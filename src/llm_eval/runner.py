"""Run orchestration: config grid -> per-example records -> run directory (design §6, §7.5, §9).

:func:`run_config` loads every task's examples once, then walks :func:`~llm_eval.config.expand_grid`
with models outermost, keeping at most one model resident. A cell that raises is logged and listed
in ``failed_cells`` while the remaining cells still run. The run directory receives
``examples.parquet`` and a provenance-rich ``run.json`` (§7.5, §8.2). This module never imports
``torch``: it only reaches an already-imported ``torch`` through ``sys.modules`` to free caches.
"""

from __future__ import annotations

import gc
import importlib
import importlib.metadata
import itertools
import logging
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from llm_eval.config import EvalConfig, GridCell, config_hash, expand_grid
from llm_eval.data.dataset import Example, Task, TaskType, get_task
from llm_eval.evaluation.metrics import primary_metric, score_multiple_choice
from llm_eval.evaluation.report import summarize
from llm_eval.evaluation.results import ExampleRecord, records_to_frame, write_run
from llm_eval.features.prompts import (
    N_EXEMPLARS,
    PromptVariant,
    check_supported,
    render_generation,
    render_mc_generation,
    render_mc_loglikelihood,
    stop_sequences,
)
from llm_eval.features.prompts import max_new_tokens as default_max_new_tokens
from llm_eval.models.base_model import (
    GenerationRequest,
    LoglikelihoodRequest,
    Model,
    ScoringMode,
)
from llm_eval.models.registry import load_model

logger = logging.getLogger("llm_eval")

SCHEMA_VERSION = 1
_CORE_PACKAGES = {
    "llm_eval": "llm-eval-harness",
    "numpy": "numpy",
    "pandas": "pandas",
    "pyarrow": "pyarrow",
    "scipy": "scipy",
    "statsmodels": "statsmodels",
    "pydantic": "pydantic",
}
_OPTIONAL_PACKAGES = ("torch", "transformers", "tokenizers", "datasets", "huggingface_hub")


@dataclass
class RunResult:
    """Where a run was written, its per-cell summary and the names of the cells that raised."""

    run_dir: Path
    summary: pd.DataFrame
    failed_cells: list[str]


def _cell_name(model_name: str, task_name: str, variant: PromptVariant) -> str:
    return f"{model_name}×{task_name}×{variant.value}"


def evaluate_cell(
    model: Model,
    task: Task,
    examples: Sequence[Example],
    variant: PromptVariant,
    scoring: ScoringMode,
    *,
    run_id: str,
    seed: int,
    exemplar_seed: int | None = None,
    max_new_tokens: int | None = None,
) -> list[ExampleRecord]:
    """Score one cell with a single ``loglikelihood`` or ``generate`` call for all its examples.

    Few-shot exemplars are drawn per example with ``exemplar_seed`` (default ``seed``), so every
    model of a run sees the same exemplars for the same example (design §7.3).
    ``max_new_tokens`` overrides the per-variant generation budget.
    """
    check_supported(
        task.task_type, variant, scoring, cell=_cell_name(model.name, task.name, variant)
    )
    shot_seed = None
    shots: dict[str, list[Example]] = {}
    if variant is PromptVariant.FEW_SHOT_3:
        shot_seed = exemplar_seed if exemplar_seed is not None else seed
        shots = {ex.id: task.exemplars(N_EXEMPLARS, shot_seed, ex.id) for ex in examples}
    base = {
        "run_id": run_id,
        "model": model.name,
        "task": task.name,
        "prompt_variant": variant.value,
        "scoring": scoring.value,
        "metric": primary_metric(task.task_type, scoring),
        "seed": seed,
        "exemplar_seed": shot_seed,
    }
    if scoring is ScoringMode.LOGLIKELIHOOD:
        return _loglikelihood_records(model, task, examples, variant, shots, base)
    budget = max_new_tokens or default_max_new_tokens(task.task_type, variant)
    return _generation_records(model, task, examples, variant, shots, base, budget)


def _shot_ids(shots: dict[str, list[Example]], example: Example) -> tuple[str, ...] | None:
    return tuple(e.id for e in shots[example.id]) if shots else None


def _loglikelihood_records(
    model: Model,
    task: Task,
    examples: Sequence[Example],
    variant: PromptVariant,
    shots: dict[str, list[Example]],
    base: dict[str, Any],
) -> list[ExampleRecord]:
    prompts = [render_mc_loglikelihood(task, ex, variant, shots.get(ex.id, ())) for ex in examples]
    requests = [LoglikelihoodRequest(p.context, c) for p in prompts for c in p.continuations]
    results = model.loglikelihood(requests)
    records: list[ExampleRecord] = []
    start = 0
    for index, (ex, prompt) in enumerate(zip(examples, prompts, strict=True)):
        k = len(prompt.continuations)
        ex_results = results[start : start + k]
        start += k
        assert ex.choices is not None  # render_mc_loglikelihood rejects choice-less examples
        score = score_multiple_choice(ex_results, ex.choices)
        prediction, prediction_norm = str(score.pred), str(score.pred_norm)
        correct = task.is_correct(prediction, ex)
        correct_norm = task.is_correct(prediction_norm, ex)
        records.append(
            ExampleRecord(
                **base,
                example_id=ex.id,
                example_index=index,
                prompt=prompt.context,
                target=ex.target,
                prediction=prediction,
                correct=correct,
                prediction_norm=prediction_norm,
                correct_norm=correct_norm,
                primary_correct=correct_norm if base["metric"] == "acc_norm" else correct,
                choice_logprobs=score.logprobs,
                choice_logprobs_norm=score.logprobs_norm,
                generation=None,
                parse_ok=None,
                exemplar_ids=_shot_ids(shots, ex),
                n_prompt_tokens=ex_results[0].n_context_tokens,
                n_generated_tokens=None,
                latency_ms=float(sum(r.latency_ms for r in ex_results)),
                status="ok",
                error=None,
            )
        )
    return records


def _generation_records(
    model: Model,
    task: Task,
    examples: Sequence[Example],
    variant: PromptVariant,
    shots: dict[str, list[Example]],
    base: dict[str, Any],
    budget: int,
) -> list[ExampleRecord]:
    render = (
        render_mc_generation if task.task_type is TaskType.MULTIPLE_CHOICE else render_generation
    )
    prompts = [render(task, ex, variant, shots.get(ex.id, ())) for ex in examples]
    stop = stop_sequences(task.task_type, variant)
    results = model.generate([GenerationRequest(p, budget, stop) for p in prompts])
    records: list[ExampleRecord] = []
    for index, (ex, prompt, res) in enumerate(zip(examples, prompts, results, strict=True)):
        failed = res.error is not None
        answer = None if failed else task.extract_answer(res.text)
        correct = None if failed else task.is_correct(answer, ex)
        records.append(
            ExampleRecord(
                **base,
                example_id=ex.id,
                example_index=index,
                prompt=prompt,
                target=ex.target,
                prediction=None if failed else (answer or ""),
                correct=correct,
                prediction_norm=None,
                correct_norm=None,
                primary_correct=correct,
                choice_logprobs=None,
                choice_logprobs_norm=None,
                generation=None if failed else res.text,
                parse_ok=None if failed else answer is not None,
                exemplar_ids=_shot_ids(shots, ex),
                n_prompt_tokens=res.n_prompt_tokens,
                n_generated_tokens=None if failed else res.n_generated_tokens,
                latency_ms=float(res.latency_ms),
                status="error" if failed else "ok",
                error=res.error,
            )
        )
    return records


def find_project_root(start: Path) -> Path:
    """Nearest ancestor of ``start`` (itself included) with a ``pyproject.toml``, else ``start``."""
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return start


def git_info(root: Path) -> tuple[str | None, bool | None]:
    """``(sha, dirty)`` of the repo at ``root``; ``LLM_EVAL_GIT_SHA`` wins; never raises (§7.5)."""
    if env_sha := os.environ.get("LLM_EVAL_GIT_SHA"):
        return env_sha, None
    try:
        sha = _git(root, "rev-parse", "HEAD").strip()
        dirty = bool(_git(root, "status", "--porcelain").strip())
    except (OSError, subprocess.SubprocessError):
        return None, None
    return sha, dirty


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True, timeout=30
    ).stdout


def library_versions() -> dict[str, str]:
    """Python and installed package versions, read from metadata without importing packages."""
    versions = {"python": platform.python_version()}
    for key, dist in _CORE_PACKAGES.items():
        versions[key] = importlib.metadata.version(dist)
    for dist in _OPTIONAL_PACKAGES:
        try:
            versions[dist] = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _release_model() -> None:
    """Collect garbage and free accelerator caches, only if ``torch`` is already imported."""
    gc.collect()
    if "torch" not in sys.modules:
        return
    torch = importlib.import_module("torch")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def _json_safe(value: Any) -> Any:
    """NaN/NA -> ``None`` and numpy scalars -> Python scalars, for ``run.json``."""
    if value is None or value is pd.NA:
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def run_config(
    cfg: EvalConfig, *, project_root: Path | None = None, now: datetime | None = None
) -> RunResult:
    """Run every cell of ``cfg`` and write ``<output_dir>/<run_name>/<run_id>/`` (§7.5, §8.2, §9).

    Task load errors (e.g. ``n_examples`` larger than the split) propagate before any model
    loads; a cell that raises is logged and recorded in ``failed_cells``.
    """
    now = now or datetime.now(UTC)
    root = project_root or find_project_root(Path.cwd())
    chash = config_hash(cfg)
    run_id = f"{now:%Y%m%dT%H%M%SZ}-{chash[:8]}"
    run_dir = root / cfg.output_dir / cfg.run_name / run_id
    exemplar_seed = cfg.exemplar_seed if cfg.exemplar_seed is not None else cfg.seed

    grid = expand_grid(cfg)
    logger.info("run %s: %d cells -> %s", run_id, len(grid), run_dir)
    tasks: dict[str, Task] = {}
    examples: dict[str, list[Example]] = {}
    for tcfg in cfg.tasks:
        task = get_task(tcfg.name)
        n = tcfg.n_examples if tcfg.n_examples is not None else cfg.n_examples
        examples[tcfg.name] = task.load(tcfg.split or task.default_split, n, cfg.seed)
        tasks[tcfg.name] = task
        info = task.dataset_info()
        logger.info(
            "task %s/%s: %d of %d rows @ %s",
            tcfg.name,
            info["split"],
            len(examples[tcfg.name]),
            info["n_rows"],
            str(info["revision"])[:7],
        )

    records: list[ExampleRecord] = []
    failed_cells: list[str] = []
    models_info: dict[str, dict[str, Any]] = {}
    progress = tqdm(total=len(grid), unit="cell", disable=not sys.stderr.isatty())
    for _, group in itertools.groupby(grid, key=lambda c: c.model.name):
        cells: list[GridCell] = list(group)
        mcfg = cells[0].model
        try:
            model = load_model(mcfg, seed=cfg.seed)  # hf seeds torch before loading (§7.5)
        except Exception:
            logger.exception("loading model %s failed; skipping its cells", mcfg.name)
            failed_cells += [_cell_name(c.model.name, c.task.name, c.variant) for c in cells]
            progress.update(len(cells))
            continue
        resolved = model.info()
        logger.info(
            "model %s: device=%s dtype=%s revision=%s",
            mcfg.name,
            resolved["device"],
            resolved["dtype"],
            resolved["revision"],
        )
        if mcfg.device == "auto" and resolved["device"] == "cpu":
            logger.warning("model %s: device auto resolved to cpu; expect a slow run", mcfg.name)
        for cell in cells:
            name = _cell_name(cell.model.name, cell.task.name, cell.variant)
            progress.set_description(name)
            started = time.perf_counter()
            try:
                cell_records = evaluate_cell(
                    model,
                    tasks[cell.task.name],
                    examples[cell.task.name],
                    cell.variant,
                    cell.scoring,
                    run_id=run_id,
                    seed=cfg.seed,
                    exemplar_seed=exemplar_seed,
                    max_new_tokens=cell.model.max_new_tokens,
                )
            except Exception:
                logger.exception("cell %s failed", name)
                failed_cells.append(name)
            else:
                records += cell_records
                n_ok = sum(r.status == "ok" for r in cell_records)
                n_right = sum(bool(r.primary_correct) for r in cell_records)
                logger.info(
                    "cell %s: %d examples, primary %.3f, %.1f s",
                    name,
                    len(cell_records),
                    n_right / max(n_ok, 1),
                    time.perf_counter() - started,
                )
            progress.update(1)
        # after the cells, so API models report the versions they actually served
        models_info[mcfg.name] = {"provider": mcfg.provider, "path": mcfg.path, **model.info()}
        del model
        _release_model()
    progress.close()

    frame = records_to_frame(records)
    summary = summarize(frame, seed=cfg.seed, n_resamples=cfg.bootstrap_resamples)
    git_sha, git_dirty = git_info(root)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "run_name": cfg.run_name,
        "config": cfg.model_dump(mode="json"),
        "config_hash": chash,
        "started_at": now.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "git_sha": git_sha,
        "git_dirty": git_dirty,
        "library_versions": library_versions(),
        "models": models_info,
        "tasks": {
            name: task.dataset_info() | {"example_ids": [e.id for e in examples[name]]}
            for name, task in tasks.items()
        },
        "failed_cells": failed_cells,
        "summary": [
            {k: _json_safe(v) for k, v in row.items()} for row in summary.to_dict("records")
        ],
    }
    write_run(run_dir, frame, metadata)
    logger.info("run %s written to %s", run_id, run_dir)
    return RunResult(run_dir=run_dir, summary=summary, failed_cells=failed_cells)
