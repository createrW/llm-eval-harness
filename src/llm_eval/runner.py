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
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from llm_eval.config import EvalConfig, GridCell, config_hash, expand_grid
from llm_eval.data.dataset import Example, Task, get_task
from llm_eval.evaluation.metrics import primary_metric, score_multiple_choice
from llm_eval.evaluation.report import summarize
from llm_eval.evaluation.results import ExampleRecord, records_to_frame, write_run
from llm_eval.features.prompts import PromptVariant, check_supported, render_mc_loglikelihood
from llm_eval.models.base_model import LoglikelihoodRequest, Model, ScoringMode
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
) -> list[ExampleRecord]:
    """Score one cell; Day 1 runs multiple-choice log-likelihood cells only.

    All ``K × n`` (context, continuation) requests go to the model in one ``loglikelihood`` call.
    ``exemplar_seed`` is unused until few-shot cells land (Day 2).
    """
    check_supported(
        task.task_type, variant, scoring, cell=_cell_name(model.name, task.name, variant)
    )
    prompts = [render_mc_loglikelihood(task, ex, variant) for ex in examples]
    requests = [
        LoglikelihoodRequest(p.context, continuation)
        for p in prompts
        for continuation in p.continuations
    ]
    results = model.loglikelihood(requests)
    metric = primary_metric(task.task_type, scoring)
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
                run_id=run_id,
                model=model.name,
                task=task.name,
                prompt_variant=variant.value,
                scoring=scoring.value,
                metric=metric,
                seed=seed,
                exemplar_seed=None,
                example_id=ex.id,
                example_index=index,
                prompt=prompt.context,
                target=ex.target,
                prediction=prediction,
                correct=correct,
                prediction_norm=prediction_norm,
                correct_norm=correct_norm,
                primary_correct=correct_norm if metric == "acc_norm" else correct,
                choice_logprobs=score.logprobs,
                choice_logprobs_norm=score.logprobs_norm,
                generation=None,
                parse_ok=None,
                exemplar_ids=None,
                n_prompt_tokens=ex_results[0].n_context_tokens,
                n_generated_tokens=None,
                latency_ms=float(sum(r.latency_ms for r in ex_results)),
                status="ok",
                error=None,
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
    tasks: dict[str, Task] = {}
    examples: dict[str, list[Example]] = {}
    for tcfg in cfg.tasks:
        task = get_task(tcfg.name)
        n = tcfg.n_examples if tcfg.n_examples is not None else cfg.n_examples
        examples[tcfg.name] = task.load(tcfg.split or task.default_split, n, cfg.seed)
        tasks[tcfg.name] = task

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
        models_info[mcfg.name] = {"provider": mcfg.provider, "path": mcfg.path, **model.info()}
        for cell in cells:
            name = _cell_name(cell.model.name, cell.task.name, cell.variant)
            progress.set_description(name)
            try:
                records += evaluate_cell(
                    model,
                    tasks[cell.task.name],
                    examples[cell.task.name],
                    cell.variant,
                    cell.scoring,
                    run_id=run_id,
                    seed=cfg.seed,
                    exemplar_seed=exemplar_seed,
                )
            except Exception:
                logger.exception("cell %s failed", name)
                failed_cells.append(name)
            progress.update(1)
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
    return RunResult(run_dir=run_dir, summary=summary, failed_cells=failed_cells)
