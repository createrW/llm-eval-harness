"""Cross-cell analysis of one or more runs (design §7.6-7.7; brief report items 6-8 and 10).

A *family* is one ``(task, scoring)`` pair: its cells score the same examples the same way, so
they can be compared example by example. For each family this module builds

- the cell table and the per-model / per-prompt tables (accuracy averaged over the other factor),
  each with a percentile bootstrap 95% CI over examples;
- every pairwise comparison: the accuracy difference with a paired bootstrap CI, the Wilcoxon
  signed-rank p and the exact McNemar p. Pairs that differ in one factor (two models under one
  prompt, or two prompts for one model) are *confirmatory* and are corrected with Bonferroni, with
  Holm as a sensitivity column; pairs that differ in both factors are *exploratory*;
- the pre-registered verdict (README, "Decision rule"): "A > B" iff the Bonferroni-corrected exact
  McNemar p < 0.05 and the paired-bootstrap CI of ``acc_A - acc_B`` excludes 0;
- the minimum detectable effect of a confirmatory comparison at the family's n.

Every bootstrap uses ``resample_indices(n, B, seed, task)`` over examples sorted by ``example_id``
(string order), so intervals for the same task and n come from one shared resample matrix.
Reruns of one config (same ``config_hash``) give the test-retest numbers. This module imports
neither torch nor datasets.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from llm_eval.data.dataset import TASKS, TaskType
from llm_eval.evaluation.results import RUN_FILE, read_run
from llm_eval.evaluation.stats import (
    adjust_pvalues,
    bootstrap_ci,
    minimum_detectable_effect,
    paired_tests,
    resample_indices,
)
from llm_eval.features.prompts import PromptVariant, max_new_tokens
from llm_eval.models.base_model import ScoringMode

ALPHA = 0.05
VARIANT_ORDER = [v.value for v in PromptVariant]
Cell = tuple[str, str]  # (model, prompt_variant)


@dataclass
class Run:
    """One run directory: its examples frame and its run.json."""

    run_dir: Path
    frame: pd.DataFrame
    meta: dict[str, Any]

    @property
    def config_hash(self) -> str:
        return str(self.meta["config_hash"])


def discover_runs(root: Path) -> list[Path]:
    """Every ``<root>/<run_name>/<run_id>/`` that holds a run.json, sorted."""
    return sorted(p.parent for p in Path(root).glob(f"*/*/{RUN_FILE}"))


def load_runs(run_dirs: Iterable[Path]) -> list[Run]:
    runs = []
    for run_dir in run_dirs:
        frame, meta = read_run(run_dir)
        runs.append(Run(Path(run_dir), frame, meta))
    return runs


def rerun_groups(runs: Sequence[Run]) -> list[list[Run]]:
    """Runs grouped by config hash, each group ordered by start time; groups in first-seen order."""
    groups: dict[str, list[Run]] = {}
    for run in runs:
        groups.setdefault(run.config_hash, []).append(run)
    return [sorted(g, key=lambda r: str(r.meta["started_at"])) for g in groups.values()]


def _outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    """Plain columns: model, task, prompt_variant, scoring, example_id, y (0/1, NaN if error)."""
    ok = (frame["status"].astype(str) == "ok").to_numpy()
    y = (
        frame["primary_correct"]
        .astype("boolean")
        .astype("Float64")
        .to_numpy(dtype=float, na_value=np.nan)
    )
    out = pd.DataFrame(
        {
            c: frame[c].astype(str).to_numpy()
            for c in ("model", "task", "prompt_variant", "scoring", "example_id")
        }
    )
    out["y"] = np.where(ok, y, np.nan)
    for c in ("prediction", "parse_ok", "n_generated_tokens"):
        out[c] = frame[c].to_numpy(dtype=object, na_value=None)
    return out


def combine(runs: Sequence[Run]) -> pd.DataFrame:
    """Outcome rows of several runs; a (cell, scoring, example) may appear in only one of them."""
    frames = [_outcomes(r.frame) for r in runs]
    if not frames:
        return pd.DataFrame(
            columns=["model", "task", "prompt_variant", "scoring", "example_id", "y"]
        )
    rows = pd.concat(frames, ignore_index=True)
    key = ["model", "task", "prompt_variant", "scoring", "example_id"]
    dup = rows.duplicated(subset=key, keep=False)
    if dup.any():
        cells = sorted({tuple(r) for r in rows.loc[dup, key[:4]].to_numpy().tolist()})
        raise ValueError(f"runs overlap in cells {cells}; analyse one run per config")
    return rows


@dataclass
class Comparison:
    a: Cell
    b: Cell
    kind: str  # "model", "prompt" or "exploratory"
    n: int
    acc_a: float
    acc_b: float
    diff: float
    diff_low: float
    diff_high: float
    n10: int
    n01: int
    wilcoxon_p: float
    mcnemar_p: float
    wilcoxon_bonf: float | None = None
    wilcoxon_holm: float | None = None
    mcnemar_bonf: float | None = None
    verdict: str = ""
    flags: list[str] = field(default_factory=list)


@dataclass
class FamilyAnalysis:
    task: str
    scoring: str
    metric: str
    seed: int
    n_resamples: int
    models: list[str]
    variants: list[str]
    example_ids: list[str]
    matrix: np.ndarray  # examples x cells, 0/1 with NaN
    cells: pd.DataFrame
    per_model: pd.DataFrame
    per_prompt: pd.DataFrame
    comparisons: list[Comparison]
    mde: dict[str, float]
    variance: dict[str, Any] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return f"{self.task} · {self.scoring}"

    def cell_list(self) -> list[Cell]:
        return [(m, v) for m in self.models for v in self.variants]


class _Resampler:
    """Shared resample matrices keyed by n, one stream per (seed, task)."""

    def __init__(self, seed: int, n_resamples: int, key: str) -> None:
        self.seed, self.n_resamples, self.key = seed, n_resamples, key
        self._cache: dict[int, np.ndarray] = {}

    def __call__(self, n: int) -> np.ndarray:
        if n not in self._cache:
            self._cache[n] = resample_indices(n, self.n_resamples, self.seed, self.key)
        return self._cache[n]


def _mean_ci(values: np.ndarray, resample: _Resampler) -> tuple[float, float, float, int]:
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    lo, hi = bootstrap_ci(values, resample(values.size))
    return float(values.mean()), lo, hi, int(values.size)


def _complete_rows(block: np.ndarray) -> np.ndarray:
    return block[~np.isnan(block).any(axis=1)]


def _metric(task: str, scoring: str) -> str:
    task_type = TASKS[task].task_type if task in TASKS else TaskType.MULTIPLE_CHOICE
    if task_type is TaskType.GENERATION:
        return "exact_match"
    return "acc_norm" if scoring == ScoringMode.LOGLIKELIHOOD.value else "acc"


def analyze_family(
    rows: pd.DataFrame,
    *,
    task: str,
    scoring: str,
    seed: int,
    n_resamples: int,
    budgets: dict[str, int | None] | None = None,
) -> FamilyAnalysis:
    """All tables and comparisons for one (task, scoring) family of ``rows`` (:func:`combine`)."""
    fam = rows[(rows["task"] == task) & (rows["scoring"] == scoring)]
    models = list(dict.fromkeys(fam["model"]))
    variants = [v for v in VARIANT_ORDER if v in set(fam["prompt_variant"])]
    pivot = fam.pivot(index="example_id", columns=["model", "prompt_variant"], values="y")
    pivot = pivot.sort_index()
    cells: list[Cell] = [(m, v) for m in models for v in variants if (m, v) in pivot.columns]
    matrix = pivot.reindex(columns=pd.MultiIndex.from_tuples(cells)).to_numpy(dtype=float)
    resample = _Resampler(seed, n_resamples, task)
    col = {c: i for i, c in enumerate(cells)}

    task_type = TASKS[task].task_type if task in TASKS else TaskType.MULTIPLE_CHOICE
    cell_rows = []
    for (m, v), i in col.items():
        acc, lo, hi, n_ok = _mean_ci(matrix[:, i], resample)
        part = fam[(fam["model"] == m) & (fam["prompt_variant"] == v)]
        parse = part["parse_ok"].dropna()
        parse_fail = None if parse.empty else float(1 - parse.astype(bool).mean())
        truncated = None
        if scoring == ScoringMode.GENERATION.value:
            budget = (budgets or {}).get(m) or max_new_tokens(task_type, PromptVariant(v))
            tokens = part["n_generated_tokens"].dropna().astype(int)
            truncated = None if tokens.empty else float((tokens >= budget).mean())
        cell_rows.append(
            {
                "model": m,
                "prompt_variant": v,
                "acc": acc,
                "ci_low": lo,
                "ci_high": hi,
                "n_ok": n_ok,
                "n_error": int(part["y"].isna().sum()),
                "parse_failure_rate": parse_fail,
                "truncation_rate": truncated,
            }
        )
    cells_table = pd.DataFrame(cell_rows)

    def aggregate(groups: dict[str, list[int]], label: str) -> pd.DataFrame:
        out = []
        for name, idx in groups.items():
            block = _complete_rows(matrix[:, idx])
            acc, lo, hi, n = (
                _mean_ci(block.mean(axis=1), resample) if block.size else (float("nan"),) * 3 + (0,)
            )
            out.append(
                {label: name, "acc": acc, "ci_low": lo, "ci_high": hi, "n": n, "cells": len(idx)}
            )
        return pd.DataFrame(out)

    per_model = aggregate(
        {m: [col[(m, v)] for v in variants if (m, v) in col] for m in models}, "model"
    )
    per_prompt = aggregate(
        {v: [col[(m, v)] for m in models if (m, v) in col] for v in variants}, "prompt_variant"
    )

    comparisons = []
    for a, b in itertools.combinations(cells, 2):
        kind = "model" if a[1] == b[1] else "prompt" if a[0] == b[0] else "exploratory"
        pair = _complete_rows(matrix[:, [col[a], col[b]]])
        n = len(pair)
        if n == 0:
            continue
        diffs = pair[:, 0] - pair[:, 1]
        lo, hi = bootstrap_ci(diffs, resample(n))
        tests = paired_tests(pair[:, 0], pair[:, 1])
        comparisons.append(
            Comparison(
                a,
                b,
                kind,
                n,
                float(pair[:, 0].mean()),
                float(pair[:, 1].mean()),
                float(diffs.mean()),
                lo,
                hi,
                int(tests["n10"]),
                int(tests["n01"]),
                float(tests["wilcoxon_p"]),
                float(tests["mcnemar_p"]),
            )
        )
    confirmatory = [c for c in comparisons if c.kind != "exploratory"]
    for c, wb, wh, mb in zip(
        confirmatory,
        adjust_pvalues([c.wilcoxon_p for c in confirmatory], "bonferroni"),
        adjust_pvalues([c.wilcoxon_p for c in confirmatory], "holm"),
        adjust_pvalues([c.mcnemar_p for c in confirmatory], "bonferroni"),
        strict=True,
    ):
        c.wilcoxon_bonf, c.wilcoxon_holm, c.mcnemar_bonf = wb, wh, mb

    k = len(confirmatory)
    if confirmatory:
        n_typ = int(np.median([c.n for c in confirmatory]))
        psi = float(np.median([(c.n10 + c.n01) / c.n for c in confirmatory]))
        mde = {
            "k": k,
            "n": n_typ,
            "discordance": psi,
            "mde": minimum_detectable_effect(n_typ, psi, alpha=ALPHA / k),
        }
    else:
        mde = {"k": 0, "n": 0, "discordance": float("nan"), "mde": float("nan")}
    for c in comparisons:
        _decide(c, mde["mde"])

    return FamilyAnalysis(
        task=task,
        scoring=scoring,
        metric=_metric(task, scoring),
        seed=seed,
        n_resamples=n_resamples,
        models=models,
        variants=variants,
        example_ids=[str(i) for i in pivot.index],
        matrix=matrix,
        cells=cells_table,
        per_model=per_model,
        per_prompt=per_prompt,
        comparisons=comparisons,
        mde=mde,
    )


def _decide(c: Comparison, mde: float) -> None:
    """Pre-registered decision rule, plus flags where the tests or corrections disagree."""
    excludes_zero = c.diff_low > 0 or c.diff_high < 0
    if c.kind == "exploratory":
        c.verdict = "exploratory"
        return
    assert (
        c.mcnemar_bonf is not None and c.wilcoxon_bonf is not None and c.wilcoxon_holm is not None
    )
    significant = c.mcnemar_bonf < ALPHA
    if significant and excludes_zero:
        c.verdict = "A > B" if c.diff > 0 else "B > A"
    elif not np.isnan(mde) and abs(c.diff) < mde:
        c.verdict = "underpowered — no conclusion"
    else:
        c.verdict = "no conclusion"
    if (c.wilcoxon_bonf < ALPHA) != significant:
        c.flags.append("Wilcoxon and McNemar disagree")
    if (c.wilcoxon_holm < ALPHA) != (c.wilcoxon_bonf < ALPHA):
        c.flags.append("Holm and Bonferroni disagree")
    if excludes_zero != significant:
        c.flags.append("CI and test disagree")


def analyze(runs: Sequence[Run], *, n_resamples: int = 1000) -> list[FamilyAnalysis]:
    """One :class:`FamilyAnalysis` per (task, scoring), from the first run of each config."""
    reference = [group[0] for group in rerun_groups(runs)]
    return _analyze_runs(reference, n_resamples=n_resamples)


def _analyze_runs(runs: Sequence[Run], *, n_resamples: int) -> list[FamilyAnalysis]:
    rows = combine(runs)
    seeds = {int(r.meta["config"]["seed"]) for r in runs}
    if len(seeds) > 1:
        raise ValueError(f"runs use different seeds {sorted(seeds)}; analyse one seed at a time")
    seed = seeds.pop() if seeds else 0
    budgets: dict[str, int | None] = {}
    for r in runs:
        for m in r.meta["config"]["models"]:
            budgets[m["name"]] = m.get("max_new_tokens")
    families = list(dict.fromkeys(zip(rows["task"], rows["scoring"], strict=True)))
    return [
        analyze_family(rows, task=t, scoring=s, seed=seed, n_resamples=n_resamples, budgets=budgets)
        for t, s in families
    ]


def retest_agreement(runs: Sequence[Run], *, n_resamples: int = 1000) -> dict[str, Any]:
    """Agreement across reruns of the same config, and stability of the confirmatory verdicts.

    *Agreement* is the fraction of (cell, example) pairs whose prediction is identical in every
    rerun (examples with an error row in any rerun are left out), pooled per provider and scoring
    mode and overall. *Conclusion stability* is the fraction of confirmatory comparisons whose
    verdict is the same in every rerun.
    """
    groups = [g for g in rerun_groups(runs) if len(g) > 1]
    if not groups:
        return {"reruns": 0, "cells": pd.DataFrame(), "pooled": pd.DataFrame(), "conclusions": None}
    n_reruns = min(len(g) for g in groups)
    providers = {}
    for g in groups:
        for m in g[0].meta["config"]["models"]:
            providers[m["name"]] = m["provider"]
    cell_rows = []
    for g in groups:
        outs = [_outcomes(r.frame) for r in g[:n_reruns]]
        key = ["model", "task", "prompt_variant", "scoring"]
        for cell, _ in outs[0].groupby(key, sort=False):
            preds, accs = [], []
            for o in outs:
                part = o[(o[key] == pd.Series(cell, index=key)).all(axis=1)].set_index("example_id")
                preds.append(part["prediction"].where(part["y"].notna()))
                accs.append(float(np.nanmean(part["y"])))
            table = pd.concat(preds, axis=1).dropna()
            agree = int((table.nunique(axis=1) == 1).sum())
            cell_rows.append(
                dict(zip(key, cell, strict=True))
                | {
                    "provider": providers.get(cell[0], "?"),
                    "n": len(table),
                    "agree": agree,
                    "agreement": agree / len(table) if len(table) else float("nan"),
                    "acc_min": min(accs),
                    "acc_max": max(accs),
                }
            )
    cells = pd.DataFrame(cell_rows)
    pooled = cells.groupby(["provider", "scoring"], sort=False)[["agree", "n"]].sum().reset_index()
    pooled["agreement"] = pooled["agree"] / pooled["n"]
    overall = {"agree": int(cells["agree"].sum()), "n": int(cells["n"].sum())}
    overall["agreement"] = overall["agree"] / overall["n"] if overall["n"] else float("nan")

    verdicts: list[dict[tuple[str, str, Cell, Cell], str]] = []
    for i in range(n_reruns):
        analyses = _analyze_runs([g[i] for g in groups], n_resamples=n_resamples)
        verdicts.append(
            {
                (fa.task, fa.scoring, c.a, c.b): c.verdict
                for fa in analyses
                for c in fa.comparisons
                if c.kind != "exploratory"
            }
        )
    common = set.intersection(*(set(v) for v in verdicts))
    stable = sum(len({v[k] for v in verdicts}) == 1 for k in common)
    conclusions = {
        "stable": stable,
        "total": len(common),
        "stability": stable / len(common) if common else float("nan"),
    }
    return {
        "reruns": n_reruns,
        "cells": cells,
        "pooled": pooled,
        "overall": overall,
        "conclusions": conclusions,
    }


def find_comparison(analyses: Sequence[FamilyAnalysis], task: str, scoring: str, a: Cell, b: Cell):
    """The comparison of cells ``a`` and ``b`` (either order) as ``(comparison, swapped)``."""
    for fa in analyses:
        if (fa.task, fa.scoring) != (task, scoring):
            continue
        for c in fa.comparisons:
            if (c.a, c.b) == (a, b):
                return c, False
            if (c.a, c.b) == (b, a):
                return c, True
    return None, False


def to_json(analyses: Sequence[FamilyAnalysis], retest: dict[str, Any] | None = None) -> str:
    """Machine-readable copy of the tables (for MLflow and the dashboard)."""

    def clean(value: Any) -> Any:
        if isinstance(value, float) and np.isnan(value):
            return None
        if isinstance(value, np.generic):
            return clean(value.item())
        return value

    payload = []
    for fa in analyses:
        payload.append(
            {
                "task": fa.task,
                "scoring": fa.scoring,
                "metric": fa.metric,
                "mde": {k: clean(v) for k, v in fa.mde.items()},
                "cells": [{k: clean(v) for k, v in r.items()} for r in fa.cells.to_dict("records")],
                "per_model": [
                    {k: clean(v) for k, v in r.items()} for r in fa.per_model.to_dict("records")
                ],
                "per_prompt": [
                    {k: clean(v) for k, v in r.items()} for r in fa.per_prompt.to_dict("records")
                ],
                "comparisons": [{k: clean(v) for k, v in vars(c).items()} for c in fa.comparisons],
                "variance": json.loads(
                    json.dumps(
                        fa.variance,
                        default=lambda o: clean(o) if not isinstance(o, np.ndarray) else o.tolist(),
                    )
                ),
            }
        )
    out: dict[str, Any] = {"families": payload}
    if retest and retest.get("reruns"):
        out["test_retest"] = {
            "reruns": retest["reruns"],
            "overall": retest["overall"],
            "conclusions": retest["conclusions"],
            "pooled": [
                {k: clean(v) for k, v in r.items()} for r in retest["pooled"].to_dict("records")
            ],
        }
    return json.dumps(out, indent=1, default=str)
