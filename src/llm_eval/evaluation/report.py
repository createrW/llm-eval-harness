"""Per-cell summary with bootstrap CIs and markdown tables (design §6, §7.6, §8.2).

A cell is one ``(model, task, prompt_variant)``. :func:`summarize` reduces the long-format results
frame to one row per cell with the §8.2 ``summary`` columns: counts, the parse-failure rate and the
point estimates of ``acc``, ``acc_norm`` and ``primary_correct``, each with a percentile bootstrap
95% interval over the cell's ``status == "ok"`` rows. Rows of a cell are ordered by ``example_id``
and every cell of a task is resampled with the same index matrix ``resample_indices(n, B, seed,
task)``, so intervals of cells over the same examples are mutually consistent. This module imports
neither torch nor datasets.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from llm_eval.evaluation.stats import bootstrap_ci, resample_indices

CELL_KEY = ["model", "task", "prompt_variant"]
CI_METHOD = "percentile_bootstrap"
SUMMARY_COLUMNS = [
    "model",
    "task",
    "prompt_variant",
    "scoring",
    "metric",
    "n",
    "n_ok",
    "n_error",
    "parse_failure_rate",
    "acc",
    "acc_ci_low",
    "acc_ci_high",
    "acc_norm",
    "acc_norm_ci_low",
    "acc_norm_ci_high",
    "primary_acc",
    "primary_ci_low",
    "primary_ci_high",
    "ci_method",
    "n_resamples",
    "ci_seed",
]
# n_ok is the n behind every interval (status == "ok" rows); n_error counts failed requests
FORMAT_COLUMNS = ["model", "task", "prompt_variant", "metric", "n_ok", "n_error", "acc", "acc_norm"]


def _estimate(values: pd.Series, idx: np.ndarray | None) -> tuple[Any, Any, Any]:
    """Return ``(mean, ci_low, ci_high)`` of a 0/1 column; ``None``s if it is empty or all null."""
    if idx is None or bool(values.isna().all()):
        return None, None, None
    arr = values.to_numpy(dtype=float, na_value=np.nan)
    lo, hi = bootstrap_ci(arr, idx)
    return float(arr.mean()), lo, hi


def summarize(frame: pd.DataFrame, *, seed: int, n_resamples: int = 1000) -> pd.DataFrame:
    """Reduce a results frame to one row per cell with the §8.2 summary columns.

    Output rows are sorted by ``(task, model, prompt_variant)``. ``acc_norm*`` are ``None`` when
    the cell's ``correct_norm`` is all null and ``parse_failure_rate`` is ``None`` when its
    ``parse_ok`` is all null.
    """
    resamples: dict[tuple[str, int], np.ndarray] = {}
    rows = []
    for (model, task, variant), cell in frame.groupby(CELL_KEY, sort=False):
        cell = cell.sort_values("example_id")
        ok = cell[cell["status"] == "ok"]
        n_ok = len(ok)
        idx = None
        if n_ok:
            key = (str(task), n_ok)
            if key not in resamples:
                resamples[key] = resample_indices(n_ok, n_resamples, seed, str(task))
            idx = resamples[key]
        parse_ok = ok["parse_ok"].dropna()
        parse_failure_rate = (
            None if parse_ok.empty else float((~parse_ok.astype(bool)).sum() / len(parse_ok))
        )
        acc, acc_lo, acc_hi = _estimate(ok["correct"], idx)
        norm, norm_lo, norm_hi = _estimate(ok["correct_norm"], idx)
        primary, primary_lo, primary_hi = _estimate(ok["primary_correct"], idx)
        rows.append(
            {
                "model": str(model),
                "task": str(task),
                "prompt_variant": str(variant),
                "scoring": str(cell["scoring"].iloc[0]),
                "metric": str(cell["metric"].iloc[0]),
                "n": len(cell),
                "n_ok": n_ok,
                "n_error": int((cell["status"] == "error").sum()),
                "parse_failure_rate": parse_failure_rate,
                "acc": acc,
                "acc_ci_low": acc_lo,
                "acc_ci_high": acc_hi,
                "acc_norm": norm,
                "acc_norm_ci_low": norm_lo,
                "acc_norm_ci_high": norm_hi,
                "primary_acc": primary,
                "primary_ci_low": primary_lo,
                "primary_ci_high": primary_hi,
                "ci_method": CI_METHOD,
                "n_resamples": n_resamples,
                "ci_seed": seed,
            }
        )
    rows.sort(key=lambda r: (r["task"], r["model"], r["prompt_variant"]))
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def _is_missing(value: Any) -> bool:
    if value is None or value is pd.NA or value is pd.NaT:
        return True
    return isinstance(value, float | np.floating) and math.isnan(value)


def _cell(value: Any, floatfmt: str) -> str:
    if _is_missing(value):
        return ""
    if isinstance(value, float | np.floating):
        return format(value, floatfmt)
    return str(value)


def to_markdown(table: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """Render ``table`` as a pipe table; floats use ``floatfmt``, ``None``/NaN render empty."""
    lines = [
        "| " + " | ".join(str(c) for c in table.columns) + " |",
        "|" + "---|" * len(table.columns),
    ]
    for row in table.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_cell(v, floatfmt) for v in row) + " |")
    return "\n".join(lines)


def _interval(point: Any, low: Any, high: Any) -> str | None:
    if _is_missing(point):
        return None
    return f"{point:.3f} [{low:.3f}, {high:.3f}]"


def format_summary(summary: pd.DataFrame) -> str:
    """Markdown view of a :func:`summarize` table with ``acc``/``acc_norm`` as ``"p [lo, hi]"``."""
    view = summary[FORMAT_COLUMNS].copy()
    for metric in ("acc", "acc_norm"):
        view[metric] = [
            _interval(p, lo, hi)
            for p, lo, hi in zip(
                summary[metric],
                summary[f"{metric}_ci_low"],
                summary[f"{metric}_ci_high"],
                strict=True,
            )
        ]
    return to_markdown(view)
