"""Variance decomposition of a family's accuracies: model vs prompt vs interaction (brief item 8).

Two views of one ``(task, scoring)`` family with a complete model × prompt grid:

1. **Cell-mean shares** (headline). The M × V matrix of cell accuracies is split as a two-way
   layout without replication: ``SS_model = V Σ a_m²``, ``SS_prompt = M Σ b_v²``,
   ``SS_interaction = Σ g_mv²``, each reported as a share of their sum, with percentile bootstrap
   95% CIs from the family's shared resample matrix over examples.
2. **Per-example ANOVA**: a linear-probability model
   ``correct ~ C(example) + C(model) * C(prompt)`` (type II sums of squares) gives F-tests for
   model, prompt and their interaction while blocking on the example, and shows how much of the
   total variance is item difficulty. F-tests on 0/1 outcomes are approximate.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from llm_eval.evaluation.analysis import FamilyAnalysis
from llm_eval.evaluation.stats import resample_indices

COMPONENTS = ("model", "prompt", "interaction")


def _shares(means: np.ndarray, n_models: int, n_variants: int) -> np.ndarray:
    """Shares of (model, prompt, interaction) for cell means shaped (..., M * V)."""
    grid = means.reshape(*means.shape[:-1], n_models, n_variants)
    grand = grid.mean(axis=(-2, -1), keepdims=True)
    a = grid.mean(axis=-1, keepdims=True) - grand
    b = grid.mean(axis=-2, keepdims=True) - grand
    g = grid - grand - a - b
    ss = np.stack(
        [
            n_variants * (a**2).sum(axis=(-2, -1)),
            n_models * (b**2).sum(axis=(-2, -1)),
            (g**2).sum(axis=(-2, -1)),
        ],
        axis=-1,
    )
    total = ss.sum(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, ss / total, np.nan)


def decompose(fa: FamilyAnalysis, *, anova: bool = True) -> dict[str, Any]:
    """Cell-mean shares with CIs and the per-example ANOVA; ``{}`` unless the grid is complete."""
    n_models, n_variants = len(fa.models), len(fa.variants)
    if n_models < 2 or n_variants < 2 or fa.matrix.shape[1] != n_models * n_variants:
        return {}
    rows = fa.matrix[~np.isnan(fa.matrix).any(axis=1)]
    n = len(rows)
    if n < 2:
        return {}
    point = _shares(rows.mean(axis=0), n_models, n_variants)
    idx = resample_indices(n, fa.n_resamples, fa.seed, fa.task)
    boot = _shares(rows[idx].mean(axis=1), n_models, n_variants)
    low, high = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    result: dict[str, Any] = {
        "n": n,
        "shares": {
            name: {"share": float(point[i]), "ci_low": float(low[i]), "ci_high": float(high[i])}
            for i, name in enumerate(COMPONENTS)
        },
    }
    if anova:
        result["anova"] = _anova(rows, fa)
    return result


def _anova(rows: np.ndarray, fa: FamilyAnalysis) -> list[dict[str, Any]]:
    import statsmodels.formula.api as smf
    from statsmodels.stats.anova import anova_lm

    n, cells = rows.shape
    labels = fa.cell_list()
    long = pd.DataFrame(
        {
            "correct": rows.reshape(-1),
            "example": np.repeat(np.arange(n), cells),
            "model": [m for _ in range(n) for m, _v in labels],
            "prompt": [v for _ in range(n) for _m, v in labels],
        }
    )
    if long["correct"].nunique() < 2:
        return []
    fit = smf.ols("correct ~ C(example) + C(model) * C(prompt)", data=long).fit()
    table = anova_lm(fit, typ=2)
    total = float(table["sum_sq"].sum())
    names = {
        "C(example)": "example (item difficulty)",
        "C(model)": "model",
        "C(prompt)": "prompt",
        "C(model):C(prompt)": "model × prompt",
        "Residual": "residual",
    }
    out = []
    for term, row in table.iterrows():
        f, p = row.get("F"), row.get("PR(>F)")
        out.append(
            {
                "term": names.get(str(term), str(term)),
                "df": int(row["df"]),
                "sum_sq": float(row["sum_sq"]),
                "share": float(row["sum_sq"]) / total if total else float("nan"),
                "F": None if pd.isna(f) else float(f),
                "p": None if pd.isna(p) else float(p),
            }
        )
    return out
