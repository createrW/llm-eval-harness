"""Data helpers behind the Streamlit dashboard (design §5: serving imports only evaluation.*).

Kept free of Streamlit so it can be tested: :func:`load` reads the committed runs and analyses
them, :func:`cell_grid` is the clickable table, :func:`paired` is the two-cell comparison with its
example-level detail. Nothing here imports torch, transformers or datasets, so the Streamlit
Community Cloud install stays small.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from llm_eval.evaluation.analysis import (
    Cell,
    Comparison,
    FamilyAnalysis,
    Run,
    analyze,
    discover_runs,
    load_runs,
    rerun_groups,
)
from llm_eval.evaluation.analysis_report import family_title, interval, label, pval
from llm_eval.evaluation.variance import decompose

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RUNS = REPO_ROOT / "data" / "samples" / "runs"
DEFAULT_RESULTS = REPO_ROOT / "docs" / "results.json"
TEXT_COLUMNS = ["prompt", "target", "prediction", "generation"]


def runs_root() -> Path:
    """``LLM_EVAL_RUNS_ROOT`` if set, else the repo's committed runs."""
    return Path(os.environ.get("LLM_EVAL_RUNS_ROOT", DEFAULT_RUNS))


@dataclass
class Dashboard:
    runs: list[Run]  # the reference (first) run of each config
    families: list[FamilyAnalysis]
    examples: pd.DataFrame  # one row per (cell, scoring, example) with its text


def load(root: Path, *, n_resamples: int = 1000) -> Dashboard:
    """Analyse the first run of every config under ``root`` (reruns only feed test-retest)."""
    all_runs = load_runs(discover_runs(root))
    runs = [group[0] for group in rerun_groups(all_runs)]
    families = analyze(runs, n_resamples=n_resamples)
    for fa in families:
        fa.variance = decompose(fa, anova=False)
    frames = []
    for run in runs:
        f = run.frame
        part = pd.DataFrame(
            {
                c: f[c].astype(str).to_numpy()
                for c in ("model", "task", "prompt_variant", "scoring", "example_id")
            }
        )
        for c in TEXT_COLUMNS:
            part[c] = f[c].to_numpy(dtype=object, na_value=None)
        part["correct"] = f["primary_correct"].to_numpy(dtype=object, na_value=None)
        part["status"] = f["status"].astype(str).to_numpy()
        frames.append(part)
    examples = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return Dashboard(runs=runs, families=families, examples=examples)


def cell_grid(fa: FamilyAnalysis) -> pd.DataFrame:
    """One row per cell, in the order of :meth:`FamilyAnalysis.cell_list`."""
    cells = fa.cells.set_index(["model", "prompt_variant"])
    rows = []
    for m, v in fa.cell_list():
        if (m, v) not in cells.index:
            continue
        r = cells.loc[(m, v)]
        rows.append(
            {
                "cell": label((m, v)),
                "model": m,
                "prompt_variant": v,
                "accuracy [95% CI]": interval(r["acc"], r["ci_low"], r["ci_high"]),
                "n_ok": int(r["n_ok"]),
                "parse failures": r["parse_failure_rate"],
            }
        )
    return pd.DataFrame(rows)


@dataclass
class PairedView:
    a: Cell
    b: Cell
    comparison: Comparison
    swapped: bool
    contingency: pd.DataFrame
    examples: pd.DataFrame

    @property
    def diff(self) -> tuple[float, float, float]:
        c = self.comparison
        return (
            (-c.diff, -c.diff_high, -c.diff_low)
            if self.swapped
            else (c.diff, c.diff_low, c.diff_high)
        )

    @property
    def verdict(self) -> str:
        v = self.comparison.verdict
        if self.swapped and v in ("A > B", "B > A"):
            v = "B > A" if v == "A > B" else "A > B"
        return v

    def tests(self) -> dict[str, str]:
        c = self.comparison
        return {
            "A right, B wrong": str(c.n01 if self.swapped else c.n10),
            "A wrong, B right": str(c.n10 if self.swapped else c.n01),
            "Wilcoxon p": pval(c.wilcoxon_p),
            "Wilcoxon p (Bonferroni)": pval(c.wilcoxon_bonf) or "exploratory",
            "McNemar exact p (Bonferroni)": pval(c.mcnemar_bonf) or "exploratory",
            "comparison": c.kind,
            "flags": "; ".join(c.flags) or "none",
        }


def paired(d: Dashboard, fa: FamilyAnalysis, a: Cell, b: Cell) -> PairedView:
    """Statistics of ``a`` vs ``b`` and their example-level outcomes, discordant pairs first."""
    lookup = {(c.a, c.b): (c, False) for c in fa.comparisons}
    lookup.update({(c.b, c.a): (c, True) for c in fa.comparisons})
    comparison, swapped = lookup[(a, b)]
    ex = d.examples[(d.examples["task"] == fa.task) & (d.examples["scoring"] == fa.scoring)]

    def side(cell: Cell, tag: str) -> pd.DataFrame:
        part = ex[(ex["model"] == cell[0]) & (ex["prompt_variant"] == cell[1])]
        part = part[part["status"] == "ok"].set_index("example_id")
        return part[["prompt", "target", "prediction", "generation", "correct"]].add_suffix(
            f" {tag}"
        )

    both = side(a, "A").join(side(b, "B"), how="inner")
    both["target"] = both.pop("target A")
    both = both.drop(columns=["target B"])
    right_a = both["correct A"].astype(bool)
    right_b = both["correct B"].astype(bool)
    contingency = pd.DataFrame(
        [
            [int((right_a & right_b).sum()), int((right_a & ~right_b).sum())],
            [int((~right_a & right_b).sum()), int((~right_a & ~right_b).sum())],
        ],
        index=["A right", "A wrong"],
        columns=["B right", "B wrong"],
    )
    both["outcome"] = np.select(
        [right_a & ~right_b, ~right_a & right_b, right_a & right_b],
        ["A only", "B only", "both right"],
        default="both wrong",
    )
    order = {"A only": 0, "B only": 1, "both wrong": 2, "both right": 3}
    both = both.sort_values("outcome", key=lambda s: s.map(order), kind="stable")
    columns = [
        "outcome",
        "target",
        "prediction A",
        "prediction B",
        "prompt A",
        "generation A",
        "generation B",
    ]
    examples = both.reset_index()[["example_id", *columns]]
    return PairedView(a, b, comparison, swapped, contingency, examples)


def significance_frame(fa: FamilyAnalysis) -> pd.DataFrame:
    """Cells × cells Bonferroni-adjusted Wilcoxon p for confirmatory pairs (NaN elsewhere)."""
    cells = fa.cell_list()
    names = [label(c) for c in cells]
    out = pd.DataFrame(np.nan, index=names, columns=names)
    for c in fa.comparisons:
        if c.kind != "exploratory" and c.wilcoxon_bonf is not None:
            out.loc[label(c.a), label(c.b)] = c.wilcoxon_bonf
            out.loc[label(c.b), label(c.a)] = c.wilcoxon_bonf
    return out


def test_retest_summary(path: Path = DEFAULT_RESULTS) -> dict[str, Any] | None:
    """Test-retest numbers from ``docs/results.json`` (written by ``make analyze``), if present."""
    if not Path(path).is_file():
        return None
    return json.loads(Path(path).read_text(encoding="utf-8")).get("test_retest")


def provenance(d: Dashboard) -> list[str]:
    lines = []
    for run in d.runs:
        meta = run.meta
        models = ", ".join(
            f"{name} ({info.get('path')}@{str(info.get('revision'))[:7]})"
            for name, info in meta.get("models", {}).items()
        )
        code = str(meta.get("git_sha"))[:7]
        lines.append(f"{meta.get('run_name')} / {meta.get('run_id')}: {models}; code {code}")
    return lines


__all__ = ["Dashboard", "PairedView", "cell_grid", "family_title", "load", "paired", "runs_root"]
