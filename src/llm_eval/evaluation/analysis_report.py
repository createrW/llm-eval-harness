"""Markdown rendering of :mod:`llm_eval.evaluation.analysis` results (README and docs tables)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from llm_eval.evaluation.analysis import Cell, Comparison, FamilyAnalysis, find_comparison
from llm_eval.evaluation.report import to_markdown

TASK_LABEL = {"hellaswag": "HellaSwag", "gsm8k": "GSM8K"}
SCORING_LABEL = {"loglikelihood": "log-likelihood", "generation": "generation"}
RETEST_NOTE = (
    "{reruns} same-seed reruns of each config. Agreement = share of (cell, example) pairs whose "
    "prediction is identical in every rerun; conclusion stability = share of confirmatory "
    "verdicts that are identical in every rerun."
)


def _missing(x: Any) -> bool:
    return x is None or (isinstance(x, float) and np.isnan(x))


def interval(p: float, lo: float, hi: float) -> str:
    return "" if _missing(p) else f"{p:.3f} [{lo:.3f}, {hi:.3f}]"


def signed_interval(d: float, lo: float, hi: float) -> str:
    return f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def pval(p: float | None) -> str:
    if _missing(p):
        return ""
    assert p is not None
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def label(cell: Cell) -> str:
    return f"{cell[0]} · {cell[1]}"


def family_title(fa: FamilyAnalysis) -> str:
    task = TASK_LABEL.get(fa.task, fa.task)
    return f"{task} · {SCORING_LABEL.get(fa.scoring, fa.scoring)} (metric: {fa.metric})"


def cells_table(fa: FamilyAnalysis) -> str:
    view = pd.DataFrame(
        {
            "model": fa.cells["model"],
            "prompt": fa.cells["prompt_variant"],
            "acc [95% CI]": [
                interval(a, lo, hi)
                for a, lo, hi in zip(
                    fa.cells["acc"], fa.cells["ci_low"], fa.cells["ci_high"], strict=True
                )
            ],
            "n_ok": fa.cells["n_ok"],
            "n_error": fa.cells["n_error"],
            "parse failures": fa.cells["parse_failure_rate"],
            "hit token budget": fa.cells["truncation_rate"],
        }
    )
    if fa.scoring != "generation":
        view = view.drop(columns=["parse failures", "hit token budget"])
    return to_markdown(view)


def factor_table(table: pd.DataFrame, factor: str) -> str:
    view = pd.DataFrame(
        {
            factor: table[factor],
            "acc [95% CI]": [
                interval(a, lo, hi)
                for a, lo, hi in zip(table["acc"], table["ci_low"], table["ci_high"], strict=True)
            ],
            "cells averaged": table["cells"],
            "n": table["n"],
        }
    )
    return to_markdown(view)


def comparisons_table(fa: FamilyAnalysis) -> str:
    rows = [
        {
            "A": label(c.a),
            "B": label(c.b),
            "factor": c.kind,
            "acc_A − acc_B [95% CI]": signed_interval(c.diff, c.diff_low, c.diff_high),
            "A only / B only": f"{c.n10} / {c.n01}",
            "Wilcoxon p (Bonf.)": pval(c.wilcoxon_bonf),
            "Holm": pval(c.wilcoxon_holm),
            "McNemar exact p (Bonf.)": pval(c.mcnemar_bonf),
            "verdict": c.verdict,
            "flags": "; ".join(c.flags),
        }
        for c in fa.comparisons
        if c.kind != "exploratory"
    ]
    return to_markdown(pd.DataFrame(rows))


def significance_matrix(fa: FamilyAnalysis) -> str:
    """Upper triangle: Bonferroni-adjusted Wilcoxon p of confirmatory pairs; ★ = verdict holds."""
    cells = fa.cell_list()
    lookup = {(c.a, c.b): c for c in fa.comparisons}
    header = ["", *[label(c) for c in cells]]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for i, row in enumerate(cells):
        entries = [label(row)]
        for j, column in enumerate(cells):
            c = lookup.get((row, column)) if i < j else None
            if i == j:
                entries.append("—")
            elif c is None:
                entries.append("")
            elif c.kind == "exploratory":
                entries.append("·")
            else:
                star = "★" if c.verdict in ("A > B", "B > A") else ""
                entries.append(f"{pval(c.wilcoxon_bonf)}{star}")
        lines.append("| " + " | ".join(entries) + " |")
    return "\n".join(lines)


def variance_section(fa: FamilyAnalysis) -> str:
    v = fa.variance
    if not v:
        return ""
    lines = []
    shares = v.get("shares")
    if shares:
        rows = [
            {
                "source": name,
                "share of cell-mean variance [95% CI]": interval(
                    s["share"], s["ci_low"], s["ci_high"]
                ),
            }
            for name, s in shares.items()
        ]
        lines += [to_markdown(pd.DataFrame(rows)), ""]
    anova = v.get("anova")
    if anova:
        rows = [
            {
                "term": t["term"],
                "df": t["df"],
                "share of total SS": t["share"],
                "F": t.get("F"),
                "p": pval(t.get("p")),
            }
            for t in anova
        ]
        lines += [
            "Per-example linear-probability ANOVA, `correct ~ C(example) + C(model) * C(prompt)`:",
            "",
            to_markdown(pd.DataFrame(rows), floatfmt=".3f"),
        ]
    return "\n".join(lines)


def hypotheses_section(analyses: Sequence[FamilyAnalysis], hypotheses: list[dict[str, Any]]) -> str:
    rows = []
    for h in hypotheses:
        a = (h["a"]["model"], h["a"]["prompt_variant"])
        b = (h["b"]["model"], h["b"]["prompt_variant"])
        c, swapped = find_comparison(analyses, h["task"], h["scoring"], a, b)
        if c is None:
            rows.append({"id": h["id"], "claim": h["claim"], "result": "not run"})
            continue
        diff, lo, hi = (
            (-c.diff, -c.diff_high, -c.diff_low) if swapped else (c.diff, c.diff_low, c.diff_high)
        )
        verdict = c.verdict
        if swapped and verdict in ("A > B", "B > A"):
            verdict = "B > A" if verdict == "A > B" else "A > B"
        result = {"A > B": "supported", "B > A": "contradicted (reverse holds)"}.get(
            verdict, f"not supported — {verdict}"
        )
        rows.append(
            {
                "id": h["id"],
                "claim": h["claim"],
                "n": str(c.n),
                "Δ [95% CI]": signed_interval(diff, lo, hi),
                "McNemar exact p (Bonf.)": pval(c.mcnemar_bonf),
                "result": result,
            }
        )
    columns = ["id", "claim", "n", "Δ [95% CI]", "McNemar exact p (Bonf.)", "result"]
    return to_markdown(pd.DataFrame(rows, columns=columns).fillna(""))


def retest_section(retest: dict[str, Any]) -> str:
    if not retest or not retest.get("reruns"):
        return "No config was rerun yet."
    overall, conclusions = retest["overall"], retest["conclusions"]
    pooled = retest["pooled"].assign(agreement=lambda d: d["agreement"].map(lambda x: f"{x:.4f}"))
    pooled = pooled.rename(columns={"agree": "identical predictions", "n": "(cell, example) pairs"})
    return "\n".join(
        [
            RETEST_NOTE.format(reruns=retest["reruns"]),
            "",
            to_markdown(
                pooled[
                    [
                        "provider",
                        "scoring",
                        "identical predictions",
                        "(cell, example) pairs",
                        "agreement",
                    ]
                ]
            ),
            "",
            f"Overall agreement: **{overall['agreement']:.2%}**"
            f" ({overall['agree']} of {overall['n']})."
            f" Conclusion stability: **{conclusions['stability']:.0%}**"
            f" ({conclusions['stable']} of {conclusions['total']} confirmatory verdicts).",
        ]
    )


def render(
    analyses: Sequence[FamilyAnalysis],
    *,
    retest: dict[str, Any] | None = None,
    hypotheses: list[dict[str, Any]] | None = None,
    provenance: str = "",
) -> str:
    """The full results document: hypotheses, then one section per family, then test-retest."""
    out = ["# Results", ""]
    if provenance:
        out += [provenance, ""]
    if hypotheses:
        out += ["## Pre-registered hypotheses", "", hypotheses_section(analyses, hypotheses), ""]
    for fa in analyses:
        mde = fa.mde
        out += [
            f"## {family_title(fa)}",
            "",
            f"{len(fa.example_ids)} examples; bootstrap B = {fa.n_resamples}, seed {fa.seed}.",
            "",
            "### Cells",
            "",
            cells_table(fa),
            "",
            "### Per model (averaged over prompt variants)",
            "",
            factor_table(fa.per_model, "model"),
            "",
            "### Per prompt variant (averaged over models)",
            "",
            factor_table(fa.per_prompt, "prompt_variant"),
            "",
            f"### Pairwise comparisons ({mde['k']} confirmatory, Bonferroni)",
            "",
            comparisons_table(fa),
            "",
            "Significance matrix (Bonferroni-adjusted Wilcoxon p; ★ = verdict under the decision"
            " rule; · = pair differs in both factors, exploratory, not tested):",
            "",
            significance_matrix(fa),
            "",
        ]
        if not _missing(mde["mde"]):
            psi = f"{mde['discordance']:.2f}"
            out += [
                f"**Power.** With n = {mde['n']} and a discordance rate of {psi},"
                f" a confirmatory comparison at α = 0.05/{mde['k']} has 80% power to"
                f" detect differences of **{mde['mde']:.3f}** or more; smaller differences"
                ' are "underpowered — no conclusion", never "no difference".',
                "",
            ]
        variance = variance_section(fa)
        if variance:
            out += ["### Variance decomposition", "", variance, ""]
    out += ["## Test–retest", "", retest_section(retest or {}), ""]
    return "\n".join(out)


def load_hypotheses(path: Path) -> list[dict[str, Any]]:
    data = yaml.safe_load(Path(path).read_text())
    return list(data or [])


__all__ = ["Comparison", "load_hypotheses", "render"]


README_START = "<!-- results:start (generated by `make analyze`; do not edit by hand) -->"
README_END = "<!-- results:end -->"


def grid_table(fa: FamilyAnalysis) -> str:
    """Model rows × prompt-variant columns of ``acc [95% CI]``."""
    cells = fa.cells.set_index(["model", "prompt_variant"])
    rows = []
    for m in fa.models:
        row = {"model": m}
        for v in fa.variants:
            if (m, v) in cells.index:
                r = cells.loc[(m, v)]
                row[v] = interval(r["acc"], r["ci_low"], r["ci_high"])
        rows.append(row)
    return to_markdown(pd.DataFrame(rows))


def readme_block(
    analyses: Sequence[FamilyAnalysis],
    *,
    retest: dict[str, Any] | None = None,
    hypotheses: list[dict[str, Any]] | None = None,
) -> str:
    """Compact results for the README; the full document is ``docs/results.md``."""
    out = []
    if hypotheses:
        out += ["**Pre-registered hypotheses** (decision rule under *Experiment protocol*):", ""]
        out += [hypotheses_section(analyses, hypotheses), ""]
    for fa in analyses:
        confirmed = [c for c in fa.comparisons if c.verdict in ("A > B", "B > A")]
        confirmatory = [c for c in fa.comparisons if c.kind != "exploratory"]
        out += [f"**{family_title(fa)}**, n = {len(fa.example_ids)} examples per cell:", ""]
        out += [grid_table(fa), ""]
        notes = [
            f"{len(confirmed)} of {len(confirmatory)} one-factor comparisons meet the decision rule"
        ]
        if not _missing(fa.mde.get("mde")):
            notes.append(f"minimum detectable difference {fa.mde['mde']:.3f}")
        shares = (fa.variance or {}).get("shares")
        if shares:
            parts = [
                f"{name} {s['share']:.0%} [{s['ci_low']:.0%}, {s['ci_high']:.0%}]"
                for name, s in shares.items()
            ]
            notes.append("variance of cell means: " + ", ".join(parts))
        out += ["; ".join(notes) + ".", ""]
    if retest and retest.get("reruns"):
        out += ["**Test–retest**: " + retest_section(retest).split("\n", 1)[0], ""]
        out += [retest_section(retest).split("\n", 1)[1].strip(), ""]
    out.append(
        "All pairwise tests, significance matrices, ANOVA tables and power: "
        "[`docs/results.md`](docs/results.md)."
    )
    return "\n".join(out)


def replace_readme_block(readme: Path, block: str) -> None:
    """Rewrite the text between the results markers in ``readme``."""
    text = Path(readme).read_text(encoding="utf-8")
    if README_START not in text or README_END not in text:
        raise ValueError(f"{readme} has no results markers")
    head, rest = text.split(README_START, 1)
    _, tail = rest.split(README_END, 1)
    Path(readme).write_text(f"{head}{README_START}\n{block}\n{README_END}{tail}", encoding="utf-8")
