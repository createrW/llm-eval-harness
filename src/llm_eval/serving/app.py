"""Streamlit dashboard: pick two cells of the model × prompt grid, compare them example by example.

Run locally with ``make dashboard``; Streamlit Community Cloud runs this file from the repo and
installs the ``requirements.txt`` next to it, which wins over the root ``uv.lock``.
"""

# ruff: noqa: E402
from __future__ import annotations

import sys
from pathlib import Path

# That requirements.txt installs the dependencies but not this package, so import it from the
# checkout: src/ is two levels up from this file. Streamlit reruns this script on every
# interaction, hence the guard; an editable install already has src/ on the path.
_SRC = str(Path(__file__).resolve().parents[2])
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import plotly.express as px
import streamlit as st

from llm_eval.evaluation.analysis_report import interval, signed_interval
from llm_eval.serving import dashboard as db

st.set_page_config(page_title="llm-eval-harness", layout="wide")


@st.cache_resource(show_spinner="Analysing the committed runs…")
def load_dashboard() -> db.Dashboard:
    return db.load(db.runs_root())


d = load_dashboard()
st.title("llm-eval-harness")
st.caption(
    "Accuracy with seeded percentile-bootstrap 95% CIs. Click two rows of the table to compare"
    " those cells on the same examples: paired tests, the difference with its CI, and every"
    " example where they disagree."
)
if not d.families:
    st.warning(f"No runs found under {db.runs_root()}.")
    st.stop()

titles = [db.family_title(fa) for fa in d.families]
fa = d.families[titles.index(st.sidebar.selectbox("Task · scoring", titles))]
st.sidebar.markdown(f"{len(fa.example_ids)} examples · {len(fa.cell_list())} cells")

grid = db.cell_grid(fa)
event = st.dataframe(
    grid.drop(columns=["model", "prompt_variant"]),
    hide_index=True,
    on_select="rerun",
    selection_mode="multi-row",
    width="stretch",
    key=f"grid-{fa.task}-{fa.scoring}",
)
rows = list(event.selection.rows)

if len(rows) != 2:
    st.info("Select exactly two rows (cells) to compare them.")
else:
    cell_a = (grid.loc[rows[0], "model"], grid.loc[rows[0], "prompt_variant"])
    cell_b = (grid.loc[rows[1], "model"], grid.loc[rows[1], "prompt_variant"])
    view = db.paired(d, fa, cell_a, cell_b)
    st.subheader(f"A = {grid.loc[rows[0], 'cell']}   vs   B = {grid.loc[rows[1], 'cell']}")
    diff, low, high = view.diff
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("accuracy A", grid.loc[rows[0], "accuracy [95% CI]"].split(" ")[0])
    c2.metric("accuracy B", grid.loc[rows[1], "accuracy [95% CI]"].split(" ")[0])
    c3.metric("A − B [95% CI]", signed_interval(diff, low, high))
    c4.metric("verdict", view.verdict)
    left, right = st.columns([1, 2])
    left.dataframe(view.contingency, width="stretch")
    right.table({k: [v] for k, v in view.tests().items()})
    only_discordant = st.toggle("Only examples where A and B disagree", value=True)
    examples = view.examples
    if only_discordant:
        examples = examples[examples["outcome"].isin(["A only", "B only"])]
    st.dataframe(examples, hide_index=True, width="stretch", height=420)

tab_matrix, tab_factors, tab_variance = st.tabs(
    ["Significance matrix", "Per model / per prompt", "Variance decomposition"]
)
with tab_matrix:
    st.caption(
        "Bonferroni-adjusted Wilcoxon p for pairs that differ in one factor; blank = pair differs"
        " in both model and prompt (exploratory)."
    )
    matrix = db.significance_frame(fa)
    fig = px.imshow(
        matrix, text_auto=".3f", zmin=0, zmax=1, color_continuous_scale="Blues_r", aspect="auto"
    )
    st.plotly_chart(fig, width="stretch")
with tab_factors:
    for table, factor in ((fa.per_model, "model"), (fa.per_prompt, "prompt_variant")):
        st.dataframe(
            table.assign(
                **{
                    "accuracy [95% CI]": [
                        interval(a, lo, hi)
                        for a, lo, hi in zip(
                            table["acc"], table["ci_low"], table["ci_high"], strict=True
                        )
                    ]
                }
            )[[factor, "accuracy [95% CI]", "cells", "n"]],
            hide_index=True,
            width="stretch",
        )
with tab_variance:
    shares = (fa.variance or {}).get("shares")
    if shares:
        st.dataframe(
            [
                {"source": k, "share [95% CI]": interval(v["share"], v["ci_low"], v["ci_high"])}
                for k, v in shares.items()
            ],
            hide_index=True,
        )
    else:
        st.caption("Needs at least two models and two prompt variants in this family.")

retest = db.test_retest_summary()
if retest:
    agreement = retest["overall"]["agreement"]
    stability = retest["conclusions"]["stability"]
    st.caption(
        f"Test–retest over {retest['reruns']} same-seed reruns: {agreement:.2%} identical"
        f" predictions; {stability:.0%} of verdicts unchanged."
    )
with st.expander("Provenance"):
    for line in db.provenance(d):
        st.text(line)
