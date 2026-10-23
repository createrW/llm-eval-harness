# tests/test_dashboard.py
from pathlib import Path

import pytest

from llm_eval.serving import dashboard as db
from tests.test_analysis import OUTCOMES, make_run

APP = Path(__file__).resolve().parents[1] / "src" / "llm_eval" / "serving" / "app.py"


@pytest.fixture
def runs_root(tmp_path):
    make_run(tmp_path, "run", OUTCOMES)
    return tmp_path


def test_cell_grid_lists_cells_in_model_then_prompt_order(runs_root):
    d = db.load(runs_root, n_resamples=100)
    [fa] = d.families
    grid = db.cell_grid(fa)
    assert grid["cell"].tolist() == [
        "big · zero_shot",
        "big · cot",
        "small · zero_shot",
        "small · cot",
    ]
    assert grid["accuracy [95% CI]"].iloc[0].startswith("0.800 [")


def test_paired_view_counts_and_orders_discordant_examples_first(runs_root):
    d = db.load(runs_root, n_resamples=100)
    [fa] = d.families
    view = db.paired(d, fa, ("small", "zero_shot"), ("big", "zero_shot"))
    assert view.swapped
    assert view.contingency.loc["A wrong", "B right"] == 100
    assert view.contingency.loc["A right", "B wrong"] == 0
    assert view.contingency.to_numpy().sum() == 200
    assert view.verdict == "B > A"
    assert view.diff[0] == pytest.approx(-0.5)
    assert view.examples["outcome"].iloc[0] == "B only"
    assert view.tests()["A wrong, B right"] == "100"


def test_significance_frame_is_symmetric_and_blank_for_exploratory_pairs(runs_root):
    [fa] = db.load(runs_root, n_resamples=100).families
    frame = db.significance_frame(fa)
    assert (
        frame.loc["big · zero_shot", "small · zero_shot"]
        == frame.loc["small · zero_shot", "big · zero_shot"]
    )
    assert frame.isna().loc["big · zero_shot", "small · cot"]


def test_streamlit_app_renders_without_errors(runs_root, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("LLM_EVAL_RUNS_ROOT", str(runs_root))
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == "llm-eval-harness"
    assert any("Select exactly two rows" in i.value for i in at.info)
