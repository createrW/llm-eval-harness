# tests/test_report.py
import pandas as pd

from llm_eval.evaluation.report import format_summary, summarize, to_markdown
from llm_eval.evaluation.results import records_to_frame
from tests.test_results import record


def two_cells():
    rows = []
    for i in range(10):
        rows.append(record(i, model="m1", correct=i < 7, correct_norm=i < 6, primary_correct=i < 6))
        rows.append(record(i, model="m2", correct=i < 4, correct_norm=i < 5, primary_correct=i < 5))
    return records_to_frame(rows)


def test_summarize_one_row_per_cell_with_point_estimates_and_cis():
    s = summarize(two_cells(), seed=42, n_resamples=200)
    assert list(s.columns) == [
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
    assert s["model"].tolist() == ["m1", "m2"]
    m1 = s.iloc[0]
    assert (m1["n"], m1["n_ok"], m1["n_error"]) == (10, 10, 0)
    assert m1["acc"] == 0.7 and m1["acc_norm"] == 0.6 and m1["primary_acc"] == 0.6
    assert (
        m1["acc_ci_low"] <= 0.7 <= m1["acc_ci_high"]
        and m1["primary_ci_low"] <= 0.6 <= m1["primary_ci_high"]
    )
    assert (
        m1["ci_method"] == "percentile_bootstrap"
        and m1["n_resamples"] == 200
        and m1["ci_seed"] == 42
    )
    assert pd.isna(m1["parse_failure_rate"])


def test_summarize_is_deterministic_and_shares_resamples_across_cells():
    a = summarize(two_cells(), seed=42, n_resamples=200)
    b = summarize(two_cells(), seed=42, n_resamples=200)
    pd.testing.assert_frame_equal(a, b)
    c = summarize(two_cells(), seed=43, n_resamples=200)
    # n=10 puts every resampled mean on a 0.1 grid, so one metric's endpoints can coincide across
    # seeds; the seed must change the intervals somewhere.
    cis = [
        "acc_ci_low",
        "acc_ci_high",
        "acc_norm_ci_low",
        "acc_norm_ci_high",
        "primary_ci_low",
        "primary_ci_high",
    ]
    assert not a[cis].equals(c[cis])


def test_cells_of_a_task_share_one_resample_matrix():
    rows = []
    for i in range(30):
        outcome = i % 3 != 0
        rows.append(
            record(i, model="m1", correct=outcome, correct_norm=outcome, primary_correct=outcome)
        )
        rows.append(
            record(i, model="m2", correct=outcome, correct_norm=outcome, primary_correct=outcome)
        )
    s = summarize(records_to_frame(rows), seed=7, n_resamples=200)
    cis = ["acc_ci_low", "acc_ci_high", "primary_ci_low", "primary_ci_high"]
    assert s.loc[0, cis].tolist() == s.loc[1, cis].tolist()


def test_to_markdown_exact():
    t = pd.DataFrame({"model": ["m1"], "acc": [0.75], "note": [None]})
    assert to_markdown(t) == "| model | acc | note |\n|---|---|---|\n| m1 | 0.750 |  |"


def test_format_summary_renders_interval_strings():
    out = format_summary(summarize(two_cells(), seed=42, n_resamples=200))
    assert out.startswith("| model | task | prompt_variant | metric | n | acc | acc_norm |")
    assert "| m1 | hellaswag | zero_shot | acc_norm | 10 | 0.700 [" in out
