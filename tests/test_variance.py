# tests/test_variance.py
import numpy as np
import pytest

from llm_eval.evaluation import analysis, variance
from tests.test_analysis import N, make_run, ones


def test_shares_of_a_pure_model_effect_and_a_pure_prompt_effect():
    pure_model = np.array([0.8, 0.8, 0.8, 0.2, 0.2, 0.2])  # 2 models x 3 prompts
    assert variance._shares(pure_model, 2, 3) == pytest.approx([1.0, 0.0, 0.0])
    pure_prompt = np.array([0.2, 0.5, 0.8, 0.2, 0.5, 0.8])
    assert variance._shares(pure_prompt, 2, 3) == pytest.approx([0.0, 1.0, 0.0])
    crossed = np.array([0.8, 0.2, 0.2, 0.8])  # 2 x 2, opposite prompt effects per model
    assert variance._shares(crossed, 2, 2) == pytest.approx([0.0, 0.0, 1.0])


def test_decompose_reports_shares_with_cis_and_a_blocked_anova(tmp_path):
    outcomes = {
        ("big", "zero_shot"): ones(160),
        ("big", "cot"): ones(150),
        ("small", "zero_shot"): ones(60),
        ("small", "cot"): ones(62),
    }
    [fa] = analysis.analyze(
        analysis.load_runs([make_run(tmp_path, "run", outcomes)]), n_resamples=200
    )
    out = variance.decompose(fa)
    assert out["n"] == N
    model = out["shares"]["model"]
    assert model["share"] > 0.95 and model["ci_low"] <= model["share"] <= model["ci_high"]
    assert sum(s["share"] for s in out["shares"].values()) == pytest.approx(1.0)
    terms = {t["term"]: t for t in out["anova"]}
    assert set(terms) == {
        "example (item difficulty)",
        "model",
        "prompt",
        "model × prompt",
        "residual",
    }
    assert terms["model"]["p"] < 1e-6
    assert sum(t["share"] for t in out["anova"]) == pytest.approx(1.0)


def test_decompose_needs_two_levels_of_each_factor(tmp_path):
    outcomes = {("big", "zero_shot"): ones(10), ("small", "zero_shot"): ones(5)}
    [fa] = analysis.analyze(
        analysis.load_runs([make_run(tmp_path, "run", outcomes)]), n_resamples=50
    )
    assert variance.decompose(fa) == {}
