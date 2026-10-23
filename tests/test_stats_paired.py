# tests/test_stats_paired.py
import numpy as np
import pytest

from llm_eval.evaluation.stats import adjust_pvalues, minimum_detectable_effect, paired_tests


def test_paired_tests_count_discordant_pairs_and_agree_with_mcnemar_formulas():
    a = np.array([1] * 30 + [0] * 10 + [1] * 40 + [0] * 20)
    b = np.array([0] * 30 + [1] * 10 + [1] * 40 + [0] * 20)
    out = paired_tests(a, b)
    assert (out["n10"], out["n01"]) == (30, 10)
    # Wilcoxon on 0/1 data == uncorrected asymptotic McNemar: z = (n10 - n01) / sqrt(n10 + n01)
    from scipy.stats import binomtest, norm

    z = (30 - 10) / np.sqrt(40)
    assert out["wilcoxon_p"] == pytest.approx(2 * norm.sf(z), rel=1e-9)
    assert out["mcnemar_p"] == pytest.approx(binomtest(30, 40, 0.5).pvalue)


def test_identical_outcomes_give_p_one():
    a = np.array([1, 0, 1, 1])
    assert paired_tests(a, a) == {"n10": 0, "n01": 0, "wilcoxon_p": 1.0, "mcnemar_p": 1.0}


def test_adjust_pvalues_bonferroni_and_holm():
    assert adjust_pvalues([0.01, 0.04, 0.5], "bonferroni") == pytest.approx([0.03, 0.12, 1.0])
    assert adjust_pvalues([0.01, 0.04, 0.5], "holm") == pytest.approx([0.03, 0.08, 0.5])
    assert adjust_pvalues([], "holm") == []


def test_minimum_detectable_effect_inverts_the_sample_size_formula():
    from scipy.stats import norm

    n, psi, alpha = 500, 0.2, 0.05 / 9
    delta = minimum_detectable_effect(n, psi, alpha=alpha)
    z_a, z_b = norm.ppf(1 - alpha / 2), norm.ppf(0.8)
    needed = (z_a * np.sqrt(psi) + z_b * np.sqrt(psi - delta**2)) ** 2 / delta**2
    assert needed == pytest.approx(n, rel=1e-6)
    assert 0.05 < delta < 0.08
    assert minimum_detectable_effect(1000, psi, alpha=alpha) < delta  # more data, smaller MDE
    assert np.isnan(minimum_detectable_effect(5, 0.01, alpha=alpha))
