# tests/test_stats.py
import numpy as np
from statsmodels.stats.proportion import proportion_confint

from llm_eval.evaluation.stats import bootstrap_ci, resample_indices


def test_resample_indices_shape_range_and_determinism():
    idx = resample_indices(7, 50, seed=3, key="hellaswag")
    assert idx.shape == (50, 7) and idx.min() >= 0 and idx.max() < 7
    assert np.array_equal(idx, resample_indices(7, 50, seed=3, key="hellaswag"))
    assert not np.array_equal(idx, resample_indices(7, 50, seed=3, key="gsm8k"))
    assert not np.array_equal(idx, resample_indices(7, 50, seed=4, key="hellaswag"))


def test_bootstrap_ci_degenerate_vectors():
    idx = resample_indices(10, 200, seed=0, key="t")
    assert bootstrap_ci(np.ones(10), idx) == (1.0, 1.0)
    assert bootstrap_ci(np.zeros(10), idx) == (0.0, 0.0)


def test_bootstrap_ci_contains_point_estimate_and_matches_wilson_within_0_02():
    for n, k in ((100, 56), (300, 42)):
        values = np.array([1.0] * k + [0.0] * (n - k))
        lo, hi = bootstrap_ci(values, resample_indices(n, 1000, seed=42, key="hellaswag"))
        wlo, whi = proportion_confint(k, n, alpha=0.05, method="wilson")
        assert lo <= k / n <= hi
        assert abs(lo - wlo) < 0.02 and abs(hi - whi) < 0.02
        assert isinstance(lo, float) and isinstance(hi, float)
