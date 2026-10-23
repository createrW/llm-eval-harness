"""Bootstrap statistics (design §7.6; Day 1 subset).

One resample index matrix ``idx[b, i]`` per task, drawn from ``child_rng(seed, "bootstrap", key)``,
is shared by every cell of that task so marginal and (Day 4) paired intervals stay mutually
consistent. The interval is the percentile interval of the resampled means.
"""

from __future__ import annotations

import numpy as np

from llm_eval.seeding import child_rng


def resample_indices(n: int, n_resamples: int, seed: int, key: str) -> np.ndarray:
    """Return the ``(n_resamples, n)`` index matrix of stream ``(seed, "bootstrap", key)``."""
    return child_rng(seed, "bootstrap", key).integers(0, n, size=(n_resamples, n))


def bootstrap_ci(values: np.ndarray, idx: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile ``1 - alpha`` interval of the mean of ``values`` over the resamples in ``idx``."""
    means = np.asarray(values, dtype=float)[idx].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_tests(a: np.ndarray, b: np.ndarray) -> dict[str, float | int]:
    """Paired tests on two 0/1 outcome vectors over the same examples (design §7.6).

    Returns the discordant counts ``n10`` (A right, B wrong) and ``n01``, the Wilcoxon
    signed-rank p (zeros dropped, normal approximation; on 0/1 data this equals the uncorrected
    asymptotic McNemar test) and the exact McNemar p (binomial test on the discordant pairs).
    """
    from scipy.stats import binomtest, wilcoxon

    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n10 = int(np.sum((a == 1) & (b == 0)))
    n01 = int(np.sum((a == 0) & (b == 1)))
    if n10 + n01 == 0:
        return {"n10": 0, "n01": 0, "wilcoxon_p": 1.0, "mcnemar_p": 1.0}
    wilcoxon_p = float(wilcoxon(a, b, zero_method="wilcox", method="approx").pvalue)
    mcnemar_p = float(binomtest(n10, n10 + n01, 0.5).pvalue)
    return {"n10": n10, "n01": n01, "wilcoxon_p": wilcoxon_p, "mcnemar_p": mcnemar_p}


def adjust_pvalues(pvalues: list[float], method: str) -> list[float]:
    """Family-wise adjusted p-values (``"bonferroni"`` or ``"holm"``), capped at 1."""
    if not pvalues:
        return []
    from statsmodels.stats.multitest import multipletests

    return [float(p) for p in multipletests(pvalues, alpha=0.05, method=method)[1]]


def minimum_detectable_effect(
    n: int, discordance: float, *, alpha: float = 0.05, power: float = 0.8
) -> float:
    """Smallest paired accuracy difference detectable with McNemar's test (Connor, 1987).

    Solves ``n = (z_{1-alpha/2} sqrt(psi) + z_{power} sqrt(psi - delta^2))^2 / delta^2`` for
    ``delta``, where ``psi`` is the proportion of discordant pairs; ``alpha`` is the per-test level
    (already divided by the number of comparisons for Bonferroni).
    """
    from scipy.stats import norm

    if n <= 0 or discordance <= 0:
        return float("nan")
    z_a, z_b = norm.ppf(1 - alpha / 2), norm.ppf(power)

    def required_n(delta: float) -> float:
        return (z_a * np.sqrt(discordance) + z_b * np.sqrt(discordance - delta**2)) ** 2 / delta**2

    low, high = 1e-9, min(discordance, 1.0) - 1e-12
    if required_n(high) > n:
        return float("nan")  # even the largest possible difference needs more examples
    for _ in range(200):
        mid = (low + high) / 2
        low, high = (mid, high) if required_n(mid) > n else (low, mid)
    return high
