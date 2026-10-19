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
