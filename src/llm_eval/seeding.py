"""Deterministic random streams (design §5).

Every random draw in the project comes from :func:`child_rng`. Streams are keyed by strings
through :func:`stable_hash` (never Python's salted ``hash()``), so independent streams such as
subset selection, few-shot exemplars and the bootstrap never depend on evaluation order.
"""

import hashlib

import numpy as np


def stable_hash(s: str) -> int:
    """Return the first 8 bytes of ``sha256(s)`` as a big-endian unsigned int (design §5)."""
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8], "big")


def child_rng(seed: int, *keys: str) -> np.random.Generator:
    """Return an independent generator for ``(seed, *keys)`` (design §5).

    The generator is ``default_rng(SeedSequence([seed, *map(stable_hash, keys)]))``: the same
    seed and keys always yield the same stream, regardless of what was drawn before.
    """
    return np.random.default_rng(np.random.SeedSequence([seed, *(stable_hash(k) for k in keys)]))
