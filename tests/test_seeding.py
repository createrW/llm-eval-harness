# tests/test_seeding.py
import hashlib

import numpy as np

from llm_eval.seeding import child_rng, stable_hash


def test_stable_hash_is_sha256_prefix_big_endian():
    expected = int.from_bytes(hashlib.sha256(b"hellaswag").digest()[:8], "big")
    assert stable_hash("hellaswag") == expected
    assert stable_hash("hellaswag") == stable_hash("hellaswag")
    assert stable_hash("hellaswag") != stable_hash("gsm8k")


def test_child_rng_same_keys_same_stream():
    a = child_rng(42, "subset", "hellaswag").permutation(10)
    b = child_rng(42, "subset", "hellaswag").permutation(10)
    assert a.tolist() == b.tolist()


def test_child_rng_different_key_or_seed_different_stream():
    base = child_rng(42, "subset", "hellaswag").permutation(50).tolist()
    assert child_rng(42, "subset", "gsm8k").permutation(50).tolist() != base
    assert child_rng(43, "subset", "hellaswag").permutation(50).tolist() != base
    assert child_rng(42, "bootstrap", "hellaswag").permutation(50).tolist() != base


def test_child_rng_is_independent_of_prior_draws():
    first = child_rng(1, "a").integers(0, 1000, size=5).tolist()
    child_rng(1, "b").integers(0, 1000, size=1000)
    again = child_rng(1, "a").integers(0, 1000, size=5).tolist()
    assert first == again


def test_child_rng_returns_generator():
    assert isinstance(child_rng(0, "x"), np.random.Generator)
