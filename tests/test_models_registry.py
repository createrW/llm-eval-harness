# tests/test_models_registry.py
import subprocess
import sys
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from llm_eval.models.base_model import (
    GenerationRequest,
    LoglikelihoodRequest,
    LoglikelihoodResult,
    ScoringMode,
)
from llm_eval.models.registry import PROVIDERS, ProviderSpec, load_model, register_provider


def test_scoring_mode_values():
    assert ScoringMode("loglikelihood") is ScoringMode.LOGLIKELIHOOD
    assert ScoringMode.GENERATION.value == "generation"


def test_dataclasses_are_frozen_with_defaults():
    req = GenerationRequest(prompt="p", max_new_tokens=8)
    assert req.stop == ()
    with pytest.raises(FrozenInstanceError):
        req.prompt = "q"  # type: ignore[misc]
    res = LoglikelihoodResult(logprob=-1.5, n_tokens=2, n_context_tokens=3, latency_ms=4.0)
    assert (res.logprob, res.n_tokens, res.n_context_tokens) == (-1.5, 2, 3)
    assert LoglikelihoodRequest("c", "x").continuation == "x"


def test_register_provider_and_load_model_dispatch():
    @register_provider("dummy_test_provider", supports_loglikelihood=False)
    def make(cfg):
        return SimpleNamespace(name=cfg.name, supports_loglikelihood=False)

    assert PROVIDERS["dummy_test_provider"] == ProviderSpec(
        factory=make, supports_loglikelihood=False
    )
    model = load_model(SimpleNamespace(provider="dummy_test_provider", name="m"))
    assert model.name == "m"


def test_load_model_unknown_provider():
    with pytest.raises(KeyError, match="unknown provider 'nope'"):
        load_model(SimpleNamespace(provider="nope", name="m"))


def test_hf_provider_is_registered_with_loglikelihood_support():
    assert PROVIDERS["hf"].supports_loglikelihood is True


def test_importing_registry_does_not_import_torch():
    code = (
        "import sys, llm_eval.models.registry; "
        "print('torch' in sys.modules, 'transformers' in sys.modules)"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    ).stdout
    assert out.strip() == "False False"
