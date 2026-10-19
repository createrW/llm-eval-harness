# tests/test_models_hf.py
import math

import pytest
import torch

from llm_eval.models.base_model import GenerationRequest, LoglikelihoodRequest
from llm_eval.models.hf_model import HFCausalLM, resolve_device, resolve_dtype


def manual_logprob(model, tokenizer, context, continuation):
    """Independent computation of §7.1: sum of log-probs of the continuation tokens."""
    n_sp = len(context) - len(context.rstrip())
    if n_sp:
        continuation, context = context[-n_sp:] + continuation, context[:-n_sp]
    whole = tokenizer(context + continuation, add_special_tokens=True).input_ids
    n_ctx = len(tokenizer(context, add_special_tokens=True).input_ids)
    ids = torch.tensor([whole])
    with torch.no_grad():
        logits = model(ids).logits[0]
    logp = torch.log_softmax(logits.float(), dim=-1)
    return (
        sum(logp[pos - 1, whole[pos]].item() for pos in range(n_ctx, len(whole))),
        len(whole) - n_ctx,
        n_ctx,
    )


@pytest.fixture
def adapter(tiny_llama, tiny_tokenizer):
    return HFCausalLM(tiny_llama, tiny_tokenizer, name="tiny", device="cpu", batch_size=2)


def test_resolve_dtype_auto_is_float32_on_every_device():
    for dev in ("cpu", "mps", "cuda"):
        assert resolve_dtype("auto", dev) is torch.float32
    assert resolve_dtype("bfloat16", "cpu") is torch.bfloat16
    assert resolve_dtype("float16", "cpu") is torch.float16


def test_resolve_device_auto_picks_an_available_backend():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in {"cuda", "mps", "cpu"}


def test_loglikelihood_matches_manual_computation(adapter, tiny_llama, tiny_tokenizer):
    req = LoglikelihoodRequest("hello world", " foo bar")
    [res] = adapter.loglikelihood([req])
    expected, n_tok, n_ctx = manual_logprob(
        tiny_llama, tiny_tokenizer, req.context, req.continuation
    )
    assert math.isclose(res.logprob, expected, abs_tol=1e-5)
    assert res.n_tokens == n_tok == 2
    assert res.n_context_tokens == n_ctx == 3  # <bos> hello world
    assert res.latency_ms > 0


def test_bos_appears_exactly_once_at_position_zero(adapter, tiny_tokenizer):
    ids = adapter._encode_pair("hello", " world")  # (whole_ids, n_ctx) helper, part of the contract
    whole, n_ctx = ids
    assert whole[0] == tiny_tokenizer.bos_token_id
    assert whole.count(tiny_tokenizer.bos_token_id) == 1
    assert n_ctx == 2 and whole[n_ctx:] == [4]


def test_loglikelihood_moves_trailing_context_spaces(adapter, tiny_llama, tiny_tokenizer):
    [res] = adapter.loglikelihood([LoglikelihoodRequest("hello world ", "foo")])
    expected, n_tok, n_ctx = manual_logprob(tiny_llama, tiny_tokenizer, "hello world ", "foo")
    assert res.n_tokens == 1 and res.n_context_tokens == 3
    assert math.isclose(res.logprob, expected, abs_tol=1e-5)


def test_loglikelihood_batched_matches_unbatched_in_request_order(
    adapter, tiny_llama, tiny_tokenizer
):
    reqs = [
        LoglikelihoodRequest("hello world foo bar", " baz"),
        LoglikelihoodRequest("hello", " world foo bar baz qux"),
        LoglikelihoodRequest("a b c", " d"),
        LoglikelihoodRequest("hello world", " foo bar baz"),
        LoglikelihoodRequest("qux", " a b c d e f g"),
    ]
    batched = adapter.loglikelihood(reqs)
    for req, res in zip(reqs, batched, strict=True):
        [single] = adapter.loglikelihood([req])
        expected, n_tok, _ = manual_logprob(
            tiny_llama, tiny_tokenizer, req.context, req.continuation
        )
        assert res.n_tokens == n_tok
        assert math.isclose(res.logprob, expected, abs_tol=1e-4)
        assert math.isclose(res.logprob, single.logprob, abs_tol=1e-4)


def test_empty_continuation_raises(adapter):
    with pytest.raises(ValueError):
        adapter.loglikelihood([LoglikelihoodRequest("hello", "")])


def test_long_context_is_truncated_from_the_left_keeping_bos(adapter, tiny_llama, tiny_tokenizer):
    # §7.1: inputs longer than the model limit (max_position_embeddings=128 here) drop tokens from
    # the left of the context only, keeping BOS, never from the continuation.
    context = " ".join(["a"] * 200)
    whole, n_ctx = adapter._encode_pair(context, " b c")
    assert len(whole) == 128 and whole[0] == tiny_tokenizer.bos_token_id
    assert whole[1:n_ctx] == [9] * 125 and whole[n_ctx:] == [10, 11] and n_ctx == 126
    [res] = adapter.loglikelihood([LoglikelihoodRequest(context, " b c")])
    assert res.n_tokens == 2 and res.n_context_tokens == 126
    with torch.no_grad():
        logp = torch.log_softmax(tiny_llama(torch.tensor([whole])).logits[0].float(), dim=-1)
    expected = logp[125, 10].item() + logp[126, 11].item()
    assert math.isclose(res.logprob, expected, abs_tol=1e-5)


def test_from_pretrained_local_dir_loads_float32_under_auto_and_reports_info(tiny_model_dir):
    m = HFCausalLM.from_pretrained(
        str(tiny_model_dir), name="tiny", dtype="auto", device="cpu", batch_size=4
    )
    assert m.model.dtype is torch.float32
    assert m.info() == {"revision": None, "dtype": "float32", "device": "cpu"}
    assert m.supports_loglikelihood is True
    [res] = m.loglikelihood([LoglikelihoodRequest("hello", " world")])
    assert res.n_tokens == 1


def test_from_pretrained_hub_path_uses_snapshot_download(monkeypatch, tiny_model_dir):
    import llm_eval.models.hf_model as hf

    calls = {}

    def fake_snapshot_download(repo_id, revision=None, allow_patterns=None):
        calls.update(repo_id=repo_id, revision=revision, allow_patterns=allow_patterns)
        fake = tiny_model_dir.parent / "snapshots" / "0123456789abcdef0123456789abcdef01234567"
        fake.parent.mkdir(parents=True, exist_ok=True)
        if not fake.exists():
            fake.symlink_to(tiny_model_dir, target_is_directory=True)
        return str(fake)

    monkeypatch.setattr(hf, "snapshot_download", fake_snapshot_download)
    m = HFCausalLM.from_pretrained("org/model", name="x", device="cpu")
    assert calls["repo_id"] == "org/model" and "*.safetensors" in calls["allow_patterns"]
    assert m.info()["revision"] == "0123456789abcdef0123456789abcdef01234567"


def test_generate_lands_on_day_2(adapter):
    with pytest.raises(NotImplementedError, match="Day 2"):
        adapter.generate([GenerationRequest("hello", 4)])


def test_registry_hf_factory_forwards_config(monkeypatch):
    from llm_eval.models import hf_model
    from llm_eval.models.registry import load_model

    seen = {}
    monkeypatch.setattr(
        hf_model.HFCausalLM,
        "from_pretrained",
        classmethod(lambda cls, path, **kw: seen.update(path=path, **kw) or "MODEL"),
    )
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        provider="hf",
        name="q",
        path="Qwen/Qwen2.5-0.5B",
        revision=None,
        dtype="auto",
        device="cpu",
        batch_size=3,
    )
    assert load_model(cfg) == "MODEL"
    assert seen == {
        "path": "Qwen/Qwen2.5-0.5B",
        "name": "q",
        "revision": None,
        "dtype": "auto",
        "device": "cpu",
        "batch_size": 3,
    }
