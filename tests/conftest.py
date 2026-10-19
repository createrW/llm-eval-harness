# tests/conftest.py
import copy

import pytest

from llm_eval.data.dataset import register_task
from llm_eval.data.hellaswag import HellaSwagTask
from llm_eval.models.base_model import LoglikelihoodResult
from llm_eval.models.registry import register_provider


@pytest.fixture(scope="session")
def tiny_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    vocab = {
        "<pad>": 0,
        "<unk>": 1,
        "<bos>": 2,
        "hello": 3,
        "world": 4,
        "foo": 5,
        "bar": 6,
        "baz": 7,
        "qux": 8,
        "a": 9,
        "b": 10,
        "c": 11,
        "d": 12,
        "e": 13,
        "f": 14,
        "g": 15,
    }
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(
        single="<bos> $A", special_tokens=[("<bos>", 2)]
    )
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<unk>",
        bos_token="<bos>",
        eos_token="<pad>",
        pad_token="<pad>",
    )  # pad == eos on purpose (§7.1)
    fast.padding_side = "right"
    return fast


@pytest.fixture(scope="session")
def tiny_llama(tiny_tokenizer):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=0,
    )  # vocab_size per design §10
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).eval()


@pytest.fixture
def tiny_model_dir(tmp_path, tiny_llama, tiny_tokenizer):
    import torch

    d = tmp_path / "tiny"
    copy.deepcopy(tiny_llama).to(torch.bfloat16).save_pretrained(
        d
    )  # checkpoint config records bf16
    tiny_tokenizer.save_pretrained(d)
    return d


RUNNER_ROWS = [
    {
        "ind": i,
        "activity_label": "T",
        "ctx_a": "Ctx a.",
        "ctx_b": "x",
        "ctx": "",
        "source_id": f"s{i}",
        "split": "val",
        "split_type": "indomain",
        "label": label,
        "endings": ["aa", "bbbb", "cccccc", "dddddddd"],
    }
    for i, label in enumerate(["0", "3", "0", "0"])
]


class FakeModel:
    """Log-likelihood = -len(continuation): shortest ending wins acc, longest wins acc_norm."""

    supports_loglikelihood = True

    def __init__(self, name="fake", fail_on_context=None):
        self.name, self.fail_on_context, self.calls = name, fail_on_context, 0

    def loglikelihood(self, requests):
        self.calls += 1
        out = []
        for r in requests:
            if self.fail_on_context and self.fail_on_context in r.context:
                raise RuntimeError("boom")
            out.append(
                LoglikelihoodResult(
                    logprob=-float(len(r.continuation)),
                    n_tokens=len(r.continuation.split()),
                    n_context_tokens=len(r.context.split()),
                    latency_ms=1.0,
                )
            )
        return out

    def generate(self, requests):
        raise NotImplementedError

    def info(self):
        return {"revision": None, "dtype": "fake", "device": "cpu"}


@register_provider("fake", supports_loglikelihood=True)
def _make_fake(cfg):
    return FakeModel(name=cfg.name, fail_on_context="BOOM" if cfg.path == "fail" else None)


@register_task
class OfflineHellaSwag(HellaSwagTask):
    name = "hellaswag_offline"

    def __init__(self, rows=None):
        super().__init__(rows=rows or (lambda split: RUNNER_ROWS))
