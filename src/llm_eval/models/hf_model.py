"""Hugging Face causal-LM adapter (design §6, §7.1, §7.5).

The only module that imports ``torch``, ``transformers`` and ``huggingface_hub`` (design §5).
Log-likelihood scoring follows the lm-eval tokenization contract of §7.1: trailing context spaces
move to the continuation, the pair is encoded as one string, and only the continuation positions
are scored. ``torch.manual_seed`` before loading is the caller's job (the runner).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_eval.models.base_model import (
    GenerationRequest,
    GenerationResult,
    LoglikelihoodRequest,
    LoglikelihoodResult,
)

_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
_ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"]


def resolve_device(device: str) -> str:
    """``"auto"`` -> ``"cuda"`` if available, else ``"mps"``, else ``"cpu"``; others unchanged."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(dtype: str, device: str) -> torch.dtype:
    """``"auto"`` -> ``float32`` on every device (§7.5); half precision only when named."""
    if dtype == "auto":
        return torch.float32
    if dtype not in _DTYPES:
        raise ValueError(
            f"unknown dtype {dtype!r} for device {device!r}; known: auto, {sorted(_DTYPES)}"
        )
    return _DTYPES[dtype]


class HFCausalLM:
    """A causal LM plus its tokenizer, scoring ``(context, continuation)`` pairs (design §6)."""

    supports_loglikelihood: bool = True

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        name: str,
        device: str,
        batch_size: int = 8,
        revision: str | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.model = model
        self.tokenizer = tokenizer
        self.name = name
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.revision = revision
        pad_id = tokenizer.pad_token_id
        self._pad_id = tokenizer.eos_token_id if pad_id is None else pad_id  # §7.1
        if self._pad_id is None:
            raise ValueError("tokenizer has neither a pad token nor an eos token")
        limits = [tokenizer.model_max_length]
        max_positions = getattr(model.config, "max_position_embeddings", None)
        if max_positions:
            limits.append(max_positions)
        self._max_length = min(limits)

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        *,
        name: str,
        revision: str | None = None,
        dtype: str = "auto",
        device: str = "auto",
        batch_size: int = 8,
    ) -> HFCausalLM:
        """Load from a local directory (revision ``None``) or a pinned hub snapshot (§7.5)."""
        device = resolve_device(device)
        if Path(path).is_dir():
            local, revision = path, None
        else:
            local = snapshot_download(path, revision=revision, allow_patterns=_ALLOW_PATTERNS)
            revision = Path(local).name  # the 40-hex commit actually loaded
        tokenizer = AutoTokenizer.from_pretrained(local)
        # Always an explicit torch dtype: omitted or "auto" means checkpoint dtype (§7.5).
        model = AutoModelForCausalLM.from_pretrained(local, dtype=resolve_dtype(dtype, device))
        model = model.to(device).eval()
        return cls(
            model, tokenizer, name=name, device=device, batch_size=batch_size, revision=revision
        )

    def info(self) -> dict[str, str | None]:
        """Resolved ``{"revision", "dtype", "device"}``, dtype read back from the loaded model."""
        return {
            "revision": self.revision,
            "dtype": str(self.model.dtype).removeprefix("torch."),
            "device": self.device,
        }

    def _encode_pair(self, context: str, continuation: str) -> tuple[list[int], int]:
        """§7.1 tokenization contract: ``(whole_ids, n_ctx)``; ``whole_ids[n_ctx:]`` is scored."""
        n_sp = len(context) - len(context.rstrip())
        if n_sp:
            continuation = context[-n_sp:] + continuation
            context = context[:-n_sp]
        whole = list(self.tokenizer(context + continuation, add_special_tokens=True).input_ids)
        n_ctx = len(self.tokenizer(context, add_special_tokens=True).input_ids)
        if n_ctx == 0:
            raise ValueError(f"context {context!r} encodes to zero tokens")
        if len(whole) <= n_ctx:
            raise ValueError(f"continuation {continuation!r} encodes to zero tokens")
        excess = len(whole) - self._max_length
        if excess > 0:  # drop from the left of the context only, keeping BOS
            bos_id = self.tokenizer.bos_token_id
            keep = 1 if bos_id is not None and whole[0] == bos_id else 0
            if excess > n_ctx - max(keep, 1):
                raise ValueError(
                    f"continuation of {len(whole) - n_ctx} tokens does not fit in "
                    f"{self._max_length} positions"
                )
            whole = whole[:keep] + whole[keep + excess :]
            n_ctx -= excess
        return whole, n_ctx

    def loglikelihood(self, requests: Sequence[LoglikelihoodRequest]) -> list[LoglikelihoodResult]:
        """Summed continuation log-probs, batched by length; results in request order (§7.1)."""
        encoded = [self._encode_pair(r.context, r.continuation) for r in requests]
        order = sorted(range(len(encoded)), key=lambda i: len(encoded[i][0]), reverse=True)
        results: dict[int, LoglikelihoodResult] = {}
        for start in range(0, len(order), self.batch_size):
            batch = order[start : start + self.batch_size]
            t0 = time.perf_counter()
            logprobs = self._score_batch([encoded[i] for i in batch])
            latency_ms = (time.perf_counter() - t0) * 1000.0 / len(batch)
            for i, logprob in zip(batch, logprobs, strict=True):
                whole, n_ctx = encoded[i]
                results[i] = LoglikelihoodResult(
                    logprob=logprob,
                    n_tokens=len(whole) - n_ctx,
                    n_context_tokens=n_ctx,
                    latency_ms=latency_ms,
                )
        return [results[i] for i in range(len(encoded))]

    def _score_batch(self, pairs: Sequence[tuple[list[int], int]]) -> list[float]:
        """Right-pad, mask from lengths, slice logits to continuation positions, log_softmax."""
        width = max(len(whole) for whole, _ in pairs)
        input_ids = torch.full((len(pairs), width), self._pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(pairs), width), dtype=torch.long)
        for row, (whole, _) in enumerate(pairs):
            input_ids[row, : len(whole)] = torch.tensor(whole, dtype=torch.long)
            attention_mask[row, : len(whole)] = 1
        with torch.inference_mode():
            logits = self.model(
                input_ids=input_ids.to(self.device), attention_mask=attention_mask.to(self.device)
            ).logits
            sums = []
            for row, (whole, n_ctx) in enumerate(pairs):
                cont = torch.tensor(whole[n_ctx:], dtype=torch.long, device=self.device)
                logp = torch.log_softmax(logits[row, n_ctx - 1 : len(whole) - 1].float(), dim=-1)
                sums.append(logp.gather(-1, cont.unsqueeze(-1)).sum())
            return torch.stack(sums).tolist()

    def generate(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        """Greedy generation (design §7.2)."""
        raise NotImplementedError("HFCausalLM.generate lands on Day 2")
