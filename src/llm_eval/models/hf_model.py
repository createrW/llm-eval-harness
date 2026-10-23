"""Hugging Face causal-LM adapter (design §6, §7.1, §7.5).

The only module that imports ``torch``, ``transformers`` and ``huggingface_hub`` (design §5).
Log-likelihood scoring follows the lm-eval tokenization contract of §7.1: trailing context spaces
move to the continuation, the pair is encoded as one string, and only the continuation positions
are scored. ``from_pretrained(..., seed=...)`` calls ``torch.manual_seed(seed)`` before loading
(§7.5); the runner passes the run seed through ``load_model`` so torch stays out of the runner.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from llm_eval.models.base_model import (
    GenerationRequest,
    GenerationResult,
    LoglikelihoodRequest,
    LoglikelihoodResult,
    cut_at_stop,
)

_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
_ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"]
LOGITS_BUDGET_BYTES = 1 << 30  # per log-likelihood forward pass


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
        seed: int | None = None,
    ) -> HFCausalLM:
        """Load from a local directory (revision ``None``) or a pinned hub snapshot (§7.5).

        A ``seed`` is applied with ``torch.manual_seed`` before anything loads (§7.5, defensive:
        nothing in eval is stochastic); ``None`` leaves the torch RNG untouched.
        """
        if seed is not None:
            torch.manual_seed(seed)
        device = resolve_device(device)
        if Path(path).is_dir():
            local, revision = path, None
        else:
            local = snapshot_download(path, revision=revision, allow_patterns=_ALLOW_PATTERNS)
            revision = Path(local).name  # the 40-hex commit actually loaded
        tokenizer = AutoTokenizer.from_pretrained(local)
        # Explicit torch dtype; "auto" is float32 everywhere, never the checkpoint's (§7.5).
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
        vocab = int(self.model.config.vocab_size)
        start = 0
        while start < len(order):
            # full-vocabulary fp32 logits for every position: cap their size per forward pass
            width = len(encoded[order[start]][0])
            rows = max(1, min(self.batch_size, LOGITS_BUDGET_BYTES // (width * vocab * 4)))
            batch = order[start : start + rows]
            start += rows
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
        """Greedy, left-padded, length-sorted batches cut at stop strings (§7.2).

        Results come back in request order.
        """
        encoded = [self._fit_prompt(r.prompt, r.max_new_tokens) for r in requests]
        order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]), reverse=True)
        results: dict[int, GenerationResult] = {}
        for start in range(0, len(order), self.batch_size):
            batch = order[start : start + self.batch_size]
            budget = max(requests[i].max_new_tokens for i in batch)
            stops = sorted({s for i in batch for s in requests[i].stop})
            t0 = time.perf_counter()
            new_ids = self._generate_batch([encoded[i] for i in batch], budget, stops)
            latency_ms = (time.perf_counter() - t0) * 1000.0 / len(batch)
            for i, ids in zip(batch, new_ids, strict=True):
                ids = ids[: requests[i].max_new_tokens]
                text = self.tokenizer.decode(ids, skip_special_tokens=True)
                results[i] = GenerationResult(
                    text=cut_at_stop(text, requests[i].stop),
                    n_prompt_tokens=len(encoded[i]),
                    n_generated_tokens=len(ids),
                    latency_ms=latency_ms,
                )
        return [results[i] for i in range(len(encoded))]

    def _fit_prompt(self, prompt: str, max_new_tokens: int) -> list[int]:
        """Prompt ids (BOS as the tokenizer adds it), left-truncated to leave room to generate."""
        ids = list(self.tokenizer(prompt, add_special_tokens=True).input_ids)
        excess = len(ids) + max_new_tokens - self._max_length
        if excess > 0:
            bos_id = self.tokenizer.bos_token_id
            keep = 1 if bos_id is not None and ids and ids[0] == bos_id else 0
            ids = ids[:keep] + ids[keep + excess :]
        return ids

    def _generate_batch(
        self, prompts: Sequence[list[int]], max_new_tokens: int, stops: Sequence[str]
    ) -> list[list[int]]:
        """New token ids per prompt, ending before the first eos/pad."""
        width = max(len(p) for p in prompts)
        input_ids = torch.full((len(prompts), width), self._pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(prompts), width), dtype=torch.long)
        for row, ids in enumerate(prompts):
            input_ids[row, width - len(ids) :] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row, width - len(ids) :] = 1
        config = GenerationConfig(  # explicit: checkpoint sampling defaults never apply
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            temperature=1.0,  # neutral values, so checkpoint sampling settings are not merged in
            top_p=1.0,
            pad_token_id=self._pad_id,
            eos_token_id=self.tokenizer.eos_token_id,
            stop_strings=list(stops) or None,
        )
        with torch.inference_mode():
            out = self.model.generate(
                input_ids=input_ids.to(self.device),
                attention_mask=attention_mask.to(self.device),
                generation_config=config,
                tokenizer=self.tokenizer if stops else None,
            )
        ends = {self._pad_id, self.tokenizer.eos_token_id} - {None}
        new_ids = []
        for row in out[:, width:].tolist():
            cut = next((j for j, t in enumerate(row) if t in ends), len(row))
            new_ids.append(row[:cut])
        return new_ids
