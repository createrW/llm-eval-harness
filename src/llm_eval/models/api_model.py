"""OpenAI and Anthropic chat adapters: generation only, one user message per prompt (design §7.4).

Each prompt goes out as a single user message, so a prompt variant is the same text for API and
local models. Requests run concurrently (``batch_size`` workers); the SDKs retry rate limits, 5xx
and connection errors with exponential backoff (``MAX_RETRIES``). A request that still fails
returns ``GenerationResult(error=...)``, which the runner records as ``status="error"`` rather than
as a wrong answer. Keys come only from the environment (``OPENAI_API_KEY``, ``ANTHROPIC_API_KEY``).

Sampling: OpenAI requests use ``temperature=0`` and a fixed ``seed``. Current Claude models accept
only the default sampling parameters, so Anthropic requests send none and switch thinking off, which
keeps the prompt variant the only treatment; their reruns therefore measure sampling variability.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any

from llm_eval.models.base_model import (
    GenerationRequest,
    GenerationResult,
    LoglikelihoodRequest,
    LoglikelihoodResult,
    cut_at_stop,
)

if TYPE_CHECKING:
    from llm_eval.config import ModelConfig

MAX_RETRIES = 6
TIMEOUT_S = 120.0

# (text, prompt tokens, generated tokens, served model id)
Completion = tuple[str, int, int, str]


class ChatAPIModel:
    """A hosted chat model behind ``complete(request) -> Completion``."""

    supports_loglikelihood: bool = False

    def __init__(
        self,
        *,
        name: str,
        model_id: str,
        complete: Callable[[GenerationRequest], Completion],
        max_workers: int = 8,
    ) -> None:
        self.name = name
        self.model_id = model_id
        self._complete = complete
        self.max_workers = max(1, max_workers)
        self._served: set[str] = set()

    def loglikelihood(self, requests: Sequence[LoglikelihoodRequest]) -> list[LoglikelihoodResult]:
        raise NotImplementedError(f"{self.name}: API models are scored by generation only")

    def generate(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        """One completion per request, ``max_workers`` at a time; results in request order."""
        with ThreadPoolExecutor(self.max_workers) as pool:
            return list(pool.map(self._one, requests))

    def _one(self, request: GenerationRequest) -> GenerationResult:
        start = time.perf_counter()
        try:
            text, n_prompt, n_generated, served = self._complete(request)
        except Exception as exc:  # the SDK has already retried what is retryable
            latency_ms = (time.perf_counter() - start) * 1000.0
            return GenerationResult("", 0, 0, latency_ms, error=f"{type(exc).__name__}: {exc}")
        self._served.add(served)
        latency_ms = (time.perf_counter() - start) * 1000.0
        return GenerationResult(cut_at_stop(text, request.stop), n_prompt, n_generated, latency_ms)

    def info(self) -> dict[str, str | None]:
        """``revision`` is the model version(s) the API reported serving, else the requested id."""
        return {
            "revision": ",".join(sorted(self._served)) or self.model_id,
            "dtype": None,
            "device": "api",
        }


def _content_stops(stop: Sequence[str]) -> list[str]:
    """Stops sent to the API; whitespace-only ones are applied client-side by ``cut_at_stop``."""
    return [s for s in stop if s.strip()]


def openai_complete(client: Any, model_id: str, seed: int | None) -> Callable[..., Completion]:
    """Chat Completions call with greedy-like sampling (``temperature=0``, fixed ``seed``)."""

    def complete(request: GenerationRequest) -> Completion:
        kwargs: dict[str, Any] = {
            "model": model_id,
            "messages": [{"role": "user", "content": request.prompt}],
            "max_completion_tokens": request.max_new_tokens,
            "temperature": 0,
        }
        if seed is not None:
            kwargs["seed"] = seed
        if stops := _content_stops(request.stop)[:4]:
            kwargs["stop"] = stops
        response = client.chat.completions.create(**kwargs)
        usage = response.usage
        text = response.choices[0].message.content or ""
        return text, usage.prompt_tokens, usage.completion_tokens, response.model

    return complete


def anthropic_complete(client: Any, model_id: str) -> Callable[..., Completion]:
    """Messages API call with default sampling and thinking off (see the module docstring)."""

    def complete(request: GenerationRequest) -> Completion:
        kwargs: dict[str, Any] = {
            "model": model_id,
            "max_tokens": request.max_new_tokens,
            "messages": [{"role": "user", "content": request.prompt}],
            "thinking": {"type": "disabled"},
        }
        if stops := _content_stops(request.stop):
            kwargs["stop_sequences"] = stops
        response = client.messages.create(**kwargs)
        text = "".join(b.text for b in response.content if b.type == "text")
        usage = response.usage
        return text, usage.input_tokens, usage.output_tokens, response.model

    return complete


def openai_model(cfg: ModelConfig, *, seed: int | None = None) -> ChatAPIModel:
    """Build from ``OPENAI_API_KEY``; checks the key and model id before any cell runs."""
    import openai

    client = openai.OpenAI(max_retries=MAX_RETRIES, timeout=TIMEOUT_S)
    client.models.retrieve(cfg.path)
    return ChatAPIModel(
        name=cfg.name,
        model_id=cfg.path,
        complete=openai_complete(client, cfg.path, seed),
        max_workers=cfg.batch_size,
    )


def anthropic_model(cfg: ModelConfig, *, seed: int | None = None) -> ChatAPIModel:
    """Build from ``ANTHROPIC_API_KEY``; checks credentials and model id before any cell runs."""
    import anthropic

    client = anthropic.Anthropic(max_retries=MAX_RETRIES, timeout=TIMEOUT_S)
    client.models.retrieve(cfg.path)
    return ChatAPIModel(
        name=cfg.name,
        model_id=cfg.path,
        complete=anthropic_complete(client, cfg.path),
        max_workers=cfg.batch_size,
    )
