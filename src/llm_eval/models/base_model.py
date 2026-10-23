"""Model protocol and request/result types (design §6).

A model adapter scores ``(context, continuation)`` pairs or generates text; it knows tokens, not
tasks or prompt layouts (design §5). This module imports nothing heavy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol


class ScoringMode(StrEnum):
    """How a cell is scored: log-likelihood of choices, or free-text generation (design §7.4)."""

    LOGLIKELIHOOD = "loglikelihood"
    GENERATION = "generation"


@dataclass(frozen=True)
class LoglikelihoodRequest:
    """Score ``continuation`` given ``context``."""

    context: str
    continuation: str


@dataclass(frozen=True)
class LoglikelihoodResult:
    """Summed log-probability of the continuation tokens, with token counts and latency."""

    logprob: float
    n_tokens: int
    n_context_tokens: int
    latency_ms: float


@dataclass(frozen=True)
class GenerationRequest:
    """Greedy-generate at most ``max_new_tokens`` after ``prompt``, cutting at any ``stop``."""

    prompt: str
    max_new_tokens: int
    stop: tuple[str, ...] = ()


@dataclass(frozen=True)
class GenerationResult:
    """Generated text, with prompt/generated token counts and latency.

    ``error`` is set (and ``text`` is empty) when the request failed after retries; the runner
    records such rows with ``status="error"`` so infrastructure failures never count as wrong
    answers (design §8.2).
    """

    text: str
    n_prompt_tokens: int
    n_generated_tokens: int
    latency_ms: float
    error: str | None = None


def cut_at_stop(text: str, stop: Sequence[str]) -> str:
    """``text`` up to the earliest occurrence of any stop string (stop strings excluded)."""
    cuts = [i for s in stop if s and (i := text.find(s)) >= 0]
    return text[: min(cuts)] if cuts else text


class Model(Protocol):
    """What the runner needs from a model adapter (design §6)."""

    name: str
    supports_loglikelihood: bool

    def loglikelihood(self, requests: Sequence[LoglikelihoodRequest]) -> list[LoglikelihoodResult]:
        """Score every request; results are returned in request order."""
        ...

    def generate(self, requests: Sequence[GenerationRequest]) -> list[GenerationResult]:
        """Generate for every request; results are returned in request order."""
        ...

    def info(self) -> dict[str, str | None]:
        """``{"revision", "dtype", "device"}``: resolved values, never ``"auto"``."""
        ...
