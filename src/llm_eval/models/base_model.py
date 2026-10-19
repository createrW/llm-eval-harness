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
    """Generated text, with prompt/generated token counts and latency."""

    text: str
    n_prompt_tokens: int
    n_generated_tokens: int
    latency_ms: float


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
