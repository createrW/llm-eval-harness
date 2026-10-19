"""Multiple-choice scoring from log-likelihoods, accuracy and the primary metric (design §6, §7.1).

``pred`` is the argmax of the summed continuation log-probabilities (lm-eval ``acc``);
``pred_norm`` is the argmax after dividing each by the *character* length of the preprocessed
choice, ``max(len(choice), 1)`` (lm-eval ``acc_norm``). Ties go to the first maximum. This module
imports nothing heavy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from llm_eval.data.dataset import TaskType
from llm_eval.models.base_model import LoglikelihoodResult, ScoringMode


@dataclass(frozen=True)
class MCScore:
    """Predictions and per-choice log-probabilities (raw and length-normalised) for one example."""

    pred: int
    pred_norm: int
    logprobs: tuple[float, ...]
    logprobs_norm: tuple[float, ...]


def score_multiple_choice(
    results: Sequence[LoglikelihoodResult], choices: Sequence[str]
) -> MCScore:
    """Score one example from its per-choice results, in choice order (design §7.1)."""
    logprobs = tuple(float(r.logprob) for r in results)
    logprobs_norm = tuple(lp / max(len(c), 1) for lp, c in zip(logprobs, choices, strict=True))
    return MCScore(
        pred=int(np.argmax(logprobs)),
        pred_norm=int(np.argmax(logprobs_norm)),
        logprobs=logprobs,
        logprobs_norm=logprobs_norm,
    )


def accuracy(correct: Sequence[bool]) -> float:
    """Fraction of ``True`` values; raises ``ValueError`` on an empty sequence."""
    if len(correct) == 0:
        raise ValueError("accuracy of an empty sequence is undefined")
    return sum(bool(c) for c in correct) / len(correct)


def primary_metric(task_type: TaskType, scoring: ScoringMode) -> str:
    """Name of the metric ``primary_correct`` measures for a cell (design §7.6)."""
    if task_type is TaskType.GENERATION:
        return "exact_match"
    if scoring is ScoringMode.LOGLIKELIHOOD:
        return "acc_norm"
    return "acc"
