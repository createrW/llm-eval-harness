"""Prompt variants, the cell support table and prompt rendering (design §6, §7.3, §7.4).

A *cell* is ``(task type, prompt variant, scoring mode)``; :data:`SUPPORTED` lists every valid
one, and anything else is rejected before any model loads. Prompts are plain text, identical for
local base models (as a completion prompt) and API chat models (as one user message), so a prompt
variant is the same treatment for every model. Rendering uses only ``task.context()`` and
``task.target_text()``; this module imports nothing heavy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from llm_eval.data.dataset import Example, Task, TaskType
from llm_eval.models.base_model import ScoringMode


class PromptVariant(StrEnum):
    """How the query is presented: bare, with 3 exemplars, or with chain-of-thought (§7.3)."""

    ZERO_SHOT = "zero_shot"
    FEW_SHOT_3 = "few_shot_3"
    COT = "cot"


_MC = TaskType.MULTIPLE_CHOICE
_GEN_TASK = TaskType.GENERATION
_LL = ScoringMode.LOGLIKELIHOOD
_GEN = ScoringMode.GENERATION
_ZERO, _FEW, _COT = PromptVariant.ZERO_SHOT, PromptVariant.FEW_SHOT_3, PromptVariant.COT

SUPPORTED: frozenset[tuple[TaskType, PromptVariant, ScoringMode]] = frozenset(
    {
        (_MC, _ZERO, _LL),
        (_MC, _FEW, _LL),
        (_MC, _ZERO, _GEN),
        (_MC, _FEW, _GEN),
        (_MC, _COT, _GEN),
        (_GEN_TASK, _ZERO, _GEN),
        (_GEN_TASK, _FEW, _GEN),
        (_GEN_TASK, _COT, _GEN),
    }
)
N_EXEMPLARS = 3
LETTERS = "ABCD"

_MC_HEADER = {
    _ZERO: "Pick the most plausible ending. Reply with one letter.",
    _FEW: "Pick the most plausible ending. Reply with one letter.",
    _COT: 'Pick the most plausible ending. Think step by step, then finish with "The answer is X."',
}
_GEN_HEADER = {
    _ZERO: 'Solve the math problem. Reply only with "The answer is N."',
    _FEW: 'Solve the math problem. Finish with "The answer is N."',
    _COT: 'Solve the math problem. Think step by step, then finish with "The answer is N."',
}
_COT_CUE = " Let's think step by step."


class UnsupportedCellError(ValueError):
    """A grid cell that is not a valid combination."""


def check_supported(
    task_type: TaskType, variant: PromptVariant, scoring: ScoringMode, *, cell: str
) -> None:
    """Raise :class:`UnsupportedCellError` naming ``cell`` unless the cell is :data:`SUPPORTED`."""
    if (task_type, variant, scoring) not in SUPPORTED:
        raise UnsupportedCellError(
            f"{cell}: {task_type.value} × {variant.value} × {scoring.value} is not a valid"
            " combination (chain-of-thought needs generation scoring; free-text tasks have no"
            " log-likelihood scoring)"
        )


def max_new_tokens(task_type: TaskType, variant: PromptVariant) -> int:
    """Generation budget: a letter for multiple choice, room to reason otherwise."""
    if task_type is _MC and variant is not _COT:
        return 8
    return 256


def stop_sequences(task_type: TaskType, variant: PromptVariant) -> tuple[str, ...]:
    """Where a completion ends: the line for a letter, the next block for longer answers."""
    if task_type is _MC:
        return ("\n",) if variant is not _COT else ("\nContext:", "\n\n\n")
    if variant is _FEW:  # exemplar answers never contain a blank line; it marks the next block
        return ("\n\n", "\nQuestion:")
    return ("\nQuestion:", "\n\n\n")


@dataclass(frozen=True)
class MCPrompt:
    """A log-likelihood MC prompt: one context and one continuation per choice."""

    context: str
    continuations: tuple[str, ...]


def _choices(example: Example) -> tuple[str, ...]:
    if example.choices is None:
        raise ValueError(f"{example.id}: a multiple-choice example has no choices")
    return example.choices


def render_mc_loglikelihood(
    task: Task, example: Example, variant: PromptVariant, exemplars: Sequence[Example] = ()
) -> MCPrompt:
    """Context plus ``" " + choice`` continuations (§7.1); few-shot prepends solved exemplars."""
    check_supported(task.task_type, variant, _LL, cell=f"{task.name}×{variant.value}")
    choices = _choices(example)
    context = task.context(example)
    if variant is _FEW:
        shots = [f"{task.context(e)} {task.target_text(e)}" for e in exemplars]
        context = "\n\n".join([*shots, context])
    return MCPrompt(context, tuple(" " + c for c in choices))


def _mc_block(task: Task, example: Example) -> str:
    options = [f"{LETTERS[i]}. {c}" for i, c in enumerate(_choices(example))]
    return "\n".join([f"Context: {task.context(example)}", *options, "Answer:"])


def render_mc_generation(
    task: Task, example: Example, variant: PromptVariant, exemplars: Sequence[Example] = ()
) -> str:
    """Lettered options A-D ending ``Answer:`` (§7.3); the reply is parsed for a letter."""
    check_supported(task.task_type, variant, _GEN, cell=f"{task.name}×{variant.value}")
    parts = [_MC_HEADER[variant]]
    if variant is _FEW:
        parts += [f"{_mc_block(task, e)} {LETTERS[int(e.target)]}" for e in exemplars]
    query = _mc_block(task, example)
    parts.append(query + _COT_CUE if variant is _COT else query)
    return "\n\n".join(parts)


def render_generation(
    task: Task, example: Example, variant: PromptVariant, exemplars: Sequence[Example] = ()
) -> str:
    """``Question:/Answer:`` prompt (§7.3); few-shot exemplars show full worked solutions."""
    check_supported(task.task_type, variant, _GEN, cell=f"{task.name}×{variant.value}")
    parts = [_GEN_HEADER[variant]]
    if variant is _FEW:
        parts += [f"Question: {task.context(e)}\nAnswer: {task.target_text(e)}" for e in exemplars]
    query = f"Question: {task.context(example)}\nAnswer:"
    parts.append(query + _COT_CUE if variant is _COT else query)
    return "\n\n".join(parts)
