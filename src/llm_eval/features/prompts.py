"""Prompt variants, the cell support table and prompt rendering (design §6, §7.3, §7.4).

A *cell* is ``(task type, prompt variant, scoring mode)``. :data:`SUPPORTED` lists the cells that
run today; :data:`LANDS_ON_DAY` names the day each other valid cell lands, so an unsupported cell is
rejected with an actionable message before any model loads. Rendering uses only
``task.context()`` (and, from Day 2, ``task.target_text()``); this module imports nothing heavy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from llm_eval.data.dataset import Example, Task, TaskType
from llm_eval.models.base_model import ScoringMode

_DAY2 = "lands on Day 2"


class PromptVariant(StrEnum):
    """How the query is presented: bare, with 3 exemplars, or with chain-of-thought (§7.3)."""

    ZERO_SHOT = "zero_shot"
    FEW_SHOT_3 = "few_shot_3"
    COT = "cot"


_MC = TaskType.MULTIPLE_CHOICE
_GEN_TASK = TaskType.GENERATION
_LL = ScoringMode.LOGLIKELIHOOD
_GEN = ScoringMode.GENERATION

SUPPORTED: frozenset[tuple[TaskType, PromptVariant, ScoringMode]] = frozenset(
    {(_MC, PromptVariant.ZERO_SHOT, _LL)}
)

LANDS_ON_DAY: dict[tuple[TaskType, PromptVariant, ScoringMode], int] = {
    (_MC, PromptVariant.FEW_SHOT_3, _LL): 2,
    (_MC, PromptVariant.ZERO_SHOT, _GEN): 3,
    (_MC, PromptVariant.FEW_SHOT_3, _GEN): 3,
    (_MC, PromptVariant.COT, _GEN): 3,
    (_GEN_TASK, PromptVariant.ZERO_SHOT, _GEN): 2,
    (_GEN_TASK, PromptVariant.FEW_SHOT_3, _GEN): 2,
    (_GEN_TASK, PromptVariant.COT, _GEN): 2,
}

MAX_NEW_TOKENS: dict[PromptVariant, int] = {
    PromptVariant.ZERO_SHOT: 256,
    PromptVariant.FEW_SHOT_3: 256,
    PromptVariant.COT: 512,
}


class UnsupportedCellError(ValueError):
    """A grid cell that is not runnable today, or not a valid combination at all."""


def check_supported(
    task_type: TaskType, variant: PromptVariant, scoring: ScoringMode, *, cell: str
) -> None:
    """Raise :class:`UnsupportedCellError` naming ``cell`` unless the cell is :data:`SUPPORTED`."""
    key = (task_type, variant, scoring)
    if key in SUPPORTED:
        return
    combo = f"{cell}: {task_type.value} × {variant.value} × {scoring.value}"
    if key in LANDS_ON_DAY:
        raise UnsupportedCellError(f"{combo} lands on Day {LANDS_ON_DAY[key]}")
    raise UnsupportedCellError(f"{combo} is not a valid combination")


@dataclass(frozen=True)
class MCPrompt:
    """A log-likelihood MC prompt: one context and one continuation per choice."""

    context: str
    continuations: tuple[str, ...]


def render_mc_loglikelihood(
    task: Task, example: Example, variant: PromptVariant, exemplars: Sequence[Example] = ()
) -> MCPrompt:
    """Context plus ``" " + choice`` continuations (§7.1); only ``zero_shot`` runs on Day 1."""
    check_supported(task.task_type, variant, _LL, cell=f"{task.name}×{variant.value}")
    if example.choices is None:
        raise ValueError(f"{example.id}: a multiple-choice example has no choices")
    return MCPrompt(task.context(example), tuple(" " + c for c in example.choices))


def render_mc_generation(
    task: Task, example: Example, variant: PromptVariant, exemplars: Sequence[Example] = ()
) -> str:
    """Lettered-choices generation prompt (§7.3)."""
    raise NotImplementedError(_DAY2)


def render_generation(
    task: Task, example: Example, variant: PromptVariant, exemplars: Sequence[Example] = ()
) -> str:
    """Free-text generation prompt, e.g. GSM8K ``Question:/Answer:`` (§7.3)."""
    raise NotImplementedError(_DAY2)
