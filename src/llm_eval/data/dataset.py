"""Task abstraction and registry (design §6).

A :class:`Task` turns raw dataset rows into fully preprocessed :class:`Example` objects, renders
their context and judges predictions. ``datasets`` is imported only inside :meth:`Task._rows`
(design §5), so importing this module stays light.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, ClassVar

from llm_eval.seeding import child_rng

_DAY2 = "lands on Day 2"


class TaskType(StrEnum):
    """How a task is scored: choose among choices, or generate free text."""

    MULTIPLE_CHOICE = "multiple_choice"
    GENERATION = "generation"


@dataclass(frozen=True)
class Example:
    """One preprocessed dataset row (design §6).

    ``id`` is ``f"{task}/{split}/{row_idx}"`` with ``row_idx`` the 0-based row position in the
    split at the pinned dataset revision. ``choices`` and ``target`` are authoritative; for
    multiple choice the gold index is ``int(target)``.
    """

    id: str
    inputs: dict[str, Any]
    target: str
    choices: tuple[str, ...] | None = None


class Task(ABC):
    """A benchmark task: loading, preprocessing, nested subsets, rendering, judging (design §6)."""

    name: ClassVar[str]
    task_type: ClassVar[TaskType]
    dataset_path: ClassVar[str]
    dataset_revision: ClassVar[str]
    default_split: ClassVar[str]
    stop_sequences: ClassVar[tuple[str, ...]] = ()

    def __init__(self, rows: Callable[[str], Sequence[Mapping[str, Any]]] | None = None) -> None:
        """``rows(split)``, when given, replaces ``datasets.load_dataset`` (test seam)."""
        self._rows_fn = rows
        self._info: dict[str, Any] | None = None

    def _rows(self, split: str) -> Sequence[Mapping[str, Any]]:
        if self._rows_fn is not None:
            return self._rows_fn(split)
        import datasets

        return datasets.load_dataset(self.dataset_path, split=split, revision=self.dataset_revision)

    @abstractmethod
    def _example(self, row: Mapping[str, Any], row_idx: int, split: str) -> Example:
        """Build the preprocessed :class:`Example` for ``row`` at position ``row_idx``."""

    def load(self, split: str, n_examples: int | None, seed: int) -> list[Example]:
        """Return the examples of ``split``, or the seeded nested subset of size ``n_examples``.

        Subset rule (design §7.5): ``perm = child_rng(seed, "subset", name).permutation(n_rows)``;
        the subset is ``sorted(int(i) for i in perm[:n])``, so equal seeds give nested subsets.
        """
        examples = [self._example(row, i, split) for i, row in enumerate(self._rows(split))]
        n_rows = len(examples)
        ids = [e.id for e in examples]
        if len(set(ids)) != n_rows:
            raise ValueError(f"duplicate example ids in {self.name}/{split}")
        if n_examples is not None and n_examples > n_rows:
            raise ValueError(
                f"n_examples={n_examples} exceeds {n_rows} rows in {self.name}/{split}"
            )
        self._info = {
            "path": self.dataset_path,
            "revision": self.dataset_revision,
            "split": split,
            "n_rows": n_rows,
        }
        if n_examples is None:
            return examples
        perm = child_rng(seed, "subset", self.name).permutation(n_rows)
        idx = sorted(int(i) for i in perm[:n_examples])
        return [examples[i] for i in idx]

    def dataset_info(self) -> dict[str, Any]:
        """Return ``{"path", "revision", "split", "n_rows"}`` of the last :meth:`load`."""
        if self._info is None:
            raise RuntimeError(f"dataset_info() of {self.name} is valid only after load()")
        return dict(self._info)

    @abstractmethod
    def context(self, example: Example) -> str:
        """Rendered question/context: no exemplars, no trailing whitespace."""

    def is_correct(self, prediction: str | None, example: Example) -> bool:
        """Whether ``prediction`` equals the canonical gold ``example.target``."""
        return prediction == example.target

    def target_text(self, example: Example) -> str:
        """Gold continuation as shown in a few-shot exemplar."""
        raise NotImplementedError(_DAY2)

    def exemplars(self, k: int, seed: int, example_id: str) -> list[Example]:
        """Per-example few-shot exemplar draw (design §7.3)."""
        raise NotImplementedError(_DAY2)

    def extract_answer(self, generation: str) -> str | None:
        """Parse a generation into a canonical answer; ``None`` means unparsable."""
        raise NotImplementedError(_DAY2)


TASKS: dict[str, type[Task]] = {}


def register_task(cls: type[Task]) -> type[Task]:
    """Class decorator: register ``cls`` in :data:`TASKS` under ``cls.name``."""
    TASKS[cls.name] = cls
    return cls


def get_task(name: str) -> Task:
    """Instantiate the registered task ``name``."""
    if name not in TASKS:
        raise KeyError(f"unknown task {name!r}; known: {sorted(TASKS)}")
    return TASKS[name]()
