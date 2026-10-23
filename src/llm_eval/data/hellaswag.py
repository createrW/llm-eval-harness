"""HellaSwag with lm-evaluation-harness preprocessing (design §7.1).

Log-likelihood cells score the four preprocessed endings; generation cells show them as lettered
options A-D and parse the chosen letter back into the gold-index string (design §7.3).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from llm_eval.data.dataset import Example, Task, TaskType, register_task

LETTERS = "ABCD"
_EXPLICIT_LETTER = re.compile(r"(?i:answer is|answer:)\s*[*(\[]*\s*([A-D])(?![A-Za-z])")
_LEADING_LETTER = re.compile(r"^[\s*(\[]*([A-D])\s*(?:[.):\]*]|$)")


def _preprocess(text: str) -> str:
    """lm-eval HellaSwag cleanup: strip, turn `` [title]`` into ``". "``, drop ``[...]`` tags."""
    text = text.strip()
    text = text.replace(" [title]", ". ")  # WikiHow artefacts
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


@register_task
class HellaSwagTask(Task):
    """HellaSwag validation split, scored as 4-way multiple choice."""

    name = "hellaswag"
    task_type = TaskType.MULTIPLE_CHOICE
    dataset_path = "Rowan/hellaswag"
    dataset_revision = "218ec52e09a7e7462a5400043bb9a69a41d06b76"
    default_split = "validation"
    exemplar_split = "train"

    def _example(self, row: Mapping[str, Any], row_idx: int, split: str) -> Example:
        ctx = row["ctx_a"] + " " + row["ctx_b"].capitalize()
        context = _preprocess(row["activity_label"] + ": " + ctx)
        inputs = {
            k: row[k]
            for k in ("ind", "source_id", "activity_label", "ctx_a", "ctx_b", "split_type")
        }
        inputs["context"] = context
        return Example(
            id=f"{self.name}/{split}/{row_idx}",
            inputs=inputs,
            target=str(int(row["label"])),
            choices=tuple(_preprocess(e) for e in row["endings"]),
        )

    def context(self, example: Example) -> str:
        return example.inputs["context"]

    def target_text(self, example: Example) -> str:
        assert example.choices is not None
        return example.choices[int(example.target)]

    def extract_answer(self, generation: str) -> str | None:
        """Gold-index string of the chosen letter, or ``None`` when no letter can be read.

        An explicit "answer is X" / "Answer: X" wins (last one, for chain-of-thought); otherwise a
        reply that starts with a bare letter ("B", "B.", "(B)", "**B**") counts. A sentence that
        merely starts with the article "A" does not.
        """
        explicit = _EXPLICIT_LETTER.findall(generation)
        if explicit:
            return str(LETTERS.index(explicit[-1]))
        leading = _LEADING_LETTER.match(generation)
        return str(LETTERS.index(leading.group(1))) if leading else None
