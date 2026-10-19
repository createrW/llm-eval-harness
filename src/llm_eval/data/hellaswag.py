"""HellaSwag with lm-evaluation-harness preprocessing (design §7.1)."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from llm_eval.data.dataset import Example, Task, TaskType, register_task


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
    stop_sequences = ("\n\n",)

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
