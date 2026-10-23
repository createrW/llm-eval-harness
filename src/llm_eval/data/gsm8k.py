"""GSM8K grade-school math, scored by exact match on the final number (design §7.2)."""

from __future__ import annotations

import re
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from typing import Any

from llm_eval.data.dataset import Example, Task, TaskType, register_task

# "1,200" (comma groups of three), "1200", "-3.5"; "3,4" reads as two numbers
_NUMBER = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")
_ANSWER_IS = re.compile(
    r"(?i:answer is)[:\s]*\**\$?\s*(-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?)"
)
_CALCULATOR = re.compile(r"<<[^>]*>>")


def normalize_number(text: str) -> str | None:
    """Canonical number string: no commas or ``$``, ``"3.0" -> "3"``; ``None`` if not a number."""
    cleaned = text.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        value = Decimal(cleaned)
    except InvalidOperation:
        return None
    if value == value.to_integral_value():
        return str(int(value))
    return format(value.normalize(), "f")


@register_task
class GSM8KTask(Task):
    """GSM8K ``main`` test split; the target is the number after ``####``."""

    name = "gsm8k"
    task_type = TaskType.GENERATION
    dataset_path = "openai/gsm8k"
    dataset_config = "main"
    dataset_revision = "740312add88f781978c0658806c59bc2815b9866"
    default_split = "test"
    exemplar_split = "train"

    def _example(self, row: Mapping[str, Any], row_idx: int, split: str) -> Example:
        rationale, _, final = str(row["answer"]).rpartition("####")
        target = normalize_number(final)
        if target is None:
            raise ValueError(f"{self.name}/{split}/{row_idx}: unparsable answer {final!r}")
        return Example(
            id=f"{self.name}/{split}/{row_idx}",
            inputs={"question": str(row["question"]).strip(), "rationale": rationale.strip()},
            target=target,
        )

    def context(self, example: Example) -> str:
        return example.inputs["question"]

    def target_text(self, example: Example) -> str:
        """The worked solution without calculator annotations, ending "The answer is N."."""
        rationale = _CALCULATOR.sub("", example.inputs["rationale"]).strip()
        return f"{rationale}\nThe answer is {example.target}."

    def extract_answer(self, generation: str) -> str | None:
        """The number after the first "answer is", else the last number in the text.

        The first stated answer wins: small base models often keep writing after it, and that
        run-on text can contain other numbers or another "answer is".
        """
        explicit = _ANSWER_IS.findall(generation)
        if explicit:
            return normalize_number(explicit[0])
        numbers = _NUMBER.findall(generation)
        return normalize_number(numbers[-1]) if numbers else None

    def is_correct(self, prediction: str | None, example: Example) -> bool:
        return prediction is not None and normalize_number(prediction) == example.target
