# tests/test_data_gsm8k.py
import pytest

import llm_eval.data  # noqa: F401  (registers tasks)
from llm_eval.data.dataset import TASKS, TaskType
from llm_eval.data.gsm8k import GSM8KTask, normalize_number

ROWS = [
    {
        "question": "Tom has 3 apples and buys 2 more. How many apples does he have?",
        "answer": "He has 3+2=<<3+2=5>>5 apples.\n#### 5",
    },
    {
        "question": "A car costs $1,200 and a bike costs $300. What is the total?",
        "answer": "1200+300=<<1200+300=1500>>1,500\n#### 1,500",
    },
    {"question": "A pen costs half of $5. How much?", "answer": "5/2=<<5/2=2.5>>2.50\n#### 2.5"},
    {"question": "It is 3 degrees and drops by 7. Temperature?", "answer": "3-7=-4\n#### -4"},
]


def make_task() -> GSM8KTask:
    return GSM8KTask(rows=lambda split: ROWS)


def test_registered_generation_task_with_pinned_revision_and_config():
    assert TASKS["gsm8k"] is GSM8KTask
    assert GSM8KTask.task_type is TaskType.GENERATION
    assert GSM8KTask.dataset_config == "main"
    assert len(GSM8KTask.dataset_revision) == 40
    task = make_task()
    task.load("test", None, seed=0)
    assert task.dataset_info()["config"] == "main"


def test_targets_are_the_normalised_number_after_hashes():
    examples = make_task().load("test", None, seed=0)
    assert [e.target for e in examples] == ["5", "1500", "2.5", "-4"]
    assert examples[0].id == "gsm8k/test/0"
    assert make_task().context(examples[0]) == ROWS[0]["question"]


def test_target_text_drops_calculator_annotations_and_states_the_answer():
    ex = make_task().load("test", None, seed=0)[0]
    assert make_task().target_text(ex) == "He has 3+2=5 apples.\nThe answer is 5."


@pytest.mark.parametrize(
    ("generation", "expected"),
    [
        ("The answer is 42.", "42"),
        ("so $1,250.50 total. The answer is $1,250.50", "1250.5"),
        ("3+2=5, then 5*2=10", "10"),
        ("The answer is 3.0", "3"),
        ("It ends at -7 degrees", "-7"),
        ("answer is: 18 apples", "18"),
        ("We get 3,4 and 5", "5"),
        ("The answer is 12. Then 99 more words", "12"),
        ("The answer is 5.\nYou are an AI. The answer is 7", "5"),
        ("no numbers", None),
    ],
)
def test_extract_answer(generation, expected):
    assert make_task().extract_answer(generation) == expected


def test_is_correct_compares_normalised_numbers():
    task = make_task()
    ex = task.load("test", None, seed=0)[1]
    assert task.is_correct("1,500", ex)
    assert task.is_correct("1500.0", ex)
    assert not task.is_correct("150", ex)
    assert not task.is_correct(None, ex)
    assert normalize_number("$2.50") == "2.5"
    assert normalize_number("abc") is None
