# tests/test_prompts.py
import pytest

from llm_eval.data.dataset import TaskType
from llm_eval.data.gsm8k import GSM8KTask
from llm_eval.data.hellaswag import HellaSwagTask
from llm_eval.features.prompts import (
    SUPPORTED,
    MCPrompt,
    PromptVariant,
    UnsupportedCellError,
    check_supported,
    max_new_tokens,
    render_generation,
    render_mc_generation,
    render_mc_loglikelihood,
    stop_sequences,
)
from llm_eval.models.base_model import ScoringMode
from tests.test_data_gsm8k import ROWS as GSM_ROWS
from tests.test_data_hellaswag import ROWS

MC, GEN_TASK = TaskType.MULTIPLE_CHOICE, TaskType.GENERATION
LL, GEN = ScoringMode.LOGLIKELIHOOD, ScoringMode.GENERATION
ZERO, FEW, COT = PromptVariant.ZERO_SHOT, PromptVariant.FEW_SHOT_3, PromptVariant.COT


@pytest.fixture
def example():
    return HellaSwagTask(rows=lambda s: ROWS).load("validation", None, seed=0)[1]


def test_zero_shot_loglikelihood_prompt_is_context_plus_space_prefixed_choices(example):
    p = render_mc_loglikelihood(
        HellaSwagTask(rows=lambda s: ROWS), example, PromptVariant.ZERO_SHOT
    )
    assert p == MCPrompt(
        context="Preparing pasta: How to cook pasta. Boil water. Add salt",
        continuations=("  Stir the pot.", " Drain. Serve.", " Wait for it.", " Eat"),
    )


def test_every_valid_cell_is_supported_and_invalid_ones_name_the_cell():
    assert len(SUPPORTED) == 8
    for bad in [(MC, COT, LL), (GEN_TASK, ZERO, LL), (GEN_TASK, FEW, LL), (GEN_TASK, COT, LL)]:
        assert bad not in SUPPORTED
    with pytest.raises(
        UnsupportedCellError,
        match=r"q×hellaswag×cot: multiple_choice × cot × loglikelihood is not a valid combination",
    ):
        check_supported(MC, COT, LL, cell="q×hellaswag×cot")
    check_supported(GEN_TASK, COT, GEN, cell="ok")


def test_generation_budget_and_stop_sequences():
    assert (max_new_tokens(MC, ZERO), max_new_tokens(MC, FEW), max_new_tokens(MC, COT)) == (
        8,
        8,
        256,
    )
    assert max_new_tokens(GEN_TASK, ZERO) == max_new_tokens(GEN_TASK, COT) == 256
    assert stop_sequences(MC, ZERO) == ("\n",)
    assert stop_sequences(MC, COT) == ("\nContext:", "\n\n\n")
    assert stop_sequences(GEN_TASK, FEW) == ("\n\n", "\nQuestion:")
    assert stop_sequences(GEN_TASK, COT) == ("\nQuestion:", "\n\n\n")


def test_few_shot_loglikelihood_prepends_solved_exemplars(example):
    task = HellaSwagTask(rows=lambda s: ROWS)
    shots = task.load("validation", None, seed=0)[2:4]
    p = render_mc_loglikelihood(task, example, FEW, shots)
    assert p.context == (
        "Playing guitar: A woman holds a guitar. She eats it.\n\n"
        "Skiing: Snow falls. A skier turns left.\n\n"
        "Preparing pasta: How to cook pasta. Boil water. Add salt"
    )
    assert p.continuations == render_mc_loglikelihood(task, example, ZERO).continuations


def test_mc_generation_prompts_letter_the_choices(example):
    task = HellaSwagTask(rows=lambda s: ROWS)
    block = (
        "Context: Preparing pasta: How to cook pasta. Boil water. Add salt\n"
        "A.  Stir the pot.\nB. Drain. Serve.\nC. Wait for it.\nD. Eat\nAnswer:"
    )
    assert render_mc_generation(task, example, ZERO) == (
        "Pick the most plausible ending. Reply with one letter.\n\n" + block
    )
    assert render_mc_generation(task, example, COT).endswith(block + " Let's think step by step.")
    shot = task.load("validation", None, seed=0)[2]
    few = render_mc_generation(task, example, FEW, [shot])
    assert "Context: Playing guitar: A woman holds a guitar. She\n" in few
    assert "D. paints it.\nAnswer: B\n\n" + block in few


def test_gsm8k_prompts_follow_question_answer_format():
    task = GSM8KTask(rows=lambda s: GSM_ROWS)
    ex, shot = task.load("test", None, seed=0)[:2]
    assert render_generation(task, ex, ZERO) == (
        'Solve the math problem. Reply only with "The answer is N."\n\n'
        f"Question: {GSM_ROWS[0]['question']}\nAnswer:"
    )
    assert render_generation(task, ex, COT).endswith("\nAnswer: Let's think step by step.")
    few = render_generation(task, ex, FEW, [shot])
    assert (
        f"Question: {GSM_ROWS[1]['question']}\nAnswer: 1200+300=1,500\nThe answer is 1500." in few
    )
