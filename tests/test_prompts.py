# tests/test_prompts.py
import pytest

from llm_eval.data.dataset import TaskType
from llm_eval.data.hellaswag import HellaSwagTask
from llm_eval.features.prompts import (
    MAX_NEW_TOKENS,
    SUPPORTED,
    MCPrompt,
    PromptVariant,
    UnsupportedCellError,
    check_supported,
    render_mc_loglikelihood,
)
from llm_eval.models.base_model import ScoringMode
from tests.test_data_hellaswag import ROWS


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


def test_supported_cells_day1_and_max_new_tokens():
    assert SUPPORTED == frozenset(  # noqa: SIM300
        {(TaskType.MULTIPLE_CHOICE, PromptVariant.ZERO_SHOT, ScoringMode.LOGLIKELIHOOD)}
    )
    assert MAX_NEW_TOKENS == {
        PromptVariant.ZERO_SHOT: 256,
        PromptVariant.FEW_SHOT_3: 256,
        PromptVariant.COT: 512,
    }


def test_check_supported_names_cell_and_day():
    with pytest.raises(
        UnsupportedCellError,
        match=r"q×hellaswag×few_shot_3: multiple_choice × few_shot_3 × loglikelihood"
        r" lands on Day 2",
    ):
        check_supported(
            TaskType.MULTIPLE_CHOICE,
            PromptVariant.FEW_SHOT_3,
            ScoringMode.LOGLIKELIHOOD,
            cell="q×hellaswag×few_shot_3",
        )
    with pytest.raises(UnsupportedCellError, match="lands on Day 3"):
        check_supported(
            TaskType.MULTIPLE_CHOICE, PromptVariant.COT, ScoringMode.GENERATION, cell="c"
        )
    with pytest.raises(UnsupportedCellError, match="is not a valid combination"):
        check_supported(
            TaskType.MULTIPLE_CHOICE, PromptVariant.COT, ScoringMode.LOGLIKELIHOOD, cell="c"
        )
    with pytest.raises(UnsupportedCellError, match="is not a valid combination"):
        check_supported(
            TaskType.GENERATION, PromptVariant.ZERO_SHOT, ScoringMode.LOGLIKELIHOOD, cell="c"
        )
    check_supported(
        TaskType.MULTIPLE_CHOICE, PromptVariant.ZERO_SHOT, ScoringMode.LOGLIKELIHOOD, cell="ok"
    )


def test_render_few_shot_raises_unsupported(example):
    with pytest.raises(UnsupportedCellError, match="Day 2"):
        render_mc_loglikelihood(
            HellaSwagTask(rows=lambda s: ROWS), example, PromptVariant.FEW_SHOT_3
        )
