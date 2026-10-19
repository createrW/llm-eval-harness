# tests/integration/test_qwen_smoke.py
import math

import pytest

pytestmark = pytest.mark.slow


def test_qwen_hellaswag_eight_examples_is_deterministic_and_batch_invariant():
    import llm_eval.data  # noqa: F401
    from llm_eval.data.dataset import get_task
    from llm_eval.features.prompts import PromptVariant
    from llm_eval.models.base_model import LoglikelihoodRequest, ScoringMode
    from llm_eval.models.hf_model import HFCausalLM
    from llm_eval.runner import evaluate_cell

    task = get_task("hellaswag")
    examples = task.load("validation", 8, seed=42)
    model = HFCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", name="qwen", batch_size=8)
    assert model.info()["dtype"] == "float32" and len(model.info()["revision"]) == 40
    first = evaluate_cell(
        model,
        task,
        examples,
        PromptVariant.ZERO_SHOT,
        ScoringMode.LOGLIKELIHOOD,
        run_id="t",
        seed=42,
    )
    second = evaluate_cell(
        model,
        task,
        examples,
        PromptVariant.ZERO_SHOT,
        ScoringMode.LOGLIKELIHOOD,
        run_id="t",
        seed=42,
    )
    assert [r.prediction for r in first] == [r.prediction for r in second]
    assert 0.0 <= sum(r.correct for r in first) / 8 <= 1.0
    req = LoglikelihoodRequest(task.context(examples[0]), " " + examples[0].choices[0])
    [single] = model.loglikelihood([req])
    assert math.isclose(single.logprob, first[0].choice_logprobs[0], abs_tol=1e-3)
