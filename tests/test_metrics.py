# tests/test_metrics.py
import pytest

from llm_eval.data.dataset import TaskType
from llm_eval.evaluation.metrics import MCScore, accuracy, primary_metric, score_multiple_choice
from llm_eval.models.base_model import LoglikelihoodResult, ScoringMode


def ll(x):
    return LoglikelihoodResult(logprob=x, n_tokens=1, n_context_tokens=1, latency_ms=1.0)


def test_score_multiple_choice_flip_case_from_design_7_1():
    s = score_multiple_choice([ll(-5.0), ll(-20.0)], ["a" * 10, "b" * 41])
    assert s.pred == 0
    assert s.pred_norm == 1
    assert s.logprobs == (-5.0, -20.0)
    assert s.logprobs_norm == pytest.approx((-0.5, -20 / 41))


def test_score_multiple_choice_returns_mcscore_and_breaks_ties_on_first():
    s = score_multiple_choice([ll(-2.0), ll(-2.0), ll(-3.0)], ["xx", "yy", "z"])
    assert isinstance(s, MCScore) and s.pred == 0 and s.pred_norm == 0


def test_score_multiple_choice_empty_choice_uses_length_one():
    s = score_multiple_choice([ll(-1.0), ll(-4.0)], ["", "abcd"])
    assert s.logprobs_norm == (-1.0, -1.0)
    assert s.pred_norm == 0


def test_accuracy():
    assert accuracy([True, False, True, True]) == 0.75
    with pytest.raises(ValueError):
        accuracy([])


def test_primary_metric_mapping():
    assert primary_metric(TaskType.MULTIPLE_CHOICE, ScoringMode.LOGLIKELIHOOD) == "acc_norm"
    assert primary_metric(TaskType.MULTIPLE_CHOICE, ScoringMode.GENERATION) == "acc"
    assert primary_metric(TaskType.GENERATION, ScoringMode.GENERATION) == "exact_match"
