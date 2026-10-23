# tests/test_analysis.py
import json

import pytest

from llm_eval.cli import main
from llm_eval.evaluation import analysis, analysis_report
from llm_eval.evaluation.results import ExampleRecord, records_to_frame, write_run

N = 200


def make_run(
    root,
    name,
    outcomes,
    *,
    config_hash="h1",
    started="2026-10-23T00:00:00",
    task="gsm8k",
    scoring="generation",
    flip=(),
):
    """Write a run whose cell (model, variant) has the given 0/1 outcomes over N examples.

    ``flip`` lists (model, variant, i) whose prediction string changes without changing y.
    """
    records = []
    for (model, variant), ys in outcomes.items():
        for i, y in enumerate(ys):
            pred = ("x" if (model, variant, i) in flip else "") + ("1" if y else "0")
            records.append(
                ExampleRecord(
                    run_id=name,
                    model=model,
                    task=task,
                    prompt_variant=variant,
                    scoring=scoring,
                    metric="exact_match",
                    seed=42,
                    exemplar_seed=None,
                    example_id=f"{task}/test/{i}",
                    example_index=i,
                    prompt="p",
                    target="1",
                    prediction=pred,
                    correct=bool(y),
                    prediction_norm=None,
                    correct_norm=None,
                    primary_correct=bool(y),
                    choice_logprobs=None,
                    choice_logprobs_norm=None,
                    generation="g",
                    parse_ok=True,
                    exemplar_ids=None,
                    n_prompt_tokens=1,
                    n_generated_tokens=1,
                    latency_ms=1.0,
                    status="ok",
                    error=None,
                )
            )
    models = list(dict.fromkeys(m for m, _ in outcomes))
    meta = {
        "config_hash": config_hash,
        "started_at": started,
        "config": {"seed": 42, "models": [{"name": m, "provider": "hf"} for m in models]},
    }
    write_run(root / name / "r", records_to_frame(records), meta)
    return root / name / "r"


def ones(k):
    return [1] * k + [0] * (N - k)


OUTCOMES = {
    ("big", "zero_shot"): ones(160),
    ("big", "cot"): ones(150),
    ("small", "zero_shot"): ones(60),
    ("small", "cot"): ones(62),
}


def test_family_tables_comparisons_and_verdicts(tmp_path):
    [fa] = analysis.analyze(analysis.load_runs([make_run(tmp_path, "run", OUTCOMES)]))
    assert (fa.task, fa.scoring, fa.metric) == ("gsm8k", "generation", "exact_match")
    assert fa.cells["acc"].tolist() == pytest.approx([0.80, 0.75, 0.30, 0.31])
    assert fa.per_model["acc"].tolist() == pytest.approx([0.775, 0.305])
    assert fa.per_prompt["prompt_variant"].tolist() == ["zero_shot", "cot"]
    kinds = [c.kind for c in fa.comparisons]
    assert kinds.count("exploratory") == 2 and len(kinds) == 6
    by = {(c.a, c.b): c for c in fa.comparisons}
    models = by[(("big", "zero_shot"), ("small", "zero_shot"))]
    assert (models.n10, models.n01) == (100, 0)
    assert models.verdict == "A > B" and models.diff == pytest.approx(0.5)
    assert models.mcnemar_bonf == pytest.approx(min(1.0, 4 * models.mcnemar_p))
    prompts = by[(("small", "zero_shot"), ("small", "cot"))]
    assert prompts.verdict in ("no conclusion", "underpowered — no conclusion")
    assert fa.mde["k"] == 4 and 0 < fa.mde["mde"] < 1


def test_combine_rejects_two_runs_with_the_same_cell(tmp_path):
    a = make_run(tmp_path, "a", {("big", "zero_shot"): ones(10)}, config_hash="x")
    b = make_run(tmp_path, "b", {("big", "zero_shot"): ones(10)}, config_hash="y")
    with pytest.raises(ValueError, match="overlap"):
        analysis.analyze(analysis.load_runs([a, b]))


def test_retest_agreement_counts_identical_predictions(tmp_path):
    first = make_run(tmp_path, "r1", OUTCOMES, started="2026-10-23T00:00:00")
    second = make_run(
        tmp_path,
        "r2",
        OUTCOMES,
        started="2026-10-23T01:00:00",
        flip={("big", "cot", 3), ("small", "cot", 7)},
    )
    retest = analysis.retest_agreement(analysis.load_runs([first, second]))
    assert retest["reruns"] == 2
    assert retest["overall"] == {
        "agree": 4 * N - 2,
        "n": 4 * N,
        "agreement": pytest.approx((4 * N - 2) / (4 * N)),
    }
    assert retest["conclusions"]["stability"] == 1.0
    # the reference analysis uses the earliest rerun only
    [fa] = analysis.analyze(analysis.load_runs([second, first]))
    assert fa.cells["n_ok"].tolist() == [N] * 4


def test_render_and_cli_write_report_with_hypothesis_result(tmp_path):
    make_run(tmp_path / "runs", "run", OUTCOMES)
    hyp = tmp_path / "h.yaml"
    zero = "prompt_variant: zero_shot"
    hyp.write_text(
        "- {id: H9, claim: big beats small, task: gsm8k, scoring: generation,"
        f" a: {{model: big, {zero}}}, b: {{model: small, {zero}}}}}\n"
        "- {id: H10, claim: small beats big, task: gsm8k, scoring: generation,"
        f" a: {{model: small, {zero}}}, b: {{model: big, {zero}}}}}\n"
    )
    out, js = tmp_path / "results.md", tmp_path / "results.json"
    assert (
        main(
            [
                "analyze",
                "--runs-root",
                str(tmp_path / "runs"),
                "--hypotheses",
                str(hyp),
                "--out",
                str(out),
                "--json",
                str(js),
                "--resamples",
                "200",
            ]
        )
        == 0
    )
    text = out.read_text()
    assert "| H9 | big beats small | 200 | +0.500 [" in text and "| supported |" in text
    assert "| H10 | small beats big |" in text and "contradicted (reverse holds)" in text
    for heading in ("### Cells", "### Per model", "### Per prompt variant", "Significance matrix"):
        assert heading in text
    payload = json.loads(js.read_text())
    assert payload["families"][0]["task"] == "gsm8k"
    assert len(payload["families"][0]["comparisons"]) == 6


def test_significance_matrix_marks_exploratory_pairs(tmp_path):
    [fa] = analysis.analyze(analysis.load_runs([make_run(tmp_path, "run", OUTCOMES)]))
    matrix = analysis_report.significance_matrix(fa)
    rows = matrix.splitlines()
    assert rows[0].startswith("|  | big · zero_shot | big · cot | small · zero_shot")
    assert "★" in rows[2]  # big · zero_shot beats small · zero_shot
    assert [cell.strip() for cell in rows[2].split("|")].count("·") == 1
