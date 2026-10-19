import dataclasses
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from llm_eval.evaluation.results import (
    RESULTS_SCHEMA,
    ExampleRecord,
    read_run,
    records_to_frame,
    write_run,
)

COLUMNS = [
    "run_id",
    "model",
    "task",
    "prompt_variant",
    "scoring",
    "metric",
    "seed",
    "exemplar_seed",
    "example_id",
    "example_index",
    "prompt",
    "target",
    "prediction",
    "correct",
    "prediction_norm",
    "correct_norm",
    "primary_correct",
    "choice_logprobs",
    "choice_logprobs_norm",
    "generation",
    "parse_ok",
    "exemplar_ids",
    "n_prompt_tokens",
    "n_generated_tokens",
    "latency_ms",
    "status",
    "error",
]


def record(i, **over):
    base = dict(
        run_id="r",
        model="m",
        task="hellaswag",
        prompt_variant="zero_shot",
        scoring="loglikelihood",
        metric="acc_norm",
        seed=42,
        exemplar_seed=None,
        example_id=f"hellaswag/validation/{i}",
        example_index=i,
        prompt="ctx",
        target="1",
        prediction="1",
        correct=True,
        prediction_norm="0",
        correct_norm=False,
        primary_correct=False,
        choice_logprobs=(-1.0, -2.0),
        choice_logprobs_norm=(-0.5, -1.0),
        generation=None,
        parse_ok=None,
        exemplar_ids=None,
        n_prompt_tokens=3,
        n_generated_tokens=None,
        latency_ms=1.5,
        status="ok",
        error=None,
    )
    base.update(over)
    return ExampleRecord(**base)


def test_schema_has_exactly_the_design_columns_in_order():
    assert RESULTS_SCHEMA.names == COLUMNS
    assert [f.name for f in dataclasses.fields(ExampleRecord)] == COLUMNS
    assert RESULTS_SCHEMA.field("choice_logprobs").type == pa.list_(pa.float64())
    assert RESULTS_SCHEMA.field("exemplar_ids").type == pa.list_(pa.string())
    assert (
        RESULTS_SCHEMA.field("correct").type == pa.bool_()
        and RESULTS_SCHEMA.field("seed").type == pa.int64()
    )
    assert (
        RESULTS_SCHEMA.field("latency_ms").type == pa.float64()
        and RESULTS_SCHEMA.field("status").nullable is False
    )
    assert RESULTS_SCHEMA.field("error").nullable is True


def test_round_trip_preserves_values_schema_and_metadata(tmp_path):
    frame = records_to_frame(
        [record(0), record(1, correct=False, choice_logprobs=None, choice_logprobs_norm=None)]
    )
    meta = {"schema_version": 1, "run_id": "r", "git_sha": None, "summary": [{"acc": 0.5}]}
    write_run(tmp_path / "run", frame, meta)
    assert pq.read_schema(tmp_path / "run" / "examples.parquet").equals(RESULTS_SCHEMA)
    back, meta_back = read_run(tmp_path / "run")
    assert list(back.columns) == COLUMNS and len(back) == 2
    assert list(back["choice_logprobs"].iloc[0]) == [-1.0, -2.0]
    assert back["choice_logprobs"].iloc[1] is None or back["choice_logprobs"].isna().iloc[1]
    assert back["correct"].tolist() == [True, False]
    assert back["generation"].isna().all() and back["n_generated_tokens"].isna().all()
    assert meta_back == json.loads(json.dumps(meta))


def test_write_run_rejects_duplicate_cell_example_pairs(tmp_path):
    frame = records_to_frame([record(0), record(0)])
    with pytest.raises(ValueError, match="duplicate"):
        write_run(tmp_path / "run", frame, {})


def test_parquet_is_zstd_compressed(tmp_path):
    write_run(tmp_path / "run", records_to_frame([record(0)]), {})
    meta = pq.ParquetFile(tmp_path / "run" / "examples.parquet").metadata
    assert meta.row_group(0).column(0).compression == "ZSTD"
