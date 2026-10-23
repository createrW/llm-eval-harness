# tests/test_smoke_sample.py
"""The committed Day 1 smoke run (design §8.3) reads back and reproduces its own summary."""

import math
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from llm_eval.cli import main
from llm_eval.evaluation.report import summarize
from llm_eval.evaluation.results import EXAMPLES_FILE, RESULTS_SCHEMA, read_run

SAMPLE_ROOT = Path(__file__).resolve().parents[1] / "data" / "samples" / "runs"


@pytest.fixture(scope="module")
def smoke_run_dir():
    dirs = sorted(p for p in (SAMPLE_ROOT / "smoke_qwen_hellaswag").glob("*") if p.is_dir())
    assert len(dirs) == 1, dirs  # found by glob, never by a hard-coded run_id (§8.3)
    return dirs[0]


def test_smoke_sample_has_the_results_schema_and_provenance(smoke_run_dir):
    assert pq.read_schema(smoke_run_dir / EXAMPLES_FILE).equals(RESULTS_SCHEMA)
    frame, meta = read_run(smoke_run_dir)
    assert len(frame) == meta["config"]["n_examples"] == 100
    assert (frame["status"] == "ok").all() and meta["failed_cells"] == []
    assert meta["run_id"] == smoke_run_dir.name
    assert list(frame["example_id"]) == meta["tasks"]["hellaswag"]["example_ids"]
    [model] = meta["models"].values()
    assert model["dtype"] == "float32" and len(model["revision"]) == 40


def test_smoke_sample_summary_is_reproducible_from_its_examples(smoke_run_dir):
    frame, meta = read_run(smoke_run_dir)
    [stored] = meta["summary"]
    [fresh] = summarize(
        frame, seed=meta["config"]["seed"], n_resamples=meta["config"]["bootstrap_resamples"]
    ).to_dict("records")
    assert fresh.keys() == stored.keys()
    for key, value in stored.items():
        if isinstance(value, float):
            assert math.isclose(fresh[key], value, abs_tol=1e-12), key
        else:
            assert fresh[key] == value, key


def test_cli_report_prints_the_smoke_sample_row(smoke_run_dir, capsys):
    assert main(["report", "--run-dir", str(smoke_run_dir)]) == 0
    assert (
        "| qwen2.5-0.5b | hellaswag | zero_shot | acc_norm | 100 | 0 |" in capsys.readouterr().out
    )
