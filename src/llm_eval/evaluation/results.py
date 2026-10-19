"""Results schema and run-directory IO (design §6, §8.2).

A run directory holds ``examples.parquet`` (one row per (cell, example), long format, written with
the explicit :data:`RESULTS_SCHEMA`, zstd-compressed) and ``run.json`` (provenance plus summary).
Frames are pandas frames backed by ``pd.ArrowDtype`` so nullable ints/bools and list columns
round-trip without float or object coercion. This module imports neither torch nor datasets.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

EXAMPLES_FILE = "examples.parquet"
RUN_FILE = "run.json"
UNIQUE_KEY = ["model", "task", "prompt_variant", "example_id"]

RESULTS_SCHEMA = pa.schema(
    [
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("model", pa.string(), nullable=False),
        pa.field("task", pa.string(), nullable=False),
        pa.field("prompt_variant", pa.string(), nullable=False),
        pa.field("scoring", pa.string(), nullable=False),
        pa.field("metric", pa.string(), nullable=False),
        pa.field("seed", pa.int64(), nullable=False),
        pa.field("exemplar_seed", pa.int64()),
        pa.field("example_id", pa.string(), nullable=False),
        pa.field("example_index", pa.int64(), nullable=False),
        pa.field("prompt", pa.string(), nullable=False),
        pa.field("target", pa.string(), nullable=False),
        pa.field("prediction", pa.string()),
        pa.field("correct", pa.bool_()),
        pa.field("prediction_norm", pa.string()),
        pa.field("correct_norm", pa.bool_()),
        pa.field("primary_correct", pa.bool_()),
        pa.field("choice_logprobs", pa.list_(pa.float64())),
        pa.field("choice_logprobs_norm", pa.list_(pa.float64())),
        pa.field("generation", pa.string()),
        pa.field("parse_ok", pa.bool_()),
        pa.field("exemplar_ids", pa.list_(pa.string())),
        pa.field("n_prompt_tokens", pa.int64()),
        pa.field("n_generated_tokens", pa.int64()),
        pa.field("latency_ms", pa.float64(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("error", pa.string()),
    ]
)


@dataclass(frozen=True)
class ExampleRecord:
    """One row of ``examples.parquet``: exactly the §8.2 columns, in schema order."""

    run_id: str
    model: str
    task: str
    prompt_variant: str
    scoring: str
    metric: str
    seed: int
    exemplar_seed: int | None
    example_id: str
    example_index: int
    prompt: str
    target: str
    prediction: str | None
    correct: bool | None
    prediction_norm: str | None
    correct_norm: bool | None
    primary_correct: bool | None
    choice_logprobs: tuple[float, ...] | None
    choice_logprobs_norm: tuple[float, ...] | None
    generation: str | None
    parse_ok: bool | None
    exemplar_ids: tuple[str, ...] | None
    n_prompt_tokens: int | None
    n_generated_tokens: int | None
    latency_ms: float
    status: str
    error: str | None


def records_to_frame(records: Sequence[ExampleRecord]) -> pd.DataFrame:
    """Build an ArrowDtype-backed frame with exactly the :data:`RESULTS_SCHEMA` columns."""
    table = pa.Table.from_pylist([asdict(r) for r in records], schema=RESULTS_SCHEMA)
    return table.to_pandas(types_mapper=pd.ArrowDtype)


def write_run(run_dir: Path, frame: pd.DataFrame, metadata: dict[str, Any]) -> None:
    """Write ``examples.parquet`` and ``run.json`` into ``run_dir`` (created if missing).

    Raises ``ValueError`` if ``(model, task, prompt_variant, example_id)`` is not unique.
    """
    duplicated = frame.duplicated(subset=UNIQUE_KEY, keep=False)
    if bool(duplicated.any()):
        keys = frame.loc[duplicated, UNIQUE_KEY].drop_duplicates().to_numpy().tolist()
        raise ValueError(f"duplicate (model, task, prompt_variant, example_id) rows: {keys}")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, schema=RESULTS_SCHEMA, preserve_index=False)
    pq.write_table(table, run_dir / EXAMPLES_FILE, compression="zstd")
    with (run_dir / RUN_FILE).open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True, default=str)


def read_run(run_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Read a run directory back as ``(ArrowDtype frame, run.json metadata)``."""
    run_dir = Path(run_dir)
    frame = pq.read_table(run_dir / EXAMPLES_FILE).to_pandas(types_mapper=pd.ArrowDtype)
    with (run_dir / RUN_FILE).open(encoding="utf-8") as fh:
        metadata = json.load(fh)
    return frame, metadata
