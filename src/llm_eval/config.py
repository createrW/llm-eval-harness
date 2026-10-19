"""Eval config schema, grid expansion and config hash (design §6, §7.4, §8.1).

A YAML file validates into an :class:`EvalConfig` (pydantic v2, ``extra="forbid"`` everywhere).
:func:`expand_grid` turns it into :class:`GridCell` objects (models outer × tasks × variants) with
the scoring mode resolved per cell, rejecting unsupported cells before any model loads.
:func:`config_hash` is location- and device-independent. This module imports nothing heavy.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

import llm_eval.data  # noqa: F401  (registers every task in TASKS)
from llm_eval.data.dataset import TASKS, TaskType
from llm_eval.features.prompts import PromptVariant, UnsupportedCellError, check_supported
from llm_eval.models.base_model import ScoringMode
from llm_eval.models.registry import PROVIDERS


def _duplicates(values: list[Any]) -> list[Any]:
    return [v for v, n in Counter(values).items() if n > 1]


class ModelConfig(BaseModel):
    """One model of the grid; ``name`` is the short id used in results."""

    model_config = ConfigDict(extra="forbid")

    name: str
    provider: str
    path: str
    revision: str | None = None
    dtype: Literal["auto", "float32", "float16", "bfloat16"] = "auto"
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    batch_size: int = Field(default=8, ge=1)
    max_new_tokens: int | None = None

    @field_validator("provider")
    @classmethod
    def _known_provider(cls, v: str) -> str:
        if v not in PROVIDERS:
            raise ValueError(f"unknown provider {v!r}; known: {sorted(PROVIDERS)}")
        return v


class TaskConfig(BaseModel):
    """One task of the grid; ``split``/``n_examples`` of ``None`` mean task default / inherit."""

    model_config = ConfigDict(extra="forbid")

    name: str
    split: str | None = None
    n_examples: int | None = Field(default=None, ge=1)

    @field_validator("name")
    @classmethod
    def _known_task(cls, v: str) -> str:
        if v not in TASKS:
            raise ValueError(f"unknown task {v!r}; known: {sorted(TASKS)}")
        return v


class EvalConfig(BaseModel):
    """A full evaluation run: models × tasks × prompt variants (design §8.1)."""

    model_config = ConfigDict(extra="forbid")

    run_name: str
    seed: int = 42
    exemplar_seed: int | None = None
    n_examples: int = Field(default=100, ge=1)
    output_dir: str = "data/processed/runs"
    scoring: Literal["auto"] | ScoringMode = "auto"
    bootstrap_resamples: int = Field(default=1000, ge=1)
    models: list[ModelConfig]
    tasks: list[TaskConfig]
    prompt_variants: list[PromptVariant] = Field(min_length=1)

    @field_validator("models")
    @classmethod
    def _unique_model_names(cls, v: list[ModelConfig]) -> list[ModelConfig]:
        if dup := _duplicates([m.name for m in v]):
            raise ValueError(f"model names must be unique; duplicated: {dup}")
        return v

    @field_validator("prompt_variants")
    @classmethod
    def _unique_variants(cls, v: list[PromptVariant]) -> list[PromptVariant]:
        if dup := _duplicates(v):
            raise ValueError(
                f"prompt_variants must be unique; duplicated: {[d.value for d in dup]}"
            )
        return v

    @model_validator(mode="after")
    def _fill_default_splits(self) -> Self:
        self.tasks = [
            t
            if t.split is not None
            else t.model_copy(update={"split": TASKS[t.name].default_split})
            for t in self.tasks
        ]
        return self


@dataclass(frozen=True)
class GridCell:
    """One (model, task, variant) cell with its resolved scoring mode."""

    model: ModelConfig
    task: TaskConfig
    variant: PromptVariant
    scoring: ScoringMode


def load_config(path: Path, *, overrides: Mapping[str, Any] | None = None) -> EvalConfig:
    """Read YAML at ``path``, apply non-``None`` ``overrides`` (CLI flags), then validate."""
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: the top level of a config must be a mapping")
    raw.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return EvalConfig.model_validate(raw)


def config_hash(cfg: EvalConfig) -> str:
    """First 12 hex of the sha256 of the canonical JSON config, ``output_dir`` excluded (§6)."""
    payload = json.dumps(
        cfg.model_dump(mode="json", exclude={"output_dir"}), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _resolve_scoring(
    requested: Literal["auto"] | ScoringMode,
    supports_loglikelihood: bool,
    task_type: TaskType,
    variant: PromptVariant,
) -> ScoringMode:
    if requested != "auto":
        return ScoringMode(requested)
    if (
        supports_loglikelihood
        and task_type is TaskType.MULTIPLE_CHOICE
        and variant is not PromptVariant.COT
    ):
        return ScoringMode.LOGLIKELIHOOD
    return ScoringMode.GENERATION


def expand_grid(cfg: EvalConfig) -> list[GridCell]:
    """Models (outer) × tasks × variants in config order, scoring resolved per cell (§7.4)."""
    cells: list[GridCell] = []
    for model in cfg.models:
        spec = PROVIDERS[model.provider]
        for task in cfg.tasks:
            task_type = TASKS[task.name].task_type
            for variant in cfg.prompt_variants:
                scoring = _resolve_scoring(
                    cfg.scoring, spec.supports_loglikelihood, task_type, variant
                )
                cell = f"{model.name}×{task.name}×{variant.value}"
                if scoring is ScoringMode.LOGLIKELIHOOD and not spec.supports_loglikelihood:
                    raise UnsupportedCellError(
                        f"{cell}: provider {model.provider!r} has no log-likelihood scoring"
                    )
                check_supported(task_type, variant, scoring, cell=cell)
                cells.append(GridCell(model, task, variant, scoring))
    return cells
