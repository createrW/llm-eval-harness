# llm-eval-harness — API specification

Canonical interface spec (design §6, §8, §9). The config JSON schema below is generated from the
code; regenerate it after any change to `src/llm_eval/config.py`:

```bash
uv run python -c "import json; from llm_eval.config import EvalConfig; print(json.dumps(EvalConfig.model_json_schema(), indent=2))"
```

## CLI

argparse, installed as the `llm-eval` console script (`llm_eval.cli:main`).

```
llm-eval run    --config configs/eval/smoke_qwen_hellaswag.yaml [--output-dir …] [--n-examples N] [--seed S] [--allow-errors]
llm-eval report --run-dir data/processed/runs/smoke_qwen_hellaswag/<run_id>      # markdown tables to stdout
scripts/eval.sh configs/eval/smoke_qwen_hellaswag.yaml                            # == uv run --extra local llm-eval run --config …
```

`llm-eval run`

| flag | required | meaning |
|---|---|---|
| `--config PATH` | yes | eval config YAML (§8.1 below) |
| `--output-dir DIR` | no | overrides `output_dir`; a relative path resolves against the project root |
| `--n-examples N` | no | overrides the top-level `n_examples` only; a per-task `n_examples` still wins |
| `--seed S` | no | overrides `seed` (subset and bootstrap streams) |
| `--allow-errors` | no | exit 0 even when example rows have `status == "error"` |

Overrides are applied before validation, so `config_hash` and `run.json["config"]` reflect them.
The project root is the nearest ancestor of the working directory that contains `pyproject.toml`,
else the working directory itself. `run` prints `run dir: <absolute path>` and then the summary
table (`acc` / `acc_norm` as `0.560 [0.470, 0.650]`, with `n`) on stdout; logs go to stderr at INFO.

Exit codes: `0` success; `1` when any cell raised (`run.json["failed_cells"]` is non-empty) or,
without `--allow-errors`, when any example row has `status == "error"`; `2` on argparse usage
errors. Completed cells are written either way.

`llm-eval report`

| flag | required | meaning |
|---|---|---|
| `--run-dir DIR` | yes | a run directory containing `examples.parquet` and `run.json` |

Prints the summary table stored in `run.json["summary"]`; exit code `0`.

## Config schema

Pydantic v2, `extra="forbid"` on every model (`additionalProperties: false`). Validation beyond
the schema: unique model names, `provider ∈ PROVIDERS`, task `name ∈ TASKS`, unique
`prompt_variants`, and the per-cell rejections of design §7.4 (raised by `expand_grid` before any
model loads). A task's `split: null` is filled with the task's default split.

```json
{
  "$defs": {
    "ModelConfig": {
      "additionalProperties": false,
      "description": "One model of the grid; ``name`` is the short id used in results.",
      "properties": {
        "name": {
          "title": "Name",
          "type": "string"
        },
        "provider": {
          "title": "Provider",
          "type": "string"
        },
        "path": {
          "title": "Path",
          "type": "string"
        },
        "revision": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Revision"
        },
        "dtype": {
          "default": "auto",
          "enum": [
            "auto",
            "float32",
            "float16",
            "bfloat16"
          ],
          "title": "Dtype",
          "type": "string"
        },
        "device": {
          "default": "auto",
          "enum": [
            "auto",
            "cpu",
            "mps",
            "cuda"
          ],
          "title": "Device",
          "type": "string"
        },
        "batch_size": {
          "default": 8,
          "minimum": 1,
          "title": "Batch Size",
          "type": "integer"
        },
        "max_new_tokens": {
          "anyOf": [
            {
              "type": "integer"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Max New Tokens"
        }
      },
      "required": [
        "name",
        "provider",
        "path"
      ],
      "title": "ModelConfig",
      "type": "object"
    },
    "PromptVariant": {
      "description": "How the query is presented: bare, with 3 exemplars, or with chain-of-thought (\u00a77.3).",
      "enum": [
        "zero_shot",
        "few_shot_3",
        "cot"
      ],
      "title": "PromptVariant",
      "type": "string"
    },
    "ScoringMode": {
      "description": "How a cell is scored: log-likelihood of choices, or free-text generation (design \u00a77.4).",
      "enum": [
        "loglikelihood",
        "generation"
      ],
      "title": "ScoringMode",
      "type": "string"
    },
    "TaskConfig": {
      "additionalProperties": false,
      "description": "One task of the grid; ``split``/``n_examples`` of ``None`` mean task default / inherit.",
      "properties": {
        "name": {
          "title": "Name",
          "type": "string"
        },
        "split": {
          "anyOf": [
            {
              "type": "string"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "Split"
        },
        "n_examples": {
          "anyOf": [
            {
              "minimum": 1,
              "type": "integer"
            },
            {
              "type": "null"
            }
          ],
          "default": null,
          "title": "N Examples"
        }
      },
      "required": [
        "name"
      ],
      "title": "TaskConfig",
      "type": "object"
    }
  },
  "additionalProperties": false,
  "description": "A full evaluation run: models \u00d7 tasks \u00d7 prompt variants (design \u00a78.1).",
  "properties": {
    "run_name": {
      "title": "Run Name",
      "type": "string"
    },
    "seed": {
      "default": 42,
      "title": "Seed",
      "type": "integer"
    },
    "exemplar_seed": {
      "anyOf": [
        {
          "type": "integer"
        },
        {
          "type": "null"
        }
      ],
      "default": null,
      "title": "Exemplar Seed"
    },
    "n_examples": {
      "default": 100,
      "minimum": 1,
      "title": "N Examples",
      "type": "integer"
    },
    "output_dir": {
      "default": "data/processed/runs",
      "title": "Output Dir",
      "type": "string"
    },
    "scoring": {
      "anyOf": [
        {
          "const": "auto",
          "type": "string"
        },
        {
          "$ref": "#/$defs/ScoringMode"
        }
      ],
      "default": "auto",
      "title": "Scoring"
    },
    "bootstrap_resamples": {
      "default": 1000,
      "minimum": 1,
      "title": "Bootstrap Resamples",
      "type": "integer"
    },
    "models": {
      "items": {
        "$ref": "#/$defs/ModelConfig"
      },
      "title": "Models",
      "type": "array"
    },
    "tasks": {
      "items": {
        "$ref": "#/$defs/TaskConfig"
      },
      "title": "Tasks",
      "type": "array"
    },
    "prompt_variants": {
      "items": {
        "$ref": "#/$defs/PromptVariant"
      },
      "minItems": 1,
      "title": "Prompt Variants",
      "type": "array"
    }
  },
  "required": [
    "run_name",
    "models",
    "tasks",
    "prompt_variants"
  ],
  "title": "EvalConfig",
  "type": "object"
}
```

Example (`configs/eval/smoke_qwen_hellaswag.yaml`):

```yaml
run_name: smoke_qwen_hellaswag      # required; results under <output_dir>/<run_name>/<run_id>/
seed: 42                            # subset + bootstrap streams (independent children, §5)
exemplar_seed: null                 # few-shot exemplar stream; null → seed
n_examples: 100                     # default for every task; TaskConfig.n_examples overrides
output_dir: data/processed/runs     # relative paths resolve against the project root; excluded from config_hash
scoring: auto                       # auto | loglikelihood | generation (resolved per cell, §7.4)
bootstrap_resamples: 1000
models:
  - name: qwen2.5-0.5b              # short id used in results; unique within a config
    provider: hf                    # must be a key of PROVIDERS (hf; Day 3: openai, anthropic)
    path: Qwen/Qwen2.5-0.5B
    revision: null                  # optional HF commit pin; the resolved commit is always recorded
    dtype: auto                     # auto (= float32) | float32 | float16 | bfloat16
    device: auto                    # auto | cpu | mps | cuda
    batch_size: 8
tasks:
  - name: hellaswag                 # must be in TASKS
    split: validation               # default comes from the task
prompt_variants: [zero_shot]        # subset of {zero_shot, few_shot_3, cot}; non-empty
```

## Results schema

`<output_dir>/<run_name>/<run_id>/examples.parquet`, `run_id = f"{started_at:%Y%m%dT%H%M%SZ}-{config_hash[:8]}"`.
One row per (cell, example), long format, written with `RESULTS_SCHEMA` (zstd). Invariant,
asserted by `write_run`: `(model, task, prompt_variant, example_id)` is unique.

| column | arrow type | meaning |
|---|---|---|
| run_id | string | directory name |
| model | string | `ModelConfig.name` |
| task | string | task name |
| prompt_variant | string | `zero_shot` / `few_shot_3` / `cot` |
| scoring | string | resolved mode |
| metric | string | `acc_norm` / `acc` / `exact_match` — what `primary_correct` measures |
| seed | int64 | config seed |
| exemplar_seed | int64, null | resolved exemplar seed for `few_shot_3`; null otherwise |
| example_id | string | `Example.id` |
| example_index | int64 | position in the selected subset (0-based) |
| prompt | string | exact rendered context / prompt sent to the model |
| target | string | canonical gold |
| prediction | string, null | canonical prediction (MC: index as str; generation: extracted answer, `""` if unparsable); null only when `status == "error"` |
| correct | bool, null | `task.is_correct(prediction)`; null only when `status == "error"` |
| prediction_norm | string, null | MC log-likelihood only |
| correct_norm | bool, null | MC log-likelihood only |
| primary_correct | bool, null | the outcome every downstream consumer uses (§7.6) |
| choice_logprobs | list<double>, null | MC only |
| choice_logprobs_norm | list<double>, null | MC only |
| generation | string, null | raw model text in generation mode |
| parse_ok | bool, null | generation mode: `extract_answer` found an answer; null in log-likelihood mode |
| exemplar_ids | list<string>, null | `few_shot_3`: exemplar `Example.id`s in prompt order |
| n_prompt_tokens | int64, null | MC: `n_context_tokens` of the shared context (counted once); generation: from `GenerationResult` |
| n_generated_tokens | int64, null | generation mode only |
| latency_ms | double | MC: sum over the example's K requests of (adapter batch wall time / requests in that batch); generation: per request |
| status | string | `ok` / `error` — `error` = infrastructure failure after retries (API error, timeout); local adapters raise instead |
| error | string, null | error message when `status == "error"` |

`read_run` returns a `pd.ArrowDtype`-backed frame; list columns come back as Python lists (`None`
when null), so wrap them in `np.asarray(...)` for numpy math.

## run.json

Provenance (design §7.5) plus the per-cell summary (§8.2):

| field | content |
|---|---|
| `schema_version` | `1` |
| `run_id` | the run directory name |
| `run_name` | `EvalConfig.run_name` |
| `config` | the post-override config, exactly as hashed (`EvalConfig.model_dump(mode="json")`) |
| `config_hash` | first 12 hex of the sha256 of the canonical JSON config, `output_dir` excluded |
| `started_at`, `finished_at` | ISO-8601 UTC timestamps |
| `git_sha`, `git_dirty` | from `git rev-parse HEAD` / `git status --porcelain`; env `LLM_EVAL_GIT_SHA` wins (then `git_dirty` is null); null when neither is available |
| `library_versions` | python, llm_eval, numpy, pandas, pyarrow, scipy, statsmodels, pydantic, and when installed torch, transformers, tokenizers, datasets, huggingface_hub |
| `models` | per model name: `{provider, path, revision, dtype, device}`, resolved (`revision` = the 40-hex snapshot commit, null for an injected model) |
| `tasks` | per task name: `{path, revision, split, n_rows, example_ids}` (`example_ids` in subset order) |
| `failed_cells` | `"model×task×variant"` names of the cells that raised |
| `summary` | one entry per cell, fields below |

`summary` entry: `{model, task, prompt_variant, scoring, metric, n, n_ok, n_error,
parse_failure_rate, acc, acc_ci_low, acc_ci_high, acc_norm, acc_norm_ci_low, acc_norm_ci_high,
primary_acc, primary_ci_low, primary_ci_high, ci_method="percentile_bootstrap", n_resamples,
ci_seed}` (`acc_norm*` null outside log-likelihood MC; `parse_failure_rate` null outside
generation mode).

## Python API

```python
# src/llm_eval/runner.py
@dataclass class RunResult: run_dir: Path; summary: pd.DataFrame; failed_cells: list[str]
def run_config(cfg: EvalConfig, *, project_root: Path | None = None, now: datetime | None = None) -> RunResult
    # project_root=None → nearest ancestor of cwd containing pyproject.toml, else cwd. Relative output_dir resolves against it.
def evaluate_cell(model: Model, task: Task, examples: Sequence[Example], variant: PromptVariant, scoring: ScoringMode,
                  *, run_id: str, seed: int, exemplar_seed: int | None = None) -> list[ExampleRecord]

# src/llm_eval/evaluation/results.py
def read_run(run_dir: Path) -> tuple[pd.DataFrame, dict]   # (ArrowDtype frame of examples.parquet, run.json)

# src/llm_eval/evaluation/report.py
def summarize(frame: pd.DataFrame, *, seed: int, n_resamples: int = 1000) -> pd.DataFrame   # one row per cell, the summary fields above
```

Minimal use:

```python
from pathlib import Path

from llm_eval.config import load_config
from llm_eval.evaluation.report import format_summary
from llm_eval.runner import run_config

result = run_config(load_config(Path("configs/eval/smoke_qwen_hellaswag.yaml"), overrides={"n_examples": 8}))
print(result.run_dir, format_summary(result.summary), sep="\n")
```
