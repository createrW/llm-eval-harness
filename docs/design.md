# llm-eval-harness — Technical Design

| | |
|---|---|
| Status | v0.1 — Day 1 of 7 (2026-10-18). Binding spec for `docs/superpowers/plans/`. |
| Owner | createrW |
| Layout standard | `/Users/zy/Project/new_sys/code_standard.md` (mapping in §4) |
| Hosting guidance | `cloud_hosting_plan_2026-10-17.md` → dashboard on Streamlit Community Cloud, eval runs local or rented GPU |

## 1. Intent

An open-source harness that evaluates small LLMs on benchmark tasks and reports results the way a
statistician would: every accuracy carries a bootstrap 95% confidence interval and an `n`; every
model-vs-model or prompt-vs-prompt claim is a *paired* test on the same examples, corrected for the
number of comparisons made; the variance in scores is decomposed into model, prompt and interaction
shares; and every number is reproducible from `(config, seed, code sha, model revision)`.

The gap it fills: typical "LLM eval" portfolio projects print a point estimate per model and call
the higher one better. With 300–500 examples, a 2–3 point gap is usually inside the noise. This
project is a showcase for Microsoft Applied Scientist / GE Vernova AI Evaluation / Mercor Research
Scientist style roles, so the statistical protocol *is* the product; the models and tasks are
deliberately small so the whole grid runs on a laptop.

## 2. Goals, non-goals, anti-goals

**Goals (from the brief)**

1. Config-driven runs: one YAML declares `models × tasks × prompt_variants`, `n_examples`, `seed`.
2. Local models via Hugging Face transformers: `Qwen/Qwen2.5-0.5B`, `meta-llama/Llama-3.2-1B`.
3. Optional API models: GPT-4o-mini (OpenAI), Claude Haiku (Anthropic). Keys only from env vars.
4. Tasks: HellaSwag (validation subset, n=500 final / n=100 Day 1) and GSM8K (test subset, n=300).
5. Prompt variants per task: `zero_shot`, `few_shot_3`, `cot`.
6. Metric: accuracy with bootstrap 95% CI (1000 resamples, seeded).
7. Pairwise significance: Wilcoxon signed-rank (as specified) alongside McNemar and a paired
   bootstrap of the difference (§7.6 explains why all three are shown), Bonferroni-corrected
   (Holm also reported).
8. Reports: per-model table, per-prompt table, pairwise significance matrix, variance
   decomposition (model / prompt / interaction / residual).
9. Streamlit dashboard: pick any two cells of the model×prompt grid → paired per-example analysis.
10. README: one-liner with the measured test–retest agreement over 3 reruns, experiment protocol
    (hypothesis → metrics → power analysis), example tables, how to add a task / model.

**Non-goals (from the brief)**: rich UI, LLM-as-judge (optional stretch), multi-GPU inference,
fine-tuning, leaderboard parity with lm-evaluation-harness (we follow its HellaSwag formatting so
numbers are comparable, but parity is not a goal).

**Anti-goals** (what would make this a failure even if it "works"):

- A reported accuracy without an interval and an `n`.
- A "model A beats model B" claim from unpaired comparisons or without multiple-comparison correction.
- Hidden nondeterminism: a rerun with the same config that cannot be compared example-by-example.
- Secrets or large artifacts in git (model weights, raw datasets, API keys).
- A dashboard that is prettier than it is correct.

## 3. Constraints

- **Stack is fixed**: Python 3.11, PyTorch + transformers (local inference), OpenAI + Anthropic
  SDKs (optional extra), statsmodels + scipy (statistics), Streamlit (dashboard), MLflow self-hosted
  file store (tracking), pytest. Package/venv management with `uv` (`uv.lock` committed;
  `requirements.txt` exported from it for Streamlit Community Cloud and for the code standard).
- **Layout** follows `code_standard.md` (§4). The package root is `src/llm_eval/` as the brief
  requires; the standard's sub-package taxonomy (`data/ models/ features/ evaluation/ serving/`)
  lives inside it.
- **Dev machine**: Apple M5, 16 GB unified memory, no CUDA. Everything must run on `mps` and on
  `cpu` (CI) and, unchanged, on a Linux CUDA box (rented GPU for the full grid).
- **Secrets**: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `HF_TOKEN` (Llama is gated) come from the
  environment only. Configs never contain keys.
- **Git**: repo is created and pushed by the user. The tree must be push-ready at the end of every
  day: `.gitignore` excludes data/runs/venv/mlruns; `data/samples/` holds one small committed run
  so tests and the dashboard work from a fresh clone with no downloads.
- **Time**: 7 days, one engineer. Day 1 scope is §11 "Day 1".

## 4. Repository layout (mapped to `code_standard.md`)

```
llm_eval_harness/
├── configs/
│   ├── eval/                       # == standard "train/": one YAML per eval run
│   │   ├── smoke_qwen_hellaswag.yaml       # Day 1: Qwen2.5-0.5B × HellaSwag × zero_shot × n=100
│   │   └── full_grid.yaml                  # Day 2+: 2 models × 2 tasks × 3 variants
│   └── deploy/
│       └── dashboard.yaml          # Day 6: which run dirs the dashboard reads
├── data/                           # .gitignore'd except samples/
│   ├── raw/                        # local dataset snapshots (optional; default is the HF cache)
│   ├── processed/runs/<run_name>/<run_id>/   # per-example parquet + run.json (§8.2)
│   └── samples/runs/...            # one committed 100-example run for tests + dashboard demo
├── docs/
│   ├── design.md                   # this document
│   ├── experiment_log.md           # every run worth keeping: date, config hash, headline numbers
│   ├── api_spec.md                 # CLI, Python API, config schema, results schema
│   └── superpowers/plans/          # implementation plans (process artifacts)
├── notebooks/                      # EDA / prototypes (Day 4+: bootstrap sanity notebook)
├── src/llm_eval/                   # the package
│   ├── __init__.py
│   ├── config.py                   # YAML → pydantic models → run grid
│   ├── runner.py                   # orchestration: grid → records → parquet
│   ├── cli.py                      # `llm-eval run|report`
│   ├── data/                       # == standard data/: tasks = dataset loading + preprocessing
│   │   ├── dataset.py              # Example, Task ABC, task registry
│   │   ├── hellaswag.py            # HellaSwagTask
│   │   └── gsm8k.py                # Day 2
│   ├── models/                     # == standard models/: inference adapters, no training
│   │   ├── base_model.py           # Model protocol + request/result dataclasses
│   │   ├── hf_model.py             # HFCausalLM (transformers, loglikelihood + generate)
│   │   ├── api_model.py            # Day 3: OpenAI / Anthropic (generate only)
│   │   └── registry.py             # provider string → adapter
│   ├── features/                   # == standard features/: prompt engineering is our feature eng.
│   │   └── prompts.py              # PromptVariant, rendering of MC / generation prompts
│   ├── evaluation/
│   │   ├── metrics.py              # accuracy, acc_norm, MC scoring from logprobs
│   │   ├── results.py              # ExampleRecord, parquet schema, read/write run dirs
│   │   ├── report.py               # summary tables (markdown)
│   │   ├── stats.py                # Day 4: bootstrap CI, Wilcoxon/McNemar, corrections
│   │   ├── variance.py             # Day 5: ANOVA decomposition
│   │   └── tracking.py             # Day 5: MLflow logging
│   └── serving/
│       └── app.py                  # Day 6: Streamlit dashboard (== standard serving/app.py)
├── tests/                          # unit tests offline; `-m slow` for real-model tests
├── scripts/
│   ├── eval.sh                     # run one config
│   └── serve.sh                    # Day 6: launch dashboard
├── pyproject.toml  uv.lock  requirements.txt  Dockerfile  .gitignore  README.md  LICENSE
```

Mapping notes: the standard's `train/` config folder becomes `eval/` (we do not train);
`features/` holds prompt construction because turning a raw example into model input is this
project's feature engineering; `serving/` is the dashboard, not a prediction API; `setup.py` is
replaced by `pyproject.toml` (same role); `requirements.txt` is generated, never hand-edited.

## 5. Architecture

```mermaid
flowchart LR
    Y[YAML config] --> C[config.py<br/>pydantic EvalConfig]
    C --> G[grid: model × task × variant]
    G --> R[runner.py]
    R --> T[data/*Task<br/>load + preprocess + subset(seed)]
    R --> P[features/prompts.py<br/>render variant]
    R --> M[models/*<br/>loglikelihood | generate]
    T --> P --> M --> S[evaluation/metrics.py<br/>score per example]
    S --> W[evaluation/results.py<br/>examples.parquet + run.json]
    W --> ST[evaluation/stats.py<br/>bootstrap · paired tests · corrections]
    W --> V[evaluation/variance.py]
    ST --> RP[report.py tables]
    V --> RP
    W --> D[serving/app.py Streamlit]
    RP --> ML[tracking.py MLflow]
```

Design rules:

- **Records are the contract.** Everything downstream (stats, dashboard, MLflow) consumes the
  long-format per-example table from §8.2. Day 1 must get this schema right because Days 4–6 only
  read it.
- **Tasks know text, models know tokens, variants know layout.** A task converts a raw dataset row
  into an `Example` and knows how to render its *context* and *choices* as strings and how to judge
  a prediction. A model adapter scores `(context, continuation)` pairs or generates text. A prompt
  variant decides how exemplars / instructions wrap the task text. None of the three imports the
  others' internals.
- **Two scoring modes.** `loglikelihood` (multiple choice; local models only) and `generation`
  (free text; all models). The capability matrix is in §7.4.
- **Registries, not if-chains.** `data.dataset.TASKS` and `models.registry.PROVIDERS` are dicts
  populated by decorators; "add a task" = one file + one `@register_task`.

## 6. Core abstractions (names are binding for the plan)

```python
# src/llm_eval/data/dataset.py
class TaskType(str, Enum): MULTIPLE_CHOICE = "multiple_choice"; GENERATION = "generation"

@dataclass(frozen=True)
class Example:
    id: str                 # f"{task}/{split}/{native_id}", stable across runs
    inputs: dict[str, Any]  # task-native fields after preprocessing
    target: str             # canonical gold answer ("2" for MC index, "42" for GSM8K)
    choices: tuple[str, ...] | None = None   # MC continuations WITHOUT the leading space

class Task(ABC):
    name: ClassVar[str]; task_type: ClassVar[TaskType]; default_split: ClassVar[str]
    def load(self, split: str, n_examples: int | None, seed: int) -> list[Example]  # §7.5 subset rule
    def context(self, example: Example) -> str          # rendered question/context, no exemplars
    def choices(self, example: Example) -> tuple[str, ...]
    def gold_index(self, example: Example) -> int       # MC only
    def is_correct(self, prediction: str, example: Example) -> bool
    def exemplars(self, k: int, seed: int) -> list[Example]        # Day 2 (few-shot pool from train)
    def extract_answer(self, generation: str) -> str               # Day 2 (generation mode)

TASKS: dict[str, type[Task]];  def register_task(cls) -> cls;  def get_task(name) -> Task
```

```python
# src/llm_eval/models/base_model.py
@dataclass(frozen=True) class LoglikelihoodRequest: context: str; continuation: str
@dataclass(frozen=True) class LoglikelihoodResult:  logprob: float; n_tokens: int   # sum over continuation tokens
@dataclass(frozen=True) class GenerationRequest:    prompt: str; max_new_tokens: int; stop: tuple[str, ...] = ()

class Model(Protocol):
    name: str
    supports_loglikelihood: bool
    def loglikelihood(self, requests: Sequence[LoglikelihoodRequest]) -> list[LoglikelihoodResult]
    def generate(self, requests: Sequence[GenerationRequest]) -> list[str]
```

```python
# src/llm_eval/models/hf_model.py
class HFCausalLM:
    def __init__(self, model, tokenizer, *, name: str, device: str, batch_size: int)   # injectable for tests
    @classmethod
    def from_pretrained(cls, path: str, *, name: str, revision: str | None, dtype: str, device: str, batch_size: int) -> "HFCausalLM"
```

```python
# src/llm_eval/features/prompts.py
class PromptVariant(str, Enum): ZERO_SHOT = "zero_shot"; FEW_SHOT_3 = "few_shot_3"; COT = "cot"
SUPPORTED_VARIANTS: frozenset[PromptVariant]          # Day 1: {ZERO_SHOT}
@dataclass(frozen=True) class MCPrompt: context: str; continuations: tuple[str, ...]   # continuation = " " + choice
def render_mc_loglikelihood(task: Task, example: Example, variant: PromptVariant, exemplars: Sequence[Example] = ()) -> MCPrompt
```

```python
# src/llm_eval/evaluation/metrics.py
@dataclass(frozen=True) class MCScore: pred: int; pred_norm: int; logprobs: tuple[float, ...]; logprobs_norm: tuple[float, ...]
def score_multiple_choice(results: Sequence[LoglikelihoodResult], continuations: Sequence[str]) -> MCScore
def accuracy(correct: Sequence[bool]) -> float          # mean; raises on empty
```

```python
# src/llm_eval/evaluation/results.py
@dataclass(frozen=True) class ExampleRecord: ...        # exactly the columns of §8.2
def records_to_frame(records) -> pd.DataFrame
def write_run(run_dir: Path, frame: pd.DataFrame, metadata: dict) -> None   # examples.parquet + run.json
def read_run(run_dir: Path) -> tuple[pd.DataFrame, dict]
```

```python
# src/llm_eval/config.py (pydantic v2, extra="forbid" everywhere)
class ModelConfig(BaseModel): name; provider: Literal["hf","openai","anthropic"]; path; revision=None; dtype="auto"; device="auto"; batch_size=8; max_new_tokens=256
class TaskConfig(BaseModel):  name; split=None; n_examples=None
class EvalConfig(BaseModel):  run_name; seed=42; n_examples=100; output_dir="data/processed/runs"; models: list[ModelConfig]; tasks: list[TaskConfig]; prompt_variants: list[PromptVariant]; scoring: Literal["auto","loglikelihood","generation"]="auto"
def load_config(path) -> EvalConfig;  def config_hash(cfg) -> str (sha256 of canonical JSON, first 12 hex);  def expand_grid(cfg) -> list[GridCell]
```

```python
# src/llm_eval/runner.py
@dataclass class RunResult: run_dir: Path; summary: pd.DataFrame
def evaluate_cell(model: Model, task: Task, examples: list[Example], variant: PromptVariant, scoring: ScoringMode) -> list[ExampleRecord]
def run_config(cfg: EvalConfig, *, project_root: Path | None = None, now: datetime | None = None) -> RunResult
```

## 7. Evaluation methodology

### 7.1 HellaSwag (multiple choice, log-likelihood)

Dataset `Rowan/hellaswag` (HF, ungated), fields `ind, activity_label, ctx_a, ctx_b, ctx, endings[4],
label, split, source_id`. We use the **validation** split (10,042 rows; `test` has no labels).

Preprocessing follows lm-evaluation-harness exactly so our numbers are comparable to published ones:

```python
def _preprocess(text: str) -> str:
    text = text.strip()
    text = text.replace(" [title]", ". ")        # WikiHow artefacts
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")

ctx     = row["ctx_a"] + " " + row["ctx_b"].capitalize()
context = _preprocess(row["activity_label"] + ": " + ctx)
choices = tuple(_preprocess(e) for e in row["endings"])
gold    = int(row["label"])
```

Scoring: for each choice `c_i`, `ll_i = Σ log p(tokens of " " + c_i | context)` (continuation gets a
leading space, context does not get a trailing one). `pred = argmax ll_i` → **acc**.
`pred_norm = argmax ll_i / len((" " + c_i).encode("utf-8"))` → **acc_norm** (byte-length
normalisation removes the bias toward short endings; it is the number leaderboards report for
HellaSwag). Both are stored per example; reports show both and label `acc_norm` as the headline for
log-likelihood scoring.

Sanity anchors (approximate, from public leaderboards; used only to catch pipeline bugs):
Qwen2.5-0.5B zero-shot `acc_norm` ≈ 0.50–0.55 and `acc` ≈ 0.38–0.45; Llama-3.2-1B `acc_norm`
≈ 0.60–0.66. At n=100 the 95% CI half-width is about ±0.10, so the Day 1 smoke run validates the
pipeline, not the model.

### 7.2 GSM8K (generation, exact match) — Day 2

Dataset `openai/gsm8k` config `main`, split `test`, n=300. Target is the number after `####` in
the answer field. Prediction = last number in the generation (after `The answer is` if present),
compared after stripping commas, `$`, trailing `.`, and normalising `3.0 == 3`. Greedy decoding,
`max_new_tokens` 256 (zero/few-shot) and 512 (cot); stop on `\n\nQuestion:`.

### 7.3 Prompt variants

| variant | HellaSwag (MC) | GSM8K (generation) |
|---|---|---|
| `zero_shot` | context → score 4 continuations | `Question: …\nAnswer:` → extract number |
| `few_shot_3` | 3 exemplars (context + correct ending) joined by `\n\n`, then the query | 3 `Question/Answer` exemplars with full solutions, then the query |
| `cot` | generation mode: lettered choices + "Let's think step by step." → extract letter | `Question: …\nAnswer: Let's think step by step.` → extract number |

Exemplars are drawn from the task's **train** split with `numpy.random.default_rng(seed)` once per
run and are identical for every model in that run (paired design). Their ids are stored in
`run.json`.

### 7.4 Scoring-mode capability matrix

| | hf × zero/few-shot MC | hf × cot | hf × GSM8K | openai / anthropic |
|---|---|---|---|---|
| `loglikelihood` | ✅ default | ✗ (cot needs generation) | ✗ | ✗ (no prompt logprobs) |
| `generation` | ✅ if forced (`scoring: generation`) | ✅ | ✅ | ✅ |

`scoring: auto` resolves per cell to `loglikelihood` when the model supports it and the cell is a
non-cot MC task, else `generation`. An explicit impossible combination is a config error, raised
before any model is loaded. The resolved mode is a column in the results so cross-mode numbers are
never silently mixed in a table.

### 7.5 Determinism and provenance

- **Subset selection**: `rng = numpy.random.default_rng(seed)`; `idx = sorted(rng.choice(len(split), size=n, replace=False))`.
  The chosen native ids are written to `run.json["example_ids"]` so a rerun is verifiable without
  trusting the RNG.
- **Decoding**: greedy, `temperature=0`, no sampling anywhere; `torch.manual_seed(seed)` set before
  model load (irrelevant for greedy, defensive for dropout-free eval).
- **Numerics**: dtype `auto` = `bfloat16` on CUDA, `float32` on MPS/CPU (bf16 kernels on MPS are
  incomplete and log-likelihood ties flip with half precision). The resolved dtype and device are in
  `run.json`; test–retest comparisons across devices are reported as such, not as "reruns".
- **Batch invariance**: log-likelihood scoring pads on the right and masks; batched and unbatched
  scores must agree to 1e-4 (a unit test asserts it on a tiny random model).
- **Provenance in `run.json`**: `git_sha`, `git_dirty`, `config` (resolved), `config_hash`,
  `library_versions` (torch, transformers, datasets), `model_revision` (HF commit hash when
  available), `dataset_revision`, `device`, `dtype`, `started_at`, `finished_at`, `example_ids`.
- **Test–retest** (README claim): agreement = fraction of (cell, example) pairs with identical
  `prediction` across 3 reruns of the same config on the same device. Expected ≈ 1.0 for
  log-likelihood cells; CoT generation on MPS may show small drift, which is exactly what the
  number is for. Measured on Day 7, placeholder until then.

### 7.6 Statistics (Day 4–5; designed now because the schema must support it)

- **Bootstrap CI** (`stats.py`): percentile interval over `n_resamples=1000` resamples of the
  per-example `correct` vector, `numpy.random.default_rng(seed)`; cross-checked in a test against
  `scipy.stats.bootstrap(method="percentile")`. BCa offered as an option.
- **Paired comparisons**: cells are compared on the *same* example ids (inner join on
  `example_id`). Per the brief we run `scipy.stats.wilcoxon` on paired 0/1 differences. On binary
  outcomes every non-zero difference has |d| = 1, so Wilcoxon reduces to the sign test on discordant
  pairs; **McNemar's exact test** (`statsmodels.stats.contingency_tables.mcnemar`) is the textbook
  test for exactly this design and is reported next to it, together with a paired bootstrap CI on
  `acc_A − acc_B`. The report prints all three; a reader who only trusts one still gets a verdict.
- **Multiple comparisons**: Bonferroni as specified plus Holm (`statsmodels.stats.multitest.multipletests`);
  the matrix cell shows corrected p and a ★ at α = 0.05.
- **Variance decomposition** (`variance.py`): two-way ANOVA (`statsmodels.formula.api.ols` +
  `anova_lm`, type II) of `correct ~ C(model) * C(prompt_variant)` per task, reporting sum-of-squares
  shares (η²) for model, prompt, interaction, residual. Caveat stated in the report: a linear
  probability model on binary outcomes is descriptive, not inferential; the paired tests carry the
  inference.
- **Power** (README protocol): for the paired McNemar design, the n needed to detect Δ = 0.05 at
  α = 0.05/k with 80% power given the observed discordance rate; computed from the smoke-run
  discordance so the README's n=500/300 choice is justified by data, not vibes.

## 8. Data contracts

### 8.1 Config (YAML)

```yaml
run_name: smoke_qwen_hellaswag      # required; results under <output_dir>/<run_name>/<run_id>/
seed: 42
n_examples: 100                     # default for every task; TaskConfig.n_examples overrides
output_dir: data/processed/runs     # relative paths resolve against the project root
scoring: auto                       # auto | loglikelihood | generation
models:
  - name: qwen2.5-0.5b              # short id used in results; unique within a config
    provider: hf                    # hf | openai | anthropic
    path: Qwen/Qwen2.5-0.5B
    revision: null                  # optional HF commit pin
    dtype: auto                     # auto | float32 | float16 | bfloat16
    device: auto                    # auto | cpu | mps | cuda
    batch_size: 8
    max_new_tokens: 256
tasks:
  - name: hellaswag                 # must be in TASKS registry
    split: validation               # default comes from the task
prompt_variants: [zero_shot]        # subset of {zero_shot, few_shot_3, cot}; non-empty
```

Validation (pydantic, `extra="forbid"`): `n_examples ≥ 1`; unique model names; known task names;
non-empty `prompt_variants`; variants not in `SUPPORTED_VARIANTS` rejected with a message naming
the day they land; `scoring: loglikelihood` with `cot` or with an API provider rejected.
`expand_grid(cfg)` returns `[(model_cfg, task_cfg, variant, resolved_scoring)]` in config order.

### 8.2 Results

`<output_dir>/<run_name>/<run_id>/` where `run_id = f"{started_at:%Y%m%dT%H%M%SZ}-{config_hash[:8]}"`.

`examples.parquet` — one row per (cell, example), long format:

| column | type | meaning |
|---|---|---|
| run_id | str | directory name |
| model | str | `ModelConfig.name` |
| task | str | task name |
| prompt_variant | str | `zero_shot` / `few_shot_3` / `cot` |
| scoring | str | resolved mode |
| seed | int | config seed |
| example_id | str | `Example.id` |
| example_index | int | position in the selected subset (0-based) |
| prompt | str | exact rendered context sent to the model |
| target | str | canonical gold |
| prediction | str | canonical prediction (MC: index as str; generation: extracted answer) |
| correct | bool | task's `is_correct(prediction)` |
| prediction_norm | str \| null | MC loglikelihood only |
| correct_norm | bool \| null | MC loglikelihood only |
| choice_logprobs | list[float] \| null | MC only |
| choice_logprobs_norm | list[float] \| null | MC only |
| generation | str \| null | raw model text in generation mode |
| n_prompt_tokens | int \| null | |
| n_generated_tokens | int \| null | |
| latency_ms | float | wall time for this example's forward passes (batch time / batch size) |

`run.json` — provenance (§7.5) plus `summary`: list of `{model, task, prompt_variant, scoring, n, acc, acc_norm}`.

### 8.3 Samples policy

`data/samples/runs/smoke_qwen_hellaswag/<run_id>/` is committed once from the real Day 1 run
(≈100 rows, < 200 KB). Unit tests for `report.py`, `stats.py` and the dashboard read it; nothing in
`tests/` touches the network unless marked `slow`.

## 9. CLI and scripts

```
llm-eval run    --config configs/eval/smoke_qwen_hellaswag.yaml [--output-dir …] [--n-examples N] [--seed S]
llm-eval report --run-dir data/processed/runs/smoke_qwen_hellaswag/<run_id>   # markdown tables to stdout
scripts/eval.sh configs/eval/smoke_qwen_hellaswag.yaml                        # == uv run llm-eval run --config …
```

`run` prints the run directory and the summary table; exit code 0 only if every cell completed.
argparse only (no click).

## 10. Testing strategy

- **Offline unit tests** (default `pytest`): config validation; HellaSwag preprocessing against
  hand-written literals; deterministic subset selection; prompt rendering exact strings; MC scoring
  from hand-computed logprobs; parquet round trip and schema; runner end-to-end with a `FakeModel`
  whose log-likelihoods are a fixed function of the continuation text (so expected accuracy is
  derivable by hand); `HFCausalLM` on a **randomly initialised tiny Llama** built from `LlamaConfig`
  with an in-memory `tokenizers` WordLevel vocabulary — no download — asserting (a) log-likelihoods
  match a manual `log_softmax` computation, (b) batched == unbatched, (c) continuation token counts.
- **Slow tests** (`pytest -m slow`): real `Qwen/Qwen2.5-0.5B` on 8 HellaSwag examples: finishes,
  accuracy ∈ [0, 1], two consecutive calls give identical predictions.
- **CI** (Day 7, GitHub Actions): `uv sync`, `ruff`, `pytest` (offline only), on `ubuntu-latest`.

## 11. Day-by-day mapping

| Day | Deliverable | Modules |
|---|---|---|
| 1 | scaffold; config schema; HellaSwag zero-shot log-likelihood on Qwen2.5-0.5B, n=100; parquet; README skeleton | config, data/dataset, data/hellaswag, models/base_model, models/hf_model, models/registry, features/prompts, evaluation/metrics, evaluation/results, evaluation/report, runner, cli |
| 2 | GSM8K; few_shot_3 + cot; generation scoring; Llama-3.2-1B | data/gsm8k, prompts, hf_model.generate, runner |
| 3 | OpenAI / Anthropic adapters (generation only), retries, cost log | models/api_model |
| 4 | bootstrap CI; Wilcoxon + McNemar + paired bootstrap; Bonferroni/Holm matrix | evaluation/stats, report |
| 5 | variance decomposition; MLflow logging of every run | evaluation/variance, tracking |
| 6 | Streamlit dashboard: grid → pick two cells → paired analysis | serving/app |
| 7 | README (protocol, power analysis, test–retest number), architecture diagram, CI, samples, push | docs |

## 12. Decisions made by the author (not in the brief)

1. **Log-likelihood scoring for HellaSwag** instead of letter generation: standard for base models
   this small (0.5B base models do not reliably emit "B"), comparable to published numbers, and it
   makes `acc_norm` available. Generation-mode MC exists for `cot` and API models.
2. **McNemar + paired bootstrap next to Wilcoxon**: Wilcoxon is kept because the brief names it,
   but on paired binary data it is the sign test; showing the textbook test prevents a reviewer from
   dismissing the analysis.
3. **numpy `default_rng` for all sampling** (not `random`): stable bit-streams across Python
   versions; ids are persisted anyway.
4. **`float32` on MPS/CPU**: correctness over speed at 0.5–1B scale.
5. **One parquet per run (long format)** rather than one per cell: the paired analyses need the
   join anyway and the dashboard loads one file.
6. **uv + `uv.lock`**; `requirements.txt` exported for the code standard / Streamlit Cloud.
7. **argparse CLI**, no extra dependency.
8. **Tiny random Llama for adapter tests**: real transformers code path without network.

## 13. Left to the builder

Internal structure of `hf_model.py` batching; progress bars; log formatting; exact wording of
error messages; whether `report.py` uses `pandas.DataFrame.to_markdown` (needs `tabulate`) or a
hand-rolled formatter (preferred: hand-rolled, no new dependency).

## 14. Risks

| Risk | Mitigation |
|---|---|
| transformers 5.x API drift (e.g. `torch_dtype` → `dtype`) | pin `transformers>=4.51,<6`; adapter test on tiny model runs in CI |
| MPS kernel gaps / nondeterminism | float32; `device: cpu` fallback documented; test–retest measured and reported |
| Llama-3.2-1B gated | `HF_TOKEN` + licence acceptance documented; Day 1 uses Qwen only |
| Streamlit Cloud 1 GB RAM | dashboard reads parquet only; no torch import in `serving/` |
| n too small for claims | CIs always shown; power analysis in README justifies n |
