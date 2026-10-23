# llm-eval-harness

Evaluate small LLMs on benchmark tasks and report the results the way a statistician would: every
accuracy carries a seeded bootstrap 95% CI and an `n`, and every "A beats B" claim is a paired,
multiplicity-corrected test under a decision rule written before the data exist.

## Results

Qwen2.5-0.5B, HellaSwag validation, 100 examples (seed 42), zero-shot log-likelihood, float32 on MPS; interval = seeded percentile bootstrap (1000 resamples).

| model | task | prompt_variant | metric | n_ok | n_error | acc | acc_norm |
|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b | hellaswag | zero_shot | acc_norm | 100 | 0 | 0.440 [0.340, 0.540] | 0.520 [0.420, 0.610] |

The run is committed at
[`data/samples/runs/smoke_qwen_hellaswag/20261019T030343Z-42a02e07/`](data/samples/runs/smoke_qwen_hellaswag/20261019T030343Z-42a02e07/)
(`uv run llm-eval report --run-dir <that dir>` reprints the table).

## Why this exists

- Typical LLM-eval projects print one point estimate per model and call the higher one better; with
  300–500 examples a 2–3 point gap is usually inside the noise, so every number here has a CI and an `n`.
- Model-vs-model and prompt-vs-prompt claims are *paired* tests on the same examples, corrected for
  the number of comparisons, and the variance in scores is decomposed into model, prompt,
  interaction and item-difficulty shares.
- Every number is reproducible from `(config, seed, code sha, model revision, dataset revision)`;
  models and tasks are deliberately small so the whole grid runs on a laptop.

## Quick start

```bash
uv sync --all-extras --all-groups      # .venv with every extra and the dev tools
make smoke                             # Qwen2.5-0.5B × HellaSwag × zero_shot × 100 examples
uv run llm-eval report --run-dir data/processed/runs/smoke_qwen_hellaswag/<run_id>
make test                              # offline unit tests (no downloads)
```

`llm-eval run` prints the run directory and the summary table; see [`docs/api_spec.md`](docs/api_spec.md)
for every flag, the config JSON schema and the results schema.

## Experiment protocol

Named hypotheses, pre-registered here before the full-grid run (the commit sha is cited next to
them once that run starts):

- **H1** `few_shot_3 > zero_shot` on GSM8K for Qwen2.5-0.5B.
- **H2** `Llama-3.2-1B > Qwen2.5-0.5B` on HellaSwag `acc_norm` (zero-shot).
- **H3** `cot > zero_shot` on GSM8K for Llama-3.2-1B.

Decision rule:

1. Primary metric per task: HellaSwag `acc_norm` (log-likelihood cells; `acc` in the
   generation-mode prompt-effect run), GSM8K `exact_match`. HellaSwag `acc` is secondary.
2. One comparison family per task: model pairs within each prompt variant plus prompt pairs within
   each model (`k = V·C(M,2) + M·C(V,2)`); pairs that differ in both factors are exploratory.
3. Verdict "A > B" iff the Bonferroni-corrected McNemar exact p < 0.05 **and** the paired-bootstrap
   95% CI of `acc_A − acc_B` excludes 0. Holm is a sensitivity column; Wilcoxon is shown and
   labelled "≡ uncorrected asymptotic McNemar on 0/1 data".
4. Any pair where the tests or the two corrections disagree is flagged in the report.

### Test–retest

Three numbers, reported separately and never conflated:

- **Determinism** — fraction of (cell, example) pairs with an identical prediction in all 3 same-seed
  reruns, by provider × scoring mode: *measured on Day 3*.
- **Exemplar-resampling stability** — per-example agreement on `few_shot_3` cells across three
  exemplar seeds, with the per-cell max accuracy gap next to its CI half-width: *measured on Day 3*.
- **Conclusion stability** — fraction of significance-matrix verdicts unchanged across the 3
  reruns: *measured on Day 3*.

## Statistical methods

- **Bootstrap CI**: one seeded resample-index matrix (B=1000) per task, shared by every cell, gives
  percentile 95% intervals of the mean of `primary_correct` (also for `acc` and `acc_norm`).
- **Paired tests**: McNemar exact on the discordant pairs is the primary test, Wilcoxon signed-rank is
  reported next to it, and a paired bootstrap of `acc_A − acc_B` gives the effect size.
- **Multiplicity**: Bonferroni over the one-factor contrast family drives the ★, with Holm-adjusted
  p as a sensitivity column.
- **Variance**: cell accuracies are decomposed into model, prompt and interaction shares with
  bootstrap CIs, plus a blocked ANOVA with an item term for example difficulty.
- **Power**: the McNemar sample-size formula gives the minimum detectable effect at the fixed n, and
  differences below it are labelled "underpowered — no conclusion", never "no difference".

## Architecture

```mermaid
flowchart LR
    Y[YAML config] --> C[config.py<br/>pydantic EvalConfig]
    C --> G[expand_grid: GridCell list<br/>model × task × variant × resolved scoring]
    G --> R[runner.py]
    R --> T[data/*Task<br/>load + preprocess + nested subset]
    R --> P[features/prompts.py<br/>render variant]
    R --> M[models/*<br/>loglikelihood | generate]
    T --> P --> M --> S[evaluation/metrics.py<br/>score per example]
    S --> W[evaluation/results.py<br/>examples.parquet + run.json]
    W --> ST[evaluation/stats.py<br/>shared bootstrap · paired tests · decision rule]
    W --> V[evaluation/variance.py]
    ST --> RP[report.py tables + leaderboard sentence]
    V --> RP
    W --> D[serving/app.py Streamlit]
    RP --> ML[tracking.py MLflow]
```

```
src/llm_eval/
├── config.py  seeding.py  runner.py  cli.py
├── data/          # tasks: dataset loading + preprocessing (Task ABC, TASKS registry)
├── models/        # inference adapters (Model protocol, PROVIDERS registry, HFCausalLM)
├── features/      # prompt variants and rendering
├── evaluation/    # metrics, stats, results schema + IO, report tables
└── serving/       # Streamlit dashboard
```

## Adding a task / a model

- **Task**: add `src/llm_eval/data/<name>.py` with a `Task` subclass decorated `@register_task`
  (set `name`, `task_type`, `dataset_path`, `dataset_revision`, `default_split`; implement `context`);
  `import llm_eval.data` registers every module in the package, so `get_task("<name>")` works.
- **Model provider**: add a factory decorated `@register_provider("<name>", supports_loglikelihood=…)`
  in `src/llm_eval/models/registry.py` that imports its adapter (a class satisfying the `Model`
  protocol) inside its body and returns it; `load_model(cfg)` then serves `provider: <name>` configs.

## Running on a GPU box

```bash
uv sync --extra local
uv pip install --torch-backend=auto torch   # CUDA wheel matching the driver
source .venv/bin/activate                   # use the activated venv, not `uv run` (it would re-sync torch)
llm-eval run --config configs/eval/smoke_qwen_hellaswag.yaml
```

Provider options, prices and setup are in [`docs/cloud_compute_options.md`](docs/cloud_compute_options.md).

## Project layout

Follows the shared `code_standard.md` layout (design §4):

| standard | here |
|---|---|
| `configs/train/`, `configs/deploy/` | `configs/eval/` (one YAML per eval run; we do not train), `configs/deploy/` |
| `data/raw/`, `processed/`, `samples/` | same; runs land in `data/processed/runs/`, committed runs in `data/samples/runs/` |
| `docs/design.md`, `experiment_log.md`, `api_spec.md` | same ([design](docs/design.md), [log](docs/experiment_log.md), [API](docs/api_spec.md)) |
| `src/` package with `data/ models/ features/ evaluation/ serving/` | `src/llm_eval/` with the same sub-packages; `features/` holds prompt construction, `serving/` the dashboard |
| `tests/`, `scripts/`, `notebooks/` | same (`scripts/eval.sh` wraps `llm-eval run`) |
| `requirements.txt`, `setup.py` | `pyproject.toml` + `uv.lock`; `requirements*.txt` generated by `make requirements` |

## License

MIT — see [LICENSE](LICENSE).
