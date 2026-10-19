# Day 1 — MVP Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A config-driven runner that scores Qwen2.5-0.5B on 100 HellaSwag examples by log-likelihood, writes a schema-checked parquet + run.json with bootstrap CIs, and is covered by an offline test suite.

**Architecture:** Tasks produce preprocessed `Example`s; prompt variants render `(context, continuation)` requests; the HF adapter scores them under the lm-eval tokenization contract; `metrics` turns log-probs into per-example records; `results` writes them with an explicit pyarrow schema; `report` summarises with shared-resample bootstrap CIs; `runner`/`cli` orchestrate. Registries (`TASKS`, `PROVIDERS`) and the `seeding` module are the only global state.

**Tech Stack:** Python 3.11 (uv venv at `.venv`, already synced with every extra), pydantic 2.14, pandas 3.0.6, pyarrow 25, numpy 2.4, scipy 1.17, statsmodels 0.15, torch 2.14.1 (MPS), transformers 5.20.1, datasets 5.1.0, pytest 8, ruff.

**Spec:** `docs/design.md` (v0.2). Section numbers below (§6, §7.1, §8.2 …) refer to it. The spec is the authority; this plan is its argument.

## Global Constraints

- Work in `/Users/zy/Project/new_sys/projects/llm_eval_harness` on branch `feat/day1-mvp-runner`. Run everything through `uv run …` (the venv is `.venv`; never `pip install`; add no dependency).
- Package root `src/llm_eval/`; sub-packages `data/ models/ features/ evaluation/ serving/` (§4). Absolute imports (`from llm_eval.data.dataset import Example`).
- Import rule (§5): `torch`/`transformers` are imported only in `src/llm_eval/models/hf_model.py`; `datasets` only inside `Task` method bodies; `huggingface_hub` only inside `hf_model.py` functions. `config.py`, `evaluation/*.py`, `features/*.py`, `data/*.py` never import torch at module level.
- Every random draw goes through `llm_eval.seeding.child_rng(seed, *keys)`; never `random`, never `np.random.seed`, never Python `hash()` (§5).
- pydantic models use `model_config = ConfigDict(extra="forbid")` (§8.1). CLI is argparse (§9).
- `dtype: auto` = `torch.float32` on every device; the adapter always passes an explicit `dtype=` kwarg to `from_pretrained`, never `torch_dtype=`, never the string `"auto"` (§7.5).
- Tokenization and batching follow §7.1 verbatim: trailing context spaces move to the continuation; `whole_ids = tok(context + continuation, add_special_tokens=True)`; `n_ctx = len(tok(context, add_special_tokens=True))`; `cont_ids = whole_ids[n_ctx:]`; pad id = `pad_token_id` else `eos_token_id`; attention mask from lengths; logits sliced to continuation positions before `log_softmax`; results returned in request order.
- Results columns and arrow types are exactly §8.2 (27 columns, in that order); `run.json` fields are exactly §7.5 + §8.2 "summary".
- Subset rule (§7.5): `perm = child_rng(seed, "subset", task.name).permutation(n_rows)`; `idx = sorted(int(i) for i in perm[:n])`.
- `acc_norm` divides by the **character** length of the preprocessed choice, `max(len(choice), 1)` (§7.1).
- Tests are offline by default; anything that downloads is `@pytest.mark.slow`. Before every commit: `uv run ruff format src tests && uv run ruff check src tests && uv run pytest -q` must be clean (no warnings from our code).
- Commit messages: conventional prefix (`feat:`, `test:`, `docs:`, `chore:`), body optional, ending with the line `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. Stage only the task's own files.
- Git identity is repo-local (`createrW <yzhao470@usc.edu>`). Never push. Never touch `main`.

## Review Focus

1. `n_examples` larger than the split (`n_examples: 20000`): `Task.load` must raise `ValueError` naming n and the row count before any model loads — pinned in Task 2 (`test_load_raises_when_n_exceeds_rows`) and surfaced by the runner as a hard error, not a failed cell — Task 10 (`test_run_config_propagates_task_load_errors`).
2. A choice that preprocesses to the empty string (ending `"[substeps]"`): scoring must not divide by zero — Task 6 (`test_score_multiple_choice_empty_choice_uses_length_one`).
3. Context with trailing whitespace (`"hello world "` + `"foo"`): the trailing space must move into the continuation and the continuation must still be scored as exactly one token — Task 4 (`test_loglikelihood_moves_trailing_context_spaces`).
4. Mixed-length requests given in shuffled order: the adapter may sort for batching but must return results in request order and match unbatched values — Task 4 (`test_loglikelihood_batched_matches_unbatched_in_request_order`).
5. Running `llm-eval run` from a directory that is not the repo (no `pyproject.toml` above it): a relative `output_dir` resolves against the cwd and the run still succeeds — Task 11 (`test_cli_run_from_foreign_cwd_uses_cwd_as_project_root`); and a checkout with no `.git` (rented VM) still writes `run.json` with `git_sha: null` — Task 10 (`test_git_info_without_repo_returns_none`).

---

### Task 1: Seeding module and sub-package skeleton

**Files:**
- Create: `src/llm_eval/seeding.py`
- Create: `src/llm_eval/data/__init__.py`, `src/llm_eval/models/__init__.py`, `src/llm_eval/features/__init__.py`, `src/llm_eval/evaluation/__init__.py`, `src/llm_eval/serving/__init__.py` (all empty except `data/__init__.py`, see Step 3)
- Create: `tests/__init__.py` (empty), `tests/conftest.py` (empty for now)
- Test: `tests/test_seeding.py`

**Interfaces:**
- Produces: `stable_hash(s: str) -> int` — `int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8], "big")`; `child_rng(seed: int, *keys: str) -> numpy.random.Generator` — `np.random.default_rng(np.random.SeedSequence([seed, *(stable_hash(k) for k in keys)]))`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_seeding.py
import hashlib

import numpy as np

from llm_eval.seeding import child_rng, stable_hash


def test_stable_hash_is_sha256_prefix_big_endian():
    expected = int.from_bytes(hashlib.sha256(b"hellaswag").digest()[:8], "big")
    assert stable_hash("hellaswag") == expected
    assert stable_hash("hellaswag") == stable_hash("hellaswag")
    assert stable_hash("hellaswag") != stable_hash("gsm8k")


def test_child_rng_same_keys_same_stream():
    a = child_rng(42, "subset", "hellaswag").permutation(10)
    b = child_rng(42, "subset", "hellaswag").permutation(10)
    assert a.tolist() == b.tolist()


def test_child_rng_different_key_or_seed_different_stream():
    base = child_rng(42, "subset", "hellaswag").permutation(50).tolist()
    assert child_rng(42, "subset", "gsm8k").permutation(50).tolist() != base
    assert child_rng(43, "subset", "hellaswag").permutation(50).tolist() != base
    assert child_rng(42, "bootstrap", "hellaswag").permutation(50).tolist() != base


def test_child_rng_is_independent_of_prior_draws():
    first = child_rng(1, "a").integers(0, 1000, size=5).tolist()
    child_rng(1, "b").integers(0, 1000, size=1000)
    again = child_rng(1, "a").integers(0, 1000, size=5).tolist()
    assert first == again


def test_child_rng_returns_generator():
    assert isinstance(child_rng(0, "x"), np.random.Generator)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_seeding.py -v`
Expected: FAIL / ERROR with `ModuleNotFoundError: No module named 'llm_eval.seeding'`

- [ ] **Step 3: Implement `src/llm_eval/seeding.py` and the `__init__.py` files**

`seeding.py` holds the two functions above with docstrings that cite §5. `src/llm_eval/data/__init__.py` imports every sibling module so task registration runs on import:

```python
import importlib
import pkgutil

for _m in pkgutil.iter_modules(__path__):
    importlib.import_module(f"{__name__}.{_m.name}")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_seeding.py -v`
Expected: 5 passed

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format src tests && uv run ruff check src tests && uv run pytest -q
git add src/llm_eval tests/__init__.py tests/conftest.py tests/test_seeding.py
git commit -m "feat: seeding module (stable_hash, child_rng) and sub-package skeleton"
```

---

### Task 2: Task ABC, registry, and HellaSwag task

**Files:**
- Create: `src/llm_eval/data/dataset.py`, `src/llm_eval/data/hellaswag.py`
- Test: `tests/test_data_hellaswag.py`

**Interfaces:**
- Consumes: `child_rng` (Task 1).
- Produces (§6): `TaskType`, `Example`, `Task`, `TASKS`, `register_task`, `get_task`, `HellaSwagTask`. `Task.__init__(self, rows=None)` stores the seam; `Task._rows(split) -> Sequence[Mapping]` returns `rows(split)` if a seam was given, else `datasets.load_dataset(self.dataset_path, split=split, revision=self.dataset_revision)` (import inside the method). `load(split, n_examples, seed)` builds every row into an `Example` via the abstract `_example(row, row_idx, split) -> Example`, then applies the subset rule; raises `ValueError(f"n_examples={n} exceeds {n_rows} rows in {name}/{split}")` and `ValueError` on duplicate ids. `dataset_info()` returns `{"path", "revision", "split", "n_rows"}` (raises `RuntimeError` before `load`). `is_correct(prediction, example)` returns `prediction == example.target`. `context(example)` is abstract. `target_text`, `exemplars`, `extract_answer` raise `NotImplementedError("lands on Day 2")`. `get_task(name)` raises `KeyError(f"unknown task {name!r}; known: {sorted(TASKS)}")`.
- HellaSwag constants: `name="hellaswag"`, `task_type=MULTIPLE_CHOICE`, `dataset_path="Rowan/hellaswag"`, `dataset_revision="218ec52e09a7e7462a5400043bb9a69a41d06b76"`, `default_split="validation"`, `stop_sequences=("\n\n",)`. `inputs` keeps `ind, source_id, activity_label, ctx_a, ctx_b, split_type` and the preprocessed `context` string; `choices` = preprocessed endings; `target = str(int(row["label"]))`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_data_hellaswag.py
import pytest

import llm_eval.data  # noqa: F401  (registers tasks)
from llm_eval.data.dataset import TASKS, Example, TaskType, get_task
from llm_eval.data.hellaswag import HellaSwagTask, _preprocess

ROWS = [
    {"ind": 24, "activity_label": "Roof shingle removal", "ctx_a": "A man is sitting on a roof.",
     "ctx_b": "he", "ctx": "A man is sitting on a roof. he",
     "endings": ["is using wrap to wrap a pair of skis.", "is ripping level tiles off.",
                 "is holding a rubik's cube.", "starts pulling up roofing on a roof."],
     "source_id": "activitynet~v_-1IBHYS3L-Y", "split": "val", "split_type": "indomain", "label": "3"},
    {"ind": 180, "activity_label": "Preparing pasta", "ctx_a": "[header] How to cook pasta [title] Boil water.",
     "ctx_b": "add salt", "ctx": "", "endings": ["[substeps] Stir the pot.", "Drain [title] Serve.", "Wait  for it.", "Eat"],
     "source_id": "wikihow~299", "split": "val", "split_type": "zeroshot", "label": "0"},
    {"ind": 180, "activity_label": "Playing guitar", "ctx_a": "A woman holds a guitar.", "ctx_b": "she",
     "ctx": "", "endings": ["strums it.", "eats it.", "throws it.", "paints it."],
     "source_id": "activitynet~v_0bosp4-pyTM", "split": "val", "split_type": "indomain", "label": "1"},
    {"ind": 5, "activity_label": "Skiing", "ctx_a": "Snow falls.", "ctx_b": "a skier", "ctx": "",
     "endings": ["turns left.", "sleeps.", "swims.", "flies."],
     "source_id": "activitynet~v_x", "split": "val", "split_type": "indomain", "label": "0"},
    {"ind": 6, "activity_label": "Baking", "ctx_a": "Flour is on the table.", "ctx_b": "a cook", "ctx": "",
     "endings": ["kneads dough.", "drives away.", "sings.", "sleeps."],
     "source_id": "activitynet~v_y", "split": "val", "split_type": "indomain", "label": "0"},
    {"ind": 7, "activity_label": "Running", "ctx_a": "A track is empty.", "ctx_b": "a runner", "ctx": "",
     "endings": ["starts a lap.", "reads a book.", "cooks rice.", "paints."],
     "source_id": "activitynet~v_z", "split": "val", "split_type": "indomain", "label": "0"},
]


def make_task() -> HellaSwagTask:
    return HellaSwagTask(rows=lambda split: ROWS)


def test_preprocess_matches_lm_eval_rules():
    assert _preprocess("[header] How to cook pasta [title] Boil water.") == " How to cook pasta. Boil water."  # strip runs before the [header] removal (§7.1)
    assert _preprocess("[substeps] Stir the pot.") == " Stir the pot."
    assert _preprocess("Wait  for it.") == "Wait for it."
    assert _preprocess("  Eat  ") == "Eat"


def test_load_builds_examples_with_row_position_ids_and_preprocessed_fields():
    examples = make_task().load("validation", None, seed=0)
    assert [e.id for e in examples] == [f"hellaswag/validation/{i}" for i in range(6)]
    first = examples[0]
    assert isinstance(first, Example)
    assert first.inputs["context"] == "Roof shingle removal: A man is sitting on a roof. He"
    assert first.choices == ("is using wrap to wrap a pair of skis.", "is ripping level tiles off.",
                             "is holding a rubik's cube.", "starts pulling up roofing on a roof.")
    assert first.target == "3"
    assert first.inputs["ind"] == 24 and first.inputs["source_id"] == "activitynet~v_-1IBHYS3L-Y"
    pasta = examples[1]
    assert pasta.inputs["context"] == "Preparing pasta: How to cook pasta. Boil water. Add salt"
    assert pasta.choices == (" Stir the pot.", "Drain. Serve.", "Wait for it.", "Eat")


def test_duplicate_native_ind_does_not_collide():
    examples = make_task().load("validation", None, seed=0)
    assert examples[1].inputs["ind"] == examples[2].inputs["ind"] == 180
    assert examples[1].id != examples[2].id


def test_context_renders_inputs_context():
    task = make_task()
    ex = task.load("validation", None, seed=0)[0]
    assert task.context(ex) == "Roof shingle removal: A man is sitting on a roof. He"


def test_subsets_are_deterministic_nested_and_int_indexed():
    task = make_task()
    two = [e.id for e in task.load("validation", 2, seed=7)]
    two_again = [e.id for e in task.load("validation", 2, seed=7)]
    four = [e.id for e in task.load("validation", 4, seed=7)]
    assert two == two_again
    assert set(two) <= set(four)
    assert four == sorted(four, key=lambda s: int(s.rsplit("/", 1)[1]))


def test_subset_matches_permutation_rule():
    from llm_eval.seeding import child_rng
    task = make_task()
    perm = child_rng(7, "subset", "hellaswag").permutation(6)
    expected = sorted(int(i) for i in perm[:3])
    assert [int(e.id.rsplit("/", 1)[1]) for e in task.load("validation", 3, seed=7)] == expected


def test_load_raises_when_n_exceeds_rows():
    with pytest.raises(ValueError, match="n_examples=7 exceeds 6 rows"):
        make_task().load("validation", 7, seed=0)


def test_dataset_info_after_load():
    task = make_task()
    with pytest.raises(RuntimeError):
        task.dataset_info()
    task.load("validation", 2, seed=0)
    assert task.dataset_info() == {"path": "Rowan/hellaswag", "revision": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
                                   "split": "validation", "n_rows": 6}


def test_is_correct_compares_canonical_target():
    task = make_task()
    ex = task.load("validation", None, seed=0)[0]
    assert task.is_correct("3", ex) is True
    assert task.is_correct("1", ex) is False
    assert task.is_correct(None, ex) is False


def test_registry_exposes_hellaswag_and_rejects_unknown():
    assert TASKS["hellaswag"] is HellaSwagTask
    assert HellaSwagTask.task_type is TaskType.MULTIPLE_CHOICE
    assert isinstance(get_task("hellaswag"), HellaSwagTask)
    with pytest.raises(KeyError, match="unknown task 'nope'"):
        get_task("nope")


def test_day2_methods_raise_not_implemented():
    task = make_task()
    ex = task.load("validation", None, seed=0)[0]
    for call in (lambda: task.target_text(ex), lambda: task.exemplars(3, 0, ex.id), lambda: task.extract_answer("x")):
        with pytest.raises(NotImplementedError, match="Day 2"):
            call()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_data_hellaswag.py -v`
Expected: ERROR with `ModuleNotFoundError: No module named 'llm_eval.data.dataset'`

- [ ] **Step 3: Implement `dataset.py` and `hellaswag.py` per the Interfaces block**

`_preprocess` and the context/choice construction are the exact code of §7.1. `ctx_b` is capitalised with `str.capitalize()`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_data_hellaswag.py -v`
Expected: 11 passed

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format src tests && uv run ruff check src tests && uv run pytest -q
git add src/llm_eval/data tests/test_data_hellaswag.py
git commit -m "feat: Task ABC, task registry and HellaSwag task with lm-eval preprocessing and nested subsets"
```

---

### Task 3: Model protocol, request/result types, provider registry

**Files:**
- Create: `src/llm_eval/models/base_model.py`, `src/llm_eval/models/registry.py`
- Test: `tests/test_models_registry.py`

**Interfaces:**
- Produces (§6): `ScoringMode`, `LoglikelihoodRequest`, `LoglikelihoodResult(logprob, n_tokens, n_context_tokens, latency_ms)`, `GenerationRequest(prompt, max_new_tokens, stop=())`, `GenerationResult(text, n_prompt_tokens, n_generated_tokens, latency_ms)`, `Model` protocol (`name`, `supports_loglikelihood`, `loglikelihood`, `generate`, `info`). `ProviderSpec(factory, supports_loglikelihood)`, `PROVIDERS: dict[str, ProviderSpec]`, `register_provider(name, *, supports_loglikelihood)` decorator, `load_model(cfg)` → `PROVIDERS[cfg.provider].factory(cfg)`, raising `KeyError(f"unknown provider {p!r}; known: {sorted(PROVIDERS)}")`. The `"hf"` provider is registered in `registry.py` with a factory that does `from llm_eval.models.hf_model import HFCausalLM` inside its body and calls `HFCausalLM.from_pretrained(cfg.path, name=cfg.name, revision=cfg.revision, dtype=cfg.dtype, device=cfg.device, batch_size=cfg.batch_size)`. `registry.py` imports `ModelConfig` only under `TYPE_CHECKING`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_models_registry.py
import subprocess
import sys
from types import SimpleNamespace

import pytest

from llm_eval.models.base_model import (GenerationRequest, LoglikelihoodRequest, LoglikelihoodResult,
                                        ScoringMode)
from llm_eval.models.registry import PROVIDERS, ProviderSpec, load_model, register_provider


def test_scoring_mode_values():
    assert ScoringMode("loglikelihood") is ScoringMode.LOGLIKELIHOOD
    assert ScoringMode.GENERATION.value == "generation"


def test_dataclasses_are_frozen_with_defaults():
    req = GenerationRequest(prompt="p", max_new_tokens=8)
    assert req.stop == ()
    with pytest.raises(Exception):
        req.prompt = "q"  # type: ignore[misc]
    res = LoglikelihoodResult(logprob=-1.5, n_tokens=2, n_context_tokens=3, latency_ms=4.0)
    assert (res.logprob, res.n_tokens, res.n_context_tokens) == (-1.5, 2, 3)
    assert LoglikelihoodRequest("c", "x").continuation == "x"


def test_register_provider_and_load_model_dispatch():
    @register_provider("dummy_test_provider", supports_loglikelihood=False)
    def make(cfg):
        return SimpleNamespace(name=cfg.name, supports_loglikelihood=False)

    assert PROVIDERS["dummy_test_provider"] == ProviderSpec(factory=make, supports_loglikelihood=False)
    model = load_model(SimpleNamespace(provider="dummy_test_provider", name="m"))
    assert model.name == "m"


def test_load_model_unknown_provider():
    with pytest.raises(KeyError, match="unknown provider 'nope'"):
        load_model(SimpleNamespace(provider="nope", name="m"))


def test_hf_provider_is_registered_with_loglikelihood_support():
    assert PROVIDERS["hf"].supports_loglikelihood is True


def test_importing_registry_does_not_import_torch():
    code = "import sys, llm_eval.models.registry; print('torch' in sys.modules, 'transformers' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True).stdout
    assert out.strip() == "False False"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_models_registry.py -v`
Expected: ERROR with `ModuleNotFoundError: No module named 'llm_eval.models.base_model'`

- [ ] **Step 3: Implement `base_model.py` and `registry.py` per the Interfaces block**

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_models_registry.py -v`
Expected: 6 passed (the hf-factory body is exercised in Task 4)

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format src tests && uv run ruff check src tests && uv run pytest -q
git add src/llm_eval/models tests/test_models_registry.py
git commit -m "feat: Model protocol, request/result types and lazy provider registry"
```

---

### Task 4: HF causal-LM adapter with tiny-model tests

**Files:**
- Create: `src/llm_eval/models/hf_model.py`
- Modify: `tests/conftest.py` (add `tiny_tokenizer`, `tiny_llama`, `tiny_model_dir` fixtures)
- Test: `tests/test_models_hf.py`

**Interfaces:**
- Consumes: Task 3 types; `ProviderSpec`/`PROVIDERS["hf"]`.
- Produces (§6): `resolve_device(device) -> str`, `resolve_dtype(dtype, device) -> torch.dtype`, `HFCausalLM(model, tokenizer, *, name, device, batch_size=8, revision=None)`, `HFCausalLM.from_pretrained(path, *, name, revision=None, dtype="auto", device="auto", batch_size=8)`. `from_pretrained`: if `Path(path).is_dir()` load from it with `revision=None` recorded; else `local = huggingface_hub.snapshot_download(path, revision=revision, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"])` and `revision = Path(local).name`; load `AutoTokenizer.from_pretrained(local)` and `AutoModelForCausalLM.from_pretrained(local, dtype=resolve_dtype(dtype, device))`, `.to(device).eval()`; `torch.manual_seed` is the caller's job (runner). `info()` returns `{"revision": revision, "dtype": str(self.model.dtype).removeprefix("torch."), "device": self.device}`. `loglikelihood(requests)` implements §7.1 tokenization + batching contracts, returns results **in request order**, `latency_ms` per request = batch wall time / requests in the batch. `generate` raises `NotImplementedError("lands on Day 2")`.

- [ ] **Step 1: Add the fixtures to `tests/conftest.py`**

```python
# tests/conftest.py
import pytest


@pytest.fixture(scope="session")
def tiny_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    vocab = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "hello": 3, "world": 4, "foo": 5, "bar": 6, "baz": 7,
             "qux": 8, "a": 9, "b": 10, "c": 11, "d": 12, "e": 13, "f": 14, "g": 15}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(single="<bos> $A", special_tokens=[("<bos>", 2)])
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", bos_token="<bos>",
                                   eos_token="<pad>", pad_token="<pad>")   # pad == eos on purpose (§7.1)
    fast.padding_side = "right"
    return fast


@pytest.fixture(scope="session")
def tiny_llama(tiny_tokenizer):
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(vocab_size=16, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=128,
                      pad_token_id=0, bos_token_id=2, eos_token_id=0)
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg).eval()


@pytest.fixture
def tiny_model_dir(tmp_path, tiny_llama, tiny_tokenizer):
    import torch

    import copy

    d = tmp_path / "tiny"
    copy.deepcopy(tiny_llama).to(torch.bfloat16).save_pretrained(d)   # checkpoint config records bf16
    tiny_tokenizer.save_pretrained(d)
    return d
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_models_hf.py
import math

import pytest
import torch

from llm_eval.models.base_model import GenerationRequest, LoglikelihoodRequest
from llm_eval.models.hf_model import HFCausalLM, resolve_device, resolve_dtype


def manual_logprob(model, tokenizer, context, continuation):
    """Independent computation of §7.1: sum of log-probs of the continuation tokens."""
    n_sp = len(context) - len(context.rstrip())
    if n_sp:
        continuation, context = context[-n_sp:] + continuation, context[:-n_sp]
    whole = tokenizer(context + continuation, add_special_tokens=True).input_ids
    n_ctx = len(tokenizer(context, add_special_tokens=True).input_ids)
    ids = torch.tensor([whole])
    with torch.no_grad():
        logits = model(ids).logits[0]
    logp = torch.log_softmax(logits.float(), dim=-1)
    return sum(logp[pos - 1, whole[pos]].item() for pos in range(n_ctx, len(whole))), len(whole) - n_ctx, n_ctx


@pytest.fixture
def adapter(tiny_llama, tiny_tokenizer):
    return HFCausalLM(tiny_llama, tiny_tokenizer, name="tiny", device="cpu", batch_size=2)


def test_resolve_dtype_auto_is_float32_on_every_device():
    for dev in ("cpu", "mps", "cuda"):
        assert resolve_dtype("auto", dev) is torch.float32
    assert resolve_dtype("bfloat16", "cpu") is torch.bfloat16
    assert resolve_dtype("float16", "cpu") is torch.float16


def test_resolve_device_auto_picks_an_available_backend():
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in {"cuda", "mps", "cpu"}


def test_loglikelihood_matches_manual_computation(adapter, tiny_llama, tiny_tokenizer):
    req = LoglikelihoodRequest("hello world", " foo bar")
    [res] = adapter.loglikelihood([req])
    expected, n_tok, n_ctx = manual_logprob(tiny_llama, tiny_tokenizer, req.context, req.continuation)
    assert math.isclose(res.logprob, expected, abs_tol=1e-5)
    assert res.n_tokens == n_tok == 2
    assert res.n_context_tokens == n_ctx == 3   # <bos> hello world
    assert res.latency_ms > 0


def test_bos_appears_exactly_once_at_position_zero(adapter, tiny_tokenizer):
    ids = adapter._encode_pair("hello", " world")   # (whole_ids, n_ctx) helper, part of the contract
    whole, n_ctx = ids
    assert whole[0] == tiny_tokenizer.bos_token_id
    assert whole.count(tiny_tokenizer.bos_token_id) == 1
    assert n_ctx == 2 and whole[n_ctx:] == [4]


def test_loglikelihood_moves_trailing_context_spaces(adapter, tiny_llama, tiny_tokenizer):
    [res] = adapter.loglikelihood([LoglikelihoodRequest("hello world ", "foo")])
    expected, n_tok, n_ctx = manual_logprob(tiny_llama, tiny_tokenizer, "hello world ", "foo")
    assert res.n_tokens == 1 and res.n_context_tokens == 3
    assert math.isclose(res.logprob, expected, abs_tol=1e-5)


def test_loglikelihood_batched_matches_unbatched_in_request_order(adapter, tiny_llama, tiny_tokenizer):
    reqs = [LoglikelihoodRequest("hello world foo bar", " baz"),
            LoglikelihoodRequest("hello", " world foo bar baz qux"),
            LoglikelihoodRequest("a b c", " d"),
            LoglikelihoodRequest("hello world", " foo bar baz"),
            LoglikelihoodRequest("qux", " a b c d e f g")]
    batched = adapter.loglikelihood(reqs)
    for req, res in zip(reqs, batched, strict=True):
        [single] = adapter.loglikelihood([req])
        expected, n_tok, _ = manual_logprob(tiny_llama, tiny_tokenizer, req.context, req.continuation)
        assert res.n_tokens == n_tok
        assert math.isclose(res.logprob, expected, abs_tol=1e-4)
        assert math.isclose(res.logprob, single.logprob, abs_tol=1e-4)


def test_empty_continuation_raises(adapter):
    with pytest.raises(ValueError):
        adapter.loglikelihood([LoglikelihoodRequest("hello", "")])


def test_from_pretrained_local_dir_loads_float32_under_auto_and_reports_info(tiny_model_dir):
    m = HFCausalLM.from_pretrained(str(tiny_model_dir), name="tiny", dtype="auto", device="cpu", batch_size=4)
    assert m.model.dtype is torch.float32
    assert m.info() == {"revision": None, "dtype": "float32", "device": "cpu"}
    assert m.supports_loglikelihood is True
    [res] = m.loglikelihood([LoglikelihoodRequest("hello", " world")])
    assert res.n_tokens == 1


def test_from_pretrained_hub_path_uses_snapshot_download(monkeypatch, tiny_model_dir):
    import llm_eval.models.hf_model as hf

    calls = {}

    def fake_snapshot_download(repo_id, revision=None, allow_patterns=None):
        calls.update(repo_id=repo_id, revision=revision, allow_patterns=allow_patterns)
        fake = tiny_model_dir.parent / "snapshots" / "0123456789abcdef0123456789abcdef01234567"
        fake.parent.mkdir(parents=True, exist_ok=True)
        if not fake.exists():
            fake.symlink_to(tiny_model_dir, target_is_directory=True)
        return str(fake)

    monkeypatch.setattr(hf, "snapshot_download", fake_snapshot_download)
    m = HFCausalLM.from_pretrained("org/model", name="x", device="cpu")
    assert calls["repo_id"] == "org/model" and "*.safetensors" in calls["allow_patterns"]
    assert m.info()["revision"] == "0123456789abcdef0123456789abcdef01234567"


def test_generate_lands_on_day_2(adapter):
    with pytest.raises(NotImplementedError, match="Day 2"):
        adapter.generate([GenerationRequest("hello", 4)])


def test_registry_hf_factory_forwards_config(monkeypatch):
    from llm_eval.models import hf_model
    from llm_eval.models.registry import load_model

    seen = {}
    monkeypatch.setattr(hf_model.HFCausalLM, "from_pretrained",
                        classmethod(lambda cls, path, **kw: seen.update(path=path, **kw) or "MODEL"))
    from types import SimpleNamespace
    cfg = SimpleNamespace(provider="hf", name="q", path="Qwen/Qwen2.5-0.5B", revision=None, dtype="auto",
                          device="cpu", batch_size=3)
    assert load_model(cfg) == "MODEL"
    assert seen == {"path": "Qwen/Qwen2.5-0.5B", "name": "q", "revision": None, "dtype": "auto",
                    "device": "cpu", "batch_size": 3}
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_models_hf.py -v`
Expected: ERROR with `ModuleNotFoundError: No module named 'llm_eval.models.hf_model'`

- [ ] **Step 4: Implement `hf_model.py`**

Module-level `from huggingface_hub import snapshot_download` is allowed here (test monkeypatches `hf.snapshot_download`). Expose `_encode_pair(context, continuation) -> tuple[list[int], int]` implementing the §7.1 contract (used by `loglikelihood`). Batching: sort requests by total length, group by `batch_size`, right-pad with the pad id, build `attention_mask` from lengths, run `model(input_ids, attention_mask)`, and for each row gather `log_softmax(logits[row, pos-1])[token]` for `pos` in the continuation range only (slice first, then `log_softmax` on `[n_cont, V]`). Keep `torch.inference_mode()`. Reorder to request order before returning.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_models_hf.py -v`
Expected: 11 passed, no warnings from `llm_eval`

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff format src tests && uv run ruff check src tests && uv run pytest -q
git add src/llm_eval/models/hf_model.py tests/conftest.py tests/test_models_hf.py
git commit -m "feat: HFCausalLM adapter with lm-eval tokenization contract, length-mask batching and float32 default"
```

---

### Task 5: Prompt variants and cell support table

**Files:**
- Create: `src/llm_eval/features/prompts.py`
- Test: `tests/test_prompts.py`

**Interfaces:**
- Consumes: `Task`, `Example`, `TaskType` (Task 2); `ScoringMode` (Task 3).
- Produces (§6): `PromptVariant`, `SUPPORTED = frozenset({(TaskType.MULTIPLE_CHOICE, PromptVariant.ZERO_SHOT, ScoringMode.LOGLIKELIHOOD)})`, `LANDS_ON_DAY = {(MC, FEW_SHOT_3, LL): 2, (MC, ZERO_SHOT, GEN): 3, (MC, FEW_SHOT_3, GEN): 3, (MC, COT, GEN): 3, (GENERATION, ZERO_SHOT, GEN): 2, (GENERATION, FEW_SHOT_3, GEN): 2, (GENERATION, COT, GEN): 2}`, `MAX_NEW_TOKENS = {ZERO_SHOT: 256, FEW_SHOT_3: 256, COT: 512}`, `class UnsupportedCellError(ValueError)`, `check_supported(task_type, variant, scoring, *, cell: str) -> None` (raises `UnsupportedCellError(f"{cell}: {task_type.value} × {variant.value} × {scoring.value} lands on Day {d}")` or `… is not a valid combination`), `MCPrompt(context, continuations)`, `render_mc_loglikelihood(task, example, variant, exemplars=())` → `MCPrompt(task.context(example), tuple(" " + c for c in example.choices))` for `ZERO_SHOT`; other variants call `check_supported` with `scoring=LOGLIKELIHOOD` (so they raise). `render_mc_generation` / `render_generation` raise `NotImplementedError("lands on Day 2")`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_prompts.py
import pytest

from llm_eval.data.dataset import TaskType
from llm_eval.data.hellaswag import HellaSwagTask
from llm_eval.features.prompts import (MAX_NEW_TOKENS, SUPPORTED, MCPrompt, PromptVariant, UnsupportedCellError,
                                       check_supported, render_mc_loglikelihood)
from llm_eval.models.base_model import ScoringMode
from tests.test_data_hellaswag import ROWS


@pytest.fixture
def example():
    return HellaSwagTask(rows=lambda s: ROWS).load("validation", None, seed=0)[1]


def test_zero_shot_loglikelihood_prompt_is_context_plus_space_prefixed_choices(example):
    p = render_mc_loglikelihood(HellaSwagTask(rows=lambda s: ROWS), example, PromptVariant.ZERO_SHOT)
    assert p == MCPrompt(context="Preparing pasta: How to cook pasta. Boil water. Add salt",
                         continuations=("  Stir the pot.", " Drain. Serve.", " Wait for it.", " Eat"))


def test_supported_cells_day1_and_max_new_tokens():
    assert SUPPORTED == frozenset({(TaskType.MULTIPLE_CHOICE, PromptVariant.ZERO_SHOT, ScoringMode.LOGLIKELIHOOD)})
    assert MAX_NEW_TOKENS == {PromptVariant.ZERO_SHOT: 256, PromptVariant.FEW_SHOT_3: 256, PromptVariant.COT: 512}


def test_check_supported_names_cell_and_day():
    with pytest.raises(UnsupportedCellError, match=r"q×hellaswag×few_shot_3: multiple_choice × few_shot_3 × loglikelihood lands on Day 2"):
        check_supported(TaskType.MULTIPLE_CHOICE, PromptVariant.FEW_SHOT_3, ScoringMode.LOGLIKELIHOOD,
                        cell="q×hellaswag×few_shot_3")
    with pytest.raises(UnsupportedCellError, match="lands on Day 3"):
        check_supported(TaskType.MULTIPLE_CHOICE, PromptVariant.COT, ScoringMode.GENERATION, cell="c")
    with pytest.raises(UnsupportedCellError, match="is not a valid combination"):
        check_supported(TaskType.MULTIPLE_CHOICE, PromptVariant.COT, ScoringMode.LOGLIKELIHOOD, cell="c")
    with pytest.raises(UnsupportedCellError, match="is not a valid combination"):
        check_supported(TaskType.GENERATION, PromptVariant.ZERO_SHOT, ScoringMode.LOGLIKELIHOOD, cell="c")
    check_supported(TaskType.MULTIPLE_CHOICE, PromptVariant.ZERO_SHOT, ScoringMode.LOGLIKELIHOOD, cell="ok")


def test_render_few_shot_raises_unsupported(example):
    with pytest.raises(UnsupportedCellError, match="Day 2"):
        render_mc_loglikelihood(HellaSwagTask(rows=lambda s: ROWS), example, PromptVariant.FEW_SHOT_3)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: ERROR with `ModuleNotFoundError: No module named 'llm_eval.features.prompts'`

- [ ] **Step 3: Implement `prompts.py` per the Interfaces block**

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_prompts.py -v`
Expected: 4 passed

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format src tests && uv run ruff check src tests && uv run pytest -q
git add src/llm_eval/features tests/test_prompts.py
git commit -m "feat: prompt variants, Day-1 cell support table and zero-shot MC rendering"
```

---

### Task 6: Metrics and bootstrap statistics

**Files:**
- Create: `src/llm_eval/evaluation/metrics.py`, `src/llm_eval/evaluation/stats.py`
- Test: `tests/test_metrics.py`, `tests/test_stats.py`

**Interfaces:**
- Consumes: `LoglikelihoodResult`, `ScoringMode` (Task 3); `TaskType` (Task 2); `child_rng` (Task 1).
- Produces (§6): `MCScore(pred, pred_norm, logprobs, logprobs_norm)`, `score_multiple_choice(results, choices)` with `logprobs_norm_i = logprob_i / max(len(choices[i]), 1)` and `argmax` taking the first maximum on ties; `accuracy(correct)` (`ValueError` on empty); `primary_metric(task_type, scoring) -> str`. `resample_indices(n, n_resamples, seed, key)` → `child_rng(seed, "bootstrap", key).integers(0, n, size=(n_resamples, n))`; `bootstrap_ci(values, idx, alpha=0.05)` → `(lo, hi)` = `np.percentile(means, [100*alpha/2, 100*(1-alpha/2)])` of `values[idx].mean(axis=1)`, as Python floats.

- [ ] **Step 1: Write the failing tests**

```python
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
```

```python
# tests/test_stats.py
import numpy as np
from statsmodels.stats.proportion import proportion_confint

from llm_eval.evaluation.stats import bootstrap_ci, resample_indices


def test_resample_indices_shape_range_and_determinism():
    idx = resample_indices(7, 50, seed=3, key="hellaswag")
    assert idx.shape == (50, 7) and idx.min() >= 0 and idx.max() < 7
    assert np.array_equal(idx, resample_indices(7, 50, seed=3, key="hellaswag"))
    assert not np.array_equal(idx, resample_indices(7, 50, seed=3, key="gsm8k"))
    assert not np.array_equal(idx, resample_indices(7, 50, seed=4, key="hellaswag"))


def test_bootstrap_ci_degenerate_vectors():
    idx = resample_indices(10, 200, seed=0, key="t")
    assert bootstrap_ci(np.ones(10), idx) == (1.0, 1.0)
    assert bootstrap_ci(np.zeros(10), idx) == (0.0, 0.0)


def test_bootstrap_ci_contains_point_estimate_and_matches_wilson_within_0_02():
    for n, k in ((100, 56), (300, 42)):
        values = np.array([1.0] * k + [0.0] * (n - k))
        lo, hi = bootstrap_ci(values, resample_indices(n, 1000, seed=42, key="hellaswag"))
        wlo, whi = proportion_confint(k, n, alpha=0.05, method="wilson")
        assert lo <= k / n <= hi
        assert abs(lo - wlo) < 0.02 and abs(hi - whi) < 0.02
        assert isinstance(lo, float) and isinstance(hi, float)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_metrics.py tests/test_stats.py -v`
Expected: ERROR with `ModuleNotFoundError` for both modules

- [ ] **Step 3: Implement `metrics.py` and `stats.py` per the Interfaces block**

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_metrics.py tests/test_stats.py -v`
Expected: 8 passed

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format src tests && uv run ruff check src tests && uv run pytest -q
git add src/llm_eval/evaluation/metrics.py src/llm_eval/evaluation/stats.py tests/test_metrics.py tests/test_stats.py
git commit -m "feat: MC scoring with lm-eval acc_norm, primary metric mapping and shared-resample bootstrap CI"
```

---

### Task 7: Config schema, grid expansion, config hash

**Files:**
- Create: `src/llm_eval/config.py`, `configs/eval/smoke_qwen_hellaswag.yaml`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: `TASKS` (Task 2, import `llm_eval.data` for side effects), `PROVIDERS`, `ScoringMode` (Task 3), `PromptVariant`, `check_supported` (Task 5).
- Produces (§6/§8.1): `ModelConfig`, `TaskConfig`, `EvalConfig`, `GridCell`, `load_config(path, *, overrides=None)`, `config_hash(cfg)`, `expand_grid(cfg)`. Validators: `n_examples ≥ 1`, `batch_size ≥ 1`, `bootstrap_resamples ≥ 1`, unique model names, `provider in PROVIDERS`, `task.name in TASKS`, non-empty unique `prompt_variants`. `expand_grid` resolves scoring per cell: `auto` → `LOGLIKELIHOOD` iff `PROVIDERS[p].supports_loglikelihood and task_type is MULTIPLE_CHOICE and variant is not COT`, else `GENERATION`; explicit value used as-is; then `check_supported(task_type, variant, scoring, cell=f"{model.name}×{task.name}×{variant.value}")`. `overrides` keys: `n_examples`, `seed`, `output_dir` (applied to the raw dict before validation; `None` values ignored).
- The smoke config is exactly §8.1's YAML with `run_name: smoke_qwen_hellaswag`, `n_examples: 100`, Qwen model, HellaSwag task, `prompt_variants: [zero_shot]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_config.py
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from llm_eval.config import EvalConfig, GridCell, config_hash, expand_grid, load_config
from llm_eval.features.prompts import PromptVariant, UnsupportedCellError
from llm_eval.models.base_model import ScoringMode

SMOKE = Path(__file__).resolve().parents[1] / "configs" / "eval" / "smoke_qwen_hellaswag.yaml"

BASE = {"run_name": "t", "models": [{"name": "q", "provider": "hf", "path": "Qwen/Qwen2.5-0.5B"}],
        "tasks": [{"name": "hellaswag"}], "prompt_variants": ["zero_shot"]}


def write(tmp_path, data):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


def test_smoke_config_loads_with_documented_values():
    cfg = load_config(SMOKE)
    assert cfg.run_name == "smoke_qwen_hellaswag" and cfg.seed == 42 and cfg.n_examples == 100
    assert cfg.exemplar_seed is None and cfg.scoring == "auto" and cfg.bootstrap_resamples == 1000
    assert cfg.output_dir == "data/processed/runs"
    assert [m.name for m in cfg.models] == ["qwen2.5-0.5b"]
    assert cfg.models[0].path == "Qwen/Qwen2.5-0.5B" and cfg.models[0].dtype == "auto"
    assert cfg.models[0].device == "auto" and cfg.models[0].batch_size == 8 and cfg.models[0].max_new_tokens is None
    assert cfg.tasks[0].name == "hellaswag" and cfg.tasks[0].split == "validation"
    assert cfg.prompt_variants == [PromptVariant.ZERO_SHOT]


@pytest.mark.parametrize("mutation, message", [
    (lambda d: d.update(extra_field=1), "extra_field"),
    (lambda d: d.update(n_examples=0), "n_examples"),
    (lambda d: d["models"][0].update(provider="nope"), "provider"),
    (lambda d: d["tasks"][0].update(name="nope"), "task"),
    (lambda d: d["models"].append(dict(d["models"][0])), "unique"),
    (lambda d: d.update(prompt_variants=[]), "prompt_variants"),
    (lambda d: d.update(prompt_variants=["zero_shot", "zero_shot"]), "unique"),
    (lambda d: d["models"][0].update(dtype="int8"), "dtype"),
])
def test_invalid_configs_are_rejected(tmp_path, mutation, message):
    data = yaml.safe_load(yaml.safe_dump(BASE))
    mutation(data)
    with pytest.raises(ValidationError, match=message):
        load_config(write(tmp_path, data))


def test_overrides_apply_before_validation(tmp_path):
    cfg = load_config(write(tmp_path, BASE), overrides={"n_examples": 5, "seed": 9, "output_dir": None})
    assert cfg.n_examples == 5 and cfg.seed == 9 and cfg.output_dir == "data/processed/runs"


def test_expand_grid_resolves_auto_to_loglikelihood_for_hf_mc_zero_shot():
    cfg = EvalConfig(**BASE)
    [cell] = expand_grid(cfg)
    assert isinstance(cell, GridCell)
    assert cell.model.name == "q" and cell.task.name == "hellaswag"
    assert cell.variant is PromptVariant.ZERO_SHOT and cell.scoring is ScoringMode.LOGLIKELIHOOD


def test_expand_grid_order_is_models_outer_then_tasks():
    data = yaml.safe_load(yaml.safe_dump(BASE))
    data["models"].append({"name": "r", "provider": "hf", "path": "x/y"})
    cells = expand_grid(EvalConfig(**data))
    assert [(c.model.name, c.task.name, c.variant.value) for c in cells] == [
        ("q", "hellaswag", "zero_shot"), ("r", "hellaswag", "zero_shot")]


def test_expand_grid_rejects_few_shot_on_day_1():
    data = yaml.safe_load(yaml.safe_dump(BASE))
    data["prompt_variants"] = ["zero_shot", "few_shot_3"]
    with pytest.raises(UnsupportedCellError, match=r"q×hellaswag×few_shot_3: .* lands on Day 2"):
        expand_grid(EvalConfig(**data))


def test_expand_grid_rejects_forced_generation_on_day_1():
    data = dict(BASE, scoring="generation")
    with pytest.raises(UnsupportedCellError, match=r"q×hellaswag×zero_shot: .* lands on Day 3"):
        expand_grid(EvalConfig(**data))


def test_config_hash_ignores_output_dir_but_not_seed(tmp_path):
    a = EvalConfig(**BASE)
    b = EvalConfig(**dict(BASE, output_dir="/elsewhere"))
    c = EvalConfig(**dict(BASE, seed=1))
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash(c)
    assert len(config_hash(a)) == 12 and int(config_hash(a), 16) >= 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_config.py -v`
Expected: ERROR with `ModuleNotFoundError: No module named 'llm_eval.config'`

- [ ] **Step 3: Implement `config.py` and write `configs/eval/smoke_qwen_hellaswag.yaml`**

`TaskConfig.split` defaults to `None` in the model but `load_config`/validation fills it from `TASKS[name].default_split` (a `model_validator(mode="after")` on `EvalConfig`), so the loaded config shows `"validation"`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_config.py -v`
Expected: 16 passed

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format src tests && uv run ruff check src tests && uv run pytest -q
git add src/llm_eval/config.py configs/eval/smoke_qwen_hellaswag.yaml tests/test_config.py
git commit -m "feat: pydantic config schema, per-cell scoring resolution and location-independent config hash"
```

---

### Task 8: Results schema and run-directory IO

**Files:**
- Create: `src/llm_eval/evaluation/results.py`
- Test: `tests/test_results.py`

**Interfaces:**
- Produces (§6/§8.2): `RESULTS_SCHEMA` (27 fields, names and arrow types exactly as the §8.2 table, `nullable=False` for `run_id, model, task, prompt_variant, scoring, metric, seed, example_id, example_index, prompt, target, latency_ms, status`), `ExampleRecord` frozen dataclass with the same 27 fields in the same order (list fields typed `tuple[float, ...] | None` / `tuple[str, ...] | None`), `records_to_frame(records)` → `pa.Table.from_pylist([asdict(r) …], schema=RESULTS_SCHEMA).to_pandas(types_mapper=pd.ArrowDtype)`, `write_run(run_dir, frame, metadata)` (creates the dir; asserts uniqueness of `(model, task, prompt_variant, example_id)` raising `ValueError("duplicate …")`; writes `examples.parquet` with `pa.Table.from_pandas(frame, schema=RESULTS_SCHEMA, preserve_index=False)` and `compression="zstd"`; writes `run.json` with `json.dump(metadata, indent=2, sort_keys=True, default=str)`), `read_run(run_dir)` → `(pq.read_table(...).to_pandas(types_mapper=pd.ArrowDtype), json.load(...))`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_results.py
import dataclasses
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from llm_eval.evaluation.results import RESULTS_SCHEMA, ExampleRecord, read_run, records_to_frame, write_run

COLUMNS = ["run_id", "model", "task", "prompt_variant", "scoring", "metric", "seed", "exemplar_seed", "example_id",
           "example_index", "prompt", "target", "prediction", "correct", "prediction_norm", "correct_norm",
           "primary_correct", "choice_logprobs", "choice_logprobs_norm", "generation", "parse_ok", "exemplar_ids",
           "n_prompt_tokens", "n_generated_tokens", "latency_ms", "status", "error"]


def record(i, **over):
    base = dict(run_id="r", model="m", task="hellaswag", prompt_variant="zero_shot", scoring="loglikelihood",
                metric="acc_norm", seed=42, exemplar_seed=None, example_id=f"hellaswag/validation/{i}",
                example_index=i, prompt="ctx", target="1", prediction="1", correct=True, prediction_norm="0",
                correct_norm=False, primary_correct=False, choice_logprobs=(-1.0, -2.0), choice_logprobs_norm=(-0.5, -1.0),
                generation=None, parse_ok=None, exemplar_ids=None, n_prompt_tokens=3, n_generated_tokens=None,
                latency_ms=1.5, status="ok", error=None)
    base.update(over)
    return ExampleRecord(**base)


def test_schema_has_exactly_the_design_columns_in_order():
    assert RESULTS_SCHEMA.names == COLUMNS
    assert [f.name for f in dataclasses.fields(ExampleRecord)] == COLUMNS
    assert RESULTS_SCHEMA.field("choice_logprobs").type == pa.list_(pa.float64())
    assert RESULTS_SCHEMA.field("exemplar_ids").type == pa.list_(pa.string())
    assert RESULTS_SCHEMA.field("correct").type == pa.bool_() and RESULTS_SCHEMA.field("seed").type == pa.int64()
    assert RESULTS_SCHEMA.field("latency_ms").type == pa.float64() and RESULTS_SCHEMA.field("status").nullable is False
    assert RESULTS_SCHEMA.field("error").nullable is True


def test_round_trip_preserves_values_schema_and_metadata(tmp_path):
    frame = records_to_frame([record(0), record(1, correct=False, choice_logprobs=None, choice_logprobs_norm=None)])
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_results.py -v`
Expected: ERROR with `ModuleNotFoundError: No module named 'llm_eval.evaluation.results'`

- [ ] **Step 3: Implement `results.py` per the Interfaces block**

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_results.py -v`
Expected: 4 passed

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format src tests && uv run ruff check src tests && uv run pytest -q
git add src/llm_eval/evaluation/results.py tests/test_results.py
git commit -m "feat: explicit pyarrow results schema, ExampleRecord and run-directory IO"
```

---

### Task 9: Per-cell summary with CIs and markdown report

**Files:**
- Create: `src/llm_eval/evaluation/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `resample_indices`, `bootstrap_ci` (Task 6); frames from `records_to_frame` (Task 8).
- Produces (§6/§8.2): `summarize(frame, *, seed, n_resamples=1000) -> pd.DataFrame` with columns `model, task, prompt_variant, scoring, metric, n, n_ok, n_error, parse_failure_rate, acc, acc_ci_low, acc_ci_high, acc_norm, acc_norm_ci_low, acc_norm_ci_high, primary_acc, primary_ci_low, primary_ci_high, ci_method, n_resamples, ci_seed` (one row per cell, rows sorted by `(task, model, prompt_variant)`; a cell's rows are sorted by `example_id` and the task's resample matrix `resample_indices(n_cell, n_resamples, seed, task)` is shared across its cells; `acc_norm*` are `None` when `correct_norm` is all null; `parse_failure_rate` is `None` when `parse_ok` is all null; `n_error` counts `status == "error"`; CIs use `status == "ok"` rows). `to_markdown(table, floatfmt=".3f") -> str` (pipe table, header + `|---|` separator, floats formatted, `None`/NaN as empty). `format_summary(summary) -> str` → `to_markdown` of a view with columns `model, task, prompt_variant, metric, n, acc, acc_norm` where `acc` is rendered as `"0.750 [0.600, 0.900]"`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_report.py
import pandas as pd

from llm_eval.evaluation.report import format_summary, summarize, to_markdown
from tests.test_results import record
from llm_eval.evaluation.results import records_to_frame


def two_cells():
    rows = []
    for i in range(10):
        rows.append(record(i, model="m1", correct=i < 7, correct_norm=i < 6, primary_correct=i < 6))
        rows.append(record(i, model="m2", correct=i < 4, correct_norm=i < 5, primary_correct=i < 5))
    return records_to_frame(rows)


def test_summarize_one_row_per_cell_with_point_estimates_and_cis():
    s = summarize(two_cells(), seed=42, n_resamples=200)
    assert list(s.columns) == ["model", "task", "prompt_variant", "scoring", "metric", "n", "n_ok", "n_error",
                               "parse_failure_rate", "acc", "acc_ci_low", "acc_ci_high", "acc_norm", "acc_norm_ci_low",
                               "acc_norm_ci_high", "primary_acc", "primary_ci_low", "primary_ci_high", "ci_method",
                               "n_resamples", "ci_seed"]
    assert s["model"].tolist() == ["m1", "m2"]
    m1 = s.iloc[0]
    assert (m1["n"], m1["n_ok"], m1["n_error"]) == (10, 10, 0)
    assert m1["acc"] == 0.7 and m1["acc_norm"] == 0.6 and m1["primary_acc"] == 0.6
    assert m1["acc_ci_low"] <= 0.7 <= m1["acc_ci_high"] and m1["primary_ci_low"] <= 0.6 <= m1["primary_ci_high"]
    assert m1["ci_method"] == "percentile_bootstrap" and m1["n_resamples"] == 200 and m1["ci_seed"] == 42
    assert pd.isna(m1["parse_failure_rate"])


def test_summarize_is_deterministic_and_shares_resamples_across_cells():
    a = summarize(two_cells(), seed=42, n_resamples=200)
    b = summarize(two_cells(), seed=42, n_resamples=200)
    pd.testing.assert_frame_equal(a, b)
    c = summarize(two_cells(), seed=43, n_resamples=200)
    assert not a[["acc_ci_low", "acc_ci_high"]].equals(c[["acc_ci_low", "acc_ci_high"]])


def test_to_markdown_exact():
    t = pd.DataFrame({"model": ["m1"], "acc": [0.75], "note": [None]})
    assert to_markdown(t) == "| model | acc | note |\n|---|---|---|\n| m1 | 0.750 |  |"


def test_format_summary_renders_interval_strings():
    out = format_summary(summarize(two_cells(), seed=42, n_resamples=200))
    assert out.startswith("| model | task | prompt_variant | metric | n | acc | acc_norm |")
    assert "| m1 | hellaswag | zero_shot | acc_norm | 10 | 0.700 [" in out
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_report.py -v`
Expected: ERROR with `ModuleNotFoundError: No module named 'llm_eval.evaluation.report'`

- [ ] **Step 3: Implement `report.py` per the Interfaces block**

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_report.py -v`
Expected: 4 passed

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format src tests && uv run ruff check src tests && uv run pytest -q
git add src/llm_eval/evaluation/report.py tests/test_report.py
git commit -m "feat: per-cell summary with shared-resample bootstrap CIs and markdown tables"
```

---

### Task 10: Runner

**Files:**
- Create: `src/llm_eval/runner.py`
- Modify: `tests/conftest.py` (add `FakeModel`, the `fake` provider, the `hellaswag_offline` task, `RUNNER_ROWS`)
- Test: `tests/test_runner.py`

**Interfaces:**
- Consumes: everything above.
- Produces (§6/§9): `RunResult(run_dir, summary, failed_cells)`, `evaluate_cell(model, task, examples, variant, scoring, *, run_id, seed, exemplar_seed=None) -> list[ExampleRecord]`, `run_config(cfg, *, project_root=None, now=None) -> RunResult`, `git_info(root) -> tuple[str | None, bool | None]`, `library_versions() -> dict[str, str]`, `find_project_root(start: Path) -> Path`.
- `evaluate_cell` (Day 1, LOGLIKELIHOOD only — other modes raise `UnsupportedCellError` via `check_supported`): renders `MCPrompt` per example, issues ONE `model.loglikelihood` call for all `K×n` requests, scores with `score_multiple_choice`, fills records: `prediction=str(pred)`, `correct=task.is_correct(prediction, ex)`, `prediction_norm=str(pred_norm)`, `correct_norm=task.is_correct(prediction_norm, ex)`, `metric=primary_metric(task.task_type, scoring)`, `primary_correct=correct_norm if metric == "acc_norm" else correct`, `n_prompt_tokens=results[k0].n_context_tokens`, `latency_ms=sum of the K request latencies`, `status="ok"`, `error=None`, `generation/parse_ok/exemplar_ids/n_generated_tokens/exemplar_seed=None`.
- `run_config`: `now = now or datetime.now(timezone.utc)`; `root = project_root or find_project_root(Path.cwd())`; `run_id = f"{now:%Y%m%dT%H%M%SZ}-{config_hash(cfg)[:8]}"`; `torch.manual_seed` is NOT called here (runner never imports torch; the hf factory seeds inside `from_pretrained`); loads every task once (`get_task(name).load(split, n, seed)`; a load error propagates); iterates `expand_grid` with models outermost, loading each model once and releasing it (`del`, `gc.collect()`, and `empty_cache` via `importlib.import_module("torch")` **only if** `"torch" in sys.modules`); a cell exception is logged with `logging.exception`, appended to `failed_cells` as the cell name, and the loop continues; writes `examples.parquet` + `run.json` (metadata exactly §7.5: `schema_version=1, run_id, run_name, config=cfg.model_dump(mode="json"), config_hash, started_at, finished_at, git_sha, git_dirty, library_versions, models={name: info()}, tasks={name: dataset_info() | {"example_ids": [...]}}, failed_cells, summary=summarize(frame, seed=cfg.seed, n_resamples=cfg.bootstrap_resamples).to_dict("records")` with NaN → None); returns `RunResult`. `git_info`: `LLM_EVAL_GIT_SHA` env var → `(sha, None)`; else `git rev-parse HEAD` + `git status --porcelain` in `root` → `(sha, dirty)`; any failure → `(None, None)`. `library_versions`: `python` plus `importlib.metadata.version` of `llm-eval-harness, numpy, pandas, pyarrow, scipy, statsmodels, pydantic` and, if importable without importing them anew (`name in sys.modules`), `torch, transformers, tokenizers, datasets, huggingface_hub` — read via `importlib.metadata.version`, which does not import the package, so include all of them when installed.

- [ ] **Step 1: Add the fixtures to `tests/conftest.py`**

```python
# append to tests/conftest.py
from llm_eval.data.dataset import register_task
from llm_eval.data.hellaswag import HellaSwagTask
from llm_eval.models.base_model import LoglikelihoodResult
from llm_eval.models.registry import register_provider

RUNNER_ROWS = [
    {"ind": i, "activity_label": "T", "ctx_a": "Ctx a.", "ctx_b": "x", "ctx": "", "source_id": f"s{i}",
     "split": "val", "split_type": "indomain", "label": label,
     "endings": ["aa", "bbbb", "cccccc", "dddddddd"]}
    for i, label in enumerate(["0", "3", "0", "0"])
]


class FakeModel:
    """Log-likelihood = -len(continuation): shortest ending wins acc, longest wins acc_norm."""

    supports_loglikelihood = True

    def __init__(self, name="fake", fail_on_context=None):
        self.name, self.fail_on_context, self.calls = name, fail_on_context, 0

    def loglikelihood(self, requests):
        self.calls += 1
        out = []
        for r in requests:
            if self.fail_on_context and self.fail_on_context in r.context:
                raise RuntimeError("boom")
            out.append(LoglikelihoodResult(logprob=-float(len(r.continuation)), n_tokens=len(r.continuation.split()),
                                           n_context_tokens=len(r.context.split()), latency_ms=1.0))
        return out

    def generate(self, requests):
        raise NotImplementedError

    def info(self):
        return {"revision": None, "dtype": "fake", "device": "cpu"}


@register_provider("fake", supports_loglikelihood=True)
def _make_fake(cfg):
    return FakeModel(name=cfg.name, fail_on_context="BOOM" if cfg.path == "fail" else None)


@register_task
class OfflineHellaSwag(HellaSwagTask):
    name = "hellaswag_offline"

    def __init__(self, rows=None):
        super().__init__(rows=rows or (lambda split: RUNNER_ROWS))
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_runner.py
import json
import subprocess
from datetime import datetime, timezone

import pytest

from llm_eval.config import EvalConfig
from llm_eval.evaluation.results import read_run
from llm_eval.runner import RunResult, evaluate_cell, find_project_root, git_info, library_versions, run_config

CFG = {"run_name": "fake_run", "seed": 3, "n_examples": 4, "bootstrap_resamples": 50,
       "models": [{"name": "fake-a", "provider": "fake", "path": "ok"}],
       "tasks": [{"name": "hellaswag_offline"}], "prompt_variants": ["zero_shot"]}
NOW = datetime(2026, 10, 18, 12, 0, 0, tzinfo=timezone.utc)


def test_run_config_writes_run_dir_with_hand_derived_accuracies(tmp_path):
    result = run_config(EvalConfig(**CFG), project_root=tmp_path, now=NOW)
    assert isinstance(result, RunResult) and result.failed_cells == []
    assert result.run_dir.parent == tmp_path / "data" / "processed" / "runs" / "fake_run"
    assert result.run_dir.name.startswith("20261018T120000Z-") and len(result.run_dir.name) == 25
    frame, meta = read_run(result.run_dir)
    assert len(frame) == 4
    # acc: shortest ending ("aa", index 0) always predicted → labels 0,3,0,0 → 3/4
    # acc_norm: longest ending (index 3) always predicted → 1/4
    assert frame["prediction"].tolist() == ["0"] * 4 and frame["prediction_norm"].tolist() == ["3"] * 4
    assert frame["correct"].tolist() == [True, False, True, True]
    assert frame["correct_norm"].tolist() == [False, True, False, False]
    assert frame["metric"].unique().tolist() == ["acc_norm"]
    assert frame["primary_correct"].tolist() == frame["correct_norm"].tolist()
    assert frame["prompt"].tolist()[0] == "T: Ctx a. X"
    assert list(frame["choice_logprobs"].iloc[0]) == [-3.0, -5.0, -7.0, -9.0]
    assert frame["n_prompt_tokens"].tolist() == [4] * 4 and frame["latency_ms"].tolist() == [4.0] * 4
    assert frame["status"].tolist() == ["ok"] * 4 and frame["scoring"].unique().tolist() == ["loglikelihood"]
    [cell] = meta["summary"]
    assert cell["acc"] == 0.75 and cell["acc_norm"] == 0.25 and cell["primary_acc"] == 0.25 and cell["n"] == 4
    assert meta["schema_version"] == 1 and meta["run_id"] == result.run_dir.name
    assert meta["config"]["seed"] == 3 and result.run_dir.name.endswith(meta["config_hash"][:8])
    assert meta["tasks"]["hellaswag_offline"]["example_ids"] == [f"hellaswag_offline/validation/{i}" for i in range(4)]
    assert meta["models"]["fake-a"] == {"revision": None, "dtype": "fake", "device": "cpu"}
    assert set(meta["library_versions"]) >= {"python", "numpy", "pandas", "pyarrow", "scipy", "statsmodels", "pydantic", "llm_eval"}
    assert "started_at" in meta and "finished_at" in meta and meta["failed_cells"] == []


def test_evaluate_cell_issues_one_loglikelihood_call_per_cell():
    from llm_eval.data.dataset import get_task
    from llm_eval.features.prompts import PromptVariant
    from llm_eval.models.base_model import ScoringMode
    from tests.conftest import FakeModel

    task = get_task("hellaswag_offline")
    examples = task.load("validation", 4, seed=0)
    model = FakeModel()
    records = evaluate_cell(model, task, examples, PromptVariant.ZERO_SHOT, ScoringMode.LOGLIKELIHOOD, run_id="r", seed=0)
    assert model.calls == 1 and len(records) == 4
    assert records[1].correct is False and records[1].correct_norm is True and records[1].example_index == 1


def test_failed_cell_is_recorded_and_other_cells_complete(tmp_path):
    cfg = dict(CFG, models=[{"name": "bad", "provider": "fake", "path": "fail"}, {"name": "good", "provider": "fake", "path": "ok"}])
    cfg["tasks"] = [{"name": "hellaswag_offline"}]
    from tests import conftest  # FakeModel raises when the context contains "BOOM"
    conftest.RUNNER_ROWS[0]["ctx_a"] = "BOOM a."
    try:
        result = run_config(EvalConfig(**cfg), project_root=tmp_path, now=NOW)
    finally:
        conftest.RUNNER_ROWS[0]["ctx_a"] = "Ctx a."
    assert result.failed_cells == ["bad×hellaswag_offline×zero_shot"]
    frame, meta = read_run(result.run_dir)
    assert frame["model"].unique().tolist() == ["good"] and meta["failed_cells"] == result.failed_cells


def test_run_config_propagates_task_load_errors(tmp_path):
    with pytest.raises(ValueError, match="n_examples=9 exceeds 4 rows"):
        run_config(EvalConfig(**dict(CFG, n_examples=9)), project_root=tmp_path, now=NOW)


def test_git_info_without_repo_returns_none(tmp_path, monkeypatch):
    monkeypatch.delenv("LLM_EVAL_GIT_SHA", raising=False)
    assert git_info(tmp_path) == (None, None)


def test_git_info_env_override_and_real_repo(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_EVAL_GIT_SHA", "abc123")
    assert git_info(tmp_path) == ("abc123", None)
    monkeypatch.delenv("LLM_EVAL_GIT_SHA")
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_path, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "x"], cwd=tmp_path, check=True)
    sha, dirty = git_info(tmp_path)
    assert isinstance(sha, str) and len(sha) == 40 and dirty is False
    (tmp_path / "f").write_text("x")
    assert git_info(tmp_path)[1] is True


def test_find_project_root_prefers_pyproject_ancestor_else_start(tmp_path):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    deep = tmp_path / "a" / "b"
    deep.mkdir(parents=True)
    assert find_project_root(deep) == tmp_path
    bare = tmp_path / "zz"
    bare.mkdir()
    (tmp_path / "pyproject.toml").unlink()
    assert find_project_root(bare) == bare


def test_library_versions_reports_installed_packages():
    v = library_versions()
    assert v["python"].startswith("3.11") and "torch" in v and "transformers" in v and "llm_eval" in v
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run pytest tests/test_runner.py -v`
Expected: ERROR with `ModuleNotFoundError: No module named 'llm_eval.runner'`

- [ ] **Step 4: Implement `runner.py` per the Interfaces block**

Cell names are `f"{model.name}×{task.name}×{variant.value}"`. Use `logging.getLogger("llm_eval")` and `tqdm` for a per-cell progress bar (disabled when not a TTY).

- [ ] **Step 5: Run the tests to verify they pass**

Run: `uv run pytest tests/test_runner.py -v`
Expected: 8 passed

- [ ] **Step 6: Lint and commit**

```bash
uv run ruff format src tests && uv run ruff check src tests && uv run pytest -q
git add src/llm_eval/runner.py tests/conftest.py tests/test_runner.py
git commit -m "feat: runner — grid orchestration, provenance-rich run.json, failed-cell handling"
```

---

### Task 11: CLI, import hygiene, README skeleton, api_spec, experiment log

**Files:**
- Create: `src/llm_eval/cli.py`, `tests/test_cli.py`, `tests/test_imports.py`, `docs/api_spec.md`, `docs/experiment_log.md`
- Modify: `README.md` (replace the placeholder)

**Interfaces:**
- Consumes: `load_config`, `run_config`, `read_run`, `format_summary`.
- Produces (§9): `main(argv=None) -> int`; `run` subcommand (`--config` required, `--output-dir`, `--n-examples`, `--seed`, `--allow-errors`) prints `run dir: <absolute path>` then `format_summary(...)`, returns 0, or 1 when `failed_cells` is non-empty or (without `--allow-errors`) any row has `status == "error"`; `report` subcommand (`--run-dir`) prints `format_summary(pd.DataFrame(meta["summary"]))`, returns 0. Logging configured at INFO to stderr.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_cli.py
import json
from pathlib import Path

import yaml

from llm_eval.cli import main
from tests.test_runner import CFG


def write_cfg(tmp_path, **over):
    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(dict(CFG, **over)))
    return p


def test_cli_run_and_report(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["run", "--config", str(write_cfg(tmp_path)), "--n-examples", "3"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("run dir: ")
    run_dir = Path(out.splitlines()[0].removeprefix("run dir: "))
    assert run_dir.is_absolute() and run_dir.is_dir()
    assert "| fake-a | hellaswag_offline | zero_shot | acc_norm | 3 |" in out
    meta = json.loads((run_dir / "run.json").read_text())
    assert meta["config"]["n_examples"] == 3
    assert main(["report", "--run-dir", str(run_dir)]) == 0
    assert "acc_norm" in capsys.readouterr().out


def test_cli_run_from_foreign_cwd_uses_cwd_as_project_root(tmp_path, capsys, monkeypatch):
    foreign = tmp_path / "elsewhere"
    foreign.mkdir()
    monkeypatch.chdir(foreign)
    assert main(["run", "--config", str(write_cfg(tmp_path))]) == 0
    run_dir = Path(capsys.readouterr().out.splitlines()[0].removeprefix("run dir: "))
    assert run_dir.parent == foreign / "data" / "processed" / "runs" / "fake_run"


def test_cli_returns_1_on_failed_cell(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from tests import conftest
    conftest.RUNNER_ROWS[0]["ctx_a"] = "BOOM a."
    try:
        code = main(["run", "--config", str(write_cfg(tmp_path, models=[{"name": "bad", "provider": "fake", "path": "fail"}]))])
    finally:
        conftest.RUNNER_ROWS[0]["ctx_a"] = "Ctx a."
    assert code == 1
```

```python
# tests/test_imports.py
import subprocess
import sys


def test_config_results_report_never_import_torch_or_datasets():
    code = ("import sys, llm_eval.config, llm_eval.evaluation.results, llm_eval.evaluation.report, llm_eval.cli;"
            "print(sorted(m for m in ('torch', 'transformers', 'datasets') if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True).stdout
    assert out.strip() == "[]"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/test_cli.py tests/test_imports.py -v`
Expected: `test_cli.py` ERROR with `ModuleNotFoundError: No module named 'llm_eval.cli'`; `test_imports.py` fails on the same import

- [ ] **Step 3: Implement `cli.py` per the Interfaces block, then the docs**

- `docs/api_spec.md`: sections *CLI* (the three command lines of §9 with every flag), *Config schema* (a fenced JSON block produced by `uv run python -c "import json; from llm_eval.config import EvalConfig; print(json.dumps(EvalConfig.model_json_schema(), indent=2))"` plus the §8.1 YAML example), *Results schema* (the §8.2 column table verbatim), *run.json* (field list of §7.5/§8.2), *Python API* (the §6 signatures of `run_config`, `evaluate_cell`, `read_run`, `summarize`).
- `docs/experiment_log.md`: header table `| date | run_name | run_id | config_hash | device/dtype | headline | notes |` with no rows yet.
- `README.md` skeleton (≈120 lines): title + one-line pitch; a **Results** section with the placeholder sentence `<!-- filled by Task 12 -->` and an empty table shell with columns `model | task | prompt | metric | n | acc [95% CI] | acc_norm [95% CI]`; **Why this exists** (3 bullets from §1); **Quick start** (`uv sync --all-extras --all-groups`, `make smoke`, `uv run llm-eval report --run-dir …`, `make test`); **Experiment protocol** with the three hypotheses of §7.7 item 5 and the decision rule of §7.7 items 1–4, plus a *Test–retest* subsection listing the three numbers of §7.5 as "measured on Day 3" placeholders; **Statistical methods** (bootstrap §7.6 bullet 3, paired tests bullet 4, multiplicity bullet 5, variance bullet 6, power bullet 7 — one sentence each); **Architecture** (the §5 mermaid block and the §4 tree down to one level); **Adding a task / a model** (one decorator each, with the §6 names); **Running on a GPU box** (`uv sync --extra local`, `uv pip install --torch-backend=auto torch`, activate the venv, `llm-eval run …`; link to `docs/cloud_compute_options.md`); **Project layout** mapping to `code_standard.md`; **License** MIT.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run pytest tests/test_cli.py tests/test_imports.py -v`
Expected: 4 passed

- [ ] **Step 5: Lint and commit**

```bash
uv run ruff format src tests && uv run ruff check src tests && uv run pytest -q
git add src/llm_eval/cli.py tests/test_cli.py tests/test_imports.py docs/api_spec.md docs/experiment_log.md README.md
git commit -m "feat: llm-eval CLI (run, report), import-hygiene test, README skeleton and API spec"
```

---

### Task 12: Real smoke run, slow integration test, committed sample, README numbers

**Files:**
- Create: `tests/integration/__init__.py`, `tests/integration/test_qwen_smoke.py`
- Create: `data/samples/runs/smoke_qwen_hellaswag/<run_id>/examples.parquet` + `run.json` (copied from the real run)
- Modify: `README.md` (Results section), `docs/experiment_log.md` (first row)

**Interfaces:**
- Consumes: everything. The HF cache already holds `Qwen/Qwen2.5-0.5B` and `Rowan/hellaswag` (`HF_HUB_OFFLINE=1` works).

- [ ] **Step 1: Write the slow test**

```python
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
    first = evaluate_cell(model, task, examples, PromptVariant.ZERO_SHOT, ScoringMode.LOGLIKELIHOOD, run_id="t", seed=42)
    second = evaluate_cell(model, task, examples, PromptVariant.ZERO_SHOT, ScoringMode.LOGLIKELIHOOD, run_id="t", seed=42)
    assert [r.prediction for r in first] == [r.prediction for r in second]
    assert 0.0 <= sum(r.correct for r in first) / 8 <= 1.0
    req = LoglikelihoodRequest(task.context(examples[0]), " " + examples[0].choices[0])
    [single] = model.loglikelihood([req])
    assert math.isclose(single.logprob, first[0].choice_logprobs[0], abs_tol=1e-3)
```

- [ ] **Step 2: Run it**

Run: `HF_HUB_OFFLINE=1 uv run pytest -m slow tests/integration -v`
Expected: 1 passed (≈10–30 s on MPS)

- [ ] **Step 3: Run the real smoke evaluation and inspect**

Run: `HF_HUB_OFFLINE=1 make smoke` then `uv run llm-eval report --run-dir <printed run dir>`
Expected: exit 0; 100 rows; `acc_norm` within 0.40–0.65 and `acc` within 0.33–0.50 (§7.1 anchors; outside that range → stop and report DONE_WITH_CONCERNS). Record wall time.

- [ ] **Step 4: Commit the sample and the numbers**

```bash
mkdir -p data/samples/runs/smoke_qwen_hellaswag && cp -r <run dir> data/samples/runs/smoke_qwen_hellaswag/
du -sh data/samples/runs/smoke_qwen_hellaswag   # must be < 1 MB
```
Paste the `format_summary` table into the README Results section (replacing the placeholder) with one sentence: "Qwen2.5-0.5B, HellaSwag validation, 100 examples (seed 42), zero-shot log-likelihood, float32 on MPS; interval = seeded percentile bootstrap (1000 resamples)." Add the first row to `docs/experiment_log.md` (date, run_name, run_id, config_hash, device/dtype, `acc_norm x.xx [lo, hi]`, wall time).

- [ ] **Step 5: Verify, lint, commit**

```bash
uv run pytest -q && uv run ruff check src tests && uv run ruff format --check src tests
git add tests/integration data/samples/runs/smoke_qwen_hellaswag README.md docs/experiment_log.md
git commit -m "test: slow Qwen2.5-0.5B smoke test; commit Day 1 sample run and README results"
```
