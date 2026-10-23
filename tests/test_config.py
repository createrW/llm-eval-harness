# tests/test_config.py
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from llm_eval.config import EvalConfig, GridCell, config_hash, expand_grid, load_config
from llm_eval.features.prompts import PromptVariant, UnsupportedCellError
from llm_eval.models.base_model import ScoringMode

SMOKE = Path(__file__).resolve().parents[1] / "configs" / "eval" / "smoke_qwen_hellaswag.yaml"

BASE = {
    "run_name": "t",
    "models": [{"name": "q", "provider": "hf", "path": "Qwen/Qwen2.5-0.5B"}],
    "tasks": [{"name": "hellaswag"}],
    "prompt_variants": ["zero_shot"],
}


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
    assert (
        cfg.models[0].device == "auto"
        and cfg.models[0].batch_size == 8
        and cfg.models[0].max_new_tokens is None
    )
    assert cfg.tasks[0].name == "hellaswag" and cfg.tasks[0].split == "validation"
    assert cfg.prompt_variants == [PromptVariant.ZERO_SHOT]


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda d: d.update(extra_field=1), "extra_field"),
        (lambda d: d.update(n_examples=0), "n_examples"),
        (lambda d: d["models"][0].update(provider="nope"), "provider"),
        (lambda d: d["tasks"][0].update(name="nope"), "task"),
        (lambda d: d["models"].append(dict(d["models"][0])), "unique"),
        (lambda d: d["tasks"].append(dict(d["tasks"][0])), "unique"),
        (lambda d: d["tasks"].append({"name": "hellaswag", "split": "train"}), "unique"),
        (lambda d: d.update(prompt_variants=[]), "prompt_variants"),
        (lambda d: d.update(prompt_variants=["zero_shot", "zero_shot"]), "unique"),
        (lambda d: d["models"][0].update(dtype="int8"), "dtype"),
        (lambda d: d["tasks"][0].update(n_examples=0), "n_examples"),
        (lambda d: d.update(seed=-1), "seed"),
        (lambda d: d.update(seed=2**63), "seed"),
        (lambda d: d.update(exemplar_seed=-1), "exemplar_seed"),
        (lambda d: d.update(exemplar_seed=2**63), "exemplar_seed"),
        (lambda d: d.update(run_name="../x"), "run_name"),
        (lambda d: d.update(run_name=""), "run_name"),
        (lambda d: d.update(models=[]), "models"),
        (lambda d: d.update(tasks=[]), "tasks"),
        (lambda d: d["models"][0].update(max_new_tokens=0), "max_new_tokens"),
    ],
)
def test_invalid_configs_are_rejected(tmp_path, mutation, message):
    data = yaml.safe_load(yaml.safe_dump(BASE))
    mutation(data)
    with pytest.raises(ValidationError, match=message):
        load_config(write(tmp_path, data))


def test_overrides_apply_before_validation(tmp_path):
    cfg = load_config(
        write(tmp_path, BASE), overrides={"n_examples": 5, "seed": 9, "output_dir": None}
    )
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
        ("q", "hellaswag", "zero_shot"),
        ("r", "hellaswag", "zero_shot"),
    ]


def test_expand_grid_rejects_few_shot_on_day_1():
    data = yaml.safe_load(yaml.safe_dump(BASE))
    data["prompt_variants"] = ["zero_shot", "few_shot_3"]
    with pytest.raises(UnsupportedCellError, match=r"q×hellaswag×few_shot_3: .* lands on Day 2"):
        expand_grid(EvalConfig(**data))


def test_expand_grid_rejects_forced_generation_on_day_1():
    data = dict(BASE, scoring="generation")
    with pytest.raises(UnsupportedCellError, match=r"q×hellaswag×zero_shot: .* lands on Day 3"):
        expand_grid(EvalConfig(**data))


def test_expand_grid_rejects_explicit_loglikelihood_on_provider_without_it(monkeypatch):
    from llm_eval.models.registry import PROVIDERS, ProviderSpec

    monkeypatch.setitem(
        PROVIDERS, "nolp", ProviderSpec(factory=lambda cfg: None, supports_loglikelihood=False)
    )
    data = yaml.safe_load(yaml.safe_dump(BASE))
    data["models"][0]["provider"] = "nolp"
    data["scoring"] = "loglikelihood"
    with pytest.raises(
        UnsupportedCellError, match=r"q×hellaswag×zero_shot: .*nolp.* log-likelihood"
    ):
        expand_grid(EvalConfig(**data))


def test_config_hash_ignores_output_dir_but_not_seed(tmp_path):
    a = EvalConfig(**BASE)
    b = EvalConfig(**dict(BASE, output_dir="/elsewhere"))
    c = EvalConfig(**dict(BASE, seed=1))
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash(c)
    assert len(config_hash(a)) == 12 and int(config_hash(a), 16) >= 0
