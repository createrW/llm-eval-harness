import subprocess
from datetime import UTC, datetime

import pytest

from llm_eval.config import EvalConfig
from llm_eval.evaluation.results import read_run
from llm_eval.runner import (
    RunResult,
    evaluate_cell,
    find_project_root,
    git_info,
    library_versions,
    run_config,
)

CFG = {
    "run_name": "fake_run",
    "seed": 3,
    "n_examples": 4,
    "bootstrap_resamples": 50,
    "models": [{"name": "fake-a", "provider": "fake", "path": "ok"}],
    "tasks": [{"name": "hellaswag_offline"}],
    "prompt_variants": ["zero_shot"],
}
NOW = datetime(2026, 10, 18, 12, 0, 0, tzinfo=UTC)


def test_run_config_writes_run_dir_with_hand_derived_accuracies(tmp_path):
    result = run_config(EvalConfig(**CFG), project_root=tmp_path, now=NOW)
    assert isinstance(result, RunResult) and result.failed_cells == []
    assert result.run_dir.parent == tmp_path / "data" / "processed" / "runs" / "fake_run"
    assert result.run_dir.name.startswith("20261018T120000Z-") and len(result.run_dir.name) == 25
    frame, meta = read_run(result.run_dir)
    assert len(frame) == 4
    # acc: shortest ending ("aa", index 0) always predicted → labels 0,3,0,0 → 3/4
    # acc_norm: longest ending (index 3) always predicted → 1/4
    assert (
        frame["prediction"].tolist() == ["0"] * 4 and frame["prediction_norm"].tolist() == ["3"] * 4
    )
    assert frame["correct"].tolist() == [True, False, True, True]
    assert frame["correct_norm"].tolist() == [False, True, False, False]
    assert frame["metric"].unique().tolist() == ["acc_norm"]
    assert frame["primary_correct"].tolist() == frame["correct_norm"].tolist()
    assert frame["prompt"].tolist()[0] == "T: Ctx a. X"
    assert list(frame["choice_logprobs"].iloc[0]) == [-3.0, -5.0, -7.0, -9.0]
    assert (
        frame["n_prompt_tokens"].tolist() == [4] * 4 and frame["latency_ms"].tolist() == [4.0] * 4
    )
    assert frame["status"].tolist() == ["ok"] * 4 and frame["scoring"].unique().tolist() == [
        "loglikelihood"
    ]
    [cell] = meta["summary"]
    assert (
        cell["acc"] == 0.75
        and cell["acc_norm"] == 0.25
        and cell["primary_acc"] == 0.25
        and cell["n"] == 4
    )
    assert meta["schema_version"] == 1 and meta["run_id"] == result.run_dir.name
    assert meta["config"]["seed"] == 3 and result.run_dir.name.endswith(meta["config_hash"][:8])
    assert meta["tasks"]["hellaswag_offline"]["example_ids"] == [
        f"hellaswag_offline/validation/{i}" for i in range(4)
    ]
    assert meta["models"][
        "fake-a"
    ] == {  # design §7.5: per model {provider, path, revision, dtype, device}
        "provider": "fake",
        "path": "ok",
        "revision": None,
        "dtype": "fake",
        "device": "cpu",
    }
    assert set(meta["library_versions"]) >= {
        "python",
        "numpy",
        "pandas",
        "pyarrow",
        "scipy",
        "statsmodels",
        "pydantic",
        "llm_eval",
    }
    assert "started_at" in meta and "finished_at" in meta and meta["failed_cells"] == []


def test_run_config_loads_each_model_with_the_config_seed(tmp_path, monkeypatch):
    import llm_eval.runner as runner

    seen = []
    real = runner.load_model

    def spy(mcfg, *, seed=None):
        seen.append((mcfg.name, seed))
        return real(mcfg, seed=seed)

    monkeypatch.setattr(runner, "load_model", spy)
    cfg = dict(CFG, models=[{"name": n, "provider": "fake", "path": "ok"} for n in ("a", "b")])
    result = run_config(EvalConfig(**cfg), project_root=tmp_path, now=NOW)
    assert seen == [("a", 3), ("b", 3)] and result.failed_cells == []


def test_evaluate_cell_issues_one_loglikelihood_call_per_cell():
    from llm_eval.data.dataset import get_task
    from llm_eval.features.prompts import PromptVariant
    from llm_eval.models.base_model import ScoringMode
    from tests.conftest import FakeModel

    task = get_task("hellaswag_offline")
    examples = task.load("validation", 4, seed=0)
    model = FakeModel()
    records = evaluate_cell(
        model,
        task,
        examples,
        PromptVariant.ZERO_SHOT,
        ScoringMode.LOGLIKELIHOOD,
        run_id="r",
        seed=0,
    )
    assert model.calls == 1 and len(records) == 4
    assert (
        records[1].correct is False
        and records[1].correct_norm is True
        and records[1].example_index == 1
    )


def test_failed_cell_is_recorded_and_other_cells_complete(tmp_path):
    cfg = dict(
        CFG,
        models=[
            {"name": "bad", "provider": "fake", "path": "fail"},
            {"name": "good", "provider": "fake", "path": "ok"},
        ],
    )
    cfg["tasks"] = [{"name": "hellaswag_offline"}]
    from tests import conftest  # FakeModel raises when the context contains "BOOM"

    conftest.RUNNER_ROWS[0]["ctx_a"] = "BOOM a."
    try:
        result = run_config(EvalConfig(**cfg), project_root=tmp_path, now=NOW)
    finally:
        conftest.RUNNER_ROWS[0]["ctx_a"] = "Ctx a."
    assert result.failed_cells == ["bad×hellaswag_offline×zero_shot"]
    frame, meta = read_run(result.run_dir)
    assert (
        frame["model"].unique().tolist() == ["good"] and meta["failed_cells"] == result.failed_cells
    )


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
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=t",
            "-c",
            "user.email=t@t",
            "commit",
            "-q",
            "--allow-empty",
            "-m",
            "x",
        ],
        cwd=tmp_path,
        check=True,
    )
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
    assert (
        v["python"].startswith("3.11") and "torch" in v and "transformers" in v and "llm_eval" in v
    )


def test_run_config_logs_progress_per_model_and_cell(tmp_path, caplog):
    caplog.set_level("INFO", logger="llm_eval")
    run_config(EvalConfig(**CFG), project_root=tmp_path, now=NOW)
    messages = [r.getMessage() for r in caplog.records if r.name == "llm_eval"]
    assert any(m.startswith("model fake-a: device=cpu") for m in messages)
    assert any(
        m.startswith("cell fake-a×hellaswag_offline×zero_shot: 4 examples") for m in messages
    )
