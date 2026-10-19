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
        code = main(
            [
                "run",
                "--config",
                str(
                    write_cfg(
                        tmp_path, models=[{"name": "bad", "provider": "fake", "path": "fail"}]
                    )
                ),
            ]
        )
    finally:
        conftest.RUNNER_ROWS[0]["ctx_a"] = "Ctx a."
    assert code == 1
