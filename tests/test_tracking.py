# tests/test_tracking.py
import pytest

from llm_eval.cli import main
from tests.test_analysis import OUTCOMES, make_run

mlflow = pytest.importorskip("mlflow")


def test_track_logs_each_run_and_the_analysis_once(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)  # MLflow puts artifacts under ./mlruns
    runs = tmp_path / "runs"
    run_dir = make_run(runs, "run", OUTCOMES)
    # make_run writes a minimal run.json; give it the fields tracking reads
    import json

    meta = json.loads((run_dir / "run.json").read_text())
    meta.update(
        run_id="r1",
        run_name="run",
        git_sha="abc",
        git_dirty=False,
        failed_cells=[],
        models={"big": {"path": "x/big", "revision": "0123456789abcdef"}},
        tasks={
            "gsm8k": {
                "path": "openai/gsm8k",
                "revision": "740312a",
                "split": "test",
                "example_ids": ["a", "b"],
            }
        },
        summary=[
            {
                "model": "big",
                "task": "gsm8k",
                "prompt_variant": "zero_shot",
                "primary_acc": 0.8,
                "primary_ci_low": 0.7,
                "primary_ci_high": 0.9,
                "n_ok": 200,
            }
        ],
    )
    meta["config"].update(
        scoring="generation", prompt_variants=["zero_shot"], bootstrap_resamples=100
    )
    (run_dir / "run.json").write_text(json.dumps(meta))
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    results = tmp_path / "results.md"
    assert (
        main(
            [
                "analyze",
                "--runs-root",
                str(runs),
                "--out",
                str(results),
                "--json",
                str(results.with_suffix(".json")),
                "--resamples",
                "50",
            ]
        )
        == 0
    )
    assert (
        main(["track", "--runs-root", str(runs), "--tracking-uri", uri, "--analysis", str(results)])
        == 0
    )
    assert (
        main(["track", "--runs-root", str(runs), "--tracking-uri", uri, "--analysis", str(results)])
        == 0
    )
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert sum(line.startswith("logged: ") for line in lines) == 1
    assert sum(line.startswith("already logged: ") for line in lines) == 1
    assert sum(line.startswith("logged analysis ") for line in lines) == 1
    assert sum(line.startswith("already logged analysis: ") for line in lines) == 1
    mlflow.set_tracking_uri(uri)
    logged = mlflow.search_runs(experiment_names=["llm-eval-harness"])
    by_name = dict(zip(logged["tags.mlflow.runName"], logged.index, strict=True))
    run_row = logged.loc[by_name["run/r1"]]
    assert run_row["metrics.big/gsm8k/zero_shot/primary_acc"] == pytest.approx(0.8)
    assert run_row["params.model/big"] == "x/big@0123456789ab"
    assert run_row["tags.config_hash"] == "h1"
    assert (logged["tags.mlflow.runName"] == "analysis/results").sum() == 1
