"""MLflow logging of run directories and of the analysis (brief item: self-hosted MLflow).

Each run directory becomes one MLflow run in experiment ``llm-eval-harness``: tags for the run id,
config hash and git state, the config and model revisions as params, every summary cell's primary
accuracy, CI and n as metrics, and ``run.json`` as an artifact. Logging is idempotent: a run whose
id is already tagged in the experiment is skipped, and so is an analysis whose tables were logged
before. The store is a local SQLite file
(``sqlite:///mlflow.db``, artifacts under ``mlruns/``), viewed with ``make mlflow-ui``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from llm_eval.evaluation.results import RUN_FILE

EXPERIMENT = "llm-eval-harness"
DEFAULT_URI = "sqlite:///mlflow.db"


def _mlflow(tracking_uri: str, experiment: str) -> Any:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    return mlflow


def _already_logged(mlflow: Any, experiment: str, tag: str, value: str) -> bool:
    found = mlflow.search_runs(
        experiment_names=[experiment], filter_string=f"tags.{tag} = '{value}'", max_results=1
    )
    return len(found) > 0


def _metric_name(*parts: str) -> str:
    return "/".join(p.replace("×", "x") for p in parts)


def log_run(
    run_dir: Path, *, tracking_uri: str = DEFAULT_URI, experiment: str = EXPERIMENT
) -> str | None:
    """Log one run directory; returns the MLflow run id, or ``None`` if it was logged before."""
    meta = json.loads((Path(run_dir) / RUN_FILE).read_text(encoding="utf-8"))
    mlflow = _mlflow(tracking_uri, experiment)
    if _already_logged(mlflow, experiment, "llm_eval_run_id", meta["run_id"]):
        return None
    config = meta["config"]
    with mlflow.start_run(run_name=f"{meta['run_name']}/{meta['run_id']}") as active:
        mlflow.set_tags(
            {
                "llm_eval_run_id": meta["run_id"],
                "config_hash": meta["config_hash"],
                "git_sha": str(meta.get("git_sha")),
                "git_dirty": str(meta.get("git_dirty")),
                "failed_cells": ",".join(meta.get("failed_cells", [])),
            }
        )
        params = {
            "run_name": meta["run_name"],
            "seed": config["seed"],
            "scoring": config["scoring"],
            "prompt_variants": ",".join(config["prompt_variants"]),
            "bootstrap_resamples": config["bootstrap_resamples"],
        }
        for name, info in meta.get("models", {}).items():
            params[f"model/{name}"] = f"{info['path']}@{str(info.get('revision'))[:12]}"
        for name, info in meta.get("tasks", {}).items():
            params[f"task/{name}"] = (
                f"{info['path']}@{str(info['revision'])[:12]} {info['split']}"
                f" n={len(info['example_ids'])}"
            )
        mlflow.log_params(params)
        metrics = {}
        for row in meta.get("summary", []):
            key = (row["model"], row["task"], row["prompt_variant"])
            if row.get("primary_acc") is not None:
                metrics[_metric_name(*key, "primary_acc")] = row["primary_acc"]
                metrics[_metric_name(*key, "primary_ci_low")] = row["primary_ci_low"]
                metrics[_metric_name(*key, "primary_ci_high")] = row["primary_ci_high"]
            metrics[_metric_name(*key, "n_ok")] = row["n_ok"]
        if metrics:
            mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(Path(run_dir) / RUN_FILE))
        return str(active.info.run_id)


def log_analysis(
    markdown: Path,
    tables: Path,
    *,
    label: str,
    tracking_uri: str = DEFAULT_URI,
    experiment: str = EXPERIMENT,
) -> str | None:
    """Log the results document and its JSON tables as one MLflow run named ``analysis/<label>``;
    returns ``None`` if these exact tables were logged before."""
    mlflow = _mlflow(tracking_uri, experiment)
    digest = hashlib.sha256(Path(tables).read_bytes()).hexdigest()[:16]
    if _already_logged(mlflow, experiment, "analysis_sha256", digest):
        return None
    payload = json.loads(Path(tables).read_text(encoding="utf-8"))
    with mlflow.start_run(run_name=f"analysis/{label}") as active:
        mlflow.set_tags({"kind": "analysis", "analysis_sha256": digest})
        metrics = {}
        for fam in payload.get("families", []):
            prefix = f"{fam['task']}/{fam['scoring']}"
            if fam["mde"].get("mde") is not None:
                metrics[f"{prefix}/mde"] = fam["mde"]["mde"]
            for share in (fam.get("variance") or {}).get("shares", {}).items():
                metrics[f"{prefix}/variance_share/{share[0]}"] = share[1]["share"]
        retest = payload.get("test_retest")
        if retest:
            metrics["test_retest/agreement"] = retest["overall"]["agreement"]
            if retest["conclusions"]["stability"] is not None:
                metrics["test_retest/conclusion_stability"] = retest["conclusions"]["stability"]
        if metrics:
            mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(markdown))
        mlflow.log_artifact(str(tables))
        return str(active.info.run_id)
