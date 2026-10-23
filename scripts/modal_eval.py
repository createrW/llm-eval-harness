"""Run the local-model grid on a cloud GPU with Modal (hosting plan Part 3.7: on-demand GPU).

This is batch inference, not training: the models score and answer the test sets; no weights
change. One job runs every config ``--reruns`` times with the same seed, rerun-major, exactly like
``scripts/run_grid.sh`` on the laptop, and records the local git commit as the code version.

One-time setup (your Modal account; the Starter plan includes $30 of credit a month):

    uv run --with modal modal setup

Run from the repo root with every tracked change committed:

    uv run --with modal modal run scripts/modal_eval.py
    uv run --with modal modal run scripts/modal_eval.py --reruns 1 \
        --configs configs/eval/local_ll.yaml
    LLM_EVAL_GPU=L4 uv run --with modal modal run scripts/modal_eval.py

The GPU is an A10G unless ``LLM_EVAL_GPU`` names another Modal GPU type. Each run is saved to the
Modal volume ``llm-eval-runs`` as soon as it finishes, so a timeout or crash keeps the runs already
done; at the end they are copied into ``data/processed/runs/``. To fetch them by hand:

    uv run --with modal modal volume get llm-eval-runs / data/processed/runs

Model and dataset downloads are cached in the volume ``llm-eval-hf-cache`` for later jobs.
"""

from __future__ import annotations

import os
import subprocess
import traceback
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/root/llm_eval_harness"
GPU = os.environ.get("LLM_EVAL_GPU", "A10G")
TORCH = "torch==2.14.1"  # the version in uv.lock; PyPI's Linux wheel is the CUDA build
IGNORE = [
    ".git",
    ".venv",
    "data",
    "docs",
    "notebooks",
    "mlruns",
    "mlflow.db",
    ".superpowers",
    "**/__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(TORCH)
    .uv_pip_install(requirements=[str(ROOT / "requirements-cuda.txt")])  # make requirements
    .add_local_dir(ROOT, REMOTE, copy=True, ignore=IGNORE)
    .run_commands(f"python -m pip install --no-deps {REMOTE}")
    .workdir(REMOTE)
    .env({"PYTHONUNBUFFERED": "1", "TOKENIZERS_PARALLELISM": "false"})
)
hf_cache = modal.Volume.from_name("llm-eval-hf-cache", create_if_missing=True)
runs = modal.Volume.from_name("llm-eval-runs", create_if_missing=True)
app = modal.App("llm-eval-harness")


@app.function(
    image=image,
    gpu=GPU,
    timeout=6 * 3600,
    volumes={"/root/.cache/huggingface": hf_cache, "/runs": runs},
)
def run_grid(configs: list[str], reruns: int, git_sha: str) -> list[str]:
    """Run each config ``reruns`` times; return the finished run dirs relative to the volume."""
    import torch

    from llm_eval.cli import main as llm_eval

    os.environ["LLM_EVAL_GIT_SHA"] = git_sha
    print(
        f"{torch.cuda.get_device_name(0)} · torch {torch.__version__} · CUDA {torch.version.cuda}"
    )
    finished: list[str] = []
    for i in range(1, reruns + 1):
        for config in configs:
            before = {p.parent for p in Path("/runs").glob("*/*/run.json")}
            print(f"=== rerun {i}/{reruns}: {config}", flush=True)
            try:
                status = str(llm_eval(["run", "--config", config, "--output-dir", "/runs"]))
            except Exception:  # keep going: the other configs and reruns may still run
                traceback.print_exc()
                status = "crashed"
            hf_cache.commit()
            runs.commit()
            new = sorted({p.parent for p in Path("/runs").glob("*/*/run.json")} - before)
            finished += [str(p.relative_to("/runs")) for p in new]
            print(f"=== rerun {i}/{reruns}: {config} -> exit {status}, saved {new}", flush=True)
    return finished


@app.local_entrypoint()
def main(
    configs: str = "configs/eval/local_ll.yaml,configs/eval/local_gen.yaml",
    reruns: int = 3,
    out: str = "data/processed/runs",
    allow_dirty: bool = False,
) -> None:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()

    sha = git("rev-parse", "HEAD")
    if git("status", "--porcelain", "--untracked-files=no") and not allow_dirty:
        raise SystemExit("Commit tracked changes first: every run records the commit it ran.")
    names = [c.strip() for c in configs.split(",") if c.strip()]
    missing = [c for c in names if not (ROOT / c).is_file()]
    if missing:
        raise SystemExit(f"configs not found: {missing}")
    print(f"Modal {GPU}: {reruns} rerun(s) of {names} at commit {sha[:7]}")
    finished = run_grid.remote(names, reruns, sha)
    for rel in finished:
        target = ROOT / out / rel
        target.mkdir(parents=True, exist_ok=True)
        for name in ("examples.parquet", "run.json"):
            with (target / name).open("wb") as fh:
                for chunk in runs.read_file(f"{rel}/{name}"):
                    fh.write(chunk)
        print(f"downloaded {target.relative_to(ROOT)}")
    print(
        f"{len(finished)} run(s) copied. Next: copy them into data/samples/runs/ and make analyze."
    )
