"""Run the local-model grid on a cloud GPU with Modal (hosting plan Part 3.7: on-demand GPU).

This is batch inference, not training: the models score and answer the test sets; no weights
change. One job brings every config up to ``--reruns`` finished runs with the same seed,
rerun-major, exactly like ``scripts/run_grid.sh`` on the laptop, and records the local git commit
as the code version.

One-time setup (your Modal account; the Starter plan includes $30 of credit a month and needs a
payment method on file):

    uv run --with modal==1.6.2 modal setup

Run from the repo root with every code change committed. ``--detach`` keeps the job running when
this terminal closes or the laptop sleeps:

    uv run --with modal==1.6.2 modal run --detach scripts/modal_eval.py
    uv run --with modal==1.6.2 modal run scripts/modal_eval.py --reruns 1 \
        --configs configs/eval/local_ll.yaml
    LLM_EVAL_GPU=L4 uv run --with modal==1.6.2 modal run --detach scripts/modal_eval.py

The GPU is an A10 unless ``LLM_EVAL_GPU`` names another Modal GPU type; keep one type for all
reruns. Each run is saved to the Modal volume ``llm-eval-runs`` as soon as it finishes. Finished
runs already there (same config, commit and GPU type, no failed cells) count, so the same command
run again after a timeout, a preemption or a closed terminal resumes the grid; at the end every
finished run is copied into ``data/processed/runs/``. To fetch the whole volume by hand instead
(the destination must be an existing directory without those runs):

    mkdir -p data/processed/modal
    uv run --with modal==1.6.2 modal volume get llm-eval-runs / data/processed/modal

Model and dataset downloads are cached in the volume ``llm-eval-hf-cache`` for later jobs.
"""

from __future__ import annotations

import json
import os
import subprocess
import traceback
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE = "/root/llm_eval_harness"
RUNS = Path("/runs")  # where the runs volume is mounted in the container
GPU = os.environ.get("LLM_EVAL_GPU", "A10").upper()  # names as in modal.com/docs/guide/gpu
TORCH = "torch==2.14.1"  # the version in uv.lock; PyPI's Linux wheel is the CUDA build
CODE = ["src", "configs", "scripts", "pyproject.toml", "requirements-cuda.txt"]
IGNORE = [
    ".git",
    ".venv",
    ".env",
    ".env.*",
    ".coverage",
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


def finished_runs(root: Path, config: str, git_sha: str, gpu_type: str) -> list[str]:
    """Runs of ``config`` under ``root`` at ``git_sha`` on Modal ``gpu_type``, no failed cells.

    Paths are relative to ``root``, oldest first (run ids start with their UTC start time).
    """
    from llm_eval.config import config_hash, load_config

    cfg = load_config(Path(config))
    want = config_hash(cfg)
    done = []
    for path in sorted(root.glob(f"{cfg.run_name}/*/run.json")):
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue  # cut short by a preemption: not finished
        if (
            meta.get("config_hash") == want
            and meta.get("git_sha") == git_sha
            and meta.get("modal_gpu") == gpu_type
            and not meta.get("failed_cells")
        ):
            done.append(str(path.parent.relative_to(root)))
    return done


def stamp(run_dir: Path, **fields: object) -> None:
    """Add launch facts the runner cannot see inside the container to ``run.json``."""
    path = run_dir / "run.json"
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:  # unreadable: finished_runs() will not count it either
        print(f"!!! {path}: {exc}", flush=True)
        return
    meta.update(fields)
    path.write_text(json.dumps(meta, indent=2, sort_keys=True, default=str), encoding="utf-8")


def commit(volume: modal.Volume) -> None:
    """Commit now; on failure rely on Modal's background commits and the commit at exit."""
    try:
        volume.commit()
    except Exception as exc:  # a failed commit must not stop the remaining runs
        print(f"!!! volume commit failed, continuing: {exc}", flush=True)


@app.function(
    image=image,
    gpu=GPU,
    timeout=6 * 3600,
    volumes={"/root/.cache/huggingface": hf_cache, str(RUNS): runs},
)
def run_grid(
    configs: list[str], reruns: int, git_sha: str, git_dirty: bool, gpu_type: str
) -> list[str]:
    """Bring each config up to ``reruns`` finished runs; return them, relative to the volume.

    Finished runs already in the volume count, so a second invocation, or Modal restarting this
    function on the same input after a preemption, resumes instead of starting over.
    ``gpu_type`` comes from the laptop: ``LLM_EVAL_GPU`` is not set inside the container.
    """
    import torch

    from llm_eval.cli import main as llm_eval

    os.environ["LLM_EVAL_GIT_SHA"] = git_sha
    gpu = torch.cuda.get_device_name(0)
    print(f"{gpu} · torch {torch.__version__} · CUDA {torch.version.cuda}")
    for i in range(1, reruns + 1):
        for config in configs:
            if len(finished_runs(RUNS, config, git_sha, gpu_type)) >= i:
                print(f"=== rerun {i}/{reruns}: {config} already in the volume", flush=True)
                continue
            before = {p.parent for p in RUNS.glob("*/*/run.json")}
            print(f"=== rerun {i}/{reruns}: {config}", flush=True)
            try:
                status = str(llm_eval(["run", "--config", config, "--output-dir", str(RUNS)]))
            except Exception:  # keep going: the other configs and reruns may still run
                traceback.print_exc()
                status = "crashed"
            new = sorted({p.parent for p in RUNS.glob("*/*/run.json")} - before)
            for run_dir in new:
                stamp(run_dir, modal_gpu=gpu_type, accelerator=gpu, git_dirty=git_dirty)
            commit(runs)
            commit(hf_cache)
            print(f"=== rerun {i}/{reruns}: {config} -> exit {status}, saved {new}", flush=True)
    finished = {c: finished_runs(RUNS, c, git_sha, gpu_type)[:reruns] for c in configs}
    if short := [c for c, done in finished.items() if len(done) < reruns]:
        print(f"!!! fewer than {reruns} finished runs (see failed_cells in run.json): {short}")
    return [rel for done in finished.values() for rel in done]


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
    # tracked and untracked changes to anything baked into the image that affects the runs
    dirty = bool(git("status", "--porcelain", "--", *CODE))
    if dirty and not allow_dirty:
        raise SystemExit(f"Commit changes under {CODE} first: every run records the commit.")
    names = [c.strip() for c in configs.split(",") if c.strip()]
    missing = [c for c in names if not (ROOT / c).is_file()]
    if missing:
        raise SystemExit(f"configs not found (paths relative to the repo root): {missing}")
    print(f"Modal {GPU}: {reruns} rerun(s) of {names} at commit {sha[:7]}")
    finished = run_grid.remote(names, reruns, sha, dirty, GPU)
    for rel in finished:
        target = ROOT / out / rel
        target.mkdir(parents=True, exist_ok=True)
        for name in ("examples.parquet", "run.json"):
            if (target / name).is_file():
                continue  # copied by an earlier invocation; finished runs never change
            part = target / f"{name}.part"
            with part.open("wb") as fh:
                for chunk in runs.read_file(f"{rel}/{name}"):
                    fh.write(chunk)
            part.replace(target / name)
        print(f"downloaded {os.path.relpath(target, ROOT)}")
    print(
        f"{len(finished)} run(s) copied. Next: move the MPS local_ll run to"
        " data/samples/mps/local_ll/, copy these runs into data/samples/runs/ and make analyze"
        " (README: Run the grid on a cloud GPU)."
    )
