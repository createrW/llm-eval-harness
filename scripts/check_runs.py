"""Check finished runs before they are committed, and copy the complete ones into the samples.

    uv run python scripts/check_runs.py data/processed/runs/local_ll data/processed/runs/local_gen
    uv run python scripts/check_runs.py data/processed/runs/local_ll data/processed/runs/local_gen \
        --copy-to data/samples/runs

Each path is a ``<run_name>/`` directory of runs or a single run directory. One line per run:
complete or not (readable, no failed cells, every configured model loaded, no error rows), config
hash, commit, device/dtype, the Modal GPU when there is one, and wall time. ``make analyze``
expects ``--keep`` (default 3) complete runs per config on one hardware set: the check exits 1
otherwise. ``--copy-to`` copies the oldest ``--keep`` complete runs of each run name to
``<root>/<run_name>/<run_id>/`` (all or nothing); pass run directories explicitly to pick others.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

RUN_FILE = "run.json"
EXAMPLES_FILE = "examples.parquet"


def run_dirs(paths: list[Path]) -> list[Path]:
    found: list[Path] = []
    for path in paths:
        if (path / RUN_FILE).is_file():
            found.append(path)
        else:
            found.extend(sorted(p.parent for p in path.glob(f"*/{RUN_FILE}")))
    return found


def inspect(run_dir: Path) -> dict:
    """Facts about one run; ``problems`` is empty when the run is complete."""
    try:
        meta = json.loads((run_dir / RUN_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"dir": run_dir, "name": run_dir.parent.name, "problems": [f"run.json: {exc}"]}
    models = meta.get("models") or {}
    wanted = [m["name"] for m in meta.get("config", {}).get("models", [])]
    problems = []
    if meta.get("failed_cells"):
        problems.append(f"{len(meta['failed_cells'])} failed cell(s)")
    if missing := [name for name in wanted if name not in models]:
        problems.append(f"not loaded: {', '.join(missing)}")
    try:
        status = pq.read_table(run_dir / EXAMPLES_FILE, columns=["status"]).column("status")
        errors = pc.sum(pc.equal(status, "error")).as_py() or 0
    except (OSError, ValueError) as exc:
        problems.append(f"examples.parquet: {exc}")
        errors = 0
    if errors:
        problems.append(f"{errors} error row(s)")
    try:
        start, end = (datetime.fromisoformat(meta[k]) for k in ("started_at", "finished_at"))
        minutes = f"{(end - start).total_seconds() / 60:.1f} min"
    except (KeyError, TypeError, ValueError):
        minutes = "?"
    hardware = tuple(  # what analysis.rerun_groups() groups reruns by
        sorted((name, str(i.get("device")), str(i.get("dtype"))) for name, i in models.items())
    )
    return {
        "dir": run_dir,
        "name": run_dir.parent.name,  # the <run_name>/<run_id> layout make analyze reads
        "config": meta.get("config_hash"),
        "sha": str(meta.get("git_sha"))[:7],
        "hardware": hardware,
        "gpu": meta.get("accelerator") or meta.get("modal_gpu") or "",
        "minutes": minutes,
        "problems": problems,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="+", type=Path, help="<run_name>/ or run directories")
    parser.add_argument("--keep", type=int, default=3, help="complete runs needed per config")
    parser.add_argument("--copy-to", type=Path, help="runs root to copy them into")
    args = parser.parse_args(argv)

    runs = [inspect(d) for d in run_dirs(args.paths)]
    if not runs:
        print(f"no runs under {', '.join(map(str, args.paths))}")
        return 1
    for r in runs:
        devices = sorted({f"{dev}/{dtype}" for _, dev, dtype in r.get("hardware", ())}) or ["-"]
        state = "complete" if not r["problems"] else "INCOMPLETE: " + "; ".join(r["problems"])
        print(
            f"{r['name']}/{r['dir'].name}  {r.get('config')}  {r.get('sha')}  {','.join(devices)}"
            f"  {r.get('gpu', '')}  {r.get('minutes', '?')}  {state}"
        )
    chosen: list[dict] = []
    ok = True
    for name in sorted({r["name"] for r in runs}):
        complete = [r for r in runs if r["name"] == name and not r["problems"]]
        extra, complete = complete[args.keep :], complete[: args.keep]
        groups = {(r["config"], r["hardware"], r["gpu"]) for r in complete}
        if len(complete) < args.keep or len(groups) > 1:
            ok = False
            why = "configs, hardware or GPUs differ" if len(groups) > 1 else "too few"
            print(f"!!! {name}: {len(complete)} complete run(s), {args.keep} needed ({why})")
        else:
            left = f"; {len(extra)} later complete run(s) left out" if extra else ""
            print(f"{name}: {len(complete)} complete run(s), config {complete[0]['config']}{left}")
        chosen.extend(complete)
    if not ok:
        return 1
    if args.copy_to:
        for r in chosen:
            target = args.copy_to / r["name"] / r["dir"].name
            shutil.copytree(r["dir"], target, dirs_exist_ok=True)
            print(f"copied {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
