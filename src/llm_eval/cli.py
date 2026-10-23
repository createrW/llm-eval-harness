"""``llm-eval`` command line: ``run`` a config, ``report`` a run directory (design §6, §9).

``run`` validates the YAML with the CLI overrides applied before validation (§8.1), runs every cell,
prints ``run dir: <absolute path>`` and the summary table, and exits 1 when a cell failed or (unless
``--allow-errors``) any example row has ``status == "error"``. ``report`` re-prints the summary
table stored in a run directory's ``run.json``. argparse only; logging goes to stderr at INFO.
This module never imports torch: the model adapter is loaded lazily by the provider registry.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from llm_eval.config import load_config
from llm_eval.evaluation.report import SUMMARY_COLUMNS, format_summary
from llm_eval.evaluation.results import read_run
from llm_eval.runner import run_config

logger = logging.getLogger("llm_eval")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-eval",
        description="Evaluate LLMs on benchmark tasks with bootstrap confidence intervals.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run every cell of an eval config")
    run.add_argument("--config", required=True, type=Path, metavar="PATH", help="eval config YAML")
    run.add_argument(
        "--output-dir", metavar="DIR", help="override output_dir (relative = project root)"
    )
    run.add_argument(
        "--n-examples", type=int, metavar="N", help="override the top-level n_examples"
    )
    run.add_argument("--seed", type=int, metavar="S", help="override seed")
    run.add_argument(
        "--allow-errors",
        action="store_true",
        help='exit 0 even when example rows have status == "error" (failed cells still exit 1)',
    )

    report = commands.add_parser("report", help="print the summary table of a run directory")
    report.add_argument(
        "--run-dir", required=True, type=Path, metavar="DIR", help="a run directory"
    )

    analyze = commands.add_parser(
        "analyze", help="cross-cell tables, paired tests, verdicts and test-retest over runs"
    )
    analyze.add_argument(
        "--runs-root", type=Path, metavar="DIR", help="analyse every <DIR>/<run_name>/<run_id>/"
    )
    analyze.add_argument(
        "--run-dir", type=Path, action="append", default=[], metavar="DIR", help="a run (repeat)"
    )
    analyze.add_argument("--hypotheses", type=Path, metavar="PATH", help="hypotheses YAML")
    analyze.add_argument("--out", type=Path, metavar="PATH", help="write the markdown here")
    analyze.add_argument("--json", type=Path, metavar="PATH", help="write the tables as JSON")
    analyze.add_argument(
        "--readme", type=Path, metavar="PATH", help="rewrite the results block of this README"
    )
    analyze.add_argument("--resamples", type=int, default=1000, metavar="B")

    track = commands.add_parser("track", help="log run directories to MLflow (idempotent)")
    track.add_argument("--runs-root", type=Path, metavar="DIR", help="log every run under DIR")
    track.add_argument("--run-dir", type=Path, action="append", default=[], metavar="DIR")
    track.add_argument(
        "--analysis", type=Path, metavar="MD", help="also log this results file and its .json"
    )
    track.add_argument("--tracking-uri", default="sqlite:///mlflow.db", metavar="URI")
    return parser


def _run(args: argparse.Namespace) -> int:
    overrides = {"output_dir": args.output_dir, "n_examples": args.n_examples, "seed": args.seed}
    cfg = load_config(args.config, overrides=overrides)
    result = run_config(cfg)
    print(f"run dir: {result.run_dir.absolute()}")
    print(format_summary(result.summary))
    code = 0
    if result.failed_cells:
        logger.error("failed cells: %s", ", ".join(result.failed_cells))
        code = 1
    n_error = int(result.summary["n_error"].sum())
    if n_error and not args.allow_errors:
        logger.error('%d example rows have status == "error" (pass --allow-errors)', n_error)
        code = 1
    return code


def _report(args: argparse.Namespace) -> int:
    _, meta = read_run(args.run_dir)
    print(format_summary(pd.DataFrame(meta["summary"], columns=SUMMARY_COLUMNS)))
    return 0


def _analyze(args: argparse.Namespace) -> int:
    from llm_eval.evaluation import analysis, analysis_report, variance

    dirs = list(args.run_dir)
    if args.runs_root:
        dirs += analysis.discover_runs(args.runs_root)
    if not dirs:
        logger.error("no runs: pass --runs-root or --run-dir")
        return 2
    runs = analysis.load_runs(dirs)
    families = analysis.analyze(runs, n_resamples=args.resamples)
    for fa in families:
        fa.variance = variance.decompose(fa)
    retest = analysis.retest_agreement(runs, n_resamples=args.resamples)
    hypotheses = analysis_report.load_hypotheses(args.hypotheses) if args.hypotheses else None
    text = analysis_report.render(families, retest=retest, hypotheses=hypotheses)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(analysis.to_json(families, retest), encoding="utf-8")
    if args.readme:
        block = analysis_report.readme_block(families, retest=retest, hypotheses=hypotheses)
        analysis_report.replace_readme_block(args.readme, block)
        print(f"updated the results block of {args.readme}")
    return 0


def _track(args: argparse.Namespace) -> int:
    from llm_eval import tracking
    from llm_eval.evaluation.analysis import discover_runs

    dirs = list(args.run_dir) + (discover_runs(args.runs_root) if args.runs_root else [])
    for run_dir in dirs:
        run_id = tracking.log_run(run_dir, tracking_uri=args.tracking_uri)
        print(f"{'logged' if run_id else 'already logged'}: {run_dir}")
    if args.analysis:
        tables = args.analysis.with_suffix(".json")
        run_id = tracking.log_analysis(
            args.analysis, tables, label=args.analysis.stem, tracking_uri=args.tracking_uri
        )
        print(f"logged analysis {args.analysis} as MLflow run {run_id}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of ``llm-eval``; returns the process exit code."""
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _build_parser().parse_args(argv)
    if args.command == "run":
        return _run(args)
    if args.command == "analyze":
        return _analyze(args)
    if args.command == "track":
        return _track(args)
    return _report(args)


if __name__ == "__main__":
    raise SystemExit(main())
