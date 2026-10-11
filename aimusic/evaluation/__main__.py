"""Command line: ``python -m aimusic.evaluation {run,report,list} ...``."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from aimusic.evaluation.config import expand_matrix, load_config, select_specs
from aimusic.evaluation.matrix import run_matrix, write_outputs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m aimusic.evaluation", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run (or resume) the benchmark matrix, then aggregate and report.")
    run.add_argument("--config", required=True, help="Versioned benchmark config JSON.")
    run.add_argument("--out", required=True, help="Output directory (resumable; refuses a different config).")
    run.add_argument("--workers", type=int, default=1, help="Parallel worker processes.")
    run.add_argument("--conditions", nargs="*", help="Subset of conditions.")
    run.add_argument("--horizons", nargs="*", type=int, help="Subset of horizons.")
    run.add_argument("--edos", nargs="*", type=int, help="Subset of EDOs.")
    run.add_argument("--seeds", nargs="*", type=int, help="Subset of seeds.")
    run.add_argument("--retry-failed", action="store_true", help="Re-run runs that previously failed.")
    run.add_argument("--dry-run", action="store_true", help="List the selected runs and exit.")

    rep = sub.add_parser("report", help="Rebuild aggregate.json and report.md from an output directory.")
    rep.add_argument("--out", required=True)

    lst = sub.add_parser("list", help="Print the expanded run matrix for a config.")
    lst.add_argument("--config", required=True)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "report":
        agg = write_outputs(Path(args.out))
        print(f"wrote report for {agg['completed_runs']}/{agg['expected_runs']} runs to {args.out}")
        return 0

    config = load_config(args.config)
    specs = expand_matrix(config)
    if args.command == "list":
        for spec in specs:
            print(spec.run_id)
        print(f"{len(specs)} runs", file=sys.stderr)
        return 0

    selected = select_specs(
        specs, conditions=args.conditions, horizons=args.horizons, edos=args.edos, seeds=args.seeds
    )
    if args.dry_run:
        for spec in selected:
            print(spec.run_id)
        print(f"{len(selected)} of {len(specs)} runs selected", file=sys.stderr)
        return 0
    out_dir = Path(args.out)
    run_matrix(
        config, specs, selected, out_dir,
        workers=args.workers, retry_failed=args.retry_failed,
        progress=lambda msg: print(msg, flush=True),
    )
    agg = write_outputs(out_dir)
    print(f"done: {agg['completed_runs']}/{agg['expected_runs']} runs; report at {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
