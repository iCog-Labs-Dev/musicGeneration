"""Subprocess entry point: evaluate one RunSpec and write its result JSON.

Running each run in its own process gives (a) a per-run peak-RSS reading that is
not contaminated by earlier runs, (b) a hard timeout, and (c) crash isolation:
an out-of-memory kill is recorded as a ``crashed`` result instead of taking the
whole matrix down.
"""
from __future__ import annotations

import json
import sys

from aimusic.evaluation.runner import evaluate_run
from aimusic.evaluation.schema import RunSpec


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: python -m aimusic.evaluation.worker SPEC_JSON OUT_JSON", file=sys.stderr)
        return 2
    spec_path, out_path = argv
    with open(spec_path, "r", encoding="utf-8") as handle:
        spec = RunSpec.from_dict(json.load(handle))
    result = evaluate_run(spec)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
