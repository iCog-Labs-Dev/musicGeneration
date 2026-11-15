"""Benchmark matrix runner: expansion, resumable execution, manifest, aggregation.

Layout of an output directory::

    manifest.json     # config + every expected run id and its status
    runs/<run_id>.json  # one raw result per run (kept forever; never overwritten)
    results.jsonl     # all raw results, one per line
    aggregate.json    # per-cell summaries, CIs, paired ablation deltas
    report.md         # concise human-readable report

A failed, timed-out or crashed run is stored as a normal result.  Re-running the
same command resumes: finished runs are skipped.  A directory created from a
different config is refused rather than overwritten.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from aimusic.evaluation.config import BenchmarkConfig
from aimusic.evaluation.runner import reproducibility_info
from aimusic.evaluation.schema import (
    AGGREGATE_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
    STATUS_CRASHED,
    STATUS_TIMEOUT,
    RunSpec,
    new_result,
)
from aimusic.evaluation.stats import aggregate


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _run_path(out_dir: Path, spec: RunSpec) -> Path:
    return out_dir / "runs" / f"{spec.run_id}.json"


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def prepare_output_dir(
    config: BenchmarkConfig, specs: Sequence[RunSpec], out_dir: Path
) -> Dict[str, Any]:
    """Create/validate the output dir and write the manifest; refuse a foreign config."""
    manifest_path = out_dir / "manifest.json"
    existing = _load_json(manifest_path)
    if existing is not None and existing.get("config_hash") != config.config_hash:
        raise FileExistsError(
            f"{out_dir} already holds results for a different config "
            f"({existing.get('config_name')!r}, hash {str(existing.get('config_hash'))[:12]}). "
            "Choose another --out directory."
        )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "config_name": config.name,
        "config_version": config.config_version,
        "config_hash": config.config_hash,
        "config": dict(config.raw),
        "created_with": reproducibility_info(),
        "expected_runs": [{"run_id": s.run_id, "spec": s.to_dict()} for s in specs],
    }
    _atomic_write_json(manifest_path, manifest)
    return manifest


def execute_run(spec: RunSpec, out_dir: Path, timeout_s: float) -> Dict[str, Any]:
    """Run one spec in a fresh subprocess and persist its result."""
    out_dir = out_dir.resolve()  # the worker runs with cwd=out_dir, so paths must be absolute
    run_dir = out_dir / "runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    spec_file = run_dir / f".{spec.run_id}.spec.json"
    out_file = run_dir / f".{spec.run_id}.out.json"
    _atomic_write_json(spec_file, spec.to_dict())
    started = time.perf_counter()
    result: Optional[Dict[str, Any]] = None
    failure: Optional[Dict[str, Any]] = None
    status = STATUS_CRASHED
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "aimusic.evaluation.worker", str(spec_file), str(out_file)],
            cwd=out_dir,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        result = _load_json(out_file)
        if result is None:
            failure = {
                "kind": "worker_crashed",
                "stage": "subprocess",
                "message": f"worker exited with code {proc.returncode} without writing a result",
                "traceback_tail": (proc.stderr or "")[-1500:],
            }
    except subprocess.TimeoutExpired:
        status = STATUS_TIMEOUT
        failure = {
            "kind": "timeout",
            "stage": "subprocess",
            "message": f"run exceeded timeout_s={timeout_s}",
            "traceback_tail": "",
        }
    if result is None:
        result = new_result(spec)
        result["status"] = status
        result["failure"] = failure
        result["runtime"] = {"wall_s": time.perf_counter() - started}
        result["reproducibility"] = reproducibility_info()
    _atomic_write_json(_run_path(out_dir, spec), result)
    for tmp in (spec_file, out_file):
        try:
            tmp.unlink()
        except OSError:
            pass
    return result


def run_matrix(
    config: BenchmarkConfig,
    all_specs: Sequence[RunSpec],
    selected: Sequence[RunSpec],
    out_dir: Path,
    *,
    workers: int = 1,
    retry_failed: bool = False,
    progress: Callable[[str], None] = print,
) -> List[Dict[str, Any]]:
    """Execute ``selected`` runs (skipping finished ones) and return all results on disk."""
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    prepare_output_dir(config, all_specs, out_dir)

    pending: List[RunSpec] = []
    for spec in selected:
        prior = _load_json(_run_path(out_dir, spec))
        if prior is not None and (prior.get("status") == "ok" or not retry_failed):
            continue
        pending.append(spec)
    skipped = len(selected) - len(pending)
    progress(f"{len(selected)} selected runs: {skipped} already finished, {len(pending)} to run")

    def work(item: tuple[int, RunSpec]) -> None:
        idx, spec = item
        res = execute_run(spec, out_dir, config.timeout_s)
        wall = (res.get("runtime") or {}).get("wall_s", float("nan"))
        progress(f"[{idx + 1}/{len(pending)}] {spec.run_id}: {res['status']} ({wall:.1f}s)")

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(work, enumerate(pending)))
    return load_results(out_dir)


def load_results(out_dir: Path) -> List[Dict[str, Any]]:
    manifest = _load_json(out_dir / "manifest.json")
    if manifest is None:
        raise FileNotFoundError(f"No manifest.json in {out_dir}.")
    results = []
    for entry in manifest["expected_runs"]:
        res = _load_json(out_dir / "runs" / f"{entry['run_id']}.json")
        if res is not None:
            results.append(res)
    return results


def write_outputs(out_dir: Path) -> Dict[str, Any]:
    """Write results.jsonl, aggregate.json and report.md from the runs on disk."""
    from aimusic.evaluation.report import render_report

    manifest = _load_json(out_dir / "manifest.json")
    if manifest is None:
        raise FileNotFoundError(f"No manifest.json in {out_dir}.")
    results = load_results(out_dir)
    with open(out_dir / "results.jsonl", "w", encoding="utf-8") as handle:
        for res in sorted(results, key=lambda r: r["run_id"]):
            handle.write(json.dumps(res, sort_keys=True) + "\n")
    agg = aggregate(results) if results else {"cells": {}, "paired_vs_control": {}}
    expected = [e["run_id"] for e in manifest["expected_runs"]]
    done = {r["run_id"] for r in results}
    agg.update(
        {
            "schema_version": AGGREGATE_SCHEMA_VERSION,
            "config_name": manifest["config_name"],
            "config_version": manifest["config_version"],
            "config_hash": manifest["config_hash"],
            "expected_runs": len(expected),
            "completed_runs": len(done),
            "missing_run_ids": [r for r in expected if r not in done],
        }
    )
    _atomic_write_json(out_dir / "aggregate.json", agg)
    (out_dir / "report.md").write_text(render_report(manifest, results, agg), encoding="utf-8")
    return agg
