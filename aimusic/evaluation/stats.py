"""Repeated-seed statistics for the MUS-28 evaluation suite (stdlib only).

Seeds are replicates *within* a condition cell.  For each numeric metric we
report mean, sample standard deviation, a two-sided 95% Student-t confidence
interval for the mean, and robust summaries (median, 5th/95th percentile).
Failure rates use a Wilson interval.  Ablation comparisons are *paired* on seed
because every condition is run with the same seeds.

The 95% t-interval assumes roughly symmetric metrics; heavy-tailed metrics
(runtime) should be read together with the median and percentile range.
"""
from __future__ import annotations

import math
import statistics
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

AGGREGATE_METRIC_SECTIONS = ("graph", "sb", "path", "tension", "gttm")

# Two-sided 95% critical values of Student's t for df = 1..30.
_T95 = (
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
)


def t_critical_95(df: int) -> float:
    if df < 1:
        raise ValueError("df must be >= 1.")
    if df <= 30:
        return _T95[df - 1]
    if df <= 40:
        return 2.021
    if df <= 60:
        return 2.000
    if df <= 120:
        return 1.980
    return 1.960


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, ``q`` in [0, 100]."""
    if not sorted_values:
        raise ValueError("percentile of empty sequence.")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    frac = pos - lo
    return float(sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac)


def summarize(values: Sequence[float]) -> Dict[str, Any]:
    vals = sorted(float(v) for v in values)
    n = len(vals)
    if n == 0:
        return {"n": 0}
    mean = statistics.fmean(vals)
    out: Dict[str, Any] = {
        "n": n,
        "mean": mean,
        "median": percentile(vals, 50),
        "p05": percentile(vals, 5),
        "p95": percentile(vals, 95),
        "min": vals[0],
        "max": vals[-1],
        "sd": None,
        "ci95_low": None,
        "ci95_high": None,
    }
    if n >= 2:
        sd = statistics.stdev(vals)
        half = t_critical_95(n - 1) * sd / math.sqrt(n)
        out.update({"sd": sd, "ci95_low": mean - half, "ci95_high": mean + half})
    return out


def wilson_interval(successes: int, n: int, z: float = 1.959964) -> Tuple[float, float]:
    """95% Wilson score interval for a proportion."""
    if n <= 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def paired_difference(treatment: Mapping[int, float], control: Mapping[int, float]) -> Dict[str, Any]:
    """Mean paired difference (treatment - control) over seeds present in both."""
    seeds = sorted(set(treatment) & set(control))
    diffs = [treatment[s] - control[s] for s in seeds]
    summary = summarize(diffs)
    summary["seeds"] = seeds
    summary["ci_excludes_zero"] = (
        None
        if summary.get("ci95_low") is None
        else bool(summary["ci95_low"] > 0 or summary["ci95_high"] < 0)
    )
    return summary


def _numeric(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)) and math.isfinite(value):
        return float(value)
    return None


def flatten_metrics(result: Mapping[str, Any]) -> Dict[str, float]:
    """Numeric leaves of a result: ``section.name`` -> value (+ ``runtime.*``)."""
    flat: Dict[str, float] = {}
    metrics = result.get("metrics", {})
    for section in AGGREGATE_METRIC_SECTIONS:
        for name, value in (metrics.get(section) or {}).items():
            if isinstance(value, dict):
                for sub, sub_value in value.items():
                    num = _numeric(sub_value)
                    if num is not None:
                        flat[f"{section}.{name}.{sub}"] = num
                continue
            num = _numeric(value)
            if num is not None:
                flat[f"{section}.{name}"] = num
    runtime = result.get("runtime") or {}
    for name in ("wall_s", "cpu_s", "peak_rss_mb", "generate_wall_s"):
        num = _numeric(runtime.get(name))
        if num is not None:
            flat[f"runtime.{name}"] = num
    peak, base = _numeric(runtime.get("peak_rss_mb")), _numeric(runtime.get("baseline_rss_mb"))
    if peak is not None and base is not None:
        flat["runtime.rss_growth_mb"] = peak - base  # memory attributable to the run itself
    return flat


def _group_key(result: Mapping[str, Any]) -> str:
    from aimusic.evaluation.schema import RunSpec

    return RunSpec.from_dict(dict(result["spec"])).group_key


def _comparison_key(spec: Mapping[str, Any]) -> Tuple[Any, ...]:
    """Cells are comparable iff everything except the condition matches."""
    return (
        spec["horizon"], spec["edo"], spec["path_mode"],
        spec["k_max"], spec["d_max"], spec["proposal_budget"],
    )


def aggregate(results: Sequence[Mapping[str, Any]], control: str = "full") -> Dict[str, Any]:
    """Aggregate raw run results into per-cell summaries and paired ablation deltas."""
    cells: Dict[str, List[Mapping[str, Any]]] = {}
    for res in results:
        cells.setdefault(_group_key(res), []).append(res)

    out_cells: Dict[str, Any] = {}
    per_seed: Dict[str, Dict[str, Dict[int, float]]] = {}
    for key in sorted(cells):
        runs = sorted(cells[key], key=lambda r: r["spec"]["seed"])
        ok = [r for r in runs if r["status"] == "ok"]
        kinds: Dict[str, int] = {}
        for r in runs:
            if r["status"] != "ok":
                kind = (r.get("failure") or {}).get("kind") or r["status"]
                kinds[kind] = kinds.get(kind, 0) + 1
        values: Dict[str, Dict[int, float]] = {}
        for r in ok:
            for name, num in flatten_metrics(r).items():
                values.setdefault(name, {})[int(r["spec"]["seed"])] = num
        n_fail = len(runs) - len(ok)
        low, high = wilson_interval(n_fail, len(runs))
        spec = dict(runs[0]["spec"])
        out_cells[key] = {
            "condition": spec["condition"],
            "horizon": spec["horizon"],
            "edo": spec["edo"],
            "path_mode": spec["path_mode"],
            "k_max": spec["k_max"],
            "d_max": spec["d_max"],
            "proposal_budget": spec["proposal_budget"],
            "n_runs": len(runs),
            "n_ok": len(ok),
            "failure_rate": n_fail / len(runs),
            "failure_rate_ci95": [low, high],
            "failure_kinds": kinds,
            "seeds_ok": sorted(int(r["spec"]["seed"]) for r in ok),
            "metrics": {name: summarize(list(v.values())) for name, v in sorted(values.items())},
        }
        per_seed[key] = values

    paired: Dict[str, Any] = {}
    controls: Dict[Tuple[Any, ...], str] = {}
    for cell_key in cells:
        cell_spec = _spec_of(cells[cell_key])
        if cell_spec["condition"] == control:
            controls[_comparison_key(cell_spec)] = cell_key
    for key in sorted(cells):
        treat_spec = _spec_of(cells[key])
        if treat_spec["condition"] == control:
            continue
        ctrl_key = controls.get(_comparison_key(treat_spec))
        if ctrl_key is None:
            continue
        deltas: Dict[str, Any] = {}
        for name, treat_vals in per_seed[key].items():
            ctrl_vals = per_seed[ctrl_key].get(name)
            if ctrl_vals:
                deltas[name] = paired_difference(treat_vals, ctrl_vals)
        if deltas:  # an all-failed cell has nothing to pair
            paired[key] = {"control": ctrl_key, "metrics": deltas}
    return {"cells": out_cells, "paired_vs_control": paired, "control_condition": control}


def _spec_of(runs: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    return runs[0]["spec"]
