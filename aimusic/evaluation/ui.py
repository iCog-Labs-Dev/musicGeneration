"""Evaluation dashboard for the MUS-28 benchmark suite (data layer, charts, job runner, Gradio page).

This module is strictly a *viewer and launcher*:

* Benchmarks are executed by the existing CLI (``python -m aimusic.evaluation ...``) in a
  subprocess, so UI and command-line results are identical.  Nothing here re-implements
  benchmark execution, statistics or reporting.
* Charts are built from the structured files the suite already writes: ``manifest.json``,
  ``runs/*.json`` / ``results.jsonl`` (run level) and ``aggregate.json`` (cell level, with the
  95% confidence intervals computed by :mod:`aimusic.evaluation.stats`).  The Markdown report
  is shown as an audit view only.
* When ``aggregate.json`` is missing or stale (e.g. a benchmark that is still running), the
  aggregate is recomputed with the same :func:`aimusic.evaluation.stats.aggregate` the CLI uses.

Layering: everything up to :func:`build_evaluation_page` is importable without Gradio or Plotly.
The ``*_chart_data`` functions are pure and return plain data; the ``make_*_chart`` functions
turn that data into Plotly figures (Plotly is imported lazily).
"""
from __future__ import annotations

import html
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, TypeGuard

from aimusic.evaluation.config import CONDITIONS, BenchmarkConfig, expand_matrix, load_config, select_specs
from aimusic.evaluation.report import CONDITION_ORDER
from aimusic.evaluation.schema import (
    STATUS_CRASHED,
    STATUS_FAILED,
    STATUS_OK,
    STATUS_TIMEOUT,
    RunSpec,
)
from aimusic.evaluation.stats import aggregate

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config" / "benchmarks"
DEFAULT_OUT_ROOT = REPO_ROOT / "outputs" / "benchmarks"
LOG_NAME = "ui_run.log"

STATUS_ORDER = (STATUS_OK, STATUS_FAILED, STATUS_TIMEOUT, STATUS_CRASHED)
STATUS_NOT_RUN = "not_run"

# Categorical slots, fixed order (never cycled); validated adjacent-pair CVD-safe palette.
CONDITION_COLORS = {
    "full": "#2a78d6",
    "no_gttm": "#eb6834",
    "no_prior": "#1baf7a",
    "no_sb": "#eda100",
    "trained_prior": "#e87ba4",
}
EDO_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#6250d6")
STATUS_COLORS = {
    STATUS_OK: "#008300",
    STATUS_FAILED: "#e34948",
    STATUS_TIMEOUT: "#eda100",
    STATUS_CRASHED: "#6250d6",
    STATUS_NOT_RUN: "#9a9a94",
}
INK = "#8a8a85"  # mid-grey text/grid: readable on light and dark Gradio themes

# metric key -> (label, unit, direction)   direction: "lower" / "higher" is better, None = neutral
METRIC_INFO: Dict[str, Tuple[str, str, Optional[str]]] = {
    "runtime.wall_s": ("Runtime", "s", "lower"),
    "runtime.cpu_s": ("CPU time", "s", "lower"),
    "runtime.peak_rss_mb": ("Peak RSS (process)", "MB", None),
    "runtime.rss_growth_mb": ("RSS growth (run only)", "MB", None),
    "graph.layer_width_mean": ("Mean layer width", "states", None),
    "tension.mean_absolute_error": ("Tension MAE", "", "lower"),
    "tension.shape_correlation": ("Tension correlation", "", "higher"),
    "path.distinct_state_fraction": ("Distinct-state fraction", "", None),
    "path.chord_change_rate": ("Chord-change rate", "", None),
    "gttm.mean_gttm_score": ("Mean GTTM score / edge", "", None),
}
RUNTIME_CHART_METRICS = (
    "runtime.wall_s", "runtime.cpu_s", "runtime.peak_rss_mb", "runtime.rss_growth_mb",
)
ABLATION_METRICS = (
    "tension.mean_absolute_error", "tension.shape_correlation", "runtime.wall_s",
    "path.distinct_state_fraction", "path.chord_change_rate",
)
SCALING_METRICS = ("runtime.wall_s", "runtime.peak_rss_mb", "runtime.rss_growth_mb")
EDO_METRICS = (
    "runtime.wall_s", "graph.layer_width_mean", "tension.mean_absolute_error",
    "path.distinct_state_fraction", "path.chord_change_rate",
)


def metric_label(key: str) -> str:
    info = METRIC_INFO.get(key)
    if not info:
        return key
    return f"{info[0]} ({info[1]})" if info[1] else info[0]


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
@dataclass
class EvaluationDataset:
    """Everything the dashboard needs from one benchmark output directory."""

    out_dir: Path
    manifest: Dict[str, Any]
    results: List[Dict[str, Any]]
    aggregate: Dict[str, Any]
    aggregate_source: str  # "aggregate.json" | "recomputed"
    report_text: Optional[str] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def expected_ids(self) -> List[str]:
        return [e["run_id"] for e in self.manifest.get("expected_runs", [])]

    @property
    def cells(self) -> Dict[str, Any]:
        return self.aggregate.get("cells", {})

    @property
    def paired(self) -> Dict[str, Any]:
        return self.aggregate.get("paired_vs_control", {})


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _read_run_files(out_dir: Path, run_ids: Sequence[str], warnings: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    runs_dir = out_dir / "runs"
    for run_id in run_ids:
        path = runs_dir / f"{run_id}.json"
        if not path.exists():
            continue
        res = _read_json(path)
        if res is None or "spec" not in res or "status" not in res:
            warnings.append(f"Skipped unreadable run file {path.name}.")
            continue
        out.append(res)
    return out


def _read_results_jsonl(path: Path, warnings: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return out
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            res = json.loads(line)
        except ValueError:
            warnings.append(f"Skipped malformed line {number} in results.jsonl.")
            continue
        if not isinstance(res, dict) or "spec" not in res or "status" not in res or "run_id" not in res:
            warnings.append(f"Skipped incomplete record on line {number} of results.jsonl.")
            continue
        out.append(res)
    return out


def load_benchmark_output(out_dir: Path | str) -> EvaluationDataset:
    """Load a benchmark directory, tolerating partial, failed and missing pieces.

    Raises ``FileNotFoundError`` only when the directory has no ``manifest.json`` (nothing to show).
    Run-level data comes from ``runs/*.json`` (always current while a benchmark is running) and falls
    back to ``results.jsonl``.
    """
    out = Path(out_dir)
    if not out.is_dir():
        raise FileNotFoundError(f"{out} is not a directory.")
    manifest = _read_json(out / "manifest.json")
    if manifest is None:
        raise FileNotFoundError(f"No readable manifest.json in {out}; is this a benchmark output directory?")
    warnings: List[str] = []
    expected = [e["run_id"] for e in manifest.get("expected_runs", []) if "run_id" in e]
    results = _read_run_files(out, expected, warnings)
    if not results and (out / "results.jsonl").exists():
        results = _read_results_jsonl(out / "results.jsonl", warnings)
        if results:
            warnings.append("runs/ is empty; run-level data was read from results.jsonl.")
    results.sort(key=lambda r: r["run_id"])

    agg = _read_json(out / "aggregate.json")
    source = "aggregate.json"
    stale = (
        agg is None
        or agg.get("config_hash") != manifest.get("config_hash")
        or agg.get("completed_runs") != len({r["run_id"] for r in results})
    )
    if stale:
        if agg is not None:
            warnings.append("aggregate.json is out of date for the runs on disk; recomputed live (same code as the CLI).")
        elif results:
            warnings.append("aggregate.json not found; computed live from the run results.")
        agg = aggregate(results) if results else {"cells": {}, "paired_vs_control": {}, "control_condition": "full"}
        source = "recomputed"
    assert agg is not None
    report_path = out / "report.md"
    report = None
    if report_path.exists():
        try:
            report = report_path.read_text(encoding="utf-8")
        except OSError:
            warnings.append("report.md exists but could not be read.")
    return EvaluationDataset(out, manifest, results, agg, source, report, warnings)


def discover_benchmark_configs(config_dir: Path = CONFIG_DIR) -> List[str]:
    """Names (file names) of the versioned benchmark configs, sorted."""
    if not config_dir.is_dir():
        return []
    return sorted(p.name for p in config_dir.glob("*.json"))


def config_path_for(name_or_path: str, config_dir: Path = CONFIG_DIR) -> Path:
    candidate = Path(name_or_path)
    if candidate.is_absolute() or candidate.exists():
        return candidate.resolve()
    return (config_dir / name_or_path).resolve()


def discover_output_dirs(root: Path = DEFAULT_OUT_ROOT) -> List[str]:
    if not root.is_dir():
        return []
    return sorted(str(p) for p in root.iterdir() if (p / "manifest.json").exists())


# --------------------------------------------------------------------------------------
# Summary / overview cards
# --------------------------------------------------------------------------------------
@dataclass
class EvaluationSummary:
    config_name: str
    config_version: str
    expected: int
    completed: int
    status_counts: Dict[str, int]  # raw statuses + "not_run"
    mean_wall_s: Optional[float]
    longest_horizon: Optional[int]
    horizons: List[int]
    edos: List[int]
    conditions: List[str]
    ci_enabled: bool
    state: str  # "empty" | "partial" | "complete"

    @property
    def ok(self) -> int:
        return self.status_counts.get(STATUS_OK, 0)

    @property
    def failed(self) -> int:
        return self.status_counts.get(STATUS_FAILED, 0)

    @property
    def timeouts(self) -> int:
        return self.status_counts.get(STATUS_TIMEOUT, 0)

    @property
    def crashed(self) -> int:
        return self.status_counts.get(STATUS_CRASHED, 0)

    @property
    def not_run(self) -> int:
        return self.status_counts.get(STATUS_NOT_RUN, 0)


def _condition_rank(name: str) -> Tuple[int, str]:
    order = {c: i for i, c in enumerate(CONDITION_ORDER)}
    return (order.get(name, len(order)), name)


def _sorted_conditions(names: Sequence[str]) -> List[str]:
    return sorted(set(names), key=_condition_rank)


def summarize_results(dataset: EvaluationDataset) -> EvaluationSummary:
    results = dataset.results
    expected = len(dataset.expected_ids)
    counts: Counter[str] = Counter(r["status"] for r in results)
    # Raw statuses only; anything unknown is surfaced rather than silently dropped.
    status_counts = {s: counts.get(s, 0) for s in STATUS_ORDER}
    for extra, n in counts.items():
        if extra not in status_counts:
            status_counts[extra] = n
    completed = len(results)
    status_counts[STATUS_NOT_RUN] = max(expected - completed, 0)
    walls = [
        float(r["runtime"]["wall_s"])
        for r in results
        if isinstance(r.get("runtime"), dict) and _finite(r["runtime"].get("wall_s"))
    ]
    specs = [e.get("spec", {}) for e in dataset.manifest.get("expected_runs", [])] or [r["spec"] for r in results]
    horizons = sorted({int(s["horizon"]) for s in specs if "horizon" in s})
    edos = sorted({int(s["edo"]) for s in specs if "edo" in s})
    conditions = _sorted_conditions([s["condition"] for s in specs if "condition" in s])
    ci = any(
        m.get("ci95_low") is not None
        for c in dataset.cells.values()
        for m in c.get("metrics", {}).values()
    )
    if completed == 0:
        state = "empty"
    elif completed < expected:
        state = "partial"
    else:
        state = "complete"
    return EvaluationSummary(
        config_name=str(dataset.manifest.get("config_name", "?")),
        config_version=str(dataset.manifest.get("config_version", "?")),
        expected=expected,
        completed=completed,
        status_counts=status_counts,
        mean_wall_s=(sum(walls) / len(walls)) if walls else None,
        longest_horizon=max(horizons) if horizons else None,
        horizons=horizons,
        edos=edos,
        conditions=conditions,
        ci_enabled=ci,
        state=state,
    )


def _fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "-"
    if seconds >= 3600:
        return f"{seconds / 3600:.1f} h"
    if seconds >= 120:
        return f"{seconds / 60:.1f} min"
    return f"{seconds:.1f} s"


def render_overview_cards(summary: Optional[EvaluationSummary], job_status: str = "") -> str:
    """HTML stat cards.  Theme-agnostic: inherits Gradio's text colour, uses translucent fills."""
    css = (
        "<style>.evc{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:10px}"
        ".evc .c{border:1px solid rgba(128,128,128,.35);border-radius:10px;padding:10px 12px}"
        ".evc .v{font-size:1.55rem;font-weight:650;line-height:1.1}"
        ".evc .k{font-size:.78rem;opacity:.75;margin-top:2px}"
        ".evc .bad .v{color:#e34948}.evc .warn .v{color:#c98500}.evc .good .v{color:#2e9d4f}"
        ".evc .note{grid-column:1/-1;font-size:.82rem;opacity:.8}</style>"
    )
    if summary is None:
        return css + '<div class="evc"><div class="c note">No benchmark loaded.</div></div>'

    def card(value: str, label: str, tone: str = "") -> str:
        return f'<div class="c {tone}"><div class="v">{html.escape(value)}</div><div class="k">{html.escape(label)}</div></div>'

    s = summary
    cards = [
        card(f"{s.completed}/{s.expected}", "runs completed", "good" if s.state == "complete" else "warn"),
        card(str(s.ok), "successful", "good" if s.ok else ""),
        card(str(s.failed), "failed", "bad" if s.failed else ""),
        card(str(s.timeouts), "timeouts", "bad" if s.timeouts else ""),
        card(str(s.crashed), "crashed", "bad" if s.crashed else ""),
        card(str(s.not_run), "not run yet", "warn" if s.not_run else ""),
        card(_fmt_duration(s.mean_wall_s), "mean runtime"),
        card(f"{s.longest_horizon} beats" if s.longest_horizon else "-", "longest horizon"),
        card(", ".join(f"{e}-EDO" for e in s.edos) or "-", "EDOs tested"),
        card("on" if s.ci_enabled else "n/a", "95% CI over seeds"),
    ]
    status = job_status or {"complete": "Complete", "partial": "Partial results", "empty": "No results yet"}[s.state]
    head = (
        f'<div class="c note"><b>{html.escape(s.config_name)}</b> v{html.escape(s.config_version)} &middot; '
        f"conditions: {html.escape(', '.join(s.conditions) or '-')} &middot; "
        f"horizons: {html.escape(', '.join(map(str, s.horizons)) or '-')} &middot; status: <b>{html.escape(status)}</b></div>"
    )
    return css + f'<div class="evc">{head}{"".join(cards)}</div>'


# --------------------------------------------------------------------------------------
# Chart data (pure)
# --------------------------------------------------------------------------------------
def _finite(value: Any) -> TypeGuard[float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


@dataclass
class Point:
    """One bar/marker: mean with an optional 95% interval (None when n < 2)."""

    mean: float
    low: Optional[float]
    high: Optional[float]
    n: int


def _point(summary: Optional[Mapping[str, Any]]) -> Optional[Point]:
    if not summary or not summary.get("n"):
        return None
    mean = summary.get("mean")
    if not _finite(mean):
        return None
    low, high = summary.get("ci95_low"), summary.get("ci95_high")
    if _finite(low) and _finite(high):
        return Point(float(mean), float(low), float(high), int(summary["n"]))
    return Point(float(mean), None, None, int(summary["n"]))


@dataclass
class GroupedChartData:
    metric: str
    categories: List[str]
    series: Dict[str, List[Optional[Point]]]  # series name -> one Point (or None) per category
    notes: List[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not any(p is not None for pts in self.series.values() for p in pts)


def _cell_axes(cells: Mapping[str, Any]) -> Tuple[bool, bool]:
    """Whether path_mode / graph-grid vary across cells (then labels must include them)."""
    modes = {c["path_mode"] for c in cells.values()}
    grids = {(c["k_max"], c["d_max"], c["proposal_budget"]) for c in cells.values()}
    return len(modes) > 1, len(grids) > 1


def _cell_label(cell: Mapping[str, Any], with_mode: bool, with_grid: bool, with_edo: bool = True) -> str:
    parts = [f"{cell['horizon']}b"]
    if with_edo:
        parts.append(f"{cell['edo']}-EDO")
    if with_mode:
        parts.append(str(cell["path_mode"]))
    if with_grid:
        parts.append(f"K{cell['k_max']}/D{cell['d_max']}/B{cell['proposal_budget']}")
    return " / ".join(parts)


def _cell_sort(cell: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (cell["edo"], cell["horizon"], cell["path_mode"], cell["k_max"], cell["d_max"], cell["proposal_budget"])


def grouped_metric_data(dataset: EvaluationDataset, metric: str) -> GroupedChartData:
    """Per-(horizon, EDO) groups, one series per condition.  EDOs are separate groups, never pooled."""
    cells = dataset.cells
    with_mode, with_grid = _cell_axes(cells)
    ordered = sorted(cells.values(), key=_cell_sort)
    categories: List[str] = []
    for cell in ordered:
        label = _cell_label(cell, with_mode, with_grid)
        if label not in categories:
            categories.append(label)
    conditions = _sorted_conditions([c["condition"] for c in ordered])
    series: Dict[str, List[Optional[Point]]] = {c: [None] * len(categories) for c in conditions}
    for cell in ordered:
        idx = categories.index(_cell_label(cell, with_mode, with_grid))
        series[cell["condition"]][idx] = _point(cell["metrics"].get(metric))
    notes: List[str] = []
    if any(p is not None and p.low is None for pts in series.values() for p in pts):
        notes.append("Groups with a single successful seed have no confidence interval.")
    return GroupedChartData(metric, categories, series, notes)


def runtime_chart_data(dataset: EvaluationDataset, metric: str = "runtime.wall_s") -> GroupedChartData:
    return grouped_metric_data(dataset, metric)


# ---- ablation deltas --------------------------------------------------------------------
def comparison_keys(dataset: EvaluationDataset) -> List[Tuple[int, int, str]]:
    """Distinct (horizon, edo, label) cells that have at least one ablation paired with ``full``."""
    cells = dataset.cells
    with_mode, with_grid = _cell_axes(cells)
    seen: Dict[Tuple[int, int], str] = {}
    for key in dataset.paired:
        cell = cells.get(key)
        if cell:
            seen[(cell["horizon"], cell["edo"])] = _cell_label(cell, with_mode, with_grid)
    return [(h, e, label) for (h, e), label in sorted(seen.items(), key=lambda kv: (kv[0][1], kv[0][0]))]


@dataclass
class DeltaBar:
    condition: str
    mean: float
    low: Optional[float]
    high: Optional[float]
    n: int
    ci_excludes_zero: Optional[bool]
    tone: str  # "better" | "worse" | "neutral"


@dataclass
class AblationChartData:
    horizon: Optional[int]
    edo: Optional[int]
    metrics: Dict[str, List[DeltaBar]]  # metric -> one bar per ablation
    notes: List[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not any(self.metrics.values())


def _tone(metric: str, mean: float, excludes_zero: Optional[bool]) -> str:
    direction = METRIC_INFO.get(metric, ("", "", None))[2]
    if direction is None or mean == 0:
        return "neutral"
    better = mean < 0 if direction == "lower" else mean > 0
    return "better" if better else "worse"


def ablation_delta_data(
    dataset: EvaluationDataset,
    horizon: Optional[int] = None,
    edo: Optional[int] = None,
    metrics: Sequence[str] = ABLATION_METRICS,
) -> AblationChartData:
    """Paired (same-seed) change of each ablation relative to ``full`` for one (horizon, EDO) cell."""
    options = comparison_keys(dataset)
    if not options:
        return AblationChartData(horizon, edo, {}, ["No ablation has a matching `full` cell yet."])
    if horizon is None or edo is None or (horizon, edo) not in {(h, e) for h, e, _ in options}:
        horizon, edo = options[0][0], options[0][1]
    out: Dict[str, List[DeltaBar]] = {m: [] for m in metrics}
    for key, entry in dataset.paired.items():
        cell = dataset.cells.get(key)
        if not cell or cell["horizon"] != horizon or cell["edo"] != edo:
            continue
        for m in metrics:
            d = entry["metrics"].get(m)
            if not d or not d.get("n") or not _finite(d.get("mean")):
                continue
            low, high = d.get("ci95_low"), d.get("ci95_high")
            has_ci = _finite(low) and _finite(high)
            out[m].append(
                DeltaBar(
                    cell["condition"], float(d["mean"]), low if has_ci else None, high if has_ci else None,
                    int(d["n"]), d.get("ci_excludes_zero"), _tone(m, float(d["mean"]), d.get("ci_excludes_zero")),
                )
            )
    for bars in out.values():
        bars.sort(key=lambda b: _condition_rank(b.condition))
    notes = ["Bars show mean paired difference vs `full` over shared seeds; whiskers are 95% intervals. "
             "Faded bars have an interval that includes 0."]
    return AblationChartData(horizon, edo, {m: b for m, b in out.items() if b}, notes)


# ---- tension --------------------------------------------------------------------------
@dataclass
class TensionCurveSeries:
    condition: str
    mean: List[float]
    low: List[float]
    high: List[float]
    n: int


@dataclass
class TensionChartData:
    horizon: Optional[int]
    edo: Optional[int]
    target: Optional[List[float]]
    curves: List[TensionCurveSeries]
    summary_mae: GroupedChartData  # fallback / companion: per-condition tension MAE
    summary_corr: GroupedChartData
    has_curves: bool
    notes: List[str] = field(default_factory=list)


def tension_cells(dataset: EvaluationDataset) -> List[Tuple[int, int, str]]:
    cells = dataset.cells
    with_mode, with_grid = _cell_axes(cells)
    seen: Dict[Tuple[int, int], str] = {}
    for cell in cells.values():
        seen[(cell["horizon"], cell["edo"])] = _cell_label(cell, with_mode, with_grid)
    return [(h, e, label) for (h, e), label in sorted(seen.items(), key=lambda kv: (kv[0][1], kv[0][0]))]


def _curve(result: Mapping[str, Any], name: str) -> Optional[List[float]]:
    values = ((result.get("metrics") or {}).get("tension") or {}).get(name)
    if isinstance(values, list) and values and all(_finite(v) for v in values):
        return [float(v) for v in values]
    return None


def tension_chart_data(
    dataset: EvaluationDataset, horizon: Optional[int] = None, edo: Optional[int] = None
) -> TensionChartData:
    """Target vs realized tension per condition for one (horizon, EDO) cell.

    Per-beat curves exist only in results produced by an evaluator that records them; for older
    outputs the chart falls back to the stored summary metrics (MAE, shape correlation).
    """
    options = tension_cells(dataset)
    mae = grouped_metric_data(dataset, "tension.mean_absolute_error")
    corr = grouped_metric_data(dataset, "tension.shape_correlation")
    if not options:
        return TensionChartData(horizon, edo, None, [], mae, corr, False, ["No results yet."])
    if horizon is None or edo is None or (horizon, edo) not in {(h, e) for h, e, _ in options}:
        horizon, edo = options[0][0], options[0][1]
    target: Optional[List[float]] = None
    by_condition: Dict[str, List[List[float]]] = {}
    for res in dataset.results:
        spec = res["spec"]
        if res["status"] != STATUS_OK or spec["horizon"] != horizon or spec["edo"] != edo:
            continue
        realized = _curve(res, "realized_curve")
        if realized is None:
            continue
        by_condition.setdefault(spec["condition"], []).append(realized)
        if target is None:
            target = _curve(res, "target_curve")
    curves: List[TensionCurveSeries] = []
    for cond in _sorted_conditions(list(by_condition)):
        runs = by_condition[cond]
        length = min(len(r) for r in runs)
        cols = list(zip(*[r[:length] for r in runs]))
        curves.append(
            TensionCurveSeries(
                cond, [sum(c) / len(c) for c in cols], [min(c) for c in cols], [max(c) for c in cols], len(runs)
            )
        )
    notes = []
    if not curves:
        notes.append(
            "Per-beat tension curves are not stored in these results (they were produced before curves were "
            "recorded). Showing the stored summary metrics instead; re-run the benchmark to get target-vs-realized curves."
        )
    else:
        notes.append("Lines are the mean over seeds; shaded bands span the min-max over seeds.")
    return TensionChartData(horizon, edo, target, curves, mae, corr, bool(curves), notes)


# ---- scaling / EDO / status -------------------------------------------------------------
@dataclass
class ScalingChartData:
    metrics: Dict[str, Dict[int, Dict[str, List[Tuple[int, Point]]]]]  # metric -> edo -> condition -> [(horizon, pt)]
    edos: List[int]
    notes: List[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not any(self.metrics.values())


def scaling_chart_data(
    dataset: EvaluationDataset, metrics: Sequence[str] = SCALING_METRICS, conditions: Optional[Sequence[str]] = None
) -> ScalingChartData:
    """Metric vs horizon, one line per condition, separate panel per EDO."""
    out: Dict[str, Dict[int, Dict[str, List[Tuple[int, Point]]]]] = {m: {} for m in metrics}
    edos: set[int] = set()
    for cell in dataset.cells.values():
        if conditions and cell["condition"] not in conditions:
            continue
        edos.add(cell["edo"])
        for m in metrics:
            pt = _point(cell["metrics"].get(m))
            if pt is not None:
                out[m].setdefault(cell["edo"], {}).setdefault(cell["condition"], []).append((cell["horizon"], pt))
    for per_edo in out.values():
        for per_cond in per_edo.values():
            for series in per_cond.values():
                series.sort(key=lambda hp: hp[0])
    notes = []
    horizons = {c["horizon"] for c in dataset.cells.values()}
    if len(horizons) < 2:
        notes.append("Only one horizon is present, so there is no scaling trend to draw yet.")
    return ScalingChartData(out, sorted(edos), notes)


@dataclass
class EdoChartData:
    horizon: Optional[int]
    metrics: Dict[str, Dict[int, Dict[str, Optional[Point]]]]  # metric -> edo -> condition -> pt
    conditions: List[str]
    edos: List[int]
    horizons: List[int]

    @property
    def empty(self) -> bool:
        return not any(
            pt is not None for per_edo in self.metrics.values() for per_cond in per_edo.values() for pt in per_cond.values()
        )


def edo_comparison_data(
    dataset: EvaluationDataset, horizon: Optional[int] = None, metrics: Sequence[str] = EDO_METRICS
) -> EdoChartData:
    """Side-by-side EDOs for one horizon.  Each EDO keeps its own bar; values are never averaged together."""
    horizons = sorted({c["horizon"] for c in dataset.cells.values()})
    if horizon not in horizons:
        horizon = horizons[-1] if horizons else None
    edos = sorted({c["edo"] for c in dataset.cells.values()})
    conditions = _sorted_conditions([c["condition"] for c in dataset.cells.values()])
    out: Dict[str, Dict[int, Dict[str, Optional[Point]]]] = {
        m: {e: {c: None for c in conditions} for e in edos} for m in metrics
    }
    for cell in dataset.cells.values():
        if cell["horizon"] != horizon:
            continue
        for m in metrics:
            out[m][cell["edo"]][cell["condition"]] = _point(cell["metrics"].get(m))
    return EdoChartData(horizon, out, conditions, edos, horizons)


@dataclass
class StatusChartData:
    conditions: List[str]
    counts: Dict[str, Dict[str, int]]  # status -> condition -> count
    totals: Dict[str, int]  # status -> count


def status_chart_data(dataset: EvaluationDataset) -> StatusChartData:
    """Raw run statuses per condition; runs without a result are shown as ``not_run``, never assumed done."""
    manifest_specs = {e["run_id"]: e.get("spec", {}) for e in dataset.manifest.get("expected_runs", [])}
    done = {r["run_id"]: r for r in dataset.results}
    for rid, res in done.items():
        manifest_specs.setdefault(rid, res["spec"])
    conditions = _sorted_conditions([s.get("condition", "?") for s in manifest_specs.values()])
    statuses = list(STATUS_ORDER) + [STATUS_NOT_RUN]
    for res in dataset.results:
        if res["status"] not in statuses:
            statuses.insert(len(STATUS_ORDER), res["status"])
    counts = {s: {c: 0 for c in conditions} for s in statuses}
    for rid, spec in manifest_specs.items():
        status = done[rid]["status"] if rid in done else STATUS_NOT_RUN
        counts[status][spec.get("condition", "?")] += 1
    totals = {s: sum(v.values()) for s, v in counts.items()}
    return StatusChartData(conditions, counts, totals)


# --------------------------------------------------------------------------------------
# Plotly figures
# --------------------------------------------------------------------------------------
def _go() -> Any:
    import plotly.graph_objects as go

    return go


def _make_subplots(**kwargs: Any) -> Any:
    from plotly.subplots import make_subplots

    return make_subplots(**kwargs)


def _style(fig: Any, title: str, height: int = 420, legend_right: bool = False) -> Any:
    legend = (
        {"orientation": "v", "yanchor": "top", "y": 1.0, "xanchor": "left", "x": 1.01}
        if legend_right
        else {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0}
    )
    fig.update_layout(
        title={"text": title, "x": 0.01, "font": {"size": 15}},
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": INK, "size": 12},
        legend=legend,
        margin={"l": 60, "r": 110 if legend_right else 20, "t": 80, "b": 60},
        hoverlabel={"namelength": -1},
    )
    fig.update_xaxes(gridcolor="rgba(128,128,128,.18)", zerolinecolor="rgba(128,128,128,.35)", linecolor="rgba(128,128,128,.35)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,.18)", zerolinecolor="rgba(128,128,128,.35)", linecolor="rgba(128,128,128,.35)")
    return fig


def _empty_figure(message: str, title: str = "") -> Any:
    go = _go()
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font={"size": 14, "color": INK}, xref="paper", yref="paper", x=0.5, y=0.5)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _style(fig, title, height=260)


def _error_arrays(points: Sequence[Optional[Point]]) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    ys: List[Optional[float]] = []
    plus: List[Optional[float]] = []
    minus: List[Optional[float]] = []
    for p in points:
        if p is None:
            ys.append(None)
            plus.append(None)
            minus.append(None)
        else:
            ys.append(p.mean)
            plus.append(0.0 if p.high is None else max(p.high - p.mean, 0.0))
            minus.append(0.0 if p.low is None else max(p.mean - p.low, 0.0))
    return ys, plus, minus


def _hover(points: Sequence[Optional[Point]], unit: str) -> List[str]:
    out = []
    for p in points:
        if p is None:
            out.append("no data")
        elif p.low is None:
            out.append(f"mean {p.mean:.4g}{unit} (n={p.n}, no CI)")
        else:
            out.append(f"mean {p.mean:.4g}{unit}<br>95% CI [{p.low:.4g}, {p.high:.4g}] (n={p.n})")
    return out


def make_runtime_chart(dataset: EvaluationDataset, metric: str = "runtime.wall_s") -> Any:
    """Grouped bars: full vs ablations per (horizon, EDO) with 95% CI whiskers."""
    data = runtime_chart_data(dataset, metric)
    title = f"{metric_label(metric)} by condition"
    if data.empty:
        return _empty_figure("No successful runs to chart yet.", title)
    go = _go()
    unit = METRIC_INFO.get(metric, ("", "", None))[1]
    fig = go.Figure()
    for cond, pts in data.series.items():
        ys, plus, minus = _error_arrays(pts)
        fig.add_bar(
            name=cond, x=data.categories, y=ys,
            error_y={"type": "data", "symmetric": False, "array": plus, "arrayminus": minus, "thickness": 1.4, "width": 3},
            marker={"color": CONDITION_COLORS.get(cond, INK)}, customdata=_hover(pts, f" {unit}".rstrip()),
            hovertemplate=f"<b>{cond}</b><br>%{{x}}<br>%{{customdata}}<extra></extra>",
        )
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.06)
    fig.update_xaxes(title="horizon / EDO")
    fig.update_yaxes(title=metric_label(metric), rangemode="tozero")
    return _style(fig, title)


def make_ablation_chart(dataset: EvaluationDataset, horizon: Optional[int] = None, edo: Optional[int] = None) -> Any:
    """One panel per metric (own x-scale), horizontal bars of ablation minus full with paired 95% CI."""
    data = ablation_delta_data(dataset, horizon, edo)
    title = "Change from full (paired by seed)"
    if data.empty:
        return _empty_figure(data.notes[0] if data.notes else "No ablation data.", title)
    go = _go()
    metrics = list(data.metrics)
    fig = _make_subplots(rows=1, cols=len(metrics), subplot_titles=[metric_label(m) for m in metrics], horizontal_spacing=0.07)
    tone_colors = {"better": "#2e9d4f", "worse": "#e34948", "neutral": "#7d8590"}
    for col, m in enumerate(metrics, start=1):
        bars = data.metrics[m]
        direction = METRIC_INFO.get(m, ("", "", None))[2]
        fig.add_trace(
            go.Bar(
                orientation="h", y=[b.condition for b in bars], x=[b.mean for b in bars], showlegend=False,
                marker={
                    "color": [tone_colors[b.tone] for b in bars],
                    "opacity": [1.0 if b.ci_excludes_zero else 0.45 for b in bars],
                },
                error_x={
                    "type": "data", "symmetric": False,
                    "array": [0 if b.high is None else b.high - b.mean for b in bars],
                    "arrayminus": [0 if b.low is None else b.mean - b.low for b in bars],
                    "thickness": 1.4, "width": 3,
                },
                customdata=[
                    f"{b.mean:+.4g}" + ("" if b.low is None else f" [{b.low:+.4g}, {b.high:+.4g}]")
                    + f" (n={b.n})" + {"better": " - better than full", "worse": " - worse than full", "neutral": ""}[b.tone]
                    for b in bars
                ],
                hovertemplate="<b>%{y}</b> vs full<br>%{customdata}<extra></extra>",
            ),
            row=1, col=col,
        )
        fig.update_xaxes(title=("lower is better" if direction == "lower" else "higher is better" if direction == "higher" else "delta"),
                         zeroline=True, row=1, col=col)
        fig.update_yaxes(autorange="reversed", showticklabels=(col == 1), row=1, col=col)
    label = f"{data.horizon} beats / {data.edo}-EDO"
    return _style(fig, f"{title}: {label}", height=360)


def make_tension_chart(dataset: EvaluationDataset, horizon: Optional[int] = None, edo: Optional[int] = None) -> Any:
    """Target vs realized tension curves; falls back to summary MAE/correlation when curves are absent."""
    data = tension_chart_data(dataset, horizon, edo)
    go = _go()
    if data.has_curves:
        fig = go.Figure()
        if data.target:
            fig.add_scatter(
                x=list(range(len(data.target))), y=data.target, name="target", mode="lines",
                line={"color": "#9a9a94", "width": 2.5, "dash": "dash"},
                hovertemplate="target<br>beat %{x}: %{y:.3f}<extra></extra>",
            )
        for c in data.curves:
            x = list(range(len(c.mean)))
            color = CONDITION_COLORS.get(c.condition, INK)
            fig.add_scatter(x=x + x[::-1], y=c.high + c.low[::-1], fill="toself", mode="none", showlegend=False,
                            fillcolor=_rgba(color, 0.15), hoverinfo="skip")
            fig.add_scatter(x=x, y=c.mean, name=f"{c.condition} (n={c.n})", mode="lines", line={"color": color, "width": 2},
                            hovertemplate=f"{c.condition}<br>beat %{{x}}: %{{y:.3f}}<extra></extra>")
        fig.update_xaxes(title="beat")
        fig.update_yaxes(title="tension")
        return _style(fig, f"Tension tracking: {data.horizon} beats / {data.edo}-EDO (target vs realized)")
    # fallback: summary metrics, two panels
    if data.summary_mae.empty and data.summary_corr.empty:
        return _empty_figure("No tension metrics available yet.", "Tension tracking")
    fig = _make_subplots(rows=1, cols=2, subplot_titles=["Tension MAE (lower is better)", "Shape correlation (higher is better)"])
    for col, gd in enumerate((data.summary_mae, data.summary_corr), start=1):
        for cond, pts in gd.series.items():
            ys, plus, minus = _error_arrays(pts)
            fig.add_trace(
                go.Bar(name=cond, x=gd.categories, y=ys, showlegend=(col == 1), marker={"color": CONDITION_COLORS.get(cond, INK)},
                       legendgroup=cond,
                       error_y={"type": "data", "symmetric": False, "array": plus, "arrayminus": minus, "thickness": 1.4, "width": 3},
                       customdata=_hover(pts, ""), hovertemplate=f"<b>{cond}</b><br>%{{x}}<br>%{{customdata}}<extra></extra>"),
                row=1, col=col,
            )
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.06)
    return _style(fig, "Tension tracking (summary metrics; per-beat curves not recorded in these results)", legend_right=True)


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def make_scaling_chart(dataset: EvaluationDataset, conditions: Optional[Sequence[str]] = None) -> Any:
    """Runtime / peak RSS / RSS growth vs horizon, lines per condition, one column per EDO."""
    data = scaling_chart_data(dataset, conditions=conditions)
    title = "Runtime and memory scaling with horizon"
    if data.empty:
        return _empty_figure("No successful runs to chart yet.", title)
    go = _go()
    metrics = list(SCALING_METRICS)
    edos = data.edos
    fig = _make_subplots(
        rows=len(metrics), cols=len(edos), shared_xaxes=True,
        subplot_titles=[f"{metric_label(m)} - {e}-EDO" for m in metrics for e in edos], vertical_spacing=0.12,
    )
    shown: set[str] = set()
    for r, m in enumerate(metrics, start=1):
        for c, edo in enumerate(edos, start=1):
            for cond, series in data.metrics[m].get(edo, {}).items():
                xs = [h for h, _ in series]
                pts = [p for _, p in series]
                ys, plus, minus = _error_arrays(pts)
                fig.add_trace(
                    go.Scatter(
                        x=xs, y=ys, mode="lines+markers", name=cond, legendgroup=cond, showlegend=cond not in shown,
                        line={"color": CONDITION_COLORS.get(cond, INK), "width": 2}, marker={"size": 8},
                        error_y={"type": "data", "symmetric": False, "array": plus, "arrayminus": minus, "thickness": 1.2, "width": 3},
                        customdata=_hover(pts, ""), hovertemplate=f"<b>{cond}</b><br>%{{x}} beats<br>%{{customdata}}<extra></extra>",
                    ),
                    row=r, col=c,
                )
                shown.add(cond)
    horizons = sorted({h for per_edo in data.metrics.values() for per_cond in per_edo.values() for s in per_cond.values() for h, _ in s})
    fig.update_xaxes(tickmode="array", tickvals=horizons)
    fig.update_xaxes(title="horizon (beats)", row=len(metrics))
    return _style(fig, title, height=250 * len(metrics) + 90, legend_right=True)


def make_edo_chart(dataset: EvaluationDataset, horizon: Optional[int] = None) -> Any:
    """Per-metric panels; x = condition, one bar per EDO (EDOs are never merged)."""
    data = edo_comparison_data(dataset, horizon)
    title = f"EDO comparison ({data.horizon} beats)" if data.horizon else "EDO comparison"
    if data.empty:
        return _empty_figure("No successful runs to chart yet.", title)
    go = _go()
    metrics = list(data.metrics)
    fig = _make_subplots(rows=1, cols=len(metrics), subplot_titles=[metric_label(m) for m in metrics], horizontal_spacing=0.06)
    for col, m in enumerate(metrics, start=1):
        for i, edo in enumerate(data.edos):
            pts = [data.metrics[m][edo][c] for c in data.conditions]
            ys, plus, minus = _error_arrays(pts)
            fig.add_trace(
                go.Bar(
                    name=f"{edo}-EDO", x=data.conditions, y=ys, legendgroup=str(edo), showlegend=(col == 1),
                    marker={"color": EDO_COLORS[i % len(EDO_COLORS)]},
                    error_y={"type": "data", "symmetric": False, "array": plus, "arrayminus": minus, "thickness": 1.2, "width": 3},
                    customdata=_hover(pts, ""), hovertemplate=f"<b>{edo}-EDO</b><br>%{{x}}<br>%{{customdata}}<extra></extra>",
                ),
                row=1, col=col,
            )
    fig.update_layout(barmode="group", bargap=0.2, bargroupgap=0.05)
    fig.update_yaxes(rangemode="tozero")
    return _style(fig, title, height=400, legend_right=True)


def make_status_chart(dataset: EvaluationDataset) -> Any:
    """Stacked horizontal bars of raw run statuses per condition (including runs not yet run)."""
    data = status_chart_data(dataset)
    title = "Run status by condition"
    if not data.conditions:
        return _empty_figure("No runs in the manifest.", title)
    go = _go()
    fig = go.Figure()
    for status, per_cond in data.counts.items():
        if not data.totals.get(status) and status not in (STATUS_OK,):
            continue
        fig.add_bar(
            name=status.replace("_", " "), orientation="h", y=data.conditions, x=[per_cond[c] for c in data.conditions],
            marker={"color": STATUS_COLORS.get(status, INK), "line": {"color": "rgba(0,0,0,0)", "width": 0}},
            hovertemplate=f"{status.replace('_', ' ')}: %{{x}}<extra>%{{y}}</extra>",
        )
    fig.update_layout(barmode="stack")
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title="runs")
    return _style(fig, title, height=120 + 50 * len(data.conditions))


# --------------------------------------------------------------------------------------
# Run inspection
# --------------------------------------------------------------------------------------
def run_choices(dataset: EvaluationDataset) -> List[Tuple[str, str]]:
    """(label, run_id) for the run selector; failures are flagged so they are easy to find."""
    out = []
    for r in sorted(
        dataset.results,
        key=lambda r: (r["spec"]["edo"], r["spec"]["horizon"], _condition_rank(r["spec"]["condition"]), r["spec"]["seed"]),
    ):
        spec = r["spec"]
        flag = "" if r["status"] == STATUS_OK else f" [{r['status'].upper()}]"
        out.append((f"{spec['condition']} | {spec['horizon']}b | {spec['edo']}-EDO | seed {spec['seed']}{flag}", r["run_id"]))
    return out


def find_run(dataset: EvaluationDataset, run_id: Optional[str]) -> Optional[Dict[str, Any]]:
    for r in dataset.results:
        if r["run_id"] == run_id:
            return r
    return None


def _kv_table(rows: Sequence[Tuple[str, Any]]) -> str:
    lines = ["| field | value |", "|---|---|"]
    for k, v in rows:
        lines.append(f"| {k} | {_fmt_value(v)} |")
    return "\n".join(lines)


def _fmt_value(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.5g}"
    if isinstance(v, (list, tuple)):
        return ", ".join(_fmt_value(x) for x in v) if v else "-"
    if isinstance(v, dict):
        return ", ".join(f"{k}: {_fmt_value(x)}" for k, x in v.items()) if v else "-"
    return str(v).replace("|", "\\|")


def _scalar_rows(section: Mapping[str, Any]) -> List[Tuple[str, Any]]:
    return [(k, v) for k, v in section.items() if not (isinstance(v, list) and len(v) > 12)]


def render_run_details(result: Optional[Mapping[str, Any]]) -> str:
    """Markdown detail view of one raw run result; failed runs show their structured failure record."""
    if not result:
        return "_Select a run to inspect it._"
    spec = result.get("spec", {})
    runtime = result.get("runtime") or {}
    metrics = result.get("metrics") or {}
    status = str(result.get("status", "?"))
    parts = [f"### `{result.get('run_id', '?')}`", f"**Status:** {status}"]
    failure = result.get("failure")
    if failure:
        parts += [
            "",
            "#### Failure",
            _kv_table([("kind", failure.get("kind")), ("stage", failure.get("stage")), ("message", failure.get("message"))]),
        ]
        tail = failure.get("traceback_tail")
        if tail:
            parts += ["", "<details><summary>Traceback tail</summary>", "", "```", str(tail), "```", "", "</details>"]
    parts += [
        "",
        "#### Configuration",
        _kv_table([
            ("condition", spec.get("condition")), ("seed", spec.get("seed")), ("horizon (beats)", spec.get("horizon")),
            ("EDO", spec.get("edo")), ("path mode", spec.get("path_mode")), ("bridge", spec.get("bridge_mode")),
            ("prior", (result.get("prior") or {}).get("mode", spec.get("prior_mode"))),
            ("prior status", (result.get("prior") or {}).get("status")),
            ("lambda_data / lambda_gttm", f"{spec.get('lambda_data')} / {spec.get('lambda_gttm')}"),
            ("K_max / D_max / proposal budget", f"{spec.get('k_max')} / {spec.get('d_max')} / {spec.get('proposal_budget')}"),
        ]),
        "",
        "#### Runtime and memory",
        _kv_table([
            ("wall time (s)", runtime.get("wall_s")), ("CPU time (s)", runtime.get("cpu_s")),
            ("generation wall (s)", runtime.get("generate_wall_s")), ("peak RSS (MB)", runtime.get("peak_rss_mb")),
            ("baseline RSS (MB)", runtime.get("baseline_rss_mb")),
            ("RSS growth (MB)", (runtime["peak_rss_mb"] - runtime["baseline_rss_mb"])
             if _finite(runtime.get("peak_rss_mb")) and _finite(runtime.get("baseline_rss_mb")) else None),
        ]),
    ]
    titles = {"graph": "Graph metrics", "sb": "SB convergence", "tension": "Tension metrics", "gttm": "GTTM score", "path": "Path metrics"}
    for key in ("graph", "sb", "tension", "gttm", "path"):
        section = metrics.get(key) or {}
        rows = _scalar_rows(section)
        parts += ["", f"#### {titles[key]}"]
        parts.append(_kv_table(rows) if rows else "_not available_")
    artifacts = result.get("artifacts") or {}
    parts += ["", "#### Generated artifacts", _kv_table(list(artifacts.items())) if artifacts else "_none recorded_"]
    parts.append(
        "\nThe benchmark records a SHA-256 of the chosen path rather than writing audio/MIDI files, "
        "so there are no file paths to open for these runs."
    )
    raw = result.get("reproducibility") or {}
    if raw:
        parts += ["", "<details><summary>Reproducibility</summary>", "", _kv_table(list(raw.items())), "", "</details>"]
    return "\n".join(parts)


def run_tension_figure(result: Optional[Mapping[str, Any]]) -> Any:
    """Small per-run tension plot: target vs realized curve, or a stored-summary bar when curves are absent."""
    go = _go()
    if not result:
        return _empty_figure("Select a run.", "Run tension")
    target, realized = _curve(result, "target_curve"), _curve(result, "realized_curve")
    spec = result.get("spec", {})
    if realized is None:
        tension = (result.get("metrics") or {}).get("tension") or {}
        keys = [k for k in ("realized_min", "realized_mean", "realized_max") if _finite(tension.get(k))]
        if not keys:
            return _empty_figure("No tension data for this run" + (" (run did not succeed)." if result.get("status") != STATUS_OK else "."), "Run tension")
        fig = go.Figure(go.Bar(x=[k.replace("realized_", "") for k in keys], y=[tension[k] for k in keys], marker={"color": "#2a78d6"}))
        fig.update_yaxes(title="realized tension", range=[0, 1])
        return _style(fig, "Realized tension summary (per-beat curve not recorded)", height=300)
    fig = go.Figure()
    if target:
        fig.add_scatter(x=list(range(len(target))), y=target, name="target", mode="lines", line={"color": "#9a9a94", "width": 2.5, "dash": "dash"})
    color = CONDITION_COLORS.get(spec.get("condition", ""), "#2a78d6")
    fig.add_scatter(x=list(range(len(realized))), y=realized, name="realized", mode="lines", line={"color": color, "width": 2})
    fig.update_xaxes(title="beat")
    fig.update_yaxes(title="tension")
    return _style(fig, "Target vs realized tension", height=320)


# --------------------------------------------------------------------------------------
# Config metadata, filter choices and validation (pure: no Gradio, no benchmark execution)
# --------------------------------------------------------------------------------------
TYPE_SMOKE = "SMOKE"
TYPE_PROFILE = "PROFILE"
TYPE_BENCHMARK = "BENCHMARK"
BENCHMARK_TYPE_TITLES = {
    TYPE_SMOKE: "Smoke test",
    TYPE_PROFILE: "Scaling/profile test",
    TYPE_BENCHMARK: "Main benchmark",
}
BENCHMARK_TYPE_COLORS = {TYPE_SMOKE: "#c98500", TYPE_PROFILE: "#6250d6", TYPE_BENCHMARK: "#2a78d6"}

# One central place for condition explanations.  A config may override or extend these with an
# optional ``"condition_descriptions": {name: text}`` object (config.raw is the source of truth);
# conditions with neither fall back to a description derived from ``config.CONDITIONS``.
CONDITION_DESCRIPTIONS: Dict[str, str] = {
    "full": "Normal configuration using GTTM, the configured prior, and SB.",
    "no_gttm": "Disables GTTM scoring while preserving legality constraints.",
    "no_prior": "Uses NullPrior to measure the contribution of the prior.",
    "no_sb": "Uses the uniform reachable-walk baseline instead of SB reweighting.",
    "trained_prior": (
        "Uses a trained neural prior; reports an explicit prior_unavailable result "
        "if no trained bundle can be loaded."
    ),
}

TERM_HELP: Dict[str, str] = {
    "Horizon": (
        "Number of beats generated. Larger horizons test longer music but require more time and memory."
    ),
    "EDO": (
        "Equal Divisions of the Octave. 12-EDO is standard semitone tuning; 19-EDO divides the octave "
        "into 19 pitch steps. It is the pitch-space resolution used throughout graph construction, "
        "scoring, decoding, and rendering."
    ),
    "Seed": "Random starting value used to reproduce a run or compare repeated runs.",
    "Condition": "A system variant used to measure the contribution of a component.",
    "Workers": (
        "Number of benchmark subprocesses running in parallel. More workers may reduce wall time "
        "but use more CPU and memory."
    ),
    "Proposal budget": "Number of candidate transitions considered before final graph retention.",
    "K_max": "Maximum graph/layer state limit.",
    "D_max": "Maximum number of outgoing edges retained per state.",
    "SB": "Schrödinger Bridge path-reweighting/inference stage.",
}


def benchmark_type(config: BenchmarkConfig) -> str:
    """``SMOKE`` | ``PROFILE`` | ``BENCHMARK`` for a config.

    An explicit ``"benchmark_type"`` key in the config wins.  Otherwise the type is inferred from
    the config name, then the description.  (The shipped configs deliberately carry no such key:
    adding one would change their ``config_hash`` and orphan existing output directories.)
    """
    explicit = str(config.raw.get("benchmark_type", "")).strip().upper()
    if explicit in BENCHMARK_TYPE_TITLES:
        return explicit
    name = config.name.lower()
    if "smoke" in name:
        return TYPE_SMOKE
    if "profile" in name:
        return TYPE_PROFILE
    desc = config.description.lower()
    if "smoke" in desc:
        return TYPE_SMOKE
    if "scaling probe" in desc or "profile" in desc:
        return TYPE_PROFILE
    return TYPE_BENCHMARK


def condition_description(condition: str, overrides: Optional[Mapping[str, str]] = None) -> str:
    """Human explanation of a condition (config override > central mapping > derived from CONDITIONS)."""
    if overrides and condition in overrides:
        return str(overrides[condition])
    if condition in CONDITION_DESCRIPTIONS:
        return CONDITION_DESCRIPTIONS[condition]
    cdef = CONDITIONS.get(condition)
    if cdef is not None:
        return f"Variant of the full system that changes: {cdef.factor}."
    return "No description is defined for this condition."


@dataclass(frozen=True)
class FilterChoices:
    """The values a config actually contains, per filterable axis."""

    conditions: Tuple[str, ...] = ()
    horizons: Tuple[int, ...] = ()
    edos: Tuple[int, ...] = ()
    seeds: Tuple[int, ...] = ()


@dataclass(frozen=True)
class FilterSelection:
    """What the user ticked.  An empty axis means "use all values available in this config"."""

    conditions: Tuple[str, ...] = ()
    horizons: Tuple[int, ...] = ()
    edos: Tuple[int, ...] = ()
    seeds: Tuple[int, ...] = ()


@dataclass(frozen=True)
class ConfigMetadata:
    file_name: str
    name: str
    version: str
    description: str
    benchmark_type: str
    horizons: Tuple[int, ...]
    edos: Tuple[int, ...]
    seeds: Tuple[int, ...]
    conditions: Tuple[str, ...]
    path_modes: Tuple[str, ...]
    k_max: Tuple[int, ...]
    d_max: Tuple[int, ...]
    proposal_budget: Tuple[int, ...]
    sb_max_iterations: int
    sb_tolerance: float
    sb_temperature: float
    timeout_s: float
    expected_runs: int
    condition_descriptions: Dict[str, str]


def _unique(values: Sequence[Any]) -> Tuple[Any, ...]:
    return tuple(dict.fromkeys(values))


def available_filter_choices(config: BenchmarkConfig) -> FilterChoices:
    return FilterChoices(
        conditions=tuple(config.conditions), horizons=tuple(config.horizons),
        edos=tuple(config.edos), seeds=tuple(config.seeds),
    )


def config_metadata(config: BenchmarkConfig, file_name: str = "") -> ConfigMetadata:
    """Describe a parsed config.  Every value comes from the config; ``expected_runs`` from ``expand_matrix``."""
    raw_desc = config.raw.get("condition_descriptions")
    overrides = {str(k): str(v) for k, v in raw_desc.items()} if isinstance(raw_desc, Mapping) else {}
    return ConfigMetadata(
        file_name=file_name or f"{config.name}.json",
        name=config.name,
        version=config.config_version,
        description=config.description,
        benchmark_type=benchmark_type(config),
        horizons=tuple(config.horizons),
        edos=tuple(config.edos),
        seeds=tuple(config.seeds),
        conditions=tuple(config.conditions),
        path_modes=tuple(config.path_modes),
        k_max=_unique([p.k_max for p in config.graph_grid]),
        d_max=_unique([p.d_max for p in config.graph_grid]),
        proposal_budget=_unique([p.proposal_budget for p in config.graph_grid]),
        sb_max_iterations=config.max_iterations,
        sb_tolerance=config.tolerance,
        sb_temperature=config.temperature,
        timeout_s=config.timeout_s,
        expected_runs=len(expand_matrix(config)),
        condition_descriptions={c: condition_description(c, overrides) for c in config.conditions},
    )


def _load_named_config(config_name: str) -> BenchmarkConfig:
    if not (config_name or "").strip():
        raise PreflightError("Choose a benchmark config.")
    path = config_path_for(config_name)
    try:
        return load_config(path)
    except (OSError, ValueError) as exc:
        raise PreflightError(f"Cannot load benchmark config {path.name}: {exc}") from exc


def load_config_metadata(config_name: str) -> ConfigMetadata:
    """Metadata for a config file name (inside ``config/benchmarks``) or path.  Raises :class:`PreflightError`."""
    return config_metadata(_load_named_config(config_name), Path(config_name).name)


def config_dropdown_choices(config_dir: Path = CONFIG_DIR) -> List[Tuple[str, str]]:
    """``(label, value)`` pairs for the config dropdown, e.g. ``("[SMOKE] m1_smoke.json", "m1_smoke.json")``."""
    out: List[Tuple[str, str]] = []
    for name in discover_benchmark_configs(config_dir):
        try:
            label = f"[{benchmark_type(load_config(config_dir / name))}] {name}"
        except (OSError, ValueError):
            label = f"[?] {name}"  # still selectable; the panel then shows why it cannot load
        out.append((label, name))
    return out


def keep_valid_selection(choices: FilterChoices, selection: FilterSelection) -> FilterSelection:
    """Drop selected values that the config does not contain (used when the config changes)."""
    return FilterSelection(
        conditions=tuple(v for v in selection.conditions if v in choices.conditions),
        horizons=tuple(v for v in selection.horizons if v in choices.horizons),
        edos=tuple(v for v in selection.edos if v in choices.edos),
        seeds=tuple(v for v in selection.seeds if v in choices.seeds),
    )


@dataclass(frozen=True)
class ConfigSwitch:
    metadata: ConfigMetadata
    choices: FilterChoices
    selection: FilterSelection  # the previous selection, minus values the new config lacks


def switch_config(config_name: str, current: FilterSelection = FilterSelection()) -> ConfigSwitch:
    """Everything the page must update when the config dropdown changes."""
    config = _load_named_config(config_name)
    choices = available_filter_choices(config)
    return ConfigSwitch(
        config_metadata(config, Path(config_name).name), choices, keep_valid_selection(choices, current)
    )


def _selected_specs(
    config: BenchmarkConfig, conditions: Sequence[str], horizons: Sequence[int],
    edos: Sequence[int], seeds: Sequence[int],
) -> List[RunSpec]:
    # The one place the UI turns filters into runs: the same expand_matrix/select_specs the CLI uses.
    return select_specs(
        expand_matrix(config), conditions=conditions or None, horizons=horizons or None,
        edos=edos or None, seeds=seeds or None,
    )


def selected_run_count(
    config: BenchmarkConfig, conditions: Sequence[str], horizons: Sequence[int],
    edos: Sequence[int], seeds: Sequence[int],
) -> int:
    """Exact number of runs the CLI would select for these filters (empty filter = all values)."""
    return len(_selected_specs(config, conditions, horizons, edos, seeds))


NO_RUNS_MESSAGE = "No runs selected."


def validate_filter_selection(
    config: BenchmarkConfig, conditions: Sequence[str], horizons: Sequence[int],
    edos: Sequence[int], seeds: Sequence[int],
) -> List[str]:
    """Problems with a filter selection (empty list = valid).  Never raises."""
    choices = available_filter_choices(config)
    problems: List[str] = []
    for label, picked, valid in (
        ("Conditions", conditions, choices.conditions), ("Horizons", horizons, choices.horizons),
        ("EDO values", edos, choices.edos), ("Seeds", seeds, choices.seeds),
    ):
        bad = [v for v in picked if v not in valid]
        if bad:
            problems.append(f"{label} not available in this config: {', '.join(map(str, bad))}.")
    if not problems and selected_run_count(config, conditions, horizons, edos, seeds) == 0:
        problems.append(NO_RUNS_MESSAGE)
    return problems


def describe_valid_values(config: BenchmarkConfig, config_label: str) -> str:
    c = available_filter_choices(config)
    join = lambda values: ", ".join(map(str, values))  # noqa: E731
    return (
        f"{config_label} supports:\n"
        f"- Horizons: {join(c.horizons)}\n"
        f"- Seeds: {join(c.seeds)}\n"
        f"- EDO: {join(c.edos)}\n"
        f"- Conditions: {join(c.conditions)}"
    )


def format_validation_error(config: BenchmarkConfig, config_label: str, problems: Sequence[str]) -> str:
    """The message shown to the user: what is wrong, which config, and what is valid."""
    if list(problems) == [NO_RUNS_MESSAGE]:
        head = NO_RUNS_MESSAGE
    else:
        head = f"Invalid selection for {config_label}:\n" + "\n".join(f"- {p}" for p in problems)
    return (
        f"{head}\n\n{describe_valid_values(config, config_label)}\n\n"
        "Choose valid values or clear the filters to use all values."
    )



def _term(label: str, key: str) -> str:
    """Label with a hover tooltip carrying the TERM_HELP text."""
    return f'<abbr title="{html.escape(TERM_HELP[key], quote=True)}" style="cursor:help">{html.escape(label)}</abbr>'


def render_config_panel(meta: Optional[ConfigMetadata], error: str = "") -> str:
    """HTML for the "Benchmark Configuration" panel.  Theme-agnostic, like the overview cards."""
    css = (
        "<style>.bcp{border:1px solid rgba(128,128,128,.35);border-radius:10px;padding:12px 14px}"
        ".bcp .h{display:flex;gap:10px;align-items:baseline;flex-wrap:wrap;font-size:1.05rem}"
        ".bcp .badge{font-size:.72rem;font-weight:700;letter-spacing:.04em;color:#fff;border-radius:6px;padding:2px 8px}"
        ".bcp .d{margin:6px 0 10px;opacity:.85}"
        ".bcp dl{display:grid;grid-template-columns:max-content 1fr;gap:4px 14px;margin:0}"
        ".bcp dt{opacity:.7}.bcp dd{margin:0}"
        ".bcp ul{margin:2px 0 0;padding-left:18px}"
        ".bcp abbr{text-decoration:underline dotted}</style>"
    )
    title = '<div style="font-weight:650;margin-bottom:6px">Benchmark Configuration</div>'
    if meta is None:
        return css + f'<div class="bcp">{title}<div class="d">{html.escape(error or "No benchmark config selected.")}</div></div>'

    def vals(values: Sequence[Any], unit: str = "") -> str:
        return html.escape(", ".join(map(str, values)) + (f" {unit}" if unit else ""))

    badge = (
        f'<span class="badge" style="background:{BENCHMARK_TYPE_COLORS[meta.benchmark_type]}">{meta.benchmark_type}</span>'
    )
    conds = "".join(
        f"<li><code>{html.escape(c)}</code>: {html.escape(meta.condition_descriptions[c])}</li>" for c in meta.conditions
    )
    rows = [
        ("Type", f"{html.escape(BENCHMARK_TYPE_TITLES[meta.benchmark_type])}"),
        ("Config version", html.escape(meta.version)),
        (_term("Horizons", "Horizon"), vals(meta.horizons, "beats")),
        (_term("EDO", "EDO"), vals(meta.edos)),
        (_term("Seeds", "Seed"), vals(meta.seeds)),
        (_term("Conditions", "Condition"), f"<ul>{conds}</ul>"),
        ("Path modes", vals(meta.path_modes)),
        (_term("K_max", "K_max"), vals(meta.k_max)),
        (_term("D_max", "D_max"), vals(meta.d_max)),
        (_term("Proposal budget", "Proposal budget"), vals(meta.proposal_budget)),
        (
            _term("SB settings", "SB"),
            html.escape(
                f"max iterations {meta.sb_max_iterations}, tolerance {meta.sb_tolerance:g}, "
                f"temperature {meta.sb_temperature:g}"
            ),
        ),
        ("Timeout", html.escape(f"{meta.timeout_s:g} s per run")),
        ("Expected total runs", f"<b>{meta.expected_runs}</b>"),
    ]
    body = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in rows)
    return (
        css + f'<div class="bcp">{title}<div class="h">{badge}<b>{html.escape(meta.file_name)}</b></div>'
        f'<div class="d"><b>Purpose:</b> {html.escape(meta.description or "(no description)")}</div><dl>{body}</dl></div>'
    )


def render_terminology_markdown() -> str:
    return "\n".join(f"- **{term}**: {text}" for term, text in TERM_HELP.items())


# --------------------------------------------------------------------------------------
# Running benchmarks through the existing CLI
# --------------------------------------------------------------------------------------
class PreflightError(ValueError):
    """The requested benchmark action is refused before anything is launched."""


@dataclass(frozen=True)
class BenchmarkRequest:
    config: str  # file name inside config/benchmarks, or a path
    out_dir: str
    workers: int = 1
    conditions: Tuple[str, ...] = ()
    horizons: Tuple[int, ...] = ()
    edos: Tuple[int, ...] = ()
    seeds: Tuple[int, ...] = ()
    retry_failed: bool = False


def resolve_out_dir(out_dir: str) -> Path:
    text = (out_dir or "").strip()
    if not text:
        raise PreflightError("Choose an output directory.")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def build_cli_command(action: str, request: BenchmarkRequest, python: Optional[str] = None) -> List[str]:
    """The exact ``python -m aimusic.evaluation`` argv for an action: run | resume | list | report."""
    py = python or sys.executable
    out = str(resolve_out_dir(request.out_dir))
    base = [py, "-m", "aimusic.evaluation"]
    if action == "report":
        return base + ["report", "--out", out]
    config = str(config_path_for(request.config))
    cmd = base + ["run", "--config", config, "--out", out]
    if action in ("run", "resume"):
        cmd += ["--workers", str(max(1, int(request.workers)))]
        if request.retry_failed:
            cmd.append("--retry-failed")
    elif action == "list":
        cmd.append("--dry-run")
    else:
        raise ValueError(f"unknown action {action!r}")
    for flag, values in (
        ("--conditions", request.conditions), ("--horizons", request.horizons),
        ("--edos", request.edos), ("--seeds", request.seeds),
    ):
        if values:
            cmd += [flag, *map(str, values)]
    return cmd


def preflight(action: str, request: BenchmarkRequest) -> None:
    """Refuse bad requests up front: unknown config, directory from a different config, resume of nothing."""
    out = resolve_out_dir(request.out_dir)
    if action == "report":
        if not (out / "manifest.json").exists():
            raise PreflightError(f"No manifest.json in {out}; nothing to report on.")
        return
    cfg_path = config_path_for(request.config)
    try:
        config = load_config(cfg_path)
    except (OSError, ValueError) as exc:
        raise PreflightError(f"Cannot load benchmark config {cfg_path.name}: {exc}") from exc
    problems = validate_filter_selection(config, request.conditions, request.horizons, request.edos, request.seeds)
    if problems:
        raise PreflightError(format_validation_error(config, cfg_path.name, problems))
    manifest = _read_json(out / "manifest.json")
    if manifest is not None and manifest.get("config_hash") != config.config_hash and action != "list":
        raise PreflightError(
            f"{out} already holds results for a different config ({manifest.get('config_name')!r}). "
            "Pick another output directory; existing results are never overwritten."
        )
    if action == "resume" and manifest is None:
        raise PreflightError(f"Nothing to resume: {out} has no manifest.json. Use Run Benchmark instead.")


# --------------------------------------------------------------------------------------
# Run plan: what Run / Resume would do with the current selection (pure; never raises)
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class OutputDirState:
    has_manifest: bool
    foreign_config: Optional[str]  # config name stored in the directory when it differs from this one
    finished: int  # selected runs that already have a result and would be skipped
    finished_not_ok: int  # ...of which did not finish with status "ok"
    pending: int  # selected runs that would be executed


@dataclass(frozen=True)
class RunPlan:
    config_label: str
    metadata: Optional[ConfigMetadata]
    errors: Tuple[str, ...]
    selected_runs: int
    matrix_text: str
    out_dir: str
    output_state: Optional[OutputDirState]
    action_text: str
    estimated_runtime_s: Optional[float]

    @property
    def ok(self) -> bool:
        return not self.errors


def _count(n: int, singular: str, plural: Optional[str] = None) -> str:
    return f"{n} {singular if n == 1 else (plural or singular + 's')}"


def matrix_text(config: BenchmarkConfig, selection: FilterSelection, selected_runs: int) -> str:
    """``2 conditions × 1 horizon × 2 seeds × 1 EDO = 4 runs`` from the effective (empty = all) selection."""
    choices = available_filter_choices(config)
    n_cond = len(selection.conditions or choices.conditions)
    n_hor = len(selection.horizons or choices.horizons)
    n_seed = len(selection.seeds or choices.seeds)
    n_edo = len(selection.edos or choices.edos)
    factors = [
        _count(n_cond, "condition"), _count(n_hor, "horizon"), _count(n_seed, "seed"), _count(n_edo, "EDO", "EDO"),
    ]
    product = n_cond * n_hor * n_seed * n_edo
    if len(config.path_modes) > 1:
        factors.append(_count(len(config.path_modes), "path mode"))
        product *= len(config.path_modes)
    if len(config.graph_grid) > 1:
        factors.append(_count(len(config.graph_grid), "graph setting"))
        product *= len(config.graph_grid)
    text = f"{' × '.join(factors)} = {_count(selected_runs, 'run')}"
    if product != selected_runs:
        text += " (combinations the config does not define are skipped)"
    return text


def _finished_wall_times(out: Path) -> Dict[Tuple[int, int], List[float]]:
    """Wall times of finished-ok runs already in ``out``, keyed by (horizon, edo)."""
    walls: Dict[Tuple[int, int], List[float]] = {}
    runs = out / "runs"
    if not runs.is_dir():
        return walls
    for path in runs.glob("*.json"):
        if path.name.startswith("."):
            continue
        res = _read_json(path)
        if not res or res.get("status") != STATUS_OK:
            continue
        spec, wall = res.get("spec") or {}, (res.get("runtime") or {}).get("wall_s")
        if _finite(wall) and isinstance(spec.get("horizon"), int) and isinstance(spec.get("edo"), int):
            walls.setdefault((spec["horizon"], spec["edo"]), []).append(float(wall))
    return walls


def estimate_runtime_s(pending: Sequence[RunSpec], out: Path, workers: int) -> Optional[float]:
    """Rough wall time for ``pending`` from finished runs with the same horizon and EDO in ``out``.

    ``None`` unless every pending run has such a reference: no guess is made from other horizons.
    """
    if not pending:
        return 0.0
    walls = _finished_wall_times(out)
    total = 0.0
    for spec in pending:
        ref = walls.get((spec.horizon, spec.edo))
        if not ref:
            return None
        total += sum(ref) / len(ref)
    return total / max(1, workers)


def inspect_output_dir(
    config: BenchmarkConfig, specs: Sequence[RunSpec], out: Path, retry_failed: bool
) -> Tuple[OutputDirState, List[RunSpec]]:
    """State of ``out`` for these specs, plus the specs a Run/Resume would actually execute."""
    manifest = _read_json(out / "manifest.json")
    foreign = None
    if manifest is not None and manifest.get("config_hash") != config.config_hash:
        foreign = str(manifest.get("config_name"))
    pending: List[RunSpec] = []
    finished = finished_bad = 0
    for spec in specs:
        prior = _read_json(out / "runs" / f"{spec.run_id}.json")
        # Same skip rule as matrix.run_matrix: a result is final unless it failed and --retry-failed is set.
        if prior is not None and (prior.get("status") == STATUS_OK or not retry_failed):
            finished += 1
            finished_bad += prior.get("status") != STATUS_OK
        else:
            pending.append(spec)
    return OutputDirState(manifest is not None, foreign, finished, finished_bad, len(pending)), pending


def _action_text(state: OutputDirState, total: int, retry_failed: bool) -> str:
    if state.foreign_config is not None:
        return "Blocked: the output directory holds results for a different config."
    if state.finished == 0:
        return f"Start new runs: {_count(total, 'run')} will be executed."
    if state.pending == 0:
        text = f"Nothing to run: all {_count(total, 'selected run')} already finished and will be skipped."
    else:
        text = (
            f"Resume: {state.finished} already finished will be skipped, "
            f"{_count(state.pending, 'run')} will be executed."
        )
    if state.finished_not_ok and not retry_failed:
        text += f" {state.finished_not_ok} of the finished runs did not succeed; tick “Retry failed runs” to re-run them."
    return text


def plan_run(request: BenchmarkRequest) -> RunPlan:
    """Preview of a Run / Resume for the current form values.  Problems are reported, never raised."""
    label = Path(request.config).name if request.config else "(no config)"
    errors: List[str] = []
    try:
        out: Optional[Path] = resolve_out_dir(request.out_dir)
    except PreflightError as exc:
        out = None
        errors.append(str(exc))
    out_text = str(out) if out is not None else "-"
    try:
        config = _load_named_config(request.config)
    except PreflightError as exc:
        return RunPlan(label, None, tuple(errors + [str(exc)]), 0, "", out_text, None, "", None)
    metadata = config_metadata(config, label)
    selection = FilterSelection(request.conditions, request.horizons, request.edos, request.seeds)
    problems = validate_filter_selection(config, request.conditions, request.horizons, request.edos, request.seeds)
    if problems:
        errors.append(format_validation_error(config, label, problems))
        return RunPlan(label, metadata, tuple(errors), 0, "", out_text, None, "", None)
    specs = _selected_specs(config, request.conditions, request.horizons, request.edos, request.seeds)
    text = matrix_text(config, selection, len(specs))
    if out is None:
        return RunPlan(label, metadata, tuple(errors), len(specs), text, out_text, None, "", None)
    state, pending = inspect_output_dir(config, specs, out, request.retry_failed)
    estimate: Optional[float] = None
    if state.foreign_config is None:
        estimate = estimate_runtime_s(pending, out, request.workers)
    else:
        errors.append(
            f"{out} already holds results for a different config ({state.foreign_config!r}). "
            "Pick another output directory; existing results are never overwritten."
        )
    return RunPlan(
        label, metadata, tuple(errors), len(specs), text, out_text, state,
        _action_text(state, len(specs), request.retry_failed), estimate,
    )


def render_run_preview(plan: RunPlan) -> str:
    """Markdown shown under the filters: the exact matrix, the directory state and what Run would do."""
    blocks: List[str] = []
    for err in plan.errors:
        blocks.append("\n".join(f"> {line}" if line else ">" for line in ("**⚠ Cannot start**", "", *err.splitlines())))
    if plan.selected_runs and plan.matrix_text:
        blocks.append(f"**Selected runs:** {plan.matrix_text}")
    state = plan.output_state
    if state is not None:
        if state.pending == 0 and state.foreign_config is None:
            est = "none (nothing to execute)"
        elif plan.estimated_runtime_s is None:
            est = "not available yet (needs finished runs with the same horizon and EDO in this directory)"
        else:
            est = f"about {_fmt_duration(plan.estimated_runtime_s)} (rough, from finished runs in this directory)"
        existing = (
            f"{state.finished} of {plan.selected_runs} selected runs already finished"
            if state.has_manifest or state.finished else "new or empty"
        )
        blocks.append(
            f"- **Output directory:** `{plan.out_dir}`\n"
            f"- **Directory state:** {existing}\n"
            f"- **This will:** {plan.action_text}\n"
            f"- **Estimated runtime:** {est}"
        )
    return "\n\n".join(blocks) if blocks else "_Choose a benchmark config._"


class BenchmarkJob:
    """One CLI subprocess whose stdout+stderr go to a log file; never blocks the caller."""

    def __init__(self, command: Sequence[str], out_dir: Path, action: str):
        self.command = list(command)
        self.out_dir = out_dir
        self.action = action
        self.log_path = out_dir / LOG_NAME
        self.started_at = time.time()
        self.finished_at: Optional[float] = None
        self.stopped = False
        self._proc: Optional[subprocess.Popen[bytes]] = None

    def start(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        # Worker subprocesses run with cwd=<out dir>; make sure they can still import aimusic.
        env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
        env["PYTHONUNBUFFERED"] = "1"
        log = open(self.log_path, "ab")
        log.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {self.action}: {' '.join(self.command)}\n".encode())
        log.flush()
        try:
            kwargs: Dict[str, Any] = {"start_new_session": True} if os.name == "posix" else {}
            self._proc = subprocess.Popen(
                self.command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, **kwargs
            )
        finally:
            log.close()

    @property
    def returncode(self) -> Optional[int]:
        if self._proc is None:
            return None
        rc = self._proc.poll()
        if rc is not None and self.finished_at is None:
            self.finished_at = time.time()
        return rc

    @property
    def running(self) -> bool:
        return self._proc is not None and self.returncode is None

    def stop(self) -> None:
        """Terminate the CLI and its worker processes; finished runs stay on disk so Resume continues."""
        if not self.running or self._proc is None:
            return
        self.stopped = True
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(self._proc.pid), signal.SIGTERM)
            else:  # pragma: no cover - non-posix
                self._proc.terminate()
        except (ProcessLookupError, PermissionError):
            pass
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                if os.name == "posix":
                    os.killpg(os.getpgid(self._proc.pid), signal.SIGKILL)
                else:  # pragma: no cover
                    self._proc.kill()
            except (ProcessLookupError, PermissionError):
                pass
        self.finished_at = self.finished_at or time.time()

    def wait(self, timeout: Optional[float] = None) -> Optional[int]:
        if self._proc is None:
            return None
        try:
            self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
        return self.returncode

    def status_text(self) -> str:
        rc = self.returncode
        elapsed = (self.finished_at or time.time()) - self.started_at
        if rc is None and self._proc is not None:
            return f"Running ({self.action}, {elapsed:.0f}s)"
        if self.stopped:
            return f"Stopped after {elapsed:.0f}s; use Resume to continue"
        if rc == 0:
            return f"Finished ({self.action}, {elapsed:.0f}s)"
        return f"Exited with code {rc} ({self.action}); see the log"

    def log_tail(self, max_bytes: int = 20000) -> str:
        try:
            with open(self.log_path, "rb") as handle:
                handle.seek(0, os.SEEK_END)
                size = handle.tell()
                handle.seek(max(0, size - max_bytes))
                return handle.read().decode("utf-8", errors="replace")
        except OSError:
            return ""


class JobRegistry:
    """Latest job per output directory (local, single-user tool).  Thread-safe."""

    def __init__(self) -> None:
        self._jobs: Dict[str, BenchmarkJob] = {}
        self._lock = threading.Lock()

    def get(self, out_dir: Path) -> Optional[BenchmarkJob]:
        with self._lock:
            return self._jobs.get(str(out_dir))

    def running_jobs(self) -> List[BenchmarkJob]:
        with self._lock:
            return [j for j in self._jobs.values() if j.running]

    def start(self, action: str, request: BenchmarkRequest) -> BenchmarkJob:
        """Validate, then launch ``action`` via the CLI.  Raises :class:`PreflightError` when refused."""
        preflight(action, request)
        out = resolve_out_dir(request.out_dir)
        with self._lock:
            existing = self._jobs.get(str(out))
            if existing is not None and existing.running:
                raise PreflightError(f"A benchmark is already running in {out}. Stop it first.")
            job = BenchmarkJob(build_cli_command(action, request), out, action)
            job.start()
            self._jobs[str(out)] = job
        return job


JOBS = JobRegistry()


def run_cli_blocking(action: str, request: BenchmarkRequest, timeout: float = 300.0) -> Tuple[int, str]:
    """Run a short CLI action (list / report) to completion and return (returncode, output)."""
    preflight(action, request)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    proc = subprocess.run(
        build_cli_command(action, request), cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=timeout, check=False
    )
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def run_benchmark_from_ui(action: str, request: BenchmarkRequest) -> Tuple[BenchmarkJob, str]:
    """Entry point used by the Run / Resume buttons.  Returns the job and a human status line."""
    job = JOBS.start(action, request)
    return job, job.status_text()


# --------------------------------------------------------------------------------------
# Gradio page
# --------------------------------------------------------------------------------------
def _parse_int_list(text: str, label: str) -> Tuple[int, ...]:
    out: List[int] = []
    for token in (text or "").replace(",", " ").split():
        try:
            out.append(int(token))
        except ValueError as exc:
            raise PreflightError(f"{label}: '{token}' is not an integer.") from exc
    return tuple(out)


_DATASET_CACHE: Dict[str, Tuple[Tuple[Any, ...], EvaluationDataset]] = {}


def _dir_signature(out: Path) -> Tuple[Any, ...]:
    def mt(p: Path) -> float:
        try:
            return p.stat().st_mtime_ns
        except OSError:
            return 0

    runs = out / "runs"
    n, newest = 0, 0
    if runs.is_dir():
        with os.scandir(runs) as it:
            for entry in it:
                if entry.name.endswith(".json") and not entry.name.startswith("."):
                    n += 1
                    newest = max(newest, entry.stat().st_mtime_ns)
    return (mt(out / "manifest.json"), mt(out / "aggregate.json"), mt(out / "report.md"), n, newest)


def load_dataset_cached(out_dir: Path | str) -> EvaluationDataset:
    out = Path(out_dir)
    sig = _dir_signature(out)
    hit = _DATASET_CACHE.get(str(out))
    if hit and hit[0] == sig:
        return hit[1]
    dataset = load_benchmark_output(out)
    _DATASET_CACHE[str(out)] = (sig, dataset)
    return dataset


def build_evaluation_page(demo: Any = None) -> None:
    """Create the Evaluation components.  Call inside ``with demo.route("Evaluation"):`` (or any Blocks)."""
    import gradio as gr

    configs = discover_benchmark_configs()
    default_config = "m1_smoke.json" if "m1_smoke.json" in configs else (configs[0] if configs else None)
    try:
        first = switch_config(default_config) if default_config else None
    except PreflightError:
        first = None
    first_choices = first.choices if first else FilterChoices()

    def _hz(values: Sequence[int]) -> List[Tuple[str, int]]:
        return [(f"{v} beats", v) for v in values]

    def _edo(values: Sequence[int]) -> List[Tuple[str, int]]:
        return [(f"{v}-EDO", v) for v in values]

    gr.Markdown(
        "## Evaluation dashboard\n"
        "Runs and views MUS-28 benchmarks through the existing `python -m aimusic.evaluation` CLI. "
        "Charts read `manifest.json`, the raw run results and `aggregate.json`; the Markdown report is a reference view."
    )
    with gr.Group():
        with gr.Row():
            config_dd = gr.Dropdown(
                label="Benchmark config", choices=config_dropdown_choices(), value=default_config, scale=2,
                info="SMOKE = quick development check, PROFILE = scaling probe, BENCHMARK = main result matrix.",
            )
            out_dir_tb = gr.Textbox(
                label="Output directory",
                value=str((DEFAULT_OUT_ROOT / (Path(default_config).stem if default_config else "run")).relative_to(REPO_ROOT)),
                info="Relative paths are resolved from the repository root.", scale=3,
            )
            workers_num = gr.Number(label="Workers", value=1, precision=0, minimum=1, scale=1, info=TERM_HELP["Workers"])
        config_panel = gr.HTML(render_config_panel(first.metadata if first else None))
        with gr.Row():
            conditions_cb = gr.CheckboxGroup(
                label="Conditions (none = all in this config)", choices=list(first_choices.conditions), value=[],
                info=TERM_HELP["Condition"],
            )
            horizons_cb = gr.CheckboxGroup(
                label="Horizons (none = all)", choices=_hz(first_choices.horizons), value=[], info=TERM_HELP["Horizon"]
            )
            edos_cb = gr.CheckboxGroup(
                label="EDO (none = all)", choices=_edo(first_choices.edos), value=[], info=TERM_HELP["EDO"]
            )
            seeds_cb = gr.CheckboxGroup(
                label="Seeds (none = all)", choices=list(first_choices.seeds), value=[], info=TERM_HELP["Seed"]
            )
        retry_cb = gr.Checkbox(label="Retry failed runs", value=False)
        preview_md = gr.Markdown(
            render_run_preview(plan_run(BenchmarkRequest(default_config or "", out_dir_tb.value)))  # type: ignore[arg-type]
        )
        with gr.Row():
            list_btn = gr.Button("List Runs")
            run_btn = gr.Button("Run Benchmark", variant="primary")
            resume_btn = gr.Button("Resume Benchmark")
            stop_btn = gr.Button("Stop", variant="stop")
            report_btn = gr.Button("Generate Report")
            load_btn = gr.Button("Load Existing Results")
    with gr.Accordion("Terminology: what the benchmark settings mean", open=False):
        gr.Markdown(render_terminology_markdown())
    status_md = gr.Markdown("Idle.")
    cards = gr.HTML(render_overview_cards(None))
    notes_md = gr.Markdown(visible=False)
    with gr.Accordion("Command output / run log", open=False):
        log_tb = gr.Textbox(label="stdout / stderr", lines=12, max_lines=24, interactive=False, autoscroll=True)

    with gr.Tab("Overview"):
        with gr.Row():
            runtime_metric = gr.Dropdown(
                label="Metric", choices=[(metric_label(m), m) for m in RUNTIME_CHART_METRICS], value="runtime.wall_s", scale=1
            )
        runtime_plot = gr.Plot(label="Runtime")
        status_plot = gr.Plot(label="Run status")
    with gr.Tab("Ablations"):
        ablation_cell = gr.Dropdown(label="Horizon / EDO", choices=[], value=None)
        ablation_plot = gr.Plot(label="Ablation deltas")
    with gr.Tab("Tension"):
        tension_cell = gr.Dropdown(label="Horizon / EDO", choices=[], value=None)
        tension_plot = gr.Plot(label="Tension tracking")
        tension_note = gr.Markdown()
    with gr.Tab("Scaling"):
        scaling_plot = gr.Plot(label="Runtime and memory scaling")
    with gr.Tab("EDO comparison"):
        edo_horizon = gr.Dropdown(label="Horizon", choices=[], value=None)
        edo_plot = gr.Plot(label="12-EDO vs 19-EDO")
    with gr.Tab("Run details"):
        run_dd = gr.Dropdown(label="Run", choices=[], value=None, filterable=True)
        run_plot = gr.Plot(label="Selected run")
        run_md = gr.Markdown("_Select a run to inspect it._")
    with gr.Tab("Report"):
        with gr.Accordion("report.md (reference view)", open=False):
            report_md = gr.Markdown("_No report loaded._")

    timer = gr.Timer(2.0, active=False)
    sig_state = gr.State("")

    # ---- handlers ---------------------------------------------------------------------
    def _request(config, out_dir, workers, conditions, horizons, edos, seeds, retry) -> BenchmarkRequest:
        return BenchmarkRequest(
            config=config or "", out_dir=out_dir, workers=int(workers or 1), conditions=tuple(conditions or ()),
            horizons=tuple(int(v) for v in horizons or ()), edos=tuple(int(v) for v in edos or ()),
            seeds=tuple(int(v) for v in seeds or ()), retry_failed=bool(retry),
        )

    dash_outputs = [
        cards, notes_md, runtime_plot, status_plot, ablation_cell, ablation_plot, tension_cell, tension_plot, tension_note,
        scaling_plot, edo_horizon, edo_plot, run_dd, report_md,
    ]

    def _cell_value(h: Optional[int], e: Optional[int]) -> Optional[str]:
        return None if h is None or e is None else f"{h}|{e}"

    def _parse_cell(value: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
        try:
            h, e = (value or "").split("|")
            return int(h), int(e)
        except ValueError:
            return None, None

    def refresh(out_dir: str, metric: str = "runtime.wall_s", job_status: str = ""):
        """Rebuild every dashboard component from disk.  Missing/partial/failed data never raises."""
        try:
            out = resolve_out_dir(out_dir)
            ds = load_dataset_cached(out)
        except (FileNotFoundError, PreflightError) as exc:
            empty = _empty_figure(str(exc))
            return (
                render_overview_cards(None), gr.update(value=f"**{exc}**", visible=True), empty, empty,
                gr.update(choices=[], value=None), empty, gr.update(choices=[], value=None), empty, "",
                empty, gr.update(choices=[], value=None), empty, gr.update(choices=[], value=None), "_No report loaded._",
            )
        summary = summarize_results(ds)
        warn = list(ds.warnings)
        if ds.aggregate_source == "recomputed":
            warn.append("Aggregates were computed live from the raw runs.")
        ab = comparison_keys(ds)
        ab_choices = [(label, _cell_value(h, e)) for h, e, label in ab]
        tc = tension_cells(ds)
        tc_choices = [(label, _cell_value(h, e)) for h, e, label in tc]
        h0, e0 = (ab[0][0], ab[0][1]) if ab else (None, None)
        th0, te0 = (tc[0][0], tc[0][1]) if tc else (None, None)
        horizons = sorted({c["horizon"] for c in ds.cells.values()})
        tension = tension_chart_data(ds, th0, te0)
        return (
            render_overview_cards(summary, job_status),
            gr.update(value="\n\n".join(f"- {w}" for w in warn), visible=bool(warn)),
            make_runtime_chart(ds, metric), make_status_chart(ds),
            gr.update(choices=ab_choices, value=_cell_value(h0, e0) if ab else None), make_ablation_chart(ds, h0, e0),
            gr.update(choices=tc_choices, value=_cell_value(th0, te0) if tc else None),
            make_tension_chart(ds, th0, te0), " ".join(tension.notes),
            make_scaling_chart(ds),
            gr.update(choices=horizons, value=horizons[-1] if horizons else None), make_edo_chart(ds, horizons[-1] if horizons else None),
            gr.update(choices=run_choices(ds), value=None),
            ds.report_text or "_No report.md in this directory yet (it is written when a run finishes or via Generate Report)._",
        )

    def on_load(out_dir, metric):
        return (*refresh(out_dir, metric), _dir_signature_str(out_dir))

    def _dir_signature_str(out_dir: str) -> str:
        try:
            return repr(_dir_signature(resolve_out_dir(out_dir)))
        except PreflightError:
            return ""

    def guard(fn):
        """Surface refusals/errors as a status line instead of crashing the interface."""
        def wrapper(*args):
            try:
                return fn(*args)
            except PreflightError as exc:
                return (f"**Refused:** {exc}",) + (gr.skip(),) * 3
            except Exception as exc:  # noqa: BLE001 - never crash the UI on a benchmark problem
                return (f"**Error:** {type(exc).__name__}: {exc}",) + (gr.skip(),) * 3
        return wrapper

    req_inputs = [config_dd, out_dir_tb, workers_num, conditions_cb, horizons_cb, edos_cb, seeds_cb, retry_cb]

    def do_start(action: str):
        def fn(*args):
            request = _request(*args)
            job, status = run_benchmark_from_ui(action, request)
            return status, job.log_tail(), gr.Timer(active=True), ""
        return fn

    start_outputs = [status_md, log_tb, timer, sig_state]
    run_btn.click(guard(do_start("run")), req_inputs, start_outputs)
    resume_btn.click(guard(do_start("resume")), req_inputs, start_outputs)

    def do_list(*args):
        request = _request(*args)
        code, text = run_cli_blocking("list", request)
        return (f"List finished (exit {code})." if code == 0 else f"**List failed (exit {code}).**"), text, gr.skip(), gr.skip()

    list_btn.click(guard(do_list), req_inputs, start_outputs)

    def do_report(*args):
        request = _request(*args)
        code, text = run_cli_blocking("report", request)
        return (f"Report written (exit {code})." if code == 0 else f"**Report failed (exit {code}).**"), text, gr.skip(), ""

    report_btn.click(guard(do_report), req_inputs, start_outputs).then(
        on_load, [out_dir_tb, runtime_metric], dash_outputs + [sig_state]
    )

    def do_stop(out_dir: str):
        job = JOBS.get(resolve_out_dir(out_dir))
        if job is None or not job.running:
            return "No benchmark is running for this directory.", gr.skip()
        job.stop()
        return job.status_text(), job.log_tail()

    stop_btn.click(do_stop, [out_dir_tb], [status_md, log_tb])
    load_btn.click(
        lambda out: ("Loaded existing results.", ""), [out_dir_tb], [status_md, log_tb]
    ).then(on_load, [out_dir_tb, runtime_metric], dash_outputs + [sig_state])

    def tick(out_dir: str, metric: str, last_sig: str):
        out = resolve_out_dir(out_dir)
        job = JOBS.get(out)
        if job is None:
            return (gr.Timer(active=False), "Idle.", "", *(gr.skip() for _ in dash_outputs), last_sig)
        status = job.status_text()
        sig = _dir_signature_str(out_dir)
        done = not job.running
        if sig != last_sig or done:
            try:
                dash = refresh(out_dir, metric, job_status=status)
            except Exception:  # noqa: BLE001 - mid-write files can be briefly unreadable; retry next tick
                dash = tuple(gr.skip() for _ in dash_outputs)
        else:
            dash = tuple(gr.skip() for _ in dash_outputs)
            # keep the status card text current even when nothing else changed
        return (gr.Timer(active=not done), status, job.log_tail(), *dash, sig)

    timer.tick(tick, [out_dir_tb, runtime_metric, sig_state], [timer, status_md, log_tb] + dash_outputs + [sig_state])

    # ---- chart selectors (reload from disk cache; cheap) ---------------------------------
    runtime_metric.change(
        lambda out, m: make_runtime_chart(load_dataset_cached(resolve_out_dir(out)), m), [out_dir_tb, runtime_metric], [runtime_plot]
    )
    ablation_cell.change(
        lambda out, cell: make_ablation_chart(load_dataset_cached(resolve_out_dir(out)), *_parse_cell(cell)),
        [out_dir_tb, ablation_cell], [ablation_plot],
    )

    def on_tension(out, cell):
        ds = load_dataset_cached(resolve_out_dir(out))
        h, e = _parse_cell(cell)
        return make_tension_chart(ds, h, e), " ".join(tension_chart_data(ds, h, e).notes)

    tension_cell.change(on_tension, [out_dir_tb, tension_cell], [tension_plot, tension_note])
    edo_horizon.change(
        lambda out, h: make_edo_chart(load_dataset_cached(resolve_out_dir(out)), int(h) if h is not None else None),
        [out_dir_tb, edo_horizon], [edo_plot],
    )

    def on_run(out, run_id):
        ds = load_dataset_cached(resolve_out_dir(out))
        res = find_run(ds, run_id)
        return run_tension_figure(res), render_run_details(res)

    run_dd.change(on_run, [out_dir_tb, run_dd], [run_plot, run_md])

    # Keep the output directory in step with the chosen config unless the user typed their own.
    def on_config(config: Optional[str], current: str, conditions, horizons, edos, seeds):
        default_names = {str((DEFAULT_OUT_ROOT / Path(c).stem).relative_to(REPO_ROOT)) for c in configs}
        out_dir = (
            str((DEFAULT_OUT_ROOT / Path(config).stem).relative_to(REPO_ROOT)) if config and current in default_names else current
        )
        try:
            sw = switch_config(
                config or "",
                FilterSelection(
                    tuple(conditions or ()), tuple(int(v) for v in horizons or ()),
                    tuple(int(v) for v in edos or ()), tuple(int(v) for v in seeds or ()),
                ),
            )
        except PreflightError as exc:
            empty = gr.update(choices=[], value=[])
            return out_dir, render_config_panel(None, str(exc)), empty, empty, empty, empty
        return (
            out_dir, render_config_panel(sw.metadata),
            gr.update(choices=list(sw.choices.conditions), value=list(sw.selection.conditions)),
            gr.update(choices=_hz(sw.choices.horizons), value=list(sw.selection.horizons)),
            gr.update(choices=_edo(sw.choices.edos), value=list(sw.selection.edos)),
            gr.update(choices=list(sw.choices.seeds), value=list(sw.selection.seeds)),
        )

    def on_preview(*args):
        return render_run_preview(plan_run(_request(*args)))

    config_dd.change(
        on_config, [config_dd, out_dir_tb, conditions_cb, horizons_cb, edos_cb, seeds_cb],
        [out_dir_tb, config_panel, conditions_cb, horizons_cb, edos_cb, seeds_cb],
    ).then(on_preview, req_inputs, [preview_md])
    gr.on(
        triggers=[c.change for c in (out_dir_tb, workers_num, conditions_cb, horizons_cb, edos_cb, seeds_cb, retry_cb)],
        fn=on_preview, inputs=req_inputs, outputs=[preview_md],
    )
