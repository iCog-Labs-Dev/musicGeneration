"""Dashboard views for the MUS-16 section-wise evaluation (MUS-28 follow-up).

Pure data builders (no plotly), Markdown renderers and plotly figures, all reading the raw run results
and ``aggregate.json`` that the evaluator writes.  Single-pass results carry no section metrics; for
them every view says so (``SECTION_WISE_UNAVAILABLE``) and shows no values, never zeros.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

from aimusic.evaluation import ui as base
from aimusic.evaluation.ui import EvaluationDataset, Point

SECTION_WISE_UNAVAILABLE = "Section-wise metrics unavailable: this run used single-pass planning."
NO_SECTION_WISE_RUNS = "Section-wise metrics unavailable: no section-wise run in this output (all cells used single-pass planning)."

# Colours for the two sectioning modes (fixed slots from the dashboard palette).
SECTIONING_COLORS = {"single_pass": base.CONDITION_COLORS["full"], "section_wise": base.CONDITION_COLORS["no_gttm"]}
SECTION_STATUS_COLORS = {
    "ok": base.STATUS_COLORS[base.STATUS_OK], "failed": base.STATUS_COLORS[base.STATUS_FAILED],
    "not_reached": base.STATUS_COLORS[base.STATUS_NOT_RUN], "unknown": base.STATUS_COLORS[base.STATUS_CRASHED],
}
LEVEL_LABELS = (
    ("sections_solved", "all sections solved"), ("joins_valid", "all joins valid"),
    ("path_assembled", "whole-song path assembled"),
)
COMPARISON_METRICS = (
    "runtime.wall_s", "runtime.peak_rss_mb", "tension_section_plan.mean_absolute_error",
    "tension_section_plan.shape_correlation", "path.distinct_state_fraction", "path.chord_change_rate",
    "gttm.mean_gttm_score", "sb.effective_entropy",
)
base.METRIC_INFO.update({
    "tension_section_plan.mean_absolute_error": ("Tension MAE (section plan)", "", "lower"),
    "tension_section_plan.shape_correlation": ("Tension correlation (section plan)", "", "higher"),
    "sb.effective_entropy": ("SB effective entropy", "nats", None),
})
CONTINUITY_METRICS = (
    ("meter_continuity", "meter"), ("groove_continuity", "groove"), ("role_continuity", "role"),
    ("head_continuity", "head"), ("boundary_continuity", "boundary"),
)


def sectioning_of(result_or_cell: Mapping[str, Any]) -> str:
    spec = result_or_cell.get("spec")
    source = spec if isinstance(spec, Mapping) else result_or_cell
    return str(source.get("sectioning", "single_pass"))


def section_wise_results(dataset: EvaluationDataset) -> List[Dict[str, Any]]:
    return [r for r in dataset.results if sectioning_of(r) == "section_wise"]


def section_wise_cells(dataset: EvaluationDataset) -> List[Tuple[str, Dict[str, Any]]]:
    cells = [(k, c) for k, c in dataset.cells.items() if c.get("sectioning") == "section_wise"]
    return sorted(cells, key=lambda kc: base._cell_sort(kc[1]))


def _cell_name(cell: Mapping[str, Any]) -> str:
    return base._cell_label(cell, True, False)


def has_section_data(result: Mapping[str, Any]) -> bool:
    return sectioning_of(result) == "section_wise" and bool(result.get("section_names"))


# --------------------------------------------------------------------------------------
# Data builders
# --------------------------------------------------------------------------------------
@dataclass
class ComparisonChartData:
    metric: str
    categories: List[str]
    series: Dict[str, List[Optional[Point]]]  # sectioning mode -> one Point per matched cell
    ci_levels: List[str]  # "paired_ci" | "descriptive" per category
    notes: List[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not any(p is not None for pts in self.series.values() for p in pts)


def comparison_data(dataset: EvaluationDataset, metric: str = "runtime.wall_s") -> ComparisonChartData:
    """Monolithic vs section-wise value of ``metric`` for every matched cell (never pooled)."""
    entries = list((dataset.aggregate.get("sectioning_comparison") or {}).values())
    categories: List[str] = []
    series: Dict[str, List[Optional[Point]]] = {"single_pass": [], "section_wise": []}
    levels: List[str] = []
    for entry in sorted(entries, key=lambda e: base._cell_sort(dataset.cells[e["section_wise_cell"]])):
        sp, sw = dataset.cells.get(entry["single_pass_cell"]), dataset.cells.get(entry["section_wise_cell"])
        if not sp or not sw:
            continue
        categories.append(_cell_name(sw))
        series["single_pass"].append(base._point(sp["metrics"].get(metric)))
        series["section_wise"].append(base._point(sw["metrics"].get(metric)))
        levels.append(entry["ci_level"])
    notes: List[str] = []
    if not entries:
        notes.append("No matched single-pass / section-wise cells in this output.")
    elif any(level != "paired_ci" for level in levels):
        notes.append("Descriptive comparison: fewer matched seeds than needed for a paired confidence interval "
                     f"({entries[0]['min_pairs_for_ci']}). Whiskers are per-mode 95% t-intervals over seeds, not a test.")
    return ComparisonChartData(metric, categories, series, levels, notes)


@dataclass
class SectionRuntimeData:
    categories: List[str]  # "<cell> / <section>"
    graph: List[Optional[Point]]
    solve: List[Optional[Point]]

    @property
    def empty(self) -> bool:
        return not any(p is not None for p in (*self.graph, *self.solve))


def section_runtime_data(dataset: EvaluationDataset) -> SectionRuntimeData:
    cats: List[str] = []
    graph: List[Optional[Point]] = []
    solve: List[Optional[Point]] = []
    cells = section_wise_cells(dataset)
    multi = len(cells) > 1
    for _, cell in cells:
        for name in cell.get("section_names") or []:
            cats.append(f"{_cell_name(cell)} / {name}" if multi else name)
            graph.append(base._point(cell["metrics"].get(f"section.{name}.runtime.graph_build_s")))
            solve.append(base._point(cell["metrics"].get(f"section.{name}.runtime.bridge_solve_s")))
    return SectionRuntimeData(cats, graph, solve)


@dataclass
class SectionStatusData:
    sections: List[str]  # "<cell> / <section>"
    counts: Dict[str, List[int]]  # status -> count per section
    levels: Dict[str, int]  # level -> runs passing it
    runs: int


def section_status_data(dataset: EvaluationDataset) -> SectionStatusData:
    """Raw per-section status (ok / failed / not_reached / unknown) over every section-wise run."""
    keys: List[str] = []
    per: Dict[str, Dict[str, int]] = {}
    levels = {name: 0 for name, _ in LEVEL_LABELS}
    runs = section_wise_results(dataset)
    multi = len({r["spec"].get("horizon") for r in runs}) > 1
    for r in sorted(runs, key=lambda r: (r["spec"]["horizon"], r["spec"]["seed"])):
        names = r.get("section_names") or []
        statuses = r.get("per_section_status") or []
        prefix = f"{r['spec']['horizon']}b / " if multi else ""
        for name, status in zip(names, statuses):
            label = prefix + name
            if label not in per:
                keys.append(label)
                per[label] = {}
            per[label][status] = per[label].get(status, 0) + 1
        st = r.get("section_wise_status") or {}
        for level, _ in LEVEL_LABELS:
            levels[level] += 1 if st.get(level) else 0
    order = ["ok", "failed", "not_reached", "unknown"]
    seen = [s for s in order if any(s in v for v in per.values())]
    seen += sorted({s for v in per.values() for s in v} - set(order))
    return SectionStatusData(keys, {s: [per[k].get(s, 0) for k in keys] for s in seen}, levels, len(runs))


@dataclass
class StitchChartData:
    joins: List[str]  # "<cell> / a -> b"
    continuity: Dict[str, List[Optional[Point]]]  # component -> point per join (1 = identical)
    jumps: Dict[str, List[Optional[Point]]]  # key / chord distance jump
    cost: List[Optional[Point]]
    tolerance: List[Optional[float]]

    @property
    def empty(self) -> bool:
        return not any(p is not None for p in self.cost)


def stitch_chart_data(dataset: EvaluationDataset) -> StitchChartData:
    joins: List[str] = []
    continuity: Dict[str, List[Optional[Point]]] = {label: [] for _, label in CONTINUITY_METRICS}
    jumps: Dict[str, List[Optional[Point]]] = {"key": [], "chord": []}
    cost: List[Optional[Point]] = []
    tol: List[Optional[float]] = []
    cells = section_wise_cells(dataset)
    for _, cell in cells:
        names = cell.get("section_names") or []
        for i in range(max(len(names) - 1, 0)):
            joins.append(f"{_cell_name(cell)} / {names[i]} → {names[i + 1]}" if len(cells) > 1 else f"{names[i]} → {names[i + 1]}")
            m = cell["metrics"]
            for metric, label in CONTINUITY_METRICS:
                continuity[label].append(base._point(m.get(f"join.{i}.{metric}")))
            jumps["key"].append(base._point(m.get(f"join.{i}.key_distance_jump")))
            jumps["chord"].append(base._point(m.get(f"join.{i}.chord_distance_jump")))
            cost.append(base._point(m.get(f"join.{i}.stitching_cost")))
            tol_pt = base._point(m.get(f"join.{i}.tolerance"))
            tol.append(tol_pt.mean if tol_pt else None)
    return StitchChartData(joins, continuity, jumps, cost, tol)


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def make_comparison_chart(dataset: EvaluationDataset, metric: str = "runtime.wall_s") -> Any:
    data = comparison_data(dataset, metric)
    title = f"Monolithic vs section-wise: {base.metric_label(metric)}"
    if data.empty:
        return base._empty_figure(NO_SECTION_WISE_RUNS if not section_wise_cells(dataset) else "No matched successful runs.", title)
    go = base._go()
    unit = base.METRIC_INFO.get(metric, ("", "", None))[1]
    fig = go.Figure()
    for mode, pts in data.series.items():
        ys, plus, minus = base._error_arrays(pts)
        fig.add_bar(
            name=mode, x=data.categories, y=ys,
            error_y={"type": "data", "symmetric": False, "array": plus, "arrayminus": minus, "thickness": 1.4, "width": 3},
            marker={"color": SECTIONING_COLORS[mode]}, customdata=base._hover(pts, f" {unit}".rstrip()),
            hovertemplate=f"<b>{mode}</b><br>%{{x}}<br>%{{customdata}}<extra></extra>",
        )
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.06)
    fig.update_xaxes(title="matched cell (horizon / EDO / path mode)")
    fig.update_yaxes(title=base.metric_label(metric))
    if metric.startswith("runtime"):
        fig.update_yaxes(rangemode="tozero")
    return base._style(fig, title)


def make_section_runtime_chart(dataset: EvaluationDataset) -> Any:
    data = section_runtime_data(dataset)
    title = "Per-section runtime (graph build + bridge solve)"
    if data.empty:
        return base._empty_figure(NO_SECTION_WISE_RUNS, title)
    go = base._go()
    fig = go.Figure()
    for name, pts, color in (("graph build", data.graph, "#2a78d6"), ("bridge solve", data.solve, "#eb6834")):
        ys, plus, minus = base._error_arrays(pts)
        fig.add_bar(
            name=name, x=data.categories, y=ys, marker={"color": color},
            error_y={"type": "data", "symmetric": False, "array": plus, "arrayminus": minus, "thickness": 1.2, "width": 3},
            customdata=base._hover(pts, " s"), hovertemplate=f"<b>{name}</b><br>%{{x}}<br>%{{customdata}}<extra></extra>",
        )
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="section")
    fig.update_yaxes(title="seconds", rangemode="tozero")
    return base._style(fig, title)


def make_section_status_chart(dataset: EvaluationDataset) -> Any:
    data = section_status_data(dataset)
    title = "Section status across section-wise runs"
    if not data.runs:
        return base._empty_figure(NO_SECTION_WISE_RUNS, title)
    fig = make_subplots_pair()
    for status, counts in data.counts.items():
        fig.add_bar(
            name=status.replace("_", " "), x=data.sections, y=counts, row=1, col=1,
            marker={"color": SECTION_STATUS_COLORS.get(status, base.INK)},
            hovertemplate=f"{status.replace('_', ' ')}: %{{y}} run(s)<extra>%{{x}}</extra>",
        )
    labels = [label for _, label in LEVEL_LABELS]
    fig.add_bar(
        name="runs passing", x=labels, y=[data.levels[k] for k, _ in LEVEL_LABELS], row=1, col=2,
        marker={"color": base.STATUS_COLORS[base.STATUS_OK]}, showlegend=False,
        hovertemplate=f"%{{y}} of {data.runs} run(s)<extra>%{{x}}</extra>",
    )
    fig.add_hline(y=data.runs, line_dash="dot", line_color=base.INK, row=1, col=2)
    fig.update_layout(barmode="stack")
    fig.update_yaxes(title="runs", rangemode="tozero", row=1, col=1)
    fig.update_yaxes(title=f"runs (of {data.runs})", rangemode="tozero", row=1, col=2)
    return base._style(fig, title)


def make_subplots_pair() -> Any:
    return base._make_subplots(rows=1, cols=2, column_widths=[0.62, 0.38], subplot_titles=("Per-section status", "Success levels"))


def make_stitch_chart(dataset: EvaluationDataset) -> Any:
    data = stitch_chart_data(dataset)
    title = "Stitching continuity at section joins"
    if data.empty:
        return base._empty_figure(NO_SECTION_WISE_RUNS, title)
    go = base._go()
    fig = base._make_subplots(
        rows=1, cols=2, column_widths=[0.62, 0.38],
        subplot_titles=("Continuity (1 = identical, worst of both edges)", "Stitching cost vs tolerance"),
    )
    palette = ("#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#eda100")
    for (label, pts), color in zip(data.continuity.items(), palette):
        ys, plus, minus = base._error_arrays(pts)
        fig.add_bar(
            name=label, x=data.joins, y=ys, marker={"color": color}, row=1, col=1,
            error_y={"type": "data", "symmetric": False, "array": plus, "arrayminus": minus, "thickness": 1, "width": 2},
            customdata=base._hover(pts, ""), hovertemplate=f"<b>{label} continuity</b><br>%{{x}}<br>%{{customdata}}<extra></extra>",
        )
    ys, plus, minus = base._error_arrays(data.cost)
    fig.add_bar(
        name="stitching cost", x=data.joins, y=ys, marker={"color": "#8a8a85"}, row=1, col=2,
        error_y={"type": "data", "symmetric": False, "array": plus, "arrayminus": minus, "thickness": 1.2, "width": 3},
        customdata=base._hover(data.cost, ""), hovertemplate="<b>cost</b><br>%{x}<br>%{customdata}<extra></extra>",
    )
    fig.add_trace(go.Scatter(
        name="tolerance (max_cost)", x=data.joins, y=data.tolerance, mode="lines+markers",
        line={"color": base.STATUS_COLORS[base.STATUS_FAILED], "dash": "dash"},
    ), row=1, col=2)
    fig.update_layout(barmode="group", bargap=0.2)
    fig.update_yaxes(title="continuity", range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(title="cost", rangemode="tozero", row=1, col=2)
    return base._style(fig, title, height=460)


# --------------------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------------------
def _p(point: Optional[Point], digits: int = 3) -> str:
    return "-" if point is None else base._ep_val(point.mean, digits)


def render_comparison_markdown(dataset: EvaluationDataset) -> str:
    entries = list((dataset.aggregate.get("sectioning_comparison") or {}).values())
    if not section_wise_cells(dataset):
        return f"_{NO_SECTION_WISE_RUNS}_"
    if not entries:
        return "_No `single_pass` cell matches a `section_wise` cell, so there is no monolithic comparison._"
    rows = ["| matched cell | seeds paired | single_pass ok | section_wise ok | only sp ok | only sw ok | comparison |", "|---|---|---|---|---|---|---|"]
    for e in entries:
        sw = dataset.cells[e["section_wise_cell"]]
        sp = dataset.cells[e["single_pass_cell"]]
        p = e["status_pairs"]
        kind = "paired CI" if e["ci_level"] == "paired_ci" else "descriptive"
        rows.append(f"| {_cell_name(sw)} | {e['n_pairs']} | {sp['n_ok']}/{sp['n_runs']} | {sw['n_ok']}/{sw['n_runs']} | "
                    f"{p['only_single_pass_ok']} | {p['only_section_wise_ok']} | {kind} |")
    return "\n".join(rows) + (
        "\n\nSection-wise succeeding is not evidence that it is better: compare the runtime, memory, tension, "
        "path and join figures above; they trade off."
    )


def render_join_table(dataset: EvaluationDataset) -> str:
    """Join-detail view over all section-wise cells (mean over successful seeds)."""
    cells = section_wise_cells(dataset)
    if not cells:
        return f"_{NO_SECTION_WISE_RUNS}_"
    rows = ["| cell | join | beat | key jump | chord jump | meter | groove | role | head | boundary | cost | tolerance | within tolerance |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, cell in cells:
        names = cell.get("section_names") or []
        m = cell["metrics"]
        for i in range(max(len(names) - 1, 0)):
            g = lambda n, i=i, m=m: _p(base._point(m.get(f"join.{i}.{n}")))  # noqa: E731
            rows.append(
                f"| {_cell_name(cell)} | {names[i]} → {names[i + 1]} | {g('time_index')} | {g('key_distance_jump')} | "
                f"{g('chord_distance_jump')} | {g('meter_continuity')} | {g('groove_continuity')} | {g('role_continuity')} | "
                f"{g('head_continuity')} | {g('boundary_continuity')} | {g('stitching_cost')} | {g('tolerance')} | "
                f"{g('within_tolerance')} |"
            )
    return "\n".join(rows) + "\n\nContinuity = 1 - mismatch (1 = identical); jumps and cost are mismatch (0 = identical)."


# ---- run inspector --------------------------------------------------------------------
def section_choices(result: Optional[Mapping[str, Any]]) -> List[str]:
    return list(result.get("section_names") or []) if result and has_section_data(result) else []


def _levels_table(result: Mapping[str, Any]) -> str:
    st = result.get("section_wise_status") or {}
    rows = [(label, st.get(key)) for key, label in LEVEL_LABELS]
    rows += [("failure class", st.get("failure_class")), ("failed section", st.get("failed_section")),
             ("failed planner stage", st.get("failed_stage"))]
    return base._kv_table(rows)


def render_section_wise_run(result: Optional[Mapping[str, Any]]) -> str:
    """The section-wise block of the run inspector (summary, per-section table, joins)."""
    if not result:
        return ""
    if not has_section_data(result):
        return f"_{SECTION_WISE_UNAVAILABLE}_"
    names = result["section_names"]
    parts = [
        f"Sections: {', '.join(f'`{n}`' for n in names)} ({result.get('section_count')} sections, horizons "
        f"{', '.join(str(h) for h in result.get('section_horizons') or [])} beats).", "",
        "**Success levels**", "", _levels_table(result), "",
        "**Per section**", "",
        "| section | beats | status | runtime s | layers | retained | SB iters | converged | tension MAE | tension corr |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    results = result.get("per_section_results") or []
    status = result.get("per_section_status") or []
    runtime = result.get("per_section_runtime") or []
    graphs = result.get("per_section_graph_metrics") or []
    sbs = result.get("per_section_sb_metrics") or []
    for i, name in enumerate(names):
        rec = results[i] if i < len(results) else {}
        g = (graphs[i] if i < len(graphs) else None) or {}
        s = (sbs[i] if i < len(sbs) else None) or {}
        t = (rec or {}).get("tension") or {}
        rt = (runtime[i] if i < len(runtime) else None) or {}
        v = base._fmt_value
        parts.append(
            f"| {name} | {v(rec.get('n_beats'))} | {status[i] if i < len(status) else '-'} | {v(rt.get('runtime_s'))} | "
            f"{v(g.get('layer_count'))} | {v(g.get('retained_count'))} | {v(s.get('iterations'))} | {v(s.get('converged'))} | "
            f"{v(t.get('mean_absolute_error'))} | {v(t.get('shape_correlation'))} |"
        )
    parts += ["", "**Joins**", "", render_run_joins(result)]
    return "\n".join(parts)


def render_run_joins(result: Mapping[str, Any]) -> str:
    joins = ((result.get("join_metrics") or {}).get("joins")) or []
    if not has_section_data(result):
        return f"_{SECTION_WISE_UNAVAILABLE}_"
    if not joins:
        return "_No join metrics: the run failed before its joins were validated._"
    v = base._fmt_value
    rows = ["| join | beat | key jump | chord jump | meter | groove | role | head | boundary | cost | tolerance | within | shared endpoint ok |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for j in joins:
        rows.append(
            f"| {j['left_section']} → {j['right_section']} | {j['time_index']} | {v(j['key_distance_jump'])} | "
            f"{v(j['chord_distance_jump'])} | {v(j['meter_continuity'])} | {v(j['groove_continuity'])} | "
            f"{v(j['role_continuity'])} | {v(j['head_continuity'])} | {v(j['boundary_continuity'])} | "
            f"{v(j['stitching_cost'])} | {v(j['tolerance'])} | {v(j['within_tolerance'])} | {v(j.get('shared_endpoint_consistent'))} |"
        )
    return "\n".join(rows)


def render_section_details(result: Optional[Mapping[str, Any]], section: Optional[str]) -> str:
    """Endpoint, SB, graph, path and tension details of one section of one run."""
    if not result:
        return "_Select a run to inspect it._"
    if not has_section_data(result):
        return f"_{SECTION_WISE_UNAVAILABLE}_"
    names = result["section_names"]
    if section not in names:
        return "_Select a section._"
    i = names.index(section)

    def at(key: str) -> Mapping[str, Any]:
        items = result.get(key) or []
        return (items[i] if i < len(items) else None) or {}

    rec = at("per_section_results")
    parts = [f"#### Section `{section}`", ""]
    parts.append(base._kv_table([
        ("status", (result.get("per_section_status") or ["-"] * len(names))[i]),
        ("beats", f"{rec.get('start_beat')} to {rec.get('end_beat')} ({rec.get('n_beats')} beats)"),
        ("RNG stream", rec.get("rng_stream_id")), ("path SHA-256", rec.get("path_sha256")),
    ]))
    if rec.get("failure"):
        parts += ["", "**Failure**", "", base._kv_table(list(rec["failure"].items()))]
    for title, key in (("Runtime", "per_section_runtime"), ("SB", "per_section_sb_metrics"),
                       ("Graph", "per_section_graph_metrics"), ("Path", "per_section_path_metrics")):
        block = at(key)
        rows = base._scalar_rows(block) if block else []
        parts += ["", f"**{title}**", "", base._kv_table(rows) if rows else "_not available (the section did not complete)_"]
    ep = at("per_section_endpoint_metrics")
    parts += ["", "**Endpoint masses**", ""]
    if ep:
        rows = [("policy", ep.get("policy")), ("request modified", ep.get("request_modified")),
                ("termination", ep.get("termination_reason")), ("bridge converged", ep.get("bridge_converged"))]
        for side in ("start", "terminal"):
            sec = ep.get(side) or {}
            rows += [(f"{side} boundary kind", sec.get("boundary_kind")),
                     (f"{side} requested / effective / realized mass",
                      " / ".join(base._ep_val(sec.get(k)) for k in ("requested_mass", "effective_mass", "realized_mass"))),
                     (f"{side} marginal L1 vs effective", sec.get("marginal_l1_vs_effective"))]
        parts.append(base._kv_table(rows))
    else:
        parts.append("_not available (the section did not complete)_")
    tension = {k: v for k, v in (rec.get("tension") or {}).items() if not isinstance(v, list)}
    parts += ["", "**Tension (whole-song curve sliced to this section)**", "",
              base._kv_table(list(tension.items())) if tension else "_not available_"]
    return "\n".join(parts)


def section_tension_figure(result: Optional[Mapping[str, Any]], section: Optional[str]) -> Any:
    """Target vs realized tension for one section of one run."""
    title = f"Tension: {section}" if section else "Tension"
    if not result or not has_section_data(result) or section not in result["section_names"]:
        return base._empty_figure(SECTION_WISE_UNAVAILABLE if result else "Select a run.", title)
    rec = (result.get("per_section_results") or [{}] * 9)[result["section_names"].index(section)] or {}
    tension = rec.get("tension") or {}
    target, realized = tension.get("target_curve"), tension.get("realized_curve")
    if not target or not realized:
        return base._empty_figure("No tension curve for this section (it did not complete).", title)
    go = base._go()
    start = rec.get("start_beat", 0)
    xs = [start + k for k in range(len(target))]
    fig = go.Figure()
    fig.add_scatter(x=xs, y=target, name="target", mode="lines+markers", line={"color": base.INK, "dash": "dash"})
    fig.add_scatter(x=xs, y=realized, name="realized", mode="lines+markers", line={"color": SECTIONING_COLORS["section_wise"]})
    fig.update_xaxes(title="beat")
    fig.update_yaxes(title="tension", range=[0, 1.05])
    return base._style(fig, title, height=300)
