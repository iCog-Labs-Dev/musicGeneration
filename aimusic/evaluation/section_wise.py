"""MUS-16 section-wise measurement for the MUS-28 evaluation suite.

Everything here is *derived* from the public MUS-16 result structures that
``aimusic.planning.plans.run_method_a`` already returns for a section-wise run
(:class:`SectionWisePlanResult`, :class:`SectionRunResult`,
:class:`~aimusic.planning.stitching.SectionJoinDiagnostics`,
:class:`~aimusic.core.diagnostics.SBDiagnostics`).  No section planning, boundary
sharing, graph construction, SB solving or stitching is reimplemented, and the
planner is never modified.

Three success levels are kept distinct, because a song can pass one and fail the next:

* ``sections_solved``  every per-section graph + SB solve succeeded
* ``joins_valid``      every shared boundary is within the configured stitching tolerance
* ``path_assembled``   the whole-song path was assembled from the section paths

A section-wise run is never flattened into a fake monolithic SB result: section solves stay
per-section, and the whole-song aggregates are explicitly labelled as such.

Per-section wall time is not part of the planner's result, so :class:`SectionTimers` measures it by
wrapping the two public solver entry points that ``plans.py`` calls once per section
(``build_sparse_graph`` and ``solve_endpoint_bridge``) for the duration of one run.  The wrappers
only time and delegate; the planner code path is unchanged.  Per-section *peak memory* is not
measurable (the process high-water mark is shared), so only whole-run memory is recorded.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from aimusic.evaluation import endpoint_metrics as ep

STATUS_SECTION_OK = "ok"
STATUS_SECTION_FAILED = "failed"
STATUS_SECTION_NOT_REACHED = "not_reached"
STATUS_SECTION_UNKNOWN = "unknown"

# Failure classes, derived from the stage annotated on ``SectionPlanningError``.
CLASS_BOUNDARY = "boundary_planning"
CLASS_SECTION_SOLVE = "section_solve"
CLASS_JOIN = "join"
CLASS_PATH = "path_assembly"
CLASS_EVALUATION = "evaluation_error"
CLASS_OTHER = "section_wise_error"
FAILURE_CLASSES = (CLASS_BOUNDARY, CLASS_SECTION_SOLVE, CLASS_JOIN, CLASS_PATH, CLASS_EVALUATION, CLASS_OTHER)

# ``plans.run_section_wise`` stage names -> failure class.
_STAGE_CLASS = {
    "boundary planning": CLASS_BOUNDARY,
    "graph construction": CLASS_SECTION_SOLVE,
    "endpoint bridge solve": CLASS_SECTION_SOLVE,
    "shared boundary validation": CLASS_JOIN,
    "join validation": CLASS_JOIN,
    "joint path extraction": CLASS_PATH,
    "path extraction": CLASS_PATH,
}

STITCH_COMPONENTS = ("key", "chord", "meter", "groove", "role", "head", "boundary")
CATEGORICAL_COMPONENTS = ("meter", "groove", "role", "head", "boundary")


def failure_kind(failure_class: str) -> str:
    """Stable ``failure.kind`` (also the key of the report's failure-kind counts)."""
    return f"{failure_class}_failed"


# --------------------------------------------------------------------------------------------
# Section plan
# --------------------------------------------------------------------------------------------
def authored_section_plan(run_config: Any, section_names: Sequence[str]) -> Tuple[Any, ...]:
    """The authored section plan for ``section_names`` over ``run_config.total_beats``.

    Uses the public ``build_section_plan`` with the section-wise strategy, so a single-pass run can be
    scored against exactly the targets a section-wise run plans for.
    """
    from aimusic.core.config import SectioningStrategy
    from aimusic.planning.plans import build_section_plan

    plan_config = replace(
        run_config.plan_config,
        sectioning_strategy=SectioningStrategy.SECTION_WISE,
        section_names=tuple(section_names),
    )
    return tuple(build_section_plan(replace(run_config, plan_config=plan_config)))


def sectioning_record(sections: Sequence[Any]) -> Dict[str, Any]:
    """``section_names`` / ``section_count`` / ``section_horizons`` of a planned section tuple."""
    return {
        "section_names": [s.name for s in sections],
        "section_count": len(sections),
        "section_horizons": [int(s.end_time - s.start_time) for s in sections],
    }


def section_identity(index: int, section: Any) -> Dict[str, Any]:
    return {
        "index": index,
        "name": section.name,
        "start_beat": int(section.start_time),
        "end_beat": int(section.end_time),
        "n_beats": int(section.end_time - section.start_time),
        "boundary_level": int(section.boundary_level),
    }


# --------------------------------------------------------------------------------------------
# Per-section timing (instrumentation only)
# --------------------------------------------------------------------------------------------
class SectionTimers:
    """Context manager that times the per-section graph build and bridge solve.

    ``plans.run_section_wise`` builds every section graph first and then solves each section, each in
    section order, so the *i*-th call of each wrapper belongs to section *i*.  Calls made before a
    failure are still recorded.  The original functions are restored on exit.
    """

    def __init__(self) -> None:
        self.graph_s: List[float] = []
        self.solve_s: List[float] = []
        self._restore: List[Tuple[Any, str, Any]] = []

    def __enter__(self) -> "SectionTimers":
        import aimusic.planning.plans as plans

        def wrap(name: str, sink: List[float]) -> None:
            original = getattr(plans, name)

            def timed(*args: Any, **kwargs: Any) -> Any:
                start = time.perf_counter()
                try:
                    return original(*args, **kwargs)
                finally:
                    sink.append(time.perf_counter() - start)

            self._restore.append((plans, name, original))
            setattr(plans, name, timed)

        wrap("build_sparse_graph", self.graph_s)
        wrap("solve_endpoint_bridge", self.solve_s)
        return self

    def __exit__(self, *exc_info: Any) -> None:
        for module, name, original in reversed(self._restore):
            setattr(module, name, original)
        self._restore.clear()

    def runtime(self, index: int) -> Dict[str, Optional[float]]:
        graph = self.graph_s[index] if index < len(self.graph_s) else None
        solve = self.solve_s[index] if index < len(self.solve_s) else None
        measured = [v for v in (graph, solve) if v is not None]
        return {
            "graph_build_s": graph,
            "bridge_solve_s": solve,
            "runtime_s": sum(measured) if measured else None,
        }

    @property
    def total_graph_s(self) -> float:
        return float(sum(self.graph_s))

    @property
    def total_solve_s(self) -> float:
        return float(sum(self.solve_s))


# --------------------------------------------------------------------------------------------
# Stitching / joins
# --------------------------------------------------------------------------------------------
def _cost_dict(cost: Any) -> Dict[str, float]:
    out = {name: float(getattr(cost, name)) for name in STITCH_COMPONENTS}
    out["total"] = float(cost.total)
    return out


def join_record(
    index: int,
    join: Any,
    *,
    left_path: Optional[Sequence[Any]] = None,
    right_path: Optional[Sequence[Any]] = None,
    whole_path: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    """Continuity metrics of one section join, from its ``SectionJoinDiagnostics``.

    The planner's ``StitchCost`` components are mismatch measures in [0, 1].  The *worst* of the
    incoming (left section's last transition) and outgoing (right section's first transition) edge is
    reported, so a bad join cannot hide behind a good side.  Categorical features report
    ``1 - mismatch`` as a continuity score.  ``within_tolerance`` is recomputed here from the recorded
    costs and tolerance, so a tolerance violation is visible even if a planner stopped enforcing it.
    """
    incoming, outgoing = _cost_dict(join.incoming), _cost_dict(join.outgoing)
    worst = {name: max(incoming[name], outgoing[name]) for name in (*STITCH_COMPONENTS, "total")}
    cost = worst["total"]
    tolerance = float(join.tolerance)
    probs = [float(p) for p in join.boundary_probabilities]
    record: Dict[str, Any] = {
        "index": index,
        "left_section": join.left_section,
        "right_section": join.right_section,
        "time_index": int(join.time_index),
        "shared_state": join.shared_state.to_dict(),
        "boundary_support_size": int(join.boundary_support_size),
        "boundary_entropy": ep.entropy_nats(probs) if probs else None,
        "incoming": incoming,
        "outgoing": outgoing,
        "key_distance_jump": worst["key"],
        "chord_distance_jump": worst["chord"],
        "meter_continuity": 1.0 - worst["meter"],
        "groove_continuity": 1.0 - worst["groove"],
        "role_continuity": 1.0 - worst["role"],
        "head_continuity": 1.0 - worst["head"],
        "boundary_continuity": 1.0 - worst["boundary"],
        "stitching_cost_incoming": incoming["total"],
        "stitching_cost_outgoing": outgoing["total"],
        "stitching_cost": cost,
        "tolerance": tolerance,
        "tolerance_margin": tolerance - cost,
        "within_tolerance": bool(cost <= tolerance),
    }
    if left_path is not None and right_path is not None:
        consistent = left_path[-1] == right_path[0] == join.shared_state
        if whole_path is not None:
            consistent = consistent and whole_path[int(join.time_index)] == join.shared_state
        record["shared_endpoint_consistent"] = bool(consistent)
    return record


def _mean(values: Sequence[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def join_summary(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Whole-song aggregate over joins (explicit ``None`` when there are no joins)."""
    if not records:
        return {"join_count": 0}
    names = (
        "key_distance_jump", "chord_distance_jump", "meter_continuity", "groove_continuity",
        "role_continuity", "head_continuity", "boundary_continuity", "stitching_cost",
        "boundary_support_size", "boundary_entropy",
    )
    summary: Dict[str, Any] = {"join_count": len(records)}
    for name in names:
        values = [float(r[name]) for r in records if r.get(name) is not None]
        summary[f"{name}_mean"] = _mean(values)
        # Worst case: the largest jump / cost, the smallest continuity.
        if values:
            summary[f"{name}_worst"] = min(values) if name.endswith("continuity") else max(values)
    inside = [bool(r["within_tolerance"]) for r in records]
    summary["within_tolerance_count"] = sum(inside)
    summary["within_tolerance_rate"] = sum(inside) / len(inside)
    summary["all_within_tolerance"] = all(inside)
    summary["min_tolerance_margin"] = min(float(r["tolerance_margin"]) for r in records)
    consistent = [bool(r["shared_endpoint_consistent"]) for r in records if "shared_endpoint_consistent" in r]
    if consistent:
        summary["all_shared_endpoints_consistent"] = all(consistent)
    return summary


def build_join_metrics(plan: Any) -> Dict[str, Any]:
    results = plan.section_results
    records = [
        join_record(
            i, join,
            left_path=results[i].path, right_path=results[i + 1].path, whole_path=plan.path,
        )
        for i, join in enumerate(plan.joins)
    ]
    config = plan.run_config.plan_config.stitching
    return {
        "stitching_config": {
            "max_cost": float(config.max_cost),
            "weights": {name: float(getattr(config, f"{name}_weight")) for name in STITCH_COMPONENTS},
        },
        "joins": records,
        "summary": join_summary(records),
    }


# --------------------------------------------------------------------------------------------
# Tension against the authored section plan
# --------------------------------------------------------------------------------------------
def tension_on_section_plan(
    path: Sequence[Any], vocabularies: Any, edo: int, sections: Sequence[Any]
) -> Dict[str, Any]:
    """Tension error of a whole-song path against the authored per-section targets.

    The realized curve is computed once over the whole path (a section's first beat is scored against
    the previous section's last state, exactly as in the monolithic run) and then sliced per section,
    so section-level numbers are comparable between ``single_pass`` and ``section_wise`` runs.
    """
    from aimusic.scoring.tension import (
        compare_tension_curves,
        realized_tension_curve,
        target_tension_curve,
    )

    decoded = list(path)[:-1]
    realized = realized_tension_curve(decoded, vocabularies, edo)
    target = target_tension_curve(sections)
    if not target:
        return {}
    whole = compare_tension_curves(target, realized, sections)
    per_section: Dict[str, Any] = {}
    for section in sections:
        lo, hi = section.start_time, section.end_time
        t_slice = [(t, v) for t, v in target if lo <= t < hi]
        r_slice = [(t, v) for t, v in realized if lo <= t < hi]
        report = compare_tension_curves(t_slice, r_slice, (section,))
        values = [v for _, v in r_slice]
        per_section[section.name] = {
            "mean_absolute_error": report.mean_absolute_error,
            "max_absolute_error": report.max_absolute_error,
            "shape_correlation": report.shape_correlation,
            "peak_timing_offset": report.peak_timing_offset,
            "realized_mean": sum(values) / len(values) if values else None,
            "target_curve": [round(float(v), 6) for _, v in t_slice],
            "realized_curve": [round(float(v), 6) for v in values],
        }
    return {
        "mean_absolute_error": whole.mean_absolute_error,
        "max_absolute_error": whole.max_absolute_error,
        "shape_correlation": whole.shape_correlation,
        "peak_timing_offset": whole.peak_timing_offset,
        "per_section": per_section,
    }


# --------------------------------------------------------------------------------------------
# Whole-song aggregates of per-section diagnostics
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class WholeSongView:
    """The whole-song endpoint reports of a section-wise plan, in the shape the metrics expect.

    The outer start comes from the first section, the outer terminal from the last; interior
    boundaries are shared and reported per section.
    """

    start_mass: Any
    end_mass: Any
    diagnostics: Any


def whole_song_view(plan: Any) -> WholeSongView:
    return WholeSongView(
        start_mass=plan.section_results[0].start_mass,
        end_mass=plan.section_results[-1].end_mass,
        diagnostics=plan.diagnostics,
    )


def combine_graph_metrics(plan: Any, per_section: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Whole-song graph metrics from the per-section ones (counts summed, fractions recomputed).

    Shared boundary layers are counted once in the layer widths (the plan's own
    ``diagnostics.graph_layer_sizes``); candidate and edge counts are sums over the independent
    section graphs.
    """
    widths = [int(w) for w in plan.diagnostics.graph_layer_sizes]
    total = lambda name: sum(int(s[name]) for s in per_section)  # noqa: E731
    ratio = lambda a, b: (float(a) / float(b)) if b else None  # noqa: E731
    raw, legal, unique, scored = total("proposed_count"), total("legal_count"), total("unique_count"), total("scored_count")
    raw_edges, kept_edges = total("raw_edge_count"), total("retained_edge_count")
    reasons: Dict[str, int] = {}
    for s in per_section:
        for key, count in s["prune_reason_counts"].items():
            reasons[key] = reasons.get(key, 0) + int(count)
    inactive = set(per_section[0]["inactive_features"])
    for s in per_section[1:]:
        inactive &= set(s["inactive_features"])
    return {
        "layer_count": len(widths),
        "layer_width_mean": sum(widths) / len(widths),
        "layer_width_min": min(widths),
        "layer_width_max": max(widths),
        "proposed_count": raw,
        "legal_count": legal,
        "unique_count": unique,
        "scored_count": scored,
        "retained_count": total("retained_count"),
        "raw_edge_count": raw_edges,
        "retained_edge_count": kept_edges,
        "outdegree_pruned_count": total("outdegree_pruned_count"),
        "legal_fraction": ratio(legal, raw),
        "unique_fraction": ratio(unique, legal),
        "scored_fraction": ratio(scored, legal),
        "retention_fraction": ratio(kept_edges, raw_edges),
        "prune_reason_counts": dict(sorted(reasons.items())),
        "inactive_feature_count": len(inactive),
        "inactive_features": sorted(inactive),
    }


def whole_song_sb_metrics(plan: Any, per_section: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Whole-song SB summary via the public ``SBDiagnostics.from_plan`` aggregator.

    ``effective_entropy`` is the planner's own aggregate (mean node-marginal entropy over the combined
    layers, shared boundaries counted once); it summarizes independent section solves and is NOT the
    entropy of a joint whole-song bridge.
    """
    from aimusic.core.diagnostics import SBDiagnostics

    diag = SBDiagnostics.from_plan(plan)
    return {
        "used_for_path": True,
        "iterations": diag.iterations_run,
        "converged": diag.converged,
        "final_max_delta": diag.final_max_delta,
        "effective_entropy": diag.effective_entropy,
        "zero_degree_pruned_nodes": diag.pruned_nodes,
        "sections_converged": sum(1 for s in per_section if s["converged"]),
        "iterations_max_section": max(int(s["iterations"]) for s in per_section),
    }


def whole_song_endpoint_metrics(plan: Any, view: WholeSongView, *, marginal_tolerance: float, mode: str) -> Dict[str, Any]:
    """Outer-endpoint metrics of the whole song (interior boundaries are per-section)."""
    if mode == "singleton":
        info = ep.singleton_endpoint_info(view)
        info["scope"] = "whole_song_outer_endpoints"
        return info
    start = ep.side_metrics(view.start_mass, marginal_tolerance)
    terminal = ep.side_metrics(view.end_mass, marginal_tolerance)
    statuses = {start["accounting_status"], terminal["accounting_status"]}
    overall = ep.ACCOUNTING_NO_MODIFICATION
    for candidate in (ep.ACCOUNTING_BLENDED, ep.ACCOUNTING_TRIMMED, ep.ACCOUNTING_REMOVED, ep.ACCOUNTING_REMOVED_TRIMMED):
        if candidate in statuses:
            overall = candidate
    if len(statuses - {ep.ACCOUNTING_NO_MODIFICATION}) > 1:
        overall = ep.ACCOUNTING_REMOVED_TRIMMED
    diag = plan.diagnostics
    metrics: Dict[str, Any] = {
        "status": "ok",
        "scope": "whole_song_outer_endpoints",
        "generation_method": "method_a_distribution_endpoints",
        "endpoint_mode": "distribution",
        "policy": plan.run_config.unreachable_policy.value,
        "accounting_status": overall,
        "request_modified": bool(diag.endpoint_request_modified),
        "modifications": list(diag.endpoint_modifications),
        "termination_reason": "per_section",
        "marginal_tolerance": float(marginal_tolerance),
        "bridge_converged": bool(diag.bridge_converged),
        "endpoint_support_size_start": start["support_size_effective"],
        "endpoint_support_size_terminal": terminal["support_size_effective"],
        "endpoint_support_size_requested_start": start["support_size_requested"],
        "endpoint_support_size_requested_terminal": terminal["support_size_requested"],
        "requested_start_mass": start["requested_mass"],
        "requested_terminal_mass": terminal["requested_mass"],
        "feasible_start_mass": start["feasible_mass"],
        "feasible_terminal_mass": terminal["feasible_mass"],
        "removed_start_mass": start["removed_mass"],
        "removed_terminal_mass": terminal["removed_mass"],
        "trimmed_start_mass": start["trimmed_mass"],
        "trimmed_terminal_mass": terminal["trimmed_mass"],
        "effective_start_mass": start["effective_mass"],
        "effective_terminal_mass": terminal["effective_mass"],
        "realized_start_mass": start["realized_mass"],
        "realized_terminal_mass": terminal["realized_mass"],
        "start": start,
        "terminal": terminal,
    }
    # No repeated draws are made in section-wise runs, so calibration is explicitly unavailable.
    ep.apply_calibration(metrics, ep.calibration_unavailable(ep.CAL_NO_SAMPLES, 0))
    return metrics


def require_section_diagnostics(plan: Any, expected_mode: str) -> None:
    """Fail loudly if the section-wise plan lacks the MUS-15/16 diagnostics evaluation needs."""
    mode = getattr(plan.diagnostics, "endpoint_mode", None)
    if mode != expected_mode:
        raise ep.EndpointDiagnosticsUnavailable(
            f"plan reports endpoint_mode={mode!r} but the run requested {expected_mode!r}; "
            "refusing to evaluate under a different mode."
        )
    for index, result in enumerate(plan.section_results):
        for name in ("start_mass", "end_mass"):
            report = getattr(result, name, None)
            if report is None or not report.realized_probabilities:
                raise ep.EndpointDiagnosticsUnavailable(
                    f"section {index} '{result.section.name}' has no realized {name} marginal; "
                    "per-section endpoint metrics cannot be computed."
                )


# --------------------------------------------------------------------------------------------
# Per-section records
# --------------------------------------------------------------------------------------------
def _boundary_kind(index: int, count: int, side: str, mode: str) -> str:
    outer = index == 0 if side == "start" else index == count - 1
    if not outer:
        return "shared_boundary"
    return "outer_singleton" if mode == "singleton" else "outer_distribution"


def section_endpoint_metrics(result: Any, index: int, count: int, *, marginal_tolerance: float, mode: str) -> Dict[str, Any]:
    """Requested / effective / realized endpoint mass of one section (both of its boundaries).

    Interior boundaries are shared multi-state distributions in *both* endpoint modes, so the mass
    accounting is meaningful per section either way; ``boundary_kind`` says which side is which.
    """
    resolution = result.endpoint_resolution
    start = ep.side_metrics(result.start_mass, marginal_tolerance)
    terminal = ep.side_metrics(result.end_mass, marginal_tolerance)
    start["boundary_kind"] = _boundary_kind(index, count, "start", mode)
    terminal["boundary_kind"] = _boundary_kind(index, count, "terminal", mode)
    return {
        "status": "ok",
        "endpoint_mode": mode,
        "policy": resolution.policy.value,
        "request_modified": bool(resolution.request_modified),
        "modifications": list(resolution.modifications),
        "termination_reason": resolution.termination_reason,
        "marginal_tolerance": float(marginal_tolerance),
        "bridge_converged": bool(result.sb_solution.trace.converged),
        "start": start,
        "terminal": terminal,
    }


def section_blocks(
    plan: Any,
    planned: Sequence[Any],
    timers: SectionTimers,
    tension: Mapping[str, Any],
    *,
    marginal_tolerance: float,
    mode: str,
    graph_fn: Callable[[Any], Dict[str, Any]],
    sb_fn: Callable[..., Dict[str, Any]],
    path_fn: Callable[[Sequence[Any], Any, int], Dict[str, Any]],
    sha_fn: Callable[[Sequence[Any]], str],
) -> Dict[str, List[Any]]:
    """The ``per_section_*`` lists of a successful section-wise run (parallel to ``section_names``)."""
    blocks: Dict[str, List[Any]] = {
        "per_section_results": [], "per_section_status": [], "per_section_runtime": [],
        "per_section_graph_metrics": [], "per_section_sb_metrics": [], "per_section_endpoint_metrics": [],
        "per_section_path_metrics": [],
    }
    per_tension = tension.get("per_section", {}) if tension else {}
    count = len(plan.section_results)
    for index, result in enumerate(plan.section_results):
        identity = section_identity(index, result.section)
        horizon = identity["n_beats"]
        runtime = timers.runtime(index)
        blocks["per_section_results"].append({
            **identity,
            "status": STATUS_SECTION_OK,
            "failure": None,
            "rng_stream_id": result.rng_stream_id,
            "path_sha256": sha_fn(result.path),
            "path_score": result.path_score,
            "tension": per_tension.get(result.section.name, {}),
        })
        blocks["per_section_status"].append(STATUS_SECTION_OK)
        blocks["per_section_runtime"].append(runtime)
        blocks["per_section_graph_metrics"].append(graph_fn(result.graph))
        blocks["per_section_sb_metrics"].append(sb_fn(result.sb_solution, used=True))
        blocks["per_section_endpoint_metrics"].append(
            section_endpoint_metrics(result, index, count, marginal_tolerance=marginal_tolerance, mode=mode)
        )
        blocks["per_section_path_metrics"].append(path_fn(result.path, result, horizon))
    return blocks


# --------------------------------------------------------------------------------------------
# Structured failures
# --------------------------------------------------------------------------------------------
def planning_error_class(exc: BaseException) -> Optional[str]:
    """Failure class of a ``SectionPlanningError``; ``None`` for any other exception."""
    from aimusic.planning.plans import SectionPlanningError

    if not isinstance(exc, SectionPlanningError):
        return None
    return _STAGE_CLASS.get(exc.stage, CLASS_OTHER)


def section_failure_state(
    exc: BaseException, planned: Sequence[Any], timers: Optional[SectionTimers] = None
) -> Dict[str, Any]:
    """Section-level state of a failed section-wise run, derived from the annotated exception.

    ``run_section_wise`` raises on the first problem and returns no partial result, so only what the
    exception states is known: which section/stage failed, and (from the stage order inside the
    planner) which sections had already been solved.  Everything else is ``not_reached``; no metric is
    invented for a section that never produced one.
    """
    from aimusic.planning.plans import SectionPlanningError

    count = len(planned)
    statuses = [STATUS_SECTION_NOT_REACHED] * count
    failure_class = CLASS_EVALUATION
    stage = None
    index: Optional[int] = None
    sections_solved: Optional[bool] = False
    joins_valid: Optional[bool] = None
    path_assembled = False
    cause: Optional[BaseException] = None
    if isinstance(exc, SectionPlanningError):
        cause = exc.__cause__
        stage = exc.stage
        index = int(exc.section_index) if 0 <= exc.section_index < count else None
        failure_class = _STAGE_CLASS.get(stage, CLASS_OTHER)
        if stage == "graph construction" and index is not None:
            statuses[index] = STATUS_SECTION_FAILED
        elif stage == "endpoint bridge solve" and index is not None:
            statuses[:index] = [STATUS_SECTION_OK] * index
            statuses[index] = STATUS_SECTION_FAILED
        elif failure_class in (CLASS_JOIN, CLASS_PATH):
            # Every bridge was solved before joins are validated and paths assembled.
            statuses = [STATUS_SECTION_OK] * count
            sections_solved = True
            if failure_class == CLASS_JOIN:
                joins_valid = False
            if stage == "path extraction" and index is not None:
                statuses[index] = STATUS_SECTION_FAILED
                statuses[index + 1:] = [STATUS_SECTION_NOT_REACHED] * (count - index - 1)
        # Boundary planning precedes every solve: no section was reached.
    elif failure_class == CLASS_EVALUATION:
        statuses = [STATUS_SECTION_UNKNOWN] * count
        sections_solved = None
    rows: List[Dict[str, Any]] = []
    runtimes: List[Any] = []
    for i, section in enumerate(planned):
        failure = None
        if i == index and statuses[i] == STATUS_SECTION_FAILED:
            failure = {"stage": stage, "kind": type(cause).__name__ if cause else None}
        rows.append({**section_identity(i, section), "status": statuses[i], "failure": failure})
        runtimes.append(timers.runtime(i) if timers is not None else None)
    return {
        "failure_class": failure_class,
        "planner_stage": stage,
        "section_index": index,
        "section_name": planned[index].name if index is not None else None,
        "cause_kind": type(cause).__name__ if cause is not None else None,
        "status": {
            "sections_solved": sections_solved,
            "joins_valid": joins_valid,
            "path_assembled": path_assembled,
            "failure_class": failure_class,
            "failed_section_index": index,
            "failed_section": planned[index].name if index is not None else None,
            "failed_stage": stage,
        },
        "per_section_results": rows,
        "per_section_status": statuses,
        "per_section_runtime": runtimes,
    }


def success_status(join_metrics: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "sections_solved": True,
        "joins_valid": bool(join_metrics["summary"].get("all_within_tolerance", False)),
        "path_assembled": True,
        "failure_class": None,
        "failed_section_index": None,
        "failed_section": None,
        "failed_stage": None,
    }
