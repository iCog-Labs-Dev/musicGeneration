"""Endpoint-distribution metrics for the MUS-15 evaluation (distribution-to-distribution Method A).

Everything here is *derived* from diagnostics MUS-15 already produces on the plan result:
``plan.start_mass`` / ``plan.end_mass`` (:class:`EndpointMassReport`), ``plan.endpoint_resolution``
(:class:`EndpointResolution`) and ``plan.diagnostics``.  No reachability, flow, trimming or
renormalization logic is repeated; the evaluator only measures and reports.

Vocabulary (per side, "start" / "terminal"; MUS-15 names in parentheses):

* ``requested_mass``  (``requested_mass``)          mass of the request before any change
* ``feasible_mass``   (``feasible_mass``)           requested mass that survived feasibility handling
* ``removed_mass``    (``removed_state_mass``)      requested mass of states removed whole
* ``trimmed_mass``    (``trimmed_mass``)            raw mass subtracted by the routable-mass trim
* ``effective_mass``  (sum of ``effective``)        what the solver was asked to match (1 after renormalization)
* ``realized_mass``   (sum of ``realized_probabilities``)  the bridge's actual marginal, measured after solving

Calibration is only ever computed from *repeated* sampled paths drawn from the one solved bridge.
A single MAP or sampled path says nothing about a distribution, so with fewer than
:data:`MIN_CALIBRATION_SAMPLES` samples the result is an explicit ``unavailable_*`` status with
``None`` values, never zeros.

Every value is JSON-compatible.  Numeric leaves live in flat per-side dicts so
:func:`aimusic.evaluation.stats.flatten_metrics` can aggregate them; per-state residual rows are
lists and are intentionally not aggregated.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SIDES = ("start", "terminal")
# Below this many repeated samples an empirical frequency is too noisy to call a calibration.
MIN_CALIBRATION_SAMPLES = 30

ACCOUNTING_NO_MODIFICATION = "no_modification"
ACCOUNTING_REMOVED = "removed_and_renormalized"
ACCOUNTING_TRIMMED = "trimmed_and_renormalized"
ACCOUNTING_REMOVED_TRIMMED = "removed_trimmed_and_renormalized"
ACCOUNTING_BLENDED = "interior_blend"
ACCOUNTING_STRICT_FAILURE = "strict_failure"

CAL_AVAILABLE = "available"
CAL_NO_SAMPLES = "unavailable_no_repeated_samples"
CAL_TOO_FEW = "unavailable_too_few_samples"
CAL_NOT_APPLICABLE = "not_applicable_singleton"


class EndpointDiagnosticsUnavailable(RuntimeError):
    """The public planning API did not provide the diagnostics distribution mode requires."""


def entropy_nats(probabilities: Sequence[float]) -> float:
    """Shannon entropy in nats; zero-probability states contribute nothing."""
    return float(-sum(p * math.log(p) for p in probabilities if p > 0.0))


def support_size(probabilities: Sequence[float]) -> int:
    return sum(1 for p in probabilities if p > 0.0)


def _by_state(dist: Any) -> Dict[Any, float]:
    return {s: float(p) for s, p in zip(dist.layer.states, dist.probabilities)}


def accounting_status(report: Any) -> str:
    """Classify what happened to one side's request (strict failure is classified by the caller)."""
    removed = bool(report.removed)
    trimmed = bool(report.trims)
    if removed and trimmed:
        return ACCOUNTING_REMOVED_TRIMMED
    if removed:
        return ACCOUNTING_REMOVED
    if trimmed:
        return ACCOUNTING_TRIMMED
    if report.interior_margin > 0.0:
        return ACCOUNTING_BLENDED
    return ACCOUNTING_NO_MODIFICATION


def require_distribution_diagnostics(plan: Any, expected_mode: str) -> None:
    """Fail loudly (never fall back to singleton) if MUS-15 diagnostics are not on the plan."""
    diag = getattr(plan, "diagnostics", None)
    mode = getattr(diag, "endpoint_mode", None)
    if mode != expected_mode:
        raise EndpointDiagnosticsUnavailable(
            f"plan reports endpoint_mode={mode!r} but the run requested {expected_mode!r}; "
            "refusing to evaluate under a different mode."
        )
    missing = [
        name
        for name in ("start_mass", "end_mass", "endpoint_resolution")
        if getattr(plan, name, None) is None
    ]
    if missing:
        raise EndpointDiagnosticsUnavailable(
            f"the planning result has no {', '.join(missing)}; the MUS-15 endpoint diagnostics "
            "are required for endpoint evaluation."
        )
    for name in ("start_mass", "end_mass"):
        if not getattr(plan, name).realized_probabilities:
            raise EndpointDiagnosticsUnavailable(
                f"{name}.realized_probabilities is empty: the realized endpoint marginals were "
                "not recorded, so requested-vs-realized error cannot be measured."
            )


def _residual_rows(report: Any, tolerance: float) -> List[Dict[str, Any]]:
    requested = _by_state(report.requested)
    effective = _by_state(report.effective)
    realized = {
        s: float(p) for s, p in zip(report.effective.layer.states, report.realized_probabilities)
    }
    order: List[Any] = list(requested)
    order += [s for s in effective if s not in requested]
    rows: List[Dict[str, Any]] = []
    for state in order:
        req, eff, real = requested.get(state, 0.0), effective.get(state, 0.0), realized.get(state, 0.0)
        rows.append(
            {
                "state": state.to_dict(),
                "requested": req,
                "effective": eff,
                "realized": real,
                "residual_vs_effective": real - eff,
                "residual_vs_requested": real - req,
                "removed": state in requested and state not in effective,
                "within_tolerance": abs(real - eff) <= tolerance,
            }
        )
    return rows


def side_metrics(report: Any, tolerance: float) -> Dict[str, Any]:
    """Mass accounting, marginal error, residual summary and entropy for one endpoint side."""
    rows = _residual_rows(report, tolerance)
    res_eff = [abs(r["residual_vs_effective"]) for r in rows]
    res_req = [abs(r["residual_vs_requested"]) for r in rows]
    eff_probs = report.effective.probabilities
    req_probs = report.requested.probabilities
    realized = report.realized_probabilities
    return {
        "support_size_requested": support_size(req_probs),
        "support_size_effective": support_size(eff_probs),
        "support_size_realized": support_size(realized),
        # mass accounting (MUS-15 names)
        "requested_mass": float(report.requested_mass),
        "feasible_mass": float(report.feasible_mass),
        "removed_mass": float(report.removed_state_mass),
        "individually_unreachable_mass": float(report.individually_unreachable_mass),
        "trimmed_mass": float(report.trimmed_mass),
        "trimmed_requested_mass": float(report.trimmed_requested_mass),
        "effective_mass": float(sum(eff_probs)),
        "realized_mass": float(sum(realized)),
        "removed_state_count": len(report.removed),
        "trimmed_state_count": len(report.trims),
        "total_variation_shift": float(report.total_variation_shift),
        "accounting_status": accounting_status(report),
        # marginal error: realized vs requested (the user's ask) and vs effective (what the solver targeted)
        "marginal_l1_vs_requested": float(sum(res_req)),
        "marginal_max_abs_vs_requested": float(max(res_req, default=0.0)),
        "marginal_l1_vs_effective": float(sum(res_eff)),
        "marginal_max_abs_vs_effective": float(max(res_eff, default=0.0)),
        # residuals vs effective with the configured tolerance
        "residual_tolerance": float(tolerance),
        "residual_max_abs": float(max(res_eff, default=0.0)),
        "residual_mean_abs": float(sum(res_eff) / len(res_eff)) if res_eff else 0.0,
        "residual_states_outside_tolerance": sum(1 for r in rows if not r["within_tolerance"]),
        "residual_within_tolerance": all(r["within_tolerance"] for r in rows),
        "requested_equals_realized_within_tolerance": all(
            abs(r["residual_vs_requested"]) <= tolerance for r in rows
        ),
        # entropy (nats)
        "entropy_requested": entropy_nats(req_probs),
        "entropy_effective": entropy_nats(eff_probs),
        "entropy_realized": entropy_nats(realized),
        "entropy_empirical": None,  # filled from repeated samples when available
        "residuals": rows,
    }


def _flow_dict(check: Any) -> Dict[str, Any]:
    return {
        "feasible": bool(check.feasible),
        "flow_value": float(check.flow_value),
        "deficit": float(check.deficit),
        "tolerance": float(check.tolerance),
    }


def endpoint_metrics(plan: Any, *, marginal_tolerance: float) -> Dict[str, Any]:
    """Full endpoint-distribution metrics for a distribution-mode plan (diagnostics already checked)."""
    start = side_metrics(plan.start_mass, marginal_tolerance)
    terminal = side_metrics(plan.end_mass, marginal_tolerance)
    resolution = plan.endpoint_resolution
    modified = bool(resolution.request_modified)
    statuses = {start["accounting_status"], terminal["accounting_status"]}
    overall = ACCOUNTING_NO_MODIFICATION
    for candidate in (ACCOUNTING_BLENDED, ACCOUNTING_TRIMMED, ACCOUNTING_REMOVED, ACCOUNTING_REMOVED_TRIMMED):
        if candidate in statuses:
            overall = candidate
    if len(statuses - {ACCOUNTING_NO_MODIFICATION}) > 1:
        overall = ACCOUNTING_REMOVED_TRIMMED
    return {
        "status": "ok",
        "generation_method": "method_a_distribution_endpoints",
        "endpoint_mode": "distribution",
        "policy": resolution.policy.value,
        "accounting_status": overall,
        "request_modified": modified,
        "modifications": list(resolution.modifications),
        "termination_reason": resolution.termination_reason,
        "marginal_tolerance": float(marginal_tolerance),
        "final_flow_check": _flow_dict(resolution.final_flow_check),
        "initial_flow_check": _flow_dict(resolution.initial_flow_check),
        "bridge_converged": bool(plan.diagnostics.bridge_converged),
        # flat, spec-named fields (also reachable under start / terminal)
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
        "calibration": calibration_unavailable(CAL_NO_SAMPLES, 0),
    }


def singleton_endpoint_info(plan: Any) -> Dict[str, Any]:
    """Factual endpoint record for singleton runs; distribution-only metrics are marked unavailable.

    Nothing distribution-specific (marginal error, entropy, calibration) is computed: for a
    one-state endpoint those would be trivially 0 and read as a result.
    """
    info: Dict[str, Any] = {
        "status": "not_applicable",
        "generation_method": "method_a_singleton_endpoints",
        "endpoint_mode": "singleton",
        "distribution_metrics": {
            "available": False,
            "reason": "Distribution metrics unavailable: singleton endpoint mode.",
        },
    }
    start, end = getattr(plan, "start_mass", None), getattr(plan, "end_mass", None)
    if start is not None and end is not None:
        info["endpoint_support_size_start"] = support_size(start.effective.probabilities)
        info["endpoint_support_size_terminal"] = support_size(end.effective.probabilities)
    return info


def calibration_unavailable(status: str, n_samples: int) -> Dict[str, Any]:
    return {
        "status": status,
        "n_samples": int(n_samples),
        "min_samples_required": MIN_CALIBRATION_SAMPLES,
    }


def _frequencies(states: Sequence[Any]) -> Dict[Any, float]:
    counts: Dict[Any, int] = {}
    for s in states:
        counts[s] = counts.get(s, 0) + 1
    n = len(states)
    return {s: c / n for s, c in counts.items()}


def _side_calibration(report: Any, observed: Sequence[Any]) -> Dict[str, Any]:
    n = len(observed)
    freq = _frequencies(observed)
    effective = _by_state(report.effective)
    requested = _by_state(report.requested)
    support = list(dict.fromkeys([*effective, *freq]))
    l1_eff = sum(abs(freq.get(s, 0.0) - effective.get(s, 0.0)) for s in support)
    max_eff = max((abs(freq.get(s, 0.0) - effective.get(s, 0.0)) for s in support), default=0.0)
    keys_req = list(dict.fromkeys([*requested, *freq]))
    l1_req = sum(abs(freq.get(s, 0.0) - requested.get(s, 0.0)) for s in keys_req)
    max_req = max((abs(freq.get(s, 0.0) - requested.get(s, 0.0)) for s in keys_req), default=0.0)
    zs = [
        abs(freq.get(s, 0.0) - p) / math.sqrt(p * (1.0 - p) / n)
        for s, p in effective.items()
        if 0.0 < p < 1.0
    ]
    # E|f - p| ~ sqrt(2 p (1-p) / (pi n)) per state: the L1 a perfectly calibrated sampler is expected to show.
    noise = sum(math.sqrt(2.0 * p * (1.0 - p) / (math.pi * n)) for p in effective.values())
    return {
        "l1_vs_effective": float(l1_eff),
        "max_abs_vs_effective": float(max_eff),
        "l1_vs_requested": float(l1_req),
        "max_abs_vs_requested": float(max_req),
        "max_abs_z_vs_effective": float(max(zs)) if zs else None,
        "expected_l1_sampling_noise": float(noise),
        "outside_effective_support_mass": float(
            sum(f for s, f in freq.items() if effective.get(s, 0.0) <= 0.0)
        ),
        "entropy_empirical": entropy_nats(list(freq.values())),
        "support_size_empirical": len(freq),
    }


def calibration_from_samples(
    plan: Any, start_states: Sequence[Any], terminal_states: Sequence[Any]
) -> Dict[str, Any]:
    """Empirical endpoint frequencies from repeated sampled paths vs requested/effective probabilities."""
    n = len(start_states)
    if n != len(terminal_states):
        raise ValueError("start and terminal sample lists must have equal length.")
    if n == 0:
        return calibration_unavailable(CAL_NO_SAMPLES, 0)
    if n < MIN_CALIBRATION_SAMPLES:
        return calibration_unavailable(CAL_TOO_FEW, n)
    return {
        "status": CAL_AVAILABLE,
        "n_samples": n,
        "min_samples_required": MIN_CALIBRATION_SAMPLES,
        "start": _side_calibration(plan.start_mass, start_states),
        "terminal": _side_calibration(plan.end_mass, terminal_states),
    }


def draw_endpoint_samples(
    plan: Any, key: Any, n: int
) -> Tuple[List[Any], List[Any]]:
    """Draw ``n`` paths from the single solved bridge with the public sampler; return their endpoints.

    Each draw uses an independent key derived from ``key`` so results do not depend on draw order.
    """
    from aimusic.planning.sb import sample_bridge_path

    starts: List[Any] = []
    terminals: List[Any] = []
    for i in range(n):
        sample, _ = sample_bridge_path(plan.bridge, key.derive(f"calibration.{i}"))
        starts.append(sample.path[0])
        terminals.append(sample.path[-1])
    return starts, terminals


def apply_calibration(metrics: Dict[str, Any], calibration: Mapping[str, Any]) -> None:
    """Attach calibration (and the empirical entropy it yields) to an endpoint metrics dict."""
    metrics["calibration"] = dict(calibration)
    metrics["calibration_status"] = calibration["status"]
    metrics["calibration_n_samples"] = calibration["n_samples"]
    if calibration["status"] == CAL_AVAILABLE:
        for side_name, label in (("start", "start"), ("terminal", "terminal")):
            side = calibration[side_name]
            target = metrics[label]
            target["entropy_empirical"] = side["entropy_empirical"]
            for name in (
                "l1_vs_effective", "max_abs_vs_effective", "l1_vs_requested", "max_abs_vs_requested",
                "max_abs_z_vs_effective", "expected_l1_sampling_noise", "outside_effective_support_mass",
            ):
                target[f"calibration_{name}"] = side[name]


def failure_endpoint_info(exc: BaseException, mode: str) -> Optional[Dict[str, Any]]:
    """Endpoint section for a run that failed during endpoint resolution.

    Only MUS-15 feasibility errors are classified (a strict failure is a legitimate, informative
    outcome: the request was not allowed to change).  Other errors return ``None``.
    """
    from aimusic.planning.endpoint_feasibility import (
        EndpointFeasibilityError,
        EndpointMassInfeasibleError,
        EndpointReachabilityError,
    )

    if not isinstance(exc, EndpointFeasibilityError):
        return None
    kind = (
        "level1_unreachable_support"
        if isinstance(exc, EndpointReachabilityError)
        else "level2_mass_infeasible" if isinstance(exc, EndpointMassInfeasibleError) else "infeasible"
    )
    info: Dict[str, Any] = {
        "status": "failed",
        "endpoint_mode": mode,
        "accounting_status": ACCOUNTING_STRICT_FAILURE,
        "failure_kind": kind,
        "message": str(exc)[:500],
        "offending_state_count": len(exc.states),
    }
    if exc.flow_check is not None:
        info["flow_check"] = _flow_dict(exc.flow_check)
    return info
