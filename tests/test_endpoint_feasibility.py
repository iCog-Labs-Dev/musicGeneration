"""Tests: reachability, global feasibility, fallbacks, restriction, validation."""
import random

import pytest

import aimusic.planning.endpoint_feasibility as ef
from aimusic.core.config import SBConfig
from aimusic.core.core_types import BeatState, Edge, EndpointDistribution, Layer
from aimusic.planning.endpoint_feasibility import (
    EndpointFallbackPolicy as Policy,
    EndpointFeasibilityError,
    EndpointMarginalError,
    EndpointMassInfeasibleError,
    EndpointReachabilityError,
    EndpointSolverNonConvergenceError,
    EndpointTolerances,
    compute_endpoint_reachability,
    global_flow_check,
    resolve_endpoint_distributions,
    routable_masses,
    solve_endpoint_bridge,
    validate_solution_marginals,
)
from aimusic.planning.graph import GraphDiagnostics, SparseGraph
from aimusic.planning.sb import build_sb_problem, solve_sb


def _s(i):
    return BeatState(0, 0, 0, i, 0, 0, 0, 0)


def _diagnostics(layer_sizes):
    # the real GraphDiagnostics requires BOTH fields
    return GraphDiagnostics(layer_sizes=tuple(layer_sizes), layer_diagnostics=())


def _dist(t, mapping):
    states = tuple(mapping)
    return EndpointDistribution(Layer(t, states), tuple(mapping[s] for s in states))


def _graph(starts, ends, edges):
    layers = (Layer(0, tuple(starts)), Layer(1, tuple(ends)))
    return SparseGraph(
        layers=layers,
        edges_by_time=(tuple(Edge(0, a, b, w) for a, b, w in edges),),
        diagnostics=_diagnostics((len(starts), len(ends))),
    )


A0, A1, A2, A3 = _s(0), _s(1), _s(2), _s(3)
B0, B1, B2, B3 = _s(10), _s(11), _s(12), _s(13)
CFG = SBConfig(horizon_t=1, tolerance=1e-12, max_iterations=2000)
FULL = [(A0, B0, 0.0), (A0, B1, -1.0), (A1, B0, -0.5), (A1, B1, 0.3)]
DISJOINT = [(A0, B0, 0.0), (A1, B1, 0.0)]  # the disconnected counter-example
DROP, TRIM, COND = (
    Policy.DROP_AND_RENORMALIZE,
    Policy.PARTIAL_TRIM_AND_RENORMALIZE,
    Policy.CONDITIONAL_DROP_THEN_TRIM,
)


def _counter_example():
    g = _graph([A0, A1], [B0, B1], DISJOINT)
    return g, _dist(0, {A0: 0.9, A1: 0.1}), _dist(1, {B0: 0.1, B1: 0.9})


def _states(dist):
    return dict(zip(dist.layer.states, dist.probabilities))


# --------------------------------------------------------------------- A -- #
def test_feasible_multistate_flow_one_and_marginals_realized():
    g = _graph([A0, A1], [B0, B1], FULL)
    res = solve_endpoint_bridge(
        g, _dist(0, {A0: 0.3, A1: 0.7}), _dist(1, {B0: 0.6, B1: 0.4}), sb_config=CFG
    )
    assert res.resolution.initial_flow_check.flow_value == pytest.approx(1.0)
    assert res.resolution.termination_reason == "feasible_as_requested"
    assert res.resolution.layer_sizes_before == res.resolution.layer_sizes_after == (2, 2)
    assert all(p > 0 for p in res.start_mass.effective.probabilities)
    assert res.start_mass.max_abs_realized_error < 1e-6
    assert res.end_mass.max_abs_realized_error < 1e-6
    assert all(p > 0 for p in res.start_mass.realized_probabilities)
    assert res.start_mass.removed_state_mass == 0.0 and res.start_mass.trimmed_mass == 0.0


def test_reachability_relation_is_exact_not_all_pairs():
    g = _graph([A0, A1], [B0, B1], DISJOINT)
    rel = compute_endpoint_reachability(g, [A0, A1], [B0, B1])
    assert rel[A0] == (B0,) and rel[A1] == (B1,)


# --------------------------------------------------------------------- B -- #
def _unreachable_case():
    g = _graph([A0, A1, A2], [B0, B1, B2], FULL)  # A2 has no edge; B2 has no incoming edge
    p = _dist(0, {A0: 0.4, A1: 0.4, A2: 0.2})
    q = _dist(1, {B0: 0.4, B1: 0.4, B2: 0.2})
    return g, p, q


def test_strict_fails_on_individually_unreachable_before_solver(monkeypatch):
    called = []
    monkeypatch.setattr(ef, "build_sb_problem", lambda *a, **k: called.append(1))
    monkeypatch.setattr(ef, "solve_sb", lambda *a, **k: called.append(1))
    g, p, q = _unreachable_case()
    with pytest.raises(EndpointReachabilityError, match="Level 1") as exc:
        solve_endpoint_bridge(g, p, q, sb_config=CFG)
    assert set(exc.value.states) == {A2, B2}
    assert called == []


def test_drop_removes_unreachable_prunes_zero_mass_and_runs_global_check():
    g, p, q = _unreachable_case()
    res = resolve_endpoint_distributions(g, p, q, policy=DROP)
    assert res.start.removed_state_mass == pytest.approx(0.2)
    assert res.start.individually_unreachable_mass == pytest.approx(0.2)
    # zero-mass states are physically absent from the restricted boundary layers
    assert res.start.effective.layer.states == (A0, A1)
    assert res.end.effective.layer.states == (B0, B1)
    assert res.graph.layers[0].states == (A0, A1) and res.graph.layers[-1].states == (B0, B1)
    assert res.layer_sizes_before == (3, 3) and res.layer_sizes_after == (2, 2)
    assert res.start.effective.probabilities == pytest.approx((0.5, 0.5))
    assert res.final_flow_check.feasible
    assert res.termination_reason == "feasible_after_individual_removal"
    assert res.start.trimmed_mass == 0.0


def test_partial_trim_alone_never_removes_unreachable_states():
    g, p, q = _unreachable_case()
    with pytest.raises(EndpointReachabilityError, match="never removes unreachable"):
        resolve_endpoint_distributions(g, p, q, policy=TRIM)


def test_conditional_policy_handles_level1_then_level2():
    # Level 1: A2/B2 unreachable. After dropping them, Level 2 is still infeasible.
    g = _graph([A0, A1, A2], [B0, B1, B2], DISJOINT)
    p = _dist(0, {A0: 0.7, A1: 0.1, A2: 0.2})
    q = _dist(1, {B0: 0.1, B1: 0.7, B2: 0.2})
    res = resolve_endpoint_distributions(g, p, q, policy=COND)
    assert res.termination_reason == "feasible_after_individual_removal_and_partial_trim"
    stages = {r.stage for r in res.start.removed + res.end.removed}
    assert stages == {"individual_reachability"}  # Level 2 used trimming, never removal
    assert res.start.trimmed_mass > 0 and res.end.trimmed_mass > 0
    assert res.initial_flow_check.feasible is False and res.final_flow_check.feasible


def test_drop_individual_removal_then_global_check_still_runs():
    g = _graph([A0, A1, A2], [B0, B1, B2], DISJOINT)
    p = _dist(0, {A0: 0.8, A1: 0.1, A2: 0.1})
    q = _dist(1, {B0: 0.1, B1: 0.8, B2: 0.1})
    res = resolve_endpoint_distributions(g, p, q, policy=DROP)
    assert res.iterations >= 1 and res.final_flow_check.feasible


# --------------------------------------------------------------------- C -- #
def test_counter_example_individually_reachable_but_globally_infeasible():
    g, p, q = _counter_example()
    rel = compute_endpoint_reachability(g, [A0, A1], [B0, B1])
    assert all(rel[s] for s in (A0, A1))
    check = global_flow_check({A0: 0.9, A1: 0.1}, {B0: 0.1, B1: 0.9}, rel)
    assert not check.feasible
    assert check.flow_value == pytest.approx(0.2)
    assert check.deficit == pytest.approx(0.8)
    assert check.violated_start_states == (A0,)
    assert check.neighbor_terminal_states == (B0,)
    assert check.excess == pytest.approx(0.8)


def test_strict_fails_before_build_sb_problem_and_solve(monkeypatch):
    called = []
    monkeypatch.setattr(ef, "build_sb_problem", lambda *a, **k: called.append("build"))
    monkeypatch.setattr(ef, "solve_sb", lambda *a, **k: called.append("solve"))
    g, p, q = _counter_example()
    with pytest.raises(EndpointMassInfeasibleError, match="Level 2") as exc:
        solve_endpoint_bridge(g, p, q, sb_config=CFG)  # STRICT is the default
    assert "max-flow=0.2" in str(exc.value) and "deficit=0.8" in str(exc.value)
    assert exc.value.flow_check.deficit == pytest.approx(0.8)
    assert called == []


def test_strict_is_default_policy():
    g, p, q = _counter_example()
    with pytest.raises(EndpointMassInfeasibleError):
        resolve_endpoint_distributions(g, p, q)


def test_plain_renormalization_is_not_a_solution():
    rel = {A0: (B0,), A1: (B1,)}
    scaled = global_flow_check({A0: 0.45, A1: 0.05}, {B0: 0.05, B1: 0.45}, rel)
    assert not scaled.feasible
    assert scaled.flow_value / scaled.total_start_mass == pytest.approx(0.2)


def test_whole_state_policy_removes_whole_states_deterministically():
    g, p, q = _counter_example()
    runs = [resolve_endpoint_distributions(g, p, q, policy=DROP) for _ in range(3)]
    first = runs[0]
    assert first.start.removed[0].state == A0  # tie .9/.9 -> start before terminal
    assert first.start.removed[0].stage == "global_feasibility"
    assert any(r.state == B0 and r.stage == "fallback_cascade" for r in first.end.removed)
    assert first.start.effective.layer.states == (A1,)
    assert first.start.effective.probabilities == pytest.approx((1.0,))
    assert first.end.effective.layer.states == (B1,)
    assert first.start.trimmed_mass == 0.0 and first.final_flow_check.feasible
    for r in runs[1:]:
        assert r.decisions == first.decisions


def test_partial_trim_counter_example_max_flow_rule():
    g, p, q = _counter_example()
    res = resolve_endpoint_distributions(g, p, q, policy=TRIM)
    assert res.termination_reason == "feasible_after_partial_trim"
    assert res.start.effective.probabilities == pytest.approx((0.5, 0.5))
    assert res.end.effective.probabilities == pytest.approx((0.5, 0.5))
    assert res.start.removed_state_mass == 0.0 and res.end.removed_state_mass == 0.0
    # raw trimmed mass on each side is exactly the minimum, 1 - F*
    assert res.start.trimmed_mass == pytest.approx(0.8)
    assert res.end.trimmed_mass == pytest.approx(0.8)
    (a0,) = res.start.trims
    assert a0.state == A0 and a0.original_mass == pytest.approx(0.9)
    assert a0.resulting_mass == pytest.approx(0.1) and a0.effective_mass == pytest.approx(0.5)
    assert a0.flow_before == pytest.approx(0.2) and a0.deficit_before == pytest.approx(0.8)
    assert a0.feasible_after is True and a0.flow_after == pytest.approx(1.0)
    # trimmed mass is not the whole change: renormalization moved A1 too
    changes = {c.state: c for c in res.start.probability_changes}
    assert changes[A0].delta == pytest.approx(-0.4) and changes[A1].delta == pytest.approx(0.4)
    assert res.start.total_variation_shift == pytest.approx(0.4)


def test_level2_with_conditional_uses_trim_not_removal():
    g, p, q = _counter_example()
    res = resolve_endpoint_distributions(g, p, q, policy=COND)
    assert res.start.removed == () and res.start.trims  # trim, never whole-state removal
    drop = resolve_endpoint_distributions(g, p, q, policy=DROP)
    assert drop.start.trims == () and drop.start.removed  # DROP keeps whole-state semantics


# --------------------------------------------------------------------- D -- #
def _zeroed_case():
    # A0,A1 compete for B0 (0.2); A2 -> B1. Max-flow gives B0's mass to ONE of A0/A1.
    g = _graph([A0, A1, A2], [B0, B1], [(A0, B0, 0), (A1, B0, 0), (A2, B1, 0)])
    return g, _dist(0, {A0: 0.4, A1: 0.4, A2: 0.2}), _dist(1, {B0: 0.2, B1: 0.8})


def test_state_trimmed_to_zero_is_recorded_as_removed_not_trimmed():
    g, p, q = _zeroed_case()
    res = resolve_endpoint_distributions(g, p, q, policy=TRIM)
    zeroed = [r for r in res.start.removed if r.stage == "partial_trim_zeroed"]
    assert len(zeroed) == 1 and zeroed[0].state == A1  # deterministic Dinic split
    assert zeroed[0].requested_mass == pytest.approx(0.4)
    assert zeroed[0].state not in {t.state for t in res.start.trims}
    assert {t.state for t in res.start.trims} == {A0}
    assert A1 not in res.graph.layers[0].states  # and it is pruned from the layer
    assert res.layer_sizes_before == (3, 2) and res.layer_sizes_after == (2, 2)
    assert res.start.removed_state_mass == pytest.approx(0.4)
    assert res.start.trimmed_mass == pytest.approx(0.2)  # A0: 0.4 -> 0.2, kept separate


def test_partial_trim_is_deterministic_and_distributions_valid():
    g, p, q = _zeroed_case()
    runs = [resolve_endpoint_distributions(g, p, q, policy=TRIM) for _ in range(3)]
    for r in runs[1:]:
        assert r.decisions == runs[0].decisions
        assert r.start.trims == runs[0].start.trims
        assert r.start.effective.probabilities == runs[0].start.effective.probabilities
    for rep in (runs[0].start, runs[0].end):
        assert all(x > 0.0 for x in rep.effective.probabilities)
        assert sum(rep.effective.probabilities) == pytest.approx(1.0)
    assert runs[0].final_flow_check.feasible
    assert len(runs[0].feasibility_history) == runs[0].iterations + 1  # rerun each iteration


def test_zero_requested_mass_is_never_given_mass():
    g = _graph([A0, A1, A2], [B0, B1], DISJOINT + [(A2, B1, 0.0)])
    p = EndpointDistribution(Layer(0, (A0, A1, A2)), (0.9, 0.1, 0.0))
    q = _dist(1, {B0: 0.1, B1: 0.9})
    res = resolve_endpoint_distributions(g, p, q, policy=TRIM)
    assert A2 not in res.start.effective.layer.states
    assert A2 not in {c.state for c in res.start.probability_changes}
    res = resolve_endpoint_distributions(g, p, q, policy=TRIM, interior_margin=0.1)
    assert A2 not in res.start.effective.layer.states


def test_trim_limits_raise_clear_errors():
    g, p, q = _counter_example()
    with pytest.raises(EndpointFeasibilityError, match="max_iterations"):
        resolve_endpoint_distributions(g, p, q, policy=TRIM, max_iterations=0)


def _lp_max_retained(p, q, relation):
    import scipy.optimize as sp
    pairs = [(s, t) for s in p for t in relation.get(s, ()) if t in q]
    c = [-1.0] * len(pairs)
    a_ub, b_ub = [], []
    for s in p:
        a_ub.append([1.0 if ps == s else 0.0 for ps, _ in pairs]); b_ub.append(p[s])
    for t in q:
        a_ub.append([1.0 if pt == t else 0.0 for _, pt in pairs]); b_ub.append(q[t])
    res = sp.linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=(0, None), method="highs")
    return -res.fun


@pytest.mark.parametrize("seed", range(12))
def test_routable_mass_trim_is_optimal_for_retained_mass(seed):
    rng = random.Random(seed)
    starts = [_s(i) for i in range(4)]
    terms = [_s(10 + i) for i in range(4)]
    relation = {s: tuple(t for t in terms if rng.random() < 0.5) for s in starts}
    p_raw = [rng.random() + 0.05 for _ in starts]
    q_raw = [rng.random() + 0.05 for _ in terms]
    p = {s: v / sum(p_raw) for s, v in zip(starts, p_raw)}
    q = {t: v / sum(q_raw) for t, v in zip(terms, q_raw)}
    rm = routable_masses(p, q, relation)
    assert rm.flow_value == pytest.approx(_lp_max_retained(p, q, relation), abs=1e-9)
    assert sum(rm.start.values()) == pytest.approx(rm.flow_value, abs=1e-9)
    assert sum(rm.terminal.values()) == pytest.approx(rm.flow_value, abs=1e-9)
    assert all(rm.start[s] <= p[s] + 1e-12 for s in p) and all(rm.terminal[t] <= q[t] + 1e-12 for t in q)


# --------------------------------------------------------------------- E -- #
def test_whole_state_survivors_keep_relative_probabilities():
    g = _graph([A0, A1, A2], [B0, B1], [(A0, B0, 0), (A1, B1, 0), (A2, B1, 0)])
    p = _dist(0, {A0: 0.6, A1: 0.3, A2: 0.1})
    q = _dist(1, {B0: 0.1, B1: 0.9})
    res = resolve_endpoint_distributions(g, p, q, policy=DROP)
    eff = _states(res.start.effective)
    if A1 in eff and A2 in eff:
        assert eff[A1] / eff[A2] == pytest.approx(0.3 / 0.1)
    assert sum(eff.values()) == pytest.approx(1.0)
    assert res.start.trimmed_mass == 0.0 and res.final_flow_check.feasible
    assert len(res.feasibility_history) == res.iterations + 1


def test_whole_state_removal_ordering_deterministic():
    cands = [("terminal", B1, 0.2), ("start", A1, 0.2), ("start", A0, 0.2), ("start", A2, 0.1)]
    assert ef._select_removal(cands) == ("start", A2, 0.1)
    assert ef._select_removal(cands[:3]) == ("start", A0, 0.2)


def test_failure_when_no_support_remains():
    g = _graph([A0], [B0], [])
    with pytest.raises(EndpointFeasibilityError, match="No feasible endpoint support"):
        resolve_endpoint_distributions(g, _dist(0, {A0: 1.0}), _dist(1, {B0: 1.0}), policy=DROP)


# --------------------------------------------------------------------- F -- #
def _resolved_and_solved(g, p, q, **kw):
    resolution = resolve_endpoint_distributions(g, p, q, **kw)
    solution = solve_sb(
        build_sb_problem(
            resolution.graph, resolution.start.effective, resolution.end.effective, sb_config=CFG
        )
    )
    return resolution, solution


def test_nonconvergence_is_failure_with_both_errors_reported():
    g = _graph([A0, A1], [B0, B1], FULL)
    p, q = _dist(0, {A0: 0.3, A1: 0.7}), _dist(1, {B0: 0.6, B1: 0.4})
    res = resolve_endpoint_distributions(g, p, q)
    short = SBConfig(horizon_t=1, tolerance=1e-15, max_iterations=1)
    sol = solve_sb(build_sb_problem(res.graph, res.start.effective, res.end.effective, sb_config=short))
    assert not sol.trace.converged
    with pytest.raises(EndpointSolverNonConvergenceError, match="start_error=.*terminal_error=") as exc:
        validate_solution_marginals(sol, res, tolerance=1e-6)
    assert exc.value.start_error is not None and exc.value.terminal_error is not None


def test_wrong_marginal_raises_and_names_failing_side():
    g = _graph([A0, A1], [B0, B1], FULL)
    p, q = _dist(0, {A0: 0.3, A1: 0.7}), _dist(1, {B0: 0.6, B1: 0.4})
    resolution, solution = _resolved_and_solved(g, p, q)
    bad_end = ef.replace(
        resolution.end,
        effective=EndpointDistribution(resolution.end.effective.layer, (0.5, 0.5)),
    )
    with pytest.raises(EndpointMarginalError, match="terminal") as exc:
        validate_solution_marginals(solution, ef.replace(resolution, end=bad_end), tolerance=1e-6)
    assert exc.value.terminal_error > 0.05 and exc.value.start_error < 1e-6


def test_no_bridge_returned_after_validation_failure():
    g = _graph([A0, A1], [B0, B1], FULL)
    p, q = _dist(0, {A0: 0.3, A1: 0.7}), _dist(1, {B0: 0.6, B1: 0.4})
    with pytest.raises(EndpointSolverNonConvergenceError):
        solve_endpoint_bridge(
            g, p, q, sb_config=SBConfig(horizon_t=1, tolerance=1e-15, max_iterations=1)
        )


def _tight_after_trim_case():
    g = _graph(
        [A0, A1, A2], [B0, B1, B2],
        [(A0, B0, 0), (A1, B0, 0), (A1, B1, 0), (A2, B2, 0)],
    )
    return g, _dist(0, {A0: 0.5, A1: 0.3, A2: 0.2}), _dist(1, {B0: 0.2, B1: 0.3, B2: 0.5})


def test_slow_convergence_on_tight_cut_and_interior_margin_fix():
    g, p, q = _tight_after_trim_case()
    cfg = SBConfig(horizon_t=1, tolerance=1e-9, max_iterations=3000)
    # tight pair: forced-zero couplings -> error only shrinks like 1/k, never converges
    with pytest.raises(EndpointSolverNonConvergenceError, match="interior_margin"):
        solve_endpoint_bridge(g, p, q, sb_config=cfg, policy=TRIM, interior_margin=0.0)
    # explicit interior blend: strictly interior -> linear convergence, change is reported
    res = solve_endpoint_bridge(
        g, p, q,
        sb_config=SBConfig(horizon_t=1, tolerance=1e-9, max_iterations=5000),
        policy=TRIM, interior_margin=0.05,
    )
    assert res.solution.trace.converged and res.solution.trace.iterations < 1000
    assert res.start_mass.interior_margin == 0.05 and res.end_mass.interior_margin == 0.05
    assert res.start_mass.max_abs_realized_error < 1e-6
    assert res.start_mass.total_variation_shift > 0.0  # the blend is visible in the report


def test_interior_margin_is_off_by_default_and_never_under_strict():
    g = _graph([A0, A1], [B0, B1], FULL)
    res = resolve_endpoint_distributions(
        g, _dist(0, {A0: 0.3, A1: 0.7}), _dist(1, {B0: 0.6, B1: 0.4}), interior_margin=0.2
    )
    assert res.start.interior_margin == 0.0  # no trim happened, so no blend
    assert res.start.effective.probabilities == pytest.approx((0.3, 0.7))


# --------------------------------------------------------------------- G -- #
def test_singleton_goes_through_same_checks():
    g = _graph([A0, A1], [B0, B1], FULL)
    res = solve_endpoint_bridge(g, _dist(0, {A0: 1.0}), _dist(1, {B1: 1.0}), sb_config=CFG)
    assert res.resolution.initial_flow_check.flow_value == pytest.approx(1.0)
    assert res.start_mass.max_abs_realized_error < 1e-6
    g2 = _graph([A0, A1], [B0, B1], DISJOINT)
    with pytest.raises(EndpointFeasibilityError):
        solve_endpoint_bridge(g2, _dist(0, {A0: 1.0}), _dist(1, {B1: 1.0}), sb_config=CFG)


def test_endpoint_mode_enum():
    assert ef.EndpointMode.DISTRIBUTION.value == "distribution"
    assert ef.EndpointMode.SINGLETON.value == "singleton"


# --------------------------------------------------------------------- H -- #
def test_tolerance_relations_are_enforced():
    EndpointTolerances()  # defaults are consistent
    with pytest.raises(ValueError, match="capacity_epsilon"):
        EndpointTolerances(flow_tolerance=1e-9, capacity_epsilon=1e-11)
    with pytest.raises(ValueError, match="flow_tolerance must be <= marginal_tolerance"):
        EndpointTolerances(flow_tolerance=1e-5, marginal_tolerance=1e-7)
    with pytest.raises(ValueError, match="positive"):
        EndpointTolerances(flow_tolerance=0.0)
    with pytest.raises(ValueError):
        global_flow_check({A0: 1.0}, {B0: 1.0}, {A0: (B0,)}, tolerance=1e-12)  # eps 1e-14 too large


def test_positivity_is_strict_tiny_positive_mass_is_kept():
    g = _graph([A0, A1], [B0, B1], FULL)
    p = EndpointDistribution(Layer(0, (A0, A1)), (1.0 - 1e-15, 1e-15))
    q = _dist(1, {B0: 0.5, B1: 0.5})
    res = resolve_endpoint_distributions(g, p, q)
    assert res.start.effective.layer.states == (A0, A1)
    assert res.start.effective.probability_of(A1) > 0.0
    assert res.graph.layers[0].states == (A0, A1)


# --------------------------------------------------------------------- I -- #
def test_restricted_graph_prunes_dead_interior_paths_and_stays_consistent():
    M0, M1 = _s(20), _s(21)
    layers = (Layer(0, (A0, A1)), Layer(1, (M0, M1)), Layer(2, (B0,)))
    g = SparseGraph(
        layers=layers,
        edges_by_time=(
            (Edge(0, A0, M0, 0.0), Edge(0, A1, M1, 0.0)),
            (Edge(1, M0, B0, 0.0),),  # M1 is a dead end
        ),
        diagnostics=_diagnostics((2, 2, 1)),
    )
    res = resolve_endpoint_distributions(
        g, _dist(0, {A0: 0.5, A1: 0.5}), _dist(2, {B0: 1.0}), policy=DROP
    )
    assert res.layer_sizes_before == (2, 2, 1) and res.layer_sizes_after == (1, 1, 1)
    assert res.graph.layers[1].states == (M0,)
    assert [len(e) for e in res.graph.edges_by_time] == [1, 1]
    assert res.graph.diagnostics.layer_sizes == (1, 1, 1)
    solved = solve_endpoint_bridge(
        g, _dist(0, {A0: 0.5, A1: 0.5}), _dist(2, {B0: 1.0}),
        sb_config=SBConfig(horizon_t=2, tolerance=1e-12), policy=DROP,
    )
    assert solved.graph.layers[0].states == (A0,)


# ---------------------------------------------------------- tight-pair blending -- #
def _tight_as_requested_case():
    # feasible as requested (max-flow == 1) but A1 -> B0 is forced to zero by the cut
    # {A0} (pi0 = 0.5 == piT(B0) = 0.5): the pair sits on a Hall cut.
    g = _graph([A0, A1], [B0, B1], [(A0, B0, 0), (A1, B0, 0), (A1, B1, 0)])
    return g, _dist(0, {A0: 0.5, A1: 0.5}), _dist(1, {B0: 0.5, B1: 0.5})


def test_detect_tight_pairs_finds_forced_zero_pairs():
    rel = {A0: (B0,), A1: (B0, B1)}
    t = ef.detect_tight_pairs({A0: 0.5, A1: 0.5}, {B0: 0.5, B1: 0.5}, rel)
    assert t.tight and t.forced_zero_pairs == ((A1, B0),)


def test_detect_tight_pairs_interior_cases_are_not_tight():
    full = {A0: (B0, B1), A1: (B0, B1)}
    assert not ef.detect_tight_pairs({A0: 0.3, A1: 0.7}, {B0: 0.6, B1: 0.4}, full).tight
    # disjoint but exactly matched: unique coupling, yet every pair is strictly positive
    diag = {A0: (B0,), A1: (B1,)}
    assert not ef.detect_tight_pairs({A0: 0.5, A1: 0.5}, {B0: 0.5, B1: 0.5}, diag).tight
    # a non-tight pair with slack: A0's cut has piT(B0)=0.6 > pi0(A0)=0.5
    rel = {A0: (B0,), A1: (B0, B1)}
    assert not ef.detect_tight_pairs({A0: 0.5, A1: 0.5}, {B0: 0.6, B1: 0.4}, rel).tight


def test_tight_pair_feasible_as_requested_is_blended_when_policy_allows():
    g, p, q = _tight_as_requested_case()
    res = resolve_endpoint_distributions(g, p, q, policy=DROP, interior_margin=0.05)
    # a blended pair is NEVER reported as exact, even though no fallback was needed
    assert res.termination_reason == "feasible_after_interior_blend"
    assert res.request_modified and not res.exact
    assert res.modifications == ("interior_blend",)
    assert res.start.modified and res.end.modified
    assert res.tightness.tight and res.tightness.forced_zero_pairs == ((A1, B0),)
    assert res.start.interior_margin == 0.05 and res.end.interior_margin == 0.05
    assert res.final_flow_check.feasible  # blended pair re-verified
    assert res.start.total_variation_shift > 0.0  # the change is reported
    assert sum(res.start.effective.probabilities) == pytest.approx(1.0)
    assert res.start.removed == () and res.start.trims == ()


def test_tight_pair_blend_restores_convergence():
    g, p, q = _tight_as_requested_case()
    cfg = SBConfig(horizon_t=1, tolerance=1e-9, max_iterations=5000)
    with pytest.raises(EndpointSolverNonConvergenceError):
        solve_endpoint_bridge(g, p, q, sb_config=cfg, policy=DROP, interior_margin=0.0)
    res = solve_endpoint_bridge(g, p, q, sb_config=cfg, policy=DROP, interior_margin=0.05)
    assert res.solution.trace.converged and res.solution.trace.iterations < 1000
    assert res.start_mass.max_abs_realized_error < 1e-6


def test_strict_never_blends_a_tight_pair():
    g, p, q = _tight_as_requested_case()
    res = resolve_endpoint_distributions(g, p, q, interior_margin=0.2)  # STRICT default
    assert res.tightness.tight  # reported ...
    assert res.start.interior_margin == 0.0  # ... but the request is untouched
    assert res.start.effective.probabilities == pytest.approx((0.5, 0.5))
    assert any("STRICT leaves it unblended" in d for d in res.decisions)
    with pytest.raises(EndpointSolverNonConvergenceError, match="never blend"):
        solve_endpoint_bridge(
            g, p, q, sb_config=SBConfig(horizon_t=1, tolerance=1e-9, max_iterations=3000)
        )


def test_non_tight_pair_is_never_blended():
    g = _graph([A0, A1], [B0, B1], FULL)
    p, q = _dist(0, {A0: 0.3, A1: 0.7}), _dist(1, {B0: 0.6, B1: 0.4})
    for policy in (DROP, TRIM, COND):
        res = resolve_endpoint_distributions(g, p, q, policy=policy, interior_margin=0.2)
        assert not res.tightness.tight
        assert res.start.interior_margin == 0.0
        assert res.start.effective.probabilities == pytest.approx((0.3, 0.7))


def test_blend_after_whole_state_removal_when_result_is_tight():
    # removal leaves a tight remainder: A0 -> B0 ; A1 -> B0,B1 ; plus unreachable A2/B2
    g = _graph(
        [A0, A1, A2], [B0, B1, B2], [(A0, B0, 0), (A1, B0, 0), (A1, B1, 0)]
    )
    p = _dist(0, {A0: 0.4, A1: 0.4, A2: 0.2})
    q = _dist(1, {B0: 0.4, B1: 0.4, B2: 0.2})
    res = resolve_endpoint_distributions(g, p, q, policy=DROP, interior_margin=0.05)
    assert res.termination_reason == "feasible_after_individual_removal_with_interior_blend"
    assert res.modifications == ("individual_removal", "interior_blend")
    assert res.tightness.tight and res.start.interior_margin == 0.05
    assert A2 not in res.start.effective.layer.states


# ------------------------------------------------- DROP_UNREACHABLE_ONLY (pipeline default) -- #
DROP_U = Policy.DROP_UNREACHABLE_ONLY


def test_drop_unreachable_only_removes_level1_states_whole():
    g, p, q = _unreachable_case()
    res = resolve_endpoint_distributions(g, p, q, policy=DROP_U)
    assert res.termination_reason == "feasible_after_individual_removal"
    assert res.start.removed_state_mass == pytest.approx(0.2)
    assert {r.stage for r in res.start.removed + res.end.removed} == {"individual_reachability"}
    assert res.start.trimmed_mass == 0.0 and res.end.trimmed_mass == 0.0
    assert res.start.effective.layer.states == (A0, A1)
    assert res.start.effective.probabilities == pytest.approx((0.5, 0.5))


def test_drop_unreachable_only_never_repairs_level2_conflicts():
    g, p, q = _counter_example()  # every endpoint reachable, masses incompatible
    with pytest.raises(EndpointMassInfeasibleError, match="drop_unreachable_only") as exc:
        resolve_endpoint_distributions(g, p, q, policy=DROP_U)
    assert exc.value.flow_check.deficit == pytest.approx(0.8)
    # unreachable states are dropped first, then the mass conflict still raises
    g2 = _graph([A0, A1, A2], [B0, B1, B2], DISJOINT)
    p2 = _dist(0, {A0: 0.7, A1: 0.1, A2: 0.2})
    q2 = _dist(1, {B0: 0.1, B1: 0.7, B2: 0.2})
    with pytest.raises(EndpointMassInfeasibleError, match="already removed whole"):
        resolve_endpoint_distributions(g2, p2, q2, policy=DROP_U)


def test_drop_unreachable_only_never_blends_a_tight_pair():
    g, p, q = _tight_as_requested_case()
    res = resolve_endpoint_distributions(g, p, q, policy=DROP_U, interior_margin=0.2)
    assert res.tightness.tight and res.start.interior_margin == 0.0
    assert res.start.effective.probabilities == pytest.approx((0.5, 0.5))


# ------------------------------------------- exactness classification (never silent) -- #
def _exact_case():
    g = _graph([A0, A1], [B0, B1], FULL)
    return g, _dist(0, {A0: 0.3, A1: 0.7}), _dist(1, {B0: 0.6, B1: 0.4})


@pytest.mark.parametrize("policy", list(Policy))
def test_feasible_non_tight_request_is_exact_under_every_policy(policy):
    g, p, q = _exact_case()
    res = resolve_endpoint_distributions(g, p, q, policy=policy, interior_margin=0.2)
    assert res.exact and not res.request_modified and res.modifications == ()
    assert res.termination_reason == "feasible_as_requested"
    assert res.start.effective.probabilities == pytest.approx(p.probabilities)
    assert res.end.effective.probabilities == pytest.approx(q.probabilities)
    assert not res.start.modified and not res.end.modified


@pytest.mark.parametrize("policy", [Policy.STRICT, DROP_U])
def test_strict_style_policies_preserve_masses_even_when_tight(policy):
    g, p, q = _tight_as_requested_case()
    res = resolve_endpoint_distributions(g, p, q, policy=policy, interior_margin=0.2)
    assert res.tightness.tight
    assert res.exact and res.termination_reason == "feasible_as_requested"
    assert res.start.effective.probabilities == pytest.approx((0.5, 0.5))  # untouched


def test_every_modification_kind_is_classified():
    g, p, q = _unreachable_case()
    assert resolve_endpoint_distributions(g, p, q, policy=DROP_U).modifications == (
        "individual_removal",
    )
    g, p, q = _counter_example()
    assert resolve_endpoint_distributions(g, p, q, policy=DROP).modifications == (
        "whole_state_removal",
    )
    assert resolve_endpoint_distributions(g, p, q, policy=TRIM).modifications == ("partial_trim",)
    blended = resolve_endpoint_distributions(
        *_tight_as_requested_case(), policy=DROP, interior_margin=0.1
    )
    assert blended.modifications == ("interior_blend",) and not blended.exact


# ---------------------------------------- retained / trimmed mass accounting (requested units) -- #
def _identity(rep):
    return rep.removed_state_mass + rep.feasible_mass + rep.trimmed_requested_mass


def test_feasible_mass_is_retained_mass_after_partial_trim():
    g, p, q = _counter_example()  # p=(.9,.1), q=(.1,.9); F* = 0.2
    res = resolve_endpoint_distributions(g, p, q, policy=TRIM)
    for rep in (res.start, res.end):
        assert rep.feasible_mass == pytest.approx(0.2)  # NOT requested - removed (== 1.0)
        assert rep.removed_state_mass == 0.0
        assert rep.trimmed_requested_mass == pytest.approx(0.8)
        assert _identity(rep) == pytest.approx(rep.requested_mass)
        assert rep.feasible_mass == pytest.approx(res.final_flow_check.flow_value * 0.2)


def test_mass_identity_with_zeroed_states_and_partial_trims():
    g, p, q = _zeroed_case()  # A1 zeroed (removed), A0 and B1 trimmed
    res = resolve_endpoint_distributions(g, p, q, policy=TRIM)
    assert res.start.removed_state_mass == pytest.approx(0.4)
    assert res.start.feasible_mass == pytest.approx(0.4)
    assert res.start.trimmed_requested_mass == pytest.approx(0.2)
    assert res.end.removed_state_mass == 0.0
    assert res.end.feasible_mass == pytest.approx(0.4)
    assert res.end.trimmed_requested_mass == pytest.approx(0.6)
    for rep in (res.start, res.end):
        assert _identity(rep) == pytest.approx(rep.requested_mass)


def test_retained_mass_is_in_requested_units_after_level1_removal_then_trim():
    # Level 1 removes A2/B2 (0.2 each) and renormalizes; the trim then runs in the
    # renormalized units. feasible_mass must still be expressed in REQUESTED units.
    g = _graph([A0, A1, A2], [B0, B1, B2], DISJOINT)
    p = _dist(0, {A0: 0.7, A1: 0.1, A2: 0.2})
    q = _dist(1, {B0: 0.1, B1: 0.7, B2: 0.2})
    res = resolve_endpoint_distributions(g, p, q, policy=COND)
    for rep in (res.start, res.end):
        assert rep.removed_state_mass == pytest.approx(0.2)
        assert rep.feasible_mass == pytest.approx(0.2)
        assert rep.trimmed_requested_mass == pytest.approx(0.6)
        assert _identity(rep) == pytest.approx(rep.requested_mass)


def test_whole_state_removal_identity_has_no_trimmed_mass():
    g, p, q = _counter_example()
    res = resolve_endpoint_distributions(g, p, q, policy=DROP)
    for rep in (res.start, res.end):
        assert rep.trimmed_requested_mass == 0.0
        assert _identity(rep) == pytest.approx(rep.requested_mass)


def test_interior_blend_does_not_change_retained_mass():
    g, p, q = _tight_as_requested_case()
    res = resolve_endpoint_distributions(g, p, q, policy=DROP, interior_margin=0.1)
    for rep in (res.start, res.end):
        assert rep.feasible_mass == pytest.approx(1.0)  # nothing removed or trimmed
        assert rep.trimmed_requested_mass == 0.0 and rep.interior_margin == 0.1
        assert rep.total_variation_shift > 0.0  # the blend is visible here instead


def test_serialization_includes_feasible_mass_and_classification():
    import json

    g, p, q = _counter_example()
    res = resolve_endpoint_distributions(g, p, q, policy=TRIM)
    data = res.to_dict()
    json.dumps(data)  # fully JSON-serializable
    assert data["request_modified"] is True and data["modifications"] == ["partial_trim"]
    assert data["termination_reason"] == "feasible_after_partial_trim"
    for side in ("start", "end"):
        entry = data[side]
        assert entry["feasible_mass"] == pytest.approx(0.2)
        assert entry["retained_mass"] == pytest.approx(0.2)
        assert entry["trimmed_requested_mass"] == pytest.approx(0.8)
        assert entry["removed_state_mass"] == 0.0 and entry["modified"] is True
    exact = resolve_endpoint_distributions(*_exact_case()).to_dict()
    assert exact["request_modified"] is False and exact["modifications"] == []
    assert exact["start"]["feasible_mass"] == pytest.approx(1.0)


def test_feasible_mass_fallback_applies_the_full_accounting_identity():
    # A report built WITHOUT a routed total must not overstate retained mass.
    layer_s = Layer(0, (A0,))
    rep = ef.EndpointMassReport(
        side="start",
        requested=EndpointDistribution(layer_s, (1.0,)),
        effective=EndpointDistribution(layer_s, (1.0,)),
        requested_mass=1.0,
        removed=(
            ef.RemovedEndpointState("start", A1, 0.20, "x", "global_feasibility", 1),
        ),
        trimmed_requested_mass=0.15,
    )
    assert rep.retained_mass is None
    assert rep.feasible_mass == pytest.approx(0.65)  # not 0.80
    assert rep.removed_state_mass + rep.feasible_mass + rep.trimmed_requested_mass == pytest.approx(1.0)
    # an explicit routed total always wins
    assert ef.replace(rep, retained_mass=0.6).feasible_mass == 0.6
    # fully clipped at zero rather than negative
    assert ef.replace(rep, trimmed_requested_mass=5.0).feasible_mass == 0.0
