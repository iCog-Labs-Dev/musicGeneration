import pytest

import aimusic.planning.plans as plans
from aimusic.core.config import SBConfig
from aimusic.core.rng import RNGKey
from aimusic.planning.plans import (
    EndpointFallbackPolicy,
    EndpointMarginalError,
    EndpointMode,
    EndpointSolverNonConvergenceError,
    MethodARunConfig,
    generate_method_a_endpoints,
    run_method_a,
)

DROP = EndpointFallbackPolicy.DROP_AND_RENORMALIZE


def test_defaults_distribution_mode_and_unreachable_only_policy():
    cfg = MethodARunConfig(total_beats=8)
    assert cfg.endpoint_mode is EndpointMode.DISTRIBUTION
    # graph-blind endpoint generation -> unreachable states are dropped whole by default,
    # but a Level 2 mass conflict is never silently repaired
    assert cfg.unreachable_policy is EndpointFallbackPolicy.DROP_UNREACHABLE_ONLY
    assert cfg.endpoint_marginal_tolerance > 0 and cfg.endpoint_flow_tolerance > 0


def test_generate_endpoints_only_generates_distributions():
    eps = generate_method_a_endpoints(MethodARunConfig(total_beats=8))
    assert not hasattr(eps, "start_choice") and not hasattr(eps, "end_choice")
    assert sum(eps.pi0.probabilities) == pytest.approx(1.0)
    assert sum(eps.piT.probabilities) == pytest.approx(1.0)


@pytest.mark.parametrize("mode", list(EndpointMode))
def test_method_a_modes_share_one_validated_pipeline(mode):
    cfg = MethodARunConfig(
        total_beats=8, seed=3, endpoint_mode=mode, unreachable_policy=DROP
    )
    result, _ = run_method_a(cfg, key=RNGKey(seed=3))
    res = result.endpoint_resolution
    assert res.final_flow_check.feasible
    assert result.diagnostics.endpoint_mode == mode.value
    assert result.diagnostics.bridge_converged
    for rep in (result.start_mass, result.end_mass):
        assert sum(rep.effective.probabilities) == pytest.approx(1.0)
        assert rep.max_abs_realized_error <= cfg.endpoint_marginal_tolerance
    if mode is EndpointMode.DISTRIBUTION:
        assert result.diagnostics.chosen_start_state is None
        assert result.diagnostics.endpoint_selection_mode == "distribution"
    else:
        assert result.diagnostics.chosen_start_state is not None
        assert sum(p > 0 for p in result.start_mass.effective.probabilities) == 1
        assert sum(p > 0 for p in result.end_mass.effective.probabilities) == 1


def test_validation_failure_blocks_path_extraction(monkeypatch):
    def boom(*a, **k):
        raise EndpointSolverNonConvergenceError("did not converge")

    def forbidden(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("path extraction ran after validation failure")

    monkeypatch.setattr(plans, "solve_endpoint_bridge", boom)
    monkeypatch.setattr(plans, "map_bridge_path", forbidden)
    monkeypatch.setattr(plans, "sample_bridge_path", forbidden)
    for sampling in (False, True):
        cfg = MethodARunConfig(total_beats=8, use_sampling=sampling, unreachable_policy=DROP)
        with pytest.raises(EndpointSolverNonConvergenceError):
            run_method_a(cfg, key=RNGKey(seed=0))


def test_unconverged_real_solve_fails_marginal_validation():
    cfg = MethodARunConfig(
        total_beats=8,
        unreachable_policy=DROP,
        sb_config=SBConfig(horizon_t=8, max_iterations=1, tolerance=1e-12),
        endpoint_marginal_tolerance=1e-12,
        endpoint_flow_tolerance=1e-12,
        endpoint_capacity_epsilon=1e-16,
    )
    with pytest.raises((EndpointSolverNonConvergenceError, EndpointMarginalError)):
        run_method_a(cfg, key=RNGKey(seed=0))


def test_config_enforces_tolerance_relations():
    with pytest.raises(ValueError, match="flow_tolerance must be <= marginal_tolerance"):
        MethodARunConfig(
            total_beats=8, endpoint_marginal_tolerance=1e-12, endpoint_flow_tolerance=1e-9
        )
    with pytest.raises(ValueError, match="capacity_epsilon"):
        MethodARunConfig(total_beats=8, endpoint_capacity_epsilon=1e-9)
    with pytest.raises(ValueError):
        MethodARunConfig(total_beats=8, endpoint_interior_margin=1.0)


def test_endpoint_sampling_never_selects_zero_mass_state(monkeypatch):
    from aimusic.core.core_types import BeatState, EndpointDistribution, Layer

    states = tuple(BeatState(0, 0, 0, i, 0, 0, 0, 0) for i in range(3))
    dist = EndpointDistribution(Layer(0, states), (0.0, 1.0, 0.0))
    for u in (0.0, 0.5, 1.0):
        monkeypatch.setattr(plans, "random_unit", lambda key, u=u: (u, key))
        idx, _ = plans._sample_index_from_distribution(dist, RNGKey(seed=0))
        assert idx == 1