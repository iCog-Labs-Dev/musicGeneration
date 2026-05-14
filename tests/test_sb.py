import math
import unittest

import numpy as np

from aimusic.core.config import SBBackend, SBConfig
from aimusic.core.core_types import BeatState, Edge, EndpointDistribution, Layer
from aimusic.planning.graph import GraphDiagnostics, LayerBuildDiagnostics, SparseGraph
from aimusic.core.rng import RNGKey
from aimusic.planning.sb import (
    SBContractError,
    SBSolverError,
    _IndexedEdgeBucket,
    _NumpySBBackend,
    _require_non_empty_log_support,
    build_sb_problem,
    map_bridge_path,
    solve_sb,
    sample_bridge_path,
    solved_bridge_from_solution,
    uniform_bridge_from_graph,
    SBDiagnostics,
    SBEdgeMarginals,
    SBIterationRecord,
    SBNodeMarginals,
    compute_sb_diagnostics,
    solve_sb_with_history,
)


def _state(beat: int, groove: int = 0) -> BeatState:
    return BeatState(
        meter_id=0,
        beat_in_bar=beat,
        boundary_lvl=0,
        key_id=0,
        chord_id=0,
        role_id=0,
        head_id=0,
        groove_id=groove,
    )


def _minimal_diagnostics(layer_count: int) -> GraphDiagnostics:
    return GraphDiagnostics(
        layer_sizes=tuple(1 for _ in range(layer_count)),
        layer_diagnostics=tuple(
            LayerBuildDiagnostics(
                time_index=time_index,
                source_state_count=1,
                raw_candidate_count=1,
                unique_candidate_count=1,
                kept_candidate_count=1,
                raw_edge_count=1,
                kept_edge_count=1,
            )
            for time_index in range(max(1, layer_count - 1))
        ),
    )


def _valid_graph() -> tuple[SparseGraph, EndpointDistribution, EndpointDistribution]:
    s0 = _state(0, groove=0)
    s1 = _state(1, groove=1)
    s2 = _state(2, groove=0)

    l0 = Layer(time_index=0, states=(s0,))
    l1 = Layer(time_index=1, states=(s1,))
    l2 = Layer(time_index=2, states=(s2,))

    e0 = Edge(time_index=0, source=s0, target=s1, log_weight=-0.1)
    e1 = Edge(time_index=1, source=s1, target=s2, log_weight=-0.2)

    graph = SparseGraph(
        layers=(l0, l1, l2),
        edges_by_time=((e0,), (e1,)),
        diagnostics=_minimal_diagnostics(3),
    )
    pi0 = EndpointDistribution(layer=l0, probabilities=(1.0,))
    piT = EndpointDistribution(layer=l2, probabilities=(1.0,))
    return graph, pi0, piT


def _branching_graph() -> tuple[SparseGraph, EndpointDistribution, EndpointDistribution]:
    start = _state(0, groove=0)
    middle_a = _state(1, groove=1)
    middle_b = _state(1, groove=2)
    end_a = _state(2, groove=1)
    end_b = _state(2, groove=2)

    l0 = Layer(time_index=0, states=(start,))
    l1 = Layer(time_index=1, states=(middle_a, middle_b))
    l2 = Layer(time_index=2, states=(end_a, end_b))

    edges_t0 = (
        Edge(time_index=0, source=start, target=middle_a, log_weight=math.log(0.7)),
        Edge(time_index=0, source=start, target=middle_b, log_weight=math.log(0.3)),
    )
    edges_t1 = (
        Edge(time_index=1, source=middle_a, target=end_a, log_weight=math.log(0.8)),
        Edge(time_index=1, source=middle_a, target=end_b, log_weight=math.log(0.2)),
        Edge(time_index=1, source=middle_b, target=end_a, log_weight=math.log(0.1)),
        Edge(time_index=1, source=middle_b, target=end_b, log_weight=math.log(0.9)),
    )

    graph = SparseGraph(
        layers=(l0, l1, l2),
        edges_by_time=(edges_t0, edges_t1),
        diagnostics=GraphDiagnostics(
            layer_sizes=(1, 2, 2),
            layer_diagnostics=(
                LayerBuildDiagnostics(
                    time_index=0,
                    source_state_count=1,
                    raw_candidate_count=2,
                    unique_candidate_count=2,
                    kept_candidate_count=2,
                    raw_edge_count=2,
                    kept_edge_count=2,
                ),
                LayerBuildDiagnostics(
                    time_index=1,
                    source_state_count=2,
                    raw_candidate_count=4,
                    unique_candidate_count=4,
                    kept_candidate_count=4,
                    raw_edge_count=4,
                    kept_edge_count=4,
                ),
            ),
        ),
    )
    pi0 = EndpointDistribution(layer=l0, probabilities=(1.0,))
    piT = EndpointDistribution(layer=l2, probabilities=(0.4, 0.6))
    return graph, pi0, piT


def _near_degenerate_graph() -> tuple[SparseGraph, EndpointDistribution, EndpointDistribution]:
    start = _state(0, groove=0)
    middle_a = _state(1, groove=1)
    middle_b = _state(1, groove=2)
    end = _state(2, groove=3)

    l0 = Layer(time_index=0, states=(start,))
    l1 = Layer(time_index=1, states=(middle_a, middle_b))
    l2 = Layer(time_index=2, states=(end,))

    edges_t0 = (
        Edge(time_index=0, source=start, target=middle_a, log_weight=math.log(1.0 - 1e-12)),
        Edge(time_index=0, source=start, target=middle_b, log_weight=math.log(1e-12)),
    )
    edges_t1 = (
        Edge(time_index=1, source=middle_a, target=end, log_weight=0.0),
        Edge(time_index=1, source=middle_b, target=end, log_weight=0.0),
    )
    graph = SparseGraph(
        layers=(l0, l1, l2),
        edges_by_time=(edges_t0, edges_t1),
        diagnostics=GraphDiagnostics(
            layer_sizes=(1, 2, 1),
            layer_diagnostics=(
                LayerBuildDiagnostics(
                    time_index=0,
                    source_state_count=1,
                    raw_candidate_count=2,
                    unique_candidate_count=2,
                    kept_candidate_count=2,
                    raw_edge_count=2,
                    kept_edge_count=2,
                ),
                LayerBuildDiagnostics(
                    time_index=1,
                    source_state_count=2,
                    raw_candidate_count=2,
                    unique_candidate_count=2,
                    kept_candidate_count=2,
                    raw_edge_count=2,
                    kept_edge_count=2,
                ),
            ),
        ),
    )
    pi0 = EndpointDistribution(layer=l0, probabilities=(1.0,))
    piT = EndpointDistribution(layer=l2, probabilities=(1.0,))
    return graph, pi0, piT


def _zero_mass_dangling_support_graph() -> tuple[SparseGraph, EndpointDistribution, EndpointDistribution]:
    active_start = _state(0, groove=0)
    inactive_start = _state(0, groove=9)
    end = _state(1, groove=1)

    l0 = Layer(time_index=0, states=(active_start, inactive_start))
    l1 = Layer(time_index=1, states=(end,))
    graph = SparseGraph(
        layers=(l0, l1),
        edges_by_time=(
            (Edge(time_index=0, source=active_start, target=end, log_weight=0.0),),
        ),
        diagnostics=GraphDiagnostics(
            layer_sizes=(2, 1),
            layer_diagnostics=(
                LayerBuildDiagnostics(
                    time_index=0,
                    source_state_count=2,
                    raw_candidate_count=1,
                    unique_candidate_count=1,
                    kept_candidate_count=1,
                    raw_edge_count=1,
                    kept_edge_count=1,
                ),
            ),
        ),
    )
    pi0 = EndpointDistribution(layer=l0, probabilities=(1.0, 0.0))
    piT = EndpointDistribution(layer=l1, probabilities=(1.0,))
    return graph, pi0, piT


class TestSBProblemContract(unittest.TestCase):
    def test_build_sb_problem_happy_path(self):
        graph, pi0, piT = _valid_graph()

        problem = build_sb_problem(
            graph=graph,
            pi0=pi0,
            piT=piT,
            sb_config=SBConfig(horizon_t=2),
        )

        self.assertEqual(problem.graph, graph)
        self.assertEqual(problem.pi0, pi0)
        self.assertEqual(problem.piT, piT)
        self.assertEqual(problem.diagnostics.horizon_t, 2)
        self.assertEqual(problem.diagnostics.layer_sizes, (1, 1, 1))
        self.assertEqual(problem.diagnostics.edge_counts_by_time, (1, 1))
        self.assertEqual(problem.diagnostics.total_edge_count, 2)

    def test_fails_on_non_contiguous_layer_time_indices(self):
        graph, pi0, piT = _valid_graph()
        s2 = _state(2, groove=0)
        bad_layer = Layer(time_index=3, states=(s2,))
        bad_graph = SparseGraph(
            layers=(graph.layers[0], graph.layers[1], bad_layer),
            edges_by_time=graph.edges_by_time,
            diagnostics=graph.diagnostics,
        )

        with self.assertRaises(SBContractError):
            build_sb_problem(bad_graph, pi0, piT)

    def test_fails_on_edges_length_mismatch(self):
        graph, pi0, piT = _valid_graph()
        bad_graph = SparseGraph(
            layers=graph.layers,
            edges_by_time=(graph.edges_by_time[0],),
            diagnostics=graph.diagnostics,
        )

        with self.assertRaises(SBContractError):
            build_sb_problem(bad_graph, pi0, piT)

    def test_fails_on_edge_bucket_time_mismatch(self):
        graph, pi0, piT = _valid_graph()
        wrong_time = Edge(
            time_index=9,
            source=graph.layers[0].states[0],
            target=graph.layers[1].states[0],
            log_weight=-0.1,
        )
        bad_graph = SparseGraph(
            layers=graph.layers,
            edges_by_time=((wrong_time,), graph.edges_by_time[1]),
            diagnostics=graph.diagnostics,
        )

        with self.assertRaises(SBContractError):
            build_sb_problem(bad_graph, pi0, piT)

    def test_fails_on_edge_target_not_in_next_layer(self):
        graph, pi0, piT = _valid_graph()
        alien_target = _state(7, groove=5)
        bad_edge = Edge(
            time_index=0,
            source=graph.layers[0].states[0],
            target=alien_target,
            log_weight=-0.3,
        )
        bad_graph = SparseGraph(
            layers=graph.layers,
            edges_by_time=((bad_edge,), graph.edges_by_time[1]),
            diagnostics=graph.diagnostics,
        )

        with self.assertRaises(SBContractError):
            build_sb_problem(bad_graph, pi0, piT)

    def test_fails_when_pi0_does_not_match_first_layer(self):
        graph, _, piT = _valid_graph()
        wrong_first_layer = Layer(time_index=0, states=(_state(0, groove=9),))
        bad_pi0 = EndpointDistribution(layer=wrong_first_layer, probabilities=(1.0,))

        with self.assertRaises(SBContractError):
            build_sb_problem(graph, bad_pi0, piT)

    def test_fails_when_piT_does_not_match_final_layer(self):
        graph, pi0, _ = _valid_graph()
        wrong_last_layer = Layer(time_index=2, states=(_state(2, groove=9),))
        bad_piT = EndpointDistribution(layer=wrong_last_layer, probabilities=(1.0,))

        with self.assertRaises(SBContractError):
            build_sb_problem(graph, pi0, bad_piT)

    def test_fails_when_intermediate_layer_has_no_outgoing_support(self):
        graph, pi0, piT = _valid_graph()
        no_outgoing_graph = SparseGraph(
            layers=graph.layers,
            edges_by_time=(graph.edges_by_time[0], tuple()),
            diagnostics=graph.diagnostics,
        )

        with self.assertRaises(SBContractError):
            build_sb_problem(no_outgoing_graph, pi0, piT)

    def test_fails_when_final_layer_has_no_incoming_support(self):
        graph, pi0, piT = _valid_graph()
        no_incoming_graph = SparseGraph(
            layers=graph.layers,
            edges_by_time=(tuple(), graph.edges_by_time[1]),
            diagnostics=graph.diagnostics,
        )

        with self.assertRaises(SBContractError):
            build_sb_problem(no_incoming_graph, pi0, piT)

    def test_fails_on_horizon_mismatch(self):
        graph, pi0, piT = _valid_graph()

        with self.assertRaises(SBContractError):
            build_sb_problem(
                graph,
                pi0,
                piT,
                sb_config=SBConfig(horizon_t=99),
            )

    def test_fails_when_piT_positive_mass_is_unreachable_from_pi0(self):
        start = _state(0, groove=0)
        reachable_end = _state(1, groove=1)
        unreachable_end = _state(1, groove=2)
        l0 = Layer(time_index=0, states=(start,))
        l1 = Layer(time_index=1, states=(reachable_end, unreachable_end))
        graph = SparseGraph(
            layers=(l0, l1),
            edges_by_time=(
                (Edge(time_index=0, source=start, target=reachable_end, log_weight=0.0),),
            ),
            diagnostics=_minimal_diagnostics(2),
        )
        pi0 = EndpointDistribution(layer=l0, probabilities=(1.0,))
        piT = EndpointDistribution(layer=l1, probabilities=(0.5, 0.5))

        with self.assertRaises(SBContractError):
            build_sb_problem(graph, pi0, piT)

    def test_fails_when_pi0_positive_mass_cannot_reach_piT(self):
        start_a = _state(0, groove=0)
        start_b = _state(0, groove=1)
        terminal = _state(1, groove=2)
        l0 = Layer(time_index=0, states=(start_a, start_b))
        l1 = Layer(time_index=1, states=(terminal,))
        graph = SparseGraph(
            layers=(l0, l1),
            edges_by_time=(
                (Edge(time_index=0, source=start_a, target=terminal, log_weight=0.0),),
            ),
            diagnostics=GraphDiagnostics(
                layer_sizes=(2, 1),
                layer_diagnostics=(
                    LayerBuildDiagnostics(
                        time_index=0,
                        source_state_count=2,
                        raw_candidate_count=1,
                        unique_candidate_count=1,
                        kept_candidate_count=1,
                        raw_edge_count=1,
                        kept_edge_count=1,
                    ),
                ),
            ),
        )
        pi0 = EndpointDistribution(layer=l0, probabilities=(0.5, 0.5))
        piT = EndpointDistribution(layer=l1, probabilities=(1.0,))

        with self.assertRaises(SBContractError):
            build_sb_problem(graph, pi0, piT)

    def test_build_is_pure_and_deterministic(self):
        graph, pi0, piT = _valid_graph()

        first = build_sb_problem(graph, pi0, piT)
        second = build_sb_problem(graph, pi0, piT)

        self.assertEqual(first, second)
        self.assertEqual(graph.layers[0].states[0].beat_in_bar, 0)


class TestSparseBackendHelpers(unittest.TestCase):
    def test_logsumexp_underflow_guard_checks_relative_shift(self):
        with self.assertRaises(SBSolverError):
            _NumpySBBackend.logsumexp(
                np.asarray((0.0, -200.0), dtype=float),
                underflow_floor=-100.0,
                context="unit_test_relative_shift",
            )

    def test_reduce_by_source_matches_dense_reference(self):
        bucket = _IndexedEdgeBucket(
            time_index=0,
            source_size=2,
            target_size=3,
            source_indices=(0, 0, 1),
            target_indices=(0, 1, 2),
            log_kernel_weights=(math.log(0.5), math.log(0.25), math.log(0.9)),
        )
        next_values = np.asarray(
            (math.log(0.2), math.log(0.8), math.log(0.3)),
            dtype=float,
        )

        reduced = _NumpySBBackend.reduce_by_source(bucket, next_values)

        expected_0 = math.log(0.5 * 0.2 + 0.25 * 0.8)
        expected_1 = math.log(0.9 * 0.3)
        self.assertTrue(np.allclose(reduced, np.asarray((expected_0, expected_1))))

    def test_reduce_by_target_matches_dense_reference(self):
        bucket = _IndexedEdgeBucket(
            time_index=0,
            source_size=3,
            target_size=2,
            source_indices=(0, 1, 2, 2),
            target_indices=(0, 0, 0, 1),
            log_kernel_weights=(
                math.log(0.6),
                math.log(0.2),
                math.log(0.1),
                math.log(0.7),
            ),
        )
        prev_values = np.asarray(
            (math.log(0.5), math.log(0.4), math.log(0.9)),
            dtype=float,
        )

        reduced = _NumpySBBackend.reduce_by_target(bucket, prev_values)

        expected_0 = math.log(0.6 * 0.5 + 0.2 * 0.4 + 0.1 * 0.9)
        expected_1 = math.log(0.7 * 0.9)
        self.assertTrue(np.allclose(reduced, np.asarray((expected_0, expected_1))))

    def test_empty_support_guard_rejects_all_negative_inf(self):
        with self.assertRaises(SBSolverError):
            _require_non_empty_log_support(
                "test_values",
                np.asarray((float("-inf"), float("-inf")), dtype=float),
            )


class TestSBSolver(unittest.TestCase):
    def test_solve_sb_converges_on_tiny_graph(self):
        graph, pi0, piT = _valid_graph()
        problem = build_sb_problem(graph, pi0, piT)

        solution = solve_sb(problem)

        self.assertTrue(solution.trace.converged)
        self.assertEqual(solution.trace.iterations, 1)
        self.assertAlmostEqual(solution.trace.final_max_delta, 0.0)
        
        # Test basic property that forward potentials + backward potentials 
        # should sum to the normalized log-distribution at endpoints
        start_mass = np.asarray(solution.log_forward_potentials[0]) + np.asarray(
            solution.log_backward_potentials[0]
        )
        end_mass = np.asarray(solution.log_forward_potentials[-1]) + np.asarray(
            solution.log_backward_potentials[-1]
        )
        self.assertTrue(np.allclose(start_mass, np.asarray((0.0,))))
        self.assertTrue(np.allclose(end_mass, np.asarray((0.0,))))

    def test_solve_sb_returns_endpoint_consistent_potentials(self):
        graph, pi0, piT = _branching_graph()
        problem = build_sb_problem(graph, pi0, piT)

        solution = solve_sb(problem)

        self.assertTrue(solution.trace.converged)
        start_mass = np.asarray(solution.log_forward_potentials[0]) + np.asarray(
            solution.log_backward_potentials[0]
        )
        end_mass = np.asarray(solution.log_forward_potentials[-1]) + np.asarray(
            solution.log_backward_potentials[-1]
        )
        self.assertTrue(
            np.allclose(start_mass, np.log(np.asarray(problem.pi0.probabilities)))
        )
        self.assertTrue(
            np.allclose(end_mass, np.log(np.asarray(problem.piT.probabilities)))
        )

    def test_solve_sb_is_deterministic(self):
        graph, pi0, piT = _branching_graph()
        problem = build_sb_problem(graph, pi0, piT)

        first = solve_sb(problem)
        second = solve_sb(problem)

        self.assertEqual(first, second)
        self.assertEqual(first.trace.residual_history, second.trace.residual_history)

    def test_solve_sb_reports_non_convergence_without_raising(self):
        graph, pi0, piT = _branching_graph()
        problem = build_sb_problem(
            graph,
            pi0,
            piT,
            sb_config=SBConfig(horizon_t=2, max_iterations=1, tolerance=1e-15),
        )

        solution = solve_sb(problem)

        self.assertEqual(solution.trace.iterations, 1)
        self.assertFalse(solution.trace.converged)
        self.assertGreater(solution.trace.final_max_delta, 0.0)

    def test_solve_sb_can_raise_on_non_convergence_when_configured(self):
        graph, pi0, piT = _branching_graph()
        problem = build_sb_problem(
            graph,
            pi0,
            piT,
            sb_config=SBConfig(
                horizon_t=2,
                max_iterations=1,
                tolerance=1e-15,
                raise_on_non_convergence=True,
            ),
        )

        with self.assertRaises(SBSolverError):
            solve_sb(problem)

    def test_solve_sb_raises_when_underflow_floor_is_crossed(self):
        graph, pi0, piT = _valid_graph()
        with self.assertRaises(ValueError):
            build_sb_problem(
                graph,
                pi0,
                piT,
                sb_config=SBConfig(horizon_t=2, log_underflow_floor=1.0),
            )

    def test_solve_sb_rejects_unsupported_backend(self):
        graph, pi0, piT = _valid_graph()
        problem = build_sb_problem(
            graph,
            pi0,
            piT,
            sb_config=SBConfig(horizon_t=2, backend_selection=SBBackend.JAX),
        )

        with self.assertRaises(NotImplementedError):
            solve_sb(problem)

    def test_solution_exposes_marginals_and_convergence_history(self):
        graph, pi0, piT = _branching_graph()
        problem = build_sb_problem(graph, pi0, piT)

        solution = solve_sb(problem)

        self.assertIsNotNone(solution.marginals)
        self.assertEqual(len(solution.marginals.node_marginals_by_layer), len(graph.layers))
        self.assertEqual(len(solution.marginals.edge_marginals_by_time), len(graph.edges_by_time))
        for layer_probs in solution.marginals.node_marginals_by_layer:
            self.assertAlmostEqual(sum(layer_probs), 1.0)
        self.assertEqual(solution.trace.iterations, len(solution.trace.residual_history))

    def test_solved_bridge_normalizes_per_source_state(self):
        graph, pi0, piT = _branching_graph()
        solution = solve_sb(build_sb_problem(graph, pi0, piT))

        bridge = solved_bridge_from_solution(solution)

        grouped = {}
        for edge, prob in zip(graph.edges_by_time[1], bridge.edge_probabilities_by_time[1]):
            grouped.setdefault(edge.source, 0.0)
            grouped[edge.source] += prob
        for total in grouped.values():
            self.assertAlmostEqual(total, 1.0)

    def test_solve_sb_converges_on_near_degenerate_graph(self):
        graph, pi0, piT = _near_degenerate_graph()

        solution = solve_sb(build_sb_problem(graph, pi0, piT))

        self.assertTrue(solution.trace.converged)
        self.assertTrue(all(math.isfinite(value) for value in solution.trace.residual_history))

    def test_solve_sb_allows_zero_mass_dangling_support(self):
        graph, pi0, piT = _zero_mass_dangling_support_graph()

        solution = solve_sb(build_sb_problem(graph, pi0, piT))
        bridge = solution.to_bridge()

        self.assertTrue(solution.trace.converged)
        self.assertEqual(solution.marginals.node_marginals_by_layer[0], (1.0, 0.0))
        self.assertEqual(bridge.edge_probabilities_by_time[0], (1.0,))


class TestSchrodingerBridgeSampler(unittest.TestCase):
    def setUp(self):
        self.start_state = _state(0, groove=0)
        mid_state_a = _state(1, groove=1)
        mid_state_b = _state(1, groove=2)
        self.end_state = _state(2, groove=1)
        
        self.start_layer = Layer(time_index=0, states=(self.start_state,))
        layer1 = Layer(time_index=1, states=(mid_state_a, mid_state_b))
        layer2 = Layer(time_index=2, states=(self.end_state,))

        edges_t0 = (
            Edge(time_index=0, source=self.start_state, target=mid_state_a, log_weight=0.0),
            Edge(time_index=0, source=self.start_state, target=mid_state_b, log_weight=0.0),
        )
        edges_t1 = (
            Edge(time_index=1, source=mid_state_a, target=self.end_state, log_weight=0.0),
            Edge(time_index=1, source=mid_state_b, target=self.end_state, log_weight=0.0),
        )

        self.graph = SparseGraph(
            layers=(self.start_layer, layer1, layer2),
            edges_by_time=(edges_t0, edges_t1),
            diagnostics=_minimal_diagnostics(3),
        )

    def test_sampling_is_reproducible_under_seed(self):
        bridge = uniform_bridge_from_graph(self.graph)
        key = RNGKey(seed=123)
        sample_a, _ = sample_bridge_path(bridge, key, include_edges=True, include_debug=True)
        sample_b, _ = sample_bridge_path(bridge, key, include_edges=True, include_debug=True)
        
        self.assertEqual(sample_a.path, sample_b.path)
        self.assertEqual(sample_a.edges, sample_b.edges)
        self.assertEqual(sample_a.debug, sample_b.debug)

    def test_sampled_path_follows_valid_edges(self):
        bridge = uniform_bridge_from_graph(self.graph)
        sampled, _ = sample_bridge_path(bridge, RNGKey(seed=9), include_edges=True)

        self.assertEqual(len(sampled.path), len(self.graph.layers))
        self.assertEqual(len(sampled.edges), len(self.graph.layers) - 1)
        self.assertEqual(sampled.path[0], self.start_state)
        self.assertEqual(sampled.path[-1], self.end_state)

        for t, edge in enumerate(sampled.edges):
            self.assertEqual(edge.time_index, t)
            self.assertEqual(edge.source, sampled.path[t])
            self.assertEqual(edge.target, sampled.path[t + 1])
            self.assertIn(edge, self.graph.edges_by_time[t])


class TestBridgeTrajectoryExtraction(unittest.TestCase):
    def test_map_bridge_path_returns_expected_best_path(self):
        graph, pi0, piT = _branching_graph()
        solution = solve_sb(build_sb_problem(graph, pi0, piT))
        bridge = solution.to_bridge()

        path, score = map_bridge_path(bridge)

        expected_path = (
            graph.layers[0].states[0],
            graph.layers[1].states[1],
            graph.layers[2].states[1],
        )
        self.assertEqual(path, expected_path)
        self.assertTrue(math.isfinite(score))

    def test_sampling_is_reproducible_from_solved_bridge(self):
        graph, pi0, piT = _branching_graph()
        bridge = solve_sb(build_sb_problem(graph, pi0, piT)).to_bridge()
        key = RNGKey(seed=77)

        sample_a, _ = sample_bridge_path(bridge, key, include_edges=True, include_debug=True)
        sample_b, _ = sample_bridge_path(bridge, key, include_edges=True, include_debug=True)

        self.assertEqual(sample_a, sample_b)
        for step in sample_a.debug:
            self.assertGreaterEqual(step["edge_probability"], 0.0)
            self.assertLessEqual(step["edge_probability"], 1.0)


# ===========================================================================
# SB-08: Marginals, iteration history, and convergence diagnostics
# ===========================================================================


class TestSBDiagnosticsNodeMarginals(unittest.TestCase):
    """Node marginals are correctly derived from forward/backward potentials."""

    def test_node_marginals_sum_to_one_per_layer(self):
        graph, pi0, piT = _valid_graph()
        problem = build_sb_problem(graph, pi0, piT)
        solution = solve_sb(problem)

        diag = compute_sb_diagnostics(solution)

        for t, layer_probs in enumerate(diag.node_marginals.marginals):
            total = sum(layer_probs)
            self.assertAlmostEqual(
                total, 1.0, places=9,
                msg=f"Layer {t} marginals do not sum to 1.0 (got {total}).",
            )

    def test_node_marginals_log_and_prob_are_consistent(self):
        graph, pi0, piT = _branching_graph()
        problem = build_sb_problem(graph, pi0, piT)
        solution = solve_sb(problem)

        diag = compute_sb_diagnostics(solution)

        for t, (log_layer, prob_layer) in enumerate(
            zip(diag.node_marginals.log_marginals, diag.node_marginals.marginals)
        ):
            for i, (lv, pv) in enumerate(zip(log_layer, prob_layer)):
                if math.isfinite(lv):
                    self.assertAlmostEqual(
                        math.exp(lv), pv, places=9,
                        msg=f"Mismatch at layer {t}, node {i}: "
                            f"exp({lv}) != {pv}.",
                    )
                else:
                    self.assertEqual(
                        pv, 0.0,
                        msg=f"Layer {t}, node {i}: -inf log should map to 0.0 prob.",
                    )

    def test_node_marginals_shape_matches_graph_layers(self):
        graph, pi0, piT = _branching_graph()
        problem = build_sb_problem(graph, pi0, piT)
        solution = solve_sb(problem)

        diag = compute_sb_diagnostics(solution)

        layer_sizes = problem.diagnostics.layer_sizes
        self.assertEqual(len(diag.node_marginals.marginals), len(layer_sizes))
        for t, expected_size in enumerate(layer_sizes):
            self.assertEqual(
                len(diag.node_marginals.marginals[t]), expected_size,
                msg=f"Layer {t} marginals size mismatch.",
            )

    def test_single_path_graph_marginals_are_all_one(self):
        # On a chain graph with a single state per layer and uniform endpoints,
        # every node marginal must be exactly 1.0.
        graph, pi0, piT = _valid_graph()
        problem = build_sb_problem(graph, pi0, piT)
        solution = solve_sb(problem)

        diag = compute_sb_diagnostics(solution)

        for t, layer_probs in enumerate(diag.node_marginals.marginals):
            for i, p in enumerate(layer_probs):
                self.assertAlmostEqual(
                    p, 1.0, places=9,
                    msg=f"Layer {t}, node {i}: expected marginal 1.0, got {p}.",
                )

    def test_branching_graph_endpoint_marginals_match_piT(self):
        # The final-layer marginals must match the piT probabilities.
        graph, pi0, piT = _branching_graph()
        problem = build_sb_problem(graph, pi0, piT)
        solution = solve_sb(problem)

        diag = compute_sb_diagnostics(solution)

        final_probs = diag.node_marginals.marginals[-1]
        expected = problem.piT.probabilities
        for i, (got, exp) in enumerate(zip(final_probs, expected)):
            self.assertAlmostEqual(
                got, exp, places=6,
                msg=f"Final layer node {i}: marginal {got} != piT prob {exp}.",
            )

    def test_branching_graph_initial_marginals_match_pi0(self):
        # The first-layer marginals must match the pi0 probabilities.
        graph, pi0, piT = _branching_graph()
        problem = build_sb_problem(graph, pi0, piT)
        solution = solve_sb(problem)

        diag = compute_sb_diagnostics(solution)

        initial_probs = diag.node_marginals.marginals[0]
        expected = problem.pi0.probabilities
        for i, (got, exp) in enumerate(zip(initial_probs, expected)):
            self.assertAlmostEqual(
                got, exp, places=6,
                msg=f"Initial layer node {i}: marginal {got} != pi0 prob {exp}.",
            )


class TestSBDiagnosticsEdgeMarginals(unittest.TestCase):
    """Edge marginals are correctly derived and structurally consistent."""

    def test_edge_marginals_not_computed_by_default(self):
        graph, pi0, piT = _valid_graph()
        problem = build_sb_problem(graph, pi0, piT)
        solution = solve_sb(problem)

        diag = compute_sb_diagnostics(solution)

        self.assertIsNone(diag.edge_marginals)

    def test_edge_marginals_computed_when_requested(self):
        graph, pi0, piT = _valid_graph()
        problem = build_sb_problem(graph, pi0, piT)
        solution = solve_sb(problem)

        diag = compute_sb_diagnostics(solution, include_edge_marginals=True)

        self.assertIsNotNone(diag.edge_marginals)
        self.assertIsInstance(diag.edge_marginals, SBEdgeMarginals)

    def test_edge_marginals_bucket_count_matches_graph(self):
        graph, pi0, piT = _branching_graph()
        problem = build_sb_problem(graph, pi0, piT)
        solution = solve_sb(problem)

        diag = compute_sb_diagnostics(solution, include_edge_marginals=True)

        expected_buckets = len(graph.edges_by_time)
        self.assertEqual(
            len(diag.edge_marginals.edge_marginals), expected_buckets,
        )

    def test_edge_marginals_count_per_bucket_matches_edges(self):
        graph, pi0, piT = _branching_graph()
        problem = build_sb_problem(graph, pi0, piT)
        solution = solve_sb(problem)

        diag = compute_sb_diagnostics(solution, include_edge_marginals=True)

        for t, (edge_group, marginal_bucket) in enumerate(
            zip(graph.edges_by_time, diag.edge_marginals.edge_marginals)
        ):
            self.assertEqual(
                len(marginal_bucket), len(edge_group),
                msg=f"Bucket {t}: edge count mismatch.",
            )

    def test_edge_marginals_are_non_negative(self):
        graph, pi0, piT = _branching_graph()
        problem = build_sb_problem(graph, pi0, piT)
        solution = solve_sb(problem)

        diag = compute_sb_diagnostics(solution, include_edge_marginals=True)

        for t, bucket in enumerate(diag.edge_marginals.edge_marginals):
            for em in bucket:
                self.assertGreaterEqual(
                    em.marginal, 0.0,
                    msg=f"Bucket {t}: negative edge marginal {em.marginal}.",
                )

    def test_edge_marginals_sum_to_one_across_all_buckets(self):
        # The sum of all edge marginals across all buckets should equal T
        # (one unit of mass per time step), but since each bucket covers one
        # transition, the sum over a single bucket equals the total node mass
        # at the source layer (= 1.0).  We verify the per-bucket sum.
        graph, pi0, piT = _branching_graph()
        problem = build_sb_problem(graph, pi0, piT)
        solution = solve_sb(problem)

        diag = compute_sb_diagnostics(solution, include_edge_marginals=True)

        for t, bucket in enumerate(diag.edge_marginals.edge_marginals):
            bucket_sum = sum(em.marginal for em in bucket)
            self.assertAlmostEqual(
                bucket_sum, 1.0, places=6,
                msg=f"Bucket {t}: edge marginals sum to {bucket_sum}, expected 1.0.",
            )

    def test_edge_marginals_log_and_prob_consistent(self):
        graph, pi0, piT = _branching_graph()
        problem = build_sb_problem(graph, pi0, piT)
        solution = solve_sb(problem)

        diag = compute_sb_diagnostics(solution, include_edge_marginals=True)

        for t, bucket in enumerate(diag.edge_marginals.edge_marginals):
            for em in bucket:
                if math.isfinite(em.log_marginal):
                    self.assertAlmostEqual(
                        math.exp(em.log_marginal), em.marginal, places=9,
                        msg=f"Bucket {t}: log/prob mismatch for edge "
                            f"({em.source_index}->{em.target_index}).",
                    )
                else:
                    self.assertEqual(em.marginal, 0.0)

    def test_single_path_graph_edge_marginals_are_all_one(self):
        # On a chain with one state per layer, every edge marginal must be 1.0.
        graph, pi0, piT = _valid_graph()
        problem = build_sb_problem(graph, pi0, piT)
        solution = solve_sb(problem)

        diag = compute_sb_diagnostics(solution, include_edge_marginals=True)

        for t, bucket in enumerate(diag.edge_marginals.edge_marginals):
            for em in bucket:
                self.assertAlmostEqual(
                    em.marginal, 1.0, places=9,
                    msg=f"Bucket {t}: expected edge marginal 1.0, got {em.marginal}.",
                )


class TestSBDiagnosticsIterationHistory(unittest.TestCase):
    """Iteration history is captured and structured correctly."""

    def test_history_not_present_without_solve_with_history(self):
        graph, pi0, piT = _valid_graph()
        problem = build_sb_problem(graph, pi0, piT)
        solution = solve_sb(problem)

        diag = compute_sb_diagnostics(solution)

        self.assertIsNone(diag.iteration_history)

    def test_solve_with_history_returns_solution_and_history(self):
        graph, pi0, piT = _branching_graph()
        problem = build_sb_problem(graph, pi0, piT)

        solution, history = solve_sb_with_history(problem)

        self.assertIsInstance(solution, type(solve_sb(problem)))
        self.assertIsInstance(history, tuple)
        self.assertGreater(len(history), 0)

    def test_history_records_are_sbiterationrecord_instances(self):
        graph, pi0, piT = _branching_graph()
        problem = build_sb_problem(graph, pi0, piT)

        _, history = solve_sb_with_history(problem)

        for record in history:
            self.assertIsInstance(record, SBIterationRecord)

    def test_history_iteration_indices_are_sequential(self):
        graph, pi0, piT = _branching_graph()
        problem = build_sb_problem(graph, pi0, piT)

        _, history = solve_sb_with_history(problem)

        for expected_idx, record in enumerate(history, start=1):
            self.assertEqual(
                record.iteration, expected_idx,
                msg=f"Expected iteration {expected_idx}, got {record.iteration}.",
            )

    def test_history_max_deltas_are_finite_and_non_negative(self):
        graph, pi0, piT = _branching_graph()
        problem = build_sb_problem(graph, pi0, piT)

        _, history = solve_sb_with_history(problem)

        for record in history:
            self.assertTrue(
                math.isfinite(record.max_delta),
                msg=f"max_delta is not finite at iteration {record.iteration}.",
            )
            self.assertGreaterEqual(record.max_delta, 0.0)

    def test_history_passed_to_diagnostics_is_accessible(self):
        graph, pi0, piT = _branching_graph()
        problem = build_sb_problem(graph, pi0, piT)

        solution, history = solve_sb_with_history(problem)
        diag = compute_sb_diagnostics(solution, iteration_history=history)

        self.assertIsNotNone(diag.iteration_history)
        self.assertEqual(len(diag.iteration_history), len(history))

    def test_history_length_matches_iterations_run(self):
        graph, pi0, piT = _branching_graph()
        problem = build_sb_problem(graph, pi0, piT)

        solution, history = solve_sb_with_history(problem)

        self.assertEqual(len(history), solution.trace.iterations)

    def test_history_final_delta_matches_solution_trace(self):
        graph, pi0, piT = _branching_graph()
        problem = build_sb_problem(graph, pi0, piT)

        solution, history = solve_sb_with_history(problem)

        self.assertAlmostEqual(
            history[-1].max_delta,
            solution.trace.final_max_delta,
            places=12,
        )

    def test_history_non_converging_case_has_max_iterations_records(self):
        s0 = _state(0, 0)
        s1a = _state(1, 1); s1b = _state(1, 2)
        s2a = _state(2, 1); s2b = _state(2, 2)
        s3 = _state(3, 0)
        l0 = Layer(time_index=0, states=(s0,))
        l1 = Layer(time_index=1, states=(s1a, s1b))
        l2 = Layer(time_index=2, states=(s2a, s2b))
        l3 = Layer(time_index=3, states=(s3,))
        edges0 = (
            Edge(0, s0, s1a, math.log(0.9)),
            Edge(0, s0, s1b, math.log(0.1)),
        )
        edges1 = (
            Edge(1, s1a, s2a, math.log(0.3)),
            Edge(1, s1a, s2b, math.log(0.7)),
            Edge(1, s1b, s2a, math.log(0.6)),
            Edge(1, s1b, s2b, math.log(0.4)),
        )
        edges2 = (
            Edge(2, s2a, s3, math.log(0.5)),
            Edge(2, s2b, s3, math.log(0.5)),
        )
        from aimusic.planning.graph import GraphDiagnostics, LayerBuildDiagnostics
        diag = GraphDiagnostics(
            layer_sizes=(1, 2, 2, 1),
            layer_diagnostics=tuple(
                LayerBuildDiagnostics(
                    time_index=t,
                    source_state_count=1,
                    raw_candidate_count=1,
                    unique_candidate_count=1,
                    kept_candidate_count=1,
                    raw_edge_count=1,
                    kept_edge_count=1,
                )
                for t in range(3)
            ),
        )
        graph = SparseGraph(
            layers=(l0, l1, l2, l3),
            edges_by_time=(edges0, edges1, edges2),
            diagnostics=diag,
        )
        pi0 = EndpointDistribution(layer=l0, probabilities=(1.0,))
        piT = EndpointDistribution(layer=l3, probabilities=(1.0,))
        problem = build_sb_problem(
            graph, pi0, piT,
            sb_config=SBConfig(horizon_t=3, max_iterations=3, tolerance=1e-300),
        )

        solution, history = solve_sb_with_history(problem)

        self.assertFalse(solution.trace.converged)
        self.assertEqual(len(history), 3)


class TestSBDiagnosticsConvergenceFields(unittest.TestCase):
    """Convergence metadata on SBDiagnostics mirrors the solution trace."""

    def test_converged_field_mirrors_trace(self):
        graph, pi0, piT = _valid_graph()
        problem = build_sb_problem(graph, pi0, piT)
        solution = solve_sb(problem)

        diag = compute_sb_diagnostics(solution)

        self.assertEqual(diag.converged, solution.trace.converged)

    def test_iterations_run_mirrors_trace(self):
        graph, pi0, piT = _branching_graph()
        problem = build_sb_problem(graph, pi0, piT)
        solution = solve_sb(problem)

        diag = compute_sb_diagnostics(solution)

        self.assertEqual(diag.iterations_run, solution.trace.iterations)

    def test_final_max_delta_mirrors_trace(self):
        graph, pi0, piT = _branching_graph()
        problem = build_sb_problem(graph, pi0, piT)
        solution = solve_sb(problem)

        diag = compute_sb_diagnostics(solution)

        self.assertAlmostEqual(
            diag.final_max_delta, solution.trace.final_max_delta, places=12,
        )

    def test_non_converged_solution_reflected_in_diagnostics(self):
        graph, pi0, piT = _branching_graph()
        problem = build_sb_problem(
            graph, pi0, piT,
            sb_config=SBConfig(horizon_t=2, max_iterations=1, tolerance=1e-15),
        )
        solution = solve_sb(problem)

        diag = compute_sb_diagnostics(solution)

        self.assertFalse(diag.converged)
        self.assertEqual(diag.iterations_run, 1)


class TestSBDiagnosticsValidation(unittest.TestCase):
    """compute_sb_diagnostics rejects bad inputs cleanly."""

    def test_rejects_non_solution_input(self):
        with self.assertRaises(TypeError):
            compute_sb_diagnostics("not a solution")  # type: ignore[arg-type]

    def test_rejects_bad_iteration_history_entry(self):
        graph, pi0, piT = _valid_graph()
        problem = build_sb_problem(graph, pi0, piT)
        solution = solve_sb(problem)

        with self.assertRaises(TypeError):
            compute_sb_diagnostics(
                solution,
                iteration_history=("not a record",),  # type: ignore[arg-type]
            )

    def test_solve_with_history_rejects_non_problem(self):
        with self.assertRaises(TypeError):
            solve_sb_with_history("not a problem")  # type: ignore[arg-type]


class TestSBDiagnosticsDeterminism(unittest.TestCase):
    """Diagnostics are deterministic and pure."""

    def test_compute_diagnostics_is_deterministic(self):
        graph, pi0, piT = _branching_graph()
        problem = build_sb_problem(graph, pi0, piT)
        solution = solve_sb(problem)

        first = compute_sb_diagnostics(solution, include_edge_marginals=True)
        second = compute_sb_diagnostics(solution, include_edge_marginals=True)

        self.assertEqual(
            first.node_marginals.marginals,
            second.node_marginals.marginals,
        )
        self.assertEqual(
            first.edge_marginals.edge_marginals,
            second.edge_marginals.edge_marginals,
        )

    def test_solve_with_history_is_deterministic(self):
        graph, pi0, piT = _branching_graph()
        problem = build_sb_problem(graph, pi0, piT)

        sol1, hist1 = solve_sb_with_history(problem)
        sol2, hist2 = solve_sb_with_history(problem)

        self.assertEqual(sol1, sol2)
        self.assertEqual(hist1, hist2)

    def test_solve_with_history_matches_solve_sb(self):
        # solve_sb_with_history must produce the same SBSolution as solve_sb.
        graph, pi0, piT = _branching_graph()
        problem = build_sb_problem(graph, pi0, piT)

        plain_solution = solve_sb(problem)
        history_solution, _ = solve_sb_with_history(problem)

        self.assertEqual(plain_solution, history_solution)


if __name__ == "__main__":
    unittest.main()
