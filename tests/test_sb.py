import math
import unittest

import numpy as np

from aimusic.core.config import SBBackend, SBConfig
from aimusic.core.core_types import BeatState, Edge, EndpointDistribution, Layer
from aimusic.planning.graph import GraphDiagnostics, LayerBuildDiagnostics, SparseGraph
from aimusic.core.rng import RNGKey
from aimusic.planning.sb import (
    SBBackendError,
    SBContractError,
    SBSolverError,
    _IndexedEdgeBucket,
    _require_non_empty_log_support,
    build_sb_problem,
    map_bridge_path,
    solve_sb,
    sample_bridge_path,
    solved_bridge_from_solution,
    uniform_bridge_from_graph,
)
from aimusic.planning.sb_backend import DenseReferenceBackend, NumpySBBackend


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
            NumpySBBackend().logsumexp(
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
        kwargs = dict(
            log_weights=np.asarray(bucket.log_kernel_weights, dtype=float),
            gather_from=np.asarray(bucket.target_indices, dtype=np.int64),
            messages=next_values,
            scatter_to=np.asarray(bucket.source_indices, dtype=np.int64),
            out_size=bucket.source_size,
        )

        reduced = NumpySBBackend().scatter_logsumexp(**kwargs)
        dense = DenseReferenceBackend().scatter_logsumexp(**kwargs)

        expected_0 = math.log(0.5 * 0.2 + 0.25 * 0.8)
        expected_1 = math.log(0.9 * 0.3)
        self.assertTrue(np.allclose(reduced, np.asarray((expected_0, expected_1))))
        self.assertTrue(np.allclose(reduced, dense))

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
        kwargs = dict(
            log_weights=np.asarray(bucket.log_kernel_weights, dtype=float),
            gather_from=np.asarray(bucket.source_indices, dtype=np.int64),
            messages=prev_values,
            scatter_to=np.asarray(bucket.target_indices, dtype=np.int64),
            out_size=bucket.target_size,
        )

        reduced = NumpySBBackend().scatter_logsumexp(**kwargs)
        dense = DenseReferenceBackend().scatter_logsumexp(**kwargs)

        expected_0 = math.log(0.6 * 0.5 + 0.2 * 0.4 + 0.1 * 0.9)
        expected_1 = math.log(0.7 * 0.9)
        self.assertTrue(np.allclose(reduced, np.asarray((expected_0, expected_1))))
        self.assertTrue(np.allclose(reduced, dense))

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

        with self.assertRaises(SBBackendError):
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


if __name__ == "__main__":
    unittest.main()
