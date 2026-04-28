import math
import unittest

import numpy as np

from config import SBBackend, SBConfig
from core_types import BeatState, Edge, EndpointDistribution, Layer
from graph import GraphDiagnostics, LayerBuildDiagnostics, SparseGraph
from sb import (
    SBContractError,
    _IndexedEdgeBucket,
    _NumpySBBackend,
    build_sb_problem,
    solve_sb,
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


class TestSBSolver(unittest.TestCase):
    def test_solve_sb_converges_on_tiny_graph(self):
        graph, pi0, piT = _valid_graph()
        problem = build_sb_problem(graph, pi0, piT)

        solution = solve_sb(problem)

        self.assertTrue(solution.trace.converged)
        self.assertEqual(solution.trace.iterations, 1)
        self.assertAlmostEqual(solution.trace.final_max_delta, 0.0)
        self.assertTrue(np.allclose(np.asarray(solution.log_forward_potentials), -np.asarray(solution.log_backward_potentials)))
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


if __name__ == "__main__":
    unittest.main()
