import unittest

from config import SBConfig
from core_types import BeatState, Edge, EndpointDistribution, Layer
from graph import GraphDiagnostics, LayerBuildDiagnostics, SparseGraph
from sb import SBContractError, build_sb_problem


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


def _minimal_diagnostics() -> GraphDiagnostics:
    return GraphDiagnostics(
        layer_sizes=(1, 1),
        layer_diagnostics=(
            LayerBuildDiagnostics(
                time_index=0,
                source_state_count=1,
                raw_candidate_count=1,
                unique_candidate_count=1,
                kept_candidate_count=1,
                raw_edge_count=1,
                kept_edge_count=1,
            ),
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
        diagnostics=_minimal_diagnostics(),
    )
    pi0 = EndpointDistribution(layer=l0, probabilities=(1.0,))
    piT = EndpointDistribution(layer=l2, probabilities=(1.0,))
    return graph, pi0, piT


class TestSBProblemContract(unittest.TestCase):
    def test_build_sb_problem_happy_path(self):
        graph, pi0, piT = _valid_graph()

        problem = build_sb_problem(graph=graph, pi0=pi0, piT=piT, sb_config=SBConfig(horizon_t=2))

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

    def test_build_is_pure_and_deterministic(self):
        graph, pi0, piT = _valid_graph()

        first = build_sb_problem(graph, pi0, piT)
        second = build_sb_problem(graph, pi0, piT)

        self.assertEqual(first, second)
        self.assertEqual(graph.layers[0].states[0].beat_in_bar, 0)


if __name__ == "__main__":
    unittest.main()
