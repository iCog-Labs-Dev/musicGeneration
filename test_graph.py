import unittest
import numpy as np

from candidates import is_legal_transition
from config import PriorWeights, SBConfig, StyleConfig
from core_types import BeatState, Layer
from graph import build_sparse_graph
from priors import NeuralPrior
from vocab import DEFAULT_VOCABULARIES


VOCABS = DEFAULT_VOCABULARIES


def state(
    *,
    meter: str = "4/4",
    beat: int = 0,
    boundary: str = "none",
    key: str = "C",
    chord: str = "Cmaj",
    role: str = "hold",
    head: str = "root",
    groove: str = "straight_8ths",
) -> BeatState:
    return BeatState(
        meter_id=VOCABS.meters.token_for_label(meter).id,
        beat_in_bar=beat,
        boundary_lvl=VOCABS.boundaries.token_for_label(boundary).id,
        key_id=VOCABS.keys.token_for_label(key).id,
        chord_id=VOCABS.chords.token_for_label(chord).id,
        role_id=VOCABS.roles.token_for_label(role).id,
        head_id=VOCABS.heads.token_for_label(head).id,
        groove_id=VOCABS.grooves.token_for_label(groove).id,
    )


class TestSparseGraphBuilder(unittest.TestCase):
    def setUp(self):
        self.style = StyleConfig(
            allowed_meters=("4/4", "5/4", "7/4"),
            groove_families=("straight", "syncopated", "swing"),
        )
        self.sb_config = SBConfig(horizon_t=3, k_max=3, d_max=2)
        self.weights = PriorWeights(lambda_data=0.75, lambda_gttm=1.0)
        self.prior = NeuralPrior()
        self.start_state = state(
            beat=1,
            key="C",
            chord="G7",
            role="prep",
            head="upper_approach",
            groove="straight_8ths",
        )
        self.end_state = state(
            beat=0,
            boundary="phrase",
            key="C",
            chord="Cmaj",
            role="cad",
            head="root",
            groove="straight_8ths",
        )
        self.start_layer = Layer(time_index=0, states=(self.start_state,))
        self.end_layer = Layer(time_index=3, states=(self.end_state,))

    def test_build_sparse_graph_bounds_layers_and_outdegrees(self):
        graph = build_sparse_graph(
            self.start_layer,
            self.end_layer,
            3,
            sb_config=self.sb_config,
            style_config=self.style,
            vocabularies=VOCABS,
            prior=self.prior,
            weights=self.weights,
            rng=np.random.default_rng(42), 
            d_max=self.sb_config.d_max
        )

        self.assertEqual(len(graph.layers), 4)
        self.assertEqual(graph.layers[0], self.start_layer)
        self.assertLessEqual(max(len(layer) for layer in graph.layers), self.sb_config.k_max)
        self.assertEqual(graph.layers[-1].time_index, self.end_layer.time_index)
        self.assertIn(self.end_state, graph.layers[-1].states)

        for edge_group in graph.edges_by_time:
            by_source = {}
            for edge in edge_group:
                by_source.setdefault(edge.source, 0)
                by_source[edge.source] += 1
                legal, reason = is_legal_transition(
                    edge.source,
                    edge.target,
                    style_config=self.style,
                    vocabularies=VOCABS,
                )
                self.assertTrue(legal, msg=reason)
                self.assertIsInstance(edge.log_weight, float)
            self.assertTrue(all(count <= self.sb_config.d_max for count in by_source.values()))

    def test_graph_diagnostics_report_rejections_and_pruning(self):
        graph = build_sparse_graph(
            self.start_layer,
            self.end_layer,
            3,
            sb_config=self.sb_config,
            style_config=self.style,
            vocabularies=VOCABS,
            prior=self.prior,
            weights=self.weights,
            rng=np.random.default_rng(42), 
            d_max=self.sb_config.d_max
        )

        self.assertGreater(graph.diagnostics.total_rejections, 0)
        self.assertEqual(graph.diagnostics.layer_sizes[0], 1)
        self.assertTrue(
            any(item.outdegree_pruned_count > 0 for item in graph.diagnostics.layer_diagnostics)
            or any(item.pruned_candidate_count > 0 for item in graph.diagnostics.layer_diagnostics)
        )
        self.assertTrue(
            any(
                pruned.reason in {"k_max_prune", "unreachable_endpoint"}
                for item in graph.diagnostics.layer_diagnostics
                for pruned in item.pruned_states
            )
            or any(item.outdegree_pruned_count > 0 for item in graph.diagnostics.layer_diagnostics)
        )


if __name__ == "__main__":
    unittest.main()
