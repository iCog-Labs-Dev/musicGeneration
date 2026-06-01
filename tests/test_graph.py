import unittest
import numpy as np

from aimusic.planning.candidates import is_legal_transition
from aimusic.core.config import PriorWeights, SBConfig, StyleConfig
from aimusic.core.core_types import BeatState, Layer
from aimusic.planning.graph import build_sparse_graph
from aimusic.scoring.priors import NeuralPrior
from aimusic.core.vocab import DEFAULT_VOCABULARIES
from aimusic.planning.graph import StitchAnchor, NeighborCache, GraphDiagnostics, _cached_endpoint_distance



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


class TestStitchAnchor(unittest.TestCase):
    def _make_state(self, key_id=0, chord_id=0):
        return state(key="C" if key_id == 0 else "G")

    def test_exact_match_returns_full_bonus(self):
        s = state(key="C", chord="Cmaj")
        anchor = StitchAnchor(prev_terminal=s, log_bonus=2.0)
        self.assertAlmostEqual(anchor.bonus_for(s), 2.0)

    def test_key_match_returns_half_bonus(self):
        terminal = state(key="C", chord="Cmaj")
        other = state(key="C", chord="G7")  # same key, different chord
        anchor = StitchAnchor(prev_terminal=terminal, log_bonus=2.0, match_key=True)
        self.assertAlmostEqual(anchor.bonus_for(other), 1.0)

    def test_no_match_returns_zero(self):
        terminal = state(key="C")
        other = state(key="G")
        anchor = StitchAnchor(prev_terminal=terminal, log_bonus=2.0)
        self.assertAlmostEqual(anchor.bonus_for(other), 0.0)

    def test_match_key_false_ignores_key_match(self):
        terminal = state(key="C", chord="Cmaj")
        other = state(key="C", chord="G7")
        anchor = StitchAnchor(prev_terminal=terminal, log_bonus=2.0, match_key=False)
        self.assertAlmostEqual(anchor.bonus_for(other), 0.0)

    def test_zero_log_bonus_is_noop(self):
        s = state(key="C")
        anchor = StitchAnchor(prev_terminal=s, log_bonus=0.0)
        self.assertAlmostEqual(anchor.bonus_for(s), 0.0)

    def test_rejects_infinite_log_bonus(self):
        s = state()
        with self.assertRaises((ValueError, Exception)):
            StitchAnchor(prev_terminal=s, log_bonus=float("inf"))


class TestNeighborCache(unittest.TestCase):
    def test_initial_state_is_zero(self):
        cache = NeighborCache()
        self.assertEqual(cache.hits, 0)
        self.assertEqual(cache.misses, 0)
        self.assertEqual(cache.size, 0)
        self.assertAlmostEqual(cache.hit_rate, 0.0)

    def test_hit_rate_zero_with_no_calls(self):
        cache = NeighborCache()
        self.assertAlmostEqual(cache.hit_rate, 0.0)

    def test_clear_resets_all_stats(self):
        cache = NeighborCache()
        s = state()
        cache._store[s] = frozenset()
        cache.hits = 7
        cache.misses = 3
        cache.clear()
        self.assertEqual(cache.hits, 0)
        self.assertEqual(cache.misses, 0)
        self.assertEqual(cache.size, 0)

    def test_hit_rate_calculation(self):
        cache = NeighborCache()
        cache.hits = 6
        cache.misses = 4
        self.assertAlmostEqual(cache.hit_rate, 0.6)

    def test_size_reflects_stored_entries(self):
        cache = NeighborCache()
        s1 = state(chord="Cmaj")
        s2 = state(chord="G7")
        cache._store[s1] = frozenset({s2})
        self.assertEqual(cache.size, 1)
        cache._store[s2] = frozenset({s1})
        self.assertEqual(cache.size, 2)


class TestGraphDiagnosticsCache(unittest.TestCase):
    def test_default_cache_stats_are_zero(self):
        diag = GraphDiagnostics(layer_sizes=(5, 6, 7), layer_diagnostics=())
        self.assertEqual(diag.neighbor_cache_hits, 0)
        self.assertEqual(diag.neighbor_cache_misses, 0)

    def test_hit_rate_zero_when_no_calls(self):
        diag = GraphDiagnostics(layer_sizes=(5,), layer_diagnostics=())
        self.assertAlmostEqual(diag.neighbor_cache_hit_rate, 0.0)

    def test_hit_rate_calculation(self):
        diag = GraphDiagnostics(
            layer_sizes=(5,), layer_diagnostics=(),
            neighbor_cache_hits=30, neighbor_cache_misses=10,
        )
        self.assertAlmostEqual(diag.neighbor_cache_hit_rate, 0.75)

    def test_hit_rate_one_when_all_hits(self):
        diag = GraphDiagnostics(
            layer_sizes=(5,), layer_diagnostics=(),
            neighbor_cache_hits=100, neighbor_cache_misses=0,
        )
        self.assertAlmostEqual(diag.neighbor_cache_hit_rate, 1.0)

    def test_hit_rate_zero_when_all_misses(self):
        diag = GraphDiagnostics(
            layer_sizes=(5,), layer_diagnostics=(),
            neighbor_cache_hits=0, neighbor_cache_misses=50,
        )
        self.assertAlmostEqual(diag.neighbor_cache_hit_rate, 0.0)


class TestCachedEndpointDistance(unittest.TestCase):
    def test_is_lru_cached(self):
        self.assertTrue(
            hasattr(_cached_endpoint_distance, "cache_info"),
            "_cached_endpoint_distance should be decorated with @lru_cache",
        )

    def test_same_inputs_return_same_result(self):
        args = (0, "maj", 0, 0, 0, 0, 0, 0, 7, "min", 7, 0, 0, 0, 0, 0, 12)
        self.assertAlmostEqual(
            _cached_endpoint_distance(*args),
            _cached_endpoint_distance(*args),
        )

    def test_same_chord_lower_distance_than_distant(self):
        same = _cached_endpoint_distance(
            0, "maj", 0, 0, 0, 0, 0, 0,
            0, "maj", 0, 0, 0, 0, 0, 0,
            12,
        )
        distant = _cached_endpoint_distance(
            0, "maj", 0, 0, 0, 0, 0, 0,
            6, "dim", 6, 2, 3, 2, 3, 1,
            12,
        )
        self.assertLess(same, distant)

    def test_cache_hit_after_first_call(self):
        _cached_endpoint_distance.cache_clear()
        args = (0, "maj", 0, 0, 0, 0, 0, 0, 1, "min", 1, 0, 0, 0, 0, 0, 12)
        _cached_endpoint_distance(*args)
        before = _cached_endpoint_distance.cache_info()
        _cached_endpoint_distance(*args)
        after = _cached_endpoint_distance.cache_info()
        self.assertGreater(after.hits, before.hits)


class TestBuildSparseGraphStitchAnchor(unittest.TestCase):
    def setUp(self):
        self.style = StyleConfig(
            allowed_meters=("4/4",),
            groove_families=("straight",),
        )
        self.sb_config = SBConfig(horizon_t=3, k_max=3, d_max=2)
        self.weights = PriorWeights(lambda_data=0.75, lambda_gttm=1.0)
        self.prior = NeuralPrior()
        self.start_state = state(beat=1, key="C", chord="G7", role="prep")
        self.end_state = state(beat=0, boundary="phrase", key="C", chord="Cmaj", role="cad")
        self.start_layer = Layer(time_index=0, states=(self.start_state,))
        self.end_layer = Layer(time_index=3, states=(self.end_state,))

    def test_stitch_anchor_none_accepted(self):
        # Should not raise
        graph = build_sparse_graph(
            self.start_layer,
            self.end_layer,
            3,
            sb_config=self.sb_config,
            style_config=self.style,
            vocabularies=VOCABS,
            prior=self.prior,
            weights=self.weights,
            rng=np.random.default_rng(0),
            d_max=self.sb_config.d_max,
            stitch_anchor=None,
        )
        self.assertEqual(len(graph.layers), 4)

    def test_stitch_anchor_accepted_in_build(self):
        anchor = StitchAnchor(prev_terminal=self.start_state, log_bonus=1.5)
        graph = build_sparse_graph(
            self.start_layer,
            self.end_layer,
            3,
            sb_config=self.sb_config,
            style_config=self.style,
            vocabularies=VOCABS,
            prior=self.prior,
            weights=self.weights,
            rng=np.random.default_rng(0),
            d_max=self.sb_config.d_max,
            stitch_anchor=anchor,
        )
        self.assertEqual(len(graph.layers), 4)

    def test_stitch_anchor_bad_type_raises_type_error(self):
        import inspect
        sig = inspect.signature(build_sparse_graph)
        self.assertIn("stitch_anchor", sig.parameters)
        self.assertIsNone(sig.parameters["stitch_anchor"].default)
        with self.assertRaises(TypeError):
            build_sparse_graph(
                self.start_layer,
                self.end_layer,
                3,
                rng=np.random.default_rng(0),
                d_max=2,
                stitch_anchor="not_an_anchor",
            )

if __name__ == "__main__":
    unittest.main()
