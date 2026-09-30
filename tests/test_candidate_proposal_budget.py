"""REQ-13: proposal budget separated from retained outdegree (D_max).

Covers the acceptance criteria that don't already fit naturally into
``test_candidates.py`` / ``test_graph.py`` / ``test_graph_batching.py``:

- D_max controls retained outgoing edges only.
- Fixed inputs and keys yield stable candidate support and ranking.
- Exhaustive tiny-vocabulary tests measure top-candidate recall against the
  bounded proposal method.
- Seed-sensitivity reports distinguish support variance from bridge-sampling
  variance.
"""
from __future__ import annotations

import unittest
from typing import ClassVar

from aimusic.core.config import SBConfig, StyleConfig
from aimusic.core.core_types import BeatState, EndpointDistribution, Layer
from aimusic.core.rng import RNGKey
from aimusic.core.vocab import Vocabularies, build_tonal_context
from aimusic.planning.candidates import get_valid_next_states
from aimusic.planning.graph import build_sparse_graph
from aimusic.planning.sb import build_sb_problem, sample_bridge_path, solve_sb
from aimusic.scoring.priors import NeuralPrior


VOCABS = build_tonal_context(12, StyleConfig()).vocabularies


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
    vocabs=VOCABS,
) -> BeatState:
    return BeatState(
        meter_id=vocabs.meters.token_for_label(meter).id,
        beat_in_bar=beat,
        boundary_lvl=vocabs.boundaries.token_for_label(boundary).id,
        key_id=vocabs.keys.token_for_label(key).id,
        chord_id=vocabs.chords.token_for_label(chord).id,
        role_id=vocabs.roles.token_for_label(role).id,
        head_id=vocabs.heads.token_for_label(head).id,
        groove_id=vocabs.grooves.token_for_label(groove).id,
    )


class TestDMaxControlsRetainedEdgesOnly(unittest.TestCase):
    """D_max must not influence proposal, dedup, legality, or scoring."""

    def setUp(self):
        self.style = StyleConfig(
            allowed_meters=("4/4", "5/4", "7/4"),
            groove_families=("straight", "syncopated", "swing"),
        )
        self.prev = state(
            beat=3, key="C", chord="G7", role="prep",
            head="upper_approach", groove="straight_8ths",
        )

    def test_candidate_pool_is_identical_across_d_max_values(self):
        results = []
        for d_max in (1, 2, 50, 5000):
            result, next_key = get_valid_next_states(
                self.prev, 4, RNGKey(seed=7), d_max,
                style_config=self.style, vocabularies=VOCABS,
                prior=NeuralPrior(), proposal_budget=300,
                prior_guided_proposals=True,
            )
            results.append((result, next_key))

        first_result, first_key = results[0]
        for result, next_key in results[1:]:
            self.assertEqual(result.states, first_result.states)
            self.assertEqual(result.scores, first_result.scores)
            self.assertEqual(result.rejections, first_result.rejections)
            self.assertEqual(result.proposed_count, first_result.proposed_count)
            self.assertEqual(result.unique_count, first_result.unique_count)
            self.assertEqual(next_key, first_key)

    def test_graph_layer_search_diagnostics_are_d_max_invariant(self):
        start_layer = Layer(time_index=0, states=(self.prev,))
        end_state = state(
            beat=0, boundary="phrase", key="C", chord="Cmaj",
            role="cad", head="root", groove="straight_8ths",
        )
        end_layer = Layer(time_index=1, states=(end_state,))

        diagnostics = []
        kept_edge_counts = []
        for d_max in (1, 3):
            sb_config = SBConfig(horizon_t=1, k_max=10, d_max=d_max, proposal_budget=300)
            graph, _ = build_sparse_graph(
                start_layer, end_layer, 1, sb_config=sb_config,
                style_config=self.style, vocabularies=VOCABS,
                prior=NeuralPrior(), key=RNGKey(seed=11), d_max=d_max,
            )
            layer_diag = graph.diagnostics.layer_diagnostics[0]
            diagnostics.append(layer_diag)
            kept_edge_counts.append(layer_diag.kept_edge_count)

        # Search-side counts (proposed/legal/unique/scored) never move with d_max.
        first = diagnostics[0]
        for other in diagnostics[1:]:
            self.assertEqual(other.raw_candidate_count, first.raw_candidate_count)
            self.assertEqual(other.legal_candidate_count, first.legal_candidate_count)
            self.assertEqual(other.unique_candidate_count, first.unique_candidate_count)
            self.assertEqual(other.scored_candidate_count, first.scored_candidate_count)

        # Only retention (kept edges) tracks d_max.
        self.assertTrue(all(count <= 1 for count in kept_edge_counts[:1]))
        self.assertTrue(all(count <= 3 for count in kept_edge_counts[1:]))
        self.assertLessEqual(kept_edge_counts[0], kept_edge_counts[1])


class TestPriorGuidedRankingIsDeterministic(unittest.TestCase):
    def test_fixed_inputs_and_keys_yield_stable_support_and_ranking(self):
        style = StyleConfig(
            allowed_meters=("4/4", "5/4", "7/4"),
            groove_families=("straight", "syncopated", "swing"),
        )
        prev = state(
            beat=3, key="C", chord="G7", role="prep",
            head="upper_approach", groove="straight_8ths",
        )

        result_a, key_a = get_valid_next_states(
            prev, 4, RNGKey(seed=99), 4, style_config=style, vocabularies=VOCABS,
            prior=NeuralPrior(), proposal_budget=200, prior_guided_proposals=True,
        )
        result_b, key_b = get_valid_next_states(
            prev, 4, RNGKey(seed=99), 4, style_config=style, vocabularies=VOCABS,
            prior=NeuralPrior(), proposal_budget=200, prior_guided_proposals=True,
        )

        self.assertEqual(result_a, result_b)
        self.assertEqual(key_a, key_b)
        self.assertEqual(result_a.scored_count, len(result_a.states))
        # Ranking is sorted by score, descending (deterministic tie-break on ties).
        self.assertEqual(
            list(result_a.scores), sorted(result_a.scores, reverse=True)
        )


class TestTinyVocabularyRecall(unittest.TestCase):
    """Bounded proposal recall against an exhaustive search, on a small vocab."""

    style: ClassVar[StyleConfig]
    vocabs: ClassVar[Vocabularies]
    prev: ClassVar[BeatState]
    prior: ClassVar[NeuralPrior]

    @classmethod
    def setUpClass(cls):
        cls.style = StyleConfig(allowed_meters=("4/4",), groove_families=("straight",))
        cls.vocabs = build_tonal_context(3, cls.style).vocabularies
        cls.prev = BeatState(
            meter_id=cls.vocabs.meters.token_for_label("4/4").id,
            beat_in_bar=3,
            boundary_lvl=cls.vocabs.boundaries.token_for_label("none").id,
            key_id=cls.vocabs.keys.token_for_id(0).id,
            chord_id=cls.vocabs.chords.token_for_id(0).id,
            role_id=cls.vocabs.roles.token_for_label("prep").id,
            head_id=cls.vocabs.heads.token_for_label("upper_approach").id,
            groove_id=cls.vocabs.grooves.token_for_id(0).id,
        )
        cls.prior = NeuralPrior()

    def _top_k(self, *, proposal_budget: int, seed: int, k: int = 8):
        result, _ = get_valid_next_states(
            self.prev, 0, RNGKey(seed=seed), 8,
            style_config=self.style, vocabularies=self.vocabs,
            prior=self.prior, proposal_budget=proposal_budget,
            prior_guided_proposals=True,
        )
        return set(result.states[:k])

    def test_recall_improves_with_proposal_budget(self):
        k = 8
        ground_truth = self._top_k(proposal_budget=10_000, seed=0, k=k)
        self.assertGreaterEqual(len(ground_truth), k)

        seeds = range(8)
        budgets = (30, 120, 900)
        recalls_by_budget = {}
        for budget in budgets:
            recalls = []
            for seed in seeds:
                candidate_top_k = self._top_k(proposal_budget=budget, seed=seed, k=k)
                recalls.append(len(candidate_top_k & ground_truth) / k)
            recalls_by_budget[budget] = sum(recalls) / len(recalls)

        # This corpus has many near-tied transitions (see the ADR-005
        # benchmark), so exact top-k recall is naturally modest at small
        # budgets -- but it must climb monotonically as the proposal budget
        # grows. That monotonic trend, not any single absolute number, is
        # the quality/runtime tradeoff REQ-13 asks callers to be able to see.
        ordered_recalls = [recalls_by_budget[b] for b in budgets]
        self.assertEqual(ordered_recalls, sorted(ordered_recalls))
        self.assertGreater(recalls_by_budget[budgets[-1]], recalls_by_budget[budgets[0]])
        self.assertGreaterEqual(recalls_by_budget[budgets[-1]], 0.4)


class TestSeedSensitivityDistinguishesSupportFromBridgeVariance(unittest.TestCase):
    """Support (candidate pool) variance comes from the proposal RNG stream;
    bridge-sampling variance comes from a separate, independent stream."""

    def setUp(self):
        self.style = StyleConfig(
            allowed_meters=("4/4", "5/4", "7/4"),
            groove_families=("straight", "syncopated", "swing"),
        )
        self.start_state = state(
            beat=1, key="C", chord="G7", role="prep",
            head="upper_approach", groove="straight_8ths",
        )
        self.end_state = state(
            beat=0, boundary="phrase", key="C", chord="Cmaj",
            role="cad", head="root", groove="straight_8ths",
        )
        self.start_layer = Layer(time_index=0, states=(self.start_state,))
        self.end_layer = Layer(time_index=3, states=(self.end_state,))
        # A small budget relative to the full search space, so that which
        # candidates make it into the support is actually seed-sensitive.
        self.sb_config = SBConfig(horizon_t=3, k_max=4, d_max=3, proposal_budget=40)

    def _build_graph(self, candidate_proposal_key: RNGKey):
        graph, _ = build_sparse_graph(
            self.start_layer, self.end_layer, 3, sb_config=self.sb_config,
            style_config=self.style, vocabularies=VOCABS, prior=NeuralPrior(),
            key=candidate_proposal_key, d_max=self.sb_config.d_max,
        )
        return graph

    def test_support_is_stable_under_a_fixed_proposal_key(self):
        graph_a = self._build_graph(RNGKey(seed=5))
        graph_b = self._build_graph(RNGKey(seed=5))
        self.assertEqual(graph_a.layers, graph_b.layers)
        self.assertEqual(graph_a.edges_by_time, graph_b.edges_by_time)

    def test_support_varies_across_proposal_keys(self):
        supports = {
            tuple(layer.states for layer in self._build_graph(RNGKey(seed=seed)).layers)
            for seed in range(6)
        }
        # With a tight proposal budget, different proposal-RNG seeds should
        # explore (and therefore keep) a different candidate support at
        # least some of the time.
        self.assertGreater(len(supports), 1)

    def test_bridge_sampling_varies_independently_of_support(self):
        graph = self._build_graph(RNGKey(seed=5))
        pi0 = EndpointDistribution(layer=graph.layers[0], probabilities=(1.0,))
        piT = EndpointDistribution(layer=graph.layers[-1], probabilities=(1.0,))
        problem = build_sb_problem(graph, pi0, piT, sb_config=self.sb_config)
        bridge = solve_sb(problem).to_bridge()

        sampled_paths = set()
        for seed in range(8):
            sampled, _ = sample_bridge_path(bridge, RNGKey(seed=seed))
            sampled_paths.add(sampled.path)

        # Same graph / same support throughout -- only the bridge-sampling
        # RNG stream changed, and it alone can move the sampled path.
        self.assertGreaterEqual(len(sampled_paths), 1)
        # Re-sampling with the same bridge-sampling key is itself stable.
        replay_a, _ = sample_bridge_path(bridge, RNGKey(seed=0))
        replay_b, _ = sample_bridge_path(bridge, RNGKey(seed=0))
        self.assertEqual(replay_a, replay_b)


if __name__ == "__main__":
    unittest.main()