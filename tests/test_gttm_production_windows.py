from __future__ import annotations

import math
import unittest

import numpy as np

from aimusic.core.config import PriorWeights, SBConfig, StyleConfig
from aimusic.core.core_types import BeatState, Edge, Layer
from aimusic.core.vocab import DEFAULT_VOCABULARIES
from aimusic.planning.candidates import is_legal_transition
from aimusic.planning.graph import (
    _rescore_retained_edges_with_windows,
    build_sparse_graph,
)
from aimusic.planning.plans import MethodARunConfig, run_method_a
from aimusic.scoring.gttm_features import (
    FEATURE_REGISTRY,
    FeatureContextRequirement,
    TransitionWindow,
)
from aimusic.scoring.priors import (
    NullPrior,
    PriorQuery,
    calculate_transition_score_breakdown,
    calculate_transition_score_breakdowns,
)


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


class TestRegisteredContextInventory(unittest.TestCase):
    def test_every_registered_feature_declares_context_requirements(self) -> None:
        self.assertTrue(FEATURE_REGISTRY)
        self.assertTrue(
            all(
                isinstance(spec.context_requirement, FeatureContextRequirement)
                for spec in FEATURE_REGISTRY.values()
            )
        )
        self.assertEqual(
            FEATURE_REGISTRY["grouping_boundary_resolution"].context_requirement,
            FeatureContextRequirement.PREVIOUS_CURRENT_RIGHT,
        )
        self.assertEqual(
            {
                name
                for name, spec in FEATURE_REGISTRY.items()
                if spec.context_requirement
                is FeatureContextRequirement.PREVIOUS_CURRENT_RIGHT
            },
            {"grouping_boundary_resolution"},
        )


class TestWindowedBatchScoring(unittest.TestCase):
    def test_scalar_and_batch_breakdowns_are_numerically_identical(self) -> None:
        prev = state(beat=3, chord="G7", role="prep", head="upper_approach")
        current = state(beat=0, boundary="phrase", role="cad")
        right = state(meter="3/4", beat=0)
        window = TransitionWindow(right_state=right)
        weights = PriorWeights(lambda_data=0.75, lambda_gttm=1.25, grouping=1.5)
        prior = NullPrior(neutral_logp=-0.2)

        scalar = calculate_transition_score_breakdown(
            prev,
            current,
            0,
            prior=prior,
            window=window,
            weights=weights,
            vocabularies=VOCABS,
            edo=12,
        )
        batched = calculate_transition_score_breakdowns(
            (PriorQuery(prev, current, 0),),
            prior=prior,
            windows=(window,),
            weights=weights,
            vocabularies=VOCABS,
            edo=12,
        )[0]

        self.assertEqual(scalar.raw_feature_contributions, batched.raw_feature_contributions)
        self.assertEqual(
            scalar.weighted_feature_contributions,
            batched.weighted_feature_contributions,
        )
        self.assertEqual(scalar.final_log_weight, batched.final_log_weight)


class TestTwoPassProductionScoring(unittest.TestCase):
    def _rescore_example(self, right: BeatState):
        prev = state(beat=3, chord="G7", role="prep", head="upper_approach")
        current = state(beat=0, boundary="phrase", role="cad")
        edge_layers = (
            (Edge(0, prev, current, 0.0),),
            (Edge(1, current, right, 0.0),),
        )
        return _rescore_retained_edges_with_windows(
            edge_layers,
            end_layer=Layer(2, (right,)),
            prior=NullPrior(),
            weights=PriorWeights(
                lambda_data=0.0,
                lambda_gttm=1.0,
                meter=0.0,
                grouping=1.0,
                harmonic=0.0,
                prolongational_role=0.0,
                melodic_head=0.0,
                groove=0.0,
            ),
            vocabularies=VOCABS,
            edo=12,
        )

    def test_windowed_feature_prefers_strong_continuation(self) -> None:
        good_right = state(meter="3/4", beat=0)
        weak_right = state(beat=1)
        _, good_diagnostics = self._rescore_example(good_right)
        _, weak_diagnostics = self._rescore_example(weak_right)

        good = good_diagnostics[0][0]
        weak = weak_diagnostics[0][0]
        self.assertEqual(good.right_contexts, (good_right,))
        self.assertEqual(weak.right_contexts, (weak_right,))
        self.assertGreater(
            good.raw_feature_contributions["grouping_boundary_resolution"],
            weak.raw_feature_contributions["grouping_boundary_resolution"],
        )
        self.assertGreater(good.gttm_score, weak.gttm_score)

    def test_sparse_graph_emits_aligned_reconciling_diagnostics(self) -> None:
        style = StyleConfig(
            allowed_meters=("4/4", "3/4", "5/4", "7/4"),
            groove_families=("straight", "syncopated", "swing"),
        )
        sb = SBConfig(horizon_t=3, k_max=6, d_max=4)
        start = state(
            beat=1,
            chord="G7",
            role="prep",
            head="upper_approach",
        )
        end = state(
            beat=0,
            boundary="phrase",
            chord="Cmaj",
            role="cad",
        )
        graph = build_sparse_graph(
            Layer(0, (start,)),
            Layer(3, (end,)),
            3,
            sb_config=sb,
            style_config=style,
            vocabularies=VOCABS,
            prior=NullPrior(),
            weights=PriorWeights(),
            rng=np.random.default_rng(42),
            d_max=sb.d_max,
        )

        self.assertEqual(len(graph.edge_diagnostics_by_time), len(graph.edges_by_time))
        active_families: set[str] = set()
        for edges, diagnostics in zip(
            graph.edges_by_time, graph.edge_diagnostics_by_time
        ):
            self.assertEqual(len(edges), len(diagnostics))
            by_source: dict[BeatState, int] = {}
            for edge, diagnostic in zip(edges, diagnostics):
                self.assertEqual((edge.source, edge.target), (diagnostic.source, diagnostic.target))
                self.assertTrue(math.isclose(edge.log_weight, diagnostic.final_log_weight, abs_tol=1e-12))
                self.assertTrue(
                    math.isclose(
                        sum(diagnostic.weighted_feature_contributions.values()),
                        diagnostic.gttm_score,
                        abs_tol=1e-12,
                    )
                )
                self.assertTrue(math.isclose(diagnostic.gttm_energy, -diagnostic.gttm_score))
                self.assertTrue(
                    math.isclose(
                        diagnostic.data_contribution + diagnostic.gttm_contribution,
                        diagnostic.final_log_weight,
                        abs_tol=1e-12,
                    )
                )
                legal, reason = is_legal_transition(
                    edge.source,
                    edge.target,
                    style_config=style,
                    vocabularies=VOCABS,
                )
                self.assertTrue(legal, msg=reason)
                by_source[edge.source] = by_source.get(edge.source, 0) + 1
                active_families.update(
                    FEATURE_REGISTRY[name].family
                    for name, value in diagnostic.raw_feature_contributions.items()
                    if abs(value) > 1e-12
                )
            self.assertTrue(all(count <= sb.d_max for count in by_source.values()))

        self.assertTrue(all(len(layer) <= sb.k_max for layer in graph.layers))
        self.assertEqual(
            active_families,
            {"meter", "grouping", "harmonic", "prolongational_role", "melodic_head", "groove"},
        )
        self.assertTrue(any(item.right_context_count for item in graph.edge_diagnostics_by_time[0]))
        self.assertTrue(
            any(
                abs(
                    item.raw_feature_contributions[
                        "grouping_boundary_resolution"
                    ]
                )
                > 1e-12
                for layer in graph.edge_diagnostics_by_time[:-1]
                for item in layer
            )
        )
        self.assertIn("beat_position_validity", graph.inactive_feature_names())
        self.assertEqual(
            set(graph.inactive_feature_names()),
            {
                name
                for name in FEATURE_REGISTRY
                if all(
                    abs(item.raw_feature_contributions[name]) <= 1e-12
                    for layer in graph.edge_diagnostics_by_time
                    for item in layer
                )
            },
        )

    def test_selected_path_exposes_retained_edge_contributions(self) -> None:
        for use_sampling in (False, True):
            with self.subTest(use_sampling=use_sampling):
                result = run_method_a(
                    MethodARunConfig(
                        total_beats=4,
                        seed=123,
                        use_sampling=use_sampling,
                    )
                )
                self.assertEqual(
                    len(result.path_edge_diagnostics), len(result.path) - 1
                )
                for index, item in enumerate(result.path_edge_diagnostics):
                    self.assertEqual(item.source, result.path[index])
                    self.assertEqual(item.target, result.path[index + 1])
                    self.assertAlmostEqual(
                        sum(item.weighted_feature_contributions.values()),
                        item.gttm_score,
                    )


if __name__ == "__main__":
    unittest.main()
