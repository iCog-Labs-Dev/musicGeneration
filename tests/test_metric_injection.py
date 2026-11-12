from __future__ import annotations

from dataclasses import dataclass, field
import unittest

from aimusic.core.core_types import BeatState
from aimusic.core.rng import RNGKey
from aimusic.planning import candidates
from aimusic.core.vocab import DEFAULT_VOCABULARIES
from aimusic.scoring.gttm_features import (
    harmonic_chord_proximity_feature,
    harmonic_key_proximity_feature,
    cadential_harmonic_motion_feature,
)
from aimusic.scoring.tension import beat_tension, realized_tension_curve
from aimusic.theory.tonal import (
    ExistingTonalMetric, chord_distance, root_distance,
    basic_space_distance, tonal_distance, nearest_roots_for_metric,
)


@dataclass
class FixedMetric:
    root_value: float = 7.0
    chord_value: float = 3.0
    root_calls: list[tuple[int, int, int]] = field(default_factory=list)
    chord_calls: list[tuple[int, str, int, str, int]] = field(default_factory=list)

    def tonal_distance(self, a_root: int, b_root: int, edo: int = 12) -> float:
        self.root_calls.append((a_root, b_root, edo))
        return self.root_value

    def basic_space_distance(
        self, a_root: int, a_quality: str,
        b_root: int, b_quality: str, edo: int = 12,
    ) -> float:
        self.chord_calls.append((a_root, a_quality, b_root, b_quality, edo))
        return self.chord_value


def state(key_root: int = 0, chord_root: int = 0) -> BeatState:
    vocabs = DEFAULT_VOCABULARIES
    return BeatState(
        meter_id=vocabs.meters.token_for_label("4/4").id,
        beat_in_bar=0,
        boundary_lvl=0,
        key_id=key_root,
        chord_id=next(c.id for c in vocabs.chords if c.root_pc == chord_root and c.quality == "maj"),
        role_id=vocabs.roles.token_for_label("hold").id,
        head_id=vocabs.heads.token_for_label("root").id,
        groove_id=vocabs.grooves.token_for_label("straight_8ths").id,
    )


class TestMetricInjection(unittest.TestCase):
    def test_cadence_uses_the_selected_distance_scale(self) -> None:
        from tests.test_graph import state as graph_state

        before = graph_state(chord="G7", role="prep")
        after = graph_state(chord="Cmaj", role="cad", boundary="phrase")
        self.assertEqual(cadential_harmonic_motion_feature(before, after, 0, tonal_metric=FixedMetric(1.0)), 1.6)
        self.assertEqual(cadential_harmonic_motion_feature(before, after, 0, tonal_metric=FixedMetric(2.0)), -0.8)

    def test_separate_unhashable_metrics_do_not_share_cached_values(self) -> None:
        first, second = FixedMetric(1.0, 2.0), FixedMetric(9.0, 10.0)
        before, after = state(), state(7, 7)
        for metric, expected in ((first, 1 / 3), (second, 1 / 11), (first, 1 / 3)):
            self.assertEqual(
                harmonic_chord_proximity_feature(before, after, 0, tonal_metric=metric),
                expected,
            )
        self.assertEqual(len(first.chord_calls), 2)
        self.assertEqual(len(second.chord_calls), 1)

    def test_default_adapter_matches_original_distances_in_both_tunings(self) -> None:
        metric = ExistingTonalMetric()
        for edo in (12, 19):
            for root in range(edo):
                self.assertEqual(root_distance(0, root, edo, tonal_metric=metric), tonal_distance(0, root, edo))
                self.assertEqual(chord_distance(0, "maj", root, "min", edo, tonal_metric=metric), basic_space_distance(0, "maj", root, "min", edo))
            self.assertEqual(nearest_roots_for_metric(0, edo, tonal_metric=metric), nearest_roots_for_metric(0, edo))

    def test_invalid_custom_distances_fail_at_the_boundary(self) -> None:
        for value in (-1.0, float("inf"), float("nan")):
            with self.subTest(value=value):
                metric = FixedMetric(value, value)
                with self.assertRaises(ValueError):
                    root_distance(0, 7, tonal_metric=metric)
                with self.assertRaises(ValueError):
                    chord_distance(0, "maj", 7, "maj", tonal_metric=metric)

    def test_tension_report_uses_the_selected_metric(self) -> None:
        metric = FixedMetric(7.0, 3.0)
        before, after = state(), state(7, 7)
        expected = 0.20 * 7 / 8 + 0.25 * 3 / 4
        self.assertAlmostEqual(beat_tension(before, after, DEFAULT_VOCABULARIES, 12, tonal_metric=metric), expected)
        self.assertEqual(realized_tension_curve((before, after), DEFAULT_VOCABULARIES, 12, tonal_metric=metric), [(0.0, 0.0), (1.0, expected)])

    def test_supplied_metric_changes_harmonic_feature_values(self) -> None:
        metric = FixedMetric()
        before, after = state(), state(7, 7)
        self.assertEqual(
            harmonic_key_proximity_feature(before, after, 0, tonal_metric=metric),
            1.0 / 8.0,
        )
        self.assertEqual(
            harmonic_chord_proximity_feature(before, after, 0, tonal_metric=metric),
            1.0 / 4.0,
        )
        self.assertEqual(metric.root_calls, [(0, 7, 12)])
        self.assertEqual(metric.chord_calls, [(0, "maj", 7, "maj", 12)])


    def test_bound_candidates_preserve_default_and_use_supplied_metric(self) -> None:
        before = state()
        expected = candidates.get_valid_next_states(before, 1, RNGKey(11), 4, proposal_budget=32)
        self.assertEqual(candidates.MetricBoundCandidates()(before, 1, RNGKey(11), 4, proposal_budget=32), expected)
        metric = FixedMetric()
        actual = candidates.MetricBoundCandidates(metric)(before, 1, RNGKey(11), 4, proposal_budget=32)
        self.assertEqual(actual, candidates.get_valid_next_states(before, 1, RNGKey(11), 4, proposal_budget=32, tonal_metric=metric))
        self.assertTrue(metric.root_calls)
        # All equal distances use ascending roots as the deterministic tie-break.
        self.assertEqual(candidates.propose_key_ids(before, 2, before.role_id, DEFAULT_VOCABULARIES, tonal_metric=metric), (0, 1, 2))
