import unittest
from dataclasses import FrozenInstanceError

from config import PriorWeights
from core_types import BeatState
from gttm_features import (
    BEATSTATE_FEATURES,
    FEATURE_REGISTRY,
    TransitionWindow,
    _cached_basic_space_distance,
    boundary_placement_feature,
    calculate_gttm_energy,
    calculate_gttm_score,
    cadential_harmonic_motion_feature,
    groove_boundary_change_feature,
    groove_continuity_feature,
    grouping_boundary_resolution_feature,
    grouping_downbeat_alignment_feature,
    grouping_onset_feature,
    harmonic_chord_proximity_feature,
    harmonic_distance_cache_info,
    harmonic_key_neighbor_feature,
    harmonic_key_proximity_feature,
    head_anchor_feature,
    head_resolution_feature,
    meter_stability_feature,
    role_meter_alignment_feature,
    role_transition_feature,
    strong_beat_bias_feature,
    tonal_neighbor_cache_info,
    transition_energy,
    transition_family_scores,
    transition_feature_vector,
    weighted_feature_breakdown,
)
from tonal import nearest_roots, tonal_distance
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


class TestTransitionWindow(unittest.TestCase):
    def test_transition_window_is_frozen(self):
        window = TransitionWindow()
        with self.assertRaises(FrozenInstanceError):
            window.right_state = state()


class TestBeatStateFeatureFamilies(unittest.TestCase):
    def test_meter_features_reward_valid_stable_strong_arrivals(self):
        prev = state(beat=3, groove="straight_8ths")
        next_good = state(beat=0, boundary="phrase", groove="straight_8ths")
        next_bad = state(meter="3/4", beat=4, boundary="phrase", groove="straight_8ths")

        self.assertGreater(meter_stability_feature(prev, next_good, 0), 0.0)
        self.assertLess(meter_stability_feature(prev, next_bad, 0), 0.0)
        self.assertGreater(boundary_placement_feature(prev, next_good, 0), 0.0)
        self.assertLess(boundary_placement_feature(prev, next_bad, 0), 0.0)
        self.assertGreater(strong_beat_bias_feature(prev, next_good, 0), 0.0)

    def test_grouping_features_reward_boundary_onsets_and_downbeats(self):
        prev = state(beat=3, boundary="none")
        next_boundary = state(beat=0, boundary="phrase")
        right_downbeat = state(beat=0, boundary="none")
        weak_boundary = state(beat=1, boundary="phrase")

        self.assertGreater(grouping_onset_feature(prev, next_boundary, 0), 0.0)
        self.assertEqual(
            grouping_boundary_resolution_feature(prev, next_boundary, 0),
            0.0,
        )
        self.assertGreater(
            grouping_boundary_resolution_feature(
                prev,
                next_boundary,
                0,
                window=TransitionWindow(right_state=right_downbeat),
            ),
            0.0,
        )
        self.assertGreater(
            grouping_downbeat_alignment_feature(prev, next_boundary, 0),
            grouping_downbeat_alignment_feature(prev, weak_boundary, 0),
        )

    def test_harmonic_features_reward_local_motion_and_cadence(self):
        prev = state(beat=3, key="C", chord="G7", role="prep")
        next_cadence = state(beat=0, key="C", chord="Cmaj", role="cad")
        next_detour = state(beat=1, key="F#", chord="F#dim", role="cad")

        self.assertGreater(
            harmonic_key_proximity_feature(prev, next_cadence, 0),
            harmonic_key_proximity_feature(prev, next_detour, 0),
        )
        self.assertGreater(
            harmonic_key_neighbor_feature(prev, next_cadence, 0),
            harmonic_key_neighbor_feature(prev, next_detour, 0),
        )
        self.assertGreater(
            harmonic_chord_proximity_feature(prev, next_cadence, 0),
            harmonic_chord_proximity_feature(prev, next_detour, 0),
        )
        self.assertGreater(
            cadential_harmonic_motion_feature(prev, next_cadence, 0),
            cadential_harmonic_motion_feature(prev, next_detour, 0),
        )

    def test_prolongational_role_features_reward_cadences_on_strong_beats(self):
        prev = state(beat=3, role="prep")
        cadential_downbeat = state(beat=0, boundary="phrase", role="cad")
        cadential_weak = state(beat=1, boundary="phrase", role="cad")

        self.assertGreater(
            role_meter_alignment_feature(prev, cadential_downbeat, 0),
            role_meter_alignment_feature(prev, cadential_weak, 0),
        )
        self.assertGreater(
            role_transition_feature(prev, cadential_downbeat, 0),
            role_transition_feature(state(role="cad"), state(role="prep"), 0),
        )

    def test_melodic_head_features_reward_anchor_and_resolution(self):
        prev = state(beat=3, head="upper_approach")
        anchored = state(beat=0, head="root", chord="Cmaj")
        unresolved = state(beat=1, head="lower_approach", chord="Cmaj")

        self.assertGreater(head_anchor_feature(prev, anchored, 0), 0.0)
        self.assertLess(head_anchor_feature(prev, unresolved, 0), head_anchor_feature(prev, anchored, 0))
        self.assertGreater(
            head_resolution_feature(prev, anchored, 0),
            head_resolution_feature(prev, unresolved, 0),
        )

    def test_groove_features_reward_continuity_and_boundary_scoped_changes(self):
        prev = state(groove="straight_8ths")
        same = state(groove="straight_8ths")
        family_change = state(groove="straight_16ths")
        hard_change = state(groove="syncopated_16ths")
        boundary_change = state(groove="syncopated_16ths", boundary="local")

        self.assertGreater(groove_continuity_feature(prev, same, 0), groove_continuity_feature(prev, family_change, 0))
        self.assertGreater(groove_continuity_feature(prev, family_change, 0), groove_continuity_feature(prev, hard_change, 0))
        self.assertGreater(
            groove_boundary_change_feature(prev, boundary_change, 0),
            groove_boundary_change_feature(prev, hard_change, 0),
        )


class TestBeatStateAggregation(unittest.TestCase):
    def setUp(self):
        self.prev = state(
            beat=3,
            key="C",
            chord="G7",
            role="prep",
            head="upper_approach",
            groove="straight_8ths",
        )
        self.next_good = state(
            beat=0,
            boundary="phrase",
            key="C",
            chord="Cmaj",
            role="cad",
            head="root",
            groove="straight_8ths",
        )
        self.next_bad = state(
            meter="3/4",
            beat=1,
            boundary="none",
            key="F#",
            chord="F#dim",
            role="change",
            head="upper_approach",
            groove="syncopated_16ths",
        )
        self.good_window = TransitionWindow(right_state=state(beat=1, role="hold"))
        self.bad_window = TransitionWindow(right_state=state(meter="3/4", beat=2, groove="syncopated_16ths"))

    def test_feature_registry_exposes_named_features_across_six_families(self):
        self.assertEqual(set(BEATSTATE_FEATURES.keys()), set(FEATURE_REGISTRY.keys()))
        self.assertEqual(
            {spec.family for spec in FEATURE_REGISTRY.values()},
            {
                "meter",
                "grouping",
                "harmonic",
                "prolongational_role",
                "melodic_head",
                "groove",
            },
        )

    def test_transition_feature_vector_and_family_scores_cover_same_signal(self):
        features = transition_feature_vector(self.prev, self.next_good, 8, window=self.good_window)
        families = transition_family_scores(self.prev, self.next_good, 8, window=self.good_window)

        self.assertEqual(set(features.keys()), set(FEATURE_REGISTRY.keys()))
        self.assertEqual(set(families.keys()), {spec.family for spec in FEATURE_REGISTRY.values()})

        for family_name, family_total in families.items():
            expected = sum(
                value
                for name, value in features.items()
                if FEATURE_REGISTRY[name].family == family_name
            )
            self.assertAlmostEqual(family_total, expected)

    def test_weighted_breakdown_applies_family_weights_without_code_changes(self):
        default = weighted_feature_breakdown(self.prev, self.next_good, 8, window=self.good_window)
        weighted = weighted_feature_breakdown(
            self.prev,
            self.next_good,
            8,
            window=self.good_window,
            weights=PriorWeights(harmonic=0.0, groove=2.0),
        )

        harmonic_features = [
            name for name, spec in FEATURE_REGISTRY.items() if spec.family == "harmonic"
        ]
        groove_features = [
            name for name, spec in FEATURE_REGISTRY.items() if spec.family == "groove"
        ]

        for name in harmonic_features:
            self.assertEqual(weighted[name], 0.0)
        for name in groove_features:
            self.assertAlmostEqual(weighted[name], default[name] * 2.0)

    def test_score_and_energy_are_consistent(self):
        breakdown = weighted_feature_breakdown(self.prev, self.next_good, 8, window=self.good_window)
        score = calculate_gttm_score(self.prev, self.next_good, 8, window=self.good_window)

        self.assertAlmostEqual(score, sum(breakdown.values()))
        self.assertAlmostEqual(transition_energy(self.prev, self.next_good, 8, window=self.good_window), score)
        self.assertAlmostEqual(
            calculate_gttm_energy(self.prev, self.next_good, 8, window=self.good_window),
            -score,
        )

    def test_aggregate_energy_prefers_cadential_arrival_over_offbeat_detour(self):
        good_score = calculate_gttm_score(self.prev, self.next_good, 8, window=self.good_window)
        bad_score = calculate_gttm_score(self.prev, self.next_bad, 8, window=self.bad_window)
        good_energy = calculate_gttm_energy(self.prev, self.next_good, 8, window=self.good_window)
        bad_energy = calculate_gttm_energy(self.prev, self.next_bad, 8, window=self.bad_window)

        self.assertGreater(good_score, bad_score)
        self.assertLess(good_energy, bad_energy)


class TestCaching(unittest.TestCase):
    def test_harmonic_distance_cache_reuses_chord_distance_queries(self):
        _cached_basic_space_distance.cache_clear()

        prev = state(chord="G7")
        next_state = state(chord="Cmaj")

        harmonic_chord_proximity_feature(prev, next_state, 0)
        harmonic_chord_proximity_feature(prev, next_state, 0)

        self.assertGreaterEqual(harmonic_distance_cache_info().hits, 1)

    def test_tonal_neighbor_cache_reuses_neighbor_queries(self):
        nearest_roots.cache_clear()
        tonal_distance.cache_clear()

        prev = state(key="C")
        next_state = state(key="G")

        harmonic_key_neighbor_feature(prev, next_state, 0)
        harmonic_key_neighbor_feature(prev, next_state, 0)

        self.assertGreaterEqual(tonal_neighbor_cache_info().hits, 1)


if __name__ == "__main__":
    unittest.main()
