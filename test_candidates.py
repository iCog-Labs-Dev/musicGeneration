import unittest
import numpy as np

from candidates import (
    apply_meter_constraints,
    apply_position_constraints,
    apply_role_constraints,
    get_valid_next_states,
    is_legal_transition,
)
from config import StyleConfig
from core_types import BeatState
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


class TestHardGatingRules(unittest.TestCase):
    def setUp(self):
        self.style = StyleConfig(
            allowed_meters=("4/4", "5/4", "7/4"),
            groove_families=("straight", "syncopated", "swing"),
        )

    def test_meter_change_mid_bar_is_rejected(self):
        prev = state(meter="4/4", beat=1, boundary="phrase")
        candidate = state(meter="5/4", beat=0, boundary="phrase")

        ok, reason = apply_meter_constraints(
            prev,
            candidate,
            style_config=self.style,
            vocabularies=VOCABS,
        )

        self.assertFalse(ok)
        self.assertEqual(reason, "meter_change_requires_downbeat_source")

    def test_phrase_boundary_on_weak_beat_is_rejected(self):
        prev = state(beat=0, role="prep")
        candidate = state(beat=1, boundary="phrase", role="cad")

        ok, reason = apply_position_constraints(
            prev,
            candidate,
            style_config=self.style,
            vocabularies=VOCABS,
        )

        self.assertFalse(ok)
        self.assertEqual(reason, "boundary_requires_strong_beat")

    def test_cadential_role_on_weak_beat_is_rejected(self):
        prev = state(beat=0, role="prep")
        candidate = state(beat=1, boundary="local", role="cad")

        ok, reason = apply_role_constraints(
            prev,
            candidate,
            vocabularies=VOCABS,  
        )

        self.assertFalse(ok)
        self.assertEqual(reason, "cadence_requires_strong_beat")


class TestCandidateGeneration(unittest.TestCase):
    def setUp(self):
        self.style = StyleConfig(
            allowed_meters=("4/4", "5/4", "7/4"),
            groove_families=("straight", "syncopated", "swing"),
        )
        # Initialize a deterministic RNG for consistent testing
        self.rng = np.random.default_rng(42)

    def test_candidate_generation_is_deduplicated_and_legal(self):
        prev = state(
            beat=3,
            key="C",
            chord="G7",
            role="prep",
            head="upper_approach",
            groove="straight_8ths",
        )

        result = get_valid_next_states(
            prev,
            4,
            self.rng,
            d_max=100,  
            style_config=self.style,
            vocabularies=VOCABS,
            prior=NeuralPrior(),
        )

        self.assertEqual(len(result.states), len(set(result.states)))
        self.assertGreater(len(result.states), 0)
        self.assertTrue(
            all(
                is_legal_transition(
                    prev,
                    candidate,
                    style_config=self.style,
                    vocabularies=VOCABS,
                )[0]
                for candidate in result.states
            )
        )

    def test_candidate_generation_includes_cadence_targets(self):
        prev = state(
            beat=3,
            key="C",
            chord="G7",
            role="prep",
            head="upper_approach",
            groove="straight_8ths",
        )
        cadence_chord_id = VOCABS.chords.token_for_label("Cmaj").id
        cadence_role_id = VOCABS.roles.token_for_label("cad").id

        result = get_valid_next_states(
            prev,
            4,
            self.rng,
            d_max=2000, # Use a high bound to guarantee we don't accidentally prune the target early
            style_config=self.style,
            vocabularies=VOCABS,
            prior=NeuralPrior(),
        )

        self.assertTrue(
            any(
                candidate.beat_in_bar == 0
                and candidate.boundary_lvl >= VOCABS.boundaries.token_for_label("phrase").id
                and candidate.chord_id == cadence_chord_id
                and candidate.role_id == cadence_role_id
                for candidate in result.states
            )
        )


if __name__ == "__main__":
    unittest.main()