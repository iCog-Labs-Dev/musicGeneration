import unittest
from aimusic.core.rng import RNGKey

from aimusic.planning.candidates import (
    apply_meter_constraints,
    apply_position_constraints,
    apply_role_constraints,
    get_valid_next_states,
    is_legal_transition,
    propose_chord_ids,
    propose_key_ids,
)
from aimusic.core.config import StyleConfig
from aimusic.core.core_types import BeatState
from aimusic.scoring.priors import NeuralPrior, NullPrior
from aimusic.core.vocab import DEFAULT_VOCABULARIES, build_tonal_context


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

    def test_configured_strong_beat_in_seven_four_is_accepted(self):
        prev = state(meter="7/4", beat=3, chord="G7", role="prep")
        candidate = state(
            meter="7/4",
            beat=4,
            boundary="local",
            chord="Cmaj",
            role="cad",
        )

        position_ok, position_reason = apply_position_constraints(
            prev,
            candidate,
            style_config=self.style,
            vocabularies=VOCABS,
        )
        role_ok, role_reason = apply_role_constraints(
            prev,
            candidate,
            vocabularies=VOCABS,
        )

        self.assertTrue(position_ok)
        self.assertIsNone(position_reason)
        self.assertTrue(role_ok)
        self.assertIsNone(role_reason)


class TestCandidateGeneration(unittest.TestCase):
    def setUp(self):
        self.style = StyleConfig(
            allowed_meters=("4/4", "5/4", "7/4"),
            groove_families=("straight", "syncopated", "swing"),
        )
        # Initialize a deterministic RNG for consistent testing
        self.key = RNGKey(seed=42)

    def test_candidate_generation_is_deduplicated_and_legal(self):
        prev = state(
            beat=3,
            key="C",
            chord="G7",
            role="prep",
            head="upper_approach",
            groove="straight_8ths",
        )

        result, _ = get_valid_next_states(
            prev,
            4,
            self.key,
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

        replay, replay_key = get_valid_next_states(
            prev, 4, RNGKey(seed=42), d_max=100, style_config=self.style,
            vocabularies=VOCABS, prior=NeuralPrior(),
        )
        self.assertEqual(replay, result)
        self.assertEqual(replay_key, _)

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

        result, _ = get_valid_next_states(
            prev,
            4,
            self.key,
            d_max=2000,
            # REQ-13: search breadth is controlled by proposal_budget, not
            # d_max. Use a high budget to guarantee we don't accidentally
            # miss the target during (shuffled) generation.
            proposal_budget=4096,
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

    def test_decoder_stream_derivation_cannot_change_candidate_replay(self):
        prev = state(
            beat=3, key="C", chord="G7", role="prep",
            head="upper_approach", groove="straight_8ths",
        )
        root = RNGKey(seed=81)
        proposal_key = root.derive("candidate_proposal")
        first, first_next = get_valid_next_states(
            prev, 4, proposal_key, d_max=100, style_config=self.style,
            vocabularies=VOCABS, prior=NeuralPrior(),
        )
        _ = root.derive("decoder.lead")
        second, second_next = get_valid_next_states(
            prev, 4, proposal_key, d_max=100, style_config=self.style,
            vocabularies=VOCABS, prior=NeuralPrior(),
        )
        self.assertEqual((first, first_next), (second, second_next))

    def test_19_edo_tonal_search_uses_19_step_fifth(self):
        context = build_tonal_context(19, self.style)
        vocabs = context.vocabularies
        prev = BeatState(
            meter_id=vocabs.meters.token_for_label("4/4").id,
            beat_in_bar=0,
            boundary_lvl=vocabs.boundaries.token_for_label("phrase").id,
            key_id=0,
            chord_id=vocabs.chords.token_for_label("pc_0maj").id,
            role_id=vocabs.roles.token_for_label("change").id,
            head_id=vocabs.heads.token_for_label("root").id,
            groove_id=vocabs.grooves.token_for_label("straight_8ths").id,
        )
        role_id = vocabs.roles.token_for_label("change").id

        key_ids = propose_key_ids(prev, 2, role_id, vocabs, edo=context.n)
        chord_ids = propose_chord_ids(
            prev,
            0,
            prev.meter_id,
            1,
            0,
            role_id,
            prev.groove_id,
            NullPrior(),
            None,
            vocabs,
            edo=context.n,
        )
        chord_roots = {vocabs.chords.token_for_id(item).root_pc for item in chord_ids}

        self.assertIn(11, key_ids)
        self.assertIn(11, chord_roots)


if __name__ == "__main__":
    unittest.main()
