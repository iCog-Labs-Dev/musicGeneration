"""REQ-11: track generators refactored into pure state/window/RNG functions.

Covers the acceptance criteria that don't already live in test_decode.py
(which is left untouched and still exercises register/density/anchoring/
overlap guarantees end-to-end, unchanged, to prove no behavior regression):

- Each track generator is independently callable and testable.
- Replaying the same decoder state and key reproduces the same events and
  next state.
- Voice-leading / motif memory is explicit in returned decoder state.
- Track order does not change generated results.
"""
from __future__ import annotations

import unittest

from aimusic.core.config import DecodeConfig
from aimusic.core.core_types import BeatState
from aimusic.core.rng import RNGKey
from aimusic.core.vocab import DEFAULT_VOCABULARIES
from aimusic.decode import (
    BassDecoderState,
    CompingDecoderState,
    DrumDecoderState,
    LeadDecoderState,
    _build_windows,
    _run_track,
    gen_bass,
    gen_comping,
    gen_drums,
    gen_lead,
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


class TestEachGeneratorIsIndependentlyCallable(unittest.TestCase):
    """No whole path required -- one beat, one window, one call."""

    def _one_beat_window(self, beat_state):
        return _build_windows((beat_state,))[0]

    def test_gen_bass_on_a_single_beat(self):
        beat_state = state(chord="Cmaj", role="hold")
        window = self._one_beat_window(beat_state)
        result = gen_bass(
            beat_state, window,
            decoder_state=BassDecoderState(), key=RNGKey(seed=1),
            decode_config=DecodeConfig(bass_density=1.0), vocabularies=VOCABS,
        )
        self.assertEqual(len(result.events), 1)
        self.assertEqual(result.events[0].track, "bass")
        self.assertIsInstance(result.decoder_state, BassDecoderState)

    def test_gen_comping_on_a_single_beat(self):
        beat_state = state(chord="G7")
        window = self._one_beat_window(beat_state)
        result = gen_comping(
            beat_state, window,
            decoder_state=CompingDecoderState(), key=RNGKey(seed=1),
            decode_config=DecodeConfig(comping_density=1.0), vocabularies=VOCABS,
        )
        self.assertGreater(len(result.events), 0)
        self.assertTrue(all(e.track == "comping" for e in result.events))

    def test_gen_lead_on_a_single_beat(self):
        beat_state = state(chord="Cmaj", head="root")
        window = self._one_beat_window(beat_state)
        result = gen_lead(
            beat_state, window,
            decoder_state=LeadDecoderState(), key=RNGKey(seed=1),
            decode_config=DecodeConfig(lead_density=1.0), vocabularies=VOCABS,
        )
        self.assertEqual(len(result.events), 1)
        self.assertEqual(result.events[0].track, "lead")

    def test_gen_drums_on_a_single_beat(self):
        beat_state = state()
        window = self._one_beat_window(beat_state)
        result = gen_drums(
            beat_state, window,
            decoder_state=DrumDecoderState(), key=RNGKey(seed=1),
            decode_config=DecodeConfig(drum_density=1.0), vocabularies=VOCABS,
        )
        self.assertGreater(len(result.events), 0)
        self.assertTrue(all(e.track == "drums" for e in result.events))


class TestReplayDeterminism(unittest.TestCase):
    """Same decoder state + key + inputs => same events and next state."""

    def _window(self, states, index):
        return _build_windows(states)[index]

    def test_gen_bass_replay_is_stable(self):
        states = (state(chord="Cmaj"), state(beat=1, chord="G7"))
        window = self._window(states, 1)
        decoder_state = BassDecoderState(prev_pitch=40)
        key = RNGKey(seed=42)
        config = DecodeConfig(bass_density=1.0)

        result_a = gen_bass(
            states[1], window, decoder_state=decoder_state, key=key,
            decode_config=config, vocabularies=VOCABS,
        )
        result_b = gen_bass(
            states[1], window, decoder_state=decoder_state, key=key,
            decode_config=config, vocabularies=VOCABS,
        )
        self.assertEqual(result_a, result_b)

    def test_run_track_replay_is_stable_for_every_generator(self):
        states = (
            state(chord="Cmaj", head="root", boundary="phrase"),
            state(beat=1, chord="G7", role="prep", head="third"),
            state(beat=2, chord="Cmaj", role="cad", head="root"),
        )
        windows = _build_windows(states)
        config = DecodeConfig(
            bass_density=0.8, comping_density=0.8, lead_density=0.8, drum_density=0.8,
        )
        cases = (
            (gen_bass, BassDecoderState()),
            (gen_comping, CompingDecoderState()),
            (gen_lead, LeadDecoderState()),
            (gen_drums, DrumDecoderState()),
        )
        for generator, initial_state in cases:
            with self.subTest(generator=generator.__name__):
                run_a = _run_track(
                    states, windows, key=RNGKey(seed=7),
                    initial_decoder_state=initial_state, generator=generator,
                    decode_config=config, vocabularies=VOCABS, edo=12,
                    ticks_per_beat=480,
                )
                run_b = _run_track(
                    states, windows, key=RNGKey(seed=7),
                    initial_decoder_state=initial_state, generator=generator,
                    decode_config=config, vocabularies=VOCABS, edo=12,
                    ticks_per_beat=480,
                )
                self.assertEqual(run_a, run_b)


class TestDecoderStateMakesVoiceLeadingExplicit(unittest.TestCase):
    """Voice-leading memory must be a value the caller can inspect, override,
    and replay -- not a variable hidden in a loop closure."""

    def test_bass_pitch_choice_depends_on_injected_decoder_state(self):
        beat_state = state(chord="Cmaj", role="hold")
        window = _build_windows((beat_state,))[0]
        config = DecodeConfig(bass_density=1.0)

        from_low = gen_bass(
            beat_state, window, decoder_state=BassDecoderState(prev_pitch=28),
            key=RNGKey(seed=1), decode_config=config, vocabularies=VOCABS,
        )
        from_high = gen_bass(
            beat_state, window, decoder_state=BassDecoderState(prev_pitch=52),
            key=RNGKey(seed=1), decode_config=config, vocabularies=VOCABS,
        )

        # Same state/window/key, only the explicit decoder state differs --
        # and that alone is enough to move the voice-led pitch choice.
        self.assertNotEqual(from_low.events[0].h, from_high.events[0].h)
        self.assertEqual(from_low.decoder_state.prev_pitch, from_low.events[0].h)
        self.assertEqual(from_high.decoder_state.prev_pitch, from_high.events[0].h)

    def test_lead_skip_carries_previous_pitch_forward_unchanged(self):
        # boundary_lvl=0 + head="rest" is a deliberate skip in gen_lead;
        # decoder state must pass through untouched, not reset.
        beat_state = state(head="rest", boundary="none")
        window = _build_windows((beat_state,))[0]
        decoder_state = LeadDecoderState(prev_pitch=64)

        result = gen_lead(
            beat_state, window, decoder_state=decoder_state, key=RNGKey(seed=3),
            decode_config=DecodeConfig(lead_density=1.0), vocabularies=VOCABS,
        )

        self.assertEqual(result.events, ())
        self.assertEqual(result.decoder_state, decoder_state)


class TestTrackOrderDoesNotChangeResults(unittest.TestCase):
    def test_running_tracks_in_different_orders_yields_the_same_events(self):
        states = (
            state(chord="Cmaj", head="root", boundary="phrase"),
            state(beat=1, chord="G7", role="prep", head="third"),
            state(beat=2, chord="Cmaj", role="cad", head="root"),
            state(beat=3, chord="Fmaj", role="change", head="fifth"),
        )
        windows = _build_windows(states)
        config = DecodeConfig()
        root = RNGKey(seed=21)
        tracks = {
            "bass": (gen_bass, BassDecoderState(), root.derive("decoder.bass")),
            "comping": (gen_comping, CompingDecoderState(), root.derive("decoder.comping")),
            "lead": (gen_lead, LeadDecoderState(), root.derive("decoder.lead")),
            "drums": (gen_drums, DrumDecoderState(), root.derive("decoder.drums")),
        }

        def run_in_order(order):
            all_events = []
            for name in order:
                generator, initial_state, key = tracks[name]
                events, _, _ = _run_track(
                    states, windows, key=key, initial_decoder_state=initial_state,
                    generator=generator, decode_config=config, vocabularies=VOCABS,
                    edo=12, ticks_per_beat=480,
                )
                all_events.append((name, events))
            return dict(all_events)

        forward = run_in_order(["bass", "comping", "lead", "drums"])
        reversed_order = run_in_order(["drums", "lead", "comping", "bass"])
        shuffled = run_in_order(["lead", "bass", "drums", "comping"])

        self.assertEqual(forward, reversed_order)
        self.assertEqual(forward, shuffled)


class TestDeterministicActivationPolicy(unittest.TestCase):
    def test_fixed_policy_is_key_independent(self):
        states = (state(chord="Cmaj"), state(beat=1, chord="G7"), state(beat=2, chord="Amin"))
        windows = _build_windows(states)
        config = DecodeConfig(lead_density=0.5)  # deterministic_activation=True (default)

        events_a, _, _ = _run_track(
            states, windows, key=RNGKey(seed=1), initial_decoder_state=LeadDecoderState(),
            generator=gen_lead, decode_config=config, vocabularies=VOCABS, edo=12,
            ticks_per_beat=480,
        )
        events_b, _, _ = _run_track(
            states, windows, key=RNGKey(seed=999), initial_decoder_state=LeadDecoderState(),
            generator=gen_lead, decode_config=config, vocabularies=VOCABS, edo=12,
            ticks_per_beat=480,
        )
        self.assertEqual(events_a, events_b)

    def test_keyed_policy_varies_with_key_but_replays_deterministically(self):
        states = tuple(state(beat=i % 4, chord="Cmaj") for i in range(12))
        windows = _build_windows(states)
        config = DecodeConfig(lead_density=0.5, deterministic_activation=False)

        events_seed1, _, _ = _run_track(
            states, windows, key=RNGKey(seed=1), initial_decoder_state=LeadDecoderState(),
            generator=gen_lead, decode_config=config, vocabularies=VOCABS, edo=12,
            ticks_per_beat=480,
        )
        events_seed2, _, _ = _run_track(
            states, windows, key=RNGKey(seed=2), initial_decoder_state=LeadDecoderState(),
            generator=gen_lead, decode_config=config, vocabularies=VOCABS, edo=12,
            ticks_per_beat=480,
        )
        replay_seed1, _, _ = _run_track(
            states, windows, key=RNGKey(seed=1), initial_decoder_state=LeadDecoderState(),
            generator=gen_lead, decode_config=config, vocabularies=VOCABS, edo=12,
            ticks_per_beat=480,
        )

        self.assertNotEqual(events_seed1, events_seed2)
        self.assertEqual(events_seed1, replay_seed1)


if __name__ == "__main__":
    unittest.main()
