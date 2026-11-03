import dataclasses
from pathlib import Path
import tempfile
import unittest
import wave

import mido
import numpy as np

from aimusic.audio.preview import PreviewNote, render_midi_preview, voice
from aimusic.core.config import DecodeConfig, StyleConfig
from aimusic.core.core_types import BeatState
from aimusic.core.rng import RNGKey
from aimusic.core.vocab import build_tonal_context
from aimusic.decode import LeadDecoderState, _build_windows, _nearest_pitch, gen_lead
from aimusic.theory.tonal import chord_pitch_classes


class TestMelodyContinuity(unittest.TestCase):
    def test_pitch_search_includes_upper_octaves(self):
        self.assertEqual(_nearest_pitch(87, (0,), (60, 96), 12), 84)

    def test_low_register_edge_uses_nearby_chord_tone_within_cap(self):
        for edo in (12, 19):
            vocabs = build_tonal_context(edo, StyleConfig()).vocabularies
            chord = next(c for c in vocabs.chords if c.root_pc == 0 and c.quality == "maj")
            state = BeatState(0, 0, 0, 0, chord.id, 0,
                              vocabs.heads.token_for_label("seventh").id, 0)
            config = DecodeConfig(lead_density=1, lead_register=(5 * edo, 7 * edo), max_lead_leap_steps=5)
            previous = 5 * edo
            result = gen_lead(state, _build_windows((state,))[0], decoder_state=LeadDecoderState(previous),
                              key=RNGKey(11), decode_config=config, vocabularies=vocabs, edo=edo)
            self.assertLessEqual(abs(result.events[0].h - previous), 5)
            self.assertIn(result.events[0].h % edo, chord_pitch_classes(0, "maj", edo))
            repeated = gen_lead(state, _build_windows((state,))[0], decoder_state=LeadDecoderState(previous),
                                key=RNGKey(11), decode_config=config, vocabularies=vocabs, edo=edo)
            self.assertEqual(result, repeated)

    def test_initial_head_uses_center_and_keeps_head_pitch_class(self):
        vocabs = build_tonal_context(12, StyleConfig()).vocabularies
        state = BeatState(0, 0, 0, 0, 0, 0, vocabs.heads.token_for_label("root").id, 0)
        result = gen_lead(state, _build_windows((state,))[0], decoder_state=LeadDecoderState(),
                          key=RNGKey(11), decode_config=DecodeConfig(lead_density=1), vocabularies=vocabs)
        self.assertEqual(result.events[0].h, 72)


class TestKickPreview(unittest.TestCase):
    def test_kick_has_audible_tail_and_upper_harmonics(self):
        note = PreviewNote(0, 0.075, 36, 100, 9)
        kick = voice(note, 44100, np.random.default_rng(11))
        self.assertGreater(len(kick) / 44100, 0.3)
        tail = kick[int(0.08 * 44100):int(0.18 * 44100)]
        self.assertGreater(float(np.sqrt(np.mean(tail ** 2))), 0.01)
        spectrum = np.abs(np.fft.rfft(kick)) ** 2
        frequencies = np.fft.rfftfreq(len(kick), 1 / 44100)
        self.assertGreater(float(spectrum[(frequencies > 100) & (frequencies < 1000)].sum() / spectrum.sum()), 0.1)
        np.testing.assert_array_equal(kick, voice(dataclasses.replace(note, duration=0.025), 44100, np.random.default_rng(11)))

    def test_export_is_deterministic_and_not_clipped(self):
        with tempfile.TemporaryDirectory() as directory:
            midi = mido.MidiFile()
            track = mido.MidiTrack()
            midi.tracks.append(track)
            track.extend([mido.Message("note_on", note=36, channel=9, velocity=100),
                          mido.Message("note_off", note=36, channel=9, time=60)])
            source = Path(directory) / "kick.mid"
            midi.save(source)
            first = render_midi_preview(source, Path(directory) / "first.wav")
            second = render_midi_preview(source, Path(directory) / "second.wav")
            self.assertEqual(first.read_bytes(), second.read_bytes())
            with wave.open(str(first)) as stream:
                samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2")
            self.assertGreater(np.abs(samples).max(), 0)
            self.assertLess(np.abs(samples).max(), 32767)
