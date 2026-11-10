"""Lossless SMF event and channel-state regression tests (no DSP/assets)."""
from __future__ import annotations

import io
import unittest
from fractions import Fraction

import mido
from hypothesis import given, strategies as st


def track(*events):
    result = mido.MidiTrack()
    previous = 0
    for tick, message in events:
        result.append(message.copy(time=tick - previous))
        previous = tick
    return result


def fixture(file_type=1):
    midi = mido.MidiFile(type=file_type, ticks_per_beat=480)
    conductor = track(
        (0, mido.MetaMessage('set_tempo', tempo=500000)),
        (0, mido.MetaMessage('time_signature', numerator=4, denominator=4)),
        (0, mido.Message('sysex', data=(0x7e, 0x7f, 8, 2, 0, 1, 60, 60, 0, 0))),
        (480, mido.MetaMessage('set_tempo', tempo=1000000)),
        (480, mido.MetaMessage('time_signature', numerator=3, denominator=4)),
        (1000, mido.MetaMessage('end_of_track')),
    )
    notes = track(
        (0, mido.MetaMessage('track_name', name='lead')),
        (0, mido.Message('program_change', channel=2, program=81)),
        (0, mido.Message('control_change', channel=2, control=101, value=0)),
        (0, mido.Message('control_change', channel=2, control=100, value=0)),
        (0, mido.Message('control_change', channel=2, control=6, value=12)),
        (0, mido.Message('control_change', channel=2, control=38, value=50)),
        (100, mido.Message('pitchwheel', channel=2, pitch=321)),
        (100, mido.Message('note_on', channel=2, note=60, velocity=90)),
        (150, mido.Message('aftertouch', channel=2, value=42)),
        (175, mido.Message('polytouch', channel=2, note=60, value=51)),
        (200, mido.Message('control_change', channel=2, control=64, value=127)),
        (500, mido.Message('note_on', channel=2, note=60, velocity=0)),
        (600, mido.Message('pitchwheel', channel=2, pitch=400)),
        (700, mido.Message('control_change', channel=2, control=64, value=0)),
        (1000, mido.MetaMessage('end_of_track')),
    )
    if file_type == 0:
        midi.tracks = [mido.merge_tracks([conductor, notes])]
    else:
        midi.tracks = [conductor, notes]
    return midi


def signature(midi):
    return (midi.type, midi.ticks_per_beat,
            [[msg.dict() for msg in tr] for tr in midi.tracks])


class TestMidiEvents(unittest.TestCase):
    def test_lossless_type_zero_and_one_with_expression_and_globals(self):
        from aimusic.audio.midi_events import MidiEvents
        for kind in (0, 1):
            with self.subTest(kind=kind):
                src = fixture(kind)
                parsed = MidiEvents.from_mido(src)
                self.assertEqual(signature(src), signature(parsed.to_mido()))
                data = io.BytesIO()
                parsed.to_mido().save(file=data)
                data.seek(0)
                self.assertEqual(signature(src), signature(mido.MidiFile(file=data)))
                note = parsed.notes[0]
                self.assertEqual((note.channel, note.start, note.end, note.sounding_end),
                                 (2, 100, 500, 700))
                self.assertEqual(note.track, kind)
                attached = [e.message.type for e in parsed.events if note.id in e.note_ids]
                self.assertIn('pitchwheel', attached)
                self.assertIn('aftertouch', attached)
                self.assertIn('polytouch', attached)
                with self.assertRaises(ValueError):
                    parsed.events[0].message.time = 9

    @given(st.lists(st.integers(0, 127), min_size=1, max_size=30))
    def test_equal_tick_controller_order_round_trips(self, values):
        from aimusic.audio.midi_events import MidiEvents
        midi = mido.MidiFile(type=0)
        midi.tracks = [track(*[(0, mido.Message('control_change', control=1, value=v))
                              for v in values], (20, mido.MetaMessage('end_of_track')))]
        self.assertEqual(signature(midi), signature(MidiEvents.from_mido(midi).to_mido()))

    def test_exact_tempo_map_conversion_and_half_even_offsets(self):
        from aimusic.audio.midi_events import MidiEvents
        tempo = MidiEvents.from_mido(fixture()).tempo_map
        self.assertEqual(tempo.seconds(720), Fraction(1))
        self.assertEqual(tempo.tick(Fraction(1)), Fraction(720))
        # Round offsets, not absolute coordinates: odd source ticks matter.
        self.assertEqual(tempo.shift(101, Fraction(1, 1920)), 101)
        self.assertEqual(tempo.shift(101, Fraction(3, 1920)), 103)
        self.assertEqual(tempo.shift(479, Fraction(1, 320)), 481)

    def test_rejects_unsupported_or_ambiguous_arrangements(self):
        from aimusic.audio.midi_events import MidiEvents, MidiEventError
        bad = []
        m = fixture(); m.type = 2; bad.append((m, 'type'))
        m = fixture(); m.ticks_per_beat = -24; bad.append((m, 'SMPTE'))
        m = fixture(); m.tracks[1][7].time = -1; bad.append((m, 'delta'))
        m = fixture(); m.tracks[1].pop(11); bad.append((m, 'unmatched'))
        m = fixture()  # unmatched off
        m.tracks[1][7] = mido.Message('note_off', channel=2, note=61, time=100)
        bad.append((m, 'unmatched'))
        m = fixture(); m.tracks[1].insert(8, mido.Message('note_on', channel=2, note=60))
        bad.append((m, 'ambiguous'))
        m = fixture(); m.tracks.append(track(
            (800, mido.Message('note_on', channel=2, note=64)),
            (900, mido.Message('note_off', channel=2, note=64))))
        bad.append((m, 'ownership'))
        m = fixture(); m.tracks[1].append(mido.Message('program_change'))
        bad.append((m, 'end_of_track'))
        m = fixture(); m.tracks[1].insert(0, mido.MetaMessage('set_tempo', tempo=600000))
        bad.append((m, 'tempo'))
        for midi, diagnostic in bad:
            with self.subTest(diagnostic=diagnostic):
                with self.assertRaisesRegex(MidiEventError, diagnostic):
                    MidiEvents.from_mido(midi)

    def test_malformed_file_has_actionable_diagnostic(self):
        import tempfile
        from pathlib import Path
        from aimusic.audio.midi_events import MidiEvents, MidiEventError
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'broken.mid'
            path.write_bytes(b'not midi')
            with self.assertRaisesRegex(MidiEventError, 'Malformed MIDI.*broken.mid'):
                MidiEvents.load(path)


def channel_fixture(*events):
    midi = mido.MidiFile(type=1, ticks_per_beat=480)
    midi.tracks = [track((0, mido.MetaMessage('set_tempo', tempo=500000))),
                   track((0, mido.MetaMessage('track_name', name='lead')), *events,
                         (2000, mido.MetaMessage('end_of_track')))]
    return midi


def message(kind, **kwargs):
    return mido.Message(kind, channel=2, **kwargs)


class TestRetiming(unittest.TestCase):
    def retime(self, midi, intervals):
        from aimusic.audio.midi_events import MidiEvents, NoteTiming, retime
        source = MidiEvents.from_mido(midi)
        return source, retime(source, {n.id: NoteTiming(*intervals[n.id], n.velocity)
                                     for n in source.notes})

    def test_expression_moves_persistent_events_and_payloads_survive(self):
        src, out = self.retime(fixture(), [(148, 548)])
        source_payloads = sorted((e.track, e.order, e.message) for e in src.events)
        self.assertEqual(source_payloads, sorted((e.track, e.order, e.message)
                                               for e in out.events if e.order >= 0))
        # Expression crossing the tempo change uses seconds, not a constant tick delta.
        expressive = {e.message.type: e.tick for e in out.events
                      if e.message.type in ('aftertouch', 'polytouch')}
        self.assertEqual(expressive, {'aftertouch': 204, 'polytouch': 232})
        self.assertEqual([(e.tick, e.message) for e in src.events
                          if not e.note_ids], [(e.tick, e.message) for e in out.events
                                              if not e.note_ids and e.order >= 0])
        self.assertEqual(out.notes[0].sounding_end, 700)

    def test_initial_rpn_program_and_controller_state_replayed_before_earlier_note(self):
        midi = channel_fixture(
            (100, message('control_change', control=0, value=3)),
            (100, message('program_change', program=81)),
            (100, message('control_change', control=7, value=80)),
            (100, message('control_change', control=101, value=0)),
            (100, message('control_change', control=100, value=0)),
            (100, message('control_change', control=6, value=12)),
            (120, message('pitchwheel', pitch=400)),
            (120, message('note_on', note=60)),
            (240, message('note_off', note=60)))
        _, out = self.retime(midi, [(60, 180)])
        before = [e.message for e in out.events if e.tick <= 60]
        onset = next(i for i, m in enumerate(before) if m.type == 'note_on')
        self.assertTrue(any(m.type == 'program_change' and m.program == 81 for m in before[:onset]))
        self.assertTrue(any(m.type == 'control_change' and m.control == 6 and m.value == 12
                            for m in before[:onset]))
        self.assertTrue(any(m.type == 'control_change' and m.control == 7 and m.value == 80
                            for m in before[:onset]))
        self.assertEqual(out.notes[0].start, 60)

    def test_reuse_incompatible_bends_rejects_with_note_diagnostics(self):
        from aimusic.audio.midi_events import ChannelStateConflict
        midi = channel_fixture(
            (100, message('pitchwheel', pitch=500)),
            (100, message('note_on', note=60)),
            (200, message('pitchwheel', pitch=700)),
            (300, message('note_off', note=60)),
            (300, message('pitchwheel', pitch=-900)),
            (300, message('note_on', note=64)),
            (500, message('note_off', note=64)))
        with self.assertRaisesRegex(ChannelStateConflict, r'track 1.*channel 2.*note'):
            self.retime(midi, [(100, 350), (250, 450)])

    def test_equal_initial_bends_but_diverging_trajectories_reject(self):
        from aimusic.audio.midi_events import ChannelStateConflict
        midi = channel_fixture(
            (100, message('pitchwheel', pitch=0)),
            (100, message('note_on', note=60)),
            (200, message('pitchwheel', pitch=400)),
            (300, message('note_off', note=60)),
            (300, message('pitchwheel', pitch=0)),
            (300, message('note_on', note=64)),
            (500, message('note_off', note=64)))
        with self.assertRaisesRegex(ChannelStateConflict, 'pitch|state'):
            self.retime(midi, [(100, 300), (150, 350)])

    def test_sustain_overlap_not_just_key_intervals_is_validated(self):
        from aimusic.audio.midi_events import ChannelStateConflict
        midi = channel_fixture(
            (100, message('pitchwheel', pitch=400)),
            (100, message('note_on', note=60)),
            (150, message('control_change', control=64, value=127)),
            (200, message('note_off', note=60)),
            (400, message('control_change', control=64, value=0)),
            (500, message('pitchwheel', pitch=-400)),
            (500, message('note_on', note=64)),
            (600, message('note_off', note=64)))
        with self.assertRaisesRegex(ChannelStateConflict, 'sustain|state'):
            self.retime(midi, [(100, 200), (300, 350)])

    def test_shared_channel_expression_with_different_offsets_rejected(self):
        from aimusic.audio.midi_events import ChannelStateConflict
        midi = channel_fixture(
            (100, message('note_on', note=60)),
            (100, message('note_on', note=64)),
            (200, message('pitchwheel', pitch=400)),
            (300, message('note_off', note=60)),
            (300, message('note_off', note=64)))
        with self.assertRaisesRegex(ChannelStateConflict, 'shared|ambiguous'):
            self.retime(midi, [(150, 350), (175, 375)])
        _, safe = self.retime(midi, [(150, 350), (150, 350)])
        self.assertEqual(len(safe.notes), 2)

    def test_safe_19edo_retiming_preserves_note_identity_within_one_cent(self):
        from aimusic.core.config import EDOConfig
        from aimusic.theory.edo import EDO
        edo = EDO(EDOConfig(n=19, pitch_bend_range=2))
        events = [(0, message('control_change', control=101, value=0)),
                  (0, message('control_change', control=100, value=0)),
                  (0, message('control_change', control=6, value=2))]
        expected = []
        for index, height in enumerate((1, 5, 8, 12)):
            pitch, bend = edo.to_midi(height)
            expected.append(edo.config.base_tuning + height * 12 / 19)
            events += [(200 + index * 400, message('pitchwheel', pitch=bend)),
                       (200 + index * 400, message('note_on', note=pitch)),
                       (400 + index * 400, message('note_off', note=pitch))]
        src, out = self.retime(channel_fixture(*events),
                               [(150 + i * 400, 350 + i * 400) for i in range(4)])
        bend, observed = 0, []
        for e in out.events:
            if e.message.type == 'pitchwheel':
                bend = e.message.pitch
            if e.message.type == 'note_on' and e.message.velocity:
                observed.append(e.message.note + bend / 8192 * 2)
        self.assertEqual([(n.id, n.channel, n.pitch) for n in src.notes],
                         [(n.id, n.channel, n.pitch) for n in out.notes])
        for intended, actual in zip(expected, observed, strict=True):
            self.assertLessEqual(abs(intended - actual) * 100, 1)

    def test_same_tick_reuse_off_and_setup_precede_next_on(self):
        midi = channel_fixture(
            (100, message('pitchwheel', pitch=400)),
            (100, message('note_on', note=60)),
            (200, message('note_off', note=60)),
            (300, message('pitchwheel', pitch=-400)),
            (300, message('note_on', note=64)),
            (400, message('note_off', note=64)))
        _, out = self.retime(midi, [(150, 250), (250, 350)])
        types = [e.message.type for e in out.events if e.tick == 250]
        self.assertLess(types.index('note_off'), types.index('note_on'))
        self.assertLess(types.index('pitchwheel'), types.index('note_on'))

    def test_release_reset_moves_with_reused_mpe_channel(self):
        midi = channel_fixture(
            (100, message('pitchwheel', pitch=400)),
            (100, message('note_on', note=60)),
            (200, message('note_off', note=60)),
            (200, message('pitchwheel', pitch=0)),
            (300, message('pitchwheel', pitch=-400)),
            (300, message('note_on', note=64)),
            (400, message('note_off', note=64)))
        _, out = self.retime(midi, [(0, 100), (150, 250)])
        self.assertEqual([e.tick for e in out.events if e.message.type == 'pitchwheel' and not e.message.pitch], [100])

    def test_poly_pressure_default_after_channel_reuse(self):
        midi = channel_fixture(
            (100, message('note_on', note=60)),
            (150, message('polytouch', note=60, value=80)),
            (200, message('note_off', note=60)),
            (300, message('note_on', note=64)),
            (400, message('note_off', note=64)))
        _, out = self.retime(midi, [(350, 450), (100, 200)])
        self.assertEqual([(n.id, n.start) for n in out.notes], [(0, 350), (1, 100)])

    def test_sustain_replay_must_not_release_another_voice_early(self):
        from aimusic.audio.midi_events import ChannelStateConflict
        midi = channel_fixture(
            (100, message('note_on', note=60)),
            (150, message('control_change', control=64, value=127)),
            (200, message('note_off', note=60)),
            (400, message('control_change', control=64, value=0)),
            (500, message('note_on', note=64)),
            (600, message('note_off', note=64)))
        with self.assertRaisesRegex(ChannelStateConflict, 'sustain'):
            self.retime(midi, [(100, 200), (300, 350)])

    def test_opaque_sysex_boundary_cannot_be_silently_crossed(self):
        from aimusic.audio.midi_events import ChannelStateConflict
        midi = channel_fixture((100, mido.Message('sysex', data=(0x7e, 0x7f, 8, 2))),
                               (150, message('note_on', note=60)),
                               (200, message('note_off', note=60)))
        with self.assertRaisesRegex(ChannelStateConflict, 'SysEx'):
            self.retime(midi, [(50, 100)])

    @given(st.integers(-90, 90), st.integers(-8192, 8191), st.integers(1, 96))
    def test_safe_mpe_shift_property_and_round_trip(self, shift, bend, bend_range):
        from aimusic.audio.midi_events import MidiEvents
        midi = channel_fixture(
            (0, message('control_change', control=101, value=0)),
            (0, message('control_change', control=100, value=0)),
            (0, message('control_change', control=6, value=bend_range)),
            (100, message('pitchwheel', pitch=bend)),
            (100, message('note_on', note=60)),
            (150, message('pitchwheel', pitch=-bend if bend != -8192 else 8191)),
            (200, message('note_off', note=60)))
        _, out = self.retime(midi, [(100 + shift, 200 + shift)])
        parsed = MidiEvents.from_mido(out.to_mido())
        self.assertEqual((parsed.notes[0].start, parsed.notes[0].end), (100 + shift, 200 + shift))
        self.assertEqual([e.message.pitch for e in parsed.events if e.message.type == 'pitchwheel'],
                         [bend, -bend if bend != -8192 else 8191])

    def test_ghost_cannot_detune_next_note_at_same_tick_release(self):
        from aimusic.audio.midi_events import MidiEvents, NoteTiming, add_ghosts
        midi = channel_fixture(
            (0, message('pitchwheel', pitch=400)),
            (10, message('note_on', note=60)),
            (20, message('note_off', note=60)),
            (30, message('pitchwheel', pitch=0)),
            (100, message('pitchwheel', pitch=-400)),
            (100, message('note_on', note=64)),
            (200, message('note_off', note=64)))
        out = add_ghosts(MidiEvents.from_mido(midi), [(0, NoteTiming(80, 100, 40))])
        bend = 0
        for e in out.events:
            if e.message.type == 'pitchwheel':
                bend = e.message.pitch
            if e.message.type == 'note_on' and e.message.note == 64:
                self.assertEqual(bend, -400)
        self.assertEqual(bend, -400)  # restore must not run after the next note's setup
        self.assertEqual(next(n.sounding_end for n in out.notes if n.pitch == 60 and n.start == 80), 100)

    def test_retiming_other_channel_does_not_reorder_sustain_at_equal_tick(self):
        midi = channel_fixture(
            (100, message('note_on', note=60)),
            (200, message('control_change', control=64, value=127)),
            (200, message('note_off', note=60)),
            (300, mido.Message('note_on', channel=3, note=64)),
            (400, mido.Message('note_off', channel=3, note=64)),
            (500, message('control_change', control=64, value=0)))
        _, out = self.retime(midi, [(100, 200), (200, 300)])
        self.assertEqual(out.notes[0].sounding_end, 500)
        messages = [e.message.type for e in out.events if e.channel == 2 and e.tick == 200]
        self.assertEqual(messages, ['control_change', 'note_off'])

    def test_reversed_source_note_order_boundary_moves_release_before_setup(self):
        midi = channel_fixture(
            (100, message('pitchwheel', pitch=400)),
            (100, message('note_on', note=60)),
            (200, message('note_off', note=60)),
            (300, message('pitchwheel', pitch=-400)),
            (300, message('note_on', note=64)),
            (400, message('note_off', note=64)),
            (400, message('pitchwheel', pitch=0)))
        _, out = self.retime(midi, [(200, 300), (100, 200)])
        at_boundary = [e.message for e in out.events if e.tick == 200]
        self.assertEqual(at_boundary[0].type, 'note_off')
        self.assertEqual(at_boundary[-1].type, 'note_on')
        self.assertEqual([m.pitch for m in at_boundary if m.type == 'pitchwheel'], [0, 400])

    def test_same_tick_sysex_order_is_part_of_boundary_validation(self):
        from aimusic.audio.midi_events import ChannelStateConflict
        midi = channel_fixture(
            (100, message('note_on', note=60)),
            (100, mido.Message('sysex', data=(0x7e, 0x7f, 8, 2))),
            (200, message('note_off', note=60)))
        with self.assertRaisesRegex(ChannelStateConflict, 'SysEx'):
            self.retime(midi, [(150, 250)])

    def test_unknown_parameter_default_is_not_invented_for_replay(self):
        from aimusic.audio.midi_events import ChannelStateConflict
        midi = channel_fixture(
            (100, message('note_on', note=60)),
            (200, message('note_off', note=60)),
            (250, message('control_change', control=99, value=1)),
            (250, message('control_change', control=98, value=2)),
            (250, message('control_change', control=6, value=42)),
            (300, message('note_on', note=64)),
            (400, message('note_off', note=64)))
        with self.assertRaisesRegex(ChannelStateConflict, 'unknown.*parameter|parameter.*unknown'):
            self.retime(midi, [(500, 600), (300, 400)])

    def test_device_dependent_relative_parameter_cannot_be_fabricated(self):
        from aimusic.audio.midi_events import ChannelStateConflict
        midi = channel_fixture(
            (100, message('control_change', control=101, value=0)),
            (100, message('control_change', control=100, value=0)),
            (100, message('control_change', control=96, value=0)),
            (150, message('note_on', note=60)),
            (200, message('note_off', note=60)))
        with self.assertRaisesRegex(ChannelStateConflict, 'relative|increment'):
            self.retime(midi, [(50, 100)])

    def test_later_note_does_not_roll_back_persistent_volume_automation(self):
        midi = channel_fixture(
            (0, message('control_change', control=7, value=80)),
            (100, message('note_on', note=60)),
            (200, message('note_off', note=60)),
            (250, message('control_change', control=7, value=100)))
        _, out = self.retime(midi, [(300, 400)])
        self.assertEqual([(e.tick, e.message.value) for e in out.events
                          if e.message.type == 'control_change' and e.message.control == 7],
                         [(0, 80), (250, 100)])

    def test_setup_replay_restores_current_parameter_selector_not_historical_one(self):
        midi = channel_fixture(
            (0, message('program_change', program=81)),
            (100, message('note_on', note=60)),
            (200, message('note_off', note=60)),
            (250, message('program_change', program=80)),
            (250, message('control_change', control=101, value=0)),
            (250, message('control_change', control=100, value=1)),
            (500, message('control_change', control=38, value=40)))
        _, out = self.retime(midi, [(300, 400)])
        selectors = [e.message.value for e in out.events if e.tick <= 500
                     and e.message.type == 'control_change' and e.message.control == 100]
        self.assertEqual(selectors[-1], 1)

    def test_ghost_insertion_retains_original_event_and_note_identities(self):
        from aimusic.audio.midi_events import MidiEvents, NoteTiming, add_ghosts
        midi = channel_fixture(
            (100, message('note_on', note=60)),
            (130, message('note_off', note=60)),
            (300, message('note_on', note=64)),
            (330, message('note_off', note=64)))
        src = MidiEvents.from_mido(midi)
        out = add_ghosts(src, [(0, NoteTiming(160, 170, 40))])
        self.assertEqual([(n.id, n.pitch, n.start) for n in out.notes[:2]],
                         [(n.id, n.pitch, n.start) for n in src.notes])
        self.assertEqual([(e.track, e.order, e.message) for e in out.events
                          if (e.track, e.order) in {(s.track, s.order) for s in src.events}],
                         [(e.track, e.order, e.message) for e in src.events])

    def test_bank_replay_reselects_same_program_and_preserves_pending_bank(self):
        midi = channel_fixture(
            (0, message('control_change', control=0, value=1)),
            (0, message('program_change', program=81)),
            (100, message('note_on', note=60)),
            (200, message('note_off', note=60)),
            (250, message('control_change', control=0, value=2)),
            (250, message('program_change', program=81)),
            (500, message('program_change', program=82)))
        _, out = self.retime(midi, [(300, 400)])
        pending, active = 0, 0
        for e in out.events:
            msg = e.message
            if msg.type == 'control_change' and msg.control == 0:
                pending = msg.value
            if msg.type == 'program_change':
                active = pending
            if msg.type == 'note_on' and msg.velocity:
                self.assertEqual(active, 1)
        self.assertEqual(active, 2)  # later source program uses its source bank

    def test_unlatched_bank_select_does_not_change_note_instrument(self):
        midi = channel_fixture(
            (0, message('control_change', control=0, value=1)),
            (0, message('program_change', program=81)),
            (50, message('control_change', control=0, value=2)),
            (100, message('note_on', note=60)),
            (200, message('note_off', note=60)))
        _, out = self.retime(midi, [(25, 125)])
        pending, active = 0, 0
        for e in out.events:
            msg = e.message
            if msg.type == 'control_change' and msg.control == 0:
                pending = msg.value
            if msg.type == 'program_change':
                active = pending
            if msg.type == 'note_on' and msg.velocity:
                self.assertEqual(active, 1)


class TestReviewRegressions(unittest.TestCase):
    retime = TestRetiming.retime

    def test_sysex_repro_expression_crosses_but_note_endpoints_do_not(self):
        from aimusic.audio.midi_events import ChannelStateConflict
        midi = channel_fixture(
            (100, message('note_on', note=60)),
            (250, mido.Message('sysex', data=(0x7e, 0x7f, 9, 1))),
            (300, message('pitchwheel', pitch=400)),
            (500, message('note_off', note=60)))
        with self.assertRaisesRegex(ChannelStateConflict, 'SysEx'):
            self.retime(midi, [(100, 300)])

    def test_sysex_repro_pre_reset_program_must_not_be_replayed(self):
        from aimusic.audio.midi_events import ChannelStateConflict
        midi = channel_fixture(
            (0, message('program_change', program=81)),
            (100, mido.Message('sysex', data=(0x7e, 0x7f, 9, 1))),
            (150, message('note_on', note=60)),
            (200, message('note_off', note=60)),
            (250, message('program_change', program=80)))
        with self.assertRaisesRegex(ChannelStateConflict, 'SysEx|unknown'):
            self.retime(midi, [(300, 350)])

    def test_ghost_cannot_copy_tuning_across_mts_even_without_replay(self):
        from aimusic.audio.midi_events import MidiEvents, NoteTiming, add_ghosts, ChannelStateConflict
        mts = mido.Message('sysex', data=(0x7f, 0x7f, 8, 2, 0, 1, 60, 61, 0, 0))
        midi = channel_fixture(
            (100, message('note_on', note=60)),
            (150, message('note_off', note=60)),
            (200, mts))
        with self.assertRaisesRegex(ChannelStateConflict, 'SysEx|unknown'):
            add_ghosts(MidiEvents.from_mido(midi), [(0, NoteTiming(250, 280, 40))])

    def test_ordinary_controller_replay_cannot_change_an_unchanged_voice(self):
        from aimusic.audio.midi_events import ChannelStateConflict
        for cc in (7, 10, 11):
            with self.subTest(cc=cc):
                midi = channel_fixture(
                    (100, message('note_on', note=60)),
                    (300, message('control_change', control=cc, value=40)),
                    (400, message('note_on', note=64)),
                    (500, message('note_off', note=64)),
                    (600, message('note_off', note=60)))
                with self.assertRaisesRegex(ChannelStateConflict, 'controller|CC|state'):
                    self.retime(midi, [(100, 600), (200, 250)])

    def test_cc74_exact_association_retiming_and_shared_conflict(self):
        from aimusic.audio.midi_events import MidiEvents, ChannelStateConflict
        midi = channel_fixture(
            (100, message('control_change', control=74, value=12)),
            (100, message('note_on', note=60)),
            (100, message('note_on', note=64)),
            (200, message('control_change', control=74, value=90)),
            (300, message('note_off', note=60)),
            (300, message('note_off', note=64)),
            (300, message('control_change', control=74, value=0)))
        src = MidiEvents.from_mido(midi)
        self.assertEqual([(e.tick, e.note_ids) for e in src.events if e.message.type == 'control_change'],
                         [(100, (0, 1)), (200, (0, 1)), (300, (0, 1))])
        _, out = self.retime(midi, [(150, 450), (150, 450)])
        self.assertEqual([(e.tick, e.message.value, e.note_ids) for e in out.events
                          if e.message.type == 'control_change' and e.message.control == 74],
                         [(150, 12, (0, 1)), (300, 90, (0, 1)), (450, 0, (0, 1))])
        with self.assertRaisesRegex(ChannelStateConflict, 'shared'):
            self.retime(midi, [(150, 450), (175, 475)])

    def test_rpn_pitch_reconstruction_includes_cents_fine_and_coarse_tuning(self):
        events = []
        for parameter, msb, lsb in ((0, 12, 50), (1, 80, 32), (2, 67, 0)):
            events.extend((100, message('control_change', control=cc, value=value))
                          for cc, value in ((101, 0), (100, parameter), (6, msb), (38, lsb)))
        events += [(150, message('pitchwheel', pitch=2048)),
                   (150, message('note_on', note=60)),
                   (250, message('pitchwheel', pitch=-1024)),
                   (350, message('note_off', note=60))]
        _, out = self.retime(channel_fixture(*events), [(50, 250)])
        # Independent MIDI decoder: no transform snapshots or helpers.
        selector = [127, 127]
        values = {0: [2, 0], 1: [64, 0], 2: [64, 0]}
        bend, sounding, observed = 0, False, []
        for event in out.events:
            msg = event.message
            if msg.type == 'control_change':
                if msg.control in (101, 100):
                    selector[msg.control == 100] = msg.value
                if msg.control in (6, 38) and selector[0] == 0 and selector[1] in values:
                    values[selector[1]][msg.control == 38] = msg.value
            if msg.type == 'pitchwheel':
                bend = msg.pitch
            if msg.type == 'note_on':
                sounding = True
            if msg.type == 'note_off':
                sounding = False
            if sounding and msg.type in ('note_on', 'pitchwheel'):
                span = values[0][0] + Fraction(values[0][1], 100)
                fine = Fraction(values[1][0] * 128 + values[1][1] - 8192, 8192)
                observed.append((event.tick, 60 + values[2][0] - 64 + fine + Fraction(bend, 8192) * span))
        self.assertEqual(observed, [(50, Fraction(60 + 3) + Fraction(2080, 8192) + Fraction(25, 8)),
                                    (150, Fraction(60 + 3) + Fraction(2080, 8192) - Fraction(25, 16))])

    def test_short_recognized_meta_payload_has_source_diagnostic(self):
        import tempfile
        from pathlib import Path
        from aimusic.audio.midi_events import MidiEvents, MidiEventError
        for kind in (0x51, 0x58, 0x59):
            with self.subTest(meta=kind), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'short-meta.mid'
                body = bytes((0, 0xff, kind, 0, 0, 0xff, 0x2f, 0))
                path.write_bytes(b'MThd\x00\x00\x00\x06\x00\x00\x00\x01\x01\xe0MTrk'
                                 + len(body).to_bytes(4, 'big') + body)
                with self.assertRaisesRegex(MidiEventError, 'Malformed MIDI.*short-meta.mid'):
                    MidiEvents.load(path)

    def test_opaque_snapshot_forgets_defaults_and_pre_reset_state(self):
        from aimusic.audio.midi_events import MidiEvents, _snapshots
        midi = channel_fixture(
            (0, message('program_change', program=81)),
            (100, mido.Message('sysex', data=(0x7e, 0x7f, 9, 1))),
            (150, message('note_on', note=60)),
            (200, message('note_off', note=60)))
        source, out = self.retime(midi, [(160, 210)])
        state = _snapshots(source.events)[source.notes[0].on]
        self.assertNotIn(('program_change', 0), state)
        self.assertNotIn(('rpn', 0), state)
        self.assertNotIn(('cc', 7), state)
        self.assertFalse(any(e.order < 0 for e in out.events))
        self.assertEqual(len(MidiEvents.from_mido(out.to_mido()).notes), 1)

    def test_ordinary_replay_must_not_leak_into_later_unchanged_voice(self):
        from aimusic.audio.midi_events import ChannelStateConflict
        midi = channel_fixture(
            (150, message('note_on', note=60)),
            (250, message('note_off', note=60)),
            (300, message('control_change', control=7, value=40)),
            (400, message('note_on', note=64)),
            (450, message('note_off', note=64)))
        with self.assertRaisesRegex(ChannelStateConflict, 'state'):
            self.retime(midi, [(150, 250), (50, 100)])

    def test_data_entry_after_sysex_requires_known_parameter_selection(self):
        from aimusic.audio.midi_events import ChannelStateConflict
        for selectors in ((), ((101, 0),), ((100, 0),)):
            with self.subTest(selectors=selectors):
                midi = channel_fixture(
                    (100, mido.Message('sysex', data=(0x7e, 0x7f, 9, 1))),
                    *((140, message('control_change', control=cc, value=value)) for cc, value in selectors),
                    (150, message('control_change', control=6, value=12)),
                    (200, message('note_on', note=60)),
                    (250, message('note_off', note=60)))
                with self.assertRaisesRegex(ChannelStateConflict, 'unknown.*SysEx|SysEx.*unknown'):
                    self.retime(midi, [(120, 170)])


class TestPartialControllerState(unittest.TestCase):
    retime = TestRetiming.retime

    def test_partial_rpn_after_sysex_cannot_fall_outside_shortened_note(self):
        from aimusic.audio.midi_events import ChannelStateConflict
        midi = channel_fixture(
            (100, mido.Message('sysex', data=(0x7e, 0x7f, 9, 1))),
            (100, message('control_change', control=101, value=0)),
            (100, message('control_change', control=100, value=0)),
            (200, message('pitchwheel', pitch=4096)),
            (200, message('note_on', note=60)),
            (300, message('control_change', control=6, value=12)),
            (400, message('note_off', note=60)))
        with self.assertRaisesRegex(ChannelStateConflict, 'trajectory|partial|unknown'):
            self.retime(midi, [(200, 250)])

    def test_ordinary_replay_cannot_contaminate_a_subsequently_retimed_voice(self):
        from aimusic.audio.midi_events import ChannelStateConflict
        for cc in (7, 10, 11):
            with self.subTest(cc=cc):
                midi = channel_fixture(
                    (150, message('note_on', note=60)),
                    (250, message('note_off', note=60)),
                    (300, message('control_change', control=cc, value=40)),
                    (400, message('note_on', note=64)),
                    (450, message('note_off', note=64)))
                with self.assertRaisesRegex(ChannelStateConflict, 'controller|state'):
                    self.retime(midi, [(151, 251), (50, 100)])

    def test_partial_parameter_zero_and_low_half_are_not_unknown_defaults(self):
        from aimusic.audio.midi_events import ChannelStateConflict
        for selectors in ((101, 100), (99, 98)):
            for cc, value in ((6, 0), (38, 0), (38, 50)):
                with self.subTest(selectors=selectors, cc=cc, value=value):
                    midi = channel_fixture(
                        (100, mido.Message('sysex', data=(0x7e, 0x7f, 9, 1))),
                        *((100, message('control_change', control=s, value=0)) for s in selectors),
                        (200, message('pitchwheel', pitch=4096)),
                        (200, message('note_on', note=60)),
                        (300, message('control_change', control=cc, value=value)),
                        (400, message('note_off', note=60)))
                    with self.assertRaisesRegex(ChannelStateConflict, 'trajectory|partial|unknown'):
                        self.retime(midi, [(200, 250)])

    def test_partial_selector_after_sysex_is_not_discarded_by_trajectory_audit(self):
        from aimusic.audio.midi_events import ChannelStateConflict
        midi = channel_fixture(
            (100, mido.Message('sysex', data=(0x7e, 0x7f, 9, 1))),
            (200, message('note_on', note=60)),
            (300, message('control_change', control=101, value=0)),
            (400, message('note_off', note=60)))
        with self.assertRaisesRegex(ChannelStateConflict, 'trajectory|partial|unknown'):
            self.retime(midi, [(200, 250)])

    def test_partial_rpn_change_kept_inside_lifetime_can_be_preserved(self):
        midi = channel_fixture(
            (100, mido.Message('sysex', data=(0x7e, 0x7f, 9, 1))),
            (100, message('control_change', control=101, value=0)),
            (100, message('control_change', control=100, value=0)),
            (200, message('pitchwheel', pitch=4096)),
            (200, message('note_on', note=60)),
            (300, message('control_change', control=6, value=12)),
            (400, message('note_off', note=60)))
        _, out = self.retime(midi, [(200, 350)])
        self.assertEqual([(e.tick, e.message.value) for e in out.events
                          if e.message.type == 'control_change' and e.message.control == 6], [(300, 12)])
        self.assertEqual(out.notes[0].end, 350)

    def test_earlier_replay_yields_to_absolute_source_automation(self):
        midi = channel_fixture(
            (100, message('control_change', control=7, value=60)),
            (300, message('control_change', control=7, value=40)),
            (400, message('note_on', note=60)),
            (500, message('note_off', note=60)))
        _, out = self.retime(midi, [(50, 450)])
        self.assertEqual([(e.tick, e.message.value) for e in out.events
                          if e.message.type == 'control_change' and e.message.control == 7],
                         [(50, 40), (100, 60), (300, 40)])

    def test_shortening_does_not_require_later_ordinary_automation(self):
        midi = channel_fixture(
            (200, message('note_on', note=60)),
            (300, message('control_change', control=7, value=40)),
            (400, message('note_off', note=60)))
        _, out = self.retime(midi, [(201, 250)])
        self.assertEqual([(e.tick, e.message.value) for e in out.events
                          if e.message.type == 'control_change' and e.message.control == 7], [(300, 40)])
        self.assertEqual((out.notes[0].start, out.notes[0].end), (201, 250))
