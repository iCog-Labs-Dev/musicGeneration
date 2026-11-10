"""Deterministic operation-count guard; no wall-clock performance assertions."""
from __future__ import annotations

import cProfile
import unittest

import mido

from aimusic.audio import midi_events as events
from tests.test_audio_midi_events import message, track


def representative(event_count):
    messages = [(0, message('program_change', program=81))]
    for i in range(event_count // 4):
        tick = 100 + i * 20
        messages.extend([
            (tick, message('pitchwheel', pitch=400 if i % 2 else -400)),
            (tick, message('note_on', note=60 + i % 12)),
            (tick + 2, message('control_change', control=74, value=i % 100)),
            (tick + 4, message('note_off', note=60 + i % 12)),
        ])
    midi = mido.MidiFile(type=1)
    midi.tracks = [track(*[(i * 1000, mido.MetaMessage('set_tempo', tempo=500000 + (i % 2) * 100000))
                          for i in range(event_count // 200)]), track(*messages)]
    return midi


def operations(function, *args):
    profile = cProfile.Profile()
    result = profile.runcall(function, *args)
    count = sum(entry.callcount for entry in profile.getstats()
                if getattr(entry.code, 'co_filename', None) == events.__file__)
    return result, count


class TestMidiScaling(unittest.TestCase):
    def test_2k_4k_event_workloads_do_not_rescan_the_whole_stream_per_note(self):
        counts = {'parse': [], 'retime': [], 'ghosts': []}
        for size in (2000, 4000):
            source, count = operations(events.MidiEvents.from_mido, representative(size))
            counts['parse'].append(count)
            timings = {n.id: events.NoteTiming(n.start + 1, n.end + 1, n.velocity) for n in source.notes}
            shifted, count = operations(events.retime, source, timings)
            counts['retime'].append(count)
            self.assertEqual(len(shifted.notes), size // 4)
            self.assertTrue(all(n.start == source.notes[n.id].start + 1 for n in shifted.notes))
            requests = [(n.id, events.NoteTiming(n.start + 8, n.start + 9, 40)) for n in source.notes]
            ghosted, count = operations(events.add_ghosts, source, requests)
            counts['ghosts'].append(count)
            self.assertEqual(len(ghosted.notes), size // 2)
            self.assertEqual([(n.id, n.pitch, n.start) for n in ghosted.notes[:len(source.notes)]],
                             [(n.id, n.pitch, n.start) for n in source.notes])
        for phase, (small, large) in counts.items():
            with self.subTest(phase=phase, operations=(small, large)):
                self.assertLessEqual(large, small * 2.7,
                                     f'{phase}: doubling 2k to 4k events approaches quadratic work')

    def test_batched_ghosts_match_incremental_insertion_and_keep_request_ids(self):
        midi = representative(40)
        source = events.MidiEvents.from_mido(midi)
        requests = [(n.id, events.NoteTiming(n.start + 8, n.start + 9, 40)) for n in reversed(source.notes)]
        batch = events.add_ghosts(source, requests)
        incremental = source
        for request in requests:
            incremental = events.add_ghosts(incremental, [request])
        self.assertEqual([[m.dict() for m in tr] for tr in batch.to_mido().tracks],
                         [[m.dict() for m in tr] for tr in incremental.to_mido().tracks])
        self.assertEqual([(n.id, n.pitch, n.start) for n in batch.notes],
                         [(n.id, n.pitch, n.start) for n in incremental.notes])

    def test_sweep_association_matches_lifetime_definition_with_sustain(self):
        messages = []
        for i in range(30):
            tick = i * 10
            messages.extend([
                (tick, message('pitchwheel', pitch=i)),
                (tick, message('note_on', note=60)),
                (tick, message('note_on', note=64)),
                (tick + 2, message('control_change', control=74, value=i)),
                (tick + 3, message('polytouch', note=60, value=i)),
                (tick + 4, message('control_change', control=64, value=127)),
                (tick + 5, message('note_off', note=60)),
                (tick + 5, message('note_off', note=64)),
                (tick + 6, message('pitchwheel', pitch=-i)),
                (tick + 7, message('control_change', control=64, value=0)),
                (tick + 7, message('aftertouch', value=0)),
            ])
        midi = mido.MidiFile(type=0)
        midi.tracks = [track(*messages)]
        parsed = events.MidiEvents.from_mido(midi)
        for i, event in enumerate(parsed.events):
            if not events.is_expression(event.message):
                continue
            relevant = [n for n in parsed.notes if n.channel == event.channel and
                        (event.message.type != 'polytouch' or n.pitch == event.message.note)]
            expected = tuple(n.id for n in relevant if (n.on < i < n.off) or
                             (n.end <= event.tick < n.sounding_end) or (n.start == event.tick and i < n.on))
            if not expected:
                expected = tuple(n.id for n in relevant if n.end == n.sounding_end == event.tick and n.off < i)
            self.assertEqual(event.note_ids, expected)
