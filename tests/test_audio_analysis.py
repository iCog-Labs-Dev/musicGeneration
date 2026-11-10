"""Imported MIDI role inference keeps GM pad and lead families distinct."""

import tempfile
import unittest
from pathlib import Path

import pretty_midi as pm

from aimusic.audio.analysis.from_midi import analyze_midi


class TestImportedRoles(unittest.TestCase):
    def test_program_boundaries_and_legacy_names(self) -> None:
        cases = [
            (0, "", "piano"), (7, "", "piano"), (8, "", "comping"),
            (31, "", "comping"), (32, "", "bass"), (39, "", "bass"),
            (40, "", "comping"), (51, "", "comping"), (79, "", "comping"),
            (80, "", "lead"), (87, "", "lead"), (88, "", "pads"),
            (95, "", "pads"), (96, "", "lead"), (103, "", "lead"),
            (104, "", "comping"), (127, "", "comping"),
            (80, "comp", "comping"), (80, "comping", "comping"),
            (88, "Lead solo", "lead"), (80, "Pads", "pads"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.mid"
            for program, name, expected in cases:
                with self.subTest(program=program, name=name):
                    midi = pm.PrettyMIDI()
                    inst = pm.Instrument(program, name=name)
                    inst.notes.append(pm.Note(90, 60, 0.0, 1.0))
                    midi.instruments.append(inst)
                    midi.write(str(path))
                    doc = analyze_midi(path, output_path=Path(tmp) / "analysis.json")
                    self.assertEqual(doc["tracks"][0]["role"], expected)

    def test_legacy_file_analysis_provides_source_identity_for_groove(self) -> None:
        import mido
        from aimusic.audio.groove.apply import apply_groove
        from aimusic.audio.midi_events import MidiEvents
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source.mid'
            midi = mido.MidiFile(type=1, ticks_per_beat=480)
            # Empty tracks make instrument indices unusable as source identities.
            midi.tracks = [mido.MidiTrack(), mido.MidiTrack(), mido.MidiTrack([
                mido.MetaMessage('track_name', name='texture'),
                mido.Message('program_change', program=80),
                mido.Message('note_on', note=60, velocity=80),
                mido.Message('note_off', note=60, time=240)])]
            midi.save(source)
            doc = analyze_midi(source, output_path=root / 'analysis.json')
            self.assertEqual([(r['source_track_index'], r['role']) for r in doc['tracks']], [(2, 'lead')])
            out = apply_groove(source, {'microtiming_ms': {'lead': {'global_offset': 12.5}}},
                               analysis=doc, output_path=root / 'out.mid')
            self.assertEqual([(n.start, n.end) for n in MidiEvents.load(out).notes], [(12, 252)])
