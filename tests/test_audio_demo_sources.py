"""Four reproducible sources: three reference fixtures plus real Method A output."""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import mido
import numpy as np
import soundfile as sf

from aimusic.audio.demo_sources import SOURCE_NAMES, build_source, write_source
from aimusic.audio.from_midi import from_midi
from aimusic.audio.midi_events import MidiEvents, NoteTiming, retime
from aimusic.audio.render.tuning import required_tuning
from tests.test_audio_entry import _generate_package, _run_cli


def files(root: Path) -> dict[str, bytes]:
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob('*') if p.is_file()}


def pitches(source: MidiEvents) -> list[tuple[int, int, int, float]]:
    """Independent oracle for the documented RPN 0, +/-2-semitone subset."""
    wheel = [0] * 16
    ranges = [2.0] * 16
    selection = [[127, 127] for _ in range(16)]
    result = []
    for event in source.events:
        msg = event.message
        if event.channel is None:
            continue
        ch = msg.channel
        if msg.type == 'control_change':
            if msg.control in (101, 100):
                selection[ch][msg.control - 100] = msg.value
            elif msg.control == 6 and selection[ch] == [0, 0]:
                ranges[ch] = msg.value
            elif msg.control == 38 and selection[ch] == [0, 0]:
                ranges[ch] += msg.value / 100
        elif msg.type == 'pitchwheel':
            wheel[ch] = msg.pitch
        elif msg.type == 'note_on' and msg.velocity:
            hz = 440 * 2 ** ((msg.note - 69 + wheel[ch] / 8192 * ranges[ch]) / 12)
            result.append((event.track, ch, msg.note, hz))
    return result


class TestDemoSources(unittest.TestCase):
    def test_three_fixtures_are_byte_reproducible_and_canonical(self) -> None:
        self.assertEqual(SOURCE_NAMES, ('fixture-12', 'reference-19', 'multitrack-19'))
        with tempfile.TemporaryDirectory() as tmp:
            for name in SOURCE_NAMES:
                with self.subTest(source=name):
                    left, right = Path(tmp) / (name + '-a'), Path(tmp) / (name + '-b')
                    write_source(name, left)
                    write_source(name, right)
                    self.assertEqual(files(left), files(right))
                    midi, intent = build_source(name)
                    self.assertEqual((left / 'source.mid').read_bytes(), midi)
                    self.assertEqual(json.loads((left / 'expected.json').read_bytes()), intent)
                    manifest = json.loads((left / 'source.json').read_bytes())
                    self.assertEqual(manifest['schema'], 'aimusic.listening-sources/1')
                    for filename, digest in manifest['files'].items():
                        self.assertEqual(digest, hashlib.sha256((left / filename).read_bytes()).hexdigest())
                    for path in left.glob('*.json'):
                        value = json.loads(path.read_bytes())
                        canonical = (json.dumps(value, sort_keys=True, separators=(',', ':'),
                                                ensure_ascii=True, allow_nan=False) + '\n').encode()
                        self.assertEqual(path.read_bytes(), canonical)
                    context = from_midi(left / 'source.mid')
                    self.assertEqual(context.events.type, 1)
                    self.assertEqual(context.events.ticks_per_beat, 480)
                    self.assertEqual(context.analysis_document()['meter_map'], [[0, 4, 4]])
                    self.assertEqual(context.events.tempo_map.changes, ((0, 500000),))
                    self.assertEqual(context.reconciliation['status'], 'not_applicable')

    def test_eight_bar_fixture_roles_programs_note_counts_and_duration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            write_source('fixture-12', Path(tmp))
            context = from_midi(Path(tmp) / 'source.mid')
            doc = context.analysis_document()
            self.assertEqual([(t['name'], t['channels'], t['program'], t['note_count']) for t in doc['tracks']],
                             [('drums', [9], 0, 96), ('bass', [0], 33, 16),
                              ('comping', [1], 0, 48), ('lead', [2], 80, 64)])
            self.assertEqual(doc['duration_sec'], 16)
            self.assertEqual(doc['bar_boundaries_sec'], list(range(0, 17, 2)))
            self.assertEqual(required_tuning(context.events, doc), 'direct_12')
            self.assertEqual({n.pitch for n in context.events.notes if n.channel == 9}, {36, 38, 42})
            self.assertEqual({e.tick for e in context.events.events if e.message.type == 'end_of_track'}, {15360})

    def test_19edo_intent_is_independent_and_survives_retime_and_isolation(self) -> None:
        for name in ('reference-19', 'multitrack-19'):
            with self.subTest(source=name):
                payload, intent = build_source(name)
                source = MidiEvents.from_bytes(payload)
                self.assertEqual(required_tuning(source, {}), 'channel_pitch')
                self.assertEqual(len(source.notes), len(intent['notes']))
                self.assertEqual(len(source.isolate_tracks()), 1 if name == 'reference-19' else 3)
                timings = {n.id: NoteTiming(n.start + 12, n.end + 12, n.velocity) for n in source.notes}
                moved = retime(source, timings)
                from aimusic.audio.groove.apply import apply_groove
                with tempfile.TemporaryDirectory() as tmp:
                    original = Path(tmp) / 'source.mid'
                    original.write_bytes(payload)
                    path = apply_groove(original, {'microtiming_ms': {
                        role: {'global_offset': 12.5} for role in ('bass', 'comp', 'lead')
                    }}, output_path=Path(tmp) / 'expressive.mid')
                    expressive = MidiEvents.load(path)
                    self.assertEqual(original.read_bytes(), payload)
                    self.assertTrue(Path(str(path) + '.diff.json').is_file())
                variants = [source, moved, expressive, *expressive.isolate_tracks().values()]
                for variant in variants:
                    buffer = io.BytesIO()
                    variant.to_mido().save(file=buffer)
                    self.assertEqual(pitches(MidiEvents.from_bytes(buffer.getvalue())), pitches(variant))
                    setup = lambda m: [(e.tick, e.track, e.message) for e in m.events
                                       if e.message.type in ('set_tempo', 'time_signature', 'program_change')]
                    self.assertEqual(setup(variant), [row for row in setup(source)
                                     if row[2].type != 'program_change' or row[1] in {n.track for n in variant.notes}])
                expected = {(n['track'], n['channel'], n['midi_note'], n['start_tick']): n for n in intent['notes']}
                # Each channel's repeated sequence is compared in onset order.
                for variant in variants:
                    oracle = pitches(variant)
                    rows = [row for row in intent['notes'] if row['track'] in {n.track for n in variant.notes}]
                    self.assertEqual(len(oracle), len(rows))
                    for actual, row in zip(oracle, rows):
                        self.assertEqual(actual[:3], (row['track'], row['channel'], row['midi_note']))
                        hz = 440 * 2 ** (row['edo_step_from_a4'] / 19)
                        self.assertAlmostEqual(row['expected_hz'], hz, places=10)
                        self.assertLessEqual(abs(1200 * math.log2(actual[3] / hz)), 1)
                self.assertEqual(len(expected), len(source.notes))
                for ch in {n.channel for n in source.notes}:
                    controls = [(e.message.control, e.message.value) for e in source.events
                                if e.channel == ch and e.message.type == 'control_change']
                    self.assertEqual(controls, [(101, 0), (100, 0), (6, 2), (38, 0), (101, 127), (100, 127)])
                    notes = [n for n in source.notes if n.channel == ch]
                    self.assertTrue(all(a.end < b.start for a, b in zip(notes, notes[1:])))
                self.assertFalse(any(e.message.type == 'sysex' for e in source.events))
                corrupted = mido.MidiFile(file=io.BytesIO(payload))
                for track in corrupted.tracks:
                    for msg in track:
                        if msg.type == 'pitchwheel':
                            msg.pitch = 0
                flat = pitches(MidiEvents.from_mido(corrupted))
                errors = [abs(1200 * math.log2(a[3] / b['expected_hz'])) for a, b in zip(flat, intent['notes'])]
                self.assertGreater(max(errors), 15)
                if name == 'reference-19':
                    self.assertEqual([r['edo_step_from_a4'] for r in intent['notes']], [1, 4, 7])
                    self.assertTrue(all(e > 15 for e in errors))
                    self.assertEqual([(w['start_sample'], w['stop_sample']) for w in intent['windows']],
                                     [(12000, 36000), (84000, 108000), (156000, 180000)])
                    self.assertEqual(intent['sample_rate'], 48000)
                    self.assertEqual([w['expected_hz'] for w in intent['windows']],
                                     [r['expected_hz'] for r in intent['notes']])
                    for window, note in zip(intent['windows'], source.notes):
                        self.assertGreater(window['start_sample'] / 48000, float(source.tempo_map.seconds(note.start)))
                        self.assertLess(window['stop_sample'] / 48000, float(source.tempo_map.seconds(note.end)))
                else:
                    self.assertEqual(intent['windows'], [])
                    self.assertEqual(intent['audio_pitch_accuracy'], 'unmeasured')

    def test_reference_sidecar_is_directly_consumable_by_tuning_checker(self) -> None:
        from aimusic.audio.scoring.tuning_check import ToneWindow, analyze_tuning
        _, intent = build_source('reference-19')
        windows = tuple(ToneWindow(**row) for row in intent['windows'])
        samples = np.zeros(216000)
        for window in windows:
            assert window.expected_hz is not None
            time = np.arange(window.stop_sample - window.start_sample) / 48000
            samples[window.start_sample:window.stop_sample] = 0.2 * np.sin(2 * np.pi * window.expected_hz * time)
        report = analyze_tuning(samples, 48000, windows)
        report.require_pass()  # synthetic PCM only; never native acceptance

    def test_output_refuses_collisions_links_and_unknown_sources_without_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for filename in ('source.mid', 'expected.json', 'source.json'):
                for kind in ('file', 'directory', 'symlink', 'dangling'):
                    with self.subTest(filename=filename, kind=kind):
                        out = root / (filename + kind)
                        out.mkdir()
                        target = out / filename
                        if kind == 'file':
                            target.write_bytes(b'keep')
                        elif kind == 'directory':
                            target.mkdir()
                        else:
                            try:
                                target.symlink_to(root / ('missing' if kind == 'dangling' else 'outside'))
                            except OSError as exc:
                                if getattr(exc, 'winerror', None) == 1314:
                                    self.skipTest('Native symlink privilege unavailable; ordinary collisions remain mandatory')
                                raise
                        before = files(out)
                        with self.assertRaises((ValueError, FileExistsError)):
                            write_source('fixture-12', out)
                        self.assertEqual(before, files(out))
                        self.assertEqual(list(out.iterdir()), [target])
            with self.subTest(kind='root-symlink'):
                link = root / 'linked'
                try:
                    link.symlink_to(root, target_is_directory=True)
                except OSError as exc:
                    if getattr(exc, 'winerror', None) == 1314:
                        self.skipTest('Native symlink privilege unavailable; native junction test remains mandatory when supported')
                    raise
                with self.assertRaisesRegex(ValueError, 'symlink|reparse'):
                    write_source('fixture-12', link)
            with self.assertRaisesRegex(ValueError, 'source'):
                write_source('other', root / 'bad')
            self.assertFalse((root / 'bad').exists())

    def test_module_command_and_seed_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'cli'
            cmd = [sys.executable, '-m', 'aimusic.audio.demo_sources', 'fixture-12', '--out', str(out)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((out / 'source.mid').read_bytes(), build_source('fixture-12')[0])
            again = subprocess.run(cmd, capture_output=True, text=True)
            self.assertNotEqual(again.returncode, 0)
            self.assertNotIn('Traceback', again.stderr)
            self.assertNotEqual(build_source('fixture-12', seed=11)[0], build_source('fixture-12', seed=12)[0])
            for seed in (-1, True):
                with self.assertRaises((TypeError, ValueError)):
                    build_source('fixture-12', seed=seed)


class TestFourSourceWorkflow(unittest.TestCase):
    def test_fresh_method_a_and_fixture_render_through_preview_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'AIMUSIC_AUDIO_BACKEND': 'native'}):
            os.environ.pop('AIMUSIC_SOUNDFONT', None)
            root = Path(tmp)
            # This is the actual planner+decoder, not randomized fixture playback.
            a = _generate_package(root / 'generated-a', seed=11, beats=4)
            b = _generate_package(root / 'generated-b', seed=11, beats=4)
            self.assertEqual((a / 'score.mid').read_bytes(), (b / 'score.mid').read_bytes())
            self.assertEqual((a / 'score.json').read_bytes(), (b / 'score.json').read_bytes())
            generated = from_midi(a / 'score.mid').analysis_document()
            self.assertEqual([(t['name'], t['program'], t['note_count']) for t in generated['tracks']],
                             [('bass', 33, 4), ('comping', 4, 12), ('drums', 0, 8), ('lead', 81, 4)])
            self.assertEqual(generated['duration_sec'], 2)
            write_source('fixture-12', root / 'fixture')
            for name, source in [('fixture', root / 'fixture/source.mid'), ('generated', a / 'score.mid')]:
                with self.subTest(source=name):
                    original = source.read_bytes()
                    context = from_midi(source)
                    self.assertEqual(required_tuning(context.events, context.analysis_document()), 'direct_12')
                    out = root / (name + '-preview')
                    result = _run_cli('render-midi', str(source), '--profile', 'preview', '--out', str(out),
                                      '--no-validate-only')
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    index = json.loads((out / 'stems/role_index.json').read_bytes())
                    self.assertEqual(len(index['stems']), 4)
                    projected = context.events.isolate_tracks()
                    for row in index['stems']:
                        stem = MidiEvents.load(out / 'stems' / row['midi'])
                        original_stem = projected[row['source_track_index']]
                        self.assertEqual(pitches(stem), pitches(original_stem))
                        self.assertEqual(stem.tempo_map.changes, original_stem.tempo_map.changes)
                        self.assertEqual(len(stem.notes), row['note_count'])
                    frames = []
                    for path in [out / 'stems' / row['wav'] for row in index['stems']] + [out / 'stems/rough_mix.wav']:
                        pcm, rate = sf.read(path)
                        frames.append(len(pcm))
                        self.assertEqual(rate, 48000)
                        self.assertTrue(np.isfinite(pcm).all())
                        active = pcm[int(rate * .1):int(rate * .4)]
                        self.assertGreater(float(np.sqrt(np.mean(active ** 2))), 1e-5)
                        self.assertLessEqual(float(np.max(np.abs(pcm))), 0.950001)
                    self.assertEqual(set(frames), {780000 if name == 'fixture' else 108000})
                    report = json.loads((out / 'render.json').read_bytes())
                    self.assertEqual(report['effective_config']['render']['backend'], 'simple')
                    for renderer in report['renderers']:
                        self.assertEqual(renderer['requested_backend'], 'simple')
                        self.assertEqual(renderer['effective_backend'], 'simple')
                        self.assertIsNone(renderer['fallback_reason'])
                        self.assertIsNone(renderer['soundfont_sha256'])
                        self.assertTrue(renderer['backend']['code_digest'])
                    self.assertEqual((out / 'source.mid').read_bytes(), original)
                    self.assertEqual(source.read_bytes(), original)
                    self.assertTrue(list(out.glob('*.expressive.mid')))

    def test_19edo_preview_and_mts_fail_without_publishing_success(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {'AIMUSIC_AUDIO_BACKEND': 'native'}):
            os.environ.pop('AIMUSIC_SOUNDFONT', None)
            root = Path(tmp)
            for name in ('reference-19', 'multitrack-19'):
                write_source(name, root / name)
                path = root / name / 'source.mid'
                original = path.read_bytes()
                out = root / (name + '-preview')
                result = _run_cli('render-midi', str(path), '--profile', 'preview', '--out', str(out), '--no-validate-only')
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('channel_pitch', result.stderr)
                self.assertFalse((out / 'render.json').exists())
                self.assertFalse(list(out.rglob('*.wav')))
                self.assertEqual(path.read_bytes(), original)
            midi = mido.MidiFile(path)
            midi.tracks[0].insert(0, mido.Message('sysex', data=(0x7e, 127, 8, 2)))
            midi.save(root / 'mts.mid')
            self.assertEqual(required_tuning(MidiEvents.from_mido(midi), {}), 'mts')
            result = _run_cli('render-midi', str(root / 'mts.mid'), '--profile', 'preview',
                              '--out', str(root / 'mts-out'), '--no-validate-only')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('mts', result.stderr)
            self.assertFalse(list((root / 'mts-out').rglob('*.wav')))
