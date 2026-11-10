"""Opt-in real licensed-asset tests. Missing prerequisites FAIL after opting in."""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import tempfile
import time
import unittest

import numpy as np
import soundfile as sf

from aimusic.audio.config import load_audio_config
from aimusic.audio.demo_sources import write_source
from aimusic.audio.from_midi import from_midi
from aimusic.audio.midi_events import MidiEvents
from aimusic.audio.orchestrator import Orchestrator
from aimusic.audio.render.registry import file_digest
from tests.audio_native_support import load_native_manifest, verify_native_runtime


@unittest.skipUnless(os.environ.get('AIMUSIC_RUN_FLUIDSYNTH_TESTS') == '1',
                     'opt in with AIMUSIC_RUN_FLUIDSYNTH_TESTS=1; see docs/audio-testing.md')
class TestFluidSynthIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = load_native_manifest()
        cls.runtime = verify_native_runtime(cls.manifest)
        parent = os.environ.get('AIMUSIC_FLUIDSYNTH_ARTIFACTS')
        cls.root = Path(tempfile.mkdtemp(prefix='fluidsynth-evidence-', dir=parent))
        print(f'Native evidence (local only): {cls.root}', flush=True)
        (cls.root / 'asset.json').write_text(json.dumps(cls.manifest, indent=2))
        cls.config = load_audio_config('listening').to_dict()
        cls.config.setdefault('paths', {})['soundfont'] = cls.manifest['soundfont_path']
        cls.config['groove'] = {'spec': {}}

    def test_eight_bar_real_instruments_cold_repeat_and_warm_cache(self):
        source = write_source('fixture-12', self.root / 'fixture-12')
        original = source.read_bytes()
        context = from_midi(source)
        expected = [('drums', [9], 0, 96), ('bass', [0], 33, 16),
                    ('comping', [1], 0, 48), ('lead', [2], 80, 64)]
        self.assertEqual([(t['role'], t['channels'], t['program'], t['note_count'])
                          for t in context.analysis_document()['tracks']], expected)
        evidence = {'runtime': self.runtime, 'platform': platform.platform(), 'python': platform.python_version(),
                    'source_sha256': file_digest(source), 'runs': {}, 'signals': {}}
        hashes = []
        for name, cache in [('cold', 'cache-a'), ('cold-repeat', 'cache-b'), ('warm', 'cache-a')]:
            out = self.root / name
            start = time.perf_counter()
            runner = Orchestrator(self.config, self.root / cache)
            runner.run(context, output_dir=out)
            elapsed = time.perf_counter() - start
            evidence['runs'][name] = {'elapsed_sec': elapsed, 'hits': runner.cache_hits, 'misses': runner.cache_misses}
            hashes.append({p.name: file_digest(p) for p in (out / 'stems').glob('*.wav')})
            self.assertEqual(runner.cache_hits.get('render', 0), int(name == 'warm'))
            self.assertEqual(runner.cache_misses.get('render', 0), int(name != 'warm'))
            manifest = json.loads((out / 'render.json').read_text())
            self.assertEqual(manifest['duration_sec'], 16)
            self.assertEqual(manifest['release_tail_sec'], .25)
            for renderer in manifest['renderers']:
                self.assertEqual(renderer['requested_backend'], 'fluidsynth')
                self.assertEqual(renderer['effective_backend'], 'fluidsynth')
                self.assertIsNone(renderer['fallback_reason'])
                self.assertEqual(renderer['soundfont_sha256'], self.manifest['soundfont_sha256'])
                self.assertEqual(dict(renderer['backend']['native_versions']), self.runtime)
            index = json.loads((out / 'stems/role_index.json').read_text())
            self.assertEqual([(r['role'], r['channels'], r['note_count']) for r in index['stems']],
                             [(r, c, n) for r, c, p, n in expected])
            waves = []
            for row, (_, channels, program, count) in zip(index['stems'], expected):
                stem = MidiEvents.load(out / 'stems' / row['midi'])
                self.assertEqual(len(stem.notes), count)
                self.assertEqual({n.channel for n in stem.notes}, set(channels))
                self.assertEqual({e.message.program for e in stem.events if e.message.type == 'program_change'}, {program})
                self.assertEqual([(n.start, n.end, n.pitch, n.velocity) for n in stem.notes],
                                 [(n.start, n.end, n.pitch, n.velocity) for n in context.events.notes
                                  if n.track == row['source_track_index']])
                pcm, rate = sf.read(out / 'stems' / row['wav'], always_2d=True)
                waves.append(pcm.mean(axis=1))
                self.check_signal(pcm, rate)
                evidence['signals'][row['role']] = self.metrics(pcm)
                sidecar = json.loads((out / 'stems' / (row['wav'] + '.json')).read_text())
                self.assertEqual(sidecar['frames'], 780000)
                self.assertEqual(sidecar['soundfont_sha256'], self.manifest['soundfont_sha256'])
            mix, rate = sf.read(out / 'stems/rough_mix.wav')
            self.check_signal(mix, rate)
            evidence['signals']['rough_mix'] = self.metrics(mix)
            total = sum(waves)
            peak = float(np.max(np.abs(total)))
            total *= min(1, .95 / peak)
            # Exact FLOAT result from source stems only, never click.
            np.testing.assert_array_equal(mix, total.astype(np.float32))
            click, rate = sf.read(out / 'stems/click.wav')
            self.assertEqual((len(click), rate), (780000, 48000))
            self.assertGreater(float(np.max(np.abs(click))), 0)
        evidence['wav_hashes'] = hashes
        evidence['cold_repeat_equal'] = hashes[0] == hashes[1]
        evidence['warm_equal'] = hashes[0] == hashes[2]
        (self.root / 'fixture-12-report.json').write_text(json.dumps(evidence, indent=2))
        self.assertEqual(source.read_bytes(), original)
        self.assertEqual(hashes[0], hashes[1], 'cold-repeat native bytes differ; do not claim determinism')
        self.assertEqual(hashes[0], hashes[2], 'warm-cache bytes differ')
        for run in evidence['runs'].values():
            self.assertLessEqual(run['elapsed_sec'], 60, 'eight-bar ~60s target exceeded; retain measured evidence')

    def check_signal(self, pcm, rate):
        self.assertEqual((len(pcm), rate), (780000, 48000))
        self.assertTrue(np.isfinite(pcm).all())
        metrics = self.metrics(pcm)
        self.assertGreater(metrics['active_rms'], 1e-5)
        self.assertLessEqual(metrics['peak'], .95)

    @staticmethod
    def metrics(pcm):
        return {'active_window_sec': [.1, .4], 'active_rms': float(np.sqrt(np.mean(pcm[4800:19200] ** 2))),
                'peak': float(np.max(np.abs(pcm)))}

    def test_raw_native_channel_pitch_proof_before_capability_promotion(self):
        """Public adapter probe, not a registry capability bypass in production."""
        import io
        import mido
        from aimusic.audio.demo_sources import build_source
        from aimusic.audio.render import fluidsynth_r
        from aimusic.audio.render.registry import RenderRequest
        from aimusic.audio.scoring.tuning_check import ToneWindow, analyze_tuning
        from tests.test_audio_stems import _event_track
        payload, intent = build_source('reference-19')
        source = self.root / 'raw-reference.mid'
        source.write_bytes(payload)
        request = RenderRequest(source, self.root / 'raw-reference.wav', {}, 'channel_pitch',
                                48000, 4.5, .25, Path(self.manifest['soundfont_path']), {})
        fluidsynth_r.render(request)
        pcm, rate = sf.read(request.output_path, always_2d=True)
        report = analyze_tuning(pcm.mean(axis=1), rate, tuple(ToneWindow(**w) for w in intent['windows']))
        (self.root / 'raw-reference-tuning.json').write_text(json.dumps(report.to_dict(), indent=2))
        report.require_pass()

        # Two independently bent voices in ONE Synth at the SAME time. Diagnostic
        # hard-pan CC10 separates mono windows; do not claim polyphonic f0 scoring.
        mono = mido.MidiFile(file=io.BytesIO(payload))
        events = []
        targets = [intent['notes'][i] for i in (0, 2)]
        wheels = [msg.pitch for msg in mono.tracks[1] if msg.type == 'pitchwheel']
        for channel, target, bend, pan in zip((1, 2), targets, (wheels[0], wheels[2]), (0, 127)):
            events.append((0, mido.Message('program_change', channel=channel, program=80)))
            for control, value in ((101, 0), (100, 0), (6, 2), (38, 0), (101, 127), (100, 127), (10, pan)):
                events.append((0, mido.Message('control_change', channel=channel, control=control, value=value)))
            events += [(0, mido.Message('pitchwheel', channel=channel, pitch=bend)),
                       (0, mido.Message('note_on', channel=channel, note=target['midi_note'], velocity=96)),
                       (960, mido.Message('note_off', channel=channel, note=target['midi_note']))]
        midi = mido.MidiFile(type=0, ticks_per_beat=480)
        midi.tracks = [_event_track(*sorted(events, key=lambda e: e[0]))]
        midi.save(self.root / 'simultaneous.mid')
        request = RenderRequest(self.root / 'simultaneous.mid', self.root / 'simultaneous.wav', {}, 'channel_pitch',
                                48000, 1, .25, Path(self.manifest['soundfont_path']), {})
        from aimusic.audio.render import registry
        selected = registry.resolve_renderer('lead', config=self.config, soundfont=request.soundfont, tuning='channel_pitch')
        registry.render_request(request, selected)
        pcm, rate = sf.read(request.output_path, always_2d=True)
        reports = []
        for column, target in enumerate(targets):
            report = analyze_tuning(pcm[:, column], rate, (
                ToneWindow(f'channel-{column + 1}', 12000, 36000, target['expected_hz']),))
            reports.append(report.to_dict())
        (self.root / 'simultaneous-tuning.json').write_text(json.dumps(reports, indent=2))
        for row in reports:
            self.assertTrue(row['passed'], row)

    def test_production_19edo_original_groove_and_flattened_same_targets(self):
        from copy import deepcopy
        import mido
        from aimusic.audio.scoring.tuning_check import ToneWindow, TuningCheckError, analyze_tuning
        source = write_source('reference-19', self.root / 'reference-19')
        expected_path = source.parent / 'expected.json'
        intent = json.loads(expected_path.read_text())
        summary = {'expected_sha256': file_digest(expected_path), 'reports': {}}
        dry_hash = None
        for name, offset, flat in [('original', 0, False), ('groove', 600, False),
                                   ('flat', 0, True), ('warm-reference', 0, False)]:
            config = deepcopy(self.config)
            config['stages']['expressivization'] = bool(offset)
            config['groove'] = {'spec': {'microtiming_ms': {'lead': {'global_offset': 12.5}}}}
            path = source
            if flat:
                midi = mido.MidiFile(source)
                for track in midi.tracks:
                    for msg in track:
                        if msg.type == 'pitchwheel':
                            msg.pitch = 0
                path = self.root / 'flattened.mid'
                midi.save(path)
            out = self.root / name
            runner = Orchestrator(config, self.root / 'reference-cache')
            runner.run(from_midi(path), output_dir=out)
            if offset:
                diff = json.loads((out / 'source.expressive.mid.diff.json').read_text())
                self.assertEqual([(r['role'], r['offset_ticks']) for r in diff['note_diffs']], [('lead', 12)] * 3)
            manifest = json.loads((out / 'render.json').read_text())
            cache_manifest = json.loads((out / 'stems/cache_manifest.json').read_text())
            self.assertEqual(cache_manifest['fingerprint'], manifest)
            choice = manifest['renderers'][0]
            self.assertEqual(choice['effective_backend'], 'fluidsynth')
            self.assertIsNone(choice['fallback_reason'])
            self.assertEqual(choice['soundfont_sha256'], self.manifest['soundfont_sha256'])
            self.assertEqual(dict(choice['backend']['native_versions']), self.runtime)
            index = json.loads((out / 'stems/role_index.json').read_text())
            self.assertEqual(len(index['stems']), 1)
            wav = out / 'stems' / index['stems'][0]['wav']
            pcm, rate = sf.read(wav, always_2d=True)
            self.assertEqual((len(pcm), rate), (228000, 48000))
            self.assertTrue(np.isfinite(pcm).all())
            self.assertLessEqual(float(np.max(np.abs(pcm))), .95)
            windows = tuple(ToneWindow(**{**w, 'start_sample': w['start_sample'] + offset,
                                         'stop_sample': w['stop_sample'] + offset}) for w in intent['windows'])
            report = analyze_tuning(pcm.mean(axis=1), rate, windows)
            (out / 'tuning-check.json').write_text(json.dumps(report.to_dict(), indent=2))
            summary['reports'][name] = report.to_dict()
            if flat:
                with self.assertRaises(TuningCheckError):
                    report.require_pass()
                # Aggregate floors must fail too, not only the five-cent gate.
                general = analyze_tuning(pcm.mean(axis=1), rate, windows, controlled=False)
                with self.assertRaises(TuningCheckError):
                    general.require_pass()
                (out / 'tuning-check-general.json').write_text(json.dumps(general.to_dict(), indent=2))
            else:
                report.require_pass()
                self.assertEqual(choice['required_tuning'], 'channel_pitch')
                evidence = choice['tuning_evidence']
                self.assertEqual([n['tick'] for n in evidence['notes']],
                                 [n['start_tick'] + offset // 50 for n in intent['notes']])
                sidecar = json.loads(Path(str(wav) + '.json').read_text())
                self.assertEqual(sidecar['required_tuning'], 'channel_pitch')
                self.assertEqual(sidecar['tuning']['event_evidence'], evidence)
                self.assertEqual(sidecar['tuning']['metadata'], manifest['analysis']['tuning'])
                self.assertTrue(sidecar['tuning']['metadata']['events'])
            if name == 'original':
                dry_hash = file_digest(wav)
            self.assertEqual(runner.cache_hits.get('render', 0), int(name == 'warm-reference'))
            if name == 'warm-reference':
                self.assertEqual(file_digest(wav), dry_hash)
        (self.root / 'reference-19-report.json').write_text(json.dumps(summary, indent=2))

    def test_mts_and_cross_track_ownership_never_publish_success(self):
        import io
        import mido
        from aimusic.audio.demo_sources import build_source
        payload, _ = build_source('reference-19')
        for name in ('mts', 'ambiguous'):
            midi = mido.MidiFile(file=io.BytesIO(payload))
            if name == 'mts':
                midi.tracks[0].insert(0, mido.Message('sysex', data=(0x7e, 127, 8, 2)))
            else:
                midi.tracks.append(midi.tracks[1].copy())
            source = self.root / (name + '.mid')
            midi.save(source)
            out = self.root / (name + '-rejected')
            runner = Orchestrator(self.config, self.root / (name + '-cache'))
            with self.assertRaisesRegex(ValueError, 'mts|channel ownership'):
                runner.run(from_midi(source), output_dir=out)
            self.assertFalse((out / 'render.json').exists())
            self.assertFalse(list(out.rglob('*.wav')))
            self.assertFalse(list((runner.cache_dir / 'render').rglob('cache_manifest.json')))

    def check_listening_run(self, out, source, frames, package, intent):
        """Check real output against original intent, not its claimed result flags."""
        import math
        from tests.test_audio_demo_sources import pitches
        from aimusic.audio.scoring.tuning_check import ToneWindow, analyze_tuning
        report = json.loads((out / 'render.json').read_text())
        self.assertEqual(json.loads((out / 'stems/cache_manifest.json').read_text())['fingerprint'], report)
        self.assertEqual(report['application_backend'], {
            'requested_backend': 'native', 'effective_backend': 'native', 'fallback_reason': None})
        self.assertEqual(report['effective_config']['render']['backend'], 'fluidsynth')
        self.assertFalse(report['effective_config']['render']['allow_simple_fallback'])
        analysis = json.loads((out / f'{source.stem}.analysis.json').read_text())
        reconciliation = json.loads((out / f'{source.stem}.reconciliation.json').read_text())
        if package:
            self.assertEqual(analysis['provenance'], 'planner')
            self.assertTrue(reconciliation['passed'])
            self.assertTrue(reconciliation['checks'])
            self.assertEqual(report['source']['package_hash'], package.package.content_hash)
            self.assertTrue(all(t['role_source'] == 'structure.json' and t['count_source'] == 'score.json'
                                for t in analysis['tracks']))
        else:
            self.assertEqual(analysis['provenance'], 'inferred')
            self.assertEqual(reconciliation['status'], 'not_applicable')
            self.assertEqual((out / 'source.mid').read_bytes(), source.read_bytes())
        original = MidiEvents.load(source)
        expressive = MidiEvents.load(out / f'{source.stem}.expressive.mid')
        signature = lambda events: [(e.tick, e.track, e.message.dict()) for e in events.events]
        self.assertEqual(signature(expressive), signature(original))  # Explicit no-op groove for this set.
        index = json.loads((out / 'stems/role_index.json').read_text())['stems']
        isolated = original.isolate_tracks()
        self.assertEqual([r['source_track_index'] for r in index], list(isolated))
        signals, waves, errors = {}, [], []
        for row, choice in zip(index, report['renderers'], strict=True):
            stem = MidiEvents.load(out / 'stems' / row['midi'])
            self.assertEqual(signature(stem), signature(isolated[row['source_track_index']]))
            self.assertEqual(len(stem.notes), row['note_count'])
            self.assertEqual(choice['effective_backend'], 'fluidsynth')
            self.assertEqual(choice['requested_backend'], 'fluidsynth')
            self.assertIsNone(choice['fallback_reason'])
            self.assertEqual(choice['soundfont_sha256'], self.manifest['soundfont_sha256'])
            self.assertEqual(dict(choice['backend']['native_versions']), self.runtime)
            expected_mode = 'channel_pitch' if intent and intent['edo'] == 19 else 'direct_12'
            self.assertEqual(choice['required_tuning'], expected_mode)
            if expected_mode == 'channel_pitch':
                expected = [n for n in intent['notes'] if n['track'] == row['source_track_index']]
                for actual, target in zip(pitches(stem), expected, strict=True):
                    self.assertEqual(actual[:3], (target['track'], target['channel'], target['midi_note']))
                    hz = 440 * 2 ** (target['edo_step_from_a4'] / 19)
                    self.assertEqual(target['expected_hz'], hz)
                    errors.append(abs(1200 * math.log2(actual[3] / hz)))
                self.assertLessEqual(max(errors), 1)
                self.assertTrue(choice['tuning_evidence']['notes'])
            wav = out / 'stems' / row['wav']
            sidecar = json.loads(Path(str(wav) + '.json').read_text())
            self.assertEqual(sidecar['frames'], frames)
            self.assertEqual(sidecar['output_sha256'], file_digest(wav))
            self.assertEqual(sidecar['required_tuning'], expected_mode)
            if expected_mode == 'channel_pitch':
                self.assertEqual(sidecar['tuning']['event_evidence'], choice['tuning_evidence'])
        for row in index + [{'wav': 'rough_mix.wav'}]:
            pcm, rate = sf.read(out / 'stems' / row['wav'], always_2d=True)
            self.assertEqual((len(pcm), rate), (frames, 48000))
            self.assertTrue(np.isfinite(pcm).all())
            metrics = self.metrics(pcm)
            self.assertGreater(metrics['active_rms'], 1e-5)
            self.assertLessEqual(metrics['peak'], .95)
            signals[row['wav']] = dict(metrics, frames=frames, sample_rate=rate, finite=True)
            waves.append(pcm.mean(axis=1))
        total = sum(waves[:-1])
        total *= min(1, .95 / float(np.max(np.abs(total))))
        np.testing.assert_array_equal(waves[-1], total.astype(np.float32))
        click, rate = sf.read(out / 'stems/click.wav')
        self.assertEqual((len(click), rate), (frames, 48000))
        self.assertTrue(np.isfinite(click).all())
        self.assertGreater(float(np.max(np.abs(click))), 0)
        self.assertLessEqual(float(np.max(np.abs(click))), .95)
        tuning = None
        if intent and intent['windows']:
            pcm, rate = sf.read(out / 'stems' / index[0]['wav'], always_2d=True)
            measured = analyze_tuning(pcm.mean(axis=1), rate, tuple(ToneWindow(**w) for w in intent['windows']))
            measured.require_pass()
            tuning = measured.to_dict()
            (out / 'tuning-check.json').write_text(json.dumps(tuning, indent=2, allow_nan=False))
        return {'signals': signals, 'click_excluded_exactly': True, 'event_payloads_equal': True,
                'event_pitch_max_abs_cents': max(errors) if errors else None,
                'polyphonic_f0': 'unmeasured', 'controlled_tuning': tuning,
                'reconciliation': reconciliation, 'renderers': report['renderers']}

    def test_four_native_listening_workflows(self):
        import subprocess
        import sys
        from aimusic.audio.from_score import from_score
        root = self.root / 'four-workflows'
        root.mkdir()
        commands = []

        def command(*args):
            argv = [sys.executable, '-m', *map(str, args)]
            result = subprocess.run(argv, text=True, capture_output=True, check=True)
            commands.append(argv)
            (root / f'command-{len(commands)}.log').write_text(result.stdout + result.stderr)

        command('aimusic.app.cli', 'generate', '--seed', 11, '--beats', 4, '--edo', 12,
                '--tempo-bpm', 120, '--meter', '4/4', '--out', root / 'generated-12')
        packages = list((root / 'generated-12').glob('run_*'))
        self.assertEqual(len(packages), 1)
        package = from_score(packages[0])
        evidence = {'schema': 'aimusic.native-listening-evidence/1', 'runtime': self.runtime,
                    'python': platform.python_version(), 'platform': platform.platform(),
                    'commands': commands, 'workflows': {}, 'human_listening': 'pending'}
        for name, frames in [('fixture-12', 780000), ('generated-12', 108000),
                             ('reference-19', 228000), ('multitrack-19', 204000)]:
            generated = name == 'generated-12'
            if not generated:
                command('aimusic.audio.demo_sources', name, '--seed', 11, '--out', root / name)
            source = package.package.midi_path if generated else root / name / 'source.mid'
            before = {p.relative_to(source.parent).as_posix(): file_digest(p) for p in source.parent.iterdir() if p.is_file()}
            intent = None if generated else json.loads((source.parent / 'expected.json').read_text())
            context = package if generated else from_midi(source)
            rows, hashes = {}, []
            for variant, cache in [('cold', 'cache-a'), ('cold-repeat', 'cache-b'), ('warm', 'cache-a')]:
                out = root / name / variant
                runner = Orchestrator(self.config, root / name / cache)
                start = time.perf_counter()
                runner.run(context, output_dir=out)
                elapsed = time.perf_counter() - start
                self.assertEqual(runner.cache_hits, {'expressivization': 1, 'render': 1} if variant == 'warm' else {})
                self.assertEqual(runner.cache_misses, {} if variant == 'warm' else {'expressivization': 1, 'render': 1})
                if name == 'fixture-12':
                    self.assertLessEqual(elapsed, 60)
                checked = self.check_listening_run(out, source, frames, package if generated else None, intent)
                digests = {p.relative_to(out).as_posix(): file_digest(p) for p in out.rglob('*') if p.is_file()}
                hashes.append(digests)
                rows[variant] = dict(checked, elapsed_sec=elapsed, hits=runner.cache_hits,
                                     misses=runner.cache_misses, artifact_sha256=digests)
            # Cold expression metadata intentionally records a new wall clock.
            # Keep both byte hashes; compare every other byte and every other field.
            sidecar = f'{source.stem}.expressive.mid.json'
            self.assertEqual({k: v for k, v in hashes[0].items() if k != sidecar},
                             {k: v for k, v in hashes[1].items() if k != sidecar})
            meta = [json.loads((root / name / v / sidecar).read_text()) for v in ('cold', 'cold-repeat')]
            for row in meta:
                self.assertIsInstance(row.pop('wall_clock_utc'), str)
            self.assertEqual(meta[0], meta[1])
            self.assertEqual(hashes[0], hashes[2], 'warm artifact bytes differ, including provenance')
            self.assertEqual(before, {p.relative_to(source.parent).as_posix(): file_digest(p)
                                     for p in source.parent.iterdir() if p.is_file()})
            evidence['workflows'][name] = {
                'source': source.relative_to(root).as_posix(), 'source_files_sha256': before,
                'seed': 11 if name in ('fixture-12', 'generated-12') else None,
                'runs': rows, 'cold_repeat_wav_equal': True, 'warm_all_artifacts_equal': True,
                'cold_repeat_metadata_exception': f'{sidecar}: wall_clock_utc only'}
        # The set is not validated unless unsupported tuning still fails closed.
        import mido
        midi = mido.MidiFile(root / 'reference-19/source.mid')
        midi.tracks[0].insert(0, mido.Message('sysex', data=(0x7e, 127, 8, 2)))
        midi.save(root / 'unsupported-mts.mid')
        runner = Orchestrator(self.config, root / 'mts-cache')
        with self.assertRaisesRegex(ValueError, 'mts') as failure:
            runner.run(from_midi(root / 'unsupported-mts.mid'), output_dir=root / 'mts-rejected')
        self.assertFalse(list((root / 'mts-rejected').rglob('*.*')))
        self.assertFalse(list((root / 'mts-cache').rglob('cache_manifest.json')))
        evidence['mts'] = {'rejected': True, 'successful_outputs': [], 'diagnostic': str(failure.exception)}
        self.assertEqual(len(evidence['workflows']), 4)
        from aimusic.audio.listening_review import write_listening_review
        from tests.test_audio_listening_review import parse_review
        import base64
        outputs = {name: root / name / 'cold' for name in evidence['workflows']}
        page = write_listening_review(outputs, self.root / 'listening-review-embedded.html')
        players = [attrs for tag, attrs in parse_review(page).tags if tag == 'audio']
        self.assertEqual(len(players), 4)
        for attrs, out in zip(players, outputs.values()):
            self.assertEqual(base64.b64decode(attrs['src'].split(',', 1)[1], validate=True),
                             (out / 'stems/rough_mix.wav').read_bytes())
        evidence['listening_review'] = {'path_from_evidence_root': page.name, 'sha256': file_digest(page),
                                        'players': 4, 'embedded_wavs_byte_identical': True}
        evidence['checks_passed'] = True  # Published only after every assertion above passes.
        (root / 'evidence.json').write_text(json.dumps(evidence, indent=2, allow_nan=False))
