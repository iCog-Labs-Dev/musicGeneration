"""Fixed candidate matrix; synthetic renderer evidence is never native evidence."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from aimusic.audio.midi_events import MidiEvents
from aimusic.audio.render import registry
from aimusic.audio.render.registry import RendererBackend, RendererCapabilities
from aimusic.audio.render.tuning import channel_pitch_evidence
from aimusic.audio.render.util import render_frames
from tests.test_audio_entry import _generate_package


def synthetic_renderer(request):
    events = MidiEvents.load(request.midi_path)
    if request.required_tuning == 'channel_pitch':
        frequencies = [row['hz'] for row in channel_pitch_evidence(events)['notes']]
    else:
        frequencies = [440 * 2 ** ((note.pitch - 69) / 12) for note in events.notes]
    pcm = np.zeros(render_frames(request.sample_rate, request.duration_sec, request.release_tail_sec))
    for note, hz in zip(events.notes, frequencies):
        a = round(float(events.tempo_map.seconds(note.start)) * request.sample_rate)
        b = round(float(events.tempo_map.seconds(note.end)) * request.sample_rate)
        pcm[a:b] += .08 * (note.velocity / 127) * np.sin(2 * np.pi * hz * np.arange(b - a) / request.sample_rate)
    sf.write(request.output_path, pcm, request.sample_rate, subtype='FLOAT')
    return request.output_path


BACKEND = RendererBackend('fluidsynth', 'synthetic-test-only',
                          RendererCapabilities(('direct_12', 'channel_pitch')),
                          'a' * 64, (('libfluidsynth', 'synthetic-test-only'),), synthetic_renderer)


class TestReviewMatrix(unittest.TestCase):
    def test_exact_manifest_has_four_mixes_four_roles_three_pairs_and_unmeasured_polyphony(self):
        from aimusic.audio.review_matrix import generate_matrix

        with tempfile.TemporaryDirectory() as tmp, patch.dict(registry._BACKENDS, {'fluidsynth': lambda asset: BACKEND}):
            root = Path(tmp)
            package = _generate_package(root / 'generated')
            asset = root / 'synthetic-test-only.sf2'
            asset.write_bytes(b'not a SoundFont: unit test boundary double only')
            out = root / 'candidates'
            matrix = generate_matrix(out, package=package, soundfont=asset)
            self.assertEqual(json.loads((out / 'matrix.json').read_bytes()), matrix.to_dict())
            self.assertEqual(matrix.schema, 'aimusic.review-matrix/1')
            self.assertEqual(len(matrix.renders), 7)
            self.assertEqual(len(matrix.exposures), 14)
            self.assertEqual({panel.purpose: len(panel.stimuli) for panel in matrix.panels},
                             {'canonical': 4, 'roles': 4, 'parity': 2, 'timing': 2, 'tuning': 2})
            names = [e.condition for e in matrix.exposures]
            self.assertEqual(len(set(names)), 14)
            self.assertEqual(tuple(matrix.panels[0].stimuli),
                             ('fixture-12', 'generated-12', 'reference-19', 'multitrack-19'))
            self.assertEqual(tuple(matrix.panels[1].stimuli),
                             ('fixture-drums', 'fixture-bass', 'fixture-comping', 'fixture-lead'))
            self.assertEqual(Counter(s for panel in matrix.panels for s in panel.stimuli), Counter(names))
            renders = {r.sha256: r for r in matrix.renders}
            sources = {s.sha256: s for s in matrix.sources}
            exposures = {e.condition: e for e in matrix.exposures}
            fixture = renders[exposures['fixture-12'].render_sha256]
            index = json.loads((out / Path(exposures['fixture-12'].wav.path).parent / 'role_index.json').read_bytes())
            for row in index['stems']:
                role = exposures['fixture-' + row['role']]
                self.assertEqual(role.render_sha256, fixture.sha256)
                self.assertEqual(Path(role.wav.path).name, row['wav'])
            for exposure in matrix.exposures:
                exposure.wav.read_verified(out)
                render = renders[exposure.render_sha256]
                render.require_admitted()
                self.assertIn(exposure.wav, render.reports)
                self.assertEqual(next(m.state for m in render.metrics if m.name == 'polyphonic_f0'), 'unmeasured')
            poly = renders[exposures['multitrack-19'].render_sha256]
            source = sources[poly.source_sha256]
            notes = MidiEvents.from_bytes(next(f for f in source.files if f.path.endswith('.mid')).read_verified(out)).notes
            self.assertGreater(sum(n.start == 0 for n in notes), 1)
            self.assertEqual(len({n.track for n in notes}), 3)
            self.assertTrue(all(r['required_tuning'] == 'channel_pitch' for r in poly.renderers))
            self.assertEqual(next(m.state for m in poly.metrics if m.name == 'controlled_tuning'), 'not_applicable')
            self.assertFalse(list(out.rglob('*.html')))  # Not a completed/public review bundle.

    def test_pairs_preserve_inputs_isolate_timing_and_measure_unchanged_tuning_targets(self):
        from aimusic.audio.review_matrix import generate_matrix, ReviewMatrix

        with tempfile.TemporaryDirectory() as tmp, patch.dict(registry._BACKENDS, {'fluidsynth': lambda asset: BACKEND}):
            root = Path(tmp)
            package = _generate_package(root / 'generated')
            original_package = {p.name: p.read_bytes() for p in package.iterdir() if p.is_file()}
            asset = root / 'synthetic-test-only.sf2'
            asset.write_bytes(b'not a SoundFont: unit test boundary double only')
            out = root / 'candidates'
            matrix = generate_matrix(out, package=package, soundfont=asset)
            self.assertEqual(ReviewMatrix.from_bytes(matrix.to_bytes()), matrix)
            exposures = {e.condition: e for e in matrix.exposures}
            renders = {r.sha256: r for r in matrix.renders}
            sources = {s.sha256: s for s in matrix.sources}

            def render(condition):
                return renders[exposures[condition].render_sha256]

            package_run, imported = render('package-12'), render('imported-12')
            self.assertEqual(sources[package_run.source_sha256].kind, 'render_package')
            self.assertEqual(sources[imported.source_sha256].kind, 'imported_midi')
            self.assertEqual(sources[package_run.source_sha256].identity['midi_sha256'],
                             sources[imported.source_sha256].identity['midi_sha256'])
            self.assertEqual(package_run.effective_config, imported.effective_config)
            # Synthetic deterministic backend proves route parity, not cross-platform native PCM equality.
            self.assertEqual(exposures['package-12'].wav.sha256, exposures['imported-12'].wav.sha256)
            self.assertEqual(next(m.state for m in package_run.metrics if m.name == 'reconciliation'), 'passed')
            self.assertEqual(next(m.state for m in imported.metrics if m.name == 'reconciliation'), 'not_applicable')

            off, on = render('groove-off'), render('groove-on')
            self.assertEqual(off.source_sha256, on.source_sha256)
            self.assertEqual(off.renderers, on.renderers)
            on_config = on.to_dict()['effective_config']
            self.assertTrue(on_config['stages']['expressivization'])
            on_config['stages']['expressivization'] = False
            self.assertEqual(on_config, off.to_dict()['effective_config'])
            original = MidiEvents.load(out / 'sources/fixture-12/source.mid')
            swung = MidiEvents.load(out / 'renders/groove-on/source.expressive.mid')
            self.assertEqual(len(original.notes), len(swung.notes))
            shifted = 0
            for before, after in zip(original.notes, swung.notes):
                self.assertEqual((before.track, before.channel, before.pitch, before.velocity, before.end - before.start),
                                 (after.track, after.channel, after.pitch, after.velocity, after.end - after.start))
                if before.start != after.start:
                    self.assertEqual(before.channel, 2)
                    self.assertEqual(after.start - before.start, 48)  # 50ms offbeat swing, not a global shift.
                    shifted += 1
            self.assertEqual(shifted, 32)
            self.assertNotEqual(exposures['groove-off'].wav.sha256, exposures['groove-on'].wav.sha256)

            good, bad = render('valid-19'), render('flattened-12-calibration-control')
            good_gate = next(m for m in good.metrics if m.name == 'controlled_tuning')
            bad_gate = next(m for m in bad.metrics if m.name == 'controlled_tuning')
            self.assertEqual((good_gate.state, bad_gate.state), ('passed', 'failed'))
            self.assertEqual(good_gate.evidence[0], bad_gate.evidence[0])
            self.assertEqual(good_gate.observations['coverage'], 1)
            self.assertEqual(bad_gate.observations['coverage'], 1)
            self.assertGreater(bad_gate.observations['p95_abs_cents'], 15)
            self.assertEqual(next(m.state for m in bad.metrics if m.name == 'negative_control'), 'passed')
            original = MidiEvents.load(out / 'sources/reference-19/source.mid')
            flat = MidiEvents.load(out / 'sources/flattened-12.mid')
            self.assertEqual(original.notes, flat.notes)
            for before, after in zip(original.events, flat.events):
                self.assertEqual((before.track, before.tick), (after.track, after.tick))
                self.assertEqual(after.message, before.message.copy(pitch=0) if before.message.type == 'pitchwheel' else before.message)

            repeat = generate_matrix(root / 'repeat', package=package, soundfont=asset)
            self.assertEqual([(e.condition, e.wav.sha256) for e in repeat.exposures],
                             [(e.condition, e.wav.sha256) for e in matrix.exposures])
            # Keep authentic timestamped groove/package metadata, not synthetic whole-run hash equality.
            self.assertEqual(repeat.panels, matrix.panels)
            self.assertEqual({p.name: p.read_bytes() for p in package.iterdir() if p.is_file()}, original_package)

    def test_non_12edo_package_is_refused_before_rendering(self):
        from aimusic.audio.review_matrix import generate_matrix
        from tests.test_audio_entry import _run_cli

        with tempfile.TemporaryDirectory() as tmp, patch.dict(registry._BACKENDS, {'fluidsynth': lambda asset: BACKEND}):
            root = Path(tmp)
            result = _run_cli('generate', '--seed', '11', '--beats', '4', '--edo', '19', '--out', str(root / 'generated'))
            self.assertEqual(result.returncode, 0, result.stderr)
            package = next((root / 'generated').glob('run_*'))
            asset = root / 'synthetic-test-only.sf2'
            asset.write_bytes(b'not a SoundFont')
            out = root / 'candidates'
            with patch('aimusic.audio.review_matrix.Orchestrator.run') as run:
                with self.assertRaisesRegex(ValueError, '12-EDO'):
                    generate_matrix(out, package=package, soundfont=asset)
                run.assert_not_called()
            self.assertFalse((out / 'matrix.json').exists())

    def test_module_command_requires_explicit_assets_without_creating_outputs(self):
        import subprocess
        import sys

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'not-created'
            command = [sys.executable, '-m', 'aimusic.audio.review_matrix']
            help_result = subprocess.run([*command, '--help'], capture_output=True, text=True)
            self.assertEqual(help_result.returncode, 0, help_result.stderr)
            self.assertIn('--package', help_result.stdout)
            self.assertIn('--soundfont', help_result.stdout)
            result = subprocess.run([*command, '--package', str(Path(tmp) / 'package'),
                                     '--soundfont', str(Path(tmp) / 'missing.sf2'), '--out', str(out)],
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('SoundFont', result.stderr)
            self.assertNotIn('Traceback', result.stderr)
            self.assertFalse(out.exists())

    def test_failed_required_candidate_and_existing_output_never_report_complete(self):
        from aimusic.audio.review_matrix import generate_matrix
        from dataclasses import replace

        def fail_polyphony(request):
            if any(n.channel == 5 for n in MidiEvents.load(request.midi_path).notes):
                raise RuntimeError('synthetic polyphonic renderer failure')
            return synthetic_renderer(request)

        backend = replace(BACKEND, render=fail_polyphony)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(registry._BACKENDS, {'fluidsynth': lambda asset: backend}):
            root = Path(tmp)
            package = _generate_package(root / 'generated')
            asset = root / 'synthetic-test-only.sf2'
            asset.write_bytes(b'not a SoundFont')
            out = root / 'candidates'
            with self.assertRaisesRegex(ValueError, 'multitrack-19.*synthetic polyphonic renderer failure'):
                generate_matrix(out, package=package, soundfont=asset)
            self.assertFalse((out / 'matrix.json').exists())
            self.assertFalse(list(out.rglob('*.html')))
            # Retain diagnostics; do not retry into a partially populated directory.
            before = {p.relative_to(out): p.read_bytes() for p in out.rglob('*') if p.is_file()}
            with self.assertRaises(FileExistsError):
                generate_matrix(out, package=package, soundfont=asset)
            self.assertEqual(before, {p.relative_to(out): p.read_bytes() for p in out.rglob('*') if p.is_file()})
