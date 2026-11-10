"""Run-wide publication and consumed-input race regressions."""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests.test_audio_stems import _audio_render_available, _make_two_pianos_one_bass_midi


@unittest.skipUnless(_audio_render_available(), "requires pip install -e '.[audio]'")
class TestRunPublication(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = _make_two_pianos_one_bass_midi(self.root / 'source.mid')
        self.config = {'groove': {'spec': {}}, 'render': {'sample_rate': 8000}}

    def symlink_or_skip(self, link, target, *, directory=False):
        try:
            link.symlink_to(target, target_is_directory=directory)
        except OSError as exc:
            if getattr(exc, 'winerror', None) == 1314:
                self.skipTest('Native symlink privilege unavailable; file/hardlink collisions and native junction tests remain')
            raise

    def run_audio(self, out='out'):
        from aimusic.audio.orchestrator import Orchestrator
        return Orchestrator(self.config, self.root / 'cache').run(self.source, output_dir=self.root / out)

    def test_every_collision_is_rejected_before_any_run_artifact_is_written(self):
        names = ['source.analysis.json', 'source.analysis.json.json', 'source.expressive.mid',
                 'source.expressive.mid.diff.json', 'source.expressive.mid.json', 'render.json',
                 'stems/track_0001_piano.mid', 'stems/rough_mix.wav', 'stems/cache_manifest.json']
        for name in names:
            for kind in ('file', 'symlink', 'source-alias'):
                with self.subTest(name=name, kind=kind):
                    out = self.root / f'collision-{names.index(name)}-{kind}'
                    target = out / name
                    target.parent.mkdir(parents=True)
                    outside = self.root / 'outside'
                    outside.write_bytes(b'KEEP')
                    original = self.source.read_bytes()
                    if kind == 'file':
                        target.write_bytes(b'KEEP')
                    elif kind == 'symlink':
                        self.symlink_or_skip(target, outside)
                    else:
                        target.hardlink_to(self.source)
                    with self.assertRaises(FileExistsError):
                        self.run_audio(out.name)
                    self.assertEqual(outside.read_bytes(), b'KEEP')
                    self.assertEqual(self.source.read_bytes(), original)
                    self.assertEqual([p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file()], [name])

    def test_output_symlink_is_rejected_before_upstream_writes(self):
        outside = self.root / 'outside'
        outside.mkdir()
        self.symlink_or_skip(self.root / 'out', outside, directory=True)
        with self.assertRaises(ValueError):
            self.run_audio()
        self.assertEqual(list(outside.iterdir()), [])

    def test_upstream_restore_checks_copied_bytes_and_expected_fingerprint(self):
        from aimusic.audio.cache import cached_stage
        self.run_audio('cold')
        real_copy = shutil.copyfile
        for stage, name in (('analysis', 'analysis.json.json'), ('expressivization', 'expressive.mid.diff.json')):
            for change in ('corrupt', 'valid-substitution'):
                with self.subTest(stage=stage, change=change):
                    cache = next(p for p in (self.root / 'cache' / stage).iterdir() if not p.name.startswith('.'))
                    names = list(json.loads((cache / 'cache_manifest.json').read_text())['outputs'])

                    def produce(directory):
                        for artifact in names:
                            (directory / artifact).write_bytes((cache / artifact).read_bytes())
                    other, _ = cached_stage(self.root / 'other-cache', {'different': stage}, names, produce)

                    def copying(src, dst, *args, **kwargs):
                        src = Path(src)
                        if src.parent == cache:
                            if change == 'valid-substitution':
                                src = other / src.name
                            elif src.name == 'source.' + name:
                                Path(dst).write_bytes(b'CORRUPT')
                                return dst
                        return real_copy(src, dst, *args, **kwargs)

                    out = f'restore-{stage}-{change}'
                    with patch('shutil.copyfile', side_effect=copying), self.assertRaisesRegex(ValueError, 'cache changed'):
                        self.run_audio(out)
                    self.assertEqual([p for p in (self.root / out).rglob('*') if p.is_file()], [])

    def test_late_collision_rolls_back_reports_and_keeps_the_other_writer(self):
        import os
        self.run_audio('cold')
        out = self.root / 'out'
        if os.name == 'nt':
            from aimusic.audio.windows_native import WindowsAPI
            operation, real_link = 'aimusic.audio.windows_native.WindowsAPI.rename', WindowsAPI.rename
        else:
            operation, real_link = 'os.link', os.link
        collided = False

        def linking(*args, **kwargs):
            nonlocal collided
            if Path(args[-1]).name == 'rough_mix.wav' and (os.name == 'nt' or kwargs.get('dst_dir_fd') is not None):
                # Only race publication into the user tree, not private staging.
                if (out / 'source.analysis.json').exists():
                    collided = True
                    if os.name == 'nt':
                        (out / 'stems' / 'rough_mix.wav').write_bytes(b'OTHER WRITER')
                    else:
                        (out / 'stems' / 'rough_mix.wav').symlink_to(self.source)
            return real_link(*args, **kwargs)

        original = self.source.read_bytes()
        with patch(operation, new=linking), self.assertRaises(FileExistsError):
            self.run_audio()
        self.assertTrue(collided)
        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual([p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file()],
                         ['stems/rough_mix.wav'])

    def test_package_reports_refuse_collisions_before_writing_other_reports(self):
        from tests.test_audio_reconcile import TestAudioReconcile
        from aimusic.audio.orchestrator import Orchestrator
        fixture = TestAudioReconcile()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        for name in ('score.analysis.json', 'score.reconciliation.json'):
            with self.subTest(name=name):
                out = self.root / name
                out.mkdir()
                (out / name).hardlink_to(self.source)  # Mandatory source-identity collision without symlink privileges.
                original = self.source.read_bytes()
                with self.assertRaises(FileExistsError):
                    Orchestrator({'stages': {'render': False, 'expressivization': False}}, self.root / 'cache').run(
                        fixture.ctx, output_dir=out)
                self.assertEqual(self.source.read_bytes(), original)
                self.assertEqual([p.name for p in out.iterdir()], [name])


@unittest.skipUnless(_audio_render_available(), "requires pip install -e '.[audio]'")
class TestConsumedInputIdentity(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = _make_two_pianos_one_bass_midi(self.root / 'source.mid')
        self.config = {'stages': {'analysis': False, 'expressivization': False},
                       'render': {'sample_rate': 8000}}

    def run_audio(self, out='out'):
        from aimusic.audio.orchestrator import Orchestrator
        orch = Orchestrator(self.config, self.root / 'cache')
        return orch, orch.run(self.source, output_dir=self.root / out)

    @staticmethod
    def mutate_midi(path):
        import mido
        midi = mido.MidiFile(path)
        next(m for tr in midi.tracks for m in tr if m.type == 'note_on').velocity = 23
        midi.save(path)

    def assert_no_success(self, stage):
        self.assertEqual(list((self.root / 'cache' / stage).rglob('cache_manifest.json')), [])
        self.assertEqual([p for p in (self.root / 'out').rglob('*') if p.is_file()], [])

    def test_midi_change_between_plan_and_digest_is_rejected_before_cache_publication(self):
        from aimusic.audio.render.stems import prepare_stem_render

        def prepare(path, **kwargs):
            plan = prepare_stem_render(path, **kwargs)
            self.mutate_midi(path)
            return plan

        with patch('aimusic.audio.render.stems.prepare_stem_render', side_effect=prepare):
            with self.assertRaisesRegex(ValueError, 'changed|snapshot'):
                self.run_audio()
        self.assert_no_success('render')

    def test_original_midi_mutation_during_analysis_cannot_publish_wrong_provenance(self):
        from aimusic.audio.analysis.from_midi import analyze_midi
        self.config['stages'].update(analysis=True, render=False)

        def analyze(*args, **kwargs):
            result = analyze_midi(*args, **kwargs)
            self.mutate_midi(self.source)
            return result

        with patch('aimusic.audio.analysis.from_midi.analyze_midi', side_effect=analyze):
            with self.assertRaisesRegex(ValueError, 'changed|snapshot'):
                self.run_audio()
        self.assert_no_success('analysis')

    def test_soundfont_mutation_inside_backend_is_rejected_before_cache_publication(self):
        from dataclasses import replace
        from aimusic.audio.render import registry
        asset = self.root / 'asset.sf2'
        asset.write_bytes(b'original')
        self.config['paths'] = {'soundfont': str(asset)}
        simple = registry.resolve_renderer('bass', config={}).backend

        calls = 0

        def render(request):
            nonlocal calls
            calls += 1
            result = simple.render(request)
            if calls == 3:  # Last stem: a pre-check on the next stem cannot save us.
                asset.write_bytes(b'changed while executing')
            return result

        with patch.dict(registry._BACKENDS, {'simple': lambda asset: replace(simple, render=render)}):
            with self.assertRaisesRegex(ValueError, 'SoundFont.*changed'):
                self.run_audio()
        self.assert_no_success('render')

    def test_package_mutation_after_validation_is_rejected_before_cache_publication(self):
        from tests.test_audio_reconcile import TestAudioReconcile
        from aimusic.audio.from_score import validate_package
        from aimusic.audio.orchestrator import Orchestrator
        fixture = TestAudioReconcile()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)

        def validate(ctx):
            result = validate_package(ctx)
            tuning = fixture.ctx.package.tuning_path
            data = json.loads(tuning.read_text())
            data['pitch_bend_range'] = 37
            tuning.write_text(json.dumps(data))
            return result

        with patch('aimusic.audio.orchestrator.validate_package', side_effect=validate):
            with self.assertRaisesRegex(ValueError, 'changed|digest mismatch'):
                Orchestrator(self.config, self.root / 'cache').run(fixture.ctx, output_dir=self.root / 'out')
        self.assert_no_success('render')

    def test_all_published_sidecar_references_are_stable_artifact_names(self):
        self.config['stages'].update(analysis=True, expressivization=True)
        self.config['groove'] = {'spec': {}}
        _, out = self.run_audio()
        for path in out.rglob('*.json'):
            with self.subTest(path=path.name):
                text = path.read_text()
                for prefix in ('.pending-', '.audio-run-', '.stems-', '.restore-', '.inputs-'):
                    self.assertNotIn(prefix, text)
                doc = json.loads(text)
                if isinstance(doc, dict) and 'output_path' in doc:
                    output = doc['output_path']
                    self.assertEqual(Path(output).name, output)
                    self.assertTrue((path.parent / output).is_file())
                    diff = doc.get('metadata', {}).get('diff_report')
                    if diff:
                        self.assertEqual(Path(diff).name, diff)
                        self.assertTrue((path.parent / diff).is_file())
        from aimusic.audio.render.registry import file_digest
        report = json.loads((out / 'render.json').read_text())
        self.assertEqual(report['source']['midi_sha256'], file_digest(self.source))
        self.assertEqual(report['expressive_midi_sha256'], file_digest(out / 'source.expressive.mid'))
        for path in (out / 'stems').glob('*.wav.json'):
            doc = json.loads(path.read_text())
            self.assertEqual(doc['midi_sha256'], file_digest(path.with_suffix('').with_suffix('.mid')))

    def test_dependency_only_changes_invalidate_the_consuming_stage(self):
        from importlib import metadata
        self.config['stages'].update(analysis=True, expressivization=True)
        self.config['groove'] = {'spec': {}}
        self.run_audio('cold')
        for dependency, misses, hits in (
            ('mido', {'analysis', 'expressivization', 'render'}, set()),
            ('pydantic', {'expressivization'}, {'analysis', 'render'}),
            ('pydantic_core', {'expressivization'}, {'analysis', 'render'}),
            ('pretty_midi', {'analysis', 'render'}, {'expressivization'}),
        ):
            with self.subTest(dependency=dependency):
                real_version = metadata.version

                def version(name):
                    return 'changed-only-this-dependency' if name == dependency else real_version(name)

                with patch('importlib.metadata.version', side_effect=version):
                    orch, _ = self.run_audio(dependency)
                self.assertEqual(set(orch.cache_misses), misses)
                self.assertEqual(set(orch.cache_hits), hits)

    def test_analysis_consumes_snapshot_even_if_original_changes_and_is_restored(self):
        import hashlib
        from aimusic.audio.analysis.from_midi import analyze_midi
        self.config['stages'].update(analysis=True, render=False)
        original = self.source.read_bytes()

        def analyze(*args, **kwargs):
            self.mutate_midi(self.source)
            try:
                return analyze_midi(*args, **kwargs)
            finally:
                self.source.write_bytes(original)

        with patch('aimusic.audio.analysis.from_midi.analyze_midi', side_effect=analyze):
            _, out = self.run_audio()
        doc = json.loads((out / 'source.analysis.json').read_text())
        self.assertEqual(doc['inputs_hash'], hashlib.sha256(original).hexdigest())

    def test_asset_change_during_warm_restore_is_rechecked_without_publishing(self):
        asset = self.root / 'asset.sf2'
        asset.write_bytes(b'original')
        self.config['paths'] = {'soundfont': str(asset)}
        self.run_audio('cold')
        cache = next((self.root / 'cache' / 'render').iterdir())
        original_manifest = (cache / 'cache_manifest.json').read_bytes()
        real_copy = shutil.copyfile

        def copy(src, dst, **kwargs):
            if Path(src) == cache / 'render.json':
                asset.write_bytes(b'changed during warm copy')
            return real_copy(src, dst, **kwargs)

        with patch('shutil.copyfile', side_effect=copy), self.assertRaisesRegex(ValueError, 'SoundFont.*changed'):
            self.run_audio()
        self.assertEqual([p for p in (self.root / 'out').rglob('*') if p.is_file()], [])
        self.assertEqual((cache / 'cache_manifest.json').read_bytes(), original_manifest)

    def test_mixed_package_snapshot_is_rejected_before_cache_access(self):
        from tests.test_audio_reconcile import TestAudioReconcile
        from aimusic.audio.orchestrator import Orchestrator
        from aimusic.core.render_package import load_render_package
        fixture = TestAudioReconcile()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        original = fixture.ctx.package.structure_path.read_bytes()
        changed = False

        def load(root):
            nonlocal changed
            package = load_render_package(root)
            if not changed:
                changed = True
                doc = json.loads(package.structure_path.read_text())
                doc['tracks'][0]['role'] = 'comping'
                package.structure_path.write_text(json.dumps(doc))
            return package

        with patch('aimusic.audio.from_score.load_render_package', side_effect=load):
            with self.assertRaisesRegex(ValueError, 'changed|digest mismatch'):
                Orchestrator(self.config, self.root / 'cache').run(fixture.ctx, output_dir=self.root / 'out')
        self.assert_no_success('render')
        self.assertEqual(fixture.ctx.package.structure_path.read_bytes(), original)

    def test_absent_optional_transitive_dependency_is_recorded_not_required(self):
        from importlib import metadata
        from aimusic.audio.orchestrator import _runtime_identity
        real_version = metadata.version

        def version(name):
            if name == 'typing-inspection':  # Not required by earlier supported Pydantic v2 releases.
                raise metadata.PackageNotFoundError(name)
            return real_version(name)

        with patch('importlib.metadata.version', side_effect=version):
            identity = _runtime_identity('expressivization')
        self.assertEqual(identity['dependencies']['typing-inspection'], 'not-installed')

    def test_standalone_plan_digest_matches_parsed_buffer_and_rejects_changed_path(self):
        import hashlib
        from aimusic.audio.midi_events import MidiEvents
        from aimusic.audio.render.stems import prepare_stem_render, render_role_stems
        original = self.source.read_bytes()
        parse = MidiEvents.from_bytes

        def parsing(data):
            self.mutate_midi(self.source)
            return parse(data)

        with patch.object(MidiEvents, 'from_bytes', side_effect=parsing):
            plan = prepare_stem_render(self.source, config=self.config)
        self.assertEqual(plan.source_sha256, hashlib.sha256(original).hexdigest())
        self.assertNotEqual(plan.stems[0].midi.notes[0].velocity, 23)
        with self.assertRaisesRegex(ValueError, 'MIDI bytes changed'):
            render_role_stems(self.source, output_dir=self.root / 'out', plan=plan)
        self.assertFalse((self.root / 'out').exists())

    def test_backend_cannot_mutate_consumed_stem_midi_and_claim_success(self):
        from dataclasses import replace
        from aimusic.audio.render import registry
        simple = registry.resolve_renderer('bass', config={}).backend

        def render(request):
            result = simple.render(request)
            self.mutate_midi(request.midi_path)
            return result

        with patch.dict(registry._BACKENDS, {'simple': lambda asset: replace(simple, render=render)}):
            with self.assertRaisesRegex(ValueError, 'MIDI bytes changed'):
                self.run_audio()
        self.assert_no_success('render')
