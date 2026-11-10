"""CLI admission, lazy imports, real preview output and publication safety."""
from __future__ import annotations

import json
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

from tests.test_audio_imported import imported_fixture
from tests.test_audio_entry import _generate_package, _run_cli


class TestAudioCLI(unittest.TestCase):
    def setUp(self) -> None:
        env = patch.dict(os.environ, {'AIMUSIC_AUDIO_BACKEND': 'native'})
        env.start()
        os.environ.pop('AIMUSIC_SOUNDFONT', None)
        self.addCleanup(env.stop)

    def test_help_and_core_imports_do_not_load_audio(self) -> None:
        for command in ('render-midi', 'render-audio'):
            with self.subTest(command=command):
                code = f'''
import sys
from unittest.mock import patch
from aimusic.app.cli import main
with patch.object(sys, 'argv', ['aimusic', {command!r}, '--help']):
    try:
        main()
    except SystemExit as exc:
        assert exc.code == 0
assert not any(n.startswith(('aimusic.audio', 'pretty_midi', 'soundfile', 'fluidsynth', 'yaml')) for n in sys.modules)
'''
                result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                for option in ('--out', '--output-dir', '--soundfont', 'preview', 'listening', '--no-validate-only'):
                    self.assertIn(option, result.stdout)

    def test_default_validation_writes_nothing_and_allocates_no_renderer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = imported_fixture(Path(tmp) / 'source.mid')
            code = f'''
import sys
from aimusic.audio.entry import render_midi
assert render_midi({str(source)!r}, profile='listening', output_dir={str(Path(tmp) / 'out')!r}) == 0
assert not any(n.startswith(('aimusic.audio.render', 'aimusic.audio.orchestrator', 'pretty_midi', 'soundfile', 'fluidsynth')) for n in sys.modules)
'''
            result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('planner reconciliation: not applicable', result.stdout)
            self.assertEqual(list(Path(tmp).iterdir()), [source])
            result = _run_cli('render-midi', str(source))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('Imported MIDI validation OK', result.stdout)
            self.assertNotIn('Audio render OK', result.stdout)

    def test_explicit_simple_cli_renders_named_readable_wavs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = imported_fixture(root / 'song.mid')
            midi = mido.MidiFile(source)
            voice = mido.MidiTrack(msg.copy(channel=2) if hasattr(msg, 'channel') else msg.copy()
                                   for msg in midi.tracks[1])
            midi.tracks.append(voice)  # Duplicate traversal names remain separate.
            midi.save(source)
            original = source.read_bytes()
            out = root / 'out'
            result = _run_cli('render-midi', str(source), '--profile', 'preview', '--out', str(out), '--no-validate-only')
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn(str(out / 'stems/rough_mix.wav'), result.stdout)
            index = json.loads((out / 'stems/role_index.json').read_text())
            self.assertEqual(len(index['stems']), 2)
            self.assertEqual(len(set(row['wav'] for row in index['stems'])), 2)
            for wav in [out / 'stems' / row['wav'] for row in index['stems']] + [out / 'stems/rough_mix.wav']:
                self.assertEqual(wav.resolve().parent, (out / 'stems').resolve())
                self.assertGreater(wav.stat().st_size, 44)
                audio, rate = sf.read(wav)
                self.assertEqual(rate, 48000)
                self.assertGreater(len(audio), 0)
                self.assertTrue(np.isfinite(audio).all())
                self.assertGreater(float(np.sqrt(np.mean(audio ** 2))), 1e-5)
            render = json.loads((out / 'render.json').read_text())
            self.assertTrue(all(r['effective_backend'] == 'simple' for r in render['renderers']))
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual((out / 'source.mid').read_bytes(), original)
            # Existing outputs must not be silently overwritten, even with a warm cache.
            files = {p: p.read_bytes() for p in out.rglob('*') if p.is_file()}
            again = _run_cli('render-midi', str(source), '--output-dir', str(out), '--no-validate-only')
            self.assertNotEqual(again.returncode, 0)
            self.assertIn('already exists', again.stderr)
            self.assertEqual(files, {p: p.read_bytes() for p in out.rglob('*') if p.is_file()})

    def test_invalid_missing_type2_and_tuning_inputs_leave_source_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            invalid = root / 'bad.mid'
            invalid.write_bytes(b'bad MIDI')
            type2 = imported_fixture(root / 'type2.mid', midi_type=2)
            bent = imported_fixture(root / 'bent.mid')
            midi = mido.MidiFile(bent)
            midi.tracks[0].insert(0, mido.Message('pitchwheel', channel=1, pitch=2048))
            midi.save(bent)
            for path, error in ((invalid, 'Malformed'), (type2, 'type 2'), (bent, 'does not support tuning'),
                                (root / 'missing.mid', 'not found')):
                with self.subTest(path=path.name):
                    original = path.read_bytes() if path.exists() else None
                    out = root / path.stem
                    result = _run_cli('render-midi', str(path), '--out', str(out), '--no-validate-only')
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn(error, result.stderr)
                    self.assertNotIn('Traceback', result.stderr)
                    self.assertNotIn('Audio render OK', result.stdout)
                    self.assertFalse(list(out.rglob('*.wav')))
                    if original is not None:
                        self.assertEqual(path.read_bytes(), original)

    def test_safe_output_refuses_source_alias_and_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = imported_fixture(root / 'source.mid')
            original = source.read_bytes()
            outside = root / 'outside'
            outside.mkdir()
            out = root / 'out'
            out.mkdir()
            for target in (root, out):
                with self.subTest(target=target):
                    if target == out:
                        try:
                            (out / 'stems').symlink_to(outside, target_is_directory=True)
                        except OSError as exc:
                            if getattr(exc, 'winerror', None) == 1314:
                                self.skipTest('Native symlink privilege unavailable; source-alias CLI check and native junction tests remain')
                            raise
                    result = _run_cli('render-midi', str(source), '--out', str(target), '--no-validate-only')
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(source.read_bytes(), original)
                    self.assertFalse(list(outside.iterdir()))
                    self.assertFalse(list(root.rglob('*.wav')))

    def test_missing_optional_dependencies_are_actionable_during_import_and_render(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = imported_fixture(Path(tmp) / 'source.mid')
            for missing in ('yaml', 'soundfile'):
                with self.subTest(missing=missing):
                    code = f'''
import sys
from importlib.abc import MetaPathFinder
class Block(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == {missing!r}:
            raise ModuleNotFoundError('test missing ' + fullname, name=fullname)
sys.meta_path.insert(0, Block())
sys.argv = ['aimusic', 'render-midi', {str(source)!r}, '--out', {str(Path(tmp) / missing)!r}, '--no-validate-only']
from aimusic.app.cli import main
main()
'''
                    result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("pip install -e '.[audio]'", result.stderr)
                    self.assertNotIn('Traceback', result.stderr)
                    self.assertFalse(list(Path(tmp).rglob('*.wav')))

    def test_listening_missing_asset_fails_without_silent_preview(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = imported_fixture(Path(tmp) / 'song.mid')
            result = _run_cli('render-midi', str(source), '--profile', 'listening',
                              '--out', str(Path(tmp) / 'out'), '--no-validate-only')
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('SoundFont', result.stderr)
            self.assertNotIn('Audio render OK', result.stdout)
            self.assertFalse(list(Path(tmp).rglob('*.wav')))

    def test_listening_with_asset_checks_binding_before_publication(self) -> None:
        from contextlib import redirect_stderr
        from io import StringIO
        from aimusic.app.cli import main
        from aimusic.audio.render.registry import MissingRendererResource

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = imported_fixture(root / 'song.mid')
            asset = root / 'selected.sf2'
            asset.write_bytes(b'asset path test only; native binding is mocked missing')
            error = MissingRendererResource('FluidSynth bindings missing: install [audio-render]')
            stderr = StringIO()
            args = ['aimusic', 'render-midi', str(source), '--profile', 'listening',
                    '--soundfont', str(asset), '--out', str(root / 'out'), '--no-validate-only']
            with patch.object(sys, 'argv', args), redirect_stderr(stderr), patch(
                'aimusic.audio.render.fluidsynth_r.load_binding', side_effect=error
            ) as binding, self.assertRaises(SystemExit) as exited:
                main()
            self.assertEqual(exited.exception.code, 1)
            binding.assert_called_once()
            self.assertIn('[audio-render]', stderr.getvalue())
            self.assertFalse(list((root / 'out').rglob('*.wav')))
            self.assertFalse((root / 'out/render.json').exists())

    def test_output_and_soundfont_precedence_reach_both_entry_paths(self) -> None:
        from aimusic.audio.entry import render_audio, render_midi

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = _generate_package(root / 'generated')
            source = imported_fixture(root / 'song.mid')
            asset = root / 'selected.sf2'
            asset.write_bytes(b'asset selection test; no native rendering')
            profile = root / 'profile.yaml'
            profile.write_text('paths:\n  soundfont: absent.sf2\n')
            with patch('aimusic.audio.orchestrator.Orchestrator') as orch:
                orch.return_value.run.return_value = root / 'out'
                for entry, path in ((render_audio, package), (render_midi, source)):
                    entry(path, profile=profile, output_dir=root / 'out', soundfont=asset, validate_only=False)
                    self.assertEqual(orch.call_args.kwargs['config']['paths']['soundfont'], str(asset.resolve()))
                    self.assertEqual(orch.return_value.run.call_args.kwargs['output_dir'], root / 'out')
            bad = _run_cli('render-midi', str(source), '--soundfont', str(root / 'missing.sf2'),
                           '--out', str(root / 'bad-out'), '--no-validate-only')
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn('--soundfont', bad.stderr)

    def test_malformed_profile_fails_cleanly_for_both_audio_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = _generate_package(root / 'generated')
            source = imported_fixture(root / 'source.mid')
            profile = root / 'broken-profile.yaml'
            profile.write_text('render: [\n')
            for command, path in (('render-midi', source), ('render-audio', package)):
                with self.subTest(command=command):
                    result = _run_cli(command, str(path), '--profile', str(profile), '--out', str(root / 'out'),
                                      '--no-validate-only')
                    self.assertEqual(result.returncode, 1)
                    self.assertIn('YAML', result.stderr)
                    self.assertIn(str(profile), result.stderr)
                    self.assertNotIn('Traceback', result.stderr)
                    self.assertNotIn('ParserError', result.stderr)
                    self.assertFalse((root / 'out').exists())
