"""Bridge backend selection tests (m2a path skipped unless installed)."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]


def _audio_extra_available() -> bool:
    try:
        import yaml  # noqa: F401
        return True
    except ImportError:
        return False


def _m2a_available() -> bool:
    try:
        import importlib.util

        return importlib.util.find_spec("m2a") is not None
    except ModuleNotFoundError:
        return False


@unittest.skipUnless(_audio_extra_available(), "requires pip install -e '.[audio]'")
class TestAudioBridge(unittest.TestCase):
    def test_resolve_backend_native_default(self) -> None:
        from aimusic.audio.bridge import resolve_backend

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AIMUSIC_AUDIO_BACKEND", None)
            self.assertEqual(resolve_backend(), "native")

    def test_resolve_backend_forced_m2a(self) -> None:
        from aimusic.audio.bridge import resolve_backend

        with mock.patch.dict(os.environ, {"AIMUSIC_AUDIO_BACKEND": "m2a"}):
            self.assertEqual(resolve_backend(), "m2a")

    def test_m2a_missing_raises_clear_error(self) -> None:
        if _m2a_available():
            self.skipTest("m2a is installed; cannot assert missing-extra error")
        from aimusic.audio.bridge import m2a_available, render_audio_from_package

        self.assertFalse(m2a_available())
        with mock.patch.dict(os.environ, {"AIMUSIC_AUDIO_BACKEND": "m2a"}):
            with tempfile.TemporaryDirectory() as tmp_s:
                tmp = Path(tmp_s)
                # Build a minimal valid RenderPackage via generate CLI.
                import subprocess
                import sys

                proc = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "aimusic.app.cli",
                        "generate",
                        "--seed",
                        "3",
                        "--beats",
                        "4",
                        "--out",
                        str(tmp),
                    ],
                    cwd=str(REPO_ROOT),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(proc.returncode, 0, msg=proc.stderr + proc.stdout)
                package = next(tmp.glob("run_*"))
                with self.assertRaises(ImportError) as ctx:
                    render_audio_from_package(package, output_dir=tmp / "out")
                self.assertIn("audio-bridge", str(ctx.exception).lower())

    def test_effective_config_does_not_reread_paths(self) -> None:
        from aimusic.audio.bridge import render_audio_from_package
        from aimusic.audio.config import load_audio_config
        from tests.test_audio_entry import _generate_package

        with tempfile.TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            package = _generate_package(tmp)
            config = load_audio_config("listening").to_dict()
            with mock.patch.dict(os.environ, {"AIMUSIC_AUDIO_BACKEND": "native"}), mock.patch(
                "aimusic.audio.orchestrator.Orchestrator"
            ) as orch, mock.patch(
                "aimusic.audio.bridge._load_yaml_config", side_effect=AssertionError("reread")
            ):
                render_audio_from_package(package, output_dir=tmp / "audio", config=config)
                self.assertEqual(orch.call_args.kwargs["config"]["render"], config["render"])

    @unittest.skipUnless(_m2a_available(), "requires pip install -e '.[audio-bridge]'")
    def test_m2a_render_from_generated_package(self) -> None:
        import subprocess
        import sys

        from aimusic.audio.bridge import render_audio_from_package

        with tempfile.TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "aimusic.app.cli",
                    "generate",
                    "--seed",
                    "11",
                    "--beats",
                    "4",
                    "--out",
                    str(tmp),
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr + proc.stdout)
            package = next(tmp.glob("run_*"))
            with mock.patch.dict(os.environ, {"AIMUSIC_AUDIO_BACKEND": "m2a"}):
                out = render_audio_from_package(
                    package,
                    output_dir=tmp / "audio_out",
                    profile_path=REPO_ROOT / "config" / "audio.default.yaml",
                )
            self.assertTrue((out / "stems" / "rough_mix.wav").is_file())


if __name__ == "__main__":
    unittest.main()


class TestExplicitBackendFailurePolicy(unittest.TestCase):
    def test_native_defect_never_calls_bridge_even_when_available(self):
        from aimusic.audio.bridge import render_audio_from_package
        from tests.test_audio_entry import _generate_package
        with tempfile.TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            package = _generate_package(tmp)
            with mock.patch.dict(os.environ, {'AIMUSIC_AUDIO_BACKEND': 'native'}), mock.patch(
                'aimusic.audio.orchestrator.Orchestrator.run', side_effect=RuntimeError('native defect')
            ), mock.patch('aimusic.audio.bridge.m2a_available', return_value=True) as available:
                with self.assertRaisesRegex(RuntimeError, 'native defect'):
                    render_audio_from_package(package, output_dir=tmp / 'out')
                available.assert_not_called()

    def test_unknown_application_backend_is_an_error(self):
        from aimusic.audio.bridge import resolve_backend
        with mock.patch.dict(os.environ, {'AIMUSIC_AUDIO_BACKEND': 'typo'}), self.assertRaisesRegex(ValueError, 'typo'):
            resolve_backend()


class TestM2APublication(unittest.TestCase):
    def setUp(self):
        import sys
        from tests.test_audio_entry import _generate_package
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.package = _generate_package(self.root / 'generated')
        self.original = {p.name: p.read_bytes() for p in self.package.iterdir() if p.is_file()}
        self.out = self.root / 'out'
        self.out.mkdir()
        self.orch = mock.Mock()
        self.orch.return_value.run.side_effect = lambda midi, *, output_dir: output_dir
        module = mock.Mock(Orchestrator=self.orch)
        for patcher in (
            mock.patch.dict(os.environ, {'AIMUSIC_AUDIO_BACKEND': 'm2a'}),
            mock.patch('aimusic.audio.bridge.m2a_available', return_value=True),
            mock.patch.dict(sys.modules, {'m2a': mock.Mock(), 'm2a.orchestrator': module}),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def symlink_or_skip(self, link, target, *, directory=False):
        try:
            link.symlink_to(target, target_is_directory=directory)
        except OSError as exc:
            if getattr(exc, 'winerror', None) == 1314:
                self.skipTest('Native symlink privilege unavailable; file/hardlink collisions and native junction tests remain')
            raise

    def render(self, out=None):
        from aimusic.audio.bridge import render_audio_from_package
        return render_audio_from_package(self.package, output_dir=out or self.out)

    def assert_source_unchanged(self):
        self.assertEqual(self.original, {p.name: p.read_bytes() for p in self.package.iterdir() if p.is_file()})

    def test_collisions_preflight_entire_bundle_without_overwrite_or_partial_copy(self):
        for name in ('score.mid', 'structure.json'):
            for kind in ('regular', 'victim_link', 'source_link', 'source_hardlink'):
                with self.subTest(name=name, kind=kind):
                    target = self.out / name
                    victim = self.root / 'victim'
                    victim.write_bytes(b'KEEP')
                    if kind == 'regular':
                        target.write_bytes(b'EXISTING')
                    elif kind == 'victim_link':
                        self.symlink_or_skip(target, victim)
                    else:
                        # Cross-source alias: copy2's same-file guard cannot help.
                        source = self.package / ('structure.json' if name == 'score.mid' else 'score.mid')
                        if kind == 'source_link':
                            self.symlink_or_skip(target, source)
                        else:
                            os.link(source, target)
                    before = target.read_bytes()
                    try:
                        with self.assertRaises((ValueError, FileExistsError)):
                            self.render()
                        self.orch.assert_not_called()
                        self.assertEqual(target.read_bytes(), before)
                        self.assertEqual(victim.read_bytes(), b'KEEP')
                        self.assert_source_unchanged()
                        self.assertEqual(list(self.out.iterdir()), [target])
                    finally:
                        target.unlink()

    def test_output_directory_alias_to_package_is_rejected(self):
        for kind in ('direct', 'symlink'):
            with self.subTest(kind=kind):
                out = self.package
                if kind == 'symlink':
                    alias = self.root / 'alias'
                    self.symlink_or_skip(alias, self.package.parent, directory=True)
                    out = alias / self.package.name
                with self.assertRaises((ValueError, FileExistsError)):
                    self.render(out)
                self.assert_source_unchanged()
        self.orch.assert_not_called()

    def test_failed_backend_leaves_no_partial_publication_or_source_changes(self):
        def fail(midi, *, output_dir):
            (output_dir / 'partial.wav').write_bytes(b'PARTIAL')
            Path(midi).write_bytes(b'BACKEND EDIT')
            raise RuntimeError('backend failed')
        self.orch.return_value.run.side_effect = fail
        with self.assertRaisesRegex(RuntimeError, 'backend failed'):
            self.render()
        self.assert_source_unchanged()
        self.assertEqual(list(self.out.iterdir()), [])

    def test_success_publishes_complete_private_backend_output(self):
        def run(midi, *, output_dir):
            self.assertNotEqual(Path(midi).resolve(), (self.package / 'score.mid').resolve())
            self.assertEqual(Path(midi).read_bytes(), self.original['score.mid'])
            self.assertEqual(list(self.out.iterdir()), [])
            (output_dir / 'stems').mkdir()
            (output_dir / 'stems/mix.wav').write_bytes(b'AUDIO')
            return output_dir
        self.orch.return_value.run.side_effect = run
        self.assertEqual(self.render(), self.out)
        self.assertEqual((self.out / 'score.mid').read_bytes(), self.original['score.mid'])
        self.assertEqual((self.out / 'structure.json').read_bytes(), self.original['structure.json'])
        self.assertEqual((self.out / 'stems/mix.wav').read_bytes(), b'AUDIO')
        self.assert_source_unchanged()

    def test_late_publication_failure_rolls_back_all_directories(self):
        def run(midi, *, output_dir):
            (output_dir / 'stems').mkdir()
            (output_dir / 'stems/mix.wav').write_bytes(b'AUDIO')
            return output_dir
        self.orch.return_value.run.side_effect = run
        if os.name == 'nt':
            from aimusic.audio.windows_native import WindowsAPI
            operation, real_link = 'aimusic.audio.windows_native.WindowsAPI.rename', WindowsAPI.rename
        else:
            operation, real_link = 'os.link', os.link

        def link(*args, **kwargs):
            if Path(args[-1]).name == 'mix.wav':
                raise OSError('publication failed')
            return real_link(*args, **kwargs)
        with mock.patch(operation, new=link), self.assertRaisesRegex(OSError, 'publication failed'):
            self.render()
        self.assertFalse([p for p in self.out.rglob('*') if p.is_file()])
        self.assert_source_unchanged()


class TestBridgeMalformedYAML(unittest.TestCase):
    def test_config_and_profile_errors_include_path_before_output_writes(self):
        from aimusic.audio.bridge import render_audio_from_package
        from tests.test_audio_entry import _generate_package

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package = _generate_package(root / 'generated')
            path = root / 'broken.yaml'
            path.write_text('render: [\n')
            for option in ('config_path', 'profile_path'):
                with self.subTest(option=option):
                    with self.assertRaisesRegex(ValueError, 'YAML') as error:
                        render_audio_from_package(package, output_dir=root / 'out', **{option: path})
                    self.assertIn(str(path), str(error.exception))
                    self.assertFalse((root / 'out').exists())
