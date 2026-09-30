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
