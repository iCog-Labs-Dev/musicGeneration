"""Tests for M1 PR2 audio entry spine (from_score + render-audio validate CLI)."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "aimusic.app.cli", *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
    )


def _generate_package(tmp: Path, *, seed: int = 11, beats: int = 4) -> Path:
    result = _run_cli(
        "generate",
        "--seed",
        str(seed),
        "--beats",
        str(beats),
        "--out",
        str(tmp),
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr + result.stdout)
    packages = list(tmp.glob("run_*"))
    if len(packages) != 1:
        raise RuntimeError(f"expected one RenderPackage, found {packages!r}")
    return packages[0]


def _audio_extra_available() -> bool:
    try:
        import yaml  # noqa: F401
        return True
    except ImportError:
        return False


@unittest.skipUnless(_audio_extra_available(), "requires pip install -e '.[audio]'")
class TestAudioEntry(unittest.TestCase):
    def test_from_score_loads_generated_package(self) -> None:
        from aimusic.audio.from_score import from_score, validate_package

        with tempfile.TemporaryDirectory() as tmp:
            package_root = _generate_package(Path(tmp))
            ctx = from_score(package_root)
            report = validate_package(ctx)

            self.assertEqual(ctx.structure.schema, "aimusic.structure/1")
            self.assertEqual(ctx.provenance, "planner")
            self.assertGreater(report.track_count, 0)
            self.assertTrue(report.track_names)
            self.assertEqual(report.package_root, package_root)

    def test_load_audio_config_default(self) -> None:
        from aimusic.audio.config import load_audio_config

        config = load_audio_config()
        self.assertEqual(config.render.backend, "simple")
        self.assertTrue(config.stages.render)
        self.assertFalse(config.stages.restyle)

    def test_validate_only_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package_root = _generate_package(Path(tmp))
            result = _run_cli(
                "render-audio",
                str(package_root),
                "--validate-only",
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)
            self.assertIn("RenderPackage validation OK", result.stdout)
            self.assertIn("provenance: planner", result.stdout)
            self.assertIn("render.backend: simple", result.stdout)

    def test_render_audio_missing_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "run_missing"
            result = _run_cli("render-audio", str(missing))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not found", result.stderr.lower())

    def test_full_render_not_implemented(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package_root = _generate_package(Path(tmp))
            result = _run_cli(
                "render-audio",
                str(package_root),
                "--no-validate-only",
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("not implemented yet", result.stderr.lower())

    def test_core_import_still_light(self) -> None:
        code = (
            "import sys\n"
            "import aimusic.core\n"
            "forbidden = {'torch', 'librosa', 'madmom', 'demucs'}\n"
            "loaded = forbidden.intersection(sys.modules)\n"
            "assert not loaded, loaded\n"
            "print('ok')\n"
        )
        env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr + proc.stdout)
        self.assertIn("ok", proc.stdout)


if __name__ == "__main__":
    unittest.main()
