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

    def test_context_retains_tuning_identity_and_count_sources(self) -> None:
        import json
        from aimusic.audio.from_score import from_score

        with tempfile.TemporaryDirectory() as tmp:
            ctx = from_score(_generate_package(Path(tmp)))
            self.assertEqual(ctx.tuning, json.loads(ctx.package.tuning_path.read_text()))
            self.assertEqual(ctx.verified_identity["package_hash"], ctx.package.content_hash)
            self.assertEqual(ctx.verified_identity["source_hash"], ctx.package.source_hash)
            self.assertEqual(ctx.track_note_counts, ctx.score.track_event_counts())
            self.assertEqual(ctx.track_roles, {t["name"]: t["role"] for t in ctx.structure.tracks})
            self.assertEqual(ctx.role_source, "structure.json")
            self.assertEqual(ctx.count_source, "score.json")

    def test_validate_package_rechecks_loaded_files(self) -> None:
        from aimusic.audio.from_score import from_score, validate_package
        from aimusic.core.render_package import ContractViolation

        with tempfile.TemporaryDirectory() as tmp:
            ctx = from_score(_generate_package(Path(tmp)))
            ctx.package.tuning_path.write_bytes(ctx.package.tuning_path.read_bytes() + b" ")
            with self.assertRaisesRegex(ContractViolation, "tuning.json.*expected=.*actual="):
                validate_package(ctx)

    def test_native_entry_receives_context_not_only_midi(self) -> None:
        from unittest.mock import patch
        from aimusic.audio.entry import render_audio
        from aimusic.audio.from_score import AudioPackageContext

        with tempfile.TemporaryDirectory() as tmp:
            package = _generate_package(Path(tmp))
            with patch("aimusic.audio.orchestrator.Orchestrator") as orch, patch.dict(
                os.environ, {"AIMUSIC_AUDIO_BACKEND": "native"}
            ):
                orch.return_value.run.return_value = Path(tmp)
                self.assertEqual(render_audio(package, validate_only=False), 0)
                context = orch.return_value.run.call_args.args[0]
                self.assertIsInstance(context, AudioPackageContext)
                self.assertEqual(context.provenance, "planner")

    def test_bridge_entry_receives_effective_named_profile(self) -> None:
        from unittest.mock import patch
        from aimusic.audio.entry import render_audio

        with tempfile.TemporaryDirectory() as tmp:
            package = _generate_package(Path(tmp))
            with patch("aimusic.audio.bridge.render_audio_from_package") as bridge, patch.dict(
                os.environ, {"AIMUSIC_AUDIO_BACKEND": "m2a"}
            ):
                bridge.return_value = Path(tmp)
                render_audio(package, profile="listening", validate_only=False)
                config = bridge.call_args.kwargs["config"]
                self.assertEqual(config["render"]["backend"], "fluidsynth")
                self.assertFalse(config["render"]["allow_simple_fallback"])

    def test_validate_only_does_not_import_synth_or_orchestrator(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            package = _generate_package(Path(tmp))
            code = (
                "import sys; from aimusic.audio.entry import render_audio; "
                f"assert render_audio({str(package)!r}) == 0; "
                "assert not any(n.startswith(('aimusic.audio.render', "
                "'aimusic.audio.orchestrator', 'pretty_midi', 'soundfile', "
                "'fluidsynth')) for n in sys.modules)"
            )
            result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

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

    def test_full_render_writes_stems(self) -> None:
        import soundfile as sf

        with tempfile.TemporaryDirectory() as tmp:
            package_root = _generate_package(Path(tmp) / 'generated')
            originals = {p: p.read_bytes() for p in package_root.iterdir() if p.is_file()}
            output = Path(tmp) / 'audio'
            result = _run_cli(
                "render-audio",
                str(package_root),
                "--profile", "preview", "--output-dir", str(output),
                "--no-validate-only",
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr + result.stdout)
            self.assertIn("Audio render OK", result.stdout)
            self.assertIn("rough_mix", result.stdout.lower())
            stems = list((output / 'stems').glob('track_*.wav'))
            self.assertTrue(stems)
            for wav in stems + [output / 'stems/rough_mix.wav']:
                self.assertGreater(wav.stat().st_size, 44)
                samples, rate = sf.read(wav)
                self.assertGreater(samples.size, 0)
                self.assertEqual(rate, 48000)
            self.assertEqual(originals, {p: p.read_bytes() for p in package_root.iterdir() if p.is_file()})

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
