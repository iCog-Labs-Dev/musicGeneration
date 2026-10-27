"""Unit tests for M1 native orchestrator (simple render)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "audio"
QUANTIZED = FIXTURES / "eight_bar_quantized.mid"


def _audio_spine_available() -> bool:
    try:
        import pretty_midi  # noqa: F401
        import soundfile  # noqa: F401
        import yaml  # noqa: F401
        return QUANTIZED.is_file()
    except ImportError:
        return False


@unittest.skipUnless(_audio_spine_available(), "requires pip install -e '.[audio]' + fixtures")
class TestAudioOrchestrator(unittest.TestCase):
    def _m1_config(self, tmp: Path) -> dict:
        return {
            "code_version": "0.1.0",
            "paths": {
                "cache_dir": str(tmp / "cache"),
                "output_dir": str(tmp / "out"),
                "soundfont": None,
            },
            "stages": {
                "analysis": True,
                "expressivization": True,
                "render": True,
                "prompts": False,
                "restyle": False,
                "scoring": False,
                "mixmaster": False,
            },
            "groove": {
                "spec": {
                    "swing": {
                        "ratio": 0.58,
                        "subdivision": 8,
                        "applies_to": ["drums", "bass", "comp"],
                    },
                }
            },
            "render": {
                "sample_rate": 48000,
                "peak_dbfs": -12.0,
                "backend": "simple",
                "allow_simple_fallback": True,
            },
            "budget": {"max_endpoint_calls": 40, "max_usd_estimate": 10.0},
        }

    def test_orchestrator_m1_and_cache_hit(self) -> None:
        from aimusic.audio.orchestrator import Orchestrator

        with tempfile.TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            cfg = self._m1_config(tmp)
            orch = Orchestrator(config=cfg, cache_dir=Path(cfg["paths"]["cache_dir"]))
            out1 = orch.run(QUANTIZED, output_dir=tmp / "out1")
            self.assertTrue((out1 / f"{QUANTIZED.stem}.analysis.json").is_file())
            self.assertTrue((out1 / f"{QUANTIZED.stem}.expressive.mid").is_file())
            self.assertTrue((out1 / "stems" / "rough_mix.wav").is_file())
            self.assertGreaterEqual(orch.cache_misses.get("render", 0), 1)

            orch2 = Orchestrator(config=cfg, cache_dir=Path(cfg["paths"]["cache_dir"]))
            out2 = orch2.run(QUANTIZED, output_dir=tmp / "out2")
            self.assertTrue((out2 / "stems" / "rough_mix.wav").is_file())
            self.assertGreaterEqual(orch2.cache_hits.get("analysis", 0), 1)
            self.assertGreaterEqual(orch2.cache_hits.get("expressivization", 0), 1)
            self.assertGreaterEqual(orch2.cache_hits.get("render", 0), 1)


@unittest.skipUnless(_audio_spine_available(), "requires pip install -e '.[audio]' + fixtures")
class TestAudioOrchestratorCacheSidecars(unittest.TestCase):
    """Regression tests for Bug B from PR #47 review.

    Invariant: for each cacheable stage, after ``Orchestrator.run`` returns,
    the user ``output_dir`` contains exactly the same set of files
    (primary artifact + every sidecar) as the corresponding cache entry.
    This must hold on both the cold path (cache miss) and the warm path
    (cache hit).
    """

    def _m1_config(self, tmp: Path) -> dict:
        return {
            "code_version": "0.1.0",
            "paths": {
                "cache_dir": str(tmp / "cache"),
                "output_dir": str(tmp / "out"),
                "soundfont": None,
            },
            "stages": {
                "analysis": True,
                "expressivization": True,
                "render": True,
                "prompts": False,
                "restyle": False,
                "scoring": False,
                "mixmaster": False,
            },
            "groove": {
                "spec": {
                    "swing": {
                        "ratio": 0.58,
                        "subdivision": 8,
                        "applies_to": ["drums", "bass", "comp"],
                    },
                }
            },
            "render": {
                "sample_rate": 48000,
                "peak_dbfs": -12.0,
                "backend": "simple",
                "allow_simple_fallback": True,
            },
            "budget": {"max_endpoint_calls": 40, "max_usd_estimate": 10.0},
        }

    def _expected_artifacts(self, out: Path) -> dict[str, Path]:
        """The set of files ``Orchestrator.run`` must produce for analysis
        and expressivization on a successful run."""
        return {
            "analysis_doc": out / f"{QUANTIZED.stem}.analysis.json",
            "analysis_manifest": out / f"{QUANTIZED.stem}.analysis.json.json",
            "expressive_midi": out / f"{QUANTIZED.stem}.expressive.mid",
            "expressive_diff": out / f"{QUANTIZED.stem}.expressive.mid.diff.json",
            "expressive_manifest": out / f"{QUANTIZED.stem}.expressive.mid.json",
        }

    def test_cold_path_copies_all_sidecars(self) -> None:
        """First run (cache miss) must copy every sidecar to the user output dir."""
        from aimusic.audio.orchestrator import Orchestrator

        with tempfile.TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            cfg = self._m1_config(tmp)
            orch = Orchestrator(config=cfg, cache_dir=Path(cfg["paths"]["cache_dir"]))
            out = orch.run(QUANTIZED, output_dir=tmp / "out1")

            for name, path in self._expected_artifacts(out).items():
                self.assertTrue(
                    path.is_file(),
                    msg=f"missing cold-path artifact: {name} -> {path}",
                )

    def test_warm_path_copies_all_sidecars(self) -> None:
        """Second run (cache hit) must copy every sidecar too."""
        from aimusic.audio.orchestrator import Orchestrator

        with tempfile.TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            cfg = self._m1_config(tmp)

            # Cold run to populate the cache.
            Orchestrator(
                config=cfg, cache_dir=Path(cfg["paths"]["cache_dir"])
            ).run(QUANTIZED, output_dir=tmp / "out_cold")

            # Warm run with a fresh output directory; everything must be
            # re-materialized from cache (primary + sidecars).
            orch2 = Orchestrator(
                config=cfg, cache_dir=Path(cfg["paths"]["cache_dir"])
            )
            out_warm = orch2.run(QUANTIZED, output_dir=tmp / "out_warm")

            self.assertGreaterEqual(orch2.cache_hits.get("analysis", 0), 1)
            self.assertGreaterEqual(
                orch2.cache_hits.get("expressivization", 0), 1
            )

            for name, path in self._expected_artifacts(out_warm).items():
                self.assertTrue(
                    path.is_file(),
                    msg=f"missing warm-path artifact: {name} -> {path}",
                )

    def test_warm_sidecars_byte_equal_to_cold(self) -> None:
        """Cache-hit sidecars must be byte-identical to cold-run sidecars."""
        from aimusic.audio.orchestrator import Orchestrator

        with tempfile.TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            cfg = self._m1_config(tmp)

            out_cold = Orchestrator(
                config=cfg, cache_dir=Path(cfg["paths"]["cache_dir"])
            ).run(QUANTIZED, output_dir=tmp / "out_cold")

            out_warm = Orchestrator(
                config=cfg, cache_dir=Path(cfg["paths"]["cache_dir"])
            ).run(QUANTIZED, output_dir=tmp / "out_warm")

            for name, cold_path in self._expected_artifacts(out_cold).items():
                warm_path = self._expected_artifacts(out_warm)[name]
                self.assertEqual(
                    cold_path.read_bytes(),
                    warm_path.read_bytes(),
                    msg=f"warm-path {name} differs byte-for-byte from cold-path",
                )


@unittest.skipUnless(_audio_spine_available(), "requires pip install -e '.[audio]' + fixtures")
class TestCopyStageArtifactsHelper(unittest.TestCase):
    """Unit tests for the ``Orchestrator._copy_stage_artifacts`` helper."""

    def _write(self, path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def test_copies_primary_and_existing_sidecars(self) -> None:
        from aimusic.audio.orchestrator import Orchestrator

        with tempfile.TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            cache = tmp / "cache"
            dest_dir = tmp / "dest"
            dest_dir.mkdir()
            self._write(cache / "stage.artifact", b"PRIMARY")
            self._write(cache / "stage.artifact.json", b"MANIFEST")
            self._write(cache / "stage.artifact.diff.json", b"DIFF")

            Orchestrator._copy_stage_artifacts(
                cache / "stage.artifact",
                dest_dir / "stage.artifact",
                ".json",
                ".diff.json",
            )

            self.assertEqual((dest_dir / "stage.artifact").read_bytes(), b"PRIMARY")
            self.assertEqual(
                (dest_dir / "stage.artifact.json").read_bytes(), b"MANIFEST"
            )
            self.assertEqual(
                (dest_dir / "stage.artifact.diff.json").read_bytes(), b"DIFF"
            )

    def test_skips_missing_sidecars(self) -> None:
        """Only sidecars that exist next to ``cached`` should be copied."""
        from aimusic.audio.orchestrator import Orchestrator

        with tempfile.TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            cache = tmp / "cache"
            dest_dir = tmp / "dest"
            dest_dir.mkdir()
            self._write(cache / "stage.artifact", b"PRIMARY")
            # Only one of the two requested sidecars exists.
            self._write(cache / "stage.artifact.json", b"MANIFEST")

            Orchestrator._copy_stage_artifacts(
                cache / "stage.artifact",
                dest_dir / "stage.artifact",
                ".json",
                ".diff.json",
            )

            self.assertEqual((dest_dir / "stage.artifact").read_bytes(), b"PRIMARY")
            self.assertEqual(
                (dest_dir / "stage.artifact.json").read_bytes(), b"MANIFEST"
            )
            self.assertFalse((dest_dir / "stage.artifact.diff.json").exists())


if __name__ == "__main__":
    unittest.main()
