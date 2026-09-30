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


if __name__ == "__main__":
    unittest.main()
