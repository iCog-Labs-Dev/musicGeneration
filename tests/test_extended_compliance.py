from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aimusic.app.cli import main
from aimusic.core.config import EDOConfig
from aimusic.core.interfaces import (
    AlgorithmicCandidateGenerator,
    AlgorithmicScoreDecoder,
    CandidateGenerator,
    MidiScoreRenderer,
    ScoreDecoder,
    ScoreRenderer,
    TonalMetric,
    RuleBasedTonalMetric,
)


class TestPublicComponentProtocols(unittest.TestCase):
    def test_default_adapters_satisfy_protocols(self) -> None:
        self.assertIsInstance(RuleBasedTonalMetric(), TonalMetric)
        self.assertIsInstance(AlgorithmicCandidateGenerator(), CandidateGenerator)
        self.assertIsInstance(AlgorithmicScoreDecoder(), ScoreDecoder)
        self.assertIsInstance(MidiScoreRenderer(EDOConfig(n=12)), ScoreRenderer)


class TestExtendedCliModes(unittest.TestCase):
    def test_section_wise_cli_generates_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            argv = [
                "aimusic", "generate", "--method", "a", "--beats", "8",
                "--section-names", "intro,theme", "--meter", "4/4",
                "--groove-family", "straight", "--out", tmpdir,
            ]
            with patch.object(sys, "argv", argv):
                main()
            manifest_path = next(Path(tmpdir).glob("*_manifest.json"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(set(manifest["sb_legs"]), {"intro", "theme"})
            self.assertTrue(manifest["sb_stats"]["converged"])

if __name__ == "__main__":
    unittest.main()
