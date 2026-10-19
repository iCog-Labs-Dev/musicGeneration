from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aimusic.app.cli import main


class TestMethodBCliWorkflow(unittest.TestCase):
    def test_generate_method_b_writes_cyclic_score_midi_and_two_leg_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            argv = [
                "aimusic",
                "generate",
                "--method",
                "b",
                "--beats",
                "4",
                "--loop-midpoint",
                "2",
                "--meter",
                "4/4",
                "--groove-family",
                "straight",
                "--seed",
                "11",
                "--out",
                tmpdir,
            ]
            with patch.object(sys, "argv", argv):
                main()

            output = Path(tmpdir)
            score_files = tuple(output.glob("*_score.json"))
            midi_files = tuple(output.glob("*.mid"))
            manifest_files = tuple(output.glob("*_manifest.json"))
            self.assertEqual(len(score_files), 1)
            self.assertEqual(len(midi_files), 1)
            self.assertEqual(len(manifest_files), 1)
            self.assertGreater(midi_files[0].stat().st_size, 0)

            manifest = json.loads(manifest_files[0].read_text(encoding="utf-8"))
            self.assertEqual(manifest["config"]["run_config"]["plan_config"]["method"], "method_b")
            self.assertTrue(manifest["sb_stats"]["converged"])
            self.assertEqual(
                set(manifest["sb_legs"]),
                {"start_to_middle", "middle_to_start"},
            )
            self.assertTrue(manifest["sb_legs"]["start_to_middle"]["converged"])
            self.assertTrue(manifest["sb_legs"]["middle_to_start"]["converged"])


if __name__ == "__main__":
    unittest.main()
