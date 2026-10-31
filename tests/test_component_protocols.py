from __future__ import annotations

import importlib
import importlib.util
import subprocess
import sys
import unittest


class TestComponentContracts(unittest.TestCase):
    def test_importing_contracts_does_not_load_concrete_backends(self) -> None:
        result = subprocess.run(
            [sys.executable, "-c", "import sys; import aimusic.core.protocols; import aimusic.planning.protocols; "
             "forbidden = ('numpy', 'mido', 'aimusic.audio', 'aimusic.planning.candidates', "
             "'aimusic.scoring.priors', 'aimusic.decode', 'aimusic.render.midi_render'); "
             "assert not any(name in sys.modules for name in forbidden)"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_shared_contracts_preserve_existing_type_identity(self) -> None:
        self.assertIsNotNone(
            importlib.util.find_spec("aimusic.core.protocols"),
            "The shared replacement interfaces must exist before wiring plugins.",
        )
        contracts = importlib.import_module("aimusic.core.protocols")
        shared = importlib.import_module("aimusic.core.component_types")
        candidates = importlib.import_module("aimusic.planning.candidates")
        rendering = importlib.import_module("aimusic.render.midi_render")
        for name in ("TonalMetric", "ScoreDecoder", "ScoreRenderer"):
            self.assertTrue(getattr(contracts, name)._is_protocol)
        for name in ("CandidateGenerationResult", "CandidateRejection"):
            self.assertIs(getattr(shared, name), getattr(candidates, name))
        for name in ("SymbolicNote", "TrackInstrumentConfig"):
            self.assertIs(getattr(shared, name), getattr(rendering, name))
