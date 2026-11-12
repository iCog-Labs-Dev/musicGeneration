import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import mido

from aimusic.app.components import load_component_config
from aimusic.core.core_types import Score
from tests.test_artifact_injection import KnownScoreDecoder


class TestComponentSelectionWorkflows(unittest.TestCase):
    def test_same_json_selects_all_four_components_in_cli_and_ui(self) -> None:
        import ui

        selection: dict[str, dict[str, object]] = {
            "tonal_metric": {"factory": "tests.test_component_factory:make_private_metric", "options": {
                "distance": 2.5, "api_key": "private-workflow-key", "endpoint": "https://example.invalid/music?signature=private-workflow-signature",
            }, "public_options": ["distance"]},
            "candidate_generator": {"factory": "tests.component_providers:make_candidates"},
            "decoder": {"factory": "tests.component_providers:make_decoder"},
            "renderer": {"factory": "tests.component_providers:make_renderer", "options": {"marker": "selected-from-json"}, "public_options": ["marker"]},
        }
        for edo in (12, 19):
            for workflow in ("cli", "ui"):
                with self.subTest(edo=edo, workflow=workflow), tempfile.TemporaryDirectory() as folder:
                    directory = Path(folder)
                    config_path = directory / "components.json"
                    config_path.write_text(json.dumps(selection), encoding="utf-8")
                    if workflow == "cli":
                        result = subprocess.run(
                            [sys.executable, "-m", "aimusic.app.cli", "generate", "--seed", "11", "--beats", "4",
                             "--edo", str(edo), "--tempo-bpm", "105", "--out", folder, "--components", str(config_path)],
                            cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=120,
                        )
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    else:
                        params = ui.GenerationParams(
                            seed=11, beats=4, edo=edo, meter="4/4", groove_family="straight", tempo_bpm=105,
                            sample_path=False, drum_density=.75, bass_density=.6, comping_density=.55,
                            lead_density=.45, pitch_bend_range=2, rendering_method="MPE", bass_program=34,
                            comping_program=5, lead_program=88, drum_track=["drums"], section_wise=False,
                        )
                        with patch.object(ui, "OUTPUT_DIR", directory), contextlib.redirect_stdout(io.StringIO()):
                            artifacts = ui._generate_artifacts(params, component_config=load_component_config(config_path))
                        self.assertIsNotNone(artifacts.render_result)
                    manifest = json.loads(next(directory.glob("*_manifest.json")).read_text(encoding="utf-8"))
                    components = manifest["config"]["components"]
                    for role, config in selection.items():
                        self.assertEqual(components[role]["source"], "configured")
                        self.assertEqual(components[role]["factory"], config["factory"])
                        options = config.get("options", {})
                        public = config.get("public_options", [])
                        assert isinstance(options, dict) and isinstance(public, list)
                        self.assertEqual(components[role]["options"], {name: options[name] for name in public})
                        self.assertEqual(components[role]["unrecorded_option_count"], len(options) - len(public))
                    self.assertEqual(components["renderer"]["construction_settings"]["edo"], edo)
                    self.assertEqual(components["decoder"]["implementation"], "tests.test_artifact_injection:KnownScoreDecoder")
                    self.assertEqual(components["renderer"]["implementation"], "tests.component_providers:MarkedMidiRenderer")
                    actual_score = Score.from_dict(json.loads(next(directory.glob("*_score.json")).read_text()))
                    self.assertEqual(actual_score, KnownScoreDecoder().score)
                    midi = mido.MidiFile(next(directory.glob("*.mid")))
                    self.assertIn("selected-from-json", [message.text for track in midi.tracks for message in track if message.type == "text"])
                    self.assertEqual(sum(message.type == "note_on" and message.velocity > 0 for track in midi.tracks for message in track), 1)
                    if workflow == "ui":
                        self.assertIn(88, [message.program for track in midi.tracks for message in track if message.type == "program_change"])
                    for manifest_path in directory.rglob("*manifest*.json"):
                        serialized = manifest_path.read_text(encoding="utf-8")
                        self.assertNotIn("private-workflow-key", serialized)
                        self.assertNotIn("private-workflow-signature", serialized)

    def test_ui_startup_reads_the_same_operator_config_file(self) -> None:
        import ui

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "components.json"
            path.write_text(json.dumps({"tonal_metric": {"factory": "tests.test_component_factory:make_metric", "options": {"distance": 3.0}}}))
            with patch.object(ui, "__name__", "__main__"), patch.object(sys, "argv", ["ui.py", "--components", str(path)]):
                self.assertEqual(ui._startup_component_config(), load_component_config(path))
