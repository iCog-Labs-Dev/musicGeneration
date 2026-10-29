from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mido

import ui
from aimusic.app.cli import handle_export, handle_generate, handle_inspect
from aimusic.core.config import EDOConfig, MicrotonalRendering
from aimusic.render import SymbolicNote, render_midi, summarize_midi
from aimusic.theory.edo import EDO


def _generate_args(output_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(
        seed=47,
        beats=4,
        edo=12,
        meter="3/4",
        groove_family="straight",
        tempo_bpm=105.0,
        sample_path=False,
        subbeats_per_beat=4,
        drum_density=0.75,
        bass_density=0.60,
        comping_density=0.55,
        lead_density=0.45,
        base_tuning=0,
        pitch_bend_range=2,
        rendering_method=MicrotonalRendering.MPE.name,
        track_program=[],
        drum_track=[],
        out=str(output_dir),
    )


def _normalized_manifest(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    manifest.pop("run_id")
    manifest.pop("timestamp")
    return manifest


class TestCliArtifactWorkflows(unittest.TestCase):
    def test_generate_is_deterministic_then_export_and_inspect_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            with contextlib.redirect_stdout(io.StringIO()):
                handle_generate(_generate_args(output_dir))
                handle_generate(_generate_args(output_dir))

            score_paths = sorted(output_dir.glob("*_score.json"))
            midi_paths = sorted(
                path for path in output_dir.glob("*.mid") if path.name != "export.mid"
            )
            manifest_paths = sorted(output_dir.glob("*_manifest.json"))
            self.assertEqual(
                (len(score_paths), len(midi_paths), len(manifest_paths)),
                (2, 2, 2),
            )

            self.assertEqual(score_paths[0].read_bytes(), score_paths[1].read_bytes())
            self.assertEqual(midi_paths[0].read_bytes(), midi_paths[1].read_bytes())
            self.assertEqual(
                _normalized_manifest(manifest_paths[0]),
                _normalized_manifest(manifest_paths[1]),
            )

            with score_paths[0].open(encoding="utf-8") as score_file:
                score_data = json.load(score_file)
            midi_summary = summarize_midi(str(midi_paths[0]))
            self.assertEqual(midi_summary.total_notes, score_data["event_count"])

            export_path = output_dir / "export.mid"
            export_args = argparse.Namespace(
                file=str(score_paths[0]),
                out=str(export_path),
                edo=12,
                base_tuning=0,
                pitch_bend_range=2,
                rendering_method=MicrotonalRendering.MPE.name,
                track_program=[],
                drum_track=[],
            )
            with contextlib.redirect_stdout(io.StringIO()) as export_output:
                handle_export(export_args)
            self.assertIn("Exported multitrack MIDI", export_output.getvalue())
            self.assertEqual(export_path.read_bytes(), midi_paths[0].read_bytes())

            with contextlib.redirect_stdout(io.StringIO()) as inspect_output:
                handle_inspect(argparse.Namespace(file=str(manifest_paths[0])))
            report = inspect_output.getvalue()
            self.assertIn("Inspection Report for Run", report)
            self.assertIn("Layer sizes:", report)
            self.assertIn("Tension Arc", report)


class TestUiArtifactWorkflow(unittest.TestCase):
    def test_builtin_preview_applies_mpe_pitch_bends(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            midi_path = Path(temp_dir) / "19-edo.mid"
            edo = EDO(EDOConfig(n=19, base_tuning=60, pitch_bend_range=2))
            render_midi(
                [SymbolicNote(pitch_height=1, start_time=0.0, end_time=1.0)],
                edo,
                str(midi_path),
            )

            preview_note = ui._extract_midi_preview_notes(midi_path)[0]
            actual_frequency = ui._midi_note_frequency(
                preview_note.midi_note,
                preview_note.pitch_bend,
                preview_note.pitch_bend_range,
            )
            expected_midi_pitch = 60 + (12 / 19)
            expected_frequency = 440.0 * (2.0 ** ((expected_midi_pitch - 69) / 12.0))
            cents_error = abs(1200.0 * math.log2(actual_frequency / expected_frequency))

        self.assertLessEqual(cents_error, 0.3)

    def test_ui_accepts_mts_generation(self) -> None:
        params = ui._normalize_inputs(
            1,
            4,
            19,
            "4/4",
            "straight",
            120,
            False,
            0.75,
            0.60,
            0.55,
            0.45,
            2,
            MicrotonalRendering.MTS.name,
            34,
            5,
            88,
            ["drums"],
        )

        self.assertEqual(params.rendering_method, MicrotonalRendering.MTS.name)

    def test_ui_rejects_mts_audio_preview_with_actionable_message(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            midi_path = temp_path / "19-edo-mts.mid"
            wav_path = temp_path / "19-edo-mts.wav"
            render_midi(
                [SymbolicNote(pitch_height=1, start_time=0.0, end_time=1.0)],
                EDO(
                    EDOConfig(
                        n=19,
                        base_tuning=60,
                        microtonal_rendering_method=MicrotonalRendering.MTS,
                    )
                ),
                str(midi_path),
            )

            with self.assertRaisesRegex(
                ui.MidiAudioConversionError,
                "MTS-compatible synthesizer",
            ):
                ui._convert_midi_to_wav(midi_path, wav_path)

    def test_ui_generation_helper_writes_complete_consistent_artifacts(self) -> None:
        params = ui.GenerationParams(
            seed=53,
            beats=4,
            edo=19,
            meter="5/4",
            groove_family="swing",
            tempo_bpm=123.0,
            sample_path=False,
            drum_density=0.65,
            bass_density=0.50,
            comping_density=0.40,
            lead_density=0.55,
            pitch_bend_range=2,
            rendering_method=MicrotonalRendering.MPE.name,
            bass_program=34,
            comping_program=5,
            lead_program=88,
            drum_track=["drums"],
            section_wise=False,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.object(ui, "OUTPUT_DIR", Path(temp_dir)):
                artifacts = ui._generate_artifacts(params)

            self.assertTrue(artifacts.score_path.is_file())
            self.assertTrue(artifacts.midi_path.is_file())
            self.assertTrue(artifacts.manifest_path.is_file())
            self.assertFalse(artifacts.wav_path.exists())

            with artifacts.score_path.open(encoding="utf-8") as score_file:
                score_data = json.load(score_file)
            with artifacts.manifest_path.open(encoding="utf-8") as manifest_file:
                manifest_data = json.load(manifest_file)
            midi_file = mido.MidiFile(artifacts.midi_path)

            note_count = sum(
                message.type == "note_on" and message.velocity > 0
                for track in midi_file.tracks
                for message in track
            )
            self.assertEqual(note_count, score_data["event_count"])
            self.assertEqual(manifest_data["seed"], params.seed)
            self.assertEqual(manifest_data["config"]["meter"], params.meter)
            self.assertEqual(
                manifest_data["config"]["rendering_method"],
                params.rendering_method,
            )


if __name__ == "__main__":
    unittest.main()
