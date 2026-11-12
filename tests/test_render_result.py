from pathlib import Path
import tempfile
import unittest

import mido

from aimusic.core.component_types import RenderRequest, RenderResult, SymbolicNote
from aimusic.core.config import EDOConfig
from aimusic.core.core_types import NoteEvent, Score
from aimusic.render.adapters import RenderResultAdapter
from aimusic.render.midi_adapter import MidiRenderer, MidiRenderSettings
from aimusic.render.midi_render import render_midi
from aimusic.planning.plans import render_exact_bridge_demo
from aimusic.theory.edo import EDO


class TestRenderResult(unittest.TestCase):
    def test_demo_exposes_the_completed_render_result(self) -> None:
        for tuning in (12, 19):
            with self.subTest(edo=tuning), tempfile.TemporaryDirectory() as folder:
                destination = str(Path(folder) / "demo.mid")
                result = render_exact_bridge_demo(
                    start_chord="C:maj", end_chord="G:maj", output_path=destination,
                    total_beats=4, seed=11, edo=tuning,
                )
                self.assertIsNotNone(result.render_result)
                assert result.render_result is not None
                self.assertEqual(result.render_result.output_path, destination)
                self.assertEqual(result.render_result.size_bytes, Path(destination).stat().st_size)
                self.assertEqual(result.render_result.format, "midi")

    def test_reporting_renderer_result_is_checked_against_the_file(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            destination = str(Path(folder) / "result.mid")
            backend = MidiRenderer(MidiRenderSettings(EDOConfig(n=19)))
            result = RenderResultAdapter(RenderResultAdapter(backend))(Score(), RenderRequest(destination, "midi"))
            self.assertEqual(result.size_bytes, Path(destination).stat().st_size)
            previous = Path(destination).read_bytes()

            def incorrect(score: Score, request: RenderRequest) -> RenderResult:
                Path(request.output_path).write_bytes(b"incorrect metadata")
                return RenderResult(request.output_path, request.format, 0)

            with self.assertRaisesRegex(ValueError, "must match"):
                RenderResultAdapter(incorrect)(Score(), RenderRequest(destination, "midi"))
            self.assertEqual(Path(destination).read_bytes(), previous)
            self.assertEqual(list(Path(folder).iterdir()), [Path(destination)])

    def test_adapter_preserves_the_existing_score_renderer_bytes(self) -> None:
        score = Score(note_events=(NoteEvent(0, 480, 5, 0.6, track="lead"),))
        for tuning in (12, 19):
            with self.subTest(edo=tuning), tempfile.TemporaryDirectory() as folder:
                destination = str(Path(folder) / "result.mid")
                config = EDOConfig(n=tuning, base_tuning=0)
                render_midi(score, EDO(config), destination)
                baseline = Path(destination).read_bytes()
                result = RenderResultAdapter(MidiRenderer(MidiRenderSettings(config)))(
                    score, RenderRequest(destination, "midi"),
                )
                self.assertEqual(result.size_bytes, len(baseline))
                self.assertEqual(Path(destination).read_bytes(), baseline)
                self.assertTrue(mido.MidiFile(destination).tracks)

    def test_legacy_symbolic_note_sequence_signature_still_works(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            destination = str(Path(folder) / "notes.mid")
            render_midi((SymbolicNote(5, 0.0, 1.0),), EDO(EDOConfig(n=19)), destination)
            self.assertEqual(sum(message.type == "note_on" and message.velocity > 0
                                 for track in mido.MidiFile(destination).tracks for message in track), 1)

    def test_failed_renderer_propagates_without_a_success_result(self) -> None:
        def fail(score: Score, request: RenderRequest) -> RenderResult:
            raise OSError("cannot save music")

        with self.assertRaisesRegex(OSError, "cannot save music"):
            RenderResultAdapter(fail)(Score(), RenderRequest("unused.mid", "midi"))

    def test_renderer_that_writes_no_file_cannot_report_success(self) -> None:
        def no_output(score: Score, request: RenderRequest) -> RenderResult:
            return RenderResult(request.output_path, request.format, 0)

        with tempfile.TemporaryDirectory() as folder, self.assertRaises(FileNotFoundError):
            RenderResultAdapter(no_output)(Score(), RenderRequest(str(Path(folder) / "absent.mid"), "midi"))

    def test_no_op_cannot_report_an_existing_file_as_a_completed_render(self) -> None:
        def no_output(score: Score, request: RenderRequest) -> RenderResult:
            return RenderResult(request.output_path, request.format, 0)

        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "existing.mid"
            destination.write_bytes(b"previous output")
            with self.assertRaises(FileNotFoundError):
                RenderResultAdapter(no_output)(Score(), RenderRequest(str(destination), "midi"))
            self.assertEqual(destination.read_bytes(), b"previous output")
            self.assertEqual(list(Path(folder).iterdir()), [destination])

    def test_failed_overwrite_preserves_old_file_and_cleans_partial_output(self) -> None:
        def fail(score: Score, request: RenderRequest) -> RenderResult:
            Path(request.output_path).write_bytes(b"partial output")
            raise OSError("render failed")

        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "existing.mid"
            destination.write_bytes(b"previous output")
            with self.assertRaisesRegex(OSError, "render failed"):
                RenderResultAdapter(fail)(Score(), RenderRequest(str(destination), "midi"))
            self.assertEqual(destination.read_bytes(), b"previous output")
            self.assertEqual(list(Path(folder).iterdir()), [destination])
