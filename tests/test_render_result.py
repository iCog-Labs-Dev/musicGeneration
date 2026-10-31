from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import mido

from aimusic.core.component_types import RenderResult, SymbolicNote
from aimusic.core.config import EDOConfig
from aimusic.core.core_types import NoteEvent, Score
from aimusic.render.adapters import RenderResultAdapter
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
                    renderer=RenderResultAdapter(),
                )
                self.assertIsNotNone(result.render_result)
                assert result.render_result is not None
                self.assertEqual(result.render_result.output_path, destination)
                self.assertEqual(result.render_result.size_bytes, Path(destination).stat().st_size)
                self.assertEqual(result.render_result.edo, tuning)

    def test_reporting_renderer_result_is_checked_against_the_file(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            destination = str(Path(folder) / "result.mid")
            edo = EDO(EDOConfig(n=19))
            result = RenderResultAdapter(RenderResultAdapter())(Score(), edo, destination)
            self.assertEqual(result.size_bytes, Path(destination).stat().st_size)
            self.assertEqual(result.edo, 19)
            previous = Path(destination).read_bytes()

            def incorrect(*args: object, **kwargs: object) -> RenderResult:
                Path(str(args[2])).write_bytes(b"incorrect metadata")
                return RenderResult(str(args[2]), "midi", 0, 12)

            with self.assertRaisesRegex(ValueError, "must match"):
                RenderResultAdapter(incorrect)(Score(), edo, destination)
            self.assertEqual(Path(destination).read_bytes(), previous)
            self.assertEqual(list(Path(folder).iterdir()), [Path(destination)])

    def test_adapter_reports_actual_file_without_changing_legacy_renderer(self) -> None:
        for tuning in (12, 19):
            for symbolic in (False, True):
                with self.subTest(edo=tuning, symbolic=symbolic), tempfile.TemporaryDirectory() as folder:
                    destination = str(Path(folder) / "result.mid")
                    edo = EDO(EDOConfig(n=tuning, base_tuning=0))
                    notes = (SymbolicNote(5, 0.0, 1.0),) if symbolic else Score(
                        note_events=(NoteEvent(0, 480, 5, 0.6, track="lead"),),
                    )
                    render_midi(notes, edo, destination)
                    baseline = Path(destination).read_bytes()
                    result = RenderResultAdapter()(notes, edo, destination)
                    self.assertEqual(result.output_path, destination)
                    self.assertEqual(result.format, "midi")
                    self.assertEqual(result.size_bytes, len(baseline))
                    self.assertEqual(result.edo, tuning)
                    self.assertEqual(Path(destination).read_bytes(), baseline)
                    self.assertTrue(mido.MidiFile(destination).tracks)

    def test_failed_renderer_propagates_without_a_success_result(self) -> None:
        def fail(*args: object, **kwargs: object) -> None:
            raise OSError("cannot save music")

        with self.assertRaisesRegex(OSError, "cannot save music"):
            RenderResultAdapter(fail)(Score(), EDO(EDOConfig(n=12)), "unused.mid")

    def test_renderer_that_writes_no_file_cannot_report_success(self) -> None:
        def no_output(*args: object, **kwargs: object) -> None:
            pass

        with tempfile.TemporaryDirectory() as folder, self.assertRaises(FileNotFoundError):
            RenderResultAdapter(no_output)(Score(), EDO(EDOConfig(n=12)), str(Path(folder) / "absent.mid"))

    def test_no_op_cannot_report_an_existing_file_as_a_completed_render(self) -> None:
        def no_output(*args: object, **kwargs: object) -> None:
            pass

        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "existing.mid"
            destination.write_bytes(b"previous output")
            with self.assertRaises(FileNotFoundError):
                RenderResultAdapter(no_output)(Score(), EDO(EDOConfig(n=12)), str(destination))
            self.assertEqual(destination.read_bytes(), b"previous output")
            self.assertEqual(list(Path(folder).iterdir()), [destination])

    def test_failed_overwrite_preserves_old_file_and_cleans_partial_output(self) -> None:
        def fail(*args: object, **kwargs: object) -> None:
            Path(str(args[2])).write_bytes(b"partial output")
            raise OSError("render failed")

        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "existing.mid"
            destination.write_bytes(b"previous output")
            with self.assertRaisesRegex(OSError, "render failed"):
                RenderResultAdapter(fail)(Score(), EDO(EDOConfig(n=12)), str(destination))
            self.assertEqual(destination.read_bytes(), b"previous output")
            self.assertEqual(list(Path(folder).iterdir()), [destination])
