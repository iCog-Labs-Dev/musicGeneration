from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from aimusic.core.component_types import RenderRequest, RenderResult
from aimusic.core.core_types import Score
from aimusic.render.adapters import RenderResultAdapter


class TextRenderer:
    def __call__(self, score: Score, request: RenderRequest) -> RenderResult:
        destination = Path(request.output_path)
        destination.write_text(f"notes={len(score.note_events)}", encoding="utf-8")
        return RenderResult(request.output_path, request.format, destination.stat().st_size)


class TestRendererContract(unittest.TestCase):
    def test_directory_cannot_be_published_as_an_artifact(self) -> None:
        class DirectoryRenderer:
            def __call__(self, score: Score, request: RenderRequest) -> RenderResult:
                destination = Path(request.output_path)
                destination.mkdir()
                return RenderResult(str(destination), request.format, destination.stat().st_size)

        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "output.txt"
            destination.write_text("old", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "regular"):
                RenderResultAdapter(DirectoryRenderer())(Score(), RenderRequest(str(destination), "text"))
            self.assertEqual(destination.read_text(encoding="utf-8"), "old")
    def test_symbolic_link_is_not_published_as_a_completed_file(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "score.txt"
            destination.write_text("old", encoding="utf-8")
            probe = Path(folder) / "probe"
            try:
                probe.symlink_to(destination)
            except OSError:
                self.skipTest("This Windows account cannot create test symlinks.")
            probe.unlink()

            class LinkedRenderer:
                def __call__(self, score: Score, request: RenderRequest) -> RenderResult:
                    path = Path(request.output_path)
                    payload = path.parent / "payload"
                    payload.write_text("linked", encoding="utf-8")
                    path.symlink_to(payload)
                    return RenderResult(str(path), request.format, payload.stat().st_size)

            with self.assertRaisesRegex(ValueError, "regular"):
                RenderResultAdapter(LinkedRenderer())(Score(), RenderRequest(str(destination), "text"))
            self.assertEqual(destination.read_text(encoding="utf-8"), "old")
    def test_general_adapter_writes_a_non_midi_format_and_reports_final_path(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "score.txt"
            result = RenderResultAdapter(TextRenderer())(Score(), RenderRequest(str(destination), "text"))
            self.assertEqual(destination.read_text(encoding="utf-8"), "notes=0")
            self.assertEqual(result, RenderResult(str(destination), "text", 7))
            self.assertEqual(list(Path(folder).iterdir()), [destination])
