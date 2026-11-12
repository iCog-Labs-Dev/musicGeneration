from pathlib import Path
from operator import setitem
import tempfile
import unittest
from typing import Callable

from aimusic.core.component_types import RenderRequest, TrackInstrumentConfig
from aimusic.core.config import EDOConfig
from aimusic.core.core_types import NoteEvent, Score
from aimusic.render.adapters import RenderResultAdapter
from aimusic.render.midi_adapter import MidiRenderer, MidiRenderSettings
from aimusic.render.midi_render import render_midi
from aimusic.theory.edo import EDO

untyped_setitem: Callable[..., None] = setitem


class TestMidiComponent(unittest.TestCase):
    def test_bound_instruments_are_copied_and_read_only(self) -> None:
        self.assertEqual(MidiRenderSettings(EDOConfig(n=19), {}), MidiRenderSettings(EDOConfig(n=19)))
        instruments = {"lead": TrackInstrumentConfig(88)}
        settings = MidiRenderSettings(EDOConfig(n=19), instruments)
        instruments["lead"] = TrackInstrumentConfig(5)
        assert settings.track_instruments is not None
        self.assertEqual(settings.track_instruments["lead"].program, 88)
        with self.assertRaises(TypeError):
            untyped_setitem(settings.track_instruments, "lead", TrackInstrumentConfig(5))

    def test_bound_midi_adapter_preserves_the_existing_renderer_bytes(self) -> None:
        score = Score(note_events=(NoteEvent(0, 480, 5, 0.6, track="lead"),))
        for tuning in (12, 19):
            with self.subTest(edo=tuning), tempfile.TemporaryDirectory() as folder:
                config = EDOConfig(n=tuning, base_tuning=0)
                original = Path(folder) / "original.mid"
                destination = Path(folder) / "result.mid"
                render_midi(score, EDO(config), str(original))
                result = RenderResultAdapter(MidiRenderer(MidiRenderSettings(config)))(
                    score, RenderRequest(str(destination), "midi"),
                )
                self.assertEqual(destination.read_bytes(), original.read_bytes())
                self.assertEqual(result.output_path, str(destination))
                self.assertEqual(result.size_bytes, len(original.read_bytes()))
                self.assertEqual(result.format, "midi")
                MidiRenderer(MidiRenderSettings(config, {}))(score, RenderRequest(str(destination), "midi"))
                self.assertEqual(destination.read_bytes(), original.read_bytes())

    def test_midi_backend_rejects_an_unsupported_format_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / "score.txt"
            with self.assertRaisesRegex(ValueError, "midi"):
                MidiRenderer(MidiRenderSettings(EDOConfig(n=12)))(
                    Score(), RenderRequest(str(destination), "text"),
                )
            self.assertEqual(list(Path(folder).iterdir()), [])
