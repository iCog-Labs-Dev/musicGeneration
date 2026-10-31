"""Result reporting around existing renderer callables."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Mapping, Sequence

from aimusic.core.component_types import RenderResult, SymbolicNote, TrackInstrumentConfig
from aimusic.core.core_types import Score
from aimusic.core.protocols import Renderer
from aimusic.render.midi_render import render_midi
from aimusic.theory.edo import EDO


@dataclass(frozen=True)
class RenderResultAdapter:
    """Keep legacy inputs/returns intact and report the actual MIDI artifact."""

    renderer: Renderer = render_midi

    def __call__(
        self,
        notes: Sequence[SymbolicNote] | Score,
        edo: EDO,
        output_path: str,
        ticks_per_beat: int = 480,
        *,
        track_instruments: Mapping[str, TrackInstrumentConfig] | None = None,
    ) -> RenderResult:
        destination = Path(output_path)
        with TemporaryDirectory(prefix=".aimusic-render-", dir=destination.parent) as temporary:
            staged = Path(temporary) / destination.name
            reported = self.renderer(
                notes, edo, str(staged), ticks_per_beat,
                track_instruments=track_instruments,
            )
            actual = RenderResult(str(staged), "midi", staged.stat().st_size, edo.config.n)
            if reported is not None and reported != actual:
                raise ValueError("renderer result must match the requested MIDI file and its actual details.")
            staged.replace(destination)
            return replace(actual, output_path=output_path)
