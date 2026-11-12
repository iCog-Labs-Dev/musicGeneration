"""Bind MIDI settings behind the format-neutral score renderer contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from aimusic.core.component_types import RenderRequest, RenderResult, TrackInstrumentConfig
from aimusic.core.config import EDOConfig
from aimusic.core.core_types import Score
from aimusic.render.midi_render import render_midi
from aimusic.theory.edo import EDO


@dataclass(frozen=True)
class MidiRenderSettings:
    edo_config: EDOConfig
    track_instruments: Mapping[str, TrackInstrumentConfig] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.edo_config, EDOConfig):
            raise TypeError("edo_config must be an EDOConfig.")
        if self.track_instruments is not None:
            if not isinstance(self.track_instruments, Mapping):
                raise TypeError("track_instruments must be a mapping.")
            for name, config in self.track_instruments.items():
                if not isinstance(name, str) or not name.strip():
                    raise ValueError("instrument track names must be nonblank strings.")
                if not isinstance(config, TrackInstrumentConfig):
                    raise TypeError("instrument settings must be TrackInstrumentConfig values.")
            instruments = dict(self.track_instruments)
            object.__setattr__(self, "track_instruments", MappingProxyType(instruments) if instruments else None)

    def to_dict(self) -> dict[str, object]:
        return {
            "edo": self.edo_config.n,
            "base_tuning": self.edo_config.base_tuning,
            "pitch_bend_range": self.edo_config.pitch_bend_range,
            "rendering_method": self.edo_config.microtonal_rendering_method.name,
            "track_instruments": None if self.track_instruments is None else {
                name: {"program": config.program, "is_drum": config.is_drum}
                for name, config in self.track_instruments.items()
            },
        }


@dataclass(frozen=True)
class MidiRenderer:
    midi_settings: MidiRenderSettings

    def __post_init__(self) -> None:
        if not isinstance(self.midi_settings, MidiRenderSettings):
            raise TypeError("midi_settings must be MidiRenderSettings.")

    def __call__(self, score: Score, request: RenderRequest) -> RenderResult:
        if request.format != "midi":
            raise ValueError("MidiRenderer supports only the midi format.")
        render_midi(
            score, EDO(self.midi_settings.edo_config), request.output_path,
            track_instruments=self.midi_settings.track_instruments,
        )
        return RenderResult(request.output_path, request.format, Path(request.output_path).stat().st_size)
