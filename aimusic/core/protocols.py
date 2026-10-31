"""Typed replacement interfaces, independent of implementation imports."""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Protocol, Sequence

from aimusic.core.component_types import RenderResult, SymbolicNote, TrackInstrumentConfig
from aimusic.core.config import DecodeConfig
from aimusic.core.core_types import BeatState, Score
from aimusic.core.rng import RNGKey
from aimusic.core.vocab import Vocabularies

if TYPE_CHECKING:
    from aimusic.theory.edo import EDO


class TonalMetric(Protocol):
    """Finite nonnegative root/chord distances in the selected EDO."""

    def tonal_distance(self, a_root: int, b_root: int, edo: int = 12) -> float: ...

    def basic_space_distance(
        self,
        a_root: int,
        a_quality: str,
        b_root: int,
        b_quality: str,
        edo: int = 12,
    ) -> float: ...


class ScoreDecoder(Protocol):
    """A whole-path decoder returning a score and explicit next random key."""

    def __call__(
        self,
        path: Sequence[BeatState],
        *,
        key: RNGKey,
        decode_config: DecodeConfig | None = None,
        vocabularies: Vocabularies,
        edo: int = 12,
        ticks_per_beat: int = 480,
        tempo_bpm: float = 120.0,
        include_terminal_state: bool = False,
    ) -> tuple[Score, RNGKey]: ...


class ScoreRenderer(Protocol):
    """Legacy renderer callable, retaining the existing None return type."""

    def __call__(
        self,
        notes: Sequence[SymbolicNote] | Score,
        edo: EDO,
        output_path: str,
        ticks_per_beat: int = 480,
        *,
        track_instruments: Mapping[str, TrackInstrumentConfig] | None = None,
    ) -> None: ...


class ReportingScoreRenderer(Protocol):
    """Write the requested artifact and report its location and details."""

    def __call__(
        self,
        notes: Sequence[SymbolicNote] | Score,
        edo: EDO,
        output_path: str,
        ticks_per_beat: int = 480,
        *,
        track_instruments: Mapping[str, TrackInstrumentConfig] | None = None,
    ) -> RenderResult: ...


Renderer = ScoreRenderer | ReportingScoreRenderer
