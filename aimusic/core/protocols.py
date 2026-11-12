"""Typed replacement interfaces, independent of implementation imports."""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from aimusic.core.component_types import RenderRequest, RenderResult
from aimusic.core.config import DecodeConfig
from aimusic.core.core_types import BeatState, Score
from aimusic.core.rng import RNGKey
from aimusic.core.vocab import Vocabularies

@runtime_checkable
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


@runtime_checkable
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


@runtime_checkable
class ScoreRenderer(Protocol):
    """Write a score in the requested format and report the completed artifact."""

    def __call__(self, score: Score, request: RenderRequest) -> RenderResult: ...
