"""Shared symbolic data used by replacement interfaces and implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from aimusic.core.core_types import BeatState


@dataclass(frozen=True)
class RenderResult:
    """A completed artifact and factual details about the written file."""

    output_path: str
    format: str
    size_bytes: int
    edo: int


@dataclass(frozen=True)
class CandidateRejection:
    """Traceable explanation for why a proposed BeatState was filtered out."""
    time_index: int
    source_state: BeatState
    candidate_state: BeatState
    reason: str


@dataclass(frozen=True)
class CandidateGenerationResult:
    """Deterministic candidate-generation output plus rejection diagnostics.

    Counts are tracked separately at every pipeline stage (see REQ-13):
    ``proposed`` -> ``unique`` -> ``legal`` (``states``) -> ``scored``.
    Retention against ``D_max`` happens one layer up, at the edge level in
    ``aimusic.planning.graph``, so it is intentionally not tracked here.
    """
    time_index: int
    source_state: BeatState
    states: Tuple[BeatState, ...]
    rejections: Tuple[CandidateRejection, ...] = ()
    proposed_count: int = 0
    unique_count: int = 0
    scores: Tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.scores and len(self.scores) != len(self.states):
            raise ValueError("scores must be empty or aligned 1:1 with states.")

    @property
    def legal_count(self) -> int:
        return len(self.states)

    @property
    def rejected_count(self) -> int:
        return len(self.rejections)

    @property
    def scored_count(self) -> int:
        return len(self.scores)


@dataclass(frozen=True)
class SymbolicNote:
    """A renderer-agnostic representation of a musical note."""

    pitch_height: int
    start_time: float
    end_time: float
    velocity: int = 64
    timbre: int | None = None
    pressure: int | None = None


@dataclass(frozen=True)
class TrackInstrumentConfig:
    """Optional per-track instrument override for score-based MIDI export."""

    program: int | None = None
    is_drum: bool = False
