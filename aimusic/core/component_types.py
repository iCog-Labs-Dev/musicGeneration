"""Shared symbolic data used by replacement interfaces and implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from aimusic.core.core_types import BeatState, _require_int, _require_real


def _nonblank(name: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value.strip():
        raise ValueError(f"{name} must not be blank.")


@dataclass(frozen=True)
class RenderRequest:
    """Destination and format, independent of any rendering backend."""

    output_path: str
    format: str

    def __post_init__(self) -> None:
        _nonblank("output_path", self.output_path)
        _nonblank("format", self.format)


@dataclass(frozen=True)
class RenderResult:
    """A completed artifact and factual details about the written file."""

    output_path: str
    format: str
    size_bytes: int

    def __post_init__(self) -> None:
        _nonblank("output_path", self.output_path)
        _nonblank("format", self.format)
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            raise TypeError("size_bytes must be an int.")
        if self.size_bytes < 0:
            raise ValueError("size_bytes must be nonnegative.")


@dataclass(frozen=True)
class CandidateRejection:
    """Traceable explanation for why a proposed BeatState was filtered out."""
    time_index: int
    source_state: BeatState
    candidate_state: BeatState
    reason: str

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        _require_int("time_index", self.time_index, minimum=0)
        if not isinstance(self.source_state, BeatState) or not isinstance(self.candidate_state, BeatState):
            raise TypeError("rejection source and candidate must be BeatState values.")
        _nonblank("reason", self.reason)


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
        self.validate()

    def validate(self) -> None:
        _require_int("time_index", self.time_index, minimum=0)
        if not isinstance(self.source_state, BeatState):
            raise TypeError("source_state must be a BeatState.")
        if not isinstance(self.states, tuple) or any(not isinstance(state, BeatState) for state in self.states):
            raise TypeError("states must be a tuple of BeatState values.")
        if len(set(self.states)) != len(self.states):
            raise ValueError("accepted states must be unique.")
        if not isinstance(self.rejections, tuple) or any(not isinstance(item, CandidateRejection) for item in self.rejections):
            raise TypeError("rejections must be a tuple of CandidateRejection values.")
        rejected_states = set()
        for rejection in self.rejections:
            rejection.validate()
            if rejection.source_state != self.source_state or rejection.time_index != self.time_index:
                raise ValueError("rejection source and time must match the candidate result.")
            if rejection.candidate_state in self.states or rejection.candidate_state in rejected_states:
                raise ValueError("accepted and rejected outcomes must be distinct unique states.")
            rejected_states.add(rejection.candidate_state)
        _require_int("proposed_count", self.proposed_count, minimum=0)
        _require_int("unique_count", self.unique_count, minimum=0)
        if self.proposed_count < self.unique_count:
            raise ValueError("proposed_count must be at least unique_count.")
        if self.unique_count != len(self.states) + len(self.rejections):
            raise ValueError("unique_count must equal accepted plus rejected outcomes.")
        if not isinstance(self.scores, tuple):
            raise TypeError("scores must be a tuple.")
        if self.scores and len(self.scores) != len(self.states):
            raise ValueError("scores must be empty or aligned 1:1 with states.")
        for score in self.scores:
            _require_real("candidate score", score)

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

    def __post_init__(self) -> None:
        _require_int("pitch_height", self.pitch_height)
        _require_real("start_time", self.start_time, minimum=0)
        _require_real("end_time", self.end_time, minimum=0)
        if self.end_time <= self.start_time:
            raise ValueError("end_time must be later than start_time.")
        _midi_value("velocity", self.velocity)
        if self.timbre is not None:
            _midi_value("timbre", self.timbre)
        if self.pressure is not None:
            _midi_value("pressure", self.pressure)


@dataclass(frozen=True)
class TrackInstrumentConfig:
    """Optional per-track instrument override for score-based MIDI export."""

    program: int | None = None
    is_drum: bool = False

    def __post_init__(self) -> None:
        if self.program is not None:
            _midi_value("program", self.program)
        if not isinstance(self.is_drum, bool):
            raise TypeError("is_drum must be a bool.")


def _midi_value(name: str, value: int) -> None:
    _require_int(name, value, minimum=0)
    if value > 127:
        raise ValueError(f"{name} must be <= 127.")
