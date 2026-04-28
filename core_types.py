from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import isclose, isfinite
from typing import TYPE_CHECKING, Any, Iterator, Tuple

if TYPE_CHECKING:
    from vocab import TokenVocabulary, Vocabularies


ExpressiveControls = Tuple[float, ...]


def _require_int(name: str, value: int, *, minimum: int | None = None) -> None:
    """Validate that a field is an integer and optionally above a floor."""
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")


def _require_real(name: str, value: float, *, minimum: float | None = None) -> None:
    """Validate that a field is a finite real number."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number.")
    if not isfinite(float(value)):
        raise ValueError(f"{name} must be finite.")
    if minimum is not None and float(value) < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")


def _safe_token_label(vocabulary: "TokenVocabulary[Any]", token_id: int) -> str | None:
    """Look up a token label while tolerating unknown ids."""
    if vocabulary.has_id(token_id):
        return vocabulary.token_for_id(token_id).label
    return None


def _format_token(field_name: str, token_id: int, label: str | None) -> str:
    """Render a token id in raw or label-aware form for logs."""
    if label is None:
        return f"{field_name}_id={token_id}"
    return f"{field_name}={label}[{token_id}]"


@dataclass(frozen=True)
class BeatState:
    """Canonical beat-level structural state.

    The project README defines:
      St = (meter_id, beat_in_bar, boundary_lvl, key_id, chord_id, role_id, head_id, groove_id)
    """

    meter_id: int
    beat_in_bar: int
    boundary_lvl: int
    key_id: int
    chord_id: int
    role_id: int
    head_id: int
    groove_id: int

    def __post_init__(self) -> None:
        for name in (
            "meter_id",
            "beat_in_bar",
            "boundary_lvl",
            "key_id",
            "chord_id",
            "role_id",
            "head_id",
            "groove_id",
        ):
            _require_int(name, getattr(self, name), minimum=0)

    def token_labels(self, vocabularies: "Vocabularies") -> dict[str, str | None]:
        """Resolve human-readable labels for the state's structural token ids."""
        return {
            "meter": _safe_token_label(vocabularies.meters, self.meter_id),
            "beat": _safe_token_label(vocabularies.beat_positions, self.beat_in_bar),
            "boundary": _safe_token_label(vocabularies.boundaries, self.boundary_lvl),
            "key": _safe_token_label(vocabularies.keys, self.key_id),
            "chord": _safe_token_label(vocabularies.chords, self.chord_id),
            "role": _safe_token_label(vocabularies.roles, self.role_id),
            "head": _safe_token_label(vocabularies.heads, self.head_id),
            "groove": _safe_token_label(vocabularies.grooves, self.groove_id),
        }

    def to_dict(self, vocabularies: "Vocabularies | None" = None) -> dict[str, object]:
        """Serialize the structural state to a log/test-friendly mapping."""
        data: dict[str, object] = {
            "meter_id": self.meter_id,
            "beat_in_bar": self.beat_in_bar,
            "boundary_lvl": self.boundary_lvl,
            "key_id": self.key_id,
            "chord_id": self.chord_id,
            "role_id": self.role_id,
            "head_id": self.head_id,
            "groove_id": self.groove_id,
        }
        if vocabularies is not None:
            labels = self.token_labels(vocabularies)
            data.update(
                {
                    "meter_label": labels["meter"],
                    "beat_label": labels["beat"],
                    "boundary_label": labels["boundary"],
                    "key_label": labels["key"],
                    "chord_label": labels["chord"],
                    "role_label": labels["role"],
                    "head_label": labels["head"],
                    "groove_label": labels["groove"],
                }
            )
        return data

    def pretty(self, vocabularies: "Vocabularies | None" = None) -> str:
        """Return a compact human-readable representation for logs."""
        if vocabularies is None:
            return (
                "BeatState("
                f"meter_id={self.meter_id}, "
                f"beat_in_bar={self.beat_in_bar}, "
                f"boundary_lvl={self.boundary_lvl}, "
                f"key_id={self.key_id}, "
                f"chord_id={self.chord_id}, "
                f"role_id={self.role_id}, "
                f"head_id={self.head_id}, "
                f"groove_id={self.groove_id})"
            )

        labels = self.token_labels(vocabularies)
        rendered = ", ".join(
            (
                _format_token("meter", self.meter_id, labels["meter"]),
                _format_token("beat", self.beat_in_bar, labels["beat"]),
                _format_token("boundary", self.boundary_lvl, labels["boundary"]),
                _format_token("key", self.key_id, labels["key"]),
                _format_token("chord", self.chord_id, labels["chord"]),
                _format_token("role", self.role_id, labels["role"]),
                _format_token("head", self.head_id, labels["head"]),
                _format_token("groove", self.groove_id, labels["groove"]),
            )
        )
        return f"BeatState({rendered})"


@dataclass(frozen=True)
class NoteEvent:
    """Canonical score-level symbolic note event.

    Matches the design spec fields:
      (ton, toff, h, v, e, track)
    """

    ton: int
    toff: int
    h: int
    v: float
    e: ExpressiveControls = ()
    track: str = "default"

    def __post_init__(self) -> None:
        _require_int("ton", self.ton, minimum=0)
        _require_int("toff", self.toff, minimum=0)
        if self.toff <= self.ton:
            raise ValueError("toff must be > ton.")
        _require_int("h", self.h)
        _require_real("v", self.v, minimum=0.0)
        if self.v > 1.0:
            raise ValueError("v must be <= 1.0.")

        expressive_controls = tuple(self.e)
        for idx, value in enumerate(expressive_controls):
            _require_real(f"e[{idx}]", value)
        object.__setattr__(self, "e", expressive_controls)

        if not isinstance(self.track, str) or not self.track.strip():
            raise ValueError("track must be a non-empty string.")

    def to_dict(self) -> dict[str, object]:
        """Serialize the note event to a JSON-friendly mapping."""
        return {
            "ton": self.ton,
            "toff": self.toff,
            "duration_ticks": self.toff - self.ton,
            "h": self.h,
            "v": self.v,
            "e": list(self.e),
            "track": self.track,
        }

    def pretty(self) -> str:
        """Return a compact, readable note-event summary."""
        return (
            "NoteEvent("
            f"track={self.track}, "
            f"ticks={self.ton}->{self.toff}, "
            f"h={self.h}, "
            f"v={self.v:.3f}, "
            f"e={list(self.e)})"
        )


@dataclass(frozen=True)
class Score:
    """Immutable symbolic score represented as note events."""

    note_events: Tuple[NoteEvent, ...] = ()
    ticks_per_beat: int = 480
    tempo_bpm: float = 120.0

    def __post_init__(self) -> None:
        note_events = tuple(self.note_events)
        if any(not isinstance(event, NoteEvent) for event in note_events):
            raise TypeError("note_events must contain only NoteEvent instances.")
        object.__setattr__(self, "note_events", note_events)

        _require_int("ticks_per_beat", self.ticks_per_beat, minimum=1)
        _require_real("tempo_bpm", self.tempo_bpm, minimum=0.0)
        if self.tempo_bpm == 0.0:
            raise ValueError("tempo_bpm must be > 0.")

    def __iter__(self) -> Iterator[NoteEvent]:
        return iter(self.note_events)

    def __len__(self) -> int:
        return len(self.note_events)

    def track_event_counts(self) -> dict[str, int]:
        """Return stable per-track event counts for diagnostics."""
        counts = Counter(event.track for event in self.note_events)
        return dict(sorted(counts.items()))

    def to_dict(self) -> dict[str, object]:
        """Serialize the score and all note events."""
        return {
            "event_count": len(self),
            "ticks_per_beat": self.ticks_per_beat,
            "tempo_bpm": self.tempo_bpm,
            "track_event_counts": self.track_event_counts(),
            "note_events": [event.to_dict() for event in self.note_events],
        }

    def pretty(self, *, max_events: int = 3) -> str:
        """Return a concise score summary with an event preview."""
        preview_events = ", ".join(event.pretty() for event in self.note_events[:max_events])
        if len(self.note_events) > max_events:
            preview_events = f"{preview_events}, ..."
        track_counts = ", ".join(
            f"{track}:{count}" for track, count in self.track_event_counts().items()
        )
        return (
            "Score("
            f"events={len(self)}, "
            f"tempo_bpm={self.tempo_bpm:.1f}, "
            f"ticks_per_beat={self.ticks_per_beat}, "
            f"tracks={{{track_counts}}}, "
            f"preview=[{preview_events}])"
        )


@dataclass(frozen=True)
class Layer:
    """Immutable collection of candidate BeatStates at a given beat index."""

    time_index: int
    states: Tuple[BeatState, ...]

    def __post_init__(self) -> None:
        _require_int("time_index", self.time_index, minimum=0)

        states = tuple(self.states)
        if any(not isinstance(state, BeatState) for state in states):
            raise TypeError("states must contain only BeatState instances.")
        if len(states) != len(set(states)):
            raise ValueError("Layer states must be unique.")
        object.__setattr__(self, "states", states)

    def __iter__(self) -> Iterator[BeatState]:
        return iter(self.states)

    def __len__(self) -> int:
        return len(self.states)

    def to_dict(self, vocabularies: "Vocabularies | None" = None) -> dict[str, object]:
        """Serialize the layer and its candidate states."""
        return {
            "time_index": self.time_index,
            "size": len(self),
            "states": [state.to_dict(vocabularies) for state in self.states],
        }

    def pretty(self, vocabularies: "Vocabularies | None" = None, *, max_states: int = 3) -> str:
        """Return a compact layer summary for logs."""
        preview = ", ".join(
            state.pretty(vocabularies) for state in self.states[:max_states]
        )
        if len(self.states) > max_states:
            preview = f"{preview}, ..."
        return f"Layer(t={self.time_index}, size={len(self)}, states=[{preview}])"


@dataclass(frozen=True)
class Edge:
    """Immutable transition edge between two BeatStates."""

    time_index: int
    source: BeatState
    target: BeatState
    log_weight: float

    def __post_init__(self) -> None:
        _require_int("time_index", self.time_index, minimum=0)
        if not isinstance(self.source, BeatState):
            raise TypeError("source must be a BeatState.")
        if not isinstance(self.target, BeatState):
            raise TypeError("target must be a BeatState.")
        _require_real("log_weight", self.log_weight)

    def to_dict(self, vocabularies: "Vocabularies | None" = None) -> dict[str, object]:
        """Serialize the edge and its endpoints."""
        return {
            "time_index": self.time_index,
            "log_weight": self.log_weight,
            "source": self.source.to_dict(vocabularies),
            "target": self.target.to_dict(vocabularies),
        }

    def pretty(self, vocabularies: "Vocabularies | None" = None) -> str:
        """Return a compact human-readable edge description."""
        return (
            "Edge("
            f"t={self.time_index}, "
            f"log_weight={self.log_weight:.3f}, "
            f"source={self.source.pretty(vocabularies)}, "
            f"target={self.target.pretty(vocabularies)})"
        )


@dataclass(frozen=True)
class EndpointDistribution:
    """Normalized probability distribution over a specific graph layer."""

    layer: Layer
    probabilities: Tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.layer, Layer):
            raise TypeError("layer must be a Layer.")
        if len(self.layer) == 0:
            raise ValueError("EndpointDistribution layer must not be empty.")

        probabilities = tuple(self.probabilities)
        if len(probabilities) != len(self.layer):
            raise ValueError("probabilities must align 1:1 with the layer states.")
        for idx, prob in enumerate(probabilities):
            _require_real(f"probabilities[{idx}]", prob, minimum=0.0)

        total = float(sum(probabilities))
        if not isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-6):
            raise ValueError("EndpointDistribution probabilities must sum to 1.0.")

        object.__setattr__(self, "probabilities", probabilities)

    def probability_of(self, state: BeatState) -> float:
        """Return the probability mass of *state*, or 0.0 if absent."""
        for candidate, probability in zip(self.layer.states, self.probabilities):
            if candidate == state:
                return probability
        return 0.0

    def to_dict(self, vocabularies: "Vocabularies | None" = None) -> dict[str, object]:
        """Serialize the endpoint distribution and its support."""
        return {
            "time_index": self.layer.time_index,
            "support_size": len(self.layer),
            "support": [
                {
                    "state": state.to_dict(vocabularies),
                    "probability": probability,
                }
                for state, probability in zip(self.layer.states, self.probabilities)
            ],
        }

    def pretty(self, vocabularies: "Vocabularies | None" = None, *, max_states: int = 3) -> str:
        """Return a compact endpoint-distribution summary."""
        support_preview = ", ".join(
            (
                f"{state.pretty(vocabularies)}@{probability:.3f}"
                for state, probability in zip(
                    self.layer.states[:max_states],
                    self.probabilities[:max_states],
                )
            )
        )
        if len(self.layer) > max_states:
            support_preview = f"{support_preview}, ..."
        return (
            "EndpointDistribution("
            f"t={self.layer.time_index}, "
            f"size={len(self.layer)}, "
            f"support=[{support_preview}])"
        )
