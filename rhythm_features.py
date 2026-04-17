from __future__ import annotations

from typing import Mapping, Optional, Sequence

from core_types import BeatState
from gttm_features import (
    DEFAULT_METERS,
    MeterSpec,
    W_BOUNDARY_ON_STRONG,
    W_BOUNDARY_ON_WEAK_PENALTY,
    W_GROOVE_CONTINUITY,
    W_ILLEGAL_BEAT_PENALTY,
    W_METER_STABILITY,
    W_STRONG_BEAT_BIAS,
    beat_position_penalty,
    beats_per_bar,
    boundary_alignment_score,
    groove_continuity_feature,
    grouping_window_score,
    is_strong_beat,
    meter_stability_feature,
    strong_beat_alignment,
)
from vocab import DEFAULT_VOCABULARIES


def _legacy_groove_continuity(a: BeatState, b: BeatState) -> float:
    grooves = DEFAULT_VOCABULARIES.grooves
    if grooves.has_id(a.groove_id) and grooves.has_id(b.groove_id):
        return groove_continuity_feature(
            a,
            b,
            0,
            vocabularies=DEFAULT_VOCABULARIES,
        )
    return 1.0 if a.groove_id == b.groove_id else -0.5


def illegal_beat_penalty(
    st: BeatState,
    meters: Optional[Mapping[int, MeterSpec]] = None,
) -> float:
    """Compatibility wrapper around the canonical BeatState meter check."""
    return beat_position_penalty(st, meters)


def strong_beat_bias(
    st: BeatState,
    meters: Optional[Mapping[int, MeterSpec]] = None,
) -> float:
    """Compatibility wrapper around the canonical strong-beat bias helper."""
    return strong_beat_alignment(st, meters)


def boundary_score(
    st: BeatState,
    meters: Optional[Mapping[int, MeterSpec]] = None,
) -> float:
    """Compatibility wrapper around the canonical boundary-placement helper."""
    return boundary_alignment_score(st, meters)


def transition_score(
    a: BeatState,
    b: BeatState,
    meters: Optional[Mapping[int, MeterSpec]] = None,
) -> float:
    """Legacy rhythm score composed from the canonical EPIC 2 feature helpers."""
    return float(
        (W_METER_STABILITY * meter_stability_feature(a, b, 0))
        + (W_GROOVE_CONTINUITY * _legacy_groove_continuity(a, b))
        + boundary_score(b, meters)
        + strong_beat_bias(b, meters)
        + illegal_beat_penalty(b, meters)
        + (0.25 * illegal_beat_penalty(a, meters))
    )


def local_window_score(
    prev: Optional[BeatState],
    curr: BeatState,
    next_: Optional[BeatState],
    meters: Optional[Mapping[int, MeterSpec]] = None,
) -> float:
    """Legacy local grouping score backed by the canonical grouping helper."""
    return grouping_window_score(prev, curr, next_, meters)


def sequence_score(
    seq: Sequence[BeatState],
    meters: Optional[Mapping[int, MeterSpec]] = None,
) -> float:
    if len(seq) < 2:
        return 0.0

    total = sum(transition_score(seq[i], seq[i + 1], meters) for i in range(len(seq) - 1))
    total += sum(
        local_window_score(seq[i - 1], seq[i], seq[i + 1], meters)
        for i in range(1, len(seq) - 1)
    )
    return float(total)
