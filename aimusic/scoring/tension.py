""" tension.py — versioned tension model (TENSION_MODEL_VERSION = "1.0.0").

Replaces dead/duplicated role+boundary-only formulas with one pure function
adding key/chord motion via tonal.py. Per-transition, not per-state: index i
means "tension of arriving at path[i]". Doesn't reuse gttm_features.py's
unbounded proximity scores (different semantics) though shares its decay shape.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple, Union

from aimusic.core.config import SectionStyleWeights
from aimusic.core.core_types import BeatState
from aimusic.core.vocab import Vocabularies
from aimusic.theory.tonal import basic_space_distance, tonal_distance

if TYPE_CHECKING:  # pragma: no cover - import-time cycle avoidance only
    from aimusic.planning.plans import PlanningSection
    from aimusic.planning.sections import SectionTimeline

TENSION_MODEL_VERSION = "1.0.0"

# Role labels are documented in aimusic/core/vocab.py:388-391. Values are
# ordered hold < prep < change < cad — a monotonically increasing sense of
# harmonic urgency — and are the base contribution before boundary/tonal
# terms are added.
_ROLE_BASE_TENSION: Dict[str, float] = {
    "hold": 0.0,
    "prep": 0.33,
    "change": 0.66,
    "cad": 1.0,
}


@dataclass(frozen=True)
class TensionWeights:
    """Weights for each contribution to `beat_tension`."""

    role: float = 0.40
    boundary: float = 0.15
    key_motion: float = 0.20
    chord_motion: float = 0.25
    head_groove: float = 0.0

    def __post_init__(self) -> None:
        for name in ("role", "boundary", "key_motion", "chord_motion", "head_groove"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"TensionWeights.{name} must be a real number.")
            if value < 0.0:
                raise ValueError(f"TensionWeights.{name} must be >= 0.")


DEFAULT_WEIGHTS = TensionWeights()


def _role_label(state: BeatState, vocabularies: Vocabularies) -> str:
    return vocabularies.roles.token_for_id(state.role_id).label


def _max_boundary_level(vocabularies: Vocabularies) -> int:
    return max(token.level for token in vocabularies.boundaries.tokens)


def _role_component(state: BeatState, vocabularies: Vocabularies) -> float:
    label = _role_label(state, vocabularies)
    return _ROLE_BASE_TENSION.get(label, 0.5)


def _boundary_component(state: BeatState, vocabularies: Vocabularies) -> float:
    max_level = _max_boundary_level(vocabularies)
    if max_level <= 0:
        return 0.0
    return min(1.0, state.boundary_lvl / max_level)


def _decay_normalize(distance: float) -> float:
    """Map a non-negative distance to [0, 1) via distance / (1 + distance)."""
    return distance / (1.0 + distance)


def _key_motion_component(
    prev_state: Optional[BeatState],
    state: BeatState,
    vocabularies: Vocabularies,
    edo: int,
) -> float:
    if prev_state is None:
        return 0.0
    prev_root = vocabularies.keys.token_for_id(prev_state.key_id).root_pc
    root = vocabularies.keys.token_for_id(state.key_id).root_pc
    distance = tonal_distance(prev_root, root, edo)
    return _decay_normalize(distance)


def _chord_motion_component(
    prev_state: Optional[BeatState],
    state: BeatState,
    vocabularies: Vocabularies,
    edo: int,
) -> float:
    if prev_state is None:
        return 0.0
    prev_chord = vocabularies.chords.token_for_id(prev_state.chord_id)
    chord = vocabularies.chords.token_for_id(state.chord_id)
    distance = basic_space_distance(
        prev_chord.root_pc, prev_chord.quality,
        chord.root_pc, chord.quality,
        edo,
    )
    return _decay_normalize(distance)


def _head_groove_component(
    prev_state: Optional[BeatState],
    state: BeatState,
    vocabularies: Vocabularies,
) -> float:
    if prev_state is None:
        return 0.0
    changed = 0.0
    if prev_state.head_id != state.head_id:
        changed += 0.5
    if prev_state.groove_id != state.groove_id:
        changed += 0.5
    return min(1.0, changed)


def beat_tension(
    prev_state: Optional[BeatState],
    state: BeatState,
    vocabularies: Vocabularies,
    edo: int,
    weights: TensionWeights = DEFAULT_WEIGHTS,
) -> float:
    """Compute tension in [0, 1] for `state`, given the beat that precedes it.

    Combines, as a weighted sum:
      - role component: authored harmonic function (hold < prep < change < cad)
      - boundary component: structural boundary level, normalized by the
        vocabulary's max boundary level
      - key motion component: circle-of-fifths distance (Lerdahl's *j*)
        between the previous and current key root, normalized by edo // 2
      - chord motion component: full Lerdahl TPS distance (*j + k*) between
        the previous and current chord, normalized by a generous fixed cap
      - head/groove component (off by default): whether the head or groove
        token changed beat-to-beat

    """
    role = _role_component(state, vocabularies)
    boundary = _boundary_component(state, vocabularies)
    key_motion = _key_motion_component(prev_state, state, vocabularies, edo)
    chord_motion = _chord_motion_component(prev_state, state, vocabularies, edo)
    head_groove = _head_groove_component(prev_state, state, vocabularies)

    total = (
        weights.role * role
        + weights.boundary * boundary
        + weights.key_motion * key_motion
        + weights.chord_motion * chord_motion
        + weights.head_groove * head_groove
    )
    return max(0.0, min(1.0, total))


def realized_tension_curve(
    path: Sequence[BeatState],
    vocabularies: Vocabularies,
    edo: int,
    weights: TensionWeights = DEFAULT_WEIGHTS,
) -> List[Tuple[float, float]]:
    """Time-indexed realized tension curve from a selected BeatState path.

    One (time, tension) sample per beat in `path`, time indexed 0..len-1 to
    match `aimusic.app.cli._segment_timeline`'s indexing of the other
    structural timelines.
    """
    curve: List[Tuple[float, float]] = []
    prev_state: Optional[BeatState] = None
    for index, state in enumerate(path):
        tension = beat_tension(prev_state, state, vocabularies, edo, weights)
        curve.append((float(index), tension))
        prev_state = state
    return curve


def target_tension_curve(
    sections: Union["SectionTimeline", Sequence["PlanningSection"]],
) -> List[Tuple[float, float]]:
    """Time-indexed target tension curve sampled from SectionTimeline.

    Delegates to SectionTimeline to eliminate duplicated interpolation logic.
    The curve index k is the ARRIVAL beat (same as realized_tension_curve),
    covering beats 0..total_beats so edge arrival states k = t + 1 align point-for-point.
    """
    from aimusic.planning.sections import SectionTimeline

    if isinstance(sections, SectionTimeline):
        timeline = sections
    else:
        timeline = SectionTimeline(tuple(sections), total_beats=sections[-1].end_time)
    return [(float(k), float(v)) for k, v in timeline.as_curve()]


def target_tension_at_time(
    sections: Union["SectionTimeline", Sequence["PlanningSection"]],
    t: int,
) -> float:
    """Return interpolated target tension value for beat state index t across sections.

    Delegates to SectionTimeline to eliminate duplicated interpolation logic.
    Index t is the beat state index (arrival beat convention k = t + 1).
    """
    from aimusic.planning.sections import SectionTimeline

    if isinstance(sections, SectionTimeline):
        timeline = sections
    else:
        timeline = SectionTimeline(tuple(sections), total_beats=sections[-1].end_time)
    return float(timeline.target_tension(t))


def transition_target_tension_energy(
    tau_hat: float | Optional[BeatState],
    tau_star: float | BeatState,
    sigma: float | int = 0.1,
    *args: Any,
    **kwargs: Any,
) -> float:
    """Compute target tension deviation energy: (tau_hat - tau_star)^2 / (2 * sigma^2).

    sigma is the tolerance in tension units: a deviation of sigma costs exactly
    0.5 nat before lambda scaling (Decision D2).
    """
    if (
        isinstance(tau_hat, (int, float))
        and not isinstance(tau_hat, bool)
        and isinstance(tau_star, (int, float))
        and not isinstance(tau_star, bool)
    ):
        if not isinstance(sigma, (int, float)) or isinstance(sigma, bool):
            raise TypeError("sigma must be a real number.")
        sig = float(sigma)
        if sig <= 0.0:
            raise ValueError("sigma must be > 0.")
        diff = float(tau_hat - tau_star)
        return float((diff * diff) / (2.0 * sig * sig))

    # Backward compatibility fallback if called with legacy positional signature:
    # (prev_state, state, t, target_tension, vocabularies, edo, ...)
    prev_state = tau_hat  # type: ignore[assignment]
    state = tau_star  # type: ignore[assignment]
    target_tension = args[0] if len(args) > 0 else kwargs.get("target_tension", 0.5)
    vocabs = args[1] if len(args) > 1 else kwargs.get("vocabularies")
    edo = args[2] if len(args) > 2 else kwargs.get("edo", 12)
    weights = args[3] if len(args) > 3 else kwargs.get("weights", DEFAULT_WEIGHTS)
    realized = beat_tension(prev_state, state, vocabs, edo, weights)  # type: ignore[arg-type]
    diff = realized - target_tension
    return float((diff * diff) / (2.0 * 0.1 * 0.1))


def _extract_section_context(
    section_view: Any,
    *,
    is_section_start: Optional[bool] = None,
    multi_meter: Optional[bool] = None,
) -> Tuple[Optional[Any], bool, bool]:
    """Extract (section, is_section_start, multi_meter) from section_view or explicit flags."""
    import aimusic.planning.sections as _sec_mod

    sec: Optional[Any] = None
    start = False
    mm = False

    if isinstance(section_view, _sec_mod.EdgeSectionView):
        sec = section_view.section
        start = section_view.is_section_start
        mm = getattr(section_view, "multi_meter", False)
    elif isinstance(section_view, _sec_mod.PlanningSection):
        sec = section_view
        start = False
        mm = False

    if is_section_start is not None:
        start = is_section_start
    if multi_meter is not None:
        mm = multi_meter
    elif sec is not None and sec.allowed_meters and len(sec.allowed_meters) > 1:
        mm = True

    return sec, start, mm


def section_style_breakdown(
    prev_state: Optional[BeatState],
    next_state: BeatState,
    section_view: Any,
    vocabularies: Vocabularies,
    edo: int = 12,
    style_weights: Optional[Any] = None,
    *,
    is_section_start: Optional[bool] = None,
) -> Dict[str, float]:
    """Return an ordered mapping of non-negative component energies for section-style deviation."""
    section, is_start, _ = _extract_section_context(
        section_view, is_section_start=is_section_start
    )

    if section is None:
        return {
            "meter": 0.0,
            "key_region": 0.0,
            "groove_family": 0.0,
            "role": 0.0,
            "boundary": 0.0,
            "start_alignment": 0.0,
        }

    # 1. meter: 1 if next meter not in section.allowed_meters else 0
    meter_comp = 0.0
    if section.allowed_meters:
        if vocabularies.meters.has_id(next_state.meter_id):
            meter_label = vocabularies.meters.token_for_id(next_state.meter_id).label
            if meter_label not in section.allowed_meters:
                meter_comp = 1.0
        else:
            meter_comp = 1.0

    # 2. key_region: min over region roots of d/(1+d), d = tonal_distance(next key root, region root, edo); 0 when no region
    key_comp = 0.0
    region = section.key_region
    if region:
        if vocabularies.keys.has_id(next_state.key_id):
            next_root = vocabularies.keys.token_for_id(next_state.key_id).root_pc
            d_min = min(tonal_distance(next_root, r, edo) for r in region)
            key_comp = float(d_min / (1.0 + d_min))
        else:
            key_comp = 0.5

    # 3. groove_family: 1 if groove family differs from section.groove_family else 0
    groove_comp = 0.0
    if section.groove_family:
        if vocabularies.grooves.has_id(next_state.groove_id):
            groove_tok = vocabularies.grooves.token_for_id(next_state.groove_id)
            if groove_tok.family != section.groove_family:
                groove_comp = 1.0
        else:
            groove_comp = 1.0

    # 4. role: 1 if role label is not in section.preferred_roles else 0
    role_comp = 0.0
    if section.preferred_roles:
        if vocabularies.roles.has_id(next_state.role_id):
            role_label = vocabularies.roles.token_for_id(next_state.role_id).label
            if role_label not in section.preferred_roles:
                role_comp = 1.0
        else:
            role_comp = 1.0

    # 5. boundary: at section-start arrival state, abs(next.boundary_lvl - section.boundary_level);
    # in interior, 1 if next.boundary_lvl >= section.boundary_level and section.boundary_level >= 2 else 0
    if is_start:
        boundary_comp = float(abs(next_state.boundary_lvl - section.boundary_level))
    else:
        boundary_comp = (
            1.0
            if (next_state.boundary_lvl >= section.boundary_level and section.boundary_level >= 2)
            else 0.0
        )

    # 6. start_alignment: at section-start arrival state, 1 if next.beat_in_bar != 0 else 0; 0 elsewhere
    if is_start:
        start_alignment_comp = 1.0 if next_state.beat_in_bar != 0 else 0.0
    else:
        start_alignment_comp = 0.0

    return {
        "meter": meter_comp,
        "key_region": key_comp,
        "groove_family": groove_comp,
        "role": role_comp,
        "boundary": boundary_comp,
        "start_alignment": start_alignment_comp,
    }


def section_style_energy(
    prev_state: Optional[BeatState],
    next_state: BeatState,
    section_view: Any,
    vocabularies: Vocabularies,
    edo: int = 12,
    style_weights: Optional[Any] = None,
    *,
    multi_meter: Optional[bool] = None,
    multi_meter_alignment_factor: float = 3.0,
    is_section_start: Optional[bool] = None,
) -> float:
    """Compute section-style deviation energy penalty for transitioning into next_state.

    Returns sum(weight_c * component_c). Unknown or unset preferences contribute 0.
    """
    resolved_weights = SectionStyleWeights() if style_weights is None else style_weights
    section, is_start, is_multi = _extract_section_context(
        section_view, is_section_start=is_section_start, multi_meter=multi_meter
    )

    components = section_style_breakdown(
        prev_state,
        next_state,
        section_view,
        vocabularies,
        edo=edo,
        style_weights=resolved_weights,
        is_section_start=is_start,
    )

    alignment_factor = multi_meter_alignment_factor if is_multi else 1.0
    total = (
        resolved_weights.meter * components["meter"]
        + resolved_weights.key_region * components["key_region"]
        + resolved_weights.groove_family * components["groove_family"]
        + resolved_weights.role * components["role"]
        + resolved_weights.boundary * components["boundary"]
        + (resolved_weights.start_alignment * alignment_factor) * components["start_alignment"]
    )
    return float(total)




@dataclass(frozen=True)
class TensionDeviationReport:
    """Comparison of a target curve against a realized curve."""

    mean_absolute_error: float
    max_absolute_error: float
    section_errors: Dict[str, float]
    target_peak_time: float
    realized_peak_time: float
    peak_timing_offset: float
    shape_correlation: float


def _as_time_value_map(curve: Sequence[Tuple[float, float]]) -> Dict[float, float]:
    return {time: value for time, value in curve}


def _pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) < 2:
        return 0.0
    if len(set(xs)) < 2 or len(set(ys)) < 2:
        # A constant series has undefined correlation; treat as no signal.
        return 0.0
    return statistics.correlation(xs, ys)


def compare_tension_curves(
    target: Sequence[Tuple[float, float]],
    realized: Sequence[Tuple[float, float]],
    sections: Union["SectionTimeline", Sequence["PlanningSection"]],
) -> TensionDeviationReport:
    """Compare target vs. realized tension curves.

    Pure function: takes two curves and the section plan, returns a report.
    Comparison is over the intersection of time points present in both
    curves, aligning target and realized by arrival beat index k.
    """
    from aimusic.planning.sections import SectionTimeline

    if isinstance(sections, SectionTimeline):
        sec_list = sections.sections
    else:
        sec_list = tuple(sections)

    target_map = _as_time_value_map(target)
    realized_map = _as_time_value_map(realized)
    shared_times = sorted(set(target_map) & set(realized_map))

    if not shared_times:
        return TensionDeviationReport(
            mean_absolute_error=0.0,
            max_absolute_error=0.0,
            section_errors={},
            target_peak_time=0.0,
            realized_peak_time=0.0,
            peak_timing_offset=0.0,
            shape_correlation=0.0,
        )

    abs_errors = [abs(target_map[t] - realized_map[t]) for t in shared_times]
    mean_abs_error = sum(abs_errors) / len(abs_errors)
    max_abs_error = max(abs_errors)

    section_errors: Dict[str, float] = {}
    for section in sec_list:
        section_times = [
            t for t in shared_times
            if (section.start_time <= t < section.end_time)
            or (t == section.end_time and section == sec_list[-1])
        ]
        if not section_times:
            continue
        section_abs_errors = [abs(target_map[t] - realized_map[t]) for t in section_times]
        section_errors[section.name] = sum(section_abs_errors) / len(section_abs_errors)

    target_peak_time = max(shared_times, key=lambda t: target_map[t])
    realized_peak_time = max(shared_times, key=lambda t: realized_map[t])

    target_series = [target_map[t] for t in shared_times]
    realized_series = [realized_map[t] for t in shared_times]
    shape_correlation = _pearson_correlation(target_series, realized_series)

    return TensionDeviationReport(
        mean_absolute_error=mean_abs_error,
        max_absolute_error=max_abs_error,
        section_errors=section_errors,
        target_peak_time=target_peak_time,
        realized_peak_time=realized_peak_time,
        peak_timing_offset=realized_peak_time - target_peak_time,
        shape_correlation=shape_correlation,
    )
