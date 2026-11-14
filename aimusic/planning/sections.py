"""sections.py — Section timeline, plan validation, and time-index convention.

Follows REQ-04 and Decision D1:
  - state s_k belongs to section(k).
  - edge t scores the arrival state s_{t+1}.
  - every edge-level lookup (section, target tension, boundary rule) uses arrival beat k = t + 1.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from aimusic.core.config import StyleConfig
from aimusic.core.vocab import Vocabularies

_logger = logging.getLogger(__name__)


def _require_int(name: str, value: int, *, minimum: int = 0) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int.")
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")


def _require_real(name: str, value: float, *, minimum: float = 0.0) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number.")
    if float(value) < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")


class SectionPlanError(ValueError):
    """Raised when a section plan or section timeline violates structural rules."""


def key_ids_for_pcs(pcs: Sequence[int], vocabularies: Vocabularies) -> tuple[int, ...]:
    """Return the IDs of vocabularies.keys tokens whose root_pc is in pcs."""
    pc_set = set(pcs)
    return tuple(k.id for k in vocabularies.keys if k.root_pc in pc_set)


@dataclass(frozen=True)
class PlanningSection:
    """Single section descriptor for structural planning and generative section guidance."""

    name: str
    start_time: int
    end_time: int
    boundary_level: int
    target_tension_arc: Tuple[float, ...] = (0.2, 0.85, 0.25)
    allowed_meters: Optional[Tuple[str, ...]] = None
    groove_family: Optional[str] = None
    target_key_id: Optional[int] = None
    key_region: Optional[Tuple[int, ...]] = None
    target_density: Optional[float] = None
    preferred_roles: Optional[Tuple[str, ...]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise SectionPlanError("name must be a non-empty string.")
        _require_int("start_time", self.start_time, minimum=0)
        _require_int("end_time", self.end_time, minimum=1)
        if self.end_time <= self.start_time:
            raise SectionPlanError(f"Section '{self.name}' end_time must be > start_time.")
        _require_int("boundary_level", self.boundary_level, minimum=0)
        if self.boundary_level > 3:
            raise SectionPlanError(
                f"Section '{self.name}' boundary_level {self.boundary_level} exceeds vocabulary maximum 3."
            )
        arc = tuple(float(item) for item in self.target_tension_arc)
        if len(arc) < 2:
            raise SectionPlanError(f"Section '{self.name}' target_tension_arc must contain at least two values.")
        for idx, value in enumerate(arc):
            if value < 0.0 or value > 1.0:
                raise SectionPlanError(f"Section '{self.name}' target_tension_arc[{idx}] value {value} must be in [0, 1].")
        object.__setattr__(self, "target_tension_arc", arc)

        if self.allowed_meters is not None:
            meters = tuple(self.allowed_meters)
            if not meters or any(not isinstance(m, str) or not m.strip() for m in meters):
                raise SectionPlanError(f"Section '{self.name}' allowed_meters must be a tuple of non-empty strings.")
            object.__setattr__(self, "allowed_meters", meters)
        if self.groove_family is not None:
            if not isinstance(self.groove_family, str) or not self.groove_family.strip():
                raise SectionPlanError(f"Section '{self.name}' groove_family must be a non-empty string.")

        if self.key_region is not None:
            region = tuple(self.key_region)
            if not region:
                raise SectionPlanError(f"Section '{self.name}' key_region must not be empty if specified.")
            for r in region:
                _require_int(f"Section '{self.name}' key_region entry", r, minimum=0)
            object.__setattr__(self, "key_region", region)
            if self.target_key_id is None:
                object.__setattr__(self, "target_key_id", region[0])
            else:
                _require_int("target_key_id", self.target_key_id, minimum=0)
        elif self.target_key_id is not None:
            _require_int("target_key_id", self.target_key_id, minimum=0)
            object.__setattr__(self, "key_region", (self.target_key_id,))

        if self.target_density is not None:
            _require_real("target_density", self.target_density, minimum=0.0)
            if self.target_density > 1.0:
                raise SectionPlanError(f"Section '{self.name}' target_density {self.target_density} must be in [0, 1].")
        if self.preferred_roles is not None:
            roles = tuple(self.preferred_roles)
            if not roles or any(not isinstance(r, str) or not r.strip() for r in roles):
                raise SectionPlanError(f"Section '{self.name}' preferred_roles must be a tuple of non-empty strings.")
            object.__setattr__(self, "preferred_roles", roles)


@dataclass(frozen=True)
class EdgeSectionView:
    """View of section context and target tension from an edge perspective (arrival state k = t+1)."""

    section: PlanningSection
    section_index: int
    offset_in_section: int
    is_section_start: bool
    target_tension: float
    arrival_time: int
    multi_meter: bool = False


def interpolate_arc(arc: Sequence[float], offset: int | float, length: int) -> float:
    """Interpolate target tension arc across a section of given length.

    First beat (offset=0) equals arc[0]; last beat (offset=length-1) equals arc[-1].
    For length <= 1, returns arc[0] without division by zero.
    Monotone arcs remain monotone.
    """
    if not arc:
        return 0.5
    if len(arc) == 1 or length <= 1:
        return float(arc[0])

    n_segments = len(arc) - 1
    clamped_offset = max(0.0, min(float(length - 1), float(offset)))
    frac = clamped_offset / float(length - 1)
    seg_pos = frac * n_segments
    seg_index = min(int(seg_pos), n_segments - 1)
    seg_frac = seg_pos - seg_index
    start_val = arc[seg_index]
    end_val = arc[seg_index + 1]
    return float(start_val + (end_val - start_val) * seg_frac)


@dataclass(frozen=True)
class SectionTimeline:
    """Validated, contiguous timeline of planning sections covering beats [0, total_beats]."""

    sections: Tuple[PlanningSection, ...]
    total_beats: int
    max_boundary_level: int = 3
    reachable_range: Optional[Tuple[float, float]] = None
    edo: int = 12
    multi_meter: bool = False

    def __post_init__(self) -> None:
        if self.sections is None or len(self.sections) == 0:
            raise SectionPlanError("Section plan is empty; at least one section is required.")

        secs = tuple(self.sections)
        object.__setattr__(self, "sections", secs)

        if not isinstance(self.total_beats, int) or isinstance(self.total_beats, bool) or self.total_beats < 1:
            raise SectionPlanError(f"total_beats must be an int >= 1, got {self.total_beats}.")

        _require_int("edo", self.edo, minimum=1)
        if not isinstance(self.multi_meter, bool):
            raise TypeError("multi_meter must be a bool.")
        if not self.multi_meter:
            if any(sec.allowed_meters and len(sec.allowed_meters) > 1 for sec in secs):
                object.__setattr__(self, "multi_meter", True)

        # 1. First section starts at 0
        if secs[0].start_time != 0:
            raise SectionPlanError(
                f"First section '{secs[0].name}' start_time {secs[0].start_time} must be 0."
            )

        # 2. Contiguity and sorting
        for i in range(1, len(secs)):
            curr = secs[i]
            prev = secs[i - 1]
            if curr.start_time < prev.end_time:
                raise SectionPlanError(
                    f"Section '{curr.name}' overlaps with section '{prev.name}': "
                    f"start {curr.start_time} < previous end {prev.end_time}."
                )
            if curr.start_time > prev.end_time:
                raise SectionPlanError(
                    f"Gap detected before section '{curr.name}': "
                    f"start {curr.start_time} > previous end {prev.end_time}."
                )

        # 3. Last section ends at total_beats
        if secs[-1].end_time != self.total_beats:
            raise SectionPlanError(
                f"Last section '{secs[-1].name}' end_time {secs[-1].end_time} "
                f"must equal total_beats {self.total_beats}."
            )

        # 4. Unique, non-empty names
        seen_names: set[str] = set()
        for sec in secs:
            if not sec.name or not sec.name.strip():
                raise SectionPlanError("Section name must be a non-empty string.")
            if sec.name in seen_names:
                raise SectionPlanError(
                    f"Duplicate section name '{sec.name}': section names must be unique."
                )
            seen_names.add(sec.name)

        # 5. Arc, density, boundary level, and key_region validations per section
        for sec in secs:
            if sec.key_region is not None:
                for r in sec.key_region:
                    if r < 0 or r >= self.edo:
                        raise SectionPlanError(
                            f"Section '{sec.name}' key_region pitch class {r} is outside [0, {self.edo})."
                        )
            if len(sec.target_tension_arc) < 2:
                raise SectionPlanError(
                    f"Section '{sec.name}' target_tension_arc must contain at least 2 points."
                )
            for idx, val in enumerate(sec.target_tension_arc):
                if val < 0.0 or val > 1.0:
                    raise SectionPlanError(
                        f"Section '{sec.name}' target_tension_arc[{idx}] value {val} must be in [0, 1]."
                    )
            if sec.target_density is not None and not (0.0 <= sec.target_density <= 1.0):
                raise SectionPlanError(
                    f"Section '{sec.name}' target_density {sec.target_density} must be in [0, 1]."
                )
            if sec.boundary_level < 0 or sec.boundary_level > self.max_boundary_level:
                raise SectionPlanError(
                    f"Section '{sec.name}' boundary_level {sec.boundary_level} "
                    f"exceeds vocabulary maximum {self.max_boundary_level}."
                )

        # 6. Reachable range warning (soft diagnostic check, not an error)
        if self.reachable_range is not None:
            r_min, r_max = self.reachable_range
            for sec in secs:
                for idx, val in enumerate(sec.target_tension_arc):
                    if val < r_min or val > r_max:
                        _logger.warning(
                            f"Section '{sec.name}' target_tension_arc[{idx}]={val:.4f} "
                            f"is outside reachable range [{r_min:.4f}, {r_max:.4f}]."
                        )

    def at_state(self, k: int) -> PlanningSection:
        """Return the PlanningSection governing beat state index k."""
        if not isinstance(k, int) or isinstance(k, bool):
            raise TypeError("State index k must be an int.")
        if k < 0 or k > self.total_beats:
            raise IndexError(f"State index {k} is outside [0, {self.total_beats}].")
        for sec in self.sections:
            if sec.start_time <= k < sec.end_time:
                return sec
        if k == self.total_beats:
            return self.sections[-1]
        raise IndexError(f"State index {k} could not be resolved.")

    def section_index(self, k: int) -> int:
        """Return the 0-based index of the section governing beat state index k."""
        if not isinstance(k, int) or isinstance(k, bool):
            raise TypeError("State index k must be an int.")
        if k < 0 or k > self.total_beats:
            raise IndexError(f"State index {k} is outside [0, {self.total_beats}].")
        for idx, sec in enumerate(self.sections):
            if sec.start_time <= k < sec.end_time:
                return idx
        if k == self.total_beats:
            return len(self.sections) - 1
        raise IndexError(f"State index {k} could not be resolved.")

    def offset_in_section(self, k: int) -> int:
        """Return the beat offset of state k relative to its section start."""
        sec = self.at_state(k)
        return k - sec.start_time

    def is_section_start(self, k: int) -> bool:
        """Check whether state index k is the exact start of a section."""
        sec = self.at_state(k)
        return k == sec.start_time

    def target_tension(self, k: int) -> float:
        """Return interpolated target tension value for beat state index k."""
        sec = self.at_state(k)
        offset = self.offset_in_section(k)
        length = sec.end_time - sec.start_time
        return interpolate_arc(sec.target_tension_arc, offset, length)

    def state_view(self, k: int) -> EdgeSectionView:
        """Return section view for beat state index k."""
        if not isinstance(k, int) or isinstance(k, bool):
            raise TypeError("State index k must be an int.")
        sec = self.at_state(k)
        idx = self.section_index(k)
        offset = self.offset_in_section(k)
        is_start = self.is_section_start(k)
        tension = self.target_tension(k)
        return EdgeSectionView(
            section=sec,
            section_index=idx,
            offset_in_section=offset,
            is_section_start=is_start,
            target_tension=tension,
            arrival_time=k,
            multi_meter=self.multi_meter,
        )

    def edge_view(self, t: int) -> EdgeSectionView:
        """Return section view for edge t, scoring arrival state k = t + 1."""
        if not isinstance(t, int) or isinstance(t, bool):
            raise TypeError("Edge time index t must be an int.")
        return self.state_view(t + 1)

    def as_curve(self) -> List[Tuple[int, float]]:
        """Return full (k, target_tension) curve for k in 0..total_beats."""
        return [(k, self.target_tension(k)) for k in range(self.total_beats + 1)]

    def validate_executable(
        self,
        style_config: StyleConfig,
        vocabularies: Vocabularies,
        *,
        snap_to_bars: bool = False,
    ) -> Any:
        """Validate alignment of section starts against meter downbeats.

        For a style with a single allowed meter, every section start must fall on a downbeat.
        If snap_to_bars=False, non-aligned starts raise SectionPlanError.
        If snap_to_bars=True, returns (new_timeline, list_of_changes).
        For multi-meter styles, bar positions are path dependent: returns warnings only.
        """
        allowed = style_config.allowed_meters
        if not allowed:
            allowed = ("4/4",)

        if len(allowed) > 1:
            warnings: list[str] = []
            for sec in self.sections[1:]:
                warnings.append(
                    f"Section '{sec.name}' start_time {sec.start_time} cannot be verified as a downbeat statically "
                    f"under multi-meter style {allowed}; bar positions are path-dependent."
                )
            if snap_to_bars:
                return self, warnings
            return warnings

        meter_label = allowed[0]
        if meter_label not in vocabularies.meters.label_map:
            raise SectionPlanError(f"Allowed meter '{meter_label}' not found in vocabulary.")
        meter_token = vocabularies.meters.token_for_label(meter_label)
        beats_per_bar = meter_token.beats_per_bar
        if beats_per_bar <= 0:
            raise SectionPlanError(f"Meter '{meter_label}' has invalid beats_per_bar={beats_per_bar}.")

        changes: list[str] = []
        new_sections: list[PlanningSection] = list(self.sections)
        needs_snap = False

        for idx, sec in enumerate(self.sections):
            if sec.start_time % beats_per_bar != 0:
                if not snap_to_bars:
                    raise SectionPlanError(
                        f"Section '{sec.name}' start_time {sec.start_time} does not fall on a downbeat "
                        f"for meter '{meter_label}' (bar length {beats_per_bar})."
                    )
                needs_snap = True
                snapped_start = sec.start_time - (sec.start_time % beats_per_bar)
                if idx > 0 and snapped_start <= new_sections[idx - 1].start_time:
                    raise SectionPlanError(
                        f"Cannot snap section '{sec.name}' start_time {sec.start_time} to {snapped_start}: "
                        f"would collide with or invert previous section '{new_sections[idx - 1].name}' start {new_sections[idx - 1].start_time}."
                    )
                changes.append(
                    f"Section '{sec.name}' start snapped from {sec.start_time} to {snapped_start} "
                    f"(previous section '{new_sections[idx - 1].name}' end adjusted from {new_sections[idx - 1].end_time} to {snapped_start})."
                )
                if idx > 0:
                    prev = new_sections[idx - 1]
                    new_sections[idx - 1] = PlanningSection(
                        name=prev.name,
                        start_time=prev.start_time,
                        end_time=snapped_start,
                        boundary_level=prev.boundary_level,
                        target_tension_arc=prev.target_tension_arc,
                        allowed_meters=prev.allowed_meters,
                        groove_family=prev.groove_family,
                        target_key_id=prev.target_key_id,
                        key_region=prev.key_region,
                        target_density=prev.target_density,
                        preferred_roles=prev.preferred_roles,
                    )
                new_sections[idx] = PlanningSection(
                    name=sec.name,
                    start_time=snapped_start,
                    end_time=sec.end_time,
                    boundary_level=sec.boundary_level,
                    target_tension_arc=sec.target_tension_arc,
                    allowed_meters=sec.allowed_meters,
                    groove_family=sec.groove_family,
                    target_key_id=sec.target_key_id,
                    key_region=sec.key_region,
                    target_density=sec.target_density,
                    preferred_roles=sec.preferred_roles,
                )

        if snap_to_bars:
            if needs_snap:
                new_timeline = SectionTimeline(
                    sections=tuple(new_sections),
                    total_beats=self.total_beats,
                    max_boundary_level=self.max_boundary_level,
                    reachable_range=self.reachable_range,
                    edo=self.edo,
                    multi_meter=self.multi_meter,
                )
                return new_timeline, changes
            return self, changes

        return []
