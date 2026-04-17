from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Callable, Dict, List, Mapping, Optional

from config import PriorWeights
from core_types import BeatState
from tonal import (
    basic_space_distance,
    is_dominant,
    is_tonic_family,
    nearest_roots,
    tonal_distance,
)
from vocab import (
    DEFAULT_VOCABULARIES,
    ChordToken,
    GrooveToken,
    HeadToken,
    KeyToken,
    MeterToken,
    RoleToken,
    Vocabularies,
)


# ==========================================
# 1. Legacy structural prototype
# Kept for the existing reduction-structure tests.
# ==========================================
@dataclass(frozen=True)
class MetricalGrid:
    beat_index: int
    level: int


@dataclass(frozen=True)
class MusicalEvent:
    root_pc: int
    quality: str
    bass_pc: int
    duration: float
    meter: MetricalGrid


@dataclass
class Group:
    level_name: str
    events: List[MusicalEvent] = field(default_factory=list)
    sub_groups: List["Group"] = field(default_factory=list)

    def get_all_events(self) -> List[MusicalEvent]:
        if not self.sub_groups:
            return self.events
        all_events: List[MusicalEvent] = []
        for sub_group in self.sub_groups:
            all_events.extend(sub_group.get_all_events())
        return all_events


@dataclass
class TimeSpanNode:
    head: Optional[MusicalEvent] = None
    children: List["TimeSpanNode"] = field(default_factory=list)


def reduce_time_span(group: Group) -> TimeSpanNode:
    """Convert a Group into a simplified time-span reduction tree."""
    node = TimeSpanNode()
    if not group.sub_groups:
        node.head = group.events[0]
        return node

    for sub_group in group.sub_groups:
        node.children.append(reduce_time_span(sub_group))

    node.head = node.children[0].head
    return node


class BranchType(Enum):
    RIGHT_TENSING = "Right-Branching (Tensing/Departure)"
    LEFT_RELAXING = "Left-Branching (Relaxing/Arrival)"
    STRONG_PROLONGATION = "Strong Prolongation (Static)"


@dataclass
class ProlongationalNode:
    event: MusicalEvent
    branch_type: Optional[BranchType] = None
    children: List["ProlongationalNode"] = field(default_factory=list)


def assign_prolongational_branching(
    tsr_node: TimeSpanNode, target_event: MusicalEvent
) -> ProlongationalNode:
    """Assign simplified prolongational branch types from a TSR tree."""
    p_node = ProlongationalNode(event=target_event)

    for child in tsr_node.children:
        if child.head is None:
            continue

        child_p_node = ProlongationalNode(event=child.head)
        if target_event.root_pc == 0 and child.head.root_pc == 7:
            child_p_node.branch_type = BranchType.RIGHT_TENSING
        elif target_event.root_pc == 7 and child.head.root_pc == 0:
            child_p_node.branch_type = BranchType.LEFT_RELAXING
        else:
            child_p_node.branch_type = BranchType.STRONG_PROLONGATION
        p_node.children.append(child_p_node)

    return p_node


# ==========================================
# 2. BeatState-centric GTTM feature library
# Primary EPIC 2 implementation.
# ==========================================
W_METER_STABILITY = 1.0
W_GROOVE_CONTINUITY = 0.6
W_BOUNDARY_ON_STRONG = 0.9
W_BOUNDARY_ON_WEAK_PENALTY = 0.7
W_STRONG_BEAT_BIAS = 0.15
W_ILLEGAL_BEAT_PENALTY = 2.0

MeterSpec = MeterToken
DEFAULT_METERS: Mapping[int, MeterSpec] = DEFAULT_VOCABULARIES.meters.id_map

ANCHOR_HEADS = frozenset({"root", "third", "fifth", "seventh"})
APPROACH_HEADS = frozenset({"upper_approach", "lower_approach"})
ROLE_PROGRESSIONS = {
    ("hold", "hold"): 0.4,
    ("hold", "prep"): 0.25,
    ("prep", "change"): 0.3,
    ("prep", "cad"): 1.1,
    ("change", "hold"): 0.25,
    ("change", "cad"): 0.55,
    ("cad", "hold"): 0.6,
}
SEVENTH_COMPATIBLE_QUALITIES = frozenset({"7", "maj7", "m7", "dim7", "m7b5"})


@dataclass(frozen=True)
class TransitionWindow:
    """Optional local context around a BeatState transition."""

    left_state: Optional[BeatState] = None
    right_state: Optional[BeatState] = None


@dataclass(frozen=True)
class GTTMFeatureSpec:
    """Metadata for one named BeatState feature."""

    name: str
    family: str
    func: Callable[..., float]
    base_weight: float = 1.0


def beats_per_bar(
    meter_id: int,
    meters: Optional[Mapping[int, MeterSpec]] = None,
) -> int:
    meter_map = DEFAULT_METERS if meters is None else meters
    spec = meter_map.get(meter_id)
    if spec is None:
        return 4
    return max(1, int(spec.beats_per_bar))


def is_strong_beat(beat_in_bar: int, beats_per_bar_value: int) -> bool:
    if beat_in_bar == 0:
        return True
    if beats_per_bar_value % 2 == 0 and beat_in_bar == (beats_per_bar_value // 2):
        return True
    return False


def beat_position_penalty(
    state: BeatState,
    meters: Optional[Mapping[int, MeterSpec]] = None,
) -> float:
    beats = beats_per_bar(state.meter_id, meters)
    if state.beat_in_bar < 0 or state.beat_in_bar >= beats:
        return -W_ILLEGAL_BEAT_PENALTY
    return 0.0


def strong_beat_alignment(
    state: BeatState,
    meters: Optional[Mapping[int, MeterSpec]] = None,
) -> float:
    beats = beats_per_bar(state.meter_id, meters)
    return W_STRONG_BEAT_BIAS if is_strong_beat(state.beat_in_bar, beats) else 0.0


def boundary_alignment_score(
    state: BeatState,
    meters: Optional[Mapping[int, MeterSpec]] = None,
) -> float:
    if state.boundary_lvl <= 0:
        return 0.0

    beats = beats_per_bar(state.meter_id, meters)
    if is_strong_beat(state.beat_in_bar, beats):
        return W_BOUNDARY_ON_STRONG * float(state.boundary_lvl)
    return -W_BOUNDARY_ON_WEAK_PENALTY * float(state.boundary_lvl)


def grouping_window_score(
    prev_state: Optional[BeatState],
    curr_state: BeatState,
    next_state: Optional[BeatState],
    meters: Optional[Mapping[int, MeterSpec]] = None,
) -> float:
    if prev_state is None or next_state is None or curr_state.boundary_lvl <= 0:
        return 0.0

    onset_bonus = 0.25 if prev_state.boundary_lvl <= 0 else 0.0
    next_beats = beats_per_bar(next_state.meter_id, meters)
    next_strong = 0.35 if is_strong_beat(next_state.beat_in_bar, next_beats) else 0.0
    return (onset_bonus + next_strong) * float(curr_state.boundary_lvl)


def _resolve_vocabularies(vocabularies: Optional[Vocabularies]) -> Vocabularies:
    return DEFAULT_VOCABULARIES if vocabularies is None else vocabularies


def _resolve_meters(meters: Optional[Mapping[int, MeterSpec]]) -> Mapping[int, MeterSpec]:
    return DEFAULT_METERS if meters is None else meters


def _resolve_edo(vocabularies: Vocabularies, edo: Optional[int]) -> int:
    return len(vocabularies.keys) if edo is None else edo


def _role_token(state: BeatState, vocabularies: Vocabularies) -> Optional[RoleToken]:
    if vocabularies.roles.has_id(state.role_id):
        return vocabularies.roles.token_for_id(state.role_id)
    return None


def _head_token(state: BeatState, vocabularies: Vocabularies) -> Optional[HeadToken]:
    if vocabularies.heads.has_id(state.head_id):
        return vocabularies.heads.token_for_id(state.head_id)
    return None


def _groove_token(state: BeatState, vocabularies: Vocabularies) -> Optional[GrooveToken]:
    if vocabularies.grooves.has_id(state.groove_id):
        return vocabularies.grooves.token_for_id(state.groove_id)
    return None


def _key_token(state: BeatState, vocabularies: Vocabularies) -> Optional[KeyToken]:
    if vocabularies.keys.has_id(state.key_id):
        return vocabularies.keys.token_for_id(state.key_id)
    return None


def _chord_token(state: BeatState, vocabularies: Vocabularies) -> Optional[ChordToken]:
    if vocabularies.chords.has_id(state.chord_id):
        return vocabularies.chords.token_for_id(state.chord_id)
    return None


@lru_cache(maxsize=4096)
def _cached_basic_space_distance(
    a_root: int,
    a_quality: str,
    b_root: int,
    b_quality: str,
    edo: int,
) -> float:
    return basic_space_distance(a_root, a_quality, b_root, b_quality, edo)


def harmonic_distance_cache_info():
    """Expose tonal-distance cache info for diagnostics/tests."""
    return _cached_basic_space_distance.cache_info()


def tonal_neighbor_cache_info():
    """Expose tonal-neighbor cache info for diagnostics/tests."""
    return nearest_roots.cache_info()


def meter_stability_feature(
    prev_state: BeatState,
    next_state: BeatState,
    t: int,
    window: Optional[TransitionWindow] = None,
    meters: Optional[Mapping[int, MeterSpec]] = None,
    vocabularies: Optional[Vocabularies] = None,
    edo: Optional[int] = None,
) -> float:
    del t, window, meters, vocabularies, edo
    return 1.0 if prev_state.meter_id == next_state.meter_id else -1.0


def beat_position_validity_feature(
    prev_state: BeatState,
    next_state: BeatState,
    t: int,
    window: Optional[TransitionWindow] = None,
    meters: Optional[Mapping[int, MeterSpec]] = None,
    vocabularies: Optional[Vocabularies] = None,
    edo: Optional[int] = None,
) -> float:
    del prev_state, t, window, vocabularies, edo
    return beat_position_penalty(next_state, meters)


def boundary_placement_feature(
    prev_state: BeatState,
    next_state: BeatState,
    t: int,
    window: Optional[TransitionWindow] = None,
    meters: Optional[Mapping[int, MeterSpec]] = None,
    vocabularies: Optional[Vocabularies] = None,
    edo: Optional[int] = None,
) -> float:
    del prev_state, t, window, vocabularies, edo
    return boundary_alignment_score(next_state, meters)


def strong_beat_bias_feature(
    prev_state: BeatState,
    next_state: BeatState,
    t: int,
    window: Optional[TransitionWindow] = None,
    meters: Optional[Mapping[int, MeterSpec]] = None,
    vocabularies: Optional[Vocabularies] = None,
    edo: Optional[int] = None,
) -> float:
    del prev_state, t, window, vocabularies, edo
    return strong_beat_alignment(next_state, meters)


def grouping_onset_feature(
    prev_state: BeatState,
    next_state: BeatState,
    t: int,
    window: Optional[TransitionWindow] = None,
    meters: Optional[Mapping[int, MeterSpec]] = None,
    vocabularies: Optional[Vocabularies] = None,
    edo: Optional[int] = None,
) -> float:
    del t, window, meters, vocabularies, edo
    if next_state.boundary_lvl <= 0:
        return 0.0
    if prev_state.boundary_lvl <= 0:
        return 0.45 * float(next_state.boundary_lvl)
    return -0.2 * float(next_state.boundary_lvl)


def grouping_boundary_resolution_feature(
    prev_state: BeatState,
    next_state: BeatState,
    t: int,
    window: Optional[TransitionWindow] = None,
    meters: Optional[Mapping[int, MeterSpec]] = None,
    vocabularies: Optional[Vocabularies] = None,
    edo: Optional[int] = None,
) -> float:
    del t, vocabularies, edo
    if window is None:
        return 0.0
    return grouping_window_score(prev_state, next_state, window.right_state, meters)


def local_grouping_feature(
    prev_state: BeatState,
    next_state: BeatState,
    t: int,
    window: Optional[TransitionWindow] = None,
    meters: Optional[Mapping[int, MeterSpec]] = None,
    vocabularies: Optional[Vocabularies] = None,
    edo: Optional[int] = None,
) -> float:
    """Backward-compatible alias for the windowed grouping-resolution feature."""
    return grouping_boundary_resolution_feature(
        prev_state,
        next_state,
        t,
        window=window,
        meters=meters,
        vocabularies=vocabularies,
        edo=edo,
    )


def grouping_downbeat_alignment_feature(
    prev_state: BeatState,
    next_state: BeatState,
    t: int,
    window: Optional[TransitionWindow] = None,
    meters: Optional[Mapping[int, MeterSpec]] = None,
    vocabularies: Optional[Vocabularies] = None,
    edo: Optional[int] = None,
) -> float:
    del prev_state, t, window, vocabularies, edo
    if next_state.boundary_lvl < 2:
        return 0.0
    beats = beats_per_bar(next_state.meter_id, meters)
    if next_state.beat_in_bar == 0:
        return 0.55 * float(next_state.boundary_lvl)
    if is_strong_beat(next_state.beat_in_bar, beats):
        return 0.2 * float(next_state.boundary_lvl)
    return -0.4 * float(next_state.boundary_lvl)


def harmonic_key_proximity_feature(
    prev_state: BeatState,
    next_state: BeatState,
    t: int,
    window: Optional[TransitionWindow] = None,
    meters: Optional[Mapping[int, MeterSpec]] = None,
    vocabularies: Optional[Vocabularies] = None,
    edo: Optional[int] = None,
) -> float:
    del t, window, meters
    resolved_vocabs = _resolve_vocabularies(vocabularies)
    key_a = _key_token(prev_state, resolved_vocabs)
    key_b = _key_token(next_state, resolved_vocabs)
    if key_a is None or key_b is None:
        return -1.0
    resolved_edo = _resolve_edo(resolved_vocabs, edo)
    distance = tonal_distance(key_a.root_pc, key_b.root_pc, resolved_edo)
    return 1.0 / (1.0 + distance)


def harmonic_key_neighbor_feature(
    prev_state: BeatState,
    next_state: BeatState,
    t: int,
    window: Optional[TransitionWindow] = None,
    meters: Optional[Mapping[int, MeterSpec]] = None,
    vocabularies: Optional[Vocabularies] = None,
    edo: Optional[int] = None,
) -> float:
    del t, window, meters
    resolved_vocabs = _resolve_vocabularies(vocabularies)
    key_a = _key_token(prev_state, resolved_vocabs)
    key_b = _key_token(next_state, resolved_vocabs)
    if key_a is None or key_b is None:
        return -0.5
    resolved_edo = _resolve_edo(resolved_vocabs, edo)
    if key_a.root_pc == key_b.root_pc:
        return 0.6
    neighbors = nearest_roots(key_a.root_pc, resolved_edo, limit=3)
    return 0.4 if key_b.root_pc in neighbors else -0.35


def harmonic_chord_proximity_feature(
    prev_state: BeatState,
    next_state: BeatState,
    t: int,
    window: Optional[TransitionWindow] = None,
    meters: Optional[Mapping[int, MeterSpec]] = None,
    vocabularies: Optional[Vocabularies] = None,
    edo: Optional[int] = None,
) -> float:
    del t, window, meters
    resolved_vocabs = _resolve_vocabularies(vocabularies)
    chord_a = _chord_token(prev_state, resolved_vocabs)
    chord_b = _chord_token(next_state, resolved_vocabs)
    if chord_a is None or chord_b is None:
        return -1.0
    resolved_edo = _resolve_edo(resolved_vocabs, edo)
    distance = _cached_basic_space_distance(
        chord_a.root_pc,
        chord_a.quality,
        chord_b.root_pc,
        chord_b.quality,
        resolved_edo,
    )
    return 1.0 / (1.0 + distance)


def cadential_harmonic_motion_feature(
    prev_state: BeatState,
    next_state: BeatState,
    t: int,
    window: Optional[TransitionWindow] = None,
    meters: Optional[Mapping[int, MeterSpec]] = None,
    vocabularies: Optional[Vocabularies] = None,
    edo: Optional[int] = None,
) -> float:
    del t, window, meters
    resolved_vocabs = _resolve_vocabularies(vocabularies)
    role = _role_token(next_state, resolved_vocabs)
    chord_a = _chord_token(prev_state, resolved_vocabs)
    chord_b = _chord_token(next_state, resolved_vocabs)
    if role is None or chord_a is None or chord_b is None:
        return 0.0

    resolved_edo = _resolve_edo(resolved_vocabs, edo)
    is_cadential_motion = (
        tonal_distance(chord_a.root_pc, chord_b.root_pc, resolved_edo) == 1
        and is_dominant(chord_a.quality)
        and is_tonic_family(chord_b.quality)
    )

    if role.label == "cad":
        return 1.6 if is_cadential_motion else -0.8
    if is_cadential_motion:
        return 0.35
    return 0.0


def role_meter_alignment_feature(
    prev_state: BeatState,
    next_state: BeatState,
    t: int,
    window: Optional[TransitionWindow] = None,
    meters: Optional[Mapping[int, MeterSpec]] = None,
    vocabularies: Optional[Vocabularies] = None,
    edo: Optional[int] = None,
) -> float:
    del prev_state, t, window, edo
    resolved_vocabs = _resolve_vocabularies(vocabularies)
    role = _role_token(next_state, resolved_vocabs)
    if role is None:
        return -0.3

    beats = beats_per_bar(next_state.meter_id, meters)
    strong = is_strong_beat(next_state.beat_in_bar, beats)
    if role.label == "cad":
        return 1.0 if strong else -1.0
    if role.label == "prep":
        return 0.45 if not strong else -0.2
    if role.label == "change":
        return 0.5 if next_state.boundary_lvl > 0 else -0.2
    return 0.35 if next_state.boundary_lvl == 0 else -0.25


def role_transition_feature(
    prev_state: BeatState,
    next_state: BeatState,
    t: int,
    window: Optional[TransitionWindow] = None,
    meters: Optional[Mapping[int, MeterSpec]] = None,
    vocabularies: Optional[Vocabularies] = None,
    edo: Optional[int] = None,
) -> float:
    del t, window, meters, edo
    resolved_vocabs = _resolve_vocabularies(vocabularies)
    prev_role = _role_token(prev_state, resolved_vocabs)
    next_role = _role_token(next_state, resolved_vocabs)
    if prev_role is None or next_role is None:
        return -0.2
    if prev_role.label == next_role.label:
        return 0.25
    return ROLE_PROGRESSIONS.get((prev_role.label, next_role.label), -0.3)


def head_anchor_feature(
    prev_state: BeatState,
    next_state: BeatState,
    t: int,
    window: Optional[TransitionWindow] = None,
    meters: Optional[Mapping[int, MeterSpec]] = None,
    vocabularies: Optional[Vocabularies] = None,
    edo: Optional[int] = None,
) -> float:
    del prev_state, t, window, edo
    resolved_vocabs = _resolve_vocabularies(vocabularies)
    head = _head_token(next_state, resolved_vocabs)
    chord = _chord_token(next_state, resolved_vocabs)
    if head is None or chord is None:
        return -0.3

    beats = beats_per_bar(next_state.meter_id, meters)
    strong = is_strong_beat(next_state.beat_in_bar, beats)
    label = head.label
    if label in {"root", "third", "fifth"}:
        return 0.9 if strong else 0.35
    if label == "seventh":
        if chord.quality in SEVENTH_COMPATIBLE_QUALITIES:
            return 0.75 if strong else 0.3
        return -0.2
    if label == "extension":
        return -0.15 if strong else 0.2
    if label in APPROACH_HEADS:
        return -0.35 if strong else 0.3
    return 0.1 if next_state.boundary_lvl > 0 else -0.05


def head_resolution_feature(
    prev_state: BeatState,
    next_state: BeatState,
    t: int,
    window: Optional[TransitionWindow] = None,
    meters: Optional[Mapping[int, MeterSpec]] = None,
    vocabularies: Optional[Vocabularies] = None,
    edo: Optional[int] = None,
) -> float:
    del t, window, meters, edo
    resolved_vocabs = _resolve_vocabularies(vocabularies)
    prev_head = _head_token(prev_state, resolved_vocabs)
    next_head = _head_token(next_state, resolved_vocabs)
    if prev_head is None or next_head is None:
        return -0.2

    if prev_head.label in APPROACH_HEADS and next_head.label in ANCHOR_HEADS:
        return 0.85
    if prev_head.label in APPROACH_HEADS and next_head.label in APPROACH_HEADS:
        return -0.55
    if next_head.label in APPROACH_HEADS and next_state.boundary_lvl > 0:
        return -0.25
    return 0.0


def groove_continuity_feature(
    prev_state: BeatState,
    next_state: BeatState,
    t: int,
    window: Optional[TransitionWindow] = None,
    meters: Optional[Mapping[int, MeterSpec]] = None,
    vocabularies: Optional[Vocabularies] = None,
    edo: Optional[int] = None,
) -> float:
    del t, window, meters, edo
    resolved_vocabs = _resolve_vocabularies(vocabularies)
    groove_a = _groove_token(prev_state, resolved_vocabs)
    groove_b = _groove_token(next_state, resolved_vocabs)
    if groove_a is None or groove_b is None:
        return -0.5
    if prev_state.groove_id == next_state.groove_id:
        return 1.0
    if groove_a.family == groove_b.family:
        return 0.4
    return -0.5


def groove_boundary_change_feature(
    prev_state: BeatState,
    next_state: BeatState,
    t: int,
    window: Optional[TransitionWindow] = None,
    meters: Optional[Mapping[int, MeterSpec]] = None,
    vocabularies: Optional[Vocabularies] = None,
    edo: Optional[int] = None,
) -> float:
    del t, window, meters, edo
    resolved_vocabs = _resolve_vocabularies(vocabularies)
    groove_a = _groove_token(prev_state, resolved_vocabs)
    groove_b = _groove_token(next_state, resolved_vocabs)
    if groove_a is None or groove_b is None:
        return -0.25
    if groove_a.family == groove_b.family:
        return 0.2 if next_state.boundary_lvl == 0 else 0.1
    if next_state.boundary_lvl > 0:
        return 0.55
    return -0.8


FEATURE_REGISTRY: Dict[str, GTTMFeatureSpec] = {
    "meter_stability": GTTMFeatureSpec("meter_stability", "meter", meter_stability_feature, 1.0),
    "beat_position_validity": GTTMFeatureSpec(
        "beat_position_validity", "meter", beat_position_validity_feature, 1.0
    ),
    "boundary_placement": GTTMFeatureSpec(
        "boundary_placement", "meter", boundary_placement_feature, 1.0
    ),
    "strong_beat_bias": GTTMFeatureSpec(
        "strong_beat_bias", "meter", strong_beat_bias_feature, 1.0
    ),
    "grouping_onset": GTTMFeatureSpec("grouping_onset", "grouping", grouping_onset_feature, 1.0),
    "grouping_boundary_resolution": GTTMFeatureSpec(
        "grouping_boundary_resolution", "grouping", grouping_boundary_resolution_feature, 1.0
    ),
    "grouping_downbeat_alignment": GTTMFeatureSpec(
        "grouping_downbeat_alignment", "grouping", grouping_downbeat_alignment_feature, 1.0
    ),
    "harmonic_key_proximity": GTTMFeatureSpec(
        "harmonic_key_proximity", "harmonic", harmonic_key_proximity_feature, 1.0
    ),
    "harmonic_key_neighbor": GTTMFeatureSpec(
        "harmonic_key_neighbor", "harmonic", harmonic_key_neighbor_feature, 0.8
    ),
    "harmonic_chord_proximity": GTTMFeatureSpec(
        "harmonic_chord_proximity", "harmonic", harmonic_chord_proximity_feature, 1.2
    ),
    "cadential_harmonic_motion": GTTMFeatureSpec(
        "cadential_harmonic_motion", "harmonic", cadential_harmonic_motion_feature, 1.0
    ),
    "role_meter_alignment": GTTMFeatureSpec(
        "role_meter_alignment", "prolongational_role", role_meter_alignment_feature, 1.0
    ),
    "role_transition": GTTMFeatureSpec(
        "role_transition", "prolongational_role", role_transition_feature, 0.9
    ),
    "head_anchor": GTTMFeatureSpec("head_anchor", "melodic_head", head_anchor_feature, 1.0),
    "head_resolution": GTTMFeatureSpec(
        "head_resolution", "melodic_head", head_resolution_feature, 1.0
    ),
    "groove_continuity": GTTMFeatureSpec(
        "groove_continuity", "groove", groove_continuity_feature, 1.0
    ),
    "groove_boundary_change": GTTMFeatureSpec(
        "groove_boundary_change", "groove", groove_boundary_change_feature, 1.0
    ),
}

BEATSTATE_FEATURES: Dict[str, Callable[..., float]] = {
    name: spec.func for name, spec in FEATURE_REGISTRY.items()
}


def _family_weight(weights: PriorWeights, family: str) -> float:
    return float(getattr(weights, family))


def transition_feature_vector(
    prev_state: BeatState,
    next_state: BeatState,
    t: int,
    window: Optional[TransitionWindow] = None,
    meters: Optional[Mapping[int, MeterSpec]] = None,
    vocabularies: Optional[Vocabularies] = None,
    edo: Optional[int] = None,
) -> Dict[str, float]:
    """Compute the raw BeatState feature vector for one transition."""
    meter_map = _resolve_meters(meters)
    resolved_vocabs = _resolve_vocabularies(vocabularies)
    resolved_edo = _resolve_edo(resolved_vocabs, edo)
    return {
        name: spec.func(
            prev_state,
            next_state,
            t,
            window,
            meter_map,
            resolved_vocabs,
            resolved_edo,
        )
        for name, spec in FEATURE_REGISTRY.items()
    }


def transition_family_scores(
    prev_state: BeatState,
    next_state: BeatState,
    t: int,
    window: Optional[TransitionWindow] = None,
    meters: Optional[Mapping[int, MeterSpec]] = None,
    vocabularies: Optional[Vocabularies] = None,
    edo: Optional[int] = None,
) -> Dict[str, float]:
    """Aggregate raw feature values by the six recommended feature families."""
    family_scores: Dict[str, float] = {
        "meter": 0.0,
        "grouping": 0.0,
        "harmonic": 0.0,
        "prolongational_role": 0.0,
        "melodic_head": 0.0,
        "groove": 0.0,
    }
    raw_features = transition_feature_vector(
        prev_state,
        next_state,
        t,
        window=window,
        meters=meters,
        vocabularies=vocabularies,
        edo=edo,
    )
    for name, value in raw_features.items():
        family_scores[FEATURE_REGISTRY[name].family] += value
    return family_scores


def weighted_feature_breakdown(
    prev_state: BeatState,
    next_state: BeatState,
    t: int,
    window: Optional[TransitionWindow] = None,
    meters: Optional[Mapping[int, MeterSpec]] = None,
    vocabularies: Optional[Vocabularies] = None,
    edo: Optional[int] = None,
    weights: Optional[PriorWeights] = None,
) -> Dict[str, float]:
    """Return the weighted contribution of every named feature."""
    resolved_weights = PriorWeights() if weights is None else weights
    raw_features = transition_feature_vector(
        prev_state,
        next_state,
        t,
        window=window,
        meters=meters,
        vocabularies=vocabularies,
        edo=edo,
    )
    breakdown: Dict[str, float] = {}
    for name, value in raw_features.items():
        spec = FEATURE_REGISTRY[name]
        breakdown[name] = spec.base_weight * _family_weight(resolved_weights, spec.family) * value
    return breakdown


def calculate_gttm_score(
    prev_state: BeatState,
    next_state: BeatState,
    t: int,
    window: Optional[TransitionWindow] = None,
    meters: Optional[Mapping[int, MeterSpec]] = None,
    vocabularies: Optional[Vocabularies] = None,
    edo: Optional[int] = None,
    weights: Optional[PriorWeights] = None,
) -> float:
    """Return the weighted structural preference score for a transition."""
    return float(
        sum(
            weighted_feature_breakdown(
                prev_state,
                next_state,
                t,
                window=window,
                meters=meters,
                vocabularies=vocabularies,
                edo=edo,
                weights=weights,
            ).values()
        )
    )


def calculate_gttm_energy(
    prev_state: BeatState,
    next_state: BeatState,
    t: int,
    window: Optional[TransitionWindow] = None,
    meters: Optional[Mapping[int, MeterSpec]] = None,
    vocabularies: Optional[Vocabularies] = None,
    edo: Optional[int] = None,
    weights: Optional[PriorWeights] = None,
) -> float:
    """Return a graph-friendly energy, where lower is structurally better."""
    return -calculate_gttm_score(
        prev_state,
        next_state,
        t,
        window=window,
        meters=meters,
        vocabularies=vocabularies,
        edo=edo,
        weights=weights,
    )


def transition_energy(
    prev_state: BeatState,
    next_state: BeatState,
    t: int,
    window: Optional[TransitionWindow] = None,
    meters: Optional[Mapping[int, MeterSpec]] = None,
    vocabularies: Optional[Vocabularies] = None,
    edo: Optional[int] = None,
    weights: Optional[PriorWeights] = None,
) -> float:
    """Backward-compatible alias for the weighted structural score."""
    return calculate_gttm_score(
        prev_state,
        next_state,
        t,
        window=window,
        meters=meters,
        vocabularies=vocabularies,
        edo=edo,
        weights=weights,
    )
