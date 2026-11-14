from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, Mapping, Optional, Sequence, Tuple
if TYPE_CHECKING:
    from aimusic.planning.plans import PlanningSection
    from aimusic.planning.sections import EdgeSectionView

from aimusic.core.config import StyleConfig
from aimusic.core.core_types import BeatState
from aimusic.core.rng import RNGKey, shuffle
from aimusic.planning.sections import key_ids_for_pcs
from aimusic.scoring.gttm_features import beats_per_bar
from aimusic.scoring.priors import NullPrior, Prior, PriorContext, PriorQuery, prior_logps
from aimusic.theory.tonal import get_fifth_steps, nearest_roots
from aimusic.core.vocab import (
    ChordToken,
    DEFAULT_VOCABULARIES,
    GrooveToken,
    Vocabularies,
    validate_vocabulary_compatibility,
)


LEGAL_ROLE_SUCCESSORS: Mapping[str, frozenset[str]] = {
    "hold": frozenset({"hold", "prep", "change"}),
    "prep": frozenset({"prep", "change", "cad"}),
    "change": frozenset({"hold", "change", "cad"}),
    "cad": frozenset({"hold", "prep"}),
}

ANCHOR_HEAD_LABELS = frozenset({"root", "third", "fifth", "seventh"})
APPROACH_HEAD_LABELS = frozenset({"upper_approach", "lower_approach"})


def _state_sort_key(state: BeatState) -> tuple[int, int, int, int, int, int, int, int]:
    return (
        state.meter_id,
        state.beat_in_bar,
        state.boundary_lvl,
        state.key_id,
        state.chord_id,
        state.role_id,
        state.head_id,
        state.groove_id,
    )


def _resolved_vocabs(vocabularies: Optional[Vocabularies]) -> Vocabularies:
    return DEFAULT_VOCABULARIES if vocabularies is None else vocabularies


def _resolved_style(style_config: Optional[StyleConfig]) -> StyleConfig:
    return StyleConfig() if style_config is None else style_config


def _resolved_prior(prior: Optional[Prior]) -> Prior:
    return NullPrior() if prior is None else prior


def _edo_size(vocabularies: Vocabularies) -> int:
    return len(vocabularies.keys)


def _meter_token(state: BeatState, vocabularies: Vocabularies):
    return vocabularies.meters.token_for_id(state.meter_id)


def _is_strong_beat(meter_id: int, beat_in_bar: int, vocabularies: Vocabularies) -> bool:
    return beat_in_bar in vocabularies.meters.token_for_id(meter_id).strong_beats


def _boundary_token(state: BeatState, vocabularies: Vocabularies):
    return vocabularies.boundaries.token_for_id(state.boundary_lvl)


def _role_label(state: BeatState, vocabularies: Vocabularies) -> str:
    return vocabularies.roles.token_for_id(state.role_id).label


def _head_label(state: BeatState, vocabularies: Vocabularies) -> str:
    return vocabularies.heads.token_for_id(state.head_id).label


def _key_root(state: BeatState, vocabularies: Vocabularies) -> int:
    return vocabularies.keys.token_for_id(state.key_id).root_pc


def _chord_token_by_id(chord_id: int, vocabularies: Vocabularies) -> ChordToken:
    return vocabularies.chords.token_for_id(chord_id)


def _groove_token_by_id(groove_id: int, vocabularies: Vocabularies) -> GrooveToken:
    return vocabularies.grooves.token_for_id(groove_id)


def _allowed_meter_ids(style_config: StyleConfig, vocabularies: Vocabularies) -> Tuple[int, ...]:
    allowed = []
    for signature in style_config.allowed_meters:
        if signature in vocabularies.meters.label_map:
            allowed.append(vocabularies.meters.token_for_label(signature).id)
    if not allowed:
        allowed.append(vocabularies.meters.token_for_id(0).id)
    return tuple(dict.fromkeys(allowed))


def _next_beat_index(
    prev_state: BeatState,
    next_meter_id: int,
    vocabularies: Vocabularies,
) -> int:
    if prev_state.meter_id == next_meter_id:
        beats = beats_per_bar(next_meter_id, vocabularies.meters.id_map)
        return (prev_state.beat_in_bar + 1) % beats
    return 0


def _head_id(label: str, vocabularies: Vocabularies) -> int:
    return vocabularies.heads.token_for_label(label).id


def _role_id(label: str, vocabularies: Vocabularies) -> int:
    return vocabularies.roles.token_for_label(label).id


def _qualities_for_role(role_label: str) -> Tuple[str, ...]:
    if role_label == "cad":
        return ("maj", "min")
    if role_label == "prep":
        return ("7", "min", "dim")
    if role_label == "change":
        return ("maj", "min", "7", "dim")
    return ("maj", "min", "7")


def _chord_ids_for_root(
    root_pc: int,
    qualities: Sequence[str],
    vocabularies: Vocabularies,
) -> Tuple[int, ...]:
    ids = []
    for chord in vocabularies.chords:
        if chord.root_pc == root_pc and chord.quality in qualities:
            ids.append(chord.id)
    return tuple(ids)


def _top_k_prior_chord_ids(
    prev_state: BeatState,
    next_meter_id: int,
    next_beat_in_bar: int,
    next_boundary_lvl: int,
    key_id: int,
    role_id: int,
    groove_id: int,
    *,
    prior: Prior,
    context: Optional[PriorContext],
    vocabularies: Vocabularies,
    top_k: int = 3,
) -> Tuple[int, ...]:
    if isinstance(prior, NullPrior):
        return ()

    anchor_head_id = _head_id("root", vocabularies)
    queries = tuple(
        PriorQuery(
            prev_state=prev_state,
            next_state=BeatState(
                meter_id=next_meter_id,
                beat_in_bar=next_beat_in_bar,
                boundary_lvl=next_boundary_lvl,
                key_id=key_id,
                chord_id=chord.id,
                role_id=role_id,
                head_id=anchor_head_id,
                groove_id=groove_id,
            ),
            time_index=0,
            context=context,
        )
        for chord in vocabularies.chords
    )
    scored = sorted(
        zip(prior_logps(prior, queries), queries),
        key=lambda item: (-item[0], item[1].next_state.chord_id),
    )
    return tuple(
        item[1].next_state.chord_id
        for item in scored[:top_k]
    )


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


def apply_meter_constraints(
    prev_state: BeatState,
    next_candidate: BeatState,
    style_config: StyleConfig,
    vocabularies: Vocabularies,
) -> tuple[bool, Optional[str]]:
    allowed_meter_ids = set(_allowed_meter_ids(style_config, vocabularies))

    if next_candidate.meter_id not in allowed_meter_ids:
        return False, "meter_not_allowed"

    if prev_state.meter_id != next_candidate.meter_id:
        if prev_state.boundary_lvl < 2:
            return False, "meter_change_requires_phrase_boundary"
        if prev_state.beat_in_bar != 0:
            return False, "meter_change_requires_downbeat_source"
        if next_candidate.beat_in_bar != 0:
            return False, "meter_change_requires_downbeat_target"

    return True, None


def apply_position_constraints(
    prev_state: BeatState,
    next_candidate: BeatState,
    style_config: Optional[StyleConfig] = None,
    vocabularies: Optional[Vocabularies] = None,
) -> tuple[bool, Optional[str]]:
    resolved_style = _resolved_style(style_config)
    resolved_vocabs = _resolved_vocabs(vocabularies)
    meter_ok, meter_reason = apply_meter_constraints(
        prev_state, next_candidate, resolved_style, resolved_vocabs
    )
    if not meter_ok:
        return False, meter_reason

    beats = beats_per_bar(next_candidate.meter_id, resolved_vocabs.meters.id_map)
    if next_candidate.beat_in_bar < 0 or next_candidate.beat_in_bar >= beats:
        return False, "invalid_beat_index"

    expected_beat = _next_beat_index(
        prev_state,
        next_candidate.meter_id,
        resolved_vocabs,
    )
    if next_candidate.beat_in_bar != expected_beat:
        return False, "non_contiguous_beat_progression"

    strong = _is_strong_beat(
        next_candidate.meter_id,
        next_candidate.beat_in_bar,
        resolved_vocabs,
    )
    if next_candidate.boundary_lvl > 0 and not strong:
        return False, "boundary_requires_strong_beat"
    if next_candidate.boundary_lvl >= 2 and next_candidate.beat_in_bar != 0:
        return False, "phrase_boundary_requires_downbeat"
    if next_candidate.boundary_lvl >= 3 and next_candidate.beat_in_bar != 0:
        return False, "section_boundary_requires_downbeat"

    return True, None


def apply_role_constraints(
    prev_state: BeatState,
    next_candidate: BeatState,
    vocabularies: Vocabularies,
) -> tuple[bool, Optional[str]]:
    prev_role = _role_label(prev_state, vocabularies)
    next_role = _role_label(next_candidate, vocabularies)

    if next_role not in LEGAL_ROLE_SUCCESSORS[prev_role]:
        return False, "illegal_role_progression"

    strong = _is_strong_beat(
        next_candidate.meter_id,
        next_candidate.beat_in_bar,
        vocabularies,
    )
    if next_role == "cad":
        if next_candidate.boundary_lvl <= 0:
            return False, "cadence_requires_boundary"
        if not strong:
            return False, "cadence_requires_strong_beat"

    if next_role == "hold" and next_candidate.boundary_lvl >= 2:
        return False, "hold_cannot_define_phrase_boundary"

    if next_role == "change":
        changed_harmony = (
            next_candidate.chord_id != prev_state.chord_id
            or next_candidate.key_id != prev_state.key_id
        )
        if next_candidate.boundary_lvl <= 0 and not changed_harmony:
            return False, "change_requires_boundary_or_harmonic_motion"

    return True, None


def apply_boundary_and_groove_constraints(
    prev_state: BeatState,
    next_candidate: BeatState,
    vocabularies: Vocabularies,
) -> tuple[bool, Optional[str]]:
    prev_groove = _groove_token_by_id(prev_state.groove_id, vocabularies)
    next_groove = _groove_token_by_id(next_candidate.groove_id, vocabularies)
    if prev_groove.family != next_groove.family and next_candidate.boundary_lvl <= 0:
        return False, "groove_family_change_requires_boundary"

    if next_candidate.key_id != prev_state.key_id:
        next_role = _role_label(next_candidate, vocabularies)
        if next_candidate.boundary_lvl < 2 and next_role not in {"change", "cad"}:
            return False, "key_change_requires_phrase_boundary_or_structural_role"

    head_label = _head_label(next_candidate, vocabularies)
    strong = _is_strong_beat(
        next_candidate.meter_id,
        next_candidate.beat_in_bar,
        vocabularies,
    )
    if head_label in APPROACH_HEAD_LABELS and (strong or next_candidate.boundary_lvl > 0):
        return False, "approach_head_requires_weak_non_boundary_position"

    chord = _chord_token_by_id(next_candidate.chord_id, vocabularies)
    if head_label == "seventh" and chord.quality != "7":
        return False, "seventh_head_requires_dominant_quality"

    return True, None


def is_legal_transition(
    prev_state: BeatState,
    next_candidate: BeatState,
    style_config: StyleConfig,
    vocabularies: Vocabularies,
) -> tuple[bool, Optional[str]]:
    checks = (
        apply_meter_constraints(prev_state, next_candidate, style_config, vocabularies),
        apply_position_constraints(prev_state, next_candidate, style_config, vocabularies),
        apply_role_constraints(prev_state, next_candidate, vocabularies),
        apply_boundary_and_groove_constraints(prev_state, next_candidate, vocabularies),
    )
    for ok, reason in checks:
        if not ok:
            return False, reason
    return True, None


def propose_meter_ids(
    prev_state: BeatState,
    style_config: StyleConfig,
    vocabularies: Vocabularies,
    section: Optional["PlanningSection"] = None,
) -> Tuple[int, ...]:
    proposals = [prev_state.meter_id]
    if prev_state.boundary_lvl >= 2 and prev_state.beat_in_bar == 0:
        proposals.extend(_allowed_meter_ids(style_config, vocabularies))
    if section is not None and section.allowed_meters:
        sec_meter_ids = [
            vocabularies.meters.token_for_label(m).id
            for m in section.allowed_meters
            if m in vocabularies.meters.label_map
        ]
        proposals = [m for m in sec_meter_ids if m in proposals] + [m for m in proposals if m not in sec_meter_ids]
    return tuple(dict.fromkeys(proposals))


def propose_boundary_levels(
    prev_state: BeatState,
    next_meter_id: int,
    next_beat_in_bar: int,
    vocabularies: Vocabularies,
    section_view: Optional[EdgeSectionView] = None,
    section: Optional[PlanningSection] = None,
    section_guided_proposals: bool = False,
    t: Optional[int] = None,
) -> Tuple[int, ...]:
    proposals = [vocabularies.boundaries.token_for_label("none").id]
    if _is_strong_beat(next_meter_id, next_beat_in_bar, vocabularies):
        proposals.append(vocabularies.boundaries.token_for_label("local").id)
    if next_beat_in_bar == 0:
        proposals.append(vocabularies.boundaries.token_for_label("phrase").id)
        proposals.append(vocabularies.boundaries.token_for_label("section").id)

    sec: Optional[PlanningSection] = None
    is_start = False
    if section_view is not None:
        sec = getattr(section_view, "section", None)
        is_start = getattr(section_view, "is_section_start", False)
    elif section is not None:
        sec = section
        is_start = bool(t is not None and getattr(sec, "start_time", None) == t + 1)

    if (
        section_guided_proposals
        and is_start
        and sec is not None
        and getattr(sec, "boundary_level", 0) > 0
    ):
        target_lvl = sec.boundary_level
        if target_lvl in proposals:
            proposals = [target_lvl] + [lvl for lvl in proposals if lvl != target_lvl]

    return tuple(dict.fromkeys(proposals))


def propose_role_ids(
    prev_state: BeatState,
    next_meter_id: int,
    next_beat_in_bar: int,
    next_boundary_lvl: int,
    vocabularies: Vocabularies,
    section: Optional["PlanningSection"] = None,
) -> Tuple[int, ...]:
    prev_role = _role_label(prev_state, vocabularies)
    allowed_labels = set(LEGAL_ROLE_SUCCESSORS[prev_role])
    strong = _is_strong_beat(next_meter_id, next_beat_in_bar, vocabularies)

    if not strong:
        allowed_labels.discard("cad")
    if next_boundary_lvl >= 2:
        allowed_labels.discard("hold")
    if next_boundary_lvl == 0:
        allowed_labels.discard("cad")

    role_ids = [
        vocabularies.roles.token_for_label(label).id
        for label in sorted(allowed_labels)
    ]
    if section is not None and section.preferred_roles:
        pref_ids = set(
            vocabularies.roles.token_for_label(r).id
            for r in section.preferred_roles
            if r in vocabularies.roles.label_map
        )
        role_ids.sort(key=lambda r_id: 0 if r_id in pref_ids else 1)
    return tuple(role_ids)


def propose_key_ids(
    prev_state: BeatState,
    next_boundary_lvl: int,
    next_role_id: int,
    vocabularies: Vocabularies,
    edo: Optional[int] = None,
    section: Optional["PlanningSection"] = None,
) -> Tuple[int, ...]:
    resolved_edo = _edo_size(vocabularies) if edo is None else edo
    validate_vocabulary_compatibility(vocabularies, resolved_edo)
    role_label = vocabularies.roles.token_for_id(next_role_id).label
    proposals = [prev_state.key_id]
    if section is not None and section.target_key_id is not None:
        target_ids = key_ids_for_pcs((section.target_key_id,), vocabularies)
        for tid in reversed(target_ids):
            proposals.insert(0, tid)
    if next_boundary_lvl >= 2 or role_label in {"change", "cad"}:
        for root_pc in nearest_roots(
            _key_root(prev_state, vocabularies), resolved_edo, limit=2
        ):
            if vocabularies.keys.has_id(root_pc):
                proposals.append(root_pc)
    return tuple(dict.fromkeys(proposals))


def propose_chord_ids(
    prev_state: BeatState,
    key_id: int,
    next_meter_id: int,
    next_beat_in_bar: int,
    next_boundary_lvl: int,
    next_role_id: int,
    groove_id: int,
    prior: Prior,
    context: Optional[PriorContext],
    vocabularies: Vocabularies,
    top_k_prior: int = 3,
    edo: Optional[int] = None,
) -> Tuple[int, ...]:
    resolved_edo = _edo_size(vocabularies) if edo is None else edo
    validate_vocabulary_compatibility(vocabularies, resolved_edo)
    role_label = vocabularies.roles.token_for_id(next_role_id).label
    prev_chord = _chord_token_by_id(prev_state.chord_id, vocabularies)
    key_root = vocabularies.keys.token_for_id(key_id).root_pc
    dominant_root = (key_root + get_fifth_steps(resolved_edo)) % resolved_edo

    proposals = [prev_state.chord_id]
    proposals.extend(_chord_ids_for_root(prev_chord.root_pc, _qualities_for_role(role_label), vocabularies))
    for root_pc in nearest_roots(prev_chord.root_pc, resolved_edo, limit=2):
        proposals.extend(_chord_ids_for_root(root_pc, _qualities_for_role(role_label), vocabularies))

    if role_label == "cad":
        proposals.extend(_chord_ids_for_root(key_root, ("maj", "min"), vocabularies))
    else:
        proposals.extend(_chord_ids_for_root(key_root, ("maj", "min"), vocabularies))
        proposals.extend(_chord_ids_for_root(dominant_root, ("7",), vocabularies))

    proposals.extend(
        _top_k_prior_chord_ids(
            prev_state, next_meter_id, next_beat_in_bar, next_boundary_lvl, key_id,
            next_role_id, groove_id, prior=prior, context=context, vocabularies=vocabularies, top_k=top_k_prior
        )
    )
    return tuple(dict.fromkeys(proposals))


def propose_head_ids(
    chord_id: int,
    next_meter_id: int,
    next_beat_in_bar: int,
    next_boundary_lvl: int,
    next_role_id: int,
    vocabularies: Vocabularies,
) -> Tuple[int, ...]:
    chord = _chord_token_by_id(chord_id, vocabularies)
    role_label = vocabularies.roles.token_for_id(next_role_id).label
    strong = _is_strong_beat(next_meter_id, next_beat_in_bar, vocabularies)

    anchor_labels = ["root", "third", "fifth"]
    if chord.quality == "7":
        anchor_labels.append("seventh")

    if role_label == "cad":
        labels = ["root", "third"]
    elif strong or next_boundary_lvl > 0:
        labels = anchor_labels
        if chord.quality == "7":
            labels.append("seventh")
    else:
        labels = ["root", "extension", "upper_approach", "lower_approach", "rest"]

    return tuple(vocabularies.heads.token_for_label(label).id for label in dict.fromkeys(labels))


def propose_groove_ids(
    prev_state: BeatState,
    next_boundary_lvl: int,
    next_role_id: int,
    vocabularies: Vocabularies,
    section: Optional["PlanningSection"] = None,
) -> Tuple[int, ...]:
    prev_groove = _groove_token_by_id(prev_state.groove_id, vocabularies)
    next_role = vocabularies.roles.token_for_id(next_role_id).label

    proposals = [prev_state.groove_id]
    for groove in vocabularies.grooves:
        if groove.family == prev_groove.family:
            proposals.append(groove.id)

    if next_boundary_lvl > 0 or next_role in {"change", "cad"}:
        seen_families = {prev_groove.family}
        for groove in vocabularies.grooves:
            if groove.family not in seen_families:
                proposals.append(groove.id)
                seen_families.add(groove.family)

    if section is not None and section.groove_family:
        sec_family = section.groove_family
        sec_grooves = [g.id for g in vocabularies.grooves if g.family == sec_family]
        if sec_grooves:
            proposals = sec_grooves + [g for g in proposals if g not in sec_grooves]

    return tuple(dict.fromkeys(proposals))


def _candidate_generator(
    prev_state: BeatState,
    style: StyleConfig,
    vocabs: Vocabularies,
    prior: Prior,
    context: Optional[PriorContext],
    key: RNGKey,
    edo: int,
    key_state: Optional[list[RNGKey]] = None,
    section_view: Optional[EdgeSectionView] = None,
    section: Optional[PlanningSection] = None,
    section_guided_proposals: bool = False,
    t: Optional[int] = None,
) -> Iterator[BeatState]:
    """Yields candidate states iteratively to avoid combinatorial memory explosions."""
    current_key = key

    sec: Optional[PlanningSection] = None
    is_start = False
    if section_view is not None:
        sec = getattr(section_view, "section", None)
        is_start = getattr(section_view, "is_section_start", False)
    elif section is not None:
        sec = section
        is_start = bool(t is not None and getattr(sec, "start_time", None) == t + 1)

    sec_meter_ids = (
        set(vocabs.meters.token_for_label(m).id for m in sec.allowed_meters if m in vocabs.meters.label_map)
        if section_guided_proposals and sec and sec.allowed_meters else None
    )
    sec_boundary_ids = (
        {sec.boundary_level}
        if section_guided_proposals and sec and is_start and sec.boundary_level > 0 and vocabs.boundaries.has_id(sec.boundary_level) else None
    )
    sec_role_ids = (
        set(vocabs.roles.token_for_label(r).id for r in sec.preferred_roles if r in vocabs.roles.label_map)
        if section_guided_proposals and sec and sec.preferred_roles else None
    )
    sec_groove_ids = (
        set(g.id for g in vocabs.grooves if g.family == sec.groove_family)
        if section_guided_proposals and sec and sec.groove_family else None
    )
    sec_key_ids = (
        set(key_ids_for_pcs(sec.key_region, vocabs))
        if section_guided_proposals and sec and sec.key_region is not None
        else (
            set(key_ids_for_pcs((sec.target_key_id,), vocabs))
            if section_guided_proposals and sec and sec.target_key_id is not None
            else None
        )
    )

    def _shuffled_prioritized(items: Sequence[int], priority_set: Optional[set[int]] = None) -> Tuple[int, ...]:
        nonlocal current_key
        if not priority_set or not items:
            shuffled, current_key = shuffle(current_key, items)
            if key_state is not None:
                key_state[0] = current_key
            return shuffled
        pref = [x for x in items if x in priority_set]
        other = [x for x in items if x not in priority_set]
        pref_shuffled, current_key = shuffle(current_key, pref)
        other_shuffled, current_key = shuffle(current_key, other)
        if key_state is not None:
            key_state[0] = current_key
        return pref_shuffled + other_shuffled

    sec_arg = sec if section_guided_proposals else None
    for meter_id in _shuffled_prioritized(propose_meter_ids(prev_state, style, vocabs, section=sec_arg), sec_meter_ids):
        beat_in_bar = _next_beat_index(prev_state, meter_id, vocabs)
        for bound_lvl in _shuffled_prioritized(
            propose_boundary_levels(
                prev_state,
                meter_id,
                beat_in_bar,
                vocabs,
                section_view=section_view,
                section=sec,
                section_guided_proposals=section_guided_proposals,
                t=t,
            ),
            sec_boundary_ids,
        ):
            for role_id in _shuffled_prioritized(propose_role_ids(prev_state, meter_id, beat_in_bar, bound_lvl, vocabs, section=sec_arg), sec_role_ids):
                for groove_id in _shuffled_prioritized(propose_groove_ids(prev_state, bound_lvl, role_id, vocabs, section=sec_arg), sec_groove_ids):
                    for key_id in _shuffled_prioritized(
                        propose_key_ids(
                            prev_state, bound_lvl, role_id, vocabs, edo=edo, section=sec_arg
                        ),
                        sec_key_ids,
                    ):
                        for chord_id in _shuffled_prioritized(propose_chord_ids(
                            prev_state, key_id, meter_id, beat_in_bar, bound_lvl,
                            role_id, groove_id, prior, context, vocabs, edo=edo
                        )):
                            for head_id in _shuffled_prioritized(propose_head_ids(chord_id, meter_id, beat_in_bar, bound_lvl, role_id, vocabs)):
                                yield BeatState(
                                    meter_id=meter_id,
                                    beat_in_bar=beat_in_bar,
                                    boundary_lvl=bound_lvl,
                                    key_id=key_id,
                                    chord_id=chord_id,
                                    role_id=role_id,
                                    head_id=head_id,
                                    groove_id=groove_id,
                                )


DEFAULT_PROPOSAL_BUDGET = 256
"""Default proposal budget (REQ-13): how many raw candidates are generated,
deduplicated, validated, and (optionally) scored before D_max is ever
consulted. Deliberately *not* derived from D_max -- D_max controls retained
outgoing edges only, and must stay decoupled from how broadly we search."""


def get_valid_next_states(
    prev_state: BeatState,
    t: int,
    key: RNGKey,
    d_max: int,
    style_config: Optional[StyleConfig] = None,
    vocabularies: Optional[Vocabularies] = None,
    prior: Optional[Prior] = None,
    context: Optional[PriorContext] = None,
    edo: Optional[int] = None,
    proposal_budget: Optional[int] = None,
    prior_guided_proposals: bool = False,
    section_view: Optional[EdgeSectionView] = None,
    section: Optional[PlanningSection] = None,
    section_guided_proposals: bool = False,
) -> tuple[CandidateGenerationResult, RNGKey]:
    """Generate a bounded pool of legal BeatState successors for one source state.

    REQ-13: proposal generation is bounded by ``proposal_budget`` (a beam
    width over *raw proposals*), not by ``d_max``. The full budgeted pool is
    deduplicated, legality-checked, and -- when ``prior_guided_proposals`` is
    set and a real prior is supplied -- batch-scored and ranked by prior
    log-probability with deterministic tie-breaking. ``d_max`` is not applied
    here at all; it is consulted exactly once, downstream, when
    ``aimusic.planning.graph.build_sparse_graph`` trims *scored edges* to the
    retained outdegree. This keeps D_max's two former meanings (proposal
    stopping condition vs. retained-edge cap) from colliding.

    ``d_max`` is still accepted (and still validated) here for API stability
    and because callers commonly want to size their own budget relative to
    it, but it no longer bounds what this function returns.

    REQ-04: Section structure is conveyed via ``section_view`` (an
    ``EdgeSectionView`` describing arrival state context at k = t + 1). When
    ``section_guided_proposals`` is enabled, proposal ordering prioritizes
    section preferences (meters, boundary levels, roles, groove families,
    and key regions). When proposal steering is disabled (the default),
    proposals follow baseline order without section bias. However, if the
    arrival state is a section start whose executable section boundary level
    is legal, the generator guarantees that at least one legal candidate with
    that boundary level is included within ``proposal_budget`` (by substituting
    the final candidate slot only when the budgeted pool would otherwise starve
    the boundary level, never bypassing legality checks).
    """
    if not isinstance(key, RNGKey):
        raise TypeError("key must be an RNGKey.")
    if not isinstance(d_max, int) or isinstance(d_max, bool) or d_max < 1:
        raise ValueError("d_max must be a positive int.")
    resolved_budget = DEFAULT_PROPOSAL_BUDGET if proposal_budget is None else proposal_budget
    if not isinstance(resolved_budget, int) or isinstance(resolved_budget, bool) or resolved_budget < 1:
        raise ValueError("proposal_budget must be a positive int.")

    resolved_vocabs = _resolved_vocabs(vocabularies)
    resolved_style = _resolved_style(style_config)
    resolved_prior = _resolved_prior(prior)
    resolved_edo = _edo_size(resolved_vocabs) if edo is None else edo
    validate_vocabulary_compatibility(resolved_vocabs, resolved_edo)

    effective_sec_view = section_view
    if effective_sec_view is None and context is not None:
        effective_sec_view = context.section_view

    sec: Optional[PlanningSection] = None
    is_start = False
    if effective_sec_view is not None:
        sec = getattr(effective_sec_view, "section", None)
        is_start = getattr(effective_sec_view, "is_section_start", False)
    elif section is not None:
        sec = section
        is_start = bool(t is not None and getattr(sec, "start_time", None) == t + 1)

    seen: set[BeatState] = set()
    accepted: list[BeatState] = []
    rejections: list[CandidateRejection] = []
    proposed_count = 0

    # 1. Generate: consume the (shuffled, lazy) generator up to the proposal
    #    budget -- this is the only place raw-proposal volume is bounded.
    key_state = [key]
    candidate_gen = _candidate_generator(
        prev_state,
        resolved_style,
        resolved_vocabs,
        resolved_prior,
        context,
        key,
        resolved_edo,
        key_state,
        section_view=effective_sec_view,
        section=sec,
        section_guided_proposals=section_guided_proposals,
        t=t,
    )

    for candidate in candidate_gen:
        if proposed_count >= resolved_budget:
            break
        proposed_count += 1

        # 2. Deduplicate.
        if candidate in seen:
            continue
        seen.add(candidate)

        # 3. Validate (legality checks are never bypassed, prior-guided or not).
        legal, reason = is_legal_transition(
            prev_state, candidate, resolved_style, resolved_vocabs
        )
        if legal:
            accepted.append(candidate)
        else:
            rejections.append(
                CandidateRejection(
                    time_index=t,
                    source_state=prev_state,
                    candidate_state=candidate,
                    reason=reason or "illegal_transition",
                )
            )

    # Section-start boundary level guarantee:
    target_bnd = getattr(sec, "boundary_level", 0) if sec is not None else 0
    if section_guided_proposals and is_start and target_bnd > 0:
        has_target_bnd = any(c.boundary_lvl == target_bnd for c in accepted)
        if not has_target_bnd:
            fallback_candidate = None
            sec_arg = sec if section_guided_proposals else None
            for m_id in propose_meter_ids(prev_state, resolved_style, resolved_vocabs, section=sec_arg):
                b_in_bar = _next_beat_index(prev_state, m_id, resolved_vocabs)
                avail_bnds = propose_boundary_levels(
                    prev_state,
                    m_id,
                    b_in_bar,
                    resolved_vocabs,
                    section_view=effective_sec_view,
                    section=sec,
                    section_guided_proposals=True,
                    t=t,
                )
                if target_bnd not in avail_bnds:
                    continue
                for r_id in propose_role_ids(
                    prev_state, m_id, b_in_bar, target_bnd, resolved_vocabs, section=sec_arg
                ):
                    for g_id in propose_groove_ids(
                        prev_state, target_bnd, r_id, resolved_vocabs, section=sec_arg
                    ):
                        for k_id in propose_key_ids(
                            prev_state, target_bnd, r_id, resolved_vocabs, edo=resolved_edo, section=sec_arg
                        ):
                            for c_id in propose_chord_ids(
                                prev_state, k_id, m_id, b_in_bar, target_bnd,
                                r_id, g_id, resolved_prior, context, resolved_vocabs, edo=resolved_edo
                            ):
                                for h_id in propose_head_ids(
                                    c_id, m_id, b_in_bar, target_bnd, r_id, resolved_vocabs
                                ):
                                    cand = BeatState(
                                        meter_id=m_id,
                                        beat_in_bar=b_in_bar,
                                        boundary_lvl=target_bnd,
                                        key_id=k_id,
                                        chord_id=c_id,
                                        role_id=r_id,
                                        head_id=h_id,
                                        groove_id=g_id,
                                    )
                                    is_leg, _ = is_legal_transition(
                                        prev_state, cand, resolved_style, resolved_vocabs
                                    )
                                    if is_leg:
                                        fallback_candidate = cand
                                        break
                                if fallback_candidate is not None:
                                    break
                            if fallback_candidate is not None:
                                break
                        if fallback_candidate is not None:
                            break
                    if fallback_candidate is not None:
                        break
                if fallback_candidate is not None:
                    break

            if fallback_candidate is not None:
                if len(accepted) >= resolved_budget:
                    accepted[-1] = fallback_candidate
                else:
                    accepted.append(fallback_candidate)
                seen.add(fallback_candidate)

    unique_count = len(seen)
    if section_guided_proposals and sec is not None:
        from aimusic.scoring.tension import section_style_energy

        legal_states = tuple(
            sorted(
                accepted,
                key=lambda s: (
                    section_style_energy(prev_state, s, effective_sec_view or sec, resolved_vocabs, resolved_edo),
                    _state_sort_key(s),
                ),
            )
        )
    else:
        legal_states = tuple(sorted(accepted, key=_state_sort_key))
    scores: Tuple[float, ...] = ()

    # 4. Batch-score (optional prior-guided ranking of the legal pool only).
    if prior_guided_proposals and legal_states and not isinstance(resolved_prior, NullPrior):
        queries = tuple(
            PriorQuery(
                prev_state=prev_state,
                next_state=candidate_state,
                time_index=t,
                context=context,
            )
            for candidate_state in legal_states
        )
        logps = prior_logps(resolved_prior, queries)
        ranked = sorted(
            zip(logps, legal_states),
            key=lambda item: (-item[0], _state_sort_key(item[1])),
        )
        legal_states = tuple(candidate_state for _, candidate_state in ranked)
        scores = tuple(float(logp) for logp, _ in ranked)

    result = CandidateGenerationResult(
        time_index=t,
        source_state=prev_state,
        states=legal_states,
        rejections=tuple(rejections),
        proposed_count=proposed_count,
        unique_count=unique_count,
        scores=scores,
    )
    # Only proposal work actually performed advances the supplied stream.
    return result, key_state[0]
