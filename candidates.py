from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

from config import StyleConfig
from core_types import BeatState
from gttm_features import beats_per_bar, is_strong_beat
from priors import NullPrior, Prior, PriorContext, PriorQuery, prior_logps
from tonal import get_fifth_steps, nearest_roots
from vocab import ChordToken, DEFAULT_VOCABULARIES, GrooveToken, Vocabularies


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

    def __post_init__(self) -> None:
        if not isinstance(self.source_state, BeatState):
            raise TypeError("source_state must be a BeatState.")
        if not isinstance(self.candidate_state, BeatState):
            raise TypeError("candidate_state must be a BeatState.")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string.")


@dataclass(frozen=True)
class CandidateGenerationResult:
    """Deterministic candidate-generation output plus rejection diagnostics."""

    time_index: int
    source_state: BeatState
    states: Tuple[BeatState, ...]
    rejections: Tuple[CandidateRejection, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.source_state, BeatState):
            raise TypeError("source_state must be a BeatState.")
        states = tuple(self.states)
        if any(not isinstance(state, BeatState) for state in states):
            raise TypeError("states must contain only BeatState instances.")
        if len(states) != len(set(states)):
            raise ValueError("states must be unique.")
        rejections = tuple(self.rejections)
        if any(not isinstance(rejection, CandidateRejection) for rejection in rejections):
            raise TypeError("rejections must contain only CandidateRejection instances.")
        object.__setattr__(self, "states", tuple(sorted(states, key=_state_sort_key)))
        object.__setattr__(self, "rejections", rejections)

    @property
    def proposed_count(self) -> int:
        return len(self.states) + len(self.rejections)


def apply_meter_constraints(
    prev_state: BeatState,
    next_candidate: BeatState,
    style_config: Optional[StyleConfig] = None,
    vocabularies: Optional[Vocabularies] = None,
) -> tuple[bool, Optional[str]]:
    resolved_vocabs = _resolved_vocabs(vocabularies)
    resolved_style = _resolved_style(style_config)
    allowed_meter_ids = set(_allowed_meter_ids(resolved_style, resolved_vocabs))

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
    resolved_vocabs = _resolved_vocabs(vocabularies)
    meter_ok, meter_reason = apply_meter_constraints(
        prev_state,
        next_candidate,
        style_config=style_config,
        vocabularies=resolved_vocabs,
    )
    if not meter_ok:
        return False, meter_reason

    beats = beats_per_bar(next_candidate.meter_id, resolved_vocabs.meters.id_map)
    if next_candidate.beat_in_bar < 0 or next_candidate.beat_in_bar >= beats:
        return False, "invalid_beat_index"

    expected_beat = _next_beat_index(prev_state, next_candidate.meter_id, resolved_vocabs)
    if next_candidate.beat_in_bar != expected_beat:
        return False, "non_contiguous_beat_progression"

    strong = is_strong_beat(next_candidate.beat_in_bar, beats)
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
    style_config: Optional[StyleConfig] = None,
    vocabularies: Optional[Vocabularies] = None,
) -> tuple[bool, Optional[str]]:
    del style_config
    resolved_vocabs = _resolved_vocabs(vocabularies)
    prev_role = _role_label(prev_state, resolved_vocabs)
    next_role = _role_label(next_candidate, resolved_vocabs)

    if next_role not in LEGAL_ROLE_SUCCESSORS[prev_role]:
        return False, "illegal_role_progression"

    beats = beats_per_bar(next_candidate.meter_id, resolved_vocabs.meters.id_map)
    strong = is_strong_beat(next_candidate.beat_in_bar, beats)
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
    vocabularies: Optional[Vocabularies] = None,
) -> tuple[bool, Optional[str]]:
    resolved_vocabs = _resolved_vocabs(vocabularies)
    prev_groove = _groove_token_by_id(prev_state.groove_id, resolved_vocabs)
    next_groove = _groove_token_by_id(next_candidate.groove_id, resolved_vocabs)
    if prev_groove.family != next_groove.family and next_candidate.boundary_lvl <= 0:
        return False, "groove_family_change_requires_boundary"

    if next_candidate.key_id != prev_state.key_id:
        next_role = _role_label(next_candidate, resolved_vocabs)
        if next_candidate.boundary_lvl < 2 and next_role not in {"change", "cad"}:
            return False, "key_change_requires_phrase_boundary_or_structural_role"

    head_label = _head_label(next_candidate, resolved_vocabs)
    beats = beats_per_bar(next_candidate.meter_id, resolved_vocabs.meters.id_map)
    strong = is_strong_beat(next_candidate.beat_in_bar, beats)
    if head_label in APPROACH_HEAD_LABELS and (strong or next_candidate.boundary_lvl > 0):
        return False, "approach_head_requires_weak_non_boundary_position"

    chord = _chord_token_by_id(next_candidate.chord_id, resolved_vocabs)
    if head_label == "seventh" and chord.quality != "7":
        return False, "seventh_head_requires_dominant_quality"

    return True, None


def is_legal_transition(
    prev_state: BeatState,
    next_candidate: BeatState,
    *,
    style_config: Optional[StyleConfig] = None,
    vocabularies: Optional[Vocabularies] = None,
) -> tuple[bool, Optional[str]]:
    resolved_vocabs = _resolved_vocabs(vocabularies)
    checks = (
        apply_meter_constraints(
            prev_state,
            next_candidate,
            style_config=style_config,
            vocabularies=resolved_vocabs,
        ),
        apply_position_constraints(
            prev_state,
            next_candidate,
            style_config=style_config,
            vocabularies=resolved_vocabs,
        ),
        apply_role_constraints(
            prev_state,
            next_candidate,
            style_config=style_config,
            vocabularies=resolved_vocabs,
        ),
        apply_boundary_and_groove_constraints(
            prev_state,
            next_candidate,
            vocabularies=resolved_vocabs,
        ),
    )
    for ok, reason in checks:
        if not ok:
            return False, reason
    return True, None


def propose_meter_ids(
    prev_state: BeatState,
    *,
    style_config: Optional[StyleConfig] = None,
    vocabularies: Optional[Vocabularies] = None,
) -> Tuple[int, ...]:
    resolved_vocabs = _resolved_vocabs(vocabularies)
    resolved_style = _resolved_style(style_config)

    proposals = [prev_state.meter_id]
    if prev_state.boundary_lvl >= 2 and prev_state.beat_in_bar == 0:
        proposals.extend(_allowed_meter_ids(resolved_style, resolved_vocabs))
    return tuple(dict.fromkeys(proposals))


def propose_boundary_levels(
    prev_state: BeatState,
    next_meter_id: int,
    next_beat_in_bar: int,
    *,
    vocabularies: Optional[Vocabularies] = None,
) -> Tuple[int, ...]:
    del prev_state
    resolved_vocabs = _resolved_vocabs(vocabularies)
    beats = beats_per_bar(next_meter_id, resolved_vocabs.meters.id_map)

    proposals = [resolved_vocabs.boundaries.token_for_label("none").id]
    if is_strong_beat(next_beat_in_bar, beats):
        proposals.append(resolved_vocabs.boundaries.token_for_label("local").id)
    if next_beat_in_bar == 0:
        proposals.append(resolved_vocabs.boundaries.token_for_label("phrase").id)
        proposals.append(resolved_vocabs.boundaries.token_for_label("section").id)
    return tuple(dict.fromkeys(proposals))


def propose_role_ids(
    prev_state: BeatState,
    next_meter_id: int,
    next_beat_in_bar: int,
    next_boundary_lvl: int,
    *,
    vocabularies: Optional[Vocabularies] = None,
) -> Tuple[int, ...]:
    resolved_vocabs = _resolved_vocabs(vocabularies)
    prev_role = _role_label(prev_state, resolved_vocabs)
    allowed_labels = set(LEGAL_ROLE_SUCCESSORS[prev_role])
    beats = beats_per_bar(next_meter_id, resolved_vocabs.meters.id_map)
    strong = is_strong_beat(next_beat_in_bar, beats)

    if not strong:
        allowed_labels.discard("cad")
    if next_boundary_lvl >= 2:
        allowed_labels.discard("hold")
    if next_boundary_lvl == 0:
        allowed_labels.discard("cad")

    return tuple(
        resolved_vocabs.roles.token_for_label(label).id
        for label in sorted(allowed_labels)
    )


def propose_key_ids(
    prev_state: BeatState,
    next_boundary_lvl: int,
    next_role_id: int,
    *,
    vocabularies: Optional[Vocabularies] = None,
) -> Tuple[int, ...]:
    resolved_vocabs = _resolved_vocabs(vocabularies)
    edo = _edo_size(resolved_vocabs)
    role_label = resolved_vocabs.roles.token_for_id(next_role_id).label

    proposals = [prev_state.key_id]
    if next_boundary_lvl >= 2 or role_label in {"change", "cad"}:
        for root_pc in nearest_roots(_key_root(prev_state, resolved_vocabs), edo, limit=2):
            if resolved_vocabs.keys.has_id(root_pc):
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
    *,
    prior: Optional[Prior] = None,
    context: Optional[PriorContext] = None,
    vocabularies: Optional[Vocabularies] = None,
    top_k_prior: int = 3,
) -> Tuple[int, ...]:
    resolved_vocabs = _resolved_vocabs(vocabularies)
    resolved_prior = _resolved_prior(prior)
    edo = _edo_size(resolved_vocabs)
    role_label = resolved_vocabs.roles.token_for_id(next_role_id).label
    prev_chord = _chord_token_by_id(prev_state.chord_id, resolved_vocabs)
    key_root = resolved_vocabs.keys.token_for_id(key_id).root_pc
    dominant_root = (key_root + get_fifth_steps(edo)) % edo

    proposals = [prev_state.chord_id]
    proposals.extend(_chord_ids_for_root(prev_chord.root_pc, _qualities_for_role(role_label), resolved_vocabs))
    for root_pc in nearest_roots(prev_chord.root_pc, edo, limit=2):
        proposals.extend(_chord_ids_for_root(root_pc, _qualities_for_role(role_label), resolved_vocabs))

    if role_label == "cad":
        proposals.extend(_chord_ids_for_root(key_root, ("maj", "min"), resolved_vocabs))
    else:
        proposals.extend(_chord_ids_for_root(key_root, ("maj", "min"), resolved_vocabs))
        proposals.extend(_chord_ids_for_root(dominant_root, ("7",), resolved_vocabs))

    proposals.extend(
        _top_k_prior_chord_ids(
            prev_state,
            next_meter_id,
            next_beat_in_bar,
            next_boundary_lvl,
            key_id,
            next_role_id,
            groove_id,
            prior=resolved_prior,
            context=context,
            vocabularies=resolved_vocabs,
            top_k=top_k_prior,
        )
    )

    return tuple(dict.fromkeys(proposals))


def propose_head_ids(
    chord_id: int,
    next_meter_id: int,
    next_beat_in_bar: int,
    next_boundary_lvl: int,
    next_role_id: int,
    *,
    vocabularies: Optional[Vocabularies] = None,
) -> Tuple[int, ...]:
    resolved_vocabs = _resolved_vocabs(vocabularies)
    chord = _chord_token_by_id(chord_id, resolved_vocabs)
    role_label = resolved_vocabs.roles.token_for_id(next_role_id).label
    beats = beats_per_bar(next_meter_id, resolved_vocabs.meters.id_map)
    strong = is_strong_beat(next_beat_in_bar, beats)

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

    return tuple(
        resolved_vocabs.heads.token_for_label(label).id
        for label in dict.fromkeys(labels)
    )


def propose_groove_ids(
    prev_state: BeatState,
    next_boundary_lvl: int,
    next_role_id: int,
    *,
    vocabularies: Optional[Vocabularies] = None,
) -> Tuple[int, ...]:
    resolved_vocabs = _resolved_vocabs(vocabularies)
    prev_groove = _groove_token_by_id(prev_state.groove_id, resolved_vocabs)
    next_role = resolved_vocabs.roles.token_for_id(next_role_id).label

    proposals = [prev_state.groove_id]
    for groove in resolved_vocabs.grooves:
        if groove.family == prev_groove.family:
            proposals.append(groove.id)

    if next_boundary_lvl > 0 or next_role in {"change", "cad"}:
        seen_families = {prev_groove.family}
        for groove in resolved_vocabs.grooves:
            if groove.family not in seen_families:
                proposals.append(groove.id)
                seen_families.add(groove.family)

    return tuple(dict.fromkeys(proposals))


def get_valid_next_states(
    prev_state: BeatState,
    t: int,
    *,
    style_config: Optional[StyleConfig] = None,
    vocabularies: Optional[Vocabularies] = None,
    prior: Optional[Prior] = None,
    context: Optional[PriorContext] = None,
) -> CandidateGenerationResult:
    """Generate legal BeatState successors for one source state."""
    resolved_vocabs = _resolved_vocabs(vocabularies)
    resolved_style = _resolved_style(style_config)
    resolved_prior = _resolved_prior(prior)

    accepted: set[BeatState] = set()
    rejections: list[CandidateRejection] = []

    for meter_id in propose_meter_ids(
        prev_state,
        style_config=resolved_style,
        vocabularies=resolved_vocabs,
    ):
        next_beat_in_bar = _next_beat_index(prev_state, meter_id, resolved_vocabs)
        for boundary_lvl in propose_boundary_levels(
            prev_state,
            meter_id,
            next_beat_in_bar,
            vocabularies=resolved_vocabs,
        ):
            for role_id in propose_role_ids(
                prev_state,
                meter_id,
                next_beat_in_bar,
                boundary_lvl,
                vocabularies=resolved_vocabs,
            ):
                for groove_id in propose_groove_ids(
                    prev_state,
                    boundary_lvl,
                    role_id,
                    vocabularies=resolved_vocabs,
                ):
                    for key_id in propose_key_ids(
                        prev_state,
                        boundary_lvl,
                        role_id,
                        vocabularies=resolved_vocabs,
                    ):
                        for chord_id in propose_chord_ids(
                            prev_state,
                            key_id,
                            meter_id,
                            next_beat_in_bar,
                            boundary_lvl,
                            role_id,
                            groove_id,
                            prior=resolved_prior,
                            context=context,
                            vocabularies=resolved_vocabs,
                        ):
                            for head_id in propose_head_ids(
                                chord_id,
                                meter_id,
                                next_beat_in_bar,
                                boundary_lvl,
                                role_id,
                                vocabularies=resolved_vocabs,
                            ):
                                candidate = BeatState(
                                    meter_id=meter_id,
                                    beat_in_bar=next_beat_in_bar,
                                    boundary_lvl=boundary_lvl,
                                    key_id=key_id,
                                    chord_id=chord_id,
                                    role_id=role_id,
                                    head_id=head_id,
                                    groove_id=groove_id,
                                )
                                legal, reason = is_legal_transition(
                                    prev_state,
                                    candidate,
                                    style_config=resolved_style,
                                    vocabularies=resolved_vocabs,
                                )
                                if legal:
                                    accepted.add(candidate)
                                else:
                                    rejections.append(
                                        CandidateRejection(
                                            time_index=t,
                                            source_state=prev_state,
                                            candidate_state=candidate,
                                            reason=reason or "illegal_transition",
                                        )
                                    )

    return CandidateGenerationResult(
        time_index=t,
        source_state=prev_state,
        states=tuple(sorted(accepted, key=_state_sort_key)),
        rejections=tuple(rejections),
    )
