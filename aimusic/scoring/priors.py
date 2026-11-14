from __future__ import annotations

import json
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple, cast, runtime_checkable

from aimusic.core.config import (
    NeuralPriorConfig,
    PlaceholderPriorMode,
    PriorFactorization,
    PriorWeights,
)
from aimusic.core.core_types import BeatState
from aimusic.core.vocab import Vocabularies
from aimusic.planning.sections import EdgeSectionView, PlanningSection
from aimusic.scoring.gttm_features import (
    TransitionWindow,
    transition_feature_vector,
    weighted_feature_breakdown,
)


STRUCTURAL_STREAM_NAMES: Tuple[str, ...] = (
    "meter",
    "beat_position",
    "boundary",
    "key",
    "chord",
    "role",
    "head",
    "groove",
)

SUPPORTED_STREAM_NAMES: Tuple[str, ...] = STRUCTURAL_STREAM_NAMES + ("section",)


MetadataPairs = Tuple[Tuple[str, str], ...]


def _require_int(name: str, value: int, *, minimum: int | None = None) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an int.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")


def _require_real(name: str, value: float) -> None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a real number.")
    if not isfinite(float(value)):
        raise ValueError(f"{name} must be finite.")


def _require_non_empty_str(name: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    if not value.strip():
        raise ValueError(f"{name} must not be empty.")


def _require_optional_path_str(name: str, value: Optional[str]) -> None:
    if value is None:
        return
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be None or a non-empty string.")


def _coerce_state_tuple(name: str, states: Sequence[BeatState]) -> Tuple[BeatState, ...]:
    items = tuple(states)
    if any(not isinstance(state, BeatState) for state in items):
        raise TypeError(f"{name} must contain only BeatState instances.")
    return items


def _coerce_metadata(name: str, metadata: Sequence[Tuple[str, str]]) -> MetadataPairs:
    items = tuple(metadata)
    normalized: list[Tuple[str, str]] = []
    for idx, item in enumerate(items):
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(f"{name}[{idx}] must be a 2-item (key, value) tuple.")
        key, value = item
        _require_non_empty_str(f"{name}[{idx}][0]", key)
        _require_non_empty_str(f"{name}[{idx}][1]", value)
        normalized.append((key, value))
    return tuple(normalized)


def _coerce_token_streams(streams: Sequence[str]) -> Tuple[str, ...]:
    items = tuple(streams)
    if not items:
        raise ValueError("token_streams must not be empty.")
    for idx, stream in enumerate(items):
        _require_non_empty_str(f"token_streams[{idx}]", stream)
        if stream not in SUPPORTED_STREAM_NAMES:
            raise ValueError(f"Unsupported token stream: {stream!r}")
    if len(items) != len(set(items)):
        raise ValueError("token_streams must not contain duplicates.")
    return items


@dataclass(frozen=True)
class SectionTokens:
    """Factorized section context tokens derived from an EdgeSectionView.

    Encapsulates timeline and section guidance for transition queries,
    following the arrival-state convention where edge t scores arrival state t + 1.
    """

    section_index: int
    section_name: str
    offset_in_section: int
    section_length: int
    is_section_start: bool
    target_tension: float
    groove_family: Optional[str] = None
    allowed_meters: Tuple[str, ...] = ()
    key_region: Optional[Tuple[int, ...]] = None
    boundary_level: Optional[int] = None
    target_density: Optional[float] = None
    preferred_roles: Optional[Tuple[str, ...]] = None
    target_key_id: Optional[int] = None

    def __post_init__(self) -> None:
        _require_int("section_index", self.section_index, minimum=0)
        _require_non_empty_str("section_name", self.section_name)
        _require_int("offset_in_section", self.offset_in_section, minimum=0)
        _require_int("section_length", self.section_length, minimum=1)
        if not isinstance(self.is_section_start, bool):
            raise TypeError("is_section_start must be a bool.")
        _require_real("target_tension", self.target_tension)
        if self.target_tension < 0.0 or self.target_tension > 1.0:
            raise ValueError("target_tension must be in [0, 1].")
        if self.groove_family is not None:
            _require_non_empty_str("groove_family", self.groove_family)
        if self.allowed_meters is not None:
            meters = tuple(self.allowed_meters)
            for idx, m in enumerate(meters):
                _require_non_empty_str(f"allowed_meters[{idx}]", m)
            object.__setattr__(self, "allowed_meters", meters)
        else:
            object.__setattr__(self, "allowed_meters", ())
        if self.key_region is not None:
            kr = tuple(self.key_region)
            for idx, k in enumerate(kr):
                _require_int(f"key_region[{idx}]", k, minimum=0)
            object.__setattr__(self, "key_region", kr)
        if self.boundary_level is not None:
            _require_int("boundary_level", self.boundary_level, minimum=0)
        if self.target_density is not None:
            _require_real("target_density", self.target_density)
        if self.preferred_roles is not None:
            roles = tuple(self.preferred_roles)
            for idx, r in enumerate(roles):
                _require_non_empty_str(f"preferred_roles[{idx}]", r)
            object.__setattr__(self, "preferred_roles", roles)
        if self.target_key_id is not None:
            _require_int("target_key_id", self.target_key_id, minimum=0)

    @classmethod
    def from_section_view(cls, view: EdgeSectionView) -> "SectionTokens":
        if not isinstance(view, EdgeSectionView):
            raise TypeError("view must be an EdgeSectionView.")
        sec = view.section
        sec_len = sec.end_time - sec.start_time
        return cls(
            section_index=view.section_index,
            section_name=sec.name,
            offset_in_section=view.offset_in_section,
            section_length=sec_len,
            is_section_start=view.is_section_start,
            target_tension=float(view.target_tension),
            groove_family=sec.groove_family,
            allowed_meters=tuple(sec.allowed_meters) if sec.allowed_meters is not None else (),
            key_region=tuple(sec.key_region) if sec.key_region is not None else None,
            boundary_level=sec.boundary_level,
            target_density=sec.target_density,
            preferred_roles=tuple(sec.preferred_roles) if sec.preferred_roles is not None else None,
            target_key_id=sec.target_key_id,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "section_index": self.section_index,
            "section_name": self.section_name,
            "offset_in_section": self.offset_in_section,
            "section_length": self.section_length,
            "is_section_start": self.is_section_start,
            "target_tension": float(self.target_tension),
            "groove_family": self.groove_family,
            "allowed_meters": list(self.allowed_meters),
            "key_region": list(self.key_region) if self.key_region is not None else None,
            "boundary_level": self.boundary_level,
            "target_density": self.target_density,
            "preferred_roles": list(self.preferred_roles) if self.preferred_roles is not None else None,
            "target_key_id": self.target_key_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SectionTokens":
        allowed = data.get("allowed_meters")
        key_reg = data.get("key_region")
        roles = data.get("preferred_roles")
        return cls(
            section_index=int(data["section_index"]),
            section_name=str(data["section_name"]),
            offset_in_section=int(data["offset_in_section"]),
            section_length=int(data["section_length"]),
            is_section_start=bool(data["is_section_start"]),
            target_tension=float(data["target_tension"]),
            groove_family=str(data["groove_family"]) if data.get("groove_family") is not None else None,
            allowed_meters=tuple(str(m) for m in allowed) if allowed is not None else (),
            key_region=tuple(int(k) for k in key_reg) if key_reg is not None else None,
            boundary_level=int(data["boundary_level"]) if data.get("boundary_level") is not None else None,
            target_density=float(data["target_density"]) if data.get("target_density") is not None else None,
            preferred_roles=tuple(str(r) for r in roles) if roles is not None else None,
            target_key_id=int(data["target_key_id"]) if data.get("target_key_id") is not None else None,
        )


@dataclass(frozen=True)
class StructuralEventTokens:
    """Factorized token ids for one BeatState event."""

    meter_id: int
    beat_position: int
    boundary_level: int
    key_id: int
    chord_id: int
    role_id: int
    head_id: int
    groove_id: int

    def __post_init__(self) -> None:
        for name in (
            "meter_id",
            "beat_position",
            "boundary_level",
            "key_id",
            "chord_id",
            "role_id",
            "head_id",
            "groove_id",
        ):
            _require_int(name, getattr(self, name), minimum=0)

    @classmethod
    def from_state(cls, state: BeatState) -> "StructuralEventTokens":
        return cls(
            meter_id=state.meter_id,
            beat_position=state.beat_in_bar,
            boundary_level=state.boundary_lvl,
            key_id=state.key_id,
            chord_id=state.chord_id,
            role_id=state.role_id,
            head_id=state.head_id,
            groove_id=state.groove_id,
        )

    def as_tuple(self) -> Tuple[int, ...]:
        return (
            self.meter_id,
            self.beat_position,
            self.boundary_level,
            self.key_id,
            self.chord_id,
            self.role_id,
            self.head_id,
            self.groove_id,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "meter_id": self.meter_id,
            "beat_position": self.beat_position,
            "boundary_level": self.boundary_level,
            "key_id": self.key_id,
            "chord_id": self.chord_id,
            "role_id": self.role_id,
            "head_id": self.head_id,
            "groove_id": self.groove_id,
        }


@dataclass(frozen=True)
class StructuralTokenSequence:
    """Factorized token streams derived from BeatState sequences.

    EPIC 3 standardizes on explicit per-dimension streams instead of one whole-state token
    so an external neural prior can consume stable vocabularies without combinatorial
    whole-state explosion.
    """

    meter_ids: Tuple[int, ...] = ()
    beat_positions: Tuple[int, ...] = ()
    boundary_levels: Tuple[int, ...] = ()
    key_ids: Tuple[int, ...] = ()
    chord_ids: Tuple[int, ...] = ()
    role_ids: Tuple[int, ...] = ()
    head_ids: Tuple[int, ...] = ()
    groove_ids: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        fields = (
            "meter_ids",
            "beat_positions",
            "boundary_levels",
            "key_ids",
            "chord_ids",
            "role_ids",
            "head_ids",
            "groove_ids",
        )
        lengths = set()
        for name in fields:
            values = tuple(getattr(self, name))
            for idx, value in enumerate(values):
                _require_int(f"{name}[{idx}]", value, minimum=0)
            object.__setattr__(self, name, values)
            lengths.add(len(values))

        if len(lengths) > 1:
            raise ValueError("All token streams must have the same length.")

    @classmethod
    def from_states(cls, states: Sequence[BeatState]) -> "StructuralTokenSequence":
        items = _coerce_state_tuple("states", states)
        return cls(
            meter_ids=tuple(state.meter_id for state in items),
            beat_positions=tuple(state.beat_in_bar for state in items),
            boundary_levels=tuple(state.boundary_lvl for state in items),
            key_ids=tuple(state.key_id for state in items),
            chord_ids=tuple(state.chord_id for state in items),
            role_ids=tuple(state.role_id for state in items),
            head_ids=tuple(state.head_id for state in items),
            groove_ids=tuple(state.groove_id for state in items),
        )

    def __len__(self) -> int:
        return len(self.meter_ids)

    def event_at(self, index: int) -> StructuralEventTokens:
        _require_int("index", index, minimum=0)
        if index >= len(self):
            raise IndexError("index out of range for token sequence.")
        return StructuralEventTokens(
            meter_id=self.meter_ids[index],
            beat_position=self.beat_positions[index],
            boundary_level=self.boundary_levels[index],
            key_id=self.key_ids[index],
            chord_id=self.chord_ids[index],
            role_id=self.role_ids[index],
            head_id=self.head_ids[index],
            groove_id=self.groove_ids[index],
        )

    def to_dict(self) -> dict[str, list[int]]:
        return {
            "meter_ids": list(self.meter_ids),
            "beat_positions": list(self.beat_positions),
            "boundary_levels": list(self.boundary_levels),
            "key_ids": list(self.key_ids),
            "chord_ids": list(self.chord_ids),
            "role_ids": list(self.role_ids),
            "head_ids": list(self.head_ids),
            "groove_ids": list(self.groove_ids),
        }


@dataclass(frozen=True, init=False)
class PriorContext:
    """Optional transition-scoring context for external priors."""

    history: Tuple[BeatState, ...] = ()
    future_hints: Tuple[BeatState, ...] = ()
    section_name: Optional[str] = None
    section_view: Optional[EdgeSectionView] = None
    metadata: MetadataPairs = ()
    history_tokens: Optional[StructuralTokenSequence] = None
    future_hint_tokens: Optional[StructuralTokenSequence] = None

    def __init__(
        self,
        history: Sequence[BeatState] = (),
        future_hints: Sequence[BeatState] = (),
        section_name: Optional[str] = None,
        section_view: Optional[EdgeSectionView] = None,
        metadata: Sequence[Tuple[str, str]] = (),
        history_tokens: Optional[StructuralTokenSequence] = None,
        future_hint_tokens: Optional[StructuralTokenSequence] = None,
        *,
        section: Optional[object] = None,
        target_tension: Optional[float] = None,
    ) -> None:
        history_coerced = _coerce_state_tuple("history", history)
        future_hints_coerced = _coerce_state_tuple("future_hints", future_hints)
        object.__setattr__(self, "history", history_coerced)
        object.__setattr__(self, "future_hints", future_hints_coerced)

        resolved_view = section_view
        if resolved_view is None and (section is not None or target_tension is not None):
            from aimusic.planning.sections import EdgeSectionView, PlanningSection

            if isinstance(section, PlanningSection):
                resolved_view = EdgeSectionView(
                    section=section,
                    section_index=0,
                    offset_in_section=0,
                    is_section_start=False,
                    target_tension=float(target_tension) if target_tension is not None else 0.5,
                    arrival_time=1,
                )

        resolved_sec_name = section_name
        if resolved_sec_name is None and resolved_view is not None:
            resolved_sec_name = resolved_view.section.name
        if resolved_sec_name is not None:
            _require_non_empty_str("section_name", resolved_sec_name)

        object.__setattr__(self, "section_name", resolved_sec_name)
        object.__setattr__(self, "section_view", resolved_view)

        meta_coerced = _coerce_metadata("metadata", metadata)
        object.__setattr__(self, "metadata", meta_coerced)

        hist_tokens = (
            StructuralTokenSequence.from_states(history_coerced)
            if history_tokens is None
            else history_tokens
        )
        fut_tokens = (
            StructuralTokenSequence.from_states(future_hints_coerced)
            if future_hint_tokens is None
            else future_hint_tokens
        )

        if not isinstance(hist_tokens, StructuralTokenSequence):
            raise TypeError("history_tokens must be a StructuralTokenSequence.")
        if not isinstance(fut_tokens, StructuralTokenSequence):
            raise TypeError("future_hint_tokens must be a StructuralTokenSequence.")
        if len(hist_tokens) != len(history_coerced):
            raise ValueError("history_tokens length must match history length.")
        if len(fut_tokens) != len(future_hints_coerced):
            raise ValueError("future_hint_tokens length must match future_hints length.")

        object.__setattr__(self, "history_tokens", hist_tokens)
        object.__setattr__(self, "future_hint_tokens", fut_tokens)

    @property
    def section(self) -> Optional[PlanningSection]:
        return self.section_view.section if self.section_view is not None else None

    @property
    def target_tension(self) -> Optional[float]:
        return self.section_view.target_tension if self.section_view is not None else None

    @property
    def section_tokens(self) -> Optional[SectionTokens]:
        return (
            SectionTokens.from_section_view(self.section_view)
            if self.section_view is not None
            else None
        )

    def to_dict(self) -> dict[str, object]:
        assert self.history_tokens is not None
        assert self.future_hint_tokens is not None
        return {
            "history_length": len(self.history),
            "future_hint_length": len(self.future_hints),
            "section_name": self.section_name,
            "metadata": list(self.metadata),
            "history_tokens": self.history_tokens.to_dict(),
            "future_hint_tokens": self.future_hint_tokens.to_dict(),
        }


@dataclass(frozen=True)
class PriorQuery:
    """One transition query against a data prior."""

    prev_state: BeatState
    next_state: BeatState
    time_index: int
    context: Optional[PriorContext] = None

    def __post_init__(self) -> None:
        if not isinstance(self.prev_state, BeatState):
            raise TypeError("prev_state must be a BeatState.")
        if not isinstance(self.next_state, BeatState):
            raise TypeError("next_state must be a BeatState.")
        _require_int("time_index", self.time_index, minimum=0)
        if self.context is not None and not isinstance(self.context, PriorContext):
            raise TypeError("context must be a PriorContext or None.")

    def tokenize(
        self,
        factorization_mode: PriorFactorization = PriorFactorization.FACTORIZED,
    ) -> "TokenizedPriorQuery":
        return TokenizedPriorQuery.from_query(self, factorization_mode=factorization_mode)


@dataclass(frozen=True)
class TokenizedPriorQuery:
    """Stable model-facing payload for a neural prior wrapper."""

    prev_event: StructuralEventTokens
    next_event: StructuralEventTokens
    time_index: int
    history_tokens: StructuralTokenSequence = field(default_factory=StructuralTokenSequence)
    future_hint_tokens: StructuralTokenSequence = field(default_factory=StructuralTokenSequence)
    section_name: Optional[str] = None
    section_tokens: Optional[SectionTokens] = None
    metadata: MetadataPairs = ()
    factorization_mode: PriorFactorization = PriorFactorization.FACTORIZED

    def __post_init__(self) -> None:
        if not isinstance(self.prev_event, StructuralEventTokens):
            raise TypeError("prev_event must be a StructuralEventTokens.")
        if not isinstance(self.next_event, StructuralEventTokens):
            raise TypeError("next_event must be a StructuralEventTokens.")
        _require_int("time_index", self.time_index, minimum=0)
        if not isinstance(self.history_tokens, StructuralTokenSequence):
            raise TypeError("history_tokens must be a StructuralTokenSequence.")
        if not isinstance(self.future_hint_tokens, StructuralTokenSequence):
            raise TypeError("future_hint_tokens must be a StructuralTokenSequence.")
        if self.section_name is not None:
            _require_non_empty_str("section_name", self.section_name)
        if self.section_tokens is not None and not isinstance(self.section_tokens, SectionTokens):
            raise TypeError("section_tokens must be a SectionTokens or None.")
        metadata = _coerce_metadata("metadata", self.metadata)
        object.__setattr__(self, "metadata", metadata)
        if not isinstance(self.factorization_mode, PriorFactorization):
            raise TypeError("factorization_mode must be a PriorFactorization value.")

    @classmethod
    def from_query(
        cls,
        query: PriorQuery,
        factorization_mode: PriorFactorization = PriorFactorization.FACTORIZED,
    ) -> "TokenizedPriorQuery":
        if not isinstance(factorization_mode, PriorFactorization):
            raise TypeError("factorization_mode must be a PriorFactorization value.")

        context = query.context or PriorContext()
        assert context.history_tokens is not None
        assert context.future_hint_tokens is not None
        section_tokens = (
            SectionTokens.from_section_view(context.section_view)
            if context.section_view is not None
            else None
        )
        return cls(
            prev_event=StructuralEventTokens.from_state(query.prev_state),
            next_event=StructuralEventTokens.from_state(query.next_state),
            time_index=query.time_index,
            history_tokens=context.history_tokens,
            future_hint_tokens=context.future_hint_tokens,
            section_name=context.section_name,
            section_tokens=section_tokens,
            metadata=context.metadata,
            factorization_mode=factorization_mode,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "factorization_mode": self.factorization_mode.value,
            "prev_event": self.prev_event.to_dict(),
            "next_event": self.next_event.to_dict(),
            "time_index": self.time_index,
            "history_tokens": self.history_tokens.to_dict(),
            "future_hint_tokens": self.future_hint_tokens.to_dict(),
            "section_name": self.section_name,
            "section_tokens": self.section_tokens.to_dict() if self.section_tokens is not None else None,
            "metadata": list(self.metadata),
        }


@runtime_checkable
class Prior(Protocol):
    """General transition-prior contract consumed by future graph code."""

    def logp_next(
        self,
        prev_state: BeatState,
        next_state: BeatState,
        t: int,
        context: Optional[PriorContext] = None,
    ) -> float:
        ...


@runtime_checkable
class BatchedPrior(Prior, Protocol):
    """Optional batch-scoring contract for priors."""

    def logp_next_batch(self, queries: Sequence[PriorQuery]) -> Tuple[float, ...]:
        ...


@runtime_checkable
class NeuralPriorModel(Protocol):
    """External neural-model contract used by the wrapper."""

    def score_transition(self, query: TokenizedPriorQuery) -> float:
        ...


@runtime_checkable
class BatchedNeuralPriorModel(NeuralPriorModel, Protocol):
    """Optional batched extension for external neural models."""

    def score_transition_batch(
        self,
        queries: Sequence[TokenizedPriorQuery],
    ) -> Tuple[float, ...]:
        ...


@dataclass(frozen=True)
class NeuralPriorManifest:
    """Serializable artifact manifest for an external neural prior bundle."""

    manifest_version: int = 2
    model_family: str = "external_neural_prior"
    model_version: str = "placeholder-v1"
    factorization_mode: PriorFactorization = PriorFactorization.FACTORIZED
    token_streams: Tuple[str, ...] = STRUCTURAL_STREAM_NAMES
    checkpoint_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    supports_batch_scoring: bool = True
    expected_edo: Optional[int] = None
    metadata: MetadataPairs = ()

    def __post_init__(self) -> None:
        _require_int("manifest_version", self.manifest_version, minimum=1)
        _require_non_empty_str("model_family", self.model_family)
        _require_non_empty_str("model_version", self.model_version)
        if not isinstance(self.factorization_mode, PriorFactorization):
            raise TypeError("factorization_mode must be a PriorFactorization value.")
        object.__setattr__(self, "token_streams", _coerce_token_streams(self.token_streams))
        _require_optional_path_str("checkpoint_path", self.checkpoint_path)
        _require_optional_path_str("tokenizer_path", self.tokenizer_path)
        if not isinstance(self.supports_batch_scoring, bool):
            raise TypeError("supports_batch_scoring must be a bool.")
        if self.expected_edo is not None:
            _require_int("expected_edo", self.expected_edo, minimum=1)
        object.__setattr__(self, "metadata", _coerce_metadata("metadata", self.metadata))

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_version": self.manifest_version,
            "model_family": self.model_family,
            "model_version": self.model_version,
            "factorization_mode": self.factorization_mode.value,
            "token_streams": list(self.token_streams),
            "checkpoint_path": self.checkpoint_path,
            "tokenizer_path": self.tokenizer_path,
            "supports_batch_scoring": self.supports_batch_scoring,
            "expected_edo": self.expected_edo,
            "metadata": [list(item) for item in self.metadata],
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "NeuralPriorManifest":
        return cls(
            manifest_version=int(cast(int, data.get("manifest_version", 1))),
            model_family=str(data.get("model_family", "external_neural_prior")),
            model_version=str(data.get("model_version", "placeholder-v1")),
            factorization_mode=PriorFactorization(
                data.get("factorization_mode", PriorFactorization.FACTORIZED.value)
            ),
            token_streams=tuple(cast(Sequence[str], data.get("token_streams", STRUCTURAL_STREAM_NAMES))),
            checkpoint_path=data.get("checkpoint_path"),  # type: ignore[arg-type]
            tokenizer_path=data.get("tokenizer_path"),  # type: ignore[arg-type]
            supports_batch_scoring=bool(data.get("supports_batch_scoring", True)),
            expected_edo=data.get("expected_edo"),  # type: ignore[arg-type]
            metadata=cast(MetadataPairs, tuple(tuple(item) for item in cast(Sequence[Tuple[str, str]], data.get("metadata", ())))),
        )


def build_neural_prior_manifest(config: NeuralPriorConfig) -> NeuralPriorManifest:
    """Build a manifest shell from runtime prior configuration."""
    if not isinstance(config, NeuralPriorConfig):
        raise TypeError("config must be a NeuralPriorConfig.")
    return NeuralPriorManifest(
        model_family=config.model_family,
        model_version=config.model_version,
        factorization_mode=config.factorization_mode,
        checkpoint_path=config.checkpoint_path,
        tokenizer_path=config.tokenizer_path,
        supports_batch_scoring=config.supports_batch_scoring,
    )


def save_neural_prior_manifest(manifest: NeuralPriorManifest, path: str) -> None:
    """Persist a neural-prior manifest as JSON."""
    if not isinstance(manifest, NeuralPriorManifest):
        raise TypeError("manifest must be a NeuralPriorManifest.")
    _require_non_empty_str("path", path)

    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_neural_prior_manifest(path: str) -> NeuralPriorManifest:
    """Load a neural-prior manifest from JSON."""
    _require_non_empty_str("path", path)
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Manifest JSON must decode to an object.")
    return NeuralPriorManifest.from_dict(data)


@dataclass(frozen=True)
class NullPrior:
    """Algorithmic-only prior that contributes a constant neutral log-probability.

    Note: This prior contributes a static neutral log-probability and does not consume
    or use SectionTokens or section context to influence its output.
    """

    neutral_logp: float = 0.0

    def __post_init__(self) -> None:
        _require_real("neutral_logp", self.neutral_logp)

    def logp_next(
        self,
        prev_state: BeatState,
        next_state: BeatState,
        t: int,
        context: Optional[PriorContext] = None,
    ) -> float:
        del prev_state, next_state, t, context
        return float(self.neutral_logp)

    def logp_next_batch(self, queries: Sequence[PriorQuery]) -> Tuple[float, ...]:
        return tuple(self.neutral_logp for _ in tuple(queries))


@dataclass(frozen=True)
class NeuralPrior:
    """Wrapper around an external neural prior or a deterministic placeholder.

    Note: The built-in deterministic placeholder mode does not consume or use
    SectionTokens to alter scoring behavior, preserving baseline transitions. External
    models supplied via the `model` argument receive full TokenizedPriorQuery objects
    and may consume section_tokens if supported.
    """

    config: NeuralPriorConfig = field(default_factory=NeuralPriorConfig)
    manifest: Optional[NeuralPriorManifest] = None
    model: Optional[NeuralPriorModel] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.config, NeuralPriorConfig):
            raise TypeError("config must be a NeuralPriorConfig.")

        manifest = build_neural_prior_manifest(self.config) if self.manifest is None else self.manifest
        if not isinstance(manifest, NeuralPriorManifest):
            raise TypeError("manifest must be a NeuralPriorManifest.")
        if manifest.factorization_mode is not self.config.factorization_mode:
            raise ValueError("config and manifest factorization_mode must match.")
        object.__setattr__(self, "manifest", manifest)

        if self.model is not None and not isinstance(self.model, NeuralPriorModel):
            raise TypeError("model must satisfy the NeuralPriorModel protocol.")

    def _score_placeholder(self, query: TokenizedPriorQuery) -> float:
        """Deterministic placeholder scoring based on factorized token agreements.

        Note: Section tokens (query.section_tokens) are explicitly ignored by this
        placeholder to preserve existing graph weights and baseline transitions.
        """
        assert self.manifest is not None
        score = float(self.config.default_logp)
        if self.config.placeholder_mode is PlaceholderPriorMode.NEUTRAL:
            return score

        active_streams = set(self.manifest.token_streams)
        prev_event = query.prev_event
        next_event = query.next_event

        if "meter" in active_streams:
            score += 0.04 if prev_event.meter_id == next_event.meter_id else -0.04
        if "beat_position" in active_streams:
            score += 0.002 * float(next_event.beat_position % 4)
        if "boundary" in active_streams:
            score += 0.015 * float(next_event.boundary_level)
        if "key" in active_streams:
            score += 0.06 if prev_event.key_id == next_event.key_id else -0.03
        if "chord" in active_streams:
            score += 0.08 if prev_event.chord_id == next_event.chord_id else -0.02
        if "role" in active_streams:
            score += 0.03 if prev_event.role_id == next_event.role_id else 0.01
        if "head" in active_streams:
            score += 0.02 if next_event.head_id != 0 else -0.01
        if "groove" in active_streams:
            score += 0.02 if prev_event.groove_id == next_event.groove_id else -0.01

        if len(query.history_tokens) > 0:
            history_tail = query.history_tokens.event_at(len(query.history_tokens) - 1)
            if "chord" in active_streams and history_tail.chord_id == next_event.chord_id:
                score += 0.015
            if "key" in active_streams and history_tail.key_id == next_event.key_id:
                score += 0.01

        if query.section_name is not None:
            score += (sum(ord(char) for char in query.section_name) % 5) * 0.001

        if query.factorization_mode is PriorFactorization.MIXED:
            score += 0.005
        elif query.factorization_mode is PriorFactorization.WHOLE_STATE:
            score -= 0.005

        return float(score)

    def _score_tokenized_query(self, query: TokenizedPriorQuery) -> float:
        if self.model is not None:
            return float(self.model.score_transition(query))
        return self._score_placeholder(query)

    def logp_next(
        self,
        prev_state: BeatState,
        next_state: BeatState,
        t: int,
        context: Optional[PriorContext] = None,
    ) -> float:
        assert self.manifest is not None
        query = PriorQuery(prev_state=prev_state, next_state=next_state, time_index=t, context=context)
        return self._score_tokenized_query(query.tokenize(self.manifest.factorization_mode))

    def logp_next_batch(self, queries: Sequence[PriorQuery]) -> Tuple[float, ...]:
        assert self.manifest is not None
        query_items = tuple(queries)
        if any(not isinstance(query, PriorQuery) for query in query_items):
            raise TypeError("queries must contain only PriorQuery instances.")

        tokenized_queries = tuple(
            query.tokenize(self.manifest.factorization_mode) for query in query_items
        )
        if (
            self.model is not None
            and self.config.supports_batch_scoring
            and self.manifest.supports_batch_scoring
            and isinstance(self.model, BatchedNeuralPriorModel)
        ):
            scores = tuple(float(score) for score in self.model.score_transition_batch(tokenized_queries))
            if len(scores) != len(tokenized_queries):
                raise ValueError("score_transition_batch must return one score per query.")
            return scores
        return tuple(self._score_tokenized_query(query) for query in tokenized_queries)


def prior_logps(prior: Prior, queries: Sequence[PriorQuery]) -> Tuple[float, ...]:
    """Score a batch of prior queries with or without a native batch API."""
    query_items = tuple(queries)
    if isinstance(prior, BatchedPrior):
        return tuple(float(score) for score in prior.logp_next_batch(query_items))
    return tuple(
        float(prior.logp_next(query.prev_state, query.next_state, query.time_index, query.context))
        for query in query_items
    )


@dataclass(frozen=True)
class TransitionScoreBreakdown:
    """All components used to produce one production transition weight."""

    raw_feature_contributions: Mapping[str, float]
    weighted_feature_contributions: Mapping[str, float]
    data_logp: float
    data_contribution: float
    gttm_score: float
    gttm_energy: float
    gttm_contribution: float
    final_log_weight: float
    target_tension: Optional[float] = None
    realized_tension: Optional[float] = None
    tension_deviation: float = 0.0
    target_tension_energy: float = 0.0
    target_tension_contribution: float = 0.0
    section_style_components: Mapping[str, float] = field(default_factory=dict)
    section_style_energy: float = 0.0
    section_style_contribution: float = 0.0
    prev_state: Optional[BeatState] = None
    next_state: Optional[BeatState] = None

    def check_decomposition(self, tol: float = 1e-9) -> None:
        """Check that final_log_weight equals data + gttm - tension_contribution - style_contribution within tol."""
        expected = (
            self.data_contribution
            + self.gttm_contribution
            - self.target_tension_contribution
            - self.section_style_contribution
        )
        diff = abs(self.final_log_weight - expected)
        if diff > tol:
            raise ValueError(
                f"TransitionScoreBreakdown decomposition violated: "
                f"final_log_weight={self.final_log_weight} != "
                f"data ({self.data_contribution}) + gttm ({self.gttm_contribution}) - "
                f"target_tension ({self.target_tension_contribution}) - "
                f"section_style ({self.section_style_contribution}). "
                f"Expected {expected}, discrepancy {diff} > tol {tol}."
            )


def score_transition_breakdown(
    query: PriorQuery,
    window: Optional[TransitionWindow],
    weights: PriorWeights,
    vocabularies: Vocabularies,
    edo: int,
    data_logp: float,
    *,
    meters=None,
) -> TransitionScoreBreakdown:
    """Compute base data and GTTM terms for one transition query and window."""
    raw = transition_feature_vector(
        query.prev_state,
        query.next_state,
        query.time_index,
        window=window,
        meters=meters,
        vocabularies=vocabularies,
        edo=edo,
    )
    weighted = weighted_feature_breakdown(
        query.prev_state,
        query.next_state,
        query.time_index,
        window=window,
        meters=meters,
        vocabularies=vocabularies,
        edo=edo,
        weights=weights,
    )
    gttm_score = float(sum(weighted.values()))
    gttm_energy = -gttm_score
    data_contribution = float(weights.lambda_data * data_logp)
    gttm_contribution = float(-weights.lambda_gttm * gttm_energy)
    final_log_weight = float(data_contribution + gttm_contribution)
    return TransitionScoreBreakdown(
        raw_feature_contributions=dict(raw),
        weighted_feature_contributions=dict(weighted),
        data_logp=float(data_logp),
        data_contribution=data_contribution,
        gttm_score=gttm_score,
        gttm_energy=gttm_energy,
        gttm_contribution=gttm_contribution,
        final_log_weight=final_log_weight,
        prev_state=query.prev_state,
        next_state=query.next_state,
    )


# Backward-compatible alias
_transition_score_breakdown_from_data_logp = score_transition_breakdown


def with_section_terms(
    breakdown: TransitionScoreBreakdown,
    section_view: Optional[EdgeSectionView],
    weights: PriorWeights,
    vocabularies: Vocabularies,
    edo: int,
    *,
    prev_state: Optional[BeatState] = None,
    next_state: Optional[BeatState] = None,
) -> TransitionScoreBreakdown:
    """Add target tension and section style terms for the arrival state once per edge."""
    prev = prev_state if prev_state is not None else breakdown.prev_state
    next_st = next_state if next_state is not None else breakdown.next_state

    if section_view is None or next_st is None:
        return breakdown

    target_tension = section_view.target_tension
    from aimusic.scoring.tension import (
        beat_tension,
        section_style_breakdown,
        section_style_energy,
        transition_target_tension_energy,
    )

    realized_tension = beat_tension(prev, next_st, vocabularies, edo)
    tension_deviation = realized_tension - target_tension
    target_tension_energy = transition_target_tension_energy(
        realized_tension, target_tension, weights.sigma_tension
    )
    target_tension_contribution = float(weights.lambda_target_tension * target_tension_energy)

    is_multi_meter = getattr(section_view, "multi_meter", False)
    if not is_multi_meter and section_view.section.allowed_meters:
        is_multi_meter = len(section_view.section.allowed_meters) > 1

    style_energy = float(
        section_style_energy(
            prev,
            next_st,
            section_view,
            vocabularies,
            edo,
            style_weights=weights.style_weights,
            multi_meter=is_multi_meter,
            multi_meter_alignment_factor=weights.multi_meter_alignment_factor,
        )
    )
    style_breakdown = section_style_breakdown(
        prev,
        next_st,
        section_view,
        vocabularies,
        edo,
        style_weights=weights.style_weights,
    )
    style_components = dict(style_breakdown)
    section_style_contribution = float(weights.lambda_section_style * style_energy)

    final_log_weight = float(
        breakdown.data_contribution
        + breakdown.gttm_contribution
        - target_tension_contribution
        - section_style_contribution
    )

    return TransitionScoreBreakdown(
        raw_feature_contributions=breakdown.raw_feature_contributions,
        weighted_feature_contributions=breakdown.weighted_feature_contributions,
        data_logp=breakdown.data_logp,
        data_contribution=breakdown.data_contribution,
        gttm_score=breakdown.gttm_score,
        gttm_energy=breakdown.gttm_energy,
        gttm_contribution=breakdown.gttm_contribution,
        final_log_weight=final_log_weight,
        target_tension=target_tension,
        realized_tension=realized_tension,
        tension_deviation=tension_deviation,
        target_tension_energy=target_tension_energy,
        target_tension_contribution=target_tension_contribution,
        section_style_components=style_components,
        section_style_energy=style_energy,
        section_style_contribution=section_style_contribution,
        prev_state=prev,
        next_state=next_st,
    )


def calculate_transition_score_breakdown(
    prev_state: BeatState,
    next_state: BeatState,
    t: int,
    *,
    prior: Prior,
    context: Optional[PriorContext] = None,
    window: Optional[TransitionWindow] = None,
    weights: Optional[PriorWeights] = None,
    meters=None,
    vocabularies=None,
    edo: Optional[int] = None,
) -> TransitionScoreBreakdown:
    """Scalar transition scoring with raw and weighted feature diagnostics."""
    resolved_weights = PriorWeights() if weights is None else weights
    from aimusic.core.vocab import DEFAULT_VOCABULARIES
    resolved_vocabs = DEFAULT_VOCABULARIES if vocabularies is None else vocabularies
    resolved_edo = len(resolved_vocabs.keys) if edo is None else edo

    query = PriorQuery(prev_state, next_state, t, context)
    data_logp = float(prior.logp_next(prev_state, next_state, t, context))
    base = score_transition_breakdown(
        query,
        window,
        resolved_weights,
        resolved_vocabs,
        resolved_edo,
        data_logp,
        meters=meters,
    )
    sec_view = context.section_view if context is not None else None
    return with_section_terms(
        base,
        sec_view,
        resolved_weights,
        resolved_vocabs,
        resolved_edo,
    )


def calculate_transition_score_breakdowns(
    queries: Sequence[PriorQuery],
    *,
    prior: Prior,
    windows: Optional[Sequence[Optional[TransitionWindow]]] = None,
    weights: Optional[PriorWeights] = None,
    meters=None,
    vocabularies=None,
    edo: Optional[int] = None,
    include_section_terms: bool = True,
) -> Tuple[TransitionScoreBreakdown, ...]:
    """Batch transition scoring with windows aligned one-to-one with queries."""
    query_items = tuple(queries)
    resolved_weights = PriorWeights() if weights is None else weights
    from aimusic.core.vocab import DEFAULT_VOCABULARIES
    resolved_vocabs = DEFAULT_VOCABULARIES if vocabularies is None else vocabularies
    resolved_edo = len(resolved_vocabs.keys) if edo is None else edo

    data_scores = prior_logps(prior, query_items)
    if len(data_scores) != len(query_items):
        raise ValueError("prior scoring must return one score per query.")

    if windows is None:
        window_items: tuple[TransitionWindow | None, ...] = (None,) * len(query_items)
    else:
        window_items = tuple(windows)
        if len(window_items) != len(query_items):
            raise ValueError("windows must align 1:1 with queries.")

    results: list[TransitionScoreBreakdown] = []
    for idx, query in enumerate(query_items):
        base = score_transition_breakdown(
            query,
            window_items[idx],
            resolved_weights,
            resolved_vocabs,
            resolved_edo,
            data_scores[idx],
            meters=meters,
        )
        if include_section_terms:
            sec_view = query.context.section_view if query.context is not None else None
            results.append(
                with_section_terms(
                    base,
                    sec_view,
                    resolved_weights,
                    resolved_vocabs,
                    resolved_edo,
                )
            )
        else:
            results.append(base)
    return tuple(results)


def calculate_transition_log_weight(
    prev_state: BeatState,
    next_state: BeatState,
    t: int,
    *,
    prior: Prior,
    context: Optional[PriorContext] = None,
    window: Optional[TransitionWindow] = None,
    weights: Optional[PriorWeights] = None,
    meters=None,
    vocabularies=None,
    edo: Optional[int] = None,
    target_tension: Optional[float] = None,
    section: Optional[object] = None,
    section_view: Optional[EdgeSectionView] = None,
) -> float:
    """Combine prior data likelihood, GTTM energy, target tension deviation, and section style energy into one edge weight."""
    effective_context = context
    effective_view = section_view
    if effective_view is None and context is not None:
        effective_view = context.section_view

    if effective_view is None and (target_tension is not None or section is not None):
        from aimusic.planning.sections import EdgeSectionView, PlanningSection

        sec_obj = section if isinstance(section, PlanningSection) else (
            context.section if context is not None and isinstance(context.section, PlanningSection) else None
        )
        if sec_obj is not None:
            tau = target_tension if target_tension is not None else (
                context.target_tension if context is not None and context.target_tension is not None else 0.5
            )
            effective_view = EdgeSectionView(
                section=sec_obj,
                section_index=0,
                offset_in_section=max(0, t + 1 - sec_obj.start_time),
                is_section_start=(t + 1 == sec_obj.start_time),
                target_tension=tau,
                arrival_time=t + 1,
            )
            history = context.history if context is not None else (prev_state,)
            effective_context = PriorContext(history=history, section_view=effective_view)

    breakdown = calculate_transition_score_breakdown(
        prev_state,
        next_state,
        t,
        prior=prior,
        context=effective_context,
        window=window,
        weights=weights,
        meters=meters,
        vocabularies=vocabularies,
        edo=edo,
    )
    return breakdown.final_log_weight


def calculate_transition_log_weights(
    queries: Sequence[PriorQuery],
    *,
    prior: Prior,
    windows: Optional[Sequence[Optional[TransitionWindow]]] = None,
    weights: Optional[PriorWeights] = None,
    meters=None,
    vocabularies=None,
    edo: Optional[int] = None,
    target_tensions: Optional[Sequence[Optional[float]]] = None,
    sections: Optional[Sequence[Optional[object]]] = None,
) -> Tuple[float, ...]:
    """Batch version of transition log-weight scoring with target tension and section style support."""
    effective_queries = list(queries)
    if target_tensions is not None or sections is not None:
        from aimusic.planning.sections import EdgeSectionView, PlanningSection

        for idx, q in enumerate(queries):
            cur_view = q.context.section_view if q.context is not None else None
            if cur_view is None:
                sec_obj = (
                    sections[idx] if sections is not None and idx < len(sections) and isinstance(sections[idx], PlanningSection)
                    else (q.context.section if q.context is not None and isinstance(q.context.section, PlanningSection) else None)
                )
                if sec_obj is not None:
                    tau = (
                        target_tensions[idx] if target_tensions is not None and idx < len(target_tensions) and target_tensions[idx] is not None
                        else (q.context.target_tension if q.context is not None and q.context.target_tension is not None else 0.5)
                    )
                    new_view = EdgeSectionView(
                        section=sec_obj,
                        section_index=0,
                        offset_in_section=max(0, q.time_index + 1 - sec_obj.start_time),
                        is_section_start=(q.time_index + 1 == sec_obj.start_time),
                        target_tension=tau,
                        arrival_time=q.time_index + 1,
                    )
                    hist = q.context.history if q.context is not None else (q.prev_state,)
                    new_ctx = PriorContext(history=hist, section_view=new_view)
                    effective_queries[idx] = PriorQuery(q.prev_state, q.next_state, q.time_index, new_ctx)

    breakdowns = calculate_transition_score_breakdowns(
        effective_queries,
        prior=prior,
        windows=windows,
        weights=weights,
        meters=meters,
        vocabularies=vocabularies,
        edo=edo,
    )
    return tuple(b.final_log_weight for b in breakdowns)


calculate_log_weight = calculate_transition_log_weight

