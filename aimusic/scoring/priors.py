from __future__ import annotations

import json
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Optional, Protocol, Sequence, Tuple, cast, runtime_checkable

from aimusic.core.config import (
    NeuralPriorConfig,
    PlaceholderPriorMode,
    PriorFactorization,
    PriorWeights,
)
from aimusic.core.core_types import BeatState
from aimusic.scoring.gttm_features import TransitionWindow, calculate_gttm_energy


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
        if stream not in STRUCTURAL_STREAM_NAMES:
            raise ValueError(f"Unsupported token stream: {stream!r}")
    if len(items) != len(set(items)):
        raise ValueError("token_streams must not contain duplicates.")
    return items


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


@dataclass(frozen=True)
class PriorContext:
    """Optional transition-scoring context for external priors."""

    history: Tuple[BeatState, ...] = ()
    future_hints: Tuple[BeatState, ...] = ()
    section_name: Optional[str] = None
    metadata: MetadataPairs = ()
    history_tokens: Optional[StructuralTokenSequence] = None
    future_hint_tokens: Optional[StructuralTokenSequence] = None

    def __post_init__(self) -> None:
        history = _coerce_state_tuple("history", self.history)
        future_hints = _coerce_state_tuple("future_hints", self.future_hints)
        object.__setattr__(self, "history", history)
        object.__setattr__(self, "future_hints", future_hints)

        if self.section_name is not None:
            _require_non_empty_str("section_name", self.section_name)

        metadata = _coerce_metadata("metadata", self.metadata)
        object.__setattr__(self, "metadata", metadata)

        history_tokens = (
            StructuralTokenSequence.from_states(history)
            if self.history_tokens is None
            else self.history_tokens
        )
        future_hint_tokens = (
            StructuralTokenSequence.from_states(future_hints)
            if self.future_hint_tokens is None
            else self.future_hint_tokens
        )

        if not isinstance(history_tokens, StructuralTokenSequence):
            raise TypeError("history_tokens must be a StructuralTokenSequence.")
        if not isinstance(future_hint_tokens, StructuralTokenSequence):
            raise TypeError("future_hint_tokens must be a StructuralTokenSequence.")
        if len(history_tokens) != len(history):
            raise ValueError("history_tokens length must match history length.")
        if len(future_hint_tokens) != len(future_hints):
            raise ValueError("future_hint_tokens length must match future_hints length.")

        object.__setattr__(self, "history_tokens", history_tokens)
        object.__setattr__(self, "future_hint_tokens", future_hint_tokens)

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
        return cls(
            prev_event=StructuralEventTokens.from_state(query.prev_state),
            next_event=StructuralEventTokens.from_state(query.next_state),
            time_index=query.time_index,
            history_tokens=context.history_tokens,
            future_hint_tokens=context.future_hint_tokens,
            section_name=context.section_name,
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

    manifest_version: int = 1
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
    """Algorithmic-only prior that contributes a constant neutral log-probability."""

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
    """Wrapper around an external neural prior or a deterministic placeholder."""

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
) -> float:
    """Combine prior data likelihood and GTTM energy into one graph-ready edge weight."""
    resolved_weights = PriorWeights() if weights is None else weights
    data_logp = float(prior.logp_next(prev_state, next_state, t, context))
    gttm_energy = calculate_gttm_energy(
        prev_state,
        next_state,
        t,
        window=window,
        meters=meters,
        vocabularies=vocabularies,
        edo=edo,
        weights=resolved_weights,
    )
    return (resolved_weights.lambda_data * data_logp) - (resolved_weights.lambda_gttm * gttm_energy)


def calculate_transition_log_weights(
    queries: Sequence[PriorQuery],
    *,
    prior: Prior,
    windows: Optional[Sequence[Optional[TransitionWindow]]] = None,
    weights: Optional[PriorWeights] = None,
    meters=None,
    vocabularies=None,
    edo: Optional[int] = None,
) -> Tuple[float, ...]:
    """Batch version of transition log-weight scoring."""
    query_items = tuple(queries)
    resolved_weights = PriorWeights() if weights is None else weights
    data_scores = prior_logps(prior, query_items)

    if windows is None:
        window_items: tuple[TransitionWindow | None, ...] = (None,) * len(query_items)
    else:
        window_items = tuple(windows)
        if len(window_items) != len(query_items):
            raise ValueError("windows must align 1:1 with queries.")

    results = []
    for idx, query in enumerate(query_items):
        gttm_energy = calculate_gttm_energy(
            query.prev_state,
            query.next_state,
            query.time_index,
            window=window_items[idx],
            meters=meters,
            vocabularies=vocabularies,
            edo=edo,
            weights=resolved_weights,
        )
        results.append(
            (resolved_weights.lambda_data * data_scores[idx])
            - (resolved_weights.lambda_gttm * gttm_energy)
        )
    return tuple(float(value) for value in results)


calculate_log_weight = calculate_transition_log_weight

# ---------------------------------------------------------------------------
# EPIC 10 - 5 : NeuralPrior integration
@dataclass(frozen=True)
class NeuralPriorScoringStats:
    """Observability record for one NeuralPrior scoring session.

    Attach this to a run result to understand whether the model was actually
    called, how many batches were dispatched, and whether any fallbacks fired.

    Attributes
    ----------
    total_queries:
        Total number of (prev, next) transition pairs scored.
    model_calls:
        Number of queries routed to the real model (0 if placeholder only).
    placeholder_calls:
        Number of queries handled by the placeholder scorer.
    batch_count:
        Number of batches sent to ``score_transition_batch`` (0 for single
        scoring or placeholder-only runs).
    fallback_count:
        Number of queries that fell back from model to placeholder after a
        scoring exception.
    factorization_mode:
        The ``PriorFactorization`` used for this session.
    """

    total_queries: int = 0
    model_calls: int = 0
    placeholder_calls: int = 0
    batch_count: int = 0
    fallback_count: int = 0
    factorization_mode: PriorFactorization = PriorFactorization.FACTORIZED

    def __post_init__(self) -> None:
        for name in ("total_queries", "model_calls", "placeholder_calls",
                     "batch_count", "fallback_count"):
            _require_int(name, getattr(self, name), minimum=0)
        if not isinstance(self.factorization_mode, PriorFactorization):
            raise TypeError("factorization_mode must be a PriorFactorization.")

    @property
    def model_fraction(self) -> float:
        """Fraction of queries handled by the real model (0–1)."""
        return self.model_calls / self.total_queries if self.total_queries > 0 else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "total_queries": self.total_queries,
            "model_calls": self.model_calls,
            "placeholder_calls": self.placeholder_calls,
            "batch_count": self.batch_count,
            "fallback_count": self.fallback_count,
            "factorization_mode": self.factorization_mode.value,
            "model_fraction": self.model_fraction,
        }


@dataclass(frozen=True)
class FactorizedStreamScorer:
    """Score a transition by summing per-stream log-probs from a factorized model."""

    model: NeuralPriorModel
    active_streams: Tuple[str, ...]
    default_logp: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.model, NeuralPriorModel):
            raise TypeError("model must satisfy the NeuralPriorModel protocol.")
        if not self.active_streams:
            raise ValueError("active_streams must not be empty.")
        _require_real("default_logp", self.default_logp)

    def score(
        self,
        query: TokenizedPriorQuery,
        *,
        fallback_scorer: "NeuralPrior",
    ) -> tuple[float, int]:
        """Return ``(total_log_prob, fallback_count)`` for *query*."""
        total = 0.0
        fallbacks = 0
        for stream in self.active_streams:
            stream_query = TokenizedPriorQuery(
                prev_event=query.prev_event,
                next_event=query.next_event,
                time_index=query.time_index,
                history_tokens=query.history_tokens,
                future_hint_tokens=query.future_hint_tokens,
                section_name=query.section_name,
                metadata=(("stream", stream),) + query.metadata,
                factorization_mode=PriorFactorization.FACTORIZED,
            )
            try:
                total += float(self.model.score_transition(stream_query))
            except Exception:
                total += fallback_scorer._score_placeholder(stream_query)
                fallbacks += 1
        return total, fallbacks


def _score_whole_state(
    model: NeuralPriorModel,
    query: TokenizedPriorQuery,
    *,
    default_logp: float,
    fallback_scorer: "NeuralPrior",
) -> tuple[float, int]:
    """Score a query in WHOLE_STATE mode with exception fallback."""
    try:
        return float(model.score_transition(query)), 0
    except Exception:
        return fallback_scorer._score_placeholder(query), 1


def _score_mixed(
    model: NeuralPriorModel,
    query: TokenizedPriorQuery,
    *,
    active_streams: Tuple[str, ...],
    default_logp: float,
    fallback_scorer: "NeuralPrior",
) -> tuple[float, int]:
    """Score a query in MIXED mode: whole-state score plus per-stream deltas."""
    base, fb_base = _score_whole_state(
        model, query, default_logp=default_logp, fallback_scorer=fallback_scorer
    )
    delta_query = TokenizedPriorQuery(
        prev_event=query.prev_event,
        next_event=query.next_event,
        time_index=query.time_index,
        history_tokens=query.history_tokens,
        future_hint_tokens=query.future_hint_tokens,
        section_name=query.section_name,
        metadata=query.metadata,
        factorization_mode=PriorFactorization.FACTORIZED,
    )
    stream_scorer = FactorizedStreamScorer(
        model=model,
        active_streams=active_streams,
        default_logp=default_logp,
    )
    delta, fb_delta = stream_scorer.score(delta_query, fallback_scorer=fallback_scorer)
    # Weight the factorized delta at 0.25 so it refines rather than dominates.
    return base + 0.25 * delta, fb_base + fb_delta


def _chunk(items: Sequence, size: int) -> list[list]:
    if size <= 0:
        raise ValueError("batch_size must be positive")
    return [list(items[i:i + size]) for i in range(0, len(items), size)]


class NeuralPriorSession:
    """Stateful scoring session for a single graph-build or section pass."""

    def __init__(self, prior: "NeuralPrior") -> None:
        if not isinstance(prior, NeuralPrior):
            raise TypeError("prior must be a NeuralPrior.")
        self._prior = prior
        self._total = 0
        self._model_calls = 0
        self._placeholder_calls = 0
        self._batch_count = 0
        self._fallbacks = 0
        self._finalized = False

    def score(
        self,
        prev_state: BeatState,
        next_state: BeatState,
        t: int,
        context: Optional[PriorContext] = None,
    ) -> float:
        """Score one transition, routing by factorization mode."""
        if self._finalized:
            raise RuntimeError("Session already finalized.")
        self._total += 1
        query = PriorQuery(
            prev_state=prev_state, next_state=next_state,
            time_index=t, context=context,
        )
        tokenized = query.tokenize(self._prior.manifest.factorization_mode)
        logp, fallbacks = self._dispatch_single(tokenized)
        self._fallbacks += fallbacks
        return logp

    def score_batch(self, queries: Sequence[PriorQuery]) -> Tuple[float, ...]:
        """Score a batch of queries, chunking to ``config.batch_size``."""
        if self._finalized:
            raise RuntimeError("Session already finalized.")
        query_items = tuple(queries)
        self._total += len(query_items)
        tokenized = tuple(
            q.tokenize(self._prior.manifest.factorization_mode) for q in query_items
        )
        return self._dispatch_batch(tokenized)

    def _dispatch_single(
        self, query: TokenizedPriorQuery
    ) -> tuple[float, int]:
        """Route one tokenized query to the correct scoring path."""
        prior = self._prior
        mode = prior.manifest.factorization_mode

        if prior.model is None:
            self._placeholder_calls += 1
            return prior._score_placeholder(query), 0

        self._model_calls += 1
        if mode is PriorFactorization.FACTORIZED:
            scorer = FactorizedStreamScorer(
                model=prior.model,
                active_streams=prior.manifest.token_streams,
                default_logp=prior.config.default_logp,
            )
            logp, fb = scorer.score(query, fallback_scorer=prior)
            return logp, fb
        
        
        elif mode is PriorFactorization.WHOLE_STATE:
            return _score_whole_state(
                prior.model, query,
                default_logp=prior.config.default_logp,
                fallback_scorer=prior,
            )
        else:  # MIXED
            return _score_mixed(
                prior.model, query,
                active_streams=prior.manifest.token_streams,
                default_logp=prior.config.default_logp,
                fallback_scorer=prior,
            )

    def _dispatch_batch(
        self, queries: Tuple[TokenizedPriorQuery, ...]
    ) -> Tuple[float, ...]:
        """Dispatch tokenized queries in chunks of ``batch_size``."""
        prior = self._prior
        mode = prior.manifest.factorization_mode
        batch_size = prior.config.batch_size

        # Placeholder-only path — no batching needed
        if prior.model is None:
            self._placeholder_calls += len(queries)
            return tuple(prior._score_placeholder(q) for q in queries)

        # Non-batch model or FACTORIZED/MIXED (scored stream-by-stream)
        use_native_batch = (
            prior.config.supports_batch_scoring
            and prior.manifest.supports_batch_scoring
            and isinstance(prior.model, BatchedNeuralPriorModel)
            and mode is PriorFactorization.WHOLE_STATE
        )

        results: list[float] = []
        for chunk in _chunk(queries, batch_size):
            if use_native_batch:
                self._batch_count += 1
                self._model_calls += len(chunk)
                try:
                    scores = prior.model.score_transition_batch(tuple(chunk))  # type: ignore[attr-defined]
                    if len(scores) != len(chunk):
                        raise ValueError("score_transition_batch length mismatch.")
                    results.extend(float(s) for s in scores)
                except Exception:
                    # Full chunk fallback
                    self._fallbacks += len(chunk)
                    self._model_calls -= len(chunk)
                    self._placeholder_calls += len(chunk)
                    results.extend(prior._score_placeholder(q) for q in chunk)
            else:
                # Per-query dispatch for FACTORIZED / MIXED or non-batched models
                for q in chunk:
                    logp, fb = self._dispatch_single(q)
                    self._fallbacks += fb
                    results.append(logp)
        return tuple(results)

    def finalize(self) -> NeuralPriorScoringStats:
        """Return accumulated stats and mark the session as done."""
        if self._finalized:
            raise RuntimeError("Session already finalized.")
        self._finalized = True
        return NeuralPriorScoringStats(
            total_queries=self._total,
            model_calls=self._model_calls,
            placeholder_calls=self._placeholder_calls,
            batch_count=self._batch_count,
            fallback_count=self._fallbacks,
            factorization_mode=self._prior.manifest.factorization_mode,
        )


def load_neural_prior(
    config: NeuralPriorConfig,
    *,
    model: Optional[NeuralPriorModel] = None,
    manifest_override: Optional[NeuralPriorManifest] = None,
) -> NeuralPrior:
    """Factory: build a ready-to-use ``NeuralPrior`` from config + optional model."""
    if not isinstance(config, NeuralPriorConfig):
        raise TypeError("config must be a NeuralPriorConfig.")
    if model is not None and not isinstance(model, NeuralPriorModel):
        raise TypeError("model must satisfy the NeuralPriorModel protocol or be None.")
    if manifest_override is not None and not isinstance(manifest_override, NeuralPriorManifest):
        raise TypeError("manifest_override must be a NeuralPriorManifest or None.")

    if manifest_override is not None:
        manifest = manifest_override
    elif config.manifest_path is not None:
        manifest = load_neural_prior_manifest(config.manifest_path)
        # Validate alignment
        if manifest.factorization_mode is not config.factorization_mode:
            raise ValueError(
                f"Loaded manifest factorization_mode={manifest.factorization_mode.value!r} "
                f"does not match config factorization_mode={config.factorization_mode.value!r}."
            )
    else:
        manifest = build_neural_prior_manifest(config)

    return NeuralPrior(config=config, manifest=manifest, model=model)


def open_neural_prior_session(prior: NeuralPrior) -> NeuralPriorSession:
    """Convenience factory for a new scoring session on *prior*."""
    return NeuralPriorSession(prior)