from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, FrozenSet, Optional, Tuple

from aimusic.planning.candidates import (
    CandidateGenerationResult,
    CandidateRejection,
    get_valid_next_states,
    is_legal_transition,
)
from aimusic.core.config import PriorWeights, SBConfig, StyleConfig
from aimusic.core.core_types import BeatState, Edge, Layer
from aimusic.scoring.priors import (
    NullPrior,
    Prior,
    PriorContext,
    PriorQuery,
    calculate_transition_log_weights,
)
from aimusic.theory.tonal import basic_space_distance, tonal_distance
from aimusic.core.vocab import DEFAULT_VOCABULARIES, Vocabularies
import numpy as np

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


def _edge_sort_key(edge: Edge) -> tuple[float, tuple[int, int, int, int, int, int, int, int]]:
    return (-edge.log_weight, _state_sort_key(edge.target))


def _resolved_vocabs(vocabularies: Optional[Vocabularies]) -> Vocabularies:
    return DEFAULT_VOCABULARIES if vocabularies is None else vocabularies


def _resolved_prior(prior: Optional[Prior]) -> Prior:
    return NullPrior() if prior is None else prior


def _edo_size(vocabularies: Vocabularies) -> int:
    return len(vocabularies.keys)


# ---------------------------------------------------------------------------
# EPIC 10 — Ticket 4: Cached neighbor sets and memoized distances

@lru_cache(maxsize=8192)
def _cached_endpoint_distance(
    src_chord_root: int,
    src_chord_quality: str,
    src_key_root: int,
    src_boundary_lvl: int,
    src_beat_in_bar: int,
    src_meter_id: int,
    src_role_id: int,
    src_groove_id: int,
    tgt_chord_root: int,
    tgt_chord_quality: str,
    tgt_key_root: int,
    tgt_boundary_lvl: int,
    tgt_beat_in_bar: int,
    tgt_meter_id: int,
    tgt_role_id: int,
    tgt_groove_id: int,
    edo: int,
) -> float:
    """Pure-value wrapper around the endpoint distance calculation."""
    
    harmonic = basic_space_distance(
        src_chord_root, src_chord_quality,
        tgt_chord_root, tgt_chord_quality,
        edo,
    )
    tonal = tonal_distance(src_key_root, tgt_key_root, edo)
    structural = (
        abs(src_boundary_lvl - tgt_boundary_lvl)
        + abs(src_beat_in_bar - tgt_beat_in_bar) * 0.25
        + (0.5 if src_meter_id != tgt_meter_id else 0.0)
        + (0.25 if src_role_id != tgt_role_id else 0.0)
        + (0.15 if src_groove_id != tgt_groove_id else 0.0)
    )
    return float(harmonic + tonal + structural)


def _fast_endpoint_distance(
    state: BeatState,
    end_layer: Layer,
    vocabularies: Vocabularies,
    edo: int,
) -> float:
    """Cached minimum distance from *state* to any state in *end_layer*."""
    
    if len(end_layer) == 0:
        return 0.0

    src_chord = vocabularies.chords.token_for_id(state.chord_id)
    src_key = vocabularies.keys.token_for_id(state.key_id)

    min_dist = float("inf")
    for target in end_layer.states:
        tgt_chord = vocabularies.chords.token_for_id(target.chord_id)
        tgt_key = vocabularies.keys.token_for_id(target.key_id)
        dist = _cached_endpoint_distance(
            src_chord.root_pc, src_chord.quality,
            src_key.root_pc,
            state.boundary_lvl, state.beat_in_bar, state.meter_id,
            state.role_id, state.groove_id,
            tgt_chord.root_pc, tgt_chord.quality,
            tgt_key.root_pc,
            target.boundary_lvl, target.beat_in_bar, target.meter_id,
            target.role_id, target.groove_id,
            edo,
        )
        if dist < min_dist:
            min_dist = dist
    return min_dist


class NeighborCache:
    """Per-graph-build cache of legal BeatState successors."""

    def __init__(self) -> None:
        self._store: Dict[BeatState, FrozenSet[BeatState]] = {}
        self.hits: int = 0
        self.misses: int = 0

    def get(
        self,
        source: BeatState,
        t: int,
        *,
        style_config: StyleConfig,
        vocabularies: Vocabularies,
        prior: "Prior",
        context: Optional["PriorContext"],
        rng: "np.random.Generator",
        d_max: int,
    ) -> "CandidateGenerationResult":
        """Return valid successors of *source*, using the cache when possible."""
        if source in self._store:
            self.hits += 1
            cached_states = tuple(
                sorted(self._store[source], key=_state_sort_key)
            )
            return CandidateGenerationResult(
                time_index=t,
                source_state=source,
                states=cached_states,
                rejections=(),
            )

        self.misses += 1
        result = get_valid_next_states(
            source,
            t,
            rng=rng,
            d_max=d_max,
            style_config=style_config,
            vocabularies=vocabularies,
            prior=prior,
            context=context,
        )
        self._store[source] = frozenset(result.states)
        return result

    @property
    def size(self) -> int:
        """Number of distinct source states currently cached."""
        return len(self._store)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a fraction in [0, 1].  Returns 0 when empty."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0

    def clear(self) -> None:
        """Evict all entries (e.g. between sections with different configs)."""
        self._store.clear()
        self.hits = 0
        self.misses = 0


def _estimate_endpoint_distance(
    state: BeatState,
    end_layer: Layer,
    vocabularies: Vocabularies,
    edo: Optional[int] = None,
) -> float:
    """Cheap distance-to-go heuristic used for K_max pruning."""
    if len(end_layer) == 0:
        return 0.0

    resolved_edo = _edo_size(vocabularies) if edo is None else edo
    source_key = vocabularies.keys.token_for_id(state.key_id)
    source_chord = vocabularies.chords.token_for_id(state.chord_id)

    distances = []
    for target in end_layer:
        target_key = vocabularies.keys.token_for_id(target.key_id)
        target_chord = vocabularies.chords.token_for_id(target.chord_id)
        harmonic = basic_space_distance(
            source_chord.root_pc,
            source_chord.quality,
            target_chord.root_pc,
            target_chord.quality,
            resolved_edo,
        )
        tonal = tonal_distance(source_key.root_pc, target_key.root_pc, resolved_edo)
        structural = (
            abs(state.boundary_lvl - target.boundary_lvl)
            + abs(state.beat_in_bar - target.beat_in_bar) * 0.25
            + (0.5 if state.meter_id != target.meter_id else 0.0)
            + (0.25 if state.role_id != target.role_id else 0.0)
            + (0.15 if state.groove_id != target.groove_id else 0.0)
        )
        distances.append(float(harmonic + tonal + structural))
    return min(distances)


def _pruning_score(
    state: BeatState,
    best_incoming_log_mass: float,
    steps_remaining: int,
    end_layer: Layer,
    vocabularies: Vocabularies,
    style_config: StyleConfig,
    edo: Optional[int] = None,
) -> float:
    if steps_remaining == 1 and not any(
        is_legal_transition(
            state,
            endpoint,
            style_config=style_config,
            vocabularies=vocabularies,
        )[0]
        for endpoint in end_layer.states
    ):
        return float("-inf")
    endpoint_distance = _fast_endpoint_distance(
        state,
        end_layer,
        vocabularies,
        edo if edo is not None else _edo_size(vocabularies),
    )
    horizon_scale = max(1, steps_remaining)
    return float(best_incoming_log_mass - (endpoint_distance / horizon_scale))


def _edge_priority_score(
    edge: Edge,
    steps_remaining: int,
    end_layer: Layer,
    vocabularies: Vocabularies,
    style_config: StyleConfig,
    edo: Optional[int] = None,
) -> float:
    if steps_remaining == 1 and not any(
        is_legal_transition(
            edge.target,
            endpoint,
            style_config=style_config,
            vocabularies=vocabularies,
        )[0]
        for endpoint in end_layer.states
    ):
        return float("-inf")
    return float(
        edge.log_weight
        - (
            _fast_endpoint_distance(
                edge.target,
                end_layer,
                vocabularies,
                edo if edo is not None else _edo_size(vocabularies),
            )
            / max(1, steps_remaining)
        )
    )


@dataclass(frozen=True)
class PrunedState:
    """Record of why a state was removed during graph construction."""

    time_index: int
    state: BeatState
    reason: str
    heuristic_score: float

    def __post_init__(self) -> None:
        if not isinstance(self.state, BeatState):
            raise TypeError("state must be a BeatState.")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string.")


@dataclass(frozen=True)
class LayerBuildDiagnostics:
    """Per-layer diagnostics for sparse graph expansion."""

    time_index: int
    source_state_count: int
    raw_candidate_count: int
    unique_candidate_count: int
    kept_candidate_count: int
    raw_edge_count: int
    kept_edge_count: int
    outdegree_pruned_count: int = 0
    rejected_proposals: Tuple[CandidateRejection, ...] = ()
    pruned_states: Tuple[PrunedState, ...] = ()

    @property
    def pruned_candidate_count(self) -> int:
        return len(self.pruned_states)


@dataclass(frozen=True)
class GraphDiagnostics:
    """Aggregate diagnostics for the full sparse graph."""

    layer_sizes: Tuple[int, ...]
    layer_diagnostics: Tuple[LayerBuildDiagnostics, ...]
    neighbor_cache_hits: int = 0
    neighbor_cache_misses: int = 0

    @property
    def total_rejections(self) -> int:
        return sum(len(item.rejected_proposals) for item in self.layer_diagnostics)

    @property
    def neighbor_cache_hit_rate(self) -> float:
        """Fraction of candidate lookups served from the neighbor cache."""
        total = self.neighbor_cache_hits + self.neighbor_cache_misses
        return self.neighbor_cache_hits / total if total > 0 else 0.0


@dataclass(frozen=True)
class SparseGraph:
    """Sparse layered graph plus diagnostics for later SB inference."""

    layers: Tuple[Layer, ...]
    edges_by_time: Tuple[Tuple[Edge, ...], ...]
    diagnostics: GraphDiagnostics


def _build_prior_context(
    source_state: BeatState,
    end_layer: Layer,
    time_index: int,
) -> PriorContext:
    future_hints = end_layer.states[: min(3, len(end_layer.states))]
    return PriorContext(
        history=(source_state,),
        future_hints=future_hints,
        metadata=(("graph_time", str(time_index)),),
    )


def _candidate_result_for_target_layer(
    prev_state: BeatState,
    target_layer: Layer,
    time_index: int,
    *,
    style_config: StyleConfig,
    vocabularies: Vocabularies,
) -> CandidateGenerationResult:
    accepted = []
    rejections = []
    for candidate in target_layer.states:
        legal, reason = is_legal_transition(
            prev_state,
            candidate,
            style_config=style_config,
            vocabularies=vocabularies,
        )
        if legal:
            accepted.append(candidate)
        else:
            rejections.append(
                CandidateRejection(
                    time_index=time_index,
                    source_state=prev_state,
                    candidate_state=candidate,
                    reason=reason or "illegal_endpoint_transition",
                )
            )
    return CandidateGenerationResult(
        time_index=time_index,
        source_state=prev_state,
        states=tuple(accepted),
        rejections=tuple(rejections),
    )


@dataclass(frozen=True)
class StitchAnchor:
    """Soft continuity constraint between adjacent sections."""

    prev_terminal: BeatState
    log_bonus: float = 1.5
    match_key: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.prev_terminal, BeatState):
            raise TypeError("prev_terminal must be a BeatState.")
        from math import isfinite
        if not isinstance(self.log_bonus, (int, float)) or not isfinite(float(self.log_bonus)):
            raise ValueError("log_bonus must be a finite real number.")
        if not isinstance(self.match_key, bool):
            raise TypeError("match_key must be a bool.")

    def bonus_for(self, source: BeatState) -> float:
        """Return the log-space bonus for *source* at the first step."""
        if source == self.prev_terminal:
            return float(self.log_bonus)
        if self.match_key and source.key_id == self.prev_terminal.key_id:
            return float(self.log_bonus) * 0.5   # half credit for key match only
        return 0.0


def build_sparse_graph(
    start_layer: Layer,
    end_layer: Layer,
    total_beats: int,
    *,
    sb_config: Optional[SBConfig] = None,
    style_config: Optional[StyleConfig] = None,
    vocabularies: Optional[Vocabularies] = None,
    prior: Optional[Prior] = None,
    weights: Optional[PriorWeights] = None,
    edo: Optional[int] = None,
    rng: np.random.Generator,
    d_max: int,
    stitch_anchor: Optional[StitchAnchor] = None,
) -> SparseGraph:
    """Build a bounded sparse graph of BeatState transitions."""
    if not isinstance(start_layer, Layer):
        raise TypeError("start_layer must be a Layer.")
    if not isinstance(end_layer, Layer):
        raise TypeError("end_layer must be a Layer.")
    if not isinstance(total_beats, int) or total_beats < 1:
        raise ValueError("total_beats must be >= 1.")
    if stitch_anchor is not None and not isinstance(stitch_anchor, StitchAnchor):
        raise TypeError("stitch_anchor must be a StitchAnchor or None.")

    resolved_sb = SBConfig() if sb_config is None else sb_config
    resolved_style = StyleConfig() if style_config is None else style_config
    resolved_vocabs = _resolved_vocabs(vocabularies)
    resolved_prior = _resolved_prior(prior)
    resolved_edo = _edo_size(resolved_vocabs) if edo is None else edo

    expected_end_time = start_layer.time_index + total_beats
    if end_layer.time_index != expected_end_time:
        raise ValueError("end_layer.time_index must equal start_layer.time_index + total_beats.")
    if len(start_layer) > resolved_sb.k_max:
        raise ValueError("start_layer size must be <= sb_config.k_max.")
    if len(end_layer) > resolved_sb.k_max:
        raise ValueError("end_layer size must be <= sb_config.k_max.")

    layers = [start_layer]
    edge_layers: list[Tuple[Edge, ...]] = []
    diagnostics: list[LayerBuildDiagnostics] = []
    neighbor_cache = NeighborCache()

    for step in range(total_beats):
        current_layer = layers[-1]
        current_time = current_layer.time_index
        next_time = current_time + 1
        final_step = next_time == end_layer.time_index
        steps_remaining = end_layer.time_index - next_time

        raw_candidate_count = 0
        raw_edge_count = 0
        outdegree_pruned_count = 0
        rejected: list[CandidateRejection] = []
        kept_edges: list[Edge] = []
        best_incoming: dict[BeatState, float] = {}

        for source_state in current_layer.states:
            if final_step:
                candidate_result = _candidate_result_for_target_layer(
                    source_state,
                    end_layer,
                    current_time,
                    style_config=resolved_style,
                    vocabularies=resolved_vocabs,
                )
            else:
                candidate_result = neighbor_cache.get(
                    source_state,
                    current_time,
                    style_config=resolved_style,
                    vocabularies=resolved_vocabs,
                    prior=resolved_prior,
                    context=_build_prior_context(source_state, end_layer, current_time),
                    rng=rng,
                    d_max=d_max,
                )

            raw_candidate_count += candidate_result.proposed_count
            rejected.extend(candidate_result.rejections)
            
            source_context = _build_prior_context(source_state, end_layer, current_time)
            is_first_step = (current_time == start_layer.time_index)
            stitch_bonus = (
                stitch_anchor.bonus_for(source_state)
                if stitch_anchor is not None and is_first_step
                else 0.0
            )
            queries = tuple(
                PriorQuery(
                    prev_state=source_state,
                    next_state=candidate_state,
                    time_index=current_time,
                    context=source_context,
                )
                for candidate_state in candidate_result.states
            )
            raw_edge_count += len(queries)
            log_weights = calculate_transition_log_weights(
                queries,
                prior=resolved_prior,
                weights=weights,
                vocabularies=resolved_vocabs,
                edo=resolved_edo,
            )
            source_edges = [
                Edge(
                    time_index=current_time,
                    source=source_state,
                    target=query.next_state,
                    log_weight=log_weight + stitch_bonus,
                )
                for query, log_weight in zip(queries, log_weights)
            ]
            source_edges.sort(
                key=lambda edge: (
                    -_edge_priority_score(
                        edge,
                        steps_remaining,
                        end_layer,
                        resolved_vocabs,
                        resolved_style,
                        edo=resolved_edo,
                    ),
                    _state_sort_key(edge.target),
                )
            )
            if len(source_edges) > resolved_sb.d_max:
                outdegree_pruned_count += len(source_edges) - resolved_sb.d_max
            trimmed_edges = source_edges[: resolved_sb.d_max]
            kept_edges.extend(trimmed_edges)
            for edge in trimmed_edges:
                if edge.target not in best_incoming or edge.log_weight > best_incoming[edge.target]:
                    best_incoming[edge.target] = edge.log_weight

        unique_candidates = tuple(sorted(best_incoming.keys(), key=_state_sort_key))
        pruned_states: list[PrunedState] = []

        if final_step:
            kept_states = unique_candidates
            unreachable_endpoints = [
                state for state in end_layer.states if state not in set(unique_candidates)
            ]
            for state in unreachable_endpoints:
                pruned_states.append(
                    PrunedState(
                        time_index=next_time,
                        state=state,
                        reason="unreachable_endpoint",
                        heuristic_score=float("-inf"),
                    )
                )
        else:
            if len(unique_candidates) > resolved_sb.k_max:
                ranked_candidates = sorted(
                    unique_candidates,
                    key=lambda state: (
                        -_pruning_score(
                            state,
                            best_incoming[state],
                            steps_remaining,
                            end_layer,
                            resolved_vocabs,
                            resolved_style,
                            edo=resolved_edo,
                        ),
                        _state_sort_key(state),
                    ),
                )
                kept_states = tuple(ranked_candidates[: resolved_sb.k_max])
                kept_state_set = set(kept_states)
                for state in ranked_candidates[resolved_sb.k_max :]:
                    pruned_states.append(
                        PrunedState(
                            time_index=next_time,
                            state=state,
                            reason="k_max_prune",
                            heuristic_score=_pruning_score(
                                state,
                                best_incoming[state],
                                steps_remaining,
                                end_layer,
                                resolved_vocabs,
                                resolved_style,
                                edo=resolved_edo,
                            ),
                        )
                    )
                kept_edges = [edge for edge in kept_edges if edge.target in kept_state_set]
            else:
                kept_states = unique_candidates

        next_layer = Layer(time_index=next_time, states=tuple(sorted(kept_states, key=_state_sort_key)))
        layers.append(next_layer)
        edge_layers.append(tuple(sorted(kept_edges, key=_edge_sort_key)))
        diagnostics.append(
            LayerBuildDiagnostics(
                time_index=next_time,
                source_state_count=len(current_layer),
                raw_candidate_count=raw_candidate_count,
                unique_candidate_count=len(unique_candidates),
                kept_candidate_count=len(next_layer),
                raw_edge_count=raw_edge_count,
                kept_edge_count=len(edge_layers[-1]),
                outdegree_pruned_count=outdegree_pruned_count,
                rejected_proposals=tuple(rejected),
                pruned_states=tuple(pruned_states),
            )
        )

    return SparseGraph(
        layers=tuple(layers),
        edges_by_time=tuple(edge_layers),
        diagnostics=GraphDiagnostics(
            layer_sizes=tuple(len(layer) for layer in layers),
            layer_diagnostics=tuple(diagnostics),
            neighbor_cache_hits=neighbor_cache.hits,
            neighbor_cache_misses=neighbor_cache.misses,
        ),
    )
