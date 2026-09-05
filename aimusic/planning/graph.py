from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

_logger = logging.getLogger(__name__)

from aimusic.planning.candidates import (
    CandidateGenerationResult,
    CandidateRejection,
    get_valid_next_states,
    is_legal_transition,
)
from aimusic.core.config import PriorWeights, SBConfig, StyleConfig
from aimusic.core.core_types import BeatState, Edge, Layer
from aimusic.core.rng import RNGKey
from aimusic.scoring.priors import (
    NullPrior,
    Prior,
    PriorContext,
    PriorQuery,
    TransitionScoreBreakdown,
    calculate_transition_log_weights,
    calculate_transition_score_breakdowns,
)
from aimusic.scoring.gttm_features import FEATURE_REGISTRY, TransitionWindow
from aimusic.theory.tonal import basic_space_distance, tonal_distance
from aimusic.core.vocab import (
    DEFAULT_VOCABULARIES,
    Vocabularies,
    validate_vocabulary_compatibility,
)

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
    endpoint_distance = _estimate_endpoint_distance(
        state,
        end_layer,
        vocabularies,
        edo=edo,
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
            _estimate_endpoint_distance(
                edge.target,
                end_layer,
                vocabularies,
                edo=edo,
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
    """Per-layer diagnostics for sparse graph expansion.

    REQ-13: proposed / legal / unique / scored / retained are tracked as
    separate counters end-to-end. ``raw_candidate_count`` is the sum of each
    source state's proposal-budget-bounded raw count (proposed);
    ``legal_candidate_count`` sums post-legality-check candidates per source
    (legal); ``unique_candidate_count`` is the layer-wide merged support
    (unique); ``scored_candidate_count`` sums prior-guided-ranking scores
    when enabled (scored, 0 otherwise); ``kept_edge_count`` is what D_max
    actually retains (retained) -- the only count D_max controls.
    """

    time_index: int
    source_state_count: int
    raw_candidate_count: int
    unique_candidate_count: int
    kept_candidate_count: int
    raw_edge_count: int
    kept_edge_count: int
    outdegree_pruned_count: int = 0
    legal_candidate_count: int = 0
    scored_candidate_count: int = 0
    rejected_proposals: Tuple[CandidateRejection, ...] = ()
    pruned_states: Tuple[PrunedState, ...] = ()

    @property
    def pruned_candidate_count(self) -> int:
        return len(self.pruned_states)

    @property
    def scored_edge_count(self) -> int:
        """Alias: every raw edge is batch-scored via calculate_transition_log_weights."""
        return self.raw_edge_count

    @property
    def retained_edge_count(self) -> int:
        """Alias: the only count D_max controls."""
        return self.kept_edge_count


@dataclass(frozen=True)
class GraphDiagnostics:
    """Aggregate diagnostics for the full sparse graph."""

    layer_sizes: Tuple[int, ...]
    layer_diagnostics: Tuple[LayerBuildDiagnostics, ...]

    @property
    def total_rejections(self) -> int:
        return sum(len(item.rejected_proposals) for item in self.layer_diagnostics)


@dataclass(frozen=True)
class EdgeScoreDiagnostics:
    """GTTM and prior contributions aligned with one retained graph edge."""

    time_index: int
    source: BeatState
    target: BeatState
    raw_feature_contributions: Mapping[str, float]
    weighted_feature_contributions: Mapping[str, float]
    data_logp: float
    data_contribution: float
    gttm_score: float
    gttm_energy: float
    gttm_contribution: float
    final_log_weight: float
    right_contexts: Tuple[BeatState, ...] = ()
    context_strategy: str = "two_pass_successor_mean"

    @property
    def right_context_count(self) -> int:
        return len(self.right_contexts)

    def to_dict(self, vocabularies: Optional[Vocabularies] = None) -> dict[str, object]:
        return {
            "time_index": self.time_index,
            "source": self.source.to_dict(vocabularies),
            "target": self.target.to_dict(vocabularies),
            "raw_feature_contributions": dict(self.raw_feature_contributions),
            "weighted_feature_contributions": dict(self.weighted_feature_contributions),
            "data_logp": self.data_logp,
            "data_contribution": self.data_contribution,
            "gttm_score": self.gttm_score,
            "gttm_energy": self.gttm_energy,
            "gttm_contribution": self.gttm_contribution,
            "final_log_weight": self.final_log_weight,
            "right_context_count": self.right_context_count,
            "right_contexts": [state.to_dict(vocabularies) for state in self.right_contexts],
            "context_strategy": self.context_strategy,
        }


@dataclass(frozen=True)
class SparseGraph:
    """Sparse layered graph plus diagnostics for later SB inference."""

    layers: Tuple[Layer, ...]
    edges_by_time: Tuple[Tuple[Edge, ...], ...]
    diagnostics: GraphDiagnostics
    edge_diagnostics_by_time: Tuple[Tuple[EdgeScoreDiagnostics, ...], ...] = ()

    def __post_init__(self) -> None:
        if not self.edge_diagnostics_by_time:
            return
        if len(self.edge_diagnostics_by_time) != len(self.edges_by_time):
            raise ValueError("edge_diagnostics_by_time must align with edges_by_time.")
        for time_index, (edges, diagnostics) in enumerate(
            zip(self.edges_by_time, self.edge_diagnostics_by_time)
        ):
            if len(edges) != len(diagnostics):
                raise ValueError(
                    f"edge_diagnostics_by_time[{time_index}] must align 1:1 with edges."
                )
            for edge, item in zip(edges, diagnostics):
                if (
                    item.time_index != edge.time_index
                    or item.source != edge.source
                    or item.target != edge.target
                ):
                    raise ValueError("edge diagnostics must describe the aligned edge.")
                if abs(item.final_log_weight - edge.log_weight) > 1e-12:
                    raise ValueError("edge diagnostic final_log_weight must match the edge.")

    def diagnostics_for_path(
        self,
        path: Sequence[BeatState],
    ) -> Tuple[EdgeScoreDiagnostics, ...]:
        """Return retained-edge diagnostics for consecutive states on a path."""
        path_items = tuple(path)
        if len(path_items) < 2 or not self.edge_diagnostics_by_time:
            return ()
        if len(path_items) != len(self.layers):
            raise ValueError("path must contain one state per graph layer.")
        selected = []
        for time_index, (source, target) in enumerate(zip(path_items, path_items[1:])):
            matches = tuple(
                item
                for edge, item in zip(
                    self.edges_by_time[time_index],
                    self.edge_diagnostics_by_time[time_index],
                )
                if edge.source == source and edge.target == target
            )
            if len(matches) != 1:
                raise ValueError(
                    f"Expected one retained edge for selected path transition {time_index}."
                )
            selected.append(matches[0])
        return tuple(selected)

    def inactive_feature_names(self, *, epsilon: float = 1e-12) -> Tuple[str, ...]:
        """List registered features with no non-zero retained-edge activation."""
        active = {
            name
            for layer in self.edge_diagnostics_by_time
            for item in layer
            for name, value in item.raw_feature_contributions.items()
            if abs(float(value)) > epsilon
        }
        return tuple(name for name in FEATURE_REGISTRY if name not in active)


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


def _mean_score_breakdowns(
    items: Sequence[TransitionScoreBreakdown],
    *,
    edge: Edge,
    right_contexts: Sequence[BeatState],
) -> EdgeScoreDiagnostics:
    """Collapse bounded successor-specific scores into one first-order edge score."""
    breakdowns = tuple(items)
    if not breakdowns:
        raise ValueError("At least one transition score breakdown is required.")
    scale = 1.0 / len(breakdowns)
    raw = {
        name: float(sum(item.raw_feature_contributions[name] for item in breakdowns) * scale)
        for name in FEATURE_REGISTRY
    }
    weighted = {
        name: float(
            sum(item.weighted_feature_contributions[name] for item in breakdowns) * scale
        )
        for name in FEATURE_REGISTRY
    }
    data_logp = float(sum(item.data_logp for item in breakdowns) * scale)
    data_contribution = float(sum(item.data_contribution for item in breakdowns) * scale)
    gttm_score = float(sum(weighted.values()))
    gttm_energy = -gttm_score
    gttm_contribution = float(sum(item.gttm_contribution for item in breakdowns) * scale)
    final_log_weight = float(data_contribution + gttm_contribution)
    return EdgeScoreDiagnostics(
        time_index=edge.time_index,
        source=edge.source,
        target=edge.target,
        raw_feature_contributions=raw,
        weighted_feature_contributions=weighted,
        data_logp=data_logp,
        data_contribution=data_contribution,
        gttm_score=gttm_score,
        gttm_energy=gttm_energy,
        gttm_contribution=gttm_contribution,
        final_log_weight=final_log_weight,
        right_contexts=tuple(right_contexts),
    )


def _rescore_retained_edges_with_windows(
    edge_layers: Sequence[Sequence[Edge]],
    *,
    end_layer: Layer,
    prior: Prior,
    weights: Optional[PriorWeights],
    vocabularies: Vocabularies,
    edo: int,
) -> tuple[Tuple[Tuple[Edge, ...], ...], Tuple[Tuple[EdgeScoreDiagnostics, ...], ...]]:
    """Second pass: score retained edges against their bounded retained successors.

    For an edge A->B, every retained B->C target supplies one right-context
    window. Their scores are averaged uniformly. Since B has at most d_max
    retained successors, this remains O(E * d_max) and does not alter support.
    """
    provisional_layers = tuple(tuple(layer) for layer in edge_layers)
    rescored_layers: list[Tuple[Edge, ...]] = []
    diagnostic_layers: list[Tuple[EdgeScoreDiagnostics, ...]] = []

    for time_index, edges in enumerate(provisional_layers):
        successor_targets: dict[BeatState, Tuple[BeatState, ...]] = {}
        if time_index + 1 < len(provisional_layers):
            grouped: dict[BeatState, set[BeatState]] = {}
            for successor_edge in provisional_layers[time_index + 1]:
                grouped.setdefault(successor_edge.source, set()).add(successor_edge.target)
            successor_targets = {
                source: tuple(sorted(targets, key=_state_sort_key))
                for source, targets in grouped.items()
            }

        flattened_queries: list[PriorQuery] = []
        flattened_windows: list[TransitionWindow | None] = []
        group_sizes: list[int] = []
        contexts_by_edge: list[Tuple[BeatState, ...]] = []
        for edge in edges:
            right_contexts = successor_targets.get(edge.target, ())
            contexts_by_edge.append(right_contexts)
            windows: tuple[TransitionWindow | None, ...]
            if right_contexts:
                windows = tuple(
                    TransitionWindow(right_state=right_state)
                    for right_state in right_contexts
                )
            else:
                windows = (None,)
            group_sizes.append(len(windows))
            query = PriorQuery(
                prev_state=edge.source,
                next_state=edge.target,
                time_index=edge.time_index,
                context=_build_prior_context(edge.source, end_layer, edge.time_index),
            )
            flattened_queries.extend(query for _ in windows)
            flattened_windows.extend(windows)

        batch_breakdowns = calculate_transition_score_breakdowns(
            flattened_queries,
            prior=prior,
            windows=flattened_windows,
            weights=weights,
            vocabularies=vocabularies,
            edo=edo,
        )
        offset = 0
        rescored_pairs: list[tuple[Edge, EdgeScoreDiagnostics]] = []
        for edge, group_size, right_contexts in zip(
            edges, group_sizes, contexts_by_edge
        ):
            edge_breakdowns = batch_breakdowns[offset : offset + group_size]
            offset += group_size
            diagnostic = _mean_score_breakdowns(
                edge_breakdowns,
                edge=edge,
                right_contexts=right_contexts,
            )
            rescored_pairs.append(
                (
                    Edge(
                        time_index=edge.time_index,
                        source=edge.source,
                        target=edge.target,
                        log_weight=diagnostic.final_log_weight,
                    ),
                    diagnostic,
                )
            )
        rescored_pairs.sort(key=lambda pair: _edge_sort_key(pair[0]))
        rescored_layers.append(tuple(pair[0] for pair in rescored_pairs))
        diagnostic_layers.append(tuple(pair[1] for pair in rescored_pairs))

    return tuple(rescored_layers), tuple(diagnostic_layers)


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
    key: RNGKey,
    d_max: int,
    proposal_budget: Optional[int] = None,
    prior_guided_proposals: Optional[bool] = None,
) -> tuple[SparseGraph, RNGKey]:
    """Build a bounded sparse graph of BeatState transitions.

    ``d_max`` controls retained outgoing edges only (REQ-13): candidate
    proposal is bounded separately by ``proposal_budget`` (falls back to
    ``sb_config.proposal_budget``). The full budgeted, deduplicated, legal
    candidate pool is batch-scored here via ``calculate_transition_log_weights``
    and only *then* trimmed to ``d_max`` by score, with deterministic
    tie-breaking (``_state_sort_key``).
    """
    if not isinstance(start_layer, Layer):
        raise TypeError("start_layer must be a Layer.")
    if not isinstance(end_layer, Layer):
        raise TypeError("end_layer must be a Layer.")
    if not isinstance(total_beats, int) or total_beats < 1:
        raise ValueError("total_beats must be >= 1.")
    if not isinstance(key, RNGKey):
        raise TypeError("key must be an RNGKey.")

    resolved_sb = SBConfig() if sb_config is None else sb_config
    resolved_style = StyleConfig() if style_config is None else style_config
    resolved_vocabs = _resolved_vocabs(vocabularies)
    resolved_prior = _resolved_prior(prior)
    resolved_edo = _edo_size(resolved_vocabs) if edo is None else edo
    validate_vocabulary_compatibility(resolved_vocabs, resolved_edo)
    resolved_proposal_budget = (
        resolved_sb.proposal_budget if proposal_budget is None else proposal_budget
    )
    resolved_prior_guided_proposals = (
        resolved_sb.prior_guided_proposals
        if prior_guided_proposals is None
        else prior_guided_proposals
    )

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
    current_key = key

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
        legal_candidate_count = 0
        scored_candidate_count = 0

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
                candidate_result, current_key = get_valid_next_states(
                    source_state,
                    current_time,
                    style_config=resolved_style,
                    vocabularies=resolved_vocabs,
                    prior=resolved_prior,
                    context=_build_prior_context(source_state, end_layer, current_time),
                    edo=resolved_edo,
                    key=current_key,
                    d_max=d_max,
                    proposal_budget=resolved_proposal_budget,
                    prior_guided_proposals=resolved_prior_guided_proposals,
                )

            raw_candidate_count += candidate_result.proposed_count
            legal_candidate_count += candidate_result.legal_count
            scored_candidate_count += candidate_result.scored_count
            rejected.extend(candidate_result.rejections)

            source_context = _build_prior_context(source_state, end_layer, current_time)
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
            # Skip the batch call entirely when there is nothing to score --
            # a source with zero surviving candidates should never trigger a
            # wasteful zero-item call into the (possibly expensive) prior.
            log_weights: Tuple[float, ...] = (
                calculate_transition_log_weights(
                    queries,
                    prior=resolved_prior,
                    weights=weights,
                    vocabularies=resolved_vocabs,
                    edo=resolved_edo,
                )
                if queries
                else ()
            )
            source_edges = [
                Edge(
                    time_index=current_time,
                    source=source_state,
                    target=query.next_state,
                    log_weight=log_weight,
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
                legal_candidate_count=legal_candidate_count,
                scored_candidate_count=scored_candidate_count,
                rejected_proposals=tuple(rejected),
                pruned_states=tuple(pruned_states),
            )
        )
        _logger.debug(f"Layer {next_time}: {len(current_layer)} sources, {raw_candidate_count} candidates, {len(next_layer)} kept, {len(edge_layers[-1])} edges")

    rescored_edge_layers, edge_diagnostics = _rescore_retained_edges_with_windows(
        edge_layers,
        end_layer=end_layer,
        prior=resolved_prior,
        weights=weights,
        vocabularies=resolved_vocabs,
        edo=resolved_edo,
    )
    return SparseGraph(
        layers=tuple(layers),
        edges_by_time=rescored_edge_layers,
        diagnostics=GraphDiagnostics(
            layer_sizes=tuple(len(layer) for layer in layers),
            layer_diagnostics=tuple(diagnostics),
        ),
        edge_diagnostics_by_time=edge_diagnostics,
    ), current_key
