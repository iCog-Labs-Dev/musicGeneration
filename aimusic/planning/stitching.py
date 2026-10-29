"""Pure boundary costs and exact shared-endpoint path assembly."""
from __future__ import annotations

from dataclasses import dataclass, replace
from math import exp, log
from typing import TYPE_CHECKING, Sequence

from aimusic.core.config import StitchingConfig
from aimusic.core.core_types import BeatState, EndpointDistribution
from aimusic.core.vocab import Vocabularies

if TYPE_CHECKING:
    from aimusic.planning.graph import SparseGraph
    from aimusic.planning.sb import SolvedBridge


def prune_section_graphs(graphs: Sequence[SparseGraph]) -> tuple[SparseGraph, ...]:
    """Remove unreachable states across joins before fixing shared marginals.

    Two linear reachability passes over existing edges; no graph rebuilds or
    repeated solves. Both copies of a shared layer receive the same support.
    """
    forward = []
    reachable = set(graphs[0].layers[0].states)
    for graph in graphs:
        reachable &= set(graph.layers[0].states)
        layers = [reachable]
        for edges in graph.edges_by_time:
            reachable = {edge.target for edge in edges if edge.source in reachable}
            layers.append(reachable)
        forward.append(layers)
    backward = []
    reachable = set(graphs[-1].layers[-1].states)
    for graph in reversed(graphs):
        reachable &= set(graph.layers[-1].states)
        layers = [reachable]
        for edges in reversed(graph.edges_by_time):
            reachable = {edge.source for edge in edges if edge.target in reachable}
            layers.append(reachable)
        backward.append(list(reversed(layers)))
    retained = []
    for graph, front, back in zip(graphs, forward, reversed(backward)):
        supports = [a & b for a, b in zip(front, back)]
        kept_layers = tuple(replace(layer, states=tuple(state for state in layer.states if state in support))
                       for layer, support in zip(graph.layers, supports))
        indices = [tuple(i for i, edge in enumerate(edges)
                         if edge.source in supports[t] and edge.target in supports[t + 1])
                   for t, edges in enumerate(graph.edges_by_time)]
        edges_by_time = tuple(tuple(edges[i] for i in kept)
                              for edges, kept in zip(graph.edges_by_time, indices))
        diagnostics = replace(graph.diagnostics, layer_sizes=tuple(len(layer) for layer in kept_layers),
                              layer_diagnostics=tuple(replace(item,
                                  source_state_count=len(kept_layers[t]), kept_candidate_count=len(kept_layers[t + 1]),
                                  kept_edge_count=len(edges_by_time[t]))
                                  for t, item in enumerate(graph.diagnostics.layer_diagnostics)))
        retained.append(replace(graph, layers=kept_layers, edges_by_time=edges_by_time, diagnostics=diagnostics,
                                edge_diagnostics_by_time=tuple(tuple(items[i] for i in kept)
                                    for items, kept in zip(graph.edge_diagnostics_by_time, indices))))
    return tuple(retained)


def shared_boundary_distribution(
    left: EndpointDistribution,
    right: EndpointDistribution,
    config: StitchingConfig,
    vocabularies: Vocabularies,
    edo: int,
) -> EndpointDistribution:
    """Reweight ending candidates by compatibility with the next start preferences.

    The start and end roles can differ, so exact support intersection is too
    restrictive. A soft distance kernel transfers right-start probability onto
    the left support. Actual incoming/outgoing edges enforce max_cost later.
    Work is bounded by endpoint_top_k squared; no graph or SB solve is needed.
    """
    scores = []
    for state, probability in zip(left.layer.states, left.probabilities):
        terms = [log(p) - stitch_cost(state, other, config, vocabularies, edo).total
                 for other, p in zip(right.layer.states, right.probabilities) if p > 0]
        peak = max(terms)
        compatibility = peak + log(sum(exp(value - peak) for value in terms))
        scores.append(log(probability) + compatibility if probability > 0 else float("-inf"))
    peak = max(scores)
    masses = [exp(score - peak) for score in scores]
    return EndpointDistribution(left.layer, tuple(mass / sum(masses) for mass in masses))


def map_section_paths(bridges: Sequence[SolvedBridge]) -> tuple[tuple[BeatState, ...], ...]:
    """One Viterbi pass across solved kernels, without a whole-song graph/solve.

    Shared marginals are counted once: each later bridge supplies conditional
    transitions, not another initial-distribution factor at the join.
    """
    first = bridges[0]
    if first.initial_distribution is None:
        raise ValueError("Section bridges require an initial distribution.")
    scores = {state: log(p) for state, p in zip(
        first.initial_distribution.layer.states, first.initial_distribution.probabilities) if p > 0}
    backpointers = []
    for bridge in bridges:
        for edges, probabilities in zip(bridge.graph.edges_by_time, bridge.edge_probabilities_by_time):
            next_scores: dict[BeatState, float] = {}
            predecessors: dict[BeatState, BeatState] = {}
            for edge, probability in zip(edges, probabilities):
                if probability <= 0 or edge.source not in scores:
                    continue
                score = scores[edge.source] + log(probability)
                if score > next_scores.get(edge.target, float("-inf")):
                    next_scores[edge.target] = score
                    predecessors[edge.target] = edge.source
            if not next_scores:
                raise ValueError("Section bridges have no connected MAP path.")
            scores = next_scores
            backpointers.append(predecessors)
    state = max(scores, key=lambda item: scores[item])
    path = [state]
    for predecessors in reversed(backpointers):
        state = predecessors[state]
        path.append(state)
    path.reverse()
    paths = []
    offset = 0
    for bridge in bridges:
        horizon = len(bridge.graph.layers) - 1
        paths.append(tuple(path[offset:offset + horizon + 1]))
        offset += horizon
    return tuple(paths)


def section_path_score(bridge: SolvedBridge, path: Sequence[BeatState], *, include_initial: bool) -> float:
    score = 0.0
    if include_initial and bridge.initial_distribution is not None:
        score = log(bridge.initial_distribution.probability_of(path[0]))
    for t, (edges, probabilities) in enumerate(zip(
            bridge.graph.edges_by_time, bridge.edge_probabilities_by_time)):
        probability = next(p for edge, p in zip(edges, probabilities)
                           if edge.source == path[t] and edge.target == path[t + 1])
        score += log(probability)
    return score


@dataclass(frozen=True)
class StitchCost:
    key: float
    chord: float
    meter: float
    groove: float
    role: float
    head: float
    boundary: float
    total: float


def stitch_cost(
    left: BeatState,
    right: BeatState,
    config: StitchingConfig,
    vocabularies: Vocabularies,
    edo: int,
) -> StitchCost:
    """Pitch distances use EDO roots, never arithmetic on vocabulary IDs.

    Chords combine root displacement and quality mismatch equally. Other
    categorical features use 0/1 mismatch; boundary levels use |delta|/3.
    Every component is in [0, 1] for the project's boundary vocabulary.
    """
    def pitch_distance(a: int, b: int) -> float:
        delta = (a - b) % edo
        return min(delta, edo - delta) / max(1, edo // 2)

    left_chord = vocabularies.chords.token_for_id(left.chord_id)
    right_chord = vocabularies.chords.token_for_id(right.chord_id)
    components = (
        pitch_distance(vocabularies.keys.token_for_id(left.key_id).root_pc,
                       vocabularies.keys.token_for_id(right.key_id).root_pc),
        0.5 * (pitch_distance(left_chord.root_pc, right_chord.root_pc)
               + float(left_chord.quality != right_chord.quality)),
        float(left.meter_id != right.meter_id),
        float(left.groove_id != right.groove_id),
        float(left.role_id != right.role_id),
        float(left.head_id != right.head_id),
        min(1.0, abs(left.boundary_lvl - right.boundary_lvl) / 3.0),
    )
    weights = (
        config.key_weight, config.chord_weight, config.meter_weight,
        config.groove_weight, config.role_weight, config.head_weight,
        config.boundary_weight,
    )
    return StitchCost(*components, total=sum(a * b for a, b in zip(components, weights)))


@dataclass(frozen=True)
class StitchPolicy:
    """Apply costs only to edges touching an internal section boundary."""

    config: StitchingConfig
    vocabularies: Vocabularies
    edo: int
    edge_times: tuple[int, ...]

    def cost(self, left: BeatState, right: BeatState, time_index: int) -> float:
        if time_index not in self.edge_times:
            return 0.0
        return stitch_cost(left, right, self.config, self.vocabularies, self.edo).total


@dataclass(frozen=True)
class SectionJoinDiagnostics:
    left_section: str
    right_section: str
    time_index: int
    shared_state: BeatState
    incoming: StitchCost
    outgoing: StitchCost
    tolerance: float
    boundary_support_size: int = 1
    boundary_states: tuple[BeatState, ...] = ()
    boundary_probabilities: tuple[float, ...] = ()


def concatenate_section_paths(paths: Sequence[Sequence[BeatState]]) -> tuple[BeatState, ...]:
    combined: list[BeatState] = []
    for path in paths:
        if len(path) < 2:
            raise ValueError("Each section must have at least one transition.")
        if combined and combined[-1] != path[0]:
            raise ValueError("Adjacent sections do not share their endpoint.")
        combined.extend(path[1:] if combined else path)
    return tuple(combined)
