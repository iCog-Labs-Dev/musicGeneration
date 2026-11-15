"""Evaluation-side baselines that reuse (never modify) the generated sparse graph.

``reachable_uniform_walk`` is the *no-SB* ablation: it walks the very same
sparse graph the full system built, choosing each next state uniformly among
the outgoing edges that can still reach the terminal state.  It therefore keeps
the same legality constraints, endpoints, horizon, prior and scores, and differs
from the full system only in that edge weights are ignored (no Schrodinger-bridge
reweighting).
"""
from __future__ import annotations

from typing import Any, Dict, List, Set, Tuple

from aimusic.core.core_types import BeatState
from aimusic.core.rng import RNGKey, choice


def coreachable_states(graph: Any) -> List[Set[BeatState]]:
    """For each layer, the states from which the terminal layer is reachable."""
    layers = graph.layers
    reach: List[Set[BeatState]] = [set() for _ in layers]
    reach[-1] = set(layers[-1].states)
    for t in range(len(layers) - 2, -1, -1):
        reach[t] = {e.source for e in graph.edges_by_time[t] if e.target in reach[t + 1]}
    return reach


def reachable_uniform_walk(graph: Any, key: RNGKey) -> Tuple[BeatState, ...]:
    """Uniform random walk over end-reachable edges of ``graph`` (no edge weights)."""
    reach = coreachable_states(graph)
    start_layer = graph.layers[0].states
    if len(start_layer) != 1:
        raise ValueError("reachable_uniform_walk expects a singleton start layer.")
    state = start_layer[0]
    if state not in reach[0]:
        raise ValueError("Terminal layer is not reachable from the start state in the sparse graph.")
    successors: List[Dict[BeatState, List[BeatState]]] = []
    for t, edges in enumerate(graph.edges_by_time):
        table: Dict[BeatState, List[BeatState]] = {}
        for edge in edges:
            if edge.target in reach[t + 1]:
                table.setdefault(edge.source, []).append(edge.target)
        successors.append(table)
    path = [state]
    rng = key
    for t in range(len(graph.edges_by_time)):
        options = successors[t].get(state)
        if not options:  # unreachable by construction; guards against a corrupted graph
            raise ValueError(f"No end-reachable successor at layer {t}.")
        state, rng = choice(rng, options)
        path.append(state)
    return tuple(path)
