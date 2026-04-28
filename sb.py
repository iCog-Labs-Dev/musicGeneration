from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from config import SBConfig
from core_types import BeatState, Edge, EndpointDistribution, Layer
from graph import SparseGraph


@dataclass(frozen=True)
class SBProblemDiagnostics:
    """Deterministic structural summary of an SB problem instance."""

    horizon_t: int
    layer_sizes: Tuple[int, ...]
    edge_counts_by_time: Tuple[int, ...]
    total_edge_count: int
    zero_outdegree_count: int
    zero_indegree_count: int
    pi0_support_size: int
    piT_support_size: int


@dataclass(frozen=True)
class SBProblem:
    """Solver-ready SB problem contract over a sparse BeatState graph."""

    graph: SparseGraph
    pi0: EndpointDistribution
    piT: EndpointDistribution
    sb_config: SBConfig
    diagnostics: SBProblemDiagnostics


class SBContractError(ValueError):
    """Raised when SB graph inputs violate the solver contract."""


def _ensure_layer_times_are_contiguous(layers: Sequence[Layer]) -> None:
    for idx, layer in enumerate(layers):
        if layer.time_index != idx:
            raise SBContractError(
                f"Layer time_index mismatch at position {idx}: "
                f"expected {idx}, got {layer.time_index}."
            )


def _validate_edges_against_layers(layers: Sequence[Layer], edges_by_time: Sequence[Sequence[Edge]]) -> None:
    if len(edges_by_time) != len(layers) - 1:
        raise SBContractError(
            "edges_by_time must have len(layers) - 1 entries "
            f"(got {len(edges_by_time)} for {len(layers)} layers)."
        )

    for t, edge_group in enumerate(edges_by_time):
        source_layer = layers[t]
        target_layer = layers[t + 1]

        source_states = set(source_layer.states)
        target_states = set(target_layer.states)

        for edge in edge_group:
            if edge.time_index != t:
                raise SBContractError(
                    f"Edge time_index mismatch in bucket {t}: got {edge.time_index}."
                )
            if edge.source not in source_states:
                raise SBContractError(
                    f"Edge source is not in layer {t} support."
                )
            if edge.target not in target_states:
                raise SBContractError(
                    f"Edge target is not in layer {t + 1} support."
                )


def _layers_have_identical_support(left: Layer, right: Layer) -> bool:
    if left.time_index != right.time_index:
        return False
    return left.states == right.states


def _validate_endpoints(layers: Sequence[Layer], pi0: EndpointDistribution, piT: EndpointDistribution) -> None:
    if not _layers_have_identical_support(pi0.layer, layers[0]):
        raise SBContractError("pi0.layer must exactly match graph.layers[0] support and time_index.")
    if not _layers_have_identical_support(piT.layer, layers[-1]):
        raise SBContractError("piT.layer must exactly match graph.layers[-1] support and time_index.")


def _compute_in_out_degrees(
    layers: Sequence[Layer],
    edges_by_time: Sequence[Sequence[Edge]],
) -> tuple[Dict[BeatState, int], Dict[BeatState, int]]:
    out_degree: Dict[BeatState, int] = {}
    in_degree: Dict[BeatState, int] = {}

    for layer in layers:
        for state in layer.states:
            out_degree[state] = 0
            in_degree[state] = 0

    for edge_group in edges_by_time:
        for edge in edge_group:
            out_degree[edge.source] += 1
            in_degree[edge.target] += 1

    return out_degree, in_degree


def _validate_reachability_sanity(
    layers: Sequence[Layer],
    out_degree: Dict[BeatState, int],
    in_degree: Dict[BeatState, int],
) -> None:
    for layer in layers[:-1]:
        if not any(out_degree[state] > 0 for state in layer.states):
            raise SBContractError(
                f"Layer {layer.time_index} has no outgoing support to the next layer."
            )

    final_layer = layers[-1]
    if not any(in_degree[state] > 0 for state in final_layer.states):
        raise SBContractError("Final layer has no incoming support.")


def build_sb_problem(
    graph: SparseGraph,
    pi0: EndpointDistribution,
    piT: EndpointDistribution,
    sb_config: Optional[SBConfig] = None,
) -> SBProblem:
    """Build and strictly validate a solver-ready SB problem contract."""

    if not isinstance(graph, SparseGraph):
        raise TypeError("graph must be a SparseGraph.")
    if not isinstance(pi0, EndpointDistribution):
        raise TypeError("pi0 must be an EndpointDistribution.")
    if not isinstance(piT, EndpointDistribution):
        raise TypeError("piT must be an EndpointDistribution.")

    resolved_config = SBConfig() if sb_config is None else sb_config
    if not isinstance(resolved_config, SBConfig):
        raise TypeError("sb_config must be an SBConfig.")

    layers = graph.layers
    edges_by_time = graph.edges_by_time

    if len(layers) < 2:
        raise SBContractError("SparseGraph must contain at least two layers.")

    _ensure_layer_times_are_contiguous(layers)
    _validate_edges_against_layers(layers, edges_by_time)
    _validate_endpoints(layers, pi0, piT)

    out_degree, in_degree = _compute_in_out_degrees(layers, edges_by_time)
    _validate_reachability_sanity(layers, out_degree, in_degree)

    layer_sizes = tuple(len(layer) for layer in layers)
    edge_counts_by_time = tuple(len(group) for group in edges_by_time)

    diagnostics = SBProblemDiagnostics(
        horizon_t=len(layers) - 1,
        layer_sizes=layer_sizes,
        edge_counts_by_time=edge_counts_by_time,
        total_edge_count=sum(edge_counts_by_time),
        zero_outdegree_count=sum(1 for layer in layers[:-1] for state in layer.states if out_degree[state] == 0),
        zero_indegree_count=sum(1 for layer in layers[1:] for state in layer.states if in_degree[state] == 0),
        pi0_support_size=len(pi0.layer),
        piT_support_size=len(piT.layer),
    )

    return SBProblem(
        graph=graph,
        pi0=pi0,
        piT=piT,
        sb_config=resolved_config,
        diagnostics=diagnostics,
    )
