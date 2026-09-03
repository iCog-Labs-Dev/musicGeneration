from __future__ import annotations

import logging
from dataclasses import dataclass
from math import isfinite, log
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

_logger = logging.getLogger(__name__)

from aimusic.core.config import SBBackend, SBConfig
from aimusic.core.core_types import BeatState, Edge, EndpointDistribution, Layer
from aimusic.planning.graph import SparseGraph
from aimusic.core.rng import RNGKey, random_unit


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


@dataclass(frozen=True)
class SBConvergenceTrace:
    """Compact convergence metadata for SB solver iterations."""

    iterations: int
    converged: bool
    final_max_delta: float
    residual_history: Tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.iterations, int) or self.iterations < 0:
            raise ValueError("iterations must be a non-negative int.")
        if not isinstance(self.converged, bool):
            raise TypeError("converged must be a bool.")
        if not isinstance(self.final_max_delta, (int, float)) or not isfinite(
            float(self.final_max_delta)
        ):
            raise ValueError("final_max_delta must be finite.")
        residual_history = tuple(float(item) for item in self.residual_history)
        for value in residual_history:
            if not isfinite(value):
                raise ValueError("residual_history must contain only finite values.")
            if value < 0.0:
                raise ValueError("residual_history must contain non-negative values.")
        object.__setattr__(self, "residual_history", residual_history)


@dataclass(frozen=True)
class BridgeMarginals:
    """Bridge mass diagnostics aligned with graph nodes and edges."""

    node_marginals_by_layer: tuple[tuple[float, ...], ...]
    edge_marginals_by_time: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        node_layers = tuple(
            tuple(float(value) for value in layer) for layer in self.node_marginals_by_layer
        )
        edge_layers = tuple(
            tuple(float(value) for value in layer) for layer in self.edge_marginals_by_time
        )
        for layer in node_layers:
            _require_probs("node_marginals_by_layer", layer)
        for layer in edge_layers:
            _require_probs("edge_marginals_by_time", layer)
        object.__setattr__(self, "node_marginals_by_layer", node_layers)
        object.__setattr__(self, "edge_marginals_by_time", edge_layers)


@dataclass(frozen=True)
class SolvedBridge:
    """Solved Schrödinger bridge represented by conditioned transition probabilities.

    `edge_probabilities_by_time[t][i]` aligns with `graph.edges_by_time[t][i]` and encodes
    the normalized conditional probability of taking that edge from its source state under
    the solved bridge.
    """

    graph: SparseGraph
    edge_probabilities_by_time: tuple[tuple[float, ...], ...]
    initial_distribution: EndpointDistribution | None = None
    marginals: BridgeMarginals | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.graph, SparseGraph):
            raise TypeError("graph must be a SparseGraph.")
        edge_probs = tuple(tuple(float(p) for p in group) for group in self.edge_probabilities_by_time)
        if len(edge_probs) != len(self.graph.edges_by_time):
            raise ValueError("edge_probabilities_by_time must align with graph.edges_by_time.")
        for t, (edges, probs) in enumerate(zip(self.graph.edges_by_time, edge_probs)):
            if len(edges) != len(probs):
                raise ValueError(
                    f"edge_probabilities_by_time[{t}] must align 1:1 with graph.edges_by_time[{t}]."
                )
            if not edges:
                raise ValueError("Bridge transition layers must not be empty.")
            _validate_conditioned_edge_probabilities(
                self.graph,
                t,
                probs,
                source_masses=(
                    None
                    if self.marginals is None
                    else self.marginals.node_marginals_by_layer[t]
                ),
            )
        object.__setattr__(self, "edge_probabilities_by_time", edge_probs)

        if self.initial_distribution is not None and self.initial_distribution.layer != self.graph.layers[0]:
            raise ValueError("initial_distribution.layer must match graph.layers[0].")
        if self.marginals is not None:
            if len(self.marginals.node_marginals_by_layer) != len(self.graph.layers):
                raise ValueError("marginals.node_marginals_by_layer must align with graph.layers.")
            if len(self.marginals.edge_marginals_by_time) != len(self.graph.edges_by_time):
                raise ValueError(
                    "marginals.edge_marginals_by_time must align with graph.edges_by_time."
                )
            for layer, probs in zip(self.graph.layers, self.marginals.node_marginals_by_layer):
                if len(layer.states) != len(probs):
                    raise ValueError("Node marginals must align with layer support sizes.")
            for edges, probs in zip(self.graph.edges_by_time, self.marginals.edge_marginals_by_time):
                if len(edges) != len(probs):
                    raise ValueError("Edge marginals must align with sparse edge counts.")


@dataclass(frozen=True)
class SampledBridgePath:
    """A single sampled trajectory through the bridge."""

    path: tuple[BeatState, ...]
    edges: tuple[Edge, ...] = ()
    debug: tuple[Mapping[str, object], ...] = ()
    log_probability: float = 0.0


@dataclass(frozen=True)
class SBSolution:
    """Solved log-space SB potentials aligned with the graph layers."""

    problem: SBProblem
    log_forward_potentials: Tuple[Tuple[float, ...], ...]
    log_backward_potentials: Tuple[Tuple[float, ...], ...]
    trace: SBConvergenceTrace
    marginals: BridgeMarginals | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.problem, SBProblem):
            raise TypeError("problem must be an SBProblem.")
        if not isinstance(self.trace, SBConvergenceTrace):
            raise TypeError("trace must be an SBConvergenceTrace.")

        forward = tuple(tuple(layer) for layer in self.log_forward_potentials)
        backward = tuple(tuple(layer) for layer in self.log_backward_potentials)
        expected_sizes = self.problem.diagnostics.layer_sizes
        if len(forward) != len(expected_sizes) or len(backward) != len(expected_sizes):
            raise ValueError(
                "Potential layers must align with the SBProblem graph layers."
            )
        for idx, (expected_size, f_layer, b_layer) in enumerate(
            zip(expected_sizes, forward, backward)
        ):
            if len(f_layer) != expected_size or len(b_layer) != expected_size:
                raise ValueError(f"Potential size mismatch at layer {idx}.")
            for value in f_layer + b_layer:
                if (
                    not isinstance(value, (int, float))
                    or not np.isfinite(value)
                    and value != float("-inf")
                ):
                    raise ValueError("Potential values must be finite or -inf.")

        object.__setattr__(self, "log_forward_potentials", forward)
        object.__setattr__(self, "log_backward_potentials", backward)

    def to_bridge(self) -> SolvedBridge:
        """Convert potentials into normalized sparse bridge transitions."""
        return solved_bridge_from_solution(self)


@dataclass(frozen=True)
class _IndexedEdgeBucket:
    """Layer-local sparse edge indices and log-kernel weights."""

    time_index: int
    source_size: int
    target_size: int
    source_indices: Tuple[int, ...]
    target_indices: Tuple[int, ...]
    log_kernel_weights: Tuple[float, ...]

    def __post_init__(self) -> None:
        fields = (
            ("time_index", self.time_index),
            ("source_size", self.source_size),
            ("target_size", self.target_size),
        )
        for name, value in fields:
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative int.")

        source_indices = tuple(self.source_indices)
        target_indices = tuple(self.target_indices)
        log_kernel_weights = tuple(float(weight) for weight in self.log_kernel_weights)
        if not (len(source_indices) == len(target_indices) == len(log_kernel_weights)):
            raise ValueError("Indexed edge bucket fields must have equal lengths.")

        for src_idx in source_indices:
            if src_idx < 0 or src_idx >= self.source_size:
                raise ValueError("source_indices contain out-of-range values.")
        for dst_idx in target_indices:
            if dst_idx < 0 or dst_idx >= self.target_size:
                raise ValueError("target_indices contain out-of-range values.")
        for weight in log_kernel_weights:
            if not isfinite(weight):
                raise ValueError("log_kernel_weights must be finite.")

        object.__setattr__(self, "source_indices", source_indices)
        object.__setattr__(self, "target_indices", target_indices)
        object.__setattr__(self, "log_kernel_weights", log_kernel_weights)


@dataclass(frozen=True)
class _IndexedSBProblem:
    """Solver-internal indexed view of an SB problem."""

    problem: SBProblem
    indexed_buckets: Tuple[_IndexedEdgeBucket, ...]
    log_pi0: Tuple[float, ...]
    log_piT: Tuple[float, ...]


class SBContractError(ValueError):
    """Raised when SB graph inputs violate the solver contract."""


class SBSolverError(ValueError):
    """Raised when SB solver iterations encounter invalid numeric conditions."""


class _NumpySBBackend:
    """Small sparse log-space backend used by the SB solver."""

    @staticmethod
    def logsumexp(
        values: np.ndarray,
        *,
        underflow_floor: Optional[float] = None,
        context: str = "logsumexp",
    ) -> float:
        if values.ndim != 1:
            raise ValueError("logsumexp expects a 1D array.")
        if values.size == 0:
            return float("-inf")
        finite_mask = np.isfinite(values)
        if not np.any(finite_mask):
            return float("-inf")
        finite_values = values[finite_mask]
        max_value = float(np.max(finite_values))
        shifts = finite_values - max_value
        if underflow_floor is not None:
            min_shift = float(np.min(shifts))
            if min_shift < underflow_floor:
                raise SBSolverError(
                    f"{context} encountered shift below underflow floor: "
                    f"min_shift={min_shift:.6g}, floor={underflow_floor:.6g}."
                )
        shifted = np.exp(shifts)
        return float(max_value + np.log(np.sum(shifted)))

    @classmethod
    def reduce_by_source(
        cls,
        bucket: _IndexedEdgeBucket,
        next_values: np.ndarray,
        *,
        underflow_floor: Optional[float] = None,
    ) -> np.ndarray:
        result = np.full(bucket.source_size, float("-inf"), dtype=float)
        edge_values = (
            np.asarray(bucket.log_kernel_weights, dtype=float)
            + next_values[np.asarray(bucket.target_indices, dtype=int)]
        )
        grouped: list[list[float]] = [[] for _ in range(bucket.source_size)]
        for src_idx, edge_value in zip(bucket.source_indices, edge_values):
            grouped[src_idx].append(float(edge_value))
        for idx, group in enumerate(grouped):
            if group:
                result[idx] = cls.logsumexp(
                    np.asarray(group, dtype=float),
                    underflow_floor=underflow_floor,
                    context=f"reduce_by_source[t={bucket.time_index},src={idx}]",
                )
        return result

    @classmethod
    def reduce_by_target(
        cls,
        bucket: _IndexedEdgeBucket,
        prev_values: np.ndarray,
        *,
        underflow_floor: Optional[float] = None,
    ) -> np.ndarray:
        result = np.full(bucket.target_size, float("-inf"), dtype=float)
        edge_values = (
            np.asarray(bucket.log_kernel_weights, dtype=float)
            + prev_values[np.asarray(bucket.source_indices, dtype=int)]
        )
        grouped: list[list[float]] = [[] for _ in range(bucket.target_size)]
        for dst_idx, edge_value in zip(bucket.target_indices, edge_values):
            grouped[dst_idx].append(float(edge_value))
        for idx, group in enumerate(grouped):
            if group:
                result[idx] = cls.logsumexp(
                    np.asarray(group, dtype=float),
                    underflow_floor=underflow_floor,
                    context=f"reduce_by_target[t={bucket.time_index},dst={idx}]",
                )
        return result


def _ensure_layer_times_are_contiguous(layers: Sequence[Layer]) -> None:
    for idx, layer in enumerate(layers):
        if layer.time_index != idx:
            raise SBContractError(
                f"Layer time_index mismatch at position {idx}: "
                f"expected {idx}, got {layer.time_index}."
            )


def _validate_edges_against_layers(
    layers: Sequence[Layer],
    edges_by_time: Sequence[Sequence[Edge]],
) -> None:
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
                raise SBContractError(f"Edge source is not in layer {t} support.")
            if edge.target not in target_states:
                raise SBContractError(f"Edge target is not in layer {t + 1} support.")


def _layers_have_identical_support(left: Layer, right: Layer) -> bool:
    if left.time_index != right.time_index:
        return False
    return left.states == right.states


def _validate_endpoints(
    layers: Sequence[Layer],
    pi0: EndpointDistribution,
    piT: EndpointDistribution,
) -> None:
    if not _layers_have_identical_support(pi0.layer, layers[0]):
        raise SBContractError(
            "pi0.layer must exactly match graph.layers[0] support and time_index."
        )
    if not _layers_have_identical_support(piT.layer, layers[-1]):
        raise SBContractError(
            "piT.layer must exactly match graph.layers[-1] support and time_index."
        )


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


def _validate_horizon_matches_graph(
    sb_config: SBConfig,
    layers: Sequence[Layer],
) -> None:
    expected_horizon = len(layers) - 1
    if sb_config.horizon_t != expected_horizon:
        raise SBContractError(
            "SBConfig horizon_t must match the sparse graph horizon "
            f"(expected {expected_horizon}, got {sb_config.horizon_t})."
        )


def _positive_mass_state_indices(endpoint: EndpointDistribution) -> Tuple[int, ...]:
    return tuple(
        idx
        for idx, probability in enumerate(endpoint.probabilities)
        if probability > 0.0
    )


def _compute_endpoint_reachability(
    layers: Sequence[Layer],
    edges_by_time: Sequence[Sequence[Edge]],
    pi0: EndpointDistribution,
    piT: EndpointDistribution,
) -> tuple[Tuple[set[BeatState], ...], Tuple[set[BeatState], ...]]:
    forward_reachable: list[set[BeatState]] = [set() for _ in layers]
    backward_reachable: list[set[BeatState]] = [set() for _ in layers]

    for idx in _positive_mass_state_indices(pi0):
        forward_reachable[0].add(layers[0].states[idx])
    for time_index, edge_group in enumerate(edges_by_time):
        next_reachable = forward_reachable[time_index + 1]
        current_reachable = forward_reachable[time_index]
        for edge in edge_group:
            if edge.source in current_reachable:
                next_reachable.add(edge.target)

    for idx in _positive_mass_state_indices(piT):
        backward_reachable[-1].add(layers[-1].states[idx])
    for reverse_index, edge_group in enumerate(reversed(edges_by_time)):
        time_index = len(edges_by_time) - 1 - reverse_index
        current_reachable = backward_reachable[time_index]
        next_reachable = backward_reachable[time_index + 1]
        for edge in edge_group:
            if edge.target in next_reachable:
                current_reachable.add(edge.source)

    return tuple(forward_reachable), tuple(backward_reachable)


def _validate_endpoint_reachability(
    layers: Sequence[Layer],
    edges_by_time: Sequence[Sequence[Edge]],
    pi0: EndpointDistribution,
    piT: EndpointDistribution,
) -> None:
    forward_reachable, backward_reachable = _compute_endpoint_reachability(
        layers,
        edges_by_time,
        pi0,
        piT,
    )

    for idx in _positive_mass_state_indices(pi0):
        state = layers[0].states[idx]
        if state not in backward_reachable[0]:
            raise SBContractError(
                "pi0 has positive mass on support that cannot reach any positive-mass "
                f"piT state at index {idx}."
            )

    final_layer_index = len(layers) - 1
    for idx in _positive_mass_state_indices(piT):
        state = layers[-1].states[idx]
        if state not in forward_reachable[final_layer_index]:
            raise SBContractError(
                "piT has positive mass on support unreachable from any positive-mass "
                f"pi0 state at index {idx}."
            )


def _endpoint_probabilities_to_logs(
    endpoint: EndpointDistribution,
) -> Tuple[float, ...]:
    return tuple(
        float("-inf") if prob == 0.0 else float(np.log(prob))
        for prob in endpoint.probabilities
    )


def _select_backend(sb_config: SBConfig) -> type[_NumpySBBackend]:
    if sb_config.backend_selection is SBBackend.NUMPY:
        return _NumpySBBackend
    if sb_config.backend_selection is SBBackend.JAX:
        raise NotImplementedError("SBBackend.JAX is not implemented yet.")
    raise SBSolverError(f"Unsupported SB backend: {sb_config.backend_selection!r}")


def _index_problem(problem: SBProblem) -> _IndexedSBProblem:
    layers = problem.graph.layers
    state_to_index = [
        {state: idx for idx, state in enumerate(layer.states)} for layer in layers
    ]
    indexed_buckets = []
    for bucket_idx, edge_group in enumerate(problem.graph.edges_by_time):
        source_indices = []
        target_indices = []
        log_kernel_weights = []
        for edge in edge_group:
            source_indices.append(state_to_index[bucket_idx][edge.source])
            target_indices.append(state_to_index[bucket_idx + 1][edge.target])
            log_kernel_weights.append(edge.log_weight / problem.sb_config.temperature)
        indexed_buckets.append(
            _IndexedEdgeBucket(
                time_index=bucket_idx,
                source_size=len(layers[bucket_idx]),
                target_size=len(layers[bucket_idx + 1]),
                source_indices=tuple(source_indices),
                target_indices=tuple(target_indices),
                log_kernel_weights=tuple(log_kernel_weights),
            )
        )

    return _IndexedSBProblem(
        problem=problem,
        indexed_buckets=tuple(indexed_buckets),
        log_pi0=_endpoint_probabilities_to_logs(problem.pi0),
        log_piT=_endpoint_probabilities_to_logs(problem.piT),
    )


def _validate_log_array(
    name: str, values: np.ndarray, allow_negative_inf: bool = True
) -> None:
    if values.ndim != 1:
        raise SBSolverError(f"{name} must be a 1D array.")
    if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
        raise SBSolverError(f"{name} contains invalid numeric values.")
    if not allow_negative_inf and np.any(~np.isfinite(values)):
        raise SBSolverError(f"{name} must be finite.")


def _require_non_empty_log_support(name: str, values: np.ndarray) -> None:
    if not np.any(np.isfinite(values)):
        raise SBSolverError(f"{name} has empty finite support.")


def _propagate_backward(
    indexed_problem: _IndexedSBProblem,
    backend: type[_NumpySBBackend],
    log_terminal_backward: np.ndarray,
    *,
    underflow_floor: float,
) -> list[np.ndarray]:
    layers = indexed_problem.problem.graph.layers
    backward = [np.full(len(layer), float("-inf"), dtype=float) for layer in layers]
    backward[-1] = log_terminal_backward.copy()
    _validate_log_array("log_terminal_backward", backward[-1])
    _require_non_empty_log_support("log_terminal_backward", backward[-1])
    for bucket in reversed(indexed_problem.indexed_buckets):
        backward[bucket.time_index] = backend.reduce_by_source(
            bucket,
            backward[bucket.time_index + 1],
            underflow_floor=underflow_floor,
        )
        _validate_log_array(
            f"log_backward_potentials[{bucket.time_index}]",
            backward[bucket.time_index],
        )
        _require_non_empty_log_support(
            f"log_backward_potentials[{bucket.time_index}]",
            backward[bucket.time_index],
        )
    return backward


def _propagate_forward(
    indexed_problem: _IndexedSBProblem,
    backend: type[_NumpySBBackend],
    log_initial_forward: np.ndarray,
    *,
    underflow_floor: float,
) -> list[np.ndarray]:
    layers = indexed_problem.problem.graph.layers
    forward = [np.full(len(layer), float("-inf"), dtype=float) for layer in layers]
    forward[0] = log_initial_forward.copy()
    _validate_log_array("log_initial_forward", forward[0])
    _require_non_empty_log_support("log_initial_forward", forward[0])
    for bucket in indexed_problem.indexed_buckets:
        forward[bucket.time_index + 1] = backend.reduce_by_target(
            bucket,
            forward[bucket.time_index],
            underflow_floor=underflow_floor,
        )
        _validate_log_array(
            f"log_forward_potentials[{bucket.time_index + 1}]",
            forward[bucket.time_index + 1],
        )
        _require_non_empty_log_support(
            f"log_forward_potentials[{bucket.time_index + 1}]",
            forward[bucket.time_index + 1],
        )
    return forward


def _require_finite_support(
    message: np.ndarray,
    endpoint_log_probs: Tuple[float, ...],
    *,
    endpoint_name: str,
) -> None:
    for idx, (message_value, endpoint_value) in enumerate(
        zip(message, endpoint_log_probs)
    ):
        if np.isfinite(endpoint_value) and not np.isfinite(message_value):
            raise SBSolverError(
                f"{endpoint_name} has positive mass on unreachable support at index {idx}."
            )


def _safe_difference(
    log_probs: Tuple[float, ...],
    message: np.ndarray,
    *,
    name: str,
    underflow_floor: float,
) -> np.ndarray:
    _require_finite_support(message, log_probs, endpoint_name=name)
    result = np.full(len(log_probs), float("-inf"), dtype=float)
    for idx, (log_prob, message_value) in enumerate(zip(log_probs, message)):
        if np.isfinite(log_prob):
            result[idx] = float(log_prob - message_value)
    _validate_log_array(name, result)
    _require_non_empty_log_support(name, result)
    return result


def _max_abs_delta(left: np.ndarray, right: np.ndarray) -> float:
    max_delta = 0.0
    for lhs, rhs in zip(left, right):
        if np.isneginf(lhs) and np.isneginf(rhs):
            continue
        if not np.isfinite(lhs) or not np.isfinite(rhs):
            return float("inf")
        max_delta = max(max_delta, abs(float(lhs - rhs)))
    return float(max_delta)


def _tuplify_layers(values: Sequence[np.ndarray]) -> Tuple[Tuple[float, ...], ...]:
    return tuple(tuple(float(item) for item in layer) for layer in values)


def _normalize_log_probs(log_values: np.ndarray) -> np.ndarray:
    normalizer = _NumpySBBackend.logsumexp(log_values, context="normalize_log_probs")
    return log_values - normalizer


def _validate_conditioned_edge_probabilities(
    graph: SparseGraph,
    time_index: int,
    probs: Sequence[float],
    *,
    source_masses: Optional[Sequence[float]] = None,
    atol: float = 1e-8,
) -> None:
    grouped_totals: dict[BeatState, float] = {}
    grouped_counts: dict[BeatState, int] = {}
    for edge, prob in zip(graph.edges_by_time[time_index], probs):
        if not isfinite(prob):
            raise ValueError(f"edge_probabilities_by_time[{time_index}] contains non-finite values.")
        if prob < 0.0:
            raise ValueError(f"edge_probabilities_by_time[{time_index}] contains negative values.")
        grouped_totals[edge.source] = grouped_totals.get(edge.source, 0.0) + float(prob)
        grouped_counts[edge.source] = grouped_counts.get(edge.source, 0) + 1
    if source_masses is not None and len(source_masses) != len(graph.layers[time_index].states):
        raise ValueError("source_masses must align with the source layer support size.")
    for source_index, source in enumerate(graph.layers[time_index].states):
        source_mass = None if source_masses is None else float(source_masses[source_index])
        is_active_source = source_mass is None or source_mass > 0.0
        if grouped_counts.get(source, 0) == 0:
            if is_active_source:
                raise ValueError(
                    f"edge_probabilities_by_time[{time_index}] has no outgoing support for an active source state."
                )
            continue
        total = grouped_totals[source]
        expected_total = 1.0 if is_active_source else 0.0
        if abs(total - expected_total) > atol:
            raise ValueError(
                f"edge_probabilities_by_time[{time_index}] must normalize over active source support."
            )


def _compute_bridge_marginals_and_conditionals(
    solution: SBSolution,
) -> tuple[tuple[tuple[float, ...], ...], BridgeMarginals]:
    graph = solution.problem.graph
    temperature = solution.problem.sb_config.temperature
    backward = solution.log_backward_potentials
    forward = solution.log_forward_potentials
    state_to_index = [
        {state: idx for idx, state in enumerate(layer.states)} for layer in graph.layers
    ]

    node_marginals_by_layer: list[tuple[float, ...]] = []
    for layer_idx, layer in enumerate(graph.layers):
        log_mass = np.asarray(forward[layer_idx], dtype=float) + np.asarray(
            backward[layer_idx], dtype=float
        )
        normalized = _normalize_log_probs(log_mass)
        probs = tuple(float(np.exp(value)) for value in normalized)
        _require_probs(f"node_marginals_by_layer[{layer_idx}]", probs)
        node_marginals_by_layer.append(probs)

    edge_conditionals_by_time: list[tuple[float, ...]] = []
    edge_marginals_by_time: list[tuple[float, ...]] = []
    for time_index, edges in enumerate(graph.edges_by_time):
        source_log_norm = np.asarray(backward[time_index], dtype=float)
        source_log_mass = np.asarray(forward[time_index], dtype=float) + source_log_norm
        conditioned_probs: list[float] = []
        edge_marginals: list[float] = []
        for edge in edges:
            source_idx = state_to_index[time_index][edge.source]
            target_idx = state_to_index[time_index + 1][edge.target]
            log_edge_marginal = (
                forward[time_index][source_idx]
                + (edge.log_weight / temperature)
                + backward[time_index + 1][target_idx]
            )
            if not np.isfinite(source_log_mass[source_idx]):
                conditioned_probs.append(0.0)
            else:
                log_conditional = (
                    (edge.log_weight / temperature)
                    + backward[time_index + 1][target_idx]
                    - source_log_norm[source_idx]
                )
                conditioned_probs.append(float(np.exp(log_conditional)))
            edge_marginals.append(float(np.exp(log_edge_marginal)))
        conditioned_tuple = tuple(conditioned_probs)
        edge_conditionals_by_time.append(conditioned_tuple)
        edge_marginals_tuple = tuple(edge_marginals)
        _require_probs(f"edge_marginals_by_time[{time_index}]", edge_marginals_tuple)
        edge_marginals_by_time.append(edge_marginals_tuple)
        _validate_conditioned_edge_probabilities(
            graph,
            time_index,
            conditioned_tuple,
            source_masses=node_marginals_by_layer[time_index],
        )

    marginals = BridgeMarginals(
        node_marginals_by_layer=tuple(node_marginals_by_layer),
        edge_marginals_by_time=tuple(edge_marginals_by_time),
    )
    return tuple(edge_conditionals_by_time), marginals


def solved_bridge_from_solution(solution: SBSolution) -> SolvedBridge:
    """Build a normalized sparse bridge object from a solved SB solution."""
    if not isinstance(solution, SBSolution):
        raise TypeError("solution must be an SBSolution.")
    edge_probabilities_by_time, marginals = _compute_bridge_marginals_and_conditionals(solution)
    return SolvedBridge(
        graph=solution.problem.graph,
        edge_probabilities_by_time=edge_probabilities_by_time,
        initial_distribution=solution.problem.pi0,
        marginals=marginals,
    )


def map_bridge_path(
    bridge: SolvedBridge,
    *,
    start_state: BeatState | None = None,
) -> tuple[tuple[BeatState, ...], float]:
    """Extract the highest-probability bridge path with deterministic tie-breaking."""
    if not isinstance(bridge, SolvedBridge):
        raise TypeError("bridge must be a SolvedBridge.")

    graph = bridge.graph
    transition_tables = _build_conditioned_transition_table(graph, bridge.edge_probabilities_by_time)
    state_to_index = [
        {state: idx for idx, state in enumerate(layer.states)} for layer in graph.layers
    ]

    if start_state is not None and start_state not in state_to_index[0]:
        raise ValueError("start_state must come from the bridge start layer.")

    initial_scores = np.full(len(graph.layers[0]), float("-inf"), dtype=float)
    if start_state is None:
        if bridge.initial_distribution is None:
            initial_scores[0] = 0.0
        else:
            for idx, prob in enumerate(bridge.initial_distribution.probabilities):
                if prob > 0.0:
                    initial_scores[idx] = float(np.log(prob))
    else:
        initial_scores[state_to_index[0][start_state]] = 0.0

    scores: list[np.ndarray] = [
        np.full(len(layer.states), float("-inf"), dtype=float) for layer in graph.layers
    ]
    backpointers: list[list[int | None]] = [
        [None for _ in layer.states] for layer in graph.layers
    ]
    scores[0] = initial_scores

    for time_index, layer in enumerate(graph.layers[:-1]):
        next_scores = np.full(len(graph.layers[time_index + 1]), float("-inf"), dtype=float)
        for source_idx, source_state in enumerate(layer.states):
            source_score = scores[time_index][source_idx]
            if not np.isfinite(source_score):
                continue
            outgoing_edges, outgoing_probs = transition_tables[time_index][source_state]
            for edge, prob in zip(outgoing_edges, outgoing_probs):
                if prob <= 0.0:
                    continue
                target_idx = state_to_index[time_index + 1][edge.target]
                candidate_score = float(source_score + np.log(prob))
                if candidate_score > next_scores[target_idx]:
                    next_scores[target_idx] = candidate_score
                    backpointers[time_index + 1][target_idx] = source_idx
        scores[time_index + 1] = next_scores

    final_scores = scores[-1]
    if not np.any(np.isfinite(final_scores)):
        raise ValueError("Bridge has no finite-support MAP path.")
    best_final_idx = int(np.argmax(final_scores))
    best_score = float(final_scores[best_final_idx])

    path_indices = [best_final_idx]
    for layer_idx in range(len(graph.layers) - 1, 0, -1):
        previous_idx = backpointers[layer_idx][path_indices[-1]]
        if previous_idx is None:
            raise ValueError("Bridge backpointer reconstruction failed.")
        path_indices.append(previous_idx)
    path_indices.reverse()

    path = tuple(
        graph.layers[layer_idx].states[state_idx]
        for layer_idx, state_idx in enumerate(path_indices)
    )
    return path, best_score


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

    layers = graph.layers
    edges_by_time = graph.edges_by_time

    if len(layers) < 2:
        raise SBContractError("SparseGraph must contain at least two layers.")

    if sb_config is None:
        resolved_config = SBConfig(horizon_t=len(layers) - 1)
    else:
        resolved_config = sb_config
    if not isinstance(resolved_config, SBConfig):
        raise TypeError("sb_config must be an SBConfig.")

    _validate_horizon_matches_graph(resolved_config, layers)
    _ensure_layer_times_are_contiguous(layers)
    _validate_edges_against_layers(layers, edges_by_time)
    _validate_endpoints(layers, pi0, piT)

    out_degree, in_degree = _compute_in_out_degrees(layers, edges_by_time)
    _validate_reachability_sanity(layers, out_degree, in_degree)
    _validate_endpoint_reachability(layers, edges_by_time, pi0, piT)

    layer_sizes = tuple(len(layer) for layer in layers)
    edge_counts_by_time = tuple(len(group) for group in edges_by_time)

    diagnostics = SBProblemDiagnostics(
        horizon_t=len(layers) - 1,
        layer_sizes=layer_sizes,
        edge_counts_by_time=edge_counts_by_time,
        total_edge_count=sum(edge_counts_by_time),
        zero_outdegree_count=sum(
            1
            for layer in layers[:-1]
            for state in layer.states
            if out_degree[state] == 0
        ),
        zero_indegree_count=sum(
            1 for layer in layers[1:] for state in layer.states if in_degree[state] == 0
        ),
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


def solve_sb(problem: SBProblem) -> SBSolution:
    """Solve SB forward/backward endpoint scaling updates in log-space."""

    if not isinstance(problem, SBProblem):
        raise TypeError("problem must be an SBProblem.")

    backend = _select_backend(problem.sb_config)
    indexed_problem = _index_problem(problem)
    log_terminal_backward = np.zeros(problem.diagnostics.piT_support_size, dtype=float)
    final_max_delta = float("inf")
    converged = False
    iterations = 0
    residual_history: list[float] = []

    for iteration in range(1, problem.sb_config.max_iterations + 1):
        backward = _propagate_backward(
            indexed_problem,
            backend,
            log_terminal_backward,
            underflow_floor=problem.sb_config.log_underflow_floor,
        )
        log_initial_forward = _safe_difference(
            indexed_problem.log_pi0,
            backward[0],
            name="log_initial_forward",
            underflow_floor=problem.sb_config.log_underflow_floor,
        )
        forward = _propagate_forward(
            indexed_problem,
            backend,
            log_initial_forward,
            underflow_floor=problem.sb_config.log_underflow_floor,
        )
        next_log_terminal_backward = _safe_difference(
            indexed_problem.log_piT,
            forward[-1],
            name="log_terminal_backward",
            underflow_floor=problem.sb_config.log_underflow_floor,
        )

        final_max_delta = _max_abs_delta(
            next_log_terminal_backward,
            log_terminal_backward,
        )
        iterations = iteration
        residual_history.append(float(final_max_delta))
        log_terminal_backward = next_log_terminal_backward
        _logger.debug(f"SB iteration {iteration}/{problem.sb_config.max_iterations}: max_delta={final_max_delta:g}")
        if final_max_delta <= problem.sb_config.tolerance:
            converged = True
            break

    _logger.info(f"SB finished: {iterations} iterations, converged={converged}, final_delta={final_max_delta:g}")
    if not converged:
        _logger.warning(f"SB did not converge after {iterations} iterations (delta={final_max_delta:g})")
        if problem.sb_config.raise_on_non_convergence:
            raise SBSolverError(
                "SB solver did not converge within max_iterations "
                f"({problem.sb_config.max_iterations})."
            )

    backward = _propagate_backward(
        indexed_problem,
        backend,
        log_terminal_backward,
        underflow_floor=problem.sb_config.log_underflow_floor,
    )
    log_initial_forward = _safe_difference(
        indexed_problem.log_pi0,
        backward[0],
        name="log_initial_forward",
        underflow_floor=problem.sb_config.log_underflow_floor,
    )
    forward = _propagate_forward(
        indexed_problem,
        backend,
        log_initial_forward,
        underflow_floor=problem.sb_config.log_underflow_floor,
    )

    trace = SBConvergenceTrace(
        iterations=iterations,
        converged=converged,
        final_max_delta=float(final_max_delta),
        residual_history=tuple(residual_history),
    )
    solution = SBSolution(
        problem=problem,
        log_forward_potentials=_tuplify_layers(forward),
        log_backward_potentials=_tuplify_layers(backward),
        trace=trace,
    )
    _, marginals = _compute_bridge_marginals_and_conditionals(solution)
    return SBSolution(
        problem=solution.problem,
        log_forward_potentials=solution.log_forward_potentials,
        log_backward_potentials=solution.log_backward_potentials,
        trace=solution.trace,
        marginals=marginals,
    )


# --- Pure Path Sampling Implementation ---

def _require_probs(name: str, values: Sequence[float]) -> tuple[float, ...]:
    probs = tuple(float(value) for value in values)
    if not probs:
        raise ValueError(f"{name} must not be empty.")
    for idx, prob in enumerate(probs):
        if not isfinite(prob):
            raise ValueError(f"{name}[{idx}] must be finite.")
        if prob < 0.0:
            raise ValueError(f"{name}[{idx}] must be >= 0.")
    total = sum(probs)
    if total <= 0.0:
        raise ValueError(f"{name} must have positive total mass.")
    return probs


def _sample_categorical(key: RNGKey, probs: Sequence[float]) -> tuple[int, RNGKey]:
    normalized = _require_probs("probs", probs)
    total = float(sum(normalized))
    threshold, next_key = random_unit(key)
    cutoff = threshold * total
    running = 0.0
    for idx, prob in enumerate(normalized):
        running += prob
        if cutoff <= running:
            return idx, next_key
    return len(normalized) - 1, next_key


def _build_conditioned_transition_table(
    graph: SparseGraph, edge_probabilities_by_time: tuple[tuple[float, ...], ...]
) -> tuple[dict[BeatState, tuple[tuple[Edge, ...], tuple[float, ...]]], ...]:
    tables: list[dict[BeatState, tuple[tuple[Edge, ...], tuple[float, ...]]]] = []
    for edges, probs in zip(graph.edges_by_time, edge_probabilities_by_time):
        by_source: dict[BeatState, list[tuple[Edge, float]]] = {}
        for edge, prob in zip(edges, probs):
            by_source.setdefault(edge.source, []).append((edge, float(prob)))
        conditioned: dict[BeatState, tuple[tuple[Edge, ...], tuple[float, ...]]] = {}
        for source, items in by_source.items():
            outgoing_edges = tuple(edge for edge, _ in items)
            raw_probs = tuple(prob for _, prob in items)
            if float(sum(raw_probs)) <= 0.0:
                continue
            outgoing_probs = _require_probs("outgoing_probs", raw_probs)
            conditioned[source] = (outgoing_edges, outgoing_probs)
        tables.append(conditioned)
    return tuple(tables)


def sample_bridge_path(
    bridge: SolvedBridge,
    rng: RNGKey,
    *,
    start_state: BeatState | None = None,
    include_edges: bool = False,
    include_debug: bool = False,
) -> tuple[SampledBridgePath, RNGKey]:
    """Sample a full path from a solved bridge using its conditioned transitions."""

    if not isinstance(bridge, SolvedBridge):
        raise TypeError("bridge must be a SolvedBridge.")
    if not isinstance(rng, RNGKey):
        raise TypeError("rng must be an RNGKey.")

    graph = bridge.graph
    transition_tables = _build_conditioned_transition_table(graph, bridge.edge_probabilities_by_time)

    log_probability = 0.0
    if start_state is None:
        if bridge.initial_distribution is None:
            start_state = graph.layers[0].states[0]
        else:
            idx, rng = _sample_categorical(rng, bridge.initial_distribution.probabilities)
            start_state = bridge.initial_distribution.layer.states[idx]
            initial_total = float(sum(bridge.initial_distribution.probabilities))
            initial_probability = float(bridge.initial_distribution.probabilities[idx]) / initial_total
            log_probability += log(initial_probability)
    else:
        if start_state not in set(graph.layers[0].states):
            raise ValueError("start_state must come from the bridge start layer.")

    path: list[BeatState] = [start_state]
    edges: list[Edge] = []
    debug: list[Mapping[str, object]] = []

    current = start_state
    for t in range(len(graph.layers) - 1):
        table = transition_tables[t]
        if current not in table:
            raise ValueError(f"No outgoing bridge transitions from state at time {t}.")
        outgoing_edges, outgoing_probs = table[current]
        chosen_idx, rng = _sample_categorical(rng, outgoing_probs)
        selected_probability = float(outgoing_probs[chosen_idx]) / float(sum(outgoing_probs))
        log_probability += log(selected_probability)
        edge = outgoing_edges[chosen_idx]
        next_state = edge.target
        path.append(next_state)
        if include_edges:
            edges.append(edge)
        if include_debug:
            debug.append(
                {
                    "time_index": t,
                    "source": edge.source,
                    "target": edge.target,
                    "edge_probability": selected_probability,
                    "edge_log_weight": edge.log_weight,
                }
            )
        current = next_state

    return (
        SampledBridgePath(
            path=tuple(path),
            edges=tuple(edges),
            debug=tuple(debug),
            log_probability=float(log_probability),
        ),
        rng,
    )


def uniform_bridge_from_graph(graph: SparseGraph) -> SolvedBridge:
    """Utility for tests: make a bridge with uniform mass on each outgoing edge."""
    probabilities: list[tuple[float, ...]] = []
    for time_index, edges in enumerate(graph.edges_by_time):
        if not edges:
            raise ValueError("graph.edges_by_time must not contain empty edge groups.")
        counts_by_source: dict[BeatState, int] = {}
        for edge in edges:
            counts_by_source[edge.source] = counts_by_source.get(edge.source, 0) + 1
        probabilities.append(
            tuple(1.0 / counts_by_source[edge.source] for edge in edges)
        )
    return SolvedBridge(graph=graph, edge_probabilities_by_time=tuple(probabilities))
