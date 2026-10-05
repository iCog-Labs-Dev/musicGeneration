"""Parity and contract tests for the public SB numeric backend."""

from __future__ import annotations

import math
import unittest
from unittest.mock import patch

import numpy as np

from aimusic.core.config import SBBackend, SBConfig
from aimusic.core.core_types import BeatState, Edge, EndpointDistribution, Layer
from aimusic.planning.graph import GraphDiagnostics, LayerBuildDiagnostics, SparseGraph
from aimusic.planning.sb import (
    SBBackendError,
    SBContractError,
    SBProblem,
    SBSolverError,
    build_sb_problem,
    solve_sb,
)
from aimusic.planning.sb_backend import (
    DenseReferenceBackend,
    NumpySBBackend,
    SBNumericBackend as BackendProtocol,
    backend_array_contract,
    invoke_backend,
    select_sb_backend,
)


_ATOL = 1e-8


def _state(beat: int, groove: int) -> BeatState:
    return BeatState(
        meter_id=0,
        beat_in_bar=beat,
        boundary_lvl=0,
        key_id=0,
        chord_id=0,
        role_id=0,
        head_id=0,
        groove_id=groove,
    )


def _layer_diagnostics(time_index: int, source_count: int, edge_count: int) -> LayerBuildDiagnostics:
    return LayerBuildDiagnostics(
        time_index=time_index,
        source_state_count=source_count,
        raw_candidate_count=edge_count,
        unique_candidate_count=edge_count,
        kept_candidate_count=min(edge_count, source_count),
        raw_edge_count=edge_count,
        kept_edge_count=edge_count,
    )


def _graph_from_edges(
    layer_sizes: tuple[int, ...],
    edges_by_time: tuple[tuple[Edge, ...], ...],
) -> SparseGraph:
    layers = tuple(
        Layer(
            time_index=t,
            states=tuple(_state(t, g) for g in range(size)),
        )
        for t, size in enumerate(layer_sizes)
    )
    diagnostics = GraphDiagnostics(
        layer_sizes=layer_sizes,
        layer_diagnostics=tuple(
            _layer_diagnostics(t, layer_sizes[t], len(edges_by_time[t]))
            for t in range(len(edges_by_time))
        ),
    )
    return SparseGraph(layers=layers, edges_by_time=edges_by_time, diagnostics=diagnostics)


def _random_reachable_problem(rng: np.random.Generator) -> SBProblem:
    horizon = int(rng.integers(1, 3))
    layer_sizes = tuple(int(rng.integers(2, 5)) for _ in range(horizon + 1))
    edge_groups: list[tuple[Edge, ...]] = []
    for t in range(horizon):
        src_n = layer_sizes[t]
        dst_n = layer_sizes[t + 1]
        chosen: set[tuple[int, int]] = set()
        for src in range(src_n):
            dst = int(rng.integers(0, dst_n))
            chosen.add((src, dst))
        for dst in range(dst_n):
            src = int(rng.integers(0, src_n))
            chosen.add((src, dst))
        extra = int(rng.integers(0, src_n * dst_n))
        for _ in range(extra):
            chosen.add((int(rng.integers(0, src_n)), int(rng.integers(0, dst_n))))
        edges = []
        for src, dst in sorted(chosen):
            weight = float(rng.uniform(-2.0, 0.5))
            edges.append(
                Edge(
                    time_index=t,
                    source=_state(t, src),
                    target=_state(t + 1, dst),
                    log_weight=weight,
                )
            )
        edge_groups.append(tuple(edges))

    graph = _graph_from_edges(layer_sizes, tuple(edge_groups))
    pi0_raw = rng.random(layer_sizes[0])
    piT_raw = rng.random(layer_sizes[-1])
    pi0 = EndpointDistribution(
        layer=graph.layers[0],
        probabilities=tuple(pi0_raw / pi0_raw.sum()),
    )
    piT = EndpointDistribution(
        layer=graph.layers[-1],
        probabilities=tuple(piT_raw / piT_raw.sum()),
    )
    return build_sb_problem(
        graph,
        pi0,
        piT,
        sb_config=SBConfig(horizon_t=horizon, max_iterations=80, tolerance=1e-10),
    )


class _RecordingBackend:
    def __init__(self, inner: BackendProtocol) -> None:
        self.inner = inner
        self.operations: list[str] = []

    def logsumexp(self, **kwargs):  # type: ignore[no-untyped-def]
        self.operations.append("logsumexp")
        return self.inner.logsumexp(**kwargs)

    def scatter_logsumexp(self, **kwargs):  # type: ignore[no-untyped-def]
        self.operations.append("scatter_logsumexp")
        return self.inner.scatter_logsumexp(**kwargs)

    def normalize_log_probs(self, **kwargs):  # type: ignore[no-untyped-def]
        self.operations.append("normalize_log_probs")
        return self.inner.normalize_log_probs(**kwargs)


class _BrokenBackend:
    def logsumexp(self, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("backend exploded")

    def scatter_logsumexp(self, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("backend exploded")

    def normalize_log_probs(self, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("backend exploded")


def _assert_close_layers(test, left, right, *, atol: float = _ATOL) -> None:
    test.assertEqual(len(left), len(right))
    for lhs, rhs in zip(left, right):
        test.assertTrue(np.allclose(np.asarray(lhs), np.asarray(rhs), atol=atol, equal_nan=False))


class TestSBBackendContract(unittest.TestCase):
    def test_numpy_and_dense_satisfy_protocol(self):
        self.assertIsInstance(NumpySBBackend(), BackendProtocol)
        self.assertIsInstance(DenseReferenceBackend(), BackendProtocol)

    def test_array_contract_documents_padding_and_neg_inf(self):
        contract = backend_array_contract()
        self.assertEqual(contract["log_dtype"], "float64")
        self.assertEqual(contract["index_dtype"], "int64")
        self.assertEqual(contract["mask_dtype"], "bool")
        self.assertEqual(contract["missing_log_value"], "-inf")
        self.assertIn("edge_mask=False", contract["padding"])
        self.assertIn("k_max", contract["node_padding"])

    def test_masked_padding_is_ignored(self):
        backend = NumpySBBackend()
        values = np.asarray((0.0, -1.0, 50.0), dtype=np.float64)
        mask = np.asarray((True, True, False))
        expected = math.log(1.0 + math.exp(-1.0))
        self.assertAlmostEqual(
            backend.logsumexp(values, mask=mask),
            expected,
            places=12,
        )
        padded = backend.scatter_logsumexp(
            log_weights=np.asarray((math.log(0.5), math.log(0.25), 99.0)),
            gather_from=np.asarray((0, 1, 0), dtype=np.int64),
            messages=np.asarray((math.log(0.2), math.log(0.8))),
            scatter_to=np.asarray((0, 0, 0), dtype=np.int64),
            out_size=1,
            edge_mask=np.asarray((True, True, False)),
        )
        self.assertTrue(
            np.allclose(padded, np.asarray((math.log(0.5 * 0.2 + 0.25 * 0.8),)))
        )

    def test_neg_inf_is_log_zero_and_nan_is_rejected(self):
        backend = NumpySBBackend()
        self.assertEqual(
            backend.logsumexp(np.asarray((float("-inf"), float("-inf")))),
            float("-inf"),
        )
        with self.assertRaises(SBSolverError):
            backend.logsumexp(np.asarray((0.0, float("nan"))))
        with self.assertRaises(SBSolverError):
            backend.logsumexp(np.asarray((0.0, float("inf"))))
        with self.assertRaises(SBSolverError):
            backend.normalize_log_probs(np.asarray((float("-inf"), float("-inf"))))

    def test_select_jax_is_stable_backend_error(self):
        with self.assertRaises(SBBackendError):
            select_sb_backend(SBBackend.JAX)

    def test_invoke_backend_translates_unexpected_errors(self):
        with self.assertRaises(SBBackendError):
            invoke_backend(_BrokenBackend(), "scatter_logsumexp", log_weights=np.zeros(1))


class TestNumpyDenseParity(unittest.TestCase):
    def test_scatter_and_normalize_ops_match_on_duplicates_and_padding(self):
        numpy_backend = NumpySBBackend()
        dense_backend = DenseReferenceBackend()
        log_weights = np.asarray((math.log(0.4), math.log(0.1), math.log(0.4), 12.0))
        gather_from = np.asarray((0, 1, 0, 1), dtype=np.int64)
        scatter_to = np.asarray((0, 0, 1, 0), dtype=np.int64)
        messages = np.asarray((math.log(0.3), math.log(0.7)))
        edge_mask = np.asarray((True, True, True, False))
        sparse = numpy_backend.scatter_logsumexp(
            log_weights=log_weights,
            gather_from=gather_from,
            messages=messages,
            scatter_to=scatter_to,
            out_size=2,
            edge_mask=edge_mask,
        )
        dense = dense_backend.scatter_logsumexp(
            log_weights=log_weights,
            gather_from=gather_from,
            messages=messages,
            scatter_to=scatter_to,
            out_size=2,
            edge_mask=edge_mask,
        )
        self.assertTrue(np.allclose(sparse, dense, atol=_ATOL))
        expected_0 = math.log(0.4 * 0.3 + 0.1 * 0.7)
        expected_1 = math.log(0.4 * 0.3)
        self.assertTrue(np.allclose(sparse, np.asarray((expected_0, expected_1)), atol=_ATOL))

        logs = np.asarray((math.log(0.25), float("-inf"), math.log(0.75)))
        mask = np.asarray((True, True, True))
        numpy_norm = numpy_backend.normalize_log_probs(logs, mask=mask)
        dense_norm = dense_backend.normalize_log_probs(logs, mask=mask)
        self.assertTrue(np.allclose(numpy_norm, dense_norm, atol=_ATOL))
        self.assertAlmostEqual(float(np.exp(numpy_norm).sum()), 1.0, places=12)

    def test_forward_backward_marginals_conditionals_on_random_graphs(self):
        rng = np.random.default_rng(11)
        numpy_backend = NumpySBBackend()
        dense_backend = DenseReferenceBackend()
        compared = 0
        for _ in range(12):
            try:
                problem = _random_reachable_problem(rng)
            except SBContractError:
                continue
            numpy_solution = solve_sb(problem, backend=numpy_backend)
            dense_solution = solve_sb(problem, backend=dense_backend)
            compared += 1

            self.assertEqual(numpy_solution.trace.converged, dense_solution.trace.converged)
            self.assertEqual(numpy_solution.trace.iterations, dense_solution.trace.iterations)
            self.assertTrue(
                np.allclose(
                    np.asarray(numpy_solution.trace.residual_history),
                    np.asarray(dense_solution.trace.residual_history),
                    atol=1e-7,
                )
            )
            _assert_close_layers(
                self,
                numpy_solution.log_forward_potentials,
                dense_solution.log_forward_potentials,
            )
            _assert_close_layers(
                self,
                numpy_solution.log_backward_potentials,
                dense_solution.log_backward_potentials,
            )
            assert numpy_solution.marginals is not None
            assert dense_solution.marginals is not None
            _assert_close_layers(
                self,
                numpy_solution.marginals.node_marginals_by_layer,
                dense_solution.marginals.node_marginals_by_layer,
            )
            _assert_close_layers(
                self,
                numpy_solution.marginals.edge_marginals_by_time,
                dense_solution.marginals.edge_marginals_by_time,
            )
            numpy_bridge = numpy_solution.to_bridge(backend=numpy_backend)
            dense_bridge = dense_solution.to_bridge(backend=dense_backend)
            _assert_close_layers(
                self,
                numpy_bridge.edge_probabilities_by_time,
                dense_bridge.edge_probabilities_by_time,
            )
        self.assertGreaterEqual(compared, 8)

    def test_non_convergence_matches_on_tight_tolerance(self):
        rng = np.random.default_rng(3)
        problem = None
        for _ in range(20):
            try:
                problem = _random_reachable_problem(rng)
                break
            except SBContractError:
                continue
        self.assertIsNotNone(problem)
        tight = build_sb_problem(
            problem.graph,
            problem.pi0,
            problem.piT,
            sb_config=SBConfig(horizon_t=problem.sb_config.horizon_t, max_iterations=1, tolerance=1e-15),
        )
        numpy_solution = solve_sb(tight, backend=NumpySBBackend())
        dense_solution = solve_sb(tight, backend=DenseReferenceBackend())
        self.assertEqual(numpy_solution.trace.converged, dense_solution.trace.converged)
        self.assertEqual(numpy_solution.trace.iterations, 1)


class TestBackendInjectionAndFailures(unittest.TestCase):
    def test_test_double_replaces_numpy_without_selection(self):
        rng = np.random.default_rng(5)
        problem = None
        for _ in range(20):
            try:
                problem = _random_reachable_problem(rng)
                break
            except SBContractError:
                continue
        self.assertIsNotNone(problem)
        recorder = _RecordingBackend(NumpySBBackend())
        with patch("aimusic.planning.sb.select_sb_backend") as selector:
            solution = solve_sb(problem, backend=recorder)
        selector.assert_not_called()
        self.assertIn("scatter_logsumexp", recorder.operations)
        self.assertIn("normalize_log_probs", recorder.operations)
        self.assertTrue(solution.trace.iterations >= 1)

    def test_incomplete_backend_is_rejected_before_math(self):
        class _IncompleteBackend:
            def logsumexp(self, **kwargs):  # type: ignore[no-untyped-def]
                return 0.0

        rng = np.random.default_rng(9)
        problem = None
        for _ in range(20):
            try:
                problem = _random_reachable_problem(rng)
                break
            except SBContractError:
                continue
        self.assertIsNotNone(problem)
        with self.assertRaises(SBBackendError):
            solve_sb(problem, backend=_IncompleteBackend())  # type: ignore[arg-type]

    def test_broken_backend_becomes_stable_solver_error(self):
        rng = np.random.default_rng(7)
        problem = None
        for _ in range(20):
            try:
                problem = _random_reachable_problem(rng)
                break
            except SBContractError:
                continue
        self.assertIsNotNone(problem)
        with self.assertRaises(SBBackendError):
            solve_sb(problem, backend=_BrokenBackend())  # type: ignore[arg-type]

    def test_underflow_floor_failure_is_solver_error(self):
        backend = NumpySBBackend()
        with self.assertRaises(SBSolverError):
            backend.scatter_logsumexp(
                log_weights=np.asarray((0.0, -200.0)),
                gather_from=np.asarray((0, 0), dtype=np.int64),
                messages=np.asarray((0.0,)),
                scatter_to=np.asarray((0, 0), dtype=np.int64),
                out_size=1,
                underflow_floor=-50.0,
            )


if __name__ == "__main__":
    unittest.main()
