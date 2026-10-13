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
_BACKENDS = (NumpySBBackend, DenseReferenceBackend)

# Fixtures


def _state(beat: int, groove: int) -> BeatState:
    return BeatState(meter_id=0, beat_in_bar=beat, boundary_lvl=0, key_id=0,
                     chord_id=0, role_id=0, head_id=0, groove_id=groove)


def _graph(layer_sizes, edges_by_time) -> SparseGraph:
    layers = tuple(
        Layer(time_index=t, states=tuple(_state(t, g) for g in range(n)))
        for t, n in enumerate(layer_sizes)
    )
    diag = GraphDiagnostics(
        layer_sizes=layer_sizes,
        layer_diagnostics=tuple(
            LayerBuildDiagnostics(
                time_index=t, source_state_count=layer_sizes[t],
                raw_candidate_count=len(edges_by_time[t]),
                unique_candidate_count=len(edges_by_time[t]),
                kept_candidate_count=len(edges_by_time[t]),
                raw_edge_count=len(edges_by_time[t]),
                kept_edge_count=len(edges_by_time[t]),
            )
            for t in range(len(edges_by_time))
        ),
    )
    return SparseGraph(layers=layers, edges_by_time=edges_by_time, diagnostics=diag)


def _random_problem(rng: np.random.Generator, *, feasible: bool):
    """Random problem. If feasible, ``piT`` comes from forward-propagating ``pi0``."""
    horizon = int(rng.integers(1, 3))
    upper = 4 if feasible else 5
    sizes = tuple(int(rng.integers(2, upper)) for _ in range(horizon + 1))

    edge_groups = []
    for t in range(horizon):
        src_n, dst_n = sizes[t], sizes[t + 1]
        chosen = {(s, int(rng.integers(0, dst_n))) for s in range(src_n)}
        chosen |= {(int(rng.integers(0, src_n)), d) for d in range(dst_n)}
        if not feasible:
            chosen |= {
                (int(rng.integers(0, src_n)), int(rng.integers(0, dst_n)))
                for _ in range(int(rng.integers(0, src_n * dst_n)))
            }
        edge_groups.append(tuple(
            Edge(time_index=t, source=_state(t, s), target=_state(t + 1, d),
                 log_weight=float(rng.uniform(-2.0, 0.5)))
            for s, d in sorted(chosen)
        ))
    graph = _graph(sizes, tuple(edge_groups))

    pi0 = np.full(sizes[0], 1.0 / sizes[0])
    if feasible:
        current = pi0.copy()
        for t in range(horizon):
            src_idx = {s: i for i, s in enumerate(graph.layers[t].states)}
            dst_idx = {s: i for i, s in enumerate(graph.layers[t + 1].states)}
            by_src: dict[BeatState, list[Edge]] = {}
            for e in graph.edges_by_time[t]:
                by_src.setdefault(e.source, []).append(e)
            nxt = np.zeros(sizes[t + 1])
            for s, si in src_idx.items():
                out = by_src.get(s, [])
                if not out:
                    continue
                w = np.exp([e.log_weight for e in out])
                w = w / w.sum()
                for e, p in zip(out, w):
                    nxt[dst_idx[e.target]] += current[si] * float(p)
            current = nxt
        piT = current / current.sum()
    else:
        raw = rng.random(sizes[-1])
        piT = raw / raw.sum()

    return build_sb_problem(
        graph,
        EndpointDistribution(layer=graph.layers[0],
                             probabilities=tuple(map(float, pi0))),
        EndpointDistribution(layer=graph.layers[-1],
                             probabilities=tuple(map(float, piT))),
        sb_config=SBConfig(horizon_t=horizon, max_iterations=500, tolerance=1e-9),
    )


def _try_random(rng, *, feasible=False):
    try:
        return _random_problem(rng, feasible=feasible)
    except SBContractError:
        return None


def _tiny_branching():
    s0 = _state(0, 0)
    ma, mb = _state(1, 0), _state(1, 1)
    ea, eb = _state(2, 0), _state(2, 1)

    e0 = (
        Edge(time_index=0, source=s0, target=ma, log_weight=math.log(0.7)),
        Edge(time_index=0, source=s0, target=mb, log_weight=math.log(0.3)),
    )
    e1 = (
        Edge(time_index=1, source=ma, target=ea, log_weight=math.log(0.8)),
        Edge(time_index=1, source=ma, target=eb, log_weight=math.log(0.2)),
        Edge(time_index=1, source=mb, target=ea, log_weight=math.log(0.1)),
        Edge(time_index=1, source=mb, target=eb, log_weight=math.log(0.9)),
    )

    graph = _graph((1, 2, 2), (e0, e1))
    pi0 = EndpointDistribution(layer=graph.layers[0], probabilities=(1.0,))
    piT = EndpointDistribution(layer=graph.layers[-1], probabilities=(0.4, 0.6))
    return graph, pi0, piT

# Test doubles


class _RecordingBackend:
    def __init__(self, inner):
        self.inner, self.operations = inner, []

    def logsumexp(self, **kw):
        self.operations.append("logsumexp")
        return self.inner.logsumexp(**kw)

    def scatter_logsumexp(self, **kw):
        self.operations.append("scatter")
        return self.inner.scatter_logsumexp(**kw)

    def normalize_log_probs(self, **kw):
        self.operations.append("normalize")
        return self.inner.normalize_log_probs(**kw)


class _BrokenBackend:
    def logsumexp(self, **kw): raise RuntimeError("boom")
    def scatter_logsumexp(self, **kw): raise RuntimeError("boom")
    def normalize_log_probs(self, **kw): raise RuntimeError("boom")


def _bad_backend(*, logsumexp=None, scatter=None, normalize=None):
    """Backend whose ops return the given value (or a NumPy default)."""
    class _Bad:
        def logsumexp(self, **kw):
            return 0.0 if logsumexp is None else logsumexp
        def scatter_logsumexp(self, **kw):
            return np.zeros(2) if scatter is None else scatter
        def normalize_log_probs(self, **kw):
            return np.zeros(1) if normalize is None else normalize
    return _Bad()


# Independent oracles

def _math_normalize(log_vals):
    """Pure-math log-softmax oracle."""
    arr = np.asarray(log_vals, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        raise SBSolverError("empty support")
    m = float(finite.max())
    log_sum = math.log(sum(math.exp(float(v) - m) for v in finite))
    return np.array(
        [float(v) - m - log_sum if np.isfinite(v) else float("-inf") for v in arr]
    )


def _enumerate_paths(graph, pi0, conds):
    """Enumerate every full path and accumulate node/edge marginals independently."""
    T = len(graph.layers) - 1
    by_src = []
    for t in range(T):
        d: dict[BeatState, list[int]] = {}
        for i, e in enumerate(graph.edges_by_time[t]):
            d.setdefault(e.source, []).append(i)
        by_src.append(d)

    idx = [{s: i for i, s in enumerate(l.states)} for l in graph.layers]
    node_m = [np.zeros(len(l.states)) for l in graph.layers]
    edge_m = [np.zeros(len(es)) for es in graph.edges_by_time]
    total = 0.0

    def dfs(t, s, w, states, edges_seq):
        nonlocal total
        if t == T:
            total += w
            for tt, ss in enumerate(states):
                node_m[tt][idx[tt][ss]] += w
            for tt, ei in enumerate(edges_seq):
                edge_m[tt][ei] += w
            return
        for ei in by_src[t].get(s, []):
            p = float(conds[t][ei])
            if p > 0:
                e = graph.edges_by_time[t][ei]
                dfs(t + 1, e.target, w * p, states + [e.target], edges_seq + [ei])

    for si, s in enumerate(graph.layers[0].states):
        p0 = float(pi0.probabilities[si])
        if p0 > 0:
            dfs(0, s, p0, [s], [])
    return total, node_m, edge_m



# Backend contract


class TestBackendContract(unittest.TestCase):
    def test_protocol_and_array_contract(self):
        for B in _BACKENDS:
            self.assertIsInstance(B(), BackendProtocol)
        c = backend_array_contract()
        self.assertEqual((c["log_dtype"], c["index_dtype"], c["mask_dtype"]),
                         ("float64", "int64", "bool"))
        self.assertEqual(c["missing_log_value"], "-inf")
        self.assertIn("edge_mask=False", c["padding"])
        self.assertIn("k_max", c["node_padding"])

    def test_masking_neg_inf_and_rejection(self):
        for B in _BACKENDS:
            b = B()
            self.assertAlmostEqual(
                b.logsumexp(np.array([0.0, -1.0, 50.0]),
                            mask=np.array([True, True, False])),
                math.log(1 + math.exp(-1)), places=12,
            )
            self.assertEqual(b.logsumexp(np.array([float("-inf")] * 2)), float("-inf"))
            with self.assertRaises(SBSolverError):
                b.logsumexp(np.array([0.0, float("nan")]))
            with self.assertRaises(SBSolverError):
                b.logsumexp(np.array([0.0, float("inf")]))
            with self.assertRaises(SBSolverError):
                b.normalize_log_probs(np.array([float("-inf"), float("-inf")]))

    def test_masked_padding_ignored_in_scatter(self):
        padded = NumpySBBackend().scatter_logsumexp(
            log_weights=np.array([math.log(0.5), math.log(0.25), 99.0]),
            gather_from=np.array([0, 1, 0]),
            messages=np.array([math.log(0.2), math.log(0.8)]),
            scatter_to=np.array([0, 0, 0]), out_size=1,
            edge_mask=np.array([True, True, False]),
        )
        self.assertAlmostEqual(padded[0], math.log(0.5 * 0.2 + 0.25 * 0.8), places=12)

    def test_select_jax_and_broken_invoke(self):
        with self.assertRaises(SBBackendError):
            select_sb_backend(SBBackend.JAX)
        with self.assertRaises(SBBackendError):
            invoke_backend(_BrokenBackend(), "scatter_logsumexp",
                           log_weights=np.zeros(1))

    def test_underflow_floor_policy_is_merge_first(self):
        # Same-cell duplicates get logaddexp'd before the floor check.
        merged = dict(log_weights=np.array([0.0, -200.0]),
                      gather_from=np.array([0, 0]),
                      messages=np.array([0.0]),
                      scatter_to=np.array([0, 0]),
                      out_size=1, underflow_floor=-50.0)
        for B in _BACKENDS:
            self.assertTrue(np.isfinite(B().scatter_logsumexp(**merged)[0]))

        # Distinct cells in one group keep their -200.0 shift → floor trips.
        distinct = dict(merged, gather_from=np.array([0, 1]),
                        messages=np.array([0.0, 0.0]))
        for B in _BACKENDS:
            with self.assertRaises(SBSolverError):
                B().scatter_logsumexp(**distinct)


# NumPy vs dense parity


class TestNumpyDenseParity(unittest.TestCase):
    def test_scatter_and_normalize_parity(self):
        kw = dict(
            log_weights=np.array([math.log(0.4), math.log(0.1), math.log(0.4), 12.0]),
            gather_from=np.array([0, 1, 0, 1]),
            scatter_to=np.array([0, 0, 1, 0]),
            messages=np.array([math.log(0.3), math.log(0.7)]),
            out_size=2, edge_mask=np.array([True, True, True, False]),
        )
        r0, r1 = (B().scatter_logsumexp(**kw) for B in _BACKENDS)
        self.assertTrue(np.allclose(r0, r1, atol=_ATOL))
        self.assertTrue(np.allclose(
            r0, [math.log(0.4 * 0.3 + 0.1 * 0.7), math.log(0.4 * 0.3)], atol=_ATOL))

        logs = np.array([math.log(0.25), float("-inf"), math.log(0.75)])
        mask = np.array([True, True, True])
        n0, n1 = (B().normalize_log_probs(logs, mask=mask) for B in _BACKENDS)
        self.assertTrue(np.allclose(n0, n1, atol=_ATOL))
        self.assertAlmostEqual(float(np.exp(n0).sum()), 1.0, places=12)

    def test_random_graphs_solver_parity(self):
        rng = np.random.default_rng(11)
        compared = 0
        for _ in range(12):
            problem = _try_random(rng)
            if problem is None:
                continue
            n = solve_sb(problem, backend=NumpySBBackend())
            d = solve_sb(problem, backend=DenseReferenceBackend())
            compared += 1
            self.assertEqual(n.trace.converged, d.trace.converged)
            self.assertEqual(n.trace.iterations, d.trace.iterations)
            for a, b in (
                (n.log_forward_potentials, d.log_forward_potentials),
                (n.log_backward_potentials, d.log_backward_potentials),
                (n.marginals.node_marginals_by_layer,
                 d.marginals.node_marginals_by_layer),
                (n.marginals.edge_marginals_by_time,
                 d.marginals.edge_marginals_by_time),
            ):
                for la, lb in zip(a, b):
                    self.assertTrue(np.allclose(la, lb, atol=1e-7))
        self.assertGreaterEqual(compared, 8)

    def test_tight_tolerance_parity(self):
        rng = np.random.default_rng(3)
        problem = next(p for p in (_try_random(rng) for _ in range(20)) if p)
        tight = build_sb_problem(
            problem.graph, problem.pi0, problem.piT,
            sb_config=SBConfig(horizon_t=problem.sb_config.horizon_t,
                               max_iterations=1, tolerance=1e-15),
        )
        n, d = (solve_sb(tight, backend=B()) for B in _BACKENDS)
        self.assertEqual(n.trace.converged, d.trace.converged)
        self.assertEqual(n.trace.iterations, 1)


# Feasible convergence + endpoint / flow invariants


class TestFeasibleConvergenceAndFlow(unittest.TestCase):
    def test_feasible_problems_converge_and_conserve_flow(self):
        rng = np.random.default_rng(42)
        checked = 0
        for _ in range(15):
            problem = _try_random(rng, feasible=True)
            if problem is None:
                continue
            solution = solve_sb(problem)
            self.assertTrue(
                solution.trace.converged,
                msg=(f"iters={solution.trace.iterations} "
                     f"delta={solution.trace.final_max_delta:g}"),
            )
            graph = problem.graph
            idx = [{s: i for i, s in enumerate(l.states)} for l in graph.layers]
            m = solution.marginals

            # Endpoints equal pi0 / piT.
            self.assertTrue(np.allclose(m.node_marginals_by_layer[0],
                                        problem.pi0.probabilities, atol=1e-7))
            self.assertTrue(np.allclose(m.node_marginals_by_layer[-1],
                                        problem.piT.probabilities, atol=1e-7))

            # Source/target flow equals node marginals at each cut.
            for t, edges in enumerate(graph.edges_by_time):
                src = np.zeros(len(graph.layers[t].states))
                dst = np.zeros(len(graph.layers[t + 1].states))
                for e, mass in zip(edges, m.edge_marginals_by_time[t]):
                    src[idx[t][e.source]] += mass
                    dst[idx[t + 1][e.target]] += mass
                self.assertTrue(np.allclose(src, m.node_marginals_by_layer[t], atol=1e-7))
                self.assertTrue(np.allclose(dst, m.node_marginals_by_layer[t + 1], atol=1e-7))
            checked += 1
        self.assertGreaterEqual(checked, 8)


# Independent oracles


class TestIndependentOracles(unittest.TestCase):
    def test_bridge_marginals_match_path_enumeration(self):
        graph, pi0, piT = _tiny_branching()
        solution = solve_sb(build_sb_problem(graph, pi0, piT))
        bridge = solution.to_bridge()
        total, node_m, edge_m = _enumerate_paths(
            graph, pi0, bridge.edge_probabilities_by_time)
        self.assertAlmostEqual(total, 1.0, places=9)
        for a, b in zip(node_m, solution.marginals.node_marginals_by_layer):
            self.assertTrue(np.allclose(a, b, atol=1e-8))
        for a, b in zip(edge_m, solution.marginals.edge_marginals_by_time):
            self.assertTrue(np.allclose(a, b, atol=1e-8))

    def test_normalize_matches_math_oracle(self):
        cases = (
            np.array([0.5, -1.0, 2.0, -3.0, 1.5]),
            np.array([0.0, float("-inf"), 1.0, float("-inf")]),
        )
        for logs in cases:
            expected = _math_normalize(logs)
            for B in _BACKENDS:
                self.assertTrue(
                    np.allclose(B().normalize_log_probs(logs), expected, atol=1e-12),
                    msg=f"{B.__name__} disagrees with math oracle",
                )


# Injection and malformed returns


class TestInjectionAndMalformed(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(5)
        self.problem = next(p for p in (_try_random(rng) for _ in range(20)) if p)

    def test_injected_backend_bypasses_selector(self):
        rec = _RecordingBackend(NumpySBBackend())
        with patch("aimusic.planning.sb.select_sb_backend") as sel:
            solve_sb(self.problem, backend=rec)
        sel.assert_not_called()
        self.assertIn("scatter", rec.operations)
        self.assertIn("normalize", rec.operations)

    def test_incomplete_and_broken_backends(self):
        class _Incomplete:
            def logsumexp(self, **kw):
                return 0.0
        with self.assertRaises(SBBackendError):
            solve_sb(self.problem, backend=_Incomplete())
        with self.assertRaises(SBBackendError):
            solve_sb(self.problem, backend=_BrokenBackend())

    def test_malformed_returns_are_translated(self):
        scatter_kw = dict(
            log_weights=np.zeros(2), gather_from=np.zeros(2, dtype=np.int64),
            messages=np.zeros(1), scatter_to=np.zeros(2, dtype=np.int64),
            out_size=2,
        )
        cases = [
            ("scatter_logsumexp", _bad_backend(scatter=np.zeros(3)),
             scatter_kw, SBBackendError),
            ("scatter_logsumexp", _bad_backend(scatter=[0.0, 0.0]),
             scatter_kw, SBBackendError),
            ("scatter_logsumexp", _bad_backend(scatter=np.zeros(2, dtype=np.float32)),
             scatter_kw, SBBackendError),
            ("logsumexp", _bad_backend(logsumexp=True),
             {"values": np.zeros(2)}, SBBackendError),
            ("logsumexp", _bad_backend(logsumexp=float("nan")),
             {"values": np.zeros(2)}, SBSolverError),
            ("normalize_log_probs", _bad_backend(normalize=np.zeros(3)),
             {"log_values": np.zeros(2)}, SBBackendError),
        ]
        for op, backend, kwargs, exc in cases:
            with self.subTest(op=op, exc=exc.__name__):
                with self.assertRaises(exc):
                    invoke_backend(backend, op, **kwargs)


if __name__ == "__main__":
    unittest.main()
