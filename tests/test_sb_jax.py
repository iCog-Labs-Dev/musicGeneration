"""Optional real-JAX parity; missing JAX skips, broken installed JAX fails."""

import importlib.util
import subprocess
import sys
import unittest
from unittest.mock import patch

import numpy as np

from tests import test_sb_backend as base
from aimusic.core.config import SBBackend, SBConfig
from aimusic.planning.sb import build_sb_problem, solve_sb
from aimusic.planning.sb_backend import (
    DenseReferenceBackend, JaxSBBackend, NumpySBBackend, SBBackendError, select_sb_backend,
)

HAS_JAX = importlib.util.find_spec("jax") is not None


class TestJaxOptionalImport(unittest.TestCase):
    def test_planning_import_does_not_load_jax(self):
        result = subprocess.run(
            [sys.executable, "-c", "import sys; import aimusic.planning.sb; "
             "assert not any(n == 'jax' or n.startswith('jax.') for n in sys.modules)"],
            capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class _JaxPairMixin:
    def setUp(self):
        self.backend_patch = patch.object(base, "_BACKENDS", (DenseReferenceBackend, JaxSBBackend))
        self.backend_patch.start()
        self.addCleanup(self.backend_patch.stop)


@unittest.skipUnless(HAS_JAX, 'Install aimusic[jax] to run JAX parity')
class TestJaxContract(_JaxPairMixin, base.TestBackendContract):
    def test_randomized_primitives_three_way(self):
        rng = np.random.default_rng(1212)
        backends = (NumpySBBackend(), DenseReferenceBackend(), JaxSBBackend())
        for sample in range(12):
            weights = rng.normal(size=18)
            weights[rng.random(18) < 0.2] = -np.inf
            messages = rng.normal(size=4)
            messages[0] = -np.inf
            args = dict(log_weights=weights, gather_from=rng.integers(0, 4, size=18),
                        messages=messages, scatter_to=rng.integers(0, 5, size=18),
                        out_size=6, edge_mask=rng.random(18) > 0.3)
            results = [b.scatter_logsumexp(**args) for b in backends]
            for result in results[1:]:
                np.testing.assert_allclose(result, results[0], atol=1e-12, rtol=1e-12,
                                           err_msg=f'scatter sample {sample}')
            values = rng.normal(size=8)
            values[0] = -np.inf
            mask = np.array([True, True, False, True, False, True, True, False])
            normalized = [b.normalize_log_probs(values, mask=mask) for b in backends]
            reductions = [b.logsumexp(values, mask=mask) for b in backends]
            np.testing.assert_allclose(reductions, reductions[0], atol=1e-12, rtol=1e-12)
            for result in normalized[1:]:
                np.testing.assert_allclose(result, normalized[0], atol=1e-12, rtol=1e-12)
        for backend in backends:
            result = backend.scatter_logsumexp(
                np.array([]), np.array([], dtype=np.int64), np.array([]),
                np.array([], dtype=np.int64), 0)
            self.assertEqual(result.shape, (0,))

    def test_selection_and_scoped_precision(self):
        backend = select_sb_backend(SBBackend.JAX)
        self.assertIsInstance(backend, JaxSBBackend)
        jax = backend._ops.jax
        before = jax.config.x64_enabled
        result = backend.normalize_log_probs(np.array([0.0, -1.0]))
        self.assertEqual(result.dtype, np.dtype(np.float64))
        self.assertEqual(jax.config.x64_enabled, before)

    def test_execution_error_translation(self):
        backend = JaxSBBackend()
        with patch.object(backend._ops, 'reduce', side_effect=RuntimeError('device failed')):
            with self.assertRaises(SBBackendError) as raised:
                backend.logsumexp(np.zeros(1))
        self.assertIsInstance(raised.exception.__cause__, RuntimeError)

    def test_masked_scatter_duplicate_floor(self):
        args = dict(log_weights=np.array([0.0, -11.0, -11.0, 100.0]),
                    gather_from=np.array([0, 1, 1, 0]), messages=np.zeros(2),
                    scatter_to=np.zeros(4, dtype=np.int64), out_size=2,
                    edge_mask=np.array([True, True, True, False]), underflow_floor=-10.5)
        np.testing.assert_allclose(JaxSBBackend().scatter_logsumexp(**args),
                                   DenseReferenceBackend().scatter_logsumexp(**args),
                                   atol=1e-12, rtol=1e-12)


@unittest.skipUnless(HAS_JAX, 'Install aimusic[jax] to run JAX parity')
class TestJaxSolverParity(_JaxPairMixin, base.TestFeasibleConvergenceAndFlow):
    def test_config_selection_end_to_end(self):
        graph, pi0, piT = base._tiny_branching()
        problem = build_sb_problem(graph, pi0, piT,
                                   sb_config=SBConfig(horizon_t=2, backend_selection=SBBackend.JAX))
        solution = solve_sb(problem)
        self.assertTrue(solution.trace.converged)
        bridge = solution.to_bridge()
        expected = solve_sb(problem, backend=DenseReferenceBackend())
        reference = expected.to_bridge(backend=DenseReferenceBackend())
        for actual, wanted in zip(bridge.edge_probabilities_by_time,
                                  reference.edge_probabilities_by_time):
            np.testing.assert_allclose(actual, wanted, atol=1e-8, rtol=1e-8)


if __name__ == '__main__':
    unittest.main()
