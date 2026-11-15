"""Controlled tests for the MUS-15 endpoint-distribution evaluation (MUS-28 follow-up).

Hand-built graphs go through the public MUS-15 ``solve_endpoint_bridge`` and the evaluation
metrics are computed from its mass reports, so the expected numbers are known exactly.  Only two
tests run a full ``evaluate_run`` (8 beats), and they share their results.
"""
import json
import unittest
from types import SimpleNamespace
from unittest import mock

from aimusic.core.config import SBConfig
from aimusic.core.core_types import BeatState, Edge, EndpointDistribution, Layer
from aimusic.core.rng import RNGKey
from aimusic.evaluation import endpoint_metrics as ep
from aimusic.evaluation.config import expand_matrix, load_config, parse_config
from aimusic.evaluation.runner import evaluate_run
from aimusic.evaluation.schema import RunSpec
from aimusic.evaluation.stats import aggregate, flatten_metrics
from aimusic.planning.endpoint_feasibility import (
    EndpointFallbackPolicy as Policy,
    EndpointMassInfeasibleError,
    EndpointReachabilityError,
    solve_endpoint_bridge,
)
from aimusic.planning.graph import GraphDiagnostics, SparseGraph


def _s(i):
    return BeatState(0, 0, 0, i, 0, 0, 0, 0)


def _dist(t, mapping):
    states = tuple(mapping)
    return EndpointDistribution(Layer(t, states), tuple(mapping[s] for s in states))


def _graph(starts, ends, edges):
    return SparseGraph(
        layers=(Layer(0, tuple(starts)), Layer(1, tuple(ends))),
        edges_by_time=(tuple(Edge(0, a, b, w) for a, b, w in edges),),
        diagnostics=GraphDiagnostics(layer_sizes=(len(starts), len(ends)), layer_diagnostics=()),
    )


A0, A1, A2 = _s(0), _s(1), _s(2)
B0, B1, B2 = _s(10), _s(11), _s(12)
CFG = SBConfig(horizon_t=1, tolerance=1e-12, max_iterations=2000)
FULL = [(A0, B0, 0.0), (A0, B1, -1.0), (A1, B0, -0.5), (A1, B1, 0.3)]
DISJOINT = [(A0, B0, 0.0), (A1, B1, 0.0)]
TOL = 1e-6


def _solve(graph, p, q, policy=Policy.STRICT):
    return solve_endpoint_bridge(graph, p, q, sb_config=CFG, policy=policy, marginal_tolerance=TOL)


def _plan(solved, mode="distribution"):
    """Duck-typed plan carrying exactly the fields the evaluator reads from MethodAPlanResult."""
    return SimpleNamespace(
        start_mass=solved.start_mass,
        end_mass=solved.end_mass,
        endpoint_resolution=solved.resolution,
        bridge=solved.bridge,
        diagnostics=SimpleNamespace(endpoint_mode=mode, bridge_converged=True),
    )


def _spec(**kw):
    base = dict(
        condition="full", horizon=8, seed=0, edo=12, path_mode="sample",
        prior_mode="placeholder_structured", lambda_data=1.0, lambda_gttm=1.0, bridge_mode="sb",
        k_max=64, d_max=8, proposal_budget=256, prior_guided_proposals=False,
        max_iterations=200, tolerance=1e-6, temperature=1.0,
    )
    base.update(kw)
    return RunSpec(**base)


class TestFeasibleMultiState(unittest.TestCase):
    def setUp(self):
        g = _graph([A0, A1], [B0, B1], FULL)
        self.solved = _solve(g, _dist(0, {A0: 0.3, A1: 0.7}), _dist(1, {B0: 0.6, B1: 0.4}))
        self.m = ep.endpoint_metrics(_plan(self.solved), marginal_tolerance=TOL)

    def test_positive_mass_preserved_and_nothing_removed(self):
        m = self.m
        self.assertEqual((m["endpoint_support_size_start"], m["endpoint_support_size_terminal"]), (2, 2))
        for side in ("start", "terminal"):
            self.assertEqual(m[f"removed_{side}_mass"], 0.0)
            self.assertEqual(m[f"trimmed_{side}_mass"], 0.0)
            self.assertAlmostEqual(m[f"requested_{side}_mass"], 1.0)
            self.assertAlmostEqual(m[f"feasible_{side}_mass"], 1.0)
            self.assertAlmostEqual(m[f"realized_{side}_mass"], 1.0, places=6)
        self.assertEqual(m["accounting_status"], ep.ACCOUNTING_NO_MODIFICATION)
        self.assertFalse(m["request_modified"])
        self.assertEqual(m["modifications"], [])

    def test_realized_marginals_within_tolerance(self):
        for side in ("start", "terminal"):
            s = self.m[side]
            self.assertLessEqual(s["marginal_max_abs_vs_requested"], TOL)
            self.assertLessEqual(s["marginal_l1_vs_requested"], 2 * TOL)
            self.assertTrue(s["residual_within_tolerance"])
            self.assertTrue(s["requested_equals_realized_within_tolerance"])
            self.assertEqual(s["residual_states_outside_tolerance"], 0)
            self.assertEqual(len(s["residuals"]), 2)
            for row in s["residuals"]:
                self.assertFalse(row["removed"])
                self.assertAlmostEqual(row["realized"], row["requested"], delta=TOL)

    def test_entropy(self):
        import math

        expected = -(0.3 * math.log(0.3) + 0.7 * math.log(0.7))
        self.assertAlmostEqual(self.m["start"]["entropy_requested"], expected)
        self.assertAlmostEqual(self.m["start"]["entropy_effective"], expected)
        self.assertIsNone(self.m["start"]["entropy_empirical"])  # no samples yet: not zero

    def test_json_compatible(self):
        self.assertEqual(json.loads(json.dumps(self.m))["endpoint_mode"], "distribution")


class TestSingletonCompatibility(unittest.TestCase):
    def test_singleton_is_a_special_case_of_the_distribution_api(self):
        g = _graph([A0, A1], [B0, B1], FULL)
        solved = _solve(g, _dist(0, {A0: 1.0}), _dist(1, {B1: 1.0}))
        info = ep.singleton_endpoint_info(_plan(solved, "singleton"))
        self.assertEqual(info["generation_method"], "method_a_singleton_endpoints")
        self.assertEqual((info["endpoint_support_size_start"], info["endpoint_support_size_terminal"]), (1, 1))
        # the same machinery measured as a distribution is exact and trivially one-state
        m = ep.endpoint_metrics(_plan(solved), marginal_tolerance=TOL)
        self.assertEqual(m["endpoint_support_size_start"], 1)
        self.assertLessEqual(m["start"]["marginal_max_abs_vs_requested"], TOL)

    def test_singleton_reports_no_distribution_metrics(self):
        g = _graph([A0], [B0], [(A0, B0, 0.0)])
        info = ep.singleton_endpoint_info(_plan(_solve(g, _dist(0, {A0: 1.0}), _dist(1, {B0: 1.0})), "singleton"))
        self.assertFalse(info["distribution_metrics"]["available"])
        self.assertIn("singleton endpoint mode", info["distribution_metrics"]["reason"])
        for absent in ("start", "terminal", "calibration", "marginal_tolerance"):
            self.assertNotIn(absent, info)

    def test_singleton_defaults_keep_legacy_identity(self):
        specs = expand_matrix(load_config("config/benchmarks/m1_smoke.json"))
        self.assertEqual(specs[0].run_id, "full-h8-e12-s0-sample-4136997f")  # id from the M1 results
        self.assertEqual(specs[0].endpoint_mode, "singleton")
        self.assertEqual(specs[0].generation_method, "method_a_singleton_endpoints")


class TestPartiallyUnreachable(unittest.TestCase):
    def setUp(self):
        # A2 has no edge and B2 has no incoming edge: Level 1 unreachable support.
        self.g = _graph([A0, A1, A2], [B0, B1, B2], FULL)
        self.p = _dist(0, {A0: 0.4, A1: 0.4, A2: 0.2})
        self.q = _dist(1, {B0: 0.4, B1: 0.4, B2: 0.2})

    def test_strict_fails_with_actionable_error(self):
        with self.assertRaises(EndpointReachabilityError) as ctx:
            _solve(self.g, self.p, self.q, Policy.STRICT)
        info = ep.failure_endpoint_info(ctx.exception, "distribution")
        assert info is not None
        self.assertEqual(info["accounting_status"], ep.ACCOUNTING_STRICT_FAILURE)
        self.assertEqual(info["failure_kind"], "level1_unreachable_support")
        self.assertGreaterEqual(info["offending_state_count"], 1)
        self.assertTrue(info["message"])

    def test_level2_strict_failure_is_classified(self):
        g = _graph([A0, A1], [B0, B1], DISJOINT)
        with self.assertRaises(EndpointMassInfeasibleError) as ctx:
            _solve(g, _dist(0, {A0: 0.9, A1: 0.1}), _dist(1, {B0: 0.1, B1: 0.9}), Policy.STRICT)
        info = ep.failure_endpoint_info(ctx.exception, "distribution")
        assert info is not None
        self.assertEqual(info["failure_kind"], "level2_mass_infeasible")
        self.assertFalse(info["flow_check"]["feasible"])

    def test_non_endpoint_errors_are_not_classified(self):
        self.assertIsNone(ep.failure_endpoint_info(ValueError("x"), "distribution"))

    def test_drop_and_renormalize_reports_exactly_what_was_removed(self):
        solved = _solve(self.g, self.p, self.q, Policy.DROP_AND_RENORMALIZE)
        m = ep.endpoint_metrics(_plan(solved), marginal_tolerance=TOL)
        self.assertTrue(m["request_modified"])
        self.assertTrue(m["modifications"])  # not hidden
        self.assertEqual(m["accounting_status"], ep.ACCOUNTING_REMOVED)
        self.assertEqual(m["policy"], "drop_and_renormalize")
        for side in ("start", "terminal"):
            s = m[side]
            self.assertAlmostEqual(s["requested_mass"], 1.0)
            self.assertAlmostEqual(s["removed_mass"], 0.2)
            self.assertAlmostEqual(s["feasible_mass"], 0.8)
            self.assertAlmostEqual(s["effective_mass"], 1.0)  # renormalized
            self.assertAlmostEqual(s["realized_mass"], 1.0, places=6)
            self.assertEqual(s["removed_state_count"], 1)
            self.assertEqual((s["support_size_requested"], s["support_size_effective"]), (3, 2))
            self.assertEqual(s["accounting_status"], ep.ACCOUNTING_REMOVED)
            # realized matches the *effective* target but, by design, not the request
            self.assertTrue(s["residual_within_tolerance"])
            self.assertFalse(s["requested_equals_realized_within_tolerance"])
            removed_rows = [r for r in s["residuals"] if r["removed"]]
            self.assertEqual(len(removed_rows), 1)
            self.assertAlmostEqual(removed_rows[0]["requested"], 0.2)
            self.assertEqual(removed_rows[0]["realized"], 0.0)
            self.assertAlmostEqual(s["effective_mass"], sum(r["effective"] for r in s["residuals"]))

    def test_trim_is_distinguished_from_removal(self):
        g = _graph([A0, A1], [B0, B1], DISJOINT)
        solved = _solve(
            g, _dist(0, {A0: 0.9, A1: 0.1}), _dist(1, {B0: 0.1, B1: 0.9}), Policy.PARTIAL_TRIM_AND_RENORMALIZE
        )
        m = ep.endpoint_metrics(_plan(solved), marginal_tolerance=TOL)
        self.assertEqual(m["start"]["accounting_status"], ep.ACCOUNTING_TRIMMED)
        self.assertGreater(m["trimmed_start_mass"], 0.0)
        self.assertEqual(m["removed_start_mass"], 0.0)


class TestRepeatedSampling(unittest.TestCase):
    def setUp(self):
        g = _graph([A0, A1], [B0, B1], FULL)
        self.solved = _solve(g, _dist(0, {A0: 0.3, A1: 0.7}), _dist(1, {B0: 0.6, B1: 0.4}))
        self.plan = _plan(self.solved)
        self.key = RNGKey(seed=5).derive("eval.calibration")

    def test_empirical_frequencies_approach_the_effective_distribution(self):
        n = 600
        starts, terminals = ep.draw_endpoint_samples(self.plan, self.key, n)
        cal = ep.calibration_from_samples(self.plan, starts, terminals)
        self.assertEqual(cal["status"], ep.CAL_AVAILABLE)
        self.assertEqual(cal["n_samples"], n)
        for side in ("start", "terminal"):
            c = cal[side]
            self.assertLess(c["max_abs_vs_effective"], 0.08)
            self.assertLess(c["l1_vs_effective"], 0.16)
            self.assertEqual(c["outside_effective_support_mass"], 0.0)
            self.assertGreater(c["expected_l1_sampling_noise"], 0.0)
        m = ep.endpoint_metrics(self.plan, marginal_tolerance=TOL)
        ep.apply_calibration(m, cal)
        self.assertEqual(m["calibration_status"], "available")
        self.assertIsNotNone(m["start"]["entropy_empirical"])
        self.assertAlmostEqual(m["start"]["entropy_empirical"], m["start"]["entropy_effective"], delta=0.05)

    def test_sampling_is_reproducible_for_a_fixed_key(self):
        a = ep.draw_endpoint_samples(self.plan, self.key, 40)
        b = ep.draw_endpoint_samples(self.plan, self.key, 40)
        self.assertEqual(a, b)
        other = ep.draw_endpoint_samples(self.plan, RNGKey(seed=6).derive("eval.calibration"), 40)
        self.assertNotEqual(a, other)

    def test_too_few_or_no_samples_is_unavailable_not_zero(self):
        none = ep.calibration_from_samples(self.plan, [], [])
        self.assertEqual(none["status"], ep.CAL_NO_SAMPLES)
        starts, terminals = ep.draw_endpoint_samples(self.plan, self.key, ep.MIN_CALIBRATION_SAMPLES - 1)
        few = ep.calibration_from_samples(self.plan, starts, terminals)
        self.assertEqual(few["status"], ep.CAL_TOO_FEW)
        for result in (none, few):
            self.assertNotIn("start", result)  # no fabricated numbers
        m = ep.endpoint_metrics(self.plan, marginal_tolerance=TOL)
        ep.apply_calibration(m, few)
        self.assertEqual(m["calibration_status"], "unavailable_too_few_samples")
        self.assertNotIn("calibration_l1_vs_effective", m["start"])
        self.assertIsNone(m["start"]["entropy_empirical"])


class TestDiagnosticsGuard(unittest.TestCase):
    def setUp(self):
        g = _graph([A0, A1], [B0, B1], FULL)
        self.plan = _plan(_solve(g, _dist(0, {A0: 0.3, A1: 0.7}), _dist(1, {B0: 0.6, B1: 0.4})))

    def test_complete_plan_passes(self):
        ep.require_distribution_diagnostics(self.plan, "distribution")

    def test_mode_mismatch_is_refused_not_collapsed(self):
        self.plan.diagnostics.endpoint_mode = "singleton"
        with self.assertRaisesRegex(ep.EndpointDiagnosticsUnavailable, "refusing"):
            ep.require_distribution_diagnostics(self.plan, "distribution")

    def test_missing_diagnostics_fail_clearly(self):
        self.plan.start_mass = None
        with self.assertRaisesRegex(ep.EndpointDiagnosticsUnavailable, "start_mass"):
            ep.require_distribution_diagnostics(self.plan, "distribution")

    def test_runner_records_a_structured_failure(self):
        fake = SimpleNamespace(
            path=(), path_edge_diagnostics=(), diagnostics=SimpleNamespace(endpoint_mode="singleton")
        )
        with mock.patch("aimusic.planning.plans.run_method_a", return_value=(fake, None)):
            res = evaluate_run(_spec(endpoint_mode="distribution"))
        self.assertEqual(res["status"], "failed")
        self.assertEqual(res["failure"]["kind"], "endpoint_diagnostics_unavailable")


class TestSchemaAndAggregates(unittest.TestCase):
    def test_generation_method_follows_endpoint_mode(self):
        self.assertEqual(_spec(endpoint_mode="distribution").generation_method, "method_a_distribution_endpoints")
        self.assertEqual(_spec().generation_method, "method_a_singleton_endpoints")
        with self.assertRaises(ValueError):
            _spec(endpoint_mode="distribution", generation_method="method_a_singleton_endpoints")
        with self.assertRaises(ValueError):
            _spec(endpoint_mode="bogus")

    def test_no_sb_cannot_run_in_distribution_mode(self):
        with self.assertRaises(ValueError):
            _spec(endpoint_mode="distribution", bridge_mode="uniform", condition="no_sb")

    def test_calibration_requires_distribution_mode(self):
        with self.assertRaises(ValueError):
            _spec(calibration_samples=10)

    def test_modes_have_distinct_run_ids_and_cells(self):
        a, b = _spec(), _spec(endpoint_mode="distribution")
        self.assertNotEqual(a.run_id, b.run_id)
        self.assertNotEqual(a.group_key, b.group_key)

    def test_config_validation(self):
        with open("config/benchmarks/m2_distribution_method_a.json", encoding="utf-8") as handle:
            raw = json.load(handle)
        self.assertEqual(parse_config(raw).endpoint_mode, "distribution")
        bad = dict(raw, conditions=["full", "no_sb"])
        with self.assertRaises(ValueError):
            parse_config(bad)
        with self.assertRaises(ValueError):
            parse_config(dict(raw, endpoint_mode="both"))
        singleton = {k: v for k, v in raw.items() if k not in ("endpoint_mode", "calibration_samples")}
        with self.assertRaises(ValueError):  # distribution-only knobs on a singleton config
            parse_config(dict(singleton, endpoint_marginal_tolerance=1e-6))

    def test_aggregate_never_mixes_modes(self):
        def fake(spec, endpoint):
            return {"status": "ok", "spec": spec.to_dict(), "failure": None, "runtime": {},
                    "metrics": {"path": {"path_length": 9}, "endpoint": endpoint}}

        dist_ep = {"start": {"marginal_l1_vs_requested": 0.1}, "terminal": {"marginal_l1_vs_requested": 0.2},
                   "requested_start_mass": 1.0, "calibration": {"status": "x"}}
        results = [
            fake(_spec(), {"status": "not_applicable", "endpoint_support_size_start": 1}),
            fake(_spec(endpoint_mode="distribution"), dist_ep),
        ]
        agg = aggregate(results)
        self.assertEqual(len(agg["cells"]), 2)
        modes = {c["endpoint_mode"] for c in agg["cells"].values()}
        self.assertEqual(modes, {"singleton", "distribution"})
        for cell in agg["cells"].values():
            has = any(k.startswith("endpoint.") for k in cell["metrics"])
            self.assertEqual(has, cell["endpoint_mode"] == "distribution")
        self.assertIn("endpoint.start.marginal_l1_vs_requested", flatten_metrics(results[1]))
        self.assertFalse(any(k.startswith("endpoint.") for k in flatten_metrics(results[0])))


class TestDashboardEndpointViews(unittest.TestCase):
    @staticmethod
    def _dataset(*results):
        from pathlib import Path

        from aimusic.evaluation.ui import EvaluationDataset

        results = list(results)
        return EvaluationDataset(Path("."), {}, results, aggregate(results), "recomputed")

    @staticmethod
    def _distribution_result(seed=0):
        g = _graph([A0, A1, A2], [B0, B1, B2], FULL)
        solved = _solve(
            g, _dist(0, {A0: 0.4, A1: 0.4, A2: 0.2}), _dist(1, {B0: 0.4, B1: 0.4, B2: 0.2}),
            Policy.DROP_AND_RENORMALIZE,
        )
        metrics = ep.endpoint_metrics(_plan(solved), marginal_tolerance=TOL)
        ep.apply_calibration(metrics, ep.calibration_unavailable(ep.CAL_NO_SAMPLES, 0))
        spec = _spec(endpoint_mode="distribution", seed=seed)
        return {"status": "ok", "run_id": spec.run_id, "spec": spec.to_dict(), "failure": None,
                "runtime": {}, "metrics": {"path": {}, "endpoint": metrics}}

    def test_singleton_dataset_shows_unavailable_message_not_zeros(self):
        from aimusic.evaluation import ui

        spec = _spec()
        res = {"status": "ok", "run_id": spec.run_id, "spec": spec.to_dict(), "failure": None, "runtime": {},
               "metrics": {"path": {}, "endpoint": {"status": "not_applicable"}}}
        text = ui.render_endpoint_overview(self._dataset(res))
        self.assertIn("Distribution metrics unavailable: singleton endpoint mode.", text)
        self.assertNotIn("0.0", text)
        self.assertIn("Distribution metrics unavailable: singleton endpoint mode.", ui.render_endpoint_run_details(res))
        fig = ui.make_endpoint_chart(self._dataset(res))
        self.assertEqual(len(fig.data), 0)

    def test_distribution_dataset_shows_mass_accounting_and_unavailable_calibration(self):
        from aimusic.evaluation import ui

        ds = self._dataset(self._distribution_result())
        text = ui.render_endpoint_overview(ds)
        self.assertIn("Start boundary", text)
        self.assertIn("removed_and_renormalized", text)
        self.assertIn("unavailable_no_repeated_samples", text)
        self.assertIn("unavailable", text)  # calibration L1 has no value: shown as unavailable, not 0
        self.assertGreater(len(ui.make_endpoint_chart(ds).data), 0)
        details = ui.render_endpoint_run_details(self._distribution_result())
        self.assertIn("Per-state residuals", details)
        self.assertIn("removed mass", details)

    def test_dashboard_never_mixes_modes(self):
        from aimusic.evaluation import ui

        spec = _spec()
        singleton = {"status": "ok", "run_id": spec.run_id, "spec": spec.to_dict(), "failure": None,
                     "runtime": {}, "metrics": {"path": {}, "endpoint": {"status": "not_applicable"}}}
        ds = self._dataset(singleton, self._distribution_result())
        self.assertEqual(len(ui.endpoint_cells(ds)), 1)
        self.assertIn("singleton cell(s) are not shown", ui.render_endpoint_overview(ds))

    def test_config_panel_and_type_show_endpoint_mode(self):
        from aimusic.evaluation import ui

        meta = ui.load_config_metadata("m2_distribution_method_a.json")
        self.assertEqual(meta.benchmark_type, ui.TYPE_EVALUATION)
        self.assertEqual(meta.endpoint_mode, "distribution")
        self.assertIn("method_a_distribution_endpoints", ui.render_config_panel(meta))
        m1 = ui.load_config_metadata("m1_smoke.json")
        self.assertEqual(m1.endpoint_mode, "singleton")
        self.assertIn("singleton", ui.render_config_panel(m1))


class TestEndToEndDeterminism(unittest.TestCase):
    """Two full 8-beat runs with identical config, seed and RNG key."""

    @classmethod
    def setUpClass(cls):
        spec = _spec(endpoint_mode="distribution", calibration_samples=40)
        cls.first, cls.second = evaluate_run(spec), evaluate_run(spec)
        cls.singleton = evaluate_run(_spec())

    def test_runs_succeed_in_distribution_mode(self):
        self.assertEqual(self.first["status"], "ok", self.first["failure"])
        self.assertEqual(self.first["spec"]["generation_method"], "method_a_distribution_endpoints")
        m = self.first["metrics"]["endpoint"]
        self.assertEqual(m["endpoint_mode"], "distribution")
        self.assertGreater(m["endpoint_support_size_start"], 1)  # not collapsed to singleton
        self.assertGreater(m["endpoint_support_size_terminal"], 1)
        self.assertTrue(m["start"]["residual_within_tolerance"] and m["terminal"]["residual_within_tolerance"])
        self.assertEqual(m["calibration_status"], "available")
        self.assertTrue(self.first["metrics"]["path"]["reaches_start"])
        self.assertTrue(self.first["metrics"]["path"]["reaches_end"])
        json.dumps(self.first)

    def test_endpoint_graph_and_path_are_identical(self):
        a, b = self.first, self.second
        self.assertEqual(a["metrics"]["endpoint"], b["metrics"]["endpoint"])  # distributions + calibration
        self.assertEqual(a["metrics"]["graph"], b["metrics"]["graph"])
        self.assertEqual(a["artifacts"]["path_sha256"], b["artifacts"]["path_sha256"])
        self.assertEqual(a["metrics"]["sb"]["iterations"], b["metrics"]["sb"]["iterations"])

    def test_map_path_is_reproducible(self):
        spec = _spec(endpoint_mode="distribution", path_mode="map", horizon=4)
        a, b = evaluate_run(spec), evaluate_run(spec)
        self.assertEqual(a["status"], "ok", a["failure"])
        self.assertEqual(a["artifacts"]["path_sha256"], b["artifacts"]["path_sha256"])
        self.assertEqual(a["metrics"]["endpoint"]["calibration_status"], "unavailable_no_repeated_samples")

    def test_singleton_run_still_works_and_is_labelled(self):
        res = self.singleton
        self.assertEqual(res["status"], "ok", res["failure"])
        self.assertEqual(res["spec"]["endpoint_mode"], "singleton")
        self.assertEqual(res["spec"]["generation_method"], "method_a_singleton_endpoints")
        self.assertEqual(res["metrics"]["endpoint"]["status"], "not_applicable")
        self.assertEqual(res["metrics"]["endpoint"]["endpoint_support_size_start"], 1)
        self.assertEqual(res["metrics"]["endpoint"]["endpoint_support_size_terminal"], 1)
        self.assertTrue(res["metrics"]["path"]["reaches_start"])


if __name__ == "__main__":
    unittest.main()
