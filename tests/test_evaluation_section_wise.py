"""MUS-16 section-wise evaluation tests (MUS-28 follow-up, REQ-18).

Fast unit tests cover configuration, the pure join/failure/aggregation logic and the report.  A small
number of real ``evaluate_run`` calls (a tiny 3-section song) are shared through ``setUpClass`` so the
planner is exercised end to end without making the suite slow.
"""
import copy
import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest import mock

from aimusic.evaluation import section_wise as sw
from aimusic.evaluation.config import expand_matrix, load_config, parse_config, select_specs
from aimusic.evaluation.report import render_report
from aimusic.evaluation.runner import evaluate_run
from aimusic.evaluation.schema import RunSpec, new_result
from aimusic.evaluation.stats import MIN_PAIRS_FOR_CI, aggregate, flatten_metrics
from aimusic.planning.graph import _keep_every_start_lineage
from aimusic.planning.plans import PlanningSection, SectionPlanningError, SectionWisePlanResult
from aimusic.planning.stitching import SectionJoinDiagnostics, StitchCost

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config" / "benchmarks"
NAMES = ["intro", "theme", "outro"]


def _raw(**overrides: Any) -> Dict[str, Any]:
    raw: Dict[str, Any] = {
        "schema_version": "mus28-config/1", "name": "t", "config_version": "1.0.0",
        "horizons": [12], "seeds": [0, 1], "edos": [12], "path_modes": ["sample"], "conditions": ["full"],
        "graph_grid": [{"k_max": 64, "d_max": 8, "proposal_budget": 256}],
        "prior_guided_proposals": False,
        "sb": {"max_iterations": 200, "tolerance": 1e-6, "temperature": 1.0},
        "timeout_s": 60,
        "endpoint_mode": "distribution",
        "sectioning": ["single_pass", "section_wise"],
        "section_names": NAMES,
        "style": {"allowed_meters": ["4/4"]},
    }
    raw.update(overrides)
    return raw


def _spec(**kw: Any) -> RunSpec:
    base: Dict[str, Any] = dict(
        condition="full", horizon=12, seed=0, edo=12, path_mode="sample",
        prior_mode="placeholder_structured", lambda_data=1.0, lambda_gttm=1.0, bridge_mode="sb",
        k_max=64, d_max=8, proposal_budget=256, prior_guided_proposals=False,
        max_iterations=200, tolerance=1e-6, temperature=1.0,
        endpoint_mode="distribution", section_names=tuple(NAMES), allowed_meters=("4/4",),
        sectioning="section_wise",
    )
    base.update(kw)
    return RunSpec(**base)


def _sections(n: int = 3, beats: int = 4) -> List[PlanningSection]:
    return [PlanningSection((NAMES + ['return'])[i], i * beats, (i + 1) * beats, 2) for i in range(n)]


class TestStartLineagePruning(unittest.TestCase):
    """``k_max`` pruning must not strand a start state of a shared boundary."""

    def test_missing_lineage_swaps_in_and_evicts_a_redundant_state(self) -> None:
        ranked = ["a", "b", "c", "d"]  # best first
        lineage = {"a": frozenset({0}), "b": frozenset({0}), "c": frozenset({0}), "d": frozenset({1})}
        self.assertEqual(_keep_every_start_lineage(ranked, 2, lineage), ("a", "d"))

    def test_covered_lineages_keep_the_plain_top_k(self) -> None:
        ranked = ["a", "b", "c"]
        lineage = {"a": frozenset({0}), "b": frozenset({1}), "c": frozenset({0, 1})}
        self.assertEqual(_keep_every_start_lineage(ranked, 2, lineage), ("a", "b"))

    def test_never_evicts_the_only_cover_of_a_lineage(self) -> None:
        ranked = ["a", "b", "c", "d"]
        lineage = {"a": frozenset({0}), "b": frozenset({1}), "c": frozenset({2}), "d": frozenset({3})}
        kept = _keep_every_start_lineage(ranked, 4, lineage)
        self.assertEqual(sorted(kept), ["a", "b", "c", "d"])


class TestSharedBoundaryRegressions(unittest.TestCase):
    """m2_section_wise_sb 32-beat failures: the next section dropped part of a shared boundary.

    Before the fix, ``k_max`` pruning removed every descendant of one start-boundary state, so MUS-15
    reported it as unreachable and the section no longer started from the authored shared distribution
    (seed 1: ``bridge``; seed 0: ``return``).
    """

    cases = {1: ("bridge", 2), 0: ("return", 3)}
    runs: Dict[int, Dict[str, Any]] = {}
    plans: Dict[int, SectionWisePlanResult] = {}

    @classmethod
    def setUpClass(cls) -> None:
        from aimusic.planning import plans
        real = plans.run_method_a
        for seed in cls.cases:
            specs = select_specs(
                expand_matrix(load_config(CONFIG_DIR / "m2_section_wise_sb.json")),
                horizons=[32], seeds=[seed], sectioning=["section_wise"],
            )
            assert len(specs) == 1
            captured: List[Any] = []

            def spy(*args: Any, **kwargs: Any) -> Any:
                out = real(*args, **kwargs)
                captured.append(out[0])
                return out

            with mock.patch.object(plans, "run_method_a", spy):
                cls.runs[seed] = evaluate_run(specs[0])
            cls.plans[seed] = captured[0]

    def test_runs_complete_successfully(self) -> None:
        for seed, (name, _) in self.cases.items():
            with self.subTest(seed=seed, section=name):
                run = self.runs[seed]
                self.assertEqual(run["status"], "ok", run.get("failure"))
                self.assertIsNone(run.get("failure"))
                self.assertEqual(run["section_count"], 4)
                self.assertEqual(set(run["per_section_status"]), {sw.STATUS_SECTION_OK})

    def test_shared_boundaries_are_preserved_exactly(self) -> None:
        for seed, (name, index) in self.cases.items():
            plan = self.plans[seed]
            with self.subTest(seed=seed, section=name):
                self.assertEqual(plan.section_results[index].section.name, name)
                for i in range(1, len(plan.section_results)):
                    left, right = plan.section_results[i - 1], plan.section_results[i]
                    self.assertFalse(left.end_mass.modified)
                    self.assertFalse(right.start_mass.modified)
                    self.assertEqual(left.end_mass.effective.layer.states,
                                     right.start_mass.effective.layer.states)
                    self.assertEqual(left.end_mass.effective.probabilities,
                                     right.start_mass.effective.probabilities)
                    join = plan.joins[i - 1]
                    # The authored request itself, not a trimmed or renormalized variant.
                    self.assertEqual(right.start_mass.requested.layer.states, join.boundary_states)
                    self.assertIn(join.shared_state, join.boundary_states)
                    self.assertEqual(right.start_mass.requested.probabilities, join.boundary_probabilities)
                    self.assertEqual(right.start_mass.effective.probabilities, join.boundary_probabilities)
                    self.assertEqual(right.start_mass.removed, ())
                    self.assertEqual(right.start_mass.trims, ())


class TestSectioningConfig(unittest.TestCase):
    def test_single_pass_parses(self) -> None:
        for value in ("single_pass", ["single_pass"]):
            cfg = parse_config({k: v for k, v in _raw(sectioning=value).items() if k not in ("section_names", "style")})
            self.assertEqual(cfg.sectioning, ("single_pass",))
            specs = expand_matrix(cfg)
            self.assertTrue(all(s.sectioning == "single_pass" and s.section_names == () for s in specs))

    def test_section_wise_parses_and_expands_both_modes(self) -> None:
        cfg = parse_config(_raw())
        self.assertEqual(cfg.sectioning, ("single_pass", "section_wise"))
        self.assertEqual(cfg.section_names, tuple(NAMES))
        specs = expand_matrix(cfg)
        self.assertEqual(len(specs), 4)  # 2 modes x 2 seeds
        self.assertEqual({s.sectioning for s in specs}, {"single_pass", "section_wise"})
        self.assertEqual(len({s.run_id for s in specs}), 4)
        only = select_specs(specs, sectioning=["section_wise"])
        self.assertEqual({s.sectioning for s in only}, {"section_wise"})

    def test_shipped_section_wise_config_loads(self) -> None:
        cfg = load_config(CONFIG_DIR / "m2_section_wise_sb.json")
        self.assertEqual(cfg.sectioning, ("single_pass", "section_wise"))
        self.assertEqual(cfg.section_names, ("intro", "theme", "bridge", "return"))
        self.assertEqual(cfg.endpoint_mode, "distribution")
        self.assertEqual(len(expand_matrix(cfg)), 12)

    def test_existing_m1_configs_are_unchanged(self) -> None:
        for path in sorted(CONFIG_DIR.glob("m1_*.json")) + [CONFIG_DIR / "m2_distribution_method_a.json"]:
            with self.subTest(config=path.name):
                raw = json.loads(path.read_text(encoding="utf-8"))
                self.assertNotIn("sectioning", raw)
                cfg = load_config(path)
                self.assertEqual(cfg.sectioning, ("single_pass",))
                self.assertEqual(cfg.section_names, ())
                for spec in expand_matrix(cfg):
                    self.assertEqual(spec.sectioning, "single_pass")
                    self.assertEqual(spec.section_names, ())
                    self.assertIsNone(spec.stitch_max_cost)
                    self.assertNotIn("sec=", spec.group_key)

    def test_m1_run_ids_are_unchanged(self) -> None:
        # Recorded from the shipped Milestone 1 benchmark output (outputs/benchmarks/mus28-rerun-64-128).
        ids = {s.run_id for s in expand_matrix(load_config(CONFIG_DIR / "m1_64_128.json"))}
        for golden in ("full-h64-e12-s1-sample-92edbacd", "no_sb-h64-e19-s0-sample-56832145",
                       "full-h128-e19-s3-sample-3bfbde32"):
            self.assertIn(golden, ids)

    def test_invalid_sectioning_values_fail_clearly(self) -> None:
        for bad, text in (("frobnicate", "unknown sectioning"), (["single_pass", "nope"], "unknown sectioning"),
                          ([], "sectioning must be"), (7, "sectioning must be"),
                          (["single_pass", "single_pass"], "must not repeat")):
            with self.subTest(value=bad):
                with self.assertRaisesRegex(ValueError, text):
                    parse_config(_raw(sectioning=bad))
        with self.assertRaisesRegex(ValueError, "sectioning must be one of"):
            _spec(sectioning="monolithic")

    def test_section_names_and_horizons_are_validated(self) -> None:
        cases: Any = (
            (dict(section_names=["a", ""]), "non-empty"),
            (dict(section_names=["a", "a"]), "unique"),
            (dict(section_names="intro"), "list of non-empty strings"),
            (dict(section_names=["only"]), "at least two"),
            (dict(section_names=[]), "at least two"),
            (dict(horizons=[2]), "shorter than"),
            (dict(conditions=["full", "no_sb"]), "no_sb"),
            (dict(calibration_samples=40), "calibration_samples"),
        )
        for overrides, text in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, text):
                    parse_config(_raw(**overrides))
        # Names are fine without section_wise (the plan still scores a single-pass tension target).
        parse_config({k: v for k, v in _raw(sectioning="single_pass", section_names=["x", "y"]).items() if k != "style"})
        with self.assertRaisesRegex(ValueError, "at least two"):
            _spec(section_names=("only",))
        with self.assertRaisesRegex(ValueError, "shorter than"):
            _spec(horizon=2)

    def test_stitching_and_style_blocks_are_validated(self) -> None:
        cfg = parse_config(_raw(stitching={"max_cost": 3.5}))
        self.assertEqual(cfg.stitch_max_cost, 3.5)
        specs = expand_matrix(cfg)
        self.assertTrue(all((s.stitch_max_cost == 3.5) == (s.sectioning == "section_wise") for s in specs))
        for overrides, text in (
            (dict(stitching={"max_cost": -1}), "max_cost"),
            (dict(stitching={"bogus": 1}), "only supports"),
            (dict(sectioning="single_pass", section_names=[], stitching={"max_cost": 1}), "only applies"),
            (dict(style={"nope": []}), "style only supports"),
            (dict(style={"allowed_meters": "4/4"}), "list of non-empty strings"),
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, text):
                    parse_config(_raw(**overrides))

    def test_group_key_and_run_id_keep_modes_apart(self) -> None:
        sp, wise = _spec(sectioning="single_pass"), _spec()
        self.assertNotIn("sec=", sp.group_key)
        self.assertTrue(wise.group_key.endswith("|sec=section_wise"))
        self.assertNotEqual(sp.run_id, wise.run_id)
        self.assertEqual(RunSpec.from_dict(json.loads(json.dumps(wise.to_dict()))), wise)  # JSON lists -> tuples


class TestMatchedConditions(unittest.TestCase):
    def test_pairs_differ_only_in_sectioning_mode(self) -> None:
        specs = expand_matrix(parse_config(_raw(seeds=[0, 1, 2], horizons=[12, 15])))
        by_key: Dict[Any, Dict[str, RunSpec]] = {}
        for spec in specs:
            by_key.setdefault((spec.horizon, spec.seed), {})[spec.sectioning] = spec
        self.assertEqual(len(by_key), 6)
        for pair in by_key.values():
            a, b = pair["single_pass"].to_dict(), pair["section_wise"].to_dict()
            differing = {k for k in a if a[k] != b[k]}
            self.assertEqual(differing, {"sectioning"})  # same seed, horizon, EDO, endpoints, graph, prior, style, plan


def _join(incoming: float, outgoing: float, tolerance: float = 3.0) -> Any:
    def cost(total: float) -> StitchCost:
        return StitchCost(0.0, 0.5, 0.0, 1.0, 1.0, 0.0, 1 / 3, total)

    state = SimpleNamespace(to_dict=lambda: {"chord_id": 1})
    return SectionJoinDiagnostics(
        "intro", "theme", 3, state, cost(incoming), cost(outgoing), tolerance, 2, (), (0.75, 0.25),  # type: ignore[arg-type]
    )


class TestJoinMetrics(unittest.TestCase):
    def test_join_record_continuity_and_cost(self) -> None:
        record = sw.join_record(0, _join(2.0, 2.5))
        self.assertEqual(record["stitching_cost"], 2.5)
        self.assertEqual(record["stitching_cost_incoming"], 2.0)
        self.assertAlmostEqual(record["chord_distance_jump"], 0.5)
        self.assertEqual(record["groove_continuity"], 0.0)
        self.assertEqual(record["meter_continuity"], 1.0)
        self.assertAlmostEqual(record["boundary_continuity"], 2 / 3)
        self.assertTrue(record["within_tolerance"])
        self.assertAlmostEqual(record["tolerance_margin"], 0.5)
        self.assertGreater(record["boundary_entropy"], 0.0)

    def test_tolerance_is_respected_and_violation_is_flagged(self) -> None:
        ok = sw.join_record(0, _join(3.0, 3.0))  # exactly at the limit is inside
        self.assertTrue(ok["within_tolerance"])
        bad = sw.join_record(1, _join(1.0, 3.5))  # outgoing edge exceeds the limit
        self.assertFalse(bad["within_tolerance"])
        self.assertLess(bad["tolerance_margin"], 0.0)
        summary = sw.join_summary([ok, bad])
        self.assertEqual(summary["within_tolerance_count"], 1)
        self.assertFalse(summary["all_within_tolerance"])
        self.assertEqual(summary["stitching_cost_worst"], 3.5)
        self.assertEqual(summary["groove_continuity_worst"], 0.0)
        status = sw.success_status({"summary": summary})
        # sections solved and path assembled, but the join is NOT valid: the levels stay distinct.
        self.assertEqual((status["sections_solved"], status["joins_valid"], status["path_assembled"]),
                         (True, False, True))

    def test_shared_endpoint_is_not_duplicated(self) -> None:
        base = _join(1.0, 1.0)
        shared = SimpleNamespace(to_dict=lambda: {})
        join = SimpleNamespace(
            left_section=base.left_section, right_section=base.right_section, time_index=1,
            shared_state=shared, incoming=base.incoming, outgoing=base.outgoing, tolerance=3.0,
            boundary_support_size=2, boundary_probabilities=(0.5, 0.5),
        )
        left, right = ("A", shared), (shared, "C")
        good = sw.join_record(0, join, left_path=left, right_path=right, whole_path=("A", shared, "C"))
        self.assertTrue(good["shared_endpoint_consistent"])
        # A naive concatenation emits the shared state twice, so it no longer sits at the join's beat.
        bad = sw.join_record(0, join, left_path=left, right_path=right, whole_path=("A", "A", shared, shared, "C"))
        self.assertFalse(bad["shared_endpoint_consistent"])

    def test_empty_summary_is_explicit(self) -> None:
        self.assertEqual(sw.join_summary([]), {"join_count": 0})


class TestSectionFailureState(unittest.TestCase):
    def _state(self, stage: str, index: int, cause: Exception = ValueError("x")) -> Dict[str, Any]:
        sections = _sections(4, 2)
        exc = SectionPlanningError.__new__(SectionPlanningError)
        try:
            raise SectionPlanningError(index, sections[index], stage, cause) from cause
        except SectionPlanningError as caught:
            exc = caught
        return sw.section_failure_state(exc, sections)

    def test_each_stage_maps_to_class_and_statuses(self) -> None:
        cases = {
            "boundary planning": (sw.CLASS_BOUNDARY, ["not_reached"] * 4, (False, None, False)),
            "graph construction": (sw.CLASS_SECTION_SOLVE, ["not_reached", "failed", "not_reached", "not_reached"], (False, None, False)),
            "endpoint bridge solve": (sw.CLASS_SECTION_SOLVE, ["ok", "failed", "not_reached", "not_reached"], (False, None, False)),
            "shared boundary validation": (sw.CLASS_JOIN, ["ok"] * 4, (True, False, False)),
            "join validation": (sw.CLASS_JOIN, ["ok"] * 4, (True, False, False)),
            "joint path extraction": (sw.CLASS_PATH, ["ok"] * 4, (True, None, False)),
            "path extraction": (sw.CLASS_PATH, ["ok", "failed", "not_reached", "not_reached"], (True, None, False)),
        }
        for stage, (klass, statuses, levels) in cases.items():
            with self.subTest(stage=stage):
                state = self._state(stage, 1)
                self.assertEqual(state["failure_class"], klass)
                self.assertEqual(state["per_section_status"], statuses)
                status = state["status"]
                self.assertEqual((status["sections_solved"], status["joins_valid"], status["path_assembled"]), levels)
                self.assertEqual(status["failed_stage"], stage)
                self.assertEqual(state["cause_kind"], "ValueError")

    def test_non_planning_error_is_not_attributed_to_a_section(self) -> None:
        state = sw.section_failure_state(RuntimeError("boom"), _sections(3))
        self.assertEqual(state["failure_class"], sw.CLASS_EVALUATION)
        self.assertEqual(state["per_section_status"], ["unknown"] * 3)
        self.assertIsNone(state["section_name"])
        self.assertIsNone(state["status"]["sections_solved"])


class TestSectionWiseEvaluation(unittest.TestCase):
    """Real (small) planner runs, evaluated and repeated once to check determinism."""

    spec: RunSpec
    sw: Dict[str, Any]
    sw_again: Dict[str, Any]
    mono: Dict[str, Any]

    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = _spec()
        cls.sw = evaluate_run(cls.spec)
        cls.sw_again = evaluate_run(cls.spec)
        cls.mono = evaluate_run(_spec(sectioning="single_pass"))

    def test_section_wise_run_is_evaluated_not_rejected(self) -> None:
        self.assertEqual(self.sw["status"], "ok", msg=self.sw["failure"])
        self.assertIsNone(self.sw["failure"])
        self.assertEqual(self.sw["sectioning_mode"], "section_wise")
        self.assertEqual(self.sw["spec"]["sectioning"], "section_wise")

    def test_section_structure_is_recorded(self) -> None:
        r = self.sw
        self.assertEqual(r["section_names"], NAMES)
        self.assertEqual(r["section_count"], 3)
        self.assertEqual(r["section_horizons"], [4, 4, 4])
        self.assertEqual(sum(r["section_horizons"]), self.spec.horizon)
        for key in ("per_section_results", "per_section_status", "per_section_runtime", "per_section_graph_metrics",
                    "per_section_sb_metrics", "per_section_endpoint_metrics", "per_section_path_metrics"):
            self.assertEqual(len(r[key]), 3, msg=key)
        self.assertEqual(r["per_section_status"], ["ok"] * 3)
        self.assertEqual((r["section_wise_status"]["sections_solved"], r["section_wise_status"]["joins_valid"],
                          r["section_wise_status"]["path_assembled"]), (True, True, True))

    def test_per_section_diagnostics_are_serialized(self) -> None:
        r = json.loads(json.dumps(self.sw))  # the full result round-trips through JSON
        beats = 0
        for i, rec in enumerate(r["per_section_results"]):
            self.assertEqual((rec["index"], rec["name"]), (i, NAMES[i]))
            self.assertEqual((rec["start_beat"], rec["end_beat"], rec["n_beats"]), (beats, beats + 4, 4))
            beats += 4
            self.assertIn("mean_absolute_error", rec["tension"])
            self.assertIn("shape_correlation", rec["tension"])
            self.assertGreater(r["per_section_runtime"][i]["runtime_s"], 0.0)
            graph = r["per_section_graph_metrics"][i]
            self.assertEqual(graph["layer_count"], 5)  # 4 beats -> 5 layers
            for k in ("layer_width_mean", "layer_width_max", "proposed_count", "legal_count", "unique_count", "retained_count"):
                self.assertIn(k, graph)
            sb = r["per_section_sb_metrics"][i]
            self.assertTrue(sb["converged"])
            self.assertGreater(sb["iterations"], 0)
            self.assertIn("effective_entropy", sb)
            ep = r["per_section_endpoint_metrics"][i]
            for side in ("start", "terminal"):
                for k in ("requested_mass", "effective_mass", "realized_mass"):
                    self.assertAlmostEqual(ep[side][k], 1.0, places=5)
            self.assertEqual(ep["start"]["boundary_kind"], "outer_distribution" if i == 0 else "shared_boundary")
            self.assertEqual(ep["terminal"]["boundary_kind"], "outer_distribution" if i == 2 else "shared_boundary")
            path = r["per_section_path_metrics"][i]
            self.assertTrue(path["length_ok"])
            for k in ("chord_change_rate", "key_change_rate", "role_change_rate", "groove_change_rate"):
                self.assertIn(k, path)

    def test_whole_song_diagnostics_are_serialized(self) -> None:
        r = json.loads(json.dumps(self.sw))
        m = r["metrics"]
        self.assertTrue(m["path"]["length_ok"])
        self.assertEqual(m["path"]["path_length"], self.spec.horizon + 1)
        self.assertEqual(m["graph"]["layer_count"], self.spec.horizon + 1)  # shared boundary layers counted once
        self.assertEqual(m["sb"]["iterations"], sum(s["iterations"] for s in r["per_section_sb_metrics"]))
        self.assertTrue(m["sb"]["converged"])
        self.assertEqual(m["sb"]["sections_converged"], 3)
        self.assertEqual(m["endpoint"]["scope"], "whole_song_outer_endpoints")
        self.assertAlmostEqual(m["endpoint"]["realized_start_mass"], 1.0, places=5)
        self.assertIn("mean_absolute_error", m["tension"])
        self.assertEqual(m["tension"]["mean_absolute_error"], m["tension_section_plan"]["mean_absolute_error"])
        self.assertEqual(set(m["tension_section_plan"]["per_section"]), set(NAMES))
        self.assertIn("mean_gttm_score", m["gttm"])
        for k in ("wall_s", "peak_rss_mb", "sections_graph_s", "sections_solve_s", "section_wise_overhead_s"):
            self.assertIn(k, r["runtime"])
        self.assertEqual(r["artifacts"]["path_sha256"], self.sw["artifacts"]["path_sha256"])

    def test_joins_are_measured_and_consistent(self) -> None:
        joins = self.sw["join_metrics"]["joins"]
        self.assertEqual(len(joins), 2)
        self.assertEqual([j["time_index"] for j in joins], [4, 8])
        for j in joins:
            for k in ("key_distance_jump", "chord_distance_jump", "meter_continuity", "groove_continuity",
                      "role_continuity", "head_continuity", "boundary_continuity", "stitching_cost"):
                self.assertTrue(0.0 <= j[k] <= 7.0, msg=k)
            self.assertAlmostEqual(j["stitching_cost"], max(j["stitching_cost_incoming"], j["stitching_cost_outgoing"]))
            self.assertTrue(j["within_tolerance"])
            self.assertEqual(j["tolerance"], 7.0)
            self.assertTrue(j["shared_endpoint_consistent"])  # no duplicated shared endpoint
            self.assertGreater(j["boundary_support_size"], 1)
        summary = self.sw["join_metrics"]["summary"]
        self.assertEqual(summary["join_count"], 2)
        self.assertTrue(summary["all_within_tolerance"])
        self.assertTrue(summary["all_shared_endpoints_consistent"])

    def test_monolithic_run_is_unchanged_in_shape(self) -> None:
        r = self.mono
        self.assertEqual(r["status"], "ok", msg=r["failure"])
        self.assertEqual(r["sectioning_mode"], "single_pass")
        self.assertEqual((r["section_count"], r["section_horizons"]), (1, [self.spec.horizon]))
        for key in ("per_section_results", "join_metrics", "section_wise_status", "per_section_runtime"):
            self.assertIsNone(r[key], msg=key)  # explicit "not applicable", never fake zeros
        self.assertIn("mean_absolute_error", r["metrics"]["tension"])
        # It is also scored against the authored section plan, so the two modes share one target.
        self.assertEqual(set(r["metrics"]["tension_section_plan"]["per_section"]), set(NAMES))
        self.assertEqual(flatten_metrics(self.mono).keys() & {k for k in flatten_metrics(self.mono) if k.startswith(("section.", "join."))}, set())

    def test_fixed_seed_runs_are_deterministic(self) -> None:
        a, b = self.sw, self.sw_again
        self.assertEqual(a["artifacts"]["path_sha256"], b["artifacts"]["path_sha256"])
        self.assertEqual([s["path_sha256"] for s in a["per_section_results"]],
                         [s["path_sha256"] for s in b["per_section_results"]])
        self.assertEqual(a["join_metrics"]["joins"], b["join_metrics"]["joins"])
        for key in ("graph", "sb", "path", "tension", "tension_section_plan", "gttm"):
            self.assertEqual(a["metrics"][key], b["metrics"][key], msg=key)
        self.assertEqual(a["per_section_sb_metrics"], b["per_section_sb_metrics"])

    def test_flattened_metrics_are_per_section_and_per_join(self) -> None:
        flat = flatten_metrics(self.sw)
        self.assertIn("section.intro.runtime.runtime_s", flat)
        self.assertIn("section.theme.sb.iterations", flat)
        self.assertIn("section.outro.tension.mean_absolute_error", flat)
        self.assertIn("section.theme.endpoint.start.realized_mass", flat)
        self.assertIn("join.stitching_cost_mean", flat)
        self.assertIn("join.1.stitching_cost", flat)
        self.assertIn("tension_section_plan.per_section.theme.mean_absolute_error", flat)


class TestStructuredFailures(unittest.TestCase):
    def _fail_with(self, stage: str, index: int, cause: Exception) -> Dict[str, Any]:
        section = _sections()[index]

        def boom(*args: Any, **kwargs: Any) -> Any:
            raise SectionPlanningError(index, section, stage, cause) from cause

        with mock.patch("aimusic.planning.plans.run_method_a", side_effect=boom):
            return evaluate_run(_spec())

    def test_join_failure_is_a_structured_result(self) -> None:
        r = self._fail_with("join validation", 1, ValueError("Selected join exceeds stitching tolerance."))
        self.assertEqual(r["status"], "failed")
        self.assertEqual(r["failure"]["kind"], "join_failed")
        self.assertEqual(r["failure"]["failure_class"], "join")
        self.assertEqual((r["failure"]["section_index"], r["failure"]["section_name"]), (1, "theme"))
        self.assertEqual(r["failure"]["planner_stage"], "join validation")
        self.assertIn("stitching tolerance", r["failure"]["message"])
        # Sections solved, join invalid, path not assembled: three distinct answers.
        st = r["section_wise_status"]
        self.assertEqual((st["sections_solved"], st["joins_valid"], st["path_assembled"]), (True, False, False))
        self.assertEqual(r["per_section_status"], ["ok", "ok", "ok"])
        self.assertEqual(r["section_names"], NAMES)
        self.assertEqual(r["section_horizons"], [4, 4, 4])
        self.assertIsNone(r["join_metrics"])  # no join metrics are invented for a failed run
        json.dumps(r)

    def test_section_failure_is_a_structured_result(self) -> None:
        r = self._fail_with("endpoint bridge solve", 1, RuntimeError("did not converge"))
        self.assertEqual(r["failure"]["kind"], "section_solve_failed")
        self.assertEqual(r["per_section_status"], ["ok", "failed", "not_reached"])
        self.assertEqual(r["per_section_results"][1]["failure"]["stage"], "endpoint bridge solve")
        self.assertIsNone(r["per_section_graph_metrics"])
        st = r["section_wise_status"]
        self.assertEqual((st["sections_solved"], st["joins_valid"], st["path_assembled"]), (False, None, False))

    def test_endpoint_infeasibility_inside_a_section_keeps_its_endpoint_class(self) -> None:
        from aimusic.planning.endpoint_feasibility import EndpointMassInfeasibleError

        r = self._fail_with("endpoint bridge solve", 0, EndpointMassInfeasibleError("marginals infeasible"))
        self.assertEqual(r["failure"]["failure_class"], "section_solve")
        endpoint = r["metrics"]["endpoint"]
        self.assertEqual(endpoint["accounting_status"], "strict_failure")
        self.assertEqual(endpoint["section_name"], "intro")

    def test_real_planner_rejection_becomes_a_structured_section_failure(self) -> None:
        # max_cost = 0 forbids every boundary transition: the planner cannot build a section graph.
        r = evaluate_run(_spec(stitch_max_cost=0.0))
        self.assertEqual(r["status"], "failed")
        self.assertIn(r["failure"]["failure_class"], ("section_solve", "join"))
        self.assertEqual(r["failure"]["kind"], sw.failure_kind(r["failure"]["failure_class"]))
        self.assertIn("section_name", r["failure"])
        self.assertEqual(r["section_count"], 3)
        self.assertEqual(len(r["per_section_status"]), 3)
        self.assertFalse(r["section_wise_status"]["path_assembled"])

    def test_planner_mode_mismatch_is_refused(self) -> None:
        # A section-wise request must never be scored as a monolithic plan (or vice versa).
        with mock.patch("aimusic.planning.plans.run_method_a", return_value=(SimpleNamespace(), None)):
            r = evaluate_run(_spec())
        self.assertEqual(r["status"], "failed")
        self.assertIn("refusing to evaluate under a different mode", r["failure"]["message"])


def _fake(spec: RunSpec, *, wall: float, mae: float, ok: bool = True, joins: bool = True) -> Dict[str, Any]:
    r = new_result(spec)
    r["runtime"] = {"wall_s": wall, "peak_rss_mb": 100.0 + wall, "baseline_rss_mb": 60.0}
    if not ok:
        r["status"] = "failed"
        r["failure"] = {"kind": "section_solve_failed", "stage": "generate", "message": "boom",
                        "failure_class": "section_solve", "section_name": "theme", "planner_stage": "endpoint bridge solve"}
        return r
    r["status"] = "ok"
    r["section_names"], r["section_count"], r["section_horizons"] = NAMES, 3, [4, 4, 4]
    r["metrics"]["tension_section_plan"] = {"mean_absolute_error": mae, "per_section": {n: {"mean_absolute_error": mae} for n in NAMES}}
    r["metrics"]["tension"] = {"mean_absolute_error": mae}
    r["metrics"]["path"] = {"distinct_state_fraction": 1.0}
    if spec.sectioning == "section_wise":
        r["per_section_status"] = ["ok"] * 3
        r["per_section_runtime"] = [{"runtime_s": wall / 3}] * 3
        r["per_section_graph_metrics"] = [{"layer_count": 4}] * 3
        r["per_section_sb_metrics"] = [{"iterations": 5, "converged": True}] * 3
        r["per_section_endpoint_metrics"] = [{"start": {"realized_mass": 1.0}, "terminal": {"realized_mass": 1.0}}] * 3
        r["per_section_path_metrics"] = [{"length_ok": True}] * 3
        r["per_section_results"] = [{"n_beats": 4, "start_beat": 4 * i, "end_beat": 4 * i + 4, "tension": {"mean_absolute_error": mae}} for i in range(3)]
        records = [sw.join_record(i, _join(2.0, 2.0)) for i in range(2)]
        r["join_metrics"] = {"joins": records, "summary": sw.join_summary(records)}
        r["section_wise_status"] = {"sections_solved": True, "joins_valid": joins, "path_assembled": True}
    return r


class TestComparisonAndReport(unittest.TestCase):
    def _results(self, seeds: int = 3, **kw: Any) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        walls_sp, walls_sw = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0], [8.0, 15.0, 31.0, 36.0, 45.0, 50.0]
        for seed in range(seeds):
            out.append(_fake(_spec(sectioning="single_pass", seed=seed), wall=walls_sp[seed], mae=0.30))
            out.append(_fake(_spec(seed=seed), wall=walls_sw[seed], mae=0.40, **kw))
        return out

    def test_aggregate_keeps_sectioning_modes_separate(self) -> None:
        agg = aggregate(self._results())
        modes = {c["sectioning"] for c in agg["cells"].values()}
        self.assertEqual(modes, {"single_pass", "section_wise"})
        self.assertEqual(len(agg["cells"]), 2)
        for key, cell in agg["cells"].items():
            self.assertEqual(cell["n_runs"], 3)
            self.assertEqual(key.endswith("|sec=section_wise"), cell["sectioning"] == "section_wise")
            has_section_metrics = any(m.startswith(("section.", "join.")) for m in cell["metrics"])
            self.assertEqual(has_section_metrics, cell["sectioning"] == "section_wise")
        # Ablation pairing must never cross sectioning modes.
        self.assertEqual(agg["paired_vs_control"], {})

    def test_paired_comparison_is_by_seed(self) -> None:
        agg = aggregate(self._results())
        (entry,) = agg["sectioning_comparison"].values()
        self.assertEqual(entry["difference"], "section_wise - single_pass")
        self.assertEqual(entry["status_pairs"]["both_ok"], 3)
        wall = entry["metrics"]["runtime.wall_s"]
        self.assertEqual(wall["seeds"], [0, 1, 2])
        self.assertAlmostEqual(wall["mean"], ((8 - 10) + (15 - 20) + (31 - 30)) / 3)
        self.assertAlmostEqual(entry["metrics"]["tension_section_plan.mean_absolute_error"]["mean"], 0.10)
        self.assertEqual(entry["ci_level"], "descriptive")  # 3 matched seeds are not enough for a paired CI

    def test_ci_level_needs_enough_matched_seeds(self) -> None:
        agg = aggregate(self._results(seeds=MIN_PAIRS_FOR_CI))
        (entry,) = agg["sectioning_comparison"].values()
        self.assertEqual((entry["n_pairs"], entry["ci_level"]), (MIN_PAIRS_FOR_CI, "paired_ci"))
        self.assertIsNotNone(entry["metrics"]["runtime.wall_s"]["ci95_low"])

    def test_pairing_counts_seeds_where_only_one_mode_succeeded(self) -> None:
        results = self._results()
        results[3] = _fake(_spec(seed=1), wall=0.0, mae=0.0, ok=False)  # section_wise seed 1 fails
        (entry,) = aggregate(results)["sectioning_comparison"].values()
        self.assertEqual(entry["status_pairs"], {"both_ok": 2, "only_single_pass_ok": 1,
                                                 "only_section_wise_ok": 0, "both_failed": 0})
        self.assertEqual(entry["metrics"]["runtime.wall_s"]["seeds"], [0, 2])  # the failed seed is not imputed

    def _report(self, results: List[Dict[str, Any]]) -> str:
        manifest = {"config_name": "t", "config_version": "1", "config_hash": "0" * 64, "created_with": {}}
        agg = aggregate(results)
        agg.update({"completed_runs": len(results), "expected_runs": len(results), "missing_run_ids": []})
        return render_report(manifest, results, agg)

    def test_report_has_section_wise_section(self) -> None:
        text = self._report(self._results())
        self.assertIn("## Section-wise SB evaluation", text)
        self.assertIn("Comparison type: descriptive", text)
        for heading in ("### Success and failure", "### Runtime and memory", "### Per-section summary",
                        "### Stitching and joins", "#### Join detail", "### Section-local failures",
                        "### Limitations of this comparison"):
            self.assertIn(heading, text)
        self.assertIn("intro -> theme", text)
        self.assertIn("Not a quality ranking", text)
        self.assertIn("full / 12b / 12-EDO / sample / distribution / section_wise", text)  # cells never merged

    def test_report_states_paired_ci_when_enough_seeds(self) -> None:
        text = self._report(self._results(seeds=MIN_PAIRS_FOR_CI))
        self.assertIn("Comparison type: paired confidence intervals", text)

    def test_report_lists_section_local_failures(self) -> None:
        results = self._results()
        results[3] = _fake(_spec(seed=1), wall=0.0, mae=0.0, ok=False)
        text = self._report(results)
        self.assertIn("section_solve", text)
        self.assertIn("endpoint bridge solve", text)
        self.assertIn("seeds [1]", text)

    def test_single_pass_only_report_says_unavailable(self) -> None:
        results = [r for r in self._results() if r["spec"]["sectioning"] == "single_pass"]
        text = self._report(results)
        self.assertIn("## Section-wise SB evaluation", text)
        self.assertIn("Section-wise metrics unavailable", text)
        self.assertIn("was **not evaluated**", text)

    def test_failed_join_stays_distinct_from_solved_sections(self) -> None:
        results = self._results()
        results[1] = _fake(_spec(seed=0), wall=8.0, mae=0.4, joins=False)
        text = self._report(results)
        self.assertIn("| 3/3 | 2/3 | 3/3 |", text)  # solved / joins valid / path assembled differ


class TestRealReport(unittest.TestCase):
    def test_real_results_render_section_tables(self) -> None:
        results = [evaluate_run(_spec(sectioning="single_pass")), evaluate_run(_spec())]
        manifest = {"config_name": "t", "config_version": "1", "config_hash": "0" * 64, "created_with": {}}
        agg = aggregate(results)
        agg.update({"completed_runs": 2, "expected_runs": 2, "missing_run_ids": []})
        text = render_report(manifest, results, copy.deepcopy(agg))
        self.assertIn("| intro |", text)
        self.assertIn("| theme |", text)
        self.assertIn("intro -> theme", text)
        self.assertIn("theme -> outro", text)


class TestSectionWiseDashboard(unittest.TestCase):
    """Dashboard data builders and renderers (no browser; plotly figures are only constructed)."""

    ui: Any
    us: Any
    ds: Any
    results: List[Dict[str, Any]]
    specs: List[Dict[str, Any]]

    @classmethod
    def setUpClass(cls) -> None:
        from aimusic.evaluation import ui, ui_sections

        cls.ui, cls.us = ui, ui_sections
        results = []
        for seed in range(3):
            results.append(_fake(_spec(sectioning="single_pass", seed=seed), wall=10.0 + seed, mae=0.30))
            results.append(_fake(_spec(seed=seed), wall=8.0 + seed, mae=0.40))
        results[3] = _fake(_spec(seed=1), wall=0.0, mae=0.0, ok=False)
        cls.results = results
        specs = [r["spec"] for r in results]
        cls.ds = ui.EvaluationDataset(
            out_dir=Path("."), manifest={"expected_runs": [{"run_id": r["run_id"], "spec": r["spec"]} for r in results]},
            results=results, aggregate=aggregate(results), aggregate_source="recomputed",
        )
        cls.specs = specs

    def test_single_pass_run_says_unavailable_without_fake_zeros(self) -> None:
        sp = next(r for r in self.results if r["spec"]["sectioning"] == "single_pass")
        for text in (self.us.render_section_wise_run(sp), self.us.render_section_details(sp, "intro"),
                     self.us.render_run_joins(sp), self.ui.render_run_details(sp)):
            self.assertIn(self.us.SECTION_WISE_UNAVAILABLE, text)
        self.assertEqual(self.us.section_choices(sp), [])
        legacy = {k: v for k, v in sp.items() if not k.startswith(("section", "per_section", "join"))}  # an M1 result
        self.assertIn(self.us.SECTION_WISE_UNAVAILABLE, self.us.render_section_wise_run(legacy))
        self.assertNotIn("| 0 |", self.us.render_section_wise_run(legacy))

    def test_modes_never_share_a_series(self) -> None:
        self.assertEqual(self.ui._sectioning_modes(self.ds), ["single_pass", "section_wise"])
        data = self.ui.grouped_metric_data(self.ds, "runtime.wall_s")
        self.assertEqual(set(data.series), {"full", "full (section_wise)"})
        self.assertAlmostEqual(data.series["full"][0].mean, 11.0)  # type: ignore[union-attr]
        self.assertAlmostEqual(data.series["full (section_wise)"][0].mean, 9.0)  # type: ignore[union-attr]  # failed seed 1 excluded: (8+10)/2
        status = self.ui.status_chart_data(self.ds)
        self.assertEqual(set(status.conditions), {"full", "full (section_wise)"})
        self.assertEqual(status.counts["failed"]["full (section_wise)"], 1)
        for make in (self.ui.make_runtime_chart, self.ui.make_status_chart, self.ui.make_scaling_chart,
                     self.ui.make_edo_chart, self.ui.make_tension_chart):
            make(self.ds)

    def test_comparison_chart_is_matched_and_descriptive(self) -> None:
        data = self.us.comparison_data(self.ds, "runtime.wall_s")
        self.assertEqual(len(data.categories), 1)
        self.assertEqual(data.ci_levels, ["descriptive"])
        self.assertTrue(any("Descriptive" in n for n in data.notes))
        self.assertIsNotNone(data.series["single_pass"][0])
        self.assertIsNotNone(data.series["section_wise"][0])
        for metric in self.us.COMPARISON_METRICS:
            self.us.make_comparison_chart(self.ds, metric)
        self.assertIn("descriptive", self.us.render_comparison_markdown(self.ds))

    def test_section_runtime_status_and_stitch_views(self) -> None:
        rt = self.us.section_runtime_data(self.ds)
        self.assertEqual(rt.categories, NAMES)
        status = self.us.section_status_data(self.ds)
        self.assertEqual(status.runs, 3)
        self.assertEqual(status.counts["ok"], [2, 2, 2])  # one failed run has no per-section status here
        self.assertEqual(status.levels["sections_solved"], 2)
        stitch = self.us.stitch_chart_data(self.ds)
        self.assertEqual(len(stitch.joins), 2)
        self.assertEqual(stitch.tolerance, [3.0, 3.0])  # the recorded join tolerance, drawn as a reference line
        table = self.us.render_join_table(self.ds)
        self.assertIn("intro → theme", table)
        self.assertIn("theme → outro", table)
        for make in (self.us.make_section_runtime_chart, self.us.make_section_status_chart, self.us.make_stitch_chart):
            make(self.ds)

    def test_section_selector_and_details(self) -> None:
        ok = next(r for r in self.results if r["status"] == "ok" and r["spec"]["sectioning"] == "section_wise")
        self.assertEqual(self.us.section_choices(ok), NAMES)
        text = self.us.render_section_details(ok, "theme")
        self.assertIn("Section `theme`", text)
        self.assertIn("tension", text.lower())
        self.us.section_tension_figure(ok, "theme")
        block = self.us.render_section_wise_run(ok)
        self.assertIn("all sections solved", block)
        self.assertIn("intro → theme", block)

    def test_failed_run_shows_levels_and_no_invented_metrics(self) -> None:
        with mock.patch("aimusic.planning.plans.run_method_a", side_effect=SectionPlanningError(
                1, _sections()[1], "join validation", ValueError("too far"))):
            failed = evaluate_run(_spec())
        text = self.us.render_section_wise_run(failed)
        self.assertIn("all joins valid | no", text)
        self.assertIn("whole-song path assembled | no", text)
        self.assertIn("No join metrics", text)
        self.assertIn("not available (the section did not complete)", self.us.render_section_details(failed, "theme"))

    def test_config_panel_shows_sectioning_mode(self) -> None:
        panel = self.ui.render_config_panel(self.ui.load_config_metadata("m2_section_wise_sb.json"))
        self.assertIn("Sectioning mode", panel)
        self.assertIn("single_pass, section_wise", panel)
        self.assertIn("intro, theme, bridge, return", panel)
        m1 = self.ui.render_config_panel(self.ui.load_config_metadata("m1_smoke.json"))
        self.assertIn("section-wise metrics are not produced", m1)

    def test_real_benchmark_output_loads_if_present(self) -> None:
        out = Path(__file__).resolve().parents[1] / "outputs" / "benchmarks" / "m2_section_wise_sb"
        if not (out / "manifest.json").exists():
            self.skipTest("benchmark output not generated")
        ds = self.ui.load_dataset_cached(out)
        self.assertTrue(self.us.section_wise_cells(ds))
        self.us.make_stitch_chart(ds)

    def test_page_builds(self) -> None:
        import gradio as gr

        with gr.Blocks():
            self.ui.build_evaluation_page()


if __name__ == "__main__":
    unittest.main()
