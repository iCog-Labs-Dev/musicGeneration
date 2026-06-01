import unittest

from unittest.mock import patch
from aimusic.core.config import PlanConfig, PlanMethod, SectioningStrategy, StyleConfig
from aimusic.planning.plans import (
    MethodARunConfig,
    MethodBRunConfig,
    build_section_plan,
    generate_end_endpoint_distribution,
    generate_method_a_endpoints,
    generate_method_b_endpoints,
    generate_start_endpoint_distribution,
    run_method_a,
    SectionSummary,
    LongFormDiagnostics,
    ComparisonRunSpec,
    ComparisonArmResult,
    ComparisonReport,
    run_comparison,
)


class TestMethodAEndpointPlanning(unittest.TestCase):
    def test_endpoint_generation_is_reproducible_and_aligned(self):
        run_config = MethodARunConfig(total_beats=4, seed=7)

        pi0 = generate_start_endpoint_distribution(run_config)
        piT = generate_end_endpoint_distribution(run_config)

        self.assertEqual(pi0.layer.time_index, 0)
        self.assertEqual(piT.layer.time_index, 4)
        self.assertAlmostEqual(sum(pi0.probabilities), 1.0)
        self.assertAlmostEqual(sum(piT.probabilities), 1.0)
        self.assertEqual(
            generate_start_endpoint_distribution(run_config),
            pi0,
        )
        self.assertEqual(
            generate_end_endpoint_distribution(run_config),
            piT,
        )

    def test_section_plan_supports_single_and_section_wise_modes(self):
        single_run = MethodARunConfig(total_beats=8)
        section_run = MethodARunConfig(
            total_beats=8,
            plan_config=PlanConfig(
                sectioning_strategy=SectioningStrategy.SECTION_WISE,
                section_names=("intro", "outro"),
            ),
        )

        single_sections = build_section_plan(single_run)
        section_wise_sections = build_section_plan(section_run)

        self.assertEqual(len(single_sections), 1)
        self.assertEqual(single_sections[0].start_time, 0)
        self.assertEqual(single_sections[0].end_time, 8)
        self.assertEqual([section.name for section in section_wise_sections], ["intro", "outro"])
        self.assertEqual(section_wise_sections[-1].end_time, 8)

    def test_section_wise_rejects_more_sections_than_beats(self):
        with self.assertRaises(ValueError):
            MethodARunConfig(
                total_beats=1,
                plan_config=PlanConfig(
                    sectioning_strategy=SectioningStrategy.SECTION_WISE,
                    section_names=("intro", "outro"),
                ),
            )


class TestMethodAOrchestration(unittest.TestCase):
    def test_generate_method_a_endpoints_returns_sections(self):
        run_config = MethodARunConfig(total_beats=4)

        endpoints = generate_method_a_endpoints(run_config)

        self.assertEqual(endpoints.pi0.layer.time_index, 0)
        self.assertEqual(endpoints.piT.layer.time_index, 4)
        self.assertEqual(len(endpoints.sections), 1)
        self.assertIn(endpoints.start_choice.state, endpoints.pi0.layer.states)
        self.assertIn(endpoints.end_choice.state, endpoints.piT.layer.states)

    def test_run_method_a_map_smoke(self):
        run_config = MethodARunConfig(
            total_beats=4,
            seed=11,
            style_config=StyleConfig(allowed_meters=("4/4",), groove_families=("straight",)),
        )

        result = run_method_a(run_config)

        self.assertEqual(len(result.path), run_config.total_beats + 1)
        self.assertEqual(result.path[0], result.diagnostics.chosen_start_state)
        self.assertEqual(result.path[-1], result.diagnostics.chosen_end_state)
        self.assertEqual(len(result.graph.layers[0].states), 1)
        self.assertEqual(len(result.graph.layers[-1].states), 1)
        self.assertEqual(result.endpoints.pi0.layer.time_index, 0)
        self.assertEqual(result.endpoints.start_choice.selection_mode, "argmax")
        self.assertEqual(result.diagnostics.endpoint_selection_mode, "argmax")
        self.assertGreater(result.diagnostics.chosen_start_probability, 0.0)
        self.assertGreater(result.diagnostics.chosen_end_probability, 0.0)
        self.assertEqual(result.diagnostics.path_mode, "map")
        self.assertIsNotNone(result.path_score)
        self.assertTrue(result.sb_solution.trace.converged)

    def test_run_method_a_sampling_is_seed_reproducible(self):
        run_config = MethodARunConfig(
            total_beats=4,
            seed=23,
            use_sampling=True,
            style_config=StyleConfig(allowed_meters=("4/4",), groove_families=("straight",)),
        )

        first = run_method_a(run_config)
        second = run_method_a(run_config)

        self.assertEqual(first.path, second.path)
        self.assertEqual(first.sampled_path, second.sampled_path)
        self.assertEqual(first.diagnostics.path_mode, "sample")
        self.assertEqual(first.diagnostics.endpoint_selection_mode, "sample")
        self.assertEqual(first.endpoints.start_choice, second.endpoints.start_choice)
        self.assertEqual(first.endpoints.end_choice, second.endpoints.end_choice)

    def test_run_method_a_section_wise_smoke(self):
        run_config = MethodARunConfig(
            total_beats=4,
            seed=5,
            style_config=StyleConfig(allowed_meters=("4/4",), groove_families=("straight",)),
            plan_config=PlanConfig(
                sectioning_strategy=SectioningStrategy.SECTION_WISE,
                section_names=("intro", "outro"),
            ),
        )

        result = run_method_a(run_config)

        self.assertEqual(result.diagnostics.section_tags, ("intro", "outro"))
        self.assertEqual(len(result.endpoints.sections), 2)
        self.assertEqual(result.endpoints.sections[-1].end_time, 4)


class TestMethodBRunConfig(unittest.TestCase):
    def test_valid_construction(self):
        cfg = MethodBRunConfig(
            total_beats=64,
            plan_config=PlanConfig(method=PlanMethod.METHOD_B, loop_midpoint=32),
        )
        self.assertEqual(cfg.loop_midpoint, 32)
        self.assertEqual(cfg.leg1_beats, 32)
        self.assertEqual(cfg.leg2_beats, 32)

    def test_leg_beats_sum_to_total(self):
        for mid in (1, 10, 31):
            cfg = MethodBRunConfig(
                total_beats=32,
                plan_config=PlanConfig(method=PlanMethod.METHOD_B, loop_midpoint=mid),
            )
            self.assertEqual(cfg.leg1_beats + cfg.leg2_beats, cfg.total_beats)

    def test_asymmetric_midpoint(self):
        cfg = MethodBRunConfig(
            total_beats=4,
            plan_config=PlanConfig(method=PlanMethod.METHOD_B, loop_midpoint=1),
        )
        self.assertEqual(cfg.leg1_beats, 1)
        self.assertEqual(cfg.leg2_beats, 3)

    def test_rejects_midpoint_equal_to_total(self):
        with self.assertRaises(ValueError):
            MethodBRunConfig(
                total_beats=32,
                plan_config=PlanConfig(method=PlanMethod.METHOD_B, loop_midpoint=32),
            )

    def test_rejects_midpoint_zero(self):
        with self.assertRaises((ValueError, Exception)):
            MethodBRunConfig(
                total_beats=32,
                plan_config=PlanConfig(method=PlanMethod.METHOD_B, loop_midpoint=0),
            )

    def test_rejects_method_a_config(self):
        with self.assertRaises((ValueError, TypeError)):
            MethodBRunConfig(
                total_beats=32,
                plan_config=PlanConfig(method=PlanMethod.METHOD_A),
            )


class TestMethodBEndpointsShape(unittest.TestCase):
    def _make_cfg(self, total=32, mid=16):
        return MethodBRunConfig(
            total_beats=total,
            plan_config=PlanConfig(method=PlanMethod.METHOD_B, loop_midpoint=mid),
        )

    def test_returns_three_distributions(self):
        endpoints = generate_method_b_endpoints(self._make_cfg())
        self.assertIsNotNone(endpoints.pi0)
        self.assertIsNotNone(endpoints.piMid)
        self.assertIsNotNone(endpoints.piT)

    def test_pi0_time_index_is_zero(self):
        endpoints = generate_method_b_endpoints(self._make_cfg())
        self.assertEqual(endpoints.pi0.layer.time_index, 0)

    def test_piMid_time_index_equals_loop_midpoint(self):
        endpoints = generate_method_b_endpoints(self._make_cfg(total=32, mid=12))
        self.assertEqual(endpoints.piMid.layer.time_index, 12)

    def test_piT_time_index_equals_total_beats(self):
        endpoints = generate_method_b_endpoints(self._make_cfg(total=32, mid=16))
        self.assertEqual(endpoints.piT.layer.time_index, 32)

    def test_two_sections_named_leg1_leg2(self):
        endpoints = generate_method_b_endpoints(self._make_cfg())
        self.assertEqual(len(endpoints.sections), 2)
        self.assertEqual(endpoints.sections[0].name, "leg1")
        self.assertEqual(endpoints.sections[1].name, "leg2")

    def test_sections_cover_full_range(self):
        endpoints = generate_method_b_endpoints(self._make_cfg(total=40, mid=15))
        self.assertEqual(endpoints.sections[0].start_time, 0)
        self.assertEqual(endpoints.sections[0].end_time, 15)
        self.assertEqual(endpoints.sections[1].start_time, 15)
        self.assertEqual(endpoints.sections[1].end_time, 40)
        self.assertEqual(endpoints.sections[0].end_time, endpoints.sections[1].start_time)

    def test_probabilities_sum_to_one(self):
        endpoints = generate_method_b_endpoints(self._make_cfg())
        for dist in (endpoints.pi0, endpoints.piMid, endpoints.piT):
            self.assertAlmostEqual(sum(dist.probabilities), 1.0, places=5)

    def test_sections_span_full_range_various_midpoints(self):
        for total, mid in [(16, 8), (32, 10), (64, 1), (64, 63)]:
            cfg = MethodBRunConfig(
                total_beats=total,
                plan_config=PlanConfig(method=PlanMethod.METHOD_B, loop_midpoint=mid),
            )
            endpoints = generate_method_b_endpoints(cfg)
            s1, s2 = endpoints.sections
            self.assertEqual(s1.start_time, 0)
            self.assertEqual(s1.end_time, mid)
            self.assertEqual(s2.start_time, mid)
            self.assertEqual(s2.end_time, total)


class TestSectionWiseGuards(unittest.TestCase):
    def test_rejects_single_pass_strategy(self):
        from aimusic.planning.plans import run_method_a_sectioned
        cfg = MethodARunConfig(
            total_beats=64,
            plan_config=PlanConfig(
                method=PlanMethod.METHOD_A,
                sectioning_strategy=SectioningStrategy.SINGLE_PASS,
            ),
        )
        with self.assertRaises(ValueError):
            run_method_a_sectioned(cfg)

    def test_rejects_single_section_name(self):
        from aimusic.planning.plans import run_method_a_sectioned
        with self.assertRaises((ValueError, Exception)):
            cfg = MethodARunConfig(
                total_beats=64,
                plan_config=PlanConfig(
                    method=PlanMethod.METHOD_A,
                    sectioning_strategy=SectioningStrategy.SECTION_WISE,
                    section_names=("intro",),
                ),
            )
            run_method_a_sectioned(cfg)


class TestSectionSummary(unittest.TestCase):
    def _make(self, label="s", start=0, end=16, sizes=(8, 8, 8),
              iters=40, converged=True, delta=1e-7, hit_rate=0.6):
        return SectionSummary(
            label=label,
            start_time=start,
            end_time=end,
            beat_count=end - start,
            graph_layer_sizes=sizes,
            mean_layer_size=sum(sizes) / len(sizes),
            min_layer_size=min(sizes),
            max_layer_size=max(sizes),
            bridge_iterations=iters,
            bridge_converged=converged,
            final_max_delta=delta,
            path_mode="map",
            neighbor_cache_hit_rate=hit_rate,
        )

    def test_to_dict_has_required_keys(self):
        d = self._make().to_dict()
        for key in ("label", "start_time", "end_time", "beat_count",
                    "mean_layer_size", "min_layer_size", "max_layer_size",
                    "bridge_iterations", "bridge_converged", "final_max_delta",
                    "path_mode", "neighbor_cache_hit_rate"):
            self.assertIn(key, d)

    def test_beat_count_consistent(self):
        summary = self._make(start=10, end=42)
        self.assertEqual(summary.beat_count, 32)

    def test_mean_layer_size_correct(self):
        summary = self._make(sizes=(10, 20, 30))
        self.assertAlmostEqual(summary.mean_layer_size, 20.0)

    def test_to_dict_is_json_serialisable(self):
        import json
        d = self._make().to_dict()
        # Must not raise
        json.dumps(d)


class TestLongFormDiagnostics(unittest.TestCase):
    def _make_section(self, label, start, end, iters=40, converged=True):
        return SectionSummary(
            label=label, start_time=start, end_time=end,
            beat_count=end - start, graph_layer_sizes=(8,) * 5,
            mean_layer_size=8.0, min_layer_size=8, max_layer_size=8,
            bridge_iterations=iters, bridge_converged=converged,
            final_max_delta=1e-7, path_mode="map", neighbor_cache_hit_rate=0.5,
        )

    def _make_diag(self, n=2, iters=40, converged=True):
        beats = 32
        sections = tuple(
            self._make_section(f"s{i}", i * beats // n, (i + 1) * beats // n, iters, converged)
            for i in range(n)
        )
        return LongFormDiagnostics(
            method="method_a", total_beats=beats, total_sections=n,
            sections=sections, all_sections_converged=converged,
            total_bridge_iterations=iters * n,
            mean_bridge_iterations=float(iters),
            total_graph_states=sum(sum(s.graph_layer_sizes) for s in sections),
            mean_neighbor_cache_hit_rate=0.5,
            path_length=beats + 1,
        )

    def test_converged_section_count(self):
        diag = self._make_diag(n=3, converged=True)
        self.assertEqual(diag.converged_section_count, 3)

    def test_slowest_section_is_max_iters(self):
        diag = self._make_diag(n=3, iters=100)
        self.assertIsNotNone(diag.slowest_section)
        self.assertEqual(diag.slowest_section.bridge_iterations, 100)

    def test_largest_section_is_max_layer(self):
        diag = self._make_diag(n=2)
        self.assertIsNotNone(diag.largest_section)
        self.assertEqual(diag.largest_section.max_layer_size, 8)

    def test_format_summary_contains_method(self):
        diag = self._make_diag()
        self.assertIn("method_a", diag.format_summary())

    def test_to_dict_has_sections_list(self):
        diag = self._make_diag(n=2)
        d = diag.to_dict()
        self.assertIn("sections", d)
        self.assertEqual(len(d["sections"]), 2)

    def test_empty_sections_returns_none_for_slowest(self):
        diag = LongFormDiagnostics(
            method="method_a", total_beats=0, total_sections=0,
            sections=(), all_sections_converged=True,
            total_bridge_iterations=0, mean_bridge_iterations=0.0,
            total_graph_states=0, mean_neighbor_cache_hit_rate=0.0,
            path_length=0,
        )
        self.assertIsNone(diag.slowest_section)
        self.assertIsNone(diag.largest_section)

    def test_to_dict_is_json_serialisable(self):
        import json
        d = self._make_diag().to_dict()
        json.dumps(d)


class TestComparisonRunSpec(unittest.TestCase):
    def test_rejects_empty_label(self):
        cfg = MethodARunConfig(total_beats=16)
        with self.assertRaises(ValueError):
            ComparisonRunSpec(label="", run_config=cfg)

    def test_rejects_wrong_config_type(self):
        with self.assertRaises(TypeError):
            ComparisonRunSpec(label="test", run_config=object())


class TestRunComparison(unittest.TestCase):
    def test_empty_specs_raises(self):
        with self.assertRaises(ValueError):
            run_comparison([])

    def test_failed_arm_captured_not_raised(self):
        cfg = MethodARunConfig(total_beats=2, seed=0)
        spec = ComparisonRunSpec(label="will_fail", run_config=cfg)
        with patch("aimusic.planning.plans.run_method_a",
                   side_effect=RuntimeError("injected failure")):
            report = run_comparison([spec], raise_on_arm_error=False)
        self.assertEqual(len(report.arms), 1)
        self.assertFalse(report.arms[0].succeeded)
        self.assertIn("injected failure", report.arms[0].error)

    def test_raise_on_arm_error_propagates(self):
        cfg = MethodARunConfig(total_beats=2, seed=0)
        spec = ComparisonRunSpec(label="will_fail", run_config=cfg)
        with patch("aimusic.planning.plans.run_method_a",
                   side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                run_comparison([spec], raise_on_arm_error=True)

    def test_format_summary_contains_label(self):
        cfg = MethodARunConfig(total_beats=2, seed=0)
        spec = ComparisonRunSpec(label="my_arm", run_config=cfg)
        with patch("aimusic.planning.plans.run_method_a",
                   side_effect=RuntimeError("fail")):
            report = run_comparison([spec])
        self.assertIn("my_arm", report.format_summary())


class TestComparisonReport(unittest.TestCase):
    def _make_section_summary(self, iters, converged):
        return SectionSummary(
            label="full", start_time=0, end_time=16, beat_count=16,
            graph_layer_sizes=(8,) * 16, mean_layer_size=8.0,
            min_layer_size=8, max_layer_size=8,
            bridge_iterations=iters, bridge_converged=converged,
            final_max_delta=1e-7, path_mode="map",
        )

    def _make_report(self, arm_labels, succeeded_flags, iters_list):
        arms = []
        for label, succeeded, iters in zip(arm_labels, succeeded_flags, iters_list):
            spec = ComparisonRunSpec(
                label=label, run_config=MethodARunConfig(total_beats=16),
            )
            sec = self._make_section_summary(iters, succeeded)
            diag = LongFormDiagnostics(
                method="method_a", total_beats=16, total_sections=1,
                sections=(sec,), all_sections_converged=succeeded,
                total_bridge_iterations=iters,
                mean_bridge_iterations=float(iters),
                total_graph_states=128, mean_neighbor_cache_hit_rate=0.5,
                path_length=17,
            )
            arm = ComparisonArmResult(
                spec=spec, result=None,
                diagnostics=diag, wall_seconds=0.1,
                error=None if succeeded else "fail",
            )
            arms.append(arm)
        converging = [(l, i) for l, s, i in zip(arm_labels, succeeded_flags, iters_list) if s]
        winner = min(converging, key=lambda x: x[1])[0] if converging else None
        return ComparisonReport(
            arms=tuple(arms), winner_label=winner,
            total_wall_seconds=sum(0.1 for _ in arms),
        )

    def test_successful_and_failed_arm_filter(self):
        report = self._make_report(["a", "b", "c"], [True, False, True], [50, 30, 80])
        self.assertEqual(len(report.successful_arms), 2)
        self.assertEqual(len(report.failed_arms), 1)

    def test_winner_label_fewest_iters(self):
        report = self._make_report(["fast", "slow"], [True, True], [30, 100])
        self.assertEqual(report.winner_label, "fast")

    def test_winner_none_when_no_convergence(self):
        report = self._make_report(["a", "b"], [False, False], [50, 80])
        self.assertIsNone(report.winner_label)

    def test_to_dict_arms_count(self):
        d = self._make_report(["x", "y"], [True, True], [40, 60]).to_dict()
        self.assertEqual(len(d["arms"]), 2)

    def test_format_summary_shows_all_labels(self):
        text = self._make_report(["arm_alpha", "arm_beta"], [True, False], [55, 0]).format_summary()
        self.assertIn("arm_alpha", text)
        self.assertIn("arm_beta", text)

if __name__ == "__main__":
    unittest.main()
