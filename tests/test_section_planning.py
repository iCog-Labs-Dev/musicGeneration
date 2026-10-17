import dataclasses
import unittest
from unittest.mock import patch

from aimusic.core.config import PlanConfig, SBConfig, SectioningStrategy, StitchingConfig, StyleConfig
from aimusic.core.diagnostics import SBDiagnostics
from aimusic.core.rng import RNGKey
from aimusic.core.core_types import BeatState
from aimusic.core.vocab import build_tonal_context
from aimusic.planning import plans
from aimusic.planning.stitching import concatenate_section_paths, stitch_cost


def section_config(edo=12, sample=False):
    return plans.MethodARunConfig(
        total_beats=16, edo=edo, use_sampling=sample,
        style_config=StyleConfig(allowed_meters=("4/4",), groove_families=("straight",)),
        sb_config=SBConfig(horizon_t=16, max_horizon_per_solve=4),
        plan_config=PlanConfig(
            sectioning_strategy=SectioningStrategy.SECTION_WISE,
            section_names=("intro", "theme", "bridge", "return"),
        ),
    )


class RecordingPrior:
    def __init__(self):
        self.contexts = []

    def logp_next(self, prev_state, next_state, t, context=None):
        self.contexts.append(context)
        return 0.0


class TestSectionPlanning(unittest.TestCase):
    def test_four_solves_contexts_joins_and_global_diagnostics(self):
        prior = RecordingPrior()
        config = section_config()
        config = dataclasses.replace(config, plan_config=dataclasses.replace(
            config.plan_config, stitching=StitchingConfig(max_cost=2.0)))
        with patch.object(plans, "solve_sb", wraps=plans.solve_sb) as solve:
            with patch.object(plans, "build_sparse_graph", wraps=plans.build_sparse_graph) as build:
                result, next_key = plans.run_method_a(config, key=RNGKey(11), prior=prior)
        self.assertIsInstance(result, plans.SectionWisePlanResult)
        self.assertEqual(solve.call_count, 4)
        self.assertEqual(build.call_count, 4)
        self.assertEqual(next_key, RNGKey(11).next_key())
        self.assertEqual(len(result.path), 17)
        self.assertEqual(len(result.joins), 3)
        self.assertEqual([item.time_index for item in result.path_edge_diagnostics], list(range(16)))
        self.assertEqual(len(result.diagnostics.graph_layer_sizes), 17)
        self.assertEqual({c.section_name for c in prior.contexts}, {"intro", "theme", "bridge", "return"})
        for context in prior.contexts:
            self.assertIn("target_tension_arc", dict(context.metadata))
            self.assertLess(int(dict(context.metadata)["graph_time"]), 4)
        for section in result.section_results:
            self.assertEqual(section.sb_solution.problem.sb_config.horizon_t, 4)
            self.assertEqual(section.graph.layers[0].time_index, 0)
            self.assertEqual(section.graph.layers[-1].time_index, 4)
            self.assertTrue(all(item.stitch_cost <= 2.0
                                for layer in section.graph.edge_diagnostics_by_time for item in layer))
            for item in section.graph.diagnostics_for_path(section.path):
                self.assertAlmostEqual(item.final_log_weight,
                                       item.data_contribution + item.gttm_contribution - item.stitch_cost)
        for left, right in zip(result.section_results, result.section_results[1:]):
            self.assertEqual(left.path[-1], right.path[0])
            self.assertEqual(left.sb_solution.problem.piT.layer.states,
                             right.sb_solution.problem.pi0.layer.states)
            self.assertEqual(left.sb_solution.problem.piT.probabilities,
                             right.sb_solution.problem.pi0.probabilities)
        for join in result.joins:
            self.assertLessEqual(join.incoming.total, join.tolerance)
            self.assertLessEqual(join.outgoing.total, join.tolerance)
        stats = SBDiagnostics.from_plan(result)
        self.assertEqual([item["name"] for item in stats.sections], ["intro", "theme", "bridge", "return"])
        self.assertEqual(len(stats.joins), 3)
        self.assertEqual(stats.iterations_run, sum(s.trace.iterations for s in result.solutions))
        self.assertEqual(len(stats.layer_sizes), 17)

    def test_budget_rejects_monolithic_but_accepts_sections(self):
        config = section_config()
        single = dataclasses.replace(config, plan_config=PlanConfig())
        with self.assertRaisesRegex(ValueError, "max_horizon_per_solve"):
            plans.run_method_a(single, key=RNGKey(11))
        result, _ = plans.run_method_a(config, key=RNGKey(11))
        self.assertEqual(len(result.path), 17)

    def test_12_and_19_edo_sampling_reproducible(self):
        for edo in (12, 19):
            with self.subTest(edo=edo):
                config = section_config(edo, sample=True)
                first, first_key = plans.run_method_a(config, key=RNGKey(7))
                second, second_key = plans.run_method_a(config, key=RNGKey(7))
                self.assertEqual(first.path, second.path)
                self.assertEqual(first.joins, second.joins)
                self.assertEqual(first.diagnostics, second.diagnostics)
                self.assertEqual(first_key, second_key)
                self.assertEqual(first.boundary_choices, second.boundary_choices)

    def test_section_failure_identifies_stage_and_cause(self):
        config = section_config()
        config = dataclasses.replace(config, sb_config=dataclasses.replace(config.sb_config, max_horizon_per_solve=3))
        with self.assertRaises(plans.SectionPlanningError) as caught:
            plans.run_method_a(config, key=RNGKey(11))
        self.assertEqual(caught.exception.section.name, "intro")
        self.assertEqual(caught.exception.stage, "graph construction")
        self.assertIsInstance(caught.exception.__cause__, ValueError)

    def test_tight_join_tolerance_fails_locally(self):
        config = section_config()
        config = dataclasses.replace(config, plan_config=dataclasses.replace(
            config.plan_config, stitching=StitchingConfig(max_cost=0.0)))
        with self.assertRaises(plans.SectionPlanningError) as caught:
            plans.run_method_a(config, key=RNGKey(11))
        self.assertEqual(caught.exception.section.name, "intro")

    def test_weights_are_applied_to_boundary_edges(self):
        config = section_config()
        result, _ = plans.run_method_a(config, key=RNGKey(11))
        self.assertTrue(any(item.stitch_cost > 0 for item in result.path_edge_diagnostics))
        for section in result.section_results:
            for edge in section.graph.diagnostics_for_path(section.path)[1:-1]:
                self.assertEqual(edge.stitch_cost, 0.0)
        zero = StitchingConfig(key_weight=0, chord_weight=0, meter_weight=0,
                               groove_weight=0, role_weight=0, head_weight=0, boundary_weight=0)
        left, right = result.path[3:5]
        self.assertEqual(stitch_cost(left, right, zero, result.vocabularies, 12).total, 0)
        with self.assertRaisesRegex(ValueError, "share their endpoint"):
            concatenate_section_paths(((left, right), (left, right)))

    def test_stitch_config_rejects_nonfinite_or_negative_values(self):
        for value in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                StitchingConfig(max_cost=value)

    def test_pitch_distance_wraps_and_each_weight_controls_its_component(self):
        zero_weights = dict(key_weight=0, chord_weight=0, meter_weight=0, groove_weight=0,
                            role_weight=0, head_weight=0, boundary_weight=0)
        for edo in (12, 19):
            vocabs = build_tonal_context(edo, StyleConfig()).vocabularies
            root_key = next(token.id for token in vocabs.keys if token.root_pc == 0)
            wrap_key = next(token.id for token in vocabs.keys if token.root_pc == edo - 1)
            root_chord = next(token.id for token in vocabs.chords if token.root_pc == 0 and token.quality == "maj")
            wrap_chord = next(token.id for token in vocabs.chords if token.root_pc == edo - 1 and token.quality == "min")
            left = BeatState(0, 0, 0, root_key, root_chord, 0, 1, 0)
            right = dataclasses.replace(left, key_id=wrap_key, chord_id=wrap_chord,
                                        meter_id=1, groove_id=1, role_id=1, head_id=2, boundary_lvl=3)
            distances = stitch_cost(left, right, StitchingConfig(), vocabs, edo)
            self.assertAlmostEqual(distances.key, 1 / (edo // 2))
            self.assertAlmostEqual(distances.chord, (1 + 1 / (edo // 2)) / 2)
            for component in ("key", "chord", "meter", "groove", "role", "head", "boundary"):
                weights = dict(zero_weights)
                weights[f"{component}_weight"] = 2
                measured = stitch_cost(left, right, StitchingConfig(**weights), vocabs, edo)
                self.assertAlmostEqual(measured.total, 2 * getattr(distances, component))

    def test_uneven_sections_preserve_global_meter_phase(self):
        config = section_config()
        config = dataclasses.replace(config, total_beats=10,
                                     sb_config=dataclasses.replace(config.sb_config, horizon_t=10))
        result, _ = plans.run_method_a(config, key=RNGKey(11))
        self.assertEqual([len(section.path) - 1 for section in result.section_results], [3, 3, 2, 2])
        self.assertEqual([state.beat_in_bar for state in result.path], [t % 4 for t in range(11)])


if __name__ == "__main__":
    unittest.main()
