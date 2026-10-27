import tempfile
import unittest
from pathlib import Path

import mido

from aimusic.core.config import (
    EDOConfig,
    PlanConfig,
    SectioningStrategy,
    StyleConfig,
)
from aimusic.decode import decode_path_to_score
from aimusic.core.rng import RNGKey
from aimusic.planning.plans import (
    EndpointFallbackPolicy,
    EndpointMode,
    MethodARunConfig,
    build_section_plan,
    generate_end_endpoint_distribution,
    generate_method_a_endpoints,
    generate_start_endpoint_distribution,
    run_method_a,
)
from aimusic.planning.sb import sample_bridge_path
from aimusic.render import render_midi
from aimusic.theory.edo import EDO


def _style() -> StyleConfig:
    return StyleConfig(allowed_meters=("4/4",), groove_families=("straight",))


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
    def test_12_and_19_edo_share_one_context_end_to_end(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            for n in (12, 19):
                with self.subTest(edo=n):
                    result, next_key = run_method_a(
                        MethodARunConfig(
                            total_beats=4,
                            seed=31,
                            edo=n,
                            style_config=StyleConfig(
                                allowed_meters=("4/4",),
                                groove_families=("straight",),
                            ),
                        ), key=RNGKey(seed=31)
                    )
                    score, _ = decode_path_to_score(
                        result.path,
                        vocabularies=result.vocabularies,
                        edo=result.tonal_context.n,
                        key=next_key,
                    )
                    midi_path = Path(temp_dir) / f"{n}-edo.mid"
                    render_midi(
                        score,
                        EDO(EDOConfig(n=result.tonal_context.n, base_tuning=0)),
                        str(midi_path),
                    )

                    chord_roots = {
                        token.root_pc for token in result.vocabularies.chords
                    }
                    self.assertEqual(result.tonal_context.n, n)
                    self.assertEqual(len(result.vocabularies.keys), n)
                    self.assertEqual(chord_roots, set(range(n)))
                    self.assertTrue(score.note_events)
                    self.assertGreater(
                        sum(
                            message.type == "note_on" and message.velocity > 0
                            for track in mido.MidiFile(midi_path).tracks
                            for message in track
                        ),
                        0,
                    )

    def test_generate_method_a_endpoints_returns_distributions_and_sections_only(self):
        run_config = MethodARunConfig(total_beats=4)

        # Endpoint generation makes no choice and needs no RNG key.
        endpoints = generate_method_a_endpoints(run_config)

        self.assertEqual(endpoints.pi0.layer.time_index, 0)
        self.assertEqual(endpoints.piT.layer.time_index, 4)
        self.assertEqual(len(endpoints.sections), 1)
        self.assertFalse(hasattr(endpoints, "start_choice"))
        self.assertFalse(hasattr(endpoints, "end_choice"))
        self.assertAlmostEqual(sum(endpoints.pi0.probabilities), 1.0)
        self.assertAlmostEqual(sum(endpoints.piT.probabilities), 1.0)

    def test_run_config_defaults(self):
        run_config = MethodARunConfig(total_beats=4)
        self.assertIs(run_config.endpoint_mode, EndpointMode.DISTRIBUTION)
        self.assertIs(
            run_config.unreachable_policy, EndpointFallbackPolicy.DROP_UNREACHABLE_ONLY
        )

    # ---- default mode: distribution-to-distribution ------------------------------- #
    def test_run_method_a_map_smoke_distribution_mode(self):
        run_config = MethodARunConfig(total_beats=4, seed=11, style_config=_style())

        input_key = RNGKey(seed=11)
        result, next_key = run_method_a(run_config, key=input_key)

        self.assertEqual(len(result.path), run_config.total_beats + 1)
        # no endpoint was chosen: the path starts/ends inside the effective supports
        start_support = {
            s for s, p in zip(result.endpoints.pi0.layer.states, result.endpoints.pi0.probabilities) if p > 0
        }
        end_support = {
            s for s, p in zip(result.endpoints.piT.layer.states, result.endpoints.piT.probabilities) if p > 0
        }
        self.assertIn(result.path[0], start_support)
        self.assertIn(result.path[-1], end_support)
        self.assertIsNone(result.diagnostics.chosen_start_state)
        self.assertIsNone(result.diagnostics.chosen_end_state)
        self.assertIsNone(result.diagnostics.chosen_start_probability)
        self.assertIsNone(result.diagnostics.chosen_end_probability)
        self.assertEqual(result.diagnostics.endpoint_mode, "distribution")
        self.assertEqual(result.diagnostics.endpoint_selection_mode, "distribution")
        # the boundary layers keep every feasible positive-mass state (not a delta)
        self.assertEqual(result.graph.layers[0].states, result.endpoints.pi0.layer.states)
        self.assertEqual(result.graph.layers[-1].states, result.endpoints.piT.layer.states)
        self.assertGreaterEqual(len(result.graph.layers[0].states), 1)
        self.assertTrue(all(p > 0.0 for p in result.endpoints.pi0.probabilities))
        self.assertTrue(all(p > 0.0 for p in result.endpoints.piT.probabilities))
        self.assertEqual(result.endpoints.pi0.layer.time_index, 0)
        self.assertEqual(result.diagnostics.path_mode, "map")
        self.assertIsNotNone(result.path_score)
        self.assertTrue(result.sb_solution.trace.converged)
        # the realized bridge marginals reproduce the effective endpoint targets
        for report in (result.start_mass, result.end_mass):
            self.assertLessEqual(
                report.max_abs_realized_error, run_config.endpoint_marginal_tolerance
            )
        self.assertTrue(result.endpoint_resolution.final_flow_check.feasible)
        # modifications are always classified, never silent
        resolution = result.endpoint_resolution
        self.assertEqual(
            result.diagnostics.endpoint_request_modified, resolution.request_modified
        )
        self.assertEqual(result.diagnostics.endpoint_modifications, resolution.modifications)
        self.assertAlmostEqual(
            result.diagnostics.endpoint_start_feasible_mass, result.start_mass.feasible_mass
        )
        self.assertAlmostEqual(
            result.diagnostics.endpoint_end_feasible_mass, result.end_mass.feasible_mass
        )
        self.assertEqual(resolution.exact, resolution.termination_reason == "feasible_as_requested")
        for report in (result.start_mass, result.end_mass):
            self.assertAlmostEqual(
                report.removed_state_mass + report.feasible_mass + report.trimmed_requested_mass,
                report.requested_mass,
            )
        self.assertEqual(next_key, input_key.next_key())
        self.assertEqual(
            result.diagnostics.rng_stream_ids,
            (
                "endpoint_choice", "candidate_proposal", "bridge_sampling",
                "decoder.comping", "decoder.bass", "decoder.lead", "decoder.drums",
            ),
        )

    def test_run_method_a_sampling_distribution_mode_is_seed_reproducible(self):
        run_config = MethodARunConfig(
            total_beats=4, seed=23, use_sampling=True, style_config=_style()
        )

        first, _ = run_method_a(run_config, key=RNGKey(seed=23))
        second, _ = run_method_a(run_config, key=RNGKey(seed=23))

        self.assertEqual(first.path, second.path)
        self.assertEqual(first.sampled_path, second.sampled_path)
        self.assertEqual(first.diagnostics.path_mode, "sample")
        # path sampling is separate from endpoint construction: no endpoint is sampled
        self.assertEqual(first.diagnostics.endpoint_selection_mode, "distribution")
        self.assertIsNone(first.diagnostics.chosen_start_state)
        # the sampled start has positive probability under the effective pi0
        self.assertGreater(first.endpoints.pi0.probability_of(first.path[0]), 0.0)
        self.assertGreater(first.endpoints.piT.probability_of(first.path[-1]), 0.0)

    # ---- singleton mode stays available as an explicit special case ---------------- #
    def test_run_method_a_map_smoke_singleton_mode(self):
        run_config = MethodARunConfig(
            total_beats=4,
            seed=11,
            style_config=_style(),
            endpoint_mode=EndpointMode.SINGLETON,
        )

        input_key = RNGKey(seed=11)
        result, next_key = run_method_a(run_config, key=input_key)

        self.assertEqual(len(result.path), run_config.total_beats + 1)
        self.assertEqual(result.path[0], result.diagnostics.chosen_start_state)
        self.assertEqual(result.path[-1], result.diagnostics.chosen_end_state)
        self.assertEqual(len(result.graph.layers[0].states), 1)
        self.assertEqual(len(result.graph.layers[-1].states), 1)
        self.assertEqual(result.diagnostics.endpoint_mode, "singleton")
        self.assertEqual(result.diagnostics.endpoint_selection_mode, "argmax")
        self.assertGreater(result.diagnostics.chosen_start_probability, 0.0)
        self.assertGreater(result.diagnostics.chosen_end_probability, 0.0)
        self.assertEqual(result.diagnostics.path_mode, "map")
        self.assertTrue(result.sb_solution.trace.converged)
        self.assertEqual(next_key, input_key.next_key())

    def test_run_method_a_sampling_singleton_mode_is_seed_reproducible(self):
        run_config = MethodARunConfig(
            total_beats=4,
            seed=23,
            use_sampling=True,
            style_config=_style(),
            endpoint_mode=EndpointMode.SINGLETON,
        )

        first, _ = run_method_a(run_config, key=RNGKey(seed=23))
        second, _ = run_method_a(run_config, key=RNGKey(seed=23))

        self.assertEqual(first.path, second.path)
        self.assertEqual(first.sampled_path, second.sampled_path)
        self.assertEqual(first.diagnostics.path_mode, "sample")
        self.assertEqual(first.diagnostics.endpoint_selection_mode, "sample")
        self.assertEqual(
            first.diagnostics.chosen_start_state, second.diagnostics.chosen_start_state
        )
        self.assertEqual(
            first.diagnostics.chosen_end_state, second.diagnostics.chosen_end_state
        )

    def test_bridge_key_cannot_change_candidate_support(self):
        run_config = MethodARunConfig(total_beats=4, style_config=_style())
        result, _ = run_method_a(run_config, key=RNGKey(seed=45))
        support = result.graph.layers
        sample_bridge_path(result.bridge, RNGKey(seed=1))
        sample_bridge_path(result.bridge, RNGKey(seed=2))
        self.assertEqual(result.graph.layers, support)

    def test_run_method_a_section_wise_smoke(self):
        run_config = MethodARunConfig(
            total_beats=4,
            seed=5,
            style_config=_style(),
            plan_config=PlanConfig(
                sectioning_strategy=SectioningStrategy.SECTION_WISE,
                section_names=("intro", "outro"),
            ),
        )

        result, _ = run_method_a(run_config, key=RNGKey(seed=5))

        self.assertEqual(result.diagnostics.section_tags, ("intro", "outro"))
        self.assertEqual(len(result.endpoints.sections), 2)
        self.assertEqual(result.endpoints.sections[-1].end_time, 4)


if __name__ == "__main__":
    unittest.main()