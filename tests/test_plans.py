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

    def test_generate_method_a_endpoints_returns_sections(self):
        run_config = MethodARunConfig(total_beats=4)

        endpoints, _ = generate_method_a_endpoints(run_config, key=RNGKey(seed=0))

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

        input_key = RNGKey(seed=11)
        result, next_key = run_method_a(run_config, key=input_key)

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
        self.assertEqual(next_key, input_key.next_key())
        self.assertEqual(
            result.diagnostics.rng_stream_ids,
            (
                "endpoint_choice", "candidate_proposal", "bridge_sampling",
                "decoder.comping", "decoder.bass", "decoder.lead", "decoder.drums",
            ),
        )

    def test_run_method_a_sampling_is_seed_reproducible(self):
        run_config = MethodARunConfig(
            total_beats=4,
            seed=23,
            use_sampling=True,
            style_config=StyleConfig(allowed_meters=("4/4",), groove_families=("straight",)),
        )

        first, _ = run_method_a(run_config, key=RNGKey(seed=23))
        second, _ = run_method_a(run_config, key=RNGKey(seed=23))

        self.assertEqual(first.path, second.path)
        self.assertEqual(first.sampled_path, second.sampled_path)
        self.assertEqual(first.diagnostics.path_mode, "sample")
        self.assertEqual(first.diagnostics.endpoint_selection_mode, "sample")
        self.assertEqual(first.endpoints.start_choice, second.endpoints.start_choice)
        self.assertEqual(first.endpoints.end_choice, second.endpoints.end_choice)

    def test_bridge_key_cannot_change_candidate_support(self):
        run_config = MethodARunConfig(
            total_beats=4,
            style_config=StyleConfig(allowed_meters=("4/4",), groove_families=("straight",)),
        )
        result, _ = run_method_a(run_config, key=RNGKey(seed=45))
        support = result.graph.layers
        sample_bridge_path(result.bridge, RNGKey(seed=1))
        sample_bridge_path(result.bridge, RNGKey(seed=2))
        self.assertEqual(result.graph.layers, support)

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

        result, _ = run_method_a(run_config, key=RNGKey(seed=5))

        self.assertEqual(result.diagnostics.section_tags, ("intro", "outro"))
        self.assertEqual(len(result.endpoints.sections), 2)
        self.assertEqual(result.endpoints.sections[-1].end_time, 4)


if __name__ == "__main__":
    unittest.main()
