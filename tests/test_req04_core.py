from __future__ import annotations

import inspect
import unittest

from aimusic.core.config import (
    DecodeConfig,
    PriorWeights,
    SBConfig,
    SectionStyleWeights,
    StitchingConfig,
    StyleConfig,
)
from aimusic.core.core_types import BeatState
from aimusic.core.rng import RNGKey
from aimusic.core.vocab import DEFAULT_VOCABULARIES, build_default_vocabularies
from aimusic.decode import effective_density
from aimusic.planning.candidates import (
    apply_boundary_and_groove_constraints,
    apply_meter_constraints,
    apply_position_constraints,
    apply_role_constraints,
    is_legal_transition,
)
from aimusic.planning.plans import MethodARunConfig, PlanningSection, run_method_a
from aimusic.planning.sections import SectionPlanError, SectionTimeline
from aimusic.planning.stitching import StitchPolicy
from aimusic.scoring.priors import (
    NullPrior,
    PriorContext,
    PriorQuery,
    calculate_transition_score_breakdown,
)
from aimusic.scoring.tension import (
    section_style_breakdown,
    section_style_energy,
    transition_target_tension_energy,
)


def _state(
    *,
    meter: int = 0,
    beat: int = 0,
    boundary: int = 0,
    key: int = 0,
    role: int = 0,
    groove: int = 0,
) -> BeatState:
    return BeatState(meter, beat, boundary, key, 0, role, 1, groove)


class TestReq04Core(unittest.TestCase):
    def setUp(self) -> None:
        self.vocabs = DEFAULT_VOCABULARIES

    def test_timeline_interpolation_and_arrival_indexing(self) -> None:
        sections = (
            PlanningSection("A", 0, 4, 2, (0.2, 0.8)),
            PlanningSection("B", 4, 8, 3, (0.8, 0.3)),
        )
        timeline = SectionTimeline(sections, total_beats=8)

        self.assertAlmostEqual(timeline.target_tension(0), 0.2)
        self.assertAlmostEqual(timeline.target_tension(3), 0.8)
        self.assertAlmostEqual(timeline.target_tension(4), 0.8)
        self.assertAlmostEqual(timeline.target_tension(8), 0.3)
        self.assertEqual(timeline.edge_view(3).section.name, "B")
        self.assertTrue(timeline.edge_view(3).is_section_start)

    def test_timeline_validation_rejects_or_snaps_off_bar_sections(self) -> None:
        style = StyleConfig(allowed_meters=("4/4",))
        vocabs = build_default_vocabularies(style)
        sections = (
            PlanningSection("A", 0, 6, 2),
            PlanningSection("B", 6, 16, 3),
        )
        timeline = SectionTimeline(sections, total_beats=16)

        with self.assertRaisesRegex(SectionPlanError, "downbeat"):
            timeline.validate_executable(style, vocabs)

        snapped, messages = timeline.validate_executable(
            style,
            vocabs,
            snap_to_bars=True,
        )
        self.assertEqual(
            tuple((section.start_time, section.end_time) for section in snapped.sections),
            ((0, 4), (4, 16)),
        )
        self.assertEqual(len(messages), 1)

    def test_method_a_rejects_authored_off_bar_boundary(self) -> None:
        config = MethodARunConfig(
            total_beats=16,
            style_config=StyleConfig(allowed_meters=("4/4",)),
            sections=(
                PlanningSection("A", 0, 6, 2),
                PlanningSection("B", 6, 16, 3),
            ),
        )

        with self.assertRaisesRegex(SectionPlanError, "downbeat"):
            run_method_a(config, key=RNGKey(seed=11))

    def test_style_components_and_ablation(self) -> None:
        meter_4_4 = self.vocabs.meters.token_for_label("4/4").id
        other_meter = next(t.id for t in self.vocabs.meters if t.label != "4/4")
        straight = next(t.id for t in self.vocabs.grooves if t.family == "straight")
        syncopated = next(t.id for t in self.vocabs.grooves if t.family == "syncopated")
        hold = self.vocabs.roles.token_for_label("hold").id
        cadence = self.vocabs.roles.token_for_label("cad").id
        key_c = next(t.id for t in self.vocabs.keys if t.root_pc == 0)
        key_g = next(t.id for t in self.vocabs.keys if t.root_pc == 7)
        section = PlanningSection(
            "A",
            0,
            4,
            3,
            allowed_meters=("4/4",),
            groove_family="straight",
            key_region=(0,),
            preferred_roles=("hold",),
        )
        satisfied = _state(
            meter=meter_4_4,
            boundary=3,
            key=key_c,
            role=hold,
            groove=straight,
        )
        violated = _state(
            meter=other_meter,
            beat=3,
            boundary=0,
            key=key_g,
            role=cadence,
            groove=syncopated,
        )

        good = section_style_breakdown(
            None, satisfied, section, self.vocabs, is_section_start=True
        )
        bad = section_style_breakdown(
            None, violated, section, self.vocabs, is_section_start=True
        )
        self.assertTrue(all(value == 0.0 for value in good.values()))
        for component in (
            "meter",
            "key_region",
            "groove_family",
            "role",
            "boundary",
            "start_alignment",
        ):
            with self.subTest(component=component):
                self.assertGreater(bad[component], 0.0)

        zero = SectionStyleWeights(
            meter=0.0,
            key_region=0.0,
            groove_family=0.0,
            role=0.0,
            boundary=0.0,
            start_alignment=0.0,
        )
        self.assertEqual(
            section_style_energy(
                None,
                violated,
                section,
                self.vocabs,
                style_weights=zero,
                is_section_start=True,
            ),
            0.0,
        )

    def test_transition_score_decomposition_and_zero_weight_baseline(self) -> None:
        section = PlanningSection("A", 0, 4, 3, (0.2, 0.8))
        timeline = SectionTimeline((section,), total_beats=4)
        previous = _state(boundary=3)
        arrival = _state(beat=1, role=1)
        context = PriorContext(history=(previous,), section_view=timeline.edge_view(0))
        weights = PriorWeights(
            lambda_data=1.2,
            lambda_gttm=0.8,
            lambda_target_tension=5.0,
            lambda_section_style=3.0,
        )

        breakdown = calculate_transition_score_breakdown(
            previous,
            arrival,
            0,
            prior=NullPrior(),
            context=context,
            weights=weights,
            vocabularies=self.vocabs,
            edo=12,
        )
        breakdown.check_decomposition(tol=1e-9)
        expected = (
            breakdown.data_contribution
            + breakdown.gttm_contribution
            - breakdown.target_tension_contribution
            - breakdown.section_style_contribution
        )
        self.assertAlmostEqual(breakdown.final_log_weight, expected)
        self.assertAlmostEqual(
            transition_target_tension_energy(0.6, 0.5, 0.1),
            0.5,
        )

        baseline = calculate_transition_score_breakdown(
            previous,
            arrival,
            0,
            prior=NullPrior(),
            context=context,
            weights=PriorWeights(
                lambda_target_tension=0.0,
                lambda_section_style=0.0,
            ),
            vocabularies=self.vocabs,
            edo=12,
        )
        self.assertEqual(baseline.target_tension_contribution, 0.0)
        self.assertEqual(baseline.section_style_contribution, 0.0)

    def test_prior_receives_arrival_section_tokens(self) -> None:
        timeline = SectionTimeline(
            (
                PlanningSection("A", 0, 4, 2),
                PlanningSection("B", 4, 8, 3),
            ),
            total_beats=8,
        )
        query = PriorQuery(
            _state(beat=2),
            _state(beat=3, role=1),
            time_index=3,
            context=PriorContext(section_view=timeline.edge_view(3)),
        ).tokenize()

        self.assertIsNotNone(query.section_tokens)
        tokens = query.section_tokens
        assert tokens is not None
        self.assertEqual(tokens.section_name, "B")
        self.assertEqual(tokens.section_index, 1)
        self.assertEqual(tokens.offset_in_section, 0)
        self.assertTrue(tokens.is_section_start)
        self.assertIsNone(
            PriorQuery(_state(), _state(beat=1), 0, PriorContext())
            .tokenize()
            .section_tokens
        )

    def test_density_and_19_edo_context(self) -> None:
        decode = DecodeConfig(reference_density=0.5)
        low = SectionTimeline(
            (PlanningSection("L", 0, 4, 3, target_density=0.2),),
            total_beats=4,
            edo=19,
        ).state_view(0)
        high = SectionTimeline(
            (PlanningSection("H", 0, 4, 3, target_density=0.8),),
            total_beats=4,
            edo=19,
        ).state_view(0)

        self.assertLess(
            effective_density(0.5, low, decode),
            effective_density(0.5, high, decode),
        )
        vocabs_19 = build_default_vocabularies(edo=19)
        root_zero = next(t.id for t in vocabs_19.keys if t.root_pc == 0)
        section = PlanningSection("19edo", 0, 4, 3, key_region=(0,))
        components = section_style_breakdown(
            None,
            _state(key=root_zero),
            section,
            vocabs_19,
            edo=19,
        )
        self.assertEqual(components["key_region"], 0.0)

    def test_hard_legality_functions_do_not_accept_section_targets(self) -> None:
        for function in (
            is_legal_transition,
            apply_meter_constraints,
            apply_position_constraints,
            apply_role_constraints,
            apply_boundary_and_groove_constraints,
        ):
            with self.subTest(function=function.__name__):
                section_parameters = [
                    name
                    for name in inspect.signature(function).parameters
                    if name.startswith("section")
                ]
                self.assertEqual(section_parameters, [])


    def test_verify_decomposition_with_stitch_policy(self) -> None:
        cfg = MethodARunConfig(
            total_beats=4,
            seed=42,
            sb_config=SBConfig(horizon_t=4, k_max=16, d_max=4),
        )
        policy = StitchPolicy(
            config=StitchingConfig(max_cost=100.0),
            vocabularies=self.vocabs,
            edo=12,
            edge_times=(1,),
        )
        res, _ = run_method_a(cfg, key=RNGKey(42), stitch_policy=policy)
        diags = res.diagnostics.transition_diagnostics
        self.assertEqual(len(diags), 4)
        for d in diags:
            self.assertTrue(d.verify_decomposition())


if __name__ == "__main__":
    unittest.main()
