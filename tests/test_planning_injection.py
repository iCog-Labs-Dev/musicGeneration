from __future__ import annotations

from dataclasses import replace
from copy import copy
import math
import unittest
from unittest.mock import Mock
from typing import Callable

from aimusic.core.component_types import CandidateGenerationResult, CandidateRejection
from aimusic.core.config import PriorWeights, SBConfig, StyleConfig
from aimusic.core.core_types import BeatState, Layer
from aimusic.core.rng import RNGKey
from aimusic.core.vocab import Vocabularies
from aimusic.planning.graph import build_sparse_graph
from aimusic.planning.candidates import get_valid_next_states
from aimusic.planning.plans import MethodAPlanResult, MethodARunConfig, render_exact_bridge_demo, run_method_a
from aimusic.scoring.priors import Prior, PriorContext
from tests.test_graph import state
from tests.test_section_planning import section_config
from aimusic.theory.tonal import basic_space_distance, tonal_distance


class ControlledCandidates:
    def __init__(self, successor: BeatState | tuple[BeatState, ...] | None, *, wrong_time: bool = False) -> None:
        self.successor = successor
        self.wrong_time = wrong_time
        self.calls: list[int] = []

    def __call__(
        self,
        prev_state: BeatState,
        t: int,
        key: RNGKey,
        d_max: int,
        style_config: StyleConfig | None = None,
        vocabularies: Vocabularies | None = None,
        prior: Prior | None = None,
        context: PriorContext | None = None,
        edo: int | None = None,
        proposal_budget: int | None = None,
        prior_guided_proposals: bool = False,
    ) -> tuple[CandidateGenerationResult, RNGKey]:
        self.calls.append(t)
        states = self.successor if isinstance(self.successor, tuple) else (() if self.successor is None else (self.successor,))
        return CandidateGenerationResult(
            time_index=t + int(self.wrong_time),
            source_state=prev_state,
            states=states,
            proposed_count=len(states),
            unique_count=len(states),
        ), key.next_key()


class FixedMetric:
    def tonal_distance(self, a_root: int, b_root: int, edo: int = 12) -> float:
        return 3.0

    def basic_space_distance(
        self, a_root: int, a_quality: str, b_root: int, b_quality: str, edo: int = 12
    ) -> float:
        return 7.0


class TracedCandidates:
    marker = "controlled_rejected_source_proposal"

    def __init__(self, *, fail: bool = False, on_call: Callable[[PriorContext | None], None] | None = None) -> None:
        self.fail = fail
        self.on_call = on_call
        self.calls: list[int] = []

    def __call__(
        self, prev_state: BeatState, t: int, key: RNGKey, d_max: int,
        style_config: StyleConfig | None = None,
        vocabularies: Vocabularies | None = None,
        prior: Prior | None = None, context: PriorContext | None = None,
        edo: int | None = None, proposal_budget: int | None = None,
        prior_guided_proposals: bool = False,
    ) -> tuple[CandidateGenerationResult, RNGKey]:
        self.calls.append(t)
        if self.on_call is not None:
            self.on_call(context)
        if self.fail:
            raise RuntimeError("candidate replacement failed")
        result, next_key = get_valid_next_states(
            prev_state, t, key, d_max, style_config, vocabularies,
            prior, context, edo, proposal_budget, prior_guided_proposals,
        )
        # One extra proposal repeats the source instead of advancing a beat.
        rejected = CandidateRejection(t, prev_state, prev_state, self.marker)
        return replace(
            result, rejections=result.rejections + (rejected,),
            proposed_count=result.proposed_count + 1,
            unique_count=result.unique_count + 1,
        ), next_key


class PreferGMetric(FixedMetric):
    def basic_space_distance(
        self, a_root: int, a_quality: str, b_root: int, b_quality: str, edo: int = 12
    ) -> float:
        return 0.0 if a_root == 7 else 100.0


class RecordingMetric:
    def __init__(self) -> None:
        self.root_calls = 0

    def tonal_distance(self, a_root: int, b_root: int, edo: int = 12) -> float:
        self.root_calls += 1
        return tonal_distance(a_root, b_root, edo)

    def basic_space_distance(self, a_root: int, a_quality: str, b_root: int, b_quality: str, edo: int = 12) -> float:
        return basic_space_distance(a_root, a_quality, b_root, b_quality, edo)


class TestPlanningInjection(unittest.TestCase):
    def test_selected_metric_and_candidates_reach_every_section_in_both_tunings(self) -> None:
        for edo in (12, 19):
            with self.subTest(edo=edo):
                metric = RecordingMetric()
                starts: list[tuple[str, int]] = []

                def observe(context: PriorContext | None) -> None:
                    assert context is not None and context.section_name is not None
                    if not starts or starts[-1][0] != context.section_name:
                        starts.append((context.section_name, metric.root_calls))

                generator = TracedCandidates(on_call=observe)
                result, _ = run_method_a(
                    section_config(edo), key=RNGKey(11), tonal_metric=metric, candidate_generator=generator,
                )
                self.assertEqual([name for name, _ in starts], ["intro", "theme", "bridge", "return"])
                counts = [count for _, count in starts] + [metric.root_calls]
                for start, end in zip(counts, counts[1:]):
                    self.assertGreater(end, start, "a section ignored the supplied metric")
                self.assertEqual(len(result.path), 17)
    def setUp(self) -> None:
        self.start = state(beat=0)
        self.middle = state(beat=1, head="third")
        self.end = state(beat=2)
        self.key = RNGKey(seed=113)

    def test_inconsistent_retention_limits_fail_before_candidate_generation(self) -> None:
        generator = ControlledCandidates(self.middle)
        with self.assertRaisesRegex(ValueError, "d_max"):
            build_sparse_graph(
                Layer(0, (self.start,)), Layer(2, (self.end,)), 2,
                sb_config=SBConfig(d_max=2), d_max=1, key=self.key,
                candidate_generator=generator,
            )
        self.assertEqual(generator.calls, [])

    def test_graph_rejects_malformed_external_results_and_rng_values(self) -> None:
        valid = CandidateGenerationResult(0, self.start, (self.middle,), proposed_count=1, unique_count=1)
        for changes in (
            {"proposed_count": 0}, {"unique_count": 2},
            {"states": (self.middle, self.middle), "unique_count": 2, "proposed_count": 2},
            {"scores": (math.nan,)}, {"scores": (math.inf,)}, {"scores": (-math.inf,)},
            {"rejections": (CandidateRejection(1, self.start, self.start, "wrong time"),)},
            {"rejections": (CandidateRejection(0, self.end, self.start, "wrong source"),)},
        ):
            # External objects can bypass constructors; the graph rechecks its boundary.
            result = copy(valid)
            for field, value in changes.items():
                object.__setattr__(result, field, value)
            with self.subTest(changes=changes), self.assertRaises((TypeError, ValueError)):
                build_sparse_graph(
                    Layer(0, (self.start,)), Layer(2, (self.end,)), 2,
                    key=self.key, d_max=4, candidate_generator=Mock(return_value=(result, self.key)),
                )
        for invalid_key in (None, "not a random key", 12):
            with self.subTest(key=invalid_key), self.assertRaises(TypeError):
                build_sparse_graph(
                    Layer(0, (self.start,)), Layer(2, (self.end,)), 2,
                    key=self.key, d_max=4, candidate_generator=Mock(return_value=(valid, invalid_key)),
                )

    def test_terminal_checks_do_not_inflate_raw_generator_proposal_counts(self) -> None:
        graph, _ = build_sparse_graph(
            Layer(0, (self.start,)), Layer(1, (self.middle,)), 1,
            key=self.key, d_max=2,
        )
        self.assertEqual(graph.diagnostics.layer_diagnostics[0].raw_candidate_count, 0)
        self.assertEqual(graph.diagnostics.layer_diagnostics[0].legal_candidate_count, 1)

    def test_run_method_a_consumes_supplied_candidate_results(self) -> None:
        generator = TracedCandidates()
        result, _ = run_method_a(
            MethodARunConfig(total_beats=4, seed=11), key=RNGKey(seed=11),
            candidate_generator=generator,
        )
        self.assertEqual(len(result.path), 5)
        assert isinstance(result, MethodAPlanResult)
        markers = [
            rejection
            for layer in result.graph.diagnostics.layer_diagnostics
            for rejection in layer.rejected_proposals
            if rejection.reason == generator.marker
        ]
        self.assertGreater(len(markers), 0, "supplied candidate diagnostics were not retained")
        self.assertEqual(set(generator.calls), {0, 1, 2})
        self.assertEqual(len(markers), len(generator.calls))

    def test_demo_forwards_candidate_errors(self) -> None:
        from pathlib import Path
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "demo.mid"
            with self.assertRaisesRegex(RuntimeError, "candidate replacement failed"):
                render_exact_bridge_demo(
                    start_chord="Cmaj", end_chord="Gmaj", output_path=str(destination),
                    total_beats=4, seed=11,
                    candidate_generator=TracedCandidates(fail=True),
                )
            self.assertFalse(destination.exists())

    def test_metric_controls_edge_retention_and_layer_pruning(self) -> None:
        preferred = state(beat=1, chord="Gmaj")
        other = state(beat=1)
        for config in (SBConfig(k_max=1, d_max=2), SBConfig(k_max=2, d_max=1)):
            with self.subTest(config=config):
                graph, _ = build_sparse_graph(
                    Layer(0, (self.start,)), Layer(2, (self.end,)), 2,
                    key=self.key, d_max=config.d_max, sb_config=config,
                    weights=PriorWeights(lambda_data=1.0, lambda_gttm=0.0),
                    tonal_metric=PreferGMetric(),
                    candidate_generator=ControlledCandidates((other, preferred)),
                )
                self.assertEqual(graph.layers[1].states, (preferred,))

    def test_generator_output_controls_graph_and_final_support(self) -> None:
        generator = ControlledCandidates(self.middle)
        graph, key = build_sparse_graph(
            Layer(0, (self.start,)), Layer(2, (self.end,)), 2,
            key=self.key, d_max=4, candidate_generator=generator,
        )
        self.assertEqual(graph.layers[1].states, (self.middle,))
        self.assertEqual(graph.layers[-1].states, (self.end,))
        self.assertEqual(generator.calls, [0])
        self.assertEqual(key, self.key.next_key())
        self.assertEqual(graph.diagnostics.layer_diagnostics[0].raw_candidate_count, 1)

    def test_empty_result_stays_empty_without_default_fallback(self) -> None:
        graph, _ = build_sparse_graph(
            Layer(0, (self.start,)), Layer(2, (self.end,)), 2,
            key=self.key, d_max=4,
            candidate_generator=ControlledCandidates(None),
        )
        self.assertEqual(graph.layers[1].states, ())
        self.assertEqual(graph.edges_by_time, ((), ()))

    def test_invalid_time_or_illegal_transition_is_rejected(self) -> None:
        for generator in (
            ControlledCandidates(self.middle, wrong_time=True),
            ControlledCandidates(replace(self.middle, beat_in_bar=3)),
            ControlledCandidates(replace(self.middle, chord_id=99999)),
        ):
            with self.subTest(generator=generator):
                with self.assertRaises(ValueError):
                    build_sparse_graph(
                        Layer(0, (self.start,)), Layer(2, (self.end,)), 2,
                        key=self.key, d_max=4, candidate_generator=generator,
                    )

    def test_metric_reaches_retained_raw_and_weighted_scores(self) -> None:
        graph, _ = build_sparse_graph(
            Layer(0, (self.start,)), Layer(2, (self.end,)), 2,
            key=self.key, d_max=4, tonal_metric=FixedMetric(),
            candidate_generator=ControlledCandidates(self.middle),
        )
        for layer in graph.edge_diagnostics_by_time:
            for item in layer:
                self.assertAlmostEqual(item.raw_feature_contributions["harmonic_key_proximity"], 0.25)
                self.assertAlmostEqual(item.raw_feature_contributions["harmonic_chord_proximity"], 0.125)
                self.assertAlmostEqual(item.weighted_feature_contributions["harmonic_key_proximity"], 0.25)
                self.assertAlmostEqual(item.weighted_feature_contributions["harmonic_chord_proximity"], 0.15)
                self.assertAlmostEqual(item.final_log_weight, item.data_contribution + item.gttm_contribution)

    def test_default_graph_replays_with_omitted_dependencies(self) -> None:
        first = build_sparse_graph(
            Layer(0, (self.start,)), Layer(2, (self.end,)), 2,
            sb_config=SBConfig(k_max=3, d_max=2), key=self.key, d_max=2,
        )
        second = build_sparse_graph(
            Layer(0, (self.start,)), Layer(2, (self.end,)), 2,
            sb_config=SBConfig(k_max=3, d_max=2), key=self.key, d_max=2,
        )
        self.assertEqual(first, second)
