from dataclasses import replace
import math
import unittest
from typing import Callable

from aimusic.core.component_types import (
    CandidateGenerationResult, CandidateRejection, RenderRequest, RenderResult,
    SymbolicNote, TrackInstrumentConfig,
)
from tests.test_graph import state


class TestComponentValidation(unittest.TestCase):
    def test_candidate_counters_describe_the_unique_outcomes(self) -> None:
        source, target = state(beat=0), state(beat=1)
        for proposed, unique in ((0, 1), (1, 2), (2, 0), (-1, 1), (True, 1)):
            with self.subTest(proposed=proposed, unique=unique), self.assertRaises((TypeError, ValueError)):
                CandidateGenerationResult(0, source, (target,), proposed_count=proposed, unique_count=unique)
        valid = CandidateGenerationResult(0, source, (target,), proposed_count=3, unique_count=1)
        self.assertEqual(valid.legal_count, 1)

    def test_candidate_states_scores_and_rejections_are_consistent(self) -> None:
        source, target = state(beat=0), state(beat=1)
        base = CandidateGenerationResult(0, source, (target,), proposed_count=1, unique_count=1)
        for changes in (
            {"states": (target, target), "proposed_count": 2, "unique_count": 2},
            {"states": [target]},
            {"scores": (math.nan,)}, {"scores": (math.inf,)}, {"scores": (-math.inf,)},
            {"scores": (True,)}, {"scores": (0.0, 1.0)},
            {"rejections": (CandidateRejection(1, source, source, "wrong time"),), "proposed_count": 2, "unique_count": 2},
            {"rejections": (CandidateRejection(0, target, source, "wrong source"),), "proposed_count": 2, "unique_count": 2},
            {"rejections": (CandidateRejection(0, source, target, "accepted and rejected"),), "proposed_count": 2, "unique_count": 2},
        ):
            with self.subTest(changes=changes), self.assertRaises((TypeError, ValueError)):
                replace(base, **changes)

    def test_rejection_requires_valid_state_time_and_reason(self) -> None:
        source = state(beat=0)
        for args in ((-1, source, source, "reason"), (True, source, source, "reason"),
                     (0, None, source, "reason"), (0, source, None, "reason"), (0, source, source, "  ")):
            with self.subTest(args=args), self.assertRaises((TypeError, ValueError)):
                CandidateRejection(*args)

    def test_symbolic_notes_reject_invalid_intervals_and_controls(self) -> None:
        note = SymbolicNote(-1, 0.0, 1.0)
        for changes in ({"pitch_height": True}, {"start_time": -1}, {"end_time": 0},
                        {"start_time": math.nan}, {"end_time": math.inf},
                        {"velocity": 128}, {"velocity": True}, {"timbre": -1}, {"pressure": 128}):
            with self.subTest(changes=changes), self.assertRaises((TypeError, ValueError)):
                replace(note, **changes)

    def test_instrument_program_and_drum_flag_are_validated(self) -> None:
        for settings in ({"program": -1}, {"program": 128}, {"program": True}, {"is_drum": 1}):
            with self.subTest(settings=settings), self.assertRaises((TypeError, ValueError)):
                TrackInstrumentConfig(**settings)
        self.assertEqual(TrackInstrumentConfig(program=127, is_drum=True).program, 127)

    def test_render_request_and_result_fields_are_validated(self) -> None:
        request_constructor: Callable[..., RenderRequest] = RenderRequest
        result_constructor: Callable[..., RenderResult] = RenderResult
        for output, format_name in (("", "text"), ("x", " "), (None, "text")):
            with self.subTest(output=output, format=format_name), self.assertRaises((TypeError, ValueError)):
                request_constructor(output, format_name)
        for size in (-1, True, 1.5):
            with self.subTest(size=size), self.assertRaises((TypeError, ValueError)):
                result_constructor("x", "text", size)
        self.assertEqual(RenderResult("empty.txt", "text", 0).size_bytes, 0)
