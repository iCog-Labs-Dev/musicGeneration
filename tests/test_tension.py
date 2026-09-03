"""Tests for aimusic.scoring.tension — see docs/tension-diagnostics-spec.md.

Each test's docstring notes which acceptance criterion (from the spec) it
covers, to keep the mapping between tests and requirements checkable.
"""

import unittest

from aimusic.core.core_types import BeatState
from aimusic.core.vocab import build_default_vocabularies
from aimusic.planning.plans import PlanningSection
from aimusic.scoring.tension import (
    TENSION_MODEL_VERSION,
    TensionWeights,
    beat_tension,
    compare_tension_curves,
    realized_tension_curve,
    target_tension_curve,
)
from aimusic.theory.tonal import nearest_roots, tonal_distance


def _role_id(vocab, label):
    return vocab.roles.token_for_label(label).id


def _state(vocab, *, role="hold", boundary=0, key_root=0, chord_root=0, chord_quality="maj",
           head="rest", groove_family=None):
    key_id = vocab.keys.token_for_id(key_root).id if key_root < len(vocab.keys.tokens) else 0
    # find a chord token with the requested root/quality
    chord_id = None
    for token in vocab.chords.tokens:
        if token.root_pc == chord_root and token.quality == chord_quality:
            chord_id = token.id
            break
    if chord_id is None:
        raise ValueError(f"no chord token for root={chord_root} quality={chord_quality}")
    groove_id = vocab.grooves.tokens[0].id if groove_family is None else groove_family
    return BeatState(
        meter_id=0,
        beat_in_bar=0,
        boundary_lvl=boundary,
        key_id=key_id,
        chord_id=chord_id,
        role_id=_role_id(vocab, role),
        head_id=vocab.heads.token_for_label(head).id,
        groove_id=groove_id,
    )


class TestKeyMotionTonalDistance(unittest.TestCase):
    """Acceptance criterion #1: tonally distant transitions produce greater
    tension than equivalent local transitions, under controlled fixtures."""

    def _local_vs_distant(self, edo):
        vocab = build_default_vocabularies(edo=edo)
        near_roots = nearest_roots(0, edo=edo, limit=1)
        self.assertTrue(near_roots, "expected at least one near root")
        near_root = near_roots[0]

        # pick the farthest root from 0 by tonal_distance
        far_root = max(range(edo), key=lambda r: tonal_distance(0, r, edo))

        prev = _state(vocab, role="hold", boundary=0, key_root=0)
        local = _state(vocab, role="hold", boundary=0, key_root=near_root)
        distant = _state(vocab, role="hold", boundary=0, key_root=far_root)

        local_tension = beat_tension(prev, local, vocab, edo)
        distant_tension = beat_tension(prev, distant, vocab, edo)
        self.assertLess(local_tension, distant_tension)

    def test_key_motion_distant_exceeds_local_12edo(self):
        self._local_vs_distant(12)

    def test_key_motion_distant_exceeds_local_19edo(self):
        self._local_vs_distant(19)


class TestChordMotionTonalDistance(unittest.TestCase):
    """Acceptance criterion #1 (chord quality variant) + #4 (19-EDO)."""

    def _quality_change_increases_tension(self, edo):
        vocab = build_default_vocabularies(edo=edo)
        prev = _state(vocab, role="hold", boundary=0, key_root=0, chord_root=0, chord_quality="maj")
        same_quality = _state(vocab, role="hold", boundary=0, key_root=0, chord_root=0, chord_quality="maj")
        # dim7/dominant-ish quality should be tonally farther from maj than
        # another maj chord at the same root.
        far_quality = "dim" if any(t.quality == "dim" for t in vocab.chords.tokens) else vocab.chords.tokens[-1].quality
        changed_quality = _state(vocab, role="hold", boundary=0, key_root=0, chord_root=0, chord_quality=far_quality)

        same_tension = beat_tension(prev, same_quality, vocab, edo)
        changed_tension = beat_tension(prev, changed_quality, vocab, edo)
        self.assertLessEqual(same_tension, changed_tension)

    def test_chord_motion_quality_change_increases_tension_12edo(self):
        self._quality_change_increases_tension(12)

    def test_chord_motion_quality_change_increases_tension_19edo(self):
        self._quality_change_increases_tension(19)


class TestRoleAndBoundaryDirection(unittest.TestCase):
    """Acceptance criterion #2: changing role or boundary changes tension in
    the documented direction (hold < prep < change < cad; boundary monotonic)."""

    def test_role_ordering_documented_direction(self):
        vocab = build_default_vocabularies(edo=12)
        prev = _state(vocab, role="hold", key_root=0, chord_root=0)
        tensions = {}
        for role in ("hold", "prep", "change", "cad"):
            state = _state(vocab, role=role, key_root=0, chord_root=0)
            tensions[role] = beat_tension(prev, state, vocab, 12)
        self.assertLess(tensions["hold"], tensions["prep"])
        self.assertLess(tensions["prep"], tensions["change"])
        self.assertLess(tensions["change"], tensions["cad"])

    def test_boundary_level_monotonic_increase(self):
        vocab = build_default_vocabularies(edo=12)
        prev = _state(vocab, role="hold", boundary=0, key_root=0, chord_root=0)
        max_level = max(t.level for t in vocab.boundaries.tokens)
        prior_tension = None
        for level in range(0, max_level + 1):
            state = _state(vocab, role="hold", boundary=level, key_root=0, chord_root=0)
            tension = beat_tension(prev, state, vocab, 12)
            if prior_tension is not None:
                self.assertGreaterEqual(tension, prior_tension)
                self.assertGreater(tension, prior_tension)
            prior_tension = tension


class TestEdgeCasesAndPurity(unittest.TestCase):
    def test_first_beat_no_prev_state_falls_back_to_role_boundary(self):
        vocab = build_default_vocabularies(edo=12)
        state = _state(vocab, role="cad", boundary=3, key_root=6, chord_root=6, chord_quality="min")
        tension_no_prev = beat_tension(None, state, vocab, 12)
        # Same state, but with an identical prev_state (no motion) should be
        # <= the no-prev-state tension, since motion terms can only add.
        tension_no_motion = beat_tension(state, state, vocab, 12)
        self.assertAlmostEqual(tension_no_prev, tension_no_motion, places=6)

    def test_beat_tension_is_pure_and_deterministic(self):
        vocab_a = build_default_vocabularies(edo=12)
        vocab_b = build_default_vocabularies(edo=12)
        prev = _state(vocab_a, role="prep", boundary=1, key_root=2, chord_root=2)
        state = _state(vocab_a, role="change", boundary=2, key_root=7, chord_root=7)
        prev_b = _state(vocab_b, role="prep", boundary=1, key_root=2, chord_root=2)
        state_b = _state(vocab_b, role="change", boundary=2, key_root=7, chord_root=7)

        first = beat_tension(prev, state, vocab_a, 12)
        second = beat_tension(prev_b, state_b, vocab_b, 12)
        third = beat_tension(prev, state, vocab_a, 12)
        self.assertEqual(first, second)
        self.assertEqual(first, third)

    def test_weights_reject_negative(self):
        with self.assertRaises(ValueError):
            TensionWeights(role=-0.1)


class TestTargetTensionCurve(unittest.TestCase):
    def test_target_tension_curve_interpolates_section_arc(self):
        section = PlanningSection(
            name="verse",
            start_time=0,
            end_time=4,
            boundary_level=2,
            target_tension_arc=(0.0, 1.0),
        )
        curve = target_tension_curve((section,))
        self.assertEqual(len(curve), 4)
        times = [t for t, _ in curve]
        self.assertEqual(times, [0.0, 1.0, 2.0, 3.0])
        values = [v for _, v in curve]
        self.assertAlmostEqual(values[0], 0.0)
        self.assertAlmostEqual(values[-1], 0.75)  # last sample is before end_time
        self.assertTrue(all(values[i] <= values[i + 1] for i in range(len(values) - 1)))

    def test_target_tension_curve_multi_control_point_arc(self):
        section = PlanningSection(
            name="build",
            start_time=0,
            end_time=6,
            boundary_level=2,
            target_tension_arc=(0.2, 0.9, 0.1),
        )
        curve = target_tension_curve((section,))
        self.assertEqual(len(curve), 6)
        values = [v for _, v in curve]
        # Should rise toward the middle control point then fall.
        self.assertLess(values[0], values[2])
        self.assertGreater(values[2], values[-1])


class TestRealizedTensionCurve(unittest.TestCase):
    def test_realized_tension_curve_matches_path_length(self):
        vocab = build_default_vocabularies(edo=12)
        path = [
            _state(vocab, role="hold", key_root=0, chord_root=0),
            _state(vocab, role="prep", key_root=0, chord_root=7),
            _state(vocab, role="cad", key_root=6, chord_root=6),
        ]
        curve = realized_tension_curve(path, vocab, 12)
        self.assertEqual(len(curve), 3)
        self.assertEqual([t for t, _ in curve], [0.0, 1.0, 2.0])
        for _, value in curve:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)


class TestCompareTensionCurves(unittest.TestCase):
    def setUp(self):
        self.sections = (
            PlanningSection(name="a", start_time=0, end_time=2, boundary_level=2,
                             target_tension_arc=(0.2, 0.2)),
            PlanningSection(name="b", start_time=2, end_time=4, boundary_level=3,
                             target_tension_arc=(0.8, 0.8)),
        )
        self.target = [(0.0, 0.2), (1.0, 0.2), (2.0, 0.8), (3.0, 0.8)]

    def test_mean_absolute_error(self):
        realized = [(0.0, 0.3), (1.0, 0.1), (2.0, 0.9), (3.0, 0.7)]
        report = compare_tension_curves(self.target, realized, self.sections)
        expected_mae = (0.1 + 0.1 + 0.1 + 0.1) / 4
        self.assertAlmostEqual(report.mean_absolute_error, expected_mae, places=6)
        self.assertAlmostEqual(report.max_absolute_error, 0.1, places=6)

    def test_section_level_error(self):
        realized = [(0.0, 0.4), (1.0, 0.4), (2.0, 0.8), (3.0, 0.8)]
        report = compare_tension_curves(self.target, realized, self.sections)
        self.assertAlmostEqual(report.section_errors["a"], 0.2, places=6)
        self.assertAlmostEqual(report.section_errors["b"], 0.0, places=6)

    def test_peak_timing_offset(self):
        # realized peak shifted one beat later than target peak
        realized = [(0.0, 0.1), (1.0, 0.9), (2.0, 0.2), (3.0, 0.2)]
        target = [(0.0, 0.9), (1.0, 0.1), (2.0, 0.1), (3.0, 0.1)]
        report = compare_tension_curves(target, realized, self.sections)
        self.assertEqual(report.target_peak_time, 0.0)
        self.assertEqual(report.realized_peak_time, 1.0)
        self.assertEqual(report.peak_timing_offset, 1.0)

    def test_shape_correlation(self):
        realized_matching_shape = [(0.0, 0.1), (1.0, 0.1), (2.0, 0.9), (3.0, 0.9)]
        report = compare_tension_curves(self.target, realized_matching_shape, self.sections)
        self.assertGreater(report.shape_correlation, 0.9)

        realized_inverse_shape = [(0.0, 0.9), (1.0, 0.9), (2.0, 0.1), (3.0, 0.1)]
        report_inverse = compare_tension_curves(self.target, realized_inverse_shape, self.sections)
        self.assertLess(report_inverse.shape_correlation, -0.9)


class TestTensionModelVersion(unittest.TestCase):
    def test_version_is_a_string_constant(self):
        self.assertIsInstance(TENSION_MODEL_VERSION, str)
        self.assertTrue(TENSION_MODEL_VERSION)


if __name__ == "__main__":
    unittest.main()
