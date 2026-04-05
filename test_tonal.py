"""Tests for the tonal module — chord templates, distances, classification.

Covers both 12-EDO and 19-EDO to verify EDO-generic arithmetic.
"""

import unittest

from config import EDOConfig
from edo import EDO
from tonal import (
    ALL_QUALITIES,
    TonalSystem,
    basic_space_distance,
    chord_pitch_classes,
    get_fifth_steps,
    get_fourth_steps,
    get_major_second_steps,
    get_major_third_steps,
    get_minor_second_steps,
    get_minor_third_steps,
    is_dominant,
    is_fifth_down,
    is_fifth_up,
    is_major_second_up,
    is_minorish,
    is_subdominant,
    is_tonic_family,
    pc,
    tonal_distance,
)


# ===================================================================
# 1. Interval helpers
# ===================================================================

class TestIntervalHelpers(unittest.TestCase):
    """Verify EDO-generic interval calculations."""

    def test_pc_wraps_correctly_12edo(self):
        self.assertEqual(pc(0, 12), 0)
        self.assertEqual(pc(12, 12), 0)
        self.assertEqual(pc(15, 12), 3)
        self.assertEqual(pc(-1, 12), 11)

    def test_pc_wraps_correctly_19edo(self):
        self.assertEqual(pc(0, 19), 0)
        self.assertEqual(pc(19, 19), 0)
        self.assertEqual(pc(23, 19), 4)
        self.assertEqual(pc(-1, 19), 18)

    def test_interval_sizes_12edo(self):
        self.assertEqual(get_minor_second_steps(12), 1)
        self.assertEqual(get_major_second_steps(12), 2)
        self.assertEqual(get_minor_third_steps(12), 3)
        self.assertEqual(get_major_third_steps(12), 4)
        self.assertEqual(get_fourth_steps(12), 5)
        self.assertEqual(get_fifth_steps(12), 7)

    def test_interval_sizes_19edo(self):
        """19-EDO has finer divisions; intervals should scale correctly."""
        self.assertEqual(get_minor_second_steps(19), 2)   # round(1*19/12) = 2
        self.assertEqual(get_major_second_steps(19), 3)   # round(2*19/12) = 3
        self.assertEqual(get_minor_third_steps(19), 5)    # round(3*19/12) = 5
        self.assertEqual(get_major_third_steps(19), 6)    # round(4*19/12) = 6
        self.assertEqual(get_fourth_steps(19), 8)         # round(5*19/12) = 8
        self.assertEqual(get_fifth_steps(19), 11)         # round(7*19/12) = 11

    def test_is_fifth_down_12edo(self):
        self.assertTrue(is_fifth_down(7, 0, edo=12))   # G -> C
        self.assertTrue(is_fifth_down(2, 7, edo=12))   # D -> G
        self.assertFalse(is_fifth_down(0, 7, edo=12))  # C -> G is fifth UP

    def test_is_fifth_up_12edo(self):
        self.assertTrue(is_fifth_up(5, 0, edo=12))     # F -> C
        self.assertTrue(is_fifth_up(0, 7, edo=12))     # C -> G
        self.assertFalse(is_fifth_up(7, 0, edo=12))    # G -> C is fifth DOWN

    def test_is_fifth_down_19edo(self):
        self.assertTrue(is_fifth_down(11, 0, edo=19))  # "G" -> "C" in 19-EDO

    def test_is_fifth_up_19edo(self):
        self.assertTrue(is_fifth_up(8, 0, edo=19))     # "F" -> "C" in 19-EDO

    def test_is_major_second_up_12edo(self):
        self.assertTrue(is_major_second_up(7, 9, edo=12))   # G -> A
        self.assertTrue(is_major_second_up(0, 2, edo=12))   # C -> D
        self.assertFalse(is_major_second_up(0, 1, edo=12))  # C -> C# is semitone

    def test_is_major_second_up_19edo(self):
        self.assertTrue(is_major_second_up(11, 14, edo=19))  # 3 steps in 19-EDO


# ===================================================================
# 2. Chord templates
# ===================================================================

class TestChordTemplates(unittest.TestCase):
    """Verify chord-to-pitch-class expansion."""

    def test_major_triad_root_0_12edo(self):
        pcs = chord_pitch_classes(0, "maj", edo=12)
        self.assertEqual(pcs, frozenset({0, 4, 7}))

    def test_minor_triad_root_0_12edo(self):
        pcs = chord_pitch_classes(0, "min", edo=12)
        self.assertEqual(pcs, frozenset({0, 3, 7}))

    def test_dom7_root_7_12edo(self):
        """G7 = {G, B, D, F} = {7, 11, 2, 5}."""
        pcs = chord_pitch_classes(7, "7", edo=12)
        self.assertEqual(pcs, frozenset({7, 11, 2, 5}))

    def test_maj7_root_0_12edo(self):
        """Cmaj7 = {C, E, G, B} = {0, 4, 7, 11}."""
        pcs = chord_pitch_classes(0, "maj7", edo=12)
        self.assertEqual(pcs, frozenset({0, 4, 7, 11}))

    def test_m7_root_2_12edo(self):
        """Dm7 = {D, F, A, C} = {2, 5, 9, 0}."""
        pcs = chord_pitch_classes(2, "m7", edo=12)
        self.assertEqual(pcs, frozenset({2, 5, 9, 0}))

    def test_dim7_root_0_12edo(self):
        """Cdim7 = {C, Eb, Gb, Bbb} = {0, 3, 6, 9}."""
        pcs = chord_pitch_classes(0, "dim7", edo=12)
        self.assertEqual(pcs, frozenset({0, 3, 6, 9}))

    def test_transposition_preserves_cardinality(self):
        """A chord transposed to any root should keep the same size."""
        for root in range(12):
            pcs = chord_pitch_classes(root, "maj7", edo=12)
            self.assertEqual(len(pcs), 4)

    def test_major_triad_root_0_19edo(self):
        """In 19-EDO, major third = 6 steps, fifth = 11 steps."""
        pcs = chord_pitch_classes(0, "maj", edo=19)
        self.assertEqual(pcs, frozenset({0, 6, 11}))

    def test_minor_triad_root_0_19edo(self):
        """In 19-EDO, minor third = 5 steps, fifth = 11 steps."""
        pcs = chord_pitch_classes(0, "min", edo=19)
        self.assertEqual(pcs, frozenset({0, 5, 11}))

    def test_dom7_root_0_19edo(self):
        """In 19-EDO: maj3=6, fifth=11, minor7=round(10*19/12)=16."""
        pcs = chord_pitch_classes(0, "7", edo=19)
        self.assertEqual(pcs, frozenset({0, 6, 11, 16}))

    def test_unknown_quality_raises(self):
        with self.assertRaises(ValueError):
            chord_pitch_classes(0, "nonexistent", edo=12)

    def test_all_qualities_are_valid(self):
        """Every quality in ALL_QUALITIES should expand without error."""
        for q in ALL_QUALITIES:
            pcs = chord_pitch_classes(0, q, edo=12)
            self.assertGreater(len(pcs), 0)


# ===================================================================
# 3. Functional classification
# ===================================================================

class TestFunctionalClassification(unittest.TestCase):

    def test_dominant_family(self):
        for q in ("7", "9", "13", "7b9", "7#11", "sus7"):
            self.assertTrue(is_dominant(q), f"{q} should be dominant")

    def test_non_dominant(self):
        for q in ("maj", "min", "m7", "dim", "sus4"):
            self.assertFalse(is_dominant(q), f"{q} should NOT be dominant")

    def test_tonic_family(self):
        for q in ("maj", "maj7", "6", "69", "maj9", "min", "m6", "m69"):
            self.assertTrue(is_tonic_family(q), f"{q} should be tonic-family")

    def test_non_tonic(self):
        for q in ("7", "dim", "m7b5"):
            self.assertFalse(is_tonic_family(q), f"{q} should NOT be tonic")

    def test_minorish(self):
        for q in ("min", "m7", "m9", "m11", "m6", "m69", "m7b5"):
            self.assertTrue(is_minorish(q), f"{q} should be minorish")

    def test_subdominant(self):
        for q in ("sus4", "sus2"):
            self.assertTrue(is_subdominant(q), f"{q} should be subdominant")

    def test_non_subdominant(self):
        for q in ("7", "dim", "aug"):
            self.assertFalse(is_subdominant(q), f"{q} should NOT be subdominant")


# ===================================================================
# 4. Tonal distance (circle of fifths, j-component)
# ===================================================================

class TestTonalDistance(unittest.TestCase):

    def test_same_root_is_zero(self):
        self.assertEqual(tonal_distance(0, 0, edo=12), 0)

    def test_fifth_is_one_step(self):
        """C (0) and G (7) are one fifth apart."""
        self.assertEqual(tonal_distance(0, 7, edo=12), 1)

    def test_tritone_is_maximum(self):
        """C (0) and F# (6) are maximally far on the circle of fifths."""
        self.assertEqual(tonal_distance(0, 6, edo=12), 6)

    def test_symmetry(self):
        self.assertEqual(
            tonal_distance(0, 7, edo=12),
            tonal_distance(7, 0, edo=12),
        )

    def test_near_vs_far(self):
        """Closely related keys should have smaller distance."""
        dist_fifth = tonal_distance(0, 7, edo=12)   # C-G
        dist_tritone = tonal_distance(0, 6, edo=12)  # C-F#
        self.assertLess(dist_fifth, dist_tritone)

    def test_19edo_same_root_zero(self):
        self.assertEqual(tonal_distance(0, 0, edo=19), 0)

    def test_19edo_fifth_is_one_step(self):
        """In 19-EDO, the fifth is 11 steps."""
        self.assertEqual(tonal_distance(0, 11, edo=19), 1)


# ===================================================================
# 5. Basic space distance (Lerdahl's δ = j + k)
# ===================================================================

class TestBasicSpaceDistance(unittest.TestCase):

    def test_same_chord_is_zero(self):
        d = basic_space_distance(0, "maj", 0, "maj", edo=12)
        self.assertEqual(d, 0.0)

    def test_same_root_different_quality(self):
        """C maj vs C min — root and fifth are shared, only third differs."""
        d = basic_space_distance(0, "maj", 0, "min", edo=12)
        # j = 0 (same root)
        # level a: {0} vs {0} → 0
        # level b: {0, 7} vs {0, 7} → 0
        # level c: {0,4,7} vs {0,3,7} → sym_diff = {3,4} → 2
        # δ = 0 + 0 + 0 + 2 = 2
        self.assertEqual(d, 2.0)

    def test_V_I_closer_than_bII_I(self):
        """V → I (G maj → C maj) should be closer than bII → I (Db maj → C maj)."""
        d_vi = basic_space_distance(7, "maj", 0, "maj", edo=12)
        d_bii = basic_space_distance(1, "maj", 0, "maj", edo=12)
        self.assertLess(d_vi, d_bii)

    def test_unknown_quality_falls_back(self):
        """Unknown quality should return tonal_distance alone."""
        d = basic_space_distance(0, "???", 7, "maj", edo=12)
        self.assertEqual(d, tonal_distance(0, 7, edo=12))

    def test_symmetry(self):
        d_ab = basic_space_distance(0, "maj7", 7, "7", edo=12)
        d_ba = basic_space_distance(7, "7", 0, "maj7", edo=12)
        self.assertEqual(d_ab, d_ba)

    def test_19edo_basic(self):
        """Basic space distance should also work in 19-EDO."""
        d = basic_space_distance(0, "maj", 0, "maj", edo=19)
        self.assertEqual(d, 0.0)

    def test_19edo_V_I(self):
        """V → I in 19-EDO (root 11 → root 0) should have small distance."""
        d = basic_space_distance(11, "maj", 0, "maj", edo=19)
        # j = 1 (one fifth)
        self.assertGreater(d, 0.0)
        # Should be less than tritone distance
        d_far = basic_space_distance(9, "maj", 0, "maj", edo=19)
        self.assertLess(d, d_far)


# ===================================================================
# 6. TonalSystem class — integration
# ===================================================================

class TestTonalSystemClass(unittest.TestCase):

    def setUp(self):
        self.ts12 = TonalSystem(EDO(EDOConfig(n=12)))
        self.ts19 = TonalSystem(EDO(EDOConfig(n=19)))

    def test_repr(self):
        self.assertIn("12", repr(self.ts12))

    def test_available_qualities_non_empty(self):
        quals = self.ts12.available_qualities()
        self.assertGreater(len(quals), 10)
        self.assertIn("maj7", quals)
        self.assertIn("m7", quals)

    def test_chord_pcs_delegates(self):
        """TonalSystem.chord_pcs should match the standalone function."""
        pcs_standalone = chord_pitch_classes(0, "maj7", edo=12)
        pcs_class = self.ts12.chord_pcs(0, "maj7")
        self.assertEqual(pcs_standalone, pcs_class)

    def test_distance_delegates(self):
        """TonalSystem.distance should match basic_space_distance."""
        d_standalone = basic_space_distance(0, "maj", 7, "7", edo=12)
        d_class = self.ts12.distance(0, "maj", 7, "7")
        self.assertEqual(d_standalone, d_class)

    def test_fifths_distance_delegates(self):
        d_standalone = tonal_distance(0, 7, edo=12)
        d_class = self.ts12.fifths_distance(0, 7)
        self.assertEqual(d_standalone, d_class)

    def test_classify_dominant(self):
        self.assertEqual(self.ts12.classify("7"), "dominant")
        self.assertEqual(self.ts12.classify("13"), "dominant")

    def test_classify_tonic(self):
        self.assertEqual(self.ts12.classify("maj7"), "tonic")
        self.assertEqual(self.ts12.classify("maj"), "tonic")

    def test_classify_diminished(self):
        self.assertEqual(self.ts12.classify("dim"), "diminished")
        self.assertEqual(self.ts12.classify("dim7"), "diminished")
        self.assertEqual(self.ts12.classify("m7b5"), "diminished")

    def test_classify_minor(self):
        # m7 is minorish but not tonic-family
        # Actually m7 is in _MINOR_QUALITIES but let's check classification
        # m7b5 → diminished takes priority; m7 → minor
        self.assertEqual(self.ts12.classify("m9"), "minor")

    def test_classify_unknown(self):
        self.assertEqual(self.ts12.classify("???"), "unknown")

    def test_19edo_chord_pcs(self):
        pcs = self.ts19.chord_pcs(0, "maj")
        self.assertEqual(pcs, frozenset({0, 6, 11}))

    def test_19edo_distance(self):
        d = self.ts19.distance(0, "maj", 0, "maj")
        self.assertEqual(d, 0.0)


if __name__ == "__main__":
    unittest.main()
