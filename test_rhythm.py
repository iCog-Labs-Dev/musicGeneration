import unittest

from rhythm_features import (
    DEFAULT_METERS,
    BeatState,
    MeterSpec,
    W_BOUNDARY_ON_STRONG,
    W_BOUNDARY_ON_WEAK_PENALTY,
    W_GROOVE_CONTINUITY,
    W_ILLEGAL_BEAT_PENALTY,
    W_METER_STABILITY,
    beats_per_bar,
    boundary_score,
    illegal_beat_penalty,
    is_strong_beat,
    local_window_score,
    sequence_score,
    strong_beat_bias,
    transition_score,
)


class TestRhythmHelpers(unittest.TestCase):
    def test_beats_per_bar_defaults(self):
        self.assertEqual(beats_per_bar(0), 4)
        self.assertEqual(beats_per_bar(1), 3)

    def test_beats_per_bar_unknown_falls_back_to_4(self):
        self.assertEqual(beats_per_bar(999), 4)

    def test_is_strong_beat_4_4(self):
        # 4/4: beat 0 and beat 2 are strong
        self.assertTrue(is_strong_beat(0, 4))
        self.assertFalse(is_strong_beat(1, 4))
        self.assertTrue(is_strong_beat(2, 4))
        self.assertFalse(is_strong_beat(3, 4))

    def test_is_strong_beat_3_4(self):
        # 3/4: only beat 0 is strong in this simple heuristic
        self.assertTrue(is_strong_beat(0, 3))
        self.assertFalse(is_strong_beat(1, 3))
        self.assertFalse(is_strong_beat(2, 3))


class TestRhythmScoring(unittest.TestCase):
    def test_boundary_score_strong_vs_weak(self):
        strong = BeatState(0, 0, 2, 0, 0, 0, 0, 0)  # meter 0 => 4/4, beat 0 strong
        weak = BeatState(0, 1, 2, 0, 0, 0, 0, 0)    # beat 1 weak

        self.assertEqual(boundary_score(strong), W_BOUNDARY_ON_STRONG * 2.0)
        self.assertEqual(boundary_score(weak), -W_BOUNDARY_ON_WEAK_PENALTY * 2.0)

    def test_illegal_beat_penalty(self):
        # 4/4 has beats 0..3 valid
        ok = BeatState(0, 3, 0, 0, 0, 0, 0, 0)
        bad = BeatState(0, 4, 0, 0, 0, 0, 0, 0)
        self.assertEqual(illegal_beat_penalty(ok), 0.0)
        self.assertEqual(illegal_beat_penalty(bad), -W_ILLEGAL_BEAT_PENALTY)

    def test_transition_score_meter_and_groove_continuity(self):
        a = BeatState(0, 0, 0, 0, 0, 0, 0, 10)
        b_same = BeatState(0, 1, 0, 0, 0, 0, 0, 10)
        b_diff = BeatState(1, 1, 0, 0, 0, 0, 0, 11)

        s_same = transition_score(a, b_same)
        s_diff = transition_score(a, b_diff)

        # The same-meter/same-groove transition must score higher.
        self.assertGreater(s_same, s_diff)

        # And it must include the positive continuity weights.
        # (We don't assert exact totals because strong-beat bias may differ.)
        self.assertGreaterEqual(s_same, W_METER_STABILITY + W_GROOVE_CONTINUITY)

    def test_local_window_boundary_onset_and_next_strong_bonus(self):
        prev = BeatState(0, 0, 0, 0, 0, 0, 0, 0)   # no boundary
        curr = BeatState(0, 1, 2, 0, 0, 0, 0, 0)   # boundary on weak beat (still allowed)
        next_ = BeatState(0, 2, 0, 0, 0, 0, 0, 0)  # strong beat in 4/4

        # Should be positive because onset_bonus + next_strong are positive.
        self.assertGreater(local_window_score(prev, curr, next_), 0.0)

    def test_sequence_score_additivity(self):
        s0 = BeatState(0, 0, 0, 0, 0, 0, 0, 1)
        s1 = BeatState(0, 1, 1, 0, 0, 0, 0, 1)
        s2 = BeatState(0, 2, 0, 0, 0, 0, 0, 1)

        expected = transition_score(s0, s1) + transition_score(s1, s2) + local_window_score(s0, s1, s2)
        self.assertAlmostEqual(sequence_score([s0, s1, s2]), expected)


if __name__ == "__main__":
    unittest.main()

