"""Asset-free, target-independent controlled-tone evidence (not native acceptance)."""
from __future__ import annotations

import json
import unittest
from unittest.mock import PropertyMock, patch

import numpy as np

from aimusic.audio.scoring.tuning_check import (
    ToneWindow, TuningCheckConfig, TuningCheckError, TuningReport,
    TuningWindowResult, analyze_tuning,
)


SR = 16000


def sine(hz: float, *, seconds: float = 0.5, phase: float = 0.31) -> np.ndarray:
    return 0.25 * np.sin(2 * np.pi * hz * np.arange(round(SR * seconds)) / SR + phase)


class TestTuningAnalysis(unittest.TestCase):
    def test_exact_12_and_19_edo_tones_report_original_targets_deterministically(self):
        for hz in (55.0, 110.0, 440.0, 440 * 2 ** (1 / 19), 440 * 2 ** (4 / 19), 1760.0):
            with self.subTest(hz=hz):
                audio = sine(hz)
                original = audio.copy()
                windows = (ToneWindow('reference', 400, len(audio) - 400, hz),)
                report = analyze_tuning(audio, SR, windows)
                row = report.windows[0]
                self.assertEqual(row.window, windows[0])
                self.assertAlmostEqual(row.measured_hz, hz, delta=hz * 0.0001)
                self.assertGreater(row.confidence, 0.99)
                self.assertEqual(row.coverage, 1.0)
                self.assertIsNone(row.exclusion)
                self.assertLess(row.absolute_cents, 0.1)
                self.assertEqual(report.coverage, 1.0)
                self.assertLess(report.mean_abs_cents, 0.1)
                self.assertLess(report.p95_abs_cents, 0.1)
                self.assertEqual(report.to_dict(), analyze_tuning(audio, SR, windows).to_dict())
                np.testing.assert_array_equal(audio, original)
                json.dumps(report.to_dict(), allow_nan=False)
                self.assertEqual(report.to_dict()['polyphonic_accuracy'], 'unmeasured')

    def test_detuning_is_measured_against_intent_not_nearest_lattice_or_output(self):
        for cents in (-23.0, 23.0, 1200.0):
            with self.subTest(cents=cents):
                hz = 440 * 2 ** (cents / 1200)
                audio = sine(hz)
                row = analyze_tuning(audio, SR, (ToneWindow('tone', 0, len(audio), 440),)).windows[0]
                self.assertAlmostEqual(row.absolute_cents, abs(cents), delta=0.1)
                other = analyze_tuning(audio, SR, (ToneWindow('tone', 0, len(audio), 400),)).windows[0]
                self.assertEqual(row.measured_hz, other.measured_hz)
                self.assertEqual(row.confidence, other.confidence)

    def test_fundamental_is_not_the_louder_second_harmonic(self):
        audio = sine(220) + 2 * sine(440)
        report = analyze_tuning(audio, SR, (ToneWindow('harmonics', 0, len(audio), 220),))
        self.assertLess(report.mean_abs_cents, 0.1)

    def test_dc_offset_and_amplitude_do_not_change_pitch(self):
        audio = 0.01 * sine(330) + 0.5
        report = analyze_tuning(audio, SR, (ToneWindow('offset', 0, len(audio), 330),))
        self.assertLess(report.mean_abs_cents, 0.1)

    def test_silence_noise_short_and_truncated_windows_are_missing_not_zero_error(self):
        fixtures = (
            (np.zeros(8000), 8000, 'silence'),
            (np.full(8000, 0.5), 8000, 'silence'),
            (np.random.default_rng(9).normal(0, 0.1, 8000), 8000, 'insufficient_confidence'),
            (sine(440, seconds=0.002), 32, 'insufficient_samples'),
            (sine(440), 9000, 'insufficient_samples'),
            (np.array([float('nan')] * 8000), 8000, 'nonfinite_samples'),
        )
        for audio, end, reason in fixtures:
            with self.subTest(reason=reason):
                report = analyze_tuning(audio, SR, (ToneWindow('missing', 0, end, 440),))
                row = report.windows[0]
                self.assertIsNone(row.measured_hz)
                self.assertIsNone(row.absolute_cents)
                self.assertEqual(row.exclusion, reason)
                self.assertEqual(row.coverage, 0.0)
                self.assertEqual(report.coverage, 0.0)
                self.assertIsNone(report.mean_abs_cents)
                self.assertIsNone(report.p95_abs_cents)
                json.dumps(report.to_dict(), allow_nan=False)

    def test_partial_coverage_does_not_pollute_measured_statistics_with_zero(self):
        audio = np.concatenate((sine(440 * 2 ** (20 / 1200)), np.zeros(8000)))
        report = analyze_tuning(audio, SR, (
            ToneWindow('detuned', 0, 8000, 440), ToneWindow('missing', 8000, 16000, 440),
        ))
        self.assertEqual(report.coverage, 0.5)
        self.assertAlmostEqual(report.mean_abs_cents, 20, delta=0.1)
        self.assertAlmostEqual(report.p95_abs_cents, 20, delta=0.1)

    def test_percussion_is_explicitly_excluded_from_frequency_statistics(self):
        audio = np.concatenate((sine(440), sine(999)))
        report = analyze_tuning(audio, SR, (
            ToneWindow('pitched', 0, 8000, 440),
            ToneWindow('GM drum key 36', 8000, 16000, None, percussion=True),
        ))
        self.assertEqual(report.coverage, 1.0)
        self.assertLess(report.mean_abs_cents, 0.1)
        drum = report.windows[1]
        self.assertEqual(drum.exclusion, 'percussion')
        self.assertIsNone(drum.measured_hz)
        self.assertIsNone(drum.absolute_cents)
        self.assertEqual(drum.coverage, 0.0)

    def test_invalid_input_is_not_silently_coerced_to_mono_or_inferred_targets(self):
        for hz in (None, 0, -1, float('nan'), float('inf')):
            with self.subTest(hz=hz), self.assertRaisesRegex(ValueError, 'expected_hz'):
                ToneWindow('bad target', 0, 8000, hz)
        for start, stop in ((-1, 10), (10, 10), (11, 10), (0.5, 10), (0, True)):
            with self.subTest(start=start, stop=stop), self.assertRaisesRegex(ValueError, 'sample'):
                ToneWindow('bad window', start, stop, 440)
        for audio in (np.zeros((10, 2)), np.zeros((1, 10)), np.array([1j])):
            with self.assertRaisesRegex(ValueError, 'mono|real'):
                analyze_tuning(audio, SR, ())
        for sr in (0, -1, 16000.5, True):
            with self.subTest(sr=sr), self.assertRaisesRegex(ValueError, 'sample_rate'):
                analyze_tuning(sine(440), sr, ())

    def test_negative_correlation_peaks_are_missing_evidence_not_an_exception(self):
        audio = np.random.default_rng(376).normal(0, 0.1, 64)
        report = analyze_tuning(audio, SR, (ToneWindow('aperiodic', 0, 64, 440),))
        self.assertFalse(report.passed)
        self.assertEqual(report.windows[0].exclusion, 'insufficient_confidence')

    def test_extreme_finite_intent_is_not_overflow_or_nan_evidence(self):
        report = analyze_tuning(sine(440), SR, (ToneWindow('wrong target', 0, 8000, 5e-324),))
        self.assertFalse(report.passed)
        self.assertTrue(np.isfinite(report.mean_abs_cents))
        self.assertTrue(np.isfinite(report.p95_abs_cents))
        json.dumps(report.to_dict(), allow_nan=False)

    def test_policy_flags_must_be_actual_booleans(self):
        for flag in ('false', None, 0, 1):
            with self.subTest(flag=flag):
                with self.assertRaisesRegex(ValueError, 'percussion'):
                    ToneWindow('drum', 0, 8000, None, percussion=flag)
                with self.assertRaisesRegex(ValueError, 'controlled'):
                    analyze_tuning(sine(440), SR, (), controlled=flag)

    def test_native_sample_rates_and_multiple_phases_with_known_19_edo_tone(self):
        hz = 440 * 2 ** (4 / 19)
        for sr in (44100, 48000):
            for phase in (0, 1.3, 3.1):
                with self.subTest(sr=sr, phase=phase):
                    audio = 0.2 * np.sin(2 * np.pi * hz * np.arange(sr // 2) / sr + phase)
                    report = analyze_tuning(audio, sr, (ToneWindow('tone', 0, len(audio), hz),))
                    report.require_pass()
                    self.assertLess(report.mean_abs_cents, 0.1)

    def test_outside_analysis_range_or_too_few_cycles_never_pass(self):
        for hz, seconds in ((20, 0.5), (3000, 0.5), (55, 0.04)):
            with self.subTest(hz=hz, seconds=seconds):
                audio = sine(hz, seconds=seconds)
                report = analyze_tuning(audio, SR, (ToneWindow('unsupported', 0, len(audio), hz),))
                self.assertFalse(report.passed)
                self.assertEqual(report.coverage, 0)

    def test_high_19_edo_and_half_sample_periods_are_not_wrong_subharmonics(self):
        for hz in (440 * 2 ** (40 / 19), SR / 8.5, SR / 9.5):
            for phase in (0, 0.2, 1.3, np.pi / 2, 3.1, 5.8):
                with self.subTest(hz=hz, phase=phase):
                    audio = sine(hz, phase=phase)
                    report = analyze_tuning(audio, SR, (ToneWindow('high tone', 0, len(audio), hz),))
                    report.require_pass()
                    self.assertEqual(report.coverage, 1)
                    self.assertLess(report.mean_abs_cents, 0.1)
                    self.assertGreaterEqual(report.windows[0].confidence, 0.9)
                    # Targets still have no influence on frequency/confidence selection.
                    other = analyze_tuning(audio, SR, (ToneWindow('wrong intent', 0, len(audio), 440),))
                    self.assertEqual(report.windows[0].measured_hz, other.windows[0].measured_hz)
                    self.assertEqual(report.windows[0].confidence, other.windows[0].confidence)
                    self.assertFalse(other.passed)

    def test_inclusive_frequency_boundaries_keep_unclamped_measurements(self):
        excursions = []
        for hz, durations in ((40, (0.1, 0.25, 0.5)), (2000, (0.004, 0.25, 0.5))):
            for seconds in durations:
                for phase in (0, 0.2, 1.3, np.pi / 2, 3.1, 5.8):
                    with self.subTest(hz=hz, seconds=seconds, phase=phase):
                        audio = sine(hz, seconds=seconds, phase=phase)
                        windows = (ToneWindow('boundary', 0, len(audio), hz),)
                        report = analyze_tuning(audio, SR, windows)
                        report.require_pass()
                        # Wider bounds expose the original estimator value: never clamp.
                        wide = analyze_tuning(audio, SR, windows, config=TuningCheckConfig(min_hz=39, max_hz=2001))
                        self.assertEqual(report.windows[0].measured_hz, wide.windows[0].measured_hz)
                        self.assertEqual(report.mean_abs_cents, wide.mean_abs_cents)
                        measured = report.windows[0].measured_hz
                        excursions.append(measured < 40 or measured > 2000)
        self.assertTrue(any(excursions))

    def test_genuinely_out_of_range_tones_are_not_admitted_by_boundary_tolerance(self):
        # Half-second windows have a 2 Hz unpadded bin; 0.01 Hz is beyond
        # the explicit 0.002-bin uncertainty (0.004 Hz), even though <5 cents.
        for hz in (39.99, 2000.01):
            for phase in (0, 0.2, 1.3, np.pi / 2, 3.1, 5.8):
                with self.subTest(hz=hz, phase=phase):
                    report = analyze_tuning(sine(hz, phase=phase), SR, (ToneWindow('outside', 0, 8000, hz),))
                    self.assertFalse(report.passed)
                    self.assertEqual(report.coverage, 0)
                    self.assertEqual(report.windows[0].exclusion, 'out_of_range')

    def test_range_uncertainty_does_not_relax_configured_cents_floor(self):
        report = analyze_tuning(sine(40), SR, (ToneWindow('boundary', 0, 8000, 40),))
        report.require_pass()
        error = report.mean_abs_cents
        self.assertGreater(error, 1e-6)
        strict = TuningReport(report.windows, SR, TuningCheckConfig(max_tone_cents=error / 2))
        self.assertFalse(strict.passed)


def exact_report(cents: list[float], *, controlled: bool = True,
                 config: TuningCheckConfig = TuningCheckConfig()) -> TuningReport:
    """Exact frequency observations isolate gate boundaries from DSP resolution."""
    rows = tuple(TuningWindowResult(ToneWindow(str(i), 0, 8000, 440),
                                   440 * 2 ** (c / 1200), 1.0)
                 for i, c in enumerate(cents))
    return TuningReport(rows, SR, config, controlled=controlled)


class TestTuningAcceptance(unittest.TestCase):
    def test_controlled_five_cent_boundary_inclusive_on_both_sides(self):
        for cents, passed in ((-5.000001, False), (-5, True), (0, True), (5, True), (5.000001, False)):
            with self.subTest(cents=cents):
                report = exact_report([cents])
                self.assertEqual(report.passed, passed)
                self.assertEqual(report.to_dict()['passed'], passed)
                if passed:
                    report.require_pass()
                else:
                    with self.assertRaisesRegex(TuningCheckError, 'tone') as caught:
                        report.require_pass()
                    self.assertIs(caught.exception.report, report)

    def test_synthesized_tones_on_either_side_of_each_floor(self):
        for controlled, limit in ((True, 5), (False, 15)):
            for sign in (-1, 1):
                for delta, passed in ((-0.1, True), (0.1, False)):
                    with self.subTest(controlled=controlled, sign=sign, delta=delta):
                        audio = sine(440 * 2 ** (sign * (limit + delta) / 1200))
                        report = analyze_tuning(audio, SR, (ToneWindow('tone', 0, len(audio), 440),),
                                                controlled=controlled)
                        self.assertEqual(report.passed, passed)

    def test_one_bad_controlled_tone_cannot_hide_in_good_mean_and_p95(self):
        report = exact_report([0.0] * 99 + [100.0])
        self.assertLess(report.mean_abs_cents, 15)
        self.assertLess(report.p95_abs_cents, 15)
        self.assertFalse(report.passed)
        self.assertTrue(any('99' in reason and 'tone' in reason for reason in report.failures))

    def test_general_mean_and_p95_fifteen_cent_boundaries(self):
        for cents, passed in ((-15.000001, False), (-15, True), (15, True), (15.000001, False)):
            with self.subTest(cents=cents):
                report = exact_report([cents], controlled=False)
                self.assertEqual(report.passed, passed)
        report = exact_report([0.0] * 18 + [20.0, 20.0], controlled=False)
        self.assertAlmostEqual(report.mean_abs_cents, 2)
        self.assertAlmostEqual(report.p95_abs_cents, 20)
        self.assertFalse(report.passed)
        self.assertTrue(any('p95' in reason for reason in report.failures))
        self.assertFalse(any('mean' in reason for reason in report.failures))

    def test_stricter_configured_limits_are_honored_independently(self):
        for field, values, controlled in (
            ('max_tone_cents', [2.0], True),
            ('max_mean_cents', [1.0, 3.0], False),
            ('max_p95_cents', [0.0, 2.0], False),
        ):
            boundary = 1.9 if field == 'max_p95_cents' else 2.0
            for limit, passed in ((boundary, True), (boundary - 0.000001, False)):
                with self.subTest(field=field, limit=limit):
                    report = exact_report(values, controlled=controlled,
                                          config=TuningCheckConfig(**{field: limit}))
                    self.assertEqual(report.passed, passed)
        self.assertTrue(exact_report([0], config=TuningCheckConfig(max_tone_cents=0)).passed)

    def test_limits_cannot_be_relaxed_or_nonfinite(self):
        for field, ceiling in (('max_tone_cents', 5), ('max_mean_cents', 15), ('max_p95_cents', 15)):
            for value in (-1, ceiling + 0.001, float('nan'), float('inf')):
                with self.subTest(field=field, value=value), self.assertRaisesRegex(ValueError, field):
                    TuningCheckConfig(**{field: value})
        for field, value in (('min_hz', 0), ('max_hz', 30), ('min_confidence', 0),
                             ('min_confidence', 1.01), ('min_rms', 0)):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                TuningCheckConfig(**{field: value})

    def test_flattened_19_edo_corruption_breaches_general_floors_using_original_targets(self):
        # Steps 1, 4 and 7 lie 36.84, 47.37 and 42.11 cents off the 12-TET grid.
        expected = [440 * 2 ** (step / 19) for step in (1, 4, 7)]
        flattened = [440 * 2 ** (round(12 * np.log2(hz / 440)) / 12) for hz in expected]
        windows = tuple(ToneWindow(str(i), i * 8000, (i + 1) * 8000, hz)
                        for i, hz in enumerate(expected))
        good = analyze_tuning(np.concatenate([sine(hz) for hz in expected]), SR, windows)
        good.require_pass()
        bad_audio = np.concatenate([sine(hz) for hz in flattened])
        for controlled in (True, False):
            with self.subTest(controlled=controlled):
                bad = analyze_tuning(bad_audio, SR, windows, controlled=controlled)
                self.assertEqual(tuple(row.window.expected_hz for row in bad.windows), tuple(expected))
                self.assertEqual(bad.coverage, 1.0)
                self.assertGreater(bad.mean_abs_cents, 35)
                self.assertGreater(bad.p95_abs_cents, 40)
                with self.assertRaises(TuningCheckError):
                    bad.require_pass()

    def test_missing_evidence_always_fails_including_empty_and_percussion_only(self):
        for windows in ((), (ToneWindow('drums', 0, 8000, None, percussion=True),),
                        (ToneWindow('silent', 0, 8000, 440),)):
            for controlled in (True, False):
                with self.subTest(windows=windows, controlled=controlled):
                    report = analyze_tuning(np.zeros(8000), SR, windows, controlled=controlled)
                    self.assertFalse(report.passed)
                    with self.assertRaisesRegex(TuningCheckError, 'evidence'):
                        report.require_pass()
        partial = analyze_tuning(np.concatenate((sine(440), np.zeros(8000))), SR, (
            ToneWindow('good', 0, 8000, 440), ToneWindow('missing', 8000, 16000, 440),
        ))
        self.assertLess(partial.mean_abs_cents, 0.1)
        with self.assertRaisesRegex(TuningCheckError, 'missing.*silence'):
            partial.require_pass()

    def test_low_confidence_never_passes_and_stricter_confidence_is_honored(self):
        window = ToneWindow('tone', 0, 8000, 440)
        noisy = analyze_tuning(np.random.default_rng(6).normal(0, 0.1, 8000), SR, (window,))
        with self.assertRaisesRegex(TuningCheckError, 'insufficient_confidence'):
            noisy.require_pass()
        clean = analyze_tuning(sine(440), SR, (window,))
        clean.require_pass()
        strict = analyze_tuning(sine(440), SR, (window,), config=TuningCheckConfig(min_confidence=1))
        self.assertFalse(strict.passed)
        self.assertIsNone(strict.mean_abs_cents)


class TestTuningEvidenceValidation(unittest.TestCase):
    def test_direct_measurements_must_be_finite_positive_numbers(self):
        window = ToneWindow('tone', 0, 8000, 440)
        for hz in (float('nan'), float('inf'), -float('inf'), 0, -1, True, '440'):
            with self.subTest(hz=hz), self.assertRaisesRegex(ValueError, 'measured_hz'):
                TuningWindowResult(window, hz, 1.0)
        for hz in (True, '440'):
            with self.subTest(target=hz), self.assertRaisesRegex(ValueError, 'expected_hz'):
                ToneWindow('tone', 0, 8000, hz)

    def test_direct_confidence_must_be_finite_and_bounded(self):
        window = ToneWindow('tone', 0, 8000, 440)
        for confidence in (float('nan'), float('inf'), -0.001, 1.001, True, '1'):
            with self.subTest(confidence=confidence), self.assertRaisesRegex(ValueError, 'confidence'):
                TuningWindowResult(window, 440, confidence)

    def test_missing_measurements_and_percussion_require_consistent_exclusions(self):
        tone = ToneWindow('tone', 0, 8000, 440)
        drum = ToneWindow('drum', 0, 8000, None, percussion=True)
        for window, hz, confidence, reason in (
            (tone, None, 0, None), (tone, None, 0, ''), (tone, 440, 1, 'silence'),
            (tone, None, 0, 'percussion'), (drum, 440, 1, None),
            (drum, None, 0, 'silence'), (drum, None, 1, 'percussion'),
        ):
            with self.subTest(window=window, hz=hz, reason=reason), self.assertRaisesRegex(ValueError, 'exclusion|percussion'):
                TuningWindowResult(window, hz, confidence, reason)

    def test_report_independently_applies_confidence_including_zero(self):
        window = ToneWindow('tone', 0, 8000, 440)
        for minimum in (0.9, 0.99):
            for confidence, passed in ((0, False), (minimum - 1e-8, False), (minimum, True), (1, True)):
                for controlled in (False, True):
                    with self.subTest(minimum=minimum, confidence=confidence, controlled=controlled):
                        row = TuningWindowResult(window, 440, confidence)
                        report = TuningReport((row,), SR, TuningCheckConfig(min_confidence=minimum), controlled)
                        self.assertEqual(report.passed, passed)
                        self.assertEqual(report.coverage, float(passed))
                        if not passed:
                            self.assertIsNone(report.mean_abs_cents)
                            self.assertIsNone(report.p95_abs_cents)
                            with self.assertRaisesRegex(TuningCheckError, 'confidence'):
                                report.require_pass()

    def test_good_statistics_cannot_hide_an_unreliable_direct_observation(self):
        window = ToneWindow('tone', 0, 8000, 440)
        report = TuningReport((TuningWindowResult(window, 440, 1),
                               TuningWindowResult(window, 440, 0)), SR, TuningCheckConfig())
        self.assertEqual(report.mean_abs_cents, 0)
        self.assertEqual(report.p95_abs_cents, 0)
        self.assertEqual(report.coverage, 0.5)
        self.assertEqual(report.to_dict()['windows'][1]['coverage'], 0)
        with self.assertRaisesRegex(TuningCheckError, 'confidence'):
            report.require_pass()

    def test_direct_report_validates_settings_and_nested_evidence(self):
        row = TuningWindowResult(ToneWindow('tone', 0, 8000, 440), 440, 1)
        for sr in (0, -1, True, 16000.5, 4000):
            with self.subTest(sr=sr), self.assertRaisesRegex(ValueError, 'sample_rate|Nyquist'):
                TuningReport((row,), sr, TuningCheckConfig())
        for controlled in ('false', 0, None):
            with self.subTest(controlled=controlled), self.assertRaisesRegex(ValueError, 'controlled'):
                TuningReport((row,), SR, TuningCheckConfig(), controlled)
        with self.assertRaisesRegex(ValueError, 'config'):
            TuningReport((row,), SR, {})
        with self.assertRaisesRegex(ValueError, 'window'):
            TuningReport(({},), SR, TuningCheckConfig())
        # Even a result corrupted after construction cannot become report evidence.
        object.__setattr__(row, 'measured_hz', float('nan'))
        with self.assertRaisesRegex(ValueError, 'measured_hz'):
            TuningReport((row,), SR, TuningCheckConfig())
        target = ToneWindow('tone', 0, 8000, 440)
        object.__setattr__(target, 'expected_hz', float('nan'))
        with self.assertRaisesRegex(ValueError, 'expected_hz'):
            TuningWindowResult(target, 440, 1)

    def test_direct_out_of_range_evidence_cannot_pass(self):
        for hz in (39.99, 2000.01, 3000):
            with self.subTest(hz=hz):
                row = TuningWindowResult(ToneWindow('outside', 0, 8000, hz), hz, 1)
                report = TuningReport((row,), SR, TuningCheckConfig())
                self.assertFalse(report.passed)
                self.assertEqual(report.coverage, 0)
                with self.assertRaisesRegex(TuningCheckError, 'out_of_range'):
                    report.require_pass()

    def test_nonfinite_aggregate_metrics_never_pass(self):
        # Exercise the gate independently of valid observations/constructor checks.
        report = exact_report([0])
        for field in ('mean_abs_cents', 'p95_abs_cents'):
            for value in (None, float('nan'), float('inf'), -float('inf')):
                with self.subTest(field=field, value=value):
                    with patch.object(TuningReport, field, new_callable=PropertyMock, return_value=value):
                        self.assertFalse(report.passed)
                        with self.assertRaisesRegex(TuningCheckError, 'nonfinite|missing'):
                            report.require_pass()


if __name__ == '__main__':
    unittest.main()
