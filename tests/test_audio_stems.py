"""Regression tests for aimusic.audio.render.stems.render_role_stems.

Covers Bug A from PR #47 review: duplicate instrument names must produce
distinct WAV files and the rough mix must sum each stem exactly once.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


def _audio_render_available() -> bool:
    try:
        import pretty_midi  # noqa: F401
        import soundfile  # noqa: F401
        return True
    except ImportError:
        return False


def _make_two_pianos_one_bass_midi(path: Path) -> Path:
    """Create a MIDI with two 'Piano' instruments and one 'Bass' instrument."""
    import pretty_midi as pm

    midi = pm.PrettyMIDI(initial_tempo=120.0)
    # Piano #1: middle C, one quarter note
    inst_a = pm.Instrument(program=0, is_drum=False, name="Piano")
    inst_a.notes.append(pm.Note(velocity=80, pitch=60, start=0.0, end=0.5))
    # Piano #2: same role name, distinct pitch (E)
    inst_b = pm.Instrument(program=0, is_drum=False, name="Piano")
    inst_b.notes.append(pm.Note(velocity=80, pitch=64, start=0.0, end=0.5))
    # Bass: distinct role
    inst_c = pm.Instrument(program=32, is_drum=False, name="Bass")
    inst_c.notes.append(pm.Note(velocity=80, pitch=40, start=0.0, end=0.5))
    midi.instruments.extend([inst_a, inst_b, inst_c])
    midi.write(str(path))
    return path


@unittest.skipUnless(_audio_render_available(), "requires pip install -e '.[audio]'")
class TestRenderRoleStemsDuplicates(unittest.TestCase):
    def test_duplicate_role_names_produce_distinct_wav_files(self) -> None:
        from aimusic.audio.render.stems import render_role_stems

        with tempfile.TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            midi_path = tmp / "src.mid"
            _make_two_pianos_one_bass_midi(midi_path)

            out_dir = tmp / "stems"
            stems = render_role_stems(midi_path, output_dir=out_dir)

            # Three role WAVs on disk
            self.assertTrue((out_dir / "piano.wav").is_file())
            self.assertTrue((out_dir / "piano_2.wav").is_file())
            self.assertTrue((out_dir / "bass.wav").is_file())

            # Dict keys distinct
            self.assertIn("piano", stems)
            self.assertIn("piano_2", stems)
            self.assertIn("bass", stems)

            # Dict values point to distinct paths
            self.assertNotEqual(stems["piano"], stems["piano_2"])
            self.assertEqual(stems["piano"].name, "piano.wav")
            self.assertEqual(stems["piano_2"].name, "piano_2.wav")
            self.assertEqual(stems["bass"].name, "bass.wav")

    def test_duplicate_role_rough_mix_sums_each_stem_once(self) -> None:
        """Regression: the second piano must not be summed twice into rough_mix.

        Build a ground-truth mix by rendering each instrument to its own WAV
        (via ``render_stem`` with ``instrument_index``) so each piano truly has
        its own audio. The rough_mix written by ``render_role_stems`` must
        equal the sum of those distinct stems — modulo the global peak
        normalization that both paths apply.
        """
        import numpy as np
        import pretty_midi as pm
        import soundfile as sf

        from aimusic.audio.render.simple_r import render_stem
        from aimusic.audio.render.stems import render_role_stems

        with tempfile.TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            midi_path = tmp / "src.mid"
            _make_two_pianos_one_bass_midi(midi_path)

            # Ground truth: render each instrument to its OWN WAV.
            truth_dir = tmp / "truth"
            truth_dir.mkdir()
            truth_paths: dict[str, Path] = {}
            midi = pm.PrettyMIDI(str(midi_path))
            for idx, inst in enumerate(midi.instruments):
                p = truth_dir / f"inst_{idx}.wav"
                render_stem(
                    midi_path, output_path=p, instrument_index=idx, sample_rate=48000
                )
                truth_paths[f"inst_{idx}"] = p

            # Read ground-truth stems, sum them, then peak-normalize to match
            # what rough_mix() does internally.
            from aimusic.audio.render.util import peak_normalize
            truth_waves: list[np.ndarray] = []
            sr: int | None = None
            for p in truth_paths.values():
                data, file_sr = sf.read(str(p), always_2d=False)
                truth_waves.append(np.asarray(data, dtype=np.float64))
                sr = int(file_sr)
            assert sr is not None
            max_len = max(w.shape[0] for w in truth_waves)
            expected = np.zeros(max_len, dtype=np.float64)
            for w in truth_waves:
                expected[: w.shape[0]] += w
            expected = peak_normalize(expected, -12.0)

            # Now exercise the buggy path
            out_dir = tmp / "stems"
            stems = render_role_stems(midi_path, output_dir=out_dir)

            mix_path = stems["rough_mix"]
            mix, mix_sr = sf.read(str(mix_path), always_2d=False)
            mix = np.asarray(mix, dtype=np.float64)
            self.assertEqual(int(mix_sr), sr)

            # Compare shapes and audio similarity. The bug (double-add of
            # piano_2 into the mix) inflates the rough_mix's energy by ~3 dB
            # relative to the ground-truth sum of three distinct stems.
            n = min(mix.shape[0], expected.shape[0])
            diff = mix[:n] - expected[:n]
            rms_diff = float(np.sqrt(np.mean(diff * diff) + 1e-12))
            rms_truth = float(np.sqrt(np.mean(expected[:n] * expected[:n]) + 1e-12))
            ratio = rms_diff / max(rms_truth, 1e-12)
            self.assertLess(
                ratio,
                0.10,
                msg=(
                    f"rough_mix differs from ground-truth sum by {ratio:.3f} "
                    f"(normalized RMS diff); piano_2 likely summed twice "
                    "or written over piano (regression of Bug A)."
                ),
            )


if __name__ == "__main__":
    unittest.main()
