"""Unit tests for groove apply (M1 PR3)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "audio"
QUANTIZED = FIXTURES / "eight_bar_quantized.mid"
EXPRESSIVE = FIXTURES / "already_expressive.mid"


def _audio_spine_available() -> bool:
    try:
        import pretty_midi  # noqa: F401
        import pydantic  # noqa: F401
        import yaml  # noqa: F401
        return QUANTIZED.is_file()
    except ImportError:
        return False


@unittest.skipUnless(_audio_spine_available(), "requires pip install -e '.[audio]' + fixtures")
class TestAudioGroove(unittest.TestCase):
    def test_note_count_preserved_without_ghosts(self) -> None:
        from aimusic.audio.groove.apply import apply_groove

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "g.mid"
            apply_groove(
                QUANTIZED,
                {
                    "swing": {
                        "ratio": 0.66,
                        "subdivision": 8,
                        "applies_to": ["drums", "bass", "comp"],
                    }
                },
                output_path=out,
            )
            report = json.loads(Path(str(out) + ".diff.json").read_text(encoding="utf-8"))
            self.assertEqual(report["ghost_inserted"], 0)
            self.assertEqual(report["notes_in"], report["notes_out"])

    def test_velocity_map_idempotent_at_depth_zero(self) -> None:
        import pretty_midi as pm
        from aimusic.audio.groove.apply import apply_groove

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "light.mid"
            apply_groove(
                QUANTIZED,
                {"velocity": {"accent_map_16": [1.2, 0.8] * 8}},
                output_path=out,
                light_cleanup=True,
                config={"code_version": "0.1.0"},
            )
            src = pm.PrettyMIDI(str(QUANTIZED))
            dst = pm.PrettyMIDI(str(out))
            self.assertEqual(
                sum(len(i.notes) for i in src.instruments),
                sum(len(i.notes) for i in dst.instruments),
            )
            for si, di in zip(src.instruments, dst.instruments, strict=True):
                for sn, dn in zip(si.notes, di.notes, strict=True):
                    self.assertEqual(sn.pitch, dn.pitch)
                    self.assertEqual(sn.velocity, dn.velocity)

    def test_already_expressive_nearly_untouched(self) -> None:
        from aimusic.audio.groove.apply import apply_groove

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ex.mid"
            apply_groove(
                EXPRESSIVE,
                {"swing": {"ratio": 0.55, "subdivision": 8}},
                output_path=out,
                light_cleanup=True,
                config={"groove": {"light_cleanup": True}},
            )
            report = json.loads(Path(str(out) + ".diff.json").read_text(encoding="utf-8"))
            self.assertLess(report["mean_abs_offset_ms"], 0.1)
            self.assertEqual(report["notes_in"], report["notes_out"])

    def test_invalid_spec_rejected(self) -> None:
        from pydantic import ValidationError

        from aimusic.audio.groove.apply import apply_groove

        with self.assertRaises(ValidationError):
            apply_groove(QUANTIZED, {"swing": {"ratio": 0.9, "subdivision": 8}})

    def test_swing_offsets_within_one_tick(self) -> None:
        import pretty_midi as pm

        from aimusic.audio.groove.apply import (
            _apply_to_midi,
            _seconds_per_beat,
            _swing_offset_sec,
        )
        from aimusic.audio.groove.spec import parse_groove_spec

        spec = parse_groove_spec(
            {
                "swing": {
                    "ratio": 0.66,
                    "subdivision": 8,
                    "applies_to": ["drums", "bass", "comp"],
                }
            }
        )
        src = pm.PrettyMIDI(str(QUANTIZED))
        spb = _seconds_per_beat(src)
        sec_per_tick = spb / float(src.resolution)
        _out, report = _apply_to_midi(src, spec, light_cleanup=False)
        for d, n_role in zip(
            report["note_diffs"],
            [
                (
                    "drums"
                    if (inst.is_drum or "drums" in (inst.name or "").lower())
                    else (
                        "bass"
                        if "bass" in (inst.name or "").lower()
                        else (
                            "lead"
                            if "lead" in (inst.name or "").lower()
                            else "comp"
                        )
                    )
                )
                for inst in src.instruments
                for _ in inst.notes
            ],
            strict=True,
        ):
            self.assertEqual(d["role"], n_role)
        # Spot-check first drum note offset matches swing helper within 1 tick
        first = src.instruments[0].notes[0]
        expected = _swing_offset_sec(first.start, spb, 0.66, 8)
        self.assertLessEqual(
            abs(report["note_diffs"][0]["offset_sec"] - expected),
            sec_per_tick + 1e-9,
        )


if __name__ == "__main__":
    unittest.main()
