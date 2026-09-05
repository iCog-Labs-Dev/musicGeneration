from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import mido

from aimusic.core.config import (
    DecodeConfig,
    EDOConfig,
    MicrotonalRendering,
    StyleConfig,
)
from aimusic.core.core_types import Score
from aimusic.core.rng import RNGKey
from aimusic.decode import decode_path_to_score
from aimusic.planning.plans import MethodARunConfig, run_method_a
from aimusic.render import render_midi
from aimusic.theory.edo import EDO


FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> dict[str, Any]:
    with (FIXTURE_DIR / name).open(encoding="utf-8") as fixture_file:
        return json.load(fixture_file)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _midi_note_ons(midi_file: mido.MidiFile) -> list[list[object]]:
    note_ons: list[list[object]] = []
    for track in midi_file.tracks:
        track_name = next(
            message.name for message in track if message.type == "track_name"
        )
        absolute_tick = 0
        for message in track:
            absolute_tick += message.time
            if message.type == "note_on" and message.velocity > 0:
                note_ons.append(
                    [
                        track_name,
                        absolute_tick,
                        message.channel,
                        message.note,
                        message.velocity,
                    ]
                )
    return note_ons


class TestShortPassageGoldenFixture(unittest.TestCase):
    def test_score_and_midi_match_checked_in_golden(self) -> None:
        fixture = _load_fixture("short_passage_golden.json")
        score = Score.from_dict(fixture["score"])
        self.assertEqual(score.to_dict(), fixture["score"])

        with tempfile.TemporaryDirectory() as temp_dir:
            midi_path = Path(temp_dir) / "short.mid"
            render_midi(
                score,
                EDO(
                    EDOConfig(
                        n=12,
                        base_tuning=0,
                        microtonal_rendering_method=MicrotonalRendering.MPE,
                    )
                ),
                str(midi_path),
            )
            midi_bytes = midi_path.read_bytes()
            midi_file = mido.MidiFile(midi_path)

        midi_fixture = fixture["midi"]
        self.assertEqual(_sha256_bytes(midi_bytes), midi_fixture["sha256"])
        self.assertEqual(
            [
                next(message.name for message in track if message.type == "track_name")
                for track in midi_file.tracks
            ],
            midi_fixture["track_names"],
        )
        self.assertEqual(
            next(
                message.tempo
                for message in midi_file.tracks[0]
                if message.type == "set_tempo"
            ),
            midi_fixture["tempo"],
        )
        self.assertEqual(_midi_note_ons(midi_file), midi_fixture["note_ons"])


class TestLongHorizonDeterministicFixture(unittest.TestCase):
    def _render_fixture(self, fixture: dict[str, Any], output_path: Path) -> dict[str, Any]:
        densities = fixture["densities"]
        decode_config = DecodeConfig(
            drum_density=densities["drum"],
            bass_density=densities["bass"],
            comping_density=densities["comping"],
            lead_density=densities["lead"],
        )
        run_config = MethodARunConfig(
            total_beats=fixture["beats"],
            seed=fixture["seed"],
            use_sampling=True,
            style_config=StyleConfig(
                allowed_meters=(fixture["meter"],),
                groove_families=(fixture["groove_family"],),
            ),
            decode_config=decode_config,
            edo=fixture["edo"],
        )
        result, next_key = run_method_a(run_config, key=RNGKey(seed=fixture["seed"]))
        score, _ = decode_path_to_score(
            result.path,
            decode_config=decode_config,
            vocabularies=result.vocabularies,
            edo=fixture["edo"],
            tempo_bpm=fixture["tempo_bpm"],
            key=next_key,
        )
        render_midi(
            score,
            EDO(
                EDOConfig(
                    n=fixture["edo"],
                    base_tuning=0,
                    pitch_bend_range=2,
                    microtonal_rendering_method=MicrotonalRendering.MPE,
                )
            ),
            str(output_path),
        )
        return {
            "path_sha256": _canonical_json_hash(
                [state.to_dict() for state in result.path]
            ),
            "score_sha256": _canonical_json_hash(score.to_dict()),
            "midi_sha256": _sha256_bytes(output_path.read_bytes()),
            "event_count": len(score),
            "track_event_counts": score.track_event_counts(),
        }

    def test_fixed_seed_long_horizon_is_stable_and_repeatable(self) -> None:
        fixture = _load_fixture("long_horizon_smoke.json")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            first = self._render_fixture(fixture, temp_path / "first.mid")
            second = self._render_fixture(fixture, temp_path / "second.mid")

        self.assertEqual(first, second)
        self.assertGreaterEqual(first["event_count"], fixture["minimum_event_count"])
        self.assertEqual(
            sorted(first["track_event_counts"]),
            sorted(fixture["expected_tracks"]),
        )
        self.assertTrue(all(first["track_event_counts"].values()))


if __name__ == "__main__":
    unittest.main()
