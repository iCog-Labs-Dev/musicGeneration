from __future__ import annotations

import argparse
import contextlib
from dataclasses import replace
import io
import json
import tempfile
import unittest
from pathlib import Path
from typing import Mapping, Sequence
from unittest.mock import patch

import mido

from aimusic.app.cli import handle_export, handle_generate
from aimusic.core.component_types import CandidateGenerationResult, SymbolicNote, TrackInstrumentConfig
from aimusic.core.config import DecodeConfig, StyleConfig
from aimusic.core.core_types import BeatState, NoteEvent, Score
from aimusic.core.protocols import ScoreDecoder, ScoreRenderer
from aimusic.core.render_package import load_render_package
from aimusic.core.rng import RNGKey
from aimusic.core.vocab import Vocabularies
from aimusic.render.midi_render import render_midi
from aimusic.render.adapters import RenderResultAdapter
from aimusic.planning.candidates import MetricBoundCandidates
from aimusic.scoring.priors import Prior, PriorContext
from aimusic.theory.edo import EDO
from tests.test_metric_injection import FixedMetric
from tests.test_planning_injection import TracedCandidates


def generate_args(out: Path, edo: int = 12) -> argparse.Namespace:
    return argparse.Namespace(
        seed=11, beats=4, edo=edo, meter="4/4", groove_family="straight",
        tempo_bpm=105.0, sample_path=False, subbeats_per_beat=4,
        drum_density=0.75, bass_density=0.6, comping_density=0.55,
        lead_density=0.45, base_tuning=0, pitch_bend_range=2,
        rendering_method="MPE", track_program=[], drum_track=[], out=str(out),
    )


class KnownScoreDecoder:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.score = Score(
            note_events=(NoteEvent(0, 480, 5, 0.6, track="lead"),),
            ticks_per_beat=480, tempo_bpm=105.0,
        )
        self.calls: list[tuple[int, int, float, RNGKey]] = []
        self.paths: list[tuple[BeatState, ...]] = []
        self.vocabularies: Vocabularies | None = None

    def __call__(
        self, path: Sequence[BeatState], *, key: RNGKey,
        decode_config: DecodeConfig | None = None,
        vocabularies: Vocabularies, edo: int = 12, ticks_per_beat: int = 480,
        tempo_bpm: float = 120.0, include_terminal_state: bool = False,
    ) -> tuple[Score, RNGKey]:
        self.calls.append((len(path), edo, tempo_bpm, key))
        self.paths.append(tuple(path))
        self.vocabularies = vocabularies
        if self.fail:
            raise RuntimeError("decoder replacement failed")
        return self.score, key.next_key()


class RecordingRenderer:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[tuple[Sequence[SymbolicNote] | Score, int, str]] = []

    def __call__(
        self, notes: Sequence[SymbolicNote] | Score, edo: EDO,
        output_path: str, ticks_per_beat: int = 480, *,
        track_instruments: Mapping[str, TrackInstrumentConfig] | None = None,
    ) -> None:
        self.calls.append((notes, edo.config.n, output_path))
        if self.fail:
            raise RuntimeError("renderer replacement failed")
        # Packaging needs real MIDI, so the double writes the captured request.
        render_midi(
            notes, edo, output_path, ticks_per_beat,
            track_instruments=track_instruments,
        )


checked_decoder: ScoreDecoder = KnownScoreDecoder()
checked_renderer: ScoreRenderer = RecordingRenderer()


class ThirdHeadCandidates:
    def __init__(self, metric: FixedMetric) -> None:
        self.generate = MetricBoundCandidates(metric)
        self.transitions: set[tuple[int, BeatState, BeatState]] = set()

    def __call__(
        self, prev_state: BeatState, t: int, key: RNGKey, d_max: int,
        style_config: StyleConfig | None = None,
        vocabularies: Vocabularies | None = None,
        prior: Prior | None = None, context: PriorContext | None = None,
        edo: int | None = None, proposal_budget: int | None = None,
        prior_guided_proposals: bool = False,
    ) -> tuple[CandidateGenerationResult, RNGKey]:
        assert vocabularies is not None
        result, next_key = self.generate(
            prev_state, t, key, d_max, style_config, vocabularies,
            prior, context, edo, proposal_budget, prior_guided_proposals,
        )
        third = vocabularies.heads.token_for_label("third").id
        # Propose a legal anchor on weak beats too, unlike the default proposal policy.
        states = tuple(dict.fromkeys(replace(state, head_id=third) for state in result.states))
        self.transitions.update((t, prev_state, state) for state in states)
        return replace(
            result, states=states, scores=(),
            unique_count=len(states) + len(result.rejections),
        ), next_key


class TestArtifactInjection(unittest.TestCase):
    def test_all_four_replacements_complete_cli_and_ui_workflows(self) -> None:
        import ui

        for workflow in ("cli", "ui"):
            for edo in (12, 19):
                with self.subTest(workflow=workflow, edo=edo), tempfile.TemporaryDirectory() as temporary:
                    directory = Path(temporary)
                    metric = FixedMetric(root_value=7.0, chord_value=3.0)
                    candidates = ThirdHeadCandidates(metric)
                    decoder = KnownScoreDecoder()
                    renderer = RecordingRenderer()
                    reporting_renderer = RenderResultAdapter(renderer)
                    if workflow == "cli":
                        with contextlib.redirect_stdout(io.StringIO()):
                            handle_generate(
                                generate_args(directory, edo), tonal_metric=metric,
                                candidate_generator=candidates, decoder=decoder, renderer=reporting_renderer,
                            )
                    else:
                        params = ui.GenerationParams(
                            seed=11, beats=4, edo=edo, meter="4/4", groove_family="straight",
                            tempo_bpm=105.0, sample_path=False, drum_density=0.75,
                            bass_density=0.6, comping_density=0.55, lead_density=0.45,
                            pitch_bend_range=2, rendering_method="MPE", bass_program=34,
                            comping_program=5, lead_program=88, drum_track=["drums"],
                        )
                        with patch.object(ui, "OUTPUT_DIR", directory):
                            artifacts = ui._generate_artifacts(
                                params, tonal_metric=metric, candidate_generator=candidates,
                                decoder=decoder, renderer=reporting_renderer,
                            )
                        self.assertIsNotNone(artifacts.render_result)
                        assert artifacts.render_result is not None
                        self.assertEqual(artifacts.render_result.output_path, str(artifacts.midi_path))
                        self.assertEqual(artifacts.render_result.size_bytes, artifacts.midi_path.stat().st_size)

                    self.assertEqual(len(decoder.paths), 1)
                    path = decoder.paths[0]
                    self.assertEqual(len(path), 5)
                    vocabs = decoder.vocabularies
                    assert vocabs is not None
                    third = vocabs.heads.token_for_label("third").id
                    self.assertTrue(candidates.transitions)
                    self.assertTrue(all(state.head_id == third for state in path[1:-1]))
                    for time_index in range(3):
                        self.assertIn((time_index, path[time_index], path[time_index + 1]), candidates.transitions)
                    self.assertTrue(metric.root_calls)
                    self.assertTrue(metric.chord_calls)
                    self.assertTrue(all(call[-1] == edo for call in metric.root_calls + metric.chord_calls))

                    self.assertEqual(len(renderer.calls), 1)
                    self.assertIs(renderer.calls[0][0], decoder.score)
                    self.assertEqual(renderer.calls[0][1], edo)
                    score_path = next(directory.glob("*_score.json"))
                    midi_path = next(directory.glob("*.mid"))
                    self.assertEqual(Score.from_dict(json.loads(score_path.read_text(encoding="utf-8"))), decoder.score)
                    midi = mido.MidiFile(midi_path)
                    self.assertEqual(sum(
                        message.type == "note_on" and message.velocity > 0
                        for track in midi.tracks for message in track
                    ), 1)
                    manifest = json.loads(next(directory.glob("*_manifest.json")).read_text(encoding="utf-8"))
                    curve = manifest["structure"]["tension_curve"]
                    self.assertEqual(len(curve), 4)
                    role_values = {"hold": 0.0, "prep": 0.33, "change": 0.66, "cad": 1.0}
                    max_boundary = max(token.level for token in vocabs.boundaries.tokens)
                    for index, state in enumerate(path[:-1]):
                        role = role_values[vocabs.roles.token_for_id(state.role_id).label]
                        motion = 0.0 if index == 0 else 0.20 * 7 / 8 + 0.25 * 3 / 4
                        expected = min(1.0, 0.40 * role + 0.15 * state.boundary_lvl / max_boundary + motion)
                        self.assertEqual(curve[index][0], float(index))
                        self.assertAlmostEqual(curve[index][1], expected)
                    if workflow == "cli":
                        package = load_render_package(next(directory.glob("run_*")))
                        self.assertEqual(Score.from_dict(json.loads(package.score_path.read_text(encoding="utf-8"))), decoder.score)
                        self.assertEqual(package.midi_path.read_bytes(), midi_path.read_bytes())
                        self.assertEqual(json.loads(package.tuning_path.read_text(encoding="utf-8"))["edo"], edo)

    def test_cli_forwards_candidate_errors_before_decoding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            decoder = KnownScoreDecoder()
            with self.assertRaisesRegex(RuntimeError, "candidate replacement failed"):
                handle_generate(
                    generate_args(Path(temporary)), decoder=decoder,
                    candidate_generator=TracedCandidates(fail=True),
                )
            self.assertEqual(decoder.calls, [])
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_cli_uses_supplied_score_and_renderer_in_both_edos(self) -> None:
        for edo in (12, 19):
            with self.subTest(edo=edo), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                decoder = KnownScoreDecoder()
                renderer = RecordingRenderer()
                with contextlib.redirect_stdout(io.StringIO()):
                    handle_generate(generate_args(directory, edo), decoder=decoder, renderer=renderer)
                self.assertEqual(len(decoder.calls), 1)
                self.assertEqual(decoder.calls[0][:3], (5, edo, 105.0))
                self.assertEqual(len(renderer.calls), 1)
                self.assertIs(renderer.calls[0][0], decoder.score)
                self.assertEqual(renderer.calls[0][1], edo)
                score_path = next(directory.glob("*_score.json"))
                self.assertEqual(
                    Score.from_dict(json.loads(score_path.read_text(encoding="utf-8"))),
                    decoder.score,
                )
                package = load_render_package(next(directory.glob("run_*")))
                midi = mido.MidiFile(package.midi_path)
                self.assertEqual(sum(
                    message.type == "note_on" and message.velocity > 0
                    for track in midi.tracks for message in track
                ), 1)
                manifest = json.loads(package.manifest_path.read_text(encoding="utf-8"))
                self.assertNotIn("decoder", manifest["config"])
                self.assertNotIn("renderer", manifest["config"])

    def test_cli_decoder_error_stops_before_rendering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            renderer = RecordingRenderer()
            with self.assertRaisesRegex(RuntimeError, "decoder replacement failed"):
                handle_generate(
                    generate_args(Path(temporary)),
                    decoder=KnownScoreDecoder(fail=True), renderer=renderer,
                )
            self.assertEqual(renderer.calls, [])
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_cli_renderer_error_does_not_write_a_success_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "renderer replacement failed"):
                handle_generate(
                    generate_args(directory), decoder=KnownScoreDecoder(),
                    renderer=RecordingRenderer(fail=True),
                )
            self.assertEqual(list(directory.glob("*_manifest.json")), [])
            self.assertEqual(list(directory.glob("run_*")), [])

    def test_export_uses_supplied_renderer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            score = KnownScoreDecoder().score
            source = directory / "score.json"
            source.write_text(json.dumps(score.to_dict()), encoding="utf-8")
            renderer = RecordingRenderer()
            args = generate_args(directory)
            args.file = str(source)
            args.out = str(directory / "export.mid")
            with contextlib.redirect_stdout(io.StringIO()):
                handle_export(args, renderer=renderer)
            self.assertEqual(len(renderer.calls), 1)
            self.assertEqual(renderer.calls[0][:2], (score, 12))
            self.assertEqual(Path(renderer.calls[0][2]).name, Path(args.out).name)
            self.assertFalse(Path(renderer.calls[0][2]).exists())
            self.assertEqual(len(mido.MidiFile(args.out).tracks), 2)

    def test_ui_helper_uses_supplied_decoder_and_renderer(self) -> None:
        import ui

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            params = ui.GenerationParams(
                seed=11, beats=4, edo=19, meter="4/4", groove_family="straight",
                tempo_bpm=105.0, sample_path=False, drum_density=0.75,
                bass_density=0.6, comping_density=0.55, lead_density=0.45,
                pitch_bend_range=2, rendering_method="MPE", bass_program=34,
                comping_program=5, lead_program=88, drum_track=["drums"],
            )
            decoder = KnownScoreDecoder()
            renderer = RecordingRenderer()
            with patch.object(ui, "OUTPUT_DIR", directory):
                with self.assertRaisesRegex(RuntimeError, "candidate replacement failed"):
                    ui._generate_artifacts(
                        params, decoder=decoder,
                        candidate_generator=TracedCandidates(fail=True),
                    )
                self.assertEqual(decoder.calls, [])
                self.assertEqual(list(directory.iterdir()), [])
                artifacts = ui._generate_artifacts(params, decoder=decoder, renderer=renderer)
            self.assertIs(renderer.calls[0][0], decoder.score)
            self.assertEqual(renderer.calls[0][1], 19)
            self.assertEqual(Path(renderer.calls[0][2]).name, artifacts.midi_path.name)
            self.assertFalse(Path(renderer.calls[0][2]).exists())
            self.assertEqual(
                Score.from_dict(json.loads(artifacts.score_path.read_text(encoding="utf-8"))),
                decoder.score,
            )
