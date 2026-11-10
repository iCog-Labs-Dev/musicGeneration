"""Source-MIDI reconciliation: planner facts are not MIDI inference."""
from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import mido

from aimusic.audio.from_score import from_score
from aimusic.core.config import EDOConfig
from aimusic.core.core_types import NoteEvent, Score
from aimusic.core.diagnostics import RunManifest
from aimusic.core.render_package import write_render_package
from aimusic.render import render_midi
from aimusic.theory.edo import EDO


class TestAudioReconcile(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ctx = self.package()

    def package(self, *, edo=12, name='texture', notes=None, track_programs=None):
        score = Score(note_events=notes or (
            NoteEvent(ton=0, toff=480, h=19, v=0.8, track=name),
            NoteEvent(ton=240, toff=720, h=21, v=0.8, track=name),
        ))
        midi = self.root / f'{edo}.mid'
        # Keep source tuning aligned with write_render_package defaults.
        render_midi(score, EDO(EDOConfig(n=edo, base_tuning=0, pitch_bend_range=2)), str(midi))
        package = write_render_package(
            self.root, score=score, midi_path=midi, manifest=RunManifest(seed=1, config_dump={}),
            edo=edo, track_programs=track_programs or ({name: 0} if name == 'texture' else None),
        )
        return from_score(package.root)

    def reconcile(self, ctx=None):
        from aimusic.audio.analysis.reconcile import reconcile
        return reconcile(ctx or self.ctx)

    def test_reports_authoritative_roles_counts_and_trailing_silence(self) -> None:
        report = self.reconcile()
        self.assertTrue(report.passed)
        self.assertEqual(report.score_occupancy_sec, 0.75)
        self.assertEqual(report.planner_extent_sec, 2.0)
        track = report.tracks[0]
        self.assertEqual(track['name'], 'texture')
        self.assertEqual(track['source_track_index'], 1)
        self.assertEqual(track['role'], 'lead')
        self.assertEqual(track['note_count'], 2)
        self.assertEqual(track['role_source'], 'structure.json')
        self.assertEqual(track['count_source'], 'score.json')
        unavailable = [c for c in report.checks if c.field == 'role_inference']
        self.assertEqual(unavailable[0].outcome, 'unavailable')
        self.assertIsNone(unavailable[0].observed)
        for check in report.to_dict()['checks']:
            self.assertTrue({'field', 'expected', 'observed', 'tolerance', 'outcome'} <= check.keys())

    def test_mpe_member_channels_count_as_one_source_track(self) -> None:
        report = self.reconcile(self.package(edo=19))
        self.assertTrue(report.passed)
        self.assertEqual(len(report.tracks), 1)
        self.assertEqual(report.tracks[0]['note_count'], 2)
        self.assertEqual(len(report.tracks[0]['channels']), 2)

    def retime_notes(self, ctx, intervals):
        """Change endpoints independently, preserving source channels/setup."""
        midi = mido.MidiFile(ctx.package.midi_path)
        track = midi.tracks[1]
        keys = [(m.channel, m.note) for m in track if m.type == 'note_on' and m.velocity > 0]
        self.assertEqual(len(keys), len(intervals))
        targets = dict(zip(keys, intervals))
        events = []
        tick = 0
        for order, msg in enumerate(track):
            tick += msg.time
            if msg.type == 'end_of_track':
                continue
            at = tick
            if msg.type in ('note_on', 'note_off'):
                endpoint = 0 if msg.type == 'note_on' and msg.velocity > 0 else 1
                at = targets[(msg.channel, msg.note)][endpoint]
            events.append((at, order, msg))
        rebuilt = mido.MidiTrack()
        previous = 0
        for at, _, msg in sorted(events, key=lambda e: e[:2]):
            rebuilt.append(msg.copy(time=at - previous))
            previous = at
        rebuilt.append(mido.MetaMessage('end_of_track'))
        midi.tracks[1] = rebuilt
        midi.save(ctx.package.midi_path)

    def test_simultaneous_intervals_match_after_one_tick_reordering(self) -> None:
        for edo in (12, 19):
            ctx = self.package(edo=edo, notes=(
                NoteEvent(ton=240, toff=480, h=19, v=0.8, track='texture'),
                NoteEvent(ton=240, toff=720, h=21, v=0.8, track='texture'),
            ))
            for offset in (-1, 1):
                with self.subTest(edo=edo, offset=offset):
                    self.retime_notes(ctx, [(240 + offset, 480 + offset), (240 - offset, 720 - offset)])
                    report = self.reconcile(ctx)
                    self.assertTrue(report.passed)
                    self.assertEqual(len(report.tracks[0]['channels']), 2 if edo == 19 else 1)
                    endpoints = [c for c in report.checks if c.field.startswith('note[')]
                    self.assertEqual(len(endpoints), 4)
                    self.assertTrue(all(c.outcome == 'pass' for c in endpoints))
                    self.assertEqual(report, self.reconcile(ctx))

    def test_interval_matching_reassigns_an_ambiguous_candidate(self) -> None:
        for edo in (12, 19):
            with self.subTest(edo=edo):
                ctx = self.package(edo=edo, notes=(
                    NoteEvent(ton=240, toff=480, h=19, v=0.8, track='texture'),
                    NoteEvent(ton=240, toff=482, h=21, v=0.8, track='texture'),
                ))
                # The earlier observation fits both notes, the later only the short one.
                self.retime_notes(ctx, [(241, 479), (239, 481)])
                report = self.reconcile(ctx)
                self.assertTrue(report.passed)
                ends = [c.observed for c in report.checks if c.field.endswith('].end_sec')]
                self.assertEqual(ends, [479 / 960, 481 / 960])

    def test_interval_matching_is_one_to_one_and_rejects_outside_tolerance(self) -> None:
        from aimusic.audio.analysis.reconcile import ReconciliationError

        ctx = self.package(edo=19, notes=(
            NoteEvent(ton=240, toff=480, h=19, v=0.8, track='texture'),
            NoteEvent(ton=240, toff=480, h=21, v=0.8, track='texture'),
            NoteEvent(ton=240, toff=720, h=23, v=0.8, track='texture'),
        ))
        for intervals in (
            [(240, 480), (240, 600), (240, 720)],  # two expectations cannot reuse one observation
            [(242, 480), (240, 480), (240, 720)],  # onset exceeds one tick
            [(240, 482), (240, 480), (240, 720)],  # release exceeds one tick
        ):
            with self.subTest(intervals=intervals):
                self.retime_notes(ctx, intervals)
                with self.assertRaisesRegex(ReconciliationError, 'texture.*note'):
                    self.reconcile(ctx)

    def test_role_alias_and_nonbinding_heuristic(self) -> None:
        ctx = self.package(name='lead')
        structure = replace(ctx.structure, tracks=[{**ctx.structure.tracks[0], 'role': 'comp'}])
        report = self.reconcile(replace(ctx, structure=structure))
        self.assertTrue(report.passed)
        self.assertEqual(report.tracks[0]['role'], 'comping')
        inferred = next(c for c in report.checks if c.field == 'role_inference')
        self.assertEqual(inferred.observed, 'lead')
        self.assertEqual(inferred.outcome, 'informational')

    def test_exact_program_count_identity_and_drum_metadata(self) -> None:
        from aimusic.audio.analysis.reconcile import ReconciliationError

        for field, value in (('program', 1), ('percussion', True), ('name', 'other')):
            with self.subTest(field=field):
                structure = replace(self.ctx.structure, tracks=[{**self.ctx.structure.tracks[0], field: value}])
                with self.assertRaisesRegex(ReconciliationError, field) as error:
                    self.reconcile(replace(self.ctx, structure=structure))
                self.assertFalse(error.exception.report.passed)
        score = replace(self.ctx.score, note_events=self.ctx.score.note_events[:1])
        with self.assertRaisesRegex(ReconciliationError, 'texture.*note_count'):
            self.reconcile(replace(self.ctx, score=score))
        drums = self.package(name='drums')
        structure = replace(drums.structure, tracks=[{**drums.structure.tracks[0], 'program': 1}])
        with self.assertRaisesRegex(ReconciliationError, 'drums.*program'):
            self.reconcile(replace(drums, structure=structure))

    def test_tempo_tolerance_is_001_bpm(self) -> None:
        from aimusic.audio.analysis.reconcile import ReconciliationError

        for delta in (0.0009, 0.001):
            structure = replace(self.ctx.structure, tempo_map=[[0.0, 120.0 + delta]])
            self.assertTrue(self.reconcile(replace(self.ctx, structure=structure)).passed)
        structure = replace(self.ctx.structure, tempo_map=[[0.0, 120.0011]])
        with self.assertRaisesRegex(ReconciliationError, 'tempo_map.*bpm'):
            self.reconcile(replace(self.ctx, structure=structure))

    def test_bar_boundary_one_source_tick_inclusive(self) -> None:
        from aimusic.audio.analysis.reconcile import ReconciliationError

        tick = 0.5 / 480
        for offset in (-tick, tick):
            structure = replace(self.ctx.structure, bar_table=[[0, 0.0], [1, 2.0 + offset]])
            report = self.reconcile(replace(self.ctx, structure=structure))
            self.assertTrue(report.passed)
            boundary = next(c for c in report.checks if c.field == 'bar_table[1].time_sec')
            self.assertAlmostEqual(boundary.tolerance, tick)
        structure = replace(self.ctx.structure, bar_table=[[0, 0.0], [1, 2.0 + tick * 1.1]])
        with self.assertRaisesRegex(ReconciliationError, 'bar_table'):
            self.reconcile(replace(self.ctx, structure=structure))

    def test_note_endpoints_and_end_of_track_do_not_use_planner_extent(self) -> None:
        from aimusic.audio.analysis.reconcile import ReconciliationError

        midi = mido.MidiFile(self.ctx.package.midi_path)
        track = midi.tracks[1]
        track[-2].time += 1  # final note-off, one tick inclusive
        midi.save(self.ctx.package.midi_path)
        self.assertTrue(self.reconcile().passed)
        track[-2].time += 1
        midi.save(self.ctx.package.midi_path)
        with self.assertRaisesRegex(ReconciliationError, 'note.*end|occupancy'):
            self.reconcile()
        track[-2].time -= 2
        track[-1].time = 1200  # EOT at planner end is not Score occupancy.
        midi.save(self.ctx.package.midi_path)
        with self.assertRaisesRegex(ReconciliationError, 'end_of_track'):
            self.reconcile()

    def test_complete_tempo_meter_maps_and_local_tick_tolerance(self) -> None:
        from aimusic.audio.analysis.reconcile import ReconciliationError

        midi = mido.MidiFile(self.ctx.package.midi_path)
        conductor = midi.tracks[0]
        conductor.insert(-1, mido.MetaMessage('set_tempo', tempo=1000000, time=1920))
        conductor.insert(-1, mido.MetaMessage('time_signature', numerator=3, denominator=4, time=0))
        midi.save(self.ctx.package.midi_path)
        structure = replace(
            self.ctx.structure, tempo_map=[[0.0, 120.0], [2.0, 60.0]],
            meter=[[0, '4/4'], [1, '3/4']], bar_table=[[0, 0.0], [1, 2.0], [2, 5.0]],
            sections=[{'bars': [0, 2]}],
        )
        ctx = replace(self.ctx, structure=structure)
        self.assertTrue(self.reconcile(ctx).passed)
        structure = replace(structure, bar_table=[[0, 0.0], [1, 2.0], [2, 5.0 + 1 / 480]])
        self.assertTrue(self.reconcile(replace(ctx, structure=structure)).passed)
        with self.assertRaisesRegex(ReconciliationError, 'tempo_map.*count|meter.*count'):
            self.reconcile()
        structure = replace(ctx.structure, meter=[[0, '4/4'], [1, '6/8']])
        with self.assertRaisesRegex(ReconciliationError, 'meter.*signature'):
            self.reconcile(replace(ctx, structure=structure))

    def test_meter_changes_snap_to_declared_bars_only_within_one_tick(self) -> None:
        from aimusic.audio.analysis.reconcile import ReconciliationError

        original = self.ctx.package.midi_path.read_bytes()
        structure = replace(
            self.ctx.structure, meter=[[0, '4/4'], [1, '3/4'], [3, '2/4']],
            bar_table=[[0, 0.0], [1, 2.0], [2, 3.5], [3, 5.0], [4, 6.0]],
            sections=[{'bars': [0, 4]}],
        )
        ctx = replace(self.ctx, structure=structure)
        for offset in (-2, -1, 0, 1, 2):
            with self.subTest(offset=offset):
                self.ctx.package.midi_path.write_bytes(original)
                midi = mido.MidiFile(self.ctx.package.midi_path)
                midi.tracks[0].insert(-1, mido.MetaMessage(
                    'time_signature', numerator=3, denominator=4, time=1920 + offset,
                ))
                midi.tracks[0].insert(-1, mido.MetaMessage(
                    'time_signature', numerator=2, denominator=4, time=2880,
                ))
                midi.save(self.ctx.package.midi_path)
                if abs(offset) > 1:
                    with self.assertRaisesRegex(ReconciliationError, r'meter\[1\].time_sec') as error:
                        self.reconcile(ctx)
                    self.assertFalse(error.exception.report.passed)
                    continue
                report = self.reconcile(ctx)
                self.assertTrue(report.passed)
                changes = [c for c in report.checks if c.field.startswith('meter[')
                           and c.field.endswith('.time_sec')]
                self.assertEqual([c.observed for c in changes], [0.0, (1920 + offset) / 960,
                                                               (4800 + offset) / 960])
                bars = [c for c in report.checks if c.field.startswith('bar_table[')]
                self.assertEqual([c.observed for c in bars], [0.0, 2.0, 3.5, 5.0, 6.0])
                # Snapping meters must not forgive a genuinely wrong planner bar time.
                bad = replace(structure, bar_table=[*structure.bar_table[:-1], [4, 6.0 + 2 / 960]])
                with self.assertRaisesRegex(ReconciliationError, r'bar_table\[4\].time_sec'):
                    self.reconcile(replace(ctx, structure=bad))

    def test_resolution_and_bar_indices_are_exact(self) -> None:
        from aimusic.audio.analysis.reconcile import ReconciliationError

        midi = mido.MidiFile(self.ctx.package.midi_path)
        midi.ticks_per_beat = 481
        midi.save(self.ctx.package.midi_path)
        with self.assertRaisesRegex(ReconciliationError, 'ticks_per_beat'):
            self.reconcile()
        midi.ticks_per_beat = 480
        midi.save(self.ctx.package.midi_path)
        structure = replace(self.ctx.structure, bar_table=[[0, 0.0], [2, 2.0]])
        with self.assertRaisesRegex(ReconciliationError, 'bar.*indices'):
            self.reconcile(replace(self.ctx, structure=structure))

    def test_note_onset_and_change_point_tolerances(self) -> None:
        from aimusic.audio.analysis.reconcile import ReconciliationError

        midi = mido.MidiFile(self.ctx.package.midi_path)
        note_on = next(m for m in midi.tracks[1] if m.type == 'note_on')
        note_on.time += 1
        midi.save(self.ctx.package.midi_path)
        self.assertTrue(self.reconcile().passed)
        note_on.time += 1
        midi.save(self.ctx.package.midi_path)
        with self.assertRaisesRegex(ReconciliationError, r'note\[0\].start_sec'):
            self.reconcile()
        note_on.time -= 2
        change = mido.MetaMessage('set_tempo', tempo=1000000, time=480)
        midi.tracks[0].insert(-1, change)
        midi.save(self.ctx.package.midi_path)
        # Move declared change by one source tick, retaining its BPM.
        structure = replace(self.ctx.structure, tempo_map=[[0, 120], [0.5 + 1 / 480, 60]],
                            bar_table=[[0, 0], [1, 3.5]])
        self.assertTrue(self.reconcile(replace(self.ctx, structure=structure)).passed)
        structure = replace(structure, tempo_map=[[0, 120], [0.5 + 2 / 480, 60]])
        with self.assertRaisesRegex(ReconciliationError, r'tempo_map\[1\].time_sec'):
            self.reconcile(replace(self.ctx, structure=structure))

    def test_drum_program_changes_rejected_not_treated_as_pitched_program(self) -> None:
        from aimusic.audio.analysis.reconcile import ReconciliationError

        ctx = self.package(name='drums')
        midi = mido.MidiFile(ctx.package.midi_path)
        midi.tracks[1].insert(1, mido.Message('program_change', program=12, channel=9))
        midi.save(ctx.package.midi_path)
        with self.assertRaisesRegex(ReconciliationError, 'drums.*program'):
            self.reconcile(ctx)

    def test_mixed_pitched_and_drum_channels_reject_pitched_metadata(self) -> None:
        from aimusic.audio.analysis.reconcile import ReconciliationError

        ctx = self.package(edo=19)
        midi = mido.MidiFile(ctx.package.midi_path)
        channels = sorted({m.channel for m in midi.tracks[1] if m.type == 'note_on'})
        self.assertEqual(len(channels), 2)
        for msg in midi.tracks[1]:
            if getattr(msg, 'channel', None) == channels[-1]:
                msg.channel = 9
        midi.save(ctx.package.midi_path)
        with self.assertRaisesRegex(ReconciliationError, 'texture.*percussion'):
            self.reconcile(ctx)

    def test_duplicate_tracks_and_ambiguous_channel_ownership_fail(self) -> None:
        from aimusic.core.render_package import ContractViolation

        midi = mido.MidiFile(self.ctx.package.midi_path)
        duplicate = mido.MidiTrack(msg.copy() for msg in midi.tracks[1])
        for msg in duplicate:
            if hasattr(msg, 'channel'):
                msg.channel = 2
        midi.tracks.append(duplicate)
        midi.save(self.ctx.package.midi_path)
        with self.assertRaisesRegex(ContractViolation, 'track_names|source_track_identity'):
            self.reconcile()
        duplicate[0].name = 'other'
        for msg in duplicate:
            if hasattr(msg, 'channel'):
                msg.channel = 1
        midi.save(self.ctx.package.midi_path)
        with self.assertRaisesRegex(ContractViolation, 'channel|pairing'):
            self.reconcile()

    def test_full_pipeline_receives_planner_context_and_warm_corruption_fails(self) -> None:
        import json
        from unittest.mock import patch
        from aimusic.audio.orchestrator import Orchestrator
        from aimusic.audio.groove.apply import apply_groove
        from aimusic.audio.render.stems import prepare_stem_render, render_role_stems
        from aimusic.core.render_package import ContractViolation

        config = {'stages': {'analysis': False, 'expressivization': True, 'render': True},
                  'groove': {'spec': {}}}  # Test context handoff, not conflicting shared-bend retiming.
        orch = Orchestrator(config=config, cache_dir=self.root / 'cache')

        # Use real stage outputs: C5 requires complete sidecars and WAVs before
        # publishing a cache entry; mocks that return no artifacts are invalid.
        with patch('aimusic.audio.analysis.from_midi.analyze_midi') as infer, patch(
            'aimusic.audio.groove.apply.apply_groove', wraps=apply_groove,
        ) as apply, patch('aimusic.audio.render.stems.prepare_stem_render', wraps=prepare_stem_render) as prepare, patch(
            'aimusic.audio.render.stems.render_role_stems', wraps=render_role_stems,
        ) as render:
            out = orch.run(self.ctx, output_dir=self.root / 'out')
            infer.assert_not_called()
            for stage in (apply, prepare):
                doc = stage.call_args.kwargs['analysis']
                self.assertEqual(doc['provenance'], 'planner')
                self.assertEqual(doc['tracks'][0]['source_track_index'], 1)
                self.assertEqual(doc['tracks'][0]['role'], 'lead')
                self.assertEqual(doc['tuning'], self.ctx.tuning)
                self.assertEqual(doc['verified_identity'], self.ctx.verified_identity)
            for call in prepare.call_args_list:
                self.assertEqual(call.kwargs['score'], self.ctx.score)
            self.assertTrue(json.loads((out / 'score.reconciliation.json').read_text())['passed'])
            self.ctx.package.midi_path.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ContractViolation, 'score.mid.*expected=.*actual='):
                orch.run(self.ctx, output_dir=self.root / 'warm')
            self.assertFalse((self.root / 'warm').exists())
            self.assertEqual(apply.call_count, 1)
            self.assertEqual(render.call_count, 1)

    def test_planner_roles_control_expressive_output_not_source_names(self) -> None:
        import json
        from aimusic.audio.midi_events import MidiEvents
        from aimusic.audio.orchestrator import Orchestrator

        config = {'stages': {'expressivization': True, 'render': True},
                  'groove': {'spec': {'microtiming_ms': {
                      'lead': {'global_offset': 12.5}, 'bass': {'global_offset': 37.5},
                      'comp': {'global_offset': 25}}}}}
        for name, role, ticks in [('texture', 'lead', 12), ('bassoon', 'lead', 12), ('comping', 'comping', 24)]:
            with self.subTest(name=name):
                ctx = self.package(name=name, track_programs={name: 4 if name == 'comping' else 0})
                original = ctx.package.midi_path.read_bytes()
                out = Orchestrator(config, self.root / 'role-cache').run(ctx, output_dir=self.root / name)
                expressive = MidiEvents.load(out / 'score.expressive.mid')
                self.assertEqual([(n.start, n.end) for n in expressive.notes],
                                 [(ticks, 480 + ticks), (240 + ticks, 720 + ticks)])
                diff = json.loads((out / 'score.expressive.mid.diff.json').read_text())
                self.assertEqual([(r['role'], r['offset_ticks']) for r in diff['note_diffs']], [(role, ticks)] * 2)
                index = json.loads((out / 'stems/role_index.json').read_text())['stems']
                self.assertEqual([(r['source_track_index'], r['role']) for r in index], [(1, role)])
                self.assertEqual(json.loads((out / 'render.json').read_text())['renderers'][0]['role'], role)
                self.assertTrue(json.loads((out / 'score.reconciliation.json').read_text())['passed'])
                self.assertEqual(ctx.package.midi_path.read_bytes(), original)

    def test_orchestrator_rejects_valid_replacement_until_context_is_reloaded(self) -> None:
        import shutil
        from aimusic.audio.from_score import validate_package
        from aimusic.audio.orchestrator import Orchestrator
        from aimusic.core.render_package import ContractViolation

        replacement = self.package(edo=19)
        for path in replacement.package.root.iterdir():
            shutil.copy2(path, self.ctx.package.root / path.name)
        reloaded = from_score(self.ctx.package.root)
        self.assertNotEqual(reloaded.package.content_hash, self.ctx.package.content_hash)
        self.assertTrue(validate_package(reloaded).reconciliation.passed)
        orch = Orchestrator(
            config={'stages': {'analysis': False, 'expressivization': False, 'render': False}},
            cache_dir=self.root / 'cache',
        )
        out = self.root / 'stale'
        with self.assertRaisesRegex(ContractViolation, 'identity changed.*expected=.*actual=') as error:
            orch.run(self.ctx, output_dir=out)
        self.assertIn(self.ctx.package.content_hash, str(error.exception))
        self.assertIn(reloaded.package.content_hash, str(error.exception))
        self.assertFalse(out.exists())
        self.assertEqual(orch.run(reloaded, output_dir=out), out)

    def test_valid_hashes_do_not_bypass_reconciliation_before_groove(self) -> None:
        from unittest.mock import patch
        from aimusic.audio.analysis.reconcile import ReconciliationError
        from aimusic.audio.orchestrator import Orchestrator

        midi = mido.MidiFile(self.ctx.package.midi_path)
        next(m for m in midi.tracks[1] if m.type == 'program_change').program = 1
        midi.save(self.root / 'mismatch.mid')
        package = write_render_package(
            self.root, score=self.ctx.score, midi_path=self.root / 'mismatch.mid',
            manifest=RunManifest(seed=2, config_dump={}), track_programs={'texture': 0},
        )
        with patch('aimusic.audio.groove.apply.apply_groove') as apply, patch(
            'aimusic.audio.render.stems.render_role_stems'
        ) as render:
            orch = Orchestrator(config={}, cache_dir=self.root / 'cache')
            with self.assertRaisesRegex(ReconciliationError, 'texture.*program'):
                orch.run(from_score(package.root), output_dir=self.root / 'out')
            apply.assert_not_called()
            render.assert_not_called()
            self.assertFalse((self.root / 'out').exists())
