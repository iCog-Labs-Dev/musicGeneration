"""Asset-free tuning admission: intent and channel evidence, never label-only."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import mido

from aimusic.audio.midi_events import MidiEvents
from aimusic.audio.render.registry import UnsupportedTuning, required_tuning
from aimusic.audio.render.stems import prepare_stem_render, render_role_stems
from aimusic.core.core_types import NoteEvent, Score


def cc(control: int, value: int, channel: int = 0) -> Any:
    return mido.Message('control_change', control=control, value=value, channel=channel)


def rpn(parameter: int, high: int, low: int = 0, channel: int = 0) -> list[Any]:
    return [cc(101, parameter // 128, channel), cc(100, parameter % 128, channel),
            cc(6, high, channel), cc(38, low, channel)]


def events(*setup: Any, channel: int = 0, name: str = 'lead', note: int = 60) -> MidiEvents:
    midi = mido.MidiFile()
    midi.tracks = [mido.MidiTrack([mido.MetaMessage('track_name', name=name), *setup,
                                  mido.Message('note_on', note=note, channel=channel),
                                  mido.Message('note_off', note=note, channel=channel, time=480)])]
    return MidiEvents.from_mido(midi)


class TestRequiredTuning(unittest.TestCase):
    def test_ordinary_direct12_and_neutral_parameter_setup(self):
        for setup in ([], rpn(0, 2), rpn(1, 64), rpn(2, 64, 127),
                      [mido.Message('pitchwheel', pitch=0)]):
            with self.subTest(setup=setup):
                self.assertEqual(required_tuning(events(*setup), {
                    'edo': 12, 'base_tuning': 60, 'tuning': {'method': 'direct_12'},
                }), 'direct_12')

    def test_fractional_base_overrides_direct12_and_false_microtonal_labels(self):
        for base in (0.000001, 60.25, -0.25):
            for metadata in ({'base_tuning': base}, {'tuning': {'base_tuning': base}}):
                with self.subTest(base=base, metadata=metadata):
                    self.assertEqual(required_tuning(events(), {
                        'edo': 12, 'method': 'direct_12', 'microtonal': False, **metadata,
                    }), 'channel_pitch')

    def test_every_nonzero_bend_requires_channel_pitch_even_outside_note_lifetime(self):
        for pitch in (-8192, -1, 1, 8191):
            with self.subTest(pitch=pitch):
                self.assertEqual(required_tuning(events(
                    mido.Message('pitchwheel', pitch=pitch),
                    mido.Message('pitchwheel', pitch=0)), {}), 'channel_pitch')

    def test_nonstandard_range_requires_capability_even_without_a_bend(self):
        for high, low in ((0, 0), (1, 0), (2, 1), (12, 50), (48, 0)):
            with self.subTest(high=high, low=low):
                self.assertEqual(required_tuning(events(*rpn(0, high, low)), {}), 'channel_pitch')

    def test_fine_coarse_and_partial_data_entry_retain_parameter_state(self):
        for setup in (rpn(1, 64, 1), rpn(1, 63), rpn(2, 65), rpn(2, 63),
                      [cc(101, 0), cc(100, 0), cc(38, 1)],
                      [*rpn(0, 12), *rpn(0, 2)]):
            with self.subTest(setup=setup):
                self.assertEqual(required_tuning(events(*setup), {}), 'channel_pitch')

    def test_rpn_null_data_is_noop_and_selectors_do_not_leak_across_channels(self):
        setup = [*rpn(1, 64), cc(101, 127), cc(100, 127), cc(6, 7), cc(96, 0),
                 cc(6, 9, channel=1), cc(101, 0, channel=2), cc(100, 0, channel=2),
                 cc(6, 2, channel=2), cc(38, 0, channel=2)]
        self.assertEqual(required_tuning(events(*setup), {}), 'direct_12')

    def test_reset_all_controllers_nulls_rpn_and_nrpn_data_targets(self):
        for selection in ([cc(101, 0), cc(100, 0)], [cc(99, 0), cc(98, 0)]):
            for control, value in ((6, 48), (38, 50), (96, 0), (97, 0)):
                with self.subTest(selection=selection, control=control):
                    self.assertEqual(required_tuning(events(
                        *selection, cc(121, 0), cc(control, value)), {}), 'direct_12')

    def test_reset_clears_both_selector_halves_including_inactive_family(self):
        # Leave both families selected before resetting; selecting either family
        # afterwards must not recover the other byte from before CC121.
        selection = [cc(101, 0), cc(100, 0), cc(99, 0), cc(98, 0)]
        for control in (101, 100, 99, 98):
            for value, expected in ((0, 'unsupported'), (127, 'direct_12')):
                with self.subTest(control=control, value=value):
                    self.assertEqual(required_tuning(events(
                        *selection, cc(121, 0), cc(control, value), cc(6, 2)), {}), expected)

    def test_reset_parameter_selection_is_channel_local(self):
        for other, expected in (([cc(101, 0, 1), cc(100, 0, 1)], 'channel_pitch'),
                                ([cc(99, 0, 1), cc(98, 0, 1)], 'unsupported')):
            with self.subTest(other=other):
                self.assertEqual(required_tuning(events(
                    cc(101, 0), cc(100, 0), *other, cc(121, 0),
                    cc(6, 48), cc(6, 48, 1)), {}), expected)

    def test_reset_keeps_accumulated_requirements_after_reselection(self):
        for setup, expected in ((rpn(0, 48, 50), 'channel_pitch'),
                                (rpn(1, 65, 1), 'channel_pitch'),
                                (rpn(2, 65), 'channel_pitch'),
                                (rpn(6, 4), 'mpe'),
                                ([cc(99, 0), cc(98, 0), cc(6, 1)], 'unsupported'),
                                ([mido.Message('pitchwheel', pitch=1)], 'channel_pitch')):
            with self.subTest(setup=setup):
                # Reselect explicitly after reset. A partial write is not a
                # new default tuning or permission to erase prior evidence.
                self.assertEqual(required_tuning(events(
                    *setup, cc(121, 0), cc(101, 0), cc(100, 0), cc(6, 2)), {}), expected)

    def test_relative_parameter_changes_are_not_assumed_neutral(self):
        for control in (96, 97):
            with self.subTest(control=control):
                self.assertEqual(required_tuning(events(*rpn(0, 2), cc(control, 0)), {}), 'channel_pitch')

    def test_unknown_rpn_and_nrpn_data_fail_closed(self):
        for setup in (rpn(5, 1), [cc(99, 0), cc(98, 1), cc(6, 64)],
                      [*rpn(0, 2), cc(99, 0), cc(98, 1), cc(38, 1)]):
            with self.subTest(setup=setup):
                self.assertEqual(required_tuning(events(*setup), {}), 'unsupported')
        self.assertEqual(required_tuning(events(cc(99, 0), cc(98, 1), *rpn(0, 2)), {}), 'direct_12')

    def test_mpe_metadata_and_zone_configuration_are_distinguished(self):
        self.assertEqual(required_tuning(events(), {'tuning': {'method': 'mpe'}}), 'channel_pitch')
        for master in (0, 15):
            with self.subTest(master=master):
                self.assertEqual(required_tuning(events(*rpn(6, 4, channel=master)), {}), 'mpe')
        self.assertEqual(required_tuning(events(*rpn(6, 0)), {}), 'direct_12')

    def test_mts_is_identified_after_channel_evidence_and_other_sysex_stays_unverified(self):
        for universal in (0x7e, 0x7f):
            with self.subTest(universal=universal):
                self.assertEqual(required_tuning(events(
                    mido.Message('pitchwheel', pitch=1),
                    mido.Message('sysex', data=(universal, 3, 8, 2))), {}), 'mts')
        self.assertEqual(required_tuning(events(), {'tuning': {'method': 'mts'}}), 'mts')
        self.assertEqual(required_tuning(events(mido.Message('sysex', data=(0x43, 8, 2))), {}), 'sysex')

    def test_track_label_cannot_hide_stronger_metadata_or_event_evidence(self):
        metadata = {'tuning': {'edo': 19, 'method': 'mpe', 'per_track': {'lead': 'direct_12'}},
                    'tracks': [{'name': 'lead', 'microtonal': False, 'method': 'direct_12'}]}
        self.assertEqual(required_tuning(events(), metadata, track_name='lead'), 'channel_pitch')
        self.assertEqual(required_tuning(events(), {
            'tuning': {'method': 'direct_12', 'per_track': {'lead': 'mts'}},
        }, track_name='lead'), 'mts')

    def test_flat_tuning_per_track_metadata_is_not_ignored(self):
        self.assertEqual(required_tuning(events(), {
            'method': 'direct_12', 'per_track': {'lead': 'mts'},
        }, track_name='lead'), 'mts')

    def test_score_pitches_use_edo_steps_not_midi_keys(self):
        for height, expected in ((1, 'channel_pitch'), (19, 'direct_12')):
            score = Score((NoteEvent(0, 480, height, 0.8, track='lead'),))
            with self.subTest(height=height):
                self.assertEqual(required_tuning(events(note=round(60 + height * 12 / 19)), {
                    'edo': 19, 'base_tuning': 60, 'method': 'direct_12',
                }, score=score, track_name='lead'), expected)
        # Without Score evidence a non-12 lattice is conservatively gated.
        self.assertEqual(required_tuning(events(), {'edo': 19}), 'channel_pitch')

    def test_score_track_selection_does_not_certify_or_contaminate_other_tracks(self):
        score = Score((NoteEvent(0, 480, 19, 0.8, track='lead'),
                       NoteEvent(0, 480, 1, 0.8, track='bass')))
        self.assertEqual(required_tuning(events(note=12), {'edo': 19}, score=score, track_name='lead'), 'direct_12')
        self.assertEqual(required_tuning(events(), {'edo': 19}, score=score, track_name='bass'), 'channel_pitch')
        with self.assertRaisesRegex(ValueError, 'missing.*Score|Score.*missing'):
            required_tuning(events(), {'edo': 19}, score=score, track_name='missing')

    def test_invalid_tuning_metadata_and_out_of_range_score_fail_explicitly(self):
        for metadata in ({'edo': 0}, {'edo': 12.5}, {'edo': float('nan')},
                         {'base_tuning': float('inf')}, {'pitch_bend_range': -1}):
            with self.subTest(metadata=metadata), self.assertRaisesRegex(ValueError, 'tuning|edo|range'):
                required_tuning(events(), metadata)
        score = Score((NoteEvent(0, 480, 128, 0.8, track='lead'),))
        with self.assertRaisesRegex(ValueError, 'pitch.*range'):
            required_tuning(events(), {}, score=score, track_name='lead')

    def test_metadata_range_and_unknown_method_are_not_hidden_by_direct_label(self):
        self.assertEqual(required_tuning(events(), {
            'tuning': {'method': 'direct_12', 'pitch_bend_range': 48},
        }), 'channel_pitch')
        self.assertEqual(required_tuning(events(), {'method': 'custom-temperament'}), 'unsupported')

    def test_drum_metadata_cannot_hide_melodic_notes_and_sysex_is_still_unverified(self):
        with self.assertRaisesRegex(ValueError, 'percussion.*melodic'):
            required_tuning(events(), {}, drums=True)
        source = events(mido.Message('sysex', data=(0x7e, 127, 8, 2)), channel=9)
        self.assertEqual(required_tuning(source, {'edo': 19}, drums=True), 'mts')

    def test_percussion_remains_drum_key_based_not_edo_pitch(self):
        source = events(*rpn(0, 48, channel=9), *rpn(1, 65, channel=9),
                        mido.Message('pitchwheel', pitch=100, channel=9), channel=9, name='drums')
        score = Score((NoteEvent(0, 480, 60, 0.8, track='drums'),))
        metadata = {'edo': 19, 'base_tuning': 60.25, 'tuning': {'method': 'mpe'}}
        for drums in (True, False):
            with self.subTest(drums=drums):
                self.assertEqual(required_tuning(source, metadata, score=score,
                                                 track_name='drums', drums=drums), 'direct_12')
        self.assertEqual(source.notes[0].pitch, 60)
        self.assertEqual(source.notes[0].channel, 9)
        # A name/role is not evidence of percussion when channel notes are melodic.
        self.assertEqual(required_tuning(events(name='drums'), metadata), 'channel_pitch')


class TestTuningPreflight(unittest.TestCase):
    def test_nonstandard_source_range_is_rejected_before_render_or_publication(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source.mid'
            events(*rpn(0, 48)).save(source)
            before = source.read_bytes()
            with patch('aimusic.audio.render.stems.render_stem') as render:
                with self.assertRaisesRegex(UnsupportedTuning, 'channel_pitch'):
                    render_role_stems(source, output_dir=root / 'out')
            render.assert_not_called()
            self.assertFalse((root / 'out').exists())
            self.assertEqual(source.read_bytes(), before)

    def test_preflight_uses_score_and_never_upgrades_real_backend_capabilities(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source.mid'
            events(note=12).save(source)
            score = Score((NoteEvent(0, 480, 19, 0.8, track='lead'),))
            plan = prepare_stem_render(source, analysis={'edo': 19}, score=score)
            self.assertEqual(plan.renderers[0].required_tuning, 'direct_12')
            self.assertEqual(plan.renderers[0].backend.capabilities.tuning_modes, ('direct_12',))
            bad = Score((NoteEvent(0, 480, 1, 0.8, track='lead'),))
            with self.assertRaisesRegex(UnsupportedTuning, 'channel_pitch'):
                prepare_stem_render(source, analysis={'edo': 19}, score=bad)

    def test_track_metadata_uses_source_identity_even_without_a_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'source.mid'
            events().save(source)
            with self.assertRaisesRegex(UnsupportedTuning, 'channel_pitch'):
                prepare_stem_render(source, analysis={
                    'tracks': [{'source_track_index': 0, 'microtonal': True}],
                })

    def test_original_midi_is_preflighted_before_groove_can_remove_evidence(self):
        from aimusic.audio.orchestrator import Orchestrator
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source.mid'
            events(*rpn(0, 48)).save(source)
            runner = Orchestrator({'stages': {'analysis': False, 'expressivization': True}}, root / 'cache')
            with patch('aimusic.audio.groove.apply.apply_groove', side_effect=AssertionError('groove ran')) as groove:
                with self.assertRaisesRegex(UnsupportedTuning, 'channel_pitch'):
                    runner.run(source, output_dir=root / 'out')
            groove.assert_not_called()
            self.assertEqual(list((root / 'out').rglob('*.wav')), [])
            self.assertEqual(list((root / 'cache').rglob('cache_manifest.json')), [])
