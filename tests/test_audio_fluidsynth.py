"""Asset-free boundary tests; these do not certify native sound or tuning."""
from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import mido
import numpy as np
import soundfile as sf

from aimusic.audio.render import registry
from tests.test_audio_stems import _event_track


class TestFluidSynth(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.asset = self.root / 'fake.sf2'
        self.asset.write_bytes(b'mock asset, not a SoundFont')
        self.source = self.root / 'source.mid'
        self.write_midi()
        self.request = registry.RenderRequest(self.source, self.root / 'out.wav', {}, 'direct_12',
                                              8000, 1.5, 0.25, self.asset, {})
        self.position = 0
        self.events = []
        self.synth = Mock()
        self.synth.sfload.return_value = 7
        for name in ('program_select', 'bank_select', 'cc', 'pitch_bend', 'noteon', 'noteoff'):
            getattr(self.synth, name).side_effect = self.record(name)
        self.synth.get_samples.side_effect = self.samples
        self.binding = SimpleNamespace(Synth=Mock(side_effect=self.create_synth))

    def record(self, name):
        def call(*args):
            self.events.append((self.position, name, args))
            return 0
        return call

    def create_synth(self, **kwargs):
        settings = {**kwargs, 'synth.sample-rate': float(kwargs['samplerate']),
                    'synth.gain': kwargs['gain'], 'synth.midi-channels': kwargs['channels']}
        self.synth.get_setting.side_effect = settings.get
        return self.synth

    def samples(self, count):
        self.assertGreater(count, 0)
        self.assertLessEqual(count, 1024)
        self.position += count
        return np.tile(np.array([1000, -2000], dtype=np.int16), count)

    def write_midi(self, extra=None):
        midi = mido.MidiFile(type=1, ticks_per_beat=480)
        controls = [(0, mido.Message('control_change', channel=2, control=c, value=v))
                    for c, v in ((0, 1), (32, 3), (101, 0), (100, 0), (6, 12), (38, 50), (64, 127))]
        if extra:
            controls.insert(0, (0, extra))
        midi.tracks = [_event_track((0, mido.MetaMessage('set_tempo', tempo=500000)),
                                    (480, mido.MetaMessage('set_tempo', tempo=1000000)),
                                    (960, mido.MetaMessage('end_of_track'))),
                       _event_track(*controls,
                                    (0, mido.Message('program_change', channel=2, program=81)),
                                    (240, mido.Message('pitchwheel', channel=2, pitch=-4096)),
                                    (240, mido.Message('note_on', channel=2, note=60, velocity=90)),
                                    (480, mido.Message('note_on', channel=2, note=60, velocity=0)),
                                    (480, mido.Message('pitchwheel', channel=2, pitch=8191)),
                                    (480, mido.Message('note_on', channel=2, note=62, velocity=80)),
                                    (720, mido.Message('note_off', channel=2, note=62)),
                                    (960, mido.Message('control_change', channel=2, control=64, value=0))),
                       _event_track((240, mido.Message('note_on', channel=9, note=36, velocity=100)),
                                    (480, mido.Message('note_off', channel=9, note=36)))]
        midi.save(self.source)

    def render(self, request=None):
        from aimusic.audio.render import fluidsynth_r
        return fluidsynth_r.render(request or self.request, binding=self.binding)

    def test_tempo_sample_positions_order_stereo_and_fixed_tail(self):
        self.render()
        self.assertEqual(self.position, 14000)
        self.assertEqual([e for e in self.events if e[1] in ('pitch_bend', 'noteon', 'noteoff')], [
            (2000, 'pitch_bend', (2, -4096)), (2000, 'noteon', (2, 60, 90)),
            (2000, 'noteon', (9, 36, 100)), (4000, 'noteoff', (2, 60)),
            (4000, 'pitch_bend', (2, 8191)), (4000, 'noteon', (2, 62, 80)),
            (4000, 'noteoff', (9, 36)), (8000, 'noteoff', (2, 62))])
        self.assertIn((0, 'program_select', (2, 7, 131, 81)), self.events)
        self.assertIn((2000, 'program_select', (9, 7, 128, 0)), self.events)
        self.assertEqual([e[2] for e in self.events if e[1] == 'cc'],
                         [(2, c, v) for c, v in ((0, 1), (32, 3), (101, 0), (100, 0),
                                                (6, 12), (38, 50), (64, 127), (64, 0))])
        samples, rate = sf.read(self.request.output_path, always_2d=True)
        self.assertEqual((samples.shape, rate), ((14000, 2), 8000))
        np.testing.assert_array_equal(samples[0], np.array([1000, -2000]) / 32768)
        self.synth.start.assert_not_called()
        self.synth.delete.assert_called_once_with()

    def write_percussion(self, messages):
        midi = mido.MidiFile(type=0, ticks_per_beat=480)
        midi.tracks = [_event_track(*[(0, msg) for msg in messages],
                                    (240, mido.Message('note_on', channel=9, note=36, velocity=100)),
                                    (480, mido.Message('note_off', channel=9, note=36)))]
        midi.save(self.source)

    def test_percussion_default_bank_with_and_without_program_change(self):
        for program in (None, 8):
            with self.subTest(program=program):
                self.setUp()
                messages = [] if program is None else [mido.Message('program_change', channel=9, program=program)]
                self.write_percussion(messages)
                self.render()
                self.assertEqual([args for _, method, args in self.events if method == 'program_select'],
                                 [(9, 7, 128, program or 0)])

    def test_percussion_explicit_mma_bank_including_zero_and_reversed_cc_order(self):
        for controls, bank in ((((0, 1), (32, 3)), 131), (((32, 3), (0, 1)), 131),
                               (((0, 0), (32, 0)), 0)):
            with self.subTest(controls=controls):
                self.setUp()
                self.write_percussion([mido.Message('control_change', channel=9, control=c, value=v)
                                       for c, v in controls] + [mido.Message('program_change', channel=9, program=8)])
                self.render()
                self.assertEqual([args for _, method, args in self.events if method == 'cc'],
                                 [(9, c, v) for c, v in controls])
                self.assertEqual([args for _, method, args in self.events if method == 'program_select'],
                                 [(9, 7, bank, 8)])

    def test_percussion_partial_banks_retain_other_controller_and_are_channel_local(self):
        # Never-seen CC halves are zero, not an invented CC0=1 for the drum default.
        for first_control, first_value, first_bank in ((0, 2, 256), (32, 3, 3), (32, 0, 0)):
            with self.subTest(first_control=first_control, first_value=first_value):
                self.setUp()
                controls = [(first_control, first_value), (0, 1), (32, 3), (0, 2), (32, 4)]
                # Program changes, other CCs (including reset-all-controllers), and
                # bank messages on another channel must not clear either bank half.
                messages = [mido.Message('program_change', channel=9, program=0)]
                for program, (control, value) in enumerate(controls, 1):
                    messages += [mido.Message('control_change', channel=9, control=control, value=value),
                                 mido.Message('control_change', channel=9, control=121, value=0),
                                 mido.Message('control_change', channel=2, control=32, value=99),
                                 mido.Message('program_change', channel=9, program=program)]
                messages.append(mido.Message('program_change', channel=9, program=6))
                self.write_percussion(messages)
                self.render()
                after_msb = 131 if first_control == 32 and first_value == 3 else 128
                expected_banks = [128, first_bank, after_msb, 131, 259, 260, 260]
                self.assertEqual([args for _, method, args in self.events if method == 'program_select'],
                                 [(9, 7, bank, program) for program, bank in enumerate(expected_banks)])

    def test_percussion_bank_ccs_wait_for_program_change(self):
        midi = mido.MidiFile(type=0, ticks_per_beat=480)
        midi.tracks = [_event_track(
            (0, mido.Message('control_change', channel=9, control=0, value=1)),
            (0, mido.Message('control_change', channel=9, control=32, value=3)),
            (0, mido.Message('note_on', channel=9, note=36, velocity=100)),
            (120, mido.Message('note_off', channel=9, note=36)),
            (240, mido.Message('program_change', channel=9, program=8)),
            (240, mido.Message('note_on', channel=9, note=38, velocity=100)),
            (360, mido.Message('control_change', channel=9, control=32, value=4)),
            (480, mido.Message('note_off', channel=9, note=38)),
            (480, mido.Message('note_on', channel=9, note=36, velocity=100)),
            (600, mido.Message('note_off', channel=9, note=36)),
            (720, mido.Message('program_change', channel=9, program=8)))]
        midi.save(self.source)
        self.render()
        self.assertEqual([event for event in self.events if event[1] == 'program_select'],
                         [(0, 'program_select', (9, 7, 128, 0)),
                          (2000, 'program_select', (9, 7, 131, 8)),
                          (6000, 'program_select', (9, 7, 132, 8))])

    def test_unavailable_explicit_percussion_preset_fails_without_substitution(self):
        self.write_percussion([mido.Message('control_change', channel=9, control=0, value=1),
                               mido.Message('control_change', channel=9, control=32, value=3),
                               mido.Message('program_change', channel=9, program=8)])
        self.synth.program_select.side_effect = None
        self.synth.program_select.return_value = -1
        with self.assertRaisesRegex(RuntimeError, 'program_select'):
            self.render()
        self.synth.program_select.assert_called_once_with(9, 7, 131, 8)
        self.synth.noteon.assert_not_called()
        self.synth.delete.assert_called_once_with()
        self.assertFalse(self.request.output_path.exists())

    def test_no_cumulative_sample_rounding_drift(self):
        midi = mido.MidiFile(type=0, ticks_per_beat=960)
        midi.tracks = [_event_track(*[(tick, mido.Message('control_change', control=7, value=90))
                                     for tick in range(1, 101)])]
        midi.save(self.source)
        self.render()
        self.assertEqual([e[0] for e in self.events if e[1] == 'cc'],
                         [round(tick * 25 / 6) for tick in range(1, 101)])

    def test_unsupported_events_fail_before_allocation_or_wav(self):
        for message in (mido.Message('sysex', data=(0x7e, 0x7f, 8, 2)),
                        mido.Message('aftertouch', value=30), mido.Message('polytouch', note=60)):
            with self.subTest(message=message):
                self.write_midi(extra=message)
                with self.assertRaisesRegex(ValueError, 'Unsupported.*' + message.type):
                    self.render()
                self.binding.Synth.assert_not_called()
                self.assertFalse(self.request.output_path.exists())

    def test_lifecycle_cleanup_on_native_errors_and_invalid_samples(self):
        for method, result in (('sfload', -1), ('program_select', -1), ('cc', -1),
                               ('noteon', -1), ('get_samples', np.zeros(3, dtype=np.int16))):
            with self.subTest(method=method):
                self.setUp()
                getattr(self.synth, method).side_effect = None
                getattr(self.synth, method).return_value = result
                with self.assertRaisesRegex(RuntimeError, 'FluidSynth'):
                    self.render()
                self.synth.delete.assert_called_once_with()
                self.assertFalse(self.request.output_path.exists())

    def test_cleanup_preserves_primary_failure_and_write_failure(self):
        self.synth.get_samples.side_effect = RuntimeError('original failure')
        self.synth.delete.side_effect = RuntimeError('cleanup failure')
        with self.assertRaisesRegex(RuntimeError, 'original failure'):
            self.render()
        self.synth.delete.assert_called_once_with()
        self.setUp()
        with patch('soundfile.write', side_effect=OSError('disk full')):
            with self.assertRaisesRegex(OSError, 'disk full'):
                self.render()
        self.synth.delete.assert_called_once_with()

    def test_rejected_setting_and_short_horizon(self):
        for request in (replace(self.request, sample_rate=1), replace(self.request, duration_sec=0.1)):
            with self.assertRaises(ValueError):
                self.render(request)
        self.binding.Synth.assert_not_called()
        self.binding.Synth.side_effect = lambda **kw: self.synth
        self.synth.get_setting.return_value = None
        with self.assertRaisesRegex(RuntimeError, 'setting'):
            self.render()
        self.synth.delete.assert_called_once_with()

    def test_noteoff_for_already_decayed_native_voice_is_not_a_render_failure(self):
        # Official noteoff docs: -1 may merely mean no voices matched the key.
        # SMF note pairing was already validated independently.
        self.synth.noteoff.side_effect = None
        self.synth.noteoff.return_value = -1
        self.render()
        self.synth.delete.assert_called_once_with()

    def test_binding_resource_classification_does_not_hide_import_defects(self):
        from importlib.metadata import PackageNotFoundError
        from aimusic.audio.render import fluidsynth_r
        with patch.object(fluidsynth_r, 'version', side_effect=PackageNotFoundError):
            with self.assertRaisesRegex(registry.MissingRendererResource, 'audio-render'):
                fluidsynth_r.load_binding()
        cases = [(ImportError("Couldn't find the FluidSynth library."), registry.MissingRendererResource),
                 (OSError('DLL load failed'), registry.MissingRendererResource),
                 (ImportError('transitive defect'), ImportError), (RuntimeError('defect'), RuntimeError)]
        for error, expected in cases:
            with self.subTest(error=error), patch.object(fluidsynth_r, 'version', return_value='1.3.4'), \
                    patch.object(fluidsynth_r.importlib, 'import_module', side_effect=error):
                with self.assertRaises(expected) as caught:
                    fluidsynth_r.load_binding()
                self.assertIs(type(caught.exception), expected)

    def test_public_native_version_query_without_allocation(self):
        from aimusic.audio.render import fluidsynth_r

        def native_version(major, minor, micro):
            major.value, minor.value, micro.value = 2, 4, 6

        self.binding.fluid_version = native_version
        with patch.object(fluidsynth_r, 'version', return_value='1.3.4'):
            self.assertIn(('libfluidsynth', '2.4.6'), fluidsynth_r.versions(self.binding))
        self.binding.Synth.assert_not_called()

    def test_registry_probes_without_synth_and_records_versions_settings(self):
        from aimusic.audio.render import fluidsynth_r
        with patch.object(fluidsynth_r, 'load_binding', return_value=self.binding), \
                patch.object(fluidsynth_r, 'versions', return_value=(('pyfluidsynth', '1.3.4'), ('libfluidsynth', 'mock-2'))):
            selected = registry.resolve_renderer('lead', config={'render': {'backend': 'fluidsynth'}}, soundfont=self.asset)
            self.binding.Synth.assert_not_called()
            self.assertEqual(selected.backend.capabilities.tuning_modes, ('direct_12',))
            # Adapter dispatch tested above is not a native MPE certification.
            with self.assertRaises(registry.UnsupportedTuning):
                registry.resolve_renderer('lead', config={'render': {'backend': 'fluidsynth'}},
                                          soundfont=self.asset, tuning='channel_pitch')
            registry.render_request(self.request, selected)
        data = json.loads(Path(str(self.request.output_path) + '.json').read_text())
        self.assertIn(['libfluidsynth', 'mock-2'], data['native_versions'])
        self.assertEqual(data['render_settings']['synth.reverb.active'], 0)
        self.assertEqual(data['render_settings']['synth.sample-rate'], 8000)
        self.assertEqual(data['frames'], 14000)
        self.assertEqual(data['effective_backend'], 'fluidsynth')
        self.assertIsNone(data['fallback_reason'])

    def test_direct_native_channel_pitch_request_cannot_bypass_subset_preflight(self):
        # Existing dispatch fixture uses unproven 12.5-semitone RPN + sustain.
        with self.assertRaisesRegex(registry.UnsupportedTuning, 'channel_pitch subset'):
            self.render(replace(self.request, required_tuning='channel_pitch'))
        self.binding.Synth.assert_not_called()
        self.assertFalse(self.request.output_path.exists())
