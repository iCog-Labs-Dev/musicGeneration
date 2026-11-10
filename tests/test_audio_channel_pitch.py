"""Asset-free boundaries for the native-proven, narrow channel-pitch subset."""
from __future__ import annotations

import io
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mido

from aimusic.audio.demo_sources import build_source
from aimusic.audio.groove.apply import apply_groove
from aimusic.audio.midi_events import MidiEvents
from aimusic.audio.render import fluidsynth_r, registry, tuning
from aimusic.audio.render.stems import prepare_stem_render


class TestChannelPitch(unittest.TestCase):
    def test_original_expected_hz_after_groove_and_isolation(self):
        for name in ('reference-19', 'multitrack-19'):
            payload, intent = build_source(name)
            with self.subTest(source=name), tempfile.TemporaryDirectory() as tmp:
                source = Path(tmp) / 'source.mid'
                source.write_bytes(payload)
                expressive = apply_groove(source, {'microtiming_ms': {
                    role: {'global_offset': 12.5} for role in ('bass', 'comp', 'lead')}},
                    output_path=Path(tmp) / 'expressive.mid')
                for path in (source, expressive):
                    for index, midi in MidiEvents.load(path).isolate_tracks().items():
                        evidence = tuning.channel_pitch_evidence(midi)
                        self.assertEqual(evidence['subset'], 'rpn0-2-semitones-v1')
                        self.assertEqual(evidence['audio_pitch_accuracy'], 'unmeasured')
                        expected = [n for n in intent['notes'] if n['track'] == index]
                        self.assertEqual(len(evidence['notes']), len(expected))
                        for actual, target in zip(evidence['notes'], expected):
                            self.assertEqual((actual['track'], actual['channel'], actual['midi_note']),
                                             (target['track'], target['channel'], target['midi_note']))
                            hz = 440 * 2 ** (target['edo_step_from_a4'] / 19)
                            self.assertLessEqual(abs(1200 * math.log2(actual['hz'] / hz)), 1)
                            self.assertEqual(actual['tick'], target['start_tick'] + (12 if path == expressive else 0))

    def test_unproven_controls_ranges_incomplete_setup_overlap_and_live_bends_reject(self):
        payload, _ = build_source('reference-19')
        cases = []
        for control, value in ((6, 12), (38, 50), (100, 1), (100, 2), (100, 6),
                               (98, 0), (99, 0), (96, 1), (97, 1), (64, 127), (1, 100)):
            midi = mido.MidiFile(file=io.BytesIO(payload))
            midi.tracks[1].insert(1, mido.Message('control_change', channel=1, control=control, value=value))
            if control == 64:
                midi.tracks[1].insert(-1, mido.Message('control_change', channel=1, control=64, value=0))
            cases.append(midi)
        for control in (101, 100, 6, 38):
            midi = mido.MidiFile(file=io.BytesIO(payload))
            index = next(i for i, m in enumerate(midi.tracks[1]) if m.type == 'control_change' and m.control == control)
            del midi.tracks[1][index]
            cases.append(midi)
        midi = mido.MidiFile(file=io.BytesIO(payload))
        index = next(i for i, m in enumerate(midi.tracks[1]) if m.type == 'note_on')
        midi.tracks[1].insert(index + 1, mido.Message('pitchwheel', channel=1, pitch=1))
        cases.append(midi)
        midi = mido.MidiFile(file=io.BytesIO(payload))
        midi.tracks[1].insert(index + 1, mido.Message('note_on', channel=1, note=80, velocity=90))
        midi.tracks[1].insert(index + 2, mido.Message('note_off', channel=1, note=80, time=1))
        midi.tracks[1][index + 3].time -= 1
        cases.append(midi)
        for i, midi in enumerate(cases):
            with self.subTest(case=i), self.assertRaisesRegex(registry.UnsupportedTuning, 'channel_pitch subset'):
                tuning.channel_pitch_evidence(MidiEvents.from_mido(midi))

    def test_only_proven_runtime_promotes_mode_and_no_simple_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            asset = Path(tmp) / 'fake.sf2'
            asset.write_bytes(b'fake for probe only')
            for version, supported in [('2.6.2', True), ('2.4.6', False), ('2.6.3', False), ('3.0.0', False)]:
                with self.subTest(version=version), patch.object(fluidsynth_r, 'load_binding'), \
                        patch.object(fluidsynth_r, 'versions', return_value=(('libfluidsynth', version), ('pyfluidsynth', '1.3.4'))):
                    config = {'render': {'backend': 'fluidsynth'}}
                    if supported:
                        choice = registry.resolve_renderer('lead', config=config, soundfont=asset, tuning='channel_pitch')
                        self.assertIn('channel_pitch', choice.backend.capabilities.tuning_modes)
                        for mode in ('mpe', 'mts', 'sysex', 'unsupported'):
                            with self.assertRaises(registry.UnsupportedTuning):
                                registry.resolve_renderer('lead', config=config, soundfont=asset, tuning=mode)
                    else:
                        with self.assertRaises(registry.UnsupportedTuning):
                            registry.resolve_renderer('lead', config=config, soundfont=asset, tuning='channel_pitch')
            with patch.object(fluidsynth_r, 'load_binding', side_effect=registry.MissingRendererResource('missing native')):
                with self.assertRaises(registry.UnsupportedTuning):
                    registry.resolve_renderer('lead', config={'render': {'backend': 'fluidsynth', 'allow_simple_fallback': True}},
                                              soundfont=asset, tuning='channel_pitch')

    def test_only_exact_binding_version_can_promote_channel_pitch(self):
        with tempfile.TemporaryDirectory() as tmp:
            asset = Path(tmp) / 'fake.sf2'
            asset.write_bytes(b'fake for probe only')
            for binding_version in ('1.3.4', '1.3.3', '1.3.5', '2.0.0', '1.3.4.post1', '1.3.4.dev1'):
                for fallback in (False, True):
                    with self.subTest(binding=binding_version, fallback=fallback), \
                            patch.object(fluidsynth_r, 'version', return_value=binding_version), \
                            patch.object(fluidsynth_r.importlib, 'import_module'), \
                            patch.object(fluidsynth_r, 'versions', return_value=(
                                ('libfluidsynth', '2.6.2'), ('pyfluidsynth', binding_version))):
                        # Keep load_binding real: its exact pin is part of the capability boundary.
                        config = {'render': {'backend': 'fluidsynth', 'allow_simple_fallback': fallback}}
                        if binding_version == '1.3.4':
                            choice = registry.resolve_renderer('lead', config=config, soundfont=asset, tuning='channel_pitch')
                            self.assertEqual(choice.backend.name, 'fluidsynth')
                            self.assertIn('channel_pitch', choice.backend.capabilities.tuning_modes)
                        else:
                            error = registry.UnsupportedTuning if fallback else registry.MissingRendererResource
                            with self.assertRaises(error):
                                registry.resolve_renderer('lead', config=config, soundfont=asset, tuning='channel_pitch')

    def test_invalid_native_subset_rejects_before_groove_cache_or_publication(self):
        from aimusic.audio.orchestrator import Orchestrator
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            asset = root / 'fake.sf2'
            asset.write_bytes(b'fake for probe only')
            payload, _ = build_source('reference-19')
            midi = mido.MidiFile(file=io.BytesIO(payload))
            next(m for m in midi.tracks[1] if m.type == 'control_change' and m.control == 6).value = 12
            source = root / 'bad.mid'
            midi.save(source)
            config = {'render': {'backend': 'fluidsynth'}, 'paths': {'soundfont': str(asset)}}
            with patch.object(fluidsynth_r, 'load_binding'), patch.object(fluidsynth_r, 'versions',
                    return_value=(('libfluidsynth', '2.6.2'), ('pyfluidsynth', '1.3.4'))), \
                    patch.object(fluidsynth_r, 'render') as native, patch('aimusic.audio.groove.apply.apply_groove') as groove:
                with self.assertRaisesRegex(registry.UnsupportedTuning, 'channel_pitch subset'):
                    prepare_stem_render(source, config=config)
                runner = Orchestrator(config, root / 'cache')
                with self.assertRaisesRegex(registry.UnsupportedTuning, 'channel_pitch subset'):
                    runner.run(source, output_dir=root / 'out')
                native.assert_not_called()
                groove.assert_not_called()
            self.assertFalse(list((root / 'out').rglob('*.wav')))
            self.assertFalse((root / 'out/render.json').exists())
            self.assertFalse(list((root / 'cache/render').rglob('cache_manifest.json')))
