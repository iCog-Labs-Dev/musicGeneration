"""Compatibility from loaded effective profiles to the existing role registry."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

import yaml

from aimusic.audio.config import load_audio_config
from aimusic.audio.orchestrator import audio_config_to_dict
from aimusic.audio.render.registry import MissingRendererResource, get_renderer, get_renderer_name


class TestAudioRegistryCompatibility(unittest.TestCase):
    def load_config(self, render: dict[str, Any]) -> dict[str, Any]:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.yaml"
            path.write_text(yaml.safe_dump({"render": render}))
            return audio_config_to_dict(load_audio_config(path))

    def test_loaded_legacy_and_canonical_routes_keep_role_overrides(self) -> None:
        for render in (
            {"role_map": {"comp": "custom"}},
            {"routing": {"comping": {"default": "custom", "microtonal": "other"}}},
        ):
            config = self.load_config(render)
            self.assertNotIn("role_map", config["render"])
            for role in ("comp", "comping"):
                with self.subTest(render=render, role=role):
                    self.assertEqual(get_renderer_name(role, config=config), "custom")
                    with self.assertRaisesRegex(KeyError, "custom"):
                        get_renderer(role, config=config)

    def test_loaded_role_route_honors_existing_fallback_policy(self) -> None:
        for allow in (True, False):
            config = self.load_config({
                "backend": "simple", "allow_simple_fallback": allow,
                "routing": {"bass": {"default": "fluidsynth"}},
            })
            with self.subTest(allow=allow):
                if allow:
                    self.assertEqual(get_renderer_name("bass", config=config), "simple")
                else:
                    with self.assertRaisesRegex(MissingRendererResource, "FluidSynth"):
                        get_renderer_name("bass", config=config)

    def test_direct_legacy_dict_and_role_alias_remain_supported(self) -> None:
        for configured_role in ("comp", "comping"):
            config = {"render": {"backend": "default", "role_map": {configured_role: "custom"}}}
            for requested_role in ("comp", "comping"):
                with self.subTest(configured=configured_role, requested=requested_role):
                    self.assertEqual(get_renderer_name(requested_role, config=config), "custom")
            self.assertEqual(get_renderer_name("bass", config=config), "default")

    def test_direct_canonical_dict_uses_default_route_or_global_backend(self) -> None:
        config = {"render": {"backend": "global", "routing": {
            "comp": {"default": "custom", "microtonal": "other"},
            "bass": {"microtonal": "other"},
        }}}
        self.assertEqual(get_renderer_name("comping", config=config), "custom")
        self.assertEqual(get_renderer_name("bass", config=config), "global")
        self.assertEqual(get_renderer_name("lead", config=config), "global")
        self.assertEqual(get_renderer_name("comp"), "simple")
        self.assertEqual(get_renderer_name("comping", config={}), "simple")


class TestVerifiedRendererContracts(unittest.TestCase):
    def test_every_stem_uses_canonical_role_route_and_records_actual_backend(self):
        import json
        from dataclasses import replace
        from unittest.mock import patch
        from aimusic.audio.render import registry
        from aimusic.audio.render.stems import render_role_stems
        from tests.test_audio_stems import _make_two_pianos_one_bass_midi
        with tempfile.TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            source = _make_two_pianos_one_bass_midi(tmp / 'source.mid')
            selected = registry.resolve_renderer('bass', config={})
            custom = replace(selected.backend, name='test-renderer', version='test-v1')
            cfg = {'render': {'routing': {'comp': {'default': 'test-renderer'}}}}
            analysis = {'tracks': [{'source_track_index': 1, 'role': 'comp', 'role_source': 'structure.json'}]}
            with patch.dict(registry._BACKENDS, {'test-renderer': lambda asset: custom}):
                outputs = render_role_stems(source, output_dir=tmp / 'out', config=cfg, analysis=analysis)
            index = json.loads(outputs['role_index'].read_text())
            self.assertEqual(index['stems'][0]['role'], 'comping')
            for row in index['stems']:
                data = json.loads((tmp / 'out' / (row['wav'] + '.json')).read_text())
                expected = 'test-renderer' if row['source_track_index'] == 1 else 'simple'
                self.assertEqual(data['requested_backend'], expected)
                self.assertEqual(data['effective_backend'], expected)
                self.assertIsNone(data['fallback_reason'])
                self.assertEqual(data['capabilities']['tuning_modes'], ['direct_12'])
                self.assertEqual(data['output_sha256'], registry.file_digest(outputs[row['id']]))

    def test_simple_rejects_metadata_and_event_tuning_before_any_output(self):
        import mido
        from aimusic.audio.render.registry import UnsupportedTuning
        from aimusic.audio.render.stems import render_role_stems
        from tests.test_audio_stems import _make_two_pianos_one_bass_midi
        cases = [({'tuning': {'edo': 19, 'method': 'mpe'}}, None),
                 ({'tuning': {'edo': 12, 'base_tuning': 0.25, 'method': 'direct_12'}}, None),
                 ({}, mido.Message('pitchwheel', pitch=123)),
                 ({}, mido.Message('sysex', data=(0x7e, 0x7f, 8, 2)))]
        for analysis, event in cases:
            with self.subTest(analysis=analysis, event=event), tempfile.TemporaryDirectory() as tmp_s:
                tmp = Path(tmp_s)
                source = _make_two_pianos_one_bass_midi(tmp / 'source.mid')
                if event:
                    midi = mido.MidiFile(source)
                    midi.tracks[1].insert(0, event)
                    midi.save(source)
                with self.assertRaisesRegex(UnsupportedTuning, 'simple.*tuning'):
                    render_role_stems(source, output_dir=tmp / 'out', analysis=analysis)
                self.assertFalse((tmp / 'out').exists())

    def test_only_classified_missing_resources_allow_compatible_fallback(self):
        from unittest.mock import patch
        from aimusic.audio.render import registry
        cfg = {'render': {'backend': 'optional', 'allow_simple_fallback': True}}
        for resource in ('binding', 'native-library', 'soundfont'):
            with self.subTest(resource=resource), patch.dict(registry._BACKENDS, {
                'optional': lambda asset: (_ for _ in ()).throw(registry.MissingRendererResource(resource))
            }):
                choice = registry.resolve_renderer('bass', config=cfg)
                self.assertEqual(choice.requested_backend, 'optional')
                self.assertEqual(choice.backend.name, 'simple')
                self.assertIn(resource, choice.fallback_reason)
                with self.assertRaises(registry.UnsupportedTuning):
                    registry.resolve_renderer('bass', config=cfg, tuning='channel_pitch')
        for error in (RuntimeError('defect'), ImportError('transitive import bug'), OSError('native crash')):
            with self.subTest(error=error), patch.dict(registry._BACKENDS, {
                'optional': lambda asset: (_ for _ in ()).throw(error)
            }), self.assertRaises(type(error)):
                registry.resolve_renderer('bass', config=cfg)

    def test_listening_and_missing_fluidsynth_never_claim_success(self):
        from aimusic.audio.render.registry import MissingRendererResource, resolve_renderer
        with self.assertRaisesRegex(MissingRendererResource, 'FluidSynth'):
            resolve_renderer('bass', config=load_audio_config('listening').to_dict())

    def test_microtonal_route_is_selected_instead_of_default(self):
        from aimusic.audio.render.registry import resolve_renderer
        config = {'render': {'routing': {'comp': {'default': 'simple', 'microtonal': 'unknown-mpe'}}}}
        with self.assertRaisesRegex(KeyError, 'unknown-mpe'):
            resolve_renderer('comping', config=config, tuning='channel_pitch')

    def test_direct12_default_bend_setup_is_not_mistaken_for_detuning(self):
        import mido
        from aimusic.audio.midi_events import MidiEvents
        from aimusic.audio.render.registry import required_tuning
        midi = mido.MidiFile()
        midi.tracks = [mido.MidiTrack([mido.Message('control_change', control=cc, value=value)
                                     for cc, value in ((101, 0), (100, 0), (6, 2), (38, 0))] +
                                    [mido.Message('pitchwheel', pitch=0), mido.Message('note_on', note=60),
                                     mido.Message('note_off', note=60, time=480)])]
        self.assertEqual(required_tuning(MidiEvents.from_mido(midi), {}), 'direct_12')
        midi.tracks[0][2].value = 48
        self.assertEqual(required_tuning(MidiEvents.from_mido(midi), {}), 'channel_pitch')
        midi.tracks[0][4].pitch = 100
        self.assertEqual(required_tuning(MidiEvents.from_mido(midi), {}), 'channel_pitch')

    def test_runtime_failure_never_falls_back_or_publishes_success(self):
        from dataclasses import replace
        from unittest.mock import patch
        from aimusic.audio.render import registry
        from aimusic.audio.render.stems import render_role_stems
        from tests.test_audio_stems import _make_two_pianos_one_bass_midi
        simple = registry.resolve_renderer('bass', config={}).backend
        for error in (registry.MissingRendererResource('resource lost at runtime'), RuntimeError('native defect')):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as tmp_s:
                tmp = Path(tmp_s)
                source = _make_two_pianos_one_bass_midi(tmp / 'source.mid')

                def fail(request):
                    raise error

                with patch.dict(registry._BACKENDS, {'broken': lambda asset: replace(simple, render=fail)}):
                    with self.assertRaises(type(error)):
                        render_role_stems(source, output_dir=tmp / 'out', config={
                            'render': {'backend': 'broken', 'allow_simple_fallback': True}})
                self.assertEqual(list((tmp / 'out').iterdir()), [])

    def test_invalid_renderer_output_never_gets_success_sidecar(self):
        import numpy as np
        import soundfile as sf
        from dataclasses import replace
        from unittest.mock import patch
        from aimusic.audio.render import registry
        from aimusic.audio.render.stems import render_role_stems
        from tests.test_audio_stems import _make_two_pianos_one_bass_midi
        simple = registry.resolve_renderer('bass', config={}).backend
        with tempfile.TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            source = _make_two_pianos_one_bass_midi(tmp / 'source.mid')

            def invalid(request):
                sf.write(request.output_path, np.array([float('nan')]), request.sample_rate, subtype='FLOAT')
                return request.output_path

            with patch.dict(registry._BACKENDS, {'broken': lambda asset: replace(simple, render=invalid)}):
                with self.assertRaisesRegex(ValueError, 'nonfinite'):
                    render_role_stems(source, output_dir=tmp / 'out', config={'render': {'backend': 'broken'}})
            self.assertEqual(list((tmp / 'out').iterdir()), [])
