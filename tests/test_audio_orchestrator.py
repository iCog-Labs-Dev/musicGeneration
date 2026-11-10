"""Unit tests for M1 native orchestrator (simple render)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "audio"
QUANTIZED = FIXTURES / "eight_bar_quantized.mid"


def _audio_spine_available() -> bool:
    try:
        import pretty_midi  # noqa: F401
        import soundfile  # noqa: F401
        import yaml  # noqa: F401
        return QUANTIZED.is_file()
    except ImportError:
        return False


@unittest.skipUnless(_audio_spine_available(), "requires pip install -e '.[audio]' + fixtures")
class TestAudioOrchestrator(unittest.TestCase):
    def _m1_config(self, tmp: Path) -> dict:
        return {
            "code_version": "0.1.0",
            "paths": {
                "cache_dir": str(tmp / "cache"),
                "output_dir": str(tmp / "out"),
                "soundfont": None,
            },
            "stages": {
                "analysis": True,
                "expressivization": True,
                "render": True,
                "prompts": False,
                "restyle": False,
                "scoring": False,
                "mixmaster": False,
            },
            "groove": {
                "spec": {
                    "swing": {
                        "ratio": 0.58,
                        "subdivision": 8,
                        "applies_to": ["drums", "bass", "comp"],
                    },
                }
            },
            "render": {
                "sample_rate": 48000,
                "peak_dbfs": -12.0,
                "backend": "simple",
                "allow_simple_fallback": True,
            },
            "budget": {"max_endpoint_calls": 40, "max_usd_estimate": 10.0},
        }

    def test_orchestrator_m1_and_cache_hit(self) -> None:
        from aimusic.audio.orchestrator import Orchestrator

        with tempfile.TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            cfg = self._m1_config(tmp)
            orch = Orchestrator(config=cfg, cache_dir=Path(cfg["paths"]["cache_dir"]))
            out1 = orch.run(QUANTIZED, output_dir=tmp / "out1")
            self.assertTrue((out1 / f"{QUANTIZED.stem}.analysis.json").is_file())
            self.assertTrue((out1 / f"{QUANTIZED.stem}.expressive.mid").is_file())
            self.assertTrue((out1 / "stems" / "rough_mix.wav").is_file())
            self.assertGreaterEqual(orch.cache_misses.get("render", 0), 1)

            orch2 = Orchestrator(config=cfg, cache_dir=Path(cfg["paths"]["cache_dir"]))
            out2 = orch2.run(QUANTIZED, output_dir=tmp / "out2")
            self.assertTrue((out2 / "stems" / "rough_mix.wav").is_file())
            self.assertGreaterEqual(orch2.cache_hits.get("analysis", 0), 1)
            self.assertGreaterEqual(orch2.cache_hits.get("expressivization", 0), 1)
            self.assertGreaterEqual(orch2.cache_hits.get("render", 0), 1)


@unittest.skipUnless(_audio_spine_available(), "requires pip install -e '.[audio]' + fixtures")
class TestAudioOrchestratorCacheSidecars(unittest.TestCase):
    """Regression tests for Bug B from PR #47 review.

    Invariant: for each cacheable stage, after ``Orchestrator.run`` returns,
    the user ``output_dir`` contains exactly the same set of files
    (primary artifact + every sidecar) as the corresponding cache entry.
    This must hold on both the cold path (cache miss) and the warm path
    (cache hit).
    """

    def _m1_config(self, tmp: Path) -> dict:
        return {
            "code_version": "0.1.0",
            "paths": {
                "cache_dir": str(tmp / "cache"),
                "output_dir": str(tmp / "out"),
                "soundfont": None,
            },
            "stages": {
                "analysis": True,
                "expressivization": True,
                "render": True,
                "prompts": False,
                "restyle": False,
                "scoring": False,
                "mixmaster": False,
            },
            "groove": {
                "spec": {
                    "swing": {
                        "ratio": 0.58,
                        "subdivision": 8,
                        "applies_to": ["drums", "bass", "comp"],
                    },
                }
            },
            "render": {
                "sample_rate": 48000,
                "peak_dbfs": -12.0,
                "backend": "simple",
                "allow_simple_fallback": True,
            },
            "budget": {"max_endpoint_calls": 40, "max_usd_estimate": 10.0},
        }

    def _expected_artifacts(self, out: Path) -> dict[str, Path]:
        """The set of files ``Orchestrator.run`` must produce for analysis
        and expressivization on a successful run."""
        return {
            "analysis_doc": out / f"{QUANTIZED.stem}.analysis.json",
            "analysis_manifest": out / f"{QUANTIZED.stem}.analysis.json.json",
            "expressive_midi": out / f"{QUANTIZED.stem}.expressive.mid",
            "expressive_diff": out / f"{QUANTIZED.stem}.expressive.mid.diff.json",
            "expressive_manifest": out / f"{QUANTIZED.stem}.expressive.mid.json",
        }

    def test_cold_path_copies_all_sidecars(self) -> None:
        """First run (cache miss) must copy every sidecar to the user output dir."""
        from aimusic.audio.orchestrator import Orchestrator

        with tempfile.TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            cfg = self._m1_config(tmp)
            orch = Orchestrator(config=cfg, cache_dir=Path(cfg["paths"]["cache_dir"]))
            out = orch.run(QUANTIZED, output_dir=tmp / "out1")

            for name, path in self._expected_artifacts(out).items():
                self.assertTrue(
                    path.is_file(),
                    msg=f"missing cold-path artifact: {name} -> {path}",
                )

    def test_warm_path_copies_all_sidecars(self) -> None:
        """Second run (cache hit) must copy every sidecar too."""
        from aimusic.audio.orchestrator import Orchestrator

        with tempfile.TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            cfg = self._m1_config(tmp)

            # Cold run to populate the cache.
            Orchestrator(
                config=cfg, cache_dir=Path(cfg["paths"]["cache_dir"])
            ).run(QUANTIZED, output_dir=tmp / "out_cold")

            # Warm run with a fresh output directory; everything must be
            # re-materialized from cache (primary + sidecars).
            orch2 = Orchestrator(
                config=cfg, cache_dir=Path(cfg["paths"]["cache_dir"])
            )
            out_warm = orch2.run(QUANTIZED, output_dir=tmp / "out_warm")

            self.assertGreaterEqual(orch2.cache_hits.get("analysis", 0), 1)
            self.assertGreaterEqual(
                orch2.cache_hits.get("expressivization", 0), 1
            )

            for name, path in self._expected_artifacts(out_warm).items():
                self.assertTrue(
                    path.is_file(),
                    msg=f"missing warm-path artifact: {name} -> {path}",
                )

    def test_warm_sidecars_byte_equal_to_cold(self) -> None:
        """Cache-hit sidecars must be byte-identical to cold-run sidecars."""
        from aimusic.audio.orchestrator import Orchestrator

        with tempfile.TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            cfg = self._m1_config(tmp)

            out_cold = Orchestrator(
                config=cfg, cache_dir=Path(cfg["paths"]["cache_dir"])
            ).run(QUANTIZED, output_dir=tmp / "out_cold")

            out_warm = Orchestrator(
                config=cfg, cache_dir=Path(cfg["paths"]["cache_dir"])
            ).run(QUANTIZED, output_dir=tmp / "out_warm")

            for name, cold_path in self._expected_artifacts(out_cold).items():
                warm_path = self._expected_artifacts(out_warm)[name]
                self.assertEqual(
                    cold_path.read_bytes(),
                    warm_path.read_bytes(),
                    msg=f"warm-path {name} differs byte-for-byte from cold-path",
                )


@unittest.skipUnless(_audio_spine_available(), "requires pip install -e '.[audio]' + fixtures")
class TestCopyStageArtifactsHelper(unittest.TestCase):
    """Unit tests for the ``Orchestrator._copy_stage_artifacts`` helper."""

    def _write(self, path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def test_copies_primary_and_existing_sidecars(self) -> None:
        from aimusic.audio.orchestrator import Orchestrator

        with tempfile.TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            cache = tmp / "cache"
            dest_dir = tmp / "dest"
            dest_dir.mkdir()
            self._write(cache / "stage.artifact", b"PRIMARY")
            self._write(cache / "stage.artifact.json", b"MANIFEST")
            self._write(cache / "stage.artifact.diff.json", b"DIFF")

            Orchestrator._copy_stage_artifacts(
                cache / "stage.artifact",
                dest_dir / "stage.artifact",
                ".json",
                ".diff.json",
            )

            self.assertEqual((dest_dir / "stage.artifact").read_bytes(), b"PRIMARY")
            self.assertEqual(
                (dest_dir / "stage.artifact.json").read_bytes(), b"MANIFEST"
            )
            self.assertEqual(
                (dest_dir / "stage.artifact.diff.json").read_bytes(), b"DIFF"
            )

    def test_skips_missing_sidecars(self) -> None:
        """Only sidecars that exist next to ``cached`` should be copied."""
        from aimusic.audio.orchestrator import Orchestrator

        with tempfile.TemporaryDirectory() as tmp_s:
            tmp = Path(tmp_s)
            cache = tmp / "cache"
            dest_dir = tmp / "dest"
            dest_dir.mkdir()
            self._write(cache / "stage.artifact", b"PRIMARY")
            # Only one of the two requested sidecars exists.
            self._write(cache / "stage.artifact.json", b"MANIFEST")

            Orchestrator._copy_stage_artifacts(
                cache / "stage.artifact",
                dest_dir / "stage.artifact",
                ".json",
                ".diff.json",
            )

            self.assertEqual((dest_dir / "stage.artifact").read_bytes(), b"PRIMARY")
            self.assertEqual(
                (dest_dir / "stage.artifact.json").read_bytes(), b"MANIFEST"
            )
            self.assertFalse((dest_dir / "stage.artifact.diff.json").exists())


if __name__ == "__main__":
    unittest.main()


class TestVerifiedCache(unittest.TestCase):
    def setUp(self):
        from tests.test_audio_stems import _make_two_pianos_one_bass_midi
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = _make_two_pianos_one_bass_midi(self.root / 'source.mid')
        self.config = {'stages': {'analysis': False, 'expressivization': False},
                       'render': {'backend': 'simple', 'allow_simple_fallback': True}}
        self.counter = 0

    def run_audio(self, config=None):
        from aimusic.audio.orchestrator import Orchestrator
        self.counter += 1
        orch = Orchestrator(config or self.config, self.root / 'cache')
        out = orch.run(self.source, output_dir=self.root / f'out{self.counter}')
        return orch, out

    def _use_fake_tuning_renderer(self):
        # Cache tests vary tuning metadata, not audio fidelity. The production
        # simple backend must no longer admit their nonstandard range request.
        from dataclasses import replace
        from unittest.mock import patch
        from aimusic.audio.render import registry
        simple = registry.resolve_renderer('bass', config={}).backend
        fake = replace(simple, name='tuning-test', capabilities=registry.RendererCapabilities(
            tuning_modes=('direct_12', 'channel_pitch')))
        provider = patch.dict(registry._BACKENDS, {'tuning-test': lambda asset: fake})
        provider.start()
        self.addCleanup(provider.stop)
        self.config['render']['backend'] = 'tuning-test'

    def test_resolves_available_renderer_again_before_warm_lookup(self):
        from dataclasses import replace
        from unittest.mock import patch
        from aimusic.audio.render import registry
        self.config['render']['backend'] = 'fluidsynth'
        self.run_audio()
        simple = registry.resolve_renderer('bass', config={}).backend
        available = replace(simple, name='fake-native', version='available-now')
        with patch.dict(registry._BACKENDS, {'fluidsynth': lambda asset: available}):
            orch, out = self.run_audio()
        self.assertEqual(orch.cache_misses.get('render'), 1)
        import json
        provenance = json.loads((out / 'render.json').read_text())
        self.assertEqual(provenance['renderers'][0]['effective_backend'], 'fake-native')
        self.assertIsNone(provenance['renderers'][0]['fallback_reason'])

    def test_each_effective_setting_and_soundfont_content_invalidates_cache(self):
        from copy import deepcopy
        baseline = deepcopy(self.config)
        self.run_audio()
        for change in ({'sample_rate': 16000}, {'peak_dbfs': -8}, {'release_tail_sec': 0.5},
                       {'allow_simple_fallback': False}, {'routing': {'bass': {'default': 'fluidsynth'}}}):
            with self.subTest(change=change):
                cfg = deepcopy(baseline)
                cfg['render'].update(change)
                orch, _ = self.run_audio(cfg)
                self.assertEqual(orch.cache_misses.get('render'), 1)
        asset = self.root / 'local.sf2'
        self.config['paths'] = {'soundfont': str(asset)}
        for data in (b'asset-one', b'asset-two'):
            asset.write_bytes(data)
            orch, _ = self.run_audio()
            self.assertEqual(orch.cache_misses.get('render'), 1)

    def test_backend_version_capability_code_native_versions_invalidate(self):
        from dataclasses import replace
        from unittest.mock import patch
        from aimusic.audio.render import registry
        simple = registry.resolve_renderer('bass', config={}).backend
        self.run_audio()
        for changes in ({'version': 'v-next'}, {'code_digest': 'changed'},
                        {'capabilities': replace(simple.capabilities, revision='next')},
                        {'native_versions': (('libsndfile', 'next'),)}):
            with self.subTest(changes=changes), patch.dict(registry._BACKENDS, {
                'simple': lambda asset: replace(simple, **changes)
            }):
                orch, _ = self.run_audio()
                self.assertEqual(orch.cache_misses.get('render'), 1)

    def test_corrupt_missing_and_incomplete_cache_are_misses(self):
        import json
        self.run_audio()
        for target in ('track_0001_piano.wav', 'track_0001_piano.wav.json', 'role_index.json', 'cache_manifest.json'):
            for corrupt in (False, True):
                with self.subTest(target=target, corrupt=corrupt):
                    cache = next(p for p in (self.root / 'cache' / 'render').iterdir() if p.is_dir())
                    path = cache / target
                    if corrupt:
                        path.write_bytes(b'CORRUPT')
                    else:
                        path.unlink()
                    orch, out = self.run_audio()
                    self.assertEqual(orch.cache_misses.get('render'), 1)
                    self.assertTrue((out / 'stems' / 'rough_mix.wav').is_file())
                    manifest = json.loads((cache / 'cache_manifest.json').read_text())
                    self.assertTrue(manifest['complete'])

    def test_warm_restore_is_byte_identical_including_all_provenance(self):
        _, cold = self.run_audio()
        warm_orch, warm = self.run_audio()
        snapshot = lambda out: {str(p.relative_to(out)): p.read_bytes() for p in out.rglob('*') if p.is_file()}
        self.assertEqual(warm_orch.cache_hits.get('render'), 1)
        self.assertEqual(snapshot(cold), snapshot(warm))
        self.assertTrue((warm / 'render.json').is_file())

    def test_output_and_asset_display_paths_do_not_invalidate(self):
        from copy import deepcopy
        asset = self.root / 'one.sf2'
        asset.write_bytes(b'same asset')
        self.config['paths'] = {'soundfont': str(asset)}
        self.run_audio()
        moved = self.root / 'two.sf2'
        moved.write_bytes(asset.read_bytes())
        cfg = deepcopy(self.config)
        cfg['paths'].update(soundfont=str(moved), output_dir='elsewhere', cache_dir='not-the-key')
        orch, _ = self.run_audio(cfg)
        self.assertEqual(orch.cache_hits.get('render'), 1)

    def test_interrupted_render_does_not_publish_complete_entry(self):
        from unittest.mock import patch
        from aimusic.audio.render import stems
        real_render = stems.render_stem
        calls = 0

        def fail_after_one(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError('interrupted')
            return real_render(*args, **kwargs)

        with patch.object(stems, 'render_stem', side_effect=fail_after_one), self.assertRaisesRegex(RuntimeError, 'interrupted'):
            self.run_audio()
        self.assertEqual(list((self.root / 'cache').rglob('cache_manifest.json')), [])
        self.assertEqual(list((self.root / 'out1').rglob('*.wav')), [])
        orch, _ = self.run_audio()
        self.assertEqual(orch.cache_misses.get('render'), 1)

    def test_source_expressive_bytes_and_semantic_metadata_change_keys(self):
        self._use_fake_tuning_renderer()
        import mido
        from copy import deepcopy
        from unittest.mock import patch
        from aimusic.audio.analysis.from_midi import analyze_midi
        self.config['stages']['analysis'] = True
        self.run_audio()
        # Metadata-only changes from an analysis producer, held source bytes fixed.
        # Remove only that stage's cache, leaving the render entry available.
        import shutil
        for extra in ({'tuning': {'pitch_bend_range': 3, 'method': 'direct_12'}},
                      {'structure_revision': 'changed'}, {'verified_identity': {'package_hash': 'other'}}):
            with self.subTest(extra=extra):
                shutil.rmtree(self.root / 'cache' / 'analysis')

                def analyze(*args, **kwargs):
                    import json
                    doc = analyze_midi(*args, **kwargs)
                    doc.update(extra)
                    kwargs['output_path'].write_text(json.dumps(doc))
                    return doc

                with patch('aimusic.audio.analysis.from_midi.analyze_midi', side_effect=analyze):
                    orch, _ = self.run_audio()
                self.assertEqual(orch.cache_misses.get('render'), 1)
        midi = mido.MidiFile(self.source)
        midi.tracks[1][2].velocity = 40
        midi.save(self.source)
        orch, _ = self.run_audio()
        self.assertEqual(orch.cache_misses.get('render'), 1)
        cfg = deepcopy(self.config)
        cfg['stages']['expressivization'] = True
        cfg['groove'] = {'spec': {'velocity': {'accent_map_16': [0.5] * 16}}}
        orch, _ = self.run_audio(cfg)
        self.assertEqual(orch.cache_misses.get('render'), 1)

    def test_upstream_corruption_is_never_served_as_warm_success(self):
        self.config['stages'].update(analysis=True, expressivization=True)
        self.run_audio()
        for stage, artifact in (('analysis', 'analysis.json'), ('analysis', 'analysis.json.json'),
                                ('expressivization', 'expressive.mid'), ('expressivization', 'expressive.mid.diff.json')):
            with self.subTest(stage=stage, artifact=artifact):
                cache = next((self.root / 'cache' / stage).iterdir())
                (cache / ('source.' + artifact)).write_bytes(b'corrupt')
                orch, _ = self.run_audio()
                self.assertEqual(orch.cache_misses.get(stage), 1)

    def test_legacy_and_canonical_routes_share_normalized_identity(self):
        self.config['render']['role_map'] = {'comp': 'simple'}
        self.run_audio()
        del self.config['render']['role_map']
        self.config['render']['routing'] = {'comping': {'default': 'simple', 'microtonal': 'simple'}}
        orch, _ = self.run_audio()
        self.assertEqual(orch.cache_hits.get('render'), 1)

    def test_verified_package_tuning_and_structure_invalidate_with_identical_midi(self):
        self._use_fake_tuning_renderer()
        from unittest.mock import patch
        from aimusic.audio.from_score import from_score
        from aimusic.audio.orchestrator import Orchestrator
        from aimusic.core.core_types import NoteEvent, Score
        from aimusic.core.diagnostics import RunManifest
        from aimusic.core.render_package import build_structure, write_render_package
        from aimusic.render import render_midi
        from aimusic.theory.edo import EDO
        from aimusic.core.config import EDOConfig
        score = Score(note_events=(NoteEvent(ton=0, toff=480, h=60, v=0.8, track='lead'),))
        render_midi(score, EDO(EDOConfig(n=12, base_tuning=0, pitch_bend_range=2)), str(self.source))
        source_bytes = self.source.read_bytes()
        for kind in ('baseline', 'tuning', 'structure'):
            with self.subTest(kind=kind):
                def structure(*args, **kwargs):
                    doc = build_structure(*args, **kwargs)
                    if kind == 'structure':
                        doc.tracks[0]['role'] = 'comping'
                    return doc
                with patch('aimusic.core.render_package.build_structure', side_effect=structure):
                    package = write_render_package(self.root / kind, score=score, midi_path=self.source,
                        manifest=RunManifest(seed=1, config_dump={}), pitch_bend_range=3 if kind == 'tuning' else 2)
                self.assertEqual(package.midi_path.read_bytes(), source_bytes)
                orch = Orchestrator(self.config, self.root / 'cache')
                out = orch.run(from_score(package.root), output_dir=self.root / f'package-{kind}')
                self.assertEqual(orch.cache_misses.get('render'), 1)
                import json
                doc = json.loads((out / 'render.json').read_text())
                self.assertEqual(doc['source']['package_hash'], package.content_hash)
                if kind == 'structure':
                    self.assertEqual(doc['renderers'][0]['role'], 'comping')

    def test_run_report_does_not_overwrite_user_file_or_symlink_target(self):
        for kind in ('file', 'symlink'):
            with self.subTest(kind=kind):
                out = self.root / f'out{self.counter + 1}'
                out.mkdir()
                report = out / 'render.json'
                outside = self.root / f'outside-{kind}'
                outside.write_bytes(b'KEEP OUTSIDE')
                if kind == 'file':
                    report.write_bytes(b'KEEP')
                else:
                    try:
                        report.symlink_to(outside)
                    except OSError as exc:
                        if getattr(exc, 'winerror', None) == 1314:
                            self.skipTest('Native symlink privilege unavailable; ordinary report collision remains mandatory')
                        raise
                with self.assertRaises(FileExistsError):
                    self.run_audio()
                self.assertEqual(outside.read_bytes(), b'KEEP OUTSIDE')
                if kind == 'file':
                    self.assertEqual(report.read_bytes(), b'KEEP')

    def test_process_death_leaves_only_unselectable_scratch_not_a_complete_cache(self):
        import subprocess
        import sys
        code = '''
import os
from pathlib import Path
from aimusic.audio.cache import cached_stage

def die(directory):
    (directory / 'audio.wav').write_bytes(b'partial')
    os._exit(23)

cached_stage(Path(__import__('sys').argv[1]), {'request': 1}, ['audio.wav'], die)
'''
        root = self.root / 'killed-cache'
        proc = subprocess.run([sys.executable, '-c', code, str(root)], capture_output=True)
        self.assertEqual(proc.returncode, 23, proc.stderr)
        self.assertEqual(list(root.rglob('cache_manifest.json')), [])
        from aimusic.audio.cache import cached_stage
        entry, hit = cached_stage(root, {'request': 1}, ['audio.wav'],
                                  lambda p: (p / 'audio.wav').write_bytes(b'complete'))
        self.assertFalse(hit)
        self.assertEqual((entry / 'audio.wav').read_bytes(), b'complete')

    def test_restore_rejects_another_valid_entry_substituted_during_copy(self):
        import shutil
        from unittest.mock import patch
        from aimusic.audio.cache import cached_stage
        from aimusic.audio.render.stems import copy_stem_artifacts
        first, _ = cached_stage(self.root / 'race-cache', {'request': 1}, ['audio.wav'],
                                lambda p: (p / 'audio.wav').write_bytes(b'first'))
        other, _ = cached_stage(self.root / 'race-cache', {'request': 2}, ['audio.wav'],
                                lambda p: (p / 'audio.wav').write_bytes(b'other'))
        real_copy = shutil.copyfile

        def substitute(src, dst):
            return real_copy(other / Path(src).name, dst)

        with patch('aimusic.audio.render.stems.shutil.copyfile', side_effect=substitute):
            with self.assertRaisesRegex(ValueError, 'cache changed'):
                copy_stem_artifacts(first, self.root / 'race-out', verify_cache=True)
        self.assertEqual(list((self.root / 'race-out').iterdir()), [])
