"""Asset-free tests of the explicitly requested native lane's setup contract."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tests.audio_native_support import load_native_manifest


class TestNativeContract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.asset = self.root / 'fake.sf2'
        self.asset.write_bytes(b'asset-free contract test, not a SoundFont')
        self.manifest = self.root / 'manifest.json'
        self.data = dict(soundfont_path=str(self.asset), soundfont_sha256=hashlib.sha256(self.asset.read_bytes()).hexdigest(),
                         soundfont_name='test asset', soundfont_version='1', source_url='https://example.org/asset',
                         source_commit='a' * 40, license_url='https://example.org/license', license_summary='test permission',
                         license_approved=True, native_version='2.6.2', binding_version='1.3.4')

    def load(self):
        self.manifest.write_text(json.dumps(self.data))
        with patch.dict(os.environ, {'AIMUSIC_FLUIDSYNTH_MANIFEST': str(self.manifest)}):
            return load_native_manifest()

    def test_every_provenance_field_required_and_bytes_verified(self):
        self.assertEqual(self.load(), self.data)
        for key in tuple(self.data):
            with self.subTest(missing=key):
                value = self.data.pop(key)
                with self.assertRaisesRegex(AssertionError, key):
                    self.load()
                self.data[key] = value
        for key, value in [('license_approved', False), ('soundfont_sha256', '0' * 64),
                           ('source_commit', ''), ('license_summary', ' '),
                           ('soundfont_path', str(self.root / 'missing.sf2'))]:
            with self.subTest(invalid=key):
                old, self.data[key] = self.data[key], value
                with self.assertRaisesRegex(AssertionError, key):
                    self.load()
                self.data[key] = old
        self.asset.write_bytes(b'same path replaced')
        with self.assertRaisesRegex(AssertionError, 'soundfont_sha256'):
            self.load()

    def test_binding_native_missing_or_wrong_version_is_failure_not_skip(self):
        from tests.audio_native_support import verify_native_runtime
        from aimusic.audio.render import fluidsynth_r
        for error in (ImportError('binding missing'), OSError('native library missing')):
            with self.subTest(error=error), patch.object(fluidsynth_r, 'load_binding', side_effect=error):
                with self.assertRaisesRegex(AssertionError, 'native prerequisites'):
                    verify_native_runtime(self.data)
        with patch.object(fluidsynth_r, 'load_binding'), patch.object(fluidsynth_r, 'versions',
                return_value=(('libfluidsynth', '2.5.0'), ('pyfluidsynth', '1.3.4'))):
            with self.assertRaisesRegex(AssertionError, 'native_version'):
                verify_native_runtime(self.data)
        with patch.object(fluidsynth_r, 'load_binding'), patch.object(fluidsynth_r, 'versions',
                return_value=(('libfluidsynth', '2.6.2'), ('pyfluidsynth', 'other'))):
            with self.assertRaisesRegex(AssertionError, 'binding_version'):
                verify_native_runtime(self.data)

    def test_default_skips_but_explicit_missing_manifest_fails_process(self):
        env = dict(os.environ)
        for key in ('AIMUSIC_RUN_FLUIDSYNTH_TESTS', 'AIMUSIC_FLUIDSYNTH_MANIFEST', 'AIMUSIC_SOUNDFONT'):
            env.pop(key, None)
        cmd = [sys.executable, '-m', 'unittest', 'tests.test_audio_fluidsynth_integration']
        default = subprocess.run(cmd, env=env, text=True, capture_output=True)
        self.assertEqual(default.returncode, 0, default.stderr)
        self.assertIn('skipped=', default.stderr)
        env['AIMUSIC_RUN_FLUIDSYNTH_TESTS'] = '1'
        explicit = subprocess.run(cmd, env=env, text=True, capture_output=True)
        self.assertNotEqual(explicit.returncode, 0)
        self.assertIn('AIMUSIC_FLUIDSYNTH_MANIFEST', explicit.stderr)
        self.assertNotIn('skipped=', explicit.stderr)
