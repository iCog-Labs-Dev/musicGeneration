"""Asset-free checks of the executable native reproduction documentation."""
from __future__ import annotations

import ast
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


class TestNativeReproductionDocs(unittest.TestCase):
    def setUp(self):
        text = (Path(__file__).resolve().parents[1] / 'docs/audio-testing.md').read_text()
        self.section = text.split('### Reproduce and locate the listening set', 1)[1].split(
            'Under the retained directory, listen to:', 1)[0]
        self.blocks = re.findall(r'```bash\n(.*?)\n```', self.section, re.S)
        self.shell = shutil.which('sh')
        if not self.shell:
            self.skipTest('POSIX shell snippets require sh (use WSL on Windows)')

    def test_snippets_are_portable_and_have_valid_shell_and_python_syntax(self):
        self.assertNotRegex(self.section, r'/(?:private/var/folders|Users|home)/')
        self.assertGreaterEqual(len(self.blocks), 2, 'include supplemental provenance commands')
        for block in self.blocks:
            result = subprocess.run([self.shell, '-n'], input=block, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            for python in re.findall(r"<<'PY'\n(.*?)\nPY", block, re.S):
                ast.parse(python)
        self.assertIn('runtime.json', self.blocks[1])
        self.assertIn('pip-freeze.txt', self.blocks[1])

    def test_reproduction_requires_caller_paths_and_preserves_spaces(self):
        block = self.blocks[0]
        # Check guards before executing so a regressed author-path recipe is never run.
        for key in ('SCRATCH', 'PYTHON', 'AIMUSIC_FLUIDSYNTH_MANIFEST'):
            self.assertIn(f': "${{{key}:?', block)
        with tempfile.TemporaryDirectory(prefix='audio docs ') as tmp:
            root = Path(tmp)
            python = root / 'chosen python'
            python.write_text('#!/bin/sh\nprintf "%s\\n" "$AIMUSIC_FLUIDSYNTH_ARTIFACTS" '
                              '"$AIMUSIC_FLUIDSYNTH_MANIFEST" "$AIMUSIC_RUN_FLUIDSYNTH_TESTS" "$@"\n')
            python.chmod(0o700)
            manifest = root / 'approved manifest.json'
            manifest.write_text('{}')  # Shell test only; no native renderer or asset.
            env = dict(os.environ, SCRATCH=str(root), PYTHON=str(python),
                       AIMUSIC_FLUIDSYNTH_MANIFEST=str(manifest))
            run = lambda e: subprocess.run([self.shell], input=block, env=e, cwd=root,
                                          capture_output=True, text=True)
            result = run(env)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines(), [str(root), str(manifest), '1', '-m', 'pytest',
                             'tests/test_audio_fluidsynth_integration.py', '-q', '-s'])
            for key in ('SCRATCH', 'PYTHON', 'AIMUSIC_FLUIDSYNTH_MANIFEST'):
                for value in (None, '', 'relative/path', str(root / 'missing')):
                    with self.subTest(key=key, value=value):
                        invalid = dict(env)
                        invalid.pop(key) if value is None else invalid.update({key: value})
                        result = run(invalid)
                        self.assertNotEqual(result.returncode, 0)
                        self.assertEqual(result.stdout, '', 'must not invoke Python with invalid paths')

    def test_imported_preview_evidence_uses_portable_scratch_path(self):
        text = (Path(__file__).resolve().parents[1] / 'docs/audio-imported-midi.md').read_text()
        evidence = text.split('## Recorded local preview check', 1)[1]
        self.assertNotRegex(evidence, r'/(?:private/var/folders|Users|home)/')
        self.assertIn('$SCRATCH/', evidence)
