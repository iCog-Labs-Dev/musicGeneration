"""Task 6.6 CLI contracts: explicit trust, bounded records, no ambient discovery."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from aimusic.audio.review_bundle import canonical_bytes
from tests.test_audio_review_archive import ArchiveCase


class TestReviewArchiveCLI(ArchiveCase):
    def setUp(self):
        super().setUp()
        self.bundle_path = self.root / 'bundle.json'
        self.assignment_path = self.root / 'reviewer.json'
        self.bundle_path.write_bytes(self.bundle.to_bytes())
        self.assignment_path.write_bytes(self.assignment.to_bytes())
        self.archive = self.base / 'review.zip'
        self.view = self.base / 'review view café'

    def cli(self, *args):
        return subprocess.run([sys.executable, '-m', 'aimusic.audio.review_archive', *map(str, args)],
                              capture_output=True, timeout=30)

    def assembly_args(self):
        return ['assemble', '--input-root', str(self.root), '--bundle', str(self.bundle_path),
                '--assignment', str(self.assignment_path), '--expected-bundle-sha256', self.bundle.sha256,
                '--expected-assignment-sha256', self.assignment.sha256, '--out', str(self.archive)]

    def result(self, result, operation, sha):
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        self.assertEqual(result.stderr, b'')
        self.assertEqual(result.stdout, canonical_bytes({
            'schema': 'aimusic.review-archive-result/1', 'operation': operation, 'archive_sha256': sha}))

    def refused(self, result):
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertEqual(result.stdout, b'')
        self.assertNotIn(b'Traceback', result.stderr)
        self.assertTrue(result.stderr)

    def test_explicit_module_commands_round_trip_canonical_results(self):
        digest = self.assemble().sha256
        self.result(self.cli(*self.assembly_args()), 'assemble', digest)
        self.assertEqual(hashlib.sha256(self.archive.read_bytes()).hexdigest(), digest)
        common = ['--archive', self.archive, '--expected-sha256', digest]
        self.result(self.cli('verify', *common), 'verify', digest)
        self.result(self.cli('extract', *common, '--out', self.view), 'extract', digest)
        self.result(self.cli('verify-extracted', *common, '--directory', self.view), 'verify-extracted', digest)
        self.refused(self.cli(*self.assembly_args()))
        self.refused(self.cli('extract', *common, '--out', self.view))
        (self.view / 'index.html').write_bytes(b'changed')
        self.refused(self.cli('verify-extracted', *common, '--directory', self.view))
        self.assertEqual(self.archive.read_bytes(), self.assemble().data)

    def test_missing_explicit_paths_digests_unknown_options_and_commands_refused(self):
        self.refused(self.cli())
        self.refused(self.cli('serve'))
        for args in (self.assembly_args(),
                     ['verify', '--archive', str(self.archive), '--expected-sha256', self.bundle.sha256],
                     ['extract', '--archive', str(self.archive), '--expected-sha256', self.bundle.sha256,
                      '--out', str(self.view)],
                     ['verify-extracted', '--archive', str(self.archive), '--expected-sha256', self.bundle.sha256,
                      '--directory', str(self.view)]):
            for index in range(1, len(args), 2):
                with self.subTest(command=args[0], omitted=args[index]):
                    self.refused(self.cli(*args[:index], *args[index + 2:]))
            self.refused(self.cli(*args, '--unexpected'))
        args = self.assembly_args()
        args[args.index('--expected-bundle-sha256')] = '--expected-bundle'
        self.refused(self.cli(*args))
        self.assertFalse(self.archive.exists())
        self.assertFalse(self.view.exists())

    def test_record_digest_and_canonical_json_required_before_publication(self):
        for path, original, flag in ((self.bundle_path, self.bundle.to_bytes(), '--expected-bundle-sha256'),
                                    (self.assignment_path, self.assignment.to_bytes(), '--expected-assignment-sha256')):
            for data in (original + b'\n', b'\xef\xbb\xbf' + original,
                         b'{"schema":1,"schema":2}', b'[]', b'\xff', b'[' * 1100,
                         canonical_bytes({**json.loads(original), 'unknown': True})):
                with self.subTest(path=path.name, sample=data[:20]):
                    path.write_bytes(data)
                    args = self.assembly_args()
                    args[args.index(flag) + 1] = hashlib.sha256(data).hexdigest()
                    self.refused(self.cli(*args))
                    self.assertFalse(self.archive.exists())
            path.write_bytes(original)
        for flag in ('--expected-bundle-sha256', '--expected-assignment-sha256'):
            for bad in ('', 'A' * 64, '0' * 64):
                args = self.assembly_args()
                args[args.index(flag) + 1] = bad
                self.refused(self.cli(*args))
        self.assertFalse(self.archive.exists())

    def test_input_records_over_budget_or_nonregular_are_refused(self):
        for path, limit, original in ((self.bundle_path, 32 * 1024 * 1024, self.bundle.to_bytes()),
                                      (self.assignment_path, 64 * 1024, self.assignment.to_bytes())):
            with path.open('wb') as stream:
                stream.truncate(limit + 1)
            self.refused(self.cli(*self.assembly_args()))
            path.unlink()
            path.mkdir()
            self.refused(self.cli(*self.assembly_args()))
            path.rmdir()
            if hasattr(os, 'mkfifo'):
                os.mkfifo(path)
                self.refused(self.cli(*self.assembly_args()))
                path.unlink()
            path.write_bytes(original)
        self.assertFalse(self.archive.exists())

    def test_reparse_records_refused_and_budget_checked_before_record_parse(self):
        from types import SimpleNamespace
        real = Path.lstat

        def reparse(path, *args, **kwargs):
            info = real(path, *args, **kwargs)
            return SimpleNamespace(st_mode=info.st_mode, st_file_attributes=0x400) if path == self.root else info

        with patch.object(Path, 'lstat', reparse), patch('sys.stderr'):
            self.assertNotEqual(self.module.main(self.assembly_args()), 0)
        with self.assignment_path.open('wb') as stream:
            stream.truncate(64 * 1024 + 1)
        with patch.object(self.module.ReviewAssignment, 'from_bytes', side_effect=AssertionError('unbounded parse')):
            with patch('sys.stderr'):
                self.assertNotEqual(self.module.main(self.assembly_args()), 0)
        self.assertFalse(self.archive.exists())

    def test_archive_trust_never_inferred_from_received_bytes_or_completion(self):
        sealed = self.assemble()
        self.archive.write_bytes(sealed.data)
        for digest in ('', 'A' * 64, '0' * 64):
            for command, tail in (('verify', []), ('extract', ['--out', self.view]),
                                  ('verify-extracted', ['--directory', self.view])):
                self.refused(self.cli(command, '--archive', self.archive, '--expected-sha256', digest, *tail))
        self.assertFalse(self.view.exists())
        self.module.extract_archive(self.archive, self.view, expected_sha256=sealed.sha256)
        (self.view / 'extra').write_text('even beside the exact completion marker')
        self.refused(self.cli('verify-extracted', '--archive', self.archive, '--directory', self.view,
                             '--expected-sha256', sealed.sha256))

    def test_import_and_help_are_stdlib_only_without_renderer_asset_or_network_imports(self):
        program = '''
import importlib.abc
import runpy
import sys
sys.path.insert(0, sys.argv[1])
class Deny(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'mido', 'yaml', 'soundfile', 'pretty_midi',
                                     'fluidsynth', 'torch', 'socket', 'http', 'urllib'}:
            raise AssertionError('unexpected dependency: ' + fullname)
sys.meta_path.insert(0, Deny())
sys.argv = ['review_archive', *sys.argv[2:]]
runpy.run_module('aimusic.audio.review_archive', run_name='__main__')
'''
        for command in (['--help'], ['prepare', '--help'], ['verify', '--help'], ['extract', '--help']):
            result = subprocess.run([sys.executable, '-I', '-S', '-c', program,
                                     str(Path(self.module.__file__).resolve().parents[2]), *command],
                                    capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr.decode())
            self.assertIn(b'usage:', result.stdout)
