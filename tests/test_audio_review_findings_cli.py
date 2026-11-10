"""Synthetic CLI collection: no private map required, no actual listening."""
from dataclasses import replace
import json
import subprocess
import sys

from tests.test_audio_review_findings import FindingCase, completed


class TestFindingCLI(FindingCase):
    def setUp(self):
        super().setUp()
        self.view = self.base / 'review view café'
        sealed = self.assemble()
        self.module.extract_archive(sealed.data, self.view, expected_sha256=sealed.sha256)
        self.history = self.base / 'history'
        self.history.mkdir()
        self.common = ['--assignment', self.view / 'assignment.json', '--media-root', self.view,
                       '--expected-assignment-sha256', self.assignment.sha256]

    def cli(self, *args):
        return subprocess.run([sys.executable, '-m', 'aimusic.audio.review_findings', *map(str, args)],
                              capture_output=True, timeout=30)

    def test_template_validate_append_in_reviewer_only_tree(self):
        first = self.template()
        before = {p.relative_to(self.view): p.read_bytes() for p in self.view.rglob('*') if p.is_file()}
        result = self.cli('template', *self.common, '--history', self.history)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['state'], 'incomplete')
        full = completed(first)
        draft = self.base / 'submitted.json'
        draft.write_bytes(full.to_bytes())
        args = [*self.common, '--finding', draft, '--expected-finding-sha256', full.sha256]
        result = self.cli('validate', *args)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['state'], 'complete')
        result = self.cli('append', *args, '--history', self.history)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(list(self.history.iterdir())), 2)
        self.assertEqual(before, {p.relative_to(self.view): p.read_bytes() for p in self.view.rglob('*') if p.is_file()})
        draft.write_bytes(replace(full, reviewer='synthetic-altered').to_bytes())
        result = self.cli('append', *args, '--history', self.history)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stdout, b'')
        self.assertNotIn(b'Traceback', result.stderr)

    def test_missing_trust_abbreviations_noncanonical_and_changed_media_refused(self):
        for args in (['template'], ['template', *self.common[:-2], '--history', self.history],
                     ['template', *self.common, '--hist', self.history]):
            result = self.cli(*args)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, b'')
        assignment = self.view / 'assignment.json'
        assignment.write_bytes(self.assignment.to_bytes() + b'\n')
        result = self.cli('template', *self.common, '--history', self.history)
        self.assertNotEqual(result.returncode, 0)
        assignment.write_bytes(self.assignment.to_bytes())
        (self.view / self.assignment.stimuli[0].media.path).write_bytes(b'changed')
        result = self.cli('template', *self.common, '--history', self.history)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(list(self.history.iterdir()), [])
