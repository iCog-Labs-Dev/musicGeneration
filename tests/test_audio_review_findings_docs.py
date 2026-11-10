"""Execute external CLI runbooks from an offline installed wheel, synthetic only."""
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
import unittest

from tests.test_audio_review_archive import fixture
from tests.test_audio_review_archive_docs import child_environment, install_wheel
from tests.test_audio_review_findings import completed, supplementary_notes
from aimusic.audio.review_findings import create_template

DOC = Path(__file__).resolve().parents[1] / 'docs/audio-review-findings.md'


def block(name):
    text = DOC.read_text(encoding='utf-8')
    found = re.search(r'<!-- findings-' + name + r' -->\n```[^\n]+\n(.*?)\n```', text, re.S)
    if found is None:
        raise AssertionError('missing runbook block: ' + name)
    return found[1]


class TestFindingRunbook(unittest.TestCase):
    def test_review_instructions_and_legacy_link_distinction(self):
        text = DOC.read_text(encoding='utf-8')
        for phrase in ('low safe listening volume', 'fatigue', 'fixed volume', 'playback failure',
                       'Never fabricate findings', 'linkability', 'not authentication', 'not tamper-proof',
                       'AssignmentReference', 'assignment_document_sha256', 'incomplete', 'withdrawn',
                       'no BOM or trailing newline', 'other', '8 singles', '3 pairs'):
            self.assertIn(phrase, text)
        for name in ('template', 'canonicalize', 'validate', 'append', 'close', 'generate'):
            lines = block('powershell-' + name).splitlines()
            self.assertTrue(lines[0].startswith('& $Python '))
            self.assertIn('if ($LASTEXITCODE -ne 0)', lines[1])

    def test_offline_installed_wheel_runbook(self):
        with tempfile.TemporaryDirectory(prefix='findings wheel café ') as tmp:
            base = Path(tmp).resolve()
            env = {**child_environment(), 'TMPDIR': str(base), 'TMP': str(base), 'TEMP': str(base)}

            def run(*args):
                result = subprocess.run(list(map(str, args)), cwd=base, env=env, capture_output=True,
                                        text=True, encoding='utf-8', timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return result.stdout

            python, site = install_wheel(base, run)
            (site / 'sitecustomize.py').write_text('''
import importlib.abc
import sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'numpy', 'mido', 'yaml', 'soundfile', 'pretty_midi', 'fluidsynth', 'torch', 'socket', 'http', 'urllib'}:
            raise AssertionError('not stdlib/offline: ' + fullname)
sys.meta_path.insert(0, Guard())
def audit(event, args):
    if event.startswith(('socket.', 'subprocess.', 'os.system', 'glob.glob', 'pathlib.Path.glob', 'pathlib.Path.rglob')):
        raise AssertionError('unexpected discovery/network/process: ' + event)
sys.addaudithook(audit)
''', encoding='utf-8')
            run(python, '-c', 'import sitecustomize,aimusic; assert hasattr(sitecustomize,"Guard"); print(aimusic.__file__)')
            root = base / 'private inputs'
            root.mkdir()
            bundle, _, assignment = fixture(root)
            (root / 'bundle.json').write_bytes(bundle.to_bytes())
            assigned = root / assignment.reference(bundle).document.path
            assigned.parent.mkdir()
            assigned.write_bytes(assignment.to_bytes())
            template = create_template(assignment, root=root, expected_assignment_sha256=assignment.sha256)
            full = completed(template)
            draft = base / 'authored.json'
            draft.write_text(json.dumps(full.to_dict(), indent=2), encoding='utf-8')
            for shell in ('posix', 'powershell'):
                out = base / shell
                out.mkdir()
                histories = out / 'histories'
                histories.mkdir()
                history = histories / assignment.sha256
                history.mkdir()
                reports = out / 'reports'
                reports.mkdir()
                values = dict(Python=str(python), Assignment=str(assigned), AssignmentSHA256=assignment.sha256,
                              View=str(root), History=str(history), Draft=str(draft), Finding=str(out / 'submitted.json'), FindingSHA256=full.sha256,
                              InputRoot=str(root), Bundle=str(root / 'bundle.json'), BundleSHA256=bundle.sha256,
                              Reviewer=assignment.reviewer, Reports=str(reports), FindingsRoot=str(histories))

                def command(name, *, expected_code=0):
                    code = block(shell + '-' + name)
                    if shell == 'posix' and os.name != 'nt':
                        result = subprocess.run(['/bin/sh', '-eu', '-c', code], cwd=base, env={**env, **values},
                                                capture_output=True, text=True, encoding='utf-8', timeout=120)
                    elif shell == 'powershell' and (shutil.which('pwsh') or shutil.which('powershell')):
                        program = '\n'.join('$' + k + "='" + v.replace("'", "''") + "'" for k, v in values.items()) + '\n' + code
                        result = subprocess.run([shutil.which('pwsh') or shutil.which('powershell'), '-NoProfile', '-Command', program],
                                                cwd=base, env=env, capture_output=True, text=True, encoding='utf-8', timeout=120)
                    else:
                        # Exact documented argv when PowerShell is unavailable; not native Windows evidence.
                        tokens = shlex.split(code.splitlines()[0])
                        if tokens[0] == '&':
                            tokens = tokens[1:]
                        args = [values.get(a[1:], a) if a.startswith('$') else a for a in tokens]
                        result = subprocess.run(args, cwd=base, env=env, capture_output=True,
                                                text=True, encoding='utf-8', timeout=120)
                    self.assertEqual(result.returncode, expected_code, result.stdout + result.stderr)
                    if expected_code:
                        self.assertEqual(result.stdout, '')
                        self.assertIn('document byte budget', result.stderr)
                        return None
                    return result.stdout.strip() if name == 'canonicalize' else json.loads(result.stdout)

                self.assertEqual(command('template')['state'], 'incomplete')
                self.assertEqual(command('canonicalize'), full.sha256)
                self.assertEqual(command('validate')['state'], 'complete')
                self.assertEqual(command('append')['state'], 'complete')
                closure = command('close')
                values.update(Closure=closure['path'], ClosureSHA256=closure['closure_sha256'])
                report = command('generate')
                doc = json.loads(Path(report['path']).read_bytes())
                self.assertEqual(doc['eligible_counts']['reviewers'], 1)
                self.assertNotIn('order_seed', json.dumps(doc))

                # Draft UTF-8 fits, but 1024 supplementary characters per note
                # expand past 128 KiB in canonical ASCII-escaped JSON.
                oversized = supplementary_notes(full)
                bad_draft = out / 'oversized-draft.json'
                raw = json.dumps(oversized.to_dict(), ensure_ascii=False).encode('utf-8')
                self.assertLess(len(raw), 128 * 1024)
                bad_draft.write_bytes(raw)
                bad_output = out / 'must-not-exist.json'
                values.update(Draft=str(bad_draft), Finding=str(bad_output))
                before = {p.relative_to(out): p.read_bytes() for p in out.rglob('*') if p.is_file()}
                command('canonicalize', expected_code=1)
                self.assertFalse(bad_output.exists())
                self.assertEqual(before, {p.relative_to(out): p.read_bytes() for p in out.rglob('*') if p.is_file()})
