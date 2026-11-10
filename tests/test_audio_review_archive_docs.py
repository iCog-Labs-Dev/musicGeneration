"""Runbook wheel tests: separate stdlib archive and provisioned synthetic audio lanes.

POSIX shell blocks execute on POSIX; PowerShell blocks execute natively where
available, otherwise their exact Python argv/scripts are exercised (not claimed
as native PowerShell/Windows evidence). Browser launching is not automated.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from urllib.request import urlopen

from tests.test_audio_review_archive import fixture

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / 'docs/audio-review-bundles.md'


def block(name):
    text = DOC.read_text(encoding='utf-8')
    match = re.search(r'<!-- archive-' + name + r' -->\n```[^\n]+\n(.*?)\n```', text, re.S)
    if match is None:
        raise AssertionError('missing documented command block: ' + name)
    return match.group(1)


def child_environment(inherited=None):
    excluded = {'PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'PYTHONUTF8', 'PYTHONIOENCODING'}
    env = {k: v for k, v in (os.environ if inherited is None else inherited).items() if k not in excluded}
    return {**env, 'PYTHONUTF8': '1'}


def install_wheel(base, run, *, audio=False):
    source, wheels, envroot = (base / name for name in ('source', 'wheels', 'venv'))
    if (ROOT / 'aimusic').is_dir():
        source.mkdir()
        shutil.copytree(ROOT / 'aimusic', source / 'aimusic', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        for name in ('pyproject.toml', 'readme.md'):
            shutil.copy2(ROOT / name, source / name)
        run(sys.executable, '-I', '-X', 'utf8', '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation',
            '--no-index', '--wheel-dir', wheels, source)
        built = list(wheels.glob('aimusic-*.whl'))
        shutil.rmtree(source)
    else:
        # Native runner copies the exact wheel and docs into external checks.
        built = list((ROOT / 'wheels').glob('aimusic-*.whl'))
    if len(built) != 1:
        raise AssertionError('expected one explicit test wheel')
    run(sys.executable, '-I', '-X', 'utf8', '-m', 'venv', '--without-pip', envroot)
    python = envroot / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    site = Path(run(python, '-I', '-X', 'utf8', '-c', "import sysconfig; print(sysconfig.get_paths()['purelib'])").strip())
    run(sys.executable, '-I', '-X', 'utf8', '-m', 'pip', 'install', '--no-deps', '--no-index', '--target', site, built[0])
    if audio:
        # Explicit offline test provisioning: reuse installed audio dependencies,
        # not their .pth files/editable checkout. The product itself is the wheel.
        dependencies = [p for p in sys.path if Path(p).name in ('site-packages', 'dist-packages')]
        (site / 'provisioned-audio.pth').write_text('import sys; sys.path.extend(' + ascii(dependencies) + ')\n',
                                                    encoding='ascii')
    return python, site


class TestArchiveRunbook(unittest.TestCase):
    def test_generation_blocks_are_explicit_and_utf8_children_are_not_inherited(self):
        env = child_environment({'PYTHONUTF8': '0', 'PYTHONIOENCODING': 'ascii', 'PYTHONPATH': 'bad'})
        self.assertEqual(env['PYTHONUTF8'], '1')
        self.assertNotIn('PYTHONIOENCODING', env)
        self.assertNotIn('PYTHONPATH', env)
        for shell in ('posix', 'powershell'):
            command = block(shell + '-generate')
            self.assertIn('-m aimusic.audio.review_matrix --package', command)
            self.assertIn('--soundfont', command)
            self.assertIn('--expected-matrix-sha256', command)
            self.assertIn('--reviewer', command)
            self.assertLess(command.index('review_matrix'), command.index('review_archive prepare'))
            self.assertLess(command.index('review_archive prepare'), command.index('review_archive assemble'))


    def test_custody_and_separate_serving_are_explicit(self):
        text = DOC.read_text(encoding='utf-8')
        for phrase in ('exactly 14 exposures', 'independent channel', 'whole provenance graph',
                       'no BOM or trailing newline', 'fresh name', 'not a human finding',
                       'private/blind-map.json', 'payload mutation', 'serving-time lock'):
            self.assertIn(phrase, text)
        posix = block('posix-serve').splitlines()
        self.assertIn('verify-extracted', posix[0])
        self.assertTrue(posix[0].endswith('&&'))
        self.assertIn('-m http.server 8000 --bind 127.0.0.1 --directory "$VIEW"', posix[1])
        for phase in ('generate', 'assemble', 'extract', 'serve'):
            lines = block('powershell-' + phase).splitlines()
            for index, line in enumerate(lines):
                if '& $Python ' in line:
                    self.assertIn("if ($LASTEXITCODE -ne 0) { throw '", lines[index + 1])
        ps = block('powershell-serve').splitlines()
        self.assertIn('verify-extracted', ps[0])
        self.assertIn('-m http.server 8000 --bind 127.0.0.1 --directory $View', ps[2])
        self.assertIn("-m webbrowser -t 'http://127.0.0.1:8000/index.html'", block('posix-open'))
        self.assertEqual(block('powershell-open'), "Start-Process 'http://127.0.0.1:8000/index.html'")

    def test_installed_wheel_runbook_commands_without_network_or_asset_discovery(self):
        with tempfile.TemporaryDirectory(prefix='archive runbook café ') as tmp:
            base = Path(tmp).resolve()
            env = {**child_environment(), 'TMPDIR': str(base), 'TEMP': str(base), 'TMP': str(base)}

            def run(*args, input=None, guarded=True):
                completed = subprocess.run(list(map(str, args)), cwd=base,
                                           env={**env, 'ARCHIVE_TEST_GUARD': '1' if guarded else '0'},
                                           input=input, text=True, encoding='utf-8', capture_output=True, timeout=120)
                self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
                return completed.stdout

            python, site = install_wheel(base, run)
            # No optional/product dependencies are installed. Forbid access to
            # network, assets, renderers and process launch during archive work.
            (site / 'sitecustomize.py').write_text('''
import importlib.abc
import os
import sys
if os.environ.get('ARCHIVE_TEST_GUARD') == '1':
    class Guard(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if (fullname.split('.')[0] in {'numpy', 'mido', 'yaml', 'soundfile', 'pretty_midi',
                                          'fluidsynth', 'torch', 'socket', 'http', 'urllib'}
                    or fullname in {'aimusic.audio.review_matrix', 'aimusic.audio.render.registry',
                                    'aimusic.audio.orchestrator'}):
                raise AssertionError('forbidden archive import: ' + fullname)
    sys.meta_path.insert(0, Guard())
    def audit(event, args):
        if event.startswith(('socket.', 'subprocess.', 'os.system', 'glob.glob', 'pathlib.Path.glob',
                             'pathlib.Path.rglob')):
            raise AssertionError('forbidden archive side effect: ' + event)
        if event == 'open' and str(args[0]).lower().endswith(('.sf2', '.sf3')):
            raise AssertionError('asset access')
    sys.addaudithook(audit)
''', encoding='utf-8')
            imported = run(python, '-c', 'import sitecustomize,aimusic; assert hasattr(sitecustomize, "Guard"); print(aimusic.__file__)').strip()
            self.assertTrue(Path(imported).is_relative_to(site))
            self.assertIn('verify-extracted', run(python, '-m', 'aimusic.audio.review_archive', '--help'))
            self.assertIn('--matrix', run(python, '-m', 'aimusic.audio.review_archive', 'prepare', '--help'))
            missing_audio = subprocess.run([str(python), '-m', 'aimusic.audio.review_archive', 'prepare',
                                            '--input-root', str(base), '--matrix', str(base / 'matrix.json'),
                                            '--expected-matrix-sha256', '0' * 64, '--reviewer', 'r01'],
                                           cwd=base, env={**env, 'ARCHIVE_TEST_GUARD': '0'},
                                           capture_output=True, timeout=30)
            self.assertEqual(missing_audio.returncode, 1)
            self.assertEqual(missing_audio.stdout, b'')
            self.assertIn('fresh private root', json.loads(missing_audio.stderr)['error'])
            private = base / 'private inputs'
            private.mkdir()
            bundle, _, assignment = fixture(private)
            (private / 'bundle.json').write_bytes(bundle.to_bytes())
            (private / 'reviewer.json').write_bytes(assignment.to_bytes())
            asset = private / 'never-discover.sf2'
            asset.write_bytes(b'not an asset; must not be read')
            env['AIMUSIC_SOUNDFONT'] = str(asset)
            env['AIMUSIC_AUDIO_BACKEND'] = 'm2a'  # Archive operations ignore renderer selection.
            for shell in ('posix', 'powershell'):
                out = base / shell
                out.mkdir()
                archive, view = out / 'review.zip', out / 'view'
                ledger, receipt = out / 'trusted.sha256', out / 'received.sha256'
                values = {'Python': str(python), 'InputRoot': str(private), 'Bundle': str(private / 'bundle.json'),
                          'Assignment': str(private / 'reviewer.json'), 'BundleSHA256': bundle.sha256,
                          'AssignmentSHA256': assignment.sha256, 'Archive': str(archive),
                          'View': str(view), 'DigestRecord': str(ledger)}

                def substituted(phase):
                    command = block(shell + '-' + phase)
                    examples = {
                        '/absolute/venv/bin/python': 'Python', '/absolute/private-inputs': 'InputRoot',
                        '/absolute/private-inputs/bundle.json': 'Bundle',
                        '/absolute/private-inputs/reviewer.json': 'Assignment',
                        '/absolute/delivery/review-r01.zip': 'Archive', '/absolute/received/review-r01.zip': 'Archive',
                        '/absolute/reviewer-views/review-r01-new': 'View',
                        '/absolute/trusted-ledger/review-r01.sha256': 'DigestRecord',
                        '/absolute/trusted-ledger/received-r01.sha256': 'DigestRecord',
                        'REPLACE_WITH_APPROVED_BUNDLE_SHA256': 'BundleSHA256',
                        'REPLACE_WITH_APPROVED_ASSIGNMENT_SHA256': 'AssignmentSHA256',
                        'REPLACE_WITH_INDEPENDENTLY_CONFIRMED_ARCHIVE_SHA256': 'TrustedSHA256',
                    }
                    for example, key in examples.items():
                        if key in values:
                            command = command.replace("'" + example + "'", "'" + values[key] + "'")
                    if shell == 'powershell':
                        for key, value in values.items():
                            command = re.sub(r'(?m)^\$' + key + r" = '[^']*'$", lambda m: '$' + key + " = '" + value + "'",
                                             command)
                    return command

                def execute_phase(phase):
                    command = substituted(phase)
                    if shell == 'posix' and os.name != 'nt':
                        return run('sh', '-c', command)
                    powershell = shutil.which('pwsh') or shutil.which('powershell')
                    if shell == 'powershell' and powershell:
                        script = out / (phase + '.ps1')
                        # BOM keeps Unicode paths correct in Windows PowerShell 5.1.
                        script.write_text(command, encoding='utf-8-sig')
                        return run(powershell, '-NoProfile', '-NonInteractive', '-File', script)
                    # Exercise exactly the documented Python argv and inline
                    # recording code when this shell is unavailable. No emulated
                    # shell is claimed as native evidence.
                    output = ''
                    for line in command.splitlines():
                        if '& $Python ' not in line:
                            continue
                        args = shlex.split(line.split('& $Python ', 1)[1])
                        args = [values[a[1:]] if a.startswith('$') else a for a in args]
                        output = run(python, *args)
                        if line.startswith('$Result = '):
                            values['ArchiveSHA256'] = json.loads(output)['archive_sha256']
                    return output

                # On Windows the POSIX contract has already been checked by the
                # POSIX lane; the native PowerShell path still executes below.
                if shell == 'posix' and os.name == 'nt':
                    continue
                execute_phase('assemble')
                digest = hashlib.sha256(archive.read_bytes()).hexdigest()
                self.assertEqual(ledger.read_bytes(), digest.encode('ascii') + b'\n')
                values.update(TrustedSHA256=digest, DigestRecord=str(receipt))
                execute_phase('extract')
                self.assertEqual(receipt.read_bytes(), ledger.read_bytes())
                run(python, '-m', 'aimusic.audio.review_archive', 'verify-extracted', '--archive', archive,
                    '--expected-sha256', digest, '--directory', view)
                self.assertFalse((view / 'private').exists())
                self.assertEqual(len(list((view / 'media').iterdir())), 14)
                # Execute the documented verify-then-serve sequence separately
                # from the archive-only guard. Use an ephemeral port for isolation.
                if shell == 'posix' and os.name != 'nt':
                    with socket.socket() as probe:
                        probe.bind(('127.0.0.1', 0))
                        port = probe.getsockname()[1]
                    prelude = '\n'.join(k + '=' + shlex.quote(v) for k, v in {
                        'PYTHON': str(python), 'ARCHIVE': str(archive), 'VIEW': str(view),
                        'TRUSTED_SHA256': digest}.items())
                    serving = prelude + '\n' + block('posix-serve').replace('8000', str(port))
                    # exec affects process ownership only, so cleanup stops the server too.
                    serving = serving.replace('"$PYTHON" -m http.server', 'exec "$PYTHON" -m http.server')
                    process = subprocess.Popen(['sh', '-c', serving], cwd=base,
                                               env={**env, 'ARCHIVE_TEST_GUARD': '0'},
                                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                    try:
                        for attempt in range(100):
                            try:
                                with urlopen(f'http://127.0.0.1:{port}/index.html', timeout=1) as response:
                                    self.assertEqual(response.read(), (view / 'index.html').read_bytes())
                                break
                            except OSError:
                                if process.poll() is not None or attempt == 99:
                                    self.fail('documented local server did not start')
                                time.sleep(.02)
                    finally:
                        process.terminate()
                        process.communicate(timeout=10)


class TestGenerationRunbook(unittest.TestCase):
    def test_installed_wheel_matrix_prepare_assemble_with_synthetic_renderer(self):
        with tempfile.TemporaryDirectory(prefix='generation runbook café ') as tmp:
            base = Path(tmp).resolve()
            env = {**child_environment(), 'TMPDIR': str(base), 'TEMP': str(base), 'TMP': str(base)}

            def run(*args):
                result = subprocess.run(list(map(str, args)), cwd=base, env=env, capture_output=True,
                                        text=True, encoding='utf-8', timeout=180)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return result.stdout

            python, site = install_wheel(base, run, audio=True)
            checks = base / 'checks'
            shutil.copytree(ROOT / 'tests', checks / 'tests', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            # The only rendering substitution lives in the test environment, never
            # the product CLI. Execute the documented -m entrypoint/argv unchanged.
            (site / 'sitecustomize.py').write_text(
                "import sys\nsys.path.insert(0, " + ascii(str(checks)) + ")\n" +
                "if 'aimusic.audio.review_matrix' in sys.orig_argv:\n"
                "    from tests.test_audio_review_matrix import BACKEND\n"
                "    from aimusic.audio.render import registry\n"
                "    registry._BACKENDS['fluidsynth'] = lambda asset: BACKEND\n"
                "sys.path.pop(0)\n"
                # Warm the stdlib runtime identity probe (uname subprocess on macOS).
                "import platform; platform.platform()\n"
                "def audit(event, args):\n"
                "    if event.startswith(('socket.', 'subprocess.', 'os.system')):\n"
                "        raise AssertionError('network/process discovery forbidden')\n"
                "    if event == 'open' and str(args[0]).lower().endswith(('.sf2', '.sf3')):\n"
                "        if 'aimusic.audio.review_matrix' not in sys.orig_argv:\n"
                "            raise AssertionError('preparation/archive asset access')\n"
                "sys.addaudithook(audit)\n", encoding='ascii')
            imported = run(python, '-c', 'import aimusic,numpy,soundfile; print(aimusic.__file__)').strip()
            self.assertTrue(Path(imported).is_relative_to(site))
            run(python, '-m', 'aimusic.app.cli', 'generate', '--seed', '11', '--beats', '4', '--out', base / 'retained')
            packages = list((base / 'retained').glob('run_*'))
            self.assertEqual(len(packages), 1)
            package = packages[0]
            asset = base / 'synthetic-only.sf2'
            asset.write_bytes(b'not a real SoundFont; synthetic renderer only')
            for shell in ('posix', 'powershell'):
                if shell == 'posix' and os.name == 'nt':
                    continue  # POSIX execution belongs to its native lane.
                out = base / shell
                out.mkdir()
                root, archive = out / 'private', out / 'review.zip'
                values = {'Python': str(python), 'Package': str(package), 'SoundFont': str(asset),
                          'InputRoot': str(root), 'Matrix': str(root / 'matrix.json'),
                          'MatrixRecord': str(out / 'matrix.sha256'), 'Reviewer': 'synthetic-r01',
                          'Archive': str(archive), 'DigestRecord': str(out / 'archive.sha256')}
                command = block(shell + '-generate')
                if shell == 'posix':
                    keys = {'Python': 'PYTHON', 'Package': 'PACKAGE', 'SoundFont': 'SOUNDFONT',
                            'InputRoot': 'INPUT_ROOT', 'Matrix': 'MATRIX', 'MatrixRecord': 'MATRIX_RECORD',
                            'Reviewer': 'REVIEWER', 'Archive': 'ARCHIVE', 'DigestRecord': 'DIGEST_RECORD'}
                    for key, value in values.items():
                        command = re.sub(r'(?m)^' + keys[key] + r"='[^']*'$",
                                         lambda m: keys[key] + '=' + shlex.quote(value), command)
                    run('sh', '-c', command)
                else:
                    for key, value in values.items():
                        command = re.sub(r'(?m)^\$' + key + r" = '[^']*'$",
                                         lambda m: '$' + key + " = '" + value.replace("'", "''") + "'", command)
                    powershell = shutil.which('pwsh') or shutil.which('powershell')
                    if powershell:
                        script = out / 'generate.ps1'
                        script.write_text(command, encoding='utf-8-sig')
                        run(powershell, '-NoProfile', '-NonInteractive', '-File', script)
                    else:
                        # Exact Python argv/code only, NOT native PowerShell evidence.
                        for line in command.splitlines():
                            if '& $Python ' not in line:
                                continue
                            args = shlex.split(line.split('& $Python ', 1)[1])
                            args = [values[a[1:]] if a.startswith('$') else a for a in args]
                            output = run(python, *args)
                            if line.startswith('$MatrixSHA256 = '):
                                values['MatrixSHA256'] = output.strip()
                            elif line.startswith('$Prepared = '):
                                prepared = json.loads(output)
                                values.update(Bundle=prepared['bundle'], BundleSHA256=prepared['bundle_sha256'],
                                              Assignment=prepared['assignment'], AssignmentSHA256=prepared['assignment_sha256'])
                            elif line.startswith('$Result = '):
                                values['ArchiveSHA256'] = json.loads(output)['archive_sha256']
                self.assertEqual(Path(values['MatrixRecord']).read_text().strip(),
                                 hashlib.sha256((root / 'matrix.json').read_bytes()).hexdigest())
                digest = hashlib.sha256(archive.read_bytes()).hexdigest()
                self.assertEqual(Path(values['DigestRecord']).read_bytes(), digest.encode('ascii') + b'\n')
                prepared = json.loads((root / 'prepared.json').read_bytes())
                bundle = json.loads(Path(prepared['bundle']).read_bytes())
                self.assertEqual(len(bundle['presentations']), 14)
                self.assertTrue(all(p['gain'] == 1 and p['start_sample'] == 0 for p in bundle['presentations']))
                view = out / 'view'
                run(python, '-m', 'aimusic.audio.review_archive', 'extract', '--archive', archive,
                    '--expected-sha256', digest, '--out', view)
                run(python, '-m', 'aimusic.audio.review_archive', 'verify-extracted', '--archive', archive,
                    '--expected-sha256', digest, '--directory', view)
                self.assertEqual(len(list((view / 'media').iterdir())), 14)
                self.assertFalse((view / 'private').exists())
