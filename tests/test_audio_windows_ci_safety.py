"""Token and PowerShell environment regressions; never create accounts in tests."""
from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]


class TestWindowsToken(unittest.TestCase):
    def test_only_unelevated_non_admin_medium_or_lower_token_is_accepted(self):
        from tests.windows_token import require_standard_user
        for elevated, admin, integrity in ((False, False, 8192), (False, False, 4096)):
            status = dict(elevated=elevated, administrator=admin, integrity_rid=integrity)
            with patch('tests.windows_token.token_status', return_value=status):
                self.assertEqual(require_standard_user(), status)
        for elevated, admin, integrity in ((True, False, 8192), (False, True, 8192),
                                           (False, False, 12288), (True, True, 16384)):
            status = dict(elevated=elevated, administrator=admin, integrity_rid=integrity)
            with self.subTest(status=status), patch('tests.windows_token.token_status', return_value=status):
                with self.assertRaisesRegex(RuntimeError, 'standard-user'):
                    require_standard_user()

    def test_optimized_python_cannot_disable_token_gate(self):
        from tests.windows_token import require_standard_user
        with patch('tests.windows_token.sys.flags', SimpleNamespace(optimize=1)):
            with self.assertRaisesRegex(RuntimeError, 'Assertions must be enabled'):
                require_standard_user()

    def test_token_query_failure_is_not_treated_as_standard_user(self):
        from tests.windows_token import require_standard_user
        with patch('tests.windows_token.token_status', side_effect=OSError('query failed')):
            with self.assertRaisesRegex(OSError, 'query failed'):
                require_standard_user()

    @unittest.skipUnless(os.name == 'nt', 'Windows token ABI')
    def test_native_token_query_matches_windows_admin_membership(self):
        import ctypes
        from tests.windows_token import token_status
        status = token_status()
        self.assertIs(status['administrator'], bool(ctypes.windll.shell32.IsUserAnAdmin()))
        self.assertIsInstance(status['elevated'], bool)
        self.assertGreater(status['integrity_rid'], 0)


class TestAssetFreeEnvironment(unittest.TestCase):
    def test_runner_rejects_empty_and_nonempty_asset_variables(self):
        from tests.windows_ci import verify_asset_free_environment, ASSET_VARIABLES
        self.assertEqual(set(ASSET_VARIABLES), {'AIMUSIC_SOUNDFONT', 'AIMUSIC_RUN_FLUIDSYNTH_TESTS',
                                               'AIMUSIC_FLUIDSYNTH_MANIFEST', 'AIMUSIC_FLUIDSYNTH_ARTIFACTS'})
        with patch.dict(os.environ, {}, clear=True):
            verify_asset_free_environment()
            for name in ASSET_VARIABLES:
                for value in ('', 'unexpected-asset'):
                    with self.subTest(name=name, value=value), patch.dict(os.environ, {name: value}):
                        with self.assertRaisesRegex(RuntimeError, name):
                            verify_asset_free_environment()

    @unittest.skipUnless(os.name == 'nt', 'PowerShell process environment')
    def test_powershell_removes_asset_variables_and_restores_original_state(self):
        # EncodedCommand carries code, paths/values are env data, never source text.
        code = r'''$ErrorActionPreference = 'Stop'
. (Join-Path $env:AIMUSIC_TEST_REPO 'scripts/windows-audio-environment.ps1')
$State = Enter-AudioTestEnvironment
'import os,json; print(json.dumps({k:v for k,v in os.environ.items() if k.startswith("AIMUSIC_") and not k.startswith("AIMUSIC_TEST_")}))' | & $env:AIMUSIC_TEST_PYTHON -
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Exit-AudioTestEnvironment $State
'import os,json; print(json.dumps({k:v for k,v in os.environ.items() if k.startswith("AIMUSIC_") and not k.startswith("AIMUSIC_TEST_")}))' | & $env:AIMUSIC_TEST_PYTHON -
exit $LASTEXITCODE
'''
        encoded = base64.b64encode(code.encode('utf-16-le')).decode('ascii')
        from tests.windows_ci import ASSET_VARIABLES
        self.assertIsNotNone(shutil.which('powershell.exe'))
        for shell in ('powershell.exe', 'pwsh.exe'):
            executable = shutil.which(shell)
            if executable is None:
                continue  # PS5 guaranteed on native Windows; additionally exercise pwsh when installed.
            for value in (None, '', 'do-not-use.sf2'):
                env = {k: v for k, v in os.environ.items() if not k.startswith('AIMUSIC_')}
                if value is not None:
                    env.update({name: value for name in ASSET_VARIABLES})
                env.update(AIMUSIC_TEST_REPO=str(ROOT), AIMUSIC_TEST_PYTHON=sys.executable)
                with self.subTest(shell=shell, initial=value):
                    result = subprocess.run([executable, '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-EncodedCommand', encoded],
                                            env=env, text=True, encoding='utf-8', errors='replace', capture_output=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    during, after = map(json.loads, result.stdout.splitlines())
                    self.assertEqual(during, {'AIMUSIC_AUDIO_BACKEND': 'native'})
                    self.assertEqual(after, {} if value is None else {name: value for name in ASSET_VARIABLES})


class TestStandardUserBootstrap(unittest.TestCase):
    def test_bootstrap_uses_credentials_not_password_argv_or_environment(self):
        script = (ROOT / 'scripts/test-windows-audio-as-standard-user.ps1').read_text(encoding='utf-8')
        self.assertIn('New-LocalUser', script)
        self.assertIn('-Credential $Credential', script)
        self.assertIn('-EncodedCommand', script)
        self.assertIn('Remove-LocalUser', script)
        self.assertIn('finally', script)
        self.assertNotIn('ConvertFrom-SecureString', script)
        self.assertNotIn('ConvertTo-SecureString', script)
        self.assertNotIn('net user', script.lower())
        self.assertNotIn('Write-Host $Password', script)
        self.assertNotIn('Set-ExecutionPolicy', script)
        self.assertIn("'ReadAndExecute'", script)
        self.assertIn("'Modify'", script)

    def test_entrypoint_verifies_token_before_build_and_runner_rechecks(self):
        script = (ROOT / 'scripts/test-windows-audio.ps1').read_text(encoding='utf-8-sig')
        self.assertLess(script.index('windows_token.py'), script.index("'-m', 'venv'"))
        runner = (ROOT / 'tests/windows_ci.py').read_text()
        self.assertLess(runner.index('token = require_standard_user()'), runner.index('verify_install(args.'))

    @unittest.skipUnless(os.name == 'nt', 'PowerShell command transport')
    def test_launch_payload_quotes_metacharacters_and_propagates_failure(self):
        # Exercise the actual encoded launch command, without provisioning users.
        code = r'''$ErrorActionPreference = 'Stop'
. (Join-Path $env:AIMUSIC_TEST_REPO 'scripts/windows-audio-standard-user-command.ps1')
$Encoded = New-AudioStandardUserCommand -Script $env:AIMUSIC_TEST_SCRIPT -Python $env:AIMUSIC_TEST_VALUE -Scratch $env:AIMUSIC_TEST_SCRATCH -Git $env:AIMUSIC_TEST_VALUE
$Shell = (Get-Process -Id $PID).Path
& $Shell -NoProfile -NonInteractive -ExecutionPolicy Bypass -EncodedCommand $Encoded
exit $LASTEXITCODE
'''
        encoded = base64.b64encode(code.encode('utf-16-le')).decode('ascii')
        with tempfile.TemporaryDirectory(prefix="audio ci caf\u00e9 \u97f3 ';&$[] ") as tmp:
            script = Path(tmp) / "check ';&$[].ps1"
            value = "C:/fake ' ; $(throw 'injection') /python.exe"
            env = dict(os.environ, AIMUSIC_TEST_REPO=str(ROOT), AIMUSIC_TEST_SCRIPT=str(script),
                       AIMUSIC_TEST_VALUE=value, AIMUSIC_TEST_SCRATCH=tmp)
            for body, success in (("@{ python = $Python; scratch = $ScratchParent; temp = $env:TEMP } | ConvertTo-Json -Compress", True),
                                  ("throw 'expected failure'", False), ("exit 7", False)):
                script.write_text('param([string]$Python, [string]$ScratchParent)\n' + body, encoding='utf-8-sig')
                for shell in ('powershell.exe', 'pwsh.exe'):
                    if shutil.which(shell) is None:
                        continue
                    with self.subTest(shell=shell, body=body):
                        result = subprocess.run([shell, '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
                                                 '-EncodedCommand', encoded], env=env, capture_output=True,
                                                text=True, encoding='utf-8', errors='replace', timeout=30)
                        if success:
                            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                            self.assertEqual(json.loads(result.stdout), dict(python=value, scratch=tmp, temp=tmp))
                        else:
                            self.assertNotEqual(result.returncode, 0)
