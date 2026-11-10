"""Static CI contracts and asset-free evidence/probe regressions."""
from __future__ import annotations

import ctypes
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import yaml

ROOT = Path(__file__).resolve().parents[1]


class TestWindowsCIContract(unittest.TestCase):
    def test_windows_job_is_explicitly_deferred_without_weakening_its_contract(self):
        workflow = yaml.safe_load((ROOT / '.github/workflows/ci.yml').read_text(encoding='utf-8'))
        job = workflow['jobs']['windows-audio']
        self.assertEqual(job['runs-on'], 'windows-latest')
        self.assertEqual(job['if'], '${{ false }}')
        self.assertNotIn('continue-on-error', job)
        self.assertEqual(job['name'], 'Windows audio (asset-free wheel)')
        commands = '\n'.join(step.get('run', '') for step in job['steps'])
        self.assertIn('scripts/test-windows-audio-as-standard-user.ps1', commands)
        self.assertNotIn('-Renderer', commands)
        self.assertNotIn('secrets.', json.dumps(job))
        self.assertEqual(job['defaults']['run']['shell'], 'pwsh')

    def test_manual_evidence_lane_is_default_dry_and_never_uploads_assets(self):
        workflow = yaml.safe_load((ROOT / '.github/workflows/windows-audio-evidence.yml').read_text(encoding='utf-8'))
        trigger = workflow.get('on', workflow.get(True))  # PyYAML YAML 1.1 bool spelling.
        self.assertEqual(set(trigger), {'workflow_dispatch'})
        self.assertIs(trigger['workflow_dispatch']['inputs']['run_renderer']['default'], False)
        render = workflow['jobs']['renderer']
        self.assertIn('inputs.run_renderer', render['if'])
        self.assertIn('self-hosted', render['runs-on'])
        self.assertEqual(render['environment'], 'windows-audio-evidence')
        self.assertEqual(workflow['permissions'], {'contents': 'read'})
        self.assertNotIn('upload-artifact', json.dumps(workflow))
        for job in workflow['jobs'].values():
            for step in job['steps']:
                self.assertNotIn('${{ inputs.', step.get('run', ''))

    def test_preflight_runs_with_stdlib_no_asset_and_no_native_import(self):
        env = {key: value for key, value in os.environ.items() if not key.startswith('AIMUSIC_')}
        result = subprocess.run([sys.executable, '-S', '-m', 'tests.windows_fluidsynth_evidence', '--preflight'],
                                cwd=ROOT, env=env, text=True, encoding='utf-8', capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report['mode'], 'preflight')
        self.assertIs(report['renderer_executed'], False)
        self.assertEqual(report['pins']['libfluidsynth'], '2.6.2')
        self.assertEqual(report['pins']['pyfluidsynth'], '1.3.4')
        self.assertIn('approved_manifest', report['required_inputs'])

    def test_enabled_missing_prerequisites_fail_not_skip(self):
        result = subprocess.run([sys.executable, '-S', '-m', 'tests.windows_fluidsynth_evidence', '--run'],
                                cwd=ROOT, text=True, encoding='utf-8', capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('--manifest', result.stderr)
        self.assertNotIn('skipped', result.stderr.lower())

    def test_only_known_probe_unavailability_is_classified(self):
        module = importlib.import_module('tests.windows_capabilities')
        for error in (5, 50, 1314):
            failure = OSError('capability denied')
            failure.winerror = error
            with patch.object(module, 'exercise', side_effect=failure):
                result = module.probe('junction', Path('unused'))
            self.assertFalse(result.available)
            self.assertIn(str(error), result.reason)
        with patch.object(module, 'exercise', side_effect=OSError('unexpected failure')):
            with self.assertRaisesRegex(OSError, 'unexpected failure'):
                module.probe('junction', Path('unused'))
        with patch.object(module, 'exercise'):
            self.assertTrue(module.probe('junction', Path('unused')).available)

    def test_required_runner_includes_portable_archive_contracts_without_posix_skips(self):
        from tests import windows_ci

        def cases(suite):
            for item in suite:
                if isinstance(item, unittest.TestSuite):
                    yield from cases(item)
                else:
                    yield item

        required = list(cases(windows_ci.load_required_suite()))
        ids = {test.id() for test in required}
        loader = unittest.TestLoader()
        expected = list(cases(loader.loadTestsFromNames([
            'tests.test_audio_review_archive',
            'tests.test_audio_review_archive_cli',
            'tests.test_audio_review_prepare',
            'tests.test_audio_review_archive_docs',
            'tests.test_audio_review_findings',
            'tests.test_audio_review_findings_cli',
            'tests.test_audio_review_findings_docs',
            'tests.test_audio_review_report',
            'tests.test_audio_review_archive_security.TestVerifiedExtraction',
            'tests.test_audio_review_archive_security.TestBackendCopyIntegrity',
            'tests.test_audio_review_archive_security.TestWindowsPublicReadback',
        ])))
        self.assertFalse(loader.errors)
        self.assertTrue(expected)
        self.assertTrue({test.id() for test in expected} <= ids)
        self.assertEqual(len(ids), len(required))
        self.assertFalse(any('_FailedTest' in name or 'TestPosix' in name or
                             'TestUnsupportedConcurrentMutation' in name for name in ids))
        for test in expected:
            self.assertFalse(getattr(type(test), '__unittest_skip__', False))
            self.assertFalse(getattr(getattr(test, test._testMethodName), '__unittest_skip__', False))
            self.assertFalse(windows_ci.allowed_skip(test.id(), 'missing capability'))

    def test_installed_runner_rejects_broad_skips(self):
        module = importlib.import_module('tests.windows_ci')
        self.assertFalse(module.allowed_skip('tests.test_architecture.TestArchitectureGuardrails.test_audio_only_imports_core_and_theory_from_aimusic', 'missing'))
        self.assertFalse(module.allowed_skip('tests.test_audio_windows_native.TestNativeWindowsPublication.test_create_only_and_replacement_complete_without_residue', 'capability unavailable'))
        prefix = 'tests.test_audio_windows_native.TestNativeWindowsPublication.'
        for name, kind in module.OPTIONAL_CASES.items():
            for code in (5, 50, 1314):
                with self.subTest(name=name, code=code):
                    self.assertTrue(module.allowed_skip(prefix + name,
                        f'CAPABILITY_UNAVAILABLE {kind}: WinError {code}; substitutes mandatory'))
            for tail in ('', '5', '51; substitutes mandatory', '1314garbage',
                         '1314; substitutes mandatory\n', '01314; substitutes mandatory',
                         '-5; substitutes mandatory', '50; substitutes mandatory; ignored',
                         '5; substitutes optional', '999; substitutes mandatory'):
                with self.subTest(name=name, tail=tail):
                    self.assertFalse(module.allowed_skip(prefix + name,
                        f'CAPABILITY_UNAVAILABLE {kind}: WinError {tail}'))
            other = 'symlink' if kind == 'junction' else 'junction'
            self.assertFalse(module.allowed_skip(prefix + name,
                f'CAPABILITY_UNAVAILABLE {other}: WinError 5; substitutes mandatory'))
        self.assertFalse(module.allowed_skip(prefix + 'not_a_test',
            'CAPABILITY_UNAVAILABLE symlink: WinError 1314; substitutes mandatory'))


class TestWindowsEvidenceRuntime(unittest.TestCase):
    def test_dll_query_uses_function_address_and_rejects_truncation(self):
        from tests.windows_fluidsynth_evidence import loaded_library
        callback = ctypes.CFUNCTYPE(ctypes.c_int)(lambda: 0)
        kernel = SimpleNamespace(GetModuleHandleExW=Mock(return_value=1), GetModuleFileNameW=Mock())
        def filename(handle, buffer, size):
            buffer.value = 'approved.dll'
            return len(buffer.value)
        kernel.GetModuleFileNameW.side_effect = filename
        self.assertEqual(loaded_library(callback, kernel), Path('approved.dll'))
        self.assertEqual(kernel.GetModuleHandleExW.call_args.args[0], 6)
        self.assertEqual(kernel.GetModuleHandleExW.call_args.args[1].value,
                         ctypes.cast(callback, ctypes.c_void_p).value)
        kernel.GetModuleFileNameW.side_effect = None
        for size in (0, 32768):
            kernel.GetModuleFileNameW.return_value = size
            with self.subTest(length=size), self.assertRaisesRegex(OSError, 'truncated'):
                loaded_library(callback, kernel)
        kernel.GetModuleHandleExW.return_value = 0
        with self.assertRaisesRegex(OSError, 'GetModuleHandleExW'):
            loaded_library(callback, kernel)

    @unittest.skipUnless(os.name == 'nt', 'native DLL path query; simulated ABI above always runs')
    def test_native_dll_query_resolves_an_actual_loaded_module(self):
        from tests.windows_fluidsynth_evidence import loaded_library
        kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        self.assertTrue(loaded_library(kernel.GetCurrentProcessId).is_file())

    def test_asset_free_script_is_external_wheel_not_editable(self):
        script = (ROOT / 'scripts/test-windows-audio.ps1').read_text(encoding='utf-8-sig')
        self.assertIn("'wheel', '--no-deps', '--no-build-isolation', '--no-index'", script)
        self.assertIn("'audio-render,dev' } else { 'audio,dev'", script)
        self.assertIn("'-m', 'venv'", script)
        self.assertIn("'tests.windows_ci'", script)
        self.assertIn("'docs'", script)
        self.assertIn("Join-Path $Checks 'wheels'", script)
        self.assertIn('[char]0x00e9', script)
        self.assertIn("'aimusic wheel caf'", script)
        self.assertNotIn("'--editable'", script)
        self.assertNotIn('Set-ExecutionPolicy', script)
        self.assertNotIn('RunAs', script)
        self.assertIn('$LASTEXITCODE -ne 0', script)
