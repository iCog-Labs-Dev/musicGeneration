"""Pinned Windows evidence wrapper. Default preflight is stdlib-only and asset-free.

Run via scripts/test-windows-audio.ps1 -Renderer for isolated installed-wheel
execution. Generated evidence and absolute machine paths must stay outside Git.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
from typing import Any
import unittest

PINS = {'libfluidsynth': '2.6.2', 'pyfluidsynth': '1.3.4', 'soundfile': '0.14.0', 'libsndfile': '1.2.2'}


def digest(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path: Path, value: Any) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def loaded_library(function: Any, kernel: Any = None) -> Path:
    """Identify the DLL containing a public function, not a same-named CLI binary.

    Win32 GetModuleHandleExW(FROM_ADDRESS|UNCHANGED_REFCOUNT), GetModuleFileNameW.
    The binding retains its library throughout this single-threaded query; do not
    FreeLibrary the borrowed HMODULE. See docs/audio-fluidsynth.md API references.
    """
    kernel = kernel if kernel is not None else getattr(ctypes, 'WinDLL')('kernel32', use_last_error=True)
    handle = ctypes.c_void_p()
    get_handle, get_name = kernel.GetModuleHandleExW, kernel.GetModuleFileNameW
    get_handle.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    get_handle.restype = ctypes.c_int32
    get_name.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint32]
    get_name.restype = ctypes.c_uint32
    if not get_handle(6, ctypes.cast(function, ctypes.c_void_p), ctypes.byref(handle)):
        raise OSError('GetModuleHandleExW could not identify the loaded FluidSynth DLL')
    buffer = ctypes.create_unicode_buffer(32768)
    length = get_name(handle, buffer, len(buffer))
    if not 0 < length < len(buffer):
        raise OSError('GetModuleFileNameW failed or truncated the loaded DLL path')
    return Path(buffer.value)


def worker() -> int:
    from tests.test_audio_fluidsynth_integration import TestFluidSynthIntegration
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TestFluidSynthIntegration)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {'tests_run': result.testsRun, 'skips': len(result.skipped),
              'passed': result.wasSuccessful() and not result.skipped and result.testsRun == 5,
              'evidence_root': str(getattr(TestFluidSynthIntegration, 'root', ''))}
    write_json(Path(os.environ['AIMUSIC_FLUIDSYNTH_ARTIFACTS']) / 'worker-results.json', report)
    return 0 if report['passed'] else 1


def run(args: argparse.Namespace) -> int:
    # Check explicit inputs before importing any optional dependency.
    for name in ('manifest', 'dll', 'dll_sha256', 'out', 'checkout'):
        if not getattr(args, name):
            raise ValueError(f'Enabled renderer requires --{name.replace("_", "-")} (see docs/audio-testing.md)')
    if sys.flags.optimize:
        raise ValueError('Evidence assertions require Python without -O / PYTHONOPTIMIZE')
    if os.name != 'nt' or sys.version_info[:2] != (3, 11) or platform.machine().lower() not in ('amd64', 'x86_64'):
        raise ValueError('Pinned evidence requires native Windows x64 with Python 3.11')
    for name in ('manifest', 'dll'):
        path = Path(getattr(args, name))
        if not path.is_absolute() or not path.is_file():
            raise ValueError(f'--{name} must be an existing absolute local file')
    parent, checkout = Path(args.out), Path(args.checkout).resolve()
    if not parent.is_absolute() or not parent.is_dir() or parent.resolve().is_relative_to(checkout):
        raise ValueError('--out must be an existing absolute external scratch directory, not the checkout')
    dll = Path(args.dll)
    if not re.fullmatch('[a-f0-9]{64}', args.dll_sha256) or digest(dll) != args.dll_sha256:
        raise ValueError('--dll-sha256 must match the explicitly approved DLL bytes')
    from tests.windows_ci import verify_install
    verify_install(checkout)
    os.environ['AIMUSIC_FLUIDSYNTH_MANIFEST'] = args.manifest
    from tests.audio_native_support import load_native_manifest, verify_native_runtime
    asset = load_native_manifest()
    for field, pin in (('native_version', 'libfluidsynth'), ('binding_version', 'pyfluidsynth')):
        if asset[field] != PINS[pin]:
            raise ValueError(f'Approved manifest {field} must pin {PINS[pin]}')
    runtime = verify_native_runtime(asset)
    if runtime != PINS:
        raise ValueError(f'Pinned runtime mismatch: expected={PINS}, actual={runtime}')
    from aimusic.audio.render.fluidsynth_r import load_binding
    binding = load_binding()
    actual_dll = loaded_library(binding.fluid_version)
    if not actual_dll.samefile(dll) or digest(actual_dll) != args.dll_sha256:
        raise ValueError('Loaded FluidSynth DLL is not the explicitly approved DLL')
    root = Path(tempfile.mkdtemp(prefix='windows-fluidsynth-', dir=parent))
    os.environ['AIMUSIC_RUN_FLUIDSYNTH_TESTS'] = '1'
    os.environ['AIMUSIC_FLUIDSYNTH_ARTIFACTS'] = str(root)
    os.environ['AIMUSIC_AUDIO_BACKEND'] = 'native'
    os.environ['PYTHONUTF8'] = '1'
    command = [sys.executable, '-m', 'tests.windows_fluidsynth_evidence', '--worker']
    info: dict[str, Any] = {'schema': 'aimusic.windows-renderer-evidence/1', 'os_build': platform.version(),
            'windows_version': tuple(getattr(sys, 'getwindowsversion')()), 'platform': platform.platform(),
            'architecture': platform.machine(), 'pointer_bits': ctypes.sizeof(ctypes.c_void_p) * 8,
            'python': sys.version, 'python_executable': sys.executable, 'runtime': runtime,
            'asset': asset, 'approved_manifest_sha256': digest(Path(args.manifest)),
            'loaded_dll': str(actual_dll), 'loaded_dll_sha256': digest(actual_dll),
            'binding_sha256': digest(Path(binding.__file__)),
            'commands': [command, [sys.executable, '-m', 'pip', 'freeze']],
            'same_machine_repetition': 'Each canonical workflow: independent cold, cold-repeat, warm-cache',
            'cross_platform_pcm_comparison': False, 'human_listening': 'pending'}
    write_json(root / 'runtime.json', info)
    freeze = subprocess.run(info['commands'][1], capture_output=True, text=True, encoding='utf-8', check=True)
    (root / 'pip-freeze.txt').write_text(freeze.stdout, encoding='utf-8')
    print(f'External evidence: {root}', flush=True)
    process = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', timeout=600)
    (root / 'native-tests.log').write_text(process.stdout + process.stderr, encoding='utf-8')
    print(process.stdout + process.stderr)
    passed = process.returncode == 0
    summary: dict[str, Any] = {'passed': passed, 'returncode': process.returncode}
    if passed:
        result = json.loads((root / 'worker-results.json').read_text(encoding='utf-8'))
        evidence = Path(result['evidence_root']) / 'four-workflows/evidence.json'
        data = json.loads(evidence.read_text(encoding='utf-8'))
        assert result['passed'] and data['checks_passed'] and len(data['workflows']) == 4
        summary['canonical_evidence'] = str(evidence.relative_to(root))
        summary['cache_and_repetition'] = {name: {
            'cold_repeat_wav_equal': item['cold_repeat_wav_equal'],
            'warm_all_artifacts_equal': item['warm_all_artifacts_equal'],
            'runs': {variant: {key: row[key] for key in ('elapsed_sec', 'hits', 'misses')}
                     for variant, row in item['runs'].items()}}
            for name, item in data['workflows'].items()}
    summary['outputs_sha256'] = {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob('*')) if p.is_file()}
    write_json(root / 'summary.json', summary)
    return 0 if passed else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--preflight', action='store_true', help='Default: plan only, no assets or optional imports')
    mode.add_argument('--run', action='store_true', help='Enable pinned native evidence; prerequisites are fatal')
    mode.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    for flag in ('manifest', 'dll', 'dll-sha256', 'out', 'checkout'):
        parser.add_argument('--' + flag)
    args = parser.parse_args()
    if args.worker:
        return worker()
    if not args.run:
        print(json.dumps({'mode': 'preflight', 'renderer_executed': False, 'pins': PINS,
                          'required_inputs': ['approved_manifest', 'dll', 'dll_sha256', 'out', 'checkout'],
                          'python': '3.11 x64 on native Windows',
                          'policy': 'No downloads, asset discovery, secrets, or native imports in preflight; '
                                    'enabled prerequisites fail, never skip'}, indent=2))
        return 0
    try:
        return run(args)
    except (AssertionError, ImportError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f'Windows renderer evidence FAILED: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
