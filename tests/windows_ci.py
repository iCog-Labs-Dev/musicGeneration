"""Run required native Windows tests against a wheel outside the checkout."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import unittest

OPTIONAL_CASES = {
    'test_junction_between_preflight_and_first_retain_never_receives_content': 'junction',
    'test_failed_handoff_cleanup_keeps_only_empty_owned_recovery_record': 'junction',
    'test_native_junction_destinations_are_refused_without_touching_target': 'junction',
    'test_native_symlink_destinations_are_refused_without_touching_target': 'symlink',
}
ASSET_VARIABLES = ('AIMUSIC_RUN_FLUIDSYNTH_TESTS', 'AIMUSIC_FLUIDSYNTH_MANIFEST',
                   'AIMUSIC_FLUIDSYNTH_ARTIFACTS', 'AIMUSIC_SOUNDFONT')

PREFIX = 'tests.test_audio_windows_native.TestNativeWindowsPublication.'

# The external checks directory contains the entire tests package, including the
# archive -> blinding -> bundle fixture helpers. Do not package them in the wheel.
# Select portable classes explicitly: POSIX descriptor/swap tests are not Windows
# requirements, and must not require broadening the allowed-skip policy.
ARCHIVE_SUITES = (
    'tests.test_audio_review_archive',
    'tests.test_audio_review_archive_cli',
    'tests.test_audio_review_prepare',
    'tests.test_audio_review_archive_docs',
    'tests.test_audio_review_archive_security.TestVerifiedExtraction',
    'tests.test_audio_review_archive_security.TestBackendCopyIntegrity',
    'tests.test_audio_review_archive_security.TestWindowsPublicReadback',
)

FINDING_SUITES = (
    'tests.test_audio_review_findings',
    'tests.test_audio_review_findings_cli',
    'tests.test_audio_review_findings_docs',
    'tests.test_audio_review_report',
)


def load_required_suite() -> unittest.TestSuite:
    return unittest.defaultTestLoader.loadTestsFromNames([
        'tests.test_architecture', 'tests.test_render_package', 'tests.test_audio_windows_native',
        *ARCHIVE_SUITES, *FINDING_SUITES,
    ])


def allowed_skip(test_id: str, reason: str) -> bool:
    kind = OPTIONAL_CASES.get(test_id.removeprefix(PREFIX)) if test_id.startswith(PREFIX) else None
    return kind is not None and reason in {
        f'CAPABILITY_UNAVAILABLE {kind}: WinError {code}; substitutes mandatory'
        for code in (5, 50, 1314)
    }


def verify_asset_free_environment() -> None:
    present = [name for name in ASSET_VARIABLES if name in os.environ]
    if present:
        raise RuntimeError(f'Asset-free environment must remove (not empty) variables: {present}')


def verify_install(checkout: Path) -> Path:
    import aimusic
    installed = Path(aimusic.__file__).resolve()
    assert not Path.cwd().resolve().is_relative_to(checkout), 'Must run outside checkout'
    assert installed.is_relative_to(Path(sys.prefix).resolve()), 'Not a venv-installed wheel'
    assert not installed.is_relative_to(checkout), 'Checkout/editable import contamination'
    assert all(not Path(p).resolve().is_relative_to(checkout) for p in sys.path), 'Checkout on sys.path'
    assert ' ' in str(installed) and not str(installed).isascii(), 'Missing space/non-ASCII installation path'
    print(f'Installed wheel verified: {installed}', flush=True)
    return installed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, required=True)
    args = parser.parse_args()
    assert os.name == 'nt', 'This is the required native Windows lane, not API simulation'
    from tests.windows_token import require_standard_user
    token = require_standard_user()
    print(f'Windows standard-user token verified: {token}', flush=True)
    verify_asset_free_environment()
    verify_install(args.checkout.resolve())
    from importlib.util import find_spec
    assert find_spec('fluidsynth') is None, 'Asset-free lane must not install FluidSynth bindings'
    from tests.import_all import import_all_executable_modules
    print(f'Imported {len(import_all_executable_modules(include_ui=False))} wheel modules (UI is not packaged)')
    assert 'fluidsynth' not in sys.modules
    suite = load_required_suite()
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    rejected = [(test.id(), reason) for test, reason in result.skipped if not allowed_skip(test.id(), reason)]
    report = {'native_windows': True, 'token': token, 'asset_environment_clean': True, 'tests_run': result.testsRun,
              'optional_unavailable': [(test.id(), reason) for test, reason in result.skipped],
              'substituted': [test.id() for test, reason in result.skipped if allowed_skip(test.id(), reason)],
              'required_invariants': 'passed' if result.wasSuccessful() and not rejected else 'failed',
              'unexpected_skips': rejected, 'passed': result.wasSuccessful() and not rejected}
    Path('windows-ci-results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
