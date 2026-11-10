"""Shared audio publication contracts for OpenSpec task 2.1.

The proposed internal API preserves the existing stem pin_output/publication
shape. ``replace=True`` opts into groove-style replacement; ``sources`` protects
input identities. The only injected implementation seam is ``_open_backend``:
backend acquisition must turn missing guarantees into an actionable error.

Tests assert filesystem outcomes, not journal filenames, platform syscalls, or
set-wide crash atomicity. No renderer, audio extra, native asset, or network is
needed. Link capabilities are probed individually, never by skipping a platform.
"""
from __future__ import annotations

import errno
from functools import lru_cache
import importlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from unittest.mock import patch


_SHARED_MODULE = 'aimusic.audio.publication'



@lru_cache(maxsize=1)
def _native_reader_api():
    from aimusic.audio.windows_native import WindowsAPI
    return WindowsAPI()


def _read_public(path: Path) -> bytes:
    """Observe live Windows rename handles without dropping DELETE sharing.

    Ordinary CRT readers do not share delete access. Use a native reader during
    the transaction, rather than skipping unreadable/partial public samples.
    POSIX still uses exactly the original read path.
    """
    if os.name != 'nt':
        return path.read_bytes()
    from aimusic.audio.windows_native import READ, SHARE_ALL
    api = _native_reader_api()
    with api.open(path, READ, share=SHARE_ALL) as handle:
        pieces = []
        while data := api.read(handle, 65536):
            pieces.append(data)
        return b''.join(pieces)


def _snapshot(root: Path) -> dict[str, tuple[str, bytes | str]]:
    """Include private entries and link identities without following leaf links."""
    result: dict[str, tuple[str, bytes | str]] = {}
    for path in root.rglob('*'):
        name = str(path.relative_to(root))
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            result[name] = ('link', str(path.readlink()))
        elif stat.S_ISDIR(info.st_mode):
            result[name] = ('directory', '')
        else:
            result[name] = ('file', path.read_bytes())
    return result


@contextmanager
def _observe_public_bytes(output: Path, permitted: dict[str, tuple[bytes | None, ...]]):
    """Sample at Python execution boundaries, not with a timing-sensitive poll.

    Tracing also enters stdlib copy helpers, so an accidental open/truncate/copy
    into a public leaf exposes its empty/partial bytes to the observer. No
    particular copy, link, rename, backend, or private staging layout is assumed.
    This is a per-file assertion: different leaves may change at different times.
    """
    violations: list[tuple[str, bytes | None]] = []
    observed: dict[str, set[bytes | None]] = {name: set() for name in permitted}

    def observe(frame, event, arg):
        # Native metadata verification has no write side effects. Trace every
        # mutation-capable wrapper line (including failed opens/short writes),
        # callers and Python copying, but not repeated read-only identity probes.
        module = frame.f_globals.get('__name__', '')
        if os.name == 'nt' and not (
                module.startswith('aimusic.audio.') or module == __name__ or
                module in ('pathlib', 'shutil', 'tempfile', 'io', '_pyio')):
            return None  # Skip pure serialization/runtime bookkeeping, not I/O.
        if os.name == 'nt' and module == 'aimusic.audio.windows_native':
            if frame.f_code.co_name == 'open' and not frame.f_locals.get('access', 0) & 0x40000000:
                return None  # Metadata/source READ opens cannot truncate bytes.
        if os.name == 'nt' and module in ('aimusic.audio.windows_native', 'aimusic.audio.publication_windows'):
            if frame.f_code.co_name not in {
                'open', 'write', 'rewrite', 'rename', 'dispose', 'stage',
                'retain', 'move', 'remove', 'cleanup', 'backup', 'publish', 'prepare',
                'acquire_pin', 'restore', 'journal',
            }:
                return None
        if event in ('line', 'return'):
            for name, allowed in permitted.items():
                try:
                    contents = _read_public(output / name)
                except FileNotFoundError:
                    contents = None
                observed[name].add(contents)
                if contents not in allowed and not violations:
                    violations.append((name, contents))
        return observe

    previous = sys.gettrace()
    sys.settrace(observe)
    try:
        yield observed
    finally:
        sys.settrace(previous)
    if violations:
        name, contents = violations[0]
        raise AssertionError(f'Public leaf {name!r} exposed incomplete bytes: {contents!r}')


class TestAudioPublicationContract(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.output = self.base / 'output'
        self.staging = self.base / 'staging'
        self.output.mkdir()
        self.staging.mkdir()
        self.source = self.base / 'source.mid'
        self.source.write_bytes(b'UNCHANGED SOURCE MIDI')
        self.payloads = {'a.mid': b'NEW MIDI\x00\xff' * 1024,
                         'a.mid.json': b'{"new": true}\n',
                         'a.mid.diff.json': b'{"changes": []}\n'}
        for name, contents in self.payloads.items():
            (self.staging / name).write_bytes(contents)

    @property
    def publication(self) -> ModuleType:
        try:
            return importlib.import_module(_SHARED_MODULE)
        except ModuleNotFoundError as exc:
            if exc.name != _SHARED_MODULE:
                raise  # A missing dependency is not an acceptable RED signal.
            self.fail('Missing shared aimusic.audio.publication abstraction (OpenSpec task 2.1)')

    def _link_or_skip(self, link: Path, target: Path, *, symbolic: bool):
        """An actual operation probes this filesystem and this process's privilege."""
        try:
            if symbolic:
                link.symlink_to(target)
            else:
                link.hardlink_to(target)
        except (OSError, NotImplementedError) as exc:
            capability = 'symbolic links' if symbolic else 'hard links'
            self.skipTest(f'{capability} unavailable in test output filesystem: {exc}')

    def test_portable_names_accept_unicode_spaces_and_internal_dots(self):
        names = ['rough_mix.wav', 'track-01.mid', 'mix.v2.json', 'café 音色.wav', '.metadata.json']
        with self.publication.pin_output(self.output) as root:
            root.check_new_files(names)
        self.assertEqual(_snapshot(self.output), {})

    def test_unsafe_names_are_rejected_before_any_public_artifact(self):
        unsafe = ['', '.', '..', '../escape.wav', r'..\escape.wav', '/absolute.wav',
                  r'C:\absolute.wav', r'folder\clip.wav', 'folder/clip.wav', 'clip:stream',
                  'a<b.wav', 'a>b.wav', 'a"b.wav', 'a|b.wav', 'a?b.wav', 'a*b.wav',
                  'clip.wav ', 'clip.wav.', 'CON', 'con.mid', 'NUL.wav', 'PRN', 'AUX.txt',
                  'COM1.wav', 'com9.mid', 'LPT1.json', 'lpt9.wav', 'COM¹.wav', 'LPT².mid']
        unsafe += [f'clip{chr(control)}.wav' for control in (*range(32), 127)]
        with self.publication.pin_output(self.output) as root:
            for name in unsafe:
                with self.subTest(name=repr(name)):
                    with self.assertRaises(ValueError):
                        with root.publication(self.staging, ['a.mid', name]):
                            self.fail('Unsafe name was accepted')
                    self.assertEqual(_snapshot(self.output), {})
        self.assertEqual(self.source.read_bytes(), b'UNCHANGED SOURCE MIDI')

    def test_duplicate_and_casefold_aliases_within_one_plan_are_rejected(self):
        with self.publication.pin_output(self.output) as root:
            for names in (['a.mid', 'a.mid'], ['Mix.wav', 'mix.wav'], ['Écho.wav', 'éCHO.wav']):
                with self.subTest(names=names), self.assertRaises(ValueError):
                    root.check_new_files(names)
        self.assertEqual(_snapshot(self.output), {})

    def test_existing_casefold_alias_is_a_collision_on_every_host(self):
        existing = self.output / 'A.MID'
        existing.write_bytes(b'KEEP CASE ALIAS')
        before = _snapshot(self.output)
        with self.assertRaises(FileExistsError):
            with self.publication.pin_output(self.output) as root:
                with root.publication(self.staging, list(self.payloads)):
                    self.fail('Case-insensitive collision was accepted')
        self.assertEqual(_snapshot(self.output), before)

    def test_create_only_refuses_existing_files_and_directories(self):
        # Non-link collision coverage is mandatory even without link privileges.
        module = self.publication
        for kind in ('file', 'directory'):
            with self.subTest(kind=kind):
                target = self.output / 'a.mid.json'
                if kind == 'file':
                    target.write_bytes(b'USER BYTES')
                else:
                    target.mkdir()
                    (target / 'keep.txt').write_bytes(b'USER CHILD')
                before = _snapshot(self.output)
                with self.assertRaises(FileExistsError):
                    with module.pin_output(self.output) as root:
                        with root.publication(self.staging, list(self.payloads)):
                            self.fail('Existing entry was overwritten')
                self.assertEqual(_snapshot(self.output), before)
                if kind == 'file':
                    target.unlink()
                else:
                    (target / 'keep.txt').unlink()
                    target.rmdir()

    def test_create_only_refuses_existing_and_dangling_symlinks(self):
        module = self.publication  # Missing abstraction must not masquerade as a link skip.
        for dangling in (False, True):
            with self.subTest(dangling=dangling):
                target = self.base / ('absent.mid' if dangling else 'source.mid')
                link = self.output / 'a.mid'
                self._link_or_skip(link, target, symbolic=True)
                before = _snapshot(self.output)
                with self.assertRaises((FileExistsError, ValueError)):
                    with module.pin_output(self.output) as root:
                        with root.publication(self.staging, list(self.payloads)):
                            self.fail('Existing symlink was overwritten')
                self.assertEqual(_snapshot(self.output), before)
                self.assertEqual(self.source.read_bytes(), b'UNCHANGED SOURCE MIDI')
                link.unlink()

    def test_create_only_refuses_existing_hardlink_without_mutating_its_target(self):
        module = self.publication
        self._link_or_skip(self.output / 'a.mid', self.source, symbolic=False)
        before = _snapshot(self.output)
        with self.assertRaises((FileExistsError, ValueError)):
            with module.pin_output(self.output) as root:
                with root.publication(self.staging, list(self.payloads)):
                    self.fail('Existing hard link was overwritten')
        self.assertEqual(_snapshot(self.output), before)
        self.assertEqual(self.source.read_bytes(), b'UNCHANGED SOURCE MIDI')

    def test_visibility_observer_detects_a_partial_public_write(self):
        with self.assertRaisesRegex(AssertionError, 'incomplete bytes'):
            with _observe_public_bytes(self.output, {'probe.wav': (None, b'COMPLETE')}):
                with (self.output / 'probe.wav').open('wb') as stream:
                    stream.write(b'PARTIAL')
                    stream.flush()

    def test_create_only_publishes_exact_complete_bytes_without_residue(self):
        expected = {name: (None, contents) for name, contents in self.payloads.items()}
        with self.publication.pin_output(self.output) as root:
            with _observe_public_bytes(self.output, expected) as observed:
                with root.publication(self.staging, list(self.payloads)):
                    for name, contents in self.payloads.items():
                        self.assertEqual(_read_public(self.output / name), contents)
        self.assertEqual(_snapshot(self.output),
                         {name: ('file', contents) for name, contents in self.payloads.items()})
        for name, contents in self.payloads.items():
            self.assertIn(None, observed[name])
            self.assertIn(contents, observed[name])
            self.assertEqual((self.staging / name).read_bytes(), contents)

    def test_transaction_yields_recorded_published_ownership_before_commit(self):
        self._assert_published_ownership(direct=True)

    def test_publication_yields_recorded_published_ownership_before_commit(self):
        self._assert_published_ownership(direct=False)

    def _assert_published_ownership(self, *, direct):
        from aimusic.audio.publication_records import FileIdentity, Journal, Ownership
        from aimusic.audio.publication_transaction import transaction
        if os.name == 'nt':
            from aimusic.audio.publication_windows import WindowsWorkspace as Workspace
        else:
            from aimusic.audio.publication_posix import PosixWorkspace as Workspace
        stage, write_journal = Workspace.stage, Workspace.journal
        staged, records = [], []
        # Deliberately not lexical order: preserve the caller's plan order.
        names = list(reversed(self.payloads))

        def record_stage(work, source, name):
            identity = stage(work, source, name)
            staged.append(Ownership(source.name, identity))
            return identity

        def record_journal(work, data):
            write_journal(work, data)
            records.append(json.loads(data))

        (self.output / 'keep.txt').write_bytes(b'UNRELATED')
        with patch.object(Workspace, 'stage', record_stage), patch.object(Workspace, 'journal', record_journal):
            with self.publication.pin_output(self.output) as root:
                context = (transaction(root.backend, self.staging, Journal('.stems-contract', tuple(names), False))
                           if direct else root.publication(self.staging, names))
                with self.assertRaisesRegex(RuntimeError, 'reject before commit'):
                    with context as published:
                        self.assertIsInstance(published, tuple)
                        self.assertEqual(published, tuple(staged))
                        self.assertEqual([owner.name for owner in published], names)
                        self.assertEqual(records[-1]['phase'], 'published')
                        self.assertEqual(records[-1]['published'], records[-1]['staged'])
                        self.assertNotIn('committed', [record['phase'] for record in records])
                        for owner, recorded in zip(published, staged):
                            self.assertIs(owner.identity, recorded.identity)
                            self.assertEqual(FileIdentity.from_stat(root.backend.info(owner.name)), owner.identity)
                            self.assertEqual(_read_public(self.output / owner.name), self.payloads[owner.name])
                        raise RuntimeError('reject before commit')
        self.assertEqual(_snapshot(self.output), {'keep.txt': ('file', b'UNRELATED')})
        self.assertNotIn('committed', [record['phase'] for record in records])

    def test_replacement_exposes_only_old_or_complete_new_bytes(self):
        old = {name: b'PREVIOUS ' + name.encode() for name in self.payloads}
        for name, contents in old.items():
            (self.output / name).write_bytes(contents)
        # Moving old entries to recovery may briefly make a leaf absent; the
        # contract is complete-file visibility, not an atomic set-wide switch.
        expected = {name: (None, old[name], contents) for name, contents in self.payloads.items()}
        with self.publication.pin_output(self.output) as root:
            with _observe_public_bytes(self.output, expected):
                with root.publication(self.staging, list(self.payloads), replace=True, sources=(self.source,)):
                    for name, contents in self.payloads.items():
                        self.assertEqual(_read_public(self.output / name), contents)
        self.assertEqual(_snapshot(self.output),
                         {name: ('file', contents) for name, contents in self.payloads.items()})
        self.assertEqual(self.source.read_bytes(), b'UNCHANGED SOURCE MIDI')

    def test_replacement_refuses_the_source_path_itself(self):
        source = self.output / 'a.mid.json'
        source.write_bytes(b'SOURCE MUST NOT CHANGE')
        before = _snapshot(self.output)
        with self.assertRaisesRegex(ValueError, 'source|input|alias'):
            with self.publication.pin_output(self.output) as root:
                with root.publication(self.staging, list(self.payloads), replace=True, sources=(source,)):
                    self.fail('Source path was replaced')
        self.assertEqual(_snapshot(self.output), before)

    def test_replacement_refuses_source_hardlink_aliases(self):
        self._assert_source_link_alias_rejected(symbolic=False)

    def test_replacement_refuses_source_symlink_aliases(self):
        self._assert_source_link_alias_rejected(symbolic=True)

    def _assert_source_link_alias_rejected(self, *, symbolic: bool):
        module = self.publication
        alias = self.output / 'a.mid.diff.json'
        self._link_or_skip(alias, self.source, symbolic=symbolic)
        (self.output / 'a.mid').write_bytes(b'PRIOR MIDI')
        before = _snapshot(self.output)
        with self.assertRaisesRegex(ValueError, 'source|input|alias|symlink'):
            with module.pin_output(self.output) as root:
                with root.publication(self.staging, list(self.payloads), replace=True, sources=(self.source,)):
                    self.fail('Source alias was replaced')
        self.assertEqual(_snapshot(self.output), before)
        self.assertEqual(self.source.read_bytes(), b'UNCHANGED SOURCE MIDI')

    def test_caller_failure_rolls_back_owned_files_but_preserves_unrelated_entries(self):
        (self.output / 'keep.txt').write_bytes(b'UNRELATED')
        (self.output / 'user-folder').mkdir()
        (self.output / 'user-folder' / 'keep.txt').write_bytes(b'UNRELATED CHILD')
        before = _snapshot(self.output)
        with self.assertRaisesRegex(RuntimeError, 'caller failed'):
            with self.publication.pin_output(self.output) as root:
                with root.publication(self.staging, list(self.payloads)):
                    self.assertTrue((self.output / 'a.mid').exists())
                    raise RuntimeError('caller failed')
        self.assertEqual(_snapshot(self.output), before)
        self.assertEqual(self.source.read_bytes(), b'UNCHANGED SOURCE MIDI')

    def test_collision_after_preflight_rolls_back_an_earlier_related_publication(self):
        (self.output / 'keep.txt').write_bytes(b'UNRELATED')
        entered = collision_created = False
        with self.assertRaises((FileExistsError, RuntimeError)):
            with self.publication.pin_output(self.output) as root:
                root.check_new_files(list(self.payloads))
                with root.publication(self.staging, ['a.mid']):
                    entered = True
                    # A related publication keeps the outer ownership rollback
                    # active, just as the existing stem context manager does.
                    try:
                        (self.output / 'a.mid.json').write_bytes(b'OTHER WRITER')
                    except PermissionError:
                        # A backend may prohibit the competing mutation instead
                        # of allowing it and detecting the collision. Exercise
                        # rollback in this branch too, without a platform skip.
                        raise RuntimeError('competitor blocked; caller failed')
                    collision_created = True
                    with root.publication(self.staging, ['a.mid.json']):
                        self.fail('Post-preflight collision was overwritten')
        self.assertTrue(entered, 'The failure must occur after owned publication')
        expected = {'keep.txt': ('file', b'UNRELATED')}
        if collision_created:
            expected['a.mid.json'] = ('file', b'OTHER WRITER')
        self.assertEqual(_snapshot(self.output), expected)

    def test_rollback_never_removes_a_substituted_unowned_leaf(self):
        (self.output / 'keep.txt').write_bytes(b'UNRELATED')
        intruder = self.base / 'other-writer.mid'
        intruder.write_bytes(b'OTHER WRITER IDENTITY')
        entered = replaced = False
        with self.assertRaises((RuntimeError, OSError, ValueError)):
            with self.publication.pin_output(self.output) as root:
                with root.publication(self.staging, list(self.payloads)):
                    entered = True
                    try:
                        intruder.replace(self.output / 'a.mid')
                    except PermissionError:
                        # Some pinned backends prevent this attack outright.
                        # Ordinary rollback still runs and must leave the
                        # would-be attacker's external file untouched.
                        self.assertEqual(_read_public(self.output / 'a.mid'), self.payloads['a.mid'])
                    else:
                        replaced = True
                    raise RuntimeError('caller failed')
        self.assertTrue(entered, 'The failure must occur after owned publication')
        expected = {'keep.txt': ('file', b'UNRELATED')}
        if replaced:
            expected['a.mid'] = ('file', b'OTHER WRITER IDENTITY')
        else:
            self.assertEqual(intruder.read_bytes(), b'OTHER WRITER IDENTITY')
        self.assertEqual(_snapshot(self.output), expected)

    def test_replacement_failure_restores_old_members_and_removes_only_new_members(self):
        (self.output / 'a.mid').write_bytes(b'PREVIOUS MIDI')
        (self.output / 'a.mid.json').write_bytes(b'PREVIOUS MANIFEST')
        (self.output / 'keep.txt').write_bytes(b'UNRELATED')
        before = _snapshot(self.output)
        with self.assertRaisesRegex(RuntimeError, 'caller failed'):
            with self.publication.pin_output(self.output) as root:
                with root.publication(self.staging, list(self.payloads), replace=True, sources=(self.source,)):
                    self.assertEqual(_read_public(self.output / 'a.mid.diff.json'), self.payloads['a.mid.diff.json'])
                    raise RuntimeError('caller failed')
        self.assertEqual(_snapshot(self.output), before)
        self.assertEqual(self.source.read_bytes(), b'UNCHANGED SOURCE MIDI')

    def test_missing_staged_member_leaves_no_partial_publication_or_private_residue(self):
        (self.output / 'keep.txt').write_bytes(b'UNRELATED')
        before = _snapshot(self.output)
        with self.assertRaises(FileNotFoundError):
            with self.publication.pin_output(self.output) as root:
                with root.publication(self.staging, ['a.mid', 'missing.mid']):
                    self.fail('Missing staged member was accepted')
        self.assertEqual(_snapshot(self.output), before)

    def test_crashed_transaction_residue_is_reported_and_never_reused_or_deleted(self):
        module = self.publication
        for replace in (False, True):
            with self.subTest(replace=replace):
                output = self.base / ('replace-residue' if replace else 'create-residue')
                output.mkdir()
                old = b'ONLY OLD RECOVERY COPY'
                if replace:
                    (output / 'a.mid').write_bytes(old)
                script = '''
import json, os, sys
from pathlib import Path
from aimusic.audio.publication import pin_output
with pin_output(Path(sys.argv[1])) as root:
    with root.publication(Path(sys.argv[2]), json.loads(sys.argv[3]), replace=sys.argv[4] == 'True'):
        os._exit(73)  # No Python finally blocks: leave a genuine unfinished transaction.
'''
                result = subprocess.run(
                    [sys.executable, '-c', script, str(output), str(self.staging),
                     json.dumps(list(self.payloads)), str(replace)],
                    cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 73, result.stdout + result.stderr)
                residue = [path for path in output.iterdir() if path.name not in self.payloads]
                self.assertTrue(residue, 'An unfinished transaction needs identifiable recovery state')
                before = _snapshot(output)
                if replace:
                    self.assertIn(('file', old), before.values(), 'The only prior copy must survive the crash')
                (self.staging / 'later.wav').write_bytes(b'UNRELATED LATER ATTEMPT')
                with self.assertRaises((OSError, ValueError, RuntimeError)) as caught:
                    with module.pin_output(output) as root:
                        # Use a fresh leaf: a normal existing-output collision
                        # must not accidentally satisfy the residue contract.
                        with root.publication(self.staging, ['later.wav']):
                            self.fail('Ambiguous transaction residue was ignored')
                diagnostic = str(caught.exception)
                self.assertRegex(diagnostic.lower(), 'residue|recovery|unfinished|transaction')
                self.assertIn(str(output), diagnostic)
                self.assertTrue(any(path.name in diagnostic for path in residue), diagnostic)
                self.assertEqual(_snapshot(output), before)
                self.assertEqual(self.source.read_bytes(), b'UNCHANGED SOURCE MIDI')

    def test_missing_backend_guarantees_fail_with_actionable_diagnostics_and_no_fallback(self):
        module = self.publication
        (self.output / 'keep.txt').write_bytes(b'UNRELATED')
        before = _snapshot(self.output)
        for failure in (NotImplementedError('required atomic publication API is unavailable'),
                        OSError(errno.ENOTSUP, 'filesystem cannot provide pinned containment')):
            with self.subTest(failure=type(failure).__name__):
                # This is the platform backend acquisition boundary, not a mock
                # of the publication policy, filesystem, or rollback journal.
                with patch.object(module, '_open_backend', side_effect=failure) as opening:
                    with self.assertRaises((OSError, RuntimeError)) as caught:
                        with module.pin_output(self.output) as root:
                            with root.publication(self.staging, list(self.payloads)):
                                self.fail('Unsupported environment reported success')
                self.assertTrue(opening.called, 'The injected capability failure must be exercised')
                diagnostic = str(caught.exception)
                self.assertIn(str(self.output), diagnostic)
                self.assertRegex(diagnostic.lower(), 'unsupported|unavailable|cannot provide')
                self.assertRegex(diagnostic.lower(), 'use |choose |move |supported .*filesystem|supported .*runtime')
                self.assertEqual(_snapshot(self.output), before)
                self.assertEqual(self.source.read_bytes(), b'UNCHANGED SOURCE MIDI')


if __name__ == '__main__':
    unittest.main()
