"""Host-independent Win32 ABI and native publication regression tests."""
from __future__ import annotations

import ctypes
import importlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock


class TestWindowsNativeAPI(unittest.TestCase):
    def api(self):
        module = importlib.import_module('aimusic.audio.windows_native')
        kernel = SimpleNamespace(**{name: Mock(return_value=1) for name in (
            'CreateFileW', 'CloseHandle', 'GetFileInformationByHandle',
            'GetVolumeInformationByHandleW', 'GetDriveTypeW', 'ReadFile', 'WriteFile',
            'FlushFileBuffers', 'SetFilePointerEx', 'SetEndOfFile', 'SetFileInformationByHandle')})
        native = SimpleNamespace(NtSetInformationFile=Mock(return_value=0),
                                 RtlNtStatusToDosError=Mock(return_value=32))
        return module, module.WindowsAPI(kernel, native, last_error=lambda: 5), kernel, native

    def test_signatures_preserve_64_bit_handles_and_32_bit_windows_scalars(self):
        m, api, kernel, native = self.api()
        self.assertIs(kernel.CreateFileW.restype, ctypes.c_void_p)
        self.assertEqual(ctypes.sizeof(native.NtSetInformationFile.restype), 4)
        self.assertEqual(ctypes.sizeof(m.DWORD), 4)
        self.assertEqual(m.RenameInfo.FileName.offset, 20 if ctypes.sizeof(ctypes.c_void_p) == 8 else 12)

    def test_rename_is_same_directory_unicode_and_never_replaces(self):
        m, api, kernel, native = self.api()
        handle = m.Handle(api, 2 ** 33)
        name = 'caf\u00e9 \U0001f3b5.wav'
        api.rename(handle, name)
        args = native.NtSetInformationFile.call_args.args
        self.assertEqual(args[0], 2 ** 33)
        self.assertEqual(args[4], 10)
        info = m.RenameInfo.from_buffer(args[2])
        self.assertFalse(info.ReplaceIfExists)
        self.assertIsNone(info.RootDirectory)
        self.assertEqual(info.FileNameLength, len(name.encode('utf-16-le')))
        self.assertEqual(ctypes.string_at(ctypes.addressof(args[2]) + m.RenameInfo.FileName.offset,
                                         info.FileNameLength), name.encode('utf-16-le'))
        kernel.SetFileInformationByHandle.assert_not_called()

    def test_rename_refuses_paths_before_native_call(self):
        m, api, kernel, native = self.api()
        for name in ('../other', r'a\b', 'a:b', '', '.', '..', 'NUL', 'name.'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                api.rename(m.Handle(api, 123), name)
        native.NtSetInformationFile.assert_not_called()

    def test_checked_errors_and_handle_closed_exactly_once(self):
        m, api, kernel, native = self.api()
        kernel.CreateFileW.return_value = ctypes.c_void_p(-1).value
        with self.assertRaises(OSError) as caught:
            api.open('C:\\safe', 0x81)
        self.assertEqual(caught.exception.winerror, 5)
        kernel.CreateFileW.return_value = 2 ** 33
        with api.open('C:\\safe', 0x81) as handle:
            self.assertEqual(handle.value, 2 ** 33)
        handle.close()
        kernel.CloseHandle.assert_called_once_with(2 ** 33)
        with self.assertRaises(ValueError):
            api.flush(handle)
        native.NtSetInformationFile.return_value = -1073741757  # STATUS_SHARING_VIOLATION
        with self.assertRaises(OSError) as caught:
            api.rename(m.Handle(api, 123), 'leaf')
        self.assertEqual(caught.exception.winerror, 32)

    def test_all_bool_failure_branches_raise_checked_errors(self):
        m, api, kernel, native = self.api()
        h = m.Handle(api, 123)
        actions = {
            'GetFileInformationByHandle': lambda: api.information(h),
            'GetVolumeInformationByHandleW': lambda: api.require_local_ntfs(h, 'C:\\out'),
            'ReadFile': lambda: api.read(h),
            'WriteFile': lambda: api.write(h, b'bytes'),
            'FlushFileBuffers': lambda: api.flush(h),
            'SetFilePointerEx': lambda: api.rewrite(h, b''),
            'SetEndOfFile': lambda: api.rewrite(h, b''),
            'SetFileInformationByHandle': lambda: api.dispose(h),
        }
        kernel.GetDriveTypeW.return_value = 3
        for name, action in actions.items():
            with self.subTest(api=name):
                function = getattr(kernel, name)
                function.return_value = 0
                with self.assertRaises(OSError) as caught:
                    action()
                self.assertEqual(caught.exception.winerror, 5)
                self.assertIn(name, str(caught.exception))
                function.return_value = 1
        kernel.CloseHandle.return_value = 0
        with self.assertRaises(OSError):
            h.close()
        h.close()
        kernel.CloseHandle.assert_called_once()

    def test_information_identity_reparse_and_volume_capability(self):
        m, api, kernel, native = self.api()
        def information(handle, pointer):
            pointer._obj.attributes = 0x10
            pointer._obj.volume = 0xfedcba98
            pointer._obj.index_high = 0x87654321
            pointer._obj.index_low = 0x12345678
            pointer._obj.links = 1
            return 1
        kernel.GetFileInformationByHandle.side_effect = information
        info = api.information(m.Handle(api, 123))
        self.assertEqual(info.identity.inode, 0x8765432112345678)
        self.assertEqual(info.identity.device, 0xfedcba98)
        info.require(directory=True)
        with self.assertRaises(ValueError):
            info.require(directory=False)
        for attributes in (0x400, 0x410):
            with self.subTest(attributes=attributes), self.assertRaises(ValueError):
                m.Information(info.identity, attributes, 0, 1).require(directory=bool(attributes & 0x10))
        for drive in (0, 1, 2, 4, 5, 6):
            kernel.GetDriveTypeW.return_value = drive
            with self.subTest(drive=drive), self.assertRaises(NotImplementedError):
                api.require_local_ntfs(m.Handle(api, 123), 'C:\\out')
        kernel.GetDriveTypeW.return_value = 3
        def volume(handle, label, length, serial, maximum, flags, filesystem, size):
            filesystem.value = 'NTFS'
            flags._obj.value = 0
            return 1
        kernel.GetVolumeInformationByHandleW.side_effect = volume
        api.require_local_ntfs(m.Handle(api, 123), 'C:\\out')
        for filesystem, flags in (('FAT32', 0), ('ReFS', 0), ('NTFS', 0x80000)):
            def unsupported(*args):
                args[6].value = filesystem
                args[5]._obj.value = flags
                return 1
            kernel.GetVolumeInformationByHandleW.side_effect = unsupported
            with self.subTest(filesystem=filesystem, flags=flags), self.assertRaises(NotImplementedError):
                api.require_local_ntfs(m.Handle(api, 123), 'C:\\out')

    def test_short_io_zero_progress_and_disposition_structure(self):
        m, api, kernel, native = self.api()
        h = m.Handle(api, 123)
        written = bytearray()
        def write(handle, buffer, count, size, overlapped):
            size._obj.value = min(2, count)
            written.extend(buffer.raw[:size._obj.value])
            return 1
        kernel.WriteFile.side_effect = write
        api.rewrite(h, b'abcde')
        self.assertEqual(written, b'abcde')
        kernel.FlushFileBuffers.assert_called_once_with(123)
        for count in (0, 999):
            def bad_write(handle, buffer, length, size, overlapped):
                size._obj.value = count
                return 1
            kernel.WriteFile.side_effect = bad_write
            with self.subTest(count=count), self.assertRaises(OSError):
                api.write(h, b'x')
        def read(handle, buffer, count, size, overlapped):
            ctypes.memmove(buffer, b'hello', 5)
            size._obj.value = 5
            return 1
        kernel.ReadFile.side_effect = read
        self.assertEqual(api.read(h, 8), b'hello')
        api.dispose(h)
        args = kernel.SetFileInformationByHandle.call_args.args
        self.assertEqual((args[0], args[1], args[2]._obj.value, args[3]), (123, 4, 1, 1))

    def test_null_handles_and_invalid_read_counts_fail_closed(self):
        m, api, kernel, native = self.api()
        kernel.CreateFileW.return_value = 0
        with self.assertRaises(OSError):
            api.open('C:\\out', m.ATTRIBUTES)
        for count in (0, -1, 2 ** 32):
            with self.subTest(count=count), self.assertRaises(ValueError):
                api.read(m.Handle(api, 123), count)
        kernel.ReadFile.assert_not_called()

    def test_unavailable_runtime_api_and_oversized_native_read_fail_closed(self):
        m, api, kernel, native = self.api()
        with patch.object(m, 'os', SimpleNamespace(name='posix')):
            with self.assertRaisesRegex(NotImplementedError, 'requires Windows'):
                m.WindowsAPI()
        del kernel.SetFileInformationByHandle
        with self.assertRaisesRegex(NotImplementedError, 'API is unavailable'):
            m.WindowsAPI(kernel, native, last_error=lambda: 5)
        def oversized(handle, buffer, count, size, overlapped):
            size._obj.value = count + 1
            return 1
        kernel.ReadFile.side_effect = oversized
        with self.assertRaisesRegex(OSError, 'invalid byte count'):
            api.read(m.Handle(api, 123), 8)

    def test_platform_paths_open_flags_and_status_mapping(self):
        m, api, kernel, native = self.api()
        for path in ('relative', 'C:relative', r'\\server\share\out', r'\\.\C:', r'C:\out\..\escape'):
            with self.subTest(path=path), self.assertRaises(NotImplementedError):
                api.open(path, m.ATTRIBUTES)
        api.open('C:\\out', m.DIRECTORY_ACCESS)
        args = kernel.CreateFileW.call_args.args
        self.assertEqual((args[1], args[2], args[4], args[5]), (0x81, 1, 3, 0x02200000))
        api.open('C:\\out\\private', m.READ | m.WRITE | m.DELETE, create=True)
        self.assertEqual(kernel.CreateFileW.call_args.args[4], 1)
        for code, exception in ((2, FileNotFoundError), (3, FileNotFoundError),
                                (80, FileExistsError), (183, FileExistsError), (32, PermissionError)):
            self.assertIsInstance(api.error('operation', code), exception)
        native.NtSetInformationFile.return_value = 0x103
        with self.assertRaises(NotImplementedError):
            api.rename(m.Handle(api, 123), 'leaf')


import os
import tempfile
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(os.name == 'nt', 'native Windows NTFS lane; ABI tests above run on every host')
class TestNativeWindowsPublication(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tests.windows_capabilities import probe
        with tempfile.TemporaryDirectory() as tmp:
            cls.capabilities = {kind: probe(kind, Path(tmp)) for kind in ('hardlink', 'junction', 'symlink')}
        for kind, capability in cls.capabilities.items():
            state = 'exercised' if capability.available else 'unavailable'
            print(f'WINDOWS_CAPABILITY {kind}: {state} ({capability.reason})', flush=True)
        if not cls.capabilities['hardlink'].available:
            raise AssertionError('Required local NTFS hardlink capability unavailable')
        print('WINDOWS_INVARIANTS mandatory: native directory pin, identity handoff, collision, '
              'rollback, mocked reparse rejection; substituted when an optional link is unavailable', flush=True)

    def require_capability(self, kind):
        capability = self.capabilities[kind]
        if not capability.available:
            self.skipTest(f'CAPABILITY_UNAVAILABLE {kind}: {capability.reason}; substitutes mandatory')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.out, self.staging = self.base / 'output', self.base / 'staging'
        self.out.mkdir()
        self.staging.mkdir()
        for name in ('a.mid', 'b.json'):
            (self.staging / name).write_bytes(('NEW ' + name).encode() * 3000)

    def publish(self, *, replace=False):
        from aimusic.audio.publication import pin_output
        with pin_output(self.out) as root:
            root.publish(self.staging, ['a.mid', 'b.json'], replace=replace)

    def test_create_only_and_replacement_complete_without_residue(self):
        self.publish()
        for name in ('a.mid', 'b.json'):
            self.assertEqual((self.out / name).read_bytes(), (self.staging / name).read_bytes())
        (self.staging / 'a.mid').write_bytes(b'REPLACEMENT')
        self.publish(replace=True)
        self.assertEqual((self.out / 'a.mid').read_bytes(), b'REPLACEMENT')
        self.assertEqual(sorted(p.name for p in self.out.iterdir()), ['a.mid', 'b.json'])

    def test_failure_mid_publish_restores_originals_and_keeps_unrelated(self):
        from aimusic.audio.windows_native import WindowsAPI
        for name in ('a.mid', 'b.json', 'keep'):
            (self.out / name).write_bytes(b'OLD ' + name.encode())
        before = {p.name: p.read_bytes() for p in self.out.iterdir()}
        rename = WindowsAPI.rename
        failed = []
        def fail(api, handle, name):
            if name == 'b.json' and not failed:
                failed.append(name)
                raise OSError('injected publish failure')
            return rename(api, handle, name)
        with patch.object(WindowsAPI, 'rename', fail), self.assertRaises(OSError):
            self.publish(replace=True)
        self.assertEqual(failed, ['b.json'])
        self.assertEqual({p.name: p.read_bytes() for p in self.out.iterdir()}, before)

    def test_unicode_spaces_long_paths_and_concurrent_collision(self):
        from aimusic.audio.publication import pin_output
        from aimusic.audio.windows_native import WindowsAPI
        out = self.base / ('long-' * 30) / ('path-' * 30) / 'caf\u00e9 \u97f3\u8272'
        name = 'm\u00fasica \U0001f3b5.wav'
        (self.staging / name).write_bytes(b'COMPLETE AUDIO')
        with pin_output(out) as root:
            root.publish(self.staging, [name])
        self.assertEqual((out / name).read_bytes(), b'COMPLETE AUDIO')
        rename = WindowsAPI.rename
        def collide(api, handle, leaf):
            if leaf == 'b.json':
                (self.out / leaf).write_bytes(b'OTHER WRITER')
            return rename(api, handle, leaf)
        with patch.object(WindowsAPI, 'rename', collide), self.assertRaises(FileExistsError):
            self.publish()
        self.assertEqual({p.name: p.read_bytes() for p in self.out.iterdir()}, {'b.json': b'OTHER WRITER'})

    def test_cleanup_failure_reports_recovery_path_and_phase(self):
        import json
        from aimusic.audio.publication import pin_output
        from aimusic.audio.publication_windows import WindowsWorkspace
        cleanup = WindowsWorkspace.cleanup
        failed = []
        def fail(work, *, keep_backups=False):
            if not keep_backups and not failed:
                failed.append(True)
                raise OSError('private cleanup failed')
            return cleanup(work, keep_backups=keep_backups)
        with patch.object(WindowsWorkspace, 'cleanup', fail), self.assertRaisesRegex(OSError, 'recovery') as caught:
            with pin_output(self.out) as root:
                with root.publication(self.staging, ['a.mid', 'b.json']):
                    raise RuntimeError('caller failed')
        record = next(self.out.glob('*.journal.json'))
        self.assertEqual(json.loads(record.read_bytes())['phase'], 'recovery')
        self.assertIn(str(record), str(caught.exception))
        self.assertEqual(list(self.out.iterdir()), [record])

    def test_exception_after_backup_rename_preserves_only_original_copy(self):
        import json
        from aimusic.audio.windows_native import WindowsAPI
        (self.out / 'a.mid').write_bytes(b'ONLY ORIGINAL COPY')
        rename = WindowsAPI.rename
        def interrupt(api, handle, name):
            rename(api, handle, name)
            if name.endswith('.backup-0'):
                raise KeyboardInterrupt('after kernel rename, before Python bookkeeping')
        with patch.object(WindowsAPI, 'rename', interrupt), self.assertRaisesRegex(OSError, 'recovery') as caught:
            self.publish(replace=True)
        record = next(self.out.glob('*.journal.json'))
        self.assertIn(str(record), str(caught.exception))
        self.assertEqual(json.loads(record.read_bytes())['phase'], 'recovery')
        self.assertEqual(next(self.out.glob('*.backup-0')).read_bytes(), b'ONLY ORIGINAL COPY')
        self.assertFalse((self.out / 'a.mid').exists())

    def test_interruption_after_publication_before_bookkeeping_rolls_back(self):
        from aimusic.audio.publication_windows import WindowsWorkspace
        from aimusic.audio.windows_native import WindowsAPI
        for boundary, target in (('publish', WindowsWorkspace), ('rename', WindowsAPI)):
            for replace in (False, True):
                with self.subTest(boundary=boundary, replace=replace):
                    self.out = self.base / f'{boundary}-{replace}'
                    self.out.mkdir()
                    (self.out / 'keep').write_bytes(b'UNRELATED')
                    if replace:
                        (self.out / 'a.mid').write_bytes(b'ORIGINAL MIDI')
                        (self.out / 'b.json').write_bytes(b'ORIGINAL JSON')
                    before = {p.name: p.read_bytes() for p in self.out.iterdir()}
                    operation = getattr(target, boundary)
                    interrupted = []
                    def interrupt(receiver, source, name, **kwargs):
                        operation(receiver, source, name, **kwargs)
                        if name == 'a.mid' and not interrupted:
                            interrupted.append(name)
                            raise KeyboardInterrupt('after publication, before bookkeeping')
                    with patch.object(target, boundary, interrupt), self.assertRaises(KeyboardInterrupt):
                        self.publish(replace=replace)
                    self.assertEqual(interrupted, ['a.mid'])
                    self.assertEqual({p.name: p.read_bytes() for p in self.out.iterdir()}, before)

    def test_interrupted_publication_cleanup_preserves_substituted_entry(self):
        import json
        from aimusic.audio.publication_windows import WindowsWorkspace
        publish = WindowsWorkspace.publish
        for replace in (False, True):
            with self.subTest(replace=replace):
                self.out = self.base / f'substitution-{replace}'
                self.out.mkdir()
                if replace:
                    (self.out / 'a.mid').write_bytes(b'ONLY ORIGINAL COPY')
                def interrupt(work, private, name, *, replace):
                    publish(work, private, name, replace=replace)
                    # Move our retained object without updating its cached name;
                    # a new child at that old name must never gain delete authority.
                    work.root.api.rename(work.stages[private].handle, 'moved-owned')
                    (self.out / name).write_bytes(b'UNRELATED SUBSTITUTE')
                    raise KeyboardInterrupt('after publication, before bookkeeping')
                with patch.object(WindowsWorkspace, 'publish', interrupt), self.assertRaises(BaseException) as caught:
                    self.publish(replace=replace)
                self.assertEqual((self.out / 'a.mid').read_bytes(), b'UNRELATED SUBSTITUTE')
                self.assertFalse((self.out / 'moved-owned').exists())
                if replace:
                    self.assertIsInstance(caught.exception, OSError)
                    record = next(self.out.glob('*.journal.json'))
                    self.assertIn(str(record), str(caught.exception))
                    self.assertEqual(json.loads(record.read_bytes())['phase'], 'recovery')
                    self.assertEqual(next(self.out.glob('*.backup-0')).read_bytes(), b'ONLY ORIGINAL COPY')
                else:
                    self.assertIsInstance(caught.exception, KeyboardInterrupt)
                    self.assertEqual(list(self.out.iterdir()), [self.out / 'a.mid'])

    def test_interrupted_publication_disposal_failure_preserves_recovery_journal(self):
        import json
        from aimusic.audio.publication import pin_output
        from aimusic.audio.publication_windows import WindowsWorkspace
        from aimusic.audio.windows_native import WindowsAPI
        publish, dispose = WindowsWorkspace.publish, WindowsAPI.dispose
        retained = []
        def interrupt(work, private, name, *, replace):
            publish(work, private, name, replace=replace)
            retained.append(work.stages[private].handle)
            raise KeyboardInterrupt('after publication, before bookkeeping')
        def fail(api, handle):
            if retained and handle is retained[0]:
                raise OSError('injected retained staged disposal failure')
            return dispose(api, handle)
        with patch.object(WindowsWorkspace, 'publish', interrupt), patch.object(WindowsAPI, 'dispose', fail), \
                self.assertRaises(BaseException) as caught:
            self.publish()
        self.assertIsInstance(caught.exception, OSError)
        record = next(self.out.glob('*.journal.json'))
        self.assertIn(str(record), str(caught.exception))
        journal = json.loads(record.read_bytes())
        self.assertEqual(journal['phase'], 'recovery')
        self.assertEqual(journal['published'], [])  # Coordinator never recorded the completed rename.
        self.assertEqual((self.out / 'a.mid').read_bytes(), (self.staging / 'a.mid').read_bytes())
        snapshot = {p.name: p.read_bytes() for p in self.out.iterdir()}
        with self.assertRaisesRegex(RuntimeError, 'residue'):
            with pin_output(self.out):
                self.fail('Unreconciled publication was accepted')
        self.assertEqual({p.name: p.read_bytes() for p in self.out.iterdir()}, snapshot)

    def test_late_source_hardlink_collision_leaves_source_and_intruder_untouched(self):
        from contextlib import ExitStack
        from aimusic.audio.groove.apply import apply_groove
        from aimusic.audio.publication_windows import WindowsWorkspace
        from aimusic.audio.windows_native import DELETE, READ, WindowsAPI
        from tests.test_audio_midi_events import fixture
        source, output = self.base / 'source.mid', self.out / 'grooved.mid'
        fixture().save(source)
        before = source.read_bytes()
        rename, acquire = WindowsAPI.rename, WindowsWorkspace.acquire_pin
        retained, collided = [], []
        with ExitStack() as handles:
            def prepare_competitor(work):
                link = self.out / 'competitor-link'
                link.hardlink_to(source)
                retained.append(handles.enter_context(work.root.api.open(link, READ | DELETE)))
                acquire(work)
            def collide(api, handle, name):
                if name == 'grooved.mid.json':
                    rename(api, retained[0], name)
                    collided.append(True)
                return rename(api, handle, name)
            with patch.object(WindowsWorkspace, 'acquire_pin', prepare_competitor), \
                    patch.object(WindowsAPI, 'rename', collide), self.assertRaises(FileExistsError):
                apply_groove(source, {}, output_path=output)
        self.assertEqual(collided, [True])
        self.assertEqual(source.read_bytes(), before)
        self.assertEqual(list(self.out.iterdir()), [self.out / 'grooved.mid.json'])
        self.assertTrue((self.out / 'grooved.mid.json').samefile(source))

    def test_identity_handoff_failure_has_no_public_mutation(self):
        from aimusic.audio.publication_windows import WindowsWorkspace
        from aimusic.audio.windows_native import WindowsAPI
        (self.out / 'a.mid').write_bytes(b'OLD SOURCE')
        acquire = WindowsWorkspace.acquire_pin
        def changed(work):
            acquire(work)
            raise ValueError('identity changed during handoff')
        with patch.object(WindowsWorkspace, 'acquire_pin', changed), \
                patch.object(WindowsAPI, 'rename') as rename, self.assertRaisesRegex(ValueError, 'handoff'):
            self.publish(replace=True)
        rename.assert_not_called()
        self.assertEqual({p.name: p.read_bytes() for p in self.out.iterdir()}, {'a.mid': b'OLD SOURCE'})

    def test_restore_failure_retains_only_old_copy_and_journal_and_refuses_reuse(self):
        import json
        from aimusic.audio.publication import pin_output
        from aimusic.audio.windows_native import WindowsAPI
        (self.out / 'a.mid').write_bytes(b'OLD MIDI')
        (self.out / 'b.json').write_bytes(b'OLD JSON')
        rename, calls = WindowsAPI.rename, []
        def fail(api, handle, name):
            calls.append(name)
            if name == 'b.json' and calls.count(name) == 1:
                raise OSError('publish failed')
            if name == 'a.mid' and calls.count(name) == 2:
                raise OSError('restore failed')
            return rename(api, handle, name)
        with patch.object(WindowsAPI, 'rename', fail), self.assertRaisesRegex(OSError, 'recovery') as caught:
            self.publish(replace=True)
        records = list(self.out.glob('*.journal.json'))
        self.assertEqual(len(records), 1)
        self.assertIn(str(records[0]), str(caught.exception))
        self.assertEqual(json.loads(records[0].read_bytes())['phase'], 'recovery')
        self.assertIn(b'OLD MIDI', [p.read_bytes() for p in self.out.iterdir()])
        snapshot = {p.name: p.read_bytes() for p in self.out.iterdir()}
        with self.assertRaisesRegex(RuntimeError, 'residue'):
            with pin_output(self.out):
                self.fail('Residue was ignored')
        self.assertEqual({p.name: p.read_bytes() for p in self.out.iterdir()}, snapshot)

    def test_pins_block_directory_write_delete_and_retained_leaf_mutation_not_child_creation(self):
        from aimusic.audio.publication_windows import WindowsWorkspace
        from aimusic.audio.windows_native import DELETE, DIRECTORY_ACCESS, SHARE_ALL, WRITE
        acquire, checked = WindowsWorkspace.acquire_pin, []
        def probe(work):
            acquire(work)
            for access in (WRITE, DELETE):
                with self.assertRaises(PermissionError):
                    with work.root.api.open(work.root.path, access | DIRECTORY_ACCESS, share=SHARE_ALL):
                        self.fail('Directory mutation handle was admitted')
            for leaf in work.leaves:
                with self.assertRaises(PermissionError):
                    (work.root.path / leaf.name).write_bytes(b'INTRUDER')
            (work.root.path / 'unrelated').write_bytes(b'ALLOWED CHILD')
            checked.append(True)
        with patch.object(WindowsWorkspace, 'acquire_pin', probe):
            self.publish()
        self.assertEqual(checked, [True])
        self.assertEqual((self.out / 'unrelated').read_bytes(), b'ALLOWED CHILD')

    def test_unsupported_filesystem_and_reparse_attributes_fail_closed(self):
        from dataclasses import replace
        from aimusic.audio.publication import pin_output
        from aimusic.audio.windows_native import WindowsAPI
        info = WindowsAPI.information
        def reparse(api, handle):
            actual = info(api, handle)
            return replace(actual, attributes=actual.attributes | 0x400)
        with patch.object(WindowsAPI, 'information', reparse), self.assertRaisesRegex(ValueError, 'reparse'):
            with pin_output(self.out):
                self.fail('Reparse directory admitted')
        with patch.object(WindowsAPI, 'require_local_ntfs', side_effect=NotImplementedError('not local NTFS')), \
                self.assertRaisesRegex(RuntimeError, 'supported local filesystem'):
            with pin_output(self.out):
                self.fail('Unsupported volume admitted')
        self.assertEqual(list(self.out.iterdir()), [])

    def test_all_existing_entry_kinds_are_preserved(self):
        from aimusic.audio.publication import pin_output
        for kind in ('file', 'directory', 'hardlink', 'casefold'):
            with self.subTest(kind=kind):
                out = self.base / kind
                out.mkdir()
                target = out / ('A.MID' if kind == 'casefold' else 'a.mid')
                if kind == 'directory':
                    target.mkdir()
                elif kind == 'hardlink':
                    target.hardlink_to(self.staging / 'a.mid')
                else:
                    target.write_bytes(b'EXISTING')
                before = target.stat()
                with self.assertRaises(FileExistsError):
                    with pin_output(out) as root:
                        root.publish(self.staging, ['a.mid'])
                self.assertEqual(target.stat().st_ino, before.st_ino)
                self.assertEqual(list(out.iterdir()), [target])

    def test_casefold_race_after_rename_rolls_back_owned_not_alias(self):
        from aimusic.audio.publication import pin_output
        from aimusic.audio.windows_native import WindowsAPI
        name = 'stra\u00dfe.wav'
        (self.staging / name).write_bytes(b'OUR BYTES')
        rename = WindowsAPI.rename
        def collide(api, handle, target):
            result = rename(api, handle, target)
            if target == name:
                (self.out / 'STRASSE.wav').write_bytes(b'OTHER WRITER')
            return result
        with patch.object(WindowsAPI, 'rename', collide), self.assertRaises(FileExistsError):
            with pin_output(self.out) as root:
                root.publish(self.staging, [name])
        self.assertEqual({p.name: p.read_bytes() for p in self.out.iterdir()}, {'STRASSE.wav': b'OTHER WRITER'})

    def test_no_path_mutation_fallback_and_all_payloads_flushed_before_first_public_rename(self):
        from aimusic.audio.windows_native import WindowsAPI
        flush, rename, flushed = WindowsAPI.flush, WindowsAPI.rename, set()
        def flushing(api, handle):
            result = flush(api, handle)
            flushed.add(handle.value)
            return result
        def renaming(api, handle, name):
            if name in ('a.mid', 'b.json'):
                self.assertIn(handle.value, flushed)
                self.assertGreaterEqual(len(flushed), 3)  # Both payloads plus journal.
            return rename(api, handle, name)
        with patch.object(WindowsAPI, 'flush', flushing), patch.object(WindowsAPI, 'rename', renaming), \
                patch('os.link', side_effect=AssertionError('unsafe link')), \
                patch('os.replace', side_effect=AssertionError('unsafe replace')), \
                patch('os.rename', side_effect=AssertionError('unsafe rename')):
            self.publish()

    def test_demo_groove_preview_stems_final_mix_and_cold_warm_cache(self):
        from aimusic.audio.demo_sources import write_source
        from aimusic.audio.groove.apply import apply_groove
        from aimusic.audio.preview import render_midi_preview
        from aimusic.audio.orchestrator import Orchestrator
        from aimusic.audio.render.stems import rough_mix, copy_stem_artifacts
        source = write_source('fixture-12', self.base / 'source caf\u00e9')
        original = source.read_bytes()
        groove = self.base / 'groove' / 'expressive.mid'
        apply_groove(source, {}, output_path=groove)
        apply_groove(source, {}, output_path=groove)  # Replacement, no O_DIRECTORY.
        preview = render_midi_preview(groove, self.base / 'preview' / 'caf\u00e9.wav')
        self.assertGreater(preview.stat().st_size, 44)
        cfg = {'stages': {'analysis': False, 'expressivization': False, 'render': True},
               'render': {'sample_rate': 8000}}
        for name in ('cold', 'warm'):
            orch = Orchestrator(cfg, self.base / 'cache')
            orch.run(source, output_dir=self.base / name)
            self.assertTrue((self.base / name / 'stems' / 'rough_mix.wav').is_file())
            if name == 'warm':
                self.assertEqual(orch.cache_hits['render'], 1)
        mix = self.base / 'cold' / 'stems' / 'rough_mix.wav'
        final = rough_mix([mix], output_path=self.base / 'final' / 'final.wav')
        self.assertGreater(final.stat().st_size, 44)
        copy_stem_artifacts(self.base / 'cold' / 'stems', self.base / 'restored')
        self.assertEqual((self.base / 'restored' / 'rough_mix.wav').read_bytes(), mix.read_bytes())
        self.assertEqual(source.read_bytes(), original)

    def test_junction_between_preflight_and_first_retain_never_receives_content(self):
        self.require_capability('junction')
        from aimusic.audio.publication_windows import WindowsWorkspace
        from aimusic.audio.windows_native import WindowsAPI
        target = self.base / 'junction-disclosure-target'
        target.mkdir()
        (target / 'KEEP').write_bytes(b'UNRELATED')
        retain, write = WindowsWorkspace.retain, WindowsAPI.write
        installed, disclosed = [], []
        def redirect(work, name, **kwargs):
            if not installed:
                self._install_junction(work.root.path, target)
                installed.append(True)
            return retain(work, name, **kwargs)
        def observe(api, handle, data):
            write(api, handle, data)
            # Observe every actual write, not just the post-rollback snapshot:
            # deleting a leaked payload/journal cannot undo disclosure.
            disclosed.extend((p.name, p.stat().st_size) for p in target.iterdir()
                             if p.name != 'KEEP' and p.stat().st_size)
        try:
            with patch.object(WindowsWorkspace, 'retain', redirect), patch.object(WindowsAPI, 'write', observe), \
                    self.assertRaisesRegex(ValueError, 'reparse'):
                self.publish()
            self.assertEqual(installed, [True])
            self.assertEqual(disclosed, [], 'Payload or journal content reached the junction target')
            self.assertEqual({p.name: p.read_bytes() for p in target.iterdir()}, {'KEEP': b'UNRELATED'})
        finally:
            if installed:
                self.out.rmdir()  # Remove our junction only, never its target.

    def test_failed_handoff_cleanup_keeps_only_empty_owned_recovery_record(self):
        self.require_capability('junction')
        from aimusic.audio.publication_windows import WindowsWorkspace
        from aimusic.audio.windows_native import WindowsAPI
        target = self.base / 'junction-empty-recovery-target'
        target.mkdir()
        (target / 'KEEP').write_bytes(b'UNRELATED')
        retain, dispose = WindowsWorkspace.retain, WindowsAPI.dispose
        installed, failed = [], []
        def redirect(work, name, **kwargs):
            if not installed:
                self._install_junction(work.root.path, target)
                installed.append(True)
            return retain(work, name, **kwargs)
        def interrupt_cleanup(api, handle):
            if not failed:
                failed.append(True)
                raise OSError('injected empty reservation disposal failure')
            return dispose(api, handle)
        try:
            with patch.object(WindowsWorkspace, 'retain', redirect), \
                    patch.object(WindowsAPI, 'dispose', interrupt_cleanup), \
                    patch.object(WindowsAPI, 'write', side_effect=AssertionError('pre-handoff content write')), \
                    self.assertRaisesRegex(OSError, 'recovery') as caught:
                self.publish()
            self.assertEqual(installed, [True])
            self.assertEqual(failed, [True])
            record = next(target.glob('*.journal.json'))
            self.assertIn(str(self.out / record.name), str(caught.exception))
            self.assertEqual({p.name: p.read_bytes() for p in target.iterdir()}, {'KEEP': b'UNRELATED', record.name: b''})
        finally:
            if installed:
                self.out.rmdir()

    def test_private_content_is_written_only_after_verified_empty_handle_handoff(self):
        from aimusic.audio.publication import pin_output
        from aimusic.audio.publication_windows import WindowsWorkspace
        from aimusic.audio.windows_native import WindowsAPI, WRITE
        acquire, write, opening = WindowsWorkspace.acquire_pin, WindowsAPI.write, WindowsAPI.open
        for fail in (False, True):
            with self.subTest(rollback=fail):
                self.out = self.base / f'content-handoff-{fail}'
                self.out.mkdir()
                retained, ready, writes = set(), [], []
                def handoff(work):
                    self.assertEqual(len(work.leaves), 3)  # Journal plus both payloads.
                    self.assertEqual([work.root.api.information(leaf.handle).size for leaf in work.leaves], [0, 0, 0])
                    retained.update(leaf.handle.value for leaf in work.leaves)
                    acquire(work)
                    ready.append(True)
                def observe(api, handle, data):
                    self.assertEqual(ready, [True], 'Content write preceded verified handoff')
                    self.assertIn(handle.value, retained, 'Content did not use a reserved handle')
                    writes.append(handle.value)
                    return write(api, handle, data)
                def no_reopen(api, path, access, **kwargs):
                    if ready and Path(path).parent == self.out:
                        self.assertFalse(access & WRITE, 'Reopened a destination path for writing after handoff')
                    return opening(api, path, access, **kwargs)
                with patch.object(WindowsWorkspace, 'acquire_pin', handoff), \
                        patch.object(WindowsAPI, 'write', observe), patch.object(WindowsAPI, 'open', no_reopen):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, 'caller failed'):
                            with pin_output(self.out) as root:
                                with root.publication(self.staging, ['a.mid', 'b.json']):
                                    raise RuntimeError('caller failed')
                    else:
                        self.publish()
                self.assertEqual(set(writes), retained)
                self.assertEqual(sorted(p.name for p in self.out.iterdir()), [] if fail else ['a.mid', 'b.json'])

    def test_private_reservation_information_failure_disposes_uninspected_handle(self):
        from aimusic.audio.windows_native import WindowsAPI
        opening, information = WindowsAPI.open, WindowsAPI.information
        for suffix in ('journal.json', '.stems-0'):
            with self.subTest(reservation=suffix):
                self.out = self.base / f'inspection-failure-{suffix}'
                self.out.mkdir()
                (self.out / 'a.mid').write_bytes(b'ORIGINAL MIDI')
                (self.out / 'keep').write_bytes(b'UNRELATED')
                before = {p.name: p.read_bytes() for p in self.out.iterdir()}
                retained = []
                def capture(api, path, access, **kwargs):
                    handle = opening(api, path, access, **kwargs)
                    if kwargs.get('create') and Path(path).name.endswith(suffix):
                        retained.append(handle)
                    return handle
                def fail(api, handle):
                    if retained and handle is retained[0]:
                        raise OSError('injected reservation information failure')
                    return information(api, handle)
                with patch.object(WindowsAPI, 'open', capture), patch.object(WindowsAPI, 'information', fail), \
                        patch.object(WindowsAPI, 'write', side_effect=AssertionError('pre-handoff content write')), \
                        patch('os.unlink', side_effect=AssertionError('path deletion fallback')), \
                        self.assertRaisesRegex(OSError, 'injected reservation information failure'):
                    self.publish(replace=True)
                self.assertEqual(len(retained), 1)
                self.assertTrue(retained[0].closed)
                self.assertEqual({p.name: p.read_bytes() for p in self.out.iterdir()}, before)

    def test_uninspected_payload_disposal_failure_retains_actionable_recovery_evidence(self):
        from aimusic.audio.publication import pin_output
        from aimusic.audio.windows_native import WindowsAPI
        opening, information, dispose = WindowsAPI.open, WindowsAPI.information, WindowsAPI.dispose
        (self.out / 'keep').write_bytes(b'UNRELATED')
        retained, reserved = [], []
        def capture(api, path, access, **kwargs):
            handle = opening(api, path, access, **kwargs)
            if kwargs.get('create') and Path(path).name.endswith('.stems-0'):
                retained.append(handle)
                reserved.append(Path(path))
            return handle
        def fail_information(api, handle):
            if retained and handle is retained[0]:
                raise OSError('injected reservation information failure')
            return information(api, handle)
        def fail_dispose(api, handle):
            if retained and handle is retained[0]:
                raise OSError('injected uninspected reservation disposal failure')
            return dispose(api, handle)
        with patch.object(WindowsAPI, 'open', capture), \
                patch.object(WindowsAPI, 'information', fail_information), \
                patch.object(WindowsAPI, 'dispose', fail_dispose), \
                patch.object(WindowsAPI, 'write', side_effect=AssertionError('pre-handoff content write')), \
                patch('os.unlink', side_effect=AssertionError('path deletion fallback')), \
                self.assertRaisesRegex(OSError, 'recovery') as caught:
            self.publish()
        self.assertEqual(len(retained), 1)
        self.assertTrue(retained[0].closed)
        record = next(self.out.glob('*.journal.json'))
        self.assertIn(str(record), str(caught.exception))
        self.assertIn(str(reserved[0]), str(caught.exception.__cause__))
        self.assertIn('uninspected', str(caught.exception.__cause__))
        snapshot = {p.name: p.read_bytes() for p in self.out.iterdir()}
        self.assertEqual(snapshot, {'keep': b'UNRELATED', record.name: b'', reserved[0].name: b''})
        with self.assertRaisesRegex(RuntimeError, 'residue'):
            with pin_output(self.out):
                self.fail('Uninspected reservation recovery evidence was ignored')
        self.assertEqual({p.name: p.read_bytes() for p in self.out.iterdir()}, snapshot)

    def test_private_reservation_collision_never_writes_or_removes_competing_leaf(self):
        from aimusic.audio.publication_windows import WindowsWorkspace
        from aimusic.audio.windows_native import WindowsAPI
        retain = WindowsWorkspace.retain
        source = self.staging / 'a.mid'
        before = source.read_bytes()
        for suffix in ('journal.json', '.stems-1'):
            with self.subTest(collision=suffix):
                self.out = self.base / f'private-collision-{suffix}'
                self.out.mkdir()
                collision = []
                def collide(work, name, **kwargs):
                    if name.endswith(suffix) and not collision:
                        intruder = self.out / name
                        intruder.hardlink_to(source)
                        collision.append(intruder)
                    return retain(work, name, **kwargs)
                with patch.object(WindowsWorkspace, 'retain', collide), \
                        patch.object(WindowsAPI, 'write', side_effect=AssertionError('pre-handoff content write')), \
                        self.assertRaises(FileExistsError):
                    self.publish()
                self.assertEqual(list(self.out.iterdir()), collision)
                self.assertEqual(len(collision), 1)
                self.assertTrue(collision[0].samefile(source))
                self.assertEqual(source.read_bytes(), before)

    def _install_junction(self, link, target):
        from tests.windows_capabilities import install_junction
        install_junction(link, target)

    def _check_link_destinations(self, kind):
        from aimusic.audio.publication import pin_output
        target = self.base / (kind + '-target')
        target.mkdir()
        (target / 'KEEP').write_bytes(b'UNRELATED')
        for position in ('root', 'ancestor', 'leaf'):
            with self.subTest(position=position):
                link = self.base / (kind + '-' + position)
                if kind == 'junction':
                    link.mkdir()
                    self._install_junction(link, target)
                else:
                    link.symlink_to(target, target_is_directory=True)
                try:
                    output = link if position == 'root' else link / 'new-output'
                    if position == 'leaf':
                        with pin_output(self.base) as root, self.assertRaises(FileExistsError):
                            root.check_new_files([link.name])
                    else:
                        with self.assertRaisesRegex(ValueError, 'reparse'):
                            with pin_output(output):
                                self.fail('Reparse component was admitted')
                    self.assertEqual({p.name: p.read_bytes() for p in target.iterdir()}, {'KEEP': b'UNRELATED'})
                finally:
                    link.rmdir()

    def test_native_junction_destinations_are_refused_without_touching_target(self):
        self.require_capability('junction')
        self._check_link_destinations('junction')

    def test_native_symlink_destinations_are_refused_without_touching_target(self):
        self.require_capability('symlink')
        self._check_link_destinations('symlink')


class TestWindowsBackendPolicy(unittest.TestCase):
    def workspace(self):
        from aimusic.audio.publication_records import FileIdentity
        from aimusic.audio.publication_windows import WindowsWorkspace
        from aimusic.audio.windows_native import Handle, Information
        api = Mock()
        identity = FileIdentity(7, 9)
        info = Information(identity, 0x10, 0, 1)
        api.open.return_value = Handle(api, 2 ** 33)
        api.information.return_value = info
        root = SimpleNamespace(api=api, path=Path('output'), identity=identity, verify=Mock(), info=Mock())
        return WindowsWorkspace(root, '.stems-invocation'), info

    def test_changed_identity_and_reparse_at_handoff_never_mutate_public_names(self):
        from dataclasses import replace
        from aimusic.audio.publication_records import FileIdentity
        for changed in ('identity', 'reparse'):
            with self.subTest(changed=changed):
                work, info = self.workspace()
                work.root.api.information.return_value = (replace(info, identity=FileIdentity(7, 10))
                                                          if changed == 'identity' else replace(info, attributes=0x410))
                with self.assertRaises(ValueError):
                    work.acquire_pin()
                self.assertFalse(work.ready)
                work.journal(b'{"phase": "recovery"}')
                with self.assertRaisesRegex(ValueError, 'pin'):
                    work.stage(Path('source'), '.stems-0')
                from aimusic.audio.publication_windows import Leaf
                with self.assertRaisesRegex(ValueError, 'pin'):
                    work.move(Leaf('.stems-0', work.pin, info.identity), 'public', public=True)
                work.root.api.write.assert_not_called()
                work.root.api.rewrite.assert_not_called()
                work.root.api.rename.assert_not_called()
                work.pin.close()
                work.root.api.kernel.CloseHandle.assert_called_once_with(2 ** 33)

    def test_mutation_before_pin_and_wrong_retained_identity_are_refused(self):
        from aimusic.audio.publication_records import FileIdentity
        from aimusic.audio.publication_windows import Leaf
        from aimusic.audio.windows_native import Handle, Information
        work, info = self.workspace()
        leaf = Leaf('.stems-ours', Handle(work.root.api, 10), FileIdentity(7, 20))
        with self.assertRaisesRegex(ValueError, 'pin'):
            work.move(leaf, 'public', public=True)
        work.root.api.information.return_value = Information(FileIdentity(7, 21), 0, 1, 1)
        with self.assertRaisesRegex(ValueError, 'identity'):
            work.remove(leaf)
        work.root.api.dispose.assert_not_called()

    def test_cleanup_uses_retained_identity_not_substituted_path(self):
        from aimusic.audio.publication_records import FileIdentity
        from aimusic.audio.publication_windows import Leaf
        from aimusic.audio.windows_native import Handle, Information
        work, info = self.workspace()
        identity = FileIdentity(7, 20)
        leaf = Leaf('.stems-ours', Handle(work.root.api, 10), identity)
        work.root.api.information.return_value = Information(identity, 0, 1, 1)
        work.root.info.return_value = Information(FileIdentity(7, 99), 0, 1, 1).stat()
        work.remove(leaf)
        work.root.api.dispose.assert_called_once_with(leaf.handle)
        work.root.api.kernel.CloseHandle.assert_called_once_with(10)
        work.root.info.assert_not_called()  # No path-based delete authority.
