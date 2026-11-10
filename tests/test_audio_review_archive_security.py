"""Task 6.5 security review: verified views and backend-copy fault boundaries."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import os
import stat
import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from aimusic.audio.publication_records import FileIdentity
from aimusic.audio.review_bundle import canonical_bytes
from tests.test_audio_review_archive import ArchiveCase


class TestVerifiedExtraction(ArchiveCase):
    def setUp(self):
        super().setUp()
        self.sealed = self.assemble()
        self.out = self.base / 'view'
        self.module.extract_archive(self.sealed.data, self.out, expected_sha256=self.sealed.sha256)

    def verify(self, **kwargs):
        return self.module.verify_extracted(self.sealed.data, self.out,
                                            expected_sha256=self.sealed.sha256, **kwargs)

    def test_valid_view_requires_independently_trusted_archive_and_is_read_only(self):
        before = {p.relative_to(self.out): p.read_bytes() for p in self.out.rglob('*') if p.is_file()}
        self.assertEqual(self.verify(), self.out)
        self.assertEqual(before, {p.relative_to(self.out): p.read_bytes() for p in self.out.rglob('*') if p.is_file()})
        with self.assertRaises(ValueError):
            self.module.verify_extracted(self.sealed.data, self.out, expected_sha256='0' * 64)
        with self.assertRaises(ValueError):
            self.module.verify_extracted(self.sealed.data[:-1], self.out, expected_sha256=self.sealed.sha256)
        absent = self.base / 'absent'
        with self.assertRaises((ValueError, OSError)):
            self.module.verify_extracted(self.sealed.data, absent, expected_sha256=self.sealed.sha256)
        self.assertFalse(absent.exists())
        with self.assertRaises(TypeError):
            self.module.verify_extracted(self.sealed.data, self.out)
        archive = self.base / 'source.zip'
        archive.write_bytes(self.sealed.data)
        self.assertEqual(self.module.verify_extracted(archive, self.out,
                                                     expected_sha256=self.sealed.sha256), self.out)
        with patch.object(self.module.os, 'scandir', side_effect=AssertionError('view read before archive check')):
            with self.assertRaises(ValueError):
                self.verify(limits=replace(self.module.ArchiveLimits(), max_members=1))

    def test_exact_marker_does_not_certify_corrupted_missing_or_extra_members(self):
        marker = (self.out / 'complete.json').read_bytes()
        for name, data in self.module.verify_archive(self.sealed.data, expected_sha256=self.sealed.sha256).members:
            path = self.out / name
            with self.subTest(name=name, fault='corrupt'):
                path.write_bytes(b'corrupt')
                with self.assertRaises(ValueError):
                    self.verify()
                path.write_bytes(data)
            with self.subTest(name=name, fault='missing'):
                path.unlink()
                with self.assertRaises((ValueError, OSError)):
                    self.verify()
                path.write_bytes(data)
        for extra in ('unlisted', 'media/unlisted.wav', 'media/empty-dir', 'private', '.stems-residue'):
            path = self.out / extra
            with self.subTest(extra=extra):
                if extra in ('private', 'media/empty-dir'):
                    path.mkdir()
                else:
                    path.write_bytes(b'extra')
                with self.assertRaises(ValueError):
                    self.verify()
                path.rmdir() if path.is_dir() else path.unlink()
        self.assertEqual((self.out / 'complete.json').read_bytes(), marker)

    def test_completion_record_requires_exact_canonical_bytes(self):
        marker = self.out / 'complete.json'
        expected = marker.read_bytes()
        for bad in (b'', expected + b'\n', expected[:-1],
                    canonical_bytes({'schema': 'aimusic.review-extraction/1', 'archive_sha256': '0' * 64}),
                    b'{"schema":"aimusic.review-extraction/1","schema":"aimusic.review-extraction/1"}'):
            marker.write_bytes(bad)
            with self.assertRaises(ValueError):
                self.verify()
        marker.unlink()
        with self.assertRaises((ValueError, OSError)):
            self.verify()

    def test_inventory_scan_stops_on_first_extra_without_recursion(self):
        from types import SimpleNamespace

        @contextmanager
        def entries(directory):
            def generate():
                yield SimpleNamespace(name='unexpected')
                raise AssertionError('unbounded or recursive enumeration')
            yield generate()

        with patch.object(self.module.os, 'scandir', entries), self.assertRaisesRegex(ValueError, 'inventory'):
            self.verify()

    def test_expected_files_reject_directory_substitutes_and_oversize_before_read(self):
        for name in ('index.html', self.assignment.stimuli[0].media.path, 'complete.json'):
            path = self.out / name
            original = path.read_bytes()
            with self.subTest(name=name):
                path.unlink()
                path.mkdir()
                with self.assertRaises(ValueError):
                    self.verify()
                path.rmdir()
                path.write_bytes(original + b'oversized')
                real = self.module.os.open

                def open_file(target, *args, **kwargs):
                    if Path(target) == path:
                        raise AssertionError('oversized member opened')
                    return real(target, *args, **kwargs)

                with patch.object(self.module.os, 'open', open_file), self.assertRaisesRegex(ValueError, 'budget'):
                    self.verify()
                path.write_bytes(original)

    def test_view_rejects_reparse_directory_and_does_not_follow_extra_links(self):
        from types import SimpleNamespace
        real = Path.lstat

        def reparse(path, *args, **kwargs):
            info = real(path, *args, **kwargs)
            if path == target:
                return SimpleNamespace(st_mode=info.st_mode, st_file_attributes=0x400)
            return info

        for target in (self.base, self.out, self.out / 'media', self.out / 'index.html',
                       self.out / 'complete.json', self.out / self.assignment.stimuli[0].media.path):
            with self.subTest(target=target):
                with patch.object(Path, 'lstat', reparse), self.assertRaises(ValueError):
                    self.verify()
        outside = self.base / 'outside'
        outside.mkdir()
        (outside / 'sentinel').write_bytes(b'outside')
        try:
            (self.out / 'extra-link').symlink_to(outside, target_is_directory=True)
        except OSError:
            return  # Mandatory reparse decision above; native links may need privilege.
        with self.assertRaises(ValueError):
            self.verify()
        self.assertEqual((outside / 'sentinel').read_bytes(), b'outside')


class TestBackendCopyIntegrity(ArchiveCase):
    def test_corrupt_second_copy_rolls_back_public_archive_and_preserves_sentinel(self):
        if os.name == 'nt':
            from aimusic.audio.publication_windows import WindowsWorkspace as Workspace
        else:
            from aimusic.audio.publication_posix import PosixWorkspace as Workspace
        real = Workspace.stage
        faults = []

        def corrupt(work, source, name):
            identity = real(work, source, name)
            faults.append(source.name)
            # This is the backend/private copy, after external staging readback.
            data = source.read_bytes()
            bad = bytes([data[0] ^ 1]) + data[1:]
            if os.name == 'nt':
                work.root.api.rewrite(work.stages[name].handle, bad)
            else:
                with os.fdopen(os.open(name, os.O_WRONLY | os.O_TRUNC, dir_fd=work.fd), 'wb') as stream:
                    stream.write(bad)
            return identity

        (self.base / 'sentinel').write_bytes(b'preserve')
        out = self.base / 'review.zip'
        with patch.object(Workspace, 'stage', corrupt), self.assertRaisesRegex(ValueError, 'published'):
            self.module.write_review_archive(self.root, out, bundle=self.bundle, assignment=self.assignment,
                                             expected_bundle_sha256=self.bundle.sha256,
                                             expected_assignment_sha256=self.assignment.sha256)
        self.assertEqual(faults, ['review.zip'])
        self.assertFalse(out.exists())
        self.assertEqual({p.name for p in self.base.iterdir()}, {'inputs', 'sentinel'})
        self.assertEqual((self.base / 'sentinel').read_bytes(), b'preserve')

    def test_backend_corruption_in_each_extraction_phase_cannot_leave_completion(self):
        if os.name == 'nt':
            from aimusic.audio.publication_windows import WindowsWorkspace as Workspace
        else:
            from aimusic.audio.publication_posix import PosixWorkspace as Workspace
        real = Workspace.stage
        sealed = self.assemble()
        expected = dict(self.module.verify_archive(sealed.data, expected_sha256=sealed.sha256).members)
        for index, target in enumerate(('index.html', Path(self.assignment.stimuli[0].media.path).name, 'complete.json')):
            out = self.base / f'view-{index}'
            faults = []

            def corrupt(work, source, name):
                identity = real(work, source, name)
                if source.name == target:
                    faults.append(target)
                    data = source.read_bytes()
                    bad = bytes([data[0] ^ 1]) + data[1:]
                    if os.name == 'nt':
                        work.root.api.rewrite(work.stages[name].handle, bad)
                    else:
                        with os.fdopen(os.open(name, os.O_WRONLY | os.O_TRUNC, dir_fd=work.fd), 'wb') as stream:
                            stream.write(bad)
                return identity

            with self.subTest(target=target):
                with patch.object(Workspace, 'stage', corrupt), self.assertRaisesRegex(ValueError, 'published'):
                    self.module.extract_archive(sealed.data, out, expected_sha256=sealed.sha256)
                self.assertEqual(faults, [target])
                self.assertFalse((out / 'complete.json').exists())
                self.assertFalse(any(p.name.startswith('.stems-') for p in out.rglob('*')))
                for path in out.rglob('*'):
                    if path.is_file():
                        self.assertEqual(path.read_bytes(), expected[path.relative_to(out).as_posix()])
                with self.assertRaises((ValueError, OSError)):
                    self.module.verify_extracted(sealed.data, out, expected_sha256=sealed.sha256)


@unittest.skipIf(os.name == 'nt', 'requires retained POSIX directory descriptors')
class TestPosixPublicReadback(unittest.TestCase):
    def setUp(self):
        from aimusic.audio import review_archive
        self.module = review_archive
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        self.addCleanup(os.close, self.fd)

    def test_readonly_descriptor_relative_nofollow_bounded_short_reads_and_close(self):
        (self.root / 'public.zip').write_bytes(b'abcdef')
        identity = FileIdentity.from_stat((self.root / 'public.zip').stat())
        read = os.read
        with patch('os.open', wraps=os.open) as opened, patch('os.close', wraps=os.close) as closed:
            with patch('os.read', side_effect=lambda fd, size: read(fd, min(2, size))) as reads:
                self.module._verify_posix_published(self.fd, 'public.zip', b'abcdef',
                                                    expected_identity=identity)
        opened.assert_called_once_with('public.zip', os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                                       dir_fd=self.fd)
        leaf = closed.call_args.args[0]
        self.assertNotEqual(leaf, self.fd)
        self.assertEqual([c.args for c in reads.call_args_list], [(leaf, n) for n in (7, 5, 3, 1)])
        closed.assert_called_once_with(leaf)
        with self.assertRaises(OSError):
            os.fstat(leaf)
        self.assertTrue(stat.S_ISDIR(os.fstat(self.fd).st_mode))

    def test_opened_fd_identity_not_path_identity_is_required_before_size_or_read(self):
        public, decoy = self.root / 'public.zip', self.root / 'decoy.zip'
        public.write_bytes(b'abc')
        identity = FileIdentity.from_stat(public.stat())
        real_open = os.open

        def open_decoy(name, flags, **kwargs):
            # Simulate substitution at open, then restoration before fstat.
            # The path identifies the owned file, but the fd identifies the decoy.
            self.assertEqual(name, 'public.zip')
            return real_open('decoy.zip', flags, **kwargs)

        for data in (b'abc', b'wrong size'):
            decoy.write_bytes(data)
            with self.subTest(data=data), patch('os.open', open_decoy), patch('os.close', wraps=os.close) as closed:
                with patch('os.read', side_effect=AssertionError('unowned leaf read')):
                    with self.assertRaisesRegex(ValueError, 'published.*identity'):
                        self.module._verify_posix_published(self.fd, 'public.zip', b'abc', expected_identity=identity)
                closed.assert_called_once()
                with self.assertRaises(OSError):
                    os.fstat(closed.call_args.args[0])
            self.assertEqual(FileIdentity.from_stat(public.stat()), identity)

    def test_identity_compares_both_device_and_inode_before_read(self):
        public = self.root / 'public.zip'
        public.write_bytes(b'abc')
        identity = FileIdentity.from_stat(public.stat())
        for wrong in (replace(identity, device=identity.device + 1), replace(identity, inode=identity.inode + 1)):
            with self.subTest(identity=wrong), patch('os.read', side_effect=AssertionError('unowned leaf read')):
                with self.assertRaisesRegex(ValueError, 'published.*identity'):
                    self.module._verify_posix_published(self.fd, 'public.zip', b'abc', expected_identity=wrong)

    def test_regular_file_and_exact_size_required_before_any_read(self):
        (self.root / 'directory').mkdir()
        os.mkfifo(self.root / 'fifo')
        (self.root / 'short').write_bytes(b'ab')
        (self.root / 'long').write_bytes(b'abcd')
        (self.root / 'link').symlink_to(self.root / 'short')
        # Match special entries' sizes so size rejection cannot mask a missing
        # regular-file check. A FIFO's matching expected payload is empty.
        for name, expected in (('directory', bytes((self.root / 'directory').stat().st_size)),
                               ('fifo', b''), ('short', b'abc'), ('long', b'abc'), ('link', b'ab')):
            identity = FileIdentity.from_stat((self.root / name).lstat())
            with self.subTest(name=name), patch('os.read', side_effect=AssertionError('invalid leaf read')):
                with self.assertRaises((ValueError, OSError)):
                    self.module._verify_posix_published(self.fd, name, expected,
                                                        expected_identity=identity)

    def test_corruption_truncation_growth_and_io_error_close_leaf(self):
        (self.root / 'public.zip').write_bytes(b'abc')
        identity = FileIdentity.from_stat((self.root / 'public.zip').stat())
        for chunks in ([b'bad'], [b'ab', b''], [b'abcX'], [b'abc', b'X'], [OSError('read fault')]):
            with self.subTest(chunks=chunks), patch('os.close', wraps=os.close) as closed:
                with patch('os.read', side_effect=chunks), self.assertRaises((ValueError, OSError)):
                    self.module._verify_posix_published(self.fd, 'public.zip', b'abc',
                                                        expected_identity=identity)
                closed.assert_called_once()
                with self.assertRaises(OSError):
                    os.fstat(closed.call_args.args[0])

    def test_empty_file_and_large_file_reads_have_fixed_ceiling(self):
        for data in (b'', b'x' * (2 * 1024 * 1024 + 7)):
            (self.root / 'public.zip').write_bytes(data)
            identity = FileIdentity.from_stat((self.root / 'public.zip').stat())
            with patch('os.read', wraps=os.read) as reads:
                self.module._verify_posix_published(self.fd, 'public.zip', data,
                                                    expected_identity=identity)
            sizes = [c.args[1] for c in reads.call_args_list]
            self.assertTrue(all(0 < n <= 1024 * 1024 for n in sizes))
            self.assertEqual(sizes[-1], 1)

    def test_only_a_single_validated_leaf_can_be_opened(self):
        for name in ('../outside', '/absolute', 'nested/file', '.', ''):
            with self.subTest(name=name), patch('os.open', side_effect=AssertionError('unsafe open')):
                with self.assertRaises(ValueError):
                    self.module._verify_posix_published(self.fd, name, b'abc', expected_identity=FileIdentity(1, 2))


@unittest.skipIf(os.name == 'nt', 'requires retained POSIX directory descriptors')
class TestPosixPinnedArchiveIntegrity(ArchiveCase):
    def test_corrupt_copy_with_temporary_leaf_substitution_and_restore_rolls_back_owned_archive(self):
        from aimusic.audio.publication_posix import PosixWorkspace
        parent = self.base / 'output'
        parent.mkdir()
        public, retained, decoy = (parent / n for n in ('review.zip', 'retained.zip', 'decoy.zip'))
        sealed = self.assemble()
        decoy.write_bytes(sealed.data)
        decoy_identity = FileIdentity.from_stat(decoy.stat())
        (parent / 'sentinel').write_bytes(b'preserve unrelated entry')
        stage, verify = PosixWorkspace.stage, self.module._verify_posix_published
        staged, swaps = [], []

        def corrupt(work, source, name):
            identity = stage(work, source, name)
            data = source.read_bytes()
            with os.fdopen(os.open(name, os.O_WRONLY | os.O_TRUNC, dir_fd=work.fd), 'wb') as stream:
                stream.write(bytes([data[0] ^ 1]) + data[1:])
            staged.append(identity)
            return identity

        def substitute(*args, **kwargs):
            self.assertEqual(FileIdentity.from_stat(public.stat()), staged[0])
            self.assertNotEqual(public.read_bytes(), sealed.data)
            public.rename(retained)
            decoy.rename(public)
            try:
                self.assertNotEqual(FileIdentity.from_stat(public.stat()), staged[0])
                self.assertEqual(public.read_bytes(), sealed.data)
                return verify(*args, **kwargs)
            finally:
                public.rename(decoy)
                retained.rename(public)  # Restore before rollback OR the commit identity post-check.
                swaps.append('restored')
                self.assertEqual(FileIdentity.from_stat(public.stat()), staged[0])

        with patch.object(PosixWorkspace, 'stage', corrupt):
            with patch.object(self.module, '_verify_posix_published', substitute):
                with self.assertRaisesRegex(ValueError, 'published.*identity'):
                    self.module.write_review_archive(self.root, public, bundle=self.bundle, assignment=self.assignment,
                                                     expected_bundle_sha256=self.bundle.sha256,
                                                     expected_assignment_sha256=self.assignment.sha256)
        self.assertEqual(len(staged), 1)
        self.assertEqual(swaps, ['restored'])
        self.assertEqual({p.name for p in parent.iterdir()}, {'sentinel', 'decoy.zip'})
        self.assertEqual((parent / 'sentinel').read_bytes(), b'preserve unrelated entry')
        self.assertEqual(decoy.read_bytes(), sealed.data)
        self.assertEqual(FileIdentity.from_stat(decoy.stat()), decoy_identity)

    def test_corrupt_second_copy_with_temporary_parent_substitution_rolls_back_owned_archive(self):
        from aimusic.audio.publication import PinnedOutput
        from aimusic.audio.publication_posix import PosixWorkspace
        parent, retained, substitute = (self.base / n for n in ('output', 'retained', 'substitute'))
        parent.mkdir()
        substitute.mkdir()
        sealed = self.assemble()
        (parent / 'sentinel').write_bytes(b'owned directory sentinel')
        (substitute / 'sentinel').write_bytes(b'substitute sentinel')
        (substitute / 'review.zip').write_bytes(sealed.data)
        original_identity = parent.stat().st_ino
        stage, verify = PosixWorkspace.stage, PinnedOutput.verify
        faults = []
        swapped = False

        def corrupt(work, source, name):
            identity = stage(work, source, name)
            data = source.read_bytes()
            with os.fdopen(os.open(name, os.O_WRONLY | os.O_TRUNC, dir_fd=work.fd), 'wb') as stream:
                stream.write(bytes([data[0] ^ 1]) + data[1:])
            faults.append(source.name)
            return identity

        def restore():
            nonlocal swapped
            if swapped:
                parent.rename(substitute)
                retained.rename(parent)
                swapped = False

        def swap_between_checks(pinned):
            nonlocal swapped
            if swapped:
                restore()  # Defeat the pathname post-check in the old implementation.
                return verify(pinned)
            verify(pinned)
            if pinned.path == parent and (parent / 'review.zip').exists():
                self.assertNotEqual((parent / 'review.zip').read_bytes(), sealed.data)
                parent.rename(retained)
                substitute.rename(parent)
                swapped = True

        try:
            with patch.object(PosixWorkspace, 'stage', corrupt), patch.object(PinnedOutput, 'verify', swap_between_checks):
                with self.assertRaisesRegex(ValueError, 'published'):
                    self.module.write_review_archive(self.root, parent / 'review.zip', bundle=self.bundle,
                                                     assignment=self.assignment,
                                                     expected_bundle_sha256=self.bundle.sha256,
                                                     expected_assignment_sha256=self.assignment.sha256)
            # Rollback must already have targeted the owned directory, not the decoy.
            self.assertTrue(swapped)
            self.assertEqual({p.name for p in retained.iterdir()}, {'sentinel'})
            self.assertEqual((parent / 'review.zip').read_bytes(), sealed.data)
        finally:
            restore()
        self.assertEqual(faults, ['review.zip'])
        self.assertEqual(parent.stat().st_ino, original_identity)
        self.assertEqual({p.name for p in parent.iterdir()}, {'sentinel'})
        self.assertEqual((parent / 'sentinel').read_bytes(), b'owned directory sentinel')
        self.assertEqual({p.name for p in substitute.iterdir()}, {'sentinel', 'review.zip'})
        self.assertEqual((substitute / 'sentinel').read_bytes(), b'substitute sentinel')
        self.assertEqual((substitute / 'review.zip').read_bytes(), sealed.data)


class TestWindowsPublicReadback(unittest.TestCase):
    """Host-independent ABI-adapter checks, not native Windows execution evidence."""
    def test_readonly_leaf_share_compatibility_and_bounded_short_reads(self):
        from aimusic.audio import review_archive as archive
        from aimusic.audio.windows_native import Information, READ, SHARE_ALL
        api = Mock()
        handle = api.open.return_value.__enter__ = Mock(return_value=Mock())
        api.open.return_value.__exit__ = Mock(return_value=False)
        api.information.return_value = Information(FileIdentity(1, 2), 0, 6, 1)
        api.read.side_effect = [b'ab', b'cdef', b'']
        path = Path('public.zip')
        with patch('aimusic.audio.windows_native.WindowsAPI', return_value=api):
            archive._verify_windows_published(path, b'abcdef', expected_identity=FileIdentity(1, 2))
        api.open.assert_called_once_with(path, READ, share=SHARE_ALL)
        self.assertEqual([call.args for call in api.read.call_args_list],
                         [(handle.return_value, 7), (handle.return_value, 5), (handle.return_value, 1)])
        api.open.return_value.__exit__.assert_called_once()
        api.write.assert_not_called()

    def test_opened_handle_identity_required_before_size_or_read(self):
        from aimusic.audio import review_archive as archive
        from aimusic.audio.windows_native import Information, READ, SHARE_ALL
        for identity in (FileIdentity(9, 2), FileIdentity(1, 9)):
            for size in (3, 99):
                with self.subTest(identity=identity, size=size):
                    api, handle = Mock(), Mock()
                    api.open.return_value.__enter__ = Mock(return_value=handle)
                    api.open.return_value.__exit__ = Mock(return_value=False)
                    api.information.return_value = Information(identity, 0, size, 1)
                    api.read.side_effect = AssertionError('unowned leaf read')
                    with patch('aimusic.audio.windows_native.WindowsAPI', return_value=api):
                        with self.assertRaisesRegex(ValueError, 'published.*identity'):
                            archive._verify_windows_published(Path('public.zip'), b'abc',
                                                              expected_identity=FileIdentity(1, 2))
                    api.open.assert_called_once_with(Path('public.zip'), READ, share=SHARE_ALL)
                    api.information.assert_called_once_with(handle)
                    api.read.assert_not_called()
                    api.open.return_value.__exit__.assert_called_once()

    def test_corruption_truncation_overrun_reparse_and_size_fail_closed(self):
        from aimusic.audio import review_archive as archive
        from aimusic.audio.windows_native import Information, REPARSE, DIRECTORY
        for attributes, size, chunks in ((0, 3, [b'bad']), (0, 3, [b'ab', b'']),
                                         (0, 3, [b'abcX']), (0, 4, []),
                                         (REPARSE, 3, []), (DIRECTORY, 3, [])):
            with self.subTest(attributes=attributes, size=size, chunks=chunks):
                api = Mock()
                api.open.return_value.__enter__ = Mock(return_value=Mock())
                api.open.return_value.__exit__ = Mock(return_value=False)
                api.information.return_value = Information(FileIdentity(1, 2), attributes, size, 1)
                api.read.side_effect = chunks
                with patch('aimusic.audio.windows_native.WindowsAPI', return_value=api):
                    with self.assertRaises(ValueError):
                        archive._verify_windows_published(Path('public.zip'), b'abc', expected_identity=FileIdentity(1, 2))
                api.open.return_value.__exit__.assert_called_once()
                if not chunks:
                    api.read.assert_not_called()


@unittest.skipIf(os.name == 'nt', 'POSIX namespace-swap characterization, not native Windows evidence')
class TestUnsupportedConcurrentMutation(ArchiveCase):
    def test_mkdir_can_escape_swapped_parent_before_postcheck_rejects(self):
        """Expected limitation, NOT a claim that post-check prevents creation."""
        parent, outside, retained = self.base / 'parent', self.base / 'outside', self.base / 'retained'
        parent.mkdir()
        outside.mkdir()
        out = parent / 'view'
        real = os.mkdir

        def swap(path, mode=0o777, **kwargs):
            if Path(path) == out and 'dir_fd' not in kwargs:
                parent.rename(retained)
                parent.symlink_to(outside, target_is_directory=True)
            return real(path, mode, **kwargs)

        sealed = self.assemble()
        with patch('os.mkdir', swap), self.assertRaises((ValueError, OSError)):
            self.module.extract_archive(sealed.data, out, expected_sha256=sealed.sha256)
        self.assertTrue((outside / 'view').is_dir())  # The path-based syscall escaped already.
        self.assertEqual(list((outside / 'view').iterdir()), [])
        self.assertFalse((outside / 'view/complete.json').exists())

    def test_mkdir_then_pin_can_adopt_an_ordinary_substitute(self):
        """Even a passing view check is not proof of custody or mkdir ownership."""
        out, displaced = self.base / 'view', self.base / 'displaced'
        real = os.mkdir
        swapped = False

        def substitute(path, mode=0o777, **kwargs):
            nonlocal swapped
            result = real(path, mode, **kwargs)
            if Path(path) == out and 'dir_fd' not in kwargs and not swapped:
                swapped = True
                out.rename(displaced)
                real(out, mode)
            return result

        sealed = self.assemble()
        with patch('os.mkdir', substitute):
            self.module.extract_archive(sealed.data, out, expected_sha256=sealed.sha256)
        self.assertTrue(swapped)
        self.assertEqual(list(displaced.iterdir()), [])
        self.assertTrue((out / 'complete.json').exists())
        # verify_extracted checks current bytes, not unsupported concurrent custody.
        self.assertEqual(self.module.verify_extracted(sealed.data, out, expected_sha256=sealed.sha256), out)
