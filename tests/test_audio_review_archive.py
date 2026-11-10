"""Sealed task-6.5 delivery: synthetic audio, no renderer or network."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import importlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
import warnings
import wave
import zipfile

from aimusic.audio.review_blinding import create_assignment
from aimusic.audio.review_bundle import Stimulus, canonical_bytes
from tests.test_audio_review_blinding import prepared_bundle
from tests.test_audio_review_bundle import ref


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fixture(root):
    bundle, blind = prepared_bundle(root)
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as stream:
        stream.setparams((1, 2, 8000, 80, 'NONE', 'not compressed'))
        stream.writeframes(struct.pack('<80h', *([4096] * 80)))
    wav = buffer.getvalue()
    presentations = {p.sha256: replace(p, output=ref(p.output.path, wav)) for p in bundle.presentations}
    bundle = replace(bundle, presentations=tuple(presentations.values()), stimuli=tuple(
        Stimulus(s.id, presentations[s.presentation_sha256].sha256) for s in bundle.stimuli))
    for p in bundle.presentations:
        (root / p.output.path).write_bytes(wav)
    assignment = create_assignment(bundle, 'synthetic-r01', root=root, expected_sha256=bundle.sha256)
    return bundle, blind, assignment


def zip_bytes(rows):
    """Independent canonical ZIP constructor; mutations preserve valid CRCs."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_STORED, allowZip64=False) as archive:
        for name, data in rows:
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Duplicate name:', category=UserWarning)
                archive.writestr(info, data)
    return buffer.getvalue()


class ArchiveCase(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name).resolve()
        self.root = self.base / 'inputs'
        self.root.mkdir()
        self.bundle, self.blind, self.assignment = fixture(self.root)
        self.module = importlib.import_module('aimusic.audio.review_archive')

    def assemble(self, **kwargs):
        return self.module.assemble_archive(self.root, bundle=self.bundle, assignment=self.assignment,
                                            expected_bundle_sha256=self.bundle.sha256,
                                            expected_assignment_sha256=self.assignment.sha256, **kwargs)

    def rows(self):
        sealed = self.assemble()
        with zipfile.ZipFile(io.BytesIO(sealed.data)) as archive:
            return [(i.filename, archive.read(i)) for i in archive.infolist()]

    def reject(self, data, **kwargs):
        with self.assertRaises(ValueError):
            self.module.verify_archive(data, expected_sha256=sha(data), **kwargs)
        out = self.base / 'attack-extraction'
        with self.assertRaises(ValueError):
            self.module.extract_archive(data, out, expected_sha256=sha(data), **kwargs)
        self.assertFalse(out.exists())


class TestSealedArchive(ArchiveCase):
    def test_deterministic_complete_zip_manifest_and_reviewer_only_members(self):
        sealed = self.assemble()
        self.assertEqual(sealed, self.assemble())
        self.assertEqual(sealed.sha256, sha(sealed.data))
        verified = self.module.verify_archive(sealed.data, expected_sha256=sealed.sha256)
        rows = self.rows()
        names = [name for name, _ in rows]
        expected = {'manifest.json', 'assignment.json', 'index.html',
                    *[p.purpose + '.html' for p in self.assignment.panels],
                    *[s.media.path for s in self.assignment.stimuli]}
        self.assertEqual(set(names), expected)
        self.assertEqual(names, sorted(names))
        self.assertEqual(dict(verified.members), dict(rows))
        manifest = json.loads(dict(rows)['manifest.json'])
        self.assertEqual(dict(rows)['manifest.json'], canonical_bytes(manifest))
        self.assertEqual(manifest['schema'], 'aimusic.review-archive/1')
        self.assertEqual([m['path'] for m in manifest['members']], sorted(expected - {'manifest.json'}))
        for item in manifest['members']:
            self.assertEqual(item, ref(item['path'], dict(rows)[item['path']]).to_dict())
        self.assertEqual(dict(rows)['assignment.json'], self.assignment.to_bytes())
        with zipfile.ZipFile(io.BytesIO(sealed.data)) as archive:
            for info in archive.infolist():
                self.assertEqual(info.date_time, (1980, 1, 1, 0, 0, 0))
                self.assertEqual(info.compress_type, zipfile.ZIP_STORED)
                self.assertEqual(info.external_attr, 0o100644 << 16)
                self.assertFalse(info.extra or info.comment or info.flag_bits)
        for secret in ('private/', 'order_seed', self.blind.order_seed, 'source_wav', 'render_sha256',
                       'blind_map', 'soundfont', *[c for _, c in self.blind.conditions]):
            self.assertNotIn(secret.encode(), sealed.data)

    def test_trusted_archive_and_member_digest_tampering(self):
        sealed = self.assemble()
        for trusted in ('0' * 64, '', 'A' * 64):
            with self.subTest(trusted=trusted), self.assertRaises(ValueError):
                self.module.verify_archive(sealed.data, expected_sha256=trusted)
        altered = bytearray(sealed.data)
        altered[100] ^= 1
        with self.assertRaises(ValueError):
            self.module.verify_archive(bytes(altered), expected_sha256=sealed.sha256)
        # Recomputed archive hash is not enough: retain the original payload manifest.
        rows = self.rows()
        self.reject(zip_bytes([(n, d + b'X' if n == 'index.html' else d) for n, d in rows]))
        manifest = json.loads(dict(rows)['manifest.json'])
        manifest['members'][0]['sha256'] = '0' * 64
        self.reject(zip_bytes([(n, canonical_bytes(manifest) if n == 'manifest.json' else d) for n, d in rows]))

    def test_missing_extra_and_unsafe_members(self):
        rows = self.rows()
        self.reject(zip_bytes(rows[:-1]))
        for name in ('extra.txt', '../outside', '/absolute', 'C:/drive', 'a\\b', 'CON', 'file.',
                     'bad /x', 'a//b', 'a/%2e', 'a\x00tail', 'a\n', 'complete.json'):
            with self.subTest(name=name):
                self.reject(zip_bytes([*rows, (name, b'x')]))
        for name in ('index.html', 'INDEX.HTML', 'Media/extra.wav', 'media'):
            with self.subTest(name=name):
                self.reject(zip_bytes([*rows, (name, b'x')]))

    def test_count_compressed_expanded_and_archive_budgets(self):
        sealed = self.assemble()
        rows = self.rows()
        total = sum(len(data) for _, data in rows)
        for key, value in (('max_members', len(rows) - 1), ('max_archive_bytes', len(sealed.data) - 1),
                           ('max_compressed_bytes', total - 1), ('max_expanded_bytes', total - 1)):
            with self.subTest(key=key):
                limits = replace(self.module.ArchiveLimits(), **{key: value})
                self.reject(sealed.data, limits=limits)
        for invalid in (0, -1, True, 10**20):
            with self.assertRaises(ValueError):
                self.module.ArchiveLimits(max_members=invalid)

    def test_malformed_metadata_links_special_compression_and_encryption(self):
        sealed = self.assemble()
        with zipfile.ZipFile(io.BytesIO(sealed.data)) as archive:
            central = archive.start_dir
        # Offsets are ZIP wire fields, not implementation hooks. Tamper each side
        # independently to expose inconsistent local/central interpretation.
        for label, offset, fmt, value in (
            ('local method', 8, '<H', 8), ('central method', central + 10, '<H', 8),
            ('encrypted', central + 8, '<H', 1), ('local encrypted', 6, '<H', 1),
            ('descriptor', central + 8, '<H', 8), ('local CRC', 14, '<I', 123),
            ('central size bomb', central + 24, '<I', 0xffffffff),
            ('local size bomb', 22, '<I', 0xffffffff), ('overlap', central + 42, '<I', 1),
            ('symlink', central + 38, '<I', 0o120777 << 16),
            ('fifo', central + 38, '<I', 0o010644 << 16),
            ('directory', central + 38, '<I', (0o040755 << 16) | 16),
            ('multi disk', len(sealed.data) - 18, '<H', 1),
            ('false count', len(sealed.data) - 12, '<H', 1),
            ('central offset', len(sealed.data) - 6, '<I', 1),
            ('time', central + 12, '<H', 1), ('version', central + 6, '<H', 63),
        ):
            with self.subTest(label=label):
                changed = bytearray(sealed.data)
                struct.pack_into(fmt, changed, offset, value)
                self.reject(bytes(changed))
        for changed in (sealed.data[:-1], sealed.data + b'trailer', b'prefix' + sealed.data,
                        b'not zip', zip_bytes(list(reversed(self.rows())))):
            self.reject(changed)
        rows = self.rows()
        for compression in (zipfile.ZIP_DEFLATED, zipfile.ZIP_BZIP2, zipfile.ZIP_LZMA):
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, 'w', compression=compression) as archive:
                for name, data in rows:
                    archive.writestr(name, data)
            self.reject(buffer.getvalue())
        for field, value in (('extra', b'\x01\x00\x00\x00'), ('comment', b'comment')):
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, 'w') as archive:
                for name, data in rows:
                    info = zipfile.ZipInfo(name)
                    setattr(info, field, value)
                    archive.writestr(info, data)
            self.reject(buffer.getvalue())

    def test_canonical_manifest_and_reviewer_semantics_not_just_hashes(self):
        rows = self.rows()
        manifest = json.loads(dict(rows)['manifest.json'])
        for changed in (dict(rows)['manifest.json'] + b' ', b'{"schema":1,"schema":2}', b'[]',
                        canonical_bytes({**manifest, 'extra': 'private'}),
                        canonical_bytes({**manifest, 'members': manifest['members'][:-1]})):
            self.reject(zip_bytes([(n, changed if n == 'manifest.json' else d) for n, d in rows]))
        # Recompute both layers: an admitted archive still cannot smuggle custom
        # HTML, arbitrary media, or a private field in the public assignment.
        for name, changed in (('index.html', b'<script>unsafe()</script>'),
                              (self.assignment.stimuli[0].media.path, b'not a wav'),
                              ('assignment.json', canonical_bytes({**self.assignment.to_dict(), 'conditions': []}))):
            payload = {n: d for n, d in rows if n != 'manifest.json'}
            payload[name] = changed
            updated = {**manifest, 'members': [ref(n, d).to_dict() for n, d in sorted(payload.items())]}
            payload['manifest.json'] = canonical_bytes(updated)
            self.reject(zip_bytes(sorted(payload.items())))


class TestArchiveBoundaries(ArchiveCase):
    def publish(self, out=None, **kwargs):
        return self.module.write_review_archive(self.root, self.base / 'review.zip' if out is None else out,
                                               bundle=self.bundle, assignment=self.assignment,
                                               expected_bundle_sha256=self.bundle.sha256,
                                               expected_assignment_sha256=self.assignment.sha256, **kwargs)

    def extract(self, data=None, out=None, **kwargs):
        sealed = self.assemble()
        return self.module.extract_archive(sealed.data if data is None else data,
                                            self.base / 'extracted' if out is None else out,
                                            expected_sha256=sealed.sha256, **kwargs)

    def test_publish_file_and_extract_only_verified_members_completion_last(self):
        from unittest.mock import patch
        from aimusic.audio.publication import PinnedOutput
        sealed = self.assemble()
        self.assertEqual(self.publish(), sealed.sha256)
        path = self.base / 'review.zip'
        self.assertEqual(path.read_bytes(), sealed.data)
        observed = []
        real = PinnedOutput.publication

        def inspect(pinned, staging, names, **kwargs):
            observed.extend(names)
            if 'complete.json' in names:
                self.assertEqual(len(observed), 23)
                for name, data in self.module.verify_archive(path, expected_sha256=sealed.sha256).members:
                    self.assertEqual((self.base / 'extracted' / name).read_bytes(), data)
            return real(pinned, staging, names, **kwargs)

        with patch.object(PinnedOutput, 'publication', inspect):
            out = self.extract(data=path)
        self.assertEqual(observed[-1], 'complete.json')
        self.assertEqual(out, self.base / 'extracted')
        self.assertEqual((out / 'complete.json').read_bytes(), canonical_bytes({
            'schema': 'aimusic.review-extraction/1', 'archive_sha256': sealed.sha256}))
        self.assertEqual({p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file()},
                         {n for n, _ in self.rows()} | {'complete.json'})
        self.assignment.verify(out, expected_sha256=self.assignment.sha256)

    def test_invalid_archive_never_creates_extraction_directory(self):
        sealed = self.assemble()
        for data in (sealed.data[:-1], b'invalid'):
            with self.assertRaises(ValueError):
                self.extract(data=data)
            self.assertFalse((self.base / 'extracted').exists())
        with self.assertRaises(ValueError):
            self.extract(limits=replace(self.module.ArchiveLimits(), max_members=1))
        self.assertFalse((self.base / 'extracted').exists())

    def test_existing_file_directory_case_alias_and_hardlink_outputs_untouched(self):
        for action, name in ((self.publish, 'review.zip'), (self.extract, 'extracted')):
            for kind in ('file', 'directory', 'casefold', 'hardlink'):
                with self.subTest(action=name, kind=kind):
                    destination = self.base / name
                    occupied = self.base / (name.upper() if kind == 'casefold' else name)
                    if kind == 'directory':
                        occupied.mkdir()
                        (occupied / 'sentinel').write_bytes(b'untouched')
                    elif kind == 'hardlink':
                        occupied.hardlink_to(self.root / self.bundle.presentations[0].output.path)
                    else:
                        occupied.write_bytes(b'untouched')
                    before = occupied.stat()
                    with self.assertRaises((ValueError, OSError)):
                        action(out=destination)
                    self.assertEqual(before.st_ino, occupied.stat().st_ino)
                    if kind == 'directory':
                        self.assertEqual((occupied / 'sentinel').read_bytes(), b'untouched')
                        (occupied / 'sentinel').unlink()
                        occupied.rmdir()
                    else:
                        self.assertEqual(occupied.read_bytes(), b'untouched' if kind != 'hardlink' else
                                         (self.root / self.bundle.presentations[0].output.path).read_bytes())
                        occupied.unlink()

    def test_link_reparse_and_source_alias_destinations_rejected(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        for action, name in ((self.publish, 'review.zip'), (self.extract, 'extracted')):
            # Mandatory simulated reparse evidence on every host, not a broad skip.
            real = Path.lstat

            def reparse(path, *args, **kwargs):
                result = real(path, *args, **kwargs)
                if path == self.base:
                    return SimpleNamespace(st_mode=result.st_mode, st_file_attributes=0x400)
                return result

            with patch.object(Path, 'lstat', reparse), self.assertRaises((ValueError, OSError)):
                action(out=self.base / name)
            self.assertFalse((self.base / name).exists())
        # Source-root identity, not spelling or case-sensitive Path comparisons.
        with self.assertRaises(ValueError):
            self.publish(out=self.root / 'new.zip')
        self.assertFalse((self.root / 'new.zip').exists())
        alias = self.base / 'INPUTS'
        if alias.exists():  # This real filesystem supplies a case-insensitive alias.
            with self.assertRaises(ValueError):
                self.publish(out=alias / 'new.zip')
            self.assertFalse((self.root / 'new.zip').exists())
        link = self.base / 'linked'
        try:
            link.symlink_to(self.root, target_is_directory=True)
        except OSError:
            return  # Reparse decision above is mandatory; native links need privilege.
        for action in (self.publish, self.extract):
            with self.assertRaises((ValueError, OSError)):
                action(out=link / 'new')
        dangling = self.base / 'dangling'
        dangling.symlink_to(self.base / 'absent')
        for action in (self.publish, self.extract):
            with self.assertRaises((ValueError, OSError)):
                action(out=dangling)
        self.assertTrue(dangling.is_symlink())

    def test_assembly_fault_failed_candidate_and_changed_input_publish_nothing(self):
        from unittest.mock import patch
        with patch.object(self.module, '_zip', side_effect=OSError('assembly fault')):
            with self.assertRaisesRegex(OSError, 'assembly fault'):
                self.publish()
        self.assertFalse((self.base / 'review.zip').exists())
        renders = {r.sha256: replace(r, metrics=tuple(replace(m, state='failed') if m.name == 'events' else m
                                                     for m in r.metrics)) for r in self.bundle.renders}
        presentations = {p.sha256: replace(p, render_sha256=renders[p.render_sha256].sha256)
                         for p in self.bundle.presentations}
        self.bundle = replace(self.bundle, renders=tuple(renders.values()), presentations=tuple(presentations.values()),
                              stimuli=tuple(replace(s, presentation_sha256=presentations[s.presentation_sha256].sha256)
                                            for s in self.bundle.stimuli))
        self.assignment = replace(self.assignment, bundle_sha256=self.bundle.sha256)
        with self.assertRaisesRegex(ValueError, 'candidate rejected'):
            self.publish()
        self.assertFalse((self.base / 'review.zip').exists())
        self.bundle, self.blind, self.assignment = fixture(self.root)
        (self.root / self.assignment.stimuli[0].media.path).write_bytes(b'tampered')
        with self.assertRaises(ValueError):
            self.publish()
        self.assertFalse((self.base / 'review.zip').exists())

    def test_input_budget_preflight_and_no_unlisted_source_reads(self):
        from unittest.mock import patch
        real = self.module.read_relative
        called = []

        def read(root, path, **kwargs):
            called.append(path)
            return real(root, path, **kwargs)

        (self.root / 'never-read-secret').write_bytes(b'secret')
        with patch.object(self.module, 'read_relative', read):
            self.assemble()
        self.assertEqual(set(called), {r.path for r in self.bundle.references()})
        self.assertEqual(len(called), len(set(called)))
        for key in ('max_input_bytes', 'max_input_files', 'max_expanded_bytes'):
            with patch.object(self.module, 'read_relative', side_effect=AssertionError('unexpected input read')):
                with self.assertRaisesRegex(ValueError, 'budget'):
                    self.assemble(limits=replace(self.module.ArchiveLimits(), **{key: 1}))

    def test_publication_failure_rolls_back_and_extraction_failure_stays_incomplete(self):
        from unittest.mock import patch
        from aimusic.audio.publication import PinnedOutput
        real = PinnedOutput.publication
        (self.base / 'sentinel').write_bytes(b'untouched')

        @contextmanager
        def fail(pinned, staging, names, **kwargs):
            with real(pinned, staging, names, **kwargs) as published:
                yield published
                raise OSError('injected publication fault')

        with patch.object(PinnedOutput, 'publication', fail), self.assertRaisesRegex(OSError, 'injected'):
            self.publish()
        self.assertFalse((self.base / 'review.zip').exists())
        calls = 0

        def interrupt(pinned, staging, names, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError('injected extraction fault')
            return real(pinned, staging, names, **kwargs)

        with patch.object(PinnedOutput, 'publication', interrupt), self.assertRaisesRegex(OSError, 'extraction fault'):
            self.extract()
        out = self.base / 'extracted'
        self.assertTrue(out.is_dir())
        self.assertFalse((out / 'complete.json').exists())
        self.assertTrue(list(out.rglob('*.html')))
        with self.assertRaises((ValueError, OSError)):
            self.extract()
        self.assertEqual((self.base / 'sentinel').read_bytes(), b'untouched')

    def test_completion_failure_and_competing_extraction_leaf_preserve_entries(self):
        from unittest.mock import patch
        from aimusic.audio.publication import PinnedOutput
        real = PinnedOutput.publication

        def fail_completion(pinned, staging, names, **kwargs):
            if 'complete.json' in names:
                raise OSError('completion fault')
            return real(pinned, staging, names, **kwargs)

        with patch.object(PinnedOutput, 'publication', fail_completion), self.assertRaisesRegex(OSError, 'completion fault'):
            self.extract()
        self.assertFalse((self.base / 'extracted/complete.json').exists())

        def collide(pinned, staging, names, **kwargs):
            if any(n.endswith('.wav') for n in names):
                (pinned.path / names[0]).write_bytes(b'competing writer')
                return real(pinned, staging, names, **kwargs)
            return real(pinned, staging, names, **kwargs)

        out = self.base / 'competed'
        with patch.object(PinnedOutput, 'publication', collide), self.assertRaises(FileExistsError):
            self.extract(out=out)
        self.assertFalse((out / 'complete.json').exists())
        self.assertEqual(next((out / 'media').iterdir()).read_bytes(), b'competing writer')

    def test_faults_before_public_visibility_and_after_copied_media_are_fail_closed(self):
        from unittest.mock import patch
        from aimusic.audio.publication import PinnedOutput
        real_zip = self.module._zip
        calls = 0

        def corrupt_assembly(members):
            nonlocal calls
            calls += 1
            result = real_zip(members)
            return result + b'corrupt' if calls == 1 else result

        with patch.object(self.module, '_zip', corrupt_assembly), self.assertRaises(ValueError):
            self.publish()
        self.assertFalse((self.base / 'review.zip').exists())
        with patch.object(self.module.os, 'fsync', side_effect=OSError('flush fault')):
            with self.assertRaisesRegex(OSError, 'flush fault'):
                self.publish()
        self.assertFalse((self.base / 'review.zip').exists())
        real = PinnedOutput.publication

        @contextmanager
        def corrupt_copied_media(pinned, staging, names, **kwargs):
            with real(pinned, staging, names, **kwargs) as published:
                yield published
            if names[0].endswith('.wav'):
                (pinned.path / names[0]).write_bytes(b'corrupted after copy')

        with patch.object(PinnedOutput, 'publication', corrupt_copied_media), self.assertRaises(ValueError):
            self.extract()
        self.assertFalse((self.base / 'extracted/complete.json').exists())

    def test_staged_bytes_are_rechecked_and_late_archive_collision_is_not_overwritten(self):
        from unittest.mock import patch
        from aimusic.audio.publication import PinnedOutput
        read = self.module.read_relative

        def corrupt_stage(root, path, **kwargs):
            data = read(root, path, **kwargs)
            return b'changed' if path == 'review.zip' else data

        with patch.object(self.module, 'read_relative', corrupt_stage), self.assertRaises(ValueError):
            self.publish()
        self.assertFalse((self.base / 'review.zip').exists())
        real = PinnedOutput.publication

        def collide(pinned, staging, names, **kwargs):
            (pinned.path / names[0]).write_bytes(b'other writer')
            return real(pinned, staging, names, **kwargs)

        with patch.object(PinnedOutput, 'publication', collide), self.assertRaises(FileExistsError):
            self.publish()
        self.assertEqual((self.base / 'review.zip').read_bytes(), b'other writer')

    def test_actual_process_exit_mid_extraction_has_no_completion_and_cannot_resume(self):
        import subprocess
        import sys
        sealed = self.assemble()
        source = self.base / 'source.zip'
        source.write_bytes(sealed.data)
        out = self.base / 'interrupted'
        program = '''
from contextlib import contextmanager
import os
from pathlib import Path
import sys
from aimusic.audio import review_archive as archive
from aimusic.audio.publication import PinnedOutput
publish = PinnedOutput.publication
@contextmanager
def interrupted(self, staging, names, **kwargs):
    with publish(self, staging, names, **kwargs) as published:
        yield published
    os._exit(19)
PinnedOutput.publication = interrupted
archive.extract_archive(Path(sys.argv[1]), Path(sys.argv[2]), expected_sha256=sys.argv[3])
'''
        completed = subprocess.run([sys.executable, '-c', program, str(source), str(out), sealed.sha256],
                                   capture_output=True, timeout=30)
        self.assertEqual(completed.returncode, 19, completed.stderr.decode())
        self.assertTrue(out.is_dir())
        self.assertFalse((out / 'complete.json').exists())
        with self.assertRaises(FileExistsError):
            self.extract(data=source, out=out)
        self.assertEqual(source.read_bytes(), sealed.data)

    def test_zip_count_budget_and_compression_refused_before_inventory_or_decoder(self):
        from unittest.mock import patch
        sealed = self.assemble()
        with patch.object(zipfile, 'ZipFile', side_effect=AssertionError('unbounded inventory')):
            with self.assertRaisesRegex(ValueError, 'budget'):
                self.module.verify_archive(sealed.data, expected_sha256=sealed.sha256,
                                           limits=replace(self.module.ArchiveLimits(), max_members=1))
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('bomb', b'0' * (1024 * 1024))
        data = buffer.getvalue()
        with patch.object(zipfile.ZipFile, 'open', side_effect=AssertionError('decoder invoked')):
            self.reject(data)
        source = self.base / 'oversized.zip'
        source.write_bytes(sealed.data)
        with patch.object(self.module.os, 'open', side_effect=AssertionError('over-budget read')):
            with self.assertRaisesRegex(ValueError, 'budget'):
                self.module.verify_archive(source, expected_sha256=sealed.sha256,
                                           limits=replace(self.module.ArchiveLimits(), max_archive_bytes=1))

    def test_wav_metadata_and_duration_checked_even_with_recomputed_trust_layers(self):
        original = dict(self.rows())
        manifest = json.loads(original.pop('manifest.json'))
        stimulus = self.assignment.stimuli[0]
        wav = original[stimulus.media.path]
        extra = b'PEAK' + struct.pack('<I', 16) + bytes(16)
        ancillary = wav[:4] + struct.pack('<I', len(wav) + len(extra) - 8) + wav[8:] + extra
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as stream:
            stream.setparams((1, 2, 1, 121, 'NONE', 'not compressed'))
            stream.writeframes(bytes(242))
        for changed in (b'not WAV', ancillary, buffer.getvalue()):
            members = dict(original)
            members[stimulus.media.path] = changed
            assignment = replace(self.assignment, stimuli=(replace(stimulus, media=ref(stimulus.media.path, changed)),
                                                          *self.assignment.stimuli[1:]))
            members['assignment.json'] = assignment.to_bytes()
            members.update(self.module.render_review_pages(assignment, expected_sha256=assignment.sha256))
            updated = {**manifest, 'assignment_sha256': assignment.sha256,
                       'members': [ref(n, d).to_dict() for n, d in sorted(members.items())]}
            members['manifest.json'] = canonical_bytes(updated)
            self.reject(zip_bytes(sorted(members.items())))

    def test_zip64_cannot_override_preflight_budgets_inside_zipfile_parser(self):
        from unittest.mock import patch
        data = self.assemble().data
        end = list(struct.unpack('<4s4H2IH', data[-22:]))
        # The small classic directory passes ordinary preflight, while ZIP64
        # overrides it inside Python's reader (allowZip64=False is write-only).
        zip64 = struct.pack('<4sQ2H2I4Q', b'PK\x06\x06', 44, 45, 45, 0, 0,
                            1000000, 1000000, 100000000, 0)
        locator = struct.pack('<4sIQI', b'PK\x06\x07', 0, len(data) - 22, 1)
        end[6] += len(zip64) + len(locator)
        hostile = data[:-22] + zip64 + locator + struct.pack('<4s4H2IH', *end)
        with patch.object(zipfile, 'ZipFile', side_effect=AssertionError('ZIP64 bypassed budget preflight')):
            self.reject(hostile)
