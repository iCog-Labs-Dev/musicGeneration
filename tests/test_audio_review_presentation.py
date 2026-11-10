"""Asset-free trim/gain presentation contracts; never touch renderer bytes."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import hashlib
import importlib
import io
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from aimusic.audio.listening_review import _validate_wav
from aimusic.audio.review_bundle import FileRef, PresentationNode
from tests.test_audio_review_bundle import example_bundle


def wav_bytes(samples, subtype='FLOAT', rate=8000):
    stream = io.BytesIO()
    sf.write(stream, samples, rate, format='WAV', subtype=subtype)
    return stream.getvalue()


def chunks(data):
    result = []
    pos = 12
    while pos < len(data):
        name, size = struct.unpack_from('<4sI', data, pos)
        result.append((name, data[pos + 8:pos + 8 + size]))
        pos += 8 + size + size % 2
    return result


class TestReviewPresentation(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        (self.root / 'private').mkdir()
        (self.root / 'media').mkdir()
        self.samples = np.tile([[.125, -.25], [.25, -.125]], (40, 1))
        self.set_source(wav_bytes(self.samples))

    def set_source(self, data):
        self.original = data
        (self.root / 'private/raw.wav').write_bytes(data)
        self.source = FileRef('private/raw.wav', hashlib.sha256(data).hexdigest(), len(data))
        self.render = replace(example_bundle().renders[0], reports=(self.source,))

    def write(self, **kwargs):
        try:
            module = importlib.import_module('aimusic.audio.review_presentation')
        except ModuleNotFoundError as exc:
            if exc.name != 'aimusic.audio.review_presentation':
                raise
            self.fail('Missing task 6.2 presentation transform')
        options = dict(render=self.render, source_wav=self.source, output='media/s00.wav',
                       start_sample=8, stop_sample=72, gain=2.0)
        options.update(kwargs)
        return module.write_presentation(self.root, **options)

    def assert_no_output(self):
        self.assertEqual(list((self.root / 'media').iterdir()), [])
        self.assertEqual((self.root / self.source.path).read_bytes(), self.original)

    def test_trim_gain_exact_audio_digest_chain_and_stable_bytes(self):
        node = self.write()
        self.assertIsInstance(node, PresentationNode)
        data = node.output.read_verified(self.root)
        pcm, rate = sf.read(io.BytesIO(data), always_2d=True)
        np.testing.assert_array_equal(pcm, self.samples[8:72] * 2)
        self.assertEqual(rate, 8000)
        self.assertEqual(node, PresentationNode(self.render.sha256, self.source,
                         FileRef('media/s00.wav', hashlib.sha256(data).hexdigest(), len(data)), 8, 72, 2.0))
        self.assertEqual(PresentationNode.from_bytes(node.to_bytes()), node)
        self.assertEqual(self.source.read_verified(self.root), self.original)
        repeated = self.write(output='media/s01.wav')
        self.assertEqual(repeated.output.read_verified(self.root), data)
        self.assertEqual(repeated.source_wav, node.source_wav)
        self.assertNotEqual(repeated.sha256, node.sha256)
        _validate_wav(data)
        self.assertEqual({name for name, _ in chunks(data)}, {b'fmt ', b'fact', b'data'})
        self.assertEqual(sf.info(io.BytesIO(data)).subtype, 'FLOAT')

    def test_all_allowed_encodings_mono_stereo_retain_format_and_strip_optional_chunks(self):
        for subtype in ('PCM_U8', 'PCM_16', 'PCM_24', 'PCM_32', 'FLOAT', 'DOUBLE'):
            for channels in (1, 2):
                with self.subTest(subtype=subtype, channels=channels):
                    samples = self.samples[:, :channels]
                    self.set_source(wav_bytes(samples, subtype))
                    node = self.write(output=f'media/s{len(subtype):x}{channels:x}.wav')
                    data = node.output.read_verified(self.root)
                    info = sf.info(io.BytesIO(data))
                    self.assertEqual((info.format, info.subtype, info.channels, info.samplerate, info.frames),
                                     ('WAV', subtype, channels, 8000, 64))
                    pcm, _ = sf.read(io.BytesIO(data), always_2d=True)
                    np.testing.assert_array_equal(pcm, samples[8:72] * 2)
                    self.assertTrue({name for name, _ in chunks(data)} <= {b'fmt ', b'fact', b'data'})
                    _validate_wav(data)
                    (self.root / node.output.path).unlink()
                    self.assert_no_output()

    def test_invalid_parameters_rejected_without_output(self):
        for options in ({'start_sample': -1}, {'start_sample': True}, {'start_sample': 1.0},
                        {'stop_sample': 8}, {'stop_sample': 81}, {'stop_sample': False},
                        {'gain': 0}, {'gain': -1}, {'gain': 17}, {'gain': True},
                        {'gain': float('nan')}, {'gain': float('inf')}, {'gain': '2'},
                        {'output': '../s00.wav'}, {'output': 'media/condition.wav'},
                        {'output': 'private/raw.wav'}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.write(**options)
            self.assert_no_output()

    def test_source_digest_and_admitted_render_link_required(self):
        for options in ({'source_wav': replace(self.source, sha256='0' * 64)},
                        {'source_wav': replace(self.source, size=self.source.size - 1)},
                        {'render': replace(self.render, reports=(replace(self.source, sha256='1' * 64),))},
                        {'render': replace(self.render, metrics=tuple(
                            replace(m, state='failed') if m.name == 'finite_audio' else m
                            for m in self.render.metrics))}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.write(**options)
            self.assert_no_output()
        (self.root / self.source.path).write_bytes(self.original[:-1] + b'x')
        with self.assertRaisesRegex(ValueError, 'digest'):
            self.write()
        self.assertEqual(list((self.root / 'media').iterdir()), [])

    def test_only_allowed_wav_containers_no_ancillary_chunks(self):
        bad = [b'not WAV', self.original[:-1], self.original + b'junk']
        for format_, subtype in (('AIFF', 'PCM_16'), ('FLAC', 'PCM_16'), ('WAV', 'ULAW'),
                                 ('WAVEX', 'PCM_16'), ('RF64', 'PCM_16')):
            stream = io.BytesIO()
            sf.write(stream, self.samples, 8000, format=format_, subtype=subtype)
            bad.append(stream.getvalue())
        for name in (b'LIST', b'bext', b'iXML', b'JUNK'):
            body = self.original[8:] + name + struct.pack('<I', 12) + b'private path'
            bad.append(b'RIFF' + struct.pack('<I', len(body)) + body)
        for data in bad:
            self.set_source(data)
            with self.subTest(header=data[:12]), self.assertRaisesRegex(ValueError, 'WAV'):
                self.write()
            self.assert_no_output()

    def test_nonfinite_or_overpeak_source_cannot_be_repaired_by_trim_or_gain(self):
        for value in (float('nan'), float('inf'), -float('inf'), .96, -.96, 1e200):
            samples = self.samples.copy()
            samples[0, 0] = value  # Outside the selected trim, still unsafe input.
            self.set_source(wav_bytes(samples, 'DOUBLE'))
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'finite|peak'):
                self.write(gain=.5)
            self.assert_no_output()

    def test_gain_cannot_clip_or_exceed_ceiling_before_or_after_encoding(self):
        with self.assertRaisesRegex(ValueError, 'peak'):
            self.write(gain=4)
        self.assert_no_output()
        for subtype, gain, ceiling in (('PCM_U8', .3, .075), ('FLOAT', .4, .1)):
            self.set_source(wav_bytes(self.samples, subtype))
            with self.subTest(subtype=subtype), self.assertRaisesRegex(ValueError, 'peak'):
                self.write(gain=gain, max_peak=ceiling)
            self.assert_no_output()
        node = self.write(max_peak=.5)  # Exact inclusive ceiling, not a tolerance.
        pcm, _ = sf.read(io.BytesIO(node.output.read_verified(self.root)))
        self.assertEqual(np.max(np.abs(pcm)), .5)

    def test_byte_duration_limits_and_inclusive_boundary(self):
        baseline = self.write()
        size = baseline.output.size
        (self.root / baseline.output.path).unlink()
        for options in ({'max_bytes': size - 1}, {'max_duration_sec': 63 / 8000}):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, 'byte|duration'):
                self.write(**options)
            self.assert_no_output()
        exact = self.write(max_bytes=size, max_duration_sec=64 / 8000)
        self.assertEqual(exact.output, baseline.output)
        (self.root / exact.output.path).unlink()
        self.set_source(wav_bytes(np.full(121, .125), rate=1))
        with self.assertRaisesRegex(ValueError, 'duration'):
            self.write(start_sample=0, stop_sample=1)  # Short trim cannot bypass input cap.
        self.assert_no_output()
        self.set_source(wav_bytes(np.full(120, .125), rate=1))
        node = self.write(start_sample=0, stop_sample=120)
        self.assertEqual(sf.info(io.BytesIO(node.output.read_verified(self.root))).duration, 120)

    def test_invalid_or_relaxed_budgets_rejected(self):
        from aimusic.audio.review_bundle import MAX_FILE_BYTES
        for name, values in (
                ('max_bytes', (0, -1, True, 100.0, MAX_FILE_BYTES + 1)),
                ('max_duration_sec', (0, -1, True, '1', float('nan'), float('inf'), 121)),
                ('max_peak', (0, -1, True, '1', float('nan'), float('inf'), .951))):
            for value in values:
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    self.write(**{name: value})
                self.assert_no_output()

    def test_oversized_source_refused_before_payload_read(self):
        from aimusic.audio.review_bundle import MAX_FILE_BYTES
        # FileRef retains the approved small size; a grown/sparse source must fail.
        with (self.root / self.source.path).open('wb') as stream:
            stream.truncate(MAX_FILE_BYTES + 1)
        with self.assertRaisesRegex(ValueError, 'budget'):
            self.write()
        self.assertEqual(list((self.root / 'media').iterdir()), [])

    def test_source_mutation_during_publication_rolls_back_output(self):
        from aimusic.audio.publication import PinnedOutput
        original = PinnedOutput.publication
        changed = wav_bytes(self.samples * .5)

        @contextmanager
        def mutate(pinned, *args, **kwargs):
            with original(pinned, *args, **kwargs):
                (self.root / self.source.path).write_bytes(changed)
                yield

        with patch.object(PinnedOutput, 'publication', mutate):
            with self.assertRaisesRegex(ValueError, 'digest'):
                self.write()
        self.assertEqual(list((self.root / 'media').iterdir()), [])
        self.assertEqual((self.root / self.source.path).read_bytes(), changed)

    def test_publication_failure_preserves_unrelated_bytes(self):
        from aimusic.audio.publication import PinnedOutput
        original = PinnedOutput.publication

        @contextmanager
        def fail(pinned, *args, **kwargs):
            with original(pinned, *args, **kwargs):
                (self.root / 'media/other.txt').write_bytes(b'another writer')
                yield
                raise OSError('injected publication failure')

        with patch.object(PinnedOutput, 'publication', fail):
            with self.assertRaisesRegex(OSError, 'injected publication failure'):
                self.write()
        self.assertEqual([p.name for p in (self.root / 'media').iterdir()], ['other.txt'])
        self.assertEqual((self.root / 'media/other.txt').read_bytes(), b'another writer')
        self.assertEqual(self.source.read_verified(self.root), self.original)

    def test_existing_file_directory_case_alias_and_source_hardlink_never_overwritten(self):
        for kind in ('file', 'directory', 'case-alias', 'hardlink'):
            target = self.root / 'media' / ('S00.WAV' if kind == 'case-alias' else 's00.wav')
            if kind == 'directory':
                target.mkdir()
            elif kind == 'hardlink':
                target.hardlink_to(self.root / self.source.path)
            else:
                target.write_bytes(b'preserve')
            with self.subTest(kind=kind), self.assertRaises(FileExistsError):
                self.write()
            if kind == 'directory':
                self.assertTrue(target.is_dir())
                target.rmdir()
            else:
                self.assertEqual(target.read_bytes(), self.original if kind == 'hardlink' else b'preserve')
                target.unlink()
            self.assert_no_output()

    def test_nonregular_source_rejected(self):
        path = self.root / self.source.path
        path.unlink()
        path.mkdir()
        with self.assertRaisesRegex(ValueError, 'nonregular'):
            self.write()
        self.assertEqual(list((self.root / 'media').iterdir()), [])

    def test_optional_symlink_source_and_destination_ancestor_refused(self):
        import errno
        path = self.root / 'probe'
        try:
            path.symlink_to(self.root / self.source.path)
        except OSError as exc:
            if getattr(exc, 'winerror', None) == 1314 or exc.errno in (errno.EACCES, errno.EPERM, errno.ENOTSUP):
                self.skipTest('file symlink capability unavailable')
            raise
        path.unlink()
        source = self.root / self.source.path
        source.rename(self.root / 'private/original.wav')
        source.symlink_to(self.root / 'private/original.wav')
        with self.assertRaisesRegex(ValueError, 'link'):
            self.write()
        source.unlink()
        (self.root / 'private/original.wav').rename(source)
        (self.root / 'media').rmdir()
        (self.root / 'other').mkdir()
        try:
            (self.root / 'media').symlink_to(self.root / 'other', target_is_directory=True)
        except OSError as exc:
            if getattr(exc, 'winerror', None) == 1314 or exc.errno in (errno.EACCES, errno.EPERM, errno.ENOTSUP):
                self.skipTest('directory symlink capability unavailable; source link rejection exercised')
            raise
        with self.assertRaises((ValueError, OSError)):
            self.write()
        self.assertEqual(list((self.root / 'other').iterdir()), [])
        self.assertEqual(self.source.read_verified(self.root), self.original)

    def test_optional_peak_timestamp_cannot_affect_presentation_bytes(self):
        first = self.write()
        original = self.original
        modified = bytearray(original)
        peak = next(payload for name, payload in chunks(original) if name == b'PEAK')
        position = original.index(b'PEAK') + 8
        stamp = struct.unpack_from('<I', peak, 4)[0]
        struct.pack_into('<I', modified, position + 4, stamp ^ 0xffffffff)
        self.set_source(bytes(modified))
        second = self.write(output='media/s01.wav')
        self.assertNotEqual(first.source_wav.sha256, second.source_wav.sha256)
        self.assertEqual(first.output.sha256, second.output.sha256)
        self.assertEqual(self.source.read_verified(self.root), bytes(modified))

    def test_single_frame_odd_pcm_padding_and_maximum_gain(self):
        self.set_source(wav_bytes(np.full(80, .015625), 'PCM_U8'))
        node = self.write(start_sample=79, stop_sample=80, gain=16)
        data = node.output.read_verified(self.root)
        _validate_wav(data)
        pcm, rate = sf.read(io.BytesIO(data))
        np.testing.assert_array_equal(pcm, [.25])
        self.assertEqual(rate, 8000)
        self.assertEqual((node.start_sample, node.stop_sample, node.gain), (79, 80, 16))
        self.assertEqual(self.source.read_verified(self.root), self.original)
