"""Asset-free contract for the four-mix, offline listening review."""
from __future__ import annotations

import base64
import errno
from html.parser import HTMLParser
import io
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import wave

from aimusic.audio.listening_review import DEMOS, MAX_TOTAL_WAV_BYTES, write_listening_review


class ReviewParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.text = []

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))

    def handle_data(self, data):
        self.text.append(data)


def parse_review(path):
    parser = ReviewParser()
    parser.feed(path.read_text(encoding='utf-8'))
    return parser


class TestListeningReview(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.outputs = {}
        self.wavs = []
        for i, (name, _, _) in enumerate(DEMOS):
            out = self.root / 'four-workflows' / name / 'cold'
            (out / 'stems').mkdir(parents=True)
            data = io.BytesIO()
            with wave.open(data, 'wb') as stream:
                stream.setparams((1, 2, 8000, 0, 'NONE', 'not compressed'))
                stream.writeframes(struct.pack('<h', i + 1) * 80)
            self.wavs.append(data.getvalue())
            (out / 'stems/rough_mix.wav').write_bytes(self.wavs[-1])
            (out / 'render.json').write_text('private path / credentials: never read')
            self.outputs[name] = out
        self.out = self.root / 'review.html'

    def require_symlink_capability(self, *, target_is_directory=False):
        # Probe file/directory support separately; never infer it from the OS name.
        kind = 'directory' if target_is_directory else 'file'
        target = self.root if target_is_directory else self.outputs[DEMOS[0][0]] / 'stems/rough_mix.wav'
        probe = self.root / 'symlink-capability-probe'
        try:
            probe.symlink_to(target, target_is_directory=target_is_directory)
        except NotImplementedError:
            self.skipTest(f'{kind} symlink capability unavailable: not implemented')
        except OSError as exc:
            winerror = getattr(exc, 'winerror', None)
            if winerror != 1314 and exc.errno not in (errno.EACCES, errno.EPERM, errno.ENOSYS, errno.ENOTSUP):
                raise
            self.skipTest(f'{kind} symlink capability unavailable: errno={exc.errno}, winerror={winerror}')
        self.assertTrue(probe.is_symlink())
        probe.unlink()

    def test_symlink_probe_skips_only_unavailable_capabilities(self):
        privilege_error = OSError(errno.EINVAL, 'symlink privilege unavailable')
        privilege_error.winerror = 1314
        unavailable = [NotImplementedError(), privilege_error]
        unavailable.extend(OSError(code, 'unavailable') for code in
                           (errno.EACCES, errno.EPERM, errno.ENOSYS, errno.ENOTSUP))
        for error in unavailable:
            with self.subTest(error=repr(error)):
                with patch.object(Path, 'symlink_to', side_effect=error):
                    with self.assertRaisesRegex(unittest.SkipTest, 'symlink capability unavailable'):
                        self.require_symlink_capability()
        for code in (errno.EIO, errno.EEXIST, errno.ENOENT, errno.EINVAL):
            error = OSError(code, 'unexpected probe failure')
            with self.subTest(errno=code):
                with patch.object(Path, 'symlink_to', side_effect=error):
                    with self.assertRaises(OSError) as caught:
                        try:
                            self.require_symlink_capability()
                        except unittest.SkipTest as skipped:
                            self.fail(f'unexpected probe error was skipped: {skipped}')
                self.assertIs(caught.exception, error)

    def test_exact_four_embedded_bytes_order_and_no_external_resources(self):
        write_listening_review(dict(reversed(list(self.outputs.items()))), self.out)
        parsed = parse_review(self.out)
        audio = [attrs for tag, attrs in parsed.tags if tag == 'audio']
        self.assertEqual(len(audio), 4)
        for attrs, expected in zip(audio, self.wavs):
            self.assertIn('controls', attrs)
            self.assertNotIn('autoplay', attrs)
            self.assertIn('aria-labelledby', attrs)
            self.assertTrue(attrs['src'].startswith('data:audio/wav;base64,'))
            self.assertEqual(base64.b64decode(attrs['src'].split(',', 1)[1], validate=True), expected)
        for tag, attrs in parsed.tags:
            self.assertNotIn(tag, ('script', 'iframe', 'link', 'object'))
            self.assertNotIn('href', attrs)
            if 'src' in attrs:
                self.assertEqual(tag, 'audio')
        text = self.out.read_text()
        self.assertNotIn(str(self.root), text)
        self.assertNotIn('credentials', text)
        self.assertNotIn('render.json', text)
        self.assertIn('polyphonic f0 unmeasured', text)
        self.assertIn('CLAP/FAD', text)
        checks = [a for t, a in parsed.tags if t == 'input']
        self.assertGreaterEqual(len(checks), 3)
        self.assertTrue(all(a['type'] == 'checkbox' and 'checked' not in a for a in checks))

    def test_reproducible_bytes_independent_of_destination_and_input_order(self):
        write_listening_review(self.outputs, self.out)
        other = self.root / 'other.html'
        write_listening_review(dict(reversed(list(self.outputs.items()))), other)
        self.assertEqual(self.out.read_bytes(), other.read_bytes())

    def test_escapes_title_and_rejects_unbounded_text(self):
        title = '\"<script>alert(1)</script> & café'
        write_listening_review(self.outputs, self.out, title=title)
        parsed = parse_review(self.out)
        self.assertNotIn('script', [t for t, _ in parsed.tags])
        self.assertIn(title, parsed.text)
        self.assertIn('&lt;script&gt;', self.out.read_text())
        with self.assertRaises(ValueError):
            write_listening_review(self.outputs, self.root / 'long.html', title='x' * 201)
        self.assertFalse((self.root / 'long.html').exists())

    def test_exact_workflow_keys_required(self):
        for values in ({}, dict(list(self.outputs.items())[:3]), dict(self.outputs, extra=self.root)):
            with self.subTest(keys=list(values)), self.assertRaisesRegex(ValueError, 'four'):
                write_listening_review(values, self.out)
            self.assertFalse(self.out.exists())

    def test_missing_nonregular_and_invalid_wav_leave_no_output(self):
        wav = self.outputs[DEMOS[0][0]] / 'stems/rough_mix.wav'
        wav.unlink()
        with self.assertRaises(OSError):
            write_listening_review(self.outputs, self.out)
        wav.mkdir()
        with self.assertRaises((OSError, ValueError)):
            write_listening_review(self.outputs, self.out)
        wav.rmdir()
        if hasattr(os, 'mkfifo'):
            os.mkfifo(wav)
            with self.assertRaises((OSError, ValueError)):
                write_listening_review(self.outputs, self.out)
            wav.unlink()
        for data in (b'not a WAV', self.wavs[0][:-1]):
            wav.write_bytes(data)
            with self.assertRaisesRegex(ValueError, 'WAV'):
                write_listening_review(self.outputs, self.out)
        self.assertFalse(self.out.exists())

    def test_chmod_unreadable_wav_leaves_no_output(self):
        wav = self.outputs[DEMOS[0][0]] / 'stems/rough_mix.wav'
        original_mode = wav.stat().st_mode
        try:
            wav.chmod(0)
            # Probe the mode bits used by the reader, not os.access(): even a
            # privileged POSIX process must reject a WAV with no read bits.
            if wav.stat().st_mode & 0o444:
                self.skipTest('chmod unreadability unavailable: chmod(0) retains read bits')
            with self.assertRaises(PermissionError):
                write_listening_review(self.outputs, self.out)
        finally:
            wav.chmod(original_mode)
        self.assertFalse(self.out.exists())
        self.assertEqual(wav.read_bytes(), self.wavs[0])

    def test_injected_read_permission_denial_leaves_no_output(self):
        # Exercise the failure invariant even when chmod cannot remove read bits
        # (Windows), and when mode bits allow reads but an ACL/open denies them.
        wav = self.outputs[DEMOS[0][0]] / 'stems/rough_mix.wav'
        original_open = os.open

        def denied_open(path, *args, **kwargs):
            if path == wav:
                raise PermissionError(errno.EACCES, 'injected WAV read denial', str(path))
            return original_open(path, *args, **kwargs)

        with patch('aimusic.audio.listening_review.os.open', denied_open):
            with self.assertRaisesRegex(PermissionError, 'injected WAV read denial'):
                write_listening_review(self.outputs, self.out)
        self.assertFalse(self.out.exists())
        self.assertEqual(wav.read_bytes(), self.wavs[0])

    def test_float_wav_and_malformed_format_validation(self):
        # Native rough_mix is FLOAT WAV, not the PCM produced by stdlib wave.
        import numpy as np
        import soundfile as sf
        wav = self.outputs[DEMOS[0][0]] / 'stems/rough_mix.wav'
        sf.write(wav, np.full(80, .01, dtype=np.float32), 8000, subtype='FLOAT')
        original = wav.read_bytes()
        write_listening_review(self.outputs, self.out)
        audio = [attrs for tag, attrs in parse_review(self.out).tags if tag == 'audio']
        self.assertEqual(base64.b64decode(audio[0]['src'].split(',', 1)[1]), original)
        for offset, value in ((20, 7), (22, 0), (32, 0)):
            bad = bytearray(original)
            struct.pack_into('<H', bad, offset, value)
            wav.write_bytes(bad)
            with self.subTest(offset=offset), self.assertRaisesRegex(ValueError, 'WAV'):
                write_listening_review(self.outputs, self.root / 'invalid.html')
        self.assertFalse((self.root / 'invalid.html').exists())

    def test_exclusive_create_refuses_a_racing_destination(self):
        original = Path.open

        def racing_open(path, mode='r', *args, **kwargs):
            if path == self.out:
                with original(path, 'wb') as stream:
                    stream.write(b'another writer')
            return original(path, mode, *args, **kwargs)

        with patch.object(Path, 'open', racing_open), self.assertRaises(FileExistsError):
            write_listening_review(self.outputs, self.out)
        self.assertEqual(self.out.read_bytes(), b'another writer')

    def test_unknown_metadata_chunks_rejected_not_embedded(self):
        wav = self.outputs[DEMOS[0][0]] / 'stems/rough_mix.wav'
        data = self.wavs[0] + b'LIST' + struct.pack('<I', 12) + b'private path'
        wav.write_bytes(data[:4] + struct.pack('<I', len(data) - 8) + data[8:])
        with self.assertRaisesRegex(ValueError, 'WAV'):
            write_listening_review(self.outputs, self.out)
        self.assertFalse(self.out.exists())

    def test_total_cap_including_multiple_files_and_no_output(self):
        total = sum(map(len, self.wavs))
        with patch('aimusic.audio.listening_review.MAX_TOTAL_WAV_BYTES', total - 1):
            with self.assertRaisesRegex(ValueError, 'byte cap'):
                write_listening_review(self.outputs, self.out)
        self.assertFalse(self.out.exists())
        with patch('aimusic.audio.listening_review.MAX_TOTAL_WAV_BYTES', total):
            write_listening_review(self.outputs, self.out)
        self.assertTrue(self.out.is_file())
        huge = self.outputs[DEMOS[0][0]] / 'stems/rough_mix.wav'
        with huge.open('wb') as stream:
            stream.truncate(MAX_TOTAL_WAV_BYTES + 1)
        with self.assertRaisesRegex(ValueError, 'byte cap'):
            write_listening_review(self.outputs, self.root / 'large.html')
        self.assertFalse((self.root / 'large.html').exists())

    def test_existing_file_and_directory_refused(self):
        self.out.write_bytes(b'preserve')
        with self.assertRaises(FileExistsError):
            write_listening_review(self.outputs, self.out)
        self.assertEqual(self.out.read_bytes(), b'preserve')
        self.out.unlink()
        self.out.mkdir()
        with self.assertRaises(FileExistsError):
            write_listening_review(self.outputs, self.out)
        self.assertTrue(self.out.is_dir())

    def test_dangling_or_live_output_symlinks_refused(self):
        self.require_symlink_capability()
        for target in (self.root / 'missing', self.outputs[DEMOS[0][0]] / 'stems/rough_mix.wav'):
            with self.subTest(target=target.name):
                before = target.read_bytes() if target.exists() else None
                self.out.symlink_to(target)
                with self.assertRaises(FileExistsError):
                    write_listening_review(self.outputs, self.out)
                self.assertTrue(self.out.is_symlink())
                self.assertEqual(target.read_bytes() if target.exists() else None, before)
                self.out.unlink()

    def test_symlink_input_refused(self):
        self.require_symlink_capability()
        wav = self.outputs[DEMOS[0][0]] / 'stems/rough_mix.wav'
        saved = wav.with_name('saved.wav')
        wav.rename(saved)
        wav.symlink_to(saved)
        with self.assertRaises((ValueError, OSError)):
            write_listening_review(self.outputs, self.out)
        self.assertTrue(wav.is_symlink())
        self.assertEqual(saved.read_bytes(), self.wavs[0])
        self.assertFalse(self.out.exists())

    def test_symlink_output_parent_refused(self):
        self.require_symlink_capability(target_is_directory=True)
        alias = self.root / 'alias'
        alias.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises((ValueError, OSError)):
            write_listening_review(self.outputs, alias / 'review.html')
        self.assertTrue(alias.is_symlink())
        self.assertFalse(self.out.exists())

    def test_module_command_evidence_root_and_explicit_directories(self):
        base = [sys.executable, '-m', 'aimusic.audio.listening_review']
        first = subprocess.run(base + ['--evidence-root', str(self.root), '--out', str(self.out)],
                               capture_output=True, text=True, timeout=10)
        self.assertEqual(first.returncode, 0, first.stderr)
        second = self.root / 'explicit.html'
        args = [item for name, out in self.outputs.items() for item in ('--' + name, str(out))]
        result = subprocess.run(base + args + ['--out', str(second)], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.out.read_bytes(), second.read_bytes())
        collision = subprocess.run(base + args + ['--out', str(second)], capture_output=True, text=True, timeout=10)
        self.assertNotEqual(collision.returncode, 0)
        self.assertNotIn('Traceback', collision.stderr)
        for invalid in ([], ['--evidence-root', str(self.root), *args]):
            result = subprocess.run(base + invalid + ['--out', str(self.root / 'bad.html')],
                                    capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / 'bad.html').exists())
