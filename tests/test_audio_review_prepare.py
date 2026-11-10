"""One-shot preparation with synthetic admitted metadata, never native evidence."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from aimusic.audio import review_archive
from aimusic.audio.review_blinding import ReviewAssignment
from aimusic.audio.review_bundle import ReviewBundle, canonical_bytes
from tests.test_audio_review_blinding import example_matrix
from tests.test_audio_review_bundle import ref
from tests.test_audio_review_presentation import wav_bytes


def matrix_fixture(root, subtype='FLOAT'):
    matrix = example_matrix()
    wav = wav_bytes(np.full((137, 2), .125), subtype=subtype)
    raw = ref('private/raw.wav', wav)
    renders = {r.sha256: replace(r, reports=tuple(raw if f.path == raw.path else f for f in r.reports))
               for r in matrix.renders}
    matrix = replace(matrix, renders=tuple(renders.values()), exposures=tuple(
        replace(e, render_sha256=renders[e.render_sha256].sha256, wav=raw) for e in matrix.exposures))
    # Fixture graph has only these private source/evidence files.
    (root / 'private').mkdir()
    for name in ('source.mid', 'render.json', 'evidence.json'):
        (root / 'private' / name).write_bytes(b'{}')
    (root / raw.path).write_bytes(wav)
    (root / 'matrix.json').write_bytes(matrix.to_bytes())
    return matrix


class TestReviewPrepare(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix='prepare café ')
        self.addCleanup(tmp.cleanup)
        self.base = Path(tmp.name).resolve()
        self.root = self.base / 'private matrix'
        self.root.mkdir()
        self.matrix = matrix_fixture(self.root)

    def args(self):
        return ['prepare', '--input-root', str(self.root), '--matrix', str(self.root / 'matrix.json'),
                '--expected-matrix-sha256', self.matrix.sha256, '--reviewer', 'synthetic-r01']

    def invoke(self, args=None):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch('sys.stdout', stdout), patch('sys.stderr', stderr):
            try:
                code = review_archive.main(self.args() if args is None else args)
            except SystemExit as exc:
                code = exc.code
        return code, stdout.getvalue(), stderr.getvalue()

    def refused(self, result):
        code, out, err = result
        self.assertEqual(code, 1, err)
        self.assertEqual(out, '')
        self.assertIn('fresh private root', json.loads(err)['error'])
        self.assertFalse((self.root / 'prepared.json').exists())

    def test_complete_full_frame_gain_one_records_then_assemble(self):
        before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        code, out, err = self.invoke()
        self.assertEqual(code, 0, err)
        result = json.loads(out)
        self.assertEqual(out.encode(), canonical_bytes(result))
        self.assertEqual((self.root / 'prepared.json').read_bytes(), out.encode())
        self.assertEqual(result['operation'], 'prepare')
        self.assertEqual(result['matrix_sha256'], self.matrix.sha256)
        bundle = ReviewBundle.from_bytes(Path(result['bundle']).read_bytes(), expected_sha256=result['bundle_sha256'])
        assignment = ReviewAssignment.from_bytes(Path(result['assignment']).read_bytes(),
                                                expected_sha256=result['assignment_sha256'])
        self.assertEqual(hashlib.sha256(Path(result['blind_map']).read_bytes()).hexdigest(), result['blind_map_sha256'])
        self.assertEqual(len(bundle.presentations), 14)
        bundle.verify(self.root, expected_sha256=bundle.sha256)
        assignment.validate_bundle(bundle)
        for p in bundle.presentations:
            self.assertEqual((p.start_sample, p.stop_sample, p.gain), (0, 137, 1.0))
            samples, rate = sf.read(self.root / p.output.path, always_2d=True)
            np.testing.assert_array_equal(samples, np.full((137, 2), .125))
            self.assertEqual(rate, 8000)
        self.assertTrue(all(p.read_bytes() == data for p, data in before.items()))
        archive = self.base / 'review.zip'
        code, _, err = self.invoke(['assemble', '--input-root', result['input_root'], '--bundle', result['bundle'],
                                   '--expected-bundle-sha256', result['bundle_sha256'], '--assignment', result['assignment'],
                                   '--expected-assignment-sha256', result['assignment_sha256'], '--out', str(archive)])
        self.assertEqual(code, 0, err)
        self.assertTrue(archive.is_file())
        snapshot = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        code, out, err = self.invoke()
        self.assertEqual(code, 1, err)
        self.assertEqual(out, '')
        self.assertTrue(all(p.read_bytes() == data for p, data in snapshot.items()))

    def test_pcm_full_clip_uses_frames_not_samples_or_float_only_wave_reader(self):
        for subtype in ('PCM_16', 'PCM_24'):
            root = self.base / subtype
            root.mkdir()
            self.root, self.matrix = root, matrix_fixture(root, subtype)
            code, out, err = self.invoke()
            self.assertEqual(code, 0, err)
            bundle = ReviewBundle.from_bytes(Path(json.loads(out)['bundle']).read_bytes())
            self.assertTrue(all(p.stop_sample == 137 for p in bundle.presentations))

    def test_explicit_arguments_and_pseudonym_required(self):
        args = self.args()
        for index in range(1, len(args), 2):
            self.assertEqual(self.invoke(args[:index] + args[index + 2:])[0], 2)
        args[-1] = '../real name'
        self.refused(self.invoke(args))
        self.assertFalse((self.root / 'media').exists())

    def test_trusted_canonical_matrix_required_inside_selected_root(self):
        path = self.root / 'matrix.json'
        original = path.read_bytes()
        self.refused(self.invoke(self.args()[:-3] + ['0' * 64, '--reviewer', 'synthetic-r01']))
        for data in (original + b'\n', b'{}', b'[' * 1100):
            path.write_bytes(data)
            args = self.args()
            args[6] = hashlib.sha256(data).hexdigest()
            self.refused(self.invoke(args))
        path.write_bytes(original)
        outside = self.base / 'matrix.json'
        outside.write_bytes(original)
        args = self.args()
        args[4] = str(outside)
        self.refused(self.invoke(args))
        self.assertFalse((self.root / 'media').exists())

    def test_oversized_and_nonregular_matrix_refused_before_parsing(self):
        from aimusic.audio.review_matrix import ReviewMatrix
        path = self.root / 'matrix.json'
        with path.open('wb') as stream:
            stream.truncate(32 * 1024 * 1024 + 1)
        with patch.object(ReviewMatrix, 'from_bytes', side_effect=AssertionError('unbounded parse')):
            self.refused(self.invoke())
            path.unlink()
            path.mkdir()
            self.refused(self.invoke())
        self.assertFalse((self.root / 'prepare-started.json').exists())

    def test_collision_preserves_existing_bytes(self):
        for name in ('bundle.json', 'prepared.json', 'prepare-started.json', 'media', 'assignments', 'private/blind-map.json'):
            path = self.root / name
            path.write_bytes(b'keep')
            code, out, err = self.invoke()
            self.assertEqual(code, 1, err)
            self.assertEqual(out, '')
            self.assertEqual(path.read_bytes(), b'keep')
            path.unlink()

    def test_failed_presentation_retains_residue_without_bundle_and_refuses_retry(self):
        from aimusic.audio.review_presentation import write_presentation
        count = 0

        def fail(*args, **kwargs):
            nonlocal count
            count += 1
            if count == 2:
                raise ValueError('synthetic write failure')
            return write_presentation(*args, **kwargs)

        with patch('aimusic.audio.review_presentation.write_presentation', side_effect=fail):
            self.refused(self.invoke())
        self.assertEqual(len(list((self.root / 'media').iterdir())), 1)
        self.assertFalse((self.root / 'bundle.json').exists())
        self.refused(self.invoke())

    def test_failed_evidence_or_metadata_publication_never_completes(self):
        (self.root / 'private/source.mid').write_bytes(b'tampered')
        self.refused(self.invoke())
        self.assertFalse((self.root / 'bundle.json').exists())
        other = self.base / 'second'
        other.mkdir()
        self.root, self.matrix = other, matrix_fixture(other)
        real = review_archive._publish_files

        def fail(directory, members, **kwargs):
            if 'prepared.json' in members:
                raise OSError('synthetic completion write failure')
            return real(directory, members, **kwargs)

        with patch.object(review_archive, '_publish_files', side_effect=fail):
            self.refused(self.invoke())
        self.refused(self.invoke())
