"""Asset-free task 6.3 contracts; synthetic pseudonyms and private maps only."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib
from pathlib import Path
import tempfile
import unittest

from aimusic.audio.review_bundle import Panel, PrivateBlindMap, canonical_bytes
from aimusic.audio.review_matrix import Exposure, ReviewMatrix
from tests.test_audio_review_bundle import example_bundle, materialize


def example_matrix():
    bundle = example_bundle()
    names = (('fixture-12', 'generated-12', 'reference-19', 'multitrack-19'),
             ('fixture-drums', 'fixture-bass', 'fixture-comping', 'fixture-lead'),
             ('package-12', 'imported-12'), ('groove-off', 'groove-on'),
             ('valid-19', 'flattened-12-calibration-control'))
    panels = tuple(Panel(p.purpose, conditions) for p, conditions in zip(bundle.panels, names))
    conditions = [s for panel in panels for s in panel.stimuli]
    exposures = tuple(Exposure(c, p.render_sha256, p.source_wav) for c, p in zip(conditions, bundle.presentations))
    return ReviewMatrix('aimusic.review-matrix/1', bundle.sources, bundle.renders, exposures, panels)


def synthetic_entropy():
    # Deliberately non-monotonic: public ordering must not preserve private indices.
    index = 0

    def draw(size):
        nonlocal index
        index += 1
        return hashlib.sha256(str(index).encode()).digest()[:size]
    return draw


class TestOpaqueNames(unittest.TestCase):
    def module(self):
        try:
            return importlib.import_module('aimusic.audio.review_blinding')
        except ModuleNotFoundError as exc:
            if exc.name != 'aimusic.audio.review_blinding':
                raise
            self.fail('Missing task 6.3 blinding implementation')

    def test_names_are_unique_independent_of_reused_wav_and_private_map_roundtrips(self):
        matrix = example_matrix()
        blind = self.module().create_blind_map(matrix, random_bytes=synthetic_entropy())
        self.assertIsInstance(blind, PrivateBlindMap)
        self.assertEqual([c for _, c in blind.conditions], [e.condition for e in matrix.exposures])
        ids = [s for s, _ in blind.conditions]
        self.assertEqual(len(set(ids)), 14)  # Even identical media get separate exposures.
        for stimulus in ids:
            self.assertRegex(stimulus, r'^s[0-9a-f]{64}$')
        self.assertRegex(blind.order_seed, r'^[0-9a-f]{64}$')
        self.assertNotIn('s' + blind.order_seed, ids)
        self.assertEqual(PrivateBlindMap.from_bytes(blind.to_bytes()), blind)
        self.assertEqual(blind, self.module().create_blind_map(matrix, random_bytes=synthetic_entropy()))
        self.assertNotEqual(blind, self.module().create_blind_map(matrix))

    def test_colliding_entropy_retried_boundedly_and_invalid_entropy_refused(self):
        module = self.module()
        values = iter([b'a' * 32, b'a' * 32, *[bytes([i]) * 32 for i in range(1, 16)]])
        blind = module.create_blind_map(example_matrix(), random_bytes=lambda n: next(values))
        self.assertEqual(len({s for s, _ in blind.conditions}), 14)
        with self.assertRaisesRegex(ValueError, 'collision'):
            module.create_blind_map(example_matrix(), random_bytes=lambda n: b'a' * n)
        for invalid in (b'', b'a' * 31, 'a' * 32, None):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'entropy'):
                module.create_blind_map(example_matrix(), random_bytes=lambda n: invalid)

    def test_public_inventory_does_not_reveal_private_pair_or_global_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle = materialize(example_bundle(), Path(tmp))
            bundle = replace(bundle, stimuli=tuple(reversed(bundle.stimuli)), panels=tuple(
                replace(p, stimuli=tuple(reversed(p.stimuli))) if len(p.stimuli) == 2 else p for p in bundle.panels))
            projection = bundle.public_projection(root=Path(tmp), expected_sha256=bundle.sha256)
        self.assertEqual([s['id'] for s in projection['stimuli']], sorted(s.id for s in bundle.stimuli))
        for public, private in zip(projection['panels'], bundle.panels):
            self.assertEqual(public['stimuli'], sorted(private.stimuli) if len(private.stimuli) == 2 else list(private.stimuli))
        for forbidden in ('conditions', 'order_seed', 'private/', 'raw.wav', 'simple', 'native', 'backend'):
            self.assertNotIn(forbidden, canonical_bytes(projection).decode())


def prepared_bundle(root):
    from aimusic.audio import review_blinding as blinding
    matrix = example_matrix()
    blind = blinding.create_blind_map(matrix, random_bytes=synthetic_entropy())
    presentations = tuple(replace(p, output=replace(p.output, path=f'media/{s}.wav'))
                          for p, (s, _) in zip(example_bundle().presentations, blind.conditions))
    bundle = blinding.bind_presentations(matrix, blind, presentations)
    for item in bundle.references():
        path = root / item.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blind.to_bytes() if item == bundle.blind_map else b'{}')
    return bundle, blind


class TestBlindedBinding(unittest.TestCase):
    def test_bind_exact_exposures_to_private_conditions_and_digest_linked_media(self):
        from aimusic.audio import review_blinding as blinding
        matrix = example_matrix()
        blind = blinding.create_blind_map(matrix, random_bytes=synthetic_entropy())
        presentations = tuple(replace(p, output=replace(p.output, path=f'media/{s}.wav'))
                              for p, (s, _) in zip(example_bundle().presentations, blind.conditions))
        before = matrix.to_bytes()
        bundle = blinding.bind_presentations(matrix, blind, tuple(reversed(presentations)))
        blind.validate_bundle(bundle)
        self.assertEqual(matrix.to_bytes(), before)
        self.assertEqual(bundle.blind_map.sha256, hashlib.sha256(blind.to_bytes()).hexdigest())
        self.assertTrue(bundle.blind_map.path.startswith('private/'))
        ids = {condition: s for s, condition in blind.conditions}
        self.assertEqual(bundle.panels, tuple(Panel(p.purpose, tuple(ids[c] for c in p.stimuli)) for p in matrix.panels))
        outputs = {p.sha256: p.output.path for p in bundle.presentations}
        for stimulus in bundle.stimuli:
            self.assertEqual(outputs[stimulus.presentation_sha256], f'media/{stimulus.id}.wav')
        self.assertEqual(bundle, blinding.bind_presentations(matrix, blind, presentations))

        for bad in (presentations[:-1], (*presentations[:-1], presentations[0]),
                    (replace(presentations[0], source_wav=replace(presentations[0].source_wav,
                                                                path='private/other.wav')), *presentations[1:]),
                    (*presentations[:-1], replace(presentations[-1], render_sha256=presentations[0].render_sha256))):
            with self.subTest(bad=bad[0].output.path), self.assertRaises(ValueError):
                blinding.bind_presentations(matrix, blind, bad)
        for bad in (replace(blind, conditions=blind.conditions[:-1]),
                    replace(blind, conditions=tuple((s, 'unknown') for s, _ in blind.conditions)),
                    replace(blind, conditions=((blind.conditions[0][0], blind.conditions[1][1]), *blind.conditions[1:]))):
            with self.assertRaises(ValueError):
                blinding.bind_presentations(matrix, bad, presentations)


class TestReviewerAssignments(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.bundle, self.blind = prepared_bundle(self.root)

    def assignment(self, reviewer='synthetic-r01', bundle=None):
        from aimusic.audio import review_blinding as blinding
        bundle = bundle or self.bundle
        return blinding.create_assignment(bundle, reviewer, root=self.root, expected_sha256=bundle.sha256)

    def test_deterministic_balanced_pair_order_each_exposure_once_without_global_rng(self):
        import random
        from aimusic.audio import review_blinding as blinding
        before = self.bundle.to_bytes()
        self.addCleanup(random.setstate, random.getstate())
        random.seed(23)
        state = random.getstate()
        patterns = set()
        for i in range(24):
            doc = self.assignment(f'synthetic-r{i:02d}')
            self.assertEqual(doc.to_bytes(), self.assignment(f'synthetic-r{i:02d}').to_bytes())
            self.assertEqual(doc, blinding.ReviewAssignment.from_bytes(doc.to_bytes(), expected_sha256=doc.sha256))
            self.assertEqual(doc.panels[:2], self.bundle.panels[:2])
            flips = tuple(a.stimuli != b.stimuli for a, b in zip(doc.panels[2:], self.bundle.panels[2:]))
            self.assertIn(sum(flips), (1, 2))
            patterns.add(flips)
            self.assertEqual(len({s for p in doc.panels for s in p.stimuli}), 14)
            doc.validate_bundle(self.bundle)
        self.assertGreater(len(patterns), 1)
        self.assertEqual(random.getstate(), state)
        self.assertEqual(self.bundle.to_bytes(), before)

    def test_bundle_identity_and_private_seed_affect_assignment_but_no_public_leak(self):
        original = [self.assignment(f'synthetic-r{i}').panels for i in range(12)]
        changed = replace(self.bundle, finding_directory='other-findings')
        self.assertNotEqual(original, [self.assignment(f'synthetic-r{i}', changed).panels for i in range(12)])
        doc = self.assignment()
        projection = self.bundle.public_projection(root=self.root, expected_sha256=self.bundle.sha256)
        reference = doc.reference(self.bundle)
        reference.validate_bundle(self.bundle)
        self.assertEqual(reference.document.sha256, doc.sha256)
        self.assertEqual(reference.document.size, len(doc.to_bytes()))
        self.assertRegex(reference.document.path, r'^assignments/a[0-9a-f]{64}\.json$')
        self.assertNotIn(doc.reviewer, reference.document.path)
        serialized = canonical_bytes((doc, reference, projection)).decode()
        for forbidden in (*[c for _, c in self.blind.conditions], self.blind.order_seed,
                          'order_seed', 'conditions', 'private/', 'source_wav', 'raw.wav',
                          'blind_map', 'blind-map', 'presentation_sha256', 'render_sha256',
                          'backend', 'native', 'simple', 'soundfont', 'runtime'):
            self.assertNotIn(forbidden, serialized)
        for stimulus in doc.stimuli:
            self.assertEqual(stimulus.media.path, f'media/{stimulus.id}.wav')

    def test_reference_honors_bundle_assignment_directory_without_embedding_it(self):
        for directory in ('reviewers', 'sessions/reviewers', 'reviews/equipe 1'):
            with self.subTest(directory=directory):
                bundle = replace(self.bundle, assignment_directory=directory)
                doc = self.assignment(bundle=bundle)
                reference = doc.reference(bundle)
                self.assertEqual(reference.document.path, f'{directory}/a{doc.sha256}.json')
                reference.validate_bundle(bundle)
                doc.validate_bundle(bundle)
                self.assertEqual(reference.document.sha256, doc.sha256)
                self.assertEqual(reference.document.size, len(doc.to_bytes()))
                self.assertEqual(set(doc.to_dict()), {'schema', 'bundle_sha256', 'reviewer', 'panels', 'stimuli'})
                self.assertNotIn(directory, doc.to_bytes().decode())
                for forbidden in ('conditions', 'order_seed', 'blind_map', 'source_wav', 'backend', 'private/'):
                    self.assertNotIn(forbidden, canonical_bytes((doc, reference)).decode())
                with self.assertRaisesRegex(ValueError, 'digest'):
                    doc.reference(self.bundle)  # Cannot rebase onto a different immutable bundle.

    def test_golden_ordering_vector(self):
        # Independently calculated HMAC vector: timing ranks first, one reversal.
        self.assertEqual(self.bundle.sha256, '7a81f7177b5ff9b8c9b52a8d723202942761143faa7ef13b263aacf685beaa0b')
        doc = self.assignment('synthetic-r01')
        self.assertEqual(tuple(p.stimuli for p in doc.panels[2:]),
                         (self.bundle.panels[2].stimuli, tuple(reversed(self.bundle.panels[3].stimuli)),
                          self.bundle.panels[4].stimuli))

    def test_reviewer_document_has_strict_allowlist_paths_and_exact_bundle_links(self):
        from aimusic.audio import review_blinding as blinding
        doc = self.assignment()
        data = doc.to_dict()
        for key in ('conditions', 'order_seed', 'backend', 'source_wav'):
            with self.assertRaises(ValueError):
                blinding.ReviewAssignment.from_bytes(canonical_bytes({**data, key: 'private'}))
        for pseudonym in ('', '../r', 'r@example.test', 'x' * 65, '<script>', 'x\n', 'a/b', 'name surname'):
            with self.subTest(pseudonym=pseudonym), self.assertRaises(ValueError):
                self.assignment(pseudonym)
        for path in ('media/groove-on.wav', 'private/s00.wav', 'media/../s00.wav', 'https://host/s00.wav',
                     'media/s00.wav'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                replace(doc.stimuli[0], media=replace(doc.stimuli[0].media, path=path))
        for altered in (replace(doc, bundle_sha256='0' * 64),
                        replace(doc, stimuli=(replace(doc.stimuli[0], media=replace(doc.stimuli[0].media,
                                                                                 sha256='0' * 64)), *doc.stimuli[1:])),
                        replace(doc, panels=(replace(doc.panels[0], stimuli=tuple(reversed(doc.panels[0].stimuli))),
                                             *doc.panels[1:]))):
            with self.assertRaises(ValueError):
                altered.validate_bundle(self.bundle)
        for change in ({'stimuli': doc.stimuli[:-1]}, {'stimuli': (*doc.stimuli[:-1], doc.stimuli[0])},
                       {'panels': doc.panels[:-1]}, {'panels': (*doc.panels[:-1], doc.panels[2])}):
            with self.assertRaises(ValueError):
                replace(doc, **change)

    def test_generation_verifies_private_graph_and_media_and_requires_seed(self):
        from aimusic.audio import review_blinding as blinding
        with self.assertRaisesRegex(ValueError, 'digest'):
            blinding.create_assignment(self.bundle, 'synthetic-r01', root=self.root, expected_sha256='f' * 64)
        media = self.root / self.bundle.presentations[0].output.path
        media.write_bytes(b'[]')
        with self.assertRaisesRegex(ValueError, 'digest'):
            self.assignment()
        media.write_bytes(b'{}')
        private_path = self.root / self.bundle.blind_map.path
        private_path.write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError, 'digest'):
            self.assignment()
        legacy = replace(self.blind, order_seed=None)
        private_path.write_bytes(legacy.to_bytes())
        bundle = replace(self.bundle, blind_map=replace(self.bundle.blind_map, sha256=legacy.sha256, size=len(legacy.to_bytes())))
        with self.assertRaisesRegex(ValueError, 'seed'):
            self.assignment(bundle=bundle)

    def test_private_seed_is_validated_and_historical_maps_remain_readable(self):
        for seed in ('', 'abc', 'A' * 64, 'g' * 64, True):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                replace(self.blind, order_seed=seed)
        legacy = PrivateBlindMap.from_bytes(canonical_bytes({'conditions': self.blind.conditions}))
        self.assertEqual(legacy.conditions, self.blind.conditions)
        self.assertIsNone(legacy.order_seed)

    def test_seed_is_not_ignored_even_if_supplied_public_inputs_are_unchanged(self):
        from unittest.mock import patch
        from aimusic.audio import review_blinding as blinding
        # Boundary substitution isolates seed use from its usual bundle-hash change.
        original = [self.assignment(f'synthetic-r{i}').panels for i in range(12)]
        changed = replace(self.blind, order_seed='f' * 64)
        with patch.object(blinding.PrivateBlindMap, 'from_bytes', return_value=changed):
            self.assertNotEqual(original, [self.assignment(f'synthetic-r{i}').panels for i in range(12)])


class TestBlindMapWireCompatibility(unittest.TestCase):
    def test_legacy_canonical_bytes_and_trusted_digest_roundtrip_on_every_serialization_path(self):
        from aimusic.audio.review_bundle import digest
        raw = b'{"conditions":[["s00","fixture-12"]]}'
        trusted = '2c21da0e4737fa1cc87c3230967b89f76a09c37815565d2fcc3b92321c1d0793'
        blind = PrivateBlindMap.from_bytes(raw, expected_sha256=trusted)
        self.assertIsNone(blind.order_seed)
        self.assertEqual(blind.to_dict(), {'conditions': [['s00', 'fixture-12']]})
        self.assertEqual(blind.to_bytes(), raw)
        self.assertEqual(canonical_bytes(blind), raw)
        self.assertEqual(canonical_bytes({'map': blind}), b'{"map":' + raw + b'}')
        self.assertEqual(blind.digest, trusted)
        self.assertEqual(blind.sha256, trusted)
        self.assertEqual(digest(blind), trusted)
        self.assertEqual(PrivateBlindMap((('s00', 'fixture-12'),)).to_bytes(), raw)
        self.assertEqual(PrivateBlindMap.from_dict(blind.to_dict()).to_bytes(), raw)

    def test_seeded_extension_is_canonical_and_null_seed_wire_shape_is_rejected(self):
        from aimusic.audio.review_bundle import digest
        blind = PrivateBlindMap((('s00', 'fixture-12'),), order_seed='f' * 64)
        raw = b'{"conditions":[["s00","fixture-12"]],"order_seed":"' + b'f' * 64 + b'"}'
        trusted = hashlib.sha256(raw).hexdigest()
        self.assertEqual(blind.to_bytes(), raw)
        self.assertEqual(canonical_bytes(blind), raw)
        self.assertEqual(digest(blind), trusted)
        self.assertEqual(PrivateBlindMap.from_bytes(raw, expected_sha256=trusted), blind)
        self.assertNotEqual(replace(blind, order_seed=None).digest, trusted)
        for invalid in ({'conditions': blind.conditions, 'order_seed': None},
                        {'conditions': blind.conditions, 'extra': True}, {'order_seed': 'f' * 64}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                PrivateBlindMap.from_bytes(canonical_bytes(invalid))


class TestBlindedMediaIntegration(unittest.TestCase):
    def test_presentation_to_assignment_verifies_from_reviewer_only_tree_without_private_fields(self):
        import numpy as np
        from aimusic.audio import review_blinding as blinding
        from aimusic.audio.review_presentation import write_presentation
        from tests.test_audio_review_presentation import wav_bytes, chunks
        from tests.test_audio_review_bundle import ref

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            original = wav_bytes(np.full(80, .125))
            source_wav = ref('renders/secret-backend-groove-on/raw.wav', original)
            matrix = example_matrix()
            renders = {r.sha256: replace(r, reports=(*r.reports, source_wav)) for r in matrix.renders}
            matrix = replace(matrix, renders=tuple(renders.values()), exposures=tuple(
                replace(e, render_sha256=renders[e.render_sha256].sha256, wav=source_wav) for e in matrix.exposures))
            raw_path = root / source_wav.path
            raw_path.parent.mkdir(parents=True)
            raw_path.write_bytes(original)
            (root / 'media').mkdir()
            blind = blinding.create_blind_map(matrix, random_bytes=synthetic_entropy())
            renders = {r.sha256: r for r in matrix.renders}
            presentations = tuple(write_presentation(root, render=renders[e.render_sha256], source_wav=e.wav,
                                                     output=f'media/{s}.wav', start_sample=8, stop_sample=72)
                                  for e, (s, _) in zip(matrix.exposures, blind.conditions))
            bundle = replace(blinding.bind_presentations(matrix, blind, presentations),
                             assignment_directory='sessions/reviewers')
            for item in bundle.references():
                path = root / item.path
                if not path.exists():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(blind.to_bytes() if item == bundle.blind_map else b'{}')
            before = {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()}
            assignment = blinding.create_assignment(bundle, 'synthetic-r01', root=root, expected_sha256=bundle.sha256)
            self.assertEqual(before, {p.relative_to(root): p.read_bytes() for p in root.rglob('*') if p.is_file()})
            self.assertEqual(source_wav.read_verified(root), original)
            reference = assignment.reference(bundle)
            reviewer = root / 'reviewer-only'
            for item in assignment.stimuli:
                data = item.media.read_verified(root)
                self.assertTrue({name for name, _ in chunks(data)} <= {b'fmt ', b'fact', b'data'})
                target = reviewer / item.media.path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            path = reviewer / reference.document.path
            path.parent.mkdir(parents=True)
            path.write_bytes(assignment.to_bytes())
            loaded = blinding.ReviewAssignment.from_bytes(reference.document.read_verified(reviewer),
                                                          expected_sha256=reference.document.sha256)
            loaded.verify(reviewer, expected_sha256=reference.document.sha256)
            self.assertFalse((reviewer / 'private').exists())
            for file in reviewer.rglob('*'):
                self.assertNotIn('secret-backend', file.as_posix())
                if file.is_file():
                    for label in ('order_seed', 'conditions', 'groove-on', 'backend', 'raw.wav', 'private/'):
                        self.assertNotIn(label.encode(), file.read_bytes())
            changed = replace(loaded, reviewer='synthetic-r02')
            with self.assertRaisesRegex(ValueError, 'digest'):
                changed.verify(reviewer, expected_sha256=reference.document.sha256)
            (reviewer / loaded.stimuli[0].media.path).write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'digest'):
                loaded.verify(reviewer, expected_sha256=reference.document.sha256)
