"""Immutable review graph contracts; synthetic metadata, never human findings."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from aimusic.audio.review_bundle import (
    AssignmentReference, ExternalFindingReference, FileRef, Metric, Panel,
    PresentationNode, PrivateBlindMap, RenderNode, ReviewBundle, SourceNode,
    Stimulus, canonical_bytes, REQUIRED_METRICS, COMMON_GATES,
)


def ref(path='private/evidence.json', data=b'{}'):
    return FileRef(path, hashlib.sha256(data).hexdigest(), len(data))


def metrics(control=False):
    rows = [Metric(n, 'unmeasured', 'not run', None, (), {}) for n in REQUIRED_METRICS]
    rows += [Metric(n, 'passed', 'synthetic test evidence', 'test', (ref(),), {}) for n in COMMON_GATES]
    rows += [Metric('reconciliation', 'not_applicable', 'imported MIDI', None, (), {}),
             Metric('controlled_tuning', 'failed' if control else 'not_applicable',
                    'calibration' if control else 'no controlled windows', 'test' if control else None,
                    (ref(),) if control else (), {}),
             Metric('negative_control', 'passed' if control else 'not_applicable',
                    'calibration' if control else 'not a control', 'test' if control else None,
                    (ref(),) if control else (), {})]
    return tuple(rows)


def example_bundle():
    source = SourceNode('imported_midi', (ref('private/source.mid'),), {'midi_sha256': ref().sha256})
    render = RenderNode(source.sha256, ref('private/render.json'), (ref(), ref('private/raw.wav')),
                        {'render': {'sample_rate': 48000, 'backend': 'simple', 'allow_simple_fallback': False,
                                    'routing': {}, 'release_tail_sec': .25, 'mix_peak': .95},
                         'stages': {'analysis': True, 'expressivization': False, 'render': True,
                                    'prompts': False, 'restyle': False, 'scoring': False, 'mixmaster': False},
                         'paths': {}},
                        {'requested_backend': 'native', 'effective_backend': 'native', 'fallback_reason': None},
                        {'python': '3.11', 'platform': 'test', 'dependencies': {'numpy': 'test'}},
                        ({'requested_backend': 'simple', 'effective_backend': 'simple', 'soundfont_sha256': None,
                          'fallback_reason': None, 'required_tuning': 'direct_12', 'source_track_index': 1,
                          'role': 'lead', 'role_source': 'inferred', 'tuning_evidence': None,
                          'backend': {'name': 'simple', 'version': '3', 'code_digest': 'a' * 64,
                                      'native_versions': [['numpy', 'test']], 'settings': [],
                                      'capabilities': {'tuning_modes': ['direct_12'], 'revision': '1',
                                                       'instrument_fidelity': False}}},), metrics())
    valid = replace(render, metrics=tuple(replace(m, state='passed', method='test', evidence=(ref(),))
                    if m.name == 'controlled_tuning' else m for m in render.metrics))
    negative = replace(render, metrics=metrics(control=True))
    renders = (render, valid, negative)
    presentations = tuple(PresentationNode((negative if i == 13 else valid if i in (2, 12) else render).sha256, ref('private/raw.wav'),
                          ref(f'media/s{i:02d}.wav'), 0, 100, 1.0) for i in range(14))
    stimuli = tuple(Stimulus(f's{i:02d}', p.sha256) for i, p in enumerate(presentations))
    panels = tuple(Panel(name, tuple(s.id for s in stimuli[a:b])) for name, a, b in (
        ('canonical', 0, 4), ('roles', 4, 8), ('parity', 8, 10), ('timing', 10, 12), ('tuning', 12, 14)))
    return ReviewBundle('aimusic.review-bundle/1', (source,), renders, presentations, stimuli, panels,
                        ref('private/blind-map.json'), 'assignments', 'findings')


def materialize(bundle, root):
    blind = PrivateBlindMap(tuple((s.id, 'flattened-12-calibration-control' if i == 13 else 'condition-' + str(i))
                                 for i, s in enumerate(bundle.stimuli)))
    bundle = replace(bundle, blind_map=ref('private/blind-map.json', blind.to_bytes()))
    for item in bundle.references():
        path = root / item.path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blind.to_bytes() if item == bundle.blind_map else b'{}')
    return bundle


class TestReviewSchema(unittest.TestCase):
    def test_round_trip_byte_stable_deeply_immutable_and_metadata_changes_identity(self):
        bundle = example_bundle()
        parsed = ReviewBundle.from_bytes(bundle.to_bytes(), expected_sha256=bundle.sha256)
        self.assertEqual(parsed, bundle)
        self.assertEqual(parsed.to_bytes(), bundle.to_bytes())
        with self.assertRaises(FrozenInstanceError):
            bundle.schema = 'other'
        with self.assertRaises(TypeError):
            bundle.renders[0].effective_config['render']['sample_rate'] = 1
        changed = replace(bundle.sources[0], files=(FileRef('private/source.mid', 'b' * 64, 2),),
                          identity={'midi_sha256': 'b' * 64})
        self.assertNotEqual(changed.sha256, bundle.sources[0].sha256)
        with self.assertRaisesRegex(ValueError, 'source'):
            replace(bundle, sources=(changed,))
        doc = json.loads(bundle.to_bytes())
        doc['assignment_directory'] = 'other'
        with self.assertRaisesRegex(ValueError, 'digest'):
            ReviewBundle.from_bytes(canonical_bytes(doc), expected_sha256=bundle.sha256)

    def test_required_fields_types_nonfinite_unknown_fields_and_duplicate_json_keys(self):
        bundle = example_bundle()
        doc = json.loads(bundle.to_bytes())
        for field in doc:
            changed = dict(doc)
            changed.pop(field)
            with self.subTest(field=field), self.assertRaises(ValueError):
                ReviewBundle.from_bytes(canonical_bytes(changed))
        for value in (float('nan'), float('inf'), True, '1'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                replace(bundle.presentations[0], gain=value)
        with self.assertRaises(ValueError):
            ReviewBundle.from_bytes(b'{"schema":"x","schema":"y"}')
        with self.assertRaises(ValueError):
            ReviewBundle.from_bytes(canonical_bytes({**doc, 'human_approved': True}))

    def test_paths_are_portable_and_contained_digest_is_sha256(self):
        for path in ('../x', '/x', 'C:/x', 'a\\b', 'a//b', './x', 'a/../b', 'a/CON.wav',
                     'a/x.', 'a/x ', 'a/x:ads', 'a/\x01', 'https://x', 'a/%2e%2e/x'):
            with self.subTest(path=path), self.assertRaises(ValueError):
                ref(path)
        for digest in ('abc', 'A' * 64, 'g' * 64):
            with self.assertRaises(ValueError):
                FileRef('a', digest, 1)
        with self.assertRaises(ValueError):
            replace(example_bundle(), blind_map=ref('media/blind.json'))

    def test_required_metrics_cannot_be_absent_or_inferred_from_humans(self):
        render = example_bundle().renders[0]
        for metric in render.metrics:
            with self.subTest(metric=metric.name), self.assertRaises(ValueError):
                replace(render, metrics=tuple(m for m in render.metrics if m != metric))
        for state in ('pending', '', 'pass', None):
            with self.assertRaises(ValueError):
                Metric('CLAP', state, 'not run', None, (), {})
        with self.assertRaises(ValueError):
            Metric('CLAP', 'passed', 'heard it', 'human', (ref(),), {})
        with self.assertRaises(ValueError):
            Metric('CLAP', 'unmeasured', 'not run', None, (), {'value': 0})
        with self.assertRaises(ValueError):
            Metric('LUFS', 'passed', 'measured', 'scorer', (), {'value': -14})

    def test_failed_gate_cannot_enter_public_projection_and_no_provenance_leaks(self):
        bundle = example_bundle()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = materialize(bundle, root)
            projection = bundle.public_projection(root=root, expected_sha256=bundle.sha256)
        text = canonical_bytes(projection).decode()
        for secret in ('private/', 'source', 'render', 'soundfont', 'effective_backend', 'blind-map'):
            self.assertNotIn(secret, text)
        rows = tuple(replace(m, state='failed') if m.name == 'finite_audio' else m
                     for m in bundle.renders[0].metrics)
        bad = replace(bundle.renders[0], metrics=rows)
        presentations = tuple(replace(p, render_sha256=bad.sha256) if p.render_sha256 == bundle.renders[0].sha256 else p
                              for p in bundle.presentations)
        stimuli = tuple(replace(s, presentation_sha256=p.sha256) for s, p in zip(bundle.stimuli, presentations))
        rejected = replace(bundle, renders=(bad, *bundle.renders[1:]), presentations=presentations, stimuli=stimuli)
        with self.assertRaisesRegex(ValueError, 'finite_audio'):
            rejected.public_projection(root=Path('nonexistent'), expected_sha256=rejected.sha256)

    def test_references_findings_are_external_without_bundle_digest_cycle(self):
        bundle = example_bundle()
        assignment = AssignmentReference(bundle.sha256, 'reviewer-01', ref('assignments/a.json'),
                                         tuple((s.id, p.output.sha256) for s, p in zip(bundle.stimuli, bundle.presentations)))
        assignment.validate_bundle(bundle)
        finding = ExternalFindingReference(bundle.sha256, assignment.sha256, 'reviewer-01', ref('findings/f.json'))
        finding.validate_assignment(assignment)
        before = bundle.to_bytes()
        with self.assertRaises(ValueError):
            replace(finding, bundle_sha256='f' * 64).validate_assignment(assignment)
        self.assertEqual(bundle.to_bytes(), before)
        blind = PrivateBlindMap(tuple((s.id, 'flattened-12-calibration-control' if i == 13 else 'condition-' + str(i)) for i, s in enumerate(bundle.stimuli)))
        blind.validate_bundle(bundle)
        with self.assertRaises(ValueError):
            PrivateBlindMap((('unknown', 'control'),)).validate_bundle(bundle)

    def test_file_verification_detects_tamper_missing_and_symlink(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'a').write_bytes(b'{}')
            item = ref('a')
            self.assertEqual(item.read_verified(root), b'{}')
            (root / 'a').write_bytes(b'[]')
            with self.assertRaisesRegex(ValueError, 'digest'):
                item.read_verified(root)
            (root / 'a').unlink()
            with self.assertRaises((OSError, ValueError)):
                item.read_verified(root)

    def test_doc_example_uses_same_validator(self):
        path = Path(__file__).resolve().parents[1] / 'docs/audio-review-bundle-example.json'
        ReviewBundle.from_bytes(path.read_bytes())


class TestRenderIdentityValidation(unittest.TestCase):
    def setUp(self):
        self.render = example_bundle().renders[0]
        self.doc = self.render.to_dict()

    def assert_invalid(self, doc):
        # Both public entry points must reject, without relying on bundle admission.
        for mode in ('constructor', 'serialized'):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                if mode == 'serialized':
                    RenderNode.from_bytes(canonical_bytes(doc))
                else:
                    replace(self.render, effective_config=doc['effective_config'],
                            application_backend=doc['application_backend'], runtime=doc['runtime'],
                            renderers=tuple(doc['renderers']))

    def test_required_identity_objects_cannot_be_junk_truthy_or_null(self):
        paths = [('effective_config',), ('application_backend',), ('runtime',), ('renderers', 0),
                 ('renderers', 0, 'backend'), ('renderers', 0, 'backend', 'capabilities')]
        for path in paths:
            for value in ({'junk': True}, None, True, 'identity', 1, []):
                doc = deepcopy(self.doc)
                target = doc
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = value
                with self.subTest(path=path, value=value):
                    self.assert_invalid(doc)

    def test_required_nested_fields_missing_null_or_unknown_rejected(self):
        paths = [('effective_config',), ('application_backend',), ('runtime',), ('renderers', 0),
                 ('renderers', 0, 'backend'), ('renderers', 0, 'backend', 'capabilities'),
                 ('effective_config', 'render'), ('effective_config', 'stages')]
        nullable = {'fallback_reason', 'soundfont_sha256', 'tuning_evidence'}
        for path in paths:
            target = self.doc
            for key in path:
                target = target[key]
            for field in (*target, 'unknown'):
                for null in (False, True):
                    if null and field in nullable:
                        continue
                    doc = deepcopy(self.doc)
                    changed = doc
                    for key in path:
                        changed = changed[key]
                    if null or field == 'unknown':
                        changed[field] = None
                    else:
                        del changed[field]
                    with self.subTest(path=path, field=field, null=null):
                        self.assert_invalid(doc)

    def test_identity_scalar_types_are_not_coerced(self):
        def leaves(value, path=()):
            if isinstance(value, dict):
                for key, child in value.items():
                    yield from leaves(child, (*path, key))
            elif isinstance(value, list):
                for key, child in enumerate(value):
                    yield from leaves(child, (*path, key))
            else:
                yield path, value
        for section in ('effective_config', 'application_backend', 'runtime', 'renderers'):
            for path, value in leaves(self.doc[section], (section,)):
                doc = deepcopy(self.doc)
                target = doc
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = 1 if type(value) is bool else True
                with self.subTest(path=path):
                    self.assert_invalid(doc)

    def test_versions_maps_capabilities_settings_and_bounds_are_validated(self):
        invalid = [
            (('runtime', 'python'), ''), (('runtime', 'platform'), ' '),
            (('runtime', 'dependencies'), {}), (('runtime', 'dependencies'), {'numpy': None}),
            (('runtime', 'dependencies'), {'': '1'}), (('runtime', 'python'), 'x' * 4097),
            (('renderers', 0, 'backend', 'version'), 'bad\nversion'),
            (('renderers', 0, 'backend', 'code_digest'), 'x' * 64),
            (('renderers', 0, 'soundfont_sha256'), 'ABC'),
            (('renderers', 0, 'backend', 'native_versions'), {'numpy': '1'}),
            (('renderers', 0, 'backend', 'native_versions'), []),
            (('renderers', 0, 'backend', 'native_versions'), [['numpy', '1'], ['numpy', '2']]),
            (('renderers', 0, 'backend', 'native_versions'), [['numpy', None]]),
            (('renderers', 0, 'backend', 'settings'), [['gain', True]]),
            (('renderers', 0, 'backend', 'settings'), [['gain', None]]),
            (('renderers', 0, 'backend', 'settings'), [['gain', .1], ['gain', .2]]),
            (('renderers', 0, 'backend', 'capabilities', 'tuning_modes'), []),
            (('renderers', 0, 'backend', 'capabilities', 'tuning_modes'), ['direct_12', 'direct_12']),
            (('renderers', 0, 'backend', 'capabilities', 'tuning_modes'), ['anything']),
            (('renderers', 0, 'required_tuning'), 'anything'),
            (('renderers', 0, 'source_track_index'), -1),
            (('effective_config', 'render', 'sample_rate'), 0),
            (('effective_config', 'render', 'mix_peak'), 1.1),
            (('effective_config', 'render', 'routing'), {'lead': {'default': True}}),
            (('effective_config', 'groove'), {'spec': {'swing': {'ratio': True}}}),
            (('effective_config', 'budget'), {'max_endpoint_calls': True, 'max_usd_estimate': 1}),
        ]
        for path, value in invalid:
            doc = deepcopy(self.doc)
            target = doc
            for key in path[:-1]:
                target = target[key]
            target[path[-1]] = value
            with self.subTest(path=path, value=value):
                self.assert_invalid(doc)

    def test_backend_and_fallback_consistency(self):
        for section in ('application_backend', 'renderer'):
            for patch in ({'effective_backend': 'other'}, {'fallback_reason': 'unnecessary'},
                          {'requested_backend': 'fluidsynth', 'fallback_reason': None}):
                doc = deepcopy(self.doc)
                target = doc['renderers'][0] if section == 'renderer' else doc[section]
                target.update(patch)
                with self.subTest(section=section, patch=patch):
                    self.assert_invalid(doc)
        for patch in ({'name': 'fluidsynth'}, {'capabilities': {'tuning_modes': ['channel_pitch'],
                     'revision': '1', 'instrument_fidelity': False}}):
            doc = deepcopy(self.doc)
            doc['renderers'][0]['backend'].update(patch)
            self.assert_invalid(doc)
        doc = deepcopy(self.doc)
        doc['effective_config']['render']['allow_simple_fallback'] = True
        doc['effective_config']['render']['backend'] = 'fluidsynth'
        doc['renderers'][0].update(requested_backend='fluidsynth', fallback_reason='missing optional renderer')
        RenderNode.from_bytes(canonical_bytes(doc))  # Explicit fallback metadata is a valid identity, not admission.


class TestGraphAdmissionBoundaries(unittest.TestCase):
    def test_source_identity_and_render_media_cannot_be_unlinked(self):
        bundle = example_bundle()
        with self.assertRaisesRegex(ValueError, 'source'):
            replace(bundle.sources[0], identity={'midi_sha256': '0' * 64})
        altered = replace(bundle.presentations[0], source_wav=ref('private/unrelated.wav'))
        with self.assertRaisesRegex(ValueError, 'render'):
            replace(bundle, presentations=(altered, *bundle.presentations[1:]))

    def test_tuning_panel_cannot_omit_applicable_tuning_gates(self):
        bundle = example_bundle()
        presentations = tuple(replace(p, render_sha256=bundle.renders[0].sha256) for p in bundle.presentations)
        stimuli = tuple(replace(s, presentation_sha256=p.sha256) for s, p in zip(bundle.stimuli, presentations))
        with self.assertRaisesRegex(ValueError, 'tuning'):
            replace(bundle, presentations=presentations, stimuli=stimuli)

    def test_every_referenced_byte_and_private_control_map_checked_before_public(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = materialize(example_bundle(), root)
            bundle.verify(root, expected_sha256=bundle.sha256)
            (root / 'private/raw.wav').write_bytes(b'[]')
            with self.assertRaisesRegex(ValueError, 'digest'):
                bundle.public_projection(root=root, expected_sha256=bundle.sha256)
            (root / 'private/raw.wav').write_bytes(b'{}')
            blind = PrivateBlindMap(tuple((s.id, 'ordinary-render') for s in bundle.stimuli))
            changed = replace(bundle, blind_map=ref('private/blind-map.json', blind.to_bytes()))
            (root / changed.blind_map.path).write_bytes(blind.to_bytes())
            with self.assertRaisesRegex(ValueError, 'control'):
                changed.verify(root, expected_sha256=changed.sha256)
