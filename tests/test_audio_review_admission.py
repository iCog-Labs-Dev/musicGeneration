"""Candidate admission replays existing evidence, never trusts success flags."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from aimusic.audio.config import load_audio_config
from aimusic.audio.demo_sources import build_source, write_source
from aimusic.audio.from_midi import from_midi
from aimusic.audio.orchestrator import Orchestrator
from aimusic.audio.review_bundle import FileRef, RenderNode
from aimusic.audio.review_admission import CandidateRequest, read_candidate, controlled_tuning
from aimusic.audio.scoring.tuning_check import ToneWindow, analyze_tuning


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestControlledAdmission(unittest.TestCase):
    def setUp(self):
        _, self.intent = build_source('reference-19')
        self.intent_bytes = json.dumps(self.intent).encode()
        self.intent_sha = hashlib.sha256(self.intent_bytes).hexdigest()
        self.rate = self.intent['sample_rate']
        self.good = np.zeros(228000)
        self.flat = self.good.copy()
        for window in self.intent['windows']:
            a, b, hz = window['start_sample'], window['stop_sample'], window['expected_hz']
            time = np.arange(b - a) / self.rate
            self.good[a:b] = .2 * np.sin(2 * np.pi * hz * time)
            nearest = 440 * 2 ** (round(12 * np.log2(hz / 440)) / 12)
            self.flat[a:b] = .2 * np.sin(2 * np.pi * nearest * time)

    def check(self, audio, **kw):
        return controlled_tuning(audio, self.rate, self.intent_bytes,
                                 expected_sha256=self.intent_sha, **kw)

    def test_positive_and_flattened_against_unchanged_targets(self):
        good = self.check(self.good)
        self.assertTrue(good.admitted)
        self.assertTrue(good.report.passed)
        negative = self.check(self.flat, flattened=True)
        self.assertTrue(negative.admitted)
        self.assertFalse(negative.report.passed)
        self.assertGreater(negative.report.p95_abs_cents, 15)
        self.assertEqual([r.window.expected_hz for r in negative.report.windows],
                         [w['expected_hz'] for w in self.intent['windows']])
        self.assertFalse(self.check(self.good, flattened=True).admitted)
        self.assertFalse(self.check(self.flat).admitted)

    def test_missing_window_and_insufficient_coverage_fail_even_for_control(self):
        windows = tuple(ToneWindow(**w) for w in self.intent['windows'])
        omitted = analyze_tuning(self.good, self.rate, windows[:-1])
        with self.assertRaisesRegex(ValueError, 'windows'):
            self.check(self.good, report=omitted)
        for flat in (False, True):
            for audio in (self.good[:1000], np.zeros_like(self.good)):
                self.assertFalse(self.check(audio, flattened=flat).admitted)

    def test_wrong_target_rate_tampered_report_and_intent_are_rejected(self):
        windows = tuple(ToneWindow(**w) for w in self.intent['windows'])
        wrong = tuple(replace(w, expected_hz=440) for w in windows)
        report = analyze_tuning(self.good, self.rate, wrong)
        with self.assertRaisesRegex(ValueError, 'windows'):
            self.check(self.good, report=report)
        correct = analyze_tuning(self.good, self.rate, windows)
        fabricated = replace(correct, windows=tuple(replace(r, measured_hz=r.window.expected_hz) for r in correct.windows))
        with self.assertRaisesRegex(ValueError, 'report'):
            self.check(self.good, report=fabricated)
        bad = deepcopy(self.intent)
        bad['notes'][0]['expected_hz'] = 440
        payload = json.dumps(bad).encode()
        with self.assertRaisesRegex(ValueError, 'target'):
            controlled_tuning(self.good, self.rate, payload, expected_sha256=hashlib.sha256(payload).hexdigest())
        with self.assertRaisesRegex(ValueError, 'digest'):
            controlled_tuning(self.good, self.rate, self.intent_bytes, expected_sha256='0' * 64)
        with self.assertRaisesRegex(ValueError, 'rate'):
            controlled_tuning(self.good, 16000, self.intent_bytes, expected_sha256=self.intent_sha)


class TestCandidateAdmission(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = write_source('fixture-12', self.root / 'source')
        config = load_audio_config().to_dict()
        config['stages']['expressivization'] = False
        config['render']['sample_rate'] = 8000
        config['render']['backend'] = 'simple'
        config['render']['routing'] = {}
        Orchestrator(config, self.root / 'cache').run(from_midi(self.source), output_dir=self.root / 'run')
        self.request = CandidateRequest('run', 'source/source.mid', 'imported_midi',
                                        {'midi_sha256': sha(self.source), 'input_kind': 'imported_midi'},
                                        sha(self.root / 'run/render.json'), 'simple', None,
                                        cache_manifest_sha256=sha(self.root / 'run/stems/cache_manifest.json'))

    def read(self):
        return read_candidate(self.root, self.request)

    def test_real_asset_free_candidate_gate_states_and_file_links(self):
        candidate = self.read()
        candidate.render.require_admitted()
        self.assertEqual(RenderNode.from_bytes(candidate.render.to_bytes()), candidate.render)
        self.assertEqual(candidate.source.identity['midi_sha256'], sha(self.source))
        states = {m.name: m.state for m in candidate.render.metrics}
        for name in ('CLAP', 'FAD', 'LUFS', 'rhythm_f_score', 'note_f1', 'polyphonic_f0'):
            self.assertEqual(states[name], 'unmeasured')
        self.assertEqual(states['reconciliation'], 'not_applicable')
        for item in candidate.render.reports:
            item.read_verified(self.root)

    def test_malformed_pinned_provenance_rejected_before_payload_reads(self):
        from aimusic.audio.review_bundle import read_relative
        manifest_path = self.root / 'run/render.json'
        original = json.loads(manifest_path.read_bytes())
        for section, value in (('effective_config', {'junk': True}),
                               ('application_backend', {'effective_backend': True}),
                               ('runtime', {'python': True, 'platform': True, 'dependencies': True}),
                               ('renderers', [{'junk': True}])):
            manifest_path.write_text(json.dumps({**original, section: value}))
            request = replace(self.request, manifest_sha256=sha(manifest_path))

            def guard(root, path, **kwargs):
                self.assertEqual(path, 'run/render.json', 'payload read before identity rejection')
                return read_relative(root, path, **kwargs)

            with self.subTest(section=section), patch('aimusic.audio.review_admission.read_relative', side_effect=guard):
                with self.assertRaises(ValueError):
                    read_candidate(self.root, request)

    def test_wrong_backend_asset_source_or_manifest_refused(self):
        for change in ({'expected_backend': 'fluidsynth'}, {'expected_asset_sha256': 'f' * 64},
                       {'expected_identity': {'midi_sha256': '0' * 64, 'input_kind': 'imported_midi'}},
                       {'manifest_sha256': '0' * 64}, {'cache_manifest_sha256': '0' * 64}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                read_candidate(self.root, replace(self.request, **change))

    def test_manifest_and_sidecar_tamper_fails_even_when_report_claims_success(self):
        index = json.loads((self.root / 'run/stems/role_index.json').read_bytes())
        sidecar = self.root / 'run/stems' / (index['stems'][0]['wav'] + '.json')
        data = json.loads(sidecar.read_bytes())
        data['frames'] += 1
        data['passed'] = True
        sidecar.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            self.read()

    def resign(self, *, approve=True):
        """Optionally reapprove fixture bytes to exercise semantic gates."""
        cache = self.root / 'run/stems/cache_manifest.json'
        doc = json.loads(cache.read_bytes())
        for name in doc['outputs']:
            doc['outputs'][name] = sha(cache.parent / name)
        cache.write_text(json.dumps(doc))
        if approve:
            self.request = replace(self.request, cache_manifest_sha256=sha(cache))

    def test_coordinated_audio_sidecar_mix_and_cache_tamper_is_not_approval(self):
        stems = self.root / 'run/stems'
        index = json.loads((stems / 'role_index.json').read_bytes())
        waves = []
        for row in index['stems']:
            wav = stems / row['wav']
            pcm, rate = sf.read(wav, always_2d=True)
            sf.write(wav, pcm * .5, rate, subtype='FLOAT')
            waves.append(sf.read(wav, always_2d=True)[0].mean(axis=1))
            sidecar = wav.with_name(wav.name + '.json')
            doc = json.loads(sidecar.read_bytes())
            doc['output_sha256'] = sha(wav)
            sidecar.write_text(json.dumps(doc))
        total = np.sum(np.stack(waves), axis=0)
        total *= min(1, .95 / float(np.max(np.abs(total))))
        sf.write(stems / 'rough_mix.wav', total, rate, subtype='FLOAT')
        self.resign(approve=False)
        with self.assertRaisesRegex(ValueError, 'cache manifest digest'):
            self.read()

    def assert_no_payload_read(self, message):
        # Approve malformed metadata only to exercise preflight, not digest gates.
        self.request = replace(self.request, cache_manifest_sha256=sha(self.root / 'run/stems/cache_manifest.json'))
        from aimusic.audio.review_bundle import read_relative
        metadata = {'run/render.json', 'run/stems/cache_manifest.json'}

        def guard(root, path, **kwargs):
            self.assertIn(path, metadata, 'payload read before inventory rejection')
            return read_relative(root, path, **kwargs)

        with patch('aimusic.audio.review_admission.read_relative', side_effect=guard):
            with self.assertRaisesRegex(ValueError, message):
                self.read()

    def test_casefold_aliases_rejected_as_collection_before_payload_reads(self):
        cache = self.root / 'run/stems/cache_manifest.json'
        doc = json.loads(cache.read_bytes())
        doc['outputs']['ROUGH_MIX.WAV'] = doc['outputs']['rough_mix.wav']
        cache.write_text(json.dumps(doc))
        self.assert_no_payload_read('casefold-alias')

    def test_cache_manifest_cannot_alias_one_of_its_own_outputs(self):
        cache = self.root / 'run/stems/cache_manifest.json'
        doc = json.loads(cache.read_bytes())
        doc['outputs']['CACHE_MANIFEST.JSON'] = '0' * 64
        cache.write_text(json.dumps(doc))
        self.assert_no_payload_read('casefold-alias')

    def test_missing_inventory_rejected_before_payload_reads(self):
        (self.root / 'run/stems/rough_mix.wav').unlink()
        self.assert_no_payload_read('complete render outputs')

    def test_extra_inventory_rejected_before_payload_reads(self):
        (self.root / 'run/stems/unexpected.wav').write_bytes(b'unapproved')
        self.assert_no_payload_read('complete render outputs')

    def test_output_count_limit_rejects_before_payload_reads(self):
        cache = self.root / 'run/stems/cache_manifest.json'
        doc = json.loads(cache.read_bytes())
        for i in range(65):
            doc['outputs'][f'extra-{i}.wav'] = '0' * 64
        cache.write_text(json.dumps(doc))
        self.assert_no_payload_read('output count')

    def test_cumulative_output_bytes_rejected_before_payload_reads(self):
        # Sparse files declare work without allocating/reading their payloads.
        cache = self.root / 'run/stems/cache_manifest.json'
        doc = json.loads(cache.read_bytes())
        for i in range(5):
            name = f'extra-{i}.wav'
            with (cache.parent / name).open('wb') as stream:
                stream.truncate(32 * 1024 * 1024)
            doc['outputs'][name] = '0' * 64
        cache.write_text(json.dumps(doc))
        self.assert_no_payload_read('total output bytes')

    def test_nonfinite_silence_peaks_and_alignment_refused_despite_new_hashes(self):
        wav = self.root / 'run/stems/rough_mix.wav'
        original = wav.read_bytes()
        for kind in ('nan', 'silence', 'loud', 'short', 'rate'):
            wav.write_bytes(original)
            pcm, rate = sf.read(wav)
            if kind == 'nan':
                pcm[10] = np.nan
            elif kind == 'silence':
                pcm[:] = 0
            elif kind == 'loud':
                pcm[10] = 1
            elif kind == 'short':
                pcm = pcm[:-1]
            else:
                rate += 1
            sf.write(wav, pcm, rate, subtype='FLOAT')
            self.resign()
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.read()

    def test_dropped_routed_events_cannot_be_hidden_by_rehash(self):
        import mido
        index = json.loads((self.root / 'run/stems/role_index.json').read_bytes())
        midi = self.root / 'run/stems' / index['stems'][0]['midi']
        events = mido.MidiFile(midi)
        for track in events.tracks:
            for message in track:
                if message.type == 'note_on':
                    message.velocity = 1
                    break
        events.save(midi)
        self.resign()
        with self.assertRaisesRegex(ValueError, 'event'):
            self.read()

    def test_source_unchanged_and_missing_reconciliation_refused(self):
        before = self.source.read_bytes()
        self.read()
        self.assertEqual(self.source.read_bytes(), before)
        (self.root / 'run/source.reconciliation.json').unlink()
        with self.assertRaises((OSError, ValueError)):
            self.read()


class TestPackageAndGrooveAdmission(unittest.TestCase):
    def test_package_reconciliation_recomputed_and_report_tamper_rejected(self):
        from tests.test_audio_reconcile import TestAudioReconcile
        from aimusic.audio.from_score import from_score
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            helper = TestAudioReconcile()
            helper.root = root
            context = helper.package()
            config = load_audio_config().to_dict()
            config['stages']['expressivization'] = False
            config['render'].update(sample_rate=8000, backend='simple', routing={})
            Orchestrator(config, root / 'cache').run(context, output_dir=root / 'run')
            manifest = json.loads((root / 'run/render.json').read_bytes())
            request = CandidateRequest('run', context.package.root.relative_to(root).as_posix(), 'render_package',
                                       manifest['source'], sha(root / 'run/render.json'), 'simple', None,
                                       cache_manifest_sha256=sha(root / 'run/stems/cache_manifest.json'))
            result = read_candidate(root, request)
            self.assertEqual(next(m.state for m in result.render.metrics if m.name == 'reconciliation'), 'passed')
            reconciliation = root / 'run/score.reconciliation.json'
            doc = json.loads(reconciliation.read_bytes())
            doc['checks'] = []
            doc['passed'] = True
            reconciliation.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, 'reconciliation'):
                read_candidate(root, request)
            from_score(context.package.root)  # original package remains valid

    def test_meaningful_groove_event_replay_and_report_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = write_source('fixture-12', root / 'source')
            config = load_audio_config().to_dict()
            config['render'].update(sample_rate=8000, backend='simple', routing={})
            config['groove'] = {'spec': {'swing': {'ratio': .6, 'subdivision': 8, 'applies_to': ['lead']}}}
            Orchestrator(config, root / 'cache').run(from_midi(source), output_dir=root / 'run')
            manifest = json.loads((root / 'run/render.json').read_bytes())
            request = CandidateRequest('run', 'source/source.mid', 'imported_midi', manifest['source'],
                                       sha(root / 'run/render.json'), 'simple', None,
                                       cache_manifest_sha256=sha(root / 'run/stems/cache_manifest.json'))
            read_candidate(root, request).render.require_admitted()
            report = root / 'run/source.expressive.mid.diff.json'
            doc = json.loads(report.read_bytes())
            doc['note_diffs'][0]['velocity_out'] += 1
            report.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, 'groove report'):
                read_candidate(root, request)

    def test_report_runtime_settings_and_role_index_membership_verified(self):
        helper = TestCandidateAdmission()
        helper.setUp()
        try:
            index_path = helper.root / 'run/stems/role_index.json'
            original = index_path.read_bytes()
            index = json.loads(original)
            index['roles'] = {'unrelated': ['missing-id']}
            index_path.write_text(json.dumps(index))
            helper.resign()
            with self.assertRaisesRegex(ValueError, 'routing'):
                helper.read()
            index_path.write_bytes(original)
            sidecar = helper.root / 'run/stems' / (index['stems'][0]['wav'] + '.json')
            doc = json.loads(sidecar.read_bytes())
            doc['render_settings']['synth.sample-rate'] = 999
            sidecar.write_text(json.dumps(doc))
            helper.resign()
            with self.assertRaisesRegex(ValueError, 'settings'):
                helper.read()
        finally:
            helper.doCleanups()

class TestNativeReportReader(unittest.TestCase):
    def setUp(self):
        self.helper = TestCandidateAdmission()
        self.helper.setUp()
        self.addCleanup(self.helper.doCleanups)
        self.root = self.helper.root
        (self.root / 'source').rename(self.root / 'fixture-12')
        (self.root / 'run').rename(self.root / 'fixture-12/cold')
        run = self.root / 'fixture-12/cold'
        self.data = {'schema': 'aimusic.native-listening-evidence/1', 'human_listening': 'accepted historical',
                     'checks_passed': True, 'workflows': {'fixture-12': {
                         'source': 'fixture-12/source.mid',
                         'source_files_sha256': {p.name: sha(p) for p in (self.root / 'fixture-12').iterdir() if p.is_file()},
                         'runs': {'cold': {'artifact_sha256': {p.relative_to(run).as_posix(): sha(p)
                                                             for p in run.rglob('*') if p.is_file()}}}}}}

    def reference(self):
        payload = json.dumps(self.data).encode()
        (self.root / 'evidence.json').write_bytes(payload)
        return FileRef('evidence.json', sha(self.root / 'evidence.json'), len(payload))

    def read(self, reference):
        from aimusic.audio.review_admission import read_native_candidate
        return read_native_candidate(self.root, reference, workflow='fixture-12', run='cold',
                                     expected_backend='simple', expected_asset_sha256=None)

    def test_pinned_native_report_links_sources_artifacts_and_never_inherits_humans(self):
        reference = self.reference()
        candidate = self.read(reference)
        candidate.render.require_admitted()
        self.assertTrue(all(m.state == 'unmeasured' for m in candidate.render.metrics if m.name == 'CLAP'))
        self.assertIn(reference, candidate.render.reports)
        (self.root / 'fixture-12/expected.json').write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError, 'digest'):
            self.read(reference)

    def test_retained_windows_paths_are_normalized_without_changing_report_pin(self):
        workflow = self.data['workflows']['fixture-12']
        workflow['source'] = workflow['source'].replace('/', '\\')
        artifacts = workflow['runs']['cold']['artifact_sha256']
        workflow['runs']['cold']['artifact_sha256'] = {p.replace('/', '\\'): sha for p, sha in artifacts.items()}
        reference = self.reference()
        before = reference.read_verified(self.root)
        candidate = self.read(reference)
        candidate.render.require_admitted()
        self.assertIn(reference, candidate.render.reports)
        self.assertTrue(all('\\' not in ref.path for ref in candidate.render.reports))
        self.assertEqual(reference.read_verified(self.root), before)

    def test_unsafe_native_paths_rejected_before_candidate_reads(self):
        paths = (r'C:\secret', r'C:secret', r'\\server\share\secret', r'\secret',
                 '/secret', '//server/share/secret', r'..\secret', r'a\..\secret',
                 r'a/..\secret', r'a\../secret', r'a/b\secret', r'a\\secret',
                 r'a\.\secret', r'a\secret:stream', r'a\%2e%2e\secret')
        for path in paths:
            for location in ('source', 'source_files_sha256', 'artifact_sha256'):
                original = deepcopy(self.data)
                workflow = self.data['workflows']['fixture-12']
                if location == 'source':
                    workflow['source'] = path
                elif location == 'source_files_sha256':
                    workflow[location][path] = '0' * 64
                else:
                    workflow['runs']['cold'][location][path] = '0' * 64
                reference = self.reference()
                with self.subTest(path=path, location=location):
                    with patch('aimusic.audio.review_admission._Reader.read') as read:
                        with self.assertRaises(ValueError):
                            self.read(reference)
                        read.assert_not_called()
                self.data = original

    def test_normalized_native_aliases_rejected_before_candidate_reads(self):
        workflow = self.data['workflows']['fixture-12']
        artifacts = workflow['runs']['cold']['artifact_sha256']
        artifacts[r'STEMS\cache_manifest.json'] = artifacts['stems/cache_manifest.json']
        with patch('aimusic.audio.review_admission._Reader.read') as read:
            with self.assertRaisesRegex(ValueError, 'alias'):
                self.read(self.reference())
            read.assert_not_called()

    def test_manifest_only_report_cannot_learn_cache_anchor_from_candidate(self):
        run = self.data['workflows']['fixture-12']['runs']['cold']
        run['artifact_sha256'] = {'render.json': run['artifact_sha256']['render.json']}
        with self.assertRaisesRegex(ValueError, 'pin.*cache manifest'):
            self.read(self.reference())

    def test_native_alias_inventory_refused_before_payload_reads(self):
        from aimusic.audio.review_bundle import read_relative
        cache = self.root / 'fixture-12/cold/stems/cache_manifest.json'
        doc = json.loads(cache.read_bytes())
        doc['outputs']['ROUGH_MIX.WAV'] = doc['outputs']['rough_mix.wav']
        cache.write_text(json.dumps(doc))
        self.data['workflows']['fixture-12']['runs']['cold']['artifact_sha256']['stems/cache_manifest.json'] = sha(cache)
        reference = self.reference()
        metadata = {'fixture-12/cold/render.json', 'fixture-12/cold/stems/cache_manifest.json'}

        def guard(root, path, **kwargs):
            self.assertIn(path, metadata, 'native payload read before inventory rejection')
            return read_relative(root, path, **kwargs)

        with patch('aimusic.audio.review_admission.read_relative', side_effect=guard):
            with self.assertRaisesRegex(ValueError, 'casefold-alias'):
                self.read(reference)


class TestAdmissionReadBudget(unittest.TestCase):
    def test_repeat_reads_consume_work_budget_and_refuse_before_next_read(self):
        from aimusic.audio.review_admission import _Reader
        from aimusic.audio.review_bundle import read_relative
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'file').write_bytes(b'1234')
            with patch('aimusic.audio.review_admission.MAX_READ_BYTES', 8):
                reader = _Reader(root)
                self.assertEqual(reader.read('file'), b'1234')
                self.assertEqual(reader.read('file'), b'1234')
                with patch('aimusic.audio.review_admission.read_relative', wraps=read_relative) as read:
                    with self.assertRaisesRegex(ValueError, 'read work budget'):
                        reader.read('file')
                    read.assert_not_called()


class TestAdmissionInventoryBounds(unittest.TestCase):
    def test_directory_enumeration_stops_at_first_excess_entry(self):
        from contextlib import nullcontext
        from types import SimpleNamespace
        from aimusic.audio.review_admission import _preflight_outputs

        def entries():
            for name in ('audio.wav', 'cache_manifest.json', 'unexpected.wav'):
                yield SimpleNamespace(name=name)
            self.fail('inventory enumeration continued beyond the bounded set')

        with patch('aimusic.audio.review_admission.os.scandir', return_value=nullcontext(entries())):
            with self.assertRaisesRegex(ValueError, 'complete render outputs'):
                _preflight_outputs(Path('unused'), {'audio.wav': '0' * 64})


class TestControlledSourceLink(unittest.TestCase):
    def test_intent_is_bound_to_original_events_and_not_flattened_original(self):
        from aimusic.audio.midi_events import MidiEvents
        from aimusic.audio.review_admission import validate_control_source
        payload, intent = build_source('reference-19')
        source = MidiEvents.from_bytes(payload)
        validate_control_source(source, intent)
        original = source.to_mido()
        for track in original.tracks:
            for index, message in enumerate(track):
                if message.type == 'pitchwheel':
                    track[index] = message.copy(pitch=0)
        with self.assertRaisesRegex(ValueError, 'target'):
            validate_control_source(MidiEvents.from_mido(original), intent)
        missing = deepcopy(intent)
        missing['notes'] = missing['notes'][:-1]
        missing['windows'] = missing['windows'][:-1]
        with self.assertRaisesRegex(ValueError, 'complete'):
            validate_control_source(source, missing)

class TestControlledCandidateIntegration(unittest.TestCase):
    def test_simulated_renderer_positive_and_calibration_admission(self):
        from unittest.mock import patch
        import mido
        from aimusic.audio.midi_events import MidiEvents
        from aimusic.audio.render import registry
        from aimusic.audio.render.registry import RendererBackend, RendererCapabilities
        from aimusic.audio.render.tuning import channel_pitch_evidence
        from aimusic.audio.render.util import render_frames

        def render(request):
            events = MidiEvents.load(request.midi_path)
            rows = channel_pitch_evidence(events)['notes']
            pcm = np.zeros(render_frames(request.sample_rate, request.duration_sec, request.release_tail_sec))
            for note, row in zip(events.notes, rows):
                a = round(float(events.tempo_map.seconds(note.start)) * request.sample_rate)
                b = round(float(events.tempo_map.seconds(note.end)) * request.sample_rate)
                pcm[a:b] = .2 * np.sin(2 * np.pi * row['hz'] * np.arange(b - a) / request.sample_rate)
            sf.write(request.output_path, pcm, request.sample_rate, subtype='FLOAT')
            return request.output_path

        backend = RendererBackend('fluidsynth', 'test', RendererCapabilities(('direct_12', 'channel_pitch')),
                                  'a' * 64, (('libfluidsynth', 'simulated-test-only'),), render)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(registry._BACKENDS, {'fluidsynth': lambda asset: backend}):
            root = Path(tmp)
            source = write_source('reference-19', root / 'source')
            font = root / 'not-an-asset.sf2'
            font.write_bytes(b'simulated test, not a licensed SoundFont')
            config = load_audio_config().to_dict()
            config['stages']['expressivization'] = False
            config['render'].update(sample_rate=48000, backend='fluidsynth', routing={})
            config['paths']['soundfont'] = str(font)
            intent = root / 'source/expected.json'
            intent_ref = FileRef('source/expected.json', sha(intent), intent.stat().st_size)
            original_ref = FileRef('source/source.mid', sha(source), source.stat().st_size)
            for flattened in (False, True):
                midi = source
                if flattened:
                    doc = mido.MidiFile(source)
                    for track in doc.tracks:
                        for msg in track:
                            if msg.type == 'pitchwheel':
                                msg.pitch = 0
                    midi = root / 'flat.mid'
                    doc.save(midi)
                out = 'flat' if flattened else 'valid'
                Orchestrator(config, root / 'cache').run(from_midi(midi), output_dir=root / out)
                manifest = json.loads((root / out / 'render.json').read_bytes())
                request = CandidateRequest(out, midi.relative_to(root).as_posix(), 'imported_midi', manifest['source'],
                                           sha(root / out / 'render.json'), 'fluidsynth', sha(font), intent_ref,
                                           original_ref if flattened else None, flattened,
                                           cache_manifest_sha256=sha(root / out / 'stems/cache_manifest.json'))
                candidate = read_candidate(root, request)
                states = {m.name: m.state for m in candidate.render.metrics}
                self.assertEqual(states['controlled_tuning'], 'failed' if flattened else 'passed')
                self.assertEqual(states['negative_control'], 'passed' if flattened else 'not_applicable')
                # A report with changed observations cannot pass just because its success flag does.
                observed = next(m for m in candidate.render.metrics if m.name == 'controlled_tuning').to_dict()['observations']
                observed['passed'] = not observed['passed']
                (root / out / 'tuning-check.json').write_text(json.dumps(observed))
                with self.assertRaisesRegex(ValueError, 'tuning report'):
                    read_candidate(root, request)
