"""Group 7 synthetic responses only: never evidence of actual listening."""
from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import importlib
import unittest

from tests.test_audio_review_archive import ArchiveCase


def completed(template, *, preference='A', defect=()):
    """Explicitly synthetic whole-assignment fixture, used only by tests."""
    return replace(template, state='complete', previous_finding_sha256=template.sha256,
                   singles=tuple(replace(s, audibility='audible', defects=defect,
                                         confidence='high', notes='Synthetic test response') for s in template.singles),
                   pairs=tuple(replace(p, perceived_difference='heard', defects_a=defect, defects_b=(),
                                       preference=preference, confidence='medium', notes='Synthetic test response')
                               for p in template.pairs))


def supplementary_notes(finding):
    """Valid per-field lengths; ASCII escaping expands beyond the wire budget."""
    notes = '🎵' * 1024
    return replace(finding, singles=tuple(replace(s, notes=notes) for s in finding.singles),
                   pairs=tuple(replace(p, notes=notes) for p in finding.pairs))


class FindingCase(ArchiveCase):
    def setUp(self):
        super().setUp()
        self.findings = importlib.import_module('aimusic.audio.review_findings')

    def template(self):
        return self.findings.create_template(self.assignment, root=self.root,
                                             expected_assignment_sha256=self.assignment.sha256)

    def verify(self, finding, **kwargs):
        return self.findings.validate_finding(finding, self.assignment, root=self.root,
                                              expected_assignment_sha256=self.assignment.sha256, **kwargs)


class TestFindingSchema(FindingCase):
    def test_fresh_template_exact_document_links_immutable_roundtrip_incomplete(self):
        before = self.bundle.to_bytes()
        finding = self.template()
        self.assertEqual(finding.schema, 'aimusic.review-finding/1')
        self.assertEqual(finding.assignment_document_sha256, self.assignment.sha256)
        self.assertNotEqual(finding.assignment_document_sha256, self.assignment.reference(self.bundle).sha256)
        self.assertEqual((finding.bundle_sha256, finding.reviewer), (self.bundle.sha256, self.assignment.reviewer))
        self.assertEqual((len(finding.singles), len(finding.pairs)), (8, 3))
        self.assertEqual(self.verify(finding), 'incomplete')
        self.assertEqual(self.verify(completed(finding)), 'complete')
        self.assertEqual(self.findings.ReviewFinding.from_bytes(finding.to_bytes()), finding)
        self.assertEqual(before, self.bundle.to_bytes())
        with self.assertRaises(FrozenInstanceError):
            finding.state = 'complete'
        media = {s.id: s for s in self.assignment.stimuli}
        self.assertEqual(tuple(s.stimulus for s in finding.singles),
                         tuple(media[s] for p in self.assignment.panels[:2] for s in p.stimuli))
        self.assertEqual(tuple((p.a, p.b) for p in finding.pairs),
                         tuple(tuple(media[s] for s in p.stimuli) for p in self.assignment.panels[2:]))

    def test_single_pair_enums_notes_null_empty_and_other_explanation(self):
        finding = self.template()
        for row, field, values in ((finding.singles[0], 'audibility', ('audible', 'partly_audible', 'inaudible', 'unsure', None)),
                                   (finding.pairs[0], 'perceived_difference', ('heard', 'not_heard', 'unsure', None)),
                                   (finding.pairs[0], 'preference', ('A', 'B', 'no_preference', 'unsure', None)),
                                   (finding.singles[0], 'confidence', ('low', 'medium', 'high', None))):
            for value in values:
                replace(row, **{field: value})
            for value in ('invalid', True, 1, [], {}):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    replace(row, **{field: value})
        for row, field in ((finding.singles[0], 'defects'), (finding.pairs[0], 'defects_a'), (finding.pairs[0], 'defects_b')):
            for value in (None, (), *[(d,) for d in self.findings.DEFECTS]):
                replace(row, **{field: value}, notes='Synthetic explanation')
            for value in (('unknown',), ('click_pop', 'click_pop'), ['dropout'], ('other',)):
                with self.assertRaises(ValueError):
                    replace(row, **{field: value})
            self.assertNotEqual(replace(row, **{field: None}).sha256, replace(row, **{field: ()}).sha256)
        for notes in ('x' * 1025, '\n', '\x7f', '', '   '):
            with self.assertRaises(ValueError):
                replace(finding.pairs[0], notes=notes)
        # Subjective combinations must not be "corrected" into consistency.
        replace(finding.pairs[0], perceived_difference='not_heard', preference='B', defects_a=('pitch_issue',))
        replace(finding.singles[0], audibility='inaudible', defects=(), confidence='high')

    def test_complete_requires_all_answers_withdrawal_requires_reason(self):
        full = completed(self.template())
        for group, fields in (('singles', ('audibility', 'defects', 'confidence')),
                              ('pairs', ('perceived_difference', 'defects_a', 'defects_b', 'preference', 'confidence'))):
            for field in fields:
                rows = getattr(full, group)
                with self.subTest(group=group, field=field), self.assertRaises(ValueError):
                    replace(full, **{group: (replace(rows[0], **{field: None}), *rows[1:])})
        for state in ('accepted', 'approved', 'consensus', '', None):
            with self.assertRaises(ValueError):
                replace(full, state=state)
        with self.assertRaises(ValueError):
            replace(full, state='withdrawn')
        withdrawn = replace(full, state='withdrawn', withdrawal_reason='Synthetic withdrawal')
        self.assertEqual(self.verify(withdrawn), 'withdrawn')
        with self.assertRaises(ValueError):
            replace(full, withdrawal_reason='not withdrawn')

    def test_wrong_links_rows_order_unknown_duplicates_and_changed_media_fail(self):
        finding = self.template()
        for field in ('bundle_sha256', 'assignment_document_sha256', 'reviewer'):
            with self.assertRaises(ValueError):
                self.verify(replace(finding, **{field: 'f' * 64}))
        for reviewer in ('', 'x@example.test', '../x', 'x' * 65):
            with self.assertRaises(ValueError):
                replace(finding, reviewer=reviewer)
        wrong = replace(finding.singles[0].stimulus, media=replace(finding.singles[0].stimulus.media, sha256='f' * 64))
        from aimusic.audio.review_blinding import AssignedStimulus
        from aimusic.audio.review_bundle import FileRef
        unknown = AssignedStimulus('sff', FileRef('media/sff.wav', wrong.media.sha256, wrong.media.size))
        for stimulus in (wrong, unknown):
            with self.assertRaises(ValueError):
                self.verify(replace(finding, singles=(replace(finding.singles[0], stimulus=stimulus), *finding.singles[1:])))
        for changes in ({'singles': finding.singles[:-1]}, {'singles': (*finding.singles[:-1], finding.singles[0])},
                        {'singles': tuple(reversed(finding.singles))}, {'pairs': tuple(reversed(finding.pairs))},
                        {'pairs': (replace(finding.pairs[0], a=finding.pairs[0].b, b=finding.pairs[0].a), *finding.pairs[1:])}):
            with self.assertRaises(ValueError):
                self.verify(replace(finding, **changes))
        with self.assertRaises(ValueError):
            self.findings.validate_finding(finding, self.assignment, root=self.root, expected_assignment_sha256='0' * 64)
        (self.root / self.assignment.stimuli[0].media.path).write_bytes(b'changed media')
        with self.assertRaisesRegex(ValueError, 'digest'):
            self.verify(finding)


if __name__ == '__main__':
    unittest.main()


class TestFindingHistory(FindingCase):
    def history(self, *rows):
        return self.findings.verify_history(rows, self.assignment, root=self.root,
                                             expected_assignment_sha256=self.assignment.sha256)

    def test_chain_order_replay_and_transitions(self):
        first = self.template()
        second = replace(first, previous_finding_sha256=first.sha256,
                         singles=(replace(first.singles[0], notes='Synthetic partial'), *first.singles[1:]))
        full = completed(second)
        withdrawn = replace(full, previous_finding_sha256=full.sha256, state='withdrawn',
                            withdrawal_reason='Synthetic withdrawal')
        self.assertEqual(self.history(full, first, second, full), (first, second, full))
        self.assertEqual(self.history(withdrawn, full, second, first)[-1], withdrawn)
        early = replace(first, previous_finding_sha256=first.sha256, state='withdrawn', withdrawal_reason='Synthetic withdrawal')
        self.assertEqual(self.history(early, first), (first, early))
        for rows in ((), (full,), (first, replace(full, previous_finding_sha256='0' * 64)),
                     (first, second, completed(first)),  # fork
                     (first, replace(first, singles=second.singles)),  # two roots
                     (first, full, completed(full)),
                     (first, early, replace(full, previous_finding_sha256=early.sha256))):
            with self.subTest(states=[r.state for r in rows]), self.assertRaises(ValueError):
                self.history(*rows)
        from aimusic.audio.review_blinding import create_assignment
        other = create_assignment(self.bundle, 'synthetic-r02', root=self.root, expected_sha256=self.bundle.sha256)
        other_finding = self.findings.create_template(other, root=self.root, expected_assignment_sha256=other.sha256)
        with self.assertRaises(ValueError):
            self.history(first, other_finding)

    def test_append_external_create_only_idempotent_tamper_fails_and_bundle_archive_unchanged(self):
        before = self.assemble()
        files = {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        directory = self.base / 'findings'
        directory.mkdir()
        first = self.template()
        full = completed(first)

        def append(row, out=directory):
            return self.findings.append_finding(row, self.assignment, root=self.root,
                                                expected_assignment_sha256=self.assignment.sha256, directory=out)
        path = append(first)
        self.assertEqual(path.name, f'f{first.sha256}.json')
        self.assertEqual(path.read_bytes(), first.to_bytes())
        self.assertEqual(append(first), path)
        append(full)
        self.assertEqual(len(list(directory.iterdir())), 2)
        with self.assertRaises(ValueError):
            append(completed(full))
        self.assertEqual(self.findings.read_history(directory), tuple(sorted((first, full), key=lambda f: f.sha256)))
        self.assertEqual(before, self.assemble())
        self.assertEqual(files, {str(p.relative_to(self.root)): p.read_bytes() for p in self.root.rglob('*') if p.is_file()})
        for out in (self.root, self.root / 'media'):
            with self.assertRaises(ValueError):
                append(first, out)
        path.write_bytes(full.to_bytes())
        with self.assertRaises(ValueError):
            append(first)

    def test_strict_canonical_bounded_parsing_and_history_inventory(self):
        import hashlib
        from aimusic.audio.review_bundle import canonical_bytes
        first = self.template()
        raw = first.to_bytes()
        for bad in (raw + b'\n', b'\xef\xbb\xbf' + raw, b'[]', b'\xff', b'[' * 1100,
                    b'{"schema":1,"schema":2}', b' ' * (128 * 1024 + 1),
                    canonical_bytes({**first.to_dict(), 'approved': True})):
            with self.assertRaises(ValueError):
                self.findings.ReviewFinding.from_bytes(bad)
        altered = replace(first, reviewer='synthetic-other').to_bytes()
        with self.assertRaises(ValueError):
            self.findings.ReviewFinding.from_bytes(altered, expected_sha256=hashlib.sha256(raw).hexdigest())
        directory = self.base / 'history'
        directory.mkdir()
        for name in ('extra.json', f'f{first.sha256.upper()}.json', '.stems-residue'):
            path = directory / name
            path.write_bytes(raw)
            with self.assertRaises(ValueError):
                self.findings.read_history(directory)
            path.unlink()


class TestFindingLimits(FindingCase):
    def test_oversized_serialization_and_append_reject_without_mutation(self):
        from aimusic.audio.review_bundle import canonical_bytes
        first = self.template()
        for finding in (supplementary_notes(first), supplementary_notes(completed(first))):
            with self.subTest(state=finding.state):
                directory = self.base / finding.state
                directory.mkdir()
                if finding.state == 'complete':
                    (directory / f'f{first.sha256}.json').write_bytes(first.to_bytes())
                self.assertGreater(len(canonical_bytes(finding.to_dict())), self.findings.MAX_DOCUMENT_BYTES)
                before = {p.name: p.read_bytes() for p in directory.iterdir()}
                with self.assertRaisesRegex(ValueError, 'document byte budget'):
                    self.findings.append_finding(finding, self.assignment, root=self.root,
                                                expected_assignment_sha256=self.assignment.sha256, directory=directory)
                self.assertEqual(before, {p.name: p.read_bytes() for p in directory.iterdir()})
                with self.assertRaisesRegex(ValueError, 'document byte budget'):
                    finding.to_bytes()

    def test_128_ordinary_snapshots_reserve_one_final_withdrawal_and_replay(self):
        rows = [self.template()]
        for _ in range(127):
            rows.append(replace(rows[-1], previous_finding_sha256=rows[-1].sha256))
        directory = self.base / 'full-history'
        directory.mkdir()
        for row in rows:
            (directory / f'f{row.sha256}.json').write_bytes(row.to_bytes())

        def append(row):
            return self.findings.append_finding(row, self.assignment, root=self.root,
                                                expected_assignment_sha256=self.assignment.sha256, directory=directory)

        before = {p.name: p.read_bytes() for p in directory.iterdir()}
        ordinary = completed(rows[-1])  # Otherwise a legal incomplete -> complete transition.
        with self.assertRaisesRegex(ValueError, 'budget'):
            append(ordinary)
        self.assertEqual(before, {p.name: p.read_bytes() for p in directory.iterdir()})
        excess = directory / f'f{ordinary.sha256}.json'
        excess.write_bytes(ordinary.to_bytes())
        with self.assertRaisesRegex(ValueError, 'budget'):
            self.findings.read_history(directory)
        excess.unlink()
        withdrawn = replace(rows[-1], previous_finding_sha256=rows[-1].sha256, state='withdrawn',
                            withdrawal_reason='Synthetic withdrawal at capacity')
        path = append(withdrawn)
        self.assertEqual(path.read_bytes(), withdrawn.to_bytes())
        self.assertEqual(append(withdrawn), path)  # Replay at 129 unique records still succeeds.
        self.assertEqual(len(self.findings.read_history(directory)), 129)
        self.assertEqual(len(list(directory.iterdir())), 129)
        with self.assertRaises(ValueError):
            append(replace(withdrawn, previous_finding_sha256=withdrawn.sha256))

    def test_complete_at_capacity_can_still_be_withdrawn(self):
        from unittest.mock import patch
        first = self.template()
        full = completed(first)
        withdrawn = replace(full, state='withdrawn', previous_finding_sha256=full.sha256,
                            withdrawal_reason='Synthetic withdrawal')
        with patch.object(self.findings, 'MAX_HISTORY', 2):
            chain = self.findings.verify_history((first, full, withdrawn, withdrawn), self.assignment, root=self.root,
                                                expected_assignment_sha256=self.assignment.sha256)
            self.assertEqual(chain, (first, full, withdrawn))

    def test_hard_linked_history_rejected_without_mutating_either_name(self):
        import os
        first = self.template()
        source = self.base / 'external-copy.json'
        source.write_bytes(first.to_bytes())
        directory = self.base / 'aliased-history'
        directory.mkdir()
        alias = directory / f'f{first.sha256}.json'
        os.link(source, alias)  # Local APFS/NTFS hard links need no symlink privilege.
        self.assertGreater(alias.stat().st_nlink, 1)
        with self.assertRaisesRegex(ValueError, 'hard.link'):
            self.findings.read_history(directory)
        with self.assertRaisesRegex(ValueError, 'hard.link'):
            self.findings.append_finding(completed(first), self.assignment, root=self.root,
                                        expected_assignment_sha256=self.assignment.sha256, directory=directory)
        self.assertEqual(list(directory.iterdir()), [alias])
        self.assertEqual(source.read_bytes(), first.to_bytes())
        self.assertEqual(alias.read_bytes(), first.to_bytes())

    def test_multi_history_exact_byte_and_file_budgets_and_inventory(self):
        from unittest.mock import patch
        from aimusic.audio import review_bundle
        first = self.template()
        raw = first.to_bytes()
        one, two = self.base / 'one', self.base / 'two'
        for directory in (one, two):
            directory.mkdir()
            (directory / f'f{first.sha256}.json').write_bytes(raw)
        for max_bytes, max_files, message in ((len(raw), 2, 'byte'), (2 * len(raw), 1, 'file'),
                                             (2 * len(raw) - 1, 2, 'byte')):
            with self.subTest(message=message, max_bytes=max_bytes), \
                    patch.object(review_bundle, 'read_relative', wraps=review_bundle.read_relative) as reads, \
                    patch.object(self.findings.ReviewFinding, 'from_bytes', wraps=self.findings.ReviewFinding.from_bytes) as parses:
                with self.assertRaisesRegex(ValueError, 'aggregate history ' + message + ' budget'):
                    self.findings.read_histories((one, two), max_bytes=max_bytes, max_files=max_files)
                self.assertEqual(reads.call_count, 1)
                self.assertEqual(parses.call_count, 1)
        self.assertEqual(self.findings.read_histories((one, two), max_bytes=2 * len(raw), max_files=2), ((first,), (first,)))
        (two / 'unexpected.txt').write_bytes(b'residue')
        with self.assertRaisesRegex(ValueError, 'inventory'):
            self.findings.read_histories((one, two), max_bytes=3 * len(raw), max_files=3)
