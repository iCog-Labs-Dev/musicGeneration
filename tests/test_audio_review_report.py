"""Group 7 authorized reporting; all judgments and identities are synthetic fixtures."""
from dataclasses import replace
import importlib

from aimusic.audio.review_blinding import create_assignment
from tests.test_audio_review_findings import FindingCase, completed, supplementary_notes


class ReportCase(FindingCase):
    def setUp(self):
        super().setUp()
        self.reporting = importlib.import_module('aimusic.audio.review_report')
        self.other = create_assignment(self.bundle, 'synthetic-r02', root=self.root, expected_sha256=self.bundle.sha256)
        self.first = self.template()
        self.second = self.findings.create_template(self.other, root=self.root, expected_assignment_sha256=self.other.sha256)
        self.histories = ((self.first, completed(self.first)), (self.second, completed(self.second, preference='B', defect=('timing_issue',))))
        self.assignments = (self.assignment, self.other)
        self.closure = self.reporting.ReviewClosure('aimusic.review-closure/1', self.bundle.sha256,
            tuple(self.reporting.ClosureEntry(a.sha256, a.reviewer, h[-1].sha256) for a, h in zip(self.assignments, self.histories)))

    def report(self, **kwargs):
        defaults = dict(bundle=self.bundle, root=self.root, expected_bundle_sha256=self.bundle.sha256,
                        closure=self.closure, expected_closure_sha256=self.closure.sha256,
                        assignments=self.assignments, histories=self.histories, authorize_unblinding=True)
        return self.reporting.build_report(**{**defaults, **kwargs})


class TestReviewReport(ReportCase):
    def test_oversized_constructed_finding_cannot_enter_report(self):
        oversized = supplementary_notes(self.histories[0][-1])
        closure = replace(self.closure, roster=(replace(self.closure.roster[0], final_finding_sha256=oversized.sha256),
                                               self.closure.roster[1]))
        with self.assertRaisesRegex(ValueError, 'document byte budget'):
            self.report(closure=closure, expected_closure_sha256=closure.sha256,
                        histories=((self.first, oversized), self.histories[1]))

    def test_preserves_individual_disagreement_exact_metrics_provenance_and_limits(self):
        before = self.assemble()
        report = self.report()
        data = report.to_dict()
        self.assertEqual(report.to_bytes(), self.report(assignments=tuple(reversed(self.assignments)),
                                                       histories=tuple(reversed(self.histories))).to_bytes())
        self.assertEqual(data['closure_sha256'], self.closure.sha256)
        self.assertEqual(data['history_counts'], {'incomplete': 2, 'complete': 2, 'withdrawn': 0})
        self.assertEqual(data['final_counts'], {'incomplete': 0, 'complete': 2, 'withdrawn': 0})
        self.assertEqual(data['eligible_counts'], {'reviewers': 2, 'single_judgments': 16, 'pair_judgments': 6})
        self.assertEqual(data['renders'], [r.to_dict() for r in self.bundle.renders])
        self.assertEqual(data['sources'], [s.to_dict() for s in self.bundle.sources])
        self.assertEqual(data['presentations'], [p.to_dict() for p in self.bundle.presentations])
        self.assertEqual(data['reviewers'][0]['assignment'], self.assignment.to_dict())
        self.assertEqual(data['reviewers'][0]['findings'], [f.to_dict() for f in self.histories[0]])
        self.assertEqual(data['reviewers'][1]['findings'], [f.to_dict() for f in self.histories[1]])
        self.assertEqual(data['category_counts']['singles'][self.first.singles[0].stimulus.id]['defects'],
                         {'none': 1, 'timing_issue': 1})
        self.assertEqual(data['category_counts']['pairs']['parity']['perceived_difference'], {'heard': 2})
        for r in data['renders']:
            self.assertEqual(next(m for m in r['metrics'] if m['name'] == 'polyphonic_f0')['state'], 'unmeasured')
        controls = [s for s in data['stimuli'] if s['negative_control']]
        self.assertEqual(len(controls), 1)
        self.assertEqual(controls[0]['condition'], 'flattened-12-calibration-control')
        self.assertIn('calibration', data['negative_control_disclosure'])
        for phrase in ('rough-demo', 'No historical acceptance', 'No mastering', 'No production-quality', 'No inferred polyphonic-f0'):
            self.assertTrue(any(phrase in v for v in data['limitations']), phrase)
        text = report.to_bytes().decode()
        self.assertNotIn('order_seed', text)
        self.assertNotIn(self.blind.order_seed, text)
        for claim in ('"approved"', '"accepted"', '"consensus"', 'unanimous'):
            self.assertNotIn(claim, text)
        self.assertEqual(before, self.assemble())

    def test_pair_preferences_follow_exact_stimuli_across_orientations(self):
        reversed_pairs = [a.a.id != b.a.id for a, b in zip(self.first.pairs, self.second.pairs)]
        self.assertIn(True, reversed_pairs)
        self.assertIn(False, reversed_pairs)
        for preference in ('A', 'B', 'no_preference', 'unsure'):
            with self.subTest(preference=preference):
                right = completed(self.second, preference=preference)
                histories = (self.histories[0], (self.second, right))
                closure = replace(self.closure, roster=(self.closure.roster[0],
                                  replace(self.closure.roster[1], final_finding_sha256=right.sha256)))
                report = self.report(closure=closure, expected_closure_sha256=closure.sha256, histories=histories).to_dict()
                for pair, flipped in zip(self.first.pairs, reversed_pairs):
                    expected = {pair.a.id: 1}
                    if preference in ('A', 'B'):
                        selected = pair.b.id if (preference == 'A') == flipped else pair.a.id
                    else:
                        selected = preference
                    expected[selected] = expected.get(selected, 0) + 1
                    self.assertEqual(report['category_counts']['pairs'][pair.panel]['preference_stimulus_or_category'], expected)
                self.assertEqual(report['reviewers'][0]['findings'][-1], histories[0][-1].to_dict())
                self.assertEqual(report['reviewers'][1]['findings'][-1], right.to_dict())

    def test_pair_side_defects_and_disagreements_follow_exact_stimuli(self):
        left = replace(completed(self.first), pairs=tuple(replace(p, defects_a=('click_pop',), defects_b=('dropout',))
                                                        for p in completed(self.first).pairs))
        right = replace(completed(self.second), pairs=tuple(replace(p, defects_a=('pitch_issue',), defects_b=('distortion',),
                                                                   perceived_difference='not_heard', confidence='low')
                                                           for p in completed(self.second).pairs))
        histories = ((self.first, left), (self.second, right))
        closure = replace(self.closure, roster=tuple(replace(r, final_finding_sha256=h[-1].sha256)
                                                    for r, h in zip(self.closure.roster, histories)))
        report = self.report(closure=closure, expected_closure_sha256=closure.sha256, histories=histories).to_dict()
        self.assertTrue(any(a.a.id != b.a.id for a, b in zip(left.pairs, right.pairs)))
        for a, b in zip(left.pairs, right.pairs):
            flipped = a.a.id != b.a.id
            counts = report['category_counts']['pairs'][a.panel]
            self.assertEqual(counts['stimuli'], {
                a.a.id: {'defects': {'click_pop': 1, ('distortion' if flipped else 'pitch_issue'): 1}},
                a.b.id: {'defects': {'dropout': 1, ('pitch_issue' if flipped else 'distortion'): 1}},
            })
            self.assertEqual(counts['perceived_difference'], {'heard': 1, 'not_heard': 1})
            self.assertEqual(counts['confidence'], {'medium': 1, 'low': 1})
        self.assertEqual(report['reviewers'][0]['findings'][-1], left.to_dict())
        self.assertEqual(report['reviewers'][1]['findings'][-1], right.to_dict())

    def test_authorization_trust_substituted_map_and_changed_media_fail_closed(self):
        for kwargs in ({'authorize_unblinding': False}, {'authorize_unblinding': 1},
                       {'expected_bundle_sha256': 'f' * 64}, {'expected_closure_sha256': 'f' * 64}):
            with self.assertRaises(ValueError):
                self.report(**kwargs)
        path = self.root / self.bundle.blind_map.path
        path.write_bytes(replace(self.blind, order_seed='0' * 64).to_bytes())
        with self.assertRaisesRegex(ValueError, 'digest'):
            self.report()
        path.write_bytes(self.blind.to_bytes())
        (self.root / self.assignment.stimuli[0].media.path).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'digest'):
            self.report()

    def test_missing_incomplete_roster_forks_and_wrong_final_block_report(self):
        for kwargs in ({'assignments': self.assignments[:1], 'histories': self.histories[:1]},
                       {'assignments': (*self.assignments, self.assignment)},
                       {'histories': ((self.first,), self.histories[1])},
                       {'histories': ((*self.histories[0], completed(self.first, preference='B')), self.histories[1])}):
            with self.assertRaises(ValueError):
                self.report(**kwargs)
        incomplete = replace(self.closure, roster=(replace(self.closure.roster[0], final_finding_sha256=self.first.sha256),
                                                  self.closure.roster[1]))
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            self.report(closure=incomplete, expected_closure_sha256=incomplete.sha256,
                        histories=((self.first,), self.histories[1]))
        for rows in ((), (self.closure.roster[0], self.closure.roster[0])):
            with self.assertRaises(ValueError):
                replace(self.closure, roster=rows)

    def test_withdrawn_retained_but_no_judgments_from_incomplete_or_withdrawn(self):
        previous = self.histories[0][-1]
        withdrawn = replace(previous, state='withdrawn', previous_finding_sha256=previous.sha256,
                            withdrawal_reason='Synthetic withdrawal')
        closure = replace(self.closure, roster=(replace(self.closure.roster[0], final_finding_sha256=withdrawn.sha256),
                                               self.closure.roster[1]))
        report = self.report(closure=closure, expected_closure_sha256=closure.sha256,
                             histories=((*self.histories[0], withdrawn), self.histories[1])).to_dict()
        self.assertEqual(report['history_counts'], {'incomplete': 2, 'complete': 2, 'withdrawn': 1})
        self.assertEqual(report['final_counts'], {'incomplete': 0, 'complete': 1, 'withdrawn': 1})
        self.assertEqual(report['eligible_counts'], {'reviewers': 1, 'single_judgments': 8, 'pair_judgments': 3})
        self.assertEqual(report['category_counts']['singles'][self.first.singles[0].stimulus.id]['defects'], {'timing_issue': 1})
        self.assertEqual(report['reviewers'][0]['findings'][-1], withdrawn.to_dict())


class TestReviewReportCLI(ReportCase):
    def cli(self, *args):
        import subprocess
        import sys
        return subprocess.run([sys.executable, '-m', 'aimusic.audio.review_report', *map(str, args)],
                              capture_output=True, timeout=30)

    def test_close_generate_trusted_canonical_create_only_with_authorization(self):
        import json
        output = self.base / 'reports'
        output.mkdir()
        findings_root = self.base / 'histories'
        findings_root.mkdir()
        entries = []
        for assignment, history in zip(self.assignments, self.histories):
            path = self.root / assignment.reference(self.bundle).document.path
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(assignment.to_bytes())
            directory = findings_root / assignment.sha256
            directory.mkdir()
            for finding in history:
                (directory / f'f{finding.sha256}.json').write_bytes(finding.to_bytes())
            entries += ['--entry', assignment.sha256, assignment.reviewer, history[-1].sha256]
        bundle = self.root / 'bundle.json'
        bundle.write_bytes(self.bundle.to_bytes())
        result = self.cli('close', '--bundle-sha256', self.bundle.sha256, '--input-root', self.root,
                          '--out', output, *entries)
        self.assertEqual(result.returncode, 0, result.stderr)
        receipt = json.loads(result.stdout)
        self.assertEqual(receipt['closure_sha256'], self.closure.sha256)
        path = output / f'c{self.closure.sha256}.json'
        self.assertEqual(path.read_bytes(), self.closure.to_bytes())
        args = ['generate', '--bundle', bundle, '--expected-bundle-sha256', self.bundle.sha256,
                '--closure', path, '--expected-closure-sha256', self.closure.sha256, '--input-root', self.root,
                '--findings-root', findings_root, '--out', output]
        refused = self.cli(*args)
        self.assertNotEqual(refused.returncode, 0)
        self.assertEqual(refused.stdout, b'')
        result = self.cli(*args, '--authorize-unblinding')
        self.assertEqual(result.returncode, 0, result.stderr)
        report = self.report()
        self.assertEqual(json.loads(result.stdout)['report_sha256'], report.sha256)
        self.assertEqual((output / f'r{report.sha256}.json').read_bytes(), report.to_bytes())
        refused = self.cli(*args, '--authorize-unblinding')
        self.assertEqual(refused.returncode, 1)  # Never overwrite a report.
        self.assertEqual(refused.stdout, b'')
        self.assertNotIn(b'Traceback', refused.stderr)
        self.assertEqual(len(list(output.iterdir())), 2)
        bad_root = self.cli(*args[:-1], self.root, '--authorize-unblinding')
        self.assertNotEqual(bad_root.returncode, 0)


    def test_aggregate_history_budget_stops_reads_and_parsing_before_later_poison(self):
        import io
        from contextlib import redirect_stderr, redirect_stdout
        from unittest.mock import patch
        from aimusic.audio import review_bundle
        third = create_assignment(self.bundle, 'synthetic-r03', root=self.root, expected_sha256=self.bundle.sha256)
        poison = self.findings.create_template(third, root=self.root, expected_assignment_sha256=third.sha256)
        closure = replace(self.closure, roster=(*self.closure.roster,
                          self.reporting.ClosureEntry(third.sha256, third.reviewer, poison.sha256)))
        output, histories = self.base / 'bounded-reports', self.base / 'large-histories'
        output.mkdir()
        histories.mkdir()
        poisoned_path = histories / third.sha256 / f'f{poison.sha256}.json'
        sizes = {}
        for assignment, first in zip((*self.assignments, third), (self.first, self.second, poison)):
            path = self.root / assignment.reference(self.bundle).document.path
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(assignment.to_bytes())
            directory = histories / assignment.sha256
            directory.mkdir()
            if assignment == third:
                poisoned_path.write_bytes(b'not JSON; must never be read')
                continue
            # Each document and history is individually under its own cap.
            row = replace(first, singles=tuple(replace(s, notes='🎵' * 900) for s in first.singles),
                          pairs=tuple(replace(p, notes='🎵' * 900) for p in first.pairs))
            for _ in range(40):
                raw = row.to_bytes()
                path = directory / f'f{row.sha256}.json'
                path.write_bytes(raw)
                sizes[path] = len(raw)
                row = replace(row, previous_finding_sha256=row.sha256)
        self.assertGreater(sum(sizes.values()), self.reporting.MAX_FINDING_BYTES)
        bundle_path, closure_path = self.root / 'bundle.json', self.base / 'closure.json'
        bundle_path.write_bytes(self.bundle.to_bytes())
        closure_path.write_bytes(closure.to_bytes())
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(review_bundle, 'read_relative', wraps=review_bundle.read_relative) as reads, \
                patch.object(self.findings.ReviewFinding, 'from_bytes', wraps=self.findings.ReviewFinding.from_bytes) as parses, \
                redirect_stdout(stdout), redirect_stderr(stderr):
            result = self.reporting.main(['generate', '--bundle', str(bundle_path), '--expected-bundle-sha256', self.bundle.sha256,
                '--closure', str(closure_path), '--expected-closure-sha256', closure.sha256, '--input-root', str(self.root),
                '--findings-root', str(histories), '--out', str(output), '--authorize-unblinding'])
        self.assertEqual(result, 1)
        self.assertIn('aggregate history byte budget', stderr.getvalue())
        self.assertEqual(stdout.getvalue(), '')
        self.assertEqual(list(output.iterdir()), [])
        paths = [args.args[0] / args.args[1] for args in reads.call_args_list]
        self.assertNotIn(poisoned_path, paths)
        self.assertLessEqual(sum(sizes.get(p, 0) for p in paths), self.reporting.MAX_FINDING_BYTES)
        self.assertLessEqual(sum(len(args.args[0]) for args in parses.call_args_list), self.reporting.MAX_FINDING_BYTES)
        self.assertGreater(len(parses.call_args_list), 40)  # The budget spans histories, not each directory.
