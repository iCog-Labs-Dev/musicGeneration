"""Private authorized unblinding. Counts and individual judgments, never consensus.

Authorization is an operator acknowledgement plus private-map custody, not user
identity authentication. A separately trusted closure freezes the intended roster
and final finding digests; do not omit an intended reviewer to obtain a report.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from aimusic.audio.review_blinding import ReviewAssignment, _reviewer
from aimusic.audio.review_bundle import (
    GraphRecord, PresentationNode, PrivateBlindMap, RenderNode, ReviewBundle, SourceNode, _sha, _unique,
)
from aimusic.audio.review_findings import MAX_HISTORY, CanonicalRecord, ReviewFinding, verify_history

MAX_REVIEWERS = 32
MAX_FINDING_BYTES = 8 * 1024 * 1024
LIMITATIONS = (
    'Only rough-demo listening evidence for these exact bytes and assignments.',
    'No historical acceptance transfers to these stimuli.',
    'No mastering claim.',
    'No production-quality claim.',
    'No inferred polyphonic-f0 accuracy or other machine metric from human responses.',
)
CONTROL_DISCLOSURE = ('The flattened-12-calibration-control is a deliberately flattened tuning calibration control, '
                      'not product output. Its failed controlled-tuning gate uses unchanged 19-EDO targets.')


@dataclass(frozen=True)
class ClosureEntry(GraphRecord):
    assignment_document_sha256: str
    reviewer: str
    final_finding_sha256: str

    def validate(self) -> None:
        _sha(self.assignment_document_sha256)
        _sha(self.final_finding_sha256)
        _reviewer(self.reviewer)


@dataclass(frozen=True)
class ReviewClosure(CanonicalRecord):
    schema: Literal['aimusic.review-closure/1']
    bundle_sha256: str
    roster: tuple[ClosureEntry, ...]

    def validate(self) -> None:
        _sha(self.bundle_sha256)
        if not 0 < len(self.roster) <= MAX_REVIEWERS:
            raise ValueError('closure requires a bounded intended reviewer roster')
        _unique((r.assignment_document_sha256 for r in self.roster), 'roster assignment')
        _unique((r.reviewer for r in self.roster), 'roster reviewer')


@dataclass(frozen=True)
class ReviewerEvidence(GraphRecord):
    assignment: ReviewAssignment
    findings: tuple[ReviewFinding, ...]  # Oldest first, including excluded snapshots.


@dataclass(frozen=True)
class UnblindedStimulus(GraphRecord):
    id: str
    condition: str
    presentation_sha256: str
    negative_control: bool


@dataclass(frozen=True)
class ReviewReport(GraphRecord):
    schema: Literal['aimusic.review-report/1']
    bundle_sha256: str
    closure_sha256: str
    closure: ReviewClosure
    blind_map_sha256: str  # Never serialize order_seed or the private map document.
    reviewers: tuple[ReviewerEvidence, ...]
    sources: tuple[SourceNode, ...]
    renders: tuple[RenderNode, ...]  # Exact Metric records, not inferred/normalized summaries.
    presentations: tuple[PresentationNode, ...]
    stimuli: tuple[UnblindedStimulus, ...]
    history_counts: Mapping[str, Any]
    final_counts: Mapping[str, Any]
    eligible_counts: Mapping[str, Any]
    category_counts: Mapping[str, Any]
    negative_control_disclosure: str = CONTROL_DISCLOSURE
    limitations: tuple[str, ...] = LIMITATIONS


def _categories(finals: tuple[ReviewFinding, ...]) -> dict[str, Any]:
    singles: dict[str, Any] = {}
    pairs: dict[str, Any] = {}

    def count(target: dict[str, Any], field: str, value: str | None) -> None:
        if value is None:
            raise ValueError('unanswered field cannot enter judgment totals')
        counts = target.setdefault(field, {})
        counts[value] = counts.get(value, 0) + 1

    for finding in finals:
        for single in finding.singles:
            target = singles.setdefault(single.stimulus.id, {})
            count(target, 'audibility', single.audibility)
            count(target, 'confidence', single.confidence)
            for defect in single.defects or ('none',):
                count(target, 'defects', defect)
        for pair in finding.pairs:
            target = pairs.setdefault(pair.panel, {})
            count(target, 'perceived_difference', pair.perceived_difference)
            count(target, 'confidence', pair.confidence)
            preference = {'A': pair.a.id, 'B': pair.b.id}.get(pair.preference or '', pair.preference)
            count(target, 'preference_stimulus_or_category', preference)
            for stimulus, defects in ((pair.a, pair.defects_a), (pair.b, pair.defects_b)):
                side = target.setdefault('stimuli', {}).setdefault(stimulus.id, {})
                for defect in defects or ('none',):
                    count(side, 'defects', defect)
    return {'singles': singles, 'pairs': pairs}


def build_report(*, bundle: ReviewBundle, root: Path, expected_bundle_sha256: str,
                 closure: ReviewClosure, expected_closure_sha256: str,
                 assignments: tuple[ReviewAssignment, ...], histories: tuple[tuple[ReviewFinding, ...], ...],
                 authorize_unblinding: bool = False) -> ReviewReport:
    """Verify trust, all intended histories, current provenance/media and private map.

    History/assignment input order is immaterial; the closure's explicit roster
    order and each original assignment/chain are preserved. Withdrawn reviewers
    remain visible but contribute no judgments, even if formerly complete.
    """
    from aimusic.audio.review_archive import ArchiveLimits
    if authorize_unblinding is not True:
        raise ValueError('explicit --authorize-unblinding acknowledgement required')
    closure.__post_init__()
    if closure.sha256 != expected_closure_sha256 or closure.bundle_sha256 != expected_bundle_sha256:
        raise ValueError('closure trusted digest/bundle mismatch')
    if not len(assignments) == len(histories) == len(closure.roster):
        raise ValueError('missing or extra intended assignments/histories')
    by_assignment = {a.sha256: a for a in assignments}
    if len(by_assignment) != len(assignments) or set(by_assignment) != {r.assignment_document_sha256 for r in closure.roster}:
        raise ValueError('duplicate or mismatched roster assignments')
    remaining = MAX_FINDING_BYTES
    for history in histories:
        if not 0 < len(history) <= MAX_HISTORY + 2:
            raise ValueError('missing or over-budget histories')
        for finding in history:
            remaining -= len(finding.to_bytes())
            if remaining < 0:
                raise ValueError('aggregate history byte budget exceeded')
    by_history = {h[0].assignment_document_sha256: h for h in histories}
    if len(by_history) != len(histories) or set(by_history) != set(by_assignment):
        raise ValueError('duplicate or mismatched assignment histories')
    refs = set(bundle.references())
    limits = ArchiveLimits()
    if len(refs) > limits.max_input_files or sum(r.size for r in refs) > limits.max_input_bytes:
        raise ValueError('provenance input budget exceeded')
    bundle.verify(root, expected_sha256=expected_bundle_sha256)
    for render in bundle.renders:
        render.require_admitted()
    blind = PrivateBlindMap.from_bytes(bundle.blind_map.read_verified(root))
    evidence = []
    history_counts = dict.fromkeys(('incomplete', 'complete', 'withdrawn'), 0)
    final_counts = history_counts.copy()
    for entry in closure.roster:
        assignment = by_assignment[entry.assignment_document_sha256]
        if assignment.reviewer != entry.reviewer:
            raise ValueError('closure reviewer mismatch')
        assignment.validate_bundle(bundle)
        chain = verify_history(by_history[assignment.sha256], assignment, root=root,
                               expected_assignment_sha256=entry.assignment_document_sha256)
        if chain[-1].sha256 != entry.final_finding_sha256:
            raise ValueError('closure final finding mismatch')
        if chain[-1].state == 'incomplete':
            raise ValueError('incomplete intended reviewer blocks report')
        evidence.append(ReviewerEvidence(assignment, chain))
        for finding in chain:
            history_counts[finding.state] += 1
        final_counts[chain[-1].state] += 1
    finals = tuple(e.findings[-1] for e in evidence if e.findings[-1].state == 'complete')
    conditions = dict(blind.conditions)
    stimuli = tuple(UnblindedStimulus(s.id, conditions[s.id], s.presentation_sha256,
                                     conditions[s.id] == 'flattened-12-calibration-control') for s in bundle.stimuli)
    return ReviewReport('aimusic.review-report/1', bundle.sha256, closure.sha256, closure, blind.sha256,
                        tuple(evidence), bundle.sources, bundle.renders, bundle.presentations, stimuli,
                        history_counts, final_counts,
                        {'reviewers': len(finals), 'single_judgments': 8 * len(finals), 'pair_judgments': 3 * len(finals)},
                        _categories(finals))


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys
    from aimusic.audio.review_archive import _cli_digest, _publish_files, _read_record, MAX_METADATA_BYTES
    from aimusic.audio.review_bundle import MAX_FILE_BYTES, canonical_bytes
    from aimusic.audio.review_findings import read_histories, require_directory
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest='operation', required=True)
    close = commands.add_parser('close', allow_abbrev=False, help='freeze explicit intended roster, not a verification claim')
    close.add_argument('--bundle-sha256', type=_cli_digest, required=True)
    close.add_argument('--entry', nargs=3, action='append', required=True,
                       metavar=('ASSIGNMENT_DOCUMENT_SHA256', 'PSEUDONYM', 'FINAL_FINDING_SHA256'))
    generate = commands.add_parser('generate', allow_abbrev=False)
    for name in ('bundle', 'closure'):
        generate.add_argument('--' + name, type=Path, required=True)
        generate.add_argument('--expected-' + name + '-sha256', type=_cli_digest, required=True)
    generate.add_argument('--findings-root', type=Path, required=True, help='external root containing assignment-digest directories')
    generate.add_argument('--authorize-unblinding', action='store_true')
    for command in (close, generate):
        command.add_argument('--input-root', type=Path, required=True)
        command.add_argument('--out', type=Path, required=True, help='existing external private output directory')
    args = parser.parse_args(argv)
    result = {'schema': 'aimusic.review-report-result/1', 'operation': args.operation}
    try:
        # Acknowledgement before any private graph or map access.
        if args.operation == 'generate' and not args.authorize_unblinding:
            raise ValueError('explicit --authorize-unblinding acknowledgement required')
        output = require_directory(args.out, outside=args.input_root)
        document: GraphRecord
        if args.operation == 'close':
            document = ReviewClosure('aimusic.review-closure/1', args.bundle_sha256,
                                     tuple(ClosureEntry(*entry) for entry in args.entry))
            prefix, kind = 'c', 'closure'
        else:
            closure = _read_record(args.closure, ReviewClosure, expected_sha256=args.expected_closure_sha256,
                                   limit=MAX_METADATA_BYTES)
            bundle = _read_record(args.bundle, ReviewBundle, expected_sha256=args.expected_bundle_sha256, limit=MAX_FILE_BYTES)
            findings_root = require_directory(args.findings_root, outside=args.input_root)
            assignments = tuple(_read_record(args.input_root / bundle.assignment_directory / f'a{r.assignment_document_sha256}.json',
                                             ReviewAssignment, expected_sha256=r.assignment_document_sha256, limit=MAX_METADATA_BYTES)
                                for r in closure.roster)
            histories = read_histories(tuple(findings_root / r.assignment_document_sha256 for r in closure.roster),
                                       max_bytes=MAX_FINDING_BYTES, max_files=MAX_REVIEWERS * (MAX_HISTORY + 1))
            document = build_report(bundle=bundle, root=args.input_root, expected_bundle_sha256=args.expected_bundle_sha256,
                                    closure=closure, expected_closure_sha256=args.expected_closure_sha256,
                                    assignments=assignments, histories=histories, authorize_unblinding=args.authorize_unblinding)
            prefix, kind = 'r', 'report'
        data = document.to_bytes()
        if len(data) > MAX_FILE_BYTES:
            raise ValueError('report byte budget exceeded')
        name = f'{prefix}{document.sha256}.json'
        _publish_files(output, {name: data})
        result.update({kind + '_sha256': document.sha256, 'path': str(output / name)})
    except (OSError, ValueError, TypeError, RuntimeError, RecursionError, OverflowError) as exc:
        sys.stderr.write(canonical_bytes({**result, 'error': str(exc)}).decode('ascii'))
        return 1
    sys.stdout.write(canonical_bytes(result).decode('ascii'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
