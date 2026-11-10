"""External whole-assignment listening snapshots. Never infer or fabricate answers.

Digests bind bytes, not identity/authenticity. Trusted-local, single-writer custody
is required; create-only publication is not a tamper-proof append-only filesystem.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeVar, get_args

from aimusic.audio.review_blinding import AssignedStimulus, ReviewAssignment, _reviewer
from aimusic.audio.review_bundle import GraphRecord, _sha, _unique

MAX_DOCUMENT_BYTES = 128 * 1024
MAX_HISTORY = 128  # Ordinary snapshots; one additional final withdrawal is reserved.
Confidence = Literal['low', 'medium', 'high']
FindingState = Literal['incomplete', 'complete', 'withdrawn']
Defect = Literal['click_pop', 'distortion', 'dropout', 'missing_voice', 'missing_notes',
                 'stuck_note', 'timing_issue', 'pitch_issue', 'cut_off_tail', 'other']
DEFECTS = get_args(Defect)
C = TypeVar('C', bound='CanonicalRecord')


class CanonicalRecord(GraphRecord):
    """Exact canonical JSON only; reject normalization, duplicate keys and excess bytes."""
    def to_bytes(self) -> bytes:
        """Bound canonical output too: escaped Unicode can outgrow an authored draft."""
        data = super().to_bytes()
        if len(data) > MAX_DOCUMENT_BYTES:
            raise ValueError('document byte budget exceeded')
        return data

    @classmethod
    def from_bytes(cls: type[C], data: bytes, *, expected_sha256: str | None = None) -> C:
        if len(data) > MAX_DOCUMENT_BYTES:
            raise ValueError('document byte budget exceeded')
        try:
            result = super().from_bytes(data, expected_sha256=expected_sha256)
        except (RecursionError, UnicodeError, OverflowError) as exc:
            raise ValueError('invalid canonical document') from exc
        if result.to_bytes() != data:
            raise ValueError('exact canonical JSON required')
        return result


def _notes(value: str | None) -> None:
    if value is not None and (not value.strip() or len(value) > 1024 or any(ord(c) < 32 or ord(c) == 127 for c in value)):
        raise ValueError('notes/reason must be nonempty bounded text without controls')


def _defects(value: tuple[Defect, ...] | None, notes: str | None) -> None:
    if value is not None:
        _unique(value, 'defect')
        if 'other' in value and notes is None:
            raise ValueError('other defect requires explanatory notes')


@dataclass(frozen=True)
class SingleFinding(GraphRecord):
    panel: Literal['canonical', 'roles']
    stimulus: AssignedStimulus
    audibility: Literal['audible', 'partly_audible', 'inaudible', 'unsure'] | None = None
    defects: tuple[Defect, ...] | None = None
    confidence: Confidence | None = None
    notes: str | None = None

    def validate(self) -> None:
        _notes(self.notes)
        _defects(self.defects, self.notes)

    def answered(self) -> bool:
        return all(v is not None for v in (self.audibility, self.defects, self.confidence))


@dataclass(frozen=True)
class PairFinding(GraphRecord):
    panel: Literal['parity', 'timing', 'tuning']
    a: AssignedStimulus
    b: AssignedStimulus
    perceived_difference: Literal['heard', 'not_heard', 'unsure'] | None = None
    defects_a: tuple[Defect, ...] | None = None
    defects_b: tuple[Defect, ...] | None = None
    preference: Literal['A', 'B', 'no_preference', 'unsure'] | None = None
    confidence: Confidence | None = None
    notes: str | None = None

    def validate(self) -> None:
        _notes(self.notes)
        _defects(self.defects_a, self.notes)
        _defects(self.defects_b, self.notes)
        _unique((self.a.id, self.b.id), 'pair stimulus')

    def answered(self) -> bool:
        return all(v is not None for v in (self.perceived_difference, self.defects_a, self.defects_b,
                                          self.preference, self.confidence))


@dataclass(frozen=True)
class ReviewFinding(CanonicalRecord):
    schema: Literal['aimusic.review-finding/1']
    assignment_document_sha256: str  # NOT the legacy AssignmentReference wrapper digest.
    bundle_sha256: str
    reviewer: str
    singles: tuple[SingleFinding, ...]
    pairs: tuple[PairFinding, ...]
    state: FindingState = 'incomplete'
    previous_finding_sha256: str | None = None
    withdrawal_reason: str | None = None

    def validate(self) -> None:
        for value in (self.assignment_document_sha256, self.bundle_sha256):
            _sha(value)
        if self.previous_finding_sha256 is not None:
            _sha(self.previous_finding_sha256)
        _reviewer(self.reviewer)
        _notes(self.withdrawal_reason)
        if (self.state == 'withdrawn') != (self.withdrawal_reason is not None):
            raise ValueError('withdrawn requires a reason; other states forbid it')
        if tuple(s.panel for s in self.singles) != ('canonical',) * 4 + ('roles',) * 4:
            raise ValueError('eight singles in exact panel order required')
        if tuple(p.panel for p in self.pairs) != ('parity', 'timing', 'tuning'):
            raise ValueError('three pairs in exact panel order required')
        _unique((*[s.stimulus.id for s in self.singles], *[s.id for p in self.pairs for s in (p.a, p.b)]), 'stimulus')
        if self.state == 'complete' and not (all(s.answered() for s in self.singles) and all(p.answered() for p in self.pairs)):
            raise ValueError('complete requires every response field answered')

    def validate_assignment(self, assignment: ReviewAssignment) -> None:
        self.__post_init__()
        self.to_bytes()  # Constructed/from_dict findings must obey the wire budget too.
        if (self.assignment_document_sha256, self.bundle_sha256, self.reviewer) != (
                assignment.sha256, assignment.bundle_sha256, assignment.reviewer):
            raise ValueError('finding assignment document/bundle/reviewer mismatch')
        media = {s.id: s for s in assignment.stimuli}
        expected_singles = tuple(media[s] for p in assignment.panels[:2] for s in p.stimuli)
        expected_pairs = tuple(tuple(media[s] for s in p.stimuli) for p in assignment.panels[2:])
        if (tuple(s.stimulus for s in self.singles) != expected_singles
                or tuple((p.a, p.b) for p in self.pairs) != expected_pairs):
            raise ValueError('finding stimulus/media/order mismatch')


def create_template(assignment: ReviewAssignment, *, root: Path, expected_assignment_sha256: str) -> ReviewFinding:
    assignment.verify(root, expected_sha256=expected_assignment_sha256)
    media = {s.id: s for s in assignment.stimuli}
    panels = {p.purpose: p.stimuli for p in assignment.panels}
    singles = tuple(SingleFinding(name, media[s]) for name in ('canonical', 'roles') for s in panels[name])
    pairs = tuple(PairFinding(name, media[panels[name][0]], media[panels[name][1]])
                  for name in ('parity', 'timing', 'tuning'))
    return ReviewFinding('aimusic.review-finding/1', assignment.sha256, assignment.bundle_sha256, assignment.reviewer,
                         singles, pairs)


def validate_finding(finding: ReviewFinding, assignment: ReviewAssignment, *, root: Path,
                     expected_assignment_sha256: str) -> FindingState:
    """Structural completeness is not history eligibility or listening authenticity."""
    assignment.verify(root, expected_sha256=expected_assignment_sha256)
    finding.validate_assignment(assignment)
    return finding.state


def _check_history_limit(findings: tuple[ReviewFinding, ...]) -> None:
    withdrawals = sum(f.state == 'withdrawn' for f in findings)
    if withdrawals > 1 or len(findings) - withdrawals > MAX_HISTORY:
        raise ValueError('finding history budget exceeded')


def verify_history(findings: tuple[ReviewFinding, ...], assignment: ReviewAssignment, *, root: Path,
                   expected_assignment_sha256: str) -> tuple[ReviewFinding, ...]:
    """Return one oldest-first chain; identical replay counts once, ambiguity rejects."""
    if not 0 < len(findings) <= MAX_HISTORY + 2:  # Capacity plus one identical replay.
        raise ValueError('missing or over-budget finding history')
    assignment.verify(root, expected_sha256=expected_assignment_sha256)
    unique = {}
    for finding in findings:
        finding.validate_assignment(assignment)
        unique[finding.sha256] = finding
    _check_history_limit(tuple(unique.values()))
    children: dict[str | None, ReviewFinding] = {}
    for finding in unique.values():
        previous = finding.previous_finding_sha256
        if previous is not None and previous not in unique:
            raise ValueError('missing or cross-assignment predecessor')
        if previous in children:
            raise ValueError('ambiguous history: fork or two roots')
        children[previous] = finding
    first = children.get(None)
    if first is None or first.state != 'incomplete':
        raise ValueError('history requires one incomplete root')
    chain = [first]
    while chain[-1].sha256 in children:
        current = children[chain[-1].sha256]
        allowed = {'incomplete': ('incomplete', 'complete', 'withdrawn'), 'complete': ('withdrawn',), 'withdrawn': ()}
        if current.state not in allowed[chain[-1].state]:
            raise ValueError('invalid finding history transition')
        chain.append(current)
    if len(chain) != len(unique):
        raise ValueError('disconnected or cyclic finding history')
    return tuple(chain)


def require_directory(directory: Path, *, outside: Path | None = None) -> Path:
    """Trusted-local path preflight, not a namespace lock. Existing directories only."""
    import stat
    directory = directory.absolute()
    for parent in (*reversed(directory.parents), directory):
        info = parent.lstat()
        if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
                or getattr(info, 'st_file_attributes', 0) & 0x400):
            raise ValueError('directory has nonregular/link/reparse component')
        if outside is not None and parent.samefile(outside):
            raise ValueError('findings/reports must remain external to input root')
    return directory


def read_histories(directories: tuple[Path, ...], *, max_bytes: int,
                   max_files: int) -> tuple[tuple[ReviewFinding, ...], ...]:
    """Charge a shared input budget before parsing; never read an excess file.

    One assignment per directory, exact inventory. Trusted-local single-writer
    custody is required, as for read_relative; this is not a namespace lock.
    """
    import os
    import re
    from aimusic.audio.review_bundle import read_relative
    histories = []
    for directory in directories:
        directory = require_directory(directory)
        rows: list[ReviewFinding] = []
        with os.scandir(directory) as entries:
            for entry in entries:
                match = re.fullmatch(r'f([0-9a-f]{64})\.json', entry.name)
                if match is None or len(rows) >= MAX_HISTORY + 1:
                    raise ValueError('unexpected or over-budget history inventory')
                if max_files <= 0:
                    raise ValueError('aggregate history file budget exceeded')
                info = entry.stat(follow_symlinks=False)
                if info.st_nlink > 1:
                    raise ValueError('hard-linked history file rejected')
                if max_bytes <= 0 or info.st_size > max_bytes:
                    raise ValueError('aggregate history byte budget exceeded')
                data = read_relative(directory, entry.name, limit=min(MAX_DOCUMENT_BYTES, max_bytes))
                max_bytes -= len(data)  # Actual bytes, before JSON parse, not canonicalized estimates.
                max_files -= 1
                rows.append(ReviewFinding.from_bytes(data, expected_sha256=match[1]))
        _check_history_limit(tuple(rows))
        histories.append(tuple(sorted(rows, key=lambda f: f.sha256)))
    return tuple(histories)


def read_history(directory: Path) -> tuple[ReviewFinding, ...]:
    """One bounded history; unknown inventory/residue is never silently ignored."""
    return read_histories((directory,), max_bytes=(MAX_HISTORY + 1) * MAX_DOCUMENT_BYTES, max_files=MAX_HISTORY + 1)[0]


def append_finding(finding: ReviewFinding, assignment: ReviewAssignment, *, root: Path,
                   expected_assignment_sha256: str, directory: Path) -> Path:
    """Verify the entire local history, then publish one digest-named file create-only.

    Custody must exclude concurrent writers across the read/verify/publish interval.
    Existing identical replay is read/verified, never rewritten. Preserve all files
    on validation failure; unknown inventory/residue requires operator inspection.
    """
    from aimusic.audio.review_archive import _publish_files
    data = finding.to_bytes()  # Reject before publication can create anything.
    directory = require_directory(directory, outside=root)
    rows = read_history(directory)
    verify_history((*rows, finding), assignment, root=root, expected_assignment_sha256=expected_assignment_sha256)
    name = f'f{finding.sha256}.json'
    if finding.sha256 not in {f.sha256 for f in rows}:
        _publish_files(directory, {name: data})
    return directory / name


def main(argv: list[str] | None = None) -> int:
    """Offline collection from an independently trusted assignment and current media.

    Keep the history and authored drafts outside the extracted archive view. No
    private graph, map, network, renderer or asset discovery is needed here.
    """
    import argparse
    import sys
    from aimusic.audio.review_archive import _cli_digest, _read_record, MAX_METADATA_BYTES
    from aimusic.audio.review_bundle import canonical_bytes
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest='operation', required=True)
    for name in ('template', 'validate', 'append'):
        command = commands.add_parser(name, allow_abbrev=False)
        command.add_argument('--assignment', type=Path, required=True)
        command.add_argument('--expected-assignment-sha256', type=_cli_digest, required=True)
        command.add_argument('--media-root', type=Path, required=True)
        if name != 'template':
            command.add_argument('--finding', type=Path, required=True)
            command.add_argument('--expected-finding-sha256', type=_cli_digest, required=True)
        if name != 'validate':
            command.add_argument('--history', type=Path, required=True, help='existing external directory, one assignment only')
    args = parser.parse_args(argv)
    result = {'schema': 'aimusic.review-finding-result/1', 'operation': args.operation}
    try:
        assignment = _read_record(args.assignment, ReviewAssignment, expected_sha256=args.expected_assignment_sha256,
                                  limit=MAX_METADATA_BYTES)
        if args.operation == 'template':
            finding = create_template(assignment, root=args.media_root, expected_assignment_sha256=args.expected_assignment_sha256)
        else:
            finding = _read_record(args.finding, ReviewFinding, expected_sha256=args.expected_finding_sha256,
                                   limit=MAX_DOCUMENT_BYTES)
        state = validate_finding(finding, assignment, root=args.media_root,
                                 expected_assignment_sha256=args.expected_assignment_sha256)
        if args.operation != 'validate':
            path = append_finding(finding, assignment, root=args.media_root,
                                  expected_assignment_sha256=args.expected_assignment_sha256, directory=args.history)
            result['path'] = str(path)
        result.update(finding_sha256=finding.sha256, assignment_document_sha256=assignment.sha256, state=state)
    except (OSError, ValueError, TypeError, RuntimeError, RecursionError, OverflowError) as exc:
        sys.stderr.write(canonical_bytes({**result, 'error': str(exc)}).decode('ascii'))
        return 1
    sys.stdout.write(canonical_bytes(result).decode('ascii'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
