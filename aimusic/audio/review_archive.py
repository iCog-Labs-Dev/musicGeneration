"""Sealed reviewer delivery, not directory publication. No discovery or network.

Only canonical, stored (uncompressed), non-ZIP64 regular files are supported.
The independently conveyed whole-archive digest is mandatory, not self-trust.
Input roots and extracted views require trusted-local custody: no concurrent
namespace or payload mutation. Completion alone is not an integrity check.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
import hashlib
import io
import os
from pathlib import Path
import struct
import stat
import sys
import tempfile
from typing import Literal, TypeVar
import zipfile

from aimusic.audio.publication import pin_output, validate_names
from aimusic.audio.publication_records import FileIdentity
from aimusic.audio.listening_review import _validate_wav
from aimusic.audio.review_blinding import ReviewAssignment
from aimusic.audio.review_bundle import (
    MAX_FILE_BYTES, FileRef, PrivateBlindMap, Record, ReviewBundle, canonical_bytes, read_relative, safe_relative,
)
from aimusic.audio.review_pages import render_review_pages

MANIFEST = 'manifest.json'
COMPLETION = 'complete.json'
MAX_METADATA_BYTES = 64 * 1024


@dataclass(frozen=True)
class ArchiveLimits:
    """Hard ceilings; callers may tighten, never enlarge them. Counts include manifest.

    Compressed/expanded totals include every ZIP member. Archive bytes also count
    headers. Input work counts unique private references, including media, once.
    Peak byte storage is bounded by four archive-sized buffers plus one input
    file; no decoder is invoked and no allocation uses unchecked ZIP sizes.
    """
    max_members: int = 32
    max_archive_bytes: int = 128 * 1024 * 1024
    max_compressed_bytes: int = 127 * 1024 * 1024
    max_expanded_bytes: int = 127 * 1024 * 1024
    max_input_files: int = 512
    max_input_bytes: int = 512 * 1024 * 1024

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if type(value) is not int or not 0 < value <= getattr(ArchiveLimits, f.name):
                raise ValueError(f'invalid archive budget: {f.name}')


@dataclass(frozen=True)
class ArchiveManifest(Record):
    schema: Literal['aimusic.review-archive/1']
    bundle_sha256: str
    assignment_sha256: str
    members: tuple[FileRef, ...]

    def validate(self) -> None:
        FileRef('digest', self.bundle_sha256, 0)
        FileRef('digest', self.assignment_sha256, 0)
        names = [r.path for r in self.members]
        _names(names)
        if names != sorted(names) or MANIFEST in names or COMPLETION in names:
            raise ValueError('manifest must list each payload, not itself or completion')


@dataclass(frozen=True)
class SealedArchive:
    data: bytes
    sha256: str


@dataclass(frozen=True)
class VerifiedArchive:
    sha256: str
    members: tuple[tuple[str, bytes], ...]


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _names(names: list[str]) -> None:
    # Also reject differently cased ancestors and file/directory prefix clashes.
    seen: dict[str, tuple[str, bool]] = {}
    for name in names:
        safe_relative(name)
        if not name.isascii() or len(name) > 128:
            raise ValueError('nonportable archive member name')
        parts = name.split('/')
        for i in range(1, len(parts) + 1):
            path, leaf = '/'.join(parts[:i]), i == len(parts)
            key = path.casefold()
            if key in seen and (seen[key] != (path, leaf) or leaf):
                raise ValueError('duplicate, casefold or file/directory collision')
            seen[key] = (path, leaf)


def _zip(members: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', compression=zipfile.ZIP_STORED, allowZip64=False) as archive:
        for name, data in sorted(members.items()):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    return buffer.getvalue()


def _wav(data: bytes) -> None:
    _validate_wav(data)
    pos, rate, align = 12, 0, 0
    while pos < len(data):
        name, size = struct.unpack_from('<4sI', data, pos)
        if name not in (b'fmt ', b'fact', b'data'):
            raise ValueError('presentation contains ancillary WAV metadata')
        if name == b'fmt ':
            _, _, rate, _, align, _ = struct.unpack_from('<HHIIHH', data, pos + 8)
        pos += 8 + size + size % 2
    # The format can follow data in RIFF; inspect duration only after finding it.
    pos = 12
    while pos < len(data):
        name, size = struct.unpack_from('<4sI', data, pos)
        if name == b'data' and size / align / rate > 120:
            raise ValueError('presentation duration budget exceeded')
        pos += 8 + size + size % 2


def _budget(sizes: list[int], names: list[str], limits: ArchiveLimits) -> None:
    total = sum(sizes)
    if (len(sizes) > limits.max_members or any(n > MAX_FILE_BYTES for n in sizes)
            or total > limits.max_compressed_bytes or total > limits.max_expanded_bytes
            or total + sum(76 + 2 * len(n) for n in names) + 22 > limits.max_archive_bytes):
        raise ValueError('archive count/compressed/expanded byte budget exceeded')


def _payloads(members: dict[str, bytes], manifest: ArchiveManifest) -> None:
    if 'assignment.json' not in members or len(members['assignment.json']) > MAX_METADATA_BYTES:
        raise ValueError('missing or over-budget assignment')
    assignment = ReviewAssignment.from_bytes(members['assignment.json'], expected_sha256=manifest.assignment_sha256)
    if assignment.to_bytes() != members['assignment.json'] or assignment.bundle_sha256 != manifest.bundle_sha256:
        raise ValueError('noncanonical or mismatched assignment')
    pages = render_review_pages(assignment, expected_sha256=manifest.assignment_sha256)
    if set(members) != {MANIFEST, 'assignment.json', *pages, *[s.media.path for s in assignment.stimuli]}:
        raise ValueError('missing or extra reviewer members')
    for name, data in pages.items():
        if members[name] != data:
            raise ValueError('page differs from reviewer-safe template')
    for stimulus in assignment.stimuli:
        data = members[stimulus.media.path]
        if len(data) != stimulus.media.size or _sha(data) != stimulus.media.sha256:
            raise ValueError('assignment media digest/size mismatch')
        _wav(data)


def verify_archive(archive: bytes | Path, *, expected_sha256: str,
                   limits: ArchiveLimits = ArchiveLimits()) -> VerifiedArchive:
    """Fully verify before exposing members. Reject every noncanonical ZIP feature.

    EOCD/count/central-directory byte caps are checked before ZipFile allocates its
    inventory. Stored sizes and cumulative budgets are checked before member reads.
    Canonical re-encoding checks *all* local/central fields, offsets, order, CRCs,
    extras, comments, flags, gaps, prefixes/trailers and unsupported ZIP64 records.
    """
    limits.__post_init__()
    FileRef('digest', expected_sha256, 0)
    data = archive if isinstance(archive, bytes) else read_relative(Path(archive).parent, Path(archive).name,
                                                               limit=limits.max_archive_bytes)
    if len(data) > limits.max_archive_bytes or _sha(data) != expected_sha256:
        raise ValueError('archive budget or trusted digest mismatch')
    try:
        if len(data) < 22:
            raise ValueError('truncated ZIP')
        signature, disk, cd_disk, disk_count, count, size, offset, comment = struct.unpack('<4s4H2IH', data[-22:])
        if (signature != b'PK\x05\x06' or disk or cd_disk or comment or disk_count != count
                or not 0 < count <= limits.max_members or size > limits.max_members * (46 + 128)
                or offset + size != len(data) - 22):
            raise ValueError('malformed or over-budget ZIP directory')
        # allowZip64=False only constrains writing. The reader otherwise lets a
        # ZIP64 locator replace the just-bounded classic directory size/count.
        if data[-42:-38] == b'PK\x06\x07':
            raise ValueError('unsupported ZIP64 directory')
        with zipfile.ZipFile(io.BytesIO(data), 'r', allowZip64=False) as stream:
            infos = stream.infolist()
            names = [i.filename for i in infos]
            _names(names)
            if len(infos) != count or names != sorted(names):
                raise ValueError('ZIP count/order mismatch')
            for info in infos:
                if (info.compress_type != zipfile.ZIP_STORED or info.flag_bits or info.extra or info.comment
                        or info.external_attr != 0o100644 << 16 or info.create_system != 3
                        or info.compress_size != info.file_size or info.orig_filename != info.filename):
                    raise ValueError('unsupported ZIP metadata, compression, encryption or special entry')
                if info.filename in (MANIFEST, 'assignment.json') and info.file_size > MAX_METADATA_BYTES:
                    raise ValueError('metadata byte budget exceeded')
            _budget([i.file_size for i in infos], names, limits)
            members = {i.filename: stream.read(i) for i in infos}
        if _zip(members) != data:
            raise ValueError('noncanonical ZIP metadata')
        if MANIFEST not in members:
            raise ValueError('missing manifest')
        manifest = ArchiveManifest.from_bytes(members[MANIFEST])
        if manifest.to_bytes() != members[MANIFEST]:
            raise ValueError('noncanonical manifest')
        if {r.path for r in manifest.members} != set(members) - {MANIFEST}:
            raise ValueError('missing or extra manifest members')
        for ref in manifest.members:
            if ref.size != len(members[ref.path]) or ref.sha256 != _sha(members[ref.path]):
                raise ValueError('member digest/size mismatch')
        _payloads(members, manifest)
        return VerifiedArchive(expected_sha256, tuple(members.items()))
    except (zipfile.BadZipFile, NotImplementedError, RuntimeError, KeyError, TypeError,
            UnicodeError, RecursionError, OverflowError, struct.error) as exc:
        raise ValueError(f'invalid sealed archive: {exc}') from exc


def assemble_archive(root: Path, *, bundle: ReviewBundle, assignment: ReviewAssignment,
                     expected_bundle_sha256: str, expected_assignment_sha256: str,
                     limits: ArchiveLimits = ArchiveLimits()) -> SealedArchive:
    """Verify the admitted private graph; build only allowlisted reviewer members.

    Never walks/copies a source tree. Private evidence is hashed then discarded;
    only exact declared presentation bytes, the assignment and generated pages go
    into the ZIP. Admission scores must come from the existing candidate reader.
    """
    limits.__post_init__()
    if type(bundle) is not ReviewBundle or type(assignment) is not ReviewAssignment:
        raise ValueError('bundle and reviewer-safe assignment required')
    bundle.__post_init__()
    assignment.__post_init__()
    if bundle.sha256 != expected_bundle_sha256 or assignment.sha256 != expected_assignment_sha256:
        raise ValueError('bundle/assignment digest mismatch')
    assignment.validate_bundle(bundle)
    for render in bundle.renders:
        render.require_admitted()
    refs: dict[str, FileRef] = {}
    for ref in bundle.references():
        key = ref.path.casefold()
        if key in refs and refs[key] != ref:
            raise ValueError('conflicting input references')
        refs[key] = ref
    if len(refs) > limits.max_input_files or sum(r.size for r in refs.values()) > limits.max_input_bytes:
        raise ValueError('input count/byte budget exceeded')
    members = {'assignment.json': assignment.to_bytes(),
               **render_review_pages(assignment, expected_sha256=expected_assignment_sha256)}
    media = {s.media.path for s in assignment.stimuli}
    # Preflight delivery sizes before loading any media or private source bytes.
    _budget([len(d) for d in members.values()] + [r.size for r in refs.values() if r.path in media],
            list(members) + sorted(media), limits)
    for ref in refs.values():
        data = read_relative(root, ref.path, limit=ref.size)
        if len(data) != ref.size or _sha(data) != ref.sha256:
            raise ValueError(f'input digest/size mismatch: {ref.path}')
        if ref == bundle.blind_map:
            PrivateBlindMap.from_bytes(data).validate_bundle(bundle)
        if ref.path in media:
            _wav(data)
            members[ref.path] = data
    manifest = ArchiveManifest('aimusic.review-archive/1', bundle.sha256, assignment.sha256,
                               tuple(FileRef(n, _sha(d), len(d)) for n, d in sorted(members.items())))
    members[MANIFEST] = manifest.to_bytes()
    if len(members[MANIFEST]) > MAX_METADATA_BYTES:
        raise ValueError('manifest byte budget exceeded')
    _budget([len(d) for d in members.values()], list(members), limits)
    data = _zip(members)
    sealed = SealedArchive(data, _sha(data))
    verify_archive(data, expected_sha256=sealed.sha256, limits=limits)
    return sealed


def _destination(out: Path, *, source_root: Path | None = None) -> Path:
    """Preflight existing local parents without creating them or resolving links."""
    out = Path(out).absolute()
    validate_names([out.name])
    for parent in reversed(out.parents):
        info = parent.lstat()
        if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
                or getattr(info, 'st_file_attributes', 0) & 0x400):
            raise ValueError('destination ancestor is not a regular directory or is a link/reparse point')
        if source_root is not None and parent.samefile(source_root):
            raise ValueError('destination aliases or lies inside source root')
    if any(p.name.casefold() == out.name.casefold() for p in out.parent.iterdir()):
        raise FileExistsError(f'destination or casefold alias exists: {out}')
    return out


def _verify_posix_published(directory_fd: int, name: str, expected: bytes, *, expected_identity: FileIdentity) -> None:
    """Check a public leaf through the transaction's retained directory, not its path."""
    from aimusic.audio.publication_posix import _required_flag
    validate_names([name])
    flags = os.O_RDONLY | _required_flag('O_NOFOLLOW') | _required_flag('O_NONBLOCK')
    fd = os.open(name, flags, dir_fd=directory_fd)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('published leaf is not a regular file')
        if FileIdentity.from_stat(info) != expected_identity:
            raise ValueError('published leaf identity differs from transaction ownership')
        if info.st_size != len(expected):
            raise ValueError('published byte size differs from verified bytes')
        offset = 0
        while chunk := os.read(fd, min(1024 * 1024, len(expected) - offset + 1)):
            if chunk != expected[offset:offset + len(chunk)]:
                raise ValueError('published bytes differ from verified bytes')
            offset += len(chunk)
        if offset != len(expected):
            raise ValueError('published bytes were truncated')
    finally:
        os.close(fd)


def _verify_windows_published(path: Path, expected: bytes, *, expected_identity: FileIdentity) -> None:
    # A public leaf still has the transaction's retained WRITE/DELETE handle.
    # A read-only handle must share those accesses (CRT os.open omits DELETE).
    # This does NOT relax sharing on retained handles or directory pins.
    from aimusic.audio.windows_native import READ, SHARE_ALL, WindowsAPI
    api = WindowsAPI()
    with api.open(path, READ, share=SHARE_ALL) as handle:
        info = api.information(handle)
        info.require(directory=False)  # OPEN_REPARSE_POINT: never follow the leaf.
        if info.identity != expected_identity:
            raise ValueError('published leaf identity differs from transaction ownership')
        if info.size != len(expected):
            raise ValueError('published byte size differs from verified bytes')
        offset = 0
        while chunk := api.read(handle, min(1024 * 1024, len(expected) - offset + 1)):
            if chunk != expected[offset:offset + len(chunk)]:
                raise ValueError('published bytes differ from verified bytes')
            offset += len(chunk)
        if offset != len(expected):
            raise ValueError('published bytes were truncated')


def _publish_files(directory: Path, members: dict[str, bytes], *, sources: tuple[Path, ...] = ()) -> None:
    # Staging is invocation-private under the existing single-file publication
    # threat model, not a directory promoted as a supposedly immutable snapshot.
    with pin_output(directory) as pinned:
        pinned.check_new_files(list(members))
        with tempfile.TemporaryDirectory(prefix='review-archive-') as tmp:
            staging = Path(tmp).resolve()
            for name, data in members.items():
                with (staging / name).open('xb') as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                if read_relative(staging, name, limit=len(data)) != data:
                    raise ValueError('staged bytes differ from verified bytes')
            with pinned.publication(staging, list(members), replace=False, sources=sources) as published:
                pinned.verify()
                # Inspect the actual public leaves, after the backend/private
                # second copy, bound to staged identities rather than fresh path stats.
                # A mismatch raises while rollback still owns the published files.
                for owner in published:
                    name, data = owner.name, members[owner.name]
                    if os.name == 'nt':
                        _verify_windows_published(pinned.path / name, data, expected_identity=owner.identity)
                    else:
                        from aimusic.audio.publication_posix import PosixBackend
                        if not isinstance(pinned.backend, PosixBackend):
                            raise NotImplementedError('retained POSIX output descriptor required')
                        _verify_posix_published(pinned.backend.fd, name, data, expected_identity=owner.identity)
                pinned.verify()


def write_review_archive(root: Path, out: Path, *, bundle: ReviewBundle, assignment: ReviewAssignment,
                         expected_bundle_sha256: str, expected_assignment_sha256: str,
                         limits: ArchiveLimits = ArchiveLimits()) -> str:
    """Publish one fully verified ZIP create-only; return its independent SHA-256.

    Send that digest by a trusted channel separate from the archive. Existing
    parents are required; input roots and their filesystem aliases are forbidden
    output locations. No reviewer-visible directory is used for publication.
    """
    out = _destination(out, source_root=Path(root))
    sealed = assemble_archive(root, bundle=bundle, assignment=assignment,
                              expected_bundle_sha256=expected_bundle_sha256,
                              expected_assignment_sha256=expected_assignment_sha256, limits=limits)
    _publish_files(out.parent, {out.name: sealed.data}, sources=tuple(Path(root) / r.path for r in bundle.references()))
    return sealed.sha256


def _completion(sha256: str) -> bytes:
    return canonical_bytes({'schema': 'aimusic.review-extraction/1', 'archive_sha256': sha256})


def verify_extracted(archive: bytes | Path, out: Path, *, expected_sha256: str,
                     limits: ArchiveLimits = ArchiveLimits()) -> Path:
    """Read-only verification of a disposable trusted-local view before serving.

    Verify the sealed archive with an independently conveyed digest, then require
    its exact bytes (including validated templates) and canonical completion, with
    no extra files or directories. Enumeration and reads are bounded; links and
    reparse points are rejected. Concurrent namespace/payload mutation is NOT
    supported. This is not a snapshot, a custody proof, or a serving-time lock.
    Re-extract into a fresh trusted location if custody is uncertain.
    """
    verified = verify_archive(archive, expected_sha256=expected_sha256, limits=limits)
    out = Path(out).absolute()
    for directory in (*reversed(out.parents), out, out / 'media'):
        info = directory.lstat()
        if (not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode)
                or getattr(info, 'st_file_attributes', 0) & 0x400):
            raise ValueError('extracted directory is not regular or is a link/reparse point')
    members = dict(verified.members)
    members[COMPLETION] = _completion(verified.sha256)
    # Only the already-validated root and media directory are enumerated. Stop at
    # the first extra entry, never recurse or allocate an unbounded directory list.
    for directory, expected in (
        (out, {n for n in members if '/' not in n} | {'media'}),
        (out / 'media', {n.removeprefix('media/') for n in members if n.startswith('media/')}),
    ):
        remaining = set(expected)
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.name not in remaining:
                    raise ValueError('extra extracted inventory entry')
                remaining.remove(entry.name)
        if remaining:
            raise ValueError('missing extracted inventory entry')
    for name, data in members.items():
        if read_relative(out, name, limit=len(data)) != data:
            raise ValueError(f'extracted bytes differ from verified archive/completion: {name}')
    return out


def extract_archive(archive: bytes | Path, out: Path, *, expected_sha256: str,
                    limits: ArchiveLimits = ArchiveLimits()) -> Path:
    """Verify completely, then exclusively create a disposable local view.

    Requires trusted-local custody: concurrent namespace OR payload mutation is
    unsupported throughout extraction, verification, and serving. Path-based mkdir
    can escape a swapped parent before a post-check, or adopt a substitute before
    pinning; it is not safely pinned child creation. No directory promotion occurs.
    Never resumes, deletes, rolls back directories, or overwrites existing leaves.
    Failure may retain an incomplete view: retry into a *new* name. Even an exact
    completion record cannot prove the current tree's integrity. Call
    verify_extracted immediately before serving; keep custody while serving.
    Re-extract from the trusted archive when custody of a local view is uncertain.
    """
    verified = verify_archive(archive, expected_sha256=expected_sha256, limits=limits)
    out = _destination(out)
    with pin_output(out.parent) as parent:
        parent.check_new_files([out.name])
        os.mkdir(out, 0o700)  # Path-based create-only, NOT pinned child creation; see contract above.
        parent.verify()
    members = dict(verified.members)
    _publish_files(out, {n: d for n, d in members.items() if '/' not in n})
    # The exact reviewer payload allowlist admits just one nested directory.
    with pin_output(out) as root:
        root.check_new_files(['media'])
        os.mkdir(out / 'media', 0o700)
        root.verify()
    _publish_files(out / 'media', {n.removeprefix('media/'): d for n, d in members.items() if n.startswith('media/')})
    # Check all copied bytes before completion, not merely publication success.
    for name, data in verified.members:
        if read_relative(out, name, limit=len(data)) != data:
            raise ValueError('extracted bytes differ from verified archive')
    _publish_files(out, {COMPLETION: _completion(verified.sha256)})
    return out


_R = TypeVar('_R', bound=Record)


def _read_record(path: Path, kind: type[_R], *, expected_sha256: str, limit: int) -> _R:
    """Pin exact bounded file bytes before parsing; do not normalize an input record."""
    FileRef('digest', expected_sha256, 0)
    data = read_relative(path.parent, path.name, limit=limit)
    if _sha(data) != expected_sha256:
        raise ValueError('record trusted digest mismatch')
    record = kind.from_bytes(data, expected_sha256=expected_sha256)
    if record.to_bytes() != data:
        raise ValueError('record must contain exact canonical JSON bytes')
    return record


def _prepare(root: Path, matrix_path: Path, *, expected_sha256: str, reviewer: str) -> dict[str, str]:
    """One-shot private preparation, not archive publication or new admission.

    The caller retains exclusive trusted-local custody of matrix/provenance and
    outputs throughout. Residue is diagnostic only: never resume or auto-delete it.
    Only this command imports the audio lane; archive operations remain stdlib-only.
    """
    import soundfile as sf
    from aimusic.audio.review_matrix import ReviewMatrix
    from aimusic.audio.review_presentation import write_presentation
    from aimusic.audio.review_blinding import _reviewer, bind_presentations, create_assignment, create_blind_map

    _reviewer(reviewer)
    root, matrix_path = root.absolute(), matrix_path.absolute()
    if matrix_path.parent != root:
        raise ValueError('matrix must be a direct child of the explicit input root')
    matrix = _read_record(matrix_path, ReviewMatrix, expected_sha256=expected_sha256, limit=MAX_FILE_BYTES)
    for name in ('prepare-started.json', 'prepared.json', 'bundle.json', 'media', 'assignments'):
        _destination(root / name)
    private = root / 'private'
    if not private.exists():
        _destination(private)
    else:
        _destination(private / 'blind-map.json')
    # Create-only reservation survives all later failures, including process death.
    # It is not a completion record; even an empty media directory forbids reuse.
    _publish_files(root, {'prepare-started.json': canonical_bytes({'matrix_sha256': expected_sha256})})
    for directory in (root / 'media', root / 'assignments'):
        directory.mkdir()
    if not private.exists():
        private.mkdir()
    blind = create_blind_map(matrix)
    renders = {r.sha256: r for r in matrix.renders}
    presentations = []
    for (stimulus, _), exposure in zip(blind.conditions, matrix.exposures):
        data = exposure.wav.read_verified(root)
        _validate_wav(data)
        # libsndfile supports both PCM and IEEE float; frames include every channel.
        with sf.SoundFile(io.BytesIO(data)) as stream:
            frames = stream.frames
        presentations.append(write_presentation(root, render=renders[exposure.render_sha256],
                             source_wav=exposure.wav, output=f'media/{stimulus}.wav',
                             start_sample=0, stop_sample=frames, gain=1.0))
    bundle = bind_presentations(matrix, blind, tuple(presentations))
    _publish_files(private, {'blind-map.json': blind.to_bytes()})
    assignment = create_assignment(bundle, reviewer, root=root, expected_sha256=bundle.sha256)
    assignment_path = root / assignment.reference(bundle).document.path
    _publish_files(assignment_path.parent, {assignment_path.name: assignment.to_bytes()})
    # Verify the input again before publishing the final private records together.
    _read_record(matrix_path, ReviewMatrix, expected_sha256=expected_sha256, limit=MAX_FILE_BYTES)
    bundle.verify(root, expected_sha256=bundle.sha256)
    assignment.reference(bundle).document.read_verified(root)
    result = {'schema': 'aimusic.review-archive-result/1', 'operation': 'prepare',
              'input_root': str(root), 'matrix': str(matrix_path), 'matrix_sha256': matrix.sha256,
              'blind_map': str(root / bundle.blind_map.path), 'blind_map_sha256': blind.sha256,
              'bundle': str(root / 'bundle.json'), 'bundle_sha256': bundle.sha256,
              'assignment': str(assignment_path), 'assignment_sha256': assignment.sha256}
    _publish_files(root, {'bundle.json': bundle.to_bytes(), 'prepared.json': canonical_bytes(result)})
    return result


def _cli_digest(value: str) -> str:
    try:
        FileRef('digest', value, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('expected independently trusted lowercase SHA-256') from exc
    return value


def main(argv: list[str] | None = None) -> int:
    """Explicit offline operations; canonical success JSON only after completion.

    Usage errors exit 2. Operational failures return 1 with canonical error JSON
    on stderr and no stdout. No operation serves, opens a browser, discovers an
    asset, or obtains trust from a received archive/completion record.
    """
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest='operation', required=True)
    prepare = commands.add_parser('prepare', allow_abbrev=False, help='one-shot private full-clip preparation; requires [audio]')
    for name in ('input-root', 'matrix'):
        prepare.add_argument('--' + name, type=Path, required=True)
    prepare.add_argument('--expected-matrix-sha256', type=_cli_digest, required=True)
    prepare.add_argument('--reviewer', required=True, help='pseudonym, not a real identity')
    assemble = commands.add_parser('assemble', allow_abbrev=False, help='publish one reviewer ZIP create-only')
    for name in ('input-root', 'bundle', 'assignment', 'out'):
        assemble.add_argument('--' + name, type=Path, required=True)
    for name in ('bundle', 'assignment'):
        assemble.add_argument('--expected-' + name + '-sha256', type=_cli_digest, required=True)
    for name in ('verify', 'extract', 'verify-extracted'):
        command = commands.add_parser(name, allow_abbrev=False)
        command.add_argument('--archive', type=Path, required=True)
        command.add_argument('--expected-sha256', type=_cli_digest, required=True,
                             help='digest received independently, never read from the archive')
        if name == 'extract':
            command.add_argument('--out', type=Path, required=True)
        elif name == 'verify-extracted':
            command.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args(argv)
    result = {'schema': 'aimusic.review-archive-result/1', 'operation': args.operation}
    try:
        if args.operation == 'prepare':
            result = _prepare(args.input_root, args.matrix, expected_sha256=args.expected_matrix_sha256,
                              reviewer=args.reviewer)
        elif args.operation == 'assemble':
            bundle = _read_record(args.bundle, ReviewBundle, expected_sha256=args.expected_bundle_sha256,
                                  limit=MAX_FILE_BYTES)
            assignment = _read_record(args.assignment, ReviewAssignment,
                                      expected_sha256=args.expected_assignment_sha256, limit=MAX_METADATA_BYTES)
            sha256 = write_review_archive(args.input_root, args.out, bundle=bundle, assignment=assignment,
                                          expected_bundle_sha256=args.expected_bundle_sha256,
                                          expected_assignment_sha256=args.expected_assignment_sha256)
        else:
            sha256 = args.expected_sha256
            if args.operation == 'verify':
                verify_archive(args.archive, expected_sha256=sha256)
            elif args.operation == 'extract':
                extract_archive(args.archive, args.out, expected_sha256=sha256)
            else:
                verify_extracted(args.archive, args.directory, expected_sha256=sha256)
    except (OSError, ValueError, TypeError, RuntimeError, OverflowError, ImportError) as exc:
        message = str(exc)
        if args.operation == 'prepare':
            message += '; preparation failed; preserve any residue and regenerate in a fresh private root'
        sys.stderr.write(canonical_bytes({**result, 'error': message}).decode('ascii'))
        return 1
    if args.operation != 'prepare':
        result['archive_sha256'] = sha256
    sys.stdout.write(canonical_bytes(result).decode('ascii'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
