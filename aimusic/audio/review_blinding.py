"""Opaque naming and external assignments. No pages or whole-tree publication.

The graph, condition map and order seed are private. Only ReviewAssignment and
ReviewBundle.public_projection are reviewer data; neither exposes provenance.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hmac
from pathlib import Path
import re
import secrets
from typing import TYPE_CHECKING, Literal

from aimusic.audio.review_bundle import (
    AssignmentReference, FileRef, GraphRecord, PANEL_SIZES, Panel, PresentationNode,
    PrivateBlindMap, ReviewBundle, Stimulus, canonical_bytes,
)
# Matrix generation loads renderers; assignments and archive readers do not.
if TYPE_CHECKING:
    from aimusic.audio.review_matrix import ReviewMatrix


def create_blind_map(matrix: ReviewMatrix, *, random_bytes: Callable[[int], bytes] = secrets.token_bytes) -> PrivateBlindMap:
    """Allocate independent 256-bit IDs and a secret seed once; retain the map.

    Entropy is explicitly injectable for tests only. IDs do not encode conditions,
    paths, hashes or exposure indices. Duplicate draws retry at most 32 times per
    token, then fail closed. Filesystem collisions are still the publisher's job.
    """
    matrix.__post_init__()
    used: set[str] = set()

    def token() -> str:
        for _ in range(32):
            value = random_bytes(32)
            if type(value) is not bytes or len(value) != 32:
                raise ValueError('expected 32 bytes of blinding entropy')
            result = value.hex()
            if result not in used:
                used.add(result)
                return result
        raise ValueError('blinding entropy collision limit exceeded')

    conditions = tuple(('s' + token(), e.condition) for e in matrix.exposures)
    return PrivateBlindMap(conditions, order_seed=token())


def bind_presentations(matrix: ReviewMatrix, blind: PrivateBlindMap,
                       presentations: tuple[PresentationNode, ...]) -> ReviewBundle:
    """Build private graph metadata only; no I/O, assembly or admission claim.

    Pass the map's IDs as media/<ID>.wav outputs to write_presentation first.
    Presentations may arrive in any order, but must cover the exact mapped slots
    and link to each slot's admitted render and source WAV. Save the map privately
    at the returned blind_map path before generating any reviewer assignment.
    """
    matrix.__post_init__()
    blind.__post_init__()
    ids = {condition: stimulus for stimulus, condition in blind.conditions}
    if len(ids) != 14 or set(ids) != {e.condition for e in matrix.exposures}:
        raise ValueError('blind map must cover exact matrix conditions once')
    by_path = {p.output.path: p for p in presentations}
    expected = {f'media/{s}.wav' for s in ids.values()}
    if len(presentations) != 14 or set(by_path) != expected:
        raise ValueError('presentations must cover exact opaque media paths once')
    stimuli = []
    for exposure in matrix.exposures:
        stimulus = ids[exposure.condition]
        presentation = by_path[f'media/{stimulus}.wav']
        if presentation.render_sha256 != exposure.render_sha256 or presentation.source_wav != exposure.wav:
            raise ValueError('presentation does not match matrix exposure')
        stimuli.append(Stimulus(stimulus, presentation.sha256))
    bundle = ReviewBundle('aimusic.review-bundle/1', matrix.sources, matrix.renders,
                          tuple(by_path[path] for path in sorted(by_path)),
                          tuple(sorted(stimuli, key=lambda s: s.id)),
                          tuple(Panel(p.purpose, tuple(ids[c] for c in p.stimuli)) for p in matrix.panels),
                          FileRef('private/blind-map.json', blind.sha256, len(blind.to_bytes())),
                          'assignments', 'findings')
    blind.validate_bundle(bundle)
    return bundle


@dataclass(frozen=True)
class AssignedStimulus(GraphRecord):
    id: str
    media: FileRef

    def validate(self) -> None:
        if not re.fullmatch('s[0-9a-f]{2,64}', self.id) or self.media.path != f'media/{self.id}.wav':
            raise ValueError('assigned stimulus requires matching opaque media path')


@dataclass(frozen=True)
class ReviewAssignment(GraphRecord):
    """Reviewer-safe document; pair stimuli are ordered A then B, never conditions.

    Arrays and fields are immutable and allowlisted. The document's content hash
    names its file. Its AssignmentReference is external to the immutable bundle.
    """
    schema: Literal['aimusic.review-assignment/1']
    bundle_sha256: str
    reviewer: str
    panels: tuple[Panel, ...]
    stimuli: tuple[AssignedStimulus, ...]

    def validate(self) -> None:
        _reviewer(self.reviewer)
        if not re.fullmatch('[0-9a-f]{64}', self.bundle_sha256):
            raise ValueError('expected bundle SHA-256 digest')
        if tuple(p.purpose for p in self.panels) != tuple(PANEL_SIZES):
            raise ValueError('assignment requires all five ordered purpose panels')
        ids = tuple(s.id for s in self.stimuli)
        exposures = tuple(s for p in self.panels for s in p.stimuli)
        if (len(ids) != 14 or len(set(ids)) != 14 or len(set(exposures)) != 14 or set(ids) != set(exposures)):
            raise ValueError('assignment requires exactly fourteen unique exposures')
        if ids != tuple(sorted(ids)):
            raise ValueError('assignment media inventory must use neutral ID order')

    def reference(self, bundle: ReviewBundle) -> AssignmentReference:
        """Locate this document using its bundle, without serializing private graph fields."""
        result = AssignmentReference(self.bundle_sha256, self.reviewer,
                                     FileRef(f'{bundle.assignment_directory}/a{self.sha256}.json',
                                             self.sha256, len(self.to_bytes())),
                                     tuple((s.id, s.media.sha256) for s in self.stimuli))
        result.validate_bundle(bundle)
        return result

    def verify(self, root: Path, *, expected_sha256: str) -> None:
        """Check trusted assignment identity and media without private graph access."""
        self.__post_init__()
        if self.sha256 != expected_sha256:
            raise ValueError('assignment digest mismatch')
        for stimulus in self.stimuli:
            stimulus.media.read_verified(root)

    def validate_bundle(self, bundle: ReviewBundle) -> None:
        self.reference(bundle)  # Verify bundle/media digests and declared document directory.
        presentations = {p.sha256: p for p in bundle.presentations}
        media = {s.id: presentations[s.presentation_sha256].output for s in bundle.stimuli}
        if any(s.media != media[s.id] for s in self.stimuli):
            raise ValueError('assignment media reference mismatch')
        panels: dict[str, Panel] = {p.purpose: p for p in bundle.panels}
        for panel in self.panels:
            expected = panels[panel.purpose].stimuli
            if set(panel.stimuli) != set(expected) or (len(expected) != 2 and panel.stimuli != expected):
                raise ValueError('assignment panel mismatch')


def _reviewer(value: str) -> None:
    if type(value) is not str or not re.fullmatch('[A-Za-z0-9_-]{1,64}', value):
        raise ValueError('bounded reviewer pseudonym required')


def create_assignment(bundle: ReviewBundle, reviewer: str, *, root: Path, expected_sha256: str) -> ReviewAssignment:
    """Verify the private graph and return one reviewer-safe assignment, without I/O writes.

    HMAC-SHA256 keyed by private map entropy ranks the three pairs using version,
    bundle identity and pseudonym. Reverse the first 1 or 2 ranked pairs (selected
    by another keyed bit). Both orientations occur; counts differ by exactly one.
    Balance is within a reviewer, not a guarantee across arbitrary reviewer sets.
    The fixed canonical/role order is unchanged. No global RNG state is consumed.
    Public inputs alone cannot recover which condition is A. Retaining the map is
    necessary for repeatability; allocating a fresh map creates a new bundle.
    """
    _reviewer(reviewer)
    public = bundle.public_projection(root=root, expected_sha256=expected_sha256)
    blind = PrivateBlindMap.from_bytes(bundle.blind_map.read_verified(root))
    if blind.order_seed is None:
        raise ValueError('private ordering seed required; allocate a new blinded bundle')

    key = bytes.fromhex(blind.order_seed)

    def rank(label: str) -> bytes:
        return hmac.digest(key,
                           canonical_bytes(('aimusic.review-assignment/1', bundle.sha256, reviewer, label)), 'sha256')

    pairs = sorted(('parity', 'timing', 'tuning'), key=lambda p: (rank(p), p))
    reversed_pairs = set(pairs[:1 + (rank('balance')[0] & 1)])
    panels: dict[str, Panel] = {p.purpose: p for p in bundle.panels}
    assigned = tuple(Panel(panels[p].purpose, tuple(reversed(panels[p].stimuli))
                           if p in reversed_pairs else panels[p].stimuli) for p in PANEL_SIZES)
    result = ReviewAssignment('aimusic.review-assignment/1', bundle.sha256, reviewer, assigned,
                              tuple(AssignedStimulus(row['id'], FileRef.from_dict(row['media'])) for row in public['stimuli']))
    result.validate_bundle(bundle)
    return result
