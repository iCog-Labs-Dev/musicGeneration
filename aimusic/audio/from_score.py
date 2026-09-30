"""Native RenderPackage loader for the audio quarantine (M1 PR2)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aimusic.core.core_types import Score
from aimusic.core.render_package import (
    STRUCTURE_SCHEMA,
    RenderPackage,
    StructureDoc,
    assert_contract_invariants,
    load_render_package,
)


@dataclass(frozen=True)
class AudioPackageContext:
    """Typed view of a RenderPackage ready for audio stages."""

    package: RenderPackage
    score: Score
    structure: StructureDoc
    provenance: str
    run_id: str


@dataclass(frozen=True)
class ValidationReport:
    """Summary of a validated RenderPackage."""

    run_id: str
    schema: str
    provenance: str
    edo: int
    track_count: int
    track_names: tuple[str, ...]
    microtonal_tracks: tuple[str, ...]
    content_hash: str
    package_root: Path


def _structure_from_dict(data: dict[str, Any]) -> StructureDoc:
    return StructureDoc(
        schema=str(data.get("schema", STRUCTURE_SCHEMA)),
        provenance=str(data.get("provenance", "planner")),
        source_hash=str(data.get("source_hash", "")),
        edo=int(data.get("edo", 12)),
        base_tuning=float(data.get("base_tuning", 0.0)),
        tempo_map=list(data.get("tempo_map") or []),
        meter=list(data.get("meter") or []),
        key=list(data.get("key") or []),
        chords=list(data.get("chords") or []),
        sections=list(data.get("sections") or []),
        bar_table=list(data.get("bar_table") or []),
        tracks=list(data.get("tracks") or []),
    )


def from_score(package_root: Path | str) -> AudioPackageContext:
    """Load and validate a RenderPackage, preferring planner ``structure.json``."""
    package = load_render_package(package_root)
    score = Score.from_dict(json.loads(package.score_path.read_text(encoding="utf-8")))
    structure_raw = json.loads(package.structure_path.read_text(encoding="utf-8"))
    structure = _structure_from_dict(structure_raw)
    manifest = json.loads(package.manifest_path.read_text(encoding="utf-8"))
    run_id = str(manifest.get("run_id", package.root.name))
    return AudioPackageContext(
        package=package,
        score=score,
        structure=structure,
        provenance=structure.provenance,
        run_id=run_id,
    )


def validate_package(ctx: AudioPackageContext) -> ValidationReport:
    """Assert contract invariants and return a human-readable summary."""
    assert_contract_invariants(ctx.package, score=ctx.score)
    microtonal = tuple(
        str(track["name"])
        for track in ctx.structure.tracks
        if track.get("microtonal")
    )
    track_names = tuple(str(track["name"]) for track in ctx.structure.tracks)
    return ValidationReport(
        run_id=ctx.run_id,
        schema=ctx.structure.schema,
        provenance=ctx.provenance,
        edo=int(ctx.structure.edo),
        track_count=len(track_names),
        track_names=track_names,
        microtonal_tracks=microtonal,
        content_hash=ctx.package.content_hash,
        package_root=ctx.package.root,
    )
