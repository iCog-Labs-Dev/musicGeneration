"""Native RenderPackage loader for the audio quarantine (M1 PR2)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aimusic.audio.analysis.reconcile import ReconciliationReport, reconcile
from aimusic.core.core_types import Score
from aimusic.core.render_package import (
    STRUCTURE_SCHEMA,
    RenderPackage,
    StructureDoc,
    ContractViolation,
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
    tuning: dict[str, Any]
    role_source: str = "structure.json"
    count_source: str = "score.json"

    @property
    def verified_identity(self) -> dict[str, str]:
        return {"source_hash": self.package.source_hash, "package_hash": self.package.content_hash}

    @property
    def track_roles(self) -> dict[str, str]:
        return {
            t["name"]: "comping" if t["role"] == "comp" else t["role"]
            for t in self.structure.tracks
        }

    @property
    def track_note_counts(self) -> dict[str, int]:
        return self.score.track_event_counts()

    def analysis_document(self) -> dict[str, Any]:
        """Planner values adapted to the existing audio-stage sidecar shape."""
        return {
            **self.structure.to_dict(),
            "provenance": self.provenance,
            "verified_identity": self.verified_identity,
            "inputs_hash": self.package.content_hash,
            "tuning": self.tuning,
            "resolution": self.score.ticks_per_beat,
            "tempo_bpm": self.score.tempo_bpm,
            "bar_boundaries_sec": [row[1] for row in self.structure.bar_table],
            "total_notes": len(self.score),
            "tracks": [
                {**track, "role": self.track_roles[track["name"]],
                 "note_count": self.track_note_counts[track["name"]],
                 "role_source": self.role_source, "count_source": self.count_source}
                for track in self.structure.tracks
            ],
        }


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
    reconciliation: ReconciliationReport


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
        tuning=json.loads(package.tuning_path.read_text(encoding="utf-8")),
    )


def validate_package(ctx: AudioPackageContext) -> ValidationReport:
    """Assert contract invariants and return a human-readable summary."""
    verified = load_render_package(ctx.package.root)
    if verified.content_hash != ctx.package.content_hash:
        raise ContractViolation(
            f"package identity changed since admission: expected={ctx.package.content_hash}, "
            f"actual={verified.content_hash}; reload the context"
        )
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
        reconciliation=reconcile(ctx),
    )
