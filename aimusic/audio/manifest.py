"""Provenance manifests and JSON sidecars for audio stages."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class StageManifest:
    stage: str
    inputs_hash: str
    config_subset: dict[str, Any] = field(default_factory=dict)
    code_version: str = "0.1.0"
    wall_clock_utc: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat()
    )
    metadata: dict[str, Any] = field(default_factory=dict)
    output_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_sidecar(self, artifact_path: Path | str) -> Path:
        path = Path(artifact_path)
        sidecar = Path(str(path) + ".json")
        sidecar.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
        )
        return sidecar


def content_hash_files(paths: list[Path | str], *extra: str) -> str:
    """SHA-256 of concatenated file bytes + optional extra strings (config/code)."""
    h = hashlib.sha256()
    for p in paths:
        data = Path(p).read_bytes()
        h.update(data)
    for e in extra:
        h.update(e.encode("utf-8"))
    return h.hexdigest()


def write_manifest(
    stage: str,
    artifact_path: Path | str,
    input_paths: list[Path | str],
    *,
    config_subset: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    code_version: str = "0.1.0",
) -> Path:
    """Create and write a provenance sidecar for an artifact."""
    cfg = config_subset or {}
    inputs_hash = content_hash_files(
        input_paths,
        json.dumps(cfg, sort_keys=True),
        code_version,
    )
    manifest = StageManifest(
        stage=stage,
        inputs_hash=inputs_hash,
        config_subset=cfg,
        code_version=code_version,
        metadata=metadata or {},
        output_path=str(artifact_path),
    )
    return manifest.write_sidecar(artifact_path)
