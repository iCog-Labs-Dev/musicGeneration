"""Private input snapshots plus mutation checks at publication boundaries."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from aimusic.audio.render.registry import file_digest


@dataclass(frozen=True)
class SnapshotFile:
    source: Path
    path: Path
    sha256: str | None

    def verify(self) -> None:
        for path in (self.source, self.path):
            if self.sha256 is None:
                if not path.exists():
                    continue
            elif path.is_file() and file_digest(path) == self.sha256:
                continue
            raise ValueError(f'input bytes changed since snapshot: {path.name}; retry')


@dataclass(frozen=True)
class InputSnapshot:
    files: tuple[SnapshotFile, ...]

    @classmethod
    def capture(cls, source: Path, directory: Path, *, package: bool = False) -> InputSnapshot:
        names = ('score.mid', 'score.json', 'structure.json', 'tuning.json', 'manifest.json', 'beatstates.jsonl')
        paths = [source / name for name in names] if package else [source]
        directory.mkdir()
        files = []
        for path in paths:
            target = directory / path.name
            if package and path.name == 'beatstates.jsonl' and not path.exists():
                digest = None  # Absence is part of a legacy package snapshot too.
            else:
                data = path.read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                target.write_bytes(data)
            files.append(SnapshotFile(path, target, digest))
        snapshot = cls(tuple(files))
        snapshot.verify()
        return snapshot

    def verify(self) -> None:
        for file in self.files:
            file.verify()
