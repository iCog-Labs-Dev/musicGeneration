"""Shared create-only/replacement publication policy for local audio artifacts.

Complete files are staged privately on the output filesystem. Each live
transaction retains identity evidence and any previous copies until the caller
accepts it. Crash residue is reported, never implicitly trusted or removed.
Windows uses retained-handle same-directory publication; no path-based mutation
fallback is provided for unsupported native operations.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import errno
import os
from pathlib import Path
import stat
from typing import ContextManager, Iterator, Protocol
from uuid import uuid4

from aimusic.audio.publication_records import (
    FileIdentity, Journal, Ownership, Phase as Phase, PRIVATE_PREFIXES, canonical as canonical, validate_names,
)

__all__ = ['FileIdentity', 'Ownership', 'Journal', 'Phase', 'canonical',
           'validate_names', 'PublicationBackend', 'PinnedOutput', 'pin_output']

class PublicationBackend(Protocol):
    @property
    def path(self) -> Path: ...

    def verify(self) -> None: ...
    def entries(self) -> list[str]: ...
    def info(self, name: str) -> os.stat_result | None: ...
    def transaction(self, staging: Path, journal: Journal) -> ContextManager[tuple[Ownership, ...]]: ...


def _unsupported(output: Path | str, exc: BaseException) -> RuntimeError:
    return RuntimeError(f'Publication unavailable for {output}: {exc}. '
                        'Use a supported local filesystem and runtime; no unsafe fallback is allowed.')


def _open_backend(output: Path | str) -> ContextManager[PublicationBackend]:
    if os.name == 'nt':
        from aimusic.audio.publication_windows import open_backend
        return open_backend(output)
    from aimusic.audio.publication_posix import open_backend as open_posix
    return open_posix(output)


@dataclass(frozen=True)
class PinnedOutput:
    backend: PublicationBackend
    _active: set[str] = field(default_factory=set, repr=False, compare=False)

    @property
    def path(self) -> Path:
        return self.backend.path

    def verify(self) -> None:
        self.backend.verify()

    def _residue(self) -> None:
        residue = sorted(name for name in self.backend.entries()
                         if name.casefold().startswith(PRIVATE_PREFIXES) and
                         not any(name == active or name.startswith(active + '.') for active in self._active))
        if residue:
            raise RuntimeError(f'Unfinished publication transaction residue in {self.path}: '
                               f'{", ".join(residue)}. Inspect and recover these entries explicitly; '
                               'they have not been reused or deleted.')

    def _check(self, names: tuple[str, ...], *, replace: bool,
               sources: tuple[Path, ...] = ()) -> tuple[Ownership, ...]:
        self.verify()
        self._residue()
        entries = self.backend.entries()
        source_ids = {FileIdentity.from_stat(source.stat()) for source in sources}
        source_paths = {source.resolve() for source in sources}
        originals = []
        identities: set[FileIdentity] = set()
        resolved_outputs: set[Path] = set()
        for name in names:
            aliases = [entry for entry in entries if entry.casefold() == name.casefold()]
            if any(entry != name for entry in aliases):
                raise FileExistsError(f'Casefold output alias already exists: {self.path / name}')
            info = self.backend.info(name)
            if not replace and info is not None:
                raise FileExistsError(f'Output already exists: {self.path / name}')
            destination = self.path / name
            resolved = destination.resolve()
            if resolved in source_paths:
                raise ValueError(f'Output aliases source/input: {destination}')
            if resolved in resolved_outputs:
                raise ValueError(f'Output aliases another output: {destination}')
            resolved_outputs.add(resolved)
            if info is None:
                continue
            # Preserve groove's POSIX leaf-link replacement (never following it
            # for a mutation). Ancestor links, staged links, directory targets,
            # source aliases, and all create-only collisions remain forbidden.
            if not (stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)) or getattr(info, 'st_file_attributes', 0) & 0x400:
                raise ValueError(f'Output is a reparse point or not a regular file: {name}')
            identity = FileIdentity.from_stat(info)
            try:
                target_info = destination.stat() if stat.S_ISLNK(info.st_mode) else info
            except FileNotFoundError:
                target_info = info  # Retain/restore a dangling leaf link too.
            if not (stat.S_ISREG(target_info.st_mode) or stat.S_ISLNK(target_info.st_mode)):
                raise ValueError(f'Output symlink target is not a regular file: {name}')
            target_identity = FileIdentity.from_stat(target_info)
            if target_identity in source_ids:
                raise ValueError(f'Output aliases source/input: {name}')
            if target_identity in identities:
                raise ValueError(f'Output aliases another output: {name}')
            identities.add(target_identity)
            originals.append(Ownership(name, identity))
        return tuple(originals)

    def check_new_files(self, names: list[str]) -> None:
        self._check(validate_names(names), replace=False)

    def publish(self, staging: Path, names: list[str], *, replace: bool = False,
                sources: tuple[Path, ...] = ()) -> None:
        with self.publication(staging, names, replace=replace, sources=sources):
            pass

    @contextmanager
    def publication(self, staging: Path, names: list[str], *, replace: bool = False,
                    sources: tuple[Path, ...] = ()) -> Iterator[tuple[Ownership, ...]]:
        """Yield recorded published ownership for caller verification before commit."""
        plan = validate_names(names)
        originals = self._check(plan, replace=replace, sources=tuple(Path(p) for p in sources))
        # Validate the *whole* staged plan before creating any transaction entry.
        for name in plan:
            info = (staging / name).lstat()
            if not stat.S_ISREG(info.st_mode) or getattr(info, 'st_file_attributes', 0) & 0x400:
                raise ValueError(f'Staged source must be a regular file, not a symlink: {name}')
        invocation = ('.groove-' if replace else '.stems-') + uuid4().hex
        journal = Journal(invocation, plan, replace, originals=originals)
        self._active.add(invocation)
        try:
            with self.backend.transaction(staging, journal) as published:
                yield published
        except NotImplementedError as exc:
            raise _unsupported(self.path, exc) from exc
        finally:
            self._active.remove(invocation)


@contextmanager
def pin_output(output: Path | str) -> Iterator[PinnedOutput]:
    try:
        backend = _open_backend(output)
        with backend as opened:
            root = PinnedOutput(opened)
            root._residue()
            yield root
    except NotImplementedError as exc:
        raise _unsupported(output, exc) from exc
    except OSError as exc:
        if exc.errno in (errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP):
            raise _unsupported(output, exc) from exc
        raise


