"""Descriptor-relative POSIX mechanics. No public file is ever opened writable.

Directory descriptors remain alive through cleanup, including when the selected
path has been swapped. This is a trusted-local-output protocol, not protection
against a hostile process able to mutate the private transaction directory.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
import os
from pathlib import Path
import shutil
import stat
from typing import ContextManager, Iterator

from aimusic.audio.publication_records import FileIdentity, Journal, Ownership
from aimusic.audio.publication_transaction import transaction


def _required_flag(name: str) -> int:
    try:
        return getattr(os, name)
    except AttributeError as exc:
        raise NotImplementedError(f'Required POSIX primitive {name} is unavailable') from exc


def _identity(info: os.stat_result) -> FileIdentity:
    return FileIdentity.from_stat(info)


@dataclass(frozen=True)
class PosixBackend:
    selected: Path
    path: Path
    fd: int
    ancestors: tuple[tuple[Path, FileIdentity], ...]

    def verify(self) -> None:
        for path, identity in self.ancestors:
            info = path.lstat()
            if not stat.S_ISDIR(info.st_mode) or _identity(info) != identity:
                raise ValueError(f'Output directory identity changed: {path}')

    def entries(self) -> list[str]:
        return os.listdir(self.fd)

    def info(self, name: str) -> os.stat_result | None:
        try:
            return os.stat(name, dir_fd=self.fd, follow_symlinks=False)
        except FileNotFoundError:
            return None

    def transaction(self, staging: Path, journal: Journal) -> ContextManager[tuple[Ownership, ...]]:
        return transaction(self, staging, journal)

    @contextmanager
    def workspace(self, name: str) -> Iterator[PosixWorkspace]:
        os.mkdir(name, 0o700, dir_fd=self.fd)
        fd = os.open(name, os.O_RDONLY | _required_flag('O_DIRECTORY') | _required_flag('O_NOFOLLOW'), dir_fd=self.fd)
        try:
            yield PosixWorkspace(self, name, fd, _identity(os.fstat(fd)))
        finally:
            os.close(fd)


@dataclass
class PosixWorkspace:
    root: PosixBackend
    name: str
    fd: int
    identity: FileIdentity
    owned: list[Ownership] = field(default_factory=list)

    @property
    def recovery_path(self) -> Path:
        return self.root.path / self.name

    def prepare(self, originals: tuple[Ownership, ...], staged: tuple[str, ...]) -> None:
        pass  # The POSIX directory descriptor is pinned at backend acquisition.

    def _create(self, name: str) -> int:
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _required_flag('O_NOFOLLOW'), 0o600, dir_fd=self.fd)
        self.owned.append(Ownership(name, _identity(os.fstat(fd))))
        return fd

    def journal(self, data: bytes) -> None:
        # Private evidence only. An interrupted write is ambiguous residue, never
        # permission to automatically recover/delete. This is not crash atomic.
        record = next((o for o in self.owned if o.name == 'journal.json'), None)
        fd = (self._create('journal.json') if record is None else
              os.open('journal.json', os.O_WRONLY | _required_flag('O_NOFOLLOW'), dir_fd=self.fd))
        with os.fdopen(fd, 'wb') as stream:
            if record is not None and _identity(os.fstat(fd)) != record.identity:
                raise ValueError('Transaction journal identity changed')
            stream.write(data)
            stream.truncate()
            stream.flush()
            os.fsync(fd)

    def stage(self, source: Path, name: str) -> FileIdentity:
        source_fd = os.open(source, os.O_RDONLY | _required_flag('O_NOFOLLOW') | _required_flag('O_NONBLOCK'))
        with os.fdopen(source_fd, 'rb') as data:
            if not stat.S_ISREG(os.fstat(source_fd).st_mode):
                raise ValueError(f'Staged source must be a regular file: {source}')
            fd = self._create(name)
            with os.fdopen(fd, 'wb') as target:
                shutil.copyfileobj(data, target)
                target.flush()
                os.fsync(fd)
                return _identity(os.fstat(fd))

    def backup(self, name: str, private: str, identity: FileIdentity) -> None:
        os.link(name, private, src_dir_fd=self.root.fd, dst_dir_fd=self.fd, follow_symlinks=False)
        actual = _identity(os.stat(private, dir_fd=self.fd, follow_symlinks=False))
        self.owned.append(Ownership(private, actual))
        if actual != identity:
            raise ValueError(f'Output identity changed before backup: {name}')

    def publish(self, private: str, name: str, *, replace: bool) -> None:
        if replace:
            os.replace(private, name, src_dir_fd=self.fd, dst_dir_fd=self.root.fd)
        else:
            os.link(private, name, src_dir_fd=self.fd, dst_dir_fd=self.root.fd, follow_symlinks=False)

    def remove_public(self, owner: Ownership) -> None:
        info = self.root.info(owner.name)
        if info is not None and stat.S_ISREG(info.st_mode) and _identity(info) == owner.identity:
            os.unlink(owner.name, dir_fd=self.root.fd)

    def restore(self, private: str, original: Ownership, published: FileIdentity | None) -> None:
        info = self.root.info(original.name)
        if info is not None:
            current = _identity(info)
            if current == original.identity:
                return  # Publication failed before replacing this leaf.
            if current != published or not stat.S_ISREG(info.st_mode):
                raise ValueError(f'Cannot restore over unowned output: {original.name}')
        backup = os.stat(private, dir_fd=self.fd, follow_symlinks=False)
        if _identity(backup) != original.identity:
            raise ValueError(f'Recovery identity changed: {private}')
        os.replace(private, original.name, src_dir_fd=self.fd, dst_dir_fd=self.root.fd)

    def cleanup(self, *, keep_backups: bool = False) -> None:
        for owner in reversed(self.owned):
            if keep_backups and (owner.name == 'journal.json' or owner.name.startswith('backup-')):
                continue
            try:
                info = os.stat(owner.name, dir_fd=self.fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            if (stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)) and _identity(info) == owner.identity:
                os.unlink(owner.name, dir_fd=self.fd)
            else:
                raise ValueError(f'Private transaction identity changed: {owner.name}')
        if not keep_backups:
            directory_info = self.root.info(self.name)
            if directory_info is None or _identity(directory_info) != self.identity or not stat.S_ISDIR(directory_info.st_mode):
                raise ValueError(f'Transaction directory identity changed: {self.name}')
            os.rmdir(self.name, dir_fd=self.root.fd)

    def discard_staged(self) -> None:
        # Delete staged create-only hardlinks before accepting the transaction,
        # while backups and journal still allow rollback after path verification.
        for owner in self.owned:
            if owner.name.startswith('.stems-'):
                try:
                    info = os.stat(owner.name, dir_fd=self.fd, follow_symlinks=False)
                except FileNotFoundError:
                    continue
                if _identity(info) != owner.identity or not stat.S_ISREG(info.st_mode):
                    raise ValueError(f'Staged identity changed: {owner.name}')
                os.unlink(owner.name, dir_fd=self.fd)


@contextmanager
def open_backend(output: Path | str) -> Iterator[PosixBackend]:
    selected = Path(os.path.abspath(output))
    if selected.is_symlink():
        raise ValueError(f'Output root is a symlink: {selected}')
    flags = os.O_RDONLY | _required_flag('O_DIRECTORY') | _required_flag('O_NOFOLLOW')
    with ExitStack() as stack:
        fd = os.open(selected.anchor, flags)
        stack.callback(os.close, fd)
        path = Path(selected.anchor)
        ancestors = [(path, _identity(os.fstat(fd)))]
        for part in selected.parts[1:]:
            try:
                os.mkdir(part, dir_fd=fd)
            except FileExistsError:
                pass
            fd = os.open(part, flags, dir_fd=fd)
            stack.callback(os.close, fd)
            path /= part
            ancestors.append((path, _identity(os.fstat(fd))))
        root = PosixBackend(selected, path, fd, tuple(ancestors))
        root.verify()
        yield root
