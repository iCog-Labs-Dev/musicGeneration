"""Platform-neutral transaction phases, ownership, and recovery policy."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace as evolve
import os
from pathlib import Path
import stat
from typing import ContextManager, Iterator, Protocol

from aimusic.audio.publication_records import FileIdentity, Journal, Ownership, Phase, canonical


class TransactionWorkspace(Protocol):
    """Private, same-filesystem storage operated through a pinned backend."""
    @property
    def recovery_path(self) -> Path: ...
    def prepare(self, originals: tuple[Ownership, ...], staged: tuple[str, ...]) -> None: ...
    def journal(self, data: bytes) -> None: ...
    def stage(self, source: Path, name: str) -> FileIdentity: ...
    def backup(self, name: str, private: str, identity: FileIdentity) -> None: ...
    def publish(self, private: str, name: str, *, replace: bool) -> None: ...
    def remove_public(self, owner: Ownership) -> None: ...
    def restore(self, private: str, original: Ownership, published: FileIdentity | None) -> None: ...
    def discard_staged(self) -> None: ...
    def cleanup(self, *, keep_backups: bool = False) -> None: ...


class TransactionBackend(Protocol):
    @property
    def path(self) -> Path: ...
    def verify(self) -> None: ...
    def entries(self) -> list[str]: ...
    def info(self, name: str) -> os.stat_result | None: ...
    def workspace(self, name: str) -> ContextManager[TransactionWorkspace]: ...


def _check_casefold(root: TransactionBackend, name: str) -> None:
    if any(entry != name and entry.casefold() == name.casefold() for entry in root.entries()):
        raise FileExistsError(f'Casefold output alias already exists: {root.path / name}')


@contextmanager
def transaction(root: TransactionBackend, staging: Path, journal: Journal) -> Iterator[tuple[Ownership, ...]]:
    """Shared journal/rollback policy; POSIX owns only descriptor operations.

    Replacement intentionally retains the old groove no-follow rename semantics.
    Concurrent replacement writers are outside the local-output threat model;
    create-only publication uses an atomic no-replace link instead.
    """
    with root.workspace(journal.invocation) as work:
        originals = {o.name: o for o in journal.originals}
        private = {name: f'.stems-{index}' for index, name in enumerate(journal.names)}
        backups = {name: f'backup-{index}' for index, name in enumerate(journal.names) if name in originals}
        try:
            # Windows must reserve empty leaves and verify its directory pin
            # before any payload or journal content can enter the output tree.
            work.prepare(journal.originals, tuple(private.values()))
            work.journal(canonical(journal))
            for name in journal.names:
                identity = work.stage(staging / name, private[name])
                journal = evolve(journal, staged=(*journal.staged, Ownership(name, identity)))
                work.journal(canonical(journal))
            root.verify()
            for name in journal.names:
                _check_casefold(root, name)
            # Preserve all prior copies before touching the first public name.
            for name, backup in backups.items():
                work.backup(name, backup, originals[name].identity)
                journal = evolve(journal, backups=(*journal.backups, Ownership(backup, originals[name].identity)))
                work.journal(canonical(journal))
            journal = evolve(journal, phase=Phase.PREPARED)
            work.journal(canonical(journal))
            for owner in journal.staged:
                root.verify()
                _check_casefold(root, owner.name)
                journal = evolve(journal, phase=Phase.PUBLISHING)
                work.journal(canonical(journal))
                work.publish(private[owner.name], owner.name, replace=journal.replace)
                journal = evolve(journal, published=(*journal.published, owner))
                work.journal(canonical(journal))
                root.verify()
                _check_casefold(root, owner.name)
            journal = evolve(journal, phase=Phase.PUBLISHED)
            work.journal(canonical(journal))
            yield journal.published  # All names exist; caller rejection still rolls back.
            work.discard_staged()
            root.verify()
            for owner in journal.published:
                _check_casefold(root, owner.name)
                info = root.info(owner.name)
                if info is None or FileIdentity.from_stat(info) != owner.identity or not stat.S_ISREG(info.st_mode):
                    raise ValueError(f'Published output identity changed: {owner.name}')
            journal = evolve(journal, phase=Phase.COMMITTED)
            work.journal(canonical(journal))
        except BaseException:
            rollback_errors = []
            journal = evolve(journal, phase=Phase.ROLLING_BACK)
            try:
                work.journal(canonical(journal))
            except (OSError, ValueError) as exc:
                rollback_errors.append(exc)
            published = {o.name: o.identity for o in journal.published}
            for owner in reversed(journal.published):
                if owner.name not in originals:
                    try:
                        work.remove_public(owner)
                    except (OSError, ValueError) as exc:
                        rollback_errors.append(exc)
            for name, backup in reversed(tuple(backups.items())):
                # A failed backup operation has not transferred ownership.
                if not any(o.name == backup for o in journal.backups):
                    continue
                try:
                    work.restore(backup, originals[name], published.get(name))
                except (OSError, ValueError) as exc:
                    rollback_errors.append(exc)
            if not rollback_errors:
                try:
                    work.cleanup()
                except (OSError, ValueError) as exc:
                    rollback_errors.append(exc)
            if rollback_errors:
                journal = evolve(journal, phase=Phase.RECOVERY)
                try:
                    work.journal(canonical(journal))
                    work.cleanup(keep_backups=True)
                except (OSError, ValueError):
                    pass  # Keep every remaining identity and never destroy recovery.
                raise OSError(f'Publication rollback failed; recovery files retained in '
                              f'{work.recovery_path}') from rollback_errors[0]
            raise
        # Commit is explicit. Cleanup failure leaves a committed journal/residue,
        # not permission to roll back after some old copies have been discarded.
        try:
            work.cleanup()
        except (OSError, ValueError) as exc:
            raise OSError(f'Publication cleanup failed; recovery files retained in '
                          f'{work.recovery_path}') from exc
