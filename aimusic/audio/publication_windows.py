"""Local NTFS publication using retained handles and same-directory renames.

Ancestor pins protect directory identity, NOT the child namespace. Private flat
leaves are reserved empty before the output pin; no content write or public
mutation precedes the identity-verified handoff. All rename/delete authority stays with retained file
handles. Unrelated children are protected by no-replace and identity checks.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import ContextManager, Iterator

from aimusic.audio.publication_records import FileIdentity, Journal, Ownership, canonical
from aimusic.audio.publication_transaction import transaction
from aimusic.audio.windows_native import (
    ATTRIBUTES, DELETE, DIRECTORY_ACCESS, READ, SHARE_ALL, WRITE,
    Handle, WindowsAPI, leaf_name, long_path,
)


@dataclass(frozen=True)
class WindowsBackend:
    path: Path
    api: WindowsAPI
    ancestors: tuple[tuple[Path, Handle, FileIdentity], ...]

    @property
    def identity(self) -> FileIdentity:
        return self.ancestors[-1][2]

    def verify(self) -> None:
        for path, handle, identity in self.ancestors:
            for info in (self.api.information(handle), self._path_information(path)):
                info.require(directory=True)
                if info.identity != identity:
                    raise ValueError(f'Output directory identity changed: {path}')

    def _path_information(self, path: Path):
        with self.api.open(path, ATTRIBUTES, share=SHARE_ALL) as handle:
            return self.api.information(handle)

    def entries(self) -> list[str]:
        self.verify()
        # Read-only enumeration; ancestors remain pinned and identity is checked
        # both sides. This never grants mutation authority based on a path check.
        result = os.listdir(long_path(self.path))
        self.verify()
        return result

    def info(self, name: str) -> os.stat_result | None:
        leaf_name(name)
        try:
            return self._path_information(self.path / name).stat()
        except FileNotFoundError:
            return None

    def transaction(self, staging: Path, journal: Journal) -> ContextManager[tuple[Ownership, ...]]:
        return transaction(self, staging, journal)

    @contextmanager
    def workspace(self, name: str) -> Iterator[WindowsWorkspace]:
        work = WindowsWorkspace(self, name)
        try:
            with work.handles:
                yield work
        finally:
            # Keep the directory pin until *all* retained leaves are closed.
            if work.pin is not None:
                work.pin.close()


@dataclass
class Leaf:
    name: str
    handle: Handle
    identity: FileIdentity
    original: bool = False
    public: bool = False


@dataclass
class WindowsWorkspace:
    root: WindowsBackend
    name: str
    handles: ExitStack = field(default_factory=ExitStack)
    leaves: list[Leaf] = field(default_factory=list)
    reservations: dict[str, Handle] = field(default_factory=dict)
    stages: dict[str, Leaf] = field(default_factory=dict)
    originals: dict[str, Leaf] = field(default_factory=dict)
    record: Leaf | None = None
    pin: Handle | None = None
    ready: bool = False
    committed: bool = False

    @property
    def recovery_path(self) -> Path:
        return self.root.path / self.private('journal.json')

    def private(self, name: str) -> str:
        return self.name + '.' + name

    def retain(self, name: str, *, original: bool = False) -> Leaf:
        if self.pin is not None:
            raise ValueError('Leaf retention must precede the directory pin')
        api = self.root.api
        handle = self.handles.enter_context(api.open(self.root.path / name,
                     READ | DELETE | (0 if original else WRITE), create=not original))
        if not original:
            # CREATE_NEW already grants disposal authority. Retain it even
            # if inspection/validation fails before a Leaf can be recorded.
            self.reservations[name] = handle
        info = api.information(handle)
        info.require(directory=False)
        if info.identity.device != self.root.identity.device:
            raise ValueError('Staging is not on the selected publication volume')
        leaf = Leaf(name, handle, info.identity, original, original)
        self.leaves.append(leaf)
        self.reservations.pop(name, None)
        return leaf

    def verify_leaf(self, leaf: Leaf) -> None:
        info = self.root.api.information(leaf.handle)
        info.require(directory=False)
        if info.identity != leaf.identity:
            raise ValueError(f'Retained file identity changed: {leaf.name}')

    def prepare(self, originals: tuple[Ownership, ...], staged: tuple[str, ...]) -> None:
        # CREATE_NEW grants ownership, not containment: an as-yet unpinned
        # directory could have become a junction. Reserve only empty leaves.
        self.record = self.retain(self.private('journal.json'))
        for name in staged:
            self.stages[name] = self.retain(self.private(name))
        for owner in originals:
            leaf = self.retain(owner.name, original=True)
            if leaf.identity != owner.identity:
                raise ValueError(f'Original output identity changed: {owner.name}')
            self.originals[owner.name] = leaf
        self.acquire_pin()

    def acquire_pin(self) -> None:
        self.pin = self.root.api.open(self.root.path, DIRECTORY_ACCESS)
        info = self.root.api.information(self.pin)
        info.require(directory=True)
        if info.identity != self.root.identity:
            raise ValueError('Publication directory identity changed during pin handoff')
        self.root.verify()
        # Verify all retained objects after handoff, before any content write.
        for leaf in self.leaves:
            self.verify_leaf(leaf)
            current = self.root.info(leaf.name)
            if current is None or FileIdentity.from_stat(current) != leaf.identity:
                raise ValueError(f'Publication leaf changed during pin handoff: {leaf.name}')
        self.ready = True

    def journal(self, data: bytes) -> None:
        if not self.ready:
            # The coordinator also journals failed preparation/rollback. No
            # content is authorized before handoff; only empty owned leaves
            # can exist, and cleanup removes them using their retained handles.
            return
        if self.record is None:
            raise ValueError('No retained journal for the verified directory pin')
        self.verify_leaf(self.record)
        payload = json.loads(data)
        payload['windows_layout'] = 'flat-same-directory'
        payload['retained_leaves'] = [dict(name=leaf.name, identity=dict(device=leaf.identity.device,
                                            inode=leaf.identity.inode), original=leaf.original)
                                      for leaf in self.leaves if not leaf.handle.closed]
        self.root.api.rewrite(self.record.handle, canonical(payload))
        self.committed = payload['phase'] == 'committed'

    def stage(self, source: Path, name: str) -> FileIdentity:
        if not self.ready:
            raise ValueError('Staging content requires the verified directory pin')
        leaf = self.stages[name]
        self.verify_leaf(leaf)
        api = self.root.api
        with api.open(source, READ) as source_handle:
            api.information(source_handle).require(directory=False)
            while data := api.read(source_handle):
                api.write(leaf.handle, data)
        api.flush(leaf.handle)
        return leaf.identity

    def move(self, leaf: Leaf, name: str, *, public: bool) -> None:
        if not self.ready:
            raise ValueError('Public mutation requires the verified directory pin')
        self.root.verify()
        self.verify_leaf(leaf)
        # NTFS no-replace is atomic. Python casefold is deliberately stricter
        # than NTFS upcase and is checked both sides by the shared coordinator.
        self.root.api.rename(leaf.handle, name)
        leaf.name, leaf.public = name, public

    def backup(self, name: str, private: str, identity: FileIdentity) -> None:
        leaf = self.originals[name]
        if leaf.identity != identity:
            raise ValueError(f'Original output identity changed: {name}')
        self.move(leaf, self.private(private), public=False)

    def publish(self, private: str, name: str, *, replace: bool) -> None:
        # Even replacement is no-replace: originals have already moved to backup.
        self.move(self.stages[private], name, public=True)

    def remove(self, leaf: Leaf) -> None:
        if not leaf.handle.closed:
            self.verify_leaf(leaf)
            self.root.api.dispose(leaf.handle)
            leaf.handle.close()

    def remove_public(self, owner: Ownership) -> None:
        for leaf in self.stages.values():
            if leaf.public and leaf.name == owner.name and leaf.identity == owner.identity:
                self.remove(leaf)
                return
        raise ValueError(f'No retained ownership for output: {owner.name}')

    def restore(self, private: str, original: Ownership, published: FileIdentity | None) -> None:
        leaf = self.originals[original.name]
        self.verify_leaf(leaf)
        if leaf.identity != original.identity or leaf.name != self.private(private):
            raise ValueError(f'Recovery identity changed: {private}')
        info = self.root.info(original.name)
        if info is not None:
            # A successful rename may precede both journal.published and the
            # cached Leaf.name/public update. Retained staged identity is the
            # authority to remove this occupant, never its name or journal alone.
            current = FileIdentity.from_stat(info)
            staged = next((item for item in self.stages.values()
                           if item.identity == current and not item.handle.closed), None)
            if staged is None:
                raise ValueError(f'Cannot restore over unowned output: {original.name}')
            self.remove(staged)
        self.move(leaf, original.name, public=True)  # Atomic refusal if a competitor won the gap.

    def discard_staged(self) -> None:
        pass  # Rename consumed the private name; retain its handle for rollback.

    def cleanup(self, *, keep_backups: bool = False) -> None:
        # Uninspected reservations have no identity proof, but their CREATE_NEW
        # handles still own those exact empty objects. Never reopen or delete by
        # path, nor extend this fallback to an original or a validated Leaf.
        # Reconcile these first so failure cannot erase the recovery record.
        for name, handle in tuple(self.reservations.items()):
            if keep_backups and name == self.private('journal.json'):
                continue
            try:
                if not handle.closed:
                    self.root.api.dispose(handle)
                    handle.close()
            except (OSError, ValueError) as exc:
                raise OSError('Cannot dispose uninspected CREATE_NEW reservation by retained handle: '
                              f'{self.root.path / name}') from exc
            del self.reservations[name]
        if self.ready and not keep_backups and not self.committed:
            # An interruption can land after the kernel rename but before the
            # coordinator records its success. Never infer restored originals
            # from that incomplete journal or from the cached Leaf.name alone.
            for name, leaf in self.originals.items():
                info = self.root.info(name)
                if info is None or FileIdentity.from_stat(info) != leaf.identity:
                    raise OSError(f'Original requires retained-handle recovery: {name}')
        # Never dispose a public original or committed new file. Uncommitted
        # stages must be removed by retained handle even if already public: the
        # coordinator may not have recorded their rename. Keep the journal last
        # so a failed reconciliation always leaves useful recovery evidence.
        for leaf in [item for item in self.leaves if item is not self.record] + ([self.record] if self.record else []):
            if leaf.public and (leaf.original or self.committed):
                continue
            if keep_backups and (leaf.original or leaf is self.record):
                continue
            self.remove(leaf)


@contextmanager
def open_backend(output: Path | str) -> Iterator[WindowsBackend]:
    selected = Path(os.path.abspath(output))
    long_path(selected)  # Reject UNC/device/relative-drive paths before creation.
    api = WindowsAPI()
    ancestors = []
    with ExitStack() as stack:
        for directory in (*reversed(selected.parents), selected):
            if directory != Path(selected.anchor):
                try:
                    os.mkdir(long_path(directory))
                except FileExistsError:
                    pass
            # Existing ancestors are pinned before opening children. The final
            # directory gets a metadata handle until originals/staging are open.
            access = ATTRIBUTES if directory == selected else DIRECTORY_ACCESS
            handle = stack.enter_context(api.open(directory, access))
            info = api.information(handle)
            info.require(directory=True)
            api.require_local_ntfs(handle, directory)
            ancestors.append((directory, handle, info.identity))
        root = WindowsBackend(selected, api, tuple(ancestors))
        root.verify()
        yield root
