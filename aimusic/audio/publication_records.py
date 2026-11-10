"""Immutable publication evidence and host-independent leaf-name policy."""
from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
import json
import os
from typing import Any, Iterable


@dataclass(frozen=True)
class FileIdentity:
    device: int
    inode: int

    @classmethod
    def from_stat(cls, info: os.stat_result) -> FileIdentity:
        return cls(info.st_dev, info.st_ino)


@dataclass(frozen=True)
class Ownership:
    name: str
    identity: FileIdentity


class Phase(str, Enum):
    PREPARING = 'preparing'
    PREPARED = 'prepared'
    PUBLISHING = 'publishing'
    PUBLISHED = 'published'
    ROLLING_BACK = 'rolling_back'
    COMMITTED = 'committed'
    RECOVERY = 'recovery'


@dataclass(frozen=True)
class Journal:
    invocation: str
    names: tuple[str, ...]
    replace: bool
    phase: Phase = Phase.PREPARING
    originals: tuple[Ownership, ...] = ()
    staged: tuple[Ownership, ...] = ()
    backups: tuple[Ownership, ...] = ()
    published: tuple[Ownership, ...] = ()
    schema_version: int = 1


def canonical(value: Any) -> bytes:
    """Canonical UTF-8 JSON, with no non-finite numbers or host-specific paths."""
    return (json.dumps(asdict(value) if is_dataclass(value) and not isinstance(value, type) else value,
                       sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                       allow_nan=False) + '\n').encode('ascii')


# Reserved on Windows even when followed by an extension, including superscript
# device digits. Do not defer to host-specific pathlib/isreserved versions.
_RESERVED = {'con', 'prn', 'aux', 'nul', 'conin$', 'conout$'} | {
    prefix + number for prefix in ('com', 'lpt') for number in '123456789¹²³'
}
PRIVATE_PREFIXES = ('.stems-', '.groove-', '.publication-')


def validate_names(names: Iterable[str]) -> tuple[str, ...]:
    result = tuple(names)
    folded: set[str] = set()
    for name in result:
        if (not isinstance(name, str) or not name or name in ('.', '..') or
                any(c in '/\\:<>"|?*' or ord(c) < 32 or 127 <= ord(c) <= 159 or
                    0xD800 <= ord(c) <= 0xDFFF for c in name) or name[-1] in ' .' or
                name.split('.')[0].rstrip(' ').casefold() in _RESERVED or
                name.casefold().startswith(PRIVATE_PREFIXES)):
            raise ValueError(f'Output violates containment or portable filename rules: {name!r}')
        key = name.casefold()
        if key in folded:
            raise ValueError(f'Duplicate or casefold-alias output name: {name!r}')
        folded.add(key)
    return result
