"""Explicit Windows link probes; only known capability errors may be unavailable.

Ordinary NTFS hard links are required. Junction/symlink omissions never waive the
native pin, identity, collision, rollback, or mocked reparse invariant tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import cast


@dataclass(frozen=True)
class Capability:
    available: bool
    reason: str


def install_junction(link: Path, target: Path) -> None:
    """Convert an existing empty directory, using the native test's FSCTL probe."""
    import ctypes as c
    import struct
    from aimusic.audio.windows_native import DWORD, HANDLE, BOOL, WRITE, WindowsAPI
    api = WindowsAPI()
    control = api.kernel.DeviceIoControl
    control.argtypes = [HANDLE, DWORD, HANDLE, DWORD, HANDLE, DWORD, c.POINTER(DWORD), HANDLE]
    control.restype = BOOL
    substitute = ('\\??\\' + str(target)).encode('utf-16-le')
    printed = str(target).encode('utf-16-le')
    paths = substitute + b'\x00\x00' + printed + b'\x00\x00'
    payload = struct.pack('<IHHHHHH', 0xa0000003, len(paths) + 8, 0,
                          0, len(substitute), len(substitute) + 2, len(printed)) + paths
    buffer, returned = c.create_string_buffer(payload), DWORD()
    with api.open(link, WRITE) as handle:
        result = control(handle.value, 0x900a4, buffer, len(payload), None, 0, c.byref(returned), None)
        if not result:
            raise api.error('FSCTL_SET_REPARSE_POINT', getattr(c, 'get_last_error')())


def exercise(kind: str, parent: Path) -> None:
    with tempfile.TemporaryDirectory(prefix='capability-', dir=parent) as tmp:
        root = Path(tmp)
        target, link = root / 'target', root / 'link'
        if kind == 'hardlink':
            target.write_bytes(b'probe')
            link.hardlink_to(target)
            assert link.samefile(target), 'hardlink probe did not alias target'
            link.unlink()
        else:
            target.mkdir()
            if kind == 'junction':
                link.mkdir()
                install_junction(link, target)
            elif kind == 'symlink':
                link.symlink_to(target, target_is_directory=True)
            else:
                raise ValueError(f'Unknown link capability: {kind}')
            try:
                # No default: missing Windows metadata is a failure, never an unavailable capability.
                attributes = cast(int, getattr(link.lstat(), 'st_file_attributes'))
                assert attributes & 0x400, 'probe did not create a reparse point'
                assert link.samefile(target), 'probe did not reach target'
            finally:
                link.rmdir()  # Remove just the link, never its target.


def probe(kind: str, parent: Path) -> Capability:
    try:
        exercise(kind, parent)
    except OSError as exc:
        code = getattr(exc, 'winerror', None)
        if code not in (5, 50, 1314):
            raise  # Unexpected setup/implementation failure is not a skip.
        return Capability(False, f'WinError {code}')
    return Capability(True, 'native creation and identity verified')
