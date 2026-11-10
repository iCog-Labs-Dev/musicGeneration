"""Small, injectable Windows ABI boundary; importing it does not load a DLL.

Documented contracts (not the Win32 relative-RootDirectory rename wrapper):
https://learn.microsoft.com/windows-hardware/drivers/ddi/ntifs/nf-ntifs-ntsetinformationfile
https://learn.microsoft.com/windows-hardware/drivers/ddi/ntifs/ns-ntifs-_file_rename_information
https://learn.microsoft.com/windows/win32/api/fileapi/nf-fileapi-createfilew
https://learn.microsoft.com/windows/win32/api/fileapi/nf-fileapi-getfileinformationbyhandle
https://learn.microsoft.com/windows/win32/api/fileapi/nf-fileapi-getvolumeinformationbyhandlew
https://learn.microsoft.com/windows/win32/api/fileapi/nf-fileapi-flushfilebuffers

Only synchronous disk handles are used. No undocumented flags, NT path parsing,
non-null rename root, writable directory sharing, or path mutation fallback.
"""
from __future__ import annotations

import ctypes as c
from dataclasses import dataclass
import errno
import os
from pathlib import PureWindowsPath
import re
import stat
from typing import Any, Callable

from aimusic.audio.publication_records import FileIdentity, validate_names

DWORD = c.c_uint32
BOOL = c.c_int32
HANDLE = c.c_void_p
WCHAR = c.c_uint16  # c_wchar is four bytes on POSIX; ABI tests run there too.
READ = 0x80000000
WRITE = 0x40000000
DELETE = 0x10000
ATTRIBUTES = 0x80
DIRECTORY_ACCESS = 0x81  # FILE_LIST_DIRECTORY | FILE_READ_ATTRIBUTES
SHARE_READ = 1
SHARE_ALL = 7  # Leaf readers only; NEVER directory pins.
OPEN_EXISTING = 3
CREATE_NEW = 1
REPARSE = 0x400
DIRECTORY = 0x10
OPEN_FLAGS = 0x02200000  # BACKUP_SEMANTICS | OPEN_REPARSE_POINT


class FileInformation(c.Structure):
    _fields_ = [('attributes', DWORD), ('times', DWORD * 6), ('volume', DWORD),
                ('size_high', DWORD), ('size_low', DWORD), ('links', DWORD),
                ('index_high', DWORD), ('index_low', DWORD)]


class RenameInfo(c.Structure):
    _fields_ = [('ReplaceIfExists', c.c_ubyte), ('RootDirectory', HANDLE),
                ('FileNameLength', DWORD), ('FileName', WCHAR * 1)]


class IOStatusBlock(c.Structure):
    # The first member is a union of NTSTATUS and pointer, hence pointer size/alignment.
    _fields_ = [('status_or_pointer', HANDLE), ('information', c.c_size_t)]


@dataclass(frozen=True)
class Information:
    identity: FileIdentity
    attributes: int
    size: int
    links: int

    def require(self, *, directory: bool) -> None:
        if self.attributes & REPARSE or bool(self.attributes & DIRECTORY) != directory:
            raise ValueError('Publication object is a reparse point or has the wrong file type')

    def stat(self) -> os.stat_result:
        # Shared policy needs only mode, device, inode and size. Every reparse
        # type is deliberately classified as unsupported, not a POSIX leaf link.
        mode = 0 if self.attributes & REPARSE else (stat.S_IFDIR if self.attributes & DIRECTORY else stat.S_IFREG)
        fields = {'st_file_attributes': self.attributes} if hasattr(os.stat_result, 'st_file_attributes') else {}
        return os.stat_result((mode, self.identity.inode, self.identity.device, self.links, 0, 0, self.size, 0, 0, 0), fields)


@dataclass
class Handle:
    api: WindowsAPI
    _value: int
    closed: bool = False

    @property
    def value(self) -> int:
        if self.closed:
            raise ValueError('Windows handle is already closed')
        return self._value

    def close(self) -> None:
        if not self.closed:
            self.closed = True  # Never double-close, even when CloseHandle fails.
            self.api.check(self.api.kernel.CloseHandle(self._value), 'CloseHandle')

    def __enter__(self) -> Handle:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def long_path(path: os.PathLike[str] | str) -> str:
    text = os.fspath(path)
    if text.startswith('\\\\?\\'):
        text = text[4:]
    p = PureWindowsPath(text)
    if not re.fullmatch('[a-zA-Z]:', p.drive) or not p.root or '..' in p.parts:
        raise NotImplementedError('Publication requires an absolute local drive path, not a network/device path')
    return '\\\\?\\' + str(p)


def leaf_name(name: str) -> None:
    # Internal invocation-prefixed names obey the same grammar as public names.
    validate_names(['x' + name])
    if not name or name in ('.', '..'):
        raise ValueError('Rename requires a simple leaf name')
    validate_names([name] if not name.startswith(('.stems-', '.groove-', '.publication-')) else ['x' + name])


class WindowsAPI:
    def __init__(self, kernel: Any = None, native: Any = None, *, last_error: Callable[[], int] | None = None):
        if kernel is None:
            if os.name != 'nt':
                raise NotImplementedError('Native Windows publication requires Windows')
            loader = getattr(c, 'WinDLL')
            kernel = loader('kernel32', use_last_error=True)
            native = loader('ntdll', use_last_error=True)
        self.kernel, self.native = kernel, native
        self.last_error = last_error or getattr(c, 'get_last_error')
        # Fixed-width types are essential: wintypes.DWORD/c_ulong is host-sized on POSIX.
        signatures = {
            'CreateFileW': ([c.c_wchar_p, DWORD, DWORD, HANDLE, DWORD, DWORD, HANDLE], HANDLE),
            'CloseHandle': ([HANDLE], BOOL),
            'GetFileInformationByHandle': ([HANDLE, c.POINTER(FileInformation)], BOOL),
            'GetVolumeInformationByHandleW': ([HANDLE, c.c_wchar_p, DWORD, c.POINTER(DWORD),
                                              c.POINTER(DWORD), c.POINTER(DWORD), c.c_wchar_p, DWORD], BOOL),
            'GetDriveTypeW': ([c.c_wchar_p], DWORD),
            'ReadFile': ([HANDLE, HANDLE, DWORD, c.POINTER(DWORD), HANDLE], BOOL),
            'WriteFile': ([HANDLE, HANDLE, DWORD, c.POINTER(DWORD), HANDLE], BOOL),
            'FlushFileBuffers': ([HANDLE], BOOL),
            'SetFilePointerEx': ([HANDLE, c.c_int64, c.POINTER(c.c_int64), DWORD], BOOL),
            'SetEndOfFile': ([HANDLE], BOOL),
            'SetFileInformationByHandle': ([HANDLE, c.c_int32, HANDLE, DWORD], BOOL),
        }
        try:
            for name, (arguments, result) in signatures.items():
                function = getattr(kernel, name)
                function.argtypes, function.restype = arguments, result
            native.NtSetInformationFile.argtypes = [HANDLE, c.POINTER(IOStatusBlock), HANDLE, DWORD, c.c_int32]
            native.NtSetInformationFile.restype = c.c_int32
            native.RtlNtStatusToDosError.argtypes = [c.c_int32]
            native.RtlNtStatusToDosError.restype = DWORD
        except AttributeError as exc:
            raise NotImplementedError('Required Windows publication API is unavailable') from exc

    def error(self, operation: str, code: int | None = None) -> OSError:
        code = self.last_error() if code is None else code
        number = {2: errno.ENOENT, 3: errno.ENOENT, 5: errno.EACCES, 32: errno.EACCES,
                  80: errno.EEXIST, 183: errno.EEXIST, 1: errno.ENOTSUP, 50: errno.ENOTSUP,
                  120: errno.ENOSYS}.get(code, errno.EIO)
        error = OSError(number, f'{operation} failed (WinError {code})')
        error.winerror = code  # type: ignore[attr-defined]
        return error

    def check(self, result: Any, operation: str) -> None:
        if not result:
            raise self.error(operation)

    def open(self, path: os.PathLike[str] | str, access: int, *, share: int = SHARE_READ,
             create: bool = False) -> Handle:
        value = self.kernel.CreateFileW(long_path(path), access, share, None,
                                        CREATE_NEW if create else OPEN_EXISTING, OPEN_FLAGS, None)
        if value in (None, 0, c.c_void_p(-1).value):
            raise self.error(f'CreateFileW({path})')
        return Handle(self, value)

    def information(self, handle: Handle) -> Information:
        info = FileInformation()
        self.check(self.kernel.GetFileInformationByHandle(handle.value, c.byref(info)), 'GetFileInformationByHandle')
        return Information(FileIdentity(info.volume, (info.index_high << 32) | info.index_low),
                           info.attributes, (info.size_high << 32) | info.size_low, info.links)

    def require_local_ntfs(self, handle: Handle, path: os.PathLike[str] | str) -> None:
        anchor = PureWindowsPath(long_path(path)[4:]).anchor
        if self.kernel.GetDriveTypeW(anchor) != 3:  # DRIVE_FIXED; exclude remote, unknown and removable.
            raise NotImplementedError('Publication requires a fixed local NTFS volume')
        filesystem = c.create_unicode_buffer(32)
        serial, maximum, flags = DWORD(), DWORD(), DWORD()
        self.check(self.kernel.GetVolumeInformationByHandleW(handle.value, None, 0, c.byref(serial),
                   c.byref(maximum), c.byref(flags), filesystem, len(filesystem)), 'GetVolumeInformationByHandleW')
        if filesystem.value != 'NTFS' or flags.value & 0x80000:  # FILE_READ_ONLY_VOLUME
            raise NotImplementedError('Publication requires a writable local NTFS filesystem')

    def read(self, handle: Handle, count: int = 1024 * 1024) -> bytes:
        if not 0 < count <= 1024 * 1024:
            raise ValueError('Native reads must be bounded to 1..1048576 bytes')
        buffer = c.create_string_buffer(count)
        size = DWORD()
        self.check(self.kernel.ReadFile(handle.value, buffer, count, c.byref(size), None), 'ReadFile')
        if size.value > count:
            raise OSError('ReadFile returned an invalid byte count')
        return buffer.raw[:size.value]

    def write(self, handle: Handle, data: bytes) -> None:
        offset = 0
        while offset < len(data):
            chunk = data[offset:offset + 1024 * 1024]
            buffer, size = c.create_string_buffer(chunk), DWORD()
            self.check(self.kernel.WriteFile(handle.value, buffer, len(chunk), c.byref(size), None), 'WriteFile')
            if not 0 < size.value <= len(chunk):
                raise OSError('WriteFile made no progress or returned an invalid byte count')
            offset += size.value

    def flush(self, handle: Handle) -> None:
        self.check(self.kernel.FlushFileBuffers(handle.value), 'FlushFileBuffers')

    def rewrite(self, handle: Handle, data: bytes) -> None:
        self.check(self.kernel.SetFilePointerEx(handle.value, 0, None, 0), 'SetFilePointerEx')
        self.write(handle, data)
        self.check(self.kernel.SetEndOfFile(handle.value), 'SetEndOfFile')
        self.flush(handle)

    def rename(self, handle: Handle, name: str) -> None:
        leaf_name(name)
        encoded = name.encode('utf-16-le')
        buffer = c.create_string_buffer(c.sizeof(RenameInfo) + len(encoded) + 2)
        info = RenameInfo.from_buffer(buffer)
        info.ReplaceIfExists, info.RootDirectory, info.FileNameLength = 0, None, len(encoded)
        c.memmove(c.addressof(buffer) + RenameInfo.FileName.offset, encoded, len(encoded))
        status = self.native.NtSetInformationFile(handle.value, c.byref(IOStatusBlock()), buffer, len(buffer), 10)
        if status != 0:
            if status == 0x103:  # STATUS_PENDING is impossible for our synchronous disk handles.
                raise NotImplementedError('Asynchronous rename completion is unsupported')
            code = self.native.RtlNtStatusToDosError(status)
            raise self.error(f'NtSetInformationFile rename {name} (NTSTATUS {status & 0xffffffff:#x})', code)

    def dispose(self, handle: Handle) -> None:
        # FILE_DISPOSITION_INFO contains BOOLEAN, not Win32 BOOL. After success
        # callers only CloseHandle; there is no path-based delete fallback.
        deleting = c.c_ubyte(1)
        self.check(self.kernel.SetFileInformationByHandle(handle.value, 4, c.byref(deleting), c.sizeof(deleting)),
                   'SetFileInformationByHandle(FileDispositionInfo)')
