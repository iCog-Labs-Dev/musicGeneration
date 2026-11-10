"""Stdlib-only native token gate, run before build and again inside wheel checks."""
from __future__ import annotations

import ctypes as c
from ctypes import wintypes as w
import json
import os
import sys
from typing import Callable, cast


def token_status() -> dict:
    if os.name != 'nt':
        raise RuntimeError('Native Windows standard-user token required')
    # These APIs exist only on Windows; required lookups preserve fail-closed behavior.
    loader = getattr(c, 'WinDLL')
    get_last_error = cast(Callable[[], int], getattr(c, 'get_last_error'))
    kernel = loader('kernel32', use_last_error=True)
    advapi = loader('advapi32', use_last_error=True)
    kernel.GetCurrentProcess.argtypes, kernel.GetCurrentProcess.restype = [], w.HANDLE
    kernel.CloseHandle.argtypes, kernel.CloseHandle.restype = [w.HANDLE], w.BOOL
    kernel.LocalFree.argtypes, kernel.LocalFree.restype = [w.HLOCAL], w.HLOCAL
    advapi.OpenProcessToken.argtypes = [w.HANDLE, w.DWORD, c.POINTER(w.HANDLE)]
    advapi.OpenProcessToken.restype = w.BOOL
    advapi.GetTokenInformation.argtypes = [w.HANDLE, c.c_int, w.LPVOID, w.DWORD, c.POINTER(w.DWORD)]
    advapi.GetTokenInformation.restype = w.BOOL
    advapi.ConvertStringSidToSidW.argtypes = [w.LPCWSTR, c.POINTER(w.LPVOID)]
    advapi.ConvertStringSidToSidW.restype = w.BOOL
    advapi.CheckTokenMembership.argtypes = [w.HANDLE, w.LPVOID, c.POINTER(w.BOOL)]
    advapi.CheckTokenMembership.restype = w.BOOL
    advapi.GetSidSubAuthorityCount.argtypes, advapi.GetSidSubAuthorityCount.restype = [w.LPVOID], c.POINTER(w.BYTE)
    advapi.GetSidSubAuthority.argtypes, advapi.GetSidSubAuthority.restype = [w.LPVOID, w.DWORD], c.POINTER(w.DWORD)

    def checked(result):
        if not result:
            raise getattr(c, 'WinError')(get_last_error())

    token = w.HANDLE()
    checked(advapi.OpenProcessToken(kernel.GetCurrentProcess(), 0x0008, c.byref(token)))  # TOKEN_QUERY
    try:
        elevated, size = w.DWORD(), w.DWORD()
        checked(advapi.GetTokenInformation(token, 20, c.byref(elevated), c.sizeof(elevated), c.byref(size)))
        # TOKEN_MANDATORY_LABEL starts with SID_AND_ATTRIBUTES (pointer + DWORD).
        if advapi.GetTokenInformation(token, 25, None, 0, c.byref(size)) or get_last_error() != 122:
            raise RuntimeError('Unexpected token integrity size query result')
        buffer = c.create_string_buffer(size.value)
        checked(advapi.GetTokenInformation(token, 25, buffer, size, c.byref(size)))
        sid = c.cast(buffer, c.POINTER(w.LPVOID))[0]
        count = advapi.GetSidSubAuthorityCount(sid)
        checked(count)
        if not count[0]:
            raise RuntimeError('Empty token integrity SID')
        rid = advapi.GetSidSubAuthority(sid, count[0] - 1)
        checked(rid)
        integrity = int(rid[0])
        admin_sid, member = w.LPVOID(), w.BOOL()
        checked(advapi.ConvertStringSidToSidW('S-1-5-32-544', c.byref(admin_sid)))
        try:
            # NULL token checks the effective token and ignores deny-only admin SIDs.
            checked(advapi.CheckTokenMembership(None, admin_sid, c.byref(member)))
        finally:
            kernel.LocalFree(admin_sid)
        return dict(elevated=bool(elevated.value), administrator=bool(member.value), integrity_rid=integrity)
    finally:
        checked(kernel.CloseHandle(token))


def require_standard_user() -> dict:
    if sys.flags.optimize:
        raise RuntimeError('Assertions must be enabled for Windows checks')
    status = token_status()
    if status['elevated'] or status['administrator'] or status['integrity_rid'] > 0x2000:
        raise RuntimeError(f'Actual checks require a standard-user/restricted token: {status}')
    return status


if __name__ == '__main__':
    print(json.dumps(require_standard_user(), sort_keys=True))
