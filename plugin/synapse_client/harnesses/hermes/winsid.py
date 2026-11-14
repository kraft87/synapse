"""This process's Windows user SID, read with ctypes (the client is stdlib only).

On Windows Hermes names the machine's owner ``sid:<sid>``: the string SID of the user of
the gateway process's own token (gateway/session_worker_reservation.py
``owner_worker_subject``; gateway/runtime_bootstrap_windows.py ``_process_sid``, which
reads ``TokenUser`` with ``GetTokenInformation`` and writes it with
``ConvertSidToStringSidW``). The control pipe mints the same text for a peer of the same
user (``_peer_subject``). This module reads the same token the same way, so the text is
the same byte for byte.

``process_sid`` never raises. Off Windows, or on any error, it answers None, and then no
session matches the owner (sessions.py ``owner_subject``).
"""

from __future__ import annotations

import os
import re

#: What ``ConvertSidToStringSidW`` writes: "S-1-", the authority, then each sub-authority.
_SID = re.compile(r"S-1-\d+(-\d+)+")
#: Whether this is Windows, the only place the lookup runs.
WINDOWS = os.name == "nt"
_TOKEN_QUERY = 0x0008
_TOKEN_USER = 1  # TOKEN_INFORMATION_CLASS.TokenUser


def process_sid() -> str | None:
    """The SID of this process's user, as ``S-1-5-21-...``; None off Windows or when it
    cannot be read."""
    if not WINDOWS:
        return None
    try:
        sid = _lookup()
    except Exception:  # any failure: no owner, so the session stays inert
        return None
    return sid if isinstance(sid, str) and _SID.fullmatch(sid) else None


def _lookup() -> str:
    """``TokenUser`` of this process's token, as text. RAISES on any failure. Each call
    loads its own WinDLL instances, so the prototypes set here never touch the host's."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)  # type: ignore[attr-defined]
    current = kernel32.GetCurrentProcess
    current.restype, current.argtypes = wintypes.HANDLE, []
    open_token = advapi32.OpenProcessToken
    open_token.restype = wintypes.BOOL
    open_token.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE)]
    token_info = advapi32.GetTokenInformation
    token_info.restype = wintypes.BOOL
    token_info.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    convert = advapi32.ConvertSidToStringSidW
    convert.restype = wintypes.BOOL
    convert.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.LPWSTR)]
    close = kernel32.CloseHandle
    close.restype, close.argtypes = wintypes.BOOL, [wintypes.HANDLE]
    local_free = kernel32.LocalFree
    local_free.restype, local_free.argtypes = ctypes.c_void_p, [ctypes.c_void_p]

    def failed() -> OSError:
        return ctypes.WinError(ctypes.get_last_error())  # type: ignore[attr-defined,no-any-return]

    token = wintypes.HANDLE()
    if not open_token(current(), _TOKEN_QUERY, ctypes.byref(token)):
        raise failed()
    try:
        size = wintypes.DWORD()
        token_info(token, _TOKEN_USER, None, 0, ctypes.byref(size))  # asks for the size
        if not size.value:
            raise failed()
        buffer = ctypes.create_string_buffer(size.value)
        if not token_info(token, _TOKEN_USER, buffer, size, ctypes.byref(size)):
            raise failed()
        # TOKEN_USER starts with SID_AND_ATTRIBUTES, whose first field points at the SID.
        sid = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_void_p))[0]
        if not sid:
            raise OSError("no user SID in the process token")
        text = wintypes.LPWSTR()
        if not convert(sid, ctypes.byref(text)):
            raise failed()
        try:
            value = text.value
        finally:
            local_free(ctypes.cast(text, ctypes.c_void_p))
        if not value:
            raise OSError("an empty SID")
        return value
    finally:
        close(token)
