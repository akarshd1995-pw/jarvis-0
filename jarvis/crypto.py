"""Encryption at rest for the activity log, using Windows DPAPI (ctypes only, no extra packages).

DPAPI ties the key to your Windows user account: only you, logged into this PC, can decrypt.
On other platforms (tests, Linux) data is stored as-is with a 'plain:' marker.
"""
from __future__ import annotations

import base64
import sys

IS_WINDOWS = sys.platform == "win32"


def _dpapi(data: bytes, protect: bool) -> bytes:
    import ctypes
    from ctypes import wintypes

    class BLOB(ctypes.Structure):
        _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_char))]

    crypt32, kernel32 = ctypes.windll.crypt32, ctypes.windll.kernel32  # type: ignore[attr-defined]
    buf = ctypes.create_string_buffer(data, len(data))
    blob_in = BLOB(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)))
    blob_out = BLOB()
    CRYPTPROTECT_UI_FORBIDDEN = 0x1
    if protect:
        ok = crypt32.CryptProtectData(ctypes.byref(blob_in), "jarvis-0", None, None, None,
                                      CRYPTPROTECT_UI_FORBIDDEN, ctypes.byref(blob_out))
    else:
        ok = crypt32.CryptUnprotectData(ctypes.byref(blob_in), None, None, None, None,
                                        CRYPTPROTECT_UI_FORBIDDEN, ctypes.byref(blob_out))
    if not ok:
        raise OSError("DPAPI call failed")
    try:
        return ctypes.string_at(blob_out.pbData, blob_out.cbData)
    finally:
        kernel32.LocalFree(blob_out.pbData)


def protect(text: str, enabled: bool = True) -> str:
    raw = text.encode("utf-8")
    if enabled and IS_WINDOWS:
        return "dpapi:" + base64.b64encode(_dpapi(raw, True)).decode("ascii")
    return "plain:" + text


def unprotect(stored: str) -> str:
    if stored.startswith("dpapi:"):
        return _dpapi(base64.b64decode(stored[6:]), False).decode("utf-8")
    if stored.startswith("plain:"):
        return stored[6:]
    return stored  # rows written before encryption existed
