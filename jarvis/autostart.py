"""Start Jarvis automatically when you log in to Windows (per-user Run key, no admin needed)."""
from __future__ import annotations

import sys
from pathlib import Path

RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"
VALUE = "Jarvis-0"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def command() -> str:
    if getattr(sys, "frozen", False):                  # the installed app (JarvisSetup.exe): its own Jarvis.exe
        return f'"{sys.executable}"'
    exe = Path(sys.executable)
    pyw = exe.with_name("pythonw.exe") if exe.name.lower() == "python.exe" else exe
    return f'"{pyw}" "{PROJECT_ROOT / "jarvis_launcher.pyw"}"'


def enable() -> str:
    import winreg
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as k:
        winreg.SetValueEx(k, VALUE, 0, winreg.REG_SZ, command())
    return "Jarvis will start automatically when you log in."


def disable() -> str:
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as k:
            winreg.DeleteValue(k, VALUE)
        return "Autostart removed."
    except FileNotFoundError:
        return "Autostart was not enabled."


def is_enabled() -> bool:
    if sys.platform != "win32":
        return False
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY) as k:
            winreg.QueryValueEx(k, VALUE)
        return True
    except FileNotFoundError:
        return False
