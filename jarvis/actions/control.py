"""Phase 5: full system control on Windows. Keystrokes and window calls via ctypes; brightness,
Wi-Fi/Bluetooth radios and sleep via built-in PowerShell; shutdown via shutdown.exe. No extra packages."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from ..collectors.focus import LAST_EXTERNAL
from ..dsl.grammar import Duration
from .base import ActionResult, Ctx

IS_WINDOWS = sys.platform == "win32"
NO_WINDOW = 0x08000000 if IS_WINDOWS else 0

VK = {"win": 0x5B, "ctrl": 0x11, "alt": 0x12, "shift": 0x10, "left": 0x25, "up": 0x26, "right": 0x27,
      "down": 0x28, "d": 0x44, "r": 0x52, "s": 0x53, "f4": 0x73, "prtsc": 0x2C,
      "play_pause": 0xB3, "next": 0xB0, "previous": 0xB1, "stop": 0xB2, "vol_down": 0xAE, "vol_up": 0xAF}
EXTENDED = {0x25, 0x26, 0x27, 0x28, 0x5B, 0x2C}


class NotOnWindows(RuntimeError):
    pass


class NoWindowFound(LookupError):
    pass


def _user32():
    if not IS_WINDOWS:
        raise NotOnWindows("this only works on Windows")
    import ctypes
    return ctypes.windll.user32  # type: ignore[attr-defined]


def chord(*keys: str) -> None:
    """Press keys together, release in reverse order (e.g. chord('win', 'left'))."""
    u = _user32()
    codes = [VK[k] for k in keys]
    for c in codes:
        u.keybd_event(c, 0, 0x1 if c in EXTENDED else 0, 0)
        time.sleep(0.02)
    for c in reversed(codes):
        u.keybd_event(c, 0, (0x1 if c in EXTENDED else 0) | 0x2, 0)
    time.sleep(0.05)


def tap(key: str, times: int = 1) -> None:
    for _ in range(times):
        chord(key)


def remember_foreground() -> None:
    """Called when the hotkey is pressed: the window the user was in, before the command bar appears."""
    if not IS_WINDOWS:
        return
    try:
        import ctypes
        from ctypes import wintypes
        u = _user32()
        hwnd = u.GetForegroundWindow()
        pid = wintypes.DWORD()
        u.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if hwnd and pid.value != os.getpid():
            LAST_EXTERNAL["hwnd"] = int(hwnd)
    except Exception:
        pass


def _process_name(hwnd: int) -> str:
    import ctypes
    from ctypes import wintypes
    u, k = _user32(), ctypes.windll.kernel32  # type: ignore[attr-defined]
    pid = wintypes.DWORD()
    u.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    h = k.OpenProcess(0x1000, False, pid.value)
    name = ""
    if h:
        size, buf = wintypes.DWORD(1024), ctypes.create_unicode_buffer(1024)
        if k.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size)):
            name = Path(buf.value).stem.lower()
        k.CloseHandle(h)
    return name


def find_window(app: str, aliases: dict[str, str] | None = None) -> int:
    """Visible top-level window whose process name or title matches the app (e.g. 'chrome', 'VS Code')."""
    import ctypes
    from ctypes import wintypes
    u = _user32()
    target = str(app).lower()
    alias_exe = Path(str((aliases or {}).get(target, ""))).stem.lower()
    wanted = {target, target.replace(" ", ""), alias_exe} - {""}
    found: list[int] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
    def cb(hwnd, _):
        if not u.IsWindowVisible(hwnd) or u.GetWindowTextLengthW(hwnd) == 0:
            return True
        n = u.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(n + 1)
        u.GetWindowTextW(hwnd, buf, n + 1)
        title, proc = buf.value.lower(), _process_name(hwnd)
        if any(w in proc or w in title for w in wanted):
            found.append(int(hwnd))
            return False
        return True

    u.EnumWindows(cb, 0)
    if not found:
        raise NoWindowFound(f"no open window for {app}")
    return found[0]


def _target(args: dict, ctx: Ctx) -> tuple[int, str]:
    if "app" in args:
        return find_window(str(args["app"]), ctx.config.apps), str(args["app"])
    hwnd = LAST_EXTERNAL.get("hwnd")
    if not hwnd:
        raise NoWindowFound("no recent window to act on; name an app, e.g. 'snap chrome to the left'")
    return hwnd, ""


def _focus(hwnd: int) -> None:
    u = _user32()
    u.ShowWindow(hwnd, 9)          # SW_RESTORE
    u.SetForegroundWindow(hwnd)
    time.sleep(0.15)


def window(args: dict, ctx: Ctx, dry_run: bool) -> ActionResult:
    action = str(args["action"])
    app = str(args["app"]) if "app" in args else ""
    data = {"action": action, "app": app}
    if dry_run:
        return ActionResult(f"Will {action} {app or 'the window'}", kind="window", data={**data, "dry": True})
    u = _user32()
    if action == "minimize_all":
        chord("win", "d")
        return ActionResult("Showing the desktop", kind="window", data=data)
    hwnd, app = _target(args, ctx)
    if action in ("snap_left", "snap_right"):
        _focus(hwnd)
        chord("win", "left" if action == "snap_left" else "right")
    elif action == "maximize":
        u.ShowWindow(hwnd, 3)
    elif action == "minimize":
        u.ShowWindow(hwnd, 6)
    elif action == "restore":
        _focus(hwnd)
    elif action == "close":
        u.PostMessageW(hwnd, 0x0010, 0, 0)    # WM_CLOSE: the app can still ask to save
    return ActionResult(f"{action} {app or 'window'}", kind="window", data=data)


def desktop(args: dict, ctx: Ctx, dry_run: bool) -> ActionResult:
    action = str(args["action"])
    if not dry_run:
        keys = {"new": ("win", "ctrl", "d"), "next": ("win", "ctrl", "right"), "previous": ("win", "ctrl", "left"),
                "close": ("win", "ctrl", "f4")}[action]
        chord(*keys)
    return ActionResult(f"desktop {action}", kind="desktop", data={"action": action})


def media(args: dict, ctx: Ctx, dry_run: bool) -> ActionResult:
    action = str(args["action"])
    if not dry_run:
        tap(action)
    return ActionResult(f"media {action}", kind="media", data={"action": action})


def _powershell(script: str, timeout: float = 20) -> str:
    if not IS_WINDOWS:
        raise NotOnWindows("this only works on Windows")
    r = subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", script],
                       capture_output=True, text=True, timeout=timeout, creationflags=NO_WINDOW)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip().splitlines()[-1] if (r.stderr or r.stdout).strip()
                           else f"PowerShell exited with {r.returncode}")
    return r.stdout.strip()


def get_brightness() -> int:
    out = _powershell("(Get-CimInstance -Namespace root/WMI -ClassName WmiMonitorBrightness).CurrentBrightness")
    if not out.strip():
        raise RuntimeError("brightness control isn't available for this display (external monitors need their own buttons)")
    return int(out.split()[0])


def set_brightness_level(level: int) -> None:
    _powershell("(Get-CimInstance -Namespace root/WMI -ClassName WmiMonitorBrightnessMethods)"
                f" | Invoke-CimMethod -MethodName WmiSetBrightness -Arguments @{{Timeout=1; Brightness={level}}}")


def brightness(args: dict, ctx: Ctx, dry_run: bool) -> ActionResult:
    if "level" in args:
        level = int(args["level"])
        if not 0 <= level <= 100:
            raise ValueError("brightness must be between 0 and 100")
        if not dry_run:
            set_brightness_level(level)
        return ActionResult(f"brightness {level}%", kind="brightness", data={"level": level, "action": "set"})
    step = 10 if str(args["action"]) == "up" else -10
    if dry_run:
        return ActionResult(f"brightness {args['action']}", kind="brightness", data={"action": str(args["action"])})
    level = max(0, min(100, get_brightness() + step))
    set_brightness_level(level)
    return ActionResult(f"brightness {level}%", kind="brightness", data={"level": level, "action": str(args["action"])})


def set_volume(args: dict, ctx: Ctx, dry_run: bool) -> ActionResult:
    level = int(args["level"])
    if not 0 <= level <= 100:
        raise ValueError("volume must be between 0 and 100")
    if not dry_run:
        tap("vol_down", 50)                 # each key press is 2%: go to 0, then up to the level
        tap("vol_up", round(level / 2))
    return ActionResult(f"volume {level}%", kind="volume_set", data={"level": level})


_RADIO_PS = r"""
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$asTaskGeneric = ([System.WindowsRuntimeSystemExtensions].GetMethods() | ? { $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
Function Await($WinRtTask, $ResultType) { $t = $asTaskGeneric.MakeGenericMethod($ResultType).Invoke($null, @($WinRtTask)); $t.Wait(-1) | Out-Null; $t.Result }
[Windows.Devices.Radios.Radio,Windows.System.Devices,ContentType=WindowsRuntime] | Out-Null
[Windows.Devices.Radios.RadioAccessStatus,Windows.System.Devices,ContentType=WindowsRuntime] | Out-Null
[Windows.Devices.Radios.RadioState,Windows.System.Devices,ContentType=WindowsRuntime] | Out-Null
Await ([Windows.Devices.Radios.Radio]::RequestAccessAsync()) ([Windows.Devices.Radios.RadioAccessStatus]) | Out-Null
$radios = Await ([Windows.Devices.Radios.Radio]::GetRadiosAsync()) ([System.Collections.Generic.IReadOnlyList[Windows.Devices.Radios.Radio]])
$r = $radios | ? { $_.Kind -eq '__KIND__' } | Select-Object -First 1
if (-not $r) { throw 'No __KIND__ radio found on this PC' }
$res = Await ($r.SetStateAsync('__STATE__')) ([Windows.Devices.Radios.RadioAccessStatus])
Write-Output $res
"""


def radio(args: dict, ctx: Ctx, dry_run: bool) -> ActionResult:
    kind, state = str(args["kind"]), str(args["state"])
    data = {"kind": kind, "state": state}
    if not dry_run:
        out = _powershell(_RADIO_PS.replace("__KIND__", "WiFi" if kind == "wifi" else "Bluetooth")
                          .replace("__STATE__", "On" if state == "on" else "Off"))
        if out and "Allowed" not in out:
            raise PermissionError(f"Windows didn't allow changing {kind} ({out})")
    return ActionResult(f"{kind} {state}", kind="radio", data={**data, "dry": dry_run})


def screenshot(args: dict, ctx: Ctx, dry_run: bool) -> ActionResult:
    mode = str(args.get("mode", "full"))
    if not dry_run:
        {"full": lambda: chord("win", "prtsc"), "window": lambda: chord("alt", "prtsc"),
         "region": lambda: chord("win", "shift", "s")}[mode]()
    return ActionResult(f"screenshot {mode}", kind="screenshot", data={"mode": mode})


def screen_record(args: dict, ctx: Ctx, dry_run: bool) -> ActionResult:
    action = str(args["action"])
    if not dry_run:
        chord("win", "alt", "r")            # Xbox Game Bar: start/stop recording the active app
    return ActionResult(f"recording {action}", kind="record", data={"action": action})


def power(args: dict, ctx: Ctx, dry_run: bool) -> ActionResult:
    action = str(args["action"])
    delay = args["after"].seconds() if isinstance(args.get("after"), Duration) else 30
    data = {"action": action, "seconds": delay}
    if dry_run:
        return ActionResult(f"Will {action}", kind="power", data={**data, "dry": True})
    if not IS_WINDOWS:
        raise NotOnWindows("this only works on Windows")
    if action == "shutdown":
        subprocess.run(["shutdown", "/s", "/t", str(delay)], creationflags=NO_WINDOW, check=True)
    elif action == "restart":
        subprocess.run(["shutdown", "/r", "/t", str(delay)], creationflags=NO_WINDOW, check=True)
    elif action == "sign_out":
        subprocess.run(["shutdown", "/l"], creationflags=NO_WINDOW, check=True)
    elif action == "cancel":
        subprocess.run(["shutdown", "/a"], creationflags=NO_WINDOW)
    elif action == "sleep":
        _powershell("Add-Type -AssemblyName System.Windows.Forms; "
                    "[System.Windows.Forms.Application]::SetSuspendState([System.Windows.Forms.PowerState]::Suspend, $false, $false)")
    return ActionResult(action, kind="power", data=data)
