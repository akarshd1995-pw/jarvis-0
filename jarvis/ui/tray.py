"""System tray icon (optional: needs `pip install pystray pillow`). Without it, Jarvis runs fine
and you use the hotkey and /commands instead."""
from __future__ import annotations

import logging
from typing import Callable

log = logging.getLogger(__name__)


def _icon_image():
    from PIL import Image, ImageDraw
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.ellipse((4, 4, 60, 60), fill=(18, 22, 30, 255), outline=(80, 200, 255, 255), width=4)
    d.ellipse((22, 22, 42, 42), fill=(80, 200, 255, 255))
    return img


def start_tray(actions: dict[str, Callable[[], None]]):
    """actions: open, hud (or None), settings, update, quit. Callbacks run on the tray thread; they should only
    post to the UI queue. Returns the icon or None if pystray isn't installed."""
    try:
        import pystray
    except ImportError:
        log.info("pystray not installed; no tray icon (pip install pystray pillow)")
        return None
    menu = pystray.Menu(
        pystray.MenuItem("Open command bar", lambda: actions["open"](), default=True),
        pystray.MenuItem("Show / hide HUD", lambda: actions["hud"](), visible=bool(actions.get("hud"))),
        pystray.MenuItem("Settings", lambda: actions["settings"]()),
        pystray.MenuItem("Check for updates", lambda: actions["update"]()),
        pystray.Menu.SEPARATOR,
        pystray.MenuItem("Quit Jarvis", lambda: actions["quit"]()),
    )
    icon = pystray.Icon("jarvis-0", _icon_image(), "Jarvis-0", menu)
    icon.run_detached()
    return icon
