"""Windows UI tests (phase 42): the real command bar, HUD, settings, review window and toasts.

They need a screen, so they run on your PC (and in every update check) and are skipped where there's no
display. JARVIS_UI_TESTS=0 skips them; JARVIS_FAKE_TK=1 runs them against a stand-in Tk that only checks
the code runs (no text to read back).
"""
from __future__ import annotations

import os
import sys
import threading
import time

import pytest

FAKE = os.environ.get("JARVIS_FAKE_TK") == "1"


def _can_show_windows() -> bool:
    if os.environ.get("JARVIS_UI_TESTS") == "0":
        return False
    if FAKE:
        return True
    try:
        import tkinter as tk
    except ImportError:
        return False
    if sys.platform != "win32" and not os.environ.get("DISPLAY"):
        return False
    try:
        r = tk.Tk()
        r.destroy()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _can_show_windows(), reason="needs a screen (runs on Windows)")


def texts(widget) -> str:
    """All label/text on screen under a widget, for checking what you'd see."""
    out = []
    try:
        if widget.winfo_class() == "Text":          # replies are in a selectable Text box, not a label
            out.append(widget.get("1.0", "end"))
        else:
            out.append(str(widget.cget("text")))
    except Exception:
        pass
    try:
        for w in widget.winfo_children():
            out.append(texts(w))
    except Exception:
        pass
    return " ".join(x for x in out if x)


def pump(root, until=lambda: True, timeout: float = 5.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        root.update()
        if FAKE or until():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture
def bar():
    from jarvis.ui.command_bar import CommandBar
    replies = {"open notepad": "Opening Notepad.", "find invoice": "Found 2 files."}
    b = CommandBar(lambda text: replies.get(text, f"echo: {text}"))
    yield b
    try:
        b.root.destroy()
    except Exception:
        pass


def test_command_bar_answers(bar):
    bar.show()
    pump(bar.root)
    bar.entry.insert(0, "open notepad")
    bar._submit()
    assert pump(bar.root, lambda: "Opening Notepad." in texts(bar.body))
    assert not bar.busy and bar.entry.get() == ""
    bar.hide()
    pump(bar.root, timeout=0.5)


def test_command_bar_result_rows_and_opening_one(bar, tmp_path, monkeypatch):
    f = tmp_path / "invoice.pdf"
    f.write_text("x")
    rows = [{"title": "invoice.pdf", "path": str(f), "source": "files", "ext": "pdf", "mtime": time.time()},
            {"title": "example", "path": "https://example.com", "source": "web", "mtime": 0}]
    bar.results_provider = lambda: rows
    opened = []
    monkeypatch.setattr(os, "startfile", opened.append, raising=False)
    bar.show()
    bar.entry.insert(0, "find invoice")
    bar._submit()
    assert pump(bar.root, lambda: "invoice.pdf" in texts(bar.body))
    bar._open_row(0)
    assert pump(bar.root, lambda: opened == [str(f)], timeout=2)
    assert pump(bar.root, lambda: "Pick another" in texts(bar.body), timeout=2)   # stays open with the rows


def test_command_bar_inline_confirmation(bar):
    from jarvis.safety import Tier
    answer = []
    bar.show()
    t = threading.Thread(target=lambda: answer.append(bar.confirm(Tier.DESTRUCTIVE, "Delete 3 files?", ["a", "b"])))
    t.start()
    assert pump(bar.root, lambda: bar._pending_confirm is not None and "Delete 3 files?" in texts(bar.body))
    bar._key_no()
    t.join(timeout=3)
    assert answer == [False]


def test_command_bar_message_from_another_thread(bar):
    threading.Thread(target=lambda: bar.post("message", "Reminder: stand up")).start()
    assert pump(bar.root, lambda: "Reminder: stand up" in texts(bar.body))


def test_toast_shows_and_closes(bar):
    from jarvis.ui.command_bar import Toast
    t = Toast(bar.root, "Set up for this PC", "Folders: 7 of 7 found.", seconds=0.3)
    pump(bar.root, timeout=0.2)
    assert FAKE or "Folders: 7 of 7 found." in texts(t.win)
    pump(bar.root, timeout=1.0)


def test_hud_draws_and_changes_state(bar):
    from jarvis.hud.metrics import Snapshot
    from jarvis.hud.view import HUD

    class Sampler:
        snapshot = Snapshot(cpu=12, ram=40, ram_used_gb=12.8, ram_total_gb=32, disk=55, disk_free_gb=400,
                            gpu=30, gpu_temp=55, jobs=[{"name": "command model", "pct": 40, "eta_min": 12}])
    h = HUD(bar.root, Sampler(), name="Jarvis", reminders=lambda: [(time.time() + 600, "stand up")],
            click_through=False)
    pump(bar.root, timeout=0.3)
    h.set_state("success", 1)
    h.set_last("Opened Notepad.")
    pump(bar.root, timeout=0.3)
    h.toggle()
    h.toggle()
    pump(bar.root, timeout=0.2)
    h.win.destroy()


def test_settings_every_page_and_save(bar, tmp_path):
    from jarvis.config import Config
    from jarvis.ui.settings import SettingsWindow
    cfg = Config(data_dir=tmp_path)
    w = SettingsWindow(bar.root, cfg)
    for page in w.PAGES:
        w.show(page)
        pump(bar.root, timeout=0.05)
    w.home_addr.set("HSR Layout, Bengaluru")
    w.save()
    if not FAKE:
        assert Config.load(tmp_path / "config.json").places["home"] == "HSR Layout, Bengaluru"


def test_review_window_one_click(bar, tmp_path):
    from jarvis.ui.review import ReviewWindow
    from jarvis.usage import ReviewSession, UsageLog
    u = UsageLog(tmp_path / "u.db", encrypt=False, corrections=tmp_path / "c.jsonl")
    u.record("play super machi", 'play_media(query="sup", site=youtube)', "model 60%",
             {"conf": 0.6, "alts": [[0.6, 'play_media(query="sup", site=youtube)'],
                                    [0.55, 'play_media(query="super machi", site=youtube)']]})
    done = []
    w = ReviewWindow(bar.root, ReviewSession(u, u.to_review(), lambda t: None), on_done=done.append)
    pump(bar.root, timeout=0.1)
    assert FAKE or "play super machi" in texts(w.card)
    w._wrong()
    w._fix('play_media(query="super machi", site=youtube)')
    assert pump(bar.root, lambda: bool(done))
    assert "super machi" in (tmp_path / "c.jsonl").read_text()
    w.close()


class _Ev:
    def __init__(self, x, y):
        self.x_root, self.y_root, self.x, self.y = x, y, 0, 0


def test_command_bar_can_be_dragged_and_remembers_where(bar):
    moved = []
    bar.on_position = lambda which, pos: moved.append((which, pos))
    bar.show()
    pump(bar.root, timeout=0.3)
    x0, y0 = bar.x, bar.y
    bar._drag_start(_Ev(500, 300))
    bar._drag_move(_Ev(440, 360))
    bar._drag_end(_Ev(440, 360))
    assert (bar.x, bar.y) == (x0 - 60, y0 + 60) and moved == [("bar", (x0 - 60, y0 + 60))]
    bar.hide()
    pump(bar.root, timeout=0.5)
    bar.show()                                        # opens where you left it
    pump(bar.root, timeout=0.3)
    assert (bar.x, bar.y) == (x0 - 60, y0 + 60)
    bar.reset_position()                              # double-click the orb: back to the default place
    assert bar.saved_pos is None and moved[-1] == ("bar", None)


def test_command_bar_minimizes_to_an_orb_and_back(bar):
    bar.show()
    pump(bar.root, timeout=0.3)
    bar.minimize()
    assert pump(bar.root, lambda: bar.minimized and bar.root.state() == "withdrawn", timeout=3)
    bar.mini._press(_Ev(10, 10))                      # a click (no drag) brings the bar back
    bar.mini._release(_Ev(10, 10))
    assert pump(bar.root, lambda: not bar.minimized and bar.root.state() == "normal", timeout=3)
    moved = []
    bar.on_position = lambda which, pos: moved.append((which, pos))
    bar.minimize()
    pump(bar.root, lambda: bar.minimized, timeout=3)
    bar.mini._press(_Ev(100, 100))                    # dragging the orb moves it instead
    bar.mini._move(_Ev(160, 140))
    bar.mini._release(_Ev(160, 140))
    assert FAKE or (moved and moved[-1][0] == "mini" and bar.minimized)
    bar.post("toggle")                                # your hotkey also brings it back
    assert pump(bar.root, lambda: not bar.minimized and bar.root.state() == "normal", timeout=3)
