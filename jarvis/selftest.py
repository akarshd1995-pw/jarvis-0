"""End-to-end self-test (phase 42): every action, in a throwaway folder, without touching your PC.

    python -m jarvis.selftest          (the updater runs this on a new version before switching to it)
    /selftest                          (the same, from the command bar)

Runs for real, inside a temporary home folder: listing, moving, copying, deleting (to Jarvis's trash), new
folders, undo, "move these" (context references), reminders, memory search, follow-ups, settings save/load.
Runs as a preview (dry run) only: everything that would reach outside that folder (opening apps and sites,
media keys, volume, Wi-Fi, power, windows, screenshots, directions...). It also checks that every action in
the command language has a handler and that every reply can be written.
Exit code 0 when everything passed.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import traceback
import webbrowser
from dataclasses import dataclass
from pathlib import Path

# A preview program for every action that must not really run here
DRY = {
    "open_app": 'open_app(name=notepad)', "open_url": 'open_url(url="example.com")',
    "open_folder": "open_folder(path=Downloads)", "search_web": 'search_web(query="jarvis", site=google)',
    "play_media": 'play_media(query="lofi beats", site=youtube)', "system": "system(action=mute)",
    "window": "window(action=minimize_all)", "desktop": "desktop(action=new)", "media": "media(action=play_pause)",
    "brightness": "brightness(level=50)", "set_volume": "set_volume(level=30)",
    "radio": "radio(kind=wifi, state=on)", "screenshot": "screenshot(mode=full)",
    "screen_record": "screen_record(action=start)", "power": "power(action=sleep)",
    "directions": 'directions(destination="airport")', "maintenance": "maintenance(action=health_check)",
    "briefing": "briefing(kind=morning)", "training": "training(action=status)",
    "unsupported": 'unsupported(request="order a pizza")', "unknown": "unknown()",
    "mood": "mood(kind=happy)", "open_result": "open_result(n=1)",
    "workspace": 'workspace(action=start, name="coding")',
}
LIVE_ACTIONS = {"list_files", "move_files", "copy_files", "delete_files", "remind", "search_memory", "undo",
                "open_result", "note", "find_notes", "delete_note"}


@dataclass
class Check:
    name: str
    ok: bool
    detail: str = ""


class _NoOutside:
    """While the self-test runs, nothing may open on your screen."""

    def __enter__(self):
        self.opened = []
        self._sf = getattr(os, "startfile", None)
        self._wb = webbrowser.open
        os.startfile = lambda p, *a: self.opened.append(str(p))     # type: ignore[attr-defined]
        webbrowser.open = lambda u, *a, **k: self.opened.append(u) or True
        return self

    def __exit__(self, *exc):
        if self._sf is None:
            del os.startfile                                         # type: ignore[attr-defined]
        else:
            os.startfile = self._sf                                  # type: ignore[attr-defined]
        webbrowser.open = self._wb


def _sandbox(root: Path):
    from .config import Config
    home = root / "home"
    folders = {n: home / n for n in ("Downloads", "Documents", "Desktop", "Pictures", "Music", "Videos")}
    folders["Archive"] = home / "Documents" / "Archive"
    for p in folders.values():
        p.mkdir(parents=True, exist_ok=True)
    dl = folders["Downloads"]
    old = time.time() - 60 * 86400
    for name in ("invoice_march.pdf", "notes.pdf", "photo.png", "song.mp3", "setup.exe", "clip.mp4"):
        (dl / name).write_text(f"selftest {name}")
    os.utime(dl / "notes.pdf", (old, old))
    cfg = Config(data_dir=root / "data", folders={k: str(v) for k, v in folders.items()},
                 allowed_roots=[str(home)], watch_folders=["Downloads"], memory_folders=["Downloads", "Documents"],
                 memory_browser=False, memory_ocr=False, memory_meaning=False, encrypt_logs=False,
                 persona_model=False, dialog_model=False, show_details=False, learn_from_usage=True,
                 model_backend="rules")             # the rules only: the same result every time
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.trash_dir.mkdir(parents=True, exist_ok=True)
    return cfg, folders


def run(log=print) -> list[Check]:
    from .context import Explorer, ScreenContext
    from .daemon import Jarvis
    from .dsl.grammar import SCHEMA, parse
    from .executor import REGISTRY
    checks: list[Check] = []

    def check(name, fn):
        try:
            r = fn()
            ok, detail = (r if isinstance(r, tuple) else (bool(r) if r is not None else True, ""))
        except Exception as e:
            ok, detail = False, f"{type(e).__name__}: {e}"
            if os.environ.get("JARVIS_SELFTEST_TRACE"):
                detail += "\n" + traceback.format_exc()
        checks.append(Check(name, ok, detail))
        log(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"  ({detail})" if detail and not ok else ""))
        return ok

    root = Path(tempfile.mkdtemp(prefix="jarvis-selftest-"))
    saved_registry = dict(REGISTRY)            # the test Jarvis registers its own handlers: put yours back after
    try:
        with _NoOutside() as outside:
            cfg, F = _sandbox(root)
            dl, arc = F["Downloads"], F["Archive"]
            j = Jarvis(cfg, confirm=lambda *a: True, data_root=root / "project-data")
            j.screen._explorer = lambda h: None
            j.screen._clip = lambda: ([], "")
            names = lambda p: sorted(x.name for x in p.iterdir() if x.is_file())     # noqa: E731

            def reply_ok(text):
                r = j.handle(text)
                bad = ("went wrong" in r.lower()) or ("couldn't" in r.lower() and "find" not in r.lower())
                return (not bad, r[:120])

            # 1. wiring: every action has a handler
            own = {"undo", "mood"}             # undo: the executor itself; mood: handled before anything runs
            check("every action has a handler", lambda: (not (set(SCHEMA) - set(REGISTRY) - own),
                                                         ", ".join(sorted(set(SCHEMA) - set(REGISTRY) - own))))
            covered = set(DRY) | LIVE_ACTIONS
            check("the self-test covers every action", lambda: (not (set(SCHEMA) - covered),
                                                                ", ".join(sorted(set(SCHEMA) - covered))))
            # 2. files, for real, in the sandbox
            check("list files", lambda: reply_ok("list_files(src=Downloads, ext=pdf)"))
            check("move old pdfs to Archive", lambda: (reply_ok("move_files(src=Downloads, ext=pdf, older_than=30d, "
                                                                "dest=Archive)")[0] and names(arc) == ["notes.pdf"],
                                                       str(names(arc))))
            check("undo the move", lambda: (reply_ok("undo()")[0] and "notes.pdf" in names(dl) and not names(arc),
                                            str(names(dl))))
            check("copy videos into a new folder", lambda: (
                reply_ok('copy_files(src=Downloads, ext=video, new_folder="Clips")')[0]
                and names(dl / "Clips") == ["clip.mp4"] and "clip.mp4" in names(dl), ""))
            check("delete (to Jarvis's trash) and undo", lambda: (
                reply_ok("delete_files(src=Downloads, ext=exe)")[0] and "setup.exe" not in names(dl)
                and reply_ok("undo()")[0] and "setup.exe" in names(dl), str(names(dl))))
            check("a sentence understood by the rules", lambda: (
                reply_ok("move the mp3 files in downloads to music")[0] and names(F["Music"]) == ["song.mp3"],
                str(names(F["Music"]))))
            check("follow-up: 'again' repeats safely", lambda: reply_ok("again")[0])
            # 3. context: "these" = the files selected in File Explorer
            j.screen._explorer = lambda h: Explorer(str(dl), [str(dl / "photo.png")])
            check("'move these to Pictures' (Explorer selection)", lambda: (
                reply_ok("move these to pictures")[0] and names(F["Pictures"]) == ["photo.png"],
                str(names(F["Pictures"]))))
            j.screen._explorer = lambda h: None
            check("'open the second one' after a list", lambda: (
                reply_ok("list_files(src=Downloads)")[0] and reply_ok("open the second one")[0]
                and len(outside.opened) == 1, str(outside.opened)))
            # 4. reminders and memory search
            check("reminder", lambda: (reply_ok('remind(text="selftest", after=10m)')[0]
                                       and j.store.upcoming_reminders(time.time(), 10)[-1][1] == "selftest", ""))

            def memory():
                from .memory import index as MI
                idx = MI.get(cfg.data_dir)
                idx.crawl(j.memory_roots())
                r = j.handle('search_memory(query="invoice")')
                rows = j.take_results()
                MI.close(cfg.data_dir)
                return any("invoice" in x["title"] for x in rows), r[:120]
            check("memory search finds a file by name", memory)

            # notes and journal (phase 30)
            def notes():
                ok1 = reply_ok('note(text="call the bank about the card")')[0]
                ok2 = reply_ok('note(text="finished the installer", kind=journal)')[0]
                r = j.handle('find_notes(query="bank")')
                found = "call the bank" in r
                gone = reply_ok('delete_note(query="bank")')[0] and \
                    "No notes" in j.handle('find_notes(query="bank")')
                back = reply_ok("undo()")[0] and "call the bank" in j.handle('find_notes(query="bank")')
                return all((ok1, ok2, found, gone, back)), r[:100]
            check("notes: save, find, delete, undo", notes)
            # 5. everything else as a preview
            from .actions.base import Ctx
            ctx = Ctx(cfg, j.store, "selftest")
            check("mood: a kind reply, nothing run", lambda: reply_ok("i'm in a great mood today"))
            for action, program in DRY.items():
                if action in ("open_result", "mood"):
                    continue

                def dry(program=program):
                    call = parse(program).calls[0]
                    if call.action == "maintenance":
                        return j.handle("/doctor") != "", ""            # the health check reads only
                    res = REGISTRY[call.action](dict(call.args), ctx, True)
                    j.persona.result(res)                              # the reply can be written
                    return True, res.summary[:80]
                check(f"preview: {action}", dry)
            check("nothing opened on screen during previews", lambda: (len(outside.opened) == 1, str(outside.opened)))
            # 6. settings survive a restart
            from .config import Config

            def settings():
                cfg.places = {"home": "Selftest Road", "work": ""}
                p = cfg.save()
                return Config.load(p).places.get("home") == "Selftest Road", ""
            check("settings are saved and loaded", settings)
            j.store._db.close()
    except Exception as e:
        checks.append(Check("self-test ran", False, f"{type(e).__name__}: {e}"))
        log(f"  FAIL self-test crashed: {e}")
    finally:
        REGISTRY.clear()
        REGISTRY.update(saved_registry)
        shutil.rmtree(root, ignore_errors=True)
    return checks


def summary(checks: list[Check]) -> str:
    bad = [c for c in checks if not c.ok]
    if not bad:
        return f"Self-test passed: {len(checks)} checks, every action works."
    return (f"Self-test: {len(bad)} of {len(checks)} checks failed:\n"
            + "\n".join(f"  - {c.name}: {c.detail}" for c in bad))


def main() -> int:
    print("Jarvis self-test (a throwaway folder; nothing on your PC is touched)")
    checks = run()
    print(summary(checks))
    return 0 if all(c.ok for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
