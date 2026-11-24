"""Phase 29: Jarvis never falls over quietly.

- setup_logging(): a rotating log file (logs/jarvis.log) even when running without a console (pythonw),
  plus a crash report file for any unhandled error, in the main thread or a background one
- SingleInstance: one Jarvis at a time; starting a second one just opens the first one's command bar
- CrashLog: the supervisor's record of recent crashes (repeated crashes -> safe mode)
- Watchdog: restarts background services (indexer, briefings, monitor, ...) if their thread dies
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import logging.handlers
import platform
import socket
import sys
import threading
import time
import traceback
from pathlib import Path

log = logging.getLogger("jarvis.reliability")
PORT = 47913                     # localhost only: how a second Jarvis finds the first


def logs_dir(data_dir: Path) -> Path:
    d = data_dir / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_crash_report(data_dir: Path, exc_type, exc, tb, where: str = "main") -> Path:
    from .updater import PROJECT_ROOT
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = logs_dir(data_dir) / f"crash-{stamp}.txt"
    try:
        head = (PROJECT_ROOT / ".git" / "HEAD").read_text().strip()
        if head.startswith("ref:"):
            head = (PROJECT_ROOT / ".git" / head.split()[1]).read_text().strip()
    except OSError:
        head = "unknown"
    text = (f"Jarvis-0 crash report  {dt.datetime.now().isoformat(timespec='seconds')}\n"
            f"where: {where}\nversion: {head[:10]}\npython: {sys.version.split()[0]}  {platform.platform()}\n\n"
            + "".join(traceback.format_exception(exc_type, exc, tb)))
    path.write_text(text, encoding="utf-8")
    old = sorted(logs_dir(data_dir).glob("crash-*.txt"))[:-30]          # keep the last 30
    for p in old:
        p.unlink(missing_ok=True)
    return path


def setup_logging(data_dir: Path, verbose: bool = False) -> Path:
    fmt = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    for h in list(root.handlers):
        root.removeHandler(h)
    path = logs_dir(data_dir) / "jarvis.log"
    fh = logging.handlers.RotatingFileHandler(path, maxBytes=2_000_000, backupCount=5, encoding="utf-8")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    if sys.stderr is not None:                           # pythonw has no console
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        root.addHandler(ch)

    def on_crash(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            return sys.__excepthook__(exc_type, exc, tb)
        p = write_crash_report(data_dir, exc_type, exc, tb)
        logging.getLogger("jarvis").critical("crashed: %s (report: %s)", exc, p)

    def on_thread_crash(args):
        p = write_crash_report(data_dir, args.exc_type, args.exc_value, args.exc_traceback,
                               where=f"thread {args.thread.name if args.thread else '?'}")
        logging.getLogger("jarvis").error("background thread %s died: %s (report: %s)",
                                          args.thread.name if args.thread else "?", args.exc_value, p)

    sys.excepthook = on_crash
    threading.excepthook = on_thread_crash
    return path


def recent_crashes(data_dir: Path, days: float = 7) -> list[Path]:
    cut = time.time() - days * 86400
    return [p for p in sorted(logs_dir(data_dir).glob("crash-*.txt")) if p.stat().st_mtime >= cut]


# ------------------------------------------------------------------ one Jarvis at a time
class SingleInstance:
    """Holds a localhost port while Jarvis runs. A second Jarvis of the SAME Windows user sends 'show' to it and
    exits. Each user has their own port (phase 44); if another user's Jarvis holds it, the next one is tried."""

    TRIES = 5

    def __init__(self, port: int | None = None, user: str | None = None):
        from .machine import user_name, user_port
        self.user = (user or user_name()).lower()
        self.port = port if port is not None else user_port(self.user)
        self.sock: socket.socket | None = None
        self.on_show = None

    def _bind(self, port: int) -> socket.socket | None:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):              # Windows: nobody else may share the port
            s.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        else:                                                   # elsewhere: reuse right after a restart
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            s.close()
            return None
        return s

    def _whose(self, port: int) -> str | None:
        """The user of the Jarvis on that port (None: not a Jarvis, or none)."""
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2) as c:
                c.sendall(b"who\n")
                reply = c.recv(128).decode(errors="ignore").strip()
                return reply[4:].lower() if reply.startswith("ok: ") else None
        except OSError:
            return None

    def acquire(self) -> bool:
        """True: this is the only Jarvis for this user. False: this user's Jarvis is already running."""
        base = self.port
        for i in range(self.TRIES):
            port = base + i
            s = self._bind(port)
            if s is not None:
                self.port, self.sock = port, s
                s.listen(4)
                threading.Thread(target=self._serve, daemon=True, name="single-instance").start()
                return True
            if self._whose(port) == self.user:                  # ours: tell it to show itself
                self.port = port
                return False
        self.port = base
        return False

    def _serve(self):
        while self.sock is not None:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            with conn:
                try:
                    msg = conn.recv(64).decode(errors="ignore").strip()
                    if msg == "who":
                        conn.sendall(f"ok: {self.user}".encode())
                        continue
                    if msg == "show" and self.on_show:
                        self.on_show()
                    conn.sendall(b"ok")
                except OSError:
                    pass

    def ping_existing(self) -> bool:
        """Ask this user's running Jarvis to show its command bar. True if one answered."""
        try:
            with socket.create_connection(("127.0.0.1", self.port), timeout=2) as c:
                c.sendall(b"show\n")
                return c.recv(8) == b"ok"
        except OSError:
            return False

    def release(self):
        if self.sock is not None:
            s, self.sock = self.sock, None
            s.close()


# ------------------------------------------------------------------ crash counting for the supervisor
class CrashLog:
    def __init__(self, path: Path):
        self.path = path

    def _load(self) -> list[float]:
        try:
            return [float(x) for x in json.loads(self.path.read_text())]
        except (OSError, ValueError, TypeError):
            return []

    def add(self, when: float | None = None) -> None:
        times = [t for t in self._load() if t > time.time() - 86400] + [when or time.time()]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(times))

    def recent(self, window_s: float = 600) -> int:
        cut = time.time() - window_s
        return sum(1 for t in self._load() if t >= cut)

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)


# ------------------------------------------------------------------ background services
class Watchdog(threading.Thread):
    """Every minute: any registered service whose thread has died is started again (at most 5 times an hour)."""

    def __init__(self, every_s: float = 60):
        super().__init__(daemon=True, name="watchdog")
        self.every_s = every_s
        self.services: dict[str, dict] = {}
        self.lock = threading.Lock()

    def register(self, name: str, thread: threading.Thread, restart) -> None:
        with self.lock:
            self.services[name] = {"thread": thread, "restart": restart, "restarts": []}

    def check(self) -> list[str]:
        restarted = []
        with self.lock:
            items = list(self.services.items())
        for name, s in items:
            if s["thread"].is_alive():
                continue
            s["restarts"] = [t for t in s["restarts"] if t > time.time() - 3600]
            if len(s["restarts"]) >= 5:
                continue
            try:
                s["thread"] = s["restart"]()
                s["restarts"].append(time.time())
                restarted.append(name)
                log.warning("restarted background service %r", name)
            except Exception as e:
                log.error("couldn't restart %r: %s", name, e)
        return restarted

    def status(self) -> dict[str, bool]:
        with self.lock:
            return {n: s["thread"].is_alive() for n, s in self.services.items()}

    def run(self):
        while True:
            time.sleep(self.every_s)
            try:
                self.check()
            except Exception as e:
                log.error("watchdog: %s", e)
