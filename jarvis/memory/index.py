"""The memory index: SQLite full-text search (FTS5, BM25 ranking) over your files and browser history.

    idx = MemoryIndex(path)
    idx.crawl([Path.home() / "Downloads", ...])     # incremental: only new or changed files are read
    idx.add_history(browser.history())
    idx.search("invoice", ext="pdf", newer_than_s=7 * 86400)

Ranking: BM25 (title weighted above body), then a boost for recent items and for matches in the name.
Images and videos are found by name, folder and date; their text comes from the OCR model (step 2).
"""
from __future__ import annotations

import math
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .. import filetypes
from .extract import extract

SKIP_DIRS = {"node_modules", ".git", "__pycache__", ".venv", "venv", "env", ".idea", ".vscode", "AppData",
             "$RECYCLE.BIN", "System Volume Information", ".cache", "site-packages", "dist", "build", ".next",
             "checkpoints", ".jarvis0", "Jarvis trash"}
SCREENSHOT = re.compile(r"^(screenshot|screen shot|screen_shot|capture|snip|scr_|img_\d{8}_)", re.I)
MAX_FILES = 60_000


@dataclass
class Hit:
    title: str
    path: str               # file path or URL
    source: str             # file | web
    ext: str
    mtime: float
    snippet: str
    screenshot: bool
    score: float = 0.0


def words_of_name(name: str) -> str:
    """'Invoice_March-2026 (1).pdf' -> 'Invoice March 2026 1 pdf'; 'myResumeFinal' -> 'my Resume Final'."""
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    return re.sub(r"[_\-.()\[\]{}+,]+", " ", s).strip()


class MemoryIndex:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.lock = threading.RLock()
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS docs (
                id INTEGER PRIMARY KEY, source TEXT, path TEXT UNIQUE, title TEXT, ext TEXT, folder TEXT,
                mtime REAL, size INTEGER, screenshot INTEGER DEFAULT 0, has_text INTEGER DEFAULT 0,
                ocr INTEGER DEFAULT 0);
            CREATE INDEX IF NOT EXISTS docs_mtime ON docs(mtime);
            CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(title, body, place,
                tokenize = 'porter unicode61 remove_diacritics 2');
            CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
            CREATE TABLE IF NOT EXISTS vecs (doc_id INTEGER, k INTEGER, v BLOB, PRIMARY KEY (doc_id, k));
        """)
        self.db.commit()
        self._vec_version = 0
        self._vec_cache: tuple[int, object, object] | None = None

    # ------------------------------------------------------------ writing
    def _upsert(self, source: str, path: str, title: str, body: str, place: str, ext: str, folder: str,
                mtime: float, size: int, screenshot: bool) -> None:
        cur = self.db.execute("SELECT id FROM docs WHERE path = ?", (path,))
        row = cur.fetchone()
        if row:
            self.db.execute("UPDATE docs SET title=?, ext=?, folder=?, mtime=?, size=?, screenshot=?, has_text=?, "
                            "ocr=0 WHERE id=?", (title, ext, folder, mtime, size, int(screenshot), int(bool(body)),
                                                 row[0]))
            self.db.execute("DELETE FROM fts WHERE rowid = ?", (row[0],))
            self.db.execute("DELETE FROM vecs WHERE doc_id = ?", (row[0],))
            self._vec_version += 1
            rid = row[0]
        else:
            rid = self.db.execute("INSERT INTO docs (source, path, title, ext, folder, mtime, size, screenshot, "
                                  "has_text) VALUES (?,?,?,?,?,?,?,?,?)",
                                  (source, path, title, ext, folder, mtime, size, int(screenshot),
                                   int(bool(body)))).lastrowid
        self.db.execute("INSERT INTO fts (rowid, title, body, place) VALUES (?,?,?,?)", (rid, title, body, place))

    def _delete(self, rid: int) -> None:
        self.db.execute("DELETE FROM vecs WHERE doc_id = ?", (rid,))
        self._vec_version += 1
        self.db.execute("DELETE FROM fts WHERE rowid = ?", (rid,))
        self.db.execute("DELETE FROM docs WHERE id = ?", (rid,))

    def crawl(self, roots: list[Path], max_files: int = MAX_FILES, budget_s: float | None = None) -> dict:
        """Index new/changed files under roots, drop deleted ones. Returns counts."""
        t0 = time.time()
        stats = {"seen": 0, "added": 0, "updated": 0, "removed": 0}
        with self.lock:
            known = {p: (rid, m) for rid, p, m in
                     self.db.execute("SELECT id, path, mtime FROM docs WHERE source = 'file'")}
        seen: set[str] = set()
        pending = 0
        for root in roots:
            if not root.is_dir():
                continue
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
                for name in filenames:
                    if stats["seen"] >= max_files or (budget_s and time.time() - t0 > budget_s):
                        break
                    p = Path(dirpath) / name
                    try:
                        st = p.stat()
                    except OSError:
                        continue
                    key = str(p)
                    seen.add(key)
                    stats["seen"] += 1
                    old = known.get(key)
                    if old and abs(old[1] - st.st_mtime) < 1:
                        continue
                    body = extract(p)
                    ext = p.suffix.lower().lstrip(".")
                    folder = Path(dirpath).name
                    shot = bool(SCREENSHOT.match(name)) or folder.lower() in ("screenshots", "screen captures")
                    with self.lock:
                        self._upsert("file", key, name, body, words_of_name(name) + " " + words_of_name(folder),
                                     ext, folder, st.st_mtime, st.st_size, shot)
                        pending += 1
                        if pending >= 500:
                            self.db.commit()
                            pending = 0
                    stats["updated" if old else "added"] += 1
        root_strs = [str(r) for r in roots]
        with self.lock:
            for key, (rid, _) in known.items():
                if key not in seen and any(key.startswith(r) for r in root_strs) and not os.path.exists(key):
                    self._delete(rid)
                    stats["removed"] += 1
            self.db.execute("INSERT OR REPLACE INTO meta VALUES ('last_crawl', ?)", (str(time.time()),))
            self.db.commit()
        return stats

    def add_history(self, rows: list[dict]) -> int:
        with self.lock:
            for r in rows:
                host = re.sub(r"^https?://(www\.)?", "", r["url"]).split("/")[0]
                self._upsert("web", r["url"], r["title"] or host, "", words_of_name(host) + " " + r.get("browser", ""),
                             "", host, r["ts"], 0, False)
            self.db.commit()
        return len(rows)

    def set_text(self, path: str, text: str, ocr: bool = True) -> None:
        """Attach recognised text to an image (OCR, step 2)."""
        with self.lock:
            row = self.db.execute("SELECT id, title, folder FROM docs WHERE path = ?", (path,)).fetchone()
            if not row:
                return
            self.db.execute("DELETE FROM fts WHERE rowid = ?", (row[0],))
            self.db.execute("INSERT INTO fts (rowid, title, body, place) VALUES (?,?,?,?)",
                            (row[0], row[1], text, words_of_name(row[1]) + " " + words_of_name(row[2] or "")))
            self.db.execute("UPDATE docs SET has_text=?, ocr=? WHERE id=?", (int(bool(text)), int(ocr), row[0]))
            if text:                                   # new words: its meaning vectors need redoing
                self.db.execute("DELETE FROM vecs WHERE doc_id = ?", (row[0],))
                self._vec_version += 1
            self.db.commit()

    def needs_ocr(self, limit: int = 100) -> list[str]:
        """Screenshots first, then other images (under 20 MB) that haven't been read yet, newest first."""
        from .ocr import IMAGE_EXT
        exts = sorted(IMAGE_EXT)
        with self.lock:
            rows = self.db.execute(
                f"SELECT path FROM docs WHERE source='file' AND ocr=0 AND size < 20000000 AND ext IN "
                f"({','.join('?' * len(exts))}) ORDER BY screenshot DESC, mtime DESC LIMIT ?", exts + [limit]).fetchall()
        return [r[0] for r in rows]

    def needs_embed(self, limit: int = 2000) -> list[tuple[int, str, str]]:
        """Items without meaning vectors yet, newest first: [(id, title, body)]."""
        with self.lock:
            return self.db.execute(
                "SELECT d.id, fts.title, fts.body FROM docs d JOIN fts ON fts.rowid = d.id "
                "WHERE NOT EXISTS (SELECT 1 FROM vecs v WHERE v.doc_id = d.id) ORDER BY d.mtime DESC LIMIT ?",
                (limit,)).fetchall()

    def add_vectors(self, doc_id: int, mat) -> None:
        import numpy as np
        with self.lock:
            self.db.execute("DELETE FROM vecs WHERE doc_id = ?", (doc_id,))
            for k, v in enumerate(np.asarray(mat, np.float16)):
                self.db.execute("INSERT INTO vecs VALUES (?,?,?)", (doc_id, k, v.tobytes()))
            self._vec_version += 1

    def commit(self) -> None:
        with self.lock:
            self.db.commit()

    def _vectors(self):
        """(doc_ids, matrix) of all meaning vectors, cached until something changes."""
        import numpy as np
        if self._vec_cache and self._vec_cache[0] == self._vec_version:
            return self._vec_cache[1], self._vec_cache[2]
        with self.lock:
            rows = self.db.execute("SELECT doc_id, v FROM vecs").fetchall()
        if not rows:
            ids, mat = np.zeros(0, np.int64), np.zeros((0, 0), np.float32)
        else:
            ids = np.array([r[0] for r in rows], np.int64)
            mat = np.stack([np.frombuffer(r[1], np.float16) for r in rows]).astype(np.float32)
        self._vec_cache = (self._vec_version, ids, mat)
        return ids, mat

    # ------------------------------------------------------------ reading
    def count(self) -> dict:
        with self.lock:
            rows = dict(self.db.execute("SELECT source, COUNT(*) FROM docs GROUP BY source").fetchall())
            shots = self.db.execute("SELECT COUNT(*) FROM docs WHERE screenshot = 1").fetchone()[0]
            read = self.db.execute("SELECT COUNT(*) FROM docs WHERE ocr = 1").fetchone()[0]
            vec = self.db.execute("SELECT COUNT(DISTINCT doc_id) FROM vecs").fetchone()[0]
        return {"files": rows.get("file", 0), "web": rows.get("web", 0), "screenshots": shots, "images_read": read,
                "with_meaning": vec}

    @staticmethod
    def _fts_query(query: str, any_word: bool = False) -> str:
        toks = [t for t in re.findall(r"\w+", query.lower()) if len(t) > 1 or t.isdigit()]
        return (" OR " if any_word else " ").join(f'"{t}"*' for t in toks)

    def search(self, query: str, ext: str | None = None, newer_than_s: float | None = None,
               older_than_s: float | None = None, where: str | None = None, limit: int = 8,
               now: float | None = None, embedder=None, min_sim: float = 0.45) -> list[Hit]:
        now = time.time() if now is None else now
        cond, args = [], []
        if where == "web":
            cond.append("d.source = 'web'")
        elif where == "screenshots":
            cond.append("d.screenshot = 1")
        elif where == "files" or ext:
            cond.append("d.source = 'file'")
        if ext:
            exts = sorted(filetypes.CATEGORIES.get(ext, {ext}))
            cond.append(f"d.ext IN ({','.join('?' * len(exts))})")
            args += exts
        if newer_than_s:
            cond.append("d.mtime >= ?")
            args.append(now - newer_than_s)
        if older_than_s:
            cond.append("d.mtime <= ?")
            args.append(now - older_than_s)
        where_sql = (" AND " + " AND ".join(cond)) if cond else ""
        q = self._fts_query(query)
        rows = []
        with self.lock:
            if q:
                for any_word in (False, True):        # all words first; if nothing, any word
                    rows = self.db.execute(
                        "SELECT d.title, d.path, d.source, d.ext, d.mtime, d.screenshot, "
                        "snippet(fts, 1, '[', ']', ' … ', 10), bm25(fts, 6.0, 1.0, 3.0) "
                        f"FROM fts JOIN docs d ON d.id = fts.rowid WHERE fts MATCH ?{where_sql} "
                        "ORDER BY bm25(fts, 6.0, 1.0, 3.0) LIMIT 60", [self._fts_query(query, any_word)] + args
                    ).fetchall()
                    if rows:
                        break
            else:                                     # "my latest screenshot": newest first
                rows = self.db.execute(
                    "SELECT d.title, d.path, d.source, d.ext, d.mtime, d.screenshot, '', 0 "
                    f"FROM docs d WHERE 1=1{where_sql} ORDER BY d.mtime DESC LIMIT ?", args + [limit]).fetchall()
        qwords = set(re.findall(r"\w+", query.lower()))

        def boosts(title, mtime):
            age_days = max(0.0, (now - (mtime or 0)) / 86400)
            name_words = set(re.findall(r"\w+", words_of_name(title).lower()))
            return len(qwords & name_words) / max(1, len(qwords)), math.exp(-age_days / 30)

        hits = []
        for title, path, source, e, mtime, shot, snip, bm in rows:
            in_name, fresh = boosts(title, mtime)
            hits.append(Hit(title, path, source, e or "", mtime or 0, snip or "", bool(shot),
                            -bm + 2.0 * in_name + 1.5 * fresh))
        hits.sort(key=lambda h: -h.score)
        if not (embedder is not None and query.strip()):
            return hits[:limit]
        # meaning search: nearest vectors, same filters, merged with the word results (reciprocal rank fusion)
        sem = self._semantic(query, embedder, where_sql, args, min_sim)
        fused: dict[str, list] = {}
        for rank, h in enumerate(hits[:60]):
            fused.setdefault(h.path, [h, 0.0])[1] += 1.0 / (60 + rank)
        for rank, (h, sim) in enumerate(sem):
            fused.setdefault(h.path, [h, 0.0])[1] += 1.0 / (60 + rank)
        out = []
        for h, rrf in fused.values():
            in_name, fresh = boosts(h.title, h.mtime)
            h.score = rrf + 0.004 * in_name + 0.002 * fresh
            out.append(h)
        out.sort(key=lambda h: -h.score)
        return out[:limit]

    def _semantic(self, query: str, embedder, where_sql: str, args: list, min_sim: float) -> list:
        import numpy as np
        ids, mat = self._vectors()
        if not len(ids):
            return []
        q = embedder.embed([query])[0]
        sims = mat @ q
        best: dict[int, float] = {}
        for i in np.argsort(-sims)[:400]:
            if sims[i] < min_sim:
                break
            d = int(ids[i])
            best[d] = max(best.get(d, -1.0), float(sims[i]))
        if not best:
            return []
        marks = ",".join("?" * len(best))
        with self.lock:
            rows = self.db.execute(
                "SELECT d.id, d.title, d.path, d.source, d.ext, d.mtime, d.screenshot, substr(fts.body, 1, 120) "
                f"FROM docs d JOIN fts ON fts.rowid = d.id WHERE d.id IN ({marks}){where_sql}",
                list(best) + args).fetchall()
        out = [(Hit(t, p, src, e or "", m or 0, (b or "")[:120], bool(sh), best[i]), best[i])
               for i, t, p, src, e, m, sh, b in rows]
        out.sort(key=lambda x: -x[1])
        return out[:60]


_instances: dict[str, MemoryIndex] = {}


def get(data_dir: Path) -> MemoryIndex:
    key = str(data_dir)
    if key not in _instances:
        _instances[key] = MemoryIndex(data_dir / "memory.sqlite")
    return _instances[key]


def close(data_dir: Path) -> None:
    """Close the cached index for data_dir (so the health check can move a damaged file aside)."""
    idx = _instances.pop(str(data_dir), None)
    if idx is not None:
        with idx.lock:
            idx.db.close()
