"""Your words (phase 43): the names and spellings you actually use, in vocab.json in Jarvis's data folder.

  aliases   nicknames you taught: "when I say dl I mean downloads" -> every "dl" is read as "downloads", now
  typos     misspellings you repeat ("donwloads"), found in your past commands; used as training examples
  apps      app names from your Start menu (Obsidian, WhatsApp Beta, Cursor ...)
  projects  your code projects (folders with a .git inside, under your user folder): "open the everway project"
  folders   folders in Documents, Desktop and Downloads: "move the pdfs to Tax 2026"

Everything except aliases is gathered automatically (refresh), at most once a week and before every improvement
run, and turned into personal training examples (datagen/personal.py), so the model learns your words.
Project and folder names also work right away as places for file commands.
"""
from __future__ import annotations

import json
import os
import re
import time
from collections import Counter
from pathlib import Path

REFRESH_S = 7 * 86400
CURRENT: "Vocab | None" = None        # the running Jarvis's vocabulary: undo changes it directly
SKIP_DIRS = {"appdata", "node_modules", ".venv", "venv", "__pycache__", ".git", ".cache", "site-packages", "$recycle.bin",
             "onedrive - personal", ".jarvis0", "anaconda3", "miniconda3", ".vscode", ".cursor", "go", ".npm"}


def norm_name(s: str) -> str:
    return re.sub(r"[\s_\-.]+", " ", s.lower()).strip()


class Vocab:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._mtime = 0.0
        self.data = {"aliases": {}, "typos": {}, "apps": [], "projects": {}, "folders": {}, "refreshed": 0}
        self._read()

    @staticmethod
    def _stamp(path: Path):
        st = path.stat()
        return st.st_mtime_ns, st.st_size

    def _read(self) -> None:
        try:
            self.data.update(json.loads(self.path.read_text(encoding="utf-8")))
            self._mtime = self._stamp(self.path)
        except (OSError, ValueError):
            pass

    def reload_if_changed(self) -> None:
        """Something else changed the file (another Jarvis process, an edit): read it again. Windows timestamps
        can stay the same for quick writes, so size counts too; undo inside Jarvis uses CURRENT instead."""
        try:
            if self._stamp(self.path) != self._mtime:
                self._read()
        except OSError:
            pass

    @classmethod
    def load(cls, data_dir: Path) -> "Vocab":
        return cls(Path(data_dir) / "vocab.json")

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)
        try:
            self._mtime = self._stamp(self.path)
        except OSError:
            pass

    # -------------------------------------------------------------- nicknames
    @staticmethod
    def _clean_word(w: str) -> str:
        return re.sub(r"\s+", " ", str(w).strip().strip("\"'").lower())

    def teach(self, word: str, means: str) -> tuple[str, str, str | None]:
        w, m = self._clean_word(word), str(means).strip().strip("\"'")
        if not w or not m or w == m.lower():
            raise ValueError("tell me the word and what it means, like \"when I say dl I mean downloads\"")
        if len(w) < 2:
            raise ValueError("that nickname is too short to recognise safely")
        before = self.data["aliases"].get(w)
        self.data["aliases"][w] = m
        self.save()
        return w, m, before

    def forget(self, word: str) -> str | None:
        before = self.data["aliases"].pop(self._clean_word(word), None)
        self.save()
        return before

    def set_alias(self, word: str, means: str | None) -> None:
        if means is None:
            self.data["aliases"].pop(word, None)
        else:
            self.data["aliases"][word] = means
        self.save()

    def expand(self, text: str) -> str:
        """Your nicknames, read as what they mean (whole words only, longest first)."""
        al = self.data.get("aliases") or {}
        if not al:
            return text
        for w in sorted(al, key=len, reverse=True):
            text = re.sub(rf"(?<![\w-]){re.escape(w)}(?![\w-])", al[w].replace("\\", "\\\\"), text, flags=re.I)
        return text

    # -------------------------------------------------------------- places
    def places(self) -> dict[str, str]:
        """name -> folder for your projects and folders (projects win)."""
        out = {norm_name(k): v for k, v in (self.data.get("folders") or {}).items()}
        out.update({norm_name(k): v for k, v in (self.data.get("projects") or {}).items()})
        return out

    # -------------------------------------------------------------- gathering
    def stale(self, now: float | None = None) -> bool:
        return (time.time() if now is None else now) - self.data.get("refreshed", 0) > REFRESH_S

    def refresh(self, config, known_words: set[str] | None = None, texts: list[str] | None = None,
                home: Path | None = None, budget_s: float = 4.0) -> dict:
        """Look around (quickly): apps, projects, folders; typos if your past commands are given."""
        home = Path(home or Path.home())
        apps_json = Path(config.data_dir) / "apps.json"
        try:
            names = [e["name"] for e in json.loads(apps_json.read_text(encoding="utf-8"))["apps"]]
            self.data["apps"] = sorted({n for n in names if 2 <= len(n) <= 40})[:600]
        except (OSError, ValueError, KeyError):
            pass
        self.data["projects"] = find_projects(home, budget_s)
        folders = {}
        for alias in ("Documents", "Desktop", "Downloads"):
            base = config.folders.get(alias)
            if not base or not Path(base).is_dir():
                continue
            try:
                for p in sorted(Path(base).iterdir())[:400]:
                    if p.is_dir() and not p.name.startswith((".", "$")) and len(p.name) <= 50:
                        folders.setdefault(p.name, str(p))
            except OSError:
                continue
        self.data["folders"] = dict(list(folders.items())[:600])
        if texts is not None and known_words:
            self.data["typos"] = learn_typos(texts, known_words | self.word_set())
        self.data["refreshed"] = time.time()
        self.save()
        return {"apps": len(self.data["apps"]), "projects": len(self.data["projects"]),
                "folders": len(self.data["folders"]), "typos": len(self.data.get("typos", {}))}

    def word_set(self) -> set[str]:
        words = set()
        for n in list(self.data.get("apps", [])) + list(self.places()):
            words |= set(re.findall(r"[a-z]+", n.lower()))
        for w, m in (self.data.get("aliases") or {}).items():
            words |= set(re.findall(r"[a-z]+", (w + " " + m).lower()))
        return words


def find_projects(home: Path, budget_s: float = 4.0, max_depth: int = 4, limit: int = 200) -> dict[str, str]:
    """Folders with a .git inside, under your user folder (skipping AppData, node_modules ...), quickly."""
    end = time.time() + budget_s
    out: dict[str, str] = {}
    stack = [(home, 0)]
    while stack and time.time() < end and len(out) < limit:
        d, depth = stack.pop()
        try:
            entries = list(os.scandir(d))
        except OSError:
            continue
        if any(e.name == ".git" for e in entries) and d != home:
            out.setdefault(d.name, str(d))
            continue                                             # a project's own subfolders aren't projects
        if depth >= max_depth:
            continue
        for e in entries:
            if e.is_dir(follow_symlinks=False) and e.name.lower() not in SKIP_DIRS and not e.name.startswith("."):
                stack.append((Path(e.path), depth + 1))
    return out


def _damerau1(a: str, b: str) -> bool:
    """One edit apart (insert, delete, substitute or swap two neighbours)."""
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        diff = [i for i in range(len(a)) if a[i] != b[i]]
        return len(diff) == 1 or (len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]]
                                  and a[diff[1]] == b[diff[0]])
    s, t = (a, b) if len(a) < len(b) else (b, a)
    for i in range(len(t)):
        if t[:i] + t[i + 1:] == s:
            return True
    return False


def learn_typos(texts: list[str], known: set[str], min_count: int = 2) -> dict[str, str]:
    """Words you typed at least twice that aren't words Jarvis knows but are one slip away from exactly one."""
    counts = Counter(w for t in texts for w in re.findall(r"[a-z]{4,}", t.lower()))
    by_first: dict[str, list[str]] = {}
    for k in known:
        if len(k) >= 3:
            by_first.setdefault(k[0], []).append(k)
    out = {}
    for w, n in counts.items():
        if n < min_count or w in known:
            continue
        cands = [k for k in by_first.get(w[0], []) if _damerau1(w, k)]
        if len(cands) == 1:
            out[w] = cands[0]
    return dict(sorted(out.items())[:300])


def known_words(samples: int = 20000, seed: int = 0) -> set[str]:
    """Every word the command model is taught (from the training data generators)."""
    import random
    from datagen.generate_intents import GENERATORS, Ctx
    rng = random.Random(seed)
    fns, weights = zip(*GENERATORS)
    words: set[str] = set()
    for _ in range(samples):
        fn = rng.choices(fns, weights)[0]
        try:
            _, text, _ = fn(Ctx(rng, "train"))
        except Exception:
            continue
        words |= set(re.findall(r"[a-z]+", text.replace("\x01", " ").replace("\x02", " ").lower()))
    return words
