"""Fonts to render training text with: the ones installed on this PC (Windows: C:/Windows/Fonts)."""
from __future__ import annotations

import os
from pathlib import Path

# Latin UI / document / code fonts; symbol and CJK-only fonts are skipped
PREFERRED = ["segoeui", "segoeuib", "segoeuisl", "seguisb", "SegUIVar", "arial", "arialbd", "calibri", "calibrib",
             "cambria", "consola", "consolab", "CascadiaCode", "CascadiaMono", "tahoma", "tahomabd", "verdana",
             "verdanab", "times", "timesbd", "georgia", "trebuc", "trebucbd", "cour", "courbd", "lucon", "corbel",
             "candara", "constan", "framd", "micross", "Inter", "DejaVuSans", "DejaVuSansMono", "DejaVuSerif",
             "LiberationSans", "LiberationMono", "Roboto", "NotoSans", "Ubuntu"]
SKIP = ("symbol", "wingding", "webding", "marlett", "holomdl", "segmdl", "seguiemj", "seguisym", "mtextra",
        "bssym", "msgothic", "msmincho", "simsun", "malgun", "mingliu", "yugoth", "meiryo", "msyh", "msjh",
        "gadugi", "nirmala", "ebrima", "javatext", "mvboli", "segoepr", "segoesc", "comic", "impact")


def font_dirs() -> list[Path]:
    dirs = [Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts",
            Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "Windows" / "Fonts",
            Path("/usr/share/fonts"), Path("/usr/local/share/fonts"), Path.home() / ".fonts",
            Path("/System/Library/Fonts"), Path("/Library/Fonts")]
    return [d for d in dirs if str(d) and d.is_dir()]


def find_fonts(limit: int = 60) -> list[str]:
    found = []
    for d in font_dirs():
        for p in d.rglob("*"):
            if p.suffix.lower() not in (".ttf", ".otf"):
                continue
            stem = p.stem.lower()
            if any(s in stem for s in SKIP):
                continue
            found.append(p)
    pref = [p for p in found if any(p.stem.lower().startswith(k.lower()) for k in PREFERRED)]
    rest = [p for p in found if p not in pref]
    out = [str(p) for p in pref] + [str(p) for p in rest]
    if not out:
        raise SystemExit("No fonts found to render training text with.")
    return out[:limit]
