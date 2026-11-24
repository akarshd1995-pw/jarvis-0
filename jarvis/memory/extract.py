"""Text from files, standard library only (pypdf is used for PDFs if it's installed).

Each extractor returns plain text, capped, or "" when the file has no readable text (images, videos,
binaries: those are found by name, folder and date; screenshots get their text from the OCR model, step 2).
"""
from __future__ import annotations

import html
import re
import zipfile
from pathlib import Path

MAX_CHARS = 60_000
TEXT_EXT = {"txt", "md", "log", "csv", "tsv", "json", "yaml", "yml", "ini", "cfg", "toml", "xml", "py", "ipynb",
            "js", "ts", "jsx", "tsx", "java", "c", "cpp", "h", "hpp", "cs", "go", "rs", "rb", "php", "sh", "ps1",
            "bat", "sql", "kt", "swift", "css", "srt", "vtt", "tex", "rtf"}
HTML_EXT = {"html", "htm"}
OFFICE_XML = {"docx": ("word/document.xml",), "pptx": ("ppt/slides/",), "xlsx": ("xl/sharedStrings.xml",),
              "odt": ("content.xml",), "odp": ("content.xml",), "ods": ("content.xml",)}


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()[:MAX_CHARS]


def _xml_text(data: bytes) -> str:
    t = data.decode("utf-8", errors="ignore")
    t = re.sub(r"</w:p>|</a:p>|<w:tab/>|<text:p[^>]*>", " \n", t)
    return html.unescape(re.sub(r"<[^>]+>", " ", t))


def office_text(path: Path, ext: str) -> str:
    parts = []
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if any(name == w or (w.endswith("/") and name.startswith(w) and name.endswith(".xml"))
                   for w in OFFICE_XML[ext]):
                parts.append(_xml_text(z.read(name)))
                if sum(map(len, parts)) > MAX_CHARS:
                    break
    return _clean(" ".join(parts))


def pdf_text(path: Path) -> str:
    try:
        import logging
        from pypdf import PdfReader
        logging.getLogger("pypdf").setLevel(logging.ERROR)       # broken PDFs are common; stay quiet
    except ImportError:
        return ""
    out = []
    reader = PdfReader(str(path))
    for page in reader.pages[:40]:
        out.append(page.extract_text() or "")
        if sum(map(len, out)) > MAX_CHARS:
            break
    return _clean(" ".join(out))


def notebook_text(path: Path) -> str:
    import json
    nb = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    return _clean(" ".join("".join(c.get("source", [])) for c in nb.get("cells", [])))


def extract(path: Path, max_bytes: int = 5_000_000) -> str:
    ext = path.suffix.lower().lstrip(".")
    try:
        if path.stat().st_size > max_bytes and ext not in ("pdf", "docx", "pptx", "xlsx"):
            return ""
        if ext == "ipynb":
            return notebook_text(path)
        if ext in TEXT_EXT:
            return _clean(path.read_text(encoding="utf-8", errors="ignore")[: MAX_CHARS * 2])
        if ext in HTML_EXT:
            t = path.read_text(encoding="utf-8", errors="ignore")[: MAX_CHARS * 4]
            t = re.sub(r"(?is)<(script|style).*?</\1>", " ", t)
            return _clean(html.unescape(re.sub(r"<[^>]+>", " ", t)))
        if ext in OFFICE_XML:
            return office_text(path, ext)
        if ext == "pdf":
            return pdf_text(path)
    except Exception:              # unreadable, locked, corrupt: findable by name only
        return ""
    return ""
