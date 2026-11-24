"""File types: single extensions (ext=pdf) and categories (ext=video means mp4, mkv, avi, mov, ...).

One definition shared by the executor (which files match), the rules and the model's safety repair
(which words mean which type), the persona (how to say it) and the training data.
"""
from __future__ import annotations

import re

CATEGORIES: dict[str, set[str]] = {
    "video": {"mp4", "mkv", "avi", "mov", "wmv", "webm", "flv", "m4v", "3gp", "mpg", "mpeg", "ts", "m2ts", "vob",
              "ogv", "mts"},
    "image": {"jpg", "jpeg", "png", "gif", "bmp", "webp", "heic", "heif", "tif", "tiff", "svg", "ico", "avif", "raw",
              "cr2", "nef", "arw", "dng"},
    "audio": {"mp3", "wav", "flac", "aac", "m4a", "ogg", "opus", "wma", "aiff", "aif", "alac", "mid", "midi", "amr"},
    "document": {"pdf", "doc", "docx", "txt", "rtf", "odt", "xls", "xlsx", "csv", "ods", "ppt", "pptx", "odp", "md",
                 "epub", "pages", "numbers", "key"},
    "archive": {"zip", "rar", "7z", "tar", "gz", "tgz", "bz2", "xz", "iso", "cab"},
    "installer": {"exe", "msi", "msix", "appx", "appxbundle", "dmg", "pkg", "apk", "deb", "rpm"},
    "code": {"py", "ipynb", "js", "ts", "jsx", "tsx", "java", "c", "cpp", "h", "hpp", "cs", "go", "rs", "rb", "php",
             "html", "css", "json", "yaml", "yml", "sh", "ps1", "bat", "sql", "kt", "swift"},
}

# What people call each category (plural forms; "video" also matches "video files")
CATEGORY_WORDS: dict[str, list[str]] = {
    "video": ["videos", "video files", "video", "movies", "movie files", "clips", "video clips", "recordings",
              "films", "vids"],
    "image": ["images", "image files", "photos", "photo files", "pics", "pictures", "picture files", "wallpapers",
              "imgs", "tasveerein", "photos aur images"],
    "audio": ["audio files", "audio", "songs", "music files", "sound files", "tracks", "voice notes", "gaane"],
    "document": ["documents", "docs", "document files", "office files", "papers"],
    "archive": ["archives", "compressed files", "zipped files", "archive files"],
    "installer": ["installers", "setup files", "setups", "installer files", "app installers"],
    "code": ["code files", "source files", "scripts", "source code"],
}
# The noun for replies: "Moved 12 videos"
NOUN = {"video": "video", "image": "image", "audio": "audio file", "document": "document", "archive": "archive",
        "installer": "installer", "code": "code file"}


def matches(suffix: str, ext: str | None) -> bool:
    """suffix like '.MP4' against ext like 'mp4' or 'video'."""
    if not ext:
        return True
    s = suffix.lower().lstrip(".")
    e = ext.lower().lstrip(".")
    return s in CATEGORIES[e] if e in CATEGORIES else s == e


_WORD_TO_CAT = sorted(((w, cat) for cat, ws in CATEGORY_WORDS.items() for w in ws), key=lambda x: -len(x[0]))
# words that are folders when they follow "to/into/from/in" or "folder"
_DEST = re.compile(r"\b(?:to|into|inside|under|in to)\b\s+.*$", re.I)
_SRC = re.compile(r"\b(?:from|in|se|me|mein)\s+(?:the\s+|my\s+)?[\w:\\/.-]+(?:\s+folder)?", re.I)


def object_words(text: str) -> str:
    """The part of a file command that names WHAT to act on: 'move all videos in downloads to videos'
    -> 'move all videos'. Destination and source clauses are dropped, so a folder called Videos
    doesn't count as asking for videos."""
    t = _DEST.sub(" ", text)
    t = _SRC.sub(" ", t)
    t = re.sub(r"\b\w+\s+(?:folder|directory|dir)\b", " ", t, flags=re.I)     # "videos folder" is a place
    return t


def category_in(text: str) -> str | None:
    """The category the user asked for, if any ('move all video files from downloads' -> 'video')."""
    obj = " " + re.sub(r"[^\w ]", " ", object_words(text).lower()) + " "
    for w, cat in _WORD_TO_CAT:
        if f" {w} " in obj:
            return cat
    return None
