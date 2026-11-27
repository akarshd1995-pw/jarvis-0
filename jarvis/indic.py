"""Indian scripts -> the Hinglish lettering the command model knows (phase 43).

    "डाउनलोड्स खोलो"   -> "downloads kholo"
    "गाना बजाओ"        -> "gana bajao"
    "कल शाम 5 बजे याद दिलाना" -> "kal sham 5 baje yad dilana"

Devanagari is written out letter by letter with Hindi's silent-vowel rules (करना -> karna, not karana). Bengali,
Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada and Malayalam share Devanagari's Unicode layout, so they're
mapped across first and read the same way (close enough for commands). Common English words typed in Devanagari
(डाउनलोड्स, फ़ोल्डर, यूट्यूब ...) come back as English. Text without Indian letters is returned unchanged.
"""
from __future__ import annotations

import re

BLOCKS = [(0x0980, 0x09FF), (0x0A00, 0x0A7F), (0x0A80, 0x0AFF), (0x0B00, 0x0B7F), (0x0B80, 0x0BFF),
          (0x0C00, 0x0C7F), (0x0C80, 0x0CFF), (0x0D00, 0x0D7F)]            # same layout as Devanagari 0900-097F
CONS = {"क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "n", "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "n",
        "ट": "t", "ठ": "th", "ड": "d", "ढ": "dh", "ण": "n", "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
        "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m", "य": "y", "र": "r", "ल": "l", "ळ": "l", "व": "v",
        "श": "sh", "ष": "sh", "स": "s", "ह": "h", "ऱ": "r", "ऴ": "zh", "क़": "k", "ख़": "kh", "ग़": "g", "ज़": "z",
        "ड़": "r", "ढ़": "rh", "फ़": "f", "य़": "y"}
NUKTA = {"क": "क़", "ख": "ख़", "ग": "ग़", "ज": "ज़", "ड": "ड़", "ढ": "ढ़", "फ": "फ़", "य": "य़"}
VOWELS = {"अ": "a", "आ": "aa", "इ": "i", "ई": "i", "उ": "u", "ऊ": "u", "ऋ": "ri", "ए": "e", "ऐ": "ai", "ओ": "o",
          "औ": "au", "ऑ": "o", "ऍ": "e", "ऎ": "e", "ऒ": "o"}
MATRAS = {"ा": "a", "ि": "i", "ी": "i", "ु": "u", "ू": "u", "ृ": "ri", "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
          "ॉ": "o", "ॅ": "e", "ॆ": "e", "ॊ": "o"}
VIRAMA, NUKTA_SIGN, ANUSVARA, CHANDRA, VISARGA = "्", "़", "ं", "ँ", "ः"
DIGITS = {chr(0x0966 + i): str(i) for i in range(10)}
# English words people type in Devanagari, as the command model knows them
LOAN = {"डाउनलोड": "download", "डाउनलोड्स": "downloads", "डॉक्यूमेंट": "document", "डॉक्यूमेंट्स": "documents",
        "डेस्कटॉप": "desktop", "फ़ोल्डर": "folder", "फोल्डर": "folder", "फ़ाइल": "file", "फाइल": "file",
        "फ़ाइलें": "files", "फाइलें": "files", "फ़ाइल्स": "files", "फाइल्स": "files", "वीडियो": "video",
        "वीडियोज़": "videos", "फोटो": "photo", "फ़ोटो": "photo", "फोटोज़": "photos", "म्यूज़िक": "music",
        "म्यूजिक": "music", "यूट्यूब": "youtube", "क्रोम": "chrome", "स्पॉटिफ़ाई": "spotify", "स्पॉटिफाई": "spotify",
        "पीडीएफ": "pdf", "स्क्रीनशॉट": "screenshot", "वॉल्यूम": "volume", "नोट": "note", "नोट्स": "notes",
        "टास्क": "task", "रिमाइंडर": "reminder", "ब्राइटनेस": "brightness", "वाईफ़ाई": "wifi", "वाईफाई": "wifi",
        "ब्लूटूथ": "bluetooth", "लॉक": "lock", "म्यूट": "mute", "मोड": "mode", "फोकस": "focus", "गेमिंग": "gaming",
        "कोडिंग": "coding", "ऑफ़िस": "office", "ऑफिस": "office", "ईमेल": "email", "प्लेलिस्ट": "playlist",
        "सॉन्ग": "song", "गेम": "game", "अपडेट": "update", "सेटिंग्स": "settings", "ओपन": "open", "प्ले": "play",
        "सर्च": "search", "मूव": "move", "कॉपी": "copy", "डिलीट": "delete", "आर्काइव": "archive", "ऐप": "app",
        "मिनट": "minute", "मिनट्स": "minutes", "बैंक": "bank", "फोन": "phone", "फ़ोन": "phone", "लोफी": "lofi",
        "लो-फ़ाई": "lofi", "सिंह": "singh", "रिपोर्ट": "report", "मीटिंग": "meeting", "पासवर्ड": "password",
        "लैपटॉप": "laptop", "कंप्यूटर": "computer", "स्क्रीन": "screen", "प्रोजेक्ट": "project", "टाइम": "time"}


def has_indic(text: str) -> bool:
    return any(0x0900 <= ord(ch) <= 0x0D7F for ch in text)


def _to_devanagari(text: str) -> str:
    out = []
    for ch in text:
        o = ord(ch)
        for lo, hi in BLOCKS:
            if lo <= o <= hi:
                ch = chr(o - lo + 0x0900)
                break
        out.append(ch)
    return "".join(out)


def _word(w: str) -> str:
    if w in LOAN:
        return LOAN[w]
    # syllables: [consonant letters, vowel text, vowel kind]  kind: inherent | matra | none (virama) | own
    syl: list[list] = []
    i = 0
    while i < len(w):
        ch = w[i]
        if ch in DIGITS:
            syl.append([DIGITS[ch], "", "own"])
        elif ch == NUKTA_SIGN and syl:
            pass
        elif ch in CONS or ch in NUKTA:
            c = ch
            if i + 1 < len(w) and w[i + 1] == NUKTA_SIGN:
                c = NUKTA.get(ch, ch)
                i += 1
            syl.append([CONS.get(c, CONS.get(ch, "")), "a", "inherent"])
        elif ch in MATRAS and syl:
            syl[-1][1], syl[-1][2] = MATRAS[ch], "matra"
        elif ch == VIRAMA and syl:
            syl[-1][1], syl[-1][2] = "", "none"
        elif ch in VOWELS:
            syl.append(["", VOWELS[ch], "own"])
        elif ch in (ANUSVARA, CHANDRA) and syl:
            syl[-1][1] += "n"
        elif ch == VISARGA and syl:
            syl[-1][1] += "h"
        elif ord(ch) < 0x0900 or ord(ch) > 0x0D7F:
            syl.append([ch, "", "own"])
        i += 1
    # Hindi drops the inherent 'a' at the end of a word, and in V-C(a)-C-V (कमरा -> kamra, करना -> karna)
    n = len(syl)
    for k in range(n):
        if syl[k][2] != "inherent":
            continue
        if k == n - 1 and n > 1:
            syl[k][1] = ""
        elif 0 < k < n - 1 and syl[k - 1][1] and syl[k + 1][2] in ("matra", "inherent") and syl[k + 1][0] \
                and not (k + 1 == n - 1 and syl[k + 1][2] == "inherent"):
            syl[k][1] = ""
    s = "".join(c + v for c, v, _ in syl)
    if w and w[0] == "व" and s.startswith("va"):
        s = "w" + s[1:]                                    # वाला -> wala
    return s


def to_latin(text: str) -> str:
    """Hinglish lettering for any Indian-script words in the text; everything else stays as it is."""
    if not has_indic(text):
        return text
    text = _to_devanagari(text).replace("।", ".")
    parts = re.split(r"(\s+|[.,!?;:()\"'])", text)
    return "".join(_word(p) if has_indic(p) else p for p in parts)
