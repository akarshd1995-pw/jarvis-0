"""YouTube lookups without an API key: reads YouTube's public search pages and channel RSS feeds.

  top_video(q)        first video in the search results            ("play dear god")
  latest_video(c)     newest upload of channel c, via its RSS feed  ("play yes theory latest video")
  playlist(q)         first playlist in the results                 ("play a lofi playlist")
  live_video(q)       first live stream in the results              ("play lofi girl live")
  channel_url(c)      the channel page                              ("open yes theory channel")

If YouTube changes its pages these return None and the caller falls back to the results page.
"""
from __future__ import annotations

import html as htmllib
import re
from urllib.parse import quote, quote_plus

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/130.0 Safari/537.36", "Accept-Language": "en-US,en;q=0.9"}
FILTERS = {"video": "EgIQAQ%3D%3D", "channel": "EgIQAg%3D%3D", "playlist": "EgIQAw%3D%3D", "live": "EgJAAQ%3D%3D"}

_VIDEO = re.compile(r'"videoRenderer":\{"videoId":"([\w-]{11})"')
_ANY_VIDEO = re.compile(r'"videoId":"([\w-]{11})"')
_CHANNEL = re.compile(r'"channelRenderer":\{"channelId":"(UC[\w-]{22})"')
_ANY_CHANNEL = re.compile(r'"channelId":"(UC[\w-]{22})"')
_EXTERNAL_ID = re.compile(r'"externalId":"(UC[\w-]{22})"|<meta itemprop="identifier" content="(UC[\w-]{22})"')
_PLAYLIST = re.compile(r'"playlistRenderer":\{"playlistId":"([\w-]+)"')
_LOCKUP_PLAYLIST = re.compile(r'"contentId":"(PL[\w-]+)"|"playlistId":"(PL[\w-]+)"')
_RSS_ENTRY = re.compile(r"<entry>.*?<yt:videoId>([\w-]{11})</yt:videoId>.*?<title>(.*?)</title>", re.S)
_RSS_CHANNEL_TITLE = re.compile(r"<feed.*?<title>(.*?)</title>", re.S)


def _get(url: str, fetch=None, timeout: float = 6.0) -> str | None:
    try:
        if fetch is not None:
            return fetch(url)
        import urllib.request
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
            return r.read().decode("utf-8", "ignore")
    except Exception:
        return None


def results_url(query: str, kind: str | None = None) -> str:
    url = f"https://www.youtube.com/results?search_query={quote_plus(query)}"
    return url + (f"&sp={FILTERS[kind]}" if kind else "")


_TITLE = re.compile(r'"title":\{"runs":\[\{"text":"((?:[^"\\]|\\.)*)"')
_OWNER = re.compile(r'"(?:ownerText|longBylineText)":\{"runs":\[\{"text":"((?:[^"\\]|\\.)*)"')
_FILLER = {"song", "songs", "video", "videos", "music", "the", "a", "an", "of", "by", "from", "full", "official",
           "lyrics", "lyric", "audio", "hd", "4k", "new", "latest", "and", "ft", "feat", "remix", "cover", "trailer",
           "track", "theme", "bgm", "live", "version", "best", "top", "hindi", "tamil", "telugu", "malayalam", "kannada",
           "how", "to", "make", "do", "does", "what", "why", "is", "are", "in", "on", "for", "with", "my", "your", "i",
           "explained", "tutorial", "guide", "easy", "minutes", "highlights", "episode", "clips", "watch", "play",
           "funny", "documentary", "review", "reviews", "unboxing", "recipe", "learn", "fix", "at", "home", "vs"}


def _unescape(s: str) -> str:
    try:
        s = s.encode("utf-8").decode("unicode_escape").encode("latin-1").decode("utf-8")
    except Exception:
        pass
    return htmllib.unescape(s)


def search_videos(query: str, fetch=None, limit: int = 6) -> list[tuple[str, str, str]] | None:
    """Top results as (video_id, title, channel). None = YouTube unreachable; [] = no videos."""
    page = _get(results_url(query), fetch)
    if page is None:
        return None
    out = []
    for chunk in page.split('"videoRenderer":{')[1:limit + 1]:
        m = re.match(r'"videoId":"([\w-]{11})"', chunk)
        if not m:
            continue
        head = chunk[:6000]
        t, o = _TITLE.search(head), _OWNER.search(head)
        out.append((m.group(1), _unescape(t.group(1)) if t else "", _unescape(o.group(1)) if o else ""))
    if not out:
        m = _ANY_VIDEO.search(page)
        if m:
            out.append((m.group(1), "", ""))
    return out


def _norm_words(s: str) -> list[str]:
    s = re.sub(r"(?<=\w)'s?\b", " ", s.lower())
    return [w for w in re.findall(r"[^\W_]+", s) if w not in _FILLER]


def _close(a: str, b: str) -> bool:
    """Same word, allowing spelling variants of transliterated names (pavazha ~ pavizha, kesariya ~ kesariyaa)."""
    from difflib import SequenceMatcher
    if a == b or a.rstrip("s") == b.rstrip("s") or (len(a) >= 4 and len(b) >= 4 and (a.startswith(b) or b.startswith(a))):
        return True
    return len(a) >= 4 and len(b) >= 4 and SequenceMatcher(None, a, b).ratio() >= 0.75


def match_score(query: str, title: str, channel: str = "") -> float:
    """Share of the query's meaningful words found (approximately) in the video's title or channel."""
    want = _norm_words(query)
    if not want:
        return 1.0
    have = _norm_words(title + " " + channel)
    return sum(any(_close(w, h) for h in have) for w in want) / len(want)


def _needed(query: str) -> float:
    """Names of 1-2 meaningful words must match fully; longer requests may miss a word ("tum hi ho by arijit singh")."""
    n = len(_norm_words(query))
    return 1.0 if n <= 2 else 0.5


def best_video(query: str, fetch=None, min_score: float | None = None):
    """('ok', vid, title) | ('nomatch', None, closest_title) | ('none', None, None) | ('offline', None, None).
    Picks the first result that actually matches what was asked, not blindly the first result."""
    results = search_videos(query, fetch)
    if results is None:
        return "offline", None, None
    if not results:
        return "none", None, None
    if not any(t for _, t, _ in results):         # YouTube changed its page: can't judge, trust the ranking
        return "ok", results[0][0], ""
    need = _needed(query) if min_score is None else min_score
    scored = [(match_score(query, t, c), -i, vid, t) for i, (vid, t, c) in enumerate(results)]
    score, _, vid, title = max(scored)                 # best match; earlier (higher-ranked) wins ties
    if score >= need:
        return "ok", vid, title
    return "nomatch", None, results[0][1]


def top_video(query: str, fetch=None) -> str | None:
    page = _get(results_url(query), fetch)
    if not page:
        return None
    m = _VIDEO.search(page) or _ANY_VIDEO.search(page)
    return m.group(1) if m else None


def find_channel(name: str, fetch=None) -> str | None:
    """Channel id (UC...) for a name or @handle."""
    name = name.strip()
    if name.startswith("@"):
        page = _get(f"https://www.youtube.com/{quote(name)}", fetch)
        m = _EXTERNAL_ID.search(page or "")
        if m:
            return m.group(1) or m.group(2)
    page = _get(results_url(name, "channel"), fetch)
    if not page:
        return None
    m = _CHANNEL.search(page) or _ANY_CHANNEL.search(page)
    return m.group(1) if m else None


def latest_video(channel: str, fetch=None) -> tuple[str, str, str] | None:
    """(video_id, video_title, channel_title) of the channel's newest upload."""
    cid = find_channel(channel, fetch)
    if not cid:
        return None
    feed = _get(f"https://www.youtube.com/feeds/videos.xml?channel_id={cid}", fetch)
    if not feed:
        return None
    m = _RSS_ENTRY.search(feed)
    if not m:
        return None
    ch = _RSS_CHANNEL_TITLE.search(feed)
    return m.group(1), htmllib.unescape(m.group(2)), htmllib.unescape(ch.group(1)) if ch else channel


def playlist(query: str, fetch=None) -> str | None:
    page = _get(results_url(query, "playlist"), fetch)
    if not page:
        return None
    m = _PLAYLIST.search(page)
    if m:
        return m.group(1)
    m = _LOCKUP_PLAYLIST.search(page)
    return (m.group(1) or m.group(2)) if m else None


def live_video(query: str, fetch=None) -> str | None:
    page = _get(results_url(query, "live"), fetch)
    if not page:
        return None
    m = _VIDEO.search(page) or _ANY_VIDEO.search(page)
    return m.group(1) if m else None


def channel_url(name: str, fetch=None) -> str | None:
    cid = find_channel(name, fetch)
    return f"https://www.youtube.com/channel/{cid}" if cid else None
