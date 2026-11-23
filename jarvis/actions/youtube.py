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
