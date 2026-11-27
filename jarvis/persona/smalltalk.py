"""Small talk Jarvis can answer without a language model: greetings, time and date, thanks,
who it is, what it can do, a joke. Everything else out of scope gets a polite decline.
Open-ended conversation arrives with Jarvis-LM (phase 27)."""
from __future__ import annotations

import datetime as dt
import re

from .persona import Persona

JOKES = [
    "I tried to tidy your Downloads folder once. It tidied me instead.",
    "My favourite exercise is a quick undo. Very low impact.",
    "I asked the PDF how it was doing. It said it had too many pages to turn.",
    "I don't sleep. I just enter a low-power state of mild judgement about your desktop icons.",
    "I would tell you a joke about zip files, but it's a bit compressed.",
]

_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("time", re.compile(r"\b(what('?s| is) the time|what time is it|time (kya|kitna) hai|kitne baje|current time)\b")),
    ("date", re.compile(r"\b(what('?s| is) (the |today'?s )?date|what day is (it|today)|aaj (kya|kaunsi) date|today'?s date)\b")),
    ("joke", re.compile(r"\b(joke|make me laugh|kuch funny|hasao)\b")),
    ("how", re.compile(r"\b(how are you|how'?s it going|kaise ho|kya haal|how do you do)\b")),
    ("who", re.compile(r"\b(who are you|what('?s| is) your name|tum kaun ho|tumhara naam|aap kaun)\b")),
    ("can", re.compile(r"\b(what can you do|what do you do|your features|help me out|^help$|kya kya kar sakte|tum kya kar)\b")),
    ("thanks", re.compile(r"\b(thanks|thank you|thx|shukriya|dhanyavaad|ty)\b")),
    ("bye", re.compile(r"\b(bye|goodbye|good night|see you|shubh ratri|chalo bye)\b")),
    ("hello", re.compile(r"\b(hi|hello|hey|namaste|yo|good (morning|afternoon|evening)|sup)\b")),
]


def _part_of_day(now: dt.datetime) -> str:
    h = now.hour
    return "morning" if 4 <= h < 12 else "afternoon" if h < 17 else "evening"


def smalltalk(text: str, p: Persona, now: dt.datetime | None = None, only: tuple[str, ...] | None = None) -> str | None:
    t = text.lower().strip()
    now = now or dt.datetime.now()
    for intent, pat in _PATTERNS:
        if only is not None and intent not in only:
            continue
        if not pat.search(t):
            continue
        a = f", {p.address}" if p.address and p.tone == "butler" else ""
        if intent == "time":
            return f"It's {now.strftime('%I:%M %p').lstrip('0')}{a}."
        if intent == "date":
            return f"Today is {now.strftime('%A, %d %B %Y').replace(' 0', ' ')}{a}."
        if intent == "joke":
            return p.rng.choice(JOKES)
        if intent == "how":
            return {"butler": f"All systems running smoothly{a}. How can I help?",
                    "friendly": "Doing great! What can I do for you?", "minimal": "Running fine."}[p.tone]
        if intent == "who":
            return f"I'm {p.name}, your personal assistant. Every model I run on was trained from scratch on this PC."
        if intent == "can":
            return ("I can move, copy, list and delete files, open apps, websites and folders, play music and "
                    "videos (or music that fits your mood: try 'i'm feeling low'), search the web, set reminders, and control volume, lock and the display. Type /help "
                    "for commands like /fix and /status.")
        if intent == "thanks":
            return {"butler": f"You're most welcome{a}.", "friendly": "Anytime!", "minimal": "Welcome."}[p.tone]
        if intent == "bye":
            if "night" in t:
                return {"butler": f"Good night{a}.", "friendly": "Night! Sleep well.", "minimal": "Good night."}[p.tone]
            return {"butler": f"Until next time{a}.", "friendly": "See you!", "minimal": "Bye."}[p.tone]
        if intent == "hello":
            return {"butler": f"Good {_part_of_day(now)}{a}. What can I do for you?",
                    "friendly": f"Hey! Good {_part_of_day(now)}. What's up?", "minimal": "Hello."}[p.tone]
    return None
