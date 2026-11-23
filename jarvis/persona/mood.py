"""Mood: "I'm feeling low", "mood kharab hai kuch gaana lagao", "play something to cheer me up".

Jarvis reads how you feel, answers kindly, and plays music that fits: a feel-good playlist when
you're low, something calm when you're stressed. If you only say how you feel, it offers music
and plays it when you say yes. Words about ending your life or self-harm never get a playlist:
they get a caring reply and a helpline number.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# mood -> (English playlist, Hindi playlist)
PLAYLISTS = {
    "low": ("feel good happy songs", "feel good hindi songs"),
    "stressed": ("calm relaxing music", "soothing hindi songs"),
    "tired": ("soft chill acoustic songs", "chill hindi songs"),
    "bored": ("upbeat party songs", "bollywood party songs"),
    "angry": ("calming lofi music", "hindi lofi chill songs"),
    "lonely": ("comforting feel good songs", "feel good hindi songs"),
    "happy": ("happy upbeat songs", "happy hindi songs"),
    "focus": ("deep focus music", "deep focus music"),
    "sleep": ("calm sleep music", "soft hindi lullabies"),
}

_I = r"(?:i(?:'m| am)?|i'?ve been|im|me|main|mai|mera|my|mujhe|feeling|feel|felt|bahut|thoda|kaafi|so|really|very|a bit|kinda)"
_FEELINGS: list[tuple[str, str]] = [
    ("low", r"sad|low|down(?! to\b| for\b)|upset|unhappy|depressed|heart ?broken|broken|hurt|blue|miserable|not (?:feeling )?(?:good|great|okay|ok|well)"
            r"|bad day|rough day|terrible day|worst day|crying|dukhi|udaas|udas|dil toota|mood (?:is )?(?:bad|off|low|kharab|down|not good)"
            r"|bad mood|mood kharab|mood off|mood thik nahi|mood theek nahi|man nahi lag"),
    ("stressed", r"stressed|stress|anxious|anxiety|tensed?|tension|nervous|overwhelmed|panicking|worried|pressure|pareshaan|pareshan"),
    ("tired", r"tired|exhausted|drained|worn out|burnt out|burned out|sleepy|thak gaya|thak gayi|thaka hua|thaki hui|neend"),
    ("bored", r"bored|boring|bore ho raha|bore ho rahi|nothing to do"),
    ("angry", r"angry|frustrated|irritated|annoyed|pissed|furious|mad at|gussa|chidh"),
    ("lonely", r"lonely|alone|akela|akeli|missing (?:someone|her|him|home|my)"),
    ("happy", r"happy|great|awesome|amazing|excited|good mood|khush|mast|celebrat"),
    ("focus", r"need to focus|want to focus|can'?t focus|cannot focus|concentrate|studying|padhai|deep work"),
    ("sleep", r"can'?t sleep|cannot sleep|need to sleep|help me sleep|neend nahi"),
]
_FEEL = [(m, re.compile(rf"\b(?:{_I}\b.{{0,25}}?\b)?(?:{p})\b", re.I)) for m, p in _FEELINGS]
_SELF = re.compile(r"\b(i|i'm|im|i am|i've|my|me|main|mai|mera|meri|mujhe|feeling|feel|mood)\b", re.I)

# Asking for music to change the mood, with or without saying the feeling
_LIFT = re.compile(r"\b(cheer me up|lift (?:up )?my mood|uplift|mood (?:fresh|thik|theek|better|accha|acha|lift|change|boost)"
                   r"|make me (?:feel )?(?:better|happy|smile)|feel better|mood ?booster|pick me up|brighten my day"
                   r"|good vibes|happy vibes|relax me|calm me down|help me relax|help me (?:focus|sleep))\b", re.I)
_MUSIC = re.compile(r"\b(music|songs?|gaan[ae]|gana|playlist|tunes?|track|play|put on|chalao|bajao|lagao|sunao|baja do|chala do"
                    r"|laga do|something|kuch)\b", re.I)
_HINDI = re.compile(r"\b(hai|hoon|hu|kuch|gaana|gana|yaar|kharab|lagao|bajao|chalao|sunao|mera|mujhe|mai|main|bahut|thoda"
                    r"|accha|acha|dukhi|udaas|udas|akela|gussa|thak|hindi|bollywood)\b", re.I)
_LIFT_MOOD = [("stressed", re.compile(r"relax|calm", re.I)), ("focus", re.compile(r"focus", re.I)),
              ("sleep", re.compile(r"sleep", re.I))]

# Never answered with a playlist
_CRISIS = re.compile(r"\b(kill myself|end (?:my|it all)|want to die|wanna die|don'?t want to (?:live|be alive)|suicid\w*"
                     r"|self[- ]?harm|hurt myself|cut myself|no reason to live|better off dead|marna chahta|marna chahti"
                     r"|mar jaana chahta|mar jaana chahti|jeena nahi|zindagi khatam)\b", re.I)

YES = re.compile(r"^(yes|yeah|yep|yup|sure|ok(?:ay)?|please|haan|ha|han|haan ji|ji|play it|go ahead|do it|chalao|lagao|why not)"
                 r"(?: please| pls| yaar| jarvis)?[.! ]*$", re.I)


@dataclass
class Mood:
    kind: str                 # low | stressed | ... | crisis
    wants_music: bool
    hindi: bool

    @property
    def playlist(self) -> str:
        en, hi = PLAYLISTS.get(self.kind, PLAYLISTS["low"])
        return hi if self.hindi else en

    @property
    def program(self) -> str:
        return f'play_media(mode=playlist, query="{self.playlist}", site=youtube)'


def detect(text: str) -> Mood | None:
    t = text.strip()
    if _CRISIS.search(t):
        return Mood("crisis", False, bool(_HINDI.search(t)))
    hindi = bool(_HINDI.search(t))
    lift = _LIFT.search(t)
    kind = None
    if _SELF.search(t):
        for m, pat in _FEEL:
            if pat.search(t):
                kind = m
                break
    if kind is None and lift:
        kind = next((m for m, p in _LIFT_MOOD if p.search(lift.group(0))), "low")
    if kind is None:
        return None
    wants = bool(lift) or bool(_MUSIC.search(t))
    # "play sad songs" is a song request, not a feeling: only treat it as mood if the user talks about themselves
    if not lift and not re.search(r"\b(i|i'm|im|i am|i've|my|me|main|mai|mera|meri|mujhe|feeling|mood)\b", t, re.I):
        return None
    return Mood(kind, wants, hindi)


def _a(p) -> str:
    return f", {p.address}" if p.address and p.tone == "butler" else ""


_EMPATHY = {
    "low": {"butler": ["I'm sorry you're feeling low{a}.", "Sorry to hear that{a}. Rough days happen."],
            "friendly": ["Aw, sorry you're feeling down.", "That sucks, I'm sorry."], "minimal": ["Sorry to hear that."]},
    "stressed": {"butler": ["That sounds like a lot{a}. Let's slow things down.", "Deep breath{a}. One thing at a time."],
                 "friendly": ["Sounds stressful. Let's take it down a notch."], "minimal": ["Understood."]},
    "tired": {"butler": ["You've earned a break{a}."], "friendly": ["Sounds like you need a breather."], "minimal": ["Noted."]},
    "bored": {"butler": ["We can fix that{a}."], "friendly": ["Let's fix that!"], "minimal": ["Noted."]},
    "angry": {"butler": ["Understandable{a}. Let's cool things down."], "friendly": ["Ugh, that's annoying. Let's cool off."],
              "minimal": ["Noted."]},
    "lonely": {"butler": ["I'm here{a}."], "friendly": ["Hey, I'm here."], "minimal": ["I'm here."]},
    "happy": {"butler": ["Splendid to hear{a}!"], "friendly": ["Love that!"], "minimal": ["Nice."]},
    "focus": {"butler": ["Let's get you in the zone{a}."], "friendly": ["Focus mode, on it."], "minimal": ["Focus mode."]},
    "sleep": {"butler": ["Let's help you wind down{a}."], "friendly": ["Let's get you some rest."], "minimal": ["Wind-down."]},
}
_PLAYING = {
    "butler": ["I've put on some {what} for you.", "Here's some {what}. I hope it helps."],
    "friendly": ["Here's some {what} for you.", "Putting on some {what}."],
    "minimal": ["Playing {what}."],
}
_WHAT = {"low": "feel-good music", "stressed": "calm music", "tired": "soft, easy music", "bored": "upbeat music",
         "angry": "calming music", "lonely": "comforting music", "happy": "happy music to match", "focus": "focus music",
         "sleep": "calm sleep music"}
_OFFER = {
    "butler": "Shall I put on some {what}? Just say yes.",
    "friendly": "Want me to put on some {what}? Just say yes.",
    "minimal": "Play {what}? Say yes.",
}


def empathy(m: Mood, p) -> str:
    lines = _EMPATHY.get(m.kind, _EMPATHY["low"])
    return p.rng.choice(lines.get(p.tone, lines["butler"])).format(a=_a(p))


def playing_line(m: Mood, p) -> str:
    return p.rng.choice(_PLAYING.get(p.tone, _PLAYING["butler"])).format(what=_WHAT.get(m.kind, "music"))


def offer_line(m: Mood, p) -> str:
    return _OFFER.get(p.tone, _OFFER["butler"]).format(what=_WHAT.get(m.kind, "music"))


def crisis_reply(m: Mood, p) -> str:
    name = f" {p.address}" if p.address and p.tone == "butler" else ""
    return ("I'm really sorry you're feeling this way" + (f",{name}" if name else "") + ". You don't have to go through it alone. "
            "Please talk to someone right now: in India you can call Tele-MANAS on 14416 (free, 24x7), "
            "or reach out to a friend or family member you trust. If you're in immediate danger, call 112.")
