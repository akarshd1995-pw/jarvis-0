"""Bootstrap parser: a few regex rules so Jarvis is usable on day 1, before the model is trained.

Order of resolution in the daemon:
  1. input that is already valid DSL       -> run directly (power-user mode)
  2. trained intent model (if checkpoint)  -> model output, validated
  3. these rules                           -> fallback
"""
from __future__ import annotations

import re

from .. import filetypes

from ..dsl.grammar import DSLError, Program, parse

_FOLDER = r"(downloads|documents|desktop|pictures|music|videos|archive)"
_UNITS = {"min": "m", "minute": "m", "minutes": "m", "mins": "m", "m": "m",
          "hour": "h", "hours": "h", "hr": "h", "hrs": "h", "h": "h",
          "day": "d", "days": "d", "d": "d", "week": "w", "weeks": "w", "w": "w",
          "month": "month", "months": "month"}


def _dur(n: str, unit: str) -> str:
    u = _UNITS[unit.lower()]
    if u == "month":
        return f"{int(n) * 30}d"
    return f"{int(n)}{u}"


def _time(h: str, m: str | None, ampm: str | None) -> str:
    hour, minute = int(h), int(m or 0)
    if ampm:
        ampm = ampm.lower()
        if ampm == "pm" and hour < 12:
            hour += 12
        if ampm == "am" and hour == 12:
            hour = 0
    return f"{hour:02d}:{minute:02d}"


KNOWN_EXT = {"pdf", "docx", "doc", "xlsx", "xls", "csv", "pptx", "txt", "md", "png", "jpg", "jpeg",
             "gif", "webp", "svg", "mp4", "mkv", "mov", "mp3", "wav", "zip", "rar", "7z", "exe",
             "msi", "iso", "py", "json", "html"}


def _ext(text: str) -> str | None:
    for word in re.findall(r"\.?[A-Za-z0-9]+", text):
        w = word.lower().lstrip(".")
        if w in KNOWN_EXT:
            return w
        if w.endswith("s") and w[:-1] in KNOWN_EXT:
            return w[:-1]
    return None


def _filters(text: str) -> str:
    parts = []
    ext = _ext(filetypes.object_words(text)) or filetypes.category_in(text)
    if ext:
        parts.append(f"ext={ext}")
    m = re.search(r"older than (\d+) (\w+)", text)
    if m and m.group(2).lower() in _UNITS:
        parts.append(f"older_than={_dur(m.group(1), m.group(2))}")
    return ", ".join(parts)


_SITES = "youtube|google|amazon|flipkart|spotify|github|wikipedia|reddit"
_THIS = {"this", "this window", "the window", "it", "the current window", "this app", "current window"}
_APP_IDENTS = {"chrome", "edge", "vscode", "notepad", "explorer", "terminal", "spotify", "calculator", "settings"}


def _window(action: str, target: str) -> str:
    t = target.strip()
    if t.lower() in _THIS:
        return f"window(action={action})"
    key = t.lower().replace(" ", "")
    key = {"vscode": "vscode", "visualstudiocode": "vscode", "fileexplorer": "explorer"}.get(key, key)
    return f"window(action={action}, app={key})" if key in _APP_IDENTS else f'window(action={action}, app="{t}")'


# ---- phase 38: "this", "these", "here", "the second one"
_THIS = (r"(?:the selected (?:files?|ones?|items?)|selected files|this file|these files|that file|those files"
         r"|ye wali files|ye files|this|these|that|those|it|them|ise|inhe)")
_THESE_KIND = r"(?:these|those|the selected|selected|ye|yeh)\s+(?!(?:to|into|in|here|files?|ones?)\b)(\w+?)(?:\s+files?)?"
_ORDINAL = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4, "fifth": 5,
            "5th": 5, "sixth": 6, "6th": 6, "pehla": 1, "pehli": 1, "dusra": 2, "dusri": 2, "teesra": 3, "teesri": 3,
            "chautha": 4, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


def _dest_clause(rest: str) -> str | None:
    """'to archive' / 'into a new folder called taxes' / 'here' -> DSL args."""
    rest = rest.strip()
    m = re.match(r"^(?:to|into|in)\s+(?:a\s+)?new folder(?:\s+(?:called|named))?\s+['\"]?(.+?)['\"]?$", rest, re.I)
    if m:
        return f'new_folder="{m.group(1).strip()}"'
    if re.fullmatch(r"(?:to |into |in )?(?:here|this folder)", rest, re.I):
        return "dest=here"
    m = re.match(rf"^(?:to|into|in)\s+{_FOLDER}(?:\s+folder)?$|^{_FOLDER}(?:\s+folder)?\s+(?:me|mein)(?:\s+\w+)*$", rest, re.I)
    if m:
        return f"dest={(m.group(1) or m.group(2)).capitalize()}"
    m = re.match(r"^(?:to|into)\s+([a-zA-Z]:[/\\].+)$", rest)
    if m:
        return f'dest="{m.group(1).replace(chr(92), "/")}"'
    return None


def _ext_word(kind: str | None) -> str | None:
    """'pdfs' -> pdf, 'videos' -> video (a category), 'files' -> None."""
    if not kind:
        return None
    k = kind.lower().strip().rstrip("s")
    if k in ("file", "one", "item", "thing", "all", "the", "every", ""):
        return None
    from ..filetypes import CATEGORY_WORDS
    for cat, words in CATEGORY_WORDS.items():
        if k == cat or any(w.rstrip("s") == k for w in words):
            return cat
    return k if re.fullmatch(r"[a-z0-9]{2,5}", k) else None


def _ref_files(m, t) -> str | None:
    verb, kind, rest = m.group(1).lower(), m.group(2), m.group(3) or ""
    op = {"copy": "copy_files", "duplicate": "copy_files"}.get(verb, "delete_files" if verb in (
        "delete", "remove", "trash", "bin") else "move_files")
    args = ["src=this"]
    if _ext_word(kind):
        args.append(f"ext={_ext_word(kind)}")
    if op != "delete_files":
        d = _dest_clause(rest)
        if not d:
            return None
        args.append(d)
    elif rest.strip():
        return None
    return f"{op}({', '.join(args)})"


def _here_files(m, t) -> str | None:
    verb, kind, rest = m.group(1).lower(), m.group(2), m.group(3)
    op = {"list": "list_files", "show": "list_files", "copy": "copy_files", "delete": "delete_files",
          "remove": "delete_files"}.get(verb, "move_files")
    args = ["src=here"]
    if _ext_word(kind):
        args.append(f"ext={_ext_word(kind)}")
    if op in ("move_files", "copy_files"):
        d = _dest_clause(rest or "")
        if not d or d == "dest=here":
            return None
        args.append(d)
    return f"{op}({', '.join(args)})"


CONTEXT_RULES: list[tuple[re.Pattern, callable]] = [
    (re.compile(rf"^(?:(?:please|jarvis,?) )?(move|shift|put|send|copy|duplicate|delete|remove|trash|bin)\s+"
                rf"(?:{_THESE_KIND}|{_THIS})(?:\s+(.+))?$", re.I), _ref_files),
    (re.compile(rf"^(?:(?:please|jarvis,?) )?(list|show|move|shift|copy|delete|remove)(?: me)?\s+(?:all\s+)?(?:the\s+)?"
                rf"(?:(\w+)\s+)?(?:files?\s+)?(?:here|in this folder|from here)(?:\s+(.+))?$", re.I), _here_files),
    (re.compile(rf"^(?:{_THIS})\s+{_FOLDER}(?:\s+folder)?\s+(?:me|mein)\s+(daal do|daalo|rakh do|rakho|move karo|move kar do"
                rf"|bhej do|shift karo|shift kar do|copy karo|copy kar do)$", re.I),
     lambda m, t: f"{'copy' if 'copy' in m.group(2).lower() else 'move'}_files(src=this, dest={m.group(1).capitalize()})"),
    (re.compile(rf"^(?:{_THIS})\s+(?:delete|hata|mita)\s*(?:karo|kar do|do)$", re.I), lambda m, t: "delete_files(src=this)"),
    (re.compile(r"^open\s+(?:this|that|it|the selected file)$|^(?:ise|ye)\s+kholo$", re.I), lambda m, t: "open_result(n=1)"),
    (re.compile(rf"^(move|copy)\s+(?:all\s+)?(?:the\s+)?(\w+\s+)?(?:files?\s+)?from\s+{_FOLDER}(?:\s+folder)?\s+(?:to\s+)?here$", re.I),
     lambda m, t: f"{m.group(1).lower()}_files(src={m.group(3).capitalize()}, dest=here"
                  + (f", ext={_ext_word(m.group(2))}" if _ext_word(m.group(2)) else "") + ")"),
    (re.compile(rf"^(?:google|search|look up|search for)\s+(?:this|that|it|what i copied)(?:\s+on\s+({_SITES}))?$"
                rf"|^(?:ise|ye|isko)\s+(google|youtube)\s+(?:karo|pe dhundo|par dhundo)$", re.I),
     lambda m, t: f"search_web(query=this, site={(m.group(1) or m.group(2) or 'google').lower()})"),
    (re.compile(r"^open\s+(?:this|that|the copied|the)\s+link(?:\s+i\s+copied)?$|^open what i copied$", re.I),
     lambda m, t: "open_url(url=this)"),
    (re.compile(r"^(?:play|chalao|bajao)\s+(?:this|that|what i copied)(?:\s+(?:song|video|link))?$|^(?:ye|ise)\s+(?:chalao|bajao)$", re.I),
     lambda m, t: "play_media(query=this, site=youtube)"),
    (re.compile(r"^(?:directions|route|navigate|take me)\s+to\s+(?:this|that)(?:\s+(?:address|place|location))?$"
                r"|^how (?:long|far)(?: is it)? to (?:this|that)(?: address| place)?$", re.I),
     lambda m, t: "directions(destination=this)"),
    (re.compile(r"^(?:find|search)\s+(?:my\s+)?files\s+(?:about|for|on)\s+(?:this|that)$", re.I),
     lambda m, t: "search_memory(query=this)"),
    (re.compile(r"^remind me (?:about|of) (?:this|that) in (\d+) ?(m|min|mins|minutes?|h|hr|hrs|hours?)$", re.I),
     lambda m, t: f"remind(text=this, after={m.group(1)}{'h' if m.group(2).lower().startswith('h') else 'm'})"),
    (re.compile(r"^open\s+(?:the\s+)?(first|1st|second|2nd|third|3rd|fourth|4th|fifth|5th|sixth|6th)"
                r"(?:\s+(?:one|result|file|link|item))?$|^open\s+(?:result|number|no\.?|#)\s*(\d)$"
                r"|^(pehla|pehli|dusra|dusri|teesra|teesri|chautha)\s+(?:wala|wali|waala)?\s*(?:kholo|open karo)$", re.I),
     lambda m, t: f"open_result(n={_ORDINAL.get((m.group(1) or m.group(3) or '').lower()) or int(m.group(2))})"),
]

# ---- phase 30: notes and journal
_REFTEXT = re.compile(r"^(?:this|that|it|what i copied|the copied text|ise|isko|ye)$", re.I)
_WHEN = {"today": "today", "todays": "today", "aaj": "today", "yesterday": "yesterday", "yesterdays": "yesterday",
         "kal": "yesterday", "this week": "week", "this weeks": "week", "this month": "month", "this months": "month"}


def _note(text: str, kind: str = "note") -> str:
    text = text.strip().strip("\"'").strip()
    if _REFTEXT.match(text):
        return f"note(text=this{', kind=journal' if kind == 'journal' else ''})"
    esc = text.replace("\\", "\\\\").replace('"', '\\"')
    return f'note(text="{esc}"{", kind=journal" if kind == "journal" else ""})'


def _find(query: str | None = None, kind: str | None = None, when: str | None = None) -> str:
    args = []
    if query:
        q = re.sub(r"^(?:the|my|a)\s+", "", query.strip().rstrip("?").strip(), flags=re.I)
        args.append(f'query="{q}"')
    if kind == "journal":
        args.append("kind=journal")
    if when:
        args.append(f"when={_WHEN[when.lower().replace(chr(39), '')]}")
    return f"find_notes({', '.join(args)})"


NOTE_RULES: list[tuple[re.Pattern, callable]] = [
    (re.compile(r"^(?:add (?:this )?to my journal|journal(?: entry)?|dear diary|write in my journal|log in my journal"
                r"|diary me likho|journal me likho)(?:\s+that)?[:,]?\s+(.+)$", re.I), lambda m, t: _note(m.group(1), "journal")),
    (re.compile(r"^(?:note(?: down)?|jot(?: down)?|make a note(?: of)?|take a note(?: of)?|add a note|quick note"
                r"|note to self|write down|save a note(?: saying)?|note karo|likh lo)(?:\s+that)?[:,]?\s+(.+)$", re.I),
     lambda m, t: _note(m.group(1))),
    (re.compile(r"^(?:save|keep) (this|that|what i copied) as a note$|^(.+?)\s+note kar(?:o| lo| do| lena)$", re.I),
     lambda m, t: _note(m.group(1) or m.group(2))),
    (re.compile(r"^what did i (?:note|jot|write)(?: down)? (?:about|on|regarding) (.+?)\??$"
                r"|^(?:show|find|read|get|search)(?: me)? (?:my )?notes? (?:about|on|for|regarding) (.+)$"
                r"|^(.+?) ke baare me kya (?:note kiya|likha) tha\??$", re.I),
     lambda m, t: _find(m.group(1) or m.group(2) or m.group(3))),
    (re.compile(r"^(?:show|read|list|what are|give me)(?: me)? (?:my |all )?(today'?s|yesterday'?s|this week'?s|this month'?s)"
                r" (notes|journal(?: entries)?)$|^(aaj|kal) ke (notes|journal) dikhao$", re.I),
     lambda m, t: _find(kind="journal" if "journal" in (m.group(2) or m.group(4)).lower() else None,
                        when=(m.group(1) or m.group(3)).lower().replace("'", ""))),
    (re.compile(r"^(?:show|read|open)(?: me)? my journal(?: from| for)?(?: (today|yesterday|this week|this month))?$"
                r"|^what did i write in my journal(?: (today|yesterday|this week|this month))?\??$", re.I),
     lambda m, t: _find(kind="journal", when=m.group(1) or m.group(2))),
    (re.compile(r"^(?:show|list|read)(?: me)? (?:all )?my notes$|^(?:my )?notes dikhao$", re.I), lambda m, t: _find()),
    (re.compile(r"^(?:delete|remove|forget|erase) (?:the|my|that) note (?:about|on|for|regarding) (.+)$"
                r"|^(.+?) wala note hata do$", re.I),
     lambda m, t: f'delete_note(query="{re.sub(r"^(?:the|my) ", "", (m.group(1) or m.group(2)).strip(), flags=re.I)}")'),
]

# ---- phase 31: workspaces ("dark/light/focus/gaming/airplane/silent mode" are other things, never workspaces)
_NOT_WS = (r"(?!(?:turn|switch|enable|disable|put|set)\b)"
           r"(?!.*\b(?:dark|light|night|focus|gaming|game|airplane|aeroplane|flight|silent|battery|power|safe|sleep"
           r"|dnd|do not disturb|saver|tablet|eco|performance|incognito|private|reading)\b)")
_WSN = rf"{_NOT_WS}([a-z0-9][a-z0-9 _-]{{0,30}}?)"


def _ws(action: str, name: str | None = None) -> str:
    return f'workspace(action={action}' + (f', name="{name.strip()}"' if name else "") + ")"


WORKSPACE_RULES: list[tuple[re.Pattern, callable]] = [
    (re.compile(rf"^(?:save|remember|store|keep) (?:this|the current|my current|these|my)(?: window)? (?:layout|windows|setup|workspace|arrangement)"
                rf" (?:as|for|called|named) (?:my |the )?{_WSN}(?: mode| workspace| layout)?$"
                rf"|^ye layout {_WSN} naam se save karo$", re.I),
     lambda m, t: _ws("save", m.group(1) or m.group(2))),
    (re.compile(rf"^(?:close|end|stop|exit|quit|shut down) (?:my |the )?{_WSN} (?:mode|workspace|setup)$"
                rf"|^{_WSN} mode band karo$|^i'?m done with (?:my )?{_WSN}(?: mode| workspace)?$", re.I),
     lambda m, t: _ws("close", m.group(1) or m.group(2) or m.group(3))),
    (re.compile(rf"^(?:delete|remove|forget) (?:my |the )?{_WSN} (?:workspace|layout|mode)$", re.I),
     lambda m, t: _ws("delete", m.group(1))),
    (re.compile(r"^(?:list|show)(?: me)? (?:my |all )?(?:workspaces|layouts|modes)$|^what (?:workspaces|layouts) do i have\??$"
                r"|^(?:my )?workspaces dikhao$", re.I), lambda m, t: _ws("list")),
    (re.compile(rf"^(?:start|open|launch|load|switch to|enter|set up(?: for)?|go to|begin|activate) (?:my |the )?{_WSN}"
                rf" (?:mode|workspace|setup|layout)$|^{_WSN} mode(?: on| please)?$|^let'?s (?:start|do) {_WSN}(?: mode)?$"
                rf"|^{_WSN} mode (?:chalu|shuru|on) karo$|^{_WSN} (?:wala )?setup lagao$", re.I),
     lambda m, t: _ws("start", next(g for g in m.groups() if g))),
]

# ---- phase 32: focus & gaming
_DUR = (r"(?:(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h|ghante?)\b|(\d+)\s*(?:minutes?|mins?|m|minute)\b|(an hour|one hour|ek ghanta)"
        r"|(half an hour|aadha ghanta|30 min)|(two hours|do ghante))")


def _minutes(text: str) -> int | None:
    m = re.search(_DUR, text, re.I)
    if not m:
        return None
    h, mins, one, half, two = m.groups()
    if h:
        return round(float(h) * 60)
    if mins:
        return int(mins)
    return 60 if one else 30 if half else 120 if two else None


def _focus(text: str) -> str:
    t = text.lower()
    args = ["action=start"]
    n = _minutes(t)
    if n:
        args.append(f"minutes={n}")
    if "pomodoro" in t:
        args.append("mode=pomodoro")
    if re.search(r"without block|no block|don'?t block|bina block", t):
        args.append("block=no")
    elif re.search(r"\bblock|strict|no distractions|distraction[- ]free", t):
        args.append("block=yes")
    if re.search(r"no interruptions at all|completely quiet|total(?:ly)? quiet|not even reminders|koi disturb nahi", t):
        args.append("quiet=total")
    return f"focus({', '.join(args)})"


FOCUS_RULES: list[tuple[re.Pattern, callable]] = [
    (re.compile(r"^(?:stop|end|cancel|finish|exit|quit) (?:the |my )?(?:focus(?: mode| session)?|pomodoro|deep work)$"
                r"|^(?:focus|pomodoro)(?: mode)? off$|^i'?m done focusing$|^focus (?:mode )?band karo$", re.I),
     lambda m, t: "focus(action=stop)"),
    (re.compile(r"^how (?:long|much (?:time|focus time)) (?:is |do i have )?left(?: in (?:my )?(?:focus|pomodoro|this session))?\??$"
                r"|^focus status$|^how long have i been focusing\??$|^(?:focus ka )?kitna time bacha hai\??$", re.I),
     lambda m, t: "focus(action=status)"),
    (re.compile(r"^(?:(?:turn on|start|enable|switch to|activate) )?(?:gaming|game) mode(?: on| please)?$"
                r"|^i'?m (?:going to|gonna) (?:play|game)(?: now)?$|^gaming mode (?:on|chalu) karo$", re.I),
     lambda m, t: "gaming(action=on)"),
    (re.compile(r"^(?:(?:turn off|stop|end|disable|exit) )?(?:gaming|game) mode(?: off)?$|^(?:i'?m )?done gaming$"
                r"|^gaming mode band karo$", re.I),
     lambda m, t: "gaming(action=off)" if re.search(r"off|stop|end|disable|exit|done|band", t, re.I) else "gaming(action=on)"),
    (re.compile(r"^(?:(?:start|begin|turn on|enable) )?(?:a |my )?(?:focus(?: mode| session)?|pomodoro|deep work)\b.*$"
                r"|^(?:i (?:need|want) to|let me|help me) focus\b.*$|^focus (?:for|karna hai|karna|karo)\b.*$"
                r"|^.*\bfocus (?:karna hai|karo)$|^do not disturb for .+$", re.I),
     lambda m, t: _focus(t)),
]

# ---- phase 33: tasks
_EVERY = [(r"\bevery ?day\b|\bdaily\b|\broz\b", "daily"), (r"\bevery weekday\b|\bon weekdays\b|\bweekdays\b", "weekdays"),
          (r"\bevery week\b|\bweekly\b", "weekly"), (r"\bevery month\b|\bmonthly\b|\bhar mahine\b", "monthly")]
_DUE_TAIL = (r"(?:\s+(?:by|on|before|due|till|until|for)\s+(.+)|\s+(.+?)\s+tak"
             r"|\s+((?:day after )?tomorrow(?: \w+)?|today(?: \w+)?|tonight|kal(?: \w+)?|parso|next week|this weekend"
             r"|(?:next |this )?(?:mon|tues|wednes|thurs|fri|satur|sun)day(?: \w+)?|at \d.*|in \d+ (?:days?|weeks?)))$")


def _task_add(body: str) -> str:
    text, args = body.strip(), []
    for pat, every in _EVERY:
        if re.search(pat, text, re.I):
            text = re.sub(pat, " ", text, flags=re.I)
            args.append(f"every={every}")
            break
    if re.search(r"\b(?:important|urgent|high priority|asap|zaroori)\b", text, re.I):
        text = re.sub(r",?\s*\b(?:it'?s |its )?(?:important|urgent|high priority|asap|zaroori)\b", " ", text, flags=re.I)
        args.append("priority=high")
    text = re.sub(r"\s+", " ", text).strip(" ,.")
    m = re.search(_DUE_TAIL, text, re.I)
    due = None
    if m:
        due = next(g for g in m.groups() if g)
        text = text[: m.start()].strip(" ,.")
    esc = lambda s: s.replace("\\", "\\\\").replace('"', '\\"')        # noqa: E731
    return "task(" + ", ".join(["action=add", f'text="{esc(text)}"'] + ([f'due="{esc(due.strip())}"'] if due else []) + args) + ")"


def _task_move(what: str, when: str) -> str | None:
    """Only a real date makes it a reschedule: "move these to archive" is a file move."""
    from ..dates import parse_due
    if parse_due(when) is None:
        return None
    return f'task(action=move, text="{_task_ref(what)}", due="{when.strip()}")'


def _task_ref(text: str) -> str:
    return re.sub(r"^(?:the|my|that|this)\s+", "", re.sub(r"\s+(?:task|item|to-?do)$", "", text.strip(" ?.!"), flags=re.I),
                  flags=re.I)


TASK_RULES: list[tuple[re.Pattern, callable]] = [
    (re.compile(r"^what'?s (?:due )?today\??$|^what(?:'s| is) due today\??$|^what do i (?:have to|need to) do today\??$"
                r"|^(?:show|list) (?:my )?tasks? (?:for |due )?today$|^today'?s tasks$|^aaj kya karna hai\??$", re.I),
     lambda m, t: "task(action=list, when=today)"),
    (re.compile(r"^(?:what'?s|what is|show(?: me)?|list) (?:my )?(?:overdue(?: tasks)?)\??$|^what(?:'s| is) overdue\??$", re.I),
     lambda m, t: "task(action=list, when=overdue)"),
    (re.compile(r"^(?:show|list) (?:my )?tasks? (?:for |due )?this week$|^what'?s due this week\??$|^this week'?s tasks$", re.I),
     lambda m, t: "task(action=list, when=week)"),
    (re.compile(r"^what did i (?:finish|complete|get done) this week\??$|^(?:show )?(?:my )?(?:completed|finished) tasks$", re.I),
     lambda m, t: "task(action=list, when=done)"),
    (re.compile(r"^(?:show|list|read|open)(?: me)? (?:all )?my (?:tasks|to-?dos?|to-?do list|task list)$|^my (?:tasks|to-?do list)$"
                r"|^(?:mere )?tasks dikhao$", re.I),
     lambda m, t: "task(action=list)"),
    (re.compile(r"^(?:(?:i'?m |i am )?done with|(?:i )?finished|completed?|tick off|check off|mark) (.+?)(?: as (?:done|complete|finished))?$"
                r"|^(.+?) (?:is|are) (?:done|finished|complete)$|^(.+?) ho gaya$", re.I),
     lambda m, t: f'task(action=done, text="{_task_ref(m.group(1) or m.group(2) or m.group(3))}")'),
    (re.compile(r"^(?:move|postpone|push|reschedule|shift) (.+?) to (.+)$|^(.+?) ko (.+?) (?:pe|par) shift karo$", re.I),
     lambda m, t: _task_move(m.group(1) or m.group(3), m.group(2) or m.group(4))),
    (re.compile(r"^(?:delete|remove|cancel) (?:the )?(?:task|to-?do) (.+)$|^remove (.+?) from my (?:to-?do |task )?list$"
                r"|^(.+?) wala task hata do$", re.I),
     lambda m, t: f'task(action=delete, text="{_task_ref(m.group(1) or m.group(2) or m.group(3))}")'),
    (re.compile(r"^(?:add (?:a |the )?(?:task|to-?do)|new task|to-?do|add to my (?:to-?do |task )?list|put on my (?:to-?do )?list"
                r"|task)[:,]?\s+(.+)$|^add (.+?) to my (?:to-?do |task )?list$|^(.+?) karna hai(?: .+ tak)?$", re.I),
     lambda m, t: _task_add(m.group(1) or m.group(2) or (m.group(3) + (" " + re.search(r"karna hai (.+) tak$", t).group(1)
                                                                          + " tak" if re.search(r"karna hai (.+) tak$", t) else "")))),
]

# ---- phase 43: your words
def _q(s: str) -> str:
    return s.strip().strip("\"'").replace("\\", "\\\\").replace('"', '\\"')


VOCAB_RULES: list[tuple[re.Pattern, callable]] = [
    (re.compile(r"^open (?:my |the )?(.+?) (?:project|repo|repository)$|^(.+?) project kholo$", re.I),
     lambda m, t: f'open_folder(path="{_q(m.group(1) or m.group(2))}")'),
    (re.compile(r"^when i say ['\"]?(.+?)['\"]?,? i mean ['\"]?(.+?)['\"]?$"
                r"|^(?:remember|learn) that ['\"]?(.+?)['\"]? (?:means|is) ['\"]?(.+?)['\"]?$"
                r"|^['\"](.+?)['\"] means ['\"]?(.+?)['\"]?(?: from now on)?$"
                r"|^(?!(?:this|that|it|what|which|who|he|she|they|there|here|everything|nothing|life|love|that's)\b)"
                r"([a-z0-9][a-z0-9_-]{1,15}) means (.+?)(?: from now on)?$"
                r"|^jab main ['\"]?(.+?)['\"]? bolun(?: to)? (?:uska matlab )?['\"]?(.+?)['\"]?(?: hai| samajhna)?$", re.I),
     lambda m, t: (lambda g: f'vocab(action=teach, word="{_q(g[0])}", means="{_q(g[1])}")')(
         [x for x in m.groups() if x is not None][:2])),
    (re.compile(r"^(?:forget|unlearn|stop using) (?:the (?:word|nickname) )?['\"]?(.+?)['\"]?(?: as a nickname)?$"
                r"|^['\"]?(.+?)['\"]? (?:is not|isn't) a nickname$", re.I),
     lambda m, t: None if re.search(r"\bnote|\btask|\blayout|\bworkspace", t, re.I)
     else f'vocab(action=forget, word="{_q(m.group(1) or m.group(2))}")'),
    (re.compile(r"^(?:show|list) (?:me )?my (?:words|nicknames|vocabulary|shortcuts)$|^what (?:words|nicknames) do you know\??$", re.I),
     lambda m, t: "vocab(action=list)"),
]

# ---- phase 34: wellbeing
def _limit(app: str, text: str) -> str:
    from ..wellbeing import parse_app_minutes
    mins = parse_app_minutes(text) or 60
    return f'screen_time(action=limit, app="{_q(app)}", minutes={mins})'


def _breaks(text: str) -> str:
    t = text.lower()
    kind = "eyes" if re.search(r"\beye", t) else "stretch" if "stretch" in t else "all"
    if re.search(r"snooze|pause|hold|mute", t):
        from ..wellbeing import parse_app_minutes
        return f"breaks(action=snooze, minutes={parse_app_minutes(t) or 60})"
    on = not re.search(r"\b(?:off|stop|disable|no more|don'?t|band|no)\b", t)
    return f"breaks(action={'on' if on else 'off'}" + (f", kind={kind}" if kind != "all" else "") + ")"


WELL_RULES: list[tuple[re.Pattern, callable]] = [
    (re.compile(r"^(?:limit|cap) (.+?) to (.+?)(?: a| per| every)? ?day$|^(?:only|just) (.+?) of (.+?) (?:a|per) day$"
                r"|^set an? (.+?) (?:daily )?limit (?:on|for) (.+)$|^(.+?) (?:daily )?limit (\d+.*)$", re.I),
     lambda m, t: _limit(m.group(1), m.group(2)) if m.group(1) else _limit(m.group(4), m.group(3)) if m.group(3)
     else _limit(m.group(6), m.group(5)) if m.group(5) else _limit(m.group(7), m.group(8))),
    (re.compile(r"^(?:remove|delete|clear|drop) the (.+?) limit$|^(?:no|remove the) limit (?:on|for) (.+)$"
                r"|^unlimit (.+)$", re.I),
     lambda m, t: f'screen_time(action=unlimit, app="{_q(m.group(1) or m.group(2) or m.group(3))}")'),
    (re.compile(r"^how (?:much|long) (?:time )?(?:have i (?:spent|been) on |did i spend on |of )?(.+?) (?:have i done |did i watch )?today\??$", re.I),
     lambda m, t: "screen_time(action=report)" if re.fullmatch(r"(?:screen ?time|time|screen)", m.group(1), re.I)
     else f'screen_time(action=report, app="{_q(m.group(1))}")'),
    (re.compile(r"^(?:(?:show )?(?:my )?screen ?time(?: today)?|how much screen ?time (?:today|did i have today)|aaj kitna screen ?time hua)\??$", re.I),
     lambda m, t: "screen_time(action=report)"),
    (re.compile(r"^(?:turn|switch) (?:on|off) (?:the )?(?:eye[- ]?)?(?:break|stretch)(?: reminders?)?$"
                r"|^(?:stop|disable|enable|start|no more|resume) (?:the )?(?:eye[- ]?)?(?:break|stretch)(?:e?s)?(?: reminders?)?$"
                r"|^(?:snooze|pause|mute|hold) (?:the )?(?:eye[- ]?)?(?:break|stretch)(?:e?s)?(?: reminders?)?(?: for .+)?$"
                r"|^(?:break|stretch) reminders? (?:on|off|band karo|chalu karo)$|^no (?:eye )?break reminders$", re.I),
     lambda m, t: _breaks(t)),
    (re.compile(r"^(?:when was my last break|break (?:reminders? )?status|how long have i been at the screen)\??$", re.I),
     lambda m, t: "breaks(action=status)"),
]

# ---- phase 37: system care
_PC = r"(?: (?:my |the )?(?:pc|computer|laptop|system|disk|drive|c drive|c:))?"
CARE_RULES: list[tuple[re.Pattern, callable]] = [
    (re.compile(r"^(?:how much (?:free )?(?:disk )?space (?:do i have|is left|is there)(?: left)?|(?:check |show )?(?:my )?disk space"
                r"|what(?:'s| is) (?:taking|using|eating) (?:up )?(?:all )?(?:my |the )?(?:disk )?space|why is my (?:c )?drive full"
                r"|disk (?:usage|report)|storage (?:report|check)|disk (?:me|mein) kitni jagah hai|kitni jagah bachi hai)\??$", re.I),
     lambda m, t: "care(action=disk)"),
    (re.compile(r"^(?:clear|clean|delete|remove|empty|wipe) (?:(?:out|up) )?(?:the |my |all )?(?:old )?(?:temp|temporary)(?: files| folder)?$"
                r"|^(?:temp|temporary) files (?:saaf karo|hata do|clear karo|delete karo)$", re.I),
     lambda m, t: "care(action=cleanup, what=temp)"),
    (re.compile(r"^(?:clear|clean up|delete|remove|get rid of) (?:the |my |all )?(?:old )?(?:installers|setup files|installer files)(?: (?:in|from) (?:my )?downloads)?$", re.I),
     lambda m, t: "care(action=cleanup, what=installers)"),
    (re.compile(r"^(?:clear|clean up|delete|remove|get rid of|find) (?:the |my |all )?duplicate(?:s| files| downloads)(?: (?:in|from) (?:my )?downloads)?$", re.I),
     lambda m, t: "care(action=cleanup, what=duplicates)"),
    (re.compile(rf"^(?:free up (?:some )?(?:disk )?space{_PC}|clean up{_PC}|clean{_PC} up|(?:pc|computer|laptop) saaf karo|jagah khali karo)$", re.I),
     lambda m, t: "care(action=cleanup, what=all)"),
    (re.compile(r"^(?:empty|clear) (?:your|jarvis(?:'s)?|the jarvis) trash$", re.I),
     lambda m, t: "care(action=empty_trash)"),
    (re.compile(r"^(?:(?:show |list )?(?:my )?startup (?:apps|programs|items)|what (?:apps |programs )?(?:start|open|run|launch)s? (?:with|when) (?:windows|my pc|i log in|the pc starts)"
                r"|what (?:runs|starts) at startup)\??$", re.I),
     lambda m, t: "care(action=startup_list)"),
    (re.compile(r"^(?:turn off|disable|stop|remove) (.+?) (?:at|on|from) (?:startup|start ?up|boot|login)$"
                r"|^(?:stop|don'?t let) (.+?) (?:from )?(?:starting|opening|launching) (?:with windows|at startup|when (?:my pc|windows|the pc) starts)$"
                r"|^(?:don'?t|do not) (?:start|open|launch) (.+?) (?:with windows|at startup)$|^startup se (.+?) hata do$", re.I),
     lambda m, t: f'care(action=startup_off, name="{_q(m.group(1) or m.group(2) or m.group(3) or m.group(4))}")'),
    (re.compile(r"^(?:turn on|enable) (.+?) (?:at|on) (?:startup|start ?up|boot|login)$"
                r"|^(?:start|open|launch) (.+?) (?:with windows|at startup|when (?:my pc|windows) starts)$", re.I),
     lambda m, t: f'care(action=startup_on, name="{_q(m.group(1) or m.group(2))}")'),
    (re.compile(r"^(?:(?:show )?(?:my )?gpu (?:temp|temperature)s?(?: history| this week| over the week)?|how hot (?:did|does|has) my (?:gpu|graphics card) (?:get|run|been)(?: this week| lately)?"
                r"|what was my (?:highest|max) gpu (?:temp|temperature)|gpu kitna garam hua)\??$", re.I),
     lambda m, t: "care(action=temps)"),
]

RULES: list[tuple[re.Pattern, callable]] = CARE_RULES + WELL_RULES + VOCAB_RULES + FOCUS_RULES + TASK_RULES + NOTE_RULES + WORKSPACE_RULES + CONTEXT_RULES + [
    (re.compile(r"^undo\b", re.I), lambda m, t: "undo()"),
    (re.compile(r"^(?!.*\btraining\b)(?:(?:hey )?jarvis,? )?(?:what(?:'s| is) the |show (?:me )?the |give me (?:the )?|get (?:me )?(?:the )?)?"
                r"(?:(?:fastest|quickest|best|shortest) )?(?:route|way|directions?|road)(?: please)? (?:from .+ )?to .+$"
                r"|^(?!.*\btraining\b)(?:(?:hey )?jarvis,? )?how (?:long|much time|far)\b.*\b(?:to|is) .+$"
                r"|^(?:navigate|take me|guide me) to .+$|^.+ (?:tak|jaane me|pahunchne me) kitna (?:time|samay|der) lagega\??$"
                r"|^.+ kaise (?:jaun|jaaun|jayen|jaye|pahunchu|pahunchun)\??$|^.+ kitna door hai\??$", re.I),
     lambda m, t: _directions(t)),
    (re.compile(r"^(?:(?:hey )?jarvis,? )?(?:run |do |start )?(?:a |the |your )?(?:health ?check|self[- ]?check|diagnostics?"
                r"|system check)(?: please| pls| now)?$|^(?:are you|is everything) (?:ok|okay|working|healthy|fine)\??$"
                r"|^check yourself$|^(?:apna )?health check karo$", re.I),
     lambda m, t: "maintenance(action=health_check)"),
    (re.compile(r"^(?:(?:hey )?jarvis,? )?(?:back ?up|take a backup of|make a backup of|save a backup of) (?:yourself|jarvis"
                r"|your (?:data|settings|brain|models))(?: now| please)?$|^(?:make|take|create) a backup(?: of jarvis)?(?: now)?$"
                r"|^jarvis ka backup (?:lo|bana do|karo)$", re.I),
     lambda m, t: "maintenance(action=backup)"),
    # briefings (phase 24)
    (re.compile(r"^(?:(?:hey |ok )?jarvis,? )?(?:good morning|gm)(?:,? jarvis)?[!. ]*$"
                r"|^(?:(?:hey )?jarvis,? )?(?:(?:give|read) me |start |run |play )?(?:my |the |a |today'?s )?morning (?:briefing|brief|summary|update|report)(?: please| pls)?$"
                r"|^(?:what'?s|what is|whats) (?:on )?(?:my |the )?(?:day|agenda|plan|schedule)(?: (?:look )?like)?(?: today)?\??$"
                r"|^what(?:'s| is| do i have) (?:on |planned )?(?:for )?today\??$"
                r"|^aaj (?:ka (?:plan|din) kya hai|kya (?:hai|plan hai)|ke liye kya hai)\??$", re.I),
     lambda m, t: "briefing(kind=morning)"),
    (re.compile(r"^(?:(?:hey )?jarvis,? )?(?:how was my day|how did my day go|summari[sz]e my day|wrap up (?:the|my) day"
                r"|(?:give me |read me )?(?:my |the |an? )?(?:evening|end of (?:the )?day|eod|daily|day) (?:briefing|brief|summary|report|recap|wrap ?up)"
                r"|what did i (?:do|work on) today|aaj (?:maine )?kya kiya|aaj ka (?:summary|hisaab)(?: do| batao)?|day recap)(?: please| pls)?\??$", re.I),
     lambda m, t: "briefing(kind=evening)"),
    (re.compile(r"^(?:(?:hey )?jarvis,? )?(?:give me |read me |start |run )?(?:my |the |a )?(?:briefing|daily brief|rundown|status update|sitrep)(?: please| pls)?\??$|^(?:(?:hey )?jarvis,? )?(?:brief me|catch me up|update me)(?: please| pls)?\??$", re.I),
     lambda m, t: "briefing()"),
    # richer play modes first: latest upload of a channel, playlist, live, open a channel
    (re.compile(r"^(?:play|watch|put on|open) (?:the )?(?:latest|newest|new|recent|most recent|last) (?:video|upload)s? "
                r"(?:of|from|by|on) (?:the )?(.+?)(?: channel)?(?: on youtube)?$", re.I),
     lambda m, t: f'play_media(query="{m.group(1)}", mode=latest, site=youtube)'),
    (re.compile(r"^(?:play|watch|put on) (.+?)(?:'s|s'|')? (?:channel'?s? )?(?:latest|newest|new|recent|most recent|last) "
                r"(?:video|upload)s?(?: on youtube)?$", re.I),
     lambda m, t: f'play_media(query="{m.group(1)}", mode=latest, site=youtube)'),
    (re.compile(r"^(.+?) (?:ka|ki|ke) (?:latest|nayi|naya|new) (?:video|vid) (?:chalao|lagao|play karo|dikhao)$", re.I),
     lambda m, t: f'play_media(query="{m.group(1)}", mode=latest, site=youtube)'),
    (re.compile(r"^open (?:the )?(.+?) (?:youtube )?channel(?: on youtube)?$", re.I),
     lambda m, t: f'play_media(query="{m.group(1)}", mode=channel, site=youtube)'),
    (re.compile(r"^(?:play|put on) (?:a |an |some )?(.+?) (?:playlist|mix)(?: on youtube)?$", re.I),
     lambda m, t: f'play_media(query="{m.group(1)}", mode=playlist, site=youtube)'),
    (re.compile(r"^(?:play|watch) (.+?) live(?: ?stream)?(?: on youtube)?$", re.I),
     lambda m, t: f'play_media(query="{m.group(1)}", mode=live, site=youtube)'),
    (re.compile(r"^(?:open (?:youtube|spotify) and )?(?:play|put on|stream) (.+?) on (youtube|spotify)$", re.I),
     lambda m, t: f'play_media(query="{m.group(1)}", site={m.group(2).lower()})'),
    (re.compile(r"^open (youtube|spotify) and (?:play|put on|stream) (.+)$", re.I),
     lambda m, t: f'play_media(query="{m.group(2)}", site={m.group(1).lower()})'),
    (re.compile(rf"^(?:open (?:{_SITES}) and )?(?:search(?: for)?|find|look up) (.+?) on ({_SITES})$", re.I),
     lambda m, t: f'search_web(query="{m.group(1)}", site={m.group(2).lower()})'),
    (re.compile(rf"^open ({_SITES}) and (?:search(?: for)?|find|look up) (.+)$", re.I),
     lambda m, t: f'search_web(query="{m.group(2)}", site={m.group(1).lower()})'),
    (re.compile(r"^(?:youtube pe |yt pe )?(.+?) (?:song |gaana |gana |video )?(?:chalao|chala do|bajao|baja do|lagao|laga do|sunao|play karo|play kar do)$", re.I),
     lambda m, t: f'play_media(query="{m.group(1)}", site=youtube)'),
    (re.compile(r"^(?:(?:jarvis,? |hey jarvis,? )(?:can you |could you |please |pls )*|(?:can you |could you |please |pls )+)(?:play|put on) (?!.* on \w+$)(.+?)(?: please| pls)?\??$", re.I),
     lambda m, t: f'play_media(query="{m.group(1)}", site=youtube)'),
    (re.compile(r"^(?:play|put on) (?!.* on \w+$)(.+)$", re.I),
     lambda m, t: f'play_media(query="{m.group(1)}", site=youtube)'),
    (re.compile(r"^(?:google|search(?: google)? for) (.+)$", re.I),
     lambda m, t: f'search_web(query="{m.group(1)}", site=google)'),
    # ---- phase 23: training control
    (re.compile(r"^(?:(?:hey )?jarvis,? )?(?:improve yourself|learn from my (?:corrections|fixes)|get better|self[- ]?improve"
                r"|(?:fine[- ]?tune|retrain|update|improve) (?:the |your |my )?(?:command )?model(?: now)?"
                r"|khud ko improve karo|apne aap ko behtar karo)(?: please| pls| now)?$", re.I),
     lambda m, t: "training(action=improve)"),
    (re.compile(r"^(?:roll ?back|revert|undo) (?:the |your )?(?:model|model update|last model|update)"
                r"|^(?:go back|switch back) to the (?:previous|old|last) model|^purana model wapas (?:lao|lagao)", re.I),
     lambda m, t: "training(action=rollback)"),
    (re.compile(r"^(?:(?:hey )?jarvis,? )?(?:let'?s |i want to |can we |open (?:the )?)?(?:review|check|go through|label)"
                r" (?:my |the |this week'?s |your )?(?:commands|mistakes|doubts|unsure commands|weekly review|review)"
                r"(?: (?:you|that you) (?:weren'?t|were not) sure about)?(?: please| now)?$"
                r"|^(?:weekly review|show (?:me )?what you (?:got wrong|weren'?t sure about))$"
                r"|^(?:mere )?commands? review karo$|^review karte hain$", re.I),
     lambda m, t: "training(action=review)"),
    (re.compile(r"^(?:pause|hold) (?:the )?training$", re.I), lambda m, t: "training(action=pause)"),
    (re.compile(r"^(?:resume|continue|unpause) (?:the )?training$", re.I), lambda m, t: "training(action=resume)"),
    (re.compile(r"^(?:training status|how(?:'s| is) (?:the )?training(?: going)?|is (?:the )?training (?:done|finished)"
                r"|training kitna hua|training ka status)\??$", re.I), lambda m, t: "training(action=status)"),
    # ---- phase 5: system control (before the generic "open/start X" rule)
    (re.compile(r"^(?:take (?:a )?|grab (?:a )?)?screenshot(?: (?:of )?(?:the |my )?(?:whole )?screen)?$", re.I),
     lambda m, t: "screenshot(mode=full)"),
    (re.compile(r"^(?:snip|screenshot an area|take a partial screenshot|open the snipping tool)", re.I),
     lambda m, t: "screenshot(mode=region)"),
    (re.compile(r"^(start|stop) (?:the )?(?:screen )?recording$", re.I),
     lambda m, t: f"screen_record(action={m.group(1).lower()})"),
    (re.compile(r"^record (?:my |the )?screen$", re.I), lambda m, t: "screen_record(action=start)"),
    (re.compile(r"^(next|previous|last) (?:song|track)$", re.I),
     lambda m, t: f"media(action={'next' if m.group(1).lower() == 'next' else 'previous'})"),
    (re.compile(r"^skip(?: (?:this )?(?:song|track))?$", re.I), lambda m, t: "media(action=next)"),
    (re.compile(r"^(?:pause|resume|play pause)(?: (?:the )?(?:music|video|song))?$", re.I),
     lambda m, t: "media(action=play_pause)"),
    (re.compile(r"^stop (?:the )?(?:music|playback|song)$", re.I), lambda m, t: "media(action=stop)"),
    (re.compile(r"^(?:set )?(?:the )?(?:screen )?brightness (?:to |at )?(\d{1,3})(?: ?%| percent)?$", re.I),
     lambda m, t: f"brightness(level={int(m.group(1))})"),
    (re.compile(r"^(?:increase|raise|turn up) (?:the )?brightness$|^brightness up$|^(?:make the screen )?brighter$", re.I),
     lambda m, t: "brightness(action=up)"),
    (re.compile(r"^(?:decrease|lower|turn down) (?:the )?brightness$|^brightness down$|^dim (?:the )?screen$", re.I),
     lambda m, t: "brightness(action=down)"),
    (re.compile(r"^(?:set )?(?:the )?volume (?:to |at )?(\d{1,3})(?: ?%| percent)?$", re.I),
     lambda m, t: f"set_volume(level={int(m.group(1))})"),
    (re.compile(r"^(?:turn|switch) (on|off) (?:the )?(wi-?fi|bluetooth)$", re.I),
     lambda m, t: f"radio(kind={m.group(2).lower().replace('-', '')}, state={m.group(1).lower()})"),
    (re.compile(r"^(enable|disable) (?:the )?(wi-?fi|bluetooth)$", re.I),
     lambda m, t: f"radio(kind={m.group(2).lower().replace('-', '')}, state={'on' if m.group(1).lower() == 'enable' else 'off'})"),
    (re.compile(r"^(wi-?fi|bluetooth) (on|off)$", re.I),
     lambda m, t: f"radio(kind={m.group(1).lower().replace('-', '')}, state={m.group(2).lower()})"),
    (re.compile(r"^(?:cancel|abort|stop) (?:the )?(?:shutdown|restart)$|^don'?t shut ?down$", re.I),
     lambda m, t: "power(action=cancel)"),
    (re.compile(r"^(?:shut ?down|power off|turn off)(?: (?:the|my) (?:pc|computer|laptop))?(?: in (\d+) ?(m|mins?|minutes?|h|hours?))?$", re.I),
     lambda m, t: "power(action=shutdown" + (f", after={m.group(1)}{m.group(2)[0].lower()})" if m.group(1) else ")")),
    (re.compile(r"^(?:restart|reboot)(?: (?:the|my) (?:pc|computer|laptop))?(?: in (\d+) ?(m|mins?|minutes?|h|hours?))?$", re.I),
     lambda m, t: "power(action=restart" + (f", after={m.group(1)}{m.group(2)[0].lower()})" if m.group(1) else ")")),
    (re.compile(r"^(?:sign|log) (?:me )?out$", re.I), lambda m, t: "power(action=sign_out)"),
    (re.compile(r"^(?:put (?:the |my )?(?:pc|computer|laptop) to )?sleep(?: mode)?$", re.I), lambda m, t: "power(action=sleep)"),
    (re.compile(r"^(new|next|previous) (?:virtual )?desktop$", re.I),
     lambda m, t: f"desktop(action={m.group(1).lower()})"),
    (re.compile(r"^close (?:this|the) (?:virtual )?desktop$", re.I), lambda m, t: "desktop(action=close)"),
    (re.compile(r"^(?:show (?:the )?desktop|minimi[sz]e (?:all|everything)(?: (?:the )?windows)?)$", re.I),
     lambda m, t: "window(action=minimize_all)"),
    (re.compile(r"^(?:snap|move|put) (.+?) (?:to the |on the )?(left|right)(?: half| side)?$", re.I),
     lambda m, t: _window(f"snap_{m.group(2).lower()}", m.group(1))),
    (re.compile(r"^(maximi[sz]e|minimi[sz]e|close|restore) (.+)$", re.I),
     lambda m, t: _window({"maximise": "maximize", "minimise": "minimize"}.get(m.group(1).lower(), m.group(1).lower()),
                          m.group(2))),
    (re.compile(r"^(?:open|launch|start) (?:the )?(?:website |site )?((?:https?://)?[\w\-]+\.[\w.\-/]+)$", re.I),
     lambda m, t: f'open_url(url="{m.group(1)}")'),
    (re.compile(rf"^(?:open|show) (?:my )?{_FOLDER}(?: folder)?$", re.I),
     lambda m, t: f"open_folder(path={m.group(1).capitalize()})"),
    (re.compile(r"^(?:open|launch|start) (?:the )?([\w\-]+)(?: app)?$", re.I),
     lambda m, t: f"open_app(name={m.group(1).lower()})"),
    (re.compile(r"^remind me (?:to )?(.+?) in (\d+) (\w+)$", re.I),
     lambda m, t: f'remind(text="{m.group(1)}", after={_dur(m.group(2), m.group(3))})'),
    (re.compile(r"^remind me (?:to )?(.+?) at (\d{1,2})(?::(\d{2}))? ?(am|pm)?$", re.I),
     lambda m, t: f'remind(text="{m.group(1)}", at={_time(m.group(2), m.group(3), m.group(4))})'),
    (re.compile(rf"^(move|copy|put|shift) .*?(?:from|in|inside) {_FOLDER} (?:in)?to (?:a |an )?new folder(?: called| named)? ['\"]?([\w .-]+?)['\"]?$", re.I),
     lambda m, t: '{}_files(src={}, new_folder="{}"{})'.format(
         "copy" if m.group(1).lower() == "copy" else "move", m.group(2).capitalize(), m.group(3).strip(),
         (", " + _filters(t)) if _filters(t) else "")),
    (re.compile(rf"^(move|copy|put|shift) .*?(?:from|in|inside) {_FOLDER} (?:in)?to (?:the |my )?{_FOLDER}(?: folder)?$", re.I),
     lambda m, t: "{}_files(src={}, dest={}{})".format(
         "copy" if m.group(1).lower() == "copy" else "move", m.group(2).capitalize(), m.group(3).capitalize(),
         (", " + _filters(t)) if _filters(t) else "")),
    (re.compile(rf"^(move|copy) .*?from {_FOLDER} to {_FOLDER}", re.I),
     lambda m, t: "{}_files(src={}, dest={}{})".format(
         m.group(1).lower(), m.group(2).capitalize(), m.group(3).capitalize(),
         (", " + _filters(t)) if _filters(t) else "")),
    (re.compile(rf"^(?:delete|remove|clean ?up) .*?(?:from|in) {_FOLDER}", re.I),
     lambda m, t: "delete_files(src={}{})".format(
         m.group(1).capitalize(), (", " + _filters(t)) if _filters(t) else "")),
    (re.compile(rf"^(?:list|show) .*?(?:in|from) {_FOLDER}", re.I),
     lambda m, t: "list_files(src={}{})".format(
         m.group(1).capitalize(), (", " + _filters(t)) if _filters(t) else "")),
    (re.compile(r"^lock(?: the)?(?: screen| pc| computer)?$", re.I), lambda m, t: "system(action=lock)"),
    (re.compile(r"^(?:mute|unmute)(?: the)?(?: sound| audio| volume)?$", re.I), lambda m, t: "system(action=mute)"),
    (re.compile(r"^(?:volume up|louder|increase volume)$", re.I), lambda m, t: "system(action=volume_up)"),
    (re.compile(r"^(?:volume down|quieter|decrease volume)$", re.I), lambda m, t: "system(action=volume_down)"),
    # memory search (phase 7): backup for "find ..." when there's no model or it's unsure
    (re.compile(rf"^(?:find|locate|dig up|track down|look for|search (?:my files|memory) for|where(?:'s| is| did i save)"
                rf"|do you remember|i'm looking for)\b(?!.*\b(?:in|from) {_FOLDER}\s*$)(?!.*\bon (?:google|youtube|amazon"
                rf"|flipkart|github|wikipedia|reddit|spotify)\b)\s+(.+?)\??$", re.I),
     lambda m, t: _memory(m.groups()[-1], t)),
]


_MODES = [(r"\b(?:by|on|via|in) (?:a |the )?(?:bus|metro|train|local|public transport|transit)\b|\b(?:bus|metro|train) se\b", "transit"),
          (r"\b(?:on|by) (?:a |my )?(?:bike|scooter|scooty|two[- ]?wheeler|motorcycle)\b|\b(?:bike|scooty) se\b"
           r"|\bride\b", "two_wheeler"),
          (r"\b(?:on foot|by walk|walking|walk)\b|\bpaidal\b", "walking"),
          (r"\b(?:by|on) (?:a )?(?:cycle|bicycle)\b|\bcycling\b", "bicycling"),
          (r"\b(?:by|in) (?:a |my )?car\b|\bdriv(?:e|ing)\b|\bcar se\b", "driving")]


def _directions(text: str) -> str:
    t = text.strip().rstrip("?.! ")
    mode = None
    for pat, m in _MODES:
        if re.search(pat, t, re.I):
            mode = m
            t = re.sub(pat, " ", t, flags=re.I)
            break
    t = re.sub(r"\s+", " ", t).strip()
    origin = dest = None
    hi = re.match(r"^(?:(.+?) se )?(.+?) (?:tak |jaane me |pahunchne me )?(?:kitna (?:time|samay|der) lagega|kaise "
                  r"(?:jaun|jaaun|jayen|jaye|pahunchu|pahunchun)|kitna door hai)$", t, re.I)
    if hi:
        origin, dest = hi.group(1), hi.group(2)
    else:
        for pat, order in ((r"\bfrom (.+?) to (.+)$", "od"), (r"\b(?:to|is) (?:get |reach |go )?(?:to )?(.+?) from (.+)$", "do"),
                           (r"\b(?:to|is) (?:get |reach |go )?(?:to )?(.+)$", "d")):
            m = re.search(pat, t, re.I)
            if m:
                if order == "od":
                    origin, dest = m.group(1), m.group(2)
                elif order == "do":
                    dest, origin = m.group(1), m.group(2)
                else:
                    dest = m.group(1)
                break
        else:
            dest = t
    dest = re.sub(r"^(?:(?:is |was )?it (?:to |till )?|the |reach |get to |go to )+", "", dest.strip(), flags=re.I)
    if origin:
        origin = re.sub(r"^(?:my |the )", "", origin.strip(), flags=re.I) if origin.strip().lower() in (
            "my home", "my office", "the office") else origin
    dest = re.sub(r"\s+(?:from here|right now|now|today)$", "", dest, flags=re.I).strip()
    if re.fullmatch(r"(?:get |go )?home|ghar", dest, re.I):
        dest = "home"
    parts = [f'destination="{dest}"'] + ([f"mode={mode}"] if mode else []) + \
        ([f'origin="{origin.strip()}"'] if origin and origin.strip().lower() not in ("here", "my location") else [])
    return "directions(" + ", ".join(parts) + ")"


def _memory(rest: str, text: str) -> str:
    from ..memory.query import split_query
    rest = re.sub(r"\b(?:in my (?:files|stuff|computer|pc)|for me|please)\b", " ", rest, flags=re.I)
    query, found = split_query(text, rest)
    parts = [f"{k}={v}" for k, v in sorted(found.items())]
    if query:
        parts.append(f'query="{query}"')
    return "search_memory(" + ", ".join(parts) + ")"


_WHOLE_SENTENCE = next(i for i, (p, _) in enumerate(RULES) if p.pattern.startswith('^(?:google|search(?: google)? for)')) + 1


def rule_parse(text: str) -> Program | None:
    text = text.strip().rstrip(".!")
    for pattern, build in RULES[:_WHOLE_SENTENCE]:  # whole-sentence play/search rules ("open youtube and play X") go first
        m = pattern.search(text)
        if m:
            built = build(m, text)
            if built is None:              # the rule matched the shape but not the details: try the others
                continue
            try:
                return parse(built)
            except DSLError:
                break
    # Compound commands: split on ' and then ' / ' then ' / ' and '
    pieces = re.split(r"\s+(?:and then|then|and)\s+(?=(?:open|launch|start|remind|move|copy|delete|remove|list|show|lock|mute|play|search|snap|maximi[sz]e|minimi[sz]e|close|take|turn|set|shut|restart|skip|pause|next|stop|enable|disable)\b)",
                      text, flags=re.I)
    dsl = []
    for piece in pieces:
        for pattern, build in RULES:
            m = pattern.search(piece.strip())
            if m:
                built = build(m, piece)
                if built is None:
                    continue
                dsl.append(built)
                break
        else:
            return None
    try:
        return parse(" ; ".join(dsl))
    except DSLError:
        return None
