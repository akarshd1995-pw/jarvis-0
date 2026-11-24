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


RULES: list[tuple[re.Pattern, callable]] = [
    (re.compile(r"^undo\b", re.I), lambda m, t: "undo()"),
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
            try:
                return parse(build(m, text))
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
                dsl.append(build(m, piece))
                break
        else:
            return None
    try:
        return parse(" ; ".join(dsl))
    except DSLError:
        return None
