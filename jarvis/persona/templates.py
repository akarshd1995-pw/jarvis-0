"""Reply templates per tone. Placeholders are filled by persona.Persona.

{A}      optional form of address, e.g. ", sir" (inserted only sometimes, and only in tones that use it)
{Count}  number at the start of a sentence (spelled out when small)   {count}  number mid-sentence
{files}  "PDFs" / "PNG files" / "files" (agrees with count)          {filt}   " older than 30 days" / ""
{src} {dest} {name} {query} {site} {text} {when} {request} {items} {action_phrase}

Lines are written to be read aloud later (phase 15 voice): no symbols, no emoji, short sentences.
"""

T: dict[str, dict[str, list[str]]] = {
    # ------------------------------------------------------------------------------ butler
    "butler": {
        "move": ["Done{A}. {Count} {files} moved to {dest}.", "{Count} {files} are now in {dest}{A}.",
                 "Moved {count} {files} from {src} to {dest}.", "All set{A}. {Count} {files} relocated to {dest}."],
        "copy": ["Copied {count} {files} to {dest}{A}.", "{Count} {files} copied to {dest}.",
                 "Done. Copies of {count} {files} are in {dest}."],
        "delete": ["Deleted {count} {files} from {src}. They're in my trash if you need them back.",
                   "{Count} {files} removed from {src}{A}. Say undo to restore them.",
                   "Done{A}. {Count} {files} cleared from {src}. Undo will bring them back."],
        "none": ["There are no {files_pl}{filt} in {src}{A}. Nothing to {opverb}.",
                 "I didn't find any {files_pl}{filt} in {src}, so there was nothing to {opverb}."],
        "list": ["You have {count} {files}{filt} in {src}{A}: {items}.", "{Count} {files}{filt} in {src}: {items}.",
                 "I found {count} {files}{filt} in {src}: {items}."],
        "list_none": ["There are no {files_pl}{filt} in {src}{A}.", "{src} has no {files_pl}{filt}."],
        "confirm_move": ["Shall I move {count} {files}{filt} from {src} to {dest}?",
                         "{Count} {files}{filt} will go from {src} to {dest}. Shall I proceed{A}?"],
        "confirm_copy": ["Shall I copy {count} {files}{filt} from {src} to {dest}?",
                         "Copy {count} {files}{filt} into {dest}{A}?"],
        "confirm_delete": ["Delete {count} {files}{filt} from {src}? I'll keep them in my trash so you can undo.",
                           "Shall I remove {count} {files}{filt} from {src}{A}? They can be restored with undo."],
        "confirm_system": ["Shall I {action_phrase}{A}?"],
        "open_app": ["Opening {name}{A}.", "Launching {name}.", "{Name} is on its way."],
        "open_url": ["Opening {name}{A}.", "Taking you to {name}."],
        "open_folder": ["Opening {name}{A}.", "Here's {name}."],
        "remind": ["I'll remind you to {text} {when}{A}.", "Reminder set{A}: {text}, {when}.",
                   "Very good. I'll remind you to {text} {when}."],
        "system_mute": ["Muted{A}.", "Sound off."], "system_volume_up": ["Turning it up{A}.", "Volume up."],
        "system_volume_down": ["Turning it down{A}.", "Volume down."],
        "system_lock": ["Locking the screen{A}.", "Locked. I'll be here when you're back."],
        "system_sleep_display": ["Display off{A}.", "Turning off the display."],
        "system_unsupported_os": ["That only works on Windows{A}."],
        "search": ["Searching {site} for {query}{A}.", "Here's what {site} has on {query}."],
        "play": ["Playing {query} on YouTube{A}.", "Now playing {query}.", "Here's {query}{A}."],
        "play_spotify": ["Opening {query} in Spotify{A}.", "Here's {query} on Spotify."],
        "play_fallback": ["I couldn't pick a video automatically{A}, so here are the YouTube results for {query}."],
        "memory": ["Memory search isn't built yet{A}. Once it is, I'll be able to find {query} for you."],
        "unsupported": ["I can't {request} yet{A}.", "I'm afraid I can't {request} yet."],
        "unknown": ["I'm not able to help with that yet{A}. I can manage files, open apps and sites, play and "
                    "search, set reminders, and control the volume.",
                    "That's beyond me for now{A}. Try a file, app, music or reminder request."],
        "undo": ["Undone{A}. {Count} changes reversed.", "Reversed {count} changes. Everything is back as it was."],
        "undo_none": ["There's nothing to undo{A}."],
        "undo_cancelled": ["Very well, I'll leave things as they are."],
        "declined": ["Understood{A}, I've left them alone.", "Very well, cancelled."],
        "err_not_found": ["I couldn't find the folder {name}{A}."],
        "err_outside": ["That folder is outside the places I'm allowed to touch{A}. You can add it in settings."],
        "err_too_many": ["That matched more files than I'll touch at once{A}. Could you narrow it down?"],
        "err_other": ["Something went wrong while trying to {action_phrase}{A}: {error}."],
        "unparsed": ["I'm sorry{A}, I didn't catch that. Could you rephrase?",
                     "I didn't quite understand that{A}. Could you say it another way?"],
        "reminder_due": ["You asked me to remind you{A}: {text}.", "A reminder{A}: {text}."],
    },
    # ------------------------------------------------------------------------------ friendly
    "friendly": {
        "move": ["Done! Moved {count} {files} to {dest}.", "{Count} {files} are in {dest} now.",
                 "Moved {count} {files} over to {dest}."],
        "copy": ["Copied {count} {files} to {dest}.", "Done, {count} {files} copied to {dest}."],
        "delete": ["Deleted {count} {files} from {src}. Say undo if you change your mind.",
                   "Gone! {Count} {files} removed from {src}. Undo brings them back."],
        "none": ["No {files_pl}{filt} in {src}, so nothing to {opverb}.", "Couldn't find any {files_pl}{filt} in {src}."],
        "list": ["You've got {count} {files}{filt} in {src}: {items}.", "Found {count} {files}{filt} in {src}: {items}."],
        "list_none": ["No {files_pl}{filt} in {src}.", "{src} doesn't have any {files_pl}{filt}."],
        "confirm_move": ["Move {count} {files}{filt} from {src} to {dest}?"],
        "confirm_copy": ["Copy {count} {files}{filt} from {src} to {dest}?"],
        "confirm_delete": ["Delete {count} {files}{filt} from {src}? You can undo it later."],
        "confirm_system": ["Want me to {action_phrase}?"],
        "open_app": ["Opening {name}!", "Here comes {name}."], "open_url": ["Opening {name}."],
        "open_folder": ["Opening {name}."],
        "remind": ["Got it, I'll remind you to {text} {when}.", "Reminder's set: {text}, {when}."],
        "system_mute": ["Muted."], "system_volume_up": ["Louder it is."], "system_volume_down": ["Turned it down."],
        "system_lock": ["Locking up. See you soon!"], "system_sleep_display": ["Screen off."],
        "system_unsupported_os": ["That one only works on Windows."],
        "search": ["Searching {site} for {query}.", "Here's {site} for {query}."],
        "play": ["Playing {query}!", "Here's {query} on YouTube."], "play_spotify": ["Opening {query} in Spotify."],
        "play_fallback": ["Couldn't pick a video, so here are the YouTube results for {query}."],
        "memory": ["Memory search is coming soon. Then I'll find {query} for you."],
        "unsupported": ["I can't {request} yet, sorry!", "Can't {request} yet."],
        "unknown": ["Hmm, I can't help with that yet. I'm good with files, apps, music, searches and reminders."],
        "undo": ["Undone! {Count} changes reversed."], "undo_none": ["Nothing to undo."],
        "undo_cancelled": ["Okay, leaving it as is."], "declined": ["Okay, cancelled."],
        "err_not_found": ["Couldn't find a folder called {name}."],
        "err_outside": ["I'm not allowed to touch that folder. You can add it in settings."],
        "err_too_many": ["That's a lot of files! Can you narrow it down?"],
        "err_other": ["Oops, couldn't {action_phrase}: {error}."],
        "unparsed": ["Sorry, didn't get that. Try saying it another way?"],
        "reminder_due": ["Hey! Reminder: {text}."],
    },
    # ------------------------------------------------------------------------------ minimal
    "minimal": {
        "move": ["Moved {count} {files} to {dest}."], "copy": ["Copied {count} {files} to {dest}."],
        "delete": ["Deleted {count} {files} from {src}. Undo available."],
        "none": ["No {files_pl}{filt} in {src}."], "list": ["{Count} {files}{filt} in {src}: {items}."],
        "list_none": ["No {files_pl}{filt} in {src}."],
        "confirm_move": ["Move {count} {files}{filt}, {src} to {dest}?"],
        "confirm_copy": ["Copy {count} {files}{filt}, {src} to {dest}?"],
        "confirm_delete": ["Delete {count} {files}{filt} from {src}?"], "confirm_system": ["{Action_phrase}?"],
        "open_app": ["Opening {name}."], "open_url": ["Opening {name}."], "open_folder": ["Opening {name}."],
        "remind": ["Reminder: {text}, {when}."], "system_mute": ["Muted."], "system_volume_up": ["Volume up."],
        "system_volume_down": ["Volume down."], "system_lock": ["Locked."], "system_sleep_display": ["Display off."],
        "system_unsupported_os": ["Windows only."], "search": ["Searching {site}: {query}."],
        "play": ["Playing {query}."], "play_spotify": ["Spotify: {query}."],
        "play_fallback": ["YouTube results: {query}."], "memory": ["Memory search not built yet."],
        "unsupported": ["Can't {request} yet."], "unknown": ["Can't do that yet."],
        "undo": ["Undid {count} changes."], "undo_none": ["Nothing to undo."], "undo_cancelled": ["Cancelled."],
        "declined": ["Cancelled."], "err_not_found": ["Folder not found: {name}."],
        "err_outside": ["Folder not allowed: {name}."], "err_too_many": ["Too many files. Narrow it down."],
        "err_other": ["Failed to {action_phrase}: {error}."], "unparsed": ["Didn't understand."],
        "reminder_due": ["Reminder: {text}."],
    },
}

TONES = list(T)

# How often each tone adds the form of address ("sir") to a sentence that offers a {A} slot.
ADDRESS_RATE = {"butler": 0.4, "friendly": 0.0, "minimal": 0.0}

SYSTEM_PHRASES = {"mute": "mute the sound", "volume_up": "turn the volume up", "volume_down": "turn the volume down",
                  "lock": "lock the screen", "sleep_display": "turn off the display"}
OP_VERBS = {"move": "move", "copy": "copy", "delete": "delete", "list": "list"}
SITE_NAMES = {"google": "Google", "youtube": "YouTube", "amazon": "Amazon", "flipkart": "Flipkart",
              "swiggy": "Swiggy", "zomato": "Zomato", "github": "GitHub", "stackoverflow": "Stack Overflow",
              "wikipedia": "Wikipedia", "spotify": "Spotify", "maps": "Google Maps", "reddit": "Reddit",
              "linkedin": "LinkedIn"}
ACTION_PHRASES = {"move_files": "move those files", "copy_files": "copy those files",
                  "delete_files": "delete those files", "list_files": "list those files",
                  "open_app": "open that app", "open_url": "open that website", "open_folder": "open that folder",
                  "remind": "set that reminder", "system": "do that", "search_web": "search for that",
                  "play_media": "play that", "search_memory": "search for that"}
