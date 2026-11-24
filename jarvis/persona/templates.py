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
        "play_nomatch": ["I couldn't find {query}{A}. The closest was \"{closest}\", so I've opened the results instead of "
                         "playing something random. Could you check the spelling or add the artist?"],
        "play_none": ["YouTube has nothing for {query}{A}. Could you check the spelling or add the artist or film?"],
        "play_offline": ["I couldn't reach YouTube{A}. Please check the internet connection; I've opened the results page for {query}."],
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
        "play_nomatch": ["Couldn't find {query}. Closest was \"{closest}\", so here are the results. Check the spelling?"],
        "play_none": ["Nothing on YouTube for {query}. Check the spelling or add the artist?"],
        "play_offline": ["Can't reach YouTube right now. Check your internet?"],
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
        "play_fallback": ["YouTube results: {query}."], "play_nomatch": ["No match for {query}. Results opened."],
        "play_none": ["No results for {query}."], "play_offline": ["YouTube unreachable."], "memory": ["Memory search not built yet."],
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

# ------------------------------------------------------------------ phase 5: system control
_SYS = {
    "butler": {
        "window_snap_left": ["{Target} is on the left{A}.", "Snapping {target} to the left."],
        "window_snap_right": ["{Target} is on the right{A}.", "Snapping {target} to the right."],
        "window_maximize": ["Maximising {target}{A}.", "{Target} is full screen now."],
        "window_minimize": ["Minimising {target}{A}.", "{Target} is tucked away."],
        "window_restore": ["Bringing {target} back{A}.", "Here's {target}."],
        "window_close": ["Closing {target}{A}.", "{Target} is closing. It may ask you to save first."],
        "window_minimize_all": ["Here's your desktop{A}.", "Everything minimised."],
        "desktop_new": ["A fresh desktop for you{A}.", "New virtual desktop ready."],
        "desktop_next": ["Next desktop{A}."], "desktop_previous": ["Previous desktop{A}."],
        "desktop_close": ["Closing this desktop{A}. Its windows move to the next one."],
        "media_play_pause": ["Play or pause, done{A}.", "Toggled playback."],
        "media_next": ["Next track{A}.", "Skipping ahead."], "media_previous": ["Previous track{A}.", "Going back one."],
        "media_stop": ["Playback stopped{A}."],
        "brightness_set": ["Brightness set to {level} percent{A}.", "Screen brightness is now {level} percent."],
        "brightness_up": ["Brighter now{A}, {level} percent.", "Brightness up to {level} percent."],
        "brightness_down": ["Dimmed to {level} percent{A}.", "Brightness down to {level} percent."],
        "volume_set": ["Volume set to {level} percent{A}.", "Volume is now {level} percent."],
        "radio_on": ["{Device} is on{A}.", "Turning {device} on."],
        "radio_off": ["{Device} is off{A}.", "Turning {device} off."],
        "screenshot_full": ["Screenshot taken{A}. It's in Pictures, Screenshots.",
                            "Captured the whole screen. You'll find it in Pictures, Screenshots."],
        "screenshot_window": ["Copied the active window to your clipboard{A}."],
        "screenshot_region": ["Select the area you'd like{A}.", "Snipping tool ready. Drag over the area."],
        "record_start": ["Recording the active app{A}. Say stop recording when you're done.",
                         "Recording started{A}."],
        "record_stop": ["Recording stopped{A}. It's in Videos, Captures."],
        "power_shutdown": ["Shutting down in {delay}{A}. Say cancel shutdown to stop it."],
        "power_restart": ["Restarting in {delay}{A}. Say cancel shutdown to stop it."],
        "power_sleep": ["Going to sleep{A}. Goodnight."], "power_sign_out": ["Signing you out{A}."],
        "power_cancel": ["Cancelled{A}. The PC stays on.", "Shutdown cancelled{A}."],
        "confirm_window_close": ["Shall I close {target}{A}?"],
        "confirm_radio_off": ["Turn off {device}{A}? You may lose your connection."],
        "confirm_power_shutdown": ["Shut down the PC in {delay}{A}?", "Shall I shut down in {delay}?"],
        "confirm_power_restart": ["Restart the PC in {delay}{A}?"],
        "confirm_power_sleep": ["Put the PC to sleep now{A}?"],
        "confirm_power_sign_out": ["Sign out now{A}? Unsaved work in open apps may be lost."],
        "err_no_window": ["I can't find an open {name} window{A}.", "There's no {name} window open{A}."],
        "err_not_windows": ["That only works on Windows{A}."],
    },
    "friendly": {
        "window_snap_left": ["{Target} snapped left!"], "window_snap_right": ["{Target} snapped right!"],
        "window_maximize": ["{Target} maximised."], "window_minimize": ["{Target} minimised."],
        "window_restore": ["Here's {target}."], "window_close": ["Closing {target}."],
        "window_minimize_all": ["Desktop's clear!"], "desktop_new": ["New desktop!"],
        "media_next": ["Next song!"], "media_previous": ["Back one song."], "media_play_pause": ["Done!"],
        "brightness_set": ["Brightness at {level} percent."], "volume_set": ["Volume at {level} percent."],
        "radio_on": ["{Device} on!"], "radio_off": ["{Device} off."],
        "screenshot_full": ["Got it! Screenshot saved in Pictures, Screenshots."],
        "power_shutdown": ["Shutting down in {delay}. Say cancel shutdown if you change your mind."],
        "power_restart": ["Restarting in {delay}. Say cancel shutdown to stop it."],
        "power_cancel": ["Phew, cancelled."], "err_no_window": ["Couldn't find a {name} window."],
    },
    "minimal": {
        "window_snap_left": ["{Target}: left."], "window_snap_right": ["{Target}: right."],
        "window_maximize": ["Maximised."], "window_minimize": ["Minimised."], "window_restore": ["Restored."],
        "window_close": ["Closed {target}."], "window_minimize_all": ["Desktop."], "desktop_new": ["New desktop."],
        "desktop_next": ["Next desktop."], "desktop_previous": ["Previous desktop."], "desktop_close": ["Desktop closed."],
        "media_play_pause": ["Play/pause."], "media_next": ["Next."], "media_previous": ["Previous."],
        "media_stop": ["Stopped."], "brightness_set": ["Brightness {level} percent."],
        "brightness_up": ["Brightness {level} percent."], "brightness_down": ["Brightness {level} percent."],
        "volume_set": ["Volume {level} percent."], "radio_on": ["{Device} on."], "radio_off": ["{Device} off."],
        "screenshot_full": ["Screenshot saved."], "screenshot_window": ["Window copied."],
        "screenshot_region": ["Select area."], "record_start": ["Recording."], "record_stop": ["Recording stopped."],
        "power_shutdown": ["Shutdown in {delay}."], "power_restart": ["Restart in {delay}."], "power_sleep": ["Sleeping."],
        "power_sign_out": ["Signing out."], "power_cancel": ["Cancelled."],
        "confirm_window_close": ["Close {target}?"], "confirm_radio_off": ["Turn off {device}?"],
        "confirm_power_shutdown": ["Shut down in {delay}?"], "confirm_power_restart": ["Restart in {delay}?"],
        "confirm_power_sleep": ["Sleep now?"], "confirm_power_sign_out": ["Sign out?"],
        "err_no_window": ["No {name} window."], "err_not_windows": ["Windows only."],
    },
}
for _tone, _keys in _SYS.items():
    T[_tone].update(_keys)

ACTION_PHRASES.update({"window": "arrange that window", "desktop": "switch desktops", "media": "control playback",
                       "brightness": "change the brightness", "set_volume": "set the volume",
                       "radio": "change that setting", "screenshot": "take a screenshot",
                       "screen_record": "record the screen", "power": "do that"})

# ------------------------------------------------------------------ phase 23: monitoring, alerts, training control
_MON = {
    "butler": {
        "training_paused": ["Training paused{A}. Say resume training when you're ready.", "The {name} run is on hold{A}."],
        "training_improve_started": ["Very good{A}. I'm fine-tuning myself on your corrections in the background, "
                                     "about half an hour. I'll only switch if the new model scores better."],
        "training_improve_running": ["I'm already improving{A}. You'll hear the result when it's done."],
        "training_improve_nomodel": ["There's no trained model to improve yet{A}. Run train_night.bat once first."],
        "training_rollback": ["Done{A}. I'm back on {version}."],
        "training_rollback_none": ["{msg}"],
        "training_resumed": ["Training resumed{A}.", "Back to work. Training has resumed."],
        "training_resume_none": ["Nothing was paused{A}."], "training_pause_none": ["No training is running right now{A}."],
        "training_status": ["The {name} run is {pct} done{A}, about {eta} to go.", "{Name} training: {pct} complete, roughly {eta} left."],
        "training_status_paused": ["The {name} run is paused at {pct}{A}."],
        "training_status_done": ["The {name} run has finished{A}. Best score {best}."],
        "training_status_none": ["No training is running at the moment{A}."],
        "alert_gpu_hot": ["The GPU is at {temp} degrees{A}. You may want to give it a rest."],
        "alert_gpu_hot_training": ["The GPU is at {temp} degrees{A}. Shall I pause training? Just say pause training."],
        "alert_disk_low": ["Your C drive is down to {free_gb} gigabytes free{A}. Time for a clean-up."],
        "alert_battery_low": ["Battery is at {pct} percent and not charging{A}."],
        "alert_ram_high": ["Memory has been at {pct} percent for a while{A}. Something may be using too much."],
        "alert_training_done": ["Training finished{A}. The {name} scored {best}.", "Good news{A}. The {name} run is done, best score {best}."],
        "alert_training_crashed": ["The {name} run has stopped with an error{A}: {error}"],
        "alert_training_stalled": ["The {name} run hasn't moved in {minutes} minutes{A}. It may be stuck."],
        "alert_unusual": ["{Metric} is at {value} percent{A}, well above your usual {typical} for this time of day."],
    },
    "friendly": {
        "training_paused": ["Training paused! Say resume training to carry on."], "training_resumed": ["And we're back! Training resumed."],
        "training_status": ["{Name} training is {pct} done, about {eta} left."],
        "alert_gpu_hot_training": ["Whoa, the GPU is at {temp} degrees. Want me to pause training?"],
        "alert_training_done": ["Training's done! The {name} scored {best}."],
        "alert_disk_low": ["Heads up, only {free_gb} gigabytes left on C."],
    },
    "minimal": {
        "training_paused": ["Training paused."], "training_resumed": ["Training resumed."],
        "training_improve_started": ["Improving in the background. ~30 min."], "training_rollback": ["Back on {version}."],
        "training_resume_none": ["Nothing paused."], "training_pause_none": ["No training running."],
        "training_status": ["{Name}: {pct}, {eta} left."], "training_status_paused": ["{Name}: paused at {pct}."],
        "training_status_done": ["{Name}: done, best {best}."], "training_status_none": ["No training running."],
        "alert_gpu_hot": ["GPU {temp} degrees."], "alert_gpu_hot_training": ["GPU {temp} degrees. Pause training?"],
        "alert_disk_low": ["C: {free_gb} GB free."], "alert_battery_low": ["Battery {pct} percent."],
        "alert_ram_high": ["RAM {pct} percent."], "alert_training_done": ["{Name} done: {best}."],
        "alert_training_crashed": ["{Name} crashed: {error}"], "alert_training_stalled": ["{Name} stalled {minutes} min."],
        "alert_unusual": ["{Metric} {value} percent (usual {typical})."],
    },
}
for _tone, _keys in _MON.items():
    T[_tone].update(_keys)
ACTION_PHRASES["training"] = "control training"

# ------------------------------------------------------------------ richer play modes
_PLAY = {
    "butler": {
        "play_latest": ["Playing the latest from {channel}{A}: {title}.", "Here's {channel}'s newest video{A}: {title}.",
                        "{Channel} just posted {title}. Playing it now."],
        "play_playlist": ["Playing a {query} playlist{A}.", "Here's a {query} playlist for you."],
        "play_live": ["Tuning in to {query} live{A}.", "Here's the {query} live stream."],
        "play_channel": ["Opening the {query} channel{A}.", "Here's {query} on YouTube."],
    },
    "friendly": {
        "play_latest": ["Here's the newest from {channel}: {title}!", "{Channel}'s latest: {title}. Enjoy!"],
        "play_playlist": ["Putting on a {query} playlist!"], "play_live": ["Going live with {query}!"],
        "play_channel": ["Here's the {query} channel."],
    },
    "minimal": {
        "play_latest": ["{Channel}: {title}."], "play_playlist": ["Playlist: {query}."], "play_live": ["Live: {query}."],
        "play_channel": ["Channel: {query}."],
    },
}
for _tone, _keys in _PLAY.items():
    T[_tone].update(_keys)
