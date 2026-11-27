# Jarvis-0 roadmap

**Vision:** a JARVIS-style assistant in the spirit of the Iron Man films. It replies in character,
runs your machines, keeps watch over them, briefs you unprompted, remembers context, answers questions
offline, and lives in an animated HUD. Voice comes last, once everything else works. Our own design,
voice and name; no film assets or actor voice.

**Rule: 100% from scratch.** Every model is trained from random weights on data we generate or on
open datasets. No pretrained models, including for conversation: open questions are handled by
offline knowledge Q&A (phase 26) plus a small conversation model of our own (phase 27).

**Build order:** 4 → 22 → 25 → 5 → 23 → 24 → 14 → 7 → **29 → 40 → 39 → 6 → 38 → 44 → 41 → 42** → 30 → 31 → 32 → 33 → 43 → 27 → 12 → 34 → 35 → 36 → 37 → 15, then the rest (26 Offline Q&A is parked for later).
- **4, 22, 25:** it feels like JARVIS. It runs properly, replies in character, and has the HUD.
- **5, 23, 24:** it runs and watches your machines, and briefs you.
- **14:** every day of use improves it automatically.
- **7, 6, 27:** it remembers and holds a conversation.
- **29, 40, 39, 6, 38, 44, 41, 42:** it's robust. It never falls over, never half-finishes an action, learns from
  how you really talk, asks when unsure, understands "this" and "here", works on any Windows setup, stays light,
  and never ships a broken update.
- **30–34:** it runs your day: notes, workspaces, focus and gaming mode, tasks and deadlines, wellbeing.
- **35–37:** it understands your stuff: spending from receipts, answers from your own documents, system care.
- **12:** it acts on its own, always with your OK.
- **15:** voice comes last and plugs into everything above (persona, alerts, briefings, conversation).

Status: ✅ done · 🔨 in progress · ⬜ planned

## Stage A: Foundation
| # | Phase | Status |
|---|---|---|
| 1 | **Agent core**: command bar (Ctrl+Alt+J), file, app, URL, folder, reminder and system actions; safety tiers, previews, confirmations, undo journal; focus and Downloads logging | ✅ |
| 2 | **Own command model**: synthetic English and Hinglish data engine, BPE tokenizer and encoder-decoder transformer from scratch, evaluation reports | ✅ |
| 3 | **Polish**: `search_web`, `unsupported` skipped-step reporting, grammar-constrained decoding, `/fix` `/good` `/wrong` corrections, negation safety data, 2M examples, 27M-param model. Result: 99.83% familiar / 99.84% unseen phrasing, 0.14% wrong actions | ✅ |

## Stage B: Feels like JARVIS
| # | Phase | What it adds | Status |
|---|---|---|---|
| 4 | **Runtime & packaging** | ONNX model (no PyTorch needed), autostart with Windows, tray icon, settings window, encrypted local store, one-click update | ✅ |
| 22 | 🆕 **Persona layer** | In-character replies for every action ("Done. Fourteen PDFs are in Archive."), confirmations and errors. Configurable name, form of address and tone. Written so voice can read the same lines later. Plus our own reply model (facts to sentence, ~6.5M params) behind a fact checker, with templates as the fallback | ✅ |
| 25 | 🆕 **HUD interface** | Animated overlay with a status orb (idle / working / done / attention / warning) and live widgets (CPU, RAM, GPU and temperature, disk, battery, training jobs with time left, next reminders, last reply). Our own visual design; listening and speaking states arrive with voice | ✅ |
## Stage C: Runs and watches your machines
| # | Phase | What it adds | Status |
|---|---|---|---|
| 5 | **Full system control** | Window snapping and layouts, virtual desktops, media keys, brightness, Wi-Fi and Bluetooth, screenshot and screen record, shutdown and restart (confirmed) | ✅ |
| 28 | 🆕 **Smart play & mood** | Play anything: top match (checked against what you asked, never a random video), a channel's latest upload, playlists, live, channels, Spotify; unfamiliar names copied in full. Mood: "I'm feeling low" gets a kind reply and fitting music (offers first, plays on yes); self-harm words get a helpline, never music | ✅ |
| 23 | 🆕 **Monitoring & alerts** | Watches CPU and GPU load and temperatures, RAM, disk, battery, network, and long jobs like training runs. An anomaly model we train flags anything unusual, with alerts on the HUD and as notifications ("GPU is at 87°C, shall I pause training?"); spoken once voice lands. Includes pause / resume / status for training runs | ✅ |
| 24 | 🆕 **Briefings** | Morning and evening briefings in character: reminders due, system health, training results, Downloads clutter, what you worked on (time per app), commands run and alerts. Ask ("good morning", "brief me", "how was my day", "aaj ka plan kya hai") or let it arrive on its own once a day each (times in Settings). Shown in the command bar, on the HUD and as a notification; spoken once voice lands | ✅ |
| 14 | **Self-improving loop** | Fine-tunes the live model (about 20–30 min, not a 5-hour retrain) on fresh data from the data engine plus your /fix corrections ×20. Scores the live and new models on the same tests, your corrections and past mistakes; switches only if better and no riskier. Every version kept: "roll back the model" / improve.bat rollback. Runs on its own at night (idle, plugged in, something new to learn), Jarvis switches models without a restart, the morning briefing reports it. Models record what they learned, so the rules only cover what they haven't | ✅ |
## Stage D: Knows things, remembers, converses
| # | Phase | What it adds | Status |
|---|---|---|---|
| 26 | 🆕 **Offline knowledge Q&A** (parked: not needed now) | "Jarvis, who invented the telephone?" A local Wikipedia copy, a retriever built on our own embedding model (shared with phase 7), and an extractive reader model trained from scratch on open question-answering datasets plus synthetic questions. Answers cite the article. Works without internet | ⬜ |
| 7 | **Memory search** | "find the invoice pdf from last week", "the screenshot with the login error". Local full-text index (SQLite FTS5, BM25) of file names and contents (text, code, Office, PDF), screenshots and browser history, with type/time/where filters the command model extracts; clickable results. Our own OCR, trained from scratch on text rendered with your fonts (line finder ~0.5M + CRNN reader ~6M), reads screenshots into the index. Our own meaning model (a small bi-encoder trained on same-meaning pairs + your own files) finds "bill" for "invoice"; results merge word and meaning matches | ✅ |
| 6 | **Conversation & clarification** | Multi-turn follow-ups for commands ("no, the other folder", "do the same for Desktop"); asks back when unsure; small dialog-state model | 🔨 built: run `train_dialog.bat` |
| 27 | 🆕 **Jarvis-LM** | Our own small conversation model: a 100–300M-parameter decoder pretrained from scratch on open text (a few billion tokens, several nights on the 5070), then tuned on persona dialogues we generate. Handles small talk and personality, and hands commands to the command model (and questions to phase 26, if it is built later). Fluent for short exchanges, deliberately not a general-purpose chatbot | ⬜ |
| 10 | **Habit model** | Causal transformer over the activity log: proactive suggestions and workspace presets ("coding mode") | ⬜ |
| 8 | **Clipboard brain** | Searchable clipboard history; a detector that never stores passwords, OTPs or card numbers | ⬜ |
| 9 | **Smart file organizer** | Learns where you file things, auto-sorts Downloads, finds duplicate photos (perceptual hashing), suggests large files to clean up | ⬜ |
| 11 | **Notification filter** | Interrupt / batch / silence classifier with online learning and a digest | ⬜ |

## Stage E: Acts on its own (always with your OK)
| # | Phase | What it adds | Status |
|---|---|---|---|
| 12 | **Automations** | Scheduled and trigger-based rules built from the same DSL ("every Friday clean Downloads", "when a PDF arrives, move it"); previewable and undoable | ⬜ |
| 13 | **Screen understanding** | UI element detector trained on synthetic UI renders; safe step-by-step UI automation ("click Submit", "copy the error on screen") | ⬜ |

## Stage F: Voice
| # | Phase | What it adds | Status |
|---|---|---|---|
| 15 | **Voice** | Wake word "Jarvis" (tiny DS-CNN), streaming command speech recognition (small conformer with CTC), and our own text-to-speech: an acoustic model plus a neural vocoder trained on a public-domain speech dataset or your own recordings, in a calm butler-style voice. Reads persona replies, alerts, briefings and answers aloud. Always-on listening stays local | ⬜ |

## Stage G: Everywhere
| # | Phase | What it adds | Status |
|---|---|---|---|
| 16 | **Phone remote** | Control your PCs from your phone over LAN or a Telegram bot ("lock my PC", "is training done?") | ⬜ |
| 17 | **Two-PC mesh** | Laptop and RTX 5070 PC share corrections and models; heavy jobs go to the 5070 automatically. Deprioritized: the RTX 5070 PC is now the single main machine | ⬜ (last) |

## Stage H: Platform
| # | Phase | What it adds | Status |
|---|---|---|---|
| 18 | **Skill SDK** | Add an action in ~20 lines of Python plus example phrases; the data engine generates training data for it automatically | ⬜ |
| 19 | **Security center** | Per-action permissions, audit log viewer, panic stop hotkey, sandboxed shell actions | ⬜ |
| 20 | **Insights dashboard** | Weekly report: time per app, focus streaks, clutter, command usage | ⬜ |
| 21 | **Personal autocomplete** | Tiny GPT trained on your own writing; Hinglish-aware suggestions across Windows | ⬜ |

## Stage I: Robust and everyday (new)
| # | Phase | What it adds | Status |
|---|---|---|---|
| 29 | 🆕 **Reliability & self-healing** | Crash log and crash reports, automatic restart (safe mode after repeated crashes), one Jarvis at a time, `/doctor` health check ("run a health check") that tests every model, index, folder and setting and repairs what it can, self-repair of a broken config or memory index, background services restarted if they die, backup and restore of your Jarvis data | 🔨 built: try /doctor |
| 30 | 🆕 **Notes & journal** | "note: call the bank about the card", "what did I note about the bank", "add to my journal…"; notes are searchable with memory search and today's notes appear in the evening briefing | 🔨 built: "note: …", "what did I note about …"; `improve.bat` teaches the model |
| 31 | 🆕 **Workspaces** | "start coding mode" opens VS Code, Chrome and a lofi playlist and arranges the windows; "save this layout as work"; per-monitor window layouts restored with one command | 🔨 built: "save this layout as work", "start coding mode" |
| 32 | 🆕 **Focus & gaming mode** | Pomodoro timer on the HUD, do-not-disturb, optional blocking of distracting sites and apps (always confirmed, always undoable); a full-screen game switches on gaming mode by itself: training paused, HUD hidden, alerts held | 🔨 built: "focus for 45 minutes", "start a pomodoro", "gaming mode on" |
| 33 | 🆕 **Tasks & deadlines** | "add task finish the report by Friday", recurring tasks, overdue nudges; tasks on the HUD and in briefings; the command model learns task phrasing | 🔨 built: "add task … by friday", "what's due today", "done with …" |
| 34 | 🆕 **Wellbeing** | Break reminders (20-20-20 for the eyes, stretch after long sessions), optional screen-time limits per app, a late-night wind-down nudge; learns your normal hours from the activity log | 🔨 built: `wellbeing.py` engine (idle-aware 20-20-20 + stretch, 5-min away resets, per-app daily limits with 80% warning / optional minimize, wind-down from your median stop time), Settings → Wellbeing, evening-briefing line, `screen_time` / `breaks` actions with undo, rules + training data — run `improve.bat` |
| 35 | 🆕 **Spending from receipts** | Invoices and receipts in Downloads (PDF text or OCR) parsed by our own small field extractor into merchant, amount and date; "how much did I spend on Swiggy this month" | ⬜ |
| 36 | 🆕 **Answers from your documents** | "what's the due date in my rent agreement?" A reader model trained from scratch answers from your own indexed files (memory search finds the passage, the reader picks the answer) and shows the source. Unlike phase 26, it only uses your files | ⬜ |
| 37 | 🆕 **System care** | Disk space advisor (biggest folders, old installers, duplicate downloads), temp-file cleanup, startup apps you can switch off, GPU temperature history; everything previewed and undoable | 🔨 built: `care.py` + `care` action — disk report (free space, folders really using it, what can go), clear temp files / old installers / duplicate downloads (asks first, goes to Jarvis trash, undo brings it back), empty old Jarvis trash (asks; permanent), startup apps on/off the Task Manager way (undoable), GPU temperature history from the monitor's samples; rules + training data — run `improve.bat` |

## Stage J: Robustness (new)
| # | Phase | What it adds | Status |
|---|---|---|---|
| 38 | 🆕 **Context awareness** | Understands "this", "here", "that file", "these": the file selected in Explorer, the active window, the open folder, what you just copied, the last search result. "move this to archive", "open the second one", "send these to a new folder". The command model learns a `this`/`these` reference and Jarvis fills it from what's on screen | 🔨 built: try it from File Explorer; `improve.bat` teaches the model |
| 39 | 🆕 **Real-usage learning** | Learns from how you actually talk, not just generated data: every command you run (text only, on this PC) becomes a private test set; the model flags the ones it was least sure about and a 2-minute weekly review lets you confirm or fix them with one click; drift alerts when real-use confidence drops; each model version gets a report card on your real commands, not only generated tests | 🔨 built: try `review my commands`, `/usage` |
| 40 | 🆕 **Crash-safe actions** | Every file action is journaled before it starts, so a crash or power cut mid-move resumes or rolls back cleanly on restart; locked or in-use files, permission errors and full disks handled with a clear message and retry; each action is checked afterwards ("did all 12 files actually arrive?") and reported honestly | 🔨 built |
| 41 | 🆕 **Speed & footprint** | int8-quantized ONNX models (smaller, about 2× faster on the CPU), models loaded on first use, Jarvis idle under ~300 MB RAM, start-up under 2 s, a per-command time budget, `/perf` shows where the time goes; training and OCR yield the CPU/GPU when you're gaming or on battery | 🔨 built: run `quantize.bat`, then `/perf` |
| 42 | 🆕 **Release safety** | Every update is tested before it switches on: the full test suite plus an end-to-end run of every action in a sandbox folder; automatic rollback if Jarvis fails to start after an update; Windows UI tests for the command bar, HUD and settings; a proper installer (one `.exe`) for a clean PC | 🔨 built: `/update` tests first; `build_installer.bat` for the laptop |
| 43 | 🆕 **Your words** | Learns your vocabulary: your app nicknames, folder names, contacts, project names and your own typos and shortcuts become training data automatically; Hindi in Devanagari script and other Indian scripts (typed and in OCR) | 🔨 built: nicknames, typos, your apps/projects/folders as training data; Devanagari & other Indian scripts typed (OCR of them: later) |
| 44 | 🆕 **Any Windows setup** | Works the same on every machine: OneDrive-redirected Documents/Desktop/Pictures, multiple monitors and mixed DPI scaling, non-English Windows folder names, multiple user accounts, different browsers and default apps, laptops on battery; a first-run check that finds your real folders and apps | 🔨 built: `/setup`, try it on the laptop |

## Backlog (small items to fold into phases)
- Copy mechanism or bigger model if very long multi-path commands still mangle names in real use
