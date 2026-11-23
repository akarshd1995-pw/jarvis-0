# Jarvis-0 roadmap

**Vision:** a JARVIS-style assistant in the spirit of the Iron Man films. It replies in character,
runs your machines, keeps watch over them, briefs you unprompted, remembers context, answers questions
offline, and lives in an animated HUD. Voice comes last, once everything else works. Our own design,
voice and name; no film assets or actor voice.

**Rule: 100% from scratch.** Every model is trained from random weights on data we generate or on
open datasets. No pretrained models, including for conversation: open questions are handled by
offline knowledge Q&A (phase 26) plus a small conversation model of our own (phase 27).

**Build order:** 4 → 22 → 25 → 5 → 23 → 24 → 14 → 26 → 7 → 6 → 27 → 12 → 15, then the rest.
- **4, 22, 25:** it feels like JARVIS. It runs properly, replies in character, and has the HUD.
- **5, 23, 24:** it runs and watches your machines, and briefs you.
- **14:** every day of use improves it automatically.
- **26, 7, 6, 27:** it knows things, remembers, and holds a conversation.
- **12:** it acts on its own, always with your OK.
- **15:** voice comes last and plugs into everything above (persona, alerts, briefings, Q&A, conversation).

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
| 4 | **Runtime & packaging** | ONNX model (no PyTorch needed), autostart with Windows, tray icon, settings window, encrypted local store, one-click update | 🔨 code done, needs first run on Windows |
| 22 | 🆕 **Persona layer** | In-character replies for every action ("Done. Fourteen PDFs are in Archive."), confirmations and errors. Configurable name, form of address and tone. Written so voice can read the same lines later | ⬜ |
| 25 | 🆕 **HUD interface** | Animated overlay with a status orb (idle / listening / thinking / speaking) and live widgets (system stats, reminders, running jobs, last action). Our own visual design | ⬜ |

## Stage C: Runs and watches your machines
| # | Phase | What it adds | Status |
|---|---|---|---|
| 5 | **Full system control** | Window snapping and layouts, virtual desktops, media keys, brightness, Wi-Fi and Bluetooth, screenshot and screen record, shutdown and restart (confirmed) | ⬜ |
| 23 | 🆕 **Monitoring & alerts** | Watches CPU and GPU load and temperatures, RAM, disk, battery, network, and long jobs like training runs. An anomaly model we train flags anything unusual, with alerts on the HUD and as notifications ("GPU is at 87°C, shall I pause training?"); spoken once voice lands | ⬜ |
| 24 | 🆕 **Briefings** | Morning and evening briefings on the HUD (spoken once voice lands): reminders due, system health, training results, Downloads clutter, what you worked on. Status updates when long jobs finish | ⬜ |
| 14 | **Self-improving loop** | The 5070 retrains on your corrections overnight; deploys only if accuracy improves and the dangerous-error rate doesn't rise; versioning and rollback | ⬜ |

## Stage D: Knows things, remembers, converses
| # | Phase | What it adds | Status |
|---|---|---|---|
| 26 | 🆕 **Offline knowledge Q&A** | "Jarvis, who invented the telephone?" A local Wikipedia copy, a retriever built on our own embedding model (shared with phase 7), and an extractive reader model trained from scratch on open question-answering datasets plus synthetic questions. Answers cite the article. Works without internet | ⬜ |
| 7 | **Memory search** | Find things across files, screenshots and browser history: contrastive embedding model, hand-written HNSW index, CRNN OCR | ⬜ |
| 6 | **Conversation & clarification** | Multi-turn follow-ups for commands ("no, the other folder", "do the same for Desktop"); asks back when unsure; small dialog-state model | ⬜ |
| 27 | 🆕 **Jarvis-LM** | Our own small conversation model: a 100–300M-parameter decoder pretrained from scratch on open text (a few billion tokens, several nights on the 5070), then tuned on persona dialogues we generate. Handles small talk and personality, and hands commands to the command model and questions to phase 26. Fluent for short exchanges, deliberately not a general-purpose chatbot | ⬜ |
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
| 17 | **Two-PC mesh** | Laptop and RTX 5070 PC share corrections and models; heavy jobs go to the 5070 automatically | ⬜ |

## Stage H: Platform
| # | Phase | What it adds | Status |
|---|---|---|---|
| 18 | **Skill SDK** | Add an action in ~20 lines of Python plus example phrases; the data engine generates training data for it automatically | ⬜ |
| 19 | **Security center** | Per-action permissions, audit log viewer, panic stop hotkey, sandboxed shell actions | ⬜ |
| 20 | **Insights dashboard** | Weekly report: time per app, focus streaks, clutter, command usage | ⬜ |
| 21 | **Personal autocomplete** | Tiny GPT trained on your own writing; Hinglish-aware suggestions across Windows | ⬜ |

## Backlog (small items to fold into phases)
- Auto-play the first YouTube result (needs the API or browser control; today `search_web` opens the results page)
- Copy mechanism or bigger model if very long multi-path commands still mangle names in real use
