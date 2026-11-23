# Jarvis-0 roadmap

The default plan for growing Jarvis-0 from a command runner into a personal operating layer for
your PCs. Every model is trained from scratch on data we generate or collect locally.

**Recommended build order:** 4 → 5 → 14 → 7 → 6 → 12, then the rest.
- 4 and 5 make it a daily driver.
- 14 makes every day of use improve the model automatically.
- 7 and 6 are the biggest jumps in smartness.
- 12 turns it from reactive into proactive.

Status: ✅ done · 🔨 in progress · ⬜ planned

## Stage A: Foundation
| # | Phase | Status |
|---|---|---|
| 1 | **Agent core**: command bar (Ctrl+Alt+J), file, app, URL, folder, reminder and system actions; safety tiers, previews, confirmations, undo journal; focus and Downloads logging | ✅ |
| 2 | **Own command model**: synthetic English and Hinglish data engine, BPE tokenizer and encoder-decoder transformer from scratch, evaluation reports | ✅ |
| 3 | **Polish**: `search_web`, `unsupported` skipped-step reporting, grammar-constrained decoding, `/fix` `/good` `/wrong` corrections, negation safety data, 2M examples, 27M-param model | ✅ (retrain pending) |

## Stage B: Daily driver
| # | Phase | What it adds | Status |
|---|---|---|---|
| 4 | **Runtime & packaging** | ONNX model (no PyTorch needed), autostart with Windows, tray icon, settings window, encrypted local store, one-click update | ⬜ |
| 5 | **Full system control** | Window snapping and layouts, virtual desktops, media keys, brightness, Wi-Fi and Bluetooth, screenshot and screen record, shutdown and restart (confirmed) | ⬜ |
| 6 | **Conversation & clarification** | Multi-turn follow-ups ("no, the other folder", "do the same for Desktop"); asks back when unsure ("move them where?"); small dialog-state model | ⬜ |

## Stage C: Knows you
| # | Phase | What it adds | Status |
|---|---|---|---|
| 7 | **Memory search** | Find things across files, screenshots and browser history: contrastive embedding model, hand-written HNSW index, CRNN OCR | ⬜ |
| 8 | **Clipboard brain** | Searchable clipboard history; a detector that never stores passwords, OTPs or card numbers | ⬜ |
| 9 | **Smart file organizer** | Learns where you file things, auto-sorts Downloads, finds duplicate photos (perceptual hashing), suggests large files to clean up | ⬜ |
| 10 | **Habit model** | Causal transformer over the activity log: proactive suggestions and workspace presets ("coding mode") | ⬜ |
| 11 | **Notification filter** | Interrupt / batch / silence classifier with online learning and a digest | ⬜ |

## Stage D: Acts on its own (always with your OK)
| # | Phase | What it adds | Status |
|---|---|---|---|
| 12 | **Automations** | Scheduled and trigger-based rules built from the same DSL ("every Friday clean Downloads", "when a PDF arrives, move it"); previewable and undoable | ⬜ |
| 13 | **Screen understanding** | UI element detector trained on synthetic UI renders; safe step-by-step UI automation ("click Submit", "copy the error on screen") | ⬜ |
| 14 | **Self-improving loop** | The 5070 retrains on your corrections overnight; deploys only if accuracy improves and the dangerous-error rate doesn't rise; versioning and rollback | ⬜ |

## Stage E: New interfaces
| # | Phase | What it adds | Status |
|---|---|---|---|
| 15 | **Voice** | Custom wake word (DS-CNN), command ASR, small TTS for spoken confirmations | ⬜ |
| 16 | **Phone remote** | Control your PCs from your phone over LAN or a Telegram bot ("lock my PC", "is training done?") | ⬜ |
| 17 | **Two-PC mesh** | Laptop and RTX 5070 PC share corrections and models; heavy jobs go to the 5070 automatically | ⬜ |

## Stage F: Platform
| # | Phase | What it adds | Status |
|---|---|---|---|
| 18 | **Skill SDK** | Add an action in ~20 lines of Python plus example phrases; the data engine generates training data for it automatically | ⬜ |
| 19 | **Security center** | Per-action permissions, audit log viewer, panic stop hotkey, sandboxed shell actions | ⬜ |
| 20 | **Insights dashboard** | Weekly report: time per app, focus streaks, clutter, command usage | ⬜ |
| 21 | **Personal autocomplete** | Tiny GPT trained on your own writing; Hinglish-aware suggestions across Windows | ⬜ |

## Backlog (small items to fold into phases)
- Auto-play the first YouTube result (needs the API or browser control; today `search_web` opens the results page)
- Copy mechanism or bigger model if very long multi-path commands still mangle names in real use
