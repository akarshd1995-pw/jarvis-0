# Jarvis-0: a personal AI agent for Windows, built on models you train yourself

Jarvis-0 is a local, private desktop agent. You type a command in plain English or Hinglish.
A small transformer **you train from scratch** turns it into a typed action program, and a
safety-checked executor runs that program on your PC.

```
"Downloads se saari pdf jo 1 month se purani hain Archive me daal do aur 6 baje yaad dilana check karna"
        │  your intent model (~10M params, trained from random weights)
        ▼
move_files(dest=Archive, ext=pdf, older_than=30d, src=Downloads) ; remind(at=18:00, text="check karna")
        │  safety layer: tier check, dry-run preview, confirmation, undo journal
        ▼
files moved + reminder scheduled
```

## What's in this repo (phases 1 and 2)

| Path | What it does |
|---|---|
| `jarvis/dsl/grammar.py` | The action language: lexer, parser, schema and type validation, canonical form |
| `jarvis/safety.py` | Permission tiers (SAFE / CONFIRM / DESTRUCTIVE), folder allowlist, file-count cap |
| `jarvis/executor.py` | Runs programs: dry-run preview, then confirm, then execute, then journal the inverse |
| `jarvis/actions/` | File operations (move, copy, recoverable delete, list), apps, URLs, folders, reminders, Windows system controls |
| `jarvis/collectors/` | Foreground-app logger and Downloads watcher. These record the data for the phase 4 routine model |
| `jarvis/nlu/` | Text-to-program chain: raw DSL, then **your trained model**, then regex fallback rules |
| `jarvis/ui/` | Spotlight-style command bar (tkinter) and global hotkey (Win32, no extra packages) |
| `datagen/` | Synthetic data engine v2: English and Hinglish, number words, time expressions, open-vocabulary slots, coreference, out-of-scope examples, realistic noise, and a held-out test set of unseen phrasings |
| `training/tokenizer.py` | Byte-level BPE tokenizer written from scratch |
| `training/model.py` | Encoder-decoder transformer written from scratch (RMSNorm, SwiGLU, tied embeddings, beam search) |
| `training/train.py` | Training loop: AdamW, warmup and cosine schedule, bf16, exact-match evaluation on val and on unseen phrasings |
| `training/evaluate.py` | Detailed report: per-action accuracy, dangerous-error rate, missed-command rate, and a dump of every mistake |

Everything stays on your machine. Data lives in `%USERPROFILE%\.jarvis0\`.

## Setup

**Both PCs** (Python 3.11+ from python.org, which includes tkinter):
```bat
cd jarvis0
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**PyTorch.** Install the right build on each machine:
```bat
:: Night PC: RTX 5070 (Blackwell) needs a CUDA 12.8+ build
pip install torch --index-url https://download.pytorch.org/whl/cu128

:: Laptop: CPU build is plenty for running the model
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Run Jarvis without PyTorch (laptop)
After training, the night script exports the model to ONNX (`checkpoints/intent_encoder.onnx`, `intent_decoder.onnx`).
Copy those two files plus `tokenizer.json` and `intent_onnx.json` into the laptop's `checkpoints\` folder, then:
```powershell
pip install -r requirements.txt        # onnxruntime + numpy, tray icon; no PyTorch needed
python -m jarvis                       # shows "(onnx model)" in the startup log
python -m jarvis --autostart on        # start automatically when you log in (off: --autostart off)
```
- **Tray icon:** open the command bar, settings, check for updates, quit.
- **Settings window** (tray or `/settings`): hotkey, confidence gate, model backend, encryption, window titles, folder and app aliases, allowed folders, autostart.
- **`/update`:** `git pull --ff-only` and restart (never overwrites local changes). **`/status`:** model, accuracy, encryption, autostart.
- **Privacy:** the activity log is encrypted at rest with Windows DPAPI (only your Windows account can read it); full window titles aren't stored unless you turn that on.

## Day 1: run the agent (laptop)
```bat
python -m jarvis            :: command bar; press Ctrl+Alt+J to toggle it
python -m jarvis --cli      :: terminal mode
```
Until you've trained a model, the regex rules handle common commands. You can also type raw DSL
directly, for example `list_files(src=Downloads, ext=pdf)`. Type `undo` to revert the last command.

## Train your model (RTX 5070)
```powershell
.\train_night.bat          # full run: 2M examples, 27M-param model, about 2-3 hours (overnight)
.\train_night.bat quick    # quick run: 1M examples, 10M-param model, about 30 minutes
```
Watch live progress (with time remaining) from a second window:
```powershell
Get-Content checkpoints\train_log.txt -Wait -Tail 5
```
You can stop with Ctrl+C at any time: `checkpoints\intent.pt` always holds the best model so far.

**If training is much slower than expected:** check `gpu mem` in the log, or run `nvidia-smi`. When GPU memory is full, Windows spills into system RAM and training slows down several times without crashing. Lower the per-batch budget, e.g. `python -m training.train --max-tokens 16000`. You can also set NVIDIA Control Panel → Manage 3D settings → *CUDA - Sysmem Fallback Policy* → **Prefer No Sysmem Fallback**, so it stops with a clear out-of-memory error instead.
That script runs four steps:
1. `python -m datagen.generate_intents --n 2000000` builds 2M training examples plus `val`, `val_ood`, `test_iid` and `test_ood` sets (about 3 min).
2. `python -m training.tokenizer --vocab 2048` trains the BPE tokenizer (under a minute).
3. `python -m training.train` trains the model (~27M parameters: 6+6 layers, d_model 384) from random weights for up to 60,000 steps. Expect roughly 2–3 hours on a 5070 (the quick mode takes about 30 minutes).
   The best checkpoint goes to `checkpoints/intent.pt`. It is picked by the average of accuracy on familiar phrasing (`val`) and on unseen sentence structures (`val_ood`). Training stops early if neither improves for 6 checks in a row.
4. `python -m training.evaluate` writes `checkpoints/eval_report.txt` and every mistake to `checkpoints/errors_*.jsonl`.

## How the data engine keeps errors low
- **Program-first generation:** every example starts as a valid program, so labels are always correct.
- **Protected slots:** noise never touches the parts the model has to copy (reminder text, file names, URLs, paths, app names).
- **Open vocabulary:** reminder texts, unknown apps, URLs, file names and folder paths are generated by combining parts, so the model learns to copy spans instead of memorizing a fixed list.
- **Many ways to say it:** about 150 templates; number words in English and Hindi (`thirty`, `do ghante`, `aadha ghanta`); times like `half past 6 pm`, `shaam 6 baje` and `noon`; nicknames like `installers` and `word docs`.
- **Follow-ups:** "move pdfs to archive **and open it**", "list zips in downloads **then delete them**".
- **Out of scope:** about 8% of examples are chit-chat, questions or unsupported requests, labeled `unknown()`. The agent politely declines instead of guessing an action.
- **Messy input:** keyboard-adjacent typos, dropped letters, chat-speak (`pls`, `u`, `kro`), missing spaces, casing changes, emoji.
- **Held-out sentence structures:** 1 in 8 sentence frames is reserved for `val_ood` (used to pick the checkpoint) and another 1 in 8 for `test_ood` (the final score, never touched during training). Only the structure is held out. Every meaningful phrase ("last modified more than X ago", "throw away", "khali karo", "A -> B") is in shared lists that training sees, and a test fails if a held-out frame contains any word training never shows.
- **Ambiguity filter:** if the same text ever maps to two different programs, it is dropped everywhere.
- **Confidence gate:** at runtime, outputs below 55% confidence are not executed. Jarvis falls back to the rules or asks you to rephrase.
- **Grammar-constrained decoding:** during beam search, any token that would make the output an invalid program is discarded, so the model can only produce well-formed commands.
- **Safety data:** negations ("don't delete…", "mat hatao") and lone or garbled words always map to `unknown()`.

## Commands it understands
| Say | Program |
|---|---|
| "move pdfs older than a month from downloads to archive" | `move_files(...)` (with preview + confirm) |
| "mute", "volume up", "copy jpgs to D:/backup" | run immediately, no confirmation (change in Settings → *Ask before*; delete always asks) |
| "play dear god on youtube" / "open youtube and play X" / "play dear god" | `play_media(query="dear god", site=youtube)`: opens the top video so it starts playing |
| "search dear god on youtube" | `search_web(query="dear god", site=youtube)`: results page |
| "google best biryani near me", "search amazon for a usb c hub" | `search_web(..., site=google/amazon/...)` |
| "open chrome and send an email to Rahul" | `open_app(name=chrome) ; unsupported(request="send an email to Rahul")`: does what it can, tells you what it skipped |
| "don't delete my pdfs" | `unknown()`: never acts on a negation |

Supported search sites: google, youtube, amazon, flipkart, swiggy, zomato, github, stackoverflow, wikipedia, spotify, maps, reddit, linkedin.

## Persona
Every reply is in character and built from the real result, so counts, folders and skipped steps are always exact:
> "Fourteen PDFs are now in Archive, sir." · "Shall I move three ZIP files older than 2 months from Downloads to Archive?" · "I can't send an email to Rahul yet." · "It's 2:37 PM, sir."

- **Tones:** `butler` (default), `friendly`, `minimal`. Set the assistant's name and how it addresses you (sir, ma'am, your name, or nothing) in Settings.
- **Small talk without a language model:** greetings, time and date, thanks, "who are you", "what can you do", and a few jokes. Open conversation comes with Jarvis-LM (phase 27).
- **Details line:** the technical command shows under each reply (`[model 98%] move_files(...)  · reply: reply-model`). Turn it off in Settings once you trust it.

### Our own reply model (trained from scratch)
A second small model (~6.5M parameters) turns facts into a sentence in the chosen tone, so wording varies beyond fixed lines.
- **Data:** random situations are turned into facts by the same code the agent uses, then paired with phrasings from a bank of ~290 lines across 37 situations and 3 tones (`jarvis/nlg/bank.py`).
- **Fact checker** (`jarvis/nlg/checker.py`): a generated sentence is shown only if it states every required fact (count, file type, folders, names, time) and invents nothing (no extra numbers, folders, sites or forms of address). Otherwise Jarvis uses the template line.
- **Train it:** `.\train_replies.bat` (about 15 minutes; also runs automatically at the end of `train_night.bat`). Its log reports the **fact-check pass rate**.
- **Check it:** `/status` shows how many replies it generated and how many the checker rejected. Turn it off in Settings to use templates only.

## HUD
A translucent overlay in a screen corner (`Ctrl+Alt+H` to show or hide, or tray → Show / hide HUD, or `/hud`):
- **Status orb:** rotating rings and a breathing core. Cyan when idle, violet while working, green when done, amber when something went wrong, red for warnings (e.g. GPU over 85°C).
- **Gauges:** CPU, RAM, GPU (with temperature, from `nvidia-smi`), disk, and battery on laptops.
- **Jobs:** live progress bars for training runs, with time left, read from the training logs.
- **Up next** (your next reminders), the **last** reply, and a clock.

It's click-through by default (the mouse passes straight through it) and stays out of the taskbar. Change the corner, opacity and click-through in Settings. Built with tkinter and Windows APIs only, no extra packages.

## Teaching it your own phrasing
The fastest way: right in the command bar, after Jarvis gets something wrong, type
```
/fix move_files(src=Downloads, dest=Archive, ext=pdf, older_than=30d)
```
That saves your last command with the correct program. `/good` confirms a correct result, `/wrong` flags one to label later, `/help` lists these.
You can also edit the file directly:
`data/user_corrections.jsonl`:
```json
{"text": "dl ki saari purani pdf archive me", "program": "move_files(src=Downloads, dest=Archive, ext=pdf, older_than=30d)"}
```
The next `train_night.bat` run oversamples these 20×, so the model learns your style quickly.

Copy `checkpoints/` to the laptop and restart Jarvis. The command bar then shows `[model]` instead of `[rules]`.

## Safety model
- **SAFE** actions (open app, URL or folder; list; remind) run immediately.
- **CONFIRM** actions (move, copy, system controls, undo) show a dry-run preview and ask first.
- **DESTRUCTIVE** actions (delete) show a warning dialog. Deleted files go to Jarvis's own trash, so `undo` restores them.
- Every action is restricted to `allowed_roots` (your user folder by default) and capped at 500 files.
- Model output only runs if it parses **and** type-checks against the DSL schema.

## Customizing
Create `%USERPROFILE%\.jarvis0\config.json`:
```json
{
  "hotkey": "ctrl+alt+j",
  "folders": {"Projects": "D:/code", "Screenshots": "C:/Users/you/Pictures/Screenshots"},
  "apps": {"obsidian": "obsidian:", "discord": "discord:"},
  "watch_folders": ["Downloads", "Desktop"]
}
```
If you add folders or apps, add them to the vocab in `datagen/generate_intents.py` too, then retrain so the model learns them.

## Tests
```bat
pip install pytest
python -m pytest -q
```

## Roadmap
Jarvis-0 is heading toward a JARVIS-style assistant in the spirit of the Iron Man films: in-character replies, an animated HUD, machine monitoring, briefings, offline Q&A, a small conversation model of its own, and voice last. Everything is built 100% from scratch. The full plan lives in [ROADMAP.md](ROADMAP.md).

Phases 1–3 are done. Next, in order: **4 Runtime → 22 Persona → 25 HUD → 5 System control → 23 Monitoring → 24 Briefings → 14 Self-improving loop → 26 Offline Q&A → 7 Memory → 6 Conversation → 27 Jarvis-LM → 12 Automations → 15 Voice**.
