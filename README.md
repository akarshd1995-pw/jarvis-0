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
| `datagen/generate_intents.py` | Synthetic data engine for English and Hinglish, with typo noise and protected slots |
| `training/tokenizer.py` | Byte-level BPE tokenizer written from scratch |
| `training/model.py` | Encoder-decoder transformer written from scratch (RMSNorm, SwiGLU, tied embeddings, beam search) |
| `training/train.py` | Training loop: AdamW, warmup and cosine schedule, bf16, exact-match evaluation |

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

## Day 1: run the agent (laptop)
```bat
python -m jarvis            :: command bar; press Ctrl+Alt+J to toggle it
python -m jarvis --cli      :: terminal mode
```
Until you've trained a model, the regex rules handle common commands. You can also type raw DSL
directly, for example `list_files(src=Downloads, ext=pdf)`. Type `undo` to revert the last command.

## Night 1: train your model (RTX 5070)
```bat
train_night.bat
```
That script runs three steps:
1. `python -m datagen.generate_intents --n 300000` builds the synthetic dataset (about 30 s).
2. `python -m training.tokenizer --vocab 1024` trains the BPE tokenizer (a few seconds).
3. `python -m training.train --steps 15000` trains the model from random weights. Expect roughly 20–40 minutes on a 5070.
   The best checkpoint goes to `checkpoints/intent.pt`, selected by exact match on the validation set.

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
- [x] Phase 1: daemon, collectors, executors, safety layer, command bar
- [x] Phase 2: synthetic data engine, BPE tokenizer, transformer, training loop (train it tonight)
- [ ] Phase 2.5: grammar-constrained decoding (mask invalid tokens during beam search), ONNX export, test-set breakdown by action
- [ ] Phase 3: memory index (contrastive embedding model, hand-written HNSW, CRNN OCR for screenshots)
- [ ] Phase 4: routine model (causal transformer over the focus and file event log) and proactive suggestions
- [ ] Phase 5: notification triage (online classifier with contextual-bandit exploration)
- [ ] v2: voice (DS-CNN wake word and command ASR)
