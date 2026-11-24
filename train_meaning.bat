@echo off
REM Phase 7 step 3: train our own meaning model for memory search, then export it.
REM Learns from generated same-meaning pairs + your own indexed files (stays on this PC).
REM   train_meaning.bat          full run, about 20-40 min
REM   train_meaning.bat quick    2-minute check that everything runs
call .venv\Scripts\activate
set STEPS=20000
if /I "%1"=="quick" set STEPS=200
echo [1/2] training the meaning model (%STEPS% steps). Watch: Get-Content checkpoints\meaning_log.txt -Wait -Tail 5
python -u -m training.meaning.train --steps %STEPS% > checkpoints\meaning_log.txt 2>&1 || (type checkpoints\meaning_log.txt & exit /b 1)
echo [2/2] exporting to ONNX...
python -u -m training.meaning.export || exit /b 1
findstr /C:"eval" /C:"your files" checkpoints\meaning_log.txt
echo Done. Jarvis adds meaning to its index in the background; searches use it right away.
