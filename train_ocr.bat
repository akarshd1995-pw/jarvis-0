@echo off
REM Phase 7 step 2: train our own OCR (line finder + reader) on text rendered with your fonts, then export.
REM   train_ocr.bat          full run: reader ~1.5-2 h, finder ~15-25 min
REM   train_ocr.bat quick    2-minute check that everything runs (models won't be good yet)
REM   train_ocr.bat finder   skip the reader (already trained): line finder + export only
REM Pause any time: "pause training" in Jarvis (or create checkpoints\PAUSE).
call .venv\Scripts\activate
set RSTEPS=40000
set FSTEPS=15000
if /I "%1"=="quick" (
  set RSTEPS=300
  set FSTEPS=200
)
if /I "%1"=="finder" goto finder
echo [1/3] training the OCR reader (%RSTEPS% steps). Watch: Get-Content checkpoints\ocr_log.txt -Wait -Tail 5
python -u -m training.ocr.train_reader --steps %RSTEPS% > checkpoints\ocr_log.txt 2>&1 || (type checkpoints\ocr_log.txt & exit /b 1)
:finder
echo [2/3] training the OCR line finder (%FSTEPS% steps)...
python -u -m training.ocr.train_finder --steps %FSTEPS% >> checkpoints\ocr_log.txt 2>&1 || (type checkpoints\ocr_log.txt & exit /b 1)
echo [3/3] exporting to ONNX...
python -u -m training.ocr.export || exit /b 1
findstr /C:"eval" checkpoints\ocr_log.txt
echo Done. Jarvis will read your screenshots in the background (restart Jarvis once).
