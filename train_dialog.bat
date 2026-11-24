@echo off
REM Train the dialog model (phase 6): follow-ups ("no, to Desktop", "same for Pictures") and answers to
REM "did you mean ...?". Small model, about 15-20 minutes on the RTX 5070. Jarvis can keep running.
REM   train_dialog.bat          full run
REM   train_dialog.bat quick    5-minute smoke test
call .venv\Scripts\activate
if not exist checkpoints\tokenizer.json ( echo Train the command model first: train_night.bat & exit /b 1 )
set N=400000
set STEPS=20000
if /I "%1"=="quick" ( set N=40000 & set STEPS=1500 )
copy /Y checkpoints\tokenizer.json checkpoints\dialog_tokenizer.json >nul
echo [D1/3] generating dialog examples...
python -u -m datagen.generate_dialog --n %N% --out data/dialog || exit /b 1
echo [D2/3] training. Watch progress in another window with:
echo        Get-Content checkpoints\dialog_log.txt -Wait -Tail 5
python -u -m training.train --data data/dialog --tokenizer checkpoints/dialog_tokenizer.json --out checkpoints/dialog.pt --steps %STEPS% --d-model 256 --layers 4 --lr 6e-4 --eval-every 2000 --patience 3 > checkpoints\dialog_log.txt 2>&1
findstr /C:"done" /C:"best" checkpoints\dialog_log.txt
echo [D3/3] exporting to ONNX...
python -u -m training.export_onnx --checkpoint checkpoints/dialog.pt --tokenizer checkpoints/dialog_tokenizer.json --prefix dialog || exit /b 1
echo Done. Jarvis uses the dialog model after a restart.
