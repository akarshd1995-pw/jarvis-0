@echo off
REM Self-improving loop (phase 14): fine-tune the live command model on your corrections and the newest
REM skills, about 20-30 minutes. The new model goes live only if it scores better; every version is kept.
REM   improve.bat            improve now
REM   improve.bat list       show model versions
REM   improve.bat rollback   go back to the previous version
call .venv\Scripts\activate
if /I "%1"=="list" ( python -m training.loop --list & exit /b )
if /I "%1"=="rollback" ( python -m training.loop --rollback %2 & exit /b )
echo Fine-tuning from the live model. Watch progress in another window with:
echo        Get-Content checkpoints\improve_log.txt -Wait -Tail 5
python -u -m training.loop --force
