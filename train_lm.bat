@echo off
REM Jarvis-LM (phase 27): our own chat model, trained from scratch. Run it each night; it continues where it
REM stopped (download -> tokenizer -> pretraining over several nights -> chat tuning) and stops at 07:30.
REM   train_lm.bat                  base model (~98M parameters, about 2-3 nights)
REM   train_lm.bat --preset small   ~34M, about one night
REM Watch progress: Get-Content checkpoints\lm_log.txt -Wait -Tail 5
call .venv\Scripts\activate
python -c "import pyarrow" 2>nul || pip install pyarrow
python -u -m training.lm.run %*
if %ERRORLEVEL%==3 echo Stopped for the day. Run train_lm.bat again tonight to continue.
if %ERRORLEVEL%==5 echo Trained, but below the quality bar: Jarvis keeps its fixed small-talk lines for now.
if %ERRORLEVEL%==0 echo Jarvis-LM is ready. Restart Jarvis to use it.
