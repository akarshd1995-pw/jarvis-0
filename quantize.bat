@echo off
REM Phase 41: make smaller, faster int8 copies of your trained models (about a minute).
REM Each copy is checked against the full model and only kept if it gives the same answers.
REM New exports do this by themselves; run this once for models you trained before.
call .venv\Scripts\activate
python -c "import onnx" 2>nul || pip install onnx
python -u -m training.quantize
echo.
echo Restart Jarvis, then type /perf to see the models it loaded.
