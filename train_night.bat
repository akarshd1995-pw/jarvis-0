@echo off
REM Training run on the RTX 5070
REM   train_night.bat          full run: 5M examples, 27M-param model, 120k steps, about 3-3.5 hours
REM   train_night.bat max      whole-night run: 8M examples, 200k steps, about 5-6 hours
REM   train_night.bat quick    quick run: 1M examples, 10M-param model, about 30 minutes
REM Pause any time: create the file checkpoints\PAUSE (or tell Jarvis "pause training"); delete it to resume.
call .venv\Scripts\activate
set N=5000000
set MODEL=--d-model 384 --layers 6 --steps 120000 --patience 6
set EST=about 3 hours
if /I "%1"=="max" (
  set N=8000000
  set MODEL=--d-model 384 --layers 6 --steps 200000 --patience 8
  set EST=about 5-6 hours
)
if /I "%1"=="quick" (
  set N=1000000
  set MODEL=--d-model 256 --layers 4 --steps 40000
  set EST=about 30 minutes
)
echo [1/5] generating %N% training examples (plus 5 x 20k eval examples)...
python -u -m datagen.generate_intents --n %N% --out data/intents || exit /b 1
echo [2/5] training tokenizer...
python -u -m training.tokenizer --data data/intents/train.jsonl --vocab 2048 --out checkpoints/tokenizer.json || exit /b 1
echo [3/5] training model, %EST%. Watch progress in another window with:
echo        Get-Content checkpoints\train_log.txt -Wait -Tail 5
python -u -m training.train --data data/intents --tokenizer checkpoints/tokenizer.json --out checkpoints/intent.pt %MODEL% > checkpoints/train_log.txt 2>&1
echo [4/5] evaluating...
python -u -m training.evaluate --split test_iid --errors checkpoints/errors_iid.jsonl > checkpoints/eval_report.txt 2>&1
python -u -m training.evaluate --split test_ood --errors checkpoints/errors_ood.jsonl >> checkpoints/eval_report.txt 2>&1
type checkpoints\eval_report.txt
echo [5/5] exporting to ONNX (needs: pip install onnx onnxruntime)...
python -u -m training.export_onnx --verify 200 || echo ONNX export skipped. Install onnx and onnxruntime, then run: python -m training.export_onnx --verify 200
echo Training the reply model next...
call train_replies.bat
