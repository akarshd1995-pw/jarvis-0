@echo off
REM Night training run on the RTX 5070
call .venv\Scripts\activate
python -m datagen.generate_intents --n 300000 --out data/intents || exit /b 1
python -m training.tokenizer --data data/intents/train.jsonl --vocab 1024 --out checkpoints/tokenizer.json || exit /b 1
python -m training.train --data data/intents --tokenizer checkpoints/tokenizer.json --out checkpoints/intent.pt --steps 15000 > checkpoints/train_log.txt 2>&1
echo Training finished. See checkpoints\train_log.txt
