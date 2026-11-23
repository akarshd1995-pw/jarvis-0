@echo off
REM Train the reply model (facts to in-character sentence), about 15 minutes on the RTX 5070
call .venv\Scripts\activate
echo [R1/4] generating reply training data...
python -u -m datagen.generate_replies --n 600000 --out data/replies || exit /b 1
echo [R2/4] training reply tokenizer...
python -u -m training.tokenizer --data data/replies/train.jsonl --fields src,tgt --vocab 2048 --out checkpoints/reply_tokenizer.json || exit /b 1
echo [R3/4] training reply model. Watch: Get-Content checkpoints\reply_log.txt -Wait -Tail 5
python -u -m training.train_replies --steps 20000 > checkpoints/reply_log.txt 2>&1
findstr /C:"done." /C:"fact-check pass" checkpoints\reply_log.txt
echo [R4/4] exporting reply model to ONNX...
python -u -m training.export_onnx --checkpoint checkpoints/reply.pt --prefix reply || echo Reply model export failed, Jarvis will keep using templates.
