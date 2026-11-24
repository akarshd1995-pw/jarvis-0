"""int8 copies of the ONNX models (phase 41): about 4x smaller and usually ~2x faster on the CPU.

    python -m training.quantize                 # every model in checkpoints/
    python -m training.quantize --only intent   # one model

Each copy is checked against the full-precision model before it's kept:
  * command / dialog models: the same program for >= 98.5% of validation sentences (data/<model>/val.jsonl),
    and the same next token for >= 97% of random inputs
  * reply model: the same next token for >= 97% of random inputs
  * meaning model: embeddings that point the same way (cosine >= 0.98)
A copy that fails is deleted, so Jarvis keeps using the full model. OCR models stay full precision.
Needs:  pip install onnx   (already there if you've exported a model)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

CK = Path("checkpoints")
SEQ2SEQ = {"intent": ("tokenizer.json", "data/intents/val.jsonl"),
           "dialog": ("dialog_tokenizer.json", "data/dialog/val.jsonl"),
           "reply": ("reply_tokenizer.json", None)}
AGREE_PROGRAMS, AGREE_TOKENS, MIN_COSINE = 0.985, 0.97, 0.98


def int8_path(p: Path) -> Path:
    return p.with_name(p.stem + ".int8" + p.suffix)


def quantize_file(src: Path) -> Path:
    from onnxruntime.quantization import QuantType, quantize_dynamic
    dst = int8_path(src)
    dst.unlink(missing_ok=True)
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
    return dst


def _sess(p: Path):
    import onnxruntime as ort
    o = ort.SessionOptions()
    o.intra_op_num_threads = 4
    return ort.InferenceSession(str(p), o, providers=["CPUExecutionProvider"])


def _vocab(tok_path: Path) -> int:
    try:
        from training.tokenizer import BPETokenizer
        return BPETokenizer.load(tok_path).vocab_size
    except Exception:
        return 1024


def token_agreement(enc: Path, dec: Path, vocab: int, n: int = 64, seed: int = 0) -> tuple[float, float, float]:
    """(share of identical next-token picks, fp32 ms, int8 ms) on random inputs."""
    rng = np.random.default_rng(seed)
    e32, d32, e8, d8 = _sess(enc), _sess(dec), _sess(int8_path(enc)), _sess(int8_path(dec))
    same, t32, t8 = 0, 0.0, 0.0
    for _ in range(n):
        s = rng.integers(4, vocab, size=(1, int(rng.integers(6, 40))), dtype=np.int64)
        pad = np.zeros_like(s, dtype=bool)
        t = rng.integers(4, vocab, size=(1, int(rng.integers(1, 12))), dtype=np.int64)
        t[0, 0] = 1                                                  # BOS
        a = time.perf_counter()
        m = e32.run(["mem"], {"src": s, "src_pad": pad})[0]
        lp32 = d32.run(["logprobs"], {"tgt_in": t, "mem": m, "src_pad": pad})[0]
        b = time.perf_counter()
        m8 = e8.run(["mem"], {"src": s, "src_pad": pad})[0]
        lp8 = d8.run(["logprobs"], {"tgt_in": t, "mem": m8, "src_pad": pad})[0]
        c = time.perf_counter()
        t32, t8 = t32 + b - a, t8 + c - b
        same += int(np.argmax(lp32[0]) == np.argmax(lp8[0]))
    return same / n, t32 / n * 1000, t8 / n * 1000


def program_agreement(enc: Path, dec: Path, tok: Path, data: Path, n: int = 400) -> float | None:
    """Share of validation sentences that give the same program with the int8 copy."""
    if not data.exists():
        return None
    from jarvis.nlu.onnx_parser import ONNXParser
    rows = [json.loads(l) for _, l in zip(range(n), open(data, encoding="utf-8"))]
    full = ONNXParser(enc, dec, tok, min_confidence=0.0, sessions=(_sess(enc), _sess(dec)))
    small = ONNXParser(enc, dec, tok, min_confidence=0.0, sessions=(_sess(int8_path(enc)), _sess(int8_path(dec))))
    same = 0
    for r in rows:
        a, _ = full.parse(r["text"])
        b, _ = small.parse(r["text"])
        same += str(a) == str(b)
    return same / max(1, len(rows))


def do_seq2seq(prefix: str, log=print, ck: Path | None = None) -> bool:
    ck = ck or CK
    enc, dec = ck / f"{prefix}_encoder.onnx", ck / f"{prefix}_decoder.onnx"
    tok_name, data = SEQ2SEQ[prefix]
    tok = ck / tok_name if (ck / tok_name).exists() else CK / tok_name   # a candidate shares the live tokenizer
    if not (enc.exists() and dec.exists() and tok.exists()):
        return False
    quantize_file(enc)
    quantize_file(dec)
    agree, ms32, ms8 = token_agreement(enc, dec, _vocab(tok))
    progs = program_agreement(enc, dec, tok, Path(data)) if data else None
    size32 = (enc.stat().st_size + dec.stat().st_size) / 2**20
    size8 = (int8_path(enc).stat().st_size + int8_path(dec).stat().st_size) / 2**20
    ok = agree >= AGREE_TOKENS and (progs is None or progs >= AGREE_PROGRAMS)
    log(f"{prefix}: {size32:.0f} MB -> {size8:.0f} MB, {ms32:.1f} -> {ms8:.1f} ms per step, "
        f"same next token {agree:.1%}" + (f", same program {progs:.1%}" if progs is not None else "")
        + ("  KEPT" if ok else "  NOT KEPT (differs too much; full model stays)"))
    if not ok:
        int8_path(enc).unlink(missing_ok=True)
        int8_path(dec).unlink(missing_ok=True)
    return ok


def do_meaning(log=print, ck: Path | None = None) -> bool:
    ck = ck or CK
    m, tok = ck / "meaning.onnx", ck / "meaning_tokenizer.json"
    if not (m.exists() and tok.exists()):
        return False
    quantize_file(m)
    s32, s8 = _sess(m), _sess(int8_path(m))
    rng = np.random.default_rng(0)
    vocab = _vocab(tok)
    cos = []
    for _ in range(48):
        ids = rng.integers(4, vocab, size=(4, int(rng.integers(6, 60))), dtype=np.int64)
        feeds = {"ids": ids, "keep": np.ones(ids.shape, np.float32)}
        a, b = s32.run(["vec"], feeds)[0], s8.run(["vec"], feeds)[0]
        a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
        b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
        cos.extend((a * b).sum(-1).ravel().tolist())
    c = float(np.mean(cos))
    ok = c >= MIN_COSINE
    log(f"meaning: {m.stat().st_size / 2**20:.0f} MB -> {int8_path(m).stat().st_size / 2**20:.0f} MB, "
        f"cosine {c:.3f}" + ("  KEPT" if ok else "  NOT KEPT"))
    if not ok:
        int8_path(m).unlink(missing_ok=True)
    return ok


def after_export(prefix: str, out_dir: Path, log=print) -> None:
    """Called by every export: drop the old int8 copy (it belongs to the previous model), then make a new one."""
    for f in (out_dir / f"{prefix}_encoder.onnx", out_dir / f"{prefix}_decoder.onnx", out_dir / f"{prefix}.onnx"):
        int8_path(f).unlink(missing_ok=True)
    try:
        import onnx  # noqa: F401
    except ImportError:
        log("(int8 copy skipped: pip install onnx to get smaller, faster models)")
        return
    try:
        if prefix in SEQ2SEQ:
            do_seq2seq(prefix, log, ck=out_dir)
        elif prefix == "meaning":
            do_meaning(log, ck=out_dir)
    except Exception as e:
        log(f"(int8 copy failed: {e}; the full model is used)")
        for f in out_dir.glob(f"{prefix}*.int8.onnx"):
            f.unlink(missing_ok=True)


def main():
    global CK
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["intent", "dialog", "reply", "meaning"])
    ap.add_argument("--checkpoints", type=Path, default=CK)
    args = ap.parse_args()
    CK = args.checkpoints
    try:
        import onnx  # noqa: F401
    except ImportError:
        raise SystemExit("Needs the onnx package:  pip install onnx")
    todo = [args.only] if args.only else ["intent", "dialog", "reply", "meaning"]
    for name in todo:
        try:
            found = do_meaning() if name == "meaning" else do_seq2seq(name)
            if not found:
                print(f"{name}: not trained yet, skipped")
        except Exception as e:                                   # one model failing never stops the others
            print(f"{name}: couldn't quantize ({e}); the full model stays")
    print("Done. The command model switches straight away; the others after a restart of Jarvis.")


if __name__ == "__main__":
    main()
