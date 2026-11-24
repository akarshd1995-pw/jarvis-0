"""Phase 14: the self-improving loop. Fine-tune the live command model, keep it only if it is better.

    python -m training.loop             # improve if there's something new to learn (new /fix corrections,
                                        #   or the data engine teaches things the live model doesn't know)
    python -m training.loop --force     # improve now regardless
    python -m training.loop --list      # model versions
    python -m training.loop --rollback  # back to the previous version (or --rollback v003)

One run (about 20-30 min on the RTX 5070):
  1. fresh data from the data engine (old + new skills, so the model doesn't forget) + your /fix corrections x20
  2. fine-tune from the live model: lower learning rate, ~15k steps, same tokenizer
  3. score live vs candidate on the SAME held-out tests + your corrections + past mistakes
  4. switch only if the candidate is better and no less safe; every live model is archived as a version
Jarvis notices the new model files and switches without a restart. The report feeds the morning briefing.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CK = ROOT / "checkpoints"
VERSIONS = CK / "versions"
LIVE = ["intent.pt", "intent_encoder.onnx", "intent_decoder.onnx", "intent_onnx.json"]
REPORT = CK / "improve_report.json"
LOG = CK / "improve_log.txt"
STATE = CK / "improve_state.json"
LOCK = CK / "improve.lock"
CORRECTIONS = ROOT / "data" / "user_corrections.jsonl"
REGRESSION = ROOT / "data" / "regression.jsonl"      # past mistakes with the right answer: must not come back

# a candidate may be at most this much worse on any test (noise), and must be better overall
EPS = 0.0015
DANGER_EPS = 0.0005


def log(msg: str) -> None:
    line = f"[{dt.datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def count_lines(path: Path) -> int:
    try:
        with open(path, encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0


# ------------------------------------------------------------------ deciding whether to run
def what_is_new() -> list[str]:
    """Reasons to improve: new corrections since the last run, or skills the live model wasn't trained on."""
    reasons = []
    state = _json(STATE, {})
    n = count_lines(CORRECTIONS)
    if n > state.get("corrections_seen", 0):
        reasons.append(f"{n - state.get('corrections_seen', 0)} new correction(s)")
    sys.path.insert(0, str(ROOT))
    from datagen.generate_intents import TEACHES
    knows = (_json(CK / "intent_onnx.json", {}) or {}).get("knows") or {}
    missing = [f for f in TEACHES if f not in knows.get("features", [])]
    if missing:
        reasons.append("new skills: " + ", ".join(missing))
    return reasons


# ------------------------------------------------------------------ versions
def versions() -> list[Path]:
    return sorted(p for p in VERSIONS.glob("v*") if p.is_dir()) if VERSIONS.exists() else []


def current_version() -> str | None:
    return _json(STATE, {}).get("live")


def archive_live(label: str, report: dict | None = None) -> Path:
    """Copy the live model files into versions/vNNN-<label>."""
    VERSIONS.mkdir(parents=True, exist_ok=True)
    n = len(versions())
    dest = VERSIONS / f"v{n:03d}-{label}"
    dest.mkdir()
    for f in LIVE:
        if (CK / f).exists():
            shutil.copy2(CK / f, dest / f)
    if report is not None:
        (dest / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return dest


def make_live(version_dir: Path) -> None:
    """Copy a version's files over the live ones (ONNX json last: Jarvis reloads when it changes)."""
    for f in LIVE:
        if (version_dir / f).exists():
            tmp = CK / (f + ".tmp")
            shutil.copy2(version_dir / f, tmp)
            tmp.replace(CK / f)
    state = _json(STATE, {})
    state["live"] = version_dir.name
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def rollback(target: str | None = None) -> str:
    vs = versions()
    if not vs:
        return "There are no earlier model versions yet."
    live = current_version()
    if target:
        match = [v for v in vs if v.name.startswith(target)]
        if not match:
            return f"No version called {target}. Versions: " + ", ".join(v.name for v in vs)
        dest = match[0]
    else:
        names = [v.name for v in vs]
        i = names.index(live) if live in names else len(vs)
        if i == 0:
            return f"{live} is already the oldest version."
        dest = vs[i - 1]
    make_live(dest)
    return f"Rolled back to {dest.name}."


# ------------------------------------------------------------------ scoring
def score_both(live_ckpt: Path, cand_ckpt: Path, data: Path, tokenizer: Path, limit: int) -> dict:
    import torch
    from training.evaluate import evaluate_rows, load_model, load_rows
    from training.tokenizer import BPETokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = BPETokenizer.load(tokenizer)
    sets = {"familiar": data / "test_iid.jsonl", "unseen": data / "test_ood.jsonl"}
    if count_lines(CORRECTIONS):
        sets["your corrections"] = CORRECTIONS
    if count_lines(REGRESSION):
        sets["past mistakes"] = REGRESSION
    out = {}
    for who, ck in (("live", live_ckpt), ("candidate", cand_ckpt)):
        model, cfg = load_model(ck, device)
        out[who] = {}
        for name, path in sets.items():
            r = evaluate_rows(model, tok, load_rows(path, tok, cfg, limit), device)
            out[who][name] = {k: r[k] for k in ("n", "exact", "dangerous", "missed")}
            if who == "live" and name in ("familiar", "unseen"):
                out.setdefault("live_errors", []).extend(r["errors"][:200])
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return out


# ------------------------------------------------------------------ report card on your real commands (phase 39)
def real_commands(limit: int = 3000) -> list[dict]:
    """Your recent real commands from Jarvis's private usage log (empty if learning is off or nothing yet)."""
    try:
        from jarvis.config import Config
        from jarvis.usage import UsageLog
        cfg = Config.load()
        if not cfg.db_path.exists():
            return []
        return UsageLog(cfg.db_path, encrypt=cfg.encrypt_logs).real_texts(limit=limit)
    except Exception as e:                                  # never block training on this
        log(f"(couldn't read your real commands: {e})")
        return []


def predict(model, tok, cfg, texts: list[str], device) -> list[str | None]:
    import torch
    from jarvis.dsl.grammar import DSLError, canonical
    from training.tokenizer import BOS, EOS
    from training.train import collate, eval_batches
    rows = [(tok.encode(t)[: cfg.max_src] or [EOS], [BOS, EOS], None, i) for i, t in enumerate(texts)]
    out: list[str | None] = [None] * len(texts)
    amp = dict(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")
    for batch in eval_batches(rows):
        src, src_pad, _, _ = collate(batch, device)
        with torch.no_grad(), torch.autocast(**amp):
            preds = model.greedy(src, src_pad, BOS, EOS)
        for p, row in zip(preds.tolist(), batch):
            raw = tok.decode(p[: p.index(EOS)] if EOS in p else p)
            try:
                out[row[3]] = canonical(raw)
            except DSLError:
                out[row[3]] = None
    return out


def _canon(p: str | None) -> str | None:
    from jarvis.dsl.grammar import DSLError, canonical
    try:
        return canonical(p) if p else None
    except DSLError:
        return None


def card(rows: list[dict], live: list, cand: list) -> dict:
    """Pure scoring (unit-tested). rows: {text, ran, label, status}; live/cand: each model's reading.
      same/changed   how many of your commands the new model reads differently
      reviewed       accuracy on the ones you confirmed or fixed in the weekly review
      everyday       agreement with what Jarvis ran and you didn't undo (a silver label)"""
    r = {"n": len(rows), "same": 0, "changed_n": 0, "changed": [], "labelled": 0, "labelled_right": 0,
         "labelled_right_live": 0, "silver": 0, "silver_live": 0, "silver_cand": 0}
    for row, a, b in zip(rows, live, cand):
        if a == b:
            r["same"] += 1
        else:
            r["changed_n"] += 1
            if len(r["changed"]) < 15:
                r["changed"].append({"text": row["text"], "live": a, "candidate": b})
        gold = _canon(row.get("label")) if row.get("status") in ("right", "fixed") else None
        if gold:
            r["labelled"] += 1
            r["labelled_right"] += b == gold
            r["labelled_right_live"] += a == gold
        elif row.get("status") == "new" and _canon(row.get("ran")) not in (None, "unknown()"):
            ran = _canon(row["ran"])
            r["silver"] += 1
            r["silver_live"] += a == ran
            r["silver_cand"] += b == ran
    return r


def real_card(live_ckpt: Path, cand_ckpt: Path, tokenizer: Path, rows: list[dict]) -> dict | None:
    if not rows:
        return None
    import torch
    from training.evaluate import load_model
    from training.tokenizer import BPETokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = BPETokenizer.load(tokenizer)
    preds = {}
    for who, ck in (("live", live_ckpt), ("candidate", cand_ckpt)):
        model, cfg = load_model(ck, device)
        preds[who] = predict(model, tok, cfg, [r["text"] for r in rows], device)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return card(rows, preds["live"], preds["candidate"])


SILVER_EPS = 0.02          # the new model may not drift away from what already works for you


def decide_real(real: dict | None) -> tuple[bool, str]:
    if not real:
        return True, ""
    if real["labelled"] and real["labelled_right"] < real["labelled_right_live"]:
        return False, (f"it got fewer of your reviewed commands right ({real['labelled_right']} vs "
                       f"{real['labelled_right_live']} of {real['labelled']})")
    if real["silver"] >= 30:
        lv, cd = real["silver_live"] / real["silver"], real["silver_cand"] / real["silver"]
        if cd < lv - SILVER_EPS:
            return False, (f"it would read {real['silver_live'] - real['silver_cand']} of your everyday commands "
                           f"differently from what already works ({cd:.0%} vs {lv:.0%})")
    return True, ""


def decide(scores: dict) -> tuple[bool, str]:
    live, cand = scores["live"], scores["candidate"]
    worse = [f"{k} {cand[k]['exact']:.2%} vs {live[k]['exact']:.2%}" for k in cand
             if cand[k]["exact"] < live[k]["exact"] - EPS]
    if worse:
        return False, "the new model was worse on " + "; ".join(worse)
    riskier = [k for k in cand if cand[k]["dangerous"] > live[k]["dangerous"] + DANGER_EPS]
    if riskier:
        return False, "the new model made more risky mistakes on " + ", ".join(riskier)
    gain = sum(cand[k]["exact"] - live[k]["exact"] for k in cand) / len(cand)
    if gain <= 0:
        return False, "the new model wasn't better overall"
    return True, f"better by {gain:+.2%} on average"


# ------------------------------------------------------------------ one run
def run(args) -> dict:
    py = sys.executable
    data = ROOT / "data" / "finetune"
    tokenizer = CK / "tokenizer.json"
    live_ckpt = CK / "intent.pt"
    cand = CK / "candidate" / "intent.pt"
    cand.parent.mkdir(parents=True, exist_ok=True)
    if not live_ckpt.exists() or not tokenizer.exists():
        raise SystemExit("No live model to improve yet: run train_night.bat once first.")
    report = {"started": dt.datetime.now().isoformat(timespec="seconds"), "reasons": args.reasons,
              "corrections": count_lines(CORRECTIONS)}
    seed = int(time.time()) % 100000
    log(f"generating {args.n:,} examples (old + new skills, plus your corrections)...")
    subprocess.run([py, "-u", "-m", "datagen.generate_intents", "--n", str(args.n), "--n-eval", str(args.n_eval),
                    "--out", str(data), "--seed", str(seed), "--extra", str(CORRECTIONS)], cwd=ROOT, check=True)
    log(f"fine-tuning from the live model for up to {args.steps:,} steps...")
    with open(LOG, "a", encoding="utf-8") as f:
        subprocess.run([py, "-u", "-m", "training.train", "--data", str(data), "--tokenizer", str(tokenizer),
                        "--out", str(cand), "--init", str(live_ckpt), "--steps", str(args.steps),
                        "--lr", str(args.lr), "--warmup", "300", "--eval-every", str(max(500, args.steps // 6)),
                        "--patience", "3"], cwd=ROOT, check=True, stdout=f, stderr=subprocess.STDOUT)
    log("scoring the live model and the new one on the same tests...")
    scores = score_both(live_ckpt, cand, data, tokenizer, args.limit)
    keep, why = decide(scores)
    log("report card: reading your real commands with both models...")
    real = real_card(live_ckpt, cand, tokenizer, real_commands())
    if real:
        log(f"  {real['n']} real commands: {real['same']} read the same, {real['changed_n']} differently; "
            f"reviewed {real['labelled_right']}/{real['labelled']} (live {real['labelled_right_live']}); "
            f"everyday {real['silver_cand']}/{real['silver']} (live {real['silver_live']})")
        ok_real, why_real = decide_real(real)
        if keep and not ok_real:
            keep, why = False, why_real
    report.update({"scores": {k: scores[k] for k in ("live", "candidate")}, "promoted": keep, "reason": why,
                   "real": real})
    _save_mistakes(scores.get("live_errors", []))
    if keep:
        if not versions():                                  # keep the original as v000 to roll back to
            make_live_state(archive_live("baseline"))
        log("exporting the new model to ONNX...")
        subprocess.run([py, "-u", "-m", "training.export_onnx", "--checkpoint", str(cand),
                        "--out", str(cand.parent), "--verify", "100", "--data", str(data / "val.jsonl")],
                       cwd=ROOT, check=True)
        vdir = VERSIONS / f"v{len(versions()):03d}-{dt.date.today():%Y%m%d}"
        vdir.mkdir(parents=True)
        shutil.copy2(cand, vdir / "intent.pt")
        for f in LIVE[1:]:
            shutil.copy2(cand.parent / f, vdir / f)
        (vdir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        make_live(vdir)
        report["version"] = vdir.name
        log(f"switched to {vdir.name}: {why}")
    else:
        log(f"kept the current model: {why}")
    state = _json(STATE, {})
    state["corrections_seen"] = report["corrections"]
    state["last_run"] = report["started"]
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    report["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def make_live_state(version_dir: Path) -> None:
    state = _json(STATE, {})
    state["live"] = version_dir.name
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _save_mistakes(errors: list[dict]) -> None:
    """The live model's mistakes on generated tests (with the right answer) become a regression set:
    future models must keep getting them right."""
    if not errors:
        return
    seen = set()
    if REGRESSION.exists():
        for line in REGRESSION.read_text(encoding="utf-8").splitlines():
            try:
                seen.add(json.loads(line)["text"])
            except (ValueError, KeyError):
                pass
    REGRESSION.parent.mkdir(parents=True, exist_ok=True)
    with open(REGRESSION, "a", encoding="utf-8") as f:
        for e in errors:
            if e["text"] not in seen and e.get("gold"):
                f.write(json.dumps({"text": e["text"], "program": e["gold"]}, ensure_ascii=False) + "\n")
                seen.add(e["text"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="improve even if nothing is new")
    ap.add_argument("--list", action="store_true", help="show model versions")
    ap.add_argument("--rollback", nargs="?", const="", default=None, help="previous version, or a name like v003")
    ap.add_argument("--n", type=int, default=1_000_000, help="fresh training examples")
    ap.add_argument("--n-eval", type=int, default=10_000)
    ap.add_argument("--steps", type=int, default=15_000)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--limit", type=int, default=10_000, help="examples per test set when scoring")
    args = ap.parse_args()
    CK.mkdir(parents=True, exist_ok=True)

    if args.list:
        live = current_version()
        for v in versions():
            r = _json(v / "report.json", {})
            note = r.get("reason", "starting point")
            print(f"{'*' if v.name == live else ' '} {v.name:22s} {note}")
        if not versions():
            print("No versions yet. The first improvement run archives the current model as v000.")
        return
    if args.rollback is not None:
        print(rollback(args.rollback or None))
        return

    args.reasons = what_is_new()
    if not args.reasons and not args.force:
        print("Nothing new to learn (no new corrections, no new skills). Use --force to run anyway.")
        return
    if LOCK.exists() and time.time() - LOCK.stat().st_mtime < 6 * 3600:
        raise SystemExit("An improvement run is already going.")
    LOCK.write_text(str(time.time()))
    LOG.write_text("", encoding="utf-8")
    try:
        log("improving because: " + (", ".join(args.reasons) or "forced"))
        report = run(args)
        print(json.dumps({k: report[k] for k in ("promoted", "reason")}, indent=2))
    finally:
        LOCK.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
