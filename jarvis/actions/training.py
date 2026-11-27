"""Control training runs from Jarvis: pause / resume / status (phase 23).

Pausing works through a flag file (checkpoints/PAUSE) that the training loops check every step,
so a paused run keeps its GPU memory and continues exactly where it left off."""
from __future__ import annotations

from pathlib import Path

from ..hud.metrics import job_status
from .base import ActionResult, Ctx

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHECKPOINTS = PROJECT_ROOT / "checkpoints"
PAUSE_FLAG = CHECKPOINTS / "PAUSE"
JOB_LOGS = {"command model": CHECKPOINTS / "train_log.txt", "reply model": CHECKPOINTS / "reply_log.txt",
            "self-improvement": CHECKPOINTS / "improve_log.txt", "dialog model": CHECKPOINTS / "dialog_log.txt",
            "Jarvis-LM": CHECKPOINTS / "lm_log.txt"}


def start_improving(force: bool = True):
    """Run the self-improving loop in the background at low priority (phase 14). Returns the Popen or None."""
    import subprocess
    import sys
    if (CHECKPOINTS / "improve.lock").exists():
        return None
    flags = 0
    if sys.platform == "win32":
        flags = 0x00004000 | 0x08000000          # BELOW_NORMAL_PRIORITY_CLASS | CREATE_NO_WINDOW
    exe = sys.executable.replace("pythonw.exe", "python.exe")
    cmd = [exe, "-u", "-m", "training.loop"] + (["--force"] if force else [])
    out = open(CHECKPOINTS / "improve_stdout.txt", "w", encoding="utf-8")
    return subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=out, stderr=subprocess.STDOUT, creationflags=flags)


def jobs(running_within_s: float = 300) -> list[dict]:
    out = []
    for name, path in JOB_LOGS.items():
        st = job_status(path) if path.exists() else None
        if st and (st["state"] != "running" or st["age_s"] < running_within_s):
            out.append({"name": name, **st})
    return out


def training(args: dict, ctx: Ctx, dry_run: bool) -> ActionResult:
    action = str(args["action"])
    running = [j for j in jobs() if j["state"] == "running"]
    if action == "status":
        recent = jobs(running_within_s=300)
        return ActionResult(f"{len(running)} training run(s) active", kind="training",
                            data={"action": "status", "jobs": recent, "paused": PAUSE_FLAG.exists()})
    if dry_run:
        return ActionResult(f"Will {action} training", kind="training", data={"action": action})
    if action == "pause":
        if not running:
            return ActionResult("No training is running", kind="training", data={"action": "pause", "none": True})
        PAUSE_FLAG.parent.mkdir(parents=True, exist_ok=True)
        PAUSE_FLAG.write_text("paused by Jarvis\n", encoding="utf-8")
        return ActionResult("Training paused", kind="training", data={"action": "pause", "name": running[0]["name"]})
    if action == "improve":
        if (CHECKPOINTS / "improve.lock").exists():
            return ActionResult("Already improving", kind="training", data={"action": "improve", "running": True})
        if not (CHECKPOINTS / "intent.pt").exists():
            return ActionResult("No model to improve yet", kind="training", data={"action": "improve", "nomodel": True})
        start_improving(force=True)
        return ActionResult("Started improving the model", kind="training", data={"action": "improve"})
    if action == "rollback":
        import sys
        sys.path.insert(0, str(PROJECT_ROOT))
        from training.loop import rollback
        msg = rollback()
        ok = msg.startswith("Rolled back")
        return ActionResult(msg, kind="training", data={"action": "rollback", "ok": ok, "msg": msg,
                                                         "version": msg.split()[-1].rstrip(".") if ok else ""})
    was = PAUSE_FLAG.exists()
    PAUSE_FLAG.unlink(missing_ok=True)
    return ActionResult("Training resumed", kind="training", data={"action": "resume", "none": not was})


def wait_if_paused(flag: Path = PAUSE_FLAG, poll: float = 2.0) -> None:
    """Called by the training loops every step. Blocks while the flag file exists."""
    import time
    if not flag.exists():
        return
    print("  paused (delete checkpoints/PAUSE or say 'resume training' to continue)", flush=True)
    while flag.exists():
        time.sleep(poll)
    print("  resumed", flush=True)
