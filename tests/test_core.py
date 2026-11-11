import os
import random
import time
from pathlib import Path

import pytest

from datagen.generate_intents import sample
from jarvis.config import Config
from jarvis.dsl.grammar import DSLError, canonical, parse
from jarvis.executor import Executor
from jarvis.nlu.rule_parser import rule_parse
from jarvis.safety import SafetyError, Tier
from jarvis.store import Store
from training.tokenizer import BPETokenizer


# ------------------------------------------------------------------ DSL
def test_parse_roundtrip_and_canonical():
    src = 'move_files(src=Downloads, dest=Archive, ext=pdf, older_than=30d) ; remind(text="call \\"mom\\"", at=18:05)'
    prog = parse(src)
    assert len(prog.calls) == 2
    assert prog.calls[1].args["text"] == 'call "mom"'
    assert canonical(str(prog)) == str(prog)
    assert str(prog).startswith("move_files(dest=Archive, ext=pdf, older_than=30d, src=Downloads)")


@pytest.mark.parametrize("bad", [
    "explode()", "move_files(src=Downloads)", "remind(text=\"x\")", "remind(text=\"x\", at=10:00, after=5m)",
    "system(action=format_c)", "open_app(name=chrome", "list_files(src=Downloads, ext=\"pdf\")",
    "remind(text=\"x\", at=25:00)", "open_url(url=\"a\", url=\"b\")",
])
def test_invalid_programs_rejected(bad):
    with pytest.raises(DSLError):
        parse(bad)


# ------------------------------------------------------------------ rules
@pytest.mark.parametrize("text,expected", [
    ("open chrome", "open_app(name=chrome)"),
    ("open github.com", 'open_url(url="github.com")'),
    ("open downloads", "open_folder(path=Downloads)"),
    ("remind me to call mom in 20 minutes", 'remind(after=20m, text="call mom")'),
    ("remind me to stretch at 6:30 pm", 'remind(at=18:30, text="stretch")'),
    ("move all pdfs older than 2 months from downloads to archive",
     "move_files(dest=Archive, ext=pdf, older_than=60d, src=Downloads)"),
    ("delete zip files from downloads", "delete_files(ext=zip, src=Downloads)"),
    ("open vscode and then remind me to push code in 2 hours",
     'open_app(name=vscode) ; remind(after=2h, text="push code")'),
    ("lock the screen", "system(action=lock)"),
])
def test_rule_parser(text, expected):
    prog = rule_parse(text)
    assert prog is not None, text
    assert str(prog) == expected


def test_rule_parser_unknown():
    assert rule_parse("write me a poem about rain") is None


# ------------------------------------------------------------------ executor + safety + undo
@pytest.fixture
def env(tmp_path):
    home = tmp_path / "home"
    dl, arc = home / "Downloads", home / "Archive"
    dl.mkdir(parents=True)
    old = time.time() - 40 * 86400
    for name in ["a.pdf", "b.pdf", "c.png", "invoice_1.pdf"]:
        (dl / name).write_text(name)
    for name in ["a.pdf", "c.png"]:
        os.utime(dl / name, (old, old))
    cfg = Config(data_dir=tmp_path / "data", folders={"Downloads": str(dl), "Archive": str(arc)},
                 allowed_roots=[str(home)])
    cfg.data_dir.mkdir()
    cfg.trash_dir.mkdir()
    return cfg, Store(cfg.db_path), dl, arc


def test_move_with_filters_and_undo(env):
    cfg, store, dl, arc = env
    asked = []
    ex = Executor(cfg, store, lambda tier, s, items: asked.append((tier, items)) or True)
    rep = ex.run(parse("move_files(src=Downloads, dest=Archive, ext=pdf, older_than=30d)"))
    assert not rep.errors
    assert asked[0][0] == Tier.CONFIRM and asked[0][1] == [f"a.pdf -> {arc.resolve()}"]
    assert (arc / "a.pdf").exists() and not (dl / "a.pdf").exists()
    ex.run(parse("undo()"))
    assert (dl / "a.pdf").exists() and not (arc / "a.pdf").exists()


def test_declined_confirmation_does_nothing(env):
    cfg, store, dl, arc = env
    ex = Executor(cfg, store, lambda *a: False)
    rep = ex.run(parse("delete_files(src=Downloads)"))
    assert rep.skipped and len(list(dl.iterdir())) == 4


def test_delete_is_recoverable(env):
    cfg, store, dl, arc = env
    ex = Executor(cfg, store, lambda *a: True)
    ex.run(parse('delete_files(src=Downloads, name_contains="invoice")'))
    assert not (dl / "invoice_1.pdf").exists()
    ex.run(parse("undo()"))
    assert (dl / "invoice_1.pdf").exists()


def test_path_outside_allowlist_blocked(env, tmp_path):
    cfg, store, dl, arc = env
    outside = tmp_path / "outside"
    outside.mkdir()
    ex = Executor(cfg, store, lambda *a: True)
    rep = ex.run(parse(f'list_files(src="{outside.as_posix()}")'))
    assert rep.errors and "outside the allowed" in rep.errors[0]


def test_reminder_scheduled(env):
    cfg, store, dl, arc = env
    ex = Executor(cfg, store, lambda *a: True)
    ex.run(parse('remind(text="stretch", after=5m)'))
    assert store.due_reminders(time.time()) == []
    assert store.due_reminders(time.time() + 301)[0][1] == "stretch"


# ------------------------------------------------------------------ datagen + tokenizer
def test_generated_programs_are_valid_and_slots_intact():
    from datagen.lexicon import KNOWN_SITES
    rng = random.Random(7)
    for split in ("train", "ood"):
        for _ in range(4000):
            ex = sample(rng, split)
            prog = parse(ex["program"])
            for call in prog.calls:
                for key in ("text", "query", "name_contains", "name", "src", "dest", "path", "url"):
                    v = call.args.get(key)
                    if isinstance(v, str) and not (key == "url" and v in KNOWN_SITES.values()):
                        assert v in ex["text"], (ex, key)


def test_splits_disjoint_and_ood_templates_held_out():
    from datagen.generate_intents import _norm, build
    splits, _ = build(3000, 500, seed=1, extra=None)
    train = {_norm(e["text"]) for e in splits["train"]}
    for name in ("val", "test_iid", "test_ood"):
        assert not train & {_norm(e["text"]) for e in splits[name]}, name
    assert any(e["program"] == "unknown()" for e in splits["train"])


def test_unknown_is_declined_politely(env):
    cfg, store, dl, arc = env
    rep = Executor(cfg, store, lambda *a: True).run(parse("unknown()"))
    assert "can't do that" in rep.text()


def test_brain_prefers_rules_over_model_unknown():
    from jarvis.nlu.brain import Brain

    class FakeModel:
        def __init__(self, out):
            self.out = out

        def parse(self, text):
            return (parse(self.out), 0.9) if self.out else (None, 0.2)

    b = Brain()
    b.model = FakeModel("unknown()")
    assert str(b.resolve("open chrome")[0]) == "open_app(name=chrome)"       # rules rescue
    assert str(b.resolve("write me a poem")[0]) == "unknown()"               # polite decline
    b.model = FakeModel(None)                                                # low confidence
    assert b.resolve("write me a poem") == (None, "unparsed")


def test_tokenizer_roundtrip():
    rng = random.Random(3)
    rows = [sample(rng) for _ in range(2000)]
    texts = [r["text"] for r in rows] + [r["program"] for r in rows]
    tok = BPETokenizer.train(texts, 600, verbose=False)
    for t in texts[:200] + ["Downloads se saari pdf hata do 🙂"]:
        assert tok.decode(tok.encode(t)) == t
