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
    for split in ("train", "val_ood", "test_ood"):
        for _ in range(4000):
            ex = sample(rng, split)
            prog = parse(ex["program"])
            for call in prog.calls:
                for key in ("text", "query", "name_contains", "name", "src", "dest", "path", "url", "request"):
                    v = call.args.get(key)
                    if isinstance(v, str) and not (key == "url" and v in KNOWN_SITES.values()):
                        assert v in ex["text"], (ex, key)


def test_splits_disjoint_and_ood_templates_held_out():
    from datagen.generate_intents import _norm, build
    splits, _ = build(3000, 500, seed=1, extra=None)
    train = {_norm(e["text"]) for e in splits["train"]}
    for name in ("val", "val_ood", "test_iid", "test_ood"):
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


def test_held_out_frames_only_use_trained_words():
    from datagen.generate_intents import unseen_frame_words
    assert unseen_frame_words() == {}


def test_frames_without_file_type_never_label_one():
    from datagen.generate_intents import Ctx, gen_file_op
    rng = random.Random(0)
    for _ in range(300):
        c = Ctx(rng, "train")
        c.hing = False
        c.T = lambda templates: "what's in {src}{filt}"
        dsl, text, _ = gen_file_op(c, "list")
        assert "ext=" not in dsl, (dsl, text)


def test_jpg_and_jpeg_not_conflated():
    from datagen.lexicon import EXT_NICKNAMES
    assert "jpegs" not in EXT_NICKNAMES["jpg"]


# ------------------------------------------------------------------ phase 2.5
def test_search_web_and_unsupported(env):
    from jarvis.actions.system import search_url
    cfg, store, dl, arc = env
    assert search_url("dear god", "youtube") == "https://www.youtube.com/results?search_query=dear+god"
    ex = Executor(cfg, store, lambda *a: True)
    rep = ex.run(parse('unsupported(request="send an email to Rahul")'))
    assert "send an email to Rahul" in rep.text() and "can't" in rep.text()
    with pytest.raises(DSLError):
        parse('search_web(query="x", site=bing)')


@pytest.mark.parametrize("text,expected", [
    ("open youtube and play dear god", 'search_web(query="dear god", site=youtube)'),
    ("play dear god on youtube", 'search_web(query="dear god", site=youtube)'),
    ("google best biryani near me", 'search_web(query="best biryani near me", site=google)'),
])
def test_rule_parser_web(text, expected):
    assert str(rule_parse(text)) == expected


def test_viable_prefix_accepts_every_prefix_of_valid_programs():
    from jarvis.dsl.grammar import viable_prefix
    rng = random.Random(21)
    for _ in range(400):
        p = sample(rng)["program"]
        assert all(viable_prefix(p[:i]) for i in range(len(p) + 1)), p


@pytest.mark.parametrize("bad", ["open_ap(", "move_files(src=Downloads)", "system(action=format",
                                 'list_files(ext="pdf"', "open_app(name=chrome) open", "unknown(x",
                                 'search_web(query="a", site=bing', 'remind(text="x", at=10:00, after='])
def test_viable_prefix_rejects_impossible(bad):
    from jarvis.dsl.grammar import viable_prefix
    assert not viable_prefix(bad)


def test_negations_never_map_to_actions():
    from datagen.generate_intents import Ctx, gen_negation
    rng = random.Random(5)
    for _ in range(200):
        d, t = gen_negation(Ctx(rng, "train"))
        assert d == "unknown()"


def test_corrections_are_saved(env, tmp_path):
    from jarvis.daemon import Jarvis
    cfg, store, dl, arc = env
    j = Jarvis(cfg, lambda *a: True, data_root=tmp_path / "d")
    assert "No previous" in j.handle("/fix undo()")
    j.handle("list files in downloads")          # safe command: nothing is launched during tests
    assert "isn't valid" in j.handle("/fix open_app(")
    assert "Saved" in j.handle("/fix list_files(src=Downloads, ext=pdf)")
    j.handle("/good")
    rows = (tmp_path / "d" / "user_corrections.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 2
    assert "ext=pdf" in rows[0] and '"list_files(src=Downloads)"' in rows[1]
