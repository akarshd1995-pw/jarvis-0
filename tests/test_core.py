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
    ("open youtube and play dear god", 'play_media(query="dear god", site=youtube)'),
    ("play dear god on youtube", 'play_media(query="dear god", site=youtube)'),
    ("play dear god", 'play_media(query="dear god", site=youtube)'),
    ("search dear god on youtube", 'search_web(query="dear god", site=youtube)'),
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


# ------------------------------------------------------------------ phase 4: ONNX runtime path (fake sessions)
def _fake_onnx(tmp_path, target_program, noise_token_text=None):
    """A fake encoder/decoder pair that 'wants' to emit target_program, optionally preferring a
    grammar-breaking token at every step, to exercise the numpy beam + constraints."""
    import numpy as np
    from jarvis.nlu.onnx_parser import ONNXParser
    from training.tokenizer import BOS, EOS, BPETokenizer

    rng = random.Random(2)
    texts = [sample(rng)["program"] for _ in range(1500)] + [target_program]
    tok = BPETokenizer.train(texts, 700, verbose=False)
    tok_path = tmp_path / "tok.json"
    tok.save(tok_path)
    target = tok.encode(target_program) + [EOS]
    noise = tok.encode(noise_token_text)[0] if noise_token_text else None
    V = tok.vocab_size

    class Enc:
        def run(self, names, feeds):
            return [np.zeros((1, feeds["src"].shape[1], 4), dtype=np.float32)]

    class Dec:
        def run(self, names, feeds):
            seqs = feeds["tgt_in"]
            out = np.full((seqs.shape[0], V), -20.0, dtype=np.float32)
            for i, s in enumerate(seqs):
                pos = len(s) - 1
                if pos < len(target):
                    out[i, target[pos]] = -0.05
                if noise is not None:
                    out[i, noise] = -0.01          # always the model's favourite, always invalid here
            return [out]

    return ONNXParser(None, None, tok_path, sessions=(Enc(), Dec()), min_confidence=0.0)


def test_onnx_parser_decodes_target(tmp_path):
    p = _fake_onnx(tmp_path, "move_files(dest=Archive, ext=pdf, src=Downloads)")
    prog, conf = p.parse("move pdfs from downloads to archive")
    assert str(prog) == "move_files(dest=Archive, ext=pdf, src=Downloads)" and conf > 0.5


def test_constraints_block_invalid_favourite_token(tmp_path):
    p = _fake_onnx(tmp_path, "open_app(name=chrome)", noise_token_text=";")
    prog, _ = p.parse("open chrome")
    assert str(prog) == "open_app(name=chrome)"


def test_brain_without_model_files_uses_rules(tmp_path):
    from jarvis.nlu.brain import Brain
    b = Brain(tmp_path / "intent.pt", tmp_path / "tokenizer.json")
    assert b.backend == "rules" and str(b.resolve("open chrome")[0]) == "open_app(name=chrome)"


# ------------------------------------------------------------------ phase 4: runtime
def test_store_payload_roundtrip_with_encryption_flag(tmp_path):
    for enc in (True, False):
        st = Store(tmp_path / f"db{enc}.db", encrypt=enc)
        st.log_event("focus", {"app": "code.exe"})
        raw = st._db.execute("SELECT payload FROM events").fetchone()[0]
        assert raw.split(":", 1)[0] in ("plain", "dpapi")
        assert st.events_since(0)[0][2] == {"app": "code.exe"}


def test_config_save_and_reload(tmp_path):
    cfg = Config(data_dir=tmp_path)
    cfg.hotkey, cfg.min_confidence = "ctrl+alt+k", 0.7
    cfg.folders["Projects"] = "D:/code"
    cfg.save()
    again = Config.load(tmp_path / "config.json")
    assert again.hotkey == "ctrl+alt+k" and again.min_confidence == 0.7
    assert again.folders["Projects"] == "D:/code"


def test_updater_refuses_non_git_folder(tmp_path):
    from jarvis.updater import update
    changed, msg = update(tmp_path)
    assert not changed and "isn't a git clone" in msg


def test_status_and_help_commands(env, tmp_path):
    from jarvis.daemon import Jarvis
    cfg, store, dl, arc = env
    j = Jarvis(cfg, lambda *a: True, data_root=tmp_path / "d")
    assert "Model:" in j.handle("/status")
    assert "/update" in j.handle("/help")


def test_fragments_never_conflict_with_lock_command():
    from datagen.generate_intents import Ctx, gen_fragment
    rng = random.Random(0)
    for _ in range(2000):
        _, t = gen_fragment(Ctx(rng, "train"))
        assert not t.lower().startswith("lock")



# ------------------------------------------------------------------ direct execution + play
def test_harmless_actions_run_without_asking(env):
    cfg, store, dl, arc = env
    asked = []
    ex = Executor(cfg, store, lambda *a: asked.append(a) or True)
    ex.run(parse("copy_files(src=Downloads, dest=Archive, ext=png)"))
    ex.run(parse('remind(text="stretch", after=5m)'))
    assert asked == []
    ex.run(parse("move_files(src=Downloads, dest=Archive, ext=png)"))   # move asks by default
    ex.run(parse("delete_files(src=Downloads, ext=png)"))               # delete always asks
    assert len(asked) == 2


def test_confirm_actions_are_configurable_but_delete_always_asks(env):
    cfg, store, dl, arc = env
    cfg.confirm_actions = []
    asked = []
    ex = Executor(cfg, store, lambda *a: asked.append(a) or True)
    ex.run(parse("move_files(src=Downloads, dest=Archive, ext=png)"))
    assert asked == []
    ex.run(parse("delete_files(src=Downloads, ext=png)"))
    assert len(asked) == 1


def test_play_media_opens_top_video(env, monkeypatch):
    import jarvis.actions.system as S
    from jarvis.actions.base import Ctx
    cfg, store, dl, arc = env
    opened = []
    monkeypatch.setattr(S.webbrowser, "open", lambda url: opened.append(url))
    html = 'x "videoRenderer":{"videoId":"abcDEF12345","thumb" y'
    r = S.play_media({"query": "dear god", "site": "youtube"}, Ctx(cfg, store, "b"), False, fetch=lambda u: html)
    assert opened[-1] == "https://www.youtube.com/watch?v=abcDEF12345" and "Playing" in r.summary
    S.play_media({"query": "dear god"}, Ctx(cfg, store, "b"), False, fetch=lambda u: 1 / 0)
    assert "results?search_query=dear+god" in opened[-1]


def test_brain_upgrades_play_requests_from_older_models():
    from jarvis.nlu.brain import upgrade_play
    p = upgrade_play(parse('search_web(query="dear god", site=youtube)'), "play dear god on youtube")
    assert str(p) == 'play_media(query="dear god", site=youtube)'
    p = upgrade_play(parse('search_web(query="dear god", site=youtube)'), "search dear god on youtube")
    assert str(p) == 'search_web(query="dear god", site=youtube)'


# ------------------------------------------------------------------ phase 22: persona
def _results():
    from jarvis.actions.base import ActionResult as AR
    fd = lambda **k: {"op": "move", "count": 14, "src": "Downloads", "dest": "Archive", "ext": "pdf",
                      "older_than": "30d", "newer_than": None, "name_contains": None, **k}
    return [
        AR("", kind="files", data=fd()), AR("", kind="files", data=fd(op="copy")),
        AR("", kind="files", data=fd(op="delete", dest=None)), AR("", kind="files", data=fd(count=0)),
        AR("", ["a.pdf", "b.pdf"], kind="list", data=fd(op="list", count=2, dest=None)),
        AR("", [], kind="list", data=fd(op="list", count=0, dest=None)),
        AR("", kind="open", data={"what": "app", "name": "chrome"}),
        AR("", kind="open", data={"what": "url", "name": "github.com"}),
        AR("", kind="open", data={"what": "folder", "name": "D:/work/college/notes/2025"}),
        AR("", kind="remind", data={"text": "call mom", "due": time.time() + 1200, "relative": True}),
        AR("", kind="remind", data={"text": "stretch", "due": time.time() + 86400, "relative": False}),
        *[AR("", kind="system", data={"action": a}) for a in ("mute", "volume_up", "volume_down", "lock", "sleep_display")],
        AR("", kind="system", data={"action": "mute", "unsupported_os": True}),
        AR("", kind="search", data={"query": "biryani", "site": "swiggy"}),
        AR("", kind="play", data={"query": "dear god", "site": "youtube"}),
        AR("", kind="play", data={"query": "lofi", "site": "spotify"}),
        AR("", kind="play", data={"query": "x", "site": "youtube", "fallback": True}),
        AR("", kind="memory", data={"query": "the blue chart"}),
        AR("", kind="unsupported", data={"request": "Send an email to Rahul"}),
        AR("", kind="unknown"), AR("", kind="undo", data={"count": 3}), AR("", kind="undo", data={"count": 0}),
        AR("", kind="undo", data={"count": 0, "cancelled": True}),
    ]


@pytest.mark.parametrize("tone", ["butler", "friendly", "minimal"])
def test_persona_renders_every_result_kind(tone):
    from jarvis.persona import Persona
    p = Persona(tone=tone, seed=0)
    for r in _results():
        out = p.result(r)
        assert out and "{" not in out and "}" not in out and "  " not in out, (tone, r.kind, out)
        assert out[0].isupper() or out[0].isdigit(), out


def test_persona_facts_are_exact():
    from jarvis.persona import Persona
    p = Persona(tone="minimal", seed=0)
    rs = _results()
    assert p.result(rs[0]) == "Moved 14 PDFs to Archive."
    assert "No PDFs older than 1 month in Downloads" in p.result(rs[3])
    unsupported = next(r for r in rs if r.kind == "unsupported")
    assert "send an email to Rahul" in p.result(unsupported)


def test_persona_avoids_repeating_itself():
    from jarvis.persona import Persona
    p = Persona(tone="butler", seed=3)
    outs = [p.result(_results()[0]) for _ in range(30)]
    assert all(a != b for a, b in zip(outs, outs[1:]))


def test_confirmation_question_comes_from_persona(env):
    from jarvis.daemon import Jarvis
    cfg, store, dl, arc = env
    asked = []
    j = Jarvis(cfg, lambda tier, q, items: asked.append(q) or False)
    reply = j.handle("move_files(src=Downloads, dest=Archive, ext=pdf)")
    assert asked and "PDF" in asked[0] and "Archive" in asked[0] and asked[0].endswith("?")
    assert "cancel" in reply.lower() or "left them" in reply.lower()


def test_smalltalk_and_details_toggle(env):
    import datetime as dt
    from jarvis.daemon import Jarvis
    from jarvis.persona import Persona, smalltalk
    p = Persona(tone="butler", address="sir", seed=0)
    now = dt.datetime(2026, 11, 23, 14, 37)
    assert smalltalk("jarvis what time is it", p, now) == "It's 2:37 PM, sir."
    assert smalltalk("hello", p, now).startswith("Good afternoon")
    assert smalltalk("write a poem about rain", p, now) is None
    cfg, store, dl, arc = env
    cfg.show_details = False
    j = Jarvis(cfg, lambda *a: True)
    assert "[" not in j.handle("thanks")


# ------------------------------------------------------------------ phase 22b: trained reply model
def test_checker_accepts_true_and_rejects_invented_facts():
    from jarvis.nlg.checker import check
    f = {"count": "fourteen", "files": "PDFs", "files_pl": "PDFs", "filt": " older than 1 month",
         "src": "Downloads", "dest": "Archive", "opverb": "move"}
    assert check("move", f, "butler", "sir", "Done, sir. Fourteen PDFs moved to Archive.")
    assert not check("move", f, "butler", "sir", "Done. Twelve PDFs moved to Archive.")        # wrong number
    assert not check("move", f, "butler", "sir", "Fourteen PDFs moved to Documents.")          # wrong folder
    assert not check("move", f, "butler", "sir", "Fourteen PDFs moved to Archive, ma'am.")     # wrong address
    assert not check("move", f, "butler", "sir", "Fourteen PDFs moved.")                       # missing destination
    assert not check("move", f, "butler", "", "Fourteen PDFs moved to Archive, sir.")          # address off


def test_every_reply_training_example_passes_its_own_fact_check():
    from datagen.generate_replies import make_example
    from jarvis.nlg.checker import check, linearize, parse_src
    rng = random.Random(11)
    for _ in range(5000):
        ex = make_example(rng)
        if ex:
            key, fields, tone, addr = parse_src(ex["src"])
            assert linearize(key, fields, tone, addr) == ex["src"]
            assert check(key, fields, tone, addr, ex["tgt"]), ex


def _fake_reply_model(tmp_path, sentence):
    import numpy as np
    from datagen.generate_replies import make_example
    from jarvis.nlg.generator import ReplyModel
    from training.tokenizer import EOS, BPETokenizer
    rng = random.Random(4)
    rows = [make_example(rng) for _ in range(1500)]
    tok = BPETokenizer.train([r["src"] for r in rows if r] + [r["tgt"] for r in rows if r] + [sentence], 900,
                             verbose=False)
    path = tmp_path / "rt.json"
    tok.save(path)
    target = tok.encode(sentence) + [EOS]

    class Enc:
        def run(self, names, feeds):
            return [np.zeros((1, feeds["src"].shape[1], 4), dtype=np.float32)]

    class Dec:
        def run(self, names, feeds):
            seqs = feeds["tgt_in"]
            out = np.full((seqs.shape[0], tok.vocab_size), -50.0, dtype=np.float32)
            for i, s in enumerate(seqs):
                out[i, target[min(len(s) - 1, len(target) - 1)]] = 0.0
            return [out]

    return ReplyModel(None, None, path, sessions=(Enc(), Dec()), seed=0)


def _move_result():
    from jarvis.actions.base import ActionResult as AR
    return AR("", kind="files", data={"op": "move", "count": 14, "src": "Downloads", "dest": "Archive", "ext": "pdf",
                                      "older_than": None, "newer_than": None, "name_contains": None})


def test_reply_model_output_used_when_facts_check_out(tmp_path):
    from jarvis.persona import Persona
    m = _fake_reply_model(tmp_path, "Right away. 14 PDFs are tucked into Archive.")
    p = Persona(tone="butler", address="", model=m, seed=0)
    assert p.result(_move_result()) == "Right away. 14 PDFs are tucked into Archive."
    assert p.last_source == "reply-model"


def test_reply_model_wrong_facts_fall_back_to_template(tmp_path):
    from jarvis.persona import Persona
    m = _fake_reply_model(tmp_path, "Right away. Twelve PDFs are tucked into Documents.")
    p = Persona(tone="minimal", address="", model=m, seed=0)
    assert p.result(_move_result()) == "Moved 14 PDFs to Archive."
    assert p.last_source == "template" and m.stats["rejected"] == m.stats["generated"] > 0
