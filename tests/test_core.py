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


# ------------------------------------------------------------------ phase 25: HUD (logic; the window itself needs Windows)
def test_hud_parsers():
    from jarvis.hud.metrics import cpu_percent, parse_nvidia_smi, parse_progress
    g = parse_nvidia_smi("87, 8412, 12227, 64, NVIDIA GeForce RTX 5070")
    assert g["gpu"] == 87 and round(g["gpu_mem"]) == 69 and g["gpu_temp"] == 64
    assert parse_nvidia_smi("[N/A], x") is None
    p = parse_progress("step 100/60000 ...\nstep  23000/60000  loss 1.0  145 min elapsed, ~234 min left  gpu mem 8.4/12 GB")
    assert p == {"pct": 23000 / 60000 * 100, "eta_min": 234, "step": 23000, "steps": 60000}
    assert parse_progress("no steps here") is None
    assert cpu_percent((100, 1000), (150, 1100)) == 50.0 and cpu_percent(None, (1, 2)) is None


def test_hud_job_progress_only_for_active_logs(tmp_path):
    import os
    from jarvis.hud.metrics import job_progress
    log = tmp_path / "train_log.txt"
    log.write_text("step  600/60000  loss 1.2  1 min elapsed, ~69 min left  gpu mem 7.4/12 GB\n")
    assert job_progress({"Training": log})[0]["eta_min"] == 69
    old = time.time() - 3600
    os.utime(log, (old, old))
    assert job_progress({"Training": log}) == []


def test_upcoming_reminders_and_hud_state_hooks(env):
    from jarvis.daemon import Jarvis
    cfg, store, dl, arc = env
    j = Jarvis(cfg, lambda *a: True)
    events = []
    j.hooks["state"] = lambda s, hold=0.0: events.append(s)
    j.hooks["last"] = lambda t: events.append("last")
    j.handle('remind(text="stretch", after=5m)')
    assert events[0] == "thinking" and "success" in events and "last" in events
    assert j.store.upcoming_reminders(time.time())[0][1] == "stretch"
    events.clear()
    j.handle("list_files(src=D:/nowhere)")
    assert "error" in events
    assert "needs the command bar" in j.handle("/hud")


# ------------------------------------------------------------------ phase 5: system control
@pytest.mark.parametrize("text,expected", [
    ("take a screenshot", "screenshot(mode=full)"), ("next song", "media(action=next)"),
    ("set brightness to 60", "brightness(level=60)"), ("set volume to 30", "set_volume(level=30)"),
    ("turn off wifi", "radio(kind=wifi, state=off)"), ("shut down in 10 minutes", "power(action=shutdown, after=10m)"),
    ("cancel shutdown", "power(action=cancel)"), ("new desktop", "desktop(action=new)"),
    ("snap chrome to the left", "window(action=snap_left, app=chrome)"), ("minimize this", "window(action=minimize)"),
    ("close spotify", "window(action=close, app=spotify)"), ("show desktop", "window(action=minimize_all)"),
])
def test_rules_system_control(text, expected):
    assert str(rule_parse(text)) == expected


def test_system_control_confirmation_policy():
    from jarvis.safety import needs_confirmation as n
    assert n("power", None, {"action": "shutdown"}) and n("power", [], {"action": "restart"})   # always asks
    assert not n("power", None, {"action": "cancel"})
    assert n("window", None, {"action": "close"}) and not n("window", None, {"action": "snap_left"})
    assert n("radio", None, {"state": "off"}) and not n("radio", None, {"state": "on"})
    assert not n("media", None, {"action": "next"}) and not n("screenshot", None, {})


def test_brain_uses_rules_first_for_system_control():
    from jarvis.nlu.brain import Brain

    class Wrong:
        def parse(self, text):
            return parse('play_media(query="next song", site=youtube)'), 0.95

    b = Brain()
    b.model = Wrong()
    assert str(b.resolve("next song")[0]) == "media(action=next)"
    assert str(b.resolve("play dear god")[0]).startswith("play_media")


def test_system_control_on_non_windows_reports_politely(env):
    import sys
    from jarvis.daemon import Jarvis
    cfg, store, dl, arc = env
    reply = Jarvis(cfg, lambda *a: True).handle("media(action=next)")
    if sys.platform != "win32":
        assert "Windows" in reply


@pytest.mark.parametrize("tone", ["butler", "friendly", "minimal"])
def test_persona_system_control_replies(tone):
    from jarvis.actions.base import ActionResult as AR
    from jarvis.persona import Persona
    p = Persona(tone=tone, seed=0)
    rs = [AR("", kind="window", data={"action": a, "app": app}) for a in
          ("snap_left", "snap_right", "maximize", "minimize", "restore", "close", "minimize_all") for app in ("", "vscode")]
    rs += [AR("", kind="desktop", data={"action": a}) for a in ("new", "next", "previous", "close")]
    rs += [AR("", kind="media", data={"action": a}) for a in ("play_pause", "next", "previous", "stop")]
    rs += [AR("", kind="brightness", data={"level": 40, "action": a}) for a in ("set", "up", "down")]
    rs += [AR("", kind="volume_set", data={"level": 30}), AR("", kind="radio", data={"kind": "wifi", "state": "on"}),
           AR("", kind="radio", data={"kind": "bluetooth", "state": "off"})]
    rs += [AR("", kind="screenshot", data={"mode": m}) for m in ("full", "window", "region")]
    rs += [AR("", kind="record", data={"action": a}) for a in ("start", "stop")]
    rs += [AR("", kind="power", data={"action": a, "seconds": 600}) for a in ("shutdown", "restart", "sleep", "sign_out", "cancel")]
    for r in rs:
        out = p.result(r)
        assert out and "{" not in out and "  " not in out, (tone, r.kind, r.data, out)
    assert "VS Code" in p.result(AR("", kind="window", data={"action": "maximize", "app": "vscode"})) or tone == "minimal"
    q = p.confirm_question(AR("", kind="power", data={"action": "shutdown", "seconds": 600, "dry": True}))
    assert q.endswith("?") and ("ten minutes" in q or "10 minutes" in q)


# ------------------------------------------------------------------ phase 23: monitoring & alerts
def _snap(**k):
    from jarvis.hud.metrics import Snapshot
    base = dict(ts=1, cpu=10, ram=40, gpu=5, gpu_temp=45, disk_free_gb=200, battery=None, charging=None)
    return Snapshot(**{**base, **k})


def test_monitor_threshold_alerts_and_cooldown():
    from jarvis.hud.metrics import MetricsSampler
    from jarvis.monitor import Monitor
    m = Monitor(MetricsSampler({}), {}, lambda a: None)
    keys = {a.key for a in m.evaluate(_snap(gpu_temp=90, disk_free_gb=4, battery=12, charging=False), 0, 10)}
    assert keys == {"gpu_hot", "disk_low", "battery_low"}
    assert m.evaluate(_snap(gpu_temp=90), 60, 10) == []                 # cooldown
    assert [a.key for a in m.evaluate(_snap(gpu_temp=90), 60 + 31 * 60, 10)] == ["gpu_hot"]
    assert m.evaluate(_snap(battery=12, charging=True), 0, 10) == []    # charging: fine


def test_monitor_training_events(tmp_path):
    import os
    from jarvis.hud.metrics import MetricsSampler
    from jarvis.monitor import Monitor
    log = tmp_path / "train_log.txt"
    log.write_text("step  600/60000  loss 1.2  1 min elapsed, ~69 min left\n")
    m = Monitor(MetricsSampler({}), {"Training command model": log}, lambda a: None)
    assert m.evaluate(_snap(), 0, 10) == []                              # first sight: no alert
    log.write_text("step 60000/60000 ...\ndone. best score = 0.9984\n")
    done = m.evaluate(_snap(), 1, 10)
    assert done[0].kind == "training_done" and done[0].data["best"] == 0.9984
    log.write_text("step  700/60000  loss 1.2  1 min elapsed, ~69 min left\n")
    m.evaluate(_snap(), 2, 10)
    old = time.time() - 3600
    os.utime(log, (old, old))
    (tmp_path / "PAUSE").write_text("x")
    assert m.evaluate(_snap(), 3, 10) == []                              # paused on purpose: no stall alert
    (tmp_path / "PAUSE").unlink()
    assert m.evaluate(_snap(), 4, 10)[0].kind == "training_stalled"
    log.write_text("Traceback (most recent call last):\nRuntimeError: CUDA out of memory\n")
    crash = m.evaluate(_snap(), 5, 10)
    assert crash[0].kind == "training_crashed" and "out of memory" in crash[0].data["error"]


def test_baseline_learns_normal_and_flags_unusual():
    from jarvis.monitor.baseline import Baseline
    b = Baseline(min_samples=50, persist=3)
    for i in range(300):
        assert b.observe({"gpu": 8 + (i % 5), "cpu": 15}, hour=14) == []
    assert b.observe({"gpu": 97}, 14) == [] and b.observe({"gpu": 97}, 14) == []
    hit = b.observe({"gpu": 97}, 14)
    assert hit and hit[0][0] == "gpu" and hit[0][2] < 20
    assert b.observe({"gpu": 97}, 3) == []                               # other hours have no history yet


def test_training_pause_resume(monkeypatch, tmp_path, env):
    import jarvis.actions.training as T
    cfg, store, dl, arc = env
    log = tmp_path / "train_log.txt"
    log.write_text("step  600/60000  loss 1.2  1 min elapsed, ~69 min left\n")
    monkeypatch.setattr(T, "PAUSE_FLAG", tmp_path / "PAUSE")
    monkeypatch.setattr(T, "JOB_LOGS", {"command model": log})
    ex = Executor(cfg, store, lambda *a: True)
    ex.run(parse("training(action=pause)"))
    assert (tmp_path / "PAUSE").exists()
    st = ex.run(parse("training(action=status)")).results[0]
    assert st.data["paused"] and st.data["jobs"][0]["eta_min"] == 69
    ex.run(parse("training(action=resume)"))
    assert not (tmp_path / "PAUSE").exists()


@pytest.mark.parametrize("tone", ["butler", "friendly", "minimal"])
def test_persona_alerts(tone):
    from jarvis.monitor import Alert
    from jarvis.persona import Persona
    p = Persona(tone=tone, seed=0)
    alerts = [Alert("a", "gpu_hot", "critical", {"temp": 88, "training": t}) for t in (True, False)] + [
        Alert("a", "disk_low", "warning", {"free_gb": 6.2}), Alert("a", "battery_low", "warning", {"pct": 14}),
        Alert("a", "ram_high", "warning", {"pct": 95}),
        Alert("a", "training_done", "info", {"name": "Training command model", "best": 0.9984}),
        Alert("a", "training_crashed", "critical", {"name": "Training reply model", "error": "RuntimeError: boom"}),
        Alert("a", "training_stalled", "warning", {"name": "Training reply model", "minutes": 12}),
        Alert("a", "unusual", "info", {"metric": "gpu", "value": 97, "typical": 12})]
    for a in alerts:
        out = p.alert(a)
        assert out and "{" not in out and "  " not in out, (tone, a.kind, out)
    assert "99.8 percent" in p.alert(alerts[5])


def test_linearize_keeps_every_fact_the_persona_can_produce():
    from datagen.generate_replies import make_example
    from jarvis.nlg.checker import FIELD_ORDER
    from jarvis.persona import Persona
    import datagen.generate_replies as G
    rng = random.Random(2)
    p = Persona(seed=0)
    for _ in range(3000):
        key, fields = G.random_case(rng, p)
        missing = {k for k in fields if k[:1].islower() and k not in FIELD_ORDER and k != "summary"}
        assert not missing, (key, missing)


# ------------------------------------------------------------------ richer play: latest / playlist / live / channel
def _yt_fetch(url):
    if "sp=EgIQAg" in url:
        return 'x "channelRenderer":{"channelId":"UCAAAAAAAAAAAAAAAAAAAAAA","title"'
    if "feeds/videos.xml" in url:
        return ("<feed><title>Yes Theory</title><entry><yt:videoId>abcDEF12345</yt:videoId>"
                "<title>We Lived With Strangers &amp; Friends</title></entry></feed>")
    if "sp=EgIQAw" in url:
        return '"playlistRenderer":{"playlistId":"PLxyz123","title"'
    if "sp=EgJAAQ" in url:
        return '"videoRenderer":{"videoId":"LIVEvid1234",'
    return '"videoRenderer":{"videoId":"TOPvid12345",'


@pytest.mark.parametrize("mode,expect", [
    ("latest", "watch?v=abcDEF12345"), ("playlist", "list=PLxyz123"), ("live", "watch?v=LIVEvid1234"),
    ("channel", "/channel/UCAAAAAAAAAAAAAAAAAAAAAA"), ("top", "watch?v=TOPvid12345")])
def test_play_modes_open_the_right_thing(env, monkeypatch, mode, expect):
    import jarvis.actions.system as S
    from jarvis.actions.base import Ctx
    cfg, store, dl, arc = env
    opened = []
    monkeypatch.setattr(S.webbrowser, "open", lambda url: opened.append(url))
    r = S.play_media({"query": "yes theory", "site": "youtube", "mode": mode}, Ctx(cfg, store, "b"), False,
                     fetch=_yt_fetch)
    assert expect in opened[-1] and not r.data.get("fallback")
    if mode == "latest":
        assert r.data["title"] == "We Lived With Strangers & Friends" and r.data["channel"] == "Yes Theory"


def test_play_latest_falls_back_to_results(env, monkeypatch):
    import jarvis.actions.system as S
    from jarvis.actions.base import Ctx
    cfg, store, dl, arc = env
    opened = []
    monkeypatch.setattr(S.webbrowser, "open", lambda url: opened.append(url))
    r = S.play_media({"query": "301 diaries", "mode": "latest"}, Ctx(cfg, store, "b"), False, fetch=lambda u: 1 / 0)
    assert r.data["fallback"] and "results?search_query=301+diaries" in opened[-1]


@pytest.mark.parametrize("text,expected", [
    ("play 301 diaries latest video", 'play_media(mode=latest, query="301 diaries", site=youtube)'),
    ("play the latest video from yes theory", 'play_media(mode=latest, query="yes theory", site=youtube)'),
    ("carryminati ki latest video chalao", 'play_media(mode=latest, query="carryminati", site=youtube)'),
    ("play a lofi playlist", 'play_media(mode=playlist, query="lofi", site=youtube)'),
    ("play lofi girl live", 'play_media(mode=live, query="lofi girl", site=youtube)'),
    ("open yes theory channel", 'play_media(mode=channel, query="yes theory", site=youtube)'),
])
def test_rules_play_modes(text, expected):
    assert str(rule_parse(text)) == expected


def test_old_model_output_is_normalized():
    from jarvis.nlu.brain import upgrade_play
    for raw, text, want in [
        ('play_media(query="301 diaries latest video", site=youtube)', "play 301 diaries latest video",
         'play_media(mode=latest, query="301 diaries", site=youtube)'),
        ('search_web(query="yes theory latest video", site=youtube)', "play yes theory latest video",
         'play_media(mode=latest, query="yes theory", site=youtube)'),
        ('play_media(query="dhruv rathee ka latest video", site=youtube)', "dhruv rathee ka latest video chalao",
         'play_media(mode=latest, query="dhruv rathee", site=youtube)'),
        ('play_media(query="dear god", site=youtube)', "play dear god", 'play_media(query="dear god", site=youtube)'),
    ]:
        assert str(upgrade_play(parse(raw), text)) == want


def test_persona_play_latest_mentions_channel_and_title():
    from jarvis.actions.base import ActionResult as AR
    from jarvis.persona import Persona
    for tone in ("butler", "friendly", "minimal"):
        out = Persona(tone=tone, seed=0).result(AR("", kind="play", data={
            "query": "yes theory", "site": "youtube", "mode": "latest", "title": "We Lived With Strangers",
            "channel": "Yes Theory"}))
        assert "Yes Theory" in out and "We Lived With Strangers" in out


# ---------------------------------------------------------------- free-text slots must be copied whole
def test_slot_faithful_catches_truncated_names():
    from jarvis.nlu.brain import slot_faithful
    assert not slot_faithful(parse('play_media(query="sup", site=youtube)'), "play super machi")
    assert slot_faithful(parse('play_media(query="super machi", site=youtube)'), "play super machi")
    assert slot_faithful(parse('play_media(mode=latest, query="Yes Theory", site=youtube)'), "play Yes Theory's newest video")


def test_brain_repairs_model_that_cuts_the_name():
    from jarvis.nlu.brain import Brain

    class Fake:
        def __init__(self, out): self.out = out
        def parse(self, text): return parse(self.out), 0.9

    for model_out in ['play_media(query="sup", site=youtube)', 'play_media(query="super", site=youtube)']:
        b = Brain(); b.model = Fake(model_out)
        prog, _ = b.resolve("play super machi")
        assert str(prog) == 'play_media(query="super machi", site=youtube)'
    b = Brain(); b.model = Fake('play_media(query="dear god", site=youtube)')
    assert str(b.resolve("play dear god song")[0]) == 'play_media(query="dear god", site=youtube)'  # dropping "song" is fine


def test_hinglish_and_polite_play_rules():
    from jarvis.nlu.rule_parser import rule_parse
    for text in ["super machi chalao", "super machi song bajao", "can you play super machi please", "jarvis play super machi"]:
        assert str(rule_parse(text)) == 'play_media(query="super machi", site=youtube)', text


# ---------------------------------------------------------------- mood -> kind reply + fitting music
def test_mood_detection():
    from jarvis.persona import mood
    assert mood.detect("i am feeling low, play some music to lift my mood").wants_music
    assert mood.detect("mood kharab hai kuch gaana lagao").playlist == "feel good hindi songs"
    assert mood.detect("i'm so stressed, play something").kind == "stressed"
    assert not mood.detect("my mood is bad").wants_music                  # offer first, play on "yes"
    assert mood.detect("i want to die").kind == "crisis"
    for t in ["play sad songs", "play dear god", "open chrome", "i'm down to play some games", "what's up"]:
        assert mood.detect(t) is None, t


def test_mood_flow_plays_and_offers(env, tmp_path):
    from jarvis.daemon import Jarvis
    cfg, store, dl, arc = env
    j = Jarvis(cfg, lambda *a: True, data_root=tmp_path / "d")
    ran = []
    real_run = j.executor.run
    j.executor.run = lambda prog: (ran.append(str(prog)), real_run(parse("list_files(src=Downloads)")))[1]
    j.handle("i am feeling low, play some music to lift my mood")
    assert ran[-1] == 'play_media(mode=playlist, query="feel good happy songs", site=youtube)'
    reply = j.handle("my mood is bad")
    assert "yes" in reply.lower() and len(ran) == 1                       # offered, nothing played yet
    j.handle("yes")
    assert len(ran) == 2
    assert "14416" in j.handle("i want to die") and len(ran) == 2         # never a playlist


def test_mood_in_training_data():
    import random
    from datagen.generate_intents import gen_mood, Ctx
    from jarvis.persona import mood
    rng = random.Random(0)
    for _ in range(500):
        d, t = gen_mood(Ctx(rng, "train"))
        prog = parse(d)                                                    # valid DSL
        assert mood.detect(t) is None or mood.detect(t).kind != "crisis"  # never self-harm text in the data
    m = mood.from_program(parse("mood(kind=stressed, music=yes)"), "bahut tension hai, kuch gaana lagao")
    assert m.wants_music and m.hindi and m.playlist == "soothing hindi songs"


# ---------------------------------------------------------------- play: never a random video for an unknown name
def _yt_page(*vids):
    return "x".join(f'"videoRenderer":{{"videoId":"{v}","title":{{"runs":[{{"text":"{t}"}}]}},'
                    f'"ownerText":{{"runs":[{{"text":"{c}"}}]}}' for v, t, c in vids)


def test_best_video_matches_what_was_asked():
    from jarvis.actions import youtube as yt
    f = lambda *v: (lambda u: _yt_page(*v))
    two = f(("a" * 11, "Pavizha Mazha | Athiran", "Muzik247"), ("b" * 11, "Pavazha Malli Full Video Song", "Sun Music"))
    assert yt.best_video("pavazha malli", two)[:2] == ("ok", "b" * 11)          # the matching one, not the first
    assert yt.best_video("pavazha malli", f(("a" * 11, "Pavizha Mazha", "M")))[0] == "nomatch"
    assert yt.best_video("qwzlorp fandango", f(("d" * 11, "Bohemian Rhapsody", "Queen")))[0] == "nomatch"
    assert yt.best_video("kesariyaa", f(("k" * 11, "Kesariya - Brahmastra", "Sony")))[0] == "ok"   # spelling variant
    assert yt.best_video("how to make biryani", f(("j" * 11, "Chicken Biryani Recipe", "Lab")))[0] == "ok"
    assert yt.best_video("funny cat videos", f(("c" * 11, "Cats being cats compilation", "Ch")))[0] == "ok"
    assert yt.best_video("x", lambda u: 1 / 0)[0] == "offline"


def test_play_unknown_song_shows_results_not_random(env, monkeypatch):
    import jarvis.actions.system as S
    from jarvis.actions.base import Ctx
    cfg, store, dl, arc = env
    opened = []
    monkeypatch.setattr(S.webbrowser, "open", lambda url: opened.append(url))
    page = _yt_page(("d" * 11, "Bohemian Rhapsody (Official Video)", "Queen"))
    r = S.play_media({"query": "qwzlorp fandango"}, Ctx(cfg, store, "b"), False, fetch=lambda u: page)
    assert "results?search_query=qwzlorp+fandango" in opened[-1] and r.data["reason"] == "nomatch"
    from jarvis.persona.persona import Persona
    reply = Persona("Jarvis", "sir", "butler").result(r)
    assert "couldn't find" in reply.lower() and "Bohemian Rhapsody" in reply


# ---------------------------------------------------------------- copy guard: quoted values come from the input
def test_copy_guard():
    from jarvis.nlu.copy_guard import copy_ok
    src = "Please pull up website datasegekot.in and play super machi"
    assert copy_ok('open_url(url="datase', src) and copy_ok('open_url(url="datasegekot.in")', src)
    assert not copy_ok('open_url(url="dudege', src)                       # garbled name: impossible
    assert copy_ok('open_url(url="zomato.c', "open zomato")                # known site address is allowed
    assert not copy_ok('play_media(query="super machi pe pe', src)        # no loops
    assert copy_ok('move_files(src="D:\\\\yides_pro\\\\mi', r"copy from D:\yides_pro\misc to desktop")
    assert copy_ok('play_media(query="சூப்', "play சூப்பர் மச்சி")         # multi-byte characters
    assert copy_ok('play_media(query="\ufffd', "play சூப்பர்")            # half-decoded character
    assert not copy_ok('play_media(query="Abhi and niyu")', "open the Abhi and Niyu channel")  # exact case


# ---------------------------------------------------------------- briefings (phase 24)
def test_briefing_rules():
    from jarvis.nlu.rule_parser import rule_parse
    assert str(rule_parse("good morning")) == "briefing(kind=morning)"
    assert str(rule_parse("aaj ka plan kya hai")) == "briefing(kind=morning)"
    assert str(rule_parse("how was my day")) == "briefing(kind=evening)"
    assert str(rule_parse("aaj maine kya kiya")) == "briefing(kind=evening)"
    assert str(rule_parse("brief me")) == "briefing()"


def test_briefing_gather_and_render(tmp_path):
    import os
    from datetime import datetime
    from jarvis import briefing as B
    from jarvis.hud.metrics import Snapshot
    from jarvis.store import Store
    st = Store(tmp_path / "db.sqlite", encrypt=False)
    now = datetime(2026, 11, 24, 21, 15).timestamp()
    st.add_reminder(datetime(2026, 11, 25, 10, 0).timestamp(), "dentist")
    st.add_reminder(datetime(2026, 11, 27, 10, 0).timestamp(), "not tomorrow")
    t = datetime(2026, 11, 24, 9, 0).timestamp()
    for app, mins in [("Code.exe", 190), ("chrome.exe", 45), ("LockApp.exe", 120), ("Spotify.exe", 25)]:
        st.log_event("focus", {"app": app}, ts=t)
        t += mins * 60
    st.log_event("command_run", {"program": "x"}, ts=t)
    dl = tmp_path / "Downloads"
    dl.mkdir()
    for i in range(30):
        (dl / f"{i}.zip").write_bytes(b"x")
        os.utime(dl / f"{i}.zip", (now - 90 * 86400,) * 2)
    f = B.gather("evening", st, Snapshot(cpu=5, gpu_temp=88, disk_free_gb=200), {}, None, dl, now)
    assert f.apps[0] == ("VS Code", 190) and all(a != "Lockapp" for a, _ in f.apps)   # lock screen isn't work
    assert [r[1] for r in f.reminders] == ["dentist"]
    text = B.render(f, "Jarvis", "sir", "butler")
    for bit in ["VS Code (3 h 10 min)", "dentist at 10:00 am", "running hot at 88", "30 are over a month old", "1 command"]:
        assert bit in text, bit
    assert not B.short(text).startswith("Good")


def test_briefing_scheduler(tmp_path):
    from datetime import datetime
    from jarvis.briefing import Scheduler
    s = Scheduler(tmp_path / "state.json", "07:00", "21:00")
    assert s.due(datetime(2026, 11, 24, 6, 30), True) is None          # too early
    assert s.due(datetime(2026, 11, 24, 8, 0), False) is None          # not at the PC
    assert s.due(datetime(2026, 11, 24, 8, 0), True) == "morning"
    s.mark("morning", datetime(2026, 11, 24, 8, 0))
    assert s.due(datetime(2026, 11, 24, 9, 0), True) is None           # once a day
    assert s.due(datetime(2026, 11, 24, 21, 5), True) == "evening"
    assert Scheduler(tmp_path / "state.json").given_today("morning", datetime(2026, 11, 24, 10, 0))   # survives restart


def test_briefing_command_end_to_end(env, tmp_path):
    from jarvis.daemon import Jarvis
    cfg, store, dl, arc = env
    j = Jarvis(cfg, lambda *a: True, data_root=tmp_path / "d")
    j.config.show_details = False
    reply = j.handle("how was my day")
    assert reply.startswith("Good") and "reminders tomorrow" in reply or "reminder" in reply
    assert j.briefings.given_today("evening", __import__("datetime").datetime.now())


# ---------------------------------------------------------------- file categories ("video files") and new folders
def test_category_matching():
    from jarvis import filetypes as FT
    assert FT.matches(".MKV", "video") and FT.matches(".mp4", "video") and not FT.matches(".zip", "video")
    assert FT.matches(".pdf", "pdf") and not FT.matches(".pdf", "mp4")
    assert FT.category_in("move all videos files in downloads to a new folder videos") == "video"
    assert FT.category_in("move everything from downloads to videos") is None        # Videos is the destination
    assert FT.category_in("move pdfs from downloads to the videos folder") is None


def test_move_video_files_to_new_folder_end_to_end(env, tmp_path):
    from jarvis.daemon import Jarvis
    cfg, store, dl, arc = env
    for name in ["trip.mp4", "clip.MKV", "talk.mov", "setup.exe", "app.zip", "notes.pdf", "song.mp3"]:
        (dl / name).write_bytes(b"x")
    j = Jarvis(cfg, lambda *a: True, data_root=tmp_path / "d")
    j.config.show_details = False
    reply = j.handle("move all videos files in downloads to a new folder videos")
    moved = sorted(p.name for p in (dl / "videos").iterdir())
    assert moved == ["clip.MKV", "talk.mov", "trip.mp4"], moved                       # never the zip or exe
    left = {p.name for p in dl.iterdir() if p.is_file()}
    assert {"app.zip", "notes.pdf", "setup.exe", "song.mp3"} <= left                   # still in Downloads
    assert "three videos" in reply.lower() and 'new folder "videos"' in reply, reply


def test_brain_adds_missing_file_type():
    from jarvis.nlu.brain import Brain

    class Fake:
        def __init__(self, out): self.out = out
        def parse(self, text): return parse(self.out), 0.95

    b = Brain(); b.model = Fake("move_files(dest=Videos, src=Downloads)")
    assert str(b.resolve("move all video files in downloads to a new folder videos")[0]) == \
        'move_files(ext=video, new_folder="videos", src=Downloads)'
    b = Brain(); b.model = Fake("move_files(dest=Videos, src=Downloads)")
    assert str(b.resolve("move everything from downloads to videos")[0]) == "move_files(dest=Videos, src=Downloads)"


# ---------------------------------------------------------------- trained model first, rules only as a backup
def test_model_decides_what_it_was_trained_on():
    from jarvis.nlu.brain import Brain

    class Fake:
        def __init__(self, out, knows=None): self.out, self.knows = out, knows
        def parse(self, text): return parse(self.out), 0.95

    old = Brain(); old.model = Fake("unknown()")                      # trained before briefings: rules cover it
    assert old.resolve("good morning")[1].startswith("rules")
    new = Brain(); new.model = Fake("briefing(kind=morning)", {"actions": ["briefing"], "features": []})
    prog, src = new.resolve("good morning")
    assert src.startswith("model") and str(prog) == "briefing(kind=morning)"
    # categories: an old model gets the repair, a trained one is trusted as-is
    old = Brain(); old.model = Fake("move_files(dest=Videos, src=Downloads)")
    assert "ext=video" in str(old.resolve("move all video files in downloads to videos")[0])
    new = Brain(); new.model = Fake("move_files(dest=Videos, ext=video, src=Downloads)",
                                    {"actions": ["move_files"], "features": ["file_categories"]})
    prog, src = new.resolve("move all video files in downloads to videos")
    assert src.startswith("model") and str(prog) == "move_files(dest=Videos, ext=video, src=Downloads)"


def test_generator_records_what_it_teaches(tmp_path):
    import json, subprocess, sys
    subprocess.run([sys.executable, "-m", "datagen.generate_intents", "--n", "3000", "--n-eval", "100",
                    "--out", str(tmp_path)], check=True, capture_output=True)
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert {"briefing", "move_files", "mood"} <= set(meta["actions"]) and "file_categories" in meta["features"]


# ---------------------------------------------------------------- self-improving loop (phase 14)
def _sc(fam, uns, dang=0.001, corr=None):
    d = {"familiar": {"exact": fam, "dangerous": dang, "n": 1000, "missed": 0},
         "unseen": {"exact": uns, "dangerous": dang, "n": 1000, "missed": 0}}
    if corr is not None:
        d["your corrections"] = {"exact": corr, "dangerous": 0, "n": 10, "missed": 0}
    return d


def test_loop_decides_only_better_and_no_riskier():
    from training.loop import decide
    assert decide({"live": _sc(.998, .997), "candidate": _sc(.9985, .998)})[0]
    assert decide({"live": _sc(.998, .997, corr=.5), "candidate": _sc(.998, .997, corr=1.0)})[0]   # learned your fixes
    ok, why = decide({"live": _sc(.998, .997), "candidate": _sc(.999, .990)})
    assert not ok and "worse on unseen" in why
    ok, why = decide({"live": _sc(.998, .997), "candidate": _sc(.999, .998, dang=0.004)})
    assert not ok and "risky" in why
    assert not decide({"live": _sc(.998, .997), "candidate": _sc(.998, .997)})[0]                # no gain: keep


def test_loop_versions_and_rollback(tmp_path, monkeypatch):
    import training.loop as L
    ck = tmp_path / "checkpoints"
    ck.mkdir()
    monkeypatch.setattr(L, "CK", ck)
    monkeypatch.setattr(L, "VERSIONS", ck / "versions")
    monkeypatch.setattr(L, "STATE", ck / "improve_state.json")
    for f in L.LIVE:
        (ck / f).write_text("original")
    v0 = L.archive_live("baseline")
    L.make_live_state(v0)
    v1 = ck / "versions" / "v001-new"
    v1.mkdir()
    for f in L.LIVE:
        (v1 / f).write_text("improved")
    L.make_live(v1)
    assert (ck / "intent.pt").read_text() == "improved" and L.current_version() == "v001-new"
    assert L.rollback() == "Rolled back to v000-baseline." and (ck / "intent.pt").read_text() == "original"
    assert "oldest" in L.rollback()
    assert L.rollback("v001").startswith("Rolled back to v001")


def test_loop_regression_set(tmp_path, monkeypatch):
    import json
    import training.loop as L
    monkeypatch.setattr(L, "REGRESSION", tmp_path / "regression.jsonl")
    errs = [{"text": "open zomato", "gold": 'open_url(url="zomato.com")', "pred": "unknown()"}]
    L._save_mistakes(errs)
    L._save_mistakes(errs)                                                # no duplicates
    rows = [json.loads(x) for x in (tmp_path / "regression.jsonl").read_text().splitlines()]
    assert rows == [{"text": "open zomato", "program": 'open_url(url="zomato.com")'}]


def test_loop_knows_when_there_is_something_new(tmp_path, monkeypatch):
    import json
    import training.loop as L
    ck = tmp_path / "ck"
    ck.mkdir()
    monkeypatch.setattr(L, "CK", ck)
    monkeypatch.setattr(L, "STATE", ck / "state.json")
    monkeypatch.setattr(L, "CORRECTIONS", tmp_path / "corr.jsonl")
    from datagen.generate_intents import TEACHES
    (ck / "intent_onnx.json").write_text(json.dumps({"knows": {"actions": [], "features": TEACHES}}))
    assert L.what_is_new() == []
    (tmp_path / "corr.jsonl").write_text('{"text": "a", "program": "undo()"}\n')
    assert L.what_is_new() == ["1 new correction(s)"]
    (ck / "intent_onnx.json").write_text(json.dumps({"knows": None}))
    assert any("new skills" in r for r in L.what_is_new())


def test_improve_and_rollback_commands():
    from jarvis.nlu.rule_parser import rule_parse
    assert str(rule_parse("improve yourself")) == "training(action=improve)"
    assert str(rule_parse("roll back the model")) == "training(action=rollback)"


def test_background_services_start(env, tmp_path):
    """The startup path (monitor, briefings, self-improvement) must at least start without errors."""
    from jarvis.daemon import Jarvis
    cfg, store, dl, arc = env
    j = Jarvis(cfg, lambda *a: True, data_root=tmp_path / "d")
    sampler, logs = j.start_monitor(lambda a, text: None)
    assert "Self-improvement" in logs and j.sampler is sampler
    assert j.start_briefings(lambda kind, text: None).is_alive()
    assert j.start_improver(lambda text: None).is_alive()


# ---------------------------------------------------------------- memory search (phase 7)
def _mem_home(tmp_path):
    import os, time, zipfile
    home = tmp_path / "memhome"
    for d in ("Downloads", "Documents", "Pictures/Screenshots"):
        (home / d).mkdir(parents=True)
    (home / "Downloads" / "Invoice_March_2026.pdf").write_bytes(b"%PDF-1.4 not really")
    (home / "Documents" / "notes.txt").write_text("Meeting with Rahul about the transformer training budget")
    old = home / "Documents" / "old_resume.txt"
    old.write_text("resume: python, ml engineer")
    os.utime(old, (time.time() - 90 * 86400,) * 2)
    (home / "Pictures" / "Screenshots" / "Screenshot 2026-11-20 101500.png").write_bytes(b"png")
    with zipfile.ZipFile(home / "Documents" / "report.docx", "w") as z:
        z.writestr("word/document.xml", "<w:p><w:t>Quarterly payment receipt for Swiggy</w:t></w:p>")
    return home


def test_memory_index_search(tmp_path):
    import time
    from jarvis.memory.index import MemoryIndex
    home = _mem_home(tmp_path)
    idx = MemoryIndex(tmp_path / "m.sqlite")
    roots = [home / "Downloads", home / "Documents", home / "Pictures"]
    assert idx.crawl(roots)["added"] == 5
    idx.add_history([{"url": "https://arxiv.org/abs/1706.03762", "title": "Attention Is All You Need",
                      "ts": time.time() - 86400, "browser": "Chrome"}])
    t = lambda hits: [h.title for h in hits]                                   # noqa: E731
    assert t(idx.search("invoices", ext="pdf")) == ["Invoice_March_2026.pdf"]   # stemming + type filter
    assert t(idx.search("transformer budget")) == ["notes.txt"]                  # inside the text
    assert t(idx.search("payment receipt", ext="document")) == ["report.docx"]   # inside a Word file
    assert t(idx.search("", where="screenshots")) == ["Screenshot 2026-11-20 101500.png"]
    assert t(idx.search("attention", where="web")) == ["Attention Is All You Need"]
    assert t(idx.search("resume", newer_than_s=7 * 86400)) == []                 # 90 days old
    assert t(idx.search("resume")) == ["old_resume.txt"]
    assert idx.crawl(roots)["added"] == 0                                       # incremental
    (home / "Documents" / "notes.txt").unlink()
    assert idx.crawl(roots)["removed"] == 1


def test_memory_repair_for_older_models():
    from jarvis.nlu.brain import Brain

    class Fake:
        def __init__(self, out): self.out, self.knows = out, None
        def parse(self, text): return parse(self.out), 0.95

    b = Brain(); b.model = Fake('search_memory(query="invoice pdf from last week")')
    assert str(b.resolve("find the invoice pdf from last week")[0]) == \
        'search_memory(ext=pdf, newer_than=7d, query="invoice")'
    b = Brain(); b.model = Fake('search_memory(query="screenshot with the login error")')
    assert str(b.resolve("where's that screenshot with the login error")[0]) == \
        'search_memory(query="login error", where=screenshots)'


def test_find_end_to_end(env, tmp_path):
    from jarvis.daemon import Jarvis
    from jarvis.memory import index as MI
    cfg, store, dl, arc = env
    home = _mem_home(tmp_path)
    j = Jarvis(cfg, lambda *a: True, data_root=tmp_path / "d")
    j.config.show_details = False
    MI.get(cfg.data_dir).crawl([home / "Downloads", home / "Documents"])
    reply = j.handle("find the invoice pdf from last week")
    assert "Invoice_March_2026.pdf" in reply and "Downloads" in reply, reply
    rows = j.take_results()
    assert rows and rows[0]["path"].endswith("Invoice_March_2026.pdf") and j.take_results() == []
    assert "couldn't find" in j.handle("find the zebra migration spreadsheet").lower()


def test_old_reply_model_never_speaks_for_newer_reply_types():
    from jarvis.actions.base import ActionResult
    from jarvis.nlg.generator import ReplyModel
    from jarvis.persona.persona import Persona

    class OldModel:                        # trained before memory search: only knows the outdated sentence
        keys = None
        knows_key = ReplyModel.knows_key
        def generate(self, key, fields, tone, address):
            return f"I can't search my memory for {fields.get('query')} just yet."

    p = Persona("Jarvis", "sir", "butler")
    p.model = OldModel()
    r = ActionResult("", kind="memory", data={"query": "login error", "count": 1, "where": "screenshots", "ext": None,
                                              "indexed": {"files": 5}, "results": [{"title": "Screenshot 1.png",
                                              "path": "C:/x/Screenshots/Screenshot 1.png", "source": "file",
                                              "mtime": 0, "screenshot": True}]})
    reply = p.result(r)
    assert "can't search" not in reply and "Screenshot 1.png" in reply, reply
    assert p.last_source == "template"


# ---------------------------------------------------------------- OCR (phase 7 step 2)
def test_ocr_training_data_renders():
    import random
    from training.ocr.charset import decode_greedy, encode
    from training.ocr.fonts import find_fonts
    from training.ocr.render import FontBank, render_line, render_page
    from training.ocr.render import _need_steps, prepare
    bank = FontBank(find_fonts())
    r = random.Random(0)
    for _ in range(50):
        a, s = render_line(r, bank)
        ids = encode(s)
        a = prepare(a, ids)
        assert a.shape[0] == 32 and a.shape[1] % 4 == 0 and a.shape[1] // 4 >= _need_steps(ids)
    assert decode_greedy([encode("h")[0], encode("h")[0], 0, encode("h")[0], encode("i")[0]]) == "hhi"
    img, mask, boxes = render_page(r, bank)
    assert img.shape == (384, 640) and mask.shape == (96, 160) and boxes and mask.max() == 1.0


def test_ocr_line_boxes_recover_lines():
    import random
    from jarvis.memory.ocr import line_boxes
    from training.ocr.fonts import find_fonts
    from training.ocr.render import FontBank, render_page

    def iou(a, b):
        ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
        iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        i = ix * iy
        return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i)

    bank = FontBank(find_fonts())
    hit = tot = 0
    for seed in range(15):
        img, mask, gt = render_page(random.Random(seed), bank)
        boxes = line_boxes(mask, 384, 640)                 # a perfect finder's output
        for g in gt:
            tot += 1
            hit += max([iou(g[:4], b) for b in boxes] or [0]) > 0.5
    assert hit / tot > 0.85, hit / tot


def test_ocr_text_is_indexed(tmp_path):
    from jarvis.memory.index import MemoryIndex
    shots = tmp_path / "Pictures" / "Screenshots"
    shots.mkdir(parents=True)
    (shots / "Screenshot 2026-11-20 101500.png").write_bytes(b"png")
    idx = MemoryIndex(tmp_path / "m.sqlite")
    idx.crawl([tmp_path / "Pictures"])
    todo = idx.needs_ocr()
    assert len(todo) == 1
    idx.set_text(todo[0], "Login failed: invalid password\nForgot password?")
    assert idx.needs_ocr() == []
    hits = idx.search("login error", where="screenshots")          # "error" isn't there, "login" is: any-word
    assert hits and hits[0].title.startswith("Screenshot") and "Login" in hits[0].snippet


def test_ocr_drops_invented_text():
    from jarvis.memory.ocr import looks_like_text
    assert looks_like_text("Login failed: invalid password", 0.97)
    assert looks_like_text("Rainfall amount (mm)", 0.93) and looks_like_text("40%", 0.95)
    for junk, conf in [("_<", 0.99), ("|{||", 0.95), ("/9QQ%#\\\\'", 0.9), ("Hello world", 0.5), ("ab", 0.8)]:
        assert not looks_like_text(junk, conf), junk


# ---------------------------------------------------------------- meaning search (phase 7 step 3)
class _ConceptEmbedder:
    """Stand-in for the meaning model: words of the same concept group share a direction."""
    def __init__(self):
        import numpy as np
        from training.meaning.concepts import CONCEPTS
        self.np = np
        self.word = {}
        for g, group in enumerate(CONCEPTS):
            for phrase in group:
                for w in phrase.split():
                    self.word.setdefault(w, g)
        self.n = len(CONCEPTS) + 1

    def embed(self, texts):
        np = self.np
        out = np.zeros((len(texts), self.n), np.float32)
        for i, t in enumerate(texts):
            for w in t.lower().replace("_", " ").split():
                out[i, self.word.get(w.strip(".,"), self.n - 1)] += 1.0 if w in self.word else 0.05
            out[i] /= max(1e-6, np.linalg.norm(out[i]))
        return out


def test_meaning_search_finds_other_words(tmp_path):
    from jarvis.memory.index import MemoryIndex
    from jarvis.memory.meaning import chunks
    d = tmp_path / "Downloads"
    d.mkdir()
    (d / "swiggy_receipt_march.txt").write_text("payment receipt order 4821 total 459 paid by upi")
    (d / "trip_photos.txt").write_text("goa vacation pictures beach")
    (d / "notes.txt").write_text("meeting minutes with rahul about the transformer training run")
    idx = MemoryIndex(tmp_path / "m.sqlite")
    idx.crawl([d])
    emb = _ConceptEmbedder()
    todo = idx.needs_embed()
    assert len(todo) == 3
    for doc_id, title, body in todo:
        idx.add_vectors(doc_id, emb.embed(chunks(title, body)))
    idx.commit()
    assert idx.needs_embed() == [] and idx.count()["with_meaning"] == 3
    assert [h.title for h in idx.search("bill")] == []                                   # words alone: nothing
    assert idx.search("bill", embedder=emb)[0].title == "swiggy_receipt_march.txt"       # meaning: the receipt
    assert idx.search("holiday", embedder=emb)[0].title == "trip_photos.txt"
    assert idx.search("notes from the call", embedder=emb)[0].title == "notes.txt"
    import os, time
    (d / "notes.txt").write_text("changed text")
    os.utime(d / "notes.txt", (time.time() + 10,) * 2)
    idx.crawl([d])
    assert len(idx.needs_embed()) == 1                                                   # re-embed what changed
