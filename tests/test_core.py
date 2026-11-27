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


def test_saved_folders_that_dont_exist_use_the_real_ones(tmp_path):
    from jarvis.config import merge_folders
    real = tmp_path / "OneDrive" / "Desktop"
    real.mkdir(parents=True)
    out = merge_folders({"Desktop": str(real), "Documents": str(tmp_path)},
                        {"Desktop": str(tmp_path / "missing" / "Desktop"), "Work": str(tmp_path)})
    assert out["Desktop"] == str(real)                 # the saved path is gone: use where Windows says it is
    assert out["Work"] == str(tmp_path)                # your own aliases stay
    assert merge_folders({"Desktop": str(real)}, {"Desktop": str(tmp_path)})["Desktop"] == str(tmp_path)  # yours win


# ---------------------------------------------------------------- reliability (phase 29)
def test_single_instance():
    from jarvis.reliability import SingleInstance
    shown = []
    import random
    port = random.randint(40000, 60000)
    a = SingleInstance(port=port)
    assert a.acquire()
    a.on_show = lambda: shown.append(1)
    b = SingleInstance(port=port)
    assert not b.acquire() and b.ping_existing()
    import time
    time.sleep(0.2)
    assert shown == [1]
    a.release()


def test_watchdog_restarts_dead_services():
    import threading
    from jarvis.reliability import Watchdog
    wd = Watchdog()
    dead = threading.Thread(target=lambda: None)
    dead.start()
    dead.join()
    alive = threading.Event()

    def restart():
        t = threading.Thread(target=alive.wait, daemon=True)
        t.start()
        return t
    wd.register("indexer", dead, restart)
    assert wd.check() == ["indexer"] and wd.status() == {"indexer": True}
    assert wd.check() == []
    alive.set()


def test_crash_reports_and_counting(tmp_path):
    import sys
    from jarvis.reliability import CrashLog, recent_crashes, write_crash_report
    try:
        raise ValueError("boom")
    except ValueError:
        p = write_crash_report(tmp_path, *sys.exc_info())
    assert "ValueError: boom" in p.read_text() and recent_crashes(tmp_path) == [p]
    c = CrashLog(tmp_path / "crashes.json")
    for _ in range(3):
        c.add()
    assert c.recent(600) == 3
    c.clear()
    assert c.recent(600) == 0


def test_supervisor_restarts_and_goes_safe(tmp_path, monkeypatch):
    import jarvis.supervisor as S
    runs = []
    codes = iter([1, 1, 1, 3, 0])                     # crash x3 -> safe mode, restart request, quit
    monkeypatch.setattr(S.subprocess, "call", lambda cmd, env=None: (runs.append(cmd), next(codes))[1])
    monkeypatch.setattr(S.time, "sleep", lambda s: None)
    assert S.run([], tmp_path) == 0
    assert ["--safe-mode" in r for r in runs] == [False, False, False, True, True]


def test_doctor_repairs(tmp_path):
    import sqlite3
    from jarvis import doctor
    from jarvis.config import Config
    cfg = Config(data_dir=tmp_path)
    (tmp_path / "config.json").write_text("{ broken json")
    (tmp_path / "memory.sqlite").write_bytes(b"this is not a database" * 100)
    con = sqlite3.connect(str(cfg.db_path))
    con.execute("create table t(x)")
    con.commit()
    con.close()
    checks = {c.name: c for c in doctor.run(cfg, {"indexer": True, "briefings": False}, ck=tmp_path / "ck")}
    assert checks["Settings"].status == "fixed" and not (tmp_path / "config.json").exists()
    assert checks["Memory index"].status == "fixed" and not (tmp_path / "memory.sqlite").exists()
    assert list(tmp_path.glob("memory.sqlite.broken-*"))                     # kept, not deleted
    assert checks["Activity log"].status == "ok"
    assert checks["Command model"].status == "warn"                          # nothing trained in this folder
    assert checks["Background services"].status == "warn" and "briefings" in checks["Background services"].detail
    text = doctor.summary(list(checks.values()))
    assert text.startswith("Health check done") and "🔧 Settings" in text


def test_backup_roundtrip(tmp_path, monkeypatch):
    import jarvis.backup as B
    from jarvis.config import Config
    proj = tmp_path / "proj"
    (proj / "data").mkdir(parents=True)
    (proj / "data" / "user_corrections.jsonl").write_text("fix1\n")
    monkeypatch.setattr(B, "PROJECT_ROOT", proj)
    data = tmp_path / "data"
    data.mkdir()
    cfg = Config(data_dir=data, folders={"Documents": str(tmp_path / "Docs")})
    (data / "config.json").write_text('{"persona_tone": "friendly"}')
    z = B.create(cfg)
    assert z.exists() and B.backups(cfg) == [z]
    (data / "config.json").write_text("{}")
    (proj / "data" / "user_corrections.jsonl").write_text("")
    msg = B.restore(cfg, z)
    assert (data / "config.json").read_text() == '{"persona_tone": "friendly"}'
    assert (proj / "data" / "user_corrections.jsonl").read_text() == "fix1\n" and "before-restore" in msg


def test_health_and_backup_commands():
    from jarvis.nlu.rule_parser import rule_parse
    assert str(rule_parse("run a health check")) == "maintenance(action=health_check)"
    assert str(rule_parse("back up yourself")) == "maintenance(action=backup)"


# ---------------------------------------------------------------- crash-safe actions (phase 40)
def _files(d, names):
    d.mkdir(parents=True, exist_ok=True)
    for n in names:
        (d / n).write_bytes(n.encode() * 50)
    return [d / n for n in names]


def test_locked_file_doesnt_stop_the_batch(tmp_path, monkeypatch):
    import shutil
    from jarvis.actions import safeops
    from jarvis.actions.files import _unique_target
    from jarvis.store import Store
    store = Store(tmp_path / "j.db", encrypt=False)
    src = _files(tmp_path / "dl", ["a.pdf", "report.xlsx", "c.pdf"])
    real_move = shutil.move

    def move(a, b):
        if a.endswith("report.xlsx"):
            e = PermissionError("The process cannot access the file because it is being used by another process")
            e.winerror = 32
            raise e
        return real_move(a, b)
    monkeypatch.setattr(safeops.shutil, "move", move)
    monkeypatch.setattr(safeops.time, "sleep", lambda s: None)
    done, failed = safeops.transfer(store, "b1", src, tmp_path / "arc", "move", _unique_target)
    assert [t.name for _, t in done] == ["a.pdf", "c.pdf"]
    assert failed == [("report.xlsx", "it's open in another app")]
    assert (tmp_path / "dl" / "report.xlsx").exists()                     # original untouched
    assert store.pending() == [] and len(store.last_batch()[1]) == 2      # only real moves are undoable


def test_not_enough_space_moves_nothing(tmp_path, monkeypatch):
    from collections import namedtuple
    from jarvis.actions import safeops
    from jarvis.actions.files import _unique_target
    from jarvis.store import Store
    store = Store(tmp_path / "j.db", encrypt=False)
    src = _files(tmp_path / "dl", ["big.iso"])
    monkeypatch.setattr(safeops.shutil, "disk_usage", lambda p: namedtuple("U", "total used free")(1, 1, 10))
    done, failed = safeops.transfer(store, "b1", src, tmp_path / "arc", "copy", _unique_target)
    assert done == [] and "disk space" in failed[0][1] and src[0].exists()


def test_copy_that_arrives_damaged_is_cleaned_up(tmp_path, monkeypatch):
    from jarvis.actions import safeops
    from jarvis.actions.files import _unique_target
    from jarvis.store import Store
    store = Store(tmp_path / "j.db", encrypt=False)
    src = _files(tmp_path / "dl", ["a.pdf"])
    monkeypatch.setattr(safeops.shutil, "copy2", lambda a, b: open(b, "wb").write(b"half"))
    done, failed = safeops.transfer(store, "b1", src, tmp_path / "arc", "copy", _unique_target)
    assert done == [] and "intact" in failed[0][1]
    assert not (tmp_path / "arc" / "a.pdf").exists() and src[0].exists()


def test_recovery_after_a_crash(tmp_path):
    from jarvis.actions import safeops
    from jarvis.store import Store
    store = Store(tmp_path / "j.db", encrypt=False)
    dl, arc = tmp_path / "dl", tmp_path / "arc"
    _files(dl, ["notstarted.pdf", "halfcopy.mp4", "copiedboth.zip"])
    _files(arc, ["finished.pdf", "copiedboth.zip"])
    (arc / "halfcopy.mp4").write_bytes(b"partial")
    size = lambda n: len(n.encode() * 50)                                  # noqa: E731
    for name in ["notstarted.pdf", "finished.pdf", "halfcopy.mp4", "copiedboth.zip"]:
        store.begin("b1", "move", {"op": "move", "src": str(arc / name), "dest": str(dl / name)},
                    {"kind": "move", "src": str(dl / name), "target": str(arc / name), "size": size(name)})
    r = safeops.recover(store)
    assert sorted(r["finished"]) == ["copiedboth.zip", "finished.pdf"]
    assert sorted(r["rolled_back"]) == ["halfcopy.mp4", "notstarted.pdf"]
    assert (dl / "halfcopy.mp4").exists() and not (arc / "halfcopy.mp4").exists()   # original kept
    assert not (dl / "copiedboth.zip").exists() and (arc / "copiedboth.zip").exists()
    assert store.pending() == [] and len(store.last_batch()[1]) == 2                  # finished ones undoable


def test_undo_carries_on_past_a_stuck_file(tmp_path, env):
    from jarvis.daemon import Jarvis
    cfg, store, dl, arc = env
    for n in ("x1.pdf", "x2.pdf"):
        (dl / n).write_bytes(b"x")
    j = Jarvis(cfg, lambda *a: True, data_root=tmp_path / "d")
    j.config.show_details = False
    j.config.confirm_actions = []
    j.handle("move pdfs from downloads to archive")
    moved = sorted(p.name for p in arc.iterdir() if p.name.startswith("x"))
    assert moved == ["x1.pdf", "x2.pdf"]
    (arc / "x1.pdf").unlink()                                                     # deleted by you since
    reply = j.handle("undo")
    assert (dl / "x2.pdf").exists() and "x1.pdf" in reply and "moved or deleted" in reply, reply


def test_failure_reply_is_honest():
    from jarvis.actions.base import ActionResult
    from jarvis.persona.persona import Persona
    p = Persona("Jarvis", "sir", "butler")
    d = {"op": "move", "count": 10, "dry": False, "src": "Downloads", "dest": "Archive", "new_folder": False,
         "ext": "pdf", "older_than": None, "newer_than": None, "name_contains": None, "requested": 12,
         "failed": [("report.xlsx", "it's open in another app"), ("big.mp4", "there isn't enough disk space")]}
    text = p.result(ActionResult("", kind="files", data=d))
    assert "2 files didn't go through" in text and "report.xlsx (it's open in another app)" in text
    none = p.result(ActionResult("", kind="files", data={**d, "count": 0}))
    assert none.startswith("I couldn't move any of them")


# ---------------------------------------------------------------- directions
def _maps_ctx(tmp_path, **kw):
    from jarvis.actions.base import Ctx
    from jarvis.config import Config
    cfg = Config(data_dir=tmp_path)
    cfg.places = {"home": "HSR Layout, Bengaluru", "work": ""}
    for k, v in kw.items():
        setattr(cfg, k, v)
    return Ctx(cfg, None, "b")


def test_directions_google_live_traffic(tmp_path):
    from jarvis.actions.maps import directions
    from jarvis.persona.persona import Persona
    opened, calls = [], []

    def http(url, data=None, headers=None, timeout=8):
        calls.append((url, data, headers))
        return {"routes": [{"duration": "3600s", "staticDuration": "2700s", "distanceMeters": 38000,
                            "description": "NH 44"},
                           {"duration": "3120s", "staticDuration": "2600s", "distanceMeters": 41000,
                            "description": "Hosur Rd"}]}
    r = directions({"destination": "the airport"}, _maps_ctx(tmp_path, maps_api_key="plain:KEY"), False,
                   http=http, open_url=opened.append)
    assert "maps/dir" in opened[0] and "destination=the+airport" in opened[0]
    url, body, headers = calls[0]
    assert body["origin"]["address"] == "HSR Layout, Bengaluru" and body["routingPreference"] == "TRAFFIC_AWARE"
    assert headers["X-Goog-Api-Key"] == "KEY"
    text = Persona("Jarvis", "sir", "butler").result(r)
    assert "52 minutes" in text and "Hosur Rd" in text and "current traffic" in text and "41 km" in text, text


def test_directions_osm_and_missing_places(tmp_path):
    from jarvis.actions.maps import directions, _geo_cache
    from jarvis.persona.persona import Persona
    import jarvis.actions.maps as M
    M._last_nominatim[0] = 0.0
    _geo_cache.clear()

    def http(url, data=None, headers=None, timeout=8):
        if "nominatim" in url:
            return [{"lat": "12.91", "lon": "77.64", "display_name": "x"}]
        return {"routes": [{"duration": 1800, "distance": 12000}]}
    p = Persona("Jarvis", "sir", "butler")
    r = directions({"destination": "indiranagar", "mode": "two_wheeler"}, _maps_ctx(tmp_path), False,
                   http=http, open_url=lambda u: None)
    assert "30 minutes" in p.result(r) and "OpenStreetMap" in p.result(r)
    r = directions({"destination": "work"}, _maps_ctx(tmp_path), False, http=http, open_url=lambda u: None)
    assert "don't know where your work is" in p.result(r)
    ctx = _maps_ctx(tmp_path)
    ctx.config.places = {}
    r = directions({"destination": "goa"}, ctx, False, http=http, open_url=lambda u: None)
    assert "Add your home address" in p.result(r)


def test_directions_rules():
    from jarvis.nlu.rule_parser import rule_parse
    assert str(rule_parse("fastest route from koramangala to the airport")) == \
        'directions(destination="airport", origin="koramangala")'
    assert str(rule_parse("airport tak kitna time lagega")) == 'directions(destination="airport")'
    assert str(rule_parse("how long to get home by metro")) == 'directions(destination="home", mode=transit)'
    assert rule_parse("how long is the training left") is None or "directions" not in str(rule_parse("how long is the training left"))


# ---------------------------------------------------------------- phase 39: real-usage learning
def test_doubt_score():
    from jarvis.usage import doubt_score
    assert doubt_score("dsl", "undo()", None, []) == 0
    assert doubt_score("unparsed", None, None, []) == 1
    sure = doubt_score("model 97%", 'open_app(name="chrome")', 0.97, [[0.97, 'open_app(name="chrome")']])
    assert sure < 0.1
    torn = doubt_score("model 81%", 'play_media(query="sup", site=youtube)', 0.81,
                       [[0.81, 'play_media(query="sup", site=youtube)'], [0.78, 'play_media(query="super machi", site=youtube)']])
    assert torn >= 0.8
    assert doubt_score("rules (model unsure)", 'directions(destination="x")', None, []) >= 0.8
    assert doubt_score("rules (model not trained on this yet)", 'directions(destination="x")', None, []) < 0.35


def _usage(tmp_path):
    from jarvis.usage import UsageLog
    return UsageLog(tmp_path / "u.db", encrypt=False, corrections=tmp_path / "corr.jsonl")


def test_usage_review_queue_and_answers(tmp_path):
    import json
    from jarvis.usage import ReviewSession
    u = _usage(tmp_path)
    alts = [[0.6, 'play_media(query="sup", site=youtube)'], [0.55, 'play_media(query="super machi", site=youtube)']]
    a = u.record("play super machi", 'play_media(query="sup", site=youtube)', "model 60%", {"conf": 0.6, "alts": alts})
    u.record("Play super machi!", 'play_media(query="sup", site=youtube)', "model 60%", {"conf": 0.6, "alts": alts})
    u.record("open chrome", 'open_app(name="chrome")', "model 98%", {"conf": 0.98, "alts": [[0.98, 'open_app(name="chrome")']]})
    b = u.record("blorp the flarn", None, "unparsed")
    items = u.to_review()
    assert [i.text for i in items] == ["blorp the flarn", "Play super machi!"]       # sure ones and repeats left out
    s = ReviewSession(u, items, parse_fn=lambda t: None)
    assert s.choices(items[1])[0][0] == 'play_media(query="super machi", site=youtube)'   # runner-up offered
    s.skip()
    s.fix(s.choices(items[1])[0][0])
    rows = [json.loads(l) for l in (tmp_path / "corr.jsonl").read_text().splitlines()]
    assert rows == [{"text": "Play super machi!", "program": 'play_media(query="super machi", site=youtube)', "from": "review"}]
    assert u.to_review() == []                     # answered phrasings don't come back
    assert "1 corrected" in s.summary()


def test_usage_undo_and_off(tmp_path):
    from jarvis.usage import UsageLog
    u = _usage(tmp_path)
    u.record("move my videos to music", 'move_files(src="Downloads", ext=video, dest="Music")', "model 95%",
             {"conf": 0.95, "alts": []})
    assert u.to_review() == []
    assert u.mark_undone()
    assert u.to_review()[0].text == "move my videos to music"
    off = UsageLog(tmp_path / "off.db", encrypt=False, enabled=False)
    assert off.record("hi there", None, "unparsed") is None and off.real_texts() == []
    assert u.forget() == 1


def test_usage_drift(tmp_path):
    import time
    from jarvis.usage import drift_line
    u = _usage(tmp_path)
    now = time.time()
    for i in range(40):                            # usual weeks: 5% unsure
        u.record(f"open app {i}", 'open_app(name="x")', "model 95%", {"conf": 0.95 if i % 20 else 0.5},
                 ts=now - 14 * 86400 - i * 3600)
    for i in range(20):                            # this week: 40% unsure
        u.record(f"thing {i}", 'open_app(name="x")', "model 95%", {"conf": 0.95 if i % 5 > 1 else 0.5},
                 ts=now - i * 3600)
    d = u.drift(now)
    assert d and "less sure" in drift_line(d)
    assert u.drift(now - 7 * 86400) is None


def test_report_card_and_gate():
    from training.loop import card, decide_real
    rows = [{"text": f"open x{i}", "ran": f'open_app(name="x{i}")', "label": None, "status": "new"} for i in range(40)]
    rows.append({"text": "play super machi", "ran": 'play_media(query="sup", site=youtube)',
                 "label": 'play_media(query="super machi", site=youtube)', "status": "fixed"})
    live = [f'open_app(name="x{i}")' for i in range(40)] + ['play_media(query="sup", site=youtube)']
    good = live[:40] + ['play_media(query="super machi", site=youtube)']
    r = card(rows, live, good)
    assert r["changed_n"] == 1 and r["labelled_right"] == 1 and r["labelled_right_live"] == 0
    assert decide_real(r)[0]
    drifted = ['unknown()'] * 5 + live[5:40] + good[40:]
    ok, why = decide_real(card(rows, live, drifted))
    assert not ok and "everyday" in why
    forgot = good[:40] + live[40:]
    r2 = card(rows, good, forgot)
    assert not decide_real(r2)[0]


def test_review_rules_and_explain():
    from jarvis.nlu.rule_parser import rule_parse
    from jarvis.nlu.explain import explain
    assert str(rule_parse("review my commands")) == "training(action=review)"
    assert explain('move_files(src="Downloads", ext=video, new_folder="Videos")') == \
        'move video files from Downloads to a new folder "Videos"'
    assert explain(None).startswith("nothing")


def test_jarvis_records_usage_and_review_cmd(tmp_path):
    from jarvis.config import Config
    from jarvis.daemon import Jarvis
    cfg = Config(data_dir=tmp_path, encrypt_logs=False)
    j = Jarvis(cfg, confirm=lambda q: True, data_root=tmp_path / "data")
    j.handle("blorp the flarn zzz")
    j.handle("/wrong")
    out = j.handle("/review")
    assert "blorp the flarn zzz" in out
    assert "means" in j.handle('/review fix 1 open_app(name="notepad")')
    assert "open_app" in (tmp_path / "data" / "user_corrections.jsonl").read_text()
    assert "Nothing to review" in j.handle("/review")
    assert "This week" in j.handle("/usage")


def test_new_enum_value_uses_rules_until_learned():
    from jarvis.nlu.brain import Brain

    class Old:
        knows = {"actions": ["training", "open_app"], "features": []}
        last_alts = []

        def parse(self, text):
            from jarvis.dsl.grammar import parse
            return parse("training(action=status)"), 0.9
    b = Brain()
    b.model, b.backend = Old(), "fake"
    prog, src = b.resolve("review my commands")
    assert str(prog) == "training(action=review)" and src.startswith("rules")
    Old.knows = {"actions": ["training"], "features": ["usage_review"]}
    assert str(b.resolve("review my commands")[0]) == "training(action=status)"   # trained model decides


# ---------------------------------------------------------------- phase 6: conversation & clarification
def test_dialog_data_valid_and_split(tmp_path):
    import json
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "datagen.generate_dialog", "--n", "3000", "--n-eval", "300",
                    "--out", str(tmp_path)], check=True, capture_output=True)
    from jarvis.dsl.grammar import canonical
    train = [json.loads(l) for l in (tmp_path / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    for ex in train[:3000]:
        canonical(ex["program"])
        assert " || " in ex["text"]
    kinds = {ex["text"].split(":")[0] for ex in train}
    assert kinds == {"PREV", "ASK", "CONFIRM"}
    unknown = sum(ex["program"] == "unknown()" for ex in train) / len(train)
    assert 0.3 < unknown < 0.55                       # plenty of "this isn't a follow-up"
    ood = {json.loads(l)["text"] for l in (tmp_path / "test_ood.jsonl").read_text(encoding="utf-8").splitlines()}
    assert not ood & {ex["text"] for ex in train}
    assert json.loads((tmp_path / "meta.json").read_text())["features"] == ["followups", "clarify"]


def test_followup_rules():
    from jarvis.dialog import Conversation
    c = Conversation()
    c.ran('move_files(src=Downloads, ext=pdf, dest=Documents)')
    assert c.read("no, to desktop")[0] == "move_files(dest=Desktop, ext=pdf, src=Downloads)"
    assert c.read("do the same for pictures")[0] == "move_files(dest=Documents, ext=pdf, src=Pictures)"
    assert c.read("again")[0] == 'move_files(dest=Documents, ext=pdf, src=Downloads)'
    assert c.read("play despacito")[0] is None
    opts = ['play_media(query="super machi", site=youtube)', 'play_media(query="sup", site=youtube)']
    c.asking("ask", opts, "play super machi", None)
    assert c.read("the second one")[0] == opts[1] and c.read("1")[0] == opts[0]
    assert c.read("super machi")[0] == opts[0] and c.read("neither")[0] == "unknown()"
    c.ran("not a program")
    assert c.active() is None
    c.ran('open_app(name=chrome)', now=0)
    assert c.active() is None                         # follow-ups expire


def test_should_ask():
    from jarvis.dialog import Conversation
    c = Conversation()
    torn = {"conf": 0.7, "alts": [[0.7, 'play_media(query="sup", site=youtube)'],
                                  [0.65, 'play_media(query="super machi", site=youtube)']]}
    assert c.should_ask('play_media(query="sup", site=youtube)', "model 70%", torn)[0] == "ask"
    mv = 'move_files(src=Downloads, ext=pdf, dest=Documents)'
    assert c.should_ask(mv, "model 65%", {"conf": 0.65, "alts": [[0.65, mv]]}) == ("confirm", [mv])
    assert c.should_ask("open_app(name=chrome)", "model 95%", {"conf": 0.95, "alts": [[0.95, "open_app(name=chrome)"]]}) is None
    assert c.should_ask(mv, "rules (model unsure)", {"conf": 0.4, "alts": []}) is None
    assert Conversation(clarify=False).should_ask('play_media(query="sup", site=youtube)', "model 70%", torn) is None


def _jarvis(tmp_path):
    from jarvis.config import Config
    from jarvis.daemon import Jarvis
    cfg = Config(data_dir=tmp_path, encrypt_logs=False)
    return Jarvis(cfg, confirm=lambda *a: True, data_root=tmp_path / "data")


def test_jarvis_followup_again(tmp_path):
    import time
    j = _jarvis(tmp_path)
    j.handle('remind(text="stretch", after=10m)')
    j.handle("again")
    assert len(j.store.upcoming_reminders(time.time(), limit=10)) == 2
    assert "No previous command" in j.handle("/fix open_app(name=chrome)")    # a follow-up isn't a /fix target


def test_jarvis_asks_when_torn_and_learns_answer(tmp_path):
    import json
    import time
    from jarvis.dsl.grammar import parse
    j = _jarvis(tmp_path)
    a, b = 'remind(text="call amma", after=10m)', 'remind(text="call anna", after=10m)'

    def torn(text):
        j.brain.last_detail = {"conf": 0.62, "alts": [[0.62, a], [0.6, b]]}
        return parse(a), "model 62%"
    j.brain.resolve = torn
    out = j.handle("remind me to call anna in 10 minutes")
    assert "which one" in out.lower() or "did you mean" in out.lower()
    assert j.store.upcoming_reminders(time.time(), limit=10) == []           # asked, didn't guess
    j.handle("2")
    assert [t for _, t in j.store.upcoming_reminders(time.time(), limit=10)] == ["call anna"]
    rows = [json.loads(l) for l in (tmp_path / "data" / "user_corrections.jsonl").read_text().splitlines()]
    assert rows[-1]["text"] == "remind me to call anna in 10 minutes" and "anna" in rows[-1]["program"]
    j.handle("remind me to call anna in 10 minutes")
    out = j.handle("neither").lower()
    assert "nothing done" in out or "won't do anything" in out
    assert len(j.store.upcoming_reminders(time.time(), limit=10)) == 1


def test_dialog_model_wrapper_reads_context():
    from jarvis.dialog import Context, Conversation, DialogModel
    from jarvis.dsl.grammar import parse
    seen = []

    class FakeParser:
        def parse(self, text):
            seen.append(text)
            return parse("move_files(src=Downloads, ext=pdf, dest=Desktop)"), 0.93
    m = DialogModel.__new__(DialogModel)
    m.parser = FakeParser()
    c = Conversation(m)
    c.ran("move_files(src=Downloads, ext=pdf, dest=Documents)")
    prog, src = c.read("nah put em on the desktop")
    assert seen == ["PREV: move_files(dest=Documents, ext=pdf, src=Downloads) || nah put em on the desktop"] or \
        seen[0].startswith("PREV: move_files(") and seen[0].endswith(" || nah put em on the desktop")
    assert "dest=Desktop" in prog and src == "follow-up 93%"


# ---------------------------------------------------------------- phase 38: context awareness
def test_ref_grammar():
    from jarvis.dsl.grammar import DSLError, parse, viable_prefix
    assert str(parse("search_web(query=this, site=google)")) == "search_web(query=this, site=google)"
    assert str(parse("move_files(src=this, dest=here)")) == "move_files(dest=here, src=this)"
    assert parse("open_result(n=2)").calls[0].args["n"] == 2
    with pytest.raises(DSLError):
        parse("search_web(query=Downloads)")            # only the reference, not any identifier
    assert viable_prefix("open_url(url=th") and not viable_prefix("open_url(url=Dow")


def _screen(sel=None, folder=None, clip=([], ""), hwnd=123):
    from jarvis.context import Explorer, ScreenContext
    ex = Explorer(folder, list(sel or [])) if (sel is not None or folder) else None
    return ScreenContext(hwnd=lambda: hwnd, explorer=lambda h: ex, desktop=lambda h: None, clip=lambda: clip)


def test_this_means_selection_then_results_then_copied(env):
    from jarvis.context import ContextError
    from jarvis.dsl.grammar import parse
    cfg, store, dl, arc = env
    a, b = str(dl / "a.pdf"), str(dl / "b.pdf")
    p = _screen(sel=[a]).fill(parse("move_files(src=this, dest=Archive)"))
    assert p.calls[0].args["_files"] == [a] and p.calls[0].args["_from"] == "selected"
    assert str(p) == "move_files(dest=Archive, src=this)"      # the file list is never printed or logged
    s = _screen()
    s.set_results([{"title": "b.pdf", "path": b, "source": "files"}])
    assert s.fill(parse("delete_files(src=this)")).calls[0].args["_files"] == [b]
    assert _screen(clip=([a], "")).fill(parse("copy_files(src=this, dest=Archive)")).calls[0].args["_from"] == "copied"
    with pytest.raises(ContextError):
        _screen().fill(parse("delete_files(src=this)"))
    s.set_results([{"title": "b.pdf", "path": b, "source": "files"}], now=0)
    with pytest.raises(ContextError):                   # old results don't count
        s.fill(parse("delete_files(src=this)"))


def test_here_and_copied_text(env):
    from jarvis.context import ContextError
    from jarvis.dsl.grammar import parse
    cfg, store, dl, arc = env
    p = _screen(folder=str(dl)).fill(parse("list_files(src=here, ext=pdf)"))
    assert p.calls[0].args["src"] == str(dl)
    with pytest.raises(ContextError):
        _screen().fill(parse("list_files(src=here)"))
    p = _screen(clip=([], "  best biryani in hyderabad ")).fill(parse("search_web(query=this, site=google)"))
    assert p.calls[0].args["query"] == "best biryani in hyderabad"
    with pytest.raises(ContextError):
        _screen(clip=([], "just some words")).fill(parse("open_url(url=this)"))
    assert _screen(clip=([], "https://x.dev/a")).fill(parse("open_url(url=this)")).calls[0].args["url"] == "https://x.dev/a"


def test_move_selection_end_to_end_with_undo(env):
    from jarvis.dsl.grammar import parse
    from jarvis.executor import Executor
    cfg, store, dl, arc = env
    ex = Executor(cfg, store, lambda *a: True, screen=_screen(sel=[str(dl / "a.pdf"), str(dl / "c.png")]))
    rep = ex.run(parse("move_files(src=this, ext=pdf, dest=Archive)"))      # "move these pdfs to archive"
    assert not rep.errors, rep.errors
    assert (arc / "a.pdf").exists() and (dl / "c.png").exists() and (dl / "b.pdf").exists()
    assert rep.results[0].data["src"] == "your selection"
    ex.run(parse("undo()"))
    assert (dl / "a.pdf").exists() and not (arc / "a.pdf").exists()
    ex.screen = _screen(sel=[str(dl / "b.pdf")])
    rep = ex.run(parse('move_files(src=this, new_folder="taxes")'))
    assert (dl / "taxes" / "b.pdf").exists()
    ex.screen = _screen()
    rep = ex.run(parse("move_files(src=this, dest=Archive)"))
    assert rep.failures and "select" in str(rep.failures[0][1]).lower()


def test_list_then_open_second_and_move_these(env, tmp_path, monkeypatch):
    from jarvis.daemon import Jarvis
    cfg, store, dl, arc = env
    j = Jarvis(cfg, lambda *a: True, data_root=tmp_path / "d")
    j.config.show_details = False
    j.screen._explorer = lambda h: None
    j.screen._clip = lambda: ([], "")
    j.handle("list_files(src=Downloads, ext=pdf)")
    rows = j.take_results()
    assert [r["title"] for r in rows] == ["a.pdf", "b.pdf", "invoice_1.pdf"]       # clickable in the bar too
    opened = []
    import os as _os
    monkeypatch.setattr(_os, "startfile", opened.append, raising=False)     # not on Windows here
    j.handle("open the second one")
    assert opened == [str(dl / "b.pdf")]
    assert "only showed 3" in j.handle("open the fifth one")
    j.handle("move these to archive")
    assert sorted(p.name for p in arc.iterdir()) == ["a.pdf", "b.pdf", "invoice_1.pdf"]
    assert "a.pdf" not in [p.name for p in dl.iterdir()]


def test_context_rules_and_model_gate():
    from jarvis.nlu.brain import Brain
    from jarvis.nlu.rule_parser import rule_parse
    cases = {"move these pdfs to archive": "move_files(dest=Archive, ext=pdf, src=this)",
             "delete this file": "delete_files(src=this)",
             "put these into a new folder called taxes": 'move_files(new_folder="taxes", src=this)',
             "show me the pdfs here": "list_files(ext=pdf, src=here)",
             "move pdfs from downloads here": "move_files(dest=here, ext=pdf, src=Downloads)",
             "google this": "search_web(query=this, site=google)",
             "open the link i copied": "open_url(url=this)",
             "open the second one": "open_result(n=2)", "dusra wala kholo": "open_result(n=2)",
             "ise documents me daal do": "move_files(dest=Documents, src=this)",
             "move the pdfs in downloads to documents": "move_files(dest=Documents, ext=pdf, src=Downloads)"}
    for text, want in cases.items():
        assert str(rule_parse(text)) == want, (text, rule_parse(text))

    class Old:
        knows = {"actions": ["move_files", "search_web"], "features": []}
        last_alts = []

        def parse(self, text):
            from jarvis.dsl.grammar import parse
            return parse('search_web(query="this", site=google)'), 0.95
    b = Brain()
    b.model, b.backend = Old(), "fake"
    assert str(b.resolve("google this")[0]) == "search_web(query=this, site=google)"


def test_context_training_data():
    import random
    from datagen.generate_intents import TEACHES, gen_context, Ctx
    from jarvis.dsl.grammar import canonical
    assert "context_refs" in TEACHES
    r = random.Random(1)
    progs = [canonical(gen_context(Ctx(r, "train"))[0]) for _ in range(2000)]
    assert any("src=this" in p for p in progs) and any("src=here" in p for p in progs)
    assert any("dest=here" in p for p in progs) and any("query=this" in p for p in progs)
    assert any(p.startswith("open_result(") for p in progs)


def test_directions_place_cleanup(tmp_path):
    from jarvis.actions.maps import clean_place, directions
    from jarvis.nlu.rule_parser import rule_parse
    assert str(rule_parse("how far is it to sira, tumkur from home")) == 'directions(destination="sira, tumkur", origin="home")'
    assert str(rule_parse("how far is it to ayodhya from home")) == 'directions(destination="ayodhya", origin="home")'
    assert clean_place("it to sira, tumkur") == "sira, tumkur" and clean_place("home") == "home"
    sent, opened = [], []

    def http(url, data=None, headers=None, timeout=8):
        sent.append(data)
        return {"routes": [{"duration": "7200s", "distanceMeters": 120000}]}
    directions({"destination": "it to ayodhya", "origin": "home"}, _maps_ctx(tmp_path, maps_api_key="plain:K"), False,
               http=http, open_url=opened.append)
    assert sent[0]["destination"]["address"] == "ayodhya" and sent[0]["origin"]["address"] == "HSR Layout, Bengaluru"
    assert "destination=ayodhya" in opened[0] and "it+to" not in opened[0]


def test_every_setting_survives_save_and_restart(tmp_path):
    from dataclasses import fields
    from jarvis.config import Config
    cfg = Config(data_dir=tmp_path)
    cfg.places = {"home": "HSR Layout, Bengaluru", "work": "Manyata Tech Park"}
    cfg.maps_api_key, cfg.maps_osm = "dpapi:abc==", False
    cfg.briefing_auto, cfg.clarify, cfg.learn_from_usage = False, False, False
    cfg.memory_folders = ["Downloads", "D:/work"]
    p = cfg.save()
    raw = p.read_text(encoding="utf-8")
    back = Config.load(p)
    for f in fields(cfg):
        if f.name in Config.NOT_SAVED or f.name == "folders":
            continue
        assert getattr(back, f.name) == getattr(cfg, f.name), f.name        # nothing is dropped on restart
    assert "HSR Layout" in raw and "dpapi:abc==" in raw
    import json
    data = json.loads(raw)
    data["from_a_newer_version"] = 1
    p.write_text(json.dumps(data), encoding="utf-8")
    Config.load(p).save(p)
    assert "from_a_newer_version" in p.read_text(encoding="utf-8")         # unknown keys are kept


# ---------------------------------------------------------------- phase 44: any Windows setup
APPS = [{"name": "Visual Studio Code", "target": "VSCODE!App", "kind": "appid"},
        {"name": "WhatsApp", "target": "5319275A.WhatsAppDesktop!App", "kind": "appid"},
        {"name": "Microsoft Word", "target": "C:/sm/Word.lnk", "kind": "shortcut"},
        {"name": "Obsidian", "target": "C:/sm/Obsidian.lnk", "kind": "shortcut"},
        {"name": "Spotify", "target": "SpotifyAB.Spotify!App", "kind": "appid"}]


def _index(tmp_path, entries=APPS):
    from jarvis.apps import AppIndex
    return AppIndex(tmp_path / "apps.json", scanners=(lambda: list(entries),))


def test_app_index_finds_apps_any_way_you_say_them(tmp_path):
    idx = _index(tmp_path)
    assert idx.find("whatsapp")["kind"] == "appid"                  # a Store app
    assert idx.find("word")["name"] == "Microsoft Word"
    assert idx.find("obsdian")["name"] == "Obsidian"                 # typo
    assert idx.find("visual studio")["name"] == "Visual Studio Code"
    assert idx.find("photoshop") is None
    assert "Obsidian" in idx.suggest("obsidan")
    assert (tmp_path / "apps.json").exists()                         # cached for next time


def test_scan_skips_uninstallers(tmp_path):
    from jarvis.apps import scan_shortcuts
    d = tmp_path / "Programs" / "Obsidian"
    d.mkdir(parents=True)
    for n in ("Obsidian.lnk", "Uninstall Obsidian.lnk", "Obsidian Help.url", "notes.txt"):
        (d / n).write_text("x")
    assert [e["name"] for e in scan_shortcuts([tmp_path / "Programs"])] == ["Obsidian"]


def test_open_app_resolution_and_browser_fallback(tmp_path, monkeypatch):
    import jarvis.apps as A
    from jarvis.apps import AppNotFound, resolve
    monkeypatch.setattr(A, "app_paths_has", lambda exe: False)     # as if nothing is registered with Windows
    monkeypatch.setattr(A.os.path, "exists", lambda p: False)
    from jarvis.config import Config
    cfg = Config(data_dir=tmp_path)
    idx = _index(tmp_path)
    edge = {"name": "Microsoft Edge", "alias": "edge", "exe": "C:/edge/msedge.exe"}
    assert resolve("vscode", cfg, idx)[0:3:2] == ("index", "Visual Studio Code")
    assert resolve("settings", cfg, idx)[1]["target"] == "ms-settings:"
    how, entry, label = resolve("chrome", cfg, _index(tmp_path, []), edge)
    assert (how, label) == ("browser", "Microsoft Edge")             # no Chrome here: your default browser
    with pytest.raises(AppNotFound) as e:
        resolve("Obsidan Notes", cfg, idx)
    assert e.value.suggestions == [] or "Obsidian" in e.value.suggestions
    from jarvis.persona.persona import Persona
    p = Persona("Jarvis", "sir", "butler")
    assert "Obsidian" in p.speak(*p.plan_error("open_app", AppNotFound("obsidan", ["Obsidian"])))
    from jarvis.actions.base import ActionResult
    txt = p.result(ActionResult("", kind="open", data={"what": "app", "name": "chrome", "instead": "Microsoft Edge"}))
    assert "isn't installed" in txt and "Microsoft Edge" in txt


def test_one_jarvis_per_windows_user():
    import random
    from jarvis.machine import user_port
    from jarvis.reliability import SingleInstance
    assert user_port("akars") == user_port("Akars") != user_port("guest")
    port = random.randint(30000, 40000)
    a = SingleInstance(port=port, user="akars")
    assert a.acquire()
    try:
        b = SingleInstance(port=port, user="guest")                   # another person signed in on the same PC
        assert b.acquire() and b.port == port + 1
        a2 = SingleInstance(port=port, user="akars")                  # you, starting Jarvis twice
        assert not a2.acquire() and a2.ping_existing()
        b.release()
    finally:
        a.release()


def test_power_and_placement(monkeypatch):
    from jarvis import machine
    from jarvis.config import Config
    cfg = Config()
    assert machine.heavy_work_ok(cfg, machine.Power(None))            # desktop
    assert machine.heavy_work_ok(cfg, machine.Power(False))           # plugged in
    assert not machine.heavy_work_ok(cfg, machine.Power(True))        # on battery
    assert not machine.heavy_work_ok(cfg, machine.Power(False, saver=True))
    cfg.pause_heavy_on_battery = False
    assert machine.heavy_work_ok(cfg, machine.Power(True))
    monkeypatch.setattr(machine, "work_area", lambda x=None, y=None: (1920, 0, 3840, 1040) if x is None else (0, 0, 1920, 1040))
    assert machine.place(760, 0, "center") == (1920 + (1920 - 760) // 2, int(1040 * 0.18))   # second monitor
    assert machine.place(340, 500, "top-right", margin=24) == (1920 - 340 - 24, 24)           # HUD: primary
    assert machine.place(380, 0, "toast") == (3840 - 380 - 20, 1040 - 20)                     # above the taskbar


def test_moved_standard_folders_are_allowed(tmp_path, monkeypatch):
    import jarvis.config as C
    docs = tmp_path / "D_drive" / "Documents"
    docs.mkdir(parents=True)
    monkeypatch.setattr(C, "known_folder", lambda n: docs if n == "Documents" else None)
    roots = C.with_standard_folders([str(tmp_path / "home")], {"Documents": str(docs), "Work": str(tmp_path / "x")})
    assert str(docs) in roots and str(tmp_path / "x") not in roots    # only Windows' own folders, at their real place


def test_setup_check_reports_and_reruns_on_new_pc(tmp_path, monkeypatch):
    from jarvis import machine
    from jarvis.config import Config
    cfg = Config(data_dir=tmp_path, folders={"Downloads": str(tmp_path), "Documents": str(tmp_path / "missing")})
    r = machine.scan(cfg, _index(tmp_path))
    assert r["apps"] == 5 and any("Documents isn't at" in n for n in r["notes"])
    assert "Set up for" in machine.summary(r)
    assert machine.needs_setup(tmp_path)
    machine.save_report(tmp_path, r)
    assert not machine.needs_setup(tmp_path)
    monkeypatch.setattr(machine, "fingerprint", lambda: "laptop|akars")      # Jarvis restored on another PC
    assert machine.needs_setup(tmp_path)


# ---------------------------------------------------------------- phase 41: speed & footprint
def test_int8_copy_is_picked_only_when_allowed(tmp_path):
    from jarvis import onnxutil
    full = tmp_path / "intent_encoder.onnx"
    full.write_bytes(b"x")
    assert onnxutil.pick(full) == full
    onnxutil.int8_path(full).write_bytes(b"x")
    assert onnxutil.pick(full).name == "intent_encoder.int8.onnx"
    assert onnxutil.pick(full, int8=False) == full
    old, onnxutil.USE_INT8 = onnxutil.USE_INT8, False
    try:
        assert onnxutil.pick(full) == full
    finally:
        onnxutil.USE_INT8 = old


def test_lazy_model_loads_once_on_first_use():
    from jarvis.lazy import Lazy
    calls = []

    class M:
        stats = {"generated": 0}

        def knows_key(self, k):
            return True
    lazy = Lazy(lambda: calls.append(1) or M(), "reply model")
    assert calls == [] and not lazy.loaded                 # nothing loaded at start-up
    assert lazy.knows_key("x") and lazy.stats == {"generated": 0} and calls == [1]
    missing = Lazy(lambda: None, "dialog model")
    assert not missing and getattr(missing, "knows_key", "none") == "none"
    broken = Lazy(lambda: 1 / 0, "x")
    assert not broken


def test_parser_time_budget(tmp_path):
    import time as _t
    p = _fake_onnx(tmp_path, "open_app(name=chrome)")
    prog, _ = p.parse("open chrome")
    assert str(prog) == "open_app(name=chrome)" and not p.timed_out
    slow_run = p.dec.run
    p.dec.run = lambda names, feeds: (_t.sleep(0.02), slow_run(names, feeds))[1]
    prog, conf = p.parse("open chrome", budget_s=0.03)
    assert prog is None and p.timed_out


def test_brain_falls_back_to_rules_over_budget_and_while_loading():
    import threading
    from jarvis.nlu.brain import Brain
    from jarvis.perf import PERF

    class Slow:
        knows = {"actions": ["open_app"], "features": []}
        timed_out = False

        def parse(self, text, budget_s=None):
            self.timed_out = True
            return None, 0.0
    b = Brain(budget_s=0.5)
    b.model, b.backend = Slow(), "fake"
    before = PERF.timeouts
    prog, src = b.resolve("open notepad")
    assert str(prog) == "open_app(name=notepad)" and "time budget" in src and PERF.timeouts == before + 1
    b2 = Brain(budget_s=None)
    b2._ready = threading.Event()                         # the model is still loading in the background
    b2._ready.wait = lambda timeout=None: False
    prog, src = b2.resolve("open notepad")
    assert str(prog) == "open_app(name=notepad)" and src == "rules (model still loading)"


def test_perf_records_stages_and_reports(tmp_path):
    from jarvis.perf import Perf, rss_mb
    p = Perf()
    p.mark("command bar ready")
    for ms in (0.01, 0.02, 0.2):
        p.begin()
        with p.stage("understand"):
            import time as _t
            _t.sleep(ms)
        with p.stage("act"):
            pass
        p.command_done("model 97%")
    p.model_loaded("intent model (encoder)", tmp_path / "intent_encoder.int8.onnx", 0.12)
    r = p.report()
    assert "Last 3 commands" in r and "understand" in r and "int8" in r and "command bar ready" in r
    assert rss_mb() is None or rss_mb() > 1


def test_jarvis_perf_command(tmp_path):
    j = _jarvis(tmp_path)
    j.handle('remind(text="stretch", after=10m)')
    out = j.handle("/perf")
    assert "Last 1 commands" in out or "commands: typical" in out
    assert "int8" in out


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")   # Stop ends the loop
def test_yield_pauses_training_while_gaming_and_respects_manual_pause(tmp_path, monkeypatch):
    import jarvis.actions.training as TR
    from jarvis import machine
    flag = tmp_path / "PAUSE"
    monkeypatch.setattr(TR, "PAUSE_FLAG", flag)
    monkeypatch.setattr(TR, "jobs", lambda *a, **k: [{"name": "command model", "state": "running"}])
    state = {"busy": "game"}
    monkeypatch.setattr(machine, "fullscreen_busy", lambda: state["busy"])
    monkeypatch.setattr(machine, "power", lambda: machine.Power(None))
    sleeps = []
    import jarvis.daemon as D

    class Stop(Exception):
        pass

    def fake_sleep(s):
        sleeps.append(s)
        if len(sleeps) == 1:
            state["busy"] = None                          # the game was closed
        if len(sleeps) >= 2:
            raise Stop()
    monkeypatch.setattr(D.time, "sleep", fake_sleep)
    j = _jarvis(tmp_path)
    notes = []
    t = j.start_yield(notes.append, every_s=0)
    t.join(timeout=5)
    assert notes and "game" in notes[0]                   # paused while gaming...
    assert not flag.exists()                              # ...and resumed afterwards
    flag.write_text("paused by Jarvis\n")                 # your own pause is never lifted
    sleeps.clear()
    state["busy"] = None
    t = j.start_yield(notes.append, every_s=0)
    t.join(timeout=5)
    assert flag.exists()


def test_rollback_never_pairs_old_model_with_new_int8(tmp_path, monkeypatch):
    import training.loop as L
    monkeypatch.setattr(L, "CK", tmp_path)
    monkeypatch.setattr(L, "STATE", tmp_path / "state.json")
    old = tmp_path / "versions" / "v000-baseline"
    old.mkdir(parents=True)
    for f in ("intent.pt", "intent_encoder.onnx", "intent_decoder.onnx", "intent_onnx.json"):
        (old / f).write_text("old")
        (tmp_path / f).write_text("new")
    for f in L.INT8:
        (tmp_path / f).write_text("new int8")
    L.make_live(old)
    assert (tmp_path / "intent_encoder.onnx").read_text() == "old"
    assert not any((tmp_path / f).exists() for f in L.INT8)


def test_after_export_drops_stale_int8(tmp_path, monkeypatch):
    import builtins
    from training.quantize import after_export
    (tmp_path / "intent_encoder.int8.onnx").write_text("stale")
    real_import = builtins.__import__

    def no_onnx(name, *a, **k):
        if name == "onnx":
            raise ImportError
        return real_import(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", no_onnx)
    logs = []
    after_export("intent", tmp_path, log=logs.append)
    assert not (tmp_path / "intent_encoder.int8.onnx").exists() and "pip install onnx" in logs[0]


# ---------------------------------------------------------------- phase 42: release safety
def _repos(tmp_path):
    """A 'GitHub' repo and your local clone of it, plus a helper to publish a new version."""
    import subprocess as sp

    def git(cwd, *a):
        return sp.run(["git", *a], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()
    origin, dev, mine = tmp_path / "origin.git", tmp_path / "dev", tmp_path / "mine"
    sp.run(["git", "init", "-q", "--bare", str(origin)], check=True)
    sp.run(["git", "clone", "-q", str(origin), str(dev)], check=True, capture_output=True)
    for d in (dev,):
        git(d, "config", "user.email", "t@t")
        git(d, "config", "user.name", "t")
    (dev / "app.txt").write_text("v1")
    git(dev, "add", ".")
    git(dev, "commit", "-qm", "v1")
    git(dev, "push", "-q", "origin", "HEAD")
    sp.run(["git", "clone", "-q", str(origin), str(mine)], check=True, capture_output=True)

    def publish(text):
        (dev / "app.txt").write_text(text)
        git(dev, "commit", "-qam", text)
        git(dev, "push", "-q", "origin", "HEAD")
        return git(dev, "rev-parse", "HEAD")
    return mine, publish, (lambda: git(mine, "rev-parse", "HEAD"))


def test_update_is_tested_before_switching(tmp_path):
    from jarvis import release
    from jarvis.updater import update
    mine, publish, head = _repos(tmp_path)
    data = tmp_path / "data"
    v1 = head()
    assert update(mine, data_dir=data, tester=lambda *a: (True, "ok"))[1] == "Already up to date."
    v2 = publish("v2 broken")
    changed, msg = update(mine, data_dir=data, tester=lambda *a: (False, "the new version failed its tests"))
    assert not changed and head() == v1 and "didn't install" in msg            # stayed on the working version
    changed, msg = update(mine, data_dir=data, tester=lambda *a: (True, "ok"))
    assert not changed and "didn't pass its checks earlier" in msg              # never retried
    v3 = publish("v3 fixed")
    seen = []
    changed, msg = update(mine, data_dir=data, tester=lambda root, commit, progress: seen.append(commit) or (True, "205 passed"))
    assert changed and head() == v3 and seen == [v3] and "Checked first: 205 passed" in msg
    assert release.pending(data)["previous"] == v1                               # remembered for a rollback


def test_rollback_after_a_failed_start_keeps_your_changes(tmp_path):
    from jarvis import release
    from jarvis.updater import update
    mine, publish, head = _repos(tmp_path)
    data = tmp_path / "data"
    v1 = head()
    (mine / "my_notes.txt").write_text("my own file")                           # something of yours
    v2 = publish("v2 crashes on start")
    assert update(mine, data_dir=data, tester=lambda *a: (True, "ok"))[0]
    done, msg = release.rollback(mine, data, why="crashed while starting (exit code 1)")
    assert done and head() == v1 and (mine / "app.txt").read_text() == "v1"
    assert (mine / "my_notes.txt").read_text() == "my own file"
    assert "crashed while starting" in release.message_once(data) and release.message_once(data) == ""
    assert "didn't pass its checks earlier" in update(mine, data_dir=data, tester=lambda *a: (True, "ok"))[1]


def test_confirmed_update_is_not_rolled_back(tmp_path):
    from jarvis import release
    from jarvis.updater import update
    mine, publish, head = _repos(tmp_path)
    data = tmp_path / "data"
    publish("v2")
    update(mine, data_dir=data, tester=lambda *a: (True, "ok"))
    assert release.confirm(data) and release.pending(data) is None
    assert release.rollback(mine, data) == (False, "nothing to roll back")


def test_supervisor_rolls_back_a_crashing_update(tmp_path, monkeypatch):
    import jarvis.supervisor as S
    codes = iter([1, 0])                                     # new version crashes, the previous one runs
    monkeypatch.setattr(S.subprocess, "call", lambda cmd, env=None: next(codes))
    state = {"pending": True}
    monkeypatch.setattr(S.release, "pending", lambda d: state if state["pending"] else None)

    def fake_rollback(root, data_dir, why=""):
        state["pending"] = False
        state["why"] = why
        return True, "rolled back"
    monkeypatch.setattr(S.release, "rollback", fake_rollback)
    monkeypatch.setattr(S.time, "sleep", lambda s: None)
    assert S.run([], tmp_path) == 0 and "exit code 1" in state["why"]


def test_selftest_passes_and_leaves_registry_alone():
    from jarvis import selftest
    from jarvis.executor import REGISTRY
    before = dict(REGISTRY)
    checks = selftest.run(log=lambda s: None)
    assert all(c.ok for c in checks), selftest.summary(checks)
    assert REGISTRY == before


def test_new_version_is_tested_in_its_own_worktree(tmp_path):
    import subprocess as sp
    from jarvis.updater import test_version
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "jarvis").mkdir()
    (repo / "jarvis" / "__init__.py").write_text("")
    (repo / "jarvis" / "selftest.py").write_text("print('Self-test passed: 1 checks')\n")
    (repo / "tests" / "test_a.py").write_text("def test_ok():\n    assert True\n")

    def git(*a):
        return sp.run(["git", *a], cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    git("init", "-q")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    git("add", ".")
    git("commit", "-qm", "good")
    good = git("rev-parse", "HEAD")
    (repo / "tests" / "test_a.py").write_text("def test_broken():\n    assert 1 == 2\n")
    git("commit", "-qam", "broken")
    broken = git("rev-parse", "HEAD")
    steps = []
    ok, msg = test_version(repo, good, steps.append)
    assert ok and "1 passed" in msg and "Self-test passed" in msg, msg
    ok, msg = test_version(repo, broken, steps.append)
    assert not ok and "test_broken" in msg
    assert "worktree" not in git("status", "--porcelain")                       # nothing left behind
    assert git("worktree", "list").count("\n") == 0


# ---------------------------------------------------------------- phase 30: notes & journal
def test_note_search_matches_words_not_exact_text(tmp_path):
    from jarvis import notes as N
    from jarvis.store import Store
    s = Store(tmp_path / "j.db", encrypt=False)
    s.add_note("note", "call the bank about the card")
    s.add_note("note", "Rahul's number is 98450 12345")
    s.add_note("journal", "good day, finished the installer")
    assert [t for _, _, _, t in N.find(s, "bank")] == ["call the bank about the card"]
    assert [t for _, _, _, t in N.find(s, "cards")] == ["call the bank about the card"]    # cards -> card
    assert [t for _, _, _, t in N.find(s, "rahul number")] == ["Rahul's number is 98450 12345"]
    assert N.find(s, "passport") == []
    assert [t for _, _, _, t in N.find(s, kind="journal")] == ["good day, finished the installer"]
    import time as _t
    s.add_note("note", "old one", ts=_t.time() - 3 * 86400)
    assert "old one" not in [t for _, _, _, t in N.find(s, when="today")]
    assert "old one" in [t for _, _, _, t in N.find(s, when="week")]


def test_notes_end_to_end_with_undo_and_delete(tmp_path):
    import time as _t
    j = _jarvis(tmp_path)
    j.config.show_details = False
    assert "call the bank" in j.handle("note: call the bank about the card")
    assert "journal" in j.handle("add to my journal: good day, finished the installer").lower()
    out = j.handle("what did i note about the bank?")
    assert "call the bank about the card" in out
    rows = j.take_results()
    assert rows and rows[0]["source"] == "note"                      # clickable in the command bar
    j.handle("note: wrong thing")
    j.handle("undo")
    assert "No notes" in j.handle("what did i note about wrong thing")   # undo removes a note you just made
    assert "Deleted" in j.handle("delete the note about the bank")
    assert "No notes" in j.handle("what did i note about the bank")
    j.handle("undo")
    assert "call the bank" in j.handle("what did i note about the bank")  # and brings a deleted one back
    assert "journal" in j.handle("read my journal").lower()


def test_note_this_uses_copied_text(tmp_path):
    j = _jarvis(tmp_path)
    j.config.show_details = False
    j.screen._clip = lambda: ([], "Flight 6E 204, PNR XK7Q2M")
    j.screen._explorer = lambda h: None
    j.handle("note this")
    assert "PNR XK7Q2M" in j.handle("what did i note about the flight")


def test_memory_search_includes_notes(tmp_path):
    j = _jarvis(tmp_path)
    j.config.show_details = False
    j.handle("note: wifi password is tiger42")
    j.handle('search_memory(query="wifi password")')
    rows = j.take_results()
    assert rows and rows[0]["source"] == "note" and "tiger42" in rows[0]["text"]


def test_notes_are_encrypted_at_rest(tmp_path):
    from jarvis.store import Store
    s = Store(tmp_path / "j.db", encrypt=True)
    s.add_note("note", "secret pin 4321")
    raw = (tmp_path / "j.db").read_bytes()
    import sys as _sys
    if _sys.platform == "win32":
        assert b"secret pin 4321" not in raw                          # DPAPI: only your Windows account
    assert s.notes()[0][3] == "secret pin 4321"


def test_evening_briefing_mentions_todays_notes():
    from jarvis.briefing import notes_lines
    lines = notes_lines([("note", "call the bank"), ("note", "buy milk"), ("journal", "good day")])
    assert "You noted 2 things today" in lines[0] and "call the bank" in lines[0]
    assert "journal entry" in lines[1]
    assert notes_lines([]) == []


def test_note_rules_and_training_data():
    import random
    from datagen.generate_intents import TEACHES, gen_notes, Ctx
    from jarvis.dsl.grammar import canonical
    from jarvis.nlu.rule_parser import rule_parse
    cases = {"note: call the bank about the card": 'note(text="call the bank about the card")',
             "add to my journal: good day": 'note(kind=journal, text="good day")',
             "note this": "note(text=this)", "what did i note about the bank?": 'find_notes(query="bank")',
             "show today's notes": "find_notes(when=today)",
             "read my journal from yesterday": "find_notes(kind=journal, when=yesterday)",
             "delete the note about the bank": 'delete_note(query="bank")',
             "kal 5 baje meeting hai note kar lo": 'note(text="kal 5 baje meeting hai")',
             "find the meeting notes pdf": 'search_memory(ext=pdf, query="meeting notes")'}
    for text, want in cases.items():
        assert str(rule_parse(text)) == want, (text, rule_parse(text))
    assert "notes" in TEACHES
    r = random.Random(4)
    progs = [canonical(gen_notes(Ctx(r, "train"))[0]) for _ in range(1500)]
    for start in ("note(", "find_notes(", "delete_note("):
        assert any(p.startswith(start) for p in progs)
    assert any("kind=journal" in p for p in progs) and any("when=" in p for p in progs)


# ---------------------------------------------------------------- phase 31: workspaces
class FakeDesktop:
    """Two monitors and some app windows; launching an app makes its window appear."""

    def __init__(self, wins=None, monitors=None):
        from jarvis.winlayout import Win
        self.Win = Win
        self.wins = wins if wins is not None else []
        self.mons = monitors or [[0, 0, 1920, 1040], [1920, 0, 3840, 1040]]
        self.placed, self.closed, self.launched, self.played = {}, [], [], []
        self._next = 100

    def add(self, exe, title, rect=(100, 100, 900, 700), state="normal"):
        self._next += 1
        w = self.Win(self._next, exe, title, list(rect), state)
        self.wins.append(w)
        return w

    def windows(self):
        return [w for w in self.wins if w.hwnd not in self.closed]

    def place(self, hwnd, rect, state):
        self.placed[hwnd] = (list(rect), state)
        for w in self.wins:
            if w.hwnd == hwnd:
                w.rect, w.state = list(rect), state

    def close(self, hwnd):
        self.closed.append(hwnd)

    def launch(self, exe):
        self.launched.append(exe)
        self.add(exe, "new " + exe)

    def open_app(self, alias):
        from jarvis.workspaces import APP_EXES
        self.launched.append(alias)
        self.add("C:/apps/" + sorted(APP_EXES.get(alias, {alias + ".exe"}))[0], alias)

    def play(self, q):
        self.played.append(q)

    def monitors(self):
        return self.mons

    def wait_new(self, known, app=None, timeout=0):
        return next((w for w in self.windows() if w.hwnd not in known and (app is None or w.app == app)), None)


def test_save_and_restore_a_layout_across_two_monitors(tmp_path):
    from jarvis.workspaces import Workspaces
    d = FakeDesktop()
    code = d.add("C:/VS Code/Code.exe", "jarvis - Visual Studio Code", (0, 0, 1920, 1040), "max")
    chrome = d.add("C:/Chrome/chrome.exe", "GitHub - Chrome", (1920, 0, 3000, 1040))
    wt = tmp_path / "WT" / "WindowsTerminal.exe"
    wt.parent.mkdir()
    wt.write_text("x")                                         # a program that exists on disk can be reopened
    term = d.add(str(wt), "PowerShell", (3000, 0, 3840, 1040))
    ws = Workspaces(tmp_path / "ws.json", d)
    key, definition, prev = ws.save("my work layout")
    assert key == "work" and prev is None and len(definition["windows"]) == 3
    assert [e["monitor"] for e in definition["windows"]] == [0, 1, 1]
    for w in d.wins:                                           # you move things around...
        w.rect, w.state = [10, 10, 500, 500], "normal"
    d.closed.append(term.hwnd)                                 # ...and close the terminal
    rep = ws.start("work mode")
    assert d.placed[code.hwnd] == ([0, 0, 1920, 1040], "max")
    assert d.placed[chrome.hwnd] == ([1920, 0, 3000, 1040], "normal")
    assert rep.opened == ["Terminal"] and str(wt) in d.launched                       # reopened
    assert not rep.missing


def test_preset_opens_missing_apps_tiles_and_plays_music(tmp_path):
    from jarvis.workspaces import Workspaces
    d = FakeDesktop(monitors=[[0, 0, 1920, 1040]])
    d.add("C:/Chrome/chrome.exe", "Gmail - Chrome")            # already open: reused, not opened again
    rep = Workspaces(tmp_path / "ws.json", d).start("coding")
    assert d.launched == ["cursor", "claude", "terminal"] and rep.opened == ["Cursor", "Claude", "Terminal"]
    assert rep.music and d.played == [rep.music]
    places = sorted(d.placed.values())
    assert ([0, 0, 1152, 1040], "normal") in places          # the main app gets 60% on the left
    assert len(rep.arranged) == 4
    d2 = FakeDesktop(monitors=[[0, 0, 1920, 1040]])            # your own list from Settings
    rep = Workspaces(tmp_path / "ws2.json", d2, starter_apps={"coding": ["claude", "chrome"]}).start("coding")
    assert d2.launched == ["claude", "chrome"] and "VS Code" not in rep.opened


def test_tiling_rules():
    from jarvis.workspaces import tile
    one = [[0, 0, 1000, 1000]]
    assert tile(1, one) == [([0, 0, 1000, 1000], "max")]
    assert tile(3, one) == [([0, 0, 600, 1000], "normal"), ([600, 0, 1000, 500], "normal"),
                            ([600, 500, 1000, 1000], "normal")]
    two = [[0, 0, 1000, 1000], [1000, 0, 2000, 1000]]
    assert tile(2, two) == [([0, 0, 1000, 1000], "max"), ([1000, 0, 2000, 1000], "max")]


def test_workspace_actions_through_jarvis_with_undo(tmp_path, monkeypatch):
    import jarvis.actions.workspace as WA
    d = FakeDesktop(monitors=[[0, 0, 1920, 1040]])
    a = d.add("C:/x/Code.exe", "proj - VS Code", (0, 0, 1200, 1040))
    b = d.add("C:/x/chrome.exe", "docs - Chrome", (1200, 0, 1920, 1040))
    monkeypatch.setattr(WA, "BACKEND", d)
    j = _jarvis(tmp_path)
    j.config.show_details = False
    out = j.handle('workspace(action=save, name="writing")')
    assert "Saved" in out and "two windows" in out and "Chrome and VS Code" in out
    a.rect, b.rect = [5, 5, 50, 50], [60, 60, 90, 90]
    out = j.handle('workspace(action=start, name="writing")')
    assert "Writing mode is ready" in out and d.placed[a.hwnd][0] == [0, 0, 1200, 1040]
    j.handle("undo")                                            # windows go back to where they were
    assert d.placed[a.hwnd][0] == [5, 5, 50, 50]
    assert "writing" in j.handle("workspace(action=list)") and "coding" in j.handle("workspace(action=list)")
    out = j.handle('workspace(action=start, name="gardening")')
    assert "don't have a gardening workspace" in out
    asked = []
    j.executor.confirm = lambda *x: asked.append(x) or True
    out = j.handle('workspace(action=close, name="writing")')
    assert asked and sorted(d.closed) == sorted([a.hwnd, b.hwnd])   # asked first, then closed both
    j.handle('workspace(action=delete, name="writing")')
    from jarvis.workspaces import Workspaces
    assert "writing" not in Workspaces(tmp_path / "workspaces.json", d).saved()
    j.handle("undo")                                            # the deleted layout comes back
    assert "writing" in Workspaces(tmp_path / "workspaces.json", d).saved()


def test_workspace_rules_and_training_data():
    import random
    from datagen.generate_intents import TEACHES, gen_workspace, Ctx
    from jarvis.dsl.grammar import canonical
    from jarvis.nlu.rule_parser import rule_parse
    cases = {"start coding mode": 'workspace(action=start, name="coding")',
             "coding mode chalu karo": 'workspace(action=start, name="coding")',
             "save this layout as work": 'workspace(action=save, name="work")',
             "close coding mode": 'workspace(action=close, name="coding")',
             "list my workspaces": "workspace(action=list)",
             "delete the work workspace": 'workspace(action=delete, name="work")'}
    for text, want in cases.items():
        assert str(rule_parse(text)) == want, (text, rule_parse(text))
    for other in ("turn on airplane mode", "dark mode", "focus mode", "gaming mode on", "switch on night mode"):
        assert "workspace" not in str(rule_parse(other))              # never mistaken for a workspace
    assert "workspaces" in TEACHES
    r = random.Random(2)
    progs = [canonical(gen_workspace(Ctx(r, "train"))[0]) for _ in range(800)]
    for act in ("start", "save", "close", "delete", "list"):
        assert any(f"action={act}" in p for p in progs)


def test_workspace_names_normalise():
    from jarvis.workspaces import norm
    assert norm("Coding Mode") == "coding" == norm("my coding workspace") == norm("code")
    assert norm("video editing") == "video editing"


# ---------------------------------------------------------------- phase 32: focus & gaming
class _Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self):
        return self.t


def test_focus_single_session_and_do_not_disturb():
    from jarvis.focus import Engine
    clk, logged = _Clock(), []
    e = Engine(clock=clk, log=lambda k, d: logged.append((k, d)))
    e.start(45)
    assert e.countdown() == "FOCUS 45:00" and e.quiet("briefing") and not e.quiet("reminder")
    assert e.hold("toast", "Model update: v003 is live") and not e.hold("reminder", "stand up")
    clk.t += 20 * 60
    assert e.tick() == [] and e.status()["focused_min"] == 20
    clk.t += 25 * 60 + 1
    (ev, summary), = e.tick()
    assert ev == "done" and summary["minutes"] == 45 and summary["held"] == [("toast", "Model update: v003 is live")]
    assert e.session is None and not e.quiet("briefing")
    assert logged == [("focus_session", {"minutes": 45.0, "mode": "single"})]   # not "focus": that's app time


def test_pomodoro_cycles_with_a_long_break_every_fourth():
    from jarvis.focus import Engine
    clk = _Clock()
    e = Engine(clock=clk)
    e.start(mode="pomodoro")
    seen = []
    for _ in range(8):
        clk.t = e.session.phase_end + 1
        for ev, d in e.tick():
            seen.append((ev, d.get("minutes")))
    assert seen == [("break", 5), ("work", 25), ("break", 5), ("work", 25), ("break", 5), ("work", 25),
                    ("break", 15), ("work", 25)]
    assert not e.quiet("toast") or e.session.phase == "work"
    summary = e.stop()
    assert summary["cycles"] == 5 and summary["minutes"] >= 100


def test_gentle_blocking_matches_whole_words_only_while_focusing():
    from jarvis.focus import Engine
    e = Engine(clock=_Clock())
    assert e.blocked("YouTube - Google Chrome", "C:/chrome.exe") is None          # not focusing
    e.start(30, block=True)
    assert e.blocked("lofi beats - YouTube - Google Chrome", "C:/Chrome/chrome.exe") == "Youtube"
    assert e.blocked("Steam", "C:\\\\Steam\\\\steam.exe") == "Steam"
    assert e.blocked("Netflixing the docs - Notion", "notion.exe") is None        # whole words only
    assert e.blocked("jarvis - Cursor", "cursor.exe") is None
    e.start(30, block=False)
    assert e.blocked("YouTube - Chrome", "chrome.exe") is None


def test_gaming_mode_holds_notices_and_summarises():
    from jarvis.focus import Engine, held_summary
    clk = _Clock()
    e = Engine(clock=clk)
    assert e.gaming_on(auto=True) and not e.gaming_on()
    assert e.hold("briefing", "Good evening. 3 reminders today.") and not e.hold("reminder", "call mom")
    clk.t += 40 * 60
    s = e.gaming_off()
    assert s["minutes"] == 40 and "1 briefing came in" in held_summary(s["held"])


def test_focus_through_jarvis_ask_status_stop_undo(tmp_path):
    j = _jarvis(tmp_path)
    j.config.show_details = False
    out = j.handle("focus for 45 minutes")
    assert "Focusing for 45 minutes" in out and "reminders still come through" in out
    assert "left" in j.handle("how long is left")
    assert not j.notify_or_hold("briefing", "Good evening")                       # held during focus
    out = j.handle("stop focus")
    assert "Focus ended" in out and "Good evening" in out                         # ...and summarised after
    asked = []
    j.executor.confirm = lambda *a: asked.append(a) or True
    out = j.handle("focus for an hour and block youtube")
    assert asked and "youtube" in str(asked[0]).lower() and "pause" in out.lower()   # blocking asks first
    j.handle("undo")
    assert j.focus.session is None                                                # undo ends it
    assert "gaming mode on" in j.handle("gaming mode on").lower()
    assert "gaming mode off" in j.handle("gaming mode off").lower()


def test_focus_ticker_minimizes_a_blocked_window(tmp_path, monkeypatch):
    import jarvis.daemon as D
    from jarvis import machine, winlayout
    j = _jarvis(tmp_path)
    j.focus.start(30, block=True)
    fg = winlayout.Win(77, "C:/Chrome/chrome.exe", "Shorts - YouTube - Google Chrome", [0, 0, 1, 1])
    monkeypatch.setattr(winlayout, "foreground", lambda: fg)
    minimized, said = [], []
    monkeypatch.setattr(winlayout, "minimize", minimized.append)
    monkeypatch.setattr(machine, "fullscreen_busy", lambda: None)

    class Stop(BaseException):
        pass

    def one_tick(s):
        raise Stop()
    monkeypatch.setattr(D.time, "sleep", one_tick)
    import threading
    errors = []
    old = threading.excepthook
    threading.excepthook = lambda a: errors.append(a.exc_type)
    try:
        j.start_focus_ticker(said.append).join(timeout=5)
    finally:
        threading.excepthook = old
    assert minimized == [77] and "on pause until" in said[0]


def test_evening_briefing_counts_focus_time_not_app_time(tmp_path):
    from jarvis.briefing import gather
    from jarvis.store import Store
    import time as _t
    s = Store(tmp_path / "j.db", encrypt=False)
    s.log_event("focus", {"app": "chrome.exe"})                                   # app-focus tracking
    s.log_event("focus_session", {"minutes": 50.0, "mode": "single"})
    s.log_event("focus_session", {"minutes": 25.0, "mode": "pomodoro"})
    f = gather("evening", s, None, {}, tmp_path / "none.txt", None, _t.time())
    assert f.focus_min == 75.0


def test_focus_rules_and_training_data():
    import random
    from datagen.generate_intents import TEACHES, gen_focus, Ctx
    from jarvis.dsl.grammar import canonical
    from jarvis.nlu.rule_parser import rule_parse
    cases = {"focus for 45 minutes": "focus(action=start, minutes=45)",
             "deep work for 1.5 hours": "focus(action=start, minutes=90)",
             "start a pomodoro": "focus(action=start, mode=pomodoro)",
             "focus for an hour and block youtube": "focus(action=start, block=yes, minutes=60)",
             "aadha ghanta focus karna hai": "focus(action=start, minutes=30)",
             "stop focus": "focus(action=stop)", "how long is left": "focus(action=status)",
             "gaming mode on": "gaming(action=on)", "done gaming": "gaming(action=off)"}
    for text, want in cases.items():
        assert str(rule_parse(text)) == want, (text, rule_parse(text))
    assert "workspace" not in str(rule_parse("focus mode")) and "workspace" not in str(rule_parse("gaming mode"))
    assert "focus_gaming" in TEACHES
    r = random.Random(3)
    progs = [canonical(gen_focus(Ctx(r, "train"))[0]) for _ in range(1000)]
    for bit in ("minutes=", "mode=pomodoro", "block=yes", "quiet=total", "action=stop", "action=status", "gaming(action=on)"):
        assert any(bit in p for p in progs), bit


# ---------------------------------------------------------------- phase 33: tasks & deadlines
def test_deadlines_in_plain_words():
    from datetime import datetime
    from jarvis.dates import next_occurrence, nice, parse_due
    now = datetime(2026, 11, 27, 9, 50)                        # a Friday
    def d(t):
        r = parse_due(t, now)
        return (datetime.fromtimestamp(r[0]).strftime("%a %d %b %H:%M"), r[1]) if r else None
    assert d("friday") == ("Fri 27 Nov 23:59", False) and d("by monday") == ("Mon 30 Nov 23:59", False)
    assert d("next monday") == ("Mon 30 Nov 23:59", False) and d("next friday") == ("Fri 04 Dec 23:59", False)
    assert d("tomorrow 5 pm") == ("Sat 28 Nov 17:00", True) and d("kal shaam") == ("Sat 28 Nov 18:00", True)
    assert d("5th december") == d("december 5") == d("5/12") == ("Sat 05 Dec 23:59", False)
    assert d("the 1st") == ("Tue 01 Dec 23:59", False) and d("end of month") == ("Mon 30 Nov 23:59", False)
    assert d("at 8 am") == ("Sat 28 Nov 08:00", True)           # already past today: tomorrow
    assert d("5 baje") == ("Fri 27 Nov 17:00", True) and d("garbage") is None
    assert nice(parse_due("tomorrow 5 pm", now)[0], True, now.timestamp()) == "tomorrow 5 pm"
    assert datetime.fromtimestamp(next_occurrence(datetime(2026, 11, 27, 18).timestamp(), "weekdays")).day == 30
    assert datetime.fromtimestamp(next_occurrence(datetime(2026, 1, 31, 18).timestamp(), "monthly")).day == 28


def test_tasks_end_to_end_with_undo(tmp_path):
    j = _jarvis(tmp_path)
    j.config.show_details = False
    out = j.handle("add task finish the report by friday")
    assert "finish the report" in out and "due" in out
    j.handle("add task pay rent every month on the 1st")
    j.handle("add task renew passport, it's important")
    out = j.handle("show my tasks")
    assert "finish the report" in out and "pay rent" in out and "renew passport" in out
    rows = j.take_results()
    assert rows and {r["source"] for r in rows} == {"task"}       # clickable rows in the command bar
    out = j.handle("done with the report")
    assert "Done" in out and "finish the report" not in j.handle("show my tasks")
    j.handle("undo")
    assert "finish the report" in j.handle("show my tasks")       # undo un-ticks it
    out = j.handle("mark rent as done")
    assert "next one is due" in out.lower()                       # a recurring task moves on instead
    assert "pay rent" in j.handle("show my tasks")
    out = j.handle("move the report to monday")
    assert "Moved" in out and ("Monday" in out or "tomorrow" in out or "today" in out)   # whatever day it is
    asked = []
    j.executor.confirm = lambda *a: asked.append(a) or True
    j.handle("delete the task passport")
    assert asked and "renew passport" not in j.handle("show my tasks")
    j.handle("undo")
    assert "renew passport" in j.handle("show my tasks")
    assert "can't find an open task" in j.handle("done with the moon landing")


def test_overdue_nudge_once_and_on_the_hud(tmp_path):
    import time as _t
    j = _jarvis(tmp_path)
    now = _t.time()
    j.store.add_task("send the invoice", due=now - 60, has_time=True)
    j.store.add_task("buy milk", due=now + 3600, has_time=True)
    said = []
    assert j.nudge_overdue(said.append, now) == 1 and "send the invoice" in said[0] and "send the" in said[0]
    assert j.nudge_overdue(said.append, now + 30) == 0              # only once
    items = j.hud_items(now)
    assert any(len(i) == 3 and i[2] == "overdue" and "send the invoice" in i[1] for i in items)


def test_tasks_in_the_briefings(tmp_path):
    import time as _t
    from datetime import datetime
    from jarvis.actions.tasks import briefing_lines
    from jarvis.store import Store
    s = Store(tmp_path / "j.db", encrypt=False)
    noon = datetime.now().replace(hour=12, minute=0, second=0, microsecond=0).timestamp()
    s.add_task("submit the assignment", due=noon + 11 * 3600, has_time=False)
    s.add_task("call the bank", due=noon - 2 * 86400)
    morning = briefing_lines(s, "morning", noon)
    assert any("Due today: submit the assignment" in l for l in morning)
    assert any("Overdue: call the bank" in l for l in morning)
    evening = briefing_lines(s, "evening", noon)
    assert any("Still open from today" in l for l in evening)


def test_task_rules_dont_steal_file_moves():
    import random
    from datagen.generate_intents import TEACHES, gen_tasks, Ctx
    from jarvis.dsl.grammar import canonical
    from jarvis.nlu.rule_parser import rule_parse
    cases = {"add task finish the report by friday": 'task(action=add, due="friday", text="finish the report")',
             "add task pay rent every month on the 1st": 'task(action=add, due="the 1st", every=monthly, text="pay rent")',
             "report karna hai friday tak": 'task(action=add, due="friday", text="report")',
             "aaj kya karna hai": "task(action=list, when=today)",
             "done with the report": 'task(action=done, text="report")',
             "move the report to monday": 'task(action=move, due="monday", text="report")',
             "what's overdue": "task(action=list, when=overdue)",
             "move these to archive": "move_files(dest=Archive, src=this)",
             "move the pdfs in downloads to documents": "move_files(dest=Documents, ext=pdf, src=Downloads)",
             "i'm done focusing": "focus(action=stop)",
             "remind me to call mom in 10 minutes": 'remind(after=10m, text="call mom")'}
    for text, want in cases.items():
        assert str(rule_parse(text)) == want, (text, rule_parse(text))
    assert "tasks" in TEACHES
    r = random.Random(6)
    progs = [canonical(gen_tasks(Ctx(r, "train"))[0]) for _ in range(1500)]
    for bit in ("action=add", "due=", "every=", "priority=high", "action=done", "action=move", "action=delete", "when=today"):
        assert any(bit in p for p in progs), bit


# ---------------------------------------------------------------- phase 43: your words
def test_indian_scripts_read_as_hinglish():
    from jarvis.indic import has_indic, to_latin
    assert to_latin("डाउनलोड्स खोलो") == "downloads kholo"
    assert to_latin("आज क्या करना है") == "aaj kya karna hai"
    assert to_latin("कमरा") == "kamra" and to_latin("चलाओ") == "chalao" and to_latin("वाला") == "wala"
    assert to_latin("अरिजीत सिंह का गाना बजाओ") == "arijit singh ka gana bajao"
    assert to_latin("१० मिनट") == "10 minute"
    assert to_latin("open डाउनलोड्स please") == "open downloads please"
    assert to_latin("ಡೌನ್ಲೋಡ್ಸ್") == "daunlods" and has_indic("ಡೌನ್")       # Kannada via the shared layout
    assert to_latin("plain english") == "plain english"


def test_nicknames_teach_use_and_undo(tmp_path):
    j = _jarvis(tmp_path)
    j.config.show_details = False
    out = j.handle("when I say dl I mean downloads")
    assert "dl" in out and "downloads" in out
    assert j.read_as("show pdfs in dl") == "show pdfs in downloads"
    assert j.read_as("handle the dl-folder") == "handle the dl-folder"      # whole words only
    assert "dl" in j.handle("show my nicknames")
    j.handle("undo")
    assert j.read_as("show pdfs in dl") == "show pdfs in dl"
    j.handle("dl means downloads")
    j.handle("forget dl")
    assert j.read_as("dl") == "dl"
    assert j.read_as("डाउनलोड्स खोलो") == "downloads kholo"


def test_typos_are_learned_from_your_commands():
    from jarvis.vocab import learn_typos
    known = {"downloads", "open", "folder", "music", "spotify", "move", "pdfs", "documents"}
    texts = ["opne donwloads", "move pdfs to documnets", "opne music", "donwloads folder", "pavazha malli",
             "pavazha malli", "documnets please", "spotfy"]
    assert learn_typos(texts, known) == {"documnets": "documents", "donwloads": "downloads", "opne": "open"}  # twice+
    assert "pavazha" not in learn_typos(texts, known)                     # not close to anything: not a typo


def test_vocab_refresh_finds_projects_and_folders(tmp_path):
    from jarvis.config import Config
    from jarvis.vocab import Vocab
    home = tmp_path / "home"
    for p in ("prototypes/everway/.git", "prototypes/jarvis-0/.git", "prototypes/jarvis-0/sub/.git",
              "AppData/Local/thing/.git", "Documents/Tax 2026", "Documents/Resumes"):
        (home / p).mkdir(parents=True)
    cfg = Config(data_dir=tmp_path / "data", folders={"Documents": str(home / "Documents")})
    cfg.data_dir.mkdir(parents=True)
    v = Vocab.load(cfg.data_dir)
    stats = v.refresh(cfg, home=home)
    assert set(v.data["projects"]) == {"everway", "jarvis-0"}              # not AppData, not a project's subfolder
    assert {"Tax 2026", "Resumes"} <= set(v.data["folders"]) and stats["projects"] == 2
    cfg.learned_places = v.places()
    assert cfg.resolve_folder("everway") == home / "prototypes" / "everway"
    assert cfg.resolve_folder("Everway project") == home / "prototypes" / "everway"
    assert cfg.resolve_folder("tax 2026") == home / "Documents" / "Tax 2026"


def test_personal_training_examples(tmp_path):
    import random
    from datagen.generate_intents import GENERATORS, Ctx
    from datagen.personal import personal_generators
    from jarvis.dsl.grammar import canonical
    vocab = {"apps": ["Obsidian", "Spotify", "WhatsApp Beta"], "projects": {"everway": "C:/p/everway"},
             "folders": {"Tax 2026": "C:/d/Tax 2026"}, "typos": {"donwloads": "downloads"}}
    gens = personal_generators(vocab, list(GENERATORS))
    assert len(gens) == 3
    r = random.Random(1)
    seen = []
    for fn, _ in gens:
        for _ in range(60):
            prog, text, _ = fn(Ctx(r, "train"))
            canonical(prog)
            seen.append((text.replace("\x01", "").replace("\x02", ""), prog))
    progs = " ".join(p for _, p in seen)
    assert 'open_app(name="obsidian")' in progs and 'name="spotify"' not in progs     # known apps keep their label
    assert 'open_folder(path="everway")' in progs and 'dest="tax 2026"' in progs
    assert any("donwloads" in t for t, _ in seen)
    assert personal_generators({}, list(GENERATORS)) == []


def test_vocab_rules():
    from jarvis.nlu.rule_parser import rule_parse
    cases = {"when I say dl I mean downloads": 'vocab(action=teach, means="downloads", word="dl")',
             "learn that ammu means amrutha": 'vocab(action=teach, means="amrutha", word="ammu")',
             "jab main dl bolun to downloads samajhna": 'vocab(action=teach, means="downloads", word="dl")',
             "forget dl": 'vocab(action=forget, word="dl")', "show my nicknames": "vocab(action=list)",
             "open my everway project": 'open_folder(path="everway")',
             "forget the note about the bank": 'delete_note(query="bank")'}
    for text, want in cases.items():
        assert str(rule_parse(text)) == want, (text, rule_parse(text))
    assert rule_parse("this means nothing") is None


def test_personal_examples_reach_the_dataset(monkeypatch):
    import random
    import datagen.generate_intents as G
    monkeypatch.setattr(G, "_FNS", G._FNS)
    monkeypatch.setattr(G, "_WEIGHTS", G._WEIGHTS)
    assert G.use_personal({"projects": {"zorblax": "C:/p/zorblax"}}) == 1
    r = random.Random(0)
    progs = [G.sample(r)["program"] for _ in range(4000)]
    assert any('"zorblax"' in p for p in progs)
