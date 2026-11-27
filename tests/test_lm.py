"""Phase 27: Jarvis-LM. The NumPy runtime, chat format and filter are tested here; tests marked `torch` check the
trainer against the runtime and run wherever PyTorch is installed (the RTX 5070 PC, including the update check)."""
import json

import numpy as np
import pytest

from training.tokenizer import BOS, EOS, SEP, BPETokenizer


def _tok():
    texts = ["U: hi there", "J: Good evening, sir.", "Jarvis|sir|butler|evening", "tell me a joke please",
             "[web] [cmd] who invented the telephone"] * 20
    return BPETokenizer.train(texts, 400, verbose=False)


def _weights(vocab, d=32, layers=2, hidden=64, seed=0):
    r = np.random.default_rng(seed)
    w = {"tok_emb": r.normal(0, 0.3, (vocab, d)), "norm": 1 + r.normal(0, 0.1, d)}
    for i in range(layers):
        w.update({f"l{i}.attn_norm": 1 + r.normal(0, 0.1, d), f"l{i}.mlp_norm": 1 + r.normal(0, 0.1, d)})
        for k, shape in (("wq", (d, d)), ("wk", (d, d)), ("wv", (d, d)), ("wo", (d, d)), ("w1", (d, hidden)),
                         ("w3", (d, hidden)), ("w2", (hidden, d))):
            w[f"l{i}.{k}"] = r.normal(0, 0.2, shape)
    return w


def test_kv_cache_matches_full_recompute():
    from jarvis.nlg.chat import Decoder
    dec = Decoder(_weights(300), 32, 2, 4)
    ids = [BOS, 10, 57, 120, 9, 200, 33, 41]
    full, _ = dec.forward(ids, all_logits=True)
    logits, cache = dec.forward(ids[:5])
    np.testing.assert_allclose(logits, full[4], rtol=1e-4, atol=1e-4)
    for i in range(5, len(ids)):                          # one token at a time, reusing the cache
        logits, cache = dec.forward([ids[i]], cache)
        np.testing.assert_allclose(logits, full[i], rtol=1e-4, atol=1e-4)
    before = cache[0][0].copy()
    dec.forward([7], cache)
    assert np.array_equal(cache[0][0], before)            # a prefill can feed several sampled replies


def test_causal_future_tokens_dont_leak():
    from jarvis.nlg.chat import Decoder
    dec = Decoder(_weights(300), 32, 2, 4)
    a, _ = dec.forward([BOS, 5, 6, 7], all_logits=True)
    b, _ = dec.forward([BOS, 5, 6, 250], all_logits=True)
    np.testing.assert_allclose(a[:3], b[:3], atol=1e-6)


def test_dialog_format_masks_only_replies_and_trims_old_turns():
    from jarvis.nlg.chatfmt import encode_dialog, header
    tok = _tok()
    head = header("Friday", "boss", "friendly", 20)
    assert head == "Friday|boss|friendly|evening" and header("A|B", "", "weird", 3) == "A B||butler|night"
    turns = [("U", "hi"), ("J", "Hey boss!"), ("U", "tell me a joke"), ("J", "Why did the PC sleep?")]
    ids, mask = encode_dialog(tok, head, turns)
    assert ids[0] == BOS and ids[-1] == EOS and len(ids) == len(mask)
    said = tok.decode([t for t, m in zip(ids, mask) if m and t >= 4])
    assert said == "Hey boss!Why did the PC sleep?"
    assert sum(1 for t, m in zip(ids, mask) if m and t == SEP) == 1   # the first reply ends with <sep>, the last with <eos>
    ids2, mask2 = encode_dialog(tok, head, turns[:3], open_reply=True)
    assert tok.decode(ids2[-3:]).endswith("J: ") and not any(mask2[-3:])
    short, _ = encode_dialog(tok, head, turns * 10, max_len=60)
    assert len(short) <= 60 and short[0] == BOS and tok.decode(short).startswith("Friday|boss")


def test_reply_filter():
    from jarvis.nlg.chatfmt import parse_reply, problems
    assert problems("Good evening, sir. What can I do for you?") == []
    assert problems("Well done! That's brilliant news.") == []
    assert "claims an action" in problems("I've deleted the files for you.")
    assert "claims an action" in problems("Done.")
    assert "states the time or date" in problems("It's 5:30 pm, sir.")
    assert "link or address" in problems("Check https://example.com for that.")
    assert "repeats itself" in problems("I am here I am here I am here I am here")
    assert "format leak" in problems("Sure. U: thanks")
    assert "language" in problems("what the fuck")
    assert problems("") == ["empty"]
    assert parse_reply("[web] who invented the telephone") == ("web", "who invented the telephone")
    assert parse_reply(" [cmd]") == ("cmd", "") and parse_reply("Hello!")[0] == "say"


def test_chat_model_reply_loop_and_load(tmp_path):
    from jarvis.nlg.chat import ChatModel, Decoder
    tok = _tok()
    w = _weights(tok.vocab_size)
    m = ChatModel(Decoder(w, 32, 2, 4), tok, seed=1)
    out = m.reply("hi", history=[("U", "hello"), ("J", "Good evening, sir.")], hour=20)
    assert out is None or out[0] in ("say", "web", "cmd")            # random weights: any outcome, never a crash
    assert m.stats["ms"] > 0
    np.savez(tmp_path / "lm_chat.npz", **{k: v.astype(np.float16) for k, v in w.items()})
    tok.save(tmp_path / "lm_tokenizer.json")
    meta = {"d_model": 32, "n_layer": 2, "n_head": 4, "max_ctx": 128, "ready": False}
    (tmp_path / "lm_chat.json").write_text(json.dumps(meta))
    assert ChatModel.load(tmp_path) is None                           # trained but below the bar: not used
    assert ChatModel.load(tmp_path, require_ready=False) is not None
    assert ChatModel.load(tmp_path / "missing") is None


def test_sampling_respects_top_p_and_greedy():
    from jarvis.nlg.chat import sample
    logits = np.array([0, 0, 0, 0, 5.0, 4.0, -3.0, 1.0])
    rng = np.random.default_rng(0)
    assert sample(logits, rng, 0, 0.9, []) == 4
    picks = {sample(logits, rng, 1.0, 0.5, []) for _ in range(50)}
    assert picks == {4}                                               # 0.73 of the mass is token 4 alone
    logits[[0, 1]] = 100                                              # <pad>/<bos> can never be produced
    assert sample(logits, rng, 0, 0.9, []) == 4


# ---------------------------------------------------------------- training helpers (no PyTorch needed)
def test_token_shards_are_deterministic_and_shifted(tmp_path):
    from training.lm.data import TokenShards
    np.arange(1000, dtype=np.uint16).tofile(tmp_path / "a.bin")
    np.arange(2000, 2600, dtype=np.uint16).tofile(tmp_path / "b.bin")
    s = TokenShards([tmp_path / "a.bin", tmp_path / "b.bin"])
    x, y = s.batch(7, 4, 16)
    x2, _ = s.batch(7, 4, 16)
    assert x.shape == (4, 16) and np.array_equal(x, x2)               # same step -> same windows (resume-safe)
    assert np.array_equal(x[:, 1:], y[:, :-1]) and (y - x == 1).all()
    assert not np.array_equal(x, s.batch(8, 4, 16)[0])
    with pytest.raises(FileNotFoundError):
        TokenShards([])


def test_sft_batch_learns_only_replies():
    from training.lm.data import IGNORE, sft_batch
    x, y = sft_batch([([1, 10, 11, 12, 13, 2], [0, 0, 0, 1, 1, 1]), ([1, 20, 21], [0, 0, 1])])
    assert x.tolist() == [[1, 10, 11, 12, 13], [1, 20, 0, 0, 0]]
    assert y.tolist() == [[IGNORE, IGNORE, 12, 13, 2], [IGNORE, 21, IGNORE, IGNORE, IGNORE]]


def test_schedule_and_night_stop():
    from datetime import datetime
    from training.lm.data import eta, lr_at, stop_time
    assert lr_at(0, 1000, 1.0, 100) == pytest.approx(0.01)
    assert lr_at(99, 1000, 1.0, 100) == pytest.approx(1.0)
    assert lr_at(1000, 1000, 1.0, 100) == pytest.approx(0.1)
    assert lr_at(550, 1000, 1.0, 100) == pytest.approx(0.55)
    night = datetime(2026, 11, 27, 23, 0)
    assert datetime.fromtimestamp(stop_time("07:30", night)) == datetime(2026, 11, 28, 7, 30)
    assert datetime.fromtimestamp(stop_time("23:30", night)) == datetime(2026, 11, 27, 23, 30)
    assert stop_time("") is None and "night" in eta(3_000_000_000, 70_000)


def test_lm_state_file(tmp_path):
    from training.lm.data import State
    s = State(tmp_path / "lm_state.json")
    assert s.stage == "none"
    s.update(stage="pretraining", tokens=5)
    assert State(tmp_path / "lm_state.json").get("tokens") == 5


def test_presets():
    from training.lm.config import PRESETS
    assert 90e6 < PRESETS["base"].params() < 110e6 and 200e6 < PRESETS["large"].params() < 240e6
    for c in PRESETS.values():
        assert c.d_model % c.n_head == 0 and (c.d_model // c.n_head) % 2 == 0


def test_corpus_download_resumes(tmp_path):
    import io
    from training.lm import corpus
    blob = bytes(range(256)) * 40

    class Resp(io.BytesIO):
        def __init__(self, data, status, total):
            super().__init__(data)
            self.status, self.headers = status, {"Content-Length": str(len(data))}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            self.close()
    seen = []

    def opener(req, timeout=0):
        rng = req.get_header("Range")
        seen.append(rng)
        start = int(rng.split("=")[1].rstrip("-")) if rng else 0
        return Resp(blob[start:], 206 if rng else 200, len(blob))
    (tmp_path / "000.part").write_bytes(blob[:1000])                  # last night's half download
    got = corpus.download(1, raw=tmp_path, url="http://x/{i}", opener=opener)
    assert got == [tmp_path / "000.parquet"] and got[0].read_bytes() == blob and seen == ["bytes=1000-"]
    corpus.download(1, raw=tmp_path, url="http://x/{i}", opener=opener)
    assert len(seen) == 1                                             # already there: nothing fetched


# ---------------------------------------------------------------- PyTorch: trainer == runtime (RTX 5070 PC)
def _torch():
    return pytest.importorskip("torch")


def test_torch_model_matches_numpy_runtime():
    torch = _torch()
    from jarvis.nlg.chat import Decoder
    from training.lm.config import PRESETS
    from training.lm.model import JarvisLM, to_numpy_weights
    torch.manual_seed(0)
    cfg = PRESETS["tiny"]
    m = JarvisLM(cfg).eval()
    with torch.no_grad():
        for p in m.parameters():                                     # bigger weights: differences would show
            p.add_(torch.randn_like(p) * 0.05)
    ids = [1, 50, 999, 3, 12000, 7, 7, 300]
    with torch.no_grad():
        want = m(torch.tensor([ids])).numpy()[0]
    dec = Decoder(to_numpy_weights(m), cfg.d_model, cfg.n_layer, cfg.n_head, cfg.rope_theta)
    got, cache = dec.forward(ids[:5], all_logits=True)
    np.testing.assert_allclose(got, want[:5], rtol=2e-3, atol=2e-3)
    for i in range(5, len(ids)):
        step, cache = dec.forward([ids[i]], cache)
        np.testing.assert_allclose(step, want[i], rtol=2e-3, atol=2e-3)


def test_pretrain_smoke_and_resume(tmp_path, monkeypatch):
    _torch()
    from training.lm import pretrain
    monkeypatch.setattr(pretrain, "wait_if_paused", lambda: None)    # a game running mustn't stall the update check
    from training.lm.data import State
    tok = _tok()
    tok.save(tmp_path / "tok.json")
    data = tmp_path / "lm"
    data.mkdir()
    r = np.random.default_rng(0)
    pattern = r.integers(4, tok.vocab_size, 50)
    np.tile(pattern, 200).astype(np.uint16).tofile(data / "000.bin")  # a repeating text: easy to learn
    np.tile(pattern, 20).astype(np.uint16).tofile(data / "val.bin")
    common = ["--preset", "tiny", "--data", str(data), "--tokenizer", str(tmp_path / "tok.json"),
              "--out", str(tmp_path / "pre.pt"), "--micro", "4", "--accum", "1", "--ctx", "32", "--warmup", "5",
              "--eval-every", "10", "--tokens", str(4 * 32 * 60)]
    assert pretrain.main(common) == 0
    st = State(tmp_path / "lm_state.json")
    assert st.stage == "pretrained" and st.get("val_loss") < 5.0
    assert pretrain.main(common) == 0                                 # finished: resumes and has nothing to do


def test_finetune_smoke(tmp_path, monkeypatch):
    _torch()
    from datagen import generate_chat
    from training.lm import finetune
    monkeypatch.setattr(finetune, "wait_if_paused", lambda: None)
    from training.lm.config import PRESETS
    from training.lm.model import JarvisLM, save_checkpoint
    generate_chat.main(["--n", "400", "--out", str(tmp_path / "chat")])
    from training.lm.corpus import chat_texts
    tok = BPETokenizer.train(chat_texts(tmp_path / "chat" / "train.jsonl"), 600, verbose=False)
    tok.save(tmp_path / "lm_tokenizer.json")
    cfg = PRESETS["tiny"]
    save_checkpoint(tmp_path / "pre.pt", JarvisLM(cfg), None, step=0, tokens=0)
    code = finetune.main(["--base", str(tmp_path / "pre.pt"), "--data", str(tmp_path / "chat"),
                          "--tokenizer", str(tmp_path / "lm_tokenizer.json"), "--out", str(tmp_path / "lm_chat.pt"),
                          "--steps", "6", "--batch", "4", "--max-len", "96", "--mix-every", "0"])
    meta = json.loads((tmp_path / "lm_chat.json").read_text())
    assert code == 5 and meta["ready"] is False and "handoff" in meta["eval"]   # 6 steps: rightly below the bar


# ---------------------------------------------------------------- persona dialogues
def test_chat_data_is_clean_and_holds_phrasing_out(tmp_path):
    from datagen import generate_chat as G
    from jarvis.nlg.chatfmt import CMD, WEB, problems
    G.main(["--n", "3000", "--out", str(tmp_path)])
    rows = {s: [json.loads(line) for line in open(tmp_path / f"{s}.jsonl", encoding="utf-8")] for s in ("train", "val", "test")}
    kinds = {r["kind"] for r in rows["train"]}
    assert kinds == {"chat", "web", "cmd"}
    for split, rs in rows.items():
        for r in rs:
            assert r["turns"][-2][0] == "U" and r["turns"][-1][0] == "J" and len(r["head"].split("|")) == 4
            for who, text in r["turns"]:
                if who == "J" and text not in (WEB, CMD):
                    assert problems(text) == [], (text, problems(text))     # never teach what Jarvis would block
            name, addr, tone, _ = r["head"].split("|")
            if r["tag"] == "name":
                assert name in r["turns"][-1][1]
            if tone == "minimal" and addr:
                raise AssertionError("minimal tone has no form of address")
    users, _ = G.TOPICS["thanks"]
    test_only = set(G.held(users, "test")) - set(G.held(users, "train"))
    assert test_only and not test_only & set(G.held(users, "train"))
    train_users = {t.lower() for r in rows["train"] for w, t in r["turns"] if w == "U"}
    assert not ({t.lower() for t in test_only} & train_users)                # those exact sentences never trained on


def test_runtime_handoff_reply_shapes():
    """What the daemon gets back for each kind of model output."""
    from jarvis.nlg.chat import ChatModel

    class Fixed(ChatModel):
        def __init__(self, outputs):
            self.outputs, self.stats, self.max_ctx = list(outputs), {"replies": 0, "rejected": 0, "web": 0, "cmd": 0, "ms": 0}, 512
            self.tok = _tok()

            class D:
                def forward(self, ids, cache=None, all_logits=False):
                    return np.zeros(10), []
            self.dec = D()

        def generate(self, *a, **k):
            return self.outputs.pop(0)
    assert Fixed(["[web] who invented"]).reply("who invented the radio") == ("web", "")
    assert Fixed(["Good evening, sir."]).reply("hi") == ("say", "Good evening, sir.")
    assert Fixed(["I've deleted them.", "Hello there!"]).reply("hi") == ("say", "Hello there!")   # 1st filtered out
    assert Fixed(["Done.", "I've opened it.", "It's 5 pm."]).reply("hi") is None
    assert Fixed(["Hi!", "[cmd]"]).reply("x") == ("say", "Hi!")


# ---------------------------------------------------------------- inside Jarvis
class _StubChat:
    def __init__(self, outs):
        self.outs, self.calls = list(outs), []

    def __bool__(self):
        return True

    def reply(self, text, history, **persona):
        self.calls.append((text, list(history), persona))
        out = self.outs.pop(0)
        if isinstance(out, Exception):
            raise out
        return out


def _jarvis(tmp_path):
    from jarvis.config import Config
    from jarvis.daemon import Jarvis
    cfg = Config(data_dir=tmp_path, encrypt_logs=False)
    cfg.show_details = False
    return Jarvis(cfg, confirm=lambda *a: True, data_root=tmp_path / "data")


def test_small_talk_goes_to_the_chat_model(tmp_path, monkeypatch):
    import webbrowser
    opened = []
    monkeypatch.setattr(webbrowser, "open", lambda u, *a, **k: opened.append(u) or True)
    j = _jarvis(tmp_path)
    j.chat = _StubChat([("say", "Splendid news, sir! Congratulations."), ("say", "Very kind of you, sir."),
                        ("web", ""), ("cmd", ""), RuntimeError("boom"), None])
    assert j.handle("I aced the interview") == "Splendid news, sir! Congratulations."
    assert j.handle("you're the best") == "Very kind of you, sir."
    text, history, persona = j.chat.calls[1]
    assert history == [("U", "I aced the interview"), ("J", "Splendid news, sir! Congratulations.")]
    assert persona == {"name": "Jarvis", "address": "sir", "tone": "butler"}
    assert "search" in j.handle("who invented the telephone")
    j.handle("yes")
    assert opened and "telephone" in opened[0]                     # the yes ran the search it offered
    assert j.handle("blorp the flibber") != ""                     # [cmd]: the usual "didn't get that"
    assert j.handle("thanks") != ""                                # model crashed: fixed small-talk line instead
    assert j.handle("thank you") != ""                             # nothing acceptable: same
    calls = len(j.chat.calls)
    assert "It's" in j.handle("what time is it") and len(j.chat.calls) == calls   # the clock never asks the model


def test_chat_model_can_be_switched_off(tmp_path):
    j = _jarvis(tmp_path)
    j.chat = _StubChat([("say", "from the model")])
    j.config.chat_model = False
    assert j.handle("I aced the interview") != "from the model" and not j.chat.calls
    for tone in ("butler", "friendly", "minimal"):
        j.persona.tone = tone
        assert "search" in j.persona.say("chat_web").lower()
