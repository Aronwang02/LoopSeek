"""Answer rule (2026-09-29): parsed answers are kept; an answer without a legal letter falls back to the option
letter with the highest next-token probability (reader.answer_fallback=letter_logits); 'none' keeps it unparsed."""
from src.reader.qwen_reader import QwenReader


def stub_reader(fallback, generated):
    r = QwenReader.__new__(QwenReader)
    r.cfg = {"batch_size": 8, "answer_max_new_tokens": 8, "observation_max_new_tokens": 8, "answer_fallback": fallback}
    r.calls = []

    def _generate(msgs, n, letters_batch=None):
        meta = [{"batch_seconds": 0.0, "batch_size": len(msgs)} for _ in msgs]
        for m, ls in zip(meta, letters_batch or []):
            m['letter_probs'] = {L: (0.7 if L == 'C' else 0.3 / (len(ls) - 1)) for L in ls}
        return list(generated[:len(msgs)]), meta

    def letter_probs(msgs, letters_batch):
        r.calls.append(len(msgs))
        return [{L: (0.7 if L == "C" else 0.3 / (len(ls) - 1)) for L in ls} for ls in letters_batch]
    r._generate, r.letter_probs = _generate, letter_probs
    return r


def job():
    return {"kind": "answer", "frames": [], "times": [], "question": "q?", "options": {"A": "a", "B": "b", "C": "c"},
            "duration": 10.0, "source": "", "interval": None, "notes": [("frames 1.0s", "Insufficient evidence.")]}


def test_fallback_only_for_unparsed():
    r = stub_reader("letter_logits", ["B", "None of the options are supported by the frames."])
    out = r.run([job(), job()])
    assert out[0]["parsed"] == "B" and out[0]["meta"]["answer_source"] == "generated"
    assert out[1]["parsed"] == "C" and out[1]["meta"]["answer_source"] == "letter_logits_fallback"
    assert abs(sum(out[1]["letter_probs"].values()) - 1.0) < 1e-9 and r.calls == []


def test_no_fallback_keeps_old_rule():
    r = stub_reader("none", ["None of the above"])
    out = r.run([job()])
    assert out[0]["parsed"] is None and out[0]["meta"]["answer_source"] == "unparsed" and r.calls == []


def test_obs_prompt_follows_reader_group():
    from src.common.config import load_config
    from conftest import ROOT
    cfg = load_config([ROOT / "configs/default.yaml"])
    picked = {}
    for key in ("reader", "reader_qwen3vl8b"):
        r = QwenReader.__new__(QwenReader)
        r.cfg, r.model_key = cfg["reader"], key
        picked[key] = r.obs_prompt()
    assert picked == {"reader": "selfassess3", "reader_qwen3vl8b": "v1"}
    r.cfg = {"obs_prompt": "v1"}                       # plain names (default.yaml, older runs) still work
    assert r.obs_prompt() == "v1"
