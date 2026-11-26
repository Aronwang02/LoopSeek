"""Settings of the current framework (configs/loopseek.yaml for inference; the options it leaves off are tested too):
  share_cap_per_place      option (a): a segment of a multi-segment row is covered at min(share, 4) kept frames
  verifier_overview_question = places / places_line   "one place or several places?" replaces the overview Q3; its answer
                           is a persistent controller line
  pick_answer_k = 24       item 20: the keep head scores every overview frame; the 24 highest are the answer set's overview
  verifier_keep_where_joint item 18: per-frame keep + where in one decision; where refers to the state after a correct keep
                           (the kept frames before it + its picks that show evidence)
  verifier_q3 = false      no Q3 anywhere; the inspection line keeps "kept m of k frames"
  skip_final_q3            no Q3 (nor label) after the check that fills the kept-frame budget
  verifier_line_compact    the verifier line without the totals / actions left
  where_wording = v2       "of that range" and the corrected enclosing question
and their SFT rows (scripts/gen_role_sft_from_trajectories.py) and dev metrics (scripts/train_sft.py)."""
import json
import re
import subprocess
import sys

from conftest import ROOT, StubEncoder, StubReader
from src.baselines.evidence import uncovered_spans, merge_gap, segment_gap
from src.baselines.teacher import EvidenceExpertController
from src.common.config import load_config
from src.env.episode_v3 import tier_position
from src.rollout.runner_v3 import RunnerV3
from test_long_flow import SAMPLE, FirstValid, _res, _text

PY = sys.executable


class ScriptedVerifier:
    """Answers every verifier question of the formal flow: places (P(several) = places_p), per-frame keep groups (P(yes)
    = keep_fn(ep, t)) on overview or selected frames, where / enclosing groups, Q3 tiers (P(T3) = 0.6). Candidate
    probabilities are normalized within each group."""
    name = "scripted_verifier"

    def __init__(self, keep_fn, places_p=0.8):
        self.keep_fn, self.places_p = keep_fn, places_p
        self.inputs = []

    def decide(self, inputs, episodes, generators):
        out = []
        for inp, ep in zip(inputs, episodes):
            self.inputs.append(inp)
            pls = inp.payloads
            probs = [0.0] * len(pls)
            n_enc = sum("enclosing" in pl for pl in pls)
            for j, pl in enumerate(pls):
                if "places" in pl:
                    probs[j] = self.places_p if pl["places"] == "several" else 1 - self.places_p
                elif "keep" in pl:
                    p = float(self.keep_fn(ep, pl["time"]))
                    probs[j] = p if pl["keep"] else 1 - p
                elif "where" in pl:
                    probs[j] = {"before": 0.1, "inside_small": 0.2, "inside_large": 0.6, "after": 0.1}[pl["where"]]
                elif "enclosing" in pl:
                    probs[j] = 1.0 / n_enc
            if any("tier" in pl for pl in pls):
                probs[tier_position(inp, 3)], probs[tier_position(inp, 1)] = 0.6, 0.4
            out.append(_res(max(range(len(pls)), key=lambda j: probs[j]), len(pls), probs))
        return out


def cfg_formal(**budget):
    c = load_config([ROOT / "configs" / "default.yaml"])
    c["budget"]["min_window_sec"] = 0.2           # the synthetic video is 6.1 s long
    c["budget"].update(answer_uniform=24, kept_frames=8, frames_per_inspect=4, record_verifier_inputs=True,
                       label_pE=True, verifier_q3_overview="answer", verifier_q3_where_joint=False, verifier_q3=True)
    c["budget"].update(budget)
    return c


def run(asset, features, spans, ver=None, n_overview=32, **budget):
    cfg = cfg_formal(**budget)
    cfg["video"]["global_frames"] = n_overview
    ann = {SAMPLE.sample_id: {"key_spans": spans, "evidence_multi_segment": len(spans) > 1}}
    in_ev = lambda ep, t: 0.9 if any(a <= t <= b for a, b in spans) else 0.1     # = the expert's evidence (pad 0)
    ver = ver or ScriptedVerifier(in_ev)
    dec = {"controller": EvidenceExpertController(ann, pad=0.0), "localize": FirstValid(), "frame": FirstValid(),
           "verifier": ver}
    runner = RunnerV3(cfg, dec, StubReader(), features, StubEncoder(), text_len=None, annotations=ann)
    t = runner.run([{"sample": SAMPLE, "asset": asset, "group_id": "g", "trajectory_id": "t0", "seed": 0}])[0]
    return t, cfg, ver


def mem(d):
    return "".join(x.get("t", "") for x in d["input"]["prefix"] if isinstance(x, dict))


# ---------------------------------------------------------------- option (a)
def test_share_cap_per_place_moves_on_after_one_inspection():
    segs = [(0.0, 60.0), (100.0, 120.0)]                    # shares of 8: 6 and 2
    kept4 = [10.0, 20.0, 30.0, 40.0]
    assert uncovered_spans(segs, kept4, 8) == segs          # without the cap segment 1 still needs 6
    assert uncovered_spans(segs, kept4, 8, per_place=4) == [(100.0, 120.0)]
    one = [(0.0, 60.0)]                                     # a single segment is never left
    assert uncovered_spans(one, kept4, 8, per_place=4) == one
    assert merge_gap([(0, 10), (35, 40)], segment_gap(400)) == [(0, 40)]       # min(30, 50) = 30 s
    assert merge_gap([(0, 10), (35, 40)], segment_gap(160)) == [(0, 10), (35, 40)]   # min(30, 20) = 20 s


def test_expert_leaves_after_four_frames_in_a_multi_segment_row(asset, features):
    spans = [[2.0, 2.6], [5.0, 5.3]]               # the dense part of the synthetic video (a frame every 50 ms)
    t, _, _ = run(asset, features, spans, share_cap_per_place=4)
    assert t["status"] == "ok", t.get("error")
    ck = [x for x in t["turns"] if x["action"] == "TENTATIVE_CHECK"]
    first = next(c for c in ck if c["kept_total"] >= 4)
    nxt = next(x for x in t["turns"][t["turns"].index(first) + 1:] if "controller_decision" in x)
    assert nxt["action"] in ("SWITCH_LOCAL", "UP"), nxt          # the expert moves to the other segment
    assert all(x.get("share_cap_per_place") == 4 for x in t["turns"] if x["action"] == "WHERE_CHECK")


# ---------------------------------------------------------------- places + pick-24
def test_overview_asks_places_and_picks_the_24_highest_frames(asset, features):
    T = asset.duration
    spans = [[T * 0.30, T * 0.40]]
    t, cfg, ver = run(asset, features, spans, verifier_overview_question="places", places_line=True, pick_answer_k=24)
    assert t["status"] == "ok", t.get("error")
    ov = [x for x in t["turns"] if x["action"] == "OVERVIEW_CHECK"]
    assert len(ov) == 1 and set(ov[0]["places"]) == {"one", "several"} and abs(ov[0]["places"]["several"] - 0.8) < 1e-9
    fr = ov[0]["frames"]
    n_ov = len(t["summary"]["overview_times"])                 # 32 requested; the sparse start of the video dedups some
    assert n_ov > 24 and len(fr) == n_ov and len(ov[0]["picked_times"]) == 24
    best = sorted(fr, key=lambda f: (-f["p_yes"], f["time"]))[:24]
    assert sorted(f["time"] for f in best) == ov[0]["picked_times"]
    assert t["summary"]["answer_overview_times"] == ov[0]["picked_times"]      # the answer set's overview frames
    assert set(ov[0]["picked_times"]) <= set(t["summary"]["answer_times"])
    vin = [i for i in ver.inputs if any("places" in pl for pl in i.payloads)]
    assert len(vin) == 1 and all("tier" not in pl for pl in vin[0].payloads)  # no Q3 at the overview
    ctrl = [d for d in t["decisions"] if d["decision_type"] == "controller"]
    line = "Verifier after the overview: evidence at one place 0.20, at several separate places 0.80"
    assert ctrl and all(line in mem(d) for d in ctrl)                          # persistent controller line
    q3 = [d for d in t["decisions"] if d["decision_type"] == "verifier" and any("tier" in p for p in d["input"]["payloads"])]
    assert q3                                                                  # Q3 (item 5) sees the picked frames
    g = [s for s in q3[0]["input"]["prefix"] if isinstance(s, dict) and "v" in s]
    picked_pts = {round(x, 6) for x in ov[0]["picked_times"]}
    assert len(picked_pts) == 24 and len([1 for s in g if round(s["v"][2], 6) in picked_pts]) >= 24


def test_exactly_k_overview_frames_are_all_taken_without_scoring(asset, features):
    T = asset.duration
    t0, _, _ = run(asset, features, [[T * 0.3, T * 0.4]], verifier_overview_question="places")
    k = len(t0["summary"]["overview_times"])                   # the overview's distinct frames on this video
    t, _, ver = run(asset, features, [[T * 0.3, T * 0.4]], verifier_overview_question="places", pick_answer_k=k)
    ov = next(x for x in t["turns"] if x["action"] == "OVERVIEW_CHECK")
    assert ov["frames"] is None and len(ov["picked_times"]) == k and ov["places"] is not None
    assert ov["picked_times"] == t["summary"]["overview_times"]
    t2, _, _ = run(asset, features, [[T * 0.3, T * 0.4]], verifier_overview_question="places", pick_answer_k=k + 1)
    ov2 = next(x for x in t2["turns"] if x["action"] == "OVERVIEW_CHECK")      # fewer than k: the uniform answer grid
    assert ov2["frames"] is None and ov2["picked_times"] is None


def test_without_pick_the_answer_grid_stays_uniform(asset, features):
    T = asset.duration
    t, _, _ = run(asset, features, [[T * 0.3, T * 0.4]], verifier_overview_question="places")
    ov = next(x for x in t["turns"] if x["action"] == "OVERVIEW_CHECK")
    assert ov["frames"] is None and ov["picked_times"] is None
    uni = sorted({asset.time(i) for i in asset.uniform_indices(0.0, T, 24)})
    assert t["summary"]["answer_overview_times"] == uni


# ---------------------------------------------------------------- item 18, skip final Q3, compact line, wording
def test_joint_keep_where_one_forward_records_the_keep_state(asset, features):
    T = asset.duration
    spans = [[T * 0.30, T * 0.40]]
    t, _, ver = run(asset, features, spans, verifier_keep_where_joint=True, skip_final_q3=True, verifier_line_compact=True,
                    where_wording="v2")
    assert t["status"] == "ok", t.get("error")
    joint = [i for i in ver.inputs if any("keep" in p for p in i.payloads) and any("where" in p for p in i.payloads)]
    assert joint                                                     # keep + where in one decision below the root
    turns = t["turns"]
    for j, x in enumerate(turns):
        if x["action"] == "TENTATIVE_CHECK" and "frames_input" in x and any("where" in p for p in x["frames_input"]["payloads"]):
            w = turns[j + 1]
            assert w["action"] == "WHERE_CHECK"                      # applied right after the keep, from that decision
            kept_after = {f["time"] for c in turns[:j + 1] if c["action"] == "TENTATIVE_CHECK" for f in c["frames"] if f["kept"]}
            kept_now = {f["time"] for f in x["frames"] if f["kept"]}
            assert set(w["kept_times"]) == kept_after - kept_now     # the kept frames before this keep
            assert w["picked_times"] == [f["time"] for f in x["frames"]]   # + its picks (the labels keep those in evidence)
    ck = [x for x in turns if x["action"] == "TENTATIVE_CHECK"]
    last = ck[-1]
    if last["kept_total"] >= 8:                                      # skip_final_q3: no Q3 after the budget is full
        assert "probs" not in last and "letter_probs" not in last
    assert any("probs" in c for c in ck[:-1]) or len(ck) == 1
    ctrl = [mem(d) for d in t["decisions"] if d["decision_type"] == "controller"]
    lines = [ln for m in ctrl for ln in m.split("\n") if ln.startswith("- Verifier after inspection attempt")]
    assert lines and all("kept in total" not in ln and "actions left" not in ln for ln in lines)
    wtxt = [_text(i) + " ".join(c[0] for c in i.candidates) for i in ver.inputs if any("where" in p for p in i.payloads)]
    assert all("of that range" in s for s in wtxt)
    enc = [s for s in wtxt if "Q0:" in s]
    assert all("searched intervals not above it" in s for s in enc)


def test_no_q3_when_switched_off(asset, features):
    T = asset.duration
    t, _, ver = run(asset, features, [[T * 0.30, T * 0.40]], verifier_q3=False, verifier_keep_where_joint=True,
                    verifier_overview_question="places", pick_answer_k=24, verifier_line_compact=True)
    assert t["status"] == "ok", t.get("error")
    assert not any(any("tier" in p for p in i.payloads) for i in ver.inputs)          # no Q3 question asked
    assert not any("letter_probs" in x or x["action"] == "VERIFIER_POINT" for x in t["turns"])   # nor labelled
    ck = [x for x in t["turns"] if x["action"] == "TENTATIVE_CHECK"]
    assert ck and all("probs" not in c for c in ck)
    ctrl = [mem(d) for d in t["decisions"] if d["decision_type"] == "controller"]
    lines = [ln for m in ctrl for ln in m.split("\n") if ln.startswith("- Verifier after inspection attempt")]
    assert lines and all(re.fullmatch(r"- Verifier after inspection attempt \d+ \(window [^)]*\): kept \d+ of \d+ frames",
                                      ln) for ln in lines), lines


def test_defaults_keep_the_old_texts(asset, features):
    T = asset.duration
    t, _, ver = run(asset, features, [[T * 0.30, T * 0.40]])
    wtxt = [_text(i) + " ".join(c[0] for c in i.candidates) for i in ver.inputs if any("where" in p for p in i.payloads)]
    assert wtxt and all("of that range" not in s for s in wtxt)
    assert not any(x["action"] == "OVERVIEW_CHECK" for x in t["turns"])


# ---------------------------------------------------------------- SFT rows and dev metrics
def test_skip_final_q3_only_after_the_budget_fills(asset, features):
    spans = [[1.6, 6.0]]                                     # most of the dense part: 8 frames are kept quickly
    for flag in (True, False):
        t, _, _ = run(asset, features, spans, skip_final_q3=flag)
        assert t["status"] == "ok", t.get("error")
        ck = [x for x in t["turns"] if x["action"] == "TENTATIVE_CHECK"]
        assert ck[-1]["kept_total"] == 8 and t["summary"]["termination_reason"] == "frame_budget_exhausted"
        assert ("probs" in ck[-1]) is (not flag)             # the check that fills the budget: Q3 only without the flag
        assert all("probs" in c for c in ck[:-1])            # every earlier check keeps its Q3


def test_pick_record_only_keeps_the_uniform_grid(asset, features):
    T = asset.duration
    t, _, _ = run(asset, features, [[T * 0.3, T * 0.4]], verifier_overview_question="places", pick_answer_k=24,
                  pick_answer_apply=False)
    ov = next(x for x in t["turns"] if x["action"] == "OVERVIEW_CHECK")
    assert ov["frames"] and len(ov["would_pick_times"]) == 24 and ov["picked_times"] is None and "input" in ov
    uni = sorted({asset.time(i) for i in asset.uniform_indices(0.0, T, 24)})
    assert t["summary"]["answer_overview_times"] == uni


def test_policy_set_overrides_travel_with_the_set(tmp_path):
    import json as _j
    from src.policy.policy_set import policy_set_overrides, read_policy_set
    p = tmp_path / "ps.json"
    p.write_text(_j.dumps({"controller": "a", "localize": "b", "frame": "b", "verifier": "c",
                           "config_overrides": ["budget.places_line=true"]}))
    assert read_policy_set(str(p)) == {"controller": "a", "localize": "b", "frame": "b", "verifier": "c"}
    assert policy_set_overrides(str(p)) == ["budget.places_line=true"]
    cfg = load_config([ROOT / "configs" / "default.yaml"], policy_set_overrides(str(p)) + ["budget.places_line=false"])
    assert cfg["budget"]["places_line"] is False                    # --set comes after the set's overrides


def test_paper_inference_config_runs_end_to_end(asset, features):
    """configs/loopseek.yaml over the default: one episode with every inference setting of the paper on (scripted deciders)."""
    cfg = load_config([ROOT / "configs" / "default.yaml", ROOT / "configs" / "loopseek.yaml"])
    cfg["budget"]["min_window_sec"] = 0.2
    cfg["budget"]["record_verifier_inputs"] = True
    spans = [[2.0, 2.6], [5.0, 5.3]]
    ann = {SAMPLE.sample_id: {"key_spans": spans, "evidence_multi_segment": True}}
    ver = ScriptedVerifier(lambda ep, t: 0.9 if any(a <= t <= b for a, b in spans) else 0.1)
    dec = {"controller": EvidenceExpertController(ann, pad=0.0), "localize": FirstValid(), "frame": FirstValid(),
           "verifier": ver}
    runner = RunnerV3(cfg, dec, StubReader(), features, StubEncoder(), text_len=None, annotations=ann)
    t = runner.run([{"sample": SAMPLE, "asset": asset, "group_id": "g", "trajectory_id": "t0", "seed": 0}])[0]
    assert t["status"] == "ok", t.get("error")
    acts = [x["action"] for x in t["turns"]]
    assert acts[0] == "OVERVIEW_CHECK" and "TENTATIVE_CHECK" in acts
    assert len(t["summary"]["answer_overview_times"]) == 24 and len(t["summary"]["answer_times"]) <= 32
    assert any(any("where" in p for p in x.get("frames_input", {}).get("payloads", [])) for x in t["turns"])
    assert all("places_line" not in d for d in t["decisions"])
    ov = next(x for x in t["turns"] if x["action"] == "OVERVIEW_CHECK")  # the overview point scores frames
    assert ov["places"] is None and ov["frames"] and len(ov["picked_times"]) == 24   # for pick-24 only, no places question
    assert not any(any("places" in p for p in i.payloads) for i in ver.inputs)
    assert cfg["budget"]["share_cap_per_place"] is None
    assert not any(any("tier" in p for p in i.payloads) for i in ver.inputs)   # no Q3
    assert not any("letter_probs" in x for x in t["turns"])
    ctrl = [mem(d) for d in t["decisions"] if d["decision_type"] == "controller"]
    assert not any("one place" in m for m in ctrl)
