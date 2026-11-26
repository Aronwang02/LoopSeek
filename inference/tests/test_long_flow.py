"""Long-video flow on the repo config (configs/default.yaml) with scripted deciders on the synthetic video:
tentative inspections (keep iff P(T3) >= 0.5, withdrawn frames never offered again, attempt cap, back-fill), the controller
state lines (full verifier distribution, locator probe) that the verifier never sees, neutral shuffled verifier options,
re-localization that excludes searched windows, the batch-size-free INSPECT text, verifier labelling, the evidence expert
and the process metrics."""
import torch

from conftest import ROOT, LabelReader, StubEncoder, StubReader
from src.common.config import load_config
from src.env.environment import PublicSample
from src.env.episode_v3 import tier_position
from src.rollout.runner_v3 import RunnerV3

SAMPLE = PublicSample("s1", "v1", "unused", "What color?", {"A": "red", "B": "green", "C": "blue", "D": "gray"})


# the flat flow (before the window tree, 2026-10-06): LOCALIZE_GLOBAL, navigation / attempt budgets, locator probe line
FLAT = dict(window_tree=False, actions=["LOCALIZE_GLOBAL", "REFINE_LOCAL", "INSPECT_FRAMES"], nav_moves=8,
            max_attempts=8, max_turns=16, locator_line=True)


def cfg_long(**budget):
    c = load_config([ROOT / "configs" / "default.yaml"])
    c["budget"]["min_window_sec"] = 0.2          # the synthetic video is 6.1 s long
    c["budget"]["verifier_q3"] = True            # most tests cover the flow with Q3; the paper's settings switch it off
    c["budget"].update(budget)
    return c


def _res(index, n, probs=None, info=None):
    probs = probs or [float(j == index) for j in range(n)]
    return {"index": index, "log_prob": 0.0, "probs": probs, "entropy": 0.0, "forced": False, "seconds": 0.0,
            "info": info or {}}


def _text(inp):
    return "".join(s for s in inp.prefix if isinstance(s, str))


class TierVerifier:
    """Scripted verifier. Q3 (4 tiers, options maybe shuffled): P(T3) from a fixed sequence per episode (overview first,
    then one per check). Per-frame question (payload "keep"): P(keep) = keep_fn(episode, frame time)."""
    name = "tier_verifier"

    def __init__(self, p3s, keep_fn=None, where_fn=None):
        self.p3s, self.n, self.texts, self.orders = list(p3s), {}, [], []
        self.keep_fn = keep_fn or (lambda ep, t: 0.9)
        self.where_fn = where_fn or (lambda ep: {"before": 0.1, "inside_small": 0.2, "inside_large": 0.6, "after": 0.1})
        self.where_calls = 0

    def decide(self, inputs, episodes, generators):
        out = []
        for inp, ep in zip(inputs, episodes):
            self.texts.append(_text(inp) + "\n".join(c[0] for c in inp.candidates))
            pls = inp.payloads
            if "keep" in pls[0] and "group" in pls[0]:   # joint per-frame question: one (yes, no) pair per frame
                probs = []
                for pl in pls:
                    p = float(self.keep_fn(ep, pl["time"]))
                    probs.append(p if pl["keep"] else 1 - p)
                out.append(_res(0, len(probs), probs))
                continue
            if "keep" in pls[0]:
                p = float(self.keep_fn(ep, pls[0]["time"]))
                probs = [p if pl["keep"] else 1 - p for pl in pls]
                out.append(_res(0 if p >= 0.5 else 1, 2, probs))
                continue
            probs = [0.0] * len(pls)
            has_where = any("where" in pl for pl in pls)
            if has_where:                                # "where is the evidence" groups (alone or joint with Q3)
                self.where_calls += 1
                d = self.where_fn(ep)
                n_enc = sum("enclosing" in pl for pl in pls)
                for j, pl in enumerate(pls):
                    if "where" in pl:
                        probs[j] = d[pl["where"]]
                    elif "enclosing" in pl:
                        probs[j] = 1.0 / n_enc
            if any("tier" in pl for pl in pls):          # Q3: advances the per-episode sequence
                k = self.n.get(id(ep), 0)
                self.n[id(ep)] = k + 1
                p3 = self.p3s[min(k, len(self.p3s) - 1)]
                probs[tier_position(inp, 3)], probs[tier_position(inp, 1)] = p3, 1 - p3
                self.orders.append([pl["tier"] for pl in pls if "tier" in pl])
                out.append(_res(tier_position(inp, 3), len(pls), probs))
            else:
                out.append(_res(max(range(len(pls)), key=lambda j: probs[j]), len(pls), probs))
        return out


class InspectController:
    """INSPECT_FRAMES when legal, else the first legal action; records its inputs."""
    name = "inspect_controller"

    def __init__(self):
        self.texts, self.cands = [], []

    def decide(self, inputs, episodes, generators):
        out = []
        for inp in inputs:
            self.texts.append(_text(inp))
            self.cands.append([c[0] for c in inp.candidates])
            legal = [inp.candidate_ids[i] for i in inp.legal_indices()]
            a = "INSPECT_FRAMES" if "INSPECT_FRAMES" in legal else (legal[0] if legal else None)
            out.append({"index": None} if a is None else _res(inp.candidate_ids.index(a), len(inp.candidates)))
        return out


class AlwaysLocalize(InspectController):
    def decide(self, inputs, episodes, generators):
        out = []
        for inp in inputs:
            self.texts.append(_text(inp))
            legal = [inp.candidate_ids[i] for i in inp.legal_indices()]
            a = "LOCALIZE_GLOBAL" if "LOCALIZE_GLOBAL" in legal else (legal[0] if legal else None)
            out.append({"index": None} if a is None else _res(inp.candidate_ids.index(a), len(inp.candidates)))
        return out


class RandomController:
    name = "random_controller"

    def decide(self, inputs, episodes, generators):
        out = []
        for inp, g in zip(inputs, generators):
            legal = inp.legal_indices()
            out.append({"index": None} if not legal else
                       _res(legal[int(torch.randint(len(legal), (1,), generator=g))], len(inp.candidates)))
        return out


class FirstValid:
    name = "first_valid"

    def decide(self, inputs, episodes, generators):
        out = []
        for inp in inputs:
            ok = [j for j, v in enumerate(inp.valid) if v]
            picks = ok[:getattr(inp, "select_k", None) or 1]
            out.append(_res(picks[0], len(inp.candidates), info={"picks": picks}) if picks else {"index": None})
        return out


def run(cfg, asset, features, ctrl, ver, loc=None, reader=None, seed=0):
    dec = {"controller": ctrl, "localize": loc or FirstValid(), "frame": FirstValid(), "verifier": ver}
    runner = RunnerV3(cfg, dec, reader or StubReader(), features, StubEncoder(), text_len=None)
    return runner.run([{"sample": SAMPLE, "asset": asset, "group_id": "g", "trajectory_id": "t0", "seed": seed}])[0]


def checks(t):
    return [x for x in t["turns"] if x["action"] == "TENTATIVE_CHECK"]


SEQ = [0.1, 0.2, 0.6, 0.3, 0.2, 0.1, 0.4]


def test_batch_mode_keep_rule_still_works(asset, features):
    t = run(cfg_long(verifier_per_frame=False), asset, features, InspectController(), TierVerifier(SEQ))
    ck = checks(t)
    assert ck and [c["kept"] for c in ck] == [SEQ[min(1 + j, len(SEQ) - 1)] >= 0.5 for j in range(len(ck))]
    assert len(t["summary"]["answer_times"]) == 32


def test_per_frame_keep_rejected_frames_and_32_frame_answer(asset, features):
    half = lambda ep, t: 0.9 if t < ep.T / 2 else 0.1
    t = run(cfg_long(), asset, features, InspectController(), TierVerifier(SEQ, half))
    assert t["status"] == "ok"
    ck = checks(t)
    assert ck
    for c in ck:
        for f in c["frames"]:                                    # keep iff P(keep) >= 0.5 (budget not binding here)
            assert f["kept"] == (f["p_keep"] >= 0.5)
        assert c["n_kept"] == sum(f["kept"] for f in c["frames"])
    picked = [f for x in t["turns"] if x.get("frame_times") for f in x["frame_times"]]
    assert len(picked) == len(set(picked))                        # rejected frames never come back
    kept = [f["time"] for c in ck for f in c["frames"] if f["kept"]]
    assert all(k in t["summary"]["answer_times"] for k in kept)
    assert len(t["summary"]["answer_times"]) == 32 and t["accounting"]["answerer_calls"] == 1


def test_per_frame_budget_ends_the_episode_when_16_frames_are_kept(asset, features):
    t = run(cfg_long(), asset, features, InspectController(), TierVerifier(SEQ, lambda ep, t: 0.9))
    ck = checks(t)
    assert ck[-1]["kept_total"] == 16 and t["summary"]["termination_reason"] == "frame_budget_exhausted"
    assert len(ck) == 2                                           # 2 batches x 8 kept


def test_per_frame_nothing_kept_backfills_best_frames(asset, features):
    score = lambda ep, t: 0.4 * t / ep.T                           # all below 0.5; later frames score higher
    t = run(cfg_long(), asset, features, InspectController(), TierVerifier(SEQ, score))
    ck = checks(t)
    assert all(c["n_kept"] == 0 for c in ck)
    rest = [x for x in t["turns"] if x["action"] == "TENTATIVE_RESTORE"]
    assert len(rest) == 1
    pool = sorted((f["p_keep"] for c in ck for f in c["frames"]), reverse=True)
    assert rest[0]["p_keep"] == pool[:len(rest[0]["p_keep"])] and len(rest[0]["p_keep"]) <= 16


def test_controller_sees_verifier_and_locator_lines_the_verifier_does_not(asset, features):
    ctrl, ver = InspectController(), TierVerifier(SEQ)
    run(cfg_long(**FLAT), asset, features, ctrl, ver)
    first = ctrl.texts[0]
    assert "Verifier after the overview: answerer's probability for the correct option: <25% 0.00, 25%-50% 0.90" in first
    assert "Locator, whole video: W1 " in first
    assert any("Verifier after inspection attempt 1" in x and "kept 8 of 8 frames, 8 of 16 kept in total" in x
               for x in ctrl.texts[1:])
    assert "- Kept frames: 0 of 16; inspection attempts left 8 of 8" in first
    assert not any("Verifier after" in x or "Locator," in x for x in ver.texts)
    assert any("picked by the last inspection: is it in or near the part of the video" in x and "E1+: E1 [" in x
               for x in ver.texts)                                              # the joint per-frame question
    v1 = TierVerifier(SEQ)
    run(cfg_long(verifier_frame_joint=False), asset, features, InspectController(), v1)
    assert any("Is selected frame E" in x and "in or near the part of the video" in x for x in v1.texts)  # old mode


def test_neutral_shuffled_verifier_options_keep_the_same_decisions(asset, features):
    v_fixed, v_neutral = TierVerifier(SEQ), TierVerifier(SEQ)
    a = run(cfg_long(verifier_per_frame=False, verifier_neutral=False), asset, features, InspectController(), v_fixed)
    b = run(cfg_long(verifier_per_frame=False), asset, features, InspectController(), v_neutral)
    assert [c["kept"] for c in checks(a)] == [c["kept"] for c in checks(b)]
    assert [round(c["p_tier3"], 6) for c in checks(a)] == [round(c["p_tier3"], 6) for c in checks(b)]
    assert len({tuple(o) for o in v_neutral.orders}) > 1 and {tuple(o) for o in v_fixed.orders} == {(0, 1, 2, 3)}
    for txt in [x for x in v_neutral.texts if "Where is the evidence" not in x]:
        assert "V1:" in txt and "wrong" not in txt and "enough evidence" not in txt and "weak" not in txt


def test_locator_probe_records_no_decision_and_keeps_the_random_stream(asset, features):
    a = run(cfg_long(**dict(FLAT, locator_line=False)), asset, features, RandomController(), TierVerifier(SEQ), seed=3)
    b = run(cfg_long(**FLAT), asset, features, RandomController(), TierVerifier(SEQ), seed=3)
    assert [x["action"] for x in a["turns"]] == [x["action"] for x in b["turns"]]
    n_loc = sum(x["action"] in ("LOCALIZE_GLOBAL", "REFINE_LOCAL", "SWITCH_LOCAL") for x in b["turns"])
    assert sum(d["decision_type"] == "localize" for d in b["decisions"]) == n_loc
    assert b["locator_log"] and not a["locator_log"]


def test_relocalization_excludes_searched_windows(asset, features):
    for excl in (False, True):
        t = run(cfg_long(relocalize_exclude_searched=excl, **FLAT), asset, features, AlwaysLocalize(), TierVerifier([0.1]))
        iv = [(x["interval"]["start"], x["interval"]["end"]) for x in t["turns"] if x.get("interval")]
        if excl:     # never a searched window again while new ones remain (5 on this video; then all are offered)
            assert len(set(iv[:5])) == 5, iv
        else:        # the first-valid locator returns to the same window
            assert len(set(iv)) < len(iv), iv


def test_inspect_text_has_no_batch_size(asset, features):
    for free, want in ((True, "pick a batch of frames in the active interval"), (False, "pick 8 frames in the active interval")):
        ctrl = InspectController()
        run(cfg_long(inspect_text_budget_free=free), asset, features, ctrl, TierVerifier(SEQ))
        assert any(want in c for c in ctrl.cands[0]), ctrl.cands[0]


def test_label_pE_at_every_check_with_tier_positions(asset, features):
    half = lambda ep, t: 0.9 if t < ep.T / 2 else 0.1
    t = run(cfg_long(label_pE=True, record_verifier_inputs=True), asset, features, InspectController(),
            TierVerifier(SEQ, half), reader=LabelReader())
    ck = checks(t)
    assert ck and all("letter_probs" in c and "verifier_input" in c for c in ck)
    for c in ck:                     # SFT rows map a tier to its (shuffled) candidate position
        inp = c["verifier_input"]
        assert all(inp["payloads"][tier_position(inp, k)]["tier"] == k for k in range(4))
        fi = c["frames_input"]                                   # one joint input per batch (verifier_frame role)
        assert sum(pl["keep"] for pl in fi["payloads"]) == len(c["frames"])


def test_evidence_expert_retries_once_then_moves_when_little_is_kept(asset, features):
    from src.baselines.teacher import EvidenceExpertController
    T = asset.duration
    ctrl = EvidenceExpertController({SAMPLE.sample_id: {"key_spans": [[0.0, T]]}}, pad=0.0)   # evidence everywhere
    t = run(cfg_long(), asset, features, ctrl, TierVerifier(SEQ, lambda ep, t: 0.1))           # nothing is kept
    rules = [d["info"]["rule"] for d in t["decisions"] if d["decision_type"] == "controller"]
    assert "kept_little_retry" in rules
    k = rules.index("kept_little_retry")
    assert any(r.startswith("kept_little_twice") for r in rules[k + 1:]), rules


def test_verifier_labels_use_only_the_frames_the_verifier_sees(asset, features):
    class CountingReader(LabelReader):
        def __init__(self):
            super().__init__()
            self.n_frames = []

        def answer_letter_probs(self, jobs):
            self.n_frames += [len(j["frames"]) for j in jobs]
            return super().answer_letter_probs(jobs)
    for no_fill in (True, False):
        r = CountingReader()
        t = run(cfg_long(label_pE=True, label_no_fill=no_fill), asset, features, InspectController(), TierVerifier(SEQ), reader=r)
        assert r.n_frames
        if no_fill:      # overview (16) + kept frames so far, never filled to 32
            assert all(n < 32 for n in r.n_frames), r.n_frames
        else:
            assert all(n == 32 for n in r.n_frames), r.n_frames


def test_oracle_frame_verifier_keeps_frames_in_the_evidence(asset, features):
    T = asset.duration if hasattr(asset, "duration") else asset.time(asset.num_frames - 1)
    span = (0.2 * T, 0.4 * T)
    cfg = cfg_long(verifier_frame_oracle_pad=0.0)
    dec = {"controller": InspectController(), "localize": FirstValid(), "frame": FirstValid(),
           "verifier": TierVerifier(SEQ, lambda ep, t: 0.0)}    # the verifier's per-frame answer would reject all
    runner = RunnerV3(cfg, dec, StubReader(), features, StubEncoder(), text_len=None,
                      annotations={SAMPLE.sample_id: {"key_spans": [span]}})
    t = runner.run([{"sample": SAMPLE, "asset": asset, "group_id": "g", "trajectory_id": "t0", "seed": 0}])[0]
    frames = [f for c in checks(t) for f in c["frames"]]
    assert frames and any(f["kept"] for f in frames)
    for f in frames:
        inside = span[0] <= f["time"] <= span[1]
        assert f["p_keep"] == float(inside) and (inside or not f["kept"])   # kept only inside (16-frame cap may bind)
    assert checks(t)[-1]["kept_total"] <= 16


def test_uncovered_segments_send_where_and_enclosing_to_the_next_segment():
    """Rows marked evidence_multi_segment (VideoITG experiment, 2026-10-20): each segment's share of the kept-frame cap
    follows its length; a covered segment drops out of the where / enclosing labels (and of the evidence expert's rules).
    Unmarked rows never call uncovered_spans."""
    from src.baselines.evidence import enclosing_label, uncovered_spans, where_label
    sp = [(100.0, 110.0), (500.0, 530.0)]                                   # 10 s + 30 s, cap 16 -> shares 4 + 12
    assert uncovered_spans(sp, [101.0, 102.0, 103.0], 16) == sp
    left = uncovered_spans(sp, [101.0, 102.0, 103.0, 104.0], 16)
    assert left == [(500.0, 530.0)]
    assert where_label(sp, 90.0, 120.0) == "inside_large" and where_label(left, 90.0, 120.0) == "after"
    assert enclosing_label(sp, (100.0, 105.0), [(0.0, 250.0), (0.0, 1000.0)], []) == 1
    assert enclosing_label(left, (100.0, 105.0), [(0.0, 250.0), (0.0, 1000.0)], []) == 2
    assert uncovered_spans(sp, [101.0, 102.0, 103.0, 104.0] + [505.0] * 12, 16) == sp      # all covered: every one
    assert uncovered_spans([(100.0, 110.0)], [101.0] * 15, 16) == [(100.0, 110.0)]         # one segment: as before
    assert uncovered_spans([(0.0, 1.0), (100.0, 400.0)], [], 16) == [(0.0, 1.0), (100.0, 400.0)]   # >= 1 frame each


def test_where_label_and_localize_targets():
    from src.baselines.evidence import localize_targets, padded_spans, where_label
    sp = padded_spans([[100.0, 110.0]], 5.0, 1000.0)                       # evidence 95-115 s
    assert where_label(sp, 0.0, 50.0) == "after" and where_label(sp, 200.0, 300.0) == "before"
    assert where_label(sp, 90.0, 120.0) == "inside_large"                  # 20 of 30 s
    assert where_label(sp, 0.0, 400.0) == "inside_small"                   # 20 of 400 s
    assert where_label(padded_spans([[10, 20], [500, 510]], 0.0, 1e9), 100.0, 200.0) == "before"   # nearest span
    wins = [{"start": 0, "end": 250}, {"start": 250, "end": 500}, {"start": 0, "end": 500}, {"start": 0, "end": 1000}]
    assert localize_targets(wins, [True] * 4, sp) == [0, 2]                # density 0.08, 0, 0.04, 0.02 (>= half of best)
    assert localize_targets(wins, [False, True, False, True], sp) == [3]
    assert localize_targets(wins[1:2], [True], sp) == []                   # nothing legal touches the evidence


class LocRecorder(FirstValid):
    def __init__(self):
        self.texts = []

    def decide(self, inputs, episodes, generators):
        self.texts += [(_text(i), i.candidate_ids) for i in inputs]
        return super().decide(inputs, episodes, generators)


def test_where_oracle_lines_controller_inside_locator_before_after(asset, features):
    T = asset.duration
    span = [T * 0.9, T * 0.95]                                             # in the last quarter only
    ctrl, loc, ver = AlwaysLocalize(), LocRecorder(), TierVerifier(SEQ)
    cfg = cfg_long(verifier_where_oracle_pad=0.0, **FLAT)
    dec = {"controller": ctrl, "localize": loc, "frame": FirstValid(), "verifier": ver}
    runner = RunnerV3(cfg, dec, StubReader(), features, StubEncoder(), text_len=None,
                      annotations={SAMPLE.sample_id: {"key_spans": [span]}})
    t = runner.run([{"sample": SAMPLE, "asset": asset, "group_id": "g", "trajectory_id": "t0", "seed": 0}])[0]
    assert ver.where_calls == 0                                            # oracle: the verifier is not asked
    wc = [x for x in t["turns"] if x["action"] == "WHERE_CHECK"]
    assert wc and all(sum(x["probs"].values()) == 1.0 for x in wc)
    first = wc[0]                                                          # FirstValid localizes to the first quarter
    assert first["window"][1] <= span[0] and first["probs"]["after"] == 1.0
    assert "Verifier, evidence vs the active interval: inside 0.00" in ctrl.texts[1] and "outside 1.00" in ctrl.texts[1]
    assert "Verifier, evidence vs" not in ctrl.texts[0]                    # full video: no where line
    glob = [x for x, ids in loc.texts if "Select the window of the whole video" in x]
    assert any("before it 0.00, after it 1.00" in x for x in glob)
    assert not any("before it" in x for x, ids in loc.texts if "of the active interval" in x)


def test_where_question_not_asked_on_the_full_video(asset, features):
    ver = TierVerifier(SEQ)
    t = run(cfg_long(tree_root_inspect=True), asset, features, InspectController(), ver)   # never leaves the full video
    assert ver.where_calls == 0 and not [x for x in t["turns"] if x["action"] == "WHERE_CHECK"]


def test_up_returns_to_parent_keeps_its_inspections_and_withdrawn_frames(asset, features):
    from src.env.candidates import Window
    T = asset.duration
    cfg = cfg_long()
    ep = __import__("src.env.episode_v3", fromlist=["EpisodeV3"]).EpisodeV3(SAMPLE, asset, features, StubEncoder(), cfg,
                                                                              key="k")
    ep.reset()
    q1 = Window(0.0, T / 4, "quarter")
    ep.apply_window("local", q1)
    ep.insp_count[q1.key()] = 1
    ep.withdrawn_idx.add(ep.unseen_candidates()[0])
    sub = ep.candidate_windows("local")[0]
    ep.apply_window("local", sub)
    assert ep.controller_mask()["UP"] and len(ep.path) == 2
    ep.apply_up()
    assert ep.active.key() == q1.key() and ep.path == [q1] and ep.inspections_here() == 1
    assert not ep.controller_mask()["UP"]                                  # level 1: UP illegal
    assert all(i not in ep.withdrawn_idx for i in ep.unseen_candidates())  # withdrawn frames never come back
    assert sub.key() not in {w.key() for w in ep.candidate_windows("local")}   # the visited child is not offered again


def test_inspect_margin_widens_candidates_not_the_window(asset, features):
    from src.env.candidates import Window
    from src.env.episode_v3 import EpisodeV3
    T = asset.duration
    cands = {}
    for m, want in ((0.0, (T / 4, T / 2)), (0.125, (T / 4 - T / 32, T / 2 + T / 32))):
        ep = EpisodeV3(SAMPLE, asset, features, StubEncoder(), cfg_long(inspect_margin=m), key="k")
        ep.reset()
        q = Window(T / 4, T / 2, "quarter")
        ep.apply_window("local", q)
        r = ep.inspect_range()
        assert abs(r.start - want[0]) < 1e-9 and abs(r.end - want[1]) < 1e-9 and ep.active.key() == q.key()
        cands[m] = ep.frame_candidates()
    assert cands[0.125] != cands[0.0]                                      # the widened range gives other candidates


def test_review_fixes_union_enclosing_and_expert_reachability(asset, features):
    from src.baselines.evidence import enclosing_label, overlap, window_cover
    from src.baselines.teacher import EvidenceExpertController
    from src.env.candidates import Window
    from src.env.episode_v3 import EpisodeV3
    sp = [(95.0, 135.0), (110.0, 145.0)]                                   # cgbench:8413 padded spans (overlap)
    assert abs(overlap(sp, 0.0, 124.0) - 29.0) < 1e-9 and abs(window_cover(sp, 0.0, 124.0) - 29.0 / 124) < 1e-9
    anc = [(0.0, 100.0), (0.0, 400.0)]                                     # parent, whole video
    assert enclosing_label([(10.0, 20.0)], (0.0, 25.0), anc, []) == 0      # only in the active window: nowhere else
    assert enclosing_label([(30.0, 40.0)], (0.0, 25.0), anc, []) == 1      # in an unvisited sibling -> SWITCH
    assert enclosing_label([(30.0, 40.0)], (0.0, 25.0), anc, [(25.0, 50.0)]) == 0   # that sibling was visited
    assert enclosing_label([(300.0, 310.0)], (0.0, 25.0), anc, []) == 2    # only above the parent -> UP
    T = asset.duration
    q1 = Window(0.0, T / 4, "quarter")
    c1 = Window(0.0, T / 16, "quarter")
    span = [T * 0.02, T * 0.04]                                            # inside c1 only
    ann = {SAMPLE.sample_id: {"key_spans": [span]}}
    ex = EvidenceExpertController(ann, pad=0.0)
    ep = EpisodeV3(SAMPLE, asset, features, StubEncoder(), cfg_long(min_window_sec=0.05), key="k")
    ep.reset()
    ep.apply_window("local", q1)
    ep.apply_window("local", c1)
    sib = [w for w in ep.candidate_windows("sibling")][0]                  # an evidence-free sibling
    ep.apply_window("sibling", sib)
    d = ex.decide([ep.controller_input()], [ep], [None])[0]
    assert d["info"]["rule"] == "off_evidence_up" and not d["info"]["dead_end"]   # UP back to q1 (evidence there)
    assert ep.controller_input().candidate_ids[d["index"]] == "UP"
    ep.apply_up()
    d = ex.decide([ep.controller_input()], [ep], [None])[0]
    act = ep.controller_input().candidate_ids[d["index"]]
    assert act == "INSPECT_FRAMES", (act, d["info"])                       # no REFINE into evidence-free children


def test_where_sees_window_frames_margin_counts_as_inside_and_no_root_inspect(asset, features):
    from src.baselines.evidence import padded_spans, where_label
    from src.baselines.teacher import EvidenceExpertController
    from src.env.candidates import Window
    from src.env.episode_v3 import EpisodeV3
    T = asset.duration
    ep = EpisodeV3(SAMPLE, asset, features, StubEncoder(), cfg_long(), key="k")
    ep.reset()
    assert not ep.controller_mask()["INSPECT_FRAMES"] and ep.controller_mask()["REFINE_LOCAL"]   # F: root -> REFINE
    root_txt = _text(ep.verifier_input(with_where=True))
    assert "A01 [" not in root_txt                                         # B: root keeps the overview only
    q = Window(T / 4, T / 2, "quarter")
    ep.apply_window("local", q)
    txt = _text(ep.verifier_where_input())
    assert "A01 [" in txt and "A16 [" in txt and "G01 [" in txt            # B: 16 frames of the inspection range
    span = [T / 2 + T / 64, T / 2 + T / 48]                                # outside q, inside its 1/8 margin
    r = ep.inspect_range()
    assert where_label(padded_spans([span], 0.0, T), r.start, r.end) in ("inside_small", "inside_large")   # D
    ex = EvidenceExpertController({SAMPLE.sample_id: {"key_spans": [span]}}, pad=0.0)
    d = ex.decide([ep.controller_input()], [ep], [None])[0]
    assert ep.controller_input().candidate_ids[d["index"]] == "INSPECT_FRAMES" and d["info"]["rule"].startswith("on_evidence")


def test_controller_sees_low_keep_count_in_this_window(asset, features):
    from src.env.candidates import Window
    from src.env.episode_v3 import EpisodeV3
    T = asset.duration
    texts = []
    for seq in ([2, 8, 2], [6, 4, 2]):                    # same attempts / last result / kept total, different low counts
        ep = EpisodeV3(SAMPLE, asset, features, StubEncoder(), cfg_long(), key="k")
        ep.reset()
        ep.apply_window("local", Window(T / 4, T / 2, "quarter"))
        w = ep.active
        for k in seq:
            ep.attempt_log.append({"turn": 0, "window": (w.start, w.end), "active_set_turn": ep.active_set_turn,
                                   "picked": 8, "kept": k})
        ep.insp_count[w.key()] = len(seq)
        texts.append(_text(ep.controller_input()))
    assert "inspection attempts in this window 3, of which kept less than half: 2" in texts[0]
    assert "inspection attempts in this window 3, of which kept less than half: 1" in texts[1]


def test_encoder_cap_makes_inspect_illegal_instead_of_crashing(asset, features):
    from src.baselines.teacher import EvidenceExpertController
    T = asset.duration
    ann = {SAMPLE.sample_id: {"key_spans": [[T * 0.30, T * 0.33]]}}
    cfg = cfg_long()
    cfg["video"]["encoder_frame_cap"] = 40                 # overview (14 here) + a little: runs out after a few windows
    dec = {"controller": EvidenceExpertController(ann, pad=0.0), "localize": FirstValid(), "frame": FirstValid(),
           "verifier": TierVerifier(SEQ, lambda ep, t: 0.1)}
    runner = RunnerV3(cfg, dec, StubReader(), features, StubEncoder(), text_len=None, annotations=ann)
    t = runner.run([{"sample": SAMPLE, "asset": asset, "group_id": "g", "trajectory_id": "t0", "seed": 0}])[0]
    assert t["status"] == "ok" and t["summary"]["cost"]["encoder_frames"] <= 40
    assert not any(x.get("tool_status") == "no_candidates" for x in t["turns"])   # the mask prevented it


def test_rule_b_refines_to_a_denser_grid_after_two_low_keeps(asset, features):
    """cgbench:10290-like: evidence covers >= 1/4 of the window, two inspections kept little (grid exhausted) -> the
    expert refines into an unvisited child whose own grid still has new evidence points, instead of leaving."""
    from src.baselines.evidence import padded_spans
    from src.baselines.teacher import EvidenceExpertController
    from src.env.candidates import Window
    from src.env.episode_v3 import EpisodeV3
    T = asset.duration
    span = [T * 0.30, T * 0.42]                                            # ~48% of the quarter [T/4, T/2]
    ep = EpisodeV3(SAMPLE, asset, features, StubEncoder(), cfg_long(min_window_sec=0.05), key="k")
    ep.reset()
    q = Window(T / 4, T / 2, "quarter")
    ep.apply_window("local", q)
    for k in (1, 0):                                                       # two low keeps in this window
        ep.attempt_log.append({"turn": 0, "window": (q.start, q.end), "active_set_turn": ep.active_set_turn,
                               "picked": 8, "kept": k})
    ep.insp_count[q.key()] = 2
    sp = padded_spans([span], 0.0, T)
    kids = ep.candidate_windows("local")
    assert any(ep.new_grid_points(w, sp) > 0 for w in kids)
    ex = EvidenceExpertController({SAMPLE.sample_id: {"key_spans": [span]}}, pad=0.0)
    d = ex.decide([ep.controller_input()], [ep], [None])[0]
    assert d["info"]["rule"] == "kept_little_twice_denser", d["info"]
    assert ep.controller_input().candidate_ids[d["index"]] == "REFINE_LOCAL"
    ep.turn = int(ep.b["max_turns"]) - 1                                   # only one action left: no denser refine
    d = ex.decide([ep.controller_input()], [ep], [None])[0]
    assert d["info"]["rule"] != "kept_little_twice_denser"


def test_feature_store_writes_only_its_own_entries_and_evicts(tmp_path):
    import torch
    from src.video.assets import FeatureStore
    a = FeatureStore(tmp_path, "tag", shard="runA", keep_dtype=True)
    a.put("v1", [1, 2], torch.ones(2, 3))
    a.flush()
    b = FeatureStore(tmp_path, "tag", shard="runB", keep_dtype=True, max_videos=1)
    assert b.has("v1", 1)                                  # reads runA's shard
    b.put("v1", [2, 3], torch.zeros(2, 3))                 # pts 2 already known: not re-stored
    b.flush()
    blob = torch.load(tmp_path / "v1" / "tag" / "feats.runB.pt")
    assert blob["pts"].tolist() == [3]                     # only its own new entry, no copy of runA's
    b.put("v2", [7], torch.ones(1, 3))                     # loads v2 -> evicts v1 (max_videos=1)
    assert list(b._mem) == ["v2"]
    b.put("v3", [8], torch.ones(1, 3))                     # evicts dirty v2: flushed first
    assert (tmp_path / "v2" / "tag" / "feats.runB.pt").exists()
    c = FeatureStore(tmp_path, "tag", shard="runB", keep_dtype=True)
    c.put("v1", [4], torch.ones(1, 3))                     # resume the same shard: keeps its earlier entries
    c.flush()
    assert torch.load(tmp_path / "v1" / "tag" / "feats.runB.pt")["pts"].tolist() == [3, 4]
    d = FeatureStore(tmp_path, "tag", shard="x", keep_dtype=True)
    assert all(d.has("v1", p) for p in (1, 2, 3, 4))


def test_feature_store_skips_an_unreadable_shard(tmp_path):
    import torch
    from src.video.assets import FeatureStore
    a = FeatureStore(tmp_path, "tag", shard="runA", keep_dtype=True)
    a.put("v1", [1, 2], torch.ones(2, 3))
    a.flush()
    good = (tmp_path / "v1" / "tag" / "feats.runA.pt").read_bytes()
    (tmp_path / "v1" / "tag" / "feats.runC.pt").write_bytes(good[: len(good) // 2])   # a truncated copy
    b = FeatureStore(tmp_path, "tag", shard="runB", keep_dtype=True)
    assert b.has("v1", 1) and not b.has("v1", 5)          # the readable shard still counts, no exception
