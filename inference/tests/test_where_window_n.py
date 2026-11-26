"""budget.where_window_n: the where question's window frames A01.. are taken evenly from the
inspection candidate grid; 32 = every candidate, so the verifier sees exactly the frames the selector then picks from;
null keeps the old 16 (every other candidate). Checked on a real episode flow: the where question asked on entering a
window vs the frame decision of the inspection that follows in the same turn."""
from fractions import Fraction

import av
import numpy as np
import pytest

from conftest import StubEncoder, StubReader, frame_color
from src.baselines.teacher import EvidenceExpertController
from src.rollout.runner_v3 import RunnerV3
from src.video.assets import VideoAsset
from test_long_flow import SAMPLE, SEQ, FirstValid, TierVerifier, cfg_long

TOL = 1e-6


def kind(d):
    pls = d["input"]["payloads"]
    if d["decision_type"] != "verifier":
        return d["decision_type"]
    if any("tier" in p for p in pls):
        return "q3+where" if any("where" in p for p in pls) else "q3"
    if any("where" in p for p in pls):
        return "where"
    return "keep" if any("keep" in p for p in pls) else "other"


@pytest.fixture(scope="module")
def dense_asset(tmp_path_factory):
    """60 s at 25 fps (constant): every inspected window holds well over 32 distinct frames, so 16 vs 32 differ
    (the VFR clip of conftest is 6.1 s; its deep windows have ~11 frames)."""
    d = tmp_path_factory.mktemp("dense")
    path = d / "cfr60.mp4"
    with av.open(str(path), "w") as out:
        st = out.add_stream("mpeg4", rate=25)
        st.width, st.height, st.pix_fmt = 64, 48, "yuv420p"
        st.time_base = Fraction(1, 1000)
        st.codec_context.time_base = Fraction(1, 1000)
        for i in range(1500):
            fr = av.VideoFrame.from_ndarray(np.tile(frame_color(i % 100), (48, 64, 1)), format="rgb24")
            fr.pts, fr.time_base = 40 * i, Fraction(1, 1000)
            for pkt in st.encode(fr):
                out.mux(pkt)
        for pkt in st.encode():
            out.mux(pkt)
    return VideoAsset(path, d / "cache", max_side=None)


def run(asset, features, **budget):
    T = asset.duration
    span = [T * 0.30, T * 0.33]
    ann = {SAMPLE.sample_id: {"key_spans": [span]}}
    keep = lambda ep, t: 0.9 if span[0] - 0.3 <= t <= span[1] + 0.3 else 0.1
    cfg = cfg_long(answer_uniform=24, kept_frames=8, frames_per_inspect=4, record_verifier_inputs=True, **budget)
    dec = {"controller": EvidenceExpertController(ann, pad=0.0), "localize": FirstValid(), "frame": FirstValid(),
           "verifier": TierVerifier(SEQ, keep)}
    runner = RunnerV3(cfg, dec, StubReader(), features, StubEncoder(), text_len=None, annotations=ann)
    return runner.run([{"sample": SAMPLE, "asset": asset, "group_id": "g", "trajectory_id": "t0", "seed": 0}])[0], cfg


def window_times(d):
    """Times of the A.. frames of a recorded verifier input (the visual right after an "A<k> [..]" label)."""
    out, take = [], False
    for s in d["input"]["prefix"]:
        if isinstance(s, dict) and "t" in s:
            last = s["t"].rstrip().split("\n")[-1]
            take = last.startswith("A") and last[1:3].isdigit()
        elif isinstance(s, dict) and "v" in s and take:
            out.append(float(s["v"][2]))
            take = False
    return out


def pairs(t):
    """(where decision on entering a window, frame decision of the inspection in the same turn)."""
    turns = {}
    for x in t["turns"]:
        turns.setdefault(x["turn_id"], []).append(x)
    out = []
    for tid, xs in turns.items():
        w = [x for x in xs if x["action"] == "WHERE_CHECK" and x["inspections_here"] == 0]
        if not w or not any(x["action"] == "INSPECT_FRAMES" for x in xs):
            continue
        wd = [d for d in t["decisions"] if d["turn_id"] == tid and kind(d) == "where"]
        fd = [d for d in t["decisions"] if d["turn_id"] == tid and d["decision_type"] == "frame"]
        if wd and fd:
            out.append((wd[0], fd[0]))
    return out


@pytest.mark.parametrize("n", [None, 32])
def test_where_window_frames_come_from_the_candidate_grid(dense_asset, features, n):
    t, cfg = run(dense_asset, features, **({} if n is None else {"where_window_n": n}))
    assert t["status"] == "ok", t.get("error")
    ps = pairs(t)
    assert ps, "no where question followed by an inspection in the same window"
    full = 0
    for wd, fd in ps:
        a = window_times(wd)
        c = [float(p["t"]) for p in fd["input"]["payloads"]]        # the selector's candidates (distinct frames)
        assert len(c) <= cfg["budget"]["frame_candidates"] == 32
        if len(c) == 32:
            full += 1
            want = c if n == 32 else c[::2]                           # 32: every candidate; default: every other one
            assert int(cfg["video"]["refine_frames"]) == 16 and len(want) == (32 if n == 32 else 16)
        else:
            k = min(len(c), n or int(cfg["video"]["refine_frames"]))
            want = c[::max(1, len(c) // k)][:k]
        assert len(a) == len(want) and all(abs(x - y) < TOL for x, y in zip(a, want)), (len(a), len(want), a[:4], want[:4])
    assert full, "no inspected window with a full grid of 32 distinct candidates: the test would not separate 16 and 32"


def test_where_window_n_reaches_the_cache_keys_and_changes_nothing_else(dense_asset, features):
    """Same controller actions with 16 and 32 window frames under scripted deciders (the A frames only feed the verifier,
    whose scripted answers do not read them)."""
    t16, _ = run(dense_asset, features)
    t32, _ = run(dense_asset, features, where_window_n=32)
    assert [x["action"] for x in t16["turns"]] == [x["action"] for x in t32["turns"]]
    a16 = [len(window_times(d)) for d in t16["decisions"] if kind(d) in ("where", "q3+where")]
    a32 = [len(window_times(d)) for d in t32["decisions"] if kind(d) in ("where", "q3+where")]
    assert len(a16) == len(a32) and all(y >= x for x, y in zip(a16, a32)) and sum(a32) > sum(a16)
