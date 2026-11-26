"""Verifier wrong-window data (2026-10-09): the oracle where line is refused in localize rows
(JSON prefix entries are dicts), and the injection wrappers (src/baselines/inject.py) enter the locator's most probable
window without evidence, force one inspection there, then let the expert act; the forced decisions make no SFT rows,
the verifier rows from the injected window are labelled outside / not kept and tagged."""
import json
import subprocess
import sys

import torch

from conftest import ROOT, StubEncoder, StubReader
from test_long_flow import SAMPLE, SEQ, FirstValid, LocRecorder, TierVerifier, _res, cfg_long
from src.baselines.evidence import EvidenceGrounding
from src.baselines.inject import InjectInspectController, InjectWrongLocalizer
from src.baselines.teacher import EvidenceExpertController
from src.common.utils import write_jsonl
from src.rollout.runner_v3 import RunnerV3


class ProbLocator:
    """Fixed probabilities by window start (share of the video): the highest-ranked legal window is executed."""
    name = "prob_locator"

    def __init__(self, T, by_quarter):
        self.T, self.by_quarter = T, by_quarter        # quarter index of the window start -> unnormalised score

    def decide(self, inputs, episodes, generators):
        out = []
        for inp in inputs:
            legal = inp.legal_indices()
            s = [self.by_quarter.get(int(round(4 * pl["start"] / self.T)) % 4, 0.01) if inp.valid[i] else 0.0
                 for i, pl in enumerate(inp.payloads)]
            z = sum(s)
            probs = [x / z for x in s]
            out.append(_res(max(legal, key=lambda i: probs[i]), len(inp.candidates), probs))
        return out


def _gen_rows(tmp_path, t, span, roles, extra=()):
    t["evaluator"] = {"correct": True}
    write_jsonl(tmp_path / "traj.jsonl", [t])
    man = tmp_path / "man"
    man.mkdir(exist_ok=True)
    write_jsonl(man / "m.jsonl", [{"sample_id": "s1", "answer": "B", "key_spans": [span]}])
    (tmp_path / "cfg.yaml").write_text(f"paths:\n  manifest_dir: {man}\n")
    out = tmp_path / "rows.jsonl"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "gen_role_sft_from_trajectories.py"), "--config",
                    str(ROOT / "configs" / "default.yaml"), str(tmp_path / "cfg.yaml"), "--trajectories",
                    str(tmp_path / "traj.jsonl"), "--manifest", "m.jsonl", "--split", "train", "--roles", *roles,
                    "--evidence-pad", "0", *extra, "--out", str(out)], check=True, capture_output=True)
    return [json.loads(l) for l in open(out)], json.load(open(str(out) + ".stats.json"))


def _text(prefix):
    return "".join(x if isinstance(x, str) else x.get("t", "") for x in prefix if isinstance(x, (str, dict)))


def test_oracle_where_line_refused_in_localize_rows_from_json(asset, features, tmp_path):
    """SWITCH locator inputs carry the oracle "before / after" line when budget.verifier_where_oracle_pad is set; those
    localize rows are refused unless --allow-oracle-inputs (before the 2026-10-09 fix none was refused)."""
    T = asset.duration
    span = [T * 0.30, T * 0.33]                        # 2nd quarter; FirstValid enters the 1st, the expert switches
    ann = {SAMPLE.sample_id: {"key_spans": [span]}}
    cfg = cfg_long(verifier_where_oracle_pad=0.0)
    dec = {"controller": EvidenceExpertController(ann, pad=0.0),
           "localize": EvidenceGrounding("localize", ann, 0.0, inner=LocRecorder()), "frame": FirstValid(),
           "verifier": TierVerifier(SEQ)}
    runner = RunnerV3(cfg, dec, StubReader(), features, StubEncoder(), text_len=None, annotations=ann)
    t = runner.run([{"sample": SAMPLE, "asset": asset, "group_id": "g", "trajectory_id": "t0", "seed": 0}])[0]
    with_line = [d for d in t["decisions"] if d["decision_type"] == "localize"
                 and "Verifier, evidence vs the interval" in _text(d["input"]["prefix"])]
    assert with_line                                   # the SWITCH locator input has the oracle line
    rows, st = _gen_rows(tmp_path, t, span, ["localize"])
    assert st.get("skip_oracle_input_localize", 0) == len(with_line)
    assert not any("Verifier, evidence vs the interval" in _text(r["input"]["prefix"]) for r in rows)
    rows2, _ = _gen_rows(tmp_path, t, span, ["localize"], ["--allow-oracle-inputs"])
    assert any("Verifier, evidence vs the interval" in _text(r["input"]["prefix"]) for r in rows2)


def _inject_run(asset, features, span, by_quarter, p=1.0, min_prob=0.05, max_inject=2, keep=0.9):
    T = asset.duration
    ann = {SAMPLE.sample_id: {"key_spans": [span]}}
    cfg = cfg_long(record_verifier_inputs=True)
    loc = InjectWrongLocalizer(EvidenceGrounding("localize", ann, 0.0, inner=ProbLocator(T, by_quarter)), ann, 0.0,
                               p=p, min_prob=min_prob, max_inject=max_inject)
    dec = {"controller": InjectInspectController(EvidenceExpertController(ann, pad=0.0)), "localize": loc,
           "frame": FirstValid(), "verifier": TierVerifier(SEQ, lambda ep, t: keep)}
    runner = RunnerV3(cfg, dec, StubReader(), features, StubEncoder(), text_len=None, annotations=ann)
    return runner.run([{"sample": SAMPLE, "asset": asset, "group_id": "g", "trajectory_id": "t0", "seed": 0}])[0]


# evidence in the 2nd quarter (the locator's first choice); the synthetic video has few frames before 1.6 s, so the
# injected windows are the 3rd (root: 0.25) and the 4th quarter (sibling call: 0.10 / 0.75 of the legal mass)
SPAN = (0.30, 0.33)
PROBS = {1: 0.60, 2: 0.25, 3: 0.10, 0: 0.05}


def test_wrong_windows_injected_at_two_positions_each_inspected_once_then_the_expert_recovers(asset, features, tmp_path):
    """Root call: the 3rd quarter (most probable without evidence, 0.25); the expert switches; sibling call: the 4th
    quarter (0.10 / 0.75 of the legal mass); the expert switches again; 3rd call: cap reached, the locator's own choice.
    The scripted verifier rejects the frames (keep 0.1), so the kept-frame budget is left for the recovery."""
    T = asset.duration
    span = [T * SPAN[0], T * SPAN[1]]
    t = _inject_run(asset, features, span, PROBS, keep=0.1)
    inj = t["injection"]
    assert inj["injected"] and [x["kind"] for x in inj["injections"]] == ["root", "sibling"]
    a, b = inj["injections"]
    assert abs(a["window"][0] - T / 2) < 1e-6 and abs(a["prob"] - 0.25) < 1e-6 and not a["natural"]
    assert abs(b["window"][0] - 0.75 * T) < 1e-6 and abs(b["prob"] - 0.10 / 0.75) < 1e-6
    assert a["inspected"] and b["inspected"] and a["label_action"] == b["label_action"] == "SWITCH_LOCAL"
    assert len(inj["calls"]) >= 3 and inj["calls"][1]["kind"] == "sibling"
    locs = [d for d in t["decisions"] if d["decision_type"] == "localize"]
    assert [d["forced"] for d in locs[:3]] == [True, True, False]
    assert abs(locs[2]["input"]["payloads"][locs[2]["action_index"]]["start"] - T / 4) < 1e-6   # back on the evidence
    ctrl = [d for d in t["decisions"] if d["decision_type"] == "controller"]
    forced = [d for d in ctrl if d["info"].get("executed") == "injected"]
    assert len(forced) == 2 and all(d["forced"] and d["action_id"] == "INSPECT_FRAMES" for d in forced)
    assert all(d["input"]["candidate_ids"][d["info"]["target_indices"][0]] == "SWITCH_LOCAL" for d in forced)
    insp = [x for x in t["turns"] if x["action"] == "TENTATIVE_CHECK"]
    assert [round(x["window"][0] / T, 3) for x in insp[:2]] == [0.5, 0.75]
    rng = lambda w: (w[0] - 0.125 * (w[1] - w[0]), w[1] + 0.125 * (w[1] - w[0]))   # noqa: E731 (inspect_margin 1/8)
    assert any(rng(x["window"])[0] < span[1] and span[0] < rng(x["window"])[1] for x in insp[2:])   # recovered
    rows, st = _gen_rows(tmp_path, t, span, ["localize", "controller", "verifier_where", "verifier_frame"])
    n_loc = sum(1 for d in locs if not d["forced"] and d["info"].get("target_indices"))
    assert sum(r["decision_type"] == "localize" for r in rows) == n_loc                 # the forced picks make no row
    crow = [r for r in rows if r["decision_type"] == "controller"]
    assert not any(r["meta"].get("executed") == "injected" for r in crow)
    assert sum(r["input"]["candidate_ids"][r["target"]["indices"][0]] == "SWITCH_LOCAL" for r in crow) >= 2
    wrows = [r for r in rows if r["source"] == "evidence_where" and r["meta"]["injected_window"]]
    assert {round(r["meta"]["window"][0] / T, 3) for r in wrows} == {0.5, 0.75}           # both positions
    assert {r["meta"]["inspections_here"] for r in wrows} == {0, 1} and {r["meta"]["where"] for r in wrows} == {"before"}
    frows = [r for r in rows if r["source"].startswith("evidence_frame") and r["meta"]["injected_window"]]
    assert len(frows) == 2 and all(not any(r["meta"]["evidence"]) if isinstance(r["meta"]["evidence"], list)
                                   else not r["meta"]["evidence"] for r in frows)       # frames there: not evidence
    assert st.get("verifier_where_injected_before", 0) == len(wrows) and st["rows_verifier_frame_injected"] == len(frows)


def test_two_injections_can_use_up_the_kept_frame_budget_but_their_verifier_states_are_recorded(asset, features):
    """With a verifier that keeps every frame (as verifier2 does in wrong windows), the two forced inspections fill the
    kept-frame budget and the episode ends; the where question after the 2nd inspection is still asked and recorded."""
    T = asset.duration
    span = [T * SPAN[0], T * SPAN[1]]
    t = _inject_run(asset, features, span, PROBS, keep=0.9)
    inj = t["injection"]["injections"]
    assert len(inj) == 2 and all(x["inspected"] for x in inj)
    assert t["summary"]["termination_reason"] == "frame_budget_exhausted"
    wc = [x for x in t["turns"] if x["action"] == "WHERE_CHECK" and abs(x["window"][0] - inj[1]["window"][0]) < 1e-6]
    assert {x["inspections_here"] for x in wc} == {0, 1}


def test_injection_cap_probability_gate_draw_and_natural_errors(asset, features):
    T = asset.duration
    span = [T * SPAN[0], T * SPAN[1]]
    t = _inject_run(asset, features, span, PROBS, max_inject=1)
    assert [x["kind"] for x in t["injection"]["injections"]] == ["root"]
    d = next(d for d in t["decisions"] if d["decision_type"] == "localize" and not d["forced"])
    assert abs(d["input"]["payloads"][d["action_index"]]["start"] - T / 4) < 1e-6        # cap: then the locator's choice
    t = _inject_run(asset, features, span, PROBS, min_prob=0.5)
    inj = t["injection"]
    assert not inj["injected"] and inj["calls"][0]["drawn"] and abs(inj["calls"][0]["best_wrong_prob"] - 0.25) < 1e-6
    assert not any(d["forced"] and d["info"].get("injected") for d in t["decisions"])
    t = _inject_run(asset, features, span, PROBS, p=0.0)
    assert not t["injection"]["injected"] and not any(c["drawn"] for c in t["injection"]["calls"])
    t = _inject_run(asset, features, span, {2: 0.7, 1: 0.2, 0: 0.05, 3: 0.05}, max_inject=1)   # the locator is wrong
    inj = t["injection"]
    assert inj["injected"] and inj["injections"][0]["natural"]
    loc0 = next(d for d in t["decisions"] if d["decision_type"] == "localize")
    assert not loc0["forced"] and "injected" not in loc0["info"]                       # an ordinary labelled decision
    assert any(d["info"].get("executed") == "injected" for d in t["decisions"] if d["decision_type"] == "controller")


def test_balance_weights_follow_the_training_mixture_and_renormalise():
    from scripts.train_sft import apply_balance, balance_weights
    main = ([{"source": "evidence_where", "meta": {"where": "inside_large"}}] * 18
            + [{"source": "evidence_where", "meta": {"where": "after"}}] * 2
            + [{"source": "evidence_frames", "meta": {"evidence": [True, True, True, False]}}] * 5
            + [{"source": "label_pE", "meta": {}}] * 5)
    rep = [{"source": "evidence_where", "meta": {"where": "before"}}] * 10
    bw = balance_weights(main, rep, n_main=30, n_rep=10, clip=4.0)
    f = bw["where"]["freq"]          # per step: 18 + 2 where rows from main, 10 before rows from replay
    assert abs(f["inside_large"] - 18 / 30) < 1e-9 and abs(f["after"] - 2 / 30) < 1e-9 and abs(f["before"] - 10 / 30) < 1e-9
    w = bw["where"]["weight"]
    assert abs(sum(f[k] * w[k] for k in f) - 1.0) < 1e-9 and w["after"] > w["before"] > w["inside_large"]
    raw = {k: min(4.0, max(0.25, 1.0 / (3 * f[k]))) for k in f}   # clipped inverse frequency, then renormalised
    assert abs(w["after"] / w["inside_large"] - raw["after"] / raw["inside_large"]) < 1e-9
    fr = bw["frame"]
    assert abs(fr["freq"]["no"] - 0.25) < 1e-9 and abs(fr["weight"]["no"] / fr["weight"]["yes"] - 3.0) < 1e-9
    rows = [dict(r, meta=dict(r["meta"])) for r in main[17:22]]
    apply_balance(rows, bw)
    assert abs(rows[0]["weight"] - w["inside_large"]) < 1e-9 and abs(rows[1]["weight"] - w["after"]) < 1e-9
    assert rows[3]["group_weights"] == [fr["weight"]["yes"]] * 3 + [fr["weight"]["no"]] and "weight" not in rows[3]


def test_group_weights_in_the_grouped_loss_and_balanced_dev_metrics(tiny_policy):
    from scripts.train_sft import _auc, decision_loss, selection_score, verifier_balanced_metrics
    from src.common.types import DecisionInput
    from src.env.environment import nanojev_candidate
    ids = ["E1+", "E1-", "E2+", "E2-"]
    inp = DecisionInput("verifier", ["State: x\n"], [nanojev_candidate(f"{c}: y{c}") for c in ids], ids, [True] * 4,
                        [{"keep": c.endswith("+"), "frame": j // 2, "time": float(j // 2), "group": j // 2}
                         for j, c in enumerate(ids)])
    with torch.no_grad():
        z = tiny_policy([inp])[0].float()[0, :4]
    pair = [torch.log_softmax(z[2 * g:2 * g + 2], -1) for g in range(2)]
    row = {"input": inp.to_json(), "target": {"indices": [0, 3]}, "sample_id": "s", "group_weights": [0.5, 3.0],
           "source": "evidence_frames", "meta": {"evidence": [True, False], "injected_window": True}}
    d = {}
    loss, _, _ = decision_loss(tiny_policy, row, torch.device("cpu"), detail=d)
    assert abs(float(loss) - float(-(0.5 * pair[0][0] + 3.0 * pair[1][1]) / 2)) < 1e-4
    assert d["group_hits"] == [bool(pair[0][0] > pair[0][1]), bool(pair[1][1] > pair[1][0])]
    wids = ["P1", "P2", "P3", "P4"]
    winp = DecisionInput("verifier", ["State: w\n"], [nanojev_candidate(f"{c}: y{c}") for c in wids], wids, [True] * 4,
                         [{"where": k, "group": 0} for k in ("before", "inside_small", "inside_large", "after")])
    wrows = [{"input": winp.to_json(), "target": {"indices": [i]}, "sample_id": "s", "source": "evidence_where",
              "meta": {"where": k, "inspections_here": 1, "injected_window": k == "after"}}
             for i, k in ((3, "after"), (2, "inside_large"))]
    rows = [row] + wrows
    dets = [{} for _ in rows]
    res = [decision_loss(tiny_policy, r, torch.device("cpu"), detail=dd) for r, dd in zip(rows, dets)]
    m = verifier_balanced_metrics(rows, res, dets)
    assert set(m) >= {"verifier_bal/where_macro_recall", "verifier_bal/frame_bal_acc", "verifier_bal/score",
                      "verifier_bal/outside_auc_post", "verifier_bal/injected_frames_kept"}
    assert abs(m["verifier_bal/score"] - (m["verifier_bal/where_macro_recall"] + m["verifier_bal/frame_bal_acc"]) / 2) < 1e-9
    assert m["verifier_bal/injected_frames_kept"] == float(not d["group_hits"][1])
    assert selection_score({"dev/verifier_bal/score": 0.42, "dev/verifier_where/argmax_in_target": 0.9}) == 0.42
    assert _auc([0.9, 0.8], [0.1, 0.8]) == 0.875 and _auc([1.0], []) is None
