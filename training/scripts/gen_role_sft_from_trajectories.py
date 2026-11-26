"""SFT rows for the controller (and optionally localize / frame) from rule-teacher trajectories of the new design
(scripts/run_agent_v3.py --controller teacher --localizer teacher --selector teacher [--teacher-mode recovery]), and
verifier rows from any trajectories run with budget.label_pE=true. CPU only.

controller : target = the teacher's action (info.target_indices); decisions that were forced (one legal action) are
             skipped; rows carry the teacher rule name for analysis. Optional action balancing (--balance-actions):
             weight = n / (K * count(action)) clipped to [0.25, 4], renormalized to mean 1.
localize   : target = info.target_indices (max-IoU windows); deliberately wrong recovery moves (empty targets) skipped.
frame      : target set = info.target_indices, picks = info.picks (selection C; the SFT samples its own order).
localize / frame rows also come from any run with --label-grounding or the `evidence` deciders (info.target_indices
             from the evidence annotation, src/baselines/evidence.py).
verifier_frame : per-frame keep rows (budget.record_verifier_inputs): yes iff the frame is in key_spans +- --evidence-pad.
verifier_where : "where is the evidence" rows (budget.record_verifier_inputs): src/baselines/evidence.where_label with
             --evidence-pad and --refine-frac; for manifest rows marked evidence_multi_segment (the VideoITG experiment,
             2026-10-20) on the segments the frames kept at that point do not cover yet (evidence.uncovered_spans).
verifier   : one row per stoppable prefix: input = the recorded verifier DecisionInput, target tier from
             p_E = letter_probs[correct letter] with cuts --cuts (0.25 0.5 0.8); needs the manifest for labels.
usage: python scripts/gen_role_sft_from_trajectories.py --trajectories runs/teach_q25_direct/trajectories.jsonl
           runs/teach_q25_recovery/trajectories.jsonl --manifest qaspan/train.jsonl --split train --roles controller
           --out data/sft/controller_q25_train.jsonl
"""
import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.config import load_config  # noqa: E402
from src.common.utils import read_jsonl, write_json, write_jsonl  # noqa: E402
from src.baselines.evidence import enclosing_label, padded_spans, uncovered_spans, where_label, window_cover  # noqa: E402
from src.env.episode_v3 import tier_position  # noqa: E402


def tier(p, cuts):
    return sum(p >= c for c in cuts)


def prefix_text(prefix):
    """Text of a DecisionInput prefix: str entries in memory, {"t": str} / {"v": [...]} entries in trajectory JSON
    (2026-10-09 fix: the oracle-line check tested isinstance(x, str) only, so it never fired on JSON trajectories)."""
    return "".join(x if isinstance(x, str) else x.get("t", "") for x in prefix if isinstance(x, (str, dict)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", default=[str(ROOT / "configs/default.yaml")])
    ap.add_argument("--trajectories", nargs="+", required=True)
    ap.add_argument("--manifest", required=True, help="manifest_dir/file with the labels (for verifier rows)")
    ap.add_argument("--split", required=True, choices=["train", "dev"])
    ap.add_argument("--roles", nargs="+", default=["controller"],
                    choices=["controller", "localize", "frame", "verifier", "verifier_frame", "verifier_where"])
    ap.add_argument("--evidence-pad", type=float, default=None,
                    help="evidence padding in seconds; defaults to budget.evidence_pad in --config")
    ap.add_argument("--refine-frac", type=float, default=0.25,
                    help="verifier_where: inside_small iff the evidence covers less than this share of the window")
    ap.add_argument("--cuts", nargs=3, type=float, default=[0.25, 0.5, 0.8])
    ap.add_argument("--balance-actions", action="store_true")
    ap.add_argument("--allow-oracle-inputs", action="store_true",
                    help="keep controller rows from trajectories run with an oracle verifier (frame / where pad), and "
                         "localize rows whose input carries an oracle where line (refused by default: the controller and "
                         "locator must be trained on the trained verifier's predictions)")
    ap.add_argument("--keep-dead-end", action="store_true",
                    help="keep controller rows the evidence expert marked dead_end (no action can reach the evidence; "
                         "dropped by default, 2026-10-06)")
    ap.add_argument("--soft-target", type=float, default=None,
                    help="controller rows: soft target with this mass on the executed action and the rest spread evenly "
                         "over the other legal actions (2026-10-05, imitation of a fixed flow that keeps entropy for RL)")
    ap.add_argument("--correct-in-mixed", action="store_true",
                    help="rejection-sampling SFT (2026-10-04): keep only correct trajectories of questions whose sampled "
                         "trajectories (all files together) are neither all right nor all wrong; target = the sampled action")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.evidence_pad is None:
        args.evidence_pad = float(cfg["budget"]["evidence_pad"])
    for path in args.trajectories:     # label definitions must match the run that produced the trajectories
        run_cfg = Path(path).parent / "config.yaml"
        if run_cfg.exists():
            import yaml
            rb = (yaml.safe_load(open(run_cfg)) or {}).get("budget", {})
            for key, val in (("evidence_pad", args.evidence_pad), ("evidence_refine_frac", args.refine_frac)):
                if key in rb and rb[key] is not None and abs(float(rb[key]) - float(val)) > 1e-9:
                    raise SystemExit(f"{run_cfg}: budget.{key}={rb[key]} but the labels would use {val}; "
                                     f"pass the matching --evidence-pad / --refine-frac")
    man = Path(cfg["paths"]["manifest_dir"]) / args.manifest
    man_rows = read_jsonl(man)
    labels = {r["sample_id"]: r["answer"] for r in man_rows}
    spans = {r["sample_id"]: r["key_spans"] for r in man_rows if r.get("key_spans")}
    multi = {r["sample_id"] for r in man_rows if r.get("evidence_multi_segment")}      # filter --multi-segment rows
    durs = {r["sample_id"]: float(r.get("duration") or 0) or float("inf") for r in man_rows}
    rows, stats = [], collections.Counter()
    mixed = None
    if args.correct_in_mixed:
        outc = collections.defaultdict(list)
        for path in args.trajectories:
            for t in read_jsonl(path):
                if t["status"] == "ok":
                    outc[t["sample_id"]].append(bool(t.get("evaluator", {}).get("correct")))
        mixed = {s for s, c in outc.items() if 0 < sum(c) < len(c)}
        stats["questions"], stats["questions_mixed"] = len(outc), len(mixed)
    for path in args.trajectories:
        for t in read_jsonl(path):
            stats["trajectories"] += 1
            if t["status"] != "ok":
                stats["skip_" + t["status"]] += 1
                continue
            if mixed is not None and (t["sample_id"] not in mixed or not t.get("evaluator", {}).get("correct")):
                stats["skip_not_correct_in_mixed"] += 1
                continue
            base = {"split": args.split, "sample_id": t["sample_id"], "trajectory_id": t["trajectory_id"], "weight": 1.0,
                    "trajectory_correct": bool(t.get("evaluator", {}).get("correct"))}
            orc = t.get("oracle") or {}
            oracle_used = any(v is not None for v in orc.values())
            inj = t.get("injection") or {}     # src/baselines/inject.py: rows from the injected wrong windows are tagged
            inj_wins = {tuple(round(v, 6) for v in x["window"]) for x in inj.get("injections", [])}
            in_inj = lambda w: w is not None and tuple(round(v, 6) for v in w) in inj_wins  # noqa: E731
            for d in t["decisions"]:
                typ = d["decision_type"]
                if typ not in args.roles or typ == "verifier":   # verifier rows come from the labelled checks below
                    continue
                if not args.allow_oracle_inputs and (
                        (typ == "controller" and oracle_used) or
                        (typ == "localize" and orc.get("where_pad") is not None and
                         "Verifier, evidence vs the interval" in prefix_text(d["input"]["prefix"]))):
                    stats[f"skip_oracle_input_{typ}"] += 1
                    continue
                if d["forced"]:
                    stats[f"skip_forced_{typ}"] += 1
                    continue
                info = d.get("info", {})
                if typ == "controller" and "target_indices" not in info:
                    # 2026-10-06 review: without an expert label the executed action would become its own target
                    # (self-imitation); run with --controller evidence_expert or --label-controller (DAgger)
                    stats["skip_unlabelled_controller"] += 1
                    continue
                if typ == "controller" and info.get("dead_end") and not args.keep_dead_end:
                    stats["skip_dead_end_controller"] += 1
                    continue
                tgt = info.get("target_indices", [d["action_index"]])
                if not tgt:
                    stats[f"skip_no_target_{typ}"] += 1
                    continue
                row = dict(base, input=d["input"], target={"indices": tgt}, decision_type=typ,
                           source=f"teacher:{typ}:{info.get('rule', info.get('mode', ''))}",
                           meta={k: info[k] for k in ("rule", "rule_state", "best_iou", "coverage_miss", "level", "fallback",
                                                      "executed") if k in info})
                if typ == "frame":
                    row["target"]["picks"] = info.get("picks", [d["action_index"]])
                if typ == "controller" and args.soft_target is not None:
                    legal = [i for i, v in enumerate(d["input"]["valid"]) if v]
                    others = [i for i in legal if i != tgt[0]]
                    q = [0.0] * len(d["input"]["valid"])
                    q[tgt[0]] = args.soft_target if others else 1.0
                    for i in others:
                        q[i] = (1.0 - args.soft_target) / len(others)
                    row["target"]["soft"] = q
                rows.append(row)
                stats[f"rows_{typ}"] += 1
            if "verifier_frame" in args.roles and t["sample_id"] in spans:   # per-frame keep labels (2026-10-06)
                pad = args.evidence_pad
                sp = [(a - pad, b + pad) for a, b in spans[t["sample_id"]]]
                for x in t["turns"]:
                    if "frames_input" in x:           # joint per-frame input (budget.verifier_frame_joint): one row,
                        inp = x["frames_input"]       # one target per frame pair
                        tg, ev = [], []
                        for k, pl in enumerate(inp["payloads"]):
                            yes = any(a <= pl["time"] <= b for a, b in sp)
                            if pl["keep"] == yes:
                                tg.append(k)
                            if pl["keep"]:
                                ev.append(yes)
                        rows.append(dict(base, input=inp, target={"indices": tg}, decision_type="verifier",
                                         source="evidence_frames",
                                         meta={"evidence": ev, "injected_window": in_inj(x.get("window"))}))
                        stats["rows_verifier_frame"] += 1
                        if in_inj(x.get("window")):
                            stats["rows_verifier_frame_injected"] += 1
                        stats["verifier_frame_yes"] += sum(ev)
                        stats["verifier_frame_no"] += len(ev) - sum(ev)
                    for f in x.get("frame_inputs", []):
                        yes = any(a <= f["time"] <= b for a, b in sp)
                        inp = f["input"]
                        tgt = next(k for k, pl in enumerate(inp["payloads"]) if pl["keep"] == yes)
                        rows.append(dict(base, input=inp, target={"indices": [tgt]}, decision_type="verifier",
                                         source="evidence_frame", meta={"time": f["time"], "evidence": yes,
                                                                        "injected_window": in_inj(x.get("window"))}))
                        stats["rows_verifier_frame"] += 1
                        stats[f"verifier_frame_{'yes' if yes else 'no'}"] += 1
            if "verifier_where" in args.roles and t["sample_id"] in spans:   # where-question rows (2026-10-06)
                for x in t["turns"]:
                    if x.get("action") != "WHERE_CHECK" or "input" not in x:
                        continue
                    # no clipping to the video needed: the window lies inside the video
                    rng = x.get("range") or x["window"]       # D: the inspection range (window + margins)
                    sp = padded_spans(spans[t["sample_id"]], args.evidence_pad, float("inf"))
                    if t["sample_id"] in multi:   # VideoITG experiment: the segments the kept frames do not cover yet,
                        # clipped to the video as in the collection's oracle (the segment shares follow their lengths)
                        sp = uncovered_spans(padded_spans(spans[t["sample_id"]], args.evidence_pad, durs[t["sample_id"]]),
                                             x.get("kept_times", []), x.get("frame_cap") or 0)
                    lab = where_label(sp, rng[0], rng[1], args.refine_frac)
                    pls = x["input"]["payloads"]
                    tgt = [next(k for k, pl in enumerate(pls) if pl.get("where") == lab)]
                    enc = None
                    if x.get("ancestors"):        # window tree: smallest enclosing interval with evidence (group 1)
                        enc = enclosing_label(sp, tuple(x["window"]), [tuple(w) for w in x["ancestors"]],
                                              [tuple(w) for w in x.get("visited", [])])
                        tgt.append(next(k for k, pl in enumerate(pls) if pl.get("enclosing") == enc))
                    rows.append(dict(base, input=x["input"], target={"indices": tgt}, decision_type="verifier",
                                     source="evidence_where", meta={"where": lab, "window": x["window"],
                                                                    "enclosing": enc,
                                                                    "inspections_here": x.get("inspections_here"),
                                                                    "injected_window": in_inj(x["window"])}))
                    stats["rows_verifier_where"] += 1
                    if in_inj(x["window"]):
                        stats[f"verifier_where_injected_{lab}"] += 1
                    stats[f"verifier_where_{lab}"] += 1
            if "verifier" in args.roles:
                label = labels.get(t["sample_id"])
                for x in t["turns"]:
                    if "letter_probs" not in x or label is None:
                        continue
                    p = float(x["letter_probs"].get(label, 0.0))
                    k = tier(p, args.cuts)
                    # target = the candidate POSITION of tier k (budget.verifier_neutral shuffles the options)
                    rows.append(dict(base, input=x["verifier_input"], target={"indices": [tier_position(x["verifier_input"], k)]},
                                     decision_type="verifier",
                                     source="label_pE", meta={"p_E": p, "tier": k, **x.get("stoppable_prefix", {})}))
                    stats["rows_verifier"] += 1
                    stats[f"verifier_tier_{k}"] += 1
    if args.balance_actions:
        ctrl = [r for r in rows if r["decision_type"] == "controller"]
        cnt = collections.Counter(r["input"]["candidate_ids"][r["target"]["indices"][0]] for r in ctrl)
        w = [min(4.0, max(0.25, len(ctrl) / (len(cnt) * cnt[r["input"]["candidate_ids"][r["target"]["indices"][0]]]))) for r in ctrl]
        m = sum(w) / max(len(w), 1)
        for r, x in zip(ctrl, w):
            r["weight"] = x / m
        stats["controller_action_counts"] = dict(cnt)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out, rows)
    write_json(str(args.out) + ".stats.json", dict(stats, args=vars(args)))
    print(json.dumps(dict(stats), indent=1))


if __name__ == "__main__":
    main()
