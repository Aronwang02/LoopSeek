"""Design-B (v3) rollouts: end-to-end evaluation, teacher trajectories for controller / verifier SFT, tau calibration.

Roles: --controller / --localizer / --selector / --verifier, each one of
  nanojev (Jev from --checkpoint: one checkpoint or a policy-set .json {controller, localize, frame, verifier}),
  teacher (privileged rule teacher, needs key_spans), uniform (selector only), random,
  fixed_inspect (controller: inspect the full video twice), fixed_li2 (controller: LOCALIZE_GLOBAL, inspect, inspect),
  locator_margin (controller rule on the locator probe; needs budget.locator_line, threshold budget.locrule_margin),
  evidence_expert (controller expert on key_spans; budget.evidence_pad, budget.evidence_refine_frac),
  evidence (localize / frame: execute the evidence label, src/baselines/evidence.py; needs key_spans),
  none (verifier only: never stop early). --verifier defaults to nanojev.
  --set budget.verifier_frame_oracle_pad=3   per-frame keep from key_spans +- 3 s instead of the verifier
--label-controller (DAgger) lets the --controller decider act and attaches the evidence expert's action as the label of
every controller decision (--dagger-beta: execute the expert's action with that probability instead).
--label-grounding attaches evidence labels (info.target_indices) to every localize / frame decision whatever decider made
it (grounding SFT rows, also on states reached by other deciders, e.g. --localizer random or a trained Jev).
Jev decisions are greedy unless --mode sample. The answerer answers once from the filled 32-frame set.
Outputs runs/<name>/: trajectories.jsonl, results.jsonl (one row per question), summary.json.
  --set budget.label_pE=true      record answerer letter probabilities at the verifier points (verifier SFT labels)
  --record-verifier-answers       record verifier probabilities and the answer stopping there would give, and
                                  continue (input of scripts/calibrate_tau.py)
usage:
  python scripts/run_agent_v3.py --name e2e_q25_vmme --config $C --manifest videomme_short --file eval.jsonl \
      --checkpoint runs/policy_sets/v3.json --verifier nanojev
  python scripts/run_agent_v3.py --name teach_v3_direct --config $C --manifest qaspan --file train.jsonl \
      --controller teacher --localizer teacher --selector teacher --set budget.label_pE=true
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.baselines.deciders import LocatorMarginController, ScriptedController  # noqa: E402
from src.baselines.registry import build_deciders  # noqa: E402
from src.common.config import load_config  # noqa: E402
from src.common.runtime import reserve_gpu_memory, gpu_mem_stats, setup_run  # noqa: E402
from src.common.utils import seed_everything, write_json, write_jsonl  # noqa: E402
from src.evaluation.evaluator import RewardEvaluator  # noqa: E402
from src.evaluation.metrics import accuracy_summary, latency_summary  # noqa: E402
from src.policy.model import apply_policy_sdpa_backend  # noqa: E402
from src.reader.qwen_reader import build_reader as QwenReader, apply_reader_backend  # noqa: E402
from src.rollout.build import AssetPool, episode_items, load_rows  # noqa: E402
from src.rollout.build_v3 import setup_v3  # noqa: E402
from src.rollout.runner_v3 import RunnerV3  # noqa: E402
from src.rollout.work_queue import Claims, completed_ids, home_names, next_batch, work_order  # noqa: E402

V3_SCRIPTS = {"fixed_inspect": ("INSPECT_FRAMES", "INSPECT_FRAMES"),
              "fixed_li2": ("LOCALIZE_GLOBAL", "INSPECT_FRAMES", "INSPECT_FRAMES"),
              # 2026-10-05 baseline for the locator_margin rule: inspect the full video, localize, then keep inspecting
              "fixed_ili": ("INSPECT_FRAMES", "LOCALIZE_GLOBAL", "INSPECT_FRAMES")}


def result_row(r, t, wall):
    sm = t["summary"]
    return {"sample_id": r["sample_id"], "group_key": r["group_key"], "subset": r.get("subset"),
            "status": t["status"], "correct": bool(t.get("evaluator", {}).get("correct", False)),
            "parsed": sm["answer"], "actions": [x["action"] for x in t["turns"]],
            "termination_reason": sm["termination_reason"], "searched": sm["searched"],
            "selected_times": sm["evidence_times"], "answer_times": sm.get("answer_times"),
            "n_answer_frames": len(sm.get("answer_times") or []), "verifier_log": t.get("verifier_log", []),
            "answer_source": (t.get("answer_meta") or {}).get("answer_source"),
            "cost": sm["cost"], "seconds": t["seconds"], "batch_wall_seconds": wall,
            **({"injection": t["injection"]} if t.get("injection") is not None else {})}


def _rows(path):
    """JSON lines of a results / trajectories file; a line cut short (a copy taken while a batch was being appended) is
    dropped, so its question runs again."""
    out = []
    for line in path.read_text().splitlines() if path.exists() else []:
        try:
            out.append(json.loads(line)) if line.strip() else None
        except ValueError:
            pass
    return out


def resume_completed(traj_path, res_path):
    """Keep completed result/trajectory pairs; retry infrastructure failures."""
    results, trajectories = _rows(res_path), _rows(traj_path)
    done = {row['sample_id'] for row in results if row.get('status') != 'infra_error'}
    done &= {row['sample_id'] for row in trajectories}
    write_jsonl(traj_path, [row for row in trajectories if row['sample_id'] in done])
    write_jsonl(res_path, [row for row in results if row['sample_id'] in done and row.get('status') != 'infra_error'])
    return done


def require_no_infrastructure_errors(results, run_dir):
    failed = [row['sample_id'] for row in results if row.get('status') == 'infra_error']
    if failed:
        write_json(run_dir / 'infra_failure.json', {'sample_ids': failed, 'count': len(failed)})
        raise SystemExit(f'{len(failed)} infrastructure failures; inspect trajectories.jsonl, fix the cause, '
                         'and rerun the same command to retry failed questions')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--controller", default="nanojev")
    ap.add_argument("--localizer", default="nanojev")
    ap.add_argument("--selector", default="nanojev")
    ap.add_argument("--verifier", default="nanojev")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--file", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--mode", default="greedy", choices=["greedy", "sample"])
    ap.add_argument("--teacher-mode", default="direct", choices=["direct", "recovery"])
    ap.add_argument("--batch-questions", type=int, default=8)
    ap.add_argument("--record-verifier-answers", action="store_true")
    ap.add_argument("--label-grounding", action="store_true")
    ap.add_argument("--label-controller", action="store_true",
                    help="DAgger: the --controller decider acts, the evidence expert labels every controller state "
                         "(info.target_indices); needs key_spans")
    ap.add_argument("--dagger-beta", type=float, default=0.0,
                    help="with --label-controller: execute the expert's action with this probability (DAgger mixing)")
    ap.add_argument("--inject-wrong", type=float, default=0.0,
                    help="collection only (2026-10-09, src/baselines/inject.py): at each locator call (root, SWITCH, child), "
                         "with this probability enter the locator's most probable window without evidence and inspect it "
                         "once before the controller acts again (verifier states in wrong windows); needs key_spans")
    ap.add_argument("--inject-max", type=int, default=2, help="with --inject-wrong: at most this many per episode")
    ap.add_argument("--inject-min-prob", type=float, default=0.05,
                    help="with --inject-wrong: inject only if the locator gives that window at least this probability "
                         "(errors the locator could make)")
    ap.add_argument("--shard", default=None, help="i/n: this process runs questions i, i+n, ... (parallel collection; "
                    "use one --name per shard and merge the run directories)")
    ap.add_argument("--queue", default=None, help="with --shard i/n and --name <base>_p<i>: node-local directory shared "
                    "by the n processes of one collection; after its own questions a process takes the unclaimed "
                    "questions of the other shards (src/rollout/work_queue.py)")
    ap.add_argument("--no-resume", action="store_true", help="start over instead of skipping questions already in "
                    "runs/<name>/results.jsonl")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    reserve_gpu_memory()                                # JEV_RESERVE_GB, shared machines only
    assert cfg["budget"].get("flow") == "v3", "use configs/v3_b32.yaml"
    apply_reader_backend(cfg)
    apply_policy_sdpa_backend(cfg)
    seed_everything(cfg["project"]["seed"])
    run_dir = setup_run(cfg, args.name)
    rows = [r for r in load_rows(cfg, manifest=args.manifest, file=args.file, limit=args.limit)
            if r.get("split") != "unavailable"]
    order, home, claims = rows, None, None       # with a queue, rows stay all questions (annotations, evaluator)
    if args.shard:
        si, sn = (int(x) for x in args.shard.split("/"))
        if args.queue:
            if not args.name.endswith(f"_p{si}"):
                raise SystemExit(f"--queue needs --name <base>_p{si} with --shard {args.shard}")
            shard_base = args.name[:-len(f"_p{si}")]     # not `base`: that name is the decider spec below
            order, home, claims = work_order(rows, si, sn), home_names(rows, shard_base, sn), Claims(args.queue)
        else:
            rows = order = rows[si::sn]
    spec = {"controller": args.controller, "localize": args.localizer, "frame": args.selector}
    if args.verifier != "none":
        spec["verifier"] = args.verifier
    jev_roles = [r for r, n in spec.items() if n == "nanojev"]
    enc, store, policies, text_len = setup_v3(cfg, args.checkpoint, jev_roles, shard=f"run_{args.name}")
    reader = QwenReader(cfg, device="cuda:0")
    # evidence_multi_segment (filter --multi-segment, the VideoITG experiment of 2026-10-20): the evidence expert and the
    # where oracle follow the row's uncovered segments (src/baselines/evidence.uncovered_spans)
    ann = {r["sample_id"]: {"key_spans": r["key_spans"],
                            **({"evidence_multi_segment": True} if r.get("evidence_multi_segment") else {})}
           for r in rows if r.get("key_spans")}
    base = {k: v for k, v in spec.items()
            if v not in V3_SCRIPTS and v not in ("locator_margin", "evidence_expert", "evidence")}
    deciders = build_deciders(base, policies or None, reader, cfg, args.mode, annotations=ann,
                              teacher_mode=args.teacher_mode)
    tree = bool(cfg["budget"].get("window_tree"))
    evidence_pad = float(cfg["budget"]["evidence_pad"])
    if tree and ("locator_margin" in spec.values() or "unitime" in spec.values()):
        raise SystemExit("locator_margin / unitime assume the flat flow (LOCALIZE_GLOBAL, full/active windows); "
                         "run them with --set budget.window_tree=false and the flat actions")
    for role, name in spec.items():
        if name in V3_SCRIPTS:
            assert role == "controller"
            script = V3_SCRIPTS[name]
            if tree:   # window tree: no LOCALIZE_GLOBAL; the scripted "localize" step is REFINE_LOCAL (from the root it
                script = tuple("REFINE_LOCAL" if a == "LOCALIZE_GLOBAL" else a for a in script)   # picks a quarter)
                print(json.dumps({"scripted_controller": name, "window_tree_script": script}), flush=True)
            deciders[role] = ScriptedController(script)
        elif name == "evidence_expert":  # 2026-10-06 expert on the evidence annotation (needs key_spans in the manifest)
            from src.baselines.teacher import EvidenceExpertController
            assert role == "controller" and ann, "evidence_expert needs key_spans in the manifest rows"
            bb = cfg["budget"]
            deciders[role] = EvidenceExpertController(ann, evidence_pad,
                                                      bb.get("expert_refine_frac", bb.get("evidence_refine_frac", 0.25)))
        elif name == "evidence":         # 2026-10-06 grounding on the evidence annotation (localize / frame)
            from src.baselines.evidence import EvidenceGrounding
            assert role in ("localize", "frame") and ann, "evidence grounding needs key_spans in the manifest rows"
            deciders[role] = EvidenceGrounding(role, ann, evidence_pad,
                                               seed=cfg["project"]["seed"])
        elif name == "locator_margin":   # 2026-10-05 rule from the randomized-controller analysis; needs locator_line
            assert role == "controller" and cfg["budget"].get("locator_line"), "locator_margin needs budget.locator_line"
            deciders[role] = LocatorMarginController(cfg["budget"].get("locrule_margin", 0.3))
    if args.label_controller:
        from src.baselines.teacher import EvidenceExpertController, ExpertLabeledController
        assert ann, "--label-controller needs key_spans in the manifest rows"
        bb = cfg["budget"]
        expert = EvidenceExpertController(ann, evidence_pad,
                                          bb.get("expert_refine_frac", bb.get("evidence_refine_frac", 0.25)))
        deciders["controller"] = ExpertLabeledController(deciders["controller"], expert, args.dagger_beta,
                                                         seed=cfg["project"]["seed"])
    if args.label_grounding:
        from src.baselines.evidence import EvidenceGrounding
        assert ann, "--label-grounding needs key_spans in the manifest rows"
        for role in ("localize", "frame"):
            if not isinstance(deciders[role], EvidenceGrounding):
                deciders[role] = EvidenceGrounding(role, ann, evidence_pad,
                                                   inner=deciders[role])
    if args.inject_wrong > 0:
        from src.baselines.inject import InjectInspectController, InjectWrongLocalizer
        assert ann and tree, "--inject-wrong needs key_spans in the manifest rows and budget.window_tree"
        deciders["localize"] = InjectWrongLocalizer(deciders["localize"], ann, evidence_pad,
                                                    args.inject_wrong, args.inject_min_prob, args.inject_max,
                                                    seed=cfg["project"]["seed"])
        deciders["controller"] = InjectInspectController(deciders["controller"])
    runner = RunnerV3(cfg, deciders, reader, store, enc, text_len=text_len,
                      record_verifier_answers=args.record_verifier_answers, annotations=ann)
    evaluator = RewardEvaluator(rows)
    assets = AssetPool(cfg)
    # 2026-10-06 review: results are appended after every batch and a rerun skips the questions already done (a crash
    # loses at most one batch); trajectories are written before results, so a question counts as done only when both are
    traj_path, res_path = run_dir / "trajectories.jsonl", run_dir / "results.jsonl"
    if args.no_resume:
        for f in (traj_path, res_path):
            f.unlink(missing_ok=True)
    done = resume_completed(traj_path, res_path)
    if claims is not None:                        # what the other shards' directories already hold
        for j in range(sn):
            if j != si:
                done |= completed_ids(run_dir.parent / f"{shard_base}_p{j}")
    own = [r for r in order if home is None or home[r["sample_id"]] == args.name]
    n_done, taken, pos = sum(r["sample_id"] in done for r in own), 0, 0
    print(json.dumps({"questions": len(own), "already_done": n_done,
                      **({"queue": args.queue, "done_in_all_shards": len(done)} if claims else {})}), flush=True)
    while True:
        chunk, pos = next_batch(order, pos, done, claims, args.batch_questions)
        if not chunk:
            break
        t0 = time.perf_counter()
        trajs = runner.run([it for r in chunk for it in episode_items(
            [r], assets, 1, cfg["project"]["seed"], home[r["sample_id"]] if home else args.name)])
        wall = (time.perf_counter() - t0) / len(chunk)
        res_rows = []
        for r, t in zip(chunk, trajs):
            evaluator.score(t)
            res_rows.append(result_row(r, t, wall))
        with open(traj_path, "a") as f:
            f.writelines(json.dumps(t) + "\n" for t in trajs)
        with open(res_path, "a") as f:
            f.writelines(json.dumps(x) + "\n" for x in res_rows)
        mine = [row for row in res_rows if home is None or home[row["sample_id"]] == args.name]
        n_done += sum(row['status'] != 'infra_error' for row in mine)
        taken += len(res_rows) - len(mine)
        print(json.dumps({"done": n_done, "of": len(own), **({"taken_from_other_shards": taken} if claims else {})}),
              flush=True)
        require_no_infrastructure_errors(res_rows, run_dir)
    ids = {r["sample_id"] for r in rows}
    results = [x for x in (json.loads(l) for l in open(res_path)) if x["sample_id"] in ids]
    ok = [x for x in results if x["status"] == "ok"]
    acts = [a for x in ok for a in x["actions"] if a != "VERIFIER_POINT"]
    summ = accuracy_summary(results)
    summ.update(spec=spec, checkpoint=args.checkpoint, mode=args.mode, manifest=args.manifest, file=args.file,
                n=len(results), failed=len(results) - len(ok),
                termination={k: sum(x["termination_reason"] == k for x in ok) for k in
                             sorted({x["termination_reason"] for x in ok})},
                action_freq={a: acts.count(a) / max(len(acts), 1) for a in sorted(set(acts))},
                inspections_mean=sum(len([a for a in x["actions"] if a == "INSPECT_FRAMES"]) for x in ok) / max(len(ok), 1),
                answer_frames_mean=sum(x["n_answer_frames"] for x in ok) / max(len(ok), 1),
                answer_fallback=sum(x["answer_source"] == "letter_logits_fallback" for x in ok),
                seconds_policy=latency_summary([x["seconds"]["policy"] for x in ok]),
                seconds_reader=latency_summary([x["seconds"]["reader"] for x in ok]),
                seconds_env=latency_summary([x["seconds"]["env"] for x in ok]),
                seconds_wall_per_question=latency_summary([x["batch_wall_seconds"] for x in ok]),
                encode_seconds=enc.encode_seconds, **gpu_mem_stats())
    write_json(run_dir / "summary.json", summ)
    (run_dir / 'infra_failure.json').unlink(missing_ok=True)
    from src.common.runtime import wandb_summary          # one W&B run per collection / evaluation (summary scalars)
    wandb_summary(args.name, "rollout", run_dir, {k: v for k, v in summ.items() if k not in ("spec",)}, cfg)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
