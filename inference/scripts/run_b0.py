"""B0: uniformly sampled frames + frozen answerer (draft §10.2). Same answer prompt, frame format,
decoding and parsing as every agent system; frames = the answerer's retained-frame cap (32). Results are appended per batch to b0_results.partial.jsonl and
a rerun with the same --name skips the questions already done (long filtering runs are resumable)."""
import argparse
import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.config import load_config  # noqa: E402
from src.common.runtime import gpu_mem_stats, init_wandb, reserve_gpu_memory, setup_run  # noqa: E402
from src.common.utils import append_jsonl, read_jsonl, seed_everything, write_json, write_jsonl  # noqa: E402
from src.evaluation.metrics import accuracy_summary  # noqa: E402
from src.evaluation.scoring import is_correct  # noqa: E402
from src.reader.qwen_reader import build_reader as QwenReader  # noqa: E402
from src.rollout.build import AssetPool, load_rows  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", default=[str(ROOT / "configs/default.yaml")])
    ap.add_argument("--name", required=True)
    ap.add_argument("--manifest", default="videor1")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--file", default="manifest.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--frames", type=int, default=None, help="default: budget.reader_retained_frames")
    ap.add_argument("--shard", default=None, help="i/n: evaluate questions i, i+n, ...; use a separate --name per shard")
    ap.add_argument("--frames-from", default=None,
                    help="trajectories.jsonl of an agent run: answer each question from its saved summary.answer_times "
                         "instead of uniform frames (2026-10-07: re-answer another run's frames with a different reader)")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    seed_everything(cfg["project"]["seed"])
    run_dir = setup_run(cfg, args.name)
    wb = init_wandb(cfg, run_dir, args.name, "eval_b0")
    n_frames = cfg["budget"]["reader_retained_frames"] if args.frames is None else args.frames
    rows = load_rows(cfg, manifest=args.manifest, split=args.split if args.file == "manifest.jsonl" else None,
                     file=args.file, limit=args.limit)
    n_unavail = sum(r.get("split") == "unavailable" for r in rows)
    rows = [r for r in rows if r.get("split") != "unavailable"]      # e.g. MVBench NTU clips (no video)
    if args.shard:
        si, sn = (int(x) for x in args.shard.split("/"))
        if not 0 <= si < sn:
            ap.error("--shard must satisfy 0 <= i < n")
        rows = rows[si::sn]
    evaluated_splits = sorted({r["split"] for r in rows if r.get("split")})
    eval_split = (args.split if args.file == "manifest.jsonl" else
                  evaluated_splits[0] if len(evaluated_splits) == 1 else None)
    saved = None
    if args.frames_from:
        saved = {t["sample_id"]: t["summary"]["answer_times"] for t in read_jsonl(args.frames_from)
                 if t.get("status", "ok") == "ok" and t["summary"].get("answer_times")}
        missing = [r["sample_id"] for r in rows if r["sample_id"] not in saved]
        assert not missing, f"{len(missing)} questions have no saved answer_times, e.g. {missing[:3]}"
    if n_unavail:
        print(json.dumps({"skipped_unavailable": n_unavail, "evaluated": len(rows)}), flush=True)
    reserve_gpu_memory()                                # JEV_RESERVE_GB, shared machines only
    reader = QwenReader(cfg, device="cuda")
    assets = AssetPool(cfg, max_assets=16)
    part = run_dir / "b0_results.partial.jsonl"
    out = read_jsonl(part) if part.exists() else []
    done = {x["sample_id"] for x in out}
    rows = [r for r in rows if r["sample_id"] not in done]
    if done:
        print(json.dumps({"resumed": len(done), "remaining": len(rows)}), flush=True)
    bs = cfg["reader"]["batch_size"]
    if cfg["reader"].get("engine") == "vllm":
        # Prepare a bounded queue with spare requests for slots freed by short answers.
        # The engine's max_num_seqs, not this CPU submission size, caps GPU concurrency.
        bs = 2 * int(cfg["reader"].get("vllm", {}).get("max_num_seqs") or bs)
    for s in range(0, len(rows), bs):
        chunk = rows[s:s + bs]
        t0 = time.perf_counter()
        jobs = []
        for r in chunk:
            a = assets.get(r["video_path"])
            if saved is not None:           # the other run's answer frames: times -> the frames shown at those times
                idx = a.indices_at(saved[r["sample_id"]])
                assert len(idx) == len(saved[r["sample_id"]]), (r["sample_id"], "duplicate frames after time->index")
            else:
                idx = a.uniform_indices(0.0, a.duration, n_frames)
            fr = a.frames(idx)
            jobs.append({"kind": "answer", "frames": [fr[i] for i in idx], "times": [a.time(i) for i in idx],
                         "question": r["question"], "options": r["options"], "duration": a.duration,
                         "source": "", "interval": None, "notes": []})
        decode_s = time.perf_counter() - t0
        res = reader.run(jobs)
        new = []
        for r, j, o in zip(chunk, jobs, res):
            new.append({"sample_id": r["sample_id"], "video_id": r["video_id"], "group_key": r["group_key"],
                        "subset": r["subset"], "frame_times": j["times"], "answer_raw": o["text"],
                        "parsed": o["parsed"], "correct": bool(is_correct(o["parsed"], r["answer"])),
                        "label": r["answer"], "reader_meta": o["meta"],
                        "decode_seconds": decode_s / len(chunk)})
        append_jsonl(part, new)
        out += new
        print(json.dumps({"done": len(out), "acc_so_far": sum(x["correct"] for x in out) / len(out)}), flush=True)
    write_jsonl(run_dir / "b0_results.jsonl", out)
    summ = accuracy_summary(out)
    summ.update(system="B0" if saved is None else "saved_frames", frames=n_frames, frames_from=args.frames_from,
                split=eval_split, manifest=args.manifest, file=args.file, **gpu_mem_stats())
    write_json(run_dir / "summary.json", summ)
    if wb:
        wb.log({f"b0/{k}": v for k, v in summ.items() if isinstance(v, (int, float))})
        wb.finish()
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
