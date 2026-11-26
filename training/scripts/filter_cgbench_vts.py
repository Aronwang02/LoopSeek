"""VTS's four-stage question filter (arXiv 2607.16189, Appendix B / C.2), applied to a manifest from build_cgbench.py
(or any manifest with key_spans, e.g. LV-Haystack with --merge-spans). The answerer is the configured reader
(Qwen3-VL-8B, greedy; VTS: Qwen3-VL-8B-Instruct, temperature 0). Stages run in order; only survivors go on.
The clue = the key spans, merged when they overlap or are at most --merge-gap apart (6 s = twice the 3 s evidence pad,
the gap the training labels' padded windows close too; CG-Bench often annotates one event in pieces
1-2 s apart, and stage 4 would otherwise sample frames inside it). As in VTS, a clue that is still several segments
after that merge drops the question (VTS_VERSION 4; versions 2 / 3 kept them, reverted 2026-10-20 after the paired
Video-MME comparison). --multi-segment keeps them (the VideoITG experiment): stage 1 then caps the
union of the segments, stages 3 / 4 sample inside / outside every segment, and every written row is marked
evidence_multi_segment, so the evidence expert and the where / enclosing labels follow its uncovered segments
(src/baselines/evidence.uncovered_spans); unmarked rows keep the single-segment labels.
  1 rule         drop: more than one clue segment (without --multi-segment); the clue (union) > --max-cover (30%, as
                 VTS) of the video; non-positive duration or no clue
  2 text-only    the answerer sees only the question and the options (no frames); drop if it answers correctly
  3 clue-only    frames at --fps (1) inside every clue segment, at most --max-frames (256) evenly; drop if it answers
                 WRONG
  4 outside-clue frames at --fps outside every clue segment (all gaps, by length), at most --max-frames; drop if it
                 answers correctly
"Answers correctly" = the predicted letter matches the ground-truth letter, as in VTS (Appendix A). Since 2026-10-19
(VTS_VERSION 3) an answer without a legal letter matches nothing: the filter's answerer runs without the letter-logit
fallback of the final answers (reader.answer_fallback), so a model that cannot tell keeps the question at stages 2 / 4
and drops it at stage 3, instead of a forced guess deciding (logged with unparsed: true, counted per stage).
--tag-only (dev splits): nothing is dropped except rows that cannot be run (stage 1 "invalid"); every
other row is written with row["vts"] = {version, kept, dropped_at, reason}. Model selection then sees the unfiltered
distribution, and the subset the filter would keep can still be scored on its own.
Per-question results go to <out>.log.jsonl (one line per question and stage: stage, kept, predicted); a rerun with the
same --out skips questions already decided and stages already passed (resume). Kept rows -> --out; counts ->
<out>.stats.json. --shard i/n (one process per GPU, train_pipeline.sh) takes every n-th question from i and logs to
<out>.log.shard<i>of<n>.jsonl without writing --out; a run without --shard then merges every log into --out.
--cache DIR (VTS_CACHE): the kept rows depend only on the input rows, the stage settings, the answerer and the frame
decoding, not on training, the TAG or the agent's frame budget. A complete unsharded run stores its result under
DIR/<key>/; a later run with the same key copies it to --out instead of filtering (--cache-restore: only that, exit 3 on
a miss). Bump VTS_VERSION when the stages, the answer prompt or its parsing change which questions survive.
VTS reports 3,000 -> 1,176 questions on CG-Bench mini; the training pool (full minus mini) uses the same filter.
usage (GPU):
  python scripts/filter_cgbench_vts.py --config configs/default.yaml --manifest data/manifests/longvid/cgbench_all.jsonl \
      --split test --out data/manifests/longvid/cgbench_mini_vts.jsonl --set reader.batch_size=1
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VTS_VERSION = 4   # 1: one clue segment (touching spans merged); 2: multi-segment clues kept; 3: + no letter-logit
                  # fallback, segments at most --merge-gap (6 s) apart merged; 4: as 3, but a clue still several segments
                  # after the merge drops the question again, unless --multi-segment (logged with "multi": true)
MERGE_GAP = 6.0   # seconds: clue segments at most this far apart are one segment (2 x budget.evidence_pad)
_EXECUTION_ONLY = {"device", "batch_size", "vision_cache_mb", "vllm"}   # reader settings that never change an answer


def merged(spans, gap=0.0):
    out = []
    for s, e in sorted((float(s), float(e)) for s, e in spans):
        if out and s <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def rule_check(row, max_cover=0.30, merge_gap=MERGE_GAP, multi_segment=False):
    """Stage 1. Returns (kept, reason). Several clue segments only with multi_segment; the cover is their union."""
    T = float(row.get("duration") or 0.0)
    spans = merged(row.get("key_spans") or [], merge_gap)
    if T <= 0 or not spans:
        return False, "invalid"
    if len(spans) > 1 and not multi_segment:
        return False, "multi_segment"
    if sum(max(0.0, min(e, T) - max(s, 0.0)) for s, e in spans) > max_cover * T:
        return False, "high_coverage"
    return True, "ok"


def _even(times, cap):
    if len(times) <= cap:
        return times
    return [times[int((j + 0.5) * len(times) / cap)] for j in range(cap)]


def clue_times(spans, fps=1.0, cap=256):
    """Stage 3: times at `fps` inside every clue segment [s, e] (bin midpoints, at least one per segment), at most
    `cap` evenly spread over all of them. spans: merged, time-ordered."""
    times = []
    for s, e in spans:
        n = max(1, int(math.ceil((e - s) * fps)))
        times += [s + (j + 0.5) * (e - s) / n for j in range(n)]
    return _even(times, cap)


def outside_times(T, spans, fps=1.0, cap=256):
    """Stage 4: times at `fps` in the gaps outside every clue segment ([0, s1), (e1, s2), ..., (en, T]), at most `cap`
    in total, shared by length. spans: merged, time-ordered."""
    edges = [0.0] + [min(max(x, 0.0), T) for span in spans for x in span] + [T]
    times = []
    for a, b in zip(edges[::2], edges[1::2]):
        n = int(math.floor((b - a) * fps))
        times += [a + (j + 0.5) * (b - a) / n for j in range(n)] if n > 0 else []
    return _even(times, cap)


def stage_jobs(chunk, stage, assets, fps, max_frames, merge_gap=MERGE_GAP):
    """Answerer jobs of one batch of questions: stage 2 without frames, 3 inside the clue, 4 outside it."""
    jobs = []
    for r in chunk:
        spans = merged(r["key_spans"], merge_gap)
        if stage == "2":
            frames, times = [], []
        else:
            a = assets.get(r["video_path"])
            T = float(r["duration"])
            want = clue_times(spans, fps, max_frames) if stage == "3" else outside_times(T, spans, fps, max_frames)
            idx = a.indices_at(want)
            got = a.frames(idx)
            frames, times = [got[i] for i in idx], [a.time(i) for i in idx]
        jobs.append({"kind": "answer", "frames": frames, "times": times, "question": r["question"],
                     "options": r["options"], "duration": float(r["duration"]), "source": "filter",
                     "interval": None, "notes": []})
    return jobs


def filter_rows(todo, stages, reader, assets, fps, max_frames, batch_questions, record, decided, passed=None,
                merge_gap=MERGE_GAP):
    """Stages 2-4 on the rows that passed stage 1; reader.run(jobs) -> [{"parsed": letter}], assets.get(path) -> a
    VideoAsset. Drops go to decided[sample_id] = False; returns the survivors. passed: sample_id -> stages an earlier
    run already kept (not asked again). One background thread decodes the next batch's frames while the answerer works
    on the current batch (2026-10-11: up to 256 frames per question were decoded with the GPU idle)."""
    from concurrent.futures import ThreadPoolExecutor
    passed = passed or {}
    for stage, keep_if_correct in (("2", False), ("3", True), ("4", False)):
        if stage not in stages or not todo:
            continue
        ask = [r for r in todo if stage not in passed.get(r["sample_id"], ())]
        keep = {r["sample_id"] for r in todo} - {r["sample_id"] for r in ask}
        chunks = [ask[b:b + batch_questions] for b in range(0, len(ask), batch_questions)]
        with ThreadPoolExecutor(max_workers=1, thread_name_prefix="vts-prefetch") as prefetch:
            ahead = (prefetch.submit(stage_jobs, chunks[0], stage, assets, fps, max_frames, merge_gap) if chunks
                     else None)
            for n, chunk in enumerate(chunks):
                jobs = ahead.result()
                if n + 1 < len(chunks):
                    ahead = prefetch.submit(stage_jobs, chunks[n + 1], stage, assets, fps, max_frames, merge_gap)
                outs = reader.run(jobs)
                for r, o, j in zip(chunk, outs, jobs):
                    if o is None:             # answerer failure: no verdict, retried on the next run
                        record(r, stage, None, predicted=None, n_frames=len(j["frames"]), error="no_answer")
                        continue
                    pred = o.get("parsed")    # None: no legal letter in the answer, which matches nothing (v3)
                    correct = pred is not None and pred == r["answer"]
                    kept = correct if keep_if_correct else not correct
                    record(r, stage, kept, predicted=pred, n_frames=len(j["frames"]),
                           **({"unparsed": True} if pred is None else {}))
                    (keep.add(r["sample_id"]) if kept else decided.__setitem__(r["sample_id"], False))
                print(json.dumps({"stage": stage, "done": min((n + 1) * batch_questions, len(ask)), "of": len(ask)}),
                      flush=True)
        todo = [r for r in todo if r["sample_id"] in keep]
    return todo


def cache_key(rows, args, cfg):
    """(key, spec) of a filter result: input rows, stage settings, answerer (model revision, prompt settings, engine)
    and frame decoding. Batch sizes, GPU memory shares and the agent's frame budget are left out."""
    rc = {**cfg["reader"], "answer_fallback": "none"}       # what the filter's answerer runs with (set in main)
    spec = {"version": VTS_VERSION, "stages": args.stages, "fps": args.fps, "max_frames": args.max_frames,
            "max_cover": args.max_cover, "merge_spans": args.merge_spans, "limit": args.limit,
            "tag_only": bool(getattr(args, "tag_only", False)), "merge_gap": getattr(args, "merge_gap", MERGE_GAP),
            "multi_segment": bool(getattr(args, "multi_segment", False)),
            "rows": hashlib.sha256("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode()).hexdigest(),
            "reader": {k: v for k, v in rc.items() if k not in _EXECUTION_ONLY},
            "filter_max_model_len": (rc.get("vllm") or {}).get("filter_max_model_len") if rc.get("engine") == "vllm" else None,
            "model": {k: v for k, v in (cfg["models"].get(rc["model"]) or {}).items() if k != "path"},
            "decode_max_side": cfg["video"]["decode_max_side"]}
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:24], spec


def _copy(src, dst):
    tmp = Path(f"{dst}.{os.getpid()}.tmp")
    tmp.write_bytes(Path(src).read_bytes())
    os.replace(tmp, dst)


def cache_restore(entry, out):
    """Copy a stored result to --out (+ .stats.json). key.json is written last, so its presence marks a complete entry."""
    if not all((entry / f).is_file() for f in ("key.json", "rows.jsonl", "stats.json")):
        return False
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    _copy(entry / "rows.jsonl", out)
    stats = {**json.loads((entry / "stats.json").read_text()), "cache": str(entry)}
    Path(f"{out}.stats.json").write_text(json.dumps(stats, indent=1))
    return True


def cache_store(entry, out, spec):
    if (entry / "key.json").is_file():
        return
    entry.mkdir(parents=True, exist_ok=True)
    _copy(out, entry / "rows.jsonl")
    _copy(f"{out}.stats.json", entry / "stats.json")
    tmp = entry / f"key.json.{os.getpid()}.tmp"
    tmp.write_text(json.dumps({**spec, "out": str(out)}, indent=1))
    os.replace(tmp, entry / "key.json")


def log_paths(out):
    """The per-question logs of --out: the unsharded one, then the --shard ones."""
    base = Path(f"{out}.log.jsonl")
    return [base] + sorted(base.parent.glob(f"{Path(out).name}.log.shard*.jsonl"))


def same_mode(x, multi_segment=False):
    """A log line of this filter's VTS_VERSION (no "v" = 1) and --multi-segment setting (no "multi" = off)."""
    return x.get("v", 1) == VTS_VERSION and bool(x.get("multi", False)) == bool(multi_segment)


def read_logs(paths, last_stage, multi_segment=False, stages=None):
    """(decided, passed): final verdicts, and the stages each undecided question already passed. Answerer failures
    (kept None), a line cut off by a killed process and lines of another version or --multi-segment setting are
    skipped; so are lines of stages not in `stages` (2026-10-21: --stages 123 over logs that also hold stage 4, whose
    drops must not decide the question)."""
    decided, passed = {}, {}
    for path in paths:
        if not path.exists():
            continue
        for line in open(path):
            try:
                x = json.loads(line)
            except ValueError:
                continue
            if x["kept"] is None or not same_mode(x, multi_segment) or (stages and x["stage"] not in stages):
                continue
            if not x["kept"] or x["stage"] == last_stage:
                decided[x["sample_id"]] = x["kept"]
            else:
                passed.setdefault(x["sample_id"], set()).add(x["stage"])
    return decided, passed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", default=[str(ROOT / "configs/default.yaml")])
    ap.add_argument("--manifest", required=True, help="manifest jsonl (absolute or relative path)")
    ap.add_argument("--split", default=None, help="only rows with this split (e.g. test = CG-Bench mini, train, dev)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stages", default="1234")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--max-frames", type=int, default=256)
    ap.add_argument("--max-cover", type=float, default=0.30)
    ap.add_argument("--merge-gap", type=float, default=MERGE_GAP, help="clue segments at most this many seconds apart "
                    "are one segment (train_pipeline.sh: 2 x budget.evidence_pad)")
    ap.add_argument("--merge-spans", action="store_true",
                    help="replace key_spans by one span [min, max] first (as VTS does for LV-Haystack / Ego4D; "
                         "train_pipeline.sh: LV-Haystack, and VideoITG unless VIDEOITG_MULTI_SEGMENT=1)")
    ap.add_argument("--multi-segment", action="store_true",
                    help="keep clues that are several segments after --merge-gap (the VideoITG experiment, 2026-10-20) "
                         "and mark the written rows evidence_multi_segment")
    ap.add_argument("--batch-questions", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", default=None, help="i/n: every n-th question from i, own log, no --out (merge after)")
    ap.add_argument("--log-tag", default=None, help="with --shard: <out>.log.shard<i>of<n>.<tag>.jsonl (one tag per job, so jobs sharing a split "
                                                    "never write the same log)")
    ap.add_argument("--shard-by", choices=("row", "video"), default="row",
                    help="with --shard: every n-th question (row), or every n-th video in path order with all its "
                         "questions (video: a job then needs its own videos only)")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--cache", default=None, help="shared result cache (VTS_CACHE); ignored with --shard")
    ap.add_argument("--cache-restore", action="store_true", help="with --cache: restore only; exit 3 on a miss")
    ap.add_argument("--tag-only", action="store_true",
                    help="dev: write every row that can be run, each with the verdict it would get "
                         "in row['vts'] (kept, dropped_at, reason), instead of dropping the rejected ones")
    args = ap.parse_args()
    if args.merge_spans and args.multi_segment:
        ap.error("--merge-spans leaves one segment; --multi-segment keeps several")
    rows = [json.loads(l) for l in open(args.manifest)]
    if args.split:
        rows = [r for r in rows if r.get("split") == args.split]
    if args.merge_spans:
        for r in rows:
            if r.get("key_spans"):
                r["key_spans"] = [[min(s for s, _ in r["key_spans"]), max(e for _, e in r["key_spans"])]]
    rows = rows[:args.limit] if args.limit else rows
    entry = spec = None
    if args.cache and not args.shard:
        from src.common.config import load_config
        key, spec = cache_key(rows, args, load_config(args.config, args.set))
        entry = Path(args.cache) / key
        if cache_restore(entry, args.out):
            print(json.dumps({"vts_cache": "hit", "entry": str(entry), "out": args.out}))
            return
        if args.cache_restore:
            print(json.dumps({"vts_cache": "miss", "entry": str(entry)}))
            raise SystemExit(3)
    decided, passed = read_logs(log_paths(args.out), args.stages[-1], args.multi_segment, args.stages)   # resume
    shard = tuple(int(x) for x in args.shard.split("/")) if args.shard else None
    tag = f".{args.log_tag}" if shard and args.log_tag else ""
    log_path = Path(f"{args.out}.log.shard{shard[0]}of{shard[1]}{tag}.jsonl" if shard else f"{args.out}.log.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = open(log_path, "a")
    synced = [time.monotonic()]

    def record(r, stage, kept, **kw):
        log.write(json.dumps({"sample_id": r["sample_id"], "stage": stage, "kept": kept, "v": VTS_VERSION,
                              **({"multi": True} if args.multi_segment else {}), **kw}) + "\n")
        log.flush()
        if time.monotonic() - synced[0] > 60:     # a network mount may upload on close / fsync only (lost on a kill otherwise)
            os.fsync(log.fileno())
            synced[0] = time.monotonic()

    if shard and args.shard_by == "video":
        nth = {v: i for i, v in enumerate(sorted({r.get("video_path") or r["sample_id"] for r in rows}))}
        position = [nth[r.get("video_path") or r["sample_id"]] for r in rows]
    else:
        position = list(range(len(rows)))
    todo = [r for k, r in enumerate(rows)
            if r["sample_id"] not in decided and (not shard or position[k] % shard[1] == shard[0])]
    if "1" in args.stages:
        nxt = []
        for r in todo:
            if "1" in passed.get(r["sample_id"], ()):
                nxt.append(r)
                continue
            ok, why = rule_check(r, args.max_cover, args.merge_gap, args.multi_segment)
            record(r, "1", ok, reason=why)
            (nxt.append(r) if ok else decided.__setitem__(r["sample_id"], False))
        todo = nxt
    reader = assets = None
    if any(s in args.stages for s in "234") and todo:
        from src.common.config import load_config
        from src.reader.qwen_reader import build_reader as QwenReader
        from src.rollout.build import AssetPool
        cfg = load_config(args.config, args.set)
        cfg['reader']['answer_fallback'] = 'none'           # v3: an answer without a legal letter is not correct
        if cfg['reader'].get('engine') == 'vllm':
            vc = cfg['reader']['vllm']
            vc['max_model_len'] = vc['filter_max_model_len']
            vc['gpu_memory_utilization'] = vc['filter_gpu_memory_utilization']
            vc['max_num_seqs'] = int(cfg['reader']['batch_size'])  # separate 256-frame VTS limit
        reader, assets = QwenReader(cfg, device="cuda:0"), AssetPool(cfg)
    todo = filter_rows(todo, args.stages, reader, assets, args.fps, args.max_frames, args.batch_questions, record, decided,
                       passed, args.merge_gap)
    for r in todo:
        decided[r["sample_id"]] = True
    if shard:
        print(json.dumps({"shard": args.shard, "kept": len(todo)}))
        return
    kept_rows = [r for r in rows if decided.get(r["sample_id"])]
    pending = sum(r["sample_id"] not in decided for r in rows)
    if pending:
        print(json.dumps({"WARNING": f"{pending} questions have no verdict (answerer failures); rerun to retry them"}))
    per_stage, dropped_at = {}, {}                    # dropped_at: sample_id -> (stage, stage-1 reason) of the drop
    for path in log_paths(args.out):
        for line in (open(path) if path.exists() else ()):
            try:
                x = json.loads(line)
            except ValueError:
                continue
            if not same_mode(x, args.multi_segment) or x["stage"] not in args.stages:
                continue
            k = f"stage{x['stage']}_" + ("no_answer" if x["kept"] is None else "kept" if x["kept"] else "dropped")
            per_stage[k] = per_stage.get(k, 0) + 1
            if x.get("unparsed"):                     # v3: answers without a legal letter, part of kept / dropped
                per_stage[f"stage{x['stage']}_unparsed"] = per_stage.get(f"stage{x['stage']}_unparsed", 0) + 1
            if x["kept"] is False:
                dropped_at[x["sample_id"]] = (x["stage"], x.get("reason"))
    out_rows = kept_rows
    if args.tag_only:     # every row that can be run (not invalid), with the verdict it would get, none dropped
        out_rows = []
        for r in rows:
            stage, reason = dropped_at.get(r["sample_id"], (None, None))
            if reason != "invalid":
                out_rows.append({**r, "vts": {"version": VTS_VERSION, "kept": decided.get(r["sample_id"]),
                                              "dropped_at": stage, "reason": reason}})
    if args.multi_segment:    # the evidence expert and the where / enclosing labels follow these rows' uncovered segments
        out_rows = [{**r, "evidence_multi_segment": True} for r in out_rows]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    stats = {"input": len(rows), "kept": len(kept_rows), "written": len(out_rows), "tag_only": args.tag_only,
             "per_stage": per_stage, "args": vars(args)}
    with open(str(args.out) + ".stats.json", "w") as f:
        json.dump(stats, f, indent=1)
    if entry is not None and not pending:             # complete results only; answerer failures are retried first
        cache_store(entry, args.out, spec)
        print(json.dumps({"vts_cache": "stored", "entry": str(entry)}))
    try:                                                  # W&B summary of the filter (counts per stage)
        from src.common.config import load_config
        from src.common.runtime import wandb_summary
        wandb_summary(Path(args.out).stem, "filter", Path(args.out).parent, {k: v for k, v in stats.items() if k != "args"},
                      load_config(args.config, args.set))
    except Exception as exc:
        print(json.dumps({"wandb_summary_failed": repr(exc)[:200]}))
    print(json.dumps({k: v for k, v in stats.items() if k != "args"}))


if __name__ == "__main__":
    main()
