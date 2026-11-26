"""VideoITG-40K multiple-choice questions -> manifests of the short-video profile (the short-video SFT
data is VideoITG only, filtered with the long-video rules, no minimum video length). CPU; videos fetched first
(scripts/fetch_videoitg.py).

VideoITG-40K (NVEagle/VideoITG-40K, Apache-2.0): LLaVA-Video-178K videos of 30 s - 3 min; per question the seconds
(1 fps frame numbers) a model-based pipeline (VidThinker) marked as evidence. Only multiple-choice questions are used:
options "A. ..." in the question text, the answer's leading letter.
Row: sample_id videoitg:<id>, video_id = the LLaVA-Video path, group_key videoitg:<path> (one video = one group),
question = the stem (option lines and the answer instruction removed), options {A: text}, answer, key_spans = each
marked second k as [k - 0.5, k + 0.5] inside the video, touching ones merged (scripts/filter_cgbench_vts.py merges
segments at most 6 s apart; the labels pad by budget.evidence_pad), evidence_times = the marked seconds, duration = the
file's timeline (src/video/decode.read_timeline, as the agent reads it).
Dropped and counted: not multiple-choice or unparseable options / answer, video missing or unreadable, no marked second
inside the video. Split: --dev-videos videos (seeded) are dev, the rest train. No deduplication.
usage: python scripts/build_videoitg.py --annotations <video_itg_data.json> --video-root <dir holding LLaVA-Video-178K/>
           --out-dir <manifest dir> [--dev-videos 300] [--seed 0] [--workers 16]
Writes <out-dir>/videoitg_{all,train,dev}.jsonl and videoitg_stats.json.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
from pathlib import Path
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_OPTION = re.compile(r"^([A-H])\.\s*(.*\S)\s*$")
_ANSWER = re.compile(r"^\(?([A-H])\b")
LETTERS = "ABCDEFGH"


def parse_mc(text, answer):
    """(stem, {letter: option}, answer letter) of a multiple-choice question, else None. The options are the lines
    "A. ...", "B. ..." in order (at least two); lines after them (the answer instruction) are dropped."""
    stem, options = [], {}
    for line in str(text).split("\n"):
        s = line.strip()
        m = _OPTION.match(s)
        if m and len(options) < len(LETTERS) and m.group(1) == LETTERS[len(options)]:
            options[m.group(1)] = m.group(2)
        elif not options and s:
            stem.append(s)
    m = _ANSWER.match(str(answer).strip())
    letter = m.group(1) if m else None
    if len(options) < 2 or not stem or letter not in options:
        return None
    return " ".join(stem), options, letter


def evidence_spans(seconds, duration):
    """Each marked second k -> [k - 0.5, k + 0.5] inside [0, duration]; touching spans merged."""
    out = []
    for k in sorted({float(k) for k in seconds if 0 <= float(k) < duration}):
        s, e = max(0.0, k - 0.5), min(duration, k + 0.5)
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [[round(s, 3), round(e, 3)] for s, e in out]


def segment_counts(rows, gap=6.0, far=20.0):
    """How many questions have a clue of 1 / 2 / 3+ segments once segments at most `gap` seconds apart are one (the VTS
    filter's --merge-gap and the labels' 2 x 3 s evidence pad close the same gaps), how many of the multi-segment ones
    have two segments more than `far` seconds apart, and the share of the video their [first, last] span covers (above
    30% --merge-spans drops them at stage 1). 2026-10-20: decides whether VideoITG needs multi-segment evidence."""
    out = Counter()
    for x in rows:
        segs = []
        for s, e in x["key_spans"]:
            if segs and s - segs[-1][1] <= gap:
                segs[-1][1] = max(segs[-1][1], e)
            else:
                segs.append([s, e])
        n = len(segs)
        out[f"clue_segments_{min(n, 3)}{'+' if n >= 3 else ''}"] += 1
        if n > 1:
            out["multi_segment_far_apart"] += any(b[0] - a[1] > far for a, b in zip(segs, segs[1:]))
            out["multi_segment_span_over_30pct"] += (segs[-1][1] - segs[0][0]) > 0.3 * x["duration"]
    return dict(out)


def probe(path):
    from src.video.decode import read_timeline
    try:
        return read_timeline(path).duration
    except Exception:                         # unreadable file: the question is dropped and counted
        return None


def build(annotations, video_root, dev_videos=300, seed=0, workers=16):
    counts, parsed = Counter(input=len(annotations)), []
    for r in annotations:
        mc = parse_mc(r.get("question", ""), r.get("answer", ""))
        if mc is None:
            counts["dropped_not_multiple_choice"] += 1
            continue
        parsed.append((r, mc))
    paths = {r["video"]: Path(video_root) / r["video"] for r, _ in parsed}
    present = {v: p for v, p in paths.items() if p.is_file() and p.stat().st_size > 0}
    # demuxing holds the GIL: processes, not threads; spawned, since a forked worker can inherit a lock another thread
    # of the caller held (it hung after tests/test_video.py had decoded in threads)
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        durations = dict(zip(present, pool.map(probe, [str(p) for p in present.values()], chunksize=16)))
    rows = []
    for r, (stem, options, letter) in parsed:
        v = r["video"]
        if v not in present:
            counts["dropped_missing_video"] += 1
            continue
        T = durations[v]
        if not T or T <= 0:
            counts["dropped_unreadable_video"] += 1
            continue
        spans = evidence_spans(r.get("frame_num") or [], T)
        if not spans:
            counts["dropped_no_evidence_in_video"] += 1
            continue
        rows.append({"sample_id": f"videoitg:{r['id']}", "video_id": v, "group_key": f"videoitg:{v}",
                     "video_path": str(paths[v]), "question": stem, "options": options, "answer": letter,
                     "source": "videoitg", "subset": v.split("/")[1], "split": "train", "duration": round(T, 3),
                     "key_spans": spans,
                     "evidence_times": sorted({float(k) for k in r["frame_num"] if 0 <= float(k) < T}),
                     "evidence_kind": "marked_seconds"})
    ids = [x["sample_id"] for x in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate VideoITG ids")
    videos = sorted({x["video_id"] for x in rows})
    held = set(random.Random(seed).sample(videos, min(dev_videos, len(videos))))
    for x in rows:
        if x["video_id"] in held:
            x["split"] = "dev"
    for sp in ("train", "dev"):
        counts[f"rows_{sp}"] = sum(x["split"] == sp for x in rows)
        counts[f"videos_{sp}"] = len({x["video_id"] for x in rows if x["split"] == sp})
    counts.update(segment_counts(rows))
    return rows, dict(counts)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--annotations", required=True, help="video_itg_data.json of NVEagle/VideoITG-40K")
    ap.add_argument("--video-root", required=True, help="directory holding LLaVA-Video-178K/")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--dev-videos", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    rows, stats = build(json.loads(Path(args.annotations).read_text()), args.video_root, args.dev_videos, args.seed,
                        args.workers)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in (("all", rows), ("train", [x for x in rows if x["split"] == "train"]),
                       ("dev", [x for x in rows if x["split"] == "dev"])):
        (out / f"videoitg_{name}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in part))
    (out / "videoitg_stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps(stats))
    if not stats.get("rows_train"):
        raise SystemExit("no VideoITG training question: are the videos under --video-root?")


if __name__ == "__main__":
    main()
