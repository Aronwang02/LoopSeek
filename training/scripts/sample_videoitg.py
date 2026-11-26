"""Draw the VideoITG questions a run trains and selects on (2026-10-21): VideoITG has 84,139 multiple-choice train
questions on 18,469 videos, 18x the earlier short-video mix (4,618), which would make every short-video collection and
SFT 18x longer. Per split: at most --per-video questions of each video (drawn at random), then at most --train / --dev
questions in all, seeded. Rows keep their content; the VTS filter then runs on the sample.
--bins (20,000 train questions, 30-60 s 20%, 1-2 min 30%, 2-3 min 50%, drawn at random, never the
first ids): --train / --dev questions split over LLaVA-Video duration categories (a row's subset without its source,
30_60_s_youtube_v0_1 -> 30_60_s) by share; each category's quota is drawn at random spread over its videos: videos in
random order, each video's questions in random order, then one question per video per pass until the quota is met (a
category with fewer questions than its quota is an error). Category statistics go to <out>.stats.json. With --bins,
--train / --dev -1 is the most questions the shares allow (the category shortest of its share sets it) and 0 none.
--rest-of FILE (VideoITG filtered in full): the train questions of --manifest not in FILE, nothing drawn.
--all: every row of the --manifest files (several are joined, a sample_id once), nothing drawn.
usage: python scripts/sample_videoitg.py --manifest videoitg_all.jsonl --out videoitg_all.jsonl [--train 12000]
           [--dev 300] [--per-video 1] [--seed 0] [--bins 30_60_s=0.2,1_2_m=0.3,2_3_m=0.5]   (--out may be --manifest)
"""
import argparse
import collections
import json
import random
from pathlib import Path


def sample(rows, limits, per_video=1, seed=0):
    rng = random.Random(seed)
    out = []
    for split, limit in limits.items():
        by_video = {}
        for r in rows:
            if r.get("split") == split:
                by_video.setdefault(r["video_id"], []).append(r)
        picked = []
        for v in sorted(by_video):
            qs = sorted(by_video[v], key=lambda r: r["sample_id"])
            picked += rng.sample(qs, min(per_video, len(qs)))
        rng.shuffle(picked)
        out += sorted(picked[:limit] if limit else picked, key=lambda r: r["sample_id"])
    return out


def category(row):
    """LLaVA-Video-178K duration category of a row: its subset without the source (30_60_s_youtube_v0_1 -> 30_60_s)."""
    return "_".join(row["subset"].split("_")[:3])


def parse_bins(text):
    """'30_60_s=0.2,1_2_m=0.3,2_3_m=0.5' -> {category: share}; the shares must sum to 1."""
    shares = {}
    for part in text.split(","):
        cat, share = part.split("=")
        shares[cat.strip()] = float(share)
    if abs(sum(shares.values()) - 1.0) > 1e-6 or min(shares.values()) <= 0:
        raise SystemExit(f"--bins shares must be positive and sum to 1: {text}")
    return shares


def quotas(n, shares):
    """Integer quotas of n by share, summing to n (largest remainder; ties in the order given)."""
    raw = {c: n * s for c, s in shares.items()}
    q = {c: int(v) for c, v in raw.items()}
    for c in sorted(raw, key=lambda c: -(raw[c] - q[c]))[:n - sum(q.values())]:
        q[c] += 1
    return q


def most_allowed(rows, shares):
    """The largest total whose quotas (by share) every category can fill."""
    have = {c: sum(category(r) == c for r in rows) for c in shares}
    n = min(int(have[c] / s) for c, s in shares.items())
    while n > 0 and any(quotas(n, shares)[c] > have[c] for c in shares):
        n -= 1
    return n


def sample_binned(rows, limits, shares, seed=0):
    """(rows, stats): per split, `limit` questions split over the duration categories by share, each category's quota
    drawn at random spread over its videos (one question per video per pass, videos and questions in random order).
    limit -1: the most the shares allow; 0: none of that split."""
    out, stats = [], {}
    for split, limit in limits.items():
        part = [r for r in rows if r.get("split") == split]
        if limit < 0:
            limit = most_allowed(part, shares)
        if limit == 0:
            continue
        for cat, k in quotas(limit, shares).items():
            rng = random.Random(f"{seed}:{split}:{cat}")
            by_video = {}
            for r in sorted((r for r in part if category(r) == cat), key=lambda r: r["sample_id"]):
                by_video.setdefault(r["video_id"], []).append(r)
            available = sum(len(qs) for qs in by_video.values())
            if available < k:
                raise SystemExit(f"{split} {cat}: {available} questions, quota {k}")
            videos = sorted(by_video)
            rng.shuffle(videos)
            for v in videos:
                rng.shuffle(by_video[v])
            picked, depth = [], 0
            while len(picked) < k:
                for v in videos:
                    if depth < len(by_video[v]):
                        picked.append(by_video[v][depth])
                        if len(picked) == k:
                            break
                depth += 1
            out += picked
            per_video = collections.Counter(r["video_id"] for r in picked)
            stats[f"{split}/{cat}"] = {"quota": k, "picked": len(picked), "available": available,
                                       "videos": len(per_video), "available_videos": len(by_video),
                                       "max_per_video": max(per_video.values())}
    return sorted(out, key=lambda r: r["sample_id"]), stats


def read_rows(paths):
    """The rows of every manifest, each sample_id once (the first file's row)."""
    rows, seen = [], set()
    for p in paths:
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                if r["sample_id"] not in seen:
                    seen.add(r["sample_id"])
                    rows.append(r)
    return rows


def rest_of(rows, selection):
    """The train rows not in `selection` (rows of a selection's manifest), in sample_id order."""
    taken = {r["sample_id"] for r in selection}
    return sorted((r for r in rows if r.get("split") == "train" and r["sample_id"] not in taken),
                  key=lambda r: r["sample_id"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", required=True, nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=12000, help="0 = no cap (with --bins: 0 none, -1 the most allowed)")
    ap.add_argument("--dev", type=int, default=300, help="0 = no cap (with --bins: 0 none, -1 the most allowed)")
    ap.add_argument("--per-video", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bins", default="", help="category=share,...: draw --train / --dev by duration category instead")
    ap.add_argument("--rest-of", default=None, help="a selection: write the train questions it does not hold")
    ap.add_argument("--all", action="store_true", help="write every row of the --manifest files, nothing drawn")
    args = ap.parse_args()
    rows = read_rows(args.manifest)
    stats = {}
    if args.rest_of:
        out = rest_of(rows, read_rows([args.rest_of]))
    elif args.all:
        out = sorted(rows, key=lambda r: r["sample_id"])
    elif args.bins:
        out, stats = sample_binned(rows, {"train": args.train, "dev": args.dev}, parse_bins(args.bins), args.seed)
    else:
        out = sample(rows, {"train": args.train, "dev": args.dev}, args.per_video, args.seed)
    Path(args.out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    summary = {"input": len(rows), "sampled": len(out), **{s: sum(r["split"] == s for r in out) for s in ("train", "dev")}}
    if args.bins and not (args.rest_of or args.all):
        summary.update(bins=args.bins, seed=args.seed, categories=stats)
        Path(f"{args.out}.stats.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
