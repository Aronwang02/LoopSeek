"""LV-Haystack parquet -> manifests for full Ego4D videos (or explicit clip mode).

Full videos use frame_indexes_video / frame_rate and video_duration. Human keyframes become +/- tau spans.
Source videos in held-out splits are excluded from earlier splits (test > val > train).
Splits: official val maps to test (VTS evaluates Haystack-Ego4D on it); dev = --dev-videos source
videos of official train, drawn with --dev-seed, so model selection never sees the evaluation questions; official test
is not used.
"""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random

import pandas as pd

SPLIT_NAME = {"train": "train", "val": "test", "test": "official_test"}   # manifest split of each official split


def spans(times, tau, duration):
    out = []
    for t in sorted(set(times)):
        start, end = max(0.0, t - tau), min(duration, t + tau)
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out


def read_split(root, split):
    root = Path(root)
    exact = root / f"{split}.parquet"
    files = [exact] if exact.exists() else sorted(root.glob(f"{split}-*.parquet"))
    if not files:
        files = sorted((root / "data").glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"missing LV-Haystack {split} parquet in {root}")
    return [r for p in files for r in pd.read_parquet(p).to_dict("records")]


def convert_rows(records, split, media_root, mode="video", tau=2.0, excluded=(), require_videos=True):
    if not math.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be positive and finite")
    rows, counts = [], Counter(input=len(records))
    for i, r in enumerate(records):
        if r["video_id"] in excluded:
            counts["dropped_heldout_video"] += 1
            continue
        meta = r["video_metadata"]
        duration = float(meta["video_duration" if mode == "video" else "vclip_duration"])
        fps = float(meta["frame_rate"])
        options = {k: str(v).strip() for k, v in sorted(dict(r["options"]).items()) if v is not None and str(v).strip()}
        question, answer = str(r["question"] or "").strip(), str(r["answer"] or "").strip().upper()
        if (not question or answer not in options or len(options) < 2 or
                not math.isfinite(duration) or duration <= 0 or not math.isfinite(fps) or fps <= 0):
            counts["dropped_invalid_qa_or_metadata"] += 1
            continue
        field = "frame_indexes_video" if mode == "video" else "frame_indexes_vclip"
        times = sorted({float(f) / fps for f in r[field] if math.isfinite(float(f)) and 0 <= float(f) / fps < duration})
        if not times:
            counts["dropped_no_valid_keyframe"] += 1
            continue
        uid = r["video_id"] if mode == "video" else r["vclip_id"]
        path = Path(media_root) / f"{uid}.mp4"
        if require_videos and (not path.is_file() or path.stat().st_size == 0):
            raise FileNotFoundError(f"missing LV-Haystack video: {path}; run prepare_lvhaystack.py --download first")
        rows.append({"sample_id": f"lvh:{split}:{i}", "video_id": uid, "group_key": f"ego4d:{r['video_id']}",
                     "video_path": str(path), "question": question, "options": options, "answer": answer,
                     "source": "lvhaystack", "subset": f"lvhaystack_{split}",
                     "split": SPLIT_NAME[split], "duration": duration,
                     "evidence_times": times, "key_spans": spans(times, tau, duration),
                     "evidence_kind": "keyframes", "evidence_tau": tau, "media_mode": mode})
    counts["rows"] = len(rows)
    return rows, dict(counts)


def build_manifests(parquet_dir, media_root, out_dir, mode="video", tau=2.0, require_videos=True,
                    splits=("train", "val"), dev_videos=50, dev_seed=0):
    # Reserve test source videos using annotations only; their media are not downloaded for training.
    raw = {s: read_split(parquet_dir, s) for s in ("train", "val", "test")}
    test_ids = {r["video_id"] for r in raw["test"]}
    val_ids = {r["video_id"] for r in raw["val"]}
    excluded = {"train": test_ids | val_ids, "val": test_ids, "test": set()}
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows, stats = [], {}
    for split in splits:
        rows, stats[split] = convert_rows(raw[split], split, media_root, mode, tau, excluded[split], require_videos)
        if split == "train" and dev_videos:   # dev: whole source videos of official train (group_key, also in clip mode)
            sources = sorted({r["group_key"] for r in rows})
            held = set(random.Random(dev_seed).sample(sources, min(dev_videos, len(sources))))
            for r in rows:
                if r["group_key"] in held:
                    r["split"] = "dev"
            stats["dev"] = {"source_videos": len(held), "rows": sum(r["split"] == "dev" for r in rows)}
        all_rows.extend(rows)
    for name in sorted({r["split"] for r in all_rows}):
        (out_dir / f"lvhaystack_{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in all_rows
                                                                  if r["split"] == name))
    (out_dir / "lvhaystack_all.jsonl").write_text("".join(json.dumps(r) + "\n" for r in all_rows))
    (out_dir / "lvhaystack_stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    return all_rows, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet-dir", required=True)
    media = ap.add_mutually_exclusive_group(required=True)
    media.add_argument("--video-root", help="full videos named <video_id>.mp4 (long-video training default)")
    media.add_argument("--clip-root", help="clips named <vclip_id>.mp4; use clip-relative labels")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val"])
    ap.add_argument("--dev-videos", type=int, default=50, help="source videos of official train held out as dev")
    ap.add_argument("--dev-seed", type=int, default=0)
    args = ap.parse_args()
    _, stats = build_manifests(args.parquet_dir, args.video_root or args.clip_root, args.out_dir,
                               "video" if args.video_root else "clip", args.tau, splits=args.splits,
                               dev_videos=args.dev_videos, dev_seed=args.dev_seed)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
