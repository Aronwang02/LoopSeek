"""Frames an SFT run's decisions reference that no readable feature shard holds, encoded again before training
(2026-10-16). A no-op when every frame is cached.

A cached shard can be lost or cut short (e.g. a copy taken while the collection rewrote the file). FeatureStore skips
such a file and a collection encodes its frames again, but train_sft.py has no encoder: an earlier run stopped in grounding1 on
"KeyError: feature missing". This step reads the frame references of the decisions train_sft.py will load (--data,
--replay, --decision-types), finds those no readable shard of the working cache holds, and encodes them with the
collection's encoder from the manifests' videos into its own shard (feats.repair_<name>.pt). It takes train_sft.py's
arguments (scripts/train_pipeline.sh sft_run) and ignores the ones it does not use.
usage: python scripts/repair_features.py --config CFG... --name NAME --data FILE [--replay FILE]
           [--decision-types T...] [--set k=v...] --manifests M.jsonl...
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def chunks(path, n):
    """[(path, start, end)]: about n byte ranges of the file, each a run of whole lines."""
    size, bounds = os.path.getsize(path), [0]
    with open(path, "rb") as f:
        for k in range(1, n):
            f.seek(max(bounds[-1], size * k // n))
            f.readline()
            bounds.append(min(f.tell(), size))
    bounds.append(size)
    return [(str(path), a, b) for a, b in zip(bounds, bounds[1:]) if b > a]


def frame_refs(task):
    """({content id: {pts}}, {content id: a sample id}) of the visual segments of the decisions in one byte range."""
    path, a, b, types = task
    need, sample = {}, {}
    with open(path, "rb") as f:
        f.seek(a)
        lines = f.read(b - a).splitlines()
    for line in lines:
        if not line.strip():
            continue
        r = json.loads(line)
        if types and r["decision_type"] not in types:
            continue
        inp = r["input"]
        for seg in [*inp["prefix"], *(s for c in inp["candidates"] for s in c)]:
            if "v" in seg:
                cid = seg["v"][0]
                need.setdefault(cid, set()).add(int(seg["v"][1]))
                sample.setdefault(cid, r["sample_id"])
    return need, sample


def missing_frames(task):
    """(content id, its pts that no readable shard file holds, how many of its shard files are unreadable)."""
    import torch
    cid, d, need, mmap = task
    held, bad = set(), 0
    for f in sorted(Path(d).glob("feats.*.pt")):
        blob = None
        if mmap:
            try:
                blob = torch.load(f, map_location="cpu", mmap=True, weights_only=True)
            except Exception:
                pass
        if blob is None:
            try:
                blob = torch.load(f, map_location="cpu", weights_only=True)
            except Exception:
                bad += 1
                continue
        held.update(int(p) for p in blob["pts"].tolist())
    return cid, [p for p in need if p not in held], bad


def main():
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--config", nargs="+", default=[str(ROOT / "configs/default.yaml")])
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--replay", default=None)
    ap.add_argument("--decision-types", nargs="+", default=None)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--manifests", nargs="*", default=[], help="manifests of the decisions' questions (video paths)")
    ap.add_argument("--workers", type=int, default=min(32, os.cpu_count() or 1))
    args, _ = ap.parse_known_args()                    # the other train_sft.py arguments
    from src.common.config import load_config
    from src.common.utils import read_jsonl
    from src.rollout.build import native_token_store
    from src.video.cache import mmap_safe
    cfg = load_config(args.config, args.set)
    t0, types = time.time(), set(args.decision_types or [])
    need, sample = {}, {}
    files = [args.data] + ([args.replay] if args.replay else [])
    with ProcessPoolExecutor(args.workers) as ex:
        tasks = [(*c, types) for f in files for c in chunks(f, args.workers)]
        for n, s in ex.map(frame_refs, tasks):
            for cid, pts in n.items():
                need.setdefault(cid, set()).update(pts)
            for cid, sid in s.items():
                sample.setdefault(cid, sid)
        _, store = native_token_store(cfg, shard=f"repair_{args.name}")
        mmap = mmap_safe(store.root)
        tasks = [(cid, str(store.root / cid / store.tag), sorted(pts), mmap) for cid, pts in need.items()]
        missing, bad = {}, 0
        for cid, miss, b in ex.map(missing_frames, tasks, chunksize=8):
            bad += b
            if miss:
                missing[cid] = miss
    report = {"repair_features": args.name, "videos": len(need), "frames": sum(map(len, need.values())),
              "unreadable_shards": bad, "missing_videos": len(missing),
              "missing_frames": sum(map(len, missing.values())), "check_seconds": round(time.time() - t0, 1)}
    print(json.dumps(report), flush=True)
    if not missing:
        return
    from src.video.assets import VideoAsset
    videos = {}
    for m in args.manifests:
        for r in read_jsonl(m):
            videos.setdefault(r["sample_id"], r["video_path"])
    enc, store = native_token_store(cfg, device="cuda:0", shard=f"repair_{args.name}")
    encoded, t1 = 0, time.time()
    for cid, pts in sorted(missing.items()):
        path = videos.get(sample[cid])
        if path is None:
            raise SystemExit(f"repair_features.py: no manifest row for {sample[cid]} (video {cid}); pass its manifest")
        asset = VideoAsset(path, cfg["paths"]["features_root"], max_side=cfg["video"]["decode_max_side"])
        if asset.content_id != cid:
            raise SystemExit(f"repair_features.py: {path} is video {asset.content_id}, not {cid}")
        encoded += store.ensure(asset, [asset.index_of_pts(p) for p in pts], enc)
    store.flush()
    print(json.dumps({"repair_features": args.name, "encoded_frames": encoded,
                      "encode_seconds": round(time.time() - t1, 1)}), flush=True)


if __name__ == "__main__":
    main()
