#!/usr/bin/env python3
"""Official Video-MME short/medium/long -> test-only manifests, without subtitles.

python scripts/prepare_videomme.py --root /path/to/videomme --download \
    --scratch /tmp/onejev --out-dir data/manifests/videomme
For existing files omit --download, optionally set --annotations /path/to/test.parquet
and --video-root /path/to/videos. Never drop questions with missing videos.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import threading
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_cgbench import Lock, write_json

REPO = "lmms-lab/Video-MME"
REVISION = "ead1408f75b618502df9a1d8e0950166bf0a2a0b"
ANNOTATION = "videomme/test-00000-of-00001.parquet"
SUBSETS = ("short", "medium", "long")


def source_rows(path):
    import pyarrow.parquet as pq
    if Path(path).suffix == ".parquet":
        data = pq.read_table(path).to_pylist()
    else:
        data = json.loads(Path(path).read_text())
    rows = [r for r in data if r["duration"] in SUBSETS]
    if Counter(r["duration"] for r in rows) != Counter(short=900, medium=900, long=900):
        raise ValueError("expected the complete official short/medium/long subbenches: 900 questions each")
    if len({r["question_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate Video-MME question_id")
    for subset in SUBSETS:
        groups = Counter(r["videoID"] for r in rows if r["duration"] == subset)
        if len(groups) != 300 or set(groups.values()) != {3}:
            raise ValueError(f"{subset}: expected 300 videos with 3 questions each")
    for r in rows:
        # The official release includes a few non-11-character media IDs; keep them verbatim.
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", r["videoID"]):
            raise ValueError(f"invalid videoID: {r['videoID']}")
        if r["answer"] not in "ABCD" or len(r["answer"]) != 1 or len(r["options"]) != 4:
            raise ValueError(f"invalid answer/options: {r['question_id']}")
        for letter, text in zip("ABCD", r["options"]):
            if not text.startswith(f"{letter}. ") or not text[3:].strip():
                raise ValueError(f"invalid original option: {r['question_id']}")
    return rows


def video_index(root):
    index = {}
    for p in sorted(Path(root).rglob("*.mp4")):
        if p.stat().st_size == 0:
            continue
        if p.stem in index:
            raise ValueError(f"ambiguous video file: {p.stem}")
        index[p.stem] = p.resolve()
    return index


def build(rows, video_root, out_dir):
    index = video_index(video_root)
    missing = sorted({r["videoID"] for r in rows} - set(index))
    if missing:
        raise ValueError(f"{len(missing)} Video-MME videos missing under {video_root}, e.g. {missing[:5]}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for subset in SUBSETS:
        manifest = []
        for r in sorted(rows, key=lambda r: r["question_id"]):
            if r["duration"] != subset:
                continue
            manifest.append({
                "sample_id": f"videomme:{r['question_id']}", "video_id": r["videoID"],
                "group_key": f"yt:{r['videoID']}", "video_path": str(index[r["videoID"]]),
                "question": r["question"], "options": {k: v[3:] for k, v in zip("ABCD", r["options"])},
                "answer": r["answer"], "source": "videomme", "subset": subset, "split": "test",
                "duration_category": subset, "domain": r["domain"], "sub_category": r["sub_category"],
                "task_type": r["task_type"], "question_id": r["question_id"],
                "dataset_revision": REVISION,
            })
        dest = out_dir / f"{subset}.jsonl"
        tmp = dest.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in manifest))
        os.replace(tmp, dest)
        print(f"{dest}: {len(manifest)} questions, 300 videos", flush=True)
    write_json(out_dir / "dataset.json", {"repo": REPO, "revision": REVISION, "subtitles": False,
                                          "subsets": {s: {"questions": 900, "videos": 300} for s in SUBSETS}})


def extract_archive(archive, video_root, needed):
    """Flatten only the needed MP4s; an interrupted extraction never leaves a final partial MP4."""
    video_root = Path(video_root)
    video_root.mkdir(parents=True, exist_ok=True)
    members = {}
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            filename = Path(info.filename).name
            if info.is_dir() or Path(filename).suffix.lower() != ".mp4" or Path(filename).stem not in needed:
                continue
            dest = video_root / filename
            if not dest.is_file() or dest.stat().st_size != info.file_size:
                tmp = dest.with_suffix(f".mp4.tmp{threading.get_ident()}")   # chunks extract in parallel
                try:
                    with zf.open(info) as src, tmp.open("wb") as out:
                        shutil.copyfileobj(src, out, 16 << 20)
                    os.replace(tmp, dest)
                finally:
                    tmp.unlink(missing_ok=True)
            members[filename] = info.file_size
    return members


def fetch_chunk(name, marker, scratch, videos, needed):
    from huggingface_hub import hf_hub_download
    t0 = time.time()
    local = Path(hf_hub_download(REPO, name, repo_type="dataset", revision=REVISION, local_dir=scratch / "chunks"))
    members = extract_archive(local, videos, needed)
    write_json(marker, {"revision": REVISION, "subsets": list(SUBSETS), "videos": members})
    local.unlink()  # archives are ~5 GB each; keep only the extracted videos
    return name, len(members), time.time() - t0


def download(root, scratch, workers=4):
    from huggingface_hub import hf_hub_download
    root, scratch = Path(root), Path(scratch) / "videomme-download"
    root.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    # The same lock protocol as scripts/prepare_cgbench.py (one download per root at a time).
    lock = Lock(root, 30)
    while lock.holder() is not None or not lock.acquire():
        print("[prepare_videomme] another job is downloading; waiting", flush=True)
        time.sleep(60)
    try:
        annotations = root / "test.parquet"
        if not annotations.is_file():
            src = hf_hub_download(REPO, ANNOTATION, repo_type="dataset", revision=REVISION, local_dir=scratch / "ann")
            shutil.copyfile(src, annotations.with_suffix(".parquet.tmp"))
            os.replace(annotations.with_suffix(".parquet.tmp"), annotations)
        rows = source_rows(annotations)
        needed = {r["videoID"] for r in rows}
        videos = root / "videos"
        done_dir = root / ".prepare/chunks"
        done_dir.mkdir(parents=True, exist_ok=True)
        if needed <= set(video_index(videos)):
            print(f"[prepare_videomme] cache hit: all {len(needed)} videos present", flush=True)
            return
        todo = []
        for i in range(1, 21):
            name = f"videos_chunked_{i:02d}.zip"
            marker = done_dir / f"{name}.json"
            if marker.is_file():
                record = json.loads(marker.read_text())
                members = record["videos"]
                if record.get("subsets") == list(SUBSETS) and all(
                        (videos / n).is_file() and (videos / n).stat().st_size == size for n, size in members.items()):
                    continue                     # a chunk extracted for other subsets is extracted again
            todo.append((name, marker))
        workers = max(1, min(workers, len(todo)))
        print(f"[prepare_videomme] {20 - len(todo)}/20 chunks already extracted; {len(todo)} to go with "
              f"{workers} workers", flush=True)
        # One chunk at a time takes ~30 min; chunks are independent.
        with ThreadPoolExecutor(workers) as pool:
            futures = [pool.submit(fetch_chunk, name, marker, scratch, videos, needed) for name, marker in todo]
            for k, fut in enumerate(as_completed(futures), 1):
                name, n, sec = fut.result()
                print(f"[prepare_videomme] [{k}/{len(todo)}] {name}: {n} videos in {sec / 60:.1f} min",
                      flush=True)
        missing = needed - set(video_index(videos))
        if missing:
            raise ValueError(f"incomplete Video-MME download: {len(missing)} videos missing")
        write_json(root / "PREPARED.json", {"repo": REPO, "revision": REVISION, "videos": len(needed)})
    finally:
        lock.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scratch", default=tempfile.gettempdir())
    ap.add_argument("--annotations", help="existing official Parquet or flat question JSON")
    ap.add_argument("--video-root", help="existing directory of <videoID>.mp4 (may have nested directories)")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--workers", type=int, default=4, help="chunks downloaded/extracted in parallel")
    args = ap.parse_args()
    if args.download:
        if args.annotations or args.video_root:
            ap.error("--download uses --root/test.parquet and --root/videos; omit custom input paths")
        download(args.root, args.scratch, args.workers)
    annotations = args.annotations or str(Path(args.root) / "test.parquet")
    videos = args.video_root or str(Path(args.root) / "videos")
    build(source_rows(annotations), videos, args.out_dir)


if __name__ == "__main__":
    main()
