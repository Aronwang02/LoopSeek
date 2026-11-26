#!/usr/bin/env python3
"""Download CG-Bench (HF dataset CG-Bench/CG-Bench, gated) once into a shared directory; later runs reuse it.

Layout under --root (default $DATA_ROOT):
  cgbench.json, cgbench_mini.json      annotations
  videos/<video_uid>.mp4               the 69 video_chunk_NN.zip volumes (~350 GB), flattened
  PREPARED.json                        written last, only when every video_uid of both annotations has its mp4
  .prepare/chunks/<chunk>.json         per-volume completion records (resume point); .prepare/lock.json
With PREPARED.json for the pinned revision and the recorded video count, the run returns at once (cache hit).
Otherwise volumes are downloaded to node-local --scratch (not HF_HOME), extracted straight into videos/, and
deleted; an interrupted run resumes at the first unfinished volume. clue_video_chunk_*.zip and subtitles.zip are
not used by this repo and are skipped. build_cgbench.py drops questions whose mp4 is missing without failing,
which is why completion is checked against the annotations here.
One job downloads at a time: others wait on .prepare/lock.json (heartbeat every minute, stale after
--stale-minutes) until PREPARED.json appears. Needs HF_TOKEN from an account that accepted the dataset terms.
usage: python scripts/prepare_cgbench.py --root <dir> [--scratch <local dir>] [--workers 4] [--check-only]
"""
import argparse
import json
import os
import re
import shutil
import socket
import sys
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = "CG-Bench/CG-Bench"
REVISION = "a3be039f6ba85e022a749eca111b64ffcafadd1d"   # dataset commit of 2025-03-31
ANNOTATIONS = ("cgbench.json", "cgbench_mini.json")
CHUNK = re.compile(r"^video_chunk_\d+\.zip$")


def log(*a):
    print("[prepare_cgbench]", *a, flush=True)


def write_json(path, obj):
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def needed_videos(root):
    uids = set()
    for name in ANNOTATIONS:
        uids |= {r["video_uid"] for r in json.loads((root / name).read_text()) if r.get("video_uid")}
    return uids


def missing_videos(root):
    have = {p.name for p in os.scandir(root / "videos") if p.name.endswith(".mp4")} \
        if (root / "videos").is_dir() else set()
    return sorted(u for u in needed_videos(root) if f"{u}.mp4" not in have), len(have)


def cache_hit(root):
    rec = read_json(root / "PREPARED.json")
    if not rec or rec.get("repo") != REPO or rec.get("revision") != REVISION:
        return False
    if not all((root / n).is_file() for n in ANNOTATIONS) or not (root / "videos").is_dir():
        return False
    return sum(1 for p in os.scandir(root / "videos") if p.name.endswith(".mp4")) >= rec["videos"]


class Lock:
    """Advisory lock on a shared directory, where flock may be unavailable."""

    def __init__(self, root, stale_minutes):
        self.path = root / ".prepare" / "lock.json"
        self.stale = stale_minutes * 60
        self.me = {"host": socket.gethostname(), "pid": os.getpid(), "job": os.environ.get("SLURM_JOB_ID", "")}
        self._stop = threading.Event()

    def holder(self):
        cur = read_json(self.path)
        if cur and cur.get("owner") != self.me and time.time() - cur.get("heartbeat", 0) < self.stale:
            return cur
        return None

    def acquire(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        write_json(self.path, {"owner": self.me, "heartbeat": time.time()})
        time.sleep(5)                     # a job that wrote at the same moment wins or loses here
        cur = read_json(self.path)
        if not cur or cur.get("owner") != self.me:
            return False
        threading.Thread(target=self._beat, daemon=True).start()
        return True

    def _beat(self):
        while not self._stop.wait(60):
            write_json(self.path, {"owner": self.me, "heartbeat": time.time()})

    def release(self):
        self._stop.set()
        cur = read_json(self.path)
        if cur and cur.get("owner") == self.me:
            self.path.unlink(missing_ok=True)


def extract_chunk(name, root, scratch, done_dir, attempts=3):
    for attempt in range(1, attempts + 1):
        try:
            return _extract_chunk(name, root, scratch, done_dir)
        except Exception as exc:          # network / truncated download; access errors are not retried
            if attempt == attempts or type(exc).__name__ in ("GatedRepoError", "RepositoryNotFoundError"):
                raise
            log(f"{name}: attempt {attempt} failed ({type(exc).__name__}: {exc}); retrying")
            time.sleep(30 * attempt)


def _extract_chunk(name, root, scratch, done_dir):
    from huggingface_hub import hf_hub_download
    t0 = time.time()
    local = Path(hf_hub_download(REPO, name, repo_type="dataset", revision=REVISION, local_dir=scratch / "chunks"))
    members = {}
    try:
        with zipfile.ZipFile(local) as zf:
            for info in zf.infolist():
                if info.is_dir() or not info.filename.lower().endswith(".mp4"):
                    continue
                dest = root / "videos" / Path(info.filename).name
                if not (dest.is_file() and dest.stat().st_size == info.file_size):
                    with zf.open(info) as src, open(dest, "wb") as out:      # zipfile checks the CRC at EOF
                        shutil.copyfileobj(src, out, 16 << 20)
                members[dest.name] = info.file_size
    finally:
        local.unlink(missing_ok=True)
    write_json(done_dir / (name + ".json"), {"chunk": name, "videos": members, "finished_at": time.time()})
    return name, len(members), sum(members.values()), time.time() - t0


def prepare(args, root, scratch):
    from huggingface_hub import HfApi, hf_hub_download
    for name in ANNOTATIONS:
        src = Path(hf_hub_download(REPO, name, repo_type="dataset", revision=REVISION, local_dir=scratch / "ann"))
        shutil.copyfile(src, root / (name + ".tmp"))
        os.replace(root / (name + ".tmp"), root / name)
    (root / "videos").mkdir(exist_ok=True)
    done_dir = root / ".prepare" / "chunks"
    done_dir.mkdir(parents=True, exist_ok=True)
    chunks = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset", revision=REVISION) if CHUNK.match(f))
    todo = [c for c in chunks if not (done_dir / (c + ".json")).is_file()]
    free_gb = shutil.disk_usage(scratch).free / 1e9
    workers = max(1, min(args.workers, len(todo) or 1, int(free_gb // 7)))   # volumes are up to ~5.4 GB
    log(f"{len(chunks) - len(todo)}/{len(chunks)} volumes already extracted; {len(todo)} to go with {workers} "
        f"workers (scratch {scratch}, {free_gb:.0f} GB free)")
    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(extract_chunk, c, root, scratch, done_dir) for c in todo]
        for k, fut in enumerate(as_completed(futures), 1):
            name, n, size, sec = fut.result()
            log(f"[{k}/{len(todo)}] {name}: {n} videos, {size / 1e9:.1f} GB in {sec / 60:.1f} min")
    missing, have = missing_videos(root)
    if missing:
        sys.exit(f"[prepare_cgbench] {len(missing)} annotated videos are still missing after all volumes "
                 f"(e.g. {missing[:5]}); not marking {root} as prepared")
    write_json(root / "PREPARED.json", {"repo": REPO, "revision": REVISION, "videos": have, "chunks": len(chunks),
                                        "needed_videos": len(needed_videos(root)), "finished_at": time.time()})
    log(f"prepared {root}: {have} videos, all annotated videos present")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.environ.get("DATA_ROOT"), required=not os.environ.get("DATA_ROOT"))
    ap.add_argument("--scratch", default=None, help="node-local download directory (default: system temp)")
    ap.add_argument("--workers", type=int, default=4, help="volumes downloaded/extracted in parallel")
    ap.add_argument("--stale-minutes", type=float, default=30, help="take over a lock without heartbeat after this")
    ap.add_argument("--wait-hours", type=float, default=24, help="give up waiting for another job's download")
    ap.add_argument("--check-only", action="store_true", help="exit 0 if prepared, 1 otherwise; download nothing")
    args = ap.parse_args()
    root = Path(args.root)
    if cache_hit(root):
        log(f"cache hit: {root} is prepared ({REPO}@{REVISION[:8]})")
        return
    if args.check_only:
        sys.exit(f"[prepare_cgbench] {root} is not prepared")
    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        sys.exit("[prepare_cgbench] CG-Bench is gated and no HF_TOKEN is set. Accept the terms at "
                 f"https://huggingface.co/datasets/{REPO} and provide that account's token "
                 "(export HF_TOKEN=...).")
    root.mkdir(parents=True, exist_ok=True)
    scratch = Path(args.scratch or tempfile.gettempdir()) / "cgbench-download"
    scratch.mkdir(parents=True, exist_ok=True)
    lock = Lock(root, args.stale_minutes)
    deadline = time.time() + args.wait_hours * 3600
    while True:
        if cache_hit(root):
            log(f"cache hit after waiting: {root} is prepared")
            return
        other = lock.holder()
        if other is None and lock.acquire():
            break
        if time.time() > deadline:
            sys.exit(f"[prepare_cgbench] gave up waiting for {other} to finish preparing {root}")
        log(f"another job is preparing {root} ({other or 'lock just taken'}); waiting")
        time.sleep(60)
    try:
        prepare(args, root, scratch)
    except Exception as exc:
        if type(exc).__name__ in ("GatedRepoError", "RepositoryNotFoundError"):
            sys.exit(f"[prepare_cgbench] {type(exc).__name__}: the HF_TOKEN account has no access to {REPO}; "
                     f"accept the terms at https://huggingface.co/datasets/{REPO} with that account")
        raise
    finally:
        lock.release()


if __name__ == "__main__":
    main()
