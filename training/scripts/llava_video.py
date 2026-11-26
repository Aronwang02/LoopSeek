"""Download of LLaVA-Video-178K videos (Hugging Face dataset lmms-lab/LLaVA-Video-178K, pinned revision), used by
scripts/fetch_videoitg.py: the tar.gz chunks of a subset are streamed and only the needed members reach the disk.
HF_TOKEN (or HUGGING_FACE_HUB_TOKEN), when set, is sent with the requests."""
import contextlib
import json
import os
import shutil
import tarfile
import time
from pathlib import Path

LLAVA_REPO, LLAVA_REV = "lmms-lab/LLaVA-Video-178K", "6d8c562dc26d70042a0d9704d1cae58c94b89098"


def log(**kw):
    print(json.dumps(kw), flush=True)


def hf_headers():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def write_atomic(src, dest, size=None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with open(tmp, "wb") as f:
        shutil.copyfileobj(src, f, 1 << 22)
    if size is not None and tmp.stat().st_size != size:
        tmp.unlink()
        raise IOError(f"{dest}: wrote {tmp.stat().st_size} bytes, expected {size}")
    os.replace(tmp, dest)


def llava_chunks(subset):
    import requests
    files = requests.get(f"https://huggingface.co/api/datasets/{LLAVA_REPO}/tree/{LLAVA_REV}/{subset}",
                         headers=hf_headers(), timeout=60).json()
    tars = sorted((x["path"] for x in files if x["path"].endswith(".tar.gz")),
                  key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
    return [(p, f"https://huggingface.co/datasets/{LLAVA_REPO}/resolve/{LLAVA_REV}/{p}") for p in tars]


def stream_chunk(name, url, members):
    """Stream one tar.gz chunk and write the members it holds (5 tries). Returns (written, streamed to its end)."""
    import requests
    got = 0
    for attempt in range(5):        # 2026-10-21: many chunks stream at once; a throttled one waits longer each try
        try:
            with requests.get(url, stream=True, timeout=(30, 300), headers=hf_headers()) as r:
                r.raise_for_status()
                r.raw.decode_content = False
                with tarfile.open(fileobj=r.raw, mode="r|gz") as tar:
                    for m in tar:
                        dest = members.get(m.name) if m.isfile() else None
                        if dest is not None and not dest.exists():
                            write_atomic(tar.extractfile(m), dest, m.size)
                            got += 1
            return got, True
        except Exception as exc:
            log(source="llava", chunk=name, attempt=attempt + 1, error=repr(exc)[:300])
            time.sleep(30 * (attempt + 1))
    return got, False


def record_chunk(done_dir, subset, name, lock=None):
    """A chunk streamed to its end: its members are on disk, a resumed run need not stream it again."""
    record = Path(done_dir) / f"{subset}.chunks"
    record.parent.mkdir(parents=True, exist_ok=True)
    with (lock or contextlib.nullcontext()):
        with open(record, "a") as f:
            f.write(name + "\n")


def recorded_chunks(done_dir, subset):
    record = Path(done_dir) / f"{subset}.chunks" if done_dir else None
    return set(record.read_text().split()) if record and record.is_file() else set()


def fetch_llava(todo, done_dir=None):
    """todo: {subset: {tar member: local path}}; every chunk of a subset is streamed until all its members are written.
    done_dir: a chunk streamed to its end is recorded in done_dir/<subset>.chunks and skipped by a later run that has
    its members back (a preempted job resumes instead of streaming again)."""
    for subset, members in todo.items():
        log(source="llava", subset=subset, to_fetch=len(members))
        done = recorded_chunks(done_dir, subset)
        for name, url in llava_chunks(subset):
            members = {m: p for m, p in members.items() if not p.exists()}
            if not members:
                break
            if name in done:
                continue
            t0 = time.time()
            got, streamed = stream_chunk(name, url, members)
            if streamed and done_dir:
                record_chunk(done_dir, subset, name)
            log(source="llava", chunk=name, written=got, left=sum(not p.exists() for p in members.values()),
                seconds=round(time.time() - t0))


def fetch_llava_parallel(todo, done_dir=None, workers=16):
    """fetch_llava with `workers` chunks streaming at once, of every subset (2026-10-21: one stream per subset gave
    ~35 MB/s each, and the VideoITG subset with ~30 needed videos per chunk took ~3 h alone). The chunks go round-robin
    over the subsets (every subset's k-th chunk before any (k+1)-th); a subset whose members are all written skips its
    remaining chunks. Records as fetch_llava."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    lock, per_subset = threading.Lock(), {}
    for subset, members in todo.items():
        done = recorded_chunks(done_dir, subset)
        per_subset[subset] = [(n, u) for n, u in llava_chunks(subset) if n not in done]
        log(source="llava", subset=subset, to_fetch=sum(not p.exists() for p in members.values()),
            chunks_left=len(per_subset[subset]))
    width = max((len(c) for c in per_subset.values()), default=0)
    tasks = [(s, *c[k]) for k in range(width) for s, c in per_subset.items() if k < len(c)]

    def one(subset, name, url):
        members = {m: p for m, p in todo[subset].items() if not p.exists()}
        if not members:
            return
        t0 = time.time()
        got, streamed = stream_chunk(name, url, members)
        if streamed and done_dir:
            record_chunk(done_dir, subset, name, lock)
        log(source="llava", chunk=name, written=got, left=sum(not p.exists() for p in todo[subset].values()),
            seconds=round(time.time() - t0))

    with ThreadPoolExecutor(max(1, workers)) as pool:
        for job in [pool.submit(one, *t) for t in tasks]:
            job.result()
