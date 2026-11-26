"""Small shared helpers: seeding, JSON IO, atomic writes, content hashing, timing."""
import hashlib
import json
import os
import random
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, obj, indent=2):
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=indent) + "\n")


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    atomic_write_text(path, "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))


def append_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def atomic_write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def file_content_id(path, chunk=1 << 20):
    """Cheap content identity: size + first/last 1 MiB. Used in feature-cache keys."""
    path = Path(path)
    size = path.stat().st_size
    h = hashlib.sha1(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(chunk))
        if size > chunk:
            f.seek(max(size - chunk, chunk))
            h.update(f.read(chunk))
    return h.hexdigest()[:20]


def stable_hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]


def git_state(root=PROJECT_ROOT):
    """Commit + dirty flag + diff hash, so every run can be traced to exact code."""
    def run(*args):
        try:
            return subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                                  text=True, timeout=30).stdout.strip()
        except Exception:
            return ""
    commit = run("rev-parse", "HEAD")
    diff = run("diff", "HEAD")
    status = run("status", "--porcelain")
    return {"commit": commit or None, "dirty": bool(status),
            "diff_sha1": hashlib.sha1(diff.encode()).hexdigest()[:12] if diff else None}


class Timer:
    """Accumulates wall-clock seconds per named component."""

    def __init__(self):
        self.totals = {}

    @contextmanager
    def __call__(self, name, sync_cuda=False):
        if sync_cuda:
            _cuda_sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if sync_cuda:
                _cuda_sync()
            self.totals[name] = self.totals.get(name, 0.0) + time.perf_counter() - t0

    def add(self, name, seconds):
        self.totals[name] = self.totals.get(name, 0.0) + seconds


def _cuda_sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass
