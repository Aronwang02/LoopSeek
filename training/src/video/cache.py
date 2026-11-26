"""Node-local working features with a durable backing directory. No model/GPU dependencies."""
import hashlib
import json
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path



def cache_roots(root):
    """The config keeps its persistent path; an execution-only environment override selects local IO."""
    persistent = Path(root)
    local = Path(os.environ.get("JEV_LOCAL_FEATURES") or root)
    return local, persistent if local.resolve() != persistent.resolve() else None


_REMOTE_FS = ("fuse", "nfs", "cifs", "smb", "9p")


def mmap_safe(path, mounts="/proc/mounts"):
    """Whether feature shards under `path` may be memory-mapped. A shard is replaced by rename, which on a local
    filesystem leaves a live mapping on the old inode; on a FUSE / network mount (blobfuse, NFS, SMB) another node can
    swap the data under the mapping. JEV_FEATURE_MMAP=0 / 1 overrides; without /proc/mounts (macOS) the path is local."""
    forced = os.environ.get("JEV_FEATURE_MMAP")
    if forced in ("0", "1"):
        return forced == "1"
    try:
        with open(mounts) as f:
            table = [line.split()[1:3] for line in f if len(line.split()) >= 3]
    except OSError:
        return True
    real, best, fstype = os.path.realpath(path), "", ""
    for mnt, fs in table:
        mnt = mnt.replace("\\040", " ")
        if (real == mnt or real.startswith(mnt.rstrip("/") + "/")) and len(mnt) >= len(best):
            best, fstype = mnt, fs                     # the deepest (last listed) mount holding the path
    return not fstype.startswith(_REMOTE_FS)


def local_video_target(root, path):
    """Where the local copy of a video goes: its absolute path mirrored under `root`."""
    return Path(root) / os.path.abspath(path).lstrip(os.sep)


@contextmanager
def _video_lock(src, root):
    """Per-video lock shared by the processes that call local_video, so concurrent users copy a video once."""
    import fcntl
    lock = Path(root) / ".copy_locks" / (hashlib.sha1(os.path.abspath(src).encode()).hexdigest() + ".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def copy_video(src, target, root):
    """Copies one video to its node-local target under the per-video lock. True when the target holds the whole file."""
    size = os.path.getsize(src)
    with _video_lock(src, root):
        target = Path(target)
        if target.is_file() and target.stat().st_size == size:
            return True
        copy_atomic(src, target)
        if target.stat().st_size == size:
            return True
        target.unlink()
        return False


def local_video(path):
    """The node-local copy of a video when JEV_LOCAL_VIDEOS is set, else `path`: decoding then seeks a local file instead
    of reading through a network mount. The copy has the same bytes, so content ids, timelines and features do not
    change. A video is copied whole when it is first opened. It is read in place when the copy would take the free space
    below JEV_LOCAL_VIDEOS_RESERVE_GB (500), when the copy fails, or with JEV_VIDEO_COPY_ON_USE=0."""
    root = os.environ.get("JEV_LOCAL_VIDEOS")
    if not root:
        return str(path)
    local = local_video_target(root, path)
    if local.is_file():
        return str(local)
    if os.environ.get("JEV_VIDEO_COPY_ON_USE", "1") == "0":
        return str(path)
    try:
        reserve = float(os.environ.get("JEV_LOCAL_VIDEOS_RESERVE_GB", "500")) * 1e9
        Path(root).mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(root).free - os.path.getsize(path) >= reserve and copy_video(path, local, root):
            return str(local)
    except OSError as exc:
        print(json.dumps({"video_copy_failed": str(path), "error": repr(exc)[:200]}), flush=True)
    return str(path)


def file_signature(path):
    """(size, mtime, inode) of a shard file, None if it is gone. Shards are only replaced by rename, so a rewrite
    changes the signature."""
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return None
    return st.st_size, st.st_mtime_ns, st.st_ino


def copy_atomic(src, dst):
    """Readers see either the old complete file or the new complete file, including after an interrupted copy."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f"{dst.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)


@contextmanager
def _publication_lock(local, relative):
    """Local filesystem locks coordinate writers/publishers on all GPUs; never flock a network mount."""
    import fcntl
    lock = local / ".publish_locks" / relative
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def commit_feature(tmp, final, local, persistent):
    if persistent is None:
        tmp.replace(final)
        return
    relative = final.relative_to(local)
    with _publication_lock(local, relative):
        marker = local / ".pending_uploads" / relative
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()                 # register BEFORE exposing a new local file to other workers
        tmp.replace(final)


def publish_pending(local, persistent, workers=4):
    """Durably publish pending local shards, including files read from another worker's LRU eviction.

    Markers are local and per file. Publishing and replacement hold the same lock, so an older upload cannot
    overwrite a newer one or acknowledge a version it did not copy. Failures leave markers for the next flush.
    """
    pending = local / ".pending_uploads"
    files = list(pending.rglob("feats.*.pt"))
    if not files:
        return
    def publish(marker):
        relative = marker.relative_to(pending)
        with _publication_lock(local, relative):
            if not marker.exists():
                return                 # another worker already completed this upload
            src = local / relative
            if src.exists():
                copy_atomic(src, persistent / relative)
            # A killed first-time writer may have registered a marker but never exposed a complete file.
            marker.unlink()
    with ThreadPoolExecutor(max_workers=min(workers, len(files))) as pool:
        list(pool.map(publish, files))
