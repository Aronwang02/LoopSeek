"""Local collect/RL feature IO, durable publication and restoration on another node; CPU only."""
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from src.video.assets import FeatureStore, VideoAsset
from src.video.cache import commit_feature, copy_atomic, mmap_safe, publish_pending


def test_ensure_refreshes_a_stale_worker_before_reencoding(tmp_path):
    from types import SimpleNamespace
    first = FeatureStore(tmp_path, 'tag', shard='first')
    second = FeatureStore(tmp_path, 'tag', shard='second')
    asset = SimpleNamespace(content_id='v', pts=lambda i: i)
    assert not second.has('v', 7)
    first.put('v', [7], torch.ones(1, 2))
    first.flush()
    assert second.ensure(asset, [7, 7], encoder=None) == 0
    assert not second._dirty
    torch.testing.assert_close(second.get([('v', 7)]), torch.ones(1, 2))


def test_failed_publication_is_retried_without_reencoding(tmp_path, monkeypatch):
    blob, local = tmp_path / "blob", tmp_path / "node"
    monkeypatch.setenv("JEV_LOCAL_FEATURES", str(local))
    store = FeatureStore(blob, "tag", shard="rl_w0", max_videos=1)
    store.put("v", [1], torch.ones(1, 2, 2))
    store.put("v2", [2], torch.ones(1, 2, 2))         # include an already evicted file
    with patch("src.video.cache.copy_atomic", side_effect=OSError("shared storage unavailable")):
        with pytest.raises(OSError, match="shared storage unavailable"):
            store.flush()
    assert len(list((local / ".pending_uploads").rglob("feats.*.pt"))) == 2
    assert not store._dirty
    store.flush()
    assert len(list(blob.rglob("feats.*.pt"))) == 2
    assert not list((local / ".pending_uploads").rglob("feats.*.pt"))


def test_reading_another_workers_unpublished_shard_commits_it_before_returning(tmp_path, monkeypatch):
    blob, local = tmp_path / "blob", tmp_path / "node"
    monkeypatch.setenv("JEV_LOCAL_FEATURES", str(local))
    writer = FeatureStore(blob, "tag", shard="worker0", max_videos=1)
    reader = FeatureStore(blob, "tag", shard="worker1")
    writer.put("v", [1], torch.ones(1, 2))
    writer.put("other", [2], torch.zeros(1, 2))   # eviction exposes v locally before worker0 finishes its rollout
    torch.testing.assert_close(reader.get([("v", 1)]), torch.ones(1, 2))
    assert not (blob / "v/tag/feats.worker0.pt").exists()
    reader.flush()                              # worker1 can commit a trajectory even if worker0 subsequently dies
    assert torch.load(blob / "v/tag/feats.worker0.pt")["pts"].tolist() == [1]
    writer.flush()
    assert (blob / "other/tag/feats.worker0.pt").exists()


def test_interrupted_copy_preserves_previous_complete_file(tmp_path):
    src, dst = tmp_path / "source", tmp_path / "blob/file.pt"
    src.write_bytes(b"new features")
    dst.parent.mkdir()
    dst.write_bytes(b"previous features")
    def partial(s, d):
        Path(d).write_bytes(b"partial")
        raise OSError("interrupted")
    with patch("src.video.cache.shutil.copyfile", side_effect=partial):
        with pytest.raises(OSError, match="interrupted"):
            copy_atomic(src, dst)
    assert dst.read_bytes() == b"previous features"
    assert not list(dst.parent.glob("*.tmp"))


def test_replacement_waits_for_upload_and_keeps_new_version_pending(tmp_path):
    from concurrent.futures import ThreadPoolExecutor, TimeoutError
    from threading import Event
    local, blob = tmp_path / "local", tmp_path / "blob"
    final = local / "v/tag/feats.worker.pt"
    final.parent.mkdir(parents=True)
    tmp = final.with_suffix(".tmp")
    tmp.write_bytes(b"version one")
    commit_feature(tmp, final, local, blob)
    started, release = Event(), Event()
    def slow_copy(src, dst):
        started.set()
        assert release.wait(5)
        copy_atomic(src, dst)
    with ThreadPoolExecutor(2) as pool, patch("src.video.cache.copy_atomic", side_effect=slow_copy):
        upload = pool.submit(publish_pending, local, blob)
        assert started.wait(5)
        tmp.write_bytes(b"version two")
        replacement = pool.submit(commit_feature, tmp, final, local, blob)
        try:
            with pytest.raises(TimeoutError):
                replacement.result(timeout=.05)
        finally:
            release.set()
        upload.result(timeout=5)
        replacement.result(timeout=5)
    assert (blob / "v/tag/feats.worker.pt").read_bytes() == b"version one"
    assert list((local / ".pending_uploads").rglob("feats.*.pt"))
    publish_pending(local, blob)
    assert (blob / "v/tag/feats.worker.pt").read_bytes() == b"version two"


def _counting_loads():
    """patch("torch.load") side effect that records every shard file read and the mmap flag it was read with."""
    load, reads = torch.load, []
    def counted(filename, **kw):
        reads.append((Path(filename).name, bool(kw.get("mmap"))))
        return load(filename, **kw)
    return counted, reads


def test_refresh_reads_only_new_or_rewritten_shards(tmp_path):
    a, b, c = (FeatureStore(tmp_path, "tag", shard=s) for s in ("a", "b", "c"))
    a.put("v", [1], torch.full((1, 2), 1.0))
    a.flush()
    c.put("v", [2], torch.full((1, 2), 2.0))
    c.flush()
    counted, reads = _counting_loads()
    with patch("torch.load", side_effect=counted):
        assert b.has("v", 1) and b.has("v", 2)                     # cold load: every shard once
        assert sorted(n for n, _ in reads) == ["feats.a.pt", "feats.c.pt"]
        reads.clear()
        with pytest.raises(KeyError):
            b.get([("v", 3)])                                       # a miss with nothing new reads no shard again
        assert reads == []
        a.put("v", [3], torch.full((1, 2), 3.0))
        a.flush()                                                   # a rewrites its shard: only that file is re-read
        torch.testing.assert_close(b.get([("v", 3)]), torch.full((1, 2), 3.0))
        assert [n for n, _ in reads] == ["feats.a.pt"]
        reads.clear()
        b.put("v", [4], torch.full((1, 2), 4.0))
        b.flush()                                                   # its own rewrite holds nothing new for b
        with pytest.raises(KeyError):
            b.get([("v", 5)])
        assert reads == []


def test_first_shard_wins_and_held_entries_keep_their_values_after_refresh(tmp_path):
    x, y, reader = (FeatureStore(tmp_path, "tag", shard=s) for s in ("x", "y", "reader"))
    reader.put("v", [7], torch.full((1, 2), 70.0))                  # held, unsaved
    y.put("v", [5, 7], torch.tensor([[5.0, 5.0], [7.0, 7.0]]))      # x and y encode pts 5 concurrently
    x.put("v", [5, 6], torch.tensor([[-5.0, -5.0], [6.0, 6.0]]))
    y.flush()
    x.flush()                                                       # feats.x.pt sorts before feats.y.pt
    got = reader.get([("v", 5), ("v", 6), ("v", 7)])
    torch.testing.assert_close(got, torch.tensor([[-5.0, -5.0], [6.0, 6.0], [70.0, 70.0]]))
    fresh = FeatureStore(tmp_path, "tag", shard="fresh")            # the same values as a cold read of every file
    torch.testing.assert_close(fresh.get([("v", 5), ("v", 6)]), got[:2])


def test_shards_are_memory_mapped_unless_disabled(tmp_path, monkeypatch):
    w = FeatureStore(tmp_path, "tag", shard="w", keep_dtype=True)
    x = torch.arange(12, dtype=torch.bfloat16).reshape(3, 2, 2)
    w.put("v", [1, 2, 3], x)
    w.flush()
    for env, mapped in (("1", True), ("0", False)):
        monkeypatch.setenv("JEV_FEATURE_MMAP", env)
        counted, reads = _counting_loads()
        with patch("torch.load", side_effect=counted):
            got = FeatureStore(tmp_path, "tag", shard="r", keep_dtype=True).get([("v", 3), ("v", 1)])
        assert reads == [("feats.w.pt", mapped)]
        torch.testing.assert_close(got, x[[2, 0]])


def test_mmap_only_on_local_filesystems(tmp_path, monkeypatch):
    monkeypatch.delenv("JEV_FEATURE_MMAP", raising=False)
    mounts = tmp_path / "mounts"
    blob = tmp_path / "blob"
    mounts.write_text(f"/dev/nvme0n1 / ext4 rw 0 0\nblobfuse2 {blob} fuse rw,allow_other 0 0\n")
    assert mmap_safe(tmp_path / "local" / "features", mounts)
    assert not mmap_safe(blob / "features", mounts)
    link = tmp_path / "data_features"
    link.symlink_to(blob / "features", target_is_directory=True)   # data/features -> a network mount
    assert not mmap_safe(link, mounts)
    mounts.write_text("server:/export /mnt/nfs nfs4 rw 0 0\n/dev/sda1 / xfs rw 0 0\n")
    assert not mmap_safe("/mnt/nfs/features", mounts) and mmap_safe("/scratch/features", mounts)
    assert mmap_safe(tmp_path, tmp_path / "missing")                # no /proc/mounts (macOS): local
    monkeypatch.setenv("JEV_FEATURE_MMAP", "1")
    assert mmap_safe(blob, mounts)


def test_a_miss_copies_that_videos_shards_from_the_shared_cache(tmp_path, monkeypatch):
    """FEATURE_CACHE may be filled in the background. SFT reading a frame its data took from another TAG's
    shard copies that video's shards itself instead of failing; collection copies before it encodes."""
    from types import SimpleNamespace
    shared, blob, local = tmp_path / "cache", tmp_path / "blob", tmp_path / "node"
    monkeypatch.delenv("JEV_LOCAL_FEATURES", raising=False)
    monkeypatch.delenv("FEATURE_CACHE", raising=False)
    other = FeatureStore(shared, "tag", shard="run_other_s1A_p0", keep_dtype=True)
    x = torch.arange(8, dtype=torch.bfloat16).reshape(2, 2, 2)
    other.put("video", [10, 20], x)
    other.put("v2", [5], x[:1])
    other.flush()
    monkeypatch.setenv("JEV_LOCAL_FEATURES", str(local))
    monkeypatch.setenv("FEATURE_CACHE", str(shared))
    sft = FeatureStore(blob, "tag", shard="sft", keep_dtype=True)
    torch.testing.assert_close(sft.get([("video", 10), ("video", 20)]), x)
    assert (local / "video/tag/feats.run_other_s1A_p0.pt").exists()
    assert not (blob / "video").exists()                       # not this run's shard: never published as its own
    with pytest.raises(KeyError):
        sft.get([("video", 30)])                                # in no shard anywhere: still an error
    collect = FeatureStore(blob, "tag", shard="collect", keep_dtype=True)
    assert collect.ensure(SimpleNamespace(content_id="v2", pts=lambda i: i), [5], encoder=None) == 0
    monkeypatch.delenv("FEATURE_CACHE")                         # FEATURE_CACHE="": no fallback
    monkeypatch.setenv("JEV_LOCAL_FEATURES", str(tmp_path / "node2"))
    with pytest.raises(KeyError):
        FeatureStore(blob, "tag", shard="plain", keep_dtype=True).get([("video", 10)])
