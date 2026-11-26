"""Per-video asset (content id, cached timeline, decoder, decoded-frame LRU) and the frozen
feature store (SigLIP2 pooled embeddings keyed by content id + exact frame pts)."""
import os
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

from ..common.utils import file_content_id, read_json, stable_hash, write_json
from .decode import (VideoDecoder, read_timeline, timeline_from_json, timeline_to_json,
                     uniform_times)
from .cache import cache_roots, commit_feature, copy_atomic, file_signature, local_video, mmap_safe, publish_pending


class VideoAsset:
    def __init__(self, path, cache_root, max_side=640, frame_cache_size=256):
        self.path = str(path)                       # the manifest path: identity, and what rollout workers receive
        self.read_path = local_video(self.path)     # what is read: the node-local copy staged at job start, if any
        self.content_id = file_content_id(self.read_path)
        local, persistent = cache_roots(cache_root)
        self.cache_dir = local / self.content_id
        self._persistent_timeline = persistent / self.content_id / "timeline.json" if persistent is not None else None
        self.timeline = self._load_timeline()
        self.decoder = VideoDecoder(self.read_path, self.timeline, max_side=max_side)
        self._frames = OrderedDict()
        self._cap = frame_cache_size
        self._lock = threading.Lock()
        self.decode_seconds = 0.0

    def _load_timeline(self):
        f = self.cache_dir / "timeline.json"
        if f.exists():
            return timeline_from_json(read_json(f))
        tl = read_timeline(self.read_path, mode="packets")
        self._save_timeline(tl)
        return tl

    def _save_timeline(self, tl):
        f = self.cache_dir / "timeline.json"
        write_json(f, timeline_to_json(tl), indent=None)
        if self._persistent_timeline is not None:
            copy_atomic(f, self._persistent_timeline)

    def rebuild_timeline_by_decoding(self):
        """Fallback when packet pts do not match decoded frames."""
        tl = read_timeline(self.read_path, mode="decode")
        self._save_timeline(tl)
        self.timeline = tl
        self.decoder = VideoDecoder(self.read_path, tl, max_side=self.decoder.max_side)
        self._frames.clear()

    @property
    def duration(self):
        return self.timeline.duration

    def time(self, idx):
        return self.timeline.times[idx]

    def pts(self, idx):
        return self.timeline.pts[idx]

    def indices_at(self, times):
        """Frame indices displayed at the given times, deduplicated, in time order."""
        return sorted({self.timeline.frame_at(t) for t in times})

    def index_of_pts(self, pts):
        if not hasattr(self, "_pts_index"):
            self._pts_index = {p: i for i, p in enumerate(self.timeline.pts)}
        return self._pts_index[int(pts)]

    def uniform_indices(self, start, end, n):
        return self.indices_at(uniform_times(start, end, n))

    def frames(self, indices):
        """{idx: RGB uint8 HxWx3}; decodes only what is not cached."""
        import time
        indices = [int(i) for i in indices]
        with self._lock:
            missing = [i for i in indices if i not in self._frames]
            if missing:
                t0 = time.perf_counter()
                got = self.decoder.get(missing)
                self.decode_seconds += time.perf_counter() - t0
                for i in missing:
                    self._frames[i] = got[i]
            out = {}
            for i in indices:
                self._frames.move_to_end(i)
                out[i] = self._frames[i]
            while len(self._frames) > self._cap:
                self._frames.popitem(last=False)
        return out


def siglip_tag(processor, revision, decode_max_side, grid=2, mode="pooled"):
    """Feature-cache key: encoder revision + processor config + decode downscale + pooling / mode."""
    proc = {k: str(v) for k, v in processor.to_dict().items()}
    base = f"siglip2b16-224-{str(revision)[:8]}-{stable_hash(proc)}-ms{decode_max_side}"
    return base + (f"-patchgrid{grid}x{grid}" if mode == "pooled" else "-global1")


class SiglipEncoder:
    """Frozen SigLIP2 image tower. Returns spatial patch features pooled on a grid x grid layout
    (update.md §3: 2x2 = 4 tokens per frame), float32 [N, grid*grid, D], row-major
    (top-left, top-right, bottom-left, bottom-right). Patch features are post-LayerNorm."""

    def __init__(self, model_path, revision, device="cuda", dtype="bfloat16", batch_size=64,
                 decode_max_side=640, grid=2, mode="pooled"):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.torch = torch
        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path, dtype=self.dtype)
        self.model = (model.vision_model if hasattr(model, "vision_model") else model).to(self.device).eval()
        self.model.requires_grad_(False)
        self.batch_size = batch_size
        self.grid = grid
        # mode "pooled": grid x grid average-pooled patch features (update.md §3); mode "global" (2026-09-30,
        # exp. 1 arm C): the official SigLIP2 image embedding = attention-pooling head over all 196 patches, 1 token
        assert mode in ("pooled", "global"), mode
        self.mode = mode
        self.dim = self.model.config.hidden_size
        self.tag = siglip_tag(self.processor, revision, decode_max_side, grid, mode)
        self.encode_seconds = 0.0

    def encode(self, frames):
        import time
        torch = self.torch
        t0 = time.perf_counter()
        outs = []
        with torch.inference_mode():
            for s in range(0, len(frames), self.batch_size):
                batch = self.processor(images=list(frames[s:s + self.batch_size]), return_tensors="pt")
                pix = batch["pixel_values"].to(self.device, self.dtype)
                out = self.model(pixel_values=pix)
                if self.mode == "global":
                    outs.append(out.pooler_output.float()[:, None, :].cpu())          # [B, 1, D]
                    continue
                h = out.last_hidden_state                                           # [B, P, D], post-LN
                n = int(round(h.shape[1] ** 0.5))
                assert n * n == h.shape[1], "expected a square patch grid"
                grid = h.float().reshape(h.shape[0], n, n, -1).permute(0, 3, 1, 2)
                pooled = torch.nn.functional.adaptive_avg_pool2d(grid, self.grid)   # [B, D, g, g]
                outs.append(pooled.flatten(2).transpose(1, 2).cpu())                # [B, g*g, D]
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.encode_seconds += time.perf_counter() - t0
        return torch.cat(outs) if outs else torch.zeros(0, self.grid * self.grid, self.dim)


class QwenVisionEncoder:
    """Frozen Qwen3.5 vision tower + its own merger (policy.frame_tokens=qwen_native, 2026-09-30, design B agreed with
    Astra). Each frame is one independent image: squashed to side x side (fixed token count; the official processor
    would keep the aspect ratio), then the official image processor (no further resize at a multiple of 32) and the
    tower. Returns the merger outputs = the LM-space image tokens, bf16 [N, (side/32)^2, out_hidden], row-major over
    the merged grid (the order Qwen3.5's M-RoPE assigns (h, w) positions in)."""

    def __init__(self, model_path, revision, device="cuda", batch_size=64, decode_max_side=640, side=256):
        import torch
        from transformers import AutoImageProcessor, AutoModel
        self.torch = torch
        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(model_path)
        full = AutoModel.from_pretrained(model_path, dtype=torch.bfloat16)
        self.model = full.visual.to(self.device).eval()
        del full
        self.model.requires_grad_(False)
        self.batch_size = batch_size
        self.side = side
        unit = self.processor.patch_size * self.processor.merge_size
        assert side % unit == 0, (side, unit)
        self.grid = side // unit
        self.dim = self.model.config.out_hidden_size
        proc = {k: str(v) for k, v in self.processor.to_dict().items()}
        self.tag = f"qwen35vis-{str(revision)[:8]}-{stable_hash(proc)}-ms{decode_max_side}-sq{side}"
        self.encode_seconds = 0.0

    def encode(self, frames):
        import time

        from PIL import Image
        torch = self.torch
        t0 = time.perf_counter()
        outs = []
        px = self.side * self.side
        with torch.inference_mode():
            for s in range(0, len(frames), self.batch_size):
                ims = [Image.fromarray(f).resize((self.side, self.side), Image.BICUBIC)
                       for f in frames[s:s + self.batch_size]]
                batch = self.processor(images=ims, return_tensors="pt", size={"shortest_edge": px, "longest_edge": px})
                thw = batch["image_grid_thw"]
                assert (thw == torch.tensor([1, 2 * self.grid, 2 * self.grid])).all(), thw
                out = self.model(batch["pixel_values"].to(self.device, torch.bfloat16), grid_thw=thw.to(self.device))
                outs.append(out.pooler_output.reshape(len(ims), self.grid * self.grid, -1).to(torch.bfloat16).cpu())
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.encode_seconds += time.perf_counter() - t0
        return torch.cat(outs) if outs else torch.zeros(0, self.grid * self.grid, self.dim, dtype=torch.bfloat16)


class FeatureStore:
    """Frozen-feature cache. Key = (content_id, exact pts) under one encoder tag.

    Entries are never overwritten, so a stored decision input rebuilds identical features.
    Each process writes its own shard file; loading merges all shards. A shard file holds only what its own shard
    encoded (2026-10-06 fix: it used to re-save everything known for the video, so every run duplicated all earlier
    runs' features -- 85 GB for 41 videos), and at most max_videos tables stay in memory (LRU; dirty ones are flushed
    before eviction).
    With JEV_LOCAL_FEATURES, root remains the durable backing directory and IO uses a node-local working cache.
    flush() publishes pending shards before the caller records completed trajectories.
    FEATURE_CACHE (optional, the features of every TAG): this run's data may reference frames only another TAG's shard
    holds; a process that misses a frame first copies that video's shards it lacks from there (once per video, never
    replacing a file).
    Shards are memory-mapped where that is safe (cache.mmap_safe): a table holds views, so only the frames a caller
    touches are read, as reclaimable page cache shared by every process (2026-10-11: a whole-file torch.load per shard
    made SFT / RL learners read every shard of a video for one decision). A refresh reads only the shard files that
    are new or were rewritten since this process read the video.
    """

    def __init__(self, root, encoder_tag, shard="0", keep_dtype=False, max_videos=64):
        self.root, self.persistent_root = cache_roots(root)
        self.tag = encoder_tag
        self.keep_dtype = keep_dtype   # store entries in their own dtype (bf16 Qwen tokens) instead of float32
        # the default shard is per Slurm job (JEV_FEATURE_SHARD)
        self.shard = str(os.environ.get("JEV_FEATURE_SHARD", "0") if shard == "0" else shard)
        from collections import OrderedDict
        self._mem = OrderedDict()   # content_id -> {pts: tensor}, LRU order
        self._own = {}              # content_id -> {pts: tensor} that belong in THIS shard's file
        self._seen = {}             # content_id -> {shard file name: file_signature} already merged into the table
        self._dirty = set()
        self._fetched = set()       # content ids whose FEATURE_CACHE shards this process copied already
        self.max_videos = int(max_videos)
        self._lock = threading.Lock()
        self._mmap = mmap_safe(self.root)

    def _dir(self, cid):
        return self.root / cid / self.tag

    def _read_shard(self, f):
        """The shard's {pts, feats}, or None for an unreadable file (2026-10-15: a corrupted copy of a cached shard
        stopped a collection; its frames are encoded again instead)."""
        import torch
        if self._mmap:
            try:
                return torch.load(f, map_location="cpu", mmap=True, weights_only=True)
            except RuntimeError:                       # not a zip-format file: read it whole
                pass
        try:
            return torch.load(f, map_location="cpu")
        except Exception as exc:
            print(f"FeatureStore: skipping unreadable shard {f}: {exc!r}"[:400], flush=True)
            return None

    def _merge(self, cid, table, own, seen):
        """Adds the shard files of `cid` that are new or rewritten since `seen` recorded them, in name order: the first
        file holding a pts wins, as when every file is read; entries already in the table keep their values."""
        d = self._dir(cid)
        if not d.exists():
            return
        mine = f"feats.{self.shard}.pt"
        for f in sorted(d.glob("feats.*.pt")):
            sig = file_signature(f)
            if sig is None or seen.get(f.name) == sig:
                continue
            blob = self._read_shard(f)
            for p, v in (zip(blob["pts"].tolist(), blob["feats"]) if blob is not None else ()):
                table.setdefault(int(p), v)
                if f.name == mine:
                    own.setdefault(int(p), v)
            seen[f.name] = sig                         # an unreadable file is read again only once it changes

    def _fetch_shared(self, cid):
        """Copies the FEATURE_CACHE shard files of `cid` that the local working cache lacks; True if it copied any."""
        shared = os.environ.get("FEATURE_CACHE")
        if not shared or cid in self._fetched:
            return False
        if self.persistent_root is None:
            return False
        self._fetched.add(cid)
        src, dst, copied = Path(shared) / cid / self.tag, self._dir(cid), 0
        try:
            if not src.is_dir() or src.resolve() == dst.resolve():
                return False
            for f in sorted(src.glob("feats.*.pt")):
                if not (dst / f.name).exists():
                    copy_atomic(f, dst / f.name)
                    copied += 1
        except OSError as exc:
            print(f"FEATURE_CACHE: copying the shards of {cid} failed: {exc!r}", flush=True)
        return copied > 0

    def _load(self, cid):
        if cid in self._mem:
            self._mem.move_to_end(cid)
            return self._mem[cid]
        table, own, seen = {}, {}, {}
        self._merge(cid, table, own, seen)
        self._mem[cid], self._own[cid], self._seen[cid] = table, own, seen
        while len(self._mem) > self.max_videos:        # LRU eviction (flush first so nothing is lost)
            old = next(iter(self._mem))
            if old == cid:
                break
            if old in self._dirty:
                self._flush_one(old)
                self._dirty.discard(old)
            self._mem.pop(old)
            self._own.pop(old, None)
            self._seen.pop(old, None)
        return table

    def _refresh(self, cid):
        """Adds the entries other processes wrote since the table was loaded (parallel RL workers, collection shards);
        entries already held, including unsaved ones, keep their values. Unchanged shard files are not read again."""
        if cid not in self._mem:
            return self._load(cid)
        self._mem.move_to_end(cid)
        self._merge(cid, self._mem[cid], self._own.setdefault(cid, {}), self._seen.setdefault(cid, {}))
        return self._mem[cid]

    def has(self, cid, pts):
        with self._lock:
            return int(pts) in self._load(cid)

    def get(self, refs):
        """refs: iterable of (content_id, pts) -> tensor [N, ...] (float32; the stored dtype with keep_dtype)."""
        import torch
        with self._lock:
            rows = []
            for cid, p in refs:
                table = self._load(cid)
                if int(p) not in table:
                    table = self._refresh(cid)
                if int(p) not in table and self._fetch_shared(cid):
                    table = self._refresh(cid)
                if int(p) not in table:
                    raise KeyError(f"feature missing: {cid} pts={p}")
                rows.append(table[int(p)])
        return torch.stack(rows) if rows else torch.zeros(0)

    def put(self, cid, pts_list, feats):
        with self._lock:
            table = self._load(cid)
            own = self._own.setdefault(cid, {})
            for p, v in zip(pts_list, feats):
                if int(p) not in table:
                    x = (v if self.keep_dtype else v.float()).clone()
                    table[int(p)] = x
                    own[int(p)] = x
            self._dirty.add(cid)

    def ensure(self, asset: VideoAsset, indices, encoder: SiglipEncoder):
        """Encode frames (by timeline index) that are not cached yet. Returns #newly encoded."""
        # Another worker may have published a new shard after this video's table
        # entered our LRU. Refresh once per request, before decoding/encoding.
        indices = list(dict.fromkeys(indices))
        with self._lock:
            table = self._load(asset.content_id)
            need = [i for i in indices if int(asset.pts(i)) not in table]
            if need:
                table = self._refresh(asset.content_id)
                need = [i for i in need if int(asset.pts(i)) not in table]
            if need and self._fetch_shared(asset.content_id):
                table = self._refresh(asset.content_id)
                need = [i for i in need if int(asset.pts(i)) not in table]
        if not need:
            return 0
        if encoder is None:
            raise KeyError(f"{len(need)} frames of {asset.content_id} not cached and no encoder")
        frames = asset.frames(need)
        feats = encoder.encode([frames[i] for i in need])
        self.put(asset.content_id, [asset.pts(i) for i in need], feats)
        return len(need)

    def _flush_one(self, cid):
        import torch
        own = self._own.get(cid) or {}
        if not own:
            return
        d = self._dir(cid)
        d.mkdir(parents=True, exist_ok=True)
        pts = sorted(own)
        blob = {"pts": torch.tensor(pts, dtype=torch.long), "feats": torch.stack([own[p] for p in pts])}
        tmp = d / f"feats.{self.shard}.pt.{os.getpid()}.tmp"
        torch.save(blob, tmp)
        final = d / f"feats.{self.shard}.pt"
        commit_feature(tmp, final, self.root, self.persistent_root)
        if cid in self._seen:                          # every entry of our own rewrite is in the table already
            self._seen[cid][final.name] = file_signature(final)

    def flush(self):
        with self._lock:
            for cid in list(self._dirty):
                self._flush_one(cid)
            self._dirty.clear()
            # RunnerV3 calls flush before returning a rollout batch. Complete durable copies before its caller
            # writes trajectories/checkpoints, so a new node can restore every referenced frame. Local shards also
            # let other workers load features without reading the shared storage, including LRU evictions.
            if self.persistent_root is not None:
                publish_pending(self.root, self.persistent_root)


def frame_array_stats(frames):
    arr = np.stack(list(frames))
    return {"n": len(arr), "shape": list(arr.shape[1:]), "mean": float(arr.mean())}
