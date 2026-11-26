"""Component construction shared by the GPU entry points."""
from collections import OrderedDict
from pathlib import Path

from ..common.utils import read_jsonl
from ..data.manifest import public_sample
from ..video.assets import FeatureStore, SiglipEncoder, VideoAsset


def load_rows(cfg, manifest="videor1", split=None, file="manifest.jsonl", limit=None):
    rows = read_jsonl(Path(cfg["paths"]["manifest_dir"]) / manifest / file)
    if split:
        rows = [r for r in rows if r["split"] == split]
    rows = sorted(rows, key=lambda r: r["sample_id"])
    return rows[:limit] if limit else rows


class AssetPool:
    """One VideoAsset per file (timeline cached on disk, decoded frames cached in memory)."""

    def __init__(self, cfg, max_assets=64):
        self.cfg = cfg
        self._pool = OrderedDict()
        self.max_assets = max_assets

    def get(self, path):
        a = self._pool.get(path)
        if a is None:
            a = VideoAsset(path, self.cfg["paths"]["features_root"], max_side=self.cfg["video"]["decode_max_side"])
            self._pool[path] = a
            while len(self._pool) > self.max_assets:
                self._pool.popitem(last=False)
        self._pool.move_to_end(path)
        return a


def build_encoder_and_store(cfg, device="cuda", shard="0"):
    m = cfg["models"]["siglip2"]
    enc = SiglipEncoder(m["path"], m["revision"], device=device,
                        decode_max_side=cfg["video"]["decode_max_side"], grid=cfg["video"]["pool_grid"])
    store = FeatureStore(cfg["paths"]["features_root"], enc.tag, shard=shard)
    return enc, store


def global_token_store(cfg, device=None, shard="0"):
    """(encoder or None, FeatureStore) of the official SigLIP2 global embeddings (1 token per frame; policy
    frame_tokens=global1). With device=None only the store is opened (tag from the processor config)."""
    from transformers import AutoImageProcessor
    from ..video.assets import siglip_tag
    m = cfg["models"]["siglip2"]
    enc = None
    if device is not None:
        enc = SiglipEncoder(m["path"], m["revision"], device=device, decode_max_side=cfg["video"]["decode_max_side"],
                            grid=cfg["video"]["pool_grid"], mode="global")
        tag = enc.tag
    else:
        tag = siglip_tag(AutoImageProcessor.from_pretrained(m["path"]), m["revision"], cfg["video"]["decode_max_side"],
                         cfg["video"]["pool_grid"], "global")
    return enc, FeatureStore(cfg["paths"]["features_root"], tag, shard=shard)


def native_token_store(cfg, device=None, shard="0"):
    """(encoder or None, FeatureStore) of the Qwen3.5 native image tokens (policy.frame_tokens=qwen_native). With
    device=None only the store is opened (tag from the processor config, same formula as QwenVisionEncoder)."""
    from transformers import AutoImageProcessor
    from ..common.utils import stable_hash
    from ..video.assets import QwenVisionEncoder
    m, side, ms = (cfg["models"][cfg["policy"].get("qwen35_model", "qwen35_4b")], cfg["policy"].get("native_side", 256),
                   cfg["video"]["decode_max_side"])
    enc = None
    if device is not None:
        enc = QwenVisionEncoder(m["path"], m["revision"], device=device, decode_max_side=ms, side=side)
        tag = enc.tag
    else:
        proc = {k: str(v) for k, v in AutoImageProcessor.from_pretrained(m["path"]).to_dict().items()}
        tag = f"qwen35vis-{str(m['revision'])[:8]}-{stable_hash(proc)}-ms{ms}-sq{side}"
    return enc, FeatureStore(cfg["paths"]["features_root"], tag, shard=shard, keep_dtype=True,
                             max_videos=int(cfg["video"].get("feature_cache_videos", 64)))


def episode_items(rows, assets, group_size, seed_base, tag):
    """G independent episodes per question; seeds are a deterministic function of the ids."""
    import hashlib
    items = []
    for r in rows:
        for g in range(group_size):
            sid = f"{tag}:{r['sample_id']}:{g}"
            seed = int(hashlib.sha1(f"{seed_base}:{sid}".encode()).hexdigest()[:8], 16)
            items.append({"sample": public_sample(r), "asset": assets.get(r["video_path"]),
                          "group_id": f"{tag}:{r['sample_id']}", "trajectory_id": sid, "seed": seed})
    return items
