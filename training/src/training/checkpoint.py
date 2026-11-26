"""Atomic, resumable checkpoints (update.md §9 item 2).

A checkpoint stores the trainable parameters (LoRA, projector, Choice head), the optimizer, RNG
states, trainer state (step/iteration, policy version, reference id, data position) and the base
model identity (NanoJev repo, revision, resolved commit, weight sha256) — the frozen backbone is
not duplicated (with policy.finetune='full' the backbone is trainable and therefore stored, ~2.4 GB
fp32). Writes go to a temp dir renamed into place; the previous complete one is kept.
"""
import hashlib
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch

from ..common.utils import read_json, write_json


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if s.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


def base_model_info(cfg, policy):
    if cfg["policy"]["init"] == "qwen35_4b":
        key = cfg["policy"].get("qwen35_model", "qwen35_4b")
        m = cfg["models"][key]
        repo = {"qwen35_4b": "Qwen/Qwen3.5-4B", "qwen35_9b": "Qwen/Qwen3.5-9B", "onejev_4b": "OmniJev/OneJev-4B"}.get(key, m.get("repo"))
        return {"init": "qwen35_4b", "qwen35_model": key, "repo": repo, "revision": m.get("revision"), "path": m.get("path"),
                "readout": getattr(policy, "readout", "head"),
                "lora": cfg["policy"]["lora"], "lora_modules": policy.lora_modules, "finetune": policy.finetune,
                "tokens_per_frame": policy.tokens_per_frame, "visual_mode": policy.visual_mode}
    nj = cfg["models"]["nanojev"]
    info = {"init": cfg["policy"]["init"], "nanojev_repo": "C-Tianyu/NanoJev",
            "nanojev_revision": nj.get("revision"), "nanojev_path": nj.get("path"),
            "resolved_commit": Path(nj["path"]).name if nj.get("path") else None,
            "lora": cfg["policy"]["lora"], "lora_modules": policy.lora_modules, "finetune": policy.finetune,
            "tokens_per_frame": policy.tokens_per_frame, "visual_mode": policy.visual_mode}
    w = Path(nj["path"]) / "best.safetensors" if nj.get("path") else None
    if w is not None and w.is_symlink():
        info["weights_sha256"] = Path(os.readlink(w)).name     # HF blob name = sha256 of the file
    return info


def save_checkpoint(ckpt_dir, policy, optimizer=None, state=None, base_info=None, keep_previous=True):
    ckpt_dir = Path(ckpt_dir)
    final, prev = ckpt_dir / "latest", ckpt_dir / "previous"
    tmp = ckpt_dir / f"tmp.{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    torch.save(policy.trainable_state_dict(), tmp / "trainable.pt")
    torch.save({"optimizer": optimizer.state_dict() if optimizer else None, "rng": rng_state(),
                "state": state or {}}, tmp / "trainer.pt")
    write_json(tmp / "base_model.json", base_info or {})
    (tmp / "COMPLETE").write_text("ok\n")
    if final.exists():
        if keep_previous:
            if prev.exists():
                shutil.rmtree(prev)
            final.rename(prev)
        else:
            shutil.rmtree(final)
    tmp.rename(final)
    return final


def complete_checkpoints(ckpt_dir):
    """The complete checkpoints of a run, newest first (latest, then previous)."""
    return [d for d in (Path(ckpt_dir) / name for name in ("latest", "previous")) if (d / "COMPLETE").exists()]


def latest_complete(ckpt_dir):
    found = complete_checkpoints(ckpt_dir)
    return found[0] if found else None


def resolve_ckpt(path):
    """A checkpoint dir, or a .txt pointer file holding one (e.g. runs/x/best_snapshot.txt)."""
    path = Path(path)
    if path.suffix == ".txt":
        path = Path(path.read_text().strip())
    return path


def checkpoint_finetune(path):
    """'lora' or 'full' as recorded in base_model.json; checkpoints without the field are LoRA."""
    path = resolve_ckpt(path)
    f = path / "base_model.json"
    return (read_json(f).get("finetune") or "lora") if f.exists() else "lora"


def load_policy_weights(path, policy):
    path = resolve_ckpt(path)
    info = read_json(path / "base_model.json") if (path / "base_model.json").exists() else {}
    mode = info.get("finetune") or "lora"
    if mode != policy.finetune:
        raise ValueError(f"checkpoint {path} was trained with finetune={mode!r} but the policy was built "
                         f"with finetune={policy.finetune!r} (build it with policy.finetune={mode})")
    if info.get("lora_modules") and info["lora_modules"] != policy.lora_modules:
        raise ValueError("checkpoint LoRA modules differ from the current policy")
    policy.load_trainable_state_dict(torch.load(path / "trainable.pt", map_location="cpu"))
    return info


def load_checkpoint(path, policy, optimizer=None, restore_rng=True):
    path = resolve_ckpt(path)
    load_policy_weights(path, policy)
    tr = torch.load(Path(path) / "trainer.pt", map_location="cpu", weights_only=False)
    if optimizer is not None and tr["optimizer"] is not None:
        optimizer.load_state_dict(tr["optimizer"])
    if restore_rng:
        set_rng_state(tr["rng"])
    return tr["state"]


def trainable_fingerprint(policy):
    """sha1 over trainable tensors: proves which parameters a record was produced with."""
    h = hashlib.sha1()
    for k, v in sorted(policy.trainable_state_dict().items()):
        h.update(k.encode())
        h.update(v.float().numpy().tobytes())
    return h.hexdigest()[:16]
