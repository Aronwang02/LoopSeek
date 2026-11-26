"""NanoJev models per decision role (controller / localize / frame): one per distinct checkpoint.

A checkpoint argument is either one checkpoint (a directory or a best_snapshot.txt pointer, shared by
every role: the original design) or a JSON policy set {"controller": ckpt, "localize": ckpt, "frame": ckpt}
written by scripts/make_policy_set.py. Roles that list the same checkpoint share one model (e.g. one
grounding model for localize + frame and a separate controller). Each model is built in the finetune mode
(lora / full) its checkpoint records.
"""
import json
from pathlib import Path

from ..training.checkpoint import checkpoint_finetune, load_policy_weights, resolve_ckpt
from .model import build_policy, share_frozen_backbone

ROLES = ("controller", "localize", "frame", "verifier")
CORE_ROLES = ROLES[:3]          # a policy set must list these; "verifier" is optional (new design, 2026-09-30)


def read_policy_set(arg):
    """{role: checkpoint} for a policy-set .json; None for one checkpoint (or none)."""
    if arg is None or not str(arg).endswith(".json"):
        return None
    ps = json.loads(Path(arg).read_text())
    unknown = set(ps) - set(ROLES)
    if unknown:
        raise ValueError(f"policy set {arg}: unknown roles {sorted(unknown)}")
    return ps


def role_checkpoint(arg, role):
    """The checkpoint a role starts from: the set's entry for a policy set, else `arg` itself."""
    ps = read_policy_set(arg)
    if ps is None:
        return arg
    if role not in ps:
        raise ValueError(f"policy set {arg} has no checkpoint for role {role!r}")
    return ps[role]


def load_policies(cfg, arg, roles, feature_store=None, device="cpu", build=build_policy):
    """{role: JevPolicy} for `roles`. One checkpoint, or none (untrained policy): the same object for
    every role. A policy set: one model per distinct checkpoint, shared by the roles that list it."""
    def make(ckpt):
        mode = checkpoint_finetune(ckpt) if ckpt else cfg["policy"].get("finetune", "lora")
        pol = build({**cfg, "policy": {**cfg["policy"], "finetune": mode}}, feature_store=feature_store,
                    device=device).eval()
        if ckpt:
            load_policy_weights(ckpt, pol)
        return pol
    if read_policy_set(arg) is None:
        shared = make(arg)
        return {r: shared for r in roles}
    models, out = {}, {}
    ps = read_policy_set(arg)
    for r in roles:
        if r not in ps and r not in CORE_ROLES:      # optional role absent from the set: not loaded
            continue
        ck = str(resolve_ckpt(role_checkpoint(arg, r)).resolve())
        if ck not in models:
            pol = make(ck)
            # Every role is built from cfg's same base snapshot. Keep separate
            # adapters while avoiding one 4B frozen-weight copy per LoRA role.
            source = next((p for p in models.values() if p.finetune == 'lora'), None)
            if source is not None:
                share_frozen_backbone(pol, source)
            models[ck] = pol
        out[r] = models[ck]
    return out
