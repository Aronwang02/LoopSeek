"""Minimal LoRA for the NanoJev backbone (update.md §2, §6: rank 16, alpha 32, dropout 0).

y = W x + (alpha / r) * B A x, with W frozen, A ~ kaiming-uniform, B = 0 at init, so a freshly
wrapped model computes exactly the original function.
"""
import math
import weakref

import torch
from torch import nn

# policy.merge_lora_inference (2026-11-03, speedup A1): base weight -> {"buf": bf16 W + scaling * B A, "module": the
# LoRALinear whose adapter is in buf, "token": its (policy id, merge version)}. One buffer per base weight: the roles
# (controller / grounding / verifier) share the frozen base (model.share_frozen_backbone), so they share the buffer and
# ensure_merged re-folds it whenever another role's forward starts. Keyed by id(base Parameter); a weak reference
# guards against a reused id (tensors cannot be WeakKeyDictionary keys: their == is elementwise).
_MERGED = {}
# policy.merge_lora_per_role (2026-11-05, speedup E1): each LoRALinear (one per role on the shared frozen base) keeps its OWN
# folded bf16 weight (module.__dict__["_own_buf"], not a registered buffer: state dicts are unchanged), so a role switch costs
# nothing; at batch 1 every decision switches role (16.4 switches / MLVU question) and the shared buffer was re-folded each
# time (248 weights through fp32 temporaries). Same fold arithmetic as the shared buffer -> identical forwards; memory: one
# bf16 copy of the adapted weights per role.
_PER_ROLE = {"on": False}


def set_merge_per_role(flag):
    _PER_ROLE["on"] = bool(flag)


def merged_entry(w):
    e = _MERGED.get(id(w))
    return e if e is not None and e["ref"]() is w else None


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.r, self.scaling = r, alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        if not torch.is_grad_enabled():
            own = self.__dict__.get("_own_buf")
            if own is not None and self.__dict__.get("_own_token") is not None:   # this role's own fold (E1)
                return nn.functional.linear(x, own, self.base.bias)
            e = merged_entry(self.base.weight)
            if e is not None and e["module"] is self:      # this module's adapter is folded in (ensure_merged)
                return nn.functional.linear(x, e["buf"], self.base.bias)
        out = self.base(x)
        a = self.lora_A.to(x.dtype) if not torch.is_autocast_enabled() else self.lora_A
        b = self.lora_B.to(x.dtype) if not torch.is_autocast_enabled() else self.lora_B
        return out + nn.functional.linear(nn.functional.linear(self.drop(x), a), b) * self.scaling


def apply_lora(model: nn.Module, targets, r, alpha, dropout=0.0):
    """Wrap every nn.Linear whose attribute name is in `targets`; freeze everything else in `model`.
    Returns the sorted list of wrapped module names (recorded in the run config)."""
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                setattr(module, child_name, LoRALinear(child, r, alpha, dropout))
                wrapped.append(f"{name}.{child_name}" if name else child_name)
    return sorted(wrapped)


def lora_parameters(model):
    return [p for n, p in model.named_parameters() if "lora_" in n]


def lora_modules(model):
    mods = model.__dict__.get("_lora_mods")
    if mods is None:
        mods = [m for m in model.modules() if isinstance(m, LoRALinear)]
        model.__dict__["_lora_mods"] = mods
    return mods


@torch.no_grad()
def ensure_merged(model, token):
    """Fold every LoRA adapter of `model` into its base weight's shared bf16 buffer unless it is already there for this
    token (policy id, merge version). buf = round(W + scaling * B @ A) computed in fp32; the module then runs one
    F.linear on buf in no-grad forwards (LoRALinear.forward). With gradients the original base + B A path runs."""
    mods = lora_modules(model)
    if not mods:
        return 0
    # in-place updates (optimizer steps, weight loading) bump a tensor's _version: any change re-folds
    token = (token, sum(m.lora_A._version + m.lora_B._version + m.base.weight._version for m in mods))
    if _PER_ROLE["on"]:
        if mods[0].__dict__.get("_own_token") == token:
            return 0                   # this role's adapters are already folded into its own buffers
        for m in mods:
            w = m.base.weight
            buf = m.__dict__.get("_own_buf")
            if buf is None or buf.shape != w.shape or buf.dtype != w.dtype or buf.device != w.device:
                buf = m.__dict__["_own_buf"] = torch.empty_like(w)
            buf.copy_((w.float() + m.scaling * (m.lora_B.float() @ m.lora_A.float())).to(buf.dtype))
            m.__dict__["_own_token"] = token
        return len(mods)
    first = merged_entry(mods[0].base.weight)
    if first is not None and first["module"] is mods[0] and first["token"] == token:
        return 0                       # all adapters of one model are always folded together
    for m in mods:
        w = m.base.weight
        e = merged_entry(w)
        if e is None:
            e = _MERGED[id(w)] = {"ref": weakref.ref(w), "buf": torch.empty_like(w), "module": None, "token": None}
        e["buf"].copy_((w.float() + m.scaling * (m.lora_B.float() @ m.lora_A.float())).to(e["buf"].dtype))
        e["module"], e["token"] = m, token
    return len(mods)


def release_merged(model):
    """Detach `model`'s adapters from the merged buffers (called whenever its weights may change: train mode, weight
    loading, static-cache invalidation), so no forward can read a fold made before an update."""
    for m in model.__dict__.get("_lora_mods") or []:
        m.__dict__["_own_token"] = None            # E1: an own fold made before an update is never used
        e = merged_entry(m.base.weight)
        if e is not None and e["module"] is m:
            e["module"], e["token"] = None, None
