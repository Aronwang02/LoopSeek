"""2026-11-05 speedup E1 (policy.merge_lora_per_role): every role keeps its own folded LoRA weight, so a role switch does not
re-fold. Checks on CPU with two roles sharing one frozen bf16 base: (1) per-role forwards equal the shared-buffer (A1)
forwards bit for bit and the unmerged forwards within bf16 rounding, (2) switching back and forth folds nothing,
(3) release_merged and an in-place adapter update invalidate the own fold."""
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.policy import lora as L  # noqa: E402


def _roles():
    torch.manual_seed(0)
    base = nn.Linear(64, 48, bias=True).to(torch.bfloat16)
    roles = []
    for seed in (1, 2):
        torch.manual_seed(seed)
        m = L.LoRALinear(base, r=4, alpha=8)
        nn.init.normal_(m.lora_B, std=0.05)          # nonzero adapters, different per role
        roles.append(nn.Sequential(m))
    return base, roles


def _run(roles, x, per_role):
    L._MERGED.clear()
    L.set_merge_per_role(per_role)
    out = []
    with torch.no_grad():
        for _ in range(2):
            for i, r in enumerate(roles):
                L.ensure_merged(r, ("role", i))
                out.append(r(x))
    return out


def test_e1_matches_shared_and_unmerged():
    _, roles = _roles()
    x = torch.randn(5, 64, dtype=torch.bfloat16)
    with torch.no_grad():
        for r in roles:
            for m in L.lora_modules(r):
                m.__dict__["_own_token"] = None
        ref = [r(x) for r in roles]                   # unmerged path (no fold yet)
    shared = _run(roles, x, per_role=False)
    for r in roles:
        L.release_merged(r)
    own = _run(roles, x, per_role=True)
    for a, b in zip(shared, own):
        assert torch.equal(a, b)                      # same fold arithmetic -> identical outputs
    for i, o in enumerate(own):
        assert torch.allclose(o.float(), ref[i % 2].float(), atol=3e-2, rtol=3e-2)
    L.set_merge_per_role(False)


def test_e1_no_refold_on_switch_and_invalidation():
    _, roles = _roles()
    L._MERGED.clear()
    L.set_merge_per_role(True)
    with torch.no_grad():
        assert L.ensure_merged(roles[0], ("role", 0)) == 1
        assert L.ensure_merged(roles[1], ("role", 1)) == 1
        assert L.ensure_merged(roles[0], ("role", 0)) == 0   # switching back: nothing folded
        assert L.ensure_merged(roles[1], ("role", 1)) == 0
        L.release_merged(roles[0])
        m0 = L.lora_modules(roles[0])[0]
        assert m0.__dict__.get("_own_token") is None          # released: forwards use the unmerged path
        assert L.ensure_merged(roles[0], ("role", 0)) == 1   # folded again on the next forward
        m0.lora_A.add_(0.01)                                  # in-place update bumps _version -> refold
        assert L.ensure_merged(roles[0], ("role", 0)) == 1
        x = torch.randn(3, 64, dtype=torch.bfloat16)
        y_own = roles[0](x)
        L.release_merged(roles[0])
        y_ref = roles[0](x)
        assert torch.allclose(y_own.float(), y_ref.float(), atol=3e-2, rtol=3e-2)
    L.set_merge_per_role(False)


if __name__ == "__main__":
    test_e1_matches_shared_and_unmerged()
    test_e1_no_refold_on_switch_and_invalidation()
    print("ok")
