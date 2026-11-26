"""Component construction for the design-B (v3) entry points (2026-10-01).

Frames reach Jev as Qwen3.5 native vision tokens. The rollout uses ONE native token store with an online encoder
(frames not cached yet are encoded on the fly, e.g. evaluation videos), and every policy reads that same store object.
"""
from ..policy.policy_set import load_policies
from .build import native_token_store


def setup_v3(cfg, policy_arg, roles, device="cuda:0", shard="0"):
    """(native encoder, native store, {role: policy}, text_len). roles = the roles played by Jev (may be empty)."""
    enc, store = native_token_store(cfg, device=device, shard=shard)
    policies = {}
    if roles:
        policies = load_policies(cfg, policy_arg, roles, feature_store=None, device=device)
        for p in {id(p): p for p in policies.values()}.values():
            p.token_store = store
        text_len = next(iter(policies.values())).text_len
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(cfg["models"][cfg["policy"].get("qwen35_model", "qwen35_4b")]["path"])
        cache = {}

        def text_len(t):
            if t not in cache:
                cache[t] = len(tok.encode(t, add_special_tokens=False))
            return cache[t]
    return enc, store, policies, text_len
