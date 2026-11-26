"""Masked Choice distribution, forced actions, exact input reconstruction, LoRA/projector/head
gradient paths (frozen base untouched), and dependence on the visual input."""
import json

import pytest
import torch

from conftest import StubEncoder, tiny_backbone
from src.common.types import DecisionInput, DecisionRecord
from src.env.environment import Episode, PublicSample
from src.policy.lora import apply_lora
from src.policy.model import JevPolicy, masked_log_softmax
from src.policy.sampler import decide

SAMPLE = PublicSample("s1", "v1", "unused", "What is shown?", {"A": "red", "B": "green", "C": "blue", "D": "gray"})


def make_episode(cfg, asset, features):
    return Episode(SAMPLE, asset, features, StubEncoder(), cfg, key="t0").reset()


def test_masked_softmax_normalization():
    logits = torch.tensor([[1.0, float("-inf"), 2.0, 0.5], [float("-inf")] * 4])
    p = masked_log_softmax(logits).exp()
    assert torch.allclose(p[0].sum(), torch.tensor(1.0)) and p[0, 1] == 0
    assert torch.isinf(masked_log_softmax(logits)[1]).all()


def test_lora_zero_init_is_identity(tokenizer, cfg, asset, features):
    """B=0 LoRA leaves the function unchanged (so the untrained policy = NanoJev + projector)."""
    torch.manual_seed(0)
    a = JevPolicy(tiny_backbone(len(tokenizer)), tokenizer, tokens_per_frame=4)
    torch.manual_seed(0)
    b = JevPolicy(tiny_backbone(len(tokenizer)), tokenizer, tokens_per_frame=4)
    names = apply_lora(b.backbone, cfg["policy"]["lora"]["targets"], 16, 32, 0.0)
    assert len(names) == 2 * 7          # 2 layers x (q,k,v,o,gate,up,down)
    b.load_state_dict({k: v for k, v in a.state_dict().items()}, strict=False)
    a.feature_store = b.feature_store = features
    ep = make_episode(cfg, asset, features)
    inp = ep.controller_input(a.text_len)
    with torch.no_grad():
        za, _ = a([inp])
        zb, _ = b([inp])
    assert torch.allclose(za, zb, atol=1e-6)
    trainable = {n for n, p in b.named_parameters() if p.requires_grad}
    assert all(("lora_" in n) or n.startswith(("projector.", "head.")) for n in trainable)
    assert not any(n.startswith("backbone.") and "lora_" not in n for n in trainable)


def test_sampling_legal_and_forced(cfg, asset, features, tiny_policy):
    ep = make_episode(cfg, asset, features)
    inp = ep.controller_input(tiny_policy.text_len)
    assert inp.candidate_ids == ["REFINE_LOCAL", "SWITCH_LOCAL", "UP", "INSPECT_FRAMES"]
    masked = DecisionInput("controller", inp.prefix, inp.candidates, inp.candidate_ids, [True, False, False, True])
    res = decide(tiny_policy, [masked] * 64, [torch.Generator().manual_seed(i) for i in range(64)])
    assert all(r["index"] not in (1, 2) and r["probs"][1] == 0.0 and abs(sum(r["probs"]) - 1) < 1e-5 for r in res)
    forced = DecisionInput("controller", inp.prefix, inp.candidates, inp.candidate_ids, [False, False, False, True])
    r = decide(tiny_policy, [forced], [torch.Generator()])[0]
    assert r["forced"] and r["index"] == 3 and r["log_prob"] == 0.0


def test_visual_tokens_live_in_prefix(cfg, asset, features, tiny_policy):
    ep = make_episode(cfg, asset, features)
    wins = ep.localize_windows("global")
    inp = ep.localize_input("global", wins, tiny_policy.text_len)
    n_refs = sum(1 for s in inp.prefix if not isinstance(s, str))
    assert n_refs == len(ep.global_idx)
    assert all(all(isinstance(x, str) for x in c) for c in inp.candidates)   # candidates are text only
    _, _, _, st = tiny_policy.build_paths([inp])
    assert st["vis_tokens"] == 4 * n_refs * len(wins)


def test_record_roundtrip_reconstructs_logprob(cfg, asset, features, tiny_policy):
    ep = make_episode(cfg, asset, features)
    wins = ep.localize_windows("global")
    inp = ep.localize_input("global", wins, tiny_policy.text_len)
    r = decide(tiny_policy, [inp], [torch.Generator().manual_seed(3)], autocast_dtype=None)[0]
    rec = DecisionRecord(0, 0, "localize", None, inp, r["index"], inp.candidate_ids[r["index"]],
                         r["log_prob"], r["probs"], r["entropy"], len(inp.legal_indices()), r["forced"], True, 0)
    back = DecisionRecord.from_json(json.loads(json.dumps(rec.to_json())))
    assert back.input.candidate_ids == inp.candidate_ids and back.input.valid == inp.valid
    logits, _ = tiny_policy([back.input])
    assert masked_log_softmax(logits)[0, back.action_index].item() == pytest.approx(back.old_log_prob, abs=1e-5)


def test_gradients_reach_lora_projector_head_only(cfg, asset, features, tiny_policy):
    ep = make_episode(cfg, asset, features)
    ins = [ep.controller_input(tiny_policy.text_len)]
    wins = ep.localize_windows("global")
    ins.append(ep.localize_input("global", wins, tiny_policy.text_len))
    ins.append(ep.frame_input(ep.frame_candidates(), [], tiny_policy.text_len))
    logits, _ = tiny_policy(ins)
    lp = masked_log_softmax(logits)
    (-sum(lp[i, d.legal_indices()[0]] for i, d in enumerate(ins))).backward()
    assert tiny_policy.head.scalar.weight.grad.abs().sum() > 0
    assert tiny_policy.projector.fc1.weight.grad.abs().sum() > 0      # pixels -> encoder -> projector
    q = tiny_policy.backbone.layers[0].self_attn.q_proj
    assert q.lora_A.grad.abs().sum() > 0 and q.base.weight.grad is None


def test_visual_input_changes_decisions(cfg, asset, features, tiny_policy):
    ep = make_episode(cfg, asset, features)
    inp = ep.localize_input("global", ep.localize_windows("global"), tiny_policy.text_len)
    store, cid = tiny_policy.feature_store, asset.content_id
    with torch.no_grad():
        a, _ = tiny_policy([inp])
        table = store._load(cid)
        keys = sorted(table)
        store._mem[cid] = {k: table[keys[(i + 5) % len(keys)]] for i, k in enumerate(keys)}
        b, _ = tiny_policy([inp])
        store._mem[cid] = table
    assert not torch.allclose(a, b) and torch.equal(torch.isinf(a), torch.isinf(b))
