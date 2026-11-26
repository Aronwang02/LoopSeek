"""2026-11-03 inference speedups on a tiny real Qwen3.5 hybrid model (CPU kernels, fla blocked), each against the
complete-prompt forward (_forward_letters_full) or the unmerged adapters:
  A2 policy.one_pass_single   single-question decisions with a static snapshot run one forward (prefix rest + suffix)
  A4 policy.suffix_batch_max  question suffixes in sub-batches, each on its own copy of the prefix-cache rows
  A3 (always on)              prefix positions once / text suffixes continue from the next position; logits in one go
  A1 policy.merge_lora        eval no-grad forwards on LoRA folded into shared base-weight buffers; switching roles,
                              in-place adapter updates and train mode all fall back to / rebuild the right weights"""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj",
           "gate_proj", "up_proj", "down_proj"]


def worker():
    sys.modules["fla"] = sys.modules["causal_conv1d"] = None
    sys.path.insert(0, str(ROOT))
    import torch
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    from src.common.types import DecisionInput, VisRef
    from src.policy.letters import QUESTION_MARK
    from src.policy.lora import _MERGED, apply_lora, lora_modules, merged_entry
    from src.policy.model import JevPolicy, frozen_copy

    torch.set_num_threads(2)

    class Tokenizer:
        pad_token_id, eos_token_id = 0, 1

        def encode(self, text, **kw):
            return [2 + ord(c) % 240 for c in text]

    cfg = Qwen3_5TextConfig(vocab_size=256, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                           num_attention_heads=4, num_key_value_heads=2, head_dim=8, linear_num_value_heads=4,
                           linear_num_key_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
                           linear_conv_kernel_dim=4, layer_types=["linear_attention"] * 3 + ["full_attention"])
    cfg._attn_implementation = "sdpa"

    class Store:
        def get(self, refs):
            return torch.stack([torch.randn(4, 32, generator=torch.Generator().manual_seed(int(pts) + 19))
                                for _, pts in refs])

    def make(seed, lora=False):
        torch.manual_seed(seed)
        model = Qwen3_5TextModel(cfg).float()
        policy = JevPolicy(model, Tokenizer(), vis_dim=32, tokens_per_frame=4)
        policy.readout, policy.native = "letters", True
        policy._chat, policy._slots = ("System\n<state>\n", "\nAssistant:"), [2 + ord(c) for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
        policy.vision_start_id, policy.vision_end_id = 250, 251
        policy.feature_store = Store()
        policy.prefix_sharing = policy.static_cache = True
        if lora:
            policy.lora_modules = apply_lora(policy.backbone, TARGETS, 4, 8.0)
            policy.finetune = "lora"
            for n, p in policy.named_parameters():
                if "lora_" in n:
                    p.data.normal_(0, 0.3)
        return policy.eval()

    def decision(tag, groups=0, memory="seen 1 window", pts=0):
        prefix = [f"Video question {tag}; duration 7200s. Overview: ", VisRef("video", pts, 0., 0., 0.),
                  f"\nMemory: {memory}\nFrames: ", VisRef("video", pts + 100, 0., 0., 0.), "\n",
                  QUESTION_MARK + "Choose?\n"]
        ids = [f"F{i}" for i in range(groups * 2)] if groups else ["UP", "GO", "STOP"]
        payloads = [{"group": i // 2, "keep": i % 2 == 0} for i in range(len(ids))] if groups else None
        d = DecisionInput("verifier" if groups else "controller", prefix,
                          [[f"{cid}: option {i}"] for i, cid in enumerate(ids)], ids,
                          [True] * len(ids) if groups else [True, False, True], payloads)
        d.static_segments = 2
        return d

    ds = [decision("A", pts=1), decision("B", memory="seen two quite different windows", pts=2),
          decision("C", groups=5, pts=3), decision("D", groups=1, pts=4)]
    tol = dict(atol=2e-5, rtol=2e-4)

    # ---- A2 + A4 + A3 against the complete prompts
    policy = make(17)
    calls = []
    original = policy.backbone.forward

    def traced(*args, **kw):
        calls.append(kw["inputs_embeds"].shape[0])
        return original(*args, **kw)

    policy.backbone.forward = traced
    with torch.no_grad():
        full, _ = policy._forward_letters_full(ds)
        policy.clear_static_cache()
        old, _ = policy(ds)                      # flags off: the former two-pass path, all rows at once
        torch.testing.assert_close(old, full, **tol)
        policy.one_pass_single, policy.suffix_batch_max = True, 2
        policy.clear_static_cache()
        calls.clear()
        fast, _ = policy(ds)
        torch.testing.assert_close(fast, full, **tol)
        # 4 cold static heads (one batch), A / B / D one pass each, C: its dynamic prefix + suffix rows 2 + 2 + 1
        assert calls == [4, 1, 1, 1, 1, 2, 2, 1], calls
        calls.clear()
        again, _ = policy(ds)                    # warm snapshots are never written to
        assert calls == [1, 1, 1, 1, 2, 2, 1], calls
        torch.testing.assert_close(again, fast)
        policy.suffix_batch_max = 64             # larger than the rows: one suffix forward, the old repeat
        calls.clear()
        wide, _ = policy(ds)
        assert calls == [1, 1, 1, 1, 5], calls
        torch.testing.assert_close(wide, full, **tol)
        # V1 + V2: suffixes sorted by length; full-attention layers read the prefix KV in place (CPU: the exact
        # concatenated-attention reference path of _shared_prefix_attention_forward)
        from src.policy import model as M
        impl = policy.backbone.config._attn_implementation
        for sb, want in ((2, [1, 1, 1, 1, 2, 2, 1]), (64, [1, 1, 1, 1, 5])):
            policy.sort_suffixes = policy.shared_prefix_attn = True
            policy.suffix_batch_max = sb
            calls.clear()
            v2, _ = policy(ds)
            assert calls == want, (sb, calls)
            torch.testing.assert_close(v2, full, **tol)
            assert policy.backbone.config._attn_implementation == impl and not M._SHARED_CTX
        policy.sort_suffixes = True               # V1 alone (copied prefix rows, sorted)
        policy.shared_prefix_attn = False
        torch.testing.assert_close(policy(ds)[0], full, **tol)
        policy.sort_suffixes = policy.shared_prefix_attn = False
    print("A2/A3/A4 match the complete prompts")
    print("V1/V2 match the complete prompts")

    # ---- A1: two roles with different adapters on one shared base
    actor = make(29, lora=True)
    ref = frozen_copy(actor)                     # shares the frozen base weights, own adapters
    with torch.no_grad():
        for n, p in ref.named_parameters():
            if "lora_" in n:
                p.normal_(0, 0.3)
    for p in (actor, ref):
        p.one_pass_single, p.suffix_batch_max = True, 3
    with torch.no_grad():
        za0, _ = actor(ds)
        zb0, _ = ref(ds)
        assert not len(_MERGED)                  # merging is opt-in
        actor.merge_lora = ref.merge_lora = True
        actor.clear_static_cache(), ref.clear_static_cache()
        za1, _ = actor(ds)
        mods = lora_modules(actor.backbone)
        assert len(_MERGED) == len(mods) and all(merged_entry(m.base.weight)["module"] is m for m in mods)
        zb1, _ = ref(ds)                         # re-folds the shared buffers with the other adapters
        assert all(merged_entry(m.base.weight)["module"] is not m for m in mods)
        za2, _ = actor(ds)
        torch.testing.assert_close(za1, za0, **tol)
        torch.testing.assert_close(zb1, zb0, **tol)
        torch.testing.assert_close(za2, za0, **tol)
        fin = torch.isfinite(za0)              # illegal candidates are -inf in every run
        assert (za0 - zb0)[fin].abs().max() > 1e-2   # the two adapters really differ
        # an in-place adapter update (an optimizer step) must re-fold even without train() / clear_static_cache();
        # the static snapshots would be stale too (in training train() clears them), so this check runs uncached
        actor.static_cache = False
        for m in mods:
            m.lora_B.mul_(1.5)
        za3, _ = actor(ds)
        assert all(merged_entry(m.base.weight)["module"] is m for m in mods)
        actor.merge_lora = False
        za3_ref, _ = actor(ds)
        torch.testing.assert_close(za3, za3_ref, **tol)
        assert (za3 - za0)[fin].abs().max() > 1e-3
        actor.merge_lora, actor.static_cache = True, True
    actor.train()                                # train mode: never the merged buffers, gradients reach the adapters
    assert all(merged_entry(m.base.weight)["module"] is not m for m in mods)
    zt, _ = actor(ds[:1])
    zt[0, 0].backward()
    assert any(m.lora_A.grad is not None and m.lora_A.grad.abs().sum() > 0 for m in mods)
    actor.eval()
    print("A1 merged adapters match the unmerged ones")


def test_speedups_1103():
    result = subprocess.run([sys.executable, __file__, "--worker"], cwd=ROOT, text=True, capture_output=True,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"}, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "A2/A3/A4 match" in result.stdout and "A1 merged adapters match" in result.stdout
    assert "V1/V2 match" in result.stdout


if __name__ == "__main__":
    worker()
