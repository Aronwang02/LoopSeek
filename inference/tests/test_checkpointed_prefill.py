"""Grouped training decisions checkpoint their shared prefix layer by layer (JevPolicy._checkpointed_prefill): same logits
and gradients as the uncheckpointed shared forward and as full prompts, far fewer activations held for backward, and the
model's layers and checkpointing flags restored. Tiny real Qwen3.5 hybrid model on CPU (fla blocked), in a subprocess."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def worker():
    sys.modules["fla"] = sys.modules["causal_conv1d"] = None
    sys.path.insert(0, str(ROOT))
    import torch
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    from src.common.types import DecisionInput, VisRef
    from src.policy.letters import QUESTION_MARK
    from src.policy.lora import apply_lora
    from src.policy.model import JevPolicy, grouped_log_softmax, payload_groups

    torch.set_num_threads(2)
    torch.manual_seed(5)

    class Tokenizer:
        pad_token_id, eos_token_id = 0, 1

        def encode(self, text, **kw):
            return [2 + ord(c) % 240 for c in text]

    cfg = Qwen3_5TextConfig(vocab_size=256, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                           num_attention_heads=4, num_key_value_heads=2, head_dim=8, linear_num_value_heads=4,
                           linear_num_key_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
                           linear_conv_kernel_dim=4, layer_types=["linear_attention"] * 3 + ["full_attention"])
    cfg._attn_implementation = "sdpa"
    model = Qwen3_5TextModel(cfg).float()
    policy = JevPolicy(model, Tokenizer(), vis_dim=32, tokens_per_frame=4)
    policy.lora_modules = apply_lora(model, ["q_proj", "v_proj", "in_proj_qkv", "out_proj", "up_proj", "down_proj"], 4, 8)
    for name, p in policy.named_parameters():          # non-zero LoRA B: the adapters change the prefix state
        if name.endswith("lora_B"):
            p.data.normal_(0, 0.05)
    policy.readout, policy.native, policy.prefix_sharing_train = "letters", True, True
    policy._chat, policy._slots = ("System\n<state>\n", "\nAssistant:"), [2 + ord(c) for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    policy.vision_start_id, policy.vision_end_id = 250, 251

    class Store:
        def get(self, refs):
            return torch.stack([torch.randn(4, 32, generator=torch.Generator().manual_seed(int(p) + 7)) for _, p in refs])

    policy.feature_store = Store()

    def verifier(tag, frames, groups):               # one yes/no question per inspected frame, a long frame prefix
        prefix = [f"Question {tag}. Overview and selected frames:"]
        for j in range(frames):
            prefix += [f" G{j:02d} [{j * 11.5:.1f}s] ", VisRef("video", 1000 * len(tag) + j, 0., 0., 0.)]
        prefix += ["\nMemory: kept 12 of 96 frames\n", QUESTION_MARK + "For each frame: is it evidence?\n"]
        ids = [f"E{g}{'+' if k == 0 else '-'}" for g in range(groups) for k in range(2)]
        return DecisionInput("verifier", prefix, [[f"{cid}: frame {cid[1:-1]}"] for cid in ids], ids,
                             [True] * len(ids), [{"group": i // 2, "keep": i % 2 == 0} for i in range(len(ids))])

    decisions = [verifier("A", 24, 5), verifier("BB", 31, 3)]

    def loss(z, d):
        lp = grouped_log_softmax(z[:len(d.candidates)], payload_groups(d))
        return -lp[torch.arange(0, len(d.candidates), 2)].mean()

    def run(mode):
        """mode: checkpointed (default path), plain (prefill not checkpointed), full (one full prompt per question).
        Also returns the bytes autograd keeps from the shared prefix forward outside checkpointed regions."""
        saved = [0]

        def pack(t):
            saved[0] += t.numel() * t.element_size()
            return t

        prefill = policy._letter_cached_forward

        def counted(*args, **kwargs):
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
                return prefill(*args, **kwargs)

        policy.zero_grad(set_to_none=True)
        original = policy._checkpointed_prefill
        if mode == "plain":
            policy._checkpointed_prefill = lambda enabled: original(False)
        policy._letter_cached_forward = counted
        policy.prefix_sharing_train = mode != "full"
        try:
            z, _ = policy(decisions)
            total = sum((i + 1) * loss(z[i], d) for i, d in enumerate(decisions))
        finally:
            policy._checkpointed_prefill = original
            del policy._letter_cached_forward
            policy.prefix_sharing_train = True
        total.backward()
        grads = {n: p.grad.clone() for n, p in policy.named_parameters() if p.grad is not None}
        return z.detach().clone(), grads, saved[0]

    def same(a, b):
        torch.testing.assert_close(a[0], b[0], atol=2e-5, rtol=2e-4)
        assert a[1].keys() == b[1].keys() and a[1]
        err = sum((a[1][n] - b[1][n]).square().sum() for n in a[1])
        norm = sum(g.square().sum() for g in b[1].values())
        assert float((err / norm).sqrt()) < 3e-4, float((err / norm).sqrt())

    policy.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    checkpointed, plain, full = run("checkpointed"), run("plain"), run("full")
    same(checkpointed, plain)
    same(checkpointed, full)
    assert checkpointed[2] * 10 < plain[2], (checkpointed[2], plain[2])  # the prefix no longer holds its activations
    assert model.is_gradient_checkpointing and all(l.gradient_checkpointing for l in model.layers)
    assert not any("forward" in vars(l) for l in model.layers)          # the wrappers are gone after the forward
    model.gradient_checkpointing_disable()                              # no checkpointing requested: prefill as before
    unwrapped = run("checkpointed")
    same(unwrapped, plain)
    assert unwrapped[2] == plain[2]
    print("OK", checkpointed[2], plain[2])


def test_checkpointed_prefill_matches_and_holds_less():
    r = subprocess.run([sys.executable, __file__], capture_output=True, text=True, timeout=600,
                       env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin", "HF_HUB_OFFLINE": "1"})
    assert r.returncode == 0 and "OK" in r.stdout, r.stdout + r.stderr


if __name__ == "__main__":
    worker()
