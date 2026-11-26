"""Cross-decision batching on a tiny real Qwen3.5 hybrid model, including native image positions and cache reuse."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def worker():
    # Exercise the same model/cache classes with CPU kernels, without downloading weights or allocating a GPU.
    sys.modules["fla"] = sys.modules["causal_conv1d"] = None
    sys.path.insert(0, str(ROOT))
    import copy
    import torch
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
    from src.common.types import DecisionInput, VisRef
    from src.policy.model import JevPolicy, grouped_log_softmax, payload_groups
    from src.policy.sampler import chunk_decisions, decide
    from src.policy.letters import QUESTION_MARK

    torch.set_num_threads(2)
    torch.manual_seed(17)

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
    policy.readout, policy.native = "letters", True
    policy._chat, policy._slots = ("System\n<state>\n", "\nAssistant:"), [2 + ord(c) for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
    policy.vision_start_id, policy.vision_end_id = 250, 251

    class Store:
        def get(self, refs):
            return torch.stack([torch.randn(4, 32, generator=torch.Generator().manual_seed(int(pts) + 19))
                                for _, pts in refs])

    policy.feature_store = Store()

    def decision(tag, groups=0, memory="seen 1 window", pts=0):
        prefix = [f"Video question {tag}; duration 7200s. Overview: ", VisRef("video", pts, 0., 0., 0.),
                  f"\nMemory: {memory}\n", QUESTION_MARK + "Choose?\n"]
        ids = [f"F{i}" for i in range(groups * 2)] if groups else ["UP", "GO", "STOP"]
        payloads = [{"group": i // 2, "keep": i % 2 == 0} for i in range(len(ids))] if groups else None
        d = DecisionInput("verifier" if groups else "controller", prefix, [[f"{cid}: option {i}"] for i, cid in enumerate(ids)],
                          ids, [True] * len(ids) if groups else [True, False, True], payloads)
        d.static_segments = 2
        return d

    ds = [decision("A", pts=1), decision("B", memory="seen two quite different windows", pts=2),
          decision("C", groups=2, pts=3)]
    calls = []
    original = model.forward

    def traced(*args, **kw):
        calls.append((kw["inputs_embeds"].shape[0], bool(kw.get("use_cache"))))
        return original(*args, **kw)

    model.forward = traced

    def loss(z, d):
        groups = payload_groups(d)
        if groups is not None:
            lp = grouped_log_softmax(z[:len(d.candidates)], groups)
            return -lp[torch.arange(0, len(d.candidates), 2)].mean()
        return -z[:len(d.candidates)].log_softmax(-1)[0]

    def logits_and_grads(decisions, batch):
        policy.zero_grad(set_to_none=True)
        if batch:
            z, _ = policy(decisions)
            total = sum((i + 1) * loss(z[i], d) for i, d in enumerate(decisions))
            values = [z[i, :len(d.candidates)].detach().clone() for i, d in enumerate(decisions)]
            total.backward()
        else:
            values = []
            for i, d in enumerate(decisions):
                z, _ = policy([d])
                values.append(z[0].detach().clone())
                ((i + 1) * loss(z[0], d)).backward()
        return values, {n: p.grad.clone() for n, p in policy.named_parameters() if p.grad is not None}

    def compare(a, b):
        for x, y in zip(a[0], b[0]):
            torch.testing.assert_close(x, y, atol=2e-5, rtol=2e-4)
        assert a[1].keys() == b[1].keys()
        err = sum((a[1][n] - b[1][n]).square().sum() for n in a[1])
        norm = sum(g.square().sum() for g in a[1].values())
        assert float((err / norm).sqrt()) < 3e-4, float((err / norm).sqrt())

    policy.train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    reference = logits_and_grads(ds, False)
    calls.clear()
    batched = logits_and_grads(ds, True)
    # The first call handles all 4 prompts; checkpoint recomputations do not call model.forward.
    assert calls == [(4, False)], calls
    compare(reference, batched)
    assert model.is_gradient_checkpointing

    # Shared training: unequal question counts repeat the correct row's KV and recurrent state.
    policy.prefix_sharing_train = True
    grouped = [decision("X", groups=2, pts=4), decision("Y", groups=3, pts=5)]
    reference = logits_and_grads(grouped, False)
    calls.clear()
    batched = logits_and_grads(grouped, True)
    assert calls == [(2, True), (5, True)], calls
    compare(reference, batched)
    assert model.is_gradient_checkpointing

    # Verifier SFT may mix grouped and single-question rows in one microbatch.
    compare(logits_and_grads(ds, False), logits_and_grads(ds, True))
    assert model.is_gradient_checkpointing

    policy.eval()
    policy.static_cache = True
    cached = [decision("X", groups=2, pts=4), decision("Y", pts=5), decision("X", groups=3, pts=4)]
    policy.clear_static_cache()
    with torch.no_grad():
        calls.clear()
        z, _ = policy(cached)
        # Two distinct static heads, three independent dynamic states, six question suffixes.
        assert calls == [(2, True), (3, True), (6, True)], calls
        assert policy.static_cache_stats == {"hit": 1, "miss": 2}
        snapshots = copy.deepcopy(policy._static_kv)
        full, _ = policy._forward_letters_full(cached)
        torch.testing.assert_close(z, full, atol=2e-5, rtol=2e-4)
        calls.clear()
        again, _ = policy(cached)
        assert calls == [(3, True), (6, True)], calls
        torch.testing.assert_close(again, z)
        assert policy.static_cache_stats == {"hit": 4, "miss": 2}
        for key, snap in snapshots.items():
            for a, b in zip(snap.layers, policy._static_kv[key].layers):
                for attr in ("keys", "values", "conv_states", "recurrent_states"):
                    x, y = getattr(a, attr, None), getattr(b, attr, None)
                    if isinstance(x, dict):
                        for k in x:
                            torch.testing.assert_close(x[k], y[k])
                    elif torch.is_tensor(x):
                        torch.testing.assert_close(x, y)
        # A longer dynamic prefix must use a separate cache-compatible batch and retain exact image positions.
        varied = [cached[0], decision("Y", memory="a longer memory and another window", pts=5)]
        z, _ = policy(varied)
        full, _ = policy._forward_letters_full(varied)
        torch.testing.assert_close(z, full, atol=2e-5, rtol=2e-4)
        # LRU eviction inside one batch must not invalidate its pending snapshots.
        policy.static_cache_max = 1
        policy.clear_static_cache()
        z, _ = policy(cached)
        full, _ = policy._forward_letters_full(cached)
        torch.testing.assert_close(z, full, atol=2e-5, rtol=2e-4)
        assert len(policy._static_kv) == 1

        # Exact inference dedup must retain every row's legal slots and sampling RNG.
        policy.deduplicate_inference = True
        repeated = [cached[0], cached[0], cached[1], cached[0]]
        policy.clear_static_cache()
        got, stats = policy(repeated)
        assert stats['deduplicated_decisions'] == 2
        full, _ = policy._forward_letters_full(repeated)
        torch.testing.assert_close(got, full, atol=2e-5, rtol=2e-4)
        gen = lambda: [torch.Generator().manual_seed(800 + i) for i in range(len(repeated))]
        together = decide(policy, repeated, generators=gen(), mode='sample', autocast_dtype=None)
        policy.deduplicate_inference = False
        alone = decide(policy, repeated, generators=gen(), mode='sample', autocast_dtype=None)
        assert [r['index'] for r in together] == [r['index'] for r in alone]
        assert policy._static_kv
        policy.train()
        assert not policy._static_kv
        policy.eval()
        policy([cached[0]])
        policy.load_trainable_state_dict(policy.trainable_state_dict())
        assert not policy._static_kv

        # SFT dev must not reuse a snapshot produced before the latest optimizer update.
        from scripts import train_sft as T
        one = cached[1]
        policy.clear_static_cache()
        policy([one])
        with torch.no_grad():
            model.layers[0].mlp.down_proj.weight.add_(.1)
        full, _ = policy._forward_letters_full([one])
        row = {"input": one.to_json(), "target": {"indices": [0]}, "sample_id": "eval",
               "decision_type": one.decision_type}
        metrics = T.evaluate(policy, [row], torch.device("cpu"), T.Ranks(env=False))
        assert abs(metrics["controller/loss"] - float(-full[0].log_softmax(-1)[0])) < 2e-5
        assert not policy._static_kv

        # Letter readout has one prompt per question, not one per candidate.
        d = ds[0]
        width = max(policy.path_lengths(d))
        assert chunk_decisions(policy, [d, d, d], 3, width * 3) == [[0, 1, 2]]
        assert chunk_decisions(policy, [d, d, d], 3, width * 2) == [[0, 1], [2]]
        policy.static_cache = False
        policy.forward_max_paths = 2
        calls.clear()
        limited, _ = policy._forward_letters_full(ds)
        assert calls == [(2, False), (2, False)], calls
        for i, ref in enumerate(logits_and_values(policy, ds)):
            torch.testing.assert_close(limited[i, :len(ref)], ref, atol=2e-5, rtol=2e-4)
        # Sampling order remains per-episode even when prompt execution is reordered into batches.
        gen = lambda: [torch.Generator().manual_seed(400 + i) for i in range(len(ds))]
        together = decide(policy, ds, generators=gen(), mode="sample", autocast_dtype=None)
        alone = [decide(policy, [d], generators=[g], mode="sample", autocast_dtype=None)[0] for d, g in zip(ds, gen())]
        assert [r["index"] for r in together] == [r["index"] for r in alone]
    print("letter batching: logits, gradients, native positions, hybrid cache, LRU and sampling passed")


def logits_and_values(policy, decisions):
    return [policy._forward_letters_full([d])[0][0] for d in decisions]


def test_letter_batching_on_hybrid_model():
    result = subprocess.run([sys.executable, __file__, "--worker"], cwd=ROOT, text=True, capture_output=True,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2"}, timeout=900)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "sampling passed" in result.stdout


if __name__ == "__main__":
    worker()
