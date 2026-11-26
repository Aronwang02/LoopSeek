"""CPU: training-time prefix sharing (JevPolicy.forward_shared, policy.prefix_sharing_train) equals the full candidate
paths on a tiny random Qwen3.5 (3 Gated DeltaNet layers + 1 full-attention layer, fp32): gradients with gradient
checkpointing on, logits in train mode without gradients (train_sft's dev evaluation), checkpointing restored, and no
reference cycle in the out-of-place cache. Runs in a subprocess with fla blocked, so transformers uses its torch
kernels (fla's Triton kernels need a GPU and use TF32)."""
import json
import subprocess
import sys

from conftest import ROOT, nanojev_tokenizer_dir

import pytest

SCRIPT = r'''
import gc, json, sys, weakref
sys.modules["fla"] = None
sys.modules["causal_conv1d"] = None
sys.path.insert(0, sys.argv[1])
import torch
from transformers import AutoTokenizer, DynamicCache
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
from src.common.types import DecisionInput
from src.policy.model import JevPolicy, out_of_place_linear_cache

tok = AutoTokenizer.from_pretrained(sys.argv[2])
torch.manual_seed(0)
cfg = Qwen3_5TextConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=16, linear_num_value_heads=4,
                        linear_num_key_heads=2, linear_key_head_dim=16, linear_value_head_dim=16,
                        linear_conv_kernel_dim=4, layer_types=["linear_attention"] * 3 + ["full_attention"])
cfg._attn_implementation = "sdpa"
m = Qwen3_5TextModel(cfg).float()
for p in m.parameters():
    p.data.normal_(0, 0.2)
pol = JevPolicy(m, tok, vis_dim=8, tokens_per_frame=4)
pol.prefix_sharing_train = True
m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
d = DecisionInput("controller", ["State: the video shows a cat chasing a red ball around a kitchen table for a while.\nChoose:"],
                  [["\nA: LOCALIZE_GLOBAL\nDecision:"], ["\nB: INSPECT_FRAMES now\nDecision:"], ["\nC: REFINE_LOCAL\nDecision:"]],
                  ["A", "B", "C"], [True, True, True])
w = torch.tensor([0.7, -1.1, 0.4])
params = [p for p in pol.parameters() if p.requires_grad]
out = {}
pol.train()
res = {}
for shared in (False, True):
    pol.prefix_sharing = shared
    pol.zero_grad(set_to_none=True)
    z, _ = pol([d])
    (w * torch.log_softmax(z[0, :3].float(), -1)).sum().backward()
    res[shared] = (z.detach()[0, :3].clone(), torch.cat([p.grad.flatten() for p in params if p.grad is not None]))
out["grad_logit_diff"] = float((res[True][0] - res[False][0]).abs().max())
out["grad_rel"] = float((res[True][1] - res[False][1]).norm() / res[False][1].norm())
out["grad_numel"] = [res[False][1].numel(), res[True][1].numel()]
with torch.no_grad():                                   # train mode, no gradients (train_sft's dev evaluation)
    pol.prefix_sharing = False
    a = pol([d])[0][0, :3]
    pol.prefix_sharing = True
    b = pol([d])[0][0, :3]
out["train_nograd_logit_diff"] = float((a - b).abs().max())
out["checkpointing_restored"] = all(l.gradient_checkpointing for l in m.layers)
gc.disable()                                            # the out-of-place cache must be freed by reference counting
cache = out_of_place_linear_cache(DynamicCache(config=cfg))
lin = [l for l in cache.layers if hasattr(l, "update_conv_state")]
refs = [weakref.ref(l) for l in lin]
out["methods_on_instance"] = sum("update_conv_state" in vars(l) for l in lin)
del cache, lin
out["layers_alive_after_del"] = sum(r() is not None for r in refs)
gc.enable()
print("RESULT " + json.dumps(out))
'''


def test_prefix_sharing_train(cfg):
    tok_dir = nanojev_tokenizer_dir(cfg)
    if tok_dir is None:
        pytest.skip("NanoJev tokenizer missing: python scripts/download_models.py nanojev_tokenizer")
    r = subprocess.run([sys.executable, "-c", SCRIPT, str(ROOT), tok_dir], capture_output=True, text=True,
                       env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin", "HF_HUB_OFFLINE": "1"}, timeout=900)
    line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT ")), None)
    assert line, r.stderr[-3000:]
    out = json.loads(line[len("RESULT "):])
    assert out["grad_numel"][0] == out["grad_numel"][1] > 0, out
    assert out["grad_logit_diff"] < 1e-4, out
    assert out["grad_rel"] < 1e-4, out                  # measured 1e-5 on this tiny model (4B GPU, strict fp32: <= 5e-4)
    assert out["train_nograd_logit_diff"] < 1e-4, out   # 0.017 before the fix (cache dropped in train mode)
    assert out["checkpointing_restored"], out
    assert out["methods_on_instance"] == 0 and out["layers_alive_after_del"] == 0, out
