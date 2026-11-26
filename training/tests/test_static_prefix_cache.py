"""CPU: the static-prefix KV cache (policy.static_prefix_cache, JevPolicy._static_then_dynamic) gives the same logits as
the uncached shared-prefix forward on a tiny random Qwen3.5 (3 Gated DeltaNet layers + 1 full-attention layer, fp32):
two decisions sharing a static head (hit), one with another head (miss), the same decision twice (the snapshot is never
written to), a decision without a static head (cache unused). Subprocess with fla blocked (torch kernels on CPU)."""
import json
import subprocess
import sys

from conftest import ROOT, nanojev_tokenizer_dir

import pytest

SCRIPT = r'''
import json, sys
sys.modules["fla"] = None
sys.modules["causal_conv1d"] = None
sys.path.insert(0, sys.argv[1])
import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
from src.common.types import DecisionInput
from src.policy.model import JevPolicy

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
pol = JevPolicy(m, tok, vis_dim=8, tokens_per_frame=4).eval()
pol.prefix_sharing = True
head = "State:\nTask: controller\nQuestion: what does the cook add to the pan right after the onions?\nOptions:\n(A) salt\n(B) garlic\n"
cands = [["\nA: REFINE_LOCAL\nDecision:"], ["\nB: SWITCH_LOCAL now\nDecision:"], ["\nC: INSPECT_FRAMES\nDecision:"]]
def dec(h, mem, ns):
    d = DecisionInput("controller", [h, mem, "Question type: choice\nQuestion:\nWhich action?\n"], cands,
                      ["A", "B", "C"], [True, True, True])
    d.static_segments = ns
    return d
ds = [dec(head, "\nMemory:\n- Kept frames: 0 of 16; actions left 24 of 24\n", 1),
      dec(head, "\nMemory:\n- Kept frames: 8 of 16; actions left 17 of 24\n- Verifier, evidence: inside 0.80\n", 1),
      dec(head.replace("onions", "garlic"), "\nMemory:\n- Kept frames: 0 of 16\n", 1),
      dec(head, "\nMemory:\n- Kept frames: 0 of 16; actions left 24 of 24\n", 1),
      dec(head, "\nMemory:\n- Kept frames: 3 of 16\n", 0)]
out = {}
with torch.no_grad():
    pol.static_cache = False
    ref = [pol([d])[0][0, :3].clone() for d in ds]
    pol.static_cache = True
    pol.clear_static_cache()
    got = [pol([d])[0][0, :3].clone() for d in ds]
out["max_diff"] = max(float((a - b).abs().max()) for a, b in zip(ref, got))
out["stats"] = dict(pol.static_cache_stats)
out["same_twice"] = float((got[0] - got[3]).abs().max())
out["dyn_matters"] = float((ref[0] - ref[1]).abs().max())
print("RESULT " + json.dumps(out))
'''


def test_static_prefix_cache_matches_uncached(cfg):
    tok_dir = nanojev_tokenizer_dir(cfg)
    if tok_dir is None:
        pytest.skip("NanoJev tokenizer missing: python scripts/download_models.py nanojev_tokenizer")
    r = subprocess.run([sys.executable, "-c", SCRIPT, str(ROOT), tok_dir], capture_output=True, text=True,
                       env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin", "HF_HUB_OFFLINE": "1"}, timeout=900)
    line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT ")), None)
    assert line, r.stderr[-3000:]
    out = json.loads(line[len("RESULT "):])
    assert out["max_diff"] < 1e-4, out                  # cached = uncached logits (fp32, CPU)
    assert out["stats"] == {"hit": 2, "miss": 2}, out   # heads: A (miss), A (hit), B (miss), A (hit); last: no head
    assert out["same_twice"] < 1e-6, out                # the snapshot is not modified by a decision
    assert out["dyn_matters"] > 1e-3, out               # the dynamic part does change the logits (test is not vacuous)
