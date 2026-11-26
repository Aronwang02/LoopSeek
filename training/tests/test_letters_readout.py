"""CPU: OneJev letter readout (policy.readout = letters, src/policy/letters.py). (1) Our rendering of a decision gives
exactly the token ids of OneJev's qev prompt (system + <state> + question suffix + generation prompt, thinking off),
with and without image blocks; (2) on a tiny random Qwen3.5 with OneJev's tokenizer, the shared-prefix branches give the
same option logits as running each full prompt in one piece (single and grouped decisions, static cache on / off), and
gradients flow. Subprocess with fla blocked (torch kernels on CPU)."""
import json
import subprocess
import sys

import pytest

from conftest import ROOT

SCRIPT = r'''
import json, sys
sys.modules["fla"] = None
sys.modules["causal_conv1d"] = None
sys.path.insert(0, sys.argv[1])
import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel
from src.common.types import DecisionInput, VisRef
from src.env.environment import merge_text, nanojev_candidate
from src.policy.letters import SYSTEM_PROMPT, chat_parts, option_name_text, question_suffix
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
pol.readout, pol.prefix_sharing = "letters", True
out = {}
head, tail = chat_parts(tok)
Q = "Question type: choice\nQuestion:\n"
cands = [nanojev_candidate("REFINE_LOCAL: go down"), nanojev_candidate("UP: go up"), nanojev_candidate("SWITCH_LOCAL: go sideways")]
d = DecisionInput("controller", ["State:\nTask: controller\nQuestion: what color is the cup?\n", "\nMemory:\n- Kept frames: 0 of 16\n",
                                 Q + "Which action should the agent take next?\n"], cands, ["REFINE_LOCAL", "UP", "SWITCH_LOCAL"],
                  [True, True, False], [{"action": "REFINE_LOCAL"}, {"action": "UP"}, {"action": "SWITCH_LOCAL"}])
suffix = question_suffix("Which action should the agent take next?", [option_name_text(cands[i], d.candidate_ids[i]) for i in (0, 1)])
out["suffix_qev"] = suffix == ("Question: Which action should the agent take next?\n\nOptions:\nA. REFINE_LOCAL: go down\n"
                               "B. UP: go up\n\nAnswer with one letter: A, B.")
state = "State:\nTask: controller\nQuestion: what color is the cup?\n\nMemory:\n- Kept frames: 0 of 16\n"
full = tok.apply_chat_template([{"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": f"<state>\n{state}\n</state>\n\n{suffix}"}],
                               tokenize=False, add_generation_prompt=True, enable_thinking=False)
ref_ids = tok.encode(full, add_special_tokens=False)
segs = merge_text([head] + list(d.prefix[:-1]) + ["\n</state>\n\n"])
ours = pol._segments_to_ids(segs, []) + pol._segments_to_ids([suffix + tail], [])
out["parity_text"] = ours == ref_ids
# with image blocks (native layout: <|vision_start|> image tokens <|vision_end|>)
pol.native, pol.vision_start_id, pol.vision_end_id = True, tok.convert_tokens_to_ids("<|vision_start|>"), tok.convert_tokens_to_ids("<|vision_end|>")
pad = tok.convert_tokens_to_ids("<|image_pad|>")
ref = VisRef("c", 0, 0.0, 0.0, 0.0)
dimg = ["State:\nTask: controller\nOverview frames (time-ordered):\nG01 [0.1s] ", ref, "\nG02 [5.0s] ", ref, "\nMemory:\n- x\n", Q + "Which?\n"]
segs = merge_text([head] + dimg[:-1] + ["\n</state>\n\n"])
ours = [pad if t < 0 else t for t in pol._segments_to_ids(segs, []) + pol._segments_to_ids([suffix + tail], [])]
blk = "<|vision_start|>" + "<|image_pad|>" * 4 + "<|vision_end|>"
state_i = "State:\nTask: controller\nOverview frames (time-ordered):\nG01 [0.1s] " + blk + "\nG02 [5.0s] " + blk + "\nMemory:\n- x\n"
full = tok.apply_chat_template([{"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": f"<state>\n{state_i}\n</state>\n\n{suffix}"}],
                               tokenize=False, add_generation_prompt=True, enable_thinking=False)
out["parity_image"] = ours == tok.encode(full, add_special_tokens=False)
# Check native M-RoPE against the official multimodal builder, independently of
# both of our cached / full-prompt text-model paths. Include the real 32-frame grid.
import math


def official_image_positions(ids, tokens_per_frame, spatial_merge_size=2):
    """Call transformers' multimodal position builder without allocating a model."""
    from types import MethodType, SimpleNamespace
    import torch
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
    side = math.isqrt(tokens_per_frame)
    assert side * side == tokens_per_frame
    image_count = sum(t < 0 for t in ids) // tokens_per_frame
    shell = SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace(
        spatial_merge_size=spatial_merge_size)))
    shell.get_vision_position_ids = MethodType(Qwen3_5Model.get_vision_position_ids, shell)
    token_types = torch.tensor([[int(t < 0) for t in ids]])
    grid = torch.tensor([[1, side * spatial_merge_size, side * spatial_merge_size]] * image_count)
    positions, _ = Qwen3_5Model.get_rope_index(
        shell, torch.tensor([[max(t, 0) for t in ids]]), token_types, image_grid_thw=grid)
    return positions[:, 0]


out["official_mrope"] = []
for count, tokens in ((2, 4), (32, 64)):
    pol.tokens_per_frame = tokens
    ids = pol._segments_to_ids([head] + [v for _ in range(count) for v in ("Frame: ", VisRef("c", 0, 0., 0., 0.), "\n")] + [suffix + tail], [])
    out["official_mrope"].append(torch.equal(pol._mrope_positions(ids), official_image_positions(ids, tokens)))
pol.tokens_per_frame = 4
pol.native = False

def reference(d, questions):
    """Each question's full prompt run in one piece; logits = last hidden . embedding rows of A, B, ..."""
    from src.policy.letters import slot_token_ids
    slots = slot_token_ids(tok)
    res = {}
    segs = merge_text([head] + list(d.prefix[:-1]) + ["\n</state>\n\n"])
    pre = pol._segments_to_ids(segs, [])
    for instr, idx in questions:
        suf = pol._segments_to_ids([question_suffix(instr, [option_name_text(d.candidates[i], d.candidate_ids[i]) for i in idx]) + tail], [])
        h = m(input_ids=torch.tensor([pre + suf])).last_hidden_state[0, -1]
        z = h @ m.get_input_embeddings().weight[slots[:len(idx)]].T
        for i, v in zip(idx, z.tolist()):
            res[i] = v
    return res

with torch.no_grad():
    got = pol([d])[0][0]
    ref1 = reference(d, [("Which action should the agent take next?", [0, 1])])
    out["single_diff"] = max(abs(float(got[i]) - v) for i, v in ref1.items())
    out["illegal_inf"] = bool(torch.isinf(got[2]))
    # grouped: two frames, yes / no each, own questions
    gc = [nanojev_candidate("E1+: E1 [1.0s] yes"), nanojev_candidate("E1-: E1 [1.0s] no"),
          nanojev_candidate("E2+: E2 [2.0s] yes"), nanojev_candidate("E2-: E2 [2.0s] no")]
    dg = DecisionInput("verifier", ["State:\nTask: verifier\nQuestion: what color is the cup?\n", "\nSelected frames: none\n",
                                    Q + "For each frame: in or near the evidence?\n"], gc, ["E1+", "E1-", "E2+", "E2-"],
                       [True] * 4, [{"keep": k, "group": g} for g in (0, 1) for k in (True, False)])
    dg.group_questions = {"0": "Is frame E1 in or near the evidence?", "1": "Is frame E2 in or near the evidence?"}
    gotg = pol([dg])[0][0]
    refg = reference(dg, [("Is frame E1 in or near the evidence?", [0, 1]), ("Is frame E2 in or near the evidence?", [2, 3])])
    out["grouped_diff"] = max(abs(float(gotg[i]) - v) for i, v in refg.items())
    # static cache: first segment static
    d.static_segments = 1
    pol.static_cache = True
    pol.clear_static_cache()
    a = pol([d])[0][0]
    b = pol([d])[0][0]
    out["static_diff"] = max(abs(float(a[i]) - v) for i, v in ref1.items())
    out["static_stats"] = dict(pol.static_cache_stats)    # text-only head: not cut (would change tokenization)
pol.static_cache = False
pol.train()
z = pol([d])[0][0]
torch.log_softmax(z[:2], -1)[0].backward()
out["grad_norm"] = float(sum(p.grad.norm() for p in m.parameters() if p.grad is not None))
grads = {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}
m.zero_grad(set_to_none=True)
from src.policy.letters import slot_token_ids
pre = pol._segments_to_ids(merge_text([head] + d.prefix[:-1] + ["\n</state>\n\n"]), [])
suf = pol._segments_to_ids([suffix + tail], [])
h = m(input_ids=torch.tensor([pre + suf])).last_hidden_state[0, -1]
full_logits = h @ m.get_input_embeddings().weight[slot_token_ids(tok)[:2]].T
torch.log_softmax(full_logits, -1)[0].backward()
params = dict(m.named_parameters())
error = sum(float((g - params[n].grad).square().sum()) for n, g in grads.items())
norm = sum(float(g.square().sum()) for g in grads.values())
out["gradient_relative_error"] = (error / norm) ** .5
# Native letter training can mix one question with a grouped verifier row. Compare
# every model gradient against three independent full prompts, and retain layer checkpointing.
from src.policy.letters import questions_of
from src.policy.model import grouped_log_softmax, payload_groups
pol.prefix_sharing_train = True
m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
calls = []
original_forward = m.forward
def traced(*args, **kw):
    calls.append({"cache": kw.get("use_cache"), "checkpointing": m.is_gradient_checkpointing})
    return original_forward(*args, **kw)
m.forward = traced
m.zero_grad(set_to_none=True)
batch_logits, batch_stats = pol([d, dg])
out["mixed_calls"] = calls.copy()
out["checkpointing_restored"] = m.is_gradient_checkpointing
loss = -torch.log_softmax(batch_logits[0, :2], -1)[0] - grouped_log_softmax(batch_logits[1], payload_groups(dg))[[0, 2]].sum()
loss.backward()
mixed_grads = {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}
m.zero_grad(set_to_none=True)
full_loss = 0
for decision in (d, dg):
    prefix = pol._segments_to_ids(merge_text([head] + decision.prefix[:-1] + ["\n</state>\n\n"]), [])
    for instr, indices in questions_of(decision):
        suffix_ids = pol._segments_to_ids([question_suffix(instr, [option_name_text(decision.candidates[i], decision.candidate_ids[i])
                                                                for i in indices]) + tail], [])
        h = m(input_ids=torch.tensor([prefix + suffix_ids]), use_cache=False).last_hidden_state[0, -1]
        z = h @ m.get_input_embeddings().weight[slot_token_ids(tok)[:len(indices)]].T
        full_loss = full_loss - z.log_softmax(-1)[0]
full_loss.backward()
params = dict(m.named_parameters())
error = sum(float((g - params[n].grad).square().sum()) for n, g in mixed_grads.items())
norm = sum(float(g.square().sum()) for g in mixed_grads.values())
out["mixed_gradient_relative_error"] = (error / norm) ** .5
pol.prefix_sharing_train = False
calls.clear()
full_grouped, _ = pol([dg])
out["full_grouped_calls"] = calls.copy()
out["full_grouped_diff"] = max(abs(float(full_grouped[0, i].detach()) - v) for i, v in refg.items())
print("RESULT " + json.dumps(out))
'''


def test_letters_readout_matches_qev_and_full_prompt(cfg):
    path = cfg["models"].get("onejev_4b", {}).get("path")
    if not path:
        pytest.skip("OneJev-4B tokenizer missing: python scripts/download_models.py onejev_4b")
    r = subprocess.run([sys.executable, "-c", SCRIPT, str(ROOT), path], capture_output=True, text=True,
                       env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin", "HF_HUB_OFFLINE": "1"}, timeout=900)
    line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT ")), None)
    assert line, r.stderr[-3000:]
    out = json.loads(line[len("RESULT "):])
    assert out["suffix_qev"] and out["parity_text"] and out["parity_image"], out   # byte-for-byte the qev prompt
    assert all(out["official_mrope"]), out
    assert out["single_diff"] < 1e-4 and out["grouped_diff"] < 1e-4, out              # shared prefix = full prompt
    assert out["illegal_inf"], out
    assert out["static_diff"] < 1e-4 and out["static_stats"] == {"hit": 0, "miss": 0}, out
    assert out["grad_norm"] > 0, out
    assert out["gradient_relative_error"] < 5e-4, out
    assert out["mixed_calls"] == [{"cache": False, "checkpointing": True},
                                   {"cache": True, "checkpointing": False},
                                   {"cache": True, "checkpointing": False}], out
    assert out["checkpointing_restored"] and out["mixed_gradient_relative_error"] < 5e-4, out
    assert out["full_grouped_calls"] == [{"cache": False, "checkpointing": True}], out
    assert out["full_grouped_diff"] < 1e-4, out
