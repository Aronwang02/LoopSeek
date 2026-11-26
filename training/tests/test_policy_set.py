"""A separate NanoJev per role: policy sets load one model per role (each in its checkpoint's finetune mode),
one checkpoint is shared by every role as before, deciders and the runner route every decision to the
model of its role, and make_policy_set.py checks roles (CPU, tiny backbone)."""
import json
import subprocess
import sys

import pytest
import torch

from conftest import ROOT, tiny_backbone
from src.baselines.registry import build_deciders
from src.policy.model import JevPolicy, configure_finetune, frozen_copy
from src.policy.policy_set import CORE_ROLES, ROLES, load_policies, read_policy_set, role_checkpoint
from src.rollout.runner import JevDecider
from src.training.checkpoint import base_model_info, save_checkpoint, trainable_fingerprint


def builder(tokenizer):
    def build(cfg, feature_store=None, device="cpu"):
        pol = JevPolicy(tiny_backbone(len(tokenizer)), tokenizer, vis_dim=768, tokens_per_frame=4)
        configure_finetune(pol, cfg["policy"])
        pol.feature_store = feature_store
        return pol
    return build


def saved(tmp_path, tokenizer, cfg, name, bias, mode="full", types=None):
    """A checkpoint whose head bias marks it; returns (dir, fingerprint)."""
    pol = builder(tokenizer)({**cfg, "policy": {**cfg["policy"], "finetune": mode}})
    with torch.no_grad():
        pol.head.scalar.bias.fill_(bias)
    info = {**base_model_info(cfg, pol), "decision_types": types}
    save_checkpoint(tmp_path / name, pol, None, base_info=info)
    return tmp_path / name / "latest", trainable_fingerprint(pol)


def test_one_checkpoint_is_shared_by_every_role(tmp_path, tokenizer, cfg, features):
    ck, fp = saved(tmp_path, tokenizer, cfg, "shared", 0.5)
    pols = load_policies(cfg, str(ck), ROLES, features, build=builder(tokenizer))
    assert len({id(p) for p in pols.values()}) == 1 and trainable_fingerprint(pols["frame"]) == fp


def test_policy_set_loads_one_model_per_role(tmp_path, tokenizer, cfg, features):
    cks = {r: saved(tmp_path, tokenizer, cfg, r, b, mode="lora" if r == "frame" else "full", types=[r])
           for r, b in zip(CORE_ROLES, (1.0, 2.0, 3.0))}
    ps = tmp_path / "set.json"
    ps.write_text(json.dumps({r: str(c) for r, (c, _) in cks.items()}))
    pols = load_policies(cfg, str(ps), ROLES, features, build=builder(tokenizer))   # verifier optional: absent
    assert set(pols) == set(CORE_ROLES) and len({id(p) for p in pols.values()}) == 3
    for r in CORE_ROLES:
        assert trainable_fingerprint(pols[r]) == cks[r][1]
        assert pols[r].feature_store is features
    assert [pols[r].finetune for r in CORE_ROLES] == ["full", "full", "lora"]  # each built in its own mode
    two = load_policies(cfg, str(ps), ("localize", "frame"), features, build=builder(tokenizer))
    assert set(two) == {"localize", "frame"}


def test_lora_roles_share_only_frozen_base(tmp_path, tokenizer, cfg, features):
    cks = {r: saved(tmp_path, tokenizer, cfg, r, b, mode='lora')
           for r, b in zip(CORE_ROLES, (1.0, 2.0, 3.0))}
    ps = tmp_path / 'lora_set.json'
    ps.write_text(json.dumps({r: str(c) for r, (c, _) in cks.items()}))
    pols = load_policies(cfg, str(ps), CORE_ROLES, features, build=builder(tokenizer))
    controller, grounding = pols['controller'], pols['localize']
    cp, gp = dict(controller.backbone.named_parameters()), dict(grounding.backbone.named_parameters())
    for name in cp:
        assert (cp[name] is gp[name]) == (not cp[name].requires_grad), name
    for r in CORE_ROLES:
        assert trainable_fingerprint(pols[r]) == cks[r][1]
    before = trainable_fingerprint(controller)
    name = next(n for n, p in gp.items() if p.requires_grad)
    with torch.no_grad():
        gp[name].add_(0.01)
    assert before == trainable_fingerprint(controller)
    assert trainable_fingerprint(grounding) != cks['localize'][1]
    ref = frozen_copy(grounding)
    rp = dict(ref.backbone.named_parameters())
    saved_adapter = rp[name].clone()
    assert rp[name] is not gp[name] and not rp[name].requires_grad
    frozen_name = next(n for n, p in gp.items() if not p.requires_grad)
    assert rp[frozen_name] is gp[frozen_name]
    with torch.no_grad():
        gp[name].add_(0.01)
    assert torch.equal(rp[name], saved_adapter)


def test_policy_set_errors(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"reader": "x"}))
    with pytest.raises(ValueError, match="unknown roles"):
        read_policy_set(str(bad))
    part = tmp_path / "part.json"
    part.write_text(json.dumps({"frame": "f"}))
    assert role_checkpoint(str(part), "frame") == "f" and role_checkpoint("ck_dir", "frame") == "ck_dir"
    with pytest.raises(ValueError, match="no checkpoint for role"):
        role_checkpoint(str(part), "localize")


def test_deciders_one_per_model(tokenizer, cfg):
    b = builder(tokenizer)
    a, c = b(cfg), b(cfg)
    spec = {r: "nanojev" for r in CORE_ROLES}
    sep = build_deciders(spec, {"controller": a, "localize": c, "frame": a}, cfg=cfg)
    assert sep["controller"] is sep["frame"] and sep["localize"] is not sep["controller"]
    assert sep["localize"].policy is c and sep["controller"].policy is a
    shared = build_deciders(spec, a, cfg=cfg)                                  # original call form
    assert len({id(d) for d in shared.values()}) == 1 and shared["frame"].policy is a


def test_runner_sends_each_decision_to_its_role_model(tokenizer, cfg, asset, features):
    """Long-video flow (RunnerV3): every decision goes to its own role model; the locator is also probed before each
    controller decision (budget.locator_line), and probes are not recorded as decisions."""
    from conftest import StubEncoder, StubReader
    from src.env.environment import PublicSample
    from src.rollout.runner_v3 import RunnerV3
    roles = CORE_ROLES + ("verifier",)
    b = builder(tokenizer)
    pols = {r: b(cfg, feature_store=features) for r in roles}
    seen = {r: [] for r in roles}
    for r, pol in pols.items():
        orig = pol.forward

        def fwd(decisions, _orig=orig, _r=r):
            seen[_r] += [d.decision_type for d in decisions]
            return _orig(decisions)
        pol.forward = fwd
    deciders = {r: JevDecider(pols[r]) for r in roles}
    sample = PublicSample("s1", "v1", "unused", "What color?", {"A": "red", "B": "green", "C": "blue", "D": "gray"})
    runner = RunnerV3(cfg, deciders, StubReader(), features, StubEncoder(), text_len=pols["controller"].text_len)
    trajs = runner.run([{"sample": sample, "asset": asset, "group_id": "g0", "trajectory_id": f"g0:{i}", "seed": i}
                        for i in range(4)])
    assert all(t["status"] == "ok" for t in trajs), [t["error"] for t in trajs]
    made = [d["decision_type"] for t in trajs for d in t["decisions"] if not d["forced"]]
    assert {r: set(v) for r, v in seen.items() if v} == {r: {r} for r in roles if seen[r]}
    for r in ("controller", "frame", "verifier"):
        assert len(seen[r]) == made.count(r) > 0, (r, len(seen[r]), made.count(r))
    assert len(seen["localize"]) >= made.count("localize")


def test_make_policy_set_checks_roles(tmp_path, tokenizer, cfg):
    loc, _ = saved(tmp_path, tokenizer, cfg, "loc", 1.0, types=["localize"])
    frm, _ = saved(tmp_path, tokenizer, cfg, "frm", 2.0, types=["frame"])
    out = tmp_path / "ps.json"
    run = [sys.executable, str(ROOT / "scripts/make_policy_set.py"), "--out", str(out)]
    ok = subprocess.run(run + ["--localize", str(loc), "--frame", str(frm)], capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr
    assert json.loads(out.read_text()) == {"localize": str(loc.resolve()), "frame": str(frm.resolve())}
    bad = subprocess.run(run + ["--localize", str(frm)], capture_output=True, text=True)
    assert bad.returncode != 0 and "trained on decision types" in bad.stderr


def test_roles_listing_one_checkpoint_share_one_model(tmp_path, tokenizer, cfg, features):
    ground, fg = saved(tmp_path, tokenizer, cfg, "ground", 1.0, types=["localize", "frame"])
    ctrl, fc = saved(tmp_path, tokenizer, cfg, "ctrl", 2.0, types=["controller"])
    ps = tmp_path / "set.json"
    ps.write_text(json.dumps({"controller": str(ctrl), "localize": str(ground), "frame": str(ground / ".." / "latest")}))
    pols = load_policies(cfg, str(ps), ROLES, features, build=builder(tokenizer))
    assert pols["localize"] is pols["frame"] and pols["controller"] is not pols["frame"]
    assert trainable_fingerprint(pols["frame"]) == fg and trainable_fingerprint(pols["controller"]) == fc
    decs = build_deciders({r: "nanojev" for r in ROLES}, pols, cfg=cfg)
    assert decs["localize"] is decs["frame"] and decs["controller"] is not decs["frame"]
