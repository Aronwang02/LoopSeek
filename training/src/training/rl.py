"""One synchronous GRPO update over complete same-question groups (update.md §6 M2).

Besides the update itself this records what the update doc asks to be able to verify:
  * groups with non-zero advantage (reward differences) vs zero-variance groups;
  * the gradient norm of the reward (clipped PG) term and of the KL term, separately;
  * recomputed log-prob at the rollout parameters vs the stored rollout log-prob;
  * reference log-probs written back into every decision record;
  * the actual parameter change of LoRA, projector and Choice head.
"""
import math
import time

import torch

from ..common.types import DecisionRecord
from ..policy.sampler import chunk_decisions
from .grpo_loss import decision_loss_parts


def collect_training_items(groups, assigner):
    """items: (DecisionRecord, advantage, trajectory index, decision dict). Groups with any
    failed trajectory (infrastructure or path budget) are dropped entirely."""
    stats = {"groups_total": len(groups), "groups_dropped": 0, "groups_zero_variance": 0,
             "groups_nonzero_advantage": 0, "trajectories": 0, "traj_zero_decisions": 0,
             "decisions_trainable": 0, "decisions_forced": 0, "reward_mean": float("nan")}
    items, rewards, n_traj = [], [], 0
    for g in groups:
        if any(t["status"] != "ok" or t.get("reward") is None for t in g):
            stats["groups_dropped"] += 1
            continue
        per_traj, st = assigner.assign(g)
        stats["groups_zero_variance"] += int(st["zero_variance"])
        stats["groups_nonzero_advantage"] += int(not st["zero_variance"])
        for j, (t, advs) in enumerate(zip(g, per_traj)):
            rewards.append(t["reward"])
            t["advantage"] = st["advantages"][j]
            n_train = 0
            for d, a in zip(t["decisions"], advs):
                d["advantage"] = a if d["trainable"] else 0.0
                stats["decisions_forced"] += int(d["forced"])
                if d["trainable"]:
                    items.append((DecisionRecord.from_json(d), a, n_traj, d))
                    n_train += 1
            stats["traj_zero_decisions"] += int(n_train == 0)
            n_traj += 1
    stats["trajectories"] = n_traj
    stats["decisions_trainable"] = len(items)
    if rewards:
        stats["reward_mean"] = sum(rewards) / len(rewards)
    return items, stats


def _group_params(policy):
    g = {"lora": [p for n, p in policy.named_parameters() if "lora_" in n and p.requires_grad],
         "projector": [p for p in policy.projector.parameters() if p.requires_grad],
         "head": [p for p in policy.head.parameters() if p.requires_grad]
         + ([p for p in policy.prior_gate.parameters() if p.requires_grad] if policy.prior_gate is not None else [])}
    backbone = policy.backbone_params()
    if backbone:                 # finetune='full' only, so LoRA runs keep their log keys
        g["backbone"] = backbone
    return g


def _norm(ts):
    ts = [t for t in ts if t is not None]
    return math.sqrt(sum(float(t.float().pow(2).sum()) for t in ts)) if ts else 0.0


def grpo_update(policy, ref_policy, optimizer, groups, assigner, rl_cfg, pol_cfg, device,
                autocast_dtype=torch.bfloat16):
    t0 = time.perf_counter()
    items, stats = collect_training_items(groups, assigner)
    if not items:
        stats.update(skipped=True, seconds=time.perf_counter() - t0)
        return stats
    n_traj = stats["trajectories"]
    d_per_traj = {}
    for _, _, ti, _ in items:
        d_per_traj[ti] = d_per_traj.get(ti, 0) + 1
    # L = -1/(BG) sum_i 1/D_i sum_d (...): per-decision weight 1/(BG * D_i)
    weights = [1.0 / (n_traj * d_per_traj[ti]) for _, _, ti, _ in items]
    params = [p for p in policy.parameters() if p.requires_grad]
    pg_buf = [torch.zeros_like(p) for p in params]
    kl_buf = [torch.zeros_like(p) for p in params]
    groups_p = _group_params(policy)
    before = {k: [p.detach().clone() for p in v] for k, v in groups_p.items()}
    policy.train()   # no dropout anywhere in the policy (attention 0, LoRA 0, head 0)
    optimizer.zero_grad(set_to_none=True)
    agg = {"pg_weighted": 0.0, "kl_weighted": 0.0, "ratio_max": 0.0, "ratio_min": math.inf,
           "logp_old_absdiff_max": 0.0, "microbatches": 0, "decisions": 0}
    kl_sum = clipped = 0.0
    inputs_all = [d.input for d, _, _, _ in items]
    for chunk in chunk_decisions(policy, inputs_all, pol_cfg["max_paths_per_forward"],
                                 pol_cfg["max_tokens_per_forward"]):
        mb = [items[i] for i in chunk]
        inputs = [inputs_all[i] for i in chunk]
        act = torch.tensor([d.action_index for d, _, _, _ in mb], device=device)
        old = torch.tensor([d.old_log_prob for d, _, _, _ in mb], device=device)
        adv = torch.tensor([a for _, a, _, _ in mb], device=device, dtype=torch.float32)
        w = torch.tensor([weights[i] for i in chunk], device=device, dtype=torch.float32)
        with torch.autocast(device.type, dtype=autocast_dtype, enabled=device.type == "cuda"):
            logits, _ = policy(inputs)
            with torch.no_grad():
                ref_logits, _ = ref_policy(inputs)
        pg, kl, st = decision_loss_parts(logits, ref_logits, act, old, adv, w, clip=rl_cfg["clip"])
        g_pg = torch.autograd.grad(pg, params, retain_graph=True, allow_unused=True)
        g_kl = torch.autograd.grad(rl_cfg["kl_coef"] * kl, params, allow_unused=True)
        for b1, b2, a1, a2 in zip(pg_buf, kl_buf, g_pg, g_kl):
            if a1 is not None:
                b1.add_(a1)
            if a2 is not None:
                b2.add_(a2)
        for (d, _, _, dd), ref_lp, ref_full, new_lp in zip(mb, st["ref_lp"].tolist(),
                                                          st["ref_logp_full"].cpu(), st["new_lp"].tolist()):
            k = len(d.input.candidates)
            dd["ref_log_prob"] = ref_lp
            dd["ref_probs"] = [float(x) for x in ref_full[:k].exp()]
            dd["recomputed_log_prob"] = new_lp
        agg["pg_weighted"] += float(pg.detach())
        agg["kl_weighted"] += float(kl.detach())
        kl_sum += float(st["kl"].sum())
        clipped += float(st["clipped"].sum())
        agg["ratio_max"] = max(agg["ratio_max"], float(st["ratio"].max()))
        agg["ratio_min"] = min(agg["ratio_min"], float(st["ratio"].min()))
        agg["logp_old_absdiff_max"] = max(agg["logp_old_absdiff_max"], float((st["new_lp"] - old).abs().max()))
        agg["microbatches"] += 1
        agg["decisions"] += len(chunk)
    for p, b1, b2 in zip(params, pg_buf, kl_buf):
        p.grad = b1 + b2
    stats["grad_norm_reward_term"] = _norm(pg_buf)
    stats["grad_norm_kl_term"] = _norm(kl_buf)
    pos = {id(p): j for j, p in enumerate(params)}
    for name, ps in groups_p.items():
        stats[f"grad_norm_reward_term_{name}"] = _norm([pg_buf[pos[id(p)]] for p in ps])
    stats["grad_norm_total_preclip"] = float(torch.nn.utils.clip_grad_norm_(params, rl_cfg["grad_clip"]))
    optimizer.step()
    for name, ps in groups_p.items():
        stats[f"param_change_{name}"] = _norm([p.detach() - q for p, q in zip(ps, before[name])])
    stats.update(agg)
    stats["kl_mean"] = kl_sum / max(agg["decisions"], 1)
    stats["clip_frac"] = clipped / max(agg["decisions"], 1)
    stats.update(skipped=False, seconds=time.perf_counter() - t0)
    return stats


def build_optimizer(policy, lr_lora, lr_new, weight_decay=0.0, lr_backbone=None):
    """AdamW groups: LoRA, projector + head, and (finetune='full') the backbone, whose learning rate
    has no default and must be given explicitly. Empty groups are left out. Learning rates are cast
    to float because PyYAML reads a --set value such as 1e-5 (no decimal point) as a string."""
    lora, new = policy.param_groups()
    groups = [{"params": lora, "lr": float(lr_lora), "name": "lora"},
              {"params": new, "lr": float(lr_new), "name": "projector_head"}]
    backbone = policy.backbone_params()
    if backbone:
        if lr_backbone is None:
            raise ValueError("finetune='full' trains the backbone: set its learning rate explicitly "
                             "(sft.backbone_lr / rl.backbone_lr)")
        groups.append({"params": backbone, "lr": float(lr_backbone), "name": "backbone"})
    return torch.optim.AdamW([g for g in groups if g["params"]], weight_decay=weight_decay)
