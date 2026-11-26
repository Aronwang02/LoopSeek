"""Categorical GRPO loss (draft §8.3).

L = -1/(BG) * sum_i 1/D_i * sum_d min(r_{i,d} A_i, clip(r_{i,d}, 1-delta, 1+delta) A_i)
    + beta * mean_i( mean_d KL(pi_theta || pi_ref)[s_{i,d}, C_{i,d}] )

Per-decision weights w_{i,d} = 1/(BG * D_i) let the caller split the sum over micro-batches:
summing `decision_loss_terms(...)` over all micro-batches equals L exactly.
KL is the exact categorical KL over the legal candidates of the stored candidate set.
"""
import torch

from ..policy.model import masked_log_softmax


def exact_kl(logp, ref_logp):
    """KL(p || q) per row over entries finite in logp; logp/ref_logp: [N, K] with -inf illegal."""
    legal = torch.isfinite(logp)
    p = logp.exp()
    diff = torch.where(legal, logp - ref_logp, torch.zeros_like(logp))
    return (torch.where(legal, p, torch.zeros_like(p)) * diff).sum(-1)


def decision_loss_parts(logits, ref_logits, action_index, old_log_prob, advantage, weight, clip=0.2):
    """Weighted sums of the clipped policy-gradient term and the KL term, kept apart so the
    reward-driven gradient can be measured on its own (update.md §6 M2)."""
    logp = masked_log_softmax(logits.float())
    ref_logp = masked_log_softmax(ref_logits.float())
    new_lp = logp.gather(1, action_index[:, None]).squeeze(1)
    ratio = torch.exp(new_lp - old_log_prob)
    pg = -torch.minimum(ratio * advantage, torch.clamp(ratio, 1 - clip, 1 + clip) * advantage)
    kl = exact_kl(logp, ref_logp)
    with torch.no_grad():
        clipped = ((ratio < 1 - clip) & (advantage < 0) | (ratio > 1 + clip) & (advantage > 0)).float()
        ref_lp = ref_logp.gather(1, action_index[:, None]).squeeze(1)
        stats = {"ratio": ratio.detach(), "kl": kl.detach(), "clipped": clipped, "new_lp": new_lp.detach(),
                 "ref_lp": ref_lp, "ref_logp_full": ref_logp.detach()}
    return (weight * pg).sum(), (weight * kl).sum(), stats


def decision_loss_terms(logits, ref_logits, action_index, old_log_prob, advantage, weight,
                        clip=0.2, kl_coef=1e-3):
    """logits/ref_logits: [N, K] (-inf at illegal); tensors of length N for the rest.
    weight = 1/(B*G*D_i) for the decision's trajectory. Returns (loss_sum, stats)."""
    logp = masked_log_softmax(logits.float())
    ref_logp = masked_log_softmax(ref_logits.float())
    new_lp = logp.gather(1, action_index[:, None]).squeeze(1)
    ratio = torch.exp(new_lp - old_log_prob)
    unclipped = ratio * advantage
    clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * advantage
    pg = -torch.minimum(unclipped, clipped)
    kl = exact_kl(logp, ref_logp)
    loss = (weight * (pg + kl_coef * kl)).sum()
    with torch.no_grad():
        clipped_frac = ((ratio < 1 - clip) & (advantage < 0) | (ratio > 1 + clip) & (advantage > 0)).float()
        stats = {"ratio_mean": ratio.mean().item(), "ratio_max": ratio.max().item(),
                 "ratio_min": ratio.min().item(), "clip_frac": clipped_frac.mean().item(),
                 "kl_mean": kl.mean().item(), "logp_new": new_lp.mean().item(),
                 "logp_old_absdiff_max": (new_lp - old_log_prob).abs().max().item(),
                 "pg_weighted": (weight * pg).sum().item(), "kl_weighted": (weight * kl).sum().item()}
    return loss, stats
