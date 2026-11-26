"""Selection C (2026-09-29): one scoring of all frame candidates, then k picks without replacement.

Candidates are in time order, so neighbours on the candidate grid are adjacent indices. Picking j masks
j-r..j+r (r = mask_radius) for the later picks of the same inspection; candidates that are illegal from
the start (frames already sent to the reader) are never pickable. If every remaining legal candidate is
masked, the neighbour mask is lifted for that step (only exact repeats stay excluded).
Each step is a softmax over the candidates still available, so the log-probability of an ordered pick
sequence is the sum of the per-step log-probabilities (Plackett-Luce with masking).
"""
import torch


def available_ex(valid, picked, radius):
    """(bool list, fallback): legal, not picked, not within radius of a pick; fallback=True when the neighbour
    mask left nothing and was lifted (then: legal and not picked)."""
    k = len(valid)
    masked = set()
    for j in picked:
        masked.update(range(max(0, j - radius), min(k, j + radius + 1)))
    av = [bool(valid[i]) and i not in masked for i in range(k)]
    if not any(av):
        return [bool(valid[i]) and i not in picked for i in range(k)], True
    return av, False


def available(valid, picked, radius):
    return available_ex(valid, picked, radius)[0]


def teacher_sequence(valid, targets, k, radius, rng):
    """Teacher pick order for SFT (2026-09-30 rule): at every step the target set is recomputed among the
    candidates currently available (permanent + neighbour masks, fallback included); if non-empty, pick uniformly
    among the targets (supervised step); if empty, pick uniformly among the available non-targets and the step
    carries no loss; supervision resumes whenever targets reappear. Returns (picks, supervised flags)."""
    picks, sup, tset = [], [], set(targets)
    n_legal = sum(bool(v) for v in valid)
    for _ in range(min(k, n_legal)):
        av = available(valid, picks, radius)
        cand = [i for i, a in enumerate(av) if a]
        if not cand:
            break
        tg = [i for i in cand if i in tset]
        picks.append(rng.choice(tg) if tg else rng.choice(cand))
        sup.append(bool(tg))
    return picks, sup


def step_log_probs(logits, valid, picked, radius):
    """logits: [K] tensor (-inf for illegal). Log-softmax over the currently available candidates."""
    av = torch.tensor(available(valid, picked, radius), device=logits.device)
    z = torch.where(av, logits, torch.full_like(logits, float("-inf")))
    return z.log_softmax(-1), av


def sequence_log_prob(logits, valid, picks, radius):
    """Exact log P(ordered picks) under masked Plackett-Luce (differentiable in logits)."""
    total = logits.new_zeros(())
    for s, j in enumerate(picks):
        lp, av = step_log_probs(logits, valid, picks[:s], radius)
        assert bool(av[j]), f"pick {j} not available at step {s}"
        total = total + lp[j]
    return total


def set_target_nll(logits, valid, picks, targets, radius, supervised=None):
    """SFT loss teacher-forced on `picks` (from teacher_sequence): at every supervised step s the loss is
    -log sum_{i in targets available at s} p_s(i) (any correct candidate is right); unsupervised steps (no target
    available) add nothing. Returns (loss, mean per-step target-set prob, number of supervised steps)."""
    loss, probs = logits.new_zeros(()), []
    tset = set(targets)
    for s in range(len(picks)):
        if supervised is not None and not supervised[s]:
            continue
        lp, av = step_log_probs(logits, valid, picks[:s], radius)
        tg = [i for i in tset if bool(av[i])]
        if not tg:
            continue
        set_lp = torch.logsumexp(lp[tg], 0)
        loss = loss - set_lp
        probs.append(float(set_lp.detach().exp()))
    return loss, (sum(probs) / len(probs) if probs else 0.0), len(probs)


@torch.no_grad()
def pick(logits, valid, k, radius, mode="sample", generator=None, info=None):
    """Returns (picks, log_prob, first-step probs, first-step entropy). logits on CPU, fp32.
    `info` (dict) receives fallback_steps = number of steps where the neighbour mask had to be lifted."""
    picks, total, first_p, first_h = [], 0.0, None, 0.0
    n_legal = sum(bool(v) for v in valid)
    for s in range(min(k, n_legal)):
        if info is not None:
            info["fallback_steps"] = info.get("fallback_steps", 0) + int(available_ex(valid, picks, radius)[1])
        lp, av = step_log_probs(logits, valid, picks, radius)
        p = lp.exp()
        if s == 0:
            first_p = p
            fin = torch.isfinite(lp)
            first_h = float(-(p[fin] * lp[fin]).sum())
        j = int(torch.multinomial(p, 1, generator=generator).item()) if mode == "sample" else int(torch.argmax(p).item())
        total += float(lp[j])
        picks.append(j)
    return picks, total, first_p, first_h
