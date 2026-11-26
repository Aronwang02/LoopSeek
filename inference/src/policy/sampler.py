"""Masked categorical decisions: sample (training rollouts, temperature 1) or argmax (eval).

A decision with exactly one legal candidate is `forced`: it is executed with probability 1, no
forward is run, and it never enters the policy loss.
"""
import time

import torch

from .model import grouped_log_softmax, masked_log_softmax, payload_groups


def _prior_info(policy, d):
    """For diagnostics: the SigLIP prior logits of the candidates (before the gate) and the gate value."""
    if getattr(policy, "prior_scorer", None) is None or not d.prior_texts:
        return {}
    with torch.no_grad():
        pr = policy.prior_scorer.scores(d, policy.feature_store)
    return {"prior": [round(float(x), 3) for x in pr[:len(d.candidates)]],
            "gate": round(float(policy.prior_gate.weight), 4) if policy.prior_gate is not None else None}


def chunk_decisions(policy, decisions, max_paths, max_tokens, max_shared_tokens=None):
    """Group decisions so that one backbone forward stays under path/token limits. max_shared_tokens: group by the
    tokens of the prefix-shared forward instead (training with prefix sharing keeps every activation of those tokens,
    no checkpointing)."""
    if max_shared_tokens:
        chunks, cur, tot = [], [], 0
        for i, d in enumerate(decisions):
            n = policy.shared_tokens(d)
            if cur and tot + n > max_shared_tokens:
                chunks.append(cur)
                cur, tot = [], 0
            cur.append(i)
            tot += n
        return chunks + ([cur] if cur else [])
    chunks, cur, cur_paths, cur_tokens = [], [], 0, 0
    for i, d in enumerate(decisions):
        if getattr(policy, "readout", "head") == "letters":
            lengths = policy.path_lengths(d)
        else:
            paths, _, _, _ = policy.build_paths([d])
            lengths = [len(p) for p in paths]
        n = len(lengths)
        width = max(lengths, default=0)
        if cur and (cur_paths + n > max_paths or (cur_paths + n) * max(width, cur_tokens) > max_tokens):
            chunks.append(cur)
            cur, cur_paths, cur_tokens = [], 0, 0
        cur.append(i)
        cur_paths += n
        cur_tokens = max(cur_tokens, width)
    if cur:
        chunks.append(cur)
    return chunks


@torch.no_grad()
def decide(policy, decisions, generators=None, mode="sample", temperature=1.0,
           autocast_dtype=torch.bfloat16, max_paths=64, max_tokens=65536):
    """Returns one dict per decision: index, log_prob, probs (full list), entropy, forced, seconds.
    `index` is None when no candidate is legal (caller must handle)."""
    out = [None] * len(decisions)
    todo = []
    for i, d in enumerate(decisions):
        legal = d.legal_indices()
        k = len(d.candidates)
        if len(legal) == 0:
            out[i] = {"index": None, "log_prob": 0.0, "probs": [0.0] * k, "entropy": 0.0,
                      "forced": True, "seconds": 0.0}
        elif len(legal) == 1 or (d.select_k > 1 and len(legal) <= d.select_k):
            if d.select_k > 1:                      # C with <= k legal frames: take them all (forced)
                probs = [0.0] * k
                for j in legal:
                    probs[j] = 1.0 / len(legal)
                out[i] = {"index": legal[0], "log_prob": 0.0, "probs": probs, "entropy": 0.0,
                          "forced": True, "seconds": 0.0, "info": {"picks": list(legal)}}
                continue
            probs = [0.0] * k
            probs[legal[0]] = 1.0
            out[i] = {"index": legal[0], "log_prob": 0.0, "probs": probs, "entropy": 0.0,
                      "forced": True, "seconds": 0.0}
        else:
            todo.append(i)
    if not todo:
        return out
    device = next(policy.parameters()).device
    sub = [decisions[i] for i in todo]
    for chunk in chunk_decisions(policy, sub, max_paths, max_tokens):
        t0 = time.perf_counter()
        batch = [sub[j] for j in chunk]
        with torch.autocast(device.type, dtype=autocast_dtype, enabled=autocast_dtype is not None
                            and device.type == "cuda"):
            logits, _ = policy(batch)
        logp = masked_log_softmax(logits.float() / temperature)
        probs = logp.exp()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        secs = (time.perf_counter() - t0) / len(chunk)
        probs_cpu, logp_cpu = probs.cpu(), logp.cpu()
        for r, j in enumerate(chunk):
            i = todo[j]
            d = decisions[i]
            k = len(d.candidates)
            p = probs_cpu[r, :k]
            lp = logp_cpu[r, :k]
            if d.select_k > 1:                      # selection C: one scoring, k masked picks
                from .selection import pick
                g = generators[i] if (generators is not None and mode == "sample") else None
                z = (logits[r, :k].float() / temperature).cpu()
                pinfo = _prior_info(policy, d)
                picks, tot, fp, fh = pick(z, d.valid, d.select_k, d.mask_radius, mode=mode, generator=g, info=pinfo)
                out[i] = {"index": picks[0], "log_prob": tot, "probs": fp.tolist(), "entropy": fh,
                          "forced": False, "seconds": secs, "info": {"picks": picks, **pinfo}}
                continue
            groups = payload_groups(d)
            if groups is not None:                  # grouped decision: one choice per group, each normalized alone
                z = (logits[r, :k].float() / temperature).cpu()
                glp = grouped_log_softmax(z, groups)
                gp = glp.exp()
                g = generators[i] if (generators is not None and mode == "sample") else None
                picks, tot, ent = [], 0.0, 0.0
                for gid in dict.fromkeys(groups):
                    mem = [j for j, x in enumerate(groups) if x == gid and d.valid[j]]
                    if not mem:
                        continue
                    q = gp[mem]
                    j = mem[int(torch.multinomial(q, 1, generator=g).item())] if mode == "sample" else \
                        mem[int(torch.argmax(q).item())]
                    picks.append(j)
                    tot += float(glp[j])
                    ent += float(-(q * glp[mem]).sum())
                out[i] = {"index": picks[0], "log_prob": tot, "probs": gp.tolist(), "entropy": ent,
                          "forced": False, "seconds": secs, "info": {"group_picks": picks}}
                continue
            if mode == "sample":
                g = generators[i] if generators is not None else None
                idx = int(torch.multinomial(p, 1, generator=g).item())
            elif mode == "greedy":
                idx = int(torch.argmax(p).item())
            else:
                raise ValueError(mode)
            assert d.valid[idx], "sampled an illegal candidate"
            finite = torch.isfinite(lp)
            ent = float(-(p[finite] * lp[finite]).sum())
            out[i] = {"index": idx, "log_prob": float(lp[idx]), "probs": p.tolist(), "entropy": ent,
                      "forced": False, "seconds": secs, "info": _prior_info(policy, d)}
    return out
