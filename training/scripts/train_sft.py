"""Short SFT of the multimodal NanoJev policy (update.md §6 M1).

Rows (scripts/gen_sft_data.py): {"input": DecisionInput json, "target": {"indices": [...]},
"decision_type", "split", ...}. Loss per decision = -log sum_{k in target set} p_k over legal
candidates (several valid frames/windows are all correct). Local microbatches combine complete decisions within
`microbatch_decisions` / `microbatch_tokens`; the global optimizer batch remains `decisions_per_step` decisions.
Trainable: LoRA (lr 2e-5), projector and Choice head (lr 1e-4); base weights and SigLIP2 frozen. With the OneJev letter
readout (policy.readout = letters, default since 2026-10-06) the Choice head and projector are unused and frozen: only
the LoRA trains, and the decision logits are the output-embedding rows of the option letters (src/policy/letters.py).
With --set policy.finetune=full: no LoRA, the whole backbone trains at sft.backbone_lr (must be set).
--overfit N first fits N training decisions (loss and target-set probability must move together).
Saves the step-0 (pre-SFT) and final checkpoints plus eval snapshots; logs parameter changes.
Rows may carry a "weight" (scripts/balance_sft_weights.py): the loss of a decision is multiplied by it,
dev metrics are reported plain and weighted (w_*), and snapshot selection then uses the weighted ones.
Resumes from runs/<name>/checkpoints/latest (weights, optimizer, RNG, data order) unless --no-resume.
SIGTERM / SIGUSR1 finish the current optimizer step and save a resumable checkpoint; incomplete runs exit 75.
Under torchrun (train_pipeline.sh, SFT_GPUS) each step's decisions and the dev rows are shared out over one rank per
GPU by prompt tokens (share_cost) and the gradients summed (same step as one process); every rank replays the
single-process RNG stream, so data order, replay picks and teacher orders match a one-GPU run, and resumes work
across GPU counts. Rank 0 writes.
--decision-types keeps only those decision types (main data, dev and replay): one run per role trains a
separate model per role (combine the runs with scripts/make_policy_set.py); --init may then be a policy set.
"""
import argparse
import collections
import datetime
import gc
import json
import os
import time
import math
import random
import signal
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.config import load_config  # noqa: E402
from src.common.runtime import reserve_gpu_memory, gpu_mem_stats, init_wandb, setup_run, wandb_log  # noqa: E402
from src.common.types import DecisionInput  # noqa: E402
from src.common.utils import append_jsonl, read_jsonl, seed_everything  # noqa: E402
from src.policy.model import apply_policy_sdpa_backend, build_policy, masked_log_softmax  # noqa: E402
from src.rollout.build import build_encoder_and_store  # noqa: E402
from src.common.utils import write_jsonl  # noqa: E402
from src.training.checkpoint import base_model_info, complete_checkpoints, load_checkpoint, save_checkpoint  # noqa: E402
from src.training.rl import _group_params, _norm, build_optimizer  # noqa: E402


train_rng = None   # set by main(): the data-order RNG, reused for teacher pick orders during training
VERIFIER_BALANCE = False   # set by main() (--verifier-balance): class-balanced verifier dev metrics select the snapshot
WHERE_OUT = ("before", "after")


def verifier_classes(row):
    """Labels used to balance a verifier row (2026-10-09): ("where", label) for a where row,
    ("frame", [True/False per frame]) for a per-frame keep row, None for Q3 rows."""
    s = row.get("source")
    if s == "evidence_where":
        return "where", row["meta"]["where"]
    if s in ("evidence_frames", "evidence_frame"):
        ev = row["meta"]["evidence"]
        return "frame", list(ev) if isinstance(ev, list) else [bool(ev)]
    return None


def balance_weights(main, replay, n_main, n_rep, clip):
    """Class weights for the verifier's where question (per row) and per-frame keep question (per frame): inverse
    frequency under the training mixture (n_main rows per step from `main`, n_rep from `replay`), clipped to
    [1/clip, clip], renormalised so the expected weight per row (where) / frame (keep) is 1."""
    import collections as _c
    eff = {"where": _c.Counter(), "frame": _c.Counter()}
    for rows, n in ((main, n_main), (replay, n_rep)):
        if not rows or n <= 0:
            continue
        for r in rows:
            k = verifier_classes(r)
            if k is None:
                continue
            if k[0] == "where":
                eff["where"][k[1]] += n / len(rows)
            else:
                for e in k[1]:
                    eff["frame"]["yes" if e else "no"] += n / len(rows)
    out = {}
    for q, cnt in eff.items():
        tot = sum(cnt.values())
        if tot <= 0:
            continue
        f = {k: v / tot for k, v in cnt.items()}
        w = {k: min(clip, max(1.0 / clip, 1.0 / (len(f) * f[k]))) for k in f}
        z = sum(f[k] * w[k] for k in f)
        out[q] = {"freq": f, "weight": {k: w[k] / z for k in f}}
    return out


def apply_balance(rows, bw):
    """Writes the class weights into training rows: where rows multiply their row weight, joint per-frame rows get
    group_weights (one per frame, in target order), single-frame rows multiply their row weight."""
    for r in rows:
        k = verifier_classes(r)
        if k is None:
            continue
        if k[0] == "where" and "where" in bw:
            r["weight"] = float(r.get("weight", 1.0)) * bw["where"]["weight"].get(k[1], 1.0)
        elif k[0] == "frame" and "frame" in bw:
            ws = [bw["frame"]["weight"].get("yes" if e else "no", 1.0) for e in k[1]]
            if r.get("source") == "evidence_frames":
                r["group_weights"] = ws
            else:
                r["weight"] = float(r.get("weight", 1.0)) * ws[0]


def _auc(pos, neg):
    """P(score of a positive > score of a negative), ties 1/2; None without both classes."""
    if not pos or not neg:
        return None
    allv = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg])
    rank, i, rs = 0.0, 0, 0.0
    while i < len(allv):                      # average ranks over ties
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        avg = (i + 1 + j) / 2.0
        rs += avg * sum(1 for x in allv[i:j] if x[1] == 1)
        i = j
    n1, n0 = len(pos), len(neg)
    return (rs - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def verifier_balanced_metrics(rows, res, dets):
    """Class-balanced dev metrics of the verifier (2026-10-09): where macro recall (the where group only), per-frame
    keep balanced accuracy (counted per frame), Q3 accuracy, and their mean `score` (snapshot selection with
    --verifier-balance). Also, for monitoring only: AUC of P(outside) = P(before) + P(after) over where rows (all /
    after an inspection / tagged injected window) and the share of injected-window frames judged keep."""
    import collections as _c
    out, rec = {}, _c.defaultdict(list)
    outside, frames, q3 = {"all": ([], []), "post": ([], [])}, [], []
    inj_where, inj_keep = [], []
    for r, x, d in zip(rows, res, dets):
        s = r.get("source")
        if s == "label_pE":
            q3.append(x[2])
        elif s == "evidence_where":
            lab = r["meta"]["where"]
            hit = d["group_hits"][0] if "group_hits" in d else x[2]
            rec[lab].append(hit)
            pr = d.get("probs")
            if pr is not None:
                po = sum(float(pr[j]) for j, pl in enumerate(r["input"]["payloads"]) if pl.get("where") in WHERE_OUT)
                keys = ["all"] + (["post"] if (r["meta"].get("inspections_here") or 0) >= 1 else [])
                for k in keys:
                    outside[k][0 if lab in WHERE_OUT else 1].append(po)
                if r["meta"].get("injected_window"):
                    inj_where.append(po)
        elif s in ("evidence_frames", "evidence_frame"):
            ev = r["meta"]["evidence"]
            ev = list(ev) if isinstance(ev, list) else [bool(ev)]
            hits = d.get("group_hits", [x[2]] * len(ev))
            frames += list(zip(ev, hits))
            if r["meta"].get("injected_window"):
                inj_keep += [not h for e, h in zip(ev, hits) if not e]    # a negative frame judged keep
    parts = []
    if rec:
        for k, v in sorted(rec.items()):
            out[f"verifier_bal/where_recall/{k}"] = sum(v) / len(v)
        out["verifier_bal/where_macro_recall"] = sum(sum(v) / len(v) for v in rec.values()) / len(rec)
        parts.append(out["verifier_bal/where_macro_recall"])
    if frames:
        tp = [h for e, h in frames if e]
        tn = [h for e, h in frames if not e]
        if tp:
            out["verifier_bal/frame_tpr"] = sum(tp) / len(tp)
        if tn:
            out["verifier_bal/frame_tnr"] = sum(tn) / len(tn)
        rates = [v for v in (out.get("verifier_bal/frame_tpr"), out.get("verifier_bal/frame_tnr")) if v is not None]
        out["verifier_bal/frame_bal_acc"] = sum(rates) / len(rates)
        parts.append(out["verifier_bal/frame_bal_acc"])
    if q3:
        out["verifier_bal/q3_acc"] = sum(q3) / len(q3)
        parts.append(out["verifier_bal/q3_acc"])
    for k, (pos, neg) in outside.items():
        a = _auc(pos, neg)
        if a is not None:
            out[f"verifier_bal/outside_auc_{k}"] = a
    if inj_where:
        out["verifier_bal/injected_where_n"] = len(inj_where)
        out["verifier_bal/injected_outside_ge_half"] = sum(p >= 0.5 for p in inj_where) / len(inj_where)
    if inj_keep:
        out["verifier_bal/injected_frames_kept"] = sum(inj_keep) / len(inj_keep)
        out["verifier_bal/injected_frames_n"] = len(inj_keep)
    if parts:
        out["verifier_bal/score"] = sum(parts) / len(parts)
    return out


def decision_loss(policy, row, device, detail=None, teacher=None, logits=None, inp=None):
    """-log P(target set) for one decision; returns (loss tensor, target-set prob, argmax in set).
    Selection-C rows (input.select_k > 1, target.picks): per-step target-set loss teacher-forced on the picks
    (src/policy/selection.py); `detail` (dict) receives the greedy picks' hit rate and the prior-only one.
    teacher: (picks, supervised) drawn beforehand by teacher_picks, instead of drawing them here."""
    inp = inp if inp is not None else DecisionInput.from_json(row["input"])
    if logits is None:
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, _ = policy([inp])
    else:
        logits = logits[:len(inp.candidates)].unsqueeze(0)
    if inp.select_k > 1:
        from src.policy.selection import pick, set_target_nll, teacher_sequence
        z = logits.float()[0, :len(inp.candidates)]
        tg = row["target"]["indices"]
        # teacher order sampled per call (uniform among the targets available at each step; unsupervised filler
        # steps when no target is available); evaluation uses a per-row fixed seed
        if teacher is None:
            rng = train_rng if (train_rng is not None and detail is None) else random.Random(row["sample_id"] + row.get("source", ""))
            teacher = teacher_sequence(inp.valid, tg, inp.select_k, inp.mask_radius, rng)
        picks, sup = teacher
        loss, p, n_sup = set_target_nll(z, inp.valid, picks, tg, inp.mask_radius, supervised=sup)
        if detail is not None:
            detail["supervised_steps"] = n_sup
            detail["short_sequence"] = int(n_sup < inp.select_k)
            g, _, _, _ = pick(z.detach().cpu(), inp.valid, inp.select_k, inp.mask_radius, mode="greedy")
            detail["hit"] = sum(j in tg for j in g) / len(g)
            if policy.prior_scorer is not None:
                pz = policy.prior_scorer.scores(inp, policy.feature_store).cpu()
                pz = torch.where(torch.tensor(inp.valid), pz, torch.full_like(pz, float("-inf")))
                pg, _, _, _ = pick(pz, inp.valid, inp.select_k, inp.mask_radius, mode="greedy")
                detail["prior_hit"] = sum(j in tg for j in pg) / len(pg)
        first = int(torch.where(torch.tensor(inp.valid, device=z.device), z, torch.full_like(z, float("-inf"))).argmax())
        return loss, p, first in tg
    from src.policy.model import grouped_log_softmax, payload_groups
    groups = payload_groups(inp)
    if groups is not None:     # grouped decision (joint per-frame verifier): one target per group, losses summed
        glp = grouped_log_softmax(logits.float()[0, :len(inp.candidates)], groups)
        tg = row["target"]["indices"]
        assert len({groups[j] for j in tg}) == len(tg) and torch.isfinite(glp[tg]).all(), "one legal target per group"
        z = glp.detach()
        hits = [int(max((j for j, x in enumerate(groups) if x == groups[t]), key=lambda j: float(z[j]))) == t for t in tg]
        if detail is not None:                     # --verifier-balance dev metrics: per-group hits, probabilities
            detail["group_hits"], detail["probs"] = hits, z.exp().cpu()
        gw = row.get("group_weights")              # --verifier-balance: per-frame class weights (mean over groups kept)
        if gw is not None:
            lt = -(glp[tg] * torch.tensor(gw, device=glp.device, dtype=glp.dtype)).mean()
        else:
            lt = -glp[tg].mean()
        return lt, float(glp[tg].exp().mean().detach()), all(hits)    # F: mean over groups (one row = one unit)
    lp = masked_log_softmax(logits.float())[0]
    if detail is not None:
        detail["probs"] = lp.detach().exp().cpu()
    tgt = torch.tensor(row["target"]["indices"], device=device)
    assert torch.isfinite(lp[tgt]).all(), "target on an illegal candidate"
    set_lp = torch.logsumexp(lp[tgt], 0)
    soft = row["target"].get("soft")
    if soft is not None:     # soft target (2026-10-05, controller imitation that keeps entropy): -sum_i q_i log p_i
        q = torch.tensor(soft, device=device, dtype=lp.dtype)[:lp.shape[0]]
        assert all(v <= 1e-9 or bool(torch.isfinite(lp[i])) for i, v in enumerate(soft)), "soft mass on an illegal candidate"
        loss = -(q * torch.where(q > 0, lp, torch.zeros_like(lp))).sum()
        return loss, float(set_lp.exp().detach()), int(lp.argmax()) in row["target"]["indices"]
    return -set_lp, float(set_lp.exp().detach()), int(lp.argmax()) in row["target"]["indices"]


def lr_factor(step, steps, sc):
    """LR multiplier at `step` (1-based) of `steps`: linear warmup over sft.warmup_steps, then constant
    (sft.lr_schedule = constant) or cosine decay to sft.min_lr_ratio x the base LR at the last step (cosine;
    LoRA lr 1e-4 + warmup + cosine)."""
    wu = int(sc.get("warmup_steps", 0) or 0)
    if wu > 0 and step <= wu:
        return step / wu
    if sc.get("lr_schedule", "constant") != "cosine":
        return 1.0
    lo = float(sc.get("min_lr_ratio", 0.1))
    t = (step - wu) / max(1, steps - wu)
    return lo + (1.0 - lo) * 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, t))))


class Ranks:
    """One process per GPU under torchrun (NCCL; gloo on CPU in tests); world 1 = the single-process run (no-ops)."""

    def __init__(self, env=True):
        self.world, self.rank = (int(os.environ.get("WORLD_SIZE", "1")), int(os.environ.get("RANK", "0"))) if env else (1, 0)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.world > 1:
            import torch.distributed as dist
            if self.device == "cuda":
                torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            dist.init_process_group("nccl" if self.device == "cuda" else "gloo", timeout=datetime.timedelta(hours=2))
            self.dist = dist

    @property
    def main(self):
        return self.rank == 0

    def share(self, costs):
        """This rank's item indices: longest-first greedy bins of similar total cost (the same split on every rank)."""
        if self.world == 1:
            return list(range(len(costs)))
        load, mine = [0.0] * self.world, []
        for i in sorted(range(len(costs)), key=lambda i: (-costs[i], i)):
            b = min(range(self.world), key=lambda j: (load[j], j))
            load[b] += costs[i]
            if b == self.rank:
                mine.append(i)
        return sorted(mine)

    def total(self, *xs):
        if self.world == 1:
            return xs
        t = torch.tensor(xs, dtype=torch.float64, device=self.device)
        self.dist.all_reduce(t)
        return tuple(t.tolist())

    def any(self, flag):
        return bool(self.total(float(flag))[0])

    def gather(self, obj):
        if self.world == 1:
            return [obj]
        out = [None] * self.world
        self.dist.all_gather_object(out, obj)
        return out

    def broadcast(self, obj):
        if self.world == 1:
            return obj
        values = [obj if self.main else None]
        self.dist.broadcast_object_list(values, src=0)
        return values[0]

    def reduce_grads(self, params):
        """Sums the gradients over ranks; a parameter no rank produced a gradient for keeps grad None (as one process)."""
        if self.world == 1:
            return
        has = torch.tensor([p.grad is not None for p in params], dtype=torch.float32, device=self.device)
        self.dist.all_reduce(has)
        flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1).float() for p in params])
        self.dist.all_reduce(flat)
        o = 0
        for p, h in zip(params, has.tolist()):
            p.grad = flat[o:o + p.numel()].view_as(p).to(p.dtype).clone() if h else None
            o += p.numel()

    def barrier(self):
        if self.world > 1:
            self.dist.barrier()


def cost(row):
    return len(row["input"]["candidates"]) + 1


def share_cost(policy, row):
    """Rank-sharing cost of a decision: the prompt tokens its training forward runs (a grouped decision's prefix once
    when the policy shares it), so ranks get similar compute; cost(row) for a policy without path lengths.
    2026-10-11: the candidate count weighed a 32-candidate frame row 6.6x a localize row of similar length, and every
    step waited for the rank that had received several short rows. Cached in the row for this policy."""
    hit = row.get("_share_cost")
    if hit is not None and hit[0] == id(policy):
        return hit[1]
    if not hasattr(policy, "path_lengths"):
        return cost(row)
    inp = DecisionInput.from_json(row["input"])
    lengths = policy.path_lengths(inp)
    shared = getattr(policy, "prefix_sharing_train", False) and (
        getattr(policy, "readout", "head") == "letters" or getattr(policy, "prefix_sharing", False))
    value = policy.shared_tokens(inp) if shared and len(lengths) > 1 else sum(lengths)
    row["_share_cost"] = (id(policy), value)
    return value


def teacher_picks(rows, rng):
    """{row index: (picks, supervised)} for the selection-C rows, drawn from rng in row order, as decision_loss draws
    them one row at a time: every rank replays the single-process RNG stream but computes only its own rows."""
    from src.policy.selection import teacher_sequence
    out = {}
    for i, r in enumerate(rows):
        inp = r["input"]
        if int(inp.get("select_k", 1)) > 1:
            out[i] = teacher_sequence([bool(v) for v in inp["valid"]], r["target"]["indices"], int(inp["select_k"]),
                                      int(inp.get("mask_radius", 0)), rng)
    return out


def path_lengths(policy, row):
    """Token lengths of a decision's paths (cost(row) for a policy without them), cached in the row for this policy:
    every dev pass otherwise tokenized every dev decision again (2026-10-14)."""
    hit = row.get("_path_lengths")
    if hit is not None and hit[0] == id(policy):
        return hit[1]
    value = policy.path_lengths(DecisionInput.from_json(row["input"])) if hasattr(policy, "path_lengths") else [cost(row)]
    row["_path_lengths"] = (id(policy), value)
    return value


def microbatches(policy, rows, indices, cap=None, budget=None):
    """Keep global sampling/rank assignment intact; only combine local rows within padded token/item limits (the
    training limits unless cap / budget are given)."""
    cap = int(cap or getattr(policy, "sft_microbatch_decisions", 1))
    budget = int(budget or getattr(policy, "sft_microbatch_tokens", 8192))
    if cap < 1 or budget < 1:
        raise ValueError("SFT microbatch limits must be positive")
    chunk, count, width = [], 0, 0
    for i in indices:
        lengths = path_lengths(policy, rows[i])
        n, size = len(lengths), max(lengths, default=0)
        if chunk and (len(chunk) >= cap or (count + n) * max(width, size) > budget):
            yield chunk
            chunk, count, width = [], 0, 0
        chunk.append(i)
        count, width = count + n, max(width, size)
    if chunk:
        yield chunk


def forward_rows(policy, rows, device):
    inputs = [DecisionInput.from_json(r["input"]) for r in rows]
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        logits, _ = policy(inputs)
    return inputs, logits


def row_eval(policy, r, device, teacher, frame_detail, prior, logits=None, inp=None):
    detail = {} if VERIFIER_BALANCE and r["decision_type"] == "verifier" else None
    loss, p, hit = decision_loss(policy, r, device, detail=detail, teacher=teacher, logits=logits, inp=inp)
    out = {"loss": float(loss), "p": float(p), "hit": bool(hit)}
    if detail is not None:
        out["verifier_detail"] = detail
    if frame_detail and r["decision_type"] == "frame":
        out["detail"] = {}
        decision_loss(policy, r, device, detail=out["detail"], logits=logits, inp=inp)
    if prior and r["decision_type"] == "localize":
        inp = DecisionInput.from_json(r["input"])
        pz = policy.prior_scorer.scores(inp, policy.feature_store).cpu()
        pz = torch.where(torch.tensor(inp.valid), pz, torch.full_like(pz, float("-inf")))
        out["prior_hit"] = int(pz.argmax()) in r["target"]["indices"]
    return out


def evaluate(policy, rows, device, ranks=None):
    """Dev metrics. Each rank computes its share of the rows and every rank aggregates the gathered results; teacher
    orders are drawn from train_rng in the single-process order (types sorted, rows in order)."""
    ranks = ranks or RANKS
    types = sorted({r["decision_type"] for r in rows})
    order = [i for typ in types for i, r in enumerate(rows) if r["decision_type"] == typ]
    teach = {}
    if train_rng is not None:
        teach = {order[j]: v for j, v in teacher_picks([rows[i] for i in order], train_rng).items()}
    frame_detail = any(r["decision_type"] == "frame" and r["input"].get("select_k", 1) > 1 for r in rows)
    prior = policy.prior_scorer is not None and any(r["decision_type"] == "localize" and r["input"].get("prior_texts")
                                                    for r in rows)
    res = {}
    clear_cache = getattr(policy, "clear_static_cache", None)
    if clear_cache:
        clear_cache()                              # optimizer steps invalidate snapshots from earlier dev passes
    # 2026-10-14: a dev pass took as long as the 100 training steps before it (mixed run's verifier1: 11 of 20 min).
    # Forward only, so its microbatches can be larger (sft.eval_microbatch_*). Rows stay in file order: a question's
    # decisions are adjacent there and share its cached static head (static_prefix_cache, LRU).
    mine = ranks.share([share_cost(policy, r) for r in rows])
    try:
        with torch.no_grad():
            for indices in microbatches(policy, rows, mine, getattr(policy, "sft_eval_microbatch_decisions", None),
                                        getattr(policy, "sft_eval_microbatch_tokens", None)):
                inputs, logits = forward_rows(policy, [rows[i] for i in indices], device)
                for j, i in enumerate(indices):
                    res[i] = row_eval(policy, rows[i], device, teach.get(i), frame_detail, prior, logits[j], inputs[j])
    finally:
        if clear_cache:
            clear_cache()                          # release inference snapshots before the next training microbatch
    for part in ranks.gather(res):
        res.update(part)
    return aggregate(rows, [res[i] for i in range(len(rows))], types)


def aggregate(rows, res, types):
    out = {}
    for typ in types:
        rs = [r for r in rows if r["decision_type"] == typ]
        xs = [x for r, x in zip(rows, res) if r["decision_type"] == typ]
        if VERIFIER_BALANCE and typ == "verifier":
            out.update(verifier_balanced_metrics(rs, [(x["loss"], x["p"], x["hit"]) for x in xs],
                                                [x["verifier_detail"] for x in xs]))
        out[f"{typ}/loss"] = sum(x["loss"] for x in xs) / len(xs)
        out[f"{typ}/target_prob"] = sum(x["p"] for x in xs) / len(xs)
        out[f"{typ}/argmax_in_target"] = sum(x["hit"] for x in xs) / len(xs)
        out[f"{typ}/n"] = len(rs)
        if typ in ("controller", "verifier"):
            # per target action / tier (stage-4 gate: agreement with the post-RL controller per action)
            by = collections.defaultdict(list)
            for r, x in zip(rs, xs):
                by[r["input"]["candidate_ids"][r["target"]["indices"][0]]].append(x["hit"])
            for k, v in sorted(by.items()):
                out[f"{typ}/argmax_in_target/{k}"] = sum(v) / len(v)
                out[f"{typ}/n/{k}"] = len(v)
        if typ == "verifier":         # 2026-10-06 review: the three verifier questions separately (Q3 rows are few)
            src_name = {"label_pE": "q3", "evidence_frames": "frame", "evidence_frame": "frame",
                        "evidence_where": "where"}
            by_src = collections.defaultdict(list)
            for r, x in zip(rs, xs):
                by_src[src_name.get(r.get("source"), "other")].append(x)
            if len(by_src) > 1:
                for k, v in sorted(by_src.items()):
                    out[f"verifier_{k}/loss"] = sum(x["loss"] for x in v) / len(v)
                    out[f"verifier_{k}/argmax_in_target"] = sum(x["hit"] for x in v) / len(v)
                    out[f"verifier_{k}/target_prob"] = sum(x["p"] for x in v) / len(v)
                    out[f"verifier_{k}/n"] = len(v)
        if typ == "localize" and any("prior_hit" in x for x in xs):
            out["localize/prior_argmax_in_target"] = sum(x["prior_hit"] for x in xs) / len(xs)
        if typ == "frame" and any("detail" in x for x in xs):
            # gate metric: hit rate of the greedy selection-C picks, per source kind and origin
            det = [(r, x["detail"]) for r, x in zip(rs, xs)]
            keys = sorted({(r.get("source", "?").split(":")[-1], "qa" if r.get("origin") != "caption" else "cap")
                           for r in rs})
            for kind, org in keys:
                sub = [d for r, d in det if r.get("source", "?").split(":")[-1] == kind
                       and ("qa" if r.get("origin") != "caption" else "cap") == org]
                for m in ("hit", "prior_hit", "short_sequence"):
                    vals = [d[m] for d in sub if m in d]
                    if vals:
                        out[f"{kind}_{org}/{m}"] = sum(vals) / len(vals)
                out[f"{kind}_{org}/n"] = len(sub)
        ws = [float(r.get("weight", 1.0)) for r in rs]
        if any(abs(w - 1.0) > 1e-9 for w in ws):
            tw = sum(ws)
            out[f"{typ}/w_loss"] = sum(w * x["loss"] for w, x in zip(ws, xs)) / tw
            out[f"{typ}/w_target_prob"] = sum(w * x["p"] for w, x in zip(ws, xs)) / tw
            out[f"{typ}/w_argmax_in_target"] = sum(w * x["hit"] for w, x in zip(ws, xs)) / tw
    return out


RANKS = Ranks(env=False)   # main() replaces it with the torchrun ranks


def selection_score(lg):
    """Dev score used both for early stopping and checkpoint selection."""
    if lg.get("dev/verifier_bal/score") is not None:
        return lg["dev/verifier_bal/score"]
    types = sorted({k.split("/")[1] for k in lg if k.startswith("dev/") and k.endswith("/argmax_in_target")})
    if any(t.startswith("verifier_") for t in types):
        types = [t for t in types if t != "verifier"]
    vals = [lg.get(f"dev/{t}/w_argmax_in_target", lg.get(f"dev/{t}/argmax_in_target")) for t in types]
    return sum(vals) / len(vals) if types and all(v is not None for v in vals) else None


class EarlyStop:
    """Stop after patience dev evaluations without improvement; patience <= 0 disables it."""

    def __init__(self, patience, min_delta=0.0):
        self.patience, self.min_delta, self.best, self.since = patience, min_delta, None, 0

    def update(self, score):
        if self.best is None or score > self.best + self.min_delta:
            self.best, self.since = score, 0
        else:
            self.since += 1
        return self.patience > 0 and self.since >= self.patience


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="+", default=[str(ROOT / "configs/default.yaml")])
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", required=True, help="sft_decisions.jsonl from gen_sft_data.py")
    ap.add_argument("--overfit", type=int, default=0, help="fit only the first N train decisions")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--epochs", type=float, default=None,
                    help="steps = ceil(epochs * train decisions / main-data decisions per step)")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--dev-limit", type=int, default=None, help="evaluate on the first N dev decisions (smoke tests)")
    ap.add_argument("--init", default=None, help="start from this checkpoint's trainable weights (or, with one "
                    "--decision-types role, from that role's entry of a policy-set .json)")
    ap.add_argument("--decision-types", nargs="+", default=None, choices=["controller", "localize", "frame", "verifier"],
                    help="train, evaluate and select on these decision types only")
    ap.add_argument("--replay", default=None, help="extra decision file mixed into every step (train split)")
    ap.add_argument("--replay-frac", type=float, default=0.5, help="fraction of each step drawn from --replay")
    ap.add_argument("--verifier-balance", action="store_true",
                    help="2026-10-09: class weights for the verifier's where (per row) and per-frame "
                         "keep (per frame) losses, and snapshot selection / early stopping on the class-balanced verifier "
                         "score (where macro recall, per-frame balanced accuracy, Q3 accuracy)")
    ap.add_argument("--balance-clip", type=float, default=4.0, help="with --verifier-balance: weights in [1/c, c]")
    ap.add_argument("--require-fit", action="store_true",
                    help="overfit check: exit 1 unless the fit-set loss halves and mean target prob >= 0.9")
    ap.add_argument("--set", nargs="*", default=[])
    args = ap.parse_args()
    global RANKS
    RANKS = ranks = Ranks()
    if not ranks.main:
        sys.stdout = open(os.devnull, "w")              # rank 0 prints, logs and saves
    reserve_gpu_memory()                                # JEV_RESERVE_GB, shared machines only
    cfg = load_config(args.config, args.set)
    if cfg["policy"].get("finetune", "lora") != "full" and cfg["policy"].get("init") != "qwen35_4b":
        # 0.6B backbone: LoRA deprecated 2026-09-29; Qwen3.5-4B (design B, 2026-09-30) is trained with LoRA
        raise SystemExit("LoRA is deprecated (2026-09-29) for the 0.6B backbone: use policy.finetune=full")
    apply_policy_sdpa_backend(cfg)
    seed_everything(cfg["project"]["seed"])
    run_dir = setup_run(cfg, args.name) if ranks.main else Path(cfg["paths"]["runs_root"]) / args.name
    ranks.barrier()
    wb = init_wandb(cfg, run_dir, args.name, "sft") if ranks.main else None
    dev = torch.device("cuda")
    sc = cfg["sft"]
    keep = (lambda r: r["decision_type"] in args.decision_types) if args.decision_types else (lambda r: True)
    data = [r for r in read_jsonl(args.data) if keep(r)]
    train = [r for r in data if r["split"] == "train"]
    held = [r for r in data if r["split"] == "dev"][: args.dev_limit]
    rng = random.Random(cfg["project"]["seed"])
    global train_rng
    train_rng = rng
    if args.overfit:
        rng.shuffle(train)
        by = {}
        for r in train:                              # keep all three decision types in the subset
            by.setdefault(r["decision_type"], []).append(r)
        inter = []
        for i in range(max(len(v) for v in by.values())):
            inter += [by[t][i] for t in sorted(by) if i < len(by[t])]
        train = inter[:args.overfit]
        held = train                                 # the fit check is measured on the same rows
    counts = {t: sum(r["decision_type"] == t for r in train) for t in ("controller", "localize", "frame", "verifier")}
    print(json.dumps({"train_by_type": counts, "dev": len(held), "overfit": args.overfit}), flush=True)
    # native (OneJev / Qwen3.5) frame tokens come from the policy's own token store (build_policy); the SigLIP store is
    # only needed by the older pooled / prior variants (2026-10-06 review: loading it made SigLIP a hard requirement)
    store = None
    if cfg["policy"].get("frame_tokens", "pooled") != "qwen_native" or cfg["policy"].get("siglip_prior"):
        _, store = build_encoder_and_store(cfg, device="cuda")  # all features are cached already
    policy = build_policy(cfg, feature_store=store, device="cuda")
    policy.sft_microbatch_decisions = int(sc.get("microbatch_decisions", 1))
    policy.sft_microbatch_tokens = int(sc.get("microbatch_tokens", 8192))
    policy.sft_eval_microbatch_decisions = int(sc.get("eval_microbatch_decisions") or policy.sft_microbatch_decisions)
    policy.sft_eval_microbatch_tokens = int(sc.get("eval_microbatch_tokens") or policy.sft_microbatch_tokens)
    if cfg["policy"]["gradient_checkpointing"]:
        policy.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.init:
        from src.policy.policy_set import read_policy_set, role_checkpoint
        from src.training.checkpoint import load_policy_weights
        if read_policy_set(args.init) is not None and len(args.decision_types or []) != 1:
            raise SystemExit("--init with a policy set needs exactly one --decision-types role")
        load_policy_weights(role_checkpoint(args.init, (args.decision_types or [None])[0]), policy)
    replay = [r for r in read_jsonl(args.replay) if r["split"] == "train" and keep(r)] if args.replay else []
    opt = build_optimizer(policy, sc["lora_lr"], sc["new_lr"], sc["weight_decay"], lr_backbone=sc.get("backbone_lr"))
    info = base_model_info(cfg, policy)
    info.update(sft_init=args.init, sft_data=args.data, sft_replay=args.replay, replay_frac=args.replay_frac if replay else 0,
                decision_types=args.decision_types)
    info["trainable_names"] = policy.trainable_names()
    n_rep = int(round(sc["decisions_per_step"] * args.replay_frac)) if replay else 0
    n_main = sc["decisions_per_step"] - n_rep
    steps = args.steps or (math.ceil(args.epochs * len(train) / n_main) if args.epochs else sc["steps"])
    info["sft_steps"] = steps
    global VERIFIER_BALANCE
    VERIFIER_BALANCE = args.verifier_balance
    if args.verifier_balance:
        bw = balance_weights(train, replay, n_main, n_rep, args.balance_clip)
        apply_balance(train, bw)
        apply_balance(replay, bw)
        info["verifier_balance"] = dict(bw, clip=args.balance_clip)
        print(json.dumps({"verifier_balance": bw}), flush=True)
    groups_p = _group_params(policy)
    order, start = [], 1
    # 2026-10-17 (Standard SLA): a latest checkpoint that does not load (cut off by an upload or a preemption) gives
    # way to the previous one; when none loads, stop rather than train on weights loaded half-way
    ck, unreadable = None, []
    for c in ([] if args.no_resume else complete_checkpoints(run_dir / "checkpoints")):
        try:
            st = load_checkpoint(c, policy, opt)
        except Exception as exc:
            unreadable.append(str(c))
            print(json.dumps({"unreadable_checkpoint": str(c), "error": repr(exc)[:300]}), flush=True)
            continue
        ck = c
        break
    if ck is None and unreadable:
        raise SystemExit(f"no checkpoint of {run_dir} loads ({unreadable}); "
                         f"remove {run_dir / 'checkpoints'} to start over")
    if ck is not None:              # resume: weights, optimizer, global RNGs, data order, sampler RNG
        start, order = st["step"] + 1, list(st["order"])
        rng.setstate(st["py_rng"])
        if ranks.main:
            kept = [lg for lg in read_jsonl(run_dir / "metrics.jsonl") if lg["step"] <= st["step"]]
            write_jsonl(run_dir / "metrics.jsonl", kept)
        print(json.dumps({"resumed_from": str(ck), "step": st["step"]}), flush=True)
    init = {k: [p.detach().clone() for p in v] for k, v in groups_p.items()}   # param change since (re)start
    if ck is None:
        log0 = {"step": 0, **{f"dev/{k}": v for k, v in evaluate(policy, held, dev).items()}}
        print(json.dumps(log0), flush=True)
        if ranks.main:
            append_jsonl(run_dir / "metrics.jsonl", [log0])
            save_checkpoint(run_dir / "ckpt_step0", policy, None, state={"step": 0}, base_info=info, keep_previous=False)
    print(json.dumps({"steps": steps, "start": start, "main_per_step": n_main, "replay_per_step": n_rep,
                      "train_decisions": len(train)}), flush=True)
    stop = {"requested": False, "signal": None}
    def request_stop(signum, _frame):
        stop.update(requested=True, signal=signum)
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGUSR1, request_stop)
    def save_progress(step):
        if not ranks.main:
            return
        state = {"step": step, "order": list(order), "py_rng": rng.getstate()}
        save_checkpoint(run_dir / "checkpoints", policy, opt, base_info=info, state=state)
        save_checkpoint(run_dir / "snapshots" / f"step_{step:04d}", policy, None, state={"step": step},
                        base_info=info, keep_previous=False)
    def interrupted(step):
        save_progress(step)
        print(json.dumps({"SFT_INTERRUPTED": True, "step": step, "target_steps": steps,
                          "signal": stop["signal"], "checkpoint": str(run_dir / "checkpoints/latest")}), flush=True)
        ranks.barrier()
        raise SystemExit(75)
    last_completed = start - 1
    params = [p for p in policy.parameters() if p.requires_grad]
    # early stopping (no step cap, an early-stop rule instead; sft.early_stop_patience = 0 is off):
    # stop once the dev selection metric has not beaten its best for `patience` consecutive dev evaluations; the
    # snapshot selection below still picks the best evaluated step. A marker file keeps a rerun from training on.
    # Fit checks must finish their requested steps. Rank 0 reads resume metadata after its checkpoint/log writes,
    # then broadcasts it so every rank restores the same stopping state even on a remote filesystem.
    stopper = EarlyStop(0 if args.overfit else int(sc.get("early_stop_patience", 0) or 0),
                        float(sc.get("early_stop_min_delta", 0.0) or 0.0))
    progress = None
    if ranks.main:
        marker = run_dir / "early_stopped.json"
        progress = {"logs": read_jsonl(run_dir / "metrics.jsonl"),
                    "stopped": json.loads(marker.read_text()) if marker.exists() and ck is not None else None}
    progress = ranks.broadcast(progress)
    for lg in progress["logs"]:       # resume: the evaluations already done count
        if lg.get("step", 0) > 0 and selection_score(lg) is not None:
            stopper.update(selection_score(lg))
    if progress["stopped"] is not None:
        print(json.dumps({"early_stopped_before": progress["stopped"]}), flush=True)
        start = steps + 1
    # A full garbage collection walks every tracked object: with the rows, feature tables and model wrappers of setup
    # (~40 GB per rank at 128 frames) one pass held a rank for 60-85 s while the other ranks waited in the all-reduce
    # (2026-10-18, a verifier run at 128 frames). Setup lives to the end, so it is frozen out of
    # the walks; objects made during training are still collected.
    gc.collect()
    gc.freeze()
    for step in range(start, steps + 1):
        if ranks.any(stop["requested"]):
            interrupted(last_completed)
        if len(order) < sc["decisions_per_step"]:      # same refill rule as the earlier runs
            more = list(range(len(train)))
            rng.shuffle(more)
            order += more
        batch = [train[i] for i in order[:n_main]] + [rng.choice(replay) for _ in range(n_rep)]
        order = order[n_main:]
        policy.train()
        for g in opt.param_groups:
            g.setdefault("base_lr", g["lr"])
            g["lr"] = g["base_lr"] * lr_factor(step, steps, sc)
        opt.zero_grad(set_to_none=True)
        t_step = time.time()
        teach = teacher_picks(batch, rng)            # every rank: the RNG stream of one process
        tot, psum = 0.0, 0.0
        for indices in microbatches(policy, batch, ranks.share([share_cost(policy, r) for r in batch])):
            inputs, logits = forward_rows(policy, [batch[i] for i in indices], dev)
            losses = []
            for j, i in enumerate(indices):
                r = batch[i]
                loss, p, _ = decision_loss(policy, r, dev, teacher=teach.get(i), logits=logits[j], inp=inputs[j])
                w = float(r.get("weight", 1.0))
                losses.append(loss * w / len(batch))  # global batch normalization; reduce_grads still sums ranks
                psum += float(p)
            combined = torch.stack(losses).sum()
            combined.backward()
            tot += float(combined.detach())
            del combined, losses, logits, inputs
        ranks.reduce_grads(params)
        tot, psum = ranks.total(tot, psum)
        gn = float(torch.nn.utils.clip_grad_norm_(params, sc["grad_clip"]))
        opt.step()
        log = {"step": step, "loss": tot, "batch_target_prob": psum / len(batch), "grad_norm": gn,
               "lr": opt.param_groups[0]["lr"], "lr_scale": opt.param_groups[0]["lr"] / opt.param_groups[0]["base_lr"], "step_sec": time.time() - t_step,
               **({"prior_gate": float(policy.prior_gate.weight)} if policy.prior_gate is not None else {}),
               **{f"param_change_{k}": _norm([p.detach() - q for p, q in zip(groups_p[k], init[k])])
                  for k in groups_p}, **gpu_mem_stats()}
        do_eval = step % sc["eval_every"] == 0 or step == steps
        snap = int(sc.get("snapshot_every", 0) or 0)     # 2026-09-30: snapshots + resume checkpoints without a dev eval
        if do_eval or (snap > 0 and step % snap == 0):
            if do_eval:
                log.update({f"dev/{k}": v for k, v in evaluate(policy, held, dev).items()})
            save_progress(step)
        if ranks.main:
            append_jsonl(run_dir / "metrics.jsonl", [log])
        wandb_log(wb, log, step=step)
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in log.items()}), flush=True)
        if not math.isfinite(tot):
            raise RuntimeError("non-finite SFT loss")
        last_completed = step
        if do_eval and selection_score(log) is not None:
            if ranks.any(stopper.update(selection_score(log))):
                early_stop = {"step": step, "best_dev_selection_metric": stopper.best,
                              "patience": stopper.patience, "min_delta": stopper.min_delta}
                if ranks.main:
                    (run_dir / "early_stopped.json").write_text(json.dumps(early_stop) + "\n")
                    print(json.dumps({"early_stop": early_stop}), flush=True)
                    if wb:
                        wb.log({"early_stop_step": step}, step=step)
                break
        if ranks.any(stop["requested"]) and step < steps:
            interrupted(step)
    ranks.barrier()
    if not ranks.main:
        return
    save_checkpoint(run_dir / "ckpt_final", policy, opt, state={"step": last_completed}, base_info=info, keep_previous=False)
    if wb:
        wb.finish()
    # snapshot selection on dev: mean over decision types of the fraction of decisions whose top-1
    # candidate is in the correct set (ties -> earlier step)
    logs = [json.loads(line) for line in open(run_dir / "metrics.jsonl")]
    types = sorted({k.split("/")[1] for lg in logs for k in lg if k.startswith("dev/") and k.endswith("/argmax_in_target")})
    if any(t.startswith("verifier_") for t in types):      # the verifier's questions weigh equally, not by row count
        types = [t for t in types if t != "verifier"]
    sel = "w_argmax_in_target" if any(k.endswith("/w_argmax_in_target") for lg in logs for k in lg) else "argmax_in_target"
    def metric(lg, t):                # the weighted metric where a type has one, else the plain one
        return lg.get(f"dev/{t}/{sel}", lg.get(f"dev/{t}/argmax_in_target"))
    scored = [(sum(metric(lg, t) for t in types) / len(types), -lg["step"], lg["step"])
              for lg in logs if types and all(metric(lg, t) is not None for t in types) and lg["step"] > 0]
    if args.verifier_balance:      # the same class-balanced score as the early stop (step 0 stays out, as before)
        sel = "verifier_bal/score"
        scored = [(lg["dev/verifier_bal/score"], -lg["step"], lg["step"]) for lg in logs
                  if lg["step"] > 0 and lg.get("dev/verifier_bal/score") is not None]
        s0 = next((lg.get("dev/verifier_bal/score") for lg in logs if lg["step"] == 0), None)
        print(json.dumps({"verifier_balanced_score_step0": s0}), flush=True)
    if scored:
        best = max(scored)
        (run_dir / "best_snapshot.txt").write_text(str(run_dir / "snapshots" / f"step_{best[2]:04d}" / "latest") + "\n")
        print(json.dumps({"best_step": best[2], "dev_selection_metric": best[0], "selection_key": sel}), flush=True)
    if args.require_fit:
        first, last = logs[0], logs[-1]
        l0 = sum(first[f"dev/{t}/loss"] for t in types) / len(types)
        l1 = sum(last[f"dev/{t}/loss"] for t in types) / len(types)
        p1 = sum(last[f"dev/{t}/target_prob"] for t in types) / len(types)
        ok = l1 <= 0.5 * l0 and p1 >= 0.9
        print(json.dumps({"fit_check": ok, "loss_start": l0, "loss_end": l1, "target_prob_end": p1}), flush=True)
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
