"""Multi-GPU SFT (train_sft.py under torchrun, SFT_GPUS): rank shares of a step and of the dev rows reproduce the
single-process run - same RNG stream, same metrics, gradients that add up. Tiny linear policy, CPU, no torch.distributed.

python -m unittest discover -s tests -p 'test_sft_ranks.py' -v
"""
from pathlib import Path
import random
import sys
import unittest
import zlib

try:
    import torch
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))
    from scripts import train_sft as T
    from src.common.types import DecisionInput
except ImportError:                              # pragma: no cover - needs torch and the repo's imports
    torch = T = None


class LinearPolicy(torch.nn.Module if torch else object):
    prior_scorer = feature_store = None

    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(8, generator=torch.Generator().manual_seed(3)))
        self.calls = 0

    def forward(self, inputs):
        self.calls += 1
        k = max(len(i.candidates) for i in inputs)
        out = torch.full((len(inputs), k), float("-inf"))
        for r, inp in enumerate(inputs):
            f = torch.stack([torch.randn(8, generator=torch.Generator().manual_seed(zlib.crc32(c.encode())))
                             for c in inp.candidate_ids])
            out[r, :len(inp.candidate_ids)] = torch.where(torch.tensor(inp.valid), f @ self.w, torch.tensor(float("-inf")))
        return out, None


def rows(n=24):
    out = []
    for i in range(n):
        kind = ("localize", "frame", "verifier")[i % 3]
        k = 6 + i % 5
        sel = 2 if kind == "frame" else 1
        inp = DecisionInput(kind, [], [[] for _ in range(k)], [f"r{i}c{c}" for c in range(k)], [c != 3 for c in range(k)],
                            select_k=sel)
        tg = [0, 2] if kind == "frame" else [i % 3]
        out.append({"input": inp.to_json(), "target": {"indices": tg}, "decision_type": kind, "sample_id": f"s{i}",
                    "source": ("label_pE", "evidence_frames", "evidence_where")[i // 3 % 3] if kind == "verifier" else "qa:clue",
                    "split": "dev", "weight": 1.0 + (i % 4) / 4})
    return out


def balanced_rows():
    out = []
    for i in range(12):
        source = ("evidence_where", "evidence_frames", "label_pE")[i % 3]
        if source == "evidence_where":
            labels = ["before", "inside_small", "inside_large", "after"]
            target = [i // 3]
            payloads = [{"where": x, "group": 0} for x in labels]
            meta = {"where": labels[target[0]], "inspections_here": 1, "injected_window": target[0] in (0, 3)}
        elif source == "evidence_frames":
            target = [i % 2, 3]
            payloads = [{"keep": j % 2 == 0, "group": j // 2} for j in range(4)]
            meta = {"evidence": [target[0] == 0, False], "injected_window": True}
        else:
            target, payloads, meta = [i // 3], [{"tier": j} for j in range(4)], {}
        inp = DecisionInput("verifier", [], [[] for _ in range(4)], [f"b{i}c{j}" for j in range(4)],
                            [True] * 4, payloads)
        out.append({"input": inp.to_json(), "target": {"indices": target}, "decision_type": "verifier",
                    "sample_id": f"b{i}", "source": source, "meta": meta})
    T.apply_balance(out, T.balance_weights(out, [], len(out), 0, 4))
    return out


def microbatch_backward(policy, rs, indices, dev, teach=None):
    for ids in T.microbatches(policy, rs, indices):
        inputs, logits = T.forward_rows(policy, [rs[i] for i in ids], dev)
        losses = [T.decision_loss(policy, rs[i], dev, teacher=(teach or {}).get(i),
                                  logits=logits[j], inp=inputs[j])[0] * rs[i].get("weight", 1.0) / len(rs)
                  for j, i in enumerate(ids)]
        torch.stack(losses).sum().backward()


def balanced_step(policy, ranks):
    previous = T.VERIFIER_BALANCE
    T.VERIFIER_BALANCE = True
    try:
        rs, dev = balanced_rows(), torch.device("cpu")
        metrics = T.evaluate(policy, rs, dev, ranks)
        policy.zero_grad(set_to_none=True)
        microbatch_backward(policy, rs, ranks.share([T.cost(r) for r in rs]), dev)
        ranks.reduce_grads([p for p in policy.parameters() if p.requires_grad])
        # Resume an already observed best score, then take two non-improving dev evaluations on every rank.
        history = ranks.broadcast([0.8] if ranks.main else None)
        stopper = T.EarlyStop(2)
        for score in history:
            stopper.update(score)
        stops = [ranks.any(stopper.update(score)) for score in (0.7, 0.6)]
        return {"metrics": metrics, "grad": policy.w.grad.clone(), "stops": stops}
    finally:
        T.VERIFIER_BALANCE = previous


class SimRanks(T.Ranks if T else object):
    """Rank `rank` of `world` in one process: gather returns the parts of the ranks evaluated so far."""

    def __init__(self, rank, world, parts):
        self.rank, self.world, self.parts = rank, world, parts

    def gather(self, obj):
        self.parts[self.rank] = obj
        return [self.parts[k] for k in sorted(self.parts)]


@unittest.skipIf(T is None, "needs torch and the repo's imports")
class SftRanks(unittest.TestCase):
    def setUp(self):
        self.policy, self.rows, self.dev = LinearPolicy(), rows(), torch.device("cpu")

    def test_share_covers_every_item_once_with_balanced_cost(self):
        costs = [T.cost(r) for r in self.rows]
        shares = [SimRanks(k, 8, {}).share(costs) for k in range(8)]
        self.assertEqual(sorted(i for s in shares for i in s), list(range(len(costs))))
        loads = [sum(costs[i] for i in s) for s in shares]
        self.assertLessEqual(max(loads) - min(loads), max(costs))
        self.assertEqual(T.Ranks(env=False).share(costs), list(range(len(costs))))

    def test_share_cost_counts_prompt_tokens(self):
        row = self.rows[0]
        self.assertEqual(T.share_cost(self.policy, row), T.cost(row))         # no path lengths: candidate count
        tokens = {r["input"]["candidate_ids"][0]: 1000 + 700 * (i % 4) for i, r in enumerate(self.rows)}
        self.policy.path_lengths = lambda d: [tokens[d.candidate_ids[0]]]
        costs = [T.share_cost(self.policy, r) for r in self.rows]
        self.assertEqual(costs, [1000 + 700 * (i % 4) for i in range(len(self.rows))])
        shares = [SimRanks(k, 8, {}).share(costs) for k in range(8)]
        loads = [sum(costs[i] for i in s) for s in shares]
        self.assertEqual(sorted(i for s in shares for i in s), list(range(len(costs))))
        self.assertLessEqual(max(loads) - min(loads), max(costs))              # similar tokens on every rank
        self.policy.path_lengths = lambda d: [1]
        self.assertEqual(T.share_cost(self.policy, row), 1000)                 # cached for this policy
        self.assertEqual(T.share_cost(LinearPolicy(), row), T.cost(row))       # not for another one

    def test_share_cost_of_grouped_decisions_follows_prefix_sharing(self):
        import types
        grouped = types.SimpleNamespace(readout="letters", path_lengths=lambda d: [3000, 3010, 3020],
                                        shared_tokens=lambda d: 3000 + 3 * 30, prefix_sharing_train=True)
        self.assertEqual(T.share_cost(grouped, dict(self.rows[0])), 3090)       # one prefix, three question suffixes
        grouped.prefix_sharing_train = False
        self.assertEqual(T.share_cost(grouped, dict(self.rows[0])), 9030)       # three full prompts
        single = types.SimpleNamespace(readout="letters", path_lengths=lambda d: [2500], prefix_sharing_train=True)
        self.assertEqual(T.share_cost(single, dict(self.rows[0])), 2500)

    def test_sharded_dev_eval_equals_one_process_and_keeps_the_rng_stream(self):
        T.train_rng = random.Random(7)
        one = T.evaluate(self.policy, self.rows, self.dev, T.Ranks(env=False))
        state_one = T.train_rng.getstate()
        parts = {}
        for k in range(3):                                        # three ranks, each with its own copy of the RNG
            T.train_rng = random.Random(7)
            if k < 2:                                             # in one process the earlier ranks lack later parts
                with self.assertRaises(KeyError):
                    T.evaluate(self.policy, self.rows, self.dev, SimRanks(k, 3, parts))
            else:
                out = T.evaluate(self.policy, self.rows, self.dev, SimRanks(k, 3, parts))
            self.assertEqual(T.train_rng.getstate(), state_one)
        self.assertEqual(out, one)                                # the last rank saw every part
        self.assertIn("verifier_q3/loss", one)
        self.assertIn("frame/w_argmax_in_target", one)

    def test_drawing_teacher_orders_first_reproduces_the_one_process_losses(self):
        batch = [r for r in self.rows if r["decision_type"] == "frame"] + self.rows[:5]
        T.train_rng = random.Random(11)                            # original: decision_loss draws row by row
        ref = [float(T.decision_loss(self.policy, r, self.dev)[0]) for r in batch]
        ref_state = T.train_rng.getstate()
        rng = random.Random(11)
        teach = T.teacher_picks(batch, rng)
        new = [float(T.decision_loss(self.policy, r, self.dev, teacher=teach.get(i))[0]) for i, r in enumerate(batch)]
        self.assertEqual(new, ref)
        self.assertEqual(rng.getstate(), ref_state)

    def test_rank_gradients_add_up_to_the_full_step(self):
        batch = self.rows[:16]
        teach = T.teacher_picks(batch, random.Random(5))

        def grads(idx):
            self.policy.zero_grad(set_to_none=True)
            for i in idx:
                loss, _, _ = T.decision_loss(self.policy, batch[i], self.dev, teacher=teach.get(i))
                (loss * batch[i]["weight"] / len(batch)).backward()
            return self.policy.w.grad.clone() if self.policy.w.grad is not None else torch.zeros(8)
        full = grads(range(len(batch)))
        costs = [T.cost(r) for r in batch]
        summed = sum(grads(SimRanks(k, 8, {}).share(costs)) for k in range(8))
        torch.testing.assert_close(summed, full)

    def test_microbatch_preserves_weighted_soft_set_and_grouped_losses(self):
        batch = self.rows[:8] + balanced_rows()
        # Different candidate counts, an illegal candidate, and controller soft labels in the same microbatch.
        batch[0]["target"]["soft"] = [.75, 0., .25, 0., 0., 0.]
        teach = T.teacher_picks(batch, random.Random(31))
        ref_loss = 0.
        for i, row in enumerate(batch):
            loss = T.decision_loss(self.policy, row, self.dev, teacher=teach.get(i))[0]
            loss = loss * row.get("weight", 1.0) / len(batch)
            ref_loss += float(loss.detach())
            loss.backward()
        ref_grad = self.policy.w.grad.clone()
        self.policy.zero_grad(set_to_none=True)
        self.policy.calls = 0
        self.policy.sft_microbatch_decisions = 2
        inputs, logits = T.forward_rows(self.policy, batch, self.dev)
        batched_loss = sum(T.decision_loss(self.policy, row, self.dev, teacher=teach.get(i),
                                          logits=logits[i], inp=inputs[i])[0] * row.get("weight", 1.0) / len(batch)
                           for i, row in enumerate(batch))
        self.assertAlmostEqual(float(batched_loss.detach()), ref_loss, places=5)
        self.policy.calls = 0
        microbatch_backward(self.policy, batch, range(len(batch)), self.dev, teach)
        self.assertEqual(self.policy.calls, (len(batch) + 1) // 2)
        torch.testing.assert_close(self.policy.w.grad, ref_grad)

    def test_microbatch_token_budget_and_oversized_row(self):
        sizes = {r["input"]["candidate_ids"][0]: n for r, n in zip(self.rows, [7, 10, 30, 8, 7])}
        self.policy.path_lengths = lambda d: [sizes[d.candidate_ids[0]]]
        self.policy.sft_microbatch_decisions = 4
        self.policy.sft_microbatch_tokens = 20
        self.assertEqual(list(T.microbatches(self.policy, self.rows, range(5))), [[0, 1], [2], [3, 4]])
        self.policy.sft_microbatch_decisions = 1
        self.assertEqual(list(T.microbatches(self.policy, self.rows, range(5))), [[i] for i in range(5)])


def _rank_main(rank, world, port, out):
    """One torchrun-like rank on CPU (gloo): dev eval, one sharded step with an unused parameter, reductions."""
    import os
    os.environ.update(WORLD_SIZE=str(world), RANK=str(rank), LOCAL_RANK=str(rank), MASTER_ADDR="127.0.0.1",
                      MASTER_PORT=str(port))
    ranks = T.Ranks()
    policy, rs, dev = LinearPolicy(), rows(), torch.device("cpu")
    policy.sft_microbatch_decisions = 2
    policy.unused = torch.nn.Parameter(torch.zeros(3))
    T.train_rng = random.Random(7)
    metrics = T.evaluate(policy, rs, dev, ranks)
    batch = rs[:16]
    teach = T.teacher_picks(batch, T.train_rng)
    microbatch_backward(policy, batch, ranks.share([T.cost(r) for r in batch]), dev, teach)
    params = [p for p in policy.parameters() if p.requires_grad]
    ranks.reduce_grads(params)
    result = {"metrics": metrics, "grad": policy.w.grad.clone(), "unused_grad": policy.unused.grad,
              "total": ranks.total(1.0, float(rank)), "any": ranks.any(rank == 1), "rng": T.train_rng.getstate()}
    result["balanced"] = balanced_step(policy, ranks)
    torch.save(result, f"{out}/rank{rank}.pt")
    ranks.barrier()


@unittest.skipIf(T is None, "needs torch and the repo's imports")
class SftRanksDistributed(unittest.TestCase):
    def test_two_gloo_ranks_match_one_process(self):
        import socket
        import tempfile
        import torch.distributed as dist
        import torch.multiprocessing as mp
        if not dist.is_available():
            self.skipTest("torch.distributed unavailable")
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        with tempfile.TemporaryDirectory() as out:
            mp.spawn(_rank_main, args=(2, port, out), nprocs=2, join=True)
            got = [torch.load(f"{out}/rank{k}.pt") for k in range(2)]
        policy, rs, dev = LinearPolicy(), rows(), torch.device("cpu")
        T.train_rng = random.Random(7)
        one = T.evaluate(policy, rs, dev, T.Ranks(env=False))
        batch = rs[:16]
        teach = T.teacher_picks(batch, T.train_rng)
        for i, r in enumerate(batch):
            loss, _, _ = T.decision_loss(policy, r, dev, teacher=teach.get(i))
            (loss * r["weight"] / len(batch)).backward()
        for g in got:
            self.assertEqual(g["metrics"], one)
            torch.testing.assert_close(g["grad"], policy.w.grad)
            self.assertIsNone(g["unused_grad"])                      # no rank had a gradient: stays None
            self.assertEqual(g["total"], (2.0, 1.0))
            self.assertTrue(g["any"])
            self.assertEqual(g["rng"], T.train_rng.getstate())
        balanced = balanced_step(policy, T.Ranks(env=False))
        self.assertIn("verifier_bal/outside_auc_post", balanced["metrics"])
        self.assertIn("verifier_bal/score", balanced["metrics"])
        for g in got:
            self.assertEqual(g["balanced"]["metrics"], balanced["metrics"])
            torch.testing.assert_close(g["balanced"]["grad"], balanced["grad"])
            self.assertEqual(g["balanced"]["stops"], [False, True])


if __name__ == "__main__":
    unittest.main()
