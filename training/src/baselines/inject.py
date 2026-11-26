"""Wrong-window injection for collection runs.

Why: the evidence expert never inspects a window without evidence, so the verifier never saw an inspection there (0
where rows labelled before / after after an inspection, ~0 per-frame negatives from such windows) and says "inside" in
every window; the trained controller then never leaves a wrong window. These wrappers create such states on purpose,
like VTS's "natural" wrong branches (the scorer's top child when it misses the ground truth):

  InjectWrongLocalizer   at every locator call of the window tree (the root's REFINE_LOCAL, every SWITCH_LOCAL, every
                         REFINE_LOCAL into a child), with probability p (deterministic per sample id and call number) and
                         at most max_inject times per episode, executes the locator's most probable legal window whose
                         inspection range (window +- inspect_margin, clipped; EpisodeV3.range_of) has no evidence
                         (key_spans +- pad), if that window's probability >= min_prob (else the locator's own choice).
                         Every position is covered on purpose: with injections only at the first window,
                         the verifier saw "outside" labels only there and could learn the position (its memory lists the
                         searched windows) instead of the frames.
  InjectInspectController in an injected window, the first controller decision while the window has no inspection yet
                         executes INSPECT_FRAMES (if legal) instead of the inner controller's action; the inner (expert)
                         label stays in info.target_indices. Later decisions are the inner controller's.

Both executed-but-not-chosen decisions are marked forced, so scripts/gen_role_sft_from_trajectories.py makes no SFT row
from them (the locator is never taught a wrong window); the verifier rows of the states they lead to (where before /
after the inspection, per-frame keep of the frames picked there) and the expert's next controller decision are kept.
A locator call whose own pick is already the most probable wrong window counts as a (natural) injection but stays an
ordinary decision. The episode records ep.inject_meta (written to the trajectory as "injection").
"""
import math
import random

from .evidence import overlap, padded_spans


def _key(w):
    return (round(float(w[0]), 6), round(float(w[1]), 6))


class InjectWrongLocalizer:
    def __init__(self, inner, annotations, pad, p=1.0, min_prob=0.05, max_inject=2, seed=0):
        self.inner, self.ann = inner, annotations
        self.pad, self.p, self.min_prob, self.max_inject = float(pad), float(p), float(min_prob), int(max_inject)
        self.seed = int(seed)
        self.name = f"{getattr(inner, 'name', 'inner')}+inject_wrong"
        self.policy = getattr(inner, "policy", None) or getattr(getattr(inner, "inner", None), "policy", None)

    def _drawn(self, sample_id, n):
        """Same draw for a question's n-th locator call on every run (resume, shards): seeded, not call-order based."""
        return random.Random(f"{self.seed}:{sample_id}:{n}").random() < self.p

    def decide(self, inputs, episodes, generators):
        from ..env.candidates import Window
        out = []
        for inp, ep, r in zip(inputs, episodes, self.inner.decide(inputs, episodes, generators)):
            if r.get("index") is None or not getattr(ep, "tree", False):
                out.append(r)
                continue
            sid = ep.sample.sample_id
            meta = getattr(ep, "inject_meta", None)
            if meta is None:
                meta = ep.inject_meta = {"injected": False, "injections": [], "calls": [], "p": self.p,
                                         "min_prob": self.min_prob, "max_inject": self.max_inject, "pad": self.pad}
            n = len(meta["calls"])
            pl0 = inp.payloads[r["index"]]
            if ep.active_is_full():
                kind, level = "root", 1
            elif abs((pl0["end"] - pl0["start"]) - ep.active.length) < 1e-6:
                kind, level = "sibling", len(ep.path)
            else:
                kind, level = "child", len(ep.path) + 1
            call = {"n": n, "kind": kind, "level": level, "orig_index": r["index"], "orig_prob": float(r["probs"][r["index"]]),
                    "drawn": sid in self.ann and self._drawn(sid, n)}
            meta["calls"].append(call)
            if not call["drawn"] or len(meta["injections"]) >= self.max_inject:
                out.append(r)
                continue
            spans = padded_spans(self.ann[sid]["key_spans"], self.pad, ep.T)
            wrong = []
            for i in inp.legal_indices():
                pl = inp.payloads[i]
                rr = ep.range_of(Window(pl["start"], pl["end"], pl.get("kind", "quarter")))
                if overlap(spans, rr.start, rr.end) <= 0.0:
                    wrong.append(i)
            call.update(n_wrong=len(wrong), orig_wrong=r["index"] in wrong)
            if not wrong:
                out.append(r)
                continue
            j = max(wrong, key=lambda i: float(r["probs"][i]))
            call["best_wrong_prob"] = float(r["probs"][j])
            if float(r["probs"][j]) < self.min_prob:
                out.append(r)
                continue
            pl = inp.payloads[j]
            inj = {"n": n, "kind": kind, "level": level, "index": j, "window": [pl["start"], pl["end"]],
                   "prob": float(r["probs"][j]), "natural": j == r["index"], "inspected": False}
            meta["injections"].append(inj)
            meta["injected"] = True
            if j != r["index"]:   # the locator's own wrong pick (natural error) stays an ordinary, labelled decision
                r = dict(r, index=j, forced=True, log_prob=math.log(max(float(r["probs"][j]), 1e-12)))
                r["info"] = dict(r.get("info") or {}, injected="wrong_window")
            out.append(r)
        return out


class InjectInspectController:
    def __init__(self, inner):
        self.inner = inner
        self.name = f"{getattr(inner, 'name', 'inner')}+inject_inspect"
        self.policy = getattr(inner, "policy", None)

    def decide(self, inputs, episodes, generators):
        out = []
        for inp, ep, r in zip(inputs, episodes, self.inner.decide(inputs, episodes, generators)):
            m = getattr(ep, "inject_meta", None)
            here = _key((ep.active.start, ep.active.end))
            inj = next((x for x in (m or {}).get("injections", []) if _key(x["window"]) == here and not x["inspected"]),
                       None)
            if (r.get("index") is None or inj is None or ep.inspections_here() > 0
                    or "INSPECT_FRAMES" not in inp.candidate_ids):
                out.append(r)
                continue
            j = inp.candidate_ids.index("INSPECT_FRAMES")
            if not inp.valid[j]:
                out.append(r)
                continue
            inj.update(inspected=True, label_action=inp.candidate_ids[r["index"]])
            info = dict(r.get("info") or {})
            info.update(target_indices=info.get("target_indices", [r["index"]]), executed="injected")
            out.append(dict(r, index=j, forced=True, log_prob=0.0, info=info))
        return out
