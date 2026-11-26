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

Path injection (path=True): at test time the agent mostly goes wrong high (a wrong
level-1 quarter), descends inside it and inspects deep, so the recovery needs UP, UP, SWITCH; the plain injection
inspects the injected window itself (a level-1 quarter or one level off the evidence path), so training had the
"switch to a sibling" recovery only. With path=True an injection above the expert's inspection level
(expert_inspect_level) is followed by forced REFINE_LOCAL decisions inside the injected window (the locator's own picks,
no injection while the path is open) down to that level, then the forced INSPECT_FRAMES; the expert acts after it.
Forced keep (keep_prob, same day): the forced inspection keeps its picks with this probability whatever P(keep) is
(RunnerV3._tentative_frames reads inject_meta), so the controller sees kept frames in a wrong window with the expert's
"leave" label; the per-frame keep labels of those frames stay "no" (they come from key_spans).
"""
import math
import random

from .evidence import overlap, padded_spans, window_cover


def expert_inspect_level(ep, spans, refine_frac=0.25):
    """Window-tree level at which the evidence expert first inspects this question (EvidenceExpertController._tree_rule
    "narrow": the root refines; below it, REFINE_LOCAL while the inspection range's evidence cover < refine_frac and a
    child holds evidence, else INSPECT_FRAMES), entering the child with the most evidence. Children = the quarters the
    window tree offers (CandidateGenerator: none shorter than video.min_window_sec)."""
    w, level = ep.full, 0
    while True:
        kids = [u for u in ep.gen.local_windows(w, ep.T) if u.kind == "quarter"]
        ev = [u for u in kids if overlap(spans, u.start, u.end) > 0]
        if level > 0:
            r = ep.range_of(w)
            if window_cover(spans, r.start, r.end) >= refine_frac or not ev:
                return level
        if not ev:
            return max(level, 1)
        w, level = max(ev, key=lambda u: overlap(spans, u.start, u.end)), level + 1


def _inside(w, outer, eps=1e-6):
    return w[0] >= outer[0] - eps and w[1] <= outer[1] + eps


def open_path(ep):
    """The path injection whose window contains the active window and that was not inspected yet (None otherwise)."""
    m = getattr(ep, "inject_meta", None) or {}
    here = (ep.active.start, ep.active.end)
    return next((x for x in m.get("injections", []) if x.get("path_level") and not x["inspected"]
                 and _inside(here, x["window"])), None)


def _key(w):
    return (round(float(w[0]), 6), round(float(w[1]), 6))


class InjectWrongLocalizer:
    def __init__(self, inner, annotations, pad, p=1.0, min_prob=0.05, max_inject=2, seed=0, path=False, refine_frac=0.25):
        self.inner, self.ann = inner, annotations
        self.pad, self.p, self.min_prob, self.max_inject = float(pad), float(p), float(min_prob), int(max_inject)
        self.seed = int(seed)
        self.path, self.refine_frac = bool(path), float(refine_frac)
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
            if self.path and open_path(ep) is not None:    # the forced descent inside an injected window: its own picks
                call["in_path"] = True
                out.append(r)
                continue
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
            if self.path:       # inspect at the expert's level inside the injected window (InjectInspectController)
                if "expert_level" not in meta:
                    meta["expert_level"] = expert_inspect_level(ep, spans, self.refine_frac)
                if level < meta["expert_level"]:
                    inj["path_level"] = meta["expert_level"]
            meta["injections"].append(inj)
            meta["injected"] = True
            if j != r["index"]:   # the locator's own wrong pick (natural error) stays an ordinary, labelled decision
                r = dict(r, index=j, forced=True, log_prob=math.log(max(float(r["probs"][j]), 1e-12)))
                r["info"] = dict(r.get("info") or {}, injected="wrong_window")
            out.append(r)
        return out


class InjectInspectController:
    def __init__(self, inner, path=False, keep_prob=0.0, seed=0):
        self.inner = inner
        self.name = f"{getattr(inner, 'name', 'inner')}+inject_inspect"
        self.policy = getattr(inner, "policy", None)
        self.path, self.keep_prob, self.seed = bool(path), float(keep_prob), int(seed)

    def _force(self, r, inp, action, executed):
        info = dict(r.get("info") or {})
        info.update(target_indices=info.get("target_indices", [r["index"]]), executed=executed)
        return dict(r, index=inp.candidate_ids.index(action), forced=True, log_prob=0.0, info=info)

    def decide(self, inputs, episodes, generators):
        out = []
        for inp, ep, r in zip(inputs, episodes, self.inner.decide(inputs, episodes, generators)):
            m = getattr(ep, "inject_meta", None)
            here = _key((ep.active.start, ep.active.end))
            inj = open_path(ep) if self.path else None
            if inj is None:
                inj = next((x for x in (m or {}).get("injections", []) if _key(x["window"]) == here
                            and not x["inspected"] and not x.get("path_level")), None)
                if inj is not None and ep.inspections_here() > 0:
                    inj = None
            if r.get("index") is None or inj is None:
                out.append(r)
                continue
            ok = lambda a: a in inp.candidate_ids and inp.valid[inp.candidate_ids.index(a)]   # noqa: E731
            if inj.get("path_level") and len(ep.path) < inj["path_level"] and ok("REFINE_LOCAL"):
                inj.setdefault("descent", []).append(list(here))     # forced descent: no SFT row (forced)
                out.append(self._force(r, inp, "REFINE_LOCAL", "injected_descent"))
                continue
            if not ok("INSPECT_FRAMES"):
                out.append(r)
                continue
            fk = self.keep_prob > 0 and random.Random(f"{self.seed}:{ep.sample.sample_id}:{inj['n']}:keep").random() \
                < self.keep_prob
            inj.update(inspected=True, label_action=inp.candidate_ids[r["index"]],
                       inspect_window=[ep.active.start, ep.active.end], force_keep=bool(fk))
            out.append(self._force(r, inp, "INSPECT_FRAMES", "injected"))
        return out
