"""Labels from the evidence annotation (2026-10-06). One definition of "evidence" for every role: the manifest key_spans
widened by `pad` seconds (budget.evidence_pad, default 3) and clipped to the video. CPU only, no model.

  merge(spans) / overlap(...)      evidence = the UNION of the padded spans (overlapping padded spans are merged;
                                   2026-10-06 review: summing them double-counted, e.g. cgbench:8413 34% vs 23%)
  window_cover(spans, a0, a1)      share of [a0, a1] covered by the evidence (the evidence expert's `cover`)
  uncovered_spans(spans, kept, cap[, per_place]) evidence segments still needing kept frames, for rows marked evidence_multi_segment
                                   only (filter --multi-segment, the VideoITG experiment of 2026-10-20; first written
                                   2026-10-14 for every row and reverted there): each segment's share of the kept-frame
                                   cap follows its length; a segment with its share of kept frames is covered and drops
                                   out (the evidence expert and the where / enclosing labels use the rest; every segment
                                   again once all are covered)
  unvisited_overlap(spans, w, excl) evidence seconds in window w outside the windows `excl` (active + visited ones)
  enclosing_label(spans, active, ancestors, visited)  window tree: 1 + index of the nearest ancestor that has evidence
                                   outside the active window and outside every visited non-ancestor window, else 0
  where_label(spans, a0, a1, frac) evidence vs the active window: before / inside_small (cover < frac) /
                                   inside_large / after (verifier "where" question; before/after = nearest span)
  localize_targets(windows, spans) windows whose evidence density (evidence seconds / window length) is >= half of
                                   the best legal window's; [] when no legal window touches the evidence
  frame_targets(inp, spans)        legal candidate frames inside the evidence

EvidenceGrounding(role, annotations, pad, inner=None) is a localize / frame decider:
  inner=None  executes the label (localize: the densest window, ties -> shorter; no target -> the legal window
              nearest to the evidence. frame: picks sampled from the targets as in SFT, src/policy/selection)
  inner=<decider>  executes the inner decider's choice (random, a trained Jev, ...) and only attaches the labels.
Either way info.target_indices holds the label set, read by scripts/gen_role_sft_from_trajectories.py (empty = no label).
"""
import random

WHERE = ("before", "inside_small", "inside_large", "after")


def padded_spans(key_spans, pad, T):
    return [(max(0.0, float(s) - pad), min(float(T), float(e) + pad)) for s, e in key_spans]


def merge(spans):
    out = []
    for s, e in sorted((float(s), float(e)) for s, e in spans if e > s):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [tuple(x) for x in out]


def merge_gap(spans, gap):
    """Spans merged when they overlap or are at most `gap` seconds apart."""
    out = []
    for s, e in sorted((float(s), float(e)) for s, e in spans if e > s):
        if out and s <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [tuple(x) for x in out]


def segment_gap(T, max_gap=30.0, frac=0.125):
    """Evidence pieces at most min(30 s, T/8) apart are one segment
    (scripts/filter_cgbench_vts.py --merge-gap auto writes the merged spans)."""
    return min(float(max_gap), float(frac) * float(T))


def overlap(spans, a0, a1):
    return sum(max(0.0, min(a1, e) - max(a0, s)) for s, e in merge(spans))


def unvisited_overlap(spans, w, excl):
    """Seconds of evidence inside window w = (a0, a1) that lie outside every window in excl."""
    a0, a1 = w
    parts = [(max(a0, s), min(a1, e)) for s, e in merge(spans) if min(a1, e) > max(a0, s)]
    for x0, x1 in merge(excl):
        nxt = []
        for s, e in parts:
            if x1 <= s or x0 >= e:
                nxt.append((s, e))
                continue
            if s < x0:
                nxt.append((s, x0))
            if x1 < e:
                nxt.append((x1, e))
        parts = nxt
    return sum(e - s for s, e in parts)


def enclosing_label(spans, active, ancestors, visited):
    """active, ancestors (nearest first, ending with the whole video), visited: (start, end) pairs. Returns k + 1 for
    the k-th ancestor that holds evidence outside the active window and outside the visited non-ancestor windows (so
    k = 0 <=> an unvisited sibling has evidence -> SWITCH; k >= 1 -> UP), or 0 when no unvisited place has any."""
    anc = {(round(a, 3), round(b, 3)) for a, b in ancestors}
    excl = [active] + [w for w in visited if (round(w[0], 3), round(w[1], 3)) not in anc]
    for k, w in enumerate(ancestors):
        if unvisited_overlap(spans, w, excl) > 1e-6:
            return k + 1
    return 0


def window_cover(spans, a0, a1):
    return overlap(spans, a0, a1) / max(a1 - a0, 1e-6)


def uncovered_spans(spans, kept_times, cap, per_place=None):
    """Merged evidence segments that still need kept frames. Segment i's share of the kept-frame cap is
    cap * len_i / total length, rounded, at least 1; a segment holding that many kept frames (kept_times inside it) is
    covered. Returns the segments not covered yet, or every segment once all are covered (or without a cap).
    per_place (budget.share_cap_per_place, option (a)): a segment is covered at
    min(share, per_place) kept frames (one inspection of 4 frames), so every multi-segment row shows the move to the next
    segment after one inspection there; None = the share alone. A single segment is never left this way (every segment
    comes back once all are covered)."""
    segs = merge(spans)
    total = sum(e - s for s, e in segs)
    if not segs or total <= 0 or not cap:
        return segs

    def need(s, e):
        n = max(1, round(cap * (e - s) / total))
        return min(n, int(per_place)) if per_place else n
    left = [(s, e) for s, e in segs if sum(s <= t <= e for t in kept_times) < need(s, e)]
    return left or segs


def where_label(spans, a0, a1, frac=0.25):
    cov = window_cover(spans, a0, a1)
    if cov > 0.0:
        return "inside_small" if cov < frac else "inside_large"
    spans = merge(spans)
    before = [a0 - e for s, e in spans if e <= a0]
    after = [s - a1 for s, e in spans if s >= a1]
    if not after or (before and min(before) <= min(after)):
        return "before"
    return "after"


def localize_targets(windows, valid, spans, tol=0.5):
    """windows: list of {start, end, ...} (DecisionInput payloads); valid: per-window legality."""
    dens = [overlap(spans, w["start"], w["end"]) / max(w["end"] - w["start"], 1e-6) if v else 0.0
            for w, v in zip(windows, valid)]
    best = max(dens, default=0.0)
    if best <= 0.0:
        return []
    return [i for i, d in enumerate(dens) if d >= tol * best and d > 0.0]


def frame_targets(inp, spans):
    return [i for i in inp.legal_indices() if any(s <= inp.payloads[i]["t"] <= e for s, e in spans)]


def _onehot(n, idx, info, forced):
    p = [0.0] * n
    p[idx] = 1.0
    return {"index": idx, "log_prob": 0.0, "probs": p, "entropy": 0.0, "forced": forced, "seconds": 0.0, "info": info}


class EvidenceGrounding:
    def __init__(self, role, annotations, pad=3.0, inner=None, seed=0):
        assert role in ("localize", "frame")
        self.role, self.ann, self.pad, self.inner = role, annotations, float(pad), inner
        self.name = f"evidence_{role}" + (f"+{getattr(inner, 'name', 'inner')}" if inner is not None else "")
        self.rng = random.Random(seed)

    def _spans(self, ep):
        return padded_spans(self.ann[ep.sample.sample_id]["key_spans"], self.pad, ep.T)

    def _targets(self, inp, spans):
        if self.role == "localize":
            return localize_targets(inp.payloads, inp.valid, spans)
        return frame_targets(inp, spans)

    def _own(self, inp, spans, targets):
        legal = inp.legal_indices()
        if self.role == "localize":
            if targets:
                idx = max(targets, key=lambda i: (overlap(spans, inp.payloads[i]["start"], inp.payloads[i]["end"])
                                                  / max(inp.payloads[i]["end"] - inp.payloads[i]["start"], 1e-6),
                                                  -(inp.payloads[i]["end"] - inp.payloads[i]["start"])))
            else:          # nothing legal touches the evidence: the legal window closest to it
                mid = lambda i: (inp.payloads[i]["start"] + inp.payloads[i]["end"]) / 2
                idx = min(legal, key=lambda i: min((abs(mid(i) - (s + e) / 2) for s, e in spans), default=0.0))
            return _onehot(len(inp.candidates), idx, {}, len(legal) == 1)
        from ..policy.selection import teacher_sequence
        picks, sup = teacher_sequence(inp.valid, targets, inp.select_k, inp.mask_radius, self.rng)
        if not picks:
            picks = legal[:inp.select_k]
        return _onehot(len(inp.candidates), picks[0], {"picks": picks, "supervised": sup}, len(legal) == 1)

    def decide(self, inputs, episodes, generators):
        out = []
        inner = self.inner.decide(inputs, episodes, generators) if self.inner is not None else [None] * len(inputs)
        for inp, ep, res in zip(inputs, episodes, inner):
            if not inp.legal_indices():
                out.append(res if res is not None else {"index": None, "log_prob": 0.0, "probs": [0.0] * len(inp.candidates),
                                                        "entropy": 0.0, "forced": True})
                continue
            spans = self._spans(ep)
            targets = self._targets(inp, spans)
            if res is None:
                res = self._own(inp, spans, targets)
            res = dict(res)
            res["info"] = dict(res.get("info") or {}, target_indices=targets, label="evidence",
                               coverage_miss=not targets)
            out.append(res)
        return out
