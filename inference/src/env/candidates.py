"""Deterministic localization candidates (update.md §4).

8 windows over a span [lo, hi]: 4 equal quarters, 3 halves with stride span/4, and the span
itself. Global localization uses the whole video; local refinement applies the same rule inside
the active interval. Candidates depend only on the timeline and the public state.
"""
from dataclasses import dataclass

TIME_DECIMALS = 3  # windows equal after rounding to 1 ms are duplicates


@dataclass(frozen=True)
class Window:
    start: float
    end: float
    kind: str = "quarter"   # quarter | half | full | active

    @property
    def length(self):
        return self.end - self.start

    def key(self):
        return (round(self.start, TIME_DECIMALS), round(self.end, TIME_DECIMALS))

    def to_json(self):
        return {"start": self.start, "end": self.end, "kind": self.kind}


def windows8(lo, hi, whole_kind):
    span = hi - lo
    q = [Window(lo + i * span / 4, lo + (i + 1) * span / 4, "quarter") for i in range(4)]
    h = [Window(lo + i * span / 4, lo + i * span / 4 + span / 2, "half") for i in range(3)]
    return q + h + [Window(lo, hi, whole_kind)]


class CandidateGenerator:
    def __init__(self, min_window_sec=1.0):
        self.min_len = min_window_sec

    def _finish(self, wins, whole_key):
        seen, out = set(), []
        for w in wins:
            if w.key() in seen or (w.length < self.min_len - 1e-9 and w.key() != whole_key):
                continue
            seen.add(w.key())
            out.append(w)
        return sorted(out, key=lambda w: (round(w.start, TIME_DECIMALS), round(w.end, TIME_DECIMALS)))

    def global_windows(self, duration):
        return self._finish(windows8(0.0, duration, "full"), (0.0, round(duration, TIME_DECIMALS)))

    def local_windows(self, active, duration):
        a, b = max(0.0, active.start), min(duration, active.end)
        return self._finish(windows8(a, b, "active"), (round(a, TIME_DECIMALS), round(b, TIME_DECIMALS)))


def best_iou(windows, spans):
    """Upper bound on localization IoU achievable with this candidate list (coverage diagnostic)."""
    from ..evaluation.localization import interval_iou
    return max((interval_iou((w.start, w.end), s) for w in windows for s in spans), default=0.0)
