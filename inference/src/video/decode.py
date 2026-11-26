"""Video decoding with real presentation timestamps (PyAV).

All times are seconds relative to the first frame's presentation time. The frame shown at time
t is the last frame whose pts <= t. The timeline is read from packet pts (no decoding), so
variable-frame-rate videos keep their true timestamps; nothing assumes a uniform FPS.
One request is decoded by up to JEV_DECODE_WORKERS containers at once (default CPU cores / 8, at least 4; 1 = a single
sequential pass): the targets are split where that pass re-seeks anyway or inside long runs of close targets, and every
group runs the same seek-and-decode procedure (2026-10-11: rollouts and the VTS filter decoded serially while the GPU
waited).
"""
import bisect
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from fractions import Fraction

import av
import numpy as np

_CORES = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 32)
# FFmpeg frame threads shared by the containers of one request: half the cores, at least 16 (48 on a 96-core node).
# 2026-10-14, 128 frames from a 68-min 1920x1440 video on a node running a collection: 4 containers / 16 threads 28.8 s,
# 12 / 16 18.4 s, 12 / 48 9.5 s; with 8 such requests at once 30 s vs 18 s on average.
CODEC_THREADS = max(16, _CORES // 2)
_POOL, _POOL_LOCK = None, threading.Lock()


def decode_workers():
    """Containers per decode request: JEV_DECODE_WORKERS, by default one per 8 CPU cores and at least 4 (the 8 GPU
    processes of a node share its cores; 2026-10-14: a 128-frame collection spent 44% of its time decoding with 4
    containers while 89% of the node's 96 cores idled); 1 = one sequential pass, the earlier behaviour."""
    try:
        return max(1, int(os.environ.get("JEV_DECODE_WORKERS") or max(4, _CORES // 8)))
    except ValueError:
        return 4


def _pool():
    global _POOL
    with _POOL_LOCK:
        if _POOL is None:
            _POOL = ThreadPoolExecutor(max_workers=decode_workers(), thread_name_prefix="jev-decode")
        return _POOL


def _reset_pool():
    global _POOL, _POOL_LOCK
    _POOL, _POOL_LOCK = None, threading.Lock()     # threads do not survive fork: a forked child starts its own


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_pool)


@dataclass
class Timeline:
    pts: list            # sorted stream pts (ints, stream time_base units), one per frame
    time_base: Fraction
    pts0: int            # pts of the first frame; t = (pts - pts0) * time_base
    duration: float      # seconds from first frame start to end of last frame
    avg_fps: float
    width: int
    height: int
    source: str = "packets"
    times: list = field(default_factory=list)
    keys: list = field(default_factory=list)   # timeline indices of keyframes (2026-11-03; [] = not read yet)

    def __post_init__(self):
        self.times = [float((p - self.pts0) * self.time_base) for p in self.pts]

    def frame_at(self, t: float) -> int:
        """Index of the frame displayed at time t (clamped to the video)."""
        i = bisect.bisect_right(self.times, t) - 1
        return min(max(i, 0), len(self.times) - 1)

    def frames_in(self, start: float, end: float):
        """Indices of frames whose pts lies in [start, end)."""
        lo = bisect.bisect_left(self.times, start)
        hi = bisect.bisect_left(self.times, end)
        return list(range(lo, hi))


def read_timeline(path, mode="packets") -> Timeline:
    """mode='packets' reads pts from the demuxer (fast); mode='decode' decodes every frame
    (slow, but is exactly what the decoder produces). Callers verify packets-mode timelines by
    decoding the frames they need and fall back to decode-mode on any mismatch."""
    with av.open(str(path)) as container:
        vs = container.streams.video[0]
        tb = vs.time_base
        pts, key_pts = [], set()
        if mode == "packets":
            for packet in container.demux(vs):
                if packet.pts is not None and packet.size > 0:
                    pts.append(packet.pts)
                    if packet.is_keyframe:            # the container's flag: read with the pts, nothing decoded
                        key_pts.add(packet.pts)
        if mode == "decode" or len(pts) < 2:
            container.seek(0)
            vs.thread_type = "AUTO"
            frames = [(f.pts, f.key_frame) for f in container.decode(vs) if f.pts is not None]
            pts, key_pts = [p for p, _ in frames], {p for p, k in frames if k}
            mode = "decode"
        if not pts:
            raise ValueError(f"no decodable frames: {path}")
        pts = sorted(set(pts))
        width, height = vs.codec_context.width, vs.codec_context.height
    diffs = np.diff(pts)
    step = float(np.median(diffs)) if len(diffs) else 0.0
    duration = float((pts[-1] - pts[0] + step) * tb)
    avg_fps = (len(pts) / duration) if duration > 0 else 0.0
    return Timeline(pts=pts, time_base=tb, pts0=pts[0], duration=duration, avg_fps=avg_fps,
                    width=width, height=height, source=mode, keys=[i for i, p in enumerate(pts) if p in key_pts])


def read_keyframes(path, tl: Timeline):
    """Timeline indices of the keyframes, from the packet flags (demux only; ~0.3 s for a 34-min video). For timelines
    cached before 2026-11-03, which carry no keyframes."""
    index = {p: i for i, p in enumerate(tl.pts)}
    with av.open(str(path)) as container:
        vs = container.streams.video[0]
        keys = {index[pk.pts] for pk in container.demux(vs) if pk.pts in index and pk.is_keyframe}
    return sorted(keys)


def timeline_to_json(tl: Timeline) -> dict:
    return {"pts": tl.pts, "tb": [tl.time_base.numerator, tl.time_base.denominator],
            "pts0": tl.pts0, "duration": tl.duration, "avg_fps": tl.avg_fps,
            "width": tl.width, "height": tl.height, "source": tl.source, "keys": list(tl.keys)}


def timeline_from_json(d: dict) -> Timeline:
    return Timeline(pts=d["pts"], time_base=Fraction(*d["tb"]), pts0=d["pts0"],
                    duration=d["duration"], avg_fps=d["avg_fps"], width=d["width"],
                    height=d["height"], source=d["source"], keys=list(d.get("keys") or []))


def uniform_times(start: float, end: float, n: int):
    """Midpoints of n equal bins in [start, end): deterministic, never hits `end` exactly."""
    if n <= 0 or end <= start:
        return []
    step = (end - start) / n
    return [start + (i + 0.5) * step for i in range(n)]


def _scaled_size(w, h, max_side):
    if not max_side or max(w, h) <= max_side:
        return w, h
    s = max_side / max(w, h)
    return max(2, int(round(w * s / 2)) * 2), max(2, int(round(h * s / 2)) * 2)


class VideoDecoder:
    """Decodes frames by timeline index; seeks only when the next target is far ahead. A request with several runs of
    close targets, or one long run, is split over up to decode_workers() containers (see _groups)."""

    split_sec = 30.0          # a run of close targets longer than 2 x this is cut into pieces of at least this span

    def __init__(self, path, timeline: Timeline, max_side=None, seek_gap_sec=2.0):
        self.path = str(path)
        self.tl = timeline
        self.max_side = max_side
        self.seek_gap = seek_gap_sec
        self.frames_decoded = 0   # every frame produced by the decoder, including skipped ones
        self._count_lock = threading.Lock()

    def get(self, indices):
        """Return {index: RGB uint8 array} for the requested timeline indices."""
        want = sorted(set(int(i) for i in indices))
        if not want:
            return {}
        tl = self.tl
        w, h = _scaled_size(tl.width, tl.height, self.max_side)
        groups = self._groups(want, decode_workers())
        if len(groups) == 1:
            out = self._decode(want, w, h)
        else:
            threads = max(1, CODEC_THREADS // len(groups))
            out = {}
            for f in [_pool().submit(self._decode, g, w, h, threads) for g in groups]:
                out.update(f.result())
        missing = [i for i in want if i not in out]
        if missing:
            out.update(self._fallback(missing, w, h))
        return out

    def _groups(self, want, k):
        """Up to k contiguous groups of sorted targets with similar decode spans. Cuts fall where one pass re-seeks
        anyway (gap > seek_gap) or inside a run of close targets longer than 2 x split_sec (pieces >= split_sec), so
        each group is decoded exactly as a single pass would decode it, from a fresh seek."""
        if k <= 1 or len(want) <= 1:
            return [want]
        t = [self.tl.times[i] for i in want]
        pieces, start = [], 0
        for j in range(1, len(want) + 1):
            if j < len(want) and t[j] - t[j - 1] <= self.seek_gap:
                continue
            span = t[j - 1] - t[start]
            n = max(1, min(k, int(span // self.split_sec)))
            bounds = [start] + [bisect.bisect_left(t, t[start] + span * m / n, start, j) for m in range(1, n)] + [j]
            pieces += [(a, b) for a, b in zip(bounds, bounds[1:]) if b > a]
            start = j
        if len(pieces) == 1:
            return [want]
        cost = [t[b - 1] - t[a] + self.seek_gap for a, b in pieces]       # decode span + one seek
        total, done, groups = sum(cost), 0.0, {}
        for piece, c in zip(pieces, cost):                                   # by the piece's cost midpoint
            groups.setdefault(min(k - 1, int((done + c / 2) * k / total)), []).append(piece)
            done += c
        return [want[g[0][0]:g[-1][1]] for _, g in sorted(groups.items())]

    def _decode(self, want, w, h, threads=0):
        """One container, one pass over sorted targets; threads > 0 caps FFmpeg's frame threads when several
        containers share a request (the decoded frames do not depend on it)."""
        out, tl, decoded = {}, self.tl, 0
        try:
            with av.open(self.path) as container:
                vs = container.streams.video[0]
                vs.thread_type = "AUTO"
                if threads:
                    vs.thread_count = threads
                k = 0
                while k < len(want):
                    target = tl.pts[want[k]]
                    container.seek(target, stream=vs, backward=True, any_frame=False)
                    for frame in container.decode(vs):
                        decoded += 1
                        if frame.pts is None:
                            continue
                        while k < len(want) and tl.pts[want[k]] < frame.pts:
                            k += 1  # target pts missing from decoded stream: give up on it below
                        if k >= len(want):
                            break
                        if frame.pts == tl.pts[want[k]]:
                            out[want[k]] = frame.to_ndarray(format="rgb24", width=w, height=h)
                            k += 1
                            if k >= len(want):
                                break
                            gap = float((tl.pts[want[k]] - frame.pts) * tl.time_base)
                            if gap > self.seek_gap:
                                break  # re-seek for the far target
                    else:
                        break  # end of stream
        finally:
            with self._count_lock:
                self.frames_decoded += decoded
        return out

    def _fallback(self, indices, w, h):
        """Linear decode from the start; used only if seeking failed to land on exact pts."""
        want = {self.tl.pts[i]: i for i in indices}
        out = {}
        with av.open(self.path) as container:
            vs = container.streams.video[0]
            for frame in container.decode(vs):
                self.frames_decoded += 1
                if frame.pts in want:
                    out[want[frame.pts]] = frame.to_ndarray(format="rgb24", width=w, height=h)
                    if len(out) == len(want):
                        break
        missing = [i for i in indices if i not in out]
        if missing:
            raise RuntimeError(f"could not decode frames {missing[:5]} of {self.path}")
        return out
