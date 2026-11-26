"""Restricted-action video QA environment for the feasibility round (update.md §3-§5).

One Episode = one independent question instance built from a PublicSample (no answer field).
Actions: LOCALIZE_GLOBAL, REFINE_LOCAL, INSPECT_FRAMES, ANSWER; at most `max_turns` controller
turns, then the environment calls the answerer (not a policy action).

Decision inputs follow NanoJev's format and the update's order:
  "State:\\n Task / Question / Options / Memory / Frames (time-ordered: 'ID [t] <4 visual tokens>')"
  "Question type: choice\\nQuestion:\\n<instruction>"      (NanoJev choice segment)
  "Candidate:\\n<id>: <description>\\nDecision:" + EOS     (one path per legal candidate)
Visual context per decision: controller and global localization see the global frames (G01..);
refinement sees frames of the active interval (L1..); frame selection sees its candidates (C1..).
Candidates refer to frames by ID; visual tokens live only in the shared prefix.
Evidence for the reader/answerer is only the frames actually sent by INSPECT_FRAMES.
"""
from dataclasses import dataclass, field
from typing import Optional

from ..common.types import DecisionInput, VisRef
from .candidates import CandidateGenerator, Window

CONTROLLER_TEXT = {
    "LOCALIZE_GLOBAL": "choose a time window anywhere in the video",
    "REFINE_LOCAL": "choose a smaller window inside the active interval",
    "SWITCH_LOCAL": "choose another window at the same level, inside the parent interval",
    "UP": "go back to the parent interval",
    "INSPECT_FRAMES": "pick {k} frames in the active interval and send them to the reader",
    "ANSWER": "stop and answer with the evidence collected so far",
}

# verifier (v2): tier k <=> p_E cut points (budget.verifier_cuts, default 0.25 / 0.5 / 0.8)
VERIFIER_TEXT = ("the answerer would most likely be wrong (below {0:.0%} for the correct option)",
                 "weak evidence ({0:.0%}-{1:.0%} for the correct option)",
                 "probably enough ({1:.0%}-{2:.0%} for the correct option)",
                 "enough evidence ({2:.0%} or more for the correct option)")
CAPTION_PREFIX = "Find the moment described: "


class PathTooLong(RuntimeError):
    pass


@dataclass(frozen=True)
class PublicSample:
    """Everything the agent may see about a question. No answer field by construction."""
    sample_id: str
    video_id: str
    video_path: str
    question: str
    options: dict          # letter -> text


@dataclass
class Observation:
    turn: int
    times: list
    text: str
    status: str = "ok"


@dataclass
class ReaderRequest:
    kind: str              # observe | answer
    episode_key: str
    frame_indices: list    # timeline indices, time-sorted
    frame_times: list
    notes: list = field(default_factory=list)   # (header, text) pairs for the answerer
    source: str = "frames"
    interval: Optional[list] = None


def _fmt_t(t):
    return f"{t:.1f}s"


def merge_text(segs):
    """Concatenate adjacent strings (text-only spans tokenize like NanoJev's inputs)."""
    out = []
    for s in segs:
        if isinstance(s, str) and out and isinstance(out[-1], str):
            out[-1] += s
        else:
            out.append(s)
    return out


def nanojev_candidate(text):
    """NanoJev leaf: 'Candidate:\\n{key}: {criteria}\\nDecision:' (+ EOS added by the model)."""
    return [f"Candidate:\n{text}\nDecision:"]


class Episode:
    # observation compression levels (chars per observation, keep the latest n observations)
    COMPRESS = ((None, None), (300, None), (150, None), (80, 2), (0, 1), (0, 0))

    def __init__(self, sample: PublicSample, asset, features, encoder, cfg, key: str):
        self.sample, self.asset, self.features, self.encoder, self.key = sample, asset, features, encoder, key
        self.b, self.v = cfg["budget"], cfg["video"]
        self.tpf = cfg["video"]["pool_grid"] ** 2
        self.gen = CandidateGenerator(self.b["min_window_sec"])
        self.T = asset.duration
        self.full = Window(0.0, self.T, "full")
        self.turn = 0
        self.active = self.full
        self.active_set_turn = -1      # turn at which the active window was last set (-1: initial)
        self.searched = []
        self.observations = []
        self.evidence = []            # indices sent to the reader (FIFO, dedup, <= retained cap)
        self.inspections = []         # (turn, [indices])
        self.encoded = set()          # indices whose features this episode paid for
        self.global_idx = []
        self.done = False
        self.termination_reason = None
        self.answer_raw = None
        self.answer = None
        self.cost = {"encoder_frames": 0, "reader_calls": 0, "reader_frames": 0}
        self.path_log = []            # (decision type, max path tokens, compression level)
        self.selection = self.b.get("selection", "sequential")     # sequential (v1) | c (v2)
        self.max_inspections = self.b.get("max_inspections")       # None: only the turn cap (v1)
        self.nav_moves = self.b.get("nav_moves")                   # v2: localize/refine moves per episode (3)
        self.use_prior = bool(cfg["policy"].get("siglip_prior"))

    # ------------------------------------------------------------------ features
    def _pay_features(self, indices, budget=True):
        """Encode `indices` (features cached) and return the ones available. budget=False (2026-11-03, the conservative
        version's separate pick-24 scoring grid, budget.pick_answer_grid): encoded outside video.encoder_frame_cap -- the
        96 scored frames came from a separate run before, so they must not crowd out the search's candidates; they are
        still counted in cost["encoder_frames"]."""
        cap = self.v["encoder_frame_cap"]
        avail = []
        if not budget:
            extra = getattr(self, "encoded_extra", None)
            if extra is None:
                extra = self.encoded_extra = set()
            avail = list(indices)
            extra.update(i for i in avail if i not in self.encoded)
        else:
            for i in indices:
                if i in self.encoded:
                    avail.append(i)
                elif len(self.encoded) < cap:
                    self.encoded.add(i)
                    avail.append(i)
        if self.encoder is not None:   # lazy mode (encoder=None) keeps identical pools; features later
            new = [i for i in avail if not self.features.has(self.asset.content_id, self.asset.pts(i))]
            if new:
                self.features.ensure(self.asset, new, self.encoder)
        self.cost["encoder_frames"] = len(self.encoded | getattr(self, "encoded_extra", set()))
        return avail

    # ------------------------------------------------------------------ v2 helpers
    def inspections_left(self):
        return None if self.max_inspections is None else self.max_inspections - len(self.inspections)

    def frame_budget_done(self):
        return self.max_inspections is not None and len(self.inspections) >= self.max_inspections

    def nav_left(self):
        return None if self.nav_moves is None else self.nav_moves - len(self.searched)

    def prior_texts(self):
        """SigLIP2 query texts: question + each option (max over them); captions: the sentence."""
        if not self.use_prior:
            return None
        q = self.sample.question.strip()
        if q.startswith(CAPTION_PREFIX):
            q = q[len(CAPTION_PREFIX):]
        return [f"{q} {v}" for v in self.sample.options.values()] if self.sample.options else [q]

    def reset(self):
        self.global_idx = self._pay_features(self.sparse_grid(0.0, self.T, self.overview_count()))
        return self

    def sparse_grid(self, start, end, n):
        """A uniform grid of n frames in [start, end); with video.keyframe_snap (2026-11-03) grids at least
        video.keyframe_snap_factor x the median keyframe gap apart take the nearest keyframes (VideoAsset.
        snapped_uniform_indices): the overview and the locator contexts. Inspection candidates / where frames stay exact."""
        if self.v.get("keyframe_snap"):
            return self.asset.snapped_uniform_indices(start, end, n, float(self.v.get("keyframe_snap_factor", 2.0)))
        return self.asset.uniform_indices(start, end, n)

    def overview_count(self):
        """video.global_frames, or (first-step overview density by time; opt-in, absent/null = off)
        with video.global_spacing_s set: one overview frame per that many seconds, a multiple of 4 (one share per
        quarter), at least video.global_frames and at most video.global_frames_max.
        budget.overview_follows_answer_uniform (opt-in): at least
        budget.answer_uniform frames, so the agent sees the answer set's uniform frames (same uniform_indices call) whenever
        that grid is the larger one (e.g. 24+8 on videos whose overview would be 16)."""
        n = int(self.v["global_frames"])
        sp = self.v.get("global_spacing_s")
        if sp:
            cap = int(self.v.get("global_frames_max") or n)
            n = min(cap, max(n, 4 * int(round(self.T / (4 * float(sp))))))
        au = self.b.get("answer_uniform")
        if self.b.get("overview_follows_answer_uniform") and au:
            n = max(n, int(au))
        return n

    def frame_for_ref(self, ref: VisRef):
        return self.asset.index_of_pts(ref.pts)

    # ------------------------------------------------------------------ rendering
    def _vis(self, i, interval: Window):
        t = self.asset.time(i)
        rel = min(max((t - interval.start) / max(interval.length, 1e-6), 0.0), 1.0)
        return VisRef(self.asset.content_id, int(self.asset.pts(i)), t, rel, t / max(self.T, 1e-6))

    def _memory(self, obs_chars, keep_last):
        searched = ", ".join(f"[{_fmt_t(w.start)}-{_fmt_t(w.end)}]" for w in self.searched) or "none"
        ev = ", ".join(_fmt_t(self.asset.time(i)) for i in sorted(self.evidence, key=self.asset.time)) or "none"
        obs = self.observations if keep_last is None else self.observations[len(self.observations) - keep_last:] \
            if keep_last else []
        lines = []
        omitted = len(self.observations) - len(obs)
        if omitted:
            lines.append(f"  ({omitted} earlier observations omitted)")
        for o in obs:
            text = o.text.strip().replace("\n", " ")
            cap = self.b["max_observation_chars"] if obs_chars is None else obs_chars
            if len(text) > cap:
                text = text[:cap] + " [...]"
            lines.append(f"  [turn {o.turn + 1}, frames {', '.join(_fmt_t(t) for t in o.times)}] {text}")
        left = self.inspections_left()
        if self.nav_moves is not None:      # v2 (2026-09-30): budget counters instead of a turn counter, so the
            counters = (f"- Navigation moves left: {self.nav_left()} of {self.nav_moves}\n"   # 8- and 32-frame
                        f"- Inspections left: {left} of {self.max_inspections}\n")            # states match
        else:
            insp = "" if left is None else f"- Inspections left: {left} of {self.max_inspections}\n"
            counters = f"- Turns left: {self.b['max_turns'] - self.turn}\n" + insp
        return (f"- Active interval: [{_fmt_t(self.active.start)}-{_fmt_t(self.active.end)}]\n"
                f"- Searched intervals: {searched}\n"
                f"- Frames sent to the reader: {ev}\n"
                f"- Reader observations:\n" + ("\n".join(lines) if lines else "  none") + "\n" + counters)

    def _prefix(self, task, frames, instruction, level):
        """frames: list of (frame id, timeline index, interval for relative position)."""
        obs_chars, keep_last = self.COMPRESS[level]
        s = self.sample
        opts = "\n".join(f"({k}) {v}" for k, v in s.options.items())
        options_block = f"Options:\n{opts}\n" if s.options else ""     # caption queries have no options
        segs = [f"State:\nTask: {task}\nQuestion: {s.question}\n{options_block}"
                f"Memory:\n{self._memory(obs_chars, keep_last)}Frames (time-ordered):\n"]
        for fid, i, interval in frames:
            segs += [f"{fid} [{_fmt_t(self.asset.time(i))}] ", self._vis(i, interval), "\n"]
        return merge_text(segs) + [f"Question type: choice\nQuestion:\n{instruction}\n"]

    def _decision(self, dtype, task, frames, instruction, cands, ids, valid, payloads, tokenizer_len):
        """Builds the input, compressing the memory by the fixed rule until every path fits the
        soft budget; errors above the hard budget (candidates and frames are never truncated)."""
        level, max_len = 0, None
        for level in range(len(self.COMPRESS)):
            prefix = self._prefix(task, frames, instruction, level)
            if tokenizer_len is None:
                break
            pre = sum(tokenizer_len(x) if isinstance(x, str) else self.tpf for x in prefix)
            max_len = pre + max(tokenizer_len(c[0]) for c in cands) + 1
            if max_len <= self.b["path_tokens_soft"]:
                break
        if max_len is not None and max_len > self.b["path_tokens_hard"]:
            raise PathTooLong(f"{dtype}: {max_len} tokens > hard cap {self.b['path_tokens_hard']}")
        self.path_log.append({"type": dtype, "turn": self.turn, "max_path_tokens": max_len,
                              "compression_level": level, "n_candidates": len(cands)})
        return DecisionInput(dtype, prefix, cands, ids, valid, payloads)

    def _id_range(self, frames, w):
        inside = [fid for fid, i, _ in frames if w.start <= self.asset.time(i) < w.end]
        if not inside:
            return "no context frame inside"
        return f"frames {inside[0]}-{inside[-1]}" if len(inside) > 1 else f"frame {inside[0]}"

    # ------------------------------------------------------------------ controller
    def active_is_full(self):
        return self.active.key() == self.full.key()

    def controller_mask(self):
        if self.selection != "c":
            return {a: True for a in self.b["actions"]}   # v1: budgets never bind before the turn cap
        # v2 masks: re-localize always; refine only after a localization; inspect while the frame budget
        # lasts and the active interval still has k frames that were not sent to the reader
        mask = {}
        nav_ok = self.nav_left() is None or self.nav_left() > 0
        for a in self.b["actions"]:
            if a == "LOCALIZE_GLOBAL":
                mask[a] = nav_ok
            elif a == "REFINE_LOCAL":
                mask[a] = nav_ok and not self.active_is_full()
            elif a == "INSPECT_FRAMES":
                grid = self.asset.uniform_indices(self.active.start, self.active.end, self.b["frame_candidates"])
                unseen = [i for i in grid if i not in self.evidence]      # no feature cost for the mask
                mask[a] = (not self.frame_budget_done()) and len(unseen) >= self.b["frames_per_inspect"]
            elif a == "SWITCH_LOCAL":                       # window tree: v3 episodes only (episode_v3)
                mask[a] = False
            else:
                mask[a] = True
        return mask

    def controller_input(self, tokenizer_len=None):
        mask = self.controller_mask()
        frames = [(f"G{j + 1:02d}", i, self.full) for j, i in enumerate(self.global_idx)]
        acts = list(self.b["actions"])
        cands = [nanojev_candidate(f"{a}: {CONTROLLER_TEXT[a].format(k=self.b['frames_per_inspect'])}") for a in acts]
        return self._decision("controller", "controller", frames,
                              "Which action should the agent take next?", cands, acts,
                              [mask[a] for a in acts], [{"action": a} for a in acts], tokenizer_len)

    # ------------------------------------------------------------------ localization
    def refine_count(self, w):
        """Frames the locator sees in interval w when it chooses among w's children: video.refine_frames; with
        video.refine_spacing_s (localization accuracy below the root rises when these frames are
        <= ~10 s apart) 4 * round(w.length / (4 * spacing)), at least refine_frames, at most video.refine_frames_max."""
        n = int(self.v["refine_frames"])
        sp = self.v.get("refine_spacing_s")
        if sp:
            cap = int(self.v.get("refine_frames_max") or n)
            n = min(cap, max(n, 4 * int(round(w.length / (4 * float(sp))))))
        return n

    def localize_context(self, kind):
        if kind == "global":
            return [(f"G{j + 1:02d}", i, self.full) for j, i in enumerate(self.global_idx)]
        idx = self._pay_features(self.sparse_grid(self.active.start, self.active.end, self.refine_count(self.active)))
        return [(f"L{j + 1}", i, self.active) for j, i in enumerate(idx)]

    def localize_windows(self, kind):
        if kind == "global":
            return self.gen.global_windows(self.T)
        wins = self.gen.local_windows(self.active, self.T)
        if self.b.get("refine_exclude_active"):          # v2: choosing the active window again wastes a turn
            wins = [w for w in wins if w.kind != "active"]
        return wins

    def localize_input(self, kind, windows, tokenizer_len=None):
        frames = self.localize_context(kind)
        ids = [f"W{j + 1}" for j in range(len(windows))]
        cands = [nanojev_candidate(f"{cid}: {_fmt_t(w.start)}-{_fmt_t(w.end)} ({w.kind}, {self._id_range(frames, w)})")
                 for cid, w in zip(ids, windows)]
        what = "the whole video" if kind == "global" else "the active interval"
        inp = self._decision("localize", "localize", frames,
                             f"Select the window of {what} that most likely contains the evidence needed "
                             "to answer the question.", cands, ids, [True] * len(windows),
                             [w.to_json() for w in windows], tokenizer_len)
        if self.use_prior:
            inp.prior_texts = self.prior_texts()
            inp.prior_refs = [[self._vis(i, self.full) for _, i, _ in frames if w.start <= self.asset.time(i) < w.end]
                              for w in windows]
        return inp

    def apply_localize(self, w: Window):
        self.active = w
        self.active_set_turn = self.turn
        self.searched.append(w)

    # ------------------------------------------------------------------ frame selection
    def frame_candidates(self):
        """Frames displayed at the bin midpoints of the active interval (distinct real frames)."""
        return self._pay_features(self.asset.uniform_indices(self.active.start, self.active.end,
                                                             self.b["frame_candidates"]))

    def frame_input(self, cands, selected, tokenizer_len=None):
        frames = [(f"C{j + 1}", i, self.active) for j, i in enumerate(cands)]
        ids = [fid for fid, _, _ in frames]
        pos = {i: fid for fid, i, _ in frames}
        chosen = ", ".join(f"{pos[i]} ({_fmt_t(self.asset.time(i))})" for i in selected) or "none"
        k = self.b["frames_per_inspect"]
        texts = [nanojev_candidate(f"{fid}: frame {fid} at {_fmt_t(self.asset.time(i))}") for fid, i, _ in frames]
        return self._decision("frame", "frame_selection", frames,
                              f"Select frame {len(selected) + 1} of {k} from the active interval to send to "
                              f"the reader. Already selected: {chosen}.", texts, ids,
                              [i not in selected for i in cands],
                              [{"index": i, "t": self.asset.time(i)} for i in cands], tokenizer_len)

    def frame_input_c(self, cands, tokenizer_len=None):
        """Selection C: one decision scores all candidates; k picks with neighbour masking (sampler).
        Frames already sent to the reader are illegal."""
        frames = [(f"C{j + 1}", i, self.active) for j, i in enumerate(cands)]
        ids = [fid for fid, _, _ in frames]
        k = self.b["frames_per_inspect"]
        texts = [nanojev_candidate(f"{fid}: frame {fid} at {_fmt_t(self.asset.time(i))}") for fid, i, _ in frames]
        inp = self._decision("frame", "frame_selection", frames,
                             f"Select {k} frames from the active interval to send to the reader "
                             "(frames already sent are excluded).", texts, ids,
                             [i not in self.evidence for i in cands],
                             [{"index": i, "t": self.asset.time(i)} for i in cands], tokenizer_len)
        inp.select_k, inp.mask_radius = k, int(self.b.get("mask_radius", 1))
        if self.use_prior:
            inp.prior_texts = self.prior_texts()
            inp.prior_refs = [[self._vis(i, self.active)] for i in cands]
        return inp

    # ------------------------------------------------------------------ verifier (v2)
    def verifier_needed(self):
        """Ask the verifier only after new evidence, while frames and turns remain (else answer anyway)."""
        return (self.b.get("verifier", False) and bool(self.inspections) and not self.frame_budget_done()
                and self.turn < self.b["max_turns"])

    def verifier_input(self, tokenizer_len=None):
        seen = sorted(self.evidence, key=self.asset.time)
        frames = [(f"E{j + 1}", i, self.full) for j, i in enumerate(seen)]
        cuts = self.b.get("verifier_cuts", [0.25, 0.5, 0.8])
        ids = [f"T{j}" for j in range(4)]
        cands = [nanojev_candidate(f"{tid}: {txt.format(*cuts)}") for tid, txt in zip(ids, VERIFIER_TEXT)]
        return self._decision("verifier", "verifier", frames,
                              "If the agent answered now with the evidence collected so far, how likely is the "
                              "answer to be correct?", cands, ids, [True] * 4,
                              [{"tier": j} for j in range(4)], tokenizer_len)

    # ------------------------------------------------------------------ reader
    def request_frames(self, selected):
        idx = sorted(selected, key=self.asset.time)
        for i in idx:
            if i not in self.evidence:
                self.evidence.append(i)
        cap = self.b["reader_retained_frames"]
        if len(self.evidence) > cap:
            self.evidence = self.evidence[-cap:]
        self.inspections.append((self.turn, idx))
        self.cost["reader_calls"] += 1
        self.cost["reader_frames"] += len(idx)
        return ReaderRequest("observe", self.key, idx, [self.asset.time(i) for i in idx])

    def receive_observation(self, req, text, status="ok"):
        self.observations.append(Observation(self.turn, list(req.frame_times), text, status))

    def request_answer(self, reason):
        self.termination_reason = reason
        idx = sorted(self.evidence, key=self.asset.time)
        notes = [(f"frames {', '.join(_fmt_t(t) for t in o.times)}", o.text.strip()) for o in self.observations]
        self.cost["reader_calls"] += 1
        return ReaderRequest("answer", self.key, idx, [self.asset.time(i) for i in idx], notes=notes)

    def receive_answer(self, raw, parsed):
        self.answer_raw, self.answer = raw, parsed
        self.done = True

    def end_turn(self):
        self.turn += 1

    def summary(self):
        return {"turns": self.turn, "termination_reason": self.termination_reason,
                "answer": self.answer, "answer_raw": self.answer_raw,
                "searched": [w.to_json() for w in self.searched],
                "evidence_times": [self.asset.time(i) for i in sorted(self.evidence, key=self.asset.time)],
                "inspections": [{"turn": t, "times": [self.asset.time(i) for i in idx]} for t, idx in self.inspections],
                "observations": [o.__dict__ for o in self.observations],
                "path_log": self.path_log,
                "cost": dict(self.cost, encoder_frames=len(self.encoded | getattr(self, "encoded_extra", set())),
                             evidence_frames=len(self.evidence))}
