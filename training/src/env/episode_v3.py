"""Design B episode (budget.flow = v3, configs/v3_b32.yaml; 2026-10-01).

Fixed answerer budget of `answer_frames` (B) frames; on 1.5-3 min videos B = 8 (training, configs/v3_b8.yaml) and
16 (inference, configs/v3_b16.yaml); configs/v3_b32.yaml alone = 32 (not used for short videos):
  overview   `video.global_frames` (16) frames uniform over the whole video (incl. near the ends); they are the
             controller's / global locator's frames G01..G16 for every B; `budget.answer_overview` (0-based
             indices into G01..G16, default all) are also in the answerer input from the start (B = 8: G02 G06 G10
             G14; B = 16: G02 G04 .. G16; B = 32: all 16);
  inspection at most `max_inspections` (2): `frame_candidates` (32) frames uniform in the active window; a candidate
             closer than a quarter of the candidate spacing to a frame already in the answer set (overview + earlier
             picks) is illegal (near-duplicate); the selector picks `frames_per_inspect` (8)
             with selection C (src/policy/selection.py); no reader call, no observation text;
  verifier   asked after the overview (before any action) and after the first inspection; P(T3) >= tau stops the
             active search;
  fill       when the search stops (verifier, 2 inspections used, no legal action, turn cap) the answer set is filled
             to `answer_frames` from a fixed `fill_grid`-point (128) uniform grid, farthest point first (max of the
             min time distance to the current set; ties -> earlier), dedup by frame index;
  answer     the answerer answers ONCE from the time-sorted answer set, no notes.
State text: question, options, active / searched windows, inspections left, navigation moves left; frames as native
vision tokens (overview G.., selected E.. tagged with their inspection). No reader observations, no text summaries.
"""
from .environment import Episode, ReaderRequest, _fmt_t, merge_text, nanojev_candidate

VERIFIER_TEXT_V3 = ("the answerer would most likely be wrong (below {0:.0%} for the correct option)",
                    "weak evidence ({0:.0%}-{1:.0%} for the correct option)",
                    "probably enough ({1:.0%}-{2:.0%} for the correct option)",
                    "enough evidence ({2:.0%} or more for the correct option)")


# budget.verifier_neutral (2026-10-05, design "rec 4"): option texts state only the probability range, no evaluative
# words; ids V1..V4 follow the (shuffled) position, so neither the id nor the order tells the tier. The tier of each
# candidate is in its payload; use tier_probs() to read probabilities in tier order.
VERIFIER_RANGE_TEXT = ("the answerer's probability for the correct option is below {0:.0%}",
                       "the answerer's probability for the correct option is between {0:.0%} and {1:.0%}",
                       "the answerer's probability for the correct option is between {1:.0%} and {2:.0%}",
                       "the answerer's probability for the correct option is {2:.0%} or more")


def enclosing_name(k, whole):
    """k = 0 for the parent, 1 for the grandparent, ...; the whole video is named as such; k = -1: nowhere else."""
    if k < 0:
        return "nowhere else"
    return "the whole video" if whole else ("the parent interval" if k == 0 else f"the interval {k + 1} levels up")


def tier_probs(inp, probs):
    """Verifier probabilities in tier order T0..T3 from probabilities in candidate order (inp: DecisionInput or its
    json; candidates carry {"tier": k} payloads). Identity for the old fixed-order format."""
    payloads = inp.payloads if hasattr(inp, "payloads") else inp["payloads"]
    tiers = [pl for pl in payloads if "tier" in pl]          # joint Q3 + where inputs carry other groups too
    out = [0.0] * len(tiers)
    for pos, pl in enumerate(payloads):
        if "tier" in pl:
            out[pl["tier"]] = probs[pos]
    return out


def tier_position(inp, tier):
    payloads = inp.payloads if hasattr(inp, "payloads") else inp["payloads"]
    return next(pos for pos, pl in enumerate(payloads) if pl.get("tier") == tier)


class EpisodeV3(Episode):
    def __init__(self, sample, asset, features, encoder, cfg, key):
        super().__init__(sample, asset, features, encoder, cfg, key)
        p = cfg["policy"]
        if p.get("frame_tokens") == "qwen_native":         # path-length estimate: tokens + vision_start/end
            self.tpf = (p.get("native_side", 256) // 32) ** 2 + 2
        self.answer_frames = int(self.b.get("answer_frames", 32))
        ov = self.b.get("answer_overview")             # indices into G01..; None = all overview frames
        self.answer_overview = None if ov is None else [int(j) for j in ov]
        # budget.answer_uniform (2026-10-19, an evaluation variant): the answer set's overview is its own uniform grid
        # of N frames instead of G01..; the policy still sees G01.. (video.global_frames). With kept_frames =
        # answer_frames - N the answerer reads N uniform frames + up to kept_frames selected ones.
        au = self.b.get("answer_uniform")
        self.answer_uniform = None if au is None else int(au)
        self.fill_grid = int(self.b.get("fill_grid", 128))
        self.verifier_points = list(self.b.get("verifier_points", ["overview", "inspection1"]))
        self.pick_inspection = {}       # frame index -> inspection number (1-based)
        self.verifier_asked = set()     # verifier points already asked
        self.fill = []                  # frames added by the uniform fill (final answer set)
        self.withdrawn = []             # tentative inspections whose frames were taken back (budget.tentative)
        # verifier-in-the-loop flow (configs/v3_ctrlver_b16.yaml, 2026-10-04)
        self.max_attempts = int(self.b["max_attempts"]) if self.b.get("tentative") and self.b.get("max_attempts") else None
        self.attempts = 0               # inspections made, withdrawn ones included (restores are not attempts)
        self.withdrawn_idx = set()      # frames taken back: never offered again as candidates (no identical retry)
        self.frames_inspected = 0       # frames picked by inspections, withdrawn ones included (accounting)
        self.n_restored = 0             # pooled inspections put back at the end (accounting)
        self.ver_line = None            # budget.verifier_line: last verifier result, shown to the controller only
        self.verifier_calls = 0         # budget.verifier_shuffle: seeds the option order of each verifier call
        # budget.verifier_per_frame (2026-10-06): the verifier keeps / rejects every inspected frame; the kept-frame budget
        # is max_inspections x frames_per_inspect; attempt_log records every inspection attempt (window, picked, kept)
        self.per_frame = bool(self.b.get("verifier_per_frame"))
        kf = self.b.get("kept_frames")                 # 2026-10-06 review: its own key (was max_inspections x batch)
        self.frame_cap = int(kf) if kf is not None else int(self.max_inspections or 0) * int(self.b.get("frames_per_inspect", 1))
        self.attempt_log = []
        self.loc_lines = []             # budget.locator_line: locator probe of this turn, shown to the controller only
        self.loc_info = {}              # the same probe as numbers (kind -> windows, probs, margin), for rules and logs
        # budget.verifier_where (2026-10-06): the verifier's answer to "where is the evidence, relative to the active
        # interval" (before / inside_small / inside_large / after -> probability); None while the active interval is the
        # full video. The controller sees inside / outside, the global locator sees before / after.
        self.where_probs = None
        self.where_window = None
        self._loc_kind = None
        # budget.window_tree: a strict window tree. The children of an interval are its 4 quarters
        # (budget.tree_quarters_only; else the 8 windows of candidates.windows8). path = windows from level 1 down to the
        # active one ([] = whole video, the root). REFINE_LOCAL enters an unvisited child (at the root: picks a level-1
        # window), SWITCH_LOCAL an unvisited sibling (another child of the parent; at level 1 another level-1 window),
        # UP returns to the parent (level >= 2 only; no locator call), INSPECT_FRAMES inspects the active window. The
        # budget is max_turns actions in total (UP included) and the kept-frame cap; budget.windows_per_level /
        # inspections_per_window (null = off) are optional extra caps.
        self.tree = bool(self.b.get("window_tree"))
        self.path = []
        lc, ic = self.b.get("windows_per_level"), self.b.get("inspections_per_window")
        self.level_cap = None if lc is None else int(lc)
        self.inspect_cap = None if ic is None else int(ic)
        self.tried = {}                 # parent window key -> children entered under it
        self.insp_count = {}            # window key -> inspection attempts made in it

    # ------------------------------------------------------------------ sets
    def overview_in_answer(self):
        """The overview frames that are in the answer set (all of G01.. unless budget.answer_overview is set;
        budget.answer_uniform replaces them by a uniform grid of that many frames)."""
        if self.answer_uniform:
            return list(dict.fromkeys(self.asset.uniform_indices(0.0, self.T, self.answer_uniform)))
        g = list(self.global_idx)
        return g if self.answer_overview is None else [g[j] for j in self.answer_overview if j < len(g)]

    def answer_set(self):
        """Current answer set before the fill: overview frames in the answer set + selected frames (dedup, insertion
        order)."""
        return list(dict.fromkeys(self.overview_in_answer() + list(self.evidence)))

    def filled_set(self, base=None):
        """`base` (default: the current answer set) filled to answer_frames by the farthest-point rule."""
        out = list(dict.fromkeys(self.answer_set() if base is None else base))
        if len(out) >= self.answer_frames:
            return sorted(out[:self.answer_frames], key=self.asset.time)
        used = set(out)
        grid = [i for i in self.asset.uniform_indices(0.0, self.T, self.fill_grid) if i not in used]
        times = {i: self.asset.time(i) for i in grid}
        selected_times = [self.asset.time(i) for i in out]
        # Each new selection can only decrease the nearest-selected-frame distance.
        # Update that minimum once, instead of rebuilding all pair distances each iteration.
        nearest = {i: min((abs(times[i] - t) for t in selected_times), default=float('inf')) for i in grid}
        while len(out) < self.answer_frames and grid:
            best = max(grid, key=lambda i: (nearest[i], -times[i]))
            out.append(best)
            grid.remove(best)
            for i in grid:
                nearest[i] = min(nearest[i], abs(times[i] - times[best]))
        return sorted(out[:self.answer_frames], key=self.asset.time)

    # ------------------------------------------------------------------ window tree
    def parent_window(self):
        """Parent of the active window (the whole video for a top-level window); None at the whole video."""
        if not self.path:
            return None
        return self.path[-2] if len(self.path) >= 2 else self.full

    def ancestors(self):
        """Windows that contain the active one, nearest first: parent, grandparent, ..., the whole video."""
        return list(reversed(self.path[:-1])) + [self.full] if self.path else []

    def apply_up(self):
        """Window tree: the parent becomes the active window again (not a new window: no locator call, not searched)."""
        self.path = self.path[:-1]
        self.active = self.path[-1] if self.path else self.full
        self.active_set_turn = self.turn

    def inspections_here(self):
        return self.insp_count.get(self.active.key(), 0)

    def apply_window(self, kind, w):
        """Make `w` the active window (kind: global | local | sibling) and update the path and the per-level counts."""
        if kind == "global":
            parent, self.path = self.full, [w]
        elif kind == "local":
            parent = self.active
            self.path = self.path + [w]
        elif kind == "sibling":
            parent = self.parent_window()
            self.path = self.path[:-1] + [w]
        else:
            raise ValueError(kind)
        self.tried[parent.key()] = self.tried.get(parent.key(), 0) + 1
        self.apply_localize(w)

    def candidate_windows(self, kind):
        """Windows offered to the locator for kind global | local | sibling. budget.relocalize_exclude_searched: drop
        exact repeats of searched windows and windows that overlap a searched window by >= 50% of their own length
        (searched windows that contain the candidates -- the whole video and, in the window tree, the path above the
        candidates' parent -- do not count). Window tree: never the whole video as a top-level window, and no fall-back
        to searched windows (an empty list makes the action illegal); flat flow: global only, all windows if none left."""
        wins = self.localize_windows(kind)
        if self.tree and kind == "global":
            wins = [w for w in wins if w.kind != "full"]
        if self.tree and self.b.get("tree_quarters_only"):     # strict tree: siblings never overlap, so only windows
            seen = {u.key() for u in self.searched}               # already visited are dropped
            return [w for w in wins if w.key() not in seen]
        if not self.b.get("relocalize_exclude_searched") or not self.searched or (not self.tree and kind != "global"):
            return wins
        keep_out = {self.full.key()}
        if self.tree and kind == "local":
            keep_out |= {u.key() for u in self.path}
        elif self.tree and kind == "sibling":
            keep_out |= {u.key() for u in self.path[:-1]}
        places = [u for u in self.searched if u.key() not in keep_out]
        seen = {u.key() for u in self.searched}
        ov = lambda w, u: max(0.0, min(w.end, u.end) - max(w.start, u.start)) / max(w.length, 1e-9)
        rest = [w for w in wins if w.key() not in seen and all(ov(w, u) < 0.5 - 1e-6 for u in places)]
        return rest if self.tree else (rest or wins)

    def where_frames(self):
        """B: frames shown to the verifier when it judges where the evidence is: the overview plus
        budget.refine_frames frames uniform in the inspection range of the active window (A01..), time-ordered; the
        overview alone at the root."""
        frames = [(f"G{j + 1:02d}", i, self.full) for j, i in enumerate(self.global_idx)]
        if self.active_is_full() or not self.b.get("where_window_frames", True):
            return frames
        r = self.inspect_range()                       # every other frame of the inspection candidate grid: encoded once,
        grid = self.asset.uniform_indices(r.start, r.end, self.b["frame_candidates"])     # reused by the inspection
        idx = self._pay_features(grid[::max(1, len(grid) // self.v["refine_frames"])][:self.v["refine_frames"]])
        frames += [(f"A{j + 1:02d}", i, r) for j, i in enumerate(idx)]
        return frames                                      # _prefix orders them (overview first if static)

    def tried_under(self, w):
        return self.tried.get(w.key(), 0) if w is not None else 0

    def _under_cap(self, w):
        return self.level_cap is None or self.tried_under(w) < self.level_cap

    def localize_windows(self, kind):
        if kind == "sibling":
            parent = self.parent_window()
            wins = [u for u in self.gen.local_windows(parent, self.T) if u.kind != "active"] if parent is not None else []
        else:
            wins = super().localize_windows(kind)
        if self.tree and self.b.get("tree_quarters_only"):
            wins = [u for u in wins if u.kind == "quarter"]
        return wins

    def localize_context(self, kind):
        # window tree: choosing among the children of the whole video (REFINE at the root, SWITCH at level 1) uses the 16
        # overview frames (already encoded; 4 per quarter); deeper levels sample budget.refine_frames in the interval whose
        # children are offered (16: 4 per quarter)
        if self.tree and ((kind == "local" and self.active_is_full()) or (kind == "sibling" and len(self.path) == 1)):
            return super().localize_context("global")
        if kind == "sibling":
            parent = self.parent_window()
            idx = self._pay_features(self.asset.uniform_indices(parent.start, parent.end, self.v["refine_frames"]))
            return [(f"L{j + 1}", i, parent) for j, i in enumerate(idx)]
        return super().localize_context(kind)

    # ------------------------------------------------------------------ state text
    def frame_budget_done(self):
        if self.per_frame:                             # kept frames, not kept batches
            return len(self.evidence) >= self.frame_cap
        return super().frame_budget_done()

    def inspections_left(self):
        if self.per_frame:
            return max(0, self.frame_cap - len(self.evidence))
        return super().inspections_left()

    def _memory(self, obs_chars=None, keep_last=None, task=None):
        searched = ", ".join(f"[{_fmt_t(w.start)}-{_fmt_t(w.end)}]" for w in self.searched) or "none"
        if self.tree and self.per_frame:
            budget = (f"- Kept frames: {len(self.evidence)} of {self.frame_cap}; actions left "
                      f"{self.b['max_turns'] - self.turn} of {self.b['max_turns']}\n")
        else:
            budget = (f"- Kept frames: {len(self.evidence)} of {self.frame_cap}; inspection attempts left "
                      f"{self.max_attempts - self.attempts} of {self.max_attempts}\n" if self.per_frame and self.max_attempts
                      else f"- Inspections left: {self.inspections_left()} of {self.max_inspections}\n")
        if self.tree:
            win = (self.active.start, self.active.end)    # A: the expert's retry / leave rule counts the
            low = sum(1 for x in self.attempt_log          # attempts here that kept less than half, so the student sees it
                      if tuple(x["window"]) == win and 2 * x["kept"] < x["picked"])
            nav = (f"- Window level: {len(self.path)} (0 = whole video); inspection attempts in this window "
                   f"{self.inspections_here()}, of which kept less than half: {low}\n")
        else:
            nav = f"- Navigation moves left: {self.nav_left()} of {self.nav_moves}\n"
        out = (f"- Active interval: [{_fmt_t(self.active.start)}-{_fmt_t(self.active.end)}]\n"
               f"- Searched intervals: {searched}\n" + budget + nav)
        if task == "controller" and self.ver_line:      # budget.verifier_line: the controller sees the verifier
            out += f"- {self.ver_line}\n"
        if task == "controller":                         # budget.locator_line: and the locator's current preference
            out += "".join(f"- {s}\n" for s in self.loc_lines)
        wp, ww = self.where_probs, self.where_window
        if wp and task == "controller":
            q = self.b.get("evidence_refine_frac", 0.25)
            inside = wp["inside_small"] + wp["inside_large"]
            out += (f"- Verifier, evidence vs the active interval: inside {inside:.2f} (covering less than {q:.0%} of it "
                    f"{wp['inside_small']:.2f}, {q:.0%} or more {wp['inside_large']:.2f}), outside {1 - inside:.2f}"
                    + ("; smallest enclosing interval with evidence: " + ", ".join(f"{n} {p:.2f}" for n, p in wp["enclosing"])
                       if wp.get("enclosing") else "") + "\n")
        if wp and task == "localize" and self._loc_kind in ("global", "sibling"):
            out += (f"- Verifier, evidence vs the interval [{_fmt_t(ww[0])}-{_fmt_t(ww[1])}]: before it "
                    f"{wp['before']:.2f}, after it {wp['after']:.2f}\n")
        return out

    def localize_input(self, kind, windows, tokenizer_len=None):
        self._loc_kind = kind                 # global / sibling locator inputs carry the verifier's before / after
        try:
            if kind != "sibling":
                return super().localize_input(kind, windows, tokenizer_len)
            from .environment import nanojev_candidate as nc
            frames = self.localize_context(kind)
            par = self.parent_window()
            ids = [f"W{j + 1}" for j in range(len(windows))]
            cands = [nc(f"{cid}: {_fmt_t(w.start)}-{_fmt_t(w.end)} ({w.kind}, {self._id_range(frames, w)})")
                     for cid, w in zip(ids, windows)]
            return self._decision("localize", "localize", frames,
                                  f"Select another window of the parent interval [{_fmt_t(par.start)}-{_fmt_t(par.end)}] "
                                  "that most likely contains the evidence needed to answer the question.", cands, ids,
                                  [True] * len(windows), [w.to_json() for w in windows], tokenizer_len)
        finally:
            self._loc_kind = None

    def _prefix(self, task, frames, instruction, level=0, selected=False):
        """budget.static_prefix_first: the part that stays the same for a role during an
        episode comes first -- task, question, options and the overview frames G01.. -- and everything that changes
        (memory, other frames, selected frames, the decision question) after it. self._static_segments is the number of
        leading prefix segments of that static head (ends right after the last overview image, 0 = none), used by the
        policy's static-prefix KV cache (src/policy/model.py)."""
        s = self.sample
        opts = "\n".join(f"({k}) {v}" for k, v in s.options.items())
        options_block = f"Options:\n{opts}\n" if s.options else ""
        self._static_segments = 0
        if self.b.get("static_prefix_first"):
            over = [f for f in frames if f[0].startswith("G")]
            rest = sorted((f for f in frames if not f[0].startswith("G")), key=lambda f: self.asset.time(f[1]))
            head = [f"State:\nTask: {task}\nQuestion: {s.question}\n{options_block}"]
            if over:
                head[0] += "Overview frames (time-ordered):\n"
                for j, (fid, i, interval) in enumerate(over):
                    head += [("\n" if j else "") + f"{fid} [{_fmt_t(self.asset.time(i))}] ", self._vis(i, interval)]
            head = merge_text(head)
            if over:
                self._static_segments = len(head)            # ends with the last overview image
            segs = head + [("\n" if over else "") + f"Memory:\n{self._memory(task=task)}"]
            if rest:
                segs.append("Frames (time-ordered):\n")
                for fid, i, interval in rest:
                    segs += [f"{fid} [{_fmt_t(self.asset.time(i))}] ", self._vis(i, interval), "\n"]
            frames = []
        else:
            segs = [f"State:\nTask: {task}\nQuestion: {s.question}\n{options_block}"
                    f"Memory:\n{self._memory(task=task)}Frames (time-ordered):\n"]
        for fid, i, interval in frames:
            segs += [f"{fid} [{_fmt_t(self.asset.time(i))}] ", self._vis(i, interval), "\n"]
        if selected:                       # controller / verifier: the frames picked so far, as images
            sel = sorted(self.evidence, key=self.asset.time)
            if sel:
                segs.append("Selected frames (time-ordered):\n")
                for j, i in enumerate(sel):
                    segs += [f"E{j + 1} [{_fmt_t(self.asset.time(i))}] (inspection {self.pick_inspection.get(i, 0)}) ",
                             self._vis(i, self.full), "\n"]
            else:
                segs.append("Selected frames: none\n")
        if self._static_segments:                          # keep the static head's segment boundaries
            n = self._static_segments
            return segs[:n] + merge_text(segs[n:]) + [f"Question type: choice\nQuestion:\n{instruction}\n"]
        return merge_text(segs) + [f"Question type: choice\nQuestion:\n{instruction}\n"]

    def _decision_v3(self, dtype, task, frames, instruction, cands, ids, valid, payloads, tokenizer_len, selected=False,
                     group_questions=None):
        from ..common.types import DecisionInput
        from .environment import PathTooLong
        prefix = self._prefix(task, frames, instruction, selected=selected)
        max_len = None
        if tokenizer_len is not None:
            pre = sum(tokenizer_len(x) if isinstance(x, str) else self.tpf for x in prefix)
            max_len = pre + max(tokenizer_len(c[0]) for c in cands) + 1
            if max_len > self.b["path_tokens_hard"]:
                raise PathTooLong(f"{dtype}: {max_len} tokens > hard cap {self.b['path_tokens_hard']}")
        self.path_log.append({"type": dtype, "turn": self.turn, "max_path_tokens": max_len, "compression_level": 0,
                              "n_candidates": len(cands)})
        inp = DecisionInput(dtype, prefix, cands, ids, valid, payloads)
        inp.static_segments = getattr(self, "_static_segments", 0)
        if group_questions:                               # letter readout: one question per group
            inp.group_questions = {str(k): v for k, v in group_questions.items()}
        return inp

    def _decision(self, dtype, task, frames, instruction, cands, ids, valid, payloads, tokenizer_len):
        return self._decision_v3(dtype, task, frames, instruction, cands, ids, valid, payloads, tokenizer_len)

    # ------------------------------------------------------------------ controller
    def _taken(self, cands):
        """Per candidate: already in the answer set, or closer than a quarter of the candidate spacing to one of its
        frames (2026-10-01: half the spacing marked every whole-video candidate as a duplicate, because the 32-point
        grid interleaves with the 16 overview frames at exactly half a spacing)."""
        span = self.inspect_range().length if hasattr(self, "inspect_range") else self.active.length
        gap = 0.25 * span / max(self.b["frame_candidates"], 1)
        taken = set(self.answer_set()) | getattr(self, "withdrawn_idx", set())   # withdrawn frames: never offered again
        ts = [self.asset.time(i) for i in taken]
        out = [i in taken or any(abs(self.asset.time(i) - t) < gap for t in ts) for i in cands]
        mg = float(self.b.get("inspect_min_gap", 0) or 0)  # G1 diagnostic (2026-10-02; 0 = off): candidates closer than
        if mg > 0 and self.evidence:                        # (mg - 0.5) grid steps to a frame picked by an earlier
            near = (mg - 0.5) * 4 * gap                     # inspection are illegal too (2 = no adjacent picks)
            es = [self.asset.time(i) for i in self.evidence]
            out = [o or any(abs(self.asset.time(i) - t) < near for t in es) for o, i in zip(out, cands)]
        return out

    def inspect_range(self):
        """Interval the inspection candidates are drawn from: the active window, widened on both sides by
        budget.inspect_margin x its length in the window tree (1/8, so evidence cut by a quarter boundary
        is still reachable from the neighbouring window), clipped to the video. The active window itself is unchanged."""
        from .candidates import Window
        m = float(self.b.get("inspect_margin") or 0.0) if self.tree else 0.0
        if m <= 0.0 or self.active_is_full():
            return self.active
        L = self.active.length
        return Window(max(0.0, self.active.start - m * L), min(self.T, self.active.end + m * L), "inspect")

    def range_of(self, w):
        """Inspection range of window w (w widened by budget.inspect_margin, clipped; w itself at the root)."""
        from .candidates import Window
        m = float(self.b.get("inspect_margin") or 0.0) if self.tree else 0.0
        if m <= 0.0 or w.key() == self.full.key():
            return w
        return Window(max(0.0, w.start - m * w.length), min(self.T, w.end + m * w.length), "inspect")

    def new_grid_points(self, w, spans):
        """Expert helper (b, 2026-10-06): how many points of window w's inspection grid lie inside the evidence spans and
        were never shown (not in the answer set, not withdrawn) -- what entering w and inspecting could still add."""
        r = self.range_of(w)
        seen = set(self.answer_set()) | set(getattr(self, "withdrawn_idx", set()))
        grid = self.asset.uniform_indices(r.start, r.end, self.b["frame_candidates"])
        return sum(1 for i in grid if i not in seen and any(s <= self.asset.time(i) <= e for s, e in spans))

    def unseen_candidates(self):
        """Candidates an inspection could offer now: not taken, and either already encoded or within the episode's
        encoder budget (video.encoder_frame_cap; frame_candidates() drops frames beyond it, 2026-10-06 fix: the
        INSPECT mask used to count frames that could no longer be encoded, giving an empty candidate list)."""
        r = self.inspect_range()
        grid = self.asset.uniform_indices(r.start, r.end, self.b["frame_candidates"])
        room = self.v["encoder_frame_cap"] - len(self.encoded)
        out = []
        for i, tk in zip(grid, self._taken(grid)):
            if i not in self.encoded:
                if room <= 0:
                    continue
                room -= 1
            if not tk:
                out.append(i)
        return out

    def frame_candidates(self):
        r = self.inspect_range()
        return self._pay_features(self.asset.uniform_indices(r.start, r.end, self.b["frame_candidates"]))

    def attempts_done(self):
        return self.max_attempts is not None and self.attempts >= self.max_attempts

    def controller_mask(self):
        nav_ok = self.nav_left() is None or self.nav_left() > 0
        cap = self.level_cap
        mask = {}
        for a in self.b["actions"]:
            if a == "LOCALIZE_GLOBAL":           # window tree: no direct jump (SWITCH at level 1 changes level-1 windows)
                mask[a] = nav_ok and not self.tree
            elif a == "REFINE_LOCAL" and self.tree:   # an unvisited child (at the root: picks a level-1 window)
                mask[a] = nav_ok and self._under_cap(self.active) and len(self.candidate_windows("local")) > 0
            elif a == "REFINE_LOCAL":            # needs at least one sub-window (very short windows have none)
                mask[a] = nav_ok and not self.active_is_full() and len(self.localize_windows("local")) > 0
            elif a == "SWITCH_LOCAL":            # window tree: an unvisited sibling (level >= 1)
                mask[a] = (self.tree and nav_ok and len(self.path) >= 1 and self._under_cap(self.parent_window())
                           and len(self.candidate_windows("sibling")) > 0)
            elif a == "UP":                      # window tree: back to the parent, level >= 2 (level 1 -> root, then
                mask[a] = self.tree and len(self.path) >= 2   # REFINE would duplicate SWITCH)
            elif a == "INSPECT_FRAMES":
                mask[a] = ((not self.frame_budget_done()) and not self.attempts_done()
                           and len(self.unseen_candidates()) >= self.b["frames_per_inspect"]
                           and (not self.tree or self.inspect_cap is None or self.inspections_here() < self.inspect_cap)
                           and not (self.tree and self.active_is_full() and not self.b.get("tree_root_inspect")))
            else:
                mask[a] = False
        return mask

    def controller_input(self, tokenizer_len=None):
        from .environment import CONTROLLER_TEXT
        mask = self.controller_mask()
        frames = [(f"G{j + 1:02d}", i, self.full) for j, i in enumerate(self.global_idx)]
        acts = list(self.b["actions"])
        k = self.b["frames_per_inspect"]
        # budget.inspect_text_budget_free (2026-10-06, long-video flow): no frame count in the text, so the controller
        # input stays the same when the batch size changes (train at 32 frames, run at 64 = 16 overview + larger batches)
        insp = ("pick a batch of frames in the active interval for the answerer" if self.b.get("inspect_text_budget_free")
                else "pick {k} frames in the active interval for the answerer")
        text = dict(CONTROLLER_TEXT, INSPECT_FRAMES=insp)
        cands = [nanojev_candidate(f"{a}: {text[a].format(k=k)}") for a in acts]
        return self._decision_v3("controller", "controller", frames, "Which action should the agent take next?",
                                 cands, acts, [mask[a] for a in acts], [{"action": a} for a in acts], tokenizer_len,
                                 selected=True)

    # ------------------------------------------------------------------ frame selection
    def frame_input_c(self, cands, tokenizer_len=None):
        frames = [(f"C{j + 1}", i, self.inspect_range()) for j, i in enumerate(cands)]
        ids = [fid for fid, _, _ in frames]
        k = self.b["frames_per_inspect"]
        taken = self._taken(cands)
        texts = [nanojev_candidate(f"{fid}: frame {fid} at {_fmt_t(self.asset.time(i))}") for fid, i, _ in frames]
        inp = self._decision_v3("frame", "frame_selection", frames,
                                (f"Select {k} frames for the answerer from the candidates of the active interval "
                                 "(widened on both sides; frames already in the answer set are excluded)."
                                 if self.inspect_range().key() != self.active.key() else
                                 f"Select {k} frames from the active interval for the answerer "
                                 "(frames already in the answer set are excluded)."), texts, ids,
                                [not tk for tk in taken],
                                [{"index": i, "t": self.asset.time(i)} for i in cands], tokenizer_len)
        inp.select_k, inp.mask_radius = k, int(self.b.get("mask_radius", 1))
        return inp

    # ------------------------------------------------------------------ verifier
    def verifier_point(self):
        """Name of the verifier point reached now, or None: 'overview' before any inspection (asked once, at the
        start), 'inspection1' right after the first inspection."""
        if not self.b.get("verifier", False):
            return None
        n = len(self.inspections)
        name = "overview" if n == 0 else f"inspection{n}"
        if name in self.verifier_points and name not in self.verifier_asked and not self.frame_budget_done():
            return name
        return None

    def verifier_input(self, tokenizer_len=None, with_where=False):
        """Q3. with_where (budget.verifier_q3_where_joint): when the active window is not the whole video,
        the "where is the evidence" groups are asked in the same decision (same prefix, one forward): Q3 = group 0,
        where = group 1, smallest enclosing interval = group 2; each group is normalized on its own."""
        joint = with_where and not self.active_is_full()
        frames = self.where_frames() if joint else [(f"G{j + 1:02d}", i, self.full) for j, i in enumerate(self.global_idx)]
        cuts = self.b.get("verifier_cuts", [0.25, 0.5, 0.8])
        if self.b.get("verifier_neutral"):
            order = list(range(4))
            if self.b.get("verifier_shuffle", True):
                import random
                random.Random(f"{self.key}|verifier|{self.verifier_calls}").shuffle(order)
            self.verifier_calls += 1
            ids = [f"V{pos + 1}" for pos in range(4)]
            cands = [nanojev_candidate(f"{ids[pos]}: {VERIFIER_RANGE_TEXT[k].format(*cuts)}") for pos, k in enumerate(order)]
            payloads = [{"tier": k} for k in order]
        else:
            ids = [f"T{j}" for j in range(4)]
            cands = [nanojev_candidate(f"{tid}: {txt.format(*cuts)}") for tid, txt in zip(ids, VERIFIER_TEXT_V3)]
            payloads = [{"tier": j} for j in range(4)]
        q = ("If the answerer saw only the overview and the selected frames above, how likely would it choose the "
             "correct option?" if self.b.get("label_no_fill") else
             "If the agent stopped searching now (the remaining answer slots are filled with uniformly spaced frames), "
             "how likely is the answerer to choose the correct option?")
        gq = None
        if joint:
            payloads = [dict(pl, group=0) for pl in payloads]
            wids, wcands, wpls, wq, wgq = self._where_options(group0=1)
            ids, cands, payloads = ids + wids, cands + wcands, payloads + wpls
            gq = {0: q, **wgq}
            q = f"{q} Also: {wq}"
        return self._decision_v3("verifier", "verifier", frames, q, cands, ids, [True] * len(cands), payloads,
                                 tokenizer_len, selected=True, group_questions=gq)

    def verifier_frame_input(self, idx, tokenizer_len=None):
        """budget.verifier_per_frame: does the selected frame `idx` (shown as E<j> among the selected frames) show evidence
        needed to answer? Two options; payload {"keep": bool}. Label (scripts/gen_role_sft_from_trajectories.py, role
        verifier_frame): the frame lies in the annotated evidence widened by budget.evidence_pad seconds."""
        frames = [(f"G{j + 1:02d}", i, self.full) for j, i in enumerate(self.global_idx)]
        sel = sorted(self.evidence, key=self.asset.time)
        name = f"E{sel.index(idx) + 1}" if idx in sel else "the frame"
        ids = ["V1", "V2"]
        cands = [nanojev_candidate("V1: yes, it is in or near the part of the video that answers the question"),
                 nanojev_candidate("V2: no, it is not")]
        return self._decision_v3("verifier", "verifier", frames,
                                 f"Is selected frame {name} [{_fmt_t(self.asset.time(idx))}] in or near the part of the "
                                 f"video that answers the question?", cands, ids, [True, True],
                                 [{"keep": True, "frame": int(idx), "time": self.asset.time(idx)},
                                  {"keep": False, "frame": int(idx), "time": self.asset.time(idx)}],
                                 tokenizer_len, selected=True)

    def verifier_frames_input(self, idxs, tokenizer_len=None):
        """budget.verifier_frame_joint (2026-10-06): one decision for a whole inspection batch. For every frame of the
        batch (E<j> among the selected frames) two candidates "E<j> yes" / "E<j> no"; payload {"keep", "frame", "time",
        "group": j}; the policy normalizes each pair on its own (src/policy/model.grouped_log_softmax), so P(keep) of a
        frame means the same as in the one-frame question, with one prefix forward for the batch."""
        frames = [(f"G{j + 1:02d}", i, self.full) for j, i in enumerate(self.global_idx)]
        sel = sorted(self.evidence, key=self.asset.time)
        order = sorted(idxs, key=self.asset.time)
        names = [f"E{sel.index(i) + 1}" if i in sel else f"F{j + 1}" for j, i in enumerate(order)]
        cands, ids, payloads = [], [], []
        for g, (i, name) in enumerate(zip(order, names)):
            t = self.asset.time(i)
            for keep in (True, False):
                ids.append(f"{name}{'+' if keep else '-'}")
                cands.append(nanojev_candidate(f"{ids[-1]}: {name} [{_fmt_t(t)}] " +
                                               ("yes, it is in or near the part of the video that answers the "
                                                "question" if keep else "no, it is not")))
                payloads.append({"keep": keep, "frame": int(i), "time": t, "group": g})
        q = (f"For each of the frames {', '.join(names)} picked by the last inspection: is it in or near the part of the "
             f"video that answers the question? (one yes/no per frame)")
        gq = {g: f"Is selected frame {name} [{_fmt_t(self.asset.time(i))}] in or near the part of the video that answers "
                 f"the question?" for g, (i, name) in enumerate(zip(order, names))}
        return self._decision_v3("verifier", "verifier", frames, q, cands, ids, [True] * len(cands), payloads,
                                 tokenizer_len, selected=True, group_questions=gq)

    def verifier_where_input(self, tokenizer_len=None):
        """budget.verifier_where: where is the evidence relative to the active interval? Four options in fixed order,
        payload {"where": before | inside_small | inside_large | after}. Label (scripts/gen_role_sft_from_trajectories.py,
        role verifier_where): src/baselines/evidence.where_label on key_spans +- budget.evidence_pad."""
        frames = self.where_frames()
        ids, cands, payloads, q, gq = self._where_options(group0=0)
        return self._decision_v3("verifier", "verifier", frames, q, cands, ids, [True] * len(cands), payloads,
                                 tokenizer_len, selected=True, group_questions=gq if len(gq) > 1 else None)

    def _where_options(self, group0=0):
        """Candidates of the "where is the evidence" question: relative to the active window (group `group0`) and, at
        level >= 2 of the window tree, the smallest enclosing interval with evidence (group `group0` + 1)."""
        q = self.b.get("evidence_refine_frac", 0.25)
        a = f"[{_fmt_t(self.active.start)}-{_fmt_t(self.active.end)}]"
        texts = {"before": "before the active interval",
                 "inside_small": f"inside the active interval or its margins, covering less than {q:.0%} of it",
                 "inside_large": f"inside the active interval or its margins, covering {q:.0%} of it or more",
                 "after": "after the active interval"}
        from ..baselines.evidence import WHERE
        ids = [f"P{j + 1}" for j in range(len(WHERE))]
        cands = [nanojev_candidate(f"{pid}: {texts[w]}") for pid, w in zip(ids, WHERE)]
        payloads = [{"where": w, "group": group0} for w in WHERE]
        q = f"Where is the evidence needed to answer the question, relative to the active interval {a}?"
        gq = {group0: q}
        if self.tree and len(self.path) >= 2:   # window tree: the smallest enclosing interval that has evidence (group 1)
            for k, w in enumerate(self.ancestors()):
                pid = f"Q{k + 1}"
                ids.append(pid)
                cands.append(nanojev_candidate(f"{pid}: {enclosing_name(k, w is self.full)} "
                                               f"[{_fmt_t(w.start)}-{_fmt_t(w.end)}]"))
                payloads.append({"enclosing": k + 1, "group": group0 + 1})
            ids.append("Q0")
            cands.append(nanojev_candidate("Q0: no evidence anywhere outside the active and the searched intervals"))
            payloads.append({"enclosing": 0, "group": group0 + 1})
            q2 = ("Which is the smallest interval above the active one that has evidence outside the active and the "
                  "searched intervals?")
            gq[group0 + 1] = q2
            q += " And w" + q2[1:-1] + "? (one answer to each)"
        return ids, cands, payloads, q, gq

    # ------------------------------------------------------------------ evidence and answer
    def request_frames(self, selected, restore=False):
        idx = sorted(selected, key=self.asset.time)
        if restore:
            self.n_restored += 1
            self.withdrawn_idx.difference_update(idx)
        else:
            self.attempts += 1
            self.frames_inspected += len(idx)
            self.insp_count[self.active.key()] = self.insp_count.get(self.active.key(), 0) + 1
        n = len(self.inspections) + 1
        for i in idx:
            if i not in self.evidence:
                self.evidence.append(i)
                self.pick_inspection[i] = n
        self.inspections.append((self.turn, idx))
        return None                                    # no reader call

    def reject_frames(self, idx):
        """budget.verifier_per_frame: take these frames of the last inspection back (they leave the evidence set and are
        never offered again); the inspection entry keeps only its kept frames (removed when none is left)."""
        idx = set(idx)
        if not idx:
            return []
        turn, last = self.inspections[-1]
        keep = [i for i in last if i not in idx]
        for i in idx:
            self.pick_inspection.pop(i, None)
            if i in self.evidence:
                self.evidence.remove(i)
        if keep:
            self.inspections[-1] = (turn, keep)
        else:
            self.inspections.pop()
        self.withdrawn_idx.update(idx)
        self.withdrawn.append({"turn": turn, "times": [self.asset.time(i) for i in sorted(idx, key=self.asset.time)]})
        return sorted(idx, key=self.asset.time)

    def withdraw_last_inspection(self):
        """budget.tentative: take the last inspection back (its frames leave the evidence set, the inspection is not
        counted); the window stays in `searched`."""
        turn, idx = self.inspections.pop()
        n = len(self.inspections) + 1
        for i in idx:
            if self.pick_inspection.get(i) == n:
                self.pick_inspection.pop(i, None)
                if i in self.evidence:
                    self.evidence.remove(i)
        self.withdrawn_idx.update(idx)
        self.withdrawn.append({"turn": turn, "times": [self.asset.time(i) for i in idx]})
        return idx

    def answer_request(self, reason, frames=None):
        """ReaderRequest of the final answer (fills the answer set); `frames` overrides (labelling)."""
        if frames is None:
            self.termination_reason = reason
            frames = self.filled_set()
            self.fill = [i for i in frames if i not in set(self.answer_set())]
            self.cost["reader_calls"] += 1
            self.cost["reader_frames"] = len(frames)
        return ReaderRequest("answer", self.key, frames, [self.asset.time(i) for i in frames], notes=[])

    def request_answer(self, reason):
        return self.answer_request(reason)

    def summary(self):
        s = super().summary()
        ans = sorted(self.answer_set() + self.fill, key=self.asset.time)
        s.update(answer_times=[self.asset.time(i) for i in ans],
                 overview_times=[self.asset.time(i) for i in self.global_idx],
                 answer_overview_times=[self.asset.time(i) for i in self.overview_in_answer()],
                 fill_times=[self.asset.time(i) for i in sorted(self.fill, key=self.asset.time)],
                 verifier_asked=sorted(self.verifier_asked), withdrawn=self.withdrawn,
                 attempts=self.attempts, frames_inspected=self.frames_inspected, n_restored=self.n_restored,
                 frames_seen=len(self.overview_in_answer()) + self.frames_inspected)
        s["cost"]["answer_frames"] = len(ans)
        return s
