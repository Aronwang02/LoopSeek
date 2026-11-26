"""Privileged teacher for SFT demonstrations (update.md §6 M1).

The teacher reads the frozen key-span annotation to pick actions; the states it visits are built
by the normal environment from public information only, so no label enters any decision input.
It also attaches the supervision set for each decision in `info["target_indices"]`:
  localize : legal windows with maximal IoU to the key spans (ties kept as a set);
  frame    : legal candidate frames inside a key span (for need='all', inside a span not yet
             covered by the frames already picked in this turn); empty = coverage miss, no label;
  controller: the teacher's action (rule below).
Controller rule (all from the annotation + public state):
  evidence sent to the reader already covers the need            -> ANSWER
  active window matches a key span (IoU >= 0.5 or contains one):
      frame candidates of the active window hit a span            -> INSPECT_FRAMES
      otherwise                                                   -> REFINE_LOCAL
  otherwise: full-video candidates already hit a span and nothing was inspected -> INSPECT_FRAMES
             else                                                 -> LOCALIZE_GLOBAL
mode='recovery' makes the first localization deliberately wrong (IoU 0) so the demonstrations
contain insufficient evidence followed by re-localization; that wrong choice is not a target.
"""
from ..evaluation.localization import evidence_hit, localization_iou


def _result(inp, idx, info):
    k = len(inp.candidates)
    p = [0.0] * k
    p[idx] = 1.0
    return {"index": idx, "log_prob": 0.0, "probs": p, "entropy": 0.0,
            "forced": len(inp.legal_indices()) == 1, "seconds": 0.0, "info": info}


class _Base:
    def __init__(self, annotations):
        self.ann = annotations

    def _ann(self, ep):
        return self.ann[ep.sample.sample_id]


class TeacherLocalizer(_Base):
    name = "teacher_localize"

    def __init__(self, annotations, mode="direct"):
        super().__init__(annotations)
        self.mode = mode

    def decide(self, inputs, episodes, generators):
        out = []
        for inp, ep in zip(inputs, episodes):
            ann = self._ann(ep)
            legal = inp.legal_indices()
            ious = {i: localization_iou((inp.payloads[i]["start"], inp.payloads[i]["end"]), ann) for i in legal}
            best = max(ious.values())
            targets = [i for i in legal if ious[i] >= best - 1e-9]
            if self.mode == "recovery" and not ep.searched and ep.turn == 0:
                wrong = [i for i in legal if ious[i] == 0.0]
                if wrong:
                    out.append(_result(inp, wrong[0], {"target_indices": [], "deliberately_wrong": True,
                                                       "ious": [ious.get(i, None) for i in range(len(inp.candidates))]}))
                    continue
            # among max-IoU windows prefer the shortest (most specific)
            idx = min(targets, key=lambda i: inp.payloads[i]["end"] - inp.payloads[i]["start"])
            out.append(_result(inp, idx, {"target_indices": targets, "best_iou": best,
                                          "ious": [ious.get(i, None) for i in range(len(inp.candidates))]}))
        return out


class TeacherFrameSelector(_Base):
    name = "teacher_frame"

    def decide(self, inputs, episodes, generators):
        out = []
        for inp, ep in zip(inputs, episodes):
            ann = self._ann(ep)
            legal = inp.legal_indices()
            times = {i: inp.payloads[i]["t"] for i in legal}
            chosen_t = [p["t"] for i, p in enumerate(inp.payloads) if not inp.valid[i]]
            spans = ann["key_spans"]
            if ann.get("need", "any") == "all":
                open_spans = [s for s in spans if not any(s[0] <= t <= s[1] for t in chosen_t)] or spans
            else:
                open_spans = spans
            targets = [i for i in legal if any(s[0] <= times[i] <= s[1] for s in open_spans)]
            if targets:
                c = [(s[0] + s[1]) / 2 for s in open_spans]
                idx = min(targets, key=lambda i: (min(abs(times[i] - x) for x in c), i))
            else:   # no candidate frame inside the needed spans: coverage miss, no supervision
                idx = legal[len(legal) // 2]
            out.append(_result(inp, idx, {"target_indices": targets, "coverage_miss": not targets}))
        return out


class TeacherController(_Base):
    """Every action is justified by what the public state shows, so the demos never teach a
    move whose reason is only visible in the hidden annotation:
      evidence already sent to the reader covers the need                 -> ANSWER
      active window chosen but not inspected yet                          -> INSPECT_FRAMES
        (direct mode: if it contains the event but its candidate frames miss it -> REFINE_LOCAL)
      nothing localized yet: full-video candidates hit the event and nothing inspected -> INSPECT_FRAMES
      inspected, evidence insufficient: window contains the event -> REFINE_LOCAL, else LOCALIZE_GLOBAL
    mode='recovery': the first action is LOCALIZE_GLOBAL (the localizer teacher then picks a wrong
    window), giving demonstrations of insufficient evidence followed by re-localization."""
    name = "teacher_controller"

    def __init__(self, annotations, mode="direct"):
        super().__init__(annotations)
        self.mode = mode

    def decide(self, inputs, episodes, generators):
        out = []
        for inp, ep in zip(inputs, episodes):
            ann = self._ann(ep)
            ev_times = [ep.asset.time(i) for i in ep.evidence]
            evidence_ok = bool(ev_times) and evidence_hit(ev_times, ann)
            full = ep.active_is_full()
            contains = (not full) and (localization_iou((ep.active.start, ep.active.end), ann) >= 0.5 or
                                       any(ep.active.start <= s and e <= ep.active.end for s, e in ann["key_spans"]))
            inspected_active = any(t >= ep.active_set_turn for t, _ in ep.inspections)
            cand_times = [ep.asset.time(i) for i in ep.asset.uniform_indices(
                ep.active.start, ep.active.end, ep.b["frame_candidates"])]
            cands_hit = any(s <= t <= e for t in cand_times for s, e in ann["key_spans"])
            if evidence_ok:
                a = "ANSWER"
            elif self.mode == "recovery" and ep.turn == 0:
                a = "LOCALIZE_GLOBAL"
            elif not full and not inspected_active:
                a = "REFINE_LOCAL" if (self.mode == "direct" and contains and not cands_hit) else "INSPECT_FRAMES"
            elif full and not ep.inspections:
                a = "INSPECT_FRAMES" if cands_hit else "LOCALIZE_GLOBAL"
            else:
                a = "REFINE_LOCAL" if contains else "LOCALIZE_GLOBAL"
            idx = inp.candidate_ids.index(a)
            out.append(_result(inp, idx, {"target_indices": [idx], "rule_state": {
                "evidence_ok": evidence_ok, "active_contains_event": contains,
                "active_inspected": inspected_active, "candidates_hit": cands_hit, "mode": self.mode}}))
        return out


# ---------------------------------------------------------------------------- new design (2026-09-30)
class TeacherFrameSelectorC(_Base):
    """Selection-C teacher: targets = legal candidates inside a key span; the pick order is sampled with the
    same rule as SFT (src/policy/selection.teacher_sequence: uniform among the targets available at each step,
    filler picks when none is available). info: picks, target_indices, supervised flags, coverage_miss."""
    name = "teacher_frame"

    def __init__(self, annotations, seed=0):
        super().__init__(annotations)
        import random
        self.rng = random.Random(seed)

    def decide(self, inputs, episodes, generators):
        from ..policy.selection import teacher_sequence
        out = []
        for inp, ep in zip(inputs, episodes):
            spans = self._ann(ep)["key_spans"]
            legal = inp.legal_indices()
            targets = [i for i in legal if any(s <= inp.payloads[i]["t"] <= e for s, e in spans)]
            picks, sup = teacher_sequence(inp.valid, targets, inp.select_k, inp.mask_radius, self.rng)
            if not picks:
                picks = legal[:inp.select_k]
            out.append(_result(inp, picks[0], {"picks": picks, "target_indices": targets, "supervised": sup,
                                               "coverage_miss": not targets}))
        return out


class EvidenceExpertController(_Base):
    """Window tree (default since 2026-10-06): see _tree_rule. The rules below are the flat flow (window_tree: false).
Evidence expert (the teacher is an expert on the evidence annotation, judged by process
    metrics). Goal: put as many kept frames as possible near the evidence (key spans widened by `pad` seconds; the
    CG-Bench upper bound: +15.5 points for 32 frames inside the clues, +3.5 for 4 swapped frames, +-15 s lost 1.5).
    Reads the annotation (only for `cover` = share of the active window covered by the evidence); the locator, frame
    selector and verifier stay the real models. Rules, in order, restricted to legal actions:
      1. active = full video                                  -> LOCALIZE_GLOBAL
      2. evidence does not overlap the active window          -> LOCALIZE_GLOBAL (searched windows are excluded)
      3. inspection results in this window (attempt_log, per-frame verifier):
         a. last attempt kept >= half of its frames            -> REFINE_LOCAL if cover < refine_frac, else INSPECT_FRAMES
         b. last attempt kept < half (or none), 1st such here  -> INSPECT_FRAMES (other candidates of the same window)
         c. 2nd attempt here keeping < half                    -> REFINE_LOCAL if cover < refine_frac (evidence inside but
                                                                  small), else LOCALIZE_GLOBAL
      4. no attempt here yet                                  -> REFINE_LOCAL if cover < refine_frac, else INSPECT_FRAMES
    Fallbacks: INSPECT_FRAMES, REFINE_LOCAL, LOCALIZE_GLOBAL in that order of the remaining legal actions.
Multi-segment evidence, only for rows marked evidence_multi_segment (scripts/filter_cgbench_vts.py --multi-segment, the
    VideoITG experiment of 2026-10-20): "evidence" in every rule is evidence.uncovered_spans, the segments that have not
    yet received their length-proportional share of the kept-frame cap, so the rules move on to the next segment. Other
    rows (all long-video questions) use every segment, as before."""
    name = "evidence_expert"

    def __init__(self, annotations, pad=3.0, refine_frac=0.25):
        super().__init__(annotations)
        self.pad, self.refine_frac = float(pad), float(refine_frac)

    def decide(self, inputs, episodes, generators):
        out = []
        for inp, ep in zip(inputs, episodes):
            legal = {inp.candidate_ids[i] for i in inp.legal_indices()}
            if not legal:
                out.append({"index": None, "log_prob": 0.0, "probs": [0.0] * len(inp.candidates), "entropy": 0.0,
                            "forced": True})
                continue
            from .evidence import padded_spans, uncovered_spans, window_cover
            ann = self._ann(ep)
            spans = padded_spans(ann["key_spans"], self.pad, ep.T)
            if ann.get("evidence_multi_segment"):            # VideoITG experiment (2026-10-20): rows the filter marked
                spans = uncovered_spans(spans, [ep.asset.time(i) for i in getattr(ep, "evidence", ())],
                                        getattr(ep, "frame_cap", 0))
            a0, a1 = ep.active.start, ep.active.end
            cover = window_cover(spans, a0, a1)               # union of the padded spans (no double counting)
            if getattr(ep, "tree", False):                    # D: evidence the inspection range can reach counts as here
                r = ep.inspect_range()
                cover = window_cover(spans, r.start, r.end)
            if getattr(ep, "tree", False):
                a, rule, info = self._tree_rule(ep, legal, spans, cover)
                idx = inp.candidate_ids.index(a)
                out.append(_result(inp, idx, {"target_indices": [idx], "rule": rule, "evidence_cover": cover, **info}))
                continue
            here = [x for x in getattr(ep, "attempt_log", []) if x.get("active_set_turn") == ep.active_set_turn]
            low = [x for x in here if 2 * x["kept"] < x["picked"]]
            narrow = "REFINE_LOCAL" if cover < self.refine_frac else "INSPECT_FRAMES"
            if ep.active_is_full():
                prefs, rule = ("LOCALIZE_GLOBAL",), "full_localize"
            elif cover <= 0.0:
                prefs, rule = ("LOCALIZE_GLOBAL",), "off_evidence_relocalize"
            elif here and 2 * here[-1]["kept"] >= here[-1]["picked"]:
                prefs, rule = (narrow,), "kept_half_" + ("refine" if narrow == "REFINE_LOCAL" else "inspect")
            elif len(low) == 1:
                prefs, rule = ("INSPECT_FRAMES",), "kept_little_retry"
            elif len(low) >= 2:
                prefs, rule = (("REFINE_LOCAL",), "kept_little_twice_refine") if cover < self.refine_frac else \
                    (("LOCALIZE_GLOBAL",), "kept_little_twice_relocalize")
            else:
                prefs, rule = (narrow,), "on_evidence_" + ("refine" if narrow == "REFINE_LOCAL" else "inspect")
            a = next(x for x in prefs + ("INSPECT_FRAMES", "REFINE_LOCAL", "LOCALIZE_GLOBAL") if x in legal)
            idx = inp.candidate_ids.index(a)
            out.append(_result(inp, idx, {"target_indices": [idx], "rule": rule, "evidence_cover": cover,
                                          "attempts_here": len(here), "fallback": a not in prefs}))
        return out


    def _tree_rule(self, ep, legal, spans, cover):
        """Strict window tree (budget.window_tree). "Reachable"
        windows are the unvisited ones the next action can enter (EpisodeV3.candidate_windows). Rules, in order:
          1. active = whole video (the root)                      -> REFINE_LOCAL (pick a level-1 window)
          2. evidence not in the active window ("move"): an unvisited sibling has evidence -> SWITCH_LOCAL; else (level
             >= 2) UP (the next turn decides again one level higher)
          3. inspection results in this window (per-frame verifier; counted over every visit of the window):
             a. last attempt kept >= half                         -> "narrow"
             b. 1st attempt here keeping < half                   -> INSPECT_FRAMES
             c. 2nd attempt here keeping < half                   -> REFINE_LOCAL if cover < refine_frac and an unvisited
                                                                     child has evidence; else REFINE_LOCAL if an unvisited
                                                                     child's own grid still has new evidence points and >= 2
                                                                     actions remain ("denser", b); else move
          4. no attempt here yet                                  -> "narrow"
          narrow = REFINE_LOCAL if cover < refine_frac and an unvisited child has evidence, else INSPECT_FRAMES
        Fallbacks: INSPECT_FRAMES, REFINE_LOCAL, SWITCH_LOCAL, UP in that order of the legal actions.
        dead_end: the active window has no evidence, no ancestor on the path below the root has evidence (UP + inspect),
        and no unvisited child of any window on the path (root .. active) has evidence."""
        from .evidence import overlap

        def ov(w):
            return overlap(spans, w.start, w.end)
        level = len(ep.path)
        sib_ev = level >= 1 and any(ov(w) > 0 for w in ep.candidate_windows("sibling"))
        kid_ev = any(ov(w) > 0 for w in ep.candidate_windows("local"))
        move, mrule = (("SWITCH_LOCAL", "UP"), "sibling") if sib_ev else (("UP", "SWITCH_LOCAL"), "up")
        win = (ep.active.start, ep.active.end)
        here = [x for x in getattr(ep, "attempt_log", []) if tuple(x["window"]) == win]
        low = [x for x in here if 2 * x["kept"] < x["picked"]]
        refine_ok = cover < self.refine_frac and kid_ev
        narrow = ("REFINE_LOCAL", "INSPECT_FRAMES") if refine_ok else ("INSPECT_FRAMES",)
        if ep.active_is_full():
            prefs, rule = ("REFINE_LOCAL",), "root_refine"
        elif cover <= 0.0:
            prefs, rule = move, "off_evidence_" + mrule
        elif here and 2 * here[-1]["kept"] >= here[-1]["picked"]:
            prefs, rule = narrow, "kept_half_" + ("refine" if refine_ok else "inspect")
        elif len(low) == 1:
            prefs, rule = ("INSPECT_FRAMES",), "kept_little_retry"
        elif len(low) >= 2:
            # (b) two low keeps here usually mean this window's grid has no evidence point left,
            # not that the evidence is absent; a finer grid (an unvisited child whose own grid still has new evidence
            # points) can add frames, if at least 2 actions remain (REFINE + INSPECT)
            left = int(ep.b.get("max_turns", 0)) - ep.turn
            denser = left >= 2 and any(ep.new_grid_points(w, spans) > 0 for w in ep.candidate_windows("local"))
            if refine_ok:
                prefs, rule = (("REFINE_LOCAL",) + move), "kept_little_twice_refine"
            elif denser:
                prefs, rule = (("REFINE_LOCAL",) + move), "kept_little_twice_denser"
            else:
                prefs, rule = move, "kept_little_twice_" + mrule
        else:
            prefs, rule = narrow, "on_evidence_" + ("refine" if refine_ok else "inspect")
        a = next(x for x in prefs + ("INSPECT_FRAMES", "REFINE_LOCAL", "SWITCH_LOCAL", "UP") if x in legal)
        seen = {u.key() for u in ep.searched}
        dead = (cover <= 0.0 and not any(ov(w) > 0 for w in ep.path[:-1])
                and not any(ov(w) > 0 and w.key() not in seen
                            for anc in [ep.full] + list(ep.path) for w in ep.gen.local_windows(anc, ep.T)
                            if w.kind == "quarter"))
        return a, rule, {"attempts_here": len(here), "level": level, "fallback": a not in prefs, "dead_end": dead}


class ExpertLabeledController:
    """DAgger collection (2026-10-06): the student controller (`inner`, e.g. a trained Jev) acts; the evidence expert
    labels every state the student reaches (info.target_indices = the expert's action, plus its rule / dead_end).
    beta > 0 executes the expert's action instead with probability beta (DAgger's mixing; 0 = the student always acts).
    scripts/gen_role_sft_from_trajectories.py turns the labelled controller decisions into SFT rows."""

    def __init__(self, inner, expert, beta=0.0, seed=0):
        import random
        self.inner, self.expert, self.beta = inner, expert, float(beta)
        self.name = f"{getattr(inner, 'name', 'student')}+expert_labels"
        self.policy = getattr(inner, "policy", None)          # static-prefix cache clearing (runner_v3.run)
        self.rng = random.Random(seed)

    def decide(self, inputs, episodes, generators):
        acts = self.inner.decide(inputs, episodes, generators)
        labs = self.expert.decide(inputs, episodes, generators)
        out = []
        for res, lab in zip(acts, labs):
            if res.get("index") is None or lab.get("index") is None:
                out.append(res)
                continue
            if self.beta > 0 and self.rng.random() < self.beta:
                res = dict(lab, info=dict(lab.get("info") or {}, executed="expert"))
            else:
                res = dict(res)
            li = lab.get("info") or {}
            res["info"] = dict(res.get("info") or {}, target_indices=li.get("target_indices", [lab["index"]]),
                               rule=li.get("rule"), dead_end=li.get("dead_end", False), fallback=li.get("fallback"),
                               label="expert", executed=(res.get("info") or {}).get("executed", "student"))
            out.append(res)
        return out


class TeacherControllerV2(_Base):
    """Controller teacher for the 3-action design (no ANSWER; the episode ends on the frame budget or the
    verifier). "On target" is RELATIVE, consistent with TeacherLocalizer (max-IoU window): the active window counts
    as found when its IoU with the key spans is >= the best IoU a new global localization could reach, or when it
    contains the event (IoU >= 0.5 or a span inside it). Rules, in order, restricted to legal actions:
      1. recovery mode, turn 0                              -> LOCALIZE_GLOBAL (the localizer teacher picks a wrong
                                                               window: an off-oracle state)
      2. INSPECT_FRAMES illegal                             -> REFINE if on target, else LOCALIZE_GLOBAL (else any legal)
      3. active = full video: the best global window is the full video, or its candidates hit a span and
         (no navigation move left or the spans cover >= 1/4 of the video)   -> INSPECT_FRAMES, else LOCALIZE_GLOBAL
      4. active window not inspected yet and not on target  -> INSPECT_FRAMES (the observation shows the evidence is
                                                               missing; a re-localization then follows from rule 6)
      5. on target: candidates hit a span                   -> INSPECT_FRAMES, else REFINE_LOCAL (else INSPECT_FRAMES)
      6. not on target (already inspected)                  -> LOCALIZE_GLOBAL (else REFINE_LOCAL, else INSPECT_FRAMES)
    (2026-09-30 fix: the earlier absolute rule "IoU >= 0.5" disagreed with the max-IoU localizer when the best window
    had IoU < 0.5, and the teacher re-localized to the same window until the navigation budget was spent.)"""
    name = "teacher_controller"

    def __init__(self, annotations, mode="direct"):
        super().__init__(annotations)
        self.mode = mode

    def decide(self, inputs, episodes, generators):
        out = []
        for inp, ep in zip(inputs, episodes):
            ann = self._ann(ep)
            spans = ann["key_spans"]
            legal = {inp.candidate_ids[i] for i in inp.legal_indices()}
            full = ep.active_is_full()
            gw = ep.localize_windows("global")
            g_iou = [localization_iou((w.start, w.end), ann) for w in gw]
            best_iou = max(g_iou) if g_iou else 0.0
            best_is_full = bool(gw) and max(range(len(gw)), key=lambda i: (g_iou[i], -(gw[i].end - gw[i].start))) \
                == next((i for i, w in enumerate(gw) if w.kind == "full"), -1)
            a_iou = localization_iou((ep.active.start, ep.active.end), ann)
            contains = (not full) and (a_iou >= 0.5 or any(ep.active.start <= s and e <= ep.active.end for s, e in spans))
            on_target = (not full) and (contains or a_iou >= best_iou - 1e-9)
            inspected_active = any(t >= ep.active_set_turn for t, _ in ep.inspections)
            cand_times = [ep.asset.time(i) for i in ep.asset.uniform_indices(
                ep.active.start, ep.active.end, ep.b["frame_candidates"])]
            cands_hit = any(s <= t <= e for t in cand_times for s, e in spans)
            span_frac = sum(min(e, ep.T) - max(s, 0.0) for s, e in spans) / max(ep.T, 1e-6)
            nav_left = ep.nav_left()
            no_nav = nav_left is not None and nav_left <= 0

            def first(*prefs):
                for a in prefs:
                    if a in legal:
                        return a
                return next(iter(legal))
            if self.mode == "recovery" and ep.turn == 0 and "LOCALIZE_GLOBAL" in legal:
                a, rule = "LOCALIZE_GLOBAL", "recovery_first_move"
            elif "INSPECT_FRAMES" not in legal:
                a, rule = (first("REFINE_LOCAL", "LOCALIZE_GLOBAL") if on_target
                           else first("LOCALIZE_GLOBAL", "REFINE_LOCAL")), "inspect_illegal"
            elif full:
                if best_is_full or (cands_hit and (no_nav or span_frac >= 0.25)):
                    a, rule = "INSPECT_FRAMES", "full_inspect"
                else:
                    a, rule = first("LOCALIZE_GLOBAL", "INSPECT_FRAMES"), "full_localize"
            elif not inspected_active and not on_target:
                a, rule = "INSPECT_FRAMES", "wrong_window_inspect_first"
            elif on_target:
                a, rule = (("INSPECT_FRAMES", "on_target_inspect") if cands_hit
                           else (first("REFINE_LOCAL", "INSPECT_FRAMES"), "on_target_refine"))
            else:
                a, rule = first("LOCALIZE_GLOBAL", "REFINE_LOCAL", "INSPECT_FRAMES"), "wrong_window_relocalize"
            idx = inp.candidate_ids.index(a)
            out.append(_result(inp, idx, {"target_indices": [idx], "rule": rule, "rule_state": {
                "active_full": full, "on_target": on_target, "active_iou": a_iou, "best_global_iou": best_iou,
                "best_global_is_full": best_is_full, "active_contains_event": contains, "candidates_hit": cands_hit,
                "active_inspected": inspected_active, "span_frac": span_frac, "nav_left": nav_left, "mode": self.mode}}))
        return out
