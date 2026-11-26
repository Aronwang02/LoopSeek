"""Batched rollouts of the design-B flow (src/env/episode_v3.py; 2026-10-01).

CURRENT DEFAULT (configs/default.yaml, 2026-10-06): strict window tree (REFINE_LOCAL / SWITCH_LOCAL / UP / INSPECT_FRAMES,
24 actions), per-frame verifier keep, the "where" question; configs/loopseek.yaml holds the inference settings of the
paper. The notes below document the earlier flat flow, still available with window_tree: false.

Per episode:  reset (overview) -> [verifier @overview] -> up to max_turns controller turns:
                LOCALIZE_GLOBAL / REFINE_LOCAL -> locator picks a window
                INSPECT_FRAMES -> selector picks k frames (selection C) -> evidence; after inspection 1:
                [verifier @inspection1]; after the last inspection: answer
              no legal action / turn cap -> answer.   Answer = answerer, once, on the filled 32-frame set, no notes.
Verifier: stop when P(T3) >= budget.verifier_tau (needs a 'verifier' decider; absent = never stop early).
Switch rule (budget.switch_tau > 0, default 0 = off; 2026-10-03 test, user): if the verifier's P(T3) after inspection 1 is
below switch_tau, the next controller action is replaced by LOCALIZE_GLOBAL over the windows that overlap every
searched window by < 50% of their own length (mostly new time; the full video stays allowed), and the
action after that by INSPECT_FRAMES (the second inspection happens in the new window). Logged as turns[].forced.
Tentative inspections (budget.tentative, default off; 2026-10-03 test, user): after every inspection the verifier is asked
again; the inspection is kept if its expected tier (sum_k k * P(Tk)) rose by >= budget.keep_delta over the last kept state
(the overview at first) or P(T3) >= budget.keep_p3. Otherwise its frames are taken back (not counted against the evidence
budget) and the next actions are forced to LOCALIZE_GLOBAL over mostly-new windows, then INSPECT_FRAMES. After
budget.max_attempts inspection attempts, or when no navigation move is left, the search ends: the inspections taken back
are a pool, and the best of them by expected tier fill the free inspection slots (best-of fallback, so a question on
which the verifier is pessimistic everywhere keeps its relatively best frames); then the episode answers. Logged as
turns[] entries with action TENTATIVE_CHECK / TENTATIVE_RESTORE.
Verifier in the loop: budget.tentative_force_switch=false removes the forced
switch/inspect (the controller navigates freely), budget.keep_delta=null keeps a batch iff P(T3) >= keep_p3, the episode ends
after max_attempts attempts, every termination back-fills free slots from the pool (best expected tier first, at most
budget.backfill_per_window batches per window), and budget.verifier_line=true shows the last verifier result to the controller
(EpisodeV3.ver_line). record_verifier_answers=True also records the answer that stopping would give at every tentative check.
Labelling (budget.label_pE): at every verifier point the answerer's option-letter probabilities on the frame set that
stopping there would give (answer set filled to 32) are recorded with the verifier DecisionInput (verifier SFT labels).
Calibration (record_verifier_answers=True): at every verifier point the verifier probabilities AND the answer that
stopping there would give are recorded, and the episode continues (tau calibration, scripts/calibrate_tau.py).
"""
import collections
import random
import time
import traceback

import torch

from ..env.episode_v3 import EpisodeV3, tier_probs
from .runner import RolloutRunner, Trajectory


class RunnerV3(RolloutRunner):
    def __init__(self, cfg, deciders, reader, features, encoder, text_len=None, log=print,
                 record_verifier_answers=False, answer_letter_probs=False, annotations=None):
        super().__init__(cfg, deciders, reader, features, encoder, text_len=text_len, log=log)
        self.annotations = annotations or {}                # sample_id -> {"key_spans": ...} (oracle frame verifier)
        self.record_verifier_answers = record_verifier_answers
        self.answer_letter_probs = answer_letter_probs     # RL: letter probabilities on the final answer frames

    # ------------------------------------------------------------------ helpers
    def _answer(self, trajs, reason):
        reqs = [t.ep.answer_request(reason) for t in trajs]
        outs = self._run_reader(trajs, reqs)
        for t, out in zip(trajs, outs):
            if out is not None:
                t.ep.receive_answer(out["text"], out["parsed"])
                t.answer_meta = out["meta"]
        if self.answer_letter_probs and trajs:
            missing = []
            for i, (t, out) in enumerate(zip(trajs, outs)):
                pr = out.get("letter_probs") if out is not None else None
                if pr is None:
                    missing.append(i)
                else:
                    t.final_letter_probs = pr
            if missing:                    # compatibility with readers without first-token logits
                selected = [trajs[i] for i in missing]
                t0 = time.perf_counter()
                self._free_cuda_cache()
                probs = self.reader.answer_letter_probs(self._reader_jobs(selected, [reqs[i] for i in missing]))
                seconds = (time.perf_counter() - t0) / len(selected)
                for t, pr in zip(selected, probs):
                    t.final_letter_probs = pr
                    t.seconds["reader"] += seconds

    def _letter_probs(self, trajs):
        """Answerer letter probabilities on the frame set that stopping now would give (one forward each)."""
        # budget.label_no_fill: the verifier sees the overview + the frames selected so far, so
        # its Q3 label is computed on exactly those frames (no uniform fill the verifier cannot see)
        frames = (lambda t: sorted(t.ep.answer_set(), key=t.ep.asset.time)) if self.cfg["budget"].get("label_no_fill") \
            else (lambda t: t.ep.filled_set())
        reqs = [t.ep.answer_request("label", frames=frames(t)) for t in trajs]
        t0 = time.perf_counter()
        jobs = self._reader_jobs(trajs, reqs)
        self._free_cuda_cache()
        probs = self.reader.answer_letter_probs(jobs)
        secs = (time.perf_counter() - t0) / max(len(trajs), 1)
        for t in trajs:
            t.seconds["reader"] += secs
        return probs, reqs

    def _overview_mode(self):
        b = self.cfg["budget"]
        return b.get("verifier_overview_question") == "places" or bool(b.get("pick_answer_k"))

    def _overview_phase(self, trajs, turn, policy_version):
        """The verifier at the overview point, in one grouped decision on the policy overview
        (EpisodeV3.verifier_overview_input):
          budget.verifier_overview_question = "places": one place / several places (replaces the overview Q3; no Q3 and
            no p_E label here); with budget.places_line every later controller state shows the answer
            ("Verifier after the overview: evidence at one place p, at several separate places q").
          budget.pick_answer_k = k: the keep question on every overview frame; the k frames with the highest P(yes) (ties:
            earlier first) become the answer set's overview frames (EpisodeV3.answer_pick). With exactly k overview
            frames all of them are taken and nothing is scored; with fewer the answer set keeps budget.answer_uniform.
        Logged as an OVERVIEW_CHECK turn (places probabilities, every scored frame's P(yes), the picked times; the input
        with budget.record_verifier_inputs, for the places / overview_frames SFT rows). Without a verifier decider (s1)
        the input is still recorded and the answer set keeps budget.answer_uniform."""
        b = self.cfg["budget"]
        places = b.get("verifier_overview_question") == "places"
        k = int(b["pick_answer_k"]) if b.get("pick_answer_k") else None
        apply = b.get("pick_answer_apply", True) is not False   # s1: score and record only, uniform answer grid
        # budget.pick_answer_final (conservative version): rank now, replace the uniform answer grid only in the final
        # answer (EpisodeV3.answer_request / final_pick); budget.pick_answer_grid: score a separate uniform grid
        final = bool(b.get("pick_answer_final"))
        grid = {id(t): t.ep.pick_grid() if k is not None else [] for t in trajs}
        score = {id(t): k is not None and len(grid[id(t)]) > k for t in trajs}
        for t in trajs:      # exactly k overview frames: all of them; fewer: the uniform grid
            if (k is not None and apply and not b.get("pick_answer_grid") and len(t.ep.global_idx) == k
                    and "verifier" in self.deciders):
                t.ep.answer_pick = sorted(t.ep.global_idx, key=t.ep.asset.time)
        ask = [t for t in trajs if places or score[id(t)]]
        if not ask:
            return
        ask, inputs = self._inputs(ask, lambda t: t.ep.verifier_overview_input(self.text_len, places=places,
                                                                                 frames=score[id(t)]))
        recs = (self._decide("verifier", ask, inputs, [None] * len(ask), turn, policy_version)
                if "verifier" in self.deciders else [None] * len(ask))
        for t, inp, rec in zip(ask, inputs, recs):
            entry = {"turn_id": turn, "action": "OVERVIEW_CHECK", "places": None, "frames": None, "picked_times": None}
            if rec is not None:
                pr = [float(p) for p in rec.old_probs]
                if places:
                    t.ep.places_probs = {pl["places"]: p for pl, p in zip(inp.payloads, pr) if "places" in pl}
                    entry["places"] = t.ep.places_probs
                    if b.get("places_line"):
                        t.ep.places_line = (f"Verifier after the overview: evidence at one place "
                                            f"{t.ep.places_probs['one']:.2f}, at several separate places "
                                            f"{t.ep.places_probs['several']:.2f}")
                if score[id(t)]:
                    yes = [(pl["frame"], p) for pl, p in zip(inp.payloads, pr) if pl.get("keep")]
                    order = sorted(yes, key=lambda x: (-x[1], t.ep.asset.time(x[0])))
                    ranked = order[:k]
                    if apply and final:
                        t.ep.answer_rank = [i for i, _ in order]
                    elif apply:
                        t.ep.answer_pick = sorted((i for i, _ in ranked), key=t.ep.asset.time)
                    entry["frames"] = [{"time": t.ep.asset.time(i), "p_yes": p} for i, p in yes]
                    entry["would_pick_times"] = sorted(t.ep.asset.time(i) for i, _ in ranked)
            if t.ep.answer_pick is not None:
                entry["picked_times"] = [t.ep.asset.time(i) for i in t.ep.answer_pick]
            if b.get("record_verifier_inputs"):
                entry["input"] = inp.to_json()
            t.turns.append(entry)

    def _verifier_phase(self, live, turn, policy_version):
        """Verifier points reached now. Returns the trajectories that stop."""
        b = self.cfg["budget"]
        pts = {id(t): t.ep.verifier_point() for t in live if t.status == "ok" and not t.ep.done}
        sel = [t for t in live if pts.get(id(t))]
        if not sel:
            return []
        for t in sel:
            t.ep.verifier_asked.add(pts[id(t)])
        if self._overview_mode():                     # 2026-11-03: the overview point asks places / scores the frames
            ov = [t for t in sel if pts[id(t)] == "overview"]
            if ov:
                self._overview_phase(ov, turn, policy_version)
                # budget.pick_answer_with_q3 (conservative version): the overview Q3 is still asked after the scoring
                keep_ov = bool(b.get("pick_answer_with_q3"))
                sel = [t for t in sel if (keep_ov or pts[id(t)] != "overview") and t.status == "ok"]
                if not sel:
                    return []
        if not b.get("verifier_q3", False):          # budget.verifier_q3 off: no Q3 anywhere, nor its labels
            return []
        rec_turn = {"turn_id": turn, "action": "VERIFIER_POINT"}
        if b.get("label_pE") and hasattr(self.reader, "answer_letter_probs"):
            try:
                probs, _ = self._letter_probs(sel)
                for t, pr in zip(sel, probs):
                    t.turns.append(dict(rec_turn, point=pts[id(t)], letter_probs=pr,
                                        verifier_input=t.ep.verifier_input(self.text_len).to_json(),
                                        stoppable_prefix={"inspections": len(t.ep.inspections),
                                                          "selected_frames": len(t.ep.evidence)}))
            except Exception as exc:     # labelling must never kill a rollout
                self.log(f"letter-prob labelling failed: {exc!r}")
        if "verifier" not in self.deciders:
            return []
        cur, inputs = self._inputs(sel, lambda t: t.ep.verifier_input(self.text_len))
        recs = self._decide("verifier", cur, inputs, [None] * len(cur), turn, policy_version)
        tau = float(b.get("verifier_tau", 0.5))
        stop = []
        for t, inp, rec in zip(cur, inputs, recs):
            if rec is None:
                continue
            pr = tier_probs(inp, rec.old_probs)              # tier order (budget.verifier_neutral shuffles options)
            p3 = pr[3]
            entry = {"point": pts[id(t)], "decision": rec.decision_id, "p_tier3": p3, "probs": pr,
                     "stop": p3 >= tau}
            t.verifier_log.append(entry)
            if b.get("verifier_line") and pts[id(t)] == "overview":
                t.ep.ver_line = self._ver_line(t, None, pr, None, None)
            if self.record_verifier_answers:
                entry["stop"] = False
            elif p3 >= tau:
                stop.append(t)
        if self.record_verifier_answers and cur:
            reqs = [t.ep.answer_request("calibration", frames=t.ep.filled_set()) for t in cur]
            outs = self._run_reader(cur, reqs)
            for t, out in zip(cur, outs):
                if out is not None:
                    t.verifier_log[-1]["answer_if_stop"] = out["parsed"]
                    t.verifier_log[-1]["answer_frames_if_stop"] = len(reqs[cur.index(t)].frame_indices)
        return stop

    def _ver_line(self, t, attempt, probs, kept, ref_et):
        """State line for the controller (budget.verifier_line): the last verifier result."""
        from ..env.environment import _fmt_t
        ep, et, p3 = t.ep, sum(k * p for k, p in enumerate(probs)), probs[3]
        left = "" if ep.max_attempts is None else f", inspection attempts left {ep.max_attempts - ep.attempts}"
        kept_n = (f"kept {len(ep.evidence)} of {ep.frame_cap} frames" if getattr(ep, "per_frame", False)
                  else f"kept {len(ep.inspections)} of {ep.max_inspections}")
        if self.cfg["budget"].get("verifier_line_full"):   # rec 4: the full tier distribution, ranges spelled out
            c = self.cfg["budget"].get("verifier_cuts", [0.25, 0.5, 0.8])
            names = (f"<{c[0]:.0%}", f"{c[0]:.0%}-{c[1]:.0%}", f"{c[1]:.0%}-{c[2]:.0%}", f">={c[2]:.0%}")
            score = ("answerer's probability for the correct option: "
                     + ", ".join(f"{n} {p:.2f}" for n, p in zip(names, probs)) + f" (expected tier {et:.2f}")
        else:
            score = f"top-tier prob {p3:.2f}, expected tier {et:.2f}"
        if attempt is None:
            return (f"Verifier after the overview: {score}" + (")" if self.cfg["budget"].get("verifier_line_full") else "")
                    + f"; {kept_n}{left}")
        w = ep.active
        tail = f"last kept {ref_et:.2f}, change {et - ref_et:+.2f}"
        score = f"{score}, {tail})" if self.cfg["budget"].get("verifier_line_full") else f"{score} ({tail})"
        return (f"Verifier after inspection attempt {attempt} (window {_fmt_t(w.start)}-{_fmt_t(w.end)}, "
                f"{'kept' if kept else 'withdrawn'}): {score}; {kept_n}{left}")

    def _windows(self, t, kind):
        """Candidate windows for LOCALIZE_GLOBAL / REFINE_LOCAL. budget.relocalize_exclude_searched (2026-10-06): global
        windows that overlap any searched window by >= 50% of their own length are dropped (the full video stays; if
        nothing is left, all windows are offered), so a re-localization cannot return to a window already searched."""
        return t.ep.candidate_windows(kind)      # exclusion rules: EpisodeV3.candidate_windows

    def _locator_probe(self, live, turn):
        """budget.locator_line (2026-10-05): before every controller decision the locator scores the windows of the
        whole video and, if the active interval is not the full video, the sub-windows of the active interval -- the
        same inputs a LOCALIZE_GLOBAL / REFINE_LOCAL would give it. The controller sees the top-n windows with their
        probabilities and, for the best window, whether it was searched and how many kept / withdrawn frames lie in it.
        The probe records no decision (nothing is trained on it) and uses its own generators, so the trajectory's
        random stream is unchanged."""
        from ..env.environment import _fmt_t
        top_n = int(self.cfg["budget"].get("locator_line_top", 3))
        for t in live:
            t.ep.loc_lines, t.ep.loc_info = [], {}
        for kind, title in (("global", "whole video"), ("local", "inside the active interval")):
            sel = [t for t in live if t.status == "ok" and not t.ep.done and (kind == "global" or not t.ep.active_is_full())]
            wins = {id(t): self._windows(t, kind) for t in sel}
            sel = [t for t in sel if wins[id(t)]]
            if not sel:
                continue
            sel, inputs = self._inputs(sel, lambda t: t.ep.localize_input(kind, wins[id(t)], self.text_len))
            res = self.deciders["localize"].decide(inputs, [t.ep for t in sel],
                                                   [torch.Generator().manual_seed(0) for _ in sel])
            for t, inp, r in zip(sel, inputs, res):
                if r.get("index") is None:
                    continue
                ep, W = t.ep, wins[id(t)]
                P = [float(p) for p in r["probs"][:len(W)]]
                order = sorted(range(len(W)), key=lambda i: -P[i])
                best = W[order[0]]
                inside = lambda idx: sum(best.start <= ep.asset.time(i) <= best.end for i in idx)
                searched = any(abs(best.start - u.start) < 1e-6 and abs(best.end - u.end) < 1e-6 for u in ep.searched)
                head = " | ".join(f"{inp.candidate_ids[i]} {_fmt_t(W[i].start)}-{_fmt_t(W[i].end)} ({W[i].kind}) {P[i]:.2f}"
                                  for i in order[:top_n])
                rest = sum(P[i] for i in order[top_n:])
                ep.loc_lines.append(
                    f"Locator, {title}: {head}" + (f" | rest {rest:.2f}" if len(W) > top_n else "")
                    + f"; best window {'searched' if searched else 'not searched'}, kept frames in it "
                      f"{inside(ep.evidence)}, withdrawn frames in it {inside(ep.withdrawn_idx)}")
                margin = P[order[0]] - (P[order[1]] if len(W) > 1 else 0.0)
                ep.loc_info[kind] = {"windows": [w.to_json() for w in W], "probs": P, "best": order[0],
                                     "margin": margin, "best_searched": searched}
                t.locator_log = getattr(t, "locator_log", [])
                t.locator_log.append({"turn": turn, "kind": kind, "probs": P, "best": order[0], "margin": margin,
                                      "best_searched": searched})

    def _tentative(self, cur, turn, policy_version):
        b = self.cfg["budget"]
        sel = [t for t in cur if t.status == "ok" and not t.ep.done and t.turns and t.turns[-1].get("frame_times")]
        if not sel:
            return
        sel, inputs = self._inputs(sel, lambda t: t.ep.verifier_input(self.text_len))
        recs = self._decide("verifier", sel, inputs, [None] * len(sel), turn, policy_version)
        if b.get("label_pE") and hasattr(self.reader, "answer_letter_probs") and sel:
            try:                                         # verifier labels at every tentative check (16-frame retrain)
                probs, _ = self._letter_probs(sel)
                for t, inp, pr in zip(sel, inputs, probs):     # the verifier input this check actually used
                    t.tent_label = {"letter_probs": pr, "verifier_input": inp.to_json(),
                                    "stoppable_prefix": {"inspections": len(t.ep.inspections),
                                                         "selected_frames": len(t.ep.evidence), "attempt": t.ep.attempts}}
            except Exception as exc:     # labelling must never kill a rollout
                self.log(f"letter-prob labelling failed: {exc!r}")
        if self.record_verifier_answers and sel:         # answer that stopping now (this batch kept) would give
            reqs = [t.ep.answer_request("calibration", frames=t.ep.filled_set()) for t in sel]
            outs = self._run_reader(sel, reqs)
            for t, out in zip(sel, outs):
                t.tent_answer = out["parsed"] if out is not None else None
        force = bool(b.get("tentative_force_switch", True))
        kd = b.get("keep_delta")
        give_up = []
        for t, inp, rec in zip(sel, inputs, recs):
            if rec is None:
                continue
            pr = tier_probs(inp, list(rec.old_probs)); et, p3 = sum(k * p for k, p in enumerate(pr)), pr[3]
            if not hasattr(t, "tent_ref"):
                ov = [e for e in t.verifier_log if e["point"] == "overview"]
                t.tent_ref = (sum(k * p for k, p in enumerate(ov[0]["probs"])), ov[0]["probs"][3]) if ov else (0.0, 0.0)
                t.tent_attempts = 0
            t.tent_attempts += 1
            keep = (kd is not None and et - t.tent_ref[0] >= float(kd)) or (p3 >= float(b.get("keep_p3", 0.5)))
            kp = b.get("tentative_keep_prob")
            if kp is not None:   # verifier relabelling (2026-10-04): keep at random, the verifier's scores stay real
                if not hasattr(t, "keep_rng"):
                    t.keep_rng = random.Random(f"{t.meta['trajectory_id']}:keep_prob")
                keep = t.keep_rng.random() < float(kp)
            rk = b.get("tentative_random_keep") or []
            if rk:   # ablation C-random (2026-10-04): keep with C's per-attempt keep rate, best-of by a random score
                if not hasattr(t, "tent_rng"):
                    t.tent_rng = random.Random(f"{t.meta['trajectory_id']}:tentative_random")
                keep = t.tent_rng.random() < float(rk[min(t.tent_attempts, len(rk)) - 1])
                et = t.tent_rng.random()
            w = t.ep.active
            entry = {"turn_id": turn, "action": "TENTATIVE_CHECK", "probs": pr, "et": et, "p_tier3": p3,
                     "et_ref": t.tent_ref[0], "attempt": t.tent_attempts, "kept": keep,
                     "window": [w.start, w.end], "kept_before": len(t.ep.inspections) - 1}
            if self.record_verifier_answers:
                entry["answer_if_stop"] = getattr(t, "tent_answer", None)
            if getattr(t, "tent_label", None) is not None:
                entry.update(t.tent_label)               # read by scripts/gen_role_sft_from_trajectories.py
                t.tent_label = None
            t.turns.append(entry)
            if b.get("verifier_line"):
                t.ep.ver_line = None             # set below, after the keep / withdraw bookkeeping
            ref_et = t.tent_ref[0]
            if keep:
                t.tent_ref = (et, p3)
            else:
                idx = t.ep.withdraw_last_inspection()
                t.tent_pool = getattr(t, "tent_pool", []) + [{"et": et, "idx": idx, "window": (w.start, w.end),
                                                               "kept_before": entry["kept_before"]}]
            if b.get("verifier_line"):
                t.ep.ver_line = self._ver_line(t, t.tent_attempts, pr, keep, ref_et)
            if force:                                    # old C rule: forced switch after a withdrawal
                if keep:
                    continue
                can_switch = bool(t.ep.controller_mask().get("LOCALIZE_GLOBAL"))
                if t.tent_attempts < int(b.get("max_attempts", 4)) and can_switch:
                    t.force_switch = True
                else:
                    give_up.append(t)
            elif t.ep.attempts_done() and not t.ep.frame_budget_done():
                give_up.append(t)                        # attempt cap reached: stop now (back-fill, answer)
        if give_up:
            self._finalize(give_up, "tentative_exhausted")

    def _tentative_frames(self, cur, turn, policy_version):
        """budget.verifier_per_frame: after an inspection the verifier judges every picked frame
        ("does it show evidence needed to answer?"); frames with P(yes) >= budget.keep_frame_tau stay (highest first,
        up to the kept-frame budget), the others go to a frame pool (never offered again; back-fill at the end). Then the
        verifier's Q3 on the kept set gives the controller line. The attempt cap ends the episode as before."""
        from ..env.environment import PathTooLong
        b = self.cfg["budget"]
        tau = float(b.get("keep_frame_tau", 0.5))
        sel = [t for t in cur if t.status == "ok" and not t.ep.done and t.turns and t.turns[-1].get("frame_times")
               and t.ep.inspections]
        if not sel:
            return
        jobs, inputs = [], []
        joint = bool(b.get("verifier_frame_joint"))
        # budget.verifier_keep_where_joint (2026-11-02 item 18): below the root the keep groups and the where groups are one
        # decision; the where answer (about the kept frames before this keep) is applied after the keep below
        kw = bool(joint and b.get("verifier_keep_where_joint") and b.get("verifier_where"))
        if joint:              # one grouped decision per batch (one prefix forward); rows (t, frame) share the input
            for t in sel:
                try:
                    inp = (t.ep.verifier_frames_where_input(t.ep.inspections[-1][1], self.text_len)
                           if kw and not t.ep.active_is_full() else
                           t.ep.verifier_frames_input(t.ep.inspections[-1][1], self.text_len))
                except PathTooLong as exc:
                    t.status, t.error = "path_too_long", str(exc)
                    continue
                for pl in inp.payloads:
                    if pl.get("keep"):            # keep groups only (item 18: the where groups follow them)
                        jobs.append((t, pl["frame"]))
                        inputs.append(inp)
        else:
            for t in sel:
                for i in t.ep.inspections[-1][1]:
                    try:
                        inputs.append(t.ep.verifier_frame_input(i, self.text_len))
                        jobs.append((t, i))
                    except PathTooLong as exc:
                        t.status, t.error = "path_too_long", str(exc)
        pad = b.get("verifier_frame_oracle_pad")
        per = collections.defaultdict(list)
        joint_rec = {}          # item 18: id(t) -> (joint input, decision record) of the joint keep (+ where) decision
        if pad is not None:     # 2026-10-06 oracle per-frame verifier (judges the controller teacher before the verifier
            for (t, i), inp in zip(jobs, inputs):   # is trained): keep = frame within key_spans +- pad seconds
                if t.status != "ok":
                    continue
                tm = t.ep.asset.time(i)
                spans = self.annotations[t.ep.sample.sample_id]["key_spans"]
                per[id(t)].append((i, float(any(s - pad <= tm <= e + pad for s, e in spans)), inp))
        elif joint:
            ts = list(dict.fromkeys(t for t, _ in jobs))
            uniq = [inputs[[t for t, _ in jobs].index(t)] for t in ts]
            recs = self._decide("verifier", ts, uniq, [None] * len(ts), turn, policy_version)
            for t, inp, rec in zip(ts, uniq, recs):
                if rec is None or t.status != "ok":
                    continue
                joint_rec[id(t)] = (inp, rec)
                for k, pl in enumerate(inp.payloads):
                    if pl.get("keep"):
                        per[id(t)].append((pl["frame"], float(rec.old_probs[k]), inp))
        else:
            recs = self._decide("verifier", [t for t, _ in jobs], inputs, [None] * len(jobs), turn, policy_version)
            for (t, i), inp, rec in zip(jobs, inputs, recs):
                if rec is None or t.status != "ok":
                    continue
                p_yes = rec.old_probs[next(k for k, pl in enumerate(inp.payloads) if pl["keep"])]
                per[id(t)].append((i, float(p_yes), inp))
        live = [t for t in sel if t.status == "ok" and per.get(id(t))]
        # budget.keep_gate_opening = g (off = null): a pick stays only if the opening-format question
        # (EpisodeV3.opening_gate_input: the picks inserted into the overview, initial-state memory) also gives P(yes) >= g;
        # CG test 11-04 (scripts/rescore_picks_overview.py): frame AUC 0.738 -> 0.819, g 0.3 keeps of wrong windows 82.5 ->
        # 61.2% at right windows' on-clue keeps 79.0 -> 77.4%
        gate_g, gate = b.get("keep_gate_opening"), {}
        if gate_g is not None and pad is None and live and "verifier" in self.deciders:
            gl, ginp = self._inputs(live, lambda t: t.ep.opening_gate_input(
                [x[0] for x in per[id(t)]], self.text_len, all_groups=b.get("keep_gate_groups") == "all"))
            grec = self._decide("verifier", gl, ginp, [None] * len(gl), turn, policy_version) if gl else []
            for t, inp, rec in zip(gl, ginp, grec):
                if rec is not None:
                    gate[id(t)] = {int(pl["frame"]): float(p) for pl, p in zip(inp.payloads, rec.old_probs)
                                   if pl.get("keep")}
            live = [t for t in live if t.status == "ok"]
        give_up = []
        for t in live:
            ep, w = t.ep, t.ep.active
            batch = per[id(t)]
            before = [i for i in ep.evidence if i not in {x[0] for x in batch}]   # kept before this inspection
            room = ep.frame_cap - (len(ep.evidence) - len(batch))
            ranked = sorted(batch, key=lambda x: -x[1])               # highest P(keep) first, within the budget
            g_open = gate.get(id(t))
            keep = {i for i, p, _ in ranked if p >= tau and (g_open is None or g_open.get(int(i), 1.0) >= float(gate_g))}
            keep = {i for i, _, _ in [x for x in ranked if x[0] in keep][:max(room, 0)]}
            forced_keep = self._forced_keep(ep, w)
            if forced_keep:      # an injected wrong-window inspection keeps its
                keep = {i for i, _, _ in ranked[:max(room, 0)]}   # picks whatever P(keep) (keep labels stay "no")
            rejected = [i for i, _, _ in batch if i not in keep]
            ep.reject_frames(rejected)
            t.tent_pool = getattr(t, "tent_pool", []) + [{"p": p, "idx": [i], "window": (w.start, w.end)}
                                                         for i, p, _ in batch if i not in keep]
            ep.attempt_log.append({"turn": turn, "window": (w.start, w.end), "active_set_turn": ep.active_set_turn,
                                   "picked": len(batch), "kept": len(keep)})
            entry = {"turn_id": turn, "action": "TENTATIVE_CHECK", "attempt": ep.attempts, "window": [w.start, w.end],
                     "frames": [{"time": ep.asset.time(i), "p_keep": p, "kept": i in keep,
                                 **({"p_open": g_open.get(int(i))} if g_open is not None else {})} for i, p, _ in batch],
                     "n_kept": len(keep), "kept": bool(keep), "kept_total": len(ep.evidence)}
            if forced_keep:
                entry["forced_keep"] = True
            if b.get("record_verifier_inputs"):  # verifier_frame SFT rows (labels from key_spans)
                if joint:                         # one grouped input for the batch
                    entry["frames_input"] = batch[0][2].to_json()
                else:
                    entry["frame_inputs"] = [{"time": ep.asset.time(i), "input": inp.to_json()} for i, _, inp in batch]
            t.turns.append(entry)
            jr = joint_rec.get(id(t))                 # item 18: the where groups of the joint decision
            if jr is not None and any("where" in pl for pl in jr[0].payloads):
                allp = list(jr[1].old_probs)
                if b.get("verifier_where_oracle_pad") is not None:
                    allp = self._where_oracle(t, jr[0], allp)
                # labels: the kept frames before this keep + the picks that show evidence,
                # i.e. where the evidence still to cover is once this keep is right (gen_role_sft_from_trajectories.py)
                self._apply_where(t, jr[0], allp, turn, kept_idx=before,
                                  picked=[x[0] for x in batch])            # WHERE_CHECK after this check
        # Q3 on the kept set: the controller line (and verifier Q3 labels with budget.label_pE)
        q3 = [t for t in live if t.status == "ok"]
        if not b.get("verifier_q3", False):          # budget.verifier_q3 off: no Q3 after an inspection; the
            for t in q3:                             # controller line keeps "kept m of k frames", no answerer labels
                if b.get("verifier_line"):
                    e = next(x for x in reversed(t.turns) if x.get("action") == "TENTATIVE_CHECK")
                    t.ep.ver_line = self._ver_line_keep(e)
            q3 = []
        if b.get("skip_final_q3"):    # the kept-frame budget is full, the episode
            q3 = [t for t in q3 if not t.ep.frame_budget_done()]   # answers now: nothing reads this Q3 (nor its label)
        # budget.verifier_q3_where_joint: the next turn's "where" question has the same prefix as Q3
        # here, so both are asked in one grouped decision (one forward); the where phase then has nothing left to ask
        joint_where = bool(b.get("verifier_q3_where_joint") and b.get("verifier_where"))
        if q3 and "verifier" not in self.deciders:   # oracle per-frame keep without a verifier model: no Q3 line
            q3 = []
        if q3:
            q3, qin = self._inputs(q3, lambda t: t.ep.verifier_input(self.text_len, with_where=joint_where))
            qrec = self._decide("verifier", q3, qin, [None] * len(q3), turn, policy_version)
            lab = {}
            if b.get("label_pE") and hasattr(self.reader, "answer_letter_probs") and q3:
                try:
                    probs, _ = self._letter_probs(q3)
                    lab = {id(t): pr for t, pr in zip(q3, probs)}
                except Exception as exc:     # labelling must never kill a rollout
                    self.log(f"letter-prob labelling failed: {exc!r}")
            for t, inp, rec in zip(q3, qin, qrec):
                # this check's entry (item 18 may have logged its WHERE_CHECK after it)
                entry = next(x for x in reversed(t.turns) if x.get("action") == "TENTATIVE_CHECK")
                if rec is not None:
                    allp = list(rec.old_probs)
                    if any("where" in pl for pl in inp.payloads):
                        if b.get("verifier_where_oracle_pad") is not None:
                            allp = self._where_oracle(t, inp, allp)
                        self._apply_where(t, inp, allp, turn)        # appends WHERE_CHECK after this check
                    pr = tier_probs(inp, allp)
                    entry.update(probs=pr, p_tier3=pr[3], et=sum(k * p for k, p in enumerate(pr)))
                    if b.get("verifier_line"):
                        t.ep.ver_line = self._ver_line_frames(t, entry, pr)
                if id(t) in lab:
                    entry.update(letter_probs=lab[id(t)], verifier_input=inp.to_json(),
                                 stoppable_prefix={"kept_frames": len(t.ep.evidence), "attempt": t.ep.attempts})
                if t.ep.attempts_done() and not t.ep.frame_budget_done():
                    give_up.append(t)
        if give_up:
            self._finalize(give_up, "tentative_exhausted")

    def _where_phase(self, live, turn, policy_version):
        """budget.verifier_where (2026-10-06): before the controller decides, the verifier says where the evidence is
        relative to the active interval (EpisodeV3.verifier_where_input). Asked again only when the active interval, the
        kept frames or the attempt count changed; not asked while the active interval is the full video.
        budget.verifier_where_oracle_pad: answer from key_spans +- pad instead (one-hot; no decision is recorded).
        Logged as turns[] entries WHERE_CHECK (with the input when budget.record_verifier_inputs)."""
        b = self.cfg["budget"]
        state = self._where_state
        need = []
        for t in live:
            if t.status != "ok" or t.ep.done:
                continue
            if t.ep.active_is_full():
                t.ep.where_probs = t.ep.where_window = None
            elif getattr(t, "where_state", None) != state(t):
                need.append(t)
        if not need:
            return
        pad = b.get("verifier_where_oracle_pad")
        if pad is None and "verifier" not in self.deciders:
            return
        need, inputs = self._inputs(need, lambda t: t.ep.verifier_where_input(self.text_len))
        if pad is not None:
            probs = [self._where_oracle(t, inp, [0.0] * len(inp.payloads)) for t, inp in zip(need, inputs)]
        else:
            recs = self._decide("verifier", need, inputs, [None] * len(need), turn, policy_version)
            probs = [None if r is None else list(r.old_probs) for r in recs]
        for t, inp, pr in zip(need, inputs, probs):
            if pr is not None:
                self._apply_where(t, inp, pr, turn)

    def _where_state(self, t):
        return (t.ep.active.key(), tuple(t.ep.evidence), t.ep.attempts)

    def _where_oracle(self, t, inp, probs):
        """budget.verifier_where_oracle_pad: replace the probabilities of the where / enclosing candidates by the
        one-hot answer from key_spans +- pad (other candidates, e.g. Q3 in a joint input, are kept). For rows marked
        evidence_multi_segment (the VideoITG experiment, 2026-10-20) only the segments not yet covered by the kept frames
        count (evidence.uncovered_spans)."""
        from ..baselines.evidence import enclosing_label, padded_spans, uncovered_spans, where_label
        b = self.cfg["budget"]
        ann = self.annotations[t.ep.sample.sample_id]
        sp = padded_spans(ann["key_spans"], float(b["verifier_where_oracle_pad"]), t.ep.T)
        if ann.get("evidence_multi_segment"):
            sp = uncovered_spans(sp, [t.ep.asset.time(i) for i in t.ep.evidence], t.ep.frame_cap,
                                 b.get("share_cap_per_place"))
        r = t.ep.inspect_range()                       # D: "inside" = within the inspection range (window + margins)
        lab = where_label(sp, r.start, r.end, float(b.get("evidence_refine_frac", 0.25)))
        ep = t.ep
        enc = enclosing_label(sp, (ep.active.start, ep.active.end), [(w.start, w.end) for w in ep.ancestors()],
                              [(w.start, w.end) for w in ep.searched])
        return [float(pl["where"] == lab) if "where" in pl else float(pl["enclosing"] == enc) if "enclosing" in pl
                else p for pl, p in zip(inp.payloads, probs)]

    def _apply_where(self, t, inp, pr, turn, kept_idx=None, picked=None):
        """Store the verifier's where / enclosing answer (controller line, SWITCH locator line) and log WHERE_CHECK.
        kept_idx / picked (budget.verifier_keep_where_joint): the frames kept before this inspection's keep and its
        picks, recorded as kept_times / picked_times; the multi-segment labels count the picks that show evidence as
        kept (the answer is about the state after a correct keep). None = the current kept frames, no picks."""
        b = self.cfg["budget"]
        if True:
            ep = t.ep
            ep.where_probs = {pl["where"]: float(p) for pl, p in zip(inp.payloads, pr) if "where" in pl}
            from ..env.episode_v3 import enclosing_name
            anc = ep.ancestors()
            enc = [(enclosing_name(pl["enclosing"] - 1, pl["enclosing"] > 0 and anc[pl["enclosing"] - 1] is ep.full),
                    float(p)) for pl, p in zip(inp.payloads, pr) if "enclosing" in pl]
            if enc:
                ep.where_probs["enclosing"] = enc
            ep.where_window = (ep.active.start, ep.active.end)
            t.where_state = self._where_state(t)
            anc = ep.ancestors() if getattr(ep, "tree", False) and len(ep.path) >= 2 else []
            rng = ep.inspect_range()
            entry = {"turn_id": turn, "action": "WHERE_CHECK", "window": [ep.active.start, ep.active.end],
                     "range": [rng.start, rng.end], "inspections_here": ep.inspections_here(),   # 0 = before any inspection
                     "ancestors": [[w.start, w.end] for w in anc], "visited": [[w.start, w.end] for w in ep.searched],
                     "probs": ep.where_probs,
                     # kept frames so far: the where / enclosing labels of evidence_multi_segment rows count only the
                     # evidence segments these do not cover yet (scripts/gen_role_sft_from_trajectories.py)
                     "kept_times": [ep.asset.time(i) for i in (ep.evidence if kept_idx is None else kept_idx)],
                     "frame_cap": ep.frame_cap, "share_cap_per_place": b.get("share_cap_per_place"),
                     **({"picked_times": [ep.asset.time(i) for i in picked]} if picked is not None else {})}
            if b.get("record_verifier_inputs"):
                entry["input"] = inp.to_json()
            t.turns.append(entry)

    @staticmethod
    def _forced_keep(ep, w):
        """src/baselines/inject.py InjectInspectController (keep_prob): this inspection is a forced one in an injected
        wrong window whose picks are kept (once per injection)."""
        for x in (getattr(ep, "inject_meta", None) or {}).get("injections", []):
            iw = x.get("inspect_window")
            if (x.get("force_keep") and not x.get("force_keep_done") and iw
                    and abs(iw[0] - w.start) < 1e-6 and abs(iw[1] - w.end) < 1e-6):
                x["force_keep_done"] = True
                return True
        return False

    def _ver_line_keep(self, entry):
        """budget.verifier_q3 = false: the inspection's line without Q3."""
        from ..env.environment import _fmt_t
        w = entry["window"]
        return (f"Verifier after inspection attempt {entry['attempt']} (window {_fmt_t(w[0])}-{_fmt_t(w[1])}): kept "
                f"{entry['n_kept']} of {len(entry['frames'])} frames")

    def _ver_line_frames(self, t, entry, probs):
        from ..env.environment import _fmt_t
        ep, et = t.ep, sum(k * p for k, p in enumerate(probs))
        c = self.cfg["budget"].get("verifier_cuts", [0.25, 0.5, 0.8])
        names = (f"<{c[0]:.0%}", f"{c[0]:.0%}-{c[1]:.0%}", f"{c[1]:.0%}-{c[2]:.0%}", f">={c[2]:.0%}")
        w = entry["window"]
        if self.cfg["budget"].get("verifier_line_compact"):   # the totals and the actions
            return (f"Verifier after inspection attempt {entry['attempt']} (window {_fmt_t(w[0])}-{_fmt_t(w[1])}): kept "
                    f"{entry['n_kept']} of {len(entry['frames'])} frames; answerer's probability for the correct option "
                    f"on the kept frames: " + ", ".join(f"{n} {p:.2f}" for n, p in zip(names, probs))
                    + f" (expected tier {et:.2f})")    # left are in the memory lines already
        return (f"Verifier after inspection attempt {entry['attempt']} (window {_fmt_t(w[0])}-{_fmt_t(w[1])}): kept "
                f"{entry['n_kept']} of {len(entry['frames'])} frames, {len(ep.evidence)} of {ep.frame_cap} kept in total; "
                f"answerer's probability for the correct option on the kept frames: "
                + ", ".join(f"{n} {p:.2f}" for n, p in zip(names, probs))
                + f" (expected tier {et:.2f}); " + (f"inspection attempts left {ep.max_attempts - ep.attempts}"
                                                     if ep.max_attempts is not None else
                                                     f"actions left {self.cfg['budget']['max_turns'] - ep.turn}"))

    def _finalize(self, trajs, reason):
        """End the episodes: with budget.tentative, back-fill the free inspection slots from the pool of withdrawn
        batches (highest expected tier first, <= budget.backfill_per_window per window), then answer once."""
        b = self.cfg["budget"]
        if b.get("tentative") and b.get("verifier_per_frame"):    # frame pool: highest P(keep) first, up to the budget
            for t in trajs:
                need = t.ep.frame_cap - len(t.ep.evidence)
                pool = sorted(getattr(t, "tent_pool", []), key=lambda x: -x["p"])[:max(need, 0)]
                if pool:
                    idx = [i for e in pool for i in e["idx"]]
                    t.ep.request_frames(idx, restore=True)
                    t.turns.append({"turn_id": None, "action": "TENTATIVE_RESTORE", "n_frames": len(idx),
                                    "p_keep": [e["p"] for e in pool], "times": [t.ep.asset.time(i) for i in idx]})
        elif b.get("tentative"):
            cap = b.get("backfill_per_window")
            for t in trajs:
                need = (t.ep.max_inspections or 0) - len(t.ep.inspections)
                per_win = collections.Counter()
                for e in sorted(getattr(t, "tent_pool", []), key=lambda x: -x["et"]):
                    if need <= 0:
                        break
                    if cap is not None and per_win[e["window"]] >= int(cap):
                        continue
                    per_win[e["window"]] += 1
                    need -= 1
                    t.ep.request_frames(e["idx"], restore=True)
                    t.turns.append({"turn_id": None, "action": "TENTATIVE_RESTORE", "et": e["et"],
                                    "window": list(e["window"]), "kept_before": e["kept_before"],
                                    "times": [t.ep.asset.time(i) for i in e["idx"]]})
        self._answer(trajs, reason)

    # ------------------------------------------------------------------ main loop
    def run(self, items, policy_version=0, reference_version=None):
        b = self.cfg["budget"]
        # Policies invalidate static snapshots on training / weight loading.
        # Frozen roles can therefore retain their bounded caches across batches.
        trajs = []
        for it in items:
            ep = EpisodeV3(it["sample"], it["asset"], self.features, self.encoder, self.cfg, key=it["trajectory_id"])
            if it.get("plan") is not None:
                ep.plan = it["plan"]                 # branch plan for BranchDecider (scripts/train_branch_rl.py)
            t = Trajectory(ep, it["group_id"], it["trajectory_id"], it["seed"], policy_version, reference_version)
            t.generator = torch.Generator().manual_seed(int(it["seed"]))
            t.verifier_log, t.answer_meta = [], None
            t0 = time.perf_counter()
            try:
                ep.reset()
            except Exception:
                t.status, t.error = "infra_error", traceback.format_exc(limit=3)
            t.seconds["env"] += time.perf_counter() - t0
            trajs.append(t)
        tl = self.text_len
        live = [t for t in trajs if t.status == "ok"]
        stop = self._verifier_phase(live, -1, policy_version)          # verifier @overview
        if stop:
            self._answer(stop, "verifier_stop")
        for turn in range(b["max_turns"]):
            live = [t for t in trajs if t.status == "ok" and not t.ep.done]
            if not live:
                break
            if b.get("verifier_where"):
                self._where_phase(live, turn, policy_version)
                live = [t for t in live if t.status == "ok"]
            if b.get("locator_line") and "localize" in self.deciders:
                self._locator_probe(live, turn)
                live = [t for t in live if t.status == "ok"]
            live, inputs = self._inputs(live, lambda t: t.ep.controller_input(tl))
            ctrl = self._decide("controller", live, inputs, [None] * len(live), turn, policy_version)
            actions, no_action = {}, []
            for t, rec in zip(live, ctrl):
                if rec is None:
                    no_action.append(t)
                    continue
                act, forced = rec.action_id, None
                if getattr(t, "force_switch", False) or getattr(t, "force_inspect", False):
                    rec.trainable = False            # the executed action is not the sampled one: no credit
                    legal = t.ep.controller_mask()
                    if getattr(t, "force_switch", False):
                        t.force_switch = False
                        if legal.get("LOCALIZE_GLOBAL"):
                            act, forced, t.exclude_searched, t.force_inspect = "LOCALIZE_GLOBAL", "switch", True, True
                    elif legal.get("INSPECT_FRAMES"):
                        act, forced, t.force_inspect = "INSPECT_FRAMES", "inspect_after_switch", False
                actions[id(t)] = (act, rec)
                t.turns.append({"turn_id": turn, "action": act, "controller_decision": rec.decision_id,
                                "tool_status": "ok", **({"forced": forced, "policy_action": rec.action_id} if forced else {})})
            if no_action:
                self._finalize(no_action, "no_legal_action")
            for t in live:                        # window tree: UP returns to the parent (no locator call)
                if t.status == "ok" and actions.get(id(t), ("",))[0] == "UP":
                    t.ep.apply_up()
                    t.turns[-1]["interval"] = t.ep.active.to_json()
            for kind, act in (("global", "LOCALIZE_GLOBAL"), ("local", "REFINE_LOCAL"), ("sibling", "SWITCH_LOCAL")):
                sel = [t for t in live if t.status == "ok" and actions.get(id(t), ("",))[0] == act]
                if not sel:
                    continue
                wins = {id(t): self._windows(t, kind) for t in sel}
                for t in sel:                     # switch rule: only windows not searched yet
                    if getattr(t, "exclude_searched", False) and kind == "global":
                        t.exclude_searched = False
                        ov = lambda w, u: max(0.0, min(w.end, u.end) - max(w.start, u.start)) / max(w.length, 1e-9)
                        rest = [w for w in wins[id(t)] if all(ov(w, u) < 0.5 for u in t.ep.searched)]   # mostly new time
                        wins[id(t)] = rest or wins[id(t)]
                sel, inputs = self._inputs(sel, lambda t: t.ep.localize_input(kind, wins[id(t)], tl))
                recs = self._decide("localize", sel, inputs, [actions[id(t)][1].decision_id for t in sel], turn,
                                    policy_version)
                for t, rec in zip(sel, recs):
                    if rec is None:
                        t.turns[-1]["tool_status"] = "no_legal_window"
                        continue
                    w = wins[id(t)][rec.action_index]
                    t.ep.apply_window(kind, w)
                    t.turns[-1]["interval"] = w.to_json()
            sel = [t for t in live if t.status == "ok" and actions.get(id(t), ("",))[0] == "INSPECT_FRAMES"]
            t0 = time.perf_counter()
            cands = {id(t): t.ep.frame_candidates() for t in sel}
            for t in sel:                                  # guard: nothing affordable to show (encoder budget)
                if not cands[id(t)]:
                    t.turns[-1]["tool_status"] = "no_candidates"
            sel = [t for t in sel if cands[id(t)]]
            self._env_seconds(sel, time.perf_counter() - t0)
            cur, inputs = self._inputs(sel, lambda t: t.ep.frame_input_c(cands[id(t)], tl))
            recs = self._decide("frame", cur, inputs, [actions[id(t)][1].decision_id for t in cur], turn, policy_version)
            for t, rec in zip(cur, recs):
                if rec is None:
                    t.turns[-1]["tool_status"] = "no_frame_selected"
                    continue
                picks = rec.info.get("picks") or [rec.action_index]
                chosen = [cands[id(t)][j] for j in picks]
                t.ep.request_frames(chosen)
                t.turns[-1]["frame_times"] = [t.ep.asset.time(i) for i in sorted(chosen, key=t.ep.asset.time)]
            for t in live:
                t.ep.end_turn()
            if b.get("tentative") and ("verifier" in self.deciders or (b.get("verifier_per_frame") and
                                                                       b.get("verifier_frame_oracle_pad") is not None)):
                if b.get("verifier_per_frame"):
                    self._tentative_frames(cur, turn, policy_version)
                else:
                    self._tentative(cur, turn, policy_version)
            live = [t for t in live if t.status == "ok" and not t.ep.done]
            stop = self._verifier_phase(live, turn, policy_version)       # verifier @inspection1
            if stop:
                self._answer(stop, "verifier_stop")
            sw = float(b.get("switch_tau", 0) or 0)
            if sw > 0:
                for t in live:
                    v = getattr(t, "verifier_log", None)
                    if (t.status == "ok" and not t.ep.done and v and v[-1]["point"] == "inspection1"
                            and not getattr(t, "switch_checked", False)):
                        t.switch_checked = True
                        t.force_switch = v[-1]["p_tier3"] < sw
            done = [t for t in live if not t.ep.done and t.ep.frame_budget_done()]
            if done:
                self._finalize(done, "frame_budget_exhausted")
        rest = [t for t in trajs if t.status == "ok" and not t.ep.done]
        if rest:
            self._finalize(rest, "budget_exhausted")
        if self.features is not None:
            self.features.flush()
        out = []
        for t in trajs:
            d = t.to_json()
            d["verifier_log"] = getattr(t, "verifier_log", [])
            d["answer_meta"] = getattr(t, "answer_meta", None)
            d["final_letter_probs"] = getattr(t, "final_letter_probs", None)
            d["locator_log"] = getattr(t, "locator_log", [])
            # which verifier answers were oracle (key_spans) rather than model predictions: SFT conversion refuses
            # controller rows (and SWITCH locator rows carrying the where line) from such trajectories
            d["oracle"] = {"frame_pad": b.get("verifier_frame_oracle_pad"), "where_pad": b.get("verifier_where_oracle_pad")}
            if getattr(t.ep, "inject_meta", None) is not None:      # collection with src/baselines/inject.py
                d["injection"] = t.ep.inject_meta
            dec = d.get("decisions", [])
            d["accounting"] = {"controller_decisions": sum(x["decision_type"] == "controller" for x in dec),
                               "verifier_calls": sum(x["decision_type"] == "verifier" for x in dec),
                               "answerer_calls": t.ep.cost.get("reader_calls", 0),
                               "answer_frames": t.ep.cost.get("reader_frames", 0),
                               "frames_seen": len(t.ep.overview_in_answer()) + t.ep.frames_inspected,
                               "jev_encoded_frames": t.ep.cost.get("encoder_frames", 0),
                               "attempts": t.ep.attempts, "withdrawn": len(t.ep.withdrawn), "restored": t.ep.n_restored}
            out.append(d)
        return out
