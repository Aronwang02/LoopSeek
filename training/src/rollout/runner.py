"""Batched multi-turn rollouts over real videos.

Episodes advance in lockstep phases so policy forwards and reader calls are batched:
  controller -> (localize | sequential frame picks | clip | answer) -> reader -> next turn.
After `max_turns` turns without ANSWER the environment forces the final answer
(termination_reason=budget_exhausted); that answer is not a policy action.

Deciders are pluggable per decision type (JEV policy, Qwen planner, UniTime, Qwen frame scorer),
which is how the B1-B4 systems share one environment.
"""
import time
import traceback

import torch

from ..common.types import DecisionRecord
from ..env.environment import Episode, ReaderRequest


class JevDecider:
    def __init__(self, policy, mode="sample", temperature=1.0, max_paths=64, max_tokens=65536):
        self.policy, self.mode, self.temperature = policy, mode, temperature
        self.max_paths, self.max_tokens = max_paths, max_tokens
        self.name = "jev"

    def decide(self, inputs, episodes, generators):
        from ..policy.sampler import decide
        return decide(self.policy, inputs, generators, mode=self.mode, temperature=self.temperature,
                      max_paths=self.max_paths, max_tokens=self.max_tokens)


class Trajectory:
    def __init__(self, ep: Episode, group_id, traj_id, seed, policy_version, reference_version):
        self.ep = ep
        self.meta = {"sample_id": ep.sample.sample_id, "video_id": ep.sample.video_id,
                     "group_id": group_id, "trajectory_id": traj_id, "seed": seed,
                     "policy_version": policy_version, "reference_version": reference_version}
        self.decisions = []
        self.turns = []
        self.status = "ok"
        self.error = None
        self.seconds = {"policy": 0.0, "reader": 0.0, "env": 0.0}

    def add_decision(self, turn, dtype, parent, inp, res, decider, policy_version):
        k = len(inp.candidates)
        rec = DecisionRecord(
            decision_id=len(self.decisions), turn_id=turn, decision_type=dtype,
            parent_decision_id=parent, input=inp, action_index=res["index"],
            action_id=inp.candidate_ids[res["index"]], old_log_prob=res["log_prob"],
            old_probs=res["probs"], entropy=res["entropy"],
            candidate_count=len(inp.legal_indices()), forced=res["forced"],
            trainable=(not res["forced"]) and decider == "jev", policy_version=policy_version,
            decider=decider, seconds=res.get("seconds", 0.0), info=res.get("info", {}))
        assert k == len(rec.old_probs)
        self.decisions.append(rec)
        self.seconds["policy"] += rec.seconds
        return rec

    def to_json(self):
        return {**self.meta, "status": self.status, "error": self.error,
                "decisions": [d.to_json() for d in self.decisions], "turns": self.turns,
                "summary": self.ep.summary(), "seconds": self.seconds}


class RolloutRunner:
    def __init__(self, cfg, deciders, reader, features, encoder, text_len=None, log=print):
        self.cfg = cfg
        self.deciders = deciders
        self.reader = reader
        self.features = features
        self.encoder = encoder
        self.text_len = text_len
        self.log = log
        self.max_retries = cfg["rollout"]["max_retries"]

    # ------------------------------------------------------------------ helpers
    def _decide(self, dtype, trajs, inputs, parents, turn, policy_version):
        decider = self.deciders[dtype]
        gens = [t.generator for t in trajs]
        res = decider.decide(inputs, [t.ep for t in trajs], gens)
        recs = []
        for t, inp, r, p in zip(trajs, inputs, res, parents):
            if r["index"] is None:
                recs.append(None)
                continue
            recs.append(t.add_decision(turn, dtype, p, inp, r, decider.name, policy_version))
        return recs

    def _reader_jobs(self, trajs, requests):
        jobs = []
        for t, req in zip(trajs, requests):
            frames = t.ep.asset.frames(req.frame_indices)
            s = t.ep.sample
            jobs.append({"kind": req.kind, "frames": [frames[i] for i in req.frame_indices],
                         "times": req.frame_times, "question": s.question, "options": s.options,
                         "duration": t.ep.T, "source": req.source, "interval": req.interval,
                         "notes": req.notes})
        return jobs

    def _free_cuda_cache(self):
        """A vLLM reader is its own process on this GPU: hand back the blocks this process's allocator caches but does
        not use before the reader encodes and prefills (2026-10-13, 128 frames: the collector's cache grew from 17.6 to
        36 GiB and the reader's vision encoder ran out of memory). Weights and static caches stay."""
        if getattr(self.reader, "proc", None) is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _run_reader(self, trajs, requests):
        """Returns reader results aligned with requests; marks trajectories infra_error on failure."""
        if not requests:
            return []
        self._free_cuda_cache()
        t0 = time.perf_counter()
        jobs = self._reader_jobs(trajs, requests)
        self._env_seconds(trajs, time.perf_counter() - t0)
        for attempt in range(self.max_retries + 1):
            try:
                out = self.reader.run(jobs)
                for t, r in zip(trajs, out):
                    t.seconds["reader"] += r["meta"]["batch_seconds"] / r["meta"]["batch_size"]
                return out
            except Exception as exc:  # infrastructure failure, not a policy outcome
                self.log(f"reader failure attempt {attempt + 1}: {exc!r}")
                if "out of memory" in str(exc).lower():
                    torch.cuda.empty_cache()
                err = traceback.format_exc(limit=3)
        for t in trajs:
            t.status, t.error = "infra_error", err
        return [None] * len(requests)

    def _inputs(self, trajs, build):
        """Build decision inputs; a path above the hard token cap ends that trajectory
        (status path_too_long: excluded from training and reported, never truncated)."""
        from ..env.environment import PathTooLong
        keep, inputs = [], []
        t0 = time.perf_counter()
        for t in trajs:
            try:
                inputs.append(build(t))
                keep.append(t)
            except PathTooLong as exc:
                t.status, t.error = "path_too_long", str(exc)
        self._env_seconds(trajs, time.perf_counter() - t0)
        return keep, inputs

    @staticmethod
    def _env_seconds(trajs, secs):
        for t in trajs:
            t.seconds["env"] += secs / max(len(trajs), 1)

    def _label_stoppable_prefixes(self, live, inspected, turn):
        """Verifier labels (budget.label_pE): at every stoppable prefix (after an inspection, budget remaining)
        record the answerer's option-letter probabilities under the answer prompt with the evidence so far, plus
        the verifier DecisionInput of that state. The correct letter is applied later by the label script from
        the evaluator's labels; nothing here sees a label."""
        if not self.cfg["budget"].get("verifier_q3", False) or not hasattr(self.reader, "answer_letter_probs"):
            return                                       # budget.verifier_q3 off: no Q3, so no labels for it
        sel = [t for t in inspected if t.status == "ok" and not t.ep.done and t.ep.verifier_needed()]
        if not sel:
            return
        reqs = []
        for t in sel:
            ep = t.ep
            idx = sorted(ep.evidence, key=ep.asset.time)
            notes = [(f"frames {', '.join(f'{x:.1f}s' for x in o.times)}", o.text.strip()) for o in ep.observations]
            reqs.append(ReaderRequest("answer", t.meta["trajectory_id"], idx, [ep.asset.time(i) for i in idx], notes=notes))
        t0 = time.perf_counter()
        jobs = self._reader_jobs(sel, reqs)
        try:
            self._free_cuda_cache()
            probs = self.reader.answer_letter_probs(jobs)
        except Exception as exc:  # labelling must never kill a rollout
            self.log(f"letter-prob labelling failed: {exc!r}")
            return
        secs = (time.perf_counter() - t0) / max(len(sel), 1)
        for t, pr in zip(sel, probs):
            t.turns[-1]["letter_probs"] = pr
            t.turns[-1]["verifier_input"] = t.ep.verifier_input(self.text_len).to_json()
            t.turns[-1]["stoppable_prefix"] = {"inspections": len(t.ep.inspections), "evidence_frames": len(t.ep.evidence)}
            t.seconds["reader"] += secs

    def _v2_stop_phase(self, live, turn, policy_version):
        """New design (2026-09-30): after the reader phase, a trajectory answers when its frame budget is used
        (termination frame_budget_exhausted) or when the verifier decider (role 'verifier', optional) puts
        P(tier 3) >= budget.verifier_tau on the evidence so far (termination verifier_stop)."""
        b = self.cfg["budget"]
        ready, reqs = [], []
        ver = [t for t in live if t.status == "ok" and not t.ep.done and t.ep.verifier_needed()]
        if ver and "verifier" in self.deciders and b.get("verifier_q3", False):   # the stop rule reads Q3
            cur, inputs = self._inputs(ver, lambda t: t.ep.verifier_input(self.text_len))
            recs = self._decide("verifier", cur, inputs, [None] * len(cur), turn, policy_version)
            tau = float(b.get("verifier_tau", 0.5))
            for t, rec in zip(cur, recs):
                if rec is None:
                    continue
                p3 = rec.old_probs[3] if len(rec.old_probs) > 3 else 0.0
                t.turns[-1]["verifier"] = {"decision": rec.decision_id, "p_tier3": p3, "stop": p3 >= tau}
                if p3 >= tau:
                    ready.append(t)
                    reqs.append(t.ep.request_answer("verifier_stop"))
        for t in live:
            if t.status == "ok" and not t.ep.done and t not in ready and t.ep.frame_budget_done():
                ready.append(t)
                reqs.append(t.ep.request_answer("frame_budget_exhausted"))
        outs = self._run_reader(ready, reqs)
        for t, out in zip(ready, outs):
            if out is not None:
                t.ep.receive_answer(out["text"], out["parsed"])
                if t.turns:
                    t.turns[-1]["answer_meta"] = out["meta"]

    # ------------------------------------------------------------------ main loop
    def run(self, items, policy_version=0, reference_version=None):
        """items: list of dicts {sample: PublicSample, asset: VideoAsset, group_id, trajectory_id,
        seed}. Every item is an independent episode (no shared history inside a group)."""
        b = self.cfg["budget"]
        v2 = b.get("selection") == "c"
        trajs = []
        for it in items:
            ep = Episode(it["sample"], it["asset"], self.features, self.encoder, self.cfg,
                         key=it["trajectory_id"])
            t = Trajectory(ep, it["group_id"], it["trajectory_id"], it["seed"], policy_version,
                           reference_version)
            t.generator = torch.Generator().manual_seed(int(it["seed"]))
            t0 = time.perf_counter()
            try:
                ep.reset()
            except Exception:
                t.status, t.error = "infra_error", traceback.format_exc(limit=3)
            t.seconds["env"] += time.perf_counter() - t0
            trajs.append(t)
        tl = self.text_len

        for turn in range(b["max_turns"]):
            live = [t for t in trajs if t.status == "ok" and not t.ep.done]
            if not live:
                break
            # ---- controller
            live, inputs = self._inputs(live, lambda t: t.ep.controller_input(tl))
            ctrl = self._decide("controller", live, inputs, [None] * len(live), turn, policy_version)
            actions = {}
            for t, rec in zip(live, ctrl):
                actions[id(t)] = (rec.action_id if rec else "ANSWER", rec)
                t.turns.append({"turn_id": turn, "action": actions[id(t)][0],
                                "controller_decision": rec.decision_id if rec else None,
                                "tool_status": "ok"})
            # ---- localization
            for kind, act in (("global", "LOCALIZE_GLOBAL"), ("local", "REFINE_LOCAL")):
                sel = [t for t in live if t.status == "ok" and actions[id(t)][0] == act]
                if not sel:
                    continue
                wins = {id(t): t.ep.localize_windows(kind) for t in sel}
                sel, inputs = self._inputs(sel, lambda t: t.ep.localize_input(kind, wins[id(t)], tl))
                recs = self._decide("localize", sel, inputs,
                                    [actions[id(t)][1].decision_id for t in sel], turn, policy_version)
                for t, rec in zip(sel, recs):
                    if rec is None:
                        t.turns[-1]["tool_status"] = "no_legal_window"
                        continue
                    w = wins[id(t)][rec.action_index]
                    t.ep.apply_localize(w)
                    t.turns[-1]["interval"] = w.to_json()
            # ---- frame selection
            sel = [t for t in live if t.status == "ok" and actions[id(t)][0] == "INSPECT_FRAMES"]
            t0 = time.perf_counter()
            cands = {id(t): t.ep.frame_candidates() for t in sel}
            chosen = {id(t): [] for t in sel}
            self._env_seconds(sel, time.perf_counter() - t0)
            if v2:   # selection C: one decision scores all candidates, k masked picks (decision info "picks")
                cur, inputs = self._inputs(sel, lambda t: t.ep.frame_input_c(cands[id(t)], tl))
                recs = self._decide("frame", cur, inputs,
                                    [actions[id(t)][1].decision_id for t in cur], turn, policy_version)
                for t, rec in zip(cur, recs):
                    if rec is not None:
                        picks = rec.info.get("picks") or [rec.action_index]
                        chosen[id(t)] = [cands[id(t)][j] for j in picks]
            else:    # v1: fixed k sequential picks without replacement, no STOP
                open_ = [t for t in sel]
                for _ in range(b["frames_per_inspect"]):
                    if not open_:
                        break
                    cur, inputs = self._inputs(open_, lambda t: t.ep.frame_input(cands[id(t)], chosen[id(t)], tl))
                    recs = self._decide("frame", cur, inputs,
                                        [actions[id(t)][1].decision_id for t in cur], turn, policy_version)
                    open_ = []
                    for t, rec in zip(cur, recs):
                        if rec is None:          # no unselected frame left in a very short interval
                            continue
                        chosen[id(t)].append(cands[id(t)][rec.action_index])
                        open_.append(t)
            # ---- reader observations and active answers
            req_t, reqs = [], []
            for t in live:
                if t.status != "ok":
                    continue
                a = actions[id(t)][0]
                if a == "INSPECT_FRAMES":
                    if chosen[id(t)]:
                        req_t.append(t)
                        reqs.append(t.ep.request_frames(chosen[id(t)]))
                    else:
                        t.turns[-1]["tool_status"] = "no_frame_selected"
                elif a == "ANSWER":
                    req_t.append(t)
                    reqs.append(t.ep.request_answer("answer_action" if actions[id(t)][1] else "no_legal_action"))
            outs = self._run_reader(req_t, reqs)
            for t, req, out in zip(req_t, reqs, outs):
                if out is None:
                    continue
                if req.kind == "observe":
                    t.ep.receive_observation(req, out["text"])
                    t.turns[-1]["reader"] = out["meta"]
                    t.turns[-1]["frame_times"] = req.frame_times
                else:
                    t.ep.receive_answer(out["text"], out["parsed"])
                    t.turns[-1]["answer_meta"] = out["meta"]
            for t in live:
                t.ep.end_turn()
            if v2:
                if b.get("label_pE"):
                    self._label_stoppable_prefixes(live, req_t, turn)
                self._v2_stop_phase(live, turn, policy_version)

        # ---- forced final answers
        rest = [t for t in trajs if t.status == "ok" and not t.ep.done]
        reqs = [t.ep.request_answer("budget_exhausted") for t in rest]
        outs = self._run_reader(rest, reqs)
        for t, out in zip(rest, outs):
            if out is not None:
                t.ep.receive_answer(out["text"], out["parsed"])
        if self.features is not None:
            self.features.flush()
        return [t.to_json() for t in trajs]
