"""Comparison deciders on the same interface, candidates and visible frames (update.md §7).

* RandomDecider          : uniform over legal candidates (seeded per trajectory).
* UniformFrameDecider    : frame selection at fixed even positions of the candidate list.
* ScriptedController     : fixed action per turn, e.g. INSPECT_FRAMES then ANSWER.
* QwenChoiceDecider      : the frozen Qwen2.5-VL reader makes the choice. It sees the same prefix
                           text with every frame reference replaced by that frame as an image, plus
                           the candidate list; it may think briefly and must end with
                           "Answer: <candidate id>". Its text never reaches the answerer.
* UniTimeDecider         : optional comparison (update.md §6 M0): UniTime window -> max-IoU window.
All return one-hot choices; their records have trainable=False.
"""
import re
import time

import requests
import torch

from ..evaluation.localization import interval_iou

_ANSWER = re.compile(r"Answer\s*:\s*\**\s*([A-Za-z_]+\d*)")
_ANSWER_LINE = re.compile(r"Answer\s*:\s*\**\s*([^\n]*)")


def _onehot(k, i):
    p = [0.0] * k
    p[i] = 1.0
    return p


def _result(inp, idx, secs, info):
    return {"index": idx, "log_prob": 0.0, "probs": _onehot(len(inp.candidates), idx), "entropy": 0.0,
            "forced": len(inp.legal_indices()) == 1, "seconds": secs, "info": info}


def _cand_text(c):
    return c[0].replace("Candidate:\n", "").replace("\nDecision:", "")


class RandomDecider:
    name = "random"

    def decide(self, inputs, episodes, generators):
        out = []
        for inp, g in zip(inputs, generators):
            legal = inp.legal_indices()
            if not legal:
                out.append({"index": None, "log_prob": 0.0, "probs": [0.0] * len(inp.candidates),
                            "entropy": 0.0, "forced": True})
                continue
            j = legal[int(torch.randint(len(legal), (1,), generator=g))]
            out.append(_result(inp, j, 0.0, {}))
        return out


class UniformFrameDecider:
    """Pick j of k goes to candidate round((j + 0.5) * n / k - 0.5) (or its nearest legal one)."""
    name = "uniform_frames"

    def __init__(self, k=2):
        self.k = k

    def decide(self, inputs, episodes, generators):
        out = []
        for inp in inputs:
            n = len(inp.candidates)
            legal = inp.legal_indices()
            if inp.select_k > 1:     # selection C: k evenly spaced legal candidates (frames already sent are illegal,
                picks = []           # so a second inspection lands between the first one's frames)
                for j in range(inp.select_k):
                    target = min(n - 1, max(0, round((j + 0.5) * n / inp.select_k - 0.5)))
                    rest = [i for i in legal if i not in picks]
                    if not rest:
                        break
                    picks.append(min(rest, key=lambda i: (abs(i - target), i)))
                if not picks:
                    out.append({"index": None, "log_prob": 0.0, "probs": [0.0] * n, "entropy": 0.0, "forced": True})
                    continue
                out.append(_result(inp, picks[0], 0.0, {"picks": picks}))
                continue
            j = n - len(legal)                             # picks already made in this turn
            target = min(n - 1, max(0, round((j + 0.5) * n / self.k - 0.5)))
            idx = min(legal, key=lambda i: (abs(i - target), i)) if legal else None
            out.append(_result(inp, idx, 0.0, {"target": target}) if idx is not None else
                       {"index": None, "log_prob": 0.0, "probs": [0.0] * n, "entropy": 0.0, "forced": True})
        return out


class UniformStratFrameDecider:
    """Selection C, stratified across inspections (G1 diagnostic, 2026-10-02): with M inspections of k frames, the
    k*M slots of the active window are (s + 0.5) / (k*M); inspection m (1-based) takes slots s = j*M + m - 1, so the
    picks of all inspections together are evenly spread (M=2, k=2, 32 candidates: 4, 20 then 12, 28) instead of the
    second inspection landing next to the first one's frames (UniformFrameDecider). Nearest legal candidate."""
    name = "uniform_strat"

    def decide(self, inputs, episodes, generators):
        out = []
        for inp, ep in zip(inputs, episodes):
            n, k = len(inp.candidates), inp.select_k
            big_m, m = max(int(ep.max_inspections), 1), len(ep.inspections) + 1
            legal, picks = inp.legal_indices(), []
            for j in range(k):
                target = min(n - 1, max(0, round((j * big_m + m - 1 + 0.5) * n / (k * big_m) - 0.5)))
                rest = [i for i in legal if i not in picks]
                if not rest:
                    break
                picks.append(min(rest, key=lambda i: (abs(i - target), i)))
            out.append(_result(inp, picks[0], 0.0, {"picks": picks}) if picks else
                       {"index": None, "log_prob": 0.0, "probs": [0.0] * n, "entropy": 0.0, "forced": True})
        return out


class ScriptedController:
    name = "scripted"

    def __init__(self, plan=("INSPECT_FRAMES", "ANSWER")):
        self.plan = list(plan)

    def decide(self, inputs, episodes, generators):
        out = []
        for inp, ep in zip(inputs, episodes):
            a = self.plan[min(ep.turn, len(self.plan) - 1)]
            legal = inp.legal_indices()
            if not legal:
                out.append({"index": None, "log_prob": 0.0, "probs": [0.0] * len(inp.candidates),
                            "entropy": 0.0, "forced": True})
                continue
            idx = inp.candidate_ids.index(a) if a in inp.candidate_ids and inp.valid[inp.candidate_ids.index(a)] \
                else legal[0]                      # planned action illegal under the v2 masks: first legal one
            out.append(_result(inp, idx, 0.0, {"plan": self.plan, "planned": a}))
        return out


class LocatorMarginController:
    """Rule controller from the randomized-controller analysis (2026-10-05; needs budget.locator_line): on the full
    video, inspect first; afterwards localize iff the locator's top-1 minus top-2 probability over the whole-video
    windows is >= `margin` and its best window was not searched yet, else inspect again. Inside a localized window:
    inspect; when nothing can be inspected there, localize if the locator's best window is new, else refine (a new
    sub-window has new candidates). Illegal choices fall back in the order INSPECT_FRAMES, LOCALIZE_GLOBAL, REFINE_LOCAL."""
    name = "locator_margin_rule"

    def __init__(self, margin=0.3):
        self.margin = float(margin)

    def decide(self, inputs, episodes, generators):
        out = []
        for inp, ep in zip(inputs, episodes):
            legal = [inp.candidate_ids[i] for i in inp.legal_indices()]
            if not legal:
                out.append({"index": None, "log_prob": 0.0, "probs": [0.0] * len(inp.candidates),
                            "entropy": 0.0, "forced": True})
                continue
            g = ep.loc_info.get("global")
            assert g is not None or not ep.active_is_full(), "locator_margin_rule needs budget.locator_line=true"
            if ep.active_is_full() and ep.attempts > 0 and g["margin"] >= self.margin and not g["best_searched"]:
                prefs, rule = ("LOCALIZE_GLOBAL", "INSPECT_FRAMES"), "localize_confident"
            elif ep.active_is_full():
                prefs, rule = ("INSPECT_FRAMES", "LOCALIZE_GLOBAL"), "inspect_full"
            elif g is not None and not g["best_searched"]:
                prefs, rule = ("INSPECT_FRAMES", "LOCALIZE_GLOBAL", "REFINE_LOCAL"), "inspect_window"
            else:                         # re-localizing would return to a searched window: refine when nothing is left
                prefs, rule = ("INSPECT_FRAMES", "REFINE_LOCAL", "LOCALIZE_GLOBAL"), "inspect_window"
            a = next((x for x in prefs + ("INSPECT_FRAMES", "LOCALIZE_GLOBAL", "REFINE_LOCAL") if x in legal), legal[0])
            out.append(_result(inp, inp.candidate_ids.index(a), 0.0,
                               {"rule": rule, "margin": None if g is None else g["margin"]}))
        return out


class QwenChoiceDecider:
    name = "qwen_decider"

    def __init__(self, reader, max_new_tokens=96, visual_tokens=None):
        # frame images: reader.decider_frame_visual_tokens converted with the reader's pixels per token
        # (QwenReader sets token_px; a reader without it is taken as Qwen2.5-VL, 28 px); null = no pre-shrink,
        # the model's official preprocessing
        visual_tokens = visual_tokens or (getattr(reader, "cfg", None) or {}).get("decider_frame_visual_tokens")
        self.reader, self.max_new_tokens = reader, max_new_tokens
        self.max_pixels = visual_tokens * getattr(reader, "token_px", 28) ** 2 if visual_tokens else None

    def _messages(self, inp, ep, strict=False):
        content = []
        refs = [s for s in inp.prefix if not isinstance(s, str)]
        frames = ep.asset.frames([ep.frame_for_ref(r) for r in refs]) if refs else {}
        for s in inp.prefix:
            if isinstance(s, str):
                content.append({"type": "text", "text": s})
            else:
                content.append({"type": "image", "image": frames[ep.frame_for_ref(s)], "max_pixels": self.max_pixels})
        legal = [_cand_text(inp.candidates[i]) for i in inp.legal_indices()]
        k = inp.select_k
        what = "<candidate id>" if k <= 1 else f"<{k} distinct candidate ids, comma-separated>"
        note = ("You are choosing the agent's next step from the candidates above (by candidate id), not answering the "
                "video question. ")       # 2026-10-01: in the v3 flow the model otherwise answered the video question
        tail = (note + f"Reply with exactly one line: Answer: {what}" if strict else
                note + f"Think briefly (at most two sentences), then end with a line: Answer: {what}")
        content.append({"type": "text", "text": "Candidates:\n" + "\n".join(legal) + "\n" + tail})
        return [{"role": "user", "content": content}]

    def _parse(self, text, inp):
        """Legal candidate ids after the last "Answer:" line; without such a line (2026-10-01, lenient fallback):
        legal ids anywhere in the reply (single choice: the last one; selection C: the first k distinct)."""
        legal = {inp.candidate_ids[i]: i for i in inp.legal_indices()}
        if inp.select_k > 1:                       # selection C: k distinct legal ids after the last "Answer:"
            m = list(_ANSWER_LINE.finditer(text))
            scope = m[-1].group(1) if m else text
            ids = [x for x in re.findall(r"[A-Za-z_]+\d*", scope) if x in legal]
            picks = list(dict.fromkeys(legal[x] for x in ids))[:inp.select_k]
            return picks or None
        found = [m for m in _ANSWER.findall(text) if m in legal]
        if not found:
            found = [x for x in re.findall(r"[A-Za-z_]+\d*", text) if x in legal]
        return legal[found[-1]] if found else None

    def decide(self, inputs, episodes, generators):
        t0 = time.perf_counter()
        texts, meta = self.reader.generate([self._messages(i, e) for i, e in zip(inputs, episodes)],
                                           self.max_new_tokens)
        picks = [self._parse(t, i) for t, i in zip(texts, inputs)]
        retry = [j for j, p in enumerate(picks) if p is None and inputs[j].legal_indices()]
        if retry:
            t2, _ = self.reader.generate([self._messages(inputs[j], episodes[j], strict=True) for j in retry], 16)
            for j, t in zip(retry, t2):
                picks[j] = self._parse(t, inputs[j])
                texts[j] += " || retry: " + t
        secs = (time.perf_counter() - t0) / max(len(inputs), 1)
        out = []
        for inp, p, t, m, g in zip(inputs, picks, texts, meta, generators):
            legal = inp.legal_indices()
            if not legal:
                out.append({"index": None, "log_prob": 0.0, "probs": [0.0] * len(inp.candidates),
                            "entropy": 0.0, "forced": True})
                continue
            failed = p is None
            if failed:   # unparsable twice: random legal choice, counted as a parse failure
                p = legal[int(torch.randint(len(legal), (1,), generator=g))]
            info = {"raw": t, "parse_failure": failed, "new_tokens": m["new_tokens"],
                    "prompt_tokens": m["prompt_tokens"], "visual_tokens": m["visual_tokens"]}
            if inp.select_k > 1:                   # fill missing picks with random legal ones (counted)
                picks = list(p) if isinstance(p, list) else [p]
                rest = [i for i in legal if i not in picks]
                while len(picks) < inp.select_k and rest:
                    picks.append(rest.pop(int(torch.randint(len(rest), (1,), generator=g))))
                info.update(picks=picks, filled=inp.select_k - (len(p) if isinstance(p, list) else 1))
                p = picks[0]
            out.append(_result(inp, p, secs, info))
        return out


class UniTimeDecider:
    name = "unitime"

    def __init__(self, url="http://127.0.0.1:9995/ground", timeout=300):
        self.url, self.timeout = url, timeout

    def decide(self, inputs, episodes, generators):
        out = []
        for inp, ep in zip(inputs, episodes):
            t0 = time.perf_counter()
            whole = [p for p in inp.payloads if p["kind"] in ("full", "active")][0]
            r = requests.post(self.url, json={"video_path": ep.asset.path, "query": ep.sample.question,
                                              "video_start": whole["start"], "video_end": whole["end"]},
                              timeout=self.timeout)
            r.raise_for_status()
            js = r.json()
            wins = js.get("windows") or []
            legal = inp.legal_indices()
            if wins:
                pred = wins[0]
                best, idx = max((interval_iou(pred, (inp.payloads[i]["start"], inp.payloads[i]["end"])), i)
                                for i in legal)
            else:
                pred, best = None, 0.0
                idx = inp.payloads.index(whole)
            out.append(_result(inp, idx, time.perf_counter() - t0,
                               {"pred_window": pred, "raw": js.get("raw"), "mapped_iou": best}))
        return out


class PriorDecider:
    """SigLIP2 prior only (2026-09-30 selection test): localize = argmax of the window prior; frame (selection C) =
    greedy masked picks on the prior scores (same pick rule as the policy). Uses the prior scorer of a JevPolicy."""
    name = "prior"

    def __init__(self, policy):
        self.policy = policy

    def decide(self, inputs, episodes, generators):
        from ..policy.selection import pick
        out = []
        for inp in inputs:
            n = len(inp.candidates)
            legal = inp.legal_indices()
            if not legal:
                out.append({"index": None, "log_prob": 0.0, "probs": [0.0] * n, "entropy": 0.0, "forced": True})
                continue
            with torch.no_grad():
                z = self.policy.prior_scorer.scores(inp, self.policy.feature_store).float().cpu()
            z = torch.where(torch.tensor(inp.valid), z, torch.full_like(z, float("-inf")))
            if inp.select_k > 1:
                picks, _, _, _ = pick(z, inp.valid, inp.select_k, inp.mask_radius, mode="greedy")
                out.append(_result(inp, picks[0], 0.0, {"picks": list(picks)}))
            else:
                out.append(_result(inp, int(z.argmax()), 0.0, {}))
        return out
