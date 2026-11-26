"""budget.keep_gate_opening: after an inspection's keep decision, the picks are
asked again in the OPENING format (EpisodeV3.opening_gate_input: the policy overview with the picks inserted in time order,
all named G01.., the initial-state memory, "Selected frames: none"); a pick stays only if that P(yes) >= the gate too."""
from conftest import ROOT, StubEncoder, StubReader
from src.baselines.teacher import EvidenceExpertController
from src.common.config import load_config
from src.rollout.runner_v3 import RunnerV3
from test_inference_settings import _res, mem
from test_long_flow import SAMPLE, FirstValid

SPANS = [[2.0, 2.6]]
on = lambda t: any(a <= t <= b for a, b in SPANS)          # noqa: E731


class FormatVerifier:
    """Keep groups on G frames (the opening scoring and the gate) answer open_fn, on E / F frames (the inspection)
    insp_fn; where / enclosing groups as in test_inference_settings.ScriptedVerifier."""
    name = "format_verifier"

    def __init__(self, insp_fn, open_fn):
        self.insp_fn, self.open_fn, self.inputs = insp_fn, open_fn, []

    def decide(self, inputs, episodes, generators):
        out = []
        for inp in inputs:
            self.inputs.append(inp)
            probs = [0.0] * len(inp.payloads)
            n_enc = sum("enclosing" in pl for pl in inp.payloads)
            for j, (cid, pl) in enumerate(zip(inp.candidate_ids, inp.payloads)):
                if "keep" in pl:
                    p = float((self.open_fn if str(cid).startswith("G") else self.insp_fn)(pl["time"]))
                    probs[j] = p if pl["keep"] else 1 - p
                elif "where" in pl:
                    probs[j] = {"before": 0.1, "inside_small": 0.2, "inside_large": 0.6, "after": 0.1}[pl["where"]]
                elif "enclosing" in pl:
                    probs[j] = 1.0 / n_enc
            out.append(_res(max(range(len(probs)), key=lambda j: probs[j]), len(probs), probs))
        return out


def run(asset, features, gate, open_fn, groups="picks"):
    cfg = load_config([ROOT / "configs" / "default.yaml", ROOT / "configs" / "loopseek.yaml"])
    cfg["budget"].update(min_window_sec=0.2, keep_gate_opening=gate, keep_gate_groups=groups)
    ann = {SAMPLE.sample_id: {"key_spans": SPANS}}
    ver = FormatVerifier(lambda t: 0.95, open_fn)            # the inspection keeps everything ("fooled")
    dec = {"controller": EvidenceExpertController(ann, pad=0.0), "localize": FirstValid(), "frame": FirstValid(),
           "verifier": ver}
    runner = RunnerV3(cfg, dec, StubReader(), features, StubEncoder(), text_len=None, annotations=ann)
    t = runner.run([{"sample": SAMPLE, "asset": asset, "group_id": "g", "trajectory_id": "t0", "seed": 0}])[0]
    assert t["status"] == "ok", t.get("error")
    return t, ver


def checks(t):
    return [x for x in t["turns"] if x.get("action") == "TENTATIVE_CHECK"]


def gate_inputs(t):
    """(the opening scoring's input, the gate decisions' inputs), as recorded in the trajectory: verifier decisions whose
    candidates are all G-frame keep options -- at the overview point (turn -1) and after inspections (turn >= 0)."""
    ov = [d for d in t["decisions"] if d["decision_type"] == "verifier"
          and all(str(c).startswith("G") for c in d["input"]["candidate_ids"])]
    return ov[0]["input"], [d["input"] for d in ov if d["turn_id"] >= 0]


def text_of(inp):
    return "".join(x.get("t", "") for x in inp["prefix"] if isinstance(x, dict))


def images(inp):
    return sum(1 for x in inp["prefix"] if isinstance(x, dict) and "v" in x)


def test_gate_drops_the_picks_the_opening_format_rejects(asset, features):
    off, _ = run(asset, features, None, on)
    gated, ver = run(asset, features, 0.3, lambda t: 0.9 if on(t) else 0.05)
    k_off = [f for x in checks(off) for f in x["frames"] if f["kept"]]
    k_gated = [f for x in checks(gated) for f in x["frames"] if f["kept"]]
    assert any(not on(f["time"]) for f in k_off)                      # without the gate: off-clue picks kept
    assert k_gated and all(on(f["time"]) for f in k_gated)            # with it: only the on-clue ones
    assert all("p_open" in f for x in checks(gated) for f in x["frames"])
    assert not any("p_open" in f for x in checks(off) for f in x["frames"])
    assert all(abs(f["p_open"] - (0.9 if on(f["time"]) else 0.05)) < 1e-6 for x in checks(gated) for f in x["frames"])


def test_gate_input_is_the_opening_format_with_the_picks_inserted(asset, features):
    t, _ = run(asset, features, 0.3, lambda t: 0.9 if on(t) else 0.05)
    opening, gates = gate_inputs(t)
    assert len(gates) == len(checks(t)) >= 1                          # one gate decision per inspection
    text_open = text_of(opening)
    memory = text_open[text_open.index("Memory:"):text_open.index("Selected frames")]
    assert "Active interval: [0.0s-" in memory and "Kept frames: 0 of 8" in memory
    open_times = {round(pl["time"], 6) for pl in opening["payloads"] if pl.get("keep")}
    for g, x in zip(gates, checks(t)):
        text = text_of(g)
        assert memory in text and "Selected frames: none" in text       # the initial state, nothing selected
        assert "Selected frames (time-ordered)" not in text and "Frames (time-ordered)" not in text
        picks = sorted(f["time"] for f in x["frames"])
        asked = sorted(pl["time"] for pl in g["payloads"] if pl.get("keep"))
        assert asked == picks                                          # only the picks' groups (picks mode)
        new = [p for p in picks if round(p, 6) not in open_times]
        assert images(g) == images(opening) + len(new)                 # the overview + the picks not already in it
        assert text.index("G01 [") < text.index("Memory:")              # inside the overview list, before the memory
        for p in picks:                                                # every pick is named in the G list
            assert any(f"[{v}s]" in text for v in (f"{p:.1f}", f"{p:.2f}"))
    t_all, _ = run(asset, features, 0.3, lambda t: 0.9 if on(t) else 0.05, groups="all")
    _, gates_all = gate_inputs(t_all)
    for g in gates_all:                                                # all mode: every shown frame is asked
        assert len(g["payloads"]) == 2 * images(g)


def test_a_gate_that_never_binds_changes_nothing_but_adds_its_decisions(asset, features):
    off, _ = run(asset, features, None, lambda t: 0.9)
    gated, _ = run(asset, features, 0.3, lambda t: 0.9)
    strip = lambda tr: [(d["decision_type"], d["action_id"], mem(d)) for d in tr["decisions"]   # noqa: E731
                        if not (d["decision_type"] == "verifier" and d["turn_id"] >= 0
                                and all(str(c).startswith("G") for c in d["input"]["candidate_ids"]))]
    assert strip(gated) == strip(off)
    assert [f["kept"] for x in checks(gated) for f in x["frames"]] == [f["kept"] for x in checks(off) for f in x["frames"]]
    assert gated["summary"]["evidence_times"] == off["summary"]["evidence_times"]
