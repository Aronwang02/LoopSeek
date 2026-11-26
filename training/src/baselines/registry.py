"""Build per-role deciders by name, so every §7 comparison runs through the same runner.

roles: controller | localize | frame
names: nanojev (policy; greedy or sample) | qwen (frozen reader decides) | random |
       uniform (frame only) | fixed (controller only: INSPECT_FRAMES then ANSWER) |
       fixed_lia (controller only: LOCALIZE_GLOBAL, INSPECT_FRAMES, ANSWER) | unitime (localize)
"""
from ..rollout.runner import JevDecider
from .deciders import (PriorDecider, QwenChoiceDecider, RandomDecider, ScriptedController, UniformFrameDecider,
                       UniformStratFrameDecider, UniTimeDecider)


def build_deciders(spec, policy=None, reader=None, cfg=None, mode="greedy", unitime_url=None,
                   annotations=None, teacher_mode="direct"):
    """spec: {"controller": name, "localize": name, "frame": name}. policy: one JevPolicy shared by the
    'nanojev' roles, or {role: JevPolicy} (src/policy/policy_set.py; a shared object gives one decider)."""
    cache = {}
    out = {}
    for role, name in spec.items():
        pol = policy.get(role) if isinstance(policy, dict) else policy
        key = (name, id(pol)) if name == "nanojev" else (name, role) if name == "teacher" else name
        if key in cache:
            out[role] = cache[key]
            continue
        if name == "nanojev":
            d = JevDecider(pol, mode=mode, max_paths=cfg["policy"]["max_paths_per_forward"],
                           max_tokens=cfg["policy"]["max_tokens_per_forward"])
        elif name == "qwen":
            d = QwenChoiceDecider(reader)
        elif name == "random":
            d = RandomDecider()
        elif name == "prior":            # SigLIP2 prior only (localize / frame); needs a policy with a prior scorer
            assert role in ("localize", "frame") and pol is not None and pol.prior_scorer is not None
            d = PriorDecider(pol)
        elif name == "uniform":
            assert role == "frame"
            d = UniformFrameDecider(cfg["budget"]["frames_per_inspect"])
        elif name == "uniform_strat":    # G1 diagnostic: evenly spread across inspections
            assert role == "frame"
            d = UniformStratFrameDecider()
        elif name == "fixed":
            assert role == "controller"
            d = ScriptedController(("INSPECT_FRAMES", "ANSWER"))
        elif name == "fixed_inspect":    # new design fixed flow: inspect the full video until the frame budget is used
            assert role == "controller"
            d = ScriptedController(("INSPECT_FRAMES",))
        elif name == "fixed_v2":         # new design: localize once, then inspect until the frame budget is used
            assert role == "controller"
            d = ScriptedController(("LOCALIZE_GLOBAL", "INSPECT_FRAMES", "INSPECT_FRAMES", "INSPECT_FRAMES", "INSPECT_FRAMES"))
        elif name == "fixed_lia":
            assert role == "controller"
            d = ScriptedController(("LOCALIZE_GLOBAL", "INSPECT_FRAMES", "ANSWER"))
        elif name == "unitime":
            assert role == "localize"
            d = UniTimeDecider(unitime_url)
        elif name == "teacher":          # privileged rule teacher (needs key_spans), new design roles
            from .teacher import TeacherControllerV2, TeacherFrameSelectorC, TeacherLocalizer
            assert annotations is not None, "teacher deciders need annotations {sample_id: {key_spans}}"
            d = {"controller": lambda: TeacherControllerV2(annotations, teacher_mode),
                 "localize": lambda: TeacherLocalizer(annotations, teacher_mode),
                 "frame": lambda: TeacherFrameSelectorC(annotations, seed=cfg["project"]["seed"])}[role]()
        else:
            raise ValueError(name)
        cache[key] = d
        out[role] = d
    return out


def needs_policy(spec):
    return "nanojev" in spec.values() or "prior" in spec.values()
