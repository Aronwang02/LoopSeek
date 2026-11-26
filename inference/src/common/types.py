"""Data contracts shared by env, policy, rollout and training.

A DecisionInput is the exact, serializable input of one policy decision. Text is stored verbatim;
visual tokens are stored as references (video content id + exact frame pts + time features) into
the frozen feature cache, so an update can rebuild the identical input without re-running the env.
"""
from dataclasses import asdict, dataclass, field
from typing import Optional

DECISION_TYPES = ("controller", "localize", "frame", "verifier")
TASK_ID = {name: i for i, name in enumerate(DECISION_TYPES)}

CONTROLLER_ACTIONS = ("LOCALIZE_GLOBAL", "REFINE_LOCAL", "INSPECT_FRAMES", "INSPECT_CLIP", "ANSWER")


@dataclass(frozen=True)
class VisRef:
    content_id: str
    pts: int            # exact stream pts of the frame (feature-cache key)
    t: float            # seconds from video start (actual decoded time)
    rel_interval: float  # position inside the interval it illustrates, in [0, 1]
    rel_video: float     # t / duration

    def to_json(self):
        return [self.content_id, self.pts, self.t, self.rel_interval, self.rel_video]

    @staticmethod
    def from_json(x):
        return VisRef(x[0], int(x[1]), float(x[2]), float(x[3]), float(x[4]))


# A segment is either a text string or a VisRef.
def seg_to_json(seg):
    return {"t": seg} if isinstance(seg, str) else {"v": seg.to_json()}


def seg_from_json(d):
    return d["t"] if "t" in d else VisRef.from_json(d["v"])


@dataclass
class DecisionInput:
    decision_type: str                 # controller | localize | frame
    prefix: list                       # segments: state + decision question
    candidates: list                   # list of segment lists, one per candidate (EOS appended by model)
    candidate_ids: list                # stable ids, in candidate order
    valid: list                        # legality mask (bool) in candidate order
    payloads: list = field(default_factory=list)  # env-side meaning of each candidate (json)
    # SigLIP2 prior (policy.siglip_prior): query texts (score = max over texts) and, per candidate, the
    # frames whose max similarity is that candidate's prior. None = no prior for this decision.
    prior_texts: Optional[list] = None
    prior_refs: Optional[list] = None                  # list (per candidate) of VisRef lists
    select_k: int = 1                                  # >1: one scoring, k picks (selection C)
    mask_radius: int = 0                               # selection C: picking j masks j-r..j+r
    static_segments: int = 0                           # leading prefix segments shared by the role in an episode (cache)
    group_questions: Optional[dict] = None             # grouped decisions: {str(group): question text} (letter readout)

    def legal_indices(self):
        return [i for i, v in enumerate(self.valid) if v]

    def to_json(self):
        return {"decision_type": self.decision_type,
                "prefix": [seg_to_json(s) for s in self.prefix],
                "candidates": [[seg_to_json(s) for s in c] for c in self.candidates],
                "candidate_ids": list(self.candidate_ids), "valid": [bool(v) for v in self.valid],
                "payloads": self.payloads,
                "prior_texts": self.prior_texts,
                "prior_refs": None if self.prior_refs is None else [[r.to_json() for r in c] for c in self.prior_refs],
                "select_k": self.select_k, "mask_radius": self.mask_radius, "static_segments": self.static_segments,
                "group_questions": self.group_questions}

    @staticmethod
    def from_json(d):
        return DecisionInput(d["decision_type"], [seg_from_json(s) for s in d["prefix"]],
                             [[seg_from_json(s) for s in c] for c in d["candidates"]],
                             list(d["candidate_ids"]), [bool(v) for v in d["valid"]],
                             d.get("payloads", []), d.get("prior_texts"),
                             None if d.get("prior_refs") is None else
                             [[VisRef.from_json(r) for r in c] for c in d["prior_refs"]],
                             int(d.get("select_k", 1)), int(d.get("mask_radius", 0)), int(d.get("static_segments", 0)),
                             d.get("group_questions"))


@dataclass
class DecisionRecord:
    """One sampled (or forced) choice, with everything needed to recompute its log-prob."""
    decision_id: int
    turn_id: int
    decision_type: str
    parent_decision_id: Optional[int]
    input: DecisionInput
    action_index: int                  # index into the full candidate list
    action_id: str
    old_log_prob: float                # log-prob under the behavior policy at sampling time
    old_probs: list                    # full distribution over the candidate list (0 for illegal)
    entropy: float
    candidate_count: int               # number of legal candidates
    forced: bool                       # only one legal candidate: not a policy choice
    trainable: bool                    # participates in the policy loss
    policy_version: int
    decider: str = "jev"               # jev | qwen_planner | unitime | qwen_frame_scorer | ...
    seconds: float = 0.0
    info: dict = field(default_factory=dict)   # decider-specific evidence (raw text, scores, IoU)

    def to_json(self):
        d = asdict(self)
        d["input"] = self.input.to_json()
        return d

    @staticmethod
    def from_json(d):
        """Training-side annotations added later (advantage, ref_log_prob, ...) are ignored."""
        from dataclasses import fields
        names = {f.name for f in fields(DecisionRecord)}
        d = {k: v for k, v in d.items() if k in names}
        d["input"] = DecisionInput.from_json(d["input"])
        return DecisionRecord(**d)
