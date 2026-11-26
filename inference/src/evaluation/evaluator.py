"""Training-side evaluator: the only component that holds answer labels.

Terminal reward R = 1[parsed answer == label]; unparsable answers are wrong. Infrastructure
failures are not rewards: they are reported separately and their groups are dropped.
"""
from .scoring import is_correct


class RewardEvaluator:
    def __init__(self, manifest_rows):
        self._labels = {r["sample_id"]: r["answer"] for r in manifest_rows}

    def score(self, traj):
        if traj["status"] != "ok":
            traj["reward"] = None
            return None
        parsed = traj["summary"]["answer"]
        gt = self._labels[traj["sample_id"]]
        correct = is_correct(parsed, gt)
        traj["evaluator"] = {"answer": gt, "parsed": parsed, "correct": bool(correct),
                             "parse_failure": parsed is None}
        traj["reward"] = 1.0 if correct else 0.0
        return traj["reward"]
