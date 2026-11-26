"""Label and metric entry points share the configured evidence boundary, including custom configurations."""
import json
import subprocess
import sys

import pytest
import yaml

from conftest import ROOT
from src.common.utils import write_jsonl


def evidence_run(tmp_path, pad):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({"budget": {"evidence_pad": pad}}))
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, [{"sample_id": "s1", "answer": "B", "duration": 1800,
                           "key_spans": [[100, 110]]}])
    times = [96.9, 97.0, 100.0, 113.0, 113.1]
    traj = {"sample_id": "s1", "trajectory_id": "t1", "status": "ok", "decisions": [],
            "summary": {"answer_times": times},
            "turns": [{"action": "TENTATIVE_CHECK", "frame_inputs": [
                {"time": t, "input": {"payloads": [{"keep": True}, {"keep": False}]}} for t in times]},
                {"action": "WHERE_CHECK", "window": [115, 120],
                 "input": {"payloads": [{"where": w} for w in
                                        ("before", "inside_small", "inside_large", "after")]}}]}
    write_jsonl(tmp_path / "trajectories.jsonl", [traj])
    write_jsonl(tmp_path / "results.jsonl", [{"sample_id": "s1", "correct": True}])
    out = tmp_path / "labels.jsonl"
    label_cmd = [sys.executable, str(ROOT / "scripts/gen_role_sft_from_trajectories.py"),
                 "--config", str(ROOT / "configs/default.yaml"), str(cfg), "--manifest", str(manifest),
                 "--trajectories", str(tmp_path / "trajectories.jsonl"), "--split", "train",
                 "--roles", "verifier_frame", "verifier_where", "--out", str(out)]
    return manifest, cfg, out, label_cmd


@pytest.mark.parametrize("pad,expected", [(3.0, [False, True, True, True, False]),
                                         (0.0, [False, False, True, False, False])])
def test_labels_follow_config_at_evidence_boundary(tmp_path, pad, expected):
    manifest, cfg, out, label_cmd = evidence_run(tmp_path, pad)
    # Omit the overlay in the default case: exercise the actual default, not a CLI value of 3.
    if pad == 3.0:
        label_cmd.remove(str(cfg))
    subprocess.run(label_cmd, check=True, capture_output=True, text=True)
    rows = [json.loads(l) for l in out.read_text().splitlines()]
    frames = [r for r in rows if r["source"] == "evidence_frame"]
    assert [r["meta"]["evidence"] for r in frames] == expected
    assert [r["target"]["indices"] for r in frames] == [[0] if yes else [1] for yes in expected]
    where = next(r for r in rows if r["source"] == "evidence_where")
    assert where["meta"]["where"] == "before"  # old +/-15 s would put evidence inside this window


def test_old_15_second_trajectories_cannot_be_relabelled_as_3_seconds(tmp_path):
    _, cfg, out, label_cmd = evidence_run(tmp_path, 15.0)
    label_cmd.remove(str(cfg))  # current labels default to 3, but the saved collection config still says 15
    proc = subprocess.run(label_cmd, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "budget.evidence_pad=15.0 but the labels would use 3.0" in proc.stdout + proc.stderr
    assert not out.exists()
