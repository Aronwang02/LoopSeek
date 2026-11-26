"""Two GPUs per inference process: configs/two_gpu.yaml changes only where System II runs, and run_agent_v3 puts it on
reader.device. Also the export / import of answer frames for answerers outside this repository."""
from pathlib import Path

from src.common.config import load_config

ROOT = Path(__file__).resolve().parents[1]
BASE = ["configs/default.yaml", "configs/loopseek.yaml"]


def flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, f"{prefix}{k}."))
        else:
            out[f"{prefix}{k}"] = v
    return out


def test_a100_overlay_changes_only_reader_placement():
    base = flat(load_config([str(ROOT / p) for p in BASE]))
    a100 = flat(load_config([str(ROOT / p) for p in BASE + ["configs/two_gpu.yaml"]]))
    assert set(base) == set(a100)
    diff = {k for k in base if base[k] != a100[k]}
    assert diff == {"reader.device", "reader.vllm.gpu_memory_utilization"}, diff
    assert a100["reader.device"] == "cuda:1"
    assert a100["reader.vllm.gpu_memory_utilization"] == 0.85


def test_run_agent_reads_reader_device():
    src = (ROOT / "scripts/run_agent_v3.py").read_text()
    assert 'QwenReader(cfg, device=cfg["reader"].get("device") or "cuda:0")' in src
    assert 'QwenReader(cfg, device="cuda:0")' not in src


def _jl(path, rows):
    import json
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_export_answer_frames(tmp_path):
    from scripts.export_answer_frames import export
    man = [{"sample_id": f"b:{i}", "bench": "b", "subset": "s", "group_key": f"b:v{i}", "video_path": f"/v{i}.mp4",
            "duration": 100.0, "question": "q", "options": {"A": "x", "B": "y"}, "answer": "B"} for i in range(3)]
    traj = [{"sample_id": "b:0", "status": "ok", "summary": {"answer_times": [9.0, 1.5, 4.0]}},
            {"sample_id": "b:1", "status": "error", "summary": {}}]
    _jl(tmp_path / "m.jsonl", man)
    _jl(tmp_path / "t.jsonl", traj)
    out, bad, missing = export(tmp_path / "t.jsonl", tmp_path / "m.jsonl")
    assert [o["sample_id"] for o in out] == ["b:0"] and out[0]["frame_times"] == [1.5, 4.0, 9.0]
    assert out[0]["options"] == {"A": "x", "B": "y"} and out[0]["answer"] == "B"
    assert bad == ["b:1"] and missing == ["b:2"]


def test_import_answers_scores_like_the_reader(tmp_path):
    import pytest
    from scripts.import_answers import score
    man = [{"sample_id": f"b:{i}", "group_key": f"b:v{i}", "subset": "s", "options": {"A": "x", "B": "y", "C": "z"},
            "answer": "B"} for i in range(4)]
    _jl(tmp_path / "m.jsonl", man)
    _jl(tmp_path / "a.jsonl", [{"sample_id": "b:0", "reply": "B"}, {"sample_id": "b:1", "reply": "(B) y"},
                               {"sample_id": "b:2", "reply": "The answer is C."}, {"sample_id": "b:3", "reply": "no idea"}])
    rows, missing = score(tmp_path / "a.jsonl", tmp_path / "m.jsonl")
    got = {r["sample_id"]: (r["parsed"], r["correct"]) for r in rows}
    assert got == {"b:0": ("B", True), "b:1": ("B", True), "b:2": ("C", False), "b:3": (None, False)} and missing == []
    _jl(tmp_path / "a2.jsonl", [{"sample_id": "b:0", "reply": "B"}])
    with pytest.raises(SystemExit):
        score(tmp_path / "a2.jsonl", tmp_path / "m.jsonl")
    rows, missing = score(tmp_path / "a2.jsonl", tmp_path / "m.jsonl", allow_missing=True)
    assert len(rows) == 1 and len(missing) == 3
