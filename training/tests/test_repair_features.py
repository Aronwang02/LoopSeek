"""scripts/repair_features.py (2026-10-16): the frames SFT decisions reference, read over byte ranges of the data file.
CPU only."""
import json

from scripts import repair_features as R


def refs(path, n, types=()):
    need = {}
    for task in R.chunks(path, n):
        got, _ = R.frame_refs((*task, set(types)))
        for cid, pts in got.items():
            need.setdefault(cid, set()).update(pts)
    return need


def test_every_decision_is_read_once_whatever_the_ranges_and_types_filter_it(tmp_path):
    def seg(cid, pts):
        return {"v": [cid, pts, 0.0, 0.0, 0.0]}
    rows = [{"sample_id": f"s{i}", "split": "train", "decision_type": "verifier" if i % 2 else "frame",
             "input": {"prefix": [{"t": "q"}, seg(f"v{i % 3}", i)],
                       "candidates": [[seg(f"v{i % 3}", 100 + i)], [{"t": "none"}]]}} for i in range(50)]
    data = tmp_path / "d.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    want = {}
    for i in range(50):
        want.setdefault(f"v{i % 3}", set()).update({i, 100 + i})
    for n in (1, 3, 7, 64):
        assert refs(data, n) == want
    assert refs(data, 5, ["verifier"]) == {cid: {p for p in pts if p % 2} for cid, pts in want.items()}
