"""CPU: VTS-style filter (scripts/filter_cgbench_vts.py) and the CG-Bench split options of scripts/build_cgbench.py."""
import json
import subprocess
import sys

from conftest import ROOT

sys.path.insert(0, str(ROOT))
from scripts.filter_cgbench_vts import (VTS_VERSION, clue_times, filter_rows, merged, outside_times,  # noqa: E402
                                        read_logs, rule_check)


def row(sid, spans, T=600.0, ans="B"):
    return {"sample_id": sid, "video_path": "v.mp4", "question": "q?", "options": {"A": "a", "B": "b"},
            "answer": ans, "duration": T, "key_spans": spans}


def test_rule_stage():
    assert rule_check(row("a", [[10, 20]])) == (True, "ok")
    assert rule_check(row("b", [[10, 20], [100, 110]]))[1] == "multi_segment"
    assert rule_check(row("c", [[10, 20], [20, 30]])) == (True, "ok")              # touching spans merge
    assert rule_check(row("d", [[0, 200]]))[1] == "high_coverage"                   # 33% of 600 s
    assert rule_check(row("h", [[0, 180]])) == (True, "ok")                        # exactly 30% is kept
    assert rule_check(row("e", []))[1] == "invalid" and rule_check(row("f", [[1, 2]], T=0))[1] == "invalid"
    pieces = row("p", [[454, 465], [466, 475], [476, 480]], T=1347.0)              # pieces 1 s apart: one clue (v4)
    assert rule_check(pieces) == (True, "ok") and rule_check(pieces, merge_gap=0.0)[1] == "multi_segment"


def test_multi_segment_option_keeps_several_segments():
    """--multi-segment (the VideoITG experiment, 2026-10-20): several segments pass stage 1, capped as a union."""
    assert rule_check(row("b", [[10, 20], [100, 110]]), multi_segment=True) == (True, "ok")
    assert rule_check(row("g", [[0, 100], [300, 400]]), multi_segment=True)[1] == "high_coverage"   # union 33% of 600 s


def test_pieces_of_one_event_are_one_clue():
    """2026-10-20: clue segments at most 6 s apart (2 x the 3 s evidence pad) are one segment, so stage 4 never samples a
    frame inside an event that CG-Bench annotated in pieces (cgbench #468: one 26 s explanation, three pieces 1 s apart)."""
    pieces = [[454, 465], [466, 475], [476, 480]]
    assert merged(pieces, 6.0) == [[454.0, 480.0]] and len(merged(pieces)) == 3
    assert not any(454 <= t <= 480 for t in outside_times(1347.0, merged(pieces, 6.0)))
    apart = [[350.8, 354.8], [398.8, 403.2]]                                           # two places 44 s apart stay two
    assert merged(apart, 6.0) == apart
    assert merged([[0, 10], [16, 20]], 6.0) == [[0.0, 20.0]] and len(merged([[0, 10], [16.5, 20]], 6.0)) == 2


def test_frame_times():
    ts = clue_times([(10.0, 20.0)])
    assert len(ts) == 10 and all(10 <= t <= 20 for t in ts)                          # 1 fps inside the clue
    assert len(clue_times([(0.0, 1000.0)])) == 256                                   # capped
    two = clue_times([(10.0, 20.0), (100.0, 105.0)])
    assert len(two) == 15 and sum(t >= 100 for t in two) == 5                        # every segment at 1 fps
    out = outside_times(600.0, [(100.0, 120.0)])
    assert len(out) == 256 and not any(100 <= t <= 120 for t in out)                 # 580 s outside -> capped
    small = outside_times(30.0, [(10.0, 20.0)])
    assert len(small) == 20 and not any(10 <= t <= 20 for t in small)
    gaps = outside_times(60.0, [(10.0, 20.0), (30.0, 40.0)])                         # before, between and after
    assert len(gaps) == 40 and not any(10 <= t <= 20 or 30 <= t <= 40 for t in gaps)


def test_logs_of_another_version_or_segment_setting_are_not_reused(tmp_path):
    log = tmp_path / "m.jsonl.log.jsonl"
    lines = [{"sample_id": "a", "stage": "1", "kept": True, "v": 3},             # v3 kept multi-segment clues
             {"sample_id": "b", "stage": "1", "kept": False},                    # unversioned = v1
             {"sample_id": "c", "stage": "1", "kept": False, "v": VTS_VERSION},
             {"sample_id": "d", "stage": "1", "kept": False, "v": VTS_VERSION, "multi": True}]
    log.write_text("".join(json.dumps(x) + "\n" for x in lines))
    assert read_logs([log], "4") == ({"c": False}, {})
    assert read_logs([log], "4", multi_segment=True) == ({"d": False}, {})


class FakeAsset:
    def indices_at(self, times):
        return list(range(len(times)))

    def frames(self, idx):
        return {i: f"img{i}" for i in idx}

    def time(self, i):
        return float(i)


class FakeReader:
    """Answers by rule: no frames -> 'A' for text-easy questions; clue frames -> right iff the id says so; outside
    frames (many) -> right iff the id says 'leak'."""
    def __init__(self, rows):
        self.rows = {r["question"]: r for r in rows}

    def run(self, jobs):
        out = []
        for j in jobs:
            r = self.rows[j["question"]]
            n = len(j["frames"])
            if n == 0:
                ok = "text" in r["sample_id"]
            elif n <= 15:                                                             # clue-only (10 frames)
                ok = "noclue" not in r["sample_id"]
            else:                                                                     # outside-clue (256 frames)
                ok = "leak" in r["sample_id"]
            out.append({"parsed": r["answer"] if ok else "A" if r["answer"] != "A" else "B"})
        return out


def test_stages_2_to_4_keep_and_drop():
    rows = [dict(row(s, [[100, 110]]), question=s) for s in ("good", "text_easy", "noclue", "leak")]
    log, decided = [], {}
    kept = filter_rows(rows, "234", FakeReader(rows), type("P", (), {"get": lambda self, p: FakeAsset()})(),
                       1.0, 256, 2, lambda r, st, k, **kw: log.append((r["sample_id"], st, k)), decided)
    assert [r["sample_id"] for r in kept] == ["good"]
    assert ("text_easy", "2", False) in log and ("noclue", "3", False) in log and ("leak", "4", False) in log


def test_build_cgbench_splits(tmp_path):
    full = [{"qid": q, "video_uid": f"v{q % 3}", "question": "q", "choices": ["x", "y"], "right_answer": "A",
             "clue_intervals": [[1, 2]], "duration": 100, "domain": "d"} for q in range(10)]
    (tmp_path / "full.json").write_text(json.dumps(full))
    (tmp_path / "mini.json").write_text(json.dumps(full[:3]))
    out = tmp_path / "m.jsonl"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_cgbench.py"), "--json", str(tmp_path / "full.json"),
                    "--video-root", str(tmp_path), "--out", str(out), "--keep-missing", "--mini-json",
                    str(tmp_path / "mini.json"), "--dev-size", "2"], check=True, capture_output=True)
    rows = [json.loads(l) for l in open(out)]
    sp = {r["sample_id"]: r["split"] for r in rows}
    assert [sp[f"cgbench:{q}"] for q in range(3)] == ["test"] * 3
    assert sorted(sp.values()).count("dev") == 2 and sorted(sp.values()).count("train") == 5


def test_answerer_failure_is_not_a_verdict():
    class Broken:
        def run(self, jobs):
            return [None for _ in jobs]
    rows = [dict(row("x", [[100, 110]]), question="x")]
    log, decided = [], {}
    kept = filter_rows(rows, "2", Broken(), None, 1.0, 256, 2, lambda r, st, k, **kw: log.append((st, k, kw)), decided)
    assert kept == [] and "x" not in decided and log[0][1] is None and log[0][2]["error"] == "no_answer"


def test_answer_without_a_letter_matches_nothing():
    """VTS: drop at stages 2 / 4 only when the predicted letter matches the ground truth, at stage 3 when it does not.
    An answer with no legal letter ("cannot tell") is a verdict, not a failure: kept at stages 2 / 4, dropped at 3."""
    class Abstains:
        def run(self, jobs):
            return [{"text": "The frames do not show this.", "parsed": None} for _ in jobs]
    asset_pool = type("P", (), {"get": lambda self, p: FakeAsset()})()
    for stages, want_kept, want_stage in (("2", True, "2"), ("4", True, "4"), ("3", False, "3")):
        rows = [dict(row("x", [[100, 110]]), question="x")]
        log, decided = [], {}
        kept = filter_rows(rows, stages, Abstains(), asset_pool, 1.0, 256, 2,
                           lambda r, st, k, **kw: log.append((st, k, kw)), decided)
        assert [r["sample_id"] for r in kept] == (["x"] if want_kept else [])
        assert log == [(want_stage, want_kept, {"predicted": None, "n_frames": log[0][2]["n_frames"], "unparsed": True})]
        assert decided == ({} if want_kept else {"x": False})


def test_next_batch_is_decoded_while_the_answerer_runs():
    import threading
    rows = [dict(row(f"good{i}", [[100, 110]]), question=f"good{i}", video_path=f"v{i}.mp4") for i in range(6)]
    decoded = {p: threading.Event() for p in (r["video_path"] for r in rows)}

    class Asset(FakeAsset):
        def __init__(self, path):
            self.path = path

        def frames(self, idx):
            decoded[self.path].set()
            return super().frames(idx)

    class Reader(FakeReader):
        def run(self, jobs):
            nxt = {"good0": "v2.mp4", "good2": "v4.mp4"}.get(jobs[0]["question"])
            if nxt is not None:                       # the following batch is being decoded meanwhile
                assert decoded[nxt].wait(5), f"{nxt} was not prefetched during the answerer call"
            return super().run(jobs)

    log, decided = [], {}
    kept = filter_rows(rows, "3", Reader(rows), type("P", (), {"get": lambda self, p: Asset(p)})(), 1.0, 256, 2,
                       lambda r, st, k, **kw: log.append((r["sample_id"], st, k)), decided)
    assert [r["sample_id"] for r in kept] == [r["sample_id"] for r in rows]
    assert [x[0] for x in log] == [r["sample_id"] for r in rows]          # verdicts in the original order
