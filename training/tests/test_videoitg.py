"""CPU: VideoITG-40K multiple-choice questions -> short-video manifests (scripts/build_videoitg.py) and the videos the
fetcher needs (scripts/fetch_videoitg.py). Real question format of the release; a 6.1 s synthetic video stands in."""
import collections
import json
import shutil
import subprocess
import sys

import pytest
from conftest import ROOT

sys.path.insert(0, str(ROOT))
from scripts.build_videoitg import build, evidence_spans, parse_mc, segment_counts  # noqa: E402
from scripts.fetch_videoitg import needed_videos  # noqa: E402

Q = ("What change occurs in the text displayed on the screen from the beginning to the end of the video?\n"
     "A. The text changes from blue to red\nB. The text changes from large to small\n"
     "C. The text changes from humorous to a call for sponsorship\nD. The text changes from English to Russian\n"
     "Please respond with only the letter of the correct answer.")


def test_parse_mc_release_formats():
    stem, options, letter = parse_mc(Q, "C")
    assert stem.startswith("What change occurs") and letter == "C" and list(options) == ["A", "B", "C", "D"]
    assert options["D"] == "The text changes from English to Russian"                     # instruction line dropped
    assert parse_mc(Q.replace("only the letter of the correct answer.", "x"), "B. The text changes")[2] == "B"
    assert parse_mc("What is the main focus of the video?", "A cooking show.") is None     # open-ended
    assert parse_mc(Q, "E") is None                                                         # answer not an option
    assert parse_mc("Which one?\nA. x\nC. y", "A") is None                                  # options out of order


def test_evidence_spans_inside_the_video():
    assert evidence_spans([0, 1, 2, 5, 9], 6.0) == [[0.0, 2.5], [4.5, 5.5]]                # touching seconds merge
    assert evidence_spans([7, 8], 6.0) == []


def test_streamed_chunks_are_recorded_and_skipped_on_resume(tmp_path, monkeypatch):
    """2026-10-20: a preempted VideoITG data job restores its videos and chunk records and streams only the rest."""
    import io
    import tarfile
    import types
    from scripts import llava_video as fsv
    chunks = {}
    for c, names in (("s/s_videos_1.tar.gz", ["a.mp4", "x.mp4"]), ("s/s_videos_2.tar.gz", ["b.mp4"])):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for n in names:
                data = n.encode() * 3
                info = tarfile.TarInfo(n)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        chunks[c] = buf.getvalue()
    streamed = []

    class Resp:
        def __init__(self, url):
            streamed.append(url)
            self.raw = io.BytesIO(chunks[url])

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def raise_for_status(self):
            pass

    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(get=lambda url, **kw: Resp(url)))
    monkeypatch.setattr(fsv, "llava_chunks", lambda subset: [(c, c) for c in chunks])
    monkeypatch.setattr(fsv, "hf_headers", lambda: {})
    want = {"a.mp4": tmp_path / "v/a.mp4", "b.mp4": tmp_path / "v/b.mp4"}
    fsv.fetch_llava({"s": dict(want)}, tmp_path / "state")
    assert streamed == list(chunks) and all(p.read_bytes() == p.name.encode() * 3 for p in want.values())
    assert (tmp_path / "state/s.chunks").read_text().split() == list(chunks)
    want["b.mp4"].unlink()                                        # chunk 2 was recorded but its video is gone:
    (tmp_path / "state/s.chunks").write_text("s/s_videos_1.tar.gz\n")   # only chunk 1's record survived the upload
    streamed.clear()
    fsv.fetch_llava({"s": dict(want)}, tmp_path / "state")
    assert streamed == ["s/s_videos_2.tar.gz"] and want["b.mp4"].exists()


def test_parallel_fetch_streams_every_subset_and_resumes(tmp_path, monkeypatch):
    """fetch_llava_parallel (2026-10-21): chunks of every subset at once, recorded per subset; a subset that has all its
    videos skips its other chunks, and a resumed run streams only unrecorded chunks."""
    import io
    import tarfile
    import threading
    import types
    from scripts import llava_video as fsv
    chunks = {}
    layout = {"s1": [["a.mp4"], ["x.mp4"], ["b.mp4"]], "s2": [["c.mp4"], ["y.mp4"]]}
    for s, parts in layout.items():
        for k, names in enumerate(parts, 1):
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tar:
                for n in names:
                    data = n.encode() * 3
                    info = tarfile.TarInfo(n)
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
            chunks[f"{s}/{s}_videos_{k}.tar.gz"] = buf.getvalue()
    streamed, lock = [], threading.Lock()

    class Resp:
        def __init__(self, url):
            with lock:
                streamed.append(url)
            self.raw = io.BytesIO(chunks[url])

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def raise_for_status(self):
            pass

    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace(get=lambda url, **kw: Resp(url)))
    monkeypatch.setattr(fsv, "llava_chunks", lambda subset: [(c, c) for c in chunks if c.startswith(subset + "/")])
    monkeypatch.setattr(fsv, "hf_headers", lambda: {})
    todo = {"s1": {"a.mp4": tmp_path / "v/a.mp4", "b.mp4": tmp_path / "v/b.mp4"}, "s2": {"c.mp4": tmp_path / "v/c.mp4"}}
    fsv.fetch_llava_parallel(todo, tmp_path / "state", workers=1)       # one worker: the round-robin order is fixed
    assert streamed == ["s1/s1_videos_1.tar.gz", "s2/s2_videos_1.tar.gz", "s1/s1_videos_2.tar.gz",
                        "s1/s1_videos_3.tar.gz"]                         # s2 is complete after its first chunk
    assert all(p.read_bytes() == p.name.encode() * 3 for m in todo.values() for p in m.values())
    assert (tmp_path / "state/s1.chunks").read_text().split() == ["s1/s1_videos_1.tar.gz", "s1/s1_videos_2.tar.gz",
                                                                  "s1/s1_videos_3.tar.gz"]
    todo["s1"]["b.mp4"].unlink()                                         # lost with the upload of s1's last record
    (tmp_path / "state/s1.chunks").write_text("s1/s1_videos_1.tar.gz\ns1/s1_videos_2.tar.gz\n")
    streamed.clear()
    fsv.fetch_llava_parallel(todo, tmp_path / "state", workers=4)
    assert streamed == ["s1/s1_videos_3.tar.gz"] and todo["s1"]["b.mp4"].exists()


def test_sample_one_question_per_video_with_caps():
    """scripts/sample_videoitg.py (2026-10-21): at most one question per video, capped per split, seeded."""
    from scripts.sample_videoitg import sample
    rows = [{"sample_id": f"videoitg:{v}{q}", "video_id": f"v{v}", "split": "dev" if v >= 8 else "train"}
            for v in range(10) for q in range(3)]
    out = sample(rows, {"train": 5, "dev": 0}, per_video=1, seed=0)
    train, dev = [r for r in out if r["split"] == "train"], [r for r in out if r["split"] == "dev"]
    assert len(train) == 5 and len({r["video_id"] for r in train}) == 5            # capped, one per video
    assert len(dev) == 2 and len({r["video_id"] for r in dev}) == 2                # no cap: every dev video once
    assert out == sample(rows, {"train": 5, "dev": 0}, per_video=1, seed=0)        # seeded


def test_binned_sample_keeps_the_duration_shares_at_random_over_videos():
    """--bins: 20% / 30% / 50% over 30-60 s / 1-2 min / 2-3 min, drawn at random (not the first ids),
    one question per video per pass: a category with fewer videos than its quota takes a second question per video."""
    from scripts.sample_videoitg import parse_bins, quotas, sample_binned
    rows = [{"sample_id": f"videoitg:{cat}-{v:03d}-{q}", "video_id": f"{cat}/{v}", "subset": f"{cat}_youtube_v0_1",
             "split": "dev" if v >= nv - 4 else "train"}
            for cat, nv in (("30_60_s", 100), ("1_2_m", 100), ("2_3_m", 40)) for v in range(nv) for q in range(4)]
    shares = parse_bins("30_60_s=0.2,1_2_m=0.3,2_3_m=0.5")
    assert quotas(100, shares) == {"30_60_s": 20, "1_2_m": 30, "2_3_m": 50} and sum(quotas(7, shares).values()) == 7
    out, stats = sample_binned(rows, {"train": 100, "dev": 10}, shares, seed=0)
    train = [r for r in out if r["split"] == "train"]
    by_cat = collections.Counter(r["subset"].split("_youtube")[0] for r in train)
    assert by_cat == {"30_60_s": 20, "1_2_m": 30, "2_3_m": 50}
    per_video = collections.Counter(r["video_id"] for r in train)
    assert max(n for v, n in per_video.items() if not v.startswith("2_3_m")) == 1    # quota < videos: one each
    assert max(n for v, n in per_video.items() if v.startswith("2_3_m")) == 2        # 50 from 36 videos: 1-2 each
    first = sorted(r["sample_id"] for r in rows if r["split"] == "train" and r["subset"].startswith("30_60_s"))[:20]
    assert sorted(r["sample_id"] for r in train if r["subset"].startswith("30_60_s")) != first   # not the first ids
    assert stats["train/2_3_m"]["videos"] == 36 and stats["dev/30_60_s"]["quota"] == 2
    assert (out, stats) == sample_binned(rows, {"train": 100, "dev": 10}, shares, seed=0)          # seeded
    assert out != sample_binned(rows, {"train": 100, "dev": 10}, shares, seed=1)[0]
    with pytest.raises(SystemExit):                                     # too few 2-3 min questions for the quota
        sample_binned(rows, {"train": 400, "dev": 10}, shares, seed=0)


def test_rest_of_a_selection_and_the_most_the_shares_allow(tmp_path):
    """VideoITG filtered in full: --rest-of writes the train questions a selection does not hold; with
    --bins, -1 is the most questions the shares allow (the shortest category of its share sets it); --all joins files."""
    from scripts.sample_videoitg import most_allowed, parse_bins, rest_of, sample_binned
    rows = [{"sample_id": f"videoitg:{cat}-{v:03d}-{q}", "video_id": f"{cat}/{v}", "subset": f"{cat}_youtube_v0_1",
             "split": "dev" if v >= nv - 2 else "train"}
            for cat, nv in (("30_60_s", 50), ("1_2_m", 40), ("2_3_m", 12)) for v in range(nv) for q in range(2)]
    shares = parse_bins("30_60_s=0.2,1_2_m=0.3,2_3_m=0.5")
    chosen, _ = sample_binned(rows, {"train": 20, "dev": 0}, shares, seed=0)
    assert {r["split"] for r in chosen} == {"train"}                      # dev 0: none
    rest = rest_of(rows, chosen)
    train = [r for r in rows if r["split"] == "train"]
    assert len(rest) == len(train) - 20 and not {r["sample_id"] for r in rest} & {r["sample_id"] for r in chosen}
    two_three = sum(r["subset"].startswith("2_3_m") for r in train)      # 20 questions: 50% of 40 at most
    assert most_allowed(train, shares) == 2 * two_three == 40
    most, stats = sample_binned(rows, {"train": -1, "dev": 0}, shares, seed=0)
    assert len(most) == 40 and stats["train/2_3_m"]["picked"] == 20
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text("".join(json.dumps(r) + "\n" for r in chosen))
    b.write_text("".join(json.dumps(r) + "\n" for r in rest + chosen[:3]))   # a repeated row is kept once
    out = tmp_path / "all.jsonl"
    subprocess.run([sys.executable, str(ROOT / "scripts/sample_videoitg.py"), "--manifest", str(a), str(b),
                    "--out", str(out), "--all"], check=True, capture_output=True)
    assert len(out.read_text().splitlines()) == len(train)


def test_segment_counts_after_the_6_s_merge():
    """The statistics that decide the VideoITG multi-segment experiment (2026-10-20)."""
    rows = [{"key_spans": [[9.5, 10.5], [14.5, 15.5]], "duration": 120.0},       # 4 s apart: one segment
            {"key_spans": [[9.5, 10.5], [59.5, 60.5]], "duration": 120.0},       # 49 s apart, span 51 s = 42%
            {"key_spans": [[9.5, 10.5], [19.5, 20.5], [29.5, 30.5]], "duration": 120.0}]   # 9 s gaps: three
    assert segment_counts(rows) == {"clue_segments_1": 1, "clue_segments_2": 1, "clue_segments_3+": 1,
                                    "multi_segment_far_apart": 1, "multi_segment_span_over_30pct": 1}
    rows[1]["duration"] = 600.0
    assert segment_counts(rows)["multi_segment_span_over_30pct"] == 0


def videoitg_row(i, video, frames, question=Q, answer="C"):
    return {"id": i, "video": video, "question": question, "frame_num": frames, "clip_num": [0], "existence": "No",
            "motion": "No", "answer": answer}


def test_build_rows_drops_and_whole_video_dev_split(tmp_path, vfr_video):
    root = tmp_path / "videos"
    vids = [f"LLaVA-Video-178K/30_60_s_youtube_v0_1/liwei_youtube_videos/videos/youtube_video_2024/ytb_v{k}.mp4"
            for k in range(3)]
    for v in vids[:2]:
        (root / v).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(vfr_video, root / v)
    ann = [videoitg_row(1, vids[0], [1, 2]), videoitg_row(2, vids[0], [5]),
           videoitg_row(3, vids[1], [0, 3, 4]),
           videoitg_row(4, vids[1], [3], question="What is the main focus of the video?", answer="A dog."),
           videoitg_row(5, vids[2], [1]),                                                   # video not fetched
           videoitg_row(6, vids[1], [30])]                                                  # beyond the 6.1 s video
    rows, stats = build(ann, root, dev_videos=1, seed=0, workers=2)
    assert stats["dropped_not_multiple_choice"] == 1 and stats["dropped_missing_video"] == 1
    assert stats["dropped_no_evidence_in_video"] == 1 and len(rows) == 3
    r = next(x for x in rows if x["sample_id"] == "videoitg:3")
    assert r["key_spans"] == [[0.0, 0.5], [2.5, 4.5]] and r["evidence_times"] == [0.0, 3.0, 4.0]
    assert r["answer"] == "C" and "Please respond" not in r["question"] and 6.0 < r["duration"] < 6.3
    assert r["source"] == "videoitg" and r["subset"] == "30_60_s_youtube_v0_1"
    dev = {x["video_id"] for x in rows if x["split"] == "dev"}
    assert len(dev) == 1 and not dev & {x["video_id"] for x in rows if x["split"] == "train"}   # whole videos
    assert stats["videos_dev"] == 1 and stats["rows_dev"] + stats["rows_train"] == 3


def test_cli_writes_split_manifests(tmp_path, vfr_video):
    v = "LLaVA-Video-178K/2_3_m_academic_v0_1/academic_source/activitynet/v_x.mp4"
    (tmp_path / "videos" / v).parent.mkdir(parents=True)
    shutil.copyfile(vfr_video, tmp_path / "videos" / v)
    ann = tmp_path / "video_itg_data.json"
    ann.write_text(json.dumps([videoitg_row(i, v, [i]) for i in range(1, 4)]))
    out = tmp_path / "man"
    subprocess.run([sys.executable, str(ROOT / "scripts/build_videoitg.py"), "--annotations", str(ann), "--video-root",
                    str(tmp_path / "videos"), "--out-dir", str(out), "--dev-videos", "0"], check=True,
                   capture_output=True, text=True)
    rows = [json.loads(l) for l in (out / "videoitg_all.jsonl").read_text().splitlines()]
    assert [r["split"] for r in rows] == ["train"] * 3 and (out / "videoitg_dev.jsonl").read_text() == ""
    assert json.loads((out / "videoitg_stats.json").read_text())["rows_train"] == 3


def test_fetcher_needs_the_multiple_choice_videos_only(tmp_path):
    ann = [videoitg_row(1, "LLaVA-Video-178K/1_2_m_youtube_v0_1/liwei_youtube_videos/videos/hdvila/a.mp4", [1]),
           videoitg_row(2, "LLaVA-Video-178K/1_2_m_nextqa/NextQA/NExTVideo/1006/b.mp4", [1],
                        question="What happens?", answer="He waves.")]
    todo = needed_videos(ann, tmp_path)
    assert todo == {"1_2_m_youtube_v0_1": {"liwei_youtube_videos/videos/hdvila/a.mp4":
                                           tmp_path / ann[0]["video"]}}                     # tar member -> local path
