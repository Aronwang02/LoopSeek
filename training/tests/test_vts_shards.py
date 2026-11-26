"""--shard of scripts/filter_cgbench_vts.py (data step on every GPU): shards + merge give the unsharded result, and a
resume skips stages already passed. Fake answerer and video, CPU, no torch.

python -m unittest discover -s tests -p 'test_vts_shards.py' -v
"""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import filter_cgbench_vts as vts  # noqa: E402

IDS = ["good", "text_easy", "noclue", "leak", "multi", "good2", "text2", "good3", "noclue2", "good4", "leak2"]


def rows():
    out = []
    for i, sid in enumerate(IDS):
        spans = [[100, 110], [300, 310]] if sid == "multi" else [[100, 110]]
        out.append({"sample_id": sid, "split": "train", "video_path": f"v{i}.mp4", "question": sid,
                    "options": {"A": "a", "B": "b"}, "answer": "B", "duration": 600.0, "key_spans": spans})
    return out


class Asset:
    def indices_at(self, times):
        return list(range(len(times)))

    def frames(self, idx):
        return {i: f"img{i}" for i in idx}

    def time(self, i):
        return float(i)


class Reader:
    """Text only: right iff 'text' in the id; clue frames (<= 25, two 10-s segments for 'multi' with --multi-segment):
    right unless 'noclue'; outside (256): right iff 'leak'.
    fail_at: frame-count kinds that get no answer (an interrupted earlier run)."""
    calls = []

    def __init__(self, fail_at=()):
        self.fail_at = fail_at

    def run(self, jobs):
        out = []
        for j in jobs:
            sid, n = j["question"], len(j["frames"])
            kind = "text" if n == 0 else "clue" if n <= 25 else "outside"
            Reader.calls.append((sid, kind))
            if kind in self.fail_at:
                out.append(None)
                continue
            ok = {"text": "text" in sid, "clue": "noclue" not in sid, "outside": "leak" in sid}[kind]
            out.append({"parsed": "B" if ok else "A"})
        return out


def run(out, manifest, *extra, fail_at=()):
    fake = {"src.reader.qwen_reader": types.SimpleNamespace(build_reader=lambda cfg, device: Reader(fail_at)),
            "src.rollout.build": types.SimpleNamespace(AssetPool=lambda cfg: types.SimpleNamespace(get=lambda p: Asset())),
            "src.common.runtime": types.SimpleNamespace()}               # no W&B summary in tests
    argv = ["filter_cgbench_vts.py", "--manifest", str(manifest), "--split", "train", "--out", str(out), *extra]
    with patch.dict(sys.modules, fake), patch.object(sys, "argv", argv):
        vts.main()


class Shards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.manifest = self.dir / "all.jsonl"
        self.manifest.write_text("".join(json.dumps(r) + "\n" for r in rows()))
        Reader.calls = []

    def tearDown(self):
        self.tmp.cleanup()

    def kept(self, out):
        return [json.loads(l)["sample_id"] for l in Path(out).read_text().splitlines()]

    def test_shards_then_merge_equal_one_unsharded_run(self):
        run(self.dir / "one.jsonl", self.manifest)
        single = list(Reader.calls)
        Reader.calls = []
        out = self.dir / "sharded.jsonl"
        for i in range(3):
            run(out, self.manifest, "--shard", f"{i}/3")
            self.assertFalse(out.exists())                                # shards never write the manifest
        self.assertEqual(sorted(Reader.calls), sorted(single))            # every question asked once overall
        Reader.calls = []
        run(out, self.manifest)                                           # merge
        self.assertEqual(Reader.calls, [])
        self.assertEqual(self.kept(out), self.kept(self.dir / "one.jsonl"))
        self.assertEqual(self.kept(out), ["good", "good2", "good3", "good4"])
        a, b = (json.loads(Path(f"{p}.stats.json").read_text()) for p in (self.dir / "one.jsonl", out))
        self.assertEqual((a["input"], a["kept"], a["per_stage"]), (b["input"], b["kept"], b["per_stage"]))

    def test_tagged_shard_logs_of_two_jobs_merge_like_one_run(self):
        """--log-tag: jobs sharing a split write their own logs, which never collide on shared storage; the plain run
        reads them all and asks nothing again."""
        run(self.dir / "one.jsonl", self.manifest)
        out = self.dir / "dist.jsonl"
        pieces = ((0, "n0"), (1, "n0"), (2, "n1"), (3, "n1"))
        for k, tag in pieces:
            run(out, self.manifest, "--shard", f"{k}/4", "--log-tag", tag)
        self.assertEqual(sorted(p.name for p in self.dir.glob("dist.jsonl.log.*")),
                         [f"dist.jsonl.log.shard{k}of4.{t}.jsonl" for k, t in pieces])
        Reader.calls = []
        run(out, self.manifest)
        self.assertEqual(Reader.calls, [])
        self.assertEqual(self.kept(out), self.kept(self.dir / "one.jsonl"))

    def test_stages_123_over_logs_with_stage_4_keep_what_passed_stage_3(self):
        """2026-10-21 (VideoITG without stage 4): --stages 123 reuses logs that also hold stage 4; a stage-4 drop does
        not decide the question there, and nothing is asked again."""
        out = self.dir / "s.jsonl"
        run(out, self.manifest)
        self.assertEqual(self.kept(out), ["good", "good2", "good3", "good4"])
        Reader.calls = []
        run(out, self.manifest, "--stages", "123")
        self.assertEqual(Reader.calls, [])
        self.assertEqual(self.kept(out), ["good", "leak", "good2", "good3", "good4", "leak2"])
        stats = json.loads(Path(f"{out}.stats.json").read_text())["per_stage"]
        self.assertNotIn("stage4_dropped", stats)

    def test_shards_by_video_hold_whole_videos(self):
        """--shard-by video: every question of a video lands in one shard, all shards together
        decide every question once."""
        manifest = self.dir / "paired.jsonl"
        manifest.write_text("".join(json.dumps(dict(r, video_path=f"v{i // 2}.mp4")) + "\n" for i, r in enumerate(rows())))
        out = self.dir / "byvideo.jsonl"
        for k in range(3):
            run(out, manifest, "--shard", f"{k}/3", "--shard-by", "video")
        video = {r["sample_id"]: f"v{i // 2}" for i, r in enumerate(rows())}
        seen = []
        for k in range(3):
            ids = {json.loads(l)["sample_id"] for l in Path(f"{out}.log.shard{k}of3.jsonl").read_text().splitlines()}
            seen.append({video[s] for s in ids})
            self.assertEqual(ids, {s for s, v in video.items() if v in seen[-1]})   # whole videos only
        self.assertEqual(sorted(v for s in seen for v in s), sorted(set(video.values())))   # each video in one shard
        Reader.calls = []
        run(out, manifest)
        self.assertEqual(Reader.calls, [])

    def test_resume_skips_passed_stages_and_survives_a_cut_log_line(self):
        out = self.dir / "m.jsonl"
        run(out, self.manifest, fail_at=("clue",))                       # interrupted in stage 3: no verdicts there
        with open(f"{out}.log.jsonl", "a") as f:
            f.write('{"sample_id": "good", "sta')                         # killed mid-write
        Reader.calls = []
        for i in range(2):
            run(out, self.manifest, "--shard", f"{i}/2")
        self.assertNotIn("text", {k for _, k in Reader.calls})            # stage 2 verdicts reused
        self.assertEqual(sorted(s for s, k in Reader.calls if k == "clue"),
                         sorted(s for s in IDS if s not in ("multi", "text_easy", "text2")))
        run(out, self.manifest)
        self.assertEqual(self.kept(out), ["good", "good2", "good3", "good4"])
        stats = json.loads(Path(f"{out}.stats.json").read_text())["per_stage"]
        self.assertEqual(stats["stage2_kept"], 8)                         # not double counted
        self.assertEqual(stats["stage3_no_answer"], 8)

    def test_tag_only_writes_every_runnable_row_with_its_verdict(self):
        """dev (2026-10-20): no row is dropped except one that cannot be run; each carries the verdict it would get."""
        manifest = self.dir / "dev.jsonl"
        bad = dict(rows()[0], sample_id="no_clue", key_spans=[])
        manifest.write_text("".join(json.dumps(r) + "\n" for r in rows() + [bad]))
        out = self.dir / "tagged.jsonl"
        run(out, manifest, "--tag-only")
        got = {json.loads(l)["sample_id"]: json.loads(l)["vts"] for l in out.read_text().splitlines()}
        self.assertEqual(sorted(got), sorted(IDS))                        # all 11, the invalid one left out
        self.assertEqual(sorted(s for s, v in got.items() if v["kept"]), ["good", "good2", "good3", "good4"])
        self.assertEqual({s: v["dropped_at"] for s, v in got.items() if not v["kept"]},
                         {"text_easy": "2", "text2": "2", "noclue": "3", "noclue2": "3", "leak": "4", "leak2": "4",
                          "multi": "1"})                                  # two clue segments 190 s apart (v4)
        self.assertEqual(got["multi"]["reason"], "multi_segment")
        stats = json.loads(Path(f"{out}.stats.json").read_text())
        self.assertEqual((stats["kept"], stats["written"], stats["tag_only"]), (4, 11, True))
        from src.common.config import load_config
        cfg = load_config([str(ROOT / "configs/default.yaml")])
        plain = types.SimpleNamespace(stages="1234", fps=1.0, max_frames=256, max_cover=0.3, merge_spans=False, limit=None)
        tagged = types.SimpleNamespace(**{**vars(plain), "tag_only": True})
        self.assertNotEqual(vts.cache_key(rows(), plain, cfg)[0], vts.cache_key(rows(), tagged, cfg)[0])

    def test_multi_segment_keeps_and_marks_multi_segment_clues(self):
        """--multi-segment (the VideoITG experiment, 2026-10-20): 'multi' passes with frames in both segments, every
        written row is marked for the evidence expert / where labels, and its verdicts and cache entry stay apart."""
        out = self.dir / "multi.jsonl"
        run(out, self.manifest, "--multi-segment")
        kept = [json.loads(l) for l in out.read_text().splitlines()]
        self.assertEqual([r["sample_id"] for r in kept], ["good", "multi", "good2", "good3", "good4"])
        self.assertTrue(all(r["evidence_multi_segment"] for r in kept))
        self.assertTrue(all(json.loads(l).get("multi") for l in Path(f"{out}.log.jsonl").read_text().splitlines()))
        from src.common.config import load_config
        cfg = load_config([str(ROOT / "configs/default.yaml")])
        plain = types.SimpleNamespace(stages="1234", fps=1.0, max_frames=256, max_cover=0.3, merge_spans=False, limit=None)
        multi = types.SimpleNamespace(**{**vars(plain), "multi_segment": True})
        self.assertNotEqual(vts.cache_key(rows(), plain, cfg)[0], vts.cache_key(rows(), multi, cfg)[0])


class Cache(unittest.TestCase):
    """--cache (VTS_CACHE): a complete result is stored once and restored by any later run with the same key."""
    setUp, tearDown = Shards.setUp, Shards.tearDown

    def test_complete_result_is_restored_by_another_run_without_answering(self):
        cache = self.dir / "cache"
        run(self.dir / "a/m.jsonl", self.manifest, "--cache", str(cache))
        self.assertEqual(len(list(cache.glob("*/key.json"))), 1)
        Reader.calls = []
        out = self.dir / "b/m.jsonl"                                      # another TAG's manifest directory
        run(out, self.manifest, "--cache", str(cache), "--cache-restore")
        self.assertEqual(Reader.calls, [])
        self.assertEqual(out.read_text(), (self.dir / "a/m.jsonl").read_text())
        stats = json.loads(Path(f"{out}.stats.json").read_text())
        self.assertEqual((stats["kept"], stats["cache"]), (4, str(next(cache.iterdir()))))

    def test_restore_only_exits_3_on_a_miss_and_incomplete_results_are_not_stored(self):
        cache, out = self.dir / "cache", self.dir / "m.jsonl"
        with self.assertRaises(SystemExit) as e:
            run(out, self.manifest, "--cache", str(cache), "--cache-restore")
        self.assertEqual((e.exception.code, Reader.calls, out.exists()), (3, [], False))
        run(out, self.manifest, "--cache", str(cache), fail_at=("clue",))   # stage-3 answers missing
        self.assertEqual(list(cache.glob("*/key.json")), [])
        for i in range(2):
            run(out, self.manifest, "--shard", f"{i}/2", "--cache", str(cache))   # shards neither restore nor store
        self.assertEqual(list(cache.glob("*/key.json")), [])
        run(out, self.manifest, "--cache", str(cache))                   # merge with every verdict: stored
        self.assertEqual(len(list(cache.glob("*/key.json"))), 1)

    def test_key_follows_data_stages_and_answerer_not_execution(self):
        from src.common.config import load_config
        cfg = load_config([str(ROOT / "configs/default.yaml")])
        args = types.SimpleNamespace(stages="1234", fps=1.0, max_frames=256, max_cover=0.3, merge_spans=False, limit=None)
        key = vts.cache_key(rows(), args, cfg)[0]

        def changed(**reader):
            c = json.loads(json.dumps(cfg))
            for k, v in reader.items():
                c["reader"][k] = v
            return c
        same = [changed(batch_size=1, device="cuda:3"),
                changed(vllm={**cfg["reader"]["vllm"], "gpu_memory_utilization": 0.3, "max_model_len": 4096})]
        self.assertTrue(all(vts.cache_key(rows(), args, c)[0] == key for c in same))
        other_rows = rows()
        other_rows[0]["answer"] = "A"
        differ = [vts.cache_key(other_rows, args, cfg), vts.cache_key(rows(), types.SimpleNamespace(**{**vars(args), "max_cover": 0.2}), cfg),
                  vts.cache_key(rows(), types.SimpleNamespace(**{**vars(args), "merge_spans": True}), cfg),
                  vts.cache_key(rows(), args, changed(engine="hf")), vts.cache_key(rows(), args, changed(answer_prompt="cot"))]
        self.assertTrue(all(k != key for k, _ in differ))
        revised = json.loads(json.dumps(cfg))
        revised["models"][cfg["reader"]["model"]] = {"path": "/elsewhere", "revision": "abc"}
        self.assertNotEqual(vts.cache_key(rows(), args, revised)[0], key)
        moved = json.loads(json.dumps(cfg))
        moved["models"][cfg["reader"]["model"]]["path"] = "/node2/snapshot"   # node-local path only
        self.assertEqual(vts.cache_key(rows(), args, moved)[0], key)


if __name__ == "__main__":
    unittest.main()
