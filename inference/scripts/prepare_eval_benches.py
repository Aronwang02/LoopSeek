"""Evaluation data of the five-benchmark evaluation: MLVU dev, LongVideoBench val and LVBench, downloaded straight
from Hugging Face at pinned revisions
(only the videos the questions use), checked, and turned into manifests. Video-MME comes from scripts/prepare_videomme.py
(short / medium / long), CG-Bench mini from the CG-Bench download (scripts/build_bench5.py puts all five together). A rerun skips finished videos, so an interrupted run resumes when started again.
  mlvu            MLVU/MVLU (gated: HF_TOKEN of an account that accepted its terms): the 7 multiple-choice tasks of dev
                  (MLVU/json/1_plotQA .. 7_topic_reasoning, 2,174 questions, 1,242 task/video files, ~198 GiB); each
                  file's size checked against the repo listing.
  longvideobench  longvideobench/LongVideoBench (gated): val (lvb_val.json, 1,337 questions, 753 videos, ~28.7 GB). The
                  31 tar parts are read as one seekable remote file, so only the tar headers and the val videos are
                  downloaded; each video's size checked against its tar header, its duration (PyAV) against
                  lvb_val.json (2 s or 2%).
  lvbench         annotations zai-org/LVBench (video_info.meta.jsonl: 103 videos, 1,549 questions, 4 options each); the
                  videos are YouTube's, taken from the Hugging Face mirror AIWinter/LVBench (mirror + check;
                  14 parts of one zip, 61.8 GB, provenance not documented), read as one seekable remote zip (CRC checked
                  by zipfile); every video must be there and last within 1% of the annotation's duration_minutes.
Manifests (--out, one row per question, the fields scripts/build_bench5.py reads): mlvu_dev.jsonl,
longvideobench_val.jsonl, lvbench.jsonl.
usage: python scripts/prepare_eval_benches.py {mlvu|longvideobench|lvbench} --root <data dir of that bench> --out <dir>
           [--workers 8] [--dry] [--build-only]"""
import argparse
import bisect
import collections
import io
import json
import os
import re
import string
import sys
import tarfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

MLVU = ("MLVU/MVLU", "06ddc388aa34746b3abba77972b7a7dfd977f7a3")
MLVU_TASKS = ["1_plotQA", "2_needle", "3_ego", "4_count", "5_order", "6_anomaly_reco", "7_topic_reasoning"]
LVB = ("longvideobench/LongVideoBench", "60d1c89c1919a198b73be39c2babb213b29d6a5c")
LVBENCH_ANN = ("zai-org/LVBench", "0caedb92002cc268bad486449e551c76f0485670")
LVBENCH_VID = ("AIWinter/LVBench", "fae6be0396740e5de4db0acaa950ca98dbdb4a77")


def log(**kw):
    print(json.dumps({"t": time.strftime("%F %T"), **kw}, ensure_ascii=False), flush=True)


def write_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    os.replace(tmp, path)
    log(written=str(path), rows=len(rows))


def video_seconds(path):
    import av
    with av.open(str(path)) as c:
        return float(c.duration) / 1e6 if c.duration else None


class ConcatRemote(io.RawIOBase):
    """Files of a Hugging Face repo revision (the parts of one archive) as one seekable byte stream: one fsspec handle
    per part, opened when first touched."""

    def __init__(self, repo, revision, parts):
        from huggingface_hub import HfFileSystem
        self.fs, self.repo, self.rev, self.parts, self.h = HfFileSystem(), repo, revision, parts, {}
        self.starts, s = [], 0
        for _, size in parts:
            self.starts.append(s)
            s += size
        self.total, self.pos = s, 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.total + off
        return self.pos

    def readinto(self, b):
        if self.pos >= self.total:
            return 0
        i = bisect.bisect_right(self.starts, self.pos) - 1
        off = self.pos - self.starts[i]
        n = min(len(b), self.parts[i][1] - off)        # never past the end of this part; BufferedReader asks again
        if i not in self.h:
            self.h[i] = self.fs.open(f"datasets/{self.repo}@{self.rev}/{self.parts[i][0]}", "rb", block_size=2 ** 16)
        self.h[i].seek(off)
        data = self.h[i].read(n)
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


def repo_files(repo, rev, prefix=None):
    from huggingface_hub import HfApi
    return {f.path: f.size for f in HfApi().list_repo_tree(repo, repo_type="dataset", revision=rev, path_in_repo=prefix,
                                                            recursive=True) if getattr(f, "size", None)}


# --- MLVU dev ---------------------------------------------------------------------------------------------------------
def mlvu_rows(root):
    rows, miss, dup = [], 0, []
    for t in MLVU_TASKS:
        for i, x in enumerate(json.load(open(root / "MLVU/json" / f"{t}.json"))):
            p = root / "MLVU/video" / t / x["video"]
            miss += not p.is_file()
            cand = [str(c) for c in x["candidates"]]
            letters = string.ascii_uppercase[:len(cand)]
            k = [c.strip() for c in cand].index(str(x["answer"]).strip())     # first occurrence (5 list it twice)
            d = sum(c.strip() == str(x["answer"]).strip() for c in cand) > 1
            sid = f"mlvu:{t}:{i:04d}"
            if d:
                dup.append(sid)
            rows.append({"sample_id": sid, "video_id": os.path.splitext(x["video"])[0], "group_key": f"mlvu:{x['video']}",
                         "video_path": str(p), "question": x["question"], "options": dict(zip(letters, cand)),
                         "answer": letters[k], "source": "mlvu", "subset": f"mlvu_{t}", "split": "dev",
                         "duration": float(x["duration"]), "question_type": x["question_type"], "duplicate_options": d})
    if miss:
        raise SystemExit(f"MLVU: {miss} videos missing under {root}")
    assert len({r["sample_id"] for r in rows}) == len(rows)
    log(mlvu_rows=len(rows), videos=len({r["group_key"] for r in rows}), duplicate_options=dup,
        subsets=dict(collections.Counter(r["subset"] for r in rows)))
    return rows


def run_mlvu(root, workers, dry, build_only):
    from huggingface_hub import hf_hub_download
    repo, rev = MLVU
    if not build_only:
        need = set()
        for t in MLVU_TASKS:
            p = hf_hub_download(repo, f"MLVU/json/{t}.json", repo_type="dataset", revision=rev, local_dir=root)
            need |= {f"MLVU/video/{t}/{x['video']}" for x in json.load(open(p))}
        sizes = repo_files(repo, rev, "MLVU/video")
        absent = sorted(need - set(sizes))
        log(what="mlvu", files=len(need), not_in_repo=absent[:10], gib=round(sum(sizes.get(p, 0) for p in need) / 2 ** 30, 1))
        if absent:
            raise SystemExit("MLVU: question videos missing from the repo listing")
        if dry:
            return
        todo = [p for p in sorted(need) if not ((root / p).is_file() and (root / p).stat().st_size == sizes[p])]

        def one(path):
            for attempt in range(1, 4):
                try:
                    q = hf_hub_download(repo, path, repo_type="dataset", revision=rev, local_dir=root)
                    if os.path.getsize(q) != sizes[path]:
                        raise OSError(f"size {os.path.getsize(q)} != {sizes[path]}")
                    return sizes[path]
                except Exception as e:                    # network hiccup: retried, finished files are kept
                    log(file=path, attempt=attempt, error=f"{type(e).__name__}: {str(e)[:200]}")
                    time.sleep(30 * attempt)
            raise RuntimeError(f"{path}: 3 attempts failed")
        done = 0.0
        with ThreadPoolExecutor(workers) as ex:
            for i, fu in enumerate(as_completed([ex.submit(one, p) for p in todo]), 1):
                done += fu.result() / 2 ** 30
                if i % 50 == 0 or i == len(todo):
                    log(what="mlvu", files_done=f"{i}/{len(todo)}", gib=round(done, 1))
        bad = [p for p in sorted(need) if not ((root / p).is_file() and (root / p).stat().st_size == sizes[p])]
        if bad:
            raise SystemExit(f"MLVU: {len(bad)} files incomplete, e.g. {bad[:3]}")
    return mlvu_rows(root)


# --- LongVideoBench val -------------------------------------------------------------------------------------------------
def lvb_rows(root):
    val = json.load(open(root / "lvb_val.json"))
    rows, miss = [], 0
    for x in val:
        p = root / "videos" / x["video_path"]
        miss += not p.is_file()
        letters = string.ascii_uppercase[:len(x["candidates"])]
        rows.append({"sample_id": f"lvb:{x['id']}", "video_id": x["video_id"], "group_key": f"lvb:{x['video_id']}",
                     "video_path": str(p), "question": x["question"], "options": dict(zip(letters, x["candidates"])),
                     "answer": letters[int(x["correct_choice"])], "source": "longvideobench",
                     "subset": f"lvb_{x['duration_group']}", "split": "val", "duration_group": x["duration_group"],
                     "duration": float(x["duration"]), "question_category": x["question_category"], "level": x["level"],
                     "topic_category": x["topic_category"]})
    if miss:
        raise SystemExit(f"LongVideoBench: {miss} val videos missing under {root}/videos")
    assert len({r["sample_id"] for r in rows}) == len(rows)
    log(lvb_rows=len(rows), videos=len({r["video_id"] for r in rows}),
        subsets=dict(sorted(collections.Counter(r["subset"] for r in rows).items())))
    return rows


def run_lvb(root, workers, dry, build_only):
    from huggingface_hub import hf_hub_download
    repo, rev = LVB
    if not build_only:
        val = json.load(open(hf_hub_download(repo, "lvb_val.json", repo_type="dataset", revision=rev, local_dir=root)))
        want = {x["video_path"] for x in val}
        dur = {x["video_path"]: float(x["duration"]) for x in val}
        parts = sorted((p, s) for p, s in repo_files(repo, rev).items() if p.startswith("videos.tar.part."))
        log(what="longvideobench", val_questions=len(val), val_videos=len(want), parts=len(parts),
            tar_gib=round(sum(s for _, s in parts) / 2 ** 30, 1))
        if dry:
            return
        (root / "videos").mkdir(parents=True, exist_ok=True)
        for attempt in range(1, 4):
            left = {v for v in want if not (root / "videos" / v).is_file()}
            if not left:
                break
            try:
                got = members = 0
                with tarfile.open(fileobj=io.BufferedReader(ConcatRemote(repo, rev, parts), buffer_size=2 ** 16),
                                  mode="r:") as tf:
                    for ti in tf:
                        members += 1
                        name = os.path.basename(ti.name)
                        if not ti.isfile() or name not in left:
                            continue
                        out = root / "videos" / name
                        src = tf.extractfile(ti)
                        with open(str(out) + ".part", "wb") as d:
                            while True:
                                b = src.read(2 ** 26)
                                if not b:
                                    break
                                d.write(b)
                        if os.path.getsize(str(out) + ".part") != ti.size:
                            raise OSError(f"{name}: size differs from its tar header")
                        os.replace(str(out) + ".part", out)
                        left.discard(name)
                        got += 1
                        if got % 25 == 0:
                            log(what="longvideobench", extracted=got, members_seen=members, left=len(left))
                        if not left:
                            break
                log(what="longvideobench", attempt=attempt, extracted=got, members_seen=members, left=len(left))
            except Exception as e:                        # network hiccup: scanned again, finished videos are skipped
                log(what="longvideobench", attempt=attempt, error=f"{type(e).__name__}: {str(e)[:200]}")
                time.sleep(30 * attempt)
        bad = []
        for v in sorted(want):
            p = root / "videos" / v
            if not p.is_file():
                bad.append((v, "missing"))
                continue
            try:
                d = video_seconds(p)
            except Exception as e:
                bad.append((v, type(e).__name__))
                continue
            if d is None or abs(d - dur[v]) > max(2.0, 0.02 * dur[v]):
                bad.append((v, f"duration {d} vs lvb_val.json {dur[v]}"))
        if bad:
            raise SystemExit(f"LongVideoBench: {len(bad)} val videos bad, e.g. {bad[:5]}")
    return lvb_rows(root)


# --- LVBench ------------------------------------------------------------------------------------------------------------
OPTION = re.compile(r"^\(([A-Z])\)\s*(.*)$")


def lvbench_question(text):
    """'stem\\n(A) x\\n(B) y ...' -> (stem, {letter: option})."""
    stem, opts = [], {}
    for line in text.split("\n"):
        m = OPTION.match(line.strip())
        if m:
            opts[m.group(1)] = m.group(2).strip()
        elif not opts:
            stem.append(line)
    return "\n".join(stem).strip(), opts


def lvbench_rows(root, ann, seconds):
    rows = []
    for v in ann:
        p = root / "videos" / f"{v['key']}.mp4"
        for q in v["qa"]:
            stem, opts = lvbench_question(q["question"])
            if len(opts) < 2 or q["answer"] not in opts:
                raise SystemExit(f"LVBench uid {q['uid']}: options {sorted(opts)} / answer {q['answer']} not parsed")
            rows.append({"sample_id": f"lvbench:{q['uid']}", "video_id": v["key"], "group_key": f"lvbench:{v['key']}",
                         "video_path": str(p), "question": stem, "options": opts, "answer": q["answer"],
                         "source": "lvbench", "subset": f"lvbench_{v['type']}", "split": "test",
                         "duration": round(seconds.get(v["key"]) or 60.0 * v["video_info"]["duration_minutes"], 3),
                         "question_type": q["question_type"], "time_reference": q.get("time_reference"),
                         "video_type": v["type"]})
    assert len({r["sample_id"] for r in rows}) == len(rows)
    log(lvbench_rows=len(rows), videos=len(ann), options=dict(collections.Counter(len(r["options"]) for r in rows)),
        subsets=dict(sorted(collections.Counter(r["subset"] for r in rows).items())))
    return rows


def run_lvbench(root, workers, dry, build_only):
    from huggingface_hub import hf_hub_download
    repo, rev = LVBENCH_ANN
    ann_path = hf_hub_download(repo, "video_info.meta.jsonl", repo_type="dataset", revision=rev, local_dir=root)
    ann = [json.loads(line) for line in open(ann_path) if line.strip()]
    keys = {v["key"]: 60.0 * v["video_info"]["duration_minutes"] for v in ann}
    vids = root / "videos"
    if not build_only:
        vrepo, vrev = LVBENCH_VID
        parts = sorted((p, s) for p, s in repo_files(vrepo, vrev).items() if re.fullmatch(r"all_videos_split\.zip\.\d+", p))
        log(what="lvbench", videos=len(keys), questions=sum(len(v["qa"]) for v in ann), parts=len(parts),
            zip_gib=round(sum(s for _, s in parts) / 2 ** 30, 1))
        if dry:
            return
        vids.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, 4):
            left = {k for k in keys if not (vids / f"{k}.mp4").is_file()}
            if not left:
                break
            try:
                with zipfile.ZipFile(io.BufferedReader(ConcatRemote(vrepo, vrev, parts), buffer_size=2 ** 20)) as zf:
                    infos = [i for i in zf.infolist() if not i.is_dir()]
                    names = collections.Counter(os.path.splitext(os.path.basename(i.filename))[0] for i in infos)
                    log(what="lvbench", zip_members=len(infos), annotated_in_zip=len(set(names) & set(keys)),
                        not_in_zip=sorted(set(keys) - set(names))[:5])
                    for info in infos:
                        stem, ext = os.path.splitext(os.path.basename(info.filename))
                        if stem not in left:
                            continue
                        if ext.lower() != ".mp4":
                            raise SystemExit(f"LVBench mirror: {info.filename} is not an .mp4")
                        out = vids / f"{stem}.mp4"
                        with zf.open(info) as s, open(str(out) + ".part", "wb") as d:   # CRC checked while reading
                            while True:
                                b = s.read(2 ** 26)
                                if not b:
                                    break
                                d.write(b)
                        if os.path.getsize(str(out) + ".part") != info.file_size:
                            raise OSError(f"{stem}: size differs from the zip entry")
                        os.replace(str(out) + ".part", out)
                        left.discard(stem)
                        log(what="lvbench", extracted=stem, gib=round(info.file_size / 2 ** 30, 2), left=len(left))
            except SystemExit:
                raise
            except Exception as e:                        # network hiccup: read again, finished videos are skipped
                log(what="lvbench", attempt=attempt, error=f"{type(e).__name__}: {str(e)[:200]}")
                time.sleep(30 * attempt)
    seconds, bad = {}, []
    for k, want in sorted(keys.items()):                  # every video, within 1% of the annotation
        p = vids / f"{k}.mp4"
        if not p.is_file():
            bad.append((k, "missing"))
            continue
        try:
            d = video_seconds(p)
        except Exception as e:
            bad.append((k, type(e).__name__))
            continue
        seconds[k] = d
        if d is None or abs(d - want) > 0.01 * want:
            bad.append((k, f"duration {d} vs annotation {round(want, 1)}"))
    log(what="lvbench", checked=len(keys), bad=len(bad), examples=bad[:5])
    if bad:
        raise SystemExit(f"LVBench: {len(bad)} videos missing or off the annotated duration, e.g. {bad[:5]}")
    return lvbench_rows(root, ann, seconds)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bench", choices=("mlvu", "longvideobench", "lvbench"))
    ap.add_argument("--root", required=True, help="data directory of this bench (videos go below it)")
    ap.add_argument("--out", required=True, help="manifest directory")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry", action="store_true", help="list what would be downloaded")
    ap.add_argument("--build-only", action="store_true", help="no download: build the manifest from what --root holds")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    run = {"mlvu": run_mlvu, "longvideobench": run_lvb, "lvbench": run_lvbench}[a.bench]
    rows = run(root, a.workers, a.dry, a.build_only)
    if rows is None:
        return
    name = {"mlvu": "mlvu_dev.jsonl", "longvideobench": "longvideobench_val.jsonl", "lvbench": "lvbench.jsonl"}[a.bench]
    write_rows(Path(a.out) / name, rows)


if __name__ == "__main__":
    sys.exit(main())
