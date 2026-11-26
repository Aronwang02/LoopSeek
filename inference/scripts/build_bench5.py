"""The five-benchmark evaluation manifest: one row per question with a "bench" field:
  cgbench_mini     CG-Bench mini, 3,000 q (the split=test rows of scripts/build_cgbench.py's cgbench_all.jsonl)
  mlvu             MLVU dev, 7 multiple-choice tasks, 2,174 q (scripts/prepare_eval_benches.py mlvu)
  longvideobench   LongVideoBench val, 1,337 q (prepare_eval_benches.py longvideobench)
  videomme_short / videomme_medium / videomme_long   Video-MME without subtitles, 900 q each (scripts/prepare_videomme.py)
  lvbench          LVBench, 1,549 q (prepare_eval_benches.py lvbench)
Every count must match and every video must exist. Writes <out>/bench5.jsonl (10,760 q), <out>/lvbench.jsonl (the
LVBench rows alone, for its uniform-32 baseline) and the smoke subsets bench5_smoke.jsonl (2 q per bench) and
lvbench_smoke.jsonl (2 q). CPU only.
usage: python scripts/build_bench5.py --cgbench-all <cgbench_all.jsonl>
           --videomme-dir <dir with short.jsonl, medium.jsonl, long.jsonl> --mlvu <mlvu_dev.jsonl> --lvb <longvideobench_val.jsonl>
           --lvbench <lvbench.jsonl> --out <dir> [--no-video-check]"""
import argparse
import collections
import json
import os
from pathlib import Path

EXPECTED = {"cgbench_mini": 3000, "mlvu": 2174, "longvideobench": 1337, "videomme_short": 900, "videomme_medium": 900,
            "videomme_long": 900, "lvbench": 1549}


def jsonl(p):
    return [json.loads(line) for line in open(p) if line.strip()]


def assemble(sources, check_videos=True):
    """sources: {bench: rows}; returns the rows with "bench" set, in bench order, after the count / id / video checks."""
    out, problems = [], []
    for bench, rows in sources.items():
        rows = [r for r in rows if r.get("split") != "unavailable"]
        if len(rows) != EXPECTED[bench]:
            problems.append(f"{bench}: {len(rows)} questions, expected {EXPECTED[bench]}")
        for r in rows:
            for k in ("sample_id", "video_path", "options", "answer", "group_key"):
                if k not in r:
                    problems.append(f"{bench} {r.get('sample_id')}: no {k}")
                    break
            else:
                if r["answer"] not in r["options"]:
                    problems.append(f"{bench} {r['sample_id']}: answer {r['answer']} not among the options")
            out.append(dict(r, bench=bench))
    ids = collections.Counter(r["sample_id"] for r in out)
    problems += [f"duplicate sample_id {s}" for s, n in ids.items() if n > 1][:5]
    if check_videos:
        missing = sorted({r["video_path"] for r in out if not os.path.isfile(r["video_path"])})
        if missing:
            problems.append(f"{len(missing)} videos missing, e.g. {missing[:3]}")
    if problems:
        raise SystemExit("build_bench5.py:\n  " + "\n  ".join(problems[:20]))
    return out


def smoke(rows, per_bench=2):
    seen, out = collections.Counter(), []
    for r in rows:
        if seen[r["bench"]] < per_bench:
            seen[r["bench"]] += 1
            out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cgbench-all", required=True)
    ap.add_argument("--videomme-dir", required=True)
    ap.add_argument("--mlvu", required=True)
    ap.add_argument("--lvb", required=True)
    ap.add_argument("--lvbench", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-video-check", action="store_true")
    a = ap.parse_args()
    vm = Path(a.videomme_dir)
    sources = {"cgbench_mini": [r for r in jsonl(a.cgbench_all) if r["split"] == "test"],
               "mlvu": jsonl(a.mlvu), "longvideobench": jsonl(a.lvb),
               "videomme_short": jsonl(vm / "short.jsonl"), "videomme_medium": jsonl(vm / "medium.jsonl"),
               "videomme_long": jsonl(vm / "long.jsonl"), "lvbench": jsonl(a.lvbench)}
    rows = assemble(sources, check_videos=not a.no_video_check)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lv = [r for r in rows if r["bench"] == "lvbench"]
    for name, part in (("bench5.jsonl", rows), ("lvbench.jsonl", lv), ("bench5_smoke.jsonl", smoke(rows)),
                       ("lvbench_smoke.jsonl", lv[:2])):
        (out / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part))
    print(json.dumps({"bench5": len(rows), "by_bench": dict(collections.Counter(r["bench"] for r in rows)),
                      "videos": len({r["video_path"] for r in rows}), "out": str(out)}))


if __name__ == "__main__":
    main()
