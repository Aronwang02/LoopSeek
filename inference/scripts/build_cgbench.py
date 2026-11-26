"""CG-Bench (HF CG-Bench/CG-Bench, cgbench.json) -> our manifest format (2026-10-06). CPU only.
key_spans = the annotated clue_intervals (seconds); evidence_times = --n-evidence times spread evenly over the total clue
length (the midpoint of each equal share), so E1 (scripts/e1_headroom.py) swaps frames from inside the clues. Options are
the `choices` list lettered A, B, ... (5-8 options); answer = right_answer. Only questions whose video file exists under
--video-root are written unless --keep-missing. CG-Bench has no official train split: its questions outside CG-Bench mini
are the training pool. No deduplication against evaluation sets.
Splits (2026-10-06, aligned with VTS, arXiv 2607.16189): --mini-json cgbench_mini.json marks the mini questions
split=test and the rest split=train (question-level split, videos shared, as the baselines do); --dev-size N moves N
random train questions to split=dev (model selection; never the test set). Then scripts/filter_cgbench_vts.py.
usage: python scripts/build_cgbench.py --json <cgbench.json> --video-root <dir with <video_uid>.mp4> --out <manifest.jsonl>
           [--mini-json cgbench_mini.json --dev-size 200] [--sample 200 --seed 0]
"""
import argparse
import json
import random
import string
from pathlib import Path


def evidence_times(intervals, n):
    total = sum(e - s for s, e in intervals)
    if total <= 0:
        return [float(s) for s, _ in intervals][:n]
    out = []
    for j in range(n):
        x = total * (j + 0.5) / n            # position along the concatenated clue time
        for s, e in intervals:
            if x <= e - s:
                out.append(float(s + x))
                break
            x -= e - s
    return sorted(set(round(t, 3) for t in out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--video-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-evidence", type=int, default=4)
    ap.add_argument("--sample", type=int, default=None, help="random sample of questions (after the existence filter)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-missing", action="store_true")
    ap.add_argument("--mini-json", default=None, help="CG-Bench mini annotation: its questions get split=test")
    ap.add_argument("--dev-size", type=int, default=0, help="random train questions moved to split=dev")
    args = ap.parse_args()
    data = json.load(open(args.json))
    mini = {r["qid"] for r in json.load(open(args.mini_json))} if args.mini_json else None
    rows, missing = [], 0
    for r in data:
        path = Path(args.video_root) / f"{r['video_uid']}.mp4"
        if not path.exists() and not args.keep_missing:
            missing += 1
            continue
        letters = string.ascii_uppercase[:len(r["choices"])]
        iv = [[float(s), float(e)] for s, e in r["clue_intervals"]]
        rows.append({"sample_id": f"cgbench:{r['qid']}", "video_id": r["video_uid"], "group_key": f"cgbench:{r['video_uid']}",
                     "video_path": str(path), "question": r["question"], "options": dict(zip(letters, r["choices"])),
                     "answer": r["right_answer"], "source": "cgbench", "subset": f"cgbench_{r['domain']}",
                     "split": ("test" if (mini is not None and r["qid"] in mini) else "train"),
                     "duration": float(r["duration"]), "key_spans": iv,
                     "evidence_times": evidence_times(iv, args.n_evidence), "evidence_kind": "clue_intervals",
                     "domain": r["domain"], "sub_category": r.get("sub_category")})
    if args.dev_size:
        train = sorted((x for x in rows if x["split"] == "train"), key=lambda x: x["sample_id"])
        for x in random.Random(args.seed).sample(train, min(args.dev_size, len(train))):
            x["split"] = "dev"
    if args.sample and len(rows) > args.sample:
        rows = random.Random(args.seed).sample(rows, args.sample)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    splits = {}
    for x in rows:
        splits[x["split"]] = splits.get(x["split"], 0) + 1
    print(json.dumps({"rows": len(rows), "missing_video": missing, "videos": len({x["video_id"] for x in rows}),
                      "splits": splits}))


if __name__ == "__main__":
    main()
