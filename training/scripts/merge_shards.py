"""Merge the run directories of one collection's shard processes into one (scripts/train_pipeline.sh collect).

With a shared queue (run_agent_v3 --queue, 2026-10-13) a question can sit in any shard's directory. One that two
directories hold (a claim lost when a launch was interrupted) is kept once, from the first directory in the given order;
lines otherwise keep their order, so without duplicates the output equals the concatenation of the shards.
usage: python scripts/merge_shards.py --out runs/<name> runs/<name>_p0 runs/<name>_p1 ...
"""
import argparse
import json
from pathlib import Path


def merge(out, shards):
    owner = {}
    for k, d in enumerate(shards):
        with open(Path(d) / "results.jsonl") as f:
            for line in f:
                if line.strip():
                    owner.setdefault(json.loads(line)["sample_id"], k)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for kind in ("trajectories", "results"):
        n = 0
        with open(out / f"{kind}.jsonl", "w") as w:
            for k, d in enumerate(shards):
                with open(Path(d) / f"{kind}.jsonl") as f:
                    for line in f:
                        if line.strip() and owner.get(json.loads(line)["sample_id"]) == k:
                            w.write(line if line.endswith("\n") else line + "\n")
                            n += 1
        counts[kind] = n
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("shards", nargs="+")
    args = ap.parse_args()
    print(json.dumps({"merged": args.out, **merge(args.out, args.shards)}), flush=True)


if __name__ == "__main__":
    main()
