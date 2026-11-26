"""Mix filtered CG-Bench and LV-Haystack train/dev; no test data or source oversampling."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random


def mix(inputs, output, split, seed):
    output = Path(output)
    if output.resolve() in {Path(p).resolve() for p in inputs}:
        raise ValueError("mixture output must differ from source manifests")
    rows, seen = [], set()
    for file in inputs:
        source = [json.loads(line) for line in Path(file).read_text().splitlines() if line.strip()]
        if not source:
            raise ValueError(f"empty training source: {file}")
        for row in source:
            if row.get("split") != split or row["sample_id"] in seen:
                raise ValueError(f"wrong split or duplicate sample: {row['sample_id']} in {file}")
            seen.add(row["sample_id"])
            rows.append(row)
    random.Random(seed).shuffle(rows)  # RL dev-limit must not see only the first concatenated dataset.
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return {"split": split, "rows": len(rows), "sources": dict(Counter(r["source"] for r in rows))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest-dir", required=True)
    ap.add_argument("--train-out", required=True)
    ap.add_argument("--dev-out", required=True)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()
    root = Path(args.manifest_dir)
    stats = []
    for split, out in (("train", args.train_out), ("dev", args.dev_out)):
        stats.append(mix([root / f"{name}_{split}_vts.jsonl" for name in ("cgbench", "lvhaystack")], out, split, args.seed))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
