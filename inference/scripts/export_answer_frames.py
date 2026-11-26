"""Export the frames our search chose for each question, for an answerer outside this repo's reader (used with
scripts/answer.sh). Input: an agent run's trajectories.jsonl (summary.answer_times =
the timestamps of the frames the answerer is shown: the 24 picked overview frames + the 8 kept frames, ascending) and the
manifest it ran on. Output, one JSON line per question: sample_id, bench, subset, group_key, video_path, duration,
question, options, answer (the label), frame_times. Questions whose run status is not ok are reported and left out.
Answers come back through scripts/import_answers.py, which scores them as this repo scores its own reader.
usage: python scripts/export_answer_frames.py --trajectories runs/loopseek_bench5/trajectories.jsonl \
           --manifest $DATA/manifests/bench5/bench5.jsonl --out runs/loopseek_bench5/answer_frames.jsonl"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common.utils import read_jsonl  # noqa: E402

KEEP = ("sample_id", "bench", "subset", "group_key", "video_path", "duration", "question", "options", "answer")


def export(trajectories, manifest):
    man = {r["sample_id"]: r for r in read_jsonl(manifest)}
    out, bad = [], []
    for t in read_jsonl(trajectories):
        times = (t.get("summary") or {}).get("answer_times")
        if t.get("status", "ok") != "ok" or not times:
            bad.append(t["sample_id"])
            continue
        r = man[t["sample_id"]]
        out.append({**{k: r.get(k) for k in KEEP}, "frame_times": sorted(float(x) for x in times)})
    missing = sorted(set(man) - {o["sample_id"] for o in out} - set(bad))
    return out, bad, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out, bad, missing = export(a.trajectories, a.manifest)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(json.dumps({"exported": len(out), "not_ok": len(bad), "not_in_run": len(missing),
                      "frames_per_question": sorted({len(o["frame_times"]) for o in out}), "out": a.out}))


if __name__ == "__main__":
    main()
