"""Score an outside answerer's replies the way this repo scores its own reader (used with scripts/answer.sh): each reply goes
through src.evaluation.scoring.parse_answer (a leading option letter or "answer is X"; no legal letter -> wrong), and the
rows are written as runs/<name>/b0_results.jsonl, which scripts/bench5_report.py reads like a scripts/run_b0.py run.
Input: JSON lines {"sample_id": ..., "reply": "<the model's text>"} (a bare letter is a reply too). Every question of the
manifest must have a reply (else nothing is written), unless --allow-missing.
usage: python scripts/import_answers.py --answers replies.jsonl --manifest data/manifests/bench5/bench5.jsonl --name ans_x
       python scripts/bench5_report.py --runs runs/ans_x --manifest data/manifests/bench5/bench5.jsonl --out runs/ans_x.json"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.common.utils import read_jsonl  # noqa: E402
from src.evaluation.scoring import is_correct, parse_answer  # noqa: E402


def score(answers, manifest, allow_missing=False):
    man = {r["sample_id"]: r for r in read_jsonl(manifest)}
    replies = {}
    for a in read_jsonl(answers):
        replies[a["sample_id"]] = a.get("reply", a.get("prediction"))
    unknown = sorted(set(replies) - set(man))
    missing = sorted(set(man) - set(replies))
    if unknown:
        raise SystemExit(f"{len(unknown)} replies are for questions not in the manifest, e.g. {unknown[:3]}")
    if missing and not allow_missing:
        raise SystemExit(f"{len(missing)} manifest questions have no reply, e.g. {missing[:3]} (--allow-missing to score the rest)")
    rows = []
    for s, text in replies.items():
        r = man[s]
        parsed = parse_answer(text, r["options"].keys())
        rows.append({"sample_id": s, "video_id": r.get("video_id"), "group_key": r.get("group_key", s),
                     "subset": r.get("subset"), "answer_raw": text, "parsed": parsed,
                     "correct": bool(is_correct(parsed, r["answer"])), "label": r["answer"]})
    return rows, missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--runs-root", default=str(ROOT / "runs"))
    ap.add_argument("--allow-missing", action="store_true")
    a = ap.parse_args()
    rows, missing = score(a.answers, a.manifest, a.allow_missing)
    out = Path(a.runs_root) / a.name
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "b0_results.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps({"scored": len(rows), "missing": len(missing), "parse_failures": sum(r["parsed"] is None for r in rows),
                      "accuracy": round(100 * sum(r["correct"] for r in rows) / max(len(rows), 1), 2),
                      "out": str(out / "b0_results.jsonl")}))


if __name__ == "__main__":
    main()
