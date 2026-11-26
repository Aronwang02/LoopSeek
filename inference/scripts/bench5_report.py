"""Five-benchmark evaluation report: scores only, no reference run. One agent run on the manifest of
scripts/build_bench5.py:
  per bench        n, accuracy with a 95% CI by resampling videos (2000 draws, seed 0), statuses other than ok,
                   latency = policy + reader seconds per question (CPU decode not counted), actions per question
  mlvu             m-avg = unweighted mean of the 7 task accuracies (the official MLVU dev metric), and each task
  videomme         short / medium / long and overall = their mean (CI resamples within each duration)
  lvbench          overall, by question type (a question may have several) and by video type; with --uniform-lvbench
                   (scripts/run_b0.py, 32 uniform frames in video mode) its accuracy and the paired
                   difference agent - uniform with a CI by resampling videos
  five_bench_mean  mean of CG-Bench mini, MLVU m-avg, LongVideoBench, Video-MME overall, LVBench
usage: python scripts/bench5_report.py --runs runs/<TAG>_bench5 [--manifest data/manifests/bench5/bench5.jsonl]
           [--uniform-lvbench runs/<TAG>_b0_lvbench] [--out JSON]"""
import argparse
import collections
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VMME = ("videomme_short", "videomme_medium", "videomme_long")
JUDGED = ("cgbench_mini", "mlvu", "longvideobench", "videomme", "lvbench")
N_BOOT = 2000


def jsonl(p):
    out = []
    try:
        with open(p) as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except ValueError:                   # a line its process is writing at this moment
                    pass
    except OSError:
        pass
    return out


def load_runs(dirs):
    """Agent runs (results.jsonl) or reader-only runs of scripts/run_b0.py (b0_results.jsonl, or the .partial file of
    one still running; no status / seconds there: status ok, no latency)."""
    out = {}
    for d in dirs:
        rows = jsonl(Path(d) / "results.jsonl")
        if not rows:
            rows = [dict(r, status="ok") for r in (jsonl(Path(d) / "b0_results.jsonl")
                                                  or jsonl(Path(d) / "b0_results.partial.jsonl"))]
        for r in rows:
            if r.get("status") == "ok" or r["sample_id"] not in out:
                out[r["sample_id"]] = r
    return out


def report(man, runs, uniform=None):
    ok = {s: r for s, r in runs.items() if r.get("status", "ok") == "ok" and s in man}
    bad = collections.Counter(r.get("status") for s, r in runs.items() if r.get("status", "ok") != "ok")

    def ids(pred):
        return sorted(s for s in ok if pred(man[s]))

    def acc(X, sel):
        return 100 * sum(bool(X[s]["correct"]) for s in sel) / len(sel)

    def mavg(X, sel):
        by = collections.defaultdict(list)
        for s in sel:
            by[man[s]["subset"]].append(bool(X[s]["correct"]))
        return 100 * statistics.mean(sum(v) / len(v) for v in by.values())

    def boot(stat, parts, seed=0):
        """95% CI of stat(list of id lists) over N_BOOT draws of whole videos (group_key), each part on its own."""
        rng = random.Random(seed)
        gs = []
        for p in parts:
            g = collections.defaultdict(list)
            for s in p:
                g[man[s]["group_key"]].append(s)
            gs.append((g, list(g)))
        draws = sorted(stat([[s for k in (keys[rng.randrange(len(keys))] for _ in keys) for s in g[k]] for g, keys in gs])
                       for _ in range(N_BOOT))
        return [round(draws[int(0.025 * N_BOOT)], 2), round(draws[int(0.975 * N_BOOT) - 1], 2)]

    def lat(sel):
        xs = [ok[s]["seconds"]["policy"] + ok[s]["seconds"]["reader"] for s in sel if ok[s].get("seconds")]
        if not xs:
            return None
        return {"policy_plus_reader_s_mean": round(statistics.mean(xs), 3), "median": round(statistics.median(xs), 3),
                "policy_s_mean": round(statistics.mean(ok[s]["seconds"]["policy"] for s in sel if ok[s].get("seconds")), 3),
                "reader_s_mean": round(statistics.mean(ok[s]["seconds"]["reader"] for s in sel if ok[s].get("seconds")), 3),
                "with_seconds": len(xs), "actions_per_q": round(statistics.mean(len(ok[s].get("actions") or []) for s in sel), 1)}

    out = {"done": len(ok), "of": len(man), "not_ok": dict(bad), "benches": {}}
    judged = {}
    for b in sorted({r["bench"] for r in man.values()}):
        sel = ids(lambda r, b=b: r["bench"] == b)
        res = {"n": len(sel), "of": sum(r["bench"] == b for r in man.values())}
        if sel:
            res.update(accuracy=round(acc(ok, sel), 2), ci95=boot(lambda ps: acc(ok, ps[0]), [sel]), latency=lat(sel))
        if b == "mlvu" and sel:
            res["m_avg"] = round(mavg(ok, sel), 2)
            res["m_avg_ci95"] = boot(lambda ps: mavg(ok, ps[0]), [sel])
            tasks = collections.defaultdict(list)
            for s in sel:
                tasks[man[s]["subset"]].append(s)
            res["per_task"] = {t: {"n": len(v), "accuracy": round(acc(ok, v), 2)} for t, v in sorted(tasks.items())}
            judged["mlvu"] = res["m_avg"]
        if b == "lvbench" and sel:
            by_type, by_video = collections.defaultdict(list), collections.defaultdict(list)
            for s in sel:
                for t in man[s].get("question_type") or ["?"]:
                    by_type[t].append(s)
                by_video[man[s].get("video_type", "?")].append(s)
            res["by_question_type"] = {t: {"n": len(v), "accuracy": round(acc(ok, v), 2)} for t, v in sorted(by_type.items())}
            res["by_video_type"] = {t: {"n": len(v), "accuracy": round(acc(ok, v), 2)} for t, v in sorted(by_video.items())}
            if uniform:
                both = [s for s in sel if s in uniform]
                if both:
                    res["uniform32_video"] = {
                        "n": len(both), "agent": round(acc(ok, both), 2), "uniform": round(acc(uniform, both), 2),
                        "diff": round(acc(ok, both) - acc(uniform, both), 2),
                        "ci95": boot(lambda ps: acc(ok, ps[0]) - acc(uniform, ps[0]), [both]),
                        "only_agent_correct": sum(bool(ok[s]["correct"]) and not uniform[s]["correct"] for s in both),
                        "only_uniform_correct": sum(bool(uniform[s]["correct"]) and not ok[s]["correct"] for s in both)}
        if b in ("cgbench_mini", "longvideobench", "lvbench") and sel:
            judged[b] = res["accuracy"]
        out["benches"][b] = res
    vs = [ids(lambda r, b=b: r["bench"] == b) for b in VMME]
    if all(vs):
        mean3 = lambda ps: statistics.mean(acc(ok, p) for p in ps)   # noqa: E731
        out["benches"]["videomme_overall"] = {"n": sum(map(len, vs)), "accuracy": round(mean3(vs), 2),
                                              "ci95": boot(mean3, vs)}
        judged["videomme"] = out["benches"]["videomme_overall"]["accuracy"]
    out["scores"] = {k: judged.get(k) for k in JUDGED}
    if all(judged.get(k) is not None for k in JUDGED):
        out["five_bench_mean"] = round(statistics.mean(judged[k] for k in JUDGED), 2)
    all_sel = sorted(ok)
    if all_sel:
        out["latency_all"] = lat(all_sel)
    out["complete"] = out["done"] >= len(man) and not bad
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--manifest", default=str(ROOT / "data/manifests/bench5/bench5.jsonl"))
    ap.add_argument("--uniform-lvbench", nargs="*", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    man = {r["sample_id"]: r for r in jsonl(a.manifest)}
    uniform = load_runs(a.uniform_lvbench) if a.uniform_lvbench else None
    if uniform is not None:
        uniform = {s: r for s, r in uniform.items() if r.get("status", "ok") == "ok"}
    txt = json.dumps(report(man, load_runs(a.runs), uniform), indent=1, ensure_ascii=False)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt + "\n")


if __name__ == "__main__":
    main()
