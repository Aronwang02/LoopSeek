"""Accuracy with video-clustered bootstrap CIs, paired system comparison, latency summaries."""
import math
import random
from collections import defaultdict


def _cluster_bootstrap(clusters, stat, n_boot=2000, seed=0):
    rng = random.Random(seed)
    keys = list(clusters)
    vals = []
    for _ in range(n_boot):
        sample = [clusters[keys[rng.randrange(len(keys))]] for _ in keys]
        vals.append(stat([x for c in sample for x in c]))
    vals.sort()
    return vals[int(0.025 * n_boot)], vals[int(0.975 * n_boot) - 1]


def accuracy_summary(rows, key="correct", cluster="group_key"):
    """rows: dicts with a boolean `correct` and a video group key."""
    n = len(rows)
    if n == 0:
        return {"n": 0}
    acc = sum(r[key] for r in rows) / n
    clusters = defaultdict(list)
    for r in rows:
        clusters[r.get(cluster, r.get("sample_id"))].append(float(r[key]))
    lo, hi = _cluster_bootstrap(clusters, lambda xs: sum(xs) / len(xs))
    out = {"n": n, "accuracy": acc, "ci95_video_bootstrap": [lo, hi], "n_video_groups": len(clusters),
           "parse_failures": sum(1 for r in rows if r.get("parsed") is None)}
    by = defaultdict(list)
    for r in rows:
        by[r.get("subset", "all")].append(r[key])
    out["accuracy_by_subset"] = {k: sum(v) / len(v) for k, v in by.items()}
    return out


def paired_difference(rows_a, rows_b, cluster="group_key", n_boot=2000, seed=0):
    """Accuracy(b) - accuracy(a) on the same questions, CI by resampling video groups."""
    a = {r["sample_id"]: r for r in rows_a}
    b = {r["sample_id"]: r for r in rows_b}
    common = sorted(set(a) & set(b))
    clusters = defaultdict(list)
    for s in common:
        clusters[a[s].get(cluster, s)].append(float(b[s]["correct"]) - float(a[s]["correct"]))
    diffs = [x for c in clusters.values() for x in c]
    mean = sum(diffs) / len(diffs) if diffs else float("nan")
    lo, hi = _cluster_bootstrap(clusters, lambda xs: sum(xs) / len(xs), n_boot, seed) if diffs else (math.nan,) * 2
    return {"n_common": len(common), "diff_b_minus_a": mean, "ci95_video_bootstrap": [lo, hi],
            "b_only_correct": sum(1 for s in common if b[s]["correct"] and not a[s]["correct"]),
            "a_only_correct": sum(1 for s in common if a[s]["correct"] and not b[s]["correct"])}


def latency_summary(values):
    xs = sorted(values)
    if not xs:
        return {}
    def q(p):
        return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]
    return {"n": len(xs), "median": q(0.5), "p95": q(0.95), "mean": sum(xs) / len(xs)}
