"""Manifest from Weaver RL QA files + locally downloaded videos (draft §7.1).

Row fields: sample_id, video_id, group_key, video_rel, video_path, duration (probed from the file),
duration_meta (from Weaver), question, options, answer, source, subset, split.
`answer` is only read by the evaluator; the agent gets a PublicSample without it.
"""
import hashlib
import random
import re
from pathlib import Path

from ..env.environment import PublicSample

_OPT = re.compile(r"^\(([A-Z])\)\s*(.*)$")
_TRAILER = "Think step by step and call tools if needed, then answer."


def parse_weaver_question(text):
    """'Question: ...\\n(A) ...\\n(B) ...\\nThink step by step ...' -> (question, {letter: text})."""
    lines = [l.rstrip() for l in text.strip().split("\n")]
    q_lines, options, last = [], {}, None
    for line in lines:
        s = line.strip()
        if s == _TRAILER or s == "Options:":
            continue
        m = _OPT.match(s)
        if m:
            options[m.group(1)] = m.group(2).strip()
            last = m.group(1)
        elif options and last and s:
            options[last] += " " + s          # wrapped option text
        elif not options:
            q_lines.append(s)
    q = " ".join(x for x in q_lines if x)
    if q.startswith("Question:"):
        q = q[len("Question:"):].strip()
    letters = list(options)
    if len(letters) < 2 or letters != [chr(65 + i) for i in range(len(letters))]:
        raise ValueError(f"unexpected option letters {letters}")
    return q, options


_YT = re.compile(r"(?:^v_|^ytb_)?([A-Za-z0-9_-]{11})$")


def group_key(video_rel):
    """Leakage group: the original source video. ActivityNet 'v_<ytid>', LLaVA youtube
    'ytb_<ytid>', youcook2 '<recipe>/<ytid>/...' all reduce to the YouTube id when present."""
    parts = video_rel.split("/")
    stem = Path(parts[-1]).stem
    m = _YT.match(stem)
    if m and (stem.startswith("v_") or stem.startswith("ytb_")):
        return "yt:" + m.group(1)
    for p in parts:
        if len(p) == 11 and re.fullmatch(r"[A-Za-z0-9_-]{11}", p) and "youcook2" in video_rel:
            return "yt:" + p
    return "file:" + "/".join(parts[1:])


def rows_from_weaver(records, source_tag, videos_root):
    rows = []
    for r in records:
        rel = r["video"]
        info = r["extra_info"]
        q, opts = parse_weaver_question(info["question"])
        ans = r["reward_model"]["ground_truth"].strip().upper()
        if ans not in opts:
            raise ValueError(f"answer {ans} not in options for index {info['index']}")
        subset = rel.split("/")[1] if rel.startswith("LLaVA-Video-178K/") else rel.split("/")[0]
        rows.append({
            "sample_id": f"{source_tag}:{info['index']}",
            "video_id": rel.split("/", 1)[1] if rel.startswith("LLaVA-Video-178K/") else rel,
            "group_key": group_key(rel),
            "video_rel": rel,
            "video_path": str(Path(videos_root) / rel),
            "duration_meta": float(info["duration"]),
            "question": q, "options": opts, "answer": ans,
            "source": source_tag, "subset": subset,
        })
    return rows


def split_by_group(rows, n_test, n_dev, seed, eligible=lambda r: True):
    """Assign whole source-video groups to test/dev/train, stratified by subset so dev and test
    have the same subset mix; no source video crosses splits."""
    groups = {}
    for r in rows:
        if eligible(r):
            groups.setdefault(r["group_key"], []).append(r)
    by_subset = {}
    for k, g in groups.items():
        primary = sorted(r["subset"] for r in g)[0]
        by_subset.setdefault(primary, []).append(k)
    total = sum(len(g) for g in groups.values())
    for sub, keys in sorted(by_subset.items()):
        n_sub = sum(len(groups[k]) for k in keys)
        tgt = {"test": round(n_test * n_sub / total), "dev": round(n_dev * n_sub / total)}
        keys = sorted(keys, key=lambda k: hashlib.sha1(f"{seed}:{k}".encode()).hexdigest())
        counts = {"test": 0, "dev": 0}
        for k in keys:
            n = len(groups[k])
            split = "test" if counts["test"] < tgt["test"] else "dev" if counts["dev"] < tgt["dev"] else "train"
            if split in counts:
                counts[split] += n
            for r in groups[k]:
                r["split"] = split
    for r in rows:
        r.setdefault("split", "excluded")
    return rows


def pick_smoke(rows, n, seed):
    """Smoke samples come from train, spread across subsets."""
    train = [r for r in rows if r["split"] == "train"]
    rng = random.Random(seed)
    by = {}
    for r in sorted(train, key=lambda r: r["sample_id"]):
        by.setdefault(r["subset"], []).append(r)
    for v in by.values():
        rng.shuffle(v)
    out, subsets = [], sorted(by)
    while len(out) < n and any(by.values()):
        for s in subsets:
            if by[s] and len(out) < n:
                out.append(by[s].pop())
    return out


def public_sample(row) -> PublicSample:
    return PublicSample(row["sample_id"], row["video_id"], row["video_path"], row["question"],
                        dict(row["options"]))
