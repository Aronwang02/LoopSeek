"""VideoITG-40K for the short-video profile, CPU + network:
  1. video_itg_data.json of NVEagle/VideoITG-40K at a pinned revision -> <root>/annotations/;
  2. the LLaVA-Video-178K videos of its multiple-choice questions -> <root>/LLaVA-Video-178K/<subset>/<member>: every
     tar.gz chunk of a needed subset is streamed from Hugging Face and only the needed members are written
     (scripts/llava_video.fetch_llava, pinned LLaVA-Video revision). Videos already there are skipped, so a rerun
     resumes; a chunk whose stream breaks is streamed again (3 tries).
The chunks cannot be indexed: a subset is streamed whole (the needed subsets are ~1.1 TB of tar.gz in all), but only
the needed videos reach the disk. Every chunk streamed to its end is recorded in <root>/state/<subset>.chunks; a run
that has its videos back (e.g. restored after a preemption) skips those chunks.
Then scripts/build_videoitg.py --video-root <root>.
usage: python scripts/fetch_videoitg.py --root <dir> [--subsets 30_60_s_youtube_v0_1 ...] [--parallel 16]
           [--list-only | --probe]   (--probe first: the tar member layout must match the VideoITG paths)
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_videoitg import parse_mc  # noqa: E402

VIDEOITG_REPO, VIDEOITG_REV = "NVEagle/VideoITG-40K", "112bdd23fc9b804d6e4afe5758d28b5129f933ca"


def needed_videos(annotations, root):
    """{subset: {tar member: local path}} of the videos of the multiple-choice questions."""
    out = defaultdict(dict)
    for r in annotations:
        if parse_mc(r.get("question", ""), r.get("answer", "")) is None:
            continue
        v = r["video"]                                   # LLaVA-Video-178K/<subset>/<member>
        head, subset, member = v.split("/", 2)
        if head != "LLaVA-Video-178K":
            raise ValueError(f"unexpected VideoITG video path: {v}")
        out[subset][member] = Path(root) / v
    return dict(out)


def probe(todo):
    """The first video member of each subset's first chunk (a few MB of the stream): its top folder must be one the
    expected members start with (paths relative to the subset folder), else the whole download would write nothing."""
    import tarfile
    import requests
    from scripts.llava_video import hf_headers, llava_chunks
    ok = True
    for subset, members in sorted(todo.items()):
        name, url = llava_chunks(subset)[0]
        with requests.get(url, stream=True, timeout=120, headers=hf_headers()) as r:
            r.raise_for_status()
            r.raw.decode_content = False
            with tarfile.open(fileobj=r.raw, mode="r|gz") as tar:
                first = next(m.name for m in tar if m.isfile() and m.name.endswith(".mp4"))
        known = any(m.split("/", 1)[0] == first.split("/", 1)[0] for m in members)
        ok &= known
        print(json.dumps({"subset": subset, "chunk": name, "first_member": first, "layout_matches": known}), flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--subsets", nargs="*", default=None, help="only these LLaVA-Video subsets (default: all needed)")
    ap.add_argument("--parallel", type=int, default=16,
                    help="chunks streamed at the same time, of every subset (round-robin; 2026-10-21, was subsets)")
    ap.add_argument("--list-only", action="store_true", help="annotation + what is needed / present; no video download")
    ap.add_argument("--probe", action="store_true", help="check each subset's tar member layout (a few MB), then stop")
    args = ap.parse_args()
    root = Path(args.root)
    from huggingface_hub import hf_hub_download
    ann = Path(hf_hub_download(VIDEOITG_REPO, "video_itg_data.json", repo_type="dataset", revision=VIDEOITG_REV,
                               local_dir=root / "annotations"))
    todo = needed_videos(json.loads(ann.read_text()), root)
    if args.subsets:
        todo = {s: m for s, m in todo.items() if s in set(args.subsets)}
    for s, members in sorted(todo.items()):
        print(json.dumps({"subset": s, "videos": len(members), "present": sum(p.exists() for p in members.values())}),
              flush=True)
    if args.list_only:
        return
    if args.probe:
        raise SystemExit(0 if probe(todo) else "tar member layout differs from the VideoITG paths: do not download")
    from scripts.llava_video import fetch_llava_parallel
    fetch_llava_parallel(dict(sorted(todo.items())), root / "state", args.parallel)
    missing = {s: sum(not p.exists() for p in m.values()) for s, m in todo.items()}
    status = {"annotations": str(ann), "revision": VIDEOITG_REV, "videos": sum(len(m) for m in todo.values()),
              "missing": missing}
    (root / "FETCHED.json").write_text(json.dumps(status, indent=1) + "\n")
    print(json.dumps(status), flush=True)
    if sum(missing.values()):
        print(json.dumps({"WARNING": "some videos are not in any chunk; build_videoitg.py drops their questions"}))


if __name__ == "__main__":
    main()
