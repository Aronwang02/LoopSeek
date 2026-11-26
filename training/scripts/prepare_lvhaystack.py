"""Download pinned LV-Haystack annotations and only the full Ego4D videos needed for train/dev.

Ego4D needs its issued AWS credentials, not a Hugging Face token. Its official downloader uses a separate CPU
environment so training's PyTorch/CUDA packages do not change. Existing videos are reused.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from build_lvhaystack import build_manifests

REPO = "MLL-Lab/LongVideoHaystack"
REVISION = "2ad05dee35c4eba578f63cbd3c72eedf91896497"
EGO4D_VERSION = "v2"
EGO4D_PACKAGE = "ego4d==1.7.3"


def download_annotations(root):
    from huggingface_hub import hf_hub_download
    for split in ("train", "val", "test"):
        filename = f"data/{split}-00000-of-00001.parquet"
        existing = root / "annotations" / filename
        if existing.is_file() and existing.stat().st_size:
            continue
        hf_hub_download(REPO, filename, repo_type="dataset", revision=REVISION,
                        local_dir=root / "annotations")


def download_videos(root, video_root, rows, scratch, profile):
    # A killed official download may leave a nonempty partial mp4. Only a successfully completed preparation
    # marks videos reusable; otherwise let the official CLI check file sizes and resume its own downloads.
    verified = set()
    marker = root / "PREPARED.json"
    if marker.exists():
        try:
            prior = json.loads(marker.read_text())
            if prior.get("revision") == REVISION and prior.get("video_root") == str(video_root):
                verified = set(prior.get("video_ids", []))
        except json.JSONDecodeError:
            pass
    missing = sorted({r["video_id"] for r in rows
                      if r["video_id"] not in verified or not Path(r["video_path"]).is_file()
                      or Path(r["video_path"]).stat().st_size == 0})
    if not missing:
        return
    expected = root / "ego4d" / EGO4D_VERSION / "full_scale"
    if video_root.resolve() != expected.resolve():
        raise RuntimeError(f"automatic download uses {expected}; use LVH_PREPARE=0 with existing custom --video-root")
    import configparser
    credentials = Path(os.environ.get("AWS_SHARED_CREDENTIALS_FILE", "~/.aws/credentials")).expanduser()
    config = configparser.RawConfigParser()
    config.read(credentials)
    if not config.has_section(profile):
        raise RuntimeError(f"Ego4D AWS profile [{profile}] is missing in {credentials}. Obtain Ego4D credentials at "
                           "https://ego4d-data.org/docs/start-here/ and store them there; HF_TOKEN cannot download these videos.")
    scratch.mkdir(parents=True, exist_ok=True)
    env = scratch / "ego4d-download-env"
    python = env / "bin/python"
    if not python.exists():
        subprocess.run([sys.executable, "-m", "venv", str(env)], check=True)
    ready = env / ".ready"
    if not ready.exists() or ready.read_text().strip() != EGO4D_PACKAGE:
        subprocess.run([str(python), "-m", "pip", "install", EGO4D_PACKAGE], check=True)
        ready.write_text(EGO4D_PACKAGE + "\n")
    uid_file = root / "missing_video_uids.txt"
    uid_file.write_text("\n".join(missing) + "\n")
    print(json.dumps({"download_videos": len(missing), "destination": str(expected)}), flush=True)
    subprocess.run([str(python), "-m", "ego4d.cli.cli", "--output_directory", str(root / "ego4d"),
                    "--datasets", "full_scale", "--version", EGO4D_VERSION, "--aws_profile_name", profile,
                    "--video_uid_file", str(uid_file), "--no-metadata", "--yes"], check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--video-root", help="existing custom videos; default ROOT/ego4d/v2/full_scale")
    ap.add_argument("--out-dir", help="default ROOT/manifests")
    ap.add_argument("--scratch", default="/tmp/onejev-data-tools")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--aws-profile", default="ego4d")
    ap.add_argument("--tau", type=float, default=2.0)
    args = ap.parse_args()
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    video_root = Path(args.video_root).resolve() if args.video_root else root / "ego4d" / EGO4D_VERSION / "full_scale"
    out_dir = Path(args.out_dir) if args.out_dir else root / "manifests"
    if args.download:
        download_annotations(root)
    rows, _ = build_manifests(root / "annotations", video_root, out_dir, tau=args.tau, require_videos=False)
    uids = sorted({r["video_id"] for r in rows})
    (root / "video_uids.txt").write_text("\n".join(uids) + "\n")
    if args.download:
        download_videos(root, video_root, rows, Path(args.scratch), args.aws_profile)
    rows, stats = build_manifests(root / "annotations", video_root, out_dir, tau=args.tau)
    if not all(stats[s]["rows"] for s in ("train", "val")):
        raise RuntimeError("LV-Haystack train/dev are empty")
    result = {"repo": REPO, "revision": REVISION, "video_root": str(video_root), "videos": len(uids),
              "video_ids": uids, "stats": stats}
    (root / "PREPARED.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "video_ids"}), flush=True)


if __name__ == "__main__":
    main()
