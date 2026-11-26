"""Haystack preparation and mixing: real parquet, fake download transport, no GPU or credentials needed."""
import json
import os
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_lvhaystack as build
import prepare_lvhaystack as prep
from mix_long_manifests import mix


def question(video="train", clip="clip"):
    return {"video_id": video, "vclip_id": clip, "question": "What is on the table?", "options": {"A": "cup", "B": "plate"},
            "answer": "B", "frame_indexes_video": [162000, 162060], "frame_indexes_vclip": [300, 360],
            "video_metadata": {"video_duration": 7200.0, "vclip_duration": 600.0, "frame_rate": 30.0}}


def annotations(root):
    root.mkdir(parents=True, exist_ok=True)
    data = {"train": [question(), question("shared"), dict(question("bad"), frame_indexes_video=[])],
            "val": [question("shared"), question("val"), question("test")], "test": [question("test")]}
    for split, rows in data.items():
        pd.DataFrame(rows).to_parquet(root / f"{split}-00000-of-00001.parquet")


def test_full_video_and_clip_use_different_time_axes(tmp_path):
    full, _ = build.convert_rows([question()], "train", tmp_path, require_videos=False)
    clip, _ = build.convert_rows([question()], "train", tmp_path, mode="clip", require_videos=False)
    assert full[0]["duration"] == 7200 and full[0]["evidence_times"] == [5400, 5402]
    assert full[0]["key_spans"] == [[5398, 5404]] and full[0]["video_path"].endswith("train.mp4")
    assert clip[0]["duration"] == 600 and clip[0]["evidence_times"] == [10, 12]
    assert clip[0]["video_path"].endswith("clip.mp4")
    assert full[0]["group_key"] == clip[0]["group_key"]


def test_out_of_range_frames_do_not_create_fake_evidence(tmp_path):
    r = question()
    r["frame_indexes_video"] = [-1, 216000, 999999]
    rows, stats = build.convert_rows([r], "train", tmp_path, require_videos=False)
    assert not rows and stats["dropped_no_valid_keyframe"] == 1


def test_official_shards_and_source_video_holdout(tmp_path):
    annotations(tmp_path / "data")
    rows, stats = build.build_manifests(tmp_path, tmp_path / "videos", tmp_path / "out", require_videos=False,
                                        dev_videos=0)
    assert {(r["split"], r["video_id"]) for r in rows} == {("train", "train"), ("test", "shared"), ("test", "val")}
    assert stats["train"]["dropped_heldout_video"] == stats["val"]["dropped_heldout_video"] == 1
    assert all(r["video_id"] != "test" for r in rows)
    with pytest.raises(FileNotFoundError, match="missing LV-Haystack video"):
        build.build_manifests(tmp_path, tmp_path / "videos", tmp_path / "out")


def test_dev_is_whole_train_source_videos_and_official_val_is_test(tmp_path):
    """2026-10-20: dev = source videos held out of official train (seeded), official val = test, official test unused."""
    root = tmp_path / "data"
    root.mkdir()
    train = [question(f"t{v}") for v in range(6) for _ in range(2)]
    data = {"train": train, "val": [question("val")], "test": [question("test")]}
    for split, rows in data.items():
        pd.DataFrame(rows).to_parquet(root / f"{split}-00000-of-00001.parquet")
    rows, stats = build.build_manifests(tmp_path, tmp_path / "videos", tmp_path / "out", require_videos=False,
                                        dev_videos=2, dev_seed=3)
    dev = {r["group_key"] for r in rows if r["split"] == "dev"}
    assert len(dev) == 2 and stats["dev"] == {"source_videos": 2, "rows": 4}
    assert not dev & {r["group_key"] for r in rows if r["split"] == "train"}           # no video in both
    assert {r["video_id"] for r in rows if r["split"] == "test"} == {"val"}
    again, _ = build.build_manifests(tmp_path, tmp_path / "videos", tmp_path / "out2", require_videos=False,
                                     dev_videos=2, dev_seed=3)
    assert [r["split"] for r in again] == [r["split"] for r in rows]                     # same seed, same dev
    out = tmp_path / "out"
    assert sorted(p.name for p in out.glob("lvhaystack_*.jsonl")) == [
        "lvhaystack_all.jsonl", "lvhaystack_dev.jsonl", "lvhaystack_test.jsonl", "lvhaystack_train.jsonl"]
    assert len((out / "lvhaystack_dev.jsonl").read_text().splitlines()) == 4


def test_download_required_videos_only_and_reuse_cache(tmp_path):
    hub, root = tmp_path / "hub", tmp_path / "cache"
    annotations(hub / "data")
    credentials = tmp_path / "credentials"
    credentials.write_text("[ego4d]\naws_access_key_id = fake\naws_secret_access_key = fake\n")
    calls, downloads = [], []

    def fake_hf(repo, filename, repo_type, revision, local_dir):
        assert (repo, repo_type, revision) == (prep.REPO, "dataset", prep.REVISION)
        downloads.append(filename)
        out = Path(local_dir) / filename
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(hub / filename, out)
        return str(out)

    def fake_run(args, **kw):
        calls.append(args)
        if "venv" in args:
            python = Path(args[-1]) / "bin/python"
            python.parent.mkdir(parents=True)
            python.touch()
        elif "ego4d.cli.cli" in args:
            uids = Path(args[args.index("--video_uid_file") + 1]).read_text().split()
            assert set(uids) == {"shared", "train", "val"}  # neither test nor the invalid question's media
            dest = root / "ego4d/v2/full_scale"
            dest.mkdir(parents=True, exist_ok=True)
            for uid in uids:
                (dest / f"{uid}.mp4").write_bytes(b"test media")

    argv = ["prepare_lvhaystack.py", "--root", str(root), "--scratch", str(tmp_path / "tools"), "--download"]
    with patch.dict(os.environ, AWS_SHARED_CREDENTIALS_FILE=str(credentials)), patch.object(sys, "argv", argv), \
            patch("huggingface_hub.hf_hub_download", side_effect=fake_hf), patch.object(prep.subprocess, "run", side_effect=fake_run):
        prep.main()
        assert len(downloads) == 3 and len(calls) == 3
        args = calls[-1]
        assert args[args.index("--datasets") + 1] == "full_scale"
        assert args[args.index("--version") + 1] == "v2" and "--yes" in args
        prep.main()
        assert len(downloads) == 3 and len(calls) == 3  # second job needs no downloads/installation
        # An interrupted first download can leave nonempty partial files, but no successful completion marker.
        (root / "PREPARED.json").unlink()
        (root / "ego4d/v2/full_scale/train.mp4").write_bytes(b"partial")
        prep.main()
        assert len(calls) == 4 and "ego4d.cli.cli" in calls[-1]
    report = json.loads((root / "PREPARED.json").read_text())
    assert report["videos"] == 3 and report["stats"]["train"]["rows"] == 1


def test_missing_ego4d_credentials_explains_required_auth(tmp_path):
    video_root = tmp_path / "ego4d/v2/full_scale"
    rows, _ = build.convert_rows([question()], "train", video_root, require_videos=False)
    with patch.dict(os.environ, AWS_SHARED_CREDENTIALS_FILE=str(tmp_path / "none")), \
            patch.object(prep.subprocess, "run") as run:
        with pytest.raises(RuntimeError, match="HF_TOKEN cannot download"):
            prep.download_videos(tmp_path, video_root, rows, tmp_path / "tools", "ego4d")
        run.assert_not_called()


def test_mixture_keeps_all_examples_and_rejects_test_leakage(tmp_path):
    inputs = [tmp_path / f"{source}.jsonl" for source in ("cgbench", "lvhaystack")]
    for p in inputs:
        p.write_text("".join(json.dumps({"sample_id": f"{p.stem}:{i}", "source": p.stem, "split": "train"}) + "\n"
                             for i in range(20)))
    out = tmp_path / "mixed.jsonl"
    stats = mix(inputs, out, "train", 17)
    assert stats["sources"] == {"cgbench": 20, "lvhaystack": 20}
    first = out.read_text()
    assert len({json.loads(l)["source"] for l in first.splitlines()[:10]}) == 2
    mix(inputs, out, "train", 17)
    assert out.read_text() == first
    with pytest.raises(ValueError, match="differ"):
        mix(inputs, inputs[0], "train", 17)
    with pytest.raises(ValueError, match="wrong split"):
        mix(inputs, out, "dev", 17)
