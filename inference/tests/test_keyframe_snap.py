"""video.keyframe_snap (2026-11-03): sparse uniform grids take the nearest keyframes; dense grids stay exact. The test
video is MPEG-4, 30 fps, 10 s, slowly changing content with scene-cut keyframes off, so a keyframe every 30 frames (1 s)
(the conftest vfr_video changes colour every frame: nearly every frame is a keyframe there)."""
import statistics
from fractions import Fraction

import av
import numpy as np
import pytest

from src.video.assets import VideoAsset
from src.video.decode import VideoDecoder, read_timeline, timeline_from_json, timeline_to_json, uniform_times


@pytest.fixture(scope="module")
def gop_video(tmp_path_factory):
    path = tmp_path_factory.mktemp("gop") / "gop30.mp4"
    with av.open(str(path), "w") as out:
        st = out.add_stream("mpeg4", rate=30)
        st.width, st.height, st.pix_fmt = 64, 48, "yuv420p"
        st.codec_context.gop_size = 30
        st.codec_context.max_b_frames = 0
        st.codec_context.options = {"sc_threshold": "1000000000"}
        base = np.linspace(0, 255, 64 * 3, dtype=np.float32).reshape(1, 64, 3)
        for i in range(300):
            img = np.clip(np.tile(base, (48, 1, 1)) + i * 0.3, 0, 255).astype(np.uint8)
            fr = av.VideoFrame.from_ndarray(img, format="rgb24")
            fr.pts = i
            for pkt in st.encode(fr):
                out.mux(pkt)
        for pkt in st.encode():
            out.mux(pkt)
    return path


@pytest.fixture()
def asset(gop_video, tmp_path):
    return VideoAsset(gop_video, tmp_path / "cache", max_side=None)


def test_timeline_records_keyframes_from_packet_flags(gop_video):
    tl = read_timeline(gop_video)
    assert tl.keys == list(range(0, 300, 30))                            # gop_size 30
    again = timeline_from_json(timeline_to_json(tl))
    assert again.keys == tl.keys
    old = timeline_to_json(tl)
    old.pop("keys")                                                      # a timeline cached before 2026-11-03
    assert timeline_from_json(old).keys == []


def test_sparse_grid_snaps_dense_grid_stays_exact(asset):
    tl = asset.timeline
    keys = set(asset.keyframe_indices())
    gap = statistics.median(b - a for a, b in zip(sorted(tl.times[i] for i in keys),
                                                  sorted(tl.times[i] for i in keys)[1:]))
    T = tl.duration
    n_sparse = max(2, int(T / (2.5 * gap)))                              # spacing 2.5 x the keyframe gap
    snapped = asset.snapped_uniform_indices(0.0, T, n_sparse)
    assert set(snapped) <= keys and len(snapped) == n_sparse
    for t, i in zip(uniform_times(0.0, T, n_sparse), snapped):           # each moves to its nearest keyframe
        assert abs(tl.times[i] - t) <= gap / 2 + 1e-6
    n_dense = int(T / (0.5 * gap))                                       # spacing half a gap: exact frames
    assert asset.snapped_uniform_indices(0.0, T, n_dense) == asset.uniform_indices(0.0, T, n_dense)
    lo, hi = T * 0.3, T * 0.9                                            # a window: frames stay inside it
    win = asset.snapped_uniform_indices(lo, hi, max(2, int((hi - lo) / (2.5 * gap))))
    assert all(lo <= tl.times[i] < hi for i in win)
    dec = VideoDecoder(asset.read_path, tl)                              # a keyframe costs one decoded frame
    dec.get([snapped[len(snapped) // 2]])
    assert dec.frames_decoded == 1


def test_old_cached_timeline_gets_keyframes(gop_video, tmp_path):
    a = VideoAsset(gop_video, tmp_path / "cache", max_side=None)
    want = list(a.timeline.keys)
    a.timeline.keys = []
    a._save_timeline(a.timeline)                                         # as cached by the earlier code
    b = VideoAsset(gop_video, tmp_path / "cache", max_side=None)
    assert b.timeline.keys == []
    assert b.keyframe_indices() == want
    assert VideoAsset(gop_video, tmp_path / "cache", max_side=None).timeline.keys == want   # re-cached


def test_episode_overview_and_locator_context_use_the_snap(asset, features, cfg):
    from conftest import StubEncoder
    from src.env.environment import PublicSample
    from src.env.episode_v3 import EpisodeV3
    keys = set(asset.keyframe_indices())
    gap = statistics.median(b - a for a, b in zip(sorted(asset.time(i) for i in keys), sorted(asset.time(i) for i in keys)[1:]))
    cfg["video"]["global_frames"] = max(2, int(asset.duration / (2.5 * gap)))
    sample = PublicSample("s", "v", str(asset.path), "q?", {"A": "a", "B": "b"})
    exact = EpisodeV3(sample, asset, features, StubEncoder(), cfg, key="t").reset()
    cfg["video"]["keyframe_snap"] = True
    snap = EpisodeV3(sample, asset, features, StubEncoder(), cfg, key="t").reset()
    assert set(snap.global_idx) <= keys and not set(exact.global_idx) <= keys
    assert len(snap.global_idx) == len(exact.global_idx)
