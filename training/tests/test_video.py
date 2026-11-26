"""Real timestamps from a variable-frame-rate file; nothing assumes a uniform FPS."""
import numpy as np

from fractions import Fraction

from conftest import VFR_PTS_MS, frame_color
from src.video.decode import Timeline, VideoDecoder, read_timeline, uniform_times


def nearest_frame_id(img):
    c = img.reshape(-1, 3).mean(0)
    return int(np.argmin([np.abs(frame_color(i)[:2] - c[:2]).sum() for i in range(len(VFR_PTS_MS))]))


def test_timeline_matches_true_pts(vfr_video):
    tl = read_timeline(vfr_video, mode="packets")
    assert [round(t * 1000) for t in tl.times] == VFR_PTS_MS
    tl2 = read_timeline(vfr_video, mode="decode")
    assert tl2.times == tl.times


def test_frame_at_uses_real_times(asset):
    tl = asset.timeline
    # 0.3 s falls in the gap 0.12 -> 0.40: the frame on screen is the one at 0.12 s (index 3)
    assert tl.frame_at(0.30) == 3
    # an index-based uniform-FPS assumption (25 fps) would have given index 7 here
    assert tl.frame_at(0.30) != int(0.30 * 25)
    assert tl.frame_at(1.0) == 6 and tl.frame_at(1.5) == 7
    assert tl.frame_at(-1) == 0 and tl.frame_at(1e9) == len(VFR_PTS_MS) - 1


def test_decoded_frames_are_the_requested_ones(asset):
    idx = [0, 3, 4, 7, 8, 50, 99]
    frames = asset.frames(idx)
    for i in idx:
        assert nearest_frame_id(frames[i]) == i


def test_uniform_times_are_bin_midpoints():
    assert uniform_times(0, 8, 4) == [1.0, 3.0, 5.0, 7.0]
    assert uniform_times(5, 5, 3) == []


def test_parallel_decode_gives_the_frames_of_one_pass(asset, monkeypatch):
    every = list(range(len(VFR_PTS_MS)))
    for idx in (every, [0, 3, 4, 7, 8, 50, 99], every[::7] + [98, 99], [5]):
        out = {}
        for workers in (1, 4):
            monkeypatch.setenv("JEV_DECODE_WORKERS", str(workers))
            dec = VideoDecoder(asset.path, asset.timeline, max_side=None, seek_gap_sec=0.1)
            dec.split_sec = 0.5                                          # also cut the dense 1.5-6.05 s run
            out[workers] = dec.get(idx)
        assert sorted(out[1]) == sorted(out[4]) == sorted(set(idx))
        for i in idx:
            assert np.array_equal(out[1][i], out[4][i]) and nearest_frame_id(out[4][i]) == i
    assert len(dec._groups(every, 4)) == 4                               # the full request used four containers


def test_groups_cut_only_where_one_pass_seeks_or_inside_long_runs():
    times = [0, 1, 2, 10, 11, 40, 41] + list(range(100, 220))          # seconds; seeks after 2, 11 and 41
    tl = Timeline(pts=[t * 1000 for t in times], time_base=Fraction(1, 1000), pts0=0, duration=221.0,
                  avg_fps=1.0, width=64, height=48)
    dec = VideoDecoder("unused.mp4", tl)
    want = list(range(len(times)))
    assert dec._groups(want, 1) == [want] and dec._groups([3], 4) == [[3]]
    assert dec._groups(list(range(7)), 4) == [[0, 1, 2], [3, 4], [5, 6]]    # short runs: cuts at the seeks only
    groups = dec._groups(want, 4)
    assert [i for g in groups for i in g] == want and 1 < len(groups) <= 4  # contiguous and complete
    long_run = [g for g in groups if times[g[-1]] >= 100]
    assert len(long_run) >= 2                                            # the 119 s run is shared out
    for g in long_run:                                                   # its pieces span >= split_sec
        assert times[g[-1]] - max(times[g[0]], 100) >= dec.split_sec
