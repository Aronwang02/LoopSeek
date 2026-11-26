"""G1 diagnostics (2026-10-02): budget.inspect_min_gap in EpisodeV3._taken and the uniform_strat frame selector."""
from types import SimpleNamespace

from src.baselines.deciders import UniformFrameDecider, UniformStratFrameDecider
from src.common.types import DecisionInput
from src.env.episode_v3 import EpisodeV3

N = 32


def _inp(valid):
    return DecisionInput("frame", [], [["c"]] * N, [f"C{j + 1}" for j in range(N)], list(valid), select_k=2, mask_radius=1)


def _stub(min_gap, evidence):
    """Active window [0, 32) with frame i at time i: grid spacing 1 s, near-duplicate gap 0.25 s."""
    ep = SimpleNamespace(b={"frame_candidates": N, "inspect_min_gap": min_gap}, active=SimpleNamespace(length=float(N)),
                         asset=SimpleNamespace(time=float), evidence=list(evidence))
    ep.answer_set = lambda: list(ep.evidence)
    return ep


def test_min_gap_off_keeps_adjacent_candidates_legal():
    taken = EpisodeV3._taken(_stub(0, [8, 24]), list(range(N)))
    assert [i for i, t in enumerate(taken) if t] == [8, 24]


def test_min_gap_2_makes_neighbours_of_earlier_picks_illegal():
    taken = EpisodeV3._taken(_stub(2, [8, 24]), list(range(N)))
    assert [i for i, t in enumerate(taken) if t] == [7, 8, 9, 23, 24, 25]


def test_uniform_strat_spreads_two_inspections():
    d = UniformStratFrameDecider()
    ep = SimpleNamespace(max_inspections=2, inspections=[])
    first = d.decide([_inp([True] * N)], [ep], [None])[0]["info"]["picks"]
    assert first == [4, 20]
    ep.inspections.append((1, first))
    valid = [i not in first for i in range(N)]
    second = d.decide([_inp(valid)], [ep], [None])[0]["info"]["picks"]
    assert second == [12, 28]


def test_old_uniform_selector_second_inspection_lands_next_to_the_first():
    d = UniformFrameDecider(k=2)                      # documents why uniform_strat exists
    first = d.decide([_inp([True] * N)], [None], [None])[0]["info"]["picks"]
    valid = [i not in first for i in range(N)]
    second = d.decide([_inp(valid)], [None], [None])[0]["info"]["picks"]
    assert all(min(abs(s - f) for f in first) == 1 for s in second)
