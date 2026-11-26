from src.env.candidates import CandidateGenerator, Window


def test_global_8_windows():
    g = CandidateGenerator(1.0)
    T = 120.0
    w = g.global_windows(T)
    keys = [x.key() for x in w]
    assert keys == [(0.0, 30.0), (0.0, 60.0), (0.0, 120.0), (30.0, 60.0), (30.0, 90.0),
                    (60.0, 90.0), (60.0, 120.0), (90.0, 120.0)]
    assert sorted(x.kind for x in w) == ["full"] + ["half"] * 3 + ["quarter"] * 4
    assert all(0 <= x.start < x.end <= T for x in w)


def test_local_same_rule_inside_active():
    g = CandidateGenerator(1.0)
    w = g.local_windows(Window(40.0, 80.0, "half"), 120.0)
    assert [x.key() for x in w] == [(40.0, 50.0), (40.0, 60.0), (40.0, 80.0), (50.0, 60.0), (50.0, 70.0),
                                    (60.0, 70.0), (60.0, 80.0), (70.0, 80.0)]
    assert [x for x in w if x.kind == "active"][0].key() == (40.0, 80.0)


def test_short_span_keeps_whole_window():
    g = CandidateGenerator(1.0)
    w = g.local_windows(Window(10.0, 11.5, "quarter"), 120.0)
    assert [x.key() for x in w] == [(10.0, 11.5)]      # quarters/halves < 1 s are dropped
