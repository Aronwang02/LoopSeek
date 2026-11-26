"""SFT early stopping (scripts/train_sft.py: EarlyStop, selection_score), on the real verifier1 dev curve of TAG=short."""
from scripts.train_sft import EarlyStop, selection_score

VERIFIER1 = [0.774, 0.806, 0.807, 0.805, 0.803, 0.810, 0.814, 0.809, 0.813, 0.811, 0.820, 0.822, 0.816, 0.823, 0.827,
             0.825, 0.825, 0.824, 0.821]          # dev selection metric every 200 steps, best at step 3000 (index 14)


def first_stop(curve, patience, min_delta=0.0):
    es = EarlyStop(patience, min_delta)
    for i, v in enumerate(curve):
        if es.update(v):
            return i
    return None


def test_patience_5_keeps_the_best_of_verifier1():
    assert first_stop(VERIFIER1, 5) is None


def test_patience_3_stops_verifier1_too_early():
    assert first_stop(VERIFIER1, 3) == 9          # step 2000, before the best at 3000


def test_off_never_stops():
    assert first_stop([0.9, 0.5, 0.4, 0.3, 0.2, 0.1], 0) is None


def test_selection_score_weights_verifier_questions_equally():
    lg = {"dev/verifier/argmax_in_target": 0.0, "dev/verifier_q3/argmax_in_target": 0.6,
          "dev/verifier_frame/argmax_in_target": 0.8, "dev/verifier_frame/w_argmax_in_target": 0.9,
          "dev/verifier_where/argmax_in_target": 0.9}
    assert abs(selection_score(lg) - (0.6 + 0.9 + 0.9) / 3) < 1e-9
    assert selection_score({"loss": 1.0}) is None
