"""Comparison deciders on the shared interface (stub reader, CPU): Qwen choice parsing and
fallback, uniform frames, scripted controller, random decider legality, localization metrics."""
from src.baselines import deciders as D
from src.evaluation.localization import evidence_hit, interval_iou, localization_iou


def test_localization_metrics():
    assert interval_iou((0, 10), (5, 15)) == 5 / 15
    ann_any = {"key_spans": [[10, 20], [50, 60]], "need": "any"}
    ann_all = {"key_spans": [[10, 20], [50, 60]], "need": "all"}
    assert localization_iou((10, 20), ann_any) == 1.0
    assert localization_iou((10, 60), ann_all) == 1.0
    assert evidence_hit([12.0, 90.0], ann_any) and not evidence_hit([12.0, 90.0], ann_all)
    assert evidence_hit([12.0, 55.0], ann_all)
