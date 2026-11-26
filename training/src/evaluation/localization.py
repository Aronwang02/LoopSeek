"""Localization and frame-selection evidence metrics (update.md §7).

Annotation per question (data/annotations/mve_keyspans.json):
  key_spans: [[s, e], ...]   time spans that contain the evidence the answer needs
  need: "any" | "all"        any span suffices, or every span (two-moment questions) is required
Localization IoU of a chosen window = max IoU over key spans (for need="all": IoU with the span
hull [min s, max e]). The full-video window usually contains the event but is not counted as a
precise localization success unless its IoU passes the threshold like any other window.
"""


def interval_iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def localization_iou(window, ann):
    spans = ann["key_spans"]
    if ann.get("need", "any") == "all" and len(spans) > 1:
        return interval_iou(window, (min(s for s, _ in spans), max(e for _, e in spans)))
    return max(interval_iou(window, s) for s in spans)


def evidence_hit(times, ann, tol=0.0):
    """Do the selected frame times cover the needed evidence? need='any': at least one frame in some
    span; need='all': every span has a selected frame."""
    covered = [any(s - tol <= t <= e + tol for t in times) for s, e in ann["key_spans"]]
    return all(covered) if ann.get("need", "any") == "all" else any(covered)
