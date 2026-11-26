"""Fixed answer-parsing rule shared by every system. No legal letter -> wrong."""
import re

# Multiple-choice only (every benchmark here is MC, checked 2026-10-04). A bare leading letter followed by a space and a
# word is NOT an option letter: option texts often start that way ("C is cooking." in EgoSchema, C = camera wearer;
# "A blue jacket ..." in Video-MME / MVBench / LLaVA-Video), and an answer that echoes such an option text would
# otherwise be read as C / A. "(C) ...", "C.", "C)", "C:", "C" still parse. Revisit this rule if a non-MC task is added.
_LEAD = re.compile(r"^\s*(?:answer\s*[:：]\s*)?(?:\(([A-Z])\)|([A-Z])(?:[\.\):,]|\s(?![A-Z])|$))", re.IGNORECASE)
_PHRASE = re.compile(r"answer\s+is\s*[:：]?\s*\(?([A-Z])\)?(?:[\s\.\):,]|$)", re.IGNORECASE)


def parse_answer(text, letters):
    """Return an option letter from `letters` or None.

    Order: leading letter ('B', '(B)', 'B.', 'Answer: B'), then 'answer is (B)', then an exact
    match of a whole option text is NOT attempted (kept strict and simple)."""
    if text is None:
        return None
    letters = {x.upper() for x in letters}
    t = text.strip()
    for rx in (_LEAD, _PHRASE):
        m = rx.search(t)
        g = next((x for x in m.groups() if x), None) if m else None
        if g and g.upper() in letters:
            return g.upper()
    return None


def is_correct(parsed, answer):
    return parsed is not None and parsed == answer
