"""parse_answer: option texts that start with a capital letter + word must not be read as that letter (2026-10-04)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.scoring import parse_answer  # noqa: E402

L = "ABCDE"


def test_plain_letters_still_parse():
    for t, want in [("B", "B"), ("(B)", "B"), ("B.", "B"), ("B)", "B"), ("B:", "B"), ("b", "B"), (" (E) ", "E"),
                    ("B. Candles.", "B"), ("(C) C is cooking.", "C"), ("Answer: D", "D"), ("Answer: (A)", "A"),
                    ("B, because ...", "B"), ("The answer is B.", "B"), ("the answer is (C)", "C"), ("B\n", "B")]:
        assert parse_answer(t, L) == want, t


def test_echoed_option_text_is_not_a_letter():
    for t in ["C is doing laundry.", "A blue jacket over a red shirt", "A set of keys.", "Answer: C is cooking.",
              "E", "None of the options", "c's actions show"]:
        got = parse_answer(t, "ABCD")
        assert got is None, (t, got)
