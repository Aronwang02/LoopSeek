"""DAgger rounds of a TAG: the DAGGER_ROUNDS recorded in runs/<TAG>_experiment.json, else 1.
2026-10-20: a new TAG trains one round, since round 2 never improved the controller (its dev imitation accuracy
moved by at most 0.7 points). An existing TAG keeps the rounds it recorded, so the earlier two-round runs resume and are
evaluated on their own final policy set (runs/<TAG>_ps3_r2.json).
usage: python3 scripts/dagger_rounds.py <TAG>     (prints the number; run from the repository root)
"""
import json
import sys
from pathlib import Path

DEFAULT = 1


def recorded_rounds(tag, runs="runs"):
    try:
        value = json.loads((Path(runs) / f"{tag}_experiment.json").read_text())["env"]["DAGGER_ROUNDS"]
    except (OSError, ValueError, KeyError, TypeError):
        return DEFAULT
    return int(value) if value not in (None, "") else DEFAULT


if __name__ == "__main__":
    print(recorded_rounds(sys.argv[1]))
