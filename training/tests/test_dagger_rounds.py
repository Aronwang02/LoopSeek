"""CPU: scripts/dagger_rounds.py, the DAgger rounds of a TAG (2026-10-20: one for a new TAG, the recorded ones else)."""
import json
import sys

from conftest import ROOT

sys.path.insert(0, str(ROOT))
from scripts.dagger_rounds import recorded_rounds  # noqa: E402


def test_new_tag_trains_one_round_and_existing_tags_keep_theirs(tmp_path):
    assert recorded_rounds("new", tmp_path) == 1
    (tmp_path / "old_experiment.json").write_text(json.dumps({"env": {"DAGGER_ROUNDS": "2"}}))
    assert recorded_rounds("old", tmp_path) == 2
    (tmp_path / "cut_experiment.json").write_text("{")              # unreadable record: the default
    assert recorded_rounds("cut", tmp_path) == 1
