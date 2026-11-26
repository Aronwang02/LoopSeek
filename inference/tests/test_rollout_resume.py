"""OOM failures must be retried and must not mark pipeline collection complete."""
import json

import pytest

from scripts.run_agent_v3 import require_no_infrastructure_errors, resume_completed
from src.common.utils import read_jsonl, write_jsonl


def test_resume_keeps_successful_pairs_and_retries_infrastructure_errors(tmp_path):
    traj, results = tmp_path / 'trajectories.jsonl', tmp_path / 'results.jsonl'
    rows = [{'sample_id': str(i), 'status': 'ok' if i < 2 else 'infra_error'} for i in range(8)]
    write_jsonl(traj, rows + [{'sample_id': 'uncommitted', 'status': 'ok'}])
    write_jsonl(results, rows)
    assert resume_completed(traj, results) == {'0', '1'}
    assert read_jsonl(traj) == read_jsonl(results) == rows[:2]
    assert resume_completed(traj, results) == {'0', '1'}


def test_result_without_trajectory_is_not_complete(tmp_path):
    traj, results = tmp_path / 'trajectories.jsonl', tmp_path / 'results.jsonl'
    write_jsonl(results, [{'sample_id': 'missing', 'status': 'ok'}])
    assert resume_completed(traj, results) == set()
    assert read_jsonl(traj) == read_jsonl(results) == []


def test_model_limit_is_completed_and_wrong_answer_is_kept(tmp_path):
    traj, results = tmp_path / 'trajectories.jsonl', tmp_path / 'results.jsonl'
    rows = [{'sample_id': 'wrong', 'status': 'ok', 'correct': False},
            {'sample_id': 'limited', 'status': 'path_too_long', 'correct': False}]
    write_jsonl(traj, rows)
    write_jsonl(results, rows)
    assert resume_completed(traj, results) == {'wrong', 'limited'}
    assert read_jsonl(results) == rows


def test_infrastructure_failure_is_nonzero_and_preserves_failure_ids(tmp_path):
    rows = [{'sample_id': 'ok', 'status': 'ok'}, {'sample_id': 'bad', 'status': 'infra_error'}]
    with pytest.raises(SystemExit, match='1 infrastructure failures'):
        require_no_infrastructure_errors(rows, tmp_path)
    assert json.loads((tmp_path / 'infra_failure.json').read_text()) == {'sample_ids': ['bad'], 'count': 1}
    require_no_infrastructure_errors(rows[:1], tmp_path)
