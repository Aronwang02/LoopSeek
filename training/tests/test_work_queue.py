"""Shared collection queue (src/rollout/work_queue.py, run_agent_v3 --queue) and the shard merge
(scripts/merge_shards.py), 2026-10-13: a process that finished its shard takes the other shards' questions. CPU only."""
import json

from scripts.merge_shards import merge
from src.rollout.work_queue import Claims, completed_ids, home_names, next_batch, work_order

ROWS = [{"sample_id": f"q{k}"} for k in range(10)]


def ids(rows):
    return [r["sample_id"] for r in rows]


def test_own_shard_first_then_the_others_from_their_ends():
    assert ids(work_order(ROWS, 1, 3)) == ["q1", "q4", "q7", "q8", "q5", "q2", "q9", "q6", "q3", "q0"]
    assert home_names(ROWS, "s3_train", 3)["q7"] == "s3_train_p1"     # seeds and ids stay those of rows[1::3]


def test_every_question_runs_once_across_processes(tmp_path):
    done, ran = {"q0"}, []                                             # q0 was done before this launch
    queues = [Claims(tmp_path / "queue") for _ in range(3)]
    orders, pos = [work_order(ROWS, i, 3) for i in range(3)], [0, 0, 0]
    while True:                                                        # interleaved batches of 2 from 3 processes
        progress = False
        for i in range(3):
            batch, pos[i] = next_batch(orders[i], pos[i], done, queues[i], 2)
            ran += ids(batch)
            progress |= bool(batch)
        if not progress:
            break
    assert sorted(ran) == sorted(f"q{k}" for k in range(1, 10))


def test_completed_ids_need_both_files_and_skip_a_line_being_written(tmp_path):
    d = tmp_path / "s_p0"
    d.mkdir()
    (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in (
        {"sample_id": "a"}, {"sample_id": "b", "status": "infra_error"}, {"sample_id": "c"})))
    (d / "trajectories.jsonl").write_text(json.dumps({"sample_id": "a"}) + "\n" + json.dumps({"sample_id": "b"})
                                          + "\n" + '{"sample_id": "c", "turns": [')
    assert completed_ids(d) == {"a"}
    assert completed_ids(tmp_path / "missing") == set()


def test_merge_keeps_shard_order_and_drops_a_duplicate(tmp_path):
    shards = []
    for k, qs in enumerate((["a", "b"], ["c", "a"])):
        d = tmp_path / f"s_p{k}"
        d.mkdir()
        for kind in ("results", "trajectories"):
            (d / f"{kind}.jsonl").write_text("".join(json.dumps({"sample_id": q, "kind": kind, "shard": k}) + "\n"
                                                     for q in qs))
        shards.append(d)
    assert merge(tmp_path / "s", shards) == {"trajectories": 3, "results": 3}
    rows = [json.loads(l) for l in (tmp_path / "s/trajectories.jsonl").read_text().splitlines()]
    assert [(r["sample_id"], r["shard"]) for r in rows] == [("a", 0), ("b", 0), ("c", 1)]
