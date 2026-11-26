"""Shared work of one collection's shard processes on one node (run_agent_v3 --queue, 2026-10-13). No model dependencies.

Static shards (run_agent_v3 --shard i/n) left GPUs idle: a shard that was slower, or that had to be collected again
(its trajectory file was truncated by an I/O error), kept one GPU busy for hours after the others had finished.
With a queue every process runs its own shard's questions first, then the unclaimed questions of the other shards,
from their ends. A question is claimed by creating a file in the node-local queue directory (O_EXCL), so no two
processes of a launch run it; a relaunch uses a new queue directory and skips what any shard directory already holds.
Results go to the directory of the process that ran the question. Episode seeds and ids keep the question's own shard
(home), so a question gives the same episode whichever process runs it.
"""
import hashlib
import json
import os
from pathlib import Path


def home_names(rows, base, n):
    """sample_id -> run name of the shard the question belongs to: rows[i::n] -> <base>_p<i>."""
    return {r["sample_id"]: f"{base}_p{k % n}" for k, r in enumerate(rows)}


def work_order(rows, si, n):
    """Candidates of process si: its own shard in order, then the other shards from their ends, starting with the next
    shard, so the processes that finished early spread over different shards."""
    order = list(rows[si::n])
    for k in range(1, n):
        order += rows[(si + k) % n::n][::-1]
    return order


def completed_ids(run_dir):
    """Questions a run directory holds both a result (not an infrastructure error) and a trajectory for. Read-only and
    tolerant of a line its own process is writing or rewriting at that moment."""
    def ids(path, keep):
        out = set()
        try:
            with open(path) as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(row, dict) and keep(row):
                        out.add(row.get("sample_id"))
        except OSError:
            pass
        return out
    run_dir = Path(run_dir)
    return (ids(run_dir / "results.jsonl", lambda r: r.get("status") != "infra_error")
            & ids(run_dir / "trajectories.jsonl", lambda r: True))


class Claims:
    """One file per claimed question in a node-local directory shared by the processes of one launch."""

    def __init__(self, queue_dir):
        self.dir = Path(queue_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def claim(self, sample_id):
        try:
            os.close(os.open(self.dir / hashlib.sha1(str(sample_id).encode()).hexdigest(),
                             os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            return True
        except FileExistsError:
            return False


def next_batch(order, pos, done, claims, size):
    """(the next batch: up to `size` rows of `order` from `pos` that are not done and this process claimed, new pos)."""
    batch = []
    while pos < len(order) and len(batch) < size:
        row = order[pos]
        pos += 1
        if row["sample_id"] in done or (claims is not None and not claims.claim(row["sample_id"])):
            continue
        batch.append(row)
    return batch, pos
