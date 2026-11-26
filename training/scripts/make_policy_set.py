"""Write a policy set: one NanoJev checkpoint per role, for run_agent / eval_components / bench_latency /
eval_caption_grounding (--checkpoint, --vision-ckpt, --text-ckpt) and train_sft --init.

python scripts/make_policy_set.py --out runs/x/policy_set.json --localize runs/a/best_snapshot.txt --frame runs/b/best_snapshot.txt
Each value (a checkpoint dir or a best_snapshot.txt pointer) is resolved to a complete checkpoint dir. When
the checkpoint records the decision types it was trained on (train_sft --decision-types), the role must be
among them.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common.utils import read_json  # noqa: E402
from src.training.checkpoint import checkpoint_finetune, resolve_ckpt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    for role in ("controller", "localize", "frame", "verifier"):
        ap.add_argument(f"--{role}", default=None)
    args = ap.parse_args()
    ps = {}
    for role in ("controller", "localize", "frame", "verifier"):
        v = getattr(args, role)
        if v is None:
            continue
        ck = resolve_ckpt(v).resolve()
        if not (ck / "COMPLETE").exists():
            raise SystemExit(f"{role}: {ck} is not a complete checkpoint")
        info = read_json(ck / "base_model.json") if (ck / "base_model.json").exists() else {}
        types = info.get("decision_types")
        if types and role not in types:
            raise SystemExit(f"{role}: {ck} was trained on decision types {types}")
        ps[role] = str(ck)
        print(json.dumps({"role": role, "checkpoint": str(ck), "finetune": checkpoint_finetune(ck),
                          "trained_on": types or "all types"}), flush=True)
    if not ps:
        raise SystemExit("give at least one of --controller / --localize / --frame")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(ps, indent=1) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
