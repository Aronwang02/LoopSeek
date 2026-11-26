#!/usr/bin/env bash
# The LoopSeek System I adapters (controller, grounding = Locator and Selector, verifier) from Hugging Face at a pinned
# revision, checked against their SHA256SUMS, then data/policy_set.json with this machine's absolute paths (read by
# scripts/infer.sh). LOOPSEEK_REPO / LOOPSEEK_REV / LOOPSEEK_DIR override the source and the destination.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source run.env; set +a
REPO=${LOOPSEEK_REPO:-Chengzhong-Wang/LoopSeek}
REV=${LOOPSEEK_REV:-4d549c1f594314e2b18189f1c373212e8ab167a8}
DST=${LOOPSEEK_DIR:-$HF_HOME/loopseek}
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
HF_HUB_OFFLINE=0 python - "$REPO" "$REV" "$DST" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, rev, dst = sys.argv[1:4]
snapshot_download(repo, revision=rev, local_dir=dst)
PY
(cd "$DST" && sha256sum -c --quiet SHA256SUMS)
python - "$DST" data/policy_set.json <<'PY'
import json, sys
from pathlib import Path
src, out = Path(sys.argv[1]), Path(sys.argv[2])
rel = json.loads((src / "policy_set.json").read_text())
ps = {role: str((src / d).resolve()) for role, d in rel.items()}
for role, d in ps.items():
    assert (Path(d) / "COMPLETE").exists() and (Path(d) / "trainable.pt").exists(), (role, d)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(ps, indent=1) + "\n")
print(out.read_text())
PY
