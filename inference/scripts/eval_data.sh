#!/usr/bin/env bash
# The five-benchmark evaluation data: CPU and network, no GPU. Every part skips what is complete, so run it again after an
# interruption. Paths from run.env: DATA (videos and manifests go below it) and CGB_ROOT (the CG-Bench download).
# A Hugging Face token must be available (HF_TOKEN, or huggingface-cli login): MLVU and LongVideoBench are gated, and its
# account must have accepted both on their pages.
#   CG-Bench (for CG-Bench mini)   CG-Bench/CG-Bench, ~350 GB                   scripts/prepare_cgbench.py
#   Video-MME short / medium / long lmms-lab/Video-MME, no subtitles            scripts/prepare_videomme.py
#   MLVU dev (7 multiple-choice tasks) MLVU/MVLU, ~198 GiB                       scripts/prepare_eval_benches.py mlvu
#   LongVideoBench val             longvideobench/LongVideoBench, ~29 GB        scripts/prepare_eval_benches.py longvideobench
#   LVBench                        zai-org/LVBench + mirror AIWinter/LVBench, 62 GB  scripts/prepare_eval_benches.py lvbench
# then scripts/build_bench5.py -> $DATA/manifests/bench5/{bench5,lvbench,bench5_smoke,lvbench_smoke}.jsonl (10,760 q).
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source run.env; set +a
E=${DATA:?set DATA in run.env}
CGB=${CGB_ROOT:-$E/cgbench}
M=$E/manifests
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
PY=$(command -v python)
export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 PYTHONUNBUFFERED=1
log() { echo "== $(date '+%F %T') $*"; }
"$PY" -c "from huggingface_hub import get_token; import sys; sys.exit(0 if get_token() else 1)" || {
  echo "no Hugging Face token: export HF_TOKEN=... (its account must accept MLVU/MVLU and longvideobench/LongVideoBench)" >&2
  exit 2; }
mkdir -p "$E" "$M"

log "CG-Bench -> $CGB"
if ! "$PY" scripts/prepare_cgbench.py --root "$CGB" --check-only; then
  "$PY" scripts/prepare_cgbench.py --root "$CGB" --scratch "$E/.scratch/cgbench" --workers 4
fi
log "Video-MME -> $E/videomme"
"$PY" scripts/prepare_videomme.py --root "$E/videomme" --download --scratch "$E/.scratch/videomme" --out-dir "$M/videomme"
log "MLVU dev -> $E/mlvu"
"$PY" scripts/prepare_eval_benches.py mlvu --root "$E/mlvu" --out "$M"
log "LongVideoBench val -> $E/longvideobench"
"$PY" scripts/prepare_eval_benches.py longvideobench --root "$E/longvideobench" --out "$M"
log "LVBench -> $E/lvbench"
"$PY" scripts/prepare_eval_benches.py lvbench --root "$E/lvbench" --out "$M"
log "CG-Bench mini rows"
"$PY" scripts/build_cgbench.py --json "$CGB/cgbench.json" --video-root "$CGB/videos" --mini-json "$CGB/cgbench_mini.json" \
    --dev-size 700 --out "$M/cgbench_all.jsonl"
log "five-benchmark manifest -> $M/bench5"
"$PY" scripts/build_bench5.py --cgbench-all "$M/cgbench_all.jsonl" --videomme-dir "$M/videomme" \
    --mlvu "$M/mlvu_dev.jsonl" --lvb "$M/longvideobench_val.jsonl" --lvbench "$M/lvbench.jsonl" --out "$M/bench5"
log "evaluation data done"
