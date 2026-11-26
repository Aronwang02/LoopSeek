#!/usr/bin/env bash
# The five benchmarks with another System II: the search runs once (bash run.sh bench5 -> runs/loopseek_bench5), System II
# is called only at the end and the search does not depend on it, so this re-answers every question from the 32 frames
# that search chose (summary.answer_times of its trajectories.jsonl) with scripts/run_b0.py --frames-from: the same
# answer prompt, frame format, greedy decoding and letter parsing; only the model changes. Then the five-benchmark scores
# (scripts/bench5_report.py) -> runs/<name>.json.
#   bash scripts/answer.sh <model> [name]
#     <model>  a Hugging Face id or a local directory: Qwen3-VL (vLLM engine) or Qwen2.5-VL / Qwen2-VL (HF engine), e.g.
#              Qwen/Qwen2.5-VL-7B-Instruct. Another family: scripts/export_answer_frames.py writes the frames per question,
#              your own code answers, scripts/import_answers.py scores (README).
#     [name]   run name (default ans_<last part of the model name>)
# One process per GPU of GPUS (run.env; "a,b" entries count as two GPUs); the model must fit one GPU (tensor parallel 1).
# FRAMES_FROM (default runs/loopseek_bench5/trajectories.jsonl) picks the search run; ANSWER_BATCH the questions per HF
# batch (default 2; the vLLM engine schedules itself). DRY=1 prints the plan.
# Check: bash scripts/answer.sh Qwen/Qwen3-VL-8B-Instruct gives the scores of bash run.sh bench5 up to numerics.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source run.env; set +a
DRY=${DRY:-0}
MODEL=${1:?usage: bash scripts/answer.sh <model: HF id or local dir> [name]}
base=${MODEL%/}; base=${base##*/}
NAME=${2:-ans_$(echo "$base" | tr -c 'A-Za-z0-9_\n-' '_')}
FRAMES_FROM=${FRAMES_FROM:-$PWD/runs/loopseek_bench5/trajectories.jsonl}
BENCH5_MAN=${BENCH5_MAN:-$DATA/manifests/bench5}
CFG="configs/default.yaml configs/loopseek.yaml"
VLLM_PYTHON=${VLLM_PYTHON:-$PWD/.venv-vllm/bin/python}
if [ "$DRY" = 1 ]; then PY=python
else source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$ENV_NAME"; PY=$(command -v python); fi
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH=$PATH:$(dirname "$VLLM_PYTHON")
log() { echo "== $(date '+%F %T') $*"; }

if [ "$DRY" = 1 ]; then MP=$MODEL; ENGINE=vllm
else
  [ -s "$FRAMES_FROM" ] || { echo "$FRAMES_FROM missing: run the search first (bash run.sh bench5)" >&2; exit 2; }
  [ -s "$BENCH5_MAN/bench5.jsonl" ] || { echo "$BENCH5_MAN/bench5.jsonl missing (bash run.sh data)" >&2; exit 2; }
  # the model: a local directory as is, else downloaded into HF_HOME; its family picks the reader engine
  read -r MP ENGINE < <(HF_HUB_OFFLINE=0 "$PY" - "$MODEL" <<'PY'
import json, os, sys
from pathlib import Path
m = sys.argv[1]
if Path(m).is_dir():
    p = str(Path(m).resolve())
else:
    from huggingface_hub import snapshot_download
    p = snapshot_download(m, token=os.environ.get("HF_TOKEN") or None)
t = json.loads((Path(p) / "config.json").read_text()).get("model_type")
engines = {"qwen3_vl": "vllm", "qwen2_5_vl": "hf", "qwen2_vl": "hf"}
if t not in engines:
    sys.exit(f"{m}: model_type {t!r} is not a Qwen2-VL / Qwen2.5-VL / Qwen3-VL checkpoint, which this repo's reader runs;"
             " use scripts/export_answer_frames.py + your own answerer + scripts/import_answers.py (README)")
print(p, engines[t])
PY
  )
fi
read -r -a groups <<< "$GPUS"
gpus=(); for g in "${groups[@]}"; do gpus+=(${g//,/ }); done
n=${#gpus[@]}
[ "$n" -ge 1 ] || { echo "GPUS is empty in run.env" >&2; exit 2; }
log "answerer $MODEL ($ENGINE engine, $MP) on GPUs ${gpus[*]}, frames from $FRAMES_FROM -> runs/$NAME"
SETS=(reader.engine=$ENGINE models.reader_qwen3vl8b.path=$MP reader.vllm.python=$VLLM_PYTHON project.wandb_mode=disabled)
if [ "$ENGINE" = hf ]; then SETS+=(reader.batch_size=${ANSWER_BATCH:-2}); fi
mkdir -p logs runs
shard() {
  local i=$1 g=${gpus[$1]}
  local cmd=(env CUDA_VISIBLE_DEVICES=$g "$PY" scripts/run_b0.py --config $CFG --manifest "$BENCH5_MAN" --file bench5.jsonl
             --frames-from "$FRAMES_FROM" --name ${NAME}_p$i --shard $i/$n --set "${SETS[@]}")
  if [ "$DRY" = 1 ]; then echo "PLAN ${cmd[*]}"; return 0; fi
  "${cmd[@]}" >> logs/${NAME}_p$i.log 2>&1
}
pids=(); dirs=()
for i in $(seq 0 $((n - 1))); do shard $i & pids+=($!); dirs+=(runs/${NAME}_p$i); done
failed=()
for k in "${!pids[@]}"; do wait "${pids[$k]}" || failed+=("$k"); done
if [ ${#failed[@]} -gt 0 ]; then      # run_b0 resumes from its partial file: a failed shard runs once more
  pids=()
  for k in "${failed[@]}"; do echo "shard $k failed, retrying once (logs/${NAME}_p$k.log)" >&2; shard $k & pids+=($!); done
  for p in "${pids[@]}"; do wait "$p" || { echo "a shard failed twice (logs/${NAME}_p*.log)" >&2; exit 1; }; done
fi
if [ "$DRY" = 1 ]; then echo "PLAN report ${dirs[*]} -> runs/$NAME.json"; exit 0; fi
"$PY" scripts/bench5_report.py --runs "${dirs[@]}" --manifest "$BENCH5_MAN/bench5.jsonl" --out runs/$NAME.json
log "five-bench scores with $MODEL: runs/$NAME.json"
