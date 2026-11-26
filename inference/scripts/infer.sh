#!/usr/bin/env bash
# LoopSeek inference on a manifest: System I searches every question for its frames, then System II answers once.
# One run_agent_v3 process per entry of GPUS (run.env): "a" puts System I and System II on GPU a; "a,b" puts System I on
# GPU a and System II on GPU b (configs/two_gpu.yaml). The processes share one question queue; their shards are merged
# into runs/<name>.
#   bash scripts/infer.sh bench5                         the five benchmarks ($DATA/manifests/bench5/bench5.jsonl, 10,760 q)
#                                                        -> runs/loopseek_bench5, scores -> runs/loopseek_bench5.json
#   bash scripts/infer.sh smoke                          2 questions per benchmark -> runs/loopseek_smoke
#   bash scripts/infer.sh <manifest dir> <file> <name>   any manifest in this repository's format -> runs/<name>
# A rerun goes on where it stopped (every process skips the questions its shard has done). DRY=1 prints the commands.
# Output per question: runs/<name>/results.jsonl (answer, correct, seconds of System I / System II / environment) and
# trajectories.jsonl (every decision of the search). Logs: logs/<name>_p<i>.log.
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source run.env; set +a
DRY=${DRY:-0}
if [ "$DRY" = 1 ]; then PY=python
else source "$(conda info --base)/etc/profile.d/conda.sh"; conda activate "$ENV_NAME"; PY=$(command -v python); fi
VLLM_PYTHON=${VLLM_PYTHON:-$PWD/.venv-vllm/bin/python}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH=$PATH:$(dirname "$VLLM_PYTHON")
BENCH5_MAN=${BENCH5_MAN:-$DATA/manifests/bench5}
POLICY_SET=${POLICY_SET:-$PWD/data/policy_set.json}
CFG="configs/default.yaml configs/loopseek.yaml"
mode=${1:-bench5}
case $mode in
  bench5) MAN=$BENCH5_MAN FILE=bench5.jsonl NAME=loopseek_bench5 ;;
  smoke)  MAN=$BENCH5_MAN FILE=bench5_smoke.jsonl NAME=loopseek_smoke ;;
  *)      MAN=$1 FILE=${2:?usage: bash scripts/infer.sh <manifest dir> <file> <name>} NAME=${3:?name} ;;
esac
if [ "$DRY" != 1 ]; then
  [ -s "$MAN/$FILE" ] || { echo "$MAN/$FILE missing (bash run.sh data builds the five-benchmark manifests)" >&2; exit 2; }
  [ -s "$POLICY_SET" ] || { echo "$POLICY_SET missing (bash run.sh weights downloads the adapters)" >&2; exit 2; }
fi
read -r -a groups <<< "$GPUS"
n=${#groups[@]}
[ "$n" -ge 1 ] || { echo "GPUS is empty in run.env" >&2; exit 2; }
for g in "${groups[@]}"; do
  [[ $g =~ ^[0-9]+(,[0-9]+)?$ ]] || { echo "GPUS entry '$g' is neither 'a' nor 'a,b'" >&2; exit 2; }
done
mkdir -p logs runs
queue=$(mktemp -d "${TMPDIR:-/tmp}/loopseek_queue.XXXXXX")
shard() {   # shard <i>: one process on GPUS entry i (stdout / stderr appended to its log)
  local i=$1 g=${groups[$1]} cfg=$CFG
  [[ $g == *,* ]] && cfg="$cfg configs/two_gpu.yaml"
  local cmd=(env CUDA_VISIBLE_DEVICES=$g "$PY" scripts/run_agent_v3.py --name ${NAME}_p$i --shard $i/$n --queue "$queue"
             --config $cfg --manifest "$MAN" --file "$FILE" --checkpoint "$POLICY_SET"
             --controller nanojev --localizer nanojev --selector nanojev --verifier nanojev --mode greedy
             --batch-questions "${BATCH_QUESTIONS:-8}" --set reader.vllm.python=$VLLM_PYTHON project.wandb_mode=disabled)
  echo "== $(date '+%F %T') ${NAME}_p$i on GPU(s) $g"
  if [ "$DRY" = 1 ]; then echo "PLAN ${cmd[*]}"; return 0; fi
  "${cmd[@]}" >> logs/${NAME}_p$i.log 2>&1
}
pids=(); shards=()
for i in $(seq 0 $((n - 1))); do shard $i & pids+=($!); shards+=(runs/${NAME}_p$i); done
failed=()
for k in "${!pids[@]}"; do wait "${pids[$k]}" || failed+=("$k"); done
# a failed shard runs once more: it resumes and retries its failed questions
if [ ${#failed[@]} -gt 0 ]; then
  pids=()
  for k in "${failed[@]}"; do echo "shard $k failed, retrying once (logs/${NAME}_p$k.log)" >&2; shard $k & pids+=($!); done
  for p in "${pids[@]}"; do wait "$p" || { echo "a shard failed twice (logs/${NAME}_p*.log)" >&2; exit 1; }; done
fi
if [ "$DRY" = 1 ]; then echo "PLAN merge ${shards[*]} -> runs/$NAME"; exit 0; fi
"$PY" scripts/merge_shards.py --out runs/$NAME "${shards[@]}"
cp runs/${NAME}_p0/config.yaml runs/$NAME/config.yaml
if [ "$mode" = bench5 ] || [ "$mode" = smoke ]; then
  "$PY" scripts/bench5_report.py --runs runs/$NAME --manifest "$MAN/$FILE" --out runs/$NAME.json
  echo "five-benchmark scores: runs/$NAME.json"
fi
echo "done: runs/$NAME (results.jsonl, trajectories.jsonl)"
