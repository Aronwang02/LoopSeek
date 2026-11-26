#!/usr/bin/env bash
# LoopSeek inference. Fill in run.env (copy run.env.example) once, then:
#   bash run.sh setup     conda env, System II's vLLM env, OneJev-4B / Qwen3-VL-8B download, CPU tests
#   bash run.sh weights   the LoopSeek adapters from Hugging Face (pinned revision, SHA256SUMS checked) -> data/policy_set.json
#   bash run.sh data      the five benchmarks (Video-MME, MLVU, LongVideoBench, LVBench, CG-Bench mini) -> $DATA/manifests/bench5
#   bash run.sh smoke     2 questions per benchmark: the whole path once (the numbers mean nothing)
#   bash run.sh bench5    all 10,760 questions -> runs/loopseek_bench5.json (scores, latency); per question runs/loopseek_bench5/
#   bash run.sh answer <model> [name]                the frames of bench5 answered by another System II (scripts/answer.sh)
#   bash run.sh infer <manifest dir> <file> <name>   any manifest in this repository's format -> runs/<name>
# Every step can be run again after an interruption: it goes on where it stopped. DRY=1 prints the commands of
# smoke / bench5 / answer / infer.
set -euo pipefail
cd "$(dirname "$0")"
cmd=${1:-help}
[ $# -gt 0 ] && shift
case $cmd in
  setup|weights|data|smoke|bench5|answer|infer)
    [ -f run.env ] || { echo "run.env missing: cp run.env.example run.env and fill in the paths" >&2; exit 2; } ;;
esac
case $cmd in
  setup)   set -a; source run.env; set +a; exec bash scripts/setup_env.sh ;;
  weights) exec bash scripts/weights.sh ;;
  data)    exec bash scripts/eval_data.sh ;;
  smoke)   exec bash scripts/infer.sh smoke ;;
  bench5)  exec bash scripts/infer.sh bench5 ;;
  answer)  exec bash scripts/answer.sh "$@" ;;
  infer)   exec bash scripts/infer.sh "$@" ;;
  *)       sed -n '2,11p' "$0"; exit 1 ;;
esac
