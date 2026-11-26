#!/usr/bin/env bash
# Entry point of System I training. Fill in run.env (copy run.env.example) once, then:
#   bash run.sh setup     conda env + models + CPU tests                        (scripts/setup_env.sh)
#   bash run.sh smoke     every training step on 16/8/8 questions, TAG=smoke     (does it run end to end? numbers meaningless)
#   bash run.sh sft       data -> SFT rounds of grounding, verifier and controller -> DAgger
#   bash run.sh status    which steps of $TAG are done
# Steps are the ones of scripts/train_pipeline.sh (see its header). A finished step leaves
# runs/.done/<TAG>.<step>; rerunning `sft` skips finished steps and resumes the unfinished one. To redo a step,
# delete its marker. Output: logs/run_<TAG>.log (this script), logs/<run>.log (each collection), runs/<run>/.
set -euo pipefail
cd "$(dirname "$0")"
if [ "${RUN_ENV_LOADED:-0}" != 1 ] && [ -f run.env ]; then set -a; source run.env; set +a; fi

ENV_NAME=${ENV_NAME:-jev_fa}
GPUS=${GPUS:-0}                       # collection: all GPUs; SFT: all with SFT_GPUS, else the first
export ENV_NAME GPUS

cmd=${1:-help}
if [ "$cmd" = setup ]; then exec bash scripts/setup_env.sh; fi
if [ "$cmd" = smoke ]; then
  export TAG=${SMOKE_TAG:-smoke}
  export WANDB_RUN_GROUP=$TAG TRAIN=${TAG}_train.jsonl DEV=${TAG}_dev.jsonl TEST=${TAG}_test.jsonl \
      DAGGER_ROUNDS=1 SFT_FIT_N=0
fi
export TAG=${TAG:-p1}
# one DAgger round for a new TAG; an existing TAG keeps the rounds it recorded
DAGGER_ROUNDS=${DAGGER_ROUNDS:-$(python3 scripts/dagger_rounds.py "$TAG")}
MAN=${MAN:-longvid}; export MAN
case $MAN in /*) MDIR=$MAN ;; *) MDIR=data/manifests/$MAN ;; esac
export TEST=${TEST:-cgbench_test_vts.jsonl}
DRY=${DRY:-0}
if [ "$cmd" = status ]; then
  shopt -s nullglob
  for m in runs/.done/"$TAG".*; do
    [[ $m == *.lock ]] && continue
    echo "${m##*/}  $(cat "$m")"
  done
  exit 0
fi

if [ "$DRY" != 1 ]; then
  if [ -z "${PY:-}" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME"
    export PY=$(command -v python)
  fi
  mkdir -p logs runs/.done
  if [ -z "${RUNSH_LOGGED:-}" ]; then exec > >(tee -a "logs/run_${TAG}.log") 2>&1; export RUNSH_LOGGED=1; fi
fi
export PY=${PY:-python}
read -r -a GPU_IDS <<< "${GPUS//,/ }"
[ ${#GPU_IDS[@]} -gt 0 ] || { echo 'GPUS is empty' >&2; exit 2; }
export GPUS="${GPU_IDS[*]}" CUDA_VISIBLE_DEVICES=${GPU_IDS[0]}
step() {      # step <marker> <command...>: run unless already done
  local m="runs/.done/${TAG}.$1"; shift
  if [ "$DRY" = 1 ]; then printf 'PLAN %s:' "${m##*/}"; printf ' %q' "$@"; printf '\n'; return; fi
  if [ -f "$m" ]; then echo "== skip $(basename "$m") (done $(cat "$m"))"; return; fi
  echo "== $(date '+%F %T') start ${m##*/}: $*"
  "$@"
  date '+%F %T' > "$m"
  echo "== $(date '+%F %T') done ${m##*/}"
}
P() { bash scripts/train_pipeline.sh "$@"; }
LAST=runs/${TAG}_ps3_r${DAGGER_ROUNDS}.json        # controller after the last DAgger round + grounding2 + verifier2

sft_rounds() {
  for s in s1_collect s1_data s1_train s2_train s3_collect s4_train s5_collect s5_train; do step $s P $s; done
  for r in $(seq 1 "$DAGGER_ROUNDS"); do step s5_dagger$r P s5_dagger $r; done
}
require_data() {
  [ "$DRY" = 1 ] && return
  : "${CGB_JSON:?set CGB_JSON in run.env}" "${CGB_MINI:?}" "${CGB_VIDEOS:?}"
}

case "$cmd" in
smoke)
  require_data
  step split "$PY" scripts/build_cgbench.py --json "${CGB_JSON:-<cgbench.json>}" --video-root "${CGB_VIDEOS:-<videos>}" --mini-json "${CGB_MINI:-<mini.json>}" \
      --dev-size 700 --out $MDIR/cgbench_all.jsonl
  if [ "${LVH_TRAIN:-0}" = 1 ]; then
    step split_lvh "$PY" scripts/build_lvhaystack.py --parquet-dir "$LVH_ROOT/annotations" --video-root "$LVH_VIDEOS" \
        --tau "${LVH_TAU:-2}" --dev-videos "${LVH_DEV_VIDEOS:-50}" --out-dir "$MDIR"
  fi
  step pick "$PY" - <<EOF
import json, os
rows = [json.loads(l) for l in open("$MDIR/cgbench_all.jsonl")]
lvh = ([json.loads(l) for l in open("$MDIR/lvhaystack_all.jsonl")] if os.environ.get('LVH_TRAIN') == '1' else [])
for split, n in (("train", 16), ("dev", 8), ("test", 8)):
    pick = [r for r in rows if r["split"] == split and len(r["key_spans"]) == 1][:n]
    if os.environ.get('LVH_TRAIN') == '1' and split != 'test':
        other = [r for r in lvh if r['split'] == split][:n // 2]
        if len(other) != n // 2:
            raise SystemExit(f'smoke needs {n // 2} LV-Haystack {split} rows, found {len(other)}')
        pick = pick[:n - len(other)] + other
    if len(pick) != n:
        raise SystemExit(f"smoke needs {n} {split} rows with one key span, found {len(pick)}")
    with open(f"$MDIR/${TAG}_{split}.jsonl", "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in pick)
    print(split, len(pick))
EOF
  sft_rounds
  if [ "$DRY" != 1 ]; then echo "smoke run finished (numbers are meaningless on 16 questions). Next: bash run.sh sft"; fi ;;
sft)
  if [ "${PREPARED_DATA:-0}" != 1 ]; then require_data; step data P data; fi
  sft_rounds ;;
*)
  sed -n '2,9p' "$0"; exit 1 ;;
esac
