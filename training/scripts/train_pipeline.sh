#!/usr/bin/env bash
# From-scratch training of System I (controller, grounding = locator + selector, verifier), one step per call.
#   bash scripts/train_pipeline.sh <step> [args]
# Steps (in this order; each one only needs the outputs of the previous ones):
#   data          CG-Bench manifests (test = mini / train / dev 700) + VTS four-stage filter of train (GPU: Qwen3-VL-8B;
#                 one shard per GPU of $GPUS); test and dev unfiltered, each row tagged with its filter verdict
#   s1_collect   expert trajectories on train + dev: A = evidence grounding, B = zero-shot OneJev grounding with
#                 evidence labels; oracle verifier (frame keep, where); verifier inputs recorded
#   s1_data       SFT rows: grounding (localize, frame) and verifier (per-frame keep, where; Q3 with budget.verifier_q3)
#   s1_train      grounding LoRA round 1 (localize + frame)
#   s2_train      verifier LoRA round 1; policy set ps1 = grounding1 + verifier1
#   s3_collect    expert controller + ps1 (no oracle): grounding labels + verifier inputs recorded
#   s4_train      grounding2 and verifier2 on the s3 states (+ replay of round 1); ps2 = grounding2 + verifier2
#   s5_collect    expert controller + ps2 (no oracle): the controller's SFT states, with the final grounding / verifier
#   s5_train      controller LoRA: SFT on the expert's actions; ps3_r0 = controller0 + ps2
#   s5_dagger R   DAgger round R (1, 2, ...): the student controller acts, the expert labels, aggregate and retrain
# SFT uses all GPUs when SFT_GPUS is set, otherwise one GPU. Collection runs over GPUS:
# one shard per GPU (run_agent_v3 --shard i/n), merged into runs/<name>/ afterwards. Collection and SFT resume when
# rerun with the same TAG. JEV_RESERVE_GB / JEV_HOLD are optional (shared machines).
# Environment overrides: PY, CFG, MAN, TAG, TRAIN, DEV, TEST, BETA, GPUS, DEV_SIZE, EVID_PAD;
#   data step: CGB_JSON, CGB_MINI, CGB_VIDEOS, LVH_DEV_VIDEOS (LV-Haystack dev source videos, 50); with SHORT_VIDEO=1
#   SHORT_SOURCE=videoitg: VIDEOITG_JSON, VIDEOITG_VIDEOS, VIDEOITG_DEV_VIDEOS (dev videos, 300), or VIDEOITG_MANIFEST
#   (a videoitg_all.jsonl already built); VIDEOITG_POOL (all / sample, below); VIDEOITG_MULTI_SEGMENT (0 / 1, below).
#   SFT: SFT_FIT_N, SFT_FIT_STEPS, SFT_EPOCHS;
#   SFT_GPUS (e.g. all): SFT on every GPU of $GPUS via torchrun (same steps as one GPU).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python}
CFG=${CFG:-configs/default.yaml}
MAN=${MAN:-longvid}                                   # data/manifests/$MAN/, or an absolute manifest directory
case $MAN in /*) MDIR=$MAN ;; *) MDIR=data/manifests/$MAN ;; esac
TAG=${TAG:-p1}                                        # run-name prefix
train_prefix=cgbench
if [ "${LVH_TRAIN:-0}" = 1 ]; then train_prefix=cgbench_lvh; fi
TRAIN=${TRAIN:-${train_prefix}_train_vts.jsonl}
DEV=${DEV:-${train_prefix}_dev_vts.jsonl}
TEST=${TEST:-cgbench_test_vts.jsonl}
BETA=${BETA:-0.3}
DEV_SIZE=${DEV_SIZE:-700}
GPUS=${GPUS:-}
SFT=data/sft
mkdir -p "$SFT" logs
CFG_PAD=$($PY -c "import sys; sys.path.insert(0, '.'); from src.common.config import load_config
print(load_config(sys.argv[1:])['budget']['evidence_pad'])" $CFG)          # evidence = key_spans +- this (seconds)
if [ -n "${EVID_PAD:-}" ] && ! $PY -c "import sys; sys.exit(abs(float(sys.argv[1]) - float(sys.argv[2])) > 1e-9)" "$EVID_PAD" "$CFG_PAD"; then
  echo "EVID_PAD=$EVID_PAD but budget.evidence_pad of $CFG is $CFG_PAD" >&2; exit 1
fi
EVID_PAD=$CFG_PAD
VTS_MERGE_GAP=$(awk -v p="$EVID_PAD" 'BEGIN { print 2 * p }')   # VTS filter: clue segments this close are one (6 s)
ORACLE="budget.verifier_frame_oracle_pad=$EVID_PAD budget.verifier_where_oracle_pad=$EVID_PAD"
RECORD="budget.record_verifier_inputs=true budget.label_pE=true"
INJECT=(--inject-wrong "${INJECT_WRONG:-0}" --inject-max "${INJECT_MAX:-2}"
        --inject-min-prob "${INJECT_MIN_PROB:-0.05}")
# SHORT_VIDEO=1: mixed long + short training. Every collection and its SFT rows (both) also run on the short-video
# manifests (SHORT_MAN) with configs/short.yaml over $CFG (short-video profile: budget halved, evidence +-3 s), as runs
# ${TAG}_sv_*; every SFT trains on the rows of both profiles. Policy sets and SFT runs are shared: ${PTAG}_* (= ${TAG}_*).
SHORT_VIDEO=${SHORT_VIDEO:-0}
# SHORT_SOURCE=videoitg: the short-video data are VideoITG-40K multiple-choice questions (scripts/build_videoitg.py from
# VIDEOITG_JSON / VIDEOITG_VIDEOS in the data step), filtered with the long-video rules: train filtered, dev tagged.
SHORT_SOURCE=${SHORT_SOURCE:-videoitg}
case $SHORT_SOURCE in
  videoitg) SHORT_MAN=${SHORT_MAN:-$PWD/data/manifests/videoitg}
       SHORT_TRAIN=${SHORT_TRAIN:-videoitg_train_vts.jsonl} SHORT_DEV=${SHORT_DEV:-videoitg_dev_vts.jsonl} ;;
  *) echo "SHORT_SOURCE must be videoitg, got $SHORT_SOURCE" >&2; exit 2 ;;
esac
PTAG=$TAG
short_profile() {   # in a subshell: config, manifests, evidence pad and run names of the short-video profile
  CFG="$CFG configs/short.yaml"; MAN=$SHORT_MAN; MDIR=$SHORT_MAN; TAG=${TAG}_sv
  EVID_PAD=$($PY -c "import sys; sys.path.insert(0, '.'); from src.common.config import load_config
print(load_config(sys.argv[1:])['budget']['evidence_pad'])" $CFG)
  ORACLE="budget.verifier_frame_oracle_pad=$EVID_PAD budget.verifier_where_oracle_pad=$EVID_PAD"
}
profile_rows() {    # profile_rows <name>: $SFT/<tag>_<name>_{train,dev}.jsonl of every profile
  local t
  for t in $TAG $([ "$SHORT_VIDEO" != 1 ] || echo ${TAG}_sv); do echo $SFT/${t}_$1_train.jsonl $SFT/${t}_$1_dev.jsonl; done
}

run() { echo "+ $*" >&2; "$@"; }
vts_args() {     # vts_args <split> [source [filter args...]]: the filter_cgbench_vts.py arguments of a split, in VTS_ARGS
  local sp=$1 source=${2:-cgbench}
  shift; [ $# -eq 0 ] || shift
  VTS_ARGS=(--config $CFG --manifest $MDIR/${source}_all.jsonl --split $sp --out $MDIR/${source}_${sp}_vts.jsonl
            --merge-gap "$VTS_MERGE_GAP" "$@")
  # LV-Haystack: one span [min, max] per question, as VTS does for Ego4D (again since 2026-10-20, with the single-segment
  # clues of VTS_VERSION 4; 2026-10-14 to 10-20 the original key-span segments were kept)
  if [ "$source" = lvhaystack ]; then VTS_ARGS+=(--merge-spans); fi
}
vts_filter() {   # vts_filter <split> [source [filter args...]]: one shard per GPU, then merge; both datasets use the same
                 # reader. VTS_CACHE: reuse a finished filter of the same rows and settings (any TAG or frame budget)
  local sp=$1 source=${2:-cgbench} ids=() pids=() i=0 cache=() args=()
  vts_args "$@"; args=("${VTS_ARGS[@]}")
  if [ -n "${VTS_CACHE:-}" ]; then
    cache=(--cache "$VTS_CACHE")
    if run $PY scripts/filter_cgbench_vts.py "${args[@]}" "${cache[@]}" --cache-restore; then return; fi
  fi
  read -r -a ids <<< "${GPUS//,/ }"
  if [ ${#ids[@]} -gt 1 ]; then
    for g in "${ids[@]}"; do
      echo "+ [GPU $g] filter_cgbench_vts --split $sp --shard $i/${#ids[@]}" >&2
      CUDA_VISIBLE_DEVICES=$g $PY scripts/filter_cgbench_vts.py "${args[@]}" --shard "$i/${#ids[@]}" \
          --set "reader.batch_size=${VTS_READER_BATCH_SIZE:-1}" > "logs/vts_${source}_${sp}_shard$i.log" 2>&1 &
      pids+=($!); i=$((i + 1))
    done
    for p in "${pids[@]}"; do wait "$p" || { echo "VTS filter shard failed (see logs/vts_${source}_${sp}_shard*.log)" >&2; exit 1; }; done
  fi
  run $PY scripts/filter_cgbench_vts.py "${args[@]}" "${cache[@]}" --set "reader.batch_size=${VTS_READER_BATCH_SIZE:-1}"
}
data_manifests() {   # the data step's manifests before the filter: CG-Bench, LV-Haystack (LVH_TRAIN=1), VideoITG
                     # (SHORT_SOURCE=videoitg; vitg_args then holds its filter options)
  : "${CGB_JSON:?set CGB_JSON=<cgbench.json>}" "${CGB_MINI:?set CGB_MINI=<cgbench_mini.json>}" "${CGB_VIDEOS:?set CGB_VIDEOS=<mp4 dir>}"
  run $PY scripts/build_cgbench.py --json "$CGB_JSON" --video-root "$CGB_VIDEOS" --mini-json "$CGB_MINI" \
      --dev-size $DEV_SIZE --out $MDIR/cgbench_all.jsonl
  if [ "${LVH_TRAIN:-0}" = 1 ]; then
    # official val = test (VTS's Haystack-Ego4D evaluation), dev = LVH_DEV_VIDEOS source videos held out of train
    run $PY scripts/build_lvhaystack.py --parquet-dir "${LVH_ROOT:?}/annotations" --video-root "${LVH_VIDEOS:?}" \
        --tau "${LVH_TAU:-2}" --dev-videos "${LVH_DEV_VIDEOS:-50}" --out-dir "$MDIR"
  fi
  vitg_args=()
  if [ "$SHORT_VIDEO" = 1 ] && [ "$SHORT_SOURCE" = videoitg ]; then   # same rules as the long videos
    if [ -n "${VIDEOITG_MANIFEST:-}" ]; then     # built already
      mkdir -p "$SHORT_MAN"; cp "$VIDEOITG_MANIFEST" "$SHORT_MAN/videoitg_all.jsonl"
    else
      run $PY scripts/build_videoitg.py --annotations "${VIDEOITG_JSON:?set VIDEOITG_JSON=<video_itg_data.json>}" \
          --video-root "${VIDEOITG_VIDEOS:?set VIDEOITG_VIDEOS=<dir holding LLaVA-Video-178K/>}" \
          --dev-videos "${VIDEOITG_DEV_VIDEOS:-300}" --out-dir "$SHORT_MAN"
    fi
    # Which VideoITG questions go to the filter (VIDEOITG_SELECTED: the manifest is a selection already, filtered as it is):
    #   VIDEOITG_POOL=all (default, the paper's data): every train question (84,139), and VIDEOITG_DEV_LIMIT (300) dev
    #     questions drawn over the duration categories of VIDEOITG_BINS (30-60 s 20%, 1-2 min 30%, 2-3 min 50%), seeded
    #   VIDEOITG_POOL=sample: at most one question per video, VIDEOITG_TRAIN_LIMIT (12000) train and VIDEOITG_DEV_LIMIT
    #     (300) dev questions, seeded (a smaller and faster data step)
    if [ -z "${VIDEOITG_SELECTED:-}" ]; then
      case ${VIDEOITG_POOL:-all} in
        all)
          run $PY scripts/sample_videoitg.py --manifest "$SHORT_MAN/videoitg_all.jsonl" \
              --out "$SHORT_MAN/videoitg_dev_drawn.jsonl" --train 0 --dev "${VIDEOITG_DEV_LIMIT:-300}" \
              --bins "${VIDEOITG_BINS:-30_60_s=0.2,1_2_m=0.3,2_3_m=0.5}"
          run $PY scripts/sample_videoitg.py --manifest "$SHORT_MAN/videoitg_all.jsonl" \
              --out "$SHORT_MAN/videoitg_train_all.jsonl" --rest-of "$SHORT_MAN/videoitg_dev_drawn.jsonl"
          run $PY scripts/sample_videoitg.py --manifest "$SHORT_MAN/videoitg_train_all.jsonl" \
              "$SHORT_MAN/videoitg_dev_drawn.jsonl" --out "$SHORT_MAN/videoitg_all.jsonl" --all ;;
        sample)
          run $PY scripts/sample_videoitg.py --manifest "$SHORT_MAN/videoitg_all.jsonl" --out "$SHORT_MAN/videoitg_all.jsonl" \
              --train "${VIDEOITG_TRAIN_LIMIT:-12000}" --dev "${VIDEOITG_DEV_LIMIT:-300}" ;;
        *) echo "VIDEOITG_POOL must be all or sample, got $VIDEOITG_POOL" >&2; exit 2 ;;
      esac
    fi
    # VIDEOITG_MULTI_SEGMENT=1 (the 2026-10-20 experiment): clues of several segments are kept and their rows marked, so
    # the evidence expert and the where labels follow the uncovered segments; 0: one span [min, max], as LV-Haystack
    vitg_args=(--merge-spans); [ "${VIDEOITG_MULTI_SEGMENT:-0}" != 1 ] || vitg_args=(--multi-segment)
    # VTS_VIDEOITG_STAGES (e.g. 123: VideoITG without stage 4, whose outside-the-clue frames answer
    # 78% of the short videos' questions): the VTS stages VideoITG is filtered with; the long videos keep all four
    if [ -n "${VTS_VIDEOITG_STAGES:-}" ]; then vitg_args+=(--stages "$VTS_VIDEOITG_STAGES"); fi
  fi
}
collect() {   # collect <name> <file> <run_agent args...>; parallel over $GPUS when set
  local name=$1 file=$2; shift 2
  if [ -z "$GPUS" ]; then
    run $PY scripts/run_agent_v3.py --name "$name" --config $CFG --manifest $MAN --file "$file" \
        --batch-questions "${COLLECT_BATCH_QUESTIONS:-8}" "$@" > "logs/$name.log" 2>&1
    return
  fi
  local ids=() i=0 pids=() shards=() queue=()
  read -r -a ids <<< "${GPUS//,/ }"
  local n=${#ids[@]}
  # COLLECT_QUEUE=1 (default, 2026-10-13): a process that finished its shard takes the unclaimed questions of the
  # others (run_agent_v3 --queue), so no GPU idles while one shard is behind; claims live in a new local directory.
  if [ "${COLLECT_QUEUE:-1}" = 1 ]; then queue=(--queue "$(mktemp -d "${TMPDIR:-/tmp}/collect_queue.XXXXXX")"); fi
  for g in "${ids[@]}"; do
    echo "+ [GPU $g] run_agent_v3 --name ${name}_p$i --shard $i/$n $*" >&2
    CUDA_VISIBLE_DEVICES=$g $PY scripts/run_agent_v3.py --name "${name}_p$i" --shard "$i/$n" ${queue[@]+"${queue[@]}"} \
        --config $CFG --manifest $MAN --file "$file" --batch-questions "${COLLECT_BATCH_QUESTIONS:-8}" "$@" \
        > "logs/${name}_p$i.log" 2>&1 &
    pids+=($!); shards+=("runs/${name}_p$i"); i=$((i + 1))
  done
  local failed=() k=0
  for p in "${pids[@]}"; do wait "$p" || failed+=("$k"); k=$((k + 1)); done
  # 2026-10-15: one question's infrastructure error (an unreadable cached feature shard) stopped a shard and with it
  # the job. A failed shard runs once more: run_agent_v3 resumes and retries its failed questions.
  if [ ${#failed[@]} -gt 0 ]; then
    pids=()
    for k in "${failed[@]}"; do
      echo "+ [GPU ${ids[$k]}] retry run_agent_v3 --name ${name}_p$k --shard $k/$n" >&2
      CUDA_VISIBLE_DEVICES=${ids[$k]} $PY scripts/run_agent_v3.py --name "${name}_p$k" --shard "$k/$n" \
          ${queue[@]+"${queue[@]}"} --config $CFG --manifest $MAN --file "$file" \
          --batch-questions "${COLLECT_BATCH_QUESTIONS:-8}" "$@" >> "logs/${name}_p$k.log" 2>&1 &
      pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || { echo "collection shard failed twice (see logs/${name}_p*.log)" >&2; exit 1; }; done
  fi
  # Merge only this invocation's shards (not stale p* directories from a different GPU count); a question a queue left
  # in two shard directories is kept once.
  run $PY scripts/merge_shards.py --out "runs/$name" "${shards[@]}"
  cp "runs/${name}_p0/config.yaml" "runs/$name/config.yaml"
}
rows() {      # rows <out> <split> <manifest file> <roles...> -- <trajectories...>
  local out=$1 split=$2 mfile=$3; shift 3
  local roles=() trajs=()
  while [ $# -gt 0 ] && [ "$1" != "--" ]; do roles+=("$1"); shift; done; shift
  trajs=("$@")
  run $PY scripts/gen_role_sft_from_trajectories.py --config $CFG --manifest "$MAN/$mfile" --split "$split" \
      --evidence-pad $EVID_PAD --roles "${roles[@]}" --trajectories "${trajs[@]}" --out "$out"
}
best() {      # the best dev snapshot of an SFT run; fail loudly if the run did not finish
  [ -f "runs/$1/best_snapshot.txt" ] || { echo "runs/$1/best_snapshot.txt missing: did $1 finish?" >&2; exit 1; }
  cat "runs/$1/best_snapshot.txt"
}
both() {      # both <stage function>: on train and dev; with SHORT_VIDEO=1 also on the short-video profile
  local sp
  for sp in train dev; do "$@" $sp "$([ $sp = train ] && echo $TRAIN || echo $DEV)"; done
  if [ "$SHORT_VIDEO" = 1 ]; then
    for sp in train dev; do ( short_profile; "$@" $sp "$([ $sp = train ] && echo $SHORT_TRAIN || echo $SHORT_DEV)" ); done
  fi
}
sft() {       # sft <name> <data> <decision types...> [-- extra train_sft args]
  local name=$1 data=$2; shift 2; local types=() extra=() role_args=()
  while [ $# -gt 0 ] && [ "$1" != "--" ]; do types+=("$1"); shift; done; [ $# -gt 0 ] && shift; extra=("$@")
  if [ "${VERIFIER_BALANCE:-0}" = 1 ] && [ "${types[*]}" = verifier ]; then
    role_args=(--verifier-balance --balance-clip "${VERIFIER_BALANCE_CLIP:-4}")
  fi
  # Fit checks use a separate run: their weights never initialize the real training.
  # Check the first SFT of each role; replay / DAgger rounds already start from a fitted role.
  if [ "${SFT_FIT_N:-0}" -gt 0 ] && [ ${#extra[@]} -eq 0 ]; then
    local fit_name=${name}_fit${SFT_FIT_N}
    if [ ! -f "runs/$fit_name/PASSED" ]; then
      sft_run --config $CFG --name "$fit_name" --data "$data" \
          --decision-types "${types[@]}" --overfit "$SFT_FIT_N" --steps "${SFT_FIT_STEPS:-200}" --require-fit \
          ${role_args[@]+"${role_args[@]}"}
      touch "runs/$fit_name/PASSED"
    fi
  fi
  sft_run --config $CFG --name "$name" --data "$data" --decision-types "${types[@]}" --epochs "${SFT_EPOCHS:-1}" \
      "${extra[@]}" ${role_args[@]+"${role_args[@]}"}
}
sft_run() {   # train_sft.py on the first GPU, or with SFT_GPUS set on every GPU of $GPUS (torchrun, one rank each)
  local ids=() overrides=()
  read -r -a ids <<< "${GPUS//,/ }"
  local mans=("$MDIR/$TRAIN" "$MDIR/$DEV")
  [ "$SHORT_VIDEO" != 1 ] || mans+=("$SHORT_MAN/$SHORT_TRAIN" "$SHORT_MAN/$SHORT_DEV")
  [ -z "${SFT_EARLY_STOP_PATIENCE:-}" ] || overrides+=("sft.early_stop_patience=$SFT_EARLY_STOP_PATIENCE")
  [ -z "${SFT_EARLY_STOP_MIN_DELTA:-}" ] || overrides+=("sft.early_stop_min_delta=$SFT_EARLY_STOP_MIN_DELTA")
  if [ ${#overrides[@]} -gt 0 ]; then overrides=(--set "${overrides[@]}"); fi
  # frames of these decisions that no readable feature shard holds are encoded again first (train_sft.py has no
  # vision encoder and stops on a missing feature)
  if [ ${#ids[@]} -gt 0 ]; then
    CUDA_VISIBLE_DEVICES=${ids[0]} run $PY scripts/repair_features.py "$@" ${overrides[@]+"${overrides[@]}"} \
        --manifests "${mans[@]}"
  else
    run $PY scripts/repair_features.py "$@" ${overrides[@]+"${overrides[@]}"} --manifests "${mans[@]}"
  fi
  if [ -n "${SFT_GPUS:-}" ] && [ ${#ids[@]} -gt 1 ]; then
    CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${ids[*]}") run $PY -m torch.distributed.run --standalone \
        --nproc_per_node=${#ids[@]} scripts/train_sft.py "$@" ${overrides[@]+"${overrides[@]}"}
  else
    run $PY scripts/train_sft.py "$@" ${overrides[@]+"${overrides[@]}"}
  fi
}

case "${1:-help}" in
data)
  data_manifests
  # dev and test keep the unfiltered distribution (model selection, evaluation); their rows only carry
  # the filter's verdict (row["vts"]), so <source>_{dev,test}_vts.jsonl are tagged, not filtered. Train is filtered.
  vts_filter test cgbench --tag-only
  vts_filter train
  vts_filter dev cgbench --tag-only
  if [ "${LVH_TRAIN:-0}" = 1 ]; then
    vts_filter train lvhaystack
    vts_filter dev lvhaystack --tag-only
    run $PY scripts/mix_long_manifests.py --manifest-dir "$MDIR" --train-out "$MDIR/$TRAIN" --dev-out "$MDIR/$DEV"
  fi
  if [ ${#vitg_args[@]} -gt 0 ]; then
    ( MDIR=$SHORT_MAN; vts_filter train videoitg "${vitg_args[@]}"; vts_filter dev videoitg --tag-only "${vitg_args[@]}" )
  fi ;;
s1_collect)
  c1() { collect ${TAG}_s1A_$1 $2 --controller evidence_expert --localizer evidence --selector evidence \
             --verifier nanojev --set $ORACLE $RECORD
         collect ${TAG}_s1B_$1 $2 --controller evidence_expert --localizer nanojev --selector nanojev \
             --verifier nanojev --label-grounding --set $ORACLE $RECORD; }
  both c1 ;;
s1_data)
  d1() { rows $SFT/${TAG}_grounding1_$1.jsonl $1 $2 localize frame -- \
             runs/${TAG}_s1A_$1/trajectories.jsonl runs/${TAG}_s1B_$1/trajectories.jsonl
         rows $SFT/${TAG}_verifier1_$1.jsonl $1 $2 verifier verifier_frame verifier_where -- \
             runs/${TAG}_s1A_$1/trajectories.jsonl runs/${TAG}_s1B_$1/trajectories.jsonl; }
  both d1
  cat $(profile_rows grounding1) > $SFT/${TAG}_grounding1.jsonl
  cat $(profile_rows verifier1) > $SFT/${TAG}_verifier1.jsonl ;;
s1_train)
  sft ${TAG}_grounding1 $SFT/${TAG}_grounding1.jsonl localize frame ;;
s2_train)
  sft ${TAG}_verifier1 $SFT/${TAG}_verifier1.jsonl verifier
  run $PY scripts/make_policy_set.py --out runs/${TAG}_ps1.json --localize "$(best ${TAG}_grounding1)" \
      --frame "$(best ${TAG}_grounding1)" --verifier "$(best ${TAG}_verifier1)" ;;
s3_collect)
  c3() { collect ${TAG}_s3_$1 $2 --controller evidence_expert --checkpoint runs/${PTAG}_ps1.json --localizer nanojev \
             --selector nanojev --verifier nanojev --label-grounding "${INJECT[@]}" --set $RECORD; }
  both c3 ;;
s4_train)
  d4() { rows $SFT/${TAG}_grounding2_$1.jsonl $1 $2 localize frame -- runs/${TAG}_s3_$1/trajectories.jsonl
         rows $SFT/${TAG}_verifier2_$1.jsonl $1 $2 verifier verifier_frame verifier_where -- runs/${TAG}_s3_$1/trajectories.jsonl; }
  both d4
  cat $(profile_rows grounding2) > $SFT/${TAG}_grounding2.jsonl
  cat $(profile_rows verifier2) > $SFT/${TAG}_verifier2.jsonl
  sft ${TAG}_grounding2 $SFT/${TAG}_grounding2.jsonl localize frame -- --init "$(best ${TAG}_grounding1)" \
      --replay $SFT/${TAG}_grounding1.jsonl
  sft ${TAG}_verifier2 $SFT/${TAG}_verifier2.jsonl verifier -- --init "$(best ${TAG}_verifier1)" \
      --replay $SFT/${TAG}_verifier1.jsonl
  run $PY scripts/make_policy_set.py --out runs/${TAG}_ps2.json --localize "$(best ${TAG}_grounding2)" \
      --frame "$(best ${TAG}_grounding2)" --verifier "$(best ${TAG}_verifier2)" ;;
s5_collect)
  c5() { collect ${TAG}_s5_$1 $2 --controller evidence_expert --checkpoint runs/${PTAG}_ps2.json --localizer nanojev \
             --selector nanojev --verifier nanojev "${INJECT[@]}"; }
  both c5 ;;
s5_train)
  d5() { rows $SFT/${TAG}_controller0_$1.jsonl $1 $2 controller -- runs/${TAG}_s5_$1/trajectories.jsonl; }
  both d5
  cat $(profile_rows controller0) > $SFT/${TAG}_controller0.jsonl
  sft ${TAG}_controller0 $SFT/${TAG}_controller0.jsonl controller
  run $PY scripts/make_policy_set.py --out runs/${TAG}_ps3_r0.json --controller "$(best ${TAG}_controller0)" \
      --localize "$(best ${TAG}_grounding2)" --frame "$(best ${TAG}_grounding2)" --verifier "$(best ${TAG}_verifier2)" ;;
s5_dagger)
  R=${2:?round number (1, 2, ...)}; P=$((R - 1))
  cd_() { collect ${TAG}_dagger${R}_$1 $2 --controller nanojev --checkpoint runs/${PTAG}_ps3_r$P.json \
              --localizer nanojev --selector nanojev --verifier nanojev --label-controller --dagger-beta $BETA "${INJECT[@]}"
          rows $SFT/${TAG}_dagger${R}_$1.jsonl $1 $2 controller -- runs/${TAG}_dagger${R}_$1/trajectories.jsonl; }
  both cd_
  cat $SFT/${TAG}_controller$P.jsonl $(profile_rows dagger$R) > $SFT/${TAG}_controller$R.jsonl  # aggregate
  sft ${TAG}_controller$R $SFT/${TAG}_controller$R.jsonl controller -- --init "$(best ${TAG}_controller$P)"
  run $PY scripts/make_policy_set.py --out runs/${TAG}_ps3_r$R.json --controller "$(best ${TAG}_controller$R)" \
      --localize "$(best ${TAG}_grounding2)" --frame "$(best ${TAG}_grounding2)" --verifier "$(best ${TAG}_verifier2)" ;;
*)
  sed -n '2,25p' "$0"; exit 1 ;;
esac
