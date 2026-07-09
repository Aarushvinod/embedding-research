#!/bin/bash
# Dispatch the FULL finalized study as SLURM jobs: 12 trainings (6 main grid + 2 boundary arms x 3
# byte sizes) + dependency-chained merge/baseline jobs. Run from the repo root on the cluster.
#
#   bash slurm/submit_all.sh
#
# Flow: 1 precompute job (balanced data + BGE-M3 targets, cached) -> all 12 model jobs depend on it
# (afterok) so nothing races the teacher pass -> one merge job per results file depends on its models;
# the MAIN merge also runs the teacher baseline.
#
# PER-SIZE GPU ROUTING (edit the three flag lines below to retarget):
#   *-large -> scavenger Hopper (H100/H200): the slow, wall-time-bound models on the fastest cards
#   *-base  -> clip rtxa6000 (48GB Ampere)
#   *-small -> clip, any Ampere (next best available; smalls fit anywhere and schedule fastest)
set -euo pipefail

MAIN_OUT="results/retrieval_bgem3.json"
TEACHER="bge-m3"; POOLING="attn"
export STEPS="${STEPS-100000}"       # the CAP; exported so train_model.sbatch inherits it
export PATIENCE="${PATIENCE-10}"     # plateau early stop: 10 x 1000-step windows (0 = off)
MAIN_MODELS=(byte-small subword-small byte-base subword-base byte-large subword-large)
ARM_MODELS=(byte-small byte-base byte-large)

LARGE_FLAGS=(--partition=scavenger --account=scavenger --qos=scavenger --constraint=Hopper --gres=gpu:1)
BASE_FLAGS=(--partition=clip --account=clip --qos=huge-long --gres=gpu:rtxa6000:1)
SMALL_FLAGS=(--partition=clip --account=clip --qos=huge-long --constraint=Ampere --gres=gpu:1)
UTIL_FLAGS=("${SMALL_FLAGS[@]}")     # precompute + merges (GPU jobs, not size-specific)

flags_for() {   # set FL[] to the size-appropriate sbatch flags for a model name
  case "$1" in
    *-large) FL=("${LARGE_FLAGS[@]}") ;;
    *-base)  FL=("${BASE_FLAGS[@]}") ;;
    *)       FL=("${SMALL_FLAGS[@]}") ;;
  esac
}

# 1) precompute (data + teacher targets + efficiency) — everything else depends on it
PRE=$(sbatch --parsable "${UTIL_FLAGS[@]}" --job-name=be-precompute --time=04:00:00 --wrap \
  "python -u -m byte_embed.run_lowresource --only '' --out $MAIN_OUT \
   --teacher $TEACHER --pooling $POOLING --no-baselines")
echo "precompute: $PRE  [${UTIL_FLAGS[*]}]"

# 2) main grid — 6 jobs, routed by size
MAIN_IDS=()
for m in "${MAIN_MODELS[@]}"; do
  flags_for "$m"
  jid=$(sbatch --parsable "${FL[@]}" --dependency=afterok:"$PRE" slurm/train_model.sbatch "$m" "$MAIN_OUT")
  MAIN_IDS+=("$jid"); echo "main $m: $jid  -> ${FL[*]}"
done

# 3) boundary arms — 3 byte sizes per arm, separate results file per arm, routed by size
declare -A ARM_IDS
for arm in teacher random; do
  out="results/retrieval_bgem3_b${arm}.json"
  ids=()
  for m in "${ARM_MODELS[@]}"; do
    flags_for "$m"
    jid=$(sbatch --parsable "${FL[@]}" --dependency=afterok:"$PRE" \
          slurm/train_model.sbatch "$m" "$out" "$arm")
    ids+=("$jid"); echo "arm-$arm $m: $jid"
  done
  ARM_IDS[$arm]=$(IFS=:; echo "${ids[*]}")
done

# 4) merges — main merge also scores the teacher baseline; arm merges skip baselines
MAIN_DEP=$(IFS=:; echo "${MAIN_IDS[*]}")
sbatch --parsable "${UTIL_FLAGS[@]}" --job-name=be-merge-main --time=08:00:00 \
  --dependency=afterok:"$MAIN_DEP" --wrap \
  "python -u -m byte_embed.run_parallel --out $MAIN_OUT --teacher $TEACHER --pooling $POOLING \
   --steps $STEPS --only ''" >/dev/null && echo "merge+baseline (main) queued"
for arm in teacher random; do
  out="results/retrieval_bgem3_b${arm}.json"
  sbatch --parsable "${UTIL_FLAGS[@]}" --job-name=be-merge-$arm --time=02:00:00 \
     --dependency=afterok:"${ARM_IDS[$arm]}" --wrap \
    "python -u -m byte_embed.run_parallel --out $out --teacher $TEACHER --pooling $POOLING \
     --steps $STEPS --boundary $arm --only '' --no-baselines" >/dev/null && echo "merge ($arm) queued"
done
echo "submitted: 1 precompute + 12 trainings + 3 merges (large->Hopper, base->A6000, small->Ampere)"
