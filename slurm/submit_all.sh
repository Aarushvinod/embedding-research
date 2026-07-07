#!/bin/bash
# Dispatch the FULL finalized study as SLURM jobs: 12 trainings (6 main grid + 2 boundary arms x 3
# byte sizes) + dependency-chained merge/baseline jobs. Run from the repo root on the cluster.
#
#   bash slurm/submit_all.sh
#
# Flow: 1 precompute job (balanced data + BGE-M3 targets, cached) -> all 12 model jobs depend on it
# (afterok) so nothing races the teacher pass -> one merge job per results file depends on its
# models; the MAIN merge also runs the teacher baseline.
set -euo pipefail

MAIN_OUT="results/retrieval_bgem3.json"
TEACHER="bge-m3"; POOLING="attn"; STEPS=50000
MAIN_MODELS=(byte-small subword-small byte-base subword-base byte-large subword-large)
ARM_MODELS=(byte-small byte-base byte-large)

# Cluster flags — DEFAULT = UMD Nexus / CLIP lab. Override via env, or set empty to omit a flag
# entirely (e.g. PARTITION= ACCOUNT= QOS= bash slurm/submit_all.sh on a different cluster).
#   GRES: default 'gpu:1' takes any free GPU; pin a type with GRES=gpu:rtxa6000:1 / gpu:a100:1.
#   QoS: huge-long is CLIP's long-job QoS (check exact limits on-cluster with `show_qos`);
#        the standard 'default' QoS caps mem at 32G, below train_model.sbatch's 48G request.
PARTITION="${PARTITION-clip}"
ACCOUNT="${ACCOUNT-clip}"
QOS="${QOS-huge-long}"
GRES="${GRES-gpu:rtxa6000:1}"       # pinned to the 48GB A6000s; loosen with GRES=gpu:1
CONSTRAINT="${CONSTRAINT-Ampere}"   # any modern clip card; excludes Pascal/Turing (see train_model.sbatch)
# H100/H200s live on CML/Vulcan nodes, NOT the clip partition — but scavenger's node list covers
# them (verified: cml[30-36] + vulcan46 are in scontrol show partition scavenger). Preemptible;
# --requeue + 5k-step checkpoints make preemption cheap.
#   ARMS_VIA_SCAVENGER=1  -> main grid on clip (SFLAGS above), boundary arms on scavenger Hopper.
#   Or route EVERYTHING there: PARTITION=scavenger ACCOUNT=scavenger QOS=scavenger \
#     CONSTRAINT=Hopper GRES=gpu:1 bash slurm/submit_all.sh
ARMS_VIA_SCAVENGER="${ARMS_VIA_SCAVENGER-0}"
SCAV_FLAGS=(--partition=scavenger --account=scavenger --qos=scavenger
            --constraint=Hopper --gres=gpu:1)
SFLAGS=()
[ -n "$PARTITION" ]  && SFLAGS+=(--partition="$PARTITION")
[ -n "$ACCOUNT" ]    && SFLAGS+=(--account="$ACCOUNT")
[ -n "$QOS" ]        && SFLAGS+=(--qos="$QOS")
[ -n "$GRES" ]       && SFLAGS+=(--gres="$GRES")
[ -n "$CONSTRAINT" ] && SFLAGS+=(--constraint="$CONSTRAINT")

sb() { sbatch --parsable "${SFLAGS[@]}" "$@"; }

# 1) precompute (data + teacher targets + efficiency) — everything else depends on it
PRE=$(sb --job-name=be-precompute --gres=gpu:1 --time=04:00:00 --wrap \
  "python -u -m byte_embed.run_lowresource --only '' --out $MAIN_OUT \
   --teacher $TEACHER --pooling $POOLING --no-baselines")
echo "precompute: $PRE"

# 2) main grid — 6 jobs
MAIN_IDS=()
for m in "${MAIN_MODELS[@]}"; do
  jid=$(sb --dependency=afterok:"$PRE" slurm/train_model.sbatch "$m" "$MAIN_OUT")
  MAIN_IDS+=("$jid"); echo "main $m: $jid"
done

# 3) boundary arms — 3 byte sizes per arm, separate results file per arm.
#    ARMS_VIA_SCAVENGER=1 routes these to preemptible Hopper (H100/H200) nodes.
declare -A ARM_IDS
for arm in teacher random; do
  out="results/retrieval_bgem3_b${arm}.json"
  ids=()
  for m in "${ARM_MODELS[@]}"; do
    if [ "$ARMS_VIA_SCAVENGER" = 1 ]; then
      jid=$(sbatch --parsable "${SCAV_FLAGS[@]}" --dependency=afterok:"$PRE" \
            slurm/train_model.sbatch "$m" "$out" "$arm")
    else
      jid=$(sb --dependency=afterok:"$PRE" slurm/train_model.sbatch "$m" "$out" "$arm")
    fi
    ids+=("$jid"); echo "arm-$arm $m: $jid"
  done
  ARM_IDS[$arm]=$(IFS=:; echo "${ids[*]}")
done

# 4) merges — main merge also scores the teacher baseline; arm merges skip baselines
MAIN_DEP=$(IFS=:; echo "${MAIN_IDS[*]}")
sb --job-name=be-merge-main --gres=gpu:1 --time=08:00:00 --dependency=afterok:"$MAIN_DEP" --wrap \
  "python -u -m byte_embed.run_parallel --out $MAIN_OUT --teacher $TEACHER --pooling $POOLING \
   --steps $STEPS --only ''" >/dev/null && echo "merge+baseline (main) queued"
for arm in teacher random; do
  out="results/retrieval_bgem3_b${arm}.json"
  sb --job-name=be-merge-$arm --gres=gpu:1 --time=02:00:00 \
     --dependency=afterok:"${ARM_IDS[$arm]}" --wrap \
    "python -u -m byte_embed.run_parallel --out $out --teacher $TEACHER --pooling $POOLING \
     --steps $STEPS --boundary $arm --only '' --no-baselines" >/dev/null && echo "merge ($arm) queued"
done
echo "submitted: 1 precompute + 12 trainings + 3 merges"
