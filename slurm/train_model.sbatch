#!/bin/bash
# One student model per job. Writes the SAME part-file convention run_parallel merges, so Colab
# sessions and SLURM jobs are interchangeable.
#
#   sbatch train_model.sbatch <model-name> <out.json> [boundary]
#   e.g. sbatch train_model.sbatch byte-large  results/retrieval_bgem3.json
#        sbatch train_model.sbatch byte-base   results/retrieval_bgem3_bteacher.json teacher
#
#SBATCH --job-name=byteembed
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
# Defaults = RTX 6000 Ada (48GB) — those live on cbcb28/29 only, so the route for CLIP members is
# the preemptible scavenger tier (--requeue + 5k-step checkpoints make preemption cost <= 5k steps).
# CLIP-partition fallback (non-preemptible A6000s):
#   sbatch --partition=clip --account=clip --qos=huge-long --gres=gpu:rtxa6000:1 slurm/train_model.sbatch ...
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtx6000ada:1

set -euo pipefail
# Content-fair char truncation makes byte sequences ~3.5x longer (non-Latin) -> ~4x the activation
# memory. expandable_segments reclaims fragmented reserved memory (the OOM error explicitly suggests
# it), which recovers the last few hundred MB; byte models still need a >=40GB card (see submit_all.sh).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MODEL="$1"
OUT="$2"
BOUNDARY="${3:-}"

# Record the allocated hardware at the top of the log — with untyped --gres=gpu:1 the scheduler may
# hand you anything from a 1080Ti to an H100, and throughput differs ~10x. Pin a class of card via
# e.g. sbatch --gres=gpu:rtxa6000:1, or (where node features are defined) accept several fast types:
#   sbatch --constraint="rtxa6000|rtx6000ada|a100" --gres=gpu:1 ...
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "node=$(hostname)  model=$MODEL  boundary=${BOUNDARY:-none}"

TEACHER="bge-m3"
POOLING="attn"
STEPS="${STEPS-100000}"        # the CAP (patience can stop earlier); env-overridable
PATIENCE="${PATIENCE-10}"      # stop after 10 x 1000-step windows w/o loss improvement (0 = off)

PART="${OUT%.json}_part_${MODEL}.json"
EXTRA=()
[ -n "$BOUNDARY" ] && EXTRA+=(--boundary "$BOUNDARY")

# AUTO-RESUME across the wall: --signal delivers SIGUSR1 ~5 min before the TIME LIMIT; the trap
# requeues THIS job, which restarts and resumes from the latest 5k-step checkpoint. run_lowresource
# writes the part file and skips an already-complete model, so requeues stop by themselves once
# training finishes -- no manual resubmit for multi-wall byte jobs. (srun is backgrounded so `wait`
# is interruptible by the trap; a foreground srun would block it.)
trap 'echo "[wall] SIGUSR1 -> requeue $SLURM_JOB_ID (resumes from checkpoint)"; scontrol requeue "$SLURM_JOB_ID" || true' USR1

srun python -u -m byte_embed.run_lowresource \
    --only "$MODEL" --out "$PART" \
    --teacher "$TEACHER" --pooling "$POOLING" --steps "$STEPS" --patience "$PATIENCE" \
    --no-baselines --no-efficiency "${EXTRA[@]}" &
wait
