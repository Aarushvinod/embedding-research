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
#SBATCH --time=36:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --requeue
# Defaults = RTX 6000 Ada (48GB) — those live on cbcb28/29 only, so the route for CLIP members is
# the preemptible scavenger tier (--requeue + 5k-step checkpoints make preemption cost <= 5k steps).
# CLIP-partition fallback (non-preemptible A6000s):
#   sbatch --partition=clip --account=clip --qos=huge-long --gres=gpu:rtxa6000:1 slurm/train_model.sbatch ...
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtx6000ada:1

set -euo pipefail
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
STEPS=50000

PART="${OUT%.json}_part_${MODEL}.json"
EXTRA=()
[ -n "$BOUNDARY" ] && EXTRA+=(--boundary "$BOUNDARY")

srun python -u -m byte_embed.run_lowresource \
    --only "$MODEL" --out "$PART" \
    --teacher "$TEACHER" --pooling "$POOLING" --steps "$STEPS" \
    --no-baselines --no-efficiency "${EXTRA[@]}"
