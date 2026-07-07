#!/bin/bash
# One student model per job. Writes the SAME part-file convention run_parallel merges, so Colab
# sessions and SLURM jobs are interchangeable.
#
#   sbatch train_model.sbatch <model-name> <out.json> [boundary]
#   e.g. sbatch train_model.sbatch byte-large  results/retrieval_bgem3.json
#        sbatch train_model.sbatch byte-base   results/retrieval_bgem3_bteacher.json teacher
#
#SBATCH --job-name=byteembed
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=36:00:00
#SBATCH --output=slurm-%x-%j.out
# Defaults = UMD Nexus / CLIP lab (override on the command line or via submit_all.sh env vars).
# NOTE: 48G mem exceeds the 'default' QoS cap (32G) -> huge-long (or medium + lower --mem).
#SBATCH --partition=clip
#SBATCH --account=clip
#SBATCH --qos=huge-long

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
