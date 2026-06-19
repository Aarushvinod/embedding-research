#!/usr/bin/env bash
# Provision + run the ByteEmbed feasibility pass on a fresh RunPod / Vast.ai GPU pod.
# Recommended image: a PyTorch CUDA 12.4+ base (Colab/cloud already ship Blackwell-safe torch).
# Recommended GPU: RTX 4090 (24 GB) is enough for the distillation route; A100 for headroom.
set -euo pipefail

REPO="${REPO:-https://github.com/Aarushvinod/embedding-research.git}"

git clone "$REPO"
cd embedding-research

pip install -q -r requirements-cloud.txt unidecode
pip install -q -e .

python scripts/check_env.py

# Full feasibility pass (~0.5-1 GPU-hr of training + eval on a 4090/A100).
python -m byte_embed.run_feasibility \
  --steps 2000 --batch 64 \
  --out results/byte_embed_cloud.json \
  --save-student results/byte_student.pt

echo "Done. Pull results/byte_embed_cloud.json before terminating the pod."
