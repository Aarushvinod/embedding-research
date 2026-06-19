# Running ByteEmbed on a rented cloud GPU

Use this if Colab is unavailable/restricted, or for a clean reproducible run.

## Cost (verified June 2026 pricing)
| GPU | Provider | ~$/hr | Full feasibility pass (~30 GPU-hr realistic) |
|-----|----------|-------|----------------------------------------------|
| RTX 4090 24 GB | RunPod | $0.34–0.69 | **~$12** (cap ~$25) |
| RTX 4090 24 GB | Vast.ai (interruptible) | $0.09–0.59 | ~$9 |
| A100 80 GB | Spheron/Vast | $1.0–1.2 | ~$30 (faster, bigger batches) |

The distillation route fits a **24 GB 4090**; you only need an A100 for larger batches or
the optional contrastive objective.

## RunPod
1. Create a pod: "RTX 4090", a PyTorch template, ~30 GB volume.
2. In the web terminal:
   ```bash
   curl -sSL https://raw.githubusercontent.com/Aarushvinod/embedding-research/main/byte_embed/cloud/setup.sh | bash
   ```
   (or `git clone` then `bash byte_embed/cloud/setup.sh`)
3. Download `results/byte_embed_cloud.json` (and `byte_student.pt`) before terminating.

## Vast.ai
Same, but pick an interruptible RTX 4090 offer for the cheapest rate; keep checkpoints on
the persistent volume in case of preemption.

## Docker (optional, fully reproducible)
Any recent `pytorch/pytorch` CUDA image works; then run `setup.sh`. A dedicated Dockerfile
can be added if you want pinned versions.
