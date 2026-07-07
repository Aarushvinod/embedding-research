"""Export trained checkpoints: strip optimizer state (~1/3 the size) and optionally push to the
Hugging Face Hub. Training checkpoints hold {model, opt, step}; the AdamW state is 2x the model, so
byte-large drops ~10.4 GB -> ~3.5 GB. Exports keep the {"model", "step"} layout, so `reeval` and the
STS warm-start load them unchanged.

  python export_checkpoints.py                                   # checkpoints/*_bge-m3*.pt -> export/
  python export_checkpoints.py --push <you>/byteembed-retrieval --private
  # (--push needs `hf auth login` with a WRITE token first)

Sizes (fp32, model-only): byte 0.9/1.7/3.5 GB, subword 0.6/1.1/2.3 GB (S/B/L) — the full 12-model
study exports to ~22 GB, far past git territory (and this repo is PUBLIC): use the Hub, not git.
Results JSONs are a few MB — commit those to the repo (`git add -f results/retrieval_*.json`;
results/ is gitignored, -f is deliberate).
"""
from __future__ import annotations

import argparse
import glob
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="checkpoints/*_bge-m3*.pt",
                    help="which checkpoints to export (default: every BGE-M3-run model incl. arms)")
    ap.add_argument("--outdir", default="export")
    ap.add_argument("--push", default=None,
                    help="HF Hub repo id to upload the exports to, e.g. <user>/byteembed-retrieval")
    ap.add_argument("--private", action="store_true", help="create the Hub repo private")
    a = ap.parse_args()

    import torch

    files = sorted(glob.glob(a.pattern))
    if not files:
        raise SystemExit(f"no checkpoints match {a.pattern!r}")
    os.makedirs(a.outdir, exist_ok=True)
    for f in files:
        ck = torch.load(f, map_location="cpu")
        out = os.path.join(a.outdir, os.path.basename(f))
        torch.save({"step": ck.get("step"), "model": ck["model"]}, out)
        print(f"  {f} -> {out}  ({os.path.getsize(out) / 1e9:.2f} GB, optimizer state dropped)")

    if a.push:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(a.push, repo_type="model", private=a.private, exist_ok=True)
        api.upload_folder(folder_path=a.outdir, repo_id=a.push, repo_type="model")
        print(f"pushed {a.outdir}/ -> https://huggingface.co/{a.push}")


if __name__ == "__main__":
    main()
