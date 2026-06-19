"""Sanity-check the environment for both experiments.

GATE-GRAFT needs only the `requirements-local.txt` stack (no GPU).
ByteEmbed needs torch + transformers and a Blackwell-compatible (cu128) build if run on a
local RTX 50-series GPU.
"""
from __future__ import annotations

import importlib
import platform

print(f"python {platform.python_version()}")

# --- GATE-GRAFT (local) deps ---
print("\n[GATE-GRAFT local stack]")
for m in ["numpy", "scipy", "sklearn", "ot", "requests", "tqdm"]:
    try:
        importlib.import_module(m)
        print(f"  {m:10} OK")
    except Exception as e:  # noqa: BLE001
        print(f"  {m:10} MISSING ({e})")

# --- ByteEmbed (GPU) deps ---
print("\n[ByteEmbed GPU stack]")
try:
    import torch

    print(f"  torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        archs = torch.cuda.get_arch_list()
        print(f"  device: {torch.cuda.get_device_name(0)}  capability sm_{cap[0]}{cap[1]}")
        print(f"  torch arch list: {archs}")
        if cap[0] >= 12 and not any("120" in a for a in archs):
            print("  WARNING: Blackwell GPU (sm_120) but torch lacks sm_120 — GPU ops may "
                  "fail.\n           Install a cu128 build:  pip install torch "
                  "--index-url https://download.pytorch.org/whl/cu128")
        else:
            x = torch.randn(2048, 2048, device="cuda")
            _ = (x @ x).sum().item()
            print("  cuda matmul OK")
    for m in ["transformers", "sentence_transformers", "datasets"]:
        try:
            importlib.import_module(m)
            print(f"  {m:22} OK")
        except Exception as e:  # noqa: BLE001
            print(f"  {m:22} MISSING ({e})")
except ImportError:
    print("  torch not installed — fine for GATE-GRAFT; install requirements-cloud.txt "
          "for ByteEmbed.")
