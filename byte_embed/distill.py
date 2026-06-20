"""Distillation: train the byte student to reproduce the frozen teacher's sentence
embeddings (cosine objective). No labels, no contrastive negatives -> memory-light, so it
fits a 24 GB 4090 and (tiny) even a 12 GB laptop. On an 80 GB A100 you can raise BATCH /
MAX_BYTES or switch to a contrastive objective."""
from __future__ import annotations

import random
import time

import torch
from torch.optim import AdamW


def distill(student, teacher, sentences, device="cuda", steps=2000, batch=64,
            lr=2e-4, log_every=100, objective="cosine"):
    opt = AdamW(student.parameters(), lr=lr)
    student.train()
    n = len(sentences)
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16 else torch.float16
    history = []
    t0 = time.time()
    for step in range(1, steps + 1):
        texts = [sentences[i] for i in random.sample(range(n), min(batch, n))]
        with torch.no_grad():
            # .clone() converts the teacher's inference-mode tensor (sentence-transformers
            # encodes under inference_mode) into a normal tensor; an inference tensor cannot
            # be saved for backward when multiplied with the student's grad-requiring output.
            t_emb = teacher.encode(texts, as_tensor=True, device=device).clone()
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            s_emb = student(texts, device=device)
            if objective == "mse":  # MSE on the (L2-normalized) embeddings
                loss = ((s_emb - t_emb) ** 2).sum(-1).mean()
            else:                   # default: cosine distance to teacher
                loss = (1.0 - (s_emb * t_emb).sum(-1)).mean()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step == 1 or step % log_every == 0:
            history.append({"step": step, "loss": float(loss.item())})
            print(f"  step {step:>5}/{steps}  loss {loss.item():.4f}  "
                  f"({(time.time()-t0)/step*1000:.0f} ms/step)")
    return history
