"""Distillation: train the byte student to reproduce the frozen teacher's sentence
embeddings (cosine objective). No labels, no contrastive negatives -> memory-light, so it
fits a 24 GB 4090 and (tiny) even a 12 GB laptop. On an 80 GB A100 you can raise BATCH /
MAX_BYTES or switch to a contrastive objective."""
from __future__ import annotations

import random
import time

import torch
import torch.nn.functional as F
from torch.optim import AdamW


def distill(student, teacher, sentences, device="cuda", steps=2000, batch=64,
            lr=2e-4, log_every=100, objective="cosine", temp=0.05,
            augment=False, queue_size=0):
    """augment=True feeds the student orthographically-noised input while the teacher targets
    CLEAN text — distilling orthographic INVARIANCE while contrastive keeps discriminativeness
    (the robustness↔retrieval tradeoff-breaker). queue_size>0 keeps a MoCo-style FIFO of past
    (frozen-)teacher embeddings as extra contrastive negatives — more negatives => sharper
    retrieval within a 12 GB budget (the frozen teacher makes queued negatives non-stale)."""
    import collections

    opt = AdamW(student.parameters(), lr=lr)
    student.train()
    n = len(sentences)
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if bf16 else torch.float16
    aug_rng = random.Random(0)
    if augment:
        from byte_embed.robustness import random_augment
    queue = collections.deque(maxlen=max(1, queue_size // batch)) if queue_size else None
    history = []
    t0 = time.time()
    for step in range(1, steps + 1):
        texts = [sentences[i] for i in random.sample(range(n), min(batch, n))]
        with torch.no_grad():
            # .clone() converts the teacher's inference-mode tensor into a normal tensor (an
            # inference tensor cannot be saved for backward). Teacher always sees CLEAN text.
            t_emb = teacher.encode(texts, as_tensor=True, device=device).clone()
        if augment:  # student sees noised input ~half the time; the clean teacher is the target
            texts = [random_augment(x, aug_rng) if aug_rng.random() < 0.5 else x for x in texts]
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            s_emb = student(texts, device=device)
            if objective == "mse":  # MSE on the (L2-normalized) embeddings
                loss = ((s_emb - t_emb) ** 2).sum(-1).mean()
            elif objective in ("contrastive", "both"):
                # in-batch contrastive distillation: student_i must match teacher_i against all
                # other teacher embeddings -> DISCRIMINATIVE (cosine alignment alone can't retrieve).
                labels = torch.arange(s_emb.size(0), device=s_emb.device)
                if queue is not None and len(queue) > 0:
                    bank = torch.cat([t_emb, *queue], 0).to(s_emb.dtype)  # in-batch + queued negs
                    loss = F.cross_entropy((s_emb @ bank.t()) / temp, labels)
                else:
                    logits = (s_emb @ t_emb.t()) / temp
                    loss = 0.5 * (F.cross_entropy(logits, labels)
                                  + F.cross_entropy(logits.t(), labels))
                if objective == "both":                       # + absolute alignment term
                    loss = loss + (1.0 - (s_emb * t_emb).sum(-1)).mean()
            else:                   # default: cosine distance to teacher
                loss = (1.0 - (s_emb * t_emb).sum(-1)).mean()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if queue is not None:
            queue.append(t_emb.detach().float())  # frozen-teacher embs -> non-stale negatives
        if step == 1 or step % log_every == 0:
            history.append({"step": step, "loss": float(loss.item())})
            print(f"  step {step:>5}/{steps}  loss {loss.item():.4f}  "
                  f"({(time.time()-t0)/step*1000:.0f} ms/step)")
    return history
