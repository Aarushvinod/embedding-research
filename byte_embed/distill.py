"""Distillation: train the byte student to reproduce the frozen teacher's sentence
embeddings (cosine objective). No labels, no contrastive negatives -> memory-light, so it
fits a 24 GB 4090 and (tiny) even a 12 GB laptop. On an 80 GB A100 you can raise BATCH /
MAX_BYTES or switch to a contrastive objective."""
from __future__ import annotations

import os
import random
import time

import torch
import torch.nn.functional as F
from torch.optim import AdamW


def distill(student, teacher, sentences, device="cuda", steps=2000, batch=64,
            lr=2e-4, log_every=100, objective="cosine", temp=0.05,
            augment=False, queue_size=0, rel_weight=0.0, optimizer="adamw",
            patience=0, min_delta=1e-3, input_transform=None,
            ckpt_path=None, ckpt_every=5000, targets=None, seed=0):
    """augment=True feeds the student orthographically-noised input while the teacher targets
    CLEAN text — distilling orthographic INVARIANCE while contrastive keeps discriminativeness
    (the robustness↔retrieval tradeoff-breaker). queue_size>0 keeps a MoCo-style FIFO of past
    (frozen-)teacher embeddings as extra contrastive negatives — more negatives => sharper
    retrieval within a 12 GB budget (the frozen teacher makes queued negatives non-stale).

    patience: early-stop on loss plateau. The per-step contrastive loss is noisy, so we compare
    WINDOW-AVERAGED loss (window = `log_every` steps): if the window average fails to improve on the
    best-so-far by more than `min_delta` for `patience` consecutive windows, training stops (and
    checkpoints). patience=0 (default) disables it — `steps` is then exact, which keeps byte/subword
    iso-step; with patience on, `steps` is a CAP and the realized step count should be reported.
    Checkpoints carry FULL training state (model, optimizer, python/torch/cuda RNG streams, the MoCo
    queue, and the plateau tracker), so a preempted or walled job resumes the exact uninterrupted
    trajectory (up to GPU-architecture float nondeterminism).

    targets: optional precomputed teacher embeddings (np.ndarray [len(sentences), d], L2-normalized
    and index-aligned with `sentences`). When given, the live `teacher` is NOT called — both the byte
    and subword students train against the SAME cached vectors (one teacher pass, identical
    supervision). `teacher` may be None in that case."""
    import collections

    # FAIRNESS/REPRODUCIBILITY: fix the minibatch order (random.sample below) and any torch RNG so the
    # byte and subword runs see the identical data-order stream — otherwise each subprocess auto-seeds
    # from OS entropy and the byte-subword gap carries uncontrolled seed noise.
    random.seed(seed)
    torch.manual_seed(seed)

    if optimizer == "adafactor":
        # Adafactor (native T5/ByT5 optimizer) keeps FACTORED second moments -> ~50 MB of state vs
        # AdamW's ~2x-params (3.2 GB for byt5-base) — lets byt5-base train in 12 GB with headroom.
        # Native self-scheduling mode (relative_step + warmup_init, lr=None) = the original T5 setup;
        # a fixed low lr under-trains here, so we let Adafactor schedule its own step size.
        from transformers.optimization import Adafactor
        opt = Adafactor(student.parameters(), scale_parameter=True,
                        relative_step=True, warmup_init=True, lr=None)
    else:
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
    start = 1
    win, best_avg, stale = [], float("inf"), 0        # loss-plateau (patience) tracker
    if ckpt_path and os.path.exists(ckpt_path):
        # FULL-state resume: model + optimizer + python/torch/cuda RNG + MoCo queue + plateau
        # tracker. Without the RNG streams a resumed job replays batches/dropout from the seed
        # origin (breaking same-seed batch-order identity across models with different preemption
        # patterns); without the queue the first ~queue_size/batch steps train with fewer negatives.
        # With them, resume continues the EXACT uninterrupted trajectory (up to GPU-arch float
        # nondeterminism). Old-format checkpoints (model/opt/step only) still load — RNG stays at
        # the fresh seed and the resume line says so.
        ck = torch.load(ckpt_path, map_location=device)
        student.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start = ck["step"] + 1
        if "rng_py" in ck:
            random.setstate(ck["rng_py"])
        if "rng_torch" in ck:
            torch.set_rng_state(ck["rng_torch"].cpu())
        if ck.get("rng_cuda") is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all([s.cpu() for s in ck["rng_cuda"]])
            except RuntimeError:      # different GPU count on the new node -> keep the seeded state
                pass
        if queue is not None:
            for q in ck.get("queue") or []:
                queue.append(q.to(device))
        best_avg, stale = ck.get("best_avg", best_avg), ck.get("stale", stale)
        print(f"  [resume] loaded {ckpt_path} -> continuing from step {start}/{steps} "
              f"(rng {'restored' if 'rng_py' in ck else 'RESEEDED: pre-fullstate ckpt'}, "
              f"queue {len(queue) if queue is not None else 0} blocks)")
    targets_t = torch.as_tensor(targets, dtype=torch.float32) if targets is not None else None

    def _ckpt():
        # ckpt_every is a multiple of log_every, so `win` is exactly empty at every periodic save —
        # the plateau tracker (best_avg/stale) checkpoints losslessly.
        torch.save({"step": step, "model": student.state_dict(), "opt": opt.state_dict(),
                    "rng_py": random.getstate(), "rng_torch": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "queue": [q.detach().cpu() for q in queue] if queue is not None else None,
                    "best_avg": best_avg, "stale": stale}, ckpt_path)

    t0 = time.time()
    for step in range(start, steps + 1):
        idx = random.sample(range(n), min(batch, n))
        texts = [sentences[i] for i in idx]
        if targets_t is not None:                    # precomputed (cached) teacher targets
            t_emb = targets_t[idx].to(device)
        else:
            with torch.no_grad():
                # .clone() converts the teacher's inference-mode tensor into a normal tensor (an
                # inference tensor cannot be saved for backward). Teacher always sees CLEAN text.
                t_emb = teacher.encode(texts, as_tensor=True, device=device).clone()
        if augment:  # student sees noised input ~half the time; the clean teacher is the target
            texts = [random_augment(x, aug_rng) if aug_rng.random() < 0.5 else x for x in texts]
        if input_transform is not None:  # boundary-injection arms etc.; teacher targets stay CLEAN
            texts = [input_transform(x) for x in texts]
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
            if rel_weight:          # relational / similarity-preserving distillation: match the
                # student's pairwise cosine matrix to the teacher's -> preserves the graded
                # similarity GEOMETRY (targets the fine-grained STS weakness).
                loss = loss + rel_weight * ((s_emb @ s_emb.t() - t_emb @ t_emb.t()) ** 2).mean()
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if queue is not None:
            queue.append(t_emb.detach().float())  # frozen-teacher embs -> non-stale negatives
        win.append(float(loss.item()))
        if step == 1 or step % log_every == 0:
            history.append({"step": step, "loss": float(loss.item())})
            print(f"  step {step:>5}/{steps}  loss {loss.item():.4f}  "
                  f"({(time.time() - t0) / (step - start + 1) * 1000:.0f} ms/step)")
        if step % log_every == 0:                     # plateau check on the window average
            avg, win = sum(win) / len(win), []
            if avg < best_avg - min_delta:
                best_avg, stale = avg, 0
            else:
                stale += 1
            if patience and stale >= patience:
                print(f"  [early-stop] window-avg loss stuck at {best_avg:.4f} "
                      f"(no {min_delta} improvement for {patience}x{log_every} steps) "
                      f"-> stopping at step {step}/{steps}")
                if ckpt_path:
                    _ckpt()
                history.append({"step": step, "loss": avg, "early_stop": True})
                break
        if ckpt_path and (step % ckpt_every == 0 or step == steps):  # disconnect-safe checkpoint
            _ckpt()
    # make history's last entry the ACTUAL last step run — callers read steps_run from it, and when
    # steps < log_every (smoke) or steps isn't a log multiple, the last log point undershoots.
    if history and history[-1]["step"] != step:
        history.append({"step": step, "loss": float(loss.item())})
    return history
