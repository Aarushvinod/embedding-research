"""Supervised STS fine-tuning for byte students: CoSENT objective + optional whitening +
test-only Spearman eval.

CoSENT (Su, 2022) is the standard modern STS objective: it's a *rank* loss that, for every pair of
training examples (i, j) with gold score_i < score_j, pushes cos_i < cos_j. It directly optimizes the
rank ordering that Spearman measures, so it suits STS far better than a plain cosine-MSE regression.

Whitening (BERT-whitening, Su 2021) is the canonical post-hoc STS boost: it removes the anisotropy that
suppresses cosine-similarity correlation. We fit ONE transform globally on the pooled eval-sentence
embeddings (transductive, unsupervised — uses sentences, never scores), which is well-conditioned, and
apply it per language.
"""
from __future__ import annotations

import os
import random
import time

import numpy as np
import torch
from torch.optim import AdamW


def cosent_loss(cos, score, tau=0.05):
    """CoSENT rank loss. cos, score: [B]. For pairs with score_i < score_j, penalize cos_i >= cos_j.
    loss = log(1 + sum_{score_i<score_j} exp((cos_i - cos_j)/tau))."""
    cos = cos / tau
    diff = cos[:, None] - cos[None, :]                    # diff[i,j] = cos_i - cos_j
    mask = (score[:, None] < score[None, :]).float()      # 1 where score_i < score_j (we want cos_i<cos_j)
    diff = diff - (1.0 - mask) * 1e12                     # keep only the ordered pairs
    diff = torch.cat([torch.zeros(1, device=cos.device), diff.flatten()])
    return torch.logsumexp(diff, dim=0)


def train_sts(student, s1, s2, score, device="cuda", steps=2000, batch=64, lr=2e-5,
              tau=0.05, log_every=100, ckpt_path=None):
    """Fine-tune `student` so cosine(emb(s1), emb(s2)) rank-correlates with the gold score (CoSENT)."""
    opt = AdamW(student.parameters(), lr=lr)
    student.train()
    n = len(s1)
    score_t = torch.tensor(score, dtype=torch.float32)
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    amp = torch.bfloat16 if bf16 else torch.float16
    start = 1
    if ckpt_path and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        student.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start = ck["step"] + 1
        print(f"  [resume] loaded {ckpt_path} -> continuing from step {start}/{steps}")
    t0 = time.time()
    for step in range(start, steps + 1):
        idx = random.sample(range(n), min(batch, n))
        b1 = [s1[i] for i in idx]
        b2 = [s2[i] for i in idx]
        sc = score_t[idx].to(device)
        with torch.autocast(device_type="cuda", dtype=amp):
            e1 = student(b1, device=device)
            e2 = student(b2, device=device)
            cos = (e1 * e2).sum(-1)               # student outputs are already L2-normalized
            loss = cosent_loss(cos, sc, tau)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step == 1 or step % log_every == 0:
            print(f"  step {step:>5}/{steps}  loss {loss.item():.4f}  "
                  f"({(time.time() - t0) / (step - start + 1) * 1000:.0f} ms/step)")
        if ckpt_path and (step % 1000 == 0 or step == steps):
            torch.save({"step": step, "model": student.state_dict(), "opt": opt.state_dict()}, ckpt_path)


def _fit_whiten(X, eps=1e-6):
    mu = X.mean(0, keepdims=True)
    xc = X - mu
    cov = (xc.T @ xc) / len(X)
    u, s, _ = np.linalg.svd(cov)
    w = u @ np.diag(1.0 / np.sqrt(s + eps))
    return mu.astype(np.float32), w.astype(np.float32)


def _apply_whiten(X, mu, w):
    y = (X - mu) @ w
    return y / (np.linalg.norm(y, axis=1, keepdims=True) + 1e-9)


def _l2(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)


def eval_sts_testonly(student, langs=None, whiten=False, device="cuda"):
    """Per-language Spearman on the TEST split only (+ mean). whiten=True fits one global whitening
    transform on the pooled test-sentence embeddings (transductive, label-free) and applies it."""
    from scipy.stats import spearmanr

    from byte_embed.sts_data import STS_LANGS, load_sts_split

    langs = langs or STS_LANGS
    cache, pooled = {}, []
    for lang in langs:
        s1, s2, sc = load_sts_split(lang, "test")
        if len(s1) < 10:
            cache[lang] = None
            continue
        e1 = student.encode(s1, device=device)
        e2 = student.encode(s2, device=device)
        cache[lang] = (e1, e2, sc)
        if whiten:
            pooled.append(e1)
            pooled.append(e2)
    mu = w = None
    if whiten and pooled:
        mu, w = _fit_whiten(np.concatenate(pooled, 0))
    per = {}
    for lang in langs:
        if cache[lang] is None:
            per[lang] = None
            continue
        e1, e2, sc = cache[lang]
        e1, e2 = (_apply_whiten(e1, mu, w), _apply_whiten(e2, mu, w)) if whiten else (_l2(e1), _l2(e2))
        cos = (e1 * e2).sum(1)
        per[lang] = round(float(spearmanr(cos, sc).correlation), 4)
        print(f"    {lang}: spearman={per[lang]} (n={len(sc)})")
    vals = [v for v in per.values() if v is not None]
    return {"per_lang": per, "mean": round(float(np.mean(vals)), 4) if vals else None}
