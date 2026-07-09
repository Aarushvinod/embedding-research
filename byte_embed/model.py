"""Frozen subword teacher + a byte-level student (ByT5 encoder)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TeacherEncoder:
    """Frozen subword sentence embedder (e5 / bge) via sentence-transformers."""

    def __init__(self, name: str, device: str = "cuda", prefix: str = "query: "):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(name, device=device)
        self.model.eval()
        self.prefix = prefix
        # get_sentence_embedding_dimension was renamed to get_embedding_dimension; support both
        self.dim = (self.model.get_embedding_dimension()
                    if hasattr(self.model, "get_embedding_dimension")
                    else self.model.get_sentence_embedding_dimension())

    @torch.no_grad()
    def encode(self, texts, batch_size=128, as_tensor=True, device="cuda"):
        emb = self.model.encode([self.prefix + t for t in texts], batch_size=batch_size,
                                normalize_embeddings=True, convert_to_numpy=not as_tensor,
                                convert_to_tensor=as_tensor, show_progress_bar=False)
        return emb.to(device) if as_tensor else emb


MAX_CHARS = 512   # CONTENT-fair truncation budget in CHARACTERS, shared by the byte AND subword
                  # students so both see the same span of text. A tokenizer-unit cap (max_length)
                  # would truncate ByT5 at N bytes but mT5 at N tokens — on a 3-byte script that hands
                  # the subword student ~3x the content, biasing every non-Latin comparison. Tunable:
                  # byte compute scales with the byte-length of this budget (Telugu ~3 bytes/char).


class ByteStudent(nn.Module):
    """Tokenizer-free student: ByT5 byte encoder -> mean pool -> projection to teacher dim."""

    def __init__(self, backbone: str, out_dim: int, max_chars: int = MAX_CHARS,
                 grad_checkpoint: bool = True, pooling: str = "mean", attn_heads: int = 8):
        super().__init__()
        from transformers import AutoConfig, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(backbone)   # byte- or subword-level
        # architecture-aware encoder load so this serves BOTH the byte student (ByT5 = "t5")
        # AND the iso-compute subword baseline (mt5-small = "mt5"); falls back to AutoModel
        # for BERT/RoBERTa-style subword encoders.
        mtype = AutoConfig.from_pretrained(backbone).model_type
        if mtype == "mt5":
            from transformers import MT5EncoderModel
            self.enc = MT5EncoderModel.from_pretrained(backbone)
        elif mtype in ("t5", "longt5", "umt5"):
            from transformers import T5EncoderModel
            self.enc = T5EncoderModel.from_pretrained(backbone)
        else:
            from transformers import AutoModel
            self.enc = AutoModel.from_pretrained(backbone)
        if grad_checkpoint:
            self.enc.gradient_checkpointing_enable()
        d = getattr(self.enc.config, "d_model", None) or self.enc.config.hidden_size
        self.proj = nn.Linear(d, out_dim)
        self.max_chars = max_chars
        self.pooling = pooling  # "mean" | "max" | "attn"
        if pooling == "attn":
            # Lightweight MULTI-HEAD attentive pooling: one learned query per head; each head
            # attends within its own d/H subspace (no Q/K/V projections), so it adds only `d`
            # parameters total — negligible, which keeps the iso-compute param-allocation story
            # intact. (A full nn.MultiheadAttention pooler would add ~4*d^2.)
            assert d % attn_heads == 0, f"d_model {d} must be divisible by attn_heads {attn_heads}"
            self.attn_heads = attn_heads
            self.attn_q = nn.Parameter(torch.randn(attn_heads, d // attn_heads)
                                       * (d // attn_heads) ** -0.5)

    def forward(self, texts, device="cuda"):
        # CONTENT-fairness: cut by CHARACTERS first (identical text for both tokenizers), then set a
        # token ceiling generous enough (4x chars >= worst-case UTF-8 bytes) never to re-clip either.
        if self.max_chars:
            texts = [t[:self.max_chars] for t in texts]
        b = self.tok(texts, padding=True, truncation=True,
                     max_length=(self.max_chars * 4 if self.max_chars else 2048),
                     return_tensors="pt").to(device)
        h = self.enc(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        if self.pooling == "max":
            pooled = h.masked_fill(m == 0, -1e9).max(dim=1).values
        elif self.pooling == "attn":                              # lightweight multi-head attentive pool
            H = self.attn_heads
            hs = h.unflatten(-1, (H, h.size(-1) // H))            # [B, L, H, d/H]
            scores = (hs * self.attn_q).sum(-1) / (h.size(-1) // H) ** 0.5   # [B, L, H]
            scores = scores.masked_fill(m == 0, -1e9)             # mask padding (m: [B, L, 1])
            alpha = scores.softmax(dim=1).unsqueeze(-1)           # softmax over L -> [B, L, H, 1]
            pooled = (hs * alpha).sum(1).flatten(1)               # [B, d]
        else:
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        return F.normalize(self.proj(pooled), dim=-1)

    @torch.no_grad()
    def encode(self, texts, batch_size=128, device="cuda") -> np.ndarray:
        self.eval()
        out = []
        for i in range(0, len(texts), batch_size):
            out.append(self(texts[i:i + batch_size], device=device).float().cpu().numpy())
        return np.concatenate(out, axis=0)
