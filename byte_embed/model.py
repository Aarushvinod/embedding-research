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


class ByteStudent(nn.Module):
    """Tokenizer-free student: ByT5 byte encoder -> mean pool -> projection to teacher dim."""

    def __init__(self, backbone: str, out_dim: int, max_bytes: int = 256,
                 grad_checkpoint: bool = True):
        super().__init__()
        from transformers import AutoTokenizer, T5EncoderModel

        self.tok = AutoTokenizer.from_pretrained(backbone)   # byte-level: no subword vocab
        self.enc = T5EncoderModel.from_pretrained(backbone)
        if grad_checkpoint:
            self.enc.gradient_checkpointing_enable()
        self.proj = nn.Linear(self.enc.config.d_model, out_dim)
        self.max_bytes = max_bytes

    def forward(self, texts, device="cuda"):
        b = self.tok(texts, padding=True, truncation=True, max_length=self.max_bytes,
                     return_tensors="pt").to(device)
        h = self.enc(**b).last_hidden_state
        m = b["attention_mask"].unsqueeze(-1).float()
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        return F.normalize(self.proj(pooled), dim=-1)

    @torch.no_grad()
    def encode(self, texts, batch_size=128, device="cuda") -> np.ndarray:
        self.eval()
        out = []
        for i in range(0, len(texts), batch_size):
            out.append(self(texts[i:i + batch_size], device=device).float().cpu().numpy())
        return np.concatenate(out, axis=0)
