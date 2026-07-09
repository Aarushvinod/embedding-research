"""Matched-transformer control: byte vs subword encoders that share an IDENTICAL transformer
(random-init), differing only in the input embedding + tokenizer.

The reviewer concern is "byte just wins because it has more parameters." Here we remove that
entirely: both variants are built from the **same mT5-{size} transformer config**, so the transformer
parameter count is byte-for-byte equal — only `vocab_size` (384 bytes vs 250k subwords) and the
tokenizer differ. If byte still wins, the advantage is the *representation*, not capacity. (In fact
byte then has far FEWER total params, since it carries no 128M+ vocab table.)

Both are random-initialized (no pretrained weights exist for a shared byte/subword transformer), so
this is a clean isolating ablation — fair on both axes (matched transformer, matched zero-pretraining)
— and its absolute scores are lower than the pretrained main runs by design.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from byte_embed.model import ByteStudent

_BYTE_VOCAB = 384   # ByT5 byte vocabulary (256 byte values + specials + sentinels)


class MatchedStudent(ByteStudent):
    """Reuses ByteStudent.forward/encode/pooling, but builds the encoder from a config (random-init)
    with the chosen input representation instead of `from_pretrained`."""

    def __init__(self, variant, size="small", out_dim=1024, max_chars=512,
                 grad_checkpoint=True, pooling="mean", attn_heads=8):
        nn.Module.__init__(self)                       # bypass ByteStudent.__init__ (no from_pretrained)
        from transformers import AutoConfig, AutoTokenizer, MT5EncoderModel

        cfg = AutoConfig.from_pretrained(f"google/mt5-{size}")     # the SHARED transformer dims
        if variant == "byte":
            self.tok = AutoTokenizer.from_pretrained("google/byt5-small")
            cfg.vocab_size = _BYTE_VOCAB
        elif variant == "subword":
            self.tok = AutoTokenizer.from_pretrained(f"google/mt5-{size}")   # vocab_size stays 250112
        else:
            raise ValueError(f"variant must be 'byte' or 'subword', got {variant!r}")

        self.enc = MT5EncoderModel(cfg)                # RANDOM INIT — identical transformer both ways
        if grad_checkpoint:
            self.enc.gradient_checkpointing_enable()
        d = cfg.d_model
        self.proj = nn.Linear(d, out_dim)
        self.max_chars = max_chars                     # char-budget used by the inherited ByteStudent.forward
        self.pooling = pooling
        self.variant = variant
        if pooling == "attn":
            assert d % attn_heads == 0, f"d_model {d} not divisible by attn_heads {attn_heads}"
            self.attn_heads = attn_heads
            self.attn_q = nn.Parameter(torch.randn(attn_heads, d // attn_heads) * (d // attn_heads) ** -0.5)

    def param_split(self):
        """(total, input_embedding, transformer). The transformer term must match across variants."""
        total = sum(p.numel() for p in self.parameters())
        emb = self.enc.get_input_embeddings().weight.numel()
        return total, emb, total - emb
