"""Tokenizer-free student whose input is **Unicode Character Database (UCD) property vectors**
instead of raw UTF-8 bytes.

Motivation (the new direction this notebook explores). The byte student (`model.ByteStudent`) feeds
raw UTF-8 bytes through a ByT5 encoder: an unseen byte is just an arbitrary id, and a non-Latin
character is shattered into 2-4 meaningless byte positions (the UTF-8 tax measured in `efficiency.py`).
A UCD representation instead encodes each **codepoint** by its *intrinsic, human-curated properties* —
`General_Category`, `Canonical_Combining_Class`, `Bidi_Class`, script/block, and a few binary flags —
plus a CANINE-style hashed-codepoint *identity* channel for discrimination. The payoff hypotheses:

  1. **One position per character** (not per byte) -> removes the UTF-8 multibyte tax for Indic/Ethiopic
     scripts (Tamil byte-seq 9.9x -> ~1x here), so sequence length sits *between* subword and byte.
  2. **Cross-lingual parameter sharing** -> a single "combining acute" / "Letter, other" feature is
     reused across every language, instead of disjoint subword vocab rows; the input table is tiny.
  3. **Structured generalisation** -> a codepoint never seen in training still arrives with a valid
     script/category/combining structure, so the representation degrades gracefully on rare/unseen
     characters (the property channels carry transferable signal a raw byte cannot).

Design. We reuse the **exact same ByT5 transformer body** as the byte student and only swap the front
end: a `UCDFeaturizer` maps each codepoint to a `d_model` vector (sum of small property embeddings +
a hashed-identity channel) which is fed to the encoder through `inputs_embeds`. So the transformer /
compute budget is identical to the byte student — only the *input representation* changes, which keeps
the iso-compute, parameter-allocation comparison clean.

Everything here is dependency-light: properties come from the stdlib `unicodedata` plus a small,
self-contained Unicode block-range table (no `regex`/`PyICU`/ICU needed, so it runs anywhere the byte
student runs). `UCDStudent` mirrors `ByteStudent`'s interface (`forward`, `encode`, mean/max/attn
pooling) so it drops into `distill.distill` and `eval_mteb.eval_battery` unchanged.
"""
from __future__ import annotations

import unicodedata
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- categorical property vocabularies (fixed; id 0 is reserved for PAD/unknown) -------------------

# Unicode General_Category: the 30 standard two-letter codes (Lu, Ll, ... Cn).
_CATS = ("Lu Ll Lt Lm Lo Mn Mc Me Nd Nl No Pc Pd Ps Pe Pi Pf Po "
         "Sm Sc Sk So Zs Zl Zp Cc Cf Cs Co Cn").split()
CAT2I = {c: i + 1 for i, c in enumerate(_CATS)}

# Bidi_Class (UAX #9) — the strong/weak/neutral/explicit-format classes.
_BIDI = ("L R AL EN ES ET AN CS NSM BN B S WS ON LRE LRO RLE RLO PDF "
         "LRI RLI FSI PDI").split()
BIDI2I = {b: i + 1 for i, b in enumerate(_BIDI)}

# Canonical_Combining_Class — the meaningful (mostly diacritic-position) values; others -> "other".
_CCC = (0, 1, 6, 7, 8, 9, 200, 202, 214, 216, 218, 220, 222, 224, 226, 228, 230, 232, 233, 234, 240)
CCC2I = {c: i + 1 for i, c in enumerate(_CCC)}
CCC_OTHER = len(_CCC) + 1

# Self-contained Unicode block -> script id table (covers every script in the study + common anchors;
# anything unlisted falls through to "other"). (lo, hi, name) inclusive ranges.
_BLOCKS = [
    (0x0000, 0x024F, "Latin"), (0x1E00, 0x1EFF, "Latin"),           # ASCII + Latin Extended A/B/Additional
    (0x0250, 0x02AF, "IPA"), (0x02B0, 0x02FF, "ModifierLetters"),
    (0x0300, 0x036F, "CombiningDiacritics"),
    (0x0370, 0x03FF, "Greek"), (0x1F00, 0x1FFF, "Greek"),
    (0x0400, 0x052F, "Cyrillic"),
    (0x0530, 0x058F, "Armenian"), (0x0590, 0x05FF, "Hebrew"),
    (0x0600, 0x06FF, "Arabic"), (0x0750, 0x077F, "Arabic"), (0x08A0, 0x08FF, "Arabic"),
    (0xFB50, 0xFDFF, "Arabic"), (0xFE70, 0xFEFF, "Arabic"),
    (0x0700, 0x074F, "Syriac"), (0x0780, 0x07BF, "Thaana"),
    (0x0900, 0x097F, "Devanagari"), (0x0980, 0x09FF, "Bengali"),
    (0x0A00, 0x0A7F, "Gurmukhi"), (0x0A80, 0x0AFF, "Gujarati"),
    (0x0B00, 0x0B7F, "Oriya"), (0x0B80, 0x0BFF, "Tamil"),
    (0x0C00, 0x0C7F, "Telugu"), (0x0C80, 0x0CFF, "Kannada"),
    (0x0D00, 0x0D7F, "Malayalam"), (0x0D80, 0x0DFF, "Sinhala"),
    (0x0E00, 0x0E7F, "Thai"), (0x0E80, 0x0EFF, "Lao"),
    (0x0F00, 0x0FFF, "Tibetan"), (0x1000, 0x109F, "Myanmar"),
    (0x10A0, 0x10FF, "Georgian"),
    (0x1200, 0x139F, "Ethiopic"),
    (0x1100, 0x11FF, "Hangul"), (0x3130, 0x318F, "Hangul"), (0xAC00, 0xD7AF, "Hangul"),
    (0x3040, 0x309F, "Hiragana"), (0x30A0, 0x30FF, "Katakana"),
    (0x3000, 0x303F, "CJKSymbols"),
    (0x3400, 0x4DBF, "CJK"), (0x4E00, 0x9FFF, "CJK"), (0xF900, 0xFAFF, "CJK"),
    (0x2000, 0x206F, "Punctuation"), (0x2070, 0x209F, "SuperSubscript"),
    (0x20A0, 0x20CF, "CurrencySymbols"), (0x2100, 0x214F, "LetterlikeSymbols"),
    (0x2190, 0x21FF, "Arrows"), (0x2200, 0x22FF, "MathOperators"),
    (0x2460, 0x24FF, "EnclosedAlphanumerics"), (0x2500, 0x257F, "BoxDrawing"),
    (0x1F300, 0x1FAFF, "Emoji"), (0x1F000, 0x1F0FF, "Emoji"),
]
_SCRIPT_NAMES = ["other"] + sorted({n for _, _, n in _BLOCKS})
SCRIPT2I = {n: i for i, n in enumerate(_SCRIPT_NAMES)}
N_SCRIPTS = len(_SCRIPT_NAMES)


def _script_id(cp: int) -> int:
    for lo, hi, name in _BLOCKS:
        if lo <= cp <= hi:
            return SCRIPT2I[name]
    return SCRIPT2I["other"]


@lru_cache(maxsize=1 << 18)
def _feature_ids(cp: int):
    """(category, combining-class, bidi, script, flags) ids for one codepoint — cached (the working
    set is small: a few thousand distinct codepoints across the study). `flags` is a 5-bit bitmask:
    bit0 alpha, bit1 digit, bit2 space, bit3 bidi-mirrored, bit4 has-canonical-decomposition."""
    ch = chr(cp)
    cat = CAT2I.get(unicodedata.category(ch), 0)
    ccc = CCC2I.get(unicodedata.combining(ch), CCC_OTHER)
    bidi = BIDI2I.get(unicodedata.bidirectional(ch), 0)
    script = _script_id(cp)
    flags = ((ch.isalpha() << 0) | (ch.isdigit() << 1) | (ch.isspace() << 2)
             | (unicodedata.mirrored(ch) << 3) | (bool(unicodedata.decomposition(ch)) << 4))
    return cat, ccc, bidi, script, flags


# fixed primes/salts for the CANINE-style codepoint hashing (deterministic across processes — we do
# NOT use Python's salted hash()).
_HASH_PRIMES = (2654435761, 2246822519, 3266489917, 668265263, 374761393, 1099511628211, 40503, 65537)


@lru_cache(maxsize=1 << 18)
def _hash_ids(cp: int, n_hashes: int, buckets: int):
    return tuple(((cp + 1) * _HASH_PRIMES[k] ^ (_HASH_PRIMES[k] >> 7)) % buckets
                 for k in range(n_hashes))


class UCDFeaturizer(nn.Module):
    """Codepoint -> `d_model` vector = sum of property embeddings + averaged hashed-identity embedding,
    then LayerNorm. Returns (inputs_embeds [B, L, d_model], attention_mask [B, L]). The property
    channels are the cross-lingual *prior*; the hashed-identity channel gives *discrimination* among
    codepoints that share all properties (e.g. two distinct Han ideographs)."""

    def __init__(self, d_model: int, max_chars: int = 256, hash_buckets: int = 8192, n_hashes: int = 4):
        super().__init__()
        self.max_chars = max_chars
        self.hash_buckets = hash_buckets
        self.n_hashes = n_hashes
        self.cat_emb = nn.Embedding(len(CAT2I) + 1, d_model, padding_idx=0)
        self.ccc_emb = nn.Embedding(CCC_OTHER + 1, d_model, padding_idx=0)
        self.bidi_emb = nn.Embedding(len(BIDI2I) + 1, d_model, padding_idx=0)
        self.script_emb = nn.Embedding(N_SCRIPTS, d_model)
        self.flag_emb = nn.Embedding(32, d_model)            # 5-bit flag bitmask
        self.hash_emb = nn.Embedding(hash_buckets, d_model)
        self.norm = nn.LayerNorm(d_model)

    @property
    def input_params(self) -> int:
        """Parameter count of the input representation (the UCD analogue of a subword vocab table)."""
        return sum(p.numel() for p in self.parameters())

    def _encode_batch(self, texts):
        L = self.max_chars
        cat_b, ccc_b, bidi_b, scr_b, fl_b, msk_b = [], [], [], [], [], []
        hash_b = [[] for _ in range(self.n_hashes)]
        for t in texts:
            cps = [ord(c) for c in t[:L]]
            feats = [_feature_ids(cp) for cp in cps]
            pad = L - len(cps)
            cat_b.append([f[0] for f in feats] + [0] * pad)
            ccc_b.append([f[1] for f in feats] + [0] * pad)
            bidi_b.append([f[2] for f in feats] + [0] * pad)
            scr_b.append([f[3] for f in feats] + [0] * pad)
            fl_b.append([f[4] for f in feats] + [0] * pad)
            msk_b.append([1] * len(cps) + [0] * pad)
            for k in range(self.n_hashes):
                hash_b[k].append([_hash_ids(cp, self.n_hashes, self.hash_buckets)[k]
                                  for cp in cps] + [0] * pad)
        return cat_b, ccc_b, bidi_b, scr_b, fl_b, msk_b, hash_b

    def forward(self, texts, device="cuda"):
        cat_b, ccc_b, bidi_b, scr_b, fl_b, msk_b, hash_b = self._encode_batch(texts)
        lt = lambda a: torch.tensor(np.asarray(a, dtype=np.int64), device=device)  # noqa: E731
        cat, ccc, bidi = lt(cat_b), lt(ccc_b), lt(bidi_b)
        scr, fl, mask = lt(scr_b), lt(fl_b), lt(msk_b)
        emb = (self.cat_emb(cat) + self.ccc_emb(ccc) + self.bidi_emb(bidi)
               + self.script_emb(scr) + self.flag_emb(fl))
        h = torch.stack([self.hash_emb(lt(hash_b[k])) for k in range(self.n_hashes)], 0).mean(0)
        emb = self.norm(emb + h)
        return emb * mask.unsqueeze(-1), mask


class UCDStudent(nn.Module):
    """Tokenizer-free student: UCD property featurizer -> (frozen-arch) ByT5 encoder body via
    `inputs_embeds` -> pool -> projection to teacher dim. Same transformer as `ByteStudent`; only the
    input representation differs (UCD property vectors instead of raw bytes)."""

    def __init__(self, backbone: str = "google/byt5-small", out_dim: int = 1024, max_chars: int = 256,
                 grad_checkpoint: bool = True, pooling: str = "mean", attn_heads: int = 8,
                 hash_buckets: int = 8192, n_hashes: int = 4):
        super().__init__()
        from transformers import AutoConfig

        # reuse the ByT5 encoder body (model_type "t5"); falls back to AutoModel for other encoders.
        mtype = AutoConfig.from_pretrained(backbone).model_type
        if mtype in ("t5", "longt5", "umt5", "mt5"):
            from transformers import T5EncoderModel
            self.enc = T5EncoderModel.from_pretrained(backbone)
        else:
            from transformers import AutoModel
            self.enc = AutoModel.from_pretrained(backbone)
        if grad_checkpoint:
            self.enc.gradient_checkpointing_enable()
        d = getattr(self.enc.config, "d_model", None) or self.enc.config.hidden_size
        self.feat = UCDFeaturizer(d, max_chars=max_chars, hash_buckets=hash_buckets, n_hashes=n_hashes)
        self.proj = nn.Linear(d, out_dim)
        self.max_chars = max_chars
        self.pooling = pooling
        self.tok = None                                      # no HF tokenizer (tokenizer-free)
        if pooling == "attn":
            assert d % attn_heads == 0, f"d_model {d} must be divisible by attn_heads {attn_heads}"
            self.attn_heads = attn_heads
            self.attn_q = nn.Parameter(torch.randn(attn_heads, d // attn_heads)
                                       * (d // attn_heads) ** -0.5)

    def forward(self, texts, device="cuda"):
        inputs_embeds, attn = self.feat(texts, device=device)
        h = self.enc(inputs_embeds=inputs_embeds, attention_mask=attn).last_hidden_state
        m = attn.unsqueeze(-1).float()
        if self.pooling == "max":
            pooled = h.masked_fill(m == 0, -1e9).max(dim=1).values
        elif self.pooling == "attn":
            H = self.attn_heads
            hs = h.unflatten(-1, (H, h.size(-1) // H))
            scores = (hs * self.attn_q).sum(-1) / (h.size(-1) // H) ** 0.5
            scores = scores.masked_fill(m == 0, -1e9)
            alpha = scores.softmax(dim=1).unsqueeze(-1)
            pooled = (hs * alpha).sum(1).flatten(1)
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
