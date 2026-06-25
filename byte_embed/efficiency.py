"""Tokenization-efficiency / equity battery — the MOTIVATION for byte-level representations.

Honest framing (byte is NOT cheaper on every axis):
  - **sequence length / compute**: byte LOSES (byte sequences are several× longer) — reported, not hidden.
  - **cross-lingual equity ("tokenization tax")**: byte WINS — a subword tokenizer charges low-resource /
    non-Latin languages many more tokens for the SAME content; the byte (UTF-8) cost is far flatter.
  - **parameter efficiency** (quality / parameter): byte WINS — measured by the scaling curve, not here.

`fertility_table` runs on FLORES-1012 (perfectly parallel, so "tax" = tokens(lang)/tokens(English) is a
fair same-content comparison). No training, no GPU. `profile_model` measures the compute COST (FLOPs /
throughput / latency / VRAM) of a trained student so the efficiency story stays honest.
"""
from __future__ import annotations

import numpy as np

from byte_embed.config import FLORES_CODE

# the three tokenizers that matter: the subword student, the byte student, and the SONAR teacher (NLLB).
MT5_TOK = "google/mt5-small"
BYT5_TOK = "google/byt5-small"
NLLB_TOK = "facebook/nllb-200-distilled-600M"   # SONAR's tokenizer (SentencePiece, 256k)


def _flores_rows():
    from datasets import load_dataset
    return load_dataset("mteb/flores", "default", split="devtest")


def fertility_table(langs, pivot="en"):
    """Per-language tokenization stats on FLORES-1012 (parallel). Returns {lang: {...}} with:
      chars, bytes_per_char, mt5_tok_per_char, nllb_tok_per_char, byte_seq_ratio (byt5/mt5 length),
      subword_tax (mt5 tokens(lang)/tokens(pivot)), byte_tax (bytes(lang)/bytes(pivot)).
    The two tax columns are the headline: subword_tax climbs steeply for low-resource/non-Latin
    languages; byte_tax stays near 1–1.5 → byte removes the cross-lingual tokenization penalty."""
    from transformers import AutoTokenizer

    mt5 = AutoTokenizer.from_pretrained(MT5_TOK)
    byt5 = AutoTokenizer.from_pretrained(BYT5_TOK)
    try:
        nllb = AutoTokenizer.from_pretrained(NLLB_TOK)
    except Exception:  # noqa: BLE001 — NLLB tokenizer optional
        nllb = None
    d = _flores_rows()

    def totals(texts):
        chars = sum(len(t) for t in texts)
        nbytes = sum(len(t.encode("utf-8")) for t in texts)
        mt5_n = sum(len(ids) for ids in mt5(texts, add_special_tokens=False)["input_ids"])
        byt5_n = sum(len(ids) for ids in byt5(texts, add_special_tokens=False)["input_ids"])
        nllb_n = (sum(len(ids) for ids in nllb(texts, add_special_tokens=False)["input_ids"])
                  if nllb is not None else None)
        return chars, nbytes, mt5_n, byt5_n, nllb_n

    base = totals(list(d[FLORES_CODE[pivot]]))   # pivot (English) totals for the tax denominators
    out = {}
    for lang in langs:
        col = FLORES_CODE.get(lang)
        if col is None or col not in d.column_names:
            out[lang] = None
            continue
        ch, by, m, bt, nl = totals(list(d[col]))
        out[lang] = {
            "chars": ch,
            "bytes_per_char": round(by / ch, 3),
            "mt5_tok_per_char": round(m / ch, 3),
            "nllb_tok_per_char": round(nl / ch, 3) if nl else None,
            "byte_seq_ratio": round(bt / m, 2),                       # how much longer byte seqs are
            "subword_tax": round(m / base[2], 3),                     # vs English, same content
            "byte_tax": round(by / base[1], 3),
        }
    return out


def print_fertility(table, pivot="en"):
    print(f"\n{'lang':6}{'b/char':>8}{'mt5/char':>9}{'nllb/char':>10}{'byte/sub':>9}"
          f"{'SUBWORD tax':>12}{'BYTE tax':>10}")
    for lang, r in table.items():
        if r is None:
            continue
        print(f"{lang:6}{r['bytes_per_char']:>8}{r['mt5_tok_per_char']:>9}"
              f"{str(r['nllb_tok_per_char']):>10}{r['byte_seq_ratio']:>9}"
              f"{r['subword_tax']:>12}{r['byte_tax']:>10}")
    print(f"  (tax = tokens(lang)/tokens({pivot}) on the SAME 1012 parallel sentences; "
          f"subword tax >> byte tax for low-resource = the tokenization penalty byte removes)")


def profile_model(model, tokenizer, sample_texts, device="cuda", max_bytes=256, repeats=3):
    """Honest compute cost of a trained student: latency (ms/sentence), throughput (sent/s), peak VRAM,
    and FLOPs/sentence (via fvcore if installed, else None). `model` is the ByteStudent; we time its
    encode path. Run on the A100 in the notebook."""
    import time

    import torch

    res = {"params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1)}
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    model.eval()
    with torch.no_grad():
        for _ in range(2):                       # warmup
            model(sample_texts[:8], device=device)
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(repeats):
            model(sample_texts, device=device)
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()
        dt = (time.time() - t0) / repeats
    res["latency_ms_per_sent"] = round(1000 * dt / len(sample_texts), 3)
    res["throughput_sent_per_s"] = round(len(sample_texts) / dt, 1)
    if device == "cuda" and torch.cuda.is_available():
        res["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    # optional FLOPs/sentence
    try:
        from fvcore.nn import FlopCountAnalysis
        b = tokenizer(sample_texts[:1], return_tensors="pt", truncation=True,
                      max_length=max_bytes, padding=True).to(device)
        flops = FlopCountAnalysis(model.enc, (b["input_ids"], b["attention_mask"])).total()
        res["gflops_per_sent"] = round(flops / 1e9, 2)
    except Exception:  # noqa: BLE001 — fvcore optional; sequence length is the proxy otherwise
        res["gflops_per_sent"] = None
    return res
