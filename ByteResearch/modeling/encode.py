# The single source of truth for how text becomes model inputs and how outputs become text.
# Every path truncates by characters first (code points, before tokenizing) and then caps tokens
# at factor * max_chars (+1 for EOS), so the byte model is never cut harder than the subword one.
# Text is encoded with split_special_tokens=True so literal markup stays text where the tokenizer
# supports it (ByT5 does; mT5's vocabulary itself contains the sentinel pieces).
# Decoder inputs are built explicitly as [start, tag, y...] with labels [-100, y..., eos], never
# through T5's shift_right, and generation prefixes are [start, tag].
# strip_prefix_and_decode removes the two-token prefix by position, cuts at EOS and re-truncates.

from __future__ import annotations

import torch

from ..core.text import truncate_chars

IGNORE = -100
PREFIX_LEN = 2


def token_cap(max_chars, factor: int = 4):
    # Inputs: character cap and token factor. Output: the content-token cap factor*max_chars
    # (None when max_chars is None).
    return None if max_chars is None else int(factor) * int(max_chars)


def _encode_content(tok, texts, max_chars, factor) -> list:
    # Inputs: tokenizer, strings, character cap and factor. Output: per-text content ids (no EOS),
    # char-truncated first and then capped at token_cap tokens.
    if not texts:
        return []
    cut = [truncate_chars(t, max_chars) for t in texts]
    cap = token_cap(max_chars, factor)
    enc = tok(cut, add_special_tokens=False, split_special_tokens=True,
              truncation=cap is not None, max_length=cap)
    return [list(ids) for ids in enc["input_ids"]]


def source_rows(tok, texts, max_chars, factor: int = 4) -> list:
    # Inputs: tokenizer, source strings, character cap and factor. Output: unpadded encoder id lists,
    # each content (<= factor*max_chars tokens) followed by EOS.
    return [ids + [tok.eos_token_id] for ids in _encode_content(tok, texts, max_chars, factor)]


def target_rows(tok, texts, tags_per_row, max_chars, factor: int = 4) -> list:
    # Inputs: tokenizer, target strings, a tag id per row (or one int for all) and the caps.
    # Output: unpadded (decoder_input_ids, labels) pairs: [start, tag, y...] and [-100, y..., eos].
    tags = _broadcast(tags_per_row, len(texts))
    start = tok.pad_token_id
    return [([start, tag] + ids, [IGNORE] + ids + [tok.eos_token_id])
            for ids, tag in zip(_encode_content(tok, texts, max_chars, factor), tags)]


def _broadcast(values, n: int) -> list:
    # Inputs: one int or a sequence, and the row count. Output: a list of n ints.
    if isinstance(values, int):
        return [values] * n
    values = [int(v) for v in values]
    if len(values) != n:
        raise ValueError(f"expected {n} per-row values, got {len(values)}")
    return values


def pad_rows(rows, value: int) -> torch.Tensor:
    # Inputs: id lists and the pad value. Output: a right-padded LongTensor [B, max_len].
    width = max((len(r) for r in rows), default=0)
    out = torch.full((len(rows), width), int(value), dtype=torch.long)
    for i, r in enumerate(rows):
        out[i, : len(r)] = torch.as_tensor(r, dtype=torch.long)
    return out


def mask_rows(rows) -> torch.Tensor:
    # Input: id lists. Output: the matching right-padded attention mask LongTensor [B, max_len].
    return pad_rows([[1] * len(r) for r in rows], 0)


def collate_sources(rows, pad_id: int) -> dict:
    # Inputs: unpadded encoder id lists and the pad id. Output: {"input_ids", "attention_mask"}.
    return {"input_ids": pad_rows(rows, pad_id), "attention_mask": mask_rows(rows)}


def collate_targets(pairs, pad_id: int) -> dict:
    # Inputs: (decoder_input_ids, labels) pairs and the pad id. Output: {"decoder_input_ids",
    # "labels", "decoder_attention_mask"} right-padded, labels padded with -100.
    dec = [p[0] for p in pairs]
    return {"decoder_input_ids": pad_rows(dec, pad_id), "labels": pad_rows([p[1] for p in pairs], IGNORE),
            "decoder_attention_mask": mask_rows(dec)}


def encode_sources(tok, texts, max_chars, factor: int = 4) -> dict:
    # Inputs: tokenizer, source strings, character cap and factor. Output: a padded batch
    # {"input_ids", "attention_mask"} (LongTensors) ready for the encoder.
    return collate_sources(source_rows(tok, texts, max_chars, factor), tok.pad_token_id)


def encode_targets(tok, texts, tags_per_row, max_chars, factor: int = 4) -> dict:
    # Inputs: tokenizer, target strings, per-row tag ids (or one int) and the caps. Output: a padded
    # batch {"decoder_input_ids": [start, tag, y...], "labels": [-100, y..., eos],
    # "decoder_attention_mask"}; the tag position is never part of the loss.
    return collate_targets(target_rows(tok, texts, tags_per_row, max_chars, factor), tok.pad_token_id)


def decoder_prefix(start_id: int, tags_per_row) -> torch.Tensor:
    # Inputs: decoder start id and per-row tag ids (sequence or 1-D tensor). Output: LongTensor
    # [B, 2] of [start, tag] rows for generation.
    tags = torch.as_tensor(tags_per_row, dtype=torch.long).reshape(-1)
    return torch.stack([torch.full_like(tags, int(start_id)), tags], dim=1)


def strip_prefix_and_decode(tok, sequences, max_chars, prefix_len: int = PREFIX_LEN) -> list:
    # Inputs: tokenizer, generated sequences [B, T] that start with the decoder prefix, output
    # character cap and prefix length. Output: decoded strings with the prefix removed by position,
    # everything from the first EOS on dropped, pads removed, then truncated to max_chars.
    out = []
    for row in torch.as_tensor(sequences).tolist():
        body = row[prefix_len:]
        if tok.eos_token_id in body:
            body = body[: body.index(tok.eos_token_id)]
        body = [t for t in body if t != tok.pad_token_id]
        out.append(truncate_chars(tok.decode(body, skip_special_tokens=True), max_chars))
    return out
