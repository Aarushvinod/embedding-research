# Target-language tag ids and the generation suppression list.
# Each language uses the pretraining sentinel <extra_id_k> with k = 86 + its registry index.
# Its id is found by ENCODING that literal string, asserting a single token and round-tripping it.
# This matters because ByT5 and mT5 number their sentinels differently, and mT5's
# convert_tokens_to_ids returns unk in transformers 5.17.
# suppress_ids lists every sentinel id plus the embedding rows beyond the tokenizer's vocabulary,
# which generation must never emit.
# assert_tags_absent checks that encoded content holds no forbidden ids.

from __future__ import annotations

import torch

from ..core.config import LANGUAGES

MAX_SENTINELS = 1000


def sentinel_string(k: int) -> str:
    # Input: sentinel index k. Output: the literal "<extra_id_{k}>".
    return f"<extra_id_{int(k)}>"


def _single_id(tok, text: str):
    # Inputs: tokenizer and a string. Output: its id if it encodes (without specials) to exactly one
    # token that decodes back to the string and re-encodes to the same id; otherwise None.
    ids = tok(text, add_special_tokens=False).input_ids
    if len(ids) != 1 or tok.decode(ids).strip() != text:
        return None
    return ids[0] if tok(tok.decode(ids).strip(), add_special_tokens=False).input_ids == ids else None


def sentinel_id(tok, k: int) -> int:
    # Inputs: tokenizer and sentinel index. Output: the single token id of <extra_id_k>; raises
    # ValueError when it is not a single round-tripping token.
    tid = _single_id(tok, sentinel_string(k))
    if tid is None:
        raise ValueError(f"{sentinel_string(k)} is not a single round-tripping token for {type(tok).__name__}")
    return tid


def tag_ids(tok, langs) -> dict:
    # Inputs: tokenizer and language codes. Output: {code: tag token id} using each language's
    # tag_k sentinel; the ids must be distinct.
    tags = {code: sentinel_id(tok, LANGUAGES[code].tag_k) for code in langs}
    if len(set(tags.values())) != len(tags):
        raise ValueError(f"tag ids collide: {tags}")
    return tags


def sentinel_ids(tok) -> list:
    # Input: tokenizer. Output: sorted ids of every <extra_id_k> sentinel, found by encoding
    # k = 0, 1, ... until the first k that is not a single token.
    found = []
    for k in range(MAX_SENTINELS):
        tid = _single_id(tok, sentinel_string(k))
        if tid is None:
            break
        found.append(tid)
    return sorted(found)


def suppress_ids(tok, model_vocab_size: int) -> list:
    # Inputs: tokenizer and the model's embedding-row count. Output: sorted ids generation must
    # suppress: all sentinels plus rows len(tok) .. model_vocab_size-1 that no token maps to.
    unused = range(len(tok), int(model_vocab_size))
    return sorted(set(sentinel_ids(tok)) | set(unused))


def assert_tags_absent(input_ids, forbidden, mask=None) -> None:
    # Inputs: token ids (tensor or nested lists), forbidden ids (iterable) and an optional mask of
    # positions to check. Raises AssertionError naming the offending ids if any checked position
    # holds a forbidden id. No output.
    ids = torch.as_tensor(input_ids)
    bad = torch.isin(ids, torch.as_tensor(sorted(set(forbidden)), dtype=ids.dtype))
    if mask is not None:
        bad &= torch.as_tensor(mask).bool()
    if bool(bad.any()):
        raise AssertionError(f"forbidden ids present in content: {sorted(set(ids[bad].tolist()))}")
