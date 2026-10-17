# Batch collation for fine-tuning: raw (source, target) texts become padded model inputs.
# Formatting is delegated entirely to modeling.encode, the single source of truth shared with
# evaluation and interpretability: char-first truncation, a token cap of factor x chars + EOS,
# split_special_tokens, decoder inputs [start, tag_X, y...] and labels [-100, y..., eos].
# Each row's tag comes from its target index, so one batch can mix all targets of a condition.
# The batch also carries target_idx (a LongTensor) for per-target loss tracking and example_idx
# (a plain list) so the trainer can fingerprint the exact example order without device syncs.
# The collator is a frozen dataclass, so it pickles cleanly into dataloader worker processes.

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..modeling.encode import collate_sources, collate_targets, source_rows, target_rows


@dataclass(frozen=True)
class PairCollator:
    # Callable collator: tokenizer, tag id per target index, character cap and token factor.
    tok: object
    tag_by_idx: tuple
    max_chars: int
    factor: int = 4

    def __call__(self, features) -> dict:
        # Input: dataset items {"src", "tgt", "target_idx", "example_idx"}. Output: {"input_ids",
        # "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels", "target_idx",
        # "example_idx"} with right padding (labels padded with -100).
        idx = [int(f["target_idx"]) for f in features]
        src = source_rows(self.tok, [f["src"] for f in features], self.max_chars, self.factor)
        tgt = target_rows(self.tok, [f["tgt"] for f in features], [self.tag_by_idx[i] for i in idx],
                          self.max_chars, self.factor)
        pad = self.tok.pad_token_id
        return {**collate_sources(src, pad), **collate_targets(tgt, pad),
                "target_idx": torch.tensor(idx, dtype=torch.long),
                "example_idx": [int(f["example_idx"]) for f in features]}
