# Batch collation for fine-tuning: raw (source, target) texts become padded model inputs.
# Formatting is delegated entirely to modeling.encode, the single source of truth shared with
# evaluation and interpretability: char-first truncation, a token cap of factor x chars + EOS and
# split_special_tokens.
# Every run trains one hub-to-target direction, so there is no target tag and no prompt: the encoder sees
# the source sentence only, decoder inputs are [start, y...] and labels [y..., eos] (standard teacher forcing).
# The labels are the only loss input, so the trainer's token-level loss covers every target token and EOS.
# The batch also carries example_idx (a plain list) so the trainer can fingerprint the exact example order
# without device syncs.
# The collator is a frozen dataclass, so it pickles cleanly into dataloader worker processes.

from __future__ import annotations

from dataclasses import dataclass

from ..modeling.encode import collate_sources, collate_targets, source_rows, target_rows


@dataclass(frozen=True)
class PairCollator:
    # Callable collator: tokenizer, character cap and token factor.
    tok: object
    max_chars: int
    factor: int = 4

    def __call__(self, features) -> dict:
        # Input: dataset items {"src", "tgt", "example_idx"}. Output: {"input_ids", "attention_mask",
        # "decoder_input_ids", "decoder_attention_mask", "labels", "example_idx"} with right padding (labels
        # padded with -100).
        src = source_rows(self.tok, [f["src"] for f in features], self.max_chars, self.factor)
        tgt = target_rows(self.tok, [f["tgt"] for f in features], None, self.max_chars, self.factor)
        pad = self.tok.pad_token_id
        return {**collate_sources(src, pad), **collate_targets(tgt, pad),
                "example_idx": [int(f["example_idx"]) for f in features]}
