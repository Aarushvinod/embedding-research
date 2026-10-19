# Core scoring primitives for interpretability readouts on fine-tuned T5 models.
# encode_pass runs the encoder once, optionally recording the embedding output and every block
# output in float32.
# lens_output applies the encoder's final layer norm to any block state, so the decoder can read
# an intermediate layer (DecoderLens).
# per_char_logprob teacher-forces the model's prefix ([start], or [start, tag] for a legacy tagged model)
# plus the reference against precomputed encoder outputs and returns the summed label log-probability
# excluding EOS divided by the reference's character count, with log-softmax in float32 over chunks of
# positions (mT5 has 250k logits).
# generate_from, the single generation path of the codebase, decodes from precomputed encoder outputs
# with the shared prefix, suppression and beam settings, and stops each beam once it has decoded the
# output character cap (CharBudget), with max_new_tokens as the hard cap.
# score and generate are convenience wrappers that also run the encoder in token-budget batches.

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch
from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList
from transformers.modeling_outputs import BaseModelOutput

from ..core.config import DecodingConfig, InterpConfig
from ..core.text import truncate_chars
from ..modeling import hooks
from ..modeling.encode import IGNORE, PREFIX_LEN, collate_sources, decoder_prefix, encode_targets, source_rows, \
    strip_prefix_and_decode


@dataclass(frozen=True)
class Batching:
    # Batch sizing: a token budget (rows x longest source length) and a hard row cap.
    budget_tokens: int = 32_768
    max_batch: int = 64


class CharLogprob(NamedTuple):
    # Per-row reference scores: per-char log-prob, summed log-prob, scored token count (EOS
    # excluded) and character count of the truncated reference; all numpy arrays.
    per_char: np.ndarray
    total: np.ndarray
    n_tokens: np.ndarray
    n_chars: np.ndarray


def batching_from(cfg: InterpConfig, generation: bool = False) -> Batching:
    # Inputs: InterpConfig and whether the batches are for generation. Output: the Batching to use.
    return Batching(cfg.budget_tokens, cfg.gen_max_batch if generation else cfg.max_batch)


def token_budget_batches(lengths, budget: int, max_batch: int) -> list:
    # Inputs: per-row lengths, token budget and row cap. Output: lists of row positions, longest rows
    # first, such that rows x longest length <= budget (an over-budget row gets its own batch).
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    batches, cur = [], []
    for i in order:
        width = lengths[cur[0]] if cur else lengths[i]
        if cur and ((len(cur) + 1) * width > budget or len(cur) >= max_batch):
            batches.append(cur)
            cur = []
        cur.append(i)
    return batches + ([cur] if cur else [])


def to_device(batch: dict, device) -> dict:
    # Inputs: a dict of tensors and a device. Output: a new dict with every tensor moved there.
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in dict(batch).items()}


def _model_dtype(model) -> torch.dtype:
    # Input: model. Output: its parameter dtype.
    return next(model.parameters()).dtype


def fresh_encoder_outputs(trained, encoder_outputs) -> BaseModelOutput:
    # Inputs: Trained and encoder outputs. Output: a NEW BaseModelOutput with the last hidden state
    # in the model's dtype and device (generate expands encoder outputs in place, so never reuse).
    h = encoder_outputs.last_hidden_state
    return BaseModelOutput(last_hidden_state=h.to(device=trained.device, dtype=_model_dtype(trained.model)))


def encode_pass(trained, enc_batch, record_blocks: bool = False):
    # Inputs: Trained, an encoder batch {"input_ids", "attention_mask"} and whether to record states.
    # Output: (BaseModelOutput with float32 last_hidden_state, attention mask on device,
    # {-1: embedding output, b: output of block b} in float32 or None).
    model, batch = trained.model, to_device(enc_batch, trained.device)
    args = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
    with torch.no_grad():
        if not record_blocks:
            out = model.encoder(**args)
            return BaseModelOutput(last_hidden_state=out.last_hidden_state.float()), batch["attention_mask"], None
        emb_mod = {-1: hooks.embed_prehook_module(model)}
        with hooks.record(emb_mod, pre=True) as emb, hooks.record_blocks(model, "enc") as rec:
            out = model.encoder(**args)
    states = {-1: emb.out[-1], **rec.out}
    return BaseModelOutput(last_hidden_state=out.last_hidden_state.float()), batch["attention_mask"], states


def lens_output(trained, h) -> BaseModelOutput:
    # Inputs: Trained and a block state [B, S, d]. Output: BaseModelOutput whose last_hidden_state is
    # the encoder final layer norm (and eval-mode dropout) applied to h, in float32.
    enc = trained.model.encoder
    with torch.no_grad():
        out = enc.dropout(enc.final_layer_norm(h.to(device=trained.device, dtype=_model_dtype(trained.model))))
    return BaseModelOutput(last_hidden_state=out.float())


def _gather_logprobs(logits_for, labels: torch.Tensor, chunk_rows: int) -> torch.Tensor:
    # Inputs: logits_for(flat positions) -> [n, V] logits, labels [B, T] (-100 ignored) and the chunk
    # size. Output: float32 [B, T] label log-probs (0 at ignored positions), log-softmax per chunk.
    flat = labels.reshape(-1)
    out = torch.zeros(flat.numel(), dtype=torch.float32, device=labels.device)
    keep = (flat != IGNORE).nonzero().squeeze(1)
    for s in range(0, keep.numel(), int(chunk_rows)):
        sel = keep[s:s + int(chunk_rows)]
        lp = torch.log_softmax(logits_for(sel).float(), dim=-1)
        out[sel] = lp.gather(1, flat[sel, None]).squeeze(1)
    return out.view(labels.shape)


def token_logprobs(logits: torch.Tensor, labels: torch.Tensor, chunk_rows: int = 2048) -> torch.Tensor:
    # Inputs: logits [B, T, V], labels [B, T] with -100 for ignored positions and the chunk size.
    # Output: float32 [B, T] log-probabilities of the label tokens (0 where ignored).
    flat = logits.reshape(-1, logits.shape[-1])
    return _gather_logprobs(lambda sel: flat[sel], labels.to(logits.device), chunk_rows)


def decoder_label_logprobs(trained, encoder_outputs, enc_mask, targets: dict, chunk_rows: int = 2048):
    # Inputs: Trained, encoder outputs, encoder mask and a target batch from encode_targets.
    # Output: float32 [B, T] label log-probs, computed from decoder hidden states with the LM head
    # applied per chunk of scored positions (full logits are never materialised).
    model = trained.model
    t = to_device(targets, trained.device)
    enc = fresh_encoder_outputs(trained, encoder_outputs).last_hidden_state
    with torch.no_grad():
        hid = model.decoder(input_ids=t["decoder_input_ids"], attention_mask=t["decoder_attention_mask"],
                            encoder_hidden_states=enc, encoder_attention_mask=enc_mask.to(trained.device),
                            use_cache=False).last_hidden_state
        if getattr(model.config, "scale_decoder_outputs", False):
            hid = hid * (model.model_dim ** -0.5)
        flat = hid.reshape(-1, hid.shape[-1])
        return _gather_logprobs(lambda sel: model.lm_head(flat[sel]), t["labels"], chunk_rows)


def _row_tags(trained, tgt_lang, n: int):
    # Inputs: Trained, a language code or per-row codes, and the row count. Output: None for a tagless
    # model, one int tag id for a single language, else a list of n tag ids.
    tags = trained.tags_for(tgt_lang)
    if tags is None:
        return None
    if isinstance(tgt_lang, str):
        return tags[0]
    if len(tags) != n:
        raise ValueError(f"expected {n} target languages, got {len(tags)}")
    return tags


def per_char_logprob(trained, encoder_outputs, enc_mask, tgt_lang, refs, max_chars: int = 512,
                     factor: int = 4, chunk_rows: int = 2048) -> CharLogprob:
    # Inputs: Trained, encoder outputs and mask for B rows, the target language (code or per-row
    # codes), B references, the character cap, token factor and chunk size. Output: CharLogprob with
    # sum of label log-probs excluding EOS / chars of the char-truncated reference (NaN if empty).
    tok = trained.tok
    targets = encode_targets(tok, refs, _row_tags(trained, tgt_lang, len(refs)), max_chars, factor)
    lp = decoder_label_logprobs(trained, encoder_outputs, enc_mask, targets, chunk_rows)
    labels = targets["labels"].to(lp.device)
    keep = (labels != IGNORE) & (labels != tok.eos_token_id)
    total = (lp * keep).sum(dim=1).double().cpu().numpy()
    n_tok = keep.sum(dim=1).cpu().numpy()
    n_chars = np.array([len(truncate_chars(r, max_chars)) for r in refs], dtype=np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_char = np.where(n_chars > 0, total / np.maximum(n_chars, 1), np.nan)
    return CharLogprob(per_char, total, n_tok, n_chars)


def generation_config(trained, decoding: DecodingConfig) -> GenerationConfig:
    # Inputs: Trained and DecodingConfig. Output: a fresh GenerationConfig (beams, length penalty,
    # token cap factor*max_output_chars+1, sentinel/unused-row suppression, no sampling), so saved
    # checkpoint generation settings never leak into decoding.
    tok = trained.tok
    return GenerationConfig(
        decoder_start_token_id=tok.pad_token_id, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id,
        num_beams=decoding.num_beams, length_penalty=decoding.length_penalty, do_sample=False,
        max_new_tokens=decoding.token_factor * decoding.max_output_chars + 1,
        suppress_tokens=list(trained.suppress), num_return_sequences=1, use_cache=True)


_CHAR_BOUNDS: dict = {}


def char_bounds(tok, arch: str, n_rows: int) -> torch.Tensor:
    # Inputs: tokenizer, architecture family and the model's vocabulary rows. Output: a LongTensor
    # [n_rows] giving each id an upper bound on the characters it adds to decoded text: for ByT5 1
    # per non-continuation byte (0 for continuation bytes and special ids), for mT5 the piece's
    # character length (0 for pad, EOS, unk and rows beyond the tokenizer). Cached per tokenizer.
    key = (id(tok), arch, int(n_rows))
    if key in _CHAR_BOUNDS and _CHAR_BOUNDS[key][0] is tok:
        return _CHAR_BOUNDS[key][1]
    ids = torch.arange(int(n_rows))
    if arch == "byt5":
        byte = ids - 3
        bounds = ((ids >= 3) & (ids < 259) & ((byte & 0xC0) != 0x80)).long()
    else:
        pieces = tok.convert_ids_to_tokens(list(range(min(len(tok), int(n_rows)))))
        skip = {tok.pad_token_id, tok.eos_token_id, tok.unk_token_id}
        lens = [0 if i in skip or p is None else len(p) for i, p in enumerate(pieces)]
        bounds = torch.tensor(lens + [0] * (int(n_rows) - len(lens)), dtype=torch.long)
    _CHAR_BOUNDS[key] = (tok, bounds)
    return bounds


class CharBudget(StoppingCriteria):
    # Per-beam stopping criterion: a sequence is finished once the text decoded after its decoder
    # prefix holds at least `budget` characters. A cheap per-id upper bound filters the rows, and only
    # rows whose bound reaches the budget are decoded exactly, so the stop never truncates early.

    def __init__(self, tok, bounds: torch.Tensor, budget: int, prefix_len: int = PREFIX_LEN):
        # Inputs: tokenizer, per-id character upper bounds, the character budget and prefix length.
        self.tok, self.bounds, self.budget, self.prefix_len = tok, bounds, int(budget), int(prefix_len)

    def __call__(self, input_ids, scores, **kwargs) -> torch.BoolTensor:
        # Inputs: the running sequences [N, T] (prefix included) and the scores (unused). Output: a
        # bool tensor [N], True where the decoded body already has at least `budget` characters.
        body = input_ids[:, self.prefix_len:]
        bound = self.bounds.to(body.device)[body.clamp(0, len(self.bounds) - 1)].sum(dim=1)
        hit = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
        rows = (bound >= self.budget).nonzero().squeeze(1)
        if rows.numel():
            texts = strip_prefix_and_decode(self.tok, input_ids[rows].cpu(), None, self.prefix_len)
            hit[rows] = torch.tensor([len(t) >= self.budget for t in texts], device=input_ids.device)
        return hit


def char_budget(trained, decoding: DecodingConfig) -> StoppingCriteriaList:
    # Inputs: Trained and DecodingConfig. Output: the stopping criteria that end each beam at
    # max_output_chars decoded characters after the model's decoder prefix.
    bounds = char_bounds(trained.tok, trained.arch or "mt5", trained.model.config.vocab_size)
    return StoppingCriteriaList([CharBudget(trained.tok, bounds, decoding.max_output_chars, trained.prefix_len)])


def generate_from(trained, encoder_outputs, enc_mask, tgt_lang, decoding: DecodingConfig = DecodingConfig(),
                  budget_stop: bool = True) -> list:
    # Inputs: Trained, encoder outputs and mask for B rows, target language (code or per-row codes),
    # DecodingConfig and whether beams stop at the character budget (always on outside tests).
    # Output: B decoded strings, prefix ([start] for a tagless model, the target is then unused) sliced off by
    # position, cut at EOS and truncated to max_output_chars. Any active decoder hooks fire on every step.
    n = enc_mask.shape[0]
    tags = _row_tags(trained, tgt_lang, n)
    prefix = decoder_prefix(trained.tok.pad_token_id, [tags] * n if isinstance(tags, int) else tags, n)
    stop = char_budget(trained, decoding) if budget_stop else None
    with torch.no_grad():
        seqs = trained.model.generate(encoder_outputs=fresh_encoder_outputs(trained, encoder_outputs),
                                      attention_mask=enc_mask.to(trained.device),
                                      decoder_input_ids=prefix.to(trained.device),
                                      generation_config=generation_config(trained, decoding), stopping_criteria=stop)
    return strip_prefix_and_decode(trained.tok, seqs, decoding.max_output_chars, trained.prefix_len)


def run_batched(trained, src_texts, rows, max_chars: int, factor: int, batching: Batching, step) -> list:
    # Inputs: Trained, all source texts, the row indices to process, character cap, token factor,
    # Batching and step(enc_batch, batch_rows) -> list of per-row results. Sources are encoded once,
    # grouped by token budget and passed to step. Output: per-row results aligned to rows.
    ids = source_rows(trained.tok, [src_texts[r] for r in rows], max_chars, factor)
    results = [None] * len(rows)
    for pos in token_budget_batches([len(x) for x in ids], batching.budget_tokens, batching.max_batch):
        enc = collate_sources([ids[p] for p in pos], trained.tok.pad_token_id)
        for p, value in zip(pos, step(enc, [rows[p] for p in pos])):
            results[p] = value
    return results


def stack_char_logprobs(items) -> CharLogprob:
    # Input: a list of per-row (per_char, total, n_tokens, n_chars) tuples. Output: one CharLogprob
    # of arrays in the same row order.
    cols = list(zip(*items)) if items else [(), (), (), ()]
    return CharLogprob(np.asarray(cols[0], dtype=np.float64), np.asarray(cols[1], dtype=np.float64),
                       np.asarray(cols[2], dtype=np.int64), np.asarray(cols[3], dtype=np.int64))


def _per_row(lang, rows):
    # Inputs: a language code or a full per-source list of codes, and batch rows. Output: the code,
    # or the batch's per-row codes.
    return lang if isinstance(lang, str) else [lang[r] for r in rows]


def score(trained, srcs, tgt_lang, refs, max_chars: int = 512, factor: int = 4, batching: Batching = Batching(),
          chunk_rows: int = 2048, rows=None) -> CharLogprob:
    # Inputs: Trained, source texts, target language (code or per-source codes), references aligned
    # to srcs, caps, Batching, chunk size and optional row subset. Output: CharLogprob for each row.
    rows = list(range(len(srcs))) if rows is None else list(rows)

    def step(enc, brows):
        # Inputs: encoder batch and its rows. Output: per-row CharLogprob tuples.
        out, mask, _ = encode_pass(trained, enc)
        res = per_char_logprob(trained, out, mask, _per_row(tgt_lang, brows), [refs[r] for r in brows],
                               max_chars, factor, chunk_rows)
        return list(zip(*res))
    return stack_char_logprobs(run_batched(trained, srcs, rows, max_chars, factor, batching, step))


def generate(trained, srcs, tgt_lang, decoding: DecodingConfig = DecodingConfig(),
             batching: Batching = Batching(max_batch=32), rows=None) -> list:
    # Inputs: Trained, source texts, target language (code or per-source codes), DecodingConfig,
    # Batching and optional row subset. Output: decoded hypotheses aligned to rows.
    rows = list(range(len(srcs))) if rows is None else list(rows)

    def step(enc, brows):
        # Inputs: encoder batch and its rows. Output: per-row hypotheses.
        out, mask, _ = encode_pass(trained, enc)
        return generate_from(trained, out, mask, _per_row(tgt_lang, brows), decoding)
    return run_batched(trained, srcs, rows, decoding.max_input_chars, decoding.token_factor, batching, step)
