# The provider/edit abstraction shared by steering, stitching and LAPE ablation.
# A provider maps an encoder batch (plus the dataset rows it holds) to encoder outputs and the
# matching attention mask: plain_provider runs the encoder unchanged, edited_provider runs it with
# hook edits on encoder blocks or act modules, and experiments may write their own (e.g. a donor
# encoder feeding a recipient's upper layers).
# Decoder edits are applied only while the decoder scores or generates, never during the encoder.
# logprob_unit runs the provider once per batch and reuses its outputs for every reference set, and
# generate_unit does the same for each requested target language.
# Results are aligned to the requested rows so units can be stored columnar.

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Callable

from ..core.config import DecodingConfig
from ..modeling import hooks
from .scoring import Batching, encode_pass, generate_from, per_char_logprob, run_batched, stack_char_logprobs

Provider = Callable


@dataclass(frozen=True)
class RefSet:
    # A reference set scored under an explicit decoder tag: tag language code and references aligned
    # to the source texts (use it when the tag language differs from the references' language).
    tag: str
    refs: list


def plain_provider(trained) -> Provider:
    # Input: Trained. Output: provider(enc_batch, rows) -> (BaseModelOutput, mask) running the
    # unmodified encoder.
    return lambda batch, rows=None: encode_pass(trained, batch)[:2]


def edited_provider(trained, enc_edits, pre: bool = False) -> Provider:
    # Inputs: Trained, encoder edits as {layer int: fn} (block outputs) or {module: fn} (e.g. act
    # modules), and whether the edits target module inputs. Output: a provider running the encoder
    # with those edits active.
    edits = hooks.resolve_edits(trained.model, "enc", enc_edits)

    def provide(batch, rows=None):
        # Inputs: encoder batch and rows. Output: (encoder outputs, mask) with the edits active.
        with hooks.edit(edits, pre=pre):
            return encode_pass(trained, batch)[:2]
    return provide


@contextlib.contextmanager
def decoder_edits(trained, dec_edits):
    # Inputs: Trained and decoder edits ({layer int: fn} for decoder blocks, {module: fn}, or None).
    # Yields with the edits active (a no-op for None or empty edits).
    with hooks.edit(hooks.resolve_edits(trained.model, "dec", dec_edits)):
        yield


def _refset(key: str, value) -> RefSet:
    # Inputs: a reference-set key and its value (a RefSet, or references scored under tag = key).
    # Output: the RefSet.
    return value if isinstance(value, RefSet) else RefSet(key, list(value))


def logprob_unit(trained, provider: Provider, dec_edits, src_texts, refs_by_tgt: dict, rows,
                 batch: Batching = Batching(), max_chars: int = 512, factor: int = 4,
                 chunk_rows: int = 2048) -> dict:
    # Inputs: Trained, a provider, decoder edits (or None), all source texts, {key: references or
    # RefSet} aligned to src_texts, the row indices, Batching, caps and chunk size. The provider runs
    # once per batch and its outputs are reused for every reference set, with dec_edits active while
    # scoring. Output: {key: CharLogprob aligned to rows}.
    sets = {k: _refset(k, v) for k, v in refs_by_tgt.items()}

    def step(enc, brows):
        # Inputs: encoder batch and its rows. Output: per-row {key: score tuple} for every reference set.
        out, mask = provider(enc, brows)
        with decoder_edits(trained, dec_edits):
            res = {k: per_char_logprob(trained, out, mask, s.tag, [s.refs[r] for r in brows], max_chars,
                                       factor, chunk_rows) for k, s in sets.items()}
        return [{k: tuple(col[i] for col in r) for k, r in res.items()} for i in range(len(brows))]

    items = run_batched(trained, src_texts, list(rows), max_chars, factor, batch, step)
    return {k: stack_char_logprobs([it[k] for it in items]) for k in sets}


def generate_unit(trained, provider: Provider, dec_edits, src_texts, tgt_langs, rows,
                  decoding: DecodingConfig = DecodingConfig(), batch: Batching = Batching(max_batch=32)) -> dict:
    # Inputs: Trained, a provider, decoder edits (or None), all source texts, the target language
    # codes to generate into, row indices, DecodingConfig and Batching. The provider runs once per
    # batch and is reused for every target; dec_edits fire on every generation step.
    # Output: {lang: hypotheses aligned to rows}.
    langs = [tgt_langs] if isinstance(tgt_langs, str) else list(tgt_langs)

    def step(enc, brows):
        # Inputs: encoder batch and its rows. Output: per-row {lang: hypothesis}.
        out, mask = provider(enc, brows)
        with decoder_edits(trained, dec_edits):
            hyps = {lang: generate_from(trained, out, mask, lang, decoding) for lang in langs}
        return [{lang: hyps[lang][i] for lang in langs} for i in range(len(brows))]

    items = run_batched(trained, src_texts, list(rows), decoding.max_input_chars, decoding.token_factor, batch, step)
    return {lang: [it[lang] for it in items] for lang in langs}
