# Steering directions: paired DiffMeans of per-sentence mean states, built on FLORES-dev half A.
# Every sentence is first averaged over its own non-padding positions at every block output, and a
# language's statistic is the sum of those sentence means plus the row count, so the paired mean of
# (English - X) over sentences equals the difference of the per-language means on the same rows.
# Encoder-site statistics come from each language's sentences fed as encoder input; decoder-site
# statistics teacher-force [start, tag_X] + the reference in each language against the real English
# source and average over decoder positions after the start and tag tokens.
# A DirectionSet holds the raw English, control and random directions of one target, their unit
# vectors, the English norm per layer and the mean state used by the mean-centered projection.
# Diagnostics report raw and scaled norms, cosines with English and the top-8-dimension energy share.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ...interp.scoring import encode_pass, fresh_encoder_outputs, run_batched, to_device
from ...modeling import hooks
from ...modeling.encode import PREFIX_LEN, encode_targets
from .settings import DIRECTIONS

TOP_K = 8
RAND_STREAM = 4242


@dataclass(frozen=True)
class DirectionSet:
    # One target's directions: raw {dir: [L, d]} (DiffMeans; rand is a unit vector), unit {dir: [L, d]},
    # the English DiffMean norm per layer [L] and the projection mean state mu [L, d].
    raw: dict
    unit: dict
    norm: np.ndarray
    mu: np.ndarray


def masked_mean(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Inputs: states [B, T, d] and a 0/1 mask [B, T]. Output: [B, d] mean over each row's masked-in
    # positions (zero for a row with no positions).
    m = mask.to(h.dtype)[..., None]
    return (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)


def block_means(trained, side: str, mask: torch.Tensor, forward) -> np.ndarray:
    # Inputs: Trained, side ("enc"/"dec"), the position mask of the pass and a zero-argument forward
    # function. Runs forward with every block output reduced to its per-row masked mean (edits
    # registered earlier on the same blocks are seen). Output: float32 array [B, L, d].
    reduce = lambda t: masked_mean(t.float(), mask).cpu()
    with hooks.record_blocks(trained.model, side, reduce=reduce) as rec, torch.no_grad():
        forward()
    n = len(hooks.blocks(trained.model, side))
    return torch.stack([rec.out[l] for l in range(n)], dim=1).numpy()


def encoder_means(trained, enc_batch) -> np.ndarray:
    # Inputs: Trained and an encoder batch. Output: per-row masked means of every encoder block
    # output, float32 [B, L, d].
    batch = to_device(enc_batch, trained.device)
    return block_means(trained, "enc", batch["attention_mask"], lambda: encode_pass(trained, batch))


def decoder_targets(trained, tag_lang: str, refs, max_chars: int, factor: int) -> dict:
    # Inputs: Trained, the tag language, references and caps. Output: the encode_targets batch on the
    # model device plus "content_mask", the decoder mask without the start and tag positions.
    t = to_device(encode_targets(trained.tok, refs, trained.tag(tag_lang), max_chars, factor), trained.device)
    content = t["decoder_attention_mask"].clone()
    content[:, :PREFIX_LEN] = 0
    return {**t, "content_mask": content}


def decode(trained, enc_out, enc_mask, targets: dict):
    # Inputs: Trained, encoder outputs and mask, and a target batch. Runs the teacher-forced decoder
    # stack (hooks fire on its blocks). Output: the decoder's last hidden state.
    enc = fresh_encoder_outputs(trained, enc_out).last_hidden_state
    return trained.model.decoder(input_ids=targets["decoder_input_ids"], attention_mask=targets["decoder_attention_mask"],
                                 encoder_hidden_states=enc, encoder_attention_mask=enc_mask.to(trained.device),
                                 use_cache=False).last_hidden_state


def decoder_means(trained, enc_out, enc_mask, tag_lang: str, refs, max_chars: int, factor: int) -> np.ndarray:
    # Inputs: Trained, encoder outputs and mask, tag language, references and caps. Output: per-row
    # masked means of every decoder block output over the reference positions, float32 [B, L, d].
    t = decoder_targets(trained, tag_lang, refs, max_chars, factor)
    return block_means(trained, "dec", t["content_mask"], lambda: decode(trained, enc_out, enc_mask, t))


def encoder_sums(trained, texts, rows, max_chars: int, factor: int, batching) -> np.ndarray:
    # Inputs: Trained, one language's texts, row indices, caps and Batching. Output: float64 [L, d]
    # sum over rows of the per-sentence encoder block means.
    per_row = run_batched(trained, texts, list(rows), max_chars, factor, batching,
                          lambda enc, brows: list(encoder_means(trained, enc)))
    return np.sum(np.asarray(per_row, dtype=np.float64), axis=0)


def decoder_sums(trained, src_texts, refs_by_lang: dict, tag_lang: str, rows, max_chars: int, factor: int,
                 batching) -> dict:
    # Inputs: Trained, English source texts, {lang: references}, the tag language, rows, caps and
    # Batching. One encoder pass per batch serves every language. Output: {lang: float64 [L, d] sum
    # of per-sentence decoder block means}.
    def step(enc, brows):
        # Inputs: encoder batch and its rows. Output: per-row {lang: [L, d]} means.
        out, mask, _ = encode_pass(trained, enc)
        means = {lang: decoder_means(trained, out, mask, tag_lang, [refs[r] for r in brows], max_chars, factor)
                 for lang, refs in refs_by_lang.items()}
        return [{lang: m[i] for lang, m in means.items()} for i in range(len(brows))]

    per_row = run_batched(trained, src_texts, list(rows), max_chars, factor, batching, step)
    return {lang: np.sum(np.asarray([p[lang] for p in per_row], dtype=np.float64), axis=0) for lang in refs_by_lang}


def stat_key(kind: str, group: str, lang: str) -> str:
    # Inputs: "sum" or "n", the statistic group ("enc" or a target code) and a language. Output: the
    # array name used in directions.npz.
    return f"{kind}__{group}__{lang}"


def lang_mean(arrays, group: str, lang: str) -> np.ndarray:
    # Inputs: the directions arrays, a group and a language. Output: the mean of its per-sentence
    # means, float64 [L, d].
    return np.asarray(arrays[stat_key("sum", group, lang)], dtype=np.float64) / float(arrays[stat_key("n", group, lang)])


def unit_rows(v: np.ndarray) -> np.ndarray:
    # Input: [L, d] vectors. Output: each row scaled to unit norm (zero rows stay zero).
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(norms > 0, norms, 1.0)


def random_units(n_layers: int, d: int, seed: int) -> np.ndarray:
    # Inputs: layer count, width and seed. Output: one isotropic Gaussian unit vector per layer
    # [L, d], deterministic in the seed.
    return unit_rows(np.random.default_rng([int(seed), RAND_STREAM]).standard_normal((n_layers, d)))


def direction_set(arrays, group: str, target: str, mu_lang: str, controls=DIRECTIONS[1:-1]) -> DirectionSet:
    # Inputs: the directions arrays, the statistic group, target X, the language whose mean state is
    # the projection centre and the control languages. Output: the DirectionSet with raw DiffMeans
    # L - X for English and controls, the stored random direction, unit vectors and English norms.
    x = lang_mean(arrays, group, target)
    raw = {lang: lang_mean(arrays, group, lang) - x for lang in ("en",) + tuple(controls)}
    raw["rand"] = np.asarray(arrays["rand"], dtype=np.float64)
    return DirectionSet(raw, {k: unit_rows(v) for k, v in raw.items()}, np.linalg.norm(raw["en"], axis=-1),
                        lang_mean(arrays, group, mu_lang))


def scaled(dset: DirectionSet, direction: str) -> np.ndarray:
    # Inputs: a DirectionSet and a direction. Output: [L, d] the direction rescaled to the English
    # norm at every layer (the vector actually subtracted at c = 1).
    return dset.unit[direction] * dset.norm[:, None]


def top_share(v: np.ndarray, k: int = TOP_K) -> np.ndarray:
    # Inputs: [L, d] vectors and k. Output: [L] share of the squared norm held by the k largest
    # squared coordinates.
    sq = np.asarray(v, dtype=np.float64) ** 2
    return np.sort(sq, axis=-1)[:, -k:].sum(axis=-1) / np.maximum(sq.sum(axis=-1), 1e-300)


def cosines(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Inputs: two [L, d] arrays. Output: [L] row-wise cosine similarities.
    return (unit_rows(a) * unit_rows(b)).sum(axis=-1)


def diagnostics(dset: DirectionSet, n_rows: int) -> dict:
    # Inputs: a DirectionSet and the number of half-A rows. Output: per-direction per-layer raw norms,
    # scaled norms, cosine with English and top-8 energy share, plus the row count.
    dirs = list(dset.raw)
    return {"n_rows": int(n_rows),
            "raw_norm": {d: np.linalg.norm(dset.raw[d], axis=-1).tolist() for d in dirs},
            "scaled_norm": {d: np.linalg.norm(scaled(dset, d), axis=-1).tolist() for d in dirs},
            "cos_en": {d: cosines(dset.raw[d], dset.raw["en"]).tolist() for d in dirs},
            "top8_share": {d: top_share(dset.raw[d]).tolist() for d in dirs}}
