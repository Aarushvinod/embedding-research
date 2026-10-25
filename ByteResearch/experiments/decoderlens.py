# DecoderLens experiment: what would a direction model's unchanged decoder say if it read an intermediate layer?
# For the embedding output and every encoder block, the lens is the encoder's final layer norm applied to that
# layer's output, handed to the decoder as its encoder outputs with the model's tagless [start] prefix; the
# source is the hub-language FLORES devtest sentence.
# Stage logprob teacher-forces the source, English, es/zh/el and target references of the same row, saves
# every per-char log-prob as one array and reports per-layer means and English-minus-language differences.
# Stage lid decodes greedily from every lens (128 characters at most), stores each output with GlotLID's top
# label and probability, buckets it as one of the six languages, other or und (low confidence or empty), and
# reports per-layer bucket shares, the labels inside other and across-layer English statistics.
# Work is resumable per chunk of rows and layer states are offloaded to the CPU while the lens runs.

from __future__ import annotations

import functools

import numpy as np
import torch

from ..core.config import CONDITIONS, CONTROLS, LID_LABEL_OVERRIDES, MODELS, DecodingConfig, InterpConfig
from ..core.registry import composite, opt, register_stage
from ..core.results import save_run_array
from ..interp.scoring import batching_from, generate_from, lens_output, per_char_logprob, run_batched, to_device
from ..interp.units import row_chunks
from ..modeling import hooks
from .hubness import ENGLISH, SPLIT, batching_config, corpus_module, devtest_rows, done_check, encoded_langs, \
    flores_texts, layer_labels, loaded, nan_to_none, open_units, relative_depth, rows_config

EXP = "interp_decoderlens"
GRID = ("model", "condition", "target")
OTHER, UND = "other", "und"
CHUNK_ROWS = 128
OPTIONS = (opt("--chunk-rows", type=int, default=None, help="DecoderLens resume chunk size in rows (default 128)"),)


def buckets_for(langs) -> list:
    # Input: the encoded languages. Output: the LID buckets: those languages, then other and und.
    return [*langs, OTHER, UND]


def gen_token_factor(model_key: str) -> int:
    # Input: registry model key. Output: tokens allowed per output character for the 128-char lens outputs:
    # 4 for ByT5 (up to 4 bytes per character), 1 for mT5 (a subword covers at least a character).
    return 4 if MODELS[model_key].arch == "byt5" else 1


def base_config(ctx, unit) -> dict:
    # Inputs: Ctx and Unit. Output: the fingerprinted settings shared by the logprob and lid measurements.
    icfg = ctx.config(InterpConfig)
    return {"model": unit.model, "condition": unit.condition, "target": unit.target,
            "source": CONDITIONS[unit.condition].hub, "langs": encoded_langs(unit), "split": SPLIT,
            **rows_config(devtest_rows(ctx)), "max_chars": icfg.max_chars, "factor": DecodingConfig().token_factor,
            "chunk_rows": int(ctx.opt("chunk_rows", CHUNK_ROWS)), "lens": "encoder.final_layer_norm(layer)",
            "layers": "emb+blocks", "checkpoint": "best"}


def logprob_config(ctx, unit) -> dict:
    # Inputs: Ctx and Unit. Output: the fingerprinted logprob config.
    return {**base_config(ctx, unit), "measure": "per_char_logprob_excl_eos",
            "chunk_logits": ctx.config(InterpConfig).logprob_chunk_rows}


def lid_decoding(ctx, unit) -> DecodingConfig:
    # Inputs: Ctx and Unit. Output: greedy decoding capped at the LID window (128 characters).
    icfg = ctx.config(InterpConfig)
    return DecodingConfig(num_beams=1, length_penalty=1.0, max_input_chars=icfg.max_chars,
                          max_output_chars=icfg.lid_window, token_factor=gen_token_factor(unit.model))


def lid_config(ctx, unit) -> dict:
    # Inputs: Ctx and Unit. Output: the fingerprinted lid config.
    icfg, dec, cfg = ctx.config(InterpConfig), lid_decoding(ctx, unit), base_config(ctx, unit)
    return {**cfg, "decoding": "greedy", "max_output_chars": dec.max_output_chars,
            "gen_token_factor": dec.token_factor, "lid": "glotlid", "lid_rule": "top label",
            "lid_threshold": icfg.lid_threshold, "lid_label_overrides": dict(LID_LABEL_OVERRIDES),
            "buckets": buckets_for(cfg["langs"])}


def lens_states(trained, enc_batch) -> tuple:
    # Inputs: Trained and an encoder batch. Output: ([embedding output, block 0, ..., block N-1] as float32
    # CPU tensors, attention mask on the model device); offloading keeps one layer's lens on the accelerator.
    batch = to_device(enc_batch, trained.device)
    offload = lambda h: h.detach().float().cpu()
    emb_mod = {-1: hooks.embed_prehook_module(trained.model)}
    with torch.no_grad(), hooks.record(emb_mod, reduce=offload, pre=True) as emb, \
            hooks.record_blocks(trained.model, "enc", reduce=offload) as rec:
        trained.model.encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    return [emb.out[-1]] + [rec.out[b] for b in range(trained.n_enc)], batch["attention_mask"]


def lens_step(trained, read):
    # Inputs: Trained and read(lens, mask, batch_rows) -> one value per batch row. Output: a run_batched step
    # returning, for every row, the list of its values over the lens layers (embedding output first).
    def step(enc, brows):
        # Inputs: encoder batch and its rows. Output: per-row lists of per-layer values.
        states, mask = lens_states(trained, enc)
        layers = [read(lens_output(trained, h), mask, brows) for h in states]
        return [[layer[i] for layer in layers] for i in range(len(brows))]
    return step


def score_langs(trained, refs: dict, target: str, icfg: InterpConfig, factor: int):
    # Inputs: Trained, {lang: all row-aligned references}, the target, InterpConfig and token factor. Output:
    # read(lens, mask, rows) -> per-row lists of per-char log-probs in the order of refs.
    def read(lens, mask, brows):
        # Inputs: lens encoder outputs, mask and batch rows. Output: per-row per-language per-char log-probs.
        res = [per_char_logprob(trained, lens, mask, target, [texts[r] for r in brows], icfg.max_chars, factor,
                                icfg.logprob_chunk_rows).per_char for texts in refs.values()]
        return [[float(v[i]) for v in res] for i in range(len(brows))]
    return read


def logprob_chunk(ctx, trained, texts: dict, cfg: dict, crows) -> dict:
    # Inputs: Ctx, Trained, {lang: devtest sentences}, the logprob config and a chunk of row ids. Output:
    # {"lp": [layer][lang][row] per-char log-probs} for the chunk.
    icfg = ctx.config(InterpConfig)
    refs = {lang: texts[lang] for lang in cfg["langs"]}
    step = lens_step(trained, score_langs(trained, refs, cfg["target"], icfg, cfg["factor"]))
    out = run_batched(trained, texts[cfg["source"]], crows, cfg["max_chars"], cfg["factor"], batching_from(icfg), step)
    n_layers, n_langs = len(out[0]), len(cfg["langs"])
    return {"lp": [[[row[li][k] for row in out] for k in range(n_langs)] for li in range(n_layers)]}


def lid_buckets(lid_model, texts, langs, threshold: float) -> tuple:
    # Inputs: a GlotLID model, generated strings, the encoded languages and the top-label probability
    # threshold. Output: (top labels, top probs, buckets) per string; empty strings are not classified and,
    # like low-confidence ones, become und.
    lid = corpus_module("lid")
    clean = [" ".join(str(t).split()) for t in texts]
    keep = [i for i, t in enumerate(clean) if t]
    preds = [None] * len(texts)
    for i, pred in zip(keep, lid.predict(lid_model, [clean[i] for i in keep], k=1) if keep else []):
        preds[i] = pred
    labels = {lang: lid.lid_label(lang) for lang in langs}
    tops = [lid.top_label(p) if p is not None else (None, 0.0) for p in preds]
    buckets = [bucket_of(top, prob, labels, threshold) for top, prob in tops]
    return [t[0] for t in tops], [float(t[1]) for t in tops], buckets


def bucket_of(top, prob: float, labels: dict, threshold: float) -> str:
    # Inputs: one string's top GlotLID label and probability (None for empty text), {lang: label} and the
    # threshold. Output: und when the top probability is below the threshold (or the text is empty), else the
    # language whose label is on top, else other.
    if top is None or prob < threshold:
        return UND
    return next((lang for lang, label in labels.items() if label == top), OTHER)


def lid_chunk(lid_model, ctx, trained, texts: dict, cfg: dict, crows, decoding: DecodingConfig) -> dict:
    # Inputs: the GlotLID model, Ctx, Trained, {lang: devtest sentences}, the lid config, a chunk of row ids
    # and the lens decoding. Output: {"text", "label", "prob", "bucket"}, each [layer][row].
    icfg = ctx.config(InterpConfig)
    read = lambda lens, mask, brows: generate_from(trained, lens, mask, cfg["target"], decoding)
    out = run_batched(trained, texts[cfg["source"]], crows, cfg["max_chars"], cfg["factor"],
                      batching_from(icfg, True), lens_step(trained, read))
    n_layers = len(out[0])
    gens = [row[li] for li in range(n_layers) for row in out]
    labels, probs, buckets = lid_buckets(lid_model, gens, cfg["langs"], cfg["lid_threshold"])
    shape = lambda flat: [flat[li * len(crows):(li + 1) * len(crows)] for li in range(n_layers)]
    return {"text": shape(gens), "label": shape(labels), "prob": shape(probs), "bucket": shape(buckets)}


def _run_missing(ctx, unit, uf, cfg: dict, make_work) -> None:
    # Inputs: Ctx, Unit, the open UnitFile, the config and make_work(ctx) -> work(ctx, trained, texts, cfg,
    # rows). Loads FLORES, any helper model and the fine-tuned model only when a chunk is missing, then runs
    # every missing chunk.
    chunks = row_chunks(devtest_rows(ctx), cfg["chunk_rows"])
    todo = [(i, c) for i, c in enumerate(chunks) if not uf.has(i)]
    if not todo:
        return
    texts = flores_texts(ctx, cfg["langs"])
    work = make_work(ctx)
    with loaded(ctx, unit) as trained:
        for i, crows in todo:
            uf.put(i, work(ctx, trained, texts, cfg, crows))


def assemble(uf, cfg: dict, fields) -> dict:
    # Inputs: a UnitFile with every chunk done, the config and field names. Output: {field: object array with
    # rows last} with chunks concatenated in row order.
    n_chunks = len(row_chunks(list(range(cfg["n_rows"])), cfg["chunk_rows"]))
    parts = [uf.get(i) for i in range(n_chunks)]
    return {f: np.concatenate([np.asarray(p[f], dtype=object) for p in parts], axis=-1) for f in fields}


def logprob_aggregate(lp: np.ndarray, langs, labels) -> dict:
    # Inputs: per-char log-probs [L, n_langs, n], the languages and layer labels. Output: per-layer mean
    # log-prob per language, per-layer English-minus-language and English-minus-control-mean differences, and
    # those differences averaged over all layers including the embedding output.
    mean = np.nanmean(lp, axis=-1)
    by_lang = {lang: mean[:, k] for k, lang in enumerate(langs)}
    en_minus = {lang: by_lang[ENGLISH] - v for lang, v in by_lang.items() if lang != ENGLISH}
    controls = by_lang[ENGLISH] - np.mean([by_lang[c] for c in CONTROLS], axis=0)
    return {"langs": list(langs), "layers": list(labels), "depth": relative_depth(labels),
            "mean_logprob": {lang: nan_to_none(v) for lang, v in by_lang.items()},
            "en_minus": {lang: nan_to_none(v) for lang, v in en_minus.items()},
            "en_minus_controls": nan_to_none(controls),
            "mean_over_layers": {"en_minus": {lang: float(np.mean(v)) for lang, v in en_minus.items()},
                                 "en_minus_controls": float(np.mean(controls))}}


@register_stage("interp.decoderlens.logprob", grid=GRID, is_done=done_check(EXP, "logprob", logprob_config),
                requires=("train",), options=OPTIONS, order=520)
def logprob(ctx, unit):
    # Inputs: Ctx and a model x condition x target Unit. Saves every lens per-char log-prob of the six
    # languages' references as one array and writes their per-layer means and English differences.
    if not ctx.force and done_check(EXP, "logprob", logprob_config)(ctx, unit):
        return
    cfg = logprob_config(ctx, unit)
    with open_units(ctx, unit, EXP, "logprob", cfg, batching_config(ctx)) as uf:
        _run_missing(ctx, unit, uf, cfg, lambda c: logprob_chunk)
        lp = assemble(uf, cfg, ("lp",))["lp"].astype(np.float64)
        labels = layer_labels(lp.shape[0] - 1)
        ref = save_run_array(ctx, EXP, unit.run_id, "logprob.npy", lp.astype(np.float32))
        per_item = {"row_ids": devtest_rows(ctx), "langs": cfg["langs"], "layers": labels,
                    "array": {"path": ref, "shape": list(lp.shape), "dtype": "float32", "axes": ["layer", "lang", "row"]}}
        uf.finalize({"source": cfg["source"], "target": cfg["target"], **logprob_aggregate(lp, cfg["langs"], labels)},
                    per_item=per_item)


def across_layers(curve, rivals) -> dict:
    # Inputs: a per-layer English curve [L] and rival curves [k, L]. Output: mean, median and max over layers,
    # the layer of the max, the fraction of layers where English is the majority (> 0.5) and where it is the
    # plurality (strictly above every rival).
    curve = np.asarray(curve, dtype=np.float64)
    rivals = np.asarray(rivals, dtype=np.float64).reshape(-1, len(curve))
    plural = curve > (rivals.max(0) if len(rivals) else -np.inf)
    return {"mean": float(np.nanmean(curve)), "median": float(np.nanmedian(curve)), "max": float(np.nanmax(curve)),
            "argmax_layer": int(np.nanargmax(curve)), "frac_majority": float(np.mean(curve > 0.5)),
            "frac_plurality": float(np.mean(plural))}


def bucket_fractions(buckets, names) -> dict:
    # Inputs: buckets [L, n] and the bucket names. Output: {bucket: per-layer fraction of all outputs [L]}.
    b = np.asarray(buckets, dtype=object)
    return {k: (b == k).mean(axis=1) for k in names}


def identified_shares(frac: dict) -> dict:
    # Input: per-layer bucket fractions. Output: {bucket: per-layer share among identified outputs} for every
    # bucket except und (NaN for a layer with no identified output).
    identified = 1.0 - frac[UND]
    with np.errstate(invalid="ignore", divide="ignore"):
        return {k: np.where(identified > 0, v / identified, np.nan) for k, v in frac.items() if k != UND}


def other_labels(buckets, labels) -> list:
    # Inputs: buckets [L, n] and top labels [L, n]. Output: per layer, {label: count} of the outputs in other,
    # most frequent first.
    out = []
    for brow, lrow in zip(np.asarray(buckets, dtype=object), np.asarray(labels, dtype=object)):
        found, counts = np.unique([str(lab) for lab in lrow[brow == OTHER]], return_counts=True)
        out.append({str(k): int(c) for k, c in sorted(zip(found, counts), key=lambda kc: (-kc[1], kc[0]))})
    return out


def lid_summary(buckets, labels, names) -> dict:
    # Inputs: buckets [L, n], top labels [L, n] and the bucket names. Output: per-layer bucket fractions of all
    # outputs and shares among identified (non-und) outputs, the labels inside other, and the across-layer
    # statistics of both English measures (rivals are the other buckets).
    frac = bucket_fractions(buckets, names)
    share = identified_shares(frac)
    return {"per_layer": {k: v.tolist() for k, v in frac.items()},
            "per_layer_identified": {k: nan_to_none(v) for k, v in share.items()},
            "other_labels": other_labels(buckets, labels),
            "en_across_layers": across_layers(frac[ENGLISH], [v for k, v in frac.items() if k != ENGLISH]),
            "en_identified_across_layers": across_layers(np.nan_to_num(share[ENGLISH]),
                                                         [np.nan_to_num(v) for k, v in share.items() if k != ENGLISH])}


def lid_worker(ctx, unit):
    # Inputs: Ctx and Unit. Output: make_work(ctx) for _run_missing that loads GlotLID once and returns the
    # lid chunk function bound to it and to the lens decoding.
    decoding = lid_decoding(ctx, unit)
    return lambda c: functools.partial(_lid_work, corpus_module("lid").load_glotlid(c), decoding)


def _lid_work(lid_model, decoding, ctx, trained, texts, cfg, crows) -> dict:
    # Inputs: the GlotLID model, lens decoding and the work arguments of _run_missing. Output: lid_chunk's
    # record for one chunk.
    return lid_chunk(lid_model, ctx, trained, texts, cfg, crows, decoding)


@register_stage("interp.decoderlens.lid", grid=GRID, is_done=done_check(EXP, "lid", lid_config),
                requires=("train",), options=OPTIONS, order=521)
def lid(ctx, unit):
    # Inputs: Ctx and a model x condition x target Unit. Writes greedy lens outputs per layer with their GlotLID
    # top labels, probabilities and buckets (per_item) and per-layer bucket shares, other labels and
    # across-layer English statistics (aggregate).
    if not ctx.force and done_check(EXP, "lid", lid_config)(ctx, unit):
        return
    cfg = lid_config(ctx, unit)
    with open_units(ctx, unit, EXP, "lid", cfg, batching_config(ctx)) as uf:
        _run_missing(ctx, unit, uf, cfg, lid_worker(ctx, unit))
        t = assemble(uf, cfg, ("text", "label", "prob", "bucket"))
        labels = layer_labels(t["bucket"].shape[0] - 1)
        agg = {"source": cfg["source"], "target": cfg["target"], "buckets": cfg["buckets"], "layers": labels,
               "depth": relative_depth(labels), **lid_summary(t["bucket"], t["label"], cfg["buckets"])}
        per_item = {"row_ids": devtest_rows(ctx), "layers": labels, "text": t["text"].tolist(),
                    "label": t["label"].tolist(), "prob": nan_to_none(t["prob"].astype(np.float64), 6),
                    "bucket": t["bucket"].tolist()}
        uf.finalize(agg, per_item=per_item)


composite("interp.decoderlens", ("interp.decoderlens.logprob", "interp.decoderlens.lid"))
