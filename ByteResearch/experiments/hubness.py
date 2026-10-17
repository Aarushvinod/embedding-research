# Hubness experiment: which language do the encoder's sentence representations gravitate to, layer by layer?
# Stage states pools each FLORES devtest sentence of the eleven study languages over its tokens (masked mean
# with EOS) at the embedding output and at every encoder block output, inside forward hooks, in float32.
# Stage similarity z-scores every dimension over all sentences of all languages at each layer, so massive
# activation dimensions cannot dominate, and scores a language pair by the mean paired cosine of the two
# translations of the same sentence.
# Stage ranks orders, for every language, the other ten by that similarity and averages each language's rank
# in the others' lists over the block layers, giving a hub ordering and nearest-neighbour fractions.
# The module also holds the helpers shared with the DecoderLens experiment: FLORES access, checkpoint
# provenance and resumable unit files whose pilot paths come from core.results.locate.

from __future__ import annotations

import functools
import importlib

import numpy as np
import torch
from scipy.stats import rankdata

from ..core.config import PINS, STUDY11, DecodingConfig, InterpConfig, config_hash, resolve_langs
from ..core.registry import composite, register_stage
from ..core.results import ResultFile, load_array, locate, provenance, run_result, save_run_array
from ..core.text import truncate_chars
from ..interp.scoring import batching_from, run_batched, to_device
from ..interp.units import UnitFile, fingerprint, measurement_done
from ..modeling import hooks
from ..modeling.loading import checkpoint_info as describe_checkpoint
from ..modeling.loading import loaded as load_model

EXP = "interp_hubness"
SPLIT = "devtest"
EMB = "emb"
GRID = ("model", "condition")


def n_flores_rows() -> int:
    # Output: the pinned number of FLORES devtest rows (1012).
    return int(PINS["flores"]["rows"][SPLIT])


def devtest_rows(ctx) -> list:
    # Input: Ctx. Output: the devtest row ids to use: all rows, or the fixed seeded pilot subset.
    return ctx.rows(n_flores_rows())


def corpus_module(name: str):
    # Input: a corpora submodule name ("flores" or "lid"). Output: the module, imported lazily through
    # sys.modules so heavy dependencies load only when needed and tests can substitute fakes.
    return importlib.import_module(f"..corpora.{name}", __package__)


def flores_texts(ctx, langs) -> dict:
    # Inputs: Ctx and language codes. Output: {code: row-aligned FLORES devtest sentences}; raises when a
    # language does not have exactly the pinned number of rows.
    data = corpus_module("flores").load_flores(ctx, SPLIT, list(langs))
    sizes = {lang: len(data.get(lang, ())) for lang in langs}
    if any(n != n_flores_rows() for n in sizes.values()):
        raise ValueError(f"FLORES {SPLIT} row counts {sizes} != {n_flores_rows()}")
    return {lang: list(data[lang]) for lang in langs}


def study_langs(ctx) -> list:
    # Input: Ctx. Output: the study languages (8 targets + en, ar, ru; smoke keeps the smoke targets).
    return resolve_langs("study11", STUDY11, ctx.smoke)


def rows_config(rows) -> dict:
    # Input: row ids. Output: their count and hash, for config fingerprints.
    return {"n_rows": len(rows), "rows_hash": config_hash(list(rows))}


def batching_config(ctx) -> dict:
    # Input: Ctx. Output: the batching settings, stored in configs but kept out of fingerprints because they
    # do not change results beyond float noise.
    icfg = ctx.config(InterpConfig)
    return {"budget_tokens": icfg.budget_tokens, "max_batch": icfg.max_batch, "gen_max_batch": icfg.gen_max_batch}


def checkpoint_info(ctx, unit) -> dict:
    # Inputs: Ctx and a model x condition Unit. Output: the checkpoint path and its meta step/lr (read from
    # byteresearch_meta.json when present) for provenance.
    info = describe_checkpoint(ctx, unit.model, unit.condition)
    return {k: info[k] for k in ("path", "which", "step", "lr")}


def open_units(ctx, unit, experiment: str, measurement: str, cfg: dict, extra_cfg=None,
               checkpoint: bool = True) -> UnitFile:
    # Inputs: Ctx, Unit, experiment, measurement, the fingerprinted config, unfingerprinted config extras and
    # whether to record the checkpoint in provenance. Output: the resumable UnitFile for this measurement.
    rf = run_result(ctx, experiment, unit.run_id, measurement, unit=unit)
    if checkpoint:
        rf.provenance_fn = functools.partial(provenance, ctx, checkpoint=checkpoint_info(ctx, unit))
    return UnitFile.open(rf, {"config": {**cfg, **(extra_cfg or {})}}, fingerprint(cfg), force=ctx.force,
                         flush_seconds=ctx.config(InterpConfig).flush_seconds)


def done_check(experiment: str, measurement: str, cfg_fn):
    # Inputs: experiment, measurement and cfg_fn(ctx, unit) -> fingerprinted config. Output: an is_done(ctx,
    # unit) that is true when the result is complete under the current fingerprint.
    def is_done(ctx, unit):
        # Inputs: Ctx and Unit. Output: whether this measurement is complete with a matching fingerprint.
        path = locate(ctx, experiment, unit.run_id, measurement)["result"]
        return measurement_done(path, fingerprint(cfg_fn(ctx, unit)))
    return is_done


def load_result(ctx, unit, experiment: str, measurement: str, cfg: dict) -> dict:
    # Inputs: Ctx, Unit, experiment, measurement and the expected fingerprinted config. Output: the complete
    # upstream result; raises when it is missing, partial or stale.
    path = locate(ctx, experiment, unit.run_id, measurement)["result"]
    if not measurement_done(path, fingerprint(cfg)):
        raise RuntimeError(f"{path} is missing, incomplete or from another config; run {measurement} first")
    return ResultFile(path).load()


def loaded(ctx, unit):
    # Inputs: Ctx and a model x condition Unit. Output: a context manager yielding the fine-tuned Trained
    # bundle (float32, prepared for interp, on ctx.device) that releases the model on exit.
    return load_model(ctx, unit.model, unit.condition, device=ctx.device)


def nan_to_none(a, digits=None):
    # Inputs: an array-like of floats and optional rounding digits. Output: nested lists with NaN as None.
    a = np.asarray(a, dtype=np.float64)
    a = np.round(a, digits) if digits is not None else a
    return np.where(np.isnan(a), None, a).tolist()


def layer_labels(n_blocks: int) -> list:
    # Input: number of encoder blocks. Output: ["emb", "0", ..., "N-1"], the layer axis of the states arrays.
    return [EMB] + [str(b) for b in range(n_blocks)]


def pooled_states(trained, enc_batch) -> tuple:
    # Inputs: Trained and an encoder batch {"input_ids", "attention_mask"}. Output: (float32 numpy
    # [B, N+1, d] masked-mean states, embedding output first then every block output, EOS included; token
    # counts [B]). Each hook reduces its tensor to the pooled vector, so full states are never kept.
    batch = to_device(enc_batch, trained.device)
    mask = batch["attention_mask"].float()
    denom = mask.sum(1, keepdim=True)
    pool = lambda h: (torch.einsum("bsd,bs->bd", h.float(), mask) / denom).cpu()
    model = trained.model
    emb_mod = {-1: hooks.embed_prehook_module(model)}
    with torch.no_grad(), hooks.record(emb_mod, reduce=pool, pre=True) as emb, \
            hooks.record_blocks(model, "enc", reduce=pool) as rec:
        model.encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    layers = [emb.out[-1]] + [rec.out[b] for b in range(trained.n_enc)]
    return torch.stack(layers, 1).numpy(), mask.sum(1).long().cpu().numpy()


def states_config(ctx, unit) -> dict:
    # Inputs: Ctx and Unit. Output: the fingerprinted config of the states measurement.
    icfg = ctx.config(InterpConfig)
    return {"model": unit.model, "condition": unit.condition, "split": SPLIT, "langs": study_langs(ctx),
            **rows_config(devtest_rows(ctx)), "max_chars": icfg.max_chars, "factor": DecodingConfig().token_factor,
            "pooling": "masked_mean_with_eos", "layers": "emb+blocks", "dtype": "float32", "checkpoint": "best"}


def pool_language(ctx, unit, trained, texts, rows, lang: str) -> dict:
    # Inputs: Ctx, Unit, Trained, all devtest sentences of one language, row ids and the language code.
    # Saves the pooled states [n_rows, N+1, d] as .npy. Output: the unit record (array ref, shape and
    # per-sentence token and character counts).
    icfg = ctx.config(InterpConfig)
    factor = DecodingConfig().token_factor
    step = lambda enc, brows: list(zip(*pooled_states(trained, enc)))
    out = run_batched(trained, texts, rows, icfg.max_chars, factor, batching_from(icfg), step)
    arr = np.stack([vec for vec, _ in out]).astype(np.float32)
    ref = save_run_array(ctx, EXP, unit.run_id, f"{lang}.npy", arr)
    return {"path": ref, "shape": list(arr.shape), "dtype": "float32", "n_tokens": [int(n) for _, n in out],
            "n_chars": [len(truncate_chars(texts[r], icfg.max_chars)) for r in rows]}


def _pool_missing(ctx, unit, uf, langs, rows) -> None:
    # Inputs: Ctx, Unit, the open UnitFile, languages and row ids. Loads the model only when a language is
    # still missing and pools every missing language (resume unit = language).
    todo = [lang for lang in langs if not uf.has(lang)]
    if not todo:
        return
    texts = flores_texts(ctx, todo)
    with loaded(ctx, unit) as trained:
        for lang in todo:
            uf.put(lang, pool_language(ctx, unit, trained, texts[lang], rows, lang))


def states_columns(uf, langs, rows) -> tuple:
    # Inputs: the UnitFile with every language done, languages and row ids. Output: (columnar per_item with
    # array refs and per-row counts, aggregate with sizes and mean counts).
    recs = {lang: uf.get(lang) for lang in langs}
    n_layers = recs[langs[0]]["shape"][1]
    per_item = {"row_ids": list(rows), "langs": list(langs), "layers": layer_labels(n_layers - 1),
                "arrays": {lang: {k: recs[lang][k] for k in ("path", "shape", "dtype")} for lang in langs},
                "n_tokens": {lang: recs[lang]["n_tokens"] for lang in langs},
                "n_chars": {lang: recs[lang]["n_chars"] for lang in langs}}
    aggregate = {"n_layers": n_layers, "d_model": recs[langs[0]]["shape"][2], "n_rows": len(rows),
                 "mean_tokens": {lang: float(np.mean(recs[lang]["n_tokens"])) for lang in langs},
                 "mean_chars": {lang: float(np.mean(recs[lang]["n_chars"])) for lang in langs}}
    return aggregate, per_item


@register_stage("interp.hubness.states", grid=GRID, is_done=done_check(EXP, "states", states_config),
                requires=("train",), order=510)
def states(ctx, unit):
    # Inputs: Ctx and a model x condition Unit. Writes pooled encoder states for every study language and
    # layer (.npy per language) and the states result JSON listing them.
    if not ctx.force and done_check(EXP, "states", states_config)(ctx, unit):
        return
    cfg = states_config(ctx, unit)
    langs, rows = cfg["langs"], devtest_rows(ctx)
    with open_units(ctx, unit, EXP, "states", cfg, batching_config(ctx)) as uf:
        _pool_missing(ctx, unit, uf, langs, rows)
        aggregate, per_item = states_columns(uf, langs, rows)
        uf.finalize(aggregate, per_item=per_item)


def zscore_paired_cosines(xs: np.ndarray) -> np.ndarray:
    # Input: states of one layer [n_langs, n_rows, d] (row-aligned translations). Every dimension is
    # z-scored over all n_langs * n_rows sentences (constant dimensions are only centred). Output: paired
    # cosines [n_langs, n_langs, n_rows] between the languages' versions of the same sentence.
    flat = xs.reshape(-1, xs.shape[-1]).astype(np.float64)
    sd = flat.std(0)
    z = (flat - flat.mean(0)) / np.where(sd > 0, sd, 1.0)
    z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    z = z.reshape(xs.shape)
    return np.einsum("lnd,mnd->lmn", z, z)


def lang_pairs(langs) -> list:
    # Input: language codes. Output: the unordered pairs (i < j) as (i, j, "a|b") in registry order.
    return [(i, j, f"{langs[i]}|{langs[j]}") for i in range(len(langs)) for j in range(i + 1, len(langs))]


def similarity_config(ctx, unit) -> dict:
    # Inputs: Ctx and Unit. Output: the fingerprinted similarity config (chained to the states fingerprint).
    return {"states": fingerprint(states_config(ctx, unit)), "standardize": "zscore_per_dim_all_langs",
            "similarity": "mean_paired_cosine"}


def layer_similarity(ctx, unit, arrays: dict, langs, li: int, label: str) -> dict:
    # Inputs: Ctx, Unit, {lang: states [n_rows, N+1, d]}, languages, the layer index and label. Saves the
    # per-sentence paired cosines [n_pairs, n_rows] as .npy. Output: the unit record (matrix and array ref).
    cos = zscore_paired_cosines(np.stack([np.asarray(arrays[lang][:, li, :]) for lang in langs]))
    pairs = lang_pairs(langs)
    per_pair = np.stack([cos[i, j] for i, j, _ in pairs]).astype(np.float32)
    ref = save_run_array(ctx, EXP, unit.run_id, f"similarity/{label}.npy", per_pair)
    return {"matrix": cos.mean(-1).tolist(), "cosine": ref}


@register_stage("interp.hubness.similarity", grid=GRID, is_done=done_check(EXP, "similarity", similarity_config),
                requires=("interp.hubness.states",), order=511)
def similarity(ctx, unit):
    # Inputs: Ctx and a model x condition Unit. Writes per-layer 11x11 similarity matrices (aggregate) and
    # per-sentence paired cosines for every language pair (per_item, .npy per layer); resume unit = layer.
    if not ctx.force and done_check(EXP, "similarity", similarity_config)(ctx, unit):
        return
    st = load_result(ctx, unit, EXP, "states", states_config(ctx, unit))["per_item"]
    langs, labels = st["langs"], st["layers"]
    arrays = {lang: load_array(ctx, st["arrays"][lang]["path"], mmap=True) for lang in langs}
    cfg = similarity_config(ctx, unit)
    with open_units(ctx, unit, EXP, "similarity", cfg, checkpoint=False) as uf:
        for li, label in enumerate(labels):
            if not uf.has(label):
                uf.put(label, layer_similarity(ctx, unit, arrays, langs, li, label))
        matrices = {label: uf.get(label)["matrix"] for label in labels}
        blocks = np.mean([matrices[label] for label in labels if label != EMB], axis=0)
        per_item = {"row_ids": st["row_ids"], "langs": langs, "layers": labels,
                    "pairs": [p for _, _, p in lang_pairs(langs)],
                    "cosine": {label: uf.get(label)["cosine"] for label in labels},
                    "cosine_shape": [len(lang_pairs(langs)), len(st["row_ids"])]}
        uf.finalize({"langs": langs, "layers": labels, "matrix": matrices, "block_mean_matrix": blocks.tolist()},
                    per_item=per_item)


def rank_layer(sim) -> dict:
    # Input: an [n, n] similarity matrix (diagonal ignored). Output: {"rank": [n, n] where rank[i, j] is
    # j's rank in i's list of the others (1 = most similar, ties averaged, NaN diagonal), "mean_rank": [n]
    # mean rank of each language in the others' lists, "nearest": [n] index of each language's nearest
    # neighbour, "nn_count": [n] how many languages have it as nearest neighbour (ties share credit),
    # "nn_frac": nn_count / (n - 1)}.
    sim = np.asarray(sim, dtype=np.float64)
    n = len(sim)
    rank, credit = np.full((n, n), np.nan), np.zeros(n)
    for i in range(n):
        others = np.array([j for j in range(n) if j != i])
        rank[i, others] = rankdata(-sim[i, others], method="average")
        best = others[sim[i, others] == sim[i, others].max()]
        credit[best] += 1.0 / len(best)
    nearest = [int(np.nanargmin(rank[i])) for i in range(n)]
    return {"rank": rank, "mean_rank": np.nanmean(rank, axis=0), "nearest": nearest, "nn_count": credit,
            "nn_frac": credit / (n - 1)}


def ranks_config(ctx, unit) -> dict:
    # Inputs: Ctx and Unit. Output: the fingerprinted ranks config (chained to the similarity fingerprint).
    return {"similarity": fingerprint(similarity_config(ctx, unit)), "ties": "average", "average_over": "blocks"}


def hub_summary(langs, labels, tables: dict) -> dict:
    # Inputs: languages, layer labels and {label: rank_layer output}. Output: the aggregate: mean rank and
    # nearest-neighbour fraction per language averaged over block layers (embedding output excluded), the
    # languages sorted by mean rank (registry order breaks ties) and the per-layer mean ranks.
    blocks = [label for label in labels if label != EMB]
    mean_rank = np.mean([tables[label]["mean_rank"] for label in blocks], axis=0)
    nn_frac = np.mean([tables[label]["nn_frac"] for label in blocks], axis=0)
    order = sorted(range(len(langs)), key=lambda i: (mean_rank[i], i))
    return {"langs": list(langs), "layers": list(labels), "averaged_layers": blocks,
            "mean_rank": {lang: float(mean_rank[i]) for i, lang in enumerate(langs)},
            "sorted_by_mean_rank": [langs[i] for i in order],
            "mean_nn_frac": {lang: float(nn_frac[i]) for i, lang in enumerate(langs)},
            "mean_rank_by_layer": {label: dict(zip(langs, tables[label]["mean_rank"].tolist())) for label in labels}}


def rank_columns(langs, labels, tables: dict) -> dict:
    # Inputs: languages, layer labels and {label: rank_layer output}. Output: per-layer rank tables (None on
    # the diagonal), nearest neighbours, nearest-neighbour counts and fractions.
    return {"langs": list(langs), "layers": list(labels),
            "rank": {label: nan_to_none(tables[label]["rank"]) for label in labels},
            "nearest": {label: {lang: langs[j] for lang, j in zip(langs, tables[label]["nearest"])} for label in labels},
            "nn_count": {label: dict(zip(langs, tables[label]["nn_count"].tolist())) for label in labels},
            "nn_frac": {label: dict(zip(langs, tables[label]["nn_frac"].tolist())) for label in labels}}


@register_stage("interp.hubness.ranks", grid=GRID, is_done=done_check(EXP, "ranks", ranks_config),
                requires=("interp.hubness.similarity",), order=512)
def ranks(ctx, unit):
    # Inputs: Ctx and a model x condition Unit. Writes per-layer rank tables and nearest neighbours
    # (per_item) and the layer-averaged mean ranks with the hub-sorted language list (aggregate).
    if not ctx.force and done_check(EXP, "ranks", ranks_config)(ctx, unit):
        return
    sim = load_result(ctx, unit, EXP, "similarity", similarity_config(ctx, unit))["aggregate"]
    langs, labels = sim["langs"], sim["layers"]
    tables = {label: rank_layer(sim["matrix"][label]) for label in labels}
    with open_units(ctx, unit, EXP, "ranks", ranks_config(ctx, unit), checkpoint=False) as uf:
        uf.finalize(hub_summary(langs, labels, tables), per_item=rank_columns(langs, labels, tables))


composite("interp.hubness", ("interp.hubness.states", "interp.hubness.similarity", "interp.hubness.ranks"))
