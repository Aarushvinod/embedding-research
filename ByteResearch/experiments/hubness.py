# Hubness experiment: does a direction model's encoder pull its source sentences towards English, layer by layer?
# Stage states pools the FLORES devtest sentences of the source (hub), English, the es/zh/el controls and the
# model's target over their tokens (masked mean with EOS) at the embedding output and every encoder block.
# Stage similarity z-scores every dimension over all six languages' sentences at each layer, saves those
# means and standard deviations, and scores each reference by paired cosine to the source, z-scored and plain.
# Stage ranks orders the five references per layer, takes the English margin over the controls, places each
# layer at its relative depth and averages ranks and margins over all layers including the embedding output.
# The module also holds the helpers shared with the DecoderLens experiment: FLORES access, checkpoint
# provenance and resumable unit files whose pilot paths come from core.results.locate.

from __future__ import annotations

import functools
import importlib

import numpy as np
import torch
from scipy.stats import rankdata

from ..core.config import CONDITIONS, CONTROLS, PINS, DecodingConfig, InterpConfig, config_hash
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
GRID = ("model", "condition", "target")
ENGLISH = "en"


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


def rows_config(rows) -> dict:
    # Input: row ids. Output: their count and hash, for config fingerprints.
    return {"n_rows": len(rows), "rows_hash": config_hash(list(rows))}


def batching_config(ctx) -> dict:
    # Input: Ctx. Output: the batching settings, stored in configs but kept out of fingerprints because they
    # do not change results beyond float noise.
    icfg = ctx.config(InterpConfig)
    return {"budget_tokens": icfg.budget_tokens, "max_batch": icfg.max_batch, "gen_max_batch": icfg.gen_max_batch}


def checkpoint_info(ctx, unit) -> dict:
    # Inputs: Ctx and a model x condition x target Unit. Output: the direction checkpoint's path and its meta
    # step/lr (read from byteresearch_meta.json when present) for provenance.
    info = describe_checkpoint(ctx, unit.model, unit.condition, target=unit.target)
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
    # Inputs: Ctx and a model x condition x target Unit. Output: a context manager yielding the direction's
    # fine-tuned Trained bundle (float32, prepared for interp, on ctx.device) that releases the model on exit.
    return load_model(ctx, unit.model, unit.condition, target=unit.target, device=ctx.device)


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


def references(unit) -> list:
    # Input: a model x condition x target Unit. Output: the reference languages: English, the controls and
    # the model's target.
    return [ENGLISH, *CONTROLS, unit.target]


def encoded_langs(unit) -> list:
    # Input: a model x condition x target Unit. Output: the source (hub) language followed by the references.
    return [CONDITIONS[unit.condition].hub, *references(unit)]


def states_config(ctx, unit) -> dict:
    # Inputs: Ctx and Unit. Output: the fingerprinted config of the states measurement.
    icfg = ctx.config(InterpConfig)
    return {"model": unit.model, "condition": unit.condition, "target": unit.target, "split": SPLIT,
            "langs": encoded_langs(unit), **rows_config(devtest_rows(ctx)), "max_chars": icfg.max_chars,
            "factor": DecodingConfig().token_factor, "pooling": "masked_mean_with_eos", "layers": "emb+blocks",
            "dtype": "float32", "checkpoint": "best"}


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
    # Inputs: Ctx and a model x condition x target Unit. Writes pooled encoder states for the source, English,
    # the controls and the target at every layer (.npy per language) and the states result JSON listing them.
    if not ctx.force and done_check(EXP, "states", states_config)(ctx, unit):
        return
    cfg = states_config(ctx, unit)
    langs, rows = cfg["langs"], devtest_rows(ctx)
    with open_units(ctx, unit, EXP, "states", cfg, batching_config(ctx)) as uf:
        _pool_missing(ctx, unit, uf, langs, rows)
        aggregate, per_item = states_columns(uf, langs, rows)
        uf.finalize(aggregate, per_item=per_item)


def layer_stats(xs: np.ndarray) -> tuple:
    # Input: states of one layer [n_langs, n_rows, d]. Output: (mean [d], std [d]) of every dimension over
    # all n_langs * n_rows sentences, in float64.
    flat = xs.reshape(-1, xs.shape[-1]).astype(np.float64)
    return flat.mean(0), flat.std(0)


def paired_cosines(xs: np.ndarray, mean=None, std=None) -> np.ndarray:
    # Inputs: states of one layer [n_langs, n_rows, d] (source first, row-aligned translations) and optional
    # per-dimension mean and std to z-score with (zero-spread dimensions are only centred). Output: cosines
    # [n_langs - 1, n_rows] between each source sentence and every other language's translation of it.
    x = xs.astype(np.float64)
    if mean is not None:
        x = (x - mean) / np.where(std > 0, std, 1.0)
    x /= np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)
    return np.einsum("nd,lnd->ln", x[0], x[1:])


def similarity_config(ctx, unit) -> dict:
    # Inputs: Ctx and Unit. Output: the fingerprinted similarity config (chained to the states fingerprint).
    return {"states": fingerprint(states_config(ctx, unit)), "standardize": "zscore_per_dim_all_langs",
            "similarity": "paired_cosine_source_vs_reference", "plain": True}


def layer_similarity(arrays: dict, langs, li: int) -> dict:
    # Inputs: {lang: states [n_rows, N+1, d]}, languages (source first) and the layer index. Output: the
    # layer's per-dimension mean and std and its z-scored and plain source-vs-reference cosines.
    xs = np.stack([np.asarray(arrays[lang][:, li, :]) for lang in langs])
    mean, std = layer_stats(xs)
    return {"mean": mean, "std": std, "cosine_z": paired_cosines(xs, mean, std), "cosine_raw": paired_cosines(xs)}


def save_similarity(ctx, unit, layers: list) -> dict:
    # Inputs: Ctx, Unit and layer_similarity outputs in layer order. Saves each quantity stacked over layers
    # (mean/std [N+1, d], cosines [N+1, n_refs, n_rows]) as float32 .npy. Output: {name: {path, shape}}.
    out = {}
    for name in ("mean", "std", "cosine_z", "cosine_raw"):
        arr = np.stack([layer[name] for layer in layers]).astype(np.float32)
        out[name] = {"path": save_run_array(ctx, EXP, unit.run_id, f"{name}.npy", arr), "shape": list(arr.shape)}
    return out


def mean_by_layer(labels, refs, layers: list, name: str) -> dict:
    # Inputs: layer labels, reference languages, layer_similarity outputs and a cosine name. Output:
    # {label: {ref: mean cosine over sentences}}.
    return {label: dict(zip(refs, layer[name].mean(-1).tolist())) for label, layer in zip(labels, layers)}


@register_stage("interp.hubness.similarity", grid=GRID, is_done=done_check(EXP, "similarity", similarity_config),
                requires=("interp.hubness.states",), order=511)
def similarity(ctx, unit):
    # Inputs: Ctx and a model x condition x target Unit. Writes the per-layer standardisation statistics and
    # the per-sentence z-scored and plain source-vs-reference cosines (.npy) and their per-layer means.
    if not ctx.force and done_check(EXP, "similarity", similarity_config)(ctx, unit):
        return
    st = load_result(ctx, unit, EXP, "states", states_config(ctx, unit))["per_item"]
    langs, labels = st["langs"], st["layers"]
    arrays = {lang: load_array(ctx, st["arrays"][lang]["path"], mmap=True) for lang in langs}
    layers = [layer_similarity(arrays, langs, li) for li in range(len(labels))]
    with open_units(ctx, unit, EXP, "similarity", similarity_config(ctx, unit), checkpoint=False) as uf:
        per_item = {"row_ids": st["row_ids"], "source": langs[0], "references": langs[1:], "layers": labels,
                    "arrays": save_similarity(ctx, unit, layers)}
        uf.finalize({"source": langs[0], "references": langs[1:], "layers": labels,
                     "mean_cosine_z": mean_by_layer(labels, langs[1:], layers, "cosine_z"),
                     "mean_cosine_raw": mean_by_layer(labels, langs[1:], layers, "cosine_raw")}, per_item=per_item)


def relative_depth(labels) -> list:
    # Input: layer labels ["emb", "0", ..., "N-1"]. Output: depths in [0, 1]: 0 for the embedding output and
    # (b + 1) / N for block b.
    n_blocks = len(labels) - 1
    return [0.0 if label == EMB else (int(label) + 1) / n_blocks for label in labels]


def rank_layer(sims: dict, refs) -> dict:
    # Inputs: {ref: similarity} of one layer and the reference order. Output: {ref: rank} (1 = most
    # similar, ties averaged).
    return dict(zip(refs, rankdata([-sims[r] for r in refs], method="average").tolist()))


def english_margin(sims: dict) -> float:
    # Input: {ref: similarity} of one layer. Output: similarity to English minus the mean over the controls.
    return float(sims[ENGLISH] - np.mean([sims[c] for c in CONTROLS]))


def rank_summary(refs, labels, by_layer: dict) -> tuple:
    # Inputs: references, layer labels and {label: {ref: mean similarity}}. Output: (averages over all layers
    # including the embedding output: mean rank per reference, references sorted by it and mean English
    # margin; per-layer ranks and margins).
    rank = {label: rank_layer(by_layer[label], refs) for label in labels}
    margin = {label: english_margin(by_layer[label]) for label in labels}
    mean_rank = {r: float(np.mean([rank[label][r] for label in labels])) for r in refs}
    order = sorted(refs, key=lambda r: (mean_rank[r], refs.index(r)))
    averages = {"mean_rank": mean_rank, "sorted_by_mean_rank": order,
                "mean_english_margin": float(np.mean(list(margin.values())))}
    return averages, {"rank": rank, "english_margin": margin}


def ranks_config(ctx, unit) -> dict:
    # Inputs: Ctx and Unit. Output: the fingerprinted ranks config (chained to the similarity fingerprint).
    return {"similarity": fingerprint(similarity_config(ctx, unit)), "ties": "average",
            "average_over": "all_layers", "margin_controls": list(CONTROLS)}


@register_stage("interp.hubness.ranks", grid=GRID, is_done=done_check(EXP, "ranks", ranks_config),
                requires=("interp.hubness.similarity",), order=512)
def ranks(ctx, unit):
    # Inputs: Ctx and a model x condition x target Unit. Writes per-layer reference ranks and English margins
    # at relative depth (per_item) and their averages over all layers (aggregate), z-scored and plain.
    if not ctx.force and done_check(EXP, "ranks", ranks_config)(ctx, unit):
        return
    sim = load_result(ctx, unit, EXP, "similarity", similarity_config(ctx, unit))["aggregate"]
    refs, labels = sim["references"], sim["layers"]
    z_avg, z_layers = rank_summary(refs, labels, sim["mean_cosine_z"])
    raw_avg, raw_layers = rank_summary(refs, labels, sim["mean_cosine_raw"])
    with open_units(ctx, unit, EXP, "ranks", ranks_config(ctx, unit), checkpoint=False) as uf:
        uf.finalize({"source": sim["source"], "references": refs, "averaged_layers": labels, **z_avg,
                     "plain": raw_avg},
                    per_item={"layers": labels, "depth": relative_depth(labels), **z_layers, "plain": raw_layers})


composite("interp.hubness", ("interp.hubness.states", "interp.hubness.similarity", "interp.hubness.ranks"))
