# DecoderLens experiment: what would the unchanged decoder say if it read an intermediate encoder layer?
# For every encoder block b, the lens is the encoder's final layer norm applied to block b's output, handed to
# the decoder as its encoder outputs; the source is the hub-language FLORES devtest sentence and the decoder
# prefix is [start, tag_X], exactly as in the trained direction.
# Stage logprob teacher-forces the English, source-language and X references of the same row and stores
# per-char log-probs, their differences and the English share (a softmax over the per-char log-probs), with
# per-layer means and across-layer statistics per target and pooled over targets.
# Stage lid decodes greedily from every lens (128 characters at most), scores the outputs with GlotLID (summed
# top-k probabilities over each language's ISO-derived label set, against that language's FLORES-calibrated
# minimum, as in the data filter) and buckets them as en, src, x, other_study, other or und; stage summary
# joins both into one per-layer table.
# Work is resumable per (target, chunk of rows), block states are offloaded to the CPU while the lens runs.

from __future__ import annotations

import functools

import numpy as np
import torch
from scipy.special import logsumexp

from ..core.config import CONDITIONS, LID_EXTRA_CODES, MODELS, PINS, STUDY11, TARGETS, DataConfig, DecodingConfig, \
    InterpConfig, resolve_langs
from ..core.registry import composite, opt, register_stage
from ..interp.scoring import batching_from, generate_from, lens_output, per_char_logprob, run_batched, to_device
from ..interp.units import fingerprint, row_chunks
from ..modeling import hooks
from .hubness import SPLIT, batching_config, corpus_module, devtest_rows, done_check, flores_texts, load_result, \
    loaded, nan_to_none, open_units, rows_config

EXP = "interp_decoderlens"
GRID = ("model", "condition")
BUCKETS = ("en", "src", "x", "other_study", "other", "und")
CHUNK_ROWS = 128
OPTIONS = (opt("--chunk-rows", type=int, default=None, help="DecoderLens resume chunk size in rows (default 128)"),)


def lens_targets(ctx, unit) -> list:
    # Inputs: Ctx and Unit. Output: the condition's target languages (smoke keeps the smoke targets).
    allowed = set(resolve_langs("all", TARGETS, ctx.smoke))
    return [t for t in CONDITIONS[unit.condition].targets if t in allowed]


def roles_for(hub: str) -> tuple:
    # Input: the hub (source) language. Output: the reference roles; C1 (hub en) has no separate source role.
    return ("en", "x") if hub == "en" else ("en", "src", "x")


def role_lang(role: str, hub: str, x: str) -> str:
    # Inputs: a role, the hub and the target. Output: the language whose FLORES sentence fills that role.
    return {"en": "en", "src": hub, "x": x}[role]


def gen_token_factor(model_key: str) -> int:
    # Input: registry model key. Output: tokens allowed per output character for the 128-char lens outputs:
    # 4 for ByT5 (up to 4 bytes per character), 1 for mT5 (a subword covers at least a character).
    return 4 if MODELS[model_key].arch == "byt5" else 1


def base_config(ctx, unit) -> dict:
    # Inputs: Ctx and Unit. Output: the fingerprinted settings shared by the logprob and lid measurements.
    icfg = ctx.config(InterpConfig)
    return {"model": unit.model, "condition": unit.condition, "hub": CONDITIONS[unit.condition].hub,
            "targets": lens_targets(ctx, unit), "split": SPLIT, **rows_config(devtest_rows(ctx)),
            "max_chars": icfg.max_chars, "factor": DecodingConfig().token_factor,
            "chunk_rows": int(ctx.opt("chunk_rows", CHUNK_ROWS)), "lens": "encoder.final_layer_norm(block_b)",
            "layers": "blocks", "checkpoint": "best"}


def logprob_config(ctx, unit) -> dict:
    # Inputs: Ctx and Unit. Output: the fingerprinted logprob config.
    cfg = base_config(ctx, unit)
    return {**cfg, "roles": list(roles_for(cfg["hub"])), "measure": "per_char_logprob_excl_eos",
            "chunk_logits": ctx.config(InterpConfig).logprob_chunk_rows}


def lid_decoding(ctx, unit) -> DecodingConfig:
    # Inputs: Ctx and Unit. Output: greedy decoding capped at the LID window (128 characters).
    icfg = ctx.config(InterpConfig)
    return DecodingConfig(num_beams=1, length_penalty=1.0, max_input_chars=icfg.max_chars,
                          max_output_chars=icfg.lid_window, token_factor=gen_token_factor(unit.model))


def lid_config(ctx, unit) -> dict:
    # Inputs: Ctx and Unit. Output: the fingerprinted lid config.
    icfg, dec = ctx.config(InterpConfig), lid_decoding(ctx, unit)
    table = corpus_module("thresholds").load_thresholds(ctx, required=False)
    langs = dict.fromkeys(("en", CONDITIONS[unit.condition].hub, *STUDY11))
    return {**base_config(ctx, unit), "decoding": "greedy", "max_output_chars": dec.max_output_chars,
            "gen_token_factor": dec.token_factor, "lid": "glotlid",
            "lid_thresholds": {lang: table.get("lid", {}).get(lang) for lang in langs},
            "lid_calibration": table.get("percentile"), "lid_other_threshold": icfg.lid_other_threshold,
            "lid_top_k": ctx.config(DataConfig).glotlid_top_k, "lid_iso639_3": PINS["iso639_3"]["sha256"],
            "lid_extra_codes": {c: list(v) for c, v in LID_EXTRA_CODES.items()}, "buckets": list(BUCKETS)}


def lens_states(trained, enc_batch) -> tuple:
    # Inputs: Trained and an encoder batch. Output: ({b: output of block b, float32 on the CPU}, attention mask
    # on the model device); offloading keeps only one layer's lens on the accelerator at a time.
    batch = to_device(enc_batch, trained.device)
    offload = lambda h: h.detach().float().cpu()
    with torch.no_grad(), hooks.record_blocks(trained.model, "enc", reduce=offload) as rec:
        trained.model.encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    return rec.out, batch["attention_mask"]


def lens_step(trained, read):
    # Inputs: Trained and read(lens, mask, batch_rows) -> one value per batch row. Output: a run_batched step
    # returning, for every row, the list of its values over encoder blocks 0..N-1.
    def step(enc, brows):
        # Inputs: encoder batch and its rows. Output: per-row lists of per-layer values.
        states, mask = lens_states(trained, enc)
        layers = [read(lens_output(trained, states[b]), mask, brows) for b in range(trained.n_enc)]
        return [[layer[i] for layer in layers] for i in range(len(brows))]
    return step


def score_roles(trained, refs: dict, x: str, icfg: InterpConfig, factor: int):
    # Inputs: Trained, {role: all row-aligned references}, the target (tag) language, InterpConfig and token
    # factor. Output: read(lens, mask, rows) -> per-row {role: per-char log-prob} under the tag of x.
    def read(lens, mask, brows):
        # Inputs: lens encoder outputs, mask and batch rows. Output: per-row {role: per-char log-prob}.
        res = {role: per_char_logprob(trained, lens, mask, x, [texts[r] for r in brows], icfg.max_chars, factor,
                                      icfg.logprob_chunk_rows).per_char for role, texts in refs.items()}
        return [{role: float(v[i]) for role, v in res.items()} for i in range(len(brows))]
    return read


def logprob_chunk(ctx, trained, texts: dict, cfg: dict, x: str, crows) -> dict:
    # Inputs: Ctx, Trained, {lang: devtest sentences}, the logprob config, the target and a chunk of row ids.
    # Output: {"lp_<role>": [layer][row] per-char log-probs} for the chunk.
    icfg, hub = ctx.config(InterpConfig), cfg["hub"]
    refs = {role: texts[role_lang(role, hub, x)] for role in cfg["roles"]}
    step = lens_step(trained, score_roles(trained, refs, x, icfg, cfg["factor"]))
    out = run_batched(trained, texts[hub], crows, cfg["max_chars"], cfg["factor"], batching_from(icfg), step)
    return {f"lp_{role}": [[row[b][role] for row in out] for b in range(trained.n_enc)] for role in cfg["roles"]}


def lid_buckets(lid_model, texts, hub: str, x: str, thresholds: dict, other: float, aliases: dict, k: int) -> tuple:
    # Inputs: a GlotLID model, generated strings, hub, target, {lang: calibrated minimum}, the top-label minimum
    # for "other", the alias table from corpora.lid.load_aliases and the top-k. Output: (top labels, top probs,
    # buckets) per string; empty strings are not classified and, like low-confidence ones, become und.
    lid = corpus_module("lid")
    clean = [" ".join(str(t).split()) for t in texts]
    keep = [i for i, t in enumerate(clean) if t]
    preds = [None] * len(texts)
    for i, pred in zip(keep, lid.predict(lid_model, [clean[i] for i in keep], k=k) if keep else []):
        preds[i] = pred
    tops = [lid.top_label(p) if p is not None else (None, 0.0) for p in preds]
    buckets = [bucket_of(p, hub, x, thresholds, other, aliases) for p in preds]
    return [t[0] for t in tops], [float(t[1]) for t in tops], buckets


def bucket_of(pred, hub: str, x: str, thresholds: dict, other: float, aliases: dict) -> str:
    # Inputs: one string's top-k GlotLID prediction (None for empty text), hub, target, {lang: calibrated
    # minimum}, the top-label minimum for "other" and the alias table. Output: the bucket: the first of en,
    # src (the hub; never reached in C1 where src = en) and x whose labels' summed probability reaches that
    # language's minimum, else other_study for another study language, else other when the top label alone
    # reaches `other`, else und.
    lid = corpus_module("lid")
    if pred is None:
        return "und"
    prob = {lang: lid.labels_prob(pred, aliases[lang]) for lang in dict.fromkeys(("en", hub, x, *STUDY11))}
    passes = lambda lang: prob[lang] >= thresholds[lang]
    named = [b for b, lang in (("en", "en"), ("src", hub), ("x", x)) if passes(lang)]
    if named:
        return named[0]
    if any(passes(lang) for lang in STUDY11):
        return "other_study"
    return "other" if lid.top_label(pred)[1] >= other else "und"


def lid_chunk(lid_model, ctx, trained, texts: dict, cfg: dict, x: str, crows, decoding: DecodingConfig) -> dict:
    # Inputs: the GlotLID model, Ctx, Trained, {lang: devtest sentences}, the lid config, the target, a chunk of
    # row ids and the lens decoding. Output: {"text", "label", "prob", "bucket"}, each [layer][row].
    icfg, hub = ctx.config(InterpConfig), cfg["hub"]
    read = lambda lens, mask, brows: generate_from(trained, lens, mask, x, decoding)
    out = run_batched(trained, texts[hub], crows, cfg["max_chars"], cfg["factor"], batching_from(icfg, True),
                      lens_step(trained, read))
    gens = [row[b] for b in range(trained.n_enc) for row in out]
    aliases, k = corpus_module("lid").load_aliases(ctx), ctx.config(DataConfig).glotlid_top_k
    thresholds = corpus_module("thresholds").load_thresholds(ctx)["lid"]
    labels, probs, buckets = lid_buckets(lid_model, gens, hub, x, thresholds, cfg["lid_other_threshold"], aliases, k)
    shape = lambda flat: [flat[b * len(crows):(b + 1) * len(crows)] for b in range(trained.n_enc)]
    return {"text": shape(gens), "label": shape(labels), "prob": shape(probs), "bucket": shape(buckets)}


def _run_missing(ctx, unit, uf, cfg: dict, make_work) -> None:
    # Inputs: Ctx, Unit, the open UnitFile, the config and make_work(ctx) -> work(ctx, trained, texts, cfg, x,
    # rows). Loads FLORES, any helper model and the fine-tuned model only when a (target, chunk) unit is
    # missing, then runs every missing unit.
    chunks = row_chunks(devtest_rows(ctx), cfg["chunk_rows"])
    todo = [(x, i, c) for x in cfg["targets"] for i, c in enumerate(chunks) if not uf.has((x, i))]
    if not todo:
        return
    texts = flores_texts(ctx, sorted({"en", cfg["hub"], *(x for x, _, _ in todo)}))
    work = make_work(ctx)
    with loaded(ctx, unit) as trained:
        for x, i, crows in todo:
            uf.put((x, i), work(ctx, trained, texts, cfg, x, crows))


def assemble(uf, cfg: dict, fields) -> dict:
    # Inputs: a UnitFile with every (target, chunk) done, the config and field names. Output: {target:
    # {field: object array [layer, row]}} with chunks concatenated in row order.
    n_chunks = len(row_chunks(list(range(cfg["n_rows"])), cfg["chunk_rows"]))
    parts = {x: [uf.get((x, i)) for i in range(n_chunks)] for x in cfg["targets"]}
    return {x: {f: np.concatenate([np.asarray(p[f], dtype=object) for p in ps], axis=1) for f in fields}
            for x, ps in parts.items()}


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


def share_fields(lp: dict, roles) -> dict:
    # Inputs: {role: per-char log-probs [L, n]} and the roles. Output: per-sentence fields [L, n]: lp_<role>,
    # delta_x = lp_en - lp_x, delta_src = lp_en - lp_src (None without a source role) and share_<role>, the
    # softmax of the per-char log-probs over the roles.
    stack = np.stack([np.asarray(lp[r], dtype=np.float64) for r in roles])
    shares = np.exp(stack - logsumexp(stack, axis=0))
    out = {f"lp_{r}": stack[k] for k, r in enumerate(roles)}
    out.update({f"share_{r}": shares[k] for k, r in enumerate(roles)})
    out["delta_x"] = out["lp_en"] - out["lp_x"]
    out["delta_src"] = out["lp_en"] - out["lp_src"] if "src" in roles else None
    return out


def logprob_summary(lp: dict, roles) -> dict:
    # Inputs: {role: [L, n]} per-char log-probs and the roles. Output: per-layer means of every field and the
    # across-layer statistics of the mean English share.
    fields = share_fields(lp, roles)
    per_layer = {k: (None if v is None else np.nanmean(v, axis=1)) for k, v in fields.items()}
    rivals = [per_layer[f"share_{r}"] for r in roles if r != "en"]
    return {"per_layer": {k: nan_to_none(v) if v is not None else None for k, v in per_layer.items()},
            "en_share_across_layers": across_layers(per_layer["share_en"], rivals)}


def pooled(tables: dict, field: str) -> np.ndarray:
    # Inputs: {target: {field: [L, n]}} and a field. Output: the field concatenated over targets [L, T*n].
    return np.concatenate([np.asarray(t[field], dtype=object) for t in tables.values()], axis=1)


def n_layers_of(tables: dict, field: str) -> int:
    # Inputs: {target: {field: [L, n]}} and a field. Output: the number of lens layers L.
    return len(next(iter(tables.values()))[field])


def logprob_finish(uf, cfg: dict, rows) -> tuple:
    # Inputs: the complete UnitFile, config and row ids. Output: (aggregate per target and pooled, columnar
    # per_item of per-sentence log-probs, deltas and shares per target, each [layer][row]).
    roles = cfg["roles"]
    tables = assemble(uf, cfg, [f"lp_{r}" for r in roles])
    as_lp = lambda t: {r: t[f"lp_{r}"].astype(np.float64) for r in roles}
    agg = {"roles": roles, "targets": cfg["targets"], "layers": list(range(n_layers_of(tables, f"lp_{roles[0]}"))),
           "by_target": {x: logprob_summary(as_lp(t), roles) for x, t in tables.items()},
           "pooled": logprob_summary({r: pooled(tables, f"lp_{r}").astype(np.float64) for r in roles}, roles)}
    per_row = {x: {k: (None if v is None else nan_to_none(v, 6)) for k, v in share_fields(as_lp(t), roles).items()}
               for x, t in tables.items()}
    return agg, {"row_ids": list(rows), "layers": agg["layers"], "roles": roles, "targets": cfg["targets"],
                 "by_target": per_row}


@register_stage("interp.decoderlens.logprob", grid=GRID, is_done=done_check(EXP, "logprob", logprob_config),
                requires=("train",), options=OPTIONS, order=520)
def logprob(ctx, unit):
    # Inputs: Ctx and a model x condition Unit. Writes per-layer lens log-probs of the English, source and X
    # references for every target (per_item) with per-layer and across-layer English-share statistics.
    if not ctx.force and done_check(EXP, "logprob", logprob_config)(ctx, unit):
        return
    cfg = logprob_config(ctx, unit)
    with open_units(ctx, unit, EXP, "logprob", cfg, batching_config(ctx)) as uf:
        _run_missing(ctx, unit, uf, cfg, lambda c: logprob_chunk)
        uf.finalize(*logprob_finish(uf, cfg, devtest_rows(ctx)))


def bucket_fractions(buckets, hub: str) -> dict:
    # Inputs: buckets [L, n] and the hub. Output: {bucket: per-layer fraction [L]} (src is None in C1).
    b = np.asarray(buckets, dtype=object)
    return {k: (None if k == "src" and hub == "en" else (b == k).mean(axis=1)) for k in BUCKETS}


def lid_summary(buckets, hub: str) -> dict:
    # Inputs: buckets [L, n] and the hub. Output: per-layer bucket fractions and the across-layer statistics of
    # the English fraction (rivals are all other buckets).
    frac = bucket_fractions(buckets, hub)
    rivals = [v for k, v in frac.items() if k != "en" and v is not None]
    return {"per_layer": {k: (None if v is None else v.tolist()) for k, v in frac.items()},
            "en_frac_across_layers": across_layers(frac["en"], rivals)}


def lid_finish(uf, cfg: dict, rows) -> tuple:
    # Inputs: the complete UnitFile, config and row ids. Output: (aggregate per target and pooled, per_item
    # with the generated text, label, probability and bucket per layer and row for each target).
    tables = assemble(uf, cfg, ("text", "label", "prob", "bucket"))
    hub = cfg["hub"]
    agg = {"buckets": list(BUCKETS), "targets": cfg["targets"], "layers": list(range(n_layers_of(tables, "bucket"))),
           "by_target": {x: lid_summary(t["bucket"], hub) for x, t in tables.items()},
           "pooled": lid_summary(pooled(tables, "bucket"), hub)}
    per_row = {x: {"text": t["text"].tolist(), "label": t["label"].tolist(),
                   "prob": nan_to_none(t["prob"].astype(np.float64), 6), "bucket": t["bucket"].tolist()}
               for x, t in tables.items()}
    return agg, {"row_ids": list(rows), "layers": agg["layers"], "targets": cfg["targets"], "by_target": per_row}


def lid_worker(ctx, unit):
    # Inputs: Ctx and Unit. Output: make_work(ctx) for _run_missing that loads GlotLID once and returns the
    # lid chunk function bound to it and to the lens decoding.
    decoding = lid_decoding(ctx, unit)
    return lambda c: functools.partial(_lid_work, corpus_module("lid").load_glotlid(c), decoding)


def _lid_work(lid_model, decoding, ctx, trained, texts, cfg, x, crows) -> dict:
    # Inputs: the GlotLID model, lens decoding and the work arguments of _run_missing. Output: lid_chunk's
    # record for one (target, chunk).
    return lid_chunk(lid_model, ctx, trained, texts, cfg, x, crows, decoding)


@register_stage("interp.decoderlens.lid", grid=GRID, is_done=done_check(EXP, "lid", lid_config),
                requires=("train",), options=OPTIONS, order=521)
def lid(ctx, unit):
    # Inputs: Ctx and a model x condition Unit. Writes greedy lens outputs per layer with GlotLID labels and
    # buckets (per_item) and per-layer bucket fractions with across-layer English statistics (aggregate).
    if not ctx.force and done_check(EXP, "lid", lid_config)(ctx, unit):
        return
    cfg = lid_config(ctx, unit)
    with open_units(ctx, unit, EXP, "lid", cfg, batching_config(ctx)) as uf:
        _run_missing(ctx, unit, uf, cfg, lid_worker(ctx, unit))
        uf.finalize(*lid_finish(uf, cfg, devtest_rows(ctx)))


def summary_config(ctx, unit) -> dict:
    # Inputs: Ctx and Unit. Output: the fingerprinted summary config (chained to both inputs' fingerprints).
    return {"logprob": fingerprint(logprob_config(ctx, unit)), "lid": fingerprint(lid_config(ctx, unit))}


def agreement(lp_rows: dict, buckets, roles) -> dict:
    # Inputs: per-sentence fields of one target ({"lp_<role>": [L][n]}), buckets [L][n] and the roles.
    # Output: per-layer agreement between the log-prob argmax role and the LID bucket, over all scored
    # sentences ("agreement") and over sentences whose bucket is one of the roles ("agreement_defined").
    stack = np.stack([np.asarray(lp_rows[f"lp_{r}"], dtype=np.float64) for r in roles])
    valid = ~np.isnan(stack).any(0)
    arg = np.asarray(roles, dtype=object)[np.argmax(np.where(np.isnan(stack), -np.inf, stack), axis=0)]
    b = np.asarray(buckets, dtype=object)
    agree, defined = (arg == b) & valid, np.isin(b, list(roles)) & valid
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"agreement": agree.sum(1) / valid.sum(1), "agreement_defined": (agree & defined).sum(1) / defined.sum(1)}


def combined_table(lp_agg: dict, lid_agg: dict, agree: dict) -> dict:
    # Inputs: one logprob summary, the matching lid summary and agreement curves. Output: the per-layer table
    # (shares, LID fractions, agreement) and the across-layer statistics of both English measures.
    shares = {k: v for k, v in lp_agg["per_layer"].items() if k.startswith("share_")}
    lid_frac = {f"lid_{k}": v for k, v in lid_agg["per_layer"].items()}
    return {"per_layer": {**shares, **lid_frac, **{k: nan_to_none(v) for k, v in agree.items()}},
            "across_layers": {"share_en": lp_agg["en_share_across_layers"], "lid_en": lid_agg["en_frac_across_layers"],
                              "agreement_mean": float(np.nanmean(agree["agreement"]))}}


@register_stage("interp.decoderlens.summary", grid=GRID, is_done=done_check(EXP, "summary", summary_config),
                requires=("interp.decoderlens.logprob", "interp.decoderlens.lid"), order=522)
def summary(ctx, unit):
    # Inputs: Ctx and a model x condition Unit. Joins the logprob and lid results into per-layer agreement
    # between the log-prob argmax role and the LID bucket and a combined across-layer table, per target and
    # pooled.
    if not ctx.force and done_check(EXP, "summary", summary_config)(ctx, unit):
        return
    lp = load_result(ctx, unit, EXP, "logprob", logprob_config(ctx, unit))
    li = load_result(ctx, unit, EXP, "lid", lid_config(ctx, unit))
    roles, targets = lp["aggregate"]["roles"], lp["aggregate"]["targets"]
    lp_rows, lid_rows = lp["per_item"]["by_target"], li["per_item"]["by_target"]
    agree = {x: agreement(lp_rows[x], lid_rows[x]["bucket"], roles) for x in targets}
    cat = lambda f, src: np.concatenate([np.asarray(src[x][f], dtype=object) for x in targets], axis=1)
    agree_pooled = agreement({f"lp_{r}": cat(f"lp_{r}", lp_rows) for r in roles}, cat("bucket", lid_rows), roles)
    agg = {"roles": roles, "targets": targets, "layers": lp["aggregate"]["layers"],
           "by_target": {x: combined_table(lp["aggregate"]["by_target"][x], li["aggregate"]["by_target"][x], agree[x])
                         for x in targets},
           "pooled": combined_table(lp["aggregate"]["pooled"], li["aggregate"]["pooled"], agree_pooled)}
    with open_units(ctx, unit, EXP, "summary", summary_config(ctx, unit), checkpoint=False) as uf:
        uf.finalize(agg, per_item={"by_target": {x: {k: nan_to_none(v) for k, v in a.items()} for x, a in agree.items()}})


composite("interp.decoderlens", ("interp.decoderlens.logprob", "interp.decoderlens.lid", "interp.decoderlens.summary"))
