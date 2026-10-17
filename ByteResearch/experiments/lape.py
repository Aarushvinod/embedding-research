# LAPE language neurons in the ENCODER of each fine-tuned model (experiment interp_lape).
# activation_probs feeds 40k Wikipedia sentences per study language through the encoder and counts,
# inside a reducing hook on every encoder FFN act module, how often each neuron fires (> 0) per
# non-padding position, plus the activation sums that mean ablation needs, resuming per sentence chunk.
# scores, selection, overlap and correlation turn those counts into Tang-style LAPE scores, per-k
# neuron sets, hub-target overlaps and cutoff-free column-centred profile correlations.
# ablation mean-ablates the shared (X and English) and English-only neurons, plus size-matched random
# controls, on FLORES devtest, first on the GPU (log-probs, hypotheses) and then through the metrics.
# ablation_compare reports damage against the baseline and against the random control with paired
# bootstrap CIs, the normalised effect compared between ByT5 and mT5; --pilot shrinks the grid to k = 1
# and writes to the core.results.locate pilot paths.

from __future__ import annotations

import contextlib
import dataclasses

import numpy as np
import torch

from ..core.config import CONDITIONS, HUBS, STUDY11, TARGETS, DecodingConfig, InterpConfig, resolve_langs
from ..core.io import sha256_json, write_npz_atomic
from ..core.registry import composite, is_known, opt, product_units, register_stage
from ..core.results import load_array, run_array_path, run_result, save_run_array
from ..interp.readouts import RefSet, edited_provider, generate_unit, logprob_unit, plain_provider
from ..interp.scoring import batching_from, encode_pass, run_batched
from ..interp.units import FingerprintMismatch, UnitFile, fingerprint, row_chunks
from ..metrics.batching import METRIC_OPTIONS, score_keyed
from ..modeling import hooks, loading
from ..stats.bootstrap import CORPUS_SCORE_FNS as SCORE_FNS
from . import lape_stats as S

EXP = "interp_lape"
PREFIX = "interp.lape"
PILOT_DEVTEST = 64
SET_NAMES = ("none",) + S.ABLATION_SETS
API_METRICS = ("chrf_pp", "spbleu", "comet", "metricx")
READOUTS = ("lp_x", "lp_en", "lp_src")


def load_wiki(ctx, lang: str) -> list:
    # Inputs: Ctx and a study language code. Output: the cached Wikipedia sentence rows
    # [{"i", "article_id", "text"}] from corpora.wiki (imported lazily; tests replace this function).
    from ..corpora.wiki import load_wiki as fn
    return fn(ctx, lang)


def load_flores(ctx, split: str, langs=None) -> dict:
    # Inputs: Ctx, the FLORES split and language codes. Output: {code: [sentences]} from
    # corpora.flores (imported lazily; tests replace this function).
    from ..corpora.flores import load_flores as fn
    return fn(ctx, split, langs)


def score_groups(ctx, groups: dict, metric: str) -> dict:
    # Inputs: Ctx, {key: rows {id, src, hyp, ref, target}} and a metric name. Output: {key: metrics.api
    # result}, one worker call for the neural metrics (imported lazily; tests replace this function).
    from ..metrics.api import score_row_groups
    return score_row_groups(ctx, groups, metric)


def k_label(k) -> str:
    # Input: k in percent. Output: its variant label, e.g. 0.5 -> "k0.5", 1.0 -> "k1".
    return f"k{float(k):g}"


def k_of(label: str) -> float:
    # Input: a variant label starting with "k<value>" (optionally "__<set>"). Output: k as a float.
    return float(label.split("__")[0][1:])


def parse_variants(sel: dict) -> tuple:
    # Input: the grid selections. Output: (k values, set names) named by --variant tokens such as
    # "k1", "shared", "none" or "k0.5__en_only"; an unnamed axis means all of it, except that the k
    # axis defaults to k = 1 alone under --pilot (the pilot simply restricts the grid).
    ks, sets = [], []
    for part in (p for tok in str(sel.get("variant") or "").split(",") for p in tok.split("__") if p):
        if part.startswith("k"):
            ks.append(float(part[1:]))
        elif part in SET_NAMES:
            sets.append(part)
        else:
            raise ValueError(f"unknown LAPE variant token {part!r}")
    ks = sorted(set(ks)) or ([1.0] if sel.get("pilot") else [float(k) for k in InterpConfig().lape_ks])
    return ks, [s for s in SET_NAMES if s in sets] or list(SET_NAMES)


def k_units(sel: dict, smoke: bool) -> list:
    # Inputs: selections and smoke flag. Output: model x condition x k units (variant "k<k>").
    ks, _ = parse_variants(sel)
    return [u for k in ks for u in product_units(("model", "condition"), sel, smoke, variant=k_label(k))]


def ablation_units(sel: dict, smoke: bool) -> list:
    # Inputs: selections and smoke flag. Output: model x condition x variant units, variant "none"
    # (the baseline shared across k) or "k<k>__<set>".
    ks, sets = parse_variants(sel)
    variants = (["none"] if "none" in sets else []) + [f"{k_label(k)}__{s}" for k in ks for s in sets if s != "none"]
    return [u for v in variants for u in product_units(("model", "condition"), sel, smoke, variant=v)]


INPUTS = {
    "scores": lambda u: [("activation_probs", None)],
    "selection": lambda u: [("scores", None)],
    "overlap": lambda u: [("selection", u.variant)],
    "correlation": lambda u: [("scores", None)],
    "ablation": lambda u: [] if u.variant == "none" else [("selection", u.variant.split("__")[0])],
    "ablation_compare": lambda u: [("ablation", "none" if s == "none" else f"{u.variant}__{s}") for s in SET_NAMES],
}


def input_stamps(ctx, unit, measurement: str) -> dict:
    # Inputs: Ctx, a unit and a measurement name. Output: {"<input>/<variant>": [fingerprint, content hash]}
    # for each upstream LAPE result the measurement reads, so an input rerun with different results makes
    # the output stale.
    stamps = {}
    for m, v in INPUTS.get(measurement, lambda u: [])(unit):
        data = run_result(ctx, EXP, unit.run_id, m, v).load() or {}
        content = sha256_json([data.get("per_item"), data.get("aggregate")]) if data else None
        stamps[f"{m}/{v}"] = [data.get("status", {}).get("fingerprint"), content]
    return stamps


def done_fn(measurement: str):
    # Input: a measurement name. Output: is_done(ctx, unit), true when the unit's (pilot-aware) result
    # is complete and was computed from the current upstream results.
    def is_done(ctx, unit):
        # Inputs: Ctx and a unit. Output: whether the result is complete and its input stamps (in status,
        # or in config for the resumable ablation) are current.
        data = run_result(ctx, EXP, unit.run_id, measurement, unit.variant).load() or {}
        status = data.get("status", {})
        stored = status.get("inputs", data.get("config", {}).get("inputs", {}))
        return bool(status.get("complete")) and stored == input_stamps(ctx, unit, measurement)
    return is_done


def known(*names) -> tuple:
    # Inputs: candidate requirement names owned by other stages. Output: those registered at import
    # time (corpora and training are discovered before experiments).
    return tuple(n for n in names if is_known(n))


def study_langs(ctx) -> tuple:
    # Input: Ctx. Output: the study languages counted for LAPE (all 11; the smoke targets plus hubs
    # under --smoke).
    return tuple(resolve_langs("study11", STUDY11, ctx.smoke))


def checkpoint_info(ctx, unit):
    # Inputs: Ctx and a model x condition unit. Output: {"path", "step", "lr"} of the best checkpoint's
    # meta, or None when it has no meta.
    info = loading.checkpoint_info(ctx, unit.model, unit.condition)
    return None if info["meta_sha256"] is None else {k: info[k] for k in ("path", "step", "lr")}


def write_result(ctx, unit, measurement: str, per_item, aggregate, config, extra=None, variant=None) -> dict:
    # Inputs: Ctx, unit, measurement, per_item, aggregate, config, extra top-level fields and the
    # variant. Writes the complete result with the checkpoint in provenance and the upstream input
    # stamps in status. Output: the envelope.
    rf = run_result(ctx, EXP, unit.run_id, measurement, variant, unit)
    extra = dict(extra or {})
    extra["provenance"] = {"checkpoint": checkpoint_info(ctx, unit)}
    extra["status"] = {**extra.get("status", {}), "fingerprint": fingerprint(config),
                       "inputs": input_stamps(ctx, unit, measurement)}
    return rf.write(per_item, aggregate, config=config, complete=True, extra=extra)


def read_result(ctx, rid: str, measurement: str, variant=None) -> dict:
    # Inputs: Ctx, run id, measurement and variant. Output: the complete result dict; raises
    # FileNotFoundError when it is missing or incomplete.
    data = run_result(ctx, EXP, rid, measurement, variant).load()
    if not data or not data.get("status", {}).get("complete"):
        raise FileNotFoundError(f"{EXP}/{rid}/{measurement} {variant or ''} (pilot={ctx.pilot}) is not complete")
    return data


def read_arrays(ctx, rid: str, measurement: str) -> dict:
    # Inputs: Ctx, run id and a measurement whose result references an .npz. Output: {name: array}.
    with load_array(ctx, read_result(ctx, rid, measurement)["arrays"]["npz"]) as z:
        return {k: z[k] for k in z.files}


def act_reducer(holder: dict):
    # Input: a dict whose "mask" entry holds the current batch's attention mask [B, T]. Output:
    # reduce(act [B, T, d_ff]) -> (fired counts [d_ff], activation sums [d_ff]) over non-padding
    # positions, computed on the act's device so no per-position values leave the hook.
    def reduce(t):
        # Input: one act output. Output: (fired count, activation sum) per unit, float32.
        m = holder["mask"].to(device=t.device, dtype=torch.float32).unsqueeze(-1)
        a = t.float()
        return ((a > 0).float() * m).sum(dim=(0, 1)), (a * m).sum(dim=(0, 1))
    return reduce


def count_activations(trained, texts, max_chars: int, factor: int, batching) -> dict:
    # Inputs: Trained, encoder input texts, character cap, token factor and Batching. Runs the encoder
    # over every text in token-budget batches with a reducing hook on each encoder act module.
    # Output: {"fired": int64 [n_layers * d_ff], "act_sum": float64 [n_layers * d_ff], "positions"}.
    n_layers, d_ff = trained.n_enc, trained.d_ff
    fired = torch.zeros(n_layers, d_ff, dtype=torch.float64, device=trained.device)
    sums = torch.zeros_like(fired)
    holder, total = {}, [0]
    mods = hooks.act_modules(trained.model, "enc")

    def step(enc, brows):
        # Inputs: an encoder batch and its rows. Accumulates the batch's counts. Output: per-row None.
        holder["mask"] = enc["attention_mask"].to(trained.device)
        with hooks.record(mods, reduce=act_reducer(holder)) as rec:
            encode_pass(trained, enc)
        fired.add_(torch.stack([rec.out[l][0] for l in range(n_layers)]).double())
        sums.add_(torch.stack([rec.out[l][1] for l in range(n_layers)]).double())
        total[0] += int(holder["mask"].sum())
        return [None] * len(brows)

    run_batched(trained, list(texts), list(range(len(texts))), max_chars, factor, batching, step)
    return {"fired": fired.round().long().reshape(-1).cpu().numpy(), "act_sum": sums.reshape(-1).cpu().numpy(),
            "positions": total[0]}


def wiki_texts(ctx, lang: str) -> list:
    # Inputs: Ctx and a language code. Output: that language's Wikipedia sentences (a fixed seeded
    # subset of InterpConfig.pilot_wiki under --pilot).
    texts = [r["text"] for r in load_wiki(ctx, lang)]
    return [texts[i] for i in ctx.rows(len(texts), k=ctx.config(InterpConfig).pilot_wiki)]


def fresh_counts(langs, n_units: int) -> dict:
    # Inputs: language codes and the neuron count. Output: zeroed counters for activation_probs.
    n = len(langs)
    return {"fired": np.zeros((n, n_units), np.int64), "act_sum": np.zeros((n, n_units), np.float64),
            "positions": np.zeros(n, np.int64), "n_sentences": np.zeros(n, np.int64)}


def load_partial(path, fp: str, force: bool):
    # Inputs: the partial .npz path, the config fingerprint and the force flag. Output: (counters
    # dict, done chunk keys) from a matching partial file, or (None, []) when absent or forced;
    # a different fingerprint raises FingerprintMismatch.
    if force or not path.exists():
        return None, []
    with np.load(path, allow_pickle=False) as z:
        if str(z["fingerprint"]) != fp:
            raise FingerprintMismatch(f"{path}: partial counts were made under another config; use --force")
        state = {k: z[k] for k in ("fired", "act_sum", "positions", "n_sentences", "n_layers", "d_ff")}
        return state, [str(d) for d in z["done"]]


def chunk_plan(corpus: dict, size: int) -> list:
    # Inputs: {lang: texts} and the chunk size. Output: [(key "lang|c", lang, start, stop)] in order.
    return [(f"{lang}|{c}", lang, s, min(s + size, len(texts)))
            for lang, texts in corpus.items() for c, s in enumerate(range(0, len(texts), size))]


def add_chunk(state: dict, j: int, counts: dict, n_sent: int) -> None:
    # Inputs: the counters, the language index, one chunk's counts and its sentence count. Adds the
    # chunk's counts into the counters in place.
    state["fired"][j] += counts["fired"]
    state["act_sum"][j] += counts["act_sum"]
    state["positions"][j] += counts["positions"]
    state["n_sentences"][j] += n_sent


def count_chunks(ctx, unit, corpus, todo, state, done, path, fp) -> dict:
    # Inputs: Ctx, unit, {lang: texts}, the remaining chunk plan, counters (None to start fresh), the
    # done keys, the partial path and fingerprint. Loads the model, counts every remaining chunk and
    # checkpoints the counters plus done keys to the partial .npz after each. Output: the counters.
    cfg, dec = ctx.config(InterpConfig), ctx.config(DecodingConfig)
    langs = list(corpus)
    rf = run_result(ctx, EXP, unit.run_id, "activation_probs", unit=unit)
    with loading.loaded(ctx, unit.model, unit.condition) as trained:
        if state is None:
            state = {**fresh_counts(langs, trained.n_enc * trained.d_ff), "n_layers": np.array(trained.n_enc),
                     "d_ff": np.array(trained.d_ff)}
        for key, lang, start, stop in todo:
            counts = count_activations(trained, corpus[lang][start:stop], cfg.max_chars, dec.token_factor,
                                       batching_from(cfg))
            add_chunk(state, langs.index(lang), counts, stop - start)
            done.append(key)
            write_npz_atomic(path, **state, done=np.array(done, dtype=str), fingerprint=np.array(fp))
            rf.write({}, {}, config={}, complete=False, extra={"status": {"units_done": done, "fingerprint": fp}})
    return state


@register_stage(f"{PREFIX}.activation_probs", grid=("model", "condition"), is_done=done_fn("activation_probs"),
                requires=known("train", "data.wiki"), order=700,
                options=(opt("--lape-chunk", type=int, default=2000, help="LAPE: sentences per resume chunk"),
                         opt("--variant", default=None, help="variant tokens, comma list")))
def activation_probs(ctx, unit) -> None:
    # Inputs: Ctx and a model x condition unit. Counts per-language fired positions, activation sums
    # and non-padding positions for every encoder FFN neuron over the Wikipedia corpus, resumable per
    # (language, chunk). Writes activation_probs.npz and a JSON with per-language mean probability.
    cfg, dec = ctx.config(InterpConfig), ctx.config(DecodingConfig)
    corpus = {lang: wiki_texts(ctx, lang) for lang in study_langs(ctx)}
    config = {"langs": list(corpus), "n_sentences": {l: len(t) for l, t in corpus.items()},
              "chunk": int(ctx.opt("lape_chunk", 2000)), "max_chars": cfg.max_chars,
              "token_factor": dec.token_factor, "corpus_sha256": sha256_json(corpus), "pilot": ctx.pilot}
    fp = fingerprint(config)
    path = run_array_path(ctx, EXP, unit.run_id, "activation_probs.partial.npz")
    state, done = load_partial(path, fp, ctx.force)
    todo = [c for c in chunk_plan(corpus, config["chunk"]) if c[0] not in set(done)]
    if todo or state is None:
        state = count_chunks(ctx, unit, corpus, todo, state, done, path, fp)
    langs = list(corpus)
    ref = save_run_array(ctx, EXP, unit.run_id, "activation_probs.npz", langs=np.array(langs, dtype=str),
                     **{k: state[k] for k in ("fired", "act_sum", "positions", "n_sentences", "n_layers", "d_ff")})
    P = S.activation_probs(state["fired"], state["positions"])
    per_item = {"lang": langs, "positions": state["positions"], "n_sentences": state["n_sentences"],
                "mean_prob": P.mean(axis=0)}
    aggregate = {"n_layers": int(state["n_layers"]), "d_ff": int(state["d_ff"]), "n_units": int(P.shape[0]),
                 "mean_prob": dict(zip(langs, P.mean(axis=0).tolist()))}
    write_result(ctx, unit, "activation_probs", per_item, aggregate, config,
                 extra={"arrays": {"npz": ref}, "status": {"units_done": done}})
    path.unlink(missing_ok=True)


@register_stage(f"{PREFIX}.scores", grid=("model", "condition"), is_done=done_fn("scores"),
                requires=(f"{PREFIX}.activation_probs",), order=710)
def scores(ctx, unit) -> None:
    # Inputs: Ctx and a model x condition unit. Computes the activation-probability matrix P, LAPE,
    # each neuron's max probability, Tang's percentile threshold and the passes-filter mask. Writes
    # scores.npz and a summary JSON.
    cfg = ctx.config(InterpConfig)
    c = read_arrays(ctx, unit.run_id, "activation_probs")
    langs, n_layers, d_ff = [str(x) for x in c["langs"]], int(c["n_layers"]), int(c["d_ff"])
    P = S.activation_probs(c["fired"], c["positions"])
    lape, thr = S.lape_entropy(P), S.tang_threshold(P, cfg.lape_percentile)
    passes, max_prob = S.passes_filter(P, thr), P.max(axis=1)
    ref = save_run_array(ctx, EXP, unit.run_id, "scores.npz", P=P, lape=lape, max_prob=max_prob,
                     passes=passes, threshold=np.array(thr), langs=np.array(langs, dtype=str),
                     n_layers=np.array(n_layers), d_ff=np.array(d_ff))
    best = np.bincount(P[passes].argmax(axis=1), minlength=len(langs)) if passes.any() else np.zeros(len(langs))
    finite = lape[np.isfinite(lape) & passes]
    aggregate = {"threshold": thr, "percentile": cfg.lape_percentile, "n_units": int(P.shape[0]),
                 "n_layers": n_layers, "d_ff": d_ff, "n_pass": int(passes.sum()),
                 "n_pass_per_layer": S.layer_counts(np.flatnonzero(passes), d_ff, n_layers),
                 "n_all_zero": int(np.isinf(lape).sum()), "best_lang_among_passing": dict(zip(langs, best.tolist())),
                 "lape_quantiles_passing": dict(zip(("q05", "q25", "q50", "q75", "q95"),
                                                    np.quantile(finite, [.05, .25, .5, .75, .95]).tolist()
                                                    if finite.size else [None] * 5))}
    per_item = {"lang": langs, "mean_prob": P.mean(axis=0), "frac_above_threshold": (P > thr).mean(axis=0)}
    write_result(ctx, unit, "scores", per_item, aggregate, {"percentile": cfg.lape_percentile, "langs": langs},
                 extra={"arrays": {"npz": ref}})


def load_selection(ctx, rid: str, label: str) -> dict:
    # Inputs: Ctx, run id and a k label. Output: {"sets": {lang: ids}, "langs", "n_selected", "d_ff",
    # "n_layers"} read back from that k's selection result.
    data = read_result(ctx, rid, "selection", label)
    pi, agg = data["per_item"], data["aggregate"]
    langs = agg["langs"]
    ids = np.asarray(pi["neuron"], dtype=np.int64)
    assign = np.array([[l in row for l in langs] for row in pi["langs"]], dtype=bool)
    return {"sets": S.language_sets(ids, assign, langs), "langs": langs, "n_selected": int(ids.size),
            "d_ff": agg["d_ff"], "n_layers": agg["n_layers"]}


@register_stage(f"{PREFIX}.selection", grid=k_units, is_done=done_fn("selection"),
                requires=(f"{PREFIX}.scores",), order=720)
def selection(ctx, unit) -> None:
    # Inputs: Ctx and a model x condition x k unit. Selects the lowest-k% LAPE neurons among those
    # passing the threshold and assigns each to every language above it. Writes the neurons with
    # their languages and the per-language, per-layer and multi-language counts.
    k = k_of(unit.variant)
    sc = read_arrays(ctx, unit.run_id, "scores")
    langs, n_layers, d_ff = [str(x) for x in sc["langs"]], int(sc["n_layers"]), int(sc["d_ff"])
    thr = float(sc["threshold"])
    ids = S.select_neurons(sc["lape"], sc["passes"], k)
    assign = S.assign_languages(sc["P"], ids, thr)
    sets = S.language_sets(ids, assign, langs)
    per_item = {"neuron": ids, "layer": ids // d_ff, "unit": ids % d_ff, "lape": sc["lape"][ids],
                "langs": [[langs[j] for j in np.flatnonzero(row)] for row in assign]}
    n_target = S.n_keep(sc["lape"].size, k)
    aggregate = {"k": k, "langs": langs, "n_layers": n_layers, "d_ff": d_ff, "threshold": thr,
                 "n_units": int(sc["lape"].size), "n_target": n_target, "n_selected": int(ids.size),
                 "shortfall": n_target - int(ids.size), "per_lang": {l: int(s.size) for l, s in sets.items()},
                 "per_layer": S.layer_counts(ids, d_ff, n_layers),
                 "per_lang_layer": {l: S.layer_counts(s, d_ff, n_layers) for l, s in sets.items()},
                 "n_multi": int((assign.sum(axis=1) > 1).sum()),
                 "n_langs_hist": np.bincount(assign.sum(axis=1), minlength=len(langs) + 1)}
    write_result(ctx, unit, "selection", per_item, aggregate, {"k": k, "threshold": thr, "langs": langs},
                 variant=unit.variant)


@register_stage(f"{PREFIX}.overlap", grid=k_units, is_done=done_fn("overlap"),
                requires=(f"{PREFIX}.selection",), order=730)
def overlap(ctx, unit) -> None:
    # Inputs: Ctx and a model x condition x k unit. For each hub, writes the overlap |A&B|/|A| with
    # every target set, its mean over all targets and over different-script targets, and chance.
    sel = load_selection(ctx, unit.run_id, unit.variant)
    targets = [t for t in TARGETS if t in sel["langs"]]
    hubs = [h for h in HUBS if h in sel["langs"]]
    summary = {h: S.overlap_summary(sel["sets"], h, targets, sel["n_selected"]) for h in hubs}
    per_item = {"target": targets, **{h: [summary[h]["per_target"][t]["overlap"] for t in targets] for h in hubs},
                "chance": [summary[hubs[0]]["per_target"][t]["chance"] for t in targets] if hubs else []}
    write_result(ctx, unit, "overlap", per_item, {"n_selected": sel["n_selected"], "hubs": summary},
                 {"k": k_of(unit.variant), "targets": targets, "hubs": hubs}, variant=unit.variant)


@register_stage(f"{PREFIX}.correlation", grid=("model", "condition"), is_done=done_fn("correlation"),
                requires=(f"{PREFIX}.scores",), order=740)
def correlation(ctx, unit) -> None:
    # Inputs: Ctx and a model x condition unit. Writes cutoff-free Spearman and cosine matrices of the
    # column-centred activation-probability profiles per layer and pooled, with hub summaries.
    sc = read_arrays(ctx, unit.run_id, "scores")
    langs, n_layers, d_ff = [str(x) for x in sc["langs"]], int(sc["n_layers"]), int(sc["d_ff"])
    targets, hubs = [t for t in TARGETS if t in langs], [h for h in HUBS if h in langs]
    summary = S.correlation_summary(sc["P"], langs, hubs, targets, n_layers, d_ff)
    pooled = summary.pop("pooled")
    write_result(ctx, unit, "correlation", {"layer": summary}, {"langs": langs, "pooled": pooled},
                 {"langs": langs, "centering": "per-neuron mean across languages"})


def ablation_edits(model, ids, values, d_ff: int) -> dict:
    # Inputs: the model, flattened neuron ids, one replacement value per id and d_ff. Output:
    # {encoder act module: set_units_fn} replacing each neuron's act output with its value at every
    # position (empty for an empty set).
    ids, values = np.asarray(ids, dtype=np.int64), np.asarray(values, dtype=np.float64)
    layers = ids // d_ff
    return {hooks.ffn_act(model, "enc", int(l)): hooks.set_units_fn((ids[layers == l] % d_ff).tolist(),
                                                                    values[layers == l].tolist())
            for l in np.unique(layers)}


def provider_for(trained, spec):
    # Inputs: Trained and (ids, values) or None. Output: the plain provider for no/empty ablation,
    # otherwise an edited provider applying the mean ablation on the encoder act modules.
    if spec is None or len(spec[0]) == 0:
        return plain_provider(trained)
    return edited_provider(trained, ablation_edits(trained.model, spec[0], spec[1], trained.d_ff))


def ablation_plan(ctx, rid: str, variant: str, targets) -> dict:
    # Inputs: Ctx, run id, variant ("none" or "k<k>__<set>") and target codes. Output: {X: (ids,
    # replacement values)} with values = the unweighted mean over languages other than X and English
    # of each language's mean activation; empty for the baseline.
    if variant == "none":
        return {}
    label, set_name = variant.split("__")
    sel, counts = load_selection(ctx, rid, label), read_arrays(ctx, rid, "activation_probs")
    langs = [str(x) for x in counts["langs"]]
    plan = {}
    for x in targets:
        ids = S.ablation_sets(sel["sets"], x, label, ctx.seed, sel["d_ff"], sel["n_layers"])[set_name]
        repl = S.mean_replacement(counts["act_sum"], counts["positions"], langs, (x, "en"))
        plan[x] = (ids, repl[ids])
    return plan


def gpu_chunk(trained, provider, flores: dict, rows, x: str, hub: str, cfg, dec) -> dict:
    # Inputs: Trained, a provider, FLORES {code: sentences}, row indices, target X, hub and the
    # Interp/Decoding configs. Output: per-char log-probs of the X, English and source references
    # (all under the X tag) and the generated X hypotheses for those rows.
    refs = {"x": RefSet(x, flores[x]), "en": RefSet(x, flores["en"])}
    if hub != "en":
        refs["src"] = RefSet(x, flores[hub])
    lp = logprob_unit(trained, provider, None, flores[hub], refs, rows, batching_from(cfg), cfg.max_chars,
                      dec.token_factor, cfg.logprob_chunk_rows)
    hyps = generate_unit(trained, provider, None, flores[hub], x, rows, dec, batching_from(cfg, generation=True))
    return {"rows": list(rows), "lp_x": lp["x"].per_char.tolist(), "lp_en": lp["en"].per_char.tolist(),
            "lp_src": lp.get("src", lp["en"]).per_char.tolist(), "hyp": hyps[x]}


def gpu_phase(ctx, unit, uf: UnitFile, plan: dict, flores: dict, chunks: list, targets, hub: str) -> None:
    # Inputs: Ctx, unit, the open UnitFile, the ablation plan, FLORES, row chunks, targets and hub.
    # Runs every unfinished (X, chunk) under the ablation and stores it; loads the model only when
    # work remains and releases it before returning.
    todo = [(x, c) for x in targets for c in range(len(chunks)) if not uf.has(("gpu", x, c))]
    if not todo:
        return
    cfg, dec = ctx.config(InterpConfig), ctx.config(DecodingConfig)
    with loading.loaded(ctx, unit.model, unit.condition) as trained:
        for x, c in todo:
            uf.put(("gpu", x, c), gpu_chunk(trained, provider_for(trained, plan.get(x)), flores, chunks[c], x, hub,
                                            cfg, dec))
    uf.flush(force=True)


def gather_gpu(uf: UnitFile, targets, n_chunks: int) -> dict:
    # Inputs: the UnitFile, targets and chunk count. Output: columnar {"id", "target", "row", readouts,
    # "hyp"} over targets in order and rows in chunk order.
    cols = {k: [] for k in ("id", "target", "row", *READOUTS, "hyp")}
    for x in targets:
        for c in range(n_chunks):
            part = uf.get(("gpu", x, c))
            cols["target"] += [x] * len(part["rows"])
            cols["id"] += [f"{x}:{r}" for r in part["rows"]]
            cols["row"] += part["rows"]
            for key in (*READOUTS, "hyp"):
                cols[key] += part[key]
    return cols


def metric_rows(cols: dict, flores: dict, hub: str, ref_lang=None) -> list:
    # Inputs: the gathered columns, FLORES, hub and the reference language (None = each row's target).
    # Output: metrics.api input rows {id, src, hyp, ref, target}.
    return [{"id": i, "src": flores[hub][r], "hyp": h, "ref": flores[ref_lang or x][r], "target": ref_lang or x}
            for i, x, r, h in zip(cols["id"], cols["target"], cols["row"], cols["hyp"])]


def metric_jobs(uf: UnitFile, cols: dict, flores: dict, hub: str) -> list:
    # Inputs: the UnitFile, gathered columns, FLORES and hub. Output: the missing (unit key, metric,
    # rows) jobs: every metric against the X references, and chrF++ against the English and source
    # references for the fallback language.
    jobs = [(("metric", m), m, metric_rows(cols, flores, hub)) for m in API_METRICS]
    jobs += [(("fallback", ref), "chrf_pp", metric_rows(cols, flores, hub, ref)) for ref in dict.fromkeys(("en", hub))]
    return [job for job in jobs if not uf.has(job[0])]


def store_scores(uf: UnitFile, key, res: dict) -> None:
    # Inputs: the UnitFile, a metric or fallback unit key and its metrics.api result. Stores the
    # per-row values (and statistics for metric units) and flushes.
    fields = {"values": np.asarray(res["values"], dtype=np.float64).tolist()}
    if key[0] == "metric":
        fields["stats"] = None if res.get("stats") is None else np.asarray(res["stats"]).tolist()
    uf.put(key, fields)
    uf.flush(force=True)


def metric_phase(ctx, states: list) -> None:
    # Inputs: Ctx and per-unit states {uf, cols, flores, hub}. Scores every unit's missing metric and
    # fallback jobs; jobs of one metric across all units share its worker calls.
    jobs = {(i, key): (m, rows) for i, st in enumerate(states)
            for key, m, rows in metric_jobs(st["uf"], st["cols"], st["flores"], st["hub"])}
    for m in dict.fromkeys(m for m, _ in jobs.values()):
        groups = {k: rows for k, (mm, rows) in jobs.items() if mm == m}
        score_keyed(ctx, groups, m, lambda k, res: store_scores(states[k[0]]["uf"], k[1], res), score_groups)


def fallback_names(hub: str) -> list:
    # Input: the hub. Output: the candidate fallback labels in tie-break order: x, en and, outside C1
    # (where the source is English), src.
    return ["x", "en"] + (["src"] if hub != "en" else [])


def argmax_labels(scores: dict, names) -> list:
    # Inputs: {label: per-row scores} and the candidate labels in tie-break order. Output: the per-row
    # label of the highest score (NaN or None never wins; ties go to the earlier label).
    stacked = np.array([scores[n] for n in names], dtype=np.float64)
    return [names[i] for i in np.where(np.isnan(stacked), -np.inf, stacked).argmax(axis=0)]


def fallback_labels(uf: UnitFile, cols: dict, hub: str) -> dict:
    # Inputs: the UnitFile, gathered columns and hub. Output: per-row chrF++ against the X, English
    # and source references, the chrF++ argmax label and the per-char log-prob argmax label
    # ("label_lp"), each "x", "en" or "src".
    chrf = {"x": uf.get(("metric", "chrf_pp"))["values"], "en": uf.get(("fallback", "en"))["values"],
            "src": uf.get(("fallback", hub))["values"]}
    names = fallback_names(hub)
    return {"chrf_x": chrf["x"], "chrf_en": chrf["en"], "chrf_src": chrf["src"],
            "label": argmax_labels(chrf, names),
            "label_lp": argmax_labels({r[3:]: cols[r] for r in READOUTS}, names)}


def metric_score(m: str, values, stats) -> float:
    # Inputs: metric name, per-row values and per-row stats (None for neural metrics). Output: the
    # corpus score from summed stats for chrF++/spBLEU, otherwise the mean value.
    if m in SCORE_FNS and stats is not None and len(stats):
        return float(SCORE_FNS[m]()(np.asarray(stats, dtype=np.float64).sum(axis=0))[0])
    return float(np.nanmean(np.asarray(values, dtype=np.float64))) if len(values) else None


def label_fractions(labels) -> dict:
    # Input: per-row fallback labels. Output: the fraction of rows per label x, en and src (None when
    # there are no rows).
    return {n: float(np.mean([l == n for l in labels])) if labels else None for n in ("x", "en", "src")}


def scope_summary(per_item: dict, mask) -> dict:
    # Inputs: the final columnar per_item and a boolean row mask. Output: the masked rows' row count,
    # mean log-prob readouts, metric scores and the chrF++ and log-prob fallback-label fractions.
    pick = np.flatnonzero(mask)
    sub = lambda col: [col[i] for i in pick]
    stats = per_item["stats"]
    return {"n": int(pick.size), **{r: S.mean_or_none(sub(per_item[r])) for r in READOUTS},
            **{m: metric_score(m, sub(v), sub(stats[m]) if stats.get(m) is not None else None)
               for m, v in per_item["metrics"].items()},
            "fallback": label_fractions(sub(per_item["fallback"]["label"])),
            "fallback_lp": label_fractions(sub(per_item["fallback"]["label_lp"]))}


def ablation_summary(per_item: dict, targets) -> dict:
    # Inputs: the final columnar per_item and targets. Output: scope_summary per target and pooled.
    tgt = np.array(per_item["target"])
    return {**{x: scope_summary(per_item, tgt == x) for x in targets},
            "pooled": scope_summary(per_item, np.ones(tgt.size, dtype=bool))}


def finalize_ablation(uf: UnitFile, cols: dict, plan: dict, targets, hub: str) -> None:
    # Inputs: the UnitFile, gathered columns, plan, targets and hub. Builds the columnar per_item
    # (readouts, hypotheses, metric values and stats, fallback, the ablated neuron sets) and the
    # aggregate, and marks the result complete.
    metrics = {m: uf.get(("metric", m)) for m in API_METRICS}
    per_item = {**cols, "metrics": {m: v["values"] for m, v in metrics.items()},
                "stats": {m: v["stats"] for m, v in metrics.items() if m in SCORE_FNS},
                "fallback": fallback_labels(uf, cols, hub),
                "sets": {x: [int(i) for i in plan[x][0]] for x in plan}}
    uf.finalize(ablation_summary(per_item, targets), per_item=per_item)


def ablation_targets(ctx, unit) -> list:
    # Inputs: Ctx and a model x condition x variant unit. Output: the targets the ablation covers: the
    # condition's targets within the --target selection.
    cond = CONDITIONS[unit.condition]
    return [t for t in resolve_langs(ctx.sel.get("target"), TARGETS, ctx.smoke) if t in cond.targets]


def ablation_done(ctx, unit) -> bool:
    # Inputs: Ctx and a model x condition x variant unit. Output: whether its (pilot-aware) ablation
    # result is complete, covers exactly the current targets (so a run over a --target subset never
    # stands in for the full one) and was computed from the current selection.
    data = run_result(ctx, EXP, unit.run_id, "ablation", unit.variant).load() or {}
    complete, cfg = bool(data.get("status", {}).get("complete")), data.get("config", {})
    current = cfg.get("targets") == ablation_targets(ctx, unit)
    return complete and current and cfg.get("inputs", {}) == input_stamps(ctx, unit, "ablation")


def ablation_state(ctx, unit, stack) -> dict:
    # Inputs: Ctx, a model x condition x variant unit and an ExitStack. Output: the unit's state {unit,
    # uf, plan, flores, chunks, targets, hub} with its resumable result opened on the stack.
    cfg, dec = ctx.config(InterpConfig), ctx.config(DecodingConfig)
    cond = CONDITIONS[unit.condition]
    targets = ablation_targets(ctx, unit)
    flores = load_flores(ctx, "devtest", list(dict.fromkeys([*targets, cond.hub, "en"])))
    rows = ctx.rows(len(flores[cond.hub]), k=cfg.pilot_rows if ctx.smoke else PILOT_DEVTEST)
    chunks = row_chunks(rows, int(ctx.opt("lape_rows_chunk", 128)))
    plan = ablation_plan(ctx, unit.run_id, unit.variant, targets)
    config = {"variant": unit.variant, "hub": cond.hub, "targets": targets, "rows": rows, "chunk": len(chunks[0]),
              "decoding": dataclasses.asdict(dec), "max_chars": cfg.max_chars, "seed": ctx.seed, "metrics": API_METRICS,
              "sets_sha256": sha256_json({x: [plan[x][0].tolist(), plan[x][1].tolist()] for x in plan}),
              "inputs": input_stamps(ctx, unit, "ablation")}
    rf = run_result(ctx, EXP, unit.run_id, "ablation", unit.variant, unit)
    uf = stack.enter_context(UnitFile.open(rf, {"config": config}, fingerprint(config), force=ctx.force,
                                           flush_seconds=cfg.flush_seconds))
    return {"unit": unit, "uf": uf, "plan": plan, "flores": flores, "chunks": chunks, "targets": targets,
            "hub": cond.hub}


def ablation_units_run(ctx, units) -> None:
    # Inputs: Ctx and model x condition x variant units. Mean-ablates each unit's neuron set on FLORES
    # devtest in the model's trained direction: a GPU phase per unit (log-probs, hypotheses) resumable
    # per (X, chunk), then, with the models freed, one metrics phase over all units resumable per
    # (unit, metric), and finally each unit's result.
    with contextlib.ExitStack() as stack:
        states = [ablation_state(ctx, u, stack) for u in units]
        for st in states:
            gpu_phase(ctx, st["unit"], st["uf"], st["plan"], st["flores"], st["chunks"], st["targets"], st["hub"])
            st["cols"] = gather_gpu(st["uf"], st["targets"], len(st["chunks"]))
        metric_phase(ctx, states)
        for st in states:
            finalize_ablation(st["uf"], st["cols"], st["plan"], st["targets"], st["hub"])


@register_stage(f"{PREFIX}.ablation", grid=ablation_units, is_done=ablation_done,
                requires=(f"{PREFIX}.selection", f"{PREFIX}.activation_probs", *known("data.flores")), order=750,
                batch=ablation_units_run,
                options=(opt("--lape-rows-chunk", type=int, default=128, help="LAPE: devtest rows per resume chunk"),
                         *METRIC_OPTIONS))
def ablation(ctx, unit) -> None:
    # Inputs: Ctx and a model x condition x variant unit. Runs that unit's ablation (ablation_units_run
    # for one unit).
    ablation_units_run(ctx, [unit])


def compare_columns(per_item: dict) -> dict:
    # Input: an ablation per_item. Output: {readout/metric name: per-row values} plus "stats:<m>"
    # entries for the corpus metrics and the fallback-to-English indicators by chrF++ argmax
    # ("fallback_en") and by per-char log-prob argmax ("fallback_en_lp"; in C1 lp_src equals lp_en, so
    # the tie goes to en).
    cols = {r: per_item[r] for r in READOUTS}
    cols.update(per_item["metrics"])
    cols.update({f"stats:{m}": s for m, s in per_item["stats"].items() if s is not None})
    cols["fallback_en"] = [float(l == "en") for l in per_item["fallback"]["label"]]
    lp_labels = argmax_labels({r[3:]: per_item[r] for r in READOUTS}, ["x", "en", "src"])
    cols["fallback_en_lp"] = [float(l == "en") for l in lp_labels]
    return cols


def scope_arrays(cols: dict, targets_col, targets, name: str):
    # Inputs: one set's columns, the per-row target column, the targets and a column name. Output:
    # {scope: array} with [n] (or [n, k]) per target and [T, n] (or [T, n, k]) for "pooled".
    tgt = np.array(targets_col)
    per = {x: np.asarray(cols[name], dtype=np.float64)[tgt == x] for x in targets}
    return {**per, "pooled": np.stack([per[x] for x in targets])}


def compare_metric(by_set: dict, targets_col, targets, name: str, ctx) -> dict:
    # Inputs: {set: columns}, the target column, targets, a metric name and Ctx. Output: per scope the
    # damage_compare result (corpus for chrF++/spBLEU via their stats, mean otherwise).
    cfg = ctx.config(InterpConfig)
    corpus = name in SCORE_FNS and all(f"stats:{name}" in c for c in by_set.values())
    col = f"stats:{name}" if corpus else name
    arrs = {s: scope_arrays(c, targets_col, targets, col) for s, c in by_set.items()}
    return {scope: S.damage_compare({s: arrs[s][scope] for s in by_set}, "corpus" if corpus else "mean",
                                    SCORE_FNS[name]() if corpus else None, cfg.n_boot, ctx.seed, cfg.alpha)
            for scope in (*targets, "pooled")}


@register_stage(f"{PREFIX}.ablation_compare", grid=k_units, is_done=done_fn("ablation_compare"),
                requires=(f"{PREFIX}.ablation",), order=760)
def ablation_compare(ctx, unit) -> None:
    # Inputs: Ctx and a model x condition x k unit. Writes per-sentence damage m(set) - m(none) and,
    # per target and pooled, the aggregated damage and its difference and ratio to the size-matched
    # random set, with paired bootstrap CIs.
    variants = {s: ("none" if s == "none" else f"{unit.variant}__{s}") for s in SET_NAMES}
    data = {s: read_result(ctx, unit.run_id, "ablation", v)["per_item"] for s, v in variants.items()}
    base = data["none"]
    if any(d["id"] != base["id"] for d in data.values()):
        raise ValueError("ablation results are not aligned on the same rows; rerun them with one config")
    targets = list(dict.fromkeys(base["target"]))
    by_set = {s: compare_columns(d) for s, d in data.items()}
    names = [n for n in by_set["none"] if not n.startswith("stats:")]
    damage = {s: {n: (np.asarray(by_set[s][n], dtype=np.float64) - np.asarray(by_set["none"][n], dtype=np.float64))
                  for n in names} for s in S.ABLATION_SETS}
    aggregate = {n: compare_metric(by_set, base["target"], targets, n, ctx) for n in names}
    per_item = {"id": base["id"], "target": base["target"], "damage": damage,
                "set_sizes": {s: {x: len(data[s]["sets"].get(x, [])) for x in targets} for s in S.ABLATION_SETS}}
    write_result(ctx, unit, "ablation_compare", per_item, {"targets": targets, "metrics": aggregate},
                 {"k": k_of(unit.variant), "variants": variants, "n_boot": ctx.config(InterpConfig).n_boot,
                  "seed": ctx.seed}, variant=unit.variant)


composite(PREFIX, [f"{PREFIX}.{m}" for m in ("activation_probs", "scores", "selection", "overlap", "correlation",
                                            "ablation", "ablation_compare")])
