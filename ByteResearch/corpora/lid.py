# Language identification with GlotLID, shared by the NLLB filters and the DecoderLens experiment.
# load_glotlid downloads cis-lmu/glotlid model.bin into ROOT/data/models/glotlid, records the resolved
# revision in its manifest, verifies its SHA-256 and loads it with fasttext.
# predict calls model.f.predict directly, because fasttext's Python wrapper breaks under numpy 2.
# load_aliases derives each language's accepted labels from ISO 639-3 macrolanguage membership, the script
# and the labels present in the model (corpora.iso639), for the study languages or the yield-only
# candidates, and lid_record describes that rule for configs.
# A text counts as language L when the probabilities of L's labels among its top-k predictions sum to at
# least the threshold: labels_prob computes that sum, and lid_probs and lid_passes apply it to raw texts.

from __future__ import annotations

import functools

from ..core.config import CANDIDATE_LANGUAGES, LANGUAGES, LID_EXTRA_CODES, LID_ONLY_CODES, PINS, DataConfig
from ..core.io import sha256_json
from .assets import asset_dir, build_hf_asset, ensure_hf_asset
from .iso639 import LABEL_PREFIX, derive_aliases, iso_table_path, parse_macrolanguages
from .manifest import verified_path

LID_RULE = "sum of GlotLID top-k probabilities over the language's labels >= min_prob"


def build_glotlid(ctx) -> dict:
    # Input: Ctx. Downloads GlotLID model.bin at the resolved (or pinned) revision. Output: the manifest.
    pin = PINS["glotlid"]
    return build_hf_asset(ctx, "glotlid", pin["repo"], files=[pin["file"]], revision=pin.get("revision"))


def glotlid_path(ctx):
    # Input: Ctx. Output: the verified local model.bin path, downloading it first when missing.
    pin = PINS["glotlid"]
    ensure_hf_asset(ctx, "glotlid", pin["repo"], files=[pin["file"]], revision=pin.get("revision"))
    return verified_path(asset_dir(ctx, "glotlid"), pin["file"])


@functools.lru_cache(maxsize=2)
def _fasttext_model(path: str):
    # Input: a fasttext model path. Output: the loaded fasttext model (cached per process).
    import fasttext
    return fasttext.load_model(path)


def load_glotlid(ctx):
    # Input: Ctx. Output: the GlotLID fasttext model (downloaded, SHA-checked and cached per process).
    return _fasttext_model(str(glotlid_path(ctx)))


def model_labels(model) -> list:
    # Input: a fasttext model. Output: its labels without the "__label__" prefix.
    return [label.removeprefix(LABEL_PREFIX) for label in model.get_labels()]


def _items(codes: dict) -> tuple:
    # Input: {code: ISO codes}. Output: its sorted (code, codes) items as a hashable cache key.
    return tuple(sorted((c, tuple(v)) for c, v in codes.items()))


@functools.lru_cache(maxsize=4)
def _aliases(iso_path: str, model_path: str, extra: tuple, only: tuple, candidates: bool) -> dict:
    # Inputs: the verified macrolanguage table path, the GlotLID model path, the extra and only codes as
    # sorted (code, codes) items and whether to derive the candidates instead of the study registry (cache
    # key). Output: {code: frozenset of labels}; raises ValueError when a language ends up with no label.
    with open(iso_path, encoding="utf-8") as fh:
        macro = parse_macrolanguages(fh.read())
    languages = CANDIDATE_LANGUAGES if candidates else LANGUAGES
    table = derive_aliases(languages, macro, model_labels(_fasttext_model(model_path)), dict(extra), dict(only))
    empty = sorted(c for c, labels in table.items() if not labels)
    if empty:
        raise ValueError(f"GlotLID has no label for {empty}; add ISO codes to LID_EXTRA_CODES in core.config")
    return table


def load_aliases(ctx, candidates: bool = False) -> dict:
    # Inputs: Ctx and whether to return the yield-only candidates instead of the study registry. Output:
    # {language code: frozenset of accepted GlotLID labels}, derived from the cached ISO table and GlotLID
    # model (both downloaded and verified when needed).
    return _aliases(str(iso_table_path(ctx)), str(glotlid_path(ctx)), _items(LID_EXTRA_CODES),
                    _items(LID_ONLY_CODES), candidates)


def pair_aliases(ctx) -> dict:
    # Input: Ctx. Output: the study and candidate alias tables merged into one {code: labels} dict (the
    # codes are disjoint), for filtering any NLLB pair.
    return {**load_aliases(ctx), **load_aliases(ctx, candidates=True)}


def lid_record(ctx, aliases: dict, langs=None) -> dict:
    # Inputs: Ctx, the alias table (load_aliases) and optional language codes (default all). Output: the LID
    # rule for manifests and result configs: rule text, top-k, threshold, ISO table pin, extra and only
    # codes, the alias table of those languages and its hash.
    table = {c: sorted(v) for c, v in aliases.items() if langs is None or c in langs}
    cfg, pin = ctx.config(DataConfig), PINS["iso639_3"]
    return {"rule": LID_RULE, "top_k": cfg.glotlid_top_k, "min_prob": cfg.glotlid_min_prob,
            "iso639_3": {"release": pin["release"], "sha256": pin["sha256"]},
            "extra_codes": {c: list(v) for c, v in LID_EXTRA_CODES.items()},
            "only_codes": {c: list(v) for c, v in LID_ONLY_CODES.items()}, "aliases": table,
            "aliases_sha256": sha256_json(table)}


def predict(model, texts, k: int = 1) -> list:
    # Inputs: a fasttext model, a list of texts and the number of labels per text. Output: one
    # (label, prob) tuple per text when k == 1, else one list of k such tuples per text; labels have no
    # "__label__" prefix and newlines are replaced by spaces before prediction.
    out = []
    for text in texts:
        pairs = model.f.predict(text.replace("\n", " ").replace("\r", " "), k, 0.0, "replace")
        ranked = [(label.removeprefix(LABEL_PREFIX), float(prob)) for prob, label in pairs]
        out.append((ranked[0] if ranked else ("", 0.0)) if k == 1 else ranked)
    return out


def ranked_pairs(pred) -> list:
    # Input: one text's prediction, either a single (label, prob) pair (k == 1) or a ranked list of them.
    # Output: the ranked list of (label, prob) pairs.
    return [tuple(pred)] if pred and isinstance(pred[0], str) else [tuple(p) for p in pred]


def top_label(pred) -> tuple:
    # Input: one text's prediction (a pair or a ranked list). Output: the top (label, prob), or ("", 0.0).
    pairs = ranked_pairs(pred)
    return pairs[0] if pairs else ("", 0.0)


def labels_prob(pred, labels) -> float:
    # Inputs: one text's prediction (a pair or a ranked top-k list) and a set of accepted labels. Output:
    # the summed probability of the predicted labels in that set, so mass split over dialect or variant
    # labels (e.g. arb/ary/arz for Arabic) still counts for the language.
    return float(sum(p for label, p in ranked_pairs(pred) if label.removeprefix(LABEL_PREFIX) in labels))


def label_matches(label: str, lang: str, aliases: dict) -> bool:
    # Inputs: a GlotLID label (with or without the prefix), a language code and the alias table. Output:
    # whether the label is one of the language's accepted labels.
    return label.removeprefix(LABEL_PREFIX) in aliases.get(lang, ())


def lid_probs(ctx, texts, lang: str) -> list:
    # Inputs: Ctx, raw texts and a language code. Output: per text, the summed top-k probability of the
    # language's labels (DataConfig.glotlid_top_k), exactly the quantity the NLLB LID filter thresholds.
    preds = predict(load_glotlid(ctx), list(texts), k=ctx.config(DataConfig).glotlid_top_k)
    labels = load_aliases(ctx)[lang]
    return [labels_prob(p, labels) for p in preds]


def lid_passes(ctx, texts, lang: str) -> list:
    # Inputs: Ctx, raw texts and a language code. Output: per text, whether it passes the NLLB LID filter
    # (lid_probs >= DataConfig.glotlid_min_prob).
    threshold = ctx.config(DataConfig).glotlid_min_prob
    return [p >= threshold for p in lid_probs(ctx, texts, lang)]
