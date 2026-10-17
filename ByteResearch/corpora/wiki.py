# Wikipedia sentence samples for LAPE, 40k per study language from wikimedia/wikipedia 20231101.
# The dump is read at a resolved repository revision through HfFileSystem, and every parquet row group of
# the language (1000 articles each) is listed.
# The groups are visited in a seeded random order covering the whole dump and a seeded random batch of
# articles is taken from each group per round, so the sample spans the dump while only the needed row
# groups are downloaded.
# Each article is split into sentences that must be 20-300 characters, free of tokenizer markup, outside the
# FLORES/NTREX exclusion set, unseen, and reach the language's FLORES-calibrated script ratio and GlotLID
# minimums (corpora.thresholds); at most five are drawn at random per article.
# Candidates are reservoir-sampled down to the target count.
# Articles scanned and contributing and any shortfall are recorded in the manifest next to
# ROOT/data/wiki/{lang}/sentences.jsonl.gz.

from __future__ import annotations

import functools
import math
import random
from dataclasses import dataclass

from ..core.config import PINS, DataConfig, language
from ..core.io import read_jsonl_gz, write_jsonl_gz_atomic
from ..core.text import has_special_markup, length_ok, normalize_for_match, script_ratio, split_sentences
from .exclusion import exclusion_set
from .http import hub_errors, retry
from .lid import labels_prob, load_glotlid, pair_aliases, predict
from .manifest import begin_build, finish_build, verified_path
from .thresholds import load_thresholds, lookup

SENTENCES_FILE = "sentences.jsonl.gz"
ARTICLES_PER_GROUP = 64
OVERSAMPLE = 2.0
REASONS = ("length", "script", "markup", "excluded", "lid", "duplicate")


@dataclass(frozen=True)
class WikiSpec:
    # Sentence filters for one language: expected script, inclusive character window, minimum script
    # ratio, maximum sentences per article, the exclusion set and the minimum summed GlotLID probability
    # (applied only when a LID function is given).
    script: str
    min_chars: int = 20
    max_chars: int = 300
    min_script: float = 0.8
    per_article: int = 5
    exclusion: frozenset = frozenset()
    lid_min: float = 0.0


def sentence_reason(sentence: str, spec: WikiSpec, seen: set, lid_fn=None):
    # Inputs: a sentence, the spec, the set of sentences already sampled and an optional lid_fn(text) ->
    # summed probability of the language. Output: the first failing filter's reason, or None when the
    # sentence is eligible.
    if not length_ok(sentence, spec.min_chars, spec.max_chars):
        return "length"
    if script_ratio(sentence, spec.script) < spec.min_script:
        return "script"
    if has_special_markup(sentence):
        return "markup"
    if normalize_for_match(sentence) in spec.exclusion:
        return "excluded"
    if lid_fn is not None and lid_fn(sentence) < spec.lid_min:
        return "lid"
    return "duplicate" if sentence in seen else None


def article_sentences(text: str, spec: WikiSpec, seen: set, rng: random.Random, rejected: dict, lid_fn=None) -> list:
    # Inputs: an article's text, the spec, the sampled-sentence set (updated in place), the seeded RNG,
    # the rejection counters (updated in place) and the optional LID function. Output: up to
    # spec.per_article eligible sentences drawn at random from the article, in article order.
    eligible = []
    for sentence in dict.fromkeys(split_sentences(text)):
        reason = sentence_reason(sentence, spec, seen, lid_fn)
        if reason:
            rejected[reason] = rejected.get(reason, 0) + 1
        else:
            eligible.append(sentence)
    picks = sorted(rng.sample(range(len(eligible)), min(spec.per_article, len(eligible))))
    chosen = [eligible[i] for i in picks]
    seen.update(chosen)
    return chosen


def reservoir_add(reservoir: list, item, n_seen: int, size: int, rng: random.Random) -> None:
    # Inputs: the reservoir list, a new item, how many items preceded it, the reservoir size and the RNG.
    # Applies one step of reservoir sampling (Algorithm R) in place.
    if len(reservoir) < size:
        reservoir.append(item)
        return
    j = rng.randrange(n_seen + 1)
    if j < size:
        reservoir[j] = item


def sample_sentences(articles, spec: WikiSpec, target: int, seed, oversample: float = OVERSAMPLE, lid_fn=None) -> tuple:
    # Inputs: an iterable of (article_id, text) in random order, the spec, the number of sentences wanted,
    # a seed, the oversampling factor and the optional LID function. Streams articles until target*oversample candidates were seen
    # (or the stream ends) and reservoir-samples `target` of them. Output: (rows {i, article_id, text} in
    # stream order, stats with articles scanned/contributing, candidates, rejections and shortfall).
    rng = random.Random(f"{seed}:sample")
    seen, rejected, reservoir = set(), dict.fromkeys(REASONS, 0), []
    scanned, contributing, n_seen, stop_at = 0, 0, 0, math.ceil(target * oversample)
    for article_id, text in articles:
        scanned += 1
        chosen = article_sentences(text or "", spec, seen, rng, rejected, lid_fn)
        contributing += bool(chosen)
        for sentence in chosen:
            reservoir_add(reservoir, (n_seen, article_id, sentence), n_seen, target, rng)
            n_seen += 1
        if n_seen >= stop_at:
            break
    rows = [{"i": i, "article_id": a, "text": t} for i, (_, a, t) in enumerate(sorted(reservoir))]
    stats = {"articles_scanned": scanned, "articles_with_candidates": contributing,
             "articles_contributing": len({r["article_id"] for r in rows}), "candidates": n_seen,
             "rejected": rejected, "kept": len(rows), "shortfall": max(target - len(rows), 0),
             "stopped_early": n_seen >= stop_at}
    return rows, stats


def row_groups(fs, root: str) -> list:
    # Inputs: an HfFileSystem and the dump directory (datasets/<repo>@<rev>/<config>). Output:
    # [(parquet path, row-group index, row count)] for every row group of every parquet file.
    import pyarrow.parquet as pq
    groups = []
    for path in sorted(fs.glob(f"{root}/*.parquet")):
        meta = retry(lambda: pq.ParquetFile(fs.open(path, "rb")).metadata, hub_errors(), label=path)
        groups.extend((path, g, meta.row_group(g).num_rows) for g in range(meta.num_row_groups))
    return groups


def _group_articles(fs, files: dict, group: tuple, seed, round_: int, per_group: int, counter: dict):
    # Inputs: the filesystem, an open-ParquetFile cache, one (path, group, rows) entry, the seed, the round
    # index, articles per group per round and a counter dict. Yields (article_id, text) for this round's
    # slice of the group's seeded article permutation, reading the row group only when the slice is not
    # empty.
    import pyarrow.parquet as pq
    path, g, n = group
    idx = random.Random(f"{seed}:{path}:{g}").sample(range(n), n)[round_ * per_group:(round_ + 1) * per_group]
    if not idx:
        return
    if path not in files:
        files[path] = retry(lambda: pq.ParquetFile(fs.open(path, "rb")), hub_errors(), label=path)
    table = retry(lambda: files[path].read_row_group(g, columns=["id", "text"]), hub_errors(), label=path)
    counter["row_groups_read"] += 1
    ids, texts = table.column("id"), table.column("text")
    for i in idx:
        yield ids[i].as_py(), texts[i].as_py()


def article_stream(fs, groups: list, seed, per_group: int = ARTICLES_PER_GROUP, counter=None):
    # Inputs: the filesystem, the row-group list, a seed, articles per group per round and a counter dict
    # (row_groups_read is incremented). Yields (article_id, text): round r visits every row group in one
    # seeded random order and takes the r-th slice of per_group articles from each, until all are used.
    counter = counter if counter is not None else {"row_groups_read": 0}
    order = random.Random(f"{seed}:groups").sample(range(len(groups)), len(groups))
    rounds = math.ceil(max((n for _, _, n in groups), default=0) / per_group)
    files = {}
    for round_ in range(rounds):
        for gi in order:
            yield from _group_articles(fs, files, groups[gi], seed, round_, per_group, counter)


def build_wiki(ctx, lang: str) -> dict:
    # Inputs: Ctx and a study or candidate language code. Samples the language's sentences from the dump at the
    # resolved revision, with its calibrated script and LID minimums, and writes sentences.jsonl.gz with its
    # manifest. Output: the manifest.
    from huggingface_hub import HfApi, HfFileSystem
    cfg, repo = ctx.config(DataConfig), PINS["wikipedia"]["repo"]
    out = begin_build(ctx, ctx.paths.wiki_dir(lang))
    resolve = lambda: HfApi().repo_info(repo, repo_type="dataset").sha
    revision = PINS["wikipedia"].get("revision") or retry(resolve, hub_errors(), label=repo)
    config_name = f"{cfg.wiki_date}.{language(lang).wiki}"
    fs, root = HfFileSystem(), f"datasets/{repo}@{revision}/{config_name}"
    groups = row_groups(fs, root)
    excl, excl_sha = exclusion_set(ctx)
    table, model, labels = load_thresholds(ctx), load_glotlid(ctx), pair_aliases(ctx)[lang]
    spec = WikiSpec(language(lang).script, cfg.wiki_min_chars, cfg.wiki_max_chars, lookup(table, "script", lang),
                    cfg.wiki_per_article, excl, lookup(table, "lid", lang))
    lid_fn = lambda text: labels_prob(predict(model, [text], k=cfg.glotlid_top_k)[0], labels)
    counter, seed = {"row_groups_read": 0}, f"{cfg.seed}:{lang}"
    rows, stats = sample_sentences(article_stream(fs, groups, seed, ARTICLES_PER_GROUP, counter), spec,
                                   cfg.wiki_per_lang, seed, lid_fn=lid_fn)
    write_jsonl_gz_atomic(out / SENTENCES_FILE, rows)
    source = {"repo": repo, "config": config_name, "revision": revision, "pinned_revision": PINS["wikipedia"].get("revision"),
              "files": sorted({p.split("/")[-1] for p, _, _ in groups}), "row_groups": len(groups),
              "articles_in_dump": sum(n for _, _, n in groups)}
    config = {"target": cfg.wiki_per_lang, "seed": seed, "articles_per_group": ARTICLES_PER_GROUP,
              "oversample": OVERSAMPLE, "exclusion_sha256": excl_sha, "lid_top_k": cfg.glotlid_top_k,
              "lid_labels": sorted(labels),
              "filters": {k: v for k, v in spec.__dict__.items() if k != "exclusion"}}
    return finish_build(ctx, out, sources=[source], config=config, outputs=[SENTENCES_FILE], stats={**stats, **counter})


@functools.lru_cache(maxsize=4)
def _read(path: str, mtime_ns: int) -> tuple:
    # Inputs: a verified sentence-file path and its mtime (cache key). Output: the rows as a tuple (cached).
    return tuple(read_jsonl_gz(path))


def load_wiki(ctx, lang: str) -> list:
    # Inputs: Ctx and a study language code. Output: the sampled rows {i, article_id, text}, after a SHA check.
    path = verified_path(ctx.paths.wiki_dir(lang), SENTENCES_FILE)
    return list(_read(str(path), path.stat().st_mtime_ns))
