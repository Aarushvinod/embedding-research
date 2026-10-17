# Public translation-metric API shared by evaluation and the interpretability experiments.
# METRICS lists the four metrics with their kind ("corpus" for chrF++/spBLEU, "mean" for
# COMET/MetricX) and direction; MetricX is the only one where lower is better.
# score_rows scores rows of {id, src, hyp, ref, target?} and returns sentence-level values,
# per-sentence sacreBLEU statistics for the corpus metrics (for stats.bootstrap) and aggregates
# per group plus their macro average and the pooled score.
# score_row_groups scores many keyed row groups (e.g. from several result files) with one COMET or
# MetricX worker call, so the neural model loads once instead of once per file.
# score_result_files stores the scores next to each file's rows under its lock, metric by metric and
# file by file, skipping metrics already scored on identical rows.

from __future__ import annotations

import numpy as np

from ..core.io import sha256_json
from ..core.results import ResultFile, now_iso
from . import neural, sacre
from .batching import score_keyed
METRICS = {
    "chrf_pp": {"kind": "corpus", "higher": True},
    "spbleu": {"kind": "corpus", "higher": True},
    "comet": {"kind": "mean", "higher": True},
    "metricx": {"kind": "mean", "higher": False},
}
ALIASES = {"chrf": "chrf_pp"}


def canonical(metric: str) -> str:
    # Input: a metric name or the accepted alias "chrf". Output: the METRICS key; raises KeyError for
    # unknown names.
    name = ALIASES.get(metric, metric)
    if name not in METRICS:
        raise KeyError(f"unknown metric {metric!r}; choose from {sorted(METRICS)}")
    return name


def as_row_list(rows) -> list:
    # Input: a list of row dicts or a columnar dict of equal-length lists. Output: a list of row dicts.
    if isinstance(rows, dict):
        keys = list(rows)
        return [dict(zip(keys, vals)) for vals in zip(*(rows[k] for k in keys))]
    return list(rows)


def group_index(rows, group_key) -> dict:
    # Inputs: row dicts and the grouping field. Output: {group: [row positions]} in first-seen order,
    # or {} when no row carries the field.
    groups = {}
    for i, row in enumerate(rows):
        if row.get(group_key) is not None:
            groups.setdefault(row[group_key], []).append(i)
    return groups


def _corpus_scorer(ctx, name: str, rows) -> tuple:
    # Inputs: Ctx, a canonical corpus metric name and row dicts. Output: (values array, stats array,
    # signature, agg(positions) -> corpus score) for that metric.
    res = sacre.sentence_scores(ctx, name, [r["hyp"] for r in rows], [r["ref"] for r in rows])
    stats, fn = res["stats"], res["score_fn"]
    return res["values"], stats, res["signature"], lambda idx: sacre.corpus_from_stats(stats[idx], fn)


def _neural_values(ctx, name: str, groups: dict) -> dict:
    # Inputs: Ctx, a neural metric name and {key: row dicts}. Runs ONE worker call over the rows of
    # every group. Output: {key: float64 values aligned to that group's rows}.
    flat = [r for rows in groups.values() for r in rows]
    values = np.asarray(neural.run_worker(ctx, name, [r["src"] for r in flat], [r["hyp"] for r in flat],
                                          [r["ref"] for r in flat]), dtype=np.float64)
    ends = np.cumsum([len(rows) for rows in groups.values()])
    return dict(zip(groups, np.split(values, ends[:-1])))


def _result(name: str, rows, values, stats, sig: str, agg, group_key: str) -> dict:
    # Inputs: canonical metric, row dicts, per-row values, stats (or None), signature, agg(positions)
    # -> score and the grouping field. Output: the score_rows result dict.
    by_group = {g: agg(np.asarray(idx)) for g, idx in group_index(rows, group_key).items()}
    pooled = agg(np.arange(len(rows)))
    overall = float(np.mean(list(by_group.values()))) if by_group else pooled
    aggregate = {"overall": overall, "by_group": by_group, "pooled": pooled, "n": len(rows)}
    return {"metric": name, "values": [float(v) for v in values],
            "stats": None if stats is None else stats.tolist(), "aggregate": aggregate,
            "signature": sig, "higher": METRICS[name]["higher"]}


def _mean_agg(values):
    # Input: per-row values. Output: agg(positions) -> mean of those rows.
    return lambda idx: float(values[idx].mean())


def score_row_groups(ctx, groups: dict, metric: str, group_key: str = "target") -> dict:
    # Inputs: Ctx, {key: rows {id, src, hyp, ref, target?}} (lists or columnar dicts), a metric name
    # and the grouping field. Scores every group; for COMET/MetricX all groups share one worker call.
    # Output: {key: score_rows result of that group}.
    name = canonical(metric)
    groups = {k: as_row_list(v) for k, v in groups.items()}
    if not groups or not all(groups.values()):
        raise ValueError("every row group needs at least one row")
    if METRICS[name]["kind"] == "corpus":
        return {k: _result(name, rows, *_corpus_scorer(ctx, name, rows), group_key) for k, rows in groups.items()}
    values, sig = _neural_values(ctx, name, groups), neural.signature(ctx, name)
    return {k: _result(name, rows, values[k], None, sig, _mean_agg(values[k]), group_key)
            for k, rows in groups.items()}


def score_rows(ctx, rows, metric: str, group_key: str = "target") -> dict:
    # Inputs: Ctx, rows {id, src, hyp, ref, target?} (list or columnar dict), a metric name and the
    # grouping field. Output: {"metric", "values": sentence scores, "stats": per-sentence sacreBLEU
    # statistics or None, "aggregate": {"overall": macro over groups (pooled when ungrouped),
    # "by_group", "pooled", "n"}, "signature", "higher"}.
    if not as_row_list(rows):
        raise ValueError("score_rows needs at least one row")
    return score_row_groups(ctx, {0: rows}, metric, group_key)[0]


def score_paths(rows_key) -> tuple:
    # Input: the rows location as a dotted string or a key tuple (e.g. "per_item.rows"). Output:
    # (rows path, per-item scores path = sibling "scores", aggregate scores path = the same location
    # under "aggregate"), each a tuple of keys.
    path = tuple(rows_key.split(".")) if isinstance(rows_key, str) else tuple(rows_key)
    inner = path[1:-1] if path[0] == "per_item" else path[:-1]
    return path, path[:-1] + ("scores",), ("aggregate",) + inner + ("scores",)


def _get(data: dict, path: tuple, default=None):
    # Inputs: a nested dict and a key path. Output: the value there, or the default when absent.
    for key in path:
        if not isinstance(data, dict) or key not in data:
            return default
        data = data[key]
    return data


def _node(data: dict, path: tuple) -> dict:
    # Inputs: a nested dict and a key path. Output: the dict at that path, created when missing.
    for key in path:
        data = data.setdefault(key, {})
    return data


def rows_digest(rows) -> str:
    # Input: row dicts. Output: SHA-256 over their ids, sources, hypotheses and references, used to
    # detect stale scores.
    return sha256_json([[r.get("id"), r.get("src"), r.get("hyp"), r.get("ref")] for r in rows])


def is_scored(data, rows_key, metric: str) -> bool:
    # Inputs: a result dict, the rows location and a metric. Output: whether that metric is stored
    # in both the per-item and aggregate scores for exactly the current rows.
    path, items, agg = score_paths(rows_key)
    name = canonical(metric)
    stored = _get(data or {}, items + (name,), {})
    rows = _get(data or {}, path)
    return rows is not None and _get(data, agg + (name,)) is not None and \
        stored.get("rows_sha256") == rows_digest(as_row_list(rows))


def _store(data: dict, rows_key, res: dict, digest: str) -> None:
    # Inputs: the result dict, the rows location, a score_rows result and the rows digest it was
    # computed on. Writes per-item values/stats, the aggregate and a status entry in place; raises
    # when the rows changed since scoring.
    path, items, agg = score_paths(rows_key)
    if rows_digest(as_row_list(_get(data, path, []))) != digest:
        raise RuntimeError(f"rows at {'.'.join(path)} changed while scoring {res['metric']}")
    name = res["metric"]
    _node(data, items)[name] = {"values": res["values"], "stats": res["stats"], "rows_sha256": digest}
    _node(data, agg)[name] = {**res["aggregate"], "signature": res["signature"], "higher": res["higher"],
                              "kind": METRICS[name]["kind"], "rows_sha256": digest, "scored": now_iso()}
    done = data.setdefault("status", {}).setdefault("scores_done", [])
    entry = f"{'.'.join(path)}|{name}"
    if entry not in done:
        done.append(entry)


def _file_rows(target, rows_key) -> list:
    # Inputs: a ResultFile or path and the rows location. Output: [ResultFile, loaded data, row
    # dicts]; raises FileNotFoundError when the file holds no rows there.
    rf = target if isinstance(target, ResultFile) else ResultFile(target)
    data = rf.load()
    path = score_paths(rows_key)[0]
    rows = as_row_list(_get(data or {}, path, []))
    if not rows:
        raise FileNotFoundError(f"no rows at {'.'.join(path)} in {rf.path}")
    return [rf, data, rows]


def _store_item(items: list, i: int, res: dict, rows_key) -> None:
    # Inputs: the [ResultFile, data, rows] items, a position, its score_rows result and the rows
    # location. Writes the scores into that file under its lock and refreshes the item's data.
    rf, _, rows = items[i]
    items[i][1] = rf.update(lambda d: _store(d, rows_key, res, rows_digest(rows)))


def score_result_files(ctx, targets, metrics, rows_key="per_item.rows", group_key: str = "target",
                       force: bool = False) -> list:
    # Inputs: Ctx, ResultFiles or paths holding rows at rows_key, metric names, the grouping field and
    # whether to rescore. For each metric, the files not yet scored on their current rows (all with
    # force) are scored together, at most --metrics-batch-rows rows per worker call, and every file is
    # written as soon as its chunk is scored, so a rerun resumes per file and metric. Output:
    # [{metric: stored aggregate}] aligned to targets.
    items = [_file_rows(t, rows_key) for t in targets]
    agg_path = score_paths(rows_key)[2]
    scorer = lambda c, groups, metric: score_row_groups(c, groups, metric, group_key)
    for name in [canonical(m) for m in metrics]:
        todo = {i: item[2] for i, item in enumerate(items) if force or not is_scored(item[1], rows_key, name)}
        score_keyed(ctx, todo, name, lambda i, res: _store_item(items, i, res, rows_key), scorer)
    return [{canonical(m): _get(item[1], agg_path + (canonical(m),)) for m in metrics} for item in items]


def score_result_rows(ctx, result_file, metrics, rows_key="per_item.rows", group_key: str = "target",
                      force: bool = False) -> dict:
    # Inputs: Ctx, one ResultFile or path, metric names, the rows location, the grouping field and the
    # force flag. Output: {metric: stored aggregate} (score_result_files for a single file).
    return score_result_files(ctx, [result_file], metrics, rows_key, group_key, force)[0]
