# Batching of neural-metric scoring across units, shared by the score stage and the interp metrics.
# COMET-22 and MetricX-24-XL take minutes to load, so rows from many result files or units are scored
# in one worker call instead of one call per file.
# score_keyed splits keyed row groups into chunks of at most --metrics-batch-rows rows (60k by default)
# and hands each chunk to a scorer that makes a single call for all of its groups.
# After every chunk a callback stores each group's result, so an interrupted run loses at most the
# chunk in flight and resumes per file or unit.
# It also declares the CLI options of metric runs and imports nothing heavy, so experiments can use it.

from __future__ import annotations

from ..core.registry import opt

BATCH_ROWS = 60_000
BATCH_OPTION = opt("--metrics-batch-rows", type=int, default=None,
                   help=f"most rows one COMET/MetricX worker call scores across units (default {BATCH_ROWS})")
DIR_OPTION = opt("--metrics-dir", default=None,
                 help="directory for the metrics venv and MetricX checkout (default $BYTERESEARCH_METRICS_DIR "
                      "or ROOT/envs; use a local disk such as /content/envs on Colab)")
METRIC_OPTIONS = (BATCH_OPTION, DIR_OPTION)


def max_batch_rows(ctx) -> int:
    # Input: Ctx. Output: the most rows one worker call takes (--metrics-batch-rows, default 60k).
    return int(ctx.opt("metrics_batch_rows", BATCH_ROWS))


def batch_chunks(sizes, max_rows: int) -> list:
    # Inputs: row counts per item and the most rows one call should take. Output: lists of item
    # positions in order, each list's total at most max_rows (an oversized item gets its own list).
    chunks, cur, total = [], [], 0
    for i, n in enumerate(sizes):
        if cur and total + n > max_rows:
            chunks.append(cur)
            cur, total = [], 0
        cur.append(i)
        total += n
    return chunks + ([cur] if cur else [])


def score_keyed(ctx, groups: dict, metric: str, on_scored, scorer) -> None:
    # Inputs: Ctx, {key: row list}, a metric name, on_scored(key, result) and scorer(ctx, {key: rows},
    # metric) -> {key: result} (one worker call per invocation). Scores the groups chunk by chunk and
    # stores every result through on_scored as soon as its chunk is done. No output.
    keys = list(groups)
    for chunk in batch_chunks([len(groups[k]) for k in keys], max_batch_rows(ctx)):
        for key, res in scorer(ctx, {keys[c]: groups[keys[c]] for c in chunk}, metric).items():
            on_scored(key, res)
