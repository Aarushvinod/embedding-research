# Evaluation test sets and the translation rows built from them.
# Three test sets are supported: FLORES devtest (the main result), NTREX-128 (the replication) and
# FLORES dev; their texts come from the corpora package, which caches the pinned files.
# A condition translates from its hub language into each target, so the sources of every row come
# from the hub column and the references from the target column of the same parallel sentence.
# Row ids are "{target}:{sentence id}", so rows of different targets and models pair by id.
# --eval-rows (default 64 in smoke mode) and --pilot restrict the sentence ids for quick runs.

from __future__ import annotations

from ..core.config import CONDITIONS
from ..core.io import sha256_json

EVAL_TESTSETS = ("flores_devtest", "ntrex", "flores_dev")
FLORES_SPLITS = {"flores_devtest": "devtest", "flores_dev": "dev"}
GROUPS = {"all": EVAL_TESTSETS, "main": ("flores_devtest", "ntrex")}


def resolve_eval_testsets(spec="all") -> list:
    # Input: a --testset selection ("all", "main" or a comma list of EVAL_TESTSETS names). Output:
    # the ordered test-set names; unknown names raise ValueError.
    chosen = set()
    for tok in [t.strip() for t in str(spec or "all").split(",") if t.strip()]:
        if tok not in GROUPS and tok not in EVAL_TESTSETS:
            raise ValueError(f"unknown test set {tok!r}; choose from {list(GROUPS) + list(EVAL_TESTSETS)}")
        chosen.update(GROUPS.get(tok, (tok,)))
    return [t for t in EVAL_TESTSETS if t in chosen]


def load_testset(ctx, testset: str, langs) -> dict:
    # Inputs: Ctx, a test-set name and language codes. Output: {code: [sentences]} aligned by
    # sentence index, from the corpora loaders (FLORES split or NTREX).
    if testset == "ntrex":
        from ..corpora.ntrex import load_ntrex
        return load_ntrex(ctx, list(langs))
    from ..corpora.flores import load_flores
    return load_flores(ctx, FLORES_SPLITS[testset], list(langs))


def select_sids(ctx, n_total: int) -> list:
    # Inputs: Ctx and the test-set size. Output: the sentence ids to evaluate: all (or the --pilot
    # subset), capped at the first --eval-rows ids (64 by default in smoke mode).
    cap = ctx.opt("eval_rows", 64 if ctx.smoke else None)
    sids = ctx.rows(n_total)
    return sids if cap is None else sids[:int(cap)]


def testset_rows(ctx, condition: str, testset: str, targets) -> list:
    # Inputs: Ctx, condition key, test-set name and target codes. Output: rows {id, target, sid, src,
    # ref} for every target and selected sentence id, sources from the condition's hub and
    # references from the target, ordered by target then sentence id.
    hub = CONDITIONS[condition].hub
    texts = load_testset(ctx, testset, [hub, *targets])
    sids = select_sids(ctx, len(texts[hub]))
    return [{"id": f"{t}:{s}", "target": t, "sid": s, "src": texts[hub][s], "ref": texts[t][s]}
            for t in targets for s in sids]


def rows_fingerprint(rows) -> str:
    # Input: testset rows. Output: SHA-256 of their ids, sources and references (the data identity
    # recorded in the translate config).
    return sha256_json([[r["id"], r["src"], r["ref"]] for r in rows])
