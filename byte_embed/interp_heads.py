"""Exp 6 — which attention heads carry LANGUAGE, and is one language's set a subset of English's?

Exps 3-5 are all structural or correlational about the thing that matters. This one intervenes: it
forces a head to emit a different language's signal and measures what happens to the output
embedding. It is also the intervention exp 3 should have been -- zeroing or replacing a head removes
d_kv dimensions of a block's attention output rather than a rank-1 direction, at a fixed site with no
fitted map to go stale, so there is no sequential-chain drift and nothing for the next block to
rebuild from a mean shift.

Design (FLORES 10-way parallel, so row k is the same sentence in every language):

  recipient   sentence k in language A, encoded normally -> z_clean
  D_self      patch head h with A_k's OWN mean          -> the mean-collapse baseline, subtracted off
  D_lang[L]   patch head h with L_k's mean              -> same meaning, different language
  D_content   patch head h with A_k''s mean (k' != k)   -> same language, different meaning

Readouts on the pooled embedding:
  lang[h][L] = cos(z_patched, mu_L) - cos(z_clean, mu_L)     movement toward the donor's LANGUAGE
  content[h] = cos(z_patched, z_k') - cos(z_clean, z_k')     movement toward the donor's SENTENCE

Both are taken net of D_self, so "this head matters at all" is separated from "this head carries the
donor's identity". Centroids mu_L come from a held-out half of the sentences.

Patching is by MASKED MEAN over positions: a donor and its recipient are different languages and so
different lengths, and there is no position correspondence to patch along. The mask matters -- scripts
differ systematically in encoded length, so padding would dilute short sentences more and a length
artifact would read as a language effect.

Screening: zeroing each head once (n_heads passes) and keeping those whose ablation actually moves the
embedding cuts the factorial to the heads that can carry anything. If NO head's ablation moves it, the
representation is too distributed for head-level analysis and that is the finding.

  python -m byte_embed.interp_heads --only byte-base --smoke     # verify the hooks on 8 sentences
  python -m byte_embed.interp_heads --only byte-base
  python -m byte_embed.interp_heads --merge
  python -m byte_embed.interp_heads --selftest
"""
from __future__ import annotations

import argparse

import numpy as np

from byte_embed.interp_common import (MAIN_MODELS, flores_parallel, l2norm, load_student, merge_parts,
                                      models_in, part_path, read_json, tokenize_like_forward,
                                      utf8_stdout, write_json)

ANALYSIS = "heads"
SCHEMA = 1
REF = "en"                         # the reference language the overlap test is run against
TOP_FRAC = 0.10                    # a language's "own" heads: the top this fraction by lang score
OWN_MARGIN = 1.25                  # a donor "owns" a head only by this factor over the runner-up
M_GRID = (0.01, 0.05, 0.10)        # joint-ablation set sizes, as a fraction of all heads
N_RAND = 5                         # random equal-size control sets drawn per set size
POOL = 500                         # parallel sentences in the retrieval pool for the task readout
MATRIX_MAX = 3                     # winner heads per model that get the donor x target matrix


# ----------------------------------------------------------------------------------------------
# module access
# ----------------------------------------------------------------------------------------------
def attn_o(student, block):
    """The output projection of block `block`'s self-attention. Its INPUT is the concatenated
    per-head outputs [B, L, n_heads*d_kv] (modeling_t5.py:366-367), which is the patch point."""
    return student.enc.encoder.block[block].layer[0].SelfAttention.o


def head_geom(student):
    """(n_blocks, n_heads, d_kv) read off the model, never assumed: d_kv is 64 across T5 variants but
    n_heads is 6/12/16 depending on backbone, so one head is 1/6 of a block's attention output in some
    models and 1/16 in others. Head COUNTS are therefore not comparable across models; fractions are."""
    sa = student.enc.encoder.block[0].layer[0].SelfAttention
    return len(student.enc.encoder.block), int(sa.n_heads), int(sa.key_value_proj_dim)


def _slice(head, d_kv):
    return slice(head * d_kv, (head + 1) * d_kv)


# ----------------------------------------------------------------------------------------------
# hooks
# ----------------------------------------------------------------------------------------------
def _record_hooks(student, store, mask, n_blocks, n_heads, d_kv):
    """Pre-hooks on every block's `o` that stash the MASKED MEAN of each head's output. One pass
    records every head, so donor caching is per-language, not per-head."""
    import torch
    handles = []
    m = mask.unsqueeze(-1).to(torch.float32)
    denom = m.sum(1).clamp(min=1.0)

    def mk(b):
        def pre(_mod, args):
            x = args[0]
            v = (x.float() * m).sum(1) / denom                 # [B, n_heads*d_kv]
            store[b] = v.reshape(v.shape[0], n_heads, d_kv).cpu().numpy()
        return pre
    for b in range(n_blocks):
        handles.append(attn_o(student, b).register_forward_pre_hook(mk(b)))
    return handles


def _patch_hook(student, block, head, d_kv, value):
    """Force `head` to emit `value` ([B, d_kv] or None to zero it) at EVERY position."""
    import torch

    def pre(_mod, args):
        x = args[0].clone()
        sl = _slice(head, d_kv)
        if value is None:
            x[:, :, sl] = 0
        else:
            v = torch.as_tensor(value, device=x.device, dtype=x.dtype)
            x[:, :, sl] = v.unsqueeze(1)
        return (x,) + tuple(args[1:])
    return [attn_o(student, block).register_forward_pre_hook(pre)]


# ----------------------------------------------------------------------------------------------
# batched passes
# ----------------------------------------------------------------------------------------------
def encode_plain(student, texts, device, batch_size):
    import torch
    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            out.append(student(texts[i:i + batch_size], device=device).float().cpu().numpy())
    return l2norm(np.concatenate(out, 0))


def donor_means(student, texts, device, batch_size, geom):
    """[N, n_blocks, n_heads, d_kv] — every head's masked-mean output for every text, in one pass
    per batch."""
    import torch
    n_blocks, n_heads, d_kv = geom
    chunks = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            bt = texts[i:i + batch_size]
            mask = tokenize_like_forward(student, bt, device)["attention_mask"]
            store, handles = {}, []
            try:
                handles = _record_hooks(student, store, mask, n_blocks, n_heads, d_kv)
                student(bt, device=device)
            finally:
                for h in handles:
                    h.remove()
            chunks.append(np.stack([store[b] for b in range(n_blocks)], 1))   # [B, nb, nh, dkv]
    return np.concatenate(chunks, 0)


def _ablate_hooks(student, head_ids, n_heads, d_kv):
    """Zero EVERY head in `head_ids` (flat indices) at once -- one pre-hook per block touched.

    Single-head ablation is bounded by what one head carries, and the screen put that at 1-cos
    <= 0.017: far too little to move a task metric, whatever the head does. The INRIA protocol
    ablates the top-m% jointly for exactly this reason, so the unit of intervention has to be a
    SET, with a random equal-size set as the control."""
    import torch  # noqa: F401  (hooks run under the caller's inference_mode)
    by_block = {}
    for j in head_ids:
        by_block.setdefault(int(j) // n_heads, []).append(int(j) % n_heads)

    def mk(hs):
        def pre(_mod, args):
            x = args[0].clone()
            for h in hs:
                x[:, :, _slice(h, d_kv)] = 0
            return (x,) + tuple(args[1:])
        return pre
    return [attn_o(student, b).register_forward_pre_hook(mk(hs)) for b, hs in by_block.items()]


def encode_ablated(student, texts, head_ids, n_heads, d_kv, device, batch_size):
    """Embeddings for `texts` with every head in `head_ids` zeroed simultaneously."""
    import torch
    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            handles = _ablate_hooks(student, head_ids, n_heads, d_kv)
            try:
                out.append(student(texts[i:i + batch_size], device=device).float().cpu().numpy())
            finally:
                for h in handles:
                    h.remove()
    return l2norm(np.concatenate(out, 0))


def encode_patched(student, texts, block, head, d_kv, values, device, batch_size):
    """Embeddings for `texts` with (block, head) forced to `values` ([N, d_kv], or None to zero)."""
    import torch
    out = []
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            v = None if values is None else values[i:i + batch_size]
            handles = _patch_hook(student, block, head, d_kv, v)
            try:
                out.append(student(texts[i:i + batch_size], device=device).float().cpu().numpy())
            finally:
                for h in handles:
                    h.remove()
    return l2norm(np.concatenate(out, 0))


# ----------------------------------------------------------------------------------------------
# statistics (pure numpy -- these are what the selftest covers)
# ----------------------------------------------------------------------------------------------
def toward(z_patched, z_clean, target):
    """Mean movement of the embedding toward `target`: cos(patched, target) - cos(clean, target)."""
    t = np.asarray(target, np.float32)
    t = l2norm(t[None, :] if t.ndim == 1 else t)      # l2norm needs 2-D; a centroid arrives 1-D
    return float(((z_patched * t).sum(1) - (z_clean * t).sum(1)).mean())


def toward_each(z_patched, z_clean, target):
    """Per-recipient movement toward `target`: cos(patched, t) - cos(clean, t), shape [N]. The
    per-row version of `toward`, so a matrix row can drop recipients already in the donor's language
    -- for those, the donor patch IS the self patch and the row would be diluted by exact zeros."""
    t = np.asarray(target, np.float32)
    t = l2norm(t[None, :] if t.ndim == 1 else t)
    return (z_patched * t).sum(1) - (z_clean * t).sum(1)


def winner_heads(res, max_n=MATRIX_MAX):
    """[(flat index, role, wins)] -- the heads the donor x target matrix is worth running on.

    winner     `best@` for at least two donors in the single-head profile: the candidates the
               per-donor table flagged as possible language-IDENTITY heads
    reference  the top head by the ablation SCREEN, when it is not already a winner: the most
               perturbation-sensitive head in the model, run so there is a known sensitivity head
               to compare the winners' matrices against

    When the top-screen head IS a winner, no separate reference is added -- and that coincidence is
    itself evidence for the sensitivity reading, so the report says so."""
    pr = res.get("profile") or {}
    if not pr.get("heads"):
        return []
    wins = {}
    for l in res.get("langs") or []:
        if l in pr["lang"]:
            j = int(np.argmax(pr["lang"][l]))
            wins[j] = wins.get(j, 0) + 1
    ranked = sorted(wins.items(), key=lambda kv: (-kv[1], kv[0]))
    out = [(j, "winner", c) for j, c in ranked if c >= 2][:max_n]
    if not out and ranked:
        out = [(ranked[0][0], "winner", ranked[0][1])]
    sc = np.asarray(res.get("screen") or [], float)
    if len(sc):
        top = int(np.argmax(sc))
        if top not in {j for j, _, _ in out}:
            out.append((top, "reference", wins.get(top, 0)))
    return out


def matrix_stats(M):
    """Directional selectivity of one head's donor x target matrix ([n_lang, n_lang], row = donor).

      diag_hits    donors whose row peaks ON the diagonal -- patching with L moves the embedding
                   toward L more than toward any other language. Chance is 1 in n_lang per row.
      p            one-sided binomial P(hits >= observed) under that chance. With 10 languages,
                   4+ hits is p < 0.05.
      diag_excess  mean diagonal minus mean off-diagonal movement.

    An IDENTITY head has a dominant diagonal. A SENSITIVITY head -- one that moves the embedding a
    lot under any patch -- moves toward every centroid about equally and has a flat matrix, which is
    exactly the case the diagonal-only profile could not tell apart from identity."""
    from math import comb
    M = np.asarray(M, float)
    n = M.shape[0]
    hits = int(sum(int(np.argmax(M[i])) == i for i in range(n)))
    p = sum(comb(n, k) * (1 / n) ** k * (1 - 1 / n) ** (n - k) for k in range(hits, n + 1))
    off = M[~np.eye(n, dtype=bool)]
    return {"diag_hits": hits, "p": round(float(p), 5),
            "diag_excess": round(float(np.diag(M).mean() - off.mean()), 6),
            "diag_mean": round(float(np.diag(M).mean()), 6)}


def top_heads(scores, frac=TOP_FRAC):
    """Indices of the top `frac` of heads by score. A FRACTION, not a count: head counts run 48 to
    384 across the grid, so a fixed k would compare a tenth of one model against a fortieth of
    another."""
    k = max(1, int(round(frac * len(scores))))
    return set(np.argsort(-np.asarray(scores, float))[:k].tolist())


def lang_sets(screen_lang, langs, frac):
    """{lang: top-`frac` heads by THAT LANGUAGE's own ablation effect}.

    The scalar screen averages over a language-balanced recipient set, so a head that matters only
    for Telugu contributes its effect on 50 of 500 sentences and is judged at a TENTH of its true
    size -- then filtered out before the language measurement ever runs. Ranking inside a language
    removes that dilution exactly, and costs nothing: the recipients were already stratified."""
    S = np.asarray(screen_lang, float)                       # [n_heads, n_lang]
    k = max(1, int(round(frac * S.shape[0])))
    return {l: sorted(int(j) for j in np.argsort(-S[:, i])[:k]) for i, l in enumerate(langs)}


def random_sets(n_total, k, n_draws, seed=0):
    """`n_draws` random head sets of size k -- the control that says whether a language set's damage
    is about WHICH heads it holds or merely HOW MANY."""
    rng = np.random.default_rng(seed)
    return [sorted(int(j) for j in rng.choice(n_total, size=k, replace=False)) for _ in range(n_draws)]


def align_scores(Z, langs):
    """{lang: mean P@1 retrieving every OTHER language from it}.

    A TASK readout, replacing `1 - cos`. Cosine movement says the embedding shifted; it does not say
    the model stopped working, and a head can move the embedding a long way without costing a single
    retrieval. The pool is index-aligned parallel text, so row i is the same sentence in every
    language and the gold match is the diagonal."""
    out = {}
    for a in langs:
        ps = []
        for b in langs:
            if a == b:
                continue
            S = Z[a] @ Z[b].T
            ps.append(float((S.argmax(1) == np.arange(len(Z[a]))).mean()))
        out[a] = round(float(np.mean(ps)), 5)
    return out


def overlap_stats(prof, langs, ref=REF, frac=TOP_FRAC):
    """Is language L's head set a subset of the REFERENCE language's, more than of a typical other?

      own(L)   = |S_L & S_ref| / |S_L|
      typical  = mean over C not in {L, ref} of |S_L & S_C| / |S_L|
      excess   = own - typical      > 0 => the reference is the general mechanism

    `prof[L]` is the per-head language score for donor language L."""
    S = {l: top_heads(prof[l], frac) for l in langs if l in prof}
    if ref not in S:
        return None
    out = {}
    for l in S:
        if l == ref:
            continue
        own = len(S[l] & S[ref]) / max(len(S[l]), 1)
        others = [len(S[l] & S[c]) / max(len(S[l]), 1) for c in S if c not in (l, ref)]
        out[l] = {"own": round(own, 3), "typical": round(float(np.mean(others)), 3) if others else None,
                  "excess": round(own - float(np.mean(others)), 3) if others else None,
                  "n_heads_in_set": len(S[l])}
    return out


def specificity(prof, langs):
    """Per head: is it language-GENERAL (responds to every donor language) or language-SPECIFIC?
    Participation ratio of its score profile across languages, normalised to [0, 1]: 1 => responds
    equally to all, 1/len(langs) => responds to exactly one."""
    have = [l for l in langs if l in prof]
    P = np.clip(np.array([prof[l] for l in have], float), 0, None)     # [n_lang, n_head]
    s = P.sum(0)
    pr = np.where(s > 0, s ** 2 / np.maximum((P ** 2).sum(0), 1e-12), np.nan)
    return pr / len(have)


# ----------------------------------------------------------------------------------------------
def run_joint(student, res, par, langs, pool_ids, geom, device, bs, outp,
              m_grid=M_GRID, n_rand=N_RAND, seed=0):
    """Joint ablation of per-language head SETS, scored by retrieval, against random sets.

    This is the INRIA ablation protocol ported to a bi-encoder: take the top-m% heads for a
    language, zero them all at once, and ask what it costs the TASK. Two numbers decide whether a
    language has its own machinery, and both have to be positive:

      specificity  drop for language L minus the mean drop for the other nine, under L's own set.
                   Negative means L's set is simply a set of important heads, not L's heads.
      vs random    drop for L under L's set minus its drop under random sets of the SAME SIZE.
                   Negative means the damage is about how many heads were removed, not which.

    Steering and BLEU do not port -- there is no generation to steer into -- but the ablation and a
    task metric do, and those were the parts whose absence made the single-head version a null."""
    n_blocks, n_heads, d_kv = geom
    total = n_blocks * n_heads
    pool = {l: [par[l][i] for i in pool_ids] for l in langs}
    npool = len(pool_ids)

    def align_for(head_ids):
        Z = {l: (encode_plain(student, pool[l], device, bs) if head_ids is None
                 else encode_ablated(student, pool[l], head_ids, n_heads, d_kv, device, bs))
             for l in langs}
        return align_scores(Z, langs)

    j = res.get("joint") or {"pool_n": npool, "m_grid": list(m_grid), "n_rand": n_rand,
                             "lang": {}, "rand": {}}
    if "clean" not in j:
        j["clean"] = align_for(None)
        res["joint"] = j
        write_json(outp, res)
        print(f"    clean align over {npool} parallel sentences: "
              f"mean P@1 {np.mean(list(j['clean'].values())):.4f}")
    sets = {m: lang_sets(res["screen_lang"], langs, m) for m in m_grid}
    for m in m_grid:
        mk, k = str(m), len(next(iter(sets[m].values())))
        for l in langs:
            if l in (j["lang"].get(mk) or {}):
                continue
            a = align_for(sets[m][l])
            j["lang"].setdefault(mk, {})[l] = {"set": sets[m][l], "align": a,
                                               "own_drop": round(j["clean"][l] - a[l], 5)}
            res["joint"] = j
            write_json(outp, res)
            print(f"    m={m:.0%} ({k} heads)  {l}: own drop "
                  f"{j['lang'][mk][l]['own_drop']:+.4f}")
        # the control sets are drawn from (m, seed) alone, so a requeue redraws exactly the same
        # ones and a half-finished grid never mixes two different null distributions
        have = len(j["rand"].get(mk) or [])
        for r, hs in enumerate(random_sets(total, k, n_rand, seed + int(m * 1000))):
            if r < have:
                continue
            a = align_for(hs)
            j["rand"].setdefault(mk, []).append({"set": hs, "align": a})
            res["joint"] = j
            write_json(outp, res)
        rd = np.mean([np.mean([j["clean"][l] - d["align"][l] for l in langs])
                      for d in j["rand"][mk]])
        print(f"    m={m:.0%}: random-set mean drop {rd:+.4f}")
    res["joint"] = j
    write_json(outp, res)


def run_one(name, results, ckpt_dir, device, n_sent=200, seed=0, screen_frac=0.25, smoke=False):
    outp = part_path(ANALYSIS, name)
    res = read_json(outp) or {}
    if res.get("schema") != SCHEMA:
        res = {"schema": SCHEMA}
    if res.get("profile") is not None and res.get("joint") and "matrix" in res and not smoke:
        print(f"=== {ANALYSIS}/{name}: already done -> skip ===")
        return
    if device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            raise SystemExit(
                "[heads] no GPU visible. Login nodes have none -- get an allocation first:" + chr(10) +
                "  srun --partition=scavenger --account=scavenger --qos=scavenger" + chr(10) +
                "       --gres=gpu:rtxa6000:1 --cpus-per-task=8 --mem=48G --time=00:30:00" + chr(10) +
                "       python -m byte_embed.interp_heads --only <model> --smoke" + chr(10) +
                "(--device cpu works for --smoke but not for a full run)")
    loaded = load_student(name, results, ckpt_dir, device)
    if loaded is None:
        return
    student, bm, _ = loaded
    geom = head_geom(student)
    n_blocks, n_heads, d_kv = geom
    bs = 16 if bm.get("kind") == "byte" else 64
    par = flores_parallel(cache_dir=ckpt_dir)
    langs = list(par)
    nsent = len(par[langs[0]])
    res.update(model=name, kind=bm.get("kind"), n_blocks=n_blocks, n_heads=n_heads, d_kv=d_kv,
               total_heads=n_blocks * n_heads, langs=langs)
    print(f"  {name}: {n_blocks} blocks x {n_heads} heads = {n_blocks * n_heads}, d_kv={d_kv}")

    rng = np.random.default_rng(seed)
    held = rng.permutation(nsent)
    cent_ids, rec_ids = held[:nsent // 2], held[nsent // 2:]
    if smoke:
        rec_ids, cent_ids = rec_ids[:8], cent_ids[:32]

    # centroids from the HELD-OUT half, so "moved toward language L" is not scored against the very
    # sentences being patched
    cent = {l: l2norm(encode_plain(student, [par[l][i] for i in cent_ids], device, bs).mean(0)[None, :])[0]
            for l in langs}
    # retrieval pool for the task readout: the SAME sentence indices in every language (FLORES rows
    # are parallel), taken from the centroid half so it is disjoint from the patched recipients
    pool_ids = [int(i) for i in cent_ids[:POOL]]

    # recipients: an equal slice of sentences per language, so no language dominates the averages
    per = max(len(rec_ids) // len(langs), 1) if not smoke else 1
    rec = [(int(i), l) for k, l in enumerate(langs) for i in rec_ids[k * per:(k + 1) * per]]
    r_txt = [par[l][i] for i, l in rec]
    z_clean = encode_plain(student, r_txt, device, bs)
    print(f"  {len(rec)} recipient sentences ({per}/language), centroids from {len(cent_ids)} held out")

    # donor caches: one pass per donor set, every head at once
    D = {l: donor_means(student, [par[l][i] for i, _ in rec], device, bs, geom) for l in langs}
    D["__self__"] = np.stack([D[l][k] for k, (_, l) in enumerate(rec)], 0)
    shuf = rng.permutation(len(rec))
    c_txt = [r_txt[j] for j in shuf]                       # same language, different sentence
    D["__content__"] = np.stack([D[rec[j][1]][j] for j in shuf], 0)
    z_content_target = z_clean[shuf]
    print(f"  donor caches: {len(D)} sets of {D[langs[0]].shape}")

    heads = [(b, h) for b in range(n_blocks) for h in range(n_heads)]

    if smoke:
        # Four heads spread across depth: enough to prove the path, cheap enough to be a smoke test.
        probe = [(b, 0) for b in sorted({0, n_blocks // 3, 2 * n_blocks // 3, n_blocks - 1})]
        print(f"  SMOKE: {len(rec)} recipients, probing heads {probe}")
        d = lambda z: float(1.0 - (z * z_clean).sum(1).mean())            # noqa: E731
        fires = lands = False
        print(f"    {'head':>12}{'ablate':>10}{'self-mean':>11}{'alt-lang':>10}{'self x50':>10}"
              f"{'|abl-self|':>12}")
        for b, h in probe:
            abl = d(encode_patched(student, r_txt, b, h, d_kv, None, device, bs))
            slf = d(encode_patched(student, r_txt, b, h, d_kv, D["__self__"][:, b, h], device, bs))
            alt = d(encode_patched(student, r_txt, b, h, d_kv, D[langs[0]][:, b, h], device, bs))
            # A hook that DROPPED its argument would be insensitive to the value's scale, so a 50x
            # value disturbing far more than zeroing is the decisive evidence that it lands.
            big = d(encode_patched(student, r_txt, b, h, d_kv,
                                   D["__self__"][:, b, h] * 50.0, device, bs))
            fires = fires or abl > 1e-5
            lands = lands or big > 3.0 * max(abl, 1e-9)
            # Early heads often have a near-ZERO mean, so substituting it is the same intervention as
            # zeroing and this spread is ~0 however correct the hook is. Context, not a criterion.
            spread = abs(abl - slf) / max(abl, slf, 1e-12)
            print(f"    b{b:<2} h{h:<8}{abl:>10.5f}{slf:>11.5f}{alt:>10.5f}{big:>10.5f}"
                  f"{spread:>12.2f}")
        # Does _slice() index the head we think it does? A wrong d_kv would straddle two heads and
        # still pass `fires` and `lands`, corrupting every per-head number with no symptom. Patch one
        # head while recording ALL heads of that block -- the record hooks go on after the patch hook
        # so they see the patched tensor -- and require that exactly that head moved. Last block, so
        # nothing downstream can contaminate it.
        ib, ih = n_blocks - 1, min(1, n_heads - 1)
        before = donor_means(student, r_txt, device, bs, geom)
        _hs = _patch_hook(student, ib, ih, d_kv, D["__self__"][:, ib, ih] * 50.0)
        try:
            after = donor_means(student, r_txt, device, bs, geom)
        finally:
            for _x in _hs:
                _x.remove()
        moved = np.abs(after[:, ib] - before[:, ib]).max(axis=(0, 2))      # [n_heads]
        others = float(np.delete(moved, ih).max()) if n_heads > 1 else 0.0
        isolates = bool(moved[ih] > 1e-4 and others < 1e-5)
        print(f"    isolation: patched b{ib} h{ih} -> that head moved {moved[ih]:.4f}, "
              f"every other head in the block moved <= {others:.2e}")
        print(f"  hooks fire: {fires}    value lands: {lands}    head slice isolates: {isolates}")
        print("  SMOKE " + ("PASS" if fires and lands and isolates else
                            "FAIL -- do not run the full job"))
        return

    # ---- screen: zero each head once, PER LANGUAGE. The recipients are already 50 per language,
    # so grouping the per-sentence cosines by language costs no extra forward passes -- the old
    # scalar simply collapsed them one step too early, and that collapse is what hid any head whose
    # importance is concentrated in one language. `screen` stays as the mean over languages, which
    # is what the equal-sized groups made the old number anyway.
    r_lang = np.array([l for _, l in rec])
    lang_idx = {l: np.flatnonzero(r_lang == l) for l in langs}
    if "screen_lang" not in res:
        sl = []
        for j, (b, h) in enumerate(heads):
            z = encode_patched(student, r_txt, b, h, d_kv, None, device, bs)
            cos = (z * z_clean).sum(1)
            sl.append([float(1.0 - cos[lang_idx[l]].mean()) for l in langs])
            if j % max(len(heads) // 8, 1) == 0:
                print(f"    screen {j + 1}/{len(heads)}  1-cos mean={np.mean(sl[-1]):.4f} "
                      f"max-lang={max(sl[-1]):.4f}")
        res["screen_lang"] = sl
        res["screen"] = [float(np.mean(v)) for v in sl]
        write_json(outp, res)
    sl = np.asarray(res["screen_lang"], float)
    spread = sl.max(1) - sl.min(1)
    print(f"  per-language screen: best single-language effect {sl.max():.4f} "
          f"(vs {np.asarray(res['screen'], float).max():.4f} averaged), "
          f"widest language spread {spread.max():.4f}")
    sc = np.array(res["screen"], float)
    keep = sorted(top_heads(sc, screen_frac))
    print(f"  screen: max 1-cos {sc.max():.4f}, median {np.median(sc):.4f}; "
          f"profiling the top {len(keep)}/{len(heads)}")
    flat = sc.max() < 1e-4
    if flat:
        print("  NO head's ablation moves the embedding -> nothing for the single-head profile. "
              "That is a result; the joint-ablation stage below still runs, and is the one that "
              "can distinguish 'no language structure' from 'structure spread across heads'.")
        res["profile"] = {}
        write_json(outp, res)

    # ---- profile the survivors: self baseline, then language and content, net of it
    prof = {l: np.zeros(len(heads)) for l in langs}
    content = np.zeros(len(heads))
    selfc = np.zeros(len(heads))
    for n, j in enumerate([] if flat or res.get("profile") else keep):
        b, h = heads[j]
        z_self = encode_patched(student, r_txt, b, h, d_kv, D["__self__"][:, b, h], device, bs)
        selfc[j] = float(1.0 - (z_self * z_clean).sum(1).mean())
        for l in langs:
            z = encode_patched(student, r_txt, b, h, d_kv, D[l][:, b, h], device, bs)
            prof[l][j] = toward(z, z_clean, cent[l]) - toward(z_self, z_clean, cent[l])
        zc = encode_patched(student, r_txt, b, h, d_kv, D["__content__"][:, b, h], device, bs)
        content[j] = (float(((zc * z_content_target).sum(1) - (z_clean * z_content_target).sum(1)).mean())
                      - float(((z_self * z_content_target).sum(1) - (z_clean * z_content_target).sum(1)).mean()))
        if n % max(len(keep) // 8, 1) == 0:
            print(f"    profile {n + 1}/{len(keep)}  block {b} head {h}  "
                  f"lang(max)={max(prof[l][j] for l in langs):+.4f}  content={content[j]:+.4f}")
    if not flat and not res.get("profile"):
        res["profile"] = {"heads": keep, "lang": {l: prof[l].tolist() for l in langs},
                          "content": content.tolist(), "self_cost": selfc.tolist(),
                          "screen_frac": screen_frac, "n_recipients": len(rec)}
        write_json(outp, res)

    # ---- joint ablation of per-language head SETS, scored by retrieval, against random sets.
    # Runs whatever the single-head profile found: its premise is that no single head carries
    # enough to matter, which is what the screen keeps reporting.
    print("  joint ablation: per-language head sets vs random sets, scored by retrieval")
    run_joint(student, res, par, langs, pool_ids, geom, device, bs, outp, seed=seed)

    # ---- donor x target matrix on the winner heads: identity head or sensitivity head?
    # The profile scored movement toward the DONOR's centroid only. A perturbation-sensitive head
    # moves toward every centroid, so it wins for most donors for that reason alone -- identity and
    # sensitivity predict the same diagonal and differ only off it. Score every target.
    if "matrix" not in res:
        heads_m = winner_heads(res)
        rows = []
        for j, role, wins in heads_m:
            b, h = heads[j]
            z_self = encode_patched(student, r_txt, b, h, d_kv, D["__self__"][:, b, h], device, bs)
            base = {t: toward_each(z_self, z_clean, cent[t]) for t in langs}
            M = np.zeros((len(langs), len(langs)))
            for a, l in enumerate(langs):
                z = encode_patched(student, r_txt, b, h, d_kv, D[l][:, b, h], device, bs)
                keep_r = r_lang != l      # a donor patch on its own language IS the self patch
                for c, t in enumerate(langs):
                    M[a, c] = float((toward_each(z, z_clean, cent[t]) - base[t])[keep_r].mean())
            st = matrix_stats(M)
            rows.append({"idx": int(j), "block": int(b), "head": int(h), "role": role,
                         "wins": int(wins), "screen": float(res["screen"][j]),
                         "M": np.round(M, 6).tolist(), **st})
            print(f"    matrix b{b}h{h} ({role}, best@ for {wins} donors): diagonal hits "
                  f"{st['diag_hits']}/{len(langs)}  p={st['p']:.4f}  "
                  f"diag excess {st['diag_excess']:+.5f}")
        res["matrix"] = {"langs": langs, "heads": rows,
                         "top_screen_is_winner": bool(rows and all(r_["role"] == "winner"
                                                                   for r_ in rows))}
        write_json(outp, res)
    print(f"  saved -> {outp}")


# ----------------------------------------------------------------------------------------------
def report_per_lang(M, models=None):
    """Per DONOR LANGUAGE, what the patching actually found -- the breakdown the summary hides.

    The aggregate reports `max language score` over all donors at once, which cannot distinguish
    "every language is equally unrepresented" from "one language has real heads and the rest do
    not". Per donor:

      best        the largest language score any profiled head reaches for this donor
      mean        its mean over the profiled heads
      owns        heads this donor wins CLEARLY (top score > OWN_MARGIN x the runner-up) -- the
                  nearest thing to "its own heads". A plain argmax would hand every head with no
                  language signal to whichever donor sorts first, which on scores this close to
                  zero is pure list order; heads without a clear winner are reported as contested
                  instead, and a high contested count is itself the finding
      best@       where that head sits (block/head), so depth is visible
      vs content  that head's language score over its OWN content score. Below 1 means the head
                  moves further when fed a different SENTENCE than a different LANGUAGE -- it is a
                  content head, not a language head.

    `vs content` is the column that decides whether any of this is a language result at all."""
    print("\n  PER-DONOR-LANGUAGE breakdown (profiled heads only)")
    over = []                                   # (model, donor, ratio, stable, head) with ratio > 1
    for n in (models or MAIN_MODELS):
        r = M.get(n)
        pr = (r or {}).get("profile") or {}
        if not pr.get("heads"):
            continue
        langs, keep = r["langs"], pr["heads"]
        prof = {l: np.array(v, float) for l, v in pr["lang"].items()}
        cont = np.array(pr["content"], float)
        nh = int(r.get("n_heads") or 1)
        have = [l for l in langs if l in prof]
        stack = np.stack([prof[l] for l in have])              # [n_lang, n_head_slots]
        # A head "belongs to" a donor only when that donor wins CLEARLY; counted over the PROFILED
        # slots only, since the rest are zeros and tie across every donor.
        owner, contested = dict.fromkeys(have, 0), 0
        for j in keep:
            col = np.sort(stack[:, j])[::-1]
            if col[0] > 0 and (len(col) < 2 or col[0] > OWN_MARGIN * max(col[1], 1e-12)):
                owner[have[int(np.nanargmax(stack[:, j]))]] += 1
            else:
                contested += 1
        ov = overlap_stats(prof, langs) or {}
        # A ratio is only as good as its denominator: a head that barely moves on content makes any
        # language score look large beside it. Mark ratios whose content score sits below the
        # model's median profiled content score, so a small-over-smaller is not read as a finding.
        med_c = float(np.median(np.abs(cont[keep])))
        print(f"\n    {n}  ({len(keep)} heads profiled, {contested} with no clear owner)")
        print(f"    {'donor':>6}{'best':>9}{'mean':>9}{'owns':>6}{'best@':>9}"
              f"{'vs content':>12}{'excess':>9}")
        for l in sorted(have):
            v = prof[l]
            j = int(np.argmax(v))
            b, h = divmod(j, nh)
            ratio = v[j] / cont[j] if abs(cont[j]) > 1e-9 else float("nan")
            stable = abs(cont[j]) >= med_c
            if ratio > 1:
                over.append((n, l, ratio, stable, f"b{b}h{h}"))
            ex = (ov.get(l) or {}).get("excess")
            at = f"b{b}h{h}"
            print(f"    {l:>6}{v[j]:>+9.4f}{float(np.mean(v[keep])):>+9.4f}{owner[l]:>6}{at:>9}"
                  f"{ratio:>11.2f}{'' if stable else '*':1}"
                  + (f"{ex:>+9.3f}" if ex is not None else f"{'-':>9}"))
    # the verdict is COMPUTED: a fixed sentence here once asserted "< 1 at every donor" directly
    # beneath a table with three donors above 1
    print("\n  * = that head's content score is below its model's median, so the ratio is unstable")
    solid = [o for o in over if o[3]]
    if not over:
        print("  vs content < 1 at every donor in every model: these are content heads, not language")
        print("  heads.")
    else:
        print(f"  vs content > 1 for {len(over)} donor(s), {len(solid)} on a stable denominator:")
        for m_, l, ratio, stable, at in over:
            print(f"    {m_:14} {l:>3} {ratio:5.2f} at {at:>7}  "
                  + ("stable -- worth a second look" if stable else "small denominator, unstable"))
        print("  A language head needs a donor whose best head moves further on LANGUAGE than on")
        print("  CONTENT on a denominator that is not itself near zero.")


def joint_stats(r, m):
    """Per language, under its OWN top-m% set: (own drop, mean drop to the other nine, mean drop
    under random sets of the same size). Returns None when that grid point is unfinished."""
    j = r.get("joint") or {}
    mk = str(m)
    per, rnd, clean = (j.get("lang") or {}).get(mk), (j.get("rand") or {}).get(mk), j.get("clean")
    if not per or not rnd or not clean:
        return None
    out = {}
    for l, d in per.items():
        others = [clean[x] - d["align"][x] for x in clean if x != l]
        rdrop = [clean[l] - q["align"][l] for q in rnd]
        out[l] = {"own": clean[l] - d["align"][l], "others": float(np.mean(others)),
                  "rand": float(np.mean(rdrop)), "n_heads": len(d["set"])}
    return out


def _sign_p(vals):
    """Exact two-sided sign test over the nonzero values: (p, n_positive, n). Across 10 languages
    only 9/10 or 10/10 of one sign reaches p < 0.05 -- low power, but it assumes nothing about the
    distribution and it is honest about how little ten numbers can say."""
    from math import comb
    v = [x for x in vals if x is not None and x != 0]
    n = len(v)
    if not n:
        return None, 0, 0
    pos = sum(1 for x in v if x > 0)
    k = min(pos, n - pos)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2.0 ** n), pos, n


def chance_jaccard(k, n):
    """EXACT expected Jaccard of two independent random k-subsets of n: the intersection size is
    hypergeometric, so E[J] = sum_i P(|A&B| = i) * i / (2k - i).

    The tempting closed form k / (2n - k) is a ratio of expectations, not the expectation of the
    ratio, and it undershoots by up to ~23% at the small k this grid produces (k=2 of 48) -- which
    would make an observed overlap look further above chance than it is."""
    from math import comb
    if k < 1 or k > n:
        return None
    tot = comb(n, k)
    return sum(comb(k, i) * comb(n - k, k - i) / tot * i / (2 * k - i) for i in range(k + 1))


def set_overlap(j, m, n_total):
    """(mean pairwise Jaccard of the per-language head sets at size m, chance for random k-sets).

    The test of the reading that specificity ~ 0 because every language's top set is the SAME set of
    generally important heads: then own ~ others by construction, and `vs random` > 0 says only that
    the screen finds important heads. An overlap far above chance is shared machinery, not
    per-language machinery."""
    per = (j.get("lang") or {}).get(str(m)) or {}
    sets = [set(d["set"]) for d in per.values()]
    if len(sets) < 2 or not n_total:
        return None, None
    js = [len(a & b) / len(a | b) for i, a in enumerate(sets) for b in sets[i + 1:] if a | b]
    return (float(np.mean(js)) if js else None), chance_jaccard(len(sets[0]), int(n_total))


def report_joint(M, models=None):
    """The INRIA ablation protocol ported: top-m% head sets zeroed jointly, scored by retrieval.

    Both columns must be positive for a language to own machinery. `specificity` <= 0 says the set
    is just important heads, not that language's heads. `vs random` <= 0 says the damage is about
    how many heads were removed rather than which ones -- the control that makes the whole thing
    interpretable, and the one the single-head version never had."""
    print("\n  JOINT ABLATION -- per-language head sets, retrieval readout, random-set control")
    for n in (models or MAIN_MODELS):
        r = M.get(n)
        j = (r or {}).get("joint")
        if not j:
            continue
        clean = j.get("clean") or {}
        tot = r.get("total_heads")
        print(f"\n    {n}  (pool {j.get('pool_n')} parallel sentences, clean mean P@1 "
              f"{np.mean(list(clean.values())):.4f})")
        print(f"    {'m':>6}{'heads':>7}{'own drop':>10}{'others':>9}{'specificity':>13}{'p':>7}"
              f"{'random':>9}{'vs random':>11}{'p':>7}{'overlap':>9}{'chance':>8}")
        grid = j.get("m_grid") or M_GRID
        for m in grid:
            st = joint_stats(r, m)
            if not st:
                print(f"    {m:>6.0%}{'-':>7}{'(unfinished)':>10}")
                continue
            own = float(np.mean([v["own"] for v in st.values()]))
            oth = float(np.mean([v["others"] for v in st.values()]))
            rnd = float(np.mean([v["rand"] for v in st.values()]))
            p_s = _sign_p([v["own"] - v["others"] for v in st.values()])[0]
            p_r = _sign_p([v["own"] - v["rand"] for v in st.values()])[0]
            ov, ch = set_overlap(j, m, tot)
            nh = next(iter(st.values()))["n_heads"]
            fp = lambda x: f"{x:>7.3f}" if x is not None else f"{'-':>7}"          # noqa: E731
            print(f"    {m:>6.0%}{nh:>7}{own:>+10.4f}{oth:>+9.4f}{own - oth:>+13.4f}{fp(p_s)}"
                  f"{rnd:>+9.4f}{own - rnd:>+11.4f}{fp(p_r)}"
                  + (f"{ov:>9.3f}{ch:>8.3f}" if ov is not None else f"{'-':>9}{'-':>8}"))
        # the per-language rows at the largest set, where effects are biggest -- averaging over
        # languages is the same too-early collapse the screen used to make, and it hides whether a
        # mean comes from one language or ten
        st = joint_stats(r, max(grid))
        if st:
            print(f"      per language at m={max(grid):.0%}:")
            print(f"      {'lang':>6}{'own drop':>10}{'others':>9}{'specificity':>13}"
                  f"{'random':>9}{'vs random':>11}")
            for l in sorted(st):
                v = st[l]
                print(f"      {l:>6}{v['own']:>+10.4f}{v['others']:>+9.4f}"
                      f"{v['own'] - v['others']:>+13.4f}{v['rand']:>+9.4f}"
                      f"{v['own'] - v['rand']:>+11.4f}")
    print("\n    specificity > 0: ablating L's heads hurts L more than the other nine.")
    print("    vs random > 0:   it hurts L more than removing the same NUMBER of arbitrary heads.")
    print("    Both positive is the claim; either one at or below 0 and there are no language heads.")
    print("    p: exact two-sided sign test over the 10 languages (only 9/10 or 10/10 reach p < 0.05).")
    print("    overlap: mean pairwise Jaccard of the per-language sets. Far above chance means every")
    print("    language picked the SAME heads -- which forces specificity to ~0 by construction and")
    print("    makes `vs random` a statement about the screen finding important heads, not language.")


def report_matrix(M, full=False, models=None):
    """Which winner heads are language-IDENTITY heads -- and so worth steering -- and which are
    merely sensitive. `full` prints each 10 x 10 matrix (row = donor, column = target centroid)."""
    print("\n  DONOR x TARGET MATRIX on the winner heads -- identity head or sensitivity head?")
    cand = []
    for n in (models or MAIN_MODELS):
        mx = ((M.get(n) or {}).get("matrix")) or {}
        rows, langs = mx.get("heads") or [], mx.get("langs") or []
        if not rows:
            continue
        print(f"\n    {n}" + ("   (the top-SCREEN head is itself a winner: leans sensitivity)"
                             if mx.get("top_screen_is_winner") else ""))
        print(f"    {'head':>8}{'role':>11}{'best@':>7}{'screen':>9}{'diag hits':>11}{'p':>8}"
              f"{'diag excess':>13}{'verdict':>14}")
        for r_ in rows:
            ident = r_["p"] < 0.05 and r_["diag_excess"] > 0
            verdict = "IDENTITY" if ident else "not selective"
            if ident:
                cand.append((n, r_))
            print(f"    {'b' + str(r_['block']) + 'h' + str(r_['head']):>8}{r_['role']:>11}"
                  f"{r_['wins']:>7}{r_['screen']:>9.4f}{str(r_['diag_hits']) + '/' + str(len(langs)):>11}"
                  f"{r_['p']:>8.4f}{r_['diag_excess']:>+13.5f}{verdict:>14}")
            if full:
                A = np.asarray(r_["M"], float)
                # every cell 9 wide, the row maximum bracketed IN that width, so columns stay aligned
                print(f"      {'donor':>6} " + "".join(f"{t:>9}" for t in langs))
                for a, l in enumerate(langs):
                    top = int(np.argmax(A[a]))
                    print(f"      {l:>6} " + "".join(
                        f"[{A[a, c]:+.4f}]" if c == top else f" {A[a, c]:+.4f} "
                        for c in range(len(langs))))
    print("\n    diag hits: donors whose row peaks on its own language (chance 1 per 10 rows);")
    print("    p: one-sided binomial, 4+/10 is p < 0.05. A flat matrix is a sensitivity head --")
    print("    it moves toward every centroid about equally under any patch.")
    if cand:
        print("\n    STEERING CANDIDATES (directionally selective):")
        for n, r_ in cand:
            print(f"      {n:14} b{r_['block']}h{r_['head']}  diag hits {r_['diag_hits']}/10  "
                  f"mean diagonal movement {r_['diag_mean']:+.5f}")
    else:
        print("\n    NO steering candidates: no winner head is directionally selective. The")
        print("    per-donor winners are sensitivity heads, not language-identity heads.")


def merge(per_lang=False):
    d = merge_parts(ANALYSIS, schema=SCHEMA)
    M = d["models"]
    print("\nEXP 6 — LANGUAGE HEADS (causal: a head is forced to emit another language's signal and the"
          "\n  pooled embedding is read). lang score = movement toward the donor language's centroid,"
          "\n  net of patching the head with its OWN mean, so generic damage is already subtracted.")
    for n in MAIN_MODELS:
        r = M.get(n)
        if not r:
            continue
        sc = np.array(r.get("screen") or [], float)
        pr = r.get("profile") or {}
        tot = r.get("total_heads")
        print(f"\n  {n}  ({r.get('n_blocks')} blocks x {r.get('n_heads')} heads = {tot}, d_kv={r.get('d_kv')})")
        if len(sc):
            print(f"    ablation screen: max 1-cos {sc.max():.4f}  median {np.median(sc):.4f}  "
                  f"top-decile {np.quantile(sc, 0.9):.4f}")
        if not pr.get("heads"):
            print("    no head-level structure to profile (see screen)")
            continue
        langs, keep = r["langs"], pr["heads"]
        prof = {l: np.array(v, float) for l, v in pr["lang"].items()}
        cont = np.array(pr["content"], float)
        spec = specificity(prof, langs)
        lm = np.max(np.stack([prof[l] for l in langs]), 0)
        print(f"    profiled {len(keep)}/{tot} heads ({len(keep)/tot:.0%})   "
              f"max language score {lm.max():+.4f}   max content score {cont.max():+.4f}")
        gen = [j for j in keep if not np.isnan(spec[j]) and spec[j] > 0.7]
        sp = [j for j in keep if not np.isnan(spec[j]) and spec[j] < 0.3]
        print(f"    of those: {len(gen)} language-GENERAL (respond to every donor language), "
              f"{len(sp)} language-SPECIFIC (respond to one)")
        ov = overlap_stats(prof, langs)
        if ov:
            print(f"    is a language's head set a subset of {REF.upper()}'s, more than of a typical "
                  f"other language's?")
            print(f"    {'lang':>6}{'own|ref':>10}{'typical':>10}{'EXCESS':>9}")
            for l in sorted(ov):
                v = ov[l]
                print(f"    {l:>6}{v['own']:>10.3f}{(v['typical'] if v['typical'] is not None else float('nan')):>10.3f}"
                      f"{(v['excess'] if v['excess'] is not None else float('nan')):>+9.3f}")
            ex = [v["excess"] for v in ov.values() if v["excess"] is not None]
            print(f"    {'MEAN':>6}{'':>10}{'':>10}{np.mean(ex):>+9.3f}"
                  f"   (> 0 => {REF} is the general language mechanism)")
    print("\n  reading: heads whose ablation moves nothing carry nothing; a language-GENERAL head "
          "carries\n  'which language' for every language, a SPECIFIC one for its own. The EXCESS is the "
          "English\n  claim -- positive means other languages' language heads are English's heads.")
    report_joint(M)
    report_matrix(M, full=per_lang)
    if per_lang:
        report_per_lang(M)


# ----------------------------------------------------------------------------------------------
def _selftest():
    # top_heads is a FRACTION, so models with different head counts are comparable
    assert top_heads([0.1, 0.9, 0.5, 0.2], 0.5) == {1, 2}
    assert top_heads([0.1, 0.9, 0.5, 0.2], 0.01) == {1}, "always keeps at least one"

    # toward(): a patched embedding pulled onto the target reads +, pushed off reads -
    t = np.array([1.0, 0.0], np.float32)
    zc = l2norm(np.array([[0.0, 1.0]], np.float32))
    assert abs(toward(l2norm(np.array([[1.0, 0.0]], np.float32)), zc, t) - 1.0) < 1e-6
    assert abs(toward(zc, zc, t)) < 1e-6

    # specificity: a head responding to ONE language is specific, one responding to all is general
    L = ["a", "b", "c", "d"]
    prof = {"a": [1.0, 1.0], "b": [0.0, 1.0], "c": [0.0, 1.0], "d": [0.0, 1.0]}
    s = specificity(prof, L)
    assert s[0] < 0.3 < 0.9 < s[1], s

    # overlap: language sets that ARE the reference's score excess > 0; disjoint ones < 0
    nh = 20
    def mk(idx):
        v = np.zeros(nh); v[list(idx)] = 1.0; return v.tolist()
    shared = {"en": mk(range(2)), "te": mk(range(2)), "bn": mk(range(2)), "zh": mk([5, 6])}
    ov = overlap_stats(shared, list(shared), frac=0.1)
    assert ov["te"]["own"] == 1.0 and ov["te"]["excess"] > 0, ov["te"]
    assert ov["zh"]["own"] == 0.0 and ov["zh"]["excess"] <= 0, ov["zh"]
    # the shape a real hub would take: English's set is the UNION of one head from each language,
    # so every language overlaps English and none overlaps another. This is the pattern the excess
    # exists to detect -- note that "every language has the SAME set" gives zero excess, correctly,
    # because then no language is privileged over any other.
    # every set must hold exactly k = round(frac * n_heads) heads, or ties fill it from the zeros
    # and manufacture overlaps that are not in the fixture.
    union = {"en": mk([1, 3, 5]), "te": mk([0, 1, 10]),
             "bn": mk([2, 3, 11]), "zh": mk([4, 5, 12])}
    ovu = overlap_stats(union, list(union), frac=0.15)
    assert all(v["own"] > 0 and v["typical"] == 0.0 for v in ovu.values()), ovu
    assert np.mean([v["excess"] for v in ovu.values()]) > 0.3, ovu
    own_lang = {"en": mk([0, 1]), "te": mk([2, 3]), "bn": mk([4, 5]), "zh": mk([6, 7])}
    ov2 = overlap_stats(own_lang, list(own_lang), frac=0.1)
    assert all(v["own"] == 0.0 for v in ov2.values()), ov2
    assert abs(np.mean([v["excess"] for v in ov2.values()])) < 1e-9, "disjoint sets -> no excess"
    # ---- joint-ablation helpers
    # lang_sets must rank INSIDE a language. Head 7 is enormous for te and nothing elsewhere; the
    # averaged screen buries it at 0.9/10 = 0.09, below head 0's flat 0.10, so the old scalar
    # ranking drops it and the per-language ranking must not.
    tenl = ["te", "bn", "en", "sw", "yo", "am", "ha", "rw", "zh", "ar"]
    sl = np.full((10, 10), 0.01)                   # 10 heads x 10 languages
    sl[0, :] = 0.10                                # a head that matters equally everywhere
    sl[7, 0] = 0.90                                # te-only, and nine times bigger than head 0
    ls = lang_sets(sl.tolist(), tenl, 0.1)
    assert ls["te"] == [7], f"per-language ranking missed the te-only head: {ls}"
    assert ls["bn"] == [0], ls
    # head 7 averages (0.90 + 9x0.01)/10 = 0.099, just UNDER head 0's flat 0.10 -- so the scalar
    # screen ranks the te-only head below a head it dwarfs, which is the bias being fixed
    assert np.argmax(sl.mean(1)) == 0, "fixture must be one the AVERAGED screen gets wrong"

    rs = random_sets(50, 6, 4, seed=0)
    assert len(rs) == 4 and all(len(set(s)) == 6 for s in rs), "controls must be distinct k-sets"
    assert all(max(s) < 50 for s in rs)
    assert rs == random_sets(50, 6, 4, seed=0), "same seed must redraw the same controls"

    # align_scores: identical embeddings across languages retrieve perfectly; shuffling one breaks it
    base = l2norm(np.random.default_rng(0).normal(size=(40, 8)).astype(np.float32))
    assert abs(align_scores({"a": base, "b": base}, ["a", "b"])["a"] - 1.0) < 1e-9
    assert align_scores({"a": base, "b": base[::-1]}, ["a", "b"])["a"] < 0.2

    # joint_stats arithmetic, and the None it must return on an unfinished grid point
    rec = {"joint": {"clean": {"te": 0.9, "bn": 0.9}, "m_grid": [0.05],
                     "lang": {"0.05": {"te": {"set": [1, 2], "align": {"te": 0.5, "bn": 0.88}}}},
                     "rand": {"0.05": [{"set": [3, 4], "align": {"te": 0.86, "bn": 0.87}}]}}}
    st = joint_stats(rec, 0.05)
    assert abs(st["te"]["own"] - 0.4) < 1e-9 and abs(st["te"]["others"] - 0.02) < 1e-9, st
    assert abs(st["te"]["rand"] - 0.04) < 1e-9, st
    assert joint_stats(rec, 0.10) is None, "an unfinished grid point must report as unfinished"

    # chance_jaccard is EXACT: checked against 20k-draw simulations at the grid's real set sizes,
    # including the small-k cases where the closed-form ratio of expectations is off by up to 23%
    for k_, n_, sim in ((2, 48, 0.0277), (4, 72, 0.0322), (22, 216, 0.0548), (38, 384, 0.0528)):
        cj = chance_jaccard(k_, n_)
        assert abs(cj - sim) / sim < 0.03, f"chance_jaccard({k_},{n_})={cj:.4f} vs simulated {sim}"
    assert chance_jaccard(0, 10) is None and chance_jaccard(11, 10) is None
    # set_overlap: identical sets read 1, disjoint read 0
    same = {"joint": {"lang": {"0.1": {l: {"set": [1, 2, 3]} for l in ("a", "b", "c")}}}}
    assert set_overlap(same["joint"], 0.1, 30)[0] == 1.0, "identical sets must overlap fully"
    apart = {"lang": {"0.1": {"a": {"set": [1, 2]}, "b": {"set": [3, 4]}, "c": {"set": [5, 6]}}}}
    assert set_overlap(apart, 0.1, 30)[0] == 0.0, "disjoint sets must not overlap"

    # _sign_p: 10/10 of one sign is significant, an even split is not, zeros are dropped
    assert _sign_p([1] * 10)[0] < 0.01 and _sign_p([-1] * 10)[0] < 0.01
    assert _sign_p([1, -1] * 5)[0] == 1.0
    assert _sign_p([0, 0, 1])[2] == 1, "zeros must not count toward n"
    assert _sign_p([1] * 8 + [-1] * 2)[0] > 0.05, "8/10 is NOT significant; the report says so"

    # ---- donor x target matrix
    # toward_each is the per-row form of toward: its mean must BE toward
    rz = np.random.default_rng(3)
    zc_ = l2norm(rz.normal(size=(20, 6)).astype(np.float32))
    zp_ = l2norm(rz.normal(size=(20, 6)).astype(np.float32))
    tg_ = rz.normal(size=6).astype(np.float32)
    assert abs(float(toward_each(zp_, zc_, tg_).mean()) - toward(zp_, zc_, tg_)) < 1e-6

    # an IDENTITY head: every donor moves the embedding toward its own language
    ident = np.full((10, 10), 0.001) + np.eye(10) * 0.01
    si = matrix_stats(ident)
    assert si["diag_hits"] == 10 and si["p"] < 1e-9 and si["diag_excess"] > 0, si
    # a SENSITIVITY head: large movement toward every centroid, no preference -> not selective,
    # even though its diagonal is as large as the identity head's (the case the profile missed)
    sens = 0.011 + rz.normal(size=(10, 10)) * 1e-4
    ss = matrix_stats(sens)
    assert ss["p"] > 0.05 and abs(ss["diag_excess"]) < 1e-3, ss
    assert np.diag(sens).mean() > np.diag(ident).mean() - 0.001, "fixture: same-size diagonals"
    # the binomial tail is exact: P(X >= 4 | n=10, p=0.1) = 0.012795...
    four = np.full((10, 10), 0.0)
    for i in range(10):
        four[i, i if i < 4 else (i + 1) % 10] = 1.0          # exactly 4 rows peak on the diagonal
    assert matrix_stats(four)["diag_hits"] == 4
    assert abs(matrix_stats(four)["p"] - 0.01280) < 1e-4, matrix_stats(four)["p"]

    # winner_heads: ranked by donors won, a screen reference added only when it is not a winner
    langs10 = ["te", "bn", "en", "sw", "yo", "am", "ha", "rw", "zh", "ar"]
    lang_prof = {l: [0.0] * 12 for l in langs10}
    for l in ("te", "bn", "en"):
        lang_prof[l][5] = 1.0                                  # head 5 wins three donors
    for l in ("sw", "yo"):
        lang_prof[l][2] = 1.0                                  # head 2 wins two
    for k, l in enumerate(("am", "ha", "rw", "zh", "ar")):
        lang_prof[l][6 + k] = 1.0                              # the rest win one each
    fx = {"langs": langs10, "profile": {"heads": list(range(12)), "lang": lang_prof},
          "screen": [0.0] * 12}
    fx["screen"][9] = 1.0                                      # most sensitive head: not a winner
    wh = winner_heads(fx)
    assert [(j, r_) for j, r_, _ in wh] == [(5, "winner"), (2, "winner"), (9, "reference")], wh
    fx["screen"] = [0.0] * 12
    fx["screen"][5] = 1.0                                      # now the top-screen head IS a winner
    assert all(r_ == "winner" for _, r_, _ in winner_heads(fx)), "no reference when it is a winner"
    assert winner_heads({"profile": {}}) == [], "an empty profile has no winners"

    print("selftest OK: fractional head sets, movement readout, generality vs specificity, "
          "reference-overlap excess (shared sets positive, per-language sets zero), "
          "per-language ranking recovers a head the averaged screen buries, seeded controls, "
          "retrieval readout, joint-ablation arithmetic")


def main():
    utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results/retrieval_bgem3.json")
    ap.add_argument("--ckpt-dir", dest="ckpt_dir", default="checkpoints")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", default=None)
    ap.add_argument("--n-sent", dest="n_sent", type=int, default=200)
    ap.add_argument("--screen-frac", dest="screen_frac", type=float, default=0.25,
                    help="fraction of heads, ranked by ablation effect, to profile")
    ap.add_argument("--smoke", action="store_true",
                    help="8 recipients, verify the hook path end to end, then exit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--per-lang", dest="per_lang", action="store_true",
                    help="--merge: add the per-donor-language breakdown")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if a.merge:
        return merge(a.per_lang)
    names = [a.only] if a.only else [n for n in MAIN_MODELS if n in models_in(a.results)[0]]
    for n in names:
        run_one(n, a.results, a.ckpt_dir, a.device, a.n_sent, a.seed, a.screen_frac, a.smoke)


if __name__ == "__main__":
    main()
