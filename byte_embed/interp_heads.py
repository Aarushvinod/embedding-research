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


def top_heads(scores, frac=TOP_FRAC):
    """Indices of the top `frac` of heads by score. A FRACTION, not a count: head counts run 48 to
    384 across the grid, so a fixed k would compare a tenth of one model against a fortieth of
    another."""
    k = max(1, int(round(frac * len(scores))))
    return set(np.argsort(-np.asarray(scores, float))[:k].tolist())


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
def run_one(name, results, ckpt_dir, device, n_sent=200, seed=0, screen_frac=0.25, smoke=False):
    outp = part_path(ANALYSIS, name)
    res = read_json(outp) or {}
    if res.get("schema") != SCHEMA:
        res = {"schema": SCHEMA}
    if res.get("profile") and not smoke:
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

    # ---- screen: zero each head once, keep those whose ablation actually moves the embedding
    if "screen" not in res:
        sc = []
        for j, (b, h) in enumerate(heads):
            z = encode_patched(student, r_txt, b, h, d_kv, None, device, bs)
            sc.append(float(1.0 - (z * z_clean).sum(1).mean()))
            if j % max(len(heads) // 8, 1) == 0:
                print(f"    screen {j + 1}/{len(heads)}  1-cos={sc[-1]:.4f}")
        res["screen"] = sc
        write_json(outp, res)
    sc = np.array(res["screen"], float)
    keep = sorted(top_heads(sc, screen_frac))
    print(f"  screen: max 1-cos {sc.max():.4f}, median {np.median(sc):.4f}; "
          f"profiling the top {len(keep)}/{len(heads)}")
    if sc.max() < 1e-4:
        print("  NO head's ablation moves the embedding -> the representation is too distributed for "
              "head-level analysis. That is the result; not profiling.")
        res["profile"] = {}
        write_json(outp, res)
        return

    # ---- profile the survivors: self baseline, then language and content, net of it
    prof = {l: np.zeros(len(heads)) for l in langs}
    content = np.zeros(len(heads))
    selfc = np.zeros(len(heads))
    for n, j in enumerate(keep):
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
    res["profile"] = {"heads": keep, "lang": {l: prof[l].tolist() for l in langs},
                      "content": content.tolist(), "self_cost": selfc.tolist(),
                      "screen_frac": screen_frac, "n_recipients": len(rec)}
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
        print(f"\n    {n}  ({len(keep)} heads profiled, {contested} with no clear owner)")
        print(f"    {'donor':>6}{'best':>9}{'mean':>9}{'owns':>6}{'best@':>9}"
              f"{'vs content':>12}{'excess':>9}")
        for l in sorted(have):
            v = prof[l]
            j = int(np.argmax(v))
            b, h = divmod(j, nh)
            ratio = v[j] / cont[j] if abs(cont[j]) > 1e-9 else float("nan")
            ex = (ov.get(l) or {}).get("excess")
            at = f"b{b}h{h}"
            print(f"    {l:>6}{v[j]:>+9.4f}{float(np.mean(v[keep])):>+9.4f}{owner[l]:>6}{at:>9}"
                  f"{ratio:>12.2f}" + (f"{ex:>+9.3f}" if ex is not None else f"{'-':>9}"))
    print("\n  vs content < 1 at every donor: these are content heads. The language-head claim needs")
    print("  at least one donor whose best head moves further on LANGUAGE than on CONTENT.")


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
    print("selftest OK: fractional head sets, movement readout, generality vs specificity, "
          "reference-overlap excess (shared sets positive, per-language sets zero)")


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
