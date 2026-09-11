"""Exp 1 — parameter allocation (capacity): why byte wins at small scale AND why it saturates.

ByT5's own hypothesis (Xue et al. 2022, §4.1): at matched size most of mT5-small's parameters are
"locked" in a 250k-row vocabulary table that any given example touches only sparsely, while ByT5
spends the same budget on dense layers activated by every example. If that is the mechanism, three
things should hold on OUR trained students: (i) byte's dense-parameter fraction >> subword's; (ii)
byte's vocabulary is fully utilized by the 10 study languages while subword touches a small fraction
of its rows; (iii) byte-small's activation spectra already fill their dimensions about as well as
byte-base's (nothing left for scale to unlock), whereas subword's effective dimensionality grows with
size. Reading rule (pre-registered): (i)+(ii) confirm the allocation story; (iii) is the saturation
signature.

  python -m byte_embed.interp_params --only subword-small          # one model (SLURM-friendly)
  python -m byte_embed.interp_params --merge                       # table over finished models
  python -m byte_embed.interp_params --selftest
"""
from __future__ import annotations

import argparse

import numpy as np

from byte_embed.interp_common import (MAIN_MODELS, effective_rank, flores_parallel, layer_pooled,
                                      load_student, merge_parts, models_in, part_path,
                                      sample_training_sentences, utf8_stdout, write_json)

ANALYSIS = "params"


def param_breakdown(student):
    """Clean split: vocabulary table / encoder blocks / final LN / projection head / attn pooler.
    (results-file `transformer_params` lumps proj+attn_q into "transformer" — this fixes that.)"""
    enc = student.enc
    vocab = enc.get_input_embeddings().weight.numel()
    blocks = sum(p.numel() for p in enc.encoder.block.parameters())
    final_ln = sum(p.numel() for p in enc.encoder.final_layer_norm.parameters())
    proj = sum(p.numel() for p in student.proj.parameters())
    attn_q = int(student.attn_q.numel()) if getattr(student, "attn_q", None) is not None else 0
    total = sum(p.numel() for p in student.parameters())
    parts = {"vocab": vocab, "blocks": blocks, "final_ln": final_ln, "proj": proj, "attn_q": attn_q}
    parts["other"] = total - sum(parts.values())          # relative-attention bias etc.
    out = {"total": total, **parts}
    out["frac"] = {k: round(v / total, 4) for k, v in parts.items()}
    out["dense_frac"] = round((total - vocab) / total, 4)
    out["vocab_rows"] = int(enc.get_input_embeddings().weight.shape[0])
    return out


def utilization_from_counts(counts, vocab_rows, rows_reachable=None):
    """Pure-numpy core: fraction of vocabulary rows ever hit (of all embedding rows, and of the rows a
    text can actually produce), and how many rows carry 99% of the token mass (a Zipf-shaped subword
    vocabulary is dominated by a small head)."""
    counts = np.asarray(counts, dtype=np.int64)
    rows_reachable = int(rows_reachable or vocab_rows)
    hit = int((counts > 0).sum())
    srt = np.sort(counts)[::-1]
    cum = np.cumsum(srt) / max(srt.sum(), 1)
    rows99 = int(np.searchsorted(cum, 0.99) + 1) if srt.sum() else 0
    return {"vocab_rows": int(vocab_rows), "rows_reachable": rows_reachable, "rows_hit": hit,
            "frac_hit": round(hit / vocab_rows, 4), "frac_hit_reachable": round(hit / rows_reachable, 4),
            "rows_for_99pct_mass": rows99,
            "frac_for_99pct_mass": round(rows99 / vocab_rows, 4), "n_tokens": int(srt.sum())}


def reachable_rows(student):
    """Embedding rows a text can produce. ByT5: 256 byte values + 3 specials = 259 (its 125 sentinel
    rows 259-383 never occur, and 13 byte values are illegal in UTF-8, so even a byte model tops out
    near 0.95 of these); subword: len(tokenizer) (mT5's 250,112 rows include 12 padding rows)."""
    return 259 if "ByT5" in type(student.tok).__name__ else len(student.tok)


def vocab_utilization(student, texts, batch=512):
    vocab_rows = int(student.enc.get_input_embeddings().weight.shape[0])
    counts = np.zeros(vocab_rows, dtype=np.int64)
    for i in range(0, len(texts), batch):
        chunk = [t[:student.max_chars] for t in texts[i:i + batch]] if student.max_chars else texts[i:i + batch]
        ids = student.tok(chunk, truncation=True,
                          max_length=(student.max_chars * 4 if student.max_chars else 2048))["input_ids"]
        flat = np.fromiter((t for row in ids for t in row), dtype=np.int64)
        counts += np.bincount(flat, minlength=vocab_rows)[:vocab_rows]
    return utilization_from_counts(counts, vocab_rows, reachable_rows(student))


def spectra_from_states(states, layer_ids, final_emb=None):
    """Pure-numpy core: per-layer effective dimensionality of mean-pooled states, on the RAW
    covariance (dominated by T5's few massive-activation dims) and on the per-dimension
    STANDARDIZED features (correlation structure — how many directions the layer really uses)."""
    rows = []
    for j, l in enumerate(layer_ids):
        X = np.asarray(states[j], dtype=np.float32)
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-6)
        rows.append({"layer": int(l), "raw": effective_rank(X), "std": effective_rank(Xs)})
    out = {"layers": rows}
    if final_emb is not None:
        out["final"] = effective_rank(np.asarray(final_emb, dtype=np.float32))
    return out


def run_one(name, results, ckpt_dir, device, n_flores=300, n_train=1000, seed=0):
    outp = part_path(ANALYSIS, name)
    if outp.exists():
        print(f"=== {ANALYSIS}/{name}: already done -> skip ===")
        return
    loaded = load_student(name, results, ckpt_dir, device)
    if loaded is None:
        return
    student, bm, _ = loaded
    rng = np.random.default_rng(seed)
    par = flores_parallel(cache_dir=ckpt_dir)
    idx = rng.choice(len(next(iter(par.values()))), size=n_flores, replace=False)
    flores_texts = [par[l][i] for l in par for i in idx]
    try:
        train_texts, _, _ = sample_training_sentences(ckpt_dir, per_lang=n_train, seed=seed,
                                                      results=results)
    except SystemExit as e:  # no cached targets on this machine -> FLORES only
        print(f"  [params] {e} -> utilization on FLORES only")
        train_texts = []

    res = {"model": name, "kind": bm.get("kind"), "backbone": bm.get("backbone"),
           "steps_run": bm.get("steps_run"), "params": param_breakdown(student)}
    res["vocab_util"] = vocab_utilization(student, flores_texts + train_texts)
    states, layer_ids, final_emb = layer_pooled(student, flores_texts, device=device,
                                                batch_size=(4 if "large" in name else 8))
    res.update(spectra_from_states(states, layer_ids, final_emb))
    write_json(outp, res)
    p, u = res["params"], res["vocab_util"]
    print(f"  saved {name}: dense_frac={p['dense_frac']}  vocab_hit={u['frac_hit']}  "
          f"final PR={res['final']['participation_ratio']}  -> {outp}")


def merge():
    d = merge_parts(ANALYSIS)
    M = d["models"]
    print("\nEXP 1 — PARAMETER ALLOCATION (dense fraction, vocab utilization, effective dimensionality)")
    print(f"  {'model':15}{'total(M)':>9}{'vocab%':>8}{'blocks%':>9}{'vocab hit':>10}{'reachable':>10}"
          f"{'rows99%':>9}{'final PR':>10}{'mid PR(std)':>12}{'last PR(std)':>13}")
    print("  (vocab hit = rows used / all embedding rows; reachable = / rows a text can produce)")
    for name in MAIN_MODELS:
        r = M.get(name)
        if not r:
            continue
        p, u, L = r["params"], r["vocab_util"], r["layers"]
        mid = L[len(L) // 2]["std"]["participation_ratio"]
        last = L[-1]["std"]["participation_ratio"]
        print(f"  {name:15}{p['total'] / 1e6:>9.0f}{100 * p['frac']['vocab']:>8.1f}"
              f"{100 * p['frac']['blocks']:>9.1f}{u['frac_hit']:>10.3f}"
              f"{u.get('frac_hit_reachable', u['frac_hit']):>10.3f}{u['frac_for_99pct_mass']:>9.4f}"
              f"{r['final']['participation_ratio']:>10.1f}{mid:>12.1f}{last:>13.1f}")
    print("  reading: byte dense%/vocab-hit >> subword confirms the allocation story; byte-small's"
          " PR ~ byte-base's is the saturation signature.")


def _selftest():
    rng = np.random.default_rng(0)
    counts = np.zeros(1000, dtype=np.int64)
    counts[:10] = 1000          # 10 rows carry all the mass
    u = utilization_from_counts(counts, 1000)
    assert u["frac_hit"] == 0.01 and u["rows_for_99pct_mass"] == 10, u
    assert utilization_from_counts(counts, 1000, rows_reachable=20)["frac_hit_reachable"] == 0.5
    states = np.stack([rng.standard_normal((400, 32)),                        # isotropic
                       np.outer(rng.standard_normal(400), rng.standard_normal(32))], 0)
    sp = spectra_from_states(states, [0, 1], final_emb=rng.standard_normal((400, 8)))
    assert sp["layers"][0]["raw"]["participation_ratio"] > 28
    assert sp["layers"][1]["raw"]["participation_ratio"] < 1.5
    assert sp["final"]["participation_ratio"] > 6
    print("selftest OK: utilization + spectra cores behave (isotropic ~d, rank-1 ~1)")


def main():
    utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results/retrieval_bgem3.json")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", default=None, help="model name (default: every finished main model)")
    ap.add_argument("--n-flores", type=int, default=300, help="FLORES sentences per language")
    ap.add_argument("--n-train", type=int, default=1000, help="cached training sentences per language")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if a.merge:
        return merge()
    names = [a.only] if a.only else [n for n in MAIN_MODELS if n in models_in(a.results)[0]]
    for n in names:
        run_one(n, a.results, a.ckpt_dir, a.device, a.n_flores, a.n_train, a.seed)


if __name__ == "__main__":
    main()
