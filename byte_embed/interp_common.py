"""Shared plumbing for the mechanistic-interpretability analyses (`byte_embed/interp_*.py`).

Everything here operates on the TRAINED students (no retraining) and must reproduce the exact
encoding path the reported numbers came from: `tokenize_like_forward` is a verbatim copy of
`ByteStudent.forward`'s char-first truncation (`text[:MAX_CHARS]`, then a token cap of 4x chars) —
deviating would silently break the content-fairness argument the whole byte-vs-subword comparison
rests on. `_pool_like_forward` replicates the trained attention pooler so per-layer extraction and the
final embedding come out of ONE forward pass.

Geometry helpers are pure numpy (LEACE and mean-difference erasure, effective rank, alignment /
uniformity, language probes) so they self-test on a torch-less machine:

  python -m byte_embed.interp_common --selftest
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

from common.eval import l2norm

MAIN_MODELS = ["byte-small", "subword-small", "byte-base", "subword-base",
               "byte-large", "subword-large"]
SCRIPT = {"te": "Telu", "bn": "Beng", "am": "Ethi", "zh": "Hans", "ar": "Arab",
          "sw": "Latn", "yo": "Latn", "ha": "Latn", "rw": "Latn", "en": "Latn"}
LATIN = [l for l, s in SCRIPT.items() if s == "Latn"]
FLORES_CACHE = "flores_devtest_10.json"


# ----------------------------------------------------------------------------------------------
# results / files
# ----------------------------------------------------------------------------------------------
def part_path(analysis, model):
    return Path(f"results/interp_{analysis}_part_{model}.json")


def merged_path(analysis):
    return Path(f"results/interp_{analysis}.json")


def read_json(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def write_json(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def utf8_stdout():
    """Windows consoles/pipes default to cp1252, which cannot print the tables' Δ/→/— glyphs
    (SLURM/Linux is UTF-8 already); call first thing in every main()."""
    import sys
    for s in (sys.stdout, sys.stderr):
        if hasattr(s, "reconfigure"):
            try:
                s.reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass


def results_meta(results="results/retrieval_bgem3.json"):
    """Top-level run metadata (`langs`, `teacher`, `n_train`, ...) from the merged results file or,
    failing that, the first part file that carries it."""
    p = Path(results)
    for f in [p] + sorted(p.parent.glob(f"{p.stem}_part_*.json")):
        d = read_json(f) if f.exists() else None
        if d and d.get("langs") and d.get("n_train"):
            return d
    return {}


def models_in(results="results/retrieval_bgem3.json", main_only=True):
    """(models, teacher, teacher_dim) across the merged results + part files; main grid only by
    default (boundary arms are dropped — exp 5 explains the boundary result from the main bytes)."""
    from byte_embed.full_eval import _models_in
    models, teacher, tdim = _models_in(results)
    if main_only:
        models = {n: bm for n, bm in models.items() if n in MAIN_MODELS and not bm.get("boundary")}
    return models, teacher, tdim


def merge_parts(analysis):
    """Glob the analysis' part files into one dict keyed by model; write the merged file."""
    out = {"models": {}}
    for f in sorted(glob.glob(str(part_path(analysis, "*")))):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        out["models"][d["model"]] = d
    write_json(merged_path(analysis), out)
    return out


# ----------------------------------------------------------------------------------------------
# model loading + exact-forward replication
# ----------------------------------------------------------------------------------------------
def load_student(name, results="results/retrieval_bgem3.json", ckpt_dir="checkpoints",
                 device="cuda", pooling="attn"):
    """Load a trained student and RETURN THE OBJECT (reeval._load_enc returns only a closure).
    Full-state checkpoints carry optimizer+RNG+queue (~3x the model; byte-large ~10 GB), so map to
    CPU, keep only `model`, and free the rest before moving to the GPU. grad-checkpointing is
    disabled — it is inert at eval but breaks forward hooks and emits warnings."""
    import torch

    from byte_embed.model import ByteStudent

    models, teacher, tdim = models_in(results, main_only=False)
    if name not in models:
        raise SystemExit(f"{name!r} not in {results} (known: {sorted(models)})")
    bm = models[name]
    tsuf = ("" if teacher == "sonar" else f"_{teacher}") + \
           (f"_b-{bm['boundary']}" if bm.get("boundary") else "")
    cpath = Path(ckpt_dir) / f"{name}_{pooling}{tsuf}.pt"
    if not cpath.exists():
        print(f"  [interp] {name}: checkpoint {cpath} not found -> skip")
        return None
    student = ByteStudent(bm["backbone"], out_dim=tdim, pooling=pooling, grad_checkpoint=False)
    ck = torch.load(cpath, map_location="cpu", weights_only=False)
    student.load_state_dict(ck["model"])
    step = ck.get("step")
    del ck
    student.to(device).eval()
    print(f"  [interp] loaded {cpath.name} (step {step}, {bm.get('kind')}, {bm['backbone']})")
    return student, bm, tsuf


def make_encode(student, device="cuda", transform=None, post=None, batch_size=128):
    """encode_fn: list[str] -> np.ndarray [n, d] (L2-normalized), with an optional TEXT transform
    (e.g. romanization) applied first and an optional numpy POST map (e.g. an erasure projection)
    applied after — the same closure shape the eval battery expects (run_lowresource.py:152-154)."""
    def enc(xs):
        if transform is not None:
            xs = [transform(x) for x in xs]
        E = student.encode(xs, batch_size=batch_size, device=device)
        if post is not None:
            E = post(E)
        return l2norm(np.asarray(E, dtype=np.float32))
    return enc


def tokenize_like_forward(student, texts, device="cuda"):
    """VERBATIM copy of ByteStudent.forward's input path (model.py:80-84)."""
    if student.max_chars:
        texts = [t[:student.max_chars] for t in texts]
    return student.tok(texts, padding=True, truncation=True,
                       max_length=(student.max_chars * 4 if student.max_chars else 2048),
                       return_tensors="pt").to(device)


def _pool_like_forward(student, h, m):
    """Replicates ByteStudent.forward's pooling + projection on a final-layer state `h` [B, L, d]
    with mask `m` [B, L, 1] -> L2-normalized [B, out_dim]. (One forward pass serves both the
    per-layer extraction and the trained embedding.)"""
    import torch.nn.functional as F
    if student.pooling == "max":
        pooled = h.masked_fill(m == 0, -1e9).max(dim=1).values
    elif student.pooling == "attn":
        H = student.attn_heads
        hs = h.unflatten(-1, (H, h.size(-1) // H))
        scores = (hs * student.attn_q).sum(-1) / (h.size(-1) // H) ** 0.5
        scores = scores.masked_fill(m == 0, -1e9)
        alpha = scores.softmax(dim=1).unsqueeze(-1)
        pooled = (hs * alpha).sum(1).flatten(1)
    else:
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
    return F.normalize(student.proj(pooled), dim=-1)


def _select_layers(n_layers, layers):
    return list(range(n_layers)) if layers is None else [l for l in layers if 0 <= l < n_layers]


def _hidden_states(student, out):
    """`out.hidden_states` with the layout every per-layer analysis assumes, verified at runtime:
    len == num_layers + 1, index 0 = embedding output (input of block 0), 1..N-1 = outputs of blocks
    0..N-2 (pre-LN residual stream), N = final-LayerNorm output == last_hidden_state. Older
    T5Stack.forward builds exactly this tuple by hand (append before each block, then the post-LN
    state); newer transformers (5.1x) record block outputs with OutputRecorder, prepend block 0's
    input and tie the last entry to last_hidden_state — same layout, but assert rather than trust
    it across versions."""
    import torch
    hs = out.hidden_states
    cfg = student.enc.config
    n = getattr(cfg, "num_layers", None) or cfg.num_hidden_layers
    if len(hs) != n + 1 or not torch.equal(hs[-1], out.last_hidden_state):
        raise SystemExit(f"[interp] unexpected hidden_states layout: {len(hs)} entries for "
                         f"{n} layers, last==last_hidden_state={torch.equal(hs[-1], out.last_hidden_state)} "
                         f"(transformers layout changed — fix interp_common._hidden_states)")
    return hs


def layer_pooled(student, texts, device="cuda", batch_size=8, layers=None, final=True):
    """Masked-MEAN-pooled residual-stream state per layer -> (states float32 [n_sel_layers, n, d],
    layer_ids, final_emb [n, out_dim] | None). Layer 0 = embedding output; the last index is the
    final-LayerNorm output (== last_hidden_state). Intermediate T5 states are pre-LN residuals whose
    norms grow with depth — normalize per layer before any cross-layer comparison. Only pooled
    vectors leave the GPU, so byte-large (36 layers, up to 2048 positions) is fine at batch 4-8.
    Stored in float32, NOT fp16: T5/mT5 residual streams carry a few massive-activation dimensions
    (|x| ~ 1e4-1e5 in mT5-large) that overflow fp16 to inf and poison every downstream statistic."""
    import torch

    if not texts:
        raise ValueError("layer_pooled: no texts")
    chunks, finals, layer_ids = None, [], None
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            b = tokenize_like_forward(student, texts[i:i + batch_size], device)
            out = student.enc(**b, output_hidden_states=True)
            hs = _hidden_states(student, out)
            m = b["attention_mask"].unsqueeze(-1).float()
            if layer_ids is None:
                layer_ids = _select_layers(len(hs), layers)
                chunks = [[] for _ in layer_ids]
            denom = m.sum(1).clamp(min=1.0)
            for j, l in enumerate(layer_ids):
                chunks[j].append(((hs[l] * m).sum(1) / denom).float().cpu().numpy())
            if final:
                finals.append(_pool_like_forward(student, hs[-1], m).float().cpu().numpy())
    states = np.stack([np.concatenate(c, 0) for c in chunks], 0)
    return states, layer_ids, (np.concatenate(finals, 0) if final else None)


def layer_positions(student, texts, sel, device="cuda", layers=None, batch_size=4):
    """Per-POSITION residual states for selected positions only -> (dict layer -> float32 [N, d],
    index list [(text_idx, pos), ...] in row order; float32 for the same overflow reason as
    layer_pooled). Nothing but the selected rows leaves the GPU.
    `sel[i]` = positions to keep in text i (byte offsets for ByT5, token offsets for mT5); every
    position must lie inside the unpadded sequence."""
    import torch

    if not texts:
        raise ValueError("layer_positions: no texts")
    store, index, layer_ids, d = None, [], None, None
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            bt = texts[i:i + batch_size]
            b = tokenize_like_forward(student, bt, device)
            out = student.enc(**b, output_hidden_states=True)
            hs = _hidden_states(student, out)
            lens = b["attention_mask"].sum(1).tolist()
            if layer_ids is None:
                layer_ids = _select_layers(len(hs), layers)
                store, d = {l: [] for l in layer_ids}, hs[0].size(-1)
            for k in range(len(bt)):
                pos = [p for p in sel[i + k] if p < lens[k]]
                if len(pos) != len(sel[i + k]):
                    print(f"  [interp] warn: {len(sel[i + k]) - len(pos)} positions past the "
                          f"sequence end were dropped for text {i + k}")
                if not pos:
                    continue
                pt = torch.as_tensor(pos, device=device)
                for l in layer_ids:
                    store[l].append(hs[l][k].index_select(0, pt).float().cpu().numpy())
                index.extend((i + k, p) for p in pos)
    return {l: (np.concatenate(v, 0) if v else np.zeros((0, d), np.float32))
            for l, v in store.items()}, index


def byte_offsets(text, max_chars=None):
    """(byte index of the START of each character, total bytes) for the text the model actually
    sees. ByT5 position k == UTF-8 byte k (ids = byte + 3, right padding, </s> at n_bytes)."""
    if max_chars is None:
        try:
            from byte_embed.model import MAX_CHARS
            max_chars = MAX_CHARS
        except ImportError:            # torch-less machine (selftest): model.py's documented default
            max_chars = 512
    text = text[:max_chars]
    starts, n = [], 0
    for ch in text:
        starts.append(n)
        n += len(ch.encode("utf-8"))
    return starts, n


# ----------------------------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------------------------
def flores_parallel(langs=None, cache_dir="checkpoints"):
    """{lang: [1012 sentences]} — the FLORES-200 devtest table is one row per parallel sentence,
    so row i is the same content in every language. Asserts every language is present (never
    silently drops one, unlike data.load_flores_parallel)."""
    from byte_embed.config import FLORES_CODE, STUDY_LANGS
    langs = list(langs or STUDY_LANGS)
    cp = Path(cache_dir) / FLORES_CACHE
    cached = read_json(cp) or {}
    if all(l in cached for l in langs):
        return {l: cached[l] for l in langs}
    from datasets import load_dataset
    d = load_dataset("mteb/flores", "default", split="devtest")
    missing = [l for l in langs if FLORES_CODE.get(l) not in d.column_names]
    assert not missing, f"FLORES devtest lacks columns for {missing}"
    par = {l: list(d[FLORES_CODE[l]]) for l in langs}
    n = {len(v) for v in par.values()}
    assert len(n) == 1, f"unequal FLORES column lengths {n}"
    cached.update(par)
    write_json(cp, cached)
    return par


def sample_training_sentences(ckpt_dir="checkpoints", per_lang=1000, seed=0, langs=None,
                              results="results/retrieval_bgem3.json"):
    """Stratified sample of the cached (sentence, language, teacher target) triples the students were
    distilled on — free positive pairs for alignment-to-teacher. The sidecar tag is rebuilt from the
    results file's own metadata (run_lowresource.py tags targets '<langs joined by ->_<per-language
    floor>' with n_train = floor x len(langs)), so a stale sidecar from another language set is never
    picked up; the lexicographic glob is only a fallback when that exact file is absent. Both halves
    (.json + .npy) must exist — a half-synced cache raises SystemExit, which callers degrade on."""
    from byte_embed.teachers import load_cached_targets, targets_exist
    meta = results_meta(results)
    teacher = meta.get("teacher") or "bge-m3"
    prefix = f"teachertargets_{teacher}_"
    tag = None
    if meta.get("langs") and meta.get("n_train"):
        cand = f"{'-'.join(meta['langs'])}_{meta['n_train'] // len(meta['langs'])}"
        if targets_exist(ckpt_dir, teacher, cand):
            tag = cand
    if tag is None:
        sidecars = [s for s in sorted(glob.glob(os.path.join(ckpt_dir, prefix + "*.json")))
                    if targets_exist(ckpt_dir, teacher, Path(s).stem[len(prefix):])]
        if not sidecars:
            raise SystemExit(f"no complete {prefix}*.json/.npy pair in {ckpt_dir}")
        tag = Path(sidecars[-1]).stem[len(prefix):]
        print(f"  [interp] warn: no cached-targets tag from {results} metadata -> using {sidecars[-1]}")
    sents, sl, T = load_cached_targets(ckpt_dir, teacher, tag)
    sl = np.asarray(sl)
    rng = np.random.default_rng(seed)
    keep = []
    for lang in (langs or sorted(set(sl.tolist()))):
        idx = np.flatnonzero(sl == lang)
        keep.extend(rng.choice(idx, size=min(per_lang, len(idx)), replace=False).tolist())
    keep = sorted(keep)
    return [sents[i] for i in keep], sl[keep].tolist(), np.asarray(T)[keep]


# ----------------------------------------------------------------------------------------------
# geometry (pure numpy)
# ----------------------------------------------------------------------------------------------
def flores_xling(emb, langs=None, k=10):
    """Cross-lingual retrieval on FLORES from precomputed L2-normed embeddings {lang: [n, d]}:
    query = sentence i of A, pool = all n sentences of B, gold = i. Returns
    {A: {B: {"p@1", "ndcg@10", "per_query"}}} for A != B (self-retrieval is trivially 1)."""
    langs = list(langs or emb)
    out = {}
    for a in langs:
        out[a] = {}
        for b in langs:
            if a == b:
                continue
            S = emb[a] @ emb[b].T
            gold = np.diag(S)
            rank = (S > gold[:, None]).sum(1)              # strictly-greater = 0-indexed rank
            nd = np.where(rank < k, 1.0 / np.log2(rank + 2), 0.0)
            out[a][b] = {"p@1": round(float((rank == 0).mean()), 4),
                         "ndcg@10": round(float(nd.mean()), 4),
                         "per_query": {f"s{i}": round(float(nd[i]), 4) for i in range(len(nd))}}
    return out


def _centroids(X, labels):
    labs = sorted(set(labels))
    labels = np.asarray(labels)
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(0)
    C = np.stack([X[labels == l].mean(0) for l in labs], 0)
    w = np.array([(labels == l).mean() for l in labs])
    return labs, mu, C, w


def fit_mean_diff(X, labels):
    """Orthogonal projection removing span{mu_k - mu} (plain centroid-difference erasure)."""
    _, mu, C, _ = _centroids(X, labels)
    D = (C - mu).T                                          # [d, K]
    U, s, _ = np.linalg.svd(D, full_matrices=False)
    r = int((s > 1e-8 * max(s.max(), 1e-12)).sum())
    Q = U[:, :r]
    P = np.eye(X.shape[1]) - Q @ Q.T
    return P.astype(np.float32), mu.astype(np.float32)


def fit_leace(X, labels, eps=1e-6):
    """LEACE (Belrose et al. 2023) closed form for a categorical concept: in whitened coordinates
    project out the span of the class-mean differences, then un-whiten:
        P = I - W+ Q Q^T W,  W = Sigma_XX^{-1/2},  Q = orth(W Sigma_XZ),  Sigma_XZ[:,k] = p_k (mu_k - mu).
    Guarantees no LINEAR classifier can recover the concept (linear guardedness) while moving the
    representations as little as possible. Apply with `apply_projection`."""
    _, mu, C, w = _centroids(X, labels)
    Xc = np.asarray(X, dtype=np.float64) - mu                # float64: W = S^{-1/2} is ill-conditioned
    S = Xc.T @ Xc / max(len(X) - 1, 1)
    lam, U = np.linalg.eigh(S)
    lam = np.maximum(lam, eps * lam.max())
    W = (U * lam ** -0.5) @ U.T
    Wp = (U * lam ** 0.5) @ U.T
    Sxz = ((C - mu) * w[:, None]).T                         # [d, K]
    M = W @ Sxz
    Uq, s, _ = np.linalg.svd(M, full_matrices=False)
    r = int((s > 1e-8 * max(s.max(), 1e-12)).sum())
    Q = Uq[:, :r]
    P = np.eye(X.shape[1]) - Wp @ Q @ Q.T @ W
    return P.astype(np.float32), mu.astype(np.float32)


def apply_projection(E, P, mu):
    return (mu + (np.asarray(E, dtype=np.float32) - mu) @ P.T).astype(np.float32)


def lang_probe(X, labels, seed=0, test_frac=0.2):
    """Linear language-ID probe accuracy (LogisticRegression max_iter=2000, C=10.0 — the
    common/eval.py:sib_probe convention) on a stratified seeded split. Features are standardized
    because residual-stream norms differ by orders of magnitude across layers."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    labels = np.asarray(labels)
    Xtr, Xte, ytr, yte = train_test_split(np.asarray(X, dtype=np.float32), labels,
                                          test_size=test_frac, random_state=seed, stratify=labels)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=10.0).fit(sc.transform(Xtr), ytr)
    return round(float(clf.score(sc.transform(Xte), yte)), 3)


def effective_rank(X):
    """Participation ratio (sum lam)^2 / sum lam^2 and entropy effective rank exp(H(lam/sum lam))
    of the covariance spectrum of X (centered). Scale-invariant."""
    Xc = np.asarray(X, dtype=np.float64)
    Xc = Xc - Xc.mean(0)
    s = np.linalg.svd(Xc, compute_uv=False)
    lam = s ** 2 / max(len(Xc) - 1, 1)
    lam = lam[lam > 0]
    p = lam / lam.sum()
    return {"participation_ratio": round(float(lam.sum() ** 2 / (lam ** 2).sum()), 2),
            "erank": round(float(np.exp(-(p * np.log(p)).sum())), 2)}


def alignment(X, Y, alpha=2):
    """Wang & Isola (2020) alignment: E ||f(x) - f(x+)||^alpha over positive pairs (rows paired)."""
    X, Y = l2norm(np.asarray(X, dtype=np.float32)), l2norm(np.asarray(Y, dtype=np.float32))
    return round(float((np.linalg.norm(X - Y, axis=1) ** alpha).mean()), 4)


def uniformity(X, t=2, n_pairs=20000, seed=0):
    """Wang & Isola (2020) uniformity: log E exp(-t ||f(x) - f(y)||^2) over random pairs
    (more negative = more uniformly spread on the hypersphere)."""
    X = l2norm(np.asarray(X, dtype=np.float32))
    n = len(X)
    rng = np.random.default_rng(seed)
    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)
    keep = i != j
    d2 = (np.linalg.norm(X[i[keep]] - X[j[keep]], axis=1) ** 2)
    return round(float(np.log(np.exp(-t * d2).mean())), 4)


# ----------------------------------------------------------------------------------------------
# selftest (pure numpy / sklearn)
# ----------------------------------------------------------------------------------------------
def _selftest():
    rng = np.random.default_rng(0)
    d, K, n = 64, 10, 200
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    lang_dirs, content_dir = Q[:, :K], Q[:, K]              # orthogonal directions
    labels = np.repeat(np.arange(K), n)
    content = rng.integers(0, 2, size=K * n)
    X = (rng.standard_normal((K * n, d)) * 0.3 + 2.0 * lang_dirs[:, labels].T
         + 3.0 * content[:, None] * content_dir[None, :])
    assert lang_probe(X, labels) > 0.95
    assert lang_probe(X, content) > 0.95
    for fit in (fit_leace, fit_mean_diff):
        P, mu = fit(X, labels)
        Xe = apply_projection(X, P, mu)
        acc = lang_probe(Xe, labels)
        assert acc <= 1.0 / K + 0.06, (fit.__name__, acc)
        assert lang_probe(Xe, content) > 0.95, fit.__name__   # content survives erasure
    iso = rng.standard_normal((5000, 32))
    assert effective_rank(iso)["participation_ratio"] > 28
    r1 = np.outer(rng.standard_normal(5000), rng.standard_normal(32)) + 1e-3 * rng.standard_normal((5000, 32))
    assert effective_rank(r1)["participation_ratio"] < 1.5
    U = l2norm(rng.standard_normal((500, 16)))
    assert alignment(U, U) == 0.0 and 1.5 < alignment(U, l2norm(rng.standard_normal((500, 16)))) < 2.5
    tight = l2norm(np.ones((300, 16)) + 1e-3 * rng.standard_normal((300, 16)))
    assert uniformity(tight) > -0.01 and uniformity(U) < -1.0
    assert byte_offsets("aé字") == ([0, 1, 3], 6)
    E = {"a": U[:100], "b": U[:100], "c": l2norm(rng.standard_normal((100, 16)))}
    xl = flores_xling(E)
    assert xl["a"]["b"]["p@1"] == 1.0 and xl["a"]["c"]["p@1"] < 0.1
    print("selftest OK: leace/mean-diff erase language to chance while content stays decodable; "
          "effective rank, alignment/uniformity, byte offsets, xling retrieval verified")


def main():
    utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        _selftest()


if __name__ == "__main__":
    main()
