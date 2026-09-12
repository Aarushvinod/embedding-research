"""Shared plumbing for the interpretability experiments (`byte_embed/interp_english.py`,
`interp_script.py`, `interp_segment.py`).

Everything operates on the TRAINED students (no retraining) and must reproduce the exact encoding
path the reported numbers came from: `tokenize_like_forward` is a verbatim copy of
`ByteStudent.forward`'s char-first truncation (`text[:MAX_CHARS]`, then a token cap of 4x chars);
the loader below reproduced every stored training-time number within 0.005 nDCG@10 (identity check
of the first interp round). Activations are read and edited through forward hooks on the encoder
blocks (`record_blocks`, `block_hook`), so an edit at block b is followed by the remaining blocks,
the final LayerNorm, the trained pooler and the projection — a real intermediate-layer intervention,
not a post-hoc edit of the output embedding.

Pure-numpy helpers (rank-1 LEACE erasers, byte offsets, part-file IO) self-test without torch:

  python -m byte_embed.interp_common --selftest
"""
from __future__ import annotations

import argparse
import glob
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from common.eval import l2norm

MAIN_MODELS = ["byte-small", "subword-small", "byte-base", "subword-base",
               "byte-large", "subword-large"]
SCRIPT = {"te": "Telu", "bn": "Beng", "am": "Ethi", "zh": "Hans", "ar": "Arab",
          "sw": "Latn", "yo": "Latn", "ha": "Latn", "rw": "Latn", "en": "Latn"}
LATIN = [l for l, s in SCRIPT.items() if s == "Latn"]
FLORES_CACHE = "flores_devtest_10.json"
DEPTH_FRACS = (0.25, 0.5, 0.75, 1.0)


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


def models_in(results="results/retrieval_bgem3.json", main_only=True):
    """(models, teacher, teacher_dim) across the merged results + part files; main grid only by
    default (boundary arms are dropped)."""
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
        if "model" in d:
            out["models"][d["model"]] = d
    write_json(merged_path(analysis), out)
    return out


def stored_results(name, results="results/retrieval_bgem3.json"):
    """The stored training-time results entry (20k pools, per-query scores) — the `none` baseline
    every intervention is paired against; the loader reproduces it within 0.005 nDCG@10."""
    return models_in(results)[0].get(name)


# ----------------------------------------------------------------------------------------------
# model loading + exact-forward replication
# ----------------------------------------------------------------------------------------------
def load_student(name, results="results/retrieval_bgem3.json", ckpt_dir="checkpoints",
                 device="cuda", pooling="attn", pretrained_only=False):
    """Load a trained student and RETURN THE OBJECT (reeval._load_enc returns only a closure).
    Full-state checkpoints carry optimizer+RNG+queue (~3x the model; byte-large ~10 GB), so map to
    CPU, keep only `model`, and free the rest before moving to the GPU. grad-checkpointing is
    disabled — inert at eval, but it breaks forward hooks. `pretrained_only=True` returns the same
    architecture with the UNTOUCHED pretrained backbone (no checkpoint; head randomly initialised),
    the baseline for "what did distillation add"."""
    import torch

    from byte_embed.model import ByteStudent

    models, teacher, tdim = models_in(results, main_only=False)
    if name not in models:
        raise SystemExit(f"{name!r} not in {results} (known: {sorted(models)})")
    bm = models[name]
    tsuf = ("" if teacher == "sonar" else f"_{teacher}") + \
           (f"_b-{bm['boundary']}" if bm.get("boundary") else "")
    student = ByteStudent(bm["backbone"], out_dim=tdim, pooling=pooling, grad_checkpoint=False)
    if pretrained_only:
        student.to(device).eval()
        print(f"  [interp] pretrained backbone only: {bm['backbone']}")
        return student, bm, tsuf
    cpath = Path(ckpt_dir) / f"{name}_{pooling}{tsuf}.pt"
    if not cpath.exists():
        print(f"  [interp] {name}: checkpoint {cpath} not found -> skip")
        return None
    ck = torch.load(cpath, map_location="cpu", weights_only=False)
    student.load_state_dict(ck["model"])
    step = ck.get("step")
    del ck
    student.to(device).eval()
    print(f"  [interp] loaded {cpath.name} (step {step}, {bm.get('kind')}, {bm['backbone']})")
    return student, bm, tsuf


def n_blocks(student):
    cfg = student.enc.config
    return int(getattr(cfg, "num_layers", None) or cfg.num_hidden_layers)


def depth_blocks(n, fracs=DEPTH_FRACS):
    """Encoder block indices at the given depth fractions (block index b = output of block b;
    1.0 = the last block, whose output feeds the final LayerNorm)."""
    return sorted({max(0, int(round(f * n)) - 1) for f in fracs})


@contextmanager
def block_hook(student, block, fn):
    """Edit the output hidden states of encoder block `block` in place of the original: `fn` maps
    a float tensor [B, L, d] to a tensor of the same shape; every later block, the final LayerNorm,
    the pooler and the projection then run on the edited activations. T5Block returns a tuple whose
    first element is the hidden state (transformers <5 and >=5 alike), so the hook rebuilds the
    tuple around the edited tensor."""
    def hook(_mod, _args, out):
        if isinstance(out, tuple):
            return (fn(out[0]),) + tuple(out[1:])
        return fn(out)
    h = student.enc.encoder.block[block].register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


@contextmanager
def record_blocks(student, blocks):
    """Stash the output hidden states of the requested encoder blocks for the forward passes run
    inside the context (`rec.out[b]` -> tensor [B, L, d] of the most recent pass)."""
    rec = type("Rec", (), {})()
    rec.out, handles = {}, []

    def mk(b):
        def hook(_mod, _args, out):
            rec.out[b] = out[0] if isinstance(out, tuple) else out
        return hook
    for b in blocks:
        handles.append(student.enc.encoder.block[b].register_forward_hook(mk(b)))
    try:
        yield rec
    finally:
        for h in handles:
            h.remove()


def make_encode(student, device="cuda", transform=None, batch_size=128, hook=None):
    """encode_fn: list[str] -> np.ndarray [n, d] (L2-normalized), with an optional TEXT transform
    (e.g. romanization) applied first and an optional activation edit `hook=(block, fn)` installed
    for the forward pass — the closure shape the eval battery expects (run_lowresource.py:152-154).
    Queries and passages go through the same closure, so an edit applies to both sides."""
    def enc(xs):
        if transform is not None:
            xs = [transform(x) for x in xs]
        if hook is None:
            E = student.encode(xs, batch_size=batch_size, device=device)
        else:
            with block_hook(student, hook[0], hook[1]):
                E = student.encode(xs, batch_size=batch_size, device=device)
        return l2norm(np.asarray(E, dtype=np.float32))
    return enc


def tokenize_like_forward(student, texts, device="cuda"):
    """VERBATIM copy of ByteStudent.forward's input path (model.py:80-84)."""
    if student.max_chars:
        texts = [t[:student.max_chars] for t in texts]
    return student.tok(texts, padding=True, truncation=True,
                       max_length=(student.max_chars * 4 if student.max_chars else 2048),
                       return_tensors="pt").to(device)


def _select_layers(n_layers, layers):
    return list(range(n_layers)) if layers is None else [l for l in layers if 0 <= l < n_layers]


def _hidden_states(student, out):
    """`out.hidden_states` with the layout every per-layer analysis assumes, verified at runtime:
    len == num_layers + 1, index 0 = embedding output (input of block 0), 1..N-1 = outputs of blocks
    0..N-2 (pre-LN residual stream), N = final-LayerNorm output == last_hidden_state. Older
    T5Stack.forward builds exactly this tuple by hand; newer transformers (5.1x) record block
    outputs with OutputRecorder, prepend block 0's input and tie the last entry to
    last_hidden_state — same layout, but assert rather than trust it across versions."""
    import torch
    hs = out.hidden_states
    n = n_blocks(student)
    if len(hs) != n + 1 or not torch.equal(hs[-1], out.last_hidden_state):
        raise SystemExit(f"[interp] unexpected hidden_states layout: {len(hs)} entries for "
                         f"{n} layers, last==last_hidden_state={torch.equal(hs[-1], out.last_hidden_state)} "
                         f"(transformers layout changed — fix interp_common._hidden_states)")
    return hs


def layer_positions(student, texts, sel, device="cuda", layers=None, batch_size=4):
    """Per-POSITION residual states for selected positions only -> (dict layer -> float32 [N, d],
    index list [(text_idx, pos), ...] in row order). Layer indices follow `_hidden_states`
    (0 = embeddings). Float32, never fp16: T5/mT5 residual streams carry massive-activation
    dimensions (|x| ~ 1e5) that overflow fp16. `sel[i]` = positions to keep in text i (byte
    offsets for ByT5); every position must lie inside the unpadded sequence."""
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


def block_states(student, texts, blocks, per_text=10, device="cuda", batch_size=8, seed=0):
    """Randomly sampled per-position output states of the requested encoder BLOCKS (the hook
    points `block_hook` edits) -> ({block: float32 [N, d]}, index [(text_idx, pos)]). `per_text`
    positions per text, drawn uniformly from the unpadded sequence minus the final </s> slot."""
    import torch

    rng = np.random.default_rng(seed)
    store, index = {b: [] for b in blocks}, []
    with torch.inference_mode(), record_blocks(student, blocks) as rec:
        for i in range(0, len(texts), batch_size):
            bt = texts[i:i + batch_size]
            b = tokenize_like_forward(student, bt, device)
            student.enc(**b)
            lens = b["attention_mask"].sum(1).tolist()
            for k in range(len(bt)):
                n_real = max(int(lens[k]) - 1, 1)                     # exclude </s>
                pos = sorted(rng.choice(n_real, size=min(per_text, n_real), replace=False).tolist())
                pt = torch.as_tensor(pos, device=device)
                for bl in blocks:
                    store[bl].append(rec.out[bl][k].index_select(0, pt).float().cpu().numpy())
                index.extend((i + k, p) for p in pos)
    return {bl: np.concatenate(v, 0) for bl, v in store.items()}, index


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
    so row i is the same content in every language. Asserts every language is present."""
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


def flores_flat(par):
    """(texts, lang per text, sentence id per text) in language-major order."""
    langs = list(par)
    n = len(par[langs[0]])
    texts = [par[l][i] for l in langs for i in range(n)]
    lang = np.array([l for l in langs for _ in range(n)])
    sid = np.array([i for _ in langs for i in range(n)])
    return texts, lang, sid


# ----------------------------------------------------------------------------------------------
# rank-1 LEACE erasers (pure numpy)
# ----------------------------------------------------------------------------------------------
def whitener(X, eps=1e-6):
    """(W = Sigma^-1/2, W+ = Sigma^1/2, mu) of X in float64; eigenvalues floored at eps*max."""
    X = np.asarray(X, dtype=np.float64)
    mu = X.mean(0)
    Xc = X - mu
    S = Xc.T @ Xc / max(len(X) - 1, 1)
    lam, U = np.linalg.eigh(S)
    lam = np.maximum(lam, eps * lam.max())
    W = (U * lam ** -0.5) @ U.T
    Wp = (U * lam ** 0.5) @ U.T
    return W, Wp, mu


def rank1_eraser(W, Wp, mu, q):
    """LEACE for a BINARY concept is a rank-1 map: with q the unit direction of the class-mean
    difference in whitened coordinates, P = I - W+ q q^T W, i.e.
        x' = x - a * (b . (x - mu)),   a = W+ q,  b = W q.
    Returns (a, b, mu) as float32 vectors."""
    q = np.asarray(q, dtype=np.float64)
    q = q / max(np.linalg.norm(q), 1e-12)
    return (Wp @ q).astype(np.float32), (W @ q).astype(np.float32), np.asarray(mu, dtype=np.float32)


def leace_direction(W, X, mask):
    """Whitened class-mean difference for concept `mask` (True = positive class) — the direction a
    binary LEACE eraser removes (equivalently the whitened cross-covariance with the one-hot label)."""
    X = np.asarray(X, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    return W @ (X[mask].mean(0) - X[~mask].mean(0))


def fit_leace_binary(X, mask, eps=1e-6):
    """(a, b, mu) rank-1 LEACE eraser for the binary concept `mask` fitted on X."""
    W, Wp, mu = whitener(X, eps)
    return rank1_eraser(W, Wp, mu, leace_direction(W, X, mask))


def apply_rank1(X, a, b, mu):
    """numpy: x' = x - a (b . (x - mu)), rowwise."""
    X = np.asarray(X, dtype=np.float32)
    return X - np.outer((X - mu) @ b, a)


def rank1_torch_fn(a, b, mu, device):
    """The same map as a torch function on [B, L, d] tensors — the `fn` for `block_hook`."""
    import torch
    ta = torch.as_tensor(a, device=device, dtype=torch.float32)
    tb = torch.as_tensor(b, device=device, dtype=torch.float32)
    tmu = torch.as_tensor(mu, device=device, dtype=torch.float32)

    def fn(x):
        return x - ((x.float() - tmu) @ tb).unsqueeze(-1) * ta
    return fn


def fit_leace(X, labels, eps=1e-6):
    """General categorical LEACE (Belrose et al. 2023): P = I - W+ Q Q^T W with Q = orth(W Sigma_XZ).
    Kept for the selftest's cross-check of the rank-1 form and for reference; returns (P, mu)."""
    labels = np.asarray(labels)
    labs = sorted(set(labels.tolist()))
    W, Wp, mu = whitener(X, eps)
    X64 = np.asarray(X, dtype=np.float64)
    C = np.stack([X64[labels == l].mean(0) for l in labs], 0)
    w = np.array([(labels == l).mean() for l in labs])
    Sxz = ((C - mu) * w[:, None]).T
    M = W @ Sxz
    Uq, s, _ = np.linalg.svd(M, full_matrices=False)
    r = int((s > 1e-8 * max(s.max(), 1e-12)).sum())
    Q = Uq[:, :r]
    P = np.eye(X.shape[1]) - Wp @ Q @ Q.T @ W
    return P.astype(np.float32), mu.astype(np.float32)


# ----------------------------------------------------------------------------------------------
# selftest (pure numpy / sklearn)
# ----------------------------------------------------------------------------------------------
def _probe_acc(X, y, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    Xtr, Xte, ytr, yte = train_test_split(np.asarray(X, np.float32), np.asarray(y), test_size=0.3,
                                          random_state=seed, stratify=y)
    sc = StandardScaler().fit(Xtr)
    return LogisticRegression(max_iter=2000, C=10.0).fit(sc.transform(Xtr), ytr).score(sc.transform(Xte), yte)


def _selftest():
    rng = np.random.default_rng(0)
    d, n = 48, 1500
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    en = rng.random(n) < 0.3                                   # the concept to erase
    content = rng.integers(0, 2, size=n)                       # a second concept that must survive
    X = (rng.standard_normal((n, d)) * 0.4 + 2.0 * en[:, None] * Q[:, 0][None, :]
         + 2.0 * content[:, None] * Q[:, 1][None, :] + 0.7 * en[:, None] * Q[:, 2][None, :])
    assert _probe_acc(X, en) > 0.97 and _probe_acc(X, content) > 0.97
    a, b, mu = fit_leace_binary(X, en)
    Xe = apply_rank1(X, a, b, mu)
    assert _probe_acc(Xe, en) < 0.75, _probe_acc(Xe, en)      # concept gone (linear guardedness)
    assert _probe_acc(Xe, content) > 0.97                       # content untouched
    assert abs(Xe[en].mean(0) - Xe[~en].mean(0)).max() < 1e-6    # class means made identical
    P, mu2 = fit_leace(X, en.astype(int))                       # rank-1 form == general closed form
    Xg = (mu2 + (X.astype(np.float32) - mu2) @ P.T)
    assert np.abs(Xg - Xe).max() < 1e-3, np.abs(Xg - Xe).max()
    W, Wp, mu3 = whitener(X)
    ra, rb, _ = rank1_eraser(W, Wp, mu3, rng.standard_normal(d))  # random direction leaves the concept
    assert _probe_acc(apply_rank1(X, ra, rb, mu3), en) > 0.95
    assert depth_blocks(12) == [2, 5, 8, 11] and depth_blocks(8) == [1, 3, 5, 7] and depth_blocks(36) == [8, 17, 26, 35]
    assert byte_offsets("aé字") == ([0, 1, 3], 6)
    texts, lang, sid = flores_flat({"en": ["a", "b"], "te": ["c", "d"]})
    assert texts == ["a", "b", "c", "d"] and lang.tolist() == ["en", "en", "te", "te"] and sid.tolist() == [0, 1, 0, 1]
    print("selftest OK: rank-1 LEACE erases a binary concept to chance and equals the general closed "
          "form; random direction is inert; depth blocks, byte offsets, flores_flat verified")


def main():
    utf8_stdout()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        _selftest()


if __name__ == "__main__":
    main()
