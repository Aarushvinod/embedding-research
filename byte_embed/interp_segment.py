"""Exp 5 — emergent segmentation in ByT5's early layers (composition): does the byte student build
word-like units on its own?

MrT5 (ICLR 2025) showed a byte encoder consolidates orthography-adaptive units by encoder layers
1-3 when trained with a deletion gate; whether VANILLA retrieval-distilled byte students do so is
untested. We extract per-byte-position residual states and train linear probes for per-position
boundary labels: `tok_end` = last byte of a BGE-M3 (teacher-tokenizer) token; `word_end` = last
byte before whitespace (space-delimited languages only); `interior` = tok_end AND NOT word_end —
the non-trivial sub-word boundary, because a word boundary is trivially decodable from the 0x20
byte sitting in the input (the "spaces trap"); `random` = count-matched random positions (chance
calibration). Chinese has no whitespace, so every boundary there had to be COMPUTED, not copied —
the cleanest test-bed. The candidate universe is character-final bytes only (a mid-character byte
can never be a boundary in a multi-byte script, and that would be a trivial signal). Reading rule
(pre-registered): interior boundaries decodable well above the random control by layers 1-3 =>
"redundancy" explanation of the boundary-injection null (markers added information the model
already had); chance-level everywhere incl. an MLP probe => "irrelevance" (segmentation isn't used
for this task at all). Teacher-boundary decodability also answers the "subword teacher biases the
byte student" reviewer risk directly. Byte models only.

  python -m byte_embed.interp_segment --only byte-small
  python -m byte_embed.interp_segment --only byte-small --langs en,zh --n-sent 50   # fast path
  python -m byte_embed.interp_segment --merge
  python -m byte_embed.interp_segment --selftest
"""
from __future__ import annotations

import argparse
import zlib

import numpy as np

from byte_embed.interp_common import (SCRIPT, byte_offsets, flores_parallel, layer_positions,
                                      load_student, merge_parts, models_in, part_path, read_json,
                                      write_json)

ANALYSIS = "segment"
LABELS = ("tok_end", "interior", "word_end", "random")
BYTE_MODELS = ["byte-small", "byte-base", "byte-large"]
NO_SPACES = {"zh"}


# ----------------------------------------------------------------------------------------------
# labels (pure python)
# ----------------------------------------------------------------------------------------------
def position_labels(text, cuts, lang, max_chars=512):
    """Per-byte-position label sets for one text. `cuts` = CHARACTER offsets of teacher-token ends
    (boundaries.boundary_cuts). Returns (labels: {name: set(byte_pos)} | None for undefined,
    universe: sorted list of character-final byte positions excluding the sentence-final one)."""
    starts, n_bytes = byte_offsets(text, max_chars)
    text = text[:max_chars]
    L = len(text)
    if L < 2:
        return None, []
    ends = [starts[c] - 1 for c in range(1, L)]           # last byte of char c-1, c = 1..L-1
    universe = sorted(set(ends))                           # sentence-final byte excluded (trivial)
    tok_end = {starts[c] - 1 for c in cuts if 0 < c < L}
    if lang in NO_SPACES:
        word_end = None
    else:
        word_end = {starts[c] - 1 for c in range(1, L) if text[c].isspace() and not text[c - 1].isspace()}
    interior = tok_end - word_end if word_end is not None else set(tok_end)
    rng = np.random.default_rng(zlib.crc32(text.encode("utf-8")))
    k = min(len(tok_end), len(universe))
    random = set(rng.choice(universe, size=k, replace=False).tolist()) if k else set()
    return {"tok_end": tok_end, "interior": interior, "word_end": word_end, "random": random}, universe


def build_dataset(texts, cuts_list, lang, rng, max_chars=512, per_text_cap=40):
    """Balanced positives/negatives per label -> (sel: positions per text, items: {label: [(text_idx,
    pos, y)]}). Negatives are drawn from the character-final universe minus that label's positives,
    count-matched; per_text_cap bounds the extraction cost."""
    sel, items = [], {l: [] for l in LABELS}
    for i, (t, cuts) in enumerate(zip(texts, cuts_list)):
        labs, universe = position_labels(t, cuts, lang, max_chars)
        keep = set()
        if labs is not None:
            for name in LABELS:
                pos = labs[name]
                if pos is None:
                    continue
                pos = sorted(pos)
                if len(pos) > per_text_cap // 2:
                    pos = sorted(rng.choice(pos, size=per_text_cap // 2, replace=False).tolist())
                neg_pool = [u for u in universe if u not in labs[name]]
                neg = sorted(rng.choice(neg_pool, size=min(len(pos), len(neg_pool)),
                                        replace=False).tolist()) if neg_pool and pos else []
                items[name].extend((i, p, 1) for p in pos)
                items[name].extend((i, p, 0) for p in neg)
                keep.update(pos); keep.update(neg)
        sel.append(sorted(keep))
    return sel, items


# ----------------------------------------------------------------------------------------------
# probes (numpy + sklearn)
# ----------------------------------------------------------------------------------------------
def probe_metrics(X, y, seed=0, mlp=False):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    y = np.asarray(y)
    if len(set(y.tolist())) < 2 or len(y) < 20:
        return None
    Xtr, Xte, ytr, yte = train_test_split(np.asarray(X, dtype=np.float32), y, test_size=0.2,
                                          random_state=seed, stratify=y)
    sc = StandardScaler().fit(Xtr)
    clf = (MLPClassifier(hidden_layer_sizes=(256,), max_iter=300, random_state=seed) if mlp
           else LogisticRegression(max_iter=2000, C=10.0))
    clf.fit(sc.transform(Xtr), ytr)
    pred = clf.predict(sc.transform(Xte))
    return {"acc": round(float((pred == yte).mean()), 3),
            "bacc": round(float(balanced_accuracy_score(yte, pred)), 3),
            "f1": round(float(f1_score(yte, pred)), 3), "n": int(len(y))}


def layer_plan(n_layers):
    """All layers for small/base; for byte-large (37 states) the early band densely + every 3rd."""
    if n_layers <= 20:
        return list(range(n_layers))
    return sorted(set([0, 1, 2, 3, 4] + list(range(7, n_layers, 3)) + [n_layers - 1]))


def run_one(name, results, ckpt_dir, device, langs=None, n_sent=None, seed=0):
    outp = part_path(ANALYSIS, name)
    res = read_json(outp) or {"model": name, "langs": {}}
    loaded = load_student(name, results, ckpt_dir, device)
    if loaded is None:
        return
    student, bm, _ = loaded
    if bm.get("kind") != "byte":
        print(f"  [segment] {name} is not a byte model -> skip")
        return
    from byte_embed.boundaries import _teacher_tok, boundary_cuts
    tok = _teacher_tok()
    par = flores_parallel(cache_dir=ckpt_dir)
    langs = langs or list(par)
    n_sent = n_sent or (300 if "large" in name else 500)
    rng = np.random.default_rng(seed)
    res.update(kind="byte", steps_run=bm.get("steps_run"))
    for lang in langs:
        if lang in res["langs"]:
            print(f"  {name}/{lang}: done -> skip")
            continue
        idx = rng.choice(len(par[lang]), size=min(n_sent, len(par[lang])), replace=False)
        texts = [par[lang][i] for i in idx]
        cuts_list = [boundary_cuts(t[:student.max_chars], tok) for t in texts]
        sel, items = build_dataset(texts, cuts_list, lang, rng, max_chars=student.max_chars)
        layers = layer_plan(student.enc.config.num_layers + 1)     # embeddings + every block
        feats, index = layer_positions(student, texts, sel, device=device, layers=layers,
                                       batch_size=(4 if "large" in name else 8))
        row_of = {k: r for r, k in enumerate(index)}
        out = {"n_sent": len(texts), "layers": []}
        for l in layers:
            entry = {"layer": int(l)}
            for lab in LABELS:
                it = [(row_of[(i, p)], y) for i, p, y in items[lab] if (i, p) in row_of]
                if not it:
                    entry[lab] = None
                    continue
                rows, y = zip(*it)
                entry[lab] = probe_metrics(feats[l][list(rows)], y, seed)
            out["layers"].append(entry)
        # MLP probe-strength check on layers 1-3 if the linear interior probe never clears random
        early = [e for e in out["layers"] if 1 <= e["layer"] <= 3 and e.get("interior") and e.get("random")]
        if early and all(e["interior"]["bacc"] < e["random"]["bacc"] + 0.05 for e in early):
            out["mlp_check"] = {}
            for e in early:
                it = [(row_of[(i, p)], y) for i, p, y in items["interior"] if (i, p) in row_of]
                rows, y = zip(*it)
                out["mlp_check"][str(e["layer"])] = probe_metrics(feats[e["layer"]][list(rows)], y,
                                                                  seed, mlp=True)
        res["langs"][lang] = out
        write_json(outp, res)
        e1 = next((e for e in out["layers"] if e["layer"] == 1), out["layers"][0])
        print(f"  {name}/{lang}: layer1 interior bacc={e1['interior']['bacc'] if e1.get('interior') else None}"
              f" random={e1['random']['bacc'] if e1.get('random') else None} -> saved")
        del feats
    print(f"  saved -> {outp}")


def merge():
    d = merge_parts(ANALYSIS)
    M = d["models"]
    print("\nEXP 5 — EMERGENT SEGMENTATION (balanced accuracy of per-byte-position boundary probes; "
          "`random` = chance control)")
    for n in BYTE_MODELS:
        r = M.get(n)
        if not r or not r.get("langs"):
            continue
        langs = list(r["langs"])
        print(f"\n  {n}  (mean over {len([l for l in langs if l not in NO_SPACES])} space-delimited langs"
              f"{' + zh separately' if 'zh' in langs else ''})")
        print(f"  {'layer':>6}{'tok_end':>9}{'interior':>10}{'word_end':>10}{'random':>8}"
              f"{'  | zh tok_end':>14}{'zh random':>11}")
        layer_ids = [e["layer"] for e in r["langs"][langs[0]]["layers"]]
        for l in layer_ids:
            vals = {lab: [] for lab in LABELS}
            zh = {}
            for lang in langs:
                e = next((x for x in r["langs"][lang]["layers"] if x["layer"] == l), None)
                if not e:
                    continue
                if lang in NO_SPACES:
                    zh = e
                    continue
                for lab in LABELS:
                    if e.get(lab):
                        vals[lab].append(e[lab]["bacc"])
            def m(lab):
                return f"{np.mean(vals[lab]):>9.3f}" if vals[lab] else f"{'—':>9}"
            zt = f"{zh['tok_end']['bacc']:>14.3f}" if zh.get("tok_end") else f"{'—':>14}"
            zr = f"{zh['random']['bacc']:>11.3f}" if zh.get("random") else f"{'—':>11}"
            print(f"  {l:>6}{m('tok_end')}{m('interior'):>10}{m('word_end'):>10}{m('random')[1:]:>8}{zt}{zr}")
        mlp = {lang: r["langs"][lang].get("mlp_check") for lang in langs if r["langs"][lang].get("mlp_check")}
        if mlp:
            print(f"  MLP probe check ran for: {sorted(mlp)} -> " +
                  ", ".join(f"{lang}: " + "/".join(f"L{k}={v['bacc']}" for k, v in mc.items())
                            for lang, mc in mlp.items()))
    print("\n  reading: interior >> random by layers 1-3 -> redundancy (markers were already known);"
          " ~random everywhere incl. MLP -> irrelevance (segmentation unused for retrieval).")


def _selftest():
    labs, uni = position_labels("ab cd", cuts=[2, 3], lang="en")
    # bytes: a0 b1 ' '2 c3 d4 ; char-final bytes excluding the last: {0,1,2,3}
    assert uni == [0, 1, 2, 3], uni
    assert labs["tok_end"] == {1, 2}, labs            # ends before chars 2 (' ') and 3 ('c')
    assert labs["word_end"] == {1}, labs               # 'b' precedes the space
    assert labs["interior"] == {2}, labs
    assert len(labs["random"]) == 2 and labs["random"] <= set(uni)
    labs_zh, _ = position_labels("字字字", cuts=[1, 2], lang="zh")
    assert labs_zh["word_end"] is None and labs_zh["tok_end"] == {2, 5}, labs_zh   # 3-byte chars
    rng = np.random.default_rng(0)
    sel, items = build_dataset(["ab cd", "ef gh"], [[2, 3], [2, 3]], "en", rng)
    assert all(len(s) > 0 for s in sel) and len(items["interior"]) == 4
    X = rng.standard_normal((400, 16)); y = (rng.random(400) > 0.5).astype(int)
    X[:, 0] += 3 * y                                     # decodable signal
    assert probe_metrics(X, y)["bacc"] > 0.9
    assert abs(probe_metrics(X, rng.permutation(y))["bacc"] - 0.5) < 0.15
    assert layer_plan(13) == list(range(13)) and 36 in layer_plan(37) and 1 in layer_plan(37)
    print("selftest OK: byte-position labels (incl. multibyte zh), balanced dataset, probes, layer plan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="results/retrieval_bgem3.json")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--only", default=None)
    ap.add_argument("--langs", default=None, help="comma list (default: all 10)")
    ap.add_argument("--n-sent", type=int, default=None, help="FLORES sentences/lang (500; 300 large)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return _selftest()
    if a.merge:
        return merge()
    langs = a.langs.split(",") if a.langs else None
    names = [a.only] if a.only else [n for n in BYTE_MODELS if n in models_in(a.results)[0]]
    for n in names:
        run_one(n, a.results, a.ckpt_dir, a.device, langs, a.n_sent, a.seed)


if __name__ == "__main__":
    main()
