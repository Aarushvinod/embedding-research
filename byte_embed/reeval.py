"""Re-eval already-trained models from their saved checkpoints — no retraining.

Rebuilds each student from the results JSON (kind/backbone/variant + the pooling you pass), loads its
checkpoint, and runs the eval battery (SIB / Belebele / FLORES / STS) + MIRACL + the new QA-retrieval,
writing the refreshed metrics back into the JSON. Use this to backfill the RAG-retrieval columns (and
get a holistic, consistent results file) for models you've already paid to train.

The orchestrators SKIP models already in results, so a plain re-run won't add new evals to old models —
this does. Pass the pooling the run used (byte/subword 50k = 'attn'; matched = 'mean'). MIRACL/QA
pools are cached on Drive, so the cost is just re-encoding.

  from byte_embed.reeval import reeval
  reeval('results/byte_lowresource_attn.json', pooling='attn')   # byte + subword
  reeval('results/matched.json',               pooling='mean')   # matched-transformer
  reeval('results/byte_lowresource_attn.json', pooling='attn', qa_only=True)  # fast: only add QA
"""
from __future__ import annotations

import json
from pathlib import Path


def _load_enc(name, bm, pooling, ckpt_dir, teacher_dim, device, tsuf=""):
    """Return an encode_fn for one saved model (rebuild + load checkpoint), or None if unavailable.
    `tsuf` is the per-teacher checkpoint suffix ('' for SONAR runs, '_bge-m3' for the retrieval run)."""
    import torch

    kind = bm.get("kind")
    if kind == "baseline":                                   # mE5 / LaBSE — no checkpoint, rebuild from id
        from sentence_transformers import SentenceTransformer
        mid = bm.get("backbone")
        prefix = "query: " if "e5" in (mid or "").lower() else ""
        mdl = SentenceTransformer(mid, device=device)
        return lambda xs, _m=mdl, _p=prefix: _m.encode([_p + x for x in xs], normalize_embeddings=True,
                                                        convert_to_numpy=True, show_progress_bar=False)

    if bm.get("variant"):                                    # matched-transformer student
        from byte_embed.matched import MatchedStudent
        student = MatchedStudent(bm["variant"], size=bm["size"], out_dim=teacher_dim, pooling=pooling)
        cpath = Path(ckpt_dir) / f"matched_{name}_{pooling}{tsuf}.pt"
    elif kind in ("byte", "subword"):
        from byte_embed.model import ByteStudent
        student = ByteStudent(bm["backbone"], out_dim=teacher_dim, pooling=pooling)
        cpath = Path(ckpt_dir) / f"{name}_{pooling}{tsuf}.pt"
    else:
        print(f"  [reeval] {name}: unknown kind {kind!r} -> skip")
        return None

    if not cpath.exists():
        print(f"  [reeval] {name}: checkpoint {cpath.name} not found -> skip")
        return None
    student.load_state_dict(torch.load(cpath, map_location=device)["model"])
    student.to(device).eval()
    print(f"  [reeval] {name}: loaded {cpath.name}")
    return lambda xs, _s=student: _s.encode(xs, device=device)


def reeval(results_path, pooling, ckpt_dir="checkpoints", device="cuda", langs=None,
           miracl_langs=("en", "zh", "ar", "te"), n_queries=250, distractors=20000,
           qa_only=False, out=None, benchmarks=None):
    """Re-eval every model in `results_path`. qa_only=True just adds the missing QA-retrieval (fast);
    otherwise also refreshes the battery + MIRACL (holistic). Saves after each model.
    `benchmarks` (tuple) restricts/extends the QA benchmarks and merges ADDITIVELY into the model's
    existing qa_retrieval dict — e.g. benchmarks=('afriqa',) backfills just the AfriQA XOR probe."""
    import torch

    from byte_embed.eval_mteb import eval_battery
    from byte_embed.miracl import eval_miracl_langs
    from byte_embed.qa_retrieval import eval_qa_retrieval

    out = out or results_path
    results = json.loads(Path(results_path).read_text(encoding="utf-8"))
    langs = langs or results.get("langs")
    teacher_dim = results.get("teacher_dim", 1024)
    teacher = results.get("teacher", "sonar")
    tsuf = "" if teacher == "sonar" else f"_{teacher}"       # per-teacher checkpoint namespace
    print(f"=== re-eval {results_path}  ({len(results.get('models', {}))} models, pooling={pooling}, "
          f"teacher={teacher}, {'QA only' if qa_only else 'full battery + MIRACL + QA'}) ===")

    for name, bm in list(results.get("models", {}).items()):
        have = bm.get("qa_retrieval") or {}
        if qa_only and (all(b in have for b in benchmarks) if benchmarks else bool(have)):
            print(f"  [reeval] {name}: already has QA -> skip")
            continue
        # boundary-arm models: per-arm checkpoint namespace + the arm's input transform at eval
        b = bm.get("boundary")
        mtsuf = tsuf + (f"_b-{b}" if b else "")
        try:
            enc = _load_enc(name, bm, pooling, ckpt_dir, teacher_dim, device, tsuf=mtsuf)
            if enc is not None and b:
                from byte_embed.boundaries import make_transform
                _t, _e = make_transform(b), enc
                enc = lambda xs, _t=_t, _e=_e: _e([_t(x) for x in xs])  # noqa: E731
        except Exception as e:  # noqa: BLE001
            print(f"  [reeval] {name}: load failed {type(e).__name__}: {e}")
            continue
        if enc is None:
            continue
        try:
            if not qa_only:
                bm.update(eval_battery(enc, langs))
                bm["miracl"] = eval_miracl_langs(enc, list(miracl_langs), n_queries=n_queries,
                                                 distractors=distractors, cache_dir=ckpt_dir)
            kw = {} if benchmarks is None else {"benchmarks": tuple(benchmarks)}
            qa_new = eval_qa_retrieval(enc, n_queries=n_queries, distractors=distractors,
                                       cache_dir=ckpt_dir, **kw)
            merged = dict(bm.get("qa_retrieval") or {})       # additive: keep other benchmarks' numbers
            merged.update(qa_new)
            bm["qa_retrieval"] = merged
        except Exception as e:  # noqa: BLE001
            print(f"  [reeval] {name}: eval failed {type(e).__name__}: {e}")
            continue
        Path(out).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        qa = bm["qa_retrieval"]
        am = (qa.get("amharicpr") or {}).get("ndcg@10_mean")
        cl = (qa.get("ciral") or {}).get("per_lang") or {}
        print(f"  [reeval] {name}: saved | AmharicPR={am} "
              f"CIRAL-ha={(cl.get('ha') or {}).get('ndcg@10')}")
        if "cuda" in str(device):
            torch.cuda.empty_cache()

    print(f"\nre-eval done -> {out}")
    return results
