"""Cloud (A100) orchestrator: the FULL SOTA push for ByteEmbed.

Three things the 12 GB laptop can't do, run on a Colab/cloud A100 via notebooks/sota_push_a100.ipynb:
  (1) SCALING CURVE — byt5 small -> base -> LARGE (1.2B; won't fit 12 GB), one teacher (mE5-base),
      matched budget, so the size effect is clean;
  (2) FLAGSHIP SOTA — byt5-large distilled HARD from the stronger mE5-LARGE teacher (higher ceiling),
      big data + many steps + more languages, to actually close the gap to mE5/LaBSE;
  (3) MIRACL — real passage retrieval (nDCG@10 / recall@100), not just Tatoeba bitext-mining.

Reuses the smoke-tested distill / ByteStudent / _benchmark; adds MIRACL + multilingual breadth.
INCREMENTAL + RESUMABLE: the results JSON is rewritten after every model and every baseline, a model
already present is skipped, and the long flagship checkpoints its model+optimizer (distill ckpt_path)
so a Colab disconnect just means re-running the cell. Training sentences are cached to disk too.

  python -m byte_embed.run_cloud --smoke            # ~5-min A100 self-test of the WHOLE pipeline
  python -m byte_embed.run_cloud --phase scaling    # the scaling curve only
  python -m byte_embed.run_cloud --phase flagship   # the flagship SOTA model only
  python -m byte_embed.run_cloud                     # everything (default), resumable
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from byte_embed import benchmark as _bm
from byte_embed.miracl import DEFAULT_EVAL as MIRACL_EVAL
from byte_embed.miracl import eval_miracl_langs
from byte_embed.run_scale import _benchmark, _f

TEACHER_BASE = "intfloat/multilingual-e5-base"
TEACHER_LARGE = "intfloat/multilingual-e5-large"

# broad, script-diverse training+eval set (overlaps SIB / Tatoeba / STS / MIRACL coverage)
LANGS = ["en", "de", "es", "fr", "it", "pl", "tr", "vi", "id", "sw",
         "ru", "uk", "bg", "ar", "fa", "hi", "bn", "ur", "te", "fi", "ko", "ja", "zh", "th"]

# extend the Tatoeba 3-letter map so bitext-mining covers the broad set (benchmark.py ships only 7)
_bm.TATOEBA_CODE.update({"de": "deu", "es": "spa", "fr": "fra", "it": "ita", "pl": "pol",
                         "uk": "ukr", "bg": "bul", "fa": "pes", "ur": "urd", "te": "tel",
                         "fi": "fin", "ko": "kor", "ja": "jpn", "zh": "cmn", "id": "ind",
                         "th": "tha"})

# the recipe that worked locally: balanced objective + big negative queue + relational (STS) term.
# AdamW on the A100 (it has the VRAM the laptop didn't) — it trained better than Adafactor.
OBJ = dict(objective="both", queue_size=8192, rel_weight=1.0, optimizer="adamw")


def _teacher(name, device, cache):
    if name not in cache:
        from byte_embed.model import TeacherEncoder
        cache[name] = TeacherEncoder(name, device=device, prefix=("query: " if "e5" in name else ""))
    return cache[name]


def _save(results, out):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(results, indent=2, ensure_ascii=False))


def _eval(enc, langs, miracl_langs, miracl_q, miracl_distractors, cache_dir):
    bm = _benchmark(enc, langs)
    if miracl_langs:
        bm["miracl"] = eval_miracl_langs(enc, miracl_langs, n_queries=miracl_q,
                                         distractors=(miracl_distractors or 20000), cache_dir=cache_dir)
    return bm


def run(phase="all", out="results/byte_cloud.json", smoke=False, device="cuda",
        scaling_steps=15000, flagship_steps=40000, n_scaling=8000, n_flagship=8000,
        miracl_q=250, miracl_extra=20000, ckpt_dir="checkpoints", teacher_id=TEACHER_BASE):
    import torch
    from sentence_transformers import SentenceTransformer

    from byte_embed.data import load_sentences
    from byte_embed.distill import distill
    from byte_embed.model import ByteStudent

    langs = ["en", "sw", "tr", "ar"] if smoke else LANGS
    miracl_langs = ["sw"] if smoke else MIRACL_EVAL  # sw corpus is small (~130k) -> fast smoke
    if smoke:
        scaling_steps = flagship_steps = 80
        n_scaling = n_flagship = 1500
        miracl_q, miracl_extra = 30, 1500

    # plan rows: name, backbone, teacher, steps, batch, n_per_lang, checkpoint?
    # FAIR size test: EQUAL batch 64 at every size (the 80 GB A100 fits byt5-large at 64 — it used
    # only 20 GB at batch 16), so token-views scale ONLY with steps; and bigger models get MORE
    # steps (compute-optimal lean), the opposite of the old 64/32/16 batch shrink that starved the
    # big models. byte-large is checkpointed (disconnect-safe: re-run cell 5 resumes it). One teacher
    # (mE5-base by default — pass teacher_id to override) keeps the size axis clean.
    #   views: byte-small 640k | byte-base 832k | byte-large 960k  (4x byte-large's old 240k)
    scaling = [("byte-small", "google/byt5-small", teacher_id, 10000, 64, n_scaling, False),
               ("byte-base", "google/byt5-base", teacher_id, 13000, 64, n_scaling, False),
               ("byte-large", "google/byt5-large", teacher_id, 15000, 64, n_scaling, True)]
    flagship = [("byte-large-flagship", "google/byt5-large", TEACHER_LARGE, flagship_steps, 16,
                 n_flagship, True)]
    if smoke:
        scaling = [("byte-small", "google/byt5-small", TEACHER_BASE, 80, 16, 1500, False)]
        flagship = [("byte-base", "google/byt5-base", TEACHER_LARGE, 80, 8, 1500, True)]
    plan = {"scaling": scaling, "flagship": flagship, "all": scaling + flagship}[phase]

    results = json.loads(Path(out).read_text(encoding="utf-8")) if Path(out).exists() \
        else {"langs": langs}
    results.setdefault("models", {})
    teachers = {}

    # cache training sentences at the largest size any planned model needs, slice down for the rest
    need_n = max((row[5] for row in plan), default=n_scaling)
    sc = Path(ckpt_dir) / f"sentences_{len(langs)}langs_{need_n}.json"
    if sc.exists():
        sents_all = json.loads(sc.read_text(encoding="utf-8"))
        print(f"  [cache] reusing {len(sents_all)} sentences from {sc}")
    else:
        print(f"loading ~{need_n}/lang for {len(langs)} langs (this can take a while) ...")
        sents_all = load_sentences(langs, n_per_lang=need_n)
        sc.parent.mkdir(parents=True, exist_ok=True)
        sc.write_text(json.dumps(sents_all, ensure_ascii=False))
    print(f"  {len(sents_all)} training sentences")

    for name, backbone, tname, steps, batch, n_per, ckpt in plan:
        if name in results["models"]:
            print(f"=== {name}: already in results -> skip ===")
            continue
        print(f"\n=== {name}  ({backbone}, teacher={tname}, {steps}x{batch}) ===")
        try:
            teacher = _teacher(tname, device, teachers)
            sents = sents_all if n_per >= need_n else sents_all[:n_per * len(langs)]
            student = ByteStudent(backbone, out_dim=teacher.dim).to(device)
            params = sum(p.numel() for p in student.parameters())
            torch.cuda.reset_peak_memory_stats()
            cpath = None
            if ckpt:
                Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
                cpath = str(Path(ckpt_dir) / f"{name}.pt")
            distill(student, teacher, sents, device=device, steps=steps, batch=batch,
                    log_every=max(200, steps // 10), ckpt_path=cpath, **OBJ)

            def s_enc(xs, _s=student):
                return _s.encode(xs, device=device)

            bm = _eval(s_enc, langs, miracl_langs, miracl_q, miracl_extra, ckpt_dir)
            bm.update(params=params, kind="byte", backbone=backbone, teacher=tname, steps=steps,
                      peak_vram_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
            results["models"][name] = bm
            _save(results, out)
            print(f"  saved {name}: SIB={bm.get('sib_mean')} Tatoeba={bm.get('tatoeba_mean')} "
                  f"STS={bm.get('sts_mean')} MIRACL={(bm.get('miracl') or {}).get('ndcg@10_mean')} "
                  f"peak={bm['peak_vram_gb']}GB")
            del student
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001 — one model erroring shouldn't lose the others
            print(f"  [{name}] FAILED: {type(e).__name__}: {e}")
            torch.cuda.empty_cache()

    baselines = [("mE5-base", TEACHER_BASE, "query: "), ("mE5-large", TEACHER_LARGE, "query: "),
                 ("LaBSE", "sentence-transformers/LaBSE", "")]
    if smoke:
        baselines = [("mE5-base", TEACHER_BASE, "query: ")]
    for name, mid, prefix in baselines:
        if name in results["models"]:
            continue
        print(f"\n=== baseline {name} ({mid}) ===")
        try:
            m = SentenceTransformer(mid, device=device)

            def b_enc(xs, _m=m, _p=prefix):
                return _m.encode([_p + x for x in xs], normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)

            bm = _eval(b_enc, langs, miracl_langs, miracl_q, miracl_extra, ckpt_dir)
            bm.update(params=sum(p.numel() for p in m.parameters()), kind="baseline", backbone=mid)
            results["models"][name] = bm
            _save(results, out)
            del m
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            print(f"  [{name}] FAILED: {type(e).__name__}: {e}")

    _summary(results)
    print(f"\nSaved -> {out}")
    return results


def _summary(results):
    M = results["models"]
    print("\n" + "=" * 94)
    print("CLOUD SOTA PUSH — scaling curve + flagship + MIRACL (real retrieval)")
    print("=" * 94)
    print(f"{'model':22}{'params(M)':>10}{'SIB':>7}{'Tatoeba':>9}{'STS':>7}{'MIRACL':>8}{'R@100':>8}")
    for name, r in M.items():
        mir = r.get("miracl") or {}
        print(f"{name:22}{(r.get('params', 0) / 1e6):>10.0f}{_f(r.get('sib_mean'), 7)}"
              f"{_f(r.get('tatoeba_mean'), 9)}{_f(r.get('sts_mean'), 7)}"
              f"{_f(mir.get('ndcg@10_mean'), 8)}{_f(mir.get('recall@100_mean'), 8)}")
    curve = [n for n in ("byte-small", "byte-base", "byte-large") if n in M]
    if len(curve) >= 2:
        print("\nSCALING (byte small -> base -> large, mE5-base teacher):")
        for n in curve:
            r = M[n]
            mir = r.get("miracl") or {}
            print(f"  {n:12} Tatoeba {r.get('tatoeba_mean')}  MIRACL nDCG@10 {mir.get('ndcg@10_mean')}")
    fl, te = M.get("byte-large-flagship"), M.get("mE5-large")
    if fl and te:
        fm, tm = (fl.get("miracl") or {}), (te.get("miracl") or {})
        if fm.get("ndcg@10_mean") and tm.get("ndcg@10_mean"):
            print(f"\nSOTA gap (flagship vs mE5-large TEACHER, its ceiling): MIRACL nDCG@10 "
                  f"{fm['ndcg@10_mean']} / {tm['ndcg@10_mean']} = "
                  f"{100 * fm['ndcg@10_mean'] / tm['ndcg@10_mean']:.0f}% of teacher")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["scaling", "flagship", "all"], default="all")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/byte_cloud.json")
    ap.add_argument("--scaling-steps", type=int, dest="scaling_steps", default=15000)
    ap.add_argument("--flagship-steps", type=int, dest="flagship_steps", default=40000)
    ap.add_argument("--miracl-q", type=int, dest="miracl_q", default=250)
    ap.add_argument("--miracl-extra", type=int, dest="miracl_extra", default=0,
                    help="extra sampled-corpus distractors per MIRACL lang (harder, more realistic)")
    a = ap.parse_args()
    run(phase=a.phase, out=a.out, smoke=a.smoke, device=a.device, scaling_steps=a.scaling_steps,
        flagship_steps=a.flagship_steps, miracl_q=a.miracl_q, miracl_extra=a.miracl_extra)


if __name__ == "__main__":
    main()
