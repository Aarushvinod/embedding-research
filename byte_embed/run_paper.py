"""Full-paper experiment for ByteEmbed (single authoritative run).

Trains a set of students from the frozen multilingual-e5 teacher and evaluates each on ONE
battery so every claim is comparable:
  - teacher-alignment cosine
  - cross-lingual retrieval: Tatoeba bitext mining (standard), clean + romanized-source
  - classification: SIB-200 (standard), clean + romanized
  - STS: STS22 (standard)
  - orthographic robustness: 8 perturbations, byte-student vs teacher, bootstrap 95% CIs

Configs isolate the paper's claims:
  byte-cosine       cosine-only distillation              (robust, weak retrieval)
  byte-contrastive  in-batch contrastive                  (discriminative -> retrieval)
  byte-both         cosine + contrastive                  (alignment + retrieval)
  subword-mt5-both  SAME recipe, mt5-small SUBWORD student = ISO-COMPUTE baseline (byte vs subword)
  byte-random       untrained byte student                = floor / causal control

Plus a tokenizer-fertility analysis (teacher subword fertility per language) — the proposed
mechanism linking subword fragmentation to the byte robustness advantage.

  python -m byte_embed.run_paper --steps 2000 --batch 24            # full (local, ~2h on a 5070 Ti)
  python -m byte_embed.run_paper --smoke                            # quick validation
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from byte_embed import robustness as R
from byte_embed.benchmark import _tatoeba
from common.eval import sib_probe, sts_probe


def _l2(a):
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)


def _align(s, t):
    return round(float((_l2(s) * _l2(t)).sum(1).mean()), 4)


def _ci(v, nb=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(v)
    if n == 0:
        return [float("nan"), float("nan")]
    m = v[rng.integers(0, n, size=(nb, n))].mean(1)
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def _mean(d):
    vals = [x for x in d.values() if x is not None]
    return round(float(np.mean(vals)), 4) if vals else None


def fertility(tokenizer, langs, mono):
    """Teacher subword fertility per language: subword tokens per character (higher = more
    fragmentation). The proposed driver of the byte robustness advantage."""
    out = {}
    for lang in langs:
        sents = mono.get(lang) or []
        if not sents:
            continue
        toks = sum(len(tokenizer.tokenize(s)) for s in sents)
        chars = sum(len(s) for s in sents)
        out[lang] = round(toks / max(chars, 1), 4)
    return out


def _teacher_refs(t_enc, langs, mono, perts):
    """Compute teacher quantities once (reused across configs)."""
    ref = {"robust": {}, "tatoeba": {}, "tatoeba_rom": {}, "sib": {}, "sib_rom": {}, "sts": {}}

    def rom(xs):
        return t_enc([R.romanize(x) for x in xs])

    for lang in langs:
        sents = mono.get(lang) or []
        if len(sents) < 16:
            continue
        ref["robust"][lang] = {p: R.stability_vec(t_enc, sents, R.PERTURBATIONS[p]) for p in perts}
        ref["tatoeba"][lang] = _tatoeba(t_enc, lang)
        ref["tatoeba_rom"][lang] = _tatoeba(t_enc, lang, rom_src=True)
        ref["sib"][lang] = sib_probe(t_enc, lang)
        ref["sib_rom"][lang] = sib_probe(rom, lang)
        ref["sts"][lang] = sts_probe(t_enc, lang)
    return ref


def _eval_student(s_enc, t_enc, langs, mono, perts, tref):
    out = {"align": {}, "tatoeba": {}, "tatoeba_rom": {}, "sib": {}, "sib_rom": {}, "sts": {},
           "robust": {}}

    def rom(xs):
        return s_enc([R.romanize(x) for x in xs])

    for lang in langs:
        sents = mono.get(lang) or []
        if len(sents) < 16:
            continue
        out["align"][lang] = _align(s_enc(sents), t_enc(sents))
        out["robust"][lang] = {}
        for p in perts:
            sv = R.stability_vec(s_enc, sents, R.PERTURBATIONS[p])
            tv = tref["robust"][lang][p]
            out["robust"][lang][p] = {"student": round(float(sv.mean()), 4),
                                      "teacher": round(float(tv.mean()), 4),
                                      "gap": round(float((sv - tv).mean()), 4),
                                      "_gv": (sv - tv).astype(np.float64)}
        out["tatoeba"][lang] = _tatoeba(s_enc, lang)
        out["tatoeba_rom"][lang] = _tatoeba(s_enc, lang, rom_src=True)
        out["sib"][lang] = sib_probe(s_enc, lang)
        out["sib_rom"][lang] = sib_probe(rom, lang)
        out["sts"][lang] = sts_probe(s_enc, lang)
    return out


def _agg_robust(robust, perts):
    agg = {}
    for p in perts:
        ls = [lng for lng in robust if p in robust[lng]]
        if not ls:
            continue
        pooled = np.concatenate([robust[lng][p]["_gv"] for lng in ls])
        agg[p] = {"student": round(float(np.mean([robust[lng][p]["student"] for lng in ls])), 4),
                  "teacher": round(float(np.mean([robust[lng][p]["teacher"] for lng in ls])), 4),
                  "gap": round(float(pooled.mean()), 4), "gap_ci95": _ci(pooled),
                  "wins": f"{sum(robust[lng][p]['gap'] > 0 for lng in ls)}/{len(ls)}"}
    return agg


def _strip(robust):
    return {lng: {p: {k: v for k, v in d.items() if k != "_gv"} for p, d in pd.items()}
            for lng, pd in robust.items()}


def run(device="cuda", out="results/byte_paper.json", smoke=False, steps=2000, batch=24,
        n_per_lang=10000):
    import torch

    from byte_embed import config as C
    from byte_embed.data import load_mono, load_sentences
    from byte_embed.distill import distill
    from byte_embed.model import ByteStudent, TeacherEncoder

    langs = ["en", "sw", "tr"] if smoke else list(C.EVAL_LANGS)
    perts = list(R.CORE_PERTURBATIONS) if smoke else list(R.PERTURBATIONS)
    n_mono = 120 if smoke else 200
    if smoke:
        steps, batch, n_per_lang = 150, 12, 1200

    configs = [
        {"name": "byte-cosine", "backbone": "google/byt5-small", "objective": "cosine"},
        {"name": "byte-contrastive", "backbone": "google/byt5-small", "objective": "contrastive"},
        {"name": "byte-both", "backbone": "google/byt5-small", "objective": "both"},
        {"name": "subword-mt5-both", "backbone": "google/mt5-small", "objective": "both"},
        {"name": "byte-random", "backbone": "google/byt5-small", "objective": "both", "train": False},
    ]

    teacher = TeacherEncoder(C.TEACHER, device=device)

    def t_enc(xs):
        return teacher.encode(xs, as_tensor=False, device=device)

    print(f"loading ~{n_per_lang}/lang sentences for {langs} ...")
    sents = load_sentences(langs, n_per_lang=n_per_lang)
    mono = {lng: load_mono(lng, n=n_mono) for lng in langs}
    print(f"  {len(sents)} train sentences; mono " + ",".join(f"{k}:{len(v)}" for k, v in mono.items()))

    try:
        tok = getattr(teacher.model, "tokenizer", None) or teacher.model._first_module().tokenizer
        fert = fertility(tok, langs, mono)
    except Exception as e:  # noqa: BLE001
        print(f"  [fertility] skipped: {type(e).__name__}: {e}")
        fert = {}
    print(f"teacher fertility (tok/char): {fert}")
    print("computing teacher reference metrics ...")
    tref = _teacher_refs(t_enc, langs, mono, perts)

    results = {"langs": langs, "perts": perts, "steps": steps, "batch": batch,
               "fertility": fert,
               "teacher": {"tatoeba": tref["tatoeba"], "tatoeba_rom": tref["tatoeba_rom"],
                           "sib": tref["sib"], "sib_rom": tref["sib_rom"], "sts": tref["sts"]},
               "configs": {}}

    for cfg in configs:
        name = cfg["name"]
        print(f"\n=== {name}  ({cfg['backbone']}, {cfg['objective']}, "
              f"train={cfg.get('train', True)}) ===")
        try:
            student = ByteStudent(cfg["backbone"], out_dim=teacher.dim,
                                  max_bytes=160 if smoke else C.MAX_BYTES).to(device)
            hist = []
            if cfg.get("train", True):
                hist = distill(student, teacher, sents, device=device, steps=steps, batch=batch,
                               lr=C.LR, objective=cfg["objective"], log_every=max(100, steps // 4))

            def s_enc(xs, _st=student):
                return _st.encode(xs, device=device)

            ev = _eval_student(s_enc, t_enc, langs, mono, perts, tref)
            results["configs"][name] = {
                "backbone": cfg["backbone"], "objective": cfg["objective"],
                "trained": cfg.get("train", True),
                "align": ev["align"], "align_mean": _mean(ev["align"]),
                "tatoeba": ev["tatoeba"], "tatoeba_mean": _mean(ev["tatoeba"]),
                "tatoeba_rom": ev["tatoeba_rom"], "tatoeba_rom_mean": _mean(ev["tatoeba_rom"]),
                "sib": ev["sib"], "sib_mean": _mean(ev["sib"]),
                "sib_rom": ev["sib_rom"], "sib_rom_mean": _mean(ev["sib_rom"]),
                "sts": ev["sts"], "sts_mean": _mean(ev["sts"]),
                "robust_agg": _agg_robust(ev["robust"], perts),
                "robust": _strip(ev["robust"]),
                "train_loss": hist,
            }
            del student
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001 — one config OOM/error shouldn't kill the run
            print(f"  [{name}] FAILED: {type(e).__name__}: {e}")
            torch.cuda.empty_cache()

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    _summary(results, perts)
    print(f"\nSaved -> {out}")
    return results


def _summary(results, perts):
    t = results["teacher"]
    print("\n" + "=" * 90)
    print("BYTEEMBED — FULL PAPER RESULTS  (teacher = multilingual-e5-base)")
    print("=" * 90)
    print(f"teacher: tatoeba={_mean(t['tatoeba'])} sib={_mean(t['sib'])} "
          f"sib_rom={_mean(t['sib_rom'])} sts={_mean(t['sts'])}")
    print(f"teacher fertility (tok/char, higher=more fragmentation): {results['fertility']}")
    print(f"\n{'config':18}{'align':>7}{'tatoeba':>8}{'tat_rom':>8}{'sib':>7}{'sib_rom':>8}"
          f"{'sts':>6}{'mean_robgap':>12}")
    for name, r in results["configs"].items():
        gaps = [r["robust_agg"][p]["gap"] for p in perts if p in r["robust_agg"]]
        mg = float(np.mean(gaps)) if gaps else float("nan")
        print(f"{name:18}{_fmt(r['align_mean'])}{_fmt(r['tatoeba_mean'])}{_fmt(r['tatoeba_rom_mean'])}"
              f"{_fmt(r['sib_mean'])}{_fmt(r['sib_rom_mean'])}{_fmt(r['sts_mean'],6)}{mg:>+12.3f}")
    print("\nrobustness gap by perturbation (byte-both student vs teacher, 95% CI · wins):")
    bb = results["configs"].get("byte-both") or next(iter(results["configs"].values()))
    for p in perts:
        a = bb["robust_agg"].get(p)
        if a:
            print(f"  {p:12}{a['student']:>8.3f}{a['teacher']:>8.3f}{a['gap']:>+8.3f}"
                  f"  [{a['gap_ci95'][0]:+.3f},{a['gap_ci95'][1]:+.3f}]  {a['wins']}")


def _fmt(x, w=8):
    return f"{x:>{w}.3f}" if isinstance(x, (int, float)) else f"{'-':>{w}}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--n-per-lang", type=int, dest="n_per_lang", default=10000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/byte_paper.json")
    a = ap.parse_args()
    run(device=a.device, out=a.out, smoke=a.smoke, steps=a.steps, batch=a.batch,
        n_per_lang=a.n_per_lang)


if __name__ == "__main__":
    main()
