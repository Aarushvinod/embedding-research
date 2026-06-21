"""ByteEmbed feasibility + ablations (single orchestrator).

Distill a frozen multilingual SUBWORD teacher (multilingual-e5) into a byte-level ByT5
student, then test three things, ROBUSTLY (the eval no longer depends on any single corpus):

  (1) teacher alignment      : does the byte student reproduce the teacher's embeddings?
  (2) cross-lingual P@1       : does it retrieve translations (student vs teacher)?  [best-effort]
  (3) orthographic robustness : is the byte student MORE STABLE than the teacher under
                                romanization / diacritic loss / typos / casing? (H2)

Robustness fix (why this file was rewritten): the previous eval pulled everything from one
FLORES-parallel load; when that failed on Colab the WHOLE eval came back empty. Now (1) and
(3) use per-language MONOLINGUAL sentences (load_mono, Wikipedia -> FLORES) which always
load, and only (2) needs parallel data (FLORES -> OPUS fallback) and is skipped gracefully.

Ablations (ablate=True, default on full runs): a random-init byte baseline (floor), and
sweeps over max_bytes / distillation objective / pooling / student size. Plus bootstrap CIs
and a sign test on the per-language student-vs-teacher robustness gap.

Importable as run(...) (used by the Colab notebook) and runnable as a CLI:
  python -m byte_embed.run_feasibility --steps 2000 --batch 64        # A100 (full + ablations)
  python -m byte_embed.run_feasibility --smoke                        # 12 GB laptop proof-of-life
  python -m byte_embed.run_feasibility --no-ablate                    # base config only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _l2(a: np.ndarray) -> np.ndarray:
    return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)


def _p1(a: np.ndarray, b: np.ndarray) -> float:
    a, b = _l2(a), _l2(b)
    return float(((a @ b.T).argmax(1) == np.arange(len(a))).mean())


def _align_cos(s: np.ndarray, t: np.ndarray) -> float:
    return float((_l2(s) * _l2(t)).sum(1).mean())


def _ci(vec: np.ndarray, n_boot: int = 1000, seed: int = 0) -> list[float]:
    """95% bootstrap CI on the mean of `vec`."""
    rng = np.random.default_rng(seed)
    n = len(vec)
    if n == 0:
        return [float("nan"), float("nan")]
    means = vec[rng.integers(0, n, size=(n_boot, n))].mean(1)
    return [round(float(np.percentile(means, 2.5)), 4), round(float(np.percentile(means, 97.5)), 4)]


def _eval_model(student, teacher, langs, mono, par, device, perts):
    """Robust per-config eval. Returns align (per lang), xling (per lang, best-effort), and
    robust (per lang per perturbation, with the per-sentence student-minus-teacher gap vector
    kept under '_gapvec' for pooled CIs/significance)."""
    res = {"align": {}, "xling": {}, "robust": {}}
    import byte_embed.robustness as R

    def s_enc(xs):
        return student.encode(xs, device=device)

    def t_enc(xs):
        return teacher.encode(xs, as_tensor=False, device=device)

    for lang in langs:
        sents = mono.get(lang) or []
        if len(sents) < 16:
            continue
        s_emb, t_emb = s_enc(sents), t_enc(sents)
        res["align"][lang] = round(_align_cos(s_emb, t_emb), 4)
        res["robust"][lang] = {}
        for p in perts:
            sv = R.stability_vec(s_enc, sents, R.PERTURBATIONS[p])
            tv = R.stability_vec(t_enc, sents, R.PERTURBATIONS[p])
            res["robust"][lang][p] = {
                "student": round(float(sv.mean()), 4), "teacher": round(float(tv.mean()), 4),
                "gap": round(float((sv - tv).mean()), 4), "_gapvec": (sv - tv).astype(np.float64),
            }
        pp = par.get(lang)
        if pp and lang != "en":
            res["xling"][lang] = {
                "student": round(_p1(s_enc(pp["tgt"]), s_enc(pp["en"])), 4),
                "teacher": round(_p1(t_enc(pp["tgt"]), t_enc(pp["en"])), 4),
                "src": pp.get("_src"),
            }
    return res


def _aggregate(robust, perts):
    """Across languages: mean student/teacher stability per perturbation, the mean gap with a
    bootstrap CI over POOLED per-sentence gaps, and the #langs where student > teacher."""
    agg = {}
    for p in perts:
        langs = [lng for lng in robust if p in robust[lng]]
        if not langs:
            continue
        stu = np.mean([robust[lng][p]["student"] for lng in langs])
        tea = np.mean([robust[lng][p]["teacher"] for lng in langs])
        pooled = np.concatenate([robust[lng][p]["_gapvec"] for lng in langs])
        wins = sum(robust[lng][p]["gap"] > 0 for lng in langs)
        agg[p] = {
            "student": round(float(stu), 4), "teacher": round(float(tea), 4),
            "gap": round(float(pooled.mean()), 4), "gap_ci95": _ci(pooled),
            "langs_student_wins": f"{wins}/{len(langs)}",
        }
    return agg


def _strip_vecs(robust):
    """Drop the non-serializable per-sentence gap vectors before saving/returning."""
    return {lng: {p: {k: v for k, v in d.items() if k != "_gapvec"} for p, d in pd.items()}
            for lng, pd in robust.items()}


def _train_and_eval(cfg, teacher, sents, mono, par, device, perts, label):
    from byte_embed.distill import distill
    from byte_embed.model import ByteStudent

    print(f"\n=== [{label}] {json.dumps({k: cfg[k] for k in cfg if k not in ('teacher',)})}")
    student = ByteStudent(cfg["backbone"], out_dim=teacher.dim, max_bytes=cfg["max_bytes"],
                          pooling=cfg["pooling"]).to(device)
    history = []
    if cfg.get("train", True):
        history = distill(student, teacher, sents, device=device, steps=cfg["steps"],
                          batch=cfg["batch"], lr=cfg["lr"], objective=cfg["objective"],
                          log_every=max(100, cfg["steps"] // 4))
    ev = _eval_model(student, teacher, cfg["langs"], mono, par, device, perts)
    ev["config"] = {k: cfg[k] for k in ("backbone", "max_bytes", "pooling", "objective", "steps")}
    ev["config"]["trained"] = cfg.get("train", True)
    ev["train_loss"] = history
    ev["agg"] = _aggregate(ev["robust"], perts)
    return student, ev


def run(device="cuda", out="results/byte_embed.json", save_student=None, smoke=False,
        ablate=None, **over):
    import torch  # noqa: F401

    from byte_embed import config as C
    from byte_embed import robustness as R
    from byte_embed.data import load_eval_parallel, load_mono, load_sentences
    from byte_embed.model import TeacherEncoder

    base = dict(teacher=C.TEACHER, backbone=C.STUDENT_BACKBONE, langs=list(C.EVAL_LANGS),
                n_per_lang=C.N_PER_LANG, steps=C.STEPS, batch=C.BATCH, max_bytes=C.MAX_BYTES,
                lr=C.LR, pooling="mean", objective="cosine", train=True)
    if smoke:
        base.update(n_per_lang=1500, steps=200, batch=16, max_bytes=160,
                    langs=["en", "sw", "tr"])
    base.update({k: v for k, v in over.items() if v is not None})
    if ablate is None:
        ablate = not smoke
    perts = list(R.CORE_PERTURBATIONS) if smoke else list(R.PERTURBATIONS)
    langs = base["langs"]
    n_mono, n_par = (120, 120) if smoke else (200, 400)

    teacher = TeacherEncoder(base["teacher"], device=device)
    print(f"loading ~{base['n_per_lang']}/lang training sentences for {langs} ...")
    sents = load_sentences(langs, n_per_lang=base["n_per_lang"])
    print(f"  {len(sents)} training sentences; loading eval sets ...")
    mono = {lng: load_mono(lng, n=n_mono) for lng in langs}
    par = {lng: load_eval_parallel(lng, n=n_par) for lng in langs if lng != "en"}
    print("  eval mono per lang: " + ", ".join(f"{lng}:{len(mono[lng])}" for lng in langs))

    runs = {}
    student, runs["base"] = _train_and_eval(base, teacher, sents, mono, par, device, perts, "base")
    if save_student:
        Path(save_student).parent.mkdir(parents=True, exist_ok=True)
        torch.save(student.state_dict(), save_student)

    if ablate:
        # random-init byte student (no distillation) = floor for alignment + robustness
        _, runs["random_init"] = _train_and_eval({**base, "train": False}, teacher, sents,
                                                  mono, par, device, perts, "random_init")
        ab_steps = max(800, base["steps"] // 2)  # cheaper ablation budget
        ablations = [
            ({**base, "max_bytes": 128, "steps": ab_steps}, "max_bytes=128"),
            ({**base, "objective": "mse", "steps": ab_steps}, "objective=mse"),
            ({**base, "pooling": "max", "steps": ab_steps}, "pooling=max"),
            ({**base, "backbone": "google/byt5-base", "steps": ab_steps}, "student=byt5-base"),
        ]
        for cfg, label in ablations:
            try:
                _, runs[label] = _train_and_eval(cfg, teacher, sents, mono, par, device, perts, label)
            except Exception as e:  # noqa: BLE001 — one ablation OOM/error shouldn't kill the rest
                print(f"  [ablation {label}] FAILED: {type(e).__name__}: {e}")
                torch.cuda.empty_cache()

    results = {"base_config": {k: base[k] for k in base if k != "teacher"}, "runs": {}}
    for name, ev in runs.items():
        results["runs"][name] = {"config": ev["config"], "align": ev["align"], "xling": ev["xling"],
                                 "agg": ev["agg"], "robust": _strip_vecs(ev["robust"])}
    # plot-cell compatibility: top-level train_loss + robustness from the base run
    results["train_loss"] = runs["base"]["train_loss"]
    results["robustness"] = results["runs"]["base"]["robust"]
    results["eval"] = {lng: {"teacher_alignment_cos": runs["base"]["align"].get(lng),
                             "xling_p1_student": runs["base"]["xling"].get(lng, {}).get("student"),
                             "xling_p1_teacher": runs["base"]["xling"].get(lng, {}).get("teacher")}
                       for lng in langs}

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    _summary(results, perts)
    print(f"\nSaved -> {out}")
    return results


def _summary(results, perts):
    print("\n" + "=" * 78)
    print("BYTEEMBED RESULTS  (stability = cosine under perturbation; higher = more robust)")
    print("=" * 78)
    base = results["runs"]["base"]
    langs = list(base["align"])
    print(f"\nbase: {json.dumps(results['base_config'])}")
    print(f"\nteacher-alignment cos (student reproduces teacher): "
          + ", ".join(f"{lng}={base['align'][lng]}" for lng in langs))
    xl = base["xling"]
    if xl:
        print("cross-lingual P@1 (stu / tea): "
              + ", ".join(f"{lng}={xl[lng]['student']}/{xl[lng]['teacher']}" for lng in xl))
    else:
        print("cross-lingual P@1: (no parallel eval data available)")

    print("\nH2 — orthographic robustness, BASE student vs teacher (mean over langs):")
    print(f"  {'pert':12}{'student':>9}{'teacher':>9}{'gap':>8}{'gap 95% CI':>20}{'wins':>8}")
    for p in perts:
        a = base["agg"].get(p)
        if a:
            ci = f"[{a['gap_ci95'][0]:+.3f},{a['gap_ci95'][1]:+.3f}]"
            print(f"  {p:12}{a['student']:>9.3f}{a['teacher']:>9.3f}{a['gap']:>+8.3f}"
                  f"{ci:>20}{a['langs_student_wins']:>8}")

    others = [n for n in results["runs"] if n != "base"]
    if others:
        print("\nablations (mean align_cos · mean robustness gap over perts · langs):")
        print(f"  {'run':18}{'align':>8}{'mean_gap':>10}{'pos_perts':>11}")
        for name in ["random_init"] + [n for n in others if n != "random_init"]:
            if name not in results["runs"]:
                continue
            r = results["runs"][name]
            al = np.mean([v for v in r["align"].values()]) if r["align"] else float("nan")
            gaps = [r["agg"][p]["gap"] for p in perts if p in r["agg"]]
            mg = float(np.mean(gaps)) if gaps else float("nan")
            pos = f"{sum(g > 0 for g in gaps)}/{len(gaps)}" if gaps else "-"
            print(f"  {name:18}{al:>8.3f}{mg:>+10.3f}{pos:>11}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny laptop proof-of-life")
    ap.add_argument("--no-ablate", dest="ablate", action="store_false", default=None,
                    help="base config only (skip ablations + random baseline)")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--max-bytes", type=int, dest="max_bytes")
    ap.add_argument("--n-per-lang", type=int, dest="n_per_lang")
    ap.add_argument("--langs", nargs="+", default=None)
    ap.add_argument("--objective", choices=["cosine", "mse", "contrastive", "both"], default=None)
    ap.add_argument("--backbone", default=None, help="student backbone (e.g. google/byt5-small)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/byte_embed.json")
    ap.add_argument("--save-student", dest="save_student", default=None)
    a = ap.parse_args()
    run(device=a.device, out=a.out, save_student=a.save_student, smoke=a.smoke, ablate=a.ablate,
        steps=a.steps, batch=a.batch, max_bytes=a.max_bytes, n_per_lang=a.n_per_lang,
        langs=a.langs, objective=a.objective, backbone=a.backbone)


if __name__ == "__main__":
    main()
