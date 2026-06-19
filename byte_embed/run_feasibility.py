"""ByteEmbed feasibility: distill a frozen multilingual subword teacher into a byte-level
student, then test three things:

  (1) teacher alignment   : does the byte student reproduce the teacher's embeddings?
  (2) cross-lingual P@1    : does it retrieve translations (student vs teacher) on FLORES?
  (3) orthographic robustness : is the byte student more stable than the teacher under
                                romanization / diacritic loss / spelling noise?  (the H2 claim)

Importable as `run(...)` (used by the Colab notebook) and runnable as a CLI:

  python -m byte_embed.run_feasibility --steps 2000 --batch 64        # A100/4090
  python -m byte_embed.run_feasibility --smoke                        # 12 GB laptop proof-of-life
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _retrieval_p1(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return float(((a @ b.T).argmax(1) == np.arange(len(a))).mean())


def run(device="cuda", out="results/byte_embed.json", save_student=None, smoke=False,
        **over):
    import torch  # noqa: F401

    from byte_embed import config as C
    from byte_embed import robustness as R
    from byte_embed.data import load_flores_parallel, load_sentences
    from byte_embed.distill import distill
    from byte_embed.model import ByteStudent, TeacherEncoder

    cfg = dict(teacher=C.TEACHER, backbone=C.STUDENT_BACKBONE, train_langs=C.TRAIN_LANGS,
               eval_langs=C.EVAL_LANGS, n_per_lang=C.N_PER_LANG, steps=C.STEPS,
               batch=C.BATCH, max_bytes=C.MAX_BYTES, lr=C.LR)
    if smoke:
        cfg.update(n_per_lang=2000, max_bytes=160, batch=16, steps=300,
                   train_langs=["en", "sw"], eval_langs=["en", "sw"])
    cfg.update({k: v for k, v in over.items() if v is not None})
    print("config:\n" + json.dumps(cfg, indent=2, ensure_ascii=False))

    teacher = TeacherEncoder(cfg["teacher"], device=device)
    student = ByteStudent(cfg["backbone"], out_dim=teacher.dim,
                          max_bytes=cfg["max_bytes"]).to(device)

    print(f"loading ~{cfg['n_per_lang']}/lang sentences for {cfg['train_langs']} ...")
    sents = load_sentences(cfg["train_langs"], n_per_lang=cfg["n_per_lang"])
    print(f"  {len(sents)} training sentences")

    history = distill(student, teacher, sents, device=device, steps=cfg["steps"],
                      batch=cfg["batch"], lr=cfg["lr"])

    results = {"config": cfg, "train_loss": history, "eval": {}, "robustness": {}}
    par = load_flores_parallel(cfg["eval_langs"], n=400)
    en_s = student.encode(par["en"], device=device) if "en" in par else None
    en_t = teacher.encode(par["en"], as_tensor=False, device=device) if "en" in par else None

    for lang in cfg["eval_langs"]:
        if lang not in par:
            continue
        s_emb = student.encode(par[lang], device=device)
        t_emb = teacher.encode(par[lang], as_tensor=False, device=device)
        sa = s_emb / (np.linalg.norm(s_emb, axis=1, keepdims=True) + 1e-9)
        ta = t_emb / (np.linalg.norm(t_emb, axis=1, keepdims=True) + 1e-9)
        results["eval"][lang] = {
            "teacher_alignment_cos": round(float((sa * ta).sum(1).mean()), 3),
            "xling_p1_student": None if lang == "en" else round(_retrieval_p1(s_emb, en_s), 3),
            "xling_p1_teacher": None if lang == "en" else round(_retrieval_p1(t_emb, en_t), 3),
        }
        rob = {}
        for pname, pfn in R.PERTURBATIONS.items():
            rob[pname] = {
                "student": round(R.stability(
                    lambda xs: student.encode(xs, device=device), par[lang][:200], pfn), 3),
                "teacher": round(R.stability(
                    lambda xs: teacher.encode(xs, as_tensor=False, device=device),
                    par[lang][:200], pfn), 3),
            }
        results["robustness"][lang] = rob

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    if save_student:
        import torch
        torch.save(student.state_dict(), save_student)

    _summary(results)
    print(f"\nSaved -> {out}")
    return results


def _summary(results):
    print("\n" + "=" * 70)
    print(f"{'lang':5} {'align_cos':>9} {'xP@1(stu)':>9} {'xP@1(tea)':>9}")
    for lang, e in results["eval"].items():
        print(f"{lang:5} {e['teacher_alignment_cos']:>9} "
              f"{str(e['xling_p1_student']):>9} {str(e['xling_p1_teacher']):>9}")
    print("\nrobustness (cosine stability under perturbation; higher = more robust)")
    print(f"{'lang':5} {'pert':12} {'student':>8} {'teacher':>8}")
    for lang, r in results["robustness"].items():
        for pname, v in r.items():
            print(f"{lang:5} {pname:12} {v['student']:>8} {v['teacher']:>8}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny laptop proof-of-life")
    ap.add_argument("--steps", type=int)
    ap.add_argument("--batch", type=int)
    ap.add_argument("--max-bytes", type=int, dest="max_bytes")
    ap.add_argument("--n-per-lang", type=int, dest="n_per_lang")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/byte_embed.json")
    ap.add_argument("--save-student", dest="save_student", default=None)
    a = ap.parse_args()
    run(device=a.device, out=a.out, save_student=a.save_student, smoke=a.smoke,
        steps=a.steps, batch=a.batch, max_bytes=a.max_bytes, n_per_lang=a.n_per_lang)


if __name__ == "__main__":
    main()
