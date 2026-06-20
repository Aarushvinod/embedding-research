"""Standard multilingual benchmarks for ByteEmbed (byte student vs subword teacher).

Scores the three MTEB multilingual task families that are tractable at feasibility scale AND
cover our languages:
  - Classification : SIB-200 (accuracy, logistic-regression probe)
  - BitextMining   : Tatoeba (find-the-translation accuracy)
  - STS            : STS22 (Spearman of cosine vs human scores)

Each is scored CLEAN and under ROMANIZATION. The romanized columns are the "does the robustness
gain matter on real tasks" evidence (the downstream test reviewers ask for): for SIB the whole
input is romanized; for Tatoeba only the non-English source is romanized (noisy query, clean
corpus). Full document retrieval (MIRACL / BEIR) needs corpus indexing and is left to the scaled
study.

Reuses the saved base student checkpoint (no retraining):
  python -m byte_embed.benchmark --student-ckpt results/byte_student_local.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from common.eval import l2norm, sib_probe, sts_probe

# Tatoeba (MTEB) 3-letter source codes (paired with eng)
TATOEBA_CODE = {"tr": "tur", "sw": "swh", "bn": "ben", "ar": "ara",
                "ru": "rus", "hi": "hin", "vi": "vie"}


def _tatoeba(encode_fn, lang, n=500, rom_src=False):
    """Find-the-translation accuracy on Tatoeba (source -> nearest English)."""
    from datasets import load_dataset

    from byte_embed.robustness import romanize

    code = TATOEBA_CODE.get(lang)
    if code is None:
        return None
    ds = None
    for cfg in (f"{code}-eng", f"eng-{code}"):
        for kw in ({}, {"trust_remote_code": True}):
            try:
                ds = load_dataset("mteb/tatoeba-bitext-mining", cfg, split="test", **kw)
                src_first = cfg.startswith(code)
                break
            except TypeError:
                continue
            except Exception:  # noqa: BLE001
                ds = None
        if ds is not None:
            break
    if ds is None:
        return None
    cols = ds.column_names
    f1 = "sentence1" if "sentence1" in cols else cols[0]
    f2 = "sentence2" if "sentence2" in cols else cols[1]
    src_col, en_col = (f1, f2) if src_first else (f2, f1)
    src = list(ds[src_col])[:n]
    en = list(ds[en_col])[:n]
    if rom_src:
        src = [romanize(x) for x in src]
    a = l2norm(encode_fn(src))
    b = l2norm(encode_fn(en))
    return round(float(((a @ b.T).argmax(1) == np.arange(len(a))).mean()), 4)


def _bench_model(encode_fn, langs):
    from byte_embed.robustness import romanize

    def rom(xs):
        return encode_fn([romanize(x) for x in xs])

    out = {"sib": {}, "sib_rom": {}, "tatoeba": {}, "tatoeba_rom": {}, "sts": {}}
    for lang in langs:
        out["sib"][lang] = sib_probe(encode_fn, lang)
        out["sib_rom"][lang] = sib_probe(rom, lang)
        out["tatoeba"][lang] = _tatoeba(encode_fn, lang)
        out["tatoeba_rom"][lang] = _tatoeba(encode_fn, lang, rom_src=True)
        out["sts"][lang] = sts_probe(encode_fn, lang)
        print(f"  {lang}: sib={out['sib'][lang]}/{out['sib_rom'][lang]} "
              f"tatoeba={out['tatoeba'][lang]}/{out['tatoeba_rom'][lang]} sts={out['sts'][lang]}")
    return out


def _mean(d):
    vals = [v for v in d.values() if v is not None]
    return round(float(np.mean(vals)), 4) if vals else None


def run(student_ckpt="results/byte_student_local.pt", langs=None,
        out="results/byte_bench.json", device="cuda"):
    import torch

    from byte_embed import config as C
    from byte_embed.model import ByteStudent, TeacherEncoder

    langs = langs or list(C.EVAL_LANGS)
    teacher = TeacherEncoder(C.TEACHER, device=device)
    student = ByteStudent(C.STUDENT_BACKBONE, out_dim=teacher.dim, max_bytes=C.MAX_BYTES,
                          pooling="mean").to(device)
    student.load_state_dict(torch.load(student_ckpt, map_location=device))
    student.eval()

    def s_enc(xs):
        return student.encode(xs, device=device)

    def t_enc(xs):
        return teacher.encode(xs, as_tensor=False, device=device)

    print("scoring TEACHER (multilingual-e5-base):")
    res = {"teacher": _bench_model(t_enc, langs)}
    print("scoring STUDENT (byte / ByT5-small):")
    res["student"] = _bench_model(s_enc, langs)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
    _summary(res, langs)
    print(f"\nSaved -> {out}")
    return res


def _summary(res, langs):
    t, s = res["teacher"], res["student"]
    print("\n" + "=" * 80)
    print("STANDARD BENCHMARKS  (student = byte ByT5-small · teacher = multilingual-e5-base)")
    print("=" * 80)
    print(f"\n{'task':22}{'teacher':>10}{'student':>10}{'stu/tea':>9}")
    rows = [("SIB-200 acc (clean)", "sib"), ("SIB-200 acc (romanized)", "sib_rom"),
            ("Tatoeba acc (clean)", "tatoeba"), ("Tatoeba acc (rom src)", "tatoeba_rom"),
            ("STS22 spearman", "sts")]
    for label, key in rows:
        tm, sm = _mean(t[key]), _mean(s[key])
        if tm is None or sm is None:
            continue
        ratio = f"{sm / tm:.2f}" if tm else "-"
        print(f"{label:22}{tm:>10.3f}{sm:>10.3f}{ratio:>9}")

    print("\nDOWNSTREAM ROBUSTNESS (accuracy retained under romanization; clean -> romanized):")
    for label, ck, rk in [("SIB-200", "sib", "sib_rom"), ("Tatoeba", "tatoeba", "tatoeba_rom")]:
        tc, tr = _mean(t[ck]), _mean(t[rk])
        sc, sr = _mean(s[ck]), _mean(s[rk])
        if None in (tc, tr, sc, sr):
            continue
        td = tc - tr
        sd = sc - sr
        print(f"  {label:10} teacher {tc:.3f}->{tr:.3f} (drop {td:+.3f}) | "
              f"student {sc:.3f}->{sr:.3f} (drop {sd:+.3f}) | student drops "
              f"{'LESS' if sd < td else 'MORE'} by {abs(td - sd):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student-ckpt", dest="student_ckpt", default="results/byte_student_local.pt")
    ap.add_argument("--langs", nargs="+", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/byte_bench.json")
    a = ap.parse_args()
    run(student_ckpt=a.student_ckpt, langs=a.langs, out=a.out, device=a.device)


if __name__ == "__main__":
    main()
