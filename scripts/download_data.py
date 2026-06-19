"""Pre-download GATE-GRAFT data (fastText vectors + MUSE dicts) so the first run is fast.

  python scripts/download_data.py --langs ca tr bn eu --k 2000
"""
from __future__ import annotations

import argparse

from common.data import load_fasttext_topk, load_muse_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["ca", "tr", "bn", "eu"])
    ap.add_argument("--k", type=int, default=2000)
    a = ap.parse_args()

    print("English vectors ...")
    load_fasttext_topk("en", k=a.k)
    for lang in a.langs:
        print(f"{lang}: vectors ...")
        load_fasttext_topk(lang, k=a.k)
        d = load_muse_dict(lang, "en")
        print(f"{lang}: MUSE {lang}-en dict entries = {len(d)}"
              + ("" if d else "  (MISSING — BLI eval will be skipped for this language)"))
    print("done.")


if __name__ == "__main__":
    main()
