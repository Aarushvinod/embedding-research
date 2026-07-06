"""ByteEmbed feasibility configuration.

Defaults are sized for a Colab A100 (80 GB) but `MAX_BYTES`/`BATCH`/`STEPS` can be cut to
fit a 24 GB 4090 (and even a 12 GB laptop for a tiny smoke test).
"""

TEACHER = "intfloat/multilingual-e5-base"  # frozen multilingual subword teacher (768-d)
STUDENT_BACKBONE = "google/byt5-small"   # byte-level encoder = the tokenizer-free student

# script- + resource-diverse set (train == eval; the student must see a language to reproduce
# the teacher there). Latin (en/tr/sw/vi), Cyrillic (ru), Arabic (ar), Devanagari (hi),
# Bengali (bn); high-resource (en/ru/ar) down to low-resource (sw); heavy diacritics (vi/tr).
EVAL_LANGS = ["en", "tr", "sw", "bn", "ar", "ru", "hi", "vi"]
TRAIN_LANGS = EVAL_LANGS  # kept for back-compat; the orchestrator trains on EVAL_LANGS

# flores-200 codes (used by data._flores for cross-lingual P@1 + monolingual fallback,
# AND as the SONAR `source_lang` + the SIB-200/Belebele/FLORES config code for each language)
FLORES_CODE = {"en": "eng_Latn", "tr": "tur_Latn", "sw": "swh_Latn", "bn": "ben_Beng",
               "ar": "arb_Arab", "ru": "rus_Cyrl", "hi": "hin_Deva", "vi": "vie_Latn",
               "ca": "cat_Latn", "fi": "fin_Latn",
               # low-resource-study languages (run_lowresource.py)
               "te": "tel_Telu", "ta": "tam_Taml", "mr": "mar_Deva", "am": "amh_Ethi",
               "ha": "hau_Latn", "rw": "kin_Latn", "zh": "zho_Hans",
               "yo": "yor_Latn", "so": "som_Latn"}

# The FINALIZED retrieval study set (deep-research verified, 2026-07): 5 low-resource (Joshi class
# 0-2) + 3 high-resource anchors. Every language has a deep-retrieval benchmark: te/sw/yo via MIRACL
# (monolingual; te/sw also Mr.TyDi), am via 2AIRTC + Amharic-PR (monolingual), ha via CIRAL
# (cross-lingual, flagged). Dropped: rw (no deep benchmark exists), ta/mr (IndicQA pool ~250 docs —
# not deep). Wikipedia floors OK for all (yo = 91k usable paragraphs, measured; old rw floor gone).
# NOTE yo is NOT in XLM-R/CC-100 (the BGE-M3 teacher's backbone) — both students inherit the same
# weakened targets there, so byte-vs-subword stays internally fair; flag it when reporting.
LOWRES_LANGS = ["te", "sw", "yo", "am", "ha"]    # Dravidian/Bantu/Niger-Congo/Semitic/Chadic
HIGHRES_LANGS = ["en", "zh", "ar"]               # anchors (incl. two non-Latin scripts)
STUDY_LANGS = LOWRES_LANGS + HIGHRES_LANGS       # the canonical 8

N_PER_LANG = 15000   # training sentences per language (× 8 langs)
MAX_BYTES = 256      # truncation length in BYTES (byte-level => longer than subword)
BATCH = 64
LR = 2e-4
STEPS = 2000         # ~ a few hundred k sentence views; a feasibility budget

# laptop smoke-test overrides (12 GB): tiny everything, proof-of-life only
SMOKE = dict(N_PER_LANG=2000, MAX_BYTES=160, BATCH=16, STEPS=300, TRAIN_LANGS=["en", "sw"])
