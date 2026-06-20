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

# flores-200 codes (used by data._flores for cross-lingual P@1 + monolingual fallback)
FLORES_CODE = {"en": "eng_Latn", "tr": "tur_Latn", "sw": "swh_Latn", "bn": "ben_Beng",
               "ar": "arb_Arab", "ru": "rus_Cyrl", "hi": "hin_Deva", "vi": "vie_Latn",
               "ca": "cat_Latn", "fi": "fin_Latn"}

N_PER_LANG = 15000   # training sentences per language (× 8 langs)
MAX_BYTES = 256      # truncation length in BYTES (byte-level => longer than subword)
BATCH = 64
LR = 2e-4
STEPS = 2000         # ~ a few hundred k sentence views; a feasibility budget

# laptop smoke-test overrides (12 GB): tiny everything, proof-of-life only
SMOKE = dict(N_PER_LANG=2000, MAX_BYTES=160, BATCH=16, STEPS=300, TRAIN_LANGS=["en", "sw"])
