"""ByteEmbed feasibility configuration.

Defaults are sized for a Colab A100 (80 GB) but `MAX_BYTES`/`BATCH`/`STEPS` can be cut to
fit a 24 GB 4090 (and even a 12 GB laptop for a tiny smoke test).
"""

TEACHER = "intfloat/multilingual-e5-base"  # frozen multilingual subword teacher (768-d)
STUDENT_BACKBONE = "google/byt5-small"   # byte-level encoder = the tokenizer-free student

# high- + low-resource mix so we can see whether byte-level helps the tail specifically
TRAIN_LANGS = ["en", "tr", "sw", "bn"]
EVAL_LANGS = ["en", "tr", "sw", "bn"]

# flores-200 codes for the eval languages (parallel cross-lingual retrieval)
FLORES_CODE = {"en": "eng_Latn", "tr": "tur_Latn", "sw": "swh_Latn", "bn": "ben_Beng"}

N_PER_LANG = 20000   # training sentences per language
MAX_BYTES = 256      # truncation length in BYTES (byte-level => longer than subword)
BATCH = 64
LR = 2e-4
STEPS = 2000         # ~ a few hundred k sentence views; a feasibility budget

# laptop smoke-test overrides (12 GB): tiny everything, proof-of-life only
SMOKE = dict(N_PER_LANG=2000, MAX_BYTES=160, BATCH=16, STEPS=300, TRAIN_LANGS=["en", "sw"])
