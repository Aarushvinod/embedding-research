# Fixed study constants and default configuration for the ByteResearch codebase.
# It defines the eleven languages (five targets, three hubs, three controls) with their FLORES,
# NTREX, CCMatrix and Wikipedia codes, scripts and decoder-tag sentinel indices, the GlotLID label
# overrides, the Unicode script ranges and sentence terminators, so a language change stays here.
# It also defines the six models, the three fine-tuning conditions, the hub x target NLLB source table, the
# candidate targets and candidate hubs (English: prepared exactly like targets and hubs so promoting one reuses
# its data, but never trained on) and the pinned revisions of every external resource, and validates them on import.
# Frozen config dataclasses carry the defaults from the approved plan for data, training (including the
# LR-sweep directions), decoding and interpretability, and smoke_overrides shrinks them for the smoke subtree.
# Resolver helpers turn CLI selections such as "all", "small" or "byt5,mt5-base" into lists.

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    # One language: short code, English name, FLORES code, NTREX code (None for controls), CCMatrix ISO
    # code (None for controls), Wikipedia subdomain, ISO 15924 script, role ("target",
    # "hub", "control", "candidate" or "candidate_hub") and the sentinel index k of its decoder tag (-1 for
    # candidates and candidate hubs, which have no tag).
    code: str
    name: str
    flores: str
    ntrex: str | None
    ccmatrix: str | None
    wiki: str | None
    script: str
    role: str
    tag_k: int


@dataclass(frozen=True)
class ModelSpec:
    # One pretrained model: registry key (e.g. "byt5-small"), Hugging Face id, architecture
    # family ("byt5" or "mt5") and size ("small", "base" or "large").
    key: str
    hf_id: str
    arch: str
    size: str


@dataclass(frozen=True)
class Condition:
    # One fine-tuning condition: key ("C1".."C3"), hub (source) language code and the tuple of
    # target language codes it translates into.
    key: str
    hub: str
    targets: tuple


@dataclass(frozen=True)
class NllbSource:
    # One NLLB-mined pair file: hub and target codes, host ("allenai" or "ccmatrix"), the URL,
    # the number of TSV columns (9 for AllenAI, 3 for CCMatrix) and whether the hub text comes
    # first among the two text columns.
    hub: str
    x: str
    host: str
    url: str
    n_cols: int
    hub_first: bool


_LANG_ROWS = (
    ("sw", "Swahili", "swh_Latn", "swa", "sw", "sw", "Latn", "target"),
    ("sr", "Serbian", "srp_Cyrl", "srp-Cyrl", "sr", "sr", "Cyrl", "target"),
    ("et", "Estonian", "est_Latn", "est", "et", "et", "Latn", "target"),
    ("ml", "Malayalam", "mal_Mlym", "mal", "ml", "ml", "Mlym", "target"),
    ("bn", "Bengali", "ben_Beng", "ben", "bn", "bn", "Beng", "target"),
    ("fr", "French", "fra_Latn", "fra", "fr", "fr", "Latn", "hub"),
    ("fa", "Persian", "pes_Arab", "fas", "fa", "fa", "Arab", "hub"),
    ("ru", "Russian", "rus_Cyrl", "rus", "ru", "ru", "Cyrl", "hub"),
    ("es", "Spanish", "spa_Latn", None, None, "es", "Latn", "control"),
    ("zh", "Chinese", "zho_Hans", None, None, "zh", "Hans", "control"),
    ("el", "Greek", "ell_Grek", None, None, "el", "Grek", "control"),
)
LID_LABEL_OVERRIDES: dict = {"fa": "fas_Arab", "et": "ekk_Latn", "zh": "cmn_Hani"}

_CANDIDATE_ROWS: tuple = ()
CANDIDATE_LANGUAGES: dict = {row[0]: Language(*row, tag_k=-1) for row in _CANDIDATE_ROWS}
CANDIDATES: tuple = tuple(CANDIDATE_LANGUAGES)
_CANDIDATE_HUB_ROWS = (
    ("en", "English", "eng_Latn", "eng", "en", "en", "Latn", "candidate_hub"),
)
CANDIDATE_HUB_LANGUAGES: dict = {row[0]: Language(*row, tag_k=-1) for row in _CANDIDATE_HUB_ROWS}
CANDIDATE_HUBS: tuple = tuple(CANDIDATE_HUB_LANGUAGES)
NLLB_UNAVAILABLE: frozenset = frozenset()

TAG_K0 = 86
MAX_TAG_K = 99

LANGUAGES: dict[str, Language] = {
    row[0]: Language(*row, tag_k=TAG_K0 + i) for i, row in enumerate(_LANG_ROWS)
}
TARGETS: tuple = tuple(c for c, l in LANGUAGES.items() if l.role == "target")
HUBS: tuple = tuple(c for c, l in LANGUAGES.items() if l.role == "hub")
CONTROLS: tuple = tuple(c for c, l in LANGUAGES.items() if l.role == "control")
STUDY_LANGS: tuple = TARGETS + HUBS
ALL_LANGS: tuple = STUDY_LANGS + CONTROLS
DATA_LANGS: tuple = ALL_LANGS + CANDIDATES + CANDIDATE_HUBS
NTREX_LANGS: tuple = STUDY_LANGS + CANDIDATES + CANDIDATE_HUBS
LID_LANGS: tuple = ALL_LANGS + CANDIDATES + CANDIDATE_HUBS
WIKI_CANDIDATES: tuple = CANDIDATES + CANDIDATE_HUBS

MODELS: dict[str, ModelSpec] = {
    f"{arch}-{size}": ModelSpec(f"{arch}-{size}", f"google/{arch}-{size}", arch, size)
    for arch in ("byt5", "mt5")
    for size in ("small", "base", "large")
}
SIZES: tuple = ("small", "base", "large")
ARCHS: tuple = ("byt5", "mt5")

CONDITIONS: dict[str, Condition] = {
    "C1": Condition("C1", "fr", TARGETS),
    "C2": Condition("C2", "fa", TARGETS),
    "C3": Condition("C3", "ru", TARGETS),
}

TESTSETS: tuple = ("flores_devtest", "ntrex")
METRICS: tuple = ("chrf_pp", "bleu", "spbleu", "comet", "metricx")

SMOKE_MODELS: tuple = ("byt5-small", "mt5-small")
SMOKE_TARGETS: tuple = ("sw", "sr")
SMOKE_CONDITIONS: tuple = ("C1", "C2")

_ALLENAI = "https://storage.googleapis.com/allennlp-data-bucket/nllb/{name}.gz"
_CCMATRIX = "http://data.statmt.org/cc-matrix/{name}.bitextf.tsv.gz"
NLLB_ALLENAI_HUBS: tuple = ("en",)
NLLB_ALLENAI_PAIRS: tuple = ()
NLLB_CCMATRIX_PAIRS: tuple = (("en", "sr"), ("en", "et"))


def language(code: str) -> Language:
    # Input: a study, candidate or candidate-hub language code. Output: its Language; raises KeyError when unknown.
    for table in (LANGUAGES, CANDIDATE_LANGUAGES, CANDIDATE_HUB_LANGUAGES):
        if code in table:
            return table[code]
    raise KeyError(code)


def _nllb_row(hub: str, x: str) -> NllbSource:
    # Inputs: hub and target codes. Output: the NllbSource for that pair. AllenAI serves every pair
    # of an NLLB_ALLENAI_HUBS hub plus NLLB_ALLENAI_PAIRS, except NLLB_CCMATRIX_PAIRS (9 columns, files
    # named by the two FLORES codes in sorted order); CCMatrix serves the rest (3 columns, files named by
    # the two ISO codes in sorted order). The hub text comes first exactly when the hub's code sorts first.
    hl, xl = language(hub), language(x)
    allenai = (hub in NLLB_ALLENAI_HUBS or (hub, x) in NLLB_ALLENAI_PAIRS) and (hub, x) not in NLLB_CCMATRIX_PAIRS
    a, b = (hl.flores, xl.flores) if allenai else (hl.ccmatrix, xl.ccmatrix)
    url = (_ALLENAI if allenai else _CCMATRIX).format(name="-".join(sorted((a, b))))
    return NllbSource(hub, x, "allenai" if allenai else "ccmatrix", url, 9 if allenai else 3, a < b)


NLLB_SOURCES: dict[tuple, NllbSource] = {(h, x): _nllb_row(h, x) for h in HUBS for x in TARGETS}
CANDIDATE_SOURCES: dict[tuple, NllbSource] = {
    (h, x): _nllb_row(h, x)
    for h, x in [(h, x) for h in HUBS for x in CANDIDATES] + [(h, x) for h in CANDIDATE_HUBS for x in TARGETS + CANDIDATES]
    if (h, x) not in NLLB_UNAVAILABLE
}

PINS: dict = {
    "flores": {
        "repo": "mteb/flores",
        "revision": "69edd3336d102dd184dd531b57defac6c9c60c8f",
        "files": {"dev": "dev.parquet", "devtest": "devtest.parquet"},
        "rows": {"dev": 997, "devtest": 1012},
    },
    "ntrex": {
        "repo": "MicrosoftTranslator/NTREX",
        "commit": "468c6b69c7f6a75d31d4743d9daba2af566cc18d",
        "ref_pattern": "NTREX-128/newstest2019-ref.{code}.txt",
        "src_file": "NTREX-128/newstest2019-src.eng.txt",
        "rows": 1997,
    },
    "wikipedia": {"repo": "wikimedia/wikipedia", "date": "20231101"},
    "metricx": {
        "repo": "google-research/metricx",
        "commit": "fc4978eb064670f7cc33e93ea4f52d38396b8ae6",
        "model": "google/metricx-24-hybrid-xl-v2p6",
        "tokenizer": "google/mt5-xl",
    },
    "comet": {"model": "Unbabel/wmt22-comet-da"},
    "glotlid": {"repo": "cis-lmu/glotlid", "file": "model.bin"},
    "labse": {"model": "sentence-transformers/LaBSE"},
    "sacrebleu": {"chrf_word_order": 2, "bleu_tokenize": "flores200"},
}

SCRIPT_RANGES: dict = {
    "Latn": ((0x41, 0x5A), (0x61, 0x7A), (0xC0, 0x24F), (0x250, 0x2AF), (0x300, 0x36F),
             (0x1E00, 0x1EFF), (0x2C60, 0x2C7F), (0xA720, 0xA7FF), (0xAB30, 0xAB6F),
             (0xFF21, 0xFF3A), (0xFF41, 0xFF5A)),
    "Arab": ((0x600, 0x6FF), (0x750, 0x77F), (0x8A0, 0x8FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)),
    "Cyrl": ((0x400, 0x52F), (0x1C80, 0x1C8F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)),
    "Geor": ((0x10A0, 0x10FF), (0x1C90, 0x1CBF), (0x2D00, 0x2D2F)),
    "Armn": ((0x530, 0x58F), (0xFB13, 0xFB17)),
    "Sinh": ((0xD80, 0xDFF), (0x111E0, 0x111FF)),
    "Deva": ((0x900, 0x97F), (0x1CD0, 0x1CFF), (0xA8E0, 0xA8FF)),
    "Taml": ((0xB80, 0xBFF), (0x11FC0, 0x11FFF)),
    "Mlym": ((0xD00, 0xD7F),),
    "Grek": ((0x370, 0x3FF), (0x1F00, 0x1FFF)),
    "Beng": ((0x980, 0x9FF),),
    "Guru": ((0xA00, 0xA7F),),
    "Gujr": ((0xA80, 0xAFF),),
    "Orya": ((0xB00, 0xB7F),),
    "Telu": ((0xC00, 0xC7F),),
    "Knda": ((0xC80, 0xCFF),),
    "Thai": ((0xE00, 0xE7F),),
    "Hebr": ((0x590, 0x5FF), (0xFB1D, 0xFB4F)),
    "Ethi": ((0x1200, 0x139F), (0x2D80, 0x2DDF), (0xAB00, 0xAB2F)),
    "Hani": ((0x2E80, 0x2FDF), (0x3005, 0x3007), (0x3021, 0x3029), (0x3038, 0x303B),
             (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x2FA1F),
             (0x30000, 0x3134F)),
}
SCRIPT_ALIASES: dict = {"Hans": "Hani", "Hant": "Hani", "Han": "Hani"}
SENTENCE_TERMINATORS: str = ".!?।॥։؟۔።"
UNSPACED_TERMINATORS: str = "。"


def ntrex_file(code: str) -> str:
    # Input: a study or candidate language code with an NTREX code. Output: the repository-relative NTREX
    # file path (English uses the source file, every other language its reference file).
    lang = language(code)
    if lang.ntrex is None:
        raise KeyError(f"{code} has no NTREX file")
    if code == "en":
        return PINS["ntrex"]["src_file"]
    return PINS["ntrex"]["ref_pattern"].format(code=lang.ntrex)


@dataclass(frozen=True)
class DataConfig:
    # Data-building defaults: NLLB cap, the fixed length window (its maximum equals the 512-character training
    # and decoding cap) and script-ratio minimum, the FLORES percentile behind every calibrated LID and LaBSE
    # minimum (no floor) and the length-ratio band's two percentiles, the number of
    # GlotLID labels read per prediction when calibrating (to see a language's probability when it is not on
    # top), the LaBSE batch size and inference dtype ("fp32" or "fp16"), Wikipedia sampling sizes, and the
    # seed for every shuffle.
    nllb_cap: int = 200_000
    min_chars: int = 20
    max_chars: int = 512
    min_script: float = 0.8
    calib_percentile: float = 1.0
    ratio_percentiles: tuple = (1.0, 99.0)
    glotlid_top_k: int = 10
    labse_batch: int = 256
    labse_dtype: str = "fp32"
    wiki_date: str = "20231101"
    wiki_per_lang: int = 40_000
    wiki_per_article: int = 5
    wiki_min_chars: int = 20
    wiki_max_chars: int = 512
    seed: int = 0


@dataclass(frozen=True)
class TrainConfig:
    # Training defaults: LR sweep grid, length and the (condition, target) directions it runs on, full-run
    # epochs, warmup, batch geometry, checkpoint/eval cadence, early stopping, truncation, optimizer and the
    # size of the fixed FLORES-dev subset each direction is selected on.
    lrs: tuple = (1e-4, 3e-4, 1e-3)
    sweep_steps: int = 3000
    sweep_directions: tuple = (("C1", "sw"), ("C2", "bn"), ("C3", "sr"))
    epochs: int = 15
    max_steps: int | None = None
    warmup_steps: int = 1000
    global_batch: int = 64
    per_device_batch: int = 16
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 4
    patience: int = 5
    max_chars: int = 512
    token_factor: int = 4
    optim: str = "adafactor"
    lr_scheduler: str = "constant_with_warmup"
    bf16: bool = True
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    logging_steps: int = 50
    dataloader_workers: int = 2
    dev_per_target: int = 200
    seed: int = 0


@dataclass(frozen=True)
class DecodingConfig:
    # Decoding defaults shared by evaluation and interpretability: beam size, length penalty,
    # input/output character caps and the token-cap factor (tokens <= factor * chars + 1).
    num_beams: int = 4
    length_penalty: float = 1.0
    max_input_chars: int = 512
    max_output_chars: int = 512
    token_factor: int = 4


@dataclass(frozen=True)
class InterpConfig:
    # Interpretability defaults: pilot subset sizes, character cap, batching budgets, bootstrap
    # settings, flush cadence, DecoderLens LID settings, LAPE cutoffs, steering grid/controls and
    # stitching donor pairs.
    seed: int = 0
    pilot_rows: int = 32
    pilot_wiki: int = 500
    max_chars: int = 512
    budget_tokens: int = 32_768
    max_batch: int = 64
    gen_max_batch: int = 32
    logprob_chunk_rows: int = 2048
    n_boot: int = 10_000
    alpha: float = 0.05
    flush_seconds: float = 60.0
    lid_threshold: float = 0.5
    lid_window: int = 128
    lape_ks: tuple = (0.5, 1.0, 2.0, 5.0)
    lape_percentile: float = 95.0
    steer_scales: tuple = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)
    steer_select_c: float = 1.0
    steer_controls: tuple = CONTROLS
    stitch_donors: tuple = (("C2", ("C1", "C3")), ("C3", ("C1", "C2")))


SMOKE: dict = {
    "DataConfig": {"nllb_cap": 2000, "wiki_per_lang": 500},
    "TrainConfig": {
        "sweep_steps": 60, "max_steps": 60, "warmup_steps": 10, "eval_steps": 20,
        "save_steps": 20, "global_batch": 8, "per_device_batch": 8, "dev_per_target": 8,
        "logging_steps": 10, "sweep_directions": (("C1", "sw"), ("C2", "sr")),
    },
    "DecodingConfig": {},
    "InterpConfig": {"pilot_rows": 8, "pilot_wiki": 100, "n_boot": 200, "flush_seconds": 5.0},
}


def smoke_overrides(cfg):
    # Input: a config dataclass instance. Output: a copy with the smoke-mode overrides for its
    # class applied (unchanged when the class has none).
    return dataclasses.replace(cfg, **SMOKE.get(type(cfg).__name__, {}))


def config_hash(obj) -> str:
    # Input: a dataclass instance, dict or other JSON-serialisable value. Output: the first 16 hex
    # characters of the SHA-256 of its canonical (sorted-key) JSON, used as a config fingerprint.
    data = dataclasses.asdict(obj) if dataclasses.is_dataclass(obj) else obj
    blob = json.dumps(data, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _resolve(spec, universe: tuple, groups: dict, restrict: tuple | None) -> list:
    # Inputs: a selection string (comma list of keys or group names, or None/"all"), the ordered
    # universe of keys, a group-name -> keys mapping and an optional restriction tuple. Output:
    # the selected keys in universe order, deduplicated; unknown names raise ValueError.
    tokens = [t.strip() for t in str(spec or "all").split(",") if t.strip()]
    chosen = set()
    for tok in tokens:
        if tok in groups:
            chosen.update(groups[tok])
        elif tok in universe:
            chosen.add(tok)
        else:
            raise ValueError(f"unknown selection {tok!r}; choose from {sorted(set(universe) | set(groups))}")
    pool = universe if restrict is None else tuple(k for k in universe if k in restrict)
    return [k for k in pool if k in chosen]


def resolve_models(spec="all", smoke: bool = False) -> list:
    # Input: model selection ("all", a size, an arch, keys, or a comma mix) and smoke flag. Output:
    # the ordered list of model keys (restricted to the small models under smoke).
    groups = {"all": tuple(MODELS)}
    groups.update({s: tuple(k for k, m in MODELS.items() if m.size == s) for s in SIZES})
    groups.update({a: tuple(k for k, m in MODELS.items() if m.arch == a) for a in ARCHS})
    return _resolve(spec, tuple(MODELS), groups, SMOKE_MODELS if smoke else None)


def resolve_conditions(spec="all", smoke: bool = False) -> list:
    # Input: condition selection ("all" or comma list of C1..C3) and smoke flag. Output: the
    # ordered list of condition keys (restricted to the smoke conditions under smoke).
    return _resolve(spec, tuple(CONDITIONS), {"all": tuple(CONDITIONS)}, SMOKE_CONDITIONS if smoke else None)


def resolve_langs(spec="all", pool: tuple = TARGETS, smoke: bool = False) -> list:
    # Input: language selection ("all" = the whole pool, group names targets/hubs/controls/study/all_langs,
    # or codes), the pool it is drawn from and smoke flag. Output: ordered language codes;
    # under smoke, targets outside SMOKE_TARGETS are dropped.
    groups = {"all": pool, "targets": TARGETS, "hubs": HUBS, "controls": CONTROLS,
              "study": STUDY_LANGS, "all_langs": ALL_LANGS}
    restrict = tuple(c for c in ALL_LANGS if c not in TARGETS or c in SMOKE_TARGETS) if smoke else None
    return _resolve(spec, ALL_LANGS, groups, restrict)


def resolve_sizes(models: list) -> list:
    # Input: a list of model keys. Output: their distinct sizes in canonical order.
    present = {MODELS[m].size for m in models}
    return [s for s in SIZES if s in present]


def resolve_testsets(spec="all") -> list:
    # Input: test-set selection ("all" or comma list). Output: ordered test-set names.
    return _resolve(spec, TESTSETS, {"all": TESTSETS}, None)


def resolve_metrics(spec="all") -> list:
    # Input: metric selection ("all", "sacre", "neural" or comma list). Output: ordered metric names.
    groups = {"all": METRICS, "sacre": ("chrf_pp", "bleu", "spbleu"), "neural": ("comet", "metricx")}
    return _resolve(spec, METRICS, groups, None)


def resolve_lrs(spec=None, cfg: TrainConfig | None = None) -> list:
    # Input: learning-rate selection (comma list of floats, or None for the sweep grid) and an
    # optional TrainConfig. Output: list of float learning rates.
    if spec in (None, "", "all"):
        return list((cfg or TrainConfig()).lrs)
    return [float(t) for t in str(spec).split(",") if t.strip()]


def model_for(arch: str, size: str) -> str:
    # Inputs: architecture family and size. Output: the model key, e.g. ("byt5", "small") -> "byt5-small".
    key = f"{arch}-{size}"
    if key not in MODELS:
        raise KeyError(key)
    return key


def lang_field(codes, attr: str) -> list:
    # Inputs: language codes and a Language attribute name. Output: that attribute for each code.
    return [getattr(LANGUAGES[c], attr) for c in codes]


def validate_registry() -> None:
    # Checks the language registry and everything derived from it: tag sentinels within mT5's 100
    # (k <= MAX_TAG_K), scripts covered by SCRIPT_RANGES, NTREX/CCMatrix/Wikipedia codes for every study
    # language, Wikipedia codes for every control, GlotLID label overrides, smoke selections, sweep directions and controls drawn from the
    # registry, one NLLB source per hub and target, and candidates that are new codes with
    # NTREX/CCMatrix/Wikipedia codes and covered scripts. Raises ValueError naming every problem. No output.
    problems = [f"tag k {l.tag_k} of {c} exceeds {MAX_TAG_K}; lower TAG_K0" for c, l in LANGUAGES.items()
                if l.tag_k > MAX_TAG_K]
    problems += [f"{c}: script {l.script} has no SCRIPT_RANGES entry" for c, l in LANGUAGES.items()
                 if SCRIPT_ALIASES.get(l.script, l.script) not in SCRIPT_RANGES]
    problems += [f"GlotLID label override for unknown language {c}" for c in LID_LABEL_OVERRIDES if c not in LID_LANGS]
    problems += [f"candidate {c} is already a study language" for c in CANDIDATES + CANDIDATE_HUBS if c in LANGUAGES]
    problems += [f"candidate hub {c}: missing NTREX/CCMatrix/Wikipedia code" for c, l in CANDIDATE_HUB_LANGUAGES.items()
                 if None in (l.ntrex, l.ccmatrix, l.wiki)]
    problems += [f"candidate {c}: missing NTREX/CCMatrix/Wikipedia code or script without SCRIPT_RANGES entry"
                 for c, l in CANDIDATE_LANGUAGES.items() if None in (l.ntrex, l.ccmatrix, l.wiki)
                 or SCRIPT_ALIASES.get(l.script, l.script) not in SCRIPT_RANGES]
    problems += [f"{c}: study language without NTREX, CCMatrix or Wikipedia code" for c in STUDY_LANGS
                 if None in (LANGUAGES[c].ntrex, LANGUAGES[c].ccmatrix, LANGUAGES[c].wiki)]
    problems += [f"control {c}: missing Wikipedia code" for c in CONTROLS if LANGUAGES[c].wiki is None]
    problems += [f"smoke target {c} is not a target" for c in SMOKE_TARGETS if c not in TARGETS]
    problems += [f"smoke condition {c} is not a condition" for c in SMOKE_CONDITIONS if c not in CONDITIONS]
    problems += [f"condition {k} uses unknown languages" for k, c in CONDITIONS.items()
                 if c.hub not in HUBS or not set(c.targets) <= set(TARGETS)]
    sweeps = TrainConfig().sweep_directions + smoke_overrides(TrainConfig()).sweep_directions
    problems += [f"sweep direction {c}-{t} is not a condition target" for c, t in sweeps
                 if c not in CONDITIONS or t not in CONDITIONS[c].targets]
    problems += [f"steering control {c} is not a control" for c in InterpConfig().steer_controls if c not in CONTROLS]
    problems += [f"NLLB source missing for {h}-{x}" for h in HUBS for x in TARGETS if (h, x) not in NLLB_SOURCES]
    if problems:
        raise ValueError("invalid language registry: " + "; ".join(problems))


validate_registry()

