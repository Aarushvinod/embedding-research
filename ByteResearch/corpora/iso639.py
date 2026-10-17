# ISO 639-3 macrolanguage membership, the basis of the GlotLID label sets accepted for each language.
# build_iso639 downloads SIL's iso-639-3-macrolanguages.tab once into ROOT/data/models/iso639_3, checks it
# against the SHA-256 pinned in core.config and records it in a manifest like every other cached asset.
# parse_macrolanguages reads the table into {macrolanguage: member codes}, and iso_family gives the codes
# that count as one language: the whole macrolanguage when the code is one or belongs to one, else itself.
# derive_aliases turns the language registry, that table, the labels present in the GlotLID model and the
# small LID_EXTRA_CODES override into {code: accepted labels}, keeping only labels in the expected script.
# Everything except the download is pure, so the derivation is unit-tested on fake tables and label lists.

from __future__ import annotations

from ..core.config import PINS
from ..core.io import write_bytes_atomic
from ..core.text import canonical_script
from .http import fetch_bytes
from .manifest import begin_build, finish_build, manifest_complete, verified_path

LABEL_PREFIX = "__label__"


def iso_dir(ctx):
    # Input: Ctx. Output: ROOT/data/models/iso639_3, the cached macrolanguage table.
    return ctx.paths.models_dir() / "iso639_3"


def build_iso639(ctx) -> dict:
    # Input: Ctx. Downloads the pinned macrolanguage table, refuses it unless its SHA-256 matches the pin,
    # and writes it with its manifest. Output: the manifest.
    pin = PINS["iso639_3"]
    out = begin_build(ctx, iso_dir(ctx))
    data, info = fetch_bytes(pin["url"])
    if info["sha256"] != pin["sha256"]:
        raise ValueError(f"{pin['url']} has SHA-256 {info['sha256']}, pinned {pin['sha256']}; update PINS['iso639_3']")
    write_bytes_atomic(out / pin["file"], data)
    return finish_build(ctx, out, sources=[{**info, "release": pin["release"]}], config={"pin": pin},
                        outputs=[pin["file"]])


def iso_table_path(ctx):
    # Input: Ctx. Output: the verified local macrolanguage table, downloading it first when missing.
    if not manifest_complete(iso_dir(ctx)):
        build_iso639(ctx)
    return verified_path(iso_dir(ctx), PINS["iso639_3"]["file"])


def parse_macrolanguages(text: str) -> dict:
    # Input: the text of iso-639-3-macrolanguages.tab (header M_Id, I_Id, I_Status, tab separated).
    # Output: {macrolanguage code: frozenset of member codes}, retired members included.
    table = {}
    for line in text.splitlines()[1:]:
        cols = [c.strip() for c in line.split("\t")]
        if len(cols) >= 2 and cols[0] and cols[1]:
            table.setdefault(cols[0], set()).add(cols[1])
    return {macro: frozenset(members) for macro, members in table.items()}


def iso_family(code: str, macro: dict) -> set:
    # Inputs: an ISO 639-3 code and the parsed macrolanguage table. Output: the codes treated as the same
    # language: the macrolanguage and all its members when the code is a macrolanguage or a member of one,
    # otherwise just the code.
    parent = code if code in macro else next((m for m, members in macro.items() if code in members), None)
    return {code} if parent is None else {parent, *macro[parent]}


def split_label(label: str):
    # Input: a GlotLID label such as "__label__arz_Arab" or "arz_Arab". Output: (language code, script), or
    # None when the label has no script part.
    parts = label.removeprefix(LABEL_PREFIX).split("_", 1)
    return tuple(parts) if len(parts) == 2 else None


def derive_aliases(languages: dict, macro: dict, labels, extra: dict | None = None) -> dict:
    # Inputs: {code: Language} (iso639_3 and script are read), the parsed macrolanguage table, the labels
    # present in the GlotLID model and {code: extra ISO codes} for cases the table does not cover. Output:
    # {code: frozenset of accepted labels}: every label whose language is in the family of the language's
    # own or extra codes and whose script matches the language's script (aliases such as Hans = Hani).
    pairs = [p for p in map(split_label, labels) if p]
    out = {}
    for code, lang in languages.items():
        family = set().union(*(iso_family(c, macro) for c in (lang.iso639_3, *(extra or {}).get(code, ()))))
        script = canonical_script(lang.script)
        out[code] = frozenset(f"{c}_{s}" for c, s in pairs if c in family and canonical_script(s) == script)
    return out
