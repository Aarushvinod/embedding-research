"""Default languages for the GATE-GRAFT feasibility run, stratified by difficulty.

The whole point is the degradation curve: alignment should be good for `ca` and decay
toward `eu`, and the gate should *predict* where it decays.
"""

# target language -> human-readable tier (English is always the anchor space)
TIERS = {
    "ca": "Near   (Romance, Latin script) — expect strong alignment",
    "tr": "Mid    (Turkic, agglutinative, Latin) — expect moderate",
    "bn": "Mid    (Indo-Aryan, Bengali script) — expect weaker",
    "fi": "Far    (Uralic, agglutinative, Latin) — distant but Latin; has a MUSE dict",
    "eu": "Far    (isolate, Latin) — expect poor; the gate should flag it",
}

DEFAULT_LANGS = ["ca", "tr", "bn", "eu"]

# alignment / gate defaults (kept small so it runs in minutes on a laptop)
DEFAULT_K = 2000          # top-k frequent words per language for GW
DEFAULT_EPSILON = 5e-3    # entropic GW regularization
DEFAULT_KNN = 10          # neighbourhood size for CSLS + the mutual-kNN gate signal
