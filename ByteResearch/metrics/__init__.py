# Translation metrics for ByteResearch: chrF++, spBLEU, COMET and MetricX.
# chrF++ and spBLEU are computed in-process with sacreBLEU and keep per-sentence statistics.
# COMET and MetricX need transformers below version 5, so they run in a separate venv.
# metrics.api is the public entry point (METRICS, score_rows, score_result_rows) used by every stage.
# The worker module is a standalone script run by that venv, so discovery must never import it.
# env.py registers the setup-metrics stage that builds the venv and runs a self-check.

DISCOVER_SKIP = ("worker",)
