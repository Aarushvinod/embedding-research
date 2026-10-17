# Interpretability experiments (implemented in a later build stage).
# This package will hold hubness, DecoderLens, LAPE, steering and cross-model stitching.
# Each measurement registers as a stage named interp.<experiment>.<measurement>.
# The prefix interp.<experiment> automatically acts as a composite of its measurements.
# Results go to ROOT/results/interp_<experiment>/<run_id>/<measurement>.json.
# It is intentionally empty until the experiment agents fill it in.
