# Steering experiment: does removing an English direction hurt translation more than removing a control?
# Directions are paired DiffMeans of English minus X on FLORES-dev half A, with Spanish, Chinese,
# Greek and random controls rescaled to the English norm at every layer.
# C2 and C3 edit the encoder residual stream; C1 edits the decoder block outputs from the tag on.
# The best single layer is chosen on dev half B, and the full scale grids plus the mean-centered
# projection run on devtest, read out as X log-probs and translation metrics.
# Importing measure, summary and compare registers the interp.steering.* stages and the composite.
