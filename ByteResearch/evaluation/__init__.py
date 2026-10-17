# Translation evaluation for ByteResearch: translate, score and compare.
# testsets.py builds hub-to-target rows from FLORES devtest, NTREX and FLORES dev.
# generate.py translates them with every trained model through the shared generation path.
# Scores come from metrics.api and are stored in the same result file as the hypotheses.
# compare.py runs the ByT5 versus mT5 paired bootstrap and the optional difference-in-differences.
# stages.py registers the stages and the eval composite with core.registry on import.
