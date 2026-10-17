# ByteResearch

Byte-level (ByT5) vs subword (mT5) encoder-decoders fine-tuned for translation from a hub language
(English, Arabic, Russian) into lower-resource targets, followed by interpretability experiments on the
fine-tuned models. All code lives in [`ByteResearch/`](ByteResearch/); see
[`ByteResearch/README.md`](ByteResearch/README.md) for installation, the stage CLI and the result layout.

```bash
pip install -r ByteResearch/requirements.txt
python -m ByteResearch.run --help
```

The Colab notebook [`ByteResearch/ByteResearch_colab.ipynb`](ByteResearch/ByteResearch_colab.ipynb)
runs data preparation, training, translation evaluation and the interp stages, saving everything to
Google Drive.
