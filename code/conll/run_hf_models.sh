#!/bin/sh
# Predictions of public CoNLL-2003 models, in an isolated environment (transformers conflicts with the
# project's huggingface-hub pin). Each model's cache is removed after use to save disk.
set -e
cd "$(dirname "$0")/../.."
for pair in "dslim/bert-base-NER bert_base_dslim" \
            "elastic/distilbert-base-cased-finetuned-conll03-english distilbert_elastic" \
            "dslim/bert-large-NER bert_large_dslim" \
            "dbmdz/bert-large-cased-finetuned-conll03-english bert_large_dbmdz" \
            "Jean-Baptiste/roberta-large-ner-english roberta_large_jb"; do
  set -- $pair
  uv run --isolated --no-project --python 3.12 --with torch --with transformers --with safetensors \
      python -I code/conll/predict_hf.py "$1" "$2" 2>&1 | grep -v -i "warn" | tail -1
  rm -rf "$HOME/.cache/huggingface/hub/models--$(echo "$1" | sed 's#/#--#')"
done
