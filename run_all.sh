#!/bin/sh
# Reproduce results, tables, figures and the PDF.
#   ./run_all.sh            start from the released derived data (data/derived/redux_helm.parquet)
#   ./run_all.sh --from-raw first re-download HELM predictions and rebuild the derived data
#   ./run_all.sh --with-conll  also rebuild the CoNLL-2003 re-analysis (downloads the corpus and audit data)
set -e
cd "$(dirname "$0")"
if [ "$1" = "--from-raw" ]; then
  uv run python -I -c "from huggingface_hub import snapshot_download as s; s('edinburgh-dawg/mmlu-redux-2.0', repo_type='dataset', local_dir='data/raw/mmlu-redux-2.0')"
  uv run python -I code/fetch_helm.py               # HELM v1.3.0 per-instance predictions (~680 MB)
  uv run python -I code/build_dataset.py            # join with MMLU-Redux 2.0
fi
if [ "$1" = "--from-raw" ] || [ "$2" = "--with-conll" ] || [ "$1" = "--with-conll" ]; then
  # CoNLL-2003 re-analysis inputs (research-use text is not redistributed; see code/conll/)
  sh code/conll/fetch_conll.sh
  (cd data/raw/conll && uv run --isolated --no-project --python 3.10 --with text-extensions-for-pandas \
      --with "pandas<2" --with "numpy<2" python -I ../../../code/conll/export_offsets.py eng.testb tp_test_tokens.csv \
   && uv run --isolated --no-project --python 3.10 --with text-extensions-for-pandas --with "pandas<2" \
      --with "numpy<2" python -I ../../../code/conll/apply_reiss_labels.py reiss/scripts eng.testb \
      reiss/corrected_labels/all_conll_corrections_combined.csv reiss_label_corrected_test.txt)
  mkdir -p data/raw/conll/entrants
  for t in bender carrerasa carrerasb chieu curran demeulder florian hammerton hendrickx klein mayfield mccallum munro whitelaw wu zhang; do
    curl -sSL -o data/raw/conll/entrants/$t.testb https://raw.githubusercontent.com/CODAIT/text-extensions-for-pandas/master/resources/conll_03/ner/results/$t/eng.testb
  done
  [ -f data/derived/conll_pred_roberta_large_jb.txt ] || sh code/conll/run_hf_models.sh
  uv run python -I code/conll/build_conll_items.py  # text-free data/derived/conll_items.csv, conll_preds.csv
fi
uv run python -I code/conll/analyse_conll.py        # results/conll.json (runs from the released derived files)
uv run python -I code/experiments.py                # results/*.json (two-phase Monte Carlo takes ~2 min)
uv run python -I code/make_lit_table.py
uv run python -I code/make_tables.py                # paper/tables/*.tex, paper/numbers.tex
uv run python -I code/figures.py                    # paper/figures/*.pdf
sh paper/build.sh                                   # needs pdfTeX (TinyTeX/TeX Live) with multirow, lastpage, lm
