# Discrepant-resolution bias in benchmark relabelling

Code and data for the paper *Only Checking Where the Model Disagrees: Discrepant-Resolution Bias in Benchmark Relabelling* (JMLR submission draft, `paper/main.pdf`; online appendix `paper/online_appendix.pdf`).

## Reproduce

```sh
./run_all.sh              # from the included derived data: experiments -> tables -> figures -> PDF
./run_all.sh --from-raw   # also re-download MMLU-Redux 2.0 and HELM v1.3.0 (~680 MB) and rebuild the data
./run_all.sh --with-conll # also rebuild the CoNLL-2003 re-analysis (Reiss et al. 2020 vs CoNLL++)
```

Requirements: [uv](https://docs.astral.sh/uv/) (it installs the Python dependencies from `pyproject.toml`/`uv.lock`), and pdfTeX for the PDF.

All numbers in the paper come from `paper/numbers.tex` and `paper/tables/*.tex`. `code/make_tables.py` generates both from `results/*.json`.

## Layout

| Path | What it holds |
|---|---|
| `code/drcore.py` | DR labels, bias identity, two-phase difference estimator, sharp sensitivity bounds |
| `code/build_dataset.py` | joins MMLU-Redux 2.0 reference labels with HELM v1.3.0 predictions (37 models) into `data/derived/redux_helm.parquet` |
| `code/experiments.py` | the experiments for Scenario A (MMLU labels) and Scenario B (LLM-generated labels): bias, two-phase, allocation, lock-in, sensitivity, rate |
| `code/conll/` | re-analysis of the published CoNLL-2003 DR audit (Reiss et al. 2020) against CoNLL++. `analyse_conll.py` runs from the released text-free files `data/derived/conll_items.csv` (eng.testb line, document and token index; original / Reiss / CoNLL++ entity types; adjudication flags) and `data/derived/conll_preds.csv` (21 models' predicted types). `build_conll_items.py` rebuilds them from the corpus. |
| `data/lit/` | coding of 22 relabelling studies (`design_coding.csv`, verbatim quotes), the verified `lit.bib`, and prior-work notes |
| `paper/` | the LaTeX source in the JMLR style (`jmlr2e.sty`, preprint option); build with `paper/build.sh` (needs TinyTeX/pdfTeX) |

## Data licences

- MMLU-Redux 2.0: CC-BY-4.0.
- HELM outputs: public, from the HELM GCS bucket.
- CoNLL-2003 (Reuters RCV1) is licensed for research use only. This repository releases token indices and labels only, never the corpus text; `./run_all.sh --with-conll` fetches the corpus to rebuild them.
- Downloaded raw data is not committed.
