# Feasibility: re-analysing a real published DR audit (2026-10-08)

This summarises the feasibility agent's findings and adds one count we made ourselves. No re-analysis has been run.

## Verdicts

| # | DR audit → independent reference | Verdict | Effort |
|---|---|---|---|
| 1 | CoNLL-2003 test: Reiss et al. 2020 → CoNLL++ (Wang et al. 2019) | **GO**, with blockers below | 3–5 days |
| 2 | CIFAR-10 test: Northcutt et al. 2021 → CIFAR-10H (Peterson et al. 2019) | **GO** (counts below) | about 1 day |
| 3 | TACRED test: Alt et al. 2020 → Re-TACRED | CONDITIONAL: schema change; Control-group IDs not released | 3–4 days |
| 4 | SciFact dev: Sylvestre 2026 | CONDITIONAL: same annotator for flagged and unflagged items, so not independent; small | half a day |
| 5 | HLE-Physics: Ansari 2026 | CONDITIONAL: no item-level release found | – |
| – | Platinum MMLU HS-math → MMLU-Redux | NO-GO: all 100 overlapping items are "ok" | – |
| – | CleanCoNLL as a reference | NO-GO: built from Reiss's version | – |

## CoNLL-2003 (Reiss → CoNLL++)

- **Reiss release** (github.com/CODAIT/Identifying-Incorrect-Labels-In-CoNLL-2003, Apache-2.0):
  - candidate lists for each flagger ensemble, with agreement counts;
  - reviewed/kept verdicts, which let us rebuild the reviewed (= DR) set at the stop-rule thresholds;
  - 444 test-fold corrections;
  - the 16 entrants' token-level predictions (in text-extensions-for-pandas).
- **CoNLL++** (github.com/ZihanWangKi/CrossWeigh `data/conllpp_test.txt`, Apache-2.0) is a full two-annotator re-check of every test sentence, with the same tokenization.
- **Independence:**
  - CoNLL++ used no model flags.
  - Reiss cites Wang et al. only to compare error rates; its candidate sources are its own ensembles.
  - Caveat: CoNLL++ annotators corrected the existing labels rather than labelling blind. That anchors them to the original and biases *against* finding hidden errors.
- **Blockers:**
  - reconcile the test rows with the paper's 1,274 flagged and 421 errors;
  - choose the flag-set definition (the reviewed set vs "<7 models");
  - handle about 284 incidental corrections of unflagged items;
  - exclude Sentence/Token error types (about 63 test rows);
  - pick the item unit (sentence 3,453; mention 5,648);
  - possibly a small third adjudication where Reiss and CoNLL++ disagree.
- **Models:** HF models such as dslim/bert-base-NER, xlm-roberta-large-finetuned-conll03-english and flair/ner-english(-large). Running them takes minutes.
- **Licence:** Reuters RCV1 text is research-only, so we publish offsets and IDs, not text.

## CIFAR-10 (Northcutt → CIFAR-10H): error count outside the flag set

Our count (script inline in the session; files in `cifar_counts/`):

| CIFAR-10H majority ≠ original label | total | among the 275 flagged | among unflagged | unflagged and flagger = original label |
|---|---|---|---|---|
| any majority | 79 | 14 | 65 | – |
| majority share > 0.5 | 54 | 12 | 42 | 30 |
| majority share > 0.6 | 31 | 8 | 23 | 15 |
| majority share > 0.7 | 12 | 5 | 7 | 4 |

The two references often disagree. Northcutt's MTurk review confirmed 54 errors among the 275 flagged items, but the CIFAR-10H majority contradicts the original label on only 14 of them.

## MedCalc-Bench claim (Ye et al. 2026): verified, with tighter wording

- Gemini 2.5 Pro recomputed labels for the full test split (1,047), with high-confidence labels for 887.
- One physician reviewed 50 instances sampled from the top of the disagreement ranking, after a domain-fit screen.
- Four newer frontier models are scored on the 887 against LLM-recomputed labels, with physician labels substituted on the adjudicated subset.

Quotes are on PDF pp. 5–9, 11–13 and 16.
