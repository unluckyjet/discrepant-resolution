# Adjudication designs in ML label-error and benchmark-relabelling studies

Companion to `design_coding.csv` (23 rows) and `lit.bib` (23 entries). Each design code rests on verbatim methods text that `scripts/build_coding.py` matches against the extracted full text. It normalises ligatures, curly quotes, whitespace and hyphens, then records the PDF page index (the page's position in the PDF file, not the printed page number). All 23 rows' quotes were found. Full texts are in `pdfs/`, page-marked extractions are in `txt_pages/`, and the SWE-bench Verified blog text is in `web/`.

## Design categories

- **DR**: only items flagged by a model or algorithm (discordant with the label) are adjudicated. Unflagged items keep the original label.
- **DR-multi**: DR where the flag is the union over a panel of models (any model disagrees).
- **RANDOM**: a random or stratified random sample is re-annotated, independent of model outputs.
- **FULL**: every item is re-annotated.
- **CRS**: flagged items plus a sample of unflagged items (composite reference standard, two-phase).
- **OTHER**: hybrid or one-sided designs, described per study.

## Counts

| Design | n | Studies |
|---|---|---|
| DR | 6 | northcutt2021pervasive, nahum2025are, vasudevan2022when, chong2022detecting, reiss2020identifying, ansari2026how |
| DR-multi | 2 | beyer2020are, vendrow2025do |
| RANDOM | 2 | gema2025are, chowdhury2024introducing |
| FULL | 5 | wang2019crossweigh, stoica2021retacred, zhai2026hleverified, brunello2026fixing, sylvestre2026gold |
| CRS | 1 | alt2020tacred (partial; see below) |
| OTHER | 6 (+1 survey) | shankar2020evaluating, wang2024mmlupro, rucker2023cleanconll, ye2026scalable, land2026auditing, aleithan2024swebench; klie2023annotation is a survey, not a relabelling |

That gives 22 relabelling or audit studies plus 1 survey.

Five of the six OTHER studies also have a discordant-only human step:

- the MMLU-Pro distractor phase;
- the CleanCoNLL cross-checking and its evaluation sample;
- MedCalc's physician adjudication;
- Land & Bikel's hand review;
- SWE-Bench+'s success-only audit.

Counting those, **13 of 22 studies send humans only to model-discordant items (or, for SWE-Bench+, only model-success items) for at least part of the adjudication.** Only one study (Alt et al. 2020) samples concordant items alongside flagged ones, and it does not use that sample to correct the accuracy estimates.

### Cross-tabs for the 8 DR/DR-multi studies

| | yes | partial | no |
|---|---|---|---|
| Re-reports accuracy or ranking on corrected labels | 7 | 1 (Chong) | 0 |
| A flagging model is also among the evaluated models | 7 | 1 (Chong) | 0 |
| Acknowledges that unflagged items may be wrong | 4 (Northcutt, Vasudevan, Vendrow, Chong) | 0 | 4 (Nahum, Beyer, Reiss, Ansari) |

Beyer et al. go further than not acknowledging it: they assert the opposite, that agreed items "can safely retain the original without re-evaluation".

## Per-study notes

**Northcutt, Athalye & Mueller 2021 (NeurIPS D&B), DR.**
- Confident learning (ResNet-50 for ImageNet; VGG, FastText and others elsewhere) flags candidates, and 5 MTurk workers review only those. Unflagged items join the "benign set" with their original labels.
- Corrected accuracy and the 34-model ImageNet ranking are recomputed, and ResNet-50, the flagging model, is in that ranking.
- Section 6 adds a CRS-like expert review of one random non-flagged image per class. It finds 16% errors among non-flagged images and revises ImageNet prevalence to about 20% (from 6%). That estimate is used only for prevalence, never to correct labels or re-rank models.
- The paper calls its analysis "a loose lower bound … due to unaccounted for label errors in the non-CL-flagged data".

**Nahum et al. 2025 (EMNLP 2025; arXiv 2410.18889), DR.**
- An ensemble of GPT-4, PaLM2, Mistral-7B and Llama-3 is averaged over prompts. Experts see only items where the ensemble prediction differs from the original label.
- Gold label = "the original label, if the LLM prediction agrees with it, or the expert resolution".
- Models are then re-ranked on these gold labels; GPT-4, PaLM2, Llama3 and Mistral are among those re-ranked. GPT-4 moves from rank 3 to 1, and DeBERTa-v3 drops from 2 to 8.
- I found no acknowledgement of errors among agreed items. The Table 2 "lower bound" is a Clopper-Pearson sampling bound.
- Venue correction: the brief says EMNLP 2024, but Crossref and ACL Anthology give EMNLP 2025 (2025.emnlp-main.1360).

**Vasudevan et al. 2022 (NeurIPS), DR.**
- A five-expert panel reviews only the 676 apparent mistakes of ViT-3B (and later Greedy Soups) on the Shankar multi-label subset.
- Multi-label accuracy of those same models is re-reported: "nearly half" of their mistakes are judged correct.
- The limitations section states the bias explicitly: "we never review any validation image whose groundtruth might be wrong, if all models evaluated also make the incorrect groundtruth prediction."

**Beyer et al. 2020 (ReaL; arXiv), DR-multi.**
- Six proposer models (selected from 19) generate label proposals. "In the event that all models agree with the original ImageNet label, we can safely retain the original without re-evaluation", which cut the annotation set from 50,000 to 24,889 images.
- The proposers (BiT-L, BiT-M, ResNeXt IG, Inception v3, CPC v2) are scored with ReaL accuracy.
- A 256-image expert gold set was used only to choose the proposer subset (97.1% proposal recall), not to audit the retained images.

**Shankar et al. 2020 (ICML), OTHER.**
- Every image in a 20k ImageNet-val subset (plus ImageNetV2) was reviewed, but only the classes predicted by at least one of 72 models were judged: full item coverage, with a model-proposed label space.
- All 72 models are scored with multi-label accuracy.
- The paper acknowledges that "each image only contained reviewed labels for classes predicted by models" and so re-reviewed the human labelers' predictions.
- Confidence is medium. The main text does not say whether the original ImageNet label was always a judged candidate, or how the 20k subset was chosen.

**Gema et al. 2025 (MMLU-Redux; NAACL 2025), RANDOM.**
- 100 random questions per subject (5,700 total) are annotated independently of model outputs.
- HELM leaderboard models are re-ranked on the five most error-prone subjects. Llama 3.1 405B goes from 16th to 1st on Virology.

**Vendrow et al. 2025 (platinum benchmarks; arXiv), DR-multi.**
- Applied to random subsets of 15 benchmarks: "manually inspect any example on which at least one LLM makes an error".
- The panel is the set of evaluated frontier models, whose error counts are then reported.
- VQA v2.0 is the exception: its subset was fully re-labelled.
- The paper explicitly says its error counts are a lower bound because "our approach could miss any bad question for which all LLM solutions happen to agree with the benchmark solution".

**Wang et al. 2024 (MMLU-Pro; NeurIPS 2024 D&B), OTHER.**
- Phase 1 is a FULL expert check of every answer key. Phase 2 is DR on distractors: Gemini-1.5-Pro flags possible false-negative options, and experts review only those.
- Gemini-1.5-Pro also appears in the results table.
- As a new benchmark, it gives no before/after accuracy on the same items.
- Confidence is medium because the two phases have different designs.

**Chong, Hong & Manning 2022 (EMNLP), DR.**
- This is a detection-method paper. Crowd verification (the Northcutt protocol) covers only top-ranked hypothesised items, and recall is computed by assuming 5% prevalence.
- It acknowledges that true recall "requires a complete re-evaluation".
- Most relevant to this paper, it warns that model-based cleaning "may bias any such benchmarks toward the models most similar to the model used to clean the data."
- Re-reporting is partial: an end-to-end cleaning experiment uses synthetic-noise splits.

**Klie, Webber & Gurevych 2023 (Computational Linguistics), OTHER (survey).**
- Not a relabelling study, and excluded from the counts.
- Cited for its method statement: AED recall needs experts to "carefully annotate a subset of the data and then use it to estimate recall overall".

**Wang et al. 2019 (CrossWeigh; EMNLP-IJCNLP), FULL.**
- Every CoNLL-2003 test sentence was checked by two of five trained annotators, then given a final verification round where they disagreed with each other or with the original.
- Models are re-evaluated on the corrected test set.

**Reiss et al. 2020 (CoNLL), DR.**
- Model ensembles flag positions where "the models agreed strongly on a particular label, but that label does not appear in the corpus". The ensembles are the 16 original shared-task entries plus two 17-model BERT random-projection ensembles, deliberately low-F1.
- Only flagged labels (plus 470 errors found nearby by chance) are reviewed.
- The 16 shared-task entries are re-scored, with no ranking change. I found no acknowledgement of unflagged errors.

**Alt, Gabryszak & Hennig 2020 (TACRED Revisited; ACL), CRS (partial).**
- The "Challenging" set (misclassified by at least half of 49 models) plus a "Control" set of up to 20 random examples per relation, drawn from items classified correctly by at least 39 models.
- This is the only two-phase design found. But the control stratum skips items with 25–38 correct models, and its 8.9% revision rate is never extrapolated to the unsampled items: the "revised test split" leaves them uncorrected.
- All 49 models are re-scored (average F1 62.1 to 70.1).
- The paper concedes that "our selection strategy was biased towards examples challenging for models".

**Stoica et al. 2021 (Re-TACRED; AAAI), FULL.**
- Re-annotates the entire TACRED dataset.
- Explicitly criticises Alt et al.'s "small and biased sample set" and their analysis over "a predominately uncorrected version".
- Models are re-scored (+14.3 F1 on average).

**Rücker & Akbik 2023 (CleanCoNLL; EMNLP), OTHER.**
- Phase 1 relabels automatically from independent expert AIDA entity links, with manual typing of 1,432 unmapped links.
- Phase 2 is DR: three rounds of cross-checking where "all predictions that differ from the annotated NER labels are flagged" and only flagged items are inspected.
- The quality evaluation samples only sentences where corpus versions differ, so it is also discordant-only.
- FLERT, the round-1 flagger's architecture, is among the four models re-evaluated.
- The limitations admit that errors could have "fell through our cross-checking approach".

**Zhai et al. 2026 (HLE-Verified; arXiv), FULL.**
- Each of the 2,500 HLE items gets expert component-wise validation, with model pass@8 runs as auxiliary evidence only.
- States outright that "high solver agreement is not regarded as proof of correctness."
- Eight frontier models are re-scored (+7–10 points overall). The solver models used in auditing are not named, so the flagger/evaluated overlap is unknown.

**Ye et al. 2026 (MedCalc-Bench stewardship; arXiv 2512.19691), OTHER.**
- Gemini 2.5 Pro recomputes every label. Physicians adjudicate only 50 instances sampled from the top of the original-vs-recomputed disagreement ranking.
- Four newer frontier models are then scored on 887 items against the LLM labels. The justification is physician agreement measured on discordant items only; no concordant items were physician-checked.
- The flagger was not itself evaluated (Gemini 3.1 is evaluated, not 2.5 Pro).
- Acknowledges that shared-model agents' "systematic biases … would not be caught by majority voting".
- The arXiv BibTeX gives year 2026 (the arXiv ID is from Dec 2025), hence the key ye2026scalable.

**Land & Bikel 2026 (IRT auditing; EMNLP 2026 per arXiv), OTHER.**
- A detection paper. Weak reference labels for every item come from GPT-5.4 aggregating strong-model judgments; humans review only items above the flag threshold. The 8% of items on which all models agree are dropped before fitting.
- Its unflagged error rate (3%) rests on LLM weak labels, not human review.
- No corrected leaderboard. Acknowledges that removing the aggregator "does not rule out biases shared across the panel".
- Confidence is medium.

**Ansari et al. 2026 (physics expert re-grading; arXiv), DR.**
- On HLE-Physics, PHYBench, PRISM-Physics and UGPhysics, the audit is restricted "to questions for which all of GPT-5.6-Sol High's attempts were evaluated as incorrect". The 252 accepted questions were never reviewed.
- CMT-Benchmark and a 56-challenge CritPt subset were fully reviewed, finding errors in 30/50 and 21/56.
- GPT-5.6-Sol is re-scored (for example HLE-Physics 47% to 79%). I found no acknowledgement of errors among accepted items.
- The model names are as printed in the paper.

**Brunello et al. 2026 (FOLIO/MALLS; EMNLP 2026 per arXiv), FULL.**
- A full human audit of the FOLIO validation split (275) and the first 100 MALLS test items. Three LLMs are re-scored (+11 to +23 points).
- The full audit is then the reference for scoring a DR-style LLM triage framework. That framework reaches "90% dataset accuracy after reviewing fewer than 20% of instances": residual errors stay in the unreviewed items.

**Sylvestre 2026 (SciFact; BioNLP@ACL 2026), FULL.**
- GPT-5.4-mini screens all 209 dev pairs. Flagged items are adjudicated, and all 152 unflagged items are also "exhaustively reviewed".
- Three of the 11 confirmed errors were in the unflagged set (auditor recall 73%). A DR design would have missed 27% of errors.
- GPT-5.4-mini, the flagger, is re-scored (+3.8 F1).
- Single annotator.

**Aleithan et al. 2024 (SWE-Bench+; arXiv), OTHER (mirror of DR).**
- Audits only instances SWE-Agent+GPT-4 "resolved" (all tests pass), looking for false passes. Failed instances are never reviewed.
- The same system's resolution rate is re-reported (12.47% to 3.97%).

**Chowdhury et al. 2024 (SWE-bench Verified; OpenAI blog), RANDOM.**
- "We annotated 1,699 random samples from the SWE-bench test set"; each sample was screened by 3 developers.
- GPT-4o is reported at 33.2% on Verified vs 16% on the original.
- Sourced from a Wayback Machine copy, because openai.com serves a bot challenge. A blog post, not peer reviewed.

## Caveats

- Page locations are PDF page indices from pypdf extraction. Quotes are verbatim apart from ligature, curly-quote and line-break-hyphen normalisation.
- `lit.bib` uses arXiv `@misc` entries for papers whose proceedings versions exist (Northcutt, Vasudevan, MMLU-Pro; Land & Bikel and Brunello are listed as accepted to EMNLP 2026). Swap these for proceedings entries if needed.
- The Stoica entry comes from DOI content negotiation; I changed only its page-range dash to `--`.
- Not covered: HellaSwag, GSM8K-only, SQuAD and Natural Questions audits. GSM8K and SQuAD2.0 appear inside Vendrow et al. One S2 search for HellaSwag audits returned nothing relevant.
