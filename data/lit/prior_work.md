# Prior work and novelty check: discrepant-resolution bias in benchmark relabelling

Compiled 2026-10-08 from Semantic Scholar (keyed API and MCP), arXiv, Crossref and full-text reads
(PDF text in `data/lit/txt/`). Bib keys refer to `paper/refs_methods.bib`. Where the claim comes
from the full text, the section or line is given. Items marked "abstract only" were not read beyond the abstract.

Setup recap: L = original label, F = flagger, Y = expert label. Adjudicate only F != L, keep L where
F = L. Contributions: (i) bias formula favouring F and correlates, with joint errors F = L != Y left uncorrected;
(ii) non-identification plus sharp bounds; (iii) two-phase stratified design with a difference/HT estimator
using L as proxy, Neyman allocation, and a Le Cam lower bound; (iv) MMLU-Redux + HELM re-analysis.

## Bottom line

- **(i) Formal bias of flagged-only adjudication in ML: not found.** Several ML relabelling papers *use*
  this design and some say in a sentence that their error counts are lower bounds or that rankings are
  "optimistic in favor of" the flagging models (Northcutt et al. 2021; Vendrow et al. 2025). None derive the
  bias, tie it to flagger-evaluated model correlation, or show it can reorder rankings. The bias itself is well
  known in diagnostic-test statistics (Hadgu 1996/1997/1999; Miller 1998; Green et al. 1998; Lipman & Astles 1998).
  So the result is a transfer and formalisation, not a discovery. Present it that way.
- **(ii) Non-identification and bounds:** partial overlap. Lam & Stork (2003) bound the true error under a
  noisy, possibly dependent test labeler. Chavoshi et al. (2025/26) give analytic best/worst cases when LLM
  labels are the reference. Ye et al. (2025) put a worst-case assumption on the unflagged stratum of a
  relabelled benchmark. None of them work under the discrepant design, where the discordant stratum is fully
  adjudicated and only the concordant stratum is unknown. Sharp bounds for that design look new.
- **(iii) Estimator and design:** **the estimator is not new.** Sampling a fraction q of concordant items,
  adjudicating every discordant item, and using L as the proxy is Tenenbein-style double sampling (1970) and the
  difference/HT estimator (Cassel, Särndal & Wretman 1976; Särndal et al. 1992). It is also a special case of
  PPI with L as the "prediction" (Angelopoulos et al. 2023; Mozer 2026 shows PPI = difference estimator). With
  strata given by the flag it is stratified PPI (Fisch et al. 2024), and with flag-dependent inclusion
  probabilities it is active inference (Zrnic & Candès 2024). IR evaluation has used stratified/HT sampling
  of relevance judgments to fix pooling bias since 2006 (Aslam et al.; Yilmaz & Aslam). What looks new:
  (a) using the **existing benchmark label L** as the proxy, with strata defined by **flagger-label discordance**;
  (b) the deterministic-design bias as the motivating failure; (c) a design-specific Neyman allocation and
  minimax rate. The paper should say plainly that the estimator is standard and claim novelty only for the
  framing, the analysis and the lower bound.
- **(iv) "Benchmark lock-in" (cleaned benchmarks inheriting the cleaners' shared errors): no paper uses this
  framing.** The closest ingredients: correlated LLM errors (Kim et al. 2025; Kohli 2026); IR pooling bias
  against systems outside the pool (Zobel 1998; Buckley et al. 2007); bias of a model-selected subset against
  the selecting models (Vasudevan et al. 2022). There is also direct evidence of missed joint errors when
  non-flagged items are audited (Northcutt 2021 Sec. 6; Sylvestre 2026).

## Closest prior works (ranked by overlap)

1. **Nahum et al. (2025), "Are LLMs Better than Reported?" (EMNLP 2025; arXiv 2410.18889)** `nahum2025are` (full text read).
   An ensemble of LLMs flags TRUE-benchmark items. Every item where the LLM prediction differs from the original
   label goes to two expert authors. The paper then states: "we define the gold label as the original label, if the
   LLM prediction agrees with it, or the expert resolution, if there was a disagreement" (Sec. 4.2). Nine
   models, mostly LLMs, are re-scored on this gold. All metrics shift upward ("LLMs are better than reported"),
   and the non-LLM DeBERTa-v3 drops from second-best to second-worst (Sec. 7.2). The paper never discusses
   that the design itself favours models correlated with the LLM flagger.
   **Verdict: textbook discrepant analysis in ML, with exactly the predicted symptom. It is the best motivating
   case study, not a competitor.**

2. **Northcutt, Athalye & Mueller (2021), "Pervasive Label Errors in Test Sets…" (NeurIPS D&B)** `northcutt2021pervasive` (full text read).
   Confident learning flags candidates, MTurk validates only flagged items, and "corrected" accuracy is computed
   on the corrected subset. The paper does say that "noise prevalence estimates are optimistic in favor of higher
   capacity models" because non-CL-flagged errors are unaccounted for (Sec. 5, end). In Sec. 6, experts review
   one random flagged and one random non-flagged image per ImageNet class: a stratified two-phase audit.
   Non-flagged images still contain errors, and flagged images are 2.6x more likely to be erroneous. The
   non-flagged sample is used only descriptively, never to build an unbiased accuracy estimator.
   **Verdict: the strongest partial overlap. They state the direction of the bias informally and run a
   concordant-stratum sample. They do not formalise the bias or estimate accuracy from the sample.**

3. **Vendrow et al. (2025), "Do Large Language Model Benchmarks Test Reliability?" (platinum benchmarks; arXiv 2502.03461)** `vendrow2025do` (full text read).
   Several frontier LLMs answer every question, and the authors "manually inspect any example on which at least
   one LLM makes an error". The flagger is the union of LLMs, and concordant items are never inspected. The paper
   notes that its error counts are "only a lower bound" because all models could agree with a bad label
   (Sec. 2). For VQA v2.0 it relabels every item, a census (App. B.1.1). No sampling, no bias analysis,
   no estimator. The evaluated models overlap with the flagger set, so the joint-error blind spot falls exactly
   where the paper claims "every model failure we report … is genuine".
   **Verdict: a prominent instance of the design. Its own caveat matches contribution (i) qualitatively.
   No formal overlap.**

4. **Wang et al. (2024), "Impact of Gold-Standard Label Errors on Evaluating Performance of Deep Learning Models in DR Screening" (JMIR; DOI 10.2196/52506)** `wang2024impact` (abstract only).
   Covers 736k fundus images. Ophthalmologists adjudicate a random sample of images where the DL algorithm
   disagrees with the human grade. The error rates found there correct the counts in the whole dataset, and the
   same DL algorithm is then re-evaluated: sensitivity rises 12.5 points and specificity 6.9 points. Concordant
   images are not sampled.
   **Verdict: discrepant analysis applied to medical-AI evaluation, with the flagger as the evaluated model.
   Another motivating example. Cite it alongside the Hadgu/Miller critique.**

5. **Fisch et al. (2024) Stratified PPI (NeurIPS) / Zrnic & Candès (2024) Active Statistical Inference (ICML) / Angelopoulos et al. (2023) PPI (Science), PPI++ (AoAS 2026)** `fisch2024stratified`, `zrnic2024active`, `angelopoulos2023prediction`, `angelopoulos2026ppi` (abstracts and known method).
   These methods combine a cheap proxy on all items with gold labels on a sample, using stratification or
   uncertainty-driven inclusion probabilities, and give unbiased estimators with valid CIs. In LLM-evaluation
   uses (Fisch, Boyeau, Chatzi, Eyre & Madras) the proxy is an autorater's verdict on the *evaluated model's
   output*, and the human labels are treated as gold. Here the proxy is instead the *pre-existing, noisy benchmark label*
   L, and the strata come from flagger-label discordance. Mathematically the two-phase estimator is a stratified
   PPI/difference estimator with L as the prediction and π = 1 on the discordant stratum.
   **Verdict: the estimator in (iii) is covered by this family, and Mozer (2026) already makes the
   survey-sampling equivalence explicit. Claim novelty for the setting, the bias and identification analysis,
   the allocation under this specific stratification, and the lower bound, not for the estimator.**

6. **Aslam, Pavlu & Yilmaz (2006) statAP; Yilmaz & Aslam (2006) infAP; Buckley et al. (2007); Zobel (1998)** `aslam2006statistical`, `yilmaz2006estimating`, `buckley2007bias`, `zobel1998how` (titles/venues verified, abstracts unavailable in S2).
   In TREC pooling, only documents retrieved by participating systems are judged and unjudged documents are
   assumed non-relevant. This favours pooled systems and penalises new ones, the IR analogue of "concordant items
   keep L". Zobel and Buckley et al. document the bias. statAP and infAP fix it by sampling judgments with
   known inclusion probabilities and using HT-type estimators.
   **Verdict: a 20-year-old precedent for both the bias (bias toward systems that built the judgment set)
   and the fix (probability sampling + HT). It is missing from the current framing and must be cited in
   related work.**

## Other relevant work, by question

### (1) Bias from adjudicating only model-flagged items

- **Diagnostic-test literature (the origin):** Hadgu 1996 Lancet `hadgu1996discrepancy`; Hadgu 1997 Stat Med
  `hadgu1997bias`; Hadgu 1999 J Clin Epidemiol `hadgu1999discrepant`; Hadgu 2000 JCM letter `hadgu2000discrepant`;
  Miller 1998 J Clin Epidemiol `miller1998bias` and CID editorial `miller1998can`; Green, Black & Johnson 1998 JCM
  `green1998evaluation`; Lipman & Astles 1998 Clin Chem `lipman1998quantifying`. These derive the bias of
  sensitivity/specificity under discrepant resolution and show it favours the new test. Alonzo & Pepe 1999
  `alonzo1999using` and Hawkins et al. 2001 `hawkins2001some` give composite or resolved reference standards.
  Reitsma et al. 2009 `reitsma2009review` survey the solutions. Begg & Greenes 1983 `begg1983assessment` cover the
  related verification bias. The ML paper's bias result in (i) is the multi-model, ranking-level version of this
  literature. Its specific additions over this literature: rankings of *many* models, the role of each model's
  correlation with F, and the joint-error term.
- **Vasudevan et al. 2022 (NeurIPS) `vasudevan2022when`** (full text read): reviews only the "mistakes" (prediction
  not in the label set) of a few top ImageNet models. That is discrepant resolution, and 44% of one model's
  initial "mistakes" turn out to be correct. They note that the models used to select ImageNet-Major "will
  comparatively perform poorly" on it, then evaluate held-out models (Sec. 5). This is a selection-bias remark
  about a *subset*, not a bias analysis of accuracy on the full set. Small overlap.
- **Beyer et al. 2020 `beyer2020are`, Shankar et al. 2020 `shankar2020evaluating`:** re-annotate all ImageNet val images
  (census), with model-generated candidate label sets (Beyer: 6 of 19 models chosen for 97% label recall). This
  is a different bias, proposal coverage, and not the discrepant design. Little overlap.
- **Ye et al. 2025 (arXiv 2512.19691, MedCalc-Bench) `ye2026scalable`** (full text, App.): an LLM pipeline flags 286 of 1047
  test items. Physicians adjudicate 50 sampled *flagged* items. For the 761 unflagged items the paper assumes the
  worst case for the new labels (original label perfect). That is a stratified estimate with a one-sided bound
  on the unadjudicated stratum, close in spirit to (ii), but ad hoc, with no concordant sample.
- **Sylvestre 2026 (BioNLP, SciFact audit)** (full text in `txt/sylvestre2026.txt`): an LLM screen flags 57 of 209 dev
  claims. The author then *exhaustively* reviews the 152 non-flagged claims and finds 3 of the 11 total errors
  there. That is direct evidence of joint errors, about 27% of errors missed by the flagger. Useful as an empirical anchor.
  Not added to the bib (handled by the case-study list). Add it if cited.
- **Brunello et al. 2026 (FOLIO/MALLS, arXiv 2606.02837)** (abstract only): an LLM framework directs reviewers to error-prone
  items and reaches 90% dataset accuracy after reviewing under 20% of items. It frames prioritisation as a
  *cleaning-efficiency* question, not an unbiased-evaluation question.
- **Lam & Stork 2003 (IJCAI) `lam2003evaluating`** (TL;DR only, no abstract in S2): noisy test labels; gives the
  number of noisy test samples equivalent to one clean sample, plus upper/lower bounds on true error when labeler
  and classifier are dependent. This is the closest ML precedent for (ii): check the bounds before claiming sharpness is new.
- **Chavoshi et al. 2025/26 (Radiology: AI) `chavoshi2026impact`** (abstract): simulation plus analytic best/worst-case
  bounds on apparent sensitivity/specificity when LLM-extracted labels are the reference standard. Covers the
  noisy reference only, with no adjudication design.

### (2) Sampling unflagged items / two-phase designs in relabelling

- Northcutt 2021 Sec. 6 (above) is the only ML relabelling paper found that samples the non-flagged stratum.
  It does so descriptively, without an estimator.
- **Tenenbein 1970 (JASA) `tenenbein1970double`**: double sampling, with a fallible classifier on all units and the true
  classifier on a random subsample. This is the statistical ancestor of the proposed design. Neyman 1938
  `neyman1938contribution` gives two-phase sampling; Neyman 1934 `neyman1934two` gives optimal allocation.
- **IR statAP/infAP (2006)** (above): stratified sampling with HT weights over judgment pools.
- **Fogliato et al. 2024 (ECCV) `fogliato2024framework`** (abstract): stratification plus model-assisted estimators
  to cut the cost of estimating accuracy. It assumes clean labels on the sample and uses predicted model
  correctness for strata. Same toolkit, different problem.
- **Kossen et al. 2021 (ICML) `kossen2021active`; Sawade et al. 2010 (ICML) `sawade2010active`; Marsigli & de Haro 2026
  (arXiv 2601.22326, stratified importance sampling for monitoring; not in bib)**: label-efficient unbiased risk
  estimation with importance weights. Test labels are assumed correct.

### (3) PPI / stratified sampling under noisy *test labels*

- All PPI-for-evaluation papers found treat the human-labelled sample as gold and correct a noisy *proxy*:
  Boyeau et al. 2025 `boyeau2025autoeval`, Chatzi et al. 2024 `chatzi2024prediction`, Eyre & Madras 2025 `eyre2025regression`,
  Fisch et al. 2024. Chen et al. 2026 `chen2026efficient` compare PPI with Rogan-Gladen misclassification correction
  for noisy LLM judges using semiparametric efficiency theory. The closest theory is the efficiency comparison of
  proxy-corrected estimators, but there is no adjudication design.
- **No paper found that treats the benchmark's own original labels as the PPI proxy for re-annotation with strata
  defined by a flagger.** MMLU-Redux (Gema et al. 2025 `gema2025are`) samples uniformly at random (no flagger), so
  it supports an unbiased estimate but makes no PPI-style use of L.
- Mozer 2026 `mozer2026ppi`: PPI mean estimator = Cassel-Särndal-Wretman difference estimator, and PPI++ = GREG.
  Cite this to pre-empt "this is just PPI": it also says PPI could borrow optimal allocation from survey sampling,
  which is what (iii) does.

### (4) Benchmark lock-in / shared errors of cleaning models

- **Kim et al. 2025 (ICML) `kim2025correlated`** (abstract): across 350+ LLMs, when two models both err they agree
  60% of the time, and larger, more accurate models are *more* correlated. This is the empirical basis for
  large joint-error mass when F is a frontier LLM.
- **Kohli 2026 "Nine Judges, Two Effective Votes" (arXiv 2605.29800; not in bib)** (abstract): nine frontier judges ≈ 2
  effective independent votes. Supports the claim that ensembling flaggers does not remove joint errors.
- **Balasubramanian, Podkopaev & Kasiviswanathan 2026 `balasubramanian2026dependence`** (abstract): dependence-aware
  (Ising) aggregation of LLM judges, showing that conditional-independence methods (Dawid-Skene `dawid1979maximum`)
  can be confidently wrong. Concerns aggregation, not adjudication design. Overlap only on the dependence theme.
- **Rahmani et al. 2025 (CIKM, arXiv 2506.10301; not in bib)** (abstract): LLM-generated test collections bias absolute
  system scores, with smaller effects on relative rankings. An IR analogue of "labels built by a model favour that model".
- **Land & Bikel 2026 (arXiv 2605.30504; not in bib)** (abstract): IRT over 114 models surfaces mislabels and finds a reward
  model agreeing with mislabels 78% of the time. Shows that model agreement with wrong labels is detectable and
  model-specific.
- No paper found uses "benchmark lock-in" or argues that iterative model-assisted cleaning makes a benchmark
  converge on the cleaning models' blind spots. The framing appears novel, but present it as a consequence of (i),
  supported by the citations above.

## Searches run (S2 unless noted)

"evaluating classifiers with noisy test labels"; "label errors test set model ranking bias relabeling";
"benchmark label errors re-annotation model flagged"; "relabeling test set label errors confident learning
evaluation"; "noisy test labels accuracy estimation bias"; "LLM detect label errors benchmark"; "label error
audit stratified sampling"; "evaluation noisy ground truth unbiased estimator"; "discrepant analysis machine
learning evaluation"; "benchmark lock-in label cleaning"; "prediction-powered inference noisy gold labels";
"test label noise model evaluation correction"; "platinum benchmarks label errors"; "active testing
label-efficient model evaluation"; "verification bias machine learning evaluation"; "imperfect reference
standard AI evaluation"; "LLMs share errors correlated mistakes"; "model-assisted relabeling bias evaluation";
"estimating classifier accuracy without gold labels latent class"; "pooling bias test collection unjudged
documents"; "double sampling misclassified data Tenenbein"; S2 snippet searches for "discrepant analysis bias
label errors benchmark", "discrepant resolution machine learning model label disagreement adjudication bias",
"discrepant analysis resolve random sample of concordant specimens". Crossref was used for DOIs and arXiv API for metadata.

Not checked: Google Scholar (unavailable), and the full texts of Lam & Stork 2003 and the IR statAP/infAP
papers. Read those before writing the related-work paragraph on bounds and on HT sampling.
