"""Build design_coding.csv and lit.bib; verify every quote against extracted full text.

Usage: python3 -I build_coding.py <lit_dir>
Each quote is checked against the source text after normalisation (ligatures,
curly quotes, all whitespace and hyphens removed) and the PDF page is recorded.
The script exits non-zero if any quote is not found verbatim.
"""
import csv, re, sys, pathlib

LIT = pathlib.Path(sys.argv[1])
TXT = LIT / "txt_pages"
BIB = LIT / "web" / "bib"

LIG = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
       "’": "'", "‘": "'", "“": '"', "”": '"'}


def norm(s):
    for a, b in LIG.items():
        s = s.replace(a, b)
    return re.sub(r"[\s\-]+", "", s).lower()


def pages(path):
    raw = path.read_text(errors="ignore")
    parts = re.split(r"=== PAGE (\d+) ===", raw)
    if len(parts) == 1:
        return [(None, raw)]
    return [(int(parts[i]), parts[i + 1]) for i in range(1, len(parts), 2)]


def locate(src, quote):
    pg = pages(TXT / src) if (TXT / src).exists() else pages(LIT / "web" / src)
    q = norm(quote)
    for num, text in pg:
        if q in norm(text):
            return num
    # quote may straddle a page break
    for (n1, t1), (n2, t2) in zip(pg, pg[1:]):
        if q in norm(t1 + t2):
            return f"{n1}-{n2}"
    return False


# Each row: fields + list of (label, src, quote) evidence tuples.
ROWS = [
 dict(key="northcutt2021pervasive", title="Pervasive Label Errors in Test Sets Destabilize Machine Learning Benchmarks", year=2021,
      venue="NeurIPS 2021 Datasets and Benchmarks Track", benchmark="10 test sets incl. ImageNet val, CIFAR-10/100, MNIST, Caltech-256, QuickDraw, 20news, IMDB, Amazon Reviews, AudioSet",
      design="DR", flagger="Confident learning on out-of-sample predicted probabilities from one model per dataset (ImageNet: ResNet-50; CIFAR: VGG; IMDB: FastText, etc.)",
      flagger_also_evaluated="yes - ResNet-50 (the ImageNet flagging model) is among the 34 ImageNet models re-ranked on corrected labels",
      rereports="yes", confidence="high", src="2103.14749.txt",
      notes="CL-flagged items sent to 5 MTurk workers; unflagged items enter the 'benign set' with original labels. Section 6 adds an expert review of 1 random non-CL-flagged image per ImageNet class (CRS-like) but it is used only to re-estimate prevalence (~20% vs 6%), not to correct labels or re-rank models.",
      ev=[("design", "We validated the algorithmically identified label errors with a Mechanical Turk (MTurk) study. For two large datasets with a large number of errors (QuickDraw and Amazon Reviews), we checked a random sample; for the rest, we checked all identified errors."),
          ("design", "The subset of benign test examples, comprising data that CL did not flag as likely label errors and data that was flagged but for which human reviewers agreed that the original label should be kept."),
          ("rerank", "Benchmark ranking comparison of 34 pre-trained ImageNet models and 13 pre-trained CIFAR-10 models"),
          ("ack", "this analysis only presents a loose lower bound on the magnitude of these issues due to unaccounted for label errors in the non-CL-flagged data"),
          ("ack", "we can more accurately estimate that the ImageNet validation set contains closer to 20% label errors (up from the 6% reported in Table 1)")]),
 dict(key="nahum2025are", title="Are LLMs Better than Reported? Detecting Label Errors and Mitigating Their Effect on Model Performance", year=2025,
      venue="EMNLP 2025 (main); arXiv 2410.18889", benchmark="TRUE (MNBM, BEGIN, VitaminC, PAWS; 160 sampled items each); SummEval",
      design="DR", flagger="Ensemble (averaged over 4 prompts) of GPT-4, PaLM2, Mistral-7B, Llama-3-8B; item flagged when the ensemble prediction differs from the original label",
      flagger_also_evaluated="yes - GPT-4, PaLM2, Llama3 and Mistral appear in the re-ranked Table 4",
      rereports="yes", confidence="high", src="2410.18889.txt",
      notes="Experts (two authors) were blinded to original and LLM labels, but only disagreement items were sent to them. The task brief lists this as EMNLP 2024; Crossref/ACL Anthology give EMNLP 2025 (DOI 10.18653/v1/2025.emnlp-main.1360). The 'lower bound' in Table 2 is a Clopper-Pearson CI over the sample, not an acknowledgement of errors among agreed items.",
      ev=[("design", "All TRUE examples where the prediction differed from the original label, regardless of confidence, were annotated by human experts."),
          ("design", "After re-annotating all conflicted examples, we define the gold label as the original label, if the LLM prediction agrees with it, or the expert resolution, if there was a disagreement."),
          ("rerank", "The first observed discrepancy is the ranking of models. For example, DeBERTa-v3 has shifted from being the second-best to the second-worst.")]),
 dict(key="vasudevan2022when", title="When does dough become a bagel? Analyzing the remaining mistakes on ImageNet", year=2022,
      venue="NeurIPS 2022", benchmark="ImageNet multi-label validation subset (20k images of Shankar et al. 2020)",
      design="DR", flagger="ViT-3B (JFT-3B pretrained) and Greedy Soups; only their apparent mistakes were panel-reviewed",
      flagger_also_evaluated="yes - the same two models' multi-label accuracy is re-reported after review",
      rereports="yes", confidence="high", src="2205.04596.txt",
      notes="Builds on Shankar et al.'s labels, which were themselves restricted to model-proposed classes. The paper is explicit that items on which all models agree with a wrong ground truth are never reviewed.",
      ev=[("design", "The model made a total of 676 apparent mistakes, which we sought to manually review in detail."),
          ("rerank", "Nearly half of each model's mistakes were deemed correct under a careful, expert multi-label re-evaluation, halving the error rate."),
          ("ack", "we never review any validation image whose groundtruth might be wrong, if all models evaluated also make the incorrect groundtruth prediction.")]),
 dict(key="beyer2020are", title="Are we done with ImageNet?", year=2020,
      venue="arXiv preprint 2006.07159", benchmark="ImageNet (ILSVRC-2012) validation set -> ReaL labels",
      design="DR-multi", flagger="Union of label proposals from 6 models selected from 19 (VGG-16, Inception v3, ResNeXt-101 32x8d IG, BiT-M, BiT-L, CPC v2 fine-tuned)",
      flagger_also_evaluated="yes - BiT-L, BiT-M, ResNeXt-101 IG, Inception v3, CPC v2 are all scored with ReaL accuracy (Appendix C, Table 5)",
      rereports="yes", confidence="high", src="2006.07159.txt",
      notes="Images where all proposing models agree with the original label are not sent to raters. A 256-image expert gold set was used only to pick the proposer subset (97.1% proposal recall), not to check retained images.",
      ev=[("design", "In the event that all models agree with the original ImageNet label, we can safely retain the original without re-evaluation. This reduces the number of images to be annotated from 50 000 to 24 889."),
          ("flagger", "we find a subset of 6 models which generates label proposals that have a recall of 97.1% and a precision of 28.3%"),
          ("rerank", "Using the proposed ReaL accuracy, we now re-assess the progress of recently proposed classifiers on the ImageNet dataset.")]),
 dict(key="shankar2020evaluating", title="Evaluating Machine Accuracy on ImageNet", year=2020,
      venue="ICML 2020 (PMLR 119)", benchmark="ImageNet val (20,000-image subset) and ImageNetV2",
      design="OTHER", flagger="Union of top-1 predictions of 72 pre-trained ImageNet models (2012-2018) defines the candidate labels judged for each image",
      flagger_also_evaluated="yes - all 72 testbed models are scored with the resulting multi-label accuracy",
      rereports="yes", confidence="medium", src="shankar20c.txt",
      notes="OTHER = full item coverage but model-proposed label space: every image in the subset was reviewed, yet only classes some model predicted were judged. Human labelers' extra predictions were reviewed later to avoid penalising non-model labels. How the 20,000-image subset was chosen is not stated in the main text.",
      ev=[("design", "The three participants then categorized every unique prediction made by the 72 models on the 40,683 images (a total of 182,597 unique predictions) into correct or incorrect"),
          ("rerank", "In Figure 4, we plot each model's top-5 and top-1 accuracy versus its multi-label accuracy."),
          ("ack", "Since each image only contained reviewed labels for classes predicted by models, to ensure a fair multi-label accuracy, the human predictions for the 2,000 images had to be manually reviewed.")]),
 dict(key="gema2025are", title="Are We Done with MMLU?", year=2025,
      venue="NAACL 2025 (long); arXiv 2406.04127", benchmark="MMLU -> MMLU-Redux (5,700 questions, 100 per subject)",
      design="RANDOM", flagger="none (annotation independent of model outputs; LLMs studied afterwards only as error detectors)",
      flagger_also_evaluated="n/a", rereports="yes", confidence="high", src="2406.04127.txt",
      notes="Subject-stratified random sample, so the annotated sample is unbiased for prevalence; re-evaluation used HELM leaderboard predictions on the 5 most error-prone subjects.",
      ev=[("design", "We randomly subsampled 100 questions per MMLU subject to be presented to the annotators."),
          ("rerank", "in the Virology subset, Llama 3.1 Instruct Turbo (405B) ranked 16th when considering all Virology instances, but ranked first when only correct instances were used.")]),
 dict(key="vendrow2025do", title="Do Large Language Model Benchmarks Test Reliability?", year=2025,
      venue="arXiv preprint 2502.03461", benchmark="15 benchmark subsets (GSM8K, SVAMP, MMLU HS Math, SQuAD2.0, HotPotQA, DROP, TabFact, Winograd WSC, BIG-bench, VQA v2.0, ...) -> platinum benchmarks",
      design="DR-multi", flagger="Panel of frontier LLMs (Table 3: GPT-4o, o1 series, Claude 3.5, Gemini, Llama 3.1, DeepSeek-R1, Mistral, etc.); item inspected if any model errs",
      flagger_also_evaluated="yes - the panel is the set of models whose error counts are reported on the platinum benchmarks",
      rereports="yes", confidence="high", src="2502.03461.txt",
      notes="Applied to random subsets of each benchmark. Exception: VQA v2.0 subset fully re-labelled. Explicitly states that items on which all models match the given answer may still be bad.",
      ev=[("design", "we give each question to several frontier LLMs. We then manually inspect any example on which at least one LLM makes an error."),
          ("design", "we manually re-label all the VQA v2.0 queries we include in our revised subset rather than only inspecting ones for which some model failed."),
          ("rerank", "Table 4 confirms significant differences in the number of errors made by models on the original and cleaned benchmarks."),
          ("ack", "Note that the number of errors that we identify for each benchmark is only a lower bound, as our approach could miss any bad question for which all LLM solutions happen to agree with the benchmark solution."),
          ("ack", "there may still be poorly written questions in our platinum benchmarks among those we did not revise, where, despite error or ambiguity, all models agreed with the stated ground truth.")]),
 dict(key="wang2024mmlupro", title="MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark", year=2024,
      venue="NeurIPS 2024 Datasets and Benchmarks Track; arXiv 2406.01574", benchmark="MMLU-Pro (built from MMLU, STEM Website, TheoremQA, SciBench)",
      design="OTHER", flagger="Phase 2 only: Gemini-1.5-Pro re-evaluates all options to flag false-negative distractors",
      flagger_also_evaluated="yes - Gemini-1.5-Pro is in the MMLU-Pro results table",
      rereports="no", confidence="medium", src="2406.01574.txt",
      notes="Two phases: (1) FULL expert check of every answer key; (2) DR on distractor validity (only Gemini-flagged options reviewed). This builds a new benchmark, so there is no before/after accuracy on the same items.",
      ev=[("design", "Phase 1: Verification of Correctness and Appropriateness involves experts verifying the accuracy of each answer"),
          ("design", "Phase 2: Ensuring Distractor Validity involves the Gemini-1.5-Pro model re-evaluating all answer options to identify potential false negatives."),
          ("design", "Subsequently, human experts rigorously review these identified options to ensure that actual distractors are indeed incorrect and distinctly different from the correct answer.")]),
 dict(key="chong2022detecting", title="Detecting Label Errors by Using Pre-Trained Language Models", year=2022,
      venue="EMNLP 2022", benchmark="IMDB, Amazon Reviews (real errors); TweetNLP, SNLI (synthetic human-originated noise)",
      design="DR", flagger="Loss ranking from fine-tuned pre-trained LMs (and ensembles) plus the confident-learning baseline; top-k hypothesised items crowd-verified",
      flagger_also_evaluated="partial - detection paper; the end-to-end cleaning experiment (Table 4) re-scores models but uses synthetic-noise splits",
      rereports="partial", confidence="medium", src="2205.12702.txt",
      notes="Real-error precision/recall is computed only over the top-ranked items up to the number CL hypothesises; recall assumes 5% prevalence. Explicitly warns that model-based cleaning can bias benchmarks toward the cleaning model's family.",
      ev=[("design", "Because IMDB and Amazon are too expensive to fully crowd verify, we cannot calculate precision and recall at the 25,000th item for each method, for each dataset, as it would require every data point to be relabeled on MTurk. Instead, we use the CL framework of predicting a fixed number of items."),
          ("ack", "Determining the true recall of a label error detection method on a real datasets is generally infeasible due to its high cost; this requires a complete re-evaluation so as to identify every label error within the dataset."),
          ("ack", "we caution against using this method to clean data intended for use in comparing performance across model families and variants: the cleaning process may bias any such benchmarks toward the models most similar to the model used to clean the data.")]),
 dict(key="klie2023annotation", title="Annotation Error Detection: Analyzing the Past and Present for a More Coherent Future", year=2023,
      venue="Computational Linguistics 49(1)", benchmark="Survey and re-implementation of 18 AED methods on corpora with known errors",
      design="OTHER", flagger="n/a (survey; no benchmark relabelled)", flagger_also_evaluated="n/a", rereports="no",
      confidence="high", src="2206.02280.txt",
      notes="Not a relabelling study. Included because it states the methodological point: AED recall needs a careful expert annotation of a subset. Excluded from the design counts of relabelling studies.",
      ev=[("ack", "Recall relies on knowing the exact number of correctly and incorrectly annotated instances. While this information may be available when developing and evaluating AED methods, it is generally not available when actually applying AED to clean real data. One solution to computing recall then is to have experts carefully annotate a subset of the data and then use it to estimate recall overall.")]),
 dict(key="wang2019crossweigh", title="CrossWeigh: Training Named Entity Tagger from Imperfect Annotations", year=2019,
      venue="EMNLP-IJCNLP 2019", benchmark="CoNLL-2003 English NER test set",
      design="FULL", flagger="none for the test-set correction (CrossWeigh is used on the training set only)",
      flagger_also_evaluated="n/a", rereports="yes", confidence="high", src="1909.01441.txt",
      notes="Every test sentence checked by two of five trained annotators, then a final verification round on sentences where the original and the two annotators did not all agree.",
      ev=[("design", "For the whole test set, we randomly split the test sentences between each pair combination of 5 annotators. In this way, each sentence in the test set is checked by exactly two annotators."),
          ("rerank", "Our re-evaluation of popular models on this corrected test set leads to more accurate assessments")]),
 dict(key="reiss2020identifying", title="Identifying Incorrect Labels in the CoNLL-2003 Corpus", year=2020,
      venue="CoNLL 2020", benchmark="CoNLL-2003 English NER (all folds)",
      design="DR", flagger="Ensembles: the 16 original CoNLL-2003 shared-task entries plus two 17-model BERT random-projection ensembles, tuned to F1 0.6-0.89; flag when models agree strongly on a label absent from the corpus",
      flagger_also_evaluated="yes - the 16 competition entries are re-scored on the corrected test fold",
      rereports="yes", confidence="high", src="reiss2020.txt",
      notes="470 extra errors found 'in close proximity' to flagged labels (opportunistic). Reports no ranking change. Contrasts its count with Wang et al. 2019's full review of the test fold.",
      ev=[("design", "We used a semi-supervised approach to flag potentially-incorrect labels in the corpus, then manually reviewed the labels thus flagged."),
          ("flagger", "we focused on cases where the models agreed strongly on a particular label, but that label does not appear in the corpus."),
          ("rerank", "Then we re-evaluated the original entries in the competition, plus selected NER models from recent work, over the corrected corpus."),
          ("rerank", "Surprisingly, we did not observe any change in the relative ranking of the models.")]),
 dict(key="alt2020tacred", title="TACRED Revisited: A Thorough Evaluation of the TACRED Relation Extraction Task", year=2020,
      venue="ACL 2020", benchmark="TACRED dev and test",
      design="CRS", flagger="49 relation-extraction models; Challenging = misclassified by >= half the models",
      flagger_also_evaluated="yes - all 49 models re-scored on the revised test split",
      rereports="yes", confidence="high", src="2004.14855.txt",
      notes="Closest ML analogue to a composite reference standard, but partial: the control stratum is a small random sample (<=20 per relation) only from items correctly classified by >=39 models (items with 25-38 correct are neither flagged nor sampled), and its 8.9% revision rate is not propagated to unsampled items. Re-TACRED (Stoica et al. 2021) criticises the sample as 'small and biased'.",
      ev=[("design", "(a) Challenging – all examples that were misclassified by at least half of the models, and (b) Control – a control group of (up to) 20 random examples per relation type, including no relation, from the set of examples classified correctly by at least 39 models."),
          ("rerank", "we evaluated all 49 models on the revised test split. The average model F1 score rises to 70.1%"),
          ("ack", "As expected, the revision rate in the Control groups is much lower, at 8.9% for Test and 8.1% for Dev."),
          ("ack", "Even though our selection strategy was biased towards examples challenging for models")]),
 dict(key="stoica2021retacred", title="Re-TACRED: Addressing Shortcomings of the TACRED Dataset", year=2021,
      venue="AAAI 2021", benchmark="TACRED (all splits)",
      design="FULL", flagger="none", flagger_also_evaluated="n/a", rereports="yes", confidence="high", src="2104.08398.txt",
      notes="Motivated explicitly by the biased partial sample of Alt et al. 2020.",
      ev=[("design", "we aim to address these shortcomings by performing a re-annotation of the entire TACRED dataset."),
          ("ack", "the broader impact of their work is restricted by both their small and biased sample set, and the fact that their analysis was performed over a predominately uncorrected version of the TACRED dataset."),
          ("rerank", "evaluating several models on our revised dataset yields an average f1-score improvement of 14.3%")]),
 dict(key="rucker2023cleanconll", title="CleanCoNLL: A Nearly Noise-Free Named Entity Recognition Dataset", year=2023,
      venue="EMNLP 2023", benchmark="CoNLL-2003 English NER (all splits, starting from Reiss et al. version)",
      design="OTHER", flagger="Phase 2 cross-checking: FLERT-style BIOES tagger (round 1), entity classifier (round 2), boundary tagger (round 3), each trained on held-out surface-form batches",
      flagger_also_evaluated="yes - FLERT is one of the four models re-evaluated",
      rereports="yes", confidence="high", src="2310.16225.txt",
      notes="Hybrid: Phase 1 relabels automatically from independent expert AIDA entity links (manual typing for 1,432 unmapped links); Phase 2 is DR (only model-label disagreements inspected, 3 rounds). The quality evaluation sample is drawn only from sentences where corpus versions disagree, so it is also discordant-only.",
      ev=[("design", "All predictions that differ from the annotated NER labels are flagged as potential errors. Manual error resolution. All annotations flagged as potential errors were then manually inspected by annotators."),
          ("design", "We gather all sentences in which there is at least one difference in annotation between at least two of the corpora. From these, we sampled 100 sentences for manual evaluation."),
          ("rerank", "Table 4 reports averaged F1-scores and standard deviation over three random seeds for each experimental setting."),
          ("ack", "It is possible, that specific instances are wrongly labeled (e.g. because of a wrong or missing Wikipedia label leading to an incorrect NER label that fell through our cross-checking approach).")]),
 dict(key="zhai2026hleverified", title="HLE-Verified: A Systematic Verification and Structured Revision of Humanity's Last Exam", year=2026,
      venue="arXiv preprint 2602.13964", benchmark="Humanity's Last Exam (2,500 items)",
      design="FULL", flagger="Model pass@8 replication checks by unnamed 'multiple frontier multimodal solvers' used as auxiliary evidence only",
      flagger_also_evaluated="unknown - solver identities not named; 8 frontier models evaluated afterwards",
      rereports="yes", confidence="high", src="2602.13964.txt",
      notes="Every item gets expert screening; models are diagnostic signals only. 668 verified, 1,143 revised, 689 uncertain.",
      ev=[("design", "In Stage I, each item is subjected to binary validation on the problem and final answer dimensions (with rationale used as an auxiliary consistency signal), combining domain-expert review and model-based cross-checks."),
          ("ack", "Conversely, high solver agreement is not regarded as proof of correctness."),
          ("rerank", "We benchmark eight state-of-the-art LLMs on HLE vs. HLE-Verified, demonstrating that verification materially alters measured performance")]),
 dict(key="ye2026scalable", title="Scalable Stewardship of an LLM-Assisted Clinical Benchmark with Physician Oversight", year=2026,
      venue="arXiv preprint 2512.19691 (arXiv BibTeX year 2026)", benchmark="MedCalc-Bench test set (1,047)",
      design="OTHER", flagger="Gemini 2.5 Pro auditor/recomputation agent (5 runs, majority)",
      flagger_also_evaluated="no - evaluated models are GPT 5.2, Opus 4.6, Grok 4.1, Gemini 3.1 (same family as flagger, different model)",
      rereports="yes", confidence="high", src="2512.19691.txt",
      notes="LLM recomputes every label; physicians adjudicate only 50 instances sampled from the top of the original-vs-recomputed disagreement ranking. Models are then re-scored on 887 items against the LLM labels, justified by physician results on the discordant 50 only. Physician review of concordant items: none.",
      ev=[("design", "Automated triage then identifies instances where the recomputed and original labels diverge most, concentrating scarce physician time on the highest-impact disagreements; physicians independently recompute scores for these contentious cases, providing ground-truth adjudication."),
          ("rerank", "Figure 4b reports accuracy on the full 887 instances for which our pipeline produced high-confidence recomputed labels."),
          ("ack", "the agents are not fully independent: they share the same underlying model and prompting strategy, so systematic biases (e.g., a consistent misinterpretation of a scoring rule) would not be caught by majority voting.")]),
 dict(key="land2026auditing", title="Auditing LLM Benchmarks with Item Response Theory", year=2026,
      venue="EMNLP 2026 (accepted, per arXiv comment); arXiv 2605.30504", benchmark="20,986 items over 32 subsets (RewardBench 2, RM-Bench, JudgeBench, MMLU-Pro, GSM-MC, GPQA Diamond, ...)",
      design="OTHER", flagger="4PL IRT forced-ceiling indicator fit on responses from 114 models",
      flagger_also_evaluated="n/a - detection paper; no corrected leaderboard",
      rereports="no", confidence="medium", src="2605.30504.txt",
      notes="Weak reference labels for every item come from GPT-5.4 aggregating strong-model judgments; human review only of items above the flag threshold. Items on which every model gives the same answer (8%) are dropped before fitting. GPQA items assumed correct.",
      ev=[("design", "For visualization and evaluation, not for fitting the IRT indicator, we assign each item a weak reference label"),
          ("design", "We reviewed every item above this threshold by hand"),
          ("design", "The constant-item filter drops 1,685 items (8%) on which every generative model gives the same answer."),
          ("ack", "Removing it from the panel leaves the ranking essentially unchanged (Appendix B.4), though this does not rule out biases shared across the panel.")]),
 dict(key="ansari2026how", title="How Good Are Frontier Models at Physics? Expert Re-Grading Reveals Broken Evaluations and Near-Saturation of Leading Benchmarks", year=2026,
      venue="arXiv preprint 2609.13009", benchmark="HLE-Physics, PHYBench, PRISM-Physics, UGPhysics (DR); CMT-Benchmark, CritPt (full review)",
      design="DR", flagger="GPT-5.6-Sol (High); a question is audited only if all its attempts were graded incorrect",
      flagger_also_evaluated="yes - GPT-5.6-Sol is one of the three models re-scored",
      rereports="yes", confidence="high", src="2609.13009.txt",
      notes="Mixed: DR for 4 benchmarks (250 rejected of 502 audited-run questions; 252 accepted never reviewed); FULL review for CMT-Benchmark (50) and a 56-challenge CritPt subset. The full-review benchmarks found errors in 30/50 and 21/56. No statement found about errors among accepted items.",
      ev=[("design", "we restrict the audit to questions for which all of GPT-5.6-Sol High's attempts were evaluated as incorrect."),
          ("design", "All questions in their audit sets are audited, regardless of the model's pre-audit response."),
          ("rerank", "Table 1 summarizes the pre-audit and validated/repaired results.")]),
 dict(key="brunello2026fixing", title="Fixing FOLIO and MALLS: Verified Annotations and an LLM-assisted Framework to Focus Human Relabeling", year=2026,
      venue="EMNLP 2026 (accepted, per arXiv comment); arXiv 2606.02837", benchmark="FOLIO validation split (275) and first 100 MALLS test instances",
      design="FULL", flagger="none for the audit; the proposed triage framework uses an LLM judge (Gemma) and is evaluated against the full audit",
      flagger_also_evaluated="partial - Gemma 4 31B-it is evaluated; it is the triage judge only in the proposed framework, not in the audit",
      rereports="yes", confidence="high", src="2606.02837.txt",
      notes="Full audit gives a reference standard against which the DR-style triage can be scored: 90% dataset accuracy after reviewing <20% of items, i.e., residual errors remain among unreviewed items.",
      ev=[("design", "We conduct a systematic human audit of the FOLIO validation split and a subset of MALLS test instances"),
          ("rerank", "Models are evaluated on both the original and curated datasets"),
          ("ack", "it is possible to achieve 90% dataset accuracy after reviewing fewer than 20% of instances, compared to over 76% required by unguided review.")]),
 dict(key="sylvestre2026gold", title="Gold Label Errors in the SciFact Benchmark: An LLM-Assisted Annotation Audit", year=2026,
      venue="BioNLP Workshop @ ACL 2026", benchmark="SciFact dev (209 claim-document pairs)",
      design="FULL", flagger="GPT-5.4-mini screening (flagged items additionally discussed with GPT-5.4)",
      flagger_also_evaluated="yes - GPT-5.4-mini is re-scored (+3.8 F1)",
      rereports="yes", confidence="high", src="sylvestre2026.txt",
      notes="Flag-assisted but every unflagged item was also reviewed; 3 of the 11 errors were among unflagged items (auditor recall 73%). Single annotator. Direct evidence that DR would have missed 27% of errors.",
      ev=[("design", "The 152 non-flagged claims were exhaustively reviewed by the same annotator without the chat-mode loop, since the screen had already implicitly endorsed the gold label."),
          ("ack", "Of the 152 non-flagged claims, exhaustive review identified 3 additional errors that the automated screening missed entirely (rating them AGREE), bringing the total to 11."),
          ("rerank", "We ran two zero-shot configurations on all 209 dev claims (oracle setting) and computed macro F1 before and after correcting the 11 errors.")]),
 dict(key="aleithan2024swebench", title="SWE-Bench+: Enhanced Coding Benchmark for LLMs", year=2024,
      venue="arXiv preprint 2410.06992", benchmark="SWE-bench Full (also Lite, Verified)",
      design="OTHER", flagger="SWE-Agent + GPT-4; only instances it 'resolved' (all tests pass) were audited",
      flagger_also_evaluated="yes - SWE-Agent+GPT-4 resolution rate is re-reported",
      rereports="yes", confidence="high", src="2410.06992.txt",
      notes="Mirror image of DR: audits only model-'success' (model agrees with the test oracle) items, looking for false passes; failed instances are not reviewed, so false failures (e.g., over-strict tests) are not counted.",
      ev=[("design", "For the model generated patches, we focused on instances where the generated patches resolved the issue and passed all associated tests."),
          ("rerank", "the correct resolution rate of SWE-Agent+GPT-4 dropped to 3.97% from 12.47%.")]),
 dict(key="chowdhury2024introducing", title="Introducing SWE-bench Verified", year=2024,
      venue="OpenAI blog post (13 Aug 2024)", benchmark="SWE-bench test set -> SWE-bench Verified (500)",
      design="RANDOM", flagger="none (each sample screened by 3 of 93 developers)",
      flagger_also_evaluated="n/a", rereports="yes", confidence="high", src="swebv_archive.txt",
      notes="Read from the Wayback Machine copy (openai.com returns a bot challenge). Not peer reviewed; BibTeX built from page metadata (no DOI exists).",
      ev=[("design", "We annotated 1,699 random samples from the SWE-bench test set to produce SWE-bench Verified."),
          ("rerank", "We found that GPT-4o's performance on the best-performing scaffold reaches 33.2% on SWE-bench Verified, more than doubling its score of 16% on the original SWE-bench.")]),
]


def main():
    bad = 0
    out_rows = []
    for r in ROWS:
        loc = {}
        for label, q in r["ev"]:
            p = locate(r["src"], q)
            if p is False:
                print(f"NOT FOUND [{r['key']}] {q[:80]}")
                bad += 1
            loc.setdefault(label, []).append((q, p))
        fmt = lambda lab: " || ".join((f'"{q}" (p.{p})' if p is not None else f'"{q}"') for q, p in loc.get(lab, []))
        design_qs = loc.get("design", []) + (loc.get("flagger", []) if not loc.get("design") else [])
        ack = fmt("ack")
        out_rows.append(dict(
            key=r["key"], title=r["title"], year=r["year"], venue=r["venue"], benchmark=r["benchmark"],
            design=r["design"], flagger=r["flagger"], flagger_also_evaluated=r["flagger_also_evaluated"],
            rereports_accuracy_or_ranking=r["rereports"] + (f": {fmt('rerank')}" if loc.get("rerank") else ""),
            acknowledges_unflagged_errors=("yes: " + ack) if ack else "no explicit statement found",
            quote_design=" || ".join(q for q, _ in (design_qs or loc.get("ack", []))),
            quote_location=(f"{r['src']} PDF page index " + ", ".join(str(p) for _, p in (design_qs or loc.get("ack", [])))) if r["src"].endswith(".txt") and not r["src"].startswith("swebv") else "openai.com/index/introducing-swe-bench-verified (Wayback copy), section 'Our Approach'",
            confidence=r["confidence"],
            notes=r["notes"] + (f" Flagger quote: {fmt('flagger')}" if loc.get("flagger") and loc.get("design") else ""),
        ))
    # Rows whose 'ack' quotes are critiques or survey points rather than self-acknowledgement
    for row in out_rows:
        if row["key"] == "stoica2021retacred":
            row["acknowledges_unflagged_errors"] = "n/a (FULL design); critiques Alt et al.: " + row["acknowledges_unflagged_errors"][5:]
        if row["key"] == "klie2023annotation":
            row["acknowledges_unflagged_errors"] = "n/a (survey); methodological point: " + row["acknowledges_unflagged_errors"][5:]
        if row["key"] == "sylvestre2026gold":
            row["acknowledges_unflagged_errors"] = "yes (measured): " + row["acknowledges_unflagged_errors"][5:]
        if row["key"] == "brunello2026fixing":
            row["acknowledges_unflagged_errors"] = "yes (measured for the triage framework): " + row["acknowledges_unflagged_errors"][5:]
    cols = ["key", "title", "year", "venue", "benchmark", "design", "flagger", "flagger_also_evaluated",
            "rereports_accuracy_or_ranking", "acknowledges_unflagged_errors", "quote_design", "quote_location",
            "confidence", "notes"]
    with open(LIT / "design_coding.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, quoting=csv.QUOTE_ALL)
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {len(out_rows)} rows; quotes not found: {bad}")
    return bad


if __name__ == "__main__":
    sys.exit(1 if main() else 0)
