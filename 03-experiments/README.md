# Experimental Results and Index

This directory organizes the PaperEval experiments in the order used by the current experiment manuscript. Each entry summarizes the question, sample, principal observations, and a link to the detailed record. Samples differ across studies and may overlap; do not add their sample sizes to infer a project-wide number of unique papers.

## Manuscript sequence

1. [Corpus and journal/source score profile](#1-corpus-and-journalsource-score-profile)
2. [Exploratory comparison across models](#2-exploratory-comparison-across-models)
3. [Reviewer structure comparison](#3-reviewer-structure-comparison)
4. [Harness validation and fault injection](#4-harness-validation-and-fault-injection)
5. [Adversarial instruction injection](#5-adversarial-instruction-injection)
6. [Repeated-run stability](#6-repeated-run-stability)
7. [Experiment summary and paper-level outcomes](#7-experiment-summary-and-paper-level-outcomes)

## 1. Corpus and journal/source score profile

The corpus audit began with **149 source PDFs**. Exact SHA-256 deduplication yielded **146 unique papers**, of which **144** had complete reports; two were blocked before scoring by PDF parsing. The 144 complete reports had a mean total score of **76.46** (SD 7.64), a median of **76.19**, and a range of **49.63–95.78**. These are descriptive scores for the audited corpus and configuration, not expert-validity estimates.

The manuscript also summarizes scores across six dimensions (A–F) and reports journal/source composition. Per-paper title-free outcomes are available in the [audit table](../04-reproducibility/original-149-v16-results.csv); the corpus counts and aggregate score summary are documented in the [audit record](../04-reproducibility/original-149-v16-audit.md). A separate journal-gradient study and its reported association with journal-tier ordering are summarized in [historical experiment evidence](experiment-evidence.md); it uses its own 12- and 15-paper samples.

| Dimension | Mean contribution | SD |
|---|---:|---:|
| A. Research question and positioning | 12.61 | 1.71 |
| B. Theoretical contribution and novelty | 14.69 | 3.69 |
| C. Theoretical rigor and hypothesis logic | 15.09 | 1.58 |
| D. Research design | 19.67 | 2.72 |
| E. Results and evidence | 11.05 | 1.10 |
| F. Discussion, boundaries, and integration | 3.21 | 0.77 |

Dimension maxima differ, so these means describe the observed score profile and should not be read as a ranking of dimension importance. Journal/source groups are also uneven, with most containing only one to four papers; their score summaries are descriptive rather than journal quality rankings.

## 2. Exploratory comparison across models

The current manuscript compares five models on three shared papers, with two recorded runs per model and paper. Model means range from **77.91 to 89.99**; average absolute differences between repeated runs range from **0.61 to 3.83** points. The runs re-extracted evidence and repeated external retrieval, so differences combine model and evidence variation. The findings are descriptive and do not establish model accuracy or general superiority.

See the [five-model summary and earlier two-model archive](cross-model-v16.md). The earlier archived subset contains 12 valid runs from two models on the same three-paper sample. The manuscript also records partial results for three additional models; these are distinguished from the complete five-model comparison in the detailed summary.

## 3. Reviewer structure comparison

Four evaluation structures were compared on the same **five papers**: full PaperEval, one unified Reviewer, six rule-level Reviewers without Harness processing and formal two-pool aggregation, and direct model scoring. Mean scores were **79.50**, **74.60**, **57.93**, and **74.00**, respectively. The conditions use different judgment and aggregation procedures, so score differences are descriptive and are not comparisons against expert accuracy.

See the [four-condition comparison and module ablations](module-ablation.md). The file also records separate one-paper module ablations and a three-paper paired comparison of evidence-gate outputs.

## 4. Harness validation and fault injection

The deterministic Harness checks were evaluated with **180 injected faulty outputs** spanning nine machine-checkable fault types and **40 clean controls**. It detected all 180 injected faults, automatically blocked or transformed 160, and routed 20 reason–judgment conflicts for review. All 40 clean controls passed without a false positive. The test measures behavior on predefined machine-checkable faults; it does not estimate error rates on naturally occurring paper assessments.

See [Harness fault-injection results](robustness-experiments.md#deterministic-harness-fault-injection) and the [technical description of validation and aggregation](../02-system/technical-appendix.md#7-harness-validation-gating-and-aggregation).

## 5. Adversarial instruction injection

The matched prompt comparison used evidence packs from **three papers**, six Reviewer dimensions, and three attack types: forced high rating, forced low rating, and fabricated evidence ID. Each prompt condition included 108 paired attack cases. Adding an evidence-boundary instruction reduced target-reaching attacks from **4/108 (3.7%) to 0/108**, rating changes from 14/108 to 7/108, and mean absolute ordinal shift from **0.250 to 0.074**. The test found no fabricated IDs in either condition's raw outputs, and it did not evaluate full-paper scores or expert accuracy.

See the [paired prompt-injection experiment](robustness-experiments.md#evidence-level-prompt-injection-comparison), which also distinguishes the matched comparison from separate production-prompt and synthetic-pilot runs.

## 6. Repeated-run stability

The current manuscript reports **20 papers scored independently twice**, producing 40 complete reports. Mean scores were 72.940 and 71.418 across the two runs. The paired mean absolute score difference was **2.382** points, the maximum difference was **8.388**, and **17/20 papers (85.0%)** retained the same quality grade. This measures repeatability for the recorded sample, not agreement with experts.

See [two-run repeatability results](robustness-experiments.md#two-run-repeatability-in-the-current-experiment-manuscript). A separate historical reliability diagnostic covers 57 archived run records across 20 papers and is reported separately in the [experiment catalog](experiment-catalog.md#reliability-and-uncertainty).

## 7. Experiment summary and paper-level outcomes

The [experiment catalog](experiment-catalog.md) maps the complete set of study families and defines each sample separately. The [reproducibility package](../04-reproducibility/README.md) links the corpus audit, title-free outcomes, code, and validation modes. The manuscript's paper-level A–F score appendix corresponds to the [title-free outcome table](../04-reproducibility/original-149-v16-results.csv), which omits paper titles and source text.

## Additional archived studies

The repository also records studies that are not separate numbered sections in the current manuscript:

- [Historical scoring and batch evidence](experiment-evidence.md): a 10-paper API batch, two journal-gradient samples (27 papers combined), three-paper evidence-budget and threshold studies, and other archived scoring comparisons.
- [Retrieval evaluation](retrieval-evaluation.md): seven target papers, seven claims, and 366 citation-linked candidate mappings. Candidate relevance was machine-prelabelled from citation links; a claim-count mismatch remains in the archived aggregate, so no overall retrieval-performance estimate is presented here.
- [Experiment catalog](experiment-catalog.md): an index of corpus audit, repeatability, model comparison, robustness, fault injection, ablations, and retrieval records, including the 17-paper trace database.

These supplemental samples overlap with one another and with other study samples. Their counts are not additive.
