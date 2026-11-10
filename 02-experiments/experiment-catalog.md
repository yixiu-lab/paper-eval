# PaperEval Experiment Catalog

## Corpus size and experimental scope

The corpus contains completed score records for **200 papers**. Their totals and dimension scores are aggregated under the revised integer-weight profile from archived rule-level judgments; the model was not rerun for this recalculation. See the paper-level outcome table for scores and source metadata.

The research archive contains trace records, run-level diagnostics, and claim-level retrieval slices. This catalog summarizes the main study families and links to the underlying records.

## Study families

| Study family | Sample and design | Current record | Public package |
|---|---|---|---|
| Paper assessment and score profile | 200 papers scored | Aggregate score and six-dimension summaries | [Evaluation results](../03-data/README.md#evaluation-results) |
| Cross-model exploratory comparison | Three shared papers × five models × two recorded runs (30 scores) | Model means range from 77.91 to 89.99; evidence was not fixed between runs, so results are descriptive | [Cross-model results](cross-model-results.md) |
| Reviewer structure comparison | Five papers, four evaluation structures | Full PaperEval, unified Reviewer, Reviewer-only, and direct scoring conditions are summarized descriptively | [PaperEval Module Ablation Records](module-ablation.md) |
| Harness fault injection | 180 injected faulty outputs and 40 clean controls across nine fault types | Deterministic checks detected all injected faults; 88.9% were automatically contained and 11.1% routed for review | [Robustness and Harness Experiments](robustness-experiments.md) |
| Prompt-injection robustness | Three papers, six reviewer dimensions, three attack types, two repeats, paired across prompt conditions | Evidence-boundary prompt comparison reduced target-hit and rating-shift counts in this pilot | [Robustness and Harness Experiments](robustness-experiments.md) |
| Repeatability and uncertainty | Independent repeated runs and archived run-level diagnostics | Current manuscript: paired mean absolute score difference 2.382, maximum 8.388, and grade agreement 85.0%; historical diagnostics are summarized below | [Two-run repeatability in the current experiment manuscript](robustness-experiments.md#two-run-repeatability-in-the-current-experiment-manuscript) |
| Journal-gradient comparison | Journal/source analyses across evaluation records | Kendall's tau = +0.500 against journal-tier ordering in the archived analysis | [Historical Experiment Evidence and Results](experiment-evidence.md) |
| Scoring-mechanism studies | Mechanism comparisons on three-paper samples | Studies examined judgment scales, two-pool scoring, evidence budgets, thresholds, and weight allocation | [Historical Experiment Evidence and Results](experiment-evidence.md) |
| External-literature retrieval | Archived query, citation, semantic, and multi-route analyses | Candidate relevance labels are derived from citation links; a claim-count mismatch in archived reports is being reconciled before quantitative conclusions are released | [External-Literature Retrieval Evaluation](retrieval-evaluation.md) |

## Reliability and uncertainty

The offline reliability analysis compares archived runs at the paper, evidence, and judgment levels. For one illustrative paper with comparable runs, the total-score mean was 69.78, standard deviation 2.86, and range 6.36. The analysis reported evidence overlap of 0.6197, judgment agreement of 0.9314, batch-ranking Spearman correlation of 0.6491, and Top-5 overlap of 0.25. The diagnostic separated run variance, evidence variance, judgment variance, and calibration uncertainty.

These diagnostics summarize archived runs only and do not form a fixed-evidence, controlled repeated-run experiment.

## Model comparison

The completed comparison reports the five models with two recorded runs on each of three shared papers. [Cross-model results](cross-model-results.md) show per-paper runs, means, and run-to-run differences. The five-model comparison reports the archived results of that focused study. The 200-paper corpus results use the revised integer-weight profile described in [Evaluation results](../03-data/README.md#evaluation-results).

## Data release boundary

This repository publishes study designs, aggregate outcomes, and title-free scores. [Data and Research Materials](../03-data/README.md) includes 51 unchanged PDFs whose source copies state a Creative Commons license. The unified paper index retains metadata and scores for all 200 papers; PDFs are not distributed when the available copy does not establish redistribution permission.

## Executable analysis material

The experiment section includes [the deterministic Harness fault-injection script](code/run_harness_fault_injection.py) and [the paper-outcome summary script](code/summarize_paper_outcomes.py). The first exercises the released validator and evidence gate on synthetic cases; the second recomputes descriptive statistics from the public title-free score table.
