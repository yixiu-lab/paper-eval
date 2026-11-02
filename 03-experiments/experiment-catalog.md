# PaperEval Experiment Catalog

## Corpus size and experimental scope

The **149-PDF** figure refers to one Corpus audit: 149 source files, 146 unique PDFs after exact deduplication, 144 complete reports, and two parser-blocked papers. It is not the count of every paper used across the project. Other studies reuse overlapping samples, so their sample sizes should not be added together as a project-wide unique-paper total.

The research archive contains a trace database covering 17 papers and 72 evaluation runs, with 2,342 agent judgments, 7,486 evidence records, and 215 reflection issues. Additional experiment folders include claim-level retrieval slices. This catalog summarizes the main study families; each row retains its own sample definition.

## Study families

| Study family | Sample and design | Current record | Public package |
|---|---|---|---|
| Corpus and journal/source score profile | 149 PDFs; 146 unique | 144 complete reports; two parser-blocked; source composition and dimension summaries | [Audit and title-free outcomes](../04-reproducibility/corpus-audit.md) |
| Cross-model exploratory comparison | Three shared papers × five models × two recorded runs (30 scores) | Model means range from 77.91 to 89.99; evidence was not fixed between runs, so results are descriptive | [Five-model comparison](cross-model-results.md) |
| Reviewer structure comparison | Five papers, four evaluation structures | Full PaperEval, unified Reviewer, Reviewer-only, and direct scoring conditions are summarized descriptively | [Module ablations](module-ablation.md) |
| Harness fault injection | 180 injected faulty outputs and 40 clean controls across nine fault types | Deterministic checks detected all injected faults; 88.9% were automatically contained and 11.1% routed for review | [Robustness experiments](robustness-experiments.md) |
| Prompt-injection robustness | Three papers, six reviewer dimensions, three attack types, two repeats, paired across prompt conditions | Evidence-boundary prompt comparison reduced target-hit and rating-shift counts in this pilot | [Robustness experiments](robustness-experiments.md) |
| Repeatability and uncertainty | Current manuscript: 20 papers × two independent runs; separate historical diagnostic: 57 run records across 20 papers | Current sample: paired mean absolute score difference 2.382, maximum 8.388, and grade agreement 85.0%; historical diagnostics are separately reported below | [Repeatability and uncertainty](robustness-experiments.md#two-run-repeatability-in-the-current-experiment-manuscript) |
| Journal-gradient comparison | First sample: 12 papers; second sample: 15 additional papers | Kendall's tau = +0.500 against journal-tier ordering in the archived analysis | [Historical experiment evidence](experiment-evidence.md) |
| Scoring-mechanism studies | Mechanism comparisons on three-paper samples | Studies examined judgment scales, two-pool scoring, evidence budgets, thresholds, and weight allocation | [Historical experiment evidence](experiment-evidence.md) |
| External-literature retrieval | Archived query/citation/semantic routes and multi-route ablations; seven-paper/seven-claim study setup | Gold candidates are machine-prelabelled from citation links; a claim-count mismatch in archived reports is being reconciled before quantitative conclusions are released | [Retrieval evaluation](retrieval-evaluation.md) |

## Reliability and uncertainty

The offline reliability analysis combines 57 historical run records across 20 papers. For one illustrative paper with three comparable runs, the total-score mean was 69.78, standard deviation 2.86, and range 6.36. The analysis reported evidence overlap of 0.6197, judgment agreement of 0.9314, batch-ranking Spearman correlation of 0.6491, and Top-5 overlap of 0.25. The diagnostic separated run variance, evidence variance, judgment variance, and calibration uncertainty.

These diagnostics summarize archived runs only. The 57 records span historical runs and do not form a fixed-evidence, five-repeat controlled experiment.

## Model comparison

The completed comparison reports the five models with two recorded runs on each of three shared papers. The [model results](cross-model-results.md) show per-paper runs, means, and run-to-run differences. All experiments use the same expert-calibrated scoring-rule set.

## Data release boundary

This repository publishes study designs, aggregate outcomes, and title-free scores. The local archive also contains PDFs, paper-linked traces, excerpts, API logs, and caches. Those raw materials are not part of this public release.
