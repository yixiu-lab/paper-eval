# Historical Experiment Evidence and Results

**Source:** the authors' curated “60/40 Two-Pool Scale” evidence package, indexed October 8, 2026. The package index reports 17 papers, 7,486 evidence records, 72 evaluation runs, and 2,342 agent judgments. This summary organizes the archived study results.

This is the earlier scoring evidence stream. The project also has cross-model, uncertainty, prompt-injection, harness fault-injection, module-ablation, and external-retrieval records. See the [full experiment catalog](experiment-catalog.md) and its linked summaries.

The trace database is a focused subset of the broader evaluation program. The primary corpus audit contains completed scores for 144 papers; two additional papers were blocked before scoring by PDF parsing. A further 54 distinct papers from journal and focused samples are included in the unified title-free outcome table, bringing the released scored-paper record to 198. See the [data overview](../04-data/README.md#scored-paper-counts), [supplementary paper index](../04-data/supplementary-paper-index.csv), and [title-free outcome table](../04-data/paper-outcomes.csv).

## Archived experiment results

| Study | Evidence in the index | Observed result |
|---|---:|---|
| API batch evaluation | 10 papers | All completed without reported failures; mean score 62.2899 under the then-current five-level scale. |
| Cross-journal gradient | 12-paper and 15-paper samples (27 combined) | Kendall's tau = +0.500 against journal-tier ordering; the index reports a bootstrap interval that did not cross zero and overestimation at the low end. |
| Repeated-run reliability | Four repeated batches, three papers per batch | Absolute scores varied across batches (reported example range 5.25 points); relative ordering was more stable; rule-level agreement reached 0.5 for some rules. |
| Scoring-mechanism comparisons | Historical scoring and prompt variants | The index reports wording and instruction changes near the observed noise level; the two-pool mechanism changed discrimination behavior. |
| Evidence-budget sensitivity | Three papers; 10 vs. 20 passages per rule | The `satisfied` share changed from 4.9% to 9.8%, `partial` from 65.7% to 47.1%, and total scores shifted upward by about 7 points. |
| Threshold and grade-band calibration | Four repeated batches on three papers | The archived analysis reports bonus thresholds of 0.65/0.30, grade bands of 88/85/74, and a margin of 0.109 (about 4.4 points). |
| Weight and pool rebalancing | 12-paper and 15-paper samples | Two replications reported unchanged rankings after weight rebalancing; pool size changed score scale while ranks remained stable. |

The source evidence package includes reports, manifests, logs, and a trace database that may contain paper excerpts. This repository publishes study descriptions, aggregate outcomes, and title-free records; source PDFs and full paper-linked traces are not included.
