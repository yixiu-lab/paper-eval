# Historical Experiment Evidence (Aggregate Summary)

**Source:** the authors' curated “60/40 Two-Pool Scale” evidence package, indexed October 8, 2026. The package index reports a trace database containing 17 papers, 7,486 evidence records, 72 evaluation runs, and 2,342 agent judgments. This page summarizes the package index; it is not an independent reanalysis of raw runs.

## Experiments recorded

| Study | Scale reported in the evidence index | Reported observation | What it supports—and does not support |
|---|---:|---|---|
| API batch evaluation | 10 papers | All completed without reported failures; mean score 62.2899 under the then-current five-level scale. | Evidence that the batch workflow ran on this sample; not evidence of score validity. The raw API run directory was not included in the curated bundle. |
| Cross-journal gradient, a weak criterion | 12-paper and 15-paper samples (27 combined) | Kendall's tau = +0.500 against journal-tier ordering; the index reports a bootstrap interval that did not cross zero. It also reports overestimation at the low end. | Preliminary association with a weak proxy, not validation against expert judgments or a causal effect. The exact interval and analysis code should be published with the raw aggregate data before quoting inferential precision. |
| Repeated-run reliability | Four batches (v23/v23b/v23c/v23d), three papers per batch | Absolute scores varied across batches (a reported example range was 5.25 points), while relative ordering was more stable; some rule-level agreement was as low as 0.5. | Documents score and judgment instability; does not establish robust reliability. |
| Scoring-mechanism comparisons | Iterations V9, V10, V11, and V12, with reruns and prompt/anchor variants | The index reports that wording and instruction changes were not distinguishable from noise, while the two-pool mechanism changed discrimination behavior. | Supports an iterative design history and a mechanism comparison on limited samples; not a definitive controlled comparison across a representative corpus. |
| Evidence-budget sensitivity | Three papers, 10 vs. 20 evidence passages per rule | The reported `satisfied` share changed from 4.9% to 9.8%, `partial` from 65.7% to 47.1%, and total scores shifted upward by about 7 points. | Shows the scoring output is sensitive to evidence budget; budget must be versioned. Does not establish that either budget is optimal. |
| Threshold and grade-band calibration | Four repeated batches on three papers | Thresholds were selected to avoid observed flips (bonus thresholds 0.65/0.30; grade bands 88/85/74). The index reports a margin of 0.109 (about 4.4 points), smaller than between-batch variation. | Shows calibration is fragile at this sample size; thresholds and grades are not externally validated. |
| Weight and pool rebalancing | 12-paper and 15-paper samples | The index reports two replications in which weight rebalancing did not alter ranking, while pool size changed score scale but not rank. | Preliminary sensitivity evidence only; not evidence that the chosen 60/40 split is optimal. |

## Explicitly untested claims and limitations

- The evidence does **not** show that 60/40 is the optimal pool ratio; a same-sample 50/50, 60/40, and 70/30 comparison was not conducted.
- The expert benchmark remains unfilled. The package identifies a planned 20-paper × 3-expert annotation effort (about 2,040 rule-level judgments), but this is a gap, not completed data.
- Existing samples are published papers. The package notes that this cannot establish whether the baseline pool rejects low-quality or off-topic work.
- The reported journal-tier relationship uses a weak proxy and exhibits low-end overestimation.
- Repeated runs show meaningful absolute-score variation and low agreement on some rules.
- These records span historical code/rule versions. The run-level code fingerprints must be mapped to the repository revision before treating the results as reproducible from the current GitHub snapshot.

## Safe sharing and reproduction status

The source evidence package contains per-paper reports, manifests, logs, and a trace database; some traces may include copyrighted paper excerpts. This repository therefore includes only the aggregate summary until the authors verify redistribution rights and prepare a sanitized, version-linked subset. The evidence package index states that 149 corpus PDFs and API credentials were not part of the package; neither should be committed.

The source inventory and exact reported file locations are documented in the authors' local evidence package index (not included in this repository). The summary above should be checked against the underlying run artifacts before use in a submission.
