# Historical Experiment Evidence and Next Studies

**Source:** the authors' curated “60/40 Two-Pool Scale” evidence package, indexed October 8, 2026. The package index reports 17 papers, 7,486 evidence records, 72 evaluation runs, and 2,342 agent judgments. This summary organizes those records into current observations and follow-up studies.

The 17-paper trace database is a focused subset. The separate V16 full-corpus audit covers 149 source PDFs, 146 unique papers, 144 complete reports, and two parse-blocked papers. See the [audit](original-149-v16-audit.md) and [title-free outcome table](original-149-v16-results.csv).

## Experiments and follow-up milestones

| Study | Evidence in the index | Current observation | Next study |
|---|---:|---|---|
| API batch evaluation | 10 papers | All completed without reported failures; mean score 62.2899 under the then-current five-level scale. | Publish the run manifest and compare the batch results with expert assessments on a fixed sample. |
| Cross-journal gradient | 12-paper and 15-paper samples (27 combined) | Kendall's tau = +0.500 against journal-tier ordering; the index reports a bootstrap interval that did not cross zero and overestimation at the low end. | Publish raw aggregate inputs, exact interval, and analysis code; test against a stronger criterion and expert ratings. |
| Repeated-run reliability | Four batches (v23/v23b/v23c/v23d), three papers per batch | Absolute scores varied across batches (reported example range 5.25 points); relative ordering was more stable; rule-level agreement reached 0.5 for some rules. | Repeat on a larger fixed sample and report score variance and rule-level agreement by dimension. |
| Scoring-mechanism comparisons | V9, V10, V11, and V12 iterations with reruns and prompt/anchor variants | The index reports wording and instruction changes near the observed noise level; the two-pool mechanism changed discrimination behavior. | Run a pre-specified comparison with shared papers, prompts, and expert references. |
| Evidence-budget sensitivity | Three papers; 10 vs. 20 passages per rule | The `satisfied` share changed from 4.9% to 9.8%, `partial` from 65.7% to 47.1%, and total scores shifted upward by about 7 points. | Compare fixed budgets on the same expanded sample and analyze the evidence behind changed judgments. |
| Threshold and grade-band calibration | Four repeated batches on three papers | Thresholds were selected to avoid observed flips (bonus thresholds 0.65/0.30; grade bands 88/85/74); reported margin was 0.109 (about 4.4 points). | Estimate calibration on expert-labeled data and report uncertainty around each threshold. |
| Weight and pool rebalancing | 12-paper and 15-paper samples | Two replications reported unchanged rankings after weight rebalancing; pool size changed score scale while ranks remained stable. | Compare 50/50, 60/40, and 70/30 on a fixed sample against expert ratings. |

## Validation sequence

1. Map each historical result to its source run, code revision, rule version, and runtime configuration.
2. Build an expert-labeled sample with a documented rater protocol and institutional review determination.
3. Pre-register agreement, stability, calibration, and subgroup analyses.
4. Repeat scoring under fixed evidence and model settings, then vary one factor at a time.
5. Extend beyond published AI/HRM papers and report results by paper type and evidence conditions.

## Sharing and access

The source evidence package contains reports, manifests, logs, and a trace database; the trace inventory indicates that records may contain paper excerpts. The team will complete source-level sharing review and prepare a minimized, version-linked release. The shareable package will pair approved aggregate data and manifests with a benchmark or synthetic substitute that supports independent reruns.
