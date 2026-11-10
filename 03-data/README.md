# Data and Research Materials

This directory contains paper-level outcomes and a source index for the PaperEval experiment. The evaluation covers **200 papers**, with one completed score per distinct source-PDF hash. The 200 paper-level results use the revised integer-weight scoring profile. Archived rule-level judgments were aggregated with this profile; model judgments were not rerun for this recalculation.

The final set contains papers with analyzable full text. Two entries whose local copies did not contain usable article text were replaced before scoring with two complete AI-HRM research papers from the same journal sample. The index and outcomes describe the final 200-paper set.

## Files

| Item | Contents |
| --- | --- |
| [`paper-outcomes.csv`](paper-outcomes.csv) | Title-free score outcomes and processing status for all 200 papers |
| [`paper-index.csv`](paper-index.csv) | One-row-per-paper index with journal and DOI metadata, score status, license note, and PDF availability |
| [`papers/`](papers/) | 51 unchanged PDFs whose source copies state a Creative Commons license |
| [`figures/`](figures/) | Figures embedded in the Technical Approach and experiment documentation |

## Evaluation results

All **200 papers** have completed scores under the revised integer-weight profile. For readability, score values in the narrative are rounded to whole points: mean **77** (SD **9**), median **76**, and range **50–96**. Exact recalculated values remain in the CSV files. The totals were calculated from 6,800 archived rule-level judgments; the underlying model judgments were not rerun.

The outcome table includes dimension scores A–F, the total score, the recorded model, the scoring profile, the calculation basis, and whether the associated local report passed the project checks. It omits titles, local paths, and source text. Each paper is identified by a checksum-derived ID.

## Outcome fields

| Field | Definition |
| --- | --- |
| `sample_id` | Identifier derived from the first 12 characters of the source PDF's SHA-256 hash |
| `source_sha256` | SHA-256 hash used to identify the source file and exact duplicates |
| `model` | Recorded scoring model |
| `rule_set` | Revised integer-weight scoring profile |
| `record_status` | Score-processing status; all 200 current records are `scored` |
| `score_total`, `dimension_A`–`dimension_F` | Recalculated total and six dimension scores |
| `score_basis` | Indicates totals were reaggregated from archived rule-level judgments using integer weights |
| `report_file_verified` | Whether the associated local report passed project checks; report files are not included here |

The [paper index](paper-index.csv) joins each outcome to DOI and journal metadata when available. It also records whether an unchanged PDF is included and the license statement identified in the source copy.
