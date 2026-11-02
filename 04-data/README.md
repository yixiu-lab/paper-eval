# Data and Research Materials

This directory contains paper-level outcomes and a source index for the PaperEval experiment. The evaluation covers **200 papers**, with one completed score per distinct source-PDF hash. All records use the same calibrated criteria and scoring model.

The final set contains papers with analyzable full text. Two entries whose local copies did not contain usable article text were replaced before scoring with two complete AI-HRM research papers from the same journal sample. The index and outcomes describe the final 200-paper set.

## Files

| Item | Contents |
| --- | --- |
| [`paper-outcomes.csv`](paper-outcomes.csv) | Title-free score outcomes and processing status for all 200 papers |
| [`paper-index.csv`](paper-index.csv) | One-row-per-paper index with journal and DOI metadata, score status, license note, and PDF availability |
| [`papers/`](papers/) | 51 unchanged PDFs whose source copies state a Creative Commons license |

## Evaluation results

All **200 papers** have completed scores. The mean total score is **76.6655** (SD 8.5668), the median is **76.4897**, and the range is **49.6342–95.7813**.

The outcome table includes dimension scores A–F, the total score, the model and criteria used, and whether the associated local report passed the project checks. It omits titles, local paths, and source text. Each paper is identified by a checksum-derived ID.

## Outcome fields

| Field | Definition |
| --- | --- |
| `sample_id` | Identifier derived from the first 12 characters of the source PDF's SHA-256 hash |
| `source_sha256` | SHA-256 hash used to identify the source file and exact duplicates |
| `model` | Recorded scoring model |
| `rule_set` | Calibrated assessment criteria |
| `record_status` | Score-processing status; all 200 current records are `scored` |
| `score_total`, `dimension_A`–`dimension_F` | Total and six dimension scores |
| `report_file_verified` | Whether the associated local report passed project checks; report files are not included here |

The [paper index](paper-index.csv) joins each outcome to DOI and journal metadata when available. It also records whether an unchanged PDF is included and the license statement identified in the source copy.

## PDF availability

The `papers/` directory contains **51 unchanged PDFs** whose source copies explicitly state a Creative Commons license. The paper index records their source hashes and repository paths. The other 149 records retain their scores and available bibliographic metadata, but have no PDF copy in the repository because redistribution permission is unconfirmed or the source copy states “All rights reserved.” Readers should follow the notice in each released PDF and the [Creative Commons license terms](https://creativecommons.org/share-your-work/cclicenses/).

## Experimental code

The experiment section includes scripts for the deterministic Harness fault-injection analysis and for recomputing descriptive statistics from the outcome table. See [`03-experiments/code/`](../03-experiments/README.md#experiment-code).

See the [system overview](../02-system/README.md) for the scoring workflow and the [ethics and data-access statement](../05-ethics-access/README.md) for the project's data-handling description.
