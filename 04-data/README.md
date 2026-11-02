# Data and Research Materials

This directory links paper-level scores, source inventories, and PDF copies that are explicitly marked for redistribution. The released outcome table combines the primary corpus audit with additional scored papers from focused journal and theory studies.

## Directory guide

| Item | Contents |
| --- | --- |
| [`paper-outcomes.csv`](paper-outcomes.csv) | Title-free scores and processing status for 200 distinct source-PDF records: 198 scored and two blocked before scoring |
| [`sample-paper-index.csv`](sample-paper-index.csv) | DOI, checksum, license note, and release status for the primary corpus input set |
| [`supplementary-paper-index.csv`](supplementary-paper-index.csv) | Source study, journal where identified, DOI, score, and PDF release status for 54 additional scored papers |
| [`sample-papers/`](sample-papers/README.md) | 29 unchanged primary-corpus PDFs with a Creative Commons license identified in the source copy |
| [`scored-papers/`](scored-papers/) | 20 unchanged supplementary PDFs with an explicit Creative Commons license in the source copy |

## Scored-paper counts

The primary corpus audit contains **144 completed paper scores** across 146 distinct source-PDF hashes; two papers were blocked before scoring by PDF parsing. The primary-corpus mean total score was **76.46** (SD 7.64), the median was **76.19**, and the range was **49.63–95.78**.

The release also includes **54 additional distinct papers** with completed score reports. These papers come from journal-comparison, multi-journal expansion, management and AI working-paper, and stratified evaluation samples. The reports record the same model and calibrated scoring criteria as the primary corpus. The additional set has a mean score of **77.00** (SD 10.72), a median of **78.18**, and a range of **53.87–93.53**.

Across the two non-overlapping sets, the title-free table contains **198 scored papers** and two primary-corpus parse blocks. The combined mean is **76.60** (SD 8.57), the median is **76.43**, and the range is **49.63–95.78**. The combined summary describes these released score records; manuscript results specific to the 144-paper primary audit remain reported separately in the experiment materials.

The paper-level outcomes omit titles, local paths, and source text. Papers are identified by a checksum-derived ID. The supplementary index provides DOI and journal metadata when available and links to a PDF only when the source copy states a Creative Commons license.

## Outcome fields

| Field | Definition |
| --- | --- |
| `sample_id` | Identifier derived from the first 12 characters of the source PDF's SHA-256 hash |
| `source_sha256` | SHA-256 hash used to identify the source file and exact duplicates |
| `evaluation_group` | `main_corpus` for the primary audit or `supplementary_experiments` for additional scored papers |
| `model` | Recorded scoring model label; blank when no scoring run was completed |
| `rule_set` | Calibrated assessment criteria used for the score; blank when scoring did not start |
| `record_status` | `scored` for a completed paper-level score; `pdf_parse_blocked` when parsing stopped before scoring |
| `score_total`, `dimension_A`–`dimension_F` | Total and dimension scores when scoring completed; blank for parse-blocked records |
| `report_file_verified` | Whether the associated local report passed the project audit checks; report files are not included here |

## PDF release

The primary sample folder contains 29 unchanged PDFs, and `scored-papers/` contains 20 unchanged supplementary PDFs. The supplementary index lists all 54 additional scored papers; it marks the other 34 as metadata-and-score records without a released PDF because the available copy either states “All rights reserved” or does not establish redistribution permission. No PDF was modified to prepare this release. Readers should follow each included PDF's license notice and the [Creative Commons license terms](https://creativecommons.org/share-your-work/cclicenses/).

The primary source inventory and PDF availability are documented separately in the [primary-paper index](sample-paper-index.csv).

## Relation to experiments

The 144-paper audit and the additional 54-paper score set are separate evaluation groups. The additional papers were scored under the same model and calibrated criteria, but they originated in different focused study samples. The [experiment index](../03-experiments/README.md#supplementary-scored-papers) reports their sample sources and links to the corresponding journal/source records. Repeated runs of a paper in other experiments are not counted again in the 198-paper outcome table.

See the [system overview](../02-system/README.md) for the scoring workflow and the [ethics and data-access statement](../05-ethics-access/README.md) for the project's data-handling description.
