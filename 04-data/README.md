# Data and Research Materials

This directory brings together PaperEval's corpus results and released source-paper files. The main evaluation corpus has **144 completed paper scores**. The files below let readers inspect the score outcomes, the source-file inventory, and the subset of PDFs available for redistribution.

## Directory guide

| Item | What it contains |
| --- | --- |
| [`paper-outcomes.csv`](paper-outcomes.csv) | Title-free scores and processing status for 146 unique source papers |
| [`sample-paper-index.csv`](sample-paper-index.csv) | DOI, checksum, license note, and release status for 149 original PDF entries representing 146 unique files |
| [`sample-papers/`](sample-papers/README.md) | 29 unchanged sample PDFs with a Creative Commons license identified in the source copy |

## Corpus results

The source inventory contains 149 original file entries and 146 unique SHA-256 hashes. Of the 146 unique papers, **144 received completed scores** and **two were blocked before scoring by PDF parsing**. No scores were imputed for the blocked papers.

Across the 144 completed reports, the mean total score was **76.46** (SD 7.64), the median was **76.19**, and the range was **49.63–95.78**. The evaluation reports also passed local checks for scoring-profile consistency, model records, dimension scores, and total-score consistency. These are descriptive results for the audited corpus and run configuration.

The outcomes table excludes paper titles, journal names, local paths, and source text. Each unique paper is identified by a checksum-derived ID. `report_file_verified` indicates that the associated report passed local audit checks; individual report files are not included in this release.

## Outcome fields

| Field | Definition |
| --- | --- |
| `sample_id` | Identifier derived from the first 12 characters of the source PDF's SHA-256 hash |
| `source_sha256` | SHA-256 hash used to identify the source file and exact duplicates |
| `record_status` | `scored` for a completed paper-level score; `pdf_parse_blocked` when parsing stopped before scoring |
| `score_total`, `dimension_A`–`dimension_F` | Total and dimension scores when scoring completed; blank for parse-blocked records |
| `report_file_verified` | Whether the associated report passed the local audit checks; the report itself is not part of this release |

In the source-paper inventory, `doi_metadata_source` identifies whether DOI metadata was checked against Crossref or a publisher record. `redistribution_license_noted_in_pdf` records the license statement identified in the source copy. A value of `Not verified` means redistribution permission was not established from that copy.

## PDF release

The repository includes 29 unchanged PDFs: 19 marked CC BY, two marked CC BY-NC, and eight marked CC BY-NC-ND. Every released file's SHA-256 checksum matches its inventory entry. The licenses have different reuse conditions; readers should consult each PDF's notice and the [Creative Commons license descriptions](https://creativecommons.org/share-your-work/cclicenses/).

The other 120 source entries are not included as downloadable PDFs because redistribution permission could not be confirmed from the available copies. DOI metadata was checked against Crossref or publisher records, but a DOI or an accessible article page does not by itself establish permission to redistribute a PDF.

## Relation to the experiments

The 144-paper count refers to completed scores in the corpus audit. Other experiment records analyze overlapping papers, repeated runs, judgments, evidence, and retrieval candidates; those records are not added to the paper count. Study designs and findings are organized in the [experiment catalog](../03-experiments/experiment-catalog.md).

See the [ethics and data-access statement](../05-ethics-access/README.md) for the project's data-handling description.
