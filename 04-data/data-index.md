# Released Study Data

This data section contains title-free outcomes for the corpus audit and a checksum-based inventory of the original PDF inputs. Paper titles, local PDF paths, and full text are excluded from the outcome table.

| File | Contents |
| --- | --- |
| [`paper-outcomes.csv`](paper-outcomes.csv) | Processing status and available total and dimension scores for 146 unique source papers |
| [`study-samples.md`](study-samples.md) | Scope of the released corpus and links to study records |
| [`sample-paper-index.csv`](sample-paper-index.csv) | 149 source-file records, SHA-256 identifiers, DOI values checked against Crossref or publisher records, license noted in the PDF, and PDF release status |
| [`sample-papers/`](sample-papers/README.md) | 29 unchanged PDFs with a Creative Commons license stated in the source copy |

The inventory contains 149 source-file entries and 146 unique SHA-256 hashes. The outcome table records 144 completed scores and two parse-blocked papers. Each of the 29 PDFs in the public collection was checksum-matched to the inventory; the remaining 120 PDFs are not included because redistribution permission could not be confirmed from the available copies. The repository does not include paper-level report files.

## Outcome fields

| Field | Definition |
| --- | --- |
| `sample_id` | Identifier derived from the first 12 characters of the source PDF's SHA-256 hash |
| `source_sha256` | SHA-256 hash of the source PDF used to identify exact duplicates |
| `record_status` | `scored` for a completed paper-level score; `pdf_parse_blocked` when parsing stopped before scoring |
| `score_total`, `dimension_A`–`dimension_F` | Total and dimension scores when scoring completed; blank for parse-blocked records |
| `report_file_verified` | Indicates whether the associated report passed the local audit checks; report files themselves are not included in this release |

In the source-paper inventory, `doi_metadata_source` identifies whether the DOI was checked against Crossref or a publisher record. `redistribution_license_noted_in_pdf` records the license statement found in the source copy; a value of `Not verified` means redistribution permission was not established from that copy.
