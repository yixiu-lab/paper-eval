# Original 149-PDF Corpus: V16 Scoring Audit

**Audit date:** October 26, 2026  
**Source record:** the final audit note and reconciled manifest in the authors' local experiment workspace. This page contains only aggregate results; the paper-level manifest, PDFs, and full reports are not included in this repository.

The accompanying [`original-149-v16-results.csv`](original-149-v16-results.csv) provides one row per unique paper with V16 scores where available. It uses a stable ID derived from the source PDF's SHA-256 checksum and omits titles, journal names, paths, and paper text.

## Corpus coverage

- The source folder contained **149 PDF files**.
- Exact SHA-256 deduplication produced **146 unique papers**; three files were exact duplicates.
- **144 of 146 unique papers** have a complete, traceable V16 report with all six dimension scores. This is 98.6% coverage of the deduplicated corpus.
- **Two unique papers** have no complete report. Each had two saved attempts marked `pdf_parse_blocked`; the evaluation harness stopped before model scoring, so no score was generated or imputed.
- The final audit states that all 144 selected reports exist and pass checks for profile, model, scale, dimensions, and total-score consistency.

## Run configuration

- Rule profile: `MGMT-LIT-EVAL-V1:V16`, 34 rules.
- Model: DeepSeek Flash; temperature 0; thinking disabled.
- Scoring scale: `MGMT-SCALE-V6`.
- A 73-paper V16 backfill was performed for the missing-report list: 71 produced complete reports; the remaining two were blocked during PDF parsing. The reconciled 144-paper result also includes complete V16 reports recovered from earlier run locations.

## Descriptive score summary

For the 144 complete unique-paper reports, the reconciled appendix reports a mean total score of **76.46** (SD **7.64**), median **76.19**, and range **49.63–95.78**. These are descriptive results for this corpus and scoring configuration. They do not establish expert agreement, criterion validity, or that the scoring scale is calibrated for manuscripts generally.

## Interpretation and limits

- The corpus consists of published AI/HRM empirical papers. It is not an expert-labeled benchmark and does not test rejection of low-quality or off-topic manuscripts.
- The corpus includes exact duplicates before deduplication; all coverage rates above use the 146 unique-paper denominator.
- Two parse-blocked papers remain unresolved; they are not treated as zero scores.
- Journal/source metadata are confirmed for 90 unique papers; 56 remain unconfirmed in the local audit and should not be used for journal-tier conclusions without further verification.
- Raw per-paper reports and paper-text traces are withheld from this repository pending copyright, confidentiality, and dataset-license review.
- The exact run-to-code-revision mapping and a clean independent rerun still need to be prepared for third-party reproduction.

## Local source records

The audit was reconciled from `PaperEval/data/original_149_v16_round/manifest.csv`, `PaperEval/runs/original_149_missing_v16_20261026/appendix_scores.csv`, the associated run logs, and complete report files distributed across the local workspace. The source audit note is `PaperEval/runs/original_149_missing_v16_20261026/progress.md`.
