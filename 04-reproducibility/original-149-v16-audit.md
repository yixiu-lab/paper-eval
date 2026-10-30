# Original 149-PDF Corpus: V16 Scoring Audit

**Audit date:** October 26, 2026  
**Source record:** final audit note and reconciled manifest in the authors' local experiment workspace.

## Profile and run configuration

The repository includes the [V16 scoring-rule snapshot](../02-system/rules/rules_v1.yaml), SHA-256 `811f6c0d992460113a88e66bcb37a4124da046b7c0d5b6725e1d653549284c0f`. The historical run used `MGMT-LIT-EVAL-V1:V16` with 34 rules, DeepSeek Flash, temperature 0, thinking disabled, and scoring scale `MGMT-SCALE-V6`.

The title-free [per-paper outcome table](original-149-v16-results.csv) contains one row per unique paper, with scores where available. Each ID is derived from the source PDF’s SHA-256 checksum; the table omits titles, journal names, paths, and paper text.

## Corpus coverage

- Source folder: **149 PDF files**.
- Exact SHA-256 deduplication: **146 unique papers**; three files were exact duplicates.
- Complete reports: **144 of 146 unique papers** (98.6%), with all six dimension scores.
- Parsing cases: **two unique papers** had two saved attempts each marked `pdf_parse_blocked`; evaluation stopped before model scoring and no score was generated.
- Report checks: all 144 selected reports pass the local audit checks for profile, model, scale, dimensions, and total-score consistency.

The audit also records a 73-paper V16 backfill: 71 produced complete reports, while two remained blocked during PDF parsing. The reconciled set includes complete reports recovered from earlier run locations.

## Descriptive score summary

Across the 144 complete unique-paper reports, the reconciled appendix reports a mean total score of **76.46** (SD **7.64**), median **76.19**, and range **49.63–95.78**. These statistics describe this corpus and configuration; they are not expert-validity estimates.

## Sample characteristics

The audited sample consists of published AI/HRM empirical papers. It does not include expert-rater annotations.

Journal/source metadata are confirmed for 90 unique papers; the local audit marks 56 as unconfirmed. Journal-tier comparisons should be interpreted with this metadata coverage in mind.

Two parse-blocked papers remain in the coverage record and have no score.

## Data access and provenance

The PDFs and full paper-level reports are held in the research workspace and are not included in this repository. Public materials are limited to the title-free outcome table and aggregate audit description.

The audit was reconciled from `PaperEval/data/original_149_v16_round/manifest.csv`, `PaperEval/runs/original_149_missing_v16_20261026/appendix_scores.csv`, associated run logs, and complete report files distributed across the local workspace. The source audit note is `PaperEval/runs/original_149_missing_v16_20261026/progress.md`.
