# Corpus Audit

This note summarizes the reconciliation of the source-paper corpus and its evaluation records. Source PDFs, extracted full text, and paper-linked reports remain in the research workspace.

## Rule profile and run configuration

The audit used the scoring profile maintained with the PaperEval system. Historical run settings are recorded in local run manifests and are not reproduced here.

The title-free [paper-level outcomes](data/paper-outcomes.csv) list each unique source paper by a checksum-derived identifier. Paper titles, journal names, local paths, and source text are omitted.

## Corpus coverage

The source set was deduplicated using exact checksum matches. Evaluation reports were completed for papers that passed PDF parsing; parse-blocked cases received no score. The outcomes table records processing status and, where available, scores.

The selected reports passed local checks for profile consistency, model records, dimension scores, and total-score consistency.

## Sample and metadata coverage

The sample consists of published empirical research in AI and human resource management. Journal and source metadata coverage is incomplete; comparisons by publication outlet should be interpreted accordingly.

## Public data scope

The repository includes aggregate descriptions and title-free outcomes. Source PDFs and paper-level traces are not included.
