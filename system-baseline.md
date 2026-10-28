# PaperEval System Baseline

## Current artifact

The repository contains the PaperEval command-line workflow and the V16 draft scoring profile used in the 149-PDF audit. The profile has 34 rules and SHA-256 `811f6c0d992460113a88e66bcb37a4124da046b7c0d5b6725e1d653549284c0f`.

The historical audit used `MGMT-LIT-EVAL-V1:V16`, DeepSeek Flash, temperature 0, thinking disabled, and `MGMT-SCALE-V6`. The checked-in rule snapshot is version-matched; the next artifact milestone is to attach the exact historical source revision and complete runtime configuration to the run manifest.

## Entry points

- Single-paper evaluation: `python -m paper_eval.cli evaluate --pdf <PDF_PATH> --rules <RULES_PATH> --out <OUTPUT_DIR>`.
- Rule-profile validation: `python -m paper_eval.cli validate-rules --rules <RULES_PATH>`.
- Offline self-checks: scripts under `scripts/`.
- Package: `paper_eval/`.

The research artifact centers on the CLI and reusable evaluation components. The local Web/API prototype is outside this repository’s scope.

## Evaluation workflow

PDF parsing → paper profiling and claim extraction → related-work retrieval → evidence-pack construction → six rule-guided reviewer dimensions → evidence/citation checks and uncertainty handling → deterministic aggregation → traceable report and reliability diagnostics.

The six dimensions cover research-question positioning, theoretical contribution, theoretical rigor, research design, results/evidence, and discussion/boundary conditions. An auxiliary novelty comparison is separate from the six scoring dimensions and does not directly contribute to the total score.

## V16 scoring design

V16 separates a 60-point baseline pool from a 40-point bonus pool. Baseline rules represent adequacy; bonus rules represent evidence beyond adequacy. Uncertain judgments are excluded from the available-score denominator and reported with completeness. The next calibration study will compare pre-specified alternatives, including 50/50, 60/40, and 70/30, against expert judgments.

## Evidence and next milestones

The V16 audit includes 149 source PDFs, 146 unique papers after exact SHA-256 deduplication, 144 complete reports, and two parse-blocked papers. The [audit note](submission/04-reproducibility/original-149-v16-audit.md) and title-free [outcome table](submission/04-reproducibility/original-149-v16-results.csv) document the reconciled results.

The next reproducibility package will link each reported result to its code revision, runtime and dependency versions, model/provider, prompts, rule fingerprint, retrieval settings, and cache state. The data workstream will build a source-level access manifest and a shareable benchmark or synthetic substitute. The validation workstream will compare judgments with expert assessments and report stability across runs and paper types.

## Data and runtime practice

The evaluation workflow sends selected evidence blocks to the configured language-model endpoint. Retrieval can send paper titles and claim-derived search terms to Crossref, OpenAlex, and Semantic Scholar. Each run manifest should record providers, model/version, access date, configuration, and data-handling terms. Credentials belong in local environment variables or secret storage; source PDFs, full-text traces, and credentials stay outside Git until their sharing status and release scope are documented.
