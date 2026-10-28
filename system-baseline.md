# PaperEval System Baseline

## Scope and version boundaries

This note describes the repository's current CLI-oriented code snapshot. It is not the same artifact revision as the historical V16 full-corpus run summarized in [`submission/04-reproducibility/original-149-v16-audit.md`](submission/04-reproducibility/original-149-v16-audit.md).

The rule profile currently checked into `评价规则库/rules_v1.yaml` is **V16, draft**, with 34 rules. Its SHA-256 is `811f6c0d992460113a88e66bcb37a4124da046b7c0d5b6725e1d653549284c0f`. This is the exact V16 rule snapshot used in the 149-PDF audit. That run used `MGMT-LIT-EVAL-V1:V16`, DeepSeek Flash, temperature 0, and thinking disabled. The run-specific source revision and full runtime configuration are not included, so the present code snapshot should not be represented as an exact rerun environment.

## Current code and entry points

- Python package: `paper_eval/`.
- Single-paper CLI: `python -m paper_eval.cli evaluate --pdf <PDF_PATH> --rules <RULES_PATH> --out <OUTPUT_DIR>`.
- Rule validation: `python -m paper_eval.cli validate-rules --rules <RULES_PATH>`.
- Offline self-checks: scripts under `scripts/`.
- The local Web/API prototype and the earlier `scripts/run_batch_evaluation.py` entry point are not included. The package retains `paper_eval/batch_runner.py`, but the current CLI has no batch subcommand.

## Evaluation workflow

PDF parsing → paper profiling and claim extraction → related-work retrieval → evidence-pack construction → six rule-guided reviewer dimensions → evidence/citation checks and uncertainty handling → deterministic aggregation → traceable report and reliability diagnostics.

The six dimensions cover research question positioning, theoretical contribution, theoretical rigor, research design, results/evidence, and discussion/boundary conditions. An auxiliary novelty comparison is separate from the six scoring dimensions and does not directly produce the total score.

## Scoring profile

The current V16 draft uses a baseline pool and a bonus pool, nominally 60 and 40 points. Baseline rules aim to represent adequacy; bonus rules represent evidence beyond adequacy. Uncertain judgments are excluded from the available-score denominator and must be reported with completeness. This is a design under study, not a validated general-purpose scale. Rule explanations and most runtime-facing rule labels remain Chinese.

## Evidence and limits

The V16 historical audit covers 149 source PDFs, 146 unique papers after exact SHA-256 deduplication, 144 complete reports, and two papers blocked before scoring by PDF parsing. Full details and title-free outcomes are linked in the audit note. The experiment does not include expert-labeled validity evidence; the selected sample consists of published AI/HRM empirical papers and cannot establish performance on rejected or off-topic manuscripts.

The source corpus and full paper-level traces are not included. A third party cannot independently reproduce the V16 run from the current repository alone. The exact code revision and runtime configuration must still be matched to the historical run, and rights for sharing paper data must be reviewed before claiming independent reproducibility.

## Security and runtime requirements

Provide provider credentials through local environment variables or untracked configuration. Never commit API keys, `.env` files, private manuscripts, full-text reports, or restricted data. Record the model/provider, code revision, rule-profile fingerprint, retrieval parameters, cache state, and failures for any new run.
