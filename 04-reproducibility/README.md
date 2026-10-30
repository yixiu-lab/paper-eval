# Reproducibility and Access Package

## Materials in this repository

- PaperEval source code and draft scoring profile V16.
- Python dependency list and example configurations.
- Validation scripts for local logic, mocked services, and live integrations.
- [Historical mechanism experiment summary](../03-experiments/experiment-evidence.md).
- [Experiment catalog](../03-experiments/experiment-catalog.md), covering the 149-PDF audit and additional model, reliability, robustness, ablation, and retrieval studies.
- [Cross-model V16 comparison](../03-experiments/cross-model-v16.md).
- [Robustness, fault-injection, and uncertainty experiments](../03-experiments/robustness-experiments.md).
- [Module ablations](../03-experiments/module-ablation.md).
- [External-literature retrieval evaluation](../03-experiments/retrieval-evaluation.md).
- [149-PDF V16 audit](original-149-v16-audit.md).
- [Title-free, per-paper outcome table](original-149-v16-results.csv), one row per unique source PDF.

## Existing research records

The authors' curated evidence package, indexed October 8, 2026, reports a 17-paper trace database with 7,486 evidence records, 72 evaluation runs, and 2,342 agent judgments across seven experiment groups. This is a focused experimental subset. The separate October 26, 2026 audit reconciles the 149-PDF corpus to 146 unique papers, 144 complete reports, and two parse-blocked papers.

The current repository presents aggregate findings and title-free outcomes. Full traces and source PDFs remain in the research workspace; the public package does not include paper-linked full-text traces.

## Current V16 run facts

The checked-in scoring profile matches the profile used in the audit. The historical run used DeepSeek Flash, temperature 0, thinking disabled, and `MGMT-SCALE-V6`. The audit reports 144 complete reports out of 146 unique papers; two papers stopped at PDF parsing and have no score.

## Run manifest fields

For each new run, record: run ID and date; source-data version and access class; code revision; Python and dependency versions; model/provider and endpoint; rule-profile fingerprint; prompt version; retrieval providers and parameters; evidence budget; cache state; random seed where applicable; output location; failures; and responsible operator. Store credentials separately from the manifest.

## Validation script guide

“Offline self-check” means a check that runs without contacting a model or external retrieval service. The repository includes both offline checks and live integration scripts, so check the mode before running:

| Check type | What it exercises | Network behavior |
|---|---|---|
| Local logic | Rule validation, scoring, evidence gates, parsing components, report rendering, reliability, and local caches | No external service calls; uses local inputs or fixtures |
| Mocked service | Model request formatting and retrieval behavior with fake fetchers or simulated responses | Calls are intercepted locally; no provider receives the test payload |
| Live integration | End-to-end evaluation through `02-system/scripts/自检_mvp.py`; optional external retrieval probe through `02-system/scripts/自检_外部检索.py --online` | Uses configured model/retrieval services and may send paper-derived text or query terms |

The `02-system/scripts/自检_mvp.py` script is a live integration run despite its “self-check” name: its source config selects a model provider, and the evaluation workflow can send selected paper evidence to that endpoint. Run it only with a source paper and provider route cleared for that processing. When run from `02-system/`, generated reports and caches go under the ignored `02-system/runs/` directory; review them before sharing.
