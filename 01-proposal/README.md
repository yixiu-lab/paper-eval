# PaperEval

### An evidence-grounded workflow for studying AI-assisted research assessment

PaperEval turns research-paper assessment into an inspectable measurement workflow. It links rule-level judgments to selected paper evidence, records uncertainty for human review, checks evidence and citation conditions, and aggregates judgments using a versioned scoring profile. The project studies how this workflow can support organizational and management research.

## Research objective

The research examines how a human-directed, evidence-grounded workflow assesses management research, how its judgments compare with expert assessments, how stable scores are across runs and evidence settings, and which paper types and conditions require additional review.

The contribution is a testable research capability: paper assessments can be examined at the rule, evidence, and aggregation levels, giving researchers a way to study AI-assisted measurement and its boundary conditions.

## Workflow

1. Parse a research-paper PDF and build a structured paper profile.
2. Extract research claims and retrieve related literature for comparison.
3. Assemble rule-specific evidence for six dimensions: research-question positioning, theoretical contribution, theoretical rigor, research design, results and evidence, and discussion/boundary conditions.
4. Generate rule-level judgments with evidence references and uncertainty indicators.
5. Apply evidence and citation checks, then aggregate results with a versioned scoring profile.
6. Produce a traceable report for researcher inspection.

The primary implementation is in [`paper_eval/`](../02-system/paper_eval). The local Web/API prototype is not part of this research artifact.

## Current empirical record

The V16 audit began with **149 source PDFs**. Exact SHA-256 deduplication yielded **146 unique papers**. The workflow produced **144 complete reports**; two papers were blocked before scoring by PDF parsing and remain without imputed scores. The audit reports a mean total score of 76.46 (SD 7.64), median 76.19, and range 49.63–95.78 across the 144 completed reports. This is one corpus audit, not the total number of papers evaluated across all project experiments.

The checked-in scoring profile matches the profile used in the audit. The historical run used DeepSeek Flash, temperature 0, thinking disabled, and `MGMT-SCALE-V6`.

The local archive also contains repeated-run and uncertainty analyses, scoring-mechanism and evidence-budget studies, a two-model comparison, prompt-injection and harness fault-injection tests, module ablations, and claim-driven retrieval experiments. The 17-paper trace database is a separate subset from the 149-PDF audit, and study samples overlap. The [experiment catalog](../03-experiments/experiment-catalog.md) maps the evidence by study family; individual summaries cover [cross-model results](../03-experiments/cross-model-v16.md), [robustness and uncertainty](../03-experiments/robustness-experiments.md), [module ablations](../03-experiments/module-ablation.md), and [retrieval evaluation](../03-experiments/retrieval-evaluation.md).

## Editor reading path

- [Technical appendix](../02-system/technical-appendix.md): architecture, V16 design, and evaluation plan.
- [Research artifact](../02-system/reviewer-guide.md): implementation and reviewer walkthrough.
- [149-PDF audit](../04-reproducibility/original-149-v16-audit.md) and [title-free outcomes](../04-reproducibility/original-149-v16-results.csv).
- [Historical experiment evidence](../03-experiments/experiment-evidence.md).
- [Full experiment catalog](../03-experiments/experiment-catalog.md), including model, robustness, uncertainty, ablation, and retrieval studies.
- [Reproducibility and data-access notes](../04-reproducibility/README.md).
- [Ethics and data-handling statement](../05-ethics-access/README.md).
- [System baseline](../02-system/system-baseline.md): entry points and version boundaries.

## Repository map

- [`02-system/`](../02-system/README.md): source code, configurations, prompts, V16 rules, technical appendix, and validation scripts.
- [`03-experiments/`](../03-experiments/README.md): experiment catalog and study summaries.
- [`04-reproducibility/`](../04-reproducibility/README.md): corpus audit, outcomes, and reproduction details.
- [`05-ethics-access/`](../05-ethics-access/README.md): ethics and data-handling statement.

## Reproduce the CLI setup

```powershell
Set-Location ..\02-system
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m paper_eval.cli --help
```

Model-backed runs use a configured provider and may incur service costs. For each new run, record provider/model/version, code revision, rule fingerprint, retrieval parameters, cache state, and failures in the run manifest.

## Ethics and access

The current corpus consists of published AI/HRM papers. Source PDFs and full paper-level traces remain in the research workspace and are not included in this repository. The [ethics and data-handling statement](../05-ethics-access/README.md) describes the audit scope, current participant-data status, and evaluation data flows.
