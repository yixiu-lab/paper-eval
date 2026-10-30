# Research Artifact: PaperEval Prototype

## What reviewers can inspect

This folder contains the PaperEval command-line workflow, reviewer-agent modules, V16 draft scoring profile, retrieval components, example configurations, prompts, and validation scripts. The local Web/API prototype is outside the artifact scope.

## Reviewer walkthrough

1. Read the [project overview](../01-proposal/README.md) for the research question and empirical record.
2. Inspect [`paper_eval/`](paper_eval/) for parsing, evidence assembly, rule-level review, scoring, and report generation.
3. Inspect the V16 profile at [`rules/rules_v1.yaml`](rules/rules_v1.yaml) and the [system baseline](system-baseline.md).
4. Review the [149-PDF audit](../04-reproducibility/original-149-v16-audit.md), [title-free outcomes](../04-reproducibility/original-149-v16-results.csv), and [experiment evidence](../03-experiments/experiment-evidence.md).
5. Review the [reproducibility and data-access notes](../04-reproducibility/README.md) and the [ethics and data-handling statement](../05-ethics-access/README.md).

## Implemented workflow

PaperEval processes a research-paper PDF, assembles evidence for rule-level judgments, retrieves related literature, aggregates judgments using declared weights, and generates traceable reports. Evidence references and uncertainty indicators let researchers inspect how a result was formed.

The checked-in scoring profile matches the one used in the 149-PDF audit. The audit completed 144 reports; two papers were blocked at PDF parsing.

## Validation scripts

The scripts cover rule-profile structure, scoring and rule applicability, evidence gates and retrieval, report rendering, reliability diagnostics, and model-request formatting. They run in three modes:

- **Local logic checks** use local fixtures and exercise parsing, scoring, reporting, or reliability components.
- **Mocked-service checks** replace network calls with captured requests and simulated responses.
- **Live integration runs** contact configured services. Run these only with papers approved for the selected processing route. Scripts may write reports, caches, and trace files to local `runs/`.

## Access pathway

The source PDFs and full paper-level traces remain in the research workspace and are not included in this repository. See the [ethics and data-handling statement](../05-ethics-access/README.md).
