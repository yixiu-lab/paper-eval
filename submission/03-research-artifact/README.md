# Research Artifact: PaperEval Prototype

## What reviewers can inspect

This repository contains the PaperEval command-line workflow, reviewer-agent modules, V16 draft scoring profile, retrieval components, example configurations, prompts, and offline self-check scripts. The local Web/API prototype is outside the artifact scope.

## Reviewer walkthrough

1. Read the [project overview](../../README.md) for the research question, empirical record, and milestones.
2. Inspect [`paper_eval/`](../../paper_eval/) for parsing, evidence assembly, rule-level review, scoring, and report generation.
3. Inspect the V16 profile at [`../../评价规则库/rules_v1.yaml`](../../评价规则库/rules_v1.yaml) and the [system baseline](../../system-baseline.md).
4. Review the [149-PDF audit](../04-reproducibility/original-149-v16-audit.md), [title-free outcomes](../04-reproducibility/original-149-v16-results.csv), and [experiment evidence](../04-reproducibility/experiment-evidence.md).
5. Follow the [reproduction plan](../04-reproducibility/README.md) and [ethics/access plan](../05-ethics-and-access/README.md) for the next study package.

## Implemented workflow

PaperEval processes a research-paper PDF, assembles evidence for rule-level judgments, retrieves related literature, aggregates judgments using declared weights, and generates traceable reports. Evidence references and uncertainty indicators let researchers inspect how a result was formed.

The checked-in scoring profile is V16, the same 34-rule snapshot used in the 149-PDF audit. The audit completed 144 unique-paper reports; two papers were blocked at PDF parsing. The historical run manifest and code revision are the next reproducibility deliverables.

## Research milestones

- **Benchmark:** prepare expert-labeled assessments under institutional review and a documented consent/privacy protocol.
- **Validity and reliability:** compare rule judgments and total scores with experts; quantify repeated-run stability.
- **Calibration:** compare the current 60/40 pool split with pre-specified alternatives.
- **Broader scope:** evaluate additional paper types and document performance by domain and evidence conditions.
- **Release:** provide a shareable dataset or synthetic substitute, version-linked run manifests, and a clean reproduction guide.

## Access pathway

The source PDFs and full paper-level traces remain in the research workspace while the team completes a source-level access inventory. The release bundle will contain items whose sharing conditions are documented, plus a clearly described access route or synthetic substitute for restricted source material. See the [ethics, data, and access plan](../05-ethics-and-access/README.md).
