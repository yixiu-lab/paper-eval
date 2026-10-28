# Organization Science Stage 1 Package

**Call:** *AI-Enabled Frontiers in Organizational Science*

**Stage 1 deadline in the call:** November 1, 2026

## Project in one minute

PaperEval is an evidence-grounded workflow for studying AI-assisted assessment of management research. It links rule-level judgments to selected paper evidence, records uncertainty for human review, and aggregates judgments with a versioned profile. The project treats assessment as a research measurement process that can be inspected and compared with expert judgments.

The V16 audit covers 149 source PDFs: 146 unique papers after exact deduplication, 144 complete reports, and two papers blocked during PDF parsing. Review the [audit](04-reproducibility/original-149-v16-audit.md) and [title-free outcomes](04-reproducibility/original-149-v16-results.csv).

## Fit with the call

PaperEval advances reusable research infrastructure and new forms of measurement. Its next study will connect evidence-grounded judgments to expert assessments, measure stability and calibration, and map performance across paper types. The proposal will explain how this capability can change the way organizational researchers study and evaluate evidence.

## Stage 1 materials and milestones

| Required item | Current asset | Next milestone |
|---|---|---|
| Short proposal (3–5 pages / 1,500–2,000 words plus references) | Research framing and evidence are documented in the repository overview. | Draft the call-aligned proposal with author-confirmed research question, contribution, validation design, and references. |
| Technical appendix or supplement | A draft describes the workflow, V16 scoring design, and run records. | Add the expert-comparison design and version-linked reproduction details. |
| Prototype and repository | PaperEval CLI, scoring profile V16, retrieval components, prompts, and offline self-checks are included. | Add a concise reviewer walkthrough and versioned release record. |
| Reproducibility evidence | Historical experiments and the 149-PDF audit are summarized; title-free outcomes are included. | Link runs to exact code/runtime manifests and prepare a shareable benchmark or synthetic substitute. |
| Ethics, data rights, and model access | An action plan is documented in [Ethics, Data, and Access](05-ethics-and-access/README.md). | Complete source-level access inventory, institutional review steps for expert raters, provider terms, retention, and editor access plan. |

## Reader path

1. Begin with the [project overview](../README.md).
2. Inspect the [technical appendix](02-technical-appendix/README.md) and [research artifact](03-research-artifact/README.md).
3. Review the [149-PDF V16 audit](04-reproducibility/original-149-v16-audit.md) and [historical experiment evidence](04-reproducibility/experiment-evidence.md).
4. Follow the [reproduction plan](04-reproducibility/README.md) and [ethics/access plan](05-ethics-and-access/README.md).

## Evidence-to-study path

The current evidence establishes a working workflow and a reconciled corpus-scale record. The next study will connect those records to expert-labeled judgments, preregister comparisons of scoring profiles, and evaluate performance across broader paper types. The two parse-blocked cases remain visible in the coverage record and will inform parser improvement.
