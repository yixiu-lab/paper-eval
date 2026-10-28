# Organization Science Stage 1 Package

**Call:** *AI-Enabled Frontiers in Organizational Science*

**Stage 1 deadline:** November 1, 2026

## Project in one minute

PaperEval is a prototype for evidence-grounded AI assessment of management research. It links rule-level judgments to selected paper evidence, applies structured evidence checks, preserves uncertainty for human review, and aggregates scores deterministically. The project is framed as reusable research infrastructure and an AI-measurement experiment. It is not presented as an automated replacement for peer review.

The current V16 audit covers 149 source PDFs: 146 unique papers after exact deduplication, 144 with complete V16 reports, and two blocked during PDF parsing before scoring. These records demonstrate a working prototype at meaningful scale, but not validity against expert judgments. See the [audit](04-reproducibility/original-149-v16-audit.md) and [title-free per-paper outcomes](04-reproducibility/original-149-v16-results.csv).

**Version boundary:** the repository currently contains a V17 draft scoring profile; the V16 corpus results are historical and were not generated with that checked-in profile. The exact V16 rule/configuration snapshot is not included here.

## Fit with the call

The project most closely aligns with **reusable research infrastructure**, **new forms of measurement**, and **critical/diagnostic boundary-setting**. Its potential contribution is to make AI-assisted research assessment more inspectable and empirically testable. The proposal must still specify the scientific question, expert-validation design, and how the resulting evidence would change organizational research.

## Stage 1 package checklist

| Required item | Status | Location |
|---|---|---|
| Short proposal (3–5 pages / 1,500–2,000 words plus references) | **MISSING — author draft required** | [`01-proposal/`](01-proposal/) |
| Technical appendix or supplement | Draft present; requires author and technical review | [`02-technical-appendix/`](02-technical-appendix/) |
| Prototype and repository | Included; local Web/API interface intentionally excluded | [`03-research-artifact/`](03-research-artifact/) and repository root |
| Reproducibility evidence | Historical runs documented; raw traces and corpus withheld pending rights review; independent reproduction package incomplete | [`04-reproducibility/`](04-reproducibility/) |
| Ethics, data rights, and model access | **MISSING — author statements required** | [`05-ethics-and-access/`](05-ethics-and-access/) |

## Reader path

1. Start with the repository [overview](../README.md) for the research question, workflow, evidence, and limits.
2. Read the [technical appendix](02-technical-appendix/README.md) to inspect the prototype design.
3. Review the [149-PDF V16 audit](04-reproducibility/original-149-v16-audit.md) and [mechanism experiment summary](04-reproducibility/experiment-evidence.md).
4. Check the [reproducibility gaps](04-reproducibility/README.md) and [ethics/access checklist](05-ethics-and-access/README.md).

## Evidence boundary

The reported runs are exploratory and span historical versions. No expert-labeled benchmark, independent validity study, or proof that the 60/40 pool ratio is optimal is included. Two unique papers could not be scored because PDF parsing was blocked; no scores were imputed. Full-text reports and source PDFs are not distributed here.
