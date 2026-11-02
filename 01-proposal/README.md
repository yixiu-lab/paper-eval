# Research Rationale and Study Design

This page sets out the research logic behind PaperEval: why AI-assisted paper assessment needs to be examined as a measurement process, what the study investigates, and how the empirical work is organized. For the project summary, see the [repository overview](../README.md). For implementation details and results, use the linked system and experiment materials below.

## Research problem

AI systems can generate fluent summaries and review-like judgments, but fluency alone does not show whether an assessment is grounded in the paper, applied consistently across criteria, or aggregated correctly. A single overall score also obscures how evidence and rule-level judgments produce the result.

PaperEval treats assessment as a sequence that can be inspected: criteria are applied to evidence from the paper; judgments are recorded against those criteria; software checks evidence references and scoring transformations; and the resulting report remains open to researcher review. This framing makes both the score and the process that produced it available for empirical analysis.

## Research objectives

The study has three objectives:

1. **Operationalize structured assessment.** Implement a criterion-level workflow that connects paper passages, judgments, validation, and aggregate scores.
2. **Examine outcome variation.** Compare results across models, repeated runs, and reviewer/workflow configurations.
3. **Characterize evidence and review behavior.** Study evidence selection, optional prior-work retrieval, robustness to instruction-like evidence, and cases routed for human inspection.

## Research questions

1. How much do paper-level scores and criterion-level judgments vary across models and repeated runs?
2. How do reviewer organization and workflow components affect judgments and aggregate scores?
3. How does the workflow select and use paper evidence and retrieved prior work?
4. Which output or evidence conditions can be detected automatically, and which should be referred for human review?

## Empirical strategy

The experiment covers **200 papers**, all with completed scores under the same calibrated criteria and scoring model. Analyses examine model behavior, workflow structure, repeatability, evidence handling, retrieval, and Harness validation. The [data overview](../04-data/README.md#evaluation-results) defines the outcomes and released materials.

The experiment records follow the sequence and evidence available in the current manuscript. Each study page describes its comparison and links to the corresponding results:

- [Corpus and journal/source score profile](../03-experiments/README.md#1-corpus-and-journalsource-score-profile)
- [Exploratory model comparison](../03-experiments/README.md#2-exploratory-comparison-across-models)
- [Reviewer structure comparison](../03-experiments/README.md#3-reviewer-structure-comparison)
- [Harness validation and fault injection](../03-experiments/README.md#4-harness-validation-and-fault-injection)
- [Adversarial instruction injection](../03-experiments/README.md#5-adversarial-instruction-injection)
- [Repeated-run stability](../03-experiments/README.md#6-repeated-run-stability)
- [Full experiment catalog](../03-experiments/experiment-catalog.md)

## Method in brief

The study uses an evidence-grounded assessment workflow. The system parses each paper into page- and section-linked evidence, assembles rule-specific evidence packages, and generates structured judgments. The Harness validates judgment structure and evidence references, flags selected cases for review, and calculates scores from the configured criteria. This page states the study design; the [system documentation](../02-system/README.md) describes the architecture, criteria, and implementation.

## Intended contribution

PaperEval contributes an inspectable way to study AI-assisted assessment as a measurement workflow. It supports analysis of how model behavior, workflow choices, evidence handling, and deterministic validation relate to assessment outputs. The project reports these findings as observed in the available corpus and experimental records; it does not treat journal standing or a model-generated score as a ground-truth measure of paper quality.

## Materials

- [System architecture and implementation](../02-system/README.md)
- [Experiment sequence and results](../03-experiments/README.md)
- [Data and released materials](../04-data/README.md)
- [Ethics and data access](../05-ethics-access/README.md)
