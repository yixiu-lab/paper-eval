# PaperEval

### An evidence-grounded workflow for AI-assisted research assessment

## Abstract

PaperEval investigates how artificial intelligence can support the assessment of management research through explicit evaluation criteria, traceable evidence, and inspectable judgments. Rather than relying on a single summary or overall score, the system translates a structured evaluation framework into rule-level assessments. For each paper, it links judgments to relevant passages, records the evidence and rationale behind them, and uses deterministic software to validate evidence references and aggregate scores into a report for researcher review.

The empirical program examines how assessment outcomes vary across models and repeated runs, how reviewer configurations and workflow components shape results, how the system handles paper and external evidence, and which conditions warrant human review. By connecting criteria, evidence, judgments, validation, and scores in a traceable workflow, PaperEval provides a basis for examining both assessment outcomes and the process through which they are produced.

The project focuses on management and AI-HRM research. It treats paper assessment as a measurement workflow: a score should be traceable to the criteria applied, the evidence considered, the judgment made for each criterion, and the software that validates and aggregates those judgments.

## Research motivation

AI systems can produce fluent paper summaries and review-like feedback, but a polished narrative or a single score does not reveal how the system reached its conclusions. PaperEval makes that process inspectable at the rule level. It links judgments to passages in the paper, records page and section context, preserves uncertainty and review indicators, and exposes the scoring transformations used to produce dimension and total scores.

This design allows the project to study both assessment outcomes and the process that produces them. Researchers can examine where the evidence came from, which criterion it supports, how the AI interpreted it, and how the final score was calculated.

## Research questions

The empirical program examines four connected questions:

1. **Score variation:** How do paper scores and rule-level judgments vary across AI models and repeated runs?
2. **Workflow structure:** How do reviewer organization and system components affect judgments and aggregate scores?
3. **Evidence handling:** How does the system select paper evidence, incorporate prior work when retrieval is enabled, and respond to instruction-like content in evidence passages?
4. **Validation and review:** Which output or evidence conditions can the Harness detect, and which cases should be surfaced for human inspection?

## Core contribution

PaperEval provides an executable, evidence-grounded assessment workflow with four linked components:

- **Explicit assessment criteria:** an operational profile containing 34 rules across six dimensions, with rule definitions, applicability, evidence requirements, judgment anchors, weights, and score mappings.
- **Rule-specific evidence:** PDF text is organized into page- and section-linked evidence records. The system builds a structured paper profile and routes relevant passages to the corresponding rule-level review tasks.
- **Structured AI judgments:** Reviewer agents return a judgment tier, rationale, confidence, and evidence references for assigned criteria. Optional claim-led retrieval can add traceable prior-work evidence for comparison-dependent judgments.
- **Deterministic validation and scoring:** the Harness checks judgment structure and evidence references, flags items for review, handles selected uncertain items through a second evidence pass, and applies the declared mappings and weights to calculate scores.

The result is a report and run record that connect criteria, evidence, judgments, validation outcomes, and scores. Model calls interpret the paper and make bounded judgments; program code validates those outputs and performs score aggregation.

## Empirical scope

The current manuscript reports completed scores for **144 papers** and focused studies of five-model variation, reviewer structure, repeated-run stability, evidence-boundary instructions, and Harness fault handling. Each study uses its own sample and comparison. The results and supporting records are summarized in the [experimental evidence section](#experimental-evidence) and organized by study in the [experiment README](03-experiments/README.md).

## System and technical approach

PaperEval turns a research-paper PDF into an inspectable sequence of evidence-backed judgments and scores:

- **Parse and profile:** extract paper text and structure, create stable evidence identifiers linked to pages and sections, and build a structured profile of the research question, claims, theory, methods, findings, and limitations. Parse failures are recorded before scoring.
- **Retrieve prior work when enabled:** extract claims and use configured scholarly metadata services to retrieve related work. Retrieved records are linked to their source and can provide external evidence for judgments about research gaps and contributions.
- **Assemble rule-specific evidence:** route bounded evidence packages to the scoring criteria across six dimensions: research-question positioning, theoretical contribution, theoretical rigor, research design, results and evidence, and discussion and boundaries.
- **Review and validate:** specialist Reviewer agents assess assigned rules and return structured judgments, rationales, confidence, and evidence references. The Harness checks schemas and evidence IDs, validates external-evidence conditions, retries selected uncertain items, and flags cases for human inspection. A unified-Reviewer configuration is also supported.
- **Aggregate and report:** deterministic scoring applies the selected rule profile to applicable judgments and records completeness and uncertainty. Reports can include dimension and total scores, evidence references, review flags, parse diagnostics, run manifests, and trace records. Optional reliability analysis compares saved runs without changing the current score.

See the [system and technical approach](02-system/README.md) for the architecture, workflow, code map, and command-line instructions.

## Experimental evidence

The current experiment manuscript presents the studies in the sequence below. Each study has its own sample and comparison; samples overlap and must not be added to estimate a project-wide count of unique papers. The [experiment README](03-experiments/README.md) provides the detailed ordered index.

1. **Corpus and journal/source score profile.** PaperEval completed scoring for **144 papers**. Two additional unique papers were blocked before scoring by PDF parsing. Across the 144 completed reports, the mean total score was **76.46** (SD 7.64), the median was **76.19**, and the range was **49.63–95.78**. Six-dimension summaries and title-free paper outcomes are in the [audit record](04-data/corpus-audit.md) and [outcome table](04-data/paper-outcomes.csv).

2. **Exploratory model comparison.** The current manuscript compares five models on three shared papers with two recorded runs per model and paper. Model means range from **77.91 to 89.99**; mean absolute differences between runs range from **0.61 to 3.83** points. Because evidence was re-extracted and retrieval rerun, the comparison is descriptive and combines model and evidence variation. See the [five-model comparison](03-experiments/cross-model-results.md).

3. **Reviewer structure comparison.** Four conditions were compared on the same five papers: full PaperEval, one unified Reviewer, six rule-level Reviewers without Harness processing and formal two-pool aggregation, and direct model scoring. Their mean scores were **79.50**, **74.60**, **57.93**, and **74.00**, respectively. These are descriptive outputs from different judgment and aggregation procedures, so they are not directly comparable as accuracy estimates. See the [four-condition comparison](03-experiments/module-ablation.md).

4. **Harness validation.** In a deterministic fault-injection test, the Harness detected **180/180** injected machine-checkable faults across nine classes, automatically blocked or transformed 160, and routed 20 reason–judgment conflicts for review. All **40 clean controls** passed without false positives. See [Harness fault-injection results](03-experiments/robustness-experiments.md#deterministic-harness-fault-injection).

5. **Adversarial instruction injection.** A paired prompt study used evidence from three papers and 108 attack cases per condition. Adding an evidence-boundary instruction reduced attacks reaching their target from **4/108 (3.7%) to 0/108**, rating changes from 14/108 to 7/108, and mean absolute ordinal shift from **0.250 to 0.074**. The results describe this test set and do not measure full-paper scoring accuracy. See the [prompt-injection comparison](03-experiments/robustness-experiments.md#evidence-level-prompt-injection-comparison).

6. **Repeated-run stability.** Twenty papers were scored independently twice, yielding **40 complete reports**. The paired mean absolute score difference was **2.382** points, the maximum difference was **8.388**, and **17/20 papers (85.0%)** retained the same quality grade. See the [two-run results](03-experiments/robustness-experiments.md#two-run-repeatability-in-the-current-experiment-manuscript).

Across the broader experimental program, the same paper corpus is revisited through model comparisons, repeated runs, workflow and scoring analyses, evidence-handling studies, retrieval analyses, and trace-based diagnostics. These analyses reuse papers and generate multiple run, judgment, and evidence records; they are linked to the completed-score corpus rather than added as separate paper totals. The main corpus comprises **144 papers with completed scores**. Study-specific designs and results are available in the [experiment catalog](03-experiments/experiment-catalog.md), [historical experiment evidence](03-experiments/experiment-evidence.md), and [retrieval record](03-experiments/retrieval-evaluation.md).

## Reading path

1. [Project overview](01-proposal/README.md)
2. [System architecture and implementation](02-system/README.md)
3. [Experiment catalog and study summaries](03-experiments/README.md)
4. [Data and experimental materials](04-data/README.md) — audit data, sample index, and redistributable sample PDFs.

## Run the command-line workflow

The implementation and example configurations are in [`02-system/`](02-system/). Run commands from that directory:

```bash
python -m venv .venv
pip install -r requirements.txt
python -m paper_eval.cli --help
python -m paper_eval.cli validate-rules --rules rules/rules_v1.yaml
```

For single-paper evaluation and rule validation, use the commands in the [system README](02-system/README.md). Model-backed runs use a configured provider; record the provider/model, code revision, criteria used, retrieval settings, and run status for each evaluation.
