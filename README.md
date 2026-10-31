# PaperEval

### An evidence-grounded workflow for studying AI-assisted research assessment

PaperEval studies how scholarly research can be assessed through explicit criteria, traceable evidence, and inspectable judgments. The project focuses initially on management and AI-HRM research and treats paper evaluation as a measurement process that can be examined at the rule, evidence, and score levels.

## Research problem

AI-assisted paper assessment can produce fluent summaries and scores, but a score alone does not show whether the assessment used relevant evidence, applied criteria consistently, or handled uncertainty appropriately. PaperEval turns assessment criteria into rule-level judgments linked to passages from the paper and, where relevant, external literature. This makes the evaluation process available for inspection and empirical study.

## Research question and contribution

The project asks when evidence-grounded AI judgments align with expert assessments, how stable they are across runs and evidence settings, and which paper types or conditions require additional review.

PaperEval contributes a research capability for studying AI-assisted assessment: expert-constructed evaluation criteria, rule-specific evidence, structured reviewer judgments, deterministic score aggregation, and reports that preserve the relationship between criteria, evidence, judgments, and scores. The system is a prototype; expert agreement, reliability, and scope are central evaluation questions.

## System and technical approach

PaperEval turns a research-paper PDF into an inspectable sequence of evidence-backed judgments and scores:

- **Parse and profile:** extract paper text and structure, create stable evidence identifiers linked to pages and sections, and build a structured profile of the research question, claims, theory, methods, findings, and limitations. Parse failures are recorded before scoring.
- **Retrieve prior work when enabled:** extract claims and use configured scholarly metadata services to retrieve related work. Retrieved records are linked to their source and can provide external evidence for judgments about research gaps and contributions.
- **Assemble rule-specific evidence:** route bounded evidence packages to expert-constructed criteria across six dimensions: research-question positioning, theoretical contribution, theoretical rigor, research design, results and evidence, and discussion and boundaries.
- **Review and validate:** specialist Reviewer agents assess assigned rules and return structured judgments, rationales, confidence, and evidence references. The Harness checks schemas and evidence IDs, validates external-evidence conditions, retries selected uncertain items, and flags cases for human inspection. A unified-Reviewer configuration is also supported.
- **Aggregate and report:** deterministic scoring applies the selected rule profile to applicable judgments and records completeness and uncertainty. Reports can include dimension and total scores, evidence references, review flags, parse diagnostics, run manifests, and trace records. Optional reliability analysis compares saved runs without changing the current score.

See the [technical architecture](02-system/technical-appendix.md), [reviewer walkthrough](02-system/reviewer-guide.md), and [system package and CLI](02-system/README.md).

## Experimental evidence

The current experiment manuscript presents the studies in the sequence below. Each study has its own sample and comparison; samples overlap and must not be added to estimate a project-wide count of unique papers. The [experiment README](03-experiments/README.md) provides the detailed ordered index.

1. **Corpus and journal/source score profile.** The audit began with **149 source PDFs**, yielding **146 unique papers** after exact deduplication. It produced **144 complete reports**; two papers were blocked before scoring by PDF parsing. The mean score across completed reports was **76.46** (SD 7.64), the median was **76.19**, and the range was **49.63–95.78**. Six-dimension summaries and title-free paper outcomes are in the [audit record](04-reproducibility/original-149-v16-audit.md) and [outcome table](04-reproducibility/original-149-v16-results.csv).

2. **Exploratory model comparison.** The current manuscript compares five models on three shared papers with two recorded runs per model and paper. Model means range from **77.91 to 89.99**; mean absolute differences between runs range from **0.61 to 3.83** points. Because evidence was re-extracted and retrieval rerun, the comparison is descriptive and combines model and evidence variation. See the [five-model comparison](03-experiments/cross-model-v16.md).

3. **Reviewer structure comparison.** Four conditions were compared on the same five papers: full PaperEval, one unified Reviewer, six rule-level Reviewers without Harness processing and formal two-pool aggregation, and direct model scoring. Their mean scores were **79.50**, **74.60**, **57.93**, and **74.00**, respectively. These are descriptive outputs from different judgment and aggregation procedures, not expert-accuracy comparisons. See the [four-condition comparison](03-experiments/module-ablation.md).

4. **Harness validation.** In a deterministic fault-injection test, the Harness detected **180/180** injected machine-checkable faults across nine classes, automatically blocked or transformed 160, and routed 20 reason–judgment conflicts for review. All **40 clean controls** passed without false positives. See [Harness fault-injection results](03-experiments/robustness-experiments.md#deterministic-harness-fault-injection).

5. **Adversarial instruction injection.** A paired prompt study used evidence from three papers and 108 attack cases per condition. Adding an evidence-boundary instruction reduced attacks reaching their target from **4/108 (3.7%) to 0/108**, rating changes from 14/108 to 7/108, and mean absolute ordinal shift from **0.250 to 0.074**. The results apply to this test set and do not measure expert accuracy. See the [prompt-injection comparison](03-experiments/robustness-experiments.md#evidence-level-prompt-injection-comparison).

6. **Repeated-run stability.** Twenty papers were scored independently twice, yielding **40 complete reports**. The paired mean absolute score difference was **2.382** points, the maximum difference was **8.388**, and **17/20 papers (85.0%)** retained the same quality grade. See the [two-run results](03-experiments/robustness-experiments.md#two-run-repeatability-in-the-current-experiment-manuscript).

The archive also contains a 10-paper API batch, two journal-gradient samples covering 27 papers, a 17-paper trace database (72 runs), a separate reliability diagnostic using 57 historical runs across 20 papers, and a retrieval study with seven target papers, seven claims, and 366 citation-linked candidate mappings. Retrieval candidates were machine-prelabelled from citations, and the archived aggregate has a claim-count mismatch; no overall retrieval-performance estimate is reported. Details and sample definitions are in the [experiment catalog](03-experiments/experiment-catalog.md), [historical experiment evidence](03-experiments/experiment-evidence.md), and [retrieval record](03-experiments/retrieval-evaluation.md).

## Reading path

1. [Project overview](01-proposal/README.md)
2. [System architecture and implementation](02-system/README.md)
3. [Experiment catalog and study summaries](03-experiments/README.md)
4. [Corpus audit and reproduction materials](04-reproducibility/README.md)

## Run the command-line workflow

The implementation and example configurations are in [`02-system/`](02-system/). Run commands from that directory:

```bash
python -m venv .venv
pip install -r requirements.txt
python -m paper_eval.cli --help
python -m paper_eval.cli validate-rules --rules <path-to-expert-criteria.yaml>
```

For a single-paper evaluation, see the [system baseline](02-system/system-baseline.md) and [reviewer guide](02-system/reviewer-guide.md). Model-backed runs use a configured provider; record the provider/model, code revision, criteria used, retrieval settings, and run status for each evaluation.
