# PaperEval

### An evidence-grounded workflow for studying AI-assisted research assessment

PaperEval examines AI-based assessment of management research through explicit criteria, traceable evidence, and inspectable judgments. Its studies examine score variation across models and repeated runs, the effect of workflow components, evidence-handling behavior, and conditions that require human review.

PaperEval operationalizes structured evaluation criteria as rule-level judgments. Each judgment is connected to paper evidence, checked by deterministic software, and aggregated into a report that researchers can inspect.

## System workflow

1. **Parse and represent the paper.** PaperEval extracts text and structure from a research-paper PDF, assigns page- and section-linked evidence identifiers, and builds a structured profile of the paper's question, claims, theory, methods, findings, and limitations. Severe parsing failures are recorded before scoring. See [PDF parsing and paper representation](../02-system/technical-appendix.md#3-pdf-parsing-and-paper-representation) and [`pdf_parser.py`](../02-system/paper_eval/pdf_parser.py).


2. **Extract claims and retrieve prior work.** When enabled, claim-led retrieval uses the paper's research question, gap, theory, and contribution claims to find related work from configured scholarly metadata services. Retrieved records are normalized, linked to their sources, and used as external evidence for comparison-dependent judgments. See [claim-led external grounding](../02-system/technical-appendix.md#5-claim-led-external-grounding) and the [retrieval study](../03-experiments/retrieval-evaluation.md).


3. **Route evidence to six evaluation dimensions.** The system assembles rule-specific evidence for research-question positioning, theoretical contribution, theoretical rigor, research design, results and evidence, and discussion and boundaries. Evidence packages retain the relevant passage, page, section, and evidence identifier. See [rule-specific evidence packages](../02-system/technical-appendix.md#4-rule-specific-evidence-packages) and the [scoring criteria](../02-system/rules/rules_v1.yaml).


4. **Generate structured reviewer judgments.** Specialist reviewers assess assigned rules and return a judgment, rationale, confidence, and cited evidence identifiers. The runtime also supports a unified-reviewer configuration. See [specialist review and judgment protocol](../02-system/technical-appendix.md#6-specialist-review-and-judgment-protocol) and the [system overview](../02-system/README.md).


5. **Validate and aggregate.** The Harness checks judgment schemas and evidence references, validates external evidence where applicable, retries selected uncertain items, and flags issues for human inspection. Deterministic scoring then aggregates applicable rule judgments and records completeness and uncertainty. See [Harness validation and aggregation](../02-system/technical-appendix.md#7-harness-validation-gating-and-aggregation), [`evidence_gate.py`](../02-system/paper_eval/evidence_gate.py), and [`scorer.py`](../02-system/paper_eval/scorer.py).


6. **Produce inspectable outputs.** Each run can produce JSON, Markdown, and HTML reports, a run manifest, and a trace bundle. Reports connect scores to criteria, evidence, rationales, validation results, and review flags; optional reliability diagnostics summarize variation across saved runs. See [reports and run records](../02-system/technical-appendix.md#8-reports-run-records-and-reliability-analysis).


The main research artifact is the [Python implementation and command-line workflow](../02-system/README.md). The local Web/API prototype is outside this repository's research-artifact scope.


## Experimental evidence

The current experiment manuscript reports results in the sequence below. Each experiment has its own sample and comparison; samples overlap and should not be added to infer a project-wide number of unique papers. The [experiment index](../03-experiments/README.md) follows the same sequence and links to each detailed record.



The audit began with **149 source PDFs**, which yielded **146 unique papers** after exact deduplication. It produced **144 complete reports**; two papers were blocked before scoring by PDF parsing. The completed reports had a mean total score of **76.46** (SD 7.64), a median of **76.19**, and a range of **49.63–95.78**. These are descriptive scores for the audited corpus and configuration. The manuscript also reports A–F dimension summaries and journal/source composition. See the [audit record](../04-data/corpus-audit.md) and [title-free outcomes](../04-data/paper-outcomes.csv).



The manuscript compares five models on three shared papers, with two recorded runs per model and paper. Model means range from **77.91 to 89.99**, and average absolute differences between repeated runs range from **0.61 to 3.83** points. Evidence and retrieval were regenerated between runs, so the observed differences combine model and evidence variation. See the [five-model summary](../03-experiments/cross-model-results.md).



Four evaluation structures were compared on the same **five papers**: full PaperEval, one unified Reviewer, six rule-level Reviewers without Harness processing and formal two-pool aggregation, and direct model scoring. Their mean scores were **79.50**, **74.60**, **57.93**, and **74.00**, respectively. These outputs are descriptive because the conditions use different judgment and aggregation procedures. See the [four-condition comparison and module ablations](../03-experiments/module-ablation.md).



The deterministic fault-injection study covered **180 injected faults** across nine machine-checkable categories and **40 clean controls**. The Harness detected all injected faults, automatically blocked or transformed 160, routed 20 reason–judgment conflicts for review, and passed all clean controls. See [fault-injection results](../03-experiments/robustness-experiments.md#deterministic-harness-fault-injection).



The paired prompt study used evidence from **three papers** and 108 attack cases per condition. Adding an evidence-boundary instruction reduced target-reaching attacks from **4/108 (3.7%) to 0/108**, rating changes from 14/108 to 7/108, and mean absolute ordinal shift from **0.250 to 0.074**. See the [prompt-injection comparison](../03-experiments/robustness-experiments.md#evidence-level-prompt-injection-comparison).



Twenty papers were scored independently twice, producing **40 complete reports**. The paired mean absolute score difference was **2.382** points, the maximum difference was **8.388**, and **17/20 papers (85.0%)** retained the same quality grade. See [two-run repeatability results](../03-experiments/robustness-experiments.md#two-run-repeatability-in-the-current-experiment-manuscript).



Additional records include a **10-paper** API batch, two journal-gradient samples with **27 papers** combined, a **17-paper** trace database, a separate reliability diagnostic with 57 historical runs across 20 papers, and a seven-paper/seven-claim retrieval setup with 366 citation-linked candidate mappings. Retrieval candidates were machine-prelabelled from citations, and the archived aggregate has a claim-count mismatch; no overall retrieval-performance estimate is reported. Per-study paper counts are itemized in the [study sample index](../04-data/study-samples.md). See the [historical experiment index](../03-experiments/experiment-evidence.md), [experiment catalog](../03-experiments/experiment-catalog.md), and [retrieval record](../03-experiments/retrieval-evaluation.md).



- [System architecture and technical method](../02-system/technical-appendix.md) — system layers, evidence flow, validation, and scoring.
- [Experiment catalog](../03-experiments/experiment-catalog.md) — study families, sample sizes, and summary outcomes.
- [Data and experimental materials](../04-data/README.md) — corpus audit, outcome table, and study samples.
- [Ethics and data handling](../05-ethics-access/README.md) — study materials, data flows, and human responsibility.
