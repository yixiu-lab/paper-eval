# PaperEval

### An evidence-grounded workflow for AI-assisted research assessment

## Research background

Assessment of management research draws on human judgment about the importance of a research question, theoretical contribution, rigor, research design, evidence, and implications. These judgments are made from different parts of a paper and need to be connected to the evidence that supports them. PaperEval studies how to make this assessment process explicit and inspectable: criteria guide the review, paper passages support rule-level judgments, and deterministic software calculates the reported scores.

The project uses one calibrated scoring-rule set with 34 rules across six dimensions. The rules translate paper-quality criteria into specific checks, evidence requirements, judgment anchors, and score mappings. PaperEval applies those rules through an AI-assisted workflow and preserves the evidence and reasoning behind each reported judgment.

## Research questions

1. **Human assessment:** How can the multidimensional judgments involved in human paper review be represented as explicit, evidence-linked criteria?
2. **AI scoring:** How can an AI workflow apply these criteria to paper evidence and produce traceable rule-level judgments and scores?
3. **Journal and source comparison:** How are the resulting score profiles distributed across journals and publication sources in the evaluated corpus?

## Technical approach

PaperEval turns a paper PDF into rule-level judgments and a score report. The method follows the sequence below; each heading links to the matching section in the full [Technical Approach](01-system/README.md).

1. [System overview](01-system/README.md#1-system-overview). The command-line Harness coordinates PDF processing, evidence routing, reviewer agents, validation, score aggregation, and report generation.
2. [Rule profile](01-system/README.md#2-rule-profile). A 34-rule profile across six dimensions specifies criteria, evidence requirements, judgment anchors, applicability, weights, and score mappings.
3. [PDF parsing and paper representation](01-system/README.md#3-pdf-parsing-and-paper-representation). The parser extracts paper structure and creates page- and section-linked evidence records and a structured paper profile.
4. [Rule-specific evidence packages](01-system/README.md#4-rule-specific-evidence-packages). The evidence router selects bounded passages and context for each rule and includes stable evidence identifiers for traceability.
5. [Claim-led external grounding](01-system/README.md#5-claim-led-external-grounding). When enabled, claims from the paper guide prior-work retrieval; selected records are linked to their source and supplied as external evidence for comparison-dependent rules.
6. [Specialist review and judgment protocol](01-system/README.md#6-specialist-review-and-judgment-protocol). Six specialist Reviewers assess assigned dimensions and return structured rule judgments, rationales, confidence, and evidence references; a unified-Reviewer configuration is also available.
7. [Harness: validation, gating, and aggregation](01-system/README.md#7-harness-validation-gating-and-aggregation). Deterministic checks validate schemas and evidence references, flag selected cases for review, retry uncertain items, and calculate dimension and total scores.

## Experimental results

The experimental results below follow the manuscript sequence. The [experiment README](02-experiments/README.md) links each section to its study record, while the [experiment catalog](02-experiments/experiment-catalog.md) indexes the broader study program.

### 3.1 Paper total score results

The evaluation contains completed score records for **200 papers**. Archived rule-level judgments were aggregated using the integer weights; the model was not rerun for this recalculation. The mean total score is **77** (SD **9**), the median is **76**, and the range is **50–96**. The [data overview](03-data/README.md#evaluation-results) and [outcome table](03-data/paper-outcomes.csv) provide dimension summaries and paper-level scores.

### 3.2 Journal score aggregation results

The paper index records **73 named journal or source groups** for 194 papers; six papers have no source field and are shown separately in Table 5. The table summarizes every group by descending mean score, with sample size, mean and standard deviation, median, and range. The corpus includes **29 papers across eight journals** on the 2026 Financial Times 50 list, including 15 in *Management Science* and four in *Academy of Management Journal*. See the [journal aggregation section](02-experiments/README.md#2-journal-score-aggregation-results) and the [paper index](03-data/paper-index.csv).

### 3.3 Cross-model comparison

Five models were compared on three shared papers, with two recorded runs for each paper–model pair. Model means range from **77.91 to 89.99**; mean absolute differences between runs range from **0.61 to 3.83** points. Public GPQA scores do not align monotonically with these run differences, so the capability–stability relationship remains a testable hypothesis. See the [five-model comparison](02-experiments/cross-model-results.md).

### 3.4 Reviewer structure comparison

Four conditions were compared on five papers: full PaperEval, one unified Reviewer, six rule-level Reviewers without Harness processing and formal two-pool aggregation, and direct model scoring. Their mean scores were **79.50**, **74.60**, **57.93**, and **74.00**, respectively. See the [comparison and module ablations](02-experiments/module-ablation.md).

### 3.5 Harness validation and fault injection

In a deterministic fault-injection test, the Harness detected **180/180** injected machine-checkable faults across nine classes; all **40 clean controls** passed. See [Harness fault-injection results](02-experiments/robustness-experiments.md#deterministic-harness-fault-injection).

### 3.6 Evidence-boundary prompt injection

In a paired prompt study with three papers and 108 attack cases per condition, adding an evidence-boundary instruction reduced attacks reaching their target from **4/108** to **0/108**. See the [prompt-injection comparison](02-experiments/robustness-experiments.md#evidence-level-prompt-injection-comparison).

### 3.7 Independent repeated-run stability

Twenty papers were scored independently twice. The paired mean absolute score difference was **2.382** points, and **17/20 papers** retained the same quality grade. See the [repeatability results](02-experiments/robustness-experiments.md#two-run-repeatability-in-the-current-experiment-manuscript).

### 3.8 Experiment summary and paper-level outcomes

Focused comparisons reuse papers from the study corpus; their repeated records are not added to the corpus count. The [experiment catalog](02-experiments/experiment-catalog.md) links study designs, samples, and detailed results. Paper-level scores are available in the [outcome table](03-data/paper-outcomes.csv).

## Data and materials

The data section contains the outcome table, the paper index covering all 200 evaluated papers, and sample PDFs with explicit redistribution licenses. The [data overview](03-data/README.md) describes the released materials and score summaries. Reusable experimental code is indexed in the [experiment README](02-experiments/README.md#experiment-code).

## Repository structure

1. [System and technical approach](01-system/README.md)
2. [Experiments and results](02-experiments/README.md)
3. [Data and research materials](03-data/README.md)
4. [Ethics and data access](04-ethics-access/README.md)

## Run the command-line workflow

Run commands from [`01-system/`](01-system/):

```bash
python -m venv .venv
pip install -r requirements.txt
python -m paper_eval.cli --help
python -m paper_eval.cli validate-rules --rules rules/rules_v1.yaml
```

For single-paper evaluation, use the commands in the [system documentation](01-system/README.md). Model-backed runs use a configured provider; record the provider and model, code revision, criteria, retrieval settings, and run status for each evaluation.
