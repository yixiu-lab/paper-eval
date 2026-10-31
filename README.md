# PaperEval

### An evidence-grounded workflow for studying AI-assisted research assessment

PaperEval studies how scholarly research can be assessed through explicit criteria, traceable evidence, and inspectable judgments. The project focuses initially on management and AI-HRM research and treats paper evaluation as a measurement process that can be examined at the rule, evidence, and score levels.

## Research problem

AI-assisted paper assessment can produce fluent summaries and scores, but a score alone does not show whether the assessment used relevant evidence, applied criteria consistently, or handled uncertainty appropriately. PaperEval turns assessment criteria into rule-level judgments linked to passages from the paper and, where relevant, external literature. This makes the evaluation process available for inspection and empirical study.

## Research question and contribution

The project asks when evidence-grounded AI judgments align with expert assessments, how stable they are across runs and evidence settings, and which paper types or conditions require additional review.

PaperEval contributes a research capability for studying AI-assisted assessment: expert-constructed evaluation criteria, rule-specific evidence, structured reviewer judgments, deterministic score aggregation, and reports that preserve the relationship between criteria, evidence, judgments, and scores. The system is a prototype; expert agreement, reliability, and scope are central evaluation questions.

## System and technical approach

For each paper, PaperEval parses the PDF into page- and section-linked evidence, builds a structured paper profile, and routes selected evidence to rule-level review tasks. When enabled, claim-led retrieval adds external prior-work evidence. Reviewer agents assess six dimensions, while the Harness validates judgment structure and evidence references, flags items for review, and computes scores using the selected rule profile.

The expert-constructed criteria cover six dimensions: research-question positioning, theoretical contribution, theoretical rigor, research design, results and evidence, and discussion and boundaries. Reports include dimension and total scores, completeness, evidence references, review flags, and run records.

See the [technical architecture](02-system/technical-appendix.md) and [system package](02-system/README.md).

## Experimental evidence

The corpus audit began with **149 source PDFs**. Exact deduplication yielded **146 unique papers**; the workflow produced **144 complete reports**, while two papers were blocked before scoring by PDF parsing. Across the 144 completed reports, the mean score was 76.46 (SD 7.64), the median was 76.19, and the range was 49.63–95.78.

Other scoring studies used additional samples: a **10-paper** API batch; **27 papers** in two journal-gradient samples; a **20-paper** reliability analysis based on 57 historical runs; and a **17-paper** trace database. Further focused studies include a three-paper model comparison (12 valid runs), a five-paper architecture comparison, three-paper robustness and evidence-budget studies, and a seven-paper retrieval study. These samples overlap, so their counts should not be added to the 146-paper corpus audit as a project-wide total. Study-level results and designs are reported in the [experiment catalog](03-experiments/experiment-catalog.md).

The Harness fault-injection study tested 180 faulty outputs and 40 clean controls. Detailed outcomes are reported study by study.

See the [experiment catalog](03-experiments/experiment-catalog.md) for study scope and the [reproducibility folder](04-reproducibility/README.md) for the corpus audit and title-free outcomes.

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
