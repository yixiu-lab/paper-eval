# Technical Appendix

## System purpose

PaperEval turns assessment of management research into an inspectable workflow. Rule-level judgments link to selected evidence; uncertainty is recorded for researcher review; a versioned profile aggregates the judgments. The research program evaluates agreement with experts, score stability, evidence support, and performance across paper types.

## Workflow architecture

The current pipeline parses a paper PDF, builds a structured profile, extracts contribution claims, retrieves related literature, assembles evidence packs for six dimensions, produces rule-level judgments, checks evidence and citation conditions, aggregates scores, and generates a traceable report with reliability diagnostics.

The six dimensions are research-question positioning, theoretical contribution, theoretical rigor, research design, results/evidence, and discussion/boundary conditions. An auxiliary novelty comparison is separate from the six scoring dimensions and does not directly contribute to the total.

## V16 scoring profile

The current draft profile uses a 60-point baseline pool and a 40-point bonus pool. Baseline rules represent adequacy; bonus rules represent evidence beyond adequacy. Uncertain judgments are excluded from the available-score denominator and reported with completeness. The profile contains 34 rules and is available at [`../../评价规则库/rules_v1.yaml`](../../评价规则库/rules_v1.yaml).

V16 is the rule snapshot used in the 149-PDF audit. The next calibration study will compare pre-specified pool ratios (50/50, 60/40, and 70/30) on a fixed sample and evaluate their relationship to expert judgments.

## Evidence assembly and retrieval

Evidence packs are assembled through deterministic selection and optional retrieval/reranking components. The workflow can send selected paper evidence blocks to the configured language-model endpoint. External literature retrieval can use Crossref, OpenAlex, and Semantic Scholar; requests can contain paper titles and claim-derived terms. Run manifests will capture provider/model, access date, retrieval settings, cache state, and data-handling terms.

## Runtime and entry points

- Single-paper CLI: `python -m paper_eval.cli evaluate` with paper, rule-profile, and output arguments.
- Runtime dependencies: [`../../requirements.txt`](../../requirements.txt).
- Example configurations: [`../../configs/`](../../configs/).
- System overview: [`../../system-baseline.md`](../../system-baseline.md).

## V16 audit and reproduction pathway

The audit covers 149 source PDFs, 146 unique papers after exact deduplication, 144 complete V16 reports, and two papers blocked before scoring by PDF parsing. The [audit](../04-reproducibility/original-149-v16-audit.md) and [title-free outcomes](../04-reproducibility/original-149-v16-results.csv) document the reconciled record.

The next reproduction release will link results to source revision, runtime and dependency versions, model/provider, prompt set, rule fingerprint, retrieval settings, and cache state. It will include a rerun guide and a shareable benchmark or synthetic substitute selected through the source-level access inventory.

## Evaluation workplan

1. **Expert agreement:** compare rule-level judgments and aggregate scores with expert ratings under an institutionally reviewed protocol.
2. **Repeatability:** repeat evaluations with fixed data and configurations; quantify score and rule-level variation.
3. **Evidence sensitivity:** compare pre-specified evidence budgets and report changes in judgments and score completeness.
4. **Scoring calibration:** compare V16 60/40 with alternative pool ratios against expert judgments.
5. **Scope:** extend the sample across paper types and report where the workflow performs consistently or requires human review.
6. **Reproduction:** conduct a clean rerun from a versioned manifest and independent researcher instructions.
