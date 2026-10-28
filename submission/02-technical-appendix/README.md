# Technical Appendix

## System purpose

PaperEval is a prototype for structured evaluation of management research papers. Its intended research value is the inspectability of an evidence-grounded, rule-based workflow and the ability to study where AI-generated judgments are reliable or fail. The repository does not yet demonstrate validity against expert judgments.

## Current workflow

The documented pipeline parses a paper PDF, builds a paper profile, extracts contribution claims, retrieves related literature, prepares evidence packs for six reviewer dimensions, requests rule-level judgments, retrieves additional evidence for uncertain cases, applies deterministic scoring, and produces a report with reliability diagnostics.

The six reviewer dimensions are: research question positioning; theoretical contribution; theoretical rigor; research design; results and evidence; and discussion and boundary conditions. An auxiliary novelty comparison is separate from the six scoring reviewers and does not directly produce the total score.

## Scoring design

The draft scoring profile separates a baseline pool (60 points) from a bonus pool (40 points). Baseline rules aim to represent adequacy thresholds; bonus rules represent evidence beyond adequacy. Uncertain judgments are excluded from the available-score denominator and should be reported with completeness. This design is an implementation choice under study, not a validated universal scale. The active profile is in `../../评价规则库/rules_v1.yaml`; its detailed design notes are currently Chinese.

## Evidence and retrieval

The code uses deterministic heuristics to assemble evidence packs and has an optional retrieval/reranking path. The baseline notes report a same-paper cached evidence-pack comparison, but these internal observations do not establish evaluator validity, inter-rater reliability, or general performance. External retrieval uses Semantic Scholar with Crossref/OpenAlex fallback paths in the implementation. Service availability and access conditions must be recorded for each reproduction.

## Runtime and entry points

- Single-paper CLI: `python -m literature_eval.cli evaluate` (provide the required paper, rule-profile, and output arguments).
- Runtime dependencies: [`../../requirements.txt`](../../requirements.txt).
- Example configurations: `../../configs/`.

The full 149-PDF corpus run is summarized in [`../04-reproducibility/original-149-v16-audit.md`](../04-reproducibility/original-149-v16-audit.md), with title-free per-paper scores in [`../04-reproducibility/original-149-v16-results.csv`](../04-reproducibility/original-149-v16-results.csv): 146 unique papers after exact deduplication, 144 complete V16 reports, and two papers blocked before scoring by PDF parsing. The per-paper reports, source corpus, and original runtime configuration are not included here, so the result is documented but not independently reproducible from this repository snapshot yet.

The local web/API prototype is intentionally excluded from this repository submission. The research artifact is the evaluation workflow and command-line components described above.

## Known limitations and open questions

- No expert-labeled benchmark or independent validity study is included.
- No shareable paper corpus or data license inventory is included.
- No end-to-end cloud run is evidenced in this repository snapshot.
- The scoring scale and default retrieval parameters are not established as generally valid.
- Uncertain-item denominator handling can affect comparability and must be reported.
- The prototype, rules, prompts, and much of the source documentation remain Chinese; English translation and terminology review are outstanding.
- The current artifact evaluates research papers; the proposal must explain why this creates a consequential new capability for organizational/social science research rather than simply automating peer review.
