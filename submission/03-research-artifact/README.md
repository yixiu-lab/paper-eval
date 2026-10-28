# Research Artifact: Prototype and Repository

The research artifact is the PaperEval evaluation workflow in this repository. It includes a command-line evaluation path, reviewer-agent modules, draft scoring rules, retrieval components, example configurations, prompts, and offline self-check scripts. The local web/API prototype is intentionally excluded.

## What the prototype currently does

It is designed to process a research-paper PDF, assemble evidence for rule-level reviewer judgments, retrieve related literature, aggregate judgments using declared weights, and generate traceable reports. Treat these as implemented workflow features, not evidence that the resulting scores are accurate or useful to editors.

The checked-in scoring profile is V16 draft, matching the rule snapshot used for the 149-PDF results summarized in the reproducibility materials. The exact historical code revision and runtime configuration are still missing, so the current checkout is not yet a complete reproduction environment.

## What is not included or established

- No public/shareable benchmark or expert labels.
- No claim that scores predict expert judgments or publication outcomes.
- No complete, independently verified end-to-end reproduction record.
- No guarantee that external services or proprietary model endpoints are currently accessible.
- The interface and scoring content are not yet fully English-language.

See the repository root README for setup cautions and the technical appendix for the documented workflow and limitations.
