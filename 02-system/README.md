# PaperEval System

This folder contains the complete technical system package.

- [Technical architecture](technical-appendix.md)
- [Reviewer guide](reviewer-guide.md)
- [System baseline and CLI entry points](system-baseline.md)
- [`paper_eval/`](paper_eval/): parsing, retrieval, reviewers, Harness validation, scoring, reports, and reliability analysis.
- [`rules/`](rules/): V16 machine-readable rule profile, Excel workbook, and design notes.
- [`configs/`](configs/): example runtime and retrieval configurations.
- [`prompts/`](prompts/): reviewer prompts.
- [`scripts/`](scripts/): local checks and retrieval utilities.
- [`requirements.txt`](requirements.txt): Python dependencies.

Run CLI commands from this folder so package and configuration paths resolve correctly:

```bash
python -m paper_eval.cli --help
python -m paper_eval.cli validate-rules --rules rules/rules_v1.yaml
```
