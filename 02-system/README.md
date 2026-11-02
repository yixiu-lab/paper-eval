# PaperEval System

PaperEval is a command-line research artifact for evaluating scholarly papers through structured, evidence-linked judgments. This page is the entry point for reviewers and collaborators inspecting the implementation.

## How the system works

1. **Parse and profile a paper.** Extract PDF text and structure, create page- and section-linked evidence records, and build a structured paper profile.
2. **Retrieve related work when enabled.** Extract paper claims and query configured scholarly metadata providers; retain source links and retrieval records.
3. **Assemble rule-specific evidence.** Route passages to six evaluation dimensions: research-question positioning, theoretical contribution, theoretical rigor, research design, results and evidence, and discussion and boundaries.
4. **Generate structured judgments.** Reviewer agents return rule-level judgments, rationales, confidence values, and evidence references.
5. **Validate and score.** The Harness checks judgment schemas and evidence references, flags items for review, and applies the selected rule profile and score mappings.
6. **Produce inspectable reports.** Runs can produce JSON, Markdown, and HTML reports, manifests, and trace records.

## Reviewer reading path

- Start with the [project overview](../01-proposal/README.md) for the research question and study context.
- Read the [technical appendix](technical-appendix.md) for the architecture, evidence flow, validation stages, and code map.
- Inspect [`paper_eval/`](paper_eval/) for the implementation.
- Review the [scoring rules overview](rules/README.md), the [operational rule profile](rules/rules_v1.yaml), and the [complete English scoring table](rules/scoring_rules.xlsx).
- See the [experiment catalog](../03-experiments/README.md) and [data materials](../04-data/README.md) for study results and released outcomes.
- See the [ethics and data-access statement](../05-ethics-access/README.md) for data handling and access scope.

## Run the command-line workflow

Run commands from this folder so package and configuration paths resolve correctly.

```bash
python -m venv .venv
pip install -r requirements.txt
python -m paper_eval.cli --help
python -m paper_eval.cli validate-rules --rules rules/rules_v1.yaml
```

Evaluate one paper:

```bash
python -m paper_eval.cli evaluate --pdf <PDF_PATH> --rules rules/rules_v1.yaml --out <OUTPUT_DIR> --agent-config configs/agent_runtime.example.yaml
```

Model-backed runs require a configured provider. Keep credentials in local environment variables or secret storage. Source PDFs and full-text run traces are not included in this repository.

## Artifact contents

- [`paper_eval/`](paper_eval/): parsing, retrieval, reviewer orchestration, validation, scoring, reporting, and reliability modules.
- [`rules/`](rules/): machine-readable rule profiles and the English scoring table.
- [`configs/`](configs/): example runtime and retrieval settings.
- [`prompts/`](prompts/): reviewer prompts.
- [`scripts/`](scripts/): local checks and utilities.
- [`requirements.txt`](requirements.txt): Python dependencies.

The local Web/API prototype is outside the scope of this research artifact.
