# PaperEval

### An evidence-grounded, auditable workflow for AI-assisted research assessment

PaperEval is a research prototype that decomposes assessment of a management research paper into rule-level judgments, requires supporting evidence, checks selected evidence and citation conditions, and aggregates scores deterministically. It is intended as research infrastructure for studying AI-based measurement and its failure boundaries—not as an automated peer-review or editorial decision system.

> **Working research question — for author confirmation:** Under what conditions can a human-directed AI workflow produce stable, traceable assessments of management research, and where do evidence retrieval, model variation, scoring design, and domain limits undermine those assessments?

## Why this project may fit the call

The *Organization Science* call for “AI-Enabled Frontiers in Organizational Science” seeks new research capabilities and research infrastructure, rather than conventional studies that merely use AI. PaperEval offers a prototype for examining whether research assessment itself can become a transparent, testable measurement process: each rule-level judgment is tied to selected evidence, uncertainty is retained for review, and the final score is computed from a declared rule profile.

The intended contribution is the workflow and the knowledge gained by testing its validity and limits. The current results do **not** establish that AI can reliably replace expert review, that the score is valid, or that the 60/40 scoring split is optimal.

## What the system does

1. Parses a research-paper PDF and builds a structured paper profile.
2. Extracts research claims and retrieves related literature for comparison.
3. Prepares rule-specific evidence for six reviewer dimensions: question positioning, theoretical contribution, theoretical rigor, research design, results and evidence, and discussion/boundary conditions.
4. Applies evidence and citation checks; uncertain or insufficiently supported judgments can be flagged for review.
5. Aggregates rule judgments using a versioned scoring profile and produces a traceable report.

The primary code is in [`paper_eval/`](paper_eval/). The local Web/API prototype is excluded from this repository. Scoring rules and some source comments remain in Chinese; the submission-facing materials are in English.

## Evidence available now

The V16 audit covers **149 source PDFs**. Exact SHA-256 deduplication leaves **146 unique papers**; **144** have complete, traceable V16 reports and **2** were blocked before scoring by PDF parsing. No scores were imputed for the blocked papers. The title-free per-paper outcomes and audit details are in [`submission/04-reproducibility/`](submission/04-reproducibility/).

**Rule version:** the checked-in scoring profile is **V16, draft**, matching the V16 profile used in the 149-PDF audit. The exact historical runtime configuration and run-specific code revision are not fully included, so the current repository is not yet a complete independent reproduction package.

The broader local experiment archive also records repeated-run, scoring-mechanism, evidence-budget, calibration, and reweighting studies. Its 17-paper trace database is a separate subset and should not be confused with the 149-PDF V16 audit. See the [experiment evidence summary](submission/04-reproducibility/experiment-evidence.md).

These are exploratory engineering and measurement records. They do not include an expert-labeled benchmark or an independent validity study. The current corpus consists of published papers and cannot establish performance on low-quality or off-topic submissions. Repeated runs show nontrivial score variation, and output changes with evidence budget. The 60/40 pool split has not been shown to be optimal.

## What editors can inspect

- [Full-corpus V16 audit](submission/04-reproducibility/original-149-v16-audit.md): corpus reconciliation, coverage, run configuration, and limitations.
- [Per-paper V16 outcome table](submission/04-reproducibility/original-149-v16-results.csv): one row per unique PDF, identified by checksum; no titles, paper text, or local paths.
- [Technical appendix](submission/02-technical-appendix/README.md): architecture, scoring logic, and known limitations.
- [Research artifact description](submission/03-research-artifact/README.md): what is implemented and what is not established.
- [Reproducibility materials](submission/04-reproducibility/README.md): available and missing materials.
- [Ethics and access checklist](submission/05-ethics-and-access/README.md): statements that still require author review.
- [Stage 1 package status](submission/README.md): checklist against the call.
- [System baseline](system-baseline.md): current CLI snapshot, historical V16 boundary, and runtime notes.

## Stage 1 status

The call requests a 1,500–2,000-word short proposal plus references, a technical appendix, and a prototype/repository. This repository contains the prototype and a draft technical appendix. **The proposal and author-approved ethics/access statements are not yet complete.** Reproduction evidence is documented, but the source corpus and full paper-level traces are withheld pending rights review; an independent reproduction package is still needed.

Stage 1 deadline in the provided call: **November 1, 2026**.

## Repository map

- `paper_eval/`: paper parsing, evidence retrieval, reviewer workflow, scoring, and reports.
- `评价规则库/`: versioned scoring profiles and design notes.
- `configs/`: example runtime and retrieval configurations.
- `prompts/`: reviewer-agent prompts.
- `scripts/`: offline self-checks and retrieval utilities.
- `submission/`: English, call-aligned materials for editors.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m paper_eval.cli --help
```

Model-backed runs require locally supplied credentials and may incur service costs. Never commit API keys, `.env` files, private manuscripts, or unapproved run traces. The source corpus and full reports are not included.

## License and data

No project license or redistributable source corpus is currently provided. Resolve code licensing, paper-data rights, and model-access alternatives before public release or third-party reproduction.
