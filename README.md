# PaperEval

### An evidence-grounded workflow for studying AI-assisted research assessment

PaperEval turns research-paper assessment into an inspectable measurement workflow. It links rule-level judgments to selected paper evidence, records uncertainty for human review, checks evidence and citation conditions, and aggregates judgments using a versioned scoring profile. The project studies how this workflow can support organizational and management research.

## Research objective

We are developing a human-directed, evidence-grounded workflow for assessing management research. The next study will establish how its judgments compare with expert assessments, how stable the scores are across runs and evidence settings, and which paper types and conditions require additional review.

The contribution is a testable research capability: paper assessments can be examined at the rule, evidence, and aggregation levels, giving researchers a way to study AI-assisted measurement and its boundary conditions.

## Workflow

1. Parse a research-paper PDF and build a structured paper profile.
2. Extract research claims and retrieve related literature for comparison.
3. Assemble rule-specific evidence for six dimensions: research-question positioning, theoretical contribution, theoretical rigor, research design, results and evidence, and discussion/boundary conditions.
4. Generate rule-level judgments with evidence references and uncertainty indicators.
5. Apply evidence and citation checks, then aggregate results with a versioned scoring profile.
6. Produce a traceable report for researcher inspection.

The primary implementation is in [`paper_eval/`](paper_eval/). The local Web/API prototype is not part of this research artifact.

## Current empirical record

The V16 audit began with **149 source PDFs**. Exact SHA-256 deduplication yielded **146 unique papers**. The workflow produced **144 complete reports**; two papers were blocked before scoring by PDF parsing and remain without imputed scores. The audit reports a mean total score of 76.46 (SD 7.64), median 76.19, and range 49.63–95.78 across the 144 completed reports.

The checked-in profile is **V16**, a draft 34-rule profile, and its SHA-256 matches the profile used in the audit. The historical run used `MGMT-LIT-EVAL-V1:V16`, DeepSeek Flash, temperature 0, thinking disabled, and `MGMT-SCALE-V6`. The next reproducibility milestone is to link the historical run to its exact source revision and full runtime manifest, then perform a clean rerun.

The local experiment archive also records repeated-run, scoring-mechanism, evidence-budget, calibration, and reweighting studies. Its 17-paper trace database is a separate subset from the 149-PDF audit. The [experiment summary](submission/04-reproducibility/experiment-evidence.md) separates each observation from its next validation step.

## Research milestones

1. **Reproducibility:** publish a version-linked run manifest, dependency lock, prompts, retrieval settings, and rerun guide.
2. **Expert comparison:** build an expert-labeled benchmark under an institutionally reviewed protocol and compare rule-level judgments and aggregate scores.
3. **Stability and calibration:** compare repeated runs and pre-specified scoring profiles, including the current 60/40 profile and alternative pool ratios.
4. **Scope:** extend evaluation beyond published AI/HRM papers to varied paper types and document where performance changes.
5. **Access:** complete a source-level rights inventory and prepare a shareable benchmark or synthetic substitute.

## Editor reading path

- [Technical appendix](submission/02-technical-appendix/README.md): architecture, V16 design, and evaluation plan.
- [Research artifact](submission/03-research-artifact/README.md): implementation and reviewer walkthrough.
- [149-PDF audit](submission/04-reproducibility/original-149-v16-audit.md) and [title-free outcomes](submission/04-reproducibility/original-149-v16-results.csv).
- [Historical experiment evidence](submission/04-reproducibility/experiment-evidence.md).
- [Reproduction plan](submission/04-reproducibility/README.md).
- [Ethics, data, and access plan](submission/05-ethics-and-access/README.md).
- [System baseline](system-baseline.md): entry points and version boundaries.

## Repository map

- `paper_eval/`: parsing, retrieval, reviewer workflow, scoring, reporting, and diagnostics.
- `评价规则库/`: versioned scoring profiles and design notes; V16 is the profile used for the 149-PDF audit.
- `configs/`: example runtime and retrieval configurations.
- `prompts/`: reviewer-agent prompts.
- `scripts/`: local logic checks, mocked-service checks, and retrieval utilities. See the [validation guide](submission/04-reproducibility/README.md) for checks that contact live model or retrieval services.
- `submission/`: English materials prepared for the *Organization Science* call.

## Reproduce the CLI setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m paper_eval.cli --help
```

Model-backed runs use a configured provider and may incur service costs. For each new run, record provider/model/version, code revision, rule fingerprint, retrieval parameters, cache state, and failures in the run manifest.

## Ethics and access

The current corpus consists of published AI/HRM papers; source PDFs and full paper-level traces remain in the research workspace while the team completes a source-level access inventory. The [ethics and access plan](submission/05-ethics-and-access/README.md) assigns the next actions for source permissions, expert-rater review, consent, privacy, provider terms, retention, and editor access. The project will provide a shareable benchmark or synthetic substitute alongside a version-linked reproduction package.
