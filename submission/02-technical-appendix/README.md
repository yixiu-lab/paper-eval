# PaperEval: System Architecture and Technical Method

## 1. System overview

PaperEval is an evidence-grounded workflow for evaluating scholarly papers. Its architecture converts a PDF into a structured evidence base, maps a versioned domain profile into executable rule-level tasks, routes evidence to specialist reviewers, and passes their judgments through a deterministic validation and aggregation layer. Each reported score is linked to a rule, cited evidence, a rationale, and any review flag or validation result.

The core processing chain is:

```mermaid
flowchart LR
    A[Research paper PDF] --> B[PDF parser]
    B --> C[Paper profile and evidence index]
    C --> D[Rule-specific evidence routing]
    C --> E[Claim extraction]
    E --> F[External prior-work retrieval]
    D --> G[Six dimension reviewers A-F]
    F --> G
    G --> H[Structured judgment validation]
    H --> I[Evidence and external-evidence checks]
    I --> J[Uncertain-item evidence retry]
    J --> K[Evidence gate]
    K --> L[Deterministic scoring and aggregation]
    L --> M[Reflection and process audit]
    M --> N[JSON, Markdown, HTML reports and run trace]
    N --> O[Optional reliability diagnosis]
```

The command-line harness coordinates the run, while the evaluation graph defines the ordered reviewer workflow. Model calls produce paper interpretation and rule-level judgments; program code validates references, applies versioned score mappings, computes scores, and records the run trace.

## 2. Rule profile: the domain adaptation layer

The rule profile defines what the system evaluates. It contains dimensions, rule-level indicators, evidence requirements, allowed judgment states, weights, score mappings, applicability conditions, and the baseline/bonus allocation. The current checked-in profile is **V16**, with 34 rules across six dimensions. Its fingerprint is SHA-256 `811f6c0d992460113a88e66bcb37a4124da046b7c0d5b6725e1d653549284c0f`; the [V16 rule file](../../评价规则库/rules_v1.yaml) is versioned with the artifact, and the [Excel scoring-rule workbook](../../评价规则库/PaperEval_V16_scoring_rules.xlsx) provides a reviewer-friendly tabular view.

| Dimension | Reviewer responsibility |
| --- | --- |
| A. Research question and positioning | Importance, clarity, researchability, and grounding of the stated gap |
| B. Theoretical contribution and novelty | Contribution through mechanisms, boundaries, context, measurement, concepts, or literature dialogue |
| C. Theoretical rigor and hypothesis logic | Construct definitions, theoretical premises, links, boundary conditions, and alternatives |
| D. Research design | Fit of sample, measures, models, identification, experiments, and robustness to the question |
| E. Results and evidence | Whether analyses and qualitative or quantitative evidence support the claims |
| F. Discussion, boundaries, and integration | Responsible treatment of implications, limitations, external validity, and grounded social relevance |

The profile separates a 60-point baseline pool from a 40-point bonus pool. The scorer maps rule judgments to construct-level rates, then aggregates available constructs according to the profile. Uncertain judgments contribute to completeness and review reporting; they are not silently treated as low-quality judgments. Rules outside a paper's recognized scope are excluded from the applicable score and recorded as out of scope. The aggregation implementation is in [`paper_eval/scorer.py`](../../paper_eval/scorer.py); the profile and validation logic are in [`paper_eval/rules.py`](../../paper_eval/rules.py).

The profile is the intended adaptation boundary: a new research domain can replace its dimensions, rules, evidence requirements, weights, and calibration mappings while retaining the PDF-to-report workflow. The historical run identifier `MGMT-LIT-EVAL-V1:V16` is retained in run records for provenance; the system and repository name is PaperEval.

## 3. PDF parsing and paper representation

The input layer parses each PDF into a `Paper` record and a collection of `Evidence` records. The parser extracts page text, detects section headings, segments text into paragraphs, and attaches page number, section, character span, stable evidence ID, and text hash to each paragraph. Reference paragraphs are tracked separately. Parse diagnostics record page coverage, section/facet counts, and warnings; a severe parse failure can stop the run before model calls.

The context layer creates a structured paper profile for downstream routing and review. It represents the paper type, central problem, research gap, research question, core claim, contribution, theory move, method and data, evidence structure, findings, boundaries, and limitations. It also builds a section map, an evidence index, and a compact cross-section context pack. The profile is model-assisted and grounded in selected full-text evidence; when profile generation fails or returns invalid controlled values, the implementation can retain heuristic defaults and record the resulting trace.

Key modules: [`paper_eval/pdf_parser.py`](../../paper_eval/pdf_parser.py), [`paper_eval/paper_context.py`](../../paper_eval/paper_context.py), and parse diagnostics in the harness.

## 4. Rule-specific evidence packages

For each rule, the evidence router uses the rule definition, section expectations, paper profile, and internal evidence index to select a bounded reading pack. Candidate selection combines section matching, rule-specific terms, evidence-form cues, page diversity, and scoring logic for different rule families. Dimension-level routing also uses the profile's evidence routes where available and a deterministic fallback when it does not.

Each package gives a reviewer the rule identifier and criterion, relevant paper context, selected evidence text, page and section metadata, and evidence IDs. The reviewer therefore evaluates a bounded question against traceable passages rather than producing only a paper-level impression. Evidence selection and payload construction are implemented in [`paper_eval/agents.py`](../../paper_eval/agents.py); context-level routing is implemented in [`paper_eval/paper_context.py`](../../paper_eval/paper_context.py).

## 5. Claim-led external grounding

When enabled by runtime configuration, the external-grounding path extracts claims from the paper profile and source passages. Claim types include the research problem, gap, theory, contribution, and related assertions; each claim retains its source evidence ID and may include concepts, theory, mechanism, context, keywords, and publication year. Claim-derived queries retrieve candidate related works from configured scholarly metadata providers. The implementation supports Crossref, OpenAlex, and Semantic Scholar, with provider status, failures, query details, and cache state represented in retrieval output.

Candidate records are normalized and deduplicated, filtered against the target paper's publication year when available, and reranked using claim and paper context. The selected prior works are converted into external evidence records with stable evidence IDs and metadata. These records can support A/B judgments about the research gap and contribution. External validation checks evidence coverage and citation conditions; a positive comparison-dependent judgment without required external support can be changed to uncertain or flagged for review.

Social-relevance grounding is a separate optional evidence input. When enabled, issue-grounding passages are added to the shared evidence collection for the F dimension. External literature and issue-grounding records enter through the same evidence interface as paper passages; they do not create additional scoring dimensions or independent final scores.

Key modules: [`paper_eval/claim_extraction.py`](../../paper_eval/claim_extraction.py), [`paper_eval/external_retrieval.py`](../../paper_eval/external_retrieval.py), [`paper_eval/external_evidence.py`](../../paper_eval/external_evidence.py), and [`paper_eval/social_relevance.py`](../../paper_eval/social_relevance.py). Availability depends on the runtime and retrieval configuration; disabled or unavailable retrieval is recorded in the run output.

## 6. Specialist review and judgment protocol

The reviewer layer applies the same rule profile and judgment schema across dimensions. In the six-reviewer configuration, A–F are assigned to Problem Framing, Contribution, Theory Logic, Research Design, Evidence, and Boundary responsibilities. A runtime configuration can also select the unified-reviewer mode. Each reviewer receives its assigned rules and evidence packages and returns structured rule-level judgments with a judgment category, rationale, confidence, cited evidence IDs, and review indicators.

The model runtime is configured through an OpenAI-compatible endpoint and example configurations. Prompt and model settings are explicit configuration inputs. The orchestration can retry uncertain judgments with a second evidence-retrieval pass, allowing the reviewer to inspect additional passages before final validation. Runtime failures are surfaced in the run result and failure report rather than converted into a score.

The graph order is defined in [`paper_eval/agent_graph.py`](../../paper_eval/agent_graph.py), reviewer evidence preparation and calls in [`paper_eval/agents.py`](../../paper_eval/agents.py), and endpoint configuration in [`paper_eval/llm.py`](../../paper_eval/llm.py). The standard graph proceeds through paper context, external retrieval, novelty evidence preparation, optional social grounding, reviewer calls, external-evidence validation, uncertain-item retry, schema validation, evidence gating, score aggregation, reflection, and process audit.

## 7. Harness: validation, gating, and aggregation

The Harness is the deterministic control layer between model judgments and reported scores. It runs the following checks and transformations:

1. **Schema validation.** Confirm rule IDs belong to the active profile; judgments use valid categories; confidence is in range; and required rationale, reviewer identity, and rule coverage are present.
2. **Evidence reference validation.** Resolve cited evidence IDs against the paper and external evidence records. A positive judgment with no valid evidence is converted to uncertain because it cannot be traced.
3. **Evidence-strength review flags.** Inspect whether cited passages provide substantive support or only a mention. Weak support, a rationale that conflicts with a high judgment, or too few cited passages can trigger human-review flags. These flags preserve the judgment and make the issue visible for review.
4. **External evidence validation.** Check the external evidence used for comparison-dependent rules, including availability, citation references, and temporal constraints where the target publication year is known. Findings are attached to the report and review queue.
5. **Uncertain-item retry.** Revisit uncertain rules with a second evidence pack, then validate the resulting judgment.
6. **Deterministic scoring.** Apply V16 score mappings and construct weights, handle out-of-scope rules, aggregate the two pools, and report completeness and uncertainty separately.
7. **Reflection and process audit.** Generate review issues from judgment/evidence/score patterns and run a separate audit of the scoring process. These outputs support reviewer inspection and do not replace the rule-level score calculation.

The evidence gate is implemented in [`paper_eval/evidence_gate.py`](../../paper_eval/evidence_gate.py); schema checks in [`paper_eval/schema_validation.py`](../../paper_eval/schema_validation.py); external checks and graph ordering in [`paper_eval/agent_graph.py`](../../paper_eval/agent_graph.py); and deterministic score calculation in [`paper_eval/scorer.py`](../../paper_eval/scorer.py). Multi-model consistency review is a post-review step orchestrated by [`paper_eval/harness.py`](../../paper_eval/harness.py); it produces agreement and review information alongside the primary judgments.

## 8. Reports, run records, and reliability analysis

Each run can produce a JSON report, a Markdown report, and an HTML report, together with a run manifest and a SQLite trace bundle. The report includes the paper profile, active rule-profile version, parse diagnostics, dimension and total scores, rule-level judgment details, rationale, cited and candidate evidence, page coverage, review flags, external retrieval and validation records, reflection issues, and any consistency or audit findings. Trace events record workflow stages and key counts/settings so a run can be inspected in sequence.

An optional reliability layer reads prior reports for the same paper and the current in-memory report. It diagnoses score and judgment variation, evidence overlap, ranking stability, and calibration sensitivity. It is post-processing: it appends a reliability report and does not change the current run's judgments or scores. With insufficient repeated runs, the report indicates that reliability cannot yet be estimated.

Key modules: [`paper_eval/report.py`](../../paper_eval/report.py), [`paper_eval/html_report.py`](../../paper_eval/html_report.py), [`paper_eval/storage.py`](../../paper_eval/storage.py), [`paper_eval/reliability.py`](../../paper_eval/reliability.py), and [`paper_eval/reliability_diagnosis.py`](../../paper_eval/reliability_diagnosis.py).

## 9. Execution interface and artifact boundary

The public research artifact is the Python package and command-line workflow. A single-paper run is invoked as follows:

```bash
python -m paper_eval.cli evaluate --pdf <PDF_PATH> --rules <V16_RULE_PROFILE> --out <OUTPUT_DIR> --agent-config configs/agent_runtime.example.yaml
```

Validate the profile before a run:

```bash
python -m paper_eval.cli validate-rules --rules <V16_RULE_PROFILE>
```

Runtime dependencies are listed in [`requirements.txt`](../../requirements.txt), and example settings are in [`configs/`](../../configs/). Credentials are supplied through local environment configuration rather than committed files. The local Web/API prototype is not part of this repository; the CLI, rule profile, prompts, and reusable evaluation modules constitute the current system artifact.

## 10. Architecture-to-code map

| System layer | Main implementation |
| --- | --- |
| Command-line interface and run coordination | [`paper_eval/cli.py`](../../paper_eval/cli.py), [`paper_eval/harness.py`](../../paper_eval/harness.py) |
| PDF parsing and evidence records | [`paper_eval/pdf_parser.py`](../../paper_eval/pdf_parser.py), [`paper_eval/models.py`](../../paper_eval/models.py) |
| Paper profile and evidence routing | [`paper_eval/paper_context.py`](../../paper_eval/paper_context.py), [`paper_eval/agents.py`](../../paper_eval/agents.py) |
| Claim extraction and prior-work retrieval | [`paper_eval/claim_extraction.py`](../../paper_eval/claim_extraction.py), [`paper_eval/external_retrieval.py`](../../paper_eval/external_retrieval.py) |
| Reviewer workflow and orchestration | [`paper_eval/agent_graph.py`](../../paper_eval/agent_graph.py), [`paper_eval/llm.py`](../../paper_eval/llm.py) |
| Validation, evidence gate, and aggregation | [`paper_eval/schema_validation.py`](../../paper_eval/schema_validation.py), [`paper_eval/evidence_gate.py`](../../paper_eval/evidence_gate.py), [`paper_eval/scorer.py`](../../paper_eval/scorer.py) |
| Reflection, process audit, and consistency | [`paper_eval/reflection.py`](../../paper_eval/reflection.py), [`paper_eval/audit_agent.py`](../../paper_eval/audit_agent.py), [`paper_eval/consistency.py`](../../paper_eval/consistency.py) |
| Reports, storage, and reliability | [`paper_eval/report.py`](../../paper_eval/report.py), [`paper_eval/storage.py`](../../paper_eval/storage.py), [`paper_eval/reliability_diagnosis.py`](../../paper_eval/reliability_diagnosis.py) |

## 11. Current empirical anchor

V16 is the profile used in the documented 149-source-PDF audit: 146 unique papers after exact deduplication, 144 complete reports, and two papers blocked before scoring by PDF parsing. The audit details and title-free outcomes are documented in the [reproducibility materials](../04-reproducibility/original-149-v16-audit.md). Other model, robustness, uncertainty, ablation, and retrieval experiments are indexed in the [experiment catalog](../04-reproducibility/experiment-catalog.md).

The historical profile identifier is preserved in run provenance. The named system, public repository, and reviewer-facing architecture in this appendix are **PaperEval**.
