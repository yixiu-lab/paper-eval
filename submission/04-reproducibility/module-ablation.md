# PaperEval Module Ablation Records

## Five-paper, four-condition comparison

The archived comparison used the same five paper inputs and model configuration across four conditions: full PaperEval, a single Reviewer, a Reviewer-only counterfactual without Harness gating/validation, and direct one-call scoring.

| Condition | Mean score | Range across five papers |
|---|---:|---:|
| Full PaperEval | 79.50 | 71.48–91.54 |
| Reviewer-only counterfactual | 57.93 | 50.74–74.22 |
| Single Reviewer | 74.60 | 60–83 |
| Direct LLM | 74.00 | 59–82 |

The scores show how the workflow conditions produce different score scales on the same sample. The next expert-benchmark study will compare each condition with the same independent ratings and pre-specified outcomes.

## Paired one-factor module ablation

A separate V16 run used one paper, a shared parsed-text evidence inventory, and the same baseline-generated context, external retrieval, model, and aggregation. Each arm disabled one module.

| Module disabled | Total score | Difference from baseline |
|---|---:|---:|
| Internal evidence retrieval | 77.025 | −15.381 |
| Boundary review | 91.469 | −0.938 |
| Social-issue evidence | 91.187 | −1.219 |
| Second pass | 92.406 | 0.000 |
| External validation | 91.187 | −1.219 |
| Prior-work grounding | 90.906 | −1.500 |
| Schema validation | 89.406 | −3.000 |
| Evidence gate | 92.406 | 0.000 |
| Unified reviewer | 91.625 | −0.781 |

The one-paper paired design isolates configured component changes for that run. The next study will repeat the ablation across a broader sample and compare rule-level outputs, review flags, and expert agreement.

## Evidence-gate output comparison

In a three-paper paired analysis, the same stored Reviewer judgments and evidence were processed with and without the evidence gate. The gate changed review/output fields for 8, 9, and 6 rules across the three papers, while the ordinal judgments remained the same. This record describes gate behavior on those runs; the next experiment will evaluate how the flags guide human reviewers.
