# Robustness and Harness Experiments

## Evidence-level prompt-injection comparison

The matched comparison used DeepSeek Flash and stored evidence packs from three papers. It covered six reviewer dimensions, three attack types (force-high, force-low, and fabricated evidence ID), and two repeats: 108 paired attack cases per prompt condition. The two conditions used the same papers, evidence, attack templates, model, and repeat numbers; the prompt differed in its evidence-boundary instruction.

| Outcome | Legacy prompt | Evidence-boundary prompt |
|---|---:|---:|
| Attacks reaching the specified target | 4/108 (3.7%) | 0/108 (0%) |
| Ratings differing from paired clean controls | 14/108 | 7/108 |
| Mean absolute ordinal shift | 0.250 | 0.074 |
| Fabricated evidence IDs remaining after normalization | 0 | 0 |

This pilot supports testing evidence-boundary instructions as a robustness measure. A rating change is distinct from reaching the attacker’s target. The next study will expand models and papers, repeat each condition, compare outputs with expert review, and test additional attack surfaces.

A separate production-prompt run recorded 5 target-reaching cases among 108 attacks (4.6%) and 20 rating changes among 108. An earlier single-rule synthetic-evidence pilot made 30 calls across four attack types and two prompt conditions. These are separate runs and are not pooled with the matched comparison above.

## Deterministic harness fault injection

The harness experiment constructed 20 examples for each of nine machine-checkable fault types and 40 clean controls, for 220 cases total.

| Outcome | Result |
|---|---:|
| Fault detection with the harness disabled | 0/180 (0%) |
| Fault detection with schema validation and evidence gate enabled | 180/180 (100%) |
| Automatic blocking or transformation | 160/180 (88.9%) |
| Flagged for human review | 20/180 (11.1%) |
| False positives on clean controls | 0/40 (0%) |

All 20 reason/judgment-conflict cases were detected and routed for review; the other eight fault classes were automatically blocked or transformed. The next test will add realistic malformed outputs and connect the checks to expert review of evidence support and paper-level judgments.

## Reliability and uncertainty diagnostics

The reliability layer was recomputed over 57 archived runs spanning 20 papers. It reports run-score variation, evidence overlap, judgment agreement, ranking stability, calibration sensitivity, and a diagnostic of the main uncertainty source. In the focal three-run example, the score range was 6.36 points and the grade changed between repeats; the batch comparison reported Spearman 0.6491 and Top-5 overlap 0.25.

The next controlled study will separate evidence variation from model-judgment variation by holding the evidence package fixed for one arm and allowing retrieval to vary in another. Calibration alternatives will be evaluated against expert labels.
