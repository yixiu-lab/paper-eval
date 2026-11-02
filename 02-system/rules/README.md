# Scoring Rules

The operational scoring profile is defined in [`rules_v1.yaml`](rules_v1.yaml). The reviewer workbook, [`scoring_rules.xlsx`](scoring_rules.xlsx), presents the same rule definitions and score mappings in a filterable format.

## How to inspect a score

1. In **Rule Guide**, locate the rule and read what it assesses, the evidence the agent must cite, the relevant paper sections, diagnostic questions, applicability, weights, and any rule-specific thresholds.
2. In **Scoring Anchors**, filter by the same Rule Key. Each row gives one allowed judgment tier, its full rule-specific anchor, the score multiplier, and whether it enters the pool calculation.
3. In **Constructs** and **Dimensions**, inspect how rule weights and construct weights are grouped and how each group reconciles to the declared dimension maximum.
4. Compare the agent's judgment and evidence references with the corresponding rule and anchor before interpreting the reported score.

## Score calculation

For each applicable rule, the agent returns a judgment tier and evidence references. The rule's declared score mapping converts the tier to a multiplier. Within each construct, the evaluator calculates a rule-weighted average from scored rules. Construct weights then determine the construct's contribution to the relevant pool rate.

- **Baseline pool (60 points):** `mostly` = 1.00, `partial` = 1.00, `weak` = 0.50, and `not_satisfied` = 0.00.
- **Bonus pool (40 points):** `satisfied` = 1.00, `mostly` = 0.50, `partial` = 0.25, and `weak` or `not_satisfied` = 0.00.
- **Total score:** `60 × baseline attainment rate + 40 × bonus attainment rate`.

`uncertain` and missing judgments are excluded from pool attainment denominators; scoring completeness is reported separately. A construct outside the paper's applicable scope is excluded as a whole. Rules with zero weight remain visible in the workbook but do not affect the total score.
