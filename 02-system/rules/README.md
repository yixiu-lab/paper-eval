# Scoring Rules

The operational rule profile is defined in [`rules_v1.yaml`](rules_v1.yaml) and loaded by the command-line evaluator. The complete reviewer-facing English table is [`scoring_rules.xlsx`](scoring_rules.xlsx).

The workbook provides the profile overview, six dimensions, 33 constructs, 34 rules, evidence requirements, scoring mappings, applicability notes, sources, retrieval terms, thresholds, and all available judgment anchors. It omits internal status and change-history fields.

The rule profile uses a 100-point scale divided into a 60-point baseline pool and a 40-point bonus pool. Each rule is assigned to one pool, and the Harness applies the declared score mapping and weights during aggregation.
