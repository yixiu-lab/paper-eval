from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any

import yaml


@dataclass(frozen=True)
class DimensionRule:
    dimension_id: str
    dimension_name: str
    weight: float
    description: str


@dataclass(frozen=True)
class IndicatorRule:
    rule_id: str
    version: str
    dimension_id: str
    dimension_name: str
    indicator_name: str
    definition: str
    required_evidence: str
    evidence_location: list[str]
    observation_points: list[str]
    judgement_values: dict[str, str]
    score_mapping: dict[str, float | None]
    weight: float
    applicable_type: str
    source_reference: list[str]
    status: str
    # Rule role: baseline = eligibility threshold (passing means full marks, only gaps will be judged); bonus = bonus points (full marks will be awarded only if satisfied).
    # Published papers are qualified by default on the baseline, so the difference should only come from bonus points - this is the core mechanism of V11.
    role: str = "baseline"
    # Multiple rules of the same construct share a cluster name; when summarizing, one cluster is only counted as one entry.
    # prevents the same construct from being weighted repeatedly. The default is to have a line of its own.
    cluster: str | None = None
    # Rule explanation: why this rule is set, what counts as enough and what counts as insufficient; and the evidence threshold for determining satisfied / mostly.
    # These will go into the report so readers can understand how the scores were derived.
    rationale: str | None = None
    threshold: str | None = None
    mostly_threshold: str | None = None
    # Tolerance description of mostly gears: Which common gaps do not block the majority to prevent the middle gear of the ladder from being elevated.
    mostly_note: str | None = None
    # English search terms for this rule, aligned with the English-language paper corpus.
    # (see the old implementation of rule_keyword_hints). English words are explicitly declared here for use in evidence retrieval (bag-of-words recall + LLM rearrangement).
    retrieval_terms: list[str] = field(default_factory=list)

    @property
    def cluster_key(self) -> str:
        return self.cluster or self.rule_id


@dataclass(frozen=True)
class ConstructRule:
    'Concept layer: the actual allocation unit of dimension weights. Rules for the same construct measure the same thing.'

    construct_id: str
    construct_name: str
    dimension_id: str
    weight: float
    note: str | None = None


@dataclass(frozen=True)
class RuleProfile:
    rule_profile_id: str
    rule_profile_name: str
    version: str
    status: str
    scoring_scale: float
    dimensions: list[DimensionRule]
    rules: list[IndicatorRule]
    constructs: list[ConstructRule] = field(default_factory=list)
    # Two-pool scale: the benchmark pool represents the benchmark score that "reaches the publication level", and the bonus pool represents the score that can be obtained beyond passing the standard.
    # The sum of the two must equal scoring_scale, enforced by validate_rule_profile.
    baseline_pool: float = 0.0
    bonus_pool: float = 0.0
    # Evidence budget: The absolute value of the score changes with the number of evidence items for each rule (the actual measurement of paragraphs 10→20 moves up by about 7 points),
    # Therefore, the budget must be versioned along with the rule base and cannot be changed quietly. Harness will compare the declared value with the actual effective value.
    max_evidence_per_rule: int | None = None

    @property
    def rules_by_id(self) -> dict[str, IndicatorRule]:
        return {rule.rule_id: rule for rule in self.rules}

    @property
    def constructs_by_id(self) -> dict[str, ConstructRule]:
        return {item.construct_id: item for item in self.constructs}


# Controlled enumeration of paper types. Rule applicability only recognizes these 5 categories and no longer uses free text matching.
PAPER_TYPE_CLASSES = ('Evidence', 'Qualitative', 'Theory', 'Summary', 'Mix')

# rule role whitelist.
RULE_ROLES = ("baseline", "bonus")

# The benchmark item is not judged as "excellent", so there is no satisfied; the bonus item must be satisfied to have room for extra points.
REQUIRED_JUDGEMENT_KEYS_BY_ROLE = {
    "baseline": ("mostly", "partial", "weak", "not_satisfied", "uncertain"),
    "bonus": ("satisfied", "mostly", "partial", "weak", "not_satisfied", "uncertain"),
}
# Compatible with old references: Complete gear set.
REQUIRED_JUDGEMENT_KEYS = REQUIRED_JUDGEMENT_KEYS_BY_ROLE["bonus"]

# old descriptive tag → controlled type
PAPER_TYPE_LABELS = {
    'Empirical Research on Management': 'Evidence',
    'Technical or machine learning paper': 'Evidence',
    'Qualitative or case study': 'Qualitative',
    'Theoretical or conceptual paper': 'Theory',
    'Review or opinion paper': 'Theory',
    'Review Paper': 'Summary',
    'not recognized': 'Mix',
}

# Rule Base applicable_type Vocabulary → Allowed paper types. None means no limit.
APPLICABLE_TYPE_ALIASES: dict[str, set[str] | None] = {
    'General': None,
    'Evidence': {'Evidence', 'Mix'},
    'Quantitative': {'Evidence', 'Mix'},
    'Qualitative': {'Qualitative', 'Mix'},
    'Theory': {'Theory', 'Mix'},
    'Comments': {'Theory', 'Mix'},
    'Summary': {'Summary'},
    'Mix': {'Mix'},
}

PAPER_TYPE_KEYWORDS = [
    ('Summary', ['Summary', 'meta-analysis', 'Literature review', "review", "meta-analysis", "systematic literature"]),
    (
        'Qualitative',
        ['Qualitative', 'Case Study', 'Interview', 'Take root', 'Ethnography', "qualitative", "case study", "interview", "ethnograph", "grounded theory"],
    ),
    (
        'Theory',
        ['Theory', 'concept', 'proposition', 'Views', 'Comments', "theoretical", "conceptual", "proposition", "commentary", "viewpoint", "essay"],
    ),
    (
        'Evidence',
        [
            'Evidence', 'Quantitative', 'Questionnaire', 'Return', 'Experiment', 'sample', 'Machine Learning', 'algorithm', 'Technology',
            "empirical", "quantitative", "survey", "regression", "experiment", "panel", "machine learning", "benchmark",
        ],
    ),
    ('Mix', ['Mix', "mixed method", "mixed-method"]),
]


def normalize_paper_type(value: Any, fallback: str | None = 'Mix') -> str | None:
    'Converge free text paper types into controlled enumerations; return fallback when unrecognized.'
    text = str(value or "").strip()
    if not text:
        return fallback
    if text in PAPER_TYPE_CLASSES:
        return text
    if text in PAPER_TYPE_LABELS:
        return PAPER_TYPE_LABELS[text]
    lowered = text.lower()
    hits = {name for name, markers in PAPER_TYPE_KEYWORDS if any(marker in lowered for marker in markers)}
    if 'Mix' in hits or {'Evidence', 'Qualitative'} <= hits:
        return 'Mix'
    for name in ('Summary', 'Qualitative', 'Theory', 'Evidence'):
        if name in hits:
            return name
    return fallback


def rule_in_scope(applicable_type: str, paper_type: str | None) -> bool:
    'Whether the rules apply to this paper type.\n\n    applicable_type allows to be written as multiple values \u200b\u200b(such as "theory/overview/mixed" and "quantitative/mixed"), split by delimiters, normalize item by item and then take the union.\n    Values \u200b\u200bnot included in the vocabulary or written as "general" will be treated as unrestricted to avoid accidental injury.'
    if not paper_type:
        return True
    text = str(applicable_type).strip()
    if not text:
        return True
    allowed: set[str] = set()
    for token in (part.strip() for part in re.split(r"[/、,，|]", text)):
        if not token:
            continue
        if token == 'General':
            return True
        values = APPLICABLE_TYPE_ALIASES.get(token)
        if values is None:
            # Not included in the word list: treated as unrestricted to avoid excluding all rules just because a new word is written.
            return True
        allowed |= values
    if not allowed:
        return True
    return paper_type in allowed


def out_of_scope_rule_ids(profile: RuleProfile, paper_type: str | None) -> set[str]:
    return {rule.rule_id for rule in profile.rules if not rule_in_scope(rule.applicable_type, paper_type)}


def load_rule_profile(path: str | Path) -> RuleProfile:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    dimensions = [DimensionRule(**item) for item in data["dimensions"]]
    rules = [IndicatorRule(**item) for item in data["rules"]]
    constructs = [ConstructRule(**item) for item in data.get("constructs") or []]
    pools = data.get("scoring_pools") or {}
    budget = data.get("evidence_budget") or {}
    return RuleProfile(
        rule_profile_id=data["rule_profile_id"],
        rule_profile_name=data["rule_profile_name"],
        version=data["version"],
        status=data["status"],
        scoring_scale=float(data["scoring_scale"]),
        dimensions=dimensions,
        rules=rules,
        constructs=constructs,
        baseline_pool=float(pools.get("baseline") or 0.0),
        bonus_pool=float(pools.get("bonus") or 0.0),
        max_evidence_per_rule=int(budget["max_evidence_per_rule"]) if budget.get("max_evidence_per_rule") else None,
    )


def validate_rule_profile(profile: RuleProfile) -> list[str]:
    'Strict verification: If the weight of any layer is not equal, any rule has no attribution, or the two pools are not equal, an error will be returned, and the harness will directly refuse to run.'
    errors: list[str] = []
    dimension_weights = {d.dimension_id: d.weight for d in profile.dimensions}
    rule_dimension_sums = {d.dimension_id: 0.0 for d in profile.dimensions}
    seen: set[str] = set()

    # Two pool scale: base pool + bonus pool = total score. The benchmark pool must exist, otherwise "passing means full score" will not fall into place.
    if abs((profile.baseline_pool + profile.bonus_pool) - profile.scoring_scale) > 1e-9:
        errors.append(
            f"The two pools are not equal: the base pool {profile.baseline_pool} + the bonus pool {profile.bonus_pool}"
            f"!= Total score {profile.scoring_scale}"
        )
    if profile.baseline_pool <= 0:
        errors.append('The benchmark pool must be greater than 0: the benchmark score is the landing point for "reaching publication level"')
    if profile.bonus_pool <= 0:
        errors.append('The bonus point pool must be greater than 0: without the bonus point pool, it is impossible to distinguish between good and bad papers.')

    for rule in profile.rules:
        if rule.rule_id in seen:
            errors.append(f"Duplicate rule_id: {rule.rule_id}")
        seen.add(rule.rule_id)
        if rule.role not in RULE_ROLES:
            errors.append(f"{rule.rule_id} has unknown role: {rule.role}")
            continue
        required_keys = REQUIRED_JUDGEMENT_KEYS_BY_ROLE[rule.role]
        for key in required_keys:
            if key not in rule.judgement_values:
                errors.append(f"{rule.rule_id} ({rule.role}) missing judgement value: {key}")
            if key not in rule.score_mapping:
                errors.append(f"{rule.rule_id} ({rule.role}) missing score mapping: {key}")
        # The benchmark items are not judged as excellent: adding "satisfied" will turn "passing means full score" into a two-level competition again.
        if rule.role == "baseline" and "satisfied" in rule.score_mapping:
            errors.append(f"{rule.rule_id} is a benchmark item, and the satisfied score should not be declared (passed means full score)")
        # The bonus points must be satisfied, otherwise there will be no room for bonus points.
        if rule.role == "bonus" and "satisfied" not in rule.score_mapping:
            errors.append(f"{rule.rule_id} is a bonus item and the satisfied score must be declared")
        if rule.dimension_id not in dimension_weights:
            errors.append(f"Unknown dimension_id for {rule.rule_id}: {rule.dimension_id}")
            continue
        if rule.weight < 0:
            errors.append(f"{rule.rule_id} has negative weight: {rule.weight}")
        rule_dimension_sums[rule.dimension_id] += float(rule.weight)

    if not any(rule.role == "baseline" and rule.weight > 0 for rule in profile.rules):
        errors.append('There is no weighted benchmark item rule')
    if not any(rule.role == "bonus" and rule.weight > 0 for rule in profile.rules):
        errors.append('does not have any weighted bonus points rules')

    if abs(sum(dimension_weights.values()) - profile.scoring_scale) > 1e-9:
        errors.append("Dimension weights do not sum to scoring scale")

    if not profile.constructs:
        # There is no construct layer: Return to "Each rule constitutes its own construct". At this time, the rule weight must be directly added to the dimension weight.
        for dimension_id, expected in dimension_weights.items():
            actual = rule_dimension_sums.get(dimension_id, 0.0)
            if abs(actual - expected) > 1e-9:
                errors.append(f"Dimension {dimension_id} rule weight mismatch: expected {expected}, got {actual}")
        return errors

    construct_by_id: dict[str, ConstructRule] = {}
    for construct in profile.constructs:
        if construct.construct_id in construct_by_id:
            errors.append(f"Duplicate construct_id: {construct.construct_id}")
        if construct.dimension_id not in dimension_weights:
            errors.append(f"Unknown dimension_id for construct {construct.construct_id}: {construct.dimension_id}")
        if construct.weight < 0:
            errors.append(f"Construct {construct.construct_id} has negative weight: {construct.weight}")
        construct_by_id[construct.construct_id] = construct

    member_weight = {cid: 0.0 for cid in construct_by_id}
    member_count = {cid: 0 for cid in construct_by_id}
    member_roles: dict[str, set[str]] = {cid: set() for cid in construct_by_id}
    for rule in profile.rules:
        construct_id = rule.cluster
        if not construct_id:
            errors.append(f"{rule.rule_id} has no construct (cluster) assigned")
            continue
        construct = construct_by_id.get(construct_id)
        if construct is None:
            errors.append(f"{rule.rule_id} references unknown construct: {construct_id}")
            continue
        if construct.dimension_id != rule.dimension_id:
            errors.append(
                f"{rule.rule_id} dimension {rule.dimension_id} does not match construct "
                f"{construct_id} dimension {construct.dimension_id}"
            )
        member_weight[construct_id] += float(rule.weight)
        member_count[construct_id] += 1
        member_roles[construct_id].add(rule.role)

    # The two pools are aggregated according to constructs (constructs are the actual allocation units of weights), so a construct cannot contain both benchmark items and bonus items.
    # Otherwise, it is unclear which pool the score of this construct should be included in.
    for construct_id, roles in member_roles.items():
        if len(roles) > 1:
            errors.append(f"Construct {construct_id} contains both {'/'.join(sorted(roles))} characters and cannot be returned to the pool")

    for construct_id, construct in construct_by_id.items():
        if member_count[construct_id] == 0:
            errors.append(f"Construct {construct_id} has no rules")
            continue
        if abs(member_weight[construct_id] - construct.weight) > 1e-9:
            errors.append(
                f"Construct {construct_id} rule weight sum {member_weight[construct_id]} "
                f"!= construct weight {construct.weight}"
            )
        if construct.weight == 0 and member_weight[construct_id] != 0:
            errors.append(f"Construct {construct_id} has zero weight but its rules carry weight")

    dimension_construct_sums = {d.dimension_id: 0.0 for d in profile.dimensions}
    for construct in profile.constructs:
        if construct.dimension_id in dimension_construct_sums:
            dimension_construct_sums[construct.dimension_id] += float(construct.weight)
    for dimension_id, expected in dimension_weights.items():
        actual = dimension_construct_sums.get(dimension_id, 0.0)
        if abs(actual - expected) > 1e-9:
            errors.append(f"Dimension {dimension_id} construct weight mismatch: expected {expected}, got {actual}")

    # The full score of the dimension is calculated by dividing the two pools (pool × the weight of the rule of the dimension / the total weight of the pool), instead of directly taking the dimension weight;
    # Only when each dimension satisfies the baseline rule weight = 0.6 × dimension weight, the apportionment result will be equal to the declared dimension weight.
    # If this is not stuck, changing the weight will silently produce a misalignment such as "B full score 25.26 / statement 20".
    if profile.baseline_pool > 0:
        baseline_share = profile.baseline_pool / profile.scoring_scale
        dimension_baseline_sums = {d.dimension_id: 0.0 for d in profile.dimensions}
        for rule in profile.rules:
            if rule.role == "baseline" and rule.dimension_id in dimension_baseline_sums:
                dimension_baseline_sums[rule.dimension_id] += float(rule.weight)
        for dimension_id, dimension_weight in dimension_weights.items():
            expected = dimension_weight * baseline_share
            actual = dimension_baseline_sums.get(dimension_id, 0.0)
            if abs(actual - expected) > 1e-9:
                errors.append(
                    f"Dimension {dimension_id} Benchmark rule weight {actual} != Dimension weight × Benchmark pool proportion {expected}"
                    f"(Each dimension must have benchmark: bonus score = 6:4, otherwise the full score of this dimension is not equal to the declared {dimension_weight})"
                )

    return errors

