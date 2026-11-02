'Certainty scoring scale: two-pool mechanism (base score + bonus points).\n\nCore mechanism (starting from V11, starting from V12, the two pools are set to 60/40):\n\n    Total score = Benchmark pool × Benchmark achievement rate + Bonus point pool × Bonus points = 60 × Benchmark achievement rate + 40 × Bonus points\n\n- **Benchmark pool 60 points** measures "adequate publication level". Published papers are qualified by default on the benchmark items, so **passing means getting full**:\n  `mostly` (qualified with no obvious gaps) and `partial` (qualified with pointed gaps) are both given 1.0;\n  Baseline points are deducted only for `weak` (not reaching publication level, 0.5) and `not_satisfied` (serious defect, 0.0).\n  The benchmark items are not set `satisfied` - the benchmark items are not judged as excellent. A paper that is qualified in all aspects is exactly 60 points, which is consistent with "60 points passing".\n- **Extra point pool 40 points** measures "whether it exceeds the qualification": `satisfied` 1.0, `mostly` 0.5, `partial` 0.25,\n  `weak`/`not_satisfied` 0.0. Bonus points are only added, not deducted. After the bonus point pool was raised from 30 to 40,\n  **The full score of dimension B (innovation and theoretical contribution) increased from 19.4 to 25.9** - because B is composed entirely of bonus points.\n- `uncertain` does not enter the numerator or denominator (loose caliber); constructs outside the applicable scope are eliminated as a whole.\n\nWhy changed to two pools: 8 reports and 272 judgments under the five-speed isometric scale, `partial` accounts for 55–61%, 27/34 rules\nNever got `satisfied`, three papers with very different methods got exactly the same bracket with 35% weight. The root cause is not that the model is lazy.\nInstead, the instruction defines `partial` as "reaching the minimum acceptable level" (i.e., passing), but the score is recorded as 60/100 - a semantic and arithmetic contradiction,\nA published paper that is qualified in all respects can only count for 62–70 points. The two-pool mechanism separates "qualification judgment" and "merit and inferiority judgment".\nLet the model only make benchmark judgments within its own reliable range (passed/gap/not up to standard).\n\nThe absolute value of the score depends on the evidence budget (in actual testing, 10→20 paragraphs for each rule will move the total score up by about 7 points), so the evidence budget changes with the rule base\nVersioned together (`evidence_budget`), the version must be upgraded to change the budget.'

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .models import AgentJudgement, Judgement, ScoreItem
from .rules import RuleProfile


# Benchmark pool grade score: full score if qualified, and will be deducted only if the standard is obviously not met.
BASELINE_MAPPING = {
    Judgement.MOSTLY: 1.0,          # qualified and no obvious gaps
    Judgement.PARTIAL: 1.0,         # Qualified but with deficiencies that can be pointed out - still qualified, no points deducted
    Judgement.WEAK: 0.5,            # Not up to publication level
    Judgement.NOT_SATISFIED: 0.0,   # Serious defect
    Judgement.SATISFIED: 1.0,       # Baseline item should not appear; defensive mapping is full score
    Judgement.UNCERTAIN: None,
}

# Bonus point pool gear points: Only those who exceed the qualification level will get points.
BONUS_MAPPING = {
    Judgement.SATISFIED: 1.0,       # obviously exceeds
    Judgement.MOSTLY: 0.5,          # Achieving but not outstanding
    Judgement.PARTIAL: 0.25,        # Only related
    Judgement.WEAK: 0.0,            # No bonus points (no extra points deduction)
    Judgement.NOT_SATISFIED: 0.0,
    Judgement.UNCERTAIN: None,
}

ROLE_MAPPINGS = {"baseline": BASELINE_MAPPING, "bonus": BONUS_MAPPING}
POOLS = ("baseline", "bonus")

# The gear that the rule base must declare. There is no satisfied basis for the benchmark items (passing means full marks, not excellence).
REQUIRED_KEYS_BY_ROLE = {
    "baseline": ("mostly", "partial", "weak", "not_satisfied", "uncertain"),
    "bonus": ("satisfied", "mostly", "partial", "weak", "not_satisfied", "uncertain"),
}

# Top publication jump threshold: the benchmark items must not have shortcomings, and the bonus items must be significantly exceeded.
# The old standard of "each dimension reaches 70% of the dimension weight" was set on a five-level equidistant scale, and the semantics under the two-pool mechanism are no longer applicable.
#
# Empirical threshold calibration (2026-09-28). Read the criteria with their limits:
# - The project has no gold-standard labels for this threshold; values were selected for repeat-run stability.
# - Evidence comes from four batches of repeated runs on three AI-HRM empirical papers.
# - Observed bonus-rate ranges: Choudhury 0.784–0.916, Blaurock 0.491–0.541, Cheng 0.427–0.455.
# - The 0.65 / 0.30 thresholds provide a minimum margin of 0.109 (about 4.4 points).
# - This margin remains below the between-batch variation for an individual paper; report ranges for repeated runs.
# - The measured bonus point range: Choudhury 0.784–0.916, Blaurock 0.491–0.541, Cheng 0.427–0.455.
#   Old value 0.60 / 0.35 Although it is a 0 flip, it is only 0.059 (≈2.4 points) away from the Blaurock upper bound and will be pushed across the line by noise at any time.
#   0.65 / 0.30 Raise the minimum margin to 0.109 (≈4.4 points), which is the widest margin that this batch of data can achieve.
# - **Limitations**: The margin of 0.109 is still smaller than the single-article inter-batch fluctuation (Choudhury bonus point swing 0.131),
#   Therefore, the "high/medium potential" determination of a single batch is inherently unreliable. To draw conclusions, at least two batches were run, and the report should be accompanied by the inter-batch range.
BONUS_GATE_HIGH = 0.65
BONUS_GATE_MID = 0.30
# Allowable short board ratio for benchmark items: 0 means any weak/not_satisfied blocks "high".
MAX_BASELINE_SHORTFALL_FOR_HIGH = 0.0
# The standard shortcomings allowed by the "mid" range.
MAX_BASELINE_SHORTFALL_FOR_MID = 0.05

# When the completeness is lower than this ratio, no conclusion on high-top publication potential will be made (the score is loose, but the missing evidence will not be endorsed).
TOP_JOURNAL_MIN_COMPLETENESS = 0.8

# ruler version: If you change the above mapping or the caliber of the two pools, you must upgrade this number.
SCALE_VERSION = "MGMT-SCALE-V6"
SCALE_ANCHOR = 'Two-pool scale: 60 points in the benchmark pool (full marks will be awarded if the benchmark items are passed, only gaps will be judged) + 40 points in the bonus pool (only satisfied will get full bonus points)'

# grade band: benchmark score 60 = qualified in all aspects = reaching publication level; lower than it means there are shortcomings in the benchmark items.
#
# Grade thresholds were selected from the archived repeated runs. Previous values (93, 85, 75) were unstable:
# Choudhury scores across four batches were 91.375 / 92.313 / 93.437 / 96.625.
# rides on 93, and the level flips between "top issue level ↔ strong issue/top issue edge".
# is selected according to "as far as possible from the total scores of all papers and all batches" (there are 3454 candidates for 0 flipping, and the middle of the platform must be selected instead of the edge):
#   Choudhury [91.375, 96.625]｜Blaurock [79.625, 81.625]｜Cheng [77.078, 78.203]
#   → 88 falls in the middle of the gap (81.625, 91.375); 85 remains 3 points away from it; 74 is 3.08 points below the Cheng lower bound.
#   The minimum margin of three points is 3.0 points, which is the upper limit that this batch of data can achieve.
# - **Limitations (must be read together)**: A margin of 3.0 points** is less than Choudhury's own inter-batch variation of 5.25 points**,
#   So as long as the paper falls near a certain line, the grade of a single batch will be flipped. Levels should be considered "judgments with uncertainty":
#   To draw conclusions, at least two batches should be run; the report should give the range between batches, not the absolute value of one batch.
GRADE_BANDS = (
    (88.0, 'Top publication level'),
    (85.0, 'Strong issue/top issue edge'),
    (74.0, 'Good paper'),
    (60.0, 'reaches publication level'),
    (50.0, 'Weak but of certain value'),
)


def rule_score_mapping(rule: Any) -> dict[str, float | None]:
    'gear → score. The single source of truth is the score_mapping of the rule base; when missing, the code default scale is returned by role.\n\n    The default scale for benchmark items **does not include** satisfied—passing means full marks, and benchmark items are not considered excellent.'
    role = str(getattr(rule, "role", "baseline"))
    mapping = {str(key): (None if value is None else float(value)) for key, value in (rule.score_mapping or {}).items()}
    required = REQUIRED_KEYS_BY_ROLE.get(role, REQUIRED_KEYS_BY_ROLE["baseline"])
    if all(key in mapping for key in required):
        return mapping
    default = dict(ROLE_MAPPINGS.get(role, BASELINE_MAPPING))
    if role == "baseline":
        default.pop(Judgement.SATISFIED, None)
    return {key.value: value for key, value in default.items()}


def lookup_score(rule: Any, judgement: Judgement) -> float | None:
    'takes the score of a certain gear.\n\n    When the rule base does not declare this gear (for example, the benchmark item is satisfied), defensive mapping is performed according to the role.\n    Instead of treating it as "not rated" - the latter will squeeze it out of the denominator and instead amplify the weight of the remaining weights.'
    mapping = rule_score_mapping(rule)
    if judgement.value in mapping:
        return mapping[judgement.value]
    if judgement == Judgement.UNCERTAIN:
        return None
    return ROLE_MAPPINGS.get(str(getattr(rule, "role", "baseline")), BASELINE_MAPPING).get(judgement)


def score_judgements(
    profile: RuleProfile,
    judgements: list[AgentJudgement],
    out_of_scope_ids: set[str] | None = None,
) -> list[ScoreItem]:
    rules = profile.rules_by_id
    skipped = out_of_scope_ids or set()
    score_items: list[ScoreItem] = []
    for judgement in judgements:
        rule = rules[judgement.indicator_id]
        out_of_scope = rule.rule_id in skipped
        raw_score = None if out_of_scope else lookup_score(rule, judgement.judgement)
        weighted_score = None if raw_score is None else float(raw_score) * float(rule.weight)
        score_items.append(
            ScoreItem(
                indicator_id=rule.rule_id,
                judgement=judgement.judgement,
                rule_id=rule.rule_id,
                raw_score=raw_score,
                weighted_score=weighted_score,
                weight=float(rule.weight),
                dimension_id=rule.dimension_id,
                dimension_name=rule.dimension_name,
                # strict_* Deprecated: There is only one set of rulers, all determined by the rule base score_mapping.
                strict_raw_score=raw_score,
                strict_weighted_score=weighted_score,
                out_of_scope=out_of_scope,
            )
        )
    return score_items


def score_judgements_with_scheme(
    profile: RuleProfile,
    judgements: list[AgentJudgement],
    scheme: dict[str, Any],
    out_of_scope_ids: set[str] | None = None,
) -> list[ScoreItem]:
    'Use ** to replace the gear mapping** and recalculate the points for calibration sensitivity detection of the Reliability Layer.\n\n    This is a read-only entry for `reliability_diagnosis.calibration_uncertainty()`:\n    It does not change `score_judgements`, the rule base, or any existing callers. It just replaces the same batch of judgments.\n    Change the mapping and calculate the points again. Agent does not rerun - the judgment record is ready.\n\n    `scheme` is in the shape of `{"baseline": {...}, "bonus": {...}}`, each item is gear → score (None means not to enter the denominator).\n    Undeclared gears are returned to the `ROLE_MAPPINGS` default scale by role, which is consistent with the defense caliber of `lookup_score`.'
    rules = profile.rules_by_id
    skipped = out_of_scope_ids or set()
    score_items: list[ScoreItem] = []
    for judgement in judgements:
        rule = rules[judgement.indicator_id]
        role = str(getattr(rule, "role", "baseline"))
        mapping = scheme.get(role) or {}
        if judgement.judgement.value in mapping:
            raw_score = mapping[judgement.judgement.value]
        elif judgement.judgement == Judgement.UNCERTAIN:
            raw_score = None
        else:
            raw_score = ROLE_MAPPINGS.get(role, BASELINE_MAPPING).get(judgement.judgement)
        if rule.rule_id in skipped:
            raw_score = None
        weighted_score = None if raw_score is None else float(raw_score) * float(rule.weight)
        score_items.append(
            ScoreItem(
                indicator_id=rule.rule_id,
                judgement=judgement.judgement,
                rule_id=rule.rule_id,
                raw_score=raw_score,
                weighted_score=weighted_score,
                weight=float(rule.weight),
                dimension_id=rule.dimension_id,
                dimension_name=rule.dimension_name,
                strict_raw_score=raw_score,
                strict_weighted_score=weighted_score,
                out_of_scope=rule.rule_id in skipped,
            )
        )
    return score_items


def classify_quality(
    score: float | None,
    baseline_rate: float,
    bonus_rate: float,
    completeness: float = 1.0,
) -> dict[str, Any]:
    'bin. The score caliber is loose, but when the completeness is insufficient, no judgment on the potential of high-ranking publications will be given to avoid missing evidence being read as excellent.'
    if score is None:
        return {
            "quality_grade": 'Unable to determine',
            "top_journal_potential": 'Unknown',
            "top_journal_gate_pass": False,
            "top_journal_gate_reasons": ['The effective scoring weight is 0 and a quality grade cannot be formed.'],
        }
    grade = 'has many problems'
    for threshold, name in GRADE_BANDS:
        if score >= threshold:
            grade = name
            break

    complete_enough = completeness >= TOP_JOURNAL_MIN_COMPLETENESS
    shortfall = 1.0 - baseline_rate
    reasons: list[str] = []

    gate_pass = (
        complete_enough
        and shortfall <= MAX_BASELINE_SHORTFALL_FOR_HIGH
        and bonus_rate >= BONUS_GATE_HIGH
    )
    if gate_pass:
        potential = 'high'
    elif complete_enough and shortfall <= MAX_BASELINE_SHORTFALL_FOR_MID and bonus_rate >= BONUS_GATE_MID:
        potential = 'Medium'
    else:
        potential = 'low'

    if shortfall > MAX_BASELINE_SHORTFALL_FOR_HIGH:
        reasons.append(
            f"There are shortcomings in the benchmark items (Benchmark achievement rate {baseline_rate:.0%}: Some rules are judged to have not reached the publication level or have serious defects)"
        )
    if bonus_rate < BONUS_GATE_MID:
        reasons.append(f"has a low bonus score ({bonus_rate:.0%}, lower than {BONUS_GATE_MID:.0%})")
    elif bonus_rate < BONUS_GATE_HIGH:
        reasons.append(f"bonus points do not reach the top publication threshold ({bonus_rate:.0%}, {BONUS_GATE_HIGH:.0%} is required)")
    if not complete_enough:
        reasons = [
            f"rating completeness {completeness:.0%} is lower than {TOP_JOURNAL_MIN_COMPLETENESS:.0%},"
            'This time is not enough to judge the top publication potential.',
            *reasons,
        ]
        # The score caliber remains loose, but high score levels must carry a completeness reminder to avoid missing evidence being read as excellent.
        if score >= 86:
            grade = f"{grade} (completeness {completeness:.0%}, conclusion to be reviewed)"
    return {
        "quality_grade": grade,
        "top_journal_potential": potential,
        "top_journal_gate_pass": gate_pass,
        "top_journal_gate_reasons": reasons,
    }


def profile_score_mapping(profile: RuleProfile) -> dict[str, float | None]:
    for rule in profile.rules:
        mapping = rule_score_mapping(rule)
        if mapping:
            return mapping
    return {key.value: value for key, value in BASELINE_MAPPING.items()}


def construct_allocation(profile: RuleProfile) -> dict[str, list[tuple[str, float, list[str]]]]:
    'returns {dimension_id: [(construct_id, construct_weight, [rule_id...])]}.\n\n    The weights are all taken from the rule base declaration; if there is no constructs layer, it returns to "each rule constitutes its own construct", and the weight is taken from the rule weight.'
    rules_by_construct: dict[str, list[str]] = {}
    for rule in profile.rules:
        rules_by_construct.setdefault(rule.cluster_key, []).append(rule.rule_id)

    if profile.constructs:
        allocation: dict[str, list[tuple[str, float, list[str]]]] = {d.dimension_id: [] for d in profile.dimensions}
        for construct in profile.constructs:
            members = rules_by_construct.get(construct.construct_id, [])
            allocation.setdefault(construct.dimension_id, []).append(
                (construct.construct_id, float(construct.weight), members)
            )
        return allocation

    allocation = {d.dimension_id: [] for d in profile.dimensions}
    for rule in profile.rules:
        allocation.setdefault(rule.dimension_id, []).append((rule.rule_id, float(rule.weight), [rule.rule_id]))
    return allocation


def aggregate_scores(profile: RuleProfile, score_items: list[ScoreItem]) -> dict[str, Any]:
    'Summary of two pools: benchmark pool × benchmark achievement rate + bonus point pool × bonus points.\n\n    -Constructs are weight distribution units and are pooled according to roles (`validate_rule_profile` ensures that a construct does not span roles).\n    - Within the construct: The 0-1 ratio is obtained by weighting the average according to the rule share; construct ratio × construct weight = contribution of the construct.\n    - Loose caliber: only constructs that have been rated ** are weighted together, and constructs that are not rated are not included in the denominator; completeness is reported separately.\n    - When a certain pool cannot be rated as a whole, use another pool to reset the total score to full score and mark it with `pool_renormalized`.\n      Avoid artificially lowering scores due to missing an entire pool.\n    - At the same time, a strict lower bound is provided (the denominator is counted as 0 points if it cannot be judged) and an optimistic upper bound (the benchmark items are counted as qualified, and the bonus items are counted as only marginal).\n    - Constructs outside the applicable scope are not included in either the numerator or the denominator.'
    rules = profile.rules_by_id
    items_by_rule = {item.rule_id: item for item in score_items}
    allocation = construct_allocation(profile)
    total_weight = {
        pool: sum(float(rule.weight) for rule in profile.rules if rule.role == pool) for pool in POOLS
    }

    agg = {pool: {"earned": 0.0, "available": 0.0, "scope": 0.0, "missing": 0.0, "uncertain": 0.0} for pool in POOLS}
    out_of_scope_weight = 0.0
    by_dimension: dict[str, dict[str, Any]] = {}

    for dimension in profile.dimensions:
        dimension_id = dimension.dimension_id
        dim = {pool: {"earned": 0.0, "available": 0.0, "scope": 0.0, "uncertain": 0.0} for pool in POOLS}
        dimension_rule_weight = {pool: 0.0 for pool in POOLS}
        construct_count = 0
        available_construct_count = 0

        for _construct_id, construct_weight, member_rule_ids in allocation.get(dimension_id, []):
            construct_count += 1
            members = [items_by_rule[rule_id] for rule_id in member_rule_ids if rule_id in items_by_rule]
            # The full score of the dimension is allocated according to the full weight of the rule base, regardless of the paper type: otherwise the same dimension will display different full scores on different papers.
            for rule_id in member_rule_ids:
                member_rule = rules.get(rule_id)
                if member_rule is not None:
                    dimension_rule_weight[member_rule.role] += float(member_rule.weight)
            in_scope = [item for item in members if not item.out_of_scope]
            if not in_scope:
                out_of_scope_weight += construct_weight
                continue
            # The construct does not cross roles (forced by verification), just take the role of the first member
            pool = rules[in_scope[0].rule_id].role
            dim[pool]["scope"] += construct_weight
            agg[pool]["scope"] += construct_weight
            uncertain = sum(item.weight for item in in_scope if item.judgement == Judgement.UNCERTAIN)
            dim[pool]["uncertain"] += uncertain
            agg[pool]["uncertain"] += uncertain

            judged = [item for item in in_scope if item.raw_score is not None]
            weight_sum = sum(item.weight for item in judged)
            if not judged or weight_sum <= 0:
                agg[pool]["missing"] += construct_weight
                continue
            ratio = sum(float(item.raw_score) * item.weight for item in judged) / weight_sum
            dim[pool]["earned"] += ratio * construct_weight
            dim[pool]["available"] += construct_weight
            agg[pool]["earned"] += ratio * construct_weight
            agg[pool]["available"] += construct_weight
            available_construct_count += 1

        # The full score of the dimension is apportioned according to the weight ratio of the two pools, ensuring that Σ the full score of each dimension = the total score
        dimension_max = {
            pool: (profile.baseline_pool if pool == "baseline" else profile.bonus_pool)
            * (dimension_rule_weight[pool] / total_weight[pool] if total_weight[pool] > 0 else 0.0)
            for pool in POOLS
        }
        dimension_rate = {
            pool: (dim[pool]["earned"] / dim[pool]["available"] if dim[pool]["available"] > 0 else None)
            for pool in POOLS
        }
        available_total = dim["baseline"]["available"] + dim["bonus"]["available"]
        dimension_score = sum(
            dimension_max[pool] * (dimension_rate[pool] or 0.0) for pool in POOLS
        )
        by_dimension[dimension_id] = {
            "dimension_name": dimension.dimension_name,
            "max_score": dimension_max["baseline"] + dimension_max["bonus"],
            "baseline_max": dimension_max["baseline"],
            "bonus_max": dimension_max["bonus"],
            "baseline_rate": dimension_rate["baseline"],
            "bonus_rate": dimension_rate["bonus"],
            "scope_weight": dim["baseline"]["scope"] + dim["bonus"]["scope"],
            "available_weight": available_total,
            "earned_score": dimension_score,
            "strict_earned_score": dimension_score,
            "normalized_score": None if available_total <= 0 else dimension_score,
            "strict_normalized_score": None if available_total <= 0 else dimension_score,
            "uncertain_weight": dim["baseline"]["uncertain"] + dim["bonus"]["uncertain"],
            "cluster_count": construct_count,
            "available_cluster_count": available_construct_count,
        }

    judgement_counts: dict[str, int] = defaultdict(int)
    for item in score_items:
        judgement_counts[item.judgement.value] += 1

    scale = profile.scoring_scale
    baseline_available, bonus_available = agg["baseline"]["available"], agg["bonus"]["available"]
    baseline_rate = agg["baseline"]["earned"] / baseline_available if baseline_available > 0 else 0.0
    bonus_rate = agg["bonus"]["earned"] / bonus_available if bonus_available > 0 else 0.0

    pool_renormalized = False
    if baseline_available > 0 and bonus_available > 0:
        total_score = profile.baseline_pool * baseline_rate + profile.bonus_pool * bonus_rate
    elif baseline_available > 0:
        total_score = scale * baseline_rate
        pool_renormalized = True
    elif bonus_available > 0:
        total_score = scale * bonus_rate
        pool_renormalized = True
    else:
        total_score = None

    scope_baseline, scope_bonus = agg["baseline"]["scope"], agg["bonus"]["scope"]
    if scope_baseline > 0 or scope_bonus > 0:
        zero_b = agg["baseline"]["earned"] / scope_baseline if scope_baseline > 0 else 0.0
        zero_g = agg["bonus"]["earned"] / scope_bonus if scope_bonus > 0 else 0.0
        na_as_zero = profile.baseline_pool * zero_b + profile.bonus_pool * zero_g
        # Optimistic upper bound: Unrated basic items will be counted as qualified, and unrated bonus items will be counted as "only marginal"
        opt_b = (
            (agg["baseline"]["earned"] + agg["baseline"]["missing"] * 1.0) / scope_baseline
            if scope_baseline > 0 else 0.0
        )
        opt_g = (
            (agg["bonus"]["earned"] + agg["bonus"]["missing"] * 0.25) / scope_bonus
            if scope_bonus > 0 else 0.0
        )
        na_as_partial = min(scale, profile.baseline_pool * opt_b + profile.bonus_pool * opt_g)
    else:
        na_as_zero = na_as_partial = None

    completeness = (
        (baseline_available + bonus_available) / (scope_baseline + scope_bonus)
        if (scope_baseline + scope_bonus) > 0 else 0.0
    )
    classification = classify_quality(total_score, baseline_rate, bonus_rate, completeness=completeness)

    return {
        "scoring_mode": (
            f"Two-pool mechanism: Benchmark pool {profile.baseline_pool:.0f} points (full score if the benchmark item is passed)"
            f"+ bonus point pool {profile.bonus_pool:.0f} points (only satisfied can get full bonus points)"
        ),
        "scale_version": SCALE_VERSION,
        "scale_anchor": SCALE_ANCHOR,
        "total_score": total_score,
        "baseline_pool": profile.baseline_pool,
        "bonus_pool": profile.bonus_pool,
        "baseline_score": profile.baseline_pool * baseline_rate,
        "bonus_score": profile.bonus_pool * bonus_rate,
        "baseline_rate": baseline_rate,
        "bonus_rate": bonus_rate,
        "pool_renormalized": pool_renormalized,
        "strict_top_journal_reference_score": na_as_zero,
        "conservative_total_score": na_as_zero,
        "optimistic_total_score": na_as_partial,
        "na_as_zero_total_score": na_as_zero,
        "na_as_partial_total_score": na_as_partial,
        "raw_earned_score": agg["baseline"]["earned"] + agg["bonus"]["earned"],
        "strict_raw_earned_score": agg["baseline"]["earned"] + agg["bonus"]["earned"],
        "available_weight": baseline_available + bonus_available,
        "available_construct_weight": baseline_available + bonus_available,
        "scope_weight": scope_baseline + scope_bonus,
        "out_of_scope_weight": out_of_scope_weight,
        "missing_construct_weight": agg["baseline"]["missing"] + agg["bonus"]["missing"],
        "scoring_completeness": completeness,
        "uncertain_weight": agg["baseline"]["uncertain"] + agg["bonus"]["uncertain"],
        "dimension_scores": by_dimension,
        "judgement_counts": dict(judgement_counts),
        **classification,
    }
