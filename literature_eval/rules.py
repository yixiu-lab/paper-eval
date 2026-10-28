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
    # 规则角色：baseline = 合格性门槛（合格即满分，只判缺口）；bonus = 加分项（只有 satisfied 才拿满）。
    # 已发表论文在基准项上默认合格，所以差异只应来自加分项——这是 V11 的核心机制。
    role: str = "baseline"
    # 同一构念的多条规则共用一个 cluster 名；汇总时一个 cluster 只算一个条目，
    # 避免同一构念被重复计权。缺省即自成一条。
    cluster: str | None = None
    # 规则说明：为什么设这条、什么算够、什么算不够；以及判定 satisfied / mostly 的证据门槛。
    # 这些会进入报告，让读者看懂分数是怎么来的。
    rationale: str | None = None
    threshold: str | None = None
    mostly_threshold: str | None = None
    # mostly 档的容差说明：哪些常见缺口不阻断 mostly，防止梯子中间档被架空。
    mostly_note: str | None = None
    # 该规则的**英文**检索词。规则锚点是中文、被评论文是英文，直接拿中文去搜英文正文恒不命中
    # （见 rule_keyword_hints 的旧实现）。这里显式声明英文词，供证据检索（词袋召回 + LLM 重排）使用。
    retrieval_terms: list[str] = field(default_factory=list)

    @property
    def cluster_key(self) -> str:
        return self.cluster or self.rule_id


@dataclass(frozen=True)
class ConstructRule:
    """构念层：维度权重的实际分配单位。同一构念下的规则测量的是同一件事。"""

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
    # 两池标尺：基准池代表「达到发表水准」的基准分，加分池是超过合格才能拿到的分数。
    # 两者相加必须等于 scoring_scale，由 validate_rule_profile 强制。
    baseline_pool: float = 0.0
    bonus_pool: float = 0.0
    # 证据预算：分数绝对值随每条规则的证据条数变化（实测 10→20 段整体上移约 7 分），
    # 所以预算必须随规则库一起版本化，不能悄悄改。Harness 会比对声明值与实际生效值。
    max_evidence_per_rule: int | None = None

    @property
    def rules_by_id(self) -> dict[str, IndicatorRule]:
        return {rule.rule_id: rule for rule in self.rules}

    @property
    def constructs_by_id(self) -> dict[str, ConstructRule]:
        return {item.construct_id: item for item in self.constructs}


# 论文类型受控枚举。规则适用性只认这 5 类，不再用自由文本匹配。
PAPER_TYPE_CLASSES = ("实证", "质性", "理论", "综述", "混合")

# 规则角色白名单。
RULE_ROLES = ("baseline", "bonus")

# 基准项不判「优秀」，所以不设 satisfied；加分项必须有 satisfied 才有加分空间。
REQUIRED_JUDGEMENT_KEYS_BY_ROLE = {
    "baseline": ("mostly", "partial", "weak", "not_satisfied", "uncertain"),
    "bonus": ("satisfied", "mostly", "partial", "weak", "not_satisfied", "uncertain"),
}
# 兼容旧引用：完整档位集合。
REQUIRED_JUDGEMENT_KEYS = REQUIRED_JUDGEMENT_KEYS_BY_ROLE["bonus"]

# 旧的描述性标签 → 受控类型
PAPER_TYPE_LABELS = {
    "管理学实证研究": "实证",
    "技术或机器学习论文": "实证",
    "质性或案例研究": "质性",
    "理论或概念论文": "理论",
    "评论或观点论文": "理论",
    "综述论文": "综述",
    "未识别": "混合",
}

# 规则库 applicable_type 词表 → 允许的论文类型。None 表示不限。
APPLICABLE_TYPE_ALIASES: dict[str, set[str] | None] = {
    "通用": None,
    "实证": {"实证", "混合"},
    "定量": {"实证", "混合"},
    "质性": {"质性", "混合"},
    "理论": {"理论", "混合"},
    "评论": {"理论", "混合"},
    "综述": {"综述"},
    "混合": {"混合"},
}

PAPER_TYPE_KEYWORDS = [
    ("综述", ["综述", "元分析", "文献回顾", "review", "meta-analysis", "systematic literature"]),
    (
        "质性",
        ["质性", "案例研究", "访谈", "扎根", "民族志", "qualitative", "case study", "interview", "ethnograph", "grounded theory"],
    ),
    (
        "理论",
        ["理论", "概念", "命题", "观点", "评论", "theoretical", "conceptual", "proposition", "commentary", "viewpoint", "essay"],
    ),
    (
        "实证",
        [
            "实证", "定量", "问卷", "回归", "实验", "样本", "机器学习", "算法", "技术",
            "empirical", "quantitative", "survey", "regression", "experiment", "panel", "machine learning", "benchmark",
        ],
    ),
    ("混合", ["混合", "mixed method", "mixed-method"]),
]


def normalize_paper_type(value: Any, fallback: str | None = "混合") -> str | None:
    """把自由文本论文类型收敛成受控枚举；无法识别时返回 fallback。"""
    text = str(value or "").strip()
    if not text:
        return fallback
    if text in PAPER_TYPE_CLASSES:
        return text
    if text in PAPER_TYPE_LABELS:
        return PAPER_TYPE_LABELS[text]
    lowered = text.lower()
    hits = {name for name, markers in PAPER_TYPE_KEYWORDS if any(marker in lowered for marker in markers)}
    if "混合" in hits or {"实证", "质性"} <= hits:
        return "混合"
    for name in ("综述", "质性", "理论", "实证"):
        if name in hits:
            return name
    return fallback


def rule_in_scope(applicable_type: str, paper_type: str | None) -> bool:
    """规则是否适用于该论文类型。

    applicable_type 允许写成多值（如「理论/综述/混合」「定量/混合」），按分隔符拆开逐项归一再取并集。
    词表未收录的取值或写成「通用」都按不限处理，避免误伤。
    """
    if not paper_type:
        return True
    text = str(applicable_type).strip()
    if not text:
        return True
    allowed: set[str] = set()
    for token in (part.strip() for part in re.split(r"[/、,，|]", text)):
        if not token:
            continue
        if token == "通用":
            return True
        values = APPLICABLE_TYPE_ALIASES.get(token)
        if values is None:
            # 未收录词表：按不限处理，避免因为写了个新词就把规则全部排除
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
    """严格校验：任何一层权重加不平、任何规则没有归属、两池加不平，都返回错误，harness 直接拒绝运行。"""
    errors: list[str] = []
    dimension_weights = {d.dimension_id: d.weight for d in profile.dimensions}
    rule_dimension_sums = {d.dimension_id: 0.0 for d in profile.dimensions}
    seen: set[str] = set()

    # 两池标尺：基准池 + 加分池 = 总分。基准池必须存在，否则「合格即满分」没有落点。
    if abs((profile.baseline_pool + profile.bonus_pool) - profile.scoring_scale) > 1e-9:
        errors.append(
            f"两池加不平：基准池 {profile.baseline_pool} + 加分池 {profile.bonus_pool} "
            f"!= 总分 {profile.scoring_scale}"
        )
    if profile.baseline_pool <= 0:
        errors.append("基准池必须大于 0：基准分是「达到发表水准」的落点")
    if profile.bonus_pool <= 0:
        errors.append("加分池必须大于 0：没有加分池就无法区分论文优劣")

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
        # 基准项不判优秀：带 satisfied 会让「合格即满分」重新变成两档竞争。
        if rule.role == "baseline" and "satisfied" in rule.score_mapping:
            errors.append(f"{rule.rule_id} 是基准项，不应声明 satisfied 分值（合格即满分）")
        # 加分项必须有 satisfied，否则没有加分空间。
        if rule.role == "bonus" and "satisfied" not in rule.score_mapping:
            errors.append(f"{rule.rule_id} 是加分项，必须声明 satisfied 分值")
        if rule.dimension_id not in dimension_weights:
            errors.append(f"Unknown dimension_id for {rule.rule_id}: {rule.dimension_id}")
            continue
        if rule.weight < 0:
            errors.append(f"{rule.rule_id} has negative weight: {rule.weight}")
        rule_dimension_sums[rule.dimension_id] += float(rule.weight)

    if not any(rule.role == "baseline" and rule.weight > 0 for rule in profile.rules):
        errors.append("没有任何有權重的基准项规则")
    if not any(rule.role == "bonus" and rule.weight > 0 for rule in profile.rules):
        errors.append("没有任何有權重的加分项规则")

    if abs(sum(dimension_weights.values()) - profile.scoring_scale) > 1e-9:
        errors.append("Dimension weights do not sum to scoring scale")

    if not profile.constructs:
        # 没有构念层：退回“每条规则自成构念”，此时规则权重必须直接加平到维度权重。
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

    # 两池按构念归集（构念是权重的实际分配单位），所以一个构念不能同时包含基准项与加分项，
    # 否则这个构念的得分该算进哪个池就说不清了。
    for construct_id, roles in member_roles.items():
        if len(roles) > 1:
            errors.append(f"Construct {construct_id} 同时包含 {'/'.join(sorted(roles))} 两种角色，无法归池")

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

    # 维度满分由两池摊分算出（pool × 该维规则权重 / 池总权重），不是直接取维度权重；
    # 只有每维都满足 基准规则权重 = 0.6 × 维度权重 时，摊分结果才等于声明的维度权重。
    # 不卡这一条，改权重就会静默产出「B 满分 25.26 / 声明 20」这类错位。
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
                    f"Dimension {dimension_id} 基准规则权重 {actual} != 维度权重 × 基准池占比 {expected}"
                    f"（每维必须 基准:加分 = 6:4，否则该维度满分不等于声明的 {dimension_weight}）"
                )

    return errors

