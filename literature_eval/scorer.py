"""确定性评分标尺：两池机制（基准分 + 加分项）。

核心机制（V11 起，V12 起两池定为 60/40）：

    总分 = 基准池 × 基准达成率 + 加分池 × 加成分 = 60 × 基准达成率 + 40 × 加成分

- **基准池 60 分**衡量「够不够发表水准」。已发表论文在基准项上默认合格，所以**合格即拿满**：
  `mostly`（合格且无明显缺口）与 `partial`（合格但有可指出的缺口）都给 1.0；
  只有 `weak`（未达发表水准，0.5）和 `not_satisfied`（严重缺陷，0.0）才扣基准分。
  基准项不设 `satisfied`——基准项不判优秀。各方面都合格的论文正好 60 分，与「60 分及格」一致。
- **加分池 40 分**衡量「有没有超出合格」：`satisfied` 1.0、`mostly` 0.5、`partial` 0.25、
  `weak`/`not_satisfied` 0.0。加分项只加分、不扣分。加分池从 30 提到 40 后，
  **B 维度（创新性与理论贡献）满分从 19.4 升到 25.9**——因为 B 全部由加分项组成。
- `uncertain` 不进分子也不进分母（宽松口径）；适用范围外的构念整体剔除。

为什么改成两池：五档等距标尺下实测 8 份报告 272 条判定，`partial` 占 55–61%，27/34 条规则
从未拿到 `satisfied`，三篇方法差异极大的论文有 35% 的权重拿到完全相同的档位。根因不是模型偷懒，
而是指令把 `partial` 定义成「达到最低可接受水准」（即合格），分数却把它记成 60/100——语义与算术矛盾，
一篇各方面都合格的已发表论文只能算出 62–70 分。两池机制把「合格性判断」与「优劣判断」分开，
让模型只在自己可靠的档位区间（合格 / 有缺口 / 不达标）上做基准项判断。

分数绝对值依赖证据预算（实测每条规则 10→20 段会让总分整体上移约 7 分），所以证据预算随规则库
一起版本化（`evidence_budget`），换预算必须升版。
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .models import AgentJudgement, Judgement, ScoreItem
from .rules import RuleProfile


# 基准池档位分值：合格即满分，只有明显不达标才扣。
BASELINE_MAPPING = {
    Judgement.MOSTLY: 1.0,          # 合格且无明显缺口
    Judgement.PARTIAL: 1.0,         # 合格但有可指出的缺口——仍然合格，不扣分
    Judgement.WEAK: 0.5,            # 未达发表水准
    Judgement.NOT_SATISFIED: 0.0,   # 严重缺陷
    Judgement.SATISFIED: 1.0,       # 基准项本不该出现；防御性映射为满分
    Judgement.UNCERTAIN: None,
}

# 加分池档位分值：只有超出合格才拿得到分。
BONUS_MAPPING = {
    Judgement.SATISFIED: 1.0,       # 明显超出
    Judgement.MOSTLY: 0.5,          # 做到但不突出
    Judgement.PARTIAL: 0.25,        # 只沾边
    Judgement.WEAK: 0.0,            # 没有加分（不额外扣分）
    Judgement.NOT_SATISFIED: 0.0,
    Judgement.UNCERTAIN: None,
}

ROLE_MAPPINGS = {"baseline": BASELINE_MAPPING, "bonus": BONUS_MAPPING}
POOLS = ("baseline", "bonus")

# 规则库必须声明的档位。基准项不设 satisfied（合格即满分，不判优秀）。
REQUIRED_KEYS_BY_ROLE = {
    "baseline": ("mostly", "partial", "weak", "not_satisfied", "uncertain"),
    "bonus": ("satisfied", "mostly", "partial", "weak", "not_satisfied", "uncertain"),
}

# 顶刊跃迁门槛：基准项不能有短板，且加分项要明显超出。
# 旧口径「每个维度达到维度权重 70%」是在五档等距标尺下定的，两池机制下语义已不适用。
#
# V16 重标定（2026-09-28）。判据与局限必须一起读：
# - 项目**没有真值**（evaluation/expert_benchmark_20.csv 尚未标注），所以阈值不是"拟合到正确答案"，
#   而是按**可复现性**选的：同一个规则库版本的多次重复跑，同一篇论文的判定不应翻转。
#   依据是 V16 的四批重复跑（runs/v23、v23b、v23c、v23d，三篇 AI-HRM 实证论文）。
# - 实测加分率区间：Choudhury 0.784–0.916、Blaurock 0.491–0.541、Cheng 0.427–0.455。
#   旧值 0.60 / 0.35 虽是 0 翻转，但离 Blaurock 上界只有 0.059（≈2.4 分），随时会被噪声推过线。
#   0.65 / 0.30 把最小裕度提到 0.109（≈4.4 分），是这批数据能达到的最宽裕度。
# - **局限**：0.109 的裕度仍小于单篇批间波动（Choudhury 加分率摆动 0.131），
#   所以单批次的「高/中潜力」判定本身不可靠。要下结论至少跑两批，报告应附批间范围。
BONUS_GATE_HIGH = 0.65
BONUS_GATE_MID = 0.30
# 基准项允许的短板比例：0 表示任何 weak/not_satisfied 都挡住「高」。
MAX_BASELINE_SHORTFALL_FOR_HIGH = 0.0
# 「中」档允许的基准短板。
MAX_BASELINE_SHORTFALL_FOR_MID = 0.05

# 完整度低于此比例时不下高顶刊潜力结论（分数口径宽松，但不给缺失证据背书）。
TOP_JOURNAL_MIN_COMPLETENESS = 0.8

# 标尺版本：改了上面的映射或两池口径就必须升这个号。
SCALE_VERSION = "MGMT-SCALE-V6"
SCALE_ANCHOR = "两池标尺：基准池 60 分（基准项合格即满分，只判缺口）+ 加分池 40 分（只有 satisfied 拿满加分）"

# 等级带：基准分 60 = 各方面都合格 = 达到发表水准；低于它说明基准项出现短板。
#
# V16 重标定。旧值 (93, 85, 75) 实测**不稳定**：Choudhury 四批总分 91.375 / 92.313 / 93.437 / 96.625
# 骑在 93 上，等级在「顶刊级 ↔ 强刊/顶刊边缘」之间翻转。
# 新值按"离所有论文所有批次的总分都尽量远"选（0 翻转的候选有 3454 个，必须挑平台中部而不是边缘）：
#   Choudhury [91.375, 96.625]｜Blaurock [79.625, 81.625]｜Cheng [77.078, 78.203]
#   → 88 落在 (81.625, 91.375) 这个空档中部；85 与它保持 3 分间距；74 低于 Cheng 下界 3.08 分。
#   三处最小裕度 3.0 分，是这批数据能达到的上限。
# - **局限（必须一起读）**：3.0 分的裕度**小于 Choudhury 自身的批间波动 5.25 分**，
#   所以只要论文落在某条线附近，单批次的等级就会翻。等级应视为"带不确定性的判定"：
#   要下结论至少跑两批；报告里应给出批间范围，而不是一个批次的绝对值。
GRADE_BANDS = (
    (88.0, "顶刊级"),
    (85.0, "强刊/顶刊边缘"),
    (74.0, "好论文"),
    (60.0, "达到发表水准"),
    (50.0, "偏弱但有一定价值"),
)


def rule_score_mapping(rule: Any) -> dict[str, float | None]:
    """档位 → 分值。单一真源是规则库的 score_mapping；缺档时才按角色退回代码默认标尺。

    基准项的默认标尺**不含** satisfied——合格即满分，基准项不判优秀。
    """
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
    """取某档位的分值。

    规则库没声明该档位时（例如基准项收到 satisfied），按角色防御性映射，
    而不是当成「未评到」——后者会把它挤出分母，反而放大其余权重的分量。
    """
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
                # strict_* 已废弃：标尺只有一套，全部由规则库 score_mapping 决定。
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
    """用**替代档位映射**重新算一遍分，供 Reliability Layer 的校准敏感性检测用。

    这是给 `reliability_diagnosis.calibration_uncertainty()` 的只读入口：
    它不改 `score_judgements`、不改规则库、不改任何已有调用方，只是把同一批 judgment
    换一套 mapping 再算一次分。Agent 不重跑——判断记录是现成的。

    `scheme` 形如 `{"baseline": {...}, "bonus": {...}}`，每项是档位 → 分值（None 表示不进分母）。
    未声明的档位按角色退回 `ROLE_MAPPINGS` 默认标尺，与 `lookup_score` 的防御口径一致。
    """
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
    """分档。分数口径宽松，但完整度不足时不给出高顶刊潜力判定，避免缺失证据被读成优秀。"""
    if score is None:
        return {
            "quality_grade": "无法判断",
            "top_journal_potential": "未知",
            "top_journal_gate_pass": False,
            "top_journal_gate_reasons": ["有效评分权重为 0，无法形成质量等级。"],
        }
    grade = "问题较多"
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
        potential = "高"
    elif complete_enough and shortfall <= MAX_BASELINE_SHORTFALL_FOR_MID and bonus_rate >= BONUS_GATE_MID:
        potential = "中"
    else:
        potential = "低"

    if shortfall > MAX_BASELINE_SHORTFALL_FOR_HIGH:
        reasons.append(
            f"基准项存在短板（基准达成率 {baseline_rate:.0%}：有规则被判未达发表水准或严重缺陷）"
        )
    if bonus_rate < BONUS_GATE_MID:
        reasons.append(f"加分项得分偏低（{bonus_rate:.0%}，低于 {BONUS_GATE_MID:.0%}）")
    elif bonus_rate < BONUS_GATE_HIGH:
        reasons.append(f"加分项未达到顶刊门槛（{bonus_rate:.0%}，需 {BONUS_GATE_HIGH:.0%}）")
    if not complete_enough:
        reasons = [
            f"评分完整度 {completeness:.0%} 低于 {TOP_JOURNAL_MIN_COMPLETENESS:.0%}，"
            "本次不足以判断顶刊潜力。",
            *reasons,
        ]
        # 分数口径保持宽松，但高分等级必须带上完整度提示，避免缺失证据被读成优秀。
        if score >= 86:
            grade = f"{grade}（完整度 {completeness:.0%}，结论待复核）"
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
    """返回 {dimension_id: [(construct_id, construct_weight, [rule_id...])]}。

    权重全部取自规则库声明；没有 constructs 层时退回“每条规则自成构念”，权重取规则权重。
    """
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
    """两池汇总：基准池 × 基准达成率 + 加分池 × 加成分。

    - 构念是权重的分配单位，按角色归池（`validate_rule_profile` 保证一个构念不跨角色）。
    - 构念内：按规则份额加权平均得到 0-1 比率；构念比率 × 构念权重 = 该构念贡献。
    - 口径宽松：只对**已评到**的构念权重归一，评不到的构念不进分母；完整度单独报告。
    - 某一池整体评不到时，用另一池把总分重归一到满分，并打 `pool_renormalized` 标记，
      避免因为一整池缺失把分数人为压低。
    - 同时给出严格下界（无法判断按 0 分占分母）与乐观上界（基准项按合格计、加分项按只沾边计）。
    - 适用范围外的构念既不进分子也不进分母。
    """
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
            # 维度满分按规则库全量权重摊分，与论文类型无关：否则同一维度在不同论文上会显示不同满分。
            for rule_id in member_rule_ids:
                member_rule = rules.get(rule_id)
                if member_rule is not None:
                    dimension_rule_weight[member_rule.role] += float(member_rule.weight)
            in_scope = [item for item in members if not item.out_of_scope]
            if not in_scope:
                out_of_scope_weight += construct_weight
                continue
            # 构念不跨角色（由校验强制），取首条成员的角色即可
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

        # 维度满分按两池权重比例摊分，保证 Σ 各维度满分 = 总分
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
        # 乐观上界：未评到的基准项按合格计，未评到的加分项按「只沾边」计
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
            f"两池机制：基准池 {profile.baseline_pool:.0f} 分（基准项合格即满分）"
            f" + 加分池 {profile.bonus_pool:.0f} 分（只有 satisfied 拿满加分）"
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
