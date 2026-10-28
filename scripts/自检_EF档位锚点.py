"""E/F 档位锚点自检：改动只在锚点层，机制字段与其它维度必须一字未动。

这份自检是这次改动的守则，不是功能测试。它钉死四件事：

1. **不越界**：E/F 11 条规则的 `role` / `weight` / `cluster` / `score_mapping` / `applicable_type` /
   `dimension_id` 全部硬编码了改动前的值，任何一条被改动都会让本自检失败。
2. **不改别人**：A/B/C/D 全部 22 条规则、以及本次不在改动范围内的 F-LIMITATIONS-V1，
   逐字段与**改动前的权威快照**比对（`评价规则库/_rules_v1_V15_pre_EF_backup.yaml`，父 agent 在两个
   agent 动手之前复制的那一份）。比的是解析后的对象本身，不比哈希——哈希会因键序/实现细节产生假阳性，
   而且硬编码的期望值读者看不出"要和哪份基线一致"。
3. **档位形状正确**：基准项不设 satisfied，加分项六档齐全（与 rules.REQUIRED_JUDGEMENT_KEYS_BY_ROLE 一致）。
4. **锚点可判别**：六个档位的锚点文本互不相同、每条 ≥ 25 字；不出现枚举式连接词；
   加分规则的 satisfied 与 mostly 锚点字符重合度（Jaccard）低于 0.60。

关于「放过的词」：本次只检查 `并`、`并且`、`且`、`全部`、`齐备`、`同时`、`以及`。
`或` 与 `、` 一律放过——`uncertain` 档的六条锚点必须写成「解析缺失、候选证据过窄，或研究范式不适用」
这类**互斥清单**，那是可选原因列表，不是「全部满足才算够」的枚举门槛。判定理由与报告里同样解释这一点。

运行：
    python ".\\scripts\\自检_EF档位锚点.py"

离线，零 API 调用。成功 exit 0，失败抛 AssertionError。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from literature_eval.rules import (  # noqa: E402
    REQUIRED_JUDGEMENT_KEYS_BY_ROLE,
    load_rule_profile,
    validate_rule_profile,
)

RULES = Path(__file__).resolve().parents[1] / "评价规则库" / "rules_v1.yaml"
# 改动前的权威快照：父 agent 在两个 agent 动手之前复制的规则库。
SNAPSHOT = Path(__file__).resolve().parents[1] / "评价规则库" / "_rules_v1_V15_pre_EF_backup.yaml"

# 本次改动范围内的 11 条规则（E 六条 + F 加分两条 + 两条基准项 + 观察项）。
# F-LIMITATIONS-V1 不在其中：它三篇已判 mostly / mostly / partial，本来就在分档，
# 且它是基准项（partial = mostly = 1.0），锚点没问题就不动。
TARGET_RULE_IDS = (
    "E-RESULTS-COMPLETENESS-V1",
    "E-RESULTS-ALIGNMENT-V1",
    "E-ROBUSTNESS-V1",
    "E-EVIDENCE-STRENGTH-V1",
    "E-CONCLUSION-VALIDITY-V1",
    "E-DISCUSSION-SYNTHESIS-V1",
    "F-REPRODUCIBILITY-V1",
    "F-DATA-CODE-TRANSPARENCY-V1",
    "F-EXTERNAL-VALIDITY-V1",
    "F-PRACTICAL-IMPLICATIONS-V1",
    "F-SOCIAL-LONGTERM-IMPACT-V1",
)

# 改动前的机制字段原值（硬编码守则）。来源：改动前的 rules_v1.yaml，逐条抄录。
EXPECTED_MECHANICS: dict[str, dict[str, object]] = {
    "E-RESULTS-COMPLETENESS-V1": {
        "role": "baseline",
        "weight": 1.8,
        "cluster": "E-COMPLETENESS",
        "applicable_type": "实证",
        "dimension_id": "E",
        "score_mapping": {"mostly": 1.0, "partial": 1.0, "weak": 0.5, "not_satisfied": 0.0, "uncertain": None},
    },
    "E-RESULTS-ALIGNMENT-V1": {
        "role": "baseline",
        "weight": 3.6,
        "cluster": "E-ALIGNMENT",
        "applicable_type": "通用",
        "dimension_id": "E",
        "score_mapping": {"mostly": 1.0, "partial": 1.0, "weak": 0.5, "not_satisfied": 0.0, "uncertain": None},
    },
    "E-ROBUSTNESS-V1": {
        "role": "bonus",
        "weight": 1.5,
        "cluster": "E-ROBUSTNESS",
        "applicable_type": "实证",
        "dimension_id": "E",
        "score_mapping": {
            "satisfied": 1.0,
            "mostly": 0.5,
            "partial": 0.25,
            "weak": 0.0,
            "not_satisfied": 0.0,
            "uncertain": None,
        },
    },
    "E-EVIDENCE-STRENGTH-V1": {
        "role": "baseline",
        "weight": 3.6,
        "cluster": "E-STRENGTH",
        "applicable_type": "通用",
        "dimension_id": "E",
        "score_mapping": {"mostly": 1.0, "partial": 1.0, "weak": 0.5, "not_satisfied": 0.0, "uncertain": None},
    },
    "E-CONCLUSION-VALIDITY-V1": {
        "role": "bonus",
        "weight": 1.5,
        "cluster": "E-CONCLUSION",
        "applicable_type": "通用",
        "dimension_id": "E",
        "score_mapping": {
            "satisfied": 1.0,
            "mostly": 0.5,
            "partial": 0.25,
            "weak": 0.0,
            "not_satisfied": 0.0,
            "uncertain": None,
        },
    },
    "E-DISCUSSION-SYNTHESIS-V1": {
        "role": "bonus",
        "weight": 3.0,
        "cluster": "E-DISCUSSION",
        "applicable_type": "通用",
        "dimension_id": "E",
        "score_mapping": {
            "satisfied": 1.0,
            "mostly": 0.5,
            "partial": 0.25,
            "weak": 0.0,
            "not_satisfied": 0.0,
            "uncertain": None,
        },
    },
    "F-REPRODUCIBILITY-V1": {
        "role": "baseline",
        "weight": 1.0,
        "cluster": "F-REPRO-TRANSPARENCY",
        "applicable_type": "通用",
        "dimension_id": "F",
        "score_mapping": {"mostly": 1.0, "partial": 1.0, "weak": 0.5, "not_satisfied": 0.0, "uncertain": None},
    },
    "F-DATA-CODE-TRANSPARENCY-V1": {
        "role": "baseline",
        "weight": 1.0,
        "cluster": "F-REPRO-TRANSPARENCY",
        "applicable_type": "通用",
        "dimension_id": "F",
        "score_mapping": {"mostly": 1.0, "partial": 1.0, "weak": 0.5, "not_satisfied": 0.0, "uncertain": None},
    },
    "F-EXTERNAL-VALIDITY-V1": {
        "role": "bonus",
        "weight": 1.0,
        "cluster": "F-EXTERNAL",
        "applicable_type": "通用",
        "dimension_id": "F",
        "score_mapping": {
            "satisfied": 1.0,
            "mostly": 0.5,
            "partial": 0.25,
            "weak": 0.0,
            "not_satisfied": 0.0,
            "uncertain": None,
        },
    },
    "F-PRACTICAL-IMPLICATIONS-V1": {
        "role": "bonus",
        "weight": 1.0,
        "cluster": "F-PRACTICAL",
        "applicable_type": "通用",
        "dimension_id": "F",
        "score_mapping": {
            "satisfied": 1.0,
            "mostly": 0.5,
            "partial": 0.25,
            "weak": 0.0,
            "not_satisfied": 0.0,
            "uncertain": None,
        },
    },
    "F-SOCIAL-LONGTERM-IMPACT-V1": {
        "role": "bonus",
        "weight": 0.0,
        "cluster": "F-SOCIAL",
        "applicable_type": "通用",
        "dimension_id": "F",
        "score_mapping": {
            "satisfied": 1.0,
            "mostly": 0.5,
            "partial": 0.25,
            "weak": 0.0,
            "not_satisfied": 0.0,
            "uncertain": None,
        },
    },
}

# 受保护的锚点字段：这些字段只有本次改动范围内的 11 条规则可以动。
PROTECTED_FIELDS = (
    "judgement_values",
    "retrieval_terms",
    "threshold",
    "mostly_threshold",
    "rationale",
    "mostly_note",
)

# 枚举式写法黑名单：这些词把档位变成「全部满足才算够」的清单，缺一项就掉档。
BANNED_ENUMERATION_WORDS = ("并", "并且", "且", "全部", "齐备", "同时", "以及")

# satisfied 与 mostly 的锚点文本字符重合度上限（Jaccard of character sets）。
# 用来机械地保证最高档不是次高档的复述。
MAX_SATISFIED_MOSTLY_OVERLAP = 0.60

MIN_ANCHOR_CHARS = 25


def field_value(rule, field: str):
    """取字段值，把空字符串与 None 归一，避免 YAML 里 `key:` 与省略写成两种"没写"。"""
    return getattr(rule, field) or None


def jaccard(left: str, right: str) -> float:
    a, b = set(left), set(right)
    return len(a & b) / len(a | b)


def main() -> int:
    profile = load_rule_profile(RULES)
    rules = profile.rules_by_id

    # 1) 规则库本身必须能加载并通过严格校验（权重加平、6:4、两池加平）
    errors = validate_rule_profile(profile)
    assert errors == [], errors
    print(f"规则库校验：{len(profile.rules)} 条规则，validate_rule_profile 无错")

    # 2) 改动范围内的 11 条规则必须全部存在
    missing = [rule_id for rule_id in TARGET_RULE_IDS if rule_id not in rules]
    assert missing == [], f"缺少规则：{missing}"

    # 3) 不越界：role / weight / cluster / score_mapping / applicable_type / dimension_id 一字未动
    for rule_id in TARGET_RULE_IDS:
        rule = rules[rule_id]
        expected = EXPECTED_MECHANICS[rule_id]
        actual = {
            "role": rule.role,
            "weight": rule.weight,
            "cluster": rule.cluster,
            "applicable_type": rule.applicable_type,
            "dimension_id": rule.dimension_id,
            "score_mapping": dict(rule.score_mapping),
        }
        assert actual == expected, (
            f"{rule_id} 的机制字段被改动了：\n  现在 {actual}\n  改动前 {expected}"
        )
    print(f"不越界：{len(TARGET_RULE_IDS)} 条 E/F 规则的 role/weight/cluster/score_mapping/applicable_type/dimension_id 与改动前一致")

    # 4) 档位形状：基准项 = 五档（不含 satisfied），加分项 = 六档（含 satisfied）
    for rule_id in TARGET_RULE_IDS:
        rule = rules[rule_id]
        expected_keys = set(REQUIRED_JUDGEMENT_KEYS_BY_ROLE[rule.role])
        actual_keys = set(rule.judgement_values)
        assert actual_keys == expected_keys, (
            f"{rule_id}（{rule.role}）档位键集合不对：\n  现在 {sorted(actual_keys)}\n  应为 {sorted(expected_keys)}"
        )
        actual_mapping_keys = set(rule.score_mapping)
        assert actual_mapping_keys == expected_keys, (
            f"{rule_id} score_mapping 键集合不对：{sorted(actual_mapping_keys)}"
        )
        if rule.role == "baseline":
            assert "satisfied" not in rule.judgement_values, f"{rule_id} 是基准项，不应有 satisfied 档"
            assert "satisfied" not in rule.score_mapping, f"{rule_id} 是基准项，不应有 satisfied 分值"
    bonus_count = sum(1 for rule_id in TARGET_RULE_IDS if rules[rule_id].role == "bonus")
    print(f"档位形状：{len(TARGET_RULE_IDS) - bonus_count} 条基准项五档（不含 satisfied）、{bonus_count} 条加分项六档")

    # 5) 禁用枚举式连接词：judgement_values / threshold / mostly_threshold / rationale
    scanned = 0
    for rule_id in TARGET_RULE_IDS:
        rule = rules[rule_id]
        fields = {
            "judgement_values.satisfied": rule.judgement_values.get("satisfied", ""),
            "judgement_values.mostly": rule.judgement_values["mostly"],
            "judgement_values.partial": rule.judgement_values["partial"],
            "judgement_values.weak": rule.judgement_values["weak"],
            "judgement_values.not_satisfied": rule.judgement_values["not_satisfied"],
            "judgement_values.uncertain": rule.judgement_values["uncertain"],
            "threshold": rule.threshold or "",
            "mostly_threshold": rule.mostly_threshold or "",
            "rationale": rule.rationale or "",
        }
        for field_name, text in fields.items():
            for word in BANNED_ENUMERATION_WORDS:
                match = re.search(re.escape(word), text)
                assert match is None, (
                    f"{rule_id} 的 {field_name} 命中枚举式连接词「{word}」："
                    f"…{text[max(0, match.start() - 8):match.start() + 10]}…"
                )
            scanned += 1
    print(f"禁用枚举式写法：扫过 {scanned} 个字段，未出现 {'/'.join(BANNED_ENUMERATION_WORDS)}")

    # 6) 六个档位的锚点互不相同，且每条至少 25 字
    for rule_id in TARGET_RULE_IDS:
        rule = rules[rule_id]
        values = rule.judgement_values
        assert len(set(values.values())) == len(values), f"{rule_id} 存在重复档位锚点（复制粘贴）"
        for key, text in values.items():
            assert len(text) >= MIN_ANCHOR_CHARS, (
                f"{rule_id} 的 {key} 档锚点只有 {len(text)} 字，低于 {MIN_ANCHOR_CHARS} 字"
            )
    print(f"档位锚点：每条规则六档互不相同，且每条锚点 ≥ {MIN_ANCHOR_CHARS} 字")

    # 7) 加分规则：satisfied 不是 mostly 的复述
    overlaps = {}
    for rule_id in TARGET_RULE_IDS:
        rule = rules[rule_id]
        if rule.role != "bonus":
            continue
        score = jaccard(rule.judgement_values["satisfied"], rule.judgement_values["mostly"])
        overlaps[rule_id] = round(score, 3)
        assert score < MAX_SATISFIED_MOSTLY_OVERLAP, (
            f"{rule_id} 的 satisfied 与 mostly 锚点字符重合度 {score:.3f} "
            f"不低于 {MAX_SATISFIED_MOSTLY_OVERLAP}，最高档像次高档的复述"
        )
    print(f"satisfied ≠ mostly 复述：重合度上限 {MAX_SATISFIED_MOSTLY_OVERLAP}，实测 {overlaps}")

    # 8) 其它维度与 F-LIMITATIONS 一字未动：逐字段与改动前的权威快照比对
    assert SNAPSHOT.exists(), f"改动前的权威快照不存在：{SNAPSHOT}"
    snapshot = load_rule_profile(SNAPSHOT)
    snapshot_rules = snapshot.rules_by_id
    protected: list[str] = []
    for rule in profile.rules:
        if rule.rule_id in TARGET_RULE_IDS:
            continue
        protected.append(rule.rule_id)
        assert rule.rule_id in snapshot_rules, f"{rule.rule_id} 不在改动前的快照里"
        before = snapshot_rules[rule.rule_id]
        for field in PROTECTED_FIELDS:
            now_value = field_value(rule, field)
            before_value = field_value(before, field)
            assert now_value == before_value, (
                f"{rule.rule_id} 的 {field} 与改动前快照不一致——"
                f"本次改动只允许动 E/F 的 {len(TARGET_RULE_IDS)} 条规则\n"
                f"  现在 {now_value!r}\n  快照 {before_value!r}"
            )
    print(
        f"不碰别人：{len(protected)} 条规则（A/B/C/D 22 条 + F-LIMITATIONS-V1）的 "
        f"{'/'.join(PROTECTED_FIELDS)} 与快照 {SNAPSHOT.name} 逐字段一致"
    )

    # 8b) 反向检查：本次改动确实只落在允许的字段上（未触碰受保护的机制字段）
    for rule_id in TARGET_RULE_IDS:
        before = snapshot_rules[rule_id]
        for field in ("role", "weight", "cluster", "score_mapping", "applicable_type", "dimension_id"):
            assert getattr(rules[rule_id], field) == getattr(before, field), (
                f"{rule_id} 的受保护字段 {field} 相对快照被改动"
            )
    print(f"反向检查：{len(TARGET_RULE_IDS)} 条 E/F 规则的机制字段相对快照未变")

    # 9) 结构侧未被牵连：维度权重、两池、构念层、证据预算仍是改动前的值
    dimension_weights = {item.dimension_id: item.weight for item in profile.dimensions}
    assert dimension_weights == {"A": 15.0, "B": 20.0, "C": 20.0, "D": 25.0, "E": 15.0, "F": 5.0}, dimension_weights
    assert (profile.baseline_pool, profile.bonus_pool) == (60.0, 40.0), (profile.baseline_pool, profile.bonus_pool)
    assert profile.max_evidence_per_rule == 10, profile.max_evidence_per_rule
    e_bonus = sum(rule.weight for rule in profile.rules if rule.dimension_id == "E" and rule.role == "bonus")
    f_bonus = sum(rule.weight for rule in profile.rules if rule.dimension_id == "F" and rule.role == "bonus")
    assert (e_bonus, f_bonus) == (6.0, 2.0), (e_bonus, f_bonus)
    print(f"结构未动：维度权重 {dimension_weights}；两池 60/40；证据预算 10；E 加分池 {e_bonus}、F 加分池 {f_bonus}")

    print("E/F 档位锚点自检通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
