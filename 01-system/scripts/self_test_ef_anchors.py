"""Offline checks for the E/F judgment anchors and rule-profile consistency.

The script checks rule structure, scoring roles, distinct judgment anchors, anchor length, and preservation of fields outside the intended scope. It makes no API calls.

Run with: python scripts/self_test_ef_anchors.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_eval.rules import (  # noqa: E402
    REQUIRED_JUDGEMENT_KEYS_BY_ROLE,
    load_rule_profile,
    validate_rule_profile,
)

RULES = Path(__file__).resolve().parents[1] / 'rules' / "rules_v1.yaml"
# Authoritative snapshot before changes: The rule base copied by the parent agent before the two agents took action.
SNAPSHOT = Path(__file__).resolve().parents[1] / 'rules' / "_rules_v1_V15_pre_EF_backup.yaml"

# 11 rules within the scope of this change (E six + F two extra points + two benchmark items + observation items).
# F-LIMITATIONS-V1 is not among them: its three articles have been judged as mostly / mostly / partial, and it is originally divided into categories.
# and it is the base item (partial = mostly = 1.0), the anchor point will not move if there is no problem.
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

# The original value of the mechanism field before changes (hard-coded rules). Source: rules_v1.yaml before changes, copied item by item.
EXPECTED_MECHANICS: dict[str, dict[str, object]] = {
    "E-RESULTS-COMPLETENESS-V1": {
        "role": "baseline",
        "weight": 1.8,
        "cluster": "E-COMPLETENESS",
        "applicable_type": 'Evidence',
        "dimension_id": "E",
        "score_mapping": {"mostly": 1.0, "partial": 1.0, "weak": 0.5, "not_satisfied": 0.0, "uncertain": None},
    },
    "E-RESULTS-ALIGNMENT-V1": {
        "role": "baseline",
        "weight": 3.6,
        "cluster": "E-ALIGNMENT",
        "applicable_type": 'General',
        "dimension_id": "E",
        "score_mapping": {"mostly": 1.0, "partial": 1.0, "weak": 0.5, "not_satisfied": 0.0, "uncertain": None},
    },
    "E-ROBUSTNESS-V1": {
        "role": "bonus",
        "weight": 1.5,
        "cluster": "E-ROBUSTNESS",
        "applicable_type": 'Evidence',
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
        "applicable_type": 'General',
        "dimension_id": "E",
        "score_mapping": {"mostly": 1.0, "partial": 1.0, "weak": 0.5, "not_satisfied": 0.0, "uncertain": None},
    },
    "E-CONCLUSION-VALIDITY-V1": {
        "role": "bonus",
        "weight": 1.5,
        "cluster": "E-CONCLUSION",
        "applicable_type": 'General',
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
        "applicable_type": 'General',
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
        "applicable_type": 'General',
        "dimension_id": "F",
        "score_mapping": {"mostly": 1.0, "partial": 1.0, "weak": 0.5, "not_satisfied": 0.0, "uncertain": None},
    },
    "F-DATA-CODE-TRANSPARENCY-V1": {
        "role": "baseline",
        "weight": 1.0,
        "cluster": "F-REPRO-TRANSPARENCY",
        "applicable_type": 'General',
        "dimension_id": "F",
        "score_mapping": {"mostly": 1.0, "partial": 1.0, "weak": 0.5, "not_satisfied": 0.0, "uncertain": None},
    },
    "F-EXTERNAL-VALIDITY-V1": {
        "role": "bonus",
        "weight": 1.0,
        "cluster": "F-EXTERNAL",
        "applicable_type": 'General',
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
        "applicable_type": 'General',
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
        "applicable_type": 'General',
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

# Protected anchor fields: These fields can only be moved by the 11 rules within the scope of this change.
PROTECTED_FIELDS = (
    "judgement_values",
    "retrieval_terms",
    "threshold",
    "mostly_threshold",
    "rationale",
    "mostly_note",
)

# Enumeration-style blacklist: These words turn the gears into a list of "only if everything is satisfied". If one item is missing, the gear will be dropped.
BANNED_ENUMERATION_WORDS = ('and', 'and', 'and', 'All', 'Complete', 'at the same time', 'and')

# satisfied and mostly The upper limit of coincidence of anchor text characters (Jaccard of character sets).
# is used to mechanically ensure that the highest gear is not a repeat of the second gear.
MAX_SATISFIED_MOSTLY_OVERLAP = 0.60

MIN_ANCHOR_CHARS = 25


def field_value(rule, field: str):
    'takes the field value and normalizes the empty string with None to prevent `key:` and omission from being written as "not written" in YAML.'
    return getattr(rule, field) or None


def jaccard(left: str, right: str) -> float:
    a, b = set(left), set(right)
    return len(a & b) / len(a | b)


def main() -> int:
    profile = load_rule_profile(RULES)
    rules = profile.rules_by_id

    # 1) The rule base itself must be able to be loaded and pass strict verification (weighted equalization, 6:4, two pools equalized)
    errors = validate_rule_profile(profile)
    assert errors == [], errors
    print(f"rule base verification: {len(profile.rules)} rules, validate_rule_profile is correct")

    # 2) All 11 rules within the scope of change must exist
    missing = [rule_id for rule_id in TARGET_RULE_IDS if rule_id not in rules]
    assert missing == [], f"Missing rule: {missing}"

    # 3) Not out of bounds: role/weight/cluster/score_mapping/applicable_type/dimension_id Not a word is touched
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
            f"The mechanism field of {rule_id} was changed: \n Now {actual}\n Before change {expected}"
        )
    print(f"does not cross the boundary: the role/weight/cluster/score_mapping/applicable_type/dimension_id of {len(TARGET_RULE_IDS)} E/F rules are the same as before the change")

    # 4) Gear shape: base item = fifth gear (excluding satisfied), bonus item = sixth gear (including satisfied)
    for rule_id in TARGET_RULE_IDS:
        rule = rules[rule_id]
        expected_keys = set(REQUIRED_JUDGEMENT_KEYS_BY_ROLE[rule.role])
        actual_keys = set(rule.judgement_values)
        assert actual_keys == expected_keys, (
            f"{rule_id} ({rule.role}) The gear key set is wrong: \n Now {sorted(actual_keys)}\n should be {sorted(expected_keys)}"
        )
        actual_mapping_keys = set(rule.score_mapping)
        assert actual_mapping_keys == expected_keys, (
            f"{rule_id} score_mapping key set is wrong: {sorted(actual_mapping_keys)}"
        )
        if rule.role == "baseline":
            assert "satisfied" not in rule.judgement_values, f"{rule_id} is the base item and should not have a satisfied file"
            assert "satisfied" not in rule.score_mapping, f"{rule_id} is the benchmark item and should not have a satisfied score"
    bonus_count = sum(1 for rule_id in TARGET_RULE_IDS if rules[rule_id].role == "bonus")
    print(f"gear shape: {len(TARGET_RULE_IDS) - bonus_count} has five basic items (excluding satisfied), {bonus_count} has six extra points")

    # 5) Disable enumeration connectives: judgment_values / threshold / mostly_threshold / rationale
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
                    f"{field_name} of {rule_id} hits the enumeration connective \"{word}\":"
                    f"…{text[max(0, match.start() - 8):match.start() + 10]}…"
                )
            scanned += 1
    print(f"Disable enumeration writing method: Scan {scanned} fields, {'/'.join(BANNED_ENUMERATION_WORDS)} does not appear")

    # 6) The anchor points of the six levels are different from each other, and each one has at least 25 words.
    for rule_id in TARGET_RULE_IDS:
        rule = rules[rule_id]
        values = rule.judgement_values
        assert len(set(values.values())) == len(values), f"{rule_id} Duplicate judgment anchors exist (copy and paste)"
        for key, text in values.items():
            assert len(text) >= MIN_ANCHOR_CHARS, (
                f"{rule_id} has only {len(text)} words, which is lower than {MIN_ANCHOR_CHARS} words."
            )
    print(f"Gear Anchor Point: The six gears of each rule are different from each other, and each anchor point ≥ {MIN_ANCHOR_CHARS} word")

    # 7) Bonus rule: satisfied is not a retelling of mostly
    overlaps = {}
    for rule_id in TARGET_RULE_IDS:
        rule = rules[rule_id]
        if rule.role != "bonus":
            continue
        score = jaccard(rule.judgement_values["satisfied"], rule.judgement_values["mostly"])
        overlaps[rule_id] = round(score, 3)
        assert score < MAX_SATISFIED_MOSTLY_OVERLAP, (
            f"{rule_id}'s satisfied and mostly anchor character coincidence degree {score:.3f}"
            f"is not lower than {MAX_SATISFIED_MOSTLY_OVERLAP}, the highest grade is like a repeat of the next highest grade"
        )
    print(f"satisfied ≠ mostly Restatement: upper limit of overlap {MAX_SATISFIED_MOSTLY_OVERLAP}, actual measurement {overlaps}")

    # 8) Other dimensions and F-LIMITATIONS remain unchanged: field by field comparison with the authoritative snapshot before the change
    assert SNAPSHOT.exists(), f"The authoritative snapshot before the change does not exist: {SNAPSHOT}"
    snapshot = load_rule_profile(SNAPSHOT)
    snapshot_rules = snapshot.rules_by_id
    protected: list[str] = []
    for rule in profile.rules:
        if rule.rule_id in TARGET_RULE_IDS:
            continue
        protected.append(rule.rule_id)
        assert rule.rule_id in snapshot_rules, f"{rule.rule_id} is not in the snapshot before the change"
        before = snapshot_rules[rule.rule_id]
        for field in PROTECTED_FIELDS:
            now_value = field_value(rule, field)
            before_value = field_value(before, field)
            assert now_value == before_value, (
                f"{field} of {rule.rule_id} is inconsistent with the snapshot before the change——"
                f"This change only allows the {len(TARGET_RULE_IDS)} rules of E/F to be moved \n"
                f"Now {now_value!r}\n Snapshot {before_value!r}"
            )
    print(
        f"Don’t touch others: {len(protected)} rules (A/B/C/D 22 + F-LIMITATIONS-V1)"
        f"{'/'.join(PROTECTED_FIELDS)} is consistent with snapshot {SNAPSHOT.name} field by field"
    )

    # 8b) Reverse check: This change indeed only falls on allowed fields (protected mechanism fields are not touched)
# The protected field of
    for rule_id in TARGET_RULE_IDS:
        before = snapshot_rules[rule_id]
        for field in ("role", "weight", "cluster", "score_mapping", "applicable_type", "dimension_id"):
            assert getattr(rules[rule_id], field) == getattr(before, field), (
                f"{rule_id} {field} is modified relative to the snapshot"
            )
    print(f"Reverse check: {len(TARGET_RULE_IDS)} mechanism fields of E/F rules remain unchanged compared to the snapshot")

    # 9) The structural side is not affected: dimension weights, two pools, construct layers, and evidence budgets are still the values before the change
    dimension_weights = {item.dimension_id: item.weight for item in profile.dimensions}
    assert dimension_weights == {"A": 15.0, "B": 20.0, "C": 20.0, "D": 25.0, "E": 15.0, "F": 5.0}, dimension_weights
    assert (profile.baseline_pool, profile.bonus_pool) == (60.0, 40.0), (profile.baseline_pool, profile.bonus_pool)
    assert profile.max_evidence_per_rule == 10, profile.max_evidence_per_rule
    e_bonus = sum(rule.weight for rule in profile.rules if rule.dimension_id == "E" and rule.role == "bonus")
    f_bonus = sum(rule.weight for rule in profile.rules if rule.dimension_id == "F" and rule.role == "bonus")
    assert (e_bonus, f_bonus) == (6.0, 2.0), (e_bonus, f_bonus)
    print(f"The structure remains unchanged: dimension weight {dimension_weights}; two pools 60/40; evidence budget 10; E bonus pool {e_bonus}, F bonus pool {f_bonus}")

    print('E/F gear anchor point self-test passed.')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
