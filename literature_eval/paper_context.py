from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from .llm import AgentRuntimeConfig, call_plain_json
from .models import Evidence, Paper
from .rules import PAPER_TYPE_CLASSES, RuleProfile, normalize_paper_type


SECTION_ALIASES = {
    "abstract": ["abstract", "摘要"],
    "introduction": ["introduction", "引言", "intro"],
    "theory": ["literature", "theory", "hypothesis", "background", "related work", "理论", "文献", "假设"],
    "method": [
        "method",
        "methods",
        "research design",
        "data",
        "sample",
        "measurement",
        "model",
        "architecture",
        "training",
        "方法",
        "设计",
        "数据",
        "样本",
        "测量",
    ],
    "results": ["result", "finding", "experiment", "evaluation", "analysis", "robust", "结果", "发现", "实验", "检验"],
    "discussion": ["discussion", "conclusion", "limitation", "future", "implication", "讨论", "结论", "局限", "启示"],
    "tables_figures": ["tables and figures", "table", "figure", "fig.", "表", "图"],
    "appendix": ["appendix", "appendices", "supplementary", "附录", "补充材料"],
}

DIMENSION_FACETS = {
    "A": ["abstract", "introduction"],
    "B": ["introduction", "theory", "discussion"],
    "C": ["theory", "introduction"],
    "D": ["method", "appendix"],
    "E": ["results", "tables_figures", "method"],
    "F": ["discussion", "appendix", "results", "method"],
}

DIMENSION_ROUTING_GUIDE = {
    "A": "选题价值、研究问题、研究范围、问题边界、可操作性。",
    "B": "理论贡献、挑战既有假设、文献分歧整合、跨理论整合、新概念或新机制。",
    "C": "概念定义、理论前提、推理链条、假设或命题、边界条件、替代解释。",
    "D": "研究设计、数据来源、样本、测量、模型设定、识别策略、方法执行。",
    "E": "结果呈现、结果与问题对应、稳健性、证据强度、结论有效性、讨论综合。",
    "F": "可复制性、数据代码透明、局限、外部效度、实践启示、长期影响。",
}

THEORETICAL_MOVE_TYPES = [
    "new_construct",
    "new_mechanism",
    "boundary_extension",
    "theory_integration",
    "challenge_assumption",
    "measurement_or_method_contribution",
    "application_only",
    "unclear",
]
THEORETICAL_MOVE_STRENGTHS = {"weak", "moderate", "strong", "breakthrough", "unclear"}

PAPER_TYPE_RULE_FOCUS = {
    "理论或概念论文": {
        "primary_dimensions": ["A", "B", "C", "F"],
        "notes": "重点看问题定位、理论对话、概念机制、命题逻辑和边界条件；不能强制要求样本、变量或回归模型。",
    },
    "管理学实证研究": {
        "primary_dimensions": ["A", "C", "D", "E", "F"],
        "notes": "重点看研究问题、设计匹配、变量测量、样本数据、识别策略、结果稳健性和结论边界。",
    },
    "质性或案例研究": {
        "primary_dimensions": ["A", "B", "C", "D", "E", "F"],
        "notes": "重点看研究范式匹配、案例选择、资料可信度、编码或解释透明度、理论生成和反身性。",
    },
    "技术或机器学习论文": {
        "primary_dimensions": ["A", "D", "E", "F"],
        "notes": "重点看任务价值、方法新意、实验设计、基准比较、可复现性和应用边界；理论贡献维度需谨慎解释。",
    },
    "综述论文": {
        "primary_dimensions": ["A", "B", "C", "F"],
        "notes": "重点看文献覆盖、分类框架、共识整合、研究议程和边界说明；不强制要求实证识别策略。",
    },
    "评论或观点论文": {
        "primary_dimensions": ["A", "B", "C", "F"],
        "notes": "重点看问题洞察、论证质量、对既有讨论的推进和实践/治理启示；方法维度通常需要人工复核。",
    },
}


def _is_reference(item: Evidence) -> bool:
    text = item.text.lower()
    return item.section.lower() == "references" or bool(re.match(r"^\s*(references|bibliography)\b", text))


def _matches(text: str, hints: list[str]) -> bool:
    lowered = text.lower()
    return any(hint.lower() in lowered for hint in hints)


def _trim(text: str, limit: int = 360) -> str:
    return re.sub(r"\s+", " ", text).strip()[:limit]


def _trim_mapping(value: Any, limits: dict[str, int]) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    trimmed: dict[str, Any] = {}
    for key, limit in limits.items():
        item = value.get(key)
        if isinstance(item, str):
            trimmed[key] = _trim(item, limit)
        elif isinstance(item, list):
            trimmed[key] = [_trim(str(entry), limit) for entry in item[:6]]
        elif isinstance(item, dict):
            trimmed[key] = item
        elif item is not None:
            trimmed[key] = item
    return trimmed


def _normalize_paper_type(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    lowered = text.lower()
    if any(marker in lowered for marker in ["empirical", "experiment", "survey", "regression", "panel"]):
        return "管理学实证研究"
    if any(marker in lowered for marker in ["conceptual", "theory", "theoretical"]):
        return "理论或概念论文"
    if any(marker in lowered for marker in ["review", "meta-analysis"]):
        return "综述论文"
    if any(marker in lowered for marker in ["qualitative", "case study", "interview"]):
        return "质性或案例研究"
    if any(marker in lowered for marker in ["commentary", "viewpoint", "perspective"]):
        return "评论或观点论文"
    if any(marker in lowered for marker in ["machine learning", "algorithm", "technical"]):
        return "技术或机器学习论文"
    return text or fallback


def _normalize_theoretical_move(value: Any) -> dict[str, str]:
    raw = value if isinstance(value, dict) else {}
    move_type = str(raw.get("type") or "unclear").strip()
    if move_type not in THEORETICAL_MOVE_TYPES:
        move_type = "unclear"
    description = _trim(str(raw.get("description") or ""), 420)
    strength = str(raw.get("strength") or "unclear").strip().lower()
    if strength not in THEORETICAL_MOVE_STRENGTHS:
        extra = _trim(str(raw.get("strength") or ""), 220)
        if extra and extra not in description:
            description = _trim(f"{description} {extra}", 420)
        extra_lower = extra.lower()
        if any(marker in extra_lower for marker in ["breakthrough", "突破", "field-changing"]):
            strength = "breakthrough"
        elif any(marker in extra_lower for marker in ["strong", "robust", "clear", "明确", "较强"]):
            strength = "strong"
        elif any(marker in extra_lower for marker in ["weak", "limited", "不足", "较弱"]):
            strength = "weak"
        elif any(marker in extra_lower for marker in ["moderate", "partial", "部分", "中等"]):
            strength = "moderate"
        else:
            strength = "unclear"
    return {"type": move_type, "description": description, "strength": strength}


def _paper_type(title: str, evidences: list[Evidence]) -> str:
    sample = " ".join(item.text.lower() for item in evidences[:20])
    text = f"{title.lower()} {sample}"
    if _matches(text, ["review", "systematic literature", "meta-analysis", "综述", "元分析"]):
        return "综述论文"
    if _matches(text, ["editorial", "commentary", "perspective", "call for action", "viewpoint", "评论", "观点"]):
        return "评论或观点论文"
    if _matches(text, ["interview", "case study", "ethnograph", "qualitative", "访谈", "案例", "质性"]):
        return "质性或案例研究"
    if _matches(text, ["regression", "survey", "panel data", "sample", "hypotheses", "empirical", "实证", "回归", "问卷"]):
        return "管理学实证研究"
    if _matches(text, ["transformer", "neural", "architecture", "algorithm", "benchmark", "dataset", "machine translation"]):
        return "技术或机器学习论文"
    if _matches(text, ["theory", "conceptual", "proposition", "framework", "理论", "概念", "命题"]):
        return "理论或概念论文"
    return "未识别"


def build_review_profile(paper_type: str) -> dict[str, Any]:
    focus = PAPER_TYPE_RULE_FOCUS.get(paper_type, {})
    nonstandard_types = {"理论或概念论文", "质性或案例研究", "综述论文", "评论或观点论文"}
    return {
        "paper_type": paper_type,
        "nonstandard_paradigm": paper_type in nonstandard_types,
        "primary_dimensions": focus.get("primary_dimensions", ["A", "B", "C", "D", "E", "F"]),
        "rule_focus_notes": focus.get("notes", "先按通用管理学论文评价，无法适配的维度需进入人工复核。"),
        "anti_homogenization_policy": (
            "识别为非常规研究范式时，不强制套用标准实证模板；但仍要求问题清楚、论证严谨、证据充分。"
        ),
        "high_score_evidence_policy": (
            "高分必须依赖多处实质证据，不能只依赖摘要、引言中的自我声明或格式化表述。"
        ),
    }


def build_section_map(evidences: list[Evidence]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Evidence]] = defaultdict(list)
    for item in evidences:
        if not _is_reference(item):
            grouped[item.section].append(item)
    rows = []
    for section, items in grouped.items():
        items.sort(key=lambda item: (item.page, item.span[0]))
        rows.append(
            {
                "section": section,
                "pages": sorted({item.page for item in items}),
                "summary": _trim(" ".join(item.text for item in items[:3]), 500),
                "key_evidence_ids": [item.evidence_id for item in items[:5]],
            }
        )
    rows.sort(key=lambda row: row["pages"][0] if row["pages"] else 9999)
    return rows


def build_evidence_index(evidences: list[Evidence]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {facet: [] for facet in SECTION_ALIASES}
    usable = [item for item in evidences if not _is_reference(item)]
    total_pages = max((item.page for item in usable), default=1)
    for item in usable:
        haystack = f"{item.section}\n{item.text}"
        for facet, hints in SECTION_ALIASES.items():
            if _matches(haystack, hints):
                index[facet].append(item.evidence_id)
    middle = [item.evidence_id for item in usable if total_pages * 0.25 <= item.page <= total_pages * 0.8]
    index["middle_body"] = middle[:30]
    return {key: value[:30] for key, value in index.items()}


def build_context_evidence_pack(evidences: list[Evidence], limit: int = 30) -> list[dict[str, Any]]:
    usable = [item for item in evidences if not _is_reference(item)]
    if not usable:
        return []
    total_pages = max(item.page for item in usable)
    early = [item for item in usable if item.page <= max(2, total_pages * 0.25)]
    middle = [item for item in usable if total_pages * 0.25 < item.page <= total_pages * 0.75]
    late = [item for item in usable if item.page > total_pages * 0.75]
    selected: list[Evidence] = []
    early_quota = max(6, limit // 4)
    late_quota = max(6, limit // 4)
    middle_quota = max(8, limit - early_quota - late_quota)
    for bucket, quota in [(early, early_quota), (middle, middle_quota), (late, late_quota)]:
        bucket.sort(key=lambda item: (item.page, item.span[0]))
        step = max(1, len(bucket) // max(1, quota))
        selected.extend(bucket[::step][:quota])
    seen: set[str] = set()
    unique = []
    for item in selected:
        if item.evidence_id in seen:
            continue
        seen.add(item.evidence_id)
        unique.append({"evidence_id": item.evidence_id, "section": item.section, "page": item.page, "text": _trim(item.text, 260)})
    return unique[:limit]


def fallback_dimension_evidence_routes(paper_context: dict[str, Any], context_pack: list[dict[str, Any]], profile: RuleProfile) -> dict[str, Any]:
    valid_by_id = {str(item.get("evidence_id")): item for item in context_pack if item.get("evidence_id")}
    evidence_index = paper_context.get("evidence_index", {}) if isinstance(paper_context, dict) else {}
    routes: dict[str, list[str]] = {}
    notes: dict[str, str] = {}
    ordered_ids = [str(item.get("evidence_id")) for item in context_pack if item.get("evidence_id")]
    for dimension in profile.dimensions:
        selected: list[str] = []
        seen: set[str] = set()
        for facet in DIMENSION_FACETS.get(dimension.dimension_id, []):
            values = evidence_index.get(facet, []) if isinstance(evidence_index, dict) else []
            if not isinstance(values, list):
                continue
            for value in values:
                evidence_id = str(value)
                if evidence_id in valid_by_id and evidence_id not in seen:
                    selected.append(evidence_id)
                    seen.add(evidence_id)
                if len(selected) >= 8:
                    break
            if len(selected) >= 8:
                break
        for evidence_id in ordered_ids:
            if len(selected) >= 8:
                break
            if evidence_id not in seen:
                selected.append(evidence_id)
                seen.add(evidence_id)
        routes[dimension.dimension_id] = selected[:8]
        notes[dimension.dimension_id] = "证据路由 Agent 不可用时，由本地章节索引生成候选证据；最终评分仍由专家 Agent 判断。"
    return {"dimension_evidence_routes": routes, "routing_notes": notes, "routing_mode": "local_index_fallback"}


def build_dimension_evidence_routes(
    run_id: str,
    paper_context: dict[str, Any],
    context_pack: list[dict[str, Any]],
    profile: RuleProfile,
    config: AgentRuntimeConfig,
) -> dict[str, Any]:
    dimensions = [
        {
            "dimension_id": item.dimension_id,
            "dimension_name": item.dimension_name,
            "routing_goal": DIMENSION_ROUTING_GUIDE.get(item.dimension_id, item.description),
        }
        for item in profile.dimensions
    ]
    prompt = (
        "你是全文证据路由 Agent。你已经看到论文全文的跨页证据包和论文画像。"
        "请不要按 PDF 页眉或章节标签机械分配，而要根据内容语义判断哪些证据应交给哪个专家。"
        "快速完成分配。每个维度至少选择 4 条、最多 8 条 evidence_id；应覆盖前部、中部、后部的相关内容。"
        "A/B/C 不得只选摘要；必须尽量包含概念定义、文献对话、理论展开或讨论段。"
        "D/E/F 必须尽量包含方法、数据、结果、讨论、局限或附录相关段。"
        "只允许引用输入 context_evidences 中存在的 evidence_id。"
        "输出 JSON："
        '{"dimension_evidence_routes":{"A":["EV-..."],"B":[],"C":[],"D":[],"E":[],"F":[]},'
        '"routing_notes":{"A":"中文说明","B":"中文说明","C":"中文说明","D":"中文说明","E":"中文说明","F":"中文说明"}}'
    )
    payload = {
        "run_id": run_id,
        "paper_profile": paper_context.get("paper_profile", {}),
        "dimensions": dimensions,
        "context_evidences": context_pack,
    }
    try:
        routed = call_plain_json(prompt, payload, config)
    except Exception as exc:
        fallback = fallback_dimension_evidence_routes(paper_context, context_pack, profile)
        fallback["routing_error"] = f"Evidence routing Agent failed: {exc}"
        return fallback

    valid_ids = {str(item["evidence_id"]) for item in context_pack if "evidence_id" in item}
    raw_routes = routed.get("dimension_evidence_routes", {}) if isinstance(routed, dict) else {}
    routes: dict[str, list[str]] = {}
    for dimension in profile.dimensions:
        values = raw_routes.get(dimension.dimension_id, []) if isinstance(raw_routes, dict) else []
        if isinstance(values, str):
            values = re.split(r"[,;，；]\s*", values)
        if not isinstance(values, list):
            values = []
        seen: set[str] = set()
        routes[dimension.dimension_id] = [
            evidence_id
            for evidence_id in (str(value) for value in values)
            if evidence_id in valid_ids and not (evidence_id in seen or seen.add(evidence_id))
        ][:8]

    if not any(routes.values()):
        fallback = fallback_dimension_evidence_routes(paper_context, context_pack, profile)
        fallback["routing_error"] = "Evidence routing Agent returned no usable evidence routes."
        return fallback

    return {
        "dimension_evidence_routes": routes,
        "routing_notes": routed.get("routing_notes", {}) if isinstance(routed, dict) else {},
        "routing_mode": "agent",
    }


PROFILE_PLACEHOLDER = "待 Agent 结合全文判断"

# 全文画像里必须由 Agent 真正填写的字段；仍是占位符就说明画像没生成成功。
REQUIRED_PROFILE_FIELDS = [
    "central_problem",
    "research_gap",
    "research_question",
    "core_claim",
    "main_contribution",
]


def incomplete_profile_fields(profile: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for field_name in REQUIRED_PROFILE_FIELDS:
        value = str(profile.get(field_name) or "").strip()
        if not value or value == PROFILE_PLACEHOLDER:
            missing.append(field_name)
    return missing


def build_global_context(
    run_id: str,
    paper: Paper,
    evidences: list[Evidence],
    profile: RuleProfile,
    config: AgentRuntimeConfig,
) -> dict[str, Any]:
    title = str(paper.metadata.get("title") or "")
    heuristic_type = _paper_type(title, evidences)
    base_context: dict[str, Any] = {
        "run_id": run_id,
        "paper_profile": {
            "title": title,
            # paper_type 是受控枚举（实证/质性/理论/综述/混合），只用于规则适用范围；
            # 描述性标签保留在 paper_type_label，供提示词与报告阅读。
            "paper_type": normalize_paper_type(heuristic_type) or "混合",
            "paper_type_label": heuristic_type,
            "central_problem": "待 Agent 结合全文判断",
            "research_gap": "待 Agent 结合全文判断",
            "research_question": "待 Agent 结合全文判断",
            "core_claim": "待 Agent 结合全文判断",
            "main_contribution": "待 Agent 结合全文判断",
            "theoretical_move": {
                "type": "unclear",
                "description": "待 Agent 结合全文判断",
                "strength": "unclear",
            },
            "method_design": {
                "design_type": "待 Agent 结合全文判断",
                "sample_data": "待 Agent 结合全文判断",
                "identification_or_validity_strategy": "待 Agent 结合全文判断",
                "limitations": "待 Agent 结合全文判断",
            },
            "method_type": "待 Agent 结合全文判断",
            "data_or_material": "待 Agent 结合全文判断",
            "evidence_structure": {
                "main_evidence": [],
                "robustness_or_validation": [],
                "cross_section_support": [],
            },
            "main_findings": "待 Agent 结合全文判断",
            "boundary_conditions": "待 Agent 结合全文判断",
            "limitations": "待 Agent 结合全文判断",
            "most_important_pages": [],
            "possible_top_journal_signals": [],
            "blocking_weaknesses": [],
        },
        "review_profile": build_review_profile(heuristic_type),
        "section_map": build_section_map(evidences),
        "evidence_index": build_evidence_index(evidences),
        "rule_profile": {
            "name": profile.rule_profile_name,
            "dimensions": [
                {"dimension_id": item.dimension_id, "dimension_name": item.dimension_name, "weight": item.weight}
                for item in profile.dimensions
            ],
        },
    }
    context_pack = build_context_evidence_pack(evidences)
    prompt = (
        "你是全文理解 Agent。请先通读给定的跨页证据包，形成论文整体画像。"
        "必须只基于证据，不要编造；不要根据期刊、作者或影响因子判断质量。"
        "你的输出会作为后续 A-F 专家 Agent 的共享上下文，因此要帮助它们理解全文，而不是只摘要前两页。"
        "输出 JSON，键包括 paper_profile 和 reading_notes。"
        "paper_profile 必须包含 title, paper_type, paper_type_detail, central_problem, research_gap, research_question, core_claim, main_contribution, "
        "theoretical_move, method_design, method_type, data_or_material, evidence_structure, main_findings, boundary_conditions, "
        "limitations, most_important_pages, possible_top_journal_signals, blocking_weaknesses。"
        "theoretical_move 必须包含 type, description, strength；type 从 "
        f"{', '.join(THEORETICAL_MOVE_TYPES)} 中选最接近的一项。"
        "method_design 必须包含 design_type, sample_data, identification_or_validity_strategy, limitations。"
        "evidence_structure 必须包含 main_evidence, robustness_or_validation, cross_section_support。"
        "possible_top_journal_signals 只记录正文内部信号，例如重要问题、新机制、新构念、强识别、证据闭环、边界清楚、可迁移框架；"
        "blocking_weaknesses 记录阻止进入 85+ 的内部短板。"
        "most_important_pages 填最能支撑评分的页码，最多 8 个。"
        f"paper_type 必须从这 5 个受控取值里选一个，只能填词本身：{'、'.join(PAPER_TYPE_CLASSES)}；"
        "不要写句子或自造类别，否则会被当作无效并退回本地启发式判断。"
        "论文范式、跨学科性质或非常规方法请写在 paper_type_detail 里。"
        "reading_notes 用中文简要说明中部章节是否充分覆盖。"
    )
    payload = {"run_id": run_id, "paper": paper.metadata, "context_evidences": context_pack}
    try:
        agent_context = call_plain_json(prompt, payload, config)
    except Exception as first_exc:
        try:
            # 缩小证据包重试一次：输出被截断是这条链路最常见的失败原因。
            agent_context = call_plain_json(
                prompt,
                {"run_id": run_id, "paper": paper.metadata, "context_evidences": context_pack[:12]},
                config,
            )
        except Exception as second_exc:
            # 仍然失败也不终止整次评价：退回本地启发式画像，并在报告里显式标记。
            base_context["context_error"] = (
                "全文理解 Agent 两次调用均失败，已退回本地启发式画像；"
                "规则级判断仍基于正文证据，但缺少全文共享上下文，建议人工复核。"
                f"首次错误：{first_exc}；重试错误：{second_exc}"
            )
            base_context["context_evidence_ids"] = [item["evidence_id"] for item in context_pack]
            routing = build_dimension_evidence_routes(run_id, base_context, context_pack, profile, config)
            base_context.update(routing)
            return base_context
    profile_out = agent_context.get("paper_profile") if isinstance(agent_context, dict) else None
    if isinstance(profile_out, dict):
        merged_profile = {**base_context["paper_profile"], **profile_out}
        theoretical_move = _normalize_theoretical_move(merged_profile.get("theoretical_move"))
        method_design = _trim_mapping(
            merged_profile.get("method_design"),
            {
                "design_type": 120,
                "sample_data": 360,
                "identification_or_validity_strategy": 420,
                "limitations": 320,
            },
        )
        evidence_structure = _trim_mapping(
            merged_profile.get("evidence_structure"),
            {"main_evidence": 260, "robustness_or_validation": 260, "cross_section_support": 260},
        )
        merged_profile["theoretical_move"] = theoretical_move
        if method_design:
            merged_profile["method_design"] = method_design
        if evidence_structure:
            merged_profile["evidence_structure"] = evidence_structure
        if isinstance(merged_profile.get("most_important_pages"), list):
            pages = []
            for page in merged_profile["most_important_pages"][:8]:
                try:
                    pages.append(int(page))
                except (TypeError, ValueError):
                    continue
            merged_profile["most_important_pages"] = pages
        # 受控类型只认模型自己填的值。模型没给有效值时不再退回关键词启发式——
        # 实测启发式把 Choudhury 2020、Blaurock 2024 这类实证论文判成「综述」，
        # 会砍掉 20% 以上的适用权重。宁可退回中性的「混合」（适用全部规则、不误排除）。
        raw_type_text = str(profile_out.get("paper_type") or "").strip()
        model_class = normalize_paper_type(raw_type_text, None)
        paper_class = model_class or "混合"
        detail = str(profile_out.get("paper_type_detail") or "").strip()
        if detail:
            label = detail[:80]
        elif raw_type_text and raw_type_text not in PAPER_TYPE_CLASSES:
            label = raw_type_text[:80]
        else:
            label = heuristic_type
        merged_profile["paper_type_label"] = label
        merged_profile["paper_type"] = paper_class
        merged_profile["paper_type_source"] = "agent" if model_class else "fallback"
        base_context["paper_profile"] = merged_profile
        base_context["review_profile"] = build_review_profile(label)
    else:
        # 模型没返回可用画像：画像本身已由 context_incomplete 标记，类型一律按「混合」，
        # 避免用不可靠的启发式去裁剪适用规则集。
        base_context["paper_profile"]["paper_type"] = "混合"
        base_context["paper_profile"]["paper_type_source"] = "fallback"
    if isinstance(agent_context, dict) and "reading_notes" in agent_context:
        base_context["reading_notes"] = agent_context["reading_notes"]
    incompleteness = incomplete_profile_fields(base_context["paper_profile"])
    if incompleteness:
        base_context["context_incomplete"] = incompleteness
    base_context["context_evidence_ids"] = [item["evidence_id"] for item in context_pack]
    routing = build_dimension_evidence_routes(run_id, base_context, context_pack, profile, config)
    base_context.update(routing)
    return base_context
