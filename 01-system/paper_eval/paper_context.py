from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from .llm import AgentRuntimeConfig, call_plain_json
from .models import Evidence, Paper
from .rules import PAPER_TYPE_CLASSES, RuleProfile, normalize_paper_type


SECTION_ALIASES = {
    "abstract": ["abstract", 'Summary'],
    "introduction": ["introduction", 'Introduction', "intro"],
    "theory": ["literature", "theory", "hypothesis", "background", "related work", 'Theory', 'Literature', 'Assumption'],
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
        'method',
        'design',
        'data',
        'sample',
        'Measurement',
    ],
    "results": ["result", "finding", "experiment", "evaluation", "analysis", "robust", 'results', 'found', 'Experiment', 'inspection'],
    "discussion": ["discussion", "conclusion", "limitation", "future", "implication", 'Discussion', 'Conclusion', 'Limitations', 'Enlightenment'],
    "tables_figures": ["tables and figures", "table", "figure", "fig.", 'table', 'diagram'],
    "appendix": ["appendix", "appendices", "supplementary", 'Appendix', 'Supplementary Materials'],
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
    "A": 'Topic value, research questions, research scope, problem boundaries, and operability.',
    "B": 'Theoretical contribution, challenging existing assumptions, integration of literature differences, cross-theoretical integration, new concepts or new mechanisms.',
    "C": 'Concept definition, theoretical premise, chain of reasoning, hypothesis or proposition, boundary conditions, alternative explanations.',
    "D": 'Research design, data sources, samples, measurements, model setting, identification strategy, method implementation.',
    "E": 'Presentation of results, correspondence between results and questions, robustness, strength of evidence, validity of conclusions, and discussion synthesis.',
    "F": 'Replicability, data coding transparency, limitations, external validity, practical implications, long-term impact.',
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
    'Theoretical or conceptual paper': {
        "primary_dimensions": ["A", "B", "C", "F"],
        "notes": 'focuses on problem positioning, theoretical dialogue, conceptual mechanisms, propositional logic and boundary conditions; samples, variables or regression models cannot be mandatory.',
    },
    'Empirical Research on Management': {
        "primary_dimensions": ["A", "C", "D", "E", "F"],
        "notes": 'focuses on research questions, design matching, variable measurement, sample data, identification strategies, result robustness and conclusion boundaries.',
    },
    'Qualitative or case study': {
        "primary_dimensions": ["A", "B", "C", "D", "E", "F"],
        "notes": 'focuses on research paradigm matching, case selection, data credibility, coding or interpretation transparency, theory generation and reflexivity.',
    },
    'Technical or machine learning paper': {
        "primary_dimensions": ["A", "D", "E", "F"],
        "notes": 'focuses on task value, method novelty, experimental design, benchmark comparison, reproducibility and application boundaries; the theoretical contribution dimension needs to be interpreted with caution.',
    },
    'Review Paper': {
        "primary_dimensions": ["A", "B", "C", "F"],
        "notes": 'focuses on literature coverage, classification framework, consensus integration, research agenda and boundary description; empirical identification strategy is not mandatory.',
    },
    'Review or opinion paper': {
        "primary_dimensions": ["A", "B", "C", "F"],
        "notes": 'focuses on problem insights, argumentation quality, advancement of existing discussions, and practical/governance implications; the method dimension usually requires manual review.',
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
        return 'Empirical Research on Management'
    if any(marker in lowered for marker in ["conceptual", "theory", "theoretical"]):
        return 'Theoretical or conceptual paper'
    if any(marker in lowered for marker in ["review", "meta-analysis"]):
        return 'Review Paper'
    if any(marker in lowered for marker in ["qualitative", "case study", "interview"]):
        return 'Qualitative or case study'
    if any(marker in lowered for marker in ["commentary", "viewpoint", "perspective"]):
        return 'Review or opinion paper'
    if any(marker in lowered for marker in ["machine learning", "algorithm", "technical"]):
        return 'Technical or machine learning paper'
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
        if any(marker in extra_lower for marker in ["breakthrough", 'Breakthrough', "field-changing"]):
            strength = "breakthrough"
        elif any(marker in extra_lower for marker in ["strong", "robust", "clear", 'clear', 'Stronger']):
            strength = "strong"
        elif any(marker in extra_lower for marker in ["weak", "limited", 'Insufficient', 'weaker']):
            strength = "weak"
        elif any(marker in extra_lower for marker in ["moderate", "partial", 'part', 'Medium']):
            strength = "moderate"
        else:
            strength = "unclear"
    return {"type": move_type, "description": description, "strength": strength}


def _paper_type(title: str, evidences: list[Evidence]) -> str:
    sample = " ".join(item.text.lower() for item in evidences[:20])
    text = f"{title.lower()} {sample}"
    if _matches(text, ["review", "systematic literature", "meta-analysis", 'Summary', 'meta-analysis']):
        return 'Review Paper'
    if _matches(text, ["editorial", "commentary", "perspective", "call for action", "viewpoint", 'Comments', 'Views']):
        return 'Review or opinion paper'
    if _matches(text, ["interview", "case study", "ethnograph", "qualitative", 'Interview', 'case', 'Qualitative']):
        return 'Qualitative or case study'
    if _matches(text, ["regression", "survey", "panel data", "sample", "hypotheses", "empirical", 'Evidence', 'Return', 'Questionnaire']):
        return 'Empirical Research on Management'
    if _matches(text, ["transformer", "neural", "architecture", "algorithm", "benchmark", "dataset", "machine translation"]):
        return 'Technical or machine learning paper'
    if _matches(text, ["theory", "conceptual", "proposition", "framework", 'Theory', 'concept', 'proposition']):
        return 'Theoretical or conceptual paper'
    return 'not recognized'


def build_review_profile(paper_type: str) -> dict[str, Any]:
    focus = PAPER_TYPE_RULE_FOCUS.get(paper_type, {})
    nonstandard_types = {'Theoretical or conceptual paper', 'Qualitative or case study', 'Review Paper', 'Review or opinion paper'}
    return {
        "paper_type": paper_type,
        "nonstandard_paradigm": paper_type in nonstandard_types,
        "primary_dimensions": focus.get("primary_dimensions", ["A", "B", "C", "D", "E", "F"]),
        "rule_focus_notes": focus.get("notes", 'First evaluate the general management paper. Dimensions that cannot be adapted need to be manually reviewed.\nWhen'),
        "anti_homogenization_policy": (
            'is identified as an unconventional research paradigm, it is not mandatory to apply the standard empirical template; but it still requires clear questions, rigorous argumentation, and sufficient evidence.'
        ),
        "high_score_evidence_policy": (
            'A high score must rely on multiple substantive evidences and cannot only rely on self-declarations or formatted statements in the abstract and introduction.'
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
        notes[dimension.dimension_id] = 'When the evidence routing agent is unavailable, candidate evidence is generated from the local chapter index; the final score is still judged by the expert agent.'
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
        "You are a full-text evidence routing agent. You've seen the full text of the paper, the double-page evidence package and the paper portrait."
        'Please do not assign mechanically according to PDF headers or chapter labels, but judge which evidence should be given to which expert based on the semantics of the content.'
        'Complete allocation quickly. Select at least 4 and up to 8 evidence_ids for each dimension; relevant content in the front, middle, and rear parts should be covered.'
        'A/B/C Do not select only abstracts; you must try to include concept definitions, literature dialogue, theoretical development or discussion paragraphs.'
        'D/E/F must try to include relevant paragraphs on methods, data, results, discussion, limitations or appendices.'
        'Only references to evidence_id present in the input context_evidences are allowed.'
        'output JSON:'
        '{"dimension_evidence_routes":{"A":["EV-..."],"B":[],"C":[],"D":[],"E":[],"F":[]},'
        '"routing_notes":{"A":"English summary","B":"English summary","C":"English summary","D":"English summary","E":"English summary","F":"English summary"}}'
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


PROFILE_PLACEHOLDER = 'To be judged by Agent based on the full text'

# A field in the full-text portrait that must be filled in by the Agent; if it is still a placeholder, it means that the portrait was not generated successfully.
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
            # paper_type is a controlled enumeration (empirical/qualitative/theoretical/review/mixed) and is only used within the scope of rule application;
            # The descriptive label is retained in paper_type_label for prompt words and report reading.
            "paper_type": normalize_paper_type(heuristic_type) or 'Mix',
            "paper_type_label": heuristic_type,
            "central_problem": 'To be judged by Agent based on the full text',
            "research_gap": 'To be judged by Agent based on the full text',
            "research_question": 'To be judged by Agent based on the full text',
            "core_claim": 'To be judged by Agent based on the full text',
            "main_contribution": 'To be judged by Agent based on the full text',
            "theoretical_move": {
                "type": "unclear",
                "description": 'To be judged by Agent based on the full text',
                "strength": "unclear",
            },
            "method_design": {
                "design_type": 'To be judged by Agent based on the full text',
                "sample_data": 'To be judged by Agent based on the full text',
                "identification_or_validity_strategy": 'To be judged by Agent based on the full text',
                "limitations": 'To be judged by Agent based on the full text',
            },
            "method_type": 'To be judged by Agent based on the full text',
            "data_or_material": 'To be judged by Agent based on the full text',
            "evidence_structure": {
                "main_evidence": [],
                "robustness_or_validation": [],
                "cross_section_support": [],
            },
            "main_findings": 'To be judged by Agent based on the full text',
            "boundary_conditions": 'To be judged by Agent based on the full text',
            "limitations": 'To be judged by Agent based on the full text',
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
        'You are the full text understanding Agent. Please first read through the given cross-page evidence package to form an overall picture of the paper.'
        'Must be based only on evidence, not fabrication; do not judge quality based on journal, author, or impact factor.'
        'Your output will serve as shared context for subsequent A-F expert agents, so help them understand the full text rather than just summarizing the first two pages.'
        'outputs JSON with keys including paper_profile and reading_notes.'
        'paper_profile must contain title, paper_type, paper_type_detail, central_problem, research_gap, research_question, core_claim, main_contribution,'
        "theoretical_move, method_design, method_type, data_or_material, evidence_structure, main_findings, boundary_conditions, "
        "limitations, most_important_pages, possible_top_journal_signals, blocking_weaknesses。"
        'theoretical_move must contain type, description, strength; type starts from'
        f"{', '.join(THEORETICAL_MOVE_TYPES)} Select the closest one."
        'method_design must contain design_type, sample_data, identification_or_validity_strategy, limitations.'
        'evidence_structure must contain main_evidence, robustness_or_validation, cross_section_support.'
        'possible_top_journal_signals only records internal signals of the text, such as important issues, new mechanisms, new constructs, strong identification, evidence closure, clear boundaries, and transferable frameworks;'
        'blocking_weaknesses records internal weaknesses that prevent entry to 85+.'
        'most_important_pages Fill in the page numbers that best support the rating, up to 8 pages.'
        f"paper_type must choose one of these 5 controlled values, and can only fill in the word itself: {'、'.join(PAPER_TYPE_CLASSES)};"
        'Do not write sentences or create your own categories, otherwise it will be regarded as invalid and the local heuristic judgment will be returned.'
        'Please write the paper paradigm, interdisciplinary nature or unconventional methods in paper_type_detail.'
        'reading_notes Briefly describe in English whether the main sections are adequately covered.'
    )
    payload = {"run_id": run_id, "paper": paper.metadata, "context_evidences": context_pack}
    try:
        agent_context = call_plain_json(prompt, payload, config)
    except Exception as first_exc:
        try:
            # Shrink Evidence Packet Retry: Output truncation is the most common cause of failure on this link.
            agent_context = call_plain_json(
                prompt,
                {"run_id": run_id, "paper": paper.metadata, "context_evidences": context_pack[:12]},
                config,
            )
        except Exception as second_exc:
            # still fails and does not terminate the entire evaluation: return the local heuristic portrait and mark it explicitly in the report.
            base_context["context_error"] = (
                'The full text understanding Agent failed to call twice and has returned to the local heuristic portrait;'
                'The rule-level judgment is still based on textual evidence, but lacks full-text sharing context. Manual review is recommended.'
                f"First error: {first_exc}; Retry error: {second_exc}"
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
        # The controlled type only recognizes the values filled in by the model itself. Keyword heuristics are no longer returned when the model does not give a valid value——
        # The measured heuristic classifies empirical papers such as Choudhury 2020 and Blaurock 2024 as "reviews".
        # will cut off more than 20% of the applicable weight. Rather fall back to a neutral "mix" (all rules apply, no false exclusions).
        raw_type_text = str(profile_out.get("paper_type") or "").strip()
        model_class = normalize_paper_type(raw_type_text, None)
        paper_class = model_class or 'Mix'
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
        # model did not return available images: the images themselves have been marked by context_incomplete, and the type must be "Mixed".
        # Avoid using unreliable heuristics to tailor the set of applicable rules.
        base_context["paper_profile"]["paper_type"] = 'Mix'
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
