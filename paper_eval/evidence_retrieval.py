"""证据检索层：词袋召回 + LLM 重排。

为什么要有这一层：`agents.build_reading_pack` 是纯确定性启发式，靠硬编码英文关键词挑候选
证据，覆盖不全。实测 25 分的 D 维（研究设计与方法质量）平均只拿到 8.9 条候选，模型反复判
partial，理由是「候选证据未覆盖信度来源、效度检验说明」「未覆盖内生性或选择偏差」——这些
内容论文里有，只是关键词表没命中，压根没进候选包。

这里的做法是两段式：先用规则声明的英文检索词做词袋召回（能覆盖关键词表没写的表述），
再让模型从召回的候选里挑出真正该进阅读包的那些（能压掉词袋召回的假阳性）。
纯标准库，复用 retrieval.py 的 tokenize / vectorize / cosine，不引入任何第三方依赖。

性能：几百段 × 几十个查询词，纯 Python 毫秒级，不做倒排索引之类的优化。
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .external_retrieval import is_external_evidence
from .llm import AgentRuntimeConfig, call_plain_json
from .models import Evidence
from .retrieval import cosine, vectorize
from .rules import IndicatorRule

# 规则 id 与 evidence_location 都不是自由文本，抽出来的词大多没有检索价值
# （dimension 前缀、版本号、纯章节名）。这些词如果混进查询向量，会把整个向量归一化
# 拉偏，让真正的检索词权重变低。所以显式列出来丢掉。
_GENERIC_TERMS = {
    "the", "and", "for", "with", "that", "this", "from", "v1", "v2", "v3",
    "abstract", "introduction", "conclusion", "discussion", "methods", "method",
    "results", "result", "findings", "finding", "theory", "appendix", "literature",
    "review", "hypotheses", "tables", "figures", "paper", "study", "research",
    "section", "practical", "implications", "limitations", "data", "analysis",
    "supplementary", "materials", "availability", "design", "sample", "measures",
    "coding", "motivation", "robustness", "checks",
}
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z-]{2,}")


def _is_reference_evidence(item: Evidence) -> bool:
    """等价于 `agents.is_reference_evidence`。

    为什么在这里重写一遍而不用 import：agents.py 之后要 import 本模块（把 select_evidence
    接进 build_reading_pack 的调用点），本模块再反向 import agents 就成环。逻辑只有两行，
    复制比引入延迟导入的复杂度更划算；改动 agents.is_reference_evidence 时这里要同步。
    """
    return item.section.lower() == "references" or item.text.strip().lower().startswith(("references", "bibliography"))


def _excluded(item: Evidence) -> bool:
    """进不了检索池的证据：参考文献块没有实质内容；EXT-* 外部文献只显式插给指定规则。"""
    return _is_reference_evidence(item) or is_external_evidence(item)


def build_evidence_vectors(evidences: list[Evidence]) -> dict[str, dict[str, int]]:
    """evidence_id → 词频向量。键上带 section：同一段话在 Methods 和 Discussion 里检索价值不同。"""
    vectors: dict[str, dict[str, int]] = {}
    for item in evidences:
        if _excluded(item):
            continue
        vectors[item.evidence_id] = vectorize(f"{item.section}\n{item.text}")
    return vectors


def build_query_string(query_terms: list[str]) -> str:
    return " ".join(term for term in query_terms if term)


def recall_evidence(
    query_terms: list[str],
    vectors: dict[str, dict[str, int]],
    evidences_by_id: dict[str, Evidence],
    top_k: int,
) -> list[tuple[float, Evidence]]:
    """词袋召回：按余弦相似度降序返回前 top_k，0 分的丢掉。

    0 分直接丢：查询词一个都没命中时，余弦相似度是 0 而不是「有点相关」，
    留下它们只会把真正命中的证据挤出 top_k。
    """
    query_vector = vectorize(build_query_string(query_terms))
    if not query_vector:
        return []
    scored: list[tuple[float, Evidence]] = []
    for evidence_id, vector in vectors.items():
        item = evidences_by_id.get(evidence_id)
        if item is None:
            continue
        score = cosine(query_vector, vector)
        if score > 0:
            scored.append((score, item))
    # 同分时按 evidence_id 稳定排序，保证召回顺序可复现（LLM 只重排不从零生成，确定性很重要）。
    scored.sort(key=lambda pair: (-pair[0], pair[1].evidence_id))
    return scored[: max(0, int(top_k))]


def build_query_terms(rule: IndicatorRule, paper_context: dict[str, Any] | None = None) -> list[str]:
    """规则的英文检索词。

    优先用 `rule.retrieval_terms`（规则库里显式声明的英文词）。为空时回退到 rule_id 里的
    英文 token + evidence_location 的英文 token（中文定义不参与：规则锚点是中文、被评论文
    是英文，拿中文去搜英文正文恒不命中）。
    """
    terms = [str(term).strip().lower() for term in (getattr(rule, "retrieval_terms", None) or []) if str(term).strip()]
    if not terms:
        fallback: list[str] = []
        fallback.extend(match.lower() for match in _TOKEN_RE.findall(str(rule.rule_id)))
        for location in getattr(rule, "evidence_location", None) or []:
            fallback.extend(match.lower() for match in _TOKEN_RE.findall(str(location)))
        fallback.extend(match.lower() for match in _TOKEN_RE.findall(str(getattr(rule, "indicator_name", ""))))
        terms = [term for term in fallback if term not in _GENERIC_TERMS]

    deduped: list[str] = []
    seen: set[str] = set()
    for term in terms:
        if term and term not in seen:
            deduped.append(term)
            seen.add(term)
    return deduped


def _config_int(config: Any, name: str, default: int) -> int:
    """读配置里的正整数。配置文件里写成空值或字符串时不至于把流程打断。"""
    try:
        value = int(getattr(config, name, default))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def build_rerank_payload(
    rule: IndicatorRule,
    candidates: list[Evidence],
    limit: int,
    max_chars: int,
) -> dict[str, Any]:
    """重排调用的 user payload。证据文本在这里截断——截的是给模型看的那一份，不是最终阅读包。"""
    judgement_values = getattr(rule, "judgement_values", None) or {}
    return {
        "indicator_name": str(getattr(rule, "indicator_name", "") or ""),
        "definition": str(getattr(rule, "definition", "") or ""),
        "judgement_values": {str(key): str(value) for key, value in judgement_values.items()},
        "budget": int(limit),
        "candidates": [
            {
                "evidence_id": item.evidence_id,
                "section": item.section,
                "page": item.page,
                "text": (item.text or "")[: max(0, int(max_chars))],
            }
            for item in candidates
        ],
    }


# 重排 prompt。要求模型只做「筛选 + 排序 + 一句话理由」，不产出任何评分内容：
# 这一步的价值是把词袋召回的假阳性压掉，越权判档会污染后面的评分链路。
_RERANK_SYSTEM_PROMPT = (
    "你是学术论文证据筛选助手。给你一条评分规则（指标名、定义、档位锚点）和一批候选证据片段，"
    "你的唯一任务是从候选里挑出评判这条规则时评委真正需要读的片段，并按重要性从高到低排序。\n"
    "判断依据：这条规则的档位锚点里提到的每个要素，候选里有没有对应表述；片段是否描述了具体的"
    "做法、数据、检验或结果，而不是背景铺垫或泛泛论断。\n"
    "约束：\n"
    "1. 只能从候选里选，不得编造 evidence_id；\n"
    f"2. 最多选 {{budget}} 条（这是硬上限），候选够用就选满，明显不相关的宁可不选；\n"
    "3. 只做筛选与排序，不要给出档位判断或分数；\n"
    "4. 每条给一句话中文理由（30 字以内），说明它覆盖了规则的哪个要素。\n"
    "只返回一个 JSON 对象，不要 markdown，不要解释，格式：\n"
    '{"selected":[{"evidence_id":"<候选里的 evidence_id>","reason":"<一句话理由>"}]}'
)


def _extract_selected_ids(parsed: Any) -> tuple[list[str], dict[str, str]]:
    """从模型返回里抠出 evidence_id 列表与理由。

    容错到「只要 id 在任意位置出现就能用」的程度：模型偶尔返回纯文本（{"ids": [...]}）、
    偶尔返回 {"selected": [...]}、偶尔带理由对象。拿不到就返回空，由调用方走降级。
    """
    if not isinstance(parsed, dict):
        return [], {}
    raw = parsed.get("selected")
    if raw is None:
        raw = parsed.get("evidence_ids")
    if raw is None:
        raw = parsed.get("ids")
    if raw is None:
        return [], {}

    ids: list[str] = []
    reasons: dict[str, str] = {}
    if isinstance(raw, dict):
        raw = list(raw.items())
    if not isinstance(raw, list):
        return [], {}
    for entry in raw:
        if isinstance(entry, str):
            ids.append(entry)
        elif isinstance(entry, dict):
            evidence_id = entry.get("evidence_id") or entry.get("id")
            if evidence_id:
                ids.append(str(evidence_id))
                reason = entry.get("reason")
                if reason:
                    reasons[str(evidence_id)] = str(reason)
        elif isinstance(entry, (list, tuple)) and entry:
            ids.append(str(entry[0]))
            if len(entry) > 1 and entry[1]:
                reasons[str(entry[0])] = str(entry[1])
    # 还没拿到理由时退一步：把顶层 evidence_ids 和 reasons 两个平行列表按位置对齐。
    if not reasons:
        flat_ids = parsed.get("evidence_ids")
        flat_reasons = parsed.get("reasons")
        if isinstance(flat_ids, list) and isinstance(flat_reasons, list):
            for index, evidence_id in enumerate(flat_ids):
                if index < len(flat_reasons):
                    reasons[str(evidence_id)] = str(flat_reasons[index])
    return [str(item) for item in ids], reasons


def _call_rerank_llm(payload: dict[str, Any], config: AgentRuntimeConfig) -> Any:
    """单独抽一层：自检里 monkeypatch 这个函数就能换掉模型响应，不必真调 API。"""
    budget = payload.get("budget", 0)
    system_prompt = _RERANK_SYSTEM_PROMPT.replace("{budget}", str(budget))
    return call_plain_json(system_prompt, payload, config)


def _fill_to_limit(ordered: list[Evidence], padding: list[Evidence], limit: int) -> list[Evidence]:
    """按 evidence_id 去重，把 ordered 补到 limit 条（不足就返回现有的）。

    补齐用启发式阅读包的结果，而不是「模型选几条就几条」。原因有两层：
    1. 阅读包条数直接影响分数绝对值（实测 10 段 → 20 段整体上移约 7 分），条数缩水的规则
       和条数满的规则不可比；
    2. 更要命的是反向效果：检索层本意是扩大证据覆盖，如果召回命中少就让包从 10 条掉到 2 条，
       恒判 partial 的规则会判得更死——比不开检索层还糟。
    """
    seen = {item.evidence_id for item in ordered}
    result = list(ordered)
    for item in padding:
        if len(result) >= limit:
            break
        if item.evidence_id in seen:
            continue
        result.append(item)
        seen.add(item.evidence_id)
    return result


def rerank_evidence(
    rule: IndicatorRule,
    candidates: list[Evidence],
    config: AgentRuntimeConfig,
    limit: int,
    fallback: list[Evidence] | None = None,
) -> list[Evidence]:
    """一次 LLM 调用重排候选；任何异常/坏返回都退回 recall 顺序，绝不抛异常打断整篇评分。

    评分流程是一条链：某条规则的重排失败不能让这篇论文的评分整体失败——失败退回召回顺序，
    最差也只是退回「词袋召回 + 启发式补齐」这个次优候选包，比整篇失败可控。
    `fallback` 是 build_reading_pack 的结果，用来补足 limit；不传就只用 candidates 补。
    """
    if not candidates:
        return (fallback or [])[:limit]
    # 补齐池 = 召回顺序在前 + 启发式包在后，_fill_to_limit 会去重。
    # 补齐无条件补满 limit，不做「按模型选中数打折」：证据条数本身会移动分数
    # （实测 10 段 → 20 段整体上移约 7 分），新旧链路条数不一致，A/B 分差就分不清
    # 是「证据更相关」还是「证据更少」。补进来的段落也都过了旧启发式的相关度门槛。
    padding = [*candidates, *(fallback or [])]
    max_chars = _config_int(config, "evidence_rerank_max_chars", 240)
    payload = build_rerank_payload(rule, candidates, limit, max_chars)
    try:
        parsed = _call_rerank_llm(payload, config)
        # 模型给的理由暂时丢弃（ponytail: 当前只需要审计「最终进包的 evidence_id 对不对」，
        # 没有消费者。等 A+B 真实跑批发现重排挑偏了，再加 rerank_evidence 的 reasons_out 出参）。
        selected_ids, _reasons = _extract_selected_ids(parsed)
    except Exception:
        # LLM 报错 / 超时 / 非法 JSON（call_plain_json 内部会抛 RuntimeError）：静默降级。
        return _fill_to_limit(candidates[:limit], padding, limit)

    known = {item.evidence_id: item for item in candidates}
    ordered: list[Evidence] = []
    seen: set[str] = set()
    for evidence_id in selected_ids:
        item = known.get(evidence_id)
        if item is None or evidence_id in seen:
            continue
        ordered.append(item)
        seen.add(evidence_id)
    if not ordered:
        # id 全不认识 / 模型返回空：同样退回召回顺序。
        return _fill_to_limit(candidates[:limit], padding, limit)

    # 模型选不满 limit 时按召回顺序、再用启发式补齐（见 _fill_to_limit 的说明）。
    return _fill_to_limit(ordered, padding, limit)


PACK_CACHE_FORMAT = 2


def evidence_cache_scope(evidences: list[Evidence], limit: int, config: Any) -> str:
    """证据包缓存的作用域：论文证据集 + 条数 + 召回/截断参数（全是确定性输入）。

    为什么不把「候选序列」放进命中条件：候选来自检索词，而检索词来自 `paper_context`，后者是
    LLM 产物、**每批都不一样**（实测同一篇论文两次跑，paper_profile 的 8 个子字段全不同）。
    要求候选一致等于要求 LLM 自洽，缓存必然全失效——2026-09-29 实测 17/34 规则仍在漂就是这个原因。
    这里改成只认「论文没变、规则没变、预算没变」，让证据包真正冻结；代价是当 paper_context
    大改（换了画像口径）时，包仍是上一批选的那批，需要人工清缓存目录来强制重选。
    """
    payload = json.dumps(
        {
            "evidence": sorted(item.evidence_id for item in evidences),
            "limit": limit,
            "top_k": _config_int(config, "evidence_recall_top_k", 30),
            "rerank_max_chars": _config_int(config, "evidence_rerank_max_chars", 400),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _rule_fingerprint(rule: IndicatorRule) -> str:
    """规则指纹：改检索词/维度/适用范围等于换了一份规则，缓存必须失效。"""
    payload = json.dumps(
        {
            "rule_id": rule.rule_id,
            "dimension": rule.dimension_id,
            "terms": sorted(str(term) for term in (getattr(rule, "retrieval_terms", None) or [])),
            "locations": sorted(str(item) for item in (getattr(rule, "evidence_location", None) or [])),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _pack_cache_path(cache_dir: str, scope: str, rule: IndicatorRule, limit: int) -> Path:
    digest = hashlib.sha1(
        f"v{PACK_CACHE_FORMAT}|{scope}|{_rule_fingerprint(rule)}|{rule.rule_id}|{limit}".encode("utf-8")
    ).hexdigest()[:16]
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"pack_{digest}.json"


def _cached_pack(path: Path, evidences_by_id: dict[str, Evidence], limit: int) -> list[Evidence] | None:
    """命中条件：格式/条数一致，且包内每个 id 在当前证据集里都能找到（找不到说明论文解析变了）。"""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if data.get("format") != PACK_CACHE_FORMAT or data.get("limit") != limit:
        return None
    pack_ids = data.get("pack") or []
    if any(evidence_id not in evidences_by_id for evidence_id in pack_ids):
        return None
    return [evidences_by_id[evidence_id] for evidence_id in pack_ids]


def _store_pack(path: Path, candidates: list[Evidence], pack: list[Evidence], limit: int) -> None:
    """只存 id 序列，不存正文（缓存文件里没有论文内容，可以安全落在 runs/ 下）。"""
    path.write_text(
        json.dumps(
            {
                "format": PACK_CACHE_FORMAT,
                "limit": limit,
                "candidates": [item.evidence_id for item in candidates],
                "pack": [item.evidence_id for item in pack],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def select_evidence(
    rule: IndicatorRule,
    evidences: list[Evidence],
    agent: Any,
    limit: int,
    paper_context: dict[str, Any] | None = None,
    config: AgentRuntimeConfig | None = None,
    cache_scope: str | None = None,
) -> list[Evidence]:
    """证据选择总入口。父 agent 把 agents.build_reading_pack 的调用点换成这里即可。

    契约：返回条数**不小于**同参数下 build_reading_pack 的条数——正文够 limit 时两者都是 limit。
    开关关闭时行为与旧的 build_reading_pack 完全一致（逐条透传，不加任何包装），
    这样「关掉新链路 = 回到旧行为」是可以被自检证明的，不是靠阅读理解。

    `cache_scope` 给定时（配合配置里的 `evidence_pack_cache_dir`），LLM 重排那一跳会被缓存复用：
    只有这一跳是非确定的，召回与补齐都是纯确定性计算。
    """
    # agents.py 之后会 import 本模块，所以这里必须函数内延迟导入，模块级 import 会成环。
    from .agents import build_reading_pack

    if config is None or not bool(getattr(config, "evidence_retrieval_enabled", False)):
        return build_reading_pack(rule, evidences, agent, limit, paper_context)

    # 启发式包在这里就算好：既做召回为空时的整包回退，也做召回命中不足时的补齐。
    legacy = build_reading_pack(rule, evidences, agent, limit, paper_context)

    vectors = build_evidence_vectors(evidences)
    evidences_by_id = {item.evidence_id: item for item in evidences}
    query_terms = build_query_terms(rule, paper_context)
    top_k = _config_int(config, "evidence_recall_top_k", 30)
    recalled = recall_evidence(query_terms, vectors, evidences_by_id, top_k)
    if not recalled:
        # 召回为空（检索词与正文完全无交集）：回退启发式，别交空包给模型。
        return legacy

    candidates = [item for _score, item in recalled]
    if not bool(getattr(config, "evidence_rerank_enabled", True)):
        return _fill_to_limit(candidates[:limit], legacy, limit)

    cache_dir = str(getattr(config, "evidence_pack_cache_dir", "") or "")
    path = _pack_cache_path(cache_dir, cache_scope, rule, limit) if (cache_dir and cache_scope) else None
    if path is not None:
        cached = _cached_pack(path, evidences_by_id, limit)
        if cached is not None:
            return cached
    pack = rerank_evidence(rule, candidates, config, limit, legacy)
    if path is not None:
        _store_pack(path, candidates, pack, limit)
    return pack


__all__ = [
    "build_evidence_vectors",
    "build_query_string",
    "build_query_terms",
    "build_rerank_payload",
    "evidence_cache_scope",
    "recall_evidence",
    "rerank_evidence",
    "select_evidence",
]
