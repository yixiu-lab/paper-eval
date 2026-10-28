from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from .claim_extraction import publication_year_of
from .models import Evidence, Paper


@dataclass(frozen=True)
class SocialIssue:
    period: str
    issue: str
    relevance: str
    keywords: tuple[str, ...]
    sources: tuple[str, ...]


SOCIAL_ISSUE_PROFILE: tuple[SocialIssue, ...] = (
    SocialIssue(
        period="2019",
        issue="可信与以人为本的 AI 治理",
        relevance="AI-HRM 论文若讨论公平、透明、可解释或责任治理，可连接到可信 AI 和人本价值议题。",
        keywords=("trustworthy ai", "human-centred", "human-centered", "fairness", "transparency", "accountability"),
        sources=("OECD AI Principles 2019",),
    ),
    SocialIssue(
        period="2020-2021",
        issue="疫情冲击下的远程工作、劳动安全与算法化组织",
        relevance="AI-HRM 论文若研究远程工作、员工安全、数字化协作或疫情后的组织调整，可连接到工作世界韧性问题。",
        keywords=("covid", "remote work", "work from home", "workplace safety", "resilience", "digital work"),
        sources=("ILO COVID-19 and the world of work monitor",),
    ),
    SocialIssue(
        period="2021-2022",
        issue="平台劳动、算法管理与劳动者自主性",
        relevance="AI-HRM 论文若研究算法管理、绩效监控、平台劳动或员工自主性，可连接到数字劳动中的权力与体面劳动问题。",
        keywords=("algorithmic management", "platform work", "digital labour", "surveillance", "autonomy", "monitoring"),
        sources=("ILO World Employment and Social Outlook 2021",),
    ),
    SocialIssue(
        period="2022-2023",
        issue="工作场所心理健康、员工福祉与 AI 压力",
        relevance="AI-HRM 论文若研究员工福祉、焦虑、工作压力、身份威胁或心理安全，可连接到 workplace well-being 问题。",
        keywords=("well-being", "wellbeing", "mental health", "stress", "anxiety", "psychological safety", "identity threat"),
        sources=("WHO/ILO mental health at work 2022",),
    ),
    SocialIssue(
        period="2023",
        issue="生成式 AI 对岗位、技能与工作重组的影响",
        relevance="AI-HRM 论文若研究 GenAI、技能替代、员工适应、岗位重塑或人机协作，可连接到工作转型与技能再配置问题。",
        keywords=("generative ai", "chatgpt", "large language model", "job redesign", "skills", "automation", "augmentation"),
        sources=("ILO Generative AI and Jobs 2023",),
    ),
    SocialIssue(
        period="2024-2025",
        issue="就业、招聘和员工管理中的高风险 AI 与合规治理",
        relevance="AI-HRM 论文若研究招聘筛选、绩效评价、员工画像或 HR analytics，可连接到高风险 AI、歧视防控和合规治理问题。",
        keywords=("recruitment", "hiring", "selection", "performance appraisal", "hr analytics", "high-risk ai", "discrimination"),
        sources=("EU AI Act 2024", "NIST AI Risk Management Framework 2023"),
    ),
)


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _score_issue(issue: SocialIssue, text: str, year: int | None) -> tuple[int, list[str]]:
    hits = [kw for kw in issue.keywords if kw in text]
    score = len(hits) * 2
    if year is not None:
        years = [int(item) for item in re.findall(r"\d{4}", issue.period)]
        if years and min(years) <= year <= max(years):
            score += 2
        elif years and abs(year - max(years)) <= 1:
            score += 1
    return score, hits


def select_social_issues(paper: Paper, paper_context: dict[str, Any] | None, limit: int = 3) -> list[dict[str, Any]]:
    profile = ((paper_context or {}).get("paper_profile") or {}) if isinstance(paper_context, dict) else {}
    parts = [
        str(paper.metadata.get("title") or ""),
        str(paper.metadata.get("abstract") or ""),
        str(profile.get("research_question") or ""),
        str(profile.get("research_gap") or ""),
        " ".join(str(item) for item in profile.get("contribution_claims") or []),
    ]
    text = _norm(" ".join(parts))
    year = publication_year_of(paper.metadata, paper.publish_date)

    rows: list[dict[str, Any]] = []
    for issue in SOCIAL_ISSUE_PROFILE:
        score, hits = _score_issue(issue, text, year)
        if score <= 0:
            continue
        rows.append(
            {
                "period": issue.period,
                "issue": issue.issue,
                "relevance": issue.relevance,
                "matched_keywords": hits,
                "sources": list(issue.sources),
                "score": score,
            }
        )
    rows.sort(key=lambda item: (-int(item["score"]), str(item["period"])))
    return rows[:limit]


def social_issue_evidence_blocks(paper: Paper, paper_context: dict[str, Any] | None, limit: int = 3) -> list[Evidence]:
    """Create auditable evidence blocks for social/global-problem relevance.

    The block is not a final judgement. It gives the F-SOCIAL bonus rule a
    versioned, inspectable prior about which public problem a paper may address.
    """
    issues = select_social_issues(paper, paper_context, limit=limit)
    blocks: list[Evidence] = []
    for idx, item in enumerate(issues, start=1):
        text = (
            f"Social issue grounding {idx}: period={item['period']}; issue={item['issue']}; "
            f"relevance={item['relevance']}; matched_keywords={', '.join(item['matched_keywords']) or 'none'}; "
            f"sources={'; '.join(item['sources'])}."
        )
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
        blocks.append(
            Evidence(
                evidence_id=f"SIV-{idx:03d}",
                paper_id=paper.paper_id,
                section="External Social Issue Grounding",
                page=0,
                span=(0, len(text)),
                text=text,
                text_hash=digest,
                source_type="social_issue_grounding",
            )
        )
    return blocks
