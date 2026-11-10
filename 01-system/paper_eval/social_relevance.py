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
        issue='Trustworthy and human-centered AI governance',
        relevance='AI-HRM papers that discuss fairness, transparency, explainability or responsible governance can be connected to issues of trusted AI and humanistic values.',
        keywords=("trustworthy ai", "human-centred", "human-centered", "fairness", "transparency", "accountability"),
        sources=("OECD AI Principles 2019",),
    ),
    SocialIssue(
        period="2020-2021",
        issue='Remote work, labor safety and algorithmic organization under the impact of the epidemic',
        relevance='AI-HRM papers can be connected to the issue of resilience in the working world if they study remote working, employee safety, digital collaboration or organizational adjustments after the epidemic.',
        keywords=("covid", "remote work", "work from home", "workplace safety", "resilience", "digital work"),
        sources=("ILO COVID-19 and the world of work monitor",),
    ),
    SocialIssue(
        period="2021-2022",
        issue='Platform labor, algorithm management and worker autonomy',
        relevance='AI-HRM papers that study algorithmic management, performance monitoring, platform labor or employee autonomy can be connected to issues of power and decent work in digital labor.',
        keywords=("algorithmic management", "platform work", "digital labour", "surveillance", "autonomy", "monitoring"),
        sources=("ILO World Employment and Social Outlook 2021",),
    ),
    SocialIssue(
        period="2022-2023",
        issue='Workplace mental health, employee well-being and AI stress',
        relevance='AI-HRM papers that study employee well-being, anxiety, work stress, identity threats or psychological safety can be connected to workplace well-being issues.',
        keywords=("well-being", "wellbeing", "mental health", "stress", "anxiety", "psychological safety", "identity threat"),
        sources=("WHO/ILO mental health at work 2022",),
    ),
    SocialIssue(
        period="2023",
        issue='The impact of generative AI on the reorganization of jobs, skills and work\nIf the',
        relevance='AI-HRM paper studies GenAI, skill substitution, employee adaptation, job reshaping or human-machine collaboration, it can be connected to the issues of work transformation and skill reallocation.',
        keywords=("generative ai", "chatgpt", "large language model", "job redesign", "skills", "automation", "augmentation"),
        sources=("ILO Generative AI and Jobs 2023",),
    ),
    SocialIssue(
        period="2024-2025",
        issue='High-risk AI and compliance governance in employment, recruitment and workforce management',
        relevance='AI-HRM papers that study recruitment screening, performance evaluation, employee profiling or HR analytics can be connected to high-risk AI, discrimination prevention and control, and compliance governance issues.',
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
