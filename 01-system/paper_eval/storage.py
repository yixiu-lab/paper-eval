from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .models import AgentJudgement, Evidence, EvaluationRun, Paper, ReflectionIssue, ScoreItem, to_dict


SCHEMA = """
CREATE TABLE IF NOT EXISTS papers (
    paper_id TEXT PRIMARY KEY,
    metadata_json TEXT NOT NULL,
    sections_json TEXT NOT NULL,
    references_json TEXT NOT NULL,
    publish_date TEXT,
    parse_version TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evidences (
    evidence_id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL,
    section TEXT NOT NULL,
    page INTEGER NOT NULL,
    span_start INTEGER NOT NULL,
    span_end INTEGER NOT NULL,
    text TEXT NOT NULL,
    text_hash TEXT NOT NULL,
    source_type TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluation_runs (
    run_id TEXT PRIMARY KEY,
    paper_id TEXT NOT NULL,
    harness_version TEXT NOT NULL,
    model_versions_json TEXT NOT NULL,
    rule_profile TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    trace_json TEXT NOT NULL,
    report_json_path TEXT,
    report_markdown_path TEXT,
    report_html_path TEXT
);

CREATE TABLE IF NOT EXISTS agent_judgements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    agent_id TEXT NOT NULL,
    indicator_id TEXT NOT NULL,
    judgement TEXT NOT NULL,
    evidence_ids_json TEXT NOT NULL,
    reason TEXT NOT NULL,
    confidence REAL NOT NULL,
    needs_review INTEGER NOT NULL,
    agent_version TEXT NOT NULL,
    model_version TEXT NOT NULL,
    review_flags_json TEXT
);

CREATE TABLE IF NOT EXISTS score_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    indicator_id TEXT NOT NULL,
    judgement TEXT NOT NULL,
    rule_id TEXT NOT NULL,
    raw_score REAL,
    weighted_score REAL,
    weight REAL NOT NULL,
    dimension_id TEXT NOT NULL,
    dimension_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reflection_issues (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    issue_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    message TEXT NOT NULL,
    related_ids_json TEXT NOT NULL,
    suggested_action TEXT NOT NULL
);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str | Path) -> None:
    with connect(db_path) as conn:
        conn.executescript(SCHEMA)
        columns = {row[1] for row in conn.execute("PRAGMA table_info(evaluation_runs)").fetchall()}
        if "report_html_path" not in columns:
            conn.execute("ALTER TABLE evaluation_runs ADD COLUMN report_html_path TEXT")
        judgement_columns = {row[1] for row in conn.execute("PRAGMA table_info(agent_judgements)").fetchall()}
        if "review_flags_json" not in judgement_columns:
            conn.execute("ALTER TABLE agent_judgements ADD COLUMN review_flags_json TEXT")


def _json(value: Any) -> str:
    return json.dumps(to_dict(value), ensure_ascii=False)


def save_run_bundle(
    db_path: str | Path,
    run: EvaluationRun,
    paper: Paper,
    evidences: list[Evidence],
    judgements: list[AgentJudgement],
    score_items: list[ScoreItem],
    reflection_issues: list[ReflectionIssue],
    report_json_path: str | Path,
    report_markdown_path: str | Path,
    report_html_path: str | Path | None = None,
) -> None:
    init_db(db_path)
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO papers
            (paper_id, metadata_json, sections_json, references_json, publish_date, parse_version)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                paper.paper_id,
                _json(paper.metadata),
                _json(paper.sections),
                _json(paper.references),
                paper.publish_date,
                paper.parse_version,
            ),
        )
        conn.executemany(
            """
            INSERT OR REPLACE INTO evidences
            (evidence_id, paper_id, section, page, span_start, span_end, text, text_hash, source_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    item.evidence_id,
                    item.paper_id,
                    item.section,
                    item.page,
                    item.span[0],
                    item.span[1],
                    item.text,
                    item.text_hash,
                    item.source_type,
                )
                for item in evidences
            ],
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO evaluation_runs
            (run_id, paper_id, harness_version, model_versions_json, rule_profile, status, started_at,
             completed_at, trace_json, report_json_path, report_markdown_path, report_html_path)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run.run_id,
                run.paper_id,
                run.harness_version,
                _json(run.model_versions),
                run.rule_profile,
                run.status.value,
                run.started_at,
                run.completed_at,
                _json(run.trace),
                str(report_json_path),
                str(report_markdown_path),
                str(report_html_path) if report_html_path else None,
            ),
        )
        conn.execute("DELETE FROM agent_judgements WHERE run_id = ?", (run.run_id,))
        conn.executemany(
            """
            INSERT INTO agent_judgements
            (run_id, agent_id, indicator_id, judgement, evidence_ids_json, reason, confidence,
             needs_review, agent_version, model_version, review_flags_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run.run_id,
                    item.agent_id,
                    item.indicator_id,
                    item.judgement.value,
                    _json(item.evidence_ids),
                    item.reason,
                    item.confidence,
                    int(item.needs_review),
                    item.agent_version,
                    item.model_version,
                    _json(item.review_flags),
                )
                for item in judgements
            ],
        )
        conn.execute("DELETE FROM score_items WHERE run_id = ?", (run.run_id,))
        conn.executemany(
            """
            INSERT INTO score_items
            (run_id, indicator_id, judgement, rule_id, raw_score, weighted_score, weight,
             dimension_id, dimension_name)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run.run_id,
                    item.indicator_id,
                    item.judgement.value,
                    item.rule_id,
                    item.raw_score,
                    item.weighted_score,
                    item.weight,
                    item.dimension_id,
                    item.dimension_name,
                )
                for item in score_items
            ],
        )
        conn.execute("DELETE FROM reflection_issues WHERE run_id = ?", (run.run_id,))
        conn.executemany(
            """
            INSERT INTO reflection_issues
            (run_id, issue_type, severity, message, related_ids_json, suggested_action)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run.run_id,
                    item.issue_type,
                    item.severity,
                    item.message,
                    _json(item.related_ids),
                    item.suggested_action,
                )
                for item in reflection_issues
            ],
        )


def get_run_summary(db_path: str | Path, run_id: str) -> dict[str, Any] | None:
    with connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM evaluation_runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        judgements = conn.execute("SELECT COUNT(*) AS count FROM agent_judgements WHERE run_id = ?", (run_id,)).fetchone()
        scores = conn.execute("SELECT COUNT(*) AS count FROM score_items WHERE run_id = ?", (run_id,)).fetchone()
        issues = conn.execute("SELECT COUNT(*) AS count FROM reflection_issues WHERE run_id = ?", (run_id,)).fetchone()
        return {
            "run_id": row["run_id"],
            "paper_id": row["paper_id"],
            "status": row["status"],
            "rule_profile": row["rule_profile"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "report_json_path": row["report_json_path"],
            "report_markdown_path": row["report_markdown_path"],
            "report_html_path": row["report_html_path"],
            "judgement_count": judgements["count"],
            "score_item_count": scores["count"],
            "reflection_issue_count": issues["count"],
        }


def list_run_summaries(db_path: str | Path, limit: int = 50) -> list[dict[str, Any]]:
    if not Path(db_path).exists():
        return []
    with connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT run_id, paper_id, status, rule_profile, started_at, completed_at,
                   report_json_path, report_markdown_path, report_html_path
            FROM evaluation_runs
            ORDER BY started_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]
