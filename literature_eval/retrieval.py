from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


FACETS = ["topic", "problem", "method", "data", "experiment", "contribution"]
FACET_HINTS = {
    "topic": ["title", "abstract", "introduction", "topic", "phenomenon", "主题", "现象"],
    "problem": ["gap", "puzzle", "problem", "question", "paradox", "研究问题", "缺口"],
    "method": ["method", "design", "measure", "model", "analysis", "方法", "设计", "测量"],
    "data": ["data", "sample", "dataset", "interview", "survey", "数据", "样本"],
    "experiment": ["result", "finding", "robust", "test", "table", "结果", "发现", "检验"],
    "contribution": ["contribution", "theory", "implication", "novel", "advance", "贡献", "理论", "启示"],
}


@dataclass
class IndexedDocument:
    doc_id: str
    title: str
    source_path: str
    publish_year: int | None
    facets: dict[str, str]
    vectors: dict[str, dict[str, int]]


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z-]{2,}|[\u4e00-\u9fff]{2,}", text.lower())
    return [token for token in tokens if token not in {"the", "and", "for", "with", "that", "this"}]


def vectorize(text: str) -> dict[str, int]:
    return dict(Counter(tokenize(text)))


def cosine(left: dict[str, int], right: dict[str, int]) -> float:
    if not left or not right:
        return 0.0
    common = set(left) & set(right)
    dot = sum(left[key] * right[key] for key in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def infer_year(text: str, fallback: str) -> int | None:
    matches = re.findall(r"\b(19[7-9]\d|20[0-3]\d)\b", fallback + "\n" + text[:2000])
    return int(matches[0]) if matches else None


def extract_facet_text(text: str, facet: str) -> str:
    paragraphs = [item.strip() for item in re.split(r"\n\s*\n", text) if len(item.strip()) >= 40]
    hints = FACET_HINTS[facet]
    selected = [p for p in paragraphs if any(hint.lower() in p.lower() for hint in hints)]
    if not selected:
        selected = paragraphs[:8]
    return "\n".join(selected[:20])


def build_index(text_dir: str | Path) -> list[IndexedDocument]:
    base = Path(text_dir)
    docs: list[IndexedDocument] = []
    for path in sorted(base.glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        facets = {facet: extract_facet_text(text, facet) for facet in FACETS}
        docs.append(
            IndexedDocument(
                doc_id=path.stem,
                title=path.stem,
                source_path=str(path),
                publish_year=infer_year(text, path.name),
                facets=facets,
                vectors={facet: vectorize(value) for facet, value in facets.items()},
            )
        )
    return docs


def save_index(docs: list[IndexedDocument], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([doc.__dict__ for doc in docs], ensure_ascii=False, indent=2), encoding="utf-8")


def load_index(path: str | Path) -> list[IndexedDocument]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [IndexedDocument(**item) for item in data]


def search_top_k(
    docs: list[IndexedDocument],
    query: str,
    facet: str = "topic",
    k: int = 5,
    before_year: int | None = None,
) -> list[dict[str, Any]]:
    if facet not in FACETS:
        raise ValueError(f"Unsupported facet: {facet}")
    query_vector = vectorize(query)
    rows = []
    for doc in docs:
        if before_year is not None and doc.publish_year is not None and doc.publish_year >= before_year:
            continue
        score = cosine(query_vector, doc.vectors.get(facet, {}))
        if score > 0:
            rows.append(
                {
                    "doc_id": doc.doc_id,
                    "title": doc.title,
                    "publish_year": doc.publish_year,
                    "facet": facet,
                    "score": score,
                    "source_path": doc.source_path,
                    "preview": doc.facets.get(facet, "")[:500],
                }
            )
    rows.sort(key=lambda item: item["score"], reverse=True)
    return rows[:k]

