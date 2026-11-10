"""Client for Semantic Scholar paper search and metadata endpoints.

Anonymous-access checks on 2026-09-27 found the relevance-ranked search endpoint consistently rate-limited, while bulk search, paper details, citations, references, and author lookup were available in the tested runs. Without an API key, the client uses bulk search. A key enables relevance-ranked search and is read from the configured environment variable.

Requests can use a disk cache, retry transient errors while respecting Retry-After, and raise S2Error with an actionable message when retries are exhausted.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GRAPH_BASE = "https://api.semanticscholar.org/graph/v1"
SEARCH_ENDPOINT = f"{GRAPH_BASE}/paper/search"
BULK_ENDPOINT = f"{GRAPH_BASE}/paper/search/bulk"
SEARCH_FIELDS = "title,abstract,year,venue,authors,citationCount,externalIds,publicationTypes"
DETAIL_FIELDS = SEARCH_FIELDS + ",referenceCount,tldr,fieldsOfStudy,publicationDate,openAccessPdf"
RETRIABLE_MARKERS = ("429", "too many requests", "503", "502", "504", "timed out", "timeout", "remote end closed")
_LAST_CALL_AT = 0.0


class S2Error(RuntimeError):
    'External retrieval failed. The message contains actionable reasons (how long to wait for current limiting/whether a key is required).'


@dataclass(frozen=True)
class S2Config:
    api_key_env: str = "SEMANTIC_SCHOLAR_API_KEY"
    timeout_seconds: int = 20
    request_retries: int = 4
    retry_backoff_seconds: float = 2.0
    max_retry_after_seconds: float = 30.0
    min_interval_seconds: float = 1.0
    cache_dir: str | None = None

    @property
    def has_key(self) -> bool:
        return bool(os.environ.get(self.api_key_env))

    @classmethod
    def from_external(cls, config: Any) -> "S2Config":
        "Adapts from ExternalRetrievalConfig. The retry and inter-source intervals are left to the pipeline's own mechanism and are not superimposed here."
        return cls(
            api_key_env=str(getattr(config, "api_key_env", cls.api_key_env)),
            timeout_seconds=int(getattr(config, "timeout_seconds", cls.timeout_seconds)),
            request_retries=1,
            min_interval_seconds=0.0,
            cache_dir=getattr(config, "cache_dir", None),
        )


def _cache_file(cache_dir: str | None, key: str) -> Path | None:
    if not cache_dir:
        return None
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"s2_{digest}.json"


def _throttle(config: S2Config) -> None:
    global _LAST_CALL_AT
    wait = config.min_interval_seconds - (time.monotonic() - _LAST_CALL_AT)
    if wait > 0:
        time.sleep(wait)
    _LAST_CALL_AT = time.monotonic()


def _retry_after_seconds(exc: Exception) -> float | None:
    headers = getattr(exc, "headers", None)
    raw = headers.get("Retry-After") if headers else None
    if not raw:
        return None
    try:
        return float(str(raw).strip())
    except ValueError:
        return None


def _http_get_json(url: str, timeout: int, api_key: str | None) -> dict:
    'The only network exit. Self-test replaces this function (see selftest).'
    headers = {"Accept": "application/json", "User-Agent": "paper-eval/1.0"}
    if api_key:
        headers["x-api-key"] = api_key
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _get(url: str, cache_key: str, config: S2Config) -> dict:
    cached = _cache_file(config.cache_dir, cache_key)
    if cached and cached.exists():
        try:
            return json.loads(cached.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    attempts = max(1, int(config.request_retries))
    last_error: Exception | None = None
    for attempt in range(attempts):
        _throttle(config)
        try:
            payload = _http_get_json(url, config.timeout_seconds, os.environ.get(config.api_key_env))
        except Exception as exc:  # noqa: BLE001 - You need to judge whether it can be retried based on the error text and response header.
            last_error = exc
            message = str(exc).lower()
            if not any(marker in message for marker in RETRIABLE_MARKERS):
                raise S2Error(f"{exc}") from exc
            wait = _retry_after_seconds(exc)
            if wait is not None and wait > config.max_retry_after_seconds:
                raise S2Error(
                    f"{exc} (required to wait {wait:.0f} seconds, exceeding the upper limit {config.max_retry_after_seconds:.0f} seconds:"
                    'Anonymous shared pool current limit, fill in SEMANTIC_SCHOLAR_API_KEY to unlock)'
                ) from exc
            if attempt + 1 >= attempts:
                raise S2Error(
                    f"{exc} (anonymous shared pool current limit, measured /paper/search year-round 429:"
                    'Use bulk endpoint instead, or fill in SEMANTIC_SCHOLAR_API_KEY)'
                ) from exc
            time.sleep(wait if wait is not None else config.retry_backoff_seconds * (2 ** attempt))
        else:
            if cached:
                cached.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            return payload
    raise S2Error(str(last_error or 'Unknown failure'))


def _year_param(year_from: int | None, year_to: int | None) -> str | None:
    if not year_from:
        return None
    return f"{year_from}-{year_to or ''}"


def search_relevance(
    query: str,
    limit: int = 20,
    config: S2Config | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
) -> list[dict[str, Any]]:
    'Relevance search /paper/search. Anonymous actual measurement 0/6 successful (429 all year round) - please use search_bulk for key-free scenarios.'
    cfg = config or S2Config()
    params = {"query": query, "limit": str(limit), "fields": SEARCH_FIELDS}
    year = _year_param(year_from, year_to)
    if year:
        params["year"] = year
    url = SEARCH_ENDPOINT + "?" + urllib.parse.urlencode(params)
    payload = _get(url, f"rel|{url}", cfg)
    return [item for item in (payload.get("data") or []) if isinstance(item, dict)]


def search_bulk(
    query: str,
    limit: int = 20,
    config: S2Config | None = None,
    year_from: int | None = None,
    year_to: int | None = None,
    sort: str = "citationCount:desc",
) -> list[dict[str, Any]]:
    'Batch search /paper/search/bulk (available without key).\n\n    ponytail: bulk does not have relevance sorting, it can only sort=citationCount:desc, and the semantics degenerate into\n    "The top N articles with the highest citations among the batch with the most hits." The upper limit is here; just use search_relevance instead after getting the key.'
    cfg = config or S2Config()
    params = {"query": query, "fields": SEARCH_FIELDS, "sort": sort}
    year = _year_param(year_from, year_to)
    if year:
        params["year"] = year
    url = BULK_ENDPOINT + "?" + urllib.parse.urlencode(params)
    payload = _get(url, f"bulk|{url}", cfg)
    rows = [item for item in (payload.get("data") or []) if isinstance(item, dict)]
    return rows[: max(1, int(limit))]


def paper_detail(paper_id: str, config: S2Config | None = None) -> dict[str, Any]:
    'Paper details /paper/{id}. Anonymous actual test 4/4 successful (occasionally 429, relying on retrying). DETAIL_FIELDS has more TLDRs/references than the search.'
    cfg = config or S2Config()
    url = f"{GRAPH_BASE}/paper/{urllib.parse.quote(paper_id)}?" + urllib.parse.urlencode({"fields": DETAIL_FIELDS})
    return _get(url, f"detail|{url}", cfg)


def _normalized_title(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()


def find_by_title(title: str, config: S2Config | None = None, year_from: int | None = None) -> dict[str, Any] | None:
    'Locate papers by title (in bulk, anonymously available). Returns the closest one, returns None if the difference is too far.'
    cfg = config or S2Config()
    cleaned = _normalized_title(title)
    if len(cleaned) < 8:
        return None
    candidates = search_bulk(title, limit=10, config=cfg, year_from=year_from)
    best: dict[str, Any] | None = None
    best_score = 0.0
    for item in candidates:
        candidate = _normalized_title(item.get("title") or "")
        if not candidate:
            continue
        shorter, longer = sorted((cleaned, candidate), key=len)
        if shorter and shorter in longer:
            score = 0.6 + 0.4 * len(shorter) / len(longer)
        else:
            left, right = set(cleaned.split()), set(candidate.split())
            score = len(left & right) / max(1, len(left | right))
        if score > best_score:
            best, best_score = item, score
    return best if best_score >= 0.6 else None


def _nested_papers(payload: dict, key: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in payload.get("data") or []:
        paper = entry.get(key) if isinstance(entry, dict) else None
        if isinstance(paper, dict) and paper.get("title"):
            rows.append(paper)
    return rows


def citations(paper_id: str, limit: int = 20, config: S2Config | None = None) -> list[dict[str, Any]]:
    'Citation context: Who cited this article (anonymous available). paper_id supports DOI:/ARXIV:/CorpusId:/naked S2 id.'
    cfg = config or S2Config()
    url = f"{GRAPH_BASE}/paper/{urllib.parse.quote(paper_id)}/citations?" + urllib.parse.urlencode(
        {"fields": SEARCH_FIELDS, "limit": str(limit)}
    )
    return _nested_papers(_get(url, f"cites|{url}", cfg), "citingPaper")[:limit]


def references(paper_id: str, limit: int = 20, config: S2Config | None = None) -> list[dict[str, Any]]:
    'Reference context: Who is cited in this article (anonymous available).'
    cfg = config or S2Config()
    url = f"{GRAPH_BASE}/paper/{urllib.parse.quote(paper_id)}/references?" + urllib.parse.urlencode(
        {"fields": SEARCH_FIELDS, "limit": str(limit)}
    )
    return _nested_papers(_get(url, f"refs|{url}", cfg), "citedPaper")[:limit]


def author(author_id: str, config: S2Config | None = None, papers_limit: int = 100) -> dict[str, Any]:
    'Author information /author/{id} (paper list + h index). Anonymous test 4/4 successful.'
    cfg = config or S2Config()
    fields = "name,hIndex,citationCount,paperCount,affiliations,papers.title,papers.year,papers.venue,papers.citationCount,papers.externalIds"
    url = f"{GRAPH_BASE}/author/{urllib.parse.quote(author_id)}?" + urllib.parse.urlencode(
        {"fields": fields, "limit": str(papers_limit)}
    )
    return _get(url, f"author|{url}", cfg)


def selftest() -> None:
    'Offline self-test: replace the only network outlet, verify parsing, downgrade, cache, and error prompts. No real network access.'

    def make_item(title: str, year: int, cites: int, doi: str) -> dict:
        return {"title": title, "year": year, "venue": "AMJ", "citationCount": cites, "externalIds": {"DOI": doi}, "abstract": "x"}

    calls: list[str] = []

    def fake(url: str, timeout: int, api_key: str | None) -> dict:
        calls.append(url)
        if "/paper/search/bulk" in url:
            return {"data": [make_item("Algorithmic Management and Work", 2021, 210, "10.1/a"),
                             make_item("Fairness in AI Hiring", 2023, 42, "10.1/b"),
                             make_item("Unrelated Topic", 1999, 1, "10.1/c")]}
        if "/paper/search?" in url:
            raise urllib.error.HTTPError(url, 429, "Too Many Requests", None, None)
        if "/citations?" in url:
            return {"data": [{"citingPaper": make_item("Citing Paper", 2024, 3, "10.2/a")}]}
        if "/references?" in url:
            return {"data": [{"citedPaper": make_item("Cited Paper", 1990, 900, "10.2/b")}]}
        if "/author/" in url:
            return {"name": "A. Scholar", "hIndex": 30, "papers": []}
        raise AssertionError(f"Uncovered URL: {url}")

    real = globals()["_http_get_json"]
    globals()["_http_get_json"] = fake
    try:
        cache_dir = str(Path(__file__).resolve().parents[1] / "runs" / "_s2_selftest_cache")
        cfg = S2Config(cache_dir=cache_dir, request_retries=1, min_interval_seconds=0.0)

        rows = search_bulk("algorithmic management", limit=2, config=cfg)
        assert len(rows) == 2, rows
        assert rows[0]["externalIds"]["DOI"] == "10.1/a", rows[0]

        found = find_by_title("Algorithmic Management and Work", config=cfg)
        assert found and found["citationCount"] == 210, found
        assert find_by_title("Totally Different Subject Entirely", config=cfg) is None

        assert [r["title"] for r in citations("DOI:10.1/a", config=cfg)] == ["Citing Paper"]
        assert [r["title"] for r in references("DOI:10.1/a", config=cfg)] == ["Cited Paper"]
        assert author("1234", config=cfg)["hIndex"] == 30

        try:
            search_relevance("algorithmic management", config=cfg)
        except S2Error as exc:
            assert "429" in str(exc) and "SEMANTIC_SCHOLAR_API_KEY" in str(exc), str(exc)
        else:
            raise AssertionError('Relevance retrieval 429 should throw S2Error')

        before = len(calls)
        search_bulk("algorithmic management", limit=2, config=cfg)
        assert len(calls) == before, 'The second time the same search should hit the disk cache, no more network calls'
    finally:
        globals()["_http_get_json"] = real
        for path in Path(cache_dir).glob("*.json"):
            path.unlink()
        Path(cache_dir).rmdir()
    print('S2 client self-test passed: bulk truncation / title matching / citation and reference unpacking / author / 429 prompt / disk cache')


if __name__ == "__main__":
    selftest()
