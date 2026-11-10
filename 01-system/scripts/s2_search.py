"""Command-line interface for Semantic Scholar search and paper metadata.

Supports topic queries, title batches, DOI or paper lookups, citations, references, author lookup, and JSON output. It uses bulk search by default when no API key is configured; a key enables relevance-ranked search. Endpoint behavior and retry handling are implemented in paper_eval/semantic_scholar.py.

Examples:
    python scripts/s2_search.py "algorithmic management employee outcomes"
    python scripts/s2_search.py -q "AI hiring fairness" -q "algorithmic management" -n 10
    python scripts/s2_search.py --titles-file papers.txt
    python scripts/s2_search.py --pdf-dir <directory>
    python scripts/s2_search.py --paper "DOI:10.5465/amj.2020.0001"
    python scripts/s2_search.py --citations "DOI:10.1177/0008125619867910" --cite-limit 20
    python scripts/s2_search.py --references "DOI:10.1177/0008125619867910"
    python scripts/s2_search.py --author 1741101
    python scripts/s2_search.py --selftest
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from paper_eval.external_retrieval import (  # noqa: E402
    ExternalRetrievalConfig,
    clean_paper_title,
    search_external,
)
from paper_eval.llm import load_local_env  # noqa: E402
from paper_eval.semantic_scholar import S2Config, S2Error  # noqa: E402
from paper_eval.semantic_scholar import author as s2_author  # noqa: E402
from paper_eval.semantic_scholar import citations as s2_citations  # noqa: E402
from paper_eval.semantic_scholar import find_by_title, paper_detail  # noqa: E402
from paper_eval.semantic_scholar import references as s2_references  # noqa: E402
from paper_eval.semantic_scholar import selftest as s2_selftest  # noqa: E402

PAPER_ID_PATTERN = re.compile(r"^(?:DOI:|ARXIV:|CORPUSID:|MAG:|ACL:|PMID:|PMCID:|URL:)|^[0-9a-f]{40}$", re.I)


def row_from_item(item: dict, query: str, provider: str | None) -> dict:
    ids = item.get("externalIds") or {}
    return {
        "query": query,
        "provider": provider,
        "title": str(item.get("title") or "").strip(),
        "year": item.get("year"),
        "venue": str(item.get("venue") or ""),
        "citations": item.get("citationCount"),
        "doi": ids.get("DOI") or "",
        "paper_id": item.get("paperId") or "",
    }


def dedupe(rows: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique: list[dict] = []
    for row in rows:
        key = (row.get("doi") or row.get("title") or "").lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def search_topic(query: str, config: ExternalRetrievalConfig, limit: int) -> tuple[list[dict], str | None]:
    'A topic search. The core module is responsible for caching, retrying, and multi-source rollback.'
    items, error, provider = search_external(query, config)
    rows = [row_from_item(item, query, provider) for item in items if isinstance(item, dict)]
    return dedupe(rows)[:limit], error


def render(rows: list[dict], query: str) -> str:
    if not rows:
        return f"--- {query}\n (no results)"
    lines = [f"--- {query} [{rows[0].get('provider')}] {len(rows)}"]
    for index, row in enumerate(rows, 1):
        year = row.get("year") or "----"
        cites = row.get("citations")
        cites = f"{cites:,}" if isinstance(cites, int) else "-"
        lines.append(f"{index:>3}. {year} cited {cites:>7} {row['title']}")
        tail = " | ".join(piece for piece in (row.get("venue"), row.get("doi")) if piece)
        if tail:
            lines.append(f"      {tail}")
    return "\n".join(lines)


def render_detail(item: dict, label: str) -> str:
    authors = [str(a.get("name")) for a in (item.get("authors") or []) if isinstance(a, dict)][:8]
    ids = item.get("externalIds") or {}
    cites = item.get("citationCount")
    lines = [
        f"--- Paper details {label}",
        f"{item.get('title') or '(Untitled)'}",
        f"Year {item.get('year') or '-'} | Cited {cites if isinstance(cites, int) else '-'} | Journal {item.get('venue') or '-'}",
    ]
    if authors:
        lines.append(f"Author {', '.join(authors)}")
    if ids.get("DOI"):
        lines.append(f"    DOI {ids['DOI']}")
    if item.get("tldr"):
        tldr = item["tldr"]
        lines.append(f"    TLDR {(tldr.get('text') if isinstance(tldr, dict) else tldr) or ''}")
    abstract = str(item.get("abstract") or "").strip()
    if abstract:
        lines.append(f"Summary {abstract[:300]}{'…' if len(abstract) > 300 else ''}")
    return "\n".join(lines)


def render_author(payload: dict, label: str) -> str:
    lines = [
        f"--- Author {payload.get('name') or label} ({label})",
        f"h Index {payload.get('hIndex') or '-'} | Paper {payload.get('paperCount') or '-'} | Total citations {payload.get('citationCount') or '-'}",
    ]
    papers = [p for p in (payload.get("papers") or []) if isinstance(p, dict)]
    papers.sort(key=lambda p: (p.get("citationCount") or 0), reverse=True)
    for index, paper in enumerate(papers[:10], 1):
        cites = paper.get("citationCount")
        cites_text = f"{cites:,}" if isinstance(cites, int) else "-"
        lines.append(
            f"{index:>3}. {paper.get('year') or '----'} cited {cites_text:>7} {paper.get('title') or ''}"
        )
    return "\n".join(lines)


def paper_titles_from_args(args: argparse.Namespace) -> list[str]:
    titles: list[str] = []
    if args.titles_file:
        raw = Path(args.titles_file).read_text(encoding="utf-8").splitlines()
        titles += [clean_paper_title(line.strip()) for line in raw if line.strip() and not line.startswith("#")]
    if args.pdf_dir:
        for pdf in sorted(Path(args.pdf_dir).glob("*.pdf")):
            titles.append(clean_paper_title(pdf.stem))
    return titles


def build_config(args: argparse.Namespace) -> ExternalRetrievalConfig:
    providers = [name.strip() for name in args.provider.split(",") if name.strip()]
    return ExternalRetrievalConfig(
        enabled=True,
        providers=providers,
        per_query_limit=args.fetch_limit,
        year_from=args.year_from,
        year_to=args.year_to,
        cache_dir=args.cache_dir,
    )


def selftest() -> None:
    'Offline self-test: client self-test + row extraction/rendering of this script. No internet.'
    s2_selftest()
    provider = "semantic_scholar_bulk"
    items = [
        {"title": "Algorithmic Management and Work", "year": 2021, "venue": "AMJ",
         "citationCount": 210, "externalIds": {"DOI": "10.1/a"}},
        {"title": "Algorithmic Management and Work", "year": 2021, "venue": "AMJ",
         "citationCount": 210, "externalIds": {"DOI": "10.1/a"}},
        {"title": "Fairness in AI Hiring", "year": 2023, "venue": "HRM",
         "citationCount": 42, "externalIds": {"DOI": "10.1/b"}},
    ]
    rows = dedupe([row_from_item(item, "algorithmic management", provider) for item in items])
    assert len(rows) == 2 and rows[0]["doi"] == "10.1/a", rows
    text = render(rows[:1], "algorithmic management")
    assert 'cited' in text and "210" in text and "Algorithmic Management" in text, text
    detail = render_detail(
        {"title": "Attention is all you need", "year": 2017, "citationCount": 100000,
         "venue": "NeurIPS", "authors": [{"name": "A. Vaswani"}], "externalIds": {"DOI": "10.1/c"},
         "abstract": "The dominant sequence transduction models…"},
        "DOI:10.1/c",
    )
    assert "A. Vaswani" in detail and "DOI 10.1/c" in detail and 'Summary' in detail, detail
    author_text = render_author({"name": "A. Scholar", "hIndex": 30, "paperCount": 2,
                                 "papers": [{"title": "P1", "year": 2020, "citationCount": 5}]}, "1234")
    assert 'h index 30' in author_text and "P1" in author_text, author_text
    assert clean_paper_title("Choudhury et al. - 2020 - Algorithmic Management and Work") == "Algorithmic Management and Work"
    assert PAPER_ID_PATTERN.search("DOI:10.1/a") and not PAPER_ID_PATTERN.search("OK computer worker perceptions")
    print('CLI self-test passed: row extraction / DOI deduplication / table, detail and author rendering / id and title discrimination')


def main() -> int:
    parser = argparse.ArgumentParser(description='Semantic Scholar search (default is to anonymously available bulk endpoint)')
    parser.add_argument("queries", nargs="*", help='theme keywords, multiple')
    parser.add_argument("-q", "--query", action="append", default=[], help='Add a topic search (repeatable)')
    parser.add_argument("-n", "--limit", type=int, default=10, help='How many results are displayed for each search (default 10)')
    parser.add_argument("--fetch-limit", type=int, default=20, help='How many items to fetch from the search source in a single request (default 20)')
    parser.add_argument("--provider", default="semantic_scholar_bulk",
                        help='Search source, comma separated and returned in order; default semantic_scholar_bulk (key-free available)')
    parser.add_argument("--year-from", type=int, default=None)
    parser.add_argument("--year-to", type=int, default=None)
    parser.add_argument("--cache-dir", default=str(ROOT / "runs" / "s2_cli_cache"), help='Retrieve cache directory')
    parser.add_argument("--titles-file", help='One paper title (or PDF file name) per line, search by topic one by one')
    parser.add_argument("--pdf-dir", help='Use the PDF file name in this directory as the title of the paper and search one by one')
    parser.add_argument("--paper", action="append", default=[],
                        help='Paper details: DOI:/ARXIV:/CorpusId: and other ids, or give the title directly (no need to use key for bulk title matching)')
    parser.add_argument("--citations", action="append", default=[], help='pulls the cited context, the value is as DOI:10.5465/amj.2020.0001')
    parser.add_argument("--references", action="append", default=[], help='Pull reference context')
    parser.add_argument("--author", action="append", default=[], help='Author information (requires key, such as 1741101)')
    parser.add_argument("--cite-limit", type=int, default=20, help='How many citations/references should be taken?')
    parser.add_argument("--sleep", type=float, default=1.0, help='Seconds between two external requests (default 1.0)')
    parser.add_argument("--json", dest="json_out", help='Write all results as JSON')
    parser.add_argument("--selftest", action="store_true", help='Offline self-test, no network connection')
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return 0

    load_local_env(ROOT)  # Make SEMANTIC_SCHOLAR_API_KEY in .env also effective in CLI

    queries = list(args.queries) + list(args.query) + paper_titles_from_args(args)
    if not any((queries, args.paper, args.citations, args.references, args.author)):
        parser.error('Give at least one topic, --titles-file / --pdf-dir / --paper / --citations / --references / --author')

    config = build_config(args)
    s2_config = S2Config(cache_dir=args.cache_dir)
    if not s2_config.has_key:
        print(f"[Anonymous] Not set {s2_config.api_key_env}: Search the bulk pool (no relevance sorting, descending order by cited)."
              'If you want to sort by relevance, fill in the key and apply for free https://www.semanticscholar.org/product/api#api-key-form', flush=True)
    collected: list[dict] = []
    failures: list[str] = []
    first = True

    def pause() -> None:
        nonlocal first
        if not first and args.sleep > 0:
            time.sleep(args.sleep)
        first = False

    for query in queries:
        pause()
        try:
            rows, error = search_topic(query, config, args.limit)
        except Exception as exc:  # Single failure does not affect the entire batch
            rows, error = [], f"{type(exc).__name__}: {exc}"
        if error:
            failures.append(f"{query} -> {error}")
            print(f"--- {query}\n Retrieval failed: {error}", flush=True)
        else:
            print(render(rows, query), flush=True)
        collected += rows

    for value in args.paper:
        pause()
        try:
            if PAPER_ID_PATTERN.search(value):
                item, label = paper_detail(value, s2_config), value
            else:
                item, label = find_by_title(value, s2_config), f"title matches \"{value[:50]}\""
                if item is None:
                    raise S2Error('bulk The title does not match a paper that is close enough (you can change the keyword or use --paper with DOI)')
            print(render_detail(item, label), flush=True)
            collected.append(row_from_item(item, f"detail:{value}", "semantic_scholar"))
        except S2Error as exc:
            failures.append(f"detail {value} -> {exc}")
            print(f"--- Paper details {value}\n Failure: {exc}", flush=True)

    def run_lookup(label: str, func, paper_id: str) -> None:
        pause()
        try:
            rows = [row_from_item(item, f"{label}:{paper_id}", "semantic_scholar") for item in func(paper_id, args.cite_limit, s2_config)]
        except S2Error as exc:
            failures.append(f"{label} {paper_id} -> {exc}")
            print(f"--- {label} {paper_id}\n Failure: {exc}", flush=True)
            return
        print(render(dedupe(rows), f"{label} {paper_id}"), flush=True)
        collected.extend(dedupe(rows))

    for paper_id in args.citations:
        run_lookup('cited', s2_citations, paper_id)
    for paper_id in args.references:
        run_lookup('References', s2_references, paper_id)

    for author_id in args.author:
        pause()
        try:
            payload = s2_author(author_id, s2_config)
        except S2Error as exc:
            failures.append(f"author {author_id} -> {exc}")
            print(f"--- Author {author_id}\n Failure: {exc}", flush=True)
            continue
        print(render_author(payload, author_id), flush=True)
        collected += [
            row_from_item(paper, f"author:{author_id}", "semantic_scholar")
            for paper in (payload.get("papers") or [])
            if isinstance(paper, dict) and paper.get("title")
        ]

    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"results": collected, "failures": failures}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nJSON: {out} ({len(collected)} items, failed {len(failures)} items)")

    return 0 if collected else 1


if __name__ == "__main__":
    raise SystemExit(main())
