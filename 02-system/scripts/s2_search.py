"""Semantic Scholar 检索 CLI：主题检索 / 逐篇论文检索 / 论文详情 / 引用脉络 / 作者。

匿名（无 key）实测可用性（2026-09-27，本机与代理出口一致）：
    /paper/search            429（共享池常年占满，退避 50 秒、6 轮重试 0/6 成功）
    /paper/search/bulk       200 稳定（6/6 轮，1.4-2.6 秒）—— 默认走这条
    /paper/{id} 详情          200（4/4 轮）
    /paper/{id}/citations    200
    /paper/{id}/references   200
    /author/{id}             200（4/4 轮）
只有「相关性排序」确实要 key（SEMANTIC_SCHOLAR_API_KEY，免费申请见 --help 输出）；其余匿名可用。
HTTP 细节都在 paper_eval/semantic_scholar.py；本脚本只做参数与展示。

用法：
    python scripts/s2_search.py "algorithmic management employee outcomes"
    python scripts/s2_search.py -q "AI hiring fairness" -q "algorithmic management" -n 10
    python scripts/s2_search.py --titles-file papers.txt          # 每行一个标题，逐篇检索
    python scripts/s2_search.py --pdf-dir <目录>                  # PDF 文件名当标题，逐篇检索
    python scripts/s2_search.py --paper "DOI:10.5465/amj.2020.0001"   # 详情（id 走 /paper/{id}）
    python scripts/s2_search.py --paper "OK computer worker perceptions of algorithmic recruitment"
    python scripts/s2_search.py --citations "DOI:10.1177/0008125619867910" --cite-limit 20
    python scripts/s2_search.py --references "DOI:10.1177/0008125619867910"
    python scripts/s2_search.py --author 1741101
    python scripts/s2_search.py "algorithmic management" --json runs/s2_out.json
    python scripts/s2_search.py --selftest                        # 离线自检，不打网络

有 key 时（SEMANTIC_SCHOLAR_API_KEY）加 --provider semantic_scholar,semantic_scholar_bulk
即可优先用相关性排序，429 时自动落到 bulk。
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
    """一次主题检索。核心模块负责缓存、重试、多源回退。"""
    items, error, provider = search_external(query, config)
    rows = [row_from_item(item, query, provider) for item in items if isinstance(item, dict)]
    return dedupe(rows)[:limit], error


def render(rows: list[dict], query: str) -> str:
    if not rows:
        return f"--- {query}\n    （无结果）"
    lines = [f"--- {query}  [{rows[0].get('provider')}]  {len(rows)} 条"]
    for index, row in enumerate(rows, 1):
        year = row.get("year") or "----"
        cites = row.get("citations")
        cites = f"{cites:,}" if isinstance(cites, int) else "-"
        lines.append(f"{index:>3}. {year}  被引 {cites:>7}  {row['title']}")
        tail = " | ".join(piece for piece in (row.get("venue"), row.get("doi")) if piece)
        if tail:
            lines.append(f"      {tail}")
    return "\n".join(lines)


def render_detail(item: dict, label: str) -> str:
    authors = [str(a.get("name")) for a in (item.get("authors") or []) if isinstance(a, dict)][:8]
    ids = item.get("externalIds") or {}
    cites = item.get("citationCount")
    lines = [
        f"--- 论文详情 {label}",
        f"    {item.get('title') or '(无标题)'}",
        f"    年份 {item.get('year') or '-'} | 被引 {cites if isinstance(cites, int) else '-'} | 期刊 {item.get('venue') or '-'}",
    ]
    if authors:
        lines.append(f"    作者 {', '.join(authors)}")
    if ids.get("DOI"):
        lines.append(f"    DOI {ids['DOI']}")
    if item.get("tldr"):
        tldr = item["tldr"]
        lines.append(f"    TLDR {(tldr.get('text') if isinstance(tldr, dict) else tldr) or ''}")
    abstract = str(item.get("abstract") or "").strip()
    if abstract:
        lines.append(f"    摘要 {abstract[:300]}{'…' if len(abstract) > 300 else ''}")
    return "\n".join(lines)


def render_author(payload: dict, label: str) -> str:
    lines = [
        f"--- 作者 {payload.get('name') or label} ({label})",
        f"    h 指数 {payload.get('hIndex') or '-'} | 论文 {payload.get('paperCount') or '-'} | 总被引 {payload.get('citationCount') or '-'}",
    ]
    papers = [p for p in (payload.get("papers") or []) if isinstance(p, dict)]
    papers.sort(key=lambda p: (p.get("citationCount") or 0), reverse=True)
    for index, paper in enumerate(papers[:10], 1):
        cites = paper.get("citationCount")
        cites_text = f"{cites:,}" if isinstance(cites, int) else "-"
        lines.append(
            f"{index:>3}. {paper.get('year') or '----'}  被引 {cites_text:>7}  {paper.get('title') or ''}"
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
    """离线自检：客户端自检 + 本脚本的行提取/渲染。不打网络。"""
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
    assert "被引" in text and "210" in text and "Algorithmic Management" in text, text
    detail = render_detail(
        {"title": "Attention is all you need", "year": 2017, "citationCount": 100000,
         "venue": "NeurIPS", "authors": [{"name": "A. Vaswani"}], "externalIds": {"DOI": "10.1/c"},
         "abstract": "The dominant sequence transduction models…"},
        "DOI:10.1/c",
    )
    assert "A. Vaswani" in detail and "DOI 10.1/c" in detail and "摘要" in detail, detail
    author_text = render_author({"name": "A. Scholar", "hIndex": 30, "paperCount": 2,
                                 "papers": [{"title": "P1", "year": 2020, "citationCount": 5}]}, "1234")
    assert "h 指数 30" in author_text and "P1" in author_text, author_text
    assert clean_paper_title("Choudhury et al. - 2020 - Algorithmic Management and Work") == "Algorithmic Management and Work"
    assert PAPER_ID_PATTERN.search("DOI:10.1/a") and not PAPER_ID_PATTERN.search("OK computer worker perceptions")
    print("CLI 自检通过：行提取 / DOI 去重 / 表格与详情与作者渲染 / id 与标题判别")


def main() -> int:
    parser = argparse.ArgumentParser(description="Semantic Scholar 检索（默认走匿名可用的 bulk 端点）")
    parser.add_argument("queries", nargs="*", help="主题关键词，可给多个")
    parser.add_argument("-q", "--query", action="append", default=[], help="追加一条主题检索（可重复）")
    parser.add_argument("-n", "--limit", type=int, default=10, help="每条检索展示多少条结果（默认 10）")
    parser.add_argument("--fetch-limit", type=int, default=20, help="单次请求从检索源取多少条（默认 20）")
    parser.add_argument("--provider", default="semantic_scholar_bulk",
                        help="检索源，逗号分隔按顺序回退；默认 semantic_scholar_bulk（免 key 可用）")
    parser.add_argument("--year-from", type=int, default=None)
    parser.add_argument("--year-to", type=int, default=None)
    parser.add_argument("--cache-dir", default=str(ROOT / "runs" / "s2_cli_cache"), help="检索缓存目录")
    parser.add_argument("--titles-file", help="每行一个论文标题（或 PDF 文件名），逐篇按主题检索")
    parser.add_argument("--pdf-dir", help="用该目录下 PDF 文件名当论文标题，逐篇检索")
    parser.add_argument("--paper", action="append", default=[],
                        help="论文详情：DOI:/ARXIV:/CorpusId: 等 id，或直接给标题（免 key 走 bulk 标题匹配）")
    parser.add_argument("--citations", action="append", default=[], help="拉被引脉络，值如 DOI:10.5465/amj.2020.0001")
    parser.add_argument("--references", action="append", default=[], help="拉参考文献脉络")
    parser.add_argument("--author", action="append", default=[], help="作者信息（需要 key，如 1741101）")
    parser.add_argument("--cite-limit", type=int, default=20, help="citations/references 各取多少条")
    parser.add_argument("--sleep", type=float, default=1.0, help="两次外部请求间隔秒数（默认 1.0）")
    parser.add_argument("--json", dest="json_out", help="把全部结果写成 JSON")
    parser.add_argument("--selftest", action="store_true", help="离线自检，不打网络")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        return 0

    load_local_env(ROOT)  # 让 .env 里的 SEMANTIC_SCHOLAR_API_KEY 在 CLI 里也生效

    queries = list(args.queries) + list(args.query) + paper_titles_from_args(args)
    if not any((queries, args.paper, args.citations, args.references, args.author)):
        parser.error("至少给一个主题、--titles-file / --pdf-dir / --paper / --citations / --references / --author")

    config = build_config(args)
    s2_config = S2Config(cache_dir=args.cache_dir)
    if not s2_config.has_key:
        print(f"[匿名] 未设 {s2_config.api_key_env}：检索走 bulk 池（无相关性排序，按被引降序）。"
              "想要相关性排序就填 key，免费申请 https://www.semanticscholar.org/product/api#api-key-form\n", flush=True)
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
        except Exception as exc:  # 单条失败不影响整批
            rows, error = [], f"{type(exc).__name__}: {exc}"
        if error:
            failures.append(f"{query} -> {error}")
            print(f"--- {query}\n    检索失败：{error}", flush=True)
        else:
            print(render(rows, query), flush=True)
        collected += rows

    for value in args.paper:
        pause()
        try:
            if PAPER_ID_PATTERN.search(value):
                item, label = paper_detail(value, s2_config), value
            else:
                item, label = find_by_title(value, s2_config), f"标题匹配「{value[:50]}」"
                if item is None:
                    raise S2Error("bulk 标题匹配不到足够接近的论文（可换关键词或 --paper 用 DOI）")
            print(render_detail(item, label), flush=True)
            collected.append(row_from_item(item, f"detail:{value}", "semantic_scholar"))
        except S2Error as exc:
            failures.append(f"detail {value} -> {exc}")
            print(f"--- 论文详情 {value}\n    失败：{exc}", flush=True)

    def run_lookup(label: str, func, paper_id: str) -> None:
        pause()
        try:
            rows = [row_from_item(item, f"{label}:{paper_id}", "semantic_scholar") for item in func(paper_id, args.cite_limit, s2_config)]
        except S2Error as exc:
            failures.append(f"{label} {paper_id} -> {exc}")
            print(f"--- {label} {paper_id}\n    失败：{exc}", flush=True)
            return
        print(render(dedupe(rows), f"{label} {paper_id}"), flush=True)
        collected.extend(dedupe(rows))

    for paper_id in args.citations:
        run_lookup("被引", s2_citations, paper_id)
    for paper_id in args.references:
        run_lookup("参考文献", s2_references, paper_id)

    for author_id in args.author:
        pause()
        try:
            payload = s2_author(author_id, s2_config)
        except S2Error as exc:
            failures.append(f"author {author_id} -> {exc}")
            print(f"--- 作者 {author_id}\n    失败：{exc}", flush=True)
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
        print(f"\nJSON: {out}（{len(collected)} 条，失败 {len(failures)} 条）")

    return 0 if collected else 1


if __name__ == "__main__":
    raise SystemExit(main())
