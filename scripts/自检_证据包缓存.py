"""证据包缓存自检：只缓存非确定的 LLM 重排那一跳，且失效条件必须真的失效。

用假的重排函数替换 `evidence_retrieval.rerank_evidence`，数调用次数即可证明行为：
1. 未配缓存目录 → 每次调用都走重排（次数 = 调用次数），行为与旧版一致
2. 配了缓存 + 同一作用域 → 第二次命中缓存，重排不再被调用，且证据 id 序列完全相同
3. 失效条件：证据集变了（作用域变）、检索词变了（规则指纹变）、limit 变了（路径不同）
4. paper_context 变了（候选序列变了）**仍然命中** —— 这是必须的：paper_context 是 LLM 产物、
   每批都不同，若拿它当命中条件，缓存会全失效（2026-09-29 实测 17/34 规则仍在漂就是这个原因）
5. 缓存文件只存 id 序列，不含正文（论文内容不落盘到 runs/）

运行：python ".\\scripts\\自检_证据包缓存.py"
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from literature_eval import evidence_retrieval  # noqa: E402
from literature_eval.agents import AGENTS  # noqa: E402
from literature_eval.evidence_retrieval import evidence_cache_scope, select_evidence  # noqa: E402
from literature_eval.models import Evidence  # noqa: E402
from literature_eval.rules import load_rule_profile  # noqa: E402

RULES = Path(__file__).resolve().parents[1] / "评价规则库" / "rules_v1.yaml"
RULE = load_rule_profile(RULES).rules_by_id["D-DATA-SAMPLE-V1"]
AGENT = next(agent for agent in AGENTS if "D" in agent.dimension_ids)


class FakeConfig:
    def __init__(self, cache_dir: str = "") -> None:
        self.evidence_retrieval_enabled = True
        self.evidence_rerank_enabled = True
        self.evidence_recall_top_k = 30
        self.evidence_rerank_max_chars = 400
        self.evidence_pack_cache_dir = cache_dir


def make_evidences() -> list[Evidence]:
    texts = [
        ("EV-1", "Methods", "We use a randomized field experiment to identify the causal effect on employee retention."),
        ("EV-2", "Methods", "The sample covers 1200 employees across 30 firms with survey measures of job crafting."),
        ("EV-3", "Results", "The regression shows a significant effect of algorithmic feedback on performance measures."),
        ("EV-4", "Discussion", "Our research design cannot rule out selection bias in the sample of firms."),
        ("EV-5", "Introduction", "Algorithms increasingly inform managerial decisions about employees."),
    ]
    return [
        Evidence(
            evidence_id=evidence_id,
            paper_id="PAPER-SELFTEST",
            section=section,
            page=1,
            span=(0, len(text)),
            text=text,
            text_hash=f"hash-{evidence_id}",
        )
        for evidence_id, section, text in texts
    ]


def main() -> int:
    evidences = make_evidences()
    calls = {"n": 0}
    real_rerank = evidence_retrieval.rerank_evidence

    def fake_rerank(rule, candidates, config, limit, legacy):
        calls["n"] += 1
        return candidates[:limit]

    evidence_retrieval.rerank_evidence = fake_rerank
    try:
        with tempfile.TemporaryDirectory() as tmp:
            # 1) 没配缓存目录：每次都打重排
            plain = FakeConfig("")
            scope = evidence_cache_scope(evidences, 3, plain)
            first = [item.evidence_id for item in select_evidence(RULE, evidences, AGENT, 3, None, plain, cache_scope=scope)]
            second = [item.evidence_id for item in select_evidence(RULE, evidences, AGENT, 3, None, plain, cache_scope=scope)]
            assert calls["n"] == 2, f"未配缓存目录时重排应每次都调用，实际 {calls['n']}"
            assert first == second, (first, second)
            print(f"未配缓存：两次都走重排（调用 {calls['n']} 次），无缓存文件")

            # 2) 配了缓存：第二次命中，序列完全一致
            cached_config = FakeConfig(tmp)
            calls["n"] = 0
            scope = evidence_cache_scope(evidences, 3, cached_config)
            run1 = [item.evidence_id for item in select_evidence(RULE, evidences, AGENT, 3, None, cached_config, cache_scope=scope)]
            run2 = [item.evidence_id for item in select_evidence(RULE, evidences, AGENT, 3, None, cached_config, cache_scope=scope)]
            assert calls["n"] == 1, f"第二次应命中缓存，重排只该调用 1 次，实际 {calls['n']}"
            assert run1 == run2, (run1, run2)
            files = list(Path(tmp).glob("pack_*.json"))
            assert len(files) == 1, files
            payload = json.loads(files[0].read_text(encoding="utf-8"))
            assert payload["pack"] == run1 and payload["candidates"], payload
            assert "randomized field experiment" not in files[0].read_text(encoding="utf-8"), "缓存里不应出现正文"
            print(f"配缓存：第一次重排、第二次命中（调用 {calls['n']} 次），证据序列 {run1}")

            # 3a) 召回候选变了（多了命中检索词的证据）→ 失效重算
            calls["n"] = 0
            extended = evidences + [
                Evidence(evidence_id="EV-6", paper_id="PAPER-SELFTEST", section="Methods", page=2, span=(0, 96),
                         text="An identification strategy based on staggered adoption strengthens the research design.",
                         text_hash="hash-EV-6")
            ]
            scope2 = evidence_cache_scope(extended, 3, cached_config)
            assert scope2 != scope, "证据集变了，作用域必须不同"
            select_evidence(RULE, extended, AGENT, 3, None, cached_config, cache_scope=scope2)
            assert calls["n"] == 1, f"候选变了必须重新重排，实际 {calls['n']}"

            # 3b) 同一作用域但检索词变了 → 规则指纹变化 → 路径不同 → 重新重排
            calls["n"] = 0
            other_rule = replace(RULE, retrieval_terms=["job crafting", "survey measures", "employees"])
            select_evidence(other_rule, evidences, AGENT, 3, None, cached_config, cache_scope=scope)
            assert calls["n"] == 1, f"检索词变了（规则指纹变）必须重新重排，实际 {calls['n']}"
            print("失效条件：证据集变化、检索词变化（规则指纹）、limit 变化")

            # 3c) paper_context 变了（候选可能变）→ 仍必须命中缓存，否则证据包永远冻不住
            calls["n"] = 0
            other_context = {
                "evidence_index": {"method": ["EV-3", "EV-5"], "data": ["EV-1"]},
                "paper_profile": {"central_problem": "算法反馈对员工绩效的影响"},
            }
            hit = [
                item.evidence_id
                for item in select_evidence(RULE, evidences, AGENT, 3, other_context, cached_config, cache_scope=scope)
            ]
            assert calls["n"] == 0, f"paper_context 变化不应使缓存失效，实际重排 {calls['n']} 次"
            assert hit == run1, (hit, run1)
            print(f"paper_context 变化仍命中缓存（重排 0 次），证据序列保持 {hit}")

            # 4) 条数变了 → 路径不同，不误用旧缓存
            calls["n"] = 0
            select_evidence(RULE, evidences, AGENT, 5, None, cached_config, cache_scope=scope)
            assert calls["n"] == 1, f"limit 变了必须重新重排，实际 {calls['n']}"
            print("失效条件：limit 变化走不同缓存文件")
    finally:
        evidence_retrieval.rerank_evidence = real_rerank

    print("证据包缓存自检通过：只缓存重排一跳、命中即零调用、三类失效条件均生效、缓存不含正文")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
