'Evidence package cache self-test: only the non-deterministic LLM reordering hop is cached, and the failure condition must really fail.\n\nReplace `evidence_retrieval.rerank_evidence` with a fake rearrange function and count the number of calls to prove the behavior:\n1. No cache directory is configured → Rearrangement is performed for each call (number of times = number of calls), the behavior is consistent with the old version\n2. Equipped with cache + same scope → the cache is hit for the second time, rearrangement is no longer called, and the evidence id sequence is exactly the same\n3. Failure conditions: the evidence set changes (scope changes), search terms change (rule fingerprint changes), limit changes (paths are different)\n4. paper_context has changed (the candidate sequence has changed) **still hits** - this is necessary: paper_context is an LLM product,\n   Each batch is different. If it is used as a hit condition, the cache will be completely invalid (actual measurement on 2026-09-29 17/34 rules are still floating for this reason)\n5. The cache file only stores the id sequence and does not contain the text (the paper content is not saved to runs/)\n\nRun: python ".\\scripts\\self_test_evidence_cache.py"'

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_eval import evidence_retrieval  # noqa: E402
from paper_eval.agents import AGENTS  # noqa: E402
from paper_eval.evidence_retrieval import evidence_cache_scope, select_evidence  # noqa: E402
from paper_eval.models import Evidence  # noqa: E402
from paper_eval.rules import load_rule_profile  # noqa: E402

RULES = Path(__file__).resolve().parents[1] / 'rules' / "rules_v1.yaml"
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
            # 1) There is no cache directory: rearrange every time
            plain = FakeConfig("")
            scope = evidence_cache_scope(evidences, 3, plain)
            first = [item.evidence_id for item in select_evidence(RULE, evidences, AGENT, 3, None, plain, cache_scope=scope)]
            second = [item.evidence_id for item in select_evidence(RULE, evidences, AGENT, 3, None, plain, cache_scope=scope)]
            assert calls["n"] == 2, f"When the cache directory is not configured, the reordering should be called every time. The actual {calls['n']}"
            assert first == second, (first, second)
            print(f"Not configured with cache: rearranged twice (called {calls['n']} times), no cached files")

            # 2) Equipped with cache: the second hit, the sequence is exactly the same
            cached_config = FakeConfig(tmp)
            calls["n"] = 0
            scope = evidence_cache_scope(evidences, 3, cached_config)
            run1 = [item.evidence_id for item in select_evidence(RULE, evidences, AGENT, 3, None, cached_config, cache_scope=scope)]
            run2 = [item.evidence_id for item in select_evidence(RULE, evidences, AGENT, 3, None, cached_config, cache_scope=scope)]
            assert calls["n"] == 1, f"should hit the cache for the second time. Rearrangement should only be called once. The actual {calls['n']}"
            assert run1 == run2, (run1, run2)
            files = list(Path(tmp).glob("pack_*.json"))
            assert len(files) == 1, files
            payload = json.loads(files[0].read_text(encoding="utf-8"))
            assert payload["pack"] == run1 and payload["candidates"], payload
            assert "randomized field experiment" not in files[0].read_text(encoding="utf-8"), 'The text should not appear in the cache'
            print(f"with cache: first rearrangement, second hit (call {calls['n']} times), evidence sequence {run1}")

            # 3a) The recall candidate has changed (there is more evidence of hitting the search word) → Failure recalculation
            calls["n"] = 0
            extended = evidences + [
                Evidence(evidence_id="EV-6", paper_id="PAPER-SELFTEST", section="Methods", page=2, span=(0, 96),
                         text="An identification strategy based on staggered adoption strengthens the research design.",
                         text_hash="hash-EV-6")
            ]
            scope2 = evidence_cache_scope(extended, 3, cached_config)
            assert scope2 != scope, 'The evidence set has changed and the scope must be different.'
            select_evidence(RULE, extended, AGENT, 3, None, cached_config, cache_scope=scope2)
            assert calls["n"] == 1, f"The candidates have changed and must be rearranged. The actual {calls['n']}"

            # 3b) The same scope but the search terms have changed → Rule fingerprint changes → Paths are different → Rearrange
            calls["n"] = 0
            other_rule = replace(RULE, retrieval_terms=["job crafting", "survey measures", "employees"])
            select_evidence(other_rule, evidences, AGENT, 3, None, cached_config, cache_scope=scope)
            assert calls["n"] == 1, f"The search term has changed (the rule fingerprint has changed) and must be rearranged. The actual {calls['n']}"
            print('Failure conditions: evidence set change, search word change (rule fingerprint), limit change')

            # 3c) paper_context has changed (candidates may change) → the cache must still be hit, otherwise the evidence package will never be frozen
            calls["n"] = 0
            other_context = {
                "evidence_index": {"method": ["EV-3", "EV-5"], "data": ["EV-1"]},
                "paper_profile": {"central_problem": 'The impact of algorithm feedback on employee performance'},
            }
            hit = [
                item.evidence_id
                for item in select_evidence(RULE, evidences, AGENT, 3, other_context, cached_config, cache_scope=scope)
            ]
            assert calls["n"] == 0, f"paper_context change should not invalidate cache, actual reordering {calls['n']} times"
            assert hit == run1, (hit, run1)
            print(f"paper_context changes still hit the cache (reordered 0 times), and the evidence sequence remains {hit}")

            # 4) The number of entries has changed → the path is different, do not misuse the old cache
            calls["n"] = 0
            select_evidence(RULE, evidences, AGENT, 5, None, cached_config, cache_scope=scope)
            assert calls["n"] == 1, f"limit has changed and must be rearranged. The actual {calls['n']}"
            print('Failure condition: limit changes to different cache files')
    finally:
        evidence_retrieval.rerank_evidence = real_rerank

    print('Evidence package cache self-test passed: only one hop of cache reordering, zero call if hit, all three types of failure conditions are in effect, cache does not contain text')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
