'Evidence retrieval self-test: Rule English search terms, Chinese rule text are no longer used as search terms, C/D dimension exclusive recall, reading package limit.\n\nBackground (three lesions):\n  1) DIMENSION_KEYWORD_HINTS / RULE_KEYWORD_HINTS are mixed with self-examination sample papers (Zhai 2023, Educational Measurement /\n     CDM / Q-matrix topic), it has almost zero hits on the AI-HRM management papers that are really to be evaluated;\n  2) rule_keyword_hints uses regular expressions to extract the Chinese rule text into "keywords", and then uses it to search the English text item.text → always misses;\n  3) The C / D dimension does not have any exclusive retrieval link in build_reading_pack. D (25 points, the largest weight) has an average measured average of only\n     8.9 candidate pieces of evidence (A/B has 14-16.7).\n\nRun:\n    python ".\\scripts\\self_test_evidence_retrieval.py"'

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from paper_eval.agents import (
    AGENTS,
    DIMENSION_KEYWORD_HINTS,
    RULE_KEYWORD_HINTS,
    batch_response_instruction,
    build_reading_pack,
    rule_keyword_hints,
    score_method_rigor_hits,
    score_theory_rigor_hits,
)
from paper_eval.models import Evidence
from paper_eval.rules import load_rule_profile

ROOT = Path(__file__).resolve().parents[1]
RULES = ROOT / 'rules' / "rules_v1.yaml"
AGENT_FOR_DIMENSION = {dimension: agent for agent in AGENTS for dimension in agent.dimension_ids}

# Self-inspection sample paper (Zhai 2023: Educational Measurement / CDM / Q-matrix / Machine-Artificial Consistency) Exclusive terminology.
# These words falling into the search word list are pollution: after replacing them with AI-HRM management papers, they will only squeeze the candidate evidence into irrelevant paragraphs.
# Or the wrong content was hit. No search terms are allowed to appear again.
POLLUTION_BLACKLIST = [
    "cognitive diagnostic", "cdm", "cdm results", "q-matrix", "cognitive skills", "cognitive patterns",
    "automatic scoring", "automated scoring", "automated scoring systems", "automated systems",
    "fine-grained", "coarse-grained", "personalized", "personalized feedback",
    "machine-human agreement", "cohen", "kappa", "srmsr", "heatmap", "probabilities",
    "aligned with prior", "diagnostic", "diagnostic information", "removed item", "test without items",
    "rubric", "participants", "real-time feedback", "tutorials", "follow-up tutorials",
    "instructional support", "private school", "public middle schools", "middle school",
    "same total ability score", "applied cdm", "timely feedback", "automatic guidance",
    "human experts encountered", "not unexpected", "not all students require", "assessment",
]

# A paragraph of more than 6 consecutive Chinese characters: the "keywords" regularly extracted from the Chinese rule text by the old implementation have this shape.
LONG_CHINESE_RUN = re.compile(r"[\u4e00-\u9fff]{6,}")

# Sample paper terms that are not allowed to reappear in the scoring prompt (batch_response_instruction).
# is listed separately from the search term blacklist: prompt directly guides the model, and the criteria are allowed to be stricter (including field words such as "educational technology/teacher diagnosis").
PROMPT_DOMAIN_BLACKLIST = [
    "Q-matrix", "SRMSR", "heatmap", "CDM", "Cohen's kappa",
    'Learning Advanced', 'Fine-grained diagnosis', 'Student mode', 'Teacher Diagnosis', 'Learning Intervention', 'Scoring Rubric', 'Educational Technology', 'Abnormal questions',
]
# The original text is extracted directly from the rule definition / indicator_name. When it appears, it means that the Chinese literal words have been extracted again.
CHINESE_RULE_TEXT_SAMPLES = [
    'Importance of research question',
    'Determine whether the paper explains',
    'Determine whether the paper',
    'needs to be located',
    'Research objects and research questions are clear',
    'observation point',
]


def blacklist_hit(text: str) -> str | None:
    'Whole word matching blacklist; returns hit words, returns None if not.'
    lowered = str(text).lower()
    for term in POLLUTION_BLACKLIST:
        if re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", lowered):
            return term
    return None


def evidence(eid: str, page: int, section: str, text: str) -> Evidence:
    return Evidence(
        evidence_id=eid,
        paper_id="PAPER-TEST",
        section=section,
        page=page,
        span=(0, len(text)),
        text=text,
        text_hash=f"h-{eid}",
    )


# Created English evidence fragment. section with "Main Body": evidence_location and agent.section_hints of any rule
# cannot match, so whether it can be included in the reading package can only be determined by the exclusive recall path, and the test can be differentiated.
FIXTURES = {
    "SAMPLE": evidence(
        "EV-SAMPLE", 1, "Main Body",
        "We collected survey data from 412 respondents in 87 firms. The data collection followed a "
        "two-stage sampling procedure, the response rate was 61 percent, and the final sample size "
        "is adequate for the estimation we report below.",
    ),
    "IDENT": evidence(
        "EV-IDENT", 2, "Main Body",
        "Identification rests on an instrumental variable strategy. We address endogeneity with "
        "two-stage least squares and a full set of control variables; the estimates survive checks "
        "for omitted variable bias and selection concerns.",
    ),
    "DESIGN": evidence(
        "EV-DESIGN", 3, "Main Body",
        "Our research design is a multi-wave longitudinal field study. The design matches the causal "
        "question because the time lag between waves lets us observe change; random assignment was "
        "not feasible, so the panel structure carries the inference.",
    ),
    "MEASURE": evidence(
        "EV-MEASURE", 4, "Main Body",
        "Every construct was measured with validated scales adapted from prior work. Each measurement "
        "scale used five items, factor analysis supports construct validity, and the reliability of "
        "every scale exceeds the conventional threshold.",
    ),
    "EXEC": evidence(
        "EV-EXEC", 5, "Main Body",
        "The procedure we followed is reported step by step. We collected the data in 2019, then ran "
        "the main regression and a structural equation model. The coding process and every data "
        "analysis decision are documented in the appendix.",
    ),
    "QUAL": evidence(
        "EV-QUAL", 6, "Main Body",
        "We adopt a qualitative case study design. We conducted 34 interviews and analyzed the "
        "transcripts with grounded theory coding; the thematic analysis followed an inductive logic "
        "and we report triangulation across informants and field notes.",
    ),
    "THEORY": evidence(
        "EV-THEORY", 7, "Main Body",
        "We define algorithmic discretion as the latitude employees have in interpreting system "
        "output. This conceptual definition is distinct from related constructs such as autonomy. "
        "We argue that the underlying assumption of prior work is taken for granted; therefore we "
        "derive hypotheses about its boundary conditions. The theoretical logic builds on agency "
        "theory, and we assume managers respond to perceived control. We also consider an "
        "alternative explanation: reverse causality and omitted variable bias could produce the "
        "same pattern, so we cannot rule out confounding.",
    ),
}

# A padding section that has nothing to do with the above five method fragments: it is used to verify that the recall is sorted by content, not by making up the numbers.
FILLER_TEXTS = [
    "The author thanks the editor and anonymous referees for their helpful comments on earlier versions of this manuscript and for their guidance.",
    "An earlier version of this paper was presented at the annual meeting of the academy and benefited from the discussion that followed.",
    "The remaining sections of this manuscript proceed as follows and introduce each part of the argument in turn for the reader.",
    "Support from the research fund of the university is gratefully acknowledged by the authors of this study and their colleagues.",
    "Any remaining errors and all interpretations offered in this article are those of the authors alone and not of their institutions.",
    "We are indebted to the two reviewers whose suggestions substantially improved the clarity of the exposition in this article.",
    "The order of authorship is alphabetical and both authors contributed equally to every part of the work reported here.",
    "A previous draft circulated widely among colleagues, and the questions we received shaped the framing adopted in this version.",
    "The views expressed in this article do not necessarily reflect those of any organization with which the authors are affiliated.",
    "We thank the participants of the doctoral consortium for their generous engagement with an earlier draft of this research note.",
]

# The method fragment for each D rule that should be recalled.
EXPECTED_D_FIXTURE = {
    "D-DESIGN-FIT-V1": "DESIGN",
    "D-DATA-SAMPLE-V1": "SAMPLE",
    "D-MEASUREMENT-VALIDITY-V1": "MEASURE",
    "D-MODEL-IDENTIFICATION-V1": "IDENT",
    "D-METHOD-EXECUTION-V1": "EXEC",
    "D-QUAL-GENRE-FIT-V1": "QUAL",
}


def method_fixture_pool() -> list[Evidence]:
    pool = list(FIXTURES.values())
    for index, text in enumerate(FILLER_TEXTS):
        pool.append(evidence(f"EV-FILL-{index}", 20 + index, "Main Body", text))
    return pool


def main() -> int:
    profile = load_rule_profile(RULES)
    assert len(profile.rules) == 34, len(profile.rules)

    # 1) All 34 rules must have non-empty retrieval_terms and not fall into the pollution blacklist
    for rule in profile.rules:
        terms = list(rule.retrieval_terms)
        assert terms, f"{rule.rule_id} does not have retrieval_terms (English search terms)"
        assert 6 <= len(terms) <= 12, f"{rule.rule_id} retrieval_terms {len(terms)} in total, should be between 6-12\nThe search term for"
        for term in terms:
            assert term.isascii(), f"{rule.rule_id} is not in English: {term}\nThe search term"
            assert ": " not in term, f"{rule.rule_id} contains ASCII colon and space, which will destroy YAML: {term}"
            hit = blacklist_hit(term)
            assert hit is None, f"{rule.rule_id}'s search term {term!r} hits the pollution blacklist {hit!r}"
    print(f"Search word list: all 34 rules are declared retrieval_terms ({min(len(r.retrieval_terms) for r in profile.rules)}"
          f"-{max(len(r.retrieval_terms) for r in profile.rules)}), none hit the sample paper pollution blacklist")

    # 2) Dimension vocabularies and rule type vocabularies are not allowed to contain polluting terms.
    for dimension_id, hints in DIMENSION_KEYWORD_HINTS.items():
        # does not lock the magic number of "15-45": the length of the vocabulary itself is not a property, and there is neither harm nor benefit in increasing the E dimension from 45 to 48.
        # What you really need to guard against are the following two items (if they break there will be functional consequences).
        assert hints, f"DIMENSION_KEYWORD_HINTS[{dimension_id}] is empty"
        assert any(hint.isascii() for hint in hints), f"{dimension_id} dimension does not have any English keywords"
        assert len(hints) <= 64, f"{dimension_id} Dimensional vocabulary {len(hints)}, suspected of uncontrolled expansion"
        for hint in hints:
            hit = blacklist_hit(hint)
            assert hit is None, f"DIMENSION_KEYWORD_HINTS[{dimension_id}] hits {hint!r} in the pollution blacklist {hit!r}"
    for marker, hints in RULE_KEYWORD_HINTS.items():
        for hint in hints:
            hit = blacklist_hit(hint)
            assert hit is None, f"RULE_KEYWORD_HINTS[{marker}] hits {hint!r} in tainted blacklist {hit!r}"
    print('Vocabulary cleaning: DIMENSION_KEYWORD_HINTS / RULE_KEYWORD_HINTS There are no exclusive terms for the sample paper.')

    # 2b) Bag-of-word truncation must not eat the search terms that come with the rule.
    # rule_keyword_hints has [:48] truncation at the end, and the etymological order is retrieval_terms → dimension general words → rule type words,
    # Therefore, dimension vocabulary expansion will only truncate the tails of common words, but will not truncate rule.retrieval_terms.
    # This property is much more important than "a few words in the vocabulary": once the etymological order is reversed, the rule-specific search terms will be silently discarded.
    for rule in profile.rules:
        hints = rule_keyword_hints(rule)
        missing = [term for term in rule.retrieval_terms if term.lower() not in hints]
        assert not missing, f"{rule.rule_id} was truncated and eaten by the word bag: {missing}"
    print('Bag-of-word truncation: all retrieval_terms of 34 rules survive in the return value of rule_keyword_hints')

    # 3) rule_keyword_hints no longer extracts literal words from Chinese rule text
# The search word bag for
    for rule in profile.rules:
        hints = rule_keyword_hints(rule)
        assert hints, f"{rule.rule_id} is empty\nA long Chinese phrase appears in the search word bag for"
        joined = " ".join(hints)
        long_run = LONG_CHINESE_RUN.search(joined)
        assert long_run is None, f"{rule.rule_id}: {long_run.group(0)!r}\nThe search word bag of"
        for sample in CHINESE_RULE_TEXT_SAMPLES:
            assert sample not in joined, f"{rule.rule_id} is mixed with Chinese rules. Original text: {sample!r}"
        for hint in hints:
            hit = blacklist_hit(hint)
            assert hit is None, f"{rule.rule_id}'s search term {hint!r} hits the pollution blacklist {hit!r}"
        # The English search terms that come with the rules must appear in the word bag (the most specific one cannot be truncated)
# The search term
        for term in rule.retrieval_terms:
            assert term.lower() in hints, f"{rule.rule_id} did not enter rule_keyword_hints"
    # Chinese common words must still be retained (useful for the Chinese text) and cannot be cut into pure English.
    assert any(re.search(r"[\u4e00-\u9fff]", hint) for hint in rule_keyword_hints(profile.rules[0]))
    print('Word bag reconstruction: only composed of English search terms + general vocabulary, literal words are no longer extracted from Chinese rule text, and Chinese general words are retained')

    # 4) C/D exclusive recall function: only effective for its own dimensions, and can recall the relevant fragments of the structure
    pool = method_fixture_pool()
    for rule in profile.rules:
        if rule.dimension_id == "C":
            assert score_method_rigor_hits(rule, pool) == [], f"{rule.rule_id} Should not go D Recall"
        elif rule.dimension_id == "D":
            assert score_theory_rigor_hits(rule, pool) == [], f"{rule.rule_id} Should not go C Recall\nD-exclusive recall of"

    for rule in profile.rules:
        if rule.dimension_id != "D":
            continue
        expected = EXPECTED_D_FIXTURE[rule.rule_id]
        hits = score_method_rigor_hits(rule, pool)
        hit_ids = [item.evidence_id for item in hits]
        assert FIXTURES[expected].evidence_id in hit_ids, f"{rule.rule_id} missed {expected}"
        # Relevant fragments must be ranked before the filler paragraph: recall is based on content, not on the order of details.
        assert hit_ids.index(FIXTURES[expected].evidence_id) < len(FILLER_TEXTS), (
            f"{rule.rule_id} come after the padding segment: {hit_ids[:6]}\nThe reading package of"
        )
        pack = build_reading_pack(rule, pool, AGENT_FOR_DIMENSION["D"], 5, None)
        pack_ids = [item.evidence_id for item in pack]
        assert FIXTURES[expected].evidence_id in pack_ids, (
            f"{rule.rule_id} was not selected into {expected}: {pack_ids}"
        )
        # D path is ranked before the bottom: the whole package of 5 must be method fragments, and not a single filler paragraph should be mixed in.
        filler_ids = {f"EV-FILL-{index}" for index in range(len(FILLER_TEXTS))}
        assert not (set(pack_ids) & filler_ids), (
            f"{rule.rule_id}’s reading bag relies on the padding section at the bottom of the pocket to make up the number: {pack_ids}"
        )
    print(f"D-dimension exclusive recall: the sample/data collection and identification/endogeneity fragments of the 6 rules are all included in the package and ranked at the top")

    # 5) C-dimension exclusive recall: theoretical rigor fragments are included in the package
    for rule in profile.rules:
        if rule.dimension_id != "C":
            continue
        hits = score_theory_rigor_hits(rule, pool)
        assert hits and hits[0].evidence_id == FIXTURES["THEORY"].evidence_id, (
            f"{rule.rule_id}’s C-exclusive recall first place is not theoretical rigor Fragment: {[i.evidence_id for i in hits[:3]]}\nThe reading package of"
        )
        pack = build_reading_pack(rule, pool, AGENT_FOR_DIMENSION["C"], 5, None)
        pack_ids = [item.evidence_id for item in pack]
        assert FIXTURES["THEORY"].evidence_id in pack_ids, f"{rule.rule_id} did not select the theoretical fragment: {pack_ids}"
    print('C-dimension exclusive recall: all the theoretical rigor fragments of the 7 rules are included in the package')

    # 6) Clips that are too short (<90 characters) are not included in C/D scoring
    short = evidence("EV-SHORT", 1, "Main Body", "We define the construct and report the sample.")
    assert len(short.text) < 90
    for rule in profile.rules:
        if rule.dimension_id in {"C", "D"}:
            hits = score_theory_rigor_hits(rule, [short]) + score_method_rigor_hits(rule, [short])
            assert hits == [], f"{rule.rule_id} also regarded short clips as candidates."
    print('Minimum length threshold: Fragments <90 characters do not participate in C/D exclusive recall')

    # 7) Limit: build_reading_pack must not be over-issued in any dimension or rule.
    big_pool = []
    for index in range(40):
        section = ["Introduction", "Literature Review", "Theory", "Methods", "Data", "Sample",
                   "Measures", "Results", "Discussion", "Conclusion", "Appendix"][index % 11]
        text = (
            f"{section} paragraph {index}. We collected survey data from respondents, measured every "
            "construct with validated scales, estimated a regression model with control variables, and "
            "report robustness checks. The theoretical contribution is developed in the discussion, "
            "together with limitations, practical implications and directions for future research."
        )
        big_pool.append(evidence(f"EV-BIG-{index}", 1 + index // 3, section, text))
    big_pool.append(evidence("EV-REF-1", 30, "References", "References " + "Smith, J. 2020. A study of organizations. " * 6))
    big_pool.append(evidence("EV-REF-2", 31, "References", "References " + "Lee, K. 2021. Another study of organizations. " * 6))

    for rule in profile.rules:
        pack = build_reading_pack(rule, big_pool, AGENT_FOR_DIMENSION[rule.dimension_id], 10, None)
        assert len(pack) <= 10, f"{rule.rule_id} Super hair: {len(pack)}"
        assert len({item.evidence_id for item in pack}) == len(pack), f"{rule.rule_id} Duplicate evidence appears"
        assert all(item.section != "References" for item in pack), f"{rule.rule_id} regards the reference paragraph as evidence"
    print('Reading package limit: 34 rules are not exceeded under the limit of 10, and there are no duplications and no reference paragraphs')

    # 8) The domain calibration text in the scoring prompt is also not allowed to contain sample paper terms.
    # This is more serious than the keyword table: it is not that the evidence cannot be retrieved, but it directly tells the model what is considered a contribution and what is considered robustness evidence.
    prompt_text = "\n".join(batch_response_instruction(agent) for agent in AGENTS)
    for banned in PROMPT_DOMAIN_BLACKLIST:
        assert banned not in prompt_text, f"There are still sample paper terms in the scoring prompt: {banned!r}"
    # The design intention must be retained: the paper cannot be judged not_satisfied just because it does not have fixed words or is not traditional regression robustness.
    for intent in [
        'does not require fixed keywords to appear',
        'E-ROBUSTNESS At least partial should be considered',
        'F-REPRODUCIBILITY is available for partial',
        'F-LIMITATIONS AND F-EXTERNAL-VALIDITY',
        'It should not be judged not_satisfied just because there is no fixed management theory wording.',
        'should fall into management practice',
    ]:
        assert intent in prompt_text, f"scoring prompt lost the original calibration intention: {intent!r}"
    # Each of the six dimensions A-F must have its own calibration text (the three blocks B/E/F must actually be injected into the prompt)
    for marker in ['B Dimension Score Calibration', 'E Dimension Score Calibration', 'F Dimension Score Calibration']:
        assert marker in prompt_text, marker
    print('Scoring prompt Decontamination: B/E/F The three blocks of calibration text have been replaced by management caliber, and the original calibration intention is retained.')

    print('Evidence retrieval self-test passed: English search terms are in place, Chinese rule text is no longer used as search terms, and C/D dimensions each have their own exclusive recall.')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
