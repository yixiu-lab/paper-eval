# PaperEval

Peer review is where AI’s “more versus better” problem is most acute. Manuscript volumes are rising faster than reviewer capacity, and early evidence from *Organization Science* links higher submission volume after the arrival of generative AI with declining writing quality ([Gartenberg et al., 2026](https://pubsonline.informs.org/doi/10.1287/orsc.2026.ed.v37.n3)). Large language models (LLMs) can extract structured information from publications and generate feedback that overlaps substantially with human reviews ([Liang et al., 2024](https://doi.org/10.1056/AIoa2400196)); separate work has examined ChatGPT’s ability to predict peer-review outcomes ([Thelwall & Yaghi, 2024](https://arxiv.org/abs/2411.09763)). Yet the naive recipe—give a model a paper and ask for a score—can produce generic judgments, uncited evidence, author claims treated as facts, and outputs that are difficult to reproduce or audit. Scaling that recipe would scale these failures.

PaperEval provides a verifiable review infrastructure. It translates management scholarship’s quality standards into explicit, versioned, machine-checkable scoring rules; anchors judgments to citable evidence from the paper and, for comparison-dependent criteria, prior literature; and passes judgments through a deterministic validation layer that can reject or downgrade unsupported outputs and route cases for human review. The output is an audit trail: for each rule, a judgment tier, rationale, evidence IDs, and review status that others can inspect. Judgments become falsifiable claims rather than impressions.

## Technical approach

PaperEval turns a paper PDF into rule-level judgments and a score report. The method follows the sequence below; each heading links to the matching section in the full [Technical Approach](01-system/README.md).

1. [System overview](01-system/README.md#1-system-overview). The command-line Harness coordinates PDF processing, evidence routing, reviewer agents, validation, score aggregation, and report generation.
2. [Rule profile](01-system/README.md#2-rule-profile). A 34-rule profile across six dimensions specifies criteria, evidence requirements, judgment anchors, applicability, weights, and score mappings.
3. [PDF parsing and paper representation](01-system/README.md#3-pdf-parsing-and-paper-representation). The parser extracts paper structure and creates page- and section-linked evidence records and a structured paper profile.
4. [Rule-specific evidence packages](01-system/README.md#4-rule-specific-evidence-packages). The evidence router selects bounded passages and context for each rule and includes stable evidence identifiers for traceability.
5. [Claim-led external grounding](01-system/README.md#5-claim-led-external-grounding). When enabled, claims from the paper guide prior-work retrieval; selected records are linked to their source and supplied as external evidence for comparison-dependent rules.
6. [Specialist review and judgment protocol](01-system/README.md#6-specialist-review-and-judgment-protocol). Six specialist Reviewers assess assigned dimensions and return structured rule judgments, rationales, confidence, and evidence references; a unified-Reviewer configuration is also available.
7. [Harness: validation, gating, and aggregation](01-system/README.md#7-harness-validation-gating-and-aggregation). Deterministic checks validate schemas and evidence references, flag selected cases for review, retry uncertain items, and calculate dimension and total scores.

## Experimental results

The results are organized into three groups: overall paper scores, journal-level score aggregation, and robustness analyses. The [experiment README](02-experiments/README.md) links each result to its detailed record; the [experiment catalog](02-experiments/experiment-catalog.md) indexes the broader study program.

### 3.1 Paper total score results

The evaluation contains completed score records for **200 papers**. Archived rule-level judgments were aggregated using the integer weights; the model was not rerun for this recalculation. The mean total score is **77** (SD **9**), the median is **76**, and the range is **50–96**. [Evaluation results](03-data/README.md#evaluation-results) and the [outcome table](03-data/paper-outcomes.csv) provide dimension summaries and paper-level scores.

### 3.2 Journal score aggregation results

The 200 evaluated papers cover **73 journals and publication sources**. The complete journal score table is sorted in descending order by mean score in [2. Journal score aggregation results](02-experiments/README.md#2-journal-score-aggregation-results).

### 3.3 Robustness results

Detailed records: [Cross-model results](02-experiments/cross-model-results.md), [PaperEval Module Ablation Records](02-experiments/papereval-module-ablation-records.md), [Deterministic harness fault injection](02-experiments/robustness-and-harness-experiments.md#deterministic-harness-fault-injection), [Evidence-level prompt-injection comparison](02-experiments/robustness-and-harness-experiments.md#evidence-level-prompt-injection-comparison), and [Two-run repeatability in the current experiment manuscript](02-experiments/robustness-and-harness-experiments.md#two-run-repeatability-in-the-current-experiment-manuscript). Focused comparisons reuse papers from the corpus; their repeated records are not added to the 200-paper count. The [PaperEval Experiment Catalog](02-experiments/experiment-catalog.md) links study designs and additional records.

## Data and materials

The data section contains the outcome table, the paper index covering all 200 evaluated papers, and sample PDFs with explicit redistribution licenses. The [data overview](03-data/README.md) describes the released materials and score summaries. Reusable experimental code is indexed in the [experiment README](02-experiments/README.md#experiment-code).

## Run the command-line workflow

Clone the repository and enter the system folder:

```text
git clone https://github.com/yixiu-lab/paper-eval.git
cd paper-eval/01-system
```

Create a virtual environment, install dependencies, set `OPENAI_API_KEY`, validate the rules, then run the evaluation. Replace the PDF path and key with your own. Do not commit API keys.

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:OPENAI_API_KEY = "YOUR_API_KEY"
python -m paper_eval.cli validate-rules --rules rules/rules_v1.yaml
python -m paper_eval.cli evaluate --pdf "C:\path\to\paper.pdf" --rules rules/rules_v1.yaml --out "runs\paper-001" --agent-config configs/agent_runtime.example.yaml
```

**macOS or Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY="YOUR_API_KEY"
python -m paper_eval.cli validate-rules --rules rules/rules_v1.yaml
python -m paper_eval.cli evaluate --pdf "/path/to/paper.pdf" --rules rules/rules_v1.yaml --out "runs/paper-001" --agent-config configs/agent_runtime.example.yaml
```

For other compatible providers or available options, see the [runtime configuration](01-system/configs/agent_runtime.example.yaml) and [Technical Approach](01-system/README.md#9-execution-interface-and-artifact-boundary).

## Ethics

### Human responsibility

PaperEval produces structured evidence and rule-level assessments for research inspection. Human researchers remain responsible for interpreting and validating results. The system's scores are not presented as editorial decisions.

PaperEval is intended to support research and exploratory analysis. It is not intended to score manuscripts for peer review, replace reviewers, generate editorial recommendations, or inform accept/reject decisions. Its outputs should be treated as supplementary research materials, not as peer reviews.

### PDF availability

The repository provides completed scores and available bibliographic metadata for **all 200 evaluated papers**. It includes **51 unchanged PDFs** whose source copies explicitly state a Creative Commons license. The paper index records each released PDF's source hash and repository path. Full-text PDFs without confirmed redistribution permission are not included. Readers should follow the notice in each released PDF and the [Creative Commons license terms](https://creativecommons.org/share-your-work/cclicenses/).
