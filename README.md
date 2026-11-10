# PaperEval

## Research background

Assessment of management research draws on human judgment about the importance of a research question, theoretical contribution, rigor, research design, evidence, and implications. These judgments are made from different parts of a paper and need to be connected to the evidence that supports them. PaperEval studies how to make this assessment process explicit and inspectable: criteria guide the review, paper passages support rule-level judgments, and deterministic software calculates the reported scores.

The project uses one calibrated scoring-rule set with 34 rules across six dimensions. The rules translate paper-quality criteria into specific checks, evidence requirements, judgment anchors, and score mappings. PaperEval applies those rules through an AI-assisted workflow and preserves the evidence and reasoning behind each reported judgment.

## Research questions

1. **Human assessment:** How can the multidimensional judgments involved in human paper review be represented as explicit, evidence-linked criteria?
2. **AI scoring:** How can an AI workflow apply these criteria to paper evidence and produce traceable rule-level judgments and scores?
3. **Journal and source comparison:** How are the resulting score profiles distributed across journals and publication sources in the evaluated corpus?

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

The evaluation contains completed score records for **200 papers**. Archived rule-level judgments were aggregated using the integer weights; the model was not rerun for this recalculation. The mean total score is **77** (SD **9**), the median is **76**, and the range is **50–96**. The [data overview](03-data/README.md#evaluation-results) and [outcome table](03-data/paper-outcomes.csv) provide dimension summaries and paper-level scores.

### 3.2 Journal score aggregation results

The paper index records **73 named journal or source groups** for 194 papers; six papers have no source field and are shown separately in Table 5. The table summarizes every group by descending mean score, with sample size, mean and standard deviation, median, and range. The corpus includes **29 papers across eight journals** on the 2026 Financial Times 50 list, including 15 in *Management Science* and four in *Academy of Management Journal*. See the [journal aggregation section](02-experiments/README.md#2-journal-score-aggregation-results) and the [paper index](03-data/paper-index.csv).

### 3.3 Robustness results

Robustness analyses examine model and reviewer configuration, Harness checks, evidence-boundary instructions, and repeated runs. Five models were compared on three shared papers; their means range from **77.91 to 89.99**, while mean absolute run differences range from **0.61 to 3.83** points. Four reviewer configurations were compared on five papers, with mean scores of **79.50**, **74.60**, **57.93**, and **74.00**. The Harness detected **180/180** injected machine-checkable faults across nine classes, and all **40** clean controls passed. In a paired prompt study, evidence-boundary instructions reduced target-reaching attacks from **4/108** to **0/108**. Across 20 papers scored twice, the paired mean absolute score difference was **2.382** points and **17/20** retained the same quality grade.

Detailed records: [five-model comparison](02-experiments/cross-model-results.md), [reviewer structure comparison](02-experiments/module-ablation.md), [Harness fault-injection results](02-experiments/robustness-experiments.md#deterministic-harness-fault-injection), [evidence-boundary prompt-injection comparison](02-experiments/robustness-experiments.md#evidence-level-prompt-injection-comparison), and [repeated-run stability](02-experiments/robustness-experiments.md#two-run-repeatability-in-the-current-experiment-manuscript). Focused comparisons reuse papers from the corpus; their repeated records are not added to the 200-paper count. The [experiment catalog](02-experiments/experiment-catalog.md) links study designs and additional records.

## Data and materials

The data section contains the outcome table, the paper index covering all 200 evaluated papers, and sample PDFs with explicit redistribution licenses. The [data overview](03-data/README.md) describes the released materials and score summaries. Reusable experimental code is indexed in the [experiment README](02-experiments/README.md#experiment-code).

## Run the command-line workflow

### 1. Get the project and open its system folder

Clone the repository, then move into the system folder:

```bash
git clone https://github.com/yixiu-lab/paper-eval.git
cd paper-eval/01-system
```

### 2. Install the requirements

Create a Python environment, activate it, and install the packages.

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

**macOS or Linux:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Check the command and scoring rules

Display the available commands and validate the included rule profile:

```bash
python -m paper_eval.cli --help
python -m paper_eval.cli validate-rules --rules rules/rules_v1.yaml
```

### 4. Configure the model and API key

The example runtime uses an OpenAI-compatible endpoint and reads its key from the `OPENAI_API_KEY` environment variable. Set the key in your current terminal session, replacing the placeholder with your own key. Keep the key out of the repository.

**Windows PowerShell:**

```powershell
$env:OPENAI_API_KEY = "your-key"
```

**macOS or Linux:**

```bash
export OPENAI_API_KEY="your-key"
```

To use another compatible service, make a local copy of the [example runtime configuration](01-system/configs/agent_runtime.example.yaml), set its provider, endpoint, model name, and key environment-variable name, then pass that file to `--agent-config`. See [Technical Approach](01-system/README.md#9-execution-interface-and-artifact-boundary) for the full command reference.

### 5. Evaluate one paper

Replace the PDF path and output folder with your own paths. The command writes the evaluation reports and run records to the output folder.

**Windows PowerShell:**

```powershell
python -m paper_eval.cli evaluate --pdf "C:\path\to\paper.pdf" --rules rules/rules_v1.yaml --out "runs\paper-001" --agent-config configs/agent_runtime.example.yaml
```

**macOS or Linux:**

```bash
python -m paper_eval.cli evaluate --pdf "/path/to/paper.pdf" --rules rules/rules_v1.yaml --out "runs/paper-001" --agent-config configs/agent_runtime.example.yaml
```

Record the provider and model, code revision, rule profile, retrieval settings, and run status with each evaluation. For the full command reference and output description, see [Technical Approach](01-system/README.md#9-execution-interface-and-artifact-boundary).
