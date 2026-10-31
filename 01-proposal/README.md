# PaperEval

### An evidence-grounded workflow for studying AI-assisted research assessment

PaperEval examines how AI can assess management research through explicit criteria, traceable evidence, and inspectable judgments. Its central research question is when evidence-grounded AI assessments align with expert judgment, how stable they are across runs and evidence settings, and where their limitations require human review.

PaperEval operationalizes expert-constructed evaluation criteria as rule-level judgments. Each judgment is connected to paper evidence, checked by deterministic software, and aggregated into a report that researchers can inspect.

## System workflow

1. **Parse and represent the paper.** PaperEval extracts text and structure from a research-paper PDF, assigns page- and section-linked evidence identifiers, and builds a structured profile of the paper's question, claims, theory, methods, findings, and limitations. Severe parsing failures are recorded before scoring. See [PDF parsing and paper representation](../02-system/technical-appendix.md#3-pdf-parsing-and-paper-representation) and [`pdf_parser.py`](../02-system/paper_eval/pdf_parser.py).

   **解析并表示论文。** 系统从研究论文 PDF 中提取文本和结构，为证据标注页码、章节及唯一编号，并构建包含研究问题、主张、理论、方法、发现和局限的结构化论文档案。严重解析失败会在评分前记录。详见[论文解析与表示](../02-system/technical-appendix.md#3-pdf-parsing-and-paper-representation)及 [`pdf_parser.py`](../02-system/paper_eval/pdf_parser.py)。

2. **Extract claims and retrieve prior work.** When enabled, claim-led retrieval uses the paper's research question, gap, theory, and contribution claims to find related work from configured scholarly metadata services. Retrieved records are normalized, linked to their sources, and used as external evidence for comparison-dependent judgments. See [claim-led external grounding](../02-system/technical-appendix.md#5-claim-led-external-grounding) and the [retrieval study](../03-experiments/retrieval-evaluation.md).

   **提取主张并检索相关研究。** 启用外部检索时，系统根据论文的研究问题、研究缺口、理论和贡献主张，从配置的学术元数据服务中查找相关研究。检索结果经过规范化并保留来源信息，作为依赖文献比较的判断所需的外部证据。详见[主张驱动的外部证据](../02-system/technical-appendix.md#5-claim-led-external-grounding)及[检索实验](../03-experiments/retrieval-evaluation.md)。

3. **Route evidence to six evaluation dimensions.** The system assembles rule-specific evidence for research-question positioning, theoretical contribution, theoretical rigor, research design, results and evidence, and discussion and boundaries. Evidence packages retain the relevant passage, page, section, and evidence identifier. See [rule-specific evidence packages](../02-system/technical-appendix.md#4-rule-specific-evidence-packages) and the [expert-constructed criteria](../02-system/rules/rules_v1.yaml).

   **将证据分配到六个评价维度。** 系统围绕研究问题定位、理论贡献、理论严谨性、研究设计、结果与证据、讨论与边界条件，分别组装规则级证据包。每个证据包保留相关段落、页码、章节和证据编号。详见[规则级证据包](../02-system/technical-appendix.md#4-rule-specific-evidence-packages)及[专家构建的评价规则](../02-system/rules/rules_v1.yaml)。

4. **Generate structured reviewer judgments.** Specialist reviewers assess assigned rules and return a judgment, rationale, confidence, and cited evidence identifiers. The runtime also supports a unified-reviewer configuration. See [specialist review and judgment protocol](../02-system/technical-appendix.md#6-specialist-review-and-judgment-protocol) and the [reviewer guide](../02-system/reviewer-guide.md).

   **生成结构化评审判断。** 不同评审模块分别处理所分配的规则，并返回判断、理由、置信度和引用的证据编号。运行配置也支持统一评审模块。详见[专业评审与判断流程](../02-system/technical-appendix.md#6-specialist-review-and-judgment-protocol)及[评审者指南](../02-system/reviewer-guide.md)。

5. **Validate and aggregate.** The Harness checks judgment schemas and evidence references, validates external evidence where applicable, retries selected uncertain items, and flags issues for human inspection. Deterministic scoring then aggregates applicable rule judgments and records completeness and uncertainty. See [Harness validation and aggregation](../02-system/technical-appendix.md#7-harness-validation-gating-and-aggregation), [`evidence_gate.py`](../02-system/paper_eval/evidence_gate.py), and [`scorer.py`](../02-system/paper_eval/scorer.py).

   **验证并汇总。** Harness 检查判断格式和证据引用，在适用时核验外部证据，对部分不确定项进行补充检索，并标记需要人工检查的问题。随后，确定性评分程序汇总适用规则的判断，同时记录覆盖度和不确定性。详见[Harness 验证与汇总](../02-system/technical-appendix.md#7-harness-validation-gating-and-aggregation)、[`evidence_gate.py`](../02-system/paper_eval/evidence_gate.py)及[`scorer.py`](../02-system/paper_eval/scorer.py)。

6. **Produce inspectable outputs.** Each run can produce JSON, Markdown, and HTML reports, a run manifest, and a trace bundle. Reports connect scores to criteria, evidence, rationales, validation results, and review flags; optional reliability diagnostics summarize variation across saved runs. See [reports and run records](../02-system/technical-appendix.md#8-reports-run-records-and-reliability-analysis).

   **生成可检查的结果。** 每次运行可生成 JSON、Markdown 和 HTML 报告、运行清单及追踪记录。报告将分数与规则、证据、理由、验证结果和复核标记关联起来；可选的可靠性诊断用于汇总已保存运行之间的差异。详见[报告、运行记录与可靠性分析](../02-system/technical-appendix.md#8-reports-run-records-and-reliability-analysis)。

The main research artifact is the [Python implementation and command-line workflow](../02-system/README.md). The local Web/API prototype is outside this repository's research-artifact scope.

本研究成果主要包括 [Python 实现和命令行工作流](../02-system/README.md)。本地 Web/API 原型不属于本仓库发布的研究成果。

## Experimental evidence

The current experiment manuscript reports results in the sequence below. Each experiment has its own sample and comparison; samples overlap and should not be added to infer a project-wide number of unique papers. The [experiment index](../03-experiments/README.md) follows the same sequence and links to each detailed record.

当前实验稿按以下顺序报告结果。各实验使用各自的样本和比较条件，样本可能重叠，因此不能将样本数相加作为项目去重后的论文总数。[实验索引](../03-experiments/README.md)按相同顺序排列，并链接到各项详细记录。

### 1. Corpus and journal/source score profile / 语料与期刊/来源评分概况

The audit began with **149 source PDFs**, which yielded **146 unique papers** after exact deduplication. It produced **144 complete reports**; two papers were blocked before scoring by PDF parsing. The completed reports had a mean total score of **76.46** (SD 7.64), a median of **76.19**, and a range of **49.63–95.78**. These are descriptive scores, not expert-validity estimates. The manuscript also reports A–F dimension summaries and journal/source composition. See the [audit record](../04-reproducibility/original-149-v16-audit.md) and [title-free outcomes](../04-reproducibility/original-149-v16-results.csv).

语料审计从 **149 份源 PDF** 开始，精确去重后得到 **146 篇独立论文**，其中 **144 篇**生成完整报告，另有两篇在评分前因 PDF 解析失败而受阻。完整报告的总分均值为 **76.46**（标准差 7.64），中位数为 **76.19**，范围为 **49.63–95.78**。这些是描述性评分，不代表专家效度。论文还报告了 A–F 维度摘要和期刊/来源构成。详见[审计记录](../04-reproducibility/original-149-v16-audit.md)和[不含标题的结果表](../04-reproducibility/original-149-v16-results.csv)。

### 2. Exploratory model comparison / 探索性模型比较

The manuscript compares five models on three shared papers, with two recorded runs per model and paper. Model means range from **77.91 to 89.99**, and average absolute differences between repeated runs range from **0.61 to 3.83** points. Evidence and retrieval were regenerated between runs, so the observed differences combine model and evidence variation. See the [five-model summary](../03-experiments/cross-model-v16.md).

论文在三篇共同样本上比较五个模型，每个模型和论文记录两轮运行。模型均分范围为 **77.91–89.99**，轮间平均绝对分差为 **0.61–3.83** 分。由于各轮重新提取证据并进行检索，观察到的差异同时包含模型和证据变化。详见[五模型汇总](../03-experiments/cross-model-v16.md)。

### 3. Reviewer structure comparison / Reviewer 结构比较

Four evaluation structures were compared on the same **five papers**: full PaperEval, one unified Reviewer, six rule-level Reviewers without Harness processing and formal two-pool aggregation, and direct model scoring. Their mean scores were **79.50**, **74.60**, **57.93**, and **74.00**, respectively. These outputs are descriptive because the conditions use different judgment and aggregation procedures and there are no independent expert scores. See the [four-condition comparison and module ablations](../03-experiments/module-ablation.md).

四种评价结构在同一组 **5 篇论文**上比较：完整 PaperEval、单一 Reviewer、跳过 Harness 处理和正式两池汇总的六个规则级 Reviewer，以及直接模型评分。四种条件的均分依次为 **79.50**、**74.60**、**57.93** 和 **74.00**。由于判断与汇总方式不同，且没有独立专家评分，这些结果用于描述条件差异。详见[四条件比较和模块消融](../03-experiments/module-ablation.md)。

### 4. Harness validation / Harness 校验

The deterministic fault-injection study covered **180 injected faults** across nine machine-checkable categories and **40 clean controls**. The Harness detected all injected faults, automatically blocked or transformed 160, routed 20 reason–judgment conflicts for review, and passed all clean controls. See [fault-injection results](../03-experiments/robustness-experiments.md#deterministic-harness-fault-injection).

确定性故障注入研究覆盖九类机器可检查错误，共 **180 个注入错误**和 **40 个正常对照**。Harness 检出全部注入错误，自动阻断或转换 160 个，将 20 个理由—判断冲突标记为复核，并使全部正常对照通过。详见[故障注入结果](../03-experiments/robustness-experiments.md#deterministic-harness-fault-injection)。

### 5. Adversarial instruction injection / 对抗性指令注入

The paired prompt study used evidence from **three papers** and 108 attack cases per condition. Adding an evidence-boundary instruction reduced target-reaching attacks from **4/108 (3.7%) to 0/108**, rating changes from 14/108 to 7/108, and mean absolute ordinal shift from **0.250 to 0.074**. See the [prompt-injection comparison](../03-experiments/robustness-experiments.md#evidence-level-prompt-injection-comparison).

配对提示词研究使用 **3 篇论文**的证据，每种条件包含 108 个攻击样例。加入证据边界指令后，命中攻击目标的案例从 **4/108（3.7%）降至 0/108**，判断变化从 14/108 降至 7/108，平均绝对序数偏移从 **0.250** 降至 **0.074**。详见[提示注入比较](../03-experiments/robustness-experiments.md#evidence-level-prompt-injection-comparison)。

### 6. Repeated-run stability / 重复运行稳定性

Twenty papers were scored independently twice, producing **40 complete reports**. The paired mean absolute score difference was **2.382** points, the maximum difference was **8.388**, and **17/20 papers (85.0%)** retained the same quality grade. See [two-run repeatability results](../03-experiments/robustness-experiments.md#two-run-repeatability-in-the-current-experiment-manuscript).

20 篇论文分别独立评分两次，共形成 **40 份完整报告**。配对平均绝对分差为 **2.382** 分，最大分差为 **8.388** 分，**17/20 篇（85.0%）**论文两轮等级一致。详见[双轮重复性结果](../03-experiments/robustness-experiments.md#two-run-repeatability-in-the-current-experiment-manuscript)。

### Additional archived studies / 其他存档研究

Additional records include a **10-paper** API batch, two journal-gradient samples with **27 papers** combined, a **17-paper** trace database, a separate reliability diagnostic with 57 historical runs across 20 papers, and a seven-paper/seven-claim retrieval setup with 366 citation-linked candidate mappings. Retrieval candidates were machine-prelabelled from citations, and the archived aggregate has a claim-count mismatch; no overall retrieval-performance estimate is reported. See the [historical experiment index](../03-experiments/experiment-evidence.md), [experiment catalog](../03-experiments/experiment-catalog.md), and [retrieval record](../03-experiments/retrieval-evaluation.md).

其他存档记录包括 **10 篇论文**的 API 批次、合计 **27 篇论文**的两个期刊梯度样本、**17 篇论文**的追踪数据库、包含 20 篇论文 57 次历史运行的独立可靠性诊断，以及包含 7 篇目标论文、7 项主张和 366 条引用候选映射的检索设置。检索候选依据引用关系由机器预标注，存档汇总存在主张数量不一致，因此此处不报告整体检索性能估计。详见[历史实验索引](../03-experiments/experiment-evidence.md)、[实验目录](../03-experiments/experiment-catalog.md)和[检索记录](../03-experiments/retrieval-evaluation.md)。

## Reviewer navigation / 审稿人阅读入口

- [System architecture and technical method](../02-system/technical-appendix.md) — system layers, evidence flow, validation, and scoring.
- [Experiment catalog](../03-experiments/experiment-catalog.md) — study families, sample sizes, and summary outcomes.
- [Reproducibility materials](../04-reproducibility/README.md) — source code, audit records, outcome table, and validation modes.
- [Ethics and data handling](../05-ethics-access/README.md) — study materials, data flows, and human responsibility.

- [系统架构与技术方法](../02-system/technical-appendix.md)：系统模块、证据流、验证和评分方式。
- [实验目录](../03-experiments/experiment-catalog.md)：研究类别、样本规模和主要结果。
- [复现材料](../04-reproducibility/README.md)：源代码、审计记录、结果表和验证方式。
- [伦理与数据处理](../05-ethics-access/README.md)：研究材料、数据流和人工责任。
