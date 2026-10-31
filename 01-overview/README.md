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

The project reports a set of studies with different samples and designs. Their samples overlap, so study counts should not be summed into a project-wide number of unique papers. The [experiment catalog](../03-experiments/experiment-catalog.md) maps the studies and provides links to each detailed record.

项目包含多项研究，各研究的样本和设计不同，且样本之间存在重叠，因此不能将各项样本数相加作为项目去重后的论文总数。[实验目录](../03-experiments/experiment-catalog.md)汇总各研究并链接到详细记录。

### Corpus audit / 语料审计

The audit started with **149 source PDFs**. Exact SHA-256 deduplication yielded **146 unique papers**. The workflow produced **144 complete reports**; two papers were blocked before scoring by PDF parsing. For the 144 completed reports, the mean total score was 76.46 (SD 7.64), the median was 76.19, and the range was 49.63–95.78. These are descriptive results for this corpus and configuration, not expert-validity estimates. See the [audit summary](../04-reproducibility/original-149-v16-audit.md) and [title-free per-paper outcomes](../04-reproducibility/original-149-v16-results.csv).

语料审计从 **149 份源 PDF** 开始。SHA-256 精确去重后得到 **146 篇独立论文**。系统生成 **144 份完整报告**；另有两篇论文在评分前因 PDF 解析失败而受阻。在已完成的 144 份报告中，总分均值为 76.46（标准差 7.64），中位数为 76.19，范围为 49.63–95.78。这些是针对该语料和配置的描述性结果，并非与专家判断一致性的效度结果。详见[审计摘要](../04-reproducibility/original-149-v16-audit.md)和[不含标题的逐篇结果](../04-reproducibility/original-149-v16-results.csv)。

### Reliability, model, and architecture studies / 可靠性、模型与架构研究

- A historical reliability analysis covers **57 runs across 20 papers** and reports score variation, evidence overlap, judgment agreement, and ranking stability. See [reliability and uncertainty results](../03-experiments/experiment-catalog.md#reliability-and-uncertainty).
- A model comparison includes **12 valid runs across three papers and two models**. The archived runs used evidence retrieved at different times, so the results describe model sensitivity but do not isolate a causal model effect. See the [cross-model comparison](../03-experiments/cross-model-v16.md).
- Architecture ablations include a **five-paper, four-condition** comparison, plus separate one-paper and three-paper analyses. These are descriptive comparisons; see [module ablations](../03-experiments/module-ablation.md).

- 一项历史可靠性分析覆盖 **20 篇论文的 57 次运行**，报告分数变化、证据重叠、判断一致性和排序稳定性。详见[可靠性与不确定性结果](../03-experiments/experiment-catalog.md#reliability-and-uncertainty)。
- 模型比较包含 **3 篇论文、2 个模型的 12 次有效运行**。存档运行使用了不同时间检索的证据，因此结果反映模型敏感性，但不能单独识别模型的因果影响。详见[跨模型比较](../03-experiments/cross-model-v16.md)。
- 架构消融包括一项 **5 篇论文、4 种条件**的比较，以及另外的单篇和三篇论文分析。这些结果为描述性比较，详见[模块消融](../03-experiments/module-ablation.md)。

### Robustness and retrieval studies / 稳健性与检索研究

- Prompt-injection tests used **three papers** and paired prompt conditions. The Harness fault-injection test covered **180 injected faulty outputs and 40 clean controls**. See [robustness and fault-injection results](../03-experiments/robustness-experiments.md).
- The retrieval study covers **seven target papers, seven claims, and 366 citation-linked candidate mappings**. Candidate relevance was machine-prelabelled from citation links, not independently judged by experts. See [retrieval evaluation](../03-experiments/retrieval-evaluation.md).
- Other archived scoring studies include a **10-paper API batch**, **27 papers** in two journal-gradient samples, and a **17-paper trace database**. Their outcomes and sample definitions are summarized in [historical experiment evidence](../03-experiments/experiment-evidence.md).

- 提示注入测试使用 **3 篇论文**并配对比较不同提示条件。Harness 故障注入测试覆盖 **180 个注入故障输出和 40 个正常对照**。详见[稳健性与故障注入结果](../03-experiments/robustness-experiments.md)。
- 检索研究涉及 **7 篇目标论文、7 项主张和 366 条引用关系候选映射**。候选相关性依据引用关系由机器预标注，并非由专家独立判断。详见[检索评估](../03-experiments/retrieval-evaluation.md)。
- 其他存档评分研究包括 **10 篇论文的 API 批次**、两个期刊梯度样本中的 **27 篇论文**，以及一个 **17 篇论文的追踪数据库**。样本口径和结果汇总见[历史实验记录](../03-experiments/experiment-evidence.md)。

## Reviewer navigation / 审稿人阅读入口

- [System architecture and technical method](../02-system/technical-appendix.md) — system layers, evidence flow, validation, and scoring.
- [Experiment catalog](../03-experiments/experiment-catalog.md) — study families, sample sizes, and summary outcomes.
- [Reproducibility materials](../04-reproducibility/README.md) — source code, audit records, outcome table, and validation modes.
- [Ethics and data handling](../05-ethics-access/README.md) — study materials, data flows, and human responsibility.

- [系统架构与技术方法](../02-system/technical-appendix.md)：系统模块、证据流、验证和评分方式。
- [实验目录](../03-experiments/experiment-catalog.md)：研究类别、样本规模和主要结果。
- [复现材料](../04-reproducibility/README.md)：源代码、审计记录、结果表和验证方式。
- [伦理与数据处理](../05-ethics-access/README.md)：研究材料、数据流和人工责任。
