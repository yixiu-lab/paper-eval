# 管理学文献质量评价研究

这是一个用于研究管理学论文质量评价的原型系统，重点是证据约束、规则化评分、外部文献检索与可追溯报告。项目仍处于研究和验证阶段，不能替代人工同行评审。

## 内容

- `literature_eval/`：评价流程、证据处理、外部检索、评分与报告模块
- `评价规则库/`：当前规则与设计说明
- `configs/`：通用运行示例和检索评测配置
- `prompts/`：评价 Agent 提示词
- `scripts/`：离线自检工具
- `web/` 与 `app.py`：本地 Web/API 原型
- `EXTERNAL_RETRIEVAL_README.md`：外部文献检索模块说明

## 快速开始

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\自检_评分标尺.py`r`npython scripts\自检_外部检索.py
```

默认示例使用 OpenAI 兼容配置。真实模型调用需要在本机设置 `OPENAI_API_KEY`，或按需修改 `configs/agent_runtime.example.yaml`。不要把 `.env`、密钥或真实运行数据提交到仓库。

## 研究状态

当前规则仍在迭代，状态可能标记为 draft。评分输出用于研究实验和方法验证，不代表论文真实质量的确定性结论。实验数据和样本原文不包含在此仓库中。


