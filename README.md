# PaperEval: Evidence-Grounded Evaluation of Management Research

PaperEval is an early-stage research prototype for evaluating management and organizational research papers. It combines rule-guided evidence retrieval, structured reviewer judgments, deterministic score aggregation, and traceable reports.

The project is being prepared as a research artifact for the *Organization Science* call for science, “AI-Enabled Frontiers in Organizational Science.” The intended contribution is an inspectable, human-directed evaluation workflow that can be studied for validity, reliability, and failure boundaries. These are research goals, not established findings. No claim of expert-level accuracy or validated measurement is made.

## Current status

- Prototype code and draft scoring rules are included.
- Offline self-check scripts are included, but their results do not establish end-to-end validity or generalizability.
- The source PDFs and an expert-judgment benchmark are not included. The full-corpus V16 audit covers 149 source PDFs (146 unique), with 144 complete reports and two parse-blocked papers; a title-free per-paper results table is linked in [`submission/04-reproducibility/`](submission/04-reproducibility/).
- The short proposal and complete raw reproduction package are still outstanding.
- The executable prototype, scoring rule labels, and some comments remain in Chinese. The English submission materials are in [`submission/`](submission/). Translation of the runtime and scoring source is still outstanding.
- Do not upload confidential manuscripts, API credentials, private run outputs, or copyrighted full texts.

## Repository map

- `literature_eval/`: evaluation workflow, evidence handling, retrieval, scoring, and reports.
- `评价规则库/`: active scoring profile and its design notes (source materials are currently Chinese).
- `configs/`: example runtime and retrieval configurations.
- `prompts/`: reviewer-agent prompts.
- `scripts/`: offline validation and evaluation utilities.
- `submission/`: call-aligned proposal, technical appendix, artifact, reproducibility, and ethics materials.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\自检_评分标尺.py
python scripts\自检_外部检索.py
```

Model-backed runs require credentials supplied through environment variables or a local configuration that is not committed. Never commit API keys, `.env` files, private manuscripts, or real run data.

## Call requirements and deadline

Stage 1 is due **November 1, 2026**. The call requests (1) a 3–5 page / 1,500–2,000 word short proposal plus references, (2) a technical appendix or supplement, and (3) a prototype and Git repository or equivalent artifact. See [`submission/README.md`](submission/README.md) for the checklist and gaps.

## License and data

No project license or redistributable evaluation dataset is currently provided. Resolve licensing and data-sharing terms before public release or reproduction claims.
