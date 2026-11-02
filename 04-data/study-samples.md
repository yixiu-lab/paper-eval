# Paper Counts by Study

Counts below are study-specific. Samples overlap; do not sum them to claim a project-wide total of unique papers. “Scored papers” counts papers with completed paper-level evaluation; test cases and model calls are reported separately.

| Study | Papers evaluated | Completed scoring / runs | Scope note |
| --- | ---: | --- | --- |
| Corpus audit | 149 source PDFs; 146 unique after exact deduplication | 144 complete scores; two unique papers stopped at PDF parsing | Three source PDFs were exact duplicates. See the [audit outcomes](paper-outcomes.csv). |
| API batch | 10 papers | Batch outcomes are summarized in the [experiment evidence](../03-experiments/experiment-evidence.md). | A separate operational batch. |
| Journal-gradient studies | 12 papers in the first sample; 15 additional papers in the second | Reported separately by sample | The combined sample is 27 papers, independent of the corpus audit. |
| Repeated-run stability | 20 papers | Two complete reports per paper (40 reports) | Paired repeatability, not a unique-paper expansion. |
| Archived reliability diagnostic | 20 papers | 57 saved run records | Reuses a focused subset; runs are not papers. |
| Trace database | 17 papers | 72 evaluation runs | Focused trace subset; overlaps other studies. |
| Five-model comparison | 3 shared papers | Two recorded runs per model and paper across five models | Descriptive comparison; evidence retrieval varied between runs. |
| Reviewer-structure comparison | 5 papers | Four evaluation conditions per paper | Scores from different workflow conditions. |
| Module ablation | 1 paper | Baseline plus nine module-disabled conditions | Single-paper diagnostic. |
| Evidence-gate comparison | 3 papers | Paired processing with and without the gate | Reuses stored judgments and evidence. |
| Prompt-injection study | 3 papers | 108 paired attack cases per prompt condition | Attack cases are not complete paper scores. |
| Harness fault injection | No paper-level sample | 180 injected faulty outputs and 40 clean controls | Software validation cases, not paper evaluations. |
| Retrieval study | 7 target papers | 7 claims and 366 candidate mappings | Relevance labels are machine-prelabelled; aggregate metric reconciliation is pending. |

The corpus audit is the only study in this index that reports source-file, deduplicated-paper, and completed-report counts together. Other samples and run records are separate and may overlap.
