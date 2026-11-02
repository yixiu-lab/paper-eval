# Data and Research Materials

This section collects the released data and supporting materials for inspecting PaperEval's experiments.

## Contents

- [Corpus audit](corpus-audit.md): corpus reconciliation, processing outcomes, and aggregate score summaries.
- [Paper-level outcomes](paper-outcomes.csv): title-free records with available total and dimension scores.
- [Study sample index](study-samples.md): paper and run counts by study; samples remain separate.
- [Released data notes](data-index.md): definitions and scope of the outcome table.
- [Sample-paper index](sample-paper-index.csv): source identifiers, DOI values where available, and release status for the corpus PDF inventory.
- [Redistributable sample PDFs](sample-papers/README.md): unchanged copies with an explicit sharing license noted in the source PDF.
- [Experiment catalog](../03-experiments/experiment-catalog.md): study designs, samples, and results across the project.
- [Experiment evidence](../03-experiments/experiment-evidence.md): supporting records for additional study families.

The released outcome table omits paper titles, journal names, local paths, and source text. Study samples overlap and their counts should not be added into one project-wide total. See the [ethics and data-access statement](../05-ethics-access/README.md) for data handling information.

## Run records

For each run, record its date, source-data access class, code revision, software dependencies, model/provider and endpoint, scoring criteria, prompt, retrieval settings, evidence budget, cache state, random seed where relevant, output location, failures, and responsible operator. Keep credentials outside the repository.

## Validation modes

| Mode | What it checks | Network behavior |
| --- | --- | --- |
| Local logic | Rule validation, scoring, evidence gates, parsing components, report rendering, and reliability logic | Uses local inputs or fixtures; no external service calls |
| Mocked service | Request formatting and retrieval behavior with simulated responses | Intercepts calls locally |
| Live integration | End-to-end evaluation or optional retrieval probes | Uses configured services and may send paper-derived text or queries |

Check a script's mode before running it. Live runs use the configured model or retrieval services and can write reports, caches, and traces to the local ignored `runs/` folder.
