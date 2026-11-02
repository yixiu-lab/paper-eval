# Reproducibility Materials

This folder indexes the records needed to inspect the corpus audit and its paper-level outcomes.

## Materials

- [Corpus audit](corpus-audit.md): corpus reconciliation, processing outcomes, and data scope.
- [Reproducibility data](data/): title-free paper outcomes and a per-study sample index.
- [Experiment catalog](../03-experiments/experiment-catalog.md): study designs, samples, and results across the project.
- [Experiment evidence](../03-experiments/experiment-evidence.md): archived evidence for historical studies.
- [Model comparison](../03-experiments/cross-model-results.md).
- [Robustness and uncertainty studies](../03-experiments/robustness-experiments.md).
- [Module ablations](../03-experiments/module-ablation.md).
- [Retrieval evaluation](../03-experiments/retrieval-evaluation.md).

## Data access

Source PDFs, extracted full text, and paper-linked run traces are retained in the research workspace and are not included here. The public outcomes omit paper titles, journal names, local paths, and source text. See the [ethics and data-access statement](../05-ethics-access/README.md) for handling details.

## Run records

For each run, record its date, source-data access class, code revision, software dependencies, model/provider and endpoint, rule profile, prompt, retrieval settings, evidence budget, cache state, random seed where relevant, output location, failures, and responsible operator. Keep credentials outside the repository.

## Validation modes

| Mode | What it checks | Network behavior |
| --- | --- | --- |
| Local logic | Rule validation, scoring, evidence gates, parsing components, report rendering, and reliability logic | Uses local inputs or fixtures; no external service calls |
| Mocked service | Request formatting and retrieval behavior with simulated responses | Intercepts calls locally |
| Live integration | End-to-end evaluation or optional retrieval probes | Uses configured services and may send paper-derived text or queries |

Check a script's mode before running it. Live runs use the configured model or retrieval services and can write reports, caches, and traces to the local ignored `runs/` folder.
