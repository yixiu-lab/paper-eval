# Ethics and Data Handling

## Study materials

The corpus audit used published AI/HRM research papers. Source PDFs, extracted full text, and paper-linked evaluation traces are held in the research workspace and are not included in this repository. The public repository contains code, study descriptions, aggregate results, and title-free outcomes.

The audited corpus did not include expert-rater annotations. This repository does not report an institutional ethics determination for a human-participant study.

## Data flows

PaperEval can send selected paper evidence to a configured language-model service. Literature retrieval can send paper titles and claim-derived search terms to Crossref, OpenAlex, and Semantic Scholar. Generated reports, traces, and caches may contain paper-derived evidence or rationales.

Local logic checks use local inputs or fixtures. Mocked-service checks use simulated responses. Live integration scripts contact configured model or retrieval services and may transmit paper-derived content; their data flow depends on the selected source and provider configuration.

## Human responsibility

PaperEval produces structured evidence and rule-level assessments for research inspection. Human researchers remain responsible for interpreting and validating results. The system's scores are not presented as editorial decisions.
