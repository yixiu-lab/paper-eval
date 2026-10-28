# Reproducibility Materials — Partial

The call expects work to be reproducible. It notes that non-shareable data is unlikely to fit unless an appropriate synthetic substitute can reproduce the insights, and that proprietary AI infrastructure should be reasonably accessible to other researchers.

## Included in this repository

- Source code and draft rule profile.
- Python dependency list (`requirements.txt`).
- Example configuration files.
- Offline self-check scripts.
- An English aggregate summary of the historical experiment evidence: [`experiment-evidence.md`](experiment-evidence.md).

## Existing records not yet published here

The authors' local workspace contains a curated evidence package assembled on October 8, 2026. Its index reports 17 papers, 7,486 evidence records, 72 evaluation runs, and 2,342 agent judgments in its trace database. It also catalogs seven experiment groups and provides run reports, summaries, manifests, scripts, and analysis notes. These are real historical experimental records and should be distinguished from the materials currently published here.

The raw bundle is not copied into this repository because run traces may contain paper excerpts and paper-level data whose sharing rights have not been established. Before sharing raw records, review copyright, confidentiality, and data licenses. The current GitHub repository also lacks a checked, complete mapping between the historical run bundle, its source-code revisions, and the uploaded repository commit.

## Remaining gaps before an independent reproduction claim

- A legally shareable benchmark corpus or a documented synthetic substitute.
- Dataset/source manifest with license, access route, version, and checksums.
- A verified mapping from each reported historical run to the exact code revision and rule profile used.
- Exact tested Python/runtime and dependency lock with versions.
- A complete configuration with secrets removed, model/provider/version, and access alternatives.
- A publication-ready run manifest for each reported result, with paper identities redacted where needed.
- Seeds, prompts, rule-profile fingerprint, retrieval parameters, cache status, logs, and failure cases.
- Expected outputs and a verified end-to-end reproduction guide.
- Independent rerun by a researcher other than the prototype author.

Do not place API keys, private manuscripts, or restricted data in this folder.
