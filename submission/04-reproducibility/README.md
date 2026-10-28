# Reproducibility Materials — Incomplete

The call expects work to be reproducible. It notes that non-shareable data is unlikely to fit unless an appropriate synthetic substitute can reproduce the insights, and that proprietary AI infrastructure should be reasonably accessible to other researchers.

## Included

- Source code and draft rule profile.
- Python dependency list (`requirements.txt`).
- Example configuration files.
- Offline self-check scripts.

## Missing before a reproducibility claim

- A legally shareable benchmark corpus or a documented synthetic substitute.
- Dataset/source manifest with license, access route, version, and checksums.
- Exact tested Python/runtime and dependency lock with versions.
- A complete configuration with secrets removed, model/provider/version, and access alternatives.
- Immutable code revision and run manifest for each reported result.
- Seeds, prompts, rule-profile fingerprint, retrieval parameters, cache status, logs, and failure cases.
- Expected outputs and a verified end-to-end reproduction guide.
- Independent rerun by a researcher other than the prototype author.

Do not place API keys, private manuscripts, or restricted data in this folder.
