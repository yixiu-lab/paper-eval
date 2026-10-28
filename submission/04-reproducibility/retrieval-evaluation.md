# External-Literature Retrieval Evaluation

## Experimental design in the archive

The retrieval study compares query, citation, and semantic routes, alone and in combination, with and without a deterministic reranker. It applies a publication-year filter and reports Recall@K, Precision@K, MRR, nDCG, Hit@K, and MAP. The archived setup uses OpenAlex reference links to discover candidate prior work and Crossref/OpenAlex/Semantic Scholar metadata services in the retrieval pipeline. The code path for this study uses deterministic retrieval and reranking rather than an LLM reranker.

The archived study describes seven target papers and seven claims, with 366 citation-linked candidate mappings. These records were machine-prelabelled from citation relationships. The next annotation pass will add expert relevance judgments to evaluate retrieval metrics against human assessment.

## Current evidence status

The local archive contains claim-level runs and multi-route ablation outputs. The generated report also contains a mismatch between the declared seven-claim sample and its claim-level aggregate counts. The team is reconciling the claim manifest, gold-set rows, and metric outputs before placing retrieval-performance numbers in the public results summary.

## Release path

The public repository includes the retrieval evaluation configuration and this design record. The next release will include a claim-count reconciliation, a versioned aggregate table, annotation provenance, and an independently reviewed relevance subset. Candidate titles, abstracts, citation traces, and provider caches will be included only after their sharing conditions are documented.
