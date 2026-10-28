# Ethics, Data, and Access Plan

## Project commitment

PaperEval will be developed as a transparent research instrument. The team will document the provenance and permitted use of study materials, protect manuscript content, make model and retrieval dependencies visible, and retain human responsibility for interpreting results and making editorial decisions.

This page is the operational plan for the next study and submission. The principal investigators should complete the author-specific decisions below before recruiting raters, processing confidential submissions, or sharing study-level records.

## Data handling pathway

| Material | Current workflow | Next action |
|---|---|---|
| Research-paper PDFs | The 149-PDF V16 audit was conducted on published AI/HRM papers. The PDFs are not part of this repository. | Create a source manifest with acquisition route, DOI, license/access basis, version, checksum, and permitted processing/sharing status. Mark each item as shareable, process-only, or excluded. |
| Extracted text and evidence passages | PaperEval builds evidence blocks from paper content and passes selected blocks to the configured language-model endpoint for rule-level judgments. | Before each run, use only material whose terms permit this processing route. Record the endpoint and data-handling setting. Use an approved local model or a shareable sample when external processing is not permitted. |
| Retrieval queries and metadata | The workflow can query Crossref, OpenAlex, and Semantic Scholar; request inputs include paper titles and claim-derived search terms. | Record providers, endpoints, request date, query fields, and applicable access conditions in the run manifest. |
| Reports, traces, and caches | Evaluation outputs can include cited evidence, rationale, scores, provider/model information, and cached retrieval/evidence selections. | Store in the project’s access-controlled research workspace. Produce a reviewed, minimized release bundle for each public or editor-facing release. |
| Credentials | Runtime configuration refers to credentials supplied through local environment variables. | Keep credentials in local secret storage; exclude them from Git history, logs, screenshots, and shared artifacts. |

## Human participants and expert assessment

The current 149-paper audit used published papers and contains no expert-rater annotations. The next validity study is planned around expert assessment. Before contacting or collecting ratings from experts, the research leads will:

1. Submit the protocol to the responsible institutional ethics office and follow its determination.
2. Prepare an information sheet and consent process covering study purpose, tasks, foreseeable risks, withdrawal, data use, and publication.
3. Define recruitment, compensation, conflicts of interest, rater training, and adjudication procedures.
4. Use coded rater IDs, restrict access to the identity key, and specify retention and deletion dates.
5. Publish only appropriately de-identified ratings and aggregate results under the approved protocol.

The research team will record the review body, determination, protocol identifier, and approval date in the submission materials when available.

## Access and sharing plan

- **Code and scoring profile:** maintain the current repository as the project’s reviewable implementation and preserve the V16 rule fingerprint with each reported run.
- **Paper corpus:** complete the source-level rights inventory before distributing PDFs, extracted text, or paper-linked traces. The inventory will identify permitted access, processing, retention, and redistribution for every source.
- **Reproduction materials:** prepare a version-linked run manifest and a legally shareable benchmark or synthetic substitute, with expected outputs and a clean rerun guide.
- **Editor access:** provide the repository and approved artifacts through the access route requested by the editors. If source materials have different access conditions, provide a clear access matrix and a shareable substitute for restricted items.
- **Model and retrieval access:** record provider, model/version, endpoint, access date, cost, rate limits, and data-handling terms. Compare the primary configuration with a reasonably accessible alternative using a fixed prompt, rule profile, and evaluation sample.
- **Retention:** set a project retention schedule for source files, working text, traces, and backups; assign an owner for access reviews and deletion requests.

## Responsible use in the study

PaperEval produces structured evidence and rule-level assessments for researchers to inspect. Human researchers remain responsible for interpretation, validation, and any downstream editorial use. The planned expert study will examine agreement, score stability, evidence support, and performance across paper types before the team makes broader claims about the measurement capability.

## Author completion record

| Decision | Owner | Status / record |
|---|---|---|
| Source-level rights and access inventory | Data lead | To complete before the next corpus run |
| Institutional review determination for expert study | Study lead | To obtain before recruitment |
| Consent, compensation, and rater privacy protocol | Study lead | To finalize with the review submission |
| Model/retrieval terms and alternative access route | Technical lead | To record in each run manifest |
| Retention, deletion, and access-control schedule | Data steward | To approve before collecting expert ratings |
| Editor-facing artifact access matrix | Corresponding author | To include with the Stage 1 package |
