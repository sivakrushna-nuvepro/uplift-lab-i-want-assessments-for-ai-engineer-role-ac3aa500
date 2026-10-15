# Private validation report publication

The validator must always write its report to `reports/report.json` first.
S3 publication is disabled for this package; retain the local report for the Factory evidence collector.

The final object key is `{tenantId}/{projectId}/{attemptId}/{reportFilename}` under the configured base URI.
Read tenant, project, and attempt identity from the environment variables declared in `report-publishing-contract.json`.
Never store credentials, session tokens, presigned URLs, or public-read ACLs in this package or report.
A publication failure must not erase the local report; return it as unverified publication evidence for SME review.
