# Security policy

## Supported status

DI-NIDS is a research artefact and is not supported as a production intrusion-prevention system.
Security fixes will be considered for the latest repository version.

## Reporting a vulnerability

Use GitHub's private vulnerability-reporting mechanism if it is enabled for the repository. If it
is not available, contact the repository owner privately rather than opening a public issue that
contains exploit details.

Include the affected version, reproduction steps, likely impact, and any suggested mitigation.

## Trust boundaries

- Do not load untrusted pickle datasets. Pickle loading is disabled unless explicitly enabled.
- Do not load model checkpoints from untrusted sources. The bundled loader uses
  `weights_only=True`, but model files should still be verified against known checksums.
- Treat network-flow datasets as potentially sensitive, even when identifiers are removed.
- Do not use research results as the sole basis for blocking traffic or making incident-response
  decisions.

