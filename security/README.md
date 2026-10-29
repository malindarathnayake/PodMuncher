# Security scan

Generated 2026-10-29 by `bin/security_scan.py` (anchore/syft:v1.54.0, anchore/grype:v0.104.0) against the local release-candidate images.

| Image | Packages | Critical | High | Medium | Low | Negligible | Unknown |
|---|---|---|---|---|---|---|---|
| media-linux-amd64 | 101 | 0 | 55 | 59 | 11 | 47 | 0 |
| asr_cpu-linux-amd64 | 122 | 0 | 56 | 63 | 11 | 47 | 0 |
| asr_cuda-linux-amd64 | 125 | 0 | 56 | 63 | 11 | 47 | 0 |

## Critical and high CVEs with a fix available

None. The remaining findings have no fixed version upstream (most are Debian "won't fix").

Full SBOMs (SPDX JSON) are in `sbom/`; full Grype reports are in `vuln/`.
