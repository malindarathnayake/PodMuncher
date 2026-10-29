#!/usr/bin/env python3
"""Local security scan: SBOM (Syft) and CVE scan (Grype) for the locally built worker images.

Run after ``bin/release_candidate.py build`` and before tagging a release. Writes
``security/sbom/<image>-<platform>.spdx.json``, ``security/vuln/<image>-<platform>.grype.json``
and a summary in ``security/README.md``; commit those files with the release.

    python bin/security_scan.py --platform linux-amd64

Syft and Grype run as pinned containers against the local Docker engine (socket mount), so
nothing needs installing. Exit status is 1 when any image has a critical CVE that has a fix.
Standard library only.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "security"
SYFT = "anchore/syft:v1.54.0"
GRYPE = "anchore/grype:v0.104.0"
SOCKET = ["-v", "/var/run/docker.sock:/var/run/docker.sock"]
TAGS = {"media": "podmuncher-media:release-local", "asr_cpu": "podmuncher-asr:cpu-release-local",
        "asr_cuda": "podmuncher-asr:cuda-release-local"}
PLATFORM_CAPABILITIES = {"linux-amd64": ("media", "asr_cpu", "asr_cuda"), "linux-arm64": ("media", "asr_cpu")}
SEVERITIES = ("Critical", "High", "Medium", "Low", "Negligible", "Unknown")


def docker(*args: str, stdin: bytes | None = None) -> bytes:
    result = subprocess.run(["docker", *args], input=stdin, capture_output=True, check=False)
    if result.returncode != 0:
        sys.exit(f"error: docker {args[0]} {' '.join(args[-2:])} failed:\n"
                 + result.stderr.decode("utf-8", "replace")[-2000:])
    return result.stdout


def image_exists(tag: str) -> bool:
    return subprocess.run(["docker", "image", "inspect", tag], capture_output=True, check=False).returncode == 0


def scan(capability: str, platform: str) -> dict:
    tag = TAGS[capability]
    sbom = docker("run", "--rm", *SOCKET, SYFT, f"docker:{tag}", "-o", "spdx-json", "-q")
    report = docker("run", "--rm", "-i", *SOCKET, GRYPE, "-o", "json", "-q", stdin=sbom)
    name = f"{capability}-{platform}"
    (OUT / "sbom" / f"{name}.spdx.json").write_bytes(sbom)
    (OUT / "vuln" / f"{name}.grype.json").write_bytes(report)
    matches = json.loads(report)["matches"]
    counts = Counter(m["vulnerability"]["severity"] for m in matches)
    fixable = sorted({(m["vulnerability"]["severity"], m["artifact"]["name"], m["artifact"]["version"],
                       m["vulnerability"]["id"], ", ".join(m["vulnerability"]["fix"]["versions"]))
                      for m in matches if m["vulnerability"]["fix"]["state"] == "fixed"
                      and m["vulnerability"]["severity"] in ("Critical", "High")})
    return {"name": name, "tag": tag, "packages": len(json.loads(sbom)["packages"]),
            "counts": counts, "fixable": fixable}


def write_summary(results: list[dict]) -> None:
    lines = ["# Security scan", "",
             f"Generated {datetime.now(timezone.utc):%Y-%m-%d} by `bin/security_scan.py` "
             f"({SYFT}, {GRYPE}) against the local release-candidate images.", "",
             "| Image | Packages | " + " | ".join(SEVERITIES) + " |",
             "|---|---|" + "---|" * len(SEVERITIES)]
    for r in results:
        lines.append(f"| {r['name']} | {r['packages']} | "
                     + " | ".join(str(r["counts"].get(s, 0)) for s in SEVERITIES) + " |")
    lines += ["", "## Critical and high CVEs with a fix available", ""]
    fixable = [(r["name"], *f) for r in results for f in r["fixable"]]
    if fixable:
        lines += ["| Image | Severity | Package | Version | CVE | Fixed in |", "|---|---|---|---|---|---|"]
        lines += [f"| {' | '.join(row)} |" for row in fixable]
    else:
        lines.append("None. The remaining findings have no fixed version upstream (most are Debian "
                     "\"won't fix\").")
    lines += ["", "Full SBOMs (SPDX JSON) are in `sbom/`; full Grype reports are in `vuln/`.", ""]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--platform", default="linux-amd64", choices=sorted(PLATFORM_CAPABILITIES))
    args = parser.parse_args(argv)
    (OUT / "sbom").mkdir(parents=True, exist_ok=True)
    (OUT / "vuln").mkdir(parents=True, exist_ok=True)
    results = []
    for capability in PLATFORM_CAPABILITIES[args.platform]:
        if not image_exists(TAGS[capability]):
            print(f"skip {capability}: {TAGS[capability]} is not built locally")
            continue
        result = scan(capability, args.platform)
        print(f"{result['name']}: " + ", ".join(f"{s} {result['counts'].get(s, 0)}" for s in SEVERITIES[:3]))
        results.append(result)
    write_summary(results)
    critical = [r["name"] for r in results if any(f[0] == "Critical" for f in r["fixable"])]
    if critical:
        print("fixable critical CVEs in: " + ", ".join(critical))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
