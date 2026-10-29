#!/usr/bin/env python3
"""Refresh the container part of release.json's provenance inventory from the built images.

Run after ``bin/release_candidate.py build`` (both platforms) and ``apply``:

    python bin/release_inventory.py --remote-arm64 podmac

For every local release-candidate image it lists the installed Debian packages (with the sha256 of
each package's copyright file) and the Python distributions (with their declared licence), then
rewrites ``provenance_inventory.container_images`` and ``container_base_images`` in release.json.
linux-arm64 images are read over SSH on the host that built them. Runtime manifests are not
touched, so no hash changes. Standard library only.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "src/podmuncher/manifests/release.json"
TAGS = {"media": "podmuncher-media:release-local", "asr_cpu": "podmuncher-asr:cpu-release-local",
        "asr_cuda": "podmuncher-asr:cuda-release-local"}
PLATFORM_CAPABILITIES = {"linux-amd64": ("media", "asr_cpu", "asr_cuda"), "linux-arm64": ("media", "asr_cpu")}
REMOTE_PREFIX = "DOCKER_CONFIG=$HOME/podmuncher-rc/docker-config PATH=/usr/local/bin:/opt/homebrew/bin:$PATH "

# Runs inside each image (python -I, network none, read-only).
PROBE = r'''
import hashlib, json, os, re
from importlib import metadata
debian = []
for block in open("/var/lib/dpkg/status", encoding="utf-8").read().split("\n\n"):
    fields = dict(re.findall(r"^([A-Za-z-]+): (.*)$", block, re.M))
    if fields.get("Status") != "install ok installed":
        continue
    path = f"/usr/share/doc/{fields['Package']}/copyright"
    digest = hashlib.sha256(open(path, "rb").read()).hexdigest() if os.path.isfile(path) else None
    debian.append({"copyright_sha256": digest, "name": fields["Package"], "version": fields["Version"]})
python = {}
for dist in metadata.distributions():
    meta = dist.metadata
    name = re.sub(r"[-_.]+", "-", meta["Name"]).lower()
    expression, declared = meta.get("License-Expression"), (meta.get("License") or "").strip()
    classifiers = [c.split(" :: ")[-1] for c in meta.get_all("Classifier") or [] if c.startswith("License ::")]
    if expression:
        license, source = expression, "License-Expression"
    elif declared and len(declared) <= 100 and "\n" not in declared:
        license, source = declared, "License"
    elif classifiers:
        license, source = "; ".join(classifiers), "Classifier"
    else:
        license, source = "UNKNOWN", "none"
    python[name] = {"name": name, "version": meta["Version"], "license": license, "license_source": source}
print(json.dumps({"debian_packages": sorted(debian, key=lambda p: p["name"]),
                  "python_packages": sorted(python.values(), key=lambda p: p["name"])}))
'''


def inspect(tag: str, host: str | None) -> dict:
    argv = ["docker", "run", "--rm", "-i", "--pull", "never", "--network", "none", "--read-only",
            "--entrypoint", "python", tag, "-I", "-"]
    if host:
        argv = ["ssh", host, REMOTE_PREFIX + " ".join(argv)]
    result = subprocess.run(argv, input=PROBE.encode(), capture_output=True, check=False)
    if result.returncode != 0:
        sys.exit(f"error: inspecting {tag}{' on ' + host if host else ''} failed:\n"
                 + result.stderr.decode("utf-8", "replace")[-2000:])
    return json.loads(result.stdout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--remote-arm64", metavar="HOST", required=True, help="SSH host holding the arm64 images")
    parser.add_argument("--manifest", default=str(MANIFEST))
    args = parser.parse_args(argv)
    path = Path(args.manifest)
    document = json.loads(path.read_text("utf-8"))
    images: dict[str, dict] = {}
    for platform, capabilities in PLATFORM_CAPABILITIES.items():
        host = args.remote_arm64 if platform == "linux-arm64" else None
        for capability in capabilities:
            images.setdefault(capability, {})[platform] = inspect(TAGS[capability], host)
            found = images[capability][platform]
            print(f"{capability} {platform}: {len(found['debian_packages'])} debian, "
                  f"{len(found['python_packages'])} python")
    inventory = document["provenance_inventory"]
    inventory["container_images"] = images
    bases = sorted({identity["base_image"] for identity in document["local_build_identities"].values()})
    if len(bases) != 1 or len(inventory["container_base_images"]) != 1:
        sys.exit("error: expected one shared base image")
    inventory["container_base_images"][0]["image"] = bases[0]
    path.write_bytes((json.dumps(document, indent=1, ensure_ascii=True) + "\n").encode("utf-8"))
    print(f"updated {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
