#!/usr/bin/env python3
"""Release-candidate identity tool: build local images, collect identities, apply them to release.json.

Subcommands: ``build`` (docker build + inspect on this machine), ``remote`` (ship a clean git archive to an
SSH host and run ``build`` there), ``apply`` (validate identity files and rewrite the release manifest).
Standard library only. Local image IDs are never labelled as registry digests. The optional ``--ca`` PEM is
passed only as the BuildKit secret ``build_ca`` and is never recorded or printed.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import tempfile
import tomllib
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST_REL = "src/podmuncher/manifests/release.json"

AMD64, ARM64 = "linux-amd64", "linux-arm64"
PLATFORM_CAPABILITIES = {AMD64: ("media", "asr_cpu", "asr_cuda"), ARM64: ("media", "asr_cpu")}
DOCKERFILES = {"media": ("containers/media.Dockerfile", None), "asr_cpu": ("containers/asr.Dockerfile", "cpu"),
               "asr_cuda": ("containers/asr.Dockerfile", "cuda")}
TAGS = {"media": "podmuncher-media:release-local", "asr_cpu": "podmuncher-asr:cpu-release-local",
        "asr_cuda": "podmuncher-asr:cuda-release-local"}
DOCKER_ARCH = {AMD64: "amd64", ARM64: "arm64"}
ENGINE_ARCH = {"amd64": "x86_64", "arm64": "aarch64"}
EXPECTED_USER = "65532:65532"
DEFAULT_KIND = "application/vnd.oci.image.index.v1+json"
COMPRESSED_METHOD = "byte count of docker save (OCI layout with compressed layer blobs)"
ARCHIVE_PATHS = ("containers/", ".dockerignore", "src/podmuncher", "requirements.lock",
                 "bin/release_candidate.py", MANIFEST_REL)
REMOTE_OUT = "rc-linux-arm64.json"
CHUNK = 4 * 1024 * 1024
PLATFORM_FIELDS = ("local_tag", "local_image_id", "local_image_kind", "compressed_bytes",
                   "compressed_bytes_method", "unpacked_bytes", "unpacked_bytes_method", "built_at", "built_on")
HASH_FIELDS = ("dockerfile_sha256", "build_inputs_sha256", "dependency_requirements_sha256")


class ToolError(Exception):
    """A step failed; the message names the step and never carries secrets."""


def run(argv, **kwargs):
    """The single injectable subprocess runner (tests monkeypatch this)."""
    return subprocess.run(list(argv), shell=False, check=False, **kwargs)


def popen(argv, **kwargs):
    """The single injectable streaming subprocess factory (tests monkeypatch this)."""
    return subprocess.Popen(list(argv), shell=False, **kwargs)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def dockerfile_requirements_sha(cap: str, repo: Path = REPO) -> str:
    text = (repo / DOCKERFILES[cap][0]).read_text("utf-8")
    blocks = dict(re.findall(r"COPY <<'REQUIREMENTS' (\S+)\n(.*?)REQUIREMENTS\n", text, re.DOTALL))
    body = {"media": blocks.get("/opt/media-requirements.txt", ""),
            "asr_cpu": blocks.get("/opt/asr-requirements.txt", ""),
            "asr_cuda": blocks.get("/opt/asr-requirements.txt", "")
            + blocks.get("/opt/asr-cuda-requirements.txt", "")}[cap]
    return sha256_hex(body.encode("utf-8"))


def build_inputs_sha(files: list[str], repo: Path = REPO) -> str:
    lines = (f"{name} {sha256_hex((repo / name).read_bytes())}" for name in files)
    return sha256_hex("\n".join(lines).encode("ascii"))


def compute_hashes(cap: str, build_inputs: list[str], repo: Path = REPO) -> dict[str, str]:
    return {"dockerfile_sha256": sha256_hex((repo / DOCKERFILES[cap][0]).read_bytes()),
            "build_inputs_sha256": build_inputs_sha(build_inputs, repo),
            "dependency_requirements_sha256": dockerfile_requirements_sha(cap, repo)}


def _load_manifest(path: Path) -> dict:
    return json.loads(path.read_text("utf-8"))


def _docker_json(step: str, *args: str):
    result = run(["docker", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise ToolError(f"{step} failed (exit {result.returncode})")
    return json.loads(result.stdout)


def save_size(tag: str) -> int:
    process = popen(["docker", "save", tag], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    total = 0
    try:
        while chunk := process.stdout.read(CHUNK):
            total += len(chunk)
        code = process.wait()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    if code != 0:
        raise ToolError(f"docker save failed for {tag}")
    return total


class _UnixConnection(http.client.HTTPConnection):
    def __init__(self, path: str):
        super().__init__("localhost")
        self._path = path

    def connect(self) -> None:
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self._path)


def engine_api_size(tag: str) -> int:
    context = _docker_json("docker context inspect", "context", "inspect")
    host = context[0]["Endpoints"]["docker"]["Host"]
    if not host.startswith("unix://"):
        raise ToolError("Engine API size needs a unix:// docker endpoint; got " + host.split("://", 1)[0] + "://")
    query = urllib.parse.quote(json.dumps({"reference": [tag]}))
    connection = _UnixConnection(host[len("unix://"):])
    try:
        connection.request("GET", "/images/json?filters=" + query)
        response = connection.getresponse()
        body = response.read()
        if response.status != 200:
            raise ToolError(f"Engine API /images/json returned {response.status}")
    finally:
        connection.close()
    return int(json.loads(body)[0]["Size"])


def _git_commit() -> str | None:
    result = run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(REPO))
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else None


def cmd_build(args) -> int:
    platform = args.platform
    arch = DOCKER_ARCH[platform]
    manifest = _load_manifest(REPO / MANIFEST_REL)
    capabilities = {}
    engine_version = None
    for cap in PLATFORM_CAPABILITIES[platform]:
        dockerfile, target = DOCKERFILES[cap]
        argv = ["docker", "build", "--platform", f"linux/{arch}", "-f", dockerfile]
        if target:
            argv += ["--target", target]
        argv += ["-t", TAGS[cap]]
        if args.ca:
            argv += ["--secret", f"id=build_ca,src={args.ca}"]
        argv.append(".")
        print(f"[build] {cap}: docker build", flush=True)
        if run(argv, cwd=str(REPO)).returncode != 0:
            raise ToolError(f"docker build step failed for {cap}")
        tag = TAGS[cap]
        inspected = _docker_json(f"docker image inspect {cap}", "image", "inspect", "--format", "{{json .}}", tag)
        compressed = save_size(tag)
        server = _docker_json("docker version", "version", "--format", "{{json .Server}}")
        version = server["Version"]
        size = int(inspected["Size"])
        if size >= compressed:
            unpacked = size
            method = f"docker image inspect Size (Docker Engine {version}, containerd snapshotter)"
        else:
            unpacked = engine_api_size(tag)
            method = (f"Engine API /images/json Size (Docker Engine {version}; "
                      "its image inspect Size reports content bytes only)")
        config = inspected.get("Config") or {}
        entry = {
            "local_tag": tag,
            "local_image_id": inspected["Id"],
            "local_image_kind": (inspected.get("Descriptor") or {}).get("mediaType") or DEFAULT_KIND,
            "compressed_bytes": compressed,
            "compressed_bytes_method": COMPRESSED_METHOD,
            "unpacked_bytes": unpacked,
            "unpacked_bytes_method": method,
            "built_at": inspected["Created"],
            "built_on": f"{args.built_on}, engine {ENGINE_ARCH.get(server.get('Arch'), server.get('Arch'))}",
            "architecture": inspected["Architecture"],
            "user": config.get("User"),
            "entrypoint": config.get("Entrypoint"),
        }
        entry.update(compute_hashes(cap, manifest["local_build_identities"][cap]["build_inputs"]))
        capabilities[cap] = entry
        engine_version = version
    document = {"platform": platform, "commit": _git_commit(), "engine_version": engine_version,
                "capabilities": capabilities}
    Path(args.out).write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"[build] wrote {args.out}")
    return 0


def cmd_remote(args) -> int:
    if args.platform != ARM64:
        raise ToolError("remote supports linux-arm64 only")
    status = run(["git", "status", "--porcelain", "--", *ARCHIVE_PATHS], capture_output=True, text=True,
                 cwd=str(REPO))
    if status.returncode != 0:
        raise ToolError("git status failed")
    if status.stdout.strip():
        raise ToolError("refusing: archive paths have uncommitted changes (archive must equal hashed bytes):\n"
                        + status.stdout.rstrip())
    commit = run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(REPO))
    if commit.returncode != 0 or not commit.stdout.strip():
        raise ToolError("git rev-parse HEAD failed")
    directory = f"podmuncher-rc/{commit.stdout.strip()}"
    quoted_dir = shlex.quote(directory)
    archive = popen(["git", "archive", "--format=tar", "HEAD", "--", *ARCHIVE_PATHS], stdout=subprocess.PIPE,
                    cwd=str(REPO))
    try:
        shipped = run(["ssh", args.host, f"mkdir -p {quoted_dir} && tar -x -C {quoted_dir}"], stdin=archive.stdout)
    finally:
        archive.stdout.close()
        archive_code = archive.wait()
    if archive_code != 0 or shipped.returncode != 0:
        raise ToolError("shipping git archive over ssh failed")
    build = (f"cd {quoted_dir} && PATH=/usr/local/bin:/opt/homebrew/bin:$PATH python3 bin/release_candidate.py "
             f"build --platform {shlex.quote(ARM64)} --built-on {shlex.quote(args.built_on)} "
             f"--out {shlex.quote(REMOTE_OUT)}")
    if run(["ssh", args.host, build]).returncode != 0:
        raise ToolError("remote build step failed")
    fetched = run(["ssh", args.host, f"cat {quoted_dir}/{REMOTE_OUT}"], capture_output=True)
    if fetched.returncode != 0:
        raise ToolError("fetching remote identity failed")
    Path(args.out).write_bytes(fetched.stdout)
    print(f"[remote] wrote {args.out}")
    return 0


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_identities(files: list[dict], manifest: dict, repo: Path) -> dict:
    """Return {(platform, cap): entry}; raise ToolError listing every problem."""
    problems: list[str] = []
    found: dict = {}
    for document in files:
        platform = document.get("platform")
        for cap, entry in (document.get("capabilities") or {}).items():
            key = (platform, cap)
            if key in found:
                problems.append(f"duplicate identity for {platform}/{cap}")
            found[key] = entry
    expected = [(p, c) for p, caps in PLATFORM_CAPABILITIES.items() for c in caps]
    for key in expected:
        if key not in found:
            problems.append(f"missing identity for {key[0]}/{key[1]}")
    for key in found:
        if key not in expected:
            problems.append(f"unexpected identity for {key[0]}/{key[1]}")
    identities = manifest["local_build_identities"]
    for key in expected:
        entry = found.get(key)
        if entry is None:
            continue
        platform, cap = key
        label = f"{platform}/{cap}"
        spec = identities[cap]
        for name, value in compute_hashes(cap, spec["build_inputs"], repo).items():
            if entry.get(name) != value:
                problems.append(f"{label}: {name} does not match the current checkout")
        if entry.get("user") != EXPECTED_USER:
            problems.append(f"{label}: user is not {EXPECTED_USER}")
        if entry.get("entrypoint") != spec["entrypoint"]:
            problems.append(f"{label}: entrypoint differs from the manifest")
        if entry.get("architecture") != DOCKER_ARCH[platform]:
            problems.append(f"{label}: architecture does not match the platform")
        compressed, unpacked = entry.get("compressed_bytes"), entry.get("unpacked_bytes")
        if not (isinstance(compressed, int) and isinstance(unpacked, int) and unpacked >= compressed > 0):
            problems.append(f"{label}: need unpacked_bytes >= compressed_bytes > 0")
    ids = [e.get("local_image_id") for k, e in found.items() if k in expected]
    if len(set(ids)) != len(ids):
        problems.append("local_image_id values are not distinct")
    if problems:
        raise ToolError("identity validation failed:\n  " + "\n  ".join(problems))
    return found


def rewrite_manifest(document: dict, found: dict, version: str, now: str) -> dict:
    identities = document["local_build_identities"]
    for (platform, cap), entry in found.items():
        record = identities[cap]
        for field in PLATFORM_FIELDS:
            record["platforms"][platform][field] = entry[field]
        for field in HASH_FIELDS:
            record[field] = entry[field]
    for record in identities.values():
        record["max_compressed_bytes"] = max(p["compressed_bytes"] for p in record["platforms"].values())
        record["max_unpacked_bytes"] = max(p["unpacked_bytes"] for p in record["platforms"].values())
    document["release_version"] = version
    for platform, runtime in document["runtime_manifests"].items():
        runtime["release_version"] = version
        for item in runtime["capabilities"]:
            record = identities[item["id"]]
            item["digest"] = record["platforms"][platform]["local_image_id"]
            item["compressed_bytes"] = record["max_compressed_bytes"]
            item["unpacked_peak_bytes"] = record["max_unpacked_bytes"]
    document["generated_at"] = now
    for platform, runtime in document["runtime_manifests"].items():
        canonical = json.dumps(runtime, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        document["runtime_manifest_sha256"][platform] = sha256_hex(canonical.encode("ascii"))
    return document


def cmd_apply(args) -> int:
    manifest_path = Path(args.manifest)
    files = [json.loads(Path(name).read_text("utf-8")) for name in args.files]
    document = _load_manifest(manifest_path)
    found = validate_identities(files, document, REPO)
    version = tomllib.loads((REPO / "pyproject.toml").read_text("utf-8"))["project"]["version"]
    document = rewrite_manifest(document, found, version, _utc_now())
    payload = (json.dumps(document, indent=1, ensure_ascii=True) + "\n").encode("utf-8")
    handle, temp = tempfile.mkstemp(dir=str(manifest_path.parent), prefix=".release-", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
        os.replace(temp, manifest_path)
    except BaseException:
        if os.path.exists(temp):
            os.unlink(temp)
        raise
    print(f"[apply] updated {manifest_path} (release_version {version})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="docker build and collect identities")
    build.add_argument("--platform", required=True, choices=sorted(PLATFORM_CAPABILITIES))
    build.add_argument("--built-on", required=True)
    build.add_argument("--out", required=True)
    build.add_argument("--ca", help="PEM passed only as the BuildKit secret build_ca")
    build.set_defaults(func=cmd_build)
    remote = sub.add_parser("remote", help="build on an ssh host from a clean git archive")
    remote.add_argument("--host", required=True)
    remote.add_argument("--platform", required=True, choices=[ARM64])
    remote.add_argument("--built-on", required=True)
    remote.add_argument("--out", required=True)
    remote.set_defaults(func=cmd_remote)
    apply = sub.add_parser("apply", help="validate identity files and update the manifest")
    apply.add_argument("files", nargs="+")
    apply.add_argument("--manifest", default=str(REPO / MANIFEST_REL))
    apply.set_defaults(func=cmd_apply)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except ToolError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
