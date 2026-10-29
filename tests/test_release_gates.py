"""P6-U6: GitHub release workflow gates and release-manifest finalization.

``.github/workflows/release.yml`` publishes the worker images to the GitHub Container Registry
and attaches the wheel to a GitHub Release; nothing is published from a workstation.  This file
is both the deterministic gate test and the small command-line tool the workflow invokes
(``python tests/test_release_gates.py <command>``):

``gate``            evaluate the namespace, version, build-input, NVIDIA and GPU gates and write
                    the gate records (``release-evidence/gates.json``) and the job outputs.
``pull-verify``     pull one published image by digest on the matching runner, check its
                    architecture, user and entrypoint, and replay one bounded ASR probe.
``finalize``        turn a copy of release.json into the published release manifest; the published
                    digests must equal the ``merge``/``cuda`` job outputs; the run fails unless the
                    result is publishable; the sha256 of the written bytes becomes a job output.
``check-finalized`` load a finalized manifest through the real loader in release mode, bind its
                    bytes to the ``finalize`` job output sha256 (and optionally its registry digests
                    to the ``merge``/``cuda`` outputs) and require ``publishable``.
``check-wheel``     check that a wheel packages exactly the finalized manifest bytes.

The finalization logic lives here, not in an inline workflow heredoc, so the same function the
workflow runs is tested against the real release.json (with fake digests marked as test data).

Publication integrity: the finalized manifest travels between jobs as a workflow artifact, so its
bytes are bound by job outputs, which belong to the run's graph rather than the artifact store.
``finalize`` exports ``manifest_sha256``; ``wheel``, ``wheel-install`` and ``release`` verify the
downloaded file (and the wheel's packaged copy) against it before copying, building or uploading.
The registry digests are bound the same way: ``merge`` and ``cuda`` export the index digests they
pushed, and the finalized registry must equal them.  ``publishable`` must be true at the
publication gate; a finalized manifest with a remaining release-critical blocker fails the run
before the ``wheel`` and ``release`` jobs (advisory blockers never block).

Release-critical blockers (``publishable`` is true only when none of them remain):
``publisher_namespace_missing`` always; ``registry_digest_missing``, ``nvidia_redistribution_review``,
``gpu_evidence_missing`` and ``worker_defect_open`` when their capability ships in the release
(media and asr_cpu always ship; asr_cuda ships only when the NVIDIA and GPU gates pass and it was
published).  ``long_duration_unvalidated`` is advisory and stays listed.  Unknown codes are
release-critical (fail closed).  A blocker is removed only when the workflow resolved it: the
namespace once it is set, a digest once that image was pushed and pull-verified, and the NVIDIA
review only together with a published CUDA image.

The workflow is read with a bounded YAML subset reader (block mappings and sequences, plain and
quoted scalars, one-line flow sequences, ``|`` literal blocks, full-line and trailing comments);
anything else is rejected rather than guessed, because PyYAML is not a dependency.

The live case (``-m live``) needs PODMUNCHER_RELEASE_MANIFEST and PODMUNCHER_RELEASE_WHEEL.  It
installs the wheel into a temporary venv outside the repository, runs ``podmuncher --help`` and
``--version``, compares the wheel's packaged release.json with the provided manifest and runs
the gate checks.  It never publishes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
import test_release_manifest as p6u5

import podmuncher
from podmuncher.manifests import ManifestError, load_manifest, resolve_capability

REPO = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO / ".github" / "workflows" / "release.yml"
MANIFEST_PATH = p6u5.MANIFEST_PATH
NEGATIVE = p6u5.NEGATIVE
SMOKE_MARKER = "PODMUNCHER_SMOKE_OK"
AMD64, ARM64 = p6u5.AMD64, p6u5.ARM64
CAPABILITIES = p6u5.CAPABILITIES
REQUIRED_CAPABILITIES = ("media", "asr_cpu")
REGISTRY = "ghcr.io"
MODEL_ORIGINS = ("https://huggingface.co",)
ADVISORY_CODES = frozenset({"long_duration_unvalidated"})
CAPABILITY_SCOPED_CRITICAL = frozenset({"registry_digest_missing", "nvidia_redistribution_review",
                                        "gpu_evidence_missing", "worker_defect_open"})
DOCKER_PLATFORM = {AMD64: "linux/amd64", ARM64: "linux/arm64"}
# Third-party actions, pinned by full commit SHA.  Source: ``gh api repos/<o>/<r>/commits/<tag>``
# (cross-checked with ``git ls-remote --tags``, peeled for annotated tags) on 2026-10-14.
PINNED_ACTIONS = {
    "actions/checkout": ("3d3c42e5aac5ba805825da76410c181273ba90b1", "v7.0.1"),
    "actions/setup-python": ("5fda3b95a4ea91299a34e894583c3862153e4b97", "v7.0.0"),
    "actions/upload-artifact": ("cf430e030ddbb5b0abf93d22962f4752f3646cd9", "v7.0.2"),
    "actions/download-artifact": ("9000827ccba6bdab643e8b6fd33ac0654aef8333", "v8.0.2"),
    "docker/setup-buildx-action": ("f87e5991a6d7451dcb8d9637bfbc97413f497069", "v4.4.1"),
    "docker/login-action": ("dbcb813823bdd20940b903addbd779551569679f", "v4.6.0"),
    "docker/build-push-action": ("c3c9e263c25d99ce0380d002d59b67737d91b0dc", "v7.4.0"),
}
GATE_JOBS = ("tests", "lint", "wheel-content")
PUSH_JOBS = frozenset({"build", "merge", "cuda"})
GATE_NAMES = frozenset({"namespace", "version", "build_inputs:media", "build_inputs:asr_cpu",
                        "build_inputs:asr_cuda", "nvidia_redistribution_review", "gpu_evidence",
                        "publish_cuda"})
# Fake registry digests for deterministic tests: TEST DATA, never a registry digest.
TEST_DATA_LABEL = "P6-U6 deterministic test data, not a registry digest"
TEST_NAMESPACE = "podmuncher-test"

_NAMESPACE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)*\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_VERSION = re.compile(r"[0-9]+(?:\.[0-9]+)*(?:(?:a|b|rc)[0-9]+)?(?:\.post[0-9]+)?(?:\.dev[0-9]+)?\Z")


class GateError(ValueError):
    """A release gate failed or a release input was rejected."""


# ---------------------------------------------------------------------- bounded YAML reader


def read_workflow_yaml(text: str) -> dict:
    """Parse the workflow's YAML subset; reject anything outside it."""
    lines: list[tuple[int, str]] = []
    for number, raw in enumerate(text.splitlines(), 1):
        body = raw.rstrip()
        leading = body[:len(body) - len(body.lstrip())]
        if "\t" in leading:
            raise GateError(f"line {number}: tab indentation")
        lines.append((len(leading), body.lstrip()))
    position = 0

    def skip() -> None:
        nonlocal position
        while position < len(lines) and (not lines[position][1] or lines[position][1].startswith("#")):
            position += 1

    def scalar(token: str) -> object:
        token = token.strip()
        if token[:1] in ("&", "*", "!", ">", "{") and token != "{}":
            raise GateError("unsupported YAML construct: " + token)
        if token == "{}":
            return {}
        if token[:1] in ("'", '"'):
            quote = token[0]
            end = 1
            while True:
                end = token.find(quote, end)
                if end < 0:
                    raise GateError("unterminated quote: " + token)
                if quote == "'" and token[end + 1:end + 2] == "'":
                    end += 2
                    continue
                if quote == '"' and token[end - 1] == "\\":
                    end += 1
                    continue
                break
            rest = token[end + 1:].strip()
            if rest and not rest.startswith("#"):
                raise GateError("text after a quoted scalar: " + token)
            inner = token[1:end]
            return inner.replace("''", "'") if quote == "'" else json.loads('"' + inner + '"')
        if token.startswith("["):
            if not token.endswith("]"):
                raise GateError("unsupported flow sequence: " + token)
            inner = token[1:-1].strip()
            return [scalar(part) for part in inner.split(",")] if inner else []
        if " #" in token:
            token = token.split(" #", 1)[0].rstrip()
        return token

    def literal(parent_indent: int) -> str:
        nonlocal position
        block: list[tuple[int, str]] = []
        while position < len(lines):
            indent, content = lines[position]
            if content and indent <= parent_indent:
                break
            block.append((indent, content))
            position += 1
        while block and not block[-1][1]:
            block.pop()
        base = min(indent for indent, content in block if content)
        return "".join((" " * (indent - base) + content if content else "") + "\n" for indent, content in block)

    def value_after_key(rest: str, indent: int) -> object:
        nonlocal position
        if rest in ("|", "|-"):
            text_value = literal(indent)
            return text_value.rstrip("\n") if rest == "|-" else text_value
        if rest and not rest.startswith("#"):
            return scalar(rest)
        skip()
        if position < len(lines) and lines[position][0] > indent:
            return block(lines[position][0])
        return None

    def mapping(indent: int, first: str | None = None) -> dict:
        nonlocal position
        result: dict = {}
        pending = first
        while True:
            if pending is None:
                skip()
                if position >= len(lines) or lines[position][0] != indent or lines[position][1].startswith("- "):
                    if position < len(lines) and lines[position][0] > indent:
                        raise GateError(f"unexpected indentation at line {position + 1}")
                    return result
                content = lines[position][1]
                position += 1
            else:
                content, pending = pending, None
            match = re.fullmatch(r"([A-Za-z0-9_.-]+):(?:\s+(.*))?", content)
            if match is None:
                raise GateError("not a mapping entry: " + content)
            key, rest = match.group(1), (match.group(2) or "").strip()
            if key in result:
                raise GateError("duplicate key: " + key)
            result[key] = value_after_key(rest, indent)

    def sequence(indent: int) -> list:
        nonlocal position
        result = []
        while True:
            skip()
            if position >= len(lines) or lines[position][0] != indent or not (
                    lines[position][1].startswith("- ") or lines[position][1] == "-"):
                return result
            item = lines[position][1][2:].strip()
            position += 1
            if re.fullmatch(r"[A-Za-z0-9_.-]+:(?:\s+.*)?", item):
                result.append(mapping(indent + 2, first=item))
            else:
                result.append(scalar(item))

    def block(indent: int) -> object:
        skip()
        content = lines[position][1]
        return sequence(indent) if content.startswith("- ") else mapping(indent)

    skip()
    document = block(0)
    skip()
    if position != len(lines) or type(document) is not dict:
        raise GateError("trailing content after the workflow document")
    return document


def load_workflow() -> tuple[dict, str]:
    text = WORKFLOW_PATH.read_text("utf-8")
    return read_workflow_yaml(text), text


def _needs(job: dict) -> list[str]:
    needs = job.get("needs") or []
    return [needs] if type(needs) is str else list(needs)


def ancestors(jobs: dict, name: str) -> set[str]:
    found: set[str] = set()
    stack = list(_needs(jobs[name]))
    while stack:
        current = stack.pop()
        if current not in found:
            found.add(current)
            stack.extend(_needs(jobs[current]))
    return found


def iter_steps(jobs: dict):
    for name, job in jobs.items():
        for step in job.get("steps") or []:
            yield name, step


# ---------------------------------------------------------------------------- gate logic


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _domain_text(value: object, what: str) -> str:
    if type(value) is not str or value in NEGATIVE or value != value.strip():
        raise GateError(f"{what} is missing or invalid")
    return value


def parse_flag(value: object, what: str) -> bool:
    """A repository-variable flag: exactly ``true`` or ``false``; unset (empty) means false."""
    if value in ("", None, "false"):
        return False
    if value == "true":
        return True
    raise GateError(f"{what} must be 'true' or 'false'")


def validate_namespace(namespace: object, owner: object | None = None) -> str:
    try:
        value = _domain_text(namespace, "publisher namespace")
    except GateError:
        raise GateError("No publisher namespace: set the PODMUNCHER_PUBLISHER_NAMESPACE repository variable "
                        "(or the namespace input). Publication fails closed without it.") from None
    if _NAMESPACE.fullmatch(value) is None or len(value) > 200:
        raise GateError("publisher namespace must be a lowercase ghcr.io path")
    if owner is not None and value.split("/")[0] != _domain_text(owner, "repository owner").lower():
        raise GateError("publisher namespace must start with the repository owner (GITHUB_TOKEN can "
                        "push only to the owner's packages)")
    return value


def image_reference(namespace: str, capability: str) -> str:
    return f"{REGISTRY}/{namespace}/podmuncher-{capability}"


def _record(gate: str, passed: bool, mandatory: bool, detail: str, **evidence) -> dict:
    return {"gate": gate, "passed": passed, "mandatory": mandatory, "detail": detail, "evidence": evidence}


def assert_gate_record(record: object, *, gate: str | None = None, passed: bool | None = None) -> dict:
    """Assert one gate record's shape (and optionally its gate and outcome)."""
    assert type(record) is dict and set(record) == {"gate", "passed", "mandatory", "detail", "evidence"}, record
    assert record["gate"] in GATE_NAMES, record["gate"]
    assert type(record["passed"]) is bool and type(record["mandatory"]) is bool
    assert type(record["detail"]) is str and record["detail"].strip() and record["detail"] not in NEGATIVE
    assert type(record["evidence"]) is dict
    for key, value in record["evidence"].items():
        assert type(key) is str and key not in NEGATIVE
        assert value is None or type(value) in (str, bool, int), (key, value)
        assert value not in NEGATIVE or value is None, (key, value)
    if gate is not None:
        assert record["gate"] == gate, (record["gate"], gate)
    if passed is not None:
        assert record["passed"] is passed, record
    return record


def build_input_binding(document: dict, capability: str) -> dict:
    """Recompute the Dockerfile and build-input hashes of this checkout against release.json."""
    identity = document["local_build_identities"][capability]
    path, target = p6u5.DOCKERFILES[capability]
    dockerfile_sha = hashlib.sha256((REPO / path).read_bytes()).hexdigest()
    try:
        inputs_sha = p6u5._build_inputs_sha(identity["build_inputs"])
    except OSError:
        inputs_sha = None
    passed = (identity["dockerfile"] == path and identity["dockerfile_target"] == target
              and identity["dockerfile_sha256"] == dockerfile_sha and identity["build_inputs_sha256"] == inputs_sha)
    detail = ("the checkout's Dockerfile and build inputs match the release candidate" if passed else
              "the Dockerfile or a build input differs from the release candidate; rebuild and regenerate release.json")
    return _record("build_inputs:" + capability, passed, capability in REQUIRED_CAPABILITIES, detail,
                   dockerfile=path, target=target, dockerfile_sha256=dockerfile_sha, build_inputs_sha256=inputs_sha)


def gpu_gate(document: dict, cuda_binding: dict) -> dict:
    """CUDA evidence must exist and bind to the release candidate built from the same inputs."""
    evidence = document.get("gpu_evidence") or {}
    local_id = document["local_build_identities"]["asr_cuda"]["platforms"][AMD64]["local_image_id"]
    probe = document["benchmarks"]["probes"].get("asr_cuda") or {}
    blocked = any(b["code"] == "gpu_evidence_missing" for b in document["release_status"]["blockers"])
    passed = (evidence.get("passthrough") is True and evidence.get("bound_image_id") == local_id
              and probe.get("bound_image_id") == local_id and probe.get("device") == "cuda"
              and bool(evidence.get("uuid")) and not blocked and cuda_binding["passed"])
    detail = ("release.json gpu_evidence binds to the CUDA candidate built from these inputs" if passed else
              "no GPU evidence bound to these CUDA build inputs; CUDA is not published (CI has no GPU)")
    return _record("gpu_evidence", passed, False, detail, bound_image_id=evidence.get("bound_image_id"),
                   local_image_id=local_id, build_inputs_sha256=cuda_binding["evidence"]["build_inputs_sha256"])


def evaluate_gates(document: dict, *, namespace: str, owner: str, version: str,
                   nvidia_reviewed: str) -> tuple[list[dict], dict[str, str]]:
    """Every gate record plus the job outputs.  Mandatory failures are raised by ``require``."""
    records = []
    try:
        validated = validate_namespace(namespace, owner)
        records.append(_record("namespace", True, True, "publisher namespace configured", namespace=validated))
    except GateError as error:
        validated = None
        records.append(_record("namespace", False, True, str(error), namespace=None))
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text("utf-8"))["project"]["version"]
    version_ok = (type(version) is str and _VERSION.fullmatch(version) is not None
                  and version == document["release_version"] == podmuncher.__version__ == pyproject)
    records.append(_record("version", version_ok, True,
                           "release version matches the package and release.json" if version_ok else
                           "release version must equal pyproject.toml, podmuncher.__version__ and release.json",
                           requested=version if type(version) is str and _VERSION.fullmatch(version) else None,
                           package=pyproject, manifest=document["release_version"]))
    bindings = {cap: build_input_binding(document, cap) for cap in CAPABILITIES}
    records.extend(bindings.values())
    reviewed = parse_flag(nvidia_reviewed, "PODMUNCHER_NVIDIA_REDISTRIBUTION_REVIEWED")
    records.append(_record("nvidia_redistribution_review", reviewed, False,
                           "owner recorded the NVIDIA redistribution review" if reviewed else
                           "NVIDIA redistribution review not recorded; CUDA is not published",
                           variable="PODMUNCHER_NVIDIA_REDISTRIBUTION_REVIEWED"))
    gpu = gpu_gate(document, bindings["asr_cuda"])
    records.append(gpu)
    publish_cuda = reviewed and gpu["passed"]
    records.append(_record("publish_cuda", publish_cuda, False,
                           "asr_cuda is published for linux-amd64" if publish_cuda else
                           "asr_cuda is skipped; its NVIDIA and digest blockers stay in the manifest",
                           nvidia_reviewed=reviewed, gpu_evidence=gpu["passed"]))
    outputs = {"namespace": validated or "", "version": version if version_ok else "",
               "publish_cuda": "true" if publish_cuda else "false"}
    return records, outputs


def require(records: list[dict]) -> None:
    failed = [r for r in records if r["mandatory"] and not r["passed"]]
    if failed:
        raise GateError("; ".join(f"{r['gate']}: {r['detail']}" for r in failed))


# -------------------------------------------------------------------------- finalization


def is_release_critical(blocker: dict, shipped: set[str]) -> bool:
    code, capability = blocker["code"], blocker["capability"]
    if code in ADVISORY_CODES:
        return False
    if code in CAPABILITY_SCOPED_CRITICAL and capability is not None:
        return capability in shipped or capability in REQUIRED_CAPABILITIES
    return True


def load_published(record: object, namespace: str, document: dict) -> dict:
    """Validate one ``published/<capability>.json`` record written by the workflow."""
    if type(record) is not dict or set(record) != {"capability", "image_reference", "digest", "platforms",
                                                   "platform_digests"}:
        raise GateError("published record keys")
    capability = record["capability"]
    if capability not in CAPABILITIES:
        raise GateError("published capability is missing or invalid")
    if record["image_reference"] != image_reference(namespace, capability):
        raise GateError("published image reference does not match the namespace")
    if type(record["digest"]) is not str or _DIGEST.fullmatch(record["digest"]) is None:
        raise GateError("published digest is missing or invalid")
    if record["platforms"] != document["registry"]["images"][capability]["platforms"]:
        raise GateError("published platforms differ from release.json registry platforms")
    digests = record["platform_digests"]
    if type(digests) is not dict or sorted(digests) != sorted(record["platforms"]) or any(
            type(d) is not str or _DIGEST.fullmatch(d) is None for d in digests.values()):
        raise GateError("published platform digests are missing or invalid")
    if capability == "asr_cuda" and digests != {AMD64: record["digest"]}:
        raise GateError("asr_cuda is a single linux-amd64 image")
    return record


def finalize(document: dict, *, namespace: str, version: str, published: dict[str, dict],
             nvidia_reviewed: bool, finalized_at: str | None = None) -> dict:
    """Return the finalized copy of release.json; the input document is not modified."""
    namespace = validate_namespace(namespace)
    if version != document["release_version"]:
        raise GateError("release version differs from release.json")
    for capability, record in published.items():
        if load_published(record, namespace, document)["capability"] != capability:
            raise GateError("published record filed under the wrong capability")
    if "asr_cuda" in published:
        if not nvidia_reviewed:
            raise GateError("asr_cuda was published without the NVIDIA redistribution review")
        if not gpu_gate(document, build_input_binding(document, "asr_cuda"))["passed"]:
            raise GateError("asr_cuda was published without GPU evidence bound to its build inputs")
    result = copy.deepcopy(document)
    result["generated_at"] = finalized_at or _now()
    registry = result["registry"]
    registry["publisher_namespace"] = namespace
    for capability, record in published.items():
        registry["images"][capability]["image_reference"] = record["image_reference"]
        registry["images"][capability]["registry_digest"] = record["digest"]
    for platform, manifest in result["runtime_manifests"].items():
        manifest["source"] = {
            "kind": "packaged_release",
            "label": f"podmuncher {version} release",
            "provenance": (f"P6-U6 GitHub release workflow for {platform}: images pushed to {REGISTRY}/{namespace} "
                           "and pull-verified by digest; models pinned by revision and per-file sha256"),
        }
        capabilities = []
        for capability in manifest["capabilities"]:
            record = published.get(capability["id"])
            if record is None:
                continue  # not published in this release: absent from its runtime manifest
            identity = result["local_build_identities"][capability["id"]]
            target = f" target {identity['dockerfile_target']}" if identity["dockerfile_target"] else ""
            capability["image_reference"] = record["image_reference"]
            capability["digest"] = record["digest"]
            capability["provenance"] = (
                f"{identity['dockerfile']}{target} built natively for {platform} by the P6-U6 GitHub release "
                f"workflow from build inputs sha256 {identity['build_inputs_sha256']}; pushed as "
                f"{record['image_reference']}@{record['digest']} and pull-verified by digest")
            capabilities.append(capability)
        manifest["capabilities"] = capabilities
    p6u5._rehash(result)
    shipped = set(published)
    remaining = []
    for blocker in result["release_status"]["blockers"]:
        code, capability = blocker["code"], blocker["capability"]
        if code == "publisher_namespace_missing":
            continue
        if code == "registry_digest_missing" and capability in shipped:
            continue
        if code == "nvidia_redistribution_review" and capability in shipped:
            continue
        remaining.append(blocker)
    result["release_status"] = {
        "publishable": not any(is_release_critical(b, shipped) for b in remaining),
        "blockers": remaining,
    }
    return result


def verify_sha256(data: bytes, expected: object, what: str) -> str:
    """``data`` must hash to ``expected`` (a job output: exactly 64 lowercase hex digits)."""
    if type(expected) is not str or _SHA256.fullmatch(expected) is None:
        raise GateError(f"{what}: the expected sha256 job output is missing or invalid")
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise GateError(f"{what}: sha256 {actual} differs from the job output {expected}")
    return actual


def parse_expected_digests(items: list[str] | None) -> dict[str, str | None] | None:
    """``capability=digest`` pairs from the merge/cuda job outputs; an empty digest means not published."""
    if items is None:
        return None
    expected: dict[str, str | None] = {}
    for item in items:
        capability, separator, digest = item.partition("=")
        if not separator or capability not in CAPABILITIES or capability in expected:
            raise GateError("expected digest must be <capability>=<digest> once per capability")
        if digest and _DIGEST.fullmatch(digest) is None:
            raise GateError(f"expected digest for {capability} is not a sha256 digest")
        expected[capability] = digest or None
    if set(expected) != set(CAPABILITIES):
        raise GateError("expected digests must name every capability: " + ", ".join(CAPABILITIES))
    for capability in REQUIRED_CAPABILITIES:
        if expected[capability] is None:
            raise GateError(f"the {capability} index digest job output is empty")
    return expected


def check_registry_digests(document: dict, expected: dict[str, str | None]) -> None:
    """The finalized registry digests must equal the digests the merge/cuda jobs pushed."""
    for capability in CAPABILITIES:
        actual = document["registry"]["images"][capability]["registry_digest"]
        if actual != expected[capability]:
            raise GateError(f"finalized registry digest for {capability} ({actual}) differs from the "
                            f"job output ({expected[capability]})")


def critical_blockers(document: dict) -> list[dict]:
    shipped = {cap for cap, image in document["registry"]["images"].items() if image["registry_digest"] is not None}
    return [b for b in document["release_status"]["blockers"] if is_release_critical(b, shipped)]


def require_publishable(document: dict) -> None:
    """The publication gate: ``publishable`` must be true; otherwise list the critical blockers."""
    status = document["release_status"]
    critical = critical_blockers(document)
    if status["publishable"] is not True or critical:
        listed = "; ".join(f"{b['code']} ({b['capability']}): {b['detail']}" for b in critical) or "none listed"
        raise GateError(f"not publishable (publishable={status['publishable']!r}); release-critical blockers "
                        f"remain, nothing is packaged or published: {listed}")


def check_finalized(document: dict, *, raw: bytes | None = None, sha256: object = None,
                    expected_digests: dict[str, str | None] | None = None) -> dict:
    """Validate a finalized manifest and load every runtime manifest in release mode.

    ``raw``/``sha256`` bind the exact file bytes to the ``finalize`` job output; ``expected_digests``
    bind the registry digests to the ``merge``/``cuda`` job outputs.
    """
    if raw is not None or sha256 is not None:
        if raw is None or p6u5._strict_json(raw) != document:
            raise GateError("the finalized manifest bytes do not parse to the checked document")
        verify_sha256(raw, sha256, "finalized release.json")
    if set(document) != p6u5.TOP_KEYS or document["document"] != "podmuncher-release-manifest":
        raise GateError("finalized document keys")
    namespace = validate_namespace(document["registry"]["publisher_namespace"])
    shipped = set()
    for capability, image in document["registry"]["images"].items():
        if image["platforms"] != p6u5.PLATFORMS[capability]:
            raise GateError("registry platforms")
        if image["registry_digest"] is not None:
            if image["image_reference"] != image_reference(namespace, capability) or _DIGEST.fullmatch(
                    image["registry_digest"]) is None:
                raise GateError("registry image reference or digest")
            shipped.add(capability)
        elif image["image_reference"] is not None:
            raise GateError("image reference without a digest")
    if expected_digests is not None:
        check_registry_digests(document, expected_digests)
    for value in (v for _, v in p6u5._strings(document)):
        if value in NEGATIVE:
            raise GateError("negative domain value in the finalized manifest")
    status = document["release_status"]
    codes = {(b["code"], b["capability"]) for b in status["blockers"]}
    if ("publisher_namespace_missing", None) in codes:
        raise GateError("namespace blocker left in a finalized manifest")
    for capability in CAPABILITIES:
        if (("registry_digest_missing", capability) in codes) == (capability in shipped):
            raise GateError("registry digest blocker disagrees with the registry for " + capability)
    if "asr_cuda" in shipped and ("nvidia_redistribution_review", "asr_cuda") in codes:
        raise GateError("asr_cuda published with an open NVIDIA review")
    if status["publishable"] != (not any(is_release_critical(b, shipped) for b in status["blockers"])):
        raise GateError("publishable disagrees with the release-critical blockers")
    loaded = {}
    with tempfile.TemporaryDirectory(prefix="pm-release-final-") as directory:
        for platform, manifest in sorted(document["runtime_manifests"].items()):
            data = p6u5.canonical(manifest)
            digest = hashlib.sha256(data).hexdigest()
            if digest != document["runtime_manifest_sha256"][platform]:
                raise GateError("runtime manifest hash mismatch")
            path = Path(directory) / f"{platform}.json"
            path.write_bytes(data)
            runtime = load_manifest(path, development=False, allowed_release_sources=(path,),
                                    allowed_asset_sources=MODEL_ORIGINS, expected_sha256=digest)
            expected = {cap for cap in shipped if platform in p6u5.PLATFORMS[cap]}
            if {c.capability.value for c in runtime.capabilities} != expected:
                raise GateError("runtime capabilities differ from the published images on " + platform)
            for capability in runtime.capabilities:
                image = document["registry"]["images"][capability.capability.value]
                if (capability.image_reference, capability.digest) != (image["image_reference"],
                                                                      image["registry_digest"]):
                    raise GateError("runtime capability does not reference the registry digest")
            loaded[platform] = runtime
    return loaded


def check_wheel(wheel: Path, manifest_bytes: bytes, sha256: object = None) -> str:
    """The wheel packages exactly this manifest (bound to the ``finalize`` job output when given)."""
    document = p6u5._strict_json(manifest_bytes)
    match = re.fullmatch(r"podmuncher-([^-]+)-py3-none-any\.whl", wheel.name)
    if match is None or match.group(1) != document["release_version"]:
        raise GateError("wheel name or version differs from the manifest release_version")
    if sha256 is not None:
        verify_sha256(manifest_bytes, sha256, "finalized release.json")
    with zipfile.ZipFile(wheel) as archive:
        packaged = archive.read("podmuncher/manifests/release.json")
    if packaged != manifest_bytes:
        raise GateError("the wheel does not package the provided release.json")
    if sha256 is not None:
        verify_sha256(packaged, sha256, f"{wheel.name} packaged release.json")
    return hashlib.sha256(packaged).hexdigest()


# ------------------------------------------------------------------------- pull-verify


def pull_verify(document: dict, record: dict, *, platform: str, probe: bool, work: Path) -> dict:
    """Pull by digest on the matching runner; check the image; replay one bounded ASR probe."""
    capability = record["capability"]
    if platform not in record["platforms"]:
        raise GateError(f"{capability} is not published for {platform}")
    reference = f"{record['image_reference']}@{record['digest']}"
    pulled = p6u5._docker("pull", "--platform", DOCKER_PLATFORM[platform], reference, timeout=1800)
    if pulled.returncode != 0:
        raise GateError(f"docker pull failed for {reference}")
    inspected = p6u5._docker("image", "inspect", reference, timeout=60)
    if inspected.returncode != 0:
        raise GateError(f"docker image inspect failed for {reference}")
    data, = json.loads(inspected.stdout)
    identity = document["local_build_identities"][capability]
    if data["Os"] != "linux" or data["Architecture"] != platform.split("-")[1]:
        raise GateError(f"{reference} is not a {platform} image")
    if data["Config"]["User"] != identity["user"] or data["Config"]["Entrypoint"] != identity["entrypoint"]:
        raise GateError(f"{reference} user or entrypoint differs from the release candidate")
    if reference not in (data.get("RepoDigests") or []):
        raise GateError(f"{reference} is not recorded as a pulled repository digest")
    result = {"capability": capability, "platform": platform, "reference": reference,
              "platform_digest": record["platform_digests"][platform], "image_id": data["Id"], "probe": None}
    if capability == "asr_cuda":
        result["probe"] = {"skipped": "GitHub-hosted runners have no NVIDIA GPU; release.json gpu_evidence gates CUDA"}
    elif capability == "asr_cpu" and probe:
        shown = p6u5.replay_probe(capability, data["Id"], work, gpu_uuid=None)
        runtime = next(c for c in document["runtime_manifests"][platform]["capabilities"] if c["id"] == capability)
        if (shown["runtime"]["name"] != "ctranslate2" or shown["runtime"]["version"] != runtime["runtime_version"]
                or shown["device"] != "cpu" or "int8" not in shown["compute_types"]):
            raise GateError(f"{reference} probe differs from the release candidate")
        result["probe"] = {"runtime": shown["runtime"], "device": shown["device"],
                           "compute_types": shown["compute_types"]}
    return result


# --------------------------------------------------------------------------------- CLI


def _read_document(path: str | Path) -> dict:
    return p6u5._strict_json(Path(path).read_bytes())


def _write_json(path: str | Path, value: object) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(value, indent=1, sort_keys=False) + "\n", encoding="utf-8")


def _github_output(path: str | None, values: dict[str, str]) -> None:
    if path:
        with open(path, "a", encoding="utf-8") as stream:
            for key, value in values.items():
                if "\n" in value:
                    raise GateError("multi-line job output")
                stream.write(f"{key}={value}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="test_release_gates.py")
    sub = parser.add_subparsers(dest="command", required=True)
    gate = sub.add_parser("gate")
    for name in ("--manifest", "--evidence"):
        gate.add_argument(name, required=True)
    for name in ("--namespace", "--owner", "--version", "--nvidia-reviewed"):
        gate.add_argument(name, default="")
    gate.add_argument("--github-output")
    verify = sub.add_parser("pull-verify")
    for name in ("--manifest", "--namespace", "--platform", "--published", "--evidence"):
        verify.add_argument(name, required=True)
    verify.add_argument("--probe", action="store_true")
    verify.add_argument("--github-output")
    final = sub.add_parser("finalize")
    for name in ("--manifest", "--published-dir", "--version", "--output"):
        final.add_argument(name, required=True)
    for name in ("--namespace", "--nvidia-reviewed", "--publish-cuda"):
        final.add_argument(name, default="")
    final.add_argument("--expected-digest", action="append", metavar="CAPABILITY=DIGEST",
                       help="index digest pushed by the merge/cuda job (job output); empty = not published")
    final.add_argument("--github-output")
    checked = sub.add_parser("check-finalized")
    checked.add_argument("--manifest", required=True)
    checked.add_argument("--sha256", required=True, help="finalize job output: sha256 of the finalized bytes")
    checked.add_argument("--expected-digest", action="append", metavar="CAPABILITY=DIGEST")
    wheel = sub.add_parser("check-wheel")
    wheel.add_argument("--manifest", required=True)
    wheel.add_argument("--wheel", required=True)
    wheel.add_argument("--sha256", required=True, help="finalize job output: sha256 of the finalized bytes")
    args = parser.parse_args(argv)
    try:
        if args.command == "gate":
            document = _read_document(args.manifest)
            records, outputs = evaluate_gates(document, namespace=args.namespace, owner=args.owner,
                                              version=args.version, nvidia_reviewed=args.nvidia_reviewed)
            _write_json(args.evidence, {"document": "podmuncher-release-gates", "checked_at": _now(),
                                        "records": records})
            for record in records:
                print(f"gate {record['gate']}: {'pass' if record['passed'] else 'FAIL'}"
                      f"{' (mandatory)' if record['mandatory'] else ''} - {record['detail']}")
            require(records)
            _github_output(args.github_output, outputs)
        elif args.command == "pull-verify":
            document = _read_document(args.manifest)
            namespace = validate_namespace(args.namespace)
            record = load_published(json.loads(Path(args.published).read_text("utf-8")), namespace, document)
            work = Path(tempfile.mkdtemp(prefix="pm-pull-verify-"))
            try:
                result = pull_verify(document, record, platform=args.platform, probe=args.probe, work=work)
            finally:
                shutil.rmtree(work, ignore_errors=True)
            _write_json(Path(args.evidence) / f"{record['capability']}-{args.platform}.json", result)
            platform_ref = f"{record['image_reference']}@{record['platform_digests'][args.platform]}"
            _github_output(args.github_output, {f"{record['capability']}_ref": platform_ref})
            print(f"verified {result['reference']} on {args.platform} probe={json.dumps(result['probe'])}")
        elif args.command == "finalize":
            document = _read_document(args.manifest)
            namespace = validate_namespace(args.namespace)
            publish_cuda = parse_flag(args.publish_cuda, "publish-cuda")
            published = {}
            for path in sorted(Path(args.published_dir).glob("*.json")):
                record = json.loads(path.read_text("utf-8"))
                if path.stem != record.get("capability"):
                    raise GateError("published file name differs from its capability")
                published[path.stem] = record
            missing = [cap for cap in REQUIRED_CAPABILITIES if cap not in published]
            if missing:
                raise GateError("required images were not published: " + ", ".join(missing))
            if ("asr_cuda" in published) != publish_cuda:
                raise GateError("asr_cuda publication disagrees with the publish_cuda gate")
            expected = parse_expected_digests(args.expected_digest)
            for capability in CAPABILITIES if expected is not None else ():
                record = published.get(capability)
                if (record["digest"] if record else None) != expected[capability]:
                    raise GateError(f"published {capability} digest differs from the merge/cuda job output")
            result = finalize(document, namespace=namespace, version=args.version, published=published,
                              nvidia_reviewed=parse_flag(args.nvidia_reviewed, "nvidia-reviewed"))
            check_finalized(result, expected_digests=expected)
            print(f"finalized publishable={result['release_status']['publishable']} blockers="
                  + ",".join(f"{b['code']}:{b['capability']}" for b in result["release_status"]["blockers"]))
            require_publishable(result)  # the publication gate: nothing downstream runs otherwise
            _write_json(args.output, result)
            raw = Path(args.output).read_bytes()
            check_finalized(p6u5._strict_json(raw), raw=raw, sha256=hashlib.sha256(raw).hexdigest(),
                            expected_digests=expected)
            _github_output(args.github_output, {"manifest_sha256": hashlib.sha256(raw).hexdigest()})
            print(f"finalized release.json sha256 {hashlib.sha256(raw).hexdigest()}")
        elif args.command == "check-finalized":
            raw = Path(args.manifest).read_bytes()
            document = p6u5._strict_json(raw)
            loaded = check_finalized(document, raw=raw, sha256=args.sha256,
                                     expected_digests=parse_expected_digests(args.expected_digest))
            require_publishable(document)
            print("release-mode load ok: " + ", ".join(sorted(loaded)) + f"; sha256 {args.sha256} bound")
        else:
            wheel_path, = [Path(p) for p in [args.wheel]]
            digest = check_wheel(wheel_path, Path(args.manifest).read_bytes(), sha256=args.sha256)
            print(f"wheel {wheel_path.name} packages release.json sha256 {digest}")
    except (GateError, ManifestError, p6u5.ReleaseManifestError, OSError, KeyError, ValueError) as error:
        reason = getattr(error, "reason_code", None)
        print(f"::error::release gate failed: {error}{f' ({reason})' if reason else ''}", file=sys.stderr)
        return 1
    return 0


# ------------------------------------------------------------------------ test helpers


def fake_digest(*parts: str) -> str:
    """A fake sha256 digest derived from TEST_DATA_LABEL: test data, never a registry digest."""
    return "sha256:" + hashlib.sha256(" ".join((TEST_DATA_LABEL, *parts)).encode("ascii")).hexdigest()


def make_release_inputs(*, publish: tuple[str, ...] = CAPABILITIES, namespace: str = TEST_NAMESPACE,
                        nvidia_reviewed: bool = True) -> dict:
    """The real release.json plus fake (test-data) published records for ``finalize``."""
    document = _read_document(MANIFEST_PATH)
    published = {}
    for capability in publish:
        platforms = document["registry"]["images"][capability]["platforms"]
        digests = {platform: fake_digest(capability, platform) for platform in platforms}
        digest = digests[AMD64] if capability == "asr_cuda" else fake_digest(capability, "index")
        published[capability] = {"capability": capability, "image_reference": image_reference(namespace, capability),
                                 "digest": digest, "platforms": list(platforms), "platform_digests": digests}
    return {"document": document, "namespace": namespace, "version": document["release_version"],
            "published": published, "nvidia_reviewed": nvidia_reviewed}


def _finalize(inputs: dict) -> dict:
    return finalize(inputs["document"], namespace=inputs["namespace"], version=inputs["version"],
                    published=inputs["published"], nvidia_reviewed=inputs["nvidia_reviewed"],
                    finalized_at="2026-10-14T00:00:00Z")


def _blocker_set(document: dict) -> set[tuple[str, str | None]]:
    return {(b["code"], b["capability"]) for b in document["release_status"]["blockers"]}


@pytest.fixture(scope="module")
def workflow():
    return load_workflow()


# ------------------------------------------------------------------- deterministic: workflow


def test_triggers_are_dispatch_with_version_and_tag_push_only(workflow):
    data, text = workflow
    assert set(data["on"]) == {"workflow_dispatch", "push"}
    version = data["on"]["workflow_dispatch"]["inputs"]["version"]
    assert version["required"] == "true" and version["type"] == "string"
    assert data["on"]["push"] == {"tags": ["v*"]}
    assert "pull_request_target" not in text and "pull_request" not in data["on"]


def test_every_action_is_pinned_to_a_full_commit_sha(workflow):
    data, text = workflow
    used = set()
    for _, step in iter_steps(data["jobs"]):
        if "uses" not in step:
            continue
        repo, _, ref = step["uses"].partition("@")
        assert repo in PINNED_ACTIONS, repo
        assert re.fullmatch(r"[0-9a-f]{40}", ref) and ref == PINNED_ACTIONS[repo][0], step["uses"]
        used.add(repo)
    assert used == set(PINNED_ACTIONS)
    for line in text.splitlines():
        if line.strip().startswith(("uses:", "- uses:")):
            repo, _sha = line.split("uses:", 1)[1].split("#")[0].strip().split("@")
            assert line.rstrip().endswith("# " + PINNED_ACTIONS[repo][1]), line
    assert "docker://" not in text and "uses: ./" not in text


def test_permissions_are_least_privilege_per_job(workflow):
    data, _ = workflow
    assert data["permissions"] == {}
    for name, job in data["jobs"].items():
        permissions = job["permissions"]
        assert type(permissions) is dict and permissions.get("contents") in ("read", "write"), name
        assert "id-token" not in permissions and "write-all" not in str(permissions)
        assert (permissions.get("packages") == "write") == (name in PUSH_JOBS), name
        assert (permissions.get("contents") == "write") == (name == "release"), name
        assert set(permissions) <= {"contents", "packages"}, name
        if name in PUSH_JOBS:
            assert any(step.get("uses", "").startswith("docker/login-action@") for step in job["steps"]), name


def test_no_secret_except_github_token_and_none_in_run_lines(workflow):
    data, text = workflow
    assert set(re.findall(r"secrets\.([A-Za-z0-9_]+)", text)) == {"GITHUB_TOKEN"}
    for name, step in iter_steps(data["jobs"]):
        if "run" in step:
            assert "${{" not in step["run"], (name, step.get("name"))  # values arrive via env only
            assert "GITHUB_TOKEN" not in step["run"] and "GH_TOKEN" not in step["run"]
        if step.get("uses", "").startswith("docker/login-action@"):
            assert step["with"] == {"registry": "ghcr.io", "username": "${{ github.actor }}",
                                    "password": "${{ secrets.GITHUB_TOKEN }}"}
        for key, value in (step.get("env") or {}).items():
            if "secrets." in value or "github.token" in value:
                assert (name, key, value) == ("release", "GH_TOKEN", "${{ github.token }}")
    for name, step in iter_steps(data["jobs"]):
        if step.get("uses", "").startswith("actions/checkout@"):
            assert step["with"]["persist-credentials"] == "false", name


def test_gate_cli_fails_closed_without_a_namespace(tmp_path, capsys):
    evidence = tmp_path / "gates.json"
    output = tmp_path / "github_output"
    code = main(["gate", "--manifest", str(MANIFEST_PATH), "--namespace", "", "--owner", TEST_NAMESPACE,
                 "--version", podmuncher.__version__, "--evidence", str(evidence), "--github-output", str(output)])
    assert code == 1
    assert "PODMUNCHER_PUBLISHER_NAMESPACE" in capsys.readouterr().err
    assert not output.exists()
    records = json.loads(evidence.read_text("utf-8"))["records"]
    assert {r["gate"] for r in records} == GATE_NAMES
    for record in records:
        assert_gate_record(record)
    code = main(["gate", "--manifest", str(MANIFEST_PATH), "--namespace", TEST_NAMESPACE, "--owner", "Podmuncher-Test",
                 "--version", podmuncher.__version__, "--evidence", str(evidence), "--github-output", str(output)])
    assert code == 0
    shown = dict(line.split("=", 1) for line in output.read_text("utf-8").splitlines())
    assert shown == {"namespace": TEST_NAMESPACE, "version": podmuncher.__version__, "publish_cuda": "false"}


def test_version_gate_binds_the_package_version():
    document = make_release_inputs()["document"]
    for version in ("9.9.9", *NEGATIVE):
        records, outputs = evaluate_gates(document, namespace=TEST_NAMESPACE, owner=TEST_NAMESPACE,
                                          version=version, nvidia_reviewed="")
        assert_gate_record(records[1], gate="version", passed=False)
        assert outputs["version"] == ""
    records, _ = evaluate_gates(document, namespace=TEST_NAMESPACE, owner=TEST_NAMESPACE,
                                version=podmuncher.__version__, nvidia_reviewed="")
    assert_gate_record(records[1], gate="version", passed=True)


def test_platforms_match_the_release_registry(workflow):
    jobs = workflow[0]["jobs"]
    registry = make_release_inputs()["document"]["registry"]["images"]
    built: dict[str, list[str]] = {}
    for entry in jobs["build"]["strategy"]["matrix"]["include"]:
        assert entry["docker_platform"] == DOCKER_PLATFORM[entry["platform"]]
        assert entry["runner"] == ("ubuntu-24.04" if entry["platform"] == AMD64 else "ubuntu-24.04-arm")
        dockerfile, target = p6u5.DOCKERFILES[entry["capability"]]
        assert (entry["dockerfile"], entry["target"] or None) == (dockerfile, target)
        built.setdefault(entry["capability"], []).append(entry["platform"])
    assert built == {cap: registry[cap]["platforms"] for cap in REQUIRED_CAPABILITIES}
    assert jobs["merge"]["strategy"]["matrix"]["capability"] == list(REQUIRED_CAPABILITIES)
    cuda = next(s for s in jobs["cuda"]["steps"] if s.get("uses", "").startswith("docker/build-push-action@"))
    assert cuda["with"]["platforms"] == "linux/amd64" and registry["asr_cuda"]["platforms"] == [AMD64]
    for name, step in iter_steps(jobs):
        if step.get("uses", "").startswith("docker/build-push-action@"):
            assert step["with"]["context"] == "." and "secrets" not in step["with"], name
            assert step["with"]["provenance"] == "false" and step["with"]["sbom"] == "false"
            assert "build-args" not in step["with"]
    for name in ("media", "asr"):
        text = (REPO / "containers" / f"{name}.Dockerfile").read_text("utf-8")
        mounts = re.findall(r"--mount=type=secret,[^ ]*", text)
        assert mounts and all("required=false" in mount for mount in mounts), name  # CI never fabricates build_ca


def test_jobs_run_in_order_and_stop_on_failure(workflow):
    jobs = workflow[0]["jobs"]
    assert set(jobs) == {"tests", "config", "build", "merge", "cuda", "release"}
    assert _needs(jobs["config"]) == ["tests"] and set(_needs(jobs["merge"])) == {"config", "build"}
    assert "if" not in jobs["cuda"] and "if" not in jobs["release"]  # CUDA always ships; any failure stops the release
    assert set(_needs(jobs["release"])) == {"config", "merge", "cuda"}
    for name, step in iter_steps(jobs):
        assert "continue-on-error" not in step, name
        for line in step.get("run", "").splitlines():
            if "pip install" in line:
                assert "--require-hashes -r" in line or "--no-deps" in line, (name, line)
    release = "".join(s.get("run", "") for s in jobs["release"]["steps"])
    assert release.index("finalize") < release.index("python -m build") < release.index("gh release create")
    assert "tar -czf upload/security.tar.gz security" in release


def test_no_package_index_publication(workflow):
    data, text = workflow
    runs = "\n".join(step.get("run", "") for _, step in iter_steps(data["jobs"])).lower()
    for forbidden in ("twine", "pypi", "upload.pypi", "--repository-url", "flit publish", "poetry publish",
                      "uv publish", "hatch publish", "docker push", "docker login"):
        assert forbidden not in runs, forbidden
    assert "gh-action-pypi-publish" not in text and "id-token" not in text
    release = data["jobs"]["release"]
    assert "gh release create" in release["steps"][-1]["run"]


# --------------------------------------------------------------- deterministic: finalize


def test_finalized_manifest_loads_in_release_mode_and_removes_exactly_the_resolved_blockers():
    inputs = make_release_inputs()
    before = _blocker_set(inputs["document"])
    original = json.dumps(inputs["document"], sort_keys=True)
    result = _finalize(inputs)
    assert json.dumps(inputs["document"], sort_keys=True) == original  # the source document is untouched
    removed = before - _blocker_set(result)
    assert removed == {("publisher_namespace_missing", None), ("nvidia_redistribution_review", "asr_cuda"),
                       *{("registry_digest_missing", cap) for cap in CAPABILITIES}}
    assert _blocker_set(result) == {b for b in before if b[0] == "long_duration_unvalidated"}
    assert result["release_status"]["publishable"] is True  # long_duration is advisory
    loaded = check_finalized(result)
    assert set(loaded) == {AMD64, ARM64}
    for platform, manifest in loaded.items():
        assert manifest.development_fixture is False and manifest.release_version == inputs["version"]
        assert manifest.sha256 == result["runtime_manifest_sha256"][platform]
        media = resolve_capability(manifest, "media")
        record = inputs["published"]["media"]
        assert media.worker_policy.image == f"{record['image_reference']}@{record['digest']}"
        assert media.worker_policy.allow_pull and not media.worker_policy.development_image
    cuda = resolve_capability(loaded[AMD64], "asr_cuda", selected_gpu_uuid=result["gpu_evidence"]["uuid"])
    assert cuda.worker_policy.image.startswith(image_reference(TEST_NAMESPACE, "asr_cuda") + "@sha256:")
    assert result["registry"]["publisher_namespace"] == TEST_NAMESPACE
    for cap in CAPABILITIES:
        image = result["registry"]["images"][cap]
        assert image["image_reference"] == f"ghcr.io/{TEST_NAMESPACE}/podmuncher-{cap}"
        assert image["registry_digest"] == inputs["published"][cap]["digest"]
    # Local build identities, model provenance, GPU evidence and the inventory are carried over unchanged.
    for key in ("local_build_identities", "model_selection", "gpu_evidence", "benchmarks", "provenance_inventory"):
        assert result[key] == inputs["document"][key], key


def test_an_absent_digest_keeps_its_blocker(capsys):
    inputs = make_release_inputs(publish=("asr_cpu", "asr_cuda"))
    result = _finalize(inputs)
    blockers = _blocker_set(result)
    assert ("registry_digest_missing", "media") in blockers
    assert ("registry_digest_missing", "asr_cpu") not in blockers
    assert result["registry"]["images"]["media"] == {"image_reference": None, "registry_digest": None,
                                                     "platforms": [AMD64, ARM64]}
    assert result["release_status"]["publishable"] is False  # media always ships: its digest is critical
    loaded = check_finalized(result)
    assert all("media" not in {c.capability.value for c in m.capabilities} for m in loaded.values())
    with pytest.raises(GateError):
        _run_finalize_cli(inputs, publish_cuda="true", drop=("media",))
    assert "required images were not published: media" in capsys.readouterr().err


def test_unreviewed_nvidia_keeps_the_cuda_blockers():
    inputs = make_release_inputs(publish=REQUIRED_CAPABILITIES, nvidia_reviewed=False)
    result = _finalize(inputs)
    blockers = _blocker_set(result)
    assert {("nvidia_redistribution_review", "asr_cuda"), ("registry_digest_missing", "asr_cuda")} <= blockers
    assert ("registry_digest_missing", "media") not in blockers
    assert result["release_status"]["publishable"] is True  # CUDA is not part of this release
    loaded = check_finalized(result)
    assert {c.capability.value for c in loaded[AMD64].capabilities} == {"media", "asr_cpu"}
    assert result["registry"]["images"]["asr_cuda"]["registry_digest"] is None
    with pytest.raises(GateError, match="NVIDIA"):
        _finalize(make_release_inputs(nvidia_reviewed=False))
    unbound = make_release_inputs()
    unbound["document"]["gpu_evidence"]["passthrough"] = False
    with pytest.raises(GateError, match="GPU evidence"):
        _finalize(unbound)


def test_release_critical_codes_are_explicit():
    def critical(code, capability, shipped=("media", "asr_cpu")):
        return is_release_critical({"code": code, "capability": capability}, set(shipped))

    assert critical("publisher_namespace_missing", None)
    assert critical("registry_digest_missing", "media") and critical("registry_digest_missing", "asr_cpu")
    assert not critical("registry_digest_missing", "asr_cuda")
    assert critical("registry_digest_missing", "asr_cuda", CAPABILITIES)
    assert not critical("nvidia_redistribution_review", "asr_cuda")
    assert critical("nvidia_redistribution_review", "asr_cuda", CAPABILITIES)
    assert critical("gpu_evidence_missing", "asr_cuda", CAPABILITIES)
    assert critical("worker_defect_open", "asr_cpu")
    assert not critical("long_duration_unvalidated", "asr_cpu")
    assert not critical("long_duration_unvalidated", "asr_cuda", CAPABILITIES)
    assert critical("some_future_code", None) and critical("some_future_code", "asr_cuda")
    assert set(p6u5.BLOCKER_CODES) == {"publisher_namespace_missing"} | CAPABILITY_SCOPED_CRITICAL | ADVISORY_CODES


@pytest.mark.parametrize("bad", NEGATIVE)
@pytest.mark.parametrize("field", ["namespace", "digest", "capability", "platform", "platform_digest", "version"])
def test_negative_domain_values_are_rejected(field, bad):
    inputs = make_release_inputs()
    record = inputs["published"]["media"]
    if field == "namespace":
        inputs["namespace"] = bad
    elif field == "digest":
        record["digest"] = bad
    elif field == "capability":
        record["capability"] = bad
    elif field == "platform":
        record["platforms"][0] = bad
    elif field == "platform_digest":
        record["platform_digests"][AMD64] = bad
    else:
        inputs["version"] = bad
    with pytest.raises(GateError):
        _finalize(inputs)


@pytest.mark.parametrize("mutation", ["runtime_digest", "publishable", "stale_hash", "dropped_digest_blocker",
                                      "namespace", "fixture_reference", "nvidia_open"])
def test_tampered_finalized_manifests_are_rejected(mutation):
    result = _finalize(make_release_inputs())
    amd64 = result["runtime_manifests"][AMD64]
    expected: type[Exception] = GateError
    if mutation == "runtime_digest":
        amd64["capabilities"][0]["digest"] = fake_digest("tampered")
        p6u5._rehash(result)
    elif mutation == "publishable":
        result["release_status"]["publishable"] = False
    elif mutation == "stale_hash":
        amd64["capabilities"][0]["worker_memory_bytes"] += 1
    elif mutation == "dropped_digest_blocker":
        result["registry"]["images"]["media"]["registry_digest"] = None
        result["registry"]["images"]["media"]["image_reference"] = None
    elif mutation == "namespace":
        result["registry"]["publisher_namespace"] = "__invalid__"
    elif mutation == "fixture_reference":
        amd64["source"]["kind"] = "development_fixture"
        p6u5._rehash(result)
        expected = ManifestError
    else:
        result["release_status"]["blockers"].append({"code": "nvidia_redistribution_review", "capability": "asr_cuda",
                                                     "resolved_by": "owner review", "detail": "reopened"})
    with pytest.raises(expected):
        check_finalized(result)


def _expected_digest_args(inputs: dict, drop: tuple[str, ...] = (), override: dict | None = None) -> list[str]:
    """``--expected-digest`` arguments as the workflow passes them from the merge/cuda job outputs."""
    arguments = []
    for capability in CAPABILITIES:
        record = inputs["published"].get(capability)
        digest = record["digest"] if record and capability not in drop else ""
        digest = (override or {}).get(capability, digest)
        arguments += ["--expected-digest", f"{capability}={digest}"]
    return arguments


def _run_finalize_cli(inputs: dict, *, publish_cuda: str, drop: tuple[str, ...] = (),
                      expected: list[str] | None = None, manifest: Path = MANIFEST_PATH) -> dict:
    with tempfile.TemporaryDirectory(prefix="pm-p6u6-cli-") as directory:
        root = Path(directory)
        published = root / "published"
        published.mkdir()
        for cap, record in inputs["published"].items():
            if cap not in drop:
                (published / f"{cap}.json").write_text(json.dumps(record), encoding="utf-8")
        output = root / "out" / "release.json"
        github_output = root / "github_output"
        arguments = _expected_digest_args(inputs, drop) if expected is None else expected
        code = main(["finalize", "--manifest", str(manifest), "--published-dir", str(published),
                     "--namespace", inputs["namespace"], "--version", inputs["version"],
                     "--nvidia-reviewed", "true" if inputs["nvidia_reviewed"] else "",
                     "--publish-cuda", publish_cuda, *arguments, "--output", str(output),
                     "--github-output", str(github_output)])
        if code != 0:
            assert not github_output.exists()  # no sha256 output for a rejected finalization
            raise GateError("finalize command failed")
        shown = dict(line.split("=", 1) for line in github_output.read_text("utf-8").splitlines())
        sha256 = hashlib.sha256(output.read_bytes()).hexdigest()
        assert shown == {"manifest_sha256": sha256}
        assert main(["check-finalized", "--manifest", str(output), "--sha256", sha256, *arguments]) == 0
        assert main(["check-finalized", "--manifest", str(output), "--sha256", sha256]) == 0
        return json.loads(output.read_text("utf-8"))


def test_finalize_command_round_trip_and_cuda_agreement():
    inputs = make_release_inputs()
    result = _run_finalize_cli(inputs, publish_cuda="true")
    assert result["registry"]["images"]["asr_cuda"]["registry_digest"] == inputs["published"]["asr_cuda"]["digest"]
    with pytest.raises(GateError):
        _run_finalize_cli(inputs, publish_cuda="false")  # a CUDA record the gate did not allow
    with pytest.raises(GateError):
        _run_finalize_cli(inputs, publish_cuda="true", drop=("asr_cuda",))  # gate allowed CUDA, nothing published
    unreviewed = make_release_inputs(publish=REQUIRED_CAPABILITIES, nvidia_reviewed=False)
    result = _run_finalize_cli(unreviewed, publish_cuda="false")
    assert ("nvidia_redistribution_review", "asr_cuda") in _blocker_set(result)


def test_finalize_command_binds_the_published_digests_to_the_job_outputs(capsys):
    inputs = make_release_inputs()
    substituted = {"media": fake_digest("substituted media index")}
    with pytest.raises(GateError):
        _run_finalize_cli(inputs, publish_cuda="true", expected=_expected_digest_args(inputs, override=substituted))
    assert "published media digest differs from the merge/cuda job output" in capsys.readouterr().err
    for override, message in (({"media": ""}, "media index digest job output is empty"),
                              ({"asr_cuda": ""}, "published asr_cuda digest differs"),
                              ({"asr_cpu": "__invalid__"}, "not a sha256 digest")):
        with pytest.raises(GateError):
            _run_finalize_cli(inputs, publish_cuda="true", expected=_expected_digest_args(inputs, override=override))
        assert message in capsys.readouterr().err
    for arguments in (_expected_digest_args(inputs)[:4], [*_expected_digest_args(inputs), "--expected-digest",
                                                             "media=" + inputs["published"]["media"]["digest"]]):
        with pytest.raises((GateError, SystemExit)):
            _run_finalize_cli(inputs, publish_cuda="true", expected=arguments)
    capsys.readouterr()
    unreviewed = make_release_inputs(publish=REQUIRED_CAPABILITIES, nvidia_reviewed=False)
    with pytest.raises(GateError):  # a CUDA digest output without a published CUDA record
        _run_finalize_cli(unreviewed, publish_cuda="false",
                          expected=_expected_digest_args(unreviewed, override={"asr_cuda": fake_digest("cuda")}))
    assert "published asr_cuda digest differs" in capsys.readouterr().err


def test_finalize_command_fails_the_publication_gate_when_not_publishable(tmp_path, capsys):
    inputs = make_release_inputs(publish=("asr_cpu", "asr_cuda"))
    assert _finalize(inputs)["release_status"]["publishable"] is False  # media is required and unpublished
    with pytest.raises(GateError):
        _run_finalize_cli(inputs, publish_cuda="true")
    assert "required images were not published: media" in capsys.readouterr().err  # refused before finalizing
    # Every image published, yet a release-critical blocker the workflow cannot resolve remains.
    inputs = make_release_inputs()
    defective = copy.deepcopy(inputs["document"])
    defective["release_status"]["blockers"].append({
        "code": "worker_defect_open", "capability": "asr_cpu", "resolved_by": "worker fix",
        "detail": "an open asr_cpu worker defect blocks this release"})
    manifest = tmp_path / "release.json"
    manifest.write_text(json.dumps(defective, indent=1) + "\n", encoding="utf-8")
    inputs["document"] = defective
    result = _finalize(inputs)
    assert result["release_status"]["publishable"] is False and check_finalized(result)
    with pytest.raises(GateError):
        _run_finalize_cli(inputs, publish_cuda="true", manifest=manifest)
    err = capsys.readouterr().err
    assert "not publishable" in err and "worker_defect_open (asr_cpu)" in err
    assert "long_duration_unvalidated" not in err  # advisory blockers are not listed as the cause


def test_check_finalized_binds_the_bytes_to_the_finalize_job_output(tmp_path, capsys):
    result = _finalize(make_release_inputs())
    raw = json.dumps(result, indent=1).encode("utf-8")
    sha256 = hashlib.sha256(raw).hexdigest()
    assert set(check_finalized(result, raw=raw, sha256=sha256)) == {AMD64, ARM64}
    # A substituted manifest with other registry/runtime digests and recomputed internal hashes is
    # internally consistent, so only the byte binding rejects it.
    substituted = copy.deepcopy(result)
    other = fake_digest("substituted media index")
    substituted["registry"]["images"]["media"]["registry_digest"] = other
    for manifest in substituted["runtime_manifests"].values():
        for capability in manifest["capabilities"]:
            if capability["id"] == "media":
                capability["digest"] = other
    p6u5._rehash(substituted)
    substituted_raw = json.dumps(substituted, indent=1).encode("utf-8")
    assert check_finalized(substituted)  # passes without the binding: the finding this test pins
    with pytest.raises(GateError, match="differs from the job output"):
        check_finalized(substituted, raw=substituted_raw, sha256=sha256)
    with pytest.raises(GateError, match="do not parse to the checked document"):
        check_finalized(result, raw=substituted_raw, sha256=sha256)
    for bad in ("", None, sha256.upper(), sha256[:-1], "__invalid__", *NEGATIVE):
        with pytest.raises(GateError, match="sha256"):
            check_finalized(result, raw=raw, sha256=bad)
    path = tmp_path / "release.json"
    path.write_bytes(substituted_raw)
    assert main(["check-finalized", "--manifest", str(path), "--sha256", sha256]) == 1
    assert "differs from the job output" in capsys.readouterr().err
    assert main(["check-finalized", "--manifest", str(path), "--sha256", hashlib.sha256(substituted_raw).hexdigest()]) == 0


def test_finalized_registry_digests_must_equal_the_job_output_digests(tmp_path, capsys):
    inputs = make_release_inputs()
    result = _finalize(inputs)
    expected = {cap: inputs["published"][cap]["digest"] for cap in CAPABILITIES}
    assert check_finalized(result, expected_digests=expected)
    for capability in CAPABILITIES:
        with pytest.raises(GateError, match=f"registry digest for {capability}"):
            check_finalized(result, expected_digests={**expected, capability: fake_digest("other", capability)})
    with pytest.raises(GateError, match="registry digest for asr_cuda"):
        check_finalized(result, expected_digests={**expected, "asr_cuda": None})
    without_cuda = _finalize(make_release_inputs(publish=REQUIRED_CAPABILITIES, nvidia_reviewed=False))
    assert check_finalized(without_cuda, expected_digests={**expected, "asr_cuda": None})
    with pytest.raises(GateError, match="registry digest for asr_cuda"):
        check_finalized(without_cuda, expected_digests=expected)
    assert parse_expected_digests(None) is None
    for items in (["media=sha256:" + "0" * 64], ["media", "asr_cpu=", "asr_cuda="], ["nope=", "media=", "asr_cpu="],
                  ["media=sha256:" + "0" * 64, "asr_cpu=", "asr_cuda="]):
        with pytest.raises(GateError):
            parse_expected_digests(items)
    raw = json.dumps(result, indent=1).encode("utf-8")
    path = tmp_path / "release.json"
    path.write_bytes(raw)
    sha256 = hashlib.sha256(raw).hexdigest()
    arguments = _expected_digest_args(inputs)
    assert main(["check-finalized", "--manifest", str(path), "--sha256", sha256, *arguments]) == 0
    code = main(["check-finalized", "--manifest", str(path), "--sha256", sha256,
                 *_expected_digest_args(inputs, override={"asr_cpu": fake_digest("other index")})])
    assert code == 1 and "registry digest for asr_cpu" in capsys.readouterr().err


def test_publication_gate_requires_publishable(tmp_path, capsys):
    publishable = _finalize(make_release_inputs())
    require_publishable(publishable)  # advisory long_duration blockers remain: still publishable
    assert _blocker_set(publishable) and critical_blockers(publishable) == []
    unpublishable = _finalize(make_release_inputs(publish=("asr_cpu", "asr_cuda")))
    assert unpublishable["release_status"]["publishable"] is False
    assert check_finalized(unpublishable)  # consistent shape, but it must not reach the publication gate
    with pytest.raises(GateError, match=r"registry_digest_missing \(media\)"):
        require_publishable(unpublishable)
    assert [b["code"] for b in critical_blockers(unpublishable)] == ["registry_digest_missing"]
    raw = json.dumps(unpublishable, indent=1).encode("utf-8")
    path = tmp_path / "release.json"
    path.write_bytes(raw)
    assert main(["check-finalized", "--manifest", str(path), "--sha256", hashlib.sha256(raw).hexdigest()]) == 1
    err = capsys.readouterr().err
    assert "not publishable" in err and "registry_digest_missing (media)" in err
    inconsistent = copy.deepcopy(publishable)
    inconsistent["release_status"]["publishable"] = False
    with pytest.raises(GateError, match="not publishable"):
        require_publishable(inconsistent)
    with pytest.raises(GateError, match="disagrees"):
        check_finalized(inconsistent)
    assert check_finalized(unpublishable) and require_publishable(publishable) is None


def test_check_wheel_compares_the_packaged_manifest(tmp_path):
    result = _finalize(make_release_inputs())
    data = json.dumps(result, indent=1).encode("utf-8")
    sha256 = hashlib.sha256(data).hexdigest()
    wheel = tmp_path / f"podmuncher-{result['release_version']}-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("podmuncher/manifests/release.json", data)
    assert check_wheel(wheel, data) == check_wheel(wheel, data, sha256=sha256) == sha256
    with pytest.raises(GateError):
        check_wheel(wheel, MANIFEST_PATH.read_bytes())
    with pytest.raises(GateError, match="differs from the job output"):
        check_wheel(wheel, data, sha256=hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest())
    for bad in ("", "__invalid__", sha256.upper(), *NEGATIVE):
        with pytest.raises(GateError, match="sha256"):
            check_wheel(wheel, data, sha256=bad)
    renamed = tmp_path / "podmuncher-9.9.9-py3-none-any.whl"
    shutil.copy(wheel, renamed)
    with pytest.raises(GateError):
        check_wheel(renamed, data)
    # The packaged copy must byte-equal the finalized file even when the provided bytes hash correctly.
    other = tmp_path / "other" / wheel.name
    other.parent.mkdir()
    with zipfile.ZipFile(other, "w") as archive:
        archive.writestr("podmuncher/manifests/release.json", json.dumps(result, indent=2).encode("utf-8"))
    with pytest.raises(GateError, match="does not package"):
        check_wheel(other, data, sha256=sha256)
    manifest_path = tmp_path / "release.json"
    manifest_path.write_bytes(data)
    assert main(["check-wheel", "--manifest", str(manifest_path), "--wheel", str(wheel), "--sha256", sha256]) == 0
    assert main(["check-wheel", "--manifest", str(manifest_path), "--wheel", str(other), "--sha256", sha256]) == 1
    assert main(["check-wheel", "--manifest", str(manifest_path), "--wheel", str(wheel), "--sha256", "0" * 64]) == 1


def test_yaml_reader_rejects_constructs_outside_its_subset():
    assert read_workflow_yaml("a:\n  b: 'x' # c\n  c: [1, 2]\n  d: |\n    one\n      two\n") == {
        "a": {"b": "x", "c": ["1", "2"], "d": "one\n  two\n"}}
    for text in ("a: &anchor 1\n", "a: *ref\n", "a: >\n  folded\n", "a: 1\na: 2\n", "a:\n\tb: 1\n",
                 "a: {b: 1}\n", "a: 'open\n", "a: 1\n    b: 2\n"):
        with pytest.raises(GateError):
            read_workflow_yaml(text)


# --------------------------------------------------------------------------------- live


@pytest.mark.live
def test_release_validation_against_the_built_wheel_and_manifest():
    began = time.monotonic()
    manifest_value = os.environ.get("PODMUNCHER_RELEASE_MANIFEST")
    wheel_value = os.environ.get("PODMUNCHER_RELEASE_WHEEL")
    assert manifest_value, "PODMUNCHER_RELEASE_MANIFEST must name the release manifest to validate"
    assert wheel_value, "PODMUNCHER_RELEASE_WHEEL must name the built podmuncher wheel to validate"
    manifest_path, wheel_path = Path(manifest_value).resolve(), Path(wheel_value).resolve()
    assert manifest_path.is_file() and wheel_path.is_file(), "the release manifest or wheel does not exist"
    manifest_bytes = manifest_path.read_bytes()
    document = p6u5._strict_json(manifest_bytes)
    finalized = all(m["source"]["kind"] == "packaged_release" for m in document["runtime_manifests"].values())
    if finalized:
        loaded = check_finalized(document)
        mode = "finalized"
    else:
        _, loaded = p6u5.load_release_manifest(manifest_path)  # local candidate: development-mode loader
        assert document["release_status"]["publishable"] is False
        mode = "candidate"
    wheel_sha = check_wheel(wheel_path, manifest_bytes)

    records, outputs = evaluate_gates(document, namespace=document["registry"]["publisher_namespace"] or "",
                                      owner=(document["registry"]["publisher_namespace"] or "").split("/")[0],
                                      version=document["release_version"], nvidia_reviewed="")
    for record in records:
        assert_gate_record(record)
    gates = {r["gate"]: r["passed"] for r in records}
    assert gates["version"] and gates["build_inputs:media"] and gates["build_inputs:asr_cpu"]
    assert gates["namespace"] is finalized  # a local candidate has no namespace: publication stays closed
    if not finalized:
        with pytest.raises(GateError, match="namespace"):
            require(records)
    assert outputs["publish_cuda"] == "false"  # no NVIDIA review is asserted from this transport

    venv_root = Path(tempfile.mkdtemp(prefix="pm-p6u6-venv-"))
    assert REPO not in (venv_root, *venv_root.parents), "the temporary venv must be outside the repository"
    try:
        subprocess.run([sys.executable, "-m", "venv", str(venv_root / "venv")], check=True, timeout=300)
        bindir = venv_root / "venv" / ("Scripts" if os.name == "nt" else "bin")
        python = bindir / ("python.exe" if os.name == "nt" else "python")
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV")}
        env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        for arguments in (["--require-hashes", "-r", str(REPO / "requirements.lock")], ["--no-deps", str(wheel_path)]):
            installed = subprocess.run([str(python), "-m", "pip", "install", "-q", *arguments], cwd=venv_root, env=env,
                                       capture_output=True, text=True, timeout=900, check=False)
            assert installed.returncode == 0, installed.stdout[-2000:] + installed.stderr[-2000:]
        cli = bindir / ("podmuncher.exe" if os.name == "nt" else "podmuncher")
        shown = subprocess.run([str(cli), "--version"], cwd=venv_root, env=env, capture_output=True, text=True,
                               timeout=120, check=False)
        assert shown.returncode == 0 and shown.stdout.strip() == f"podmuncher {document['release_version']}", shown
        helped = subprocess.run([str(cli), "--help"], cwd=venv_root, env=env, capture_output=True, text=True,
                                timeout=120, check=False)
        assert helped.returncode == 0 and "podmuncher" in helped.stdout and "--version" in helped.stdout
        probe = ("import hashlib, importlib.resources as r, podmuncher; "
                 "p = r.files('podmuncher').joinpath('manifests', 'release.json'); "
                 "print(hashlib.sha256(p.read_bytes()).hexdigest(), podmuncher.__file__)")
        packaged = subprocess.run([str(python), "-I", "-c", probe], cwd=venv_root, env=env, capture_output=True,
                                  text=True, timeout=120, check=False)
        assert packaged.returncode == 0, packaged.stderr[-2000:]
        digest, module = packaged.stdout.split()
        assert digest == wheel_sha == hashlib.sha256(manifest_bytes).hexdigest()
        assert Path(module).resolve().is_relative_to(venv_root.resolve())
    finally:
        shutil.rmtree(venv_root, ignore_errors=True)
    assert not venv_root.exists()
    elapsed = time.monotonic() - began
    print(f"manifest {mode} {manifest_path.name} sha256={hashlib.sha256(manifest_bytes).hexdigest()[:16]} "
          f"runtime_manifests={','.join(sorted(loaded))}")
    print(f"wheel {wheel_path.name} packaged release.json matches; clean venv --version: {shown.stdout.strip()}")
    for record in records:
        print(f"gate {record['gate']}={'pass' if record['passed'] else 'fail'} - {record['detail']}")
    blockers = ", ".join(sorted(f"{b['code']}:{b['capability']}" for b in document["release_status"]["blockers"]))
    print(f"release publishable={document['release_status']['publishable']} blockers=[{blockers}] "
          f"publish=none elapsed={elapsed:.1f}s")
    print(SMOKE_MARKER)


if __name__ == "__main__":
    sys.exit(main())
