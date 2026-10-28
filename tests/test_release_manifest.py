"""P6-U5: verified local release-candidate manifest, license inventory and live asset comparison.

``src/podmuncher/manifests/release.json`` is a release document that wraps strict runtime
manifests, one per Docker engine platform (``linux-amd64``, ``linux-arm64``).
``manifests.load_manifest`` rejects unknown keys, so the release status (blockers), local build
identities, model provenance, benchmark summaries and the provenance inventory sit beside the
runtime manifests.  Each embedded runtime manifest is serialized canonically and loaded through the
real loader in development mode: release mode needs a registry ``image_reference@digest``, and the
registry digests and the publisher namespace are explicit release blockers until the P6-U6 GitHub
release workflow publishes the images (SPEC_GAP-14).  Local image IDs are local build identities,
never registry digests.

Model choice (SPEC_GAP-15/16) uses the shipped hardware table; every table model is pinned here
(exact Hugging Face revision, per-file size and sha256, model-card licence, languages).  Measured
envelopes are optional and appear only where real evidence exists: CPU tiny/base/small on the
current linux-amd64 build.  Images are multi-arch (SPEC_GAP-17): media and asr_cpu for linux-amd64
and linux-arm64, asr_cuda for linux-amd64 only; image sizes are the maximum over architectures.

Deterministic cases need no network or Docker.  The live case (``-m live``) requires
PODMUNCHER_RELEASE_MANIFEST (the manifest to verify) and PODMUNCHER_TEST_MODEL_ROOT (a ModelCache
root outside the repository).  It compares the local images for the engine's platform, re-verifies
the pinned model files (tiny always, every other model only when it is already cached; the rest
are checked deterministically for shape) and replays one bounded capability probe per built ASR
image through the owned DockerRuntime.

The checks accept two states of the same document.  The local (unpublished) state is the file
in the repository: ``development_fixture`` runtime manifests bound to local build identities, no
registry digests.  The finalized (published) state is what the P6-U6 release workflow writes with
``tests/test_release_gates.py finalize``: ``packaged_release`` runtime manifests loaded in release
mode, registry index digests in place, and the resolved blockers removed.  In both states
``publishable`` may be true only when no release-critical blocker remains; the classification
mirrors ``test_release_gates.is_release_critical`` (``long_duration_unvalidated`` is advisory,
unknown codes are critical) and a deterministic test keeps the two in agreement.  For a finalized
manifest the live case reads each published index anonymously with ``docker buildx imagetools
inspect --raw`` (never a login, push or tag) and replays no probe: the workflow's pull-verify step
probes the published images on matching runners.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from dataclasses import asdict
from importlib import resources
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit
from uuid import uuid4

import pytest

from podmuncher.domains import Capability, DomainValidationError, ModelTier
from podmuncher.manifests import (
    MIN_WORKER_OVERHEAD_BYTES,
    ManifestError,
    RuntimeManifest,
    engine_arch_platform,
    load_manifest,
    resolve_capability,
    supports_engine_arch,
)

REPO = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO / "src" / "podmuncher" / "manifests" / "release.json"
HARDWARE_TABLE = REPO / "src" / "podmuncher" / "assets" / "model_hardware.json"
LICENSE_PATH = REPO / "LICENSE"
APACHE_2_SHA256 = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
NEGATIVE = json.loads((REPO / "Docs" / "contract-domains.json").read_text("utf-8"))["negative_examples"]
SMOKE_MARKER = "PODMUNCHER_SMOKE_OK"
SIX_HOURS = 21600
GIB = 1024**3

AMD64, ARM64 = "linux-amd64", "linux-arm64"
TOP_KEYS = {
    "document", "document_version", "release_version", "generated_at", "release_status", "registry",
    "local_build_identities", "runtime_manifests", "runtime_manifest_sha256", "model_selection",
    "gpu_evidence", "benchmarks", "provenance_inventory",
}
CAPABILITIES = ("media", "asr_cpu", "asr_cuda")
PLATFORMS = {"media": [AMD64, ARM64], "asr_cpu": [AMD64, ARM64], "asr_cuda": [AMD64]}
# Blocker classification; mirrors tests/test_release_gates.py (agreement is tested below).
ADVISORY_CODES = frozenset({"long_duration_unvalidated"})
CAPABILITY_SCOPED_CRITICAL = frozenset({"registry_digest_missing", "nvidia_redistribution_review",
                                        "gpu_evidence_missing", "worker_defect_open"})
BLOCKER_CODES = {"publisher_namespace_missing"} | CAPABILITY_SCOPED_CRITICAL | ADVISORY_CODES
REQUIRED_CAPABILITIES = ("media", "asr_cpu")  # always ship; asr_cuda ships only when published
REGISTRY = "ghcr.io"
MODEL_ORIGINS = ("https://huggingface.co",)
LOCAL, FINALIZED = "local", "finalized"
INDEX_MEDIA_TYPES = {"application/vnd.oci.image.index.v1+json",
                     "application/vnd.docker.distribution.manifest.list.v2+json"}
IMAGE_MEDIA_TYPES = {"application/vnd.oci.image.manifest.v1+json",
                     "application/vnd.docker.distribution.manifest.v2+json"}
CODECS = {"wav", "mp3", "mp4_aac"}
POLICY_COMPUTE = {"asr_cpu": "int8", "asr_cuda": "int8_float16"}
DOCKERFILES = {"media": ("containers/media.Dockerfile", None), "asr_cpu": ("containers/asr.Dockerfile", "cpu"),
               "asr_cuda": ("containers/asr.Dockerfile", "cuda")}
# faster_whisper.utils._MODELS (faster-whisper 1.2.1, read inside the ASR image) for the table models.
MODEL_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "distil-large-v3": "Systran/faster-distil-whisper-large-v3",
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "large-v2": "Systran/faster-whisper-large-v2",
    "large-v3": "Systran/faster-whisper-large-v3",
}
# faster_whisper.utils.download_model allow_patterns.
MODEL_FILE_PATTERNS = re.compile(r"(?:config\.json|preprocessor_config\.json|model\.bin|tokenizer\.json"
                                 r"|vocabulary\.[a-z]+)\Z")
# The P3-U6 smoke fixture pins tiny at this revision (tests/test_asr_worker.py FIXTURE_REVISION).
TINY_REVISION = "d90ca5fe260221311c53c58e660288d3deb8d356"
TINY_WEIGHTS_SHA256 = "dcb76c6586fc06cbdac6dd21f14cfd129cc4cdd9dce19bf4ffa62e59cbe6e6d1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
_LOCAL_TAG = re.compile(r"[a-z0-9][a-z0-9._-]{0,100}:[A-Za-z0-9_][A-Za-z0-9_.-]{0,100}\Z")
_PLACEHOLDER = re.compile(r"(?i)\b(?:todo|tbd|fixme|xxx+|placeholder|changeme|lorem|dummy|example\.com)\b")
_REPEATED_HEX = re.compile(r"(?:sha256:)?([0-9a-f])\1{15,}\Z")
_NAMESPACE = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*(?:/[a-z0-9]+(?:[._-][a-z0-9]+)*)*\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
DOCKER = shutil.which("docker") or ("/usr/local/bin/docker" if os.path.isfile("/usr/local/bin/docker")
                                    else "docker")


class ReleaseManifestError(ValueError):
    """A release document failure (wrapper level); runtime manifests raise ManifestError."""


# ------------------------------------------------------------------------------ helpers


def _strict_json(raw: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ReleaseManifestError("duplicate key")
            result[key] = value
        return result

    def constant(_value):
        raise ReleaseManifestError("non-finite number")

    try:
        value = json.loads(raw.decode("utf-8", errors="strict"), object_pairs_hook=pairs,
                           parse_constant=constant)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ReleaseManifestError("malformed JSON") from error
    if type(value) is not dict:
        raise ReleaseManifestError("not an object")
    return value


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _nonempty(value: object, what: str) -> str:
    if type(value) is not str or not value.strip() or value in NEGATIVE:
        raise ReleaseManifestError(what)
    return value


def is_release_critical(blocker: dict, shipped: set[str]) -> bool:
    """Same rule as test_release_gates.is_release_critical: advisory never blocks, unknown codes do."""
    code, capability = blocker["code"], blocker["capability"]
    if code in ADVISORY_CODES:
        return False
    if code in CAPABILITY_SCOPED_CRITICAL and capability is not None:
        return capability in shipped or capability in REQUIRED_CAPABILITIES
    return True


def release_state(document: dict) -> str:
    """FINALIZED when every runtime manifest is ``packaged_release``, else LOCAL (development loader)."""
    manifests = document.get("runtime_manifests")
    if type(manifests) is not dict or not manifests:
        raise ReleaseManifestError("runtime manifest set")
    kinds = set()
    for manifest in manifests.values():
        source = manifest.get("source") if type(manifest) is dict else None
        kinds.add(source.get("kind") if type(source) is dict else None)
    return FINALIZED if kinds == {"packaged_release"} else LOCAL


def _check_registry(document: dict, state: str) -> set[str]:
    """Check the registry block for ``state``; return the capabilities with a registry digest."""
    registry = document["registry"]
    if type(registry) is not dict or set(registry) != {"publisher_namespace", "images"} \
            or type(registry["images"]) is not dict or set(registry["images"]) != set(CAPABILITIES):
        raise ReleaseManifestError("registry")
    namespace = registry["publisher_namespace"]
    if state == FINALIZED and (type(namespace) is not str or len(namespace) > 200
                               or _NAMESPACE.fullmatch(namespace) is None):
        raise ReleaseManifestError("finalized manifest without a valid publisher namespace")
    shipped = set()
    for cap, image in registry["images"].items():
        if type(image) is not dict or set(image) != {"image_reference", "registry_digest", "platforms"}:
            raise ReleaseManifestError("registry image keys")
        if image["platforms"] != PLATFORMS[cap]:
            raise ReleaseManifestError("registry image platforms")
        if image["registry_digest"] is not None and (image["image_reference"] is None or namespace is None):
            raise ReleaseManifestError("registry digest without a registry reference")
        if state == LOCAL:
            if image["registry_digest"] is not None:
                raise ReleaseManifestError("registry digest in an unpublished (development) manifest")
            continue
        if image["registry_digest"] is None:
            if image["image_reference"] is not None:
                raise ReleaseManifestError("registry reference without a digest")
            continue
        if type(image["registry_digest"]) is not str or _DIGEST.fullmatch(image["registry_digest"]) is None:
            raise ReleaseManifestError("malformed registry digest")
        if image["image_reference"] != f"{REGISTRY}/{namespace}/podmuncher-{cap}":
            raise ReleaseManifestError("registry reference does not match the publisher namespace")
        shipped.add(cap)
    return shipped


def _check_wrapper(document: dict) -> str:
    """Check the release wrapper; return the document state (LOCAL or FINALIZED)."""
    if set(document) != TOP_KEYS:
        raise ReleaseManifestError("top-level keys")
    if document["document"] != "podmuncher-release-manifest" or document["document_version"] != 2:
        raise ReleaseManifestError("document kind")
    state = release_state(document)
    status = document["release_status"]
    if type(status) is not dict or set(status) != {"publishable", "blockers"} \
            or type(status["publishable"]) is not bool or type(status["blockers"]) is not list:
        raise ReleaseManifestError("release_status")
    for blocker in status["blockers"]:
        if type(blocker) is not dict or set(blocker) != {"code", "capability", "resolved_by", "detail"}:
            raise ReleaseManifestError("blocker keys")
        if blocker["code"] not in BLOCKER_CODES:
            raise ReleaseManifestError("blocker code")
        if blocker["capability"] is not None and blocker["capability"] not in CAPABILITIES:
            raise ReleaseManifestError("blocker capability")
        _nonempty(blocker["resolved_by"], "blocker resolved_by")
        _nonempty(blocker["detail"], "blocker detail")
    shipped = _check_registry(document, state)
    critical = [b for b in status["blockers"] if is_release_critical(b, shipped)]
    if status["publishable"] and critical:
        raise ReleaseManifestError("publishable with release-critical blockers")
    if state == FINALIZED:
        codes = {(b["code"], b["capability"]) for b in status["blockers"]}
        if ("publisher_namespace_missing", None) in codes:
            raise ReleaseManifestError("namespace blocker left in a finalized manifest")
        for cap in CAPABILITIES:
            if (("registry_digest_missing", cap) in codes) == (cap in shipped):
                raise ReleaseManifestError("registry_digest_missing disagrees with the registry for " + cap)
        if "asr_cuda" in shipped and ("nvidia_redistribution_review", "asr_cuda") in codes:
            raise ReleaseManifestError("asr_cuda published with an open NVIDIA review")
        if status["publishable"] != (not critical):
            raise ReleaseManifestError("publishable disagrees with the release-critical blockers")
    identities = document["local_build_identities"]
    if set(identities) != set(CAPABILITIES):
        raise ReleaseManifestError("local identities")
    for cap, identity in identities.items():
        if identity["identity_kind"] != "local_image_id" or set(identity["platforms"]) != set(PLATFORMS[cap]):
            raise ReleaseManifestError("local identity kind or platforms")
        for local in identity["platforms"].values():
            if not _IMAGE_ID.fullmatch(local["local_image_id"]):
                raise ReleaseManifestError("local image ID")
            if not _LOCAL_TAG.fullmatch(local["local_tag"]) or "/" in local["local_tag"]:
                raise ReleaseManifestError("local tag carries a registry namespace")
    if set(document["runtime_manifests"]) != {AMD64, ARM64}:
        raise ReleaseManifestError("runtime manifest set")
    if set(document["runtime_manifest_sha256"]) != set(document["runtime_manifests"]):
        raise ReleaseManifestError("runtime manifest hashes")
    return state


def shipped_capabilities(document: dict) -> set[str]:
    """Capabilities with a registry digest (always empty for a local manifest)."""
    return {cap for cap, image in document["registry"]["images"].items() if image["registry_digest"] is not None}


def load_release_manifest(source: str | os.PathLike | None = None) -> tuple[dict, dict[str, RuntimeManifest]]:
    """Strictly parse the release document and load each embedded runtime manifest.

    ``source`` defaults to the packaged resource (``importlib.resources``).  Every runtime
    manifest must hash to its recorded canonical sha256 and load through the real loader:
    in development mode for a local manifest (no registry digests yet), and in release mode
    for a finalized manifest, with the same allowed sources as test_release_gates.check_finalized
    and each capability bound to its registry ``image_reference@digest``.
    Returns the document and the loaded runtime manifests keyed by platform.
    """
    if source is None:
        raw = resources.files("podmuncher").joinpath("manifests", "release.json").read_bytes()
    else:
        raw = Path(source).read_bytes()
    document = _strict_json(raw)
    state = _check_wrapper(document)
    shipped = shipped_capabilities(document)
    loaded: dict[str, RuntimeManifest] = {}
    with tempfile.TemporaryDirectory(prefix="pm-release-") as directory:
        for platform, manifest in sorted(document["runtime_manifests"].items()):
            data = canonical(manifest)
            digest = hashlib.sha256(data).hexdigest()
            if digest != document["runtime_manifest_sha256"][platform]:
                raise ReleaseManifestError("runtime manifest hash mismatch")
            path = Path(directory) / (platform + ".json")
            path.write_bytes(data)
            if state == LOCAL:
                loaded[platform] = load_manifest(path, development=True, expected_sha256=digest)
            else:
                runtime = load_manifest(path, development=False, allowed_release_sources=(path,),
                                        allowed_asset_sources=MODEL_ORIGINS, expected_sha256=digest)
                expected = {cap for cap in shipped if platform in PLATFORMS[cap]}
                if {c.capability.value for c in runtime.capabilities} != expected:
                    raise ReleaseManifestError("runtime capabilities differ from the published images on " + platform)
                for capability in runtime.capabilities:
                    image = document["registry"]["images"][capability.capability.value]
                    if (capability.image_reference, capability.digest) != (image["image_reference"],
                                                                          image["registry_digest"]):
                        raise ReleaseManifestError("runtime capability does not reference the registry digest")
                loaded[platform] = runtime
            # Development mode admits any non-empty licence/provenance text; a release document must
            # still never carry the contract-domains negatives anywhere in its runtime manifests.
            if any(value in NEGATIVE for _, value in _strings(manifest)):
                raise ReleaseManifestError("negative domain value in a runtime manifest")
    return document, loaded


@functools.cache
def _docker_endpoint() -> str:
    """The current Docker context endpoint (Windows npipe or Mac/Linux unix socket)."""
    try:
        result = subprocess.run((DOCKER, "context", "inspect"), shell=False, check=False, capture_output=True,
                                text=True, encoding="utf-8", errors="replace", timeout=30)
        data, = json.loads(result.stdout)
        return data["Endpoints"]["docker"]["Host"]
    except (OSError, subprocess.SubprocessError, ValueError, LookupError, TypeError) as error:
        raise AssertionError('Could not resolve the current Docker context with "docker context inspect"; '
                             "start Docker and select a local context") from error


def _docker(*arguments: str, timeout: float = 120) -> subprocess.CompletedProcess:
    return subprocess.run((DOCKER, "--host", _docker_endpoint(), *arguments), shell=False, check=False,
                          capture_output=True, timeout=timeout)


def _save_size(tag: str, timeout: float) -> int:
    process = subprocess.Popen((DOCKER, "--host", _docker_endpoint(), "save", tag), stdout=subprocess.PIPE,
                               stderr=subprocess.DEVNULL)
    total = 0
    deadline = time.monotonic() + timeout
    try:
        while chunk := process.stdout.read(4 * 1024 * 1024):
            total += len(chunk)
            if time.monotonic() > deadline:
                raise AssertionError("docker save exceeded its bound for " + tag)
        assert process.wait(timeout=60) == 0, "docker save failed for " + tag
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    return total


def engine_platform() -> str:
    """The manifest platform id of the local Docker engine (docker info Architecture)."""
    shown = _docker("info", "--format", "{{.Architecture}}", timeout=60)
    assert shown.returncode == 0, "docker info failed; start Docker"
    platform = engine_arch_platform(shown.stdout.decode().strip())
    assert platform in (AMD64, ARM64), "The Docker engine architecture is not linux-amd64 or linux-arm64"
    return platform


def inspect_registry_raw(reference: str) -> bytes:
    """Read one published manifest anonymously (``docker buildx imagetools inspect --raw``).

    Read-only: never logs in, pushes or tags.  Fails when the registry is unreachable or the
    package is private, because the finalized check must not pass without evidence.
    """
    shown = _docker("buildx", "imagetools", "inspect", "--raw", reference, timeout=120)
    assert shown.returncode == 0, (
        f"anonymous registry read failed for {reference}; this check never logs in, so the package must be "
        "public and the registry reachable: " + shown.stderr.decode("utf-8", "replace")[-500:])
    return shown.stdout


def compare_registry_assets(document: dict, platform: str, *, inspect=inspect_registry_raw) -> dict[str, dict]:
    """Check each published image for ``platform`` against its registry manifest (finalized state).

    The raw manifest must hash to the recorded registry digest; an index must list exactly the
    recorded platforms (attestation entries aside); a single image manifest is accepted only for a
    single-platform capability (asr_cuda), whose platform the workflow recorded at publication.
    """
    observed = {}
    for cap, image in sorted(document["registry"]["images"].items()):
        if image["registry_digest"] is None or platform not in image["platforms"]:
            continue
        reference = f"{image['image_reference']}@{image['registry_digest']}"
        raw = inspect(reference)
        assert "sha256:" + hashlib.sha256(raw).hexdigest() == image["registry_digest"], (
            cap, "registry manifest bytes do not hash to the recorded digest")
        data = json.loads(raw)
        media_type = data.get("mediaType")
        if media_type in INDEX_MEDIA_TYPES:
            listed = set()
            for entry in data["manifests"]:
                annotations = entry.get("annotations") or {}
                if annotations.get("vnd.docker.reference.type") == "attestation-manifest":
                    continue
                shown = entry["platform"]
                listed.add(f"{shown['os']}-{shown['architecture']}")
            assert listed == set(image["platforms"]), (cap, sorted(listed), image["platforms"])
            kind = "index"
        else:
            assert media_type in IMAGE_MEDIA_TYPES, (cap, media_type)
            assert image["platforms"] == [platform], (cap, "a multi-platform image must be an index")
            kind = "image"
        observed[cap] = {"id": image["registry_digest"], "kind": kind, "platforms": list(image["platforms"])}
    return observed


def compare_live_assets(document: dict, platform: str, *, save_timeout: float = 240,
                        inspect=inspect_registry_raw) -> dict[str, dict]:
    """Inspect each release candidate for ``platform``.

    Local state: the local image ID, the docker-save size and (where recorded with ``docker image
    inspect``) the unpacked size must match the recorded local build identity.  Finalized state:
    the registry references are checked with :func:`compare_registry_assets` (read-only).
    """
    if release_state(document) == FINALIZED:
        return compare_registry_assets(document, platform, inspect=inspect)
    observed = {}
    for cap, identity in sorted(document["local_build_identities"].items()):
        local = identity["platforms"].get(platform)
        if local is None:
            continue  # not built for this architecture (asr_cuda on linux-arm64)
        inspected = _docker("image", "inspect", local["local_tag"])
        assert inspected.returncode == 0, f"local image {local['local_tag']} is missing; rebuild it first"
        data, = json.loads(inspected.stdout)
        assert data["Id"] == local["local_image_id"], (cap, data["Id"], local["local_image_id"])
        assert data["Architecture"] == platform.split("-")[1]
        assert data["Config"]["User"] == "65532:65532"
        assert data["Config"]["Entrypoint"] == identity["entrypoint"]
        if local["unpacked_bytes_method"].startswith("docker image inspect Size"):
            assert data["Size"] == local["unpacked_bytes"], (cap, data["Size"], local["unpacked_bytes"])
        saved = _save_size(local["local_tag"], save_timeout)
        assert saved == local["compressed_bytes"], (cap, saved, local["compressed_bytes"])
        observed[cap] = {"id": data["Id"], "unpacked": data["Size"], "compressed": saved}
    return observed


def _strings(value, path=""):
    if type(value) is str:
        yield path, value
    elif type(value) is dict:
        for key, item in value.items():
            yield path + "/" + key, key
            yield from _strings(item, path + "/" + key)
    elif type(value) is list:
        for index, item in enumerate(value):
            yield from _strings(item, f"{path}[{index}]")


def _pairs(value, path=""):
    if type(value) is dict:
        for key, item in value.items():
            yield path + "/" + key, key, item
            yield from _pairs(item, path + "/" + key)
    elif type(value) is list:
        for index, item in enumerate(value):
            yield from _pairs(item, f"{path}[{index}]")


def _dockerfile_requirements_sha(cap: str) -> str:
    text = (REPO / DOCKERFILES[cap][0]).read_text("utf-8")
    blocks = dict(re.findall(r"COPY <<'REQUIREMENTS' (\S+)\n(.*?)REQUIREMENTS\n", text, re.DOTALL))
    body = {"media": blocks.get("/opt/media-requirements.txt", ""),
            "asr_cpu": blocks.get("/opt/asr-requirements.txt", ""),
            "asr_cuda": blocks.get("/opt/asr-requirements.txt", "")
            + blocks.get("/opt/asr-cuda-requirements.txt", "")}[cap]
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def _build_inputs_sha(files: list[str]) -> str:
    lines = (f"{name} {hashlib.sha256((REPO / name).read_bytes()).hexdigest()}" for name in files)
    return hashlib.sha256("\n".join(lines).encode("ascii")).hexdigest()


def _current(records: list[dict], cap: str, model_id: str | None = None) -> list[dict]:
    return [r for r in records if r["capability"] == cap and r["bound_to_current_build"]
            and (model_id is None or r["model_id"] == model_id)]


@pytest.fixture(scope="module")
def release():
    return load_release_manifest(MANIFEST_PATH)


def _write(tmp_path: Path, document: dict) -> Path:
    path = tmp_path / "release.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def _rehash(document: dict) -> dict:
    document["runtime_manifest_sha256"] = {
        key: hashlib.sha256(canonical(value)).hexdigest() for key, value in document["runtime_manifests"].items()}
    return document


# ------------------------------------------------------------------------ deterministic


def test_packaged_resource_is_the_repository_file():
    packaged = resources.files("podmuncher").joinpath("manifests", "release.json")
    assert packaged.is_file()
    assert packaged.read_bytes() == MANIFEST_PATH.read_bytes()
    assert not (REPO / "src" / "podmuncher" / "manifests" / "__init__.py").exists()
    document, loaded = load_release_manifest()  # default: importlib.resources
    assert set(loaded) == set(document["runtime_manifests"]) == {AMD64, ARM64}


def test_runtime_manifests_load_and_resolve_with_local_identities(release):
    document, loaded = release
    identities = document["local_build_identities"]
    for platform, manifest in loaded.items():
        assert manifest.development_fixture is True
        assert manifest.sha256 == document["runtime_manifest_sha256"][platform]
        assert manifest.platforms == (AMD64, ARM64)
        expected_caps = {cap for cap in CAPABILITIES if platform in PLATFORMS[cap]}
        assert {c.capability.value for c in manifest.capabilities} == expected_caps
        for capability in manifest.capabilities:
            cap = capability.capability.value
            identity = identities[cap]
            assert capability.platform == platform
            assert list(capability.platforms) == PLATFORMS[cap]
            assert capability.image_reference == "fixture-local"
            assert capability.digest == identity["platforms"][platform]["local_image_id"]
            # Sizes are the maximum over architectures.
            assert capability.compressed_bytes == identity["max_compressed_bytes"] == max(
                local["compressed_bytes"] for local in identity["platforms"].values())
            assert capability.unpacked_peak_bytes == identity["max_unpacked_bytes"] == max(
                local["unpacked_bytes"] for local in identity["platforms"].values())
            assert capability.worker_overhead_bytes >= MIN_WORKER_OVERHEAD_BYTES
            assert capability.worker_memory_bytes == document["model_selection"]["worker_memory_bytes"][cap]
            assert capability.dependency_lock_sha256 == identity["dependency_requirements_sha256"]
            assert capability.dependency_lock_sha256 == _dockerfile_requirements_sha(cap)
            assert list(capability.entrypoint) == identity["entrypoint"]
            assert (capability.driver_version is not None) == (cap == "asr_cuda")
        media = resolve_capability(manifest, "media")
        assert media.worker_policy.image == identities["media"]["platforms"][platform]["local_image_id"]
        assert media.worker_policy.development_image and not media.worker_policy.allow_pull
        cpu = resolve_capability(manifest, "asr_cpu")
        assert cpu.worker_policy.image == identities["asr_cpu"]["platforms"][platform]["local_image_id"]
        assert len(cpu.artifacts) == len(MODEL_REPOS)
        measured = {e.artifact.model_id for e in cpu.envelopes}
        if platform == AMD64:
            assert measured == set(document["model_selection"]["measured_envelopes"]["models"])
            assert measured == {"tiny", "base", "small"}
            assert {e.compute_type for e in cpu.envelopes} == {"int8"}
            assert all(e.release_tested and not e.synthetic and e.worker_peak_vram_bytes == 0
                       for e in cpu.envelopes)
            cuda = resolve_capability(manifest, "asr_cuda", selected_gpu_uuid=document["gpu_evidence"]["uuid"])
            assert cuda.worker_policy.image == identities["asr_cuda"]["platforms"][AMD64]["local_image_id"]
            assert cuda.envelopes == ()  # CUDA fit comes from the hardware table
            assert supports_engine_arch(cuda, "x86_64") and not supports_engine_arch(cuda, "aarch64")
        else:
            assert measured == set()  # amd64 measurements never bind to the arm64 build
            with pytest.raises(ManifestError):
                resolve_capability(manifest, "asr_cuda", selected_gpu_uuid=document["gpu_evidence"]["uuid"])
        for resolved in (media, cpu):
            assert supports_engine_arch(resolved, "x86_64") and supports_engine_arch(resolved, "aarch64")


def test_local_build_identities_bind_the_owned_dockerfiles(release):
    document, _ = release
    ids = []
    for cap, identity in document["local_build_identities"].items():
        path, target = DOCKERFILES[cap]
        assert identity["dockerfile"] == path and identity["dockerfile_target"] == target
        text = (REPO / path).read_text("utf-8")
        assert identity["dockerfile_sha256"] == hashlib.sha256((REPO / path).read_bytes()).hexdigest(), (
            cap, "the Dockerfile changed since the release candidate was built; rebuild and regenerate")
        assert identity["build_inputs_sha256"] == _build_inputs_sha(identity["build_inputs"]), (
            cap, "a build input changed since the release candidate was built; rebuild and regenerate")
        assert path in identity["build_inputs"] and "src/podmuncher/__init__.py" in identity["build_inputs"]
        assert re.search(r"^FROM " + re.escape(identity["base_image"]) + r"\b", text, re.MULTILINE)
        assert re.fullmatch(r"python@sha256:[0-9a-f]{64}", identity["base_image"])
        assert identity["user"] == "65532:65532"
        assert "not a registry digest" in identity["identity_note"]
        for platform, local in identity["platforms"].items():
            assert local["compressed_bytes"] > 0 and local["unpacked_bytes"] >= local["compressed_bytes"]
            assert local["local_tag"].endswith("release-local")
            assert local["compressed_bytes_method"] and local["unpacked_bytes_method"]
            assert ("x86_64" if platform == AMD64 else "aarch64") in local["built_on"]
            ids.append(local["local_image_id"])
    assert len(set(ids)) == len(ids) == 5


def test_blockers_are_exactly_the_missing_publication_review_and_evidence(release):
    document, _ = release
    status = document["release_status"]
    assert status["publishable"] is False
    found = [(b["code"], b["capability"]) for b in status["blockers"]]
    assert len(set(found)) == len(found)
    expected = {("publisher_namespace_missing", None), ("nvidia_redistribution_review", "asr_cuda")}
    expected |= {("registry_digest_missing", cap) for cap in CAPABILITIES}
    records = document["benchmarks"]["records"]
    cuda_ok = [r for r in _current(records, "asr_cuda") if r["outcome"] == "completed"]
    assert bool(cuda_ok) == document["gpu_evidence"]["passthrough"]
    if cuda_ok:
        if max(r["fixture_seconds"] for r in cuda_ok) < SIX_HOURS:
            expected.add(("long_duration_unvalidated", "asr_cuda"))
    else:
        expected.add(("gpu_evidence_missing", "asr_cuda"))
    models = document["runtime_manifests"][AMD64]["models"]
    if max(m.get("validated_max_duration_seconds") or 0 for m in models) < SIX_HOURS:
        expected.add(("long_duration_unvalidated", "asr_cpu"))
    for cap in ("asr_cpu", "asr_cuda"):
        if any(r["outcome"].startswith("worker_failure:") for r in _current(records, cap)):
            expected.add(("worker_defect_open", cap))
    assert set(found) == expected
    nvidia = next(b for b in status["blockers"] if b["code"] == "nvidia_redistribution_review")
    assert all(word in nvidia["detail"] for word in ("cuBLAS", "cuDNN", "NVRTC"))
    registry = document["registry"]
    assert registry["publisher_namespace"] is None
    for cap, image in registry["images"].items():
        assert image == {"image_reference": None, "registry_digest": None, "platforms": PLATFORMS[cap]}


def test_no_field_claims_a_registry_digest(release):
    document, _ = release
    for path, key, value in _pairs(document):
        if "registry" in key.lower() and "digest" in key.lower():
            assert value is None, path
        if key in ("RepoDigests", "repo_digests", "image_digest"):
            raise AssertionError("registry digest field " + path)
    for path, value in _strings(document):
        assert "@sha256:" not in value or value.startswith("python@sha256:"), path


def test_no_placeholder_or_negative_domain_strings(release):
    document, _ = release
    for path, value in _strings(document):
        assert value not in NEGATIVE, path
        assert not _PLACEHOLDER.search(value), (path, value)
        assert not _REPEATED_HEX.fullmatch(value), (path, value)
        assert value == value.strip(), path


def test_every_table_model_is_pinned_hashed_sourced_and_licensed(release):
    document, loaded = release
    table = json.loads(HARDWARE_TABLE.read_text("utf-8"))
    table_ids = [row["model_id"] for row in table["models"]]
    assert set(table_ids) == set(MODEL_REPOS) == set(document["model_selection"]["pinned_models"])
    inventory = {m["id"]: m for m in document["provenance_inventory"]["models"]}
    files_by_platform = {p: {m.artifact.model_id: m.files for m in manifest.models}
                         for p, manifest in loaded.items()}
    assert files_by_platform[AMD64] == files_by_platform[ARM64]
    tiers = {row["model_id"]: row["tier"] for row in table["models"]}
    for manifest in loaded.values():
        assert [m.artifact.model_id for m in manifest.models] == document["model_selection"]["pinned_models"]
        for model in manifest.models:
            artifact = model.artifact
            entry = inventory[artifact.model_id]
            assert artifact.tier is ModelTier.parse(tiers[artifact.model_id])
            assert re.fullmatch(r"[0-9a-f]{40}", artifact.revision)
            assert entry["revision"] == artifact.revision and entry["repo"] == MODEL_REPOS[artifact.model_id]
            assert model.license == "MIT" == entry["license"] and "mit" in entry["license_evidence"]
            assert _SHA256.fullmatch(entry["model_card_sha256"])
            assert not model.synthetic
            names = [f.relative_path for f in model.files]
            assert all(MODEL_FILE_PATTERNS.fullmatch(name) for name in names), names
            assert {"config.json", "model.bin", "tokenizer.json"} <= set(names)
            assert any(name.startswith("vocabulary.") for name in names)
            assert set(entry["file_hash_basis"]) == set(names)
            for name, basis in entry["file_hash_basis"].items():
                # Large files are pinned by the Hub LFS sha256; small files were downloaded and hashed.
                assert basis == ("hf_lfs_oid" if name == "model.bin" else "downloaded_bytes"), (name, basis)
            assert artifact.sha256 == next(f.sha256 for f in model.files if f.relative_path == "model.bin")
            weights = next(f for f in model.files if f.relative_path == "model.bin")
            assert weights.bytes > 50 * 1024**2
            for url, declared in zip(model.source_urls, model.files, strict=True):
                parts = urlsplit(url)
                assert parts.scheme == "https" and parts.hostname == "huggingface.co" and not parts.query
                assert parts.path == (f"/{MODEL_REPOS[artifact.model_id]}/resolve/"
                                      f"{artifact.revision}/{declared.relative_path}")
                assert _SHA256.fullmatch(declared.sha256) and declared.bytes > 0
                assert not _REPEATED_HEX.fullmatch(declared.sha256)
            if artifact.model_id == "tiny":
                assert artifact.revision == TINY_REVISION and artifact.sha256 == TINY_WEIGHTS_SHA256


def test_language_registry_matches_the_pinned_models(release):
    document, loaded = release
    inventory = {m["id"]: m for m in document["provenance_inventory"]["models"]}
    per_model = {model_id: entry["languages"] for model_id, entry in inventory.items()}
    longest = max(per_model.values(), key=len)
    for model_id, languages in per_model.items():
        assert languages == longest[:len(languages)], model_id  # tokenizer order, prefix of the full list
        assert inventory[model_id]["multilingual"] is True and len(languages) in (99, 100)
    assert len(longest) == 100 and longest[-1] == "yue"
    common = [code for code in longest if all(code in languages for languages in per_model.values())]
    for manifest in loaded.values():
        assert list(manifest.languages) == common
        assert len(manifest.languages) == len(set(manifest.languages)) == 99
        assert {"en", "de", "fr", "es", "ja", "zh"} <= set(manifest.languages)
        assert all(re.fullmatch(r"[a-z]{2,3}", code) for code in manifest.languages)
        assert "auto" not in manifest.languages
    # The v3 family adds Cantonese; distil-large-v3 is trained for English per its model card.
    assert "yue" in per_model["large-v3"] and "yue" not in per_model["large-v2"]
    assert inventory["distil-large-v3"]["model_card_languages"] == ["en"]


def test_envelopes_and_codec_matrix_match_the_measured_records(release):
    document, _ = release
    records = document["benchmarks"]["records"]
    assert len({r["record_sha256"] for r in records}) == len(records)
    current_ids = {cap: identity["platforms"][AMD64]["local_image_id"]
                   for cap, identity in document["local_build_identities"].items()}
    for record in records:
        # Evidence keeps the image it was measured on; older builds are never relabelled.
        assert record["bound_to_current_build"] == (record["bound_image_id"] == current_ids[record["capability"]])
        assert record["platform"] == AMD64
    fixtures = {(f["codec"], f["seconds"]) for f in document["benchmarks"]["fixtures"]}
    assert {codec for codec, _ in fixtures} == CODECS
    manifest = document["runtime_manifests"][AMD64]
    asr = next(c for c in manifest["capabilities"] if c["id"] == "asr_cpu")
    measured = []
    for model in manifest["models"]:
        mine = [r for r in _current(records, "asr_cpu", model["id"]) if r["memory_cap_bytes"] == 2 * GIB]
        ok = [r for r in mine if r["outcome"] == "completed"]
        if "evidence_sha256" not in model:
            assert not ok, model["id"]  # never drop real evidence
            assert model["codec_matrix"] == sorted(CODECS)
            continue
        measured.append(model["id"])
        expected = hashlib.sha256("\n".join(sorted(r["record_sha256"] for r in mine)).encode("ascii"))
        assert model["evidence_sha256"] == expected.hexdigest()
        assert set(model["codec_matrix"]) == {r["codec"] for r in ok} <= CODECS
        assert model["worker_peak_ram_bytes"] == max(r["peak_rss_bytes"] for r in ok)
        assert model["worker_peak_ram_bytes"] < asr["worker_memory_bytes"]
        failed = [r["fixture_seconds"] for r in mine if r["outcome"].startswith("worker_failure:")]
        below = [r["fixture_seconds"] for r in ok if not failed or r["fixture_seconds"] < min(failed)]
        assert model["validated_max_duration_seconds"] == max(below)
        assert model["validated_max_duration_seconds"] < SIX_HOURS  # never certified from short fixtures
        assert model["worker_peak_vram_bytes"] == 0 and model["compute_types"] == ["int8"]
        assert model["release_tested"] is True
        for record in ok:
            assert (record["codec"], record["fixture_seconds"]) in fixtures
            assert math.isclose(record["real_time_factor"],
                                record["transcribe_wall_seconds"] / record["audio_seconds"], rel_tol=1e-3)
            assert abs(record["audio_seconds"] - record["fixture_seconds"]) < 1.0
            assert record["memory_cap_bytes"] >= record["peak_rss_bytes"]
    assert measured == document["model_selection"]["measured_envelopes"]["models"]
    for model in document["runtime_manifests"][ARM64]["models"]:
        assert "evidence_sha256" not in model  # no arm64 measurements exist
    for record in records:
        assert record["outcome"] == "completed" or record["outcome"].startswith(
            ("worker_failure:", "host_guard_failure:"))
    probes = document["benchmarks"]["probes"]
    for cap in ("asr_cpu", "asr_cuda"):
        probe = probes[cap]
        assert probe["bound_image_id"] == current_ids[cap]
        capability = next(c for c in manifest["capabilities"] if c["id"] == cap)
        assert capability["runtime_version"] == probe["runtime"]["version"]
        assert POLICY_COMPUTE[cap] in probe["compute_types"]
        assert probe["device"] == ("cuda" if cap == "asr_cuda" else "cpu")
    cuda_ok = [r for r in _current(records, "asr_cuda") if r["outcome"] == "completed"]
    for record in cuda_ok:
        assert record["worker_vram_delta_bytes"] > 0
    gpu = document["gpu_evidence"]
    if gpu["passthrough"]:
        assert gpu["bound_image_id"] == current_ids["asr_cuda"]
        assert re.fullmatch(r"GPU-[0-9a-f-]{36}", gpu["uuid"])
        assert "int8_float16" in gpu["container_cuda_compute_types"]
        cuda = next(c for c in manifest["capabilities"] if c["id"] == "asr_cuda")
        assert cuda["driver_version"] == gpu["driver_version"] and cuda["hardware_fingerprint"] == gpu["name"]


def test_provenance_inventory_is_complete(release):
    document, _ = release
    inventory = document["provenance_inventory"]
    assert inventory["own_source"]["license"] == "Apache-2.0"
    assert inventory["own_source"]["license_sha256"] == hashlib.sha256(LICENSE_PATH.read_bytes()).hexdigest()
    lock_text = (REPO / "requirements.lock").read_text("utf-8")
    assert inventory["python_lock"]["sha256"] == hashlib.sha256((REPO / "requirements.lock").read_bytes()).hexdigest()
    pins = set(re.findall(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)", lock_text, re.MULTILINE))
    entries = inventory["python_lock"]["packages"]
    assert {(p["name"], p["version"]) for p in entries} == pins
    assert {p["scope"] for p in entries} == {"runtime", "development"}
    for package in entries:
        _nonempty(package["license"], "lock license")
    runtime = {p["name"] for p in entries if p["scope"] == "runtime"}
    assert {"textual", "aiohttp", "psutil", "platformdirs", "opentelemetry-sdk"} <= runtime
    assert {"pytest", "ruff", "hatchling"}.isdisjoint(runtime)
    for cap in CAPABILITIES:
        images = inventory["container_images"][cap]
        assert set(images) == set(PLATFORMS[cap])
        for image in images.values():
            assert image["python_packages"] and image["debian_packages"]
            for package in image["python_packages"]:
                _nonempty(package["name"], "image package")
                _nonempty(package["version"], "image package version")
                _nonempty(package["license"], "image package license " + package["name"])
            for package in image["debian_packages"]:
                assert package["copyright_sha256"] is None or _SHA256.fullmatch(package["copyright_sha256"])
            names = {p["name"].lower().replace("_", "-") for p in image["python_packages"]}
            if cap == "media":
                assert {"yt-dlp", "yt-dlp-ejs", "deno"} <= names
                assert "ffmpeg" in {p["name"] for p in image["debian_packages"]}
            else:
                assert {"faster-whisper", "ctranslate2", "av", "onnxruntime"} <= names
            if cap == "asr_cuda":
                assert {"nvidia-cublas-cu12", "nvidia-cudnn-cu12", "nvidia-cuda-nvrtc-cu12"} <= names
        # Python package versions are identical across architectures (one hash-pinned list).
        versions = [{(p["name"], p["version"]) for p in image["python_packages"]} for image in images.values()]
        assert all(v == versions[0] for v in versions)
    nvidia = {p["name"] for p in inventory["nvidia_components"]}
    assert nvidia == {"nvidia-cublas-cu12", "nvidia-cudnn-cu12", "nvidia-cuda-nvrtc-cu12"}
    bases = {b["image"] for b in inventory["container_base_images"]}
    assert bases == {identity["base_image"] for identity in document["local_build_identities"].values()}
    artwork = {a["path"]: a for a in inventory["artwork"]}
    for path, item in artwork.items():
        assert item["sha256"] == hashlib.sha256((REPO / path).read_bytes()).hexdigest(), path
        assert item["license"] == "Apache-2.0" and item["origin"].startswith("original")
    assert "src/podmuncher/assets/welcome.txt" in artwork
    assert not any("mascot" in path.lower() or path.lower().endswith((".png", ".svg")) for path in artwork)
    assert any("mascot" in item["name"] for item in inventory["excluded_artwork"])


def test_license_is_the_apache_2_text():
    raw = LICENSE_PATH.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == APACHE_2_SHA256
    text = raw.decode("ascii")
    assert text.lstrip().startswith("Apache License") and "Version 2.0, January 2004" in text
    assert "Copyright [yyyy] [name of copyright owner]" in text  # no invented copyright holder


@pytest.mark.parametrize("bad", NEGATIVE)
@pytest.mark.parametrize("field", ["capability", "tier", "language", "platform", "compute_type", "license"])
def test_negative_domain_values_are_rejected_by_the_loader(tmp_path, field, bad):
    document = json.loads(MANIFEST_PATH.read_text("utf-8"))
    manifest = document["runtime_manifests"][AMD64]
    if field == "capability":
        manifest["capabilities"][1]["id"] = bad
    elif field == "tier":
        manifest["models"][0]["tier"] = bad
    elif field == "language":
        manifest["languages"][0] = bad
    elif field == "platform":
        manifest["platforms"][0] = bad
    elif field == "compute_type":
        manifest["models"][0]["compute_types"][0] = bad
    else:
        manifest["capabilities"][1]["licenses"][0] = bad
    # The loader rejects every case except a development-mode licence, which the wrapper rejects.
    expected = ReleaseManifestError if (field == "license" and bad) else ManifestError
    with pytest.raises(expected):
        load_release_manifest(_write(tmp_path, _rehash(document)))


@pytest.mark.parametrize("bad", NEGATIVE)
@pytest.mark.parametrize("field", ["blocker_code", "blocker_capability", "registry_platform", "domain_parse"])
def test_negative_domain_values_are_rejected_by_the_wrapper(tmp_path, field, bad):
    if field == "domain_parse":
        for parse in (Capability.parse, ModelTier.parse):
            with pytest.raises(DomainValidationError):
                parse(bad)
        return
    document = json.loads(MANIFEST_PATH.read_text("utf-8"))
    if field == "registry_platform":
        document["registry"]["images"]["media"]["platforms"][0] = bad
    else:
        document["release_status"]["blockers"][0]["code" if field == "blocker_code" else "capability"] = bad
    with pytest.raises(ReleaseManifestError):
        load_release_manifest(_write(tmp_path, document))


@pytest.mark.parametrize("mutation", [
    "drop_blockers", "registry_digest_without_reference", "publishable", "stale_hash", "extra_key",
    "registry_tag", "duplicate_key", "release_mode_digest", "partial_envelope", "capability_platform_outside_root",
    "unknown_platform", "missing_arm64_identity",
])
def test_tampered_release_documents_are_rejected(tmp_path, mutation):
    raw = MANIFEST_PATH.read_text("utf-8")
    document = json.loads(raw)
    expected: type[Exception] = ReleaseManifestError
    amd64 = document["runtime_manifests"][AMD64]
    if mutation == "drop_blockers":
        document["release_status"]["blockers"] = []
        document["release_status"]["publishable"] = True
        load_release_manifest(_write(tmp_path, document))  # structurally valid, so the content test must catch it
        with pytest.raises(AssertionError):
            test_blockers_are_exactly_the_missing_publication_review_and_evidence((document, None))
        return
    if mutation == "registry_digest_without_reference":
        document["registry"]["images"]["media"]["registry_digest"] = (
            document["local_build_identities"]["media"]["platforms"][AMD64]["local_image_id"])
    elif mutation == "publishable":
        document["release_status"]["publishable"] = True
    elif mutation == "stale_hash":
        next(m for m in amd64["models"] if "evidence_sha256" in m)["worker_peak_ram_bytes"] += 1
    elif mutation == "extra_key":
        amd64["release_status"] = {}
        _rehash(document)
        expected = ManifestError
    elif mutation == "registry_tag":
        document["local_build_identities"]["media"]["platforms"][AMD64]["local_tag"] = "ghcr.io/someone/pm:1"
    elif mutation == "release_mode_digest":
        amd64["capabilities"][0]["image_reference"] = "ghcr.io/x/podmuncher-media"
        _rehash(document)
        expected = ManifestError
    elif mutation == "partial_envelope":
        del next(m for m in amd64["models"] if "evidence_sha256" in m)["decode_timeout_seconds"]
        _rehash(document)
        expected = ManifestError
    elif mutation == "capability_platform_outside_root":
        cuda = next(c for c in amd64["capabilities"] if c["id"] == "asr_cuda")
        cuda["platforms"] = [AMD64, ARM64, "linux-riscv64"]
        _rehash(document)
        expected = ManifestError
    elif mutation == "unknown_platform":
        document["runtime_manifests"]["linux-riscv64"] = amd64
        _rehash(document)
    elif mutation == "missing_arm64_identity":
        del document["local_build_identities"]["asr_cpu"]["platforms"][ARM64]
    if mutation == "duplicate_key":
        path = tmp_path / "release.json"
        path.write_text(raw.replace('"document": ', '"document": "x", "document": ', 1), encoding="utf-8")
    else:
        path = _write(tmp_path, document)
    with pytest.raises(expected):
        load_release_manifest(path)


def test_release_json_ships_in_the_built_wheel_and_loads_when_installed(tmp_path):
    result = subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--no-isolation", "--outdir", str(tmp_path / "dist"), str(REPO)],
        capture_output=True, text=True, timeout=180, check=False)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    wheel, = (tmp_path / "dist").glob("podmuncher-*.whl")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        assert "podmuncher/manifests/release.json" in names
        assert "podmuncher/manifests/__init__.py" not in names
        assert archive.read("podmuncher/manifests/release.json") == MANIFEST_PATH.read_bytes()
        site = tmp_path / "site"
        archive.extractall(site)
    probe = ("import hashlib, importlib.resources as r, podmuncher; "
             "p = r.files('podmuncher').joinpath('manifests', 'release.json'); "
             "print(hashlib.sha256(p.read_bytes()).hexdigest(), podmuncher.__file__)")
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    env["PYTHONPATH"] = str(site)
    shown = subprocess.run([sys.executable, "-c", probe], cwd=tmp_path, env=env, capture_output=True,
                           text=True, timeout=60, check=False)
    assert shown.returncode == 0, shown.stderr[-2000:]
    digest, module = shown.stdout.split()
    assert digest == hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    assert Path(module).resolve().is_relative_to(site.resolve())


# ------------------------------------------------------------- deterministic: finalized state


def _gates():
    """test_release_gates imports this module at load time, so it is imported lazily here."""
    import test_release_gates

    return test_release_gates


def _finalized(**kwargs) -> tuple[dict, dict]:
    """A finalized copy of release.json made by the P6-U6 finalize function with fake test-data digests."""
    gates = _gates()
    inputs = gates.make_release_inputs(**kwargs)
    return inputs, gates._finalize(inputs)


def test_blocker_classification_matches_the_release_gates():
    gates = _gates()
    assert ADVISORY_CODES == gates.ADVISORY_CODES
    assert CAPABILITY_SCOPED_CRITICAL == gates.CAPABILITY_SCOPED_CRITICAL
    assert REQUIRED_CAPABILITIES == gates.REQUIRED_CAPABILITIES
    assert (REGISTRY, MODEL_ORIGINS) == (gates.REGISTRY, gates.MODEL_ORIGINS)
    assert (_NAMESPACE.pattern, _DIGEST.pattern) == (gates._NAMESPACE.pattern, gates._DIGEST.pattern)
    subsets = [set(c) for n in range(len(CAPABILITIES) + 1) for c in itertools.combinations(CAPABILITIES, n)]
    for code in sorted(BLOCKER_CODES | {"some_future_code"}):
        for capability in (None, *CAPABILITIES):
            for shipped in subsets:
                blocker = {"code": code, "capability": capability}
                assert is_release_critical(blocker, shipped) == gates.is_release_critical(blocker, shipped), (
                    code, capability, shipped)
    assert not is_release_critical({"code": "long_duration_unvalidated", "capability": "asr_cpu"}, set())
    assert is_release_critical({"code": "some_future_code", "capability": None}, set())  # unknown: critical


def test_finalized_copy_passes_the_p6u5_checks(tmp_path):
    inputs, result = _finalized()
    gates = _gates()
    document, loaded = load_release_manifest(_write(tmp_path, result))
    assert release_state(document) == FINALIZED and shipped_capabilities(document) == set(CAPABILITIES)
    status = document["release_status"]
    assert status["publishable"] is True and status["blockers"]
    assert {b["code"] for b in status["blockers"]} == {"long_duration_unvalidated"}  # advisory only
    for cap, image in document["registry"]["images"].items():
        record = inputs["published"][cap]
        assert image["registry_digest"] == record["digest"]
        assert record["digest"] in {gates.fake_digest(cap, "index"), gates.fake_digest(cap, AMD64)}  # test data
    for platform, manifest in loaded.items():
        assert manifest.development_fixture is False and manifest.release_version == inputs["version"]
        assert manifest.sha256 == document["runtime_manifest_sha256"][platform]
        for capability in manifest.capabilities:
            image = document["registry"]["images"][capability.capability.value]
            assert (capability.image_reference, capability.digest) == (image["image_reference"],
                                                                      image["registry_digest"])
        media = resolve_capability(manifest, "media")
        assert media.worker_policy.image == "{image_reference}@{registry_digest}".format(
            **document["registry"]["images"]["media"])
        assert media.worker_policy.allow_pull and not media.worker_policy.development_image
    # The state-independent P6-U5 content checks hold for the finalized copy too.
    for check in (test_local_build_identities_bind_the_owned_dockerfiles, test_no_placeholder_or_negative_domain_strings,
                  test_every_table_model_is_pinned_hashed_sourced_and_licensed,
                  test_language_registry_matches_the_pinned_models,
                  test_envelopes_and_codec_matrix_match_the_measured_records, test_provenance_inventory_is_complete):
        check((document, loaded))
    with pytest.raises(AssertionError):
        test_no_field_claims_a_registry_digest((document, loaded))  # local-state check: digests now exist
    assert set(gates.check_finalized(result)) == set(loaded)  # both checkers accept it

    # A release without CUDA (NVIDIA review open) and one missing a required image are also well-formed.
    _, partial = _finalized(publish=REQUIRED_CAPABILITIES, nvidia_reviewed=False)
    document, loaded = load_release_manifest(_write(tmp_path, partial))
    assert document["release_status"]["publishable"] is True
    assert {c.capability.value for c in loaded[AMD64].capabilities} == {"media", "asr_cpu"}
    _, missing = _finalized(publish=("asr_cpu", "asr_cuda"))
    document, loaded = load_release_manifest(_write(tmp_path, missing))
    assert document["release_status"]["publishable"] is False  # media is required: its digest blocker is critical
    assert all("media" not in {c.capability.value for c in m.capabilities} for m in loaded.values())


@pytest.mark.parametrize("case", ["open_worker_defect", "missing_required_image", "unknown_code_kept"])
def test_finalized_copy_with_a_critical_blocker_and_publishable_is_rejected(tmp_path, case):
    if case == "missing_required_image":
        _, result = _finalized(publish=("asr_cpu", "asr_cuda"))
    else:
        _, result = _finalized()
    if case == "open_worker_defect":
        result["release_status"]["blockers"].append({
            "code": "worker_defect_open", "capability": "asr_cpu", "resolved_by": "worker fix",
            "detail": "P6-U5 deterministic test data: a reopened worker defect"})
    elif case == "unknown_code_kept":
        result["release_status"]["blockers"].append({
            "code": "some_future_code", "capability": None, "resolved_by": "owner",
            "detail": "P6-U5 deterministic test data: an unknown blocker code"})
    result["release_status"]["publishable"] = True
    with pytest.raises(ReleaseManifestError):
        load_release_manifest(_write(tmp_path, result))
    if case != "unknown_code_kept":  # the gates checker agrees (it does not validate codes structurally)
        with pytest.raises(_gates().GateError):
            _gates().check_finalized(result)


def test_advisory_only_blockers_with_publishable_are_accepted(tmp_path):
    _, result = _finalized()
    assert result["release_status"]["publishable"] is True
    assert {b["code"] for b in result["release_status"]["blockers"]} == ADVISORY_CODES
    load_release_manifest(_write(tmp_path, result))
    result["release_status"]["publishable"] = False  # a finalized manifest must state the computed value
    with pytest.raises(ReleaseManifestError, match="disagrees"):
        load_release_manifest(_write(tmp_path, result))
    # The same rule holds for the local state: advisory blockers never block, critical ones do.
    local = json.loads(MANIFEST_PATH.read_text("utf-8"))
    advisory = [b for b in local["release_status"]["blockers"] if b["code"] in ADVISORY_CODES]
    assert advisory
    local["release_status"] = {"publishable": True, "blockers": advisory}
    document, _ = load_release_manifest(_write(tmp_path, local))
    assert release_state(document) == LOCAL
    local["release_status"]["blockers"].append(
        next(b for b in json.loads(MANIFEST_PATH.read_text("utf-8"))["release_status"]["blockers"]
             if b["code"] == "registry_digest_missing"))
    with pytest.raises(ReleaseManifestError, match="release-critical"):
        load_release_manifest(_write(tmp_path, local))


@pytest.mark.parametrize("mutation", [
    "malformed_digest", "foreign_reference", "digest_blocker_for_published_image", "namespace_blocker",
    "no_namespace", "runtime_digest", "dropped_runtime_capability", "mixed_kinds", "reference_without_digest",
    "nvidia_open_with_cuda",
])
def test_tampered_finalized_copies_are_rejected(tmp_path, mutation):
    _, result = _finalized()
    images = result["registry"]["images"]
    amd64 = result["runtime_manifests"][AMD64]
    blocker = {"capability": None, "resolved_by": "P6-U6 release workflow", "detail": "P6-U5 deterministic test data"}
    expected: type[Exception] = ReleaseManifestError
    if mutation == "malformed_digest":
        images["media"]["registry_digest"] = "sha256:" + "ab" * 16
    elif mutation == "foreign_reference":
        images["media"]["image_reference"] = "ghcr.io/someone-else/podmuncher-media"
    elif mutation == "digest_blocker_for_published_image":
        result["release_status"]["blockers"].append({**blocker, "code": "registry_digest_missing", "capability": "media"})
        result["release_status"]["publishable"] = False
    elif mutation == "namespace_blocker":
        result["release_status"]["blockers"].append({**blocker, "code": "publisher_namespace_missing"})
        result["release_status"]["publishable"] = False
    elif mutation == "no_namespace":
        result["registry"]["publisher_namespace"] = None
    elif mutation == "runtime_digest":
        amd64["capabilities"][0]["digest"] = _gates().fake_digest("tampered")
        _rehash(result)
    elif mutation == "dropped_runtime_capability":
        amd64["capabilities"] = [c for c in amd64["capabilities"] if c["id"] != "media"]
        _rehash(result)
    elif mutation == "mixed_kinds":
        result["runtime_manifests"][ARM64]["source"] = {"kind": "development_fixture", "label": "fixture",
                                                        "provenance": "P6-U5 deterministic test data"}
        _rehash(result)  # the document is then local, and a local document carries no registry digest
    elif mutation == "reference_without_digest":
        images["asr_cuda"]["registry_digest"] = None
    else:
        result["release_status"]["blockers"].append({**blocker, "code": "nvidia_redistribution_review",
                                                     "capability": "asr_cuda"})
        result["release_status"]["publishable"] = False
    with pytest.raises(expected):
        load_release_manifest(_write(tmp_path, result))


def _fake_registry_manifest(cap: str, platforms: list[str]) -> bytes:
    """Fake registry manifest bytes (P6-U6 TEST_DATA_LABEL): test data, never a published manifest."""
    gates = _gates()
    label = {"org.opencontainers.image.description": gates.TEST_DATA_LABEL}
    if cap == "asr_cuda":
        value = {"schemaVersion": 2, "mediaType": "application/vnd.oci.image.manifest.v1+json",
                 "config": {"mediaType": "application/vnd.oci.image.config.v1+json",
                            "digest": gates.fake_digest(cap, "config"), "size": 1},
                 "layers": [], "annotations": label}
    else:
        entries = [{"mediaType": "application/vnd.oci.image.manifest.v1+json", "digest": gates.fake_digest(cap, p),
                    "size": 1, "platform": {"os": "linux", "architecture": p.split("-")[1]}} for p in platforms]
        entries.append({"mediaType": "application/vnd.oci.image.manifest.v1+json",
                        "digest": gates.fake_digest(cap, "attestation"), "size": 1,
                        "platform": {"os": "unknown", "architecture": "unknown"},
                        "annotations": {"vnd.docker.reference.type": "attestation-manifest"}})
        value = {"schemaVersion": 2, "mediaType": "application/vnd.oci.image.index.v1+json", "manifests": entries,
                 "annotations": label}
    return canonical(value)


def _finalized_with_registry(tmp_path: Path, index_platforms: dict[str, list[str]] | None = None
                             ) -> tuple[dict, dict[str, bytes]]:
    """Finalize with digests that are the sha256 of fake manifest bytes, so the registry branch can verify them."""
    gates = _gates()
    document = json.loads(MANIFEST_PATH.read_text("utf-8"))
    published, raws = {}, {}
    for cap in CAPABILITIES:
        platforms = list(PLATFORMS[cap])
        raw = _fake_registry_manifest(cap, (index_platforms or {}).get(cap, platforms))
        digest = "sha256:" + hashlib.sha256(raw).hexdigest()
        reference = gates.image_reference(gates.TEST_NAMESPACE, cap)
        platform_digests = ({AMD64: digest} if cap == "asr_cuda"
                            else {p: gates.fake_digest(cap, p) for p in platforms})
        published[cap] = {"capability": cap, "image_reference": reference, "digest": digest,
                          "platforms": platforms, "platform_digests": platform_digests}
        raws[f"{reference}@{digest}"] = raw
    result = gates.finalize(document, namespace=gates.TEST_NAMESPACE, version=document["release_version"],
                            published=published, nvidia_reviewed=True, finalized_at="2026-10-14T00:00:00Z")
    loaded_document, _ = load_release_manifest(_write(tmp_path, result))
    return loaded_document, raws


def test_finalized_live_branch_checks_registry_manifests_read_only(tmp_path, monkeypatch):
    module = sys.modules[__name__]
    monkeypatch.setattr(module, "_docker", lambda *a, **k: pytest.fail("the finalized branch ran docker: " + str(a)))
    document, raws = _finalized_with_registry(tmp_path)
    seen = []

    def fake_inspect(reference):
        seen.append(reference)
        return raws[reference]

    amd64 = compare_live_assets(document, AMD64, inspect=fake_inspect)
    assert set(amd64) == set(CAPABILITIES)
    assert {cap: v["kind"] for cap, v in amd64.items()} == {"media": "index", "asr_cpu": "index", "asr_cuda": "image"}
    arm64 = compare_live_assets(document, ARM64, inspect=fake_inspect)
    assert set(arm64) == {"media", "asr_cpu"}
    assert set(seen) == set(raws)  # every read is a recorded image_reference@digest
    with pytest.raises(AssertionError, match="hash"):
        compare_live_assets(document, AMD64, inspect=lambda reference: raws[reference] + b" ")
    short, short_raws = _finalized_with_registry(tmp_path, {"media": [AMD64]})  # index lacks linux-arm64
    with pytest.raises(AssertionError):
        compare_live_assets(short, AMD64, inspect=short_raws.__getitem__)

    calls = []

    def fake_docker(*arguments, timeout=120):
        calls.append(arguments)
        return subprocess.CompletedProcess(arguments, 0, b"{}", b"")

    monkeypatch.setattr(module, "_docker", fake_docker)
    reference = next(iter(raws))
    assert inspect_registry_raw(reference) == b"{}"
    assert calls == [("buildx", "imagetools", "inspect", "--raw", reference)]
    assert not {"login", "push", "tag", "pull"} & {word for call in calls for word in call}


# --------------------------------------------------------------------------------- live


class _Audit:
    def record(self, **_values):
        return None


class _Lease:
    def __init__(self):
        self.held = False
        self.released = 0

    def try_acquire(self, owner, plan):
        assert plan.admitted
        if self.held:
            return False
        self.held = True
        return True

    def release(self, owner, *, writers_reaped):
        assert writers_reaped and self.held
        self.held = False
        self.released += 1


def _probe_request(capability: str):
    from podmuncher.config import Limits
    from podmuncher.protocol import WorkerRequest

    return WorkerRequest.from_dict({
        "schema_version": 1, "job_id": str(uuid4()), "attempt_id": str(uuid4()), "stage_id": str(uuid4()),
        "capability": capability, "operation": "probe", "assigned_slots": {"probe_metadata": "probe.json"},
        "source": None, "outputs": {"transcript": True, "video": False, "video_quality": "source_best"},
        "input_basename": None, "model_snapshot_id": None, "limits": asdict(Limits()),
        "selection": {"device": "auto", "profile": "balanced", "model": "auto", "language": "auto"}})


def replay_probe(capability, image_id: str, work: Path, *, gpu_uuid: str | None) -> dict:
    """One bounded probe through the owned DockerRuntime (network none, no input, no model)."""
    from podmuncher.admission import WorkloadGrowth
    from podmuncher.config import Limits
    from podmuncher.docker_storage import make_docker_storage_provider
    from podmuncher.guards import CancellationCoordinator, WriteGuard
    from podmuncher.runtime import (
        DockerRuntime,
        DockerTarget,
        NativeDockerCLI,
        RuntimeStorage,
        WorkerMounts,
        WorkerPolicy,
    )

    context = json.loads(subprocess.run((DOCKER, "context", "inspect"), capture_output=True, text=True,
                                        timeout=30, check=False).stdout)[0]
    target = DockerTarget(context["Name"], _docker_endpoint())

    def proof():
        shown = _docker("image", "inspect", "--format", "{{.Id}}", image_id, timeout=30)
        return shown.returncode == 0 and shown.stdout.decode().strip() == image_id

    policy = WorkerPolicy(capability, image_id, ("python", "-I", "-m", "podmuncher.asr_worker"), 0, 0, proof,
                          development_image=True, allow_pull=False, memory_bytes=2 * GIB, pids=256,
                          selected_gpu_uuid=gpu_uuid)
    scratch = work / "scratch"
    paths = {role: work / role for role in ("cache", "output", "state")}
    for path in (scratch, *paths.values()):
        path.mkdir(parents=True)
    provider = make_docker_storage_provider(timeout=60)
    session = provider.prepare(manual_backing="")
    cli = NativeDockerCLI(target)
    instance = "release-probe-" + uuid4().hex
    try:
        async def run():
            growth = WorkloadGrowth(0, 0, 0, 0, (0,), 0, 0, 0, 0, (1024**2,), 0, 0)
            storage = RuntimeStorage(session, provider.volumes, paths={**paths, "scratch": scratch}, growth=growth,
                                     limits=Limits(), sample_budget_seconds=2.5)
            lease = _Lease()
            coordinator = CancellationCoordinator(instance, lease, stop_grace_seconds=12, reap_timeout_seconds=12,
                                                  callback_timeout_seconds=12)
            guard = WriteGuard(coordinator, storage.sample, byte_limit=1024**2, timeout_seconds=120,
                               sample_timeout_seconds=2.9)
            runtime = DockerRuntime(cli, guard, instance=instance, policy=policy, mounts=WorkerMounts(scratch))
            result = await runtime.run(_probe_request(str(capability)))
            assert result.completed and result.exit_code == 0 and lease.released == 1
            listed = await cli.run("ps", "-a", "--no-trunc", "--format", "{{.ID}}", "--filter",
                                   "label=podmuncher.instance=" + instance, owner="release-verify-" + uuid4().hex)
            assert listed.returncode == 0 and not listed.stdout.split(), "probe container left behind"
            return result

        result = asyncio.run(run())
    finally:
        session.close()
    claims = [claim.payload for claim in result.artifact_claims]
    assert [claim["slot"] for claim in claims] == ["probe_metadata"]
    data = (scratch / "probe.json").read_bytes()
    assert claims[0]["sha256"] == hashlib.sha256(data).hexdigest()
    return json.loads(data)


def verify_models(manifest: RuntimeManifest, model_root: Path) -> tuple[dict[str, str], list[str]]:
    """Re-verify the pinned model files: tiny always (acquired if absent), others only when cached."""
    from podmuncher.model_cache import ModelCache

    verified, deterministic_only = {}, []
    cache = ModelCache(model_root, audit=_Audit(), timeout_seconds=600,
                       admission=lambda **v: v["declared_bytes"] <= 600 * 1024**2
                       and v["allocated_bytes"] <= 600 * 1024**2)
    try:
        cached = {snapshot.files for snapshot in cache.snapshots()}
        for model in manifest.models:
            if model.artifact.model_id != "tiny" and model.files not in cached:
                deterministic_only.append(model.artifact.model_id)
                continue
            snapshot = cache.acquire(model)  # cached snapshots are re-verified, an absent tiny is acquired
            for declared in model.files:
                path = snapshot.path / PurePosixPath(declared.relative_path)
                digest = hashlib.sha256()
                with path.open("rb") as stream:
                    while chunk := stream.read(8 * 1024 * 1024):
                        digest.update(chunk)
                assert path.stat().st_size == declared.bytes
                assert digest.hexdigest() == declared.sha256
            verified[model.artifact.model_id] = snapshot.snapshot_id
    finally:
        cache.close()
    return verified, deterministic_only


@pytest.mark.live
def test_release_manifest_live_assets(tmp_path):
    began = time.monotonic()
    manifest_value = os.environ.get("PODMUNCHER_RELEASE_MANIFEST")
    model_root_value = os.environ.get("PODMUNCHER_TEST_MODEL_ROOT")
    assert manifest_value, ("PODMUNCHER_RELEASE_MANIFEST must name the release manifest to verify "
                            "(src/podmuncher/manifests/release.json); the live smoke never skips")
    assert Path(manifest_value).is_file(), "PODMUNCHER_RELEASE_MANIFEST does not name an existing file"
    assert model_root_value, ("PODMUNCHER_TEST_MODEL_ROOT (a ModelCache root outside the repository) must be set "
                              "to verify the pinned model files")
    model_root = Path(model_root_value).resolve()
    assert REPO not in (model_root, *model_root.parents), "The model root must be outside the repository"
    document, loaded = load_release_manifest(manifest_value)

    state = release_state(document)
    platform = engine_platform()
    manifest = loaded[platform]
    observed = compare_live_assets(document, platform)  # local: IDs and sizes; finalized: registry manifests
    built = {cap for cap in CAPABILITIES if platform in PLATFORMS[cap]}
    assert set(observed) == (built if state == LOCAL else built & shipped_capabilities(document))

    verified, deterministic_only = verify_models(manifest, model_root)
    assert "tiny" in verified

    probes = {}
    work = tmp_path / "probes"
    try:
        for capability in manifest.capabilities:
            cap = capability.capability.value
            if cap == "media" or state == FINALIZED:
                continue  # finalized: the workflow's pull-verify step probes the published images
            gpu_uuid = None
            if cap == "asr_cuda":
                evidence = document["gpu_evidence"]
                if not evidence["passthrough"]:
                    continue  # no CUDA evidence: the gpu_evidence_missing blocker stands, no GPU probe
                gpu_uuid = evidence["uuid"]
            probe = replay_probe(cap, capability.digest, work / cap, gpu_uuid=gpu_uuid)
            assert probe["runtime"]["name"] == "ctranslate2"
            assert probe["runtime"]["version"] == capability.runtime_version
            assert POLICY_COMPUTE[cap] in probe["compute_types"], (probe["compute_types"], cap)
            envelope_types = {t for m in manifest.models if m.measured and m.worker_peak_vram_bytes == 0
                              for t in m.compute_types}
            if cap == "asr_cpu":
                assert envelope_types <= set(probe["compute_types"])
            assert probe["device"] == ("cuda" if cap == "asr_cuda" else "cpu")
            recorded = document["benchmarks"]["probes"][cap]
            assert probe["compute_types"] == recorded["compute_types"]
            probes[cap] = probe
    finally:
        shutil.rmtree(work, ignore_errors=True)
    assert not work.exists()
    elapsed = time.monotonic() - began
    assert elapsed < 600, "Live smoke exceeded its declared 600 s bound"
    print(f"platform {platform} state {state}")
    for cap, values in observed.items():
        if state == LOCAL:
            print(f"image {cap} id={values['id'][7:19]} unpacked={values['unpacked']} "
                  f"compressed={values['compressed']}")
        else:
            print(f"registry {cap} digest={values['id'][7:19]} {values['kind']} platforms={','.join(values['platforms'])}")
    print("models verified " + " ".join(f"{k}={v}" for k, v in sorted(verified.items())))
    print("models shape-checked only (not cached) " + " ".join(deterministic_only))
    for cap, probe in probes.items():
        print(f"probe {cap} runtime={probe['runtime']['version']} device={probe['device']} "
              f"compute_types={','.join(probe['compute_types'])}")
    blockers = ", ".join(sorted(f"{b['code']}:{b['capability']}" for b in document["release_status"]["blockers"]))
    print(f"release publishable={document['release_status']['publishable']} blockers=[{blockers}] "
          f"elapsed={elapsed:.1f}s")
    print(SMOKE_MARKER)
