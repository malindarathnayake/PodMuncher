"""Product composition: the service factory and the client-startup binding.

Phase 6 unit P6-U2 (Docs/spec.md:1624).  Two entry points:

* ``make_service(state_dir)`` is the factory that ``service.DEFAULT_FACTORY`` names.  The
  serving process calls it after taking the singleton lock.  ``startup()`` then builds, in
  this order: the state-volume preflight, the configuration and runtime manifest, the
  repository, the telemetry hub, the model cache, startup reconciliation, the scheduler, its
  pump task and the API actions.  A failure before readiness closes whatever was opened and
  raises a closed ``ServiceError``; no traceback, path or secret leaves this module.
* ``client_startup()`` binds the final CLI presentation adapter and the verified registry
  provider.  It never starts, contacts or discovers the daemon.

Runtime manifest policy (SPEC_GAP-11, SPEC_GAP-14): an explicit development fixture named
by ``PODMUNCHER_RUNTIME_MANIFEST`` always wins and is loaded with ``development=True``.
Without it the packaged release document (``podmuncher/manifests/release.json``, P6-U5) is
read: when it is publishable, the runtime manifest for the Docker engine's architecture is
loaded through the real loader in release mode (``image_reference@digest``).  A release that
is not publishable, an invalid wrapper, an unknown engine or a platform without images gives
no manifest: the registries stay ``None`` (auto-only), every image-backed capability is not
ready with a closed reason, nothing is pulled, and setup guidance names the blocker codes
(``ManifestResolution.next_action``).  Resolution happens once, at startup, right after the
engine architecture is sampled through the hardware port.

Interface bridges (one section each below):

* API route names are dotted (``jobs.submit``); audit actions are ``[a-z][a-z0-9_]*``.  Dots
  become underscores; ``jobs.cancel`` and ``service.stop`` map to the exempt ``cancel`` and
  ``service_stop`` so that an audit-loss latch still lets them through.  New work (submit,
  retry, clone, setup apply) writes an intent audit first and is refused on audit loss.
* ``Scheduler.retry`` runs a whole attempt inline, so the action checks the durable replay,
  starts the retry as a background task and returns the attempt id once it is committed.
* ``artifacts.write_manifest`` takes ``PublishedArtifact`` values; the publisher keeps them
  per artifact id (or re-proves them from the existing sidecar) and derives the job folder.
* There is no snapshot builder or artifact reader elsewhere: both live here.  Unmeasured
  values are null with a reason, never zero.  Artifact reads are contained, no-follow,
  bounded and re-hashed.
* ``setup.SetupService.apply`` is synchronous: it runs in a worker thread under
  ``Scheduler.run_setup`` (the single execution lease), with bounded plan/operation maps.
* Automatic model selection needs same-worker ``PolicyEvidence`` (SPEC_GAP-12).  It comes
  from a short-lived ASR **probe** worker: ``_ProbeEvidence`` runs the image's ``probe``
  operation, reads its runtime facts from the private scratch slot, samples the held
  container with ``hardware.HardwareSampler`` and hands fresh evidence to
  ``model_policy.select_model`` inside the hold window.  ``ModelProvider.select`` is
  synchronous on the loop, so it only records a deferred choice; the async ``prepare``
  probes, selects and acquires.  Setup ``choose_model`` and ``smoke`` use the same port.

Blocking work (fsync, Docker, filesystem hashing, network) never runs on the API loop from
an action: it goes through ``asyncio.to_thread``.  Telemetry carries ids and reason codes only.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .domains import (
    ArtifactFormat,
    ArtifactSlot,
    Capability,
    DomainValidationError,
    FieldStatus,
    JobState,
    ModelTier,
    ReasonCode,
    ServiceState,
    Stage,
    WorkerSlot,
)

_LOG = logging.getLogger("podmuncher.bootstrap")

MANIFEST_ENV = "PODMUNCHER_RUNTIME_MANIFEST"
CONFIG_ENV = "PODMUNCHER_CONFIG"
DATABASE_NAME = "podmuncher.db"
VERIFIED_DIRNAME = "verified"
MIB = 1024 * 1024
# The packaged release document (P6-U5 shape, finalized by the P6-U6 release workflow).
RELEASE_RESOURCE = ("manifests", "release.json")
RELEASE_DOCUMENT = "podmuncher-release-manifest"
RELEASE_DOCUMENT_VERSION = 2
MAX_RELEASE_DOCUMENT_BYTES = 4 * MIB
MAX_RELEASE_PLATFORMS = 8
MAX_RELEASE_BLOCKERS = 64
NAMED_BLOCKERS = 4  # blocker codes named in operator guidance
# The only model asset origin a packaged release may pin (the release gates load the
# finalized runtime manifests with the same allowlist).
PACKAGED_ASSET_ORIGINS = ("https://huggingface.co",)
_BLOCKER_CODE_RE = re.compile(r"[a-z][a-z0-9_]{0,63}\Z")
_RELEASE_PLATFORM_RE = re.compile(r"linux-(?:amd64|arm64)\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
CHUNK_BYTES = 64 * 1024
MAX_RECORD_BYTES = 1024 * 1024
MAX_PROBE_BYTES = 64 * 1024
PUMP_IDLE_SECONDS = 0.5
RETRY_COMMIT_SECONDS = 5.0
SNAPSHOT_REUSE_SECONDS = 0.5
DOCKER_REFRESH_SECONDS = 5.0
DOCKER_FIRST_WAIT_SECONDS = 1.5
PLAN_TTL_SECONDS = 900.0
_ADMISSION_RESAMPLE_BYTES = 8 * MIB  # model acquisition: growth between two host plans
_ADMISSION_RESAMPLE_SECONDS = 1.0
MAX_RETAINED_IDENTITIES = 64  # per job: stage identities -> the attempt holding their outputs
MAX_PLANS = 32
MAX_OPERATIONS = 64
MAX_KEYS = 256
SHUTDOWN_TASK_SECONDS = 10.0
# Probe evidence port (SPEC_GAP-12): the worker's probe artifact cap, the wait for it, the
# whole single-use run, the in-worker sample bound and the freshness we re-sample below.
PROBE_ARTIFACT_BYTES = 4096
PROBE_ARTIFACT_WAIT_SECONDS = 45.0
PROBE_TIMEOUT_SECONDS = 90.0
PROBE_END_SECONDS = 30.0
PROBE_SAMPLE_SECONDS = 10.0
# Storage watchdog budget for every worker run (probe, smoke and stage workers), as the live
# runtime/ASR smokes budget it: a real sample measures 0.7-1.2 s on this host, above runtime's
# 0.95 s default, so the default latched storage_unknown.  At 1.45 s, 38% of idle samples on
# this host still ran over (slowest 1.83 s); none of 60 exceeded 2.5 s.  Stays within
# RuntimeStorage's 3 s cap.
WORKER_STORAGE_SAMPLE_SECONDS = 2.5
EVIDENCE_MAX_AGE_SECONDS = 3.0  # model_policy.select_model's own freshness bound
EVIDENCE_FRESH_SECONDS = 2.75  # older than this at return: one re-sample while still held
SETUP_PROBE_SECONDS = 180.0
SMOKE_FIXTURE_SECONDS = 2.0
# Setup PULL of a release image: the configured download timeout, capped here, bounds the
# guarded pull; the loop wait allows the guard its finish (stop grace, reaping) on top.
SETUP_PULL_TIMEOUT_SECONDS = 3600.0
SETUP_PULL_GRACE_SECONDS = 30.0
IMAGE_INSPECT_SECONDS = 30.0
_STAMP = "%Y-%m-%dT%H:%M:%S.%fZ"
_COMPUTE_TYPE_RE = re.compile(r"[a-z0-9_]{1,32}\Z")
_VERSION_RE = re.compile(r"[0-9A-Za-z][0-9A-Za-z.+_-]{0,63}\Z")
_GPU_UUID_RE = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")
# Probe-run failures that keep their own closed reason (storage, Docker, cancellation,
# unresolved writers); every other probe failure is ``envelope_unvalidated``.
_PASS_THROUGH_REASONS = frozenset({
    ReasonCode.DOCKER_UNAVAILABLE, ReasonCode.DOCKER_CONTEXT_UNSUPPORTED,
    ReasonCode.SERVICE_UNAVAILABLE, ReasonCode.STORAGE_INSUFFICIENT, ReasonCode.STORAGE_UNKNOWN,
    ReasonCode.STORAGE_STALE, ReasonCode.STORAGE_GUARD_TRIPPED, ReasonCode.QUOTA_EXCEEDED,
    ReasonCode.BACKING_UNRESOLVED, ReasonCode.CANCELLED, ReasonCode.WRITER_UNRESOLVED,
    ReasonCode.REQUEST_TIMEOUT,
})
# Reasons that never fall back from a CUDA candidate to the CPU one on ``device = auto``.
_NO_FALLBACK_REASONS = _PASS_THROUGH_REASONS | frozenset({
    ReasonCode.DURATION_UNKNOWN, ReasonCode.DURATION_EXCEEDED, ReasonCode.MODEL_OOM,
})
_ACTION_RE = re.compile(r"[a-z][a-z0-9_]{0,47}\Z")
_ID_RE = re.compile(r"(?:snapshot-[0-9a-f]{32}|[0-9a-f]{32}|"
                    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\Z")
_OFFSET_RE = re.compile(r"[0-9]{1,6}\Z")
_ROLES = ("scratch", "cache", "output", "state")
_MEDIA_SLOTS = frozenset({WorkerSlot.DOWNLOADED_MEDIA, WorkerSlot.DOWNLOADED_AUDIO,
                          WorkerSlot.DOWNLOADED_VIDEO, WorkerSlot.MERGED_VIDEO})
_FINAL_STATES = frozenset({JobState.COMPLETED, JobState.PARTIAL, JobState.FAILED,
                           JobState.CANCELLED})
_AUDIT_ALIASES = {"jobs.cancel": "cancel", "service.stop": "service_stop"}
_CONTENT_TYPES = {
    ArtifactFormat.TXT: "text/plain; charset=utf-8",
    ArtifactFormat.TIMESTAMPED_TXT: "text/plain; charset=utf-8",
    ArtifactFormat.READABLE_TXT: "text/plain; charset=utf-8",
    ArtifactFormat.SRT: "application/x-subrip; charset=utf-8",
    ArtifactFormat.JSON: "application/json; charset=utf-8",
}
# Fixed published names of transcript exports (artifacts.publish's naming contract).
_TRANSCRIPT_NAMES = {
    ArtifactSlot.TRANSCRIPT_TXT: "transcript.txt",
    ArtifactSlot.TRANSCRIPT_TIMESTAMPED_TXT: "transcript.timestamped.txt",
    ArtifactSlot.TRANSCRIPT_READABLE_TXT: "transcript.readable.txt",
    ArtifactSlot.TRANSCRIPT_SRT: "transcript.srt",
    ArtifactSlot.TRANSCRIPT_JSON: "transcript.json",
}
_VIDEO_TYPES = {"mp4": "video/mp4", "mkv": "video/x-matroska", "webm": "video/webm"}
_RETRY_PULL = "check your connection and retry the setup; the image is public"
# Operator guidance per closed reason of a setup PULL (``_Provisioning._pull_release``).
_PULL_GUIDANCE = {
    ReasonCode.DOCKER_UNAVAILABLE: "start Docker Desktop",
    ReasonCode.DOCKER_CONTEXT_UNSUPPORTED: "start Docker Desktop and use its local context",
    ReasonCode.STORAGE_INSUFFICIENT: "free space on the Docker disk",
    ReasonCode.STORAGE_GUARD_TRIPPED: "free space on the Docker disk",
    ReasonCode.STORAGE_UNKNOWN: "retry the setup once Docker storage can be measured",
    ReasonCode.STORAGE_STALE: "retry the setup once Docker storage can be measured",
    ReasonCode.BACKING_UNRESOLVED: "verify the Docker data location, then retry the setup",
    ReasonCode.UPSTREAM_UNAVAILABLE: "check your connection; the image is public",
    ReasonCode.DOWNLOAD_TIMEOUT: _RETRY_PULL,
    ReasonCode.REQUEST_TIMEOUT: _RETRY_PULL,
    ReasonCode.ASSET_VERIFICATION_FAILED: "the pulled image does not carry the manifest digest; "
                                          "retry the setup",
    ReasonCode.CANCELLED: "the pull was cancelled; retry the setup",
    ReasonCode.WRITER_UNRESOLVED: "a Docker pull may still be running; let it finish, restart the "
                                  "service, then retry the setup",
    ReasonCode.ENVELOPE_UNVALIDATED: "restart the service, then retry the setup",
}
_SETUP_CONFIG_KEYS = frozenset({"paths", "manual_backing", "model"})
_SETUP_PATH_KEYS = frozenset({"data", "output", "cache", "scratch", "docker_backing"})
_NO_MANIFEST_ACTION = "podmuncher --setup"


# ------------------------------------------------------------------------------ helpers


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _stamp(moment: datetime | None = None) -> str:
    return (moment or _utcnow()).astimezone(UTC).strftime(_STAMP)


class _Clock:
    """scheduler/recovery/telemetry clock port: UTC stamps and a monotonic source."""

    def now(self) -> str:
        return _stamp()

    def monotonic(self) -> float:
        return time.monotonic()


def _reason_of(error: BaseException, default: ReasonCode = ReasonCode.SERVICE_UNAVAILABLE
               ) -> ReasonCode:
    """The closed reason an error carries (``reason_code``, or a discovery ``reason``)."""
    for name in ("reason_code", "reason"):
        reason = getattr(error, name, None)
        if isinstance(reason, ReasonCode):
            return reason
    return default


def _closed(error: BaseException, default: ReasonCode = ReasonCode.SERVICE_UNAVAILABLE
            ) -> Exception:
    """A closed ``ServiceError`` for any startup failure; never the original message."""
    from .service import ServiceError

    return ServiceError(_reason_of(error, default))


def _identifier(value: object) -> str | None:
    return value if type(value) is str and _ID_RE.fullmatch(value) else None


def _write_json_atomic(path: str, document: Mapping[str, Any]) -> None:
    """Private atomic replace in the state directory (temp file, fsync, replace)."""
    payload = json.dumps(dict(document), sort_keys=True, allow_nan=False).encode("utf-8")
    if len(payload) > MAX_RECORD_BYTES:
        raise DomainValidationError(ReasonCode.STATE_WRITE_FAILED)
    temp = f"{path}.{uuid.uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    fd = os.open(temp, flags, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temp)
        raise


def _read_json_bounded(path: str) -> dict | None:
    try:
        with open(path, "rb") as handle:
            raw = handle.read(MAX_RECORD_BYTES + 1)
    except FileNotFoundError:
        return None
    if len(raw) > MAX_RECORD_BYTES:
        raise DomainValidationError(ReasonCode.ARTIFACT_REJECTED)
    try:
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise DomainValidationError(ReasonCode.ARTIFACT_REJECTED) from None
    return data if type(data) is dict else None


def _private_dir(path: str | os.PathLike) -> str:
    text = os.fspath(path)
    os.makedirs(text, mode=0o700, exist_ok=True)
    return text


# ------------------------------------------------------------ manifest, config, registries


def config_path(environ: Mapping[str, str] | None = None) -> Path:
    """Saved preferences path: ``PODMUNCHER_CONFIG`` or the platform default (as the CLI)."""
    env = os.environ if environ is None else environ
    value = env.get(CONFIG_ENV)
    if value:
        return Path(value)
    from .setup import default_config_path

    return default_config_path()


def load_runtime_manifest(environ: Mapping[str, str] | None = None) -> Any:
    """The explicit development fixture manifest named by ``PODMUNCHER_RUNTIME_MANIFEST``,
    or None.  The packaged release manifest is resolved by ``resolve_runtime_manifest``."""
    env = os.environ if environ is None else environ
    value = env.get(MANIFEST_ENV, "")
    if not value:
        return None
    from .manifests import load_manifest

    return load_manifest(value, development=True)


def registries_of(manifest: Any) -> tuple[frozenset[str] | None, frozenset[str] | None]:
    """``(verified model ids, verified languages)``; ``(None, None)`` keeps auto-only."""
    if manifest is None:
        return None, None
    models = frozenset(model.artifact.model_id for model in manifest.models)
    return models, frozenset(manifest.languages)


@dataclass(frozen=True)
class ManifestResolution:
    """Where the runtime manifest came from, or the closed reason there is none.

    ``manifest`` is the loaded ``RuntimeManifest`` or None.  ``source`` is ``development``
    (the explicit fixture) or ``packaged`` (the release document).  Without a manifest,
    ``reason`` and ``next_action`` are the closed operator guidance every "no manifest"
    step repeats; ``deferred`` marks the one case that is not final for this build: the
    release is publishable but the Docker engine's architecture is not known yet.
    ``blockers`` are the release's blocker codes (closed identifiers, never free text).
    ``common_registries`` are the model ids and languages every loadable platform manifest
    of a publishable release verifies: the client-side registry when no engine is selected.
    """

    manifest: Any
    source: str
    reason: ReasonCode = ReasonCode.OK
    next_action: str = ""
    deferred: bool = False
    blockers: tuple[str, ...] = ()
    common_registries: tuple[frozenset[str] | None, frozenset[str] | None] = (None, None)

    @property
    def registries(self) -> tuple[frozenset[str] | None, frozenset[str] | None]:
        return registries_of(self.manifest)


def _canonical_json(value: object) -> bytes:
    """The release document's canonical form of an embedded runtime manifest (P6-U5)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def _read_packaged_release() -> bytes:
    """The packaged ``release.json`` bytes, bounded; a missing resource is a verification
    failure like any other (a wheel always ships it)."""
    from importlib import resources

    resource = resources.files("podmuncher").joinpath(*RELEASE_RESOURCE)
    try:
        with resource.open("rb") as handle:
            raw = handle.read(MAX_RELEASE_DOCUMENT_BYTES + 1)
    except (OSError, TypeError, ValueError):
        raise DomainValidationError(ReasonCode.ASSET_VERIFICATION_FAILED) from None
    if not raw or len(raw) > MAX_RELEASE_DOCUMENT_BYTES:
        raise DomainValidationError(ReasonCode.ASSET_VERIFICATION_FAILED)
    return raw


def _strict_document(raw: bytes) -> dict[str, Any]:
    """The release document as a dict: strict UTF-8, no duplicate keys, no NaN/Infinity."""

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise DomainValidationError(ReasonCode.ASSET_VERIFICATION_FAILED)
            result[key] = value
        return result

    def constant(_value: str) -> Any:
        raise DomainValidationError(ReasonCode.ASSET_VERIFICATION_FAILED)

    try:
        document = json.loads(raw.decode("utf-8", errors="strict"), object_pairs_hook=pairs,
                              parse_constant=constant)
    except (UnicodeError, ValueError, RecursionError, MemoryError, TypeError):
        raise DomainValidationError(ReasonCode.ASSET_VERIFICATION_FAILED) from None
    if type(document) is not dict:
        raise DomainValidationError(ReasonCode.ASSET_VERIFICATION_FAILED)
    return document


class _PackagedRelease:
    """The release document that ships as ``podmuncher/manifests/release.json``.

    Validated here: the document name and version, ``release_status`` (``publishable`` and
    closed ``blockers[].code`` identifiers), one runtime manifest per Docker engine platform
    under ``runtime_manifests`` and, for each, that ``runtime_manifest_sha256`` matches the
    sha256 of its canonical JSON (P6-U5's rule, recomputed).  Anything else in the wrapper
    (registry, build identities, benchmarks) is the release workflow's business.  A platform
    manifest is loaded lazily through the real loader in release mode with the exact bytes
    that were hashed, so the loader's own source and schema policy decides what runs.
    """

    def __init__(self, raw: bytes) -> None:
        document = _strict_document(raw)
        failure = DomainValidationError(ReasonCode.ASSET_VERIFICATION_FAILED)
        if (document.get("document") != RELEASE_DOCUMENT
                or type(document.get("document_version")) is not int
                or document["document_version"] != RELEASE_DOCUMENT_VERSION):
            raise failure
        status = document.get("release_status")
        if type(status) is not dict or type(status.get("publishable")) is not bool \
                or type(status.get("blockers")) is not list \
                or len(status["blockers"]) > MAX_RELEASE_BLOCKERS:
            raise failure
        codes: list[str] = []
        for blocker in status["blockers"]:
            code = blocker.get("code") if type(blocker) is dict else None
            if type(code) is not str or _BLOCKER_CODE_RE.fullmatch(code) is None:
                raise failure
            if code not in codes:
                codes.append(code)
        manifests, digests = document.get("runtime_manifests"), document.get("runtime_manifest_sha256")
        if (type(manifests) is not dict or type(digests) is not dict or not manifests
                or len(manifests) > MAX_RELEASE_PLATFORMS or set(manifests) != set(digests)):
            raise failure
        self._manifests: dict[str, bytes] = {}
        self._digests: dict[str, str] = {}
        for platform, manifest in manifests.items():
            digest = digests[platform]
            if (_RELEASE_PLATFORM_RE.fullmatch(platform) is None or type(manifest) is not dict
                    or type(digest) is not str or _SHA256_RE.fullmatch(digest) is None):
                raise failure
            data = _canonical_json(manifest)
            if not hmac.compare_digest(hashlib.sha256(data).hexdigest(), digest):
                raise failure
            self._manifests[platform] = data
            self._digests[platform] = digest
        self.publishable: bool = status["publishable"]
        self.blocker_codes: tuple[str, ...] = tuple(codes)
        self._loaded: dict[str, Any] = {}
        self._lock = threading.Lock()

    @property
    def platforms(self) -> tuple[str, ...]:
        return tuple(sorted(self._manifests))

    def _load(self, platform: str) -> Any:
        from .manifests import load_manifest

        # The loader takes a path and proves its identity before and after reading, so the
        # hashed bytes go through a private file that exists only for this call.
        directory = tempfile.mkdtemp(prefix="podmuncher-manifest-")
        try:
            path = os.path.join(directory, f"{platform}.json")
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
            with os.fdopen(os.open(path, flags, 0o600), "wb") as handle:
                handle.write(self._manifests[platform])
            return load_manifest(path, development=False, allowed_release_sources=(path,),
                                 allowed_asset_sources=PACKAGED_ASSET_ORIGINS,
                                 expected_sha256=self._digests[platform])
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def manifest_for(self, platform: str) -> Any:
        """The platform's runtime manifest loaded in release mode; None when the release has
        no manifest for it.  A manifest that fails the loader raises its closed error."""
        if platform not in self._manifests:
            return None
        with self._lock:
            outcome = self._loaded.get(platform)
            if outcome is None:
                try:
                    outcome = ("ok", self._load(platform))
                except DomainValidationError as error:
                    outcome = ("error", error.reason_code)
                except (OSError, ValueError, TypeError):
                    outcome = ("error", ReasonCode.ASSET_VERIFICATION_FAILED)
                self._loaded[platform] = outcome
        if outcome[0] != "ok":
            raise DomainValidationError(outcome[1])
        return outcome[1]

    def common_registries(self) -> tuple[frozenset[str] | None, frozenset[str] | None]:
        """The model ids and languages every loadable platform manifest verifies."""
        loaded = []
        for platform in self.platforms:
            try:
                loaded.append(self.manifest_for(platform))
            except DomainValidationError:
                continue
        if not loaded:
            return None, None
        models, languages = registries_of(loaded[0])
        for manifest in loaded[1:]:
            more_models, more_languages = registries_of(manifest)
            models, languages = models & more_models, languages & more_languages
        return models, languages


def resolve_runtime_manifest(environ: Mapping[str, str] | None = None, *,
                             engine_arch: Callable[[], str | None] | str | None = None,
                             reader: Callable[[], bytes] | None = None) -> ManifestResolution:
    """The runtime manifest this process runs with, or the closed reason there is none.

    ``PODMUNCHER_RUNTIME_MANIFEST`` set: the development fixture, exactly as
    ``load_runtime_manifest`` (its errors propagate: a named fixture that fails is a
    startup failure).  Unset: the packaged release document (``reader`` injects its bytes;
    the default reads the package resource).  A publishable release selects the platform
    manifest for the Docker engine architecture (``engine_arch``: a value, or a callable
    sampled only when needed, mapped by ``manifests.engine_arch_platform``) and loads it in
    release mode.  Every failure of the packaged path is reported, never raised.
    """
    from .manifests import engine_arch_platform

    env = os.environ if environ is None else environ
    if env.get(MANIFEST_ENV, ""):
        return ManifestResolution(load_runtime_manifest(env), "development")
    unverified = (f"this build's packaged release manifest failed verification; reinstall a "
                  f"release build or set {MANIFEST_ENV}")
    try:
        packaged = _PackagedRelease((reader or _read_packaged_release)())
    except DomainValidationError as error:
        return ManifestResolution(None, "packaged", error.reason_code, unverified)
    except Exception:  # noqa: BLE001 - a reader fault is a closed verification failure
        return ManifestResolution(None, "packaged", ReasonCode.ASSET_VERIFICATION_FAILED, unverified)
    if not packaged.publishable:
        codes = ", ".join(packaged.blocker_codes[:NAMED_BLOCKERS]) or "unpublished"
        return ManifestResolution(
            None, "packaged", ReasonCode.SERVICE_UNAVAILABLE,
            f"this build has no published images ({codes}); install a release build or set "
            f"{MANIFEST_ENV}", blockers=packaged.blocker_codes)
    arch: object = engine_arch
    if callable(engine_arch):
        try:
            arch = engine_arch()
        except Exception:  # noqa: BLE001 - an unsampled engine is unknown, never a guess
            arch = None
    platform = engine_arch_platform(arch)
    if platform is None:
        return ManifestResolution(
            None, "packaged", ReasonCode.DOCKER_UNAVAILABLE,
            "start Docker so the engine architecture is known, then restart the service",
            deferred=True, blockers=packaged.blocker_codes,
            common_registries=packaged.common_registries())
    try:
        manifest = packaged.manifest_for(platform)
    except DomainValidationError as error:
        return ManifestResolution(
            None, "packaged", error.reason_code,
            f"this build's {platform} runtime manifest failed verification; reinstall a release "
            f"build or set {MANIFEST_ENV}", blockers=packaged.blocker_codes)
    if manifest is None:
        return ManifestResolution(
            None, "packaged", ReasonCode.SERVICE_UNAVAILABLE,
            f"this build has no images for a {platform} Docker engine; install a release build "
            f"for it or set {MANIFEST_ENV}", blockers=packaged.blocker_codes)
    return ManifestResolution(manifest, "packaged", blockers=packaged.blocker_codes)


def load_config(environ: Mapping[str, str] | None,
                registries: tuple[frozenset[str] | None, frozenset[str] | None]) -> Any:
    """Saved configuration; a missing file means defaults.  Errors are closed ConfigErrors."""
    from .config import Config, ConfigError

    path = config_path(environ)
    try:
        with open(path, "rb") as handle:
            raw = handle.read()
    except FileNotFoundError:
        raw = None
    except OSError:
        raise ConfigError() from None
    models, languages = registries
    kwargs = {"verified_models": models, "verified_languages": languages}
    return Config.from_toml(raw, **kwargs) if raw is not None else Config.from_dict(None, **kwargs)


class _LazyRegistry:
    """CLI registry provider: resolves the manifest only when a request is built.

    The client never contacts Docker, so it does not know the engine architecture: with a
    publishable packaged release it admits the model ids and languages every platform
    manifest verifies (``common_registries``); the service re-validates every request
    against the manifest it selected for the engine.
    """

    def __init__(self, environ: Mapping[str, str] | None) -> None:
        self._environ = environ
        self._value: tuple[frozenset[str] | None, frozenset[str] | None] | None = None
        self._lock = threading.Lock()

    def __call__(self) -> tuple[frozenset[str] | None, frozenset[str] | None]:
        with self._lock:
            if self._value is None:
                resolution = resolve_runtime_manifest(self._environ)
                self._value = (resolution.registries if resolution.manifest is not None
                               else resolution.common_registries)
            return self._value


def client_startup(environ: Mapping[str, str] | None = None) -> None:
    """Client-only composition run by the CLI hook: never starts or contacts the daemon."""
    from . import cli, presentation

    cli.configure_presentation(presentation.default_adapter())
    cli.set_registry_provider(_LazyRegistry(environ))


# ----------------------------------------------------------------------- seams, context


@dataclass
class Seams:
    """Injection points for tests.  Every None means the real adapter."""

    volumes: Any = None
    docker_storage: Any = None
    containers: Any = None
    launcher: Callable[[Any], Any] | None = None
    models: Callable[[Any], Any] | None = None
    hardware: Any = None
    provisioning: Callable[[Any], Any] | None = None
    probe: Callable[[Any], Any] | None = None  # the probe/smoke launch port (``_DockerLaunch``)
    acquire: Callable[[Any], Callable[[Any], Any]] | None = None  # ``ModelCache.acquire``
    repository_fault_hook: Callable[[str], None] | None = None
    quiesce_timeout: float = 30.0


@dataclass
class _Context:
    state_dir: str
    environ: Mapping[str, str]
    clock: _Clock
    volumes: Any
    docker_storage: Any
    config: Any = None
    manifest: Any = None
    manifest_source: Any = None  # the ``ManifestResolution`` behind ``manifest`` (guidance)
    registries: tuple[Any, Any] = (None, None)
    repo: Any = None
    hub: Any = None
    cache: Any = None
    scratch: Any = None
    models: Any = None
    acquire_model: Callable[[Any], Any] | None = None  # manifest model -> verified snapshot
    instance: str = ""
    records_dir: str = ""

    def capability(self, capability: Capability | str) -> Any:
        """The manifest capability entry, or None when no manifest is loaded."""
        if self.manifest is None:
            return None
        wanted = Capability.parse(capability)
        return next((c for c in self.manifest.capabilities if c.capability is wanted), None)

    def resolved(self, capability: Capability | str) -> Any:
        """``manifests.resolve_capability`` or a closed not-ready failure (never a pull)."""
        from .manifests import resolve_capability
        from .runtime import RuntimeFailure

        if self.capability(capability) is None:
            raise RuntimeFailure(ReasonCode.SERVICE_UNAVAILABLE)
        return resolve_capability(self.manifest, capability)


class _HeldRootVolumes:
    """Volume provider that samples the model-cache root through its parent directory.

    ``ModelCache`` holds its root directory open for its lifetime without write sharing, so
    the Windows permission probe on that exact directory cannot open it and reports
    ``writable`` as unknown.  The cache root is never a reparse point (the cache refuses
    one), so its parent is on the same volume: identity, capacity and free space are the
    same evidence.  Every other path is sampled unchanged.

    The detour is Windows-only (``platform`` is ``os.name``): a POSIX probe opens a held
    directory fine, and a cache root that is its own mount point (a Linux volume mounted
    at the cache path) must be budgeted against its own filesystem, not its parent's.
    """

    def __init__(self, inner: Any, held_roots: Iterable[str], *, platform: str = os.name) -> None:
        self._inner = inner
        self._held = ({os.path.normcase(os.path.abspath(root)) for root in held_roots}
                      if platform == "nt" else set())

    def _map(self, path: str | os.PathLike) -> str:
        text = os.path.abspath(os.fspath(path))
        return os.path.dirname(text) if os.path.normcase(text) in self._held else text

    def sample(self, path: str | os.PathLike, **kwargs: Any) -> Any:
        return self._inner.sample(self._map(path), **kwargs)

    def sample_many(self, paths: tuple[str, ...]) -> tuple[Any, ...]:
        return tuple(self.sample(path) for path in paths)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _MemoryLease:
    """Nested in-process lease for one worker or publication guard.

    The scheduler already holds the durable single execution lease for the whole job (or
    setup operation); a coordinator inside that operation needs its own lease object.
    """

    def __init__(self) -> None:
        self._holder: str | None = None

    def try_acquire(self, operation_id: str, plan: Any) -> bool:
        if self._holder is not None or not getattr(plan, "admitted", False):
            return False
        self._holder = operation_id
        return True

    def release(self, operation_id: str, *, writers_reaped: bool) -> None:
        if self._holder == operation_id and writers_reaped:
            self._holder = None


# ------------------------------------------------------------------------- scratch index


class _ScratchIndex:
    """Host-assigned scratch names to their private output directories.

    Every ``assign`` gets a fresh directory ``<scratch>/<attempt_id>/<hex>/`` so a worker's
    writable mount never contains the input file of a later stage.  Names are unique
    (uuid hex) and never chosen by a worker.
    """

    def __init__(self, root: str) -> None:
        self.root = root
        self._dirs: dict[str, str] = {}
        self._lock = threading.Lock()

    def new_dir(self, attempt_id: str) -> str:
        from .domains import validate_uuid_string

        validate_uuid_string(attempt_id)
        return _private_dir(os.path.join(self.root, attempt_id, uuid.uuid4().hex))

    def register(self, name: str, directory: str) -> None:
        with self._lock:
            self._dirs[name] = directory

    def directory_of(self, name: str) -> str:
        with self._lock:
            directory = self._dirs.get(name)
        if directory is None:
            from .artifacts import ArtifactError

            raise ArtifactError(ReasonCode.ARTIFACT_REJECTED)
        return directory

    def path_of(self, name: str) -> str:
        return os.path.join(self.directory_of(name), name)

    def attempt_root(self, attempt_id: str) -> str:
        from .domains import validate_uuid_string

        validate_uuid_string(attempt_id)
        return os.path.join(self.root, attempt_id)


# ---------------------------------------------------------------------------- preflight


def _limits_for(ctx: _Context, request: Any) -> Any:
    """Configured limits narrowed to the request's media/duration/export caps."""
    limits = ctx.config.limits
    return replace(limits, media_bytes=min(limits.media_bytes, request.limits.media_bytes),
                   duration_seconds=min(limits.duration_seconds, request.limits.duration_seconds),
                   export_bytes=min(limits.export_bytes, request.limits.export_bytes))


def _cached_model_files(ctx: _Context) -> set[tuple]:
    if ctx.cache is None:
        return set()
    try:
        return {tuple(snapshot.files) for snapshot in ctx.cache.snapshots()}
    except Exception:  # noqa: BLE001 - an unreadable cache simply credits nothing
        return set()


def _growth(ctx: _Context, request: Any, capabilities: Iterable[Capability], *,
            image_presence: Callable[[str], bool] | None = None) -> tuple[Any, dict]:
    """Conservative workload growth plus allocation credits for one job request.

    Image bytes are credited as allocated for development images (local-only, never
    pulled) and for a release image that ``image_presence`` verifies present locally (a
    bounded, read-only inspect; see ``_release_image_present``).  Any other release image
    budgets its full compressed plus unpacked growth, so the job's admission covers the
    pull the runtime will do.  ``image_presence`` None never asks: a caller that discards
    the credits (the launcher) must not block on Docker for them.
    """
    from .admission import WorkloadGrowth

    entries = [ctx.capability(c) for c in capabilities]
    entries = [e for e in entries if e is not None]
    compressed = sum(e.compressed_bytes for e in entries)
    unpacked = sum(e.unpacked_peak_bytes for e in entries)
    if ctx.manifest is not None and ctx.manifest.development_fixture:
        present = entries
    elif image_presence is not None:
        present = [e for e in entries if image_presence(_release_image(e))]
    else:
        present = []
    allocated = {"image_compressed": sum(e.compressed_bytes for e in present),
                 "image_unpacked": sum(e.unpacked_peak_bytes for e in present)}
    model_total = 0
    if request.outputs.transcript and ctx.manifest is not None and ctx.manifest.models:
        cached = _cached_model_files(ctx)
        sizes = []
        for model in ctx.manifest.models:
            size = sum(item.bytes for item in model.files)
            sizes.append((size, tuple(model.files) in cached))
        model_total, present = max(sizes)
        if present:
            allocated.update(model_partial=model_total, model_unpacked=model_total)
    remote = request.source.kind != "local"
    media = request.limits.media_bytes
    video = request.outputs.video
    growth = WorkloadGrowth(
        image_compressed=compressed, image_unpacked=unpacked,
        model_partial=model_total, model_unpacked=model_total,
        source_streams=(None,) if remote else (0,),
        merge=media if remote and video else 0, copied_source=media if video and not remote else 0,
        copied_video=media if video else 0, publication_staging=media if video else 0,
        transcript_outputs=(request.limits.export_bytes,) if request.outputs.transcript else (),
        sidecars=MIB, state=MIB)
    return growth, allocated


def _capabilities_for(request: Any) -> tuple[Capability, ...]:
    caps = [Capability.MEDIA]
    if request.outputs.transcript:
        caps.append(Capability.ASR_CUDA if request.selection.device == "cuda" else Capability.ASR_CPU)
    return tuple(caps)


def _blocked_plan(reason: ReasonCode) -> Any:
    from .admission import AdmissionPlan
    from .snapshots import Measurement

    return AdmissionPlan(Measurement(None, "preflight", FieldStatus.ERROR, _stamp(), reason), (), ())


def _sample_roles(ctx: _Context, paths: Mapping[str, str]) -> dict[str, Any]:
    return {role: ctx.volumes.sample(path, sqlite_state=role == "state")
            for role, path in paths.items()}


class _StoragePreflight:
    """``scheduler.Preflight``: fresh volume + Docker storage evidence on every call."""

    def __init__(self, ctx: _Context) -> None:
        self._ctx = ctx

    def plan(self, job: Any, stage: Stage) -> Any:
        from .admission import StoragePlacement, make_workload_plan

        ctx = self._ctx
        if ctx.manifest is None:
            return _blocked_plan(ReasonCode.SERVICE_UNAVAILABLE)
        request = job.request
        caps = _capabilities_for(request)
        if any(ctx.capability(c) is None for c in caps):
            return _blocked_plan(ReasonCode.SERVICE_UNAVAILABLE)
        try:
            docker = ctx.docker_storage.sample(manual_backing=ctx.config.paths.docker_backing)
        except Exception as error:  # noqa: BLE001 - discovery failure blocks with its reason
            return _blocked_plan(_reason_of(error, ReasonCode.DOCKER_UNAVAILABLE))
        if not docker.can_create_cached:
            return _blocked_plan(docker.reason or ReasonCode.STORAGE_UNKNOWN)
        paths = {"scratch": ctx.config.paths.scratch, "cache": ctx.config.paths.cache,
                 "output": request.output_root, "state": ctx.state_dir}
        selected = _sample_roles(ctx, paths)
        roles = {role: c.volume.identity.value for c in docker.constraints for role in c.roles}
        vm = roles.get("docker_vm", roles.get("docker_data"))
        backing = roles.get("docker_backing", vm)
        identities = [selected[k].identity.value for k in _ROLES]
        if vm is None or any(type(v) is not str or not v for v in identities):
            return _blocked_plan(ReasonCode.STORAGE_UNKNOWN if vm is not None
                                 else ReasonCode.BACKING_UNRESOLVED)
        placement = StoragePlacement(*identities, vm, backing)
        # Off the loop (the scheduler runs this plan in a thread): a release image not yet
        # pulled is budgeted, one verified present is credited.
        growth, allocated = _growth(ctx, request, caps, image_presence=_release_image_present)
        optional = frozenset(identities) | {backing} if os.name == "nt" else frozenset()
        return make_workload_plan(
            tuple(selected[k] for k in _ROLES), growth, placement, _limits_for(ctx, request),
            now=_utcnow(), docker=docker, state_evidence=selected["state"], allocated=allocated,
            inode_optional_identities=optional)


def _host_plan(ctx: _Context, role: str, path: str, total: int, limits: Any = None) -> Any:
    """Host-only admission for one bounded write (publication copy, setup) on one volume."""
    from .admission import GrowthInterval, make_plan

    evidence = ctx.volumes.sample(path)
    state = ctx.volumes.sample(ctx.state_dir, sqlite_state=True)
    identity = evidence.identity.value
    if type(identity) is not str or not identity:
        return _blocked_plan(ReasonCode.STORAGE_UNKNOWN)
    interval = GrowthInterval(role, identity, total, 0, 0, 1)
    optional = frozenset({identity, state.identity.value or ""} - {""}) if os.name == "nt" else frozenset()
    return make_plan((evidence,), (interval,), limits or ctx.config.limits, now=_utcnow(),
                     state_evidence=state, inode_optional_identities=optional)


def _plan_allowance(plan: Any) -> int | None:
    """Bytes the planned volume can still take: its measured ``free - reserved - reserve``."""
    identity = plan.intervals[0].volume_identity if plan.intervals else None
    for requirement in plan.volumes:
        if requirement.volume_identity != identity:
            continue
        values = (requirement.free.value, requirement.reserved.value, requirement.reserve.value)
        if all(type(value) is int for value in values):
            return values[0] - values[1] - values[2]
        return None
    return None


class _ModelAdmission:
    """``ModelCache`` admission: one host plan per 8 MiB (or second) of growth, fail-closed.

    The cache asks admission once per received chunk.  A host plan takes two volume samples
    (about 10 ms), so planning on every call adds about a minute to a 78 MB model and runs
    the acquisition into its deadline.  Between plans the last plan's allowance (the cache
    volume's measured free space minus its reserve) is compared with the bytes projected
    since that sample: what was allocated meanwhile plus what remains, which is exactly the
    growth the sampled free space still has to hold.  A non-admitting plan refuses, and a
    sample error propagates (the cache fails closed on it); both drop the cached allowance.
    """

    def __init__(self, ctx: _Context, *, path: str | None = None,
                 monotonic: Callable[[], float] = time.monotonic) -> None:
        self._ctx = ctx
        self._path = path  # the cache root admitted; None means the configured cache
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._asset: str | None = None
        self._base = 0  # allocated bytes when the allowance was measured
        self._at = 0.0
        self._allowance: int | None = None

    def __call__(self, *, asset_id: str, declared_bytes: int, allocated_bytes: int) -> bool:
        declared, allocated = int(declared_bytes), int(allocated_bytes)
        with self._lock:
            now = self._monotonic()
            if (self._allowance is not None and asset_id == self._asset
                    and 0 <= allocated - self._base < _ADMISSION_RESAMPLE_BYTES
                    and 0 <= now - self._at < _ADMISSION_RESAMPLE_SECONDS):
                return max(0, declared - self._base) <= self._allowance
            self._allowance = None  # fail closed until a fresh plan admits
            remaining = max(0, declared - allocated)
            plan = _host_plan(self._ctx, "model_unpacked", self._path or self._ctx.config.paths.cache,
                              remaining)
            if not plan.admitted:
                return False
            self._asset, self._base, self._at = asset_id, allocated, now
            self._allowance = _plan_allowance(plan)
            return True


def _same_path(left: str | None, right: str | None) -> bool:
    if left is None or right is None:
        return left is right
    return os.path.normcase(os.path.abspath(left)) == os.path.normcase(os.path.abspath(right))


def _open_model_cache(ctx: _Context, path: str | None = None) -> Any:
    """A ``ModelCache`` at ``path`` (the configured cache when None) with the composed
    audit, admission, spans, CA bundle and download timeout; None when it cannot open.

    Setup opens one at the cache path it is provisioning; the service's own cache is the
    configured one.  ``ctx.volumes`` must already sample ``path`` through its parent.
    """
    from .model_cache import ModelCache

    root = path if path is not None else ctx.config.paths.cache
    # The whole acquisition (every file of one model) gets the configured download
    # timeout; the cache's own default (30 s) is for tests.  Its ceiling is 3600 s.
    timeout = float(min(ctx.config.limits.download_timeout_seconds, 3600))
    try:
        return ModelCache(root, audit=ctx.hub.cache_audit(),
                          admission=_ModelAdmission(ctx, path=path), spans=ctx.hub.cache_spans(),
                          ca_bundle=ctx.config.tls.ca_bundle or None, timeout_seconds=timeout)
    except DomainValidationError:
        return None  # cache routes answer cache_unknown; jobs needing a model block


async def _started_guard(ctx: _Context, path: str, total: int, *, timeout: float) -> Any:
    """A started publication WriteGuard with its own nested lease and host sampler."""
    from .guards import CancellationCoordinator, StorageSample, WriteGuard

    async def sampler() -> Any:
        plan = await asyncio.to_thread(_host_plan, ctx, "publication_staging", path, total)
        return StorageSample(plan, time.monotonic())

    coordinator = CancellationCoordinator(f"publish-{uuid.uuid4().hex}", _MemoryLease())
    guard = WriteGuard(coordinator, sampler, byte_limit=total, timeout_seconds=timeout,
                       sample_timeout_seconds=2.9, cap_reason=ReasonCode.EXPORT_LIMIT_REACHED)
    await guard.start()
    return guard


# ---------------------------------------------------------------------------- publisher


class _Verified:
    """A host-verified scratch file.  Large media are verified lazily, off the loop."""

    def __init__(self, *, job_id: str, slot: WorkerSlot, name: str, directory: str,
                 verify: Callable[[], Any] | None = None, file: Any = None,
                 provenance: Any = None) -> None:
        self.job_id, self.slot, self.name, self.directory = job_id, slot, name, directory
        self.provenance = provenance
        self._verify, self._file = verify, file
        self._error: BaseException | None = None
        self._lock = threading.Lock()

    def resolve(self) -> Any:
        with self._lock:
            if self._file is None and self._error is None:
                try:
                    self._file = self._verify()
                except Exception as error:  # noqa: BLE001 - re-raised to every consumer
                    self._error = error
            if self._error is not None:
                raise self._error
            return self._file

    @property
    def sha256(self) -> str:
        return self.resolve().sha256


def _never_same(source_volume: str, output_volume: str) -> bool:
    """``artifacts.publish`` volume chooser that forces the copy path."""
    return False


@dataclass(frozen=True)
class _LocalVideo:
    """Deferred local-source video publication input (copied and verified off the loop)."""

    job: Any
    attempt_id: str


@dataclass(frozen=True)
class _ExportRef:
    directory: str
    name: str
    size: int
    sha256: str
    provenance: Any


class _Publisher:
    """``scheduler.Publisher`` over ``artifacts`` with contained, host-assigned scratch."""

    def __init__(self, ctx: _Context) -> None:
        self._ctx = ctx
        self._published: OrderedDict[str, tuple[Any, str]] = OrderedDict()
        self._media: dict[str, _Verified] = {}
        self._roots: dict[str, str] = {}
        self._record_lock = threading.Lock()

    # -- helpers ---------------------------------------------------------------------
    def _max_bytes(self, job: Any, slot: WorkerSlot) -> int:
        limits = _limits_for(self._ctx, job.request)
        return limits.media_bytes if slot in _MEDIA_SLOTS else limits.export_bytes

    def _record_path(self, attempt_id: str) -> str:
        from .domains import validate_uuid_string

        validate_uuid_string(attempt_id)
        return os.path.join(self._ctx.records_dir, f"{attempt_id}.json")

    def _remember(self, attempt_id: str, verified: _Verified, file: Any) -> None:
        """Durable private record of a verified output, for later re-verification (reuse)."""
        with self._record_lock:
            path = self._record_path(attempt_id)
            document = _read_json_bounded(path) or {"entries": {}}
            entries = document.get("entries") if type(document.get("entries")) is dict else {}
            entries[verified.name] = {"slot": verified.slot.value, "content": file.content.value,
                                      "bytes": file.bytes, "sha256": file.sha256,
                                      "directory": verified.directory}
            _write_json_atomic(path, {"entries": entries})

    def _verify_file(self, directory: str, name: str, slot: WorkerSlot, size: int, sha256: str,
                     *, writers_reaped: bool, max_bytes: int, content: Any = None) -> Any:
        from .artifacts import verify_artifact

        return verify_artifact(directory, {slot: name}, slot, size, sha256,
                               writers_reaped=writers_reaped, max_bytes=max_bytes,
                               content=content, audit=self._ctx.hub.artifact_audit())

    def _wrap(self, job: Any, attempt_id: str, slot: WorkerSlot, name: str, directory: str,
              size: int, sha256: str, *, writers_reaped: bool, content: Any = None) -> _Verified:
        max_bytes = self._max_bytes(job, slot)
        holder: dict[str, _Verified] = {}

        def verify() -> Any:
            file = self._verify_file(directory, name, slot, size, sha256,
                                     writers_reaped=writers_reaped, max_bytes=max_bytes,
                                     content=content)
            self._remember(attempt_id, holder["v"], file)
            return file

        verified = _Verified(job_id=job.job_id, slot=slot, name=name, directory=directory,
                             verify=verify)
        holder["v"] = verified
        return verified

    # -- scheduler port ----------------------------------------------------------------
    def assign(self, job: Any, attempt: Any, slots: Iterable[WorkerSlot]) -> dict[WorkerSlot, str]:
        from .artifacts import assign_slots

        directory = self._ctx.scratch.new_dir(attempt.attempt_id)
        assigned = assign_slots(directory, slots)
        for name in assigned.values():
            self._ctx.scratch.register(name, directory)
        self._roots[job.job_id] = job.request.output_root
        return assigned

    def verify(self, job: Any, attempt: Any, assignment: Mapping[WorkerSlot, str], claim: Any, *,
               writers_reaped: bool) -> _Verified:
        from .artifacts import ArtifactError

        try:
            slot = WorkerSlot.parse(claim.payload.get("slot"))
            name = assignment[slot]
            size, sha256 = claim.payload["bytes"], claim.payload["sha256"]
        except (DomainValidationError, KeyError, AttributeError, TypeError):
            raise ArtifactError(ReasonCode.ARTIFACT_REJECTED) from None
        if writers_reaped is not True:
            raise ArtifactError(ReasonCode.WRITER_UNRESOLVED)
        directory = self._ctx.scratch.directory_of(name)
        verified = self._wrap(job, attempt.attempt_id, slot, name, directory, size, sha256,
                              writers_reaped=True)
        if slot in _MEDIA_SLOTS:
            self._media[job.job_id] = verified  # hashed off the loop by source_identity/publish
        else:
            verified.resolve()  # small JSON outputs: verify now
        return verified

    def verify_local_video(self, job: Any, attempt: Any) -> _LocalVideo:
        return _LocalVideo(job, attempt.attempt_id)

    def _copy_local_video(self, local: _LocalVideo) -> tuple[_Verified, str]:
        """Copy the local source into private scratch with a cap, then verify it as video."""
        from .artifacts import VIDEO_EXTENSIONS, ArtifactError

        job = local.job
        source = job.request.source.input
        ext = os.path.splitext(source)[1].lower().lstrip(".")
        if ext not in VIDEO_EXTENSIONS:
            raise ArtifactError(ReasonCode.INVALID_INPUT)
        cap = self._max_bytes(job, WorkerSlot.DOWNLOADED_MEDIA)
        directory = self._ctx.scratch.new_dir(local.attempt_id)
        name = f"{WorkerSlot.DOWNLOADED_MEDIA.value}-{uuid.uuid4().hex}.bin"
        target = os.path.join(directory, name)
        digest, total = hashlib.sha256(), 0
        if os.path.islink(source) or not os.path.isfile(source):
            raise ArtifactError(ReasonCode.INPUT_REJECTED)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
        with open(source, "rb") as reader, os.fdopen(os.open(target, flags, 0o600), "wb") as writer:
            while True:
                chunk = reader.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > cap:
                    raise ArtifactError(ReasonCode.MEDIA_LIMIT_REACHED)
                digest.update(chunk)
                writer.write(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        self._ctx.scratch.register(name, directory)
        verified = self._wrap(job, local.attempt_id, WorkerSlot.DOWNLOADED_MEDIA, name, directory,
                              total, digest.hexdigest(), writers_reaped=True)
        verified.resolve()
        return verified, ext

    def verify_export(self, job: Any, attempt: Any, slot: ArtifactSlot, ref: Any) -> _Verified:
        from .artifacts import ArtifactError

        if type(ref) is not _ExportRef:
            raise ArtifactError(ReasonCode.ARTIFACT_REJECTED)
        file = self._verify_file(ref.directory, ref.name, WorkerSlot.TRANSCRIPT_SEGMENTS,
                                 ref.size, ref.sha256, writers_reaped=True,
                                 max_bytes=self._max_bytes(job, WorkerSlot.TRANSCRIPT_SEGMENTS),
                                 content=ArtifactSlot.parse(slot))
        return _Verified(job_id=job.job_id, slot=WorkerSlot.TRANSCRIPT_SEGMENTS, name=ref.name,
                         directory=ref.directory, file=file, provenance=ref.provenance)

    def probe_facts(self, verified: Any) -> Any:
        from . import transitions as tr
        from .volumes import open_contained

        if type(verified) is not _Verified or verified.slot is not WorkerSlot.PROBE_METADATA:
            raise DomainValidationError(ReasonCode.MEDIA_PROBE_FAILED)
        file = verified.resolve()
        with open_contained(file.attempt_dir, file.basename, expected=file.identity) as held:
            raw = held.read(MAX_PROBE_BYTES)
        try:
            doc = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise DomainValidationError(ReasonCode.MEDIA_PROBE_FAILED) from None
        if (type(doc) is not dict or doc.get("schema_version") != 1
                or type(doc.get("has_audio")) is not bool or type(doc.get("has_video")) is not bool):
            raise DomainValidationError(ReasonCode.MEDIA_PROBE_FAILED)
        duration = doc.get("duration_seconds")
        if duration is None:
            raise DomainValidationError(ReasonCode.DURATION_UNKNOWN)
        return tr.MediaFacts(doc["has_audio"], doc["has_video"], duration)

    def source_identity(self, job: Any) -> str:
        from .artifacts import ArtifactError

        if job.request.source.kind == "local":
            path = job.request.source.input
            if os.path.islink(path) or not os.path.isfile(path):
                raise ArtifactError(ReasonCode.INPUT_REJECTED)
            cap = self._max_bytes(job, WorkerSlot.DOWNLOADED_MEDIA)
            digest, total = hashlib.sha256(), 0
            with open(path, "rb") as handle:
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > cap:
                        raise ArtifactError(ReasonCode.MEDIA_LIMIT_REACHED)
                    digest.update(chunk)
            return digest.hexdigest()
        verified = self._media.get(job.job_id)
        if verified is None:
            raise ArtifactError(ReasonCode.ARTIFACT_REJECTED)
        return verified.sha256  # verifies the download (thread) before it is trusted

    # -- retained-output identity -------------------------------------------------------
    # An export retry carries the verified transcribing record into its new attempt under
    # the new attempt id, while the retained outputs (and their record file) belong to the
    # attempt that transcribed.  The publisher keeps, per job, a bounded private index from
    # a stage record's verified identity (input hashes, settings, output slots) to that
    # attempt, written whenever a reuse succeeds, so every later export retry resolves the
    # same retained outputs and re-verifies them instead of transcribing again.
    def _index_path(self, job_id: str) -> str:
        from .domains import validate_uuid_string

        validate_uuid_string(job_id)
        return os.path.join(self._ctx.records_dir, f"job-{job_id}.json")

    @staticmethod
    def _record_identity(record: Any) -> str:
        hashes = sorted((str(name), str(value)) for name, value in record.verified_input_hashes)
        slots = sorted(WorkerSlot.parse(slot).value for slot in record.output_slots)
        document = json.dumps([hashes, str(record.settings_hash), slots], sort_keys=True)
        return hashlib.sha256(document.encode("utf-8")).hexdigest()

    def _retained_attempt(self, job: Any, record: Any) -> str:
        """The attempt whose private record file holds this stage record's outputs."""
        from .artifacts import ArtifactError
        from .domains import validate_uuid_string

        if _read_json_bounded(self._record_path(record.attempt_id)) is not None:
            return record.attempt_id
        index = _read_json_bounded(self._index_path(job.job_id)) or {}
        retained = index.get(self._record_identity(record))
        try:
            validate_uuid_string(retained)
        except DomainValidationError:
            raise ArtifactError(ReasonCode.ARTIFACT_REJECTED) from None
        return retained

    def _remember_identity(self, job: Any, record: Any, attempt_id: str) -> None:
        with self._record_lock:
            path = self._index_path(job.job_id)
            stored = _read_json_bounded(path) or {}
            index = {key: value for key, value in stored.items()
                     if type(key) is str and type(value) is str}
            key = self._record_identity(record)
            if index.get(key) == attempt_id:
                return
            index.pop(key, None)
            index[key] = attempt_id
            while len(index) > MAX_RETAINED_IDENTITIES:
                index.pop(next(iter(index)))  # oldest first
            _write_json_atomic(path, index)

    def reuse(self, job: Any, record: Any) -> dict[WorkerSlot, tuple[str, _Verified]]:
        from .artifacts import ArtifactError

        retained = self._retained_attempt(job, record)
        document = _read_json_bounded(self._record_path(retained)) or {}
        entries = document.get("entries") if type(document.get("entries")) is dict else {}
        found: dict[WorkerSlot, tuple[str, dict]] = {}
        for name, entry in entries.items():
            if type(entry) is dict and entry.get("slot") in {s.value for s in record.output_slots}:
                found[WorkerSlot.parse(entry["slot"])] = (name, entry)
        result: dict[WorkerSlot, tuple[str, _Verified]] = {}
        for slot in record.output_slots:
            if slot not in found:
                raise ArtifactError(ReasonCode.ARTIFACT_REJECTED)
            name, entry = found[slot]
            directory = entry.get("directory")
            if type(directory) is not str or os.path.dirname(directory) != \
                    self._ctx.scratch.attempt_root(retained):
                raise ArtifactError(ReasonCode.ARTIFACT_REJECTED)
            content = ArtifactSlot.parse(entry["content"])
            verified = self._wrap(job, retained, slot, name, directory, entry["bytes"],
                                  entry["sha256"], writers_reaped=True, content=content)
            if slot not in _MEDIA_SLOTS:
                verified.resolve()
            else:
                self._media[job.job_id] = verified
            self._ctx.scratch.register(name, directory)
            result[slot] = (name, verified)
        self._remember_identity(job, record, retained)
        return result

    def discard(self, job: Any, attempt: Any) -> None:
        root = self._ctx.scratch.attempt_root(attempt.attempt_id)
        if os.path.isdir(root) and not os.path.islink(root):
            shutil.rmtree(root, ignore_errors=True)

    @staticmethod
    def _keeps_scratch_media(job: Any, verified: _Verified, slot: Any) -> bool:
        """True when ASR still needs this file: the download of a remote combined job."""
        request = job.request
        return (slot is ArtifactSlot.ORIGINAL_VIDEO and verified.slot in _MEDIA_SLOTS
                and request.source.kind != "local" and bool(request.outputs.transcript))

    async def publish(self, job: Any, attempt: Any, verified: Any, slot: ArtifactSlot) -> Any:
        from . import artifacts
        from .volumes import nearest_existing_ancestor

        video_ext = "mp4"
        if type(verified) is _LocalVideo:
            verified, video_ext = await asyncio.to_thread(self._copy_local_video, verified)
        if type(verified) is not _Verified:
            raise artifacts.ArtifactError(ReasonCode.ARTIFACT_REJECTED)
        file = await asyncio.to_thread(verified.resolve)
        root = job.request.output_root
        await asyncio.to_thread(os.makedirs, root, exist_ok=True)

        def output_volume() -> str:
            with nearest_existing_ancestor(root) as held:
                return held.identity.volume

        same = await asyncio.to_thread(output_volume) == file.identity.volume
        # A remote combined job transcribes from the scratch download it publishes as the
        # video.  On the same volume the publication would move that file away before ASR
        # starts, so it is copied instead: the cross-volume path, admitted and budgeted by
        # the same started guard.  The scratch copy stays until the attempt is discarded.
        copy = same and self._keeps_scratch_media(job, verified, slot)
        guard = None
        if not same or copy:
            limits = _limits_for(self._ctx, job.request)
            guard = await _started_guard(self._ctx, root, file.bytes,
                                         timeout=float(limits.download_timeout_seconds))
        try:
            published = await artifacts.publish(file, slot, job_id=job.job_id, output_root=root,
                                                guard=guard, audit=self._ctx.hub.artifact_audit(),
                                                video_ext=video_ext,
                                                same_volume=_never_same if copy else None)
        finally:
            if guard is not None:
                await guard.finish()
        self._published[published.artifact_id] = (published, root)
        while len(self._published) > 4096:
            self._published.popitem(last=False)
        self._roots[job.job_id] = root
        return published.to_artifact(attempt.attempt_id, verified.provenance)

    def _published_entry(self, job_dir: str, artifact: Any, listed: Mapping[str, Any]) -> Any:
        """The PublishedArtifact for one artifact: remembered, or re-proved from the sidecar."""
        from .artifacts import ArtifactError, PublishedArtifact
        from .volumes import open_contained

        remembered = self._published.get(artifact.artifact_id)
        if remembered is not None:
            return remembered[0]
        entry = listed.get(artifact.artifact_id)
        name = entry.get("file_name") if type(entry) is dict else None
        if type(name) is not str or os.path.basename(name) != name or not name:
            raise ArtifactError(ReasonCode.PUBLICATION_FAILED)
        with open_contained(job_dir, name) as held:
            identity = held.identity
        return PublishedArtifact(artifact.artifact_id, artifact.slot, artifact.format,
                                 artifact.bytes, artifact.sha256, name, identity)

    def write_manifest(self, job_id: str, attempt_id: str, published: list[Any],
                       provenance: Any) -> Any:
        from .artifacts import MANIFEST_NAME, write_manifest

        root = self._roots.get(job_id)
        if root is None:
            job = self._ctx.repo.get_job(job_id)
            if job is None:
                from .artifacts import ArtifactError

                raise ArtifactError(ReasonCode.PUBLICATION_FAILED)
            root = job.request.output_root
        job_dir = os.path.join(root, f"podmuncher-{job_id}")
        listed: dict[str, Any] = {}
        if any(a.artifact_id not in self._published for a in published):
            existing = _read_json_bounded(os.path.join(job_dir, MANIFEST_NAME)) or {}
            for item in existing.get("artifacts") or ():
                if type(item) is dict and type(item.get("artifact_id")) is str:
                    listed[item["artifact_id"]] = item
        items = [self._published_entry(job_dir, artifact, listed) for artifact in published]
        return write_manifest(job_dir, job_id, attempt_id, items, provenance)


# ----------------------------------------------------------------------------- exporter


class _Exporter:
    """``scheduler.Exporter``: validated record -> host-stamped exports in private scratch."""

    def __init__(self, ctx: _Context) -> None:
        self._ctx = ctx

    async def export(self, segments: Any, slots: frozenset[ArtifactSlot]) -> dict[ArtifactSlot, _ExportRef]:
        return await asyncio.to_thread(self._export, segments, slots)

    def _stamp_source(self, segments: _Verified, record: Any) -> tuple[str, str]:
        """(image digest, model revision) from host knowledge only; refuse otherwise."""
        from .exports import ExportError

        prepared = self._ctx.models.prepared(segments.job_id) if self._ctx.models else None
        if prepared is not None:
            snapshot_id, capability = prepared
        else:  # export retry after a restart: the record's snapshot must be a cached one
            snapshot_id = record.provenance.model_snapshot_id
            known = {s.snapshot_id for s in self._ctx.cache.snapshots()} if self._ctx.cache else set()
            if snapshot_id not in known:
                raise ExportError(ReasonCode.ARTIFACT_REJECTED)
            capability = Capability.ASR_CUDA if record.provenance.device == "cuda" else Capability.ASR_CPU
        entry = self._ctx.capability(capability)
        if entry is None or record.provenance.model_snapshot_id != snapshot_id:
            raise ExportError(ReasonCode.ARTIFACT_REJECTED)
        return entry.digest, snapshot_id

    def _export(self, segments: Any, slots: frozenset[ArtifactSlot]) -> dict[ArtifactSlot, _ExportRef]:
        from . import exports
        from .snapshots import Provenance
        from .volumes import open_contained

        if type(segments) is not _Verified or segments.slot is not WorkerSlot.TRANSCRIPT_SEGMENTS:
            raise exports.ExportError(ReasonCode.ARTIFACT_REJECTED)
        file = segments.resolve()
        job = self._ctx.repo.get_job(segments.job_id)
        if job is None:
            raise exports.ExportError(ReasonCode.ARTIFACT_REJECTED)
        cap = _limits_for(self._ctx, job.request).export_bytes
        raw = bytearray()
        with open_contained(file.attempt_dir, file.basename, expected=file.identity) as held:
            while True:
                chunk = held.read(1024 * 1024)
                if not chunk:
                    break
                raw.extend(chunk)
                if len(raw) > max(cap, file.bytes):
                    raise exports.ExportError(ReasonCode.EXPORT_LIMIT_REACHED)
        if len(raw) != file.bytes or hashlib.sha256(raw).hexdigest() != file.sha256:
            raise exports.ExportError(ReasonCode.ARTIFACT_REJECTED)
        record = exports.parse_record(bytes(raw), max_bytes=max(cap, file.bytes))
        digest, revision = self._stamp_source(segments, record)
        stamped = exports.with_host_provenance(record, image_digest=digest, model_revision=revision)
        if stamped.provenance.image_digest is None or stamped.provenance.model_revision is None:
            raise exports.ExportError(ReasonCode.ARTIFACT_REJECTED)  # never an unstamped export
        outputs = exports.export_all(stamped, slots, cap_bytes=cap)
        sidecar = json.loads(exports.provenance_sidecar(stamped, outputs).decode("utf-8"))
        for slot, data in outputs.items():
            entry = sidecar["outputs"][slot.value]
            if entry["bytes"] != len(data) or entry["sha256"] != hashlib.sha256(data).hexdigest():
                raise exports.ExportError(ReasonCode.ARTIFACT_REJECTED)
        source = sidecar["provenance"]
        provenance = Provenance(source["model_id"], source["model_revision"], source["image_digest"],
                                source["device"], source["compute_type"], source["source_sha256"],
                                source["attempt_id"])
        directory = self._ctx.scratch.new_dir(job.attempt_id or segments.job_id)
        refs: dict[ArtifactSlot, _ExportRef] = {}
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
        for slot, data in outputs.items():
            name = f"{slot.value}-{uuid.uuid4().hex}.bin"
            with os.fdopen(os.open(os.path.join(directory, name), flags, 0o600), "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            self._ctx.scratch.register(name, directory)
            refs[slot] = _ExportRef(directory, name, len(data), hashlib.sha256(data).hexdigest(),
                                    provenance)
        return refs


# ------------------------------------------------------------------------------- models


@dataclass(frozen=True)
class _TrackedChoice:
    job_id: str
    inner: Any


class _TrackedModels:
    """Wraps any ``scheduler.ModelProvider`` and records the host-prepared snapshot per job."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self._prepared: OrderedDict[str, tuple[str, str]] = OrderedDict()
        self._lock = threading.Lock()

    def select(self, job: Any, *, facts: Any, lower_than: object | None = None) -> _TrackedChoice:
        inner_lower = lower_than.inner if type(lower_than) is _TrackedChoice else lower_than
        return _TrackedChoice(job.job_id, self._inner.select(job, facts=facts, lower_than=inner_lower))

    def capability(self, choice: _TrackedChoice) -> str:
        return self._inner.capability(choice.inner)

    def selection(self, choice: _TrackedChoice) -> tuple[str, str]:
        return self._inner.selection(choice.inner)

    async def prepare(self, choice: _TrackedChoice) -> str:
        snapshot = await self._inner.prepare(choice.inner)
        with self._lock:
            self._prepared[choice.job_id] = (str(snapshot), self._inner.capability(choice.inner))
            while len(self._prepared) > MAX_KEYS:
                self._prepared.popitem(last=False)
        return snapshot

    def prepared(self, job_id: str) -> tuple[str, str] | None:
        with self._lock:
            return self._prepared.get(job_id)


def _asr_candidates(ctx: _Context, device: str) -> tuple[Capability, ...]:
    """ASR capabilities to probe, in order.  One manifest serves both backends (SPEC_GAP-16):
    on ``auto`` CUDA is probed first when the manifest has an ``asr_cuda`` entry, and the CPU
    candidate is the fallback.  A usable GPU is checked by the probe port before any CUDA
    worker starts, so a host without one lands on CPU without a CUDA worker."""
    if device == "cuda":
        return (Capability.ASR_CUDA,)
    if device == "cpu":
        return (Capability.ASR_CPU,)
    present = tuple(c for c in (Capability.ASR_CUDA, Capability.ASR_CPU) if ctx.capability(c) is not None)
    return present or (Capability.ASR_CPU,)


# Host and engine identity the setup steps show the operator (SPEC_GAP-17): e.g. "Apple M4
# (arm64), Docker arm64".  Read from ``hardware.HardwareEvidence``; None when unmeasured.
_HOST_IDENTITY_FIELDS = ("host_os", "host_arch", "host_cpu_model", "engine_arch")


def _measured_text(sample: Any, name: str) -> str | None:
    """A measured (status OK) string field of a hardware sample, else None."""
    field = getattr(sample, name, None)
    value = getattr(field, "value", None)
    if getattr(field, "status", None) is not FieldStatus.OK or type(value) is not str or not value:
        return None
    return value


def _host_identity(hardware: Mapping[str, Any]) -> dict[str, str | None]:
    """The identity keys of a ``probe_hardware`` result, string-valued or None."""
    return {name: hardware.get(name) if type(hardware.get(name)) is str else None
            for name in _HOST_IDENTITY_FIELDS}


def _supported_candidates(ctx: _Context, candidates: tuple[Capability, ...],
                          engine_arch: str | None) -> tuple[Capability, ...]:
    """The candidates whose image runs on the Docker engine's architecture (SPEC_GAP-17).

    ``engine_arch`` is the measured ``docker info`` architecture (``amd64``/``arm64``, or
    the raw ``x86_64``/``aarch64``); ``manifests.supports_engine_arch`` maps it onto the
    entry's ``platforms``.  A development fixture is never gated: its platform is a fixture
    id, not an architecture (``worker_policy.development_image`` is set from the same
    ``development_fixture`` flag).  An unmeasured engine gates nothing: the probe then
    fails closed on its own Docker evidence.  On an arm64 engine ``asr_cuda``
    (linux-amd64 only) is dropped here, before any GPU check, so it is never probed.
    A candidate without a manifest entry stays: the caller refuses it as unavailable.
    """
    from .manifests import supports_engine_arch

    manifest = ctx.manifest
    if engine_arch is None or manifest is None or manifest.development_fixture:
        return candidates
    return tuple(c for c in candidates
                 if ctx.capability(c) is None or supports_engine_arch(ctx.capability(c), engine_arch))


def _model_options(request: Any, evidence: Any, resolved: Any) -> list[dict[str, Any]]:
    """Setup guidance (SPEC_GAP-16, owner rule 7): every verified model, highest rank first.

    Each option carries the requirement the policy would judge it by on the probed backend
    (an exact measured envelope, else the shipped table row; ``None`` when unknown), its
    basis and source, and ``fits``: whether the live evidence fits it.  Fit comes from the
    real policy, walked down ``lower_than`` from the automatic pick, never re-implemented
    here.  Runs inside the probe hold window, where the evidence is fresh.  Bounded by the
    manifest's model list; JSON-safe values only.
    """
    from .domains import Backend
    from .model_policy import ModelPolicyError, load_hardware_table, select_model

    backend = Backend.CUDA if resolved.capability is Capability.ASR_CUDA else Backend.CPU
    table = load_hardware_table()
    tiers = tuple(ModelTier)
    fits: set[str] = set()
    lower = None
    auto = replace(request, model="auto")
    while True:
        try:
            lower = select_model(auto, evidence, resolved.envelopes, lower_than=lower)
        except ModelPolicyError:
            break
        fits.add(lower.artifact.model_id)

    def rank(artifact: Any) -> int:
        # The policy's order: the table rank, else the tier (model_policy._rank).
        row = table.row(artifact.model_id)
        return row.rank if row is not None else (tiers.index(artifact.tier) + 1) * 10

    options = []
    for artifact in sorted(evidence.artifacts, key=lambda a: (rank(a), a.model_id), reverse=True):
        row = table.row(artifact.model_id)
        need = row.requirement(backend) if row is not None else None
        option: dict[str, Any] = {"id": artifact.model_id, "tier": artifact.tier.value,
                                  "rank": rank(artifact), "backend": backend.value}
        try:
            judged = select_model(replace(request, model=artifact.model_id, device=backend.value),
                                  evidence, resolved.envelopes)
        except ModelPolicyError:
            judged = None
        if judged is not None:
            option["compute_type"] = judged.compute_type
            if backend is Backend.CUDA:
                option["peak_vram_bytes"] = judged.peak_vram_bytes
            option["peak_ram_bytes"] = judged.peak_ram_bytes
            measured = judged.measured_envelope is not None
            option["basis"] = "measured" if measured else need.basis if need is not None else "unknown"
            option["source"] = judged.source
        else:
            option["compute_type"] = need.compute_type if need is not None else None
            if backend is Backend.CUDA:
                option["peak_vram_bytes"] = need.peak_vram_bytes if need is not None else None
            option["peak_ram_bytes"] = need.peak_ram_bytes if need is not None else None
            option["basis"] = need.basis if need is not None else "unknown"
            option["source"] = need.source if need is not None else "unknown"
        option["fits"] = artifact.model_id in fits
        options.append(option)
    return options


async def _choose_with_probe(ctx: _Context, probe: Any, candidates: tuple[Capability, ...], *,
                             duration: Any, selection: Any, output_mode: Any,
                             duration_limit: float, lower_than: Any = None,
                             job_id: str | None = None, attempt_id: str | None = None,
                             guidance: dict[str, Any] | None = None,
                             scratch: Any = None, manual_backing: str | None = None) -> tuple[Any, Any]:
    """Probe each candidate capability in turn and select inside its hold window.

    Returns ``(ModelChoice, ResolvedCapability)``.  On ``device = auto`` a CUDA candidate
    that fails for a GPU/envelope reason falls back to the CPU candidate; storage, Docker,
    cancellation and duration failures stop immediately with their own closed reason.
    ``scratch`` (setup only) is the ``_ScratchIndex`` the probe worker mounts when the
    wizard is provisioning another scratch path; None is the service's configured scratch.

    ``guidance`` (setup only) receives, for the last probe whose evidence was collected,
    ``resolved``, ``options`` (``_model_options``) and ``failure``: the policy error raised
    by that selection, or None.  A raised error identical to ``failure`` means the probe
    succeeded and no model fits automatically; the operator may still choose one.  It also
    receives ``engine_arch`` (the measured engine architecture, or None), ``skipped``
    (candidates whose image does not run on it) and ``unsupported`` (True when nothing
    could be probed for that reason).

    Engine architecture (SPEC_GAP-17): candidates whose image does not run on the Docker
    engine are skipped (``_supported_candidates``).  When none is left, an explicit CUDA
    request is refused as ``gpu_unavailable`` (there is no CUDA image for this engine),
    anything else as ``service_unavailable``.
    """
    from .domains import OutputMode
    from .model_policy import ModelPolicyError, PolicyRequest, select_model

    if OutputMode.parse(output_mode) is OutputMode.VIDEO:
        raise ModelPolicyError(ReasonCode.CONFLICT)  # no model is ever needed for video only
    if duration is None:
        raise ModelPolicyError(ReasonCode.DURATION_UNKNOWN)  # never start a worker to learn this
    engine_arch = None
    if ctx.manifest is not None and not ctx.manifest.development_fixture:
        engine_arch = await probe.engine_arch()
    supported = _supported_candidates(ctx, candidates, engine_arch)
    if guidance is not None:
        guidance.update(engine_arch=engine_arch, unsupported=not supported,
                        skipped=[c.value for c in candidates if c not in supported])
    if not supported:
        raise ModelPolicyError(ReasonCode.GPU_UNAVAILABLE if all(c is Capability.ASR_CUDA for c in candidates)
                               else ReasonCode.SERVICE_UNAVAILABLE)
    candidates = supported
    failure: ModelPolicyError | None = None
    for capability in candidates:
        entry = ctx.capability(capability)
        if entry is None:
            raise ModelPolicyError(ReasonCode.SERVICE_UNAVAILABLE)
        request = PolicyRequest(duration, entry.worker_memory_bytes, device=selection.device,
                                profile=selection.profile, model=selection.model,
                                output_mode=output_mode, duration_limit_seconds=duration_limit,
                                decode_timeout_seconds=float(ctx.config.limits.decode_timeout_seconds))
        if request.duration_seconds > request.duration_limit_seconds:
            raise ModelPolicyError(ReasonCode.DURATION_EXCEEDED)
        chosen: dict[str, Any] = {}

        def use(evidence: Any, resolved: Any, *, request: Any = request,
                chosen: dict[str, Any] = chosen) -> None:
            # Called while the probe container is still held: the evidence is seconds old.
            if guidance is not None:
                guidance.update(resolved=resolved, failure=None,
                                options=_model_options(request, evidence, resolved))
            try:
                chosen["choice"] = select_model(request, evidence, resolved.envelopes,
                                                lower_than=lower_than)
            except ModelPolicyError as error:
                if guidance is not None:
                    guidance["failure"] = error
                raise
            chosen["resolved"] = resolved

        extra = {"scratch": scratch} if scratch is not None else {}
        if manual_backing is not None:
            extra["manual_backing"] = manual_backing
        try:
            await probe.collect(capability, use=use, job_id=job_id, attempt_id=attempt_id, **extra)
        except ModelPolicyError as error:
            fallback = (capability is Capability.ASR_CUDA and len(candidates) > 1
                        and error.reason_code not in _NO_FALLBACK_REASONS)
            if not fallback:
                raise
            failure = error
            continue
        return chosen["choice"], chosen["resolved"]
    raise failure if failure is not None else ModelPolicyError(ReasonCode.ENVELOPE_UNVALIDATED)


@dataclass
class _Deferred:
    """``_ManifestModels.select``'s token: the request is recorded, nothing is probed yet.

    ``prepare`` fills ``choice``/``resolved``/``snapshot`` exactly once; ``capability`` and
    ``selection`` are only valid after that (the scheduler reads them in TRANSCRIBING).
    """

    job: Any
    facts: Any
    lower_than: _Deferred | None = None
    choice: Any = None
    resolved: Any = None
    snapshot: str | None = None


class _ManifestModels:
    """Real ``scheduler.ModelProvider``: probe evidence + manifest envelopes + ``select_model``.

    ``select`` runs synchronously on the loop in PREPARING, so it only returns a deferred
    token.  ``prepare`` (awaited by the scheduler) runs the ASR probe through
    ``_ProbeEvidence``, calls ``select_model`` inside the hold window, acquires the chosen
    model's verified snapshot in a thread and returns the snapshot id.  ``lower_than`` is
    the previous attempt's token, so the OOM fallback picks a strictly smaller model.
    """

    def __init__(self, ctx: _Context, probe: Any) -> None:
        self._ctx = ctx
        self._probe = probe

    def select(self, job: Any, *, facts: Any, lower_than: object | None = None) -> _Deferred:
        from .model_policy import ModelPolicyError

        ctx = self._ctx
        if ctx.manifest is None or all(ctx.capability(c) is None
                                       for c in (Capability.ASR_CPU, Capability.ASR_CUDA)):
            raise ModelPolicyError(ReasonCode.SERVICE_UNAVAILABLE)
        if lower_than is not None and (type(lower_than) is not _Deferred or lower_than.choice is None):
            raise ModelPolicyError(ReasonCode.MODEL_OOM)  # a fallback needs the proven prior choice
        return _Deferred(job, facts, lower_than)

    @staticmethod
    def _resolved(choice: Any) -> _Deferred:
        from .model_policy import ModelPolicyError

        if type(choice) is not _Deferred or choice.choice is None or choice.resolved is None:
            raise ModelPolicyError(ReasonCode.CONFLICT)  # used before prepare: never guess
        return choice

    def capability(self, choice: Any) -> str:
        return self._resolved(choice).resolved.capability.value

    def selection(self, choice: Any) -> tuple[str, str]:
        chosen = self._resolved(choice).choice
        return chosen.backend.value, chosen.artifact.model_id

    async def prepare(self, choice: Any) -> str:
        from .model_cache import CacheError
        from .model_policy import ModelPolicyError

        ctx = self._ctx
        if type(choice) is not _Deferred or choice.choice is not None:
            raise ModelPolicyError(ReasonCode.CONFLICT)  # one probe per token
        if ctx.cache is None or ctx.manifest is None or ctx.acquire_model is None:
            raise CacheError(ReasonCode.CACHE_UNKNOWN)
        job, facts = choice.job, choice.facts
        selection = job.request.selection
        lower = choice.lower_than.choice if choice.lower_than is not None else None
        chosen, resolved = await _choose_with_probe(
            ctx, self._probe, _asr_candidates(ctx, selection.device),
            duration=facts.duration_seconds or None, selection=selection,
            output_mode=job.request.output_mode,
            duration_limit=float(job.request.limits.duration_seconds), lower_than=lower,
            job_id=job.job_id, attempt_id=job.attempt_id)
        model = next((m for m in ctx.manifest.models if m.artifact == chosen.artifact), None)
        if model is None:
            raise CacheError(ReasonCode.ASSET_VERIFICATION_FAILED)
        snapshot = await asyncio.to_thread(ctx.acquire_model, model)
        choice.choice, choice.resolved, choice.snapshot = chosen, resolved, snapshot.snapshot_id
        return snapshot.snapshot_id


# ------------------------------------------------------------------- probe evidence port


def _host_gpu_uuid(timeout: float = 10.0) -> str | None:
    """The first host GPU's UUID from ``nvidia-smi``, or None (never raises)."""
    flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    try:
        result = subprocess.run(("nvidia-smi", "--id=0", "--query-gpu=uuid",
                                 "--format=csv,noheader,nounits"), capture_output=True,
                                timeout=timeout, check=False, stdin=subprocess.DEVNULL, **flags)
        value = result.stdout.decode("ascii", "replace").strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return value if not result.returncode and _GPU_UUID_RE.fullmatch(value) else None


def _write_tone_wav(path: str, seconds: float = SMOKE_FIXTURE_SECONDS) -> int:
    """A generated 16 kHz mono PCM tone burst between silences (no recording); its size."""
    import wave

    rate = 16000
    total = int(seconds * rate)
    frames = bytearray()
    for index in range(total):
        active = rate // 4 <= index < total - rate // 4
        value = int(3000 * math.sin(2 * math.pi * 440 * index / rate)) if active else 0
        frames += value.to_bytes(2, "little", signed=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    with os.fdopen(os.open(path, flags, 0o600), "wb") as handle:
        with wave.open(handle, "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(rate)
            writer.writeframes(bytes(frames))
        handle.flush()
        os.fsync(handle.fileno())
    return os.path.getsize(path)


async def _inspect_image(cli: Any, image: str) -> Any:
    """One bounded, read-only ``docker image inspect`` through the service's CLI port."""
    return await cli.run("image", "inspect", image, owner=f"inspect-{uuid.uuid4().hex}", timeout=15)


async def _image_listed(cli: Any, image: str) -> bool:
    """Is the release image ``image`` (``repo@digest``) verified present (``RepoDigests``)?"""
    result = await _inspect_image(cli, image)
    return _repo_digest_listed(result.returncode, result.stdout, image)


async def _require_image(cli: Any, image: str, *, development: bool) -> None:
    """The exact image must already be present locally (never a pull), else a closed failure.

    A development image is a local image id: its ``Id`` must match.  A release image is a
    pulled ``repo@digest`` reference: its ``Id`` is the config digest, which never equals
    the pulled index digest, so presence is proven by ``RepoDigests`` listing the exact
    reference (``_repo_digest_listed``).
    """
    from .runtime import RuntimeFailure

    result = await _inspect_image(cli, image)
    if not development:
        if not _repo_digest_listed(result.returncode, result.stdout, image):
            raise RuntimeFailure(ReasonCode.SERVICE_UNAVAILABLE)
        return
    try:
        found, = json.loads(result.stdout)
        if result.returncode or found["Id"] != image:
            raise ValueError
    except (ValueError, TypeError, KeyError):
        raise RuntimeFailure(ReasonCode.SERVICE_UNAVAILABLE) from None


# ------------------------------------------------------------------ release image pulls


def _release_image(entry: Any) -> str:
    """The pulled reference of a packaged-release capability: ``image_reference@digest``
    (the same value ``manifests.resolve_capability`` puts in ``worker_policy.image``)."""
    return f"{entry.image_reference}@{entry.digest}"


def _repo_digest_listed(returncode: int, stdout: bytes, image: str) -> bool:
    """True when ``docker image inspect`` output proves ``image`` (``repo@digest``) is local:
    the command succeeded, exactly one image was listed and its ``RepoDigests`` names that
    reference.  The image ``Id`` is the config digest, never the pulled index digest, so
    it is not what a release reference is checked against."""
    try:
        found, = json.loads(stdout)
        digests = found["RepoDigests"]
    except (ValueError, TypeError, KeyError):
        return False
    return not returncode and type(digests) is list and image in digests


def _release_image_present(image: str, timeout: float = IMAGE_INSPECT_SECONDS) -> bool:
    """Bounded, read-only: is the release image ``image`` (``repo@digest``) verified present?

    Blocking (``docker image inspect`` against the local context): call it off the loop.
    Any Docker failure is "not verified present", never a pull and never an exception.
    Nothing is cached: an image pruned after a pull must stop being credited at the next
    admission (its pull is then budgeted again), so every admission asks the daemon.
    """
    flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    try:
        target = resolve_docker_target(timeout)
        result = subprocess.run(("docker", "--host", target.endpoint, "image", "inspect", image),
                                capture_output=True, timeout=timeout, check=False,
                                stdin=subprocess.DEVNULL, **flags)
    except (DomainValidationError, OSError, subprocess.SubprocessError, ValueError):
        return False
    return _repo_digest_listed(result.returncode, result.stdout, image)


def _no_pull_policy(policy: Any) -> Any:
    """``policy`` for a run whose release image was verified present just before it.

    The runtime pulls under ``allow_pull`` before every run, so a present image would
    still be pulled and an unreachable registry (offline, which setup's outcome must
    survive) would fail the job.  With presence verified, the run is launched without a
    pull; ``cached_proof`` re-verifies that presence (read-only, ``RepoDigests``) should
    the runtime ask for it.  Only an absent image pulls.  A development image, or a
    policy that already does not pull, is returned unchanged.
    """
    if policy.development_image or not policy.allow_pull:
        return policy
    image = policy.image

    def present() -> bool:
        return _release_image_present(image)

    return replace(policy, allow_pull=False, cached_proof=present)


def _image_growth(compressed: int, unpacked: int) -> Any:
    """Workload growth of one image pull: its compressed download and unpacked peak."""
    from .admission import WorkloadGrowth

    return WorkloadGrowth(compressed, unpacked, 0, 0, (0,), 0, 0, 0, 0, (), 0, MIB)


class _SetupPull:
    """The setup pull's registered operation (``guards.OwnedOperation``).

    ``runtime._PullRecovery``'s semantics, with one difference: a pull whose CLI exited
    with any non-negative status is resolved, because the daemon reported the end of its
    pull (a registry or network failure is then a closed reason, not an unresolved
    writer).  A CLI that ended by signal (a negative status), raised, was cancelled or
    never reported completion is unresolved: the daemon's pull may still be running.  An
    interrupted or unfinished pull keeps its lease, as the runtime's does: a daemon pull
    has no cancellation API.
    """

    def __init__(self, owner: str, image: str) -> None:
        from .guards import OperationBinding

        self._binding = OperationBinding(owner, "pull:" + image)
        self.task: asyncio.Task | None = None

    @property
    def binding(self) -> Any:
        return self._binding

    async def terminate(self, expected: Any) -> None:
        from .runtime import RuntimeFailure

        if expected is not self._binding:
            raise RuntimeFailure(ReasonCode.WRITER_UNRESOLVED)

    async def kill(self, expected: Any) -> None:
        await self.terminate(expected)

    async def reap(self, expected: Any) -> bool:
        await self.terminate(expected)
        if self.task is None:
            return True
        if not self.task.done() or self.task.cancelled():
            return False
        try:
            code = self.task.result().returncode
        except Exception:  # noqa: BLE001 - a raised CLI run never reported the pull's end
            return False
        return type(code) is int and code >= 0  # a signal exit (negative) is unresolved


async def _guarded_pull(cli: Any, guard: Any, image: str) -> dict[str, Any]:
    """Pull one release image under ``guard``, then prove it carries ``image``'s digest.

    The guard's initial sample is the admission (a short Docker disk fails here, before
    any pull) and its watchdog runs throughout.  The pull IO is registered with the
    coordinator as ``DockerRuntime`` registers its own (``_SetupPull``): a cancelled or
    timed-out pull keeps its lease until the daemon's pull is resolved.  After the pull,
    ``docker image inspect`` must list ``image`` in ``RepoDigests``; anything else fails
    closed.  Every failure is a ``RuntimeFailure`` with a closed reason: a nonzero pull is
    ``upstream_unavailable`` (the daemon answered; the registry or the network did not).
    """
    from .runtime import RuntimeFailure, _reason

    coordinator = guard.coordinator
    owner = coordinator.operation_id
    reason = ReasonCode.OK
    try:
        await guard.start()
        cli.bind_owner(owner, coordinator)
        pull = _SetupPull(owner, image)
        coordinator.register_operation(pull)
        pull.task = coordinator.track(cli.run("pull", image, owner=owner,
                                              timeout=guard.timeout_seconds, coordinator=coordinator))
        pulled = await asyncio.shield(pull.task)
        if pulled.returncode:
            raise RuntimeFailure(ReasonCode.UPSTREAM_UNAVAILABLE)
        await guard.transition()
        inspected = await guard.io(lambda: cli.run("image", "inspect", image, owner=owner,
                                                   coordinator=coordinator))
        if not _repo_digest_listed(inspected.returncode, inspected.stdout, image):
            raise RuntimeFailure(ReasonCode.ASSET_VERIFICATION_FAILED)
        await guard.transition()
    except BaseException as error:
        reason = _reason(error)
        if reason is ReasonCode.OK:
            reason = ReasonCode.DOCKER_UNAVAILABLE
    finally:
        stopped = await guard.finish(reason)
        if stopped.reason is not ReasonCode.OK:
            reason = stopped.reason
    if reason is not ReasonCode.OK:
        raise RuntimeFailure(reason) from None
    return {"present": True, "pulled": True}


def _read_claimed(directory: str, name: str, claim: Mapping[str, Any], slot: WorkerSlot,
                  max_bytes: int) -> bytes:
    """Contained, size-capped read of a worker's claimed output; the claim must match."""
    from .model_policy import ModelPolicyError
    from .volumes import open_contained

    size, sha256 = claim.get("bytes"), claim.get("sha256")
    if (claim.get("slot") != slot.value or type(size) is not int or not 0 < size <= max_bytes
            or type(sha256) is not str):
        raise ModelPolicyError(ReasonCode.ENVELOPE_UNVALIDATED)
    raw = bytearray()
    try:
        with open_contained(directory, name) as held:
            while len(raw) <= max_bytes:  # bounded: one chunk past the cap proves oversize
                chunk = held.read(min(CHUNK_BYTES, max_bytes + 1 - len(raw)))
                if not chunk:
                    break
                raw.extend(chunk)
    except (DomainValidationError, OSError, ValueError):
        raise ModelPolicyError(ReasonCode.ENVELOPE_UNVALIDATED) from None
    if len(raw) != size or hashlib.sha256(raw).hexdigest() != sha256:
        raise ModelPolicyError(ReasonCode.ENVELOPE_UNVALIDATED)
    return bytes(raw)


def _parse_probe_artifact(raw: bytes, device: str, gpu_uuid: str | None) -> tuple[str, tuple[str, ...]]:
    """Strict parse of the worker's probe JSON: ``(runtime version, compute types)``.

    The worker wrote this file; after the sha check it is trusted only in this exact shape
    (schema 1, known short compute-type labels, a label-shaped version, the expected device).
    """
    from .model_policy import ModelPolicyError

    try:
        doc = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise ModelPolicyError(ReasonCode.ENVELOPE_UNVALIDATED) from None
    if type(doc) is not dict or doc.get("schema_version") != 1:
        raise ModelPolicyError(ReasonCode.ENVELOPE_UNVALIDATED)
    runtime = doc.get("runtime")
    version = runtime.get("version") if type(runtime) is dict else None
    types = doc.get("compute_types")
    if (type(version) is not str or _VERSION_RE.fullmatch(version) is None
            or type(types) is not list or not 1 <= len(types) <= 16
            or any(type(t) is not str or _COMPUTE_TYPE_RE.fullmatch(t) is None for t in types)
            or len(set(types)) != len(types)
            or doc.get("device") != device or doc.get("cuda_device_uuid") != gpu_uuid):
        raise ModelPolicyError(ReasonCode.ENVELOPE_UNVALIDATED)
    return version, tuple(types)


def _evidence_age(hardware: Any, now: datetime) -> float:
    """Age in seconds of the oldest reading ``select_model`` will consume."""
    ages = []
    for name in ("worker_ram_limit", "worker_ram_current", "worker_ram_available",
                 "worker_gpu_visible", "runtime_compute_types"):
        field = getattr(hardware, name)
        stamp = datetime.fromisoformat(field.sample_time)
        ages.append((now - stamp).total_seconds())
    return max(ages)


def _worker_readings_ok(hardware: Any, compute_types: tuple[str, ...]) -> bool:
    fields = ("worker_ram_limit", "worker_ram_current", "worker_ram_available", "worker_gpu_visible",
              "runtime_compute_types")
    if any(getattr(hardware, name).status is not FieldStatus.OK for name in fields):
        return False
    return tuple(hardware.runtime_compute_types.value) == compute_types


class _ProbeRuntimeCapabilities:
    """``hardware.RuntimeCapabilities`` adapter: compute types from the parsed probe artifact."""

    def __init__(self, identity: Any, backend: Any, compute_types: tuple[str, ...]) -> None:
        self.identity, self._backend, self._types = identity, backend, compute_types

    def compute_types(self, backend: Any, *, deadline: float) -> Any:
        from .domains import Backend
        from .hardware import RuntimeComputeEvidence

        if Backend.parse(backend) is not self._backend:
            raise ValueError("backend mismatch")
        gpu = self.identity.gpu_uuid if self._backend is Backend.CUDA else None
        return RuntimeComputeEvidence(self.identity, self._backend, self._types, gpu)


class _Launched:
    """One started single-use worker run: its task, admitted reference and resource close."""

    def __init__(self, runtime: Any, task: asyncio.Task, close: Callable[[], Any]) -> None:
        self.runtime, self.task, self._close = runtime, task, close

    @property
    def reference(self) -> Any:
        return self.runtime.reference

    def obsolete(self) -> None:
        self.runtime.obsolete()

    async def close(self) -> None:
        await self._close()


class _DockerLaunch:
    """Real launch port for probe and smoke runs: a fresh ``runtime.DockerRuntime`` each.

    The guard/coordinator construction is the launcher's: a nested in-process lease, the
    prepared Docker storage session and the real volume sampler.  The container carries
    this instance's label, so startup reconciliation would reap a survivor.
    """

    def __init__(self, ctx: _Context, docker: _DockerAccess) -> None:
        self._ctx, self._docker = ctx, docker

    async def verify_image(self, image: str) -> None:
        """The probe/smoke image must be present: a development image by id, a release
        image by its ``repo@digest`` in ``RepoDigests`` (the manifest kind decides)."""
        manifest = self._ctx.manifest
        development = bool(manifest is not None and manifest.development_fixture)
        await _require_image(await self._docker.cli(), image, development=development)

    async def pull_image(self, image: str, *, growth: Any, timeout: float,
                         manual_backing: str | None = None) -> dict[str, Any]:
        """Admitted, guarded pull of one release image (``repo@digest``); see ``_guarded_pull``.

        A reference already verified present (read-only inspect, before any admission) is
        not pulled again: ``{"present": True, "pulled": False}``.  Otherwise the guard is
        built like a worker's: the prepared Docker storage session, ``RuntimeStorage`` with
        ``growth`` (the image's compressed and unpacked bytes) and a nested in-process lease.
        ``manual_backing`` is the Docker data location the admission verifies (setup's
        BACKING step, which may differ from the configured one); None is the configured one.
        """
        from .guards import CancellationCoordinator, WriteGuard
        from .runtime import RuntimeStorage

        ctx = self._ctx
        cli = await self._docker.cli()
        if await _image_listed(cli, image):
            return {"present": True, "pulled": False}
        limits = ctx.config.limits
        backing = ctx.config.paths.docker_backing if manual_backing is None else manual_backing
        session = await asyncio.to_thread(ctx.docker_storage.prepare, manual_backing=backing)
        try:
            paths = {"scratch": Path(ctx.config.paths.scratch), "cache": Path(ctx.config.paths.cache),
                     "output": Path(ctx.config.paths.output), "state": Path(ctx.state_dir)}
            storage = RuntimeStorage(session, ctx.volumes, paths=paths, growth=growth, limits=limits,
                                     sample_budget_seconds=WORKER_STORAGE_SAMPLE_SECONDS)
            coordinator = CancellationCoordinator(f"pull-{uuid.uuid4().hex}", _MemoryLease(),
                                                  stop_grace_seconds=limits.worker_stop_grace_seconds)
            guard = WriteGuard(coordinator, storage.sample, byte_limit=0,
                               timeout_seconds=float(timeout), sample_timeout_seconds=2.9)
            return await _guarded_pull(cli, guard, image)
        finally:
            await asyncio.to_thread(session.close)

    async def launch(self, policy: Any, mounts: Any, request: Any, on_event: Callable[[Any], None], *,
                     byte_limit: int, timeout: float, growth: Any,
                     manual_backing: str | None = None) -> _Launched:
        """One probe/smoke run.  ``manual_backing`` is the Docker data location the run's
        storage admission verifies (setup's BACKING step, which may differ from the
        configured one, as ``pull_image``); None is the configured one."""
        from .guards import CancellationCoordinator, WriteGuard
        from .runtime import DockerRuntime, RuntimeStorage

        ctx = self._ctx
        cli = await self._docker.cli()
        limits = ctx.config.limits
        backing = ctx.config.paths.docker_backing if manual_backing is None else manual_backing
        session = await asyncio.to_thread(ctx.docker_storage.prepare, manual_backing=backing)
        try:
            paths = {"scratch": mounts.scratch, "cache": Path(ctx.config.paths.cache),
                     "output": Path(ctx.config.paths.output), "state": Path(ctx.state_dir)}
            # The in-worker hardware sample (cgroup reader, docker stats) runs concurrently with
            # the guard's storage watchdog; the wider live budget keeps that watchdog honest.
            storage = RuntimeStorage(session, ctx.volumes, paths=paths, growth=growth, limits=limits,
                                     sample_budget_seconds=WORKER_STORAGE_SAMPLE_SECONDS)
            coordinator = CancellationCoordinator(f"probe-{uuid.uuid4().hex}", _MemoryLease(),
                                                  stop_grace_seconds=limits.worker_stop_grace_seconds)
            guard = WriteGuard(coordinator, storage.sample, byte_limit=byte_limit,
                               timeout_seconds=float(timeout), sample_timeout_seconds=2.9)
            runtime = DockerRuntime(cli, guard, instance=ctx.instance, policy=policy, mounts=mounts)
        except BaseException:
            await asyncio.to_thread(session.close)
            raise
        task = asyncio.get_running_loop().create_task(runtime.run(request, on_event=on_event))
        return _Launched(runtime, task, lambda: asyncio.to_thread(session.close))

    def sampler(self, reference: Any, adapter: Any) -> Any:
        from .hardware import DockerCgroupMemory, HardwareSampler

        # The host CPU window is not selection evidence: skipping its one-second wait keeps
        # the worker readings a second fresher inside the probe's 5 s hold.
        return HardwareSampler(worker=DockerCgroupMemory(reference), runtime=adapter,
                               timeout=PROBE_SAMPLE_SECONDS, sleep=lambda _seconds: None)


class _ProbeEvidence:
    """Same-worker ``PolicyEvidence`` from a short-lived ASR probe worker (SPEC_GAP-12).

    ``collect`` resolves the capability from the loaded manifest, requires the exact image
    digest locally (never a pull), runs the ``probe`` operation in a network-none container
    with only a fresh private scratch mount, reads the <= 4 KiB probe artifact contained and
    sha-checked, then samples the still-held container (``HardwareSampler`` over
    ``DockerCgroupMemory`` plus the compute-types adapter) in a thread.  A sample older
    than ``EVIDENCE_FRESH_SECONDS`` is taken once more while the hold lasts; when the hold
    is exceeded the probe fails closed (``envelope_unvalidated``), never fabricated.  The
    optional ``use`` callback (``select_model``) runs inside the hold window.  The
    container is reaped on every path by the runtime's own finalization; a failure also
    makes the run obsolete so the hold ends at once.  ``smoke`` runs one short real
    transcription of a generated fixture through the same launch port.
    """

    def __init__(self, ctx: _Context, launch: Any, *, hardware: Any = None,
                 gpu_uuid: Callable[[], str | None] | None = None) -> None:
        self._ctx, self._launch = ctx, launch
        self._hardware = hardware
        self._gpu_uuid = gpu_uuid or _host_gpu_uuid
        self._lock = asyncio.Lock()
        self._active: _Launched | None = None
        self._reaped = True

    @property
    def running(self) -> bool:
        return self._active is not None

    @staticmethod
    def _task_reaped(task: asyncio.Task) -> bool:
        """A finished runtime run has reaped its container unless it says otherwise."""
        if not task.done():
            return False
        if task.cancelled():
            return True
        error = task.exception()
        return error is None or _reason_of(error, ReasonCode.OK) is not ReasonCode.WRITER_UNRESOLVED

    @property
    def writers_reaped(self) -> bool:
        active = self._active
        return self._reaped if active is None else self._task_reaped(active.task)

    async def stop(self, reason: ReasonCode) -> bool:
        """Make any in-flight probe/smoke obsolete and wait (bounded) for its reaping."""
        active = self._active
        if active is None:
            return self._reaped
        active.obsolete()
        await asyncio.wait({active.task}, timeout=PROBE_END_SECONDS)
        return self._task_reaped(active.task)

    # -- helpers ---------------------------------------------------------------------
    async def engine_arch(self) -> str | None:
        """The Docker engine's measured architecture (``docker info``), else None.

        Sampled from the host hardware port off the loop; an unsampled or unmeasured
        engine is None, which gates no capability (``_supported_candidates``).
        """
        if self._hardware is None:
            return None
        try:
            sample = await asyncio.to_thread(self._hardware.sample)
        except Exception:  # noqa: BLE001 - an unsampled engine gates nothing
            return None
        return _measured_text(sample, "engine_arch")

    async def _usable_gpu(self) -> str | None:
        """A host GPU UUID when the hardware sample reports a usable GPU, else None."""
        if self._hardware is None:
            return None
        try:
            sample = await asyncio.to_thread(self._hardware.sample)
        except Exception:  # noqa: BLE001 - an unsampled host has no usable GPU
            return None
        for name in ("gpu_name", "gpu_driver", "gpu_vram_total", "gpu_vram_free"):
            field = getattr(sample, name, None)
            if field is None or getattr(field, "status", None) is not FieldStatus.OK:
                return None
        try:
            value = await asyncio.to_thread(self._gpu_uuid)
        except Exception:  # noqa: BLE001
            return None
        return value if type(value) is str and _GPU_UUID_RE.fullmatch(value) else None

    async def _resolve(self, capability: Capability | str) -> tuple[Any, Any, str | None]:
        """``(manifest entry, ResolvedCapability, gpu uuid)`` for one ASR capability."""
        from .manifests import resolve_capability
        from .model_policy import ModelPolicyError

        ctx = self._ctx
        wanted = Capability.parse(capability)
        entry = ctx.capability(wanted)
        if wanted is Capability.MEDIA or entry is None:
            raise ModelPolicyError(ReasonCode.SERVICE_UNAVAILABLE)
        gpu = None
        if wanted is Capability.ASR_CUDA:
            gpu = await self._usable_gpu()
            if gpu is None:
                raise ModelPolicyError(ReasonCode.GPU_UNAVAILABLE)
        try:
            resolved = resolve_capability(ctx.manifest, wanted, selected_gpu_uuid=gpu)
        except DomainValidationError as error:
            raise ModelPolicyError(error.reason_code) from None
        return entry, resolved, gpu

    def _request(self, capability: Capability, operation: str, slots: Mapping[str, str], *,
                 job_id: str, attempt_id: str, selection: Any, input_basename: str | None = None,
                 snapshot: str | None = None) -> Any:
        from .protocol import WorkerRequest
        from .requests import RequestOutputs

        return WorkerRequest(1, job_id, attempt_id, str(uuid.uuid4()), capability.value, operation,
                             dict(slots), None, RequestOutputs(), input_basename, snapshot,
                             self._ctx.config.limits, selection)

    async def _begin(self, policy: Any, mounts: Any, request: Any, on_event: Callable[[Any], None], *,
                     byte_limit: int, timeout: float, growth: Any,
                     manual_backing: str | None = None) -> _Launched:
        """Launch through the port.  ``_verify`` proved the image present just before, so
        a release policy is launched without a pull (``_no_pull_policy``); ``manual_backing``
        (setup's verified Docker backing) is forwarded to the launch's storage admission."""
        from .model_policy import ModelPolicyError

        extra = {} if manual_backing is None else {"manual_backing": manual_backing}
        try:
            launched = await self._launch.launch(_no_pull_policy(policy), mounts, request, on_event,
                                                 byte_limit=byte_limit, timeout=timeout, growth=growth,
                                                 **extra)
        except DomainValidationError as error:
            raise ModelPolicyError(_closed_probe_reason(error.reason_code)) from None
        self._active, self._reaped = launched, False
        return launched

    async def _end(self, launched: _Launched, timeout: float) -> ReasonCode:
        """Wait (bounded) for the run to finish, record reaping and close its resources."""
        task = launched.task
        done, _ = await asyncio.wait({task}, timeout=timeout)
        if not done:
            launched.obsolete()
            done, _ = await asyncio.wait({task}, timeout=PROBE_END_SECONDS)
        if task.done():
            if task.cancelled():
                reason = ReasonCode.CANCELLED
            elif task.exception() is not None:
                reason = _reason_of(task.exception(), ReasonCode.WORKER_PROTOCOL_MISMATCH)
            else:
                reason = ReasonCode.OK if task.result().completed is True else ReasonCode.WORKER_PROTOCOL_MISMATCH
        else:
            reason = ReasonCode.WRITER_UNRESOLVED
        self._reaped = reason is not ReasonCode.WRITER_UNRESOLVED
        self._active = None
        await launched.close()
        _LOG.debug("probe run ended reason=%s", reason.value)
        return reason

    @staticmethod
    def _ended_early(task: asyncio.Task) -> Exception:
        """The closed error for a run that ended before its evidence was taken.

        A cancelled/storage/Docker failure keeps its reason; a worker that simply did not
        hold (or exited) leaves no evidence: ``envelope_unvalidated``.
        """
        from .model_policy import ModelPolicyError

        error = None if task.cancelled() else task.exception()
        reason = _reason_of(error, ReasonCode.WORKER_PROTOCOL_MISMATCH) if error else \
            ReasonCode.CANCELLED if task.cancelled() else ReasonCode.WORKER_PROTOCOL_MISMATCH
        return ModelPolicyError(_closed_probe_reason(reason))

    async def _wait_artifact(self, event: asyncio.Event, task: asyncio.Task) -> None:
        from .model_policy import ModelPolicyError

        waiter = asyncio.ensure_future(event.wait())
        try:
            await asyncio.wait({waiter, task}, timeout=PROBE_ARTIFACT_WAIT_SECONDS,
                               return_when=asyncio.FIRST_COMPLETED)
        finally:
            if not waiter.done():
                waiter.cancel()
        if task.done():  # the run ended before (or with) the artifact: nothing can be sampled
            raise self._ended_early(task)
        if not event.is_set():
            raise ModelPolicyError(ReasonCode.ENVELOPE_UNVALIDATED)

    async def _sample(self, sampler: Any, task: asyncio.Task) -> Any:
        from .model_policy import ModelPolicyError

        if task.done():
            raise self._ended_early(task)  # hold exceeded: never fake
        try:
            hardware = await asyncio.to_thread(sampler.sample)
        except Exception:  # noqa: BLE001 - an unsampled worker is no evidence
            raise ModelPolicyError(ReasonCode.ENVELOPE_UNVALIDATED) from None
        if task.done():
            raise self._ended_early(task)
        return hardware

    async def _evidence(self, launched: _Launched, artifact: asyncio.Event, claims: list[dict],
                        scratch: str, name: str, entry: Any, resolved: Any, backend: Any,
                        gpu: str | None) -> Any:
        from .domains import Backend
        from .model_policy import (
            ModelPolicyError,
            PolicyEvidence,
            RuntimeFingerprint,
            WorkerEvidence,
        )

        task = launched.task
        began = time.monotonic()
        await self._wait_artifact(artifact, task)
        reference = launched.reference
        if reference is None or len(claims) != 1:
            raise ModelPolicyError(ReasonCode.ENVELOPE_UNVALIDATED)
        raw = await asyncio.to_thread(_read_claimed, scratch, name, claims[0], WorkerSlot.PROBE_METADATA,
                                      PROBE_ARTIFACT_BYTES)
        version, types = _parse_probe_artifact(raw, backend.value, gpu)
        if (version != entry.runtime_version or reference.gpu_visible is not (backend is Backend.CUDA)
                or reference.gpu_uuid != gpu):
            raise ModelPolicyError(ReasonCode.ENVELOPE_UNVALIDATED)
        artifact_at = time.monotonic()
        adapter = _ProbeRuntimeCapabilities(reference, backend, types)
        sampler = self._launch.sampler(reference, adapter)
        hardware = await self._sample(sampler, task)
        if _evidence_age(hardware, _utcnow()) > EVIDENCE_FRESH_SECONDS:
            hardware = await self._sample(sampler, task)  # one fresh sample, still held
        sampled_at = time.monotonic()
        if (not _worker_readings_ok(hardware, types)
                or _evidence_age(hardware, _utcnow()) > EVIDENCE_MAX_AGE_SECONDS):
            raise ModelPolicyError(ReasonCode.ENVELOPE_UNVALIDATED)  # stale: closed, never faked
        try:
            fingerprint = RuntimeFingerprint(entry.digest, entry.runtime_version, entry.driver_version,
                                             entry.hardware_fingerprint)
            runtime = adapter.compute_types(backend, deadline=time.monotonic() + 1.0)
            worker = WorkerEvidence(reference, hardware, runtime, fingerprint,
                                    hardware.runtime_compute_types.sample_time)
            evidence = PolicyEvidence((worker,), resolved.artifacts)
        except (DomainValidationError, ValueError, TypeError):
            raise ModelPolicyError(ReasonCode.ENVELOPE_UNVALIDATED) from None
        _LOG.debug("probe evidence capability=%s artifact_seconds=%.2f sample_seconds=%.2f",
                   resolved.capability.value, artifact_at - began, sampled_at - artifact_at)
        return evidence

    # -- port ------------------------------------------------------------------------
    async def collect(self, capability: Capability | str, *,
                      use: Callable[[Any, Any], Any] | None = None,
                      job_id: str | None = None, attempt_id: str | None = None,
                      scratch: Any = None, manual_backing: str | None = None) -> Any:
        """Fresh ``PolicyEvidence`` for one ASR capability; ``use(evidence, resolved)`` runs
        while the probe container is still held (that is where ``select_model`` belongs).
        ``scratch`` is the ``_ScratchIndex`` the probe worker mounts (setup passes the one
        for the paths it is provisioning); None is the service's configured scratch.
        ``manual_backing`` is the Docker data location the run's admission verifies (setup's
        verified backing); None is the configured one."""
        from .artifacts import assign_slots
        from .domains import Backend
        from .model_policy import ModelPolicyError
        from .requests import RequestSelection
        from .runtime import WorkerMounts

        async with self._lock:
            entry, resolved, gpu = await self._resolve(capability)
            wanted = resolved.capability
            backend = Backend.CUDA if wanted is Capability.ASR_CUDA else Backend.CPU
            await self._verify(resolved.worker_policy.image)
            job_id, attempt_id = job_id or str(uuid.uuid4()), attempt_id or str(uuid.uuid4())
            index = scratch if scratch is not None else self._ctx.scratch
            scratch = await asyncio.to_thread(index.new_dir, attempt_id)
            name = assign_slots(scratch, (WorkerSlot.PROBE_METADATA,))[WorkerSlot.PROBE_METADATA]
            request = self._request(wanted, "probe", {WorkerSlot.PROBE_METADATA.value: name},
                                    job_id=job_id, attempt_id=attempt_id,
                                    selection=RequestSelection(device=backend.value))
            artifact, claims = asyncio.Event(), []

            def on_event(event: Any) -> None:
                if event.event == "artifact":
                    claims.append(dict(event.payload))
                    artifact.set()

            launched = await self._begin(resolved.worker_policy, WorkerMounts(Path(scratch)), request,
                                         on_event, byte_limit=PROBE_ARTIFACT_BYTES,
                                         timeout=PROBE_TIMEOUT_SECONDS, growth=_probe_growth(),
                                         manual_backing=manual_backing)
            failed = False
            try:
                evidence = await self._evidence(launched, artifact, claims, scratch, name, entry,
                                                resolved, backend, gpu)
                if use is not None:
                    use(evidence, resolved)
            except BaseException:
                failed = True
                launched.obsolete()  # end the hold now; the runtime reaps on its own exit path
                raise
            finally:
                outcome = await self._end(launched, PROBE_TIMEOUT_SECONDS)
                if not failed and outcome is not ReasonCode.OK:
                    raise ModelPolicyError(_closed_probe_reason(outcome))
            return evidence

    async def _verify(self, image: str) -> None:
        from .model_policy import ModelPolicyError

        try:
            await self._launch.verify_image(image)
        except DomainValidationError as error:
            raise ModelPolicyError(_closed_probe_reason(error.reason_code)) from None

    async def pull(self, capability: Capability | str, *, timeout: float,
                   manual_backing: str | None = None) -> dict[str, Any]:
        """Setup PULL of one packaged-release image through the launch port (``pull_image``).

        The reference and its sizes come from the manifest entry itself, not from
        ``resolve_capability``: pulling the CUDA image must not require a usable GPU.  A
        development manifest never pulls (closed ``service_unavailable``).  Serialised with
        the probe runs by the same lock; failures are closed ``RuntimeFailure`` reasons.
        ``manual_backing`` is the verified Docker data location the pull's admission samples.
        """
        from .runtime import RuntimeFailure

        ctx = self._ctx
        entry = ctx.capability(capability)
        if entry is None or ctx.manifest.development_fixture:
            raise RuntimeFailure(ReasonCode.SERVICE_UNAVAILABLE)
        growth = _image_growth(entry.compressed_bytes, entry.unpacked_peak_bytes)
        async with self._lock:
            return await self._launch.pull_image(_release_image(entry), growth=growth,
                                                 timeout=timeout, manual_backing=manual_backing)

    async def smoke(self, capability: Capability | str, model: Any, snapshot: Any, *,
                    scratch: Any = None, manual_backing: str | None = None) -> dict[str, Any]:
        """One short real transcription of a generated fixture: ``{model, device, segments,
        duration_seconds}`` or a closed ``ModelPolicyError``.

        ``scratch`` is the ``_ScratchIndex`` the worker mounts (setup passes the one for the
        paths it is provisioning); None is the service's configured scratch.  ``manual_backing``
        is the Docker data location the run's admission verifies; None is the configured one.
        """
        from .artifacts import assign_slots
        from .domains import Backend
        from .model_policy import ModelPolicyError
        from .requests import RequestSelection
        from .runtime import WorkerMounts

        async with self._lock:
            _entry, resolved, _gpu = await self._resolve(capability)
            wanted = resolved.capability
            backend = Backend.CUDA if wanted is Capability.ASR_CUDA else Backend.CPU
            await self._verify(resolved.worker_policy.image)
            ctx = self._ctx
            limits = ctx.config.limits
            job_id, attempt_id = str(uuid.uuid4()), str(uuid.uuid4())
            index = scratch if scratch is not None else ctx.scratch
            scratch = await asyncio.to_thread(index.new_dir, attempt_id)
            inputs = await asyncio.to_thread(index.new_dir, attempt_id)
            input_name = "smoke.wav"
            await asyncio.to_thread(_write_tone_wav, os.path.join(inputs, input_name))
            name = assign_slots(scratch, (WorkerSlot.TRANSCRIPT_SEGMENTS,))[WorkerSlot.TRANSCRIPT_SEGMENTS]
            request = self._request(wanted, "transcribe", {WorkerSlot.TRANSCRIPT_SEGMENTS.value: name},
                                    job_id=job_id, attempt_id=attempt_id,
                                    selection=RequestSelection(device=backend.value,
                                                               model=model.artifact.model_id),
                                    input_basename=input_name, snapshot=snapshot.snapshot_id)
            mounts = WorkerMounts(Path(scratch), input_file=Path(inputs, input_name),
                                  model=Path(snapshot.path))
            claims: list[dict] = []

            def on_event(event: Any) -> None:
                if event.event == "artifact":
                    claims.append(dict(event.payload))

            launched = await self._begin(resolved.worker_policy, mounts, request, on_event,
                                         byte_limit=limits.export_bytes,
                                         timeout=float(limits.decode_timeout_seconds),
                                         growth=_smoke_growth(limits.export_bytes),
                                         manual_backing=manual_backing)
            try:
                outcome = await self._end(launched, float(limits.decode_timeout_seconds))
            except BaseException:
                launched.obsolete()
                raise
            if outcome is not ReasonCode.OK:
                raise ModelPolicyError(_closed_probe_reason(outcome))
            if len(claims) != 1:
                raise ModelPolicyError(ReasonCode.ENVELOPE_UNVALIDATED)
            raw = await asyncio.to_thread(_read_claimed, scratch, name, claims[0],
                                          WorkerSlot.TRANSCRIPT_SEGMENTS, limits.export_bytes)
            try:
                doc = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                raise ModelPolicyError(ReasonCode.ARTIFACT_REJECTED) from None
            segments, duration = doc.get("segments"), doc.get("duration_seconds")
            if (type(doc) is not dict or doc.get("schema_version") != 1 or type(segments) is not list
                    or type(duration) not in (int, float) or not math.isfinite(duration)):
                raise ModelPolicyError(ReasonCode.ARTIFACT_REJECTED)
            return {"model": model.artifact.model_id, "device": backend.value,
                    "segments": len(segments), "duration_seconds": float(duration)}


def _closed_probe_reason(reason: ReasonCode) -> ReasonCode:
    """Storage/Docker/cancel reasons keep their meaning; anything else is unvalidated."""
    return reason if reason in _PASS_THROUGH_REASONS else ReasonCode.ENVELOPE_UNVALIDATED


def _probe_growth() -> Any:
    from .admission import WorkloadGrowth

    return WorkloadGrowth(0, 0, 0, 0, (0,), 0, 0, 0, 0, (PROBE_ARTIFACT_BYTES,), 0, MIB)


def _smoke_growth(export_bytes: int) -> Any:
    from .admission import WorkloadGrowth

    return WorkloadGrowth(0, 0, 0, 0, (0,), 0, 0, 0, 0, (export_bytes,), MIB, MIB)


# ------------------------------------------------------------------- docker and launcher


def resolve_docker_target(timeout: float = 10.0) -> Any:
    """The active local Docker context as a ``runtime.DockerTarget`` (closed failures)."""
    from .runtime import DockerTarget, RuntimeFailure

    flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
    try:
        shown = subprocess.run(("docker", "context", "show"), capture_output=True, timeout=timeout,
                               check=False, stdin=subprocess.DEVNULL, **flags)
        name = shown.stdout.decode("ascii", "replace").strip()
        if shown.returncode or not name:
            raise RuntimeFailure(ReasonCode.DOCKER_UNAVAILABLE)
        inspected = subprocess.run(("docker", "context", "inspect", name), capture_output=True,
                                   timeout=timeout, check=False, stdin=subprocess.DEVNULL, **flags)
        values = json.loads(inspected.stdout)
        if inspected.returncode or len(values) != 1:
            raise RuntimeFailure(ReasonCode.DOCKER_UNAVAILABLE)
        endpoint = values[0]["Endpoints"]["docker"]["Host"]
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError, IndexError):
        raise RuntimeFailure(ReasonCode.DOCKER_UNAVAILABLE) from None
    return DockerTarget(name, endpoint)


class _DockerAccess:
    """One lazily resolved ``NativeDockerCLI`` shared by the launcher and reconciliation."""

    def __init__(self) -> None:
        self._cli: Any = None
        self._lock = asyncio.Lock()

    async def cli(self) -> Any:
        from .runtime import NativeDockerCLI

        async with self._lock:
            if self._cli is None:
                target = await asyncio.to_thread(resolve_docker_target)
                self._cli = NativeDockerCLI(target)
            return self._cli


class _LazyContainers:
    """``recovery.OwnedContainers`` that resolves Docker only when a recorded instance exists."""

    def __init__(self, docker: _DockerAccess) -> None:
        self._docker = docker
        self._inner: Any = None

    async def _get(self) -> Any:
        from .recovery import NativeOwnedContainers, RecoveryError

        if self._inner is None:
            try:
                cli = await self._docker.cli()
            except Exception:  # noqa: BLE001 - unresolved Docker is not proof of absence
                raise RecoveryError(ReasonCode.DOCKER_UNAVAILABLE) from None
            self._inner = NativeOwnedContainers(cli, owner=f"reconcile-{uuid.uuid4().hex}")
        return self._inner

    async def list(self, instance: str) -> Any:
        return await (await self._get()).list(instance)

    async def stop(self, ref: Any) -> None:
        await (await self._get()).stop(ref)

    async def kill(self, ref: Any) -> None:
        await (await self._get()).kill(ref)

    async def remove(self, ref: Any) -> None:
        await (await self._get()).remove(ref)

    async def exists(self, ref: Any) -> bool | None:
        return await (await self._get()).exists(ref)


class DockerWorkerLauncher:
    """``scheduler.WorkerLauncher`` over a fresh ``runtime.DockerRuntime`` per stage.

    Images come only from the loaded manifest.  A development image is local and never
    pulled: ``prepare`` requires it by id.  A packaged-release image (``repo@digest``) is
    verified by ``RepoDigests``; one verified present at run time is launched without a
    pull (the registry is not needed: offline jobs run); one that is absent (pruned after
    setup) does not fail the job: the runtime pulls it under ``allow_pull`` and the run's
    guard budgets that pull.
    Mounts are host-assigned: the per-assignment scratch directory, the job's local input
    or a verified scratch input, and the verified model snapshot.
    """

    def __init__(self, ctx: _Context, docker: _DockerAccess, probe: Any = None) -> None:
        self._ctx = ctx
        self._docker = docker
        self._probe = probe  # the probe/smoke runs are stopped and reaped with the stage worker
        self._ready: set[str] = set()
        self._reaped = True
        self._active: tuple[Any, Any] | None = None

    @property
    def writers_reaped(self) -> bool:
        return self._reaped and (self._probe is None or self._probe.writers_reaped)

    @property
    def running(self) -> bool:
        return self._active is not None or bool(self._probe is not None and self._probe.running)

    def ready(self, capability: str) -> bool:
        return capability in self._ready

    async def prepare(self, capability: str) -> None:
        policy = self._ctx.resolved(capability).worker_policy
        cli = await self._docker.cli()
        if policy.development_image:
            await _require_image(cli, policy.image, development=True)
        else:
            # Verified presence or not: the inspect proves Docker answers; an absent release
            # image is pulled by the runtime (``allow_pull``), budgeted at run time.
            await _image_listed(cli, policy.image)
        self._ready.add(str(capability))

    def _mounts(self, request: Any) -> Any:
        from .runtime import RuntimeFailure, WorkerMounts

        names = list(request.assigned_slots.values())
        if not names:
            raise RuntimeFailure(ReasonCode.WORKER_PROTOCOL_MISMATCH)
        directories = {self._ctx.scratch.directory_of(name) for name in names}
        if len(directories) != 1:
            raise RuntimeFailure(ReasonCode.UNAUTHORIZED)
        scratch = Path(directories.pop())
        input_file = None
        if request.input_basename is not None:
            try:
                input_file = Path(self._ctx.scratch.path_of(request.input_basename))
            except DomainValidationError:
                job = self._ctx.repo.get_job(request.job_id)
                if job is None or job.request.source.kind != "local" or \
                        os.path.basename(job.request.source.input) != request.input_basename:
                    raise RuntimeFailure(ReasonCode.UNAUTHORIZED) from None
                input_file = Path(job.request.source.input)
        model = None
        if request.model_snapshot_id is not None:
            snapshots = {s.snapshot_id: s for s in self._ctx.cache.snapshots()} if self._ctx.cache else {}
            snapshot = snapshots.get(request.model_snapshot_id)
            if snapshot is None:
                raise RuntimeFailure(ReasonCode.CACHE_UNKNOWN)
            model = Path(snapshot.path)
        return WorkerMounts(scratch, input_file=input_file, model=model)

    async def run(self, request: Any, *, stage: Stage, on_event: Callable[[Any], None]) -> Any:
        from .guards import CancellationCoordinator, WriteGuard
        from .runtime import DockerRuntime, RuntimeFailure, RuntimeStorage

        ctx = self._ctx
        policy = ctx.resolved(request.capability).worker_policy
        mounts = await asyncio.to_thread(self._mounts, request)
        job = await asyncio.to_thread(ctx.repo.get_job, request.job_id)
        if job is None:
            raise RuntimeFailure(ReasonCode.CONFLICT)
        cli = await self._docker.cli()
        limits = _limits_for(ctx, job.request)
        session = await asyncio.to_thread(
            ctx.docker_storage.prepare, manual_backing=ctx.config.paths.docker_backing)
        try:
            growth, _allocated = _growth(ctx, job.request, (Capability.parse(request.capability),))
            # Any model snapshot is already acquired: it cannot grow during this worker's
            # run.  A development image is local (prepare).  A release image verified
            # present now (one async inspect) cannot grow either and is launched without
            # a pull (``_no_pull_policy``: offline, the registry is not needed); an absent
            # one is pulled by the runtime under ``allow_pull``, so its compressed and
            # unpacked growth stays in this run's budget.  The credits are discarded.
            present = policy.development_image or await _image_listed(cli, policy.image)
            if present:
                policy = _no_pull_policy(policy)
            growth = replace(growth, model_partial=0, model_unpacked=0,
                             **({"image_compressed": 0, "image_unpacked": 0} if present else {}))
            paths = {"scratch": mounts.scratch, "cache": Path(ctx.config.paths.cache),
                     "output": Path(job.request.output_root), "state": Path(ctx.state_dir)}
            storage = RuntimeStorage(session, ctx.volumes, paths=paths, growth=growth, limits=limits,
                                     sample_budget_seconds=WORKER_STORAGE_SAMPLE_SECONDS)
            coordinator = CancellationCoordinator(
                f"worker-{uuid.uuid4().hex}", _MemoryLease(),
                stop_grace_seconds=limits.worker_stop_grace_seconds)
            remote = request.operation in ("metadata", "download")
            byte_limit = limits.media_bytes if remote else limits.export_bytes
            timeout = limits.download_timeout_seconds if remote else limits.decode_timeout_seconds
            guard = WriteGuard(coordinator, storage.sample, byte_limit=byte_limit,
                               timeout_seconds=float(timeout), sample_timeout_seconds=2.9)
            runtime = DockerRuntime(cli, guard, instance=ctx.instance, policy=policy, mounts=mounts)
            self._active = (runtime, coordinator)
            self._reaped = False
            try:
                result = await runtime.run(request, on_event=on_event)
            except RuntimeFailure as error:
                self._reaped = error.reason_code is not ReasonCode.WRITER_UNRESOLVED
                raise
            self._reaped = True
            return result
        finally:
            self._active = None
            await asyncio.to_thread(session.close)

    async def stop(self, reason: ReasonCode) -> bool:
        probe_reaped = True if self._probe is None else await self._probe.stop(reason)
        active = self._active
        if active is None:
            return self._reaped and probe_reaped
        runtime, coordinator = active
        runtime.obsolete()
        result = await coordinator.stop(reason)
        self._reaped = bool(result.writers_reaped)
        return self._reaped and probe_reaped


# ------------------------------------------------------------------------------ snapshot


class _DockerStatus:
    """Cached Docker evidence for snapshots: refreshed off the loop, never on demand."""

    def __init__(self, provider: Any, backing: str) -> None:
        self._provider, self._backing = provider, backing
        self.value: tuple[str, Any] | None = None  # ("ok", evidence) | ("error", reason)
        self.at: float | None = None
        self._task: asyncio.Task | None = None

    def _probe(self) -> tuple[str, Any]:
        try:
            return "ok", self._provider.sample(manual_backing=self._backing)
        except Exception as error:  # noqa: BLE001 - closed reason only
            return "error", _reason_of(error, ReasonCode.DOCKER_UNAVAILABLE)

    async def _refresh(self) -> None:
        value = await asyncio.to_thread(self._probe)
        self.value, self.at = value, time.monotonic()

    def kick(self) -> asyncio.Task | None:
        if self._task is not None and not self._task.done():
            return self._task
        if self.at is not None and time.monotonic() - self.at < DOCKER_REFRESH_SECONDS:
            return None
        self._task = asyncio.get_running_loop().create_task(self._refresh())
        return self._task

    async def close(self) -> None:
        task = self._task
        if task is not None and not task.done():
            await asyncio.wait({task}, timeout=SHUTDOWN_TASK_SECONDS)


def _measure(value: Any, source: str, stamp: str, reason: ReasonCode = ReasonCode.SERVICE_UNAVAILABLE,
             status: FieldStatus = FieldStatus.UNKNOWN) -> Any:
    """OK when a value was measured; otherwise null with a closed reason (never zero)."""
    from .snapshots import Measurement

    if value is not None:
        return Measurement(value, source, FieldStatus.OK, stamp, None)
    return Measurement(None, source, status, stamp, reason)


def _from_hardware(field_value: Any, stamp: str, source: str) -> Any:
    from .snapshots import Measurement

    if field_value is None:
        return _measure(None, source, stamp, ReasonCode.SERVICE_UNAVAILABLE)
    if field_value.status is FieldStatus.OK:
        return Measurement(field_value.value, source, FieldStatus.OK, stamp, None)
    return Measurement(None, source, field_value.status, stamp,
                       field_value.reason or ReasonCode.STORAGE_UNKNOWN)


class _Snapshots:
    """Assembles ``snapshots.ResourceSnapshot`` from the providers and scheduler progress."""

    def __init__(self, comp: Composition) -> None:
        self._comp = comp
        self._lock = asyncio.Lock()
        self._cached: Any = None
        self._cached_at = 0.0

    def _sample(self) -> dict[str, Any]:
        comp, ctx = self._comp, self._comp.ctx
        hardware = None
        try:
            hardware = comp.hardware.sample()
        except Exception:  # noqa: BLE001 - unknown hardware is reported as unknown
            hardware = None
        roots = {"scratch": ctx.config.paths.scratch, "cache": ctx.config.paths.cache,
                 "output": ctx.config.paths.output, "state": ctx.state_dir}
        volumes: dict[str, Any] = {}
        for role, path in roots.items():
            try:
                volumes[role] = ctx.volumes.sample(path, sqlite_state=role == "state")
            except Exception:  # noqa: BLE001
                volumes[role] = None
        active = attempt = None
        with contextlib.suppress(Exception):
            jobs = ctx.repo.active_jobs(1)
            active = jobs[0] if jobs else None
            if active is not None and active.attempt_id is not None:
                attempt = ctx.repo.get_attempt(active.attempt_id)
        return {"hardware": hardware, "volumes": volumes, "active": active, "attempt": attempt}

    def _storage(self, volumes: Mapping[str, Any], docker: Any, stamp: str) -> tuple[list, bool]:
        from .admission import reserve_bytes
        from .domains import Context, VolumeSlot
        from .snapshots import StorageEntry

        entries, complete = [], True
        na = FieldStatus.NOT_APPLICABLE
        limits = self._comp.ctx.config.limits

        def entry(slot: int, evidence: Any, context: Any) -> Any:
            nonlocal complete
            capacity = evidence.capacity.value if evidence is not None else None
            free = evidence.free.value if evidence is not None else None
            identity = evidence.identity.value if evidence is not None else None
            if capacity is None or free is None:
                complete = False
            reserve = reserve_bytes(capacity, limits) if type(capacity) is int else None
            return StorageEntry(
                VolumeSlot(slot), identity or f"unknown-{slot}", context,
                _measure(capacity, "host_volume", stamp, ReasonCode.STORAGE_UNKNOWN),
                _measure(free, "host_volume", stamp, ReasonCode.STORAGE_UNKNOWN),
                _measure(reserve, "admission", stamp, ReasonCode.STORAGE_UNKNOWN),
                _measure(None, "admission", stamp, ReasonCode.STORAGE_UNKNOWN, na),
                _measure(None, "admission", stamp, ReasonCode.STORAGE_UNKNOWN, na),
                _measure(None, "admission", stamp, ReasonCode.STORAGE_UNKNOWN, na),
                "host_volume", stamp, None)

        for slot, role in enumerate(_ROLES):
            entries.append(entry(slot, volumes.get(role), Context.HOST))
        if docker is not None:
            for index, constraint in enumerate(docker.constraints[:4]):
                entries.append(entry(len(_ROLES) + index, constraint.volume, Context.DOCKER_VM))
        else:
            complete = False
        return entries, complete

    async def build(self) -> Any:
        async with self._lock:
            if self._cached is not None and time.monotonic() - self._cached_at < SNAPSHOT_REUSE_SECONDS:
                return self._cached
            comp = self._comp
            task = comp.docker_status.kick()
            if comp.docker_status.value is None and task is not None:
                await asyncio.wait({task}, timeout=DOCKER_FIRST_WAIT_SECONDS)
            data = await asyncio.to_thread(self._sample)
            snapshot = self._assemble(data)
            self._cached, self._cached_at = snapshot, time.monotonic()
            return snapshot

    def _assemble(self, data: Mapping[str, Any]) -> Any:
        from .domains import BudgetConfidence
        from .snapshots import ResourceSnapshot

        comp = self._comp
        stamp = _stamp()
        na = FieldStatus.NOT_APPLICABLE
        service = _measure(comp.service_state.value, "service", stamp)
        status = comp.docker_status
        docker_evidence = None
        if status.value is None:
            docker = _measure(None, "docker", stamp, ReasonCode.SERVICE_UNAVAILABLE)
        elif status.value[0] == "ok":
            evidence = status.value[1]
            docker_evidence = evidence
            if evidence.context.status is FieldStatus.OK:
                docker = _measure(True, "docker", stamp)
            elif evidence.context.reason is ReasonCode.DOCKER_UNAVAILABLE:
                docker = _measure(False, "docker", stamp)
            else:
                docker = _measure(None, "docker", stamp, evidence.context.reason or ReasonCode.STORAGE_UNKNOWN)
        elif status.value[1] is ReasonCode.DOCKER_UNAVAILABLE:
            docker = _measure(False, "docker", stamp)
        else:
            docker = _measure(None, "docker", stamp, status.value[1])
        if status.at is not None and time.monotonic() - status.at > \
                DOCKER_REFRESH_SECONDS + comp.ctx.config.monitoring.stale_after_seconds:
            docker = _measure(None, "docker", stamp, ReasonCode.STORAGE_STALE, FieldStatus.STALE)
        hw = data["hardware"]
        worker_running = bool(getattr(comp.launcher, "running", False))
        worker = _measure(worker_running, "launcher", stamp)
        host_ram = _from_hardware(getattr(hw, "host_ram_used", None), stamp, "host_psutil")
        cpu = _from_hardware(getattr(hw, "host_cpu_percent", None), stamp, "host_psutil")
        if cpu.value is not None and cpu.value > 100:
            cpu = _measure(None, "host_psutil", stamp, ReasonCode.STORAGE_UNKNOWN)
        gpu_name = getattr(hw, "gpu_name", None)
        if gpu_name is not None and gpu_name.status is FieldStatus.OK:
            gpu_capability = _measure(True, "host_nvidia_smi", stamp)
        elif gpu_name is not None and gpu_name.reason is ReasonCode.GPU_UNAVAILABLE:
            gpu_capability = _measure(False, "host_nvidia_smi", stamp)
        else:
            gpu_capability = _measure(None, "host_nvidia_smi", stamp, ReasonCode.GPU_UNAVAILABLE)
        if gpu_capability.value is True:
            gpu_util = _from_hardware(getattr(hw, "gpu_utilization_percent", None), stamp, "host_nvidia_smi")
            gpu_vram = _from_hardware(getattr(hw, "gpu_vram_used", None), stamp, "host_nvidia_smi")
        else:
            gpu_util = _measure(None, "host_nvidia_smi", stamp, ReasonCode.GPU_UNAVAILABLE, na)
            gpu_vram = _measure(None, "host_nvidia_smi", stamp, ReasonCode.GPU_UNAVAILABLE, na)
        active, attempt = data["active"], data["attempt"]
        worker_ram = _measure(None, "worker", stamp, ReasonCode.SERVICE_UNAVAILABLE, na)
        model = _measure(None, "scheduler", stamp, ReasonCode.SERVICE_UNAVAILABLE, na)
        progress = _measure(None, "scheduler", stamp, ReasonCode.DURATION_UNKNOWN, na)
        elapsed = _measure(None, "scheduler", stamp, ReasonCode.DURATION_UNKNOWN, na)
        eta = _measure(None, "scheduler", stamp, ReasonCode.DURATION_UNKNOWN, na)
        job_id = attempt_id = stage = None
        if active is not None:
            job_id, attempt_id = active.job_id, active.attempt_id
            with contextlib.suppress(DomainValidationError):
                stage = Stage.parse(active.state.value)
            if worker_running:
                worker_ram = _measure(None, "worker", stamp, ReasonCode.SERVICE_UNAVAILABLE)
            prepared = comp.ctx.models.prepared(job_id) if comp.ctx.models else None
            model = _measure(prepared[0] if prepared else None, "scheduler", stamp,
                             ReasonCode.SERVICE_UNAVAILABLE)
            percent = comp.scheduler.progress(job_id) if comp.scheduler is not None else None
            progress = _measure(percent if percent is not None and 0 <= percent <= 100 else None,
                                "worker_progress", stamp, ReasonCode.DURATION_UNKNOWN)
            running = next((r for r in (attempt.stage_records if attempt else ())
                            if r.ended_at is None), None)
            seconds = None
            if running is not None:
                with contextlib.suppress(ValueError, TypeError):
                    started = datetime.fromisoformat(running.started_at)
                    seconds = max(0.0, (_utcnow() - started).total_seconds())
            elapsed = _measure(seconds, "scheduler", stamp, ReasonCode.DURATION_UNKNOWN)
            eta = _measure(None, "scheduler", stamp, ReasonCode.DURATION_UNKNOWN)
        storage, complete = self._storage(data["volumes"], docker_evidence, stamp)
        confidence = BudgetConfidence.MEASURED if complete else BudgetConfidence.UNKNOWN
        return ResourceSnapshot(1, stamp, FieldStatus.OK, service, docker, worker, host_ram,
                                worker_ram, cpu, gpu_capability, gpu_util, gpu_vram, model,
                                tuple(storage), confidence, job_id, attempt_id, stage,
                                progress, elapsed, eta)


# --------------------------------------------------------------------------------- setup


def _step(ok: bool, reason: ReasonCode = ReasonCode.OK, data: Mapping[str, Any] | None = None,
          next_action: str = "") -> Any:
    from .setup import StepResult

    return StepResult(ok, reason, dict(data or {}), next_action)


class _Provisioning:
    """``setup.ProvisioningPort`` bound to real adapters (SPEC_GAP-9).

    Without a loaded manifest every step is a closed ``service_unavailable`` failure.  The
    fixture manifest's images are local development images: ``pull`` verifies the exact
    digest is present and never pulls.  A packaged release's images are pulled by digest
    through the probe port (``_ProbeEvidence.pull``: admitted, guarded, verified by
    ``RepoDigests``; one already present is not pulled).  ``choose_model`` runs the ASR probe through the
    composed ``_ProbeEvidence`` port and selects for the longest admissible job; ``smoke``
    runs one short real transcription of a generated fixture through the same port.  Both
    run on the service loop (this port is called from a setup worker thread) and fail
    closed without the port.
    """

    def __init__(self, ctx: _Context, hardware: Any = None, *, probe: Any = None,
                 loop: asyncio.AbstractEventLoop | None = None) -> None:
        self._ctx = ctx
        self._hardware = hardware
        self._probe, self._loop = probe, loop
        self._lock = threading.Lock()
        # The wizard's accepted paths (``bind_paths``); None means the configured ones.
        self._bound: dict[str, str] | None = None
        # Setup-scoped cache at a bound cache path other than the configured one: opened
        # on first use (after CONFIRM, never at PATHS), closed on rebind and shutdown.
        self._setup_cache: Any = None
        self._setup_cache_path: str | None = None
        # Release pulls whose daemon-side fate is unknown (``writer_unresolved``, or the
        # loop-side wait timed out): the setup lease must outlive them (``writers_reaped``).
        self._unresolved: set[str] = set()

    @property
    def writers_reaped(self) -> bool:
        """False while a setup pull may still be running in the Docker daemon.

        ``runtime._PullRecovery``'s rule: a daemon pull has no cancellation API, so an
        unresolved one keeps the lease it ran under until the service is restarted and
        startup reconciliation proves the daemon idle.  Never cleared here.
        """
        return not self._unresolved

    # -- bound paths -------------------------------------------------------------------
    def bind_paths(self, paths: Mapping[str, str]) -> None:
        """Provision for the wizard's paths; the service's own cache and scratch stay as they are.

        Every bind closes an open setup-scoped cache, even when the cache path is unchanged:
        going back to PATHS re-runs the host and plan checks, which sample the bound cache
        path as a plain directory, and on Windows a cache still holding that root open makes
        the sample ``storage_unknown``.  The cache reopens lazily on the next use.
        """
        bound = {k: str(v) for k, v in paths.items() if type(v) is str and v}
        with self._lock:
            self._bound = bound
            self._close_setup_cache()

    def _bound_path(self, key: str) -> str | None:
        """The bound path for ``key`` when it differs from the configured one, else None."""
        bound = (self._bound or {}).get(key)
        config = getattr(self._ctx, "config", None)
        if bound is None or config is None or _same_path(bound, getattr(config.paths, key)):
            return None
        return bound

    def _cache_path(self) -> str:
        return self._bound_path("cache") or self._ctx.config.paths.cache

    def _docker_backing(self) -> str:
        """The Docker data location the pull admission samples: the wizard's (what its
        BACKING step verified; "" is auto discovery) once paths are bound, else the configured one."""
        if self._bound is None:
            return self._ctx.config.paths.docker_backing
        return self._bound.get("docker_backing", "")

    def release_cache(self) -> None:
        """Close the setup-scoped cache so the bound cache root is sampled as a plain
        directory again (the machine calls it before a pre-confirmation check re-runs)."""
        self.close()

    def _plan_ctx(self) -> _Context:
        """The context host plans sample through: a bound cache root is held open like the
        configured one, so it is sampled through its parent (``_HeldRootVolumes``)."""
        bound = self._bound_path("cache")
        if bound is None:
            return self._ctx
        return replace(self._ctx, volumes=_HeldRootVolumes(self._ctx.volumes, (bound,)))

    def _cache(self) -> tuple[Any, Callable[[Any], Any] | None]:
        """``(cache, acquire)`` for the cache path being provisioned: the service's when the
        bound path is the configured one (or nothing is bound), else the setup-scoped one."""
        bound = self._bound_path("cache")
        if bound is None:
            return self._ctx.cache, self._ctx.acquire_model
        with self._lock:
            if self._setup_cache is None or not _same_path(self._setup_cache_path, bound):
                self._close_setup_cache()
                self._setup_cache = _open_model_cache(self._plan_ctx(), bound)
                self._setup_cache_path = bound
            cache = self._setup_cache
        return cache, (cache.acquire if cache is not None else None)

    def _close_setup_cache(self) -> None:
        cache, self._setup_cache, self._setup_cache_path = self._setup_cache, None, None
        if cache is not None:
            with contextlib.suppress(Exception):
                cache.close()

    def close(self) -> None:
        with self._lock:
            self._close_setup_cache()

    def setup_snapshots(self) -> tuple[Any, ...]:
        """Snapshots of the cache being provisioned (the bound one when it differs)."""
        cache, _acquire = self._cache()
        return tuple(cache.snapshots()) if cache is not None else ()

    def _no_manifest(self) -> Any:
        """Closed "no runtime manifest" step: the composition's resolution says what to do
        (the release's blocker codes, the engine, the platform), else the fixture hint."""
        source = getattr(self._ctx, "manifest_source", None)
        guidance = getattr(source, "next_action", "") if getattr(source, "manifest", None) is None else ""
        return _step(False, ReasonCode.SERVICE_UNAVAILABLE,
                     next_action=guidance or f"set {MANIFEST_ENV} to a verified runtime manifest")

    def _on_loop(self, start: Callable[[], Any], timeout: float) -> Any:
        """Run a probe-port coroutine on the service loop from this worker thread."""
        if self._probe is None or self._loop is None:
            raise DomainValidationError(ReasonCode.ENVELOPE_UNVALIDATED)
        future = asyncio.run_coroutine_threadsafe(start(), self._loop)
        try:
            return future.result(timeout)
        except TimeoutError:
            future.cancel()
            raise DomainValidationError(ReasonCode.REQUEST_TIMEOUT) from None

    def pull(self, capability: Capability) -> Any:
        entry = self._ctx.capability(capability)
        if entry is None:
            return self._no_manifest()
        if not self._ctx.manifest.development_fixture:
            return self._pull_release(capability, entry)
        try:
            image = self._ctx.resolved(capability).worker_policy.image
            target = resolve_docker_target()
            flags = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
            result = subprocess.run(("docker", "--host", target.endpoint, "image", "inspect", image),
                                    capture_output=True, timeout=30, check=False,
                                    stdin=subprocess.DEVNULL, **flags)
            found, = json.loads(result.stdout)
            if result.returncode or found["Id"] != image:
                raise ValueError
        except DomainValidationError as error:
            return _step(False, error.reason_code, next_action="start Docker Desktop yourself")
        except (OSError, subprocess.SubprocessError, ValueError, TypeError, KeyError):
            return _step(False, ReasonCode.ASSET_VERIFICATION_FAILED,
                         next_action="build the local development image named by the manifest")
        return _step(True, data={"capability": str(capability), "present": True, "pulled": False})

    def _pull_release(self, capability: Capability, entry: Any) -> Any:
        """Setup PULL of a packaged-release image: an admitted, guarded pull on the service
        loop, bounded by the configured download timeout (capped).  ``bytes`` is the
        download this step made: the manifest's compressed bytes, or 0 when the image was
        already present and verified."""
        limits = self._ctx.config.limits
        timeout = min(float(limits.download_timeout_seconds), SETUP_PULL_TIMEOUT_SECONDS)
        data: dict[str, Any] = {"capability": str(capability)}
        backing = self._docker_backing()
        try:
            outcome = self._on_loop(lambda: self._probe.pull(capability, timeout=timeout,
                                                             manual_backing=backing),
                                    timeout + SETUP_PULL_GRACE_SECONDS)
        except DomainValidationError as error:
            reason = error.reason_code
            if reason in (ReasonCode.WRITER_UNRESOLVED, ReasonCode.REQUEST_TIMEOUT):
                # The daemon may still be pulling (the guard could not reap it, or the
                # loop-side wait gave up): the setup lease stays held (``writers_reaped``).
                self._unresolved.add(_release_image(entry))
            return _step(False, reason, data=data,
                         next_action=_PULL_GUIDANCE.get(reason, "retry the setup later"))
        except Exception:  # noqa: BLE001 - closed reason only
            return _step(False, ReasonCode.SERVICE_UNAVAILABLE, data=data,
                         next_action="retry the setup later")
        pulled = bool(outcome.get("pulled"))
        data.update(present=True, pulled=pulled, bytes=entry.compressed_bytes if pulled else 0)
        return _step(True, data=data)

    def probe_hardware(self) -> Any:
        from .hardware import HardwareSampler

        if self._ctx.manifest is None:
            return self._no_manifest()
        sampler = self._hardware or HardwareSampler()
        try:
            evidence = sampler.sample()
        except Exception as error:  # noqa: BLE001
            return _step(False, _reason_of(error, ReasonCode.STORAGE_UNKNOWN))
        data = {name: getattr(getattr(evidence, name, None), "value", None) for name in
                ("host_cpu_cores", "host_ram_total", "host_ram_available", "gpu_vram_total")}
        # Host and engine identity for the operator (SPEC_GAP-17): measured strings or None.
        data.update({name: _measured_text(evidence, name) for name in _HOST_IDENTITY_FIELDS})
        return _step(True, data=data)

    def choose_model(self, hardware: Mapping[str, Any]) -> Any:
        from .domains import OutputMode
        from .requests import RequestSelection

        ctx = self._ctx
        if ctx.manifest is None:
            return self._no_manifest()
        if not ctx.manifest.models:
            return _step(False, ReasonCode.ASSET_VERIFICATION_FAILED,
                         next_action="the manifest lists no model; choose video only")
        # The longest job this setup must admit: the configured cap, within what any
        # measured model was validated for (unmeasured models carry no such bound).
        validated = [m.validated_max_duration_seconds for m in ctx.manifest.models
                     if m.validated_max_duration_seconds is not None]
        duration = float(min([ctx.config.limits.duration_seconds, *validated]))
        selection = RequestSelection()
        guidance: dict[str, Any] = {}
        # Host and engine identity from the hardware step (SPEC_GAP-17); the engine
        # architecture the probe measured itself takes precedence when it has one.
        identity = _host_identity(hardware)

        def shown() -> dict[str, Any]:
            return {**identity, "engine_arch": guidance.get("engine_arch") or identity["engine_arch"],
                    "skipped_capabilities": list(guidance.get("skipped") or [])}

        scratch = self._bound_path("scratch")
        backing = self._docker_backing()  # the admission samples what BACKING verified
        try:
            # The probe worker mounts the scratch being provisioned, like the smoke does.
            index = _ScratchIndex(_private_dir(scratch)) if scratch is not None else None
            choice, resolved = self._on_loop(lambda: _choose_with_probe(
                ctx, self._probe, _asr_candidates(ctx, selection.device), duration=duration,
                selection=selection, output_mode=OutputMode.TRANSCRIPT,
                duration_limit=float(ctx.config.limits.duration_seconds), guidance=guidance,
                scratch=index, manual_backing=backing),
                SETUP_PROBE_SECONDS)
        except DomainValidationError as error:
            if error is guidance.get("failure"):
                # The probe succeeded; no model fits automatically.  The operator decides
                # (owner rule 7): the options still carry every verified model.
                resolved = guidance["resolved"]
                device = "cuda" if resolved.capability is Capability.ASR_CUDA else "cpu"
                return _step(True, data={"choice": None, "device": device, "compute_type": None,
                                         "capability": resolved.capability.value,
                                         "reason": error.reason_code.value,
                                         "duration_seconds": duration,
                                         "options": guidance["options"], **shown()})
            if guidance.get("unsupported"):
                # No ASR image in the manifest runs on this Docker engine's architecture.
                arch = guidance.get("engine_arch") or "unknown"
                return _step(False, error.reason_code, data=shown(),
                             next_action=f"the manifest has no ASR image for a {arch} Docker "
                                         "engine; choose video only")
            return _step(False, error.reason_code, data=shown(),
                         next_action="model selection needs in-worker evidence; choose video only")
        except Exception:  # noqa: BLE001 - closed reason only
            return _step(False, ReasonCode.SERVICE_UNAVAILABLE, next_action="retry the setup later")
        return _step(True, data={"choice": choice.artifact.model_id, "device": choice.backend.value,
                                 "compute_type": choice.compute_type,
                                 "capability": resolved.capability.value, "reason": choice.reason,
                                 "duration_seconds": duration, "options": guidance["options"],
                                 **shown()})

    def _model(self, choice: str) -> Any:
        if self._ctx.manifest is None:
            return None
        return next((m for m in self._ctx.manifest.models if m.artifact.model_id == choice), None)

    def model_capacity(self, choice: str) -> Any:
        model = self._model(choice)
        if model is None:
            return self._no_manifest() if self._ctx.manifest is None else \
                _step(False, ReasonCode.ASSET_VERIFICATION_FAILED)
        total = sum(item.bytes for item in model.files)
        plan = _host_plan(self._plan_ctx(), "model_unpacked", self._cache_path(), total)
        if not plan.admitted:
            return _step(False, plan.reason, next_action="free space in the cache folder")
        return _step(True, data={"model": choice, "bytes": total})

    def _acquire(self, choice: str) -> tuple[Any, Any] | Any:
        """``(model, snapshot)`` for a manifest model id, or the closed StepResult."""
        ctx = self._ctx
        model = self._model(choice)
        cache, acquire = self._cache() if model is not None else (None, None)
        if model is None or cache is None or acquire is None:
            return self._no_manifest() if ctx.manifest is None else \
                _step(False, ReasonCode.CACHE_UNKNOWN if model is not None
                      else ReasonCode.ASSET_VERIFICATION_FAILED)
        try:
            return model, acquire(model)
        except DomainValidationError as error:
            return _step(False, error.reason_code, next_action="retry the setup later")

    def acquire_model(self, choice: str) -> Any:
        acquired = self._acquire(choice)
        if type(acquired) is not tuple:
            return acquired
        return _step(True, data={"snapshot": acquired[1].snapshot_id})

    def smoke(self, choice: str) -> Any:
        ctx = self._ctx
        if ctx.manifest is None:
            return self._no_manifest()
        acquired = self._acquire(choice)
        if type(acquired) is not tuple:
            return acquired
        model, snapshot = acquired
        capability = Capability.ASR_CPU if ctx.capability(Capability.ASR_CPU) is not None \
            else Capability.ASR_CUDA
        scratch = self._bound_path("scratch")
        backing = self._docker_backing()  # the admission samples what BACKING verified
        try:
            # The worker mounts the scratch being provisioned (a bound path proves it mounts).
            extra = {"scratch": _ScratchIndex(_private_dir(scratch))} if scratch is not None else {}
            summary = self._on_loop(lambda: self._probe.smoke(capability, model, snapshot,
                                                              manual_backing=backing, **extra),
                                    SETUP_PROBE_SECONDS)
        except DomainValidationError as error:
            return _step(False, error.reason_code,
                         next_action="the inference smoke failed; fix the worker image or model")
        except Exception:  # noqa: BLE001 - closed reason only
            return _step(False, ReasonCode.SERVICE_UNAVAILABLE, next_action="retry the setup later")
        return _step(True, data=summary)


def _setup_audit(hub: Any) -> Callable[[str, str, ReasonCode], None]:
    def record(action: str, target_id: str, reason: ReasonCode) -> None:
        ids = {"target_id": target_id} if _identifier(target_id) else {}
        if not hub.audit(action, reason_code=reason, ids=ids):
            from .telemetry import TelemetryError

            raise TelemetryError(ReasonCode.AUDIT_LOSS)
    return record


class _ImageSizes(Mapping[str, int]):
    """One release capability's setup sizes, decided when read (PLAN time, in a thread):
    an image already verified present costs nothing more on the Docker disk, so its
    ``download_bytes`` (the compressed download) and ``docker_bytes`` (the unpacked peak
    while Docker extracts it) are 0; otherwise the manifest's bytes.  Both land on the
    Docker store (``setup._COST_DESTINATIONS``), the same compressed-plus-unpacked growth
    the pull's own guard admits.  ``LocalSetupActions`` copies these into plain dicts at
    every PLAN, which is when the presence is asked."""

    _KEYS = ("download_bytes", "docker_bytes", "reserve_bytes")

    def __init__(self, image: str, download: int, unpacked: int, reserve: int,
                 present: Callable[[str], bool]) -> None:
        self._image, self._download, self._unpacked = image, download, unpacked
        self._reserve, self._present = reserve, present

    def __getitem__(self, key: str) -> int:
        if key == "download_bytes":
            return 0 if self._present(self._image) else self._download
        if key == "docker_bytes":
            return 0 if self._present(self._image) else self._unpacked
        if key == "reserve_bytes":
            return self._reserve
        raise KeyError(key)

    def __iter__(self) -> Any:
        return iter(self._KEYS)

    def __len__(self) -> int:
        return len(self._KEYS)


def _setup_sizes(manifest: Any, image_present: Callable[[str], bool] | None = None
                 ) -> dict[str, Mapping[str, int]]:
    """Per-capability setup sizes.  ``image_present`` (``_release_image_present`` in the
    composition) makes a release image's download cost 0 once it is verified present;
    without it a release image always costs its compressed bytes."""
    if manifest is None:
        return {}
    model = max((sum(item.bytes for item in m.files) for m in manifest.models), default=0)
    sizes: dict[str, Mapping[str, int]] = {}
    for entry in manifest.capabilities:
        reserve = entry.worker_overhead_bytes + (model if entry.capability is not Capability.MEDIA else 0)
        # Development images are local: no download bytes, only the worker reserve.  A
        # packaged release image is pulled by digest: its compressed bytes are the download
        # and its unpacked peak the Docker-store growth while it is extracted, unless it is
        # already present when PLAN runs.
        if manifest.development_fixture:
            sizes[entry.capability.value] = {"download_bytes": 0, "reserve_bytes": reserve}
        elif image_present is None:
            sizes[entry.capability.value] = {"download_bytes": entry.compressed_bytes,
                                             "docker_bytes": entry.unpacked_peak_bytes,
                                             "reserve_bytes": reserve}
        else:
            sizes[entry.capability.value] = _ImageSizes(_release_image(entry), entry.compressed_bytes,
                                                        entry.unpacked_peak_bytes, reserve, image_present)
    return sizes


# ------------------------------------------------------------------------------- actions


def _cursor(cursor: str | None) -> int:
    from .api import ApiActionError

    if cursor is None:
        return 0
    if _OFFSET_RE.fullmatch(cursor) is None:
        raise ApiActionError(ReasonCode.INVALID_INPUT)
    return int(cursor)


def cancel_status(job: Any) -> str:
    """Map ``Scheduler.cancel``'s returned job to the API's closed status word."""
    return "cancelled" if job.state is JobState.CANCELLED else "intent"


class _ArtifactStream:
    """Async iterator over one contained, held artifact handle; re-hashed as it streams."""

    def __init__(self, stack: contextlib.ExitStack, held: Any, size: int, sha256: str) -> None:
        self._stack, self._held, self._size, self._sha = stack, held, size, sha256
        self._digest = hashlib.sha256()
        self._sent = 0
        self._closed = False

    def __aiter__(self) -> _ArtifactStream:
        return self

    async def __anext__(self) -> bytes:
        from .artifacts import ArtifactError

        if self._closed:
            raise StopAsyncIteration
        chunk = await asyncio.to_thread(self._held.read, CHUNK_BYTES)
        if not chunk:
            if self._sent != self._size or self._digest.hexdigest() != self._sha:
                raise ArtifactError(ReasonCode.ARTIFACT_REJECTED)
            await self.aclose()
            raise StopAsyncIteration
        self._sent += len(chunk)
        if self._sent > self._size:
            raise ArtifactError(ReasonCode.ARTIFACT_REJECTED)
        self._digest.update(chunk)
        return chunk

    async def aclose(self) -> None:
        if not self._closed:
            self._closed = True
            await asyncio.to_thread(self._stack.close)


def _open_published(job: Any, artifact: Any) -> tuple[contextlib.ExitStack, Any, str]:
    """Open the published file contained below the job folder; returns (stack, held, type)."""
    from .artifacts import MANIFEST_NAME, ArtifactError
    from .volumes import open_contained

    root = job.request.output_root
    job_name = f"podmuncher-{job.job_id}"
    name = None
    with contextlib.suppress(DomainValidationError, OSError, ValueError, UnicodeDecodeError):
        with open_contained(root, f"{job_name}/{MANIFEST_NAME}") as held:
            raw = held.read(1024 * 1024)
        doc = json.loads(raw.decode("utf-8"))
        for item in doc.get("artifacts") or ():
            if type(item) is dict and item.get("artifact_id") == artifact.artifact_id:
                name = item.get("file_name")
    if name is None and artifact.slot in _TRANSCRIPT_NAMES:
        name = _TRANSCRIPT_NAMES[artifact.slot]
    if type(name) is not str or not name or os.path.basename(name) != name:
        raise ArtifactError(ReasonCode.ARTIFACT_REJECTED)
    if artifact.format is ArtifactFormat.VIDEO:
        content_type = _VIDEO_TYPES.get(os.path.splitext(name)[1].lstrip(".").lower())
    else:
        content_type = _CONTENT_TYPES.get(artifact.format)
    if content_type is None:
        raise ArtifactError(ReasonCode.ARTIFACT_REJECTED)
    stack = contextlib.ExitStack()
    try:
        held = stack.enter_context(open_contained(root, f"{job_name}/{name}"))
    except BaseException:
        stack.close()
        raise
    return stack, held, content_type


class _Actions:
    """``api.ServiceActions`` over the real repository, scheduler, cache and setup service."""

    def __init__(self, comp: Composition) -> None:
        self._c = comp
        self._clone_keys: OrderedDict[str, tuple[str, str, str]] = OrderedDict()
        # key -> (plan_id, operation_id, the plan's canonical inputs) for idempotent replays
        self._setup_keys: OrderedDict[str, tuple[str, str, dict]] = OrderedDict()
        self._plans: OrderedDict[str, tuple[float, dict]] = OrderedDict()
        self._operations: OrderedDict[str, dict] = OrderedDict()

    @property
    def _repo(self) -> Any:
        return self._c.ctx.repo

    async def _job(self, job_id: str) -> Any:
        from .api import ApiActionError

        job = await asyncio.to_thread(self._repo.get_job, job_id)
        if job is None:
            raise ApiActionError(ReasonCode.INVALID_INPUT, not_found=True)
        return job

    async def _admit(self, action: str) -> None:
        """Refuse new work when stopping, on persistence loss or on audit loss (fail closed)."""
        from .api import ApiActionError

        comp = self._c
        if comp.stopping:
            raise ApiActionError(ReasonCode.SERVICE_STOPPED)
        lost = comp.persistence_loss()
        if lost is not None:
            raise ApiActionError(lost)
        await comp.audit_intent(action)

    # -- read routes -------------------------------------------------------------------
    async def identity(self) -> Mapping[str, Any]:
        return {"pid": os.getpid(), "service_state": self._c.service_state}

    async def snapshot(self) -> Any:
        return await self._c.snapshots.build()

    async def list_jobs(self, limit: int, cursor: str | None) -> tuple[list, str | None]:
        return await asyncio.to_thread(self._repo.list_jobs, limit, cursor)

    async def get_job(self, job_id: str) -> Any:
        return await self._job(job_id)

    # -- job routes --------------------------------------------------------------------
    async def submit(self, request: Any, key: str) -> tuple[Any, bool]:
        from .snapshots import Job

        await self._admit("job_submit")
        now = _stamp()
        job = Job(str(uuid.uuid4()), request, request.request_hash, JobState.QUEUED,
                  ReasonCode.OK, None, None, now, now, (), None)
        stored, replayed = await asyncio.to_thread(self._repo.submit, job, key)
        self._c.wake()
        return stored, replayed

    async def recheck(self, job_id: str) -> tuple[Any, Mapping[str, Any], bool]:
        from . import transitions as tr

        job = await self._job(job_id)
        attempt = None if job.attempt_id is None else \
            await asyncio.to_thread(self._repo.get_attempt, job.attempt_id)
        tr.reduce(job, attempt, tr.RecheckRequested(job_id), now=_stamp())  # legality only
        try:
            plan = await asyncio.to_thread(self._c.preflight.plan, job, Stage.CHECKING)
        except Exception as error:  # noqa: BLE001 - an unknown sample blocks, it never passes
            plan = _blocked_plan(_reason_of(error, ReasonCode.STORAGE_UNKNOWN))
        volumes = [{"required_bytes": v.required.value, "available_after_job_bytes":
                    v.available_after_job.value, "reason_code": v.reason.value} for v in plan.volumes]
        preflight = {"admitted": plan.admitted, "reason_code": plan.reason.value,
                     "checked_at": plan.admission.sample_time, "volumes": volumes}
        return job, preflight, plan.admitted

    async def retry(self, job_id: str, key: str) -> tuple[Any, str, bool]:
        from .api import ApiActionError

        await self._admit("job_retry")
        known = await asyncio.to_thread(self._repo.get_retry_attempt, key)
        if known is not None:
            if known[0] != job_id:
                raise ApiActionError(ReasonCode.CONFLICT)
            return await self._job(job_id), known[1].attempt_id, True
        await self._job(job_id)
        task = self._c.spawn(self._c.scheduler.retry(job_id, key))
        deadline = time.monotonic() + RETRY_COMMIT_SECONDS
        await asyncio.sleep(0)
        while True:
            if task.done():
                break
            known = await asyncio.to_thread(self._repo.get_retry_attempt, key)
            if known is not None or time.monotonic() >= deadline:
                break
            await asyncio.sleep(0.01)
        if known is None and task.done():
            error = task.exception()
            if error is not None:
                raise error
            known = (job_id, task.result())
        if known is None:
            raise ApiActionError(ReasonCode.SERVICE_UNAVAILABLE)
        return await self._job(job_id), known[1].attempt_id, False

    async def clone(self, job_id: str, request: Any, key: str) -> Any:
        from .api import ApiActionError

        await self._admit("job_clone")
        stored = self._clone_keys.get(key)
        if stored is not None:
            if stored[:2] != (job_id, request.request_hash):
                raise ApiActionError(ReasonCode.CONFLICT)
            return await self._job(stored[2])
        prior = await self._job(job_id)
        old, new = prior.request.to_dict(), request.to_dict()
        overrides = {name: value for name, value in new.items()
                     if name != "schema_version" and value != old.get(name)}
        job = await self._c.scheduler.clone(job_id, overrides or None)
        self._clone_keys[key] = (job_id, request.request_hash, job.job_id)
        while len(self._clone_keys) > MAX_KEYS:
            self._clone_keys.popitem(last=False)
        self._c.wake()
        return job

    async def cancel(self, job_id: str) -> tuple[Any, str]:
        from . import transitions as tr

        job = await self._job(job_id)
        if job.state in _FINAL_STATES:
            return job, "already_terminal"
        try:
            final = await self._c.scheduler.cancel(job_id)
        except tr.TransitionError as error:
            if error.reason_code is ReasonCode.ALREADY_TERMINAL:
                return await self._job(job_id), "already_terminal"
            raise
        return final, cancel_status(final)

    async def open_artifact(self, job_id: str, artifact_id: str) -> tuple[int, str, Any]:
        from .api import ApiActionError

        job = await self._job(job_id)
        artifact = next((a for a in job.artifacts
                         if a.artifact_id == artifact_id and a.available), None)
        if artifact is None:
            raise ApiActionError(ReasonCode.INVALID_INPUT, not_found=True)
        stack, held, content_type = await asyncio.to_thread(_open_published, job, artifact)
        return artifact.bytes, content_type, _ArtifactStream(stack, held, artifact.bytes,
                                                              artifact.sha256)

    # -- cache routes ------------------------------------------------------------------
    def _require_cache(self) -> Any:
        from .api import ApiActionError

        cache = self._c.ctx.cache
        if cache is None:
            raise ApiActionError(ReasonCode.CACHE_UNKNOWN)
        return cache

    async def list_cache(self, limit: int, cursor: str | None) -> tuple[list, str | None]:
        cache = self._require_cache()
        start = _cursor(cursor)
        snapshots = await asyncio.to_thread(cache.snapshots)
        page = snapshots[start:start + limit]
        entries = [{"entry_id": s.snapshot_id, "bytes": s.total_bytes, "files": len(s.files)}
                   for s in page]
        following = start + limit
        return entries, (str(following) if following < len(snapshots) else None)

    async def remove_cache(self, entry_id: str) -> Any:
        from .api import ApiActionError

        cache = self._require_cache()
        known = {s.snapshot_id for s in await asyncio.to_thread(cache.snapshots)}
        if entry_id not in known:
            raise ApiActionError(ReasonCode.INVALID_INPUT, not_found=True)
        active = await asyncio.to_thread(self._repo.active_jobs, 1)
        if active or self._c.setup_running():
            raise ApiActionError(ReasonCode.CACHE_IN_USE)
        await asyncio.to_thread(cache.remove, entry_id)
        return {"entry_id": entry_id, "removed": True}

    # -- setup routes ------------------------------------------------------------------
    def _expire(self) -> None:
        now = time.monotonic()
        for plan_id in [p for p, (at, _) in self._plans.items() if now - at > PLAN_TTL_SECONDS]:
            self._plans.pop(plan_id, None)
        while len(self._plans) > MAX_PLANS:
            self._plans.popitem(last=False)
        while len(self._operations) > MAX_OPERATIONS:
            oldest = next(iter(self._operations))
            if self._operations[oldest]["state"] == "running":
                break
            self._operations.pop(oldest)

    def _setup_inputs(self, config: Mapping[str, Any]) -> tuple[dict, str | None, str | None]:
        from .api import ApiActionError

        if not set(config) <= _SETUP_CONFIG_KEYS:
            raise ApiActionError(ReasonCode.INVALID_INPUT)
        paths = config.get("paths", {})
        manual, model = config.get("manual_backing"), config.get("model")
        if (type(paths) is not dict or not set(paths) <= _SETUP_PATH_KEYS
                or any(type(v) is not str for v in paths.values())
                or any(v is not None and type(v) is not str for v in (manual, model))):
            raise ApiActionError(ReasonCode.INVALID_INPUT)
        current = self._c.ctx.config.paths
        merged = {"data": current.data, "output": current.output, "cache": current.cache,
                  "scratch": current.scratch, "docker_backing": current.docker_backing}
        merged.update(paths)
        return merged, manual, model

    async def setup_plan(self, capabilities: list[Capability], config: Mapping[str, Any]) -> Any:
        from .setup import parse_capabilities

        paths, manual, model = self._setup_inputs(config)
        caps = tuple(sorted(c.value for c in parse_capabilities([c.value for c in capabilities])))
        result = await asyncio.to_thread(self._c.setup_actions.plan, caps, paths)
        plan_id = str(uuid.uuid4())
        self._plans[plan_id] = (time.monotonic(), {"capabilities": caps, "paths": paths,
                                                   "manual_backing": manual, "model": model})
        self._expire()
        return {"plan_id": plan_id, "capabilities": list(caps), "ready": bool(result.ok),
                "reason_code": result.reason_code.value, "next_action": result.next_action,
                "size_known": bool(result.data.get("size_known")),
                "sizes": dict(result.data.get("sizes") or {})}

    async def setup_apply(self, plan_id: str, key: str) -> Any:
        from .api import ApiActionError

        await self._admit("setup_apply")
        self._expire()
        stored = self._setup_keys.get(key)
        if stored is not None:
            # Idempotent replay: the key's original plan, or another plan with the same
            # canonical inputs (an answers-file re-run plans afresh: plan ids are per
            # request), returns the original operation's status.  The same key with
            # different inputs, or an unknown plan, is a conflict.
            stored_plan, operation_id, inputs = stored
            if stored_plan != plan_id:
                planned = self._plans.get(plan_id)
                if planned is None or planned[1] != inputs:
                    raise ApiActionError(ReasonCode.CONFLICT)
            return dict(self._operations.get(operation_id) or {"operation_id": operation_id, "state": "done"})
        entry = self._plans.get(plan_id)
        if entry is None:
            raise ApiActionError(ReasonCode.INVALID_INPUT, not_found=True)
        if self._c.setup_running():
            raise ApiActionError(ReasonCode.CONFLICT)
        inputs = entry[1]
        operation_id = str(uuid.uuid4())
        # The lease admission samples the cache this setup provisions (the plan's bound
        # path, not the configuration the service started with).  An earlier setup may
        # still hold a setup-scoped cache open on that path (closed only on rebind): on
        # Windows that sample would be ``storage_unknown``, so it is released first.
        release = getattr(getattr(self._c.setup_actions, "provision", None), "release_cache", None)
        if callable(release):
            await asyncio.to_thread(release)
        admission = await asyncio.to_thread(_host_plan, self._c.ctx, "setup", inputs["paths"]["cache"], 0)
        status = {"operation_id": operation_id, "plan_id": plan_id, "state": "running",
                  "reason_code": ReasonCode.OK.value, "next_action": ""}
        self._operations[operation_id] = status
        task = self._c.spawn(self._run_setup(operation_id, key, inputs, admission))
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        if task.done() and task.exception() is not None:
            self._operations.pop(operation_id, None)
            raise task.exception()
        self._setup_keys[key] = (plan_id, operation_id, dict(inputs))
        while len(self._setup_keys) > MAX_KEYS:
            self._setup_keys.popitem(last=False)
        return dict(status)

    async def _run_setup(self, operation_id: str, key: str, inputs: Mapping[str, Any],
                         admission: Any) -> None:
        status = self._operations[operation_id]
        comp = self._c

        def reaped() -> bool:
            """Every writer this setup started is proven finished: the probe/smoke runs
            (through the launcher) and the release pulls (the provisioning port).  False
            keeps the durable setup lease held (``Scheduler.run_setup``), so no job
            overlaps a pull the daemon may still be running."""
            provision = getattr(comp.setup_actions, "provision", None)
            return bool(comp.launcher.writers_reaped) and bool(getattr(provision, "writers_reaped", True))

        async def action() -> bool:
            try:
                result = await asyncio.to_thread(
                    comp.setup_service.apply, key, capabilities=inputs["capabilities"],
                    paths=inputs["paths"], accept_plan=True,
                    manual_backing=inputs["manual_backing"], model=inputs["model"])
            except Exception as error:  # noqa: BLE001 - closed reason only
                status.update(state="failed", reason_code=_reason_of(error).value)
                return reaped()
            status.update(state="done" if result.applied else "failed",
                          reason_code=str(result.reason_code), next_action=result.next_action,
                          setup_state=str(result.state), exit_code=int(result.exit_code))
            return reaped()

        try:
            await self._c.scheduler.run_setup(operation_id, admission, action)
        except Exception as error:  # lease refused or lost: closed reason, re-raised
            if status["state"] == "running":
                status.update(state="failed", reason_code=_reason_of(error).value)
            raise

    async def setup_status(self, operation_id: str) -> Mapping[str, Any]:
        from .api import ApiActionError

        status = self._operations.get(operation_id)
        if status is None:
            raise ApiActionError(ReasonCode.INVALID_INPUT, not_found=True)
        return dict(status)

    # -- service -----------------------------------------------------------------------
    async def stop(self) -> tuple[bool, str]:
        """Accept the stop request; the real stop work runs in ``Composition.shutdown``."""
        if not self._c.launcher.writers_reaped:
            return False, ReasonCode.WRITER_UNRESOLVED.value
        self._c.stopping = True
        return True, ReasonCode.OK.value


# --------------------------------------------------------------------------- composition


class Composition:
    """``service.ServiceComposition`` for the product."""

    def __init__(self, state_dir: str | os.PathLike, *, environ: Mapping[str, str] | None = None,
                 seams: Seams | None = None) -> None:
        self.state_dir = os.path.abspath(os.fspath(state_dir))
        self._environ = dict(os.environ if environ is None else environ)
        self._seams = seams or Seams()
        self.ctx: _Context | None = None
        self.actions: _Actions | None = None
        self.scheduler: Any = None
        self.launcher: Any = None
        self.preflight: Any = None
        self.hardware: Any = None
        self.probe: _ProbeEvidence | None = None
        self.snapshots: _Snapshots | None = None
        self.docker_status: _DockerStatus | None = None
        self.setup_actions: Any = None
        self.setup_service: Any = None
        self.stopping = False
        self._registries: tuple[Any, Any] = (None, None)
        self._recovery: Any = None
        self._pump: asyncio.Task | None = None
        self._wake: asyncio.Event | None = None
        self._tasks: set[asyncio.Task] = set()
        self._ready = False
        self._shutdown: tuple[bool, str] | None = None
        self._shutdown_lock: asyncio.Lock | None = None

    # -- service protocol --------------------------------------------------------------
    def registries(self) -> tuple[frozenset[str] | None, frozenset[str] | None]:
        return self._registries

    @property
    def service_state(self) -> ServiceState:
        return ServiceState.STOPPING if self.stopping else ServiceState.RUNNING

    async def audit(self, route_name: str, reason_code: str, ids: Mapping[str, str]) -> Any:
        """API audit sink: dotted route names become closed audit actions (fsync off-loop)."""
        ctx = self.ctx
        if ctx is None or ctx.hub is None or self._shutdown is not None:
            return None
        action = _AUDIT_ALIASES.get(route_name) or str(route_name).replace(".", "_")
        if _ACTION_RE.fullmatch(action) is None:
            action = "api_unknown"
        return await asyncio.to_thread(ctx.hub.audit, action, reason_code=reason_code,
                                       ids=dict(ids))

    async def audit_intent(self, action: str) -> None:
        """Durable intent before new work; audit loss maps to a closed API refusal."""
        from .api import ApiActionError

        try:
            await asyncio.to_thread(self.ctx.hub.audit, action, reason_code=ReasonCode.OK, ids={})
        except DomainValidationError as error:
            raise ApiActionError(error.reason_code) from None

    def persistence_loss(self) -> ReasonCode | None:
        repo_latch = self.ctx.repo.admission_blocked if self.ctx and self.ctx.repo else None
        fault = getattr(self.scheduler, "fault", None)
        return repo_latch or fault

    def setup_running(self) -> bool:
        actions = self.actions
        return actions is not None and any(
            op.get("state") == "running" for op in actions._operations.values())

    def wake(self) -> None:
        if self._wake is not None:
            self._wake.set()

    def spawn(self, coroutine: Any) -> asyncio.Task:
        task = asyncio.get_running_loop().create_task(coroutine)
        self._tasks.add(task)
        task.add_done_callback(self._task_done)
        return task

    def _task_done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            _LOG.debug("background task ended reason=%s", _reason_of(task.exception()).value)

    # -- startup -----------------------------------------------------------------------
    async def startup(self) -> None:
        try:
            await self._startup()
        except BaseException as error:
            await self._close_resources()
            if isinstance(error, asyncio.CancelledError):
                raise
            raise _closed(error) from None

    async def _startup(self) -> None:
        from . import recovery
        from .scheduler import Scheduler
        from .service import preflight_state_path
        from .volumes import make_volume_provider

        seams = self._seams
        volumes = seams.volumes or make_volume_provider()
        # 1. State-volume preflight before any state or runtime write of ours.
        await asyncio.to_thread(preflight_state_path, self.state_dir, volume_provider=volumes)
        if seams.docker_storage is None:
            from .docker_storage import make_docker_storage_provider

            docker_storage = make_docker_storage_provider(volumes=volumes)
        else:
            docker_storage = seams.docker_storage
        ctx = self.ctx = _Context(self.state_dir, self._environ, _Clock(), volumes, docker_storage)
        # 2. Runtime manifest and configuration.  The explicit fixture wins; otherwise the
        # packaged release manifest for the Docker engine's architecture, which the host
        # hardware port samples off the loop (bounded) only when the release is publishable.
        if seams.hardware is not None:
            self.hardware = seams.hardware
        else:
            from .hardware import HardwareSampler

            self.hardware = HardwareSampler()
        hardware = self.hardware

        def engine_arch() -> str | None:
            try:
                return _measured_text(hardware.sample(), "engine_arch")
            except Exception:  # noqa: BLE001 - an unsampled engine is unknown
                return None

        ctx.manifest_source = await asyncio.to_thread(resolve_runtime_manifest, self._environ,
                                                      engine_arch=engine_arch)
        ctx.manifest = ctx.manifest_source.manifest
        ctx.registries = self._registries = registries_of(ctx.manifest)
        ctx.config = await asyncio.to_thread(load_config, self._environ, ctx.registries)
        ctx.volumes = _HeldRootVolumes(volumes, (ctx.config.paths.cache,))
        # 3. Repository.
        await asyncio.to_thread(self._open_repository)
        # 4. Telemetry hub.
        from .telemetry import make_telemetry

        ctx.hub = await asyncio.to_thread(make_telemetry, self.state_dir, ctx.config.telemetry,
                                          clock=ctx.clock, db_bytes=ctx.repo.db_bytes)
        ctx.records_dir = await asyncio.to_thread(
            _private_dir, os.path.join(self.state_dir, VERIFIED_DIRNAME))
        ctx.scratch = _ScratchIndex(await asyncio.to_thread(_private_dir, ctx.config.paths.scratch))
        ctx.cache = await asyncio.to_thread(self._open_cache)
        # 5. Startup reconciliation of any previous instance (containers, jobs, partials).
        docker = _DockerAccess()
        ctx.instance = f"pm-{uuid.uuid4().hex}"
        self._recovery = recovery.Recovery(self.state_dir, instance_nonce=ctx.instance)
        containers = seams.containers or _LazyContainers(docker)
        report = await recovery.startup_reconcile(
            self.state_dir, ctx.repo, containers, process_identity=recovery.psutil_identity,
            audit=ctx.hub.recovery_audit(), clock=ctx.clock, output_roots=(ctx.config.paths.output,),
            recovery=self._recovery)
        if not report.ok:
            from .service import ServiceError

            raise ServiceError(report.reason_code)
        # Every earlier instance is proven clear: record only ours for the next reconcile.
        await asyncio.to_thread(recovery.write_intent, self.state_dir, {
            "action": "running", "instance_nonce": ctx.instance,
            "instance_nonces": [ctx.instance], "ts": ctx.clock.now()})
        # 6. Scheduler and its ports: the probe evidence port feeds model selection and setup.
        if seams.hardware is not None:
            probe_hardware = seams.hardware
        else:
            from .hardware import HardwareSampler

            # Separate instances: a sampler refuses overlapping samples, and the snapshot
            # builder (``self.hardware``, which also sampled the engine architecture in
            # step 2) must never make the probe's GPU check (or vice versa) fail spuriously.
            probe_hardware = HardwareSampler()
        launch = seams.probe(ctx) if seams.probe else _DockerLaunch(ctx, docker)
        self.probe = _ProbeEvidence(ctx, launch, hardware=probe_hardware)
        if seams.acquire is not None:
            ctx.acquire_model = seams.acquire(ctx)
        elif ctx.cache is not None:
            ctx.acquire_model = ctx.cache.acquire
        inner_models = seams.models(ctx) if seams.models else _ManifestModels(ctx, self.probe)
        ctx.models = _TrackedModels(inner_models)
        self.launcher = (seams.launcher(ctx) if seams.launcher
                         else DockerWorkerLauncher(ctx, docker, probe=self.probe))
        self.preflight = _StoragePreflight(ctx)
        self.scheduler = Scheduler(ctx.repo, preflight=self.preflight, launcher=self.launcher,
                                   models=ctx.models, exporter=_Exporter(ctx),
                                   publisher=_Publisher(ctx), clock=ctx.clock,
                                   telemetry=ctx.hub.scheduler_telemetry(),
                                   limits=ctx.config.limits)
        self._build_setup()
        self.docker_status = _DockerStatus(docker_storage, ctx.config.paths.docker_backing)
        self.snapshots = _Snapshots(self)
        # 7. Pump task, 8. actions.
        self._wake = asyncio.Event()
        self._shutdown_lock = asyncio.Lock()
        self._pump = asyncio.get_running_loop().create_task(self._run_pump())
        self.actions = _Actions(self)
        self._ready = True

    def _open_repository(self) -> None:
        from .repository import Repository

        ctx = self.ctx
        volumes = ctx.volumes

        def state_volume(path: str) -> bool | None:
            return volumes.sample(path, sqlite_state=True).state_allowed.value

        models, languages = ctx.registries
        ctx.repo = Repository.open(os.path.join(self.state_dir, DATABASE_NAME), clock=_utcnow,
                                   state_volume=state_volume,
                                   fault_hook=self._seams.repository_fault_hook,
                                   limits=ctx.config.limits, verified_models=models,
                                   verified_languages=languages)

    def _open_cache(self, path: str | None = None) -> Any:
        return _open_model_cache(self.ctx, path)

    def _build_setup(self) -> None:
        from .setup import LocalSetupActions, SetupService

        ctx = self.ctx
        provisioning = (self._seams.provisioning(ctx) if self._seams.provisioning
                        else _Provisioning(ctx, self._seams.hardware, probe=self.probe,
                                           loop=asyncio.get_running_loop()))
        audit = _setup_audit(ctx.hub)

        def cached_snapshot(choice: str) -> str | None:
            if ctx.manifest is None:
                return None
            model = next((m for m in ctx.manifest.models if m.artifact.model_id == choice), None)
            if model is None:
                return None
            # The cache being provisioned: a port bound to another cache path answers for
            # that one, so a model cached only in the service's cache is acquired afresh.
            listing = getattr(provisioning, "setup_snapshots", None)
            if callable(listing):
                snapshots = listing()
            elif ctx.cache is not None:
                snapshots = ctx.cache.snapshots()
            else:
                return None
            return next((s.snapshot_id for s in snapshots if tuple(s.files) == tuple(model.files)), None)

        self.setup_actions = LocalSetupActions(
            provisioning=provisioning, config_path=config_path(self._environ), audit=audit,
            volumes=ctx.volumes, docker=ctx.docker_storage, snapshot_source=cached_snapshot,
            sizes=_setup_sizes(ctx.manifest, _release_image_present))
        self.setup_service = SetupService(self.setup_actions, audit=audit)

    async def _run_pump(self) -> None:
        """Background run loop: one queued job at a time, until shutdown quiesces."""
        while not self.stopping:
            ran = False
            if self.persistence_loss() is None:
                try:
                    ran = await self.scheduler.run_once()
                except Exception as error:  # noqa: BLE001 - recorded as a closed reason
                    _LOG.warning("scheduler run failed reason=%s", _reason_of(error).value)
                    with contextlib.suppress(Exception):
                        self.ctx.hub.log(logging.WARNING, "scheduler_run_failed",
                                         component="service", reason_code=_reason_of(error))
            if self.stopping:
                return
            if not ran:
                self._wake.clear()
                with contextlib.suppress(TimeoutError):
                    async with asyncio.timeout(PUMP_IDLE_SECONDS):
                        await self._wake.wait()

    # -- shutdown ----------------------------------------------------------------------
    async def shutdown(self) -> tuple[bool, str]:
        """Graceful stop, then close hub, repository and cache.  Idempotent."""
        if self._shutdown is not None:
            return self._shutdown
        if self._shutdown_lock is None:
            self._shutdown_lock = asyncio.Lock()
        async with self._shutdown_lock:
            if self._shutdown is None:
                self._shutdown = await self._stop()
        return self._shutdown

    async def _stop(self) -> tuple[bool, str]:
        from . import recovery

        self.stopping = True
        self.wake()
        ok, reason = True, ReasonCode.SERVICE_STOPPED
        ctx = self.ctx
        if self._ready and ctx is not None:
            try:
                report = await recovery.graceful_stop(
                    self.scheduler, ctx.repo, self.launcher, audit=ctx.hub.recovery_audit(),
                    clock=ctx.clock, recovery=self._recovery,
                    quiesce_timeout=self._seams.quiesce_timeout)
                ok, reason = report.ok, report.reason_code
            except Exception as error:  # noqa: BLE001 - an unproven stop is a failure
                ok, reason = False, _reason_of(error, ReasonCode.WRITER_UNRESOLVED)
            pending = [t for t in (self._pump, *self._tasks) if t is not None and not t.done()]
            if pending:
                _done, still = await asyncio.wait(pending, timeout=SHUTDOWN_TASK_SECONDS)
                if still:
                    ok, reason = False, ReasonCode.WRITER_UNRESOLVED
                    for task in still:
                        task.cancel()
            if ok:
                # Writers proven reaped and jobs recorded: no instance is left to reconcile.
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(recovery.write_intent, self.state_dir, {
                        "action": "stopped", "ts": ctx.clock.now()})
            if self.docker_status is not None:
                await self.docker_status.close()
        await self._close_resources()
        return ok, reason.value

    async def _close_resources(self) -> None:
        ctx = self.ctx
        if ctx is None:
            return
        closer = getattr(getattr(self.setup_actions, "provision", None), "close", None)
        if callable(closer):  # the setup-scoped cache, when setup bound another cache path
            with contextlib.suppress(Exception):
                await asyncio.to_thread(closer)
        if ctx.cache is not None:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(ctx.cache.close)
        if ctx.hub is not None:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(ctx.hub.shutdown)
        if ctx.repo is not None:
            with contextlib.suppress(Exception):
                await asyncio.to_thread(ctx.repo.close)


def make_service(state_dir: str) -> Composition:
    """The service factory seam (``service.DEFAULT_FACTORY``)."""
    return Composition(state_dir)


def attempt_scratch(scratch_root: str, attempt_id: str) -> str:
    """Root of one attempt's private scratch (per-assignment folders live below it)."""
    return _ScratchIndex(scratch_root).attempt_root(attempt_id)


__all__ = (
    "CONFIG_ENV",
    "MANIFEST_ENV",
    "PACKAGED_ASSET_ORIGINS",
    "RELEASE_DOCUMENT",
    "RELEASE_DOCUMENT_VERSION",
    "Composition",
    "DockerWorkerLauncher",
    "ManifestResolution",
    "Seams",
    "attempt_scratch",
    "cancel_status",
    "client_startup",
    "config_path",
    "load_config",
    "load_runtime_manifest",
    "make_service",
    "registries_of",
    "resolve_docker_target",
    "resolve_runtime_manifest",
)
