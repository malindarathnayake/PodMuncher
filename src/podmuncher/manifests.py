"""Strict, bounded runtime and model manifest loading.

Development manifests are accepted only when explicitly requested and labelled
as fixtures.  Release manifests are accepted only from an exact caller
allowlist.  This module performs no network access.

A development manifest may declare each model either as a synthetic fixture
(``synthetic: true`` with ``fixture:`` sources verified against local files)
or as a real pinned model (``synthetic: false`` with HTTPS sources under the
same URL, origin and per-file size/sha256 rules that release manifests use).
Real development models are verified at acquisition by the model cache, not
here, and never make the manifest a release: ``development_fixture`` stays
true and development images stay local identities.

Measured envelopes are optional per model (SPEC_GAP-16).  A model either
carries the complete measured set (``worker_peak_ram_bytes``,
``worker_peak_vram_bytes``, ``validated_max_duration_seconds``,
``decode_bytes_per_second``, ``decode_timeout_seconds``, ``evidence_sha256``
and ``release_tested``) or none of it (keys absent or null).  Partial sets are
rejected.  Artifact identity, files, sources and licences stay required, so
integrity is unchanged.  Model fit for a model without an envelope comes from
the shipped hardware table and live worker evidence, not from this loader.

Multi-architecture images (SPEC_GAP-17).  A capability entry is keyed by its
image *index* digest and always names one ``platform``.  It may also carry an
optional ``platforms`` list (every id in the root ``platforms`` list, unique,
and including ``platform``) when that index resolves on more than one engine
architecture.  ``ResolvedCapability.platforms`` is the effective tuple and
``supports_engine_arch`` maps a Docker engine architecture onto it, so
bootstrap can skip an image the engine cannot run.
"""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import SplitResult, urlsplit

from .domains import Backend, Capability, DomainValidationError, ModelTier, ReasonCode
from .model_policy import (
    DECODE_BYTES_PER_SECOND,
    MeasuredEnvelope,
    ModelArtifact,
    RuntimeFingerprint,
)
from .runtime import WorkerPolicy

MAX_MANIFEST_BYTES = 1024 * 1024
MAX_JSON_DEPTH = 12
MAX_COLLECTION_ITEMS = 1024
MAX_STRING_BYTES = 4096
MAX_FIXTURE_FILE_BYTES = 64 * 1024**2
MIN_WORKER_OVERHEAD_BYTES = 256 * 1024**2 + 2 * 10 * 1024**2
_MAX_BYTES = 2**63 - 1
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_IMAGE = re.compile(r"[a-z0-9][a-z0-9./:_-]{0,240}\Z")
_HOST_LABEL = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\Z")
_GPU_UUID = re.compile(r"GPU-[a-fA-F0-9]{8}(?:-[a-fA-F0-9]{4}){3}-[a-fA-F0-9]{12}\Z")
_PLACEHOLDER = re.compile(r"(?:todo|tbd|unknown|unresolved|placeholder)", re.IGNORECASE)
_REPARSE_POINT = 0x400
# Docker engine architecture (``docker info --format {{.Architecture}}`` or a
# normalised amd64/arm64 value) to the manifest platform id that runs on it.
# Bounded on purpose: the images are published for these two only.
ENGINE_ARCH_PLATFORMS: dict[str, str] = {
    "x86_64": "linux-amd64",
    "amd64": "linux-amd64",
    "aarch64": "linux-arm64",
    "arm64": "linux-arm64",
}
# The measured-envelope keys: all present (a measured envelope) or all absent
# or null (no envelope).  Any other combination is a partial envelope and fails.
ENVELOPE_KEYS = frozenset({
    "worker_peak_ram_bytes", "worker_peak_vram_bytes", "validated_max_duration_seconds",
    "decode_bytes_per_second", "decode_timeout_seconds", "evidence_sha256", "release_tested",
})


class ManifestError(DomainValidationError):
    """A manifest failure whose message never contains source or input data."""

    def __init__(self, reason_code: ReasonCode) -> None:
        super().__init__(reason_code)


@dataclass(frozen=True, slots=True)
class ManifestFile:
    relative_path: str
    bytes: int
    sha256: str


@dataclass(frozen=True, slots=True)
class ManifestModel:
    """One verified model.  The envelope fields are all set or all ``None``."""

    artifact: ModelArtifact
    source_urls: tuple[str, ...]
    files: tuple[ManifestFile, ...]
    license: str
    provenance: str
    compute_types: tuple[str, ...]
    worker_peak_ram_bytes: int | None
    worker_peak_vram_bytes: int | None
    validated_max_duration_seconds: float | None
    decode_bytes_per_second: int | None
    decode_timeout_seconds: float | None
    codec_matrix: tuple[str, ...]
    evidence_sha256: str | None
    release_tested: bool | None
    synthetic: bool

    @property
    def measured(self) -> bool:
        """True when the model carries a complete measured envelope."""
        return self.evidence_sha256 is not None


@dataclass(frozen=True, slots=True)
class ManifestCapability:
    capability: Capability
    platform: str
    platforms: tuple[str, ...]
    image_reference: str
    digest: str
    entrypoint: tuple[str, ...]
    compressed_bytes: int
    unpacked_peak_bytes: int
    worker_overhead_bytes: int
    worker_memory_bytes: int
    dependency_lock_sha256: str
    licenses: tuple[str, ...]
    provenance: str
    runtime_version: str
    driver_version: str | None
    hardware_fingerprint: str


@dataclass(frozen=True, slots=True)
class RuntimeManifest:
    release_version: str
    development_fixture: bool
    platforms: tuple[str, ...]
    capabilities: tuple[ManifestCapability, ...]
    models: tuple[ManifestModel, ...]
    languages: tuple[str, ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class ResolvedCapability:
    """Values ready for runtime and model-policy consumers."""

    capability: Capability
    worker_policy: WorkerPolicy
    artifacts: tuple[ModelArtifact, ...]
    envelopes: tuple[MeasuredEnvelope, ...]
    model_files: tuple[tuple[str, tuple[ManifestFile, ...]], ...]
    languages: tuple[str, ...]
    worker_overhead_bytes: int
    dependency_lock_sha256: str
    licenses: tuple[str, ...]
    provenance: str
    platforms: tuple[str, ...]


def _failure(reason: ReasonCode = ReasonCode.ASSET_VERIFICATION_FAILED) -> ManifestError:
    return ManifestError(reason)


def _object(
    value: object,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    if type(value) is not dict or not required <= set(value) <= required | optional:
        raise _failure()
    return value


def _string(value: object, *, identifier: bool = False, release: bool = False) -> str:
    if type(value) is not str or not value or len(value.encode("utf-8")) > MAX_STRING_BYTES:
        raise _failure()
    if any(ord(character) < 32 for character in value):
        raise _failure()
    if identifier and _ID.fullmatch(value) is None:
        raise _failure()
    if release and (_PLACEHOLDER.search(value) is not None
                    or value.lower() in {"fixture", "development", "dev", "test", "local", "__invalid__"}):
        raise _failure()
    return value


def _sha256(value: object) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise _failure()
    return value


def _image_digest(value: object) -> str:
    if type(value) is not str or not value.startswith("sha256:"):
        raise _failure()
    _sha256(value[7:])
    return value


def _integer(value: object, *, minimum: int = 0) -> int:
    if type(value) is not int or not minimum <= value <= _MAX_BYTES:
        raise _failure()
    return value


def _positive_number(value: object) -> float:
    if type(value) not in (int, float):
        raise _failure()
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise _failure() from None
    if not math.isfinite(result) or not 0 < result <= _MAX_BYTES:
        raise _failure()
    return result


def _tuple(value: object, parser, *, nonempty: bool = True) -> tuple:
    if type(value) is not list or len(value) > MAX_COLLECTION_ITEMS or (nonempty and not value):
        raise _failure()
    return tuple(parser(item) for item in value)


def _unique(values: tuple, key) -> None:
    keys = tuple(key(value) for value in values)
    if len(set(keys)) != len(keys):
        raise _failure()


def _safe_relative_path(value: object) -> str:
    path = _string(value)
    pure = PurePosixPath(path)
    if (path.startswith(("/", "\\")) or "\\" in path or ":" in path or pure.is_absolute()
            or str(pure) != path or path in {".", ".."}
            or any(part in {"", ".", ".."} for part in pure.parts)
            or len(pure.parts) > 16):
        raise _failure()
    return path


def _bounded_json(value: object, depth: int = 0) -> None:
    if depth > MAX_JSON_DEPTH:
        raise _failure()
    if type(value) is str:
        _string(value)
    elif value is None or type(value) in (bool, int):
        return
    elif type(value) is float:
        if not math.isfinite(value):
            raise _failure()
    elif type(value) is list:
        if len(value) > MAX_COLLECTION_ITEMS:
            raise _failure()
        for item in value:
            _bounded_json(item, depth + 1)
    elif type(value) is dict:
        if len(value) > MAX_COLLECTION_ITEMS:
            raise _failure()
        for key, item in value.items():
            _string(key)
            _bounded_json(item, depth + 1)
    else:
        raise _failure()


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise _failure()
        result[key] = value
    return result


def _load_json(raw: bytes) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(_failure()),
        )
        _bounded_json(value)
    except ManifestError:
        raise
    except (UnicodeError, json.JSONDecodeError, RecursionError, MemoryError, ValueError, TypeError):
        raise _failure() from None
    if type(value) is not dict:
        raise _failure()
    return value


def _https_url(value: object) -> tuple[str, SplitResult, str]:
    try:
        url = _string(value, release=True)
    except ManifestError:
        raise _failure(ReasonCode.ASSET_SOURCE_REJECTED) from None
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        raise _failure(ReasonCode.ASSET_SOURCE_REJECTED) from None
    if (parsed.scheme != "https" or not parsed.netloc or hostname is None
            or parsed.username is not None or parsed.password is not None
            or parsed.fragment or "\\" in url or any(ord(character) > 127 for character in hostname)):
        raise _failure(ReasonCode.ASSET_SOURCE_REJECTED)
    host = hostname.lower()
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        labels = host.rstrip(".").split(".")
        if (host.endswith((".", ".localhost", ".local", ".lan", ".home", ".internal"))
                or len(host) > 253 or len(labels) < 2
                or any(_HOST_LABEL.fullmatch(label) is None for label in labels)
                or labels[-1].isdigit()
                or host == "localhost"):
            raise _failure(ReasonCode.ASSET_SOURCE_REJECTED) from None
    else:
        if not address.is_global:
            raise _failure(ReasonCode.ASSET_SOURCE_REJECTED)
    origin = f"https://[{host}]" if ":" in host else f"https://{host}"
    if port is not None:
        origin += f":{port}"
    return url, parsed, origin


def _asset_allowlist(values: tuple[str, ...]) -> tuple[tuple[str, str], ...]:
    if type(values) is not tuple or len(values) > MAX_COLLECTION_ITEMS:
        raise _failure(ReasonCode.ASSET_SOURCE_REJECTED)
    parsed_values = []
    for value in values:
        url, parsed, origin = _https_url(value)
        if parsed.path in ("", "/") and not parsed.query:
            parsed_values.append(("origin", origin))
        else:
            parsed_values.append(("url", url))
    if len(set(parsed_values)) != len(parsed_values):
        raise _failure(ReasonCode.ASSET_SOURCE_REJECTED)
    return tuple(parsed_values)


def _source_url(
    value: object,
    *,
    fixture: bool,
    allowed_assets: tuple[tuple[str, str], ...],
    enforce_allowlist: bool,
) -> str:
    """One model source: ``fixture:`` for synthetic models, pinned HTTPS otherwise.

    ``enforce_allowlist`` is always true for release manifests (an empty
    allowlist admits nothing) and true for development manifests only when the
    caller supplied ``allowed_asset_sources``; a development manifest without
    an allowlist still admits only URLs that pass every ``_https_url`` rule.
    """
    try:
        url = _string(value, release=not fixture)
    except ManifestError:
        raise _failure(ReasonCode.ASSET_SOURCE_REJECTED) from None
    if fixture:
        if not url.startswith("fixture:"):
            raise _failure(ReasonCode.ASSET_SOURCE_REJECTED)
    else:
        url, _parsed, origin = _https_url(url)
        if enforce_allowlist and (
                ("url", url) not in allowed_assets and ("origin", origin) not in allowed_assets):
            raise _failure(ReasonCode.ASSET_SOURCE_REJECTED)
    return url


def _license(value: object, *, release: bool) -> str:
    return _string(value, release=release)


def _file(value: object) -> ManifestFile:
    item = _object(value, frozenset({"relative_path", "bytes", "sha256"}))
    return ManifestFile(
        _safe_relative_path(item["relative_path"]),
        _integer(item["bytes"], minimum=1),
        _sha256(item["sha256"]),
    )


def _model(
    value: object,
    *,
    fixture: bool,
    allowed_assets: tuple[tuple[str, str], ...],
) -> ManifestModel:
    item = _object(value, frozenset({
        "id", "tier", "revision", "digest", "source_urls", "files", "license", "provenance",
        "compute_types", "codec_matrix", "synthetic",
    }), optional=ENVELOPE_KEYS)
    given = frozenset(key for key in ENVELOPE_KEYS if item.get(key) is not None)
    if given and given != ENVELOPE_KEYS:
        # A partial envelope is neither measured nor absent.
        raise _failure()
    measured = bool(given)
    try:
        tier = ModelTier.parse(item["tier"])
    except DomainValidationError:
        raise _failure() from None
    files = _tuple(item["files"], _file)
    _unique(files, lambda entry: entry.relative_path)
    compute_types = _tuple(item["compute_types"], lambda entry: _string(entry, identifier=True))
    codecs = _tuple(item["codec_matrix"], lambda entry: _string(entry, identifier=True))
    _unique(compute_types, lambda entry: entry)
    _unique(codecs, lambda entry: entry)
    synthetic = item["synthetic"]
    if type(synthetic) is not bool:
        raise _failure()
    if synthetic and not fixture:
        # Synthetic fixtures exist only in development manifests.
        raise _failure()
    release_tested = item["release_tested"] if measured else None
    if measured:
        if type(release_tested) is not bool:
            raise _failure()
        if release_tested is not True and (synthetic or not fixture):
            # A measured envelope on a synthetic fixture or in a release
            # manifest must be release tested.  A real development model
            # carries release_tested verbatim: resolve_capability hands only
            # tested envelopes to the model policy.
            raise _failure()
    # Real models (development or release) use pinned HTTPS sources, one per
    # declared file, so the model cache can verify size and sha256 per download.
    source_urls = _tuple(
        item["source_urls"],
        lambda entry: _source_url(
            entry,
            fixture=synthetic,
            allowed_assets=allowed_assets,
            enforce_allowlist=not fixture or bool(allowed_assets),
        ),
    )
    if not synthetic and len(source_urls) != len(files):
        raise _failure()
    try:
        artifact = ModelArtifact(
            _string(item["id"], identifier=True),
            tier,
            _string(item["revision"], release=not fixture),
            _sha256(item["digest"]),
        )
    except DomainValidationError:
        raise _failure() from None
    if measured:
        peak_ram = _integer(item["worker_peak_ram_bytes"], minimum=1)
        peak_vram = _integer(item["worker_peak_vram_bytes"])
        max_duration = _positive_number(item["validated_max_duration_seconds"])
        decode_rate = _integer(item["decode_bytes_per_second"], minimum=DECODE_BYTES_PER_SECOND)
        decode_timeout = _positive_number(item["decode_timeout_seconds"])
        evidence = _sha256(item["evidence_sha256"])
    else:
        peak_ram = peak_vram = max_duration = decode_rate = decode_timeout = evidence = None
    return ManifestModel(
        artifact,
        source_urls,
        files,
        _license(item["license"], release=not fixture),
        _string(item["provenance"], release=not fixture),
        compute_types,
        peak_ram,
        peak_vram,
        max_duration,
        decode_rate,
        decode_timeout,
        codecs,
        evidence,
        release_tested,
        synthetic,
    )


def _capability(value: object, *, fixture: bool, platforms: tuple[str, ...]) -> ManifestCapability:
    item = _object(value, frozenset({
        "id", "platform", "image_reference", "digest", "entrypoint", "compressed_bytes",
        "unpacked_peak_bytes", "worker_overhead_bytes", "worker_memory_bytes",
        "dependency_lock_sha256", "licenses", "provenance", "runtime_version",
        "driver_version", "hardware_fingerprint",
    }), optional=frozenset({"platforms"}))
    try:
        capability = Capability.parse(item["id"])
    except DomainValidationError:
        raise _failure() from None
    platform = _string(item["platform"], identifier=True)
    if platform not in platforms:
        raise _failure()
    # Optional multi-architecture list (SPEC_GAP-17).  Absent or null keeps the
    # single-platform form.  When given it is a non-empty, unique list of root
    # platform ids that includes ``platform``, so the two never contradict.
    if item.get("platforms") is None:
        entry_platforms = (platform,)
    else:
        entry_platforms = _tuple(
            item["platforms"], lambda entry: _string(entry, identifier=True),
        )
        _unique(entry_platforms, lambda entry: entry)
        if (platform not in entry_platforms
                or any(entry not in platforms for entry in entry_platforms)):
            raise _failure()
    digest = _image_digest(item["digest"])
    reference = _string(item["image_reference"], release=not fixture)
    if fixture:
        if reference != "fixture-local":
            raise _failure(ReasonCode.ASSET_SOURCE_REJECTED)
    elif _IMAGE.fullmatch(reference) is None or "@sha256:" in reference:
        raise _failure(ReasonCode.ASSET_SOURCE_REJECTED)
    compressed = _integer(item["compressed_bytes"], minimum=1)
    unpacked = _integer(item["unpacked_peak_bytes"], minimum=1)
    overhead = _integer(item["worker_overhead_bytes"], minimum=MIN_WORKER_OVERHEAD_BYTES)
    memory = _integer(item["worker_memory_bytes"], minimum=64 * 1024**2)
    if unpacked < compressed or overhead > 128 * 1024**3 or memory > 128 * 1024**3:
        raise _failure()
    entrypoint = _tuple(item["entrypoint"], lambda entry: _string(entry), nonempty=True)
    if len(entrypoint) > 16 or any(len(argument) > 1024 for argument in entrypoint):
        raise _failure()
    licenses = _tuple(item["licenses"], lambda entry: _license(entry, release=not fixture))
    _unique(licenses, lambda entry: entry)
    driver = item["driver_version"]
    if driver is not None:
        driver = _string(driver, release=not fixture)
    if capability is Capability.ASR_CUDA and driver is None:
        raise _failure()
    if capability is not Capability.ASR_CUDA and driver is not None:
        raise _failure()
    return ManifestCapability(
        capability,
        platform,
        entry_platforms,
        reference,
        digest,
        entrypoint,
        compressed,
        unpacked,
        overhead,
        memory,
        _sha256(item["dependency_lock_sha256"]),
        licenses,
        _string(item["provenance"], release=not fixture),
        _string(item["runtime_version"], release=not fixture),
        driver,
        _string(item["hardware_fingerprint"], release=not fixture),
    )


@dataclass(slots=True)
class _FixtureRootAnchor:
    path: Path
    descriptor: int | None = None
    handle: int | None = None
    identity: tuple[int, int] | None = None

    def close(self) -> None:
        if self.descriptor is not None:
            descriptor = self.descriptor
            self.descriptor = None
            os.close(descriptor)
        if self.handle is not None:
            handle = self.handle
            self.handle = None
            _close_windows_handle(handle)


def _open_fixture_root(path: Path) -> _FixtureRootAnchor:
    root = path.parent
    if os.name == "posix":
        directory_flag = getattr(os, "O_DIRECTORY", 0)
        nofollow_flag = getattr(os, "O_NOFOLLOW", 0)
        if not directory_flag or not nofollow_flag:
            raise _failure(ReasonCode.ASSET_SOURCE_REJECTED)
        descriptor = None
        try:
            before = root.stat(follow_symlinks=False)
            if not stat.S_ISDIR(before.st_mode) or _is_reparse(before):
                raise _failure(ReasonCode.ASSET_SOURCE_REJECTED)
            descriptor = os.open(root, os.O_RDONLY | directory_flag | nofollow_flag)
            opened = os.fstat(descriptor)
            if (not stat.S_ISDIR(opened.st_mode) or _is_reparse(opened)
                    or _identity(opened) != _identity(before)):
                raise _failure(ReasonCode.ASSET_SOURCE_REJECTED)
            return _FixtureRootAnchor(root, descriptor=descriptor, identity=_identity(opened))
        except ManifestError:
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    raise _failure(ReasonCode.ASSET_SOURCE_REJECTED) from None
            raise
        except (OSError, TypeError, ValueError):
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            raise _failure(ReasonCode.ASSET_SOURCE_REJECTED) from None
    if os.name == "nt":
        try:
            before = root.stat(follow_symlinks=False)
            if not stat.S_ISDIR(before.st_mode) or _is_reparse(before):
                raise _failure(ReasonCode.ASSET_SOURCE_REJECTED)
            handle, _handle_identity = _windows_directory_handle(root)
            try:
                if (_normalized_windows_handle_path(handle)
                        != _normalized_windows_declared_path(root)):
                    raise OSError("root handle path mismatch")
            except Exception:
                _close_windows_handle(handle)
                raise
            after = root.stat(follow_symlinks=False)
            if (not stat.S_ISDIR(after.st_mode) or _is_reparse(after)
                    or _identity(after) != _identity(before)):
                _close_windows_handle(handle)
                raise _failure(ReasonCode.ASSET_SOURCE_REJECTED)
            return _FixtureRootAnchor(
                root,
                handle=handle,
                identity=_identity(before),
            )
        except ManifestError:
            raise
        except (OSError, TypeError, ValueError):
            raise _failure(ReasonCode.ASSET_SOURCE_REJECTED) from None
    raise _failure(ReasonCode.ASSET_SOURCE_REJECTED)


def _read_fixture_source(path: Path, anchor: _FixtureRootAnchor) -> bytes:
    descriptor = None
    try:
        before = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or _is_reparse(before):
            raise _failure(ReasonCode.ASSET_SOURCE_REJECTED)
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        if os.name == "posix":
            if anchor.descriptor is None:
                raise _failure(ReasonCode.ASSET_SOURCE_REJECTED)
            descriptor = os.open(path.name, flags, dir_fd=anchor.descriptor)
        else:
            descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            opened = os.fstat(stream.fileno())
            if (not stat.S_ISREG(opened.st_mode) or _is_reparse(opened)
                    or _identity(opened) != _identity(before)):
                raise _failure(ReasonCode.ASSET_SOURCE_REJECTED)
            if os.name == "nt":
                _verify_windows_open_path(anchor, stream.fileno(), (path.name,))
            raw = stream.read(MAX_MANIFEST_BYTES + 1)
            if os.name == "nt":
                _verify_windows_open_path(anchor, stream.fileno(), (path.name,))
    except ManifestError:
        raise
    except (OSError, TypeError, ValueError):
        raise _failure(ReasonCode.ASSET_SOURCE_REJECTED) from None
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
    if not raw or len(raw) > MAX_MANIFEST_BYTES:
        raise _failure()
    return raw


def _read_release_source(
    path: Path,
    allowed_sources: tuple[Path, ...],
) -> bytes:
    try:
        before = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or _is_reparse(before):
            raise _failure(ReasonCode.ASSET_SOURCE_REJECTED)
        allowed_identity = None
        for allowed in allowed_sources:
            allowed_info = allowed.stat(follow_symlinks=False)
            if not stat.S_ISREG(allowed_info.st_mode) or _is_reparse(allowed_info):
                raise _failure(ReasonCode.ASSET_SOURCE_REJECTED)
            if path == allowed:
                allowed_identity = _identity(allowed_info)
        if allowed_identity is None or _identity(before) != allowed_identity:
            raise _failure(ReasonCode.ASSET_SOURCE_REJECTED)
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            opened = os.fstat(stream.fileno())
            if (not stat.S_ISREG(opened.st_mode) or _is_reparse(opened)
                    or _identity(opened) != allowed_identity):
                raise _failure(ReasonCode.ASSET_SOURCE_REJECTED)
            raw = stream.read(MAX_MANIFEST_BYTES + 1)
        after = path.stat(follow_symlinks=False)
        if (not stat.S_ISREG(after.st_mode) or _is_reparse(after)
                or _identity(after) != allowed_identity):
            raise _failure(ReasonCode.ASSET_SOURCE_REJECTED)
    except ManifestError:
        raise
    except (OSError, ValueError):
        raise _failure(ReasonCode.ASSET_SOURCE_REJECTED) from None
    if not raw or len(raw) > MAX_MANIFEST_BYTES:
        raise _failure()
    return raw


def _identity(info: os.stat_result) -> tuple[int, int]:
    return info.st_dev, info.st_ino


def _is_reparse(info: os.stat_result) -> bool:
    return bool(getattr(info, "st_file_attributes", 0) & _REPARSE_POINT)


def _read_declared_file(stream, declared: ManifestFile) -> tuple[int, str]:
    digest = hashlib.sha256()
    count = 0
    while True:
        chunk = stream.read(min(1024 * 1024, declared.bytes - count + 1))
        if not chunk:
            break
        count += len(chunk)
        if count > declared.bytes:
            raise _failure()
        digest.update(chunk)
    return count, digest.hexdigest()


def _verify_posix_fixture_file(
    root_descriptor: int,
    parts: tuple[str, ...],
    declared: ManifestFile,
    opened_identities: set[tuple[int, int]],
) -> tuple[int, str]:
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    nofollow_flag = getattr(os, "O_NOFOLLOW", 0)
    if not directory_flag or not nofollow_flag:
        raise _failure()
    directory_descriptors: list[tuple[int, tuple[int, int]]] = []
    file_descriptor: int | None = None
    try:
        root_opened = os.fstat(root_descriptor)
        if not stat.S_ISDIR(root_opened.st_mode) or _is_reparse(root_opened):
            raise _failure()
        parent_descriptor = root_descriptor
        for part in parts[:-1]:
            descriptor = os.open(
                part,
                os.O_RDONLY | directory_flag | nofollow_flag,
                dir_fd=parent_descriptor,
            )
            directory_descriptors.append((descriptor, (-1, -1)))
            opened = os.fstat(descriptor)
            if not stat.S_ISDIR(opened.st_mode) or _is_reparse(opened):
                raise _failure()
            directory_descriptors[-1] = (descriptor, _identity(opened))
            parent_descriptor = descriptor

        before = os.stat(parts[-1], dir_fd=parent_descriptor, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or _is_reparse(before):
            raise _failure()
        file_descriptor = os.open(
            parts[-1],
            os.O_RDONLY | getattr(os, "O_BINARY", 0) | nofollow_flag,
            dir_fd=parent_descriptor,
        )
        opened = os.fstat(file_descriptor)
        opened_identity = _identity(opened)
        if (not stat.S_ISREG(opened.st_mode) or _is_reparse(opened)
                or opened_identity != _identity(before)
                or opened_identity in opened_identities):
            raise _failure()
        opened_identities.add(opened_identity)
        with os.fdopen(file_descriptor, "rb") as stream:
            file_descriptor = None
            count, digest = _read_declared_file(stream, declared)
            after_opened = os.fstat(stream.fileno())
            if (not stat.S_ISREG(after_opened.st_mode) or _is_reparse(after_opened)
                    or _identity(after_opened) != opened_identity):
                raise _failure()
            after = os.stat(parts[-1], dir_fd=parent_descriptor, follow_symlinks=False)
            if (not stat.S_ISREG(after.st_mode) or _is_reparse(after)
                    or _identity(after) != opened_identity):
                raise _failure()
        for descriptor, identity in directory_descriptors:
            current = os.fstat(descriptor)
            if (not stat.S_ISDIR(current.st_mode) or _is_reparse(current)
                    or _identity(current) != identity):
                raise _failure()
        return count, digest
    finally:
        close_error = None
        if file_descriptor is not None:
            try:
                os.close(file_descriptor)
            except OSError as error:
                close_error = error
        for descriptor, _identity_value in reversed(directory_descriptors):
            try:
                os.close(descriptor)
            except OSError as error:
                close_error = error
        if close_error is not None:
            raise close_error


def _windows_directory_handle(path: Path) -> tuple[int, tuple[int, int]]:
    import ctypes
    from ctypes import wintypes

    class ByHandleFileInformation(ctypes.Structure):
        _fields_ = [
            ("file_attributes", wintypes.DWORD),
            ("creation_time", wintypes.FILETIME),
            ("last_access_time", wintypes.FILETIME),
            ("last_write_time", wintypes.FILETIME),
            ("volume_serial_number", wintypes.DWORD),
            ("file_size_high", wintypes.DWORD),
            ("file_size_low", wintypes.DWORD),
            ("number_of_links", wintypes.DWORD),
            ("file_index_high", wintypes.DWORD),
            ("file_index_low", wintypes.DWORD),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = (
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    )
    create_file.restype = wintypes.HANDLE
    get_information = kernel32.GetFileInformationByHandle
    get_information.argtypes = (wintypes.HANDLE, ctypes.POINTER(ByHandleFileInformation))
    get_information.restype = wintypes.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL

    handle = create_file(
        os.fspath(path),
        0x80,  # FILE_READ_ATTRIBUTES
        0x1 | 0x2 | 0x4,  # FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE
        None,
        3,  # OPEN_EXISTING
        0x02000000 | 0x00200000,  # BACKUP_SEMANTICS | OPEN_REPARSE_POINT
        None,
    )
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    information = ByHandleFileInformation()
    if not get_information(handle, ctypes.byref(information)):
        error = ctypes.get_last_error()
        close_handle(handle)
        raise ctypes.WinError(error)
    if not information.file_attributes & stat.FILE_ATTRIBUTE_DIRECTORY:
        close_handle(handle)
        raise OSError("not a directory")
    if information.file_attributes & _REPARSE_POINT:
        close_handle(handle)
        raise OSError("reparse directory")
    identity = (
        information.volume_serial_number,
        (information.file_index_high << 32) | information.file_index_low,
    )
    return handle, identity


def _close_windows_handle(handle: int) -> None:
    import ctypes
    from ctypes import wintypes

    close_handle = ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    if not close_handle(handle):
        raise ctypes.WinError(ctypes.get_last_error())


def _normalized_windows_handle_path(handle: int) -> str:
    import ctypes
    import ntpath
    from ctypes import wintypes

    get_final_path = ctypes.WinDLL(
        "kernel32", use_last_error=True,
    ).GetFinalPathNameByHandleW
    get_final_path.argtypes = (
        wintypes.HANDLE,
        wintypes.LPWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
    )
    get_final_path.restype = wintypes.DWORD

    required = get_final_path(handle, None, 0, 0)
    if required == 0:
        raise ctypes.WinError(ctypes.get_last_error())
    capacity = required + 1
    buffer = ctypes.create_unicode_buffer(capacity)
    written = get_final_path(handle, buffer, capacity, 0)
    if written == 0:
        raise ctypes.WinError(ctypes.get_last_error())
    if written >= capacity:
        raise OSError("truncated final path")

    value = buffer.value.replace("/", "\\")
    lowered = value.lower()
    if lowered.startswith("\\\\?\\unc\\"):
        value = "\\\\" + value[8:]
    elif lowered.startswith("\\\\?\\"):
        value = value[4:]
    elif lowered.startswith("\\??\\unc\\"):
        value = "\\\\" + value[8:]
    elif lowered.startswith("\\??\\"):
        value = value[4:]
    elif lowered.startswith("\\\\.\\"):
        raise OSError("unsupported device path")
    normalized = ntpath.normpath(value)
    if not ntpath.isabs(normalized):
        raise OSError("non-absolute final path")
    return ntpath.normcase(normalized)


def _normalized_windows_declared_path(path: Path) -> str:
    import ntpath

    normalized = ntpath.normpath(os.path.abspath(os.fspath(path))).replace("/", "\\")
    if not ntpath.isabs(normalized):
        raise OSError("non-absolute declared path")
    return ntpath.normcase(normalized)


def _verify_windows_handle_path(
    anchor: _FixtureRootAnchor,
    handle: int,
    parts: tuple[str, ...],
) -> None:
    import ntpath

    if anchor.handle is None or not parts:
        raise _failure()
    root_path = _normalized_windows_handle_path(anchor.handle)
    declared_root = _normalized_windows_declared_path(anchor.path)
    if root_path != declared_root:
        raise _failure()
    expected = ntpath.normcase(ntpath.normpath(ntpath.join(root_path, *parts)))
    prefix = root_path.rstrip("\\") + "\\"
    if not expected.startswith(prefix) or _normalized_windows_handle_path(handle) != expected:
        raise _failure()


def _verify_windows_open_path(
    anchor: _FixtureRootAnchor,
    descriptor: int,
    parts: tuple[str, ...],
) -> None:
    import msvcrt

    handle = msvcrt.get_osfhandle(descriptor)
    if handle == -1:
        raise OSError("invalid file descriptor")
    _verify_windows_handle_path(anchor, handle, parts)


def _verify_windows_fixture_file(
    anchor: _FixtureRootAnchor,
    parts: tuple[str, ...],
    declared: ManifestFile,
    opened_identities: set[tuple[int, int]],
) -> tuple[int, str]:
    handles: list[tuple[int, Path, tuple[int, int], tuple[int, int]]] = []
    file_descriptor: int | None = None
    try:
        root_current = anchor.path.stat(follow_symlinks=False)
        if (anchor.identity is None or not stat.S_ISDIR(root_current.st_mode)
                or _is_reparse(root_current)
                or _identity(root_current) != anchor.identity):
            raise _failure()
        for index in range(1, len(parts)):
            current = anchor.path.joinpath(*parts[:index])
            before = current.stat(follow_symlinks=False)
            if not stat.S_ISDIR(before.st_mode) or _is_reparse(before):
                raise _failure()
            handle, handle_identity = _windows_directory_handle(current)
            try:
                after = current.stat(follow_symlinks=False)
                if (not stat.S_ISDIR(after.st_mode) or _is_reparse(after)
                        or _identity(after) != _identity(before)):
                    raise _failure()
            except Exception:
                _close_windows_handle(handle)
                raise
            _verify_windows_handle_path(anchor, handle, parts[:index])
            handles.append((handle, current, _identity(before), handle_identity))

        candidate = anchor.path.joinpath(*parts)
        before = candidate.stat(follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or _is_reparse(before):
            raise _failure()
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        file_descriptor = os.open(candidate, flags)
        with os.fdopen(file_descriptor, "rb") as stream:
            file_descriptor = None
            opened = os.fstat(stream.fileno())
            opened_identity = _identity(opened)
            if (not stat.S_ISREG(opened.st_mode) or _is_reparse(opened)
                    or opened_identity != _identity(before)
                    or opened_identity in opened_identities):
                raise _failure()
            opened_identities.add(opened_identity)
            _verify_windows_open_path(anchor, stream.fileno(), parts)
            count, digest = _read_declared_file(stream, declared)
            _verify_windows_open_path(anchor, stream.fileno(), parts)
            after = candidate.stat(follow_symlinks=False)
            if (not stat.S_ISREG(after.st_mode) or _is_reparse(after)
                    or _identity(after) != opened_identity):
                raise _failure()
        for index, (_handle, path, path_identity, _handle_identity) in enumerate(handles, 1):
            _verify_windows_handle_path(anchor, _handle, parts[:index])
            current = path.stat(follow_symlinks=False)
            if (not stat.S_ISDIR(current.st_mode) or _is_reparse(current)
                    or _identity(current) != path_identity):
                raise _failure()
        return count, digest
    finally:
        close_error = None
        if file_descriptor is not None:
            try:
                os.close(file_descriptor)
            except OSError as error:
                close_error = error
        for handle, _path, _path_identity, _handle_identity in reversed(handles):
            try:
                _close_windows_handle(handle)
            except OSError as error:
                close_error = error
        if close_error is not None:
            raise close_error


def _verify_fixture_files(
    anchor: _FixtureRootAnchor,
    models: tuple[ManifestModel, ...],
) -> None:
    opened_identities: set[tuple[int, int]] = set()
    for model in models:
        if not model.synthetic:
            # Real development models have no local fixture files: the model
            # cache downloads and verifies them against the pinned size and
            # sha256, under the release per-file bounds, at acquisition.
            continue
        for declared in model.files:
            if declared.bytes > MAX_FIXTURE_FILE_BYTES:
                raise _failure()
            parts = PurePosixPath(declared.relative_path).parts
            try:
                if os.name == "posix":
                    if anchor.descriptor is None:
                        raise _failure()
                    count, digest = _verify_posix_fixture_file(
                        anchor.descriptor, parts, declared, opened_identities,
                    )
                elif os.name == "nt":
                    count, digest = _verify_windows_fixture_file(
                        anchor, parts, declared, opened_identities,
                    )
                else:
                    raise _failure()
            except ManifestError:
                raise
            except (OSError, TypeError, ValueError):
                raise _failure() from None
            if count != declared.bytes or not hmac.compare_digest(digest, declared.sha256):
                raise _failure()


def _parse_manifest(
    raw: bytes,
    *,
    development: bool,
    allowed_assets: tuple[tuple[str, str], ...],
    expected_sha256: str | None,
    fixture_anchor: _FixtureRootAnchor | None,
) -> RuntimeManifest:
    manifest_hash = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and (
            type(expected_sha256) is not str
            or _SHA256.fullmatch(expected_sha256) is None
            or not hmac.compare_digest(manifest_hash, expected_sha256)):
        raise _failure()
    data = _load_json(raw)
    root = _object(data, frozenset({
        "schema_version", "release_version", "source", "platforms",
        "capabilities", "models", "languages",
    }))
    if type(root["schema_version"]) is not int or root["schema_version"] != 1:
        raise _failure()
    source_data = _object(root["source"], frozenset({"kind", "label", "provenance"}))
    expected_kind = "development_fixture" if development else "packaged_release"
    if source_data["kind"] != expected_kind:
        raise _failure(ReasonCode.ASSET_SOURCE_REJECTED)
    if development:
        if source_data["label"] != "fixture":
            raise _failure(ReasonCode.ASSET_SOURCE_REJECTED)
    else:
        _string(source_data["label"], release=True)
    _string(source_data["provenance"], release=not development)
    platforms = _tuple(root["platforms"], lambda entry: _string(entry, identifier=True))
    _unique(platforms, lambda entry: entry)
    capabilities = _tuple(
        root["capabilities"],
        lambda entry: _capability(entry, fixture=development, platforms=platforms),
    )
    models = _tuple(
        root["models"],
        lambda entry: _model(entry, fixture=development, allowed_assets=allowed_assets),
        nonempty=False,
    )
    languages = _tuple(root["languages"], lambda entry: _string(entry, identifier=True))
    _unique(capabilities, lambda entry: entry.capability)
    _unique(models, lambda entry: entry.artifact.model_id)
    _unique(languages, lambda entry: entry)
    if any(capability.capability in (Capability.ASR_CPU, Capability.ASR_CUDA)
           for capability in capabilities) and not models:
        raise _failure()
    # No static worker-memory fit check here: the operator chooses the model
    # and selection checks fit against live worker evidence (SPEC_GAP-16).
    if development:
        if fixture_anchor is None:
            raise _failure()
        _verify_fixture_files(fixture_anchor, models)
    return RuntimeManifest(
        _string(root["release_version"], release=not development),
        development,
        platforms,
        capabilities,
        models,
        languages,
        manifest_hash,
    )


def load_manifest(
    source: str | Path,
    *,
    development: bool = False,
    allowed_release_sources: tuple[str | Path, ...] = (),
    allowed_asset_sources: tuple[str, ...] = (),
    expected_sha256: str | None = None,
) -> RuntimeManifest:
    """Load one local manifest after source-policy and bounded schema checks."""

    if (type(development) is not bool or type(allowed_release_sources) is not tuple
            or len(allowed_release_sources) > MAX_COLLECTION_ITEMS):
        raise _failure(ReasonCode.ASSET_SOURCE_REJECTED)
    # Release manifests admit only allowlisted asset origins or URLs.  A
    # development manifest honours the allowlist when one is given and
    # otherwise admits any source that passes the strict HTTPS/origin rules.
    allowed_assets = _asset_allowlist(allowed_asset_sources)
    if not development and (
            type(expected_sha256) is not str or _SHA256.fullmatch(expected_sha256) is None):
        raise _failure()
    try:
        source_path = Path(os.path.abspath(os.fspath(source)))
    except (TypeError, ValueError, OSError):
        raise _failure(ReasonCode.ASSET_SOURCE_REJECTED) from None
    if not development:
        try:
            allowlist = tuple(
                Path(os.path.abspath(os.fspath(path)))
                for path in allowed_release_sources
            )
        except (TypeError, ValueError, OSError):
            raise _failure(ReasonCode.ASSET_SOURCE_REJECTED) from None
        raw = _read_release_source(source_path, allowlist)
        return _parse_manifest(
            raw,
            development=False,
            allowed_assets=allowed_assets,
            expected_sha256=expected_sha256,
            fixture_anchor=None,
        )

    anchor = _open_fixture_root(source_path)
    active_error = False
    close_reason = ReasonCode.ASSET_SOURCE_REJECTED
    try:
        raw = _read_fixture_source(source_path, anchor)
        close_reason = ReasonCode.ASSET_VERIFICATION_FAILED
        return _parse_manifest(
            raw,
            development=True,
            allowed_assets=allowed_assets,
            expected_sha256=expected_sha256,
            fixture_anchor=anchor,
        )
    except BaseException:
        active_error = True
        raise
    finally:
        try:
            anchor.close()
        except OSError:
            if not active_error:
                raise _failure(close_reason) from None


def resolve_capability(
    manifest: RuntimeManifest,
    capability: Capability | str,
    *,
    selected_gpu_uuid: str | None = None,
) -> ResolvedCapability:
    """Resolve exactly one admitted capability into immutable consumer values."""

    if type(manifest) is not RuntimeManifest:
        raise _failure()
    try:
        requested = Capability.parse(capability)
    except DomainValidationError:
        raise _failure() from None
    matches = tuple(item for item in manifest.capabilities if item.capability is requested)
    if len(matches) != 1:
        raise _failure()
    selected = matches[0]
    if (requested is Capability.ASR_CUDA and (
            type(selected_gpu_uuid) is not str or _GPU_UUID.fullmatch(selected_gpu_uuid) is None)):
        raise _failure()
    if requested is not Capability.ASR_CUDA and selected_gpu_uuid is not None:
        raise _failure()
    image = selected.digest if manifest.development_fixture else f"{selected.image_reference}@{selected.digest}"
    try:
        policy = WorkerPolicy(
            requested,
            image,
            selected.entrypoint,
            selected.compressed_bytes,
            selected.unpacked_peak_bytes,
            allow_pull=not manifest.development_fixture,
            memory_bytes=selected.worker_memory_bytes,
            selected_gpu_uuid=selected_gpu_uuid,
            development_image=manifest.development_fixture,
        )
        artifacts = tuple(model.artifact for model in manifest.models)
        envelopes: list[MeasuredEnvelope] = []
        if requested in (Capability.ASR_CPU, Capability.ASR_CUDA):
            backend = Backend.CUDA if requested is Capability.ASR_CUDA else Backend.CPU
            runtime = RuntimeFingerprint(
                selected.digest,
                selected.runtime_version,
                selected.driver_version,
                selected.hardware_fingerprint,
            )
            # Every verified model is an artifact for every ASR backend, so one
            # manifest serves CPU and CUDA.  An envelope is built only from a
            # complete, release-tested measurement taken on this backend (CUDA
            # measurements carry VRAM, CPU measurements carry none); any other
            # model is skipped here and fitted from the hardware table instead.
            for model in manifest.models:
                if (not model.measured or model.release_tested is not True
                        or (backend is Backend.CUDA) != (model.worker_peak_vram_bytes != 0)):
                    continue
                for compute_type in model.compute_types:
                    envelopes.append(MeasuredEnvelope(
                        model.artifact,
                        runtime,
                        backend,
                        compute_type,
                        model.worker_peak_ram_bytes,
                        model.worker_peak_vram_bytes,
                        model.validated_max_duration_seconds,
                        model.decode_bytes_per_second,
                        model.decode_timeout_seconds,
                        model.evidence_sha256,
                        model.release_tested,
                        model.synthetic,
                    ))
        return ResolvedCapability(
            requested,
            policy,
            artifacts,
            tuple(envelopes),
            tuple((model.artifact.model_id, model.files) for model in manifest.models),
            manifest.languages,
            selected.worker_overhead_bytes,
            selected.dependency_lock_sha256,
            selected.licenses,
            selected.provenance,
            selected.platforms,
        )
    except ManifestError:
        raise
    except (DomainValidationError, ValueError, TypeError, OverflowError):
        raise _failure() from None


def engine_arch_platform(arch: object) -> str | None:
    """The manifest platform id for one Docker engine architecture, or ``None``.

    Accepts ``docker info`` values (``x86_64``, ``aarch64``) and the OCI
    names (``amd64``, ``arm64``), case-insensitively, with surrounding
    whitespace ignored.  Anything else (an empty value, ``Unknown``, a
    variant such as ``arm/v7``) maps to ``None``.
    """
    if type(arch) is not str:
        return None
    return ENGINE_ARCH_PLATFORMS.get(arch.strip().lower())


def supports_engine_arch(
    resolved: ResolvedCapability | ManifestCapability,
    arch: object,
) -> bool:
    """True when the capability's image runs on this Docker engine architecture.

    ``arch`` is the engine architecture (``docker info --format
    {{.Architecture}}``: ``x86_64`` or ``aarch64``, or the normalised
    ``amd64``/``arm64``).  It is mapped through ``engine_arch_platform`` and
    looked up in ``platforms``.  An unmapped architecture is never supported,
    so a caller fails closed rather than pulling an image the engine cannot
    run.  A development fixture names a fixture platform, not an
    architecture: callers that run fixtures decide for themselves (for
    example via ``worker_policy.development_image``) rather than through
    this check.
    """
    if type(resolved) not in (ResolvedCapability, ManifestCapability):
        raise _failure()
    platform = engine_arch_platform(arch)
    return platform is not None and platform in resolved.platforms
