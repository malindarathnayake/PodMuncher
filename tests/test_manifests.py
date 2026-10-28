"""Synthetic manifest fixtures only; no production asset claim or network access."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import podmuncher.manifests as manifests_module
from podmuncher.domains import Backend, Capability, ModelTier, ReasonCode
from podmuncher.manifests import (
    ENGINE_ARCH_PLATFORMS,
    ENVELOPE_KEYS,
    MAX_FIXTURE_FILE_BYTES,
    MIN_WORKER_OVERHEAD_BYTES,
    ManifestError,
    engine_arch_platform,
    load_manifest,
    resolve_capability,
    supports_engine_arch,
)
from podmuncher.model_cache import CacheError, ModelCache
from podmuncher.model_policy import (
    DECODE_BYTES_PER_SECOND,
    MeasuredEnvelope,
    ModelArtifact,
)
from podmuncher.runtime import WorkerPolicy

GPU_UUID = "GPU-12345678-1234-1234-1234-123456789abc"
REAL_MODEL_BYTES = b"real pinned development model payload"
REAL_MODEL_SOURCE = "https://packages.podmuncher.org/development/model.bin"


def make_manifest(*, capability="asr_cpu", tier="tiny"):
    """Return a complete, explicitly synthetic development fixture."""
    model_bytes = b"synthetic model fixture"
    return {
        "schema_version": 1,
        "release_version": "fixture",
        "source": {
            "kind": "development_fixture",
            "label": "fixture",
            "provenance": "fixture:test_manifests",
        },
        "platforms": ["fixture-platform"],
        "capabilities": [{
            "id": capability,
            "platform": "fixture-platform",
            "image_reference": "fixture-local",
            "digest": "sha256:" + "1" * 64,
            "entrypoint": ["python", "-m", "fixture_worker"],
            "compressed_bytes": 100,
            "unpacked_peak_bytes": 200,
            "worker_overhead_bytes": MIN_WORKER_OVERHEAD_BYTES,
            "worker_memory_bytes": 2 * 1024**3,
            "dependency_lock_sha256": "2" * 64,
            "licenses": ["fixture-license"],
            "provenance": "fixture:local-image",
            "runtime_version": "fixture-runtime",
            "driver_version": "fixture-driver" if capability == "asr_cuda" else None,
            "hardware_fingerprint": "fixture-hardware",
        }],
        "models": [{
            "id": "fixture-model",
            "tier": tier,
            "revision": "fixture-revision",
            "digest": "3" * 64,
            "source_urls": ["fixture:model.bin"],
            "files": [{
                "relative_path": "model.bin",
                "bytes": len(model_bytes),
                "sha256": hashlib.sha256(model_bytes).hexdigest(),
            }],
            "license": "fixture-license",
            "provenance": "fixture:generated-test-data",
            "compute_types": ["int8_float16" if capability == "asr_cuda" else "int8"],
            "worker_peak_ram_bytes": 512 * 1024**2,
            "worker_peak_vram_bytes": 1024**3 if capability == "asr_cuda" else 0,
            "validated_max_duration_seconds": 3600,
            "decode_bytes_per_second": DECODE_BYTES_PER_SECOND,
            "decode_timeout_seconds": 120,
            "codec_matrix": ["fixture_codec"],
            "evidence_sha256": "4" * 64,
            "release_tested": True,
            "synthetic": True,
        }],
        "languages": ["en"],
    }


def make_release_manifest():
    manifest = make_manifest()
    manifest["release_version"] = "1.0.0"
    manifest["source"] = {
        "kind": "packaged_release",
        "label": "release-1.0.0",
        "provenance": "publisher-build-2026-10",
    }
    capability = manifest["capabilities"][0]
    capability.update({
        "image_reference": "ghcr.io/podmuncher/asr",
        "licenses": ["Apache-2.0"],
        "provenance": "publisher-container-build-2026-10",
        "runtime_version": "runtime-1.0.0",
        "hardware_fingerprint": "validated-amd64-host",
    })
    model = manifest["models"][0]
    model.update({
        "revision": "model-revision-1",
        "source_urls": ["https://packages.podmuncher.org/releases/model.bin"],
        "license": "MIT",
        "provenance": "publisher-model-build-2026-10",
        "synthetic": False,
    })
    return manifest


def strip_envelope(model, *, null=False):
    """Remove the measured envelope from one model dict (keys absent, or all null)."""
    for key in ENVELOPE_KEYS:
        if null:
            model[key] = None
        else:
            del model[key]
    return model


def make_dual_manifest():
    """One development fixture with both ASR capabilities and one unmeasured model."""
    manifest = make_manifest()
    cuda = dict(make_manifest(capability="asr_cuda")["capabilities"][0])
    manifest["capabilities"].append(cuda)
    strip_envelope(manifest["models"][0])
    return manifest


def make_development_real_manifest(*, capability="asr_cpu", release_tested=True):
    """A development fixture whose one model is real: pinned HTTPS source, bytes, sha256."""
    manifest = make_manifest(capability=capability)
    manifest["models"][0].update({
        "id": "real-development-model",
        "source_urls": [REAL_MODEL_SOURCE],
        "files": [{
            "relative_path": "model.bin",
            "bytes": len(REAL_MODEL_BYTES),
            "sha256": hashlib.sha256(REAL_MODEL_BYTES).hexdigest(),
        }],
        "release_tested": release_tested,
        "synthetic": False,
    })
    return manifest


class _StubSocket:
    """Socket shape ModelCache expects on a response; fileno < 0 skips timeouts."""

    @staticmethod
    def fileno():
        return -1

    @staticmethod
    def settimeout(_value):
        return None


class _StubResponse:
    """In-memory HTTPS response: Content-Length header plus read1/close."""

    def __init__(self, payload):
        self._payload = payload
        self._offset = 0
        self.headers = {"Content-Length": str(len(payload))}
        self._sock = _StubSocket()

    def read1(self, size):
        chunk = self._payload[self._offset:self._offset + size]
        self._offset += len(chunk)
        return chunk

    def close(self):
        return None


class _Audit:
    def __init__(self):
        self.events = []

    def record(self, *, action, reason, asset_id, failure_code):
        self.events.append((action, reason, asset_id, failure_code))


def stubbed_cache(tmp_path, monkeypatch, payload):
    """ModelCache on a temp root whose transport is replaced; any real request fails the test."""
    opened = []

    def fail_request(_self, *_args, **_kwargs):
        raise AssertionError("ModelCache attempted a network request")

    def stub_open(_self, url, _deadline):
        opened.append(url)
        return _StubResponse(payload)

    monkeypatch.setattr(ModelCache, "_request", fail_request)
    monkeypatch.setattr(ModelCache, "_resolve_addresses", fail_request)
    monkeypatch.setattr(ModelCache, "_open", stub_open)
    cache = ModelCache(
        tmp_path / "cache",
        audit=_Audit(),
        admission=lambda **_values: True,
        max_entries=4,
        chunk_size=8,
        timeout_seconds=30,
    )
    return cache, opened


def write_manifest(tmp_path, value=None):
    path = tmp_path / "manifest.json"
    raw = json.dumps(make_manifest() if value is None else value, separators=(",", ":")).encode()
    path.write_bytes(raw)
    (tmp_path / "model.bin").write_bytes(b"synthetic model fixture")
    return path, raw


def reason(expected, callback):
    with pytest.raises(ManifestError) as error:
        callback()
    assert error.value.reason_code is expected
    assert str(error.value) == "Invalid input."


def test_load_and_resolve_immutable_typed_values(tmp_path):
    path, raw = write_manifest(tmp_path)
    manifest = load_manifest(
        path,
        development=True,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
    )
    resolved = resolve_capability(manifest, Capability.ASR_CPU)
    assert manifest.sha256 == hashlib.sha256(raw).hexdigest()
    assert isinstance(resolved.worker_policy, WorkerPolicy)
    assert resolved.worker_policy.image == "sha256:" + "1" * 64
    assert resolved.worker_policy.development_image is True
    assert resolved.worker_policy.allow_pull is False
    assert isinstance(resolved.artifacts[0], ModelArtifact)
    assert isinstance(resolved.envelopes[0], MeasuredEnvelope)
    assert resolved.envelopes[0].backend is Backend.CPU
    assert resolved.envelopes[0].synthetic is True
    assert resolved.languages == ("en",)
    with pytest.raises(FrozenInstanceError):
        resolved.languages = ()


def test_cuda_resolution_retains_measured_vram_and_driver(tmp_path):
    path, _ = write_manifest(tmp_path, make_manifest(capability="asr_cuda"))
    resolved = resolve_capability(
        load_manifest(path, development=True),
        "asr_cuda",
        selected_gpu_uuid=GPU_UUID,
    )
    assert resolved.envelopes[0].backend is Backend.CUDA
    assert resolved.envelopes[0].worker_peak_vram_bytes == 1024**3
    assert resolved.envelopes[0].runtime.driver_version == "fixture-driver"
    assert resolved.worker_policy.selected_gpu_uuid == GPU_UUID


@pytest.mark.parametrize("raw", [b"{", b"[]", b'{"x":NaN}', b'{"x":1,"x":2}', b"\xff"])
def test_malformed_json_is_verification_failure_without_input_leak(tmp_path, raw):
    path = tmp_path / "private-token.json"
    path.write_bytes(raw)
    reason(ReasonCode.ASSET_VERIFICATION_FAILED,
           lambda: load_manifest(path, development=True))


def test_missing_unknown_and_duplicate_ids_fail_verification(tmp_path):
    missing = make_manifest()
    del missing["models"][0]["digest"]
    path, _ = write_manifest(tmp_path, missing)
    reason(ReasonCode.ASSET_VERIFICATION_FAILED,
           lambda: load_manifest(path, development=True))

    unknown = make_manifest()
    unknown["unexpected"] = True
    path, _ = write_manifest(tmp_path, unknown)
    reason(ReasonCode.ASSET_VERIFICATION_FAILED,
           lambda: load_manifest(path, development=True))

    duplicate = make_manifest()
    duplicate["models"].append(dict(duplicate["models"][0]))
    path, _ = write_manifest(tmp_path, duplicate)
    reason(ReasonCode.ASSET_VERIFICATION_FAILED,
           lambda: load_manifest(path, development=True))


def test_hash_format_and_expected_manifest_hash_mismatch_fail(tmp_path):
    value = make_manifest()
    value["models"][0]["files"][0]["sha256"] = "A" * 64
    path, _ = write_manifest(tmp_path, value)
    reason(ReasonCode.ASSET_VERIFICATION_FAILED,
           lambda: load_manifest(path, development=True))

    value = make_manifest()
    value["models"][0]["files"][0]["sha256"] = "5" * 64
    path, _ = write_manifest(tmp_path, value)
    reason(ReasonCode.ASSET_VERIFICATION_FAILED,
           lambda: load_manifest(path, development=True))

    path, _ = write_manifest(tmp_path)
    reason(ReasonCode.ASSET_VERIFICATION_FAILED,
           lambda: load_manifest(path, development=True, expected_sha256="f" * 64))


@pytest.mark.parametrize("relative_path", [
    "../model.bin", "/model.bin", r"..\model.bin", r"C:\model.bin", ".", "a/../../model.bin",
    "./model.bin", "folder//model.bin",
])
def test_model_file_paths_reject_traversal_and_absolute_forms(tmp_path, relative_path):
    value = make_manifest()
    value["models"][0]["files"][0]["relative_path"] = relative_path
    path, _ = write_manifest(tmp_path, value)
    reason(ReasonCode.ASSET_VERIFICATION_FAILED,
           lambda: load_manifest(path, development=True))


def test_fixture_file_rejects_linked_parent(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "model.bin").write_bytes(b"synthetic model fixture")
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")
    value = make_manifest()
    value["models"][0]["files"][0]["relative_path"] = "linked/model.bin"
    path, _ = write_manifest(tmp_path, value)
    reason(ReasonCode.ASSET_VERIFICATION_FAILED,
           lambda: load_manifest(path, development=True))


def test_fixture_file_rejects_hardlink_alias_across_models(tmp_path):
    value = make_manifest()
    duplicate = json.loads(json.dumps(value["models"][0]))
    duplicate["id"] = "fixture-model-copy"
    duplicate["files"][0]["relative_path"] = "alias.bin"
    value["models"].append(duplicate)
    path, _ = write_manifest(tmp_path, value)
    try:
        os.link(tmp_path / "model.bin", tmp_path / "alias.bin")
    except OSError:
        pytest.skip("hard links are unavailable")
    reason(ReasonCode.ASSET_VERIFICATION_FAILED,
           lambda: load_manifest(path, development=True))


def test_fixture_file_rejects_check_open_swap(tmp_path, monkeypatch):
    path, _ = write_manifest(tmp_path)
    original_open = manifests_module.os.open
    swapped = False

    def swapping_open(candidate, flags, *, dir_fd=None):
        nonlocal swapped
        is_candidate = (
            Path(candidate) == tmp_path / "model.bin"
            or Path(candidate) == Path("model.bin") and dir_fd is not None
        )
        if is_candidate and not swapped:
            replacement = tmp_path / "replacement.bin"
            replacement.write_bytes(b"synthetic model fixture")
            os.replace(replacement, tmp_path / "model.bin")
            swapped = True
        if dir_fd is None:
            return original_open(candidate, flags)
        return original_open(candidate, flags, dir_fd=dir_fd)

    monkeypatch.setattr(manifests_module.os, "open", swapping_open)
    reason(ReasonCode.ASSET_VERIFICATION_FAILED,
           lambda: load_manifest(path, development=True))
    assert swapped


def test_fixture_file_parent_swap_cannot_admit_outside_matching_bytes(tmp_path, monkeypatch):
    value = make_manifest()
    value["models"][0]["files"][0]["relative_path"] = "inside/model.bin"
    path, _ = write_manifest(tmp_path, value)
    inside = tmp_path / "inside"
    outside = tmp_path / "outside"
    parked = tmp_path / "parked"
    inside.mkdir()
    outside.mkdir()
    (inside / "model.bin").write_bytes(b"x" * len(b"synthetic model fixture"))
    (outside / "model.bin").write_bytes(b"synthetic model fixture")

    original_open = manifests_module.os.open
    race_attempted = False

    def racing_open(candidate, flags, *, dir_fd=None):
        nonlocal race_attempted
        is_candidate = (
            Path(candidate) == inside / "model.bin"
            or Path(candidate) == Path("model.bin") and dir_fd is not None
        )
        if not is_candidate or race_attempted:
            if dir_fd is None:
                return original_open(candidate, flags)
            return original_open(candidate, flags, dir_fd=dir_fd)

        race_attempted = True
        try:
            os.replace(inside, parked)
            try:
                os.replace(outside, inside)
            except Exception:
                os.replace(parked, inside)
                raise
        except OSError:
            return original_open(candidate, flags) if dir_fd is None else original_open(
                candidate, flags, dir_fd=dir_fd,
            )

        try:
            descriptor = (
                original_open(candidate, flags)
                if dir_fd is None
                else original_open(candidate, flags, dir_fd=dir_fd)
            )
        finally:
            os.replace(inside, outside)
            os.replace(parked, inside)
        return descriptor

    monkeypatch.setattr(manifests_module.os, "open", racing_open)
    reason(
        ReasonCode.ASSET_VERIFICATION_FAILED,
        lambda: load_manifest(path, development=True),
    )
    assert race_attempted


def test_fixture_root_replacement_after_manifest_read_cannot_admit_outside_bytes(
    tmp_path,
    monkeypatch,
):
    workspace = tmp_path / "workspace"
    fixture_root = workspace / "fixture"
    outside_root = workspace / "outside"
    parked_root = workspace / "parked"
    fixture_root.mkdir(parents=True)
    outside_root.mkdir()
    path, raw = write_manifest(fixture_root)
    (fixture_root / "model.bin").write_bytes(b"x" * len(b"synthetic model fixture"))
    (outside_root / "manifest.json").write_bytes(raw)
    (outside_root / "model.bin").write_bytes(b"synthetic model fixture")

    original_load_json = manifests_module._load_json
    replacement_attempted = False
    replacement_completed = False
    replacement_blocked = False

    def replace_root_after_read(manifest_raw):
        nonlocal replacement_attempted, replacement_completed, replacement_blocked
        parsed = original_load_json(manifest_raw)
        replacement_attempted = True
        try:
            os.replace(fixture_root, parked_root)
            try:
                os.replace(outside_root, fixture_root)
            except Exception:
                os.replace(parked_root, fixture_root)
                raise
        except OSError:
            replacement_blocked = True
        else:
            replacement_completed = True
        return parsed

    monkeypatch.setattr(manifests_module, "_load_json", replace_root_after_read)
    try:
        reason(
            ReasonCode.ASSET_VERIFICATION_FAILED,
            lambda: load_manifest(path, development=True),
        )
    finally:
        if replacement_completed:
            os.replace(fixture_root, outside_root)
            os.replace(parked_root, fixture_root)

    assert replacement_attempted
    assert replacement_blocked != replacement_completed
    if os.name == "posix":
        assert replacement_completed


@pytest.mark.skipif(os.name != "nt", reason="Windows handle-path regression")
def test_windows_fixture_root_replace_restore_before_file_open_is_rejected(
    tmp_path,
    monkeypatch,
):
    workspace = tmp_path / "workspace"
    fixture_root = workspace / "fixture"
    outside_root = workspace / "outside"
    parked_root = workspace / "parked"
    fixture_root.mkdir(parents=True)
    outside_root.mkdir()
    path, _ = write_manifest(fixture_root)
    (fixture_root / "model.bin").write_bytes(b"x" * len(b"synthetic model fixture"))
    (outside_root / "model.bin").write_bytes(b"synthetic model fixture")

    original_open = manifests_module.os.open
    replacement_attempted = False

    def replace_restore_open(candidate, flags, *, dir_fd=None):
        nonlocal replacement_attempted
        if Path(candidate) != fixture_root / "model.bin" or replacement_attempted:
            if dir_fd is None:
                return original_open(candidate, flags)
            return original_open(candidate, flags, dir_fd=dir_fd)

        replacement_attempted = True
        os.replace(fixture_root, parked_root)
        descriptor = None
        try:
            os.replace(outside_root, fixture_root)
            descriptor = original_open(candidate, flags)
            os.replace(fixture_root, outside_root)
            os.replace(parked_root, fixture_root)
            return descriptor
        except BaseException:
            if descriptor is not None:
                os.close(descriptor)
            if fixture_root.exists() and not outside_root.exists():
                os.replace(fixture_root, outside_root)
            if parked_root.exists() and not fixture_root.exists():
                os.replace(parked_root, fixture_root)
            raise

    monkeypatch.setattr(manifests_module.os, "open", replace_restore_open)
    reason(
        ReasonCode.ASSET_VERIFICATION_FAILED,
        lambda: load_manifest(path, development=True),
    )
    assert replacement_attempted


@pytest.mark.skipif(os.name != "nt", reason="Windows 8.3 short-name regression")
def test_fixture_root_accepts_short_name_path(tmp_path):
    import ctypes
    from ctypes import wintypes

    long_root = tmp_path / "long directory name for short form"
    long_root.mkdir()
    get_short = ctypes.WinDLL("kernel32", use_last_error=True).GetShortPathNameW
    get_short.argtypes = (wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD)
    get_short.restype = wintypes.DWORD
    buffer = ctypes.create_unicode_buffer(1024)
    written = get_short(str(long_root), buffer, 1024)
    if written == 0 or written >= 1024 or buffer.value.lower() == str(long_root).lower():
        pytest.skip("volume does not provide 8.3 short names")
    short_root = Path(buffer.value)

    path, _ = write_manifest(short_root)
    manifest = load_manifest(path, development=True)
    assert resolve_capability(manifest, Capability.ASR_CPU) is not None


@pytest.mark.skipif(os.name != "nt", reason="Windows junction regression")
def test_windows_junctioned_fixture_root_is_rejected(tmp_path):
    import _winapi

    real_root = tmp_path / "real"
    real_root.mkdir()
    write_manifest(real_root)
    junction = tmp_path / "junction"
    _winapi.CreateJunction(str(real_root), str(junction))

    reason(
        ReasonCode.ASSET_SOURCE_REJECTED,
        lambda: load_manifest(junction / "manifest.json", development=True),
    )


def test_capability_and_model_tier_reject_registry_negative_examples(tmp_path):
    registry = json.loads(
        (Path(__file__).parents[1] / "Docs" / "contract-domains.json").read_text(encoding="utf-8")
    )
    assert registry["capability"] == [member.value for member in Capability]
    assert registry["model_tier"] == [member.value for member in ModelTier]
    for invalid in registry["negative_examples"]:
        capability = make_manifest(capability=invalid)
        path, _ = write_manifest(tmp_path, capability)
        reason(ReasonCode.ASSET_VERIFICATION_FAILED,
               lambda path=path: load_manifest(path, development=True))
        model = make_manifest(tier=invalid)
        path, _ = write_manifest(tmp_path, model)
        reason(ReasonCode.ASSET_VERIFICATION_FAILED,
               lambda path=path: load_manifest(path, development=True))
        # An unmeasured model keeps the same identity rules.
        unmeasured = make_manifest(tier=invalid)
        strip_envelope(unmeasured["models"][0])
        path, _ = write_manifest(tmp_path, unmeasured)
        reason(ReasonCode.ASSET_VERIFICATION_FAILED,
               lambda path=path: load_manifest(path, development=True))
        unmeasured_capability = make_manifest(capability=invalid)
        strip_envelope(unmeasured_capability["models"][0])
        path, _ = write_manifest(tmp_path, unmeasured_capability)
        reason(ReasonCode.ASSET_VERIFICATION_FAILED,
               lambda path=path: load_manifest(path, development=True))


def test_source_policy_rejections_have_distinct_reason(tmp_path):
    path, raw = write_manifest(tmp_path)
    reason(
        ReasonCode.ASSET_SOURCE_REJECTED,
        lambda: load_manifest(path, expected_sha256=hashlib.sha256(raw).hexdigest()),
    )

    wrong_label = make_manifest()
    wrong_label["source"]["label"] = "not-a-fixture"
    path, _ = write_manifest(tmp_path, wrong_label)
    reason(ReasonCode.ASSET_SOURCE_REJECTED,
           lambda: load_manifest(path, development=True))

    wrong_model_source = make_manifest()
    wrong_model_source["models"][0]["source_urls"] = ["https://example.invalid/model"]
    path, _ = write_manifest(tmp_path, wrong_model_source)
    reason(ReasonCode.ASSET_SOURCE_REJECTED,
           lambda: load_manifest(path, development=True))


@pytest.mark.parametrize("asset_allowlist", [
    ("https://packages.podmuncher.org",),
    ("https://packages.podmuncher.org/releases/model.bin",),
])
def test_allowlisted_packaged_release_loads(tmp_path, asset_allowlist):
    path, raw = write_manifest(tmp_path, make_release_manifest())
    manifest = load_manifest(
        path,
        allowed_release_sources=(path,),
        allowed_asset_sources=asset_allowlist,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
    )
    assert manifest.development_fixture is False
    assert manifest.models[0].source_urls == (
        "https://packages.podmuncher.org/releases/model.bin",
    )


@pytest.mark.parametrize("expected_sha256", [None, "", "A" * 64])
def test_packaged_release_requires_valid_expected_sha256(tmp_path, expected_sha256):
    path, _ = write_manifest(tmp_path, make_release_manifest())
    reason(
        ReasonCode.ASSET_VERIFICATION_FAILED,
        lambda: load_manifest(
            path,
            allowed_release_sources=(path,),
            expected_sha256=expected_sha256,
        ),
    )


def test_packaged_release_rejects_check_open_replacement(tmp_path, monkeypatch):
    path, raw = write_manifest(tmp_path, make_release_manifest())
    original_open = manifests_module.os.open
    replacement_attempted = threading.Event()

    def swapping_open(candidate, flags, *, dir_fd=None):
        if Path(candidate) == path and not replacement_attempted.is_set():
            replacement = tmp_path / "replacement-manifest.json"
            replacement.write_bytes(raw)
            os.replace(replacement, candidate)
            replacement_attempted.set()
        if dir_fd is None:
            return original_open(candidate, flags)
        return original_open(candidate, flags, dir_fd=dir_fd)

    monkeypatch.setattr(manifests_module.os, "open", swapping_open)
    reason(
        ReasonCode.ASSET_SOURCE_REJECTED,
        lambda: load_manifest(
            path,
            allowed_release_sources=(path,),
            expected_sha256=hashlib.sha256(raw).hexdigest(),
        ),
    )
    assert replacement_attempted.is_set()


@pytest.mark.parametrize("source_url,allowlist", [
    ("https:///model.bin", ("https://packages.podmuncher.org",)),
    ("https://user@packages.podmuncher.org/model.bin", ("https://packages.podmuncher.org",)),
    ("https://127.0.0.1/model.bin", ("https://127.0.0.1",)),
    ("https://10.0.0.1/model.bin", ("https://10.0.0.1",)),
    ("https://packages.podmuncher.org/model.bin", ("https://cdn.podmuncher.org",)),
])
def test_packaged_asset_sources_reject_malformed_private_and_unlisted(
    tmp_path,
    source_url,
    allowlist,
):
    value = make_release_manifest()
    value["models"][0]["source_urls"] = [source_url]
    path, raw = write_manifest(tmp_path, value)
    reason(
        ReasonCode.ASSET_SOURCE_REJECTED,
        lambda: load_manifest(
            path,
            allowed_release_sources=(path,),
            allowed_asset_sources=allowlist,
            expected_sha256=hashlib.sha256(raw).hexdigest(),
        ),
    )


@pytest.mark.parametrize(("field", "value"), [
    ("release_version", ""),
    ("release_version", "unresolved"),
])
def test_allowlisted_packaged_release_rejects_unresolved_fields(tmp_path, field, value):
    manifest = make_manifest()
    manifest["source"] = {
        "kind": "packaged_release",
        "label": "unresolved",
        "provenance": "unresolved",
    }
    manifest[field] = value
    path, raw = write_manifest(tmp_path, manifest)
    reason(ReasonCode.ASSET_VERIFICATION_FAILED,
           lambda: load_manifest(
               path,
               allowed_release_sources=(path,),
               expected_sha256=hashlib.sha256(raw).hexdigest(),
           ))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("compressed_bytes", 0),
        ("unpacked_peak_bytes", 99),
        ("worker_overhead_bytes", MIN_WORKER_OVERHEAD_BYTES - 1),
        ("worker_memory_bytes", 64 * 1024**2 - 1),
        ("digest", ""),
        ("dependency_lock_sha256", "__invalid__"),
        ("licenses", []),
        ("provenance", ""),
    ],
)
def test_incomplete_or_inconsistent_capability_evidence_fails(tmp_path, field, value):
    manifest = make_manifest()
    manifest["capabilities"][0][field] = value
    path, _ = write_manifest(tmp_path, manifest)
    reason(ReasonCode.ASSET_VERIFICATION_FAILED,
           lambda: load_manifest(path, development=True))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("digest", ""),
        ("files", []),
        ("license", ""),
        ("provenance", ""),
        ("compute_types", []),
        ("worker_peak_ram_bytes", 0),
        ("validated_max_duration_seconds", 0),
        ("codec_matrix", []),
        ("evidence_sha256", ""),
        ("release_tested", False),
    ],
)
def test_incomplete_measured_model_evidence_fails(tmp_path, field, value):
    manifest = make_manifest()
    manifest["models"][0][field] = value
    path, _ = write_manifest(tmp_path, manifest)
    reason(ReasonCode.ASSET_VERIFICATION_FAILED,
           lambda: load_manifest(path, development=True))


@pytest.mark.parametrize("field", [
    "validated_max_duration_seconds",
    "decode_timeout_seconds",
])
def test_huge_integer_time_values_map_to_verification_failure(tmp_path, field):
    value = make_manifest()
    value["models"][0][field] = 10**400
    path, _ = write_manifest(tmp_path, value)
    reason(ReasonCode.ASSET_VERIFICATION_FAILED,
           lambda: load_manifest(path, development=True))


def test_worker_memory_is_explicit_and_storage_overhead_is_not_runtime_ram(tmp_path):
    value = make_manifest()
    value["capabilities"][0]["worker_memory_bytes"] = 2 * 1024**3
    path, _ = write_manifest(tmp_path, value)
    resolved = resolve_capability(load_manifest(path, development=True), "asr_cpu")
    assert resolved.worker_policy.memory_bytes == 2 * 1024**3
    assert resolved.worker_policy.memory_bytes != MIN_WORKER_OVERHEAD_BYTES
    assert resolved.worker_overhead_bytes == MIN_WORKER_OVERHEAD_BYTES


def test_worker_memory_is_not_statically_fitted_against_measured_envelopes(tmp_path):
    # SPEC_GAP-16: the operator chooses the model and selection checks live
    # memory, so a measured envelope larger than the worker cap still loads.
    value = make_manifest()
    value["capabilities"][0]["worker_memory_bytes"] = 2 * 1024**3
    value["models"][0]["worker_peak_ram_bytes"] = 8 * 1024**3
    path, _ = write_manifest(tmp_path, value)
    manifest = load_manifest(path, development=True)
    assert manifest.models[0].worker_peak_ram_bytes == 8 * 1024**3
    resolved = resolve_capability(manifest, "asr_cpu")
    assert resolved.envelopes[0].worker_peak_ram_bytes == 8 * 1024**3


def test_cuda_resolution_requires_valid_selected_gpu_uuid(tmp_path):
    path, _ = write_manifest(tmp_path, make_manifest(capability="asr_cuda"))
    manifest = load_manifest(path, development=True)
    for selected in (None, "all", "GPU-invalid"):
        reason(
            ReasonCode.ASSET_VERIFICATION_FAILED,
            lambda selected=selected: resolve_capability(
                manifest,
                "asr_cuda",
                selected_gpu_uuid=selected,
            ),
        )


def test_cpu_resolution_rejects_gpu_uuid_and_skips_cuda_measured_envelope(tmp_path):
    path, _ = write_manifest(tmp_path)
    manifest = load_manifest(path, development=True)
    reason(
        ReasonCode.ASSET_VERIFICATION_FAILED,
        lambda: resolve_capability(manifest, "asr_cpu", selected_gpu_uuid=GPU_UUID),
    )

    # A CUDA measurement (vram != 0) is not a CPU envelope: the model is still
    # an artifact for the CPU capability, with no envelope, never a failure.
    value = make_manifest()
    value["models"][0]["worker_peak_vram_bytes"] = 1
    path, _ = write_manifest(tmp_path, value)
    resolved = resolve_capability(load_manifest(path, development=True), "asr_cpu")
    assert [artifact.model_id for artifact in resolved.artifacts] == ["fixture-model"]
    assert resolved.envelopes == ()


def test_cuda_resolution_skips_cpu_measured_envelope(tmp_path):
    value = make_manifest(capability="asr_cuda")
    value["models"][0]["worker_peak_vram_bytes"] = 0
    path, _ = write_manifest(tmp_path, value)
    resolved = resolve_capability(
        load_manifest(path, development=True), "asr_cuda", selected_gpu_uuid=GPU_UUID,
    )
    assert [artifact.model_id for artifact in resolved.artifacts] == ["fixture-model"]
    assert resolved.envelopes == ()


def test_missing_requested_capability_and_invalid_request_map_to_verification(tmp_path):
    path, _ = write_manifest(tmp_path)
    manifest = load_manifest(path, development=True)
    reason(ReasonCode.ASSET_VERIFICATION_FAILED,
           lambda: resolve_capability(manifest, "media"))
    reason(ReasonCode.ASSET_VERIFICATION_FAILED,
           lambda: resolve_capability(manifest, "__invalid__"))


# ---- SPEC_GAP-13: development manifests may declare pinned real models ----------------


def test_development_real_model_loads_resolves_and_is_acquired_without_network(
    tmp_path,
    monkeypatch,
):
    path, raw = write_manifest(tmp_path, make_development_real_manifest())
    manifest = load_manifest(
        path,
        development=True,
        expected_sha256=hashlib.sha256(raw).hexdigest(),
    )
    assert manifest.development_fixture is True
    model = manifest.models[0]
    assert model.synthetic is False
    assert model.release_tested is True
    assert model.source_urls == (REAL_MODEL_SOURCE,)
    assert len(model.files) == len(model.source_urls)

    resolved = resolve_capability(manifest, Capability.ASR_CPU)
    assert resolved.worker_policy.development_image is True
    assert resolved.worker_policy.allow_pull is False
    assert resolved.worker_policy.image == "sha256:" + "1" * 64
    assert resolved.envelopes[0].synthetic is False
    assert resolved.envelopes[0].release_tested is True
    assert resolved.model_files == (("real-development-model", model.files),)

    cache, opened = stubbed_cache(tmp_path, monkeypatch, REAL_MODEL_BYTES)
    snapshot = cache.acquire(model)
    assert opened == [REAL_MODEL_SOURCE]
    assert snapshot.total_bytes == len(REAL_MODEL_BYTES)
    assert snapshot.path.joinpath("model.bin").read_bytes() == REAL_MODEL_BYTES


def test_development_real_model_acquisition_still_verifies_pinned_bytes(tmp_path, monkeypatch):
    path, _ = write_manifest(tmp_path, make_development_real_manifest())
    model = load_manifest(path, development=True).models[0]
    tampered = b"tampered payload with the same length"
    assert len(tampered) == len(REAL_MODEL_BYTES)
    cache, opened = stubbed_cache(tmp_path, monkeypatch, tampered)
    with pytest.raises(CacheError) as error:
        cache.acquire(model)
    assert error.value.reason_code is ReasonCode.ASSET_VERIFICATION_FAILED
    assert opened == [REAL_MODEL_SOURCE]


def test_synthetic_development_model_is_still_refused_by_model_cache(tmp_path, monkeypatch):
    path, _ = write_manifest(tmp_path)
    model = load_manifest(path, development=True).models[0]
    assert model.synthetic is True
    cache, opened = stubbed_cache(tmp_path, monkeypatch, b"synthetic model fixture")
    with pytest.raises(CacheError) as error:
        cache.acquire(model)
    assert error.value.reason_code is ReasonCode.ASSET_VERIFICATION_FAILED
    assert opened == []


def test_mixed_development_manifest_verifies_fixture_and_keeps_real_model(tmp_path):
    value = make_development_real_manifest()
    value["models"].append(make_manifest()["models"][0])
    path, _ = write_manifest(tmp_path, value)
    manifest = load_manifest(path, development=True)
    assert [model.synthetic for model in manifest.models] == [False, True]
    assert manifest.development_fixture is True
    resolved = resolve_capability(manifest, "asr_cpu")
    assert [envelope.synthetic for envelope in resolved.envelopes] == [False, True]

    # The synthetic sibling is still verified against its local fixture bytes.
    (tmp_path / "model.bin").write_bytes(b"x" * len(b"synthetic model fixture"))
    reason(ReasonCode.ASSET_VERIFICATION_FAILED,
           lambda: load_manifest(path, development=True))


def test_development_real_model_uses_release_file_limit_not_fixture_cap(tmp_path):
    value = make_development_real_manifest()
    value["models"][0]["files"][0]["bytes"] = 75_538_270
    assert 75_538_270 > MAX_FIXTURE_FILE_BYTES
    path, _ = write_manifest(tmp_path, value)
    manifest = load_manifest(path, development=True)
    assert manifest.models[0].files[0].bytes == 75_538_270

    synthetic = make_manifest()
    synthetic["models"][0]["files"][0]["bytes"] = MAX_FIXTURE_FILE_BYTES + 1
    path, _ = write_manifest(tmp_path, synthetic)
    reason(ReasonCode.ASSET_VERIFICATION_FAILED,
           lambda: load_manifest(path, development=True))


@pytest.mark.parametrize("asset_allowlist", [
    ("https://packages.podmuncher.org",),
    (REAL_MODEL_SOURCE,),
    ("https://cdn.podmuncher.org", "https://packages.podmuncher.org"),
])
def test_development_real_model_honours_given_asset_allowlist(tmp_path, asset_allowlist):
    path, _ = write_manifest(tmp_path, make_development_real_manifest())
    manifest = load_manifest(path, development=True, allowed_asset_sources=asset_allowlist)
    assert manifest.models[0].source_urls == (REAL_MODEL_SOURCE,)


@pytest.mark.parametrize("source_url,allowlist", [
    ("http://packages.podmuncher.org/development/model.bin", ()),
    ("fixture:model.bin", ()),
    ("https:///model.bin", ()),
    ("https://user@packages.podmuncher.org/model.bin", ()),
    ("https://packages.podmuncher.org/model.bin#fragment", ()),
    ("https://127.0.0.1/model.bin", ()),
    ("https://10.0.0.1/model.bin", ()),
    ("https://localhost/model.bin", ()),
    ("https://models.internal/model.bin", ()),
    (REAL_MODEL_SOURCE, ("https://cdn.podmuncher.org",)),
    (REAL_MODEL_SOURCE, ("https://packages.podmuncher.org/releases/model.bin",)),
])
def test_development_real_model_rejects_unpinned_private_and_unlisted_sources(
    tmp_path,
    source_url,
    allowlist,
):
    value = make_development_real_manifest()
    value["models"][0]["source_urls"] = [source_url]
    path, _ = write_manifest(tmp_path, value)
    reason(
        ReasonCode.ASSET_SOURCE_REJECTED,
        lambda: load_manifest(path, development=True, allowed_asset_sources=allowlist),
    )


def test_development_real_model_requires_pinned_size_and_hash_per_source(tmp_path):
    missing_sha = make_development_real_manifest()
    del missing_sha["models"][0]["files"][0]["sha256"]
    path, _ = write_manifest(tmp_path, missing_sha)
    reason(ReasonCode.ASSET_VERIFICATION_FAILED,
           lambda: load_manifest(path, development=True))

    for field, bad in (("sha256", ""), ("sha256", "A" * 64), ("bytes", 0), ("bytes", "1")):
        value = make_development_real_manifest()
        value["models"][0]["files"][0][field] = bad
        path, _ = write_manifest(tmp_path, value)
        reason(ReasonCode.ASSET_VERIFICATION_FAILED,
               lambda path=path: load_manifest(path, development=True))

    more_sources = make_development_real_manifest()
    more_sources["models"][0]["source_urls"] = [REAL_MODEL_SOURCE, REAL_MODEL_SOURCE]
    path, _ = write_manifest(tmp_path, more_sources)
    reason(ReasonCode.ASSET_VERIFICATION_FAILED,
           lambda: load_manifest(path, development=True))

    more_files = make_development_real_manifest()
    more_files["models"][0]["files"].append({
        "relative_path": "tokenizer.json",
        "bytes": 1,
        "sha256": "6" * 64,
    })
    path, _ = write_manifest(tmp_path, more_files)
    reason(ReasonCode.ASSET_VERIFICATION_FAILED,
           lambda: load_manifest(path, development=True))

    no_files = make_development_real_manifest()
    no_files["models"][0]["files"] = []
    no_files["models"][0]["source_urls"] = []
    path, _ = write_manifest(tmp_path, no_files)
    reason(ReasonCode.ASSET_VERIFICATION_FAILED,
           lambda: load_manifest(path, development=True))


def test_development_real_model_keeps_release_tested_meaning(tmp_path):
    path, _ = write_manifest(tmp_path, make_development_real_manifest(release_tested=False))
    manifest = load_manifest(path, development=True)
    assert manifest.development_fixture is True
    assert manifest.models[0].release_tested is False
    assert manifest.models[0].measured is True
    # The loader does not demand the claim; an untested measurement is never
    # handed to the model policy as an envelope.  The model stays an artifact.
    resolved = resolve_capability(manifest, "asr_cpu")
    assert [artifact.model_id for artifact in resolved.artifacts] == ["real-development-model"]
    assert resolved.envelopes == ()


def test_development_real_model_cannot_claim_release_source(tmp_path):
    value = make_development_real_manifest()
    value["source"] = {
        "kind": "packaged_release",
        "label": "release-1.0.0",
        "provenance": "publisher-build-2026-10",
    }
    path, raw = write_manifest(tmp_path, value)
    reason(ReasonCode.ASSET_SOURCE_REJECTED,
           lambda: load_manifest(path, development=True))
    # Loaded as a release it still fails: fixture-quality strings are not release evidence.
    reason(
        ReasonCode.ASSET_VERIFICATION_FAILED,
        lambda: load_manifest(
            path,
            allowed_release_sources=(path,),
            allowed_asset_sources=("https://packages.podmuncher.org",),
            expected_sha256=hashlib.sha256(raw).hexdigest(),
        ),
    )


@pytest.mark.parametrize(("field", "value", "expected"), [
    ("synthetic", True, ReasonCode.ASSET_VERIFICATION_FAILED),
    ("release_tested", False, ReasonCode.ASSET_VERIFICATION_FAILED),
    ("source_urls", ["fixture:model.bin"], ReasonCode.ASSET_SOURCE_REJECTED),
    (
        "source_urls",
        ["https://packages.podmuncher.org/releases/model.bin"] * 2,
        ReasonCode.ASSET_VERIFICATION_FAILED,
    ),
])
def test_packaged_release_model_rules_are_unchanged(tmp_path, field, value, expected):
    manifest = make_release_manifest()
    manifest["models"][0][field] = value
    path, raw = write_manifest(tmp_path, manifest)
    reason(
        expected,
        lambda: load_manifest(
            path,
            allowed_release_sources=(path,),
            allowed_asset_sources=("https://packages.podmuncher.org",),
            expected_sha256=hashlib.sha256(raw).hexdigest(),
        ),
    )


def test_packaged_release_still_requires_an_asset_allowlist(tmp_path):
    path, raw = write_manifest(tmp_path, make_release_manifest())
    reason(
        ReasonCode.ASSET_SOURCE_REJECTED,
        lambda: load_manifest(
            path,
            allowed_release_sources=(path,),
            expected_sha256=hashlib.sha256(raw).hexdigest(),
        ),
    )


# ---- SPEC_GAP-16: measured envelopes are optional; one manifest serves CPU and CUDA ----


@pytest.mark.parametrize("null", [False, True])
def test_unmeasured_model_loads_and_resolves_for_cpu_and_cuda(tmp_path, null):
    value = make_manifest()
    cuda = dict(make_manifest(capability="asr_cuda")["capabilities"][0])
    value["capabilities"].append(cuda)
    strip_envelope(value["models"][0], null=null)
    path, raw = write_manifest(tmp_path, value)
    manifest = load_manifest(
        path, development=True, expected_sha256=hashlib.sha256(raw).hexdigest(),
    )
    model = manifest.models[0]
    assert model.measured is False
    assert model.release_tested is None
    assert model.evidence_sha256 is None
    assert model.worker_peak_ram_bytes is None
    assert model.worker_peak_vram_bytes is None
    assert model.validated_max_duration_seconds is None
    assert model.decode_bytes_per_second is None
    assert model.decode_timeout_seconds is None
    # Identity, files, sources and licences are untouched.
    assert model.artifact.model_id == "fixture-model"
    assert model.artifact.sha256 == "3" * 64
    assert model.files[0].relative_path == "model.bin"
    assert model.source_urls == ("fixture:model.bin",)
    assert model.license == "fixture-license"

    cpu = resolve_capability(manifest, "asr_cpu")
    cuda_resolved = resolve_capability(manifest, "asr_cuda", selected_gpu_uuid=GPU_UUID)
    for resolved in (cpu, cuda_resolved):
        assert [artifact.model_id for artifact in resolved.artifacts] == ["fixture-model"]
        assert resolved.envelopes == ()
        assert resolved.model_files == (("fixture-model", model.files),)
    assert cpu.worker_policy.selected_gpu_uuid is None
    assert cuda_resolved.worker_policy.selected_gpu_uuid == GPU_UUID


def test_unmeasured_model_fixture_bytes_are_still_verified(tmp_path):
    path, _ = write_manifest(tmp_path, make_dual_manifest())
    (tmp_path / "model.bin").write_bytes(b"x" * len(b"synthetic model fixture"))
    reason(ReasonCode.ASSET_VERIFICATION_FAILED,
           lambda: load_manifest(path, development=True))


@pytest.mark.parametrize("missing", sorted(ENVELOPE_KEYS))
@pytest.mark.parametrize("null", [False, True])
def test_partial_envelope_is_rejected(tmp_path, missing, null):
    value = make_manifest()
    if null:
        value["models"][0][missing] = None
    else:
        del value["models"][0][missing]
    path, _ = write_manifest(tmp_path, value)
    reason(ReasonCode.ASSET_VERIFICATION_FAILED,
           lambda: load_manifest(path, development=True))


@pytest.mark.parametrize("present", sorted(ENVELOPE_KEYS))
def test_single_envelope_key_on_unmeasured_model_is_rejected(tmp_path, present):
    value = make_manifest()
    kept = value["models"][0][present]
    strip_envelope(value["models"][0])
    value["models"][0][present] = kept
    path, _ = write_manifest(tmp_path, value)
    reason(ReasonCode.ASSET_VERIFICATION_FAILED,
           lambda: load_manifest(path, development=True))


@pytest.mark.parametrize(("field", "value"), [
    ("worker_peak_ram_bytes", 0),
    ("worker_peak_ram_bytes", "1"),
    ("worker_peak_vram_bytes", -1),
    ("validated_max_duration_seconds", 0),
    ("decode_bytes_per_second", DECODE_BYTES_PER_SECOND - 1),
    ("decode_timeout_seconds", 0),
    ("evidence_sha256", "A" * 64),
    ("release_tested", "yes"),
    ("release_tested", 1),
])
def test_complete_envelope_is_still_validated(tmp_path, field, value):
    manifest = make_manifest()
    manifest["models"][0][field] = value
    path, _ = write_manifest(tmp_path, manifest)
    reason(ReasonCode.ASSET_VERIFICATION_FAILED,
           lambda: load_manifest(path, development=True))


def test_unmeasured_model_rejects_unknown_keys_and_loose_identity(tmp_path):
    unknown = make_dual_manifest()
    unknown["models"][0]["peak_bytes"] = 1
    path, _ = write_manifest(tmp_path, unknown)
    reason(ReasonCode.ASSET_VERIFICATION_FAILED,
           lambda: load_manifest(path, development=True))

    for field in ("id", "tier", "revision", "digest", "source_urls", "files",
                  "license", "provenance", "compute_types", "codec_matrix", "synthetic"):
        value = make_dual_manifest()
        del value["models"][0][field]
        path, _ = write_manifest(tmp_path, value)
        reason(ReasonCode.ASSET_VERIFICATION_FAILED,
               lambda path=path: load_manifest(path, development=True))

    bad_hash = make_dual_manifest()
    bad_hash["models"][0]["digest"] = "A" * 64
    path, _ = write_manifest(tmp_path, bad_hash)
    reason(ReasonCode.ASSET_VERIFICATION_FAILED,
           lambda: load_manifest(path, development=True))


def test_one_manifest_serves_cpu_and_cuda_with_mixed_envelopes(tmp_path):
    value = make_dual_manifest()
    cpu_measured = dict(make_manifest()["models"][0])
    cpu_measured["id"] = "cpu-measured"
    cpu_measured["files"] = [dict(cpu_measured["files"][0], relative_path="cpu.bin")]
    cpu_measured["source_urls"] = ["fixture:cpu.bin"]
    cuda_measured = dict(make_manifest(capability="asr_cuda")["models"][0])
    cuda_measured["id"] = "cuda-measured"
    cuda_measured["files"] = [dict(cuda_measured["files"][0], relative_path="cuda.bin")]
    cuda_measured["source_urls"] = ["fixture:cuda.bin"]
    value["models"].extend([cpu_measured, cuda_measured])
    path, _ = write_manifest(tmp_path, value)
    (tmp_path / "cpu.bin").write_bytes(b"synthetic model fixture")
    (tmp_path / "cuda.bin").write_bytes(b"synthetic model fixture")
    manifest = load_manifest(path, development=True)
    assert [model.measured for model in manifest.models] == [False, True, True]

    cpu = resolve_capability(manifest, "asr_cpu")
    cuda = resolve_capability(manifest, "asr_cuda", selected_gpu_uuid=GPU_UUID)
    expected_ids = ["fixture-model", "cpu-measured", "cuda-measured"]
    assert [artifact.model_id for artifact in cpu.artifacts] == expected_ids
    assert [artifact.model_id for artifact in cuda.artifacts] == expected_ids
    assert [envelope.artifact.model_id for envelope in cpu.envelopes] == ["cpu-measured"]
    assert cpu.envelopes[0].backend is Backend.CPU
    assert cpu.envelopes[0].worker_peak_vram_bytes == 0
    assert [envelope.artifact.model_id for envelope in cuda.envelopes] == ["cuda-measured"]
    assert cuda.envelopes[0].backend is Backend.CUDA
    assert cuda.envelopes[0].worker_peak_vram_bytes == 1024**3
    assert cuda.envelopes[0].runtime.driver_version == "fixture-driver"
    assert [model_id for model_id, _files in cpu.model_files] == expected_ids


def test_packaged_release_accepts_unmeasured_model_without_release_tested(tmp_path):
    value = make_release_manifest()
    strip_envelope(value["models"][0])
    path, raw = write_manifest(tmp_path, value)
    manifest = load_manifest(
        path,
        allowed_release_sources=(path,),
        allowed_asset_sources=("https://packages.podmuncher.org",),
        expected_sha256=hashlib.sha256(raw).hexdigest(),
    )
    assert manifest.development_fixture is False
    assert manifest.models[0].measured is False
    assert manifest.models[0].release_tested is None
    resolved = resolve_capability(manifest, "asr_cpu")
    assert resolved.worker_policy.allow_pull is True
    assert [artifact.model_id for artifact in resolved.artifacts] == ["fixture-model"]
    assert resolved.envelopes == ()


def test_packaged_release_integrity_is_unchanged_for_unmeasured_models(tmp_path):
    # Hash pins, https sources, the asset allowlist and the expected manifest
    # hash all still gate an unmeasured release model.
    for field, bad, expected in (
        ("digest", "A" * 64, ReasonCode.ASSET_VERIFICATION_FAILED),
        ("source_urls", ["http://packages.podmuncher.org/releases/model.bin"],
         ReasonCode.ASSET_SOURCE_REJECTED),
        ("source_urls", ["https://cdn.example.org/model.bin"], ReasonCode.ASSET_SOURCE_REJECTED),
        ("synthetic", True, ReasonCode.ASSET_VERIFICATION_FAILED),
        ("revision", "unresolved", ReasonCode.ASSET_VERIFICATION_FAILED),
    ):
        value = make_release_manifest()
        strip_envelope(value["models"][0])
        value["models"][0][field] = bad
        path, raw = write_manifest(tmp_path, value)
        reason(
            expected,
            lambda path=path, raw=raw: load_manifest(
                path,
                allowed_release_sources=(path,),
                allowed_asset_sources=("https://packages.podmuncher.org",),
                expected_sha256=hashlib.sha256(raw).hexdigest(),
            ),
        )

    value = make_release_manifest()
    strip_envelope(value["models"][0])
    path, raw = write_manifest(tmp_path, value)
    reason(
        ReasonCode.ASSET_VERIFICATION_FAILED,
        lambda: load_manifest(
            path,
            allowed_release_sources=(path,),
            allowed_asset_sources=("https://packages.podmuncher.org",),
            expected_sha256="f" * 64,
        ),
    )
    reason(
        ReasonCode.ASSET_SOURCE_REJECTED,
        lambda: load_manifest(
            path,
            allowed_release_sources=(path,),
            expected_sha256=hashlib.sha256(raw).hexdigest(),
        ),
    )


def test_packaged_release_measured_model_still_requires_release_tested(tmp_path):
    for bad in (False, None):
        value = make_release_manifest()
        value["models"][0]["release_tested"] = bad
        path, raw = write_manifest(tmp_path, value)
        reason(
            ReasonCode.ASSET_VERIFICATION_FAILED,
            lambda path=path, raw=raw: load_manifest(
                path,
                allowed_release_sources=(path,),
                allowed_asset_sources=("https://packages.podmuncher.org",),
                expected_sha256=hashlib.sha256(raw).hexdigest(),
            ),
        )


def test_synthetic_measured_model_still_requires_release_tested(tmp_path):
    value = make_manifest()
    value["models"][0]["release_tested"] = False
    path, _ = write_manifest(tmp_path, value)
    reason(ReasonCode.ASSET_VERIFICATION_FAILED,
           lambda: load_manifest(path, development=True))


def test_unmeasured_development_real_model_is_acquired_without_network(tmp_path, monkeypatch):
    value = make_development_real_manifest()
    strip_envelope(value["models"][0])
    path, _ = write_manifest(tmp_path, value)
    manifest = load_manifest(path, development=True)
    model = manifest.models[0]
    assert model.measured is False and model.synthetic is False
    resolved = resolve_capability(manifest, Capability.ASR_CPU)
    assert resolved.envelopes == ()
    assert resolved.model_files == (("real-development-model", model.files),)
    cache, opened = stubbed_cache(tmp_path, monkeypatch, REAL_MODEL_BYTES)
    snapshot = cache.acquire(model)
    assert opened == [REAL_MODEL_SOURCE]
    assert snapshot.path.joinpath("model.bin").read_bytes() == REAL_MODEL_BYTES


# ---- SPEC_GAP-17: multi-architecture images declare a platforms list -------------------


MULTI_PLATFORMS = ["linux-amd64", "linux-arm64"]


def make_multiarch_manifest(*, capability="asr_cpu", entry_platforms=None):
    """A development fixture whose root platforms are the two published architectures."""
    manifest = make_manifest(capability=capability)
    manifest["platforms"] = list(MULTI_PLATFORMS)
    entry = manifest["capabilities"][0]
    entry["platform"] = "linux-amd64"
    if entry_platforms is not None:
        entry["platforms"] = entry_platforms
    return manifest


def test_multi_platform_capability_loads_and_resolves(tmp_path):
    value = make_multiarch_manifest(entry_platforms=list(MULTI_PLATFORMS))
    path, raw = write_manifest(tmp_path, value)
    manifest = load_manifest(
        path, development=True, expected_sha256=hashlib.sha256(raw).hexdigest(),
    )
    entry = manifest.capabilities[0]
    assert entry.platform == "linux-amd64"
    assert entry.platforms == ("linux-amd64", "linux-arm64")
    resolved = resolve_capability(manifest, "asr_cpu")
    assert resolved.platforms == ("linux-amd64", "linux-arm64")
    with pytest.raises(FrozenInstanceError):
        resolved.platforms = ()
    for arch in ("x86_64", "amd64", "aarch64", "arm64", " AArch64 ", "AMD64"):
        assert supports_engine_arch(resolved, arch) is True
        assert supports_engine_arch(entry, arch) is True


def test_single_platform_form_is_unchanged(tmp_path):
    # No ``platforms`` key: the entry is the one platform it names.
    path, _ = write_manifest(tmp_path)
    manifest = load_manifest(path, development=True)
    assert manifest.capabilities[0].platform == "fixture-platform"
    assert manifest.capabilities[0].platforms == ("fixture-platform",)
    resolved = resolve_capability(manifest, "asr_cpu")
    assert resolved.platforms == ("fixture-platform",)
    # A fixture platform names no architecture, so no engine supports it here.
    assert supports_engine_arch(resolved, "x86_64") is False
    assert supports_engine_arch(resolved, "aarch64") is False

    # An explicit null is the single-platform form too.
    nulled = make_manifest()
    nulled["capabilities"][0]["platforms"] = None
    path, _ = write_manifest(tmp_path, nulled)
    assert load_manifest(path, development=True).capabilities[0].platforms == (
        "fixture-platform",
    )

    # One architecture, declared either way, supports only that engine.
    single = make_multiarch_manifest()
    path, _ = write_manifest(tmp_path, single)
    resolved = resolve_capability(load_manifest(path, development=True), "asr_cpu")
    assert resolved.platforms == ("linux-amd64",)
    assert supports_engine_arch(resolved, "x86_64") is True
    assert supports_engine_arch(resolved, "aarch64") is False
    listed = make_multiarch_manifest(entry_platforms=["linux-amd64"])
    path, _ = write_manifest(tmp_path, listed)
    resolved = resolve_capability(load_manifest(path, development=True), "asr_cpu")
    assert resolved.platforms == ("linux-amd64",)
    assert supports_engine_arch(resolved, "arm64") is False


@pytest.mark.parametrize("entry_platforms", [
    ["linux-amd64", "linux-riscv64"],  # not in the root list
    ["linux-arm64"],  # omits the entry's own ``platform``
    [],  # empty
    ["linux-amd64", "linux-amd64"],  # duplicate
    "linux-amd64",  # not a list
    ["linux-amd64", 1],  # not a string
    ["linux-amd64", "not an identifier"],
    [["linux-amd64"]],
])
def test_platforms_list_rejects_unknown_and_malformed_entries(tmp_path, entry_platforms):
    value = make_multiarch_manifest(entry_platforms=entry_platforms)
    path, _ = write_manifest(tmp_path, value)
    reason(ReasonCode.ASSET_VERIFICATION_FAILED,
           lambda: load_manifest(path, development=True))


def test_platforms_list_keeps_capability_ids_unique(tmp_path):
    value = make_multiarch_manifest(entry_platforms=["linux-amd64"])
    arm = dict(value["capabilities"][0])
    arm["platform"] = "linux-arm64"
    arm["platforms"] = ["linux-arm64"]
    value["capabilities"].append(arm)
    path, _ = write_manifest(tmp_path, value)
    reason(ReasonCode.ASSET_VERIFICATION_FAILED,
           lambda: load_manifest(path, development=True))


def test_release_manifest_accepts_multi_platform_and_single_platform_cuda(tmp_path):
    value = make_release_manifest()
    value["platforms"] = list(MULTI_PLATFORMS)
    cpu = value["capabilities"][0]
    cpu["platform"] = "linux-amd64"
    cpu["platforms"] = list(MULTI_PLATFORMS)
    cuda = dict(make_release_manifest()["capabilities"][0])
    cuda.update({
        "id": "asr_cuda",
        "platform": "linux-amd64",
        "platforms": ["linux-amd64"],
        "driver_version": "driver-1.0",
    })
    value["capabilities"].append(cuda)
    path, raw = write_manifest(tmp_path, value)
    manifest = load_manifest(
        path,
        allowed_release_sources=(path,),
        allowed_asset_sources=("https://packages.podmuncher.org",),
        expected_sha256=hashlib.sha256(raw).hexdigest(),
    )
    cpu_resolved = resolve_capability(manifest, "asr_cpu")
    cuda_resolved = resolve_capability(manifest, "asr_cuda", selected_gpu_uuid=GPU_UUID)
    assert cpu_resolved.platforms == ("linux-amd64", "linux-arm64")
    assert cuda_resolved.platforms == ("linux-amd64",)
    # An Apple Silicon engine (docker info Architecture aarch64) runs the CPU
    # image and skips CUDA; an amd64 engine runs both.
    assert supports_engine_arch(cpu_resolved, "aarch64") is True
    assert supports_engine_arch(cuda_resolved, "aarch64") is False
    assert supports_engine_arch(cpu_resolved, "x86_64") is True
    assert supports_engine_arch(cuda_resolved, "x86_64") is True


@pytest.mark.parametrize(("arch", "expected"), [
    ("x86_64", "linux-amd64"),
    ("amd64", "linux-amd64"),
    ("aarch64", "linux-arm64"),
    ("arm64", "linux-arm64"),
    ("AArch64", "linux-arm64"),
    (" x86_64\n", "linux-amd64"),
    ("arm/v7", None),
    ("armv7l", None),
    ("linux/arm64", None),
    ("Unknown", None),
    ("", None),
    (None, None),
    (b"x86_64", None),
    (64, None),
])
def test_engine_arch_platform_mapping(arch, expected):
    assert engine_arch_platform(arch) == expected
    assert set(ENGINE_ARCH_PLATFORMS.values()) == set(MULTI_PLATFORMS)


def test_supports_engine_arch_fails_closed_on_unmapped_arch_and_wrong_type(tmp_path):
    value = make_multiarch_manifest(entry_platforms=list(MULTI_PLATFORMS))
    path, _ = write_manifest(tmp_path, value)
    resolved = resolve_capability(load_manifest(path, development=True), "asr_cpu")
    for arch in ("arm/v7", "", None, "Unknown", "linux/arm64"):
        assert supports_engine_arch(resolved, arch) is False
    for wrong in (None, "asr_cpu", resolved.worker_policy, resolved.platforms):
        reason(ReasonCode.ASSET_VERIFICATION_FAILED,
               lambda wrong=wrong: supports_engine_arch(wrong, "x86_64"))
