"""Product composition (P6-U2): real loopback service, CLI binding and the live smoke."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import types
import uuid
import wave
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar, Self

import pytest

from podmuncher import bootstrap, cli, service
from podmuncher.domains import (
    ArtifactSlot,
    Capability,
    DomainValidationError,
    FieldStatus,
    JobState,
    ReasonCode,
    ServiceState,
    WorkerSlot,
)
from podmuncher.service import ClientError

ROOT = Path(__file__).resolve().parents[1]
# The repo venv's console script: ``Scripts\podmuncher.exe`` on Windows, ``bin/podmuncher`` on
# POSIX (the Mac test venv is installed with ``uv pip install -e .``).
CONSOLE = ROOT / ".venv" / ("Scripts" if os.name == "nt" else "bin") / (
    "podmuncher.exe" if os.name == "nt" else "podmuncher")
MODEL_BYTES = b"synthetic model fixture"
DIGEST = "sha256:" + "1" * 64
SNAPSHOT = "snapshot-" + "a" * 32
TRANSCRIPT_SLOTS = sorted(s.value for s in ArtifactSlot if s is not ArtifactSlot.ORIGINAL_VIDEO)
HOSTILE = "evil\x1b[31m [bold]markup[/bold] \u202edesrever\x07 end"
FACTORY_MODULE = "podmuncher_p6u2_test_factory"
MIB, GIB = 1024**2, 1024**3
STAMP = "%Y-%m-%dT%H:%M:%S.%fZ"
RUNTIME_VERSION = "fixture-runtime"
GPU_UUID = "GPU-0123abcd-4567-89ab-cdef-0123456789ab"
PROBE_DOCUMENT = {"schema_version": 1, "runtime": {"name": "ctranslate2", "version": RUNTIME_VERSION},
                  "compute_types": ["float32", "int8"], "device": "cpu", "cuda_device_uuid": None,
                  "faster_whisper_version": "1.2.1"}
_TIERS = {"tiny": "tiny", "base": "base", "small": "small", "medium": "medium", "large-v3": "large"}


# --------------------------------------------------------------------------- environment


def _capability(cap: str, worker_memory_bytes: int = 4 * GIB) -> dict:
    return {
        "id": cap, "platform": "fixture-platform", "image_reference": "fixture-local",
        "digest": DIGEST, "entrypoint": ["python", "-m", "fixture_worker"],
        "compressed_bytes": 100, "unpacked_peak_bytes": 200,
        "worker_overhead_bytes": 256 * 1024**2 + 2 * 10 * 1024**2,
        "worker_memory_bytes": worker_memory_bytes, "dependency_lock_sha256": "2" * 64,
        "licenses": ["fixture-license"], "provenance": "fixture:local-image",
        "runtime_version": RUNTIME_VERSION, "driver_version": None,
        "hardware_fingerprint": "fixture-hardware",
    }


def _model(model_id: str, *, measured: bool = True) -> dict:
    """A verified fixture model; ``measured`` adds the complete CPU envelope (SPEC_GAP-16:
    an unmeasured model carries none of the envelope keys and is fitted from the table)."""
    doc = {
        "id": model_id, "tier": _TIERS.get(model_id, "small"), "revision": "fixture-revision",
        "digest": "3" * 64, "source_urls": [f"fixture:{model_id}.bin"],
        "files": [{"relative_path": f"{model_id}.bin", "bytes": len(MODEL_BYTES),
                   "sha256": hashlib.sha256(MODEL_BYTES).hexdigest()}],
        "license": "fixture-license", "provenance": "fixture:generated-test-data",
        "compute_types": ["int8"], "codec_matrix": ["fixture_codec"], "synthetic": True,
    }
    if measured:
        doc.update({"worker_peak_ram_bytes": 512 * 1024**2, "worker_peak_vram_bytes": 0,
                    "validated_max_duration_seconds": 3600, "decode_bytes_per_second": 256000,
                    "decode_timeout_seconds": 120, "evidence_sha256": "4" * 64,
                    "release_tested": True})
    return doc


def make_manifest_doc(model_id: str = "small", *, models: tuple[str, ...] | None = None,
                      worker_memory_bytes: int = 4 * GIB, unmeasured: tuple[str, ...] = ()) -> dict:
    """Explicitly synthetic development fixture (shape of manifests.py:101)."""
    return {
        "schema_version": 1, "release_version": "fixture",
        "source": {"kind": "development_fixture", "label": "fixture",
                   "provenance": "fixture:test_bootstrap"},
        "platforms": ["fixture-platform"],
        "capabilities": [_capability("media"), _capability("asr_cpu", worker_memory_bytes)],
        "models": [_model(m, measured=m not in unmeasured) for m in (models or (model_id,))],
        "languages": ["en"],
    }


def with_cuda(doc: dict) -> dict:
    """The same manifest with an ``asr_cuda`` capability (one manifest serves both backends)."""
    cuda = _capability("asr_cuda")
    cuda["driver_version"] = "fixture-driver"
    doc["capabilities"].append(cuda)
    return doc


def _toml_path(path: Path) -> str:
    return "'" + str(path) + "'"


def make_bootstrap_env(tmp_path: Path, *, model_id: str = "small",
                       manifest_doc: dict | None = None) -> SimpleNamespace:
    """Tiny temp roots, an explicit fixture manifest, a config TOML and the three env vars."""
    roots = {name: tmp_path / name for name in ("data", "output", "cache", "scratch", "state",
                                                "manifest", "input")}
    for path in roots.values():
        path.mkdir(parents=True, exist_ok=True)
    manifest = roots["manifest"] / "manifest.json"
    doc = manifest_doc or make_manifest_doc(model_id)
    manifest.write_text(json.dumps(doc), encoding="utf-8")
    for model in doc["models"]:
        for item in model["files"]:
            (roots["manifest"] / item["relative_path"]).write_bytes(MODEL_BYTES)
    config = tmp_path / "config.toml"
    config.write_text("\n".join([
        "schema_version = 1", "[paths]",
        *(f"{name} = {_toml_path(roots[name])}" for name in ("data", "output", "cache", "scratch")),
        "[limits]", "media_bytes = 8388608", "duration_seconds = 600", "export_bytes = 1048576",
        "reserve_fraction = 0.0", "reserve_floor_bytes = 0", "reserve_ceiling_bytes = 0", "",
    ]), encoding="utf-8")
    source = roots["input"] / "talk.mp4"
    source.write_bytes(b"\x00\x00\x00\x18ftypmp42 synthetic media bytes")
    env = {"PODMUNCHER_STATE_DIR": str(roots["state"]), "PODMUNCHER_CONFIG": str(config),
           "PODMUNCHER_RUNTIME_MANIFEST": str(manifest)}
    return SimpleNamespace(roots=roots, manifest=manifest, config=config, source=source, env=env)


def job_request(env: SimpleNamespace, *, model: str = "small", language: str = "auto") -> dict:
    return {"schema_version": 1, "source": {"kind": "local", "input": str(env.source)},
            "outputs": {"transcript": True, "video": False, "video_quality": "1080p"},
            "selection": {"device": "auto", "profile": "balanced", "model": model,
                          "language": language},
            "output_root": str(env.roots["output"]),
            "limits": {"media_bytes": 8388608, "duration_seconds": 600, "export_bytes": 1048576}}


# ------------------------------------------------------------------------- fake Docker seam


def _measurement(value: Any, source: str = "fake") -> Any:
    from podmuncher.snapshots import Measurement

    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    return Measurement(value, source, FieldStatus.OK, stamp, None)


class FakeDockerStorage:
    """Docker storage evidence on the real temp volume (pattern: docker_storage tests)."""

    def __init__(self, root: Path) -> None:
        from podmuncher.volumes import make_volume_provider

        self.root, self.volumes, self.calls = root, make_volume_provider(), 0

    def sample(self, *, manual_backing: str = "") -> Any:
        from podmuncher.docker_storage import Constraint, DockerStorageEvidence

        self.calls += 1
        volume = self.volumes.sample(str(self.root))
        return DockerStorageEvidence(
            _measurement("desktop-linux"), _measurement("/var/lib/docker"), _measurement("4.0"),
            (Constraint(("docker_data",), ("/var/lib/docker",), volume),), (),
            _measurement(True), time.monotonic())


class FakeContainers:
    """recovery.OwnedContainers (pattern: tests/test_recovery.py FakeContainers)."""

    def __init__(self) -> None:
        self.log: list = []

    async def list(self, instance: str) -> list:
        self.log.append(("list", instance))
        return []

    async def stop(self, ref: Any) -> None: ...

    async def kill(self, ref: Any) -> None: ...

    async def remove(self, ref: Any) -> None: ...

    async def exists(self, ref: Any) -> bool:
        return False


class FakeLauncher:
    """scheduler.WorkerLauncher (pattern: tests/test_scheduler.py FakeLauncher).

    It writes real worker outputs into the host-assigned private scratch, so the real
    publisher verifies, exports and publishes them.
    """

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx
        self.ops: list[str] = []
        self.selections: list[tuple[str, str]] = []
        self.prepared: set[str] = set()
        self.gates: dict[str, threading.Event] = {}
        self.entered: dict[str, threading.Event] = {}
        self.fail_once: set[str] = set()
        self.oom_once: set[str] = set()
        self.stopped = False
        self.stops: list = []
        self.has_video = False  # what the fake probe reports
        self.inputs: list[tuple[str, str | None, bool]] = []  # (op, input basename, exists)

    @property
    def writers_reaped(self) -> bool:
        return True

    @property
    def running(self) -> bool:
        return False

    def ready(self, capability: str) -> bool:
        return capability in self.prepared

    async def prepare(self, capability: str) -> None:
        self.prepared.add(capability)

    async def run(self, request: Any, *, stage: Any, on_event: Any) -> Any:
        from podmuncher.hardware import AdmittedWorkerRef
        from podmuncher.protocol import WorkerEventRecord
        from podmuncher.runtime import RuntimeFailure, WorkerResult

        op = request.operation
        self.ops.append(op)
        self.stopped = False
        self.entered.setdefault(op, threading.Event()).set()
        gate = self.gates.get(op)
        while gate is not None and not gate.is_set() and not self.stopped:
            await asyncio.sleep(0.01)
        if self.stopped:
            raise RuntimeFailure(ReasonCode.CANCELLED)
        if op in self.fail_once:
            self.fail_once.discard(op)
            raise RuntimeFailure(ReasonCode.WORKER_EXIT_UNEXPECTED)
        if op == "transcribe":
            self.selections.append((request.selection.device, request.selection.model))
        if op in self.oom_once:
            self.oom_once.discard(op)
            raise RuntimeFailure(ReasonCode.MODEL_OOM)
        basename = request.input_basename
        if basename is not None:  # a scratch input must still exist when the worker starts
            try:
                exists = Path(self.ctx.scratch.path_of(basename)).is_file()
            except DomainValidationError:  # a local source: mounted from its own path
                exists = None
            self.inputs.append((op, basename, exists))
        slot, name = next(iter(request.assigned_slots.items()))
        if op == "download":  # the media slot the scheduler verifies: merged, else the only one
            slot = "merged_video" if "merged_video" in request.assigned_slots else slot
            name = request.assigned_slots[slot]
            data = b"\x00\x00\x00\x18ftypmp42 downloaded media " + b"\x01" * 4096
        elif op == "probe":
            data = json.dumps({"schema_version": 1, "operation": "probe", "source_kind": None,
                               "duration_seconds": 4.0, "has_audio": True,
                               "has_video": self.has_video, "video_height": None,
                               "audio_streams": 1, "video_streams": int(self.has_video),
                               "estimated_bytes": None}).encode()
        else:
            data = json.dumps({
                "schema_version": 1, "language": "en", "language_probability": 0.9,
                "duration_seconds": 4.0, "no_speech_detected": False,
                "segments": [{"start_seconds": 0.0, "end_seconds": 1.5, "text": HOSTILE},
                             {"start_seconds": 1.5, "end_seconds": 3.5, "text": "second line"}],
                "provenance": {"model_id": request.selection.model,
                               "model_snapshot_id": request.model_snapshot_id,
                               "device": "cpu", "compute_type": "int8",
                               "source_sha256": "5" * 64, "attempt_id": request.attempt_id,
                               "image_digest": None},
                "measurements": {}}).encode()
        Path(self.ctx.scratch.path_of(name)).write_bytes(data)
        claim = WorkerEventRecord(1, request.job_id, request.attempt_id, request.stage_id, 1,
                                  "artifact", {"slot": slot, "bytes": len(data),
                                               "sha256": hashlib.sha256(data).hexdigest()})
        ref = AdmittedWorkerRef("ctx", "npipe:////./pipe/docker_engine", "b" * 64, "inst", "job", "att")
        return WorkerResult(ref, True, 0, (claim,))

    async def stop(self, reason: Any) -> bool:
        self.stops.append(reason)
        self.stopped = True
        return True


class FakeModels:
    """scheduler.ModelProvider: no in-worker evidence exists in the fake seam."""

    def select(self, job: Any, *, facts: Any, lower_than: Any = None) -> str:
        return "small"

    def capability(self, choice: Any) -> str:
        return "asr_cpu"

    def selection(self, choice: Any) -> tuple[str, str]:
        return "cpu", "small"

    async def prepare(self, choice: Any) -> str:
        return SNAPSHOT


def fake_hardware() -> Any:
    from podmuncher.snapshots import Measurement

    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    gone = Measurement(None, "host_nvidia_smi", FieldStatus.UNKNOWN, stamp, ReasonCode.GPU_UNAVAILABLE)
    return SimpleNamespace(sample=lambda: SimpleNamespace(
        host_ram_used=_measurement(8 * 1024**3), host_cpu_percent=_measurement(12.5),
        gpu_name=gone, gpu_utilization_percent=gone, gpu_vram_used=gone))


# ------------------------------------------------------------------- fake probe seam


def hardware_evidence(ref: Any, compute: Any, *, stamp: str, ram: tuple[int, int, int],
                      gpu: tuple[str, int, int] | None = None) -> Any:
    """A complete ``HardwareEvidence`` for an admitted worker, stamped as given.

    ``gpu`` is ``(driver, vram total, vram free)`` for a CUDA worker; a CPU worker's GPU
    readings are unknown."""
    from podmuncher.hardware import HardwareEvidence
    from podmuncher.snapshots import Measurement

    def ok(value: Any, source: str = "fake") -> Any:
        return Measurement(value, source, FieldStatus.OK, stamp, None)

    gone = Measurement(None, "host_nvidia_smi", FieldStatus.UNKNOWN, stamp, ReasonCode.GPU_UNAVAILABLE)
    limit, current, available = ram
    readings = {name: gone for name in ("gpu_name", "gpu_driver", "gpu_vram_total", "gpu_vram_free",
                                        "gpu_vram_used", "gpu_utilization_percent")}
    if gpu is not None:
        driver, total, free = gpu
        readings = {"gpu_name": ok("fixture-gpu"), "gpu_driver": ok(driver), "gpu_vram_total": ok(total),
                    "gpu_vram_free": ok(free), "gpu_vram_used": ok(total - free),
                    "gpu_utilization_percent": ok(0.0)}
    return HardwareEvidence(
        host_cpu_percent=ok(12.5), host_cpu_cores=ok(8), host_ram_total=ok(32 * GIB),
        host_ram_available=ok(16 * GIB), host_ram_used=ok(16 * GIB), worker_cpu_percent=ok(1.0),
        worker_cpu_raw_percent=ok(1.0), worker_cpu_limit_cores=ok(1.0), worker_ram_limit=ok(limit),
        worker_ram_current=ok(current), worker_ram_available=ok(available),
        worker_gpu_visible=ok(ref.gpu_visible), **readings,
        runtime_compute_types=ok(list(compute), "admitted_runtime"))


def gpu_host(total: int = 8 * GIB, free: int = 6 * GIB, driver: str = "fixture-driver") -> Any:
    """A host hardware sampler reporting a usable NVIDIA GPU (``_ProbeEvidence._usable_gpu``)."""
    from podmuncher.snapshots import Measurement

    stamp = datetime.now(UTC).strftime(STAMP)
    return SimpleNamespace(sample=lambda: SimpleNamespace(
        gpu_name=Measurement("fixture-gpu", "host_nvidia_smi", FieldStatus.OK, stamp, None),
        gpu_driver=Measurement(driver, "host_nvidia_smi", FieldStatus.OK, stamp, None),
        gpu_vram_total=Measurement(total, "host_nvidia_smi", FieldStatus.OK, stamp, None),
        gpu_vram_free=Measurement(free, "host_nvidia_smi", FieldStatus.OK, stamp, None)))


def identity_host(engine_arch: str | None, *, host_os: str = "Darwin", host_arch: str = "arm64",
                  host_cpu_model: str = "Apple M4", gpu: bool = False) -> Any:
    """A host hardware sampler with the SPEC_GAP-17 identity fields (Apple M4 shape).

    ``engine_arch`` None is the unmeasured engine (``docker info`` failed); ``gpu`` adds
    a usable NVIDIA GPU so the CUDA path can be exercised on an engine of either arch."""
    from podmuncher.snapshots import Measurement

    stamp = datetime.now(UTC).strftime(STAMP)

    def ok(value: Any, source: str) -> Any:
        return Measurement(value, source, FieldStatus.OK, stamp, None)

    fields = {"host_os": ok(host_os, "host_platform"), "host_arch": ok(host_arch, "host_platform"),
              "host_cpu_model": ok(host_cpu_model, "host_sysctl"),
              "engine_arch": ok(engine_arch, "docker_info") if engine_arch is not None else
              Measurement(None, "docker_info", FieldStatus.UNKNOWN, stamp, ReasonCode.DOCKER_UNAVAILABLE)}
    if gpu:
        fields.update(gpu_name=ok("fixture-gpu", "host_nvidia_smi"),
                      gpu_driver=ok("fixture-driver", "host_nvidia_smi"),
                      gpu_vram_total=ok(8 * GIB, "host_nvidia_smi"),
                      gpu_vram_free=ok(6 * GIB, "host_nvidia_smi"))
    return SimpleNamespace(sample=lambda: SimpleNamespace(**fields))


def release_manifest(ctx: Any, platforms: dict[str, tuple[str, ...]]) -> None:
    """Reshape the loaded fixture manifest as a release: not a development fixture, with the
    given per-capability ``platforms`` (every other value stays the fixture's).  The engine
    architecture gate only applies to release images; fixtures always pass."""
    capabilities = tuple(replace(c, platforms=platforms.get(c.capability.value, c.platforms))
                         for c in ctx.manifest.capabilities)
    ctx.manifest = replace(ctx.manifest, development_fixture=False, capabilities=capabilities)


class FakeProbeRun:
    """One fake single-use worker run: started -> artifact (written to scratch) -> hold -> completed."""

    def __init__(self, owner: FakeProbeLaunch, policy: Any, mounts: Any, request: Any,
                 on_event: Any) -> None:
        self.owner, self.policy, self.mounts, self.request = owner, policy, mounts, request
        self.on_event = on_event
        self.reference: Any = None
        self.task: asyncio.Task | None = None
        self.obsoleted = False
        self.closed = False
        self.ended_at: float | None = None

    def obsolete(self) -> None:
        self.obsoleted = True

    async def close(self) -> None:
        self.closed = True

    def _event(self, sequence: int, name: str, payload: dict) -> Any:
        from podmuncher.protocol import WorkerEventRecord

        request = self.request
        return WorkerEventRecord(1, request.job_id, request.attempt_id, request.stage_id, sequence,
                                 name, payload)

    @property
    def cuda(self) -> bool:
        """This run is a CUDA worker: the fake GPU is set and the request asked for it."""
        return self.owner.gpu is not None and self.request.selection.device == "cuda"

    def _document(self) -> bytes:
        request = self.request
        if request.operation == "probe":
            document = self.owner.document
            if self.cuda and type(document) is dict:
                document = {**document, "device": "cuda", "cuda_device_uuid": self.owner.gpu[0],
                            "compute_types": [*document["compute_types"], "int8_float16"]}
            return document if type(document) is bytes else json.dumps(document).encode()
        return json.dumps({
            "schema_version": 1, "language": "en", "language_probability": 0.9,
            "duration_seconds": 2.0, "no_speech_detected": False,
            "segments": [{"start_seconds": 0.0, "end_seconds": 1.0, "text": "tone"},
                         {"start_seconds": 1.0, "end_seconds": 1.5, "text": "burst"}],
            "provenance": {"model_id": request.selection.model,
                           "model_snapshot_id": request.model_snapshot_id, "device": "cpu",
                           "compute_type": "int8", "source_sha256": "5" * 64,
                           "attempt_id": request.attempt_id, "image_digest": None},
            "measurements": {}}).encode()

    async def run(self) -> Any:
        from podmuncher.hardware import AdmittedWorkerRef
        from podmuncher.runtime import RuntimeFailure, WorkerResult

        request = self.request
        try:
            self.reference = AdmittedWorkerRef(
                "desktop-linux", "npipe:////./pipe/docker_engine",
                hashlib.sha256(request.stage_id.encode()).hexdigest(), self.owner.ctx.instance,
                request.job_id, request.attempt_id, self.cuda,
                self.owner.gpu[0] if self.cuda else None)
            self.on_event(self._event(1, "started", {}))
            slot, name = next(iter(request.assigned_slots.items()))
            data = self._document()
            Path(self.mounts.scratch, name).write_bytes(data)
            claim = {"slot": slot, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            if self.owner.tamper is not None:
                claim = self.owner.tamper(claim)
            artifact = self._event(2, "artifact", claim)
            self.on_event(artifact)
            end = time.monotonic() + (self.owner.hold if request.operation == "probe" else 0.0)
            while time.monotonic() < end and not self.obsoleted:
                await asyncio.sleep(0.01)
            if self.obsoleted:
                raise RuntimeFailure(ReasonCode.CANCELLED)
            self.on_event(self._event(3, "completed", {}))
            return WorkerResult(self.reference, True, 0, (artifact,))
        finally:
            self.ended_at = time.monotonic()  # the runtime reaps its container on every exit


class FakeSampler:
    """``HardwareSampler`` stand-in: it drives the real compute-types adapter and stamps freshly."""

    def __init__(self, owner: FakeProbeLaunch, reference: Any, adapter: Any) -> None:
        self.owner, self.reference, self.adapter = owner, reference, adapter

    def sample(self) -> Any:
        from podmuncher.domains import Backend

        owner = self.owner
        owner.samples += 1
        if owner.sample_delay:
            time.sleep(owner.sample_delay)
        cuda = self.reference.gpu_visible
        evidence = self.adapter.compute_types(Backend.CUDA if cuda else Backend.CPU,
                                              deadline=time.monotonic() + 1.0)
        assert evidence.identity == self.reference
        stamp = (datetime.now(UTC) - timedelta(seconds=owner.stale)).strftime(STAMP)
        return hardware_evidence(self.reference, evidence.compute_types, stamp=stamp, ram=owner.ram,
                                 gpu=owner.gpu[1:] if cuda else None)


class FakeProbeLaunch:
    """The probe/smoke launch port without Docker (bootstrap._DockerLaunch's shape)."""

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx
        self.document: Any = dict(PROBE_DOCUMENT)
        self.tamper: Any = None
        self.hold = 0.4
        self.stale = 0.0
        self.sample_delay = 0.0
        self.ram = (4 * GIB, 100 * MIB, 3800 * MIB)
        # (uuid, driver, vram total, vram free): set, the fake worker answers a CUDA probe as a
        # CUDA worker; unset, it is always a CPU worker (a CUDA probe then has no evidence).
        self.gpu: tuple[str, str, int, int] | None = None
        self.fail_image: ReasonCode | None = None
        self.verified: list[str] = []
        self.launches: list[FakeProbeRun] = []
        self.samples = 0
        # Release pulls (``_DockerLaunch.pull_image``): recorded, answered as configured.
        self.pulls: list[tuple[str, Any, float]] = []
        self.pull_backings: list[str | None] = []  # the verified Docker backing per pull
        self.pull_outcome: Any = {"present": True, "pulled": True}
        self.launch_backings: list[str | None] = []  # the verified Docker backing per launch

    async def verify_image(self, image: str) -> None:
        from podmuncher.runtime import RuntimeFailure

        self.verified.append(image)
        if self.fail_image is not None:
            raise RuntimeFailure(self.fail_image)

    async def pull_image(self, image: str, *, growth: Any, timeout: float,
                         manual_backing: str | None = None) -> dict:
        self.pulls.append((image, growth, timeout))
        self.pull_backings.append(manual_backing)
        if isinstance(self.pull_outcome, BaseException):
            raise self.pull_outcome
        return dict(self.pull_outcome)

    async def launch(self, policy: Any, mounts: Any, request: Any, on_event: Any, *,
                     byte_limit: int, timeout: float, growth: Any,
                     manual_backing: str | None = None) -> FakeProbeRun:
        self.launch_backings.append(manual_backing)
        run = FakeProbeRun(self, policy, mounts, request, on_event)
        run.task = asyncio.get_running_loop().create_task(run.run())
        self.launches.append(run)
        return run

    def sampler(self, reference: Any, adapter: Any) -> FakeSampler:
        return FakeSampler(self, reference, adapter)


class FakeAcquire:
    """``ModelCache.acquire`` stand-in: a verified-looking snapshot directory per model."""

    def __init__(self, root: Path) -> None:
        self.root, self.calls = root, []

    def __call__(self, model: Any) -> Any:
        self.calls.append(model.artifact.model_id)
        path = self.root / model.artifact.model_id
        path.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(snapshot_id=SNAPSHOT, path=path)


def make_ctx(env: SimpleNamespace, tmp_path: Path) -> Any:
    """A real ``_Context`` over the fixture manifest and config, without a service."""
    ctx = bootstrap._Context(str(env.roots["state"]), env.env, bootstrap._Clock(), None, None)
    ctx.manifest = bootstrap.load_runtime_manifest(env.env)
    ctx.registries = bootstrap.registries_of(ctx.manifest)
    ctx.config = bootstrap.load_config(env.env, ctx.registries)
    ctx.scratch = bootstrap._ScratchIndex(bootstrap._private_dir(ctx.config.paths.scratch))
    ctx.cache = SimpleNamespace(snapshots=lambda: ())
    ctx.acquire_model = FakeAcquire(tmp_path / "snapshots")
    ctx.instance = "pm-test"
    return ctx


def make_seams(env: SimpleNamespace, **overrides: Any) -> tuple[bootstrap.Seams, dict]:
    made: dict[str, Any] = {}

    def launcher(ctx: Any) -> FakeLauncher:
        made["launcher"] = FakeLauncher(ctx)
        return made["launcher"]

    def probe(ctx: Any) -> FakeProbeLaunch:
        made["probe"] = FakeProbeLaunch(ctx)
        return made["probe"]

    def acquire(ctx: Any) -> FakeAcquire:
        made["acquire"] = FakeAcquire(env.roots["cache"].parent / "fake-snapshots")
        return made["acquire"]

    seams = bootstrap.Seams(docker_storage=FakeDockerStorage(env.roots["state"]),
                            containers=FakeContainers(), launcher=launcher,
                            models=lambda ctx: FakeModels(), hardware=fake_hardware(),
                            probe=probe, acquire=acquire, quiesce_timeout=5.0)
    for name, value in overrides.items():
        setattr(seams, name, value)
    return seams, made


class LoopbackService:
    """The real ``service._serve_async`` (singleton lock, API, runtime file) in a thread."""

    def __init__(self, env: SimpleNamespace, seams: bootstrap.Seams) -> None:
        self.env, self.seams = env, seams
        self.state_dir = str(env.roots["state"])
        self.composition: bootstrap.Composition | None = None
        self.code: int | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> Self:
        module = types.ModuleType(FACTORY_MODULE)

        def make_service(state_dir: str) -> bootstrap.Composition:
            self.composition = bootstrap.Composition(state_dir, environ=self.env.env,
                                                     seams=self.seams)
            return self.composition

        module.make_service = make_service
        sys.modules[FACTORY_MODULE] = module

        def run() -> None:
            self.code = asyncio.run(service._serve_async(self.state_dir,
                                                         f"{FACTORY_MODULE}:make_service", None))

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        deadline = time.monotonic() + 30
        while service.discover(self.state_dir) is None:
            assert self._thread.is_alive(), f"service exited early with {self.code}"
            assert time.monotonic() < deadline, "service did not become ready"
            time.sleep(0.05)
        return self

    def client(self) -> service.ServiceClient:
        client = service.connect(self.state_dir)
        assert client is not None
        return client

    def stop(self) -> int | None:
        if self._thread is not None and self._thread.is_alive():
            client = self.client()
            try:
                client.stop()
            finally:
                client.close()
            self._thread.join(60)
        return self.code

    def __exit__(self, *exc: object) -> None:
        if self._thread is not None and self._thread.is_alive():
            try:
                self.stop()
            except ClientError:
                pass
            self._thread.join(60)
        sys.modules.pop(FACTORY_MODULE, None)


def wait_scheduler_idle(svc: LoopbackService, timeout: float = 20.0) -> None:
    """Wait until the scheduler's run of a just-cancelled job has unwound.

    ``cancel`` returns once the job is final and its lease released, while the run
    coroutine that owned it is still winding down for a few loop steps (its launcher raises
    cancelled, abandon runs, ``run_once`` clears its busy flag).  In that window the
    scheduler is busy and refuses setup apply and retry with a closed ``conflict`` (the
    operator is told to check state and try again).  On the Mac the window outlives the
    test's next calls, since volume sampling there is far quicker than on Windows, so the
    test waits for idleness instead of racing the scheduler.
    """
    scheduler = svc.composition.scheduler
    deadline = time.monotonic() + timeout
    while scheduler._busy:
        assert time.monotonic() < deadline, "the scheduler stayed busy after cancel"
        time.sleep(0.02)


def wait_job(client: service.ServiceClient, job_id: str, states: set[str],
             timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        job = client.get_job(job_id)["job"]
        if job["state"] in states:
            return job
        assert time.monotonic() < deadline, f"job stuck in {job['state']}"
        time.sleep(0.05)


def reason(callback: Any) -> ClientError:
    with pytest.raises(ClientError) as raised:
        callback()
    return raised.value


# ---------------------------------------------------------------------- loopback service


def test_full_loopback_service_drives_every_route(tmp_path: Path) -> None:
    env = make_bootstrap_env(tmp_path)
    seams, made = make_seams(env)
    with LoopbackService(env, seams) as svc:
        client = svc.client()
        launcher: FakeLauncher = made["launcher"]

        identity = client.identity()
        assert identity["pid"] == os.getpid() and identity["service_state"] == "running"

        snapshot = client.snapshot()["snapshot"]
        assert snapshot["service"]["value"] == "running" and snapshot["docker"]["value"] is True
        assert snapshot["host_ram"]["value"] == 8 * 1024**3
        for name in ("worker_ram", "progress", "elapsed_seconds", "stage_eta_seconds",
                     "gpu_utilization", "gpu_vram", "model"):
            assert snapshot[name]["value"] is None, name  # never a fabricated zero
            assert snapshot[name]["status"] != "ok" and snapshot[name]["reason"], name
        assert snapshot["gpu_capability"]["value"] is False
        assert snapshot["storage"] and all(type(e["free"]["value"]) is int for e in snapshot["storage"])
        assert all(e["reserved"]["value"] is None for e in snapshot["storage"])
        assert "volume_identity" not in snapshot["storage"][0]

        # submit -> completed with five transcript exports and the manifest sidecar
        key = str(uuid.uuid4())
        submitted = client.submit(job_request(env), key)
        assert submitted["replayed"] is False
        job_id = submitted["job"]["job_id"]
        assert client.submit(job_request(env), key)["replayed"] is True
        job = wait_job(client, job_id, {"completed", "failed", "partial", "blocked"})
        assert job["state"] == "completed", job
        assert sorted(a["slot"] for a in job["artifacts"] if a["available"]) == TRANSCRIPT_SLOTS
        assert launcher.ops == ["probe", "transcribe"]
        job_dir = env.roots["output"] / f"podmuncher-{job_id}"
        sidecar = json.loads((job_dir / "podmuncher-manifest.json").read_text(encoding="utf-8"))
        assert sorted(a["slot"] for a in sidecar["artifacts"]) == TRANSCRIPT_SLOTS
        assert sidecar["provenance"]["image_digest"] == DIGEST
        assert sidecar["provenance"]["model_revision"] == SNAPSHOT
        assert sidecar["provenance"]["model_id"] == "small"

        listed = client.list_jobs(10)
        assert job_id in [j["job_id"] for j in listed["jobs"]]

        # artifact bytes stream exactly (contained, hashed) and keep the original text
        txt = next(a for a in job["artifacts"] if a["slot"] == "transcript_txt")
        received = bytearray()
        count = client.download_artifact(job_id, txt["artifact_id"], received.extend)
        assert bytes(received) == (job_dir / "transcript.txt").read_bytes()
        assert count == len(received) and HOSTILE.encode() in bytes(received)
        assert reason(lambda: client.download_artifact(job_id, str(uuid.uuid4()),
                                                       lambda _: None)).status == 404

        # recheck: fresh preflight, no attempt
        rechecked = client.recheck(job_id)
        assert rechecked["ready_to_retry"] is True and rechecked["preflight"]["reason_code"] == "ok"
        assert rechecked["job"]["attempt_id"] == job["attempt_id"]

        # retry of a completed job is refused with a closed reason
        assert reason(lambda: client.retry(job_id, str(uuid.uuid4()))).reason_code \
            is ReasonCode.ALREADY_TERMINAL

        # a failed job retries: background attempt, durable replay of the same key
        launcher.fail_once.add("transcribe")
        failed_id = client.submit(job_request(env), str(uuid.uuid4()))["job"]["job_id"]
        failed = wait_job(client, failed_id, {"completed", "failed"})
        assert failed["state"] == "failed" and failed["reason_code"] == "worker_exit_unexpected"
        retry_key = str(uuid.uuid4())
        retried = client.retry(failed_id, retry_key)
        assert retried["replayed"] is False and retried["attempt_id"] != failed["attempt_id"]
        again = client.retry(failed_id, retry_key)
        assert again["replayed"] is True and again["attempt_id"] == retried["attempt_id"]
        assert wait_job(client, failed_id, {"completed", "failed"})["state"] == "completed"

        # clone: a new job, deduplicated per key; the probe proof is re-verified and reused
        clone_key = str(uuid.uuid4())
        cloned = client.clone(job_id, job_request(env, language="en"), clone_key)["job"]
        assert cloned["job_id"] != job_id
        assert client.clone(job_id, job_request(env, language="en"), clone_key)["job"]["job_id"] \
            == cloned["job_id"]
        assert reason(lambda: client.clone(job_id, job_request(env), clone_key)).reason_code \
            is ReasonCode.CONFLICT
        assert wait_job(client, cloned["job_id"], {"completed", "failed"})["state"] == "completed"

        # cancel: terminal -> already_terminal; active -> cancelled after the writer stops
        assert client.cancel(job_id)["status"] == "already_terminal"
        launcher.gates["probe"] = threading.Event()
        launcher.entered.pop("probe", None)
        active_id = client.submit(job_request(env), str(uuid.uuid4()))["job"]["job_id"]
        deadline = time.monotonic() + 20
        while not launcher.entered.get("probe", threading.Event()).is_set():
            assert time.monotonic() < deadline
            time.sleep(0.02)
        cancelled = client.cancel(active_id)
        assert cancelled["status"] == "cancelled" and cancelled["job"]["state"] == "cancelled"
        launcher.gates.clear()
        wait_scheduler_idle(svc)

        # model registry: only manifest-listed models are admitted
        assert reason(lambda: client.submit(job_request(env, model="medium"),
                                            str(uuid.uuid4()))).reason_code is ReasonCode.INVALID_INPUT

        # cache: real ModelCache (empty); removing an unknown entry is a closed 404
        assert client.list_cache()["entries"] == []
        assert reason(lambda: client.remove_cache(SNAPSHOT)).status == 404

        # setup: plan, apply (provisioning without a manifest -> closed failure), status
        plan = client.setup_plan(["media"], {})["plan"]
        assert plan["capabilities"] == ["media"] and plan["plan_id"]
        apply_key = str(uuid.uuid4())
        operation = client.setup_apply(plan["plan_id"], apply_key)["operation"]
        assert client.setup_apply(plan["plan_id"], apply_key)["operation"]["operation_id"] \
            == operation["operation_id"]
        deadline = time.monotonic() + 30
        status = client.setup_status(operation["operation_id"])
        while status["state"] == "running":
            assert time.monotonic() < deadline
            time.sleep(0.05)
            status = client.setup_status(operation["operation_id"])
        assert status["state"] == "failed" and status["reason_code"] == "service_unavailable"
        assert reason(lambda: client.setup_status(str(uuid.uuid4()))).status == 404

        assert client.stop() == {"schema_version": 1, "instance_nonce": client.instance_nonce,
                                 "state": "stopped"}
        client.close()
        svc._thread.join(60)
        assert svc.code == service.EXIT_OK
    state = env.roots["state"]
    assert not (state / "runtime.json").exists()
    audit = (state / "diagnostics" / "audit.jsonl").read_text(encoding="utf-8")
    actions = {json.loads(line)["action"] for line in audit.splitlines()}
    assert {"identity", "jobs_submit", "job_submit", "cancel", "service_stop"} <= actions
    for name in ("audit.jsonl", "logs.jsonl", "traces.jsonl"):
        text = (state / "diagnostics" / name).read_text(encoding="utf-8") \
            if (state / "diagnostics" / name).exists() else ""
        assert str(tmp_path) not in text and str(tmp_path).replace("\\", "\\\\") not in text
        assert "markup" not in text  # no transcript text in telemetry


def _provisioning_without_manifest(ctx: Any) -> Any:
    return bootstrap._Provisioning(replace(ctx, manifest=None), fake_hardware())


@pytest.fixture(autouse=True)
def _no_manifest_provisioning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test service binds setup provisioning without a manifest (never real Docker)."""
    original = bootstrap.Seams

    def seams(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("provisioning", _provisioning_without_manifest)
        return original(*args, **kwargs)

    monkeypatch.setattr(bootstrap, "Seams", seams)


def test_audit_loss_blocks_new_work_but_not_cancel_or_stop(tmp_path: Path) -> None:
    env = make_bootstrap_env(tmp_path)
    seams, made = make_seams(env)
    with LoopbackService(env, seams) as svc:
        client = svc.client()
        launcher: FakeLauncher = made["launcher"]
        launcher.gates["probe"] = threading.Event()
        active = client.submit(job_request(env), str(uuid.uuid4()))["job"]["job_id"]
        deadline = time.monotonic() + 20
        while not launcher.entered.get("probe", threading.Event()).is_set():
            assert time.monotonic() < deadline
            time.sleep(0.02)
        hub = svc.composition.ctx.hub

        def broken(line: bytes) -> None:
            raise OSError("disk gone")

        hub._audit_sink.write = broken  # the durable sink fails: the latch trips on next audit
        error = reason(lambda: client.submit(job_request(env), str(uuid.uuid4())))
        assert error.reason_code is ReasonCode.AUDIT_LOSS and error.status == 503
        assert reason(lambda: client.retry(active, str(uuid.uuid4()))).reason_code \
            is ReasonCode.AUDIT_LOSS
        plan = client.setup_plan(["media"], {})["plan"]
        assert reason(lambda: client.setup_apply(plan["plan_id"], str(uuid.uuid4()))).reason_code \
            is ReasonCode.AUDIT_LOSS
        assert hub.admission_blocked is True
        cancelled = client.cancel(active)
        assert cancelled["status"] == "cancelled"
        assert svc.stop() == service.EXIT_OK


def test_persistence_failure_stops_new_work(tmp_path: Path) -> None:
    env = make_bootstrap_env(tmp_path)
    armed = {"on": False}

    def fault(op: str) -> None:
        if armed["on"] and op == "submit":
            raise sqlite3.OperationalError("disk I/O error")

    seams, _ = make_seams(env, repository_fault_hook=fault)
    with LoopbackService(env, seams) as svc:
        client = svc.client()
        armed["on"] = True
        first = reason(lambda: client.submit(job_request(env), str(uuid.uuid4())))
        assert first.reason_code is ReasonCode.STATE_WRITE_FAILED and first.status == 503
        armed["on"] = False  # the latch holds: no new work until a restart
        again = reason(lambda: client.submit(job_request(env), str(uuid.uuid4())))
        assert again.reason_code is ReasonCode.STATE_WRITE_FAILED
        assert client.identity()["service_state"] == "running"
        assert svc.stop() == service.EXIT_OK


def test_startup_failure_is_closed_and_releases_everything(tmp_path: Path) -> None:
    env = make_bootstrap_env(tmp_path)
    env.manifest.write_text("{not json", encoding="utf-8")
    seams, _ = make_seams(env)
    composition = bootstrap.Composition(env.roots["state"], environ=env.env, seams=seams)
    with pytest.raises(service.ServiceError) as raised:
        asyncio.run(composition.startup())
    assert raised.value.reason_code is ReasonCode.ASSET_VERIFICATION_FAILED
    assert str(env.manifest) not in str(raised.value)
    assert not (env.roots["state"] / bootstrap.DATABASE_NAME).exists()  # failed before state


def test_state_volume_preflight_runs_before_any_state_write(tmp_path: Path) -> None:
    env = make_bootstrap_env(tmp_path)

    class RefusingVolumes:
        def sample(self, path: str, **kwargs: Any) -> Any:
            return SimpleNamespace(state_allowed=SimpleNamespace(value=False))

    seams, _ = make_seams(env, volumes=RefusingVolumes())
    composition = bootstrap.Composition(env.roots["state"], environ=env.env, seams=seams)
    with pytest.raises(service.ServiceError) as raised:
        asyncio.run(composition.startup())
    assert raised.value.reason_code is ReasonCode.INVALID_CONFIG
    assert list(env.roots["state"].iterdir()) == []


def test_audit_action_names_and_cancel_status_mapping(tmp_path: Path) -> None:
    from podmuncher.snapshots import Job

    seen: list = []
    composition = bootstrap.Composition(tmp_path)
    composition.ctx = SimpleNamespace(hub=SimpleNamespace(
        audit=lambda action, **kw: seen.append((action, kw["reason_code"], kw["ids"])) or True))
    for route in ("jobs.submit", "jobs.cancel", "service.stop", "artifacts.get", "unknown"):
        asyncio.run(composition.audit(route, "ok", {"job_id": "1" * 32}))
    assert [a for a, _, _ in seen] == ["jobs_submit", "cancel", "service_stop", "artifacts_get",
                                       "unknown"]
    active = SimpleNamespace(state=JobState.PROBING)
    assert bootstrap.cancel_status(active) == "intent"
    assert bootstrap.cancel_status(SimpleNamespace(state=JobState.CANCELLED)) == "cancelled"
    assert Job  # imported for the public record type the mapping receives


def test_registries_follow_the_explicit_fixture_manifest(tmp_path: Path) -> None:
    env = make_bootstrap_env(tmp_path)
    assert bootstrap.registries_of(None) == (None, None)
    manifest = bootstrap.load_runtime_manifest(env.env)
    assert bootstrap.registries_of(manifest) == (frozenset({"small"}), frozenset({"en"}))
    assert bootstrap.load_runtime_manifest({}) is None  # the fixture loader: never the release
    assert bootstrap.load_runtime_manifest({bootstrap.MANIFEST_ENV: ""}) is None


# ------------------------------------------------------- packaged release manifest (P6-U2)


AMD64, ARM64 = "linux-amd64", "linux-arm64"
RELEASE_PLATFORMS = {"media": (AMD64, ARM64), "asr_cpu": (AMD64, ARM64), "asr_cuda": (AMD64,)}
RELEASE_IMAGE = "ghcr.io/podmuncher-test/podmuncher-{cap}"
MODEL_URL = "https://huggingface.co/Systran/faster-whisper-{model}/resolve/{rev}/{name}"
RELEASE_REVISION = "f" * 40


def fake_digest(label: str) -> str:
    """TEST DATA: a digest-shaped value derived from its label; never a registry digest."""
    return "sha256:" + hashlib.sha256(f"P6-U2 test data, not a registry digest: {label}".encode()).hexdigest()


def release_capability(cap: str, platform: str) -> dict:
    """A capability entry of the packaged_release kind: ``image_reference@digest``."""
    doc = _capability(cap)
    doc.update(platform=platform, platforms=list(RELEASE_PLATFORMS[cap]),
               image_reference=RELEASE_IMAGE.format(cap=cap.replace("_", "-")),
               digest=fake_digest(f"{cap} {platform}"), licenses=["Apache-2.0"],
               provenance=f"synthetic {cap} image for {platform} built by the P6-U2 tests; "
                          "its digest is test data",
               runtime_version="1.2.1", hardware_fingerprint=f"{platform} synthetic worker")
    if cap == "asr_cuda":
        doc["driver_version"] = "580.0"
    return doc


def release_model(model_id: str) -> dict:
    """A real (non-synthetic) pinned model: one HTTPS source per file, under the allowlist."""
    doc = _model(model_id, measured=False)
    doc.update(revision=RELEASE_REVISION, synthetic=False, license="MIT",
               provenance="synthetic release model pinned by the P6-U2 tests",
               source_urls=[MODEL_URL.format(model=model_id, rev=RELEASE_REVISION, name=f"{model_id}.bin")])
    return doc


def release_runtime_manifest(platform: str, *, models: tuple[str, ...] = ("small",),
                             languages: tuple[str, ...] = ("en",)) -> dict:
    caps = [cap for cap, platforms in RELEASE_PLATFORMS.items() if platform in platforms]
    return {
        "schema_version": 1, "release_version": "0.1.0",
        "source": {"kind": "packaged_release", "label": "podmuncher 0.1.0 release",
                   "provenance": f"P6-U2 synthetic release wrapper for {platform}"},
        "platforms": [AMD64, ARM64],
        "capabilities": [release_capability(cap, platform) for cap in caps],
        "models": [release_model(m) for m in models], "languages": list(languages),
    }


def canonical(value: object) -> bytes:
    """P6-U5's canonical JSON (tests/test_release_manifest.py): the hashed form."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def make_release_wrapper(*, publishable: bool = True, blockers: tuple[str, ...] = (),
                         manifests: dict[str, dict] | None = None) -> dict:
    """A synthetic ``release.json`` wrapper (document version 2) around runtime manifests."""
    if manifests is None:
        manifests = {platform: release_runtime_manifest(platform) for platform in (AMD64, ARM64)}
    return {
        "document": "podmuncher-release-manifest", "document_version": 2, "release_version": "0.1.0",
        "release_status": {"publishable": publishable,
                           "blockers": [{"code": code, "capability": None, "resolved_by": "the tests",
                                         "detail": "synthetic blocker"} for code in blockers]},
        "registry": {"publisher_namespace": "podmuncher-test", "images": {}},
        "runtime_manifests": manifests,
        "runtime_manifest_sha256": {p: hashlib.sha256(canonical(m)).hexdigest()
                                    for p, m in manifests.items()},
    }


def wrapper_reader(document: dict) -> Any:
    raw = json.dumps(document).encode("utf-8")
    return lambda: raw


def _bare_env(tmp_path: Path) -> SimpleNamespace:
    """The usual env without ``PODMUNCHER_RUNTIME_MANIFEST``: the packaged release applies."""
    env = make_bootstrap_env(tmp_path)
    env.env.pop(bootstrap.MANIFEST_ENV)
    return env


def test_resolution_prefers_the_explicit_development_manifest(tmp_path: Path) -> None:
    env = make_bootstrap_env(tmp_path)

    def never() -> bytes:
        raise AssertionError("the packaged release is not read when the env names a fixture")

    resolution = bootstrap.resolve_runtime_manifest(env.env, engine_arch=lambda: "amd64", reader=never)
    assert resolution.source == "development" and resolution.manifest.development_fixture is True
    assert resolution.reason is ReasonCode.OK and resolution.next_action == ""
    assert resolution.deferred is False and resolution.blockers == ()
    assert resolution.registries == (frozenset({"small"}), frozenset({"en"}))
    assert bootstrap._LazyRegistry(env.env)() == resolution.registries
    # A fixture that fails still fails loudly (startup is closed on it), never falls back.
    env.manifest.write_text("{not json", encoding="utf-8")
    with pytest.raises(DomainValidationError):
        bootstrap.resolve_runtime_manifest(env.env, reader=never)


def test_packaged_release_in_this_tree_is_not_publishable_and_names_its_blockers() -> None:
    sampled: list[str] = []
    resolution = bootstrap.resolve_runtime_manifest({}, engine_arch=lambda: sampled.append("x") or "amd64")
    assert resolution.manifest is None and resolution.source == "packaged"
    assert resolution.reason is ReasonCode.SERVICE_UNAVAILABLE and resolution.deferred is False
    assert sampled == []  # an unpublishable release never samples the engine
    assert "registry_digest_missing" in resolution.blockers
    assert all(re.fullmatch(r"[a-z][a-z0-9_]*", code) for code in resolution.blockers)  # closed codes
    assert resolution.next_action.startswith("this build has no published images (")
    assert "registry_digest_missing" in resolution.next_action
    assert resolution.next_action.endswith(f"install a release build or set {bootstrap.MANIFEST_ENV}")
    assert resolution.registries == (None, None) and bootstrap._LazyRegistry({})() == (None, None)
    # Every setup step without a manifest repeats that guidance, not the fixture variable.
    ctx = SimpleNamespace(manifest=None, manifest_source=resolution, capability=lambda c: None)
    port = bootstrap._Provisioning(ctx, fake_hardware())
    for result in (port.pull(Capability.MEDIA), port.probe_hardware(), port.choose_model({}),
                   port.model_capacity("small"), port.acquire_model("small"), port.smoke("small")):
        assert result.ok is False and result.reason_code is ReasonCode.SERVICE_UNAVAILABLE
        assert result.next_action == resolution.next_action
        assert bootstrap.MANIFEST_ENV not in result.next_action.split(";")[0]


@pytest.mark.parametrize(("arch", "platform"), [("x86_64", AMD64), ("amd64", AMD64),
                                                 ("aarch64", ARM64), ("ARM64 ", ARM64)])
def test_publishable_release_loads_the_engine_platform_manifest_in_release_mode(
        tmp_path: Path, arch: str, platform: str) -> None:
    from podmuncher.manifests import resolve_capability

    wrapper = make_release_wrapper()
    env = _bare_env(tmp_path)
    resolution = bootstrap.resolve_runtime_manifest(env.env, engine_arch=lambda: arch,
                                                    reader=wrapper_reader(wrapper))
    manifest = resolution.manifest
    assert manifest is not None and resolution.source == "packaged", resolution
    assert resolution.reason is ReasonCode.OK and resolution.next_action == "" and not resolution.deferred
    assert manifest.development_fixture is False
    assert manifest.sha256 == wrapper["runtime_manifest_sha256"][platform]
    assert {c.capability.value for c in manifest.capabilities} == {
        cap for cap, platforms in RELEASE_PLATFORMS.items() if platform in platforms}
    assert all(c.platform == platform for c in manifest.capabilities)
    media = resolve_capability(manifest, "media")
    assert media.worker_policy.image == RELEASE_IMAGE.format(cap="media") + "@" + fake_digest(f"media {platform}")
    assert media.worker_policy.allow_pull and not media.worker_policy.development_image
    # Registries, config and setup sizes follow the selected platform manifest.
    assert resolution.registries == bootstrap.registries_of(manifest) == (frozenset({"small"}), frozenset({"en"}))
    config = bootstrap.load_config(env.env, resolution.registries)
    assert config.paths.cache == str(env.roots["cache"])
    sizes = bootstrap._setup_sizes(manifest)
    assert sizes["media"]["download_bytes"] == 100  # a pulled image: its compressed bytes
    assert sizes["asr_cpu"]["reserve_bytes"] > sizes["media"]["reserve_bytes"]


@pytest.mark.parametrize("mutation", ["sha256", "document", "version", "blocker_code", "blockers",
                                      "missing_platform", "malformed", "duplicate_key", "unloadable"])
def test_tampered_or_unpublishable_wrappers_give_no_manifest_and_guidance(mutation: str) -> None:
    wrapper = make_release_wrapper()
    raw: bytes | None = None
    arch = "amd64"
    expected = ReasonCode.ASSET_VERIFICATION_FAILED
    if mutation == "sha256":
        wrapper["runtime_manifests"][AMD64]["languages"].append("de")  # content changed, hash stale
    elif mutation == "document":
        wrapper["document"] = "podmuncher-other-document"
    elif mutation == "version":
        wrapper["document_version"] = 3
    elif mutation == "blocker_code":
        wrapper = make_release_wrapper(publishable=False, blockers=("Not a closed code!",))
    elif mutation == "blockers":
        wrapper = make_release_wrapper(publishable=False,
                                       blockers=("registry_digest_missing", "gpu_evidence_missing"))
        expected = ReasonCode.SERVICE_UNAVAILABLE
    elif mutation == "missing_platform":
        wrapper = make_release_wrapper(manifests={AMD64: release_runtime_manifest(AMD64)})
        arch, expected = "aarch64", ReasonCode.SERVICE_UNAVAILABLE
    elif mutation == "malformed":
        raw = b"{not json"
    elif mutation == "duplicate_key":
        raw = json.dumps(wrapper).encode().replace(b'"document": ', b'"document": "x", "document": ', 1)
    elif mutation == "unloadable":  # hashes fine, but the release-mode loader refuses a fixture source
        manifest = wrapper["runtime_manifests"][AMD64]
        manifest["source"] = {"kind": "development_fixture", "label": "fixture", "provenance": "fixture"}
        wrapper["runtime_manifest_sha256"][AMD64] = hashlib.sha256(canonical(manifest)).hexdigest()
        expected = ReasonCode.ASSET_SOURCE_REJECTED
    reader = (lambda: raw) if raw is not None else wrapper_reader(wrapper)
    resolution = bootstrap.resolve_runtime_manifest({}, engine_arch=lambda: arch, reader=reader)
    assert resolution.manifest is None and resolution.source == "packaged"
    assert resolution.reason is expected and resolution.deferred is False, resolution
    assert resolution.next_action and bootstrap.MANIFEST_ENV in resolution.next_action
    assert resolution.registries == (None, None)
    if mutation == "blockers":
        assert "(registry_digest_missing, gpu_evidence_missing)" in resolution.next_action
    if mutation == "missing_platform":
        assert ARM64 in resolution.next_action and AMD64 not in resolution.next_action


def test_unknown_engine_architecture_defers_the_packaged_manifest(monkeypatch: pytest.MonkeyPatch) -> None:
    amd = release_runtime_manifest(AMD64, models=("small", "medium"), languages=("en", "de"))
    arm = release_runtime_manifest(ARM64, models=("small",), languages=("en",))
    reader = wrapper_reader(make_release_wrapper(manifests={AMD64: amd, ARM64: arm}))

    def failing() -> str:
        raise RuntimeError("docker info failed")

    for engine in (None, lambda: None, lambda: "Unknown", lambda: "arm/v7", failing):
        resolution = bootstrap.resolve_runtime_manifest({}, engine_arch=engine, reader=reader)
        assert resolution.manifest is None and resolution.deferred is True
        assert resolution.reason is ReasonCode.DOCKER_UNAVAILABLE
        assert resolution.next_action.startswith("start Docker")
        assert resolution.registries == (None, None)
        # The ids every platform build verifies: what the client may name before the engine is known.
        assert resolution.common_registries == (frozenset({"small"}), frozenset({"en"}))
    monkeypatch.setattr(bootstrap, "_read_packaged_release", reader)
    assert bootstrap._LazyRegistry({})() == (frozenset({"small"}), frozenset({"en"}))
    # With the engine known, the platform's own registry applies.
    selected = bootstrap.resolve_runtime_manifest({}, engine_arch="x86_64", reader=reader)
    assert selected.registries == (frozenset({"small", "medium"}), frozenset({"en", "de"}))
    assert bootstrap.resolve_runtime_manifest({}, engine_arch="arm64", reader=reader).registries \
        == (frozenset({"small"}), frozenset({"en"}))


def test_startup_resolves_the_packaged_manifest_for_the_sampled_engine(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bootstrap, "_read_packaged_release", wrapper_reader(make_release_wrapper()))

    async def started(env: SimpleNamespace, hardware: Any) -> Any:
        seams, _ = make_seams(env, hardware=hardware)
        composition = bootstrap.Composition(env.roots["state"], environ=env.env, seams=seams)
        await composition.startup()
        try:
            assert composition.hardware is hardware  # the engine was sampled through the port
            return composition.ctx.manifest_source, composition.ctx.manifest, composition.registries()
        finally:
            await composition.shutdown()

    env = _bare_env(tmp_path / "arm")
    source, manifest, registries = asyncio.run(started(env, identity_host("arm64")))
    assert source.source == "packaged" and manifest is not None and manifest.development_fixture is False
    assert {c.capability.value for c in manifest.capabilities} == {"media", "asr_cpu"}
    assert all(c.platform == ARM64 for c in manifest.capabilities)
    assert registries == (frozenset({"small"}), frozenset({"en"}))
    # An engine that cannot be measured (Docker down): the service starts without a
    # manifest and the setup guidance says to start Docker.
    env = _bare_env(tmp_path / "down")
    source, manifest, registries = asyncio.run(started(env, identity_host(None)))
    assert manifest is None and source.deferred is True and registries == (None, None)
    assert source.reason is ReasonCode.DOCKER_UNAVAILABLE and source.next_action.startswith("start Docker")
    # The explicit fixture still wins over a publishable release.
    env = make_bootstrap_env(tmp_path / "fixture")
    source, manifest, registries = asyncio.run(started(env, identity_host("amd64")))
    assert source.source == "development" and manifest.development_fixture is True


def test_real_launcher_mounts_are_host_assigned_and_fail_closed(tmp_path: Path) -> None:
    from podmuncher.runtime import RuntimeFailure

    env = make_bootstrap_env(tmp_path)
    scratch = bootstrap._ScratchIndex(str(env.roots["scratch"]))
    attempt = str(uuid.uuid4())
    out_dir, in_dir = scratch.new_dir(attempt), scratch.new_dir(attempt)
    scratch.register("transcript_segments-" + "c" * 32 + ".bin", out_dir)
    media = "downloaded_media-" + "d" * 32 + ".bin"
    scratch.register(media, in_dir)
    Path(in_dir, media).write_bytes(b"media")
    job = SimpleNamespace(request=SimpleNamespace(source=SimpleNamespace(
        kind="local", input=str(env.source))))
    ctx = SimpleNamespace(scratch=scratch, cache=None, manifest=None,
                          repo=SimpleNamespace(get_job=lambda job_id: job))
    launcher = bootstrap.DockerWorkerLauncher(ctx, bootstrap._DockerAccess())

    def request(input_basename: str | None) -> Any:
        return SimpleNamespace(assigned_slots={"transcript_segments": "transcript_segments-"
                                               + "c" * 32 + ".bin"},
                               input_basename=input_basename, job_id=str(uuid.uuid4()),
                               model_snapshot_id=None)

    mounts = launcher._mounts(request(media))
    assert mounts.scratch == Path(out_dir) and mounts.input_file == Path(in_dir, media)
    assert launcher._mounts(request("talk.mp4")).input_file == env.source  # the job's own input
    with pytest.raises(RuntimeFailure) as refused:
        launcher._mounts(request("other.mp4"))  # never a worker-chosen host path
    assert refused.value.reason_code is ReasonCode.UNAUTHORIZED
    assert asyncio.run(launcher.stop(ReasonCode.CANCELLED)) is True and launcher.writers_reaped
    assert not launcher.ready("media")
    ctx.capability = lambda capability: None
    ctx.resolved = bootstrap._Context.resolved.__get__(ctx)
    with pytest.raises(RuntimeFailure) as no_manifest:
        asyncio.run(launcher.prepare("media"))  # no manifest: not ready, nothing pulled
    assert no_manifest.value.reason_code is ReasonCode.SERVICE_UNAVAILABLE


def test_real_launcher_budgets_the_storage_watchdog_like_the_probe(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A stage worker's RuntimeStorage gets the shared live sample budget, not runtime's
    0.95 s default (a real sample measures 0.7-1.2 s here and latched storage_unknown)."""
    from podmuncher import runtime
    from podmuncher.requests import JobRequest

    class Constructed(Exception):
        pass

    built: list[dict[str, Any]] = []
    closed: list[bool] = []

    class RecordingStorage:
        def __init__(self, session: Any, volumes: Any, **kwargs: Any) -> None:
            built.append({"session": session, "volumes": volumes, **kwargs})
            raise Constructed  # the launcher's own cleanup must still close the session

    env = make_bootstrap_env(tmp_path)
    ctx = make_ctx(env, tmp_path)
    request = JobRequest.from_dict(job_request(env), verified_models=ctx.registries[0],
                                   verified_languages=ctx.registries[1])
    ctx.repo = SimpleNamespace(get_job=lambda job_id: SimpleNamespace(request=request))
    ctx.volumes = object()
    session = SimpleNamespace(close=lambda: closed.append(True))
    ctx.docker_storage = SimpleNamespace(prepare=lambda **kw: session)
    monkeypatch.setattr(runtime, "RuntimeStorage", RecordingStorage)

    async def cli() -> Any:
        return object()

    launcher = bootstrap.DockerWorkerLauncher(ctx, SimpleNamespace(cli=cli))
    launcher._mounts = lambda request: SimpleNamespace(scratch=tmp_path / "scratch")
    worker = SimpleNamespace(capability="media", job_id=str(uuid.uuid4()), operation="metadata")
    with pytest.raises(Constructed):
        asyncio.run(launcher.run(worker, stage=None, on_event=lambda event: None))
    (storage,) = built
    assert storage["session"] is session and storage["volumes"] is ctx.volumes
    assert storage["sample_budget_seconds"] == bootstrap.WORKER_STORAGE_SAMPLE_SECONDS == 2.5
    assert 0.95 < bootstrap.WORKER_STORAGE_SAMPLE_SECONDS <= 3  # above the default, within the cap
    assert runtime.DEFAULT_SAMPLE_BUDGET_SECONDS == 0.95
    assert closed == [True] and launcher.writers_reaped and not launcher.running


def test_held_cache_root_is_sampled_through_its_parent(tmp_path: Path) -> None:
    seen: list[str] = []
    inner = SimpleNamespace(sample=lambda path, **kw: seen.append(path) or path)
    cache = tmp_path / "cache"
    volumes = bootstrap._HeldRootVolumes(inner, (str(cache),), platform="nt")
    assert volumes.sample(str(cache)) == str(tmp_path)
    assert volumes.sample(str(tmp_path / "scratch"), sqlite_state=True) == str(tmp_path / "scratch")
    assert volumes.sample_many((str(cache),)) == (str(tmp_path),)
    # The detour is for the Windows permission probe only: on POSIX the held cache root is
    # sampled as itself, so a cache that is its own mount point is budgeted on its own
    # filesystem rather than its parent's.
    posix = bootstrap._HeldRootVolumes(inner, (str(cache),), platform="posix")
    assert posix.sample(str(cache)) == str(cache) and posix.sample_many((str(cache),)) == (str(cache),)
    default = bootstrap._HeldRootVolumes(inner, (str(cache),))
    assert default.sample(str(cache)) == (str(tmp_path) if os.name == "nt" else str(cache))


def test_provisioning_without_manifest_is_a_closed_failure(tmp_path: Path) -> None:
    ctx = SimpleNamespace(manifest=None, capability=lambda c: None)
    port = bootstrap._Provisioning(ctx, fake_hardware())
    for result in (port.pull(Capability.MEDIA), port.probe_hardware(), port.choose_model({}),
                   port.model_capacity("small"), port.acquire_model("small"), port.smoke("small")):
        assert result.ok is False and result.reason_code is ReasonCode.SERVICE_UNAVAILABLE
        assert result.next_action  # never a fake success


# ------------------------------------------------------------- release image pulls (P6-U2)


RELEASE_REF = RELEASE_IMAGE.format(cap="media") + "@" + fake_digest("media linux-amd64")
FIXTURE_REF = "fixture-local@" + DIGEST  # the fixture manifest reshaped as a release
DOCKER_ENDPOINT = "npipe:////./pipe/docker_engine"


class FakePullCli:
    """``NativeDockerCLI`` stand-in for release pulls: records commands and owners, answers
    ``pull`` and ``image inspect`` as configured.  Never a real Docker, never a registry."""

    def __init__(self, image: str, *, present: bool = False, pull_code: int = 0,
                 lists: tuple[str, ...] | None = None) -> None:
        from podmuncher.runtime import CommandResult

        self._result = CommandResult
        self.image, self.pull_code = image, pull_code
        self.listed: list[str] = [image] if present else []
        self.after_pull = list(lists) if lists is not None else [image]
        self.hold_pull = False  # the pull stays in flight until the coordinator stops
        self.pulled = 0
        self.calls: list[tuple[tuple, str, float]] = []
        self.bound: list[tuple[str, Any]] = []

    def bind_owner(self, owner: str, coordinator: Any = None) -> None:
        self.bound.append((owner, coordinator))

    async def run(self, *args: str, owner: str, timeout: float = 15, coordinator: Any = None,
                  **kwargs: Any) -> Any:
        self.calls.append((args, owner, timeout))
        if args[0] == "pull":
            if self.hold_pull:
                await coordinator.stopped.wait()
            self.pulled += 1
            if self.pull_code == 0:
                self.listed = list(self.after_pull)
            return self._result(self.pull_code, b"")
        if args[:2] == ("image", "inspect"):
            if not self.listed:
                return self._result(1, b"[]")
            return self._result(0, json.dumps([{"Id": "sha256:" + "c" * 64,
                                                "RepoDigests": self.listed}]).encode())
        raise AssertionError(args)


def pull_guard(reason: ReasonCode = ReasonCode.OK, *, timeout: float = 10.0) -> Any:
    """A real ``WriteGuard`` over a fake storage sampler whose plan carries ``reason``."""
    from podmuncher.admission import AdmissionPlan
    from podmuncher.guards import CancellationCoordinator, StorageSample, WriteGuard
    from podmuncher.snapshots import Measurement

    okay = reason is ReasonCode.OK

    async def sample() -> Any:
        plan = AdmissionPlan(Measurement(True if okay else None, "test",
                                         FieldStatus.OK if okay else FieldStatus.ERROR,
                                         datetime.now(UTC).strftime(STAMP), None if okay else reason),
                             (), ())
        return StorageSample(plan, time.monotonic())

    coordinator = CancellationCoordinator("pull-test", bootstrap._MemoryLease(), stop_grace_seconds=0.2,
                                          reap_timeout_seconds=0.2, callback_timeout_seconds=0.2)
    return WriteGuard(coordinator, sample, byte_limit=0, timeout_seconds=timeout)


def test_release_pull_pulls_then_verifies_the_repo_digest() -> None:
    cli = FakePullCli(RELEASE_REF)
    guard = pull_guard()
    coordinator = guard.coordinator
    result = asyncio.run(bootstrap._guarded_pull(cli, guard, RELEASE_REF))
    assert result == {"present": True, "pulled": True}
    assert [c[0] for c in cli.calls] == [("pull", RELEASE_REF), ("image", "inspect", RELEASE_REF)]
    assert cli.calls[0][2] == guard.timeout_seconds  # the pull gets the guard's whole budget
    assert all(owner == coordinator.operation_id for _, owner, _ in cli.calls)
    assert cli.bound == [(coordinator.operation_id, coordinator)]  # the CLI owner is the coordinator's
    assert [item.binding.target_id for item in coordinator._operations] == ["pull:" + RELEASE_REF]
    assert coordinator.stopped.is_set() and coordinator.reason is ReasonCode.OK
    assert not coordinator.active


def test_release_pull_digest_mismatch_fails_closed() -> None:
    from podmuncher.runtime import RuntimeFailure

    other = RELEASE_IMAGE.format(cap="media") + "@" + fake_digest("some other build")
    cli = FakePullCli(RELEASE_REF, lists=(other,))
    with pytest.raises(RuntimeFailure) as failed:
        asyncio.run(bootstrap._guarded_pull(cli, pull_guard(), RELEASE_REF))
    assert failed.value.reason_code is ReasonCode.ASSET_VERIFICATION_FAILED
    assert cli.pulled == 1
    # The proof is the exact ``repo@digest`` in RepoDigests: an Id alone, a failed command,
    # malformed output or another digest never proves a release reference.
    listed = json.dumps([{"RepoDigests": [RELEASE_REF]}]).encode()
    assert bootstrap._repo_digest_listed(0, listed, RELEASE_REF) is True
    assert bootstrap._repo_digest_listed(1, listed, RELEASE_REF) is False
    assert bootstrap._repo_digest_listed(0, json.dumps([{"Id": RELEASE_REF}]).encode(), RELEASE_REF) is False
    assert bootstrap._repo_digest_listed(0, json.dumps([{"RepoDigests": RELEASE_REF}]).encode(),
                                         RELEASE_REF) is False
    assert bootstrap._repo_digest_listed(0, json.dumps([{"RepoDigests": [other]}]).encode(), RELEASE_REF) is False
    assert bootstrap._repo_digest_listed(0, b"not json", RELEASE_REF) is False
    assert bootstrap._repo_digest_listed(0, b"[]", RELEASE_REF) is False


def test_release_pull_insufficient_storage_blocks_before_pulling() -> None:
    from podmuncher.runtime import RuntimeFailure

    cli = FakePullCli(RELEASE_REF)
    guard = pull_guard(ReasonCode.STORAGE_INSUFFICIENT)
    with pytest.raises(RuntimeFailure) as failed:
        asyncio.run(bootstrap._guarded_pull(cli, guard, RELEASE_REF))
    assert failed.value.reason_code is ReasonCode.STORAGE_INSUFFICIENT
    assert cli.calls == [] and cli.bound == [] and cli.pulled == 0  # admission comes first
    assert guard.coordinator.stopped.is_set()


def test_release_pull_failure_is_a_closed_reason() -> None:
    from podmuncher.runtime import RuntimeFailure

    cli = FakePullCli(RELEASE_REF, pull_code=1)
    with pytest.raises(RuntimeFailure) as failed:
        asyncio.run(bootstrap._guarded_pull(cli, pull_guard(), RELEASE_REF))
    assert failed.value.reason_code is ReasonCode.UPSTREAM_UNAVAILABLE
    assert [c[0] for c in cli.calls] == [("pull", RELEASE_REF)]  # nothing verified after a failed pull


def test_release_pull_cancel_goes_through_the_coordinator() -> None:
    from podmuncher.runtime import RuntimeFailure

    cli = FakePullCli(RELEASE_REF)
    cli.hold_pull = True
    guard = pull_guard()
    coordinator = guard.coordinator

    async def scenario() -> None:
        task = asyncio.get_running_loop().create_task(bootstrap._guarded_pull(cli, guard, RELEASE_REF))
        while cli.pulled == 0 and not cli.calls:
            await asyncio.sleep(0.01)  # the pull is in flight
        assert coordinator.active
        task.cancel()
        with pytest.raises(RuntimeFailure) as failed:
            await task
        # The cancel stops the coordinator, which reaps the registered pull operation; the
        # outcome is a closed reason, never a silent success.
        assert failed.value.reason_code in (ReasonCode.CANCELLED, ReasonCode.WRITER_UNRESOLVED)
        assert coordinator.stopped.is_set() and coordinator.reason is ReasonCode.CANCELLED
        assert [item.binding.target_id for item in coordinator._operations] == ["pull:" + RELEASE_REF]
        assert cli.pulled == 1 and not coordinator.active

    asyncio.run(scenario())
    assert [c[0] for c in cli.calls] == [("pull", RELEASE_REF)]  # no verification after a cancel


def test_setup_pull_reap_counts_only_a_non_negative_cli_exit_as_resolved() -> None:
    """The setup pull's registered operation: a CLI that exited with a status (any
    non-negative one: the daemon reported the pull's end) is resolved; one that ended by
    signal, raised, was cancelled or never reported completion is unresolved, so the
    setup lease stays held (the daemon's pull may still be running)."""
    from podmuncher.runtime import CommandResult, RuntimeFailure

    async def ended(result: Any) -> Any:
        return result

    async def raised() -> Any:
        raise RuntimeFailure(ReasonCode.REQUEST_TIMEOUT)

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        pull = bootstrap._SetupPull("owner", RELEASE_REF)
        assert pull.binding.target_id == "pull:" + RELEASE_REF
        assert await pull.reap(pull.binding) is True  # nothing started: nothing to reap
        for code, resolved in ((0, True), (1, True), (125, True), (-15, False), (-9, False)):
            pull.task = loop.create_task(ended(CommandResult(code, b"")))
            await pull.task
            assert await pull.reap(pull.binding) is resolved, code
        pull.task = loop.create_task(raised())  # the CLI run raised: no end reported
        await asyncio.wait({pull.task})
        assert await pull.reap(pull.binding) is False
        pull.task = loop.create_task(ended(CommandResult(0, b"")))
        pull.task.cancel()
        await asyncio.wait({pull.task})
        assert pull.task.cancelled() and await pull.reap(pull.binding) is False
        pull.task = loop.create_future()  # never reports completion
        assert await pull.reap(pull.binding) is False
        pull.task.cancel()
        with pytest.raises(RuntimeFailure) as foreign:
            await pull.reap(object())  # not this operation's binding
        assert foreign.value.reason_code is ReasonCode.WRITER_UNRESOLVED

    asyncio.run(scenario())


def test_launch_port_skips_a_present_image_and_builds_the_guard_otherwise(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from podmuncher import runtime
    from podmuncher.guards import CancellationCoordinator

    env = make_bootstrap_env(tmp_path)
    ctx = make_ctx(env, tmp_path)
    ctx.volumes = object()
    prepared: list[dict] = []
    closed: list[bool] = []
    session = SimpleNamespace(close=lambda: closed.append(True))
    ctx.docker_storage = SimpleNamespace(prepare=lambda **kw: prepared.append(kw) or session)
    holder = [FakePullCli(RELEASE_REF, present=True)]

    async def cli() -> Any:
        return holder[0]

    launch = bootstrap._DockerLaunch(ctx, SimpleNamespace(cli=cli))
    growth = bootstrap._image_growth(100, 200)
    assert (growth.image_compressed, growth.image_unpacked) == (100, 200)
    assert growth.model_partial == growth.model_unpacked == 0 and growth.source_streams == (0,)
    # Present and verified: one read-only inspect, no storage session, no guard, no pull.
    result = asyncio.run(launch.pull_image(RELEASE_REF, growth=growth, timeout=42.0))
    assert result == {"present": True, "pulled": False}
    assert [c[0] for c in holder[0].calls] == [("image", "inspect", RELEASE_REF)]
    assert holder[0].pulled == 0 and prepared == []
    # Absent: the guard is built like a worker's and handed to the guarded pull.
    built: list[dict] = []
    handed: list[tuple] = []

    class RecordingStorage:
        def __init__(self, session: Any, volumes: Any, **kwargs: Any) -> None:
            built.append({"session": session, "volumes": volumes, **kwargs})

        async def sample(self) -> Any:
            raise AssertionError("the guarded pull is faked here; nothing samples")

    async def fake_pull(cli_: Any, guard: Any, image: str) -> dict:
        handed.append((cli_, guard, image))
        return {"present": True, "pulled": True}

    monkeypatch.setattr(runtime, "RuntimeStorage", RecordingStorage)
    monkeypatch.setattr(bootstrap, "_guarded_pull", fake_pull)
    holder[0] = FakePullCli(RELEASE_REF)
    result = asyncio.run(launch.pull_image(RELEASE_REF, growth=growth, timeout=42.0))
    assert result == {"present": True, "pulled": True}
    assert prepared == [{"manual_backing": ctx.config.paths.docker_backing}] and closed == [True]
    (storage,) = built
    assert storage["session"] is session and storage["volumes"] is ctx.volumes
    assert storage["growth"] is growth and storage["limits"] is ctx.config.limits
    assert storage["sample_budget_seconds"] == bootstrap.WORKER_STORAGE_SAMPLE_SECONDS
    assert storage["paths"] == {"scratch": Path(ctx.config.paths.scratch), "cache": Path(ctx.config.paths.cache),
                                "output": Path(ctx.config.paths.output), "state": Path(ctx.state_dir)}
    (cli_, guard, image), = handed
    assert cli_ is holder[0] and image == RELEASE_REF
    assert guard.timeout_seconds == 42.0 and guard.byte_limit == 0 and guard.sample_timeout_seconds == 2.9
    assert type(guard.coordinator) is CancellationCoordinator and guard.coordinator.operation_id.startswith("pull-")
    assert guard.coordinator.stop_grace_seconds == ctx.config.limits.worker_stop_grace_seconds
    assert [c[0] for c in holder[0].calls] == [("image", "inspect", RELEASE_REF)]  # the pre-admission inspect
    # The admission samples the Docker backing the setup's BACKING step verified, when given.
    holder[0] = FakePullCli(RELEASE_REF)
    asyncio.run(launch.pull_image(RELEASE_REF, growth=growth, timeout=42.0, manual_backing="D:\\docker-data"))
    assert prepared[-1] == {"manual_backing": "D:\\docker-data"}
    asyncio.run(launch.pull_image(RELEASE_REF, growth=growth, timeout=42.0, manual_backing=""))
    assert prepared[-1] == {"manual_backing": ""}  # the wizard's auto discovery, not the configured one


def test_probe_port_pulls_release_images_from_the_manifest_entry(tmp_path: Path) -> None:
    from podmuncher.runtime import RuntimeFailure

    env = make_bootstrap_env(tmp_path, manifest_doc=with_cuda(make_manifest_doc()))
    ctx, launch, probe = _probe(env, tmp_path)
    # A development manifest never pulls.
    with pytest.raises(RuntimeFailure) as development:
        asyncio.run(probe.pull(Capability.MEDIA, timeout=5.0))
    assert development.value.reason_code is ReasonCode.SERVICE_UNAVAILABLE and launch.pulls == []
    release_manifest(ctx, {})
    # The CUDA image is pulled from its manifest entry: no GPU resolution is needed for that.
    result = asyncio.run(probe.pull(Capability.ASR_CUDA, timeout=5.0))
    assert result == {"present": True, "pulled": True}
    (image, growth, timeout), = launch.pulls
    assert image == FIXTURE_REF and timeout == 5.0 and launch.pull_backings == [None]
    assert (growth.image_compressed, growth.image_unpacked) == (100, 200)
    asyncio.run(probe.pull(Capability.MEDIA, timeout=5.0, manual_backing="D:\\docker"))
    assert launch.pull_backings == [None, "D:\\docker"]  # forwarded to the launch port's admission
    launch.pull_outcome = RuntimeFailure(ReasonCode.STORAGE_INSUFFICIENT)
    with pytest.raises(RuntimeFailure) as blocked:
        asyncio.run(probe.pull("media", timeout=5.0))
    assert blocked.value.reason_code is ReasonCode.STORAGE_INSUFFICIENT  # the port's reason, unchanged
    assert probe.writers_reaped and not probe.running


def test_provisioning_pull_release_runs_the_guarded_pull_on_the_loop(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from podmuncher.runtime import RuntimeFailure

    env = make_bootstrap_env(tmp_path)
    env.config.write_text(env.config.read_text(encoding="utf-8") + "download_timeout_seconds = 300\n",
                          encoding="utf-8")
    ctx, launch, probe = _probe(env, tmp_path)
    release_manifest(ctx, {})

    def never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the development inspect path must not run for a release manifest")

    monkeypatch.setattr(bootstrap, "resolve_docker_target", never)
    expectations = [
        (ReasonCode.DOCKER_UNAVAILABLE, "start Docker Desktop"),
        (ReasonCode.STORAGE_INSUFFICIENT, "free space on the Docker disk"),
        (ReasonCode.UPSTREAM_UNAVAILABLE, "check your connection; the image is public"),
        (ReasonCode.ASSET_VERIFICATION_FAILED,
         "the pulled image does not carry the manifest digest; retry the setup"),
        (ReasonCode.WRITER_UNRESOLVED,
         "a Docker pull may still be running; let it finish, restart the service, then retry the setup"),
    ]

    async def scenario() -> None:
        port = bootstrap._Provisioning(ctx, fake_hardware(), probe=probe, loop=asyncio.get_running_loop())
        pulled = await asyncio.to_thread(port.pull, Capability.MEDIA)
        assert pulled.ok is True, pulled
        assert pulled.data == {"capability": "media", "present": True, "pulled": True, "bytes": 100}
        (image, growth, timeout), = launch.pulls
        assert image == FIXTURE_REF and (growth.image_compressed, growth.image_unpacked) == (100, 200)
        assert timeout == 300.0  # the configured download timeout
        assert launch.pull_backings == [ctx.config.paths.docker_backing]  # nothing bound: the configured one
        monkeypatch.setattr(bootstrap, "SETUP_PULL_TIMEOUT_SECONDS", 120.0)
        launch.pull_outcome = {"present": True, "pulled": False}
        again = await asyncio.to_thread(port.pull, Capability.ASR_CPU)
        assert again.ok is True, again
        assert again.data == {"capability": "asr_cpu", "present": True, "pulled": False, "bytes": 0}
        assert launch.pulls[-1][2] == 120.0  # capped
        # The wizard's verified Docker backing drives the pull's admission (bound paths);
        # bound without one, the wizard's auto discovery ("") does, not the configured value.
        port.bind_paths({"cache": ctx.config.paths.cache, "docker_backing": "D:\\docker-data"})
        await asyncio.to_thread(port.pull, Capability.MEDIA)
        assert launch.pull_backings[-1] == "D:\\docker-data"
        port.bind_paths({"cache": ctx.config.paths.cache, "docker_backing": ""})
        await asyncio.to_thread(port.pull, Capability.MEDIA)
        assert launch.pull_backings[-1] == ""
        assert port.writers_reaped is True
        for reason, action in expectations:
            launch.pull_outcome = RuntimeFailure(reason)
            failed = await asyncio.to_thread(port.pull, Capability.MEDIA)
            assert failed.ok is False and failed.reason_code is reason, failed
            assert failed.next_action == action and failed.data == {"capability": "media"}
            # Only a pull the daemon may still be running leaves the port's writers unreaped.
            assert port.writers_reaped is (reason is not ReasonCode.WRITER_UNRESOLVED)
            port._unresolved.clear()
        launch.pull_outcome = RuntimeError("not a closed failure")
        failed = await asyncio.to_thread(port.pull, Capability.MEDIA)
        assert failed.ok is False and failed.reason_code is ReasonCode.SERVICE_UNAVAILABLE
        assert failed.next_action == "retry the setup later"
        # The loop-side wait giving up (the guard's own timeout did not end the pull) is
        # unresolved too: the daemon's pull fate is unknown, and it stays unknown.
        launch.pull_outcome = RuntimeFailure(ReasonCode.REQUEST_TIMEOUT)
        timed_out = await asyncio.to_thread(port.pull, Capability.MEDIA)
        assert timed_out.reason_code is ReasonCode.REQUEST_TIMEOUT and port.writers_reaped is False
        launch.pull_outcome = {"present": True, "pulled": False}
        assert (await asyncio.to_thread(port.pull, Capability.MEDIA)).ok and port.writers_reaped is False

    asyncio.run(scenario())
    # Without the probe port and loop the release pull fails closed (never a bare subprocess).
    unbound = bootstrap._Provisioning(ctx, fake_hardware()).pull(Capability.MEDIA)
    assert unbound.ok is False and unbound.reason_code is ReasonCode.ENVELOPE_UNVALIDATED
    assert unbound.next_action == "restart the service, then retry the setup"


def test_provisioning_pull_development_manifest_is_inspect_only(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = make_bootstrap_env(tmp_path)
    ctx = make_ctx(env, tmp_path)
    ran: list[tuple] = []
    answer = {"Id": DIGEST}
    monkeypatch.setattr(bootstrap, "resolve_docker_target",
                        lambda *a, **k: SimpleNamespace(endpoint=DOCKER_ENDPOINT))

    def run(args: tuple, **kwargs: Any) -> Any:
        ran.append(args)
        return SimpleNamespace(returncode=0, stdout=json.dumps([answer]).encode())

    monkeypatch.setattr(subprocess, "run", run)
    port = bootstrap._Provisioning(ctx, fake_hardware())  # no probe port, no loop: not needed
    result = port.pull(Capability.MEDIA)
    assert result.ok is True and result.data == {"capability": "media", "present": True, "pulled": False}
    answer["Id"] = "sha256:" + "9" * 64
    refused = port.pull(Capability.MEDIA)
    assert refused.ok is False and refused.reason_code is ReasonCode.ASSET_VERIFICATION_FAILED
    assert refused.next_action == "build the local development image named by the manifest"
    assert ran == [("docker", "--host", DOCKER_ENDPOINT, "image", "inspect", DIGEST)] * 2  # never a pull


def test_release_image_presence_is_an_uncached_read_only_inspect(monkeypatch: pytest.MonkeyPatch) -> None:
    from podmuncher.runtime import RuntimeFailure

    ran: list[tuple] = []
    listed: list[str] = []
    monkeypatch.setattr(bootstrap, "resolve_docker_target",
                        lambda *a, **k: SimpleNamespace(endpoint=DOCKER_ENDPOINT))

    def run(args: tuple, **kwargs: Any) -> Any:
        ran.append(args)
        assert kwargs["timeout"] == bootstrap.IMAGE_INSPECT_SECONDS and kwargs["stdin"] is subprocess.DEVNULL
        return SimpleNamespace(returncode=0 if listed else 1,
                               stdout=json.dumps([{"RepoDigests": listed}]).encode())

    monkeypatch.setattr(subprocess, "run", run)
    assert bootstrap._release_image_present(RELEASE_REF) is False
    assert bootstrap._release_image_present(RELEASE_REF) is False and len(ran) == 2  # absence is re-asked
    listed.append(RELEASE_REF)
    assert bootstrap._release_image_present(RELEASE_REF) is True and len(ran) == 3
    assert bootstrap._release_image_present(RELEASE_REF) is True and len(ran) == 4  # presence is re-asked too
    # Pruned after a pull: the next admission sees it gone (nothing is remembered).
    listed.clear()
    assert bootstrap._release_image_present(RELEASE_REF) is False and len(ran) == 5
    assert all(args == ("docker", "--host", DOCKER_ENDPOINT, "image", "inspect", RELEASE_REF) for args in ran)

    def down(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeFailure(ReasonCode.DOCKER_UNAVAILABLE)

    monkeypatch.setattr(bootstrap, "resolve_docker_target", down)
    listed.append(RELEASE_REF)
    assert bootstrap._release_image_present(RELEASE_REF) is False  # Docker down: not verified, no error
    assert not hasattr(bootstrap, "_PRESENT_IMAGES")


def _release_entries(ctx: Any) -> None:
    """Reshape the fixture manifest as a release whose two images are distinct references."""
    capabilities = tuple(replace(c, image_reference=RELEASE_IMAGE.format(cap=c.capability.value.replace("_", "-")))
                         for c in ctx.manifest.capabilities)
    ctx.manifest = replace(ctx.manifest, development_fixture=False, capabilities=capabilities)


def test_growth_budgets_release_images_until_verified_present(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from podmuncher.requests import JobRequest
    from podmuncher.volumes import make_volume_provider

    env = make_bootstrap_env(tmp_path)
    ctx = make_ctx(env, tmp_path)
    request = JobRequest.from_dict(job_request(env), verified_models=ctx.registries[0],
                                   verified_languages=ctx.registries[1])
    caps = bootstrap._capabilities_for(request)
    assert caps == (Capability.MEDIA, Capability.ASR_CPU)
    asked: list[str] = []
    present: set[str] = set()

    def presence(image: str) -> bool:
        asked.append(image)
        return image in present

    def credits(allocated: dict) -> tuple[int, int]:
        return allocated["image_compressed"], allocated["image_unpacked"]

    # Development images: credited in full, nobody is asked.
    growth, allocated = bootstrap._growth(ctx, request, caps, image_presence=presence)
    assert (growth.image_compressed, growth.image_unpacked) == (200, 400)
    assert credits(allocated) == (200, 400) and asked == []
    # Release images not yet pulled: the full growth is budgeted, nothing is credited.
    _release_entries(ctx)
    media = RELEASE_IMAGE.format(cap="media") + "@" + DIGEST
    asr = RELEASE_IMAGE.format(cap="asr-cpu") + "@" + DIGEST
    growth, allocated = bootstrap._growth(ctx, request, caps, image_presence=presence)
    assert (growth.image_compressed, growth.image_unpacked) == (200, 400)
    assert credits(allocated) == (0, 0) and asked == [media, asr]
    # One verified present: its bytes are credited, the other image's are still budgeted.
    present.add(media)
    growth, allocated = bootstrap._growth(ctx, request, caps, image_presence=presence)
    assert (growth.image_compressed, growth.image_unpacked) == (200, 400) and credits(allocated) == (100, 200)
    # Pruned since (``docker image prune``): the next admission budgets its pull again.
    present.discard(media)
    growth, allocated = bootstrap._growth(ctx, request, caps, image_presence=presence)
    assert (growth.image_compressed, growth.image_unpacked) == (200, 400) and credits(allocated) == (0, 0)
    # No presence callable (the launcher, on the loop): nothing asked, nothing credited.
    asked.clear()
    growth, allocated = bootstrap._growth(ctx, request, caps)
    assert credits(allocated) == (0, 0) and asked == []
    # The job preflight asks the real presence check and carries the credit into the plan.
    present.add(media)
    monkeypatch.setattr(bootstrap, "_release_image_present", presence)
    ctx.volumes = make_volume_provider()
    ctx.docker_storage = FakeDockerStorage(env.roots["state"])
    plan = bootstrap._StoragePreflight(ctx).plan(SimpleNamespace(request=request), None)
    credited = {i.asset_key: i.allocated_bytes for i in plan.intervals
                if i.asset_key in ("image_compressed", "image_unpacked") and i.allocated_bytes}
    assert credited == {"image_compressed": 100, "image_unpacked": 200}
    assert {i.asset_key: i.total_bytes for i in plan.intervals if i.asset_key in credited} == {
        "image_compressed": 200, "image_unpacked": 400}
    assert asked == [media, asr]


def test_setup_sizes_charge_no_download_for_a_present_release_image(tmp_path: Path) -> None:
    from podmuncher.setup import _destination_costs

    env = make_bootstrap_env(tmp_path)
    ctx = make_ctx(env, tmp_path)
    present: set[str] = set()
    reserve = ctx.capability("media").worker_overhead_bytes
    # Development images: never a download, whatever the presence check says.
    sizes = bootstrap._setup_sizes(ctx.manifest, lambda image: True)
    assert dict(sizes["media"]) == {"download_bytes": 0, "reserve_bytes": reserve}
    _release_entries(ctx)
    media = RELEASE_IMAGE.format(cap="media") + "@" + DIGEST
    # Without a presence check (the pure sizes) a release image always costs its compressed
    # download plus its unpacked peak on the Docker store (what the pull's guard admits).
    assert bootstrap._setup_sizes(ctx.manifest)["media"] == {"download_bytes": 100, "docker_bytes": 200,
                                                             "reserve_bytes": reserve}
    sizes = bootstrap._setup_sizes(ctx.manifest, lambda image: image in present)
    assert type(sizes["media"]) is bootstrap._ImageSizes and len(sizes["media"]) == 3
    assert dict(sizes["media"]) == {"download_bytes": 100, "docker_bytes": 200, "reserve_bytes": reserve}
    assert sizes["media"].get("download_bytes") == 100 and sizes["media"].get("other") is None
    with pytest.raises(KeyError):
        sizes["media"]["other"]
    costs = _destination_costs([dict(s) for s in sizes.values()])
    assert costs["docker"] == 600  # both images are pulled: PLAN budgets both downloads and unpacks
    # Decided when read (PLAN time): a present image costs nothing more on the Docker disk.
    present.add(media)
    assert dict(sizes["media"])["download_bytes"] == 0 and dict(sizes["media"])["docker_bytes"] == 0
    assert dict(sizes["asr_cpu"])["download_bytes"] == 100 and dict(sizes["asr_cpu"])["docker_bytes"] == 200
    assert _destination_costs([dict(s) for s in sizes.values()])["docker"] == 300
    present.add(RELEASE_IMAGE.format(cap="asr-cpu") + "@" + DIGEST)
    assert "docker" not in _destination_costs([dict(s) for s in sizes.values()])
    assert costs["docker"] == 600  # the copies PLAN took earlier are plain dicts, unchanged


def test_composition_binds_the_presence_check_to_setup_sizes(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bootstrap, "_read_packaged_release", wrapper_reader(make_release_wrapper()))
    env = _bare_env(tmp_path)

    async def started() -> Any:
        seams, _ = make_seams(env, hardware=identity_host("amd64"))
        composition = bootstrap.Composition(env.roots["state"], environ=env.env, seams=seams)
        await composition.startup()
        try:
            assert composition.ctx.manifest.development_fixture is False
            return dict(composition.setup_actions._sizes)
        finally:
            await composition.shutdown()

    sizes = asyncio.run(started())
    assert set(sizes) == {"media", "asr_cpu", "asr_cuda"}
    for capability, size in sizes.items():
        assert type(size) is bootstrap._ImageSizes and size._present is bootstrap._release_image_present
        assert size._image == RELEASE_IMAGE.format(cap=capability.replace("_", "-")) + "@" + fake_digest(
            f"{capability} {AMD64}")


# ------------------------------------------------------ release image verification (P6-U2)


class ReleaseCli:
    """``NativeDockerCLI`` stand-in for a Docker engine holding release images: exactly as
    Docker answers, ``image inspect`` of a present ``repo@digest`` lists it in ``RepoDigests``
    and carries a config ``Id`` that never equals that reference; an absent one fails.  A
    ``pull`` makes the reference present.  Never a real Docker, never a registry."""

    CONFIG_ID = "sha256:" + "c" * 64

    def __init__(self, *present: str) -> None:
        self.listed: set[str] = set(present)
        self.calls: list[tuple] = []
        self.bound: list[tuple[str, Any]] = []

    def bind_owner(self, owner: str, coordinator: Any = None) -> None:
        self.bound.append((owner, coordinator))

    async def run(self, *args: str, owner: str, timeout: float = 15, coordinator: Any = None,
                  **kwargs: Any) -> Any:
        from podmuncher.runtime import CommandResult

        self.calls.append(args)
        if args[:2] == ("image", "inspect"):
            image = args[2]
            if image not in self.listed:
                return CommandResult(1, b"[]")
            # Inspected by a local id (a development image) Docker echoes that id; a pulled
            # reference carries the config digest as its Id and the reference in RepoDigests.
            by_id = "@" not in image
            return CommandResult(0, json.dumps([{"Id": image if by_id else self.CONFIG_ID,
                                                 "RepoDigests": [] if by_id else [image]}]).encode())
        if args[0] == "pull":
            self.listed.add(args[1])
            return CommandResult(0, b"")
        raise AssertionError(args)

    def inspects(self, image: str) -> int:
        return sum(1 for call in self.calls if call == ("image", "inspect", image))


def _access(cli: Any) -> Any:
    async def get() -> Any:
        return cli

    return SimpleNamespace(cli=get)


def test_require_image_proves_a_release_reference_by_repo_digest_and_a_development_image_by_id() -> None:
    """A release reference is ``repo@digest``; Docker's ``Id`` for it is the config digest, so
    an ``Id`` comparison can never pass: presence is the reference in ``RepoDigests``."""
    from podmuncher.runtime import CommandResult, RuntimeFailure

    cli = ReleaseCli(RELEASE_REF)
    asyncio.run(bootstrap._require_image(cli, RELEASE_REF, development=False))  # present: no failure
    assert cli.calls == [("image", "inspect", RELEASE_REF)]
    with pytest.raises(RuntimeFailure) as absent:
        asyncio.run(bootstrap._require_image(cli, RELEASE_REF.replace("media", "asr-cpu"), development=False))
    assert absent.value.reason_code is ReasonCode.SERVICE_UNAVAILABLE
    # The same inspect output read as a development image fails (its Id is not the reference),
    # and a development image is proven by its Id, never by RepoDigests.
    with pytest.raises(RuntimeFailure):
        asyncio.run(bootstrap._require_image(cli, RELEASE_REF, development=True))

    class DevelopmentCli:
        async def run(self, *args: str, **kwargs: Any) -> Any:
            return CommandResult(0, json.dumps([{"Id": DIGEST, "RepoDigests": []}]).encode())

    asyncio.run(bootstrap._require_image(DevelopmentCli(), DIGEST, development=True))
    with pytest.raises(RuntimeFailure):
        asyncio.run(bootstrap._require_image(DevelopmentCli(), DIGEST, development=False))
    assert asyncio.run(bootstrap._image_listed(cli, RELEASE_REF)) is True
    assert asyncio.run(bootstrap._image_listed(cli, "other@" + DIGEST)) is False


def test_launch_port_verifies_images_by_the_manifest_kind(tmp_path: Path) -> None:
    from podmuncher.runtime import RuntimeFailure

    env = make_bootstrap_env(tmp_path)
    ctx = make_ctx(env, tmp_path)
    cli = ReleaseCli(FIXTURE_REF)
    launch = bootstrap._DockerLaunch(ctx, _access(cli))
    # Development manifest: the fixture's local image id must be the Id (never RepoDigests).
    with pytest.raises(RuntimeFailure) as development:
        asyncio.run(launch.verify_image(DIGEST))
    assert development.value.reason_code is ReasonCode.SERVICE_UNAVAILABLE
    # Release manifest: the probe and smoke image is proven by ``RepoDigests``.
    release_manifest(ctx, {})
    image = ctx.resolved(Capability.ASR_CPU).worker_policy.image
    assert image == FIXTURE_REF
    asyncio.run(launch.verify_image(image))
    cli.listed.clear()  # pruned: the probe cannot run on an absent image (closed, no pull)
    with pytest.raises(RuntimeFailure) as pruned:
        asyncio.run(launch.verify_image(image))
    assert pruned.value.reason_code is ReasonCode.SERVICE_UNAVAILABLE
    assert all(call[:2] == ("image", "inspect") for call in cli.calls)  # never a pull here


def test_real_launcher_prepare_and_run_budget_an_absent_release_image(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``prepare`` verifies a release image by ``RepoDigests`` and never fails the job when it
    is absent (pruned after setup): the runtime pulls it under ``allow_pull``, and the run's
    storage guard then budgets that pull (compressed plus unpacked); a present image budgets
    nothing, as a development image never does."""
    from podmuncher import runtime
    from podmuncher.requests import JobRequest
    from podmuncher.runtime import RuntimeFailure

    class Constructed(Exception):
        pass

    built: list[dict[str, Any]] = []

    class RecordingStorage:
        def __init__(self, session: Any, volumes: Any, **kwargs: Any) -> None:
            built.append(kwargs)
            raise Constructed

    env = make_bootstrap_env(tmp_path)
    ctx = make_ctx(env, tmp_path)
    request = JobRequest.from_dict(job_request(env), verified_models=ctx.registries[0],
                                   verified_languages=ctx.registries[1])
    ctx.repo = SimpleNamespace(get_job=lambda job_id: SimpleNamespace(request=request))
    ctx.volumes = object()
    ctx.docker_storage = SimpleNamespace(prepare=lambda **kw: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(runtime, "RuntimeStorage", RecordingStorage)
    cli = ReleaseCli()
    launcher = bootstrap.DockerWorkerLauncher(ctx, _access(cli))
    launcher._mounts = lambda request: SimpleNamespace(scratch=tmp_path / "scratch")
    worker = SimpleNamespace(capability="media", job_id=str(uuid.uuid4()), operation="metadata")

    def growth() -> tuple[int, int]:
        with pytest.raises(Constructed):
            asyncio.run(launcher.run(worker, stage=None, on_event=lambda event: None))
        image = built[-1]["growth"]
        return image.image_compressed, image.image_unpacked

    # Development image: absent is a closed failure at prepare; present budgets nothing.
    with pytest.raises(RuntimeFailure) as missing:
        asyncio.run(launcher.prepare("media"))
    assert missing.value.reason_code is ReasonCode.SERVICE_UNAVAILABLE and not launcher.ready("media")
    cli.listed.add(DIGEST)
    assert cli.calls[-1] == ("image", "inspect", DIGEST)
    asyncio.run(launcher.prepare("media"))
    assert launcher.ready("media") and growth() == (0, 0)
    # Release image, absent: prepare passes (the runtime will pull) and the run budgets the pull.
    release_manifest(ctx, {})
    launcher._ready.clear()
    asyncio.run(launcher.prepare("media"))
    assert launcher.ready("media") and cli.calls[-1] == ("image", "inspect", FIXTURE_REF)
    assert ctx.resolved("media").worker_policy.allow_pull is True
    assert growth() == (100, 200)
    assert built[-1]["growth"].model_partial == built[-1]["growth"].model_unpacked == 0
    # Present (pulled by setup, or by the previous run): nothing can grow for the image.
    cli.listed.add(FIXTURE_REF)
    assert growth() == (0, 0)
    assert all(call[0] != "pull" for call in cli.calls)  # the launcher itself never pulls


class OfflineReleaseCli(ReleaseCli):
    """``ReleaseCli`` plus the container lifecycle the real ``DockerRuntime`` drives, with
    the registry unreachable: every ``pull`` fails (exit 1, recorded) and never makes the
    image present.  The scripted container starts, emits ``started``/``completed`` for the
    run's own request, exits 0 and is removed.  Never a real Docker, never a registry."""

    CID = "e" * 64

    def __init__(self, *present: str) -> None:
        from podmuncher.runtime import DockerTarget

        super().__init__(*present)
        self.target = DockerTarget("desktop-linux", DOCKER_ENDPOINT)
        self.request: Any = None
        self.labels: dict[str, str] = {}
        self.exists = False

    async def run(self, *args: str, owner: str, timeout: float = 15, coordinator: Any = None,
                  stdin: Any = None, on_stdout: Any = None, **kwargs: Any) -> Any:
        from podmuncher.protocol import WorkerEventRecord
        from podmuncher.runtime import CommandResult

        if args[:2] == ("image", "inspect"):
            return await super().run(*args, owner=owner, timeout=timeout, coordinator=coordinator)
        self.calls.append(args)
        if args[0] == "pull":
            return CommandResult(1, b"")  # the registry is unreachable
        if args[0] == "create":
            self.labels = dict(args[i + 1].split("=", 1) for i, v in enumerate(args) if v == "--label")
            self.exists = True
            return CommandResult(0, (self.CID + "\n").encode())
        if args[0] == "ps":
            return CommandResult(0, (self.CID + "\n").encode() if self.exists else b"")
        if args[:2] == ("container", "inspect"):
            doc = [{"Id": self.CID, "Config": {"Labels": dict(self.labels)},
                    "HostConfig": {"DeviceRequests": None},
                    "State": {"Running": False, "Restarting": False}}]
            return CommandResult(0 if self.exists else 1, json.dumps(doc).encode())
        if args[0] == "start":
            request = self.request
            for sequence, kind in enumerate(("started", "completed")):
                record = WorkerEventRecord(1, request.job_id, request.attempt_id, request.stage_id,
                                           sequence, kind, {})
                await on_stdout(record.to_json() + b"\n")
            return CommandResult(0, b"")
        if args[0] == "wait":
            return CommandResult(0, b"0\n")
        if args[0] in ("stop", "kill"):
            return CommandResult(0, b"")
        if args[0] == "rm":
            self.exists = False
            return CommandResult(0, b"")
        raise AssertionError(args)


def test_real_launcher_runs_a_present_release_image_offline_without_a_pull(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Offline replay: the release image setup pulled is present (``RepoDigests``) and the
    registry is unreachable.  The job's worker runs through the real ``DockerRuntime`` and
    completes with no pull attempted: the launcher hands the runtime a policy that does
    not pull (``allow_pull`` False, a read-only presence proof).  Only an absent image
    pulls, and that pull's failure is the closed reason it always was."""
    from podmuncher import runtime
    from podmuncher.admission import AdmissionPlan
    from podmuncher.config import Limits
    from podmuncher.guards import StorageSample
    from podmuncher.protocol import WorkerRequest
    from podmuncher.requests import JobRequest, RequestOutputs, RequestSelection
    from podmuncher.runtime import RuntimeFailure, WorkerMounts
    from podmuncher.snapshots import Measurement

    class AdmittedStorage:
        def __init__(self, session: Any, volumes: Any, **kwargs: Any) -> None:
            pass

        async def sample(self) -> Any:
            plan = AdmissionPlan(Measurement(True, "test", FieldStatus.OK,
                                             datetime.now(UTC).strftime(STAMP), None), (), ())
            return StorageSample(plan, time.monotonic())

    policies: list[Any] = []

    class RecordingRuntime(runtime.DockerRuntime):
        def __init__(self, cli: Any, guard: Any, *, policy: Any, **kwargs: Any) -> None:
            policies.append(policy)
            super().__init__(cli, guard, policy=policy, **kwargs)

    env = make_bootstrap_env(tmp_path)
    ctx = make_ctx(env, tmp_path)
    release_manifest(ctx, {})
    job = JobRequest.from_dict(job_request(env), verified_models=ctx.registries[0],
                               verified_languages=ctx.registries[1])
    ctx.repo = SimpleNamespace(get_job=lambda job_id: SimpleNamespace(request=job))
    ctx.volumes = object()
    ctx.docker_storage = SimpleNamespace(prepare=lambda **kw: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(runtime, "RuntimeStorage", AdmittedStorage)
    monkeypatch.setattr(runtime, "DockerRuntime", RecordingRuntime)
    cli = OfflineReleaseCli(FIXTURE_REF)
    # The runtime's own presence proof reads the same engine (the launcher never calls it
    # itself: the policy's ``cached_proof`` is for the runtime, should it ask).
    monkeypatch.setattr(bootstrap, "_release_image_present", lambda image, timeout=0: image in cli.listed)
    scratch = tmp_path / "worker-scratch"
    scratch.mkdir()
    launcher = bootstrap.DockerWorkerLauncher(ctx, _access(cli))
    launcher._mounts = lambda request: WorkerMounts(scratch, input_file=env.source)  # the job's own input
    policy = ctx.resolved("media").worker_policy
    assert policy.allow_pull is True and policy.development_image is False  # the manifest's

    def worker() -> Any:
        request = WorkerRequest(1, str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4()), "media",
                                "probe", {"probe_metadata": "probe_metadata-" + "a" * 32 + ".bin"},
                                None, RequestOutputs(), env.source.name, None, Limits(),
                                RequestSelection(device="cpu"))
        cli.request = request
        return request

    asyncio.run(launcher.prepare("media"))
    result = asyncio.run(launcher.run(worker(), stage=None, on_event=lambda event: None))
    assert result.completed is True and result.exit_code == 0
    assert all(call[0] != "pull" for call in cli.calls), cli.calls  # offline: nothing pulled
    assert [c[0] for c in cli.calls if c[0] in ("create", "start", "wait", "rm")] == ["create", "start", "wait", "rm"]
    assert launcher.writers_reaped and not launcher.running
    (handed,) = policies
    assert handed.allow_pull is False and handed.development_image is False
    assert handed.image == policy.image == FIXTURE_REF and handed.entrypoint == policy.entrypoint
    assert callable(handed.cached_proof) and handed.cached_proof() is True  # re-verifies presence
    assert (handed.compressed_bytes, handed.unpacked_bytes) == (policy.compressed_bytes, policy.unpacked_bytes)
    # Absent (pruned): the runtime still pulls, and the unreachable registry is the closed
    # failure it always was (``docker_unavailable``, then ``_PullRecovery`` keeps the failed
    # pull's lease: ``writer_unresolved``).  The launcher's rule only removes the pull of a
    # present image.
    cli.listed.discard(FIXTURE_REF)
    with pytest.raises(RuntimeFailure) as failed:
        asyncio.run(launcher.run(worker(), stage=None, on_event=lambda event: None))
    assert failed.value.reason_code in (ReasonCode.DOCKER_UNAVAILABLE, ReasonCode.WRITER_UNRESOLVED)
    assert ("pull", FIXTURE_REF) in cli.calls and policies[-1].allow_pull is True
    assert policies[-1] == policy and handed.cached_proof() is False  # the manifest's, unchanged
    # The rule itself: a development image or a non-pulling policy is unchanged.
    assert bootstrap._no_pull_policy(policy) is not policy
    assert bootstrap._no_pull_policy(handed) is handed
    development = replace(policy, image=DIGEST, development_image=True, allow_pull=False)
    assert bootstrap._no_pull_policy(development) is development


def test_provisioning_release_cache_closes_the_setup_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from podmuncher import model_cache

    env = make_bootstrap_env(tmp_path, model_id="tiny")
    ctx, launch, probe = _probe(env, tmp_path)
    ctx.volumes = HeldAwareVolumes()
    ctx.hub = SimpleNamespace(cache_audit=lambda: "audit", cache_spans=lambda: "spans")
    monkeypatch.setattr(model_cache, "ModelCache", FakeModelCache)
    monkeypatch.setattr(FakeModelCache, "opened", [])
    new_cache = tmp_path / "chosen" / "cache"

    async def scenario() -> None:
        port = bootstrap._Provisioning(ctx, fake_hardware(), probe=probe, loop=asyncio.get_running_loop())
        port.bind_paths({"cache": str(new_cache), "scratch": str(tmp_path / "chosen" / "scratch")})
        port.release_cache()  # nothing open yet: harmless
        acquired = await asyncio.to_thread(port.acquire_model, "tiny")
        assert acquired.ok is True
        bound, = FakeModelCache.opened
        assert not bound.closed
        if os.name == "nt":
            with pytest.raises(OSError):
                ctx.volumes.sample(str(new_cache))  # held: what a re-run PLAN would hit
        port.release_cache()  # what the machine calls when back() crosses below CONFIRM
        assert bound.closed and ctx.volumes.sample(str(new_cache)).free.value > 0
        assert bootstrap._host_plan(ctx, "model_unpacked", str(new_cache), MIB).admitted
        again = await asyncio.to_thread(port.acquire_model, "tiny")  # reopens lazily
        assert again.ok is True and len(FakeModelCache.opened) == 2
        port.close()

    asyncio.run(scenario())


# ------------------------------------------------------------ the release path (P6-U2)


class ReleaseLaunch(FakeProbeLaunch):
    """The fake probe/smoke runs with the real ``_DockerLaunch`` image verification and pull
    admission over a ``ReleaseCli`` (``RepoDigests`` are what the release path checks)."""

    def __init__(self, ctx: Any, cli: ReleaseCli) -> None:
        super().__init__(ctx)
        self.real = bootstrap._DockerLaunch(ctx, _access(cli))
        self.document = {**PROBE_DOCUMENT, "runtime": {"name": "ctranslate2", "version": "1.2.1"}}

    async def verify_image(self, image: str) -> None:
        self.verified.append(image)
        await self.real.verify_image(image)

    async def pull_image(self, image: str, *, growth: Any, timeout: float,
                         manual_backing: str | None = None) -> dict:
        self.pulls.append((image, growth, timeout))
        self.pull_backings.append(manual_backing)
        return await self.real.pull_image(image, growth=growth, timeout=timeout, manual_backing=manual_backing)


class ReleaseLauncher(FakeLauncher):
    """The fake stage worker with the real ``DockerWorkerLauncher.prepare`` over a ``ReleaseCli``."""

    def __init__(self, ctx: Any, cli: ReleaseCli) -> None:
        super().__init__(ctx)
        self._ctx, self._docker, self._ready = ctx, _access(cli), set()

    def ready(self, capability: str) -> bool:
        return bootstrap.DockerWorkerLauncher.ready(self, capability)

    async def prepare(self, capability: str) -> None:
        await bootstrap.DockerWorkerLauncher.prepare(self, capability)
        self.prepared.add(capability)


async def _settled(actions: Any, operation: Mapping[str, Any], timeout: float = 60.0) -> dict:
    deadline = time.monotonic() + timeout
    status = dict(operation)
    while status["state"] == "running":
        assert time.monotonic() < deadline, status
        await asyncio.sleep(0.05)
        status = dict(await actions.setup_status(status["operation_id"]))
    return status


async def _final_job(actions: Any, job_id: str, timeout: float = 60.0) -> Any:
    deadline = time.monotonic() + timeout
    while True:
        job = await actions.get_job(job_id)
        if job.state in (JobState.COMPLETED, JobState.FAILED, JobState.PARTIAL, JobState.BLOCKED):
            return job
        assert time.monotonic() < deadline, job.state
        await asyncio.sleep(0.05)


def _release_seams(env: SimpleNamespace, cli: ReleaseCli, monkeypatch: pytest.MonkeyPatch,
                   **overrides: Any) -> tuple[bootstrap.Seams, dict]:
    """A composition over the synthetic publishable release: real setup provisioning, real
    image verification and pull admission over ``cli``; the guarded pull itself, the worker
    runs and the storage sampler are faked (no Docker, no registry)."""
    from podmuncher import runtime

    monkeypatch.setattr(bootstrap, "_read_packaged_release", wrapper_reader(make_release_wrapper()))
    # Admission's presence check (PLAN sizes, the job preflight) reads the same fake engine.
    monkeypatch.setattr(bootstrap, "_release_image_present", lambda image, timeout=0: image in cli.listed)
    async def never_sampled() -> Any:
        raise AssertionError("the guarded pull is faked here; nothing samples")

    monkeypatch.setattr(runtime, "RuntimeStorage",
                        lambda session, volumes, **kw: SimpleNamespace(sample=never_sampled, **kw))

    async def guarded_pull(cli_: Any, guard: Any, image: str) -> dict:
        pulled = await cli_.run("pull", image, owner=guard.coordinator.operation_id)
        assert pulled.returncode == 0
        return {"present": True, "pulled": True}

    monkeypatch.setattr(bootstrap, "_guarded_pull", guarded_pull)
    made: dict[str, Any] = {}

    def launcher(ctx: Any) -> ReleaseLauncher:
        made["launcher"] = ReleaseLauncher(ctx, cli)
        return made["launcher"]

    def probe(ctx: Any) -> ReleaseLaunch:
        made["probe"] = ReleaseLaunch(ctx, cli)
        return made["probe"]

    seams, inner = make_seams(env, hardware=identity_host("amd64"), provisioning=None, launcher=launcher,
                              probe=probe, **overrides)
    seams.docker_storage.prepare = lambda **kw: SimpleNamespace(close=lambda: None)
    made["inner"] = inner  # make_seams's own records (the acquire seam), filled at startup
    return seams, made


def test_release_path_setup_pulls_probes_and_a_job_prepares(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The release path end to end, with fakes that answer exactly as Docker does: a
    publishable wrapper resolves at startup; setup pulls both images (PULL), probes the ASR
    image (MODEL_SELECT verifies it by ``RepoDigests``), acquires the model and smokes; then
    a job's ``prepare`` verifies the media image the same way, and a pruned image does not
    fail a later job.  This is the test that would have caught the ``Id`` comparison."""
    from podmuncher.requests import JobRequest

    env = _bare_env(tmp_path)
    cli = ReleaseCli()
    seams, made = _release_seams(env, cli, monkeypatch)
    media_ref = RELEASE_IMAGE.format(cap="media") + "@" + fake_digest(f"media {AMD64}")
    asr_ref = RELEASE_IMAGE.format(cap="asr-cpu") + "@" + fake_digest(f"asr_cpu {AMD64}")
    reserve = 256 * 1024**2 + 2 * 10 * 1024**2

    async def scenario() -> None:
        composition = bootstrap.Composition(env.roots["state"], environ=env.env, seams=seams)
        await composition.startup()
        try:
            ctx, actions = composition.ctx, composition.actions
            assert ctx.manifest.development_fixture is False and ctx.manifest_source.source == "packaged"
            assert ctx.resolved("media").worker_policy.image == media_ref
            launch: ReleaseLaunch = made["probe"]
            launcher: ReleaseLauncher = made["launcher"]

            # PLAN before any pull: both images cost their download and unpacked peak.
            plan = await actions.setup_plan([Capability.MEDIA, Capability.ASR_CPU], {})
            assert plan["ready"] is True and plan["capabilities"] == ["asr_cpu", "media"], plan
            assert plan["sizes"]["media"] == {"download_bytes": 100, "docker_bytes": 200, "reserve_bytes": reserve}
            assert plan["sizes"]["asr_cpu"]["download_bytes"] == 100

            # Setup through the service: PULL, HARDWARE, MODEL_SELECT (probe), MODEL_CAPACITY,
            # MODEL_ACQUIRE, SMOKE, SAVE_CONFIG.
            status = await _settled(actions, await actions.setup_apply(plan["plan_id"], "setup-key"))
            assert status["state"] == "done" and status["setup_state"] == "done", status
            assert cli.listed == {asr_ref, media_ref}  # both pulled, by their manifest references
            assert [p[0] for p in launch.pulls] == [asr_ref, media_ref]
            assert launch.pull_backings == [ctx.config.paths.docker_backing] * 2
            # The probe and the smoke verified the ASR image by RepoDigests (its Id differs).
            assert launch.verified == [asr_ref, asr_ref]
            assert [r.request.operation for r in launch.launches] == ["probe", "transcribe"]
            assert cli.inspects(asr_ref) >= 3  # before the pull, the probe, the smoke
            assert made["inner"]["acquire"].calls == ["small", "small"]  # MODEL_ACQUIRE, then the smoke's mount
            assert ctx.repo.lease_holders() == ()  # every writer reaped: the setup lease is released

            # PLAN again: the pulled images cost nothing more on the Docker store.
            plan = await actions.setup_plan([Capability.MEDIA, Capability.ASR_CPU], {})
            assert plan["sizes"]["media"] == {"download_bytes": 0, "docker_bytes": 0, "reserve_bytes": reserve}

            # A job: prepare verifies the media release image by RepoDigests and the job runs.
            request = JobRequest.from_dict(job_request(env), verified_models=ctx.registries[0],
                                           verified_languages=ctx.registries[1])
            job, _replayed = await actions.submit(request, "job-key")
            final = await _final_job(actions, job.job_id)
            assert final.state is JobState.COMPLETED, (final.state, final.reason_code)
            assert launcher.prepared == {"media"} and launcher.ops == ["probe", "transcribe"]
            assert cli.inspects(media_ref) >= 2  # the pull verification and prepare

            # Pruned after setup: the next job still prepares (the runtime would pull).
            cli.listed.discard(media_ref)
            launcher._ready.clear()
            before = len(cli.calls)
            job, _replayed = await actions.submit(request, "job-key-2")
            final = await _final_job(actions, job.job_id)
            assert final.state is JobState.COMPLETED, (final.state, final.reason_code)
            since = cli.calls[before:]
            assert ("image", "inspect", media_ref) in since and all(call[0] != "pull" for call in since)
        finally:
            await composition.shutdown()

    asyncio.run(scenario())


def test_setup_apply_replays_the_same_answers_under_a_pinned_key(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An answers-file re-run with a pinned idempotency key plans afresh (plan ids are per
    request) and applies the new plan under the same key: the same canonical inputs are
    that replay, returning the original operation's outcome (exit 0 twice, nothing
    re-provisioned); the same key with different inputs stays a conflict."""
    from podmuncher import setup
    from podmuncher.api import ApiActionError

    env = _bare_env(tmp_path)
    cli = ReleaseCli()
    seams, made = _release_seams(env, cli, monkeypatch)

    async def scenario() -> None:
        composition = bootstrap.Composition(env.roots["state"], environ=env.env, seams=seams)
        await composition.startup()
        try:
            actions = composition.actions
            launch: ReleaseLaunch = made["probe"]
            plan = await actions.setup_plan([Capability.MEDIA, Capability.ASR_CPU], {})
            first = await _settled(actions, await actions.setup_apply(plan["plan_id"], "setup-key"))
            assert first["state"] == "done" and first["exit_code"] == 0, first
            assert setup._remote_result(first).exit_code == 0
            provisioned = (len(launch.pulls), len(launch.launches), len(cli.calls))
            # The replay: the same answers, a new plan id, the same key.
            again = await actions.setup_plan([Capability.MEDIA, Capability.ASR_CPU], {})
            assert again["plan_id"] != plan["plan_id"]
            replay = await actions.setup_apply(again["plan_id"], "setup-key")
            assert replay["operation_id"] == first["operation_id"] and replay == first, replay
            assert setup._remote_result(replay).exit_code == 0  # what the CLI exits with
            assert (len(launch.pulls), len(launch.launches), len(cli.calls)) == provisioned  # nothing re-run
            # The original plan id replays too (the plan itself has not expired).
            same = await actions.setup_apply(plan["plan_id"], "setup-key")
            assert same["operation_id"] == first["operation_id"]
            # Changed inputs under the same key: a conflict, nothing applied.
            for changed in ((await actions.setup_plan([Capability.MEDIA], {}))["plan_id"],
                            (await actions.setup_plan([Capability.MEDIA, Capability.ASR_CPU],
                                                      {"model": "small"}))["plan_id"],
                            str(uuid.uuid4())):  # an unknown plan proves nothing
                with pytest.raises(ApiActionError) as conflict:
                    await actions.setup_apply(changed, "setup-key")
                assert conflict.value.reason_code is ReasonCode.CONFLICT
            assert (len(launch.pulls), len(launch.launches), len(cli.calls)) == provisioned
            # The same answers under another key is a new operation.
            fresh = await actions.setup_plan([Capability.MEDIA, Capability.ASR_CPU], {})
            second = await _settled(actions, await actions.setup_apply(fresh["plan_id"], "setup-key-2"))
            assert second["operation_id"] != first["operation_id"] and second["state"] == "done", second
            assert len(launch.launches) > provisioned[1]
        finally:
            await composition.shutdown()

    asyncio.run(scenario())


def test_release_setup_keeps_the_lease_after_an_unresolved_pull(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A setup pull the daemon may still be running (``writer_unresolved``) is an unreaped
    writer: the durable setup lease stays held, so no job starts under it and another setup
    is refused, until the service restarts and reconciles (``_PullRecovery``'s rule)."""
    from podmuncher.api import ApiActionError
    from podmuncher.requests import JobRequest
    from podmuncher.runtime import RuntimeFailure

    env = _bare_env(tmp_path)
    cli = ReleaseCli()
    seams, made = _release_seams(env, cli, monkeypatch)

    async def unresolved(cli_: Any, guard: Any, image: str) -> dict:
        raise RuntimeFailure(ReasonCode.WRITER_UNRESOLVED)

    monkeypatch.setattr(bootstrap, "_guarded_pull", unresolved)

    async def scenario() -> None:
        composition = bootstrap.Composition(env.roots["state"], environ=env.env, seams=seams)
        await composition.startup()
        try:
            ctx, actions = composition.ctx, composition.actions
            plan = await actions.setup_plan([Capability.MEDIA], {})
            status = await _settled(actions, await actions.setup_apply(plan["plan_id"], "setup-key"))
            assert status["state"] == "failed" and status["setup_state"] == "blocked", status
            assert status["reason_code"] == "writer_unresolved"
            assert status["next_action"].startswith("a Docker pull may still be running")
            assert composition.setup_actions.provision.writers_reaped is False
            holders = ctx.repo.lease_holders()
            assert len(holders) == 1 and holders[0][1] is True  # held; unreaped until a release proves otherwise
            assert composition.scheduler._setup_unreaped is True
            # Another setup is refused while the pull's lease is held ...
            plan = await actions.setup_plan([Capability.MEDIA], {})
            with pytest.raises(Exception) as refused:  # the scheduler's closed refusal, re-raised
                await actions.setup_apply(plan["plan_id"], "setup-key-2")
            assert not isinstance(refused.value, ApiActionError)
            assert refused.value.reason_code is ReasonCode.CONFLICT
            # ... and a job never starts under it.
            request = JobRequest.from_dict(job_request(env), verified_models=ctx.registries[0],
                                           verified_languages=ctx.registries[1])
            job, _ = await actions.submit(request, "job-key")
            assert await composition.scheduler.run_once() is False
            await asyncio.sleep(0.3)
            assert (await actions.get_job(job.job_id)).state is JobState.QUEUED
            assert made["launcher"].ops == []
            ok, reason = await composition.shutdown()
            assert ok is False and reason == "writer_unresolved"  # the stop reports the unreaped writer
        finally:
            await composition.shutdown()

    asyncio.run(scenario())
    # The reaped/unreaped verdict is the port's and the launcher's, not a constant.
    assert bootstrap._Provisioning(SimpleNamespace(manifest=None), fake_hardware()).writers_reaped is True


def test_setup_apply_admits_on_the_bound_cache_after_releasing_the_port_cache(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The setup lease's admission samples the cache the plan provisions (a bound path), not
    the configured one, and a setup-scoped cache an earlier setup left open there is released
    first; a running setup refuses a second apply before anything is released."""
    from podmuncher.api import ApiActionError

    env = _bare_env(tmp_path)
    cli = ReleaseCli()
    seams, made = _release_seams(env, cli, monkeypatch)
    planned: list[tuple[str, str, int]] = []
    real_plan = bootstrap._host_plan

    def host_plan(ctx: Any, role: str, path: str, total: int, limits: Any = None) -> Any:
        planned.append((role, path, total))
        return real_plan(ctx, role, path, total, limits)

    monkeypatch.setattr(bootstrap, "_host_plan", host_plan)
    chosen = tmp_path / "chosen-cache"
    chosen.mkdir()

    async def scenario() -> None:
        gate = asyncio.Event()  # holds the setup in PULL until the test lets it go

        async def gated_pull(cli_: Any, guard: Any, image: str) -> dict:
            await gate.wait()
            await cli_.run("pull", image, owner=guard.coordinator.operation_id)
            return {"present": True, "pulled": True}

        monkeypatch.setattr(bootstrap, "_guarded_pull", gated_pull)
        composition = bootstrap.Composition(env.roots["state"], environ=env.env, seams=seams)
        await composition.startup()
        try:
            actions, ctx = composition.actions, composition.ctx
            provision = composition.setup_actions.provision
            released: list[int] = []
            inner = provision.release_cache
            provision.release_cache = lambda: released.append(len(planned)) or inner()
            # Both plans first: sampling a path from two threads at once (a plan while the
            # setup's HOST_CHECK samples it) is itself a Windows storage_unknown.
            plan = await actions.setup_plan([Capability.MEDIA], {"paths": {"cache": str(chosen)}})
            again = await actions.setup_plan([Capability.MEDIA], {})
            assert plan["ready"] is True, plan
            operation = await actions.setup_apply(plan["plan_id"], "setup-key")
            assert planned[-1] == ("setup", str(chosen), 0) and released == [len(planned) - 1]
            # The setup is running (held in PULL): a second apply is refused before any
            # release or admission.
            running = await actions.setup_status(operation["operation_id"])
            assert running["state"] == "running", (running.get("setup_state"), running["reason_code"],
                                                   running.get("next_action"))
            with pytest.raises(ApiActionError) as busy:
                await actions.setup_apply(again["plan_id"], "setup-key-2")
            assert busy.value.reason_code is ReasonCode.CONFLICT and len(released) == 1
            gate.set()
            status = await _settled(actions, operation)
            assert status["state"] == "done", (status.get("setup_state"), status["reason_code"],
                                               status.get("next_action"))
            assert ctx.config.paths.cache != str(chosen)
            assert all(path != ctx.config.paths.cache for role, path, _ in planned if role == "setup")
        finally:
            await composition.shutdown()

    asyncio.run(scenario())


# ------------------------------------------------------------- probe evidence (SPEC_GAP-12)


def _probe(env: SimpleNamespace, tmp_path: Path, **settings: Any) -> tuple[Any, FakeProbeLaunch, Any]:
    ctx = make_ctx(env, tmp_path)
    launch = FakeProbeLaunch(ctx)
    for name, value in settings.items():
        setattr(launch, name, value)
    return ctx, launch, bootstrap._ProbeEvidence(ctx, launch, hardware=fake_hardware())


def test_probe_evidence_is_selected_inside_the_hold_and_reaped(tmp_path: Path) -> None:
    from podmuncher.model_policy import (
        ModelChoice,
        PolicyEvidence,
        PolicyRequest,
        select_model,
    )

    env = make_bootstrap_env(tmp_path)
    _ctx, launch, probe = _probe(env, tmp_path)
    seen: dict[str, Any] = {}

    async def scenario() -> Any:
        began = time.monotonic()

        def use(evidence: Any, resolved: Any) -> None:
            run = launch.launches[0]
            seen["held"] = not run.task.done() and run.ended_at is None  # still inside the hold
            seen["used_at"] = time.monotonic() - began
            seen["choice"] = select_model(PolicyRequest(4.0, 4 * GIB, model="small"), evidence,
                                          resolved.envelopes)

        evidence = await probe.collect("asr_cpu", use=use)
        seen["returned_at"] = time.monotonic() - began
        seen["running_after"] = probe.running
        return evidence

    evidence = asyncio.run(scenario())
    assert type(evidence) is PolicyEvidence and len(evidence.workers) == 1
    worker = evidence.workers[0]
    assert worker.runtime.compute_types == ("float32", "int8")
    assert worker.hardware.runtime_compute_types.value == ("float32", "int8")
    assert worker.fingerprint.image_digest == DIGEST
    assert worker.fingerprint.runtime_version == RUNTIME_VERSION
    assert {a.model_id for a in evidence.artifacts} == {"small"}
    assert seen["held"] is True and type(seen["choice"]) is ModelChoice
    assert seen["choice"].identity == worker.identity and seen["used_at"] < launch.hold
    assert seen["returned_at"] >= launch.hold  # the run completed naturally after the hold
    assert launch.verified == [DIGEST] and launch.samples == 1  # fresh at the first sample
    run = launch.launches[0]
    assert run.request.operation == "probe" and run.request.input_basename is None
    assert run.request.model_snapshot_id is None and run.request.selection.device == "cpu"
    assert run.mounts.input_file is None and run.mounts.model is None
    assert Path(run.mounts.scratch).is_relative_to(env.roots["scratch"])
    assert run.ended_at is not None and run.closed and not run.obsoleted  # reaped on success
    assert probe.writers_reaped and seen["running_after"] is False


@pytest.mark.parametrize("case", ["bad_sha", "oversize", "invalid_json", "empty_types",
                                  "bad_version", "wrong_device", "stale", "hold_exceeded"])
def test_probe_failures_are_envelope_unvalidated_and_reaped(tmp_path: Path, case: str) -> None:
    from podmuncher.model_policy import ModelPolicyError

    env = make_bootstrap_env(tmp_path)
    settings: dict[str, Any] = {}
    if case == "bad_sha":
        settings["tamper"] = lambda claim: {**claim, "sha256": "f" * 64}
    elif case == "oversize":
        settings["document"] = json.dumps({**PROBE_DOCUMENT, "pad": "x" * 5000}).encode()
    elif case == "invalid_json":
        settings["document"] = b"{not json"
    elif case == "empty_types":
        settings["document"] = {**PROBE_DOCUMENT, "compute_types": []}
    elif case == "bad_version":
        settings["document"] = {**PROBE_DOCUMENT, "runtime": {"name": "ctranslate2", "version": "9.9.9"}}
    elif case == "wrong_device":
        settings["document"] = {**PROBE_DOCUMENT, "device": "cuda"}
    elif case == "stale":
        settings["stale"] = 5.0  # every sample is older than select_model accepts
    else:
        settings.update(hold=0.0, sample_delay=0.2)  # the hold ends while sampling
    _ctx, launch, probe = _probe(env, tmp_path, **settings)
    used = []

    async def scenario() -> None:
        with pytest.raises(ModelPolicyError) as raised:
            await probe.collect("asr_cpu", use=lambda *a: used.append(a))
        assert raised.value.reason_code is ReasonCode.ENVELOPE_UNVALIDATED
        assert not probe.running and probe.writers_reaped

    asyncio.run(scenario())
    assert used == []  # no selection ran on bad evidence
    run, = launch.launches
    assert run.ended_at is not None and run.closed  # reaped on failure too
    if case == "stale":
        assert launch.samples == 2  # one fresh re-sample while still held, then closed
    if case in ("bad_sha", "oversize", "invalid_json", "empty_types", "bad_version", "wrong_device"):
        assert run.obsoleted  # a failure ends the hold at once


def test_probe_image_and_cancel_failures_keep_their_closed_reason(tmp_path: Path) -> None:
    from podmuncher.model_policy import ModelPolicyError

    env = make_bootstrap_env(tmp_path)
    ctx, launch, probe = _probe(env, tmp_path, fail_image=ReasonCode.SERVICE_UNAVAILABLE)

    async def missing_image() -> None:
        with pytest.raises(ModelPolicyError) as raised:
            await probe.collect("asr_cpu")
        assert raised.value.reason_code is ReasonCode.SERVICE_UNAVAILABLE
        assert launch.launches == []  # nothing started, nothing pulled
        with pytest.raises(ModelPolicyError) as unknown:
            await probe.collect("media")
        assert unknown.value.reason_code is ReasonCode.SERVICE_UNAVAILABLE
        with pytest.raises(ModelPolicyError) as cuda:
            await probe.collect("asr_cuda")  # no asr_cuda manifest entry: never a GPU guess
        assert cuda.value.reason_code is ReasonCode.SERVICE_UNAVAILABLE

    asyncio.run(missing_image())

    ctx, launch, probe = _probe(env, tmp_path, hold=5.0, sample_delay=0.3)
    launcher = bootstrap.DockerWorkerLauncher(ctx, bootstrap._DockerAccess(), probe=probe)

    async def cancelled() -> None:
        task = asyncio.get_running_loop().create_task(probe.collect("asr_cpu"))
        while not launch.launches or launch.launches[0].reference is None:
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.05)
        assert launcher.running  # the probe counts as a running writer for the scheduler
        assert await launcher.stop(ReasonCode.CANCELLED) is True
        with pytest.raises(ModelPolicyError) as raised:
            await task
        assert raised.value.reason_code is ReasonCode.CANCELLED
        run, = launch.launches
        assert run.obsoleted and run.ended_at is not None and run.closed
        assert launcher.writers_reaped and not launcher.running

    asyncio.run(cancelled())


def test_manifest_models_select_defers_and_prepare_probes(tmp_path: Path) -> None:
    from podmuncher import transitions as tr
    from podmuncher.model_policy import ModelChoice, ModelPolicyError
    from podmuncher.requests import JobRequest

    env = make_bootstrap_env(tmp_path)
    ctx, launch, probe = _probe(env, tmp_path)
    models = bootstrap._ManifestModels(ctx, probe)
    request = JobRequest.from_dict(job_request(env), verified_models=ctx.registries[0],
                                   verified_languages=ctx.registries[1])
    job = SimpleNamespace(job_id=str(uuid.uuid4()), attempt_id=str(uuid.uuid4()), request=request)
    token = models.select(job, facts=tr.MediaFacts(True, False, 4.0))
    assert launch.launches == []  # select never probes: it runs synchronously on the loop
    for read in (models.capability, models.selection):
        with pytest.raises(ModelPolicyError) as early:
            read(token)
        assert early.value.reason_code is ReasonCode.CONFLICT  # only valid after prepare
    assert asyncio.run(models.prepare(token)) == SNAPSHOT
    assert type(token.choice) is ModelChoice and token.snapshot == SNAPSHOT
    assert models.capability(token) == "asr_cpu" and models.selection(token) == ("cpu", "small")
    assert ctx.acquire_model.calls == ["small"]
    run, = launch.launches
    assert (run.request.job_id, run.request.attempt_id) == (job.job_id, job.attempt_id)
    with pytest.raises(ModelPolicyError) as again:
        asyncio.run(models.prepare(token))  # one probe per token
    assert again.value.reason_code is ReasonCode.CONFLICT
    with pytest.raises(ModelPolicyError) as unknown:
        asyncio.run(models.prepare(models.select(job, facts=tr.MediaFacts(True, False, 0.0))))
    assert unknown.value.reason_code is ReasonCode.DURATION_UNKNOWN and len(launch.launches) == 1
    with pytest.raises(ModelPolicyError) as oom:
        models.select(job, facts=tr.MediaFacts(True, False, 4.0), lower_than=object())
    assert oom.value.reason_code is ReasonCode.MODEL_OOM
    no_manifest = bootstrap._ManifestModels(replace(ctx, manifest=None), probe)
    with pytest.raises(ModelPolicyError) as closed:
        no_manifest.select(job, facts=tr.MediaFacts(True, False, 4.0))
    assert closed.value.reason_code is ReasonCode.SERVICE_UNAVAILABLE


def _job(ctx: Any, env: SimpleNamespace, *, device: str = "auto", model: str = "small") -> Any:
    from podmuncher.requests import JobRequest

    request = JobRequest.from_dict({**job_request(env), "selection": {
        "device": device, "profile": "balanced", "model": model, "language": "auto"}},
        verified_models=ctx.registries[0], verified_languages=ctx.registries[1])
    return SimpleNamespace(job_id=str(uuid.uuid4()), attempt_id=None, request=request)


def test_auto_device_probes_cuda_first_and_lands_on_cpu_without_a_usable_gpu(tmp_path: Path) -> None:
    from podmuncher import transitions as tr
    from podmuncher.model_policy import ModelPolicyError

    env = make_bootstrap_env(tmp_path, manifest_doc=with_cuda(make_manifest_doc()))
    ctx, launch, probe = _probe(env, tmp_path)
    assert bootstrap._asr_candidates(ctx, "auto") == (Capability.ASR_CUDA, Capability.ASR_CPU)
    assert bootstrap._asr_candidates(ctx, "cpu") == (Capability.ASR_CPU,)
    models = bootstrap._ManifestModels(ctx, probe)
    facts = tr.MediaFacts(True, False, 4.0)

    # No usable host GPU: the CUDA candidate is refused before any worker starts and auto
    # falls back to the CPU probe; an explicit cuda request is refused with gpu_unavailable.
    token = models.select(_job(ctx, env), facts=facts)
    assert asyncio.run(models.prepare(token)) == SNAPSHOT
    assert models.capability(token) == "asr_cpu" and models.selection(token) == ("cpu", "small")
    assert [r.request.capability for r in launch.launches] == ["asr_cpu"]
    with pytest.raises(ModelPolicyError) as refused:
        asyncio.run(models.prepare(models.select(_job(ctx, env, device="cuda"), facts=facts)))
    assert refused.value.reason_code is ReasonCode.GPU_UNAVAILABLE and len(launch.launches) == 1

    # A reported GPU with a UUID (SPEC_GAP-16: one manifest serves both backends): the CUDA
    # candidate is probed first.  This fake worker answers as a CPU runtime, so the CUDA probe
    # yields no CUDA evidence (envelope_unvalidated) and auto falls back to the CPU probe; the
    # job still lands on CPU, never on a guessed GPU.
    uuid_calls: list[int] = []

    def gpu_uuid() -> str:
        uuid_calls.append(1)
        return GPU_UUID

    with_gpu = bootstrap._ProbeEvidence(ctx, launch, hardware=gpu_host(), gpu_uuid=gpu_uuid)
    token = bootstrap._ManifestModels(ctx, with_gpu).select(_job(ctx, env), facts=facts)
    assert asyncio.run(bootstrap._ManifestModels(ctx, with_gpu).prepare(token)) == SNAPSHOT
    assert uuid_calls == [1] and token.resolved.capability is Capability.ASR_CPU
    assert models.selection(token) == ("cpu", "small")
    assert [r.request.capability for r in launch.launches] == ["asr_cpu", "asr_cuda", "asr_cpu"]
    assert all(r.ended_at is not None and r.closed for r in launch.launches)  # every run reaped


def test_engine_architecture_gates_release_capabilities(tmp_path: Path) -> None:
    """SPEC_GAP-17: a release image that does not run on the Docker engine's architecture is
    never probed.  ``asr_cuda`` is linux-amd64 only, so an arm64 engine (Docker Desktop on
    Apple silicon) lands on the CPU image without a CUDA worker, GPU or not."""
    from podmuncher import transitions as tr
    from podmuncher.model_policy import ModelPolicyError

    both = (Capability.ASR_CUDA, Capability.ASR_CPU)
    env = make_bootstrap_env(tmp_path, manifest_doc=with_cuda(make_manifest_doc()))
    ctx, launch, _ = _probe(env, tmp_path)
    release_manifest(ctx, {"asr_cuda": ("linux-amd64",), "asr_cpu": ("linux-amd64", "linux-arm64")})
    launch.gpu = (GPU_UUID, "fixture-driver", 8 * GIB, 6 * GIB)  # the fake worker answers CUDA
    facts = tr.MediaFacts(True, False, 4.0)

    # docker info values and OCI names both map; an unmapped or unmeasured arch gates nothing.
    for arch in ("arm64", "aarch64", "ARM64 "):
        assert bootstrap._supported_candidates(ctx, both, arch) == (Capability.ASR_CPU,)
    for arch in ("amd64", "x86_64", None):
        assert bootstrap._supported_candidates(ctx, both, arch) == both
    assert bootstrap._supported_candidates(ctx, both, "riscv64") == ()

    # arm64 engine with a host GPU reported: asr_cuda is dropped before any GPU check, so
    # only the CPU image is probed and the job lands on CPU.
    arm = bootstrap._ProbeEvidence(ctx, launch, hardware=identity_host("arm64", gpu=True),
                                   gpu_uuid=lambda: GPU_UUID)
    models = bootstrap._ManifestModels(ctx, arm)
    token = models.select(_job(ctx, env), facts=facts)
    assert asyncio.run(models.prepare(token)) == SNAPSHOT
    assert models.capability(token) == "asr_cpu" and models.selection(token) == ("cpu", "small")
    assert [r.request.capability for r in launch.launches] == ["asr_cpu"]
    # An explicit cuda request on that engine: no CUDA image can run, refused as gpu_unavailable.
    with pytest.raises(ModelPolicyError) as refused:
        asyncio.run(models.prepare(models.select(_job(ctx, env, device="cuda"), facts=facts)))
    assert refused.value.reason_code is ReasonCode.GPU_UNAVAILABLE and len(launch.launches) == 1

    # amd64 engine: unchanged, CUDA is probed first and wins with the usable GPU.
    amd = bootstrap._ProbeEvidence(ctx, launch, hardware=identity_host("x86_64", gpu=True),
                                   gpu_uuid=lambda: GPU_UUID)
    token = bootstrap._ManifestModels(ctx, amd).select(_job(ctx, env), facts=facts)
    assert asyncio.run(bootstrap._ManifestModels(ctx, amd).prepare(token)) == SNAPSHOT
    assert token.resolved.capability is Capability.ASR_CUDA
    assert [r.request.capability for r in launch.launches] == ["asr_cpu", "asr_cuda"]

    # Unmeasured engine (docker info failed): nothing is gated; the probe fails closed itself.
    unknown = bootstrap._ProbeEvidence(ctx, launch, hardware=identity_host(None, gpu=True),
                                       gpu_uuid=lambda: GPU_UUID)
    token = bootstrap._ManifestModels(ctx, unknown).select(_job(ctx, env), facts=facts)
    assert asyncio.run(bootstrap._ManifestModels(ctx, unknown).prepare(token)) == SNAPSHOT
    assert [r.request.capability for r in launch.launches] == ["asr_cpu", "asr_cuda", "asr_cuda"]

    # No image at all for the engine (a CPU image published for amd64 only): a closed
    # service_unavailable before any worker starts.
    release_manifest(ctx, {"asr_cpu": ("linux-amd64",)})
    with pytest.raises(ModelPolicyError) as nothing:
        asyncio.run(models.prepare(models.select(_job(ctx, env, device="cpu"), facts=facts)))
    assert nothing.value.reason_code is ReasonCode.SERVICE_UNAVAILABLE and len(launch.launches) == 3
    assert all(r.ended_at is not None and r.closed for r in launch.launches)

    # The development fixture is never gated: its platform is a fixture id, not an arch.
    fixture = make_ctx(env, tmp_path)
    assert fixture.manifest.development_fixture is True
    assert bootstrap._supported_candidates(fixture, both, "arm64") == both


def test_setup_steps_show_host_and_engine_identity(tmp_path: Path) -> None:
    """SPEC_GAP-17 (b): ``probe_hardware`` and ``choose_model`` carry the host OS, arch, CPU
    model and the engine architecture, so the operator sees "Apple M4 (arm64), Docker arm64";
    an unmeasured value is None, never a guess.  On an engine no ASR image runs on, the model
    step is a closed failure that names the architecture (the operator may choose video only)."""
    identity = {"host_os": "Darwin", "host_arch": "arm64", "host_cpu_model": "Apple M4",
                "engine_arch": "arm64"}
    env = make_bootstrap_env(tmp_path, model_id="tiny")
    ctx, launch, _ = _probe(env, tmp_path)
    host = identity_host("arm64")

    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        port = bootstrap._Provisioning(ctx, host, probe=bootstrap._ProbeEvidence(ctx, launch, hardware=host),
                                       loop=loop)
        probed = await asyncio.to_thread(port.probe_hardware)
        assert probed.ok is True and {k: probed.data[k] for k in identity} == identity
        assert probed.data["host_cpu_cores"] is None  # unmeasured by this sampler: no guess
        chosen = await asyncio.to_thread(port.choose_model, probed.data)
        assert chosen.ok is True and chosen.data["choice"] == "tiny", chosen
        assert {k: chosen.data[k] for k in identity} == identity
        assert chosen.data["skipped_capabilities"] == []  # a fixture manifest is never gated
        assert json.loads(json.dumps(chosen.data)) == chosen.data
        assert [r.request.capability for r in launch.launches] == ["asr_cpu"]

        # The hardware step without the identity readings: every key present, all None.
        bare = bootstrap._Provisioning(ctx, fake_hardware(),
                                       probe=bootstrap._ProbeEvidence(ctx, launch, hardware=fake_hardware()),
                                       loop=loop)
        plain = await asyncio.to_thread(bare.probe_hardware)
        assert plain.ok is True and {k: plain.data[k] for k in identity} == dict.fromkeys(identity)
        chosen = await asyncio.to_thread(bare.choose_model, {})
        assert chosen.ok is True and {k: chosen.data[k] for k in identity} == dict.fromkeys(identity)

        # A release whose only ASR image is amd64, on an arm64 engine: the probe measured the
        # architecture itself, no worker starts, and the guidance names the engine.
        release_manifest(ctx, {"asr_cpu": ("linux-amd64",)})
        refused = await asyncio.to_thread(port.choose_model, probed.data)
        assert refused.ok is False and refused.reason_code is ReasonCode.SERVICE_UNAVAILABLE
        assert refused.data["engine_arch"] == "arm64" and refused.data["skipped_capabilities"] == ["asr_cpu"]
        assert "arm64" in refused.next_action and "video" in refused.next_action
        assert len(launch.launches) == 2

    asyncio.run(scenario())


def test_usable_gpu_picks_cuda_with_the_largest_model_that_fits_the_table(tmp_path: Path) -> None:
    from podmuncher import transitions as tr

    doc = with_cuda(make_manifest_doc(models=("tiny", "base", "small", "medium", "large-v3")))
    env = make_bootstrap_env(tmp_path, manifest_doc=doc)
    ctx, launch, _ = _probe(env, tmp_path)
    # 8 GiB GPU with 4 GiB free: the table's CUDA rows need large-v3 2.953 GB, medium 3.0 GB,
    # small 1.5 GB plus the 20% (1.6 GiB) headroom, so small is the largest fit.
    launch.gpu = (GPU_UUID, "fixture-driver", 8 * GIB, 4 * GIB)
    probe = bootstrap._ProbeEvidence(ctx, launch, hardware=gpu_host(8 * GIB, 4 * GIB),
                                     gpu_uuid=lambda: GPU_UUID)
    models = bootstrap._ManifestModels(ctx, probe)
    token = models.select(_job(ctx, env, model="auto"), facts=tr.MediaFacts(True, False, 4.0))
    assert asyncio.run(models.prepare(token)) == SNAPSHOT
    assert models.capability(token) == "asr_cuda" and models.selection(token) == ("cuda", "small")
    choice = token.choice
    assert choice.backend.value == "cuda" and choice.compute_type == "int8_float16"
    assert choice.reason == "auto_largest_fit" and choice.basis == "estimated"
    assert choice.peak_vram_bytes == 1_500_000_000 and choice.measured_envelope is None
    assert choice.identity.gpu_uuid == GPU_UUID and choice.identity.gpu_visible is True
    assert [r.request.capability for r in launch.launches] == ["asr_cuda"]  # no CPU fallback
    assert ctx.acquire_model.calls == ["small"]
    # More free VRAM: the largest model by rank, large-v3, fits and wins.
    launch.gpu = (GPU_UUID, "fixture-driver", 8 * GIB, 6 * GIB)
    roomy = bootstrap._ProbeEvidence(ctx, launch, hardware=gpu_host(8 * GIB, 6 * GIB),
                                     gpu_uuid=lambda: GPU_UUID)
    token = bootstrap._ManifestModels(ctx, roomy).select(_job(ctx, env, model="auto"),
                                                         facts=tr.MediaFacts(True, False, 4.0))
    assert asyncio.run(bootstrap._ManifestModels(ctx, roomy).prepare(token)) == SNAPSHOT
    assert models.selection(token) == ("cuda", "large-v3") and token.choice.basis == "published"


def test_explicit_model_is_honoured_through_bootstrap_without_a_fit_check(tmp_path: Path) -> None:
    from podmuncher import transitions as tr

    # large-v3 is unmeasured; its table CPU figure (4.43 GB, estimated) exceeds this 4 GiB
    # worker, so auto never picks it.  An explicit request still gets exactly that model.
    doc = make_manifest_doc(models=("tiny", "large-v3"), unmeasured=("large-v3",))
    env = make_bootstrap_env(tmp_path, manifest_doc=doc)
    ctx, launch, probe = _probe(env, tmp_path)
    models = bootstrap._ManifestModels(ctx, probe)
    facts = tr.MediaFacts(True, False, 4.0)
    auto = models.select(_job(ctx, env, model="auto"), facts=facts)
    assert asyncio.run(models.prepare(auto)) == SNAPSHOT and models.selection(auto) == ("cpu", "tiny")
    explicit = models.select(_job(ctx, env, model="large-v3"), facts=facts)
    assert asyncio.run(models.prepare(explicit)) == SNAPSHOT
    assert models.selection(explicit) == ("cpu", "large-v3") and models.capability(explicit) == "asr_cpu"
    choice = explicit.choice
    assert choice.reason == "explicit_model" and choice.basis == "explicit"
    assert choice.peak_ram_bytes == 4_429_500_000  # the table figure, reported but not enforced
    assert choice.required_ram_bytes > choice.worker_memory_cap_bytes  # would not have fitted
    assert ctx.acquire_model.calls == ["tiny", "large-v3"]
    assert all(r.ended_at is not None and r.closed for r in launch.launches)


def test_job_completes_through_real_model_selection(tmp_path: Path) -> None:
    env = make_bootstrap_env(tmp_path)
    seams, made = make_seams(env, models=None)  # the real _ManifestModels over the fake seam
    with LoopbackService(env, seams) as svc:
        client = svc.client()
        job_id = client.submit(job_request(env), str(uuid.uuid4()))["job"]["job_id"]
        job = wait_job(client, job_id, {"completed", "failed", "partial", "blocked"})
        assert job["state"] == "completed", job
        launcher: FakeLauncher = made["launcher"]
        probe: FakeProbeLaunch = made["probe"]
        assert launcher.ops == ["probe", "transcribe"] and launcher.selections == [("cpu", "small")]
        run, = probe.launches
        assert run.request.operation == "probe" and run.request.job_id == job_id
        assert run.request.attempt_id == job["attempt_id"]
        assert run.ended_at is not None and run.closed and not run.obsoleted
        assert made["acquire"].calls == ["small"]
        assert svc.composition.ctx.models.prepared(job_id) == (SNAPSHOT, "asr_cpu")
        job_dir = env.roots["output"] / f"podmuncher-{job_id}"
        sidecar = json.loads((job_dir / "podmuncher-manifest.json").read_text(encoding="utf-8"))
        assert sidecar["provenance"]["model_revision"] == SNAPSHOT
        assert sidecar["provenance"]["image_digest"] == DIGEST
        assert svc.stop() == service.EXIT_OK
    state = env.roots["state"]
    for name in ("audit.jsonl", "logs.jsonl", "traces.jsonl"):
        path = state / "diagnostics" / name
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        assert str(tmp_path) not in text and str(tmp_path).replace("\\", "\\\\") not in text


def test_oom_fallback_honours_lower_than(tmp_path: Path) -> None:
    doc = make_manifest_doc(models=("base", "tiny"), worker_memory_bytes=6 * GIB)
    env = make_bootstrap_env(tmp_path, manifest_doc=doc)
    seams, made = make_seams(env, models=None)
    with LoopbackService(env, seams) as svc:
        client = svc.client()
        probe: FakeProbeLaunch = made["probe"]
        probe.ram = (8 * GIB, 100 * MIB, 7500 * MIB)
        launcher: FakeLauncher = made["launcher"]
        launcher.oom_once.add("transcribe")
        job_id = client.submit(job_request(env, model="auto"), str(uuid.uuid4()))["job"]["job_id"]
        job = wait_job(client, job_id, {"completed", "failed", "partial", "blocked"})
        assert job["state"] == "completed", job
        # Automatic choice: base first (balanced threshold); after the OOM a strictly lower tier.
        assert launcher.selections == [("cpu", "base"), ("cpu", "tiny")]
        assert [r.request.operation for r in probe.launches] == ["probe", "probe"]
        assert all(r.ended_at is not None and r.closed for r in probe.launches)
        assert made["acquire"].calls == ["base", "tiny"]
        assert svc.stop() == service.EXIT_OK


# ------------------------------------------------- P6-U2-a4: publication, reuse, acquisition


def remote_request(env: SimpleNamespace) -> dict:
    """A remote combined job (video + transcript) whose output is on the scratch volume."""
    request = job_request(env)
    request["source"] = {"kind": "direct", "input": "https://example.com/talk.mp4"}
    request["outputs"] = {"transcript": True, "video": True, "video_quality": "1080p"}
    return request


def test_remote_combined_job_publishes_the_video_by_copy_and_keeps_the_asr_input(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = make_bootstrap_env(tmp_path)
    seams, made = make_seams(env)
    guards: list[tuple[str, int]] = []
    started = bootstrap._started_guard

    async def started_guard(ctx: Any, path: str, total: int, *, timeout: float) -> Any:
        guards.append((path, total))
        return await started(ctx, path, total, timeout=timeout)

    monkeypatch.setattr(bootstrap, "_started_guard", started_guard)
    with LoopbackService(env, seams) as svc:
        client = svc.client()
        launcher: FakeLauncher = made["launcher"]
        launcher.has_video = True
        job_id = client.submit(remote_request(env), str(uuid.uuid4()))["job"]["job_id"]
        job = wait_job(client, job_id, {"completed", "failed", "partial", "blocked"})
        assert job["state"] == "completed", job
        assert launcher.ops == ["download", "probe", "transcribe"]
        # The scheduler launches ASR with the scratch download as input: it still exists then.
        assert [(op, exists) for op, _, exists in launcher.inputs] == [("probe", True),
                                                                      ("transcribe", True)]
        assert sorted(a["slot"] for a in job["artifacts"] if a["available"]) \
            == sorted(TRANSCRIPT_SLOTS + ["original_video"])
        job_dir = env.roots["output"] / f"podmuncher-{job_id}"
        video = job_dir / "video.mp4"
        assert video.read_bytes().startswith(b"\x00\x00\x00\x18ftypmp42 downloaded media ")
        # Same volume (both under tmp_path), published by copy: the scratch download remains
        # under the attempt's private scratch and the copy went through the admitted guard.
        scratch_root = Path(bootstrap.attempt_scratch(str(env.roots["scratch"]), job["attempt_id"]))
        _, basename, _ = launcher.inputs[-1]
        download = next(p for p in scratch_root.rglob(basename))
        assert download.read_bytes() == video.read_bytes()
        assert guards == [(str(env.roots["output"]), len(video.read_bytes()))]
        sidecar = json.loads((job_dir / "podmuncher-manifest.json").read_text(encoding="utf-8"))
        assert sorted(a["slot"] for a in sidecar["artifacts"]) \
            == sorted(TRANSCRIPT_SLOTS + ["original_video"])
        # The copy lives inside the attempt root the existing discard path removes.
        publisher = svc.composition.scheduler._publisher
        publisher.discard(SimpleNamespace(job_id=job_id), SimpleNamespace(attempt_id=job["attempt_id"]))
        assert not scratch_root.exists() and video.is_file()
        assert svc.stop() == service.EXIT_OK


def test_second_export_only_retry_still_never_transcribes(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from podmuncher import artifacts

    env = make_bootstrap_env(tmp_path)
    seams, made = make_seams(env)
    failing: set[str] = {"transcript_srt"}
    real_publish = artifacts.publish

    async def publish(verified: Any, final_slot: Any, **kwargs: Any) -> Any:
        if ArtifactSlot.parse(final_slot).value in failing:
            raise artifacts.ArtifactError(ReasonCode.PUBLICATION_FAILED)
        return await real_publish(verified, final_slot, **kwargs)

    monkeypatch.setattr(artifacts, "publish", publish)
    with LoopbackService(env, seams) as svc:
        client = svc.client()
        launcher: FakeLauncher = made["launcher"]
        job_id = client.submit(job_request(env), str(uuid.uuid4()))["job"]["job_id"]
        first = wait_job(client, job_id, {"completed", "failed", "partial", "blocked"})
        assert first["state"] == "partial" and launcher.ops == ["probe", "transcribe"], \
            (first["state"], first["reason_code"], launcher.ops, first["artifacts"])
        assert len([a for a in first["artifacts"] if a["available"]]) == 4

        # First export-only retry: the retained transcript is re-verified; the srt still fails.
        retried = client.retry(job_id, str(uuid.uuid4()))
        second = wait_job(client, job_id, {"completed", "failed", "partial", "blocked"})
        assert second["state"] == "partial" and second["attempt_id"] == retried["attempt_id"]
        assert launcher.ops == ["probe", "transcribe"]

        # Second export-only retry: the adopted transcribing record carries the new attempt
        # id; the publisher resolves the retained outputs by identity and ASR never runs.
        failing.clear()
        retried = client.retry(job_id, str(uuid.uuid4()))
        third = wait_job(client, job_id, {"completed", "failed", "partial", "blocked"})
        assert third["state"] == "completed" and third["attempt_id"] == retried["attempt_id"]
        assert launcher.ops == ["probe", "transcribe"]
        assert sorted(a["slot"] for a in third["artifacts"] if a["available"]) == TRANSCRIPT_SLOTS
        repo = svc.composition.ctx.repo
        attempts = [repo.get_attempt(a) for a in (first["attempt_id"], second["attempt_id"],
                                                  third["attempt_id"])]
        assert [a.ordinal for a in attempts] == [1, 2, 3]
        for attempt in attempts[1:]:
            assert [r.stage.value for r in attempt.stage_records] \
                == ["checking", "transcribing", "exporting"]
        transcribing = [next(r for r in a.stage_records if r.stage.value == "transcribing")
                        for a in attempts]
        assert {r.settings_hash for r in transcribing} == {transcribing[0].settings_hash}
        assert {r.verified_input_hashes for r in transcribing} == {transcribing[0].verified_input_hashes}
        records = env.roots["state"] / bootstrap.VERIFIED_DIRNAME
        assert (records / f"{first['attempt_id']}.json").is_file()
        assert not (records / f"{third['attempt_id']}.json").exists()
        index = json.loads((records / f"job-{job_id}.json").read_text(encoding="utf-8"))
        assert list(index.values()) == [first["attempt_id"]]
        assert svc.stop() == service.EXIT_OK


class CountingVolumes:
    """A volume provider over the real one that counts samples and can shrink or fail."""

    def __init__(self) -> None:
        from podmuncher.volumes import make_volume_provider

        self.inner = make_volume_provider()
        self.samples = 0
        self.free: int | None = None
        self.error: Exception | None = None

    def sample(self, path: Any, **kwargs: Any) -> Any:
        self.samples += 1
        if self.error is not None:
            raise self.error
        evidence = self.inner.sample(path, **kwargs)
        if self.free is None:
            return evidence
        return replace(evidence, free=replace(evidence.free, value=self.free))


def test_model_acquisition_admission_samples_per_8_mib_and_fails_closed(tmp_path: Path) -> None:
    env = make_bootstrap_env(tmp_path)
    ctx = make_ctx(env, tmp_path)
    ctx.volumes = volumes = CountingVolumes()
    clock = [100.0]
    admission = bootstrap._ModelAdmission(ctx, monotonic=lambda: clock[0])
    total, chunk = 78 * 1000 * 1000, 16 * 1024
    calls = 0
    for allocated in range(0, total + chunk, chunk):
        calls += 1
        assert admission(asset_id="snap", declared_bytes=total, allocated_bytes=min(allocated, total))
    plans = volumes.samples // 2  # one host plan is two samples (cache volume + state volume)
    assert calls >= 4700 and plans <= -(-total // (8 * MIB)) + 2, (calls, plans)
    assert volumes.samples % 2 == 0

    # Time rule: a plan older than a second is re-sampled even without growth.
    before = volumes.samples
    assert admission(asset_id="snap", declared_bytes=total, allocated_bytes=total)
    assert volumes.samples == before
    clock[0] += 1.0
    assert admission(asset_id="snap", declared_bytes=total, allocated_bytes=total)
    assert volumes.samples == before + 2

    # Projected bytes beyond the measured allowance refuse without a new sample.
    volumes.free = 12 * MIB
    clock[0] += 1.0
    before = volumes.samples
    assert admission(asset_id="other", declared_bytes=10 * MIB, allocated_bytes=0)
    assert volumes.samples == before + 2
    assert admission(asset_id="other", declared_bytes=10 * MIB, allocated_bytes=MIB)
    assert not admission(asset_id="other", declared_bytes=20 * MIB, allocated_bytes=MIB)
    assert volumes.samples == before + 2
    # A fresh plan that does not admit refuses too (free below the remaining growth).
    clock[0] += 1.0
    assert not admission(asset_id="other", declared_bytes=20 * MIB, allocated_bytes=MIB)
    assert volumes.samples == before + 4

    # A sample error propagates (the cache maps it to storage_unknown) and drops the cache.
    volumes.free = None
    clock[0] += 1.0
    assert admission(asset_id="snap", declared_bytes=total, allocated_bytes=0)
    volumes.error = OSError("sample failed")
    clock[0] += 1.0  # the next call must re-plan: it meets the failing sampler
    with pytest.raises(OSError):
        admission(asset_id="snap", declared_bytes=total, allocated_bytes=chunk)
    volumes.error = None
    before = volumes.samples
    assert admission(asset_id="snap", declared_bytes=total, allocated_bytes=2 * chunk)
    assert volumes.samples == before + 2  # the failed call left no cached allowance behind


def test_open_cache_passes_the_configured_download_timeout(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from podmuncher import model_cache

    env = make_bootstrap_env(tmp_path)
    env.config.write_text(env.config.read_text(encoding="utf-8") + "download_timeout_seconds = 300\n",
                          encoding="utf-8")
    ctx = make_ctx(env, tmp_path)
    assert ctx.config.limits.download_timeout_seconds == 300
    ctx.volumes = CountingVolumes()
    ctx.hub = SimpleNamespace(cache_audit=lambda: "audit", cache_spans=lambda: "spans")
    received: dict[str, Any] = {}
    real_cache = model_cache.ModelCache

    class FakeModelCache:
        def __init__(self, root: Any, **kwargs: Any) -> None:
            received.update(kwargs, root=root)

    monkeypatch.setattr(model_cache, "ModelCache", FakeModelCache)
    composition = bootstrap.Composition(env.roots["state"], environ=env.env)
    composition.ctx = ctx
    assert type(composition._open_cache()) is FakeModelCache
    assert received["timeout_seconds"] == 300.0 == float(ctx.config.limits.download_timeout_seconds)
    assert type(received["admission"]) is bootstrap._ModelAdmission
    assert received["root"] == ctx.config.paths.cache and received["audit"] == "audit"
    # The real cache accepts the composed admission and the configured timeout.
    real = real_cache(tmp_path / "real-cache", audit=SimpleNamespace(record=lambda **k: None),
                      admission=received["admission"], timeout_seconds=received["timeout_seconds"])
    try:
        assert real.timeout_seconds == 300.0
    finally:
        real.close()


class FakeModelCache:
    """``ModelCache`` stand-in recording how it was opened; acquires into its own root."""

    opened: ClassVar[list[FakeModelCache]] = []

    def __init__(self, root: Any, **kwargs: Any) -> None:
        self.root, self.kwargs, self.closed = str(root), kwargs, False
        self.acquire = FakeAcquire(Path(root) / "snapshots")
        self.listed = 0
        FakeModelCache.opened.append(self)

    def snapshots(self) -> tuple:
        self.listed += 1
        return ()

    def close(self) -> None:
        self.closed = True


class HeldAwareVolumes(CountingVolumes):
    """Refuses to sample a root an open ``FakeModelCache`` still holds: the Windows symptom
    (``ModelCache`` holds its root without write sharing; the permission probe cannot open
    it).  POSIX opens a held directory fine, so the refusal is Windows-only, matching
    ``_HeldRootVolumes``, which samples held roots directly on POSIX."""

    def sample(self, path: Any, **kwargs: Any) -> Any:
        held = {os.path.normcase(os.path.abspath(c.root)) for c in FakeModelCache.opened if not c.closed}
        if os.name == "nt" and os.path.normcase(os.path.abspath(str(path))) in held:
            raise OSError("the cache root is held open")
        return super().sample(path, **kwargs)


def test_setup_provisioning_follows_the_bound_cache_and_scratch_paths(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The wizard's cache path, not the one the service started with, is where capacity is
    checked, the model lands and the smoke mounts from; the service's cache is untouched."""
    from podmuncher import model_cache

    env = make_bootstrap_env(tmp_path, model_id="tiny")
    env.config.write_text(env.config.read_text(encoding="utf-8") + "download_timeout_seconds = 300\n"
                          + "[tls]\nca_bundle = " + _toml_path(tmp_path / "corp-ca.pem") + "\n",
                          encoding="utf-8")
    ctx, launch, probe = _probe(env, tmp_path)
    ctx.volumes = HeldAwareVolumes()
    ctx.hub = SimpleNamespace(cache_audit=lambda: "audit", cache_spans=lambda: "spans")
    service_cache, service_acquire = ctx.cache, ctx.acquire_model
    monkeypatch.setattr(model_cache, "ModelCache", FakeModelCache)
    monkeypatch.setattr(FakeModelCache, "opened", [])
    new_cache, new_scratch = tmp_path / "chosen" / "cache", tmp_path / "chosen" / "scratch"
    chosen_paths = {"data": str(env.roots["data"]), "output": str(env.roots["output"]),
                    "cache": str(new_cache), "scratch": str(new_scratch), "docker_backing": ""}

    async def scenario() -> None:
        port = bootstrap._Provisioning(ctx, fake_hardware(), probe=probe, loop=asyncio.get_running_loop())
        port.bind_paths(chosen_paths)
        assert FakeModelCache.opened == []  # binding opens nothing (PATHS is before CONFIRM)
        assert not new_cache.exists() and not new_scratch.exists()

        # MODEL_SELECT: the probe worker mounts the scratch being provisioned, like the smoke.
        chosen = await asyncio.to_thread(port.choose_model, {})
        assert chosen.ok is True and chosen.data["choice"] == "tiny", chosen
        probe_run, = launch.launches
        assert probe_run.request.operation == "probe"
        assert probe_run.mounts.scratch.is_relative_to(new_scratch)
        assert not any(env.roots["scratch"].iterdir())  # the configured scratch stays untouched

        capacity = await asyncio.to_thread(port.model_capacity, "tiny")
        assert capacity.ok is True and capacity.data["model"] == "tiny", capacity
        assert FakeModelCache.opened == []  # capacity is a host plan on the bound path only
        assert port.setup_snapshots() == ()
        acquired = await asyncio.to_thread(port.acquire_model, "tiny")
        assert acquired.ok is True and acquired.data["snapshot"] == SNAPSHOT, acquired
        bound, = FakeModelCache.opened
        assert bound.root == str(new_cache) and bound.listed == 1
        assert bound.acquire.calls == ["tiny"] and (new_cache / "snapshots" / "tiny").is_dir()
        # Same composition as the service's cache: audit, spans, CA bundle, timeout, admission
        # on the bound root (sampled through its parent while the cache holds it open).
        assert bound.kwargs["audit"] == "audit" and bound.kwargs["spans"] == "spans"
        assert bound.kwargs["ca_bundle"] == str((tmp_path / "corp-ca.pem").resolve())
        assert bound.kwargs["timeout_seconds"] == 300.0
        admission = bound.kwargs["admission"]
        assert type(admission) is bootstrap._ModelAdmission and admission._path == str(new_cache)
        assert type(admission._ctx.volumes) is bootstrap._HeldRootVolumes
        assert admission._ctx.volumes.sample(str(new_cache)) is not None  # the parent exists
        assert admission(asset_id="snap", declared_bytes=MIB, allocated_bytes=0) is True

        smoked = await asyncio.to_thread(port.smoke, "tiny")
        assert smoked.ok is True and smoked.data["model"] == "tiny", smoked
        assert bound.acquire.calls == ["tiny", "tiny"]
        run = launch.launches[-1]
        assert run.mounts.model == new_cache / "snapshots" / "tiny"
        assert run.mounts.scratch.is_relative_to(new_scratch)
        assert run.mounts.input_file.is_relative_to(new_scratch)
        # The running service's cache and acquisition were never touched.
        assert ctx.cache is service_cache and ctx.acquire_model is service_acquire
        assert service_acquire.calls == [] and not (service_acquire.root / "tiny").exists()
        assert not any(env.roots["scratch"].iterdir())
        assert not bound.closed

        # Going back to PATHS rebinds the same paths: every bind closes the setup cache, so
        # the host and plan checks that follow sample the bound cache root as a plain
        # directory (held open, the Windows sample is storage_unknown).  The cache reopens
        # lazily on the next use.
        if os.name == "nt":
            with pytest.raises(OSError):
                ctx.volumes.sample(str(new_cache))  # still held: the symptom the rebind must cure
        port.bind_paths(chosen_paths)
        assert bound.closed and FakeModelCache.opened == [bound]
        assert ctx.volumes.sample(str(new_cache)).free.value > 0
        plan = bootstrap._host_plan(ctx, "model_unpacked", str(new_cache), MIB)
        assert plan.admitted
        again = await asyncio.to_thread(port.acquire_model, "tiny")
        assert again.ok is True
        first, reopened = FakeModelCache.opened
        assert first is bound and reopened.root == str(new_cache) and not reopened.closed
        assert reopened.acquire.calls == ["tiny"]

        # Rebinding to the configured cache path closes the setup cache and reuses the
        # service's cache (nothing new is opened); the configured scratch is used again.
        port.bind_paths({"cache": ctx.config.paths.cache, "scratch": ctx.config.paths.scratch})
        assert reopened.closed and FakeModelCache.opened == [bound, reopened]
        again = await asyncio.to_thread(port.acquire_model, "tiny")
        assert again.ok is True and service_acquire.calls == ["tiny"]
        assert FakeModelCache.opened == [bound, reopened]
        smoked = await asyncio.to_thread(port.smoke, "tiny")
        assert smoked.ok is True
        assert launch.launches[-1].mounts.scratch.is_relative_to(Path(ctx.config.paths.scratch))
        chosen = await asyncio.to_thread(port.choose_model, {})
        assert chosen.ok is True
        assert launch.launches[-1].mounts.scratch.is_relative_to(Path(ctx.config.paths.scratch))
        port.close()

    asyncio.run(scenario())
    # ``Composition._open_cache`` opens the configured cache or an explicit path the same way.
    composition = bootstrap.Composition(env.roots["state"], environ=env.env)
    composition.ctx = ctx
    assert composition._open_cache().root == ctx.config.paths.cache
    assert composition._open_cache(str(new_cache)).root == str(new_cache)


def test_setup_choose_model_and_smoke_use_the_probe_port(tmp_path: Path) -> None:
    env = make_bootstrap_env(tmp_path, model_id="tiny")
    ctx, launch, probe = _probe(env, tmp_path)

    async def scenario() -> None:
        port = bootstrap._Provisioning(ctx, fake_hardware(), probe=probe,
                                       loop=asyncio.get_running_loop())
        chosen = await asyncio.to_thread(port.choose_model, {})
        assert chosen.ok is True and chosen.reason_code is ReasonCode.OK, chosen
        assert chosen.data["choice"] == "tiny" and chosen.data["device"] == "cpu"
        assert chosen.data["capability"] == "asr_cpu" and chosen.data["compute_type"] == "int8"
        assert chosen.data["duration_seconds"] == 600.0  # the config cap, within tiny's 3600 s
        option, = chosen.data["options"]  # the measured fixture envelope, not the table row
        assert option == {"id": "tiny", "tier": "tiny", "rank": 10, "backend": "cpu",
                          "compute_type": "int8", "peak_ram_bytes": 512 * MIB, "basis": "measured",
                          "source": "measured envelope sha256:" + "4" * 64, "fits": True}
        assert json.loads(json.dumps(chosen.data)) == chosen.data  # JSON-safe for the TUI/CLI
        smoked = await asyncio.to_thread(port.smoke, "tiny")
        assert smoked.ok is True and smoked.data["model"] == "tiny", smoked
        assert smoked.data["segments"] == 2 and smoked.data["duration_seconds"] == 2.0
        assert [r.request.operation for r in launch.launches] == ["probe", "transcribe"]
        smoke = launch.launches[1]
        assert smoke.request.selection.model == "tiny" and smoke.request.model_snapshot_id == SNAPSHOT
        assert smoke.mounts.model == ctx.acquire_model.root / "tiny"
        assert smoke.mounts.input_file.name == "smoke.wav"
        assert not smoke.mounts.input_file.is_relative_to(smoke.mounts.scratch)
        with wave.open(str(smoke.mounts.input_file), "rb") as fixture:
            assert fixture.getframerate() == 16000 and fixture.getnchannels() == 1
            assert fixture.getnframes() == 32000  # a 2 s generated tone, no recording
        assert all(r.ended_at is not None and r.closed for r in launch.launches)
        missing = await asyncio.to_thread(port.smoke, "medium")
        assert missing.ok is False and missing.reason_code is ReasonCode.ASSET_VERIFICATION_FAILED
        launch.fail_image = ReasonCode.SERVICE_UNAVAILABLE
        refused = await asyncio.to_thread(port.choose_model, {})
        assert refused.ok is False and refused.reason_code is ReasonCode.SERVICE_UNAVAILABLE
        unbound = bootstrap._Provisioning(ctx, fake_hardware())  # no port: closed, never a guess
        for result in (unbound.choose_model({}), unbound.smoke("tiny")):
            assert result.ok is False and result.reason_code is ReasonCode.ENVELOPE_UNVALIDATED
            assert result.next_action
        absent = bootstrap._Provisioning(replace(ctx, manifest=None), fake_hardware(), probe=probe,
                                         loop=asyncio.get_running_loop())
        for result in (absent.choose_model({}), absent.smoke("tiny")):
            assert result.ok is False and result.reason_code is ReasonCode.SERVICE_UNAVAILABLE

    asyncio.run(scenario())


def test_setup_probe_and_smoke_admit_on_the_verified_docker_backing(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """MODEL_SELECT's probe and SMOKE's run prepare their Docker storage on the backing the
    setup's BACKING step verified (bound paths; "" is the wizard's auto discovery), as PULL
    does: nothing bound, the configured one.  A release image, verified present just
    before, is launched without a pull.  The real launch port forwards that backing into
    ``docker_storage.prepare``."""
    from podmuncher import runtime

    env = make_bootstrap_env(tmp_path, model_id="tiny")
    ctx, launch, probe = _probe(env, tmp_path)

    async def scenario() -> None:
        port = bootstrap._Provisioning(ctx, fake_hardware(), probe=probe, loop=asyncio.get_running_loop())
        assert (await asyncio.to_thread(port.choose_model, {})).ok is True
        assert (await asyncio.to_thread(port.smoke, "tiny")).ok is True
        configured = ctx.config.paths.docker_backing
        assert launch.launch_backings == [configured, configured]  # nothing bound
        port.bind_paths({"cache": ctx.config.paths.cache, "docker_backing": "D:\\docker-data"})
        assert (await asyncio.to_thread(port.choose_model, {})).ok is True
        assert (await asyncio.to_thread(port.smoke, "tiny")).ok is True
        assert launch.launch_backings[-2:] == ["D:\\docker-data"] * 2  # the BACKING override
        port.bind_paths({"cache": ctx.config.paths.cache, "docker_backing": ""})
        assert (await asyncio.to_thread(port.choose_model, {})).ok is True
        assert (await asyncio.to_thread(port.smoke, "tiny")).ok is True
        assert launch.launch_backings[-2:] == ["", ""]  # auto discovery, not the configured one
        assert [r.request.operation for r in launch.launches] == ["probe", "transcribe"] * 3
        # The fixture's development policies are launched as they are (never pulled).
        assert all(r.policy.development_image and not r.policy.allow_pull for r in launch.launches)
        # A release policy, verified present by the port just before, is launched without a pull.
        release_manifest(ctx, {})
        assert ctx.resolved("asr_cpu").worker_policy.allow_pull is True
        assert (await asyncio.to_thread(port.choose_model, {})).ok is True
        assert (await asyncio.to_thread(port.smoke, "tiny")).ok is True
        for run in launch.launches[-2:]:
            assert run.policy.image == FIXTURE_REF and run.policy.allow_pull is False
            assert run.policy.development_image is False and callable(run.policy.cached_proof)
        assert launch.verified[-2:] == [FIXTURE_REF] * 2
        port.close()

    asyncio.run(scenario())
    # The real launch port: the verified backing reaches ``docker_storage.prepare``.
    from podmuncher.protocol import WorkerRequest
    from podmuncher.requests import RequestOutputs, RequestSelection
    from podmuncher.runtime import WorkerMounts

    class Constructed(Exception):
        pass

    class RecordingStorage:
        def __init__(self, session: Any, volumes: Any, **kwargs: Any) -> None:
            raise Constructed

    prepared: list[dict] = []
    closed: list[bool] = []
    ctx.volumes = object()
    ctx.docker_storage = SimpleNamespace(
        prepare=lambda **kw: prepared.append(kw) or SimpleNamespace(close=lambda: closed.append(True)))
    monkeypatch.setattr(runtime, "RuntimeStorage", RecordingStorage)
    real = bootstrap._DockerLaunch(ctx, _access(ReleaseCli(FIXTURE_REF)))
    policy = ctx.resolved("asr_cpu").worker_policy
    request = WorkerRequest(1, str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4()), "asr_cpu", "probe",
                            {"probe_metadata": "probe_metadata-" + "a" * 32 + ".bin"}, None,
                            RequestOutputs(), None, None, ctx.config.limits, RequestSelection(device="cpu"))
    scratch = tmp_path / "launch-scratch"
    scratch.mkdir()
    for backing, expected in ((None, ctx.config.paths.docker_backing), ("D:\\docker-data", "D:\\docker-data"),
                              ("", "")):
        extra = {} if backing is None else {"manual_backing": backing}
        with pytest.raises(Constructed):
            asyncio.run(real.launch(policy, WorkerMounts(scratch), request, lambda event: None,
                                    byte_limit=4096, timeout=5.0, growth=bootstrap._probe_growth(), **extra))
        assert prepared[-1] == {"manual_backing": expected}
    assert closed == [True] * 3  # the session is closed on the failed construction


def _choose(ctx: Any, probe: Any) -> Any:
    async def scenario() -> Any:
        port = bootstrap._Provisioning(ctx, fake_hardware(), probe=probe, loop=asyncio.get_running_loop())
        return await asyncio.to_thread(port.choose_model, {})

    return asyncio.run(scenario())


def test_setup_choose_model_offers_every_verified_model_with_fit_guidance(tmp_path: Path) -> None:
    # tiny keeps its measured envelope; the rest are unmeasured and judged by the table's CPU
    # rows.  On this 4 GiB worker (3800 MiB available, 20% headroom, 600 s decode) small is the
    # largest fit; medium (4.5 GB) and large-v3 (4.43 GB) do not fit but are still offered.
    doc = make_manifest_doc(models=("tiny", "base", "small", "medium", "large-v3"),
                            unmeasured=("base", "small", "medium", "large-v3"))
    env = make_bootstrap_env(tmp_path, manifest_doc=doc)
    ctx, launch, probe = _probe(env, tmp_path)
    chosen = _choose(ctx, probe)
    assert chosen.ok is True and chosen.data["choice"] == "small", chosen
    assert chosen.data["device"] == "cpu" and chosen.data["reason"] == "auto_largest_fit"
    options = chosen.data["options"]
    assert [o["id"] for o in options] == ["large-v3", "medium", "small", "base", "tiny"]  # rank order
    assert [o["rank"] for o in options] == [50, 40, 30, 20, 10]
    assert [o["fits"] for o in options] == [False, False, True, True, True]
    assert [o["basis"] for o in options] == ["estimated", "estimated", "published", "measured", "measured"]
    by_id = {o["id"]: o for o in options}
    assert by_id["small"]["peak_ram_bytes"] == 1_477_000_000
    assert by_id["small"]["source"] == "https://github.com/SYSTRAN/faster-whisper#benchmark"
    assert by_id["large-v3"]["peak_ram_bytes"] == 4_429_500_000
    assert by_id["base"]["source"] == "podmuncher measured 2026-10-13"  # the table's own measurement
    assert by_id["tiny"]["source"].startswith("measured envelope sha256:")  # the manifest envelope
    assert by_id["tiny"]["peak_ram_bytes"] == 512 * MIB
    assert all(o["backend"] == "cpu" and o["compute_type"] == "int8" and "peak_vram_bytes" not in o
               for o in options)
    assert all(type(o["source"]) is str and len(o["source"]) <= 512 for o in options)  # bounded text
    assert json.loads(json.dumps(chosen.data)) == chosen.data
    run, = launch.launches
    assert run.ended_at is not None and run.closed  # the options were built inside the hold


def test_setup_choose_model_duration_is_none_safe_for_unmeasured_models(tmp_path: Path) -> None:
    # Every model unmeasured: the setup selects for the configured duration cap (this crashed
    # with TypeError on None before SPEC_GAP-16).
    doc = make_manifest_doc(models=("tiny", "base"), unmeasured=("tiny", "base"))
    env = make_bootstrap_env(tmp_path, manifest_doc=doc)
    ctx, _launch, probe = _probe(env, tmp_path)
    chosen = _choose(ctx, probe)
    assert chosen.ok is True and chosen.data["choice"] == "base", chosen
    assert chosen.data["duration_seconds"] == 600.0 == float(ctx.config.limits.duration_seconds)
    assert [o["basis"] for o in chosen.data["options"]] == ["measured", "measured"]  # table rows
    # One measured model validated below the cap bounds the setup's duration; the unmeasured
    # one contributes no bound.
    doc = make_manifest_doc(models=("tiny", "base"), unmeasured=("base",))
    doc["models"][0]["validated_max_duration_seconds"] = 300
    env = make_bootstrap_env(tmp_path / "bounded", manifest_doc=doc)
    ctx, _launch, probe = _probe(env, tmp_path / "bounded")
    chosen = _choose(ctx, probe)
    assert chosen.ok is True and chosen.data["duration_seconds"] == 300.0, chosen


def test_setup_choose_model_without_an_automatic_fit_still_offers_the_models(tmp_path: Path) -> None:
    # A 1 GiB worker fits none of the table's CPU rows (tiny needs 855 MB + decode + headroom):
    # the step succeeds with no suggestion, the operator may still pick a model (owner rule 7).
    doc = make_manifest_doc(models=("tiny", "base"), unmeasured=("tiny", "base"), worker_memory_bytes=GIB)
    env = make_bootstrap_env(tmp_path, manifest_doc=doc)
    ctx, launch, probe = _probe(env, tmp_path, ram=(GIB, 100 * MIB, 900 * MIB))
    chosen = _choose(ctx, probe)
    assert chosen.ok is True and chosen.reason_code is ReasonCode.OK, chosen
    assert chosen.data["choice"] is None and chosen.data["reason"] == ReasonCode.RAM_INSUFFICIENT.value
    assert chosen.data["device"] == "cpu" and chosen.data["capability"] == "asr_cpu"
    assert [(o["id"], o["fits"]) for o in chosen.data["options"]] == [("base", False), ("tiny", False)]
    assert chosen.data["options"][1]["peak_ram_bytes"] == 855_638_016  # the numbers stay shown
    run, = launch.launches
    assert run.ended_at is not None and run.closed and run.obsoleted  # reaped; the hold ended at once
    # A probe that fails before any evidence stays a closed failure: no options, no guess.
    launch.fail_image = ReasonCode.SERVICE_UNAVAILABLE
    refused = _choose(ctx, probe)
    assert refused.ok is False and refused.reason_code is ReasonCode.SERVICE_UNAVAILABLE
    assert "options" not in refused.data and refused.next_action


def test_accepted_domains_reject_contract_negatives(tmp_path: Path) -> None:
    registry = json.loads((ROOT / "Docs" / "contract-domains.json").read_text(encoding="utf-8"))
    negatives = registry["negative_examples"]
    assert len(negatives) == 2
    from podmuncher import domains

    exercised = {"capability": Capability, "artifact_slot": ArtifactSlot, "job_state": JobState,
                 "service_state": ServiceState, "field_status": FieldStatus,
                 "worker_slot": WorkerSlot, "reason_code": ReasonCode,
                 "stage": domains.Stage, "context": domains.Context}
    for name, kind in exercised.items():
        for value in registry[name]:
            kind.parse(value)
        for value in negatives:
            with pytest.raises(DomainValidationError):
                kind.parse(value)
    env = make_bootstrap_env(tmp_path)
    seams, _ = make_seams(env)
    with LoopbackService(env, seams) as svc:
        client = svc.client()
        try:
            for value in negatives:
                assert reason(lambda value=value: client.setup_plan([value], {})).reason_code \
                    is ReasonCode.INVALID_INPUT
                assert reason(lambda value=value: client.cancel(value)).status in (400, 404)
        finally:
            client.close()
        assert svc.stop() == service.EXIT_OK


# --------------------------------------------------------------------------- client binding


def test_client_startup_binds_presentation_and_registry_without_daemon(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from argparse import Namespace

    from podmuncher import presentation

    env = make_bootstrap_env(tmp_path)
    args = Namespace(input=str(env.source), device=None, profile=None, model="small",
                     language=None, output=None, video=False, no_transcript=False)
    try:
        monkeypatch.delenv(bootstrap.MANIFEST_ENV, raising=False)
        bootstrap.client_startup({})
        with pytest.raises(DomainValidationError):
            cli.build_request(args, {"PODMUNCHER_CONFIG": str(env.config)})
        bootstrap.client_startup(env.env)
        assert isinstance(cli._adapter, type(presentation.default_adapter()))
        request = cli.build_request(args, {"PODMUNCHER_CONFIG": str(env.config)})
        assert request["selection"]["model"] == "small"
        assert service.discover(env.roots["state"]) is None
        assert list(env.roots["state"].iterdir()) == []  # nothing started or written
    finally:
        cli.set_registry_provider(None)
        cli.configure_presentation(None)


def test_package_main_installs_the_lazy_hook(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                             capsys: pytest.CaptureFixture) -> None:
    import podmuncher

    env = make_bootstrap_env(tmp_path)
    for key, value in env.env.items():
        monkeypatch.setenv(key, value)
    try:
        assert podmuncher.main(["status", "--json"]) == 0
        assert cli._client_startup is podmuncher._client_startup
        out = json.loads(capsys.readouterr().out)
        assert out["service_state"] == "stopped" and out["stale"] is True
        assert not (env.roots["state"] / "runtime.json").exists()
    finally:
        cli.set_client_startup(None)
        cli.set_registry_provider(None)
        cli.configure_presentation(None)


def test_tui_uses_the_final_presentation_functions() -> None:
    from podmuncher import presentation, tui

    assert tui.safe(HOSTILE) == presentation.escape_display(HOSTILE)
    assert "\x1b" not in tui.safe(HOSTILE) and "\\u202e" in tui.safe(HOSTILE)
    family = "\U0001f468‍\U0001f469‍\U0001f467"  # one grapheme, never split
    assert tui.fit(family * 3, 3) == presentation.truncate_display(family * 3, 3)
    assert presentation.key_action("q", in_text_input=True) is None


@pytest.mark.asyncio
async def test_tui_welcome_wordmark_only_on_the_empty_overview() -> None:
    from podmuncher import presentation, tui

    class EmptyService:
        def snapshot(self) -> dict:
            return {"snapshot": None}

        def list_jobs(self, limit: int | None = None, cursor: str | None = None) -> dict:
            return {"jobs": [], "next_cursor": None}

        def close(self) -> None:
            pass

    profile = presentation.TerminalProfile(color="full", theme="dark", reduced_motion=True,
                                           unicode=True, fullscreen_ok=True)
    app = tui.PodmuncherApp(EmptyService, poll_seconds=0.05, profile=profile,
                            doctor=lambda: {"status": "ok", "checks": {}})
    async with app.run_test(size=(100, 30)) as pilot:
        for _ in range(100):
            if app.model.phase == "connected":
                break
            await pilot.pause(0.02)
        head = str(app.screen.query_one("#jobs-head").content)
        assert presentation.welcome_text(profile) in head and "none yet" in head
        assert app.theme == "textual-dark"


def _console_env(env: SimpleNamespace, tmp_path: Path) -> dict[str, str]:
    full = os.environ.copy()
    full.update(env.env, PYTHONDONTWRITEBYTECODE="1", TERM="xterm")
    full.pop("NO_COLOR", None)
    return full


def run_console(*args: str, env: dict[str, str], cwd: Path, timeout: float = 120) -> Any:
    assert CONSOLE.exists(), f"console script missing: {CONSOLE}"
    return subprocess.run([str(CONSOLE), *args], cwd=cwd, env=env, capture_output=True,
                          timeout=timeout, check=False, stdin=subprocess.DEVNULL)


@pytest.mark.parametrize("argument", ["--help", "--version"])
def test_cli_help_and_version_stay_lazy(argument: str, tmp_path: Path) -> None:
    env = make_bootstrap_env(tmp_path)
    shutil.rmtree(env.roots["state"])
    code = ("import sys; from podmuncher import main\n"
            "try:\n    main(sys.argv[1:])\nexcept SystemExit:\n    pass\n"
            "print('BOOT', 'podmuncher.bootstrap' in sys.modules, 'aiohttp' in sys.modules)")
    result = subprocess.run([sys.executable, "-c", code, argument], cwd=tmp_path,
                            env={**_console_env(env, tmp_path), "PYTHONPATH": str(ROOT / "src")},
                            capture_output=True, text=True, timeout=60, check=False)
    assert result.stdout.strip().splitlines()[-1] == "BOOT False False", result.stderr
    console = run_console(argument, env=_console_env(env, tmp_path), cwd=tmp_path)
    assert console.returncode == 0 and b"podmuncher" in console.stdout
    assert not env.roots["state"].exists()  # nothing created, nothing started


def test_cli_status_renders_final_escaping_and_keeps_json_original(tmp_path: Path) -> None:
    env = make_bootstrap_env(tmp_path)
    assert service.save_snapshot(env.roots["state"], {"note": HOSTILE, "count": 3})
    before = sorted(p.name for p in env.roots["state"].iterdir())
    full = _console_env(env, tmp_path)
    plain = run_console("status", env=full, cwd=tmp_path)
    assert plain.returncode == 0, plain.stderr
    text = plain.stdout.decode("utf-8")
    assert "\x1b" not in text and "\u202e" not in text and "\x07" not in text
    assert "\\x1b" in text and "\\u202e" in text and "\\x07" in text
    assert "\\[bold\\]" in text and "[bold]" not in text.replace("\\[bold\\]", "")
    as_json = run_console("status", "--json", env=full, cwd=tmp_path)
    assert as_json.returncode == 0
    document = json.loads(as_json.stdout.decode("utf-8"))
    assert document["snapshot"]["note"] == HOSTILE  # the original data, unescaped
    assert document["service_state"] == "stopped" and document["stale"] is True
    assert sorted(p.name for p in env.roots["state"].iterdir()) == before  # never started
    assert service.discover(env.roots["state"]) is None


# ---------------------------------------------------------------------------- live smoke


class _PullSpy:
    """Watches for any ``docker pull`` process while the smoke runs (psutil polling)."""

    def __init__(self) -> None:
        self.seen: list[list[str]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        import psutil

        while not self._stop.is_set():
            for proc in psutil.process_iter(["name", "cmdline"]):
                try:
                    name = (proc.info.get("name") or "").lower()
                    argv = [str(a).lower() for a in (proc.info.get("cmdline") or ())]
                except (psutil.Error, OSError):
                    continue
                if name.startswith("docker") and "pull" in argv:
                    self.seen.append(argv)
            self._stop.wait(0.05)

    def __enter__(self) -> Self:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(10)


def _engine_pull_events(since: str, until: str) -> list[str] | None:
    """Image pull events recorded by the Docker engine in the window, or None if unreachable."""
    docker = shutil.which("docker")
    if docker is None:
        return None
    try:
        result = subprocess.run([docker, "events", "--since", since, "--until", until,
                                 "--filter", "type=image", "--filter", "event=pull",
                                 "--format", "{{.ID}}"], capture_output=True, text=True,
                                timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return [line for line in result.stdout.splitlines() if line.strip()]


@pytest.mark.live
def test_own_transport_smoke(tmp_path: Path) -> None:
    import psutil

    assert CONSOLE.exists(), f"prerequisite missing: console script {CONSOLE}"
    env = make_bootstrap_env(tmp_path)
    full = _console_env(env, tmp_path)
    pid = None
    since = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    try:
        with _PullSpy() as spy:
            started = run_console("service", "start", "--json", env=full, cwd=tmp_path)
            assert started.returncode == 0, started.stdout + started.stderr
            pid = json.loads(started.stdout)["pid"]
            assert psutil.pid_exists(pid)

            status = run_console("status", "--json", env=full, cwd=tmp_path)
            assert status.returncode == 0, status.stdout + status.stderr
            status_doc = json.loads(status.stdout)
            assert status_doc["service_state"] == "running" and status_doc["stale"] is False
            snapshot = status_doc["snapshot"]
            assert snapshot["service"]["value"] == "running"
            assert snapshot["schema_version"] == 1 and snapshot["storage"]

            doctor = run_console("doctor", "--json", env=full, cwd=tmp_path, timeout=180)
            doctor_doc = json.loads(doctor.stdout)
            assert doctor.returncode in (0, 3), doctor_doc
            names = [check["name"] for check in doctor_doc["checks"]]
            assert names[:2] == ["service", "snapshot"] and "docker" in names
            assert doctor_doc["service_state"] == "running"

            stopped = run_console("service", "stop", "--json", env=full, cwd=tmp_path)
            assert stopped.returncode == 0, stopped.stdout + stopped.stderr
            assert json.loads(stopped.stdout)["outcome"] == "stopped"
            assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == "zombie"
            pid = None
            after = run_console("status", "--json", env=full, cwd=tmp_path)
            assert json.loads(after.stdout)["service_state"] == "stopped"
        assert spy.seen == [], "a docker pull ran during the smoke"
        until = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        events = _engine_pull_events(since, until)
        assert events in (None, []), "the Docker engine recorded an image pull"
        assert not (env.roots["state"] / "runtime.json").exists()
        downloaded = [p for p in env.roots["cache"].rglob("*") if p.is_file()
                      and "snapshots" in p.parts]
        assert downloaded == [], "a model download happened"
    finally:
        if pid is not None and psutil.pid_exists(pid):
            run_console("service", "stop", "--json", env=full, cwd=tmp_path)
            with __import__("contextlib").suppress(psutil.Error):
                psutil.Process(pid).kill()
        shutil.rmtree(tmp_path, ignore_errors=True)
    assert not tmp_path.exists() or not any(tmp_path.iterdir())
    print("PODMUNCHER_SMOKE_OK")


@pytest.mark.live
def test_probe_evidence_live(tmp_path: Path) -> None:
    """Optional live probe against the real local ASR image (never built or pulled here).

    Not part of the contract smoke: it needs an explicit ``PODMUNCHER_TEST_PROBE_LIVE=1``
    opt-in plus ``PODMUNCHER_TEST_ASR_IMAGE`` and ``PODMUNCHER_TEST_MODEL_ROOT``.
    """
    if os.environ.get("PODMUNCHER_TEST_PROBE_LIVE") != "1":
        pytest.skip("live probe not selected: set PODMUNCHER_TEST_PROBE_LIVE=1 together with "
                    "PODMUNCHER_TEST_ASR_IMAGE and PODMUNCHER_TEST_MODEL_ROOT")
    tag = os.environ.get("PODMUNCHER_TEST_ASR_IMAGE")
    model_root = os.environ.get("PODMUNCHER_TEST_MODEL_ROOT")
    assert tag and model_root, ("PODMUNCHER_TEST_ASR_IMAGE (a local build tag) and "
                                "PODMUNCHER_TEST_MODEL_ROOT must be set for the live probe")
    from podmuncher.docker_storage import make_docker_storage_provider
    from podmuncher.model_policy import ModelChoice, PolicyRequest, select_model
    from podmuncher.volumes import make_volume_provider

    target = bootstrap.resolve_docker_target()

    def docker(*args: str, timeout: float = 120) -> Any:
        return subprocess.run(("docker", "--host", target.endpoint, *args), capture_output=True,
                              text=True, encoding="utf-8", errors="replace", timeout=timeout,
                              check=False, stdin=subprocess.DEVNULL)

    inspected = docker("image", "inspect", "--format", "{{.Id}} {{.Size}}", tag)
    assert inspected.returncode == 0, (f"the ASR image {tag} is not present locally; build it with "
                                       "the tests/test_asr_worker.py live smoke first (nothing is "
                                       "pulled or built here)")
    image_id, size = inspected.stdout.split()
    version = docker("run", "--rm", "--pull", "never", "--network", "none", "--entrypoint", "python",
                     image_id, "-I", "-c", "import ctranslate2; print(ctranslate2.__version__)")
    assert version.returncode == 0 and version.stdout.strip(), version.stderr[-300:]
    doc = make_manifest_doc()
    for entry in doc["capabilities"]:
        if entry["id"] == "asr_cpu":
            entry.update(digest=image_id, entrypoint=["python", "-I", "-m", "podmuncher.asr_worker"],
                         compressed_bytes=int(size), unpacked_peak_bytes=int(size),
                         runtime_version=version.stdout.strip(), worker_memory_bytes=2 * GIB)
    env = make_bootstrap_env(tmp_path, manifest_doc=doc)
    ctx = make_ctx(env, tmp_path)
    ctx.instance = f"pm-{uuid.uuid4().hex}"
    volumes = make_volume_provider()
    ctx.volumes = bootstrap._HeldRootVolumes(volumes, (ctx.config.paths.cache,))
    ctx.docker_storage = make_docker_storage_provider(volumes=volumes)
    probe = bootstrap._ProbeEvidence(ctx, bootstrap._DockerLaunch(ctx, bootstrap._DockerAccess()))
    timings: dict[str, Any] = {}

    async def scenario() -> Any:
        began = time.monotonic()

        def use(evidence: Any, resolved: Any) -> None:
            timings["used_at"] = time.monotonic() - began
            worker = evidence.workers[0]
            timings["age_at_use"] = bootstrap._evidence_age(worker.hardware, datetime.now(UTC))
            timings["choice"] = select_model(PolicyRequest(4.0, 2 * GIB, model="small"), evidence,
                                             resolved.envelopes)

        evidence = await probe.collect("asr_cpu", use=use)
        timings["returned_at"] = time.monotonic() - began
        return evidence

    evidence = asyncio.run(scenario())
    worker = evidence.workers[0]
    assert worker.runtime.compute_types and {"int8", "float32"} & set(worker.runtime.compute_types)
    assert worker.hardware.worker_ram_limit.status is FieldStatus.OK
    assert 0 < worker.hardware.worker_ram_limit.value <= 2 * GIB
    assert worker.hardware.worker_ram_available.value > 0
    assert worker.fingerprint.image_digest == image_id and timings["age_at_use"] <= 3.0
    assert type(timings["choice"]) is ModelChoice and timings["choice"].artifact.model_id == "small"
    assert probe.writers_reaped and not probe.running
    left = docker("ps", "-a", "--no-trunc", "--format", "{{.ID}}", "--filter",
                  f"label=podmuncher.instance={ctx.instance}")
    assert left.returncode == 0 and not left.stdout.strip(), "the probe container was not reaped"
    print(f"PODMUNCHER_PROBE_LIVE_OK runtime={worker.fingerprint.runtime_version} "
          f"compute_types={list(worker.runtime.compute_types)} "
          f"ram_limit={worker.hardware.worker_ram_limit.value} "
          f"ram_available={worker.hardware.worker_ram_available.value} "
          f"age_at_use={timings['age_at_use']:.2f}s used_at={timings['used_at']:.2f}s "
          f"returned_at={timings['returned_at']:.2f}s")
