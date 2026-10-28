"""Deterministic adversarial failure/recovery matrix (P6-U3).

Every case drives real product modules - the composed ``bootstrap.Composition`` behind the
real loopback API (``service._serve_async``), the real repository, scheduler, publisher,
exporter, recovery, CLI rendering, protocol and runtime - over a temporary filesystem.
Only the outer seams are faked, inside this file: the Docker storage sampler, the volume
sampler, owned-container listing, the worker launcher (it writes real outputs into the
host-assigned scratch), the probe launch port, the model acquisition, the Docker CLI
process factory and the clock.  No helper is imported from another test module.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import http.client
import io
import json
import os
import sys
import threading
import time
import types
import uuid
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self

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
    Stage,
    WorkerSlot,
)
from podmuncher.service import ClientError

ROOT = Path(__file__).resolve().parents[1]
MIB, GIB = 1024**2, 1024**3
DIGEST = "sha256:" + "1" * 64
SNAPSHOT = "snapshot-" + "a" * 32
MODEL_BYTES = b"synthetic model fixture"
STAMP = "%Y-%m-%dT%H:%M:%S.%fZ"
RUNTIME_VERSION = "fixture-runtime"
TRANSCRIPT_SLOTS = sorted(s.value for s in ArtifactSlot if s is not ArtifactSlot.ORIGINAL_VIDEO)
HOSTILE = "evil\x1b[31m [bold]markup[/bold] \u202edesrever\x07 end"
FACTORY_MODULE = "podmuncher_p6u3_matrix_factory"
PROBE_DOCUMENT = {"schema_version": 1, "runtime": {"name": "ctranslate2", "version": RUNTIME_VERSION},
                  "compute_types": ["float32", "int8"], "device": "cpu", "cuda_device_uuid": None,
                  "faster_whisper_version": "1.2.1"}
_TIERS = {"tiny": "tiny", "base": "base", "small": "small", "medium": "medium"}
TERMINAL = {"completed", "partial", "failed", "cancelled", "blocked", "interrupted"}


# ------------------------------------------------------------------- factories: environment


def _capability(cap: str, memory: int = 4 * GIB) -> dict:
    return {
        "id": cap, "platform": "fixture-platform", "image_reference": "fixture-local",
        "digest": DIGEST, "entrypoint": ["python", "-m", "fixture_worker"],
        "compressed_bytes": 100, "unpacked_peak_bytes": 200,
        "worker_overhead_bytes": 256 * MIB + 2 * 10 * MIB, "worker_memory_bytes": memory,
        "dependency_lock_sha256": "2" * 64, "licenses": ["fixture-license"],
        "provenance": "fixture:local-image", "runtime_version": RUNTIME_VERSION,
        "driver_version": None, "hardware_fingerprint": "fixture-hardware",
    }


def _model(model_id: str) -> dict:
    return {
        "id": model_id, "tier": _TIERS.get(model_id, "small"), "revision": "fixture-revision",
        "digest": "3" * 64, "source_urls": [f"fixture:{model_id}.bin"],
        "files": [{"relative_path": f"{model_id}.bin", "bytes": len(MODEL_BYTES),
                   "sha256": hashlib.sha256(MODEL_BYTES).hexdigest()}],
        "license": "fixture-license", "provenance": "fixture:generated-test-data",
        "compute_types": ["int8"], "worker_peak_ram_bytes": 512 * MIB,
        "worker_peak_vram_bytes": 0, "validated_max_duration_seconds": 3600,
        "decode_bytes_per_second": 256000, "decode_timeout_seconds": 120,
        "codec_matrix": ["fixture_codec"], "evidence_sha256": "4" * 64,
        "release_tested": True, "synthetic": True,
    }


def manifest_doc(models: tuple[str, ...] = ("small",), memory: int = 4 * GIB) -> dict:
    return {
        "schema_version": 1, "release_version": "fixture",
        "source": {"kind": "development_fixture", "label": "fixture",
                   "provenance": "fixture:test_failure_matrix"},
        "platforms": ["fixture-platform"],
        "capabilities": [_capability("media"), _capability("asr_cpu", memory)],
        "models": [_model(m) for m in models], "languages": ["en"],
    }


def make_env(tmp_path: Path, *, doc: dict | None = None, manifest: bool = True,
             export_bytes: int = 1048576) -> SimpleNamespace:
    """Tiny temp roots, an optional explicit fixture manifest and a config TOML."""
    roots = {name: tmp_path / name for name in ("data", "output", "cache", "scratch", "state",
                                                "manifest", "input")}
    for path in roots.values():
        path.mkdir(parents=True, exist_ok=True)
    path = roots["manifest"] / "manifest.json"
    document = doc or manifest_doc()
    path.write_text(json.dumps(document), encoding="utf-8")
    for model in document["models"]:
        for item in model["files"]:
            (roots["manifest"] / item["relative_path"]).write_bytes(MODEL_BYTES)
    config = tmp_path / "config.toml"
    config.write_text("\n".join([
        "schema_version = 1", "[paths]",
        *(f"{n} = '{roots[n]}'" for n in ("data", "output", "cache", "scratch")),
        "[limits]", "media_bytes = 8388608", "duration_seconds = 600",
        f"export_bytes = {export_bytes}", "reserve_fraction = 0.0", "reserve_floor_bytes = 0",
        "reserve_ceiling_bytes = 0", "",
    ]), encoding="utf-8")
    source = roots["input"] / "talk.mp4"
    source.write_bytes(b"\x00\x00\x00\x18ftypmp42 synthetic media bytes for the matrix")
    env = {"PODMUNCHER_STATE_DIR": str(roots["state"]), "PODMUNCHER_CONFIG": str(config)}
    if manifest:
        env["PODMUNCHER_RUNTIME_MANIFEST"] = str(path)
    return SimpleNamespace(roots=roots, manifest=path, config=config, source=source, env=env,
                           export_bytes=export_bytes)


def job_request(env: SimpleNamespace, *, transcript: bool = True, video: bool = False,
                model: str = "auto") -> dict:
    return {"schema_version": 1, "source": {"kind": "local", "input": str(env.source)},
            "outputs": {"transcript": transcript, "video": video, "video_quality": "1080p"},
            "selection": {"device": "auto", "profile": "balanced", "model": model,
                          "language": "auto"},
            "output_root": str(env.roots["output"]),
            "limits": {"media_bytes": 8388608, "duration_seconds": 600,
                       "export_bytes": env.export_bytes}}


# --------------------------------------------------------------- factories: storage seams


def _measure(value: Any, status: FieldStatus = FieldStatus.OK, reason: ReasonCode | None = None,
             source: str = "fake", age: float = 0.0) -> Any:
    from podmuncher.snapshots import Measurement

    stamp = (datetime.now(UTC) - timedelta(seconds=age)).strftime(STAMP)
    return Measurement(value, source, status, stamp, reason)


class SwitchDockerStorage:
    """Docker storage evidence on the real temp volume; ``mode`` injects stale/unknown."""

    def __init__(self, root: Path) -> None:
        from podmuncher.volumes import make_volume_provider

        self.root, self.volumes, self.mode, self.calls = root, make_volume_provider(), "ok", 0

    def sample(self, *, manual_backing: str = "") -> Any:
        from podmuncher.docker_storage import Constraint, DockerStorageEvidence

        self.calls += 1
        volume = self.volumes.sample(str(self.root))
        if self.mode == "stale":
            admission = _measure(None, FieldStatus.STALE, ReasonCode.STORAGE_STALE)
        elif self.mode == "unknown":
            admission = _measure(None, FieldStatus.UNKNOWN, ReasonCode.STORAGE_UNKNOWN)
        else:
            admission = _measure(True)
        return DockerStorageEvidence(
            _measure("desktop-linux"), _measure("/var/lib/docker"), _measure("4.0"),
            (Constraint(("docker_data",), ("/var/lib/docker",), volume),), (), admission,
            time.monotonic())


class SwitchVolumes:
    """The real volume provider; ``mode = "full"`` reports only 1 MiB free on every volume."""

    def __init__(self) -> None:
        from podmuncher.volumes import make_volume_provider

        self.inner, self.mode = make_volume_provider(), "ok"

    def _shape(self, evidence: Any) -> Any:
        if self.mode == "full":
            return replace(evidence, free=_measure(MIB, source=evidence.free.source))
        return evidence

    def sample(self, path: Any, **kwargs: Any) -> Any:
        return self._shape(self.inner.sample(path, **kwargs))

    def sample_many(self, paths: tuple[str, ...]) -> tuple[Any, ...]:
        return tuple(self._shape(e) for e in self.inner.sample_many(paths))

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class FakeContainers:
    """recovery.OwnedContainers; ``unreachable`` models Docker gone after a crash."""

    def __init__(self, *, unreachable: bool = False, refs: tuple = ()) -> None:
        self.unreachable, self.refs, self.log = unreachable, list(refs), []

    async def list(self, instance: str) -> list:
        self.log.append(("list", instance))
        if self.unreachable:
            raise RuntimeError("docker unreachable")
        return [r for r in self.refs if r.labels.get("podmuncher.instance") == instance]

    async def stop(self, ref: Any) -> None:
        self.log.append(("stop", ref.container_id))

    async def kill(self, ref: Any) -> None:
        self.log.append(("kill", ref.container_id))

    async def remove(self, ref: Any) -> None:
        self.log.append(("remove", ref.container_id))
        self.refs = [r for r in self.refs if r.container_id != ref.container_id]

    async def exists(self, ref: Any) -> bool:
        return any(r.container_id == ref.container_id for r in self.refs)


# --------------------------------------------------------------- factories: worker seams


class FakeLauncher:
    """scheduler.WorkerLauncher writing real outputs into the host-assigned scratch."""

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx
        self.ops: list[str] = []
        self.selections: list[tuple[str, str]] = []
        self.prepared: set[str] = set()
        self.gates: dict[str, threading.Event] = {}
        self.entered: dict[str, threading.Event] = {}
        self.fail: dict[str, ReasonCode] = {}
        self.oom: int = 0
        self.facts = {"has_audio": True, "has_video": True, "duration_seconds": 4.0}
        self.text = "first line"
        self.pad = 0
        self.link_target: Path | None = None
        self.stopped = False
        self.stops: list = []

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

    def _document(self, request: Any) -> bytes:
        if request.operation == "probe":
            return json.dumps({"schema_version": 1, "operation": "probe", "source_kind": None,
                               "audio_streams": int(self.facts["has_audio"]),
                               "video_streams": int(self.facts["has_video"]),
                               "video_height": 120 if self.facts["has_video"] else None,
                               "estimated_bytes": None, **self.facts}).encode()
        return json.dumps({
            "schema_version": 1, "language": "en", "language_probability": 0.9,
            "duration_seconds": 4.0, "no_speech_detected": False,
            "segments": [{"start_seconds": 0.0, "end_seconds": 1.5,
                          "text": self.text + "x" * self.pad},
                         {"start_seconds": 1.5, "end_seconds": 3.5, "text": "second line"}],
            "provenance": {"model_id": request.selection.model,
                           "model_snapshot_id": request.model_snapshot_id, "device": "cpu",
                           "compute_type": "int8", "source_sha256": "5" * 64,
                           "attempt_id": request.attempt_id, "image_digest": None},
            "measurements": {}}).encode()

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
        if op in self.fail:
            raise RuntimeFailure(self.fail[op])
        if op == "transcribe":
            self.selections.append((request.selection.device, request.selection.model))
            if self.oom > 0:
                self.oom -= 1
                raise RuntimeFailure(ReasonCode.MODEL_OOM)
        slot, name = next(iter(request.assigned_slots.items()))
        data = self._document(request)
        target = Path(self.ctx.scratch.path_of(name))
        if op == "transcribe" and self.link_target is not None:
            self.link_target.write_bytes(data)  # a hostile worker links outside its scratch
            try:
                os.symlink(self.link_target, target)
            except OSError:
                os.link(self.link_target, target)
        else:
            target.write_bytes(data)
        claim = WorkerEventRecord(1, request.job_id, request.attempt_id, request.stage_id, 1,
                                  "artifact", {"slot": slot, "bytes": len(data),
                                               "sha256": hashlib.sha256(data).hexdigest()})
        ref = AdmittedWorkerRef("ctx", "npipe:////./pipe/docker_engine", "b" * 64, "inst", "j", "a")
        return WorkerResult(ref, True, 0, (claim,))

    async def stop(self, reason: Any) -> bool:
        self.stops.append(reason)
        self.stopped = True
        return True


class FakeModels:
    """scheduler.ModelProvider without probe evidence (selection is covered separately)."""

    def select(self, job: Any, *, facts: Any, lower_than: Any = None) -> str:
        return "small"

    def capability(self, choice: Any) -> str:
        return "asr_cpu"

    def selection(self, choice: Any) -> tuple[str, str]:
        return "cpu", "small"

    async def prepare(self, choice: Any) -> str:
        return SNAPSHOT


def fake_hardware() -> Any:
    gone = _measure(None, FieldStatus.UNKNOWN, ReasonCode.GPU_UNAVAILABLE, "host_nvidia_smi")
    return SimpleNamespace(sample=lambda: SimpleNamespace(
        host_ram_used=_measure(8 * GIB), host_cpu_percent=_measure(12.5),
        gpu_name=gone, gpu_utilization_percent=gone, gpu_vram_used=gone))


def hardware_evidence(ref: Any, compute: Any, *, age: float, ram: tuple[int, int, int]) -> Any:
    from podmuncher.hardware import HardwareEvidence

    def ok(value: Any, source: str = "fake") -> Any:
        return _measure(value, source=source, age=age)

    gone = _measure(None, FieldStatus.UNKNOWN, ReasonCode.GPU_UNAVAILABLE, "host_nvidia_smi", age)
    limit, current, available = ram
    return HardwareEvidence(
        host_cpu_percent=ok(12.5), host_cpu_cores=ok(8), host_ram_total=ok(32 * GIB),
        host_ram_available=ok(16 * GIB), host_ram_used=ok(16 * GIB), worker_cpu_percent=ok(1.0),
        worker_cpu_raw_percent=ok(1.0), worker_cpu_limit_cores=ok(1.0), worker_ram_limit=ok(limit),
        worker_ram_current=ok(current), worker_ram_available=ok(available),
        worker_gpu_visible=ok(ref.gpu_visible), gpu_name=gone, gpu_driver=gone, gpu_vram_total=gone,
        gpu_vram_free=gone, gpu_vram_used=gone, gpu_utilization_percent=gone,
        runtime_compute_types=ok(list(compute), "admitted_runtime"))


class FakeProbeRun:
    """One single-use probe worker: started -> artifact -> hold -> completed; always reaped."""

    def __init__(self, owner: FakeProbeLaunch, mounts: Any, request: Any, on_event: Any) -> None:
        self.owner, self.mounts, self.request, self.on_event = owner, mounts, request, on_event
        self.reference: Any = None
        self.task: asyncio.Task | None = None
        self.obsoleted = self.closed = False
        self.ended_at: float | None = None

    def obsolete(self) -> None:
        self.obsoleted = True

    async def close(self) -> None:
        self.closed = True

    def _event(self, sequence: int, name: str, payload: dict) -> Any:
        from podmuncher.protocol import WorkerEventRecord

        r = self.request
        return WorkerEventRecord(1, r.job_id, r.attempt_id, r.stage_id, sequence, name, payload)

    async def run(self) -> Any:
        from podmuncher.hardware import AdmittedWorkerRef
        from podmuncher.runtime import RuntimeFailure, WorkerResult

        request = self.request
        try:
            self.reference = AdmittedWorkerRef(
                "desktop-linux", "npipe:////./pipe/docker_engine",
                hashlib.sha256(request.stage_id.encode()).hexdigest(), self.owner.ctx.instance,
                request.job_id, request.attempt_id)
            self.on_event(self._event(1, "started", {}))
            slot, name = next(iter(request.assigned_slots.items()))
            data = json.dumps(self.owner.document).encode()
            Path(self.mounts.scratch, name).write_bytes(data)
            claim = {"slot": slot, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            artifact = self._event(2, "artifact", claim)
            self.on_event(artifact)
            end = time.monotonic() + self.owner.hold
            while time.monotonic() < end and not self.obsoleted:
                await asyncio.sleep(0.01)
            if self.obsoleted:
                raise RuntimeFailure(ReasonCode.CANCELLED)
            self.on_event(self._event(3, "completed", {}))
            return WorkerResult(self.reference, True, 0, (artifact,))
        finally:
            self.ended_at = time.monotonic()


class FakeSampler:
    def __init__(self, owner: FakeProbeLaunch, reference: Any, adapter: Any) -> None:
        self.owner, self.reference, self.adapter = owner, reference, adapter

    def sample(self) -> Any:
        from podmuncher.domains import Backend

        self.owner.samples += 1
        evidence = self.adapter.compute_types(Backend.CPU, deadline=time.monotonic() + 1.0)
        return hardware_evidence(self.reference, evidence.compute_types, age=self.owner.stale,
                                 ram=self.owner.ram)


class FakeProbeLaunch:
    """The probe launch port (``bootstrap._DockerLaunch`` shape) without Docker."""

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx
        self.document: Any = dict(PROBE_DOCUMENT)
        self.hold, self.stale, self.samples = 0.3, 0.0, 0
        self.ram = (4 * GIB, 100 * MIB, 3800 * MIB)
        self.launches: list[FakeProbeRun] = []

    async def verify_image(self, image: str) -> None:
        return None

    async def launch(self, policy: Any, mounts: Any, request: Any, on_event: Any, *,
                     byte_limit: int, timeout: float, growth: Any) -> FakeProbeRun:
        run = FakeProbeRun(self, mounts, request, on_event)
        run.task = asyncio.get_running_loop().create_task(run.run())
        self.launches.append(run)
        return run

    def sampler(self, reference: Any, adapter: Any) -> FakeSampler:
        return FakeSampler(self, reference, adapter)


class FakeAcquire:
    """``ModelCache.acquire`` stand-in: a snapshot directory per model."""

    def __init__(self, root: Path) -> None:
        self.root, self.calls = root, []

    def __call__(self, model: Any) -> Any:
        self.calls.append(model.artifact.model_id)
        path = self.root / model.artifact.model_id
        path.mkdir(parents=True, exist_ok=True)
        return SimpleNamespace(snapshot_id=SNAPSHOT, path=path)


def _no_manifest_provisioning(ctx: Any) -> Any:
    return bootstrap._Provisioning(replace(ctx, manifest=None), fake_hardware())


def make_seams(env: SimpleNamespace, **overrides: Any) -> tuple[bootstrap.Seams, dict]:
    made: dict[str, Any] = {"docker": SwitchDockerStorage(env.roots["state"]),
                            "volumes": SwitchVolumes(), "containers": FakeContainers()}

    def launcher(ctx: Any) -> FakeLauncher:
        made["launcher"] = FakeLauncher(ctx)
        return made["launcher"]

    def probe(ctx: Any) -> FakeProbeLaunch:
        made["probe"] = FakeProbeLaunch(ctx)
        return made["probe"]

    def acquire(ctx: Any) -> FakeAcquire:
        made["acquire"] = FakeAcquire(env.roots["cache"].parent / "fake-snapshots")
        return made["acquire"]

    seams = bootstrap.Seams(volumes=made["volumes"], docker_storage=made["docker"],
                            containers=made["containers"], launcher=launcher,
                            models=lambda ctx: FakeModels(), hardware=fake_hardware(),
                            provisioning=_no_manifest_provisioning, probe=probe,
                            acquire=acquire, quiesce_timeout=5.0)
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
            with contextlib.suppress(ClientError):
                self.stop()
            self._thread.join(60)
        sys.modules.pop(FACTORY_MODULE, None)


def wait_job(client: Any, job_id: str, states: set[str] = TERMINAL, timeout: float = 30.0) -> dict:
    deadline = time.monotonic() + timeout
    while True:
        job = client.get_job(job_id)["job"]
        if job["state"] in states:
            return job
        assert time.monotonic() < deadline, f"job stuck in {job['state']}"
        time.sleep(0.05)


def wait_event(owner: dict, key: str, timeout: float = 20) -> None:
    deadline = time.monotonic() + timeout
    while not owner.get(key, threading.Event()).is_set():
        assert time.monotonic() < deadline, f"{key} never started"
        time.sleep(0.02)


def refused(callback: Any) -> ClientError:
    with pytest.raises(ClientError) as raised:
        callback()
    return raised.value


def available(job: dict) -> list[str]:
    return sorted(a["slot"] for a in job["artifacts"] if a["available"])


def published(env: SimpleNamespace, job_id: str) -> list[str]:
    job_dir = env.roots["output"] / f"podmuncher-{job_id}"
    return sorted(p.name for p in job_dir.iterdir()) if job_dir.is_dir() else []


# -------------------------------------------------------------------------- outcomes


OUTCOMES = [
    # (case, transcript, video, has_audio, has_video, fail_op, state, reason, slots)
    ("transcript_ok", True, False, True, False, None, "completed", "ok", TRANSCRIPT_SLOTS),
    ("combined_ok", True, True, True, True, None, "completed", "ok",
     sorted(["original_video", *TRANSCRIPT_SLOTS])),
    ("video_only_silent", False, True, False, True, None, "completed", "ok", ["original_video"]),
    ("combined_silent", True, True, False, True, None, "partial", "no_audio", ["original_video"]),
    ("transcript_no_audio", True, False, False, True, None, "failed", "no_audio", []),
    ("audio_only_wants_video", False, True, True, False, None, "failed", "no_video", []),
    ("combined_asr_fails", True, True, True, True, "transcribe", "partial",
     "worker_exit_unexpected", ["original_video"]),
    ("transcript_asr_fails", True, False, True, False, "transcribe", "failed",
     "worker_exit_unexpected", []),
]


@pytest.mark.parametrize("case,transcript,video,audio,has_video,fail,state,reason,slots",
                         OUTCOMES, ids=[c[0] for c in OUTCOMES])
def test_outcome_matrix(tmp_path: Path, case: str, transcript: bool, video: bool, audio: bool,
                        has_video: bool, fail: str | None, state: str, reason: str,
                        slots: list[str]) -> None:
    """completed / partial / failed per spec.md:140 through the composed service."""
    env = make_env(tmp_path)
    seams, made = make_seams(env)
    with LoopbackService(env, seams) as svc:
        client = svc.client()
        launcher: FakeLauncher = made["launcher"]
        launcher.facts.update(has_audio=audio, has_video=has_video)
        if fail:
            launcher.fail[fail] = ReasonCode.WORKER_EXIT_UNEXPECTED
        job_id = client.submit(job_request(env, transcript=transcript, video=video),
                               str(uuid.uuid4()))["job"]["job_id"]
        job = wait_job(client, job_id)
        assert (job["state"], job["reason_code"]) == (state, reason), job
        assert available(job) == slots
        files = published(env, job_id)
        if slots:  # bounded: see test_terminal_state_is_never_visible_before_the_sidecar
            deadline = time.monotonic() + 5
            while "podmuncher-manifest.json" not in files and time.monotonic() < deadline:
                time.sleep(0.05)
                files = published(env, job_id)
            assert "podmuncher-manifest.json" in files
        if "original_video" in slots:
            video_file = next(n for n in files if n.endswith(".mp4"))
            assert (env.roots["output"] / f"podmuncher-{job_id}" / video_file).read_bytes() \
                == env.source.read_bytes()
        if not transcript or not audio:  # video-only or silent media never provisions ASR
            assert "transcribe" not in launcher.ops
        assert env.source.exists()  # the original source is preserved
        assert svc.stop() == service.EXIT_OK


@pytest.mark.parametrize("transcript", [False, True], ids=["video_only", "combined_silent"])
def test_terminal_state_is_never_visible_before_the_sidecar(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, transcript: bool) -> None:
    """completed/partial must mean every requested artifact is published and listed."""
    from podmuncher.artifacts import ArtifactError

    gate, entered = threading.Event(), threading.Event()

    def failing_manifest(self: Any, *args: Any, **kwargs: Any) -> Any:
        entered.set()
        gate.wait(10)
        raise ArtifactError(ReasonCode.PUBLICATION_FAILED)

    monkeypatch.setattr(bootstrap._Publisher, "write_manifest", failing_manifest)
    env = make_env(tmp_path)
    seams, made = make_seams(env)
    with LoopbackService(env, seams) as svc:
        client = svc.client()
        made["launcher"].facts.update(has_audio=False)  # the video publication is the last one
        job_id = client.submit(job_request(env, transcript=transcript, video=True),
                               str(uuid.uuid4()))["job"]["job_id"]
        assert entered.wait(20)
        during = client.get_job(job_id)["job"]["state"]
        gate.set()
        after = wait_job(client, job_id)
        assert svc.stop() == service.EXIT_OK
    assert during not in ("completed", "partial"), "final state committed before the sidecar"
    assert after["state"] not in ("completed", "partial") or \
        "podmuncher-manifest.json" in published(env, job_id)


# --------------------------------------------------------------------------- storage


@pytest.mark.parametrize("mode,reason", [("stale", "storage_stale"), ("unknown", "storage_unknown")])
def test_stale_or_unknown_capacity_blocks_the_next_stage(tmp_path: Path, mode: str,
                                                         reason: str) -> None:
    env = make_env(tmp_path)
    seams, made = make_seams(env)
    with LoopbackService(env, seams) as svc:
        client = svc.client()
        launcher: FakeLauncher = made["launcher"]
        launcher.gates["probe"] = threading.Event()
        job_id = client.submit(job_request(env), str(uuid.uuid4()))["job"]["job_id"]
        wait_event(launcher.entered, "probe")
        made["docker"].mode = mode  # capacity turns stale/unknown while probing
        launcher.gates["probe"].set()
        job = wait_job(client, job_id)
        assert (job["state"], job["reason_code"]) == ("blocked", reason), job
        assert launcher.ops == ["probe"]  # no next stage was started
        assert available(job) == []
        made["docker"].mode = "ok"  # recheck is explicit; no background retry happened
        assert client.get_job(job_id)["job"]["state"] == "blocked"
        assert client.recheck(job_id)["ready_to_retry"] is True
        assert svc.stop() == service.EXIT_OK


def test_external_disk_drop_mid_job(tmp_path: Path) -> None:
    """A guard trip mid-transcribe: blocked without output, partial once video is published."""
    env = make_env(tmp_path)
    seams, made = make_seams(env)
    with LoopbackService(env, seams) as svc:
        client = svc.client()
        launcher: FakeLauncher = made["launcher"]
        launcher.fail["transcribe"] = ReasonCode.STORAGE_GUARD_TRIPPED
        only = client.submit(job_request(env), str(uuid.uuid4()))["job"]["job_id"]
        job = wait_job(client, only)
        assert (job["state"], job["reason_code"]) == ("blocked", "storage_guard_tripped"), job
        combined = client.submit(job_request(env, video=True), str(uuid.uuid4()))["job"]["job_id"]
        job = wait_job(client, combined)
        assert (job["state"], job["reason_code"]) == ("partial", "storage_guard_tripped"), job
        assert available(job) == ["original_video"]
        assert svc.stop() == service.EXIT_OK


def test_reservation_exhaustion_blocks_before_any_worker(tmp_path: Path) -> None:
    env = make_env(tmp_path)
    seams, made = make_seams(env)
    with LoopbackService(env, seams) as svc:
        client = svc.client()
        made["volumes"].mode = "full"  # 1 MiB free cannot cover the job's peak growth
        job_id = client.submit(job_request(env, video=True), str(uuid.uuid4()))["job"]["job_id"]
        job = wait_job(client, job_id)
        assert (job["state"], job["reason_code"]) == ("blocked", "storage_insufficient"), job
        assert made["launcher"].ops == [] and published(env, job_id) == []
        made["volumes"].mode = "ok"
        assert svc.stop() == service.EXIT_OK


def test_export_cap_reached(tmp_path: Path) -> None:
    env = make_env(tmp_path, export_bytes=4096)
    seams, made = make_seams(env)
    with LoopbackService(env, seams) as svc:
        client = svc.client()
        made["launcher"].pad = 8192  # transcript record larger than the export cap
        job_id = client.submit(job_request(env), str(uuid.uuid4()))["job"]["job_id"]
        job = wait_job(client, job_id)
        assert (job["state"], job["reason_code"]) == ("failed", "export_limit_reached"), job
        assert available(job) == [] and published(env, job_id) == []
        assert svc.stop() == service.EXIT_OK


# -------------------------------------------------------------------------- recovery


async def _start(env: SimpleNamespace, seams: bootstrap.Seams) -> bootstrap.Composition:
    composition = bootstrap.Composition(env.roots["state"], environ=env.env, seams=seams)
    await composition.startup()
    return composition


async def _crash(composition: bootstrap.Composition) -> None:
    """Abandon the process state without a graceful stop (no 'stopped' intent is written)."""
    tasks = [t for t in (composition._pump, *composition._tasks) if t is not None]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await composition._close_resources()


def _submit(composition: bootstrap.Composition, env: SimpleNamespace) -> Any:
    from podmuncher.requests import JobRequest

    models, languages = composition.registries()
    request = JobRequest.from_dict(job_request(env), verified_models=models,
                                   verified_languages=languages)
    return composition.actions.submit(request, str(uuid.uuid4()))


def test_crash_mid_transcribe_reconciles_to_interrupted(tmp_path: Path) -> None:
    from podmuncher.recovery import ContainerRef

    env = make_env(tmp_path)

    async def scenario() -> None:
        seams, made = make_seams(env)
        first = await _start(env, seams)
        launcher: FakeLauncher = made["launcher"]
        launcher.gates["transcribe"] = threading.Event()
        job, _ = await _submit(first, env)
        while not launcher.entered.get("transcribe", threading.Event()).is_set():
            await asyncio.sleep(0.02)
        instance = first.ctx.instance
        await _crash(first)  # the service dies while the ASR worker is running
        survivor = ContainerRef("c" * 64, {"podmuncher.instance": instance})
        containers = FakeContainers(refs=(survivor,))
        seams2, _ = make_seams(env, containers=containers)
        second = await _start(env, seams2)
        try:
            restored = second.ctx.repo.get_job(job.job_id)
            assert restored.state is JobState.INTERRUPTED
            assert restored.reason_code is ReasonCode.SERVICE_LOST
            assert ("list", instance) in containers.log and containers.refs == []
            assert second.ctx.repo.lease_holders() == ()
        finally:
            await second.shutdown()

    asyncio.run(scenario())


def test_unreachable_docker_after_crash_refuses_start(tmp_path: Path) -> None:
    env = make_env(tmp_path)

    async def scenario() -> None:
        seams, made = make_seams(env)
        first = await _start(env, seams)
        made["launcher"].gates["probe"] = threading.Event()
        await _submit(first, env)
        while not made["launcher"].entered.get("probe", threading.Event()).is_set():
            await asyncio.sleep(0.02)
        await _crash(first)
        seams2, _ = make_seams(env, containers=FakeContainers(unreachable=True))
        blocked = bootstrap.Composition(env.roots["state"], environ=env.env, seams=seams2)
        with pytest.raises(service.ServiceError) as raised:
            await blocked.startup()
        assert raised.value.reason_code is ReasonCode.WRITER_UNRESOLVED

    asyncio.run(scenario())


def test_clean_stop_lets_the_next_start_skip_docker(tmp_path: Path) -> None:
    env = make_env(tmp_path)

    async def scenario() -> None:
        seams, _ = make_seams(env)
        first = await _start(env, seams)
        assert await first.shutdown() == (True, "service_stopped")
        containers = FakeContainers(unreachable=True)  # Docker is gone, and not needed
        seams2, _ = make_seams(env, containers=containers)
        second = await _start(env, seams2)
        try:
            assert containers.log == []  # no earlier instance to reconcile
        finally:
            await second.shutdown()

    asyncio.run(scenario())


# -------------------------------------------------------------------------- security


def test_hostile_text_is_escaped_in_plain_output_and_kept_in_json(
        capsys: pytest.CaptureFixture) -> None:
    """[CWE-150] terminal escapes, bidi and Rich markup through the real CLI handlers."""
    job_id = str(uuid.uuid4())
    job = {"job_id": job_id, "state": "completed", "reason_code": "ok", "title": HOSTILE,
           "artifacts": [{"slot": "transcript_txt", "basename": HOSTILE, "available": True}]}

    class Client:
        def get_job(self, wanted: str) -> dict:
            assert wanted == job_id
            return {"job": job}

        def close(self) -> None:
            pass

    access = SimpleNamespace(connect=lambda: Client(), ensure=lambda: Client())
    try:
        bootstrap.client_startup({})  # the final presentation adapter
        assert cli.main(["job", job_id], access=access, environ={}) == 0
        plain = capsys.readouterr().out
        assert cli.main(["job", job_id, "--json"], access=access, environ={}) == 0
        document = json.loads(capsys.readouterr().out)
    finally:
        cli.set_registry_provider(None)
        cli.configure_presentation(None)
    for raw in ("\x1b", "\u202e", "\x07"):
        assert raw not in plain
    assert "\\x1b" in plain and "\\u202e" in plain and "\\x07" in plain
    assert "[bold]" not in plain.replace("\\[bold\\]", "")
    assert document["job"]["title"] == HOSTILE
    assert document["job"]["artifacts"][0]["basename"] == HOSTILE


def test_worker_output_link_outside_scratch_is_rejected(tmp_path: Path) -> None:
    """[CWE-22] a worker that links its assigned output to a file outside scratch."""
    env = make_env(tmp_path)
    seams, made = make_seams(env)
    outside = tmp_path / "outside-secret.json"
    with LoopbackService(env, seams) as svc:
        client = svc.client()
        made["launcher"].link_target = outside
        job_id = client.submit(job_request(env), str(uuid.uuid4()))["job"]["job_id"]
        job = wait_job(client, job_id)
        assert (job["state"], job["reason_code"]) == ("failed", "artifact_rejected"), job
        assert published(env, job_id) == []
        assert svc.stop() == service.EXIT_OK
    assert outside.exists()  # never moved or published


def test_traversal_names_are_rejected_by_protocol_and_verification(tmp_path: Path) -> None:
    """[CWE-22] worker-chosen names never resolve outside the assigned scratch."""
    from podmuncher.artifacts import ArtifactError, verify_artifact
    from podmuncher.protocol import ProtocolError, WorkerRequest

    base = _worker_request()
    for bad in ("../escape.bin", "..\\escape.bin", "/etc/passwd", "C:\\x.bin"):
        with pytest.raises(ProtocolError):
            WorkerRequest.from_dict({**base.to_dict(), "operation": "probe",
                                     "input_basename": bad})
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (tmp_path / "escape.bin").write_bytes(b"{}")
    for name in ("../escape.bin", "..", "escape.bin/.."):
        with pytest.raises(ArtifactError) as raised:
            verify_artifact(str(scratch), {WorkerSlot.TRANSCRIPT_SEGMENTS: name},
                            WorkerSlot.TRANSCRIPT_SEGMENTS, 2,
                            hashlib.sha256(b"{}").hexdigest(), writers_reaped=True, max_bytes=100)
        assert raised.value.reason_code is ReasonCode.ARTIFACT_REJECTED


def _worker_request(**changes: Any) -> Any:
    from podmuncher.config import Limits
    from podmuncher.protocol import WorkerRequest

    data = {"schema_version": 1, "job_id": str(uuid.uuid4()), "attempt_id": str(uuid.uuid4()),
                "stage_id": str(uuid.uuid4()), "capability": "asr_cpu", "operation": "probe",
                "assigned_slots": {"probe_metadata": "probe_metadata-" + "a" * 32 + ".bin"},
                "source": None, "outputs": {"transcript": True, "video": False, "video_quality": "1080p"},
                "input_basename": None, "model_snapshot_id": None, "limits": asdict(Limits()),
                "selection": {"device": "cpu", "profile": "balanced", "model": "auto", "language": "auto"}}
    data.update(changes)
    return WorkerRequest.from_dict(data)


class FakeDockerProcess:
    def __init__(self, out: bytes, code: int = 0) -> None:
        self.pid = 4242
        self.returncode = code
        self.stdout, self.stderr, self.stdin = io.BytesIO(out), io.BytesIO(b""), io.BytesIO()

    def poll(self) -> int:
        return self.returncode

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode


class FakeDockerCLI:
    """The Docker CLI process seam: a worker whose stdout is one oversize event line."""

    CID = "a" * 64
    IMAGE = "sha256:" + "b" * 64

    def __init__(self, payload: bytes) -> None:
        self.payload, self.exists, self.labels, self.calls = payload, False, {}, []

    def identity(self, pid: int) -> Any:
        return SimpleNamespace(create_time=lambda: float(pid))

    def __call__(self, argv: tuple, **kwargs: Any) -> FakeDockerProcess:
        assert kwargs["shell"] is False
        command = argv[3:]
        self.calls.append(command[:2])
        out, code = b"", 0
        if command[:2] == ("context", "inspect"):
            out = json.dumps([{"Name": "desktop-linux", "Endpoints": {"docker": {
                "Host": "npipe:////./pipe/dockerDesktopLinuxEngine"}}}]).encode()
        elif command[0] == "create":
            self.exists = True
            self.labels = dict(command[i + 1].split("=", 1)
                               for i, v in enumerate(command) if v == "--label")
            out = (self.CID + "\n").encode()
        elif command[:2] == ("image", "inspect"):
            out = json.dumps([{"Id": self.IMAGE}]).encode()
        elif command[0] == "ps":
            out = (self.CID + "\n").encode() if self.exists else b""
        elif command[:2] == ("container", "inspect"):
            out = json.dumps([{"Id": self.CID, "Config": {"Labels": self.labels},
                               "HostConfig": {"DeviceRequests": None},
                               "State": {"Running": False, "Restarting": False}}]).encode()
            code = 0 if self.exists else 1
        elif command[0] == "start":
            out = self.payload
        elif command[0] == "wait":
            out = b"0\n"
        elif command[0] == "rm":
            self.exists = False
        return FakeDockerProcess(out, code)


def test_oversize_worker_event_is_rejected_and_the_container_reaped(tmp_path: Path) -> None:
    from podmuncher.admission import AdmissionPlan
    from podmuncher.guards import CancellationCoordinator, StorageSample, WriteGuard
    from podmuncher.runtime import (
        DockerRuntime,
        DockerTarget,
        NativeDockerCLI,
        RuntimeFailure,
        WorkerMounts,
        WorkerPolicy,
    )

    request = _worker_request()
    started = json.dumps({"schema_version": 1, "job_id": request.job_id,
                          "attempt_id": request.attempt_id, "stage_id": request.stage_id,
                          "sequence": 0, "event": "started", "payload": {}}).encode()
    factory = FakeDockerCLI(started + b"\n" + b"x" * 70000 + b"\n")

    class Lease:
        held = False
        released = 0

        def try_acquire(self, owner: str, plan: Any) -> bool:
            self.held = True
            return True

        def release(self, owner: str, *, writers_reaped: bool) -> None:
            self.held = False
            self.released += 1

    async def sample() -> Any:
        return StorageSample(AdmissionPlan(_measure(True), (), ()), time.monotonic())

    lease = Lease()
    guard = WriteGuard(CancellationCoordinator("matrix-op", lease, stop_grace_seconds=0.2,
                                               reap_timeout_seconds=0.2,
                                               callback_timeout_seconds=0.2),
                       sample, byte_limit=65536, timeout_seconds=10)
    target = DockerTarget("desktop-linux", "npipe:////./pipe/dockerDesktopLinuxEngine")
    docker = NativeDockerCLI(target, factory=factory, identity=factory.identity, environ={})
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    policy = WorkerPolicy("asr_cpu", FakeDockerCLI.IMAGE, ("python", "-I", "-m", "w"), 0, 0,
                          lambda: True, development_image=True, allow_pull=False)
    runtime = DockerRuntime(docker, guard, instance="matrix-instance", policy=policy,
                            mounts=WorkerMounts(scratch))
    with pytest.raises(RuntimeFailure) as raised:
        asyncio.run(runtime.run(request))
    assert raised.value.reason_code in (ReasonCode.EVENT_LIMIT_REACHED,
                                        ReasonCode.WORKER_PROTOCOL_MISMATCH)
    assert not factory.exists and lease.released == 1 and not lease.held  # reaped, released


def _raw(port: int, path: str, headers: dict[str, str]) -> tuple[int, dict]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        connection.putrequest("GET", path, skip_host=True, skip_accept_encoding=True)
        for name, value in headers.items():
            connection.putheader(name, value)
        connection.endheaders()
        response = connection.getresponse()
        body = response.read()
        try:
            return response.status, json.loads(body or b"{}")
        except ValueError:
            return response.status, {}
    finally:
        connection.close()


def test_loopback_auth_refusals(tmp_path: Path) -> None:
    env = make_env(tmp_path)
    seams, _ = make_seams(env)
    with LoopbackService(env, seams) as svc:
        record = service.discover(svc.state_dir)
        host = f"127.0.0.1:{record.port}"
        good = {"Host": host, "X-Podmuncher-Schema": "1",
                "Authorization": f"Bearer {record.credential}"}
        status, _ = _raw(record.port, "/v1/identity", good)
        assert status == 200
        cases = {
            "no_token": ({k: v for k, v in good.items() if k != "Authorization"}, 401,
                         "unauthorized"),
            "wrong_token": ({**good, "Authorization": "Bearer " + "0" * 43}, 401, "unauthorized"),
            "wrong_host": ({**good, "Host": f"localhost:{record.port}"}, 403, "invalid_host"),
            "origin_set": ({**good, "Origin": "http://127.0.0.1"}, 403,
                           "browser_origin_rejected"),
        }
        for name, (headers, want, code) in cases.items():
            status, body = _raw(record.port, "/v1/identity", headers)
            assert status == want, name
            assert body.get("error", {}).get("reason_code") == code, (name, body)
            assert record.credential not in json.dumps(body)
        assert svc.stop() == service.EXIT_OK


# ----------------------------------------------------------------------------- model


def test_auto_oom_falls_back_once_to_a_smaller_model(tmp_path: Path) -> None:
    env = make_env(tmp_path, doc=manifest_doc(("base", "tiny"), memory=6 * GIB))
    seams, made = make_seams(env, models=None)  # the real _ManifestModels over the probe seam
    with LoopbackService(env, seams) as svc:
        client = svc.client()
        made["probe"].ram = (8 * GIB, 100 * MIB, 7500 * MIB)
        launcher: FakeLauncher = made["launcher"]
        launcher.oom = 1
        job = wait_job(client, client.submit(job_request(env), str(uuid.uuid4()))["job"]["job_id"])
        assert job["state"] == "completed", job
        assert launcher.selections == [("cpu", "base"), ("cpu", "tiny")]
        assert all(r.ended_at is not None and r.closed for r in made["probe"].launches)
        launcher.oom, launcher.selections = 2, []
        job = wait_job(client, client.submit(job_request(env), str(uuid.uuid4()))["job"]["job_id"])
        assert (job["state"], job["reason_code"]) == ("failed", "model_oom"), job
        assert launcher.selections == [("cpu", "base"), ("cpu", "tiny")]  # exactly one fallback
        assert svc.stop() == service.EXIT_OK


def test_stale_probe_evidence_is_envelope_unvalidated_and_reaped(tmp_path: Path) -> None:
    env = make_env(tmp_path)
    seams, made = make_seams(env, models=None)
    with LoopbackService(env, seams) as svc:
        client = svc.client()
        made["probe"].stale = 10.0  # every hardware sample is older than select_model accepts
        job = wait_job(client, client.submit(job_request(env), str(uuid.uuid4()))["job"]["job_id"])
        assert job["reason_code"] == "envelope_unvalidated", job
        assert job["state"] in ("failed", "blocked"), job
        assert "transcribe" not in made["launcher"].ops
        runs = made["probe"].launches
        assert runs and all(r.ended_at is not None and r.closed for r in runs)  # reaped
        assert not svc.composition.probe.running and svc.composition.probe.writers_reaped
        assert svc.stop() == service.EXIT_OK


def test_no_manifest_is_auto_only_and_refuses_named_models(tmp_path: Path) -> None:
    from argparse import Namespace

    env = make_env(tmp_path, manifest=False)
    args = Namespace(input=str(env.source), device=None, profile=None, model="small",
                     language=None, output=None, video=False, no_transcript=False)
    try:
        bootstrap.client_startup(env.env)
        with pytest.raises(DomainValidationError):
            cli.build_request(args, env.env)
        assert cli.build_request(replace_ns(args, model=None), env.env)["selection"]["model"] \
            == "auto"
    finally:
        cli.set_registry_provider(None)
        cli.configure_presentation(None)
    seams, made = make_seams(env)
    with LoopbackService(env, seams) as svc:
        client = svc.client()
        assert refused(lambda: client.submit(job_request(env, model="small"),
                                             str(uuid.uuid4()))).reason_code \
            is ReasonCode.INVALID_INPUT
        job = wait_job(client, client.submit(job_request(env), str(uuid.uuid4()))["job"]["job_id"])
        assert (job["state"], job["reason_code"]) == ("blocked", "service_unavailable"), job
        assert made["launcher"].ops == []  # nothing pulled, probed or run without a manifest
        assert svc.stop() == service.EXIT_OK


def replace_ns(ns: Any, **changes: Any) -> Any:
    from argparse import Namespace

    return Namespace(**{**vars(ns), **changes})


# -------------------------------------------------------------------- audit, idempotency


def test_audit_loss_blocks_submit_but_not_cancel_or_stop(tmp_path: Path) -> None:
    """[CWE-778] a failing durable audit sink stops new work; cancel and stop still work."""
    env = make_env(tmp_path)
    seams, made = make_seams(env)
    with LoopbackService(env, seams) as svc:
        client = svc.client()
        launcher: FakeLauncher = made["launcher"]
        launcher.gates["probe"] = threading.Event()
        active = client.submit(job_request(env), str(uuid.uuid4()))["job"]["job_id"]
        wait_event(launcher.entered, "probe")
        hub = svc.composition.ctx.hub

        def broken(line: bytes) -> None:
            raise OSError("disk gone")

        hub._audit_sink.write = broken
        error = refused(lambda: client.submit(job_request(env), str(uuid.uuid4())))
        assert error.reason_code is ReasonCode.AUDIT_LOSS and error.status == 503
        assert refused(lambda: client.retry(active, str(uuid.uuid4()))).reason_code \
            is ReasonCode.AUDIT_LOSS
        assert refused(lambda: client.clone(active, job_request(env), str(uuid.uuid4()))) \
            .reason_code is ReasonCode.AUDIT_LOSS
        assert client.cancel(active)["status"] == "cancelled"
        assert svc.stop() == service.EXIT_OK


def test_same_key_replays_submit_retry_and_clone(tmp_path: Path) -> None:
    env = make_env(tmp_path)
    seams, made = make_seams(env)
    with LoopbackService(env, seams) as svc:
        client = svc.client()
        launcher: FakeLauncher = made["launcher"]
        key = str(uuid.uuid4())
        first = client.submit(job_request(env), key)
        again = client.submit(job_request(env), key)
        assert first["replayed"] is False and again["replayed"] is True
        assert again["job"]["job_id"] == first["job"]["job_id"]
        assert refused(lambda: client.submit(job_request(env, video=True), key)).reason_code \
            is ReasonCode.CONFLICT  # one key never names two requests
        job_id = first["job"]["job_id"]
        assert wait_job(client, job_id)["state"] == "completed"

        launcher.fail["transcribe"] = ReasonCode.WORKER_EXIT_UNEXPECTED
        failed_id = client.submit(job_request(env), str(uuid.uuid4()))["job"]["job_id"]
        assert wait_job(client, failed_id)["state"] == "failed"
        del launcher.fail["transcribe"]
        retry_key = str(uuid.uuid4())
        retried = client.retry(failed_id, retry_key)
        replay = client.retry(failed_id, retry_key)
        assert retried["replayed"] is False and replay["replayed"] is True
        assert replay["attempt_id"] == retried["attempt_id"]
        assert wait_job(client, failed_id)["state"] == "completed"

        clone_key = str(uuid.uuid4())
        clone = client.clone(job_id, job_request(env, video=True), clone_key)["job"]
        assert client.clone(job_id, job_request(env, video=True), clone_key)["job"]["job_id"] \
            == clone["job_id"] != job_id
        assert wait_job(client, clone["job_id"])["state"] == "completed"
        assert len(client.list_jobs(10)["jobs"]) == 3  # replays never created a job
        assert svc.stop() == service.EXIT_OK


# ---------------------------------------------------------------------------- domains


def test_accepted_domains_reject_contract_negatives(tmp_path: Path) -> None:
    from podmuncher import domains

    registry = json.loads((ROOT / "Docs" / "contract-domains.json").read_text(encoding="utf-8"))
    negatives = registry["negative_examples"]
    assert negatives == ["__invalid__", ""]
    exercised = {"capability": Capability, "artifact_slot": ArtifactSlot, "job_state": JobState,
                 "service_state": ServiceState, "field_status": FieldStatus,
                 "worker_slot": WorkerSlot, "reason_code": ReasonCode, "stage": Stage,
                 "output_mode": domains.OutputMode, "backend": domains.Backend,
                 "model_tier": domains.ModelTier}
    for name, kind in exercised.items():
        for value in registry[name]:
            kind.parse(value)
        for value in negatives:
            with pytest.raises(DomainValidationError):
                kind.parse(value)
    env = make_env(tmp_path)
    seams, _ = make_seams(env)
    with LoopbackService(env, seams) as svc:
        client = svc.client()
        for value in negatives:
            for field_name, field_value in (("device", value), ("profile", value)):
                request = job_request(env)
                request["selection"][field_name] = field_value
                assert refused(lambda r=request: client.submit(r, str(uuid.uuid4()))).status \
                    in (400, 422)
            assert refused(lambda v=value: client.setup_plan([v], {})).reason_code \
                is ReasonCode.INVALID_INPUT
            assert refused(lambda v=value: client.cancel(v)).status in (400, 404)
        assert svc.stop() == service.EXIT_OK
