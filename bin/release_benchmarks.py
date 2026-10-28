#!/usr/bin/env python3
"""Release benchmarks: measure the current ASR candidates and apply the evidence to release.json.

Subcommands:

``run --model-root DIR --out FILE [--manifest PATH]``
    Generates synthetic speech fixtures inside the current media image (network none, ffmpeg flite voice
    ``slt`` from the original text below), runs the benchmark matrix through the owned ``DockerRuntime``
    (network none, read-only, 65532:65532, 1 CPU, 2 GiB, development image, no pull), replays one probe per
    ASR candidate and collects GPU passthrough evidence. Every record is appended as its canonical JSONL
    line to ``FILE.records.jsonl`` as it completes; the full result is written to ``FILE``.

``apply FILE [--manifest PATH]``
    Validates a ``run`` output against the manifest's current amd64 build identities, replaces the benchmark
    evidence, recomputes the amd64 tiny/base/small envelopes and the conditional blockers, rehashes the
    runtime manifests and writes the manifest through a temp file and ``os.replace``. A validation failure
    leaves the manifest untouched. ``publishable`` stays false and the long-duration blockers are never
    dropped by short fixtures.
"""
from __future__ import annotations

import argparse
import asyncio
import functools
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

REPO = Path(__file__).resolve().parents[1]
MANIFEST_REL = "src/podmuncher/manifests/release.json"
AMD64 = "linux-amd64"
GIB = 1024**3
MIB = 1024**2
SIX_HOURS = 21600
MEMORY_CAP = 2 * GIB
CPUS = 1.0
SAMPLE_BUDGET = 2.5
SERIES = "attempt-3"
OUTPUT_BYTES = 64 * MIB
HOST_GUARD_RETRIES = 2
VRAM_POLL_SECONDS = 0.25
MEASURED_MODELS = ("tiny", "base", "small")
FIXTURE_SECONDS = (60, 300, 600)
# codec name -> (file extension, ffmpeg encoder arguments)
CODECS = {
    "wav": ("wav", ("-c:a", "pcm_s16le")),
    "mp3": ("mp3", ("-c:a", "libmp3lame", "-b:a", "64k")),
    "mp4_aac": ("m4a", ("-c:a", "aac", "-b:a", "64k")),
}
FIXTURE_GENERATOR = ("Synthetic speech from an original English text (bin/release_benchmarks.py) via ffmpeg flite "
                     "(voice slt) inside the media image; no recordings")
GPU_METHOD = "docker run --gpus all on the CUDA release candidate (nvidia-smi and ctranslate2 inside the container)"
CONDITIONAL_CODES = frozenset({"gpu_evidence_missing", "long_duration_unvalidated", "worker_defect_open"})
RECORD_FIELDS = (
    "record_sha256", "series", "name", "capability", "platform", "bound_image_id", "bound_image_basis",
    "bound_to_current_build", "model_id", "codec", "fixture_seconds", "memory_cap_bytes", "cpus",
    "storage_sample_budget_seconds", "outcome", "load_seconds", "transcribe_wall_seconds", "audio_seconds",
    "real_time_factor", "peak_rss_bytes", "worker_vram_delta_bytes", "segments", "detected_language",
    "recorded_at",
)
MEASUREMENT_FIELDS = ("load_seconds", "transcribe_wall_seconds", "audio_seconds", "real_time_factor",
                      "peak_rss_bytes", "worker_vram_delta_bytes", "segments", "detected_language")
GPU_FIELDS = ("passthrough", "method", "name", "driver_version", "uuid", "memory_total_bytes", "bound_image_id",
              "container_cuda_device_count", "container_cuda_compute_types", "checked_at")
POLICY_COMPUTE = {"asr_cpu": "int8", "asr_cuda": "int8_float16"}
# Reasons raised by the host side (storage watchdog, guard, Docker, deadlines), never by the worker.
HOST_GUARD_REASONS = frozenset({
    "storage_insufficient", "storage_unknown", "storage_stale", "storage_guard_tripped", "quota_exceeded",
    "state_write_failed", "audit_loss", "docker_unavailable", "docker_context_unsupported",
    "backing_unresolved", "gpu_unavailable", "request_timeout", "cancelled", "writer_unresolved",
    "invalid_input", "unauthorized", "conflict",
})

# The matrix of the previous current-build set: every point needs at least one completed record.
MATRIX = tuple(
    [("asr_cpu", model, codec, seconds) for model in MEASURED_MODELS
     for codec, seconds in (("mp3", 60), ("mp4_aac", 60), ("wav", 60), ("mp3", 600))]
    + [("asr_cuda", model, "wav", 60) for model in MEASURED_MODELS]
)

# Original English prose written for this tool; repeated so flite renders well past 600 s.
SPEECH_TEXT = """\
On the morning the river flooded, the town woke to a sound it had never heard before. It was not the roar
of water, which everyone expected, but a soft and steady hiss, like rain falling on a field of dry grass. The
baker was the first to open his door. He stood on the step with flour on his hands and watched the street
turn slowly into a shallow canal. Bread could wait, he decided, and he went back inside to carry the sacks of
grain up to the attic.

By noon the whole valley had gathered on the hill behind the school. Children counted the boats that drifted
past, and their parents argued about whether the old bridge would hold. A retired engineer drew diagrams in
the mud with a stick, explaining how the stone arches spread the weight of the current into the banks. Nobody
fully understood him, but his calm voice made the afternoon feel shorter.

When the evening came, the water stopped rising. It lay across the meadows like a sheet of polished metal,
reflecting the first stars. Someone lit a fire, someone else found a guitar, and the long wait became a kind
of festival. Neighbours who had not spoken in years shared blankets and soup. The mayor gave a short speech
that was mostly about gratitude and partly about drainage.

Three days later the river returned to its usual path. The streets were brown with silt, the gardens were
ruined, and the library had lost its lowest shelf of atlases. Yet people remembered that week with a strange
affection. They had learned how quickly a town can become a single household, and how much patience fits
inside an ordinary person. The baker reopened on the fourth morning, and the first loaf he sold went to the
engineer, who still had mud on his boots and a new diagram folded in his pocket.
"""
SPEECH_REPEATS = 8


class ToolError(Exception):
    """A step failed; the message names the step."""


def _require(condition: object, message: str) -> None:
    if not condition:
        raise ToolError(message)


# ----------------------------------------------------------------------------- pure functions


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def record_line(record: dict) -> str:
    """The raw JSONL line of a record: canonical JSON of every field except record_sha256."""
    body = {key: value for key, value in record.items() if key != "record_sha256"}
    return canonical_bytes(body).decode("ascii")


def record_hash(record: dict) -> str:
    return hashlib.sha256(record_line(record).encode("ascii")).hexdigest()


def seal_record(fields: dict) -> dict:
    """Return the record in RECORD_FIELDS order with its record_sha256 first."""
    body = {key: fields[key] for key in RECORD_FIELDS if key != "record_sha256"}
    return {"record_sha256": record_hash(body), **body}


def record_name(capability: str, model: str, codec: str, seconds: int, memory_cap: int = MEMORY_CAP) -> str:
    return f"{capability}-{model}-{codec}-{seconds}s-{memory_cap // MIB}m"


def current_records(records: list[dict], capability: str, model_id: str | None = None) -> list[dict]:
    return [r for r in records if r["capability"] == capability and r["bound_to_current_build"]
            and (model_id is None or r["model_id"] == model_id)]


def model_envelope(records: list[dict], model_id: str) -> dict | None:
    """Envelope fields of one amd64 CPU model, as checked by tests/test_release_manifest.py:723-763.

    Returns None when the model has no completed current 2 GiB record (its envelope is left untouched).
    """
    mine = [r for r in current_records(records, "asr_cpu", model_id) if r["memory_cap_bytes"] == MEMORY_CAP]
    ok = [r for r in mine if r["outcome"] == "completed"]
    if not ok:
        return None
    failed = [r["fixture_seconds"] for r in mine if r["outcome"].startswith("worker_failure:")]
    below = [r["fixture_seconds"] for r in ok if not failed or r["fixture_seconds"] < min(failed)]
    if not below:
        raise ToolError(f"envelope {model_id}: no completed record below the first worker failure duration")
    evidence = hashlib.sha256("\n".join(sorted(r["record_sha256"] for r in mine)).encode("ascii")).hexdigest()
    return {
        "evidence_sha256": evidence,
        "codec_matrix": sorted({r["codec"] for r in ok}),
        "worker_peak_ram_bytes": max(r["peak_rss_bytes"] for r in ok),
        "validated_max_duration_seconds": max(below),
    }


def expected_conditional_blockers(records: list[dict], models: list[dict]) -> list[tuple[str, str]]:
    """Conditional blockers per tests/test_release_manifest.py:608-630, in a stable order."""
    expected = []
    cuda_ok = [r for r in current_records(records, "asr_cuda") if r["outcome"] == "completed"]
    if cuda_ok:
        if max(r["fixture_seconds"] for r in cuda_ok) < SIX_HOURS:
            expected.append(("long_duration_unvalidated", "asr_cuda"))
    else:
        expected.append(("gpu_evidence_missing", "asr_cuda"))
    if max(m.get("validated_max_duration_seconds") or 0 for m in models) < SIX_HOURS:
        expected.append(("long_duration_unvalidated", "asr_cpu"))
    for capability in ("asr_cpu", "asr_cuda"):
        if any(r["outcome"].startswith("worker_failure:") for r in current_records(records, capability)):
            expected.append(("worker_defect_open", capability))
    return expected


def _blocker_text(code: str, capability: str, records: list[dict], models: list[dict]) -> tuple[str, str]:
    if code == "long_duration_unvalidated" and capability == "asr_cpu":
        measured = max(m.get("validated_max_duration_seconds") or 0 for m in models)
        return ("long-duration benchmark", f"CPU envelopes are measured up to {measured} s; the {SIX_HOURS} s "
                "product limit is not certified from short fixtures.")
    if code == "long_duration_unvalidated":
        ok = [r for r in current_records(records, "asr_cuda") if r["outcome"] == "completed"]
        measured = max(r["fixture_seconds"] for r in ok)
        return ("long-duration benchmark", f"CUDA runs are measured up to {measured} s; no CUDA envelope is "
                "certified.")
    if code == "gpu_evidence_missing":
        return ("GPU evidence run", "No completed CUDA benchmark run exists for the current asr_cuda build.")
    return ("worker defect fix", f"A {capability} benchmark run on the current build ended in a worker failure; "
            "the worker defect must be fixed and the run repeated.")


def rebuild_blockers(blockers: list[dict], records: list[dict], models: list[dict]) -> list[dict]:
    """Keep unconditional blockers unchanged, keep/add/remove conditional ones from the evidence."""
    expected = expected_conditional_blockers(records, models)
    result = []
    for blocker in blockers:
        key = (blocker["code"], blocker["capability"])
        if blocker["code"] not in CONDITIONAL_CODES:
            result.append(dict(blocker))
        elif key in expected:
            kept = dict(blocker)
            if blocker["code"] == "long_duration_unvalidated":
                kept["detail"] = _blocker_text(*key, records, models)[1]
            result.append(kept)
    present = {(b["code"], b["capability"]) for b in result}
    for code, capability in expected:
        if (code, capability) not in present:
            resolved_by, detail = _blocker_text(code, capability, records, models)
            result.append({"code": code, "capability": capability, "resolved_by": resolved_by, "detail": detail})
    return result


def classify_failure(reason: str, worker_started: bool) -> str:
    """Outcome of a failed run: host guard failures are retried, worker failures are not."""
    if worker_started and reason not in HOST_GUARD_REASONS:
        return "worker_failure:" + reason
    return "host_guard_failure:" + reason


def current_image_ids(document: dict) -> dict[str, str]:
    return {cap: identity["platforms"][AMD64]["local_image_id"]
            for cap, identity in document["local_build_identities"].items() if AMD64 in identity["platforms"]}


def validate_run(result: dict, document: dict) -> None:
    """Every check that must pass before apply writes; raises ToolError naming the first failure."""
    _require(type(result) is dict and set(result) == {"fixtures", "records", "probes", "gpu_evidence"},
             "run output keys")
    current = current_image_ids(document)
    manifest = document["runtime_manifests"][AMD64]
    capabilities = {c["id"]: c for c in manifest["capabilities"]}
    fixtures = result["fixtures"]
    _require(fixtures and all(set(f) == {"name", "codec", "seconds", "bytes", "sha256"} for f in fixtures),
             "fixture keys")
    _require({f["codec"] for f in fixtures} == set(CODECS), "fixture codecs")
    fixture_points = {(f["codec"], f["seconds"]) for f in fixtures}
    records = result["records"]
    _require(records, "no records")
    hashes = []
    for record in records:
        _require(tuple(record) == RECORD_FIELDS, f"record fields of {record.get('name')}")
        _require(record["capability"] in ("asr_cpu", "asr_cuda"), f"record capability of {record['name']}")
        _require(record["bound_image_id"] == current[record["capability"]],
                 f"record {record['name']} is bound to {record['bound_image_id']}, not the current "
                 f"{record['capability']} build")
        _require(record["bound_to_current_build"] is True and record["platform"] == AMD64,
                 f"record {record['name']} build binding")
        _require(record["record_sha256"] == record_hash(record), f"record_sha256 of {record['name']}")
        outcome = record["outcome"]
        _require(outcome == "completed" or outcome.startswith(("worker_failure:", "host_guard_failure:")),
                 f"outcome of {record['name']}")
        if outcome == "completed":
            _require((record["codec"], record["fixture_seconds"]) in fixture_points,
                     f"record {record['name']} has no fixture")
            _require(type(record["peak_rss_bytes"]) is int and 0 < record["peak_rss_bytes"]
                     <= record["memory_cap_bytes"], f"peak_rss_bytes of {record['name']}")
            _require(type(record["audio_seconds"]) in (int, float)
                     and abs(record["audio_seconds"] - record["fixture_seconds"]) < 1.0,
                     f"audio_seconds of {record['name']}")
            if record["capability"] == "asr_cuda":
                _require(type(record["worker_vram_delta_bytes"]) is int and record["worker_vram_delta_bytes"] > 0,
                         f"worker_vram_delta_bytes of {record['name']}")
        hashes.append(record["record_sha256"])
    _require(len(set(hashes)) == len(hashes), "duplicate record_sha256")
    completed = {(r["capability"], r["model_id"], r["codec"], r["fixture_seconds"])
                 for r in records if r["outcome"] == "completed" and r["memory_cap_bytes"] == MEMORY_CAP}
    for point in MATRIX:
        _require(point in completed, "matrix point without a completed record: " + record_name(*point))
    probes = result["probes"]
    _require(set(probes) == {"asr_cpu", "asr_cuda"}, "probe set")
    for cap, probe in probes.items():
        _require(probe["bound_image_id"] == current[cap], f"probe {cap} is not bound to the current build")
        _require(probe["runtime"]["version"] == capabilities[cap]["runtime_version"], f"probe {cap} runtime")
        _require(POLICY_COMPUTE[cap] in probe["compute_types"], f"probe {cap} compute types")
        _require(probe["device"] == ("cuda" if cap == "asr_cuda" else "cpu"), f"probe {cap} device")
    gpu = result["gpu_evidence"]
    _require(tuple(gpu) == GPU_FIELDS, "gpu evidence fields")
    _require(gpu["bound_image_id"] == current["asr_cuda"], "gpu evidence is not bound to the current asr_cuda build")
    cuda_ok = [r for r in records if r["capability"] == "asr_cuda" and r["outcome"] == "completed"]
    _require(gpu["passthrough"] == bool(cuda_ok), "gpu passthrough disagrees with the CUDA records")
    _require("int8_float16" in gpu["container_cuda_compute_types"], "gpu compute types")


def apply_run(document: dict, result: dict) -> dict:
    """Validate ``result`` and return the updated document (the input is not modified)."""
    validate_run(result, document)
    document = json.loads(json.dumps(document))
    benchmarks = document["benchmarks"]
    benchmarks["fixtures"] = result["fixtures"]
    benchmarks["fixture_generator"] = FIXTURE_GENERATOR
    benchmarks["records"] = result["records"]
    benchmarks["probes"] = result["probes"]
    document["gpu_evidence"] = result["gpu_evidence"]
    records = result["records"]
    manifest = document["runtime_manifests"][AMD64]
    limit = next(c for c in manifest["capabilities"] if c["id"] == "asr_cpu")["worker_memory_bytes"]
    measured = []
    for model in manifest["models"]:
        if model["id"] not in MEASURED_MODELS:
            continue
        envelope = model_envelope(records, model["id"])
        if envelope is None:
            continue
        _require(envelope["worker_peak_ram_bytes"] < limit,
                 f"envelope {model['id']}: peak RAM {envelope['worker_peak_ram_bytes']} is not below the "
                 f"asr_cpu worker memory {limit}")
        _require(envelope["validated_max_duration_seconds"] < SIX_HOURS, "validated duration claims 6 h")
        model.update(envelope)
        measured.append(model["id"])
    document["model_selection"]["measured_envelopes"]["models"] = measured
    gpu = result["gpu_evidence"]
    cuda = next(c for c in manifest["capabilities"] if c["id"] == "asr_cuda")
    cuda["driver_version"] = gpu["driver_version"]
    cuda["hardware_fingerprint"] = gpu["name"]
    status = document["release_status"]
    status["blockers"] = rebuild_blockers(status["blockers"], records, manifest["models"])
    status["publishable"] = False
    for platform, runtime in document["runtime_manifests"].items():
        document["runtime_manifest_sha256"][platform] = hashlib.sha256(canonical_bytes(runtime)).hexdigest()
    return document


def write_manifest(path: Path, document: dict) -> None:
    payload = (json.dumps(document, indent=1, ensure_ascii=True) + "\n").encode("utf-8")
    handle, temp = tempfile.mkstemp(dir=str(path.parent), prefix=".release-", suffix=".tmp")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
        os.replace(temp, path)
    except BaseException:
        if os.path.exists(temp):
            os.unlink(temp)
        raise


def cmd_apply(args) -> int:
    path = Path(args.manifest)
    document = json.loads(path.read_text("utf-8"))
    result = json.loads(Path(args.file).read_text("utf-8"))
    updated = apply_run(document, result)
    write_manifest(path, updated)
    codes = ", ".join(f"{b['code']}:{b['capability']}" for b in updated["release_status"]["blockers"])
    print(f"[apply] updated {path}; measured {updated['model_selection']['measured_envelopes']['models']}; "
          f"blockers [{codes}]")
    return 0


# ----------------------------------------------------------------------------- live (Docker) path

DOCKER = shutil.which("docker") or "docker"
NVIDIA_SMI = shutil.which("nvidia-smi") or "nvidia-smi"


@functools.cache
def _docker_endpoint() -> tuple[str, str]:
    """The current Docker context name and endpoint (Windows npipe or Mac/Linux unix socket)."""
    result = subprocess.run((DOCKER, "context", "inspect"), shell=False, check=False, capture_output=True,
                            text=True, encoding="utf-8", errors="replace", timeout=30)
    try:
        data, = json.loads(result.stdout)
        return data["Name"], data["Endpoints"]["docker"]["Host"]
    except (ValueError, LookupError, TypeError) as error:
        raise ToolError("docker context inspect: start Docker and select a local context") from error


def _docker(*arguments: str, timeout: float = 120) -> subprocess.CompletedProcess:
    return subprocess.run((DOCKER, "--host", _docker_endpoint()[1], *arguments), shell=False, check=False,
                          capture_output=True, timeout=timeout)


def _nvidia_smi(*arguments: str) -> str:
    result = subprocess.run((NVIDIA_SMI, *arguments), shell=False, check=False, capture_output=True, text=True,
                            timeout=30)
    _require(result.returncode == 0, "nvidia-smi " + " ".join(arguments))
    return result.stdout.strip()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class _Audit:
    def record(self, **_values):
        return None


class _Lease:
    def __init__(self):
        self.held = False
        self.released = 0

    def try_acquire(self, owner, plan):
        if not plan.admitted or self.held:
            return False
        self.held = True
        return True

    def release(self, owner, *, writers_reaped):
        if not (writers_reaped and self.held):
            raise ToolError("lease released without reaped writers")
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


def _transcribe_request(capability: str, fixture: Path, snapshot_id: str, model: str):
    from podmuncher.config import Limits
    from podmuncher.protocol import WorkerRequest

    return WorkerRequest.from_dict({
        "schema_version": 1, "job_id": str(uuid4()), "attempt_id": str(uuid4()), "stage_id": str(uuid4()),
        "capability": capability, "operation": "transcribe",
        "assigned_slots": {"transcript_segments": "segments.json"},
        "source": None, "outputs": {"transcript": True, "video": False, "video_quality": "source_best"},
        "input_basename": fixture.name, "model_snapshot_id": snapshot_id, "limits": asdict(Limits()),
        "selection": {"device": "cuda" if capability == "asr_cuda" else "cpu", "profile": "balanced",
                      "model": model, "language": "auto"}})


def _policy(capability: str, image_id: str, gpu_uuid: str | None):
    from podmuncher.runtime import WorkerPolicy

    def proof():
        shown = _docker("image", "inspect", "--format", "{{.Id}}", image_id, timeout=30)
        return shown.returncode == 0 and shown.stdout.decode().strip() == image_id

    return WorkerPolicy(capability, image_id, ("python", "-I", "-m", "podmuncher.asr_worker"), 0, 0, proof,
                        development_image=True, allow_pull=False, cpus=CPUS, memory_bytes=MEMORY_CAP, pids=256,
                        selected_gpu_uuid=gpu_uuid if capability == "asr_cuda" else None)


def _owned_run(capability: str, image_id: str, work: Path, request, mounts_kwargs: dict, *, gpu_uuid: str | None,
               input_bytes: int, byte_limit: int, timeout_seconds: float, started: list) -> object:
    """One run through the owned DockerRuntime, built as replay_probe builds it; returns the WorkerResult."""
    from podmuncher.admission import WorkloadGrowth
    from podmuncher.config import Limits
    from podmuncher.docker_storage import make_docker_storage_provider
    from podmuncher.guards import CancellationCoordinator, WriteGuard
    from podmuncher.runtime import DockerRuntime, DockerTarget, NativeDockerCLI, RuntimeStorage, WorkerMounts

    name, endpoint = _docker_endpoint()
    target = DockerTarget(name, endpoint)
    policy = _policy(capability, image_id, gpu_uuid)
    scratch = work / "scratch"
    paths = {role: work / role for role in ("cache", "output", "state")}
    for path in (scratch, *paths.values()):
        path.mkdir(parents=True)
    provider = make_docker_storage_provider(timeout=60)
    session = provider.prepare(manual_backing="")
    cli = NativeDockerCLI(target)
    instance = "release-bench-" + uuid4().hex
    try:
        async def run():
            growth = WorkloadGrowth(0, 0, 0, 0, (input_bytes,), 0, 0, 0, 0, (byte_limit,), 0, 0)
            storage = RuntimeStorage(session, provider.volumes, paths={**paths, "scratch": scratch}, growth=growth,
                                     limits=Limits(), sample_budget_seconds=SAMPLE_BUDGET)
            lease = _Lease()
            coordinator = CancellationCoordinator(instance, lease, stop_grace_seconds=12, reap_timeout_seconds=12,
                                                  callback_timeout_seconds=12)
            guard = WriteGuard(coordinator, storage.sample, byte_limit=byte_limit, timeout_seconds=timeout_seconds,
                               sample_timeout_seconds=2.9)
            runtime = DockerRuntime(cli, guard, instance=instance, policy=policy,
                                    mounts=WorkerMounts(scratch, **mounts_kwargs))

            def consume(item):
                if item.event == "started":
                    started.append(True)

            try:
                return await runtime.run(request, on_event=consume)
            finally:
                listed = await cli.run("ps", "-a", "--no-trunc", "--format", "{{.ID}}", "--filter",
                                       "label=podmuncher.instance=" + instance, owner="release-verify-" + uuid4().hex)
                _require(listed.returncode == 0 and not listed.stdout.split(),
                         f"container left behind by {instance}")

        return asyncio.run(run())
    finally:
        session.close()


def replay_probe(capability: str, image_id: str, work: Path, *, gpu_uuid: str | None) -> tuple[dict, bytes]:
    """One bounded probe through the owned DockerRuntime (network none, no input, no model)."""
    started: list = []
    result = _owned_run(capability, image_id, work, _probe_request(capability), {}, gpu_uuid=gpu_uuid,
                        input_bytes=0, byte_limit=MIB, timeout_seconds=120, started=started)
    _require(result.completed and result.exit_code == 0, f"probe {capability} did not complete")
    claims = [claim.payload for claim in result.artifact_claims]
    _require([claim["slot"] for claim in claims] == ["probe_metadata"], f"probe {capability} claims")
    data = (work / "scratch" / "probe.json").read_bytes()
    _require(claims[0]["sha256"] == hashlib.sha256(data).hexdigest(), f"probe {capability} artifact hash")
    return json.loads(data), data


def probe_record(capability: str, image_id: str, work: Path, gpu_uuid: str | None) -> dict:
    from podmuncher.domains import DomainValidationError

    try:
        probe, data = replay_probe(capability, image_id, work, gpu_uuid=gpu_uuid)
    except DomainValidationError as error:
        raise ToolError(f"probe {capability}: {error.reason_code.value}") from None
    return {"record_sha256": hashlib.sha256(data).hexdigest(), "bound_image_id": image_id,
            "runtime": {"name": probe["runtime"]["name"], "version": probe["runtime"]["version"]},
            "compute_types": sorted(probe["compute_types"]), "device": probe["device"],
            "faster_whisper_version": probe["faster_whisper_version"]}


class _VramSampler:
    """Polls host nvidia-smi memory.used (MiB) every 0.25 s while a CUDA run is active."""

    def __init__(self, uuid: str):
        self.uuid = uuid
        self.baseline = self._read()
        self.peak = self.baseline
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _read(self) -> int:
        return int(_nvidia_smi("--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", self.uuid))

    def _loop(self):
        while not self._stop.wait(VRAM_POLL_SECONDS):
            try:
                self.peak = max(self.peak, self._read())
            except (ToolError, ValueError, OSError, subprocess.SubprocessError):
                continue

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._stop.set()
        self._thread.join(timeout=30)

    @property
    def delta_bytes(self) -> int:
        return (self.peak - self.baseline) * MIB


def _null_measurements() -> dict:
    return dict.fromkeys(MEASUREMENT_FIELDS)


def benchmark_once(point: tuple, image_id: str, fixture: Path, snapshot, work: Path, gpu_uuid: str | None) -> dict:
    """One matrix run; returns the sealed record (never raises for a measured failure)."""
    from podmuncher.domains import DomainValidationError

    capability, model, codec, seconds = point
    request = _transcribe_request(capability, fixture, snapshot.snapshot_id, model)
    timeout = min(3600, 600 + 6 * seconds)
    started: list = []
    sampler = _VramSampler(gpu_uuid) if capability == "asr_cuda" else None
    measurements = _null_measurements()
    try:
        if sampler is not None:
            with sampler:
                result = _owned_run(capability, image_id, work, request,
                                    {"input_file": fixture, "model": snapshot.path}, gpu_uuid=gpu_uuid,
                                    input_bytes=fixture.stat().st_size, byte_limit=OUTPUT_BYTES,
                                    timeout_seconds=timeout, started=started)
        else:
            result = _owned_run(capability, image_id, work, request, {"input_file": fixture, "model": snapshot.path},
                                gpu_uuid=None, input_bytes=fixture.stat().st_size, byte_limit=OUTPUT_BYTES,
                                timeout_seconds=timeout, started=started)
    except DomainValidationError as error:
        outcome = classify_failure(error.reason_code.value, bool(started))
    else:
        outcome, measurements = _read_result(result, work / "scratch" / "segments.json", sampler)
    return seal_record({
        "series": SERIES, "name": record_name(capability, model, codec, seconds), "capability": capability,
        "platform": AMD64, "bound_image_id": image_id, "bound_image_basis": "record image_id",
        "bound_to_current_build": True, "model_id": model, "codec": codec, "fixture_seconds": seconds,
        "memory_cap_bytes": MEMORY_CAP, "cpus": CPUS, "storage_sample_budget_seconds": SAMPLE_BUDGET,
        "outcome": outcome, **measurements, "recorded_at": _utc_now().isoformat(),
    })


def _read_result(result, path: Path, sampler: _VramSampler | None) -> tuple[str, dict]:
    if not (result.completed and result.exit_code == 0):
        return "worker_failure:worker_exit_unexpected", _null_measurements()
    claims = [claim.payload for claim in result.artifact_claims]
    try:
        data = path.read_bytes()
        if [c["slot"] for c in claims] != ["transcript_segments"] or claims[0]["sha256"] != \
                hashlib.sha256(data).hexdigest():
            raise ValueError("claim")
        document = json.loads(data)
        measured = document["measurements"]
        wall, audio = float(measured["transcribe_wall_seconds"]), float(measured["audio_seconds"])
        if audio <= 0:
            raise ValueError("audio")
        values = {
            "load_seconds": measured["load_seconds"], "transcribe_wall_seconds": measured["transcribe_wall_seconds"],
            "audio_seconds": measured["audio_seconds"], "real_time_factor": round(wall / audio, 4),
            "peak_rss_bytes": measured["peak_rss_bytes"],
            "worker_vram_delta_bytes": None if sampler is None else sampler.delta_bytes,
            "segments": len(document["segments"]), "detected_language": document["language"],
        }
    except (OSError, ValueError, LookupError, TypeError):
        return "worker_failure:artifact_rejected", _null_measurements()
    return "completed", values


def generate_fixtures(media_image: str, out: Path) -> list[dict]:
    """Render the original text with flite inside the media image, then encode every fixture."""
    (out / "text.txt").write_text("\n".join([SPEECH_TEXT] * SPEECH_REPEATS), encoding="utf-8", newline="\n")
    base = ("run", "--rm", "--pull", "never", "--network", "none", "--read-only", "--user", "65532:65532",
            "--tmpfs", "/tmp", "--mount", f"type=bind,source={out},target=/out", "--entrypoint", "ffmpeg",
            media_image, "-nostdin", "-hide_banner", "-loglevel", "error", "-y")
    longest = max(FIXTURE_SECONDS)
    master = _docker(*base, "-f", "lavfi", "-i", "flite=textfile=/out/text.txt:voice=slt",
                     "-af", "aresample=16000,apad", "-ac", "1", "-ar", "16000", "-t", str(longest),
                     "-c:a", "pcm_s16le", "/out/master.wav", timeout=1800)
    _require(master.returncode == 0, "fixture generation (flite master): "
             + master.stderr.decode("utf-8", "replace")[-400:])
    fixtures = []
    for seconds in FIXTURE_SECONDS:
        for codec, (extension, encoder) in CODECS.items():
            name = f"speech-{seconds}.{extension}"
            encoded = _docker(*base, "-i", "/out/master.wav", "-t", str(seconds), "-ac", "1", "-ar", "16000",
                              *encoder, "/out/" + name, timeout=1800)
            _require(encoded.returncode == 0, f"fixture generation ({name}): "
                     + encoded.stderr.decode("utf-8", "replace")[-400:])
            data = (out / name).read_bytes()
            fixtures.append({"name": name, "codec": codec, "seconds": seconds, "bytes": len(data),
                             "sha256": hashlib.sha256(data).hexdigest()})
    (out / "master.wav").unlink()
    (out / "text.txt").unlink()
    return sorted(fixtures, key=lambda f: f["name"])


def gpu_evidence(cuda_image: str) -> dict:
    line = _nvidia_smi("--query-gpu=name,driver_version,uuid,memory.total", "--format=csv,noheader,nounits")
    name, driver, uuid, memory = [part.strip() for part in line.splitlines()[0].split(",")]
    script = ("import ctranslate2,json; print(json.dumps({'count': ctranslate2.get_cuda_device_count(), "
              "'types': sorted(ctranslate2.get_supported_compute_types('cuda'))}))")
    shown = _docker("run", "--rm", "--pull", "never", "--gpus", "all", "--network", "none", "--entrypoint", "python",
                    cuda_image, "-c", script, timeout=300)
    _require(shown.returncode == 0, "gpu evidence container: " + shown.stderr.decode("utf-8", "replace")[-400:])
    inside = json.loads(shown.stdout.decode("utf-8").strip().splitlines()[-1])
    return {"passthrough": inside["count"] >= 1, "method": GPU_METHOD, "name": name, "driver_version": driver,
            "uuid": uuid, "memory_total_bytes": int(memory) * MIB, "bound_image_id": cuda_image,
            "container_cuda_device_count": inside["count"], "container_cuda_compute_types": inside["types"],
            "checked_at": _utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")}


def _load_amd64_runtime(document: dict):
    from podmuncher.manifests import load_manifest

    with tempfile.TemporaryDirectory(prefix="pm-bench-manifest-") as directory:
        path = Path(directory) / "linux-amd64.json"
        data = canonical_bytes(document["runtime_manifests"][AMD64])
        path.write_bytes(data)
        return load_manifest(path, development=True, expected_sha256=hashlib.sha256(data).hexdigest())


def cmd_run(args) -> int:
    from podmuncher.model_cache import ModelCache

    document = json.loads(Path(args.manifest).read_text("utf-8"))
    ids = current_image_ids(document)
    model_root = Path(args.model_root).resolve()
    _require(REPO not in (model_root, *model_root.parents), "--model-root must be outside the repository")
    model_root.mkdir(parents=True, exist_ok=True)
    out = Path(args.out).resolve()
    jsonl = out.with_name(out.name + ".records.jsonl")
    runtime = _load_amd64_runtime(document)
    models = {m.artifact.model_id: m for m in runtime.models if m.artifact.model_id in MEASURED_MODELS}
    _require(set(models) == set(MEASURED_MODELS), "the manifest lacks a measured model")
    cache = ModelCache(model_root, audit=_Audit(), admission=lambda **v: True, timeout_seconds=600)
    work_root = Path(tempfile.mkdtemp(prefix="pm-bench-")).resolve()
    _require(REPO not in work_root.parents, "the work directory must be outside the repository")
    try:
        snapshots = {model_id: cache.acquire(model) for model_id, model in models.items()}
        fixture_dir = work_root / "fixtures"
        fixture_dir.mkdir()
        fixtures = generate_fixtures(ids["media"], fixture_dir)
        print(f"[run] fixtures {len(fixtures)} generated")
        by_point = {(f["codec"], f["seconds"]): fixture_dir / f["name"] for f in fixtures}
        gpu = gpu_evidence(ids["asr_cuda"])
        print(f"[run] gpu passthrough={gpu['passthrough']} {gpu['name']} driver {gpu['driver_version']}")
        probes = {}
        for cap in ("asr_cpu", "asr_cuda"):
            probes[cap] = probe_record(cap, ids[cap], work_root / ("probe-" + cap),
                                       gpu["uuid"] if cap == "asr_cuda" else None)
            print(f"[run] probe {cap} runtime={probes[cap]['runtime']['version']} device={probes[cap]['device']}")
        records = []
        for index, point in enumerate(MATRIX):
            capability, model, codec, seconds = point
            for attempt in range(1 + HOST_GUARD_RETRIES):
                work = work_root / f"run-{index}-{attempt}"
                began = time.monotonic()
                record = benchmark_once(point, ids[capability], by_point[(codec, seconds)], snapshots[model], work,
                                        gpu["uuid"] if capability == "asr_cuda" else None)
                shutil.rmtree(work, ignore_errors=True)
                records.append(record)
                with jsonl.open("a", encoding="ascii", newline="\n") as stream:
                    stream.write(record_line(record) + "\n")
                print(f"{record['name']} {record['outcome']} {time.monotonic() - began:.1f}s", flush=True)
                if not record["outcome"].startswith("host_guard_failure:"):
                    break
        result = {"fixtures": fixtures, "records": records, "probes": probes, "gpu_evidence": gpu}
        out.write_text(json.dumps(result, indent=1, ensure_ascii=True) + "\n", encoding="utf-8", newline="\n")
        print(f"[run] wrote {out} ({len(records)} records)")
    finally:
        cache.close()
        shutil.rmtree(work_root, ignore_errors=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="generate fixtures, run the benchmark matrix, probes and GPU evidence")
    run.add_argument("--model-root", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--manifest", default=str(REPO / MANIFEST_REL))
    run.set_defaults(func=cmd_run)
    apply = sub.add_parser("apply", help="validate a run output and update the manifest")
    apply.add_argument("file")
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
