"""Contract tests for immutable state codecs and safe projections."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import pytest

from podmuncher.domains import (
    ArtifactFormat, ArtifactSlot, DomainValidationError,
    Device, FieldStatus, JobState, ReasonCode, Stage, StageOutcome, VolumeSlot, WorkerSlot,
)
from podmuncher.requests import JobRequest
from podmuncher.snapshots import (
    Artifact, Attempt, Job, Measurement, Provenance, ResourceSnapshot,
    SnapshotHistory, StageRecord, StorageEntry,
)


ID1 = "11111111-1111-4111-8111-111111111111"
ID2 = "22222222-2222-4222-8222-222222222222"
ID3 = "33333333-3333-4333-8333-333333333333"
TS = "2026-10-09T20:00:00Z"
SHA = "a" * 64
SECRET = "sensitive-path-secret"


def m(value=1, status="ok", reason=None):
    return {"value": value, "source": SECRET, "status": status,
            "sample_time": TS, "reason": reason}


def make_storage(slot=0):
    return {"volume_slot": slot, "volume_identity": f"/private/{SECRET}",
            "context": "host", "capacity": m(100), "free": m(90),
            "reserve": m(5), "reserved": m(4), "required": m(3),
            "available_after_job": m(78), "source": SECRET,
            "sample_time": TS, "backing_volume_slot": None}


def make_snapshot():
    return {"schema_version": 1, "sample_time": TS, "freshness": "ok",
            "service": m("running"), "docker": m(True), "worker": m(False),
            "host_ram": m(100), "worker_ram": m(50), "cpu": m(30.5),
            "gpu_capability": m(True), "gpu_utilization": m(40),
            "gpu_vram": m(1000), "model": m("tiny"),
            "storage": [make_storage()], "budget_confidence": "measured",
            "job_id": ID1, "attempt_id": ID2, "stage": "preparing",
            "progress": m(0), "elapsed_seconds": m(0), "stage_eta_seconds": m(2)}


def make_request():
    return JobRequest.from_dict({"schema_version": 1,
        "source": {"kind": "local", "input": "./input.mp4"},
        "outputs": {"transcript": True, "video": False, "video_quality": "1080p"},
        "selection": {"device": "auto", "profile": "balanced", "model": "auto", "language": "auto"},
        "output_root": "./out", "limits": {"media_bytes": 100, "duration_seconds": 20,
                                          "export_bytes": 10}})


def make_job():
    request = make_request()
    return Job(ID1, request, request.request_hash, JobState.QUEUED, ReasonCode.OK,
               None, ID2, TS, TS, (Artifact(ID3, ArtifactSlot.TRANSCRIPT_TXT,
               ArtifactFormat.TXT, 5, SHA, True, ID2,
               Provenance(model_id="tiny", source_sha256=SHA, attempt_id=ID2),
               f"C:\\private\\{SECRET}\\transcript.txt"),),
               {"secret": [SECRET, {"nested": SECRET}]})


def test_measurement_all_statuses_and_absent_is_never_zero():
    for status in FieldStatus:
        data = m(0) if status is FieldStatus.OK else m(None, status.value, "storage_unknown")
        item = Measurement.from_json(json.dumps(data))
        assert Measurement.from_dict(item.to_dict()) == item
        assert item.to_dict()["value"] == (0 if status is FieldStatus.OK else None)
    with pytest.raises(DomainValidationError):
        Measurement.from_dict(m(0, "unknown", "storage_unknown"))
    with pytest.raises(DomainValidationError):
        Measurement.from_dict(m(None, "unknown", None))


def test_snapshot_roundtrip_and_storage_boundaries():
    data = make_snapshot()
    snapshot = ResourceSnapshot.from_json(json.dumps(data))
    assert ResourceSnapshot.from_dict(snapshot.to_dict()) == snapshot
    data["storage"] = [make_storage(i) for i in range(8)]
    assert len(ResourceSnapshot.from_dict(data).storage) == 8
    with pytest.raises(DomainValidationError):
        ResourceSnapshot.from_dict({**data, "storage": data["storage"] + [make_storage(0)]})
    with pytest.raises(DomainValidationError):
        ResourceSnapshot.from_dict({**data, "storage": [make_storage(0), make_storage(0)]})


def test_history_caps_600_and_is_value_style():
    sample = ResourceSnapshot.from_dict(make_snapshot())
    history = SnapshotHistory()
    for _ in range(601):
        history = history.append(sample)
    assert len(history.samples) == 600
    assert len(SnapshotHistory.from_json(history.to_json()).samples) == 600
    with pytest.raises(DomainValidationError):
        SnapshotHistory.from_dict({"samples": [sample.to_dict()] * 601})


def test_job_attempt_stage_artifact_roundtrip_and_immutability():
    job = make_job()
    assert Job.from_json(job.to_json()) == job
    attempt = Attempt(ID2, ID1, 1, "tiny", None, TS, None, JobState.PREPARING,
                      ReasonCode.OK, (StageRecord(ID3, ID2, Stage.PREPARING,
                      StageOutcome.OK, TS, TS, (("input", SHA),), SHA,
                      (WorkerSlot.DOWNLOADED_MEDIA,)),), 1)
    assert Attempt.from_json(attempt.to_json()) == attempt
    with pytest.raises(FrozenInstanceError):
        job.state = JobState.COMPLETED
    with pytest.raises(TypeError):
        job.preflight["secret"] = "changed"
    with pytest.raises(TypeError):
        job.preflight["secret"][1]["nested"] = "changed"
    exported = job.to_dict()
    exported["preflight"]["secret"][1]["nested"] = "changed"
    assert job.to_dict()["preflight"]["secret"][1]["nested"] == SECRET


def test_public_and_telemetry_projections_redact_sensitive_data():
    job = make_job()
    snapshot = ResourceSnapshot.from_dict(make_snapshot())
    assert job.public_dict()["artifacts"][0]["basename"] == "transcript.txt"
    assert SECRET not in json.dumps(job.public_dict())
    assert SECRET not in json.dumps(job.telemetry_dict())
    assert SECRET not in json.dumps(snapshot.public_dict())
    assert SECRET not in json.dumps(snapshot.telemetry_dict())
    assert SECRET not in json.dumps(job.artifacts[0].telemetry_dict())


@pytest.mark.parametrize("path,value", [
    (("schema_version",), True), (("schema_version",), 2),
    (("sample_time",), "2026-10-09T20:00:00+01:00"),
    (("job_id",), "bad"), (("freshness",), "__invalid__"),
    (("budget_confidence",), "__invalid__"), (("stage",), "__invalid__"),
    (("storage", 0, "volume_slot"), True),
    (("storage", 0, "context"), "__invalid__"),
    (("storage", 0, "capacity", "value"), -1),
    (("cpu", "value"), 101), (("progress", "value"), float("nan")),
    (("elapsed_seconds", "value"), -1),
    (("docker", "value"), 1), (("service", "value"), "__invalid__"),
])
def test_snapshot_invalid_domains_and_types(path, value):
    data = make_snapshot()
    cursor = data
    for part in path[:-1]:
        cursor = cursor[part]
    cursor[path[-1]] = value
    with pytest.raises(DomainValidationError):
        ResourceSnapshot.from_dict(data)


def test_unknown_keys_deep_json_utf8_and_duplicate_json():
    data = make_snapshot()
    data["extra"] = 1
    with pytest.raises(DomainValidationError):
        ResourceSnapshot.from_dict(data)
    data = make_snapshot()
    data["model"]["source"] = "\ud800"
    with pytest.raises(DomainValidationError):
        ResourceSnapshot.from_dict(data)
    with pytest.raises(DomainValidationError):
        ResourceSnapshot.from_json('{"schema_version":1,"schema_version":1}')
    with pytest.raises(DomainValidationError):
        ResourceSnapshot.from_json('{"x": NaN}')
    with pytest.raises(DomainValidationError):
        Job.from_dict({**make_job().to_dict(), "preflight": {"x": {"y": {"z": {"a": {"b": {"c": {"d": {"e": 1}}}}}}}}})


def test_request_hash_rechecked_and_verified_selection_required():
    data = make_job().to_dict()
    data["request_hash"] = "b" * 64
    with pytest.raises(DomainValidationError):
        Job.from_dict(data)
    data = make_job().to_dict()
    data["request"]["selection"]["model"] = "verified-model"
    with pytest.raises(DomainValidationError):
        Job.from_dict(data)
    request = JobRequest.from_dict(data["request"], verified_models={"verified-model"})
    data["request_hash"] = request.request_hash
    restored = Job.from_dict(data, verified_models={"verified-model"})
    assert restored.request.selection.model == "verified-model"
    with pytest.raises(DomainValidationError):
        Job.from_json(restored.to_json())


def test_registry_negatives_rejected_in_relevant_fields():
    registry = json.loads(Path("Docs/contract-domains.json").read_text(encoding="utf-8"))
    for bad in registry["negative_examples"]:
        data = make_snapshot()
        for key in ("freshness", "budget_confidence", "stage"):
            altered = deepcopy(data)
            altered[key] = bad
            with pytest.raises(DomainValidationError):
                ResourceSnapshot.from_dict(altered)
        altered = make_job().to_dict()
        altered["state"] = bad
        with pytest.raises(DomainValidationError):
            Job.from_dict(altered)
        altered = make_job().to_dict()
        altered["artifacts"][0]["slot"] = bad
        with pytest.raises(DomainValidationError):
            Job.from_dict(altered)
        for field in ("format",):
            altered = make_job().to_dict()
            altered["artifacts"][0][field] = bad
            with pytest.raises(DomainValidationError):
                Job.from_dict(altered)
        for field in ("status", "reason"):
            altered = m(None, "unknown", "storage_unknown")
            altered[field] = bad
            with pytest.raises(DomainValidationError):
                Measurement.from_dict(altered)
        for field in ("volume_slot", "context", "backing_volume_slot"):
            altered = make_storage()
            altered[field] = bad
            with pytest.raises(DomainValidationError):
                StorageEntry.from_dict(altered)
        for field in ("service",):
            altered = make_snapshot()
            altered[field]["value"] = bad
            with pytest.raises(DomainValidationError):
                ResourceSnapshot.from_dict(altered)
        with pytest.raises(DomainValidationError):
            Provenance(device=bad)
        stage = StageRecord(ID3, ID2, Stage.PREPARING, StageOutcome.OK, TS, None,
                            (("input", SHA),), SHA, (WorkerSlot.DOWNLOADED_MEDIA,))
        for field in ("stage", "status", "output_slots"):
            altered = stage.to_dict()
            if field == "output_slots":
                altered[field] = [bad]
            else:
                altered[field] = bad
            with pytest.raises(DomainValidationError):
                StageRecord.from_dict(altered)
        attempt = Attempt(ID2, ID1, 1, None, None, TS, None, JobState.PREPARING,
                          ReasonCode.OK, (stage,), 0)
        for field in ("state", "reason_code"):
            altered = attempt.to_dict()
            altered[field] = bad
            with pytest.raises(DomainValidationError):
                Attempt.from_dict(altered)


def test_direct_construction_normalizes_before_serialization():
    missing = Measurement(None, "probe", "unknown", TS, "storage_unknown")
    assert missing.status is FieldStatus.UNKNOWN
    assert missing.reason is ReasonCode.STORAGE_UNKNOWN
    assert Measurement.from_json(missing.to_json()) == missing

    storage = StorageEntry(**make_storage())
    assert storage.volume_slot is VolumeSlot.VALUE_0
    assert type(storage.capacity) is Measurement
    assert type(storage.capacity.value) is int
    assert StorageEntry.from_json(storage.to_json()) == storage

    snapshot = ResourceSnapshot(**{**make_snapshot(), "storage": [make_storage()]})
    assert snapshot.freshness is FieldStatus.OK
    assert type(snapshot.storage[0]) is StorageEntry
    assert type(snapshot.elapsed_seconds) is Measurement
    assert ResourceSnapshot.from_json(snapshot.to_json()) == snapshot

    provenance = Provenance(device="cpu", image_digest="sha256:" + SHA)
    assert provenance.device == Device.CPU.value
    assert Provenance.from_json(provenance.to_json()) == provenance
    artifact = Artifact(ID3, "transcript_txt", "txt", 5, SHA, True, ID2,
                        provenance.to_dict())
    assert artifact.slot is ArtifactSlot.TRANSCRIPT_TXT
    assert type(artifact.provenance) is Provenance
    assert Artifact.from_json(artifact.to_json()) == artifact
    stage = StageRecord(ID3, ID2, "preparing", "ok", TS, None,
                        (("input", SHA),), SHA, ("downloaded_media",))
    assert stage.output_slots == (WorkerSlot.DOWNLOADED_MEDIA,)
    assert StageRecord.from_json(stage.to_json()) == stage
    attempt = Attempt(ID2, ID1, 1, None, provenance.to_dict(), TS, None,
                      "preparing", "ok", (stage.to_dict(),), 0)
    assert type(attempt.stage_records[0]) is StageRecord
    assert Attempt.from_json(attempt.to_json()) == attempt
    job = make_job()
    raw_job = Job(job.job_id, job.request, job.request_hash, "queued", "ok",
                  None, ID2, TS, TS, [artifact.to_dict()], {})
    assert type(raw_job.artifacts[0]) is Artifact
    assert Job.from_json(raw_job.to_json()) == raw_job


def test_measurement_value_is_frozen_and_validated_at_construction():
    raw = {"items": [{"name": "before"}]}
    item = Measurement(raw, "probe", "ok", TS, None)
    raw["items"][0]["name"] = "after"
    assert item.to_dict()["value"]["items"][0]["name"] == "before"
    with pytest.raises(TypeError):
        item.value["items"][0]["name"] = "changed"
    with pytest.raises(DomainValidationError):
        Measurement(10 ** 5000, "probe", "ok", TS, None)


@pytest.mark.parametrize("value", [0.5, 0, 1, 10 ** 100])
def test_duration_accepts_nonnegative_finite_numbers(value):
    data = make_snapshot()
    data["elapsed_seconds"]["value"] = value
    data["stage_eta_seconds"]["value"] = value
    snapshot = ResourceSnapshot.from_dict(data)
    assert snapshot.elapsed_seconds.value == value
    assert ResourceSnapshot.from_json(snapshot.to_json()) == snapshot


@pytest.mark.parametrize("value", [True, -0.5, float("inf"), float("nan")])
def test_duration_rejects_bool_negative_and_nonfinite(value):
    data = make_snapshot()
    data["elapsed_seconds"]["value"] = value
    with pytest.raises(DomainValidationError):
        ResourceSnapshot.from_dict(data)


@pytest.mark.parametrize("device,digest", [
    ("__invalid__", None), ("", None), (None, "sha256:bad"),
    (None, SHA), (None, "sha256:" + "A" * 64),
])
def test_provenance_rejects_invalid_device_and_digest(device, digest):
    with pytest.raises(DomainValidationError):
        Provenance(device=device, image_digest=digest)


def test_whole_record_depth_and_string_limits_preserve_reason_codes():
    job = make_job().to_dict()
    job["preflight"] = {"a": {"b": {"c": {"d": {"e": {"f": 1}}}}}}
    assert Job.from_json(json.dumps(job)).to_dict()["preflight"] == job["preflight"]
    job["preflight"]["a"]["b"]["c"]["d"]["e"]["f"] = {"g": {"h": 1}}
    with pytest.raises(DomainValidationError) as error:
        Job.from_dict(job)
    assert error.value.reason_code is ReasonCode.NESTING_LIMIT
    with pytest.raises(DomainValidationError) as error:
        Job.from_json(json.dumps(job))
    assert error.value.reason_code is ReasonCode.NESTING_LIMIT
    job = make_job().to_dict()
    job["preflight"] = {"x": "x" * 8193}
    with pytest.raises(DomainValidationError):
        Job.from_dict(job)


def test_large_bounded_history_roundtrips_and_601_rejects():
    data = make_snapshot()
    data["model"]["value"] = "m" * 1200
    history = SnapshotHistory((ResourceSnapshot.from_dict(data),) * 600)
    payload = history.to_json()
    assert len(payload.encode("utf-8")) > 262144
    assert SnapshotHistory.from_json(payload) == history
    with pytest.raises(DomainValidationError):
        SnapshotHistory.from_json(json.dumps({"samples": [data] * 601}))
