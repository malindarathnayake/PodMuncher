"""Wire contract and adversarial byte-stream tests."""

from dataclasses import FrozenInstanceError
from dataclasses import asdict
import json
from uuid import uuid4

import pytest

from podmuncher.config import Limits
from podmuncher.domains import ReasonCode
from podmuncher.protocol import (MAX_RECORD_BYTES, ProtocolError, WorkerEvent, WorkerEventRecord,
                                  WorkerEventStream, WorkerRequest, emit_event)


def make_worker_request(**changes):
    data = {"schema_version": 1, "job_id": str(uuid4()), "attempt_id": str(uuid4()),
            "stage_id": str(uuid4()), "capability": "media", "operation": "download",
            "assigned_slots": {"downloaded_media": "private.mp4"},
            "source": {"kind": "direct", "input": "https://example.com/video.mp4"},
            "outputs": {"transcript": True, "video": False, "video_quality": "source_best"},
            "input_basename": None, "model_snapshot_id": None,
            "limits": asdict(Limits()),
            "selection": {"device": "auto", "profile": "quality", "model": "auto", "language": "auto"}}
    data.update(changes)
    return WorkerRequest.from_dict(data)


def make_event(request, sequence=0, event="started", payload=None):
    return WorkerEventRecord(1, request.job_id, request.attempt_id, request.stage_id,
                             sequence, event, {} if payload is None else payload)


def bounded_bytes(event):
    record = event.to_json() + b"\n"
    assert len(record) <= MAX_RECORD_BYTES + 1
    return record


def test_remote_request_round_trip_and_private_mapping():
    request = make_worker_request()
    assert WorkerRequest.from_json(request.to_json()) == request
    assert request.to_dict()["outputs"]["video_quality"] == "source_best"
    assert request.to_dict()["limits"] == asdict(Limits())
    slots = {"downloaded_media": "private.mp4"}
    source = {"kind": "youtube", "input": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}
    alternate = make_worker_request(assigned_slots=slots, source=source, operation="metadata")
    slots["downloaded_media"] = "changed.mp4"
    source["input"] = "https://example.com/changed"
    assert alternate.assigned_slots["downloaded_media"] == "private.mp4"
    assert alternate.source["kind"] == "youtube"
    with pytest.raises(TypeError):
        alternate.assigned_slots["downloaded_media"] = "other"
    with pytest.raises(FrozenInstanceError):
        alternate.operation = "other"


@pytest.mark.parametrize("bad_value", ["nested", "cyclic", "invalid_scalar"])
def test_malformed_limits_rejected_at_direct_and_dict_boundaries(bad_value):
    request = make_worker_request()
    data = request.to_dict()
    if bad_value == "nested":
        value = []
        for _ in range(600):
            value = [value]
    elif bad_value == "cyclic":
        value = []
        value.append(value)
    else:
        value = "600"
    data["limits"]["media_bytes"] = value

    with pytest.raises(ProtocolError):
        WorkerRequest.from_dict(data)
    with pytest.raises(ProtocolError):
        WorkerRequest(**{**data, "limits": Limits(**data["limits"]),
                         "outputs": request.outputs, "selection": request.selection})

    assert WorkerRequest.from_json(request.to_json()) == request


@pytest.mark.parametrize("source", [
    {"kind": "local", "input": "C:\\private\\video.mp4"},
    {"kind": "direct", "input": "http://localhost/video"},
    {"kind": "direct", "input": "C:\\private\\video.mp4"},
    {"kind": "direct", "input": "https://example.com/video" , "path": "C:\\private"},
])
def test_source_local_path_and_extra_fields_rejected(source):
    with pytest.raises(ProtocolError):
        make_worker_request(source=source)


@pytest.mark.parametrize("changes", [
    {"operation": "download", "source": None},
    {"operation": "download", "input_basename": "file.mp4"},
    {"operation": "probe", "source": None, "input_basename": None},
    {"operation": "transcribe", "source": None, "input_basename": None, "capability": "asr_cpu"},
    {"operation": "transcribe", "source": {"kind": "direct", "input": "https://example.com/a"}, "input_basename": "file.mp4"},
    {"operation": "anything", "source": {"kind": "direct", "input": "https://example.com/a"}},
    {"operation": "../../escape"},
    {"model_snapshot_id": "C:\\models\\tiny"},
    {"selection": {"device": "auto", "profile": "balanced", "model": "C:\\model", "language": "auto"}},
    {"assigned_slots": {"downloaded_media": "../escape"}},
    {"assigned_slots": {"downloaded_media": "C:escape"}},
    {"assigned_slots": {"downloaded_media": "a\\b"}},
    {"assigned_slots": {"downloaded_media": "same", "downloaded_video": "same"}},
    {"assigned_slots": {"original_video": "final.mp4"}},
    {"limits": {**asdict(Limits()), "download_attempts": 4}},
    {"limits": {**asdict(Limits()), "pending_jobs": 101}},
    {"limits": {**asdict(Limits()), "reserve_fraction": float("nan")}},
])
def test_request_admission_rejects_sidechannels_and_invalid_limits(changes):
    with pytest.raises(ProtocolError):
        make_worker_request(**changes)


def test_local_file_and_asr_runtime_probe():
    local = make_worker_request(operation="transcribe", capability="asr_cpu", source=None,
                                input_basename="private.wav", model_snapshot_id="snapshot.1")
    assert WorkerRequest.from_json(local.to_json()) == local
    probe = make_worker_request(operation="probe", capability="asr_cuda", source=None)
    assert probe.input_basename is None


def test_event_payloads_and_round_trip():
    request = make_worker_request()
    examples = [("started", {}), ("progress", {"percent": 100.0}),
                ("artifact", {"slot": "downloaded_media", "bytes": 0, "sha256": "a" * 64}),
                ("warning", {"reason_code": "no_audio", "summary": "Audio absent"}),
                ("error", {"reason_code": "media_probe_failed", "summary": "Probe failed"}),
                ("completed", {})]
    stream = WorkerEventStream(request)
    for sequence, (event, payload) in enumerate(examples):
        record = make_event(request, sequence, event, payload)
        assert WorkerEventRecord.from_json(record.to_json()) == record
        stream.feed(bounded_bytes(record))
    assert [stream.pop().event for _ in examples] == [kind for kind, _ in examples]
    assert stream.pop() is None
    stream.finish()


@pytest.mark.parametrize("kind,reason,untrusted,canonical", [
    ("warning", "no_audio", "open('/home/alice/private/transcript.txt') failed", "No audio"),
    ("error", "media_probe_failed", "Authorization: Bearer fake-test-token", "Media probe failed"),
    ("warning", "no_audio", "The secret launch code is purple kiwi", "No audio"),
])
def test_warning_error_summaries_are_canonical_at_every_boundary(kind, reason, untrusted, canonical):
    request = make_worker_request()
    payload = {"reason_code": reason, "summary": untrusted}
    record = make_event(request, event=kind, payload=payload)
    raw_data = {**make_event(request).to_dict(), "event": kind, "payload": payload}
    raw_json = json.dumps(raw_data).encode("utf-8")

    from_dict = WorkerEventRecord.from_dict(raw_data)
    from_json = WorkerEventRecord.from_json(raw_json)
    for admitted in (record, from_dict, from_json):
        assert admitted.to_dict()["payload"] == {"reason_code": reason, "summary": canonical}
        assert WorkerEventRecord.from_json(admitted.to_json()) == admitted
        assert untrusted.encode("utf-8") not in admitted.to_json()
    assert payload["summary"] == untrusted

    emitted = []
    emit_event(record, emitted.append)
    assert untrusted.encode("utf-8") not in emitted[0]
    assert WorkerEventRecord.from_json(emitted[0][:-1]) == record

    stream = WorkerEventStream(request)
    stream.feed(raw_json + b"\n")
    assert stream.pop() == record
    assert stream.pop() is None
    stream.finish()


@pytest.mark.parametrize("event,payload", [
    ("progress", {"percent": float("nan")}), ("progress", {"percent": 101}),
    ("progress", {"percent": 100, "completed": True}),
    ("artifact", {"slot": "original_video", "bytes": 1, "sha256": "a" * 64}),
    ("artifact", {"slot": "downloaded_media", "bytes": -1, "sha256": "a" * 64}),
    ("artifact", {"slot": "downloaded_media", "bytes": 1, "sha256": "A" * 64}),
    ("artifact", {"slot": "downloaded_media", "bytes": 1, "sha256": "a" * 64, "path": "C:\\secret"}),
    ("warning", {"reason_code": "invalid", "summary": "oops"}),
    ("warning", {"reason_code": "no_audio", "summary": "secret\npath"}),
    ("warning", {"reason_code": "no_audio", "summary": "see C:\\private\\secret"}),
    ("warning", {"reason_code": "no_audio", "summary": "URL https://example.com/token"}),
    ("warning", {"reason_code": "no_audio", "summary": "TOKEN=secret"}),
    ("warning", {"reason_code": "no_audio", "summary": "unsafe\u202epath"}),
    ("warning", {"reason_code": "no_audio", "summary": "unsafe\ud800path"}),
    ("warning", {"reason_code": "no_audio", "summary": 42}),
    ("error", {"reason_code": "no_audio", "summary": "x" * 1025}),
    ("completed", {"percent": 100}),
])
def test_invalid_payloads(event, payload):
    with pytest.raises(ProtocolError):
        make_event(make_worker_request(), event=event, payload=payload)


def test_json_and_record_boundaries():
    request = make_worker_request()
    event = make_event(request)
    for payload in (b"\xff", b"{", b'{"schema_version":1,"schema_version":1}',
                    event.to_json().replace(b'"schema_version":1', b'"schema_version":2'),
                    event.to_json()[:-1] + b',"extra":1}', b'{"a":NaN}',
                    b'[' * 10 + b'0' + b']' * 10):
        with pytest.raises(ProtocolError):
            WorkerEventRecord.from_json(payload)
    assert len(b" " * MAX_RECORD_BYTES) == MAX_RECORD_BYTES
    with pytest.raises(ProtocolError):
        WorkerEventRecord.from_json(b" " * (MAX_RECORD_BYTES + 1))
    padded = event.to_json() + b" " * (MAX_RECORD_BYTES - len(event.to_json()))
    assert WorkerEventRecord.from_json(padded) == event
    with pytest.raises(ProtocolError) as error:
        WorkerEventRecord.from_json(padded + b" ")
    assert error.value.reason_code == ReasonCode.EVENT_LIMIT_REACHED
    stream = WorkerEventStream(request)
    stream.feed(b" " * MAX_RECORD_BYTES)
    with pytest.raises(ProtocolError) as error:
        stream.feed(b" ")
    assert error.value.reason_code == ReasonCode.EVENT_LIMIT_REACHED
    truncated = WorkerEventStream(request)
    truncated.feed(event.to_json())
    with pytest.raises(ProtocolError):
        truncated.finish()
    complete = WorkerEventStream(request)
    complete.feed(bounded_bytes(event))
    complete.finish()
    with pytest.raises(ProtocolError):
        complete.feed(bounded_bytes(make_event(request, 1)))


def test_identity_sequence_slot_and_failure_latch():
    request = make_worker_request()
    stream = WorkerEventStream(request)
    stream.feed(bounded_bytes(make_event(request)))
    with pytest.raises(ProtocolError):
        stream.feed(bounded_bytes(make_event(request)))
    with pytest.raises(ProtocolError):
        stream.feed(bounded_bytes(make_event(request, 1)))
    other = make_worker_request()
    with pytest.raises(ProtocolError):
        WorkerEventStream(request).feed(bounded_bytes(make_event(other)))
    with pytest.raises(ProtocolError):
        WorkerEventStream(request).feed(bounded_bytes(make_event(request, event="artifact", payload={"slot": "probe_metadata", "bytes": 1, "sha256": "a" * 64})))


def test_fault_emitter_receives_bounded_reason_once():
    request = make_worker_request()
    reasons = []
    stream = WorkerEventStream(request, fault_emitter=reasons.append)
    with pytest.raises(ProtocolError):
        stream.feed(b"not json\n")
    with pytest.raises(ProtocolError):
        stream.feed(bounded_bytes(make_event(request)))
    assert reasons == [ReasonCode.WORKER_PROTOCOL_MISMATCH]
    assert WorkerEvent is WorkerEventRecord
    capped = WorkerEventStream(request, fault_emitter=reasons.append)
    with pytest.raises(ProtocolError):
        capped.feed(b"x" * (MAX_RECORD_BYTES + 1))
    assert reasons[-1] == ReasonCode.EVENT_LIMIT_REACHED


def test_queue_event_and_byte_caps():
    request = make_worker_request()
    stream = WorkerEventStream(request, clock=lambda: 0.0)
    for i in range(256):
        stream.feed(bounded_bytes(make_event(request, i)))
    with pytest.raises(ProtocolError) as error:
        stream.feed(bounded_bytes(make_event(request, 256)))
    assert error.value.reason_code == ReasonCode.EVENT_LIMIT_REACHED
    stream = WorkerEventStream(request, clock=lambda: 0.0)
    # At most 64 warnings; completed records can fill the remaining byte budget.
    # Exercise the independent byte guard using a long but valid request identity
    # only via the stored raw queue accounting boundary.
    stream._queue_bytes = 4 * 1024 * 1024
    with pytest.raises(ProtocolError) as error:
        stream.feed(bounded_bytes(make_event(request)))
    assert error.value.reason_code == ReasonCode.EVENT_LIMIT_REACHED


def test_rate_and_stage_counters_with_fake_clock():
    request = make_worker_request()
    tick = [0.0]
    stream = WorkerEventStream(request, clock=lambda: tick[0])
    for i in range(10):
        stream.feed(bounded_bytes(make_event(request, i, "progress", {"percent": i})))
    with pytest.raises(ProtocolError) as error:
        stream.feed(bounded_bytes(make_event(request, 10, "progress", {"percent": 10})))
    assert error.value.reason_code == ReasonCode.EVENT_LIMIT_REACHED
    tick[0] = 1.0
    stream = WorkerEventStream(request, clock=lambda: tick[0])
    for i in range(64):
        stream.feed(bounded_bytes(make_event(request, i, "warning", {"reason_code": "no_audio", "summary": "x"})))
    with pytest.raises(ProtocolError) as error:
        stream.feed(bounded_bytes(make_event(request, 64, "error", {"reason_code": "no_audio", "summary": "x"})))
    assert error.value.reason_code == ReasonCode.EVENT_LIMIT_REACHED
    stream = WorkerEventStream(request)
    for i in range(64):
        stream.feed(bounded_bytes(make_event(request, i, "artifact", {"slot": "downloaded_media", "bytes": i, "sha256": "a" * 64})))
    with pytest.raises(ProtocolError) as error:
        stream.feed(bounded_bytes(make_event(request, 64, "artifact", {"slot": "downloaded_media", "bytes": 64, "sha256": "a" * 64})))
    assert error.value.reason_code == ReasonCode.EVENT_LIMIT_REACHED


def test_emitter_and_payload_immutability():
    request = make_worker_request()
    payload = {"percent": 50}
    event = make_event(request, event="progress", payload=payload)
    payload["percent"] = 100
    assert event.payload["percent"] == 50
    with pytest.raises(TypeError):
        event.payload["percent"] = 100
    records = []
    emit_event(event, records.append)
    assert WorkerEventRecord.from_json(records[0][:-1]) == event
