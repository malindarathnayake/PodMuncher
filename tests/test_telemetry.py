"""Bounded local telemetry and audit: labels, redaction, caps, export, latch, pressure."""
import json
import shutil
import socket
import time
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from opentelemetry.sdk.metrics.export import MetricExportResult
from opentelemetry.sdk.trace.export import SpanExportResult

from podmuncher import config, telemetry
from podmuncher.domains import (
    Component,
    DomainValidationError,
    ReasonCode,
    StageOutcome,
)
from podmuncher.telemetry import MIB, TelemetryError

ROOT = Path(__file__).resolve().parents[1]
SID = "snapshot-" + "a" * 32
_HUBS = []


class FakeClock:
    def __init__(self):
        self.t = 0.0
        self._base = datetime(2026, 1, 1, tzinfo=UTC)

    def now(self):
        return (self._base + timedelta(seconds=self.t)).isoformat()

    def monotonic(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


class RealClock:
    def now(self):
        return datetime.now(UTC).isoformat()

    def monotonic(self):
        return time.monotonic()


class FakeCollector:
    """Scripted exporter: each call pops "ok"/"fail"; empty script means fail."""

    def __init__(self, clock, script=()):
        self.clock = clock
        self.script = list(script)
        self.attempts = []
        self.shut = False

    def export(self, payload, *args, **kwargs):
        self.attempts.append(self.clock.monotonic())
        ok = bool(self.script) and self.script.pop(0) == "ok"
        if hasattr(payload, "resource_metrics"):
            return MetricExportResult.SUCCESS if ok else MetricExportResult.FAILURE
        return SpanExportResult.SUCCESS if ok else SpanExportResult.FAILURE

    def shutdown(self):
        self.shut = True


@pytest.fixture
def fake_clock():
    return FakeClock()


@pytest.fixture
def fake_collector(fake_clock):
    def build(script=()):
        collectors = {}

        def factory(kind, endpoint, timeout):
            assert timeout <= 5.0
            collectors[kind] = FakeCollector(fake_clock, script)
            return collectors[kind]

        factory.collectors = collectors
        return factory

    return build


@pytest.fixture(autouse=True)
def _close_hubs():
    yield
    while _HUBS:
        _HUBS.pop().shutdown()


def make_sink(tmp_path, *, otlp=False, exporter_factory=None, clock=None, **kwargs):
    clock = clock or FakeClock()
    kwargs.setdefault("background", False)
    kwargs.setdefault("sleep", getattr(clock, "sleep", time.sleep))
    hub = telemetry.make_telemetry(
        tmp_path,
        config.Telemetry(otlp_enabled=otlp, otlp_endpoint="127.0.0.1:4317" if otlp else ""),
        clock=clock,
        exporter_factory=exporter_factory,
        **kwargs,
    )
    _HUBS.append(hub)
    return hub


def lines(hub, name):
    hub.flush()
    hub.flush()
    path = hub.diagnostics_dir / name
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_bytes().splitlines()]


def diag_bytes(hub):
    hub.flush()
    hub.flush()
    return b"".join(p.read_bytes() for p in sorted(hub.diagnostics_dir.iterdir()))


def valid_labels(name):
    return {key: min(values, key=str) for key, values in telemetry.METRICS[name][2].items()}


def test_high_cardinality_and_unknown_labels_rejected_and_counted(tmp_path):
    hub = make_sink(tmp_path)
    for name, (kind, _unit, labels) in telemetry.METRICS.items():
        value = 1 if kind == "counter" else 1.5
        before = hub.dropped["metric"]
        assert hub.metric(name, value, valid_labels(name)) is True
        assert hub.metric_points(name)
        assert hub.dropped["metric"] == before
        bad = [
            {**valid_labels(name), "job_id": str(uuid.uuid4())},
            {k: v for k, v in list(valid_labels(name).items())[1:]},
            {},
        ]
        for key in labels:
            bad.append({**valid_labels(name), key: "x" * 40})
            bad.append({**valid_labels(name), key: 999})
            bad.append({**valid_labels(name), key: True})
        for attrs in bad:
            assert hub.metric(name, value, attrs) is False
        assert hub.dropped["metric"] == before + len(bad)
        assert hub.metric(name, float("nan"), valid_labels(name)) is False
        assert hub.metric(name, -1, valid_labels(name)) is False
    assert hub.metric("podmuncher.unknown", 1, {}) is False
    assert hub.metric("podmuncher.jobs", 1.5, valid_labels("podmuncher.jobs")) is False
    assert hub.metric("podmuncher.jobs", 1, {"outcome": "queued", "source": "local"}) is False


def test_metric_label_sets_are_exact_and_use_registry_values():
    assert {k: set(v[2]) for k, v in telemetry.METRICS.items()} == {
        "podmuncher.jobs": {"outcome", "source"},
        "podmuncher.stage.duration": {"stage", "outcome"},
        "podmuncher.resource.used": {"resource", "device_slot"},
        "podmuncher.storage.free": {"volume_slot", "context"},
        "podmuncher.storage.reserved": {"volume_slot", "context"},
        "podmuncher.storage.required": {"volume_slot", "context"},
        "podmuncher.preflight": {"result", "reason"},
        "podmuncher.processing.speed": {"backend", "model_tier"},
    }
    jobs = telemetry.METRICS["podmuncher.jobs"][2]
    assert jobs["outcome"] == {"completed", "partial", "failed", "cancelled", "interrupted"}
    assert jobs["source"] == {"local", "direct", "youtube"}
    assert telemetry.METRICS["podmuncher.storage.free"][2]["volume_slot"] == set(range(8))
    assert telemetry.METRICS["podmuncher.resource.used"][2]["device_slot"] == {0}
    assert telemetry.METRICS["podmuncher.processing.speed"][2]["backend"] == {"cpu", "cuda"}


def test_unknown_span_names_are_noop_and_counted(tmp_path):
    hub = make_sink(tmp_path)
    for name in ("segment-42", "transcribe_segment", "", "Job"):
        with hub.span(name, {"stage": "transcribing"}) as span:
            assert span.is_noop
            span.set_attribute("x", 1)
            span.add_event("e")
    assert hub.dropped["span"] == 4
    assert lines(hub, "traces.jsonl") == []
    for name in sorted(telemetry.SPAN_NAMES):
        with hub.span(name) as span:
            assert not span.is_noop
    assert [r["name"] for r in lines(hub, "traces.jsonl")] == [
        "acquire_asset", "download", "export", "job", "preflight", "stage", "transcribe"]


@pytest.mark.parametrize("value", [
    "Bearer abc123SECRET", "Basic dXNlcjpwYXNzU0VDUkVU", "C:\\Users\\bob\\clip.wav",
    "d:/Users/bob/clip.wav", "/home/bob/clip.wav", "\\\\fileserver\\share\\bob.wav",
    "https://evil.example/a?token=zzz", "credential=hunter2 tail", "password: hunter2",
])
def test_redact_unit(value):
    out = telemetry.redact("before " + value + " after")
    assert out.startswith("before ") and "[redacted]" in out
    for secret in ("abc123SECRET", "dXNlcjpwYXNzU0VDUkVU", "bob", "fileserver", "evil",
                   "hunter2", "zzz"):
        assert secret not in out


def test_redact_keeps_plain_text():
    assert telemetry.redact("and/or 3/4 stage ok") == "and/or 3/4 stage ok"


def test_redaction_in_written_files(tmp_path):
    hub = make_sink(tmp_path)
    nasty = ("Bearer abc123SECRET C:\\Users\\bob\\clip.wav /home/bob/clip.wav "
             "\\\\fileserver\\share\\bob.wav https://evil.example/a?token=zzz")
    with hub.span("job", {
        "note": nasty, "title": "Secret Title", "transcript": "spoken words here",
        "text": "spoken words here", "url": "https://evil.example/x", "path": "C:\\Users\\bob",
        "credential": "hunter2", "stage": "transcribing",
    }) as span:
        span.set_attribute("source_url", "https://evil.example/y")
        span.add_event("note", {"detail": nasty, "title": "Secret Title"})
        hub.log(20, "redaction_check", component="telemetry", reason_code="ok", message=nasty,
                title="Secret Title", transcript="spoken words here", detail=nasty)
        with pytest.raises(ValueError), hub.span("stage", {"stage": "checking"}) as inner:
            inner.set_attribute("note", nasty)
            raise ValueError(nasty)
    data = diag_bytes(hub)
    assert b"[redacted]" in data
    for secret in (b"abc123SECRET", b"bob", b"fileserver", b"evil", b"Secret Title",
                   b"spoken words", b"hunter2", b"zzz", b"title", b"transcript"):
        assert secret not in data, secret
    trace = lines(hub, "traces.jsonl")
    job = next(r for r in trace if r["name"] == "job")
    assert job["attributes"]["stage"] == "transcribing"
    assert not {"title", "transcript", "text", "url", "path", "credential",
                "source_url"} & set(job["attributes"])
    stage = next(r for r in trace if r["name"] == "stage")
    exc = next(e for e in stage["events"] if e["name"] == "exception")
    assert exc["attributes"]["exception.type"] == "ValueError"
    assert len(exc["attributes"]["exception.summary"]) <= len("ValueError: ") + 256


def test_exception_summary_is_bounded():
    summary = telemetry.exception_summary(RuntimeError("x" * 5000))
    assert summary == "RuntimeError: " + "x" * 256


def test_bounded_queue_by_count_and_bytes():
    queue = telemetry._BoundedQueue(2048, 8 * MIB)
    assert [queue.offer(3 * MIB, i) for i in range(4)] == [True, True, False, False]
    assert queue.queued_bytes == 6 * MIB
    queue = telemetry._BoundedQueue(2048, 8 * MIB)
    assert sum(queue.offer(1, i) for i in range(3000)) == 2048
    assert len(queue) == 2048 and len(queue.drain()) == 2048 and len(queue) == 0


def test_span_queue_bound_at_2048_records(tmp_path):
    hub = make_sink(tmp_path)
    for _ in range(2048 + 5):
        with hub.span("export"):
            pass
    assert hub.dropped["span"] == 5
    assert len(hub._span_queue) == 2048
    hub.flush()
    assert len(lines(hub, "traces.jsonl")) == 2048


def test_span_queue_bound_at_8_mib(tmp_path):
    hub = make_sink(tmp_path)
    big = {f"k{i}": "a" * 1000 for i in range(32)}
    for _ in range(600):
        with hub.span("export", big):
            pass
    assert hub._span_queue.queued_bytes <= 8 * MIB
    assert 0 < len(hub._span_queue) < 2048
    assert hub.dropped["span"] == 600 - len(hub._span_queue)


def test_log_queue_bound(tmp_path):
    hub = make_sink(tmp_path)
    for _ in range(2048 + 7):
        hub.log(20, "bulk", component="telemetry", reason_code="ok")
    assert len(hub._log_queue) == 2048
    assert hub.dropped["log"] == 7


def test_export_three_attempts_then_drop_without_spool(tmp_path, fake_clock, fake_collector):
    factory = fake_collector()
    hub = make_sink(tmp_path, otlp=True, exporter_factory=factory, clock=fake_clock)
    for _ in range(3):
        with hub.span("export"):
            pass
    assert fake_clock.monotonic() == 0.0  # the caller was never blocked
    assert factory.collectors["span"].attempts == []
    hub.flush()
    attempts = factory.collectors["span"].attempts
    assert len(attempts) == 3 and attempts[-1] - attempts[0] <= 30.0
    assert hub.dropped["export"] == 3 and hub.export_attempts == 3
    assert any(r["event_name"] == "telemetry_dropped" and r["reason_code"] == "telemetry_dropped"
               for r in lines(hub, "logs.jsonl"))
    hub.flush()
    assert len(factory.collectors["span"].attempts) == 3  # nothing was retained for retry
    names = {p.name for p in tmp_path.rglob("*") if p.is_file()}
    assert names <= {"traces.jsonl", "logs.jsonl", "audit.jsonl"}


def test_export_succeeds_on_retry_within_window(tmp_path, fake_clock, fake_collector):
    factory = fake_collector(["fail", "ok"])
    hub = make_sink(tmp_path, otlp=True, exporter_factory=factory, clock=fake_clock)
    with hub.span("export"):
        pass
    hub.flush()
    assert len(factory.collectors["span"].attempts) == 2
    assert hub.dropped["export"] == 0


def test_failing_metric_export_is_bounded_and_counted(tmp_path, fake_clock, fake_collector):
    factory = fake_collector()
    hub = make_sink(tmp_path, otlp=True, exporter_factory=factory, clock=fake_clock)
    hub.metric("podmuncher.preflight", 1, {"result": "pass", "reason": "disk"})
    hub.flush()
    attempts = factory.collectors["metric"].attempts
    assert len(attempts) == 3 and attempts[-1] - attempts[0] <= 30.0
    assert hub.dropped["export"] == 1


def test_otlp_is_off_by_default(tmp_path, fake_clock, fake_collector):
    factory = fake_collector()
    hub = make_sink(tmp_path, otlp=False, exporter_factory=factory, clock=fake_clock)
    with hub.span("export"):
        pass
    hub.flush()
    assert factory.collectors == {} and hub.export_attempts == 0


def test_rotation_and_retention_keep_exact_segment_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(telemetry, "TRACE_CAP", (600, 2))
    monkeypatch.setattr(telemetry, "LOG_CAP", (500, 3))
    monkeypatch.setattr(telemetry, "AUDIT_CAP", (400, 3))
    hub = make_sink(tmp_path)
    for index in range(30):
        with hub.span("export", {"n": index}):
            pass
        hub.log(20, "rotate", component="telemetry", reason_code="ok", n=index)
        hub.audit("access", reason_code=ReasonCode.OK, ids={"artifact_id": str(uuid.uuid4())})
        hub.flush()
    names = sorted(p.name for p in hub.diagnostics_dir.iterdir())
    assert names == ["audit.jsonl", "audit.jsonl.1", "audit.jsonl.2", "logs.jsonl", "logs.jsonl.1",
                     "logs.jsonl.2", "traces.jsonl", "traces.jsonl.1"]
    caps = {"traces": 600, "logs": 500, "audit": 400}
    for path in hub.diagnostics_dir.iterdir():
        assert path.stat().st_size <= caps[path.name.split(".")[0]]


def test_production_caps_match_spec():
    assert telemetry.TRACE_CAP == (10 * MIB, 2)
    assert telemetry.LOG_CAP == (20 * MIB, 3)
    assert telemetry.AUDIT_CAP == (10 * MIB, 3)
    assert telemetry.TOTAL_CAP_BYTES == 512 * MIB
    assert (telemetry.QUEUE_MAX_RECORDS, telemetry.QUEUE_MAX_BYTES) == (2048, 8 * MIB)
    assert (telemetry.SPAN_MAX_EVENTS, telemetry.SPAN_MAX_ATTRIBUTES,
            telemetry.ATTRIBUTE_MAX_BYTES) == (64, 32, 1024)
    assert telemetry.EXPORT_ATTEMPTS == 3 and telemetry.EXPORT_WINDOW_SECONDS == 30.0


def test_span_limits(tmp_path):
    hub = make_sink(tmp_path)
    with hub.span("job", {f"k{i}": i for i in range(40)}) as span:
        for i in range(70):
            span.add_event(f"e{i}", {"v": i})
        span.set_attribute("wide", "é" * 700)
    record = lines(hub, "traces.jsonl")[0]
    assert len(record["attributes"]) == 32
    assert len(record["events"]) == 64
    with hub.span("job") as span:
        span.set_attribute("wide", "é" * 700)
        span.set_attribute("ascii", "a" * 3000)
    wide = lines(hub, "traces.jsonl")[1]["attributes"]
    assert len(wide["wide"].encode()) <= 1024 and len(wide["ascii"].encode()) <= 1024


def test_root_versus_child_contexts(tmp_path):
    hub = make_sink(tmp_path)
    job_id = str(uuid.uuid4())
    with hub.job_span(job_id):
        with hub.span("stage", {"stage": "checking"}), hub.span("download"):
            pass
        with hub.service_span("export"):
            pass
    with hub.service_span("acquire_asset"):
        pass
    by_name = {r["name"]: r for r in lines(hub, "traces.jsonl")}
    job, stage, download = by_name["job"], by_name["stage"], by_name["download"]
    assert job["parent_span_id"] is None and job["attributes"]["job_id"] == job_id
    assert stage["parent_span_id"] == job["span_id"] and stage["trace_id"] == job["trace_id"]
    assert download["parent_span_id"] == stage["span_id"]
    assert by_name["export"]["parent_span_id"] is None
    assert by_name["export"]["trace_id"] != job["trace_id"]
    assert by_name["acquire_asset"]["parent_span_id"] is None


def test_log_carries_core_and_project_fields(tmp_path):
    hub = make_sink(tmp_path)
    job_id, attempt_id = str(uuid.uuid4()), str(uuid.uuid4())
    with hub.job_span(job_id):
        hub.log(30, "stage_slow", component=Component.WORKER, reason_code=ReasonCode.OK,
                job_id=job_id, attempt_id=attempt_id, duration_ms=12, message="slow")
    hub.log(20, "outside", component="api", reason_code="ok")
    trace_id = lines(hub, "traces.jsonl")[0]["trace_id"]
    inside, outside = lines(hub, "logs.jsonl")
    assert inside["trace_id"] == trace_id and len(inside["span_id"]) == 16
    assert inside["severity_number"] == 13 and inside["event_name"] == "stage_slow"
    assert (inside["component"], inside["reason_code"], inside["job_id"],
            inside["attempt_id"], inside["duration_ms"]) == ("worker", "ok", job_id, attempt_id, 12)
    assert outside["trace_id"] is None and outside["severity_number"] == 9
    assert hub.log(20, "bad", component="not_a_component", reason_code="ok") is False
    assert hub.log(20, "bad", component="api", reason_code="not_a_reason") is False
    assert hub.dropped["log"] == 2


def test_warning_log_rate_limit(tmp_path, fake_clock):
    hub = make_sink(tmp_path, clock=fake_clock)
    sent = sum(hub.log(30, "burst", component="api", reason_code="ok") for _ in range(60))
    assert sent == 50 and hub.dropped["log"] == 10
    for _ in range(5):
        assert hub.log(20, "info", component="api", reason_code="ok")  # info is not limited
    fake_clock.sleep(1.0)
    assert sum(hub.log(40, "again", component="api", reason_code="ok") for _ in range(30)) == 10
    assert hub.dropped["log"] == 30


def test_audit_latch_exempt_actions_and_reset(tmp_path, monkeypatch):
    hub = make_sink(tmp_path)
    ident = {"artifact_id": str(uuid.uuid4())}
    assert hub.audit("access", reason_code=ReasonCode.OK, ids=ident) is True
    record = lines(hub, "audit.jsonl")[0]
    assert set(record) == {"ts", "action", "reason_code", "ids"} and record["ids"] == ident

    real_fsync = telemetry.os.fsync

    def broken(_fd):
        raise OSError("disk gone")

    monkeypatch.setattr(telemetry.os, "fsync", broken)
    with pytest.raises(TelemetryError) as caught:
        hub.audit("publish", reason_code=ReasonCode.OK, ids=ident)
    assert caught.value.reason_code is ReasonCode.AUDIT_LOSS
    assert hub.audit_blocked and hub.admission_blocked
    for action in ("cancel", "reap", "stop", "known_unused_removal"):
        assert hub.audit(action, reason_code=ReasonCode.OK, ids=ident) is False
    for _ in range(40):
        assert hub.audit("cancel", reason_code=ReasonCode.OK) is False
    notices = hub.audit_notices
    assert len(notices) == telemetry.AUDIT_NOTICE_LIMIT
    assert all(n.reason_code is ReasonCode.AUDIT_LOSS for n in notices)
    for action in ("submit", "config_change", "acquire", "publish"):
        with pytest.raises(TelemetryError) as caught:
            hub.audit(action, reason_code=ReasonCode.OK)
        assert caught.value.reason_code is ReasonCode.AUDIT_LOSS
    assert hub.reset_audit() is False and hub.audit_blocked  # probe fails while broken
    monkeypatch.setattr(telemetry.os, "fsync", real_fsync)
    assert hub.reset_audit() is True and not hub.admission_blocked
    assert hub.audit("submit", reason_code=ReasonCode.OK, ids=ident) is True
    assert any(r["reason_code"] == "audit_loss" for r in lines(hub, "logs.jsonl"))


def test_audit_rejects_malformed_input_without_latching(tmp_path):
    hub = make_sink(tmp_path)
    for kwargs in (
        {"action": "Bad Action", "reason_code": ReasonCode.OK},
        {"action": "access", "reason_code": "nope"},
        {"action": "access", "reason_code": ReasonCode.OK, "ids": {"artifact_id": "/etc/passwd"}},
        {"action": "access", "reason_code": ReasonCode.OK, "ids": {"Bad Key": str(uuid.uuid4())}},
    ):
        action = kwargs.pop("action")
        with pytest.raises(TelemetryError) as caught:
            hub.audit(action, **kwargs)
        assert caught.value.reason_code is ReasonCode.INVALID_INPUT
    assert not hub.audit_blocked


def test_pressure_policy_deletion_order(tmp_path):
    hub = make_sink(tmp_path, db_bytes=lambda: 50)
    diagnostics = hub.diagnostics_dir
    sizes = {"traces.jsonl": 100, "traces.jsonl.1": 100, "logs.jsonl": 100, "logs.jsonl.1": 100,
             "logs.jsonl.2": 100, "audit.jsonl": 100, "audit.jsonl.1": 100, "audit.jsonl.2": 100}
    for name, size in sizes.items():
        (diagnostics / name).write_bytes(b"x" * size)
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"a" * 10_000)

    assert hub.pressure_check() == telemetry.PressureReport(850, (), False, None)

    hub.total_cap = 700  # 850 -> drop the rotated trace first, then the oldest log segment
    report = hub.pressure_check()
    assert report.deleted == ("traces.jsonl.1", "logs.jsonl.2") and not report.blocked
    assert report.total_bytes == 650
    assert not (diagnostics / "traces.jsonl.1").exists()
    assert (diagnostics / "logs.jsonl.1").exists()

    hub.total_cap = 100  # cannot get there without touching audit/current files
    report = hub.pressure_check()
    assert report.deleted == ("logs.jsonl.1",)
    assert report.blocked and report.reason_code is ReasonCode.STORAGE_INSUFFICIENT
    assert report.total_bytes == 550
    assert sorted(p.name for p in diagnostics.iterdir()) == [
        "audit.jsonl", "audit.jsonl.1", "audit.jsonl.2", "logs.jsonl", "traces.jsonl"]
    assert artifact.stat().st_size == 10_000


def test_pressure_policy_counts_database_bytes(tmp_path):
    hub = make_sink(tmp_path, db_bytes=lambda: telemetry.TOTAL_CAP_BYTES + 1)
    report = hub.pressure_check()
    assert report.blocked and report.reason_code is ReasonCode.STORAGE_INSUFFICIENT


def test_scheduler_adapter(tmp_path):
    hub = make_sink(tmp_path)
    tel = hub.scheduler_telemetry()
    job_id, attempt_id, stage_id = (str(uuid.uuid4()) for _ in range(3))
    with tel.span("job", {"job_id": job_id}):
        with tel.span("preflight", {"stage": "checking", "job_id": job_id}):
            pass
        tel.count("preflight", {"result": "pass"})
        tel.count("preflight", {"result": "blocked", "reason": "disk"})
        with tel.span("stage", {"stage": "transcribing", "job_id": job_id,
                                "attempt_id": attempt_id, "stage_id": stage_id}):
            pass
        tel.count("stage", {"stage": "transcribing", "outcome": "ok"})
        tel.count("stage", {"stage": "exporting", "outcome": "error"})  # no span duration
    before = hub.dropped["metric"]
    tel.count("job", {"state": "queued"})  # not a terminal outcome: ignored
    assert hub.dropped["metric"] == before
    tel.count("job", {"state": "completed"})  # no source: dropped and counted
    assert hub.dropped["metric"] == before + 1
    tel.count("job", {"state": "completed", "source": "local"})
    tel.count("unknown", {})
    assert hub.metric_points("podmuncher.jobs") == [({"outcome": "completed", "source": "local"}, 1)]
    assert hub.metric_points("podmuncher.stage.duration") == [
        ({"stage": "transcribing", "outcome": "ok"}, 1)]
    assert sorted((a["result"], a["reason"]) for a, _ in hub.metric_points("podmuncher.preflight")) == [
        ("blocked", "disk"), ("pass", "unknown")]
    trace = {r["name"]: r for r in lines(hub, "traces.jsonl")}
    assert trace["job"]["parent_span_id"] is None
    assert trace["stage"]["parent_span_id"] == trace["job"]["span_id"]
    assert trace["stage"]["attributes"]["stage_id"] == stage_id


def test_artifact_and_recovery_audit_adapters(tmp_path, monkeypatch):
    hub = make_sink(tmp_path)
    artifact, recovery = hub.artifact_audit(), hub.recovery_audit()
    artifact("access", None, ReasonCode.OK)
    artifact("publish", str(uuid.uuid4()), ReasonCode.OK)
    recovery("startup_reconcile", None, ReasonCode.OK)
    assert [r["action"] for r in lines(hub, "audit.jsonl")] == [
        "access", "publish", "startup_reconcile"]
    with pytest.raises(TelemetryError):
        artifact("publish", "not-an-id", ReasonCode.OK)

    def broken(_fd):
        raise OSError

    monkeypatch.setattr(telemetry.os, "fsync", broken)
    with pytest.raises(TelemetryError):
        artifact("access", None, ReasonCode.OK)
    for adapter, action in ((recovery, "service_stop"), (artifact, "known_unused_removal")):
        # exempt under the latch: audit() returns False, so the adapter must report the loss
        with pytest.raises(TelemetryError) as caught:
            adapter(action, None, ReasonCode.OK)
        assert caught.value.reason_code is ReasonCode.AUDIT_LOSS
    with pytest.raises(TelemetryError):
        recovery("startup_reconcile", None, ReasonCode.OK)
    assert {n.action for n in hub.audit_notices} >= {"service_stop", "known_unused_removal"}


def test_cache_adapters(tmp_path, monkeypatch):
    hub = make_sink(tmp_path)
    sink, spans = hub.cache_audit(), hub.cache_spans()
    sink.record(action="stage", reason="verified_manifest_acquisition", asset_id=SID,
                failure_code="ok")
    sink.record(action="initialize", reason="cache_initialize", asset_id="cache", failure_code="ok")
    with spans.span("acquire_asset", asset_id=SID):
        pass
    with spans.span("acquire_asset", asset_id="C:\\Users\\bob\\cache"):
        pass
    audit = lines(hub, "audit.jsonl")
    assert audit[0]["ids"] == {"asset_id": SID} and audit[1]["ids"] == {}
    trace = lines(hub, "traces.jsonl")
    assert trace[0]["attributes"]["asset_id"] == SID and "asset_id" not in trace[1]["attributes"]
    assert b"bob" not in diag_bytes(hub)

    def broken(_fd):
        raise OSError

    monkeypatch.setattr(telemetry.os, "fsync", broken)
    with pytest.raises(TelemetryError):
        sink.record(action="stage", reason="x", asset_id=SID, failure_code="ok")
    sink.record(action="remove", reason="evict", asset_id=SID, failure_code="ok")  # exempt
    assert hub.audit_notices[-1].action == "known_unused_removal"


def test_preflight_adapter(tmp_path):
    hub = make_sink(tmp_path)
    pre = hub.preflight_telemetry()
    with pre.span("preflight", component=Component.STORAGE.value):
        pre.result(result="blocked", reason="disk", component="storage")
    with pre.span("preflight", component="https://evil.example"):
        pass
    assert hub.metric_points("podmuncher.preflight") == [({"result": "blocked", "reason": "disk"}, 1)]
    trace = lines(hub, "traces.jsonl")
    assert trace[0]["attributes"]["component"] == "storage"
    assert "component" not in trace[1]["attributes"]


def test_domain_negative_examples_rejected_by_real_parse():
    domains = json.loads((ROOT / "Docs" / "contract-domains.json").read_text(encoding="utf-8"))
    assert domains["negative_examples"]
    for enum in (ReasonCode, Component, StageOutcome):
        for bad in domains["negative_examples"]:
            with pytest.raises(DomainValidationError):
                enum.parse(bad)
    assert {c.value for c in Component} == set(domains["component"])
    assert {o.value for o in StageOutcome} == set(domains["stage_outcome"])


def test_background_pump_and_shutdown(tmp_path):
    hub = make_sink(tmp_path, clock=RealClock(), sleep=time.sleep, background=True)
    with hub.span("job"):
        hub.log(20, "pumped", component="api", reason_code="ok")
    hub.audit("access", reason_code=ReasonCode.OK)
    hub.shutdown()
    hub.shutdown()  # idempotent
    assert (hub.diagnostics_dir / "traces.jsonl").stat().st_size > 0
    assert b"pumped" in (hub.diagnostics_dir / "logs.jsonl").read_bytes()
    with hub.span("job") as span:  # after shutdown: no-op, never raises
        assert span.is_noop


@pytest.mark.parametrize("key", [
    "token", "password", "passwd", "secret", "api_key", "apikey", "auth", "authorization",
    "cookie", "Cookie", "access_token", "client_secret", "signing_key", "AUTH_header",
    "user.password",
])
def test_credential_named_keys_are_dropped(tmp_path, key):
    assert telemetry._clean_attrs({key: "hunter2", "index": 1}) == {"index": 1}
    hub = make_sink(tmp_path)
    with hub.span("stage", {key: "hunter2", "index": 1}) as span:
        span.set_attribute(key, "hunter2")
    hub.log(20, "x", component="runtime", reason_code="ok", **{key: "hunter2"})
    assert b"hunter2" not in diag_bytes(hub)


def test_exempt_actions_are_an_explicit_set(tmp_path):
    assert telemetry.EXEMPT_ACTIONS == {
        "cancel", "reap", "stop", "known_unused_removal", "service_stop"}
    for action in ("job_cancel", "orphan_reap", "worker_stop", "cancel_all"):
        assert telemetry.TelemetryHub._exempt(action) is False
    assert telemetry.TelemetryHub._exempt("service_stop") is True


def test_always_off_environment_sampler_still_records_spans(tmp_path, monkeypatch):
    monkeypatch.setenv("OTEL_TRACES_SAMPLER", "always_off")
    hub = make_sink(tmp_path)
    with hub.span("stage", {"stage": "checking"}) as span:
        span.set_attribute("index", 3)
        span.add_event("tick")
    record = next(r for r in lines(hub, "traces.jsonl") if r["name"] == "stage")
    assert record["attributes"]["index"] == 3 and record["trace_id"]


def test_span_handle_tolerates_non_recording_span():
    from opentelemetry import trace

    handle = telemetry._SpanHandle(trace.NonRecordingSpan(trace.INVALID_SPAN_CONTEXT))
    handle.set_attribute("index", 1)
    handle.set_attributes({"index": 2})
    handle.add_event("tick", {"index": 1})
    handle.record_exception(RuntimeError("x"))


def test_span_registry_attributes_are_validated_and_counted(tmp_path):
    hub = make_sink(tmp_path)
    with hub.span("stage", {"stage": "not_a_stage", "component": "runtime",
                            "outcome": "ok", "index": 1}) as span:
        span.set_attributes({"reason_code": "made_up", "component": "nope"})
        span.set_attribute("outcome", "completed")  # terminal job state is registry-valid
    attrs = next(r for r in lines(hub, "traces.jsonl") if r["name"] == "stage")["attributes"]
    assert "stage" not in attrs and "reason_code" not in attrs
    assert attrs["component"] == "runtime" and attrs["outcome"] == "completed"
    assert attrs["index"] == 1
    assert hub.dropped["span"] == 3


def test_pressure_check_blocks_when_storage_size_unknown(tmp_path):
    def broken():
        raise OSError("stat failed")

    report = make_sink(tmp_path, db_bytes=broken).pressure_check()
    assert report.blocked and report.reason_code is ReasonCode.STORAGE_UNKNOWN


def test_pressure_check_blocks_when_entry_stat_fails(tmp_path, monkeypatch):
    hub = make_sink(tmp_path)
    real = telemetry.os.scandir

    class Broken:
        name = "traces.jsonl"

        def is_file(self, **_kw):
            raise OSError("stat failed")

    monkeypatch.setattr(telemetry.os, "scandir", lambda path: [Broken(), *real(path)])
    report = hub.pressure_check()
    assert report.blocked and report.reason_code is ReasonCode.STORAGE_UNKNOWN


def test_default_exporters_disable_sdk_retries():
    from grpc import StatusCode

    for kind in ("span", "metric"):
        exporter = telemetry._default_exporter_factory(kind, "127.0.0.1:4317", 5.0)
        try:
            assert exporter._retryable_error_codes == frozenset({StatusCode.OK})
        finally:
            exporter.shutdown()


def test_real_exporter_makes_one_rpc_per_hub_attempt(tmp_path, fake_clock):
    """Hub is the single retry owner: 3 outer attempts == 3 RPCs, not 3 x SDK retries."""
    import grpc

    rpcs = []

    class Unavailable(grpc.RpcError):
        def code(self):
            return grpc.StatusCode.UNAVAILABLE

        def details(self):
            return "http://secret-host:4317 down"

        def trailing_metadata(self):
            return ()

    class Stub:
        def Export(self, *args, **kwargs):
            rpcs.append(1)
            raise Unavailable

    def factory(kind, endpoint, timeout):
        exporter = telemetry._default_exporter_factory(kind, endpoint, timeout)
        exporter._client = Stub()
        exporter._initialize_channel_and_stub = lambda: None  # keep the fake stub
        return exporter

    hub = make_sink(tmp_path, otlp=True, exporter_factory=factory, clock=fake_clock)
    with hub.span("stage", {"stage": "checking"}):
        pass
    hub.flush()
    assert len(rpcs) == telemetry.EXPORT_ATTEMPTS == hub.export_attempts
    assert hub.dropped["export"] >= 1


def test_exporter_logger_is_redacted_and_rate_limited(tmp_path, fake_clock, caplog):
    hub = make_sink(tmp_path, otlp=True, exporter_factory=lambda *a: None, clock=fake_clock)
    log = telemetry.logging.getLogger(telemetry.EXPORTER_LOGGER)
    seen = []

    class Capture(telemetry.logging.Handler):
        def emit(self, record):
            seen.append(record.getMessage())

    handler = Capture()
    log.addHandler(handler)
    try:
        log.error("Failed to export to %s: %s", "10.1.2.3:4317", "token=abc")
        log.error("Failed to export to %s: %s", "10.1.2.3:4317", "token=abc")
        fake_clock.t += telemetry.EXPORTER_LOG_INTERVAL_SECONDS
        log.warning("Transient error to %s", "10.1.2.3:4317")
    finally:
        log.removeHandler(handler)
    assert len(seen) == 2
    assert all("10.1.2.3" not in m and "token" not in m for m in seen)
    hub.shutdown()
    assert not [f for f in log.filters if isinstance(f, telemetry._ExporterLogFilter)]


def test_providers_are_private():
    from opentelemetry import metrics, trace
    from opentelemetry._logs import get_logger_provider

    assert type(trace.get_tracer_provider()).__name__ != "TracerProvider"
    assert type(metrics.get_meter_provider()).__name__ != "MeterProvider"
    assert type(get_logger_provider()).__name__ != "LoggerProvider"


@pytest.mark.live
def test_own_transport_smoke(tmp_path, monkeypatch):
    started = time.monotonic()
    monkeypatch.setattr(telemetry, "TRACE_CAP", (1500, 2))
    monkeypatch.setattr(telemetry, "LOG_CAP", (1500, 3))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]  # closed again: nothing listens on it
    attempts = []

    def recording_factory(kind, endpoint, timeout):
        real = telemetry._default_exporter_factory(kind, endpoint, timeout)

        class Recording:
            def export(self, payload, *args, **kwargs):
                attempts.append((kind, time.monotonic()))
                return real.export(payload, *args, **kwargs)

            def shutdown(self):
                real.shutdown()

        return Recording()

    state = tmp_path / "state"
    hub = telemetry.make_telemetry(
        state,
        config.Telemetry(otlp_enabled=True, otlp_endpoint=f"127.0.0.1:{port}"),
        clock=RealClock(),
        exporter_factory=recording_factory,
        background=False,
    )
    try:
        job_id = str(uuid.uuid4())
        with hub.job_span(job_id):
            for index in range(12):
                with hub.span("stage", {"stage": "checking", "index": index}):
                    hub.log(20, "smoke", component="runtime", reason_code="ok", index=index)
        hub.metric("podmuncher.jobs", 1, {"outcome": "completed", "source": "local"})
        hub.audit("access", reason_code=ReasonCode.OK, ids={"artifact_id": str(uuid.uuid4())})
        hub.flush()
        diagnostics = state / "diagnostics"
        assert (diagnostics / "traces.jsonl.1").exists()  # rotation was forced
        assert sorted(p.name for p in diagnostics.iterdir() if p.name.startswith("traces")) == [
            "traces.jsonl", "traces.jsonl.1"]
        span_attempts = [stamp for kind, stamp in attempts if kind == "span"]
        assert 1 <= len(span_attempts) <= 3
        assert span_attempts[-1] - span_attempts[0] <= 30.0
        assert hub.dropped["export"] > 0
        assert hub.metric_points("podmuncher.jobs")
    finally:
        hub.shutdown()
    names = {p.name for p in state.rglob("*") if p.is_file()}
    assert names <= {"traces.jsonl", "traces.jsonl.1", "logs.jsonl", "logs.jsonl.1",
                     "logs.jsonl.2", "audit.jsonl"}
    duration = time.monotonic() - started
    assert duration < 45.0
    shutil.rmtree(tmp_path, ignore_errors=True)
    print(f"export_attempts={len(span_attempts)} duration_s={duration:.2f} "
          f"dropped_export={hub.dropped['export']}")
    print("PODMUNCHER_SMOKE_OK")
