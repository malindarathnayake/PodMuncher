"""Bounded local telemetry and audit built on the OpenTelemetry API/SDK.

Contract (Docs/spec.md, quoted):

* spec.md:195 - "OTEL API/SDK for spans and metrics, with its logging bridge for
  structured logs; bounded local sinks; optional batched OTLP/gRPC, off by
  default. Core trace_id/span_id/severity_number; project fields event_name,
  component, reason_code, job_id, attempt_id, duration_ms (never claimed as
  backend reserved names). component and reason_code values come from the
  registry."
* spec.md:197 - spans job, stage, preflight, acquire_asset, download,
  transcribe, export. Stage spans are children of job; non-job service
  operations get root contexts. Span attributes are bounded
  component/stage/outcome; correlation ids appear only in logs/spans. Never
  include transcript text, raw URLs, titles, credentials, absolute user paths
  or raw downloader stderr.
* spec.md:199-206 - metrics with exact label sets: podmuncher.jobs,
  podmuncher.stage.duration, podmuncher.resource.used,
  podmuncher.storage.free/.reserved/.required, podmuncher.preflight and
  podmuncher.processing.speed (see ``METRICS`` below).
* spec.md:208 - caps: logs 20 MiB x 3, audit 10 MiB x 3, traces 10 MiB x 2;
  total control/diagnostics <= 512 MiB. Audit is an append-only stream. Audit
  loss blocks new jobs, acquisitions and config changes. Cancel, reap, stop and
  contained removal of already-classified unused cache remain possible and emit
  a bounded in-memory audit_loss, without claiming durability.
* spec.md:210 - each span/log queue holds <= 2048 records AND <= 8 MiB. Export:
  <= 3 attempts within 30 s, no persistent retry spool. Per span: <= 64 events,
  <= 32 attributes, <= 1024 UTF-8 bytes per attribute, with a bounded exception
  summary. Under pressure, drop traces/exports before eligible old logs/history;
  track the drop count with no high-cardinality labels. No per-segment traces.
* spec.md:174 - an unreachable OTLP collector gets bounded retries, then
  ``telemetry_dropped``; it must never stall a job or grow a spool.

Design notes.  The hub owns private tracer, meter and logger providers and never
sets a global provider.  Span ends and log emits only serialize a record and
offer it to a queue bounded by count and bytes; a daemon pump thread (or
``flush()``) writes the local sinks and runs the opt-in export, so a caller is
never blocked by disk pressure or an unreachable collector.  A segment cap such
as "10 MiB x 2" is the total number of files (current plus rotated).
"""

from __future__ import annotations

import collections
import contextlib
import json
import logging
import math
import os
import re
import threading
import time
import warnings
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

from opentelemetry import trace as trace_api
from opentelemetry.context import Context
from opentelemetry.sdk._logs import (
    LoggerProvider,
    LoggingHandler,
    LogRecordProcessor,
    ReadWriteLogRecord,
)
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import (
    ReadableSpan,
    SpanLimits,
    SpanProcessor,
    TracerProvider,
)
from opentelemetry.sdk.trace.sampling import ALWAYS_ON

from . import config as _config
from .domains import (
    Backend,
    Component,
    DomainValidationError,
    JobState,
    ModelTier,
    PreflightReason,
    PreflightResult,
    ReasonCode,
    Resource,
    Source,
    Stage,
    StageOutcome,
    VolumeSlot,
)
from .domains import (
    Context as DomainContext,
)

__all__ = [
    "EXEMPT_ACTIONS",
    "METRICS",
    "SPAN_NAMES",
    "AuditNotice",
    "PressureReport",
    "TelemetryError",
    "TelemetryHub",
    "make_telemetry",
    "redact",
]

MIB = 1024 * 1024
TRACE_CAP = (10 * MIB, 2)
LOG_CAP = (20 * MIB, 3)
AUDIT_CAP = (10 * MIB, 3)
TOTAL_CAP_BYTES = 512 * MIB
QUEUE_MAX_RECORDS = 2048
QUEUE_MAX_BYTES = 8 * MIB
EXPORT_ATTEMPTS = 3
EXPORT_WINDOW_SECONDS = 30.0
EXPORT_CALL_TIMEOUT_SECONDS = 5.0
EXPORT_BACKOFF_SECONDS = 0.5
EXPORT_BACKOFF_MAX_SECONDS = 4.0
EXPORT_BATCH_RECORDS = 512
METRIC_EXPORT_INTERVAL_SECONDS = 60.0
SHUTDOWN_JOIN_SECONDS = 40.0
PUMP_IDLE_SECONDS = 0.25
SPAN_MAX_ATTRIBUTES = 32
SPAN_MAX_EVENTS = 64
ATTRIBUTE_MAX_BYTES = 1024
EXCEPTION_SUMMARY_CHARS = 256
MAX_REDACT_CHARS = 8192
LOG_RATE_PER_SECOND = 10.0
LOG_RATE_BURST = 50.0
AUDIT_NOTICE_LIMIT = 32
DIAGNOSTICS_DIRNAME = "diagnostics"
REDACTED = "[redacted]"

SPAN_NAMES = frozenset(
    ("job", "stage", "preflight", "acquire_asset", "download", "transcribe", "export")
)
DROP_KINDS = ("span", "log", "metric", "export")
EXEMPT_ACTIONS = frozenset(
    ("cancel", "reap", "stop", "known_unused_removal", "service_stop")
)
_FORBIDDEN_KEY_PARTS = frozenset(
    ("title", "transcript", "text", "url", "path", "credential", "credentials",
     "token", "password", "passwd", "secret", "apikey", "auth", "authorization",
     "cookie")
)
# A trailing ``_key`` / ``_token`` / ``_secret`` marks a credential-named attribute.
_FORBIDDEN_KEY_SUFFIXES = frozenset(("key", "token", "secret"))
EXPORTER_LOGGER = "opentelemetry.exporter.otlp.proto.grpc.exporter"
EXPORTER_LOG_INTERVAL_SECONDS = 60.0
_PROJECT_LOG_FIELDS = (
    "event_name", "component", "reason_code", "job_id", "attempt_id", "duration_ms",
)
_TERMINAL_OUTCOMES = frozenset(
    state.value
    for state in (
        JobState.COMPLETED, JobState.PARTIAL, JobState.FAILED,
        JobState.CANCELLED, JobState.INTERRUPTED,
    )
)


def _values(enum: type[Enum]) -> frozenset[Any]:
    return frozenset(member.value for member in enum)


# name -> (kind, unit, {label: allowed values}); label sets are exact.
METRICS: Mapping[str, tuple[str, str, Mapping[str, frozenset[Any]]]] = {
    "podmuncher.jobs": ("counter", "1", {
        "outcome": _TERMINAL_OUTCOMES, "source": _values(Source)}),
    "podmuncher.stage.duration": ("histogram", "s", {
        "stage": _values(Stage), "outcome": _values(StageOutcome)}),
    "podmuncher.resource.used": ("gauge", "By", {
        "resource": _values(Resource), "device_slot": frozenset((0,))}),
    "podmuncher.storage.free": ("gauge", "By", {
        "volume_slot": _values(VolumeSlot), "context": _values(DomainContext)}),
    "podmuncher.storage.reserved": ("gauge", "By", {
        "volume_slot": _values(VolumeSlot), "context": _values(DomainContext)}),
    "podmuncher.storage.required": ("gauge", "By", {
        "volume_slot": _values(VolumeSlot), "context": _values(DomainContext)}),
    "podmuncher.preflight": ("counter", "1", {
        "result": _values(PreflightResult), "reason": _values(PreflightReason)}),
    "podmuncher.processing.speed": ("gauge", "1", {
        "backend": _values(Backend), "model_tier": _values(ModelTier)}),
}

# ----------------------------------------------------------------------------- errors


class TelemetryError(DomainValidationError):
    """A closed telemetry failure; the message never carries caller text."""

    def __init__(self, reason_code: ReasonCode) -> None:
        super().__init__(ReasonCode.parse(reason_code))


class Clock(Protocol):
    def now(self) -> str: ...

    def monotonic(self) -> float: ...


# ----------------------------------------------------------------------------- redaction

_URL = re.compile(r"\b[A-Za-z][A-Za-z0-9+.\-]*://\S*")
_UNC = re.compile(r"\\\\[^\s\\/]+(?:[\\/][^\s\\/\"']*)*")
_WINDOWS = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/][^\s\"'<>|]*")
_BEARER = re.compile(r"(?i)\b(?:bearer|basic)\s+[A-Za-z0-9._~+/=\-]+")
_CREDENTIAL = re.compile(
    r"(?i)(?:(?:credential|password|passwd|secret|token|api[_\-]?key)[\w\-]*"
    r"|authorization|auth)\s*[=:]\s*\S+"
)
_POSIX = re.compile(r"(?<![\w./:~\-])//?(?:[\w.@%+~\-]+/?)+")
_KEY = re.compile(r"[a-z][a-z0-9_.]{0,63}\Z")
_ID = re.compile(
    r"(?:snapshot-[0-9a-f]{32}|[0-9a-f]{32}|"
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\Z",
    re.IGNORECASE,
)
_SNAPSHOT_ID = re.compile(r"snapshot-[0-9a-f]{32}\Z")
_ACTION = re.compile(r"[a-z][a-z0-9_]{0,47}\Z")
_ROTATED = re.compile(r"(traces|logs)\.jsonl\.(\d+)\Z")


def redact(value: str) -> str:
    """Replace paths, URLs, bearer/basic tokens and credential pairs."""
    if type(value) is not str:
        value = str(value)
    value = value[:MAX_REDACT_CHARS]  # bound regex work on hostile input
    for pattern in (_URL, _UNC, _WINDOWS, _BEARER, _CREDENTIAL, _POSIX):
        value = pattern.sub(REDACTED, value)
    return value


def _clip(value: str, limit: int) -> str:
    raw = value.encode("utf-8")
    if len(raw) <= limit:
        return value
    return raw[:limit].decode("utf-8", "ignore")


def _forbidden_key(key: str) -> bool:
    parts = re.split(r"[_.\-]", key.lower())
    return any(part in _FORBIDDEN_KEY_PARTS for part in parts) or (
        len(parts) > 1 and parts[-1] in _FORBIDDEN_KEY_SUFFIXES)


def _clean_value(value: object) -> str | bool | int | float | None:
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, str):
        return _clip(redact(value), ATTRIBUTE_MAX_BYTES)
    return None


def _clean_attrs(attrs: Mapping[str, object] | None) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in (attrs or {}).items():
        if type(key) is not str or _KEY.fullmatch(key) is None or _forbidden_key(key):
            continue
        item = _clean_value(value)
        if item is not None:
            cleaned[key] = item
    return cleaned


_SPAN_REGISTRY_ATTRS: Mapping[str, frozenset[Any]] = {
    "component": _values(Component),
    "reason_code": _values(ReasonCode),
    "stage": _values(Stage),
    "outcome": _values(StageOutcome) | _TERMINAL_OUTCOMES,
}


def _clean_span_attrs(attrs: Mapping[str, object] | None) -> tuple[dict[str, Any], int]:
    """Clean span attributes; registry-backed names must carry registry values.

    Returns the kept attributes and how many registry-bound attributes were dropped.
    """
    cleaned = _clean_attrs(attrs)
    rejected = 0
    for key, allowed in _SPAN_REGISTRY_ATTRS.items():
        if key in cleaned and cleaned[key] not in allowed:
            del cleaned[key]
            rejected += 1
    return cleaned, rejected


def exception_summary(error: BaseException) -> str:
    """Exception type plus at most 256 redacted characters of its message."""
    message = redact(str(error))[:EXCEPTION_SUMMARY_CHARS]
    return f"{type(error).__name__}: {message}" if message else type(error).__name__


def _ident(value: object) -> str | None:
    return value if type(value) is str and _ID.fullmatch(value) else None


# ----------------------------------------------------------------------------- queue


class _BoundedQueue:
    """FIFO bounded by record count AND serialized bytes; overflow drops the newest."""

    def __init__(self, max_records: int, max_bytes: int) -> None:
        self._items: collections.deque[tuple[int, Any]] = collections.deque()
        self._max_records = max_records
        self._max_bytes = max_bytes
        self._bytes = 0
        self._lock = threading.Lock()

    def offer(self, size: int, item: Any) -> bool:
        with self._lock:
            if len(self._items) >= self._max_records or self._bytes + size > self._max_bytes:
                return False
            self._items.append((size, item))
            self._bytes += size
            return True

    def drain(self) -> list[Any]:
        with self._lock:
            items = [item for _size, item in self._items]
            self._items.clear()
            self._bytes = 0
            return items

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    @property
    def queued_bytes(self) -> int:
        with self._lock:
            return self._bytes


# ----------------------------------------------------------------------------- sinks


class _RotatingSink:
    """Append-only JSON-lines file with ``count`` total segments of ``max_bytes``."""

    def __init__(self, path: Path, max_bytes: int, count: int, *, durable: bool = False) -> None:
        self.path = path
        self._max_bytes = max_bytes
        self._count = max(1, count)
        self._durable = durable
        self._lock = threading.Lock()
        self._fh: Any = None
        self._size = 0

    def _open(self) -> None:
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_BINARY", 0)
        fd = os.open(self.path, flags, 0o600)
        self._fh = os.fdopen(fd, "ab")
        self._size = self._fh.seek(0, os.SEEK_END)
        if os.name == "posix":
            os.chmod(self.path, 0o600)

    def _segment(self, index: int) -> Path:
        return self.path if index == 0 else self.path.with_name(f"{self.path.name}.{index}")

    def _rotate(self) -> None:
        self._close()
        if self._count == 1:
            self.path.unlink(missing_ok=True)
        else:
            for index in range(self._count - 2, -1, -1):
                source = self._segment(index)
                if source.exists():
                    os.replace(source, self._segment(index + 1))
        self._open()

    def write(self, line: bytes) -> None:
        with self._lock:
            try:
                if self._fh is None:
                    self._open()
                if self._size > 0 and self._size + len(line) > self._max_bytes:
                    self._rotate()
                self._fh.write(line)
                self._fh.flush()
                if self._durable:
                    os.fsync(self._fh.fileno())
                self._size += len(line)
            except BaseException:
                self._close()
                raise

    def _close(self) -> None:
        fh, self._fh = self._fh, None
        if fh is not None:
            try:
                fh.close()
            except OSError:
                pass

    def close(self) -> None:
        with self._lock:
            self._close()


# ----------------------------------------------------------------------------- spans


class _NoopSpan:
    """Returned for an unknown span name; every operation is a no-op."""

    def set_attribute(self, key: str, value: object) -> None:
        return None

    def set_attributes(self, attrs: Mapping[str, object]) -> None:
        return None

    def add_event(self, name: str, attrs: Mapping[str, object] | None = None) -> None:
        return None

    def record_exception(self, error: BaseException) -> None:
        return None

    def get_span_context(self) -> trace_api.SpanContext:
        return trace_api.INVALID_SPAN_CONTEXT

    is_noop = True


class _SpanHandle:
    """Sanitizing facade over one SDK span (redaction and key policy)."""

    is_noop = False

    def __init__(
        self, span: trace_api.Span, on_reject: Callable[[int], None] | None = None,
    ) -> None:
        self.span = span
        self._on_reject = on_reject

    def _recording(self) -> bool:
        """A non-recording span (foreign sampler, ended span) accepts nothing."""
        return bool(self.span.is_recording())

    def _room(self, key: str) -> bool:
        """True when the SDK can take ``key`` without evicting (and warning about) another."""
        current = getattr(self.span, "attributes", None) or {}
        return key in current or len(current) < SPAN_MAX_ATTRIBUTES

    def _clean(self, attrs: Mapping[str, object] | None) -> dict[str, Any]:
        cleaned, rejected = _clean_span_attrs(attrs)
        if rejected and self._on_reject is not None:
            self._on_reject(rejected)
        return cleaned

    def set_attribute(self, key: str, value: object) -> None:
        self.set_attributes({key: value})

    def set_attributes(self, attrs: Mapping[str, object]) -> None:
        if not self._recording():
            return
        for name, item in self._clean(attrs).items():
            if self._room(name):
                self.span.set_attribute(name, item)

    def _event_room(self) -> bool:
        return len(getattr(self.span, "events", ())) < SPAN_MAX_EVENTS

    def add_event(self, name: str, attrs: Mapping[str, object] | None = None) -> None:
        if (type(name) is str and _KEY.fullmatch(name) and self._recording()
                and self._event_room()):
            self.span.add_event(
                name, dict(list(self._clean(attrs).items())[:SPAN_MAX_ATTRIBUTES]))

    def record_exception(self, error: BaseException) -> None:
        if not self._recording():
            return
        if self._event_room():
            self.span.add_event(
                "exception",
                {"exception.type": type(error).__name__,
                 "exception.summary": exception_summary(error)},
            )
        self.span.set_status(trace_api.Status(trace_api.StatusCode.ERROR))

    def get_span_context(self) -> trace_api.SpanContext:
        return self.span.get_span_context()


def _hex_ids(context: trace_api.SpanContext | None) -> tuple[str | None, str | None]:
    if context is None or not context.is_valid:
        return None, None
    return f"{context.trace_id:032x}", f"{context.span_id:016x}"


def _span_line(span: ReadableSpan) -> bytes:
    trace_id, span_id = _hex_ids(span.context)
    _parent_trace, parent_id = _hex_ids(span.parent)
    events = [
        {"name": event.name, "ts_ns": event.timestamp,
         "attributes": {k: v for k, v in dict(event.attributes or {}).items()}}
        for event in span.events
    ]
    record = {
        "trace_id": trace_id,
        "span_id": span_id,
        "parent_span_id": parent_id,
        "name": span.name,
        "start_ns": span.start_time,
        "end_ns": span.end_time,
        "status": span.status.status_code.name,
        "attributes": dict(span.attributes or {}),
        "events": events,
    }
    return (json.dumps(record, separators=(",", ":"), default=str) + "\n").encode("utf-8")


class _SpanProcessor(SpanProcessor):
    def __init__(self, hub: TelemetryHub) -> None:
        self._hub = hub

    def on_end(self, span: ReadableSpan) -> None:
        self._hub._enqueue_span(span)

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


class _LogProcessor(LogRecordProcessor):
    def __init__(self, hub: TelemetryHub) -> None:
        self._hub = hub

    def on_emit(self, log_record: ReadWriteLogRecord) -> None:
        self._hub._enqueue_log(log_record)

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def _log_line(log_record: ReadWriteLogRecord, ts: str) -> bytes:
    record = log_record.log_record
    attrs = dict(record.attributes or {})
    trace_id = f"{record.trace_id:032x}" if record.trace_id else None
    span_id = f"{record.span_id:016x}" if record.span_id else None
    body = record.body if isinstance(record.body, str) else str(record.body)
    out: dict[str, Any] = {
        "ts": ts,
        "trace_id": trace_id,
        "span_id": span_id,
        "severity_number": record.severity_number.value if record.severity_number else None,
        "severity_text": record.severity_text,
    }
    for key in _PROJECT_LOG_FIELDS:
        if key in attrs:
            item = _clean_value(attrs[key])
            if item is not None:
                out[key] = item
    out["message"] = _clip(redact(body), ATTRIBUTE_MAX_BYTES * 4)
    fields = {
        key[len("fields."):]: attrs[key]
        for key in attrs
        if key.startswith("fields.")
    }
    out["fields"] = _clean_attrs(fields)
    return (json.dumps(out, separators=(",", ":"), default=str) + "\n").encode("utf-8")


# ----------------------------------------------------------------------------- reports


@dataclass(frozen=True, slots=True)
class AuditNotice:
    ts: str
    action: str
    reason_code: ReasonCode
    ids: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class PressureReport:
    total_bytes: int
    deleted: tuple[str, ...]
    blocked: bool
    reason_code: ReasonCode | None


class _TokenBucket:
    def __init__(self, rate: float, burst: float, clock: Clock) -> None:
        self._rate, self._burst, self._clock = rate, burst, clock
        self._tokens = burst
        self._last = clock.monotonic()
        self._lock = threading.Lock()

    def take(self) -> bool:
        with self._lock:
            now = self._clock.monotonic()
            self._tokens = min(self._burst, self._tokens + max(0.0, now - self._last) * self._rate)
            self._last = now
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False


# ----------------------------------------------------------------------------- hub


class _ExporterLogFilter(logging.Filter):
    """Replace SDK exporter log text (endpoint, error details) and rate-limit it."""

    def __init__(self, clock: Clock) -> None:
        super().__init__()
        self._clock = clock
        self._last: float | None = None
        self._lock = threading.Lock()

    def filter(self, record: logging.LogRecord) -> bool:
        with self._lock:
            now = self._clock.monotonic()
            if self._last is not None and now - self._last < EXPORTER_LOG_INTERVAL_SECONDS:
                return False
            self._last = now
        record.msg = f"otlp exporter {record.levelname.lower()} (details withheld)"
        record.args = None
        record.exc_info = None
        record.exc_text = None
        record.stack_info = None
        return True


def _default_exporter_factory(kind: str, endpoint: str, timeout: float) -> Any:
    if not endpoint:
        return None
    # The SDK exporter retries only codes in this set, and an empty set silently restores
    # its defaults; a set no failed RPC can match makes the hub the single retry owner.
    from grpc import StatusCode
    no_retry = frozenset({StatusCode.OK})
    host = endpoint.split("://", 1)[-1].rsplit(":", 1)[0].strip("[]")
    insecure = True if endpoint.startswith("http://") or host in (
        "127.0.0.1", "localhost", "::1") else None
    if kind == "span":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
            OTLPSpanExporter,
        )
        return OTLPSpanExporter(
            endpoint=endpoint, insecure=insecure, timeout=timeout,
            retryable_error_codes=no_retry)
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
        OTLPMetricExporter,
    )
    return OTLPMetricExporter(
        endpoint=endpoint, insecure=insecure, timeout=timeout,
        retryable_error_codes=no_retry)


class TelemetryHub:
    """Owns the private OTEL providers, bounded sinks, audit latch and drop counters."""

    def __init__(
        self,
        state_dir: str | os.PathLike[str],
        config: _config.Telemetry,
        *,
        clock: Clock,
        exporter_factory: Callable[[str, str, float], Any] | None = None,
        db_bytes: Callable[[], int] = lambda: 0,
        background: bool = True,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._clock = clock
        self._sleep = sleep
        self._db_bytes = db_bytes
        self.total_cap = TOTAL_CAP_BYTES
        self.dropped: dict[str, int] = {kind: 0 for kind in DROP_KINDS}
        self.export_attempts = 0
        self._counter_lock = threading.Lock()
        self._last_drop_note: dict[str, float] = {}
        self._stage_seconds: dict[str, float] = {}

        self.diagnostics_dir = Path(state_dir) / DIAGNOSTICS_DIRNAME
        self.diagnostics_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._traces = _RotatingSink(self.diagnostics_dir / "traces.jsonl", *TRACE_CAP)
        self._logs = _RotatingSink(self.diagnostics_dir / "logs.jsonl", *LOG_CAP)
        self._audit_sink = _RotatingSink(
            self.diagnostics_dir / "audit.jsonl", *AUDIT_CAP, durable=True)

        self._span_queue = _BoundedQueue(QUEUE_MAX_RECORDS, QUEUE_MAX_BYTES)
        self._log_queue = _BoundedQueue(QUEUE_MAX_RECORDS, QUEUE_MAX_BYTES)
        self._drain_lock = threading.Lock()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._closed = False
        self._bucket = _TokenBucket(LOG_RATE_PER_SECOND, LOG_RATE_BURST, clock)

        self._audit_lock = threading.Lock()
        self.audit_blocked = False
        self._notices: collections.deque[AuditNotice] = collections.deque(
            maxlen=AUDIT_NOTICE_LIMIT)

        self._metric_reader = InMemoryMetricReader()
        self._meter_provider = MeterProvider(
            metric_readers=[self._metric_reader], shutdown_on_exit=False)
        self._tracer_provider = TracerProvider(
            span_limits=SpanLimits(
                max_span_attributes=SPAN_MAX_ATTRIBUTES,
                max_events=SPAN_MAX_EVENTS,
                max_links=0,
                max_event_attributes=SPAN_MAX_ATTRIBUTES,
                max_span_attribute_length=ATTRIBUTE_MAX_BYTES,
            ),
            sampler=ALWAYS_ON,  # every job/stage trace, whatever OTEL_TRACES_SAMPLER says
            shutdown_on_exit=False,
            meter_provider=self._meter_provider,
        )
        self._tracer_provider.add_span_processor(_SpanProcessor(self))
        self._tracer = self._tracer_provider.get_tracer("podmuncher.telemetry")
        self._logger_provider = LoggerProvider(
            shutdown_on_exit=False, meter_provider=self._meter_provider)
        self._logger_provider.add_log_record_processor(_LogProcessor(self))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            handler = LoggingHandler(level=logging.NOTSET, logger_provider=self._logger_provider)
        self._pylogger = logging.Logger(  # noqa: LOG001 - unregistered: no global logger-manager entry
            f"podmuncher.telemetry.{id(self):x}", logging.DEBUG)
        self._pylogger.addHandler(handler)
        self._pylogger.propagate = False

        meter = self._meter_provider.get_meter("podmuncher.telemetry")
        self._instruments: dict[str, Any] = {}
        for name, (kind, unit, _labels) in METRICS.items():
            make = {"counter": meter.create_counter, "histogram": meter.create_histogram,
                    "gauge": meter.create_gauge}[kind]
            self._instruments[name] = make(name, unit=unit)

        self._span_exporter: Any = None
        self._metric_exporter: Any = None
        self._next_metric_export = clock.monotonic() + METRIC_EXPORT_INTERVAL_SECONDS
        self._exporter_filter: _ExporterLogFilter | None = None
        if config.otlp_enabled:
            self._exporter_filter = _ExporterLogFilter(clock)
            logging.getLogger(EXPORTER_LOGGER).addFilter(self._exporter_filter)
            factory = exporter_factory or _default_exporter_factory
            try:
                self._span_exporter = factory(
                    "span", config.otlp_endpoint, EXPORT_CALL_TIMEOUT_SECONDS)
                self._metric_exporter = factory(
                    "metric", config.otlp_endpoint, EXPORT_CALL_TIMEOUT_SECONDS)
            except Exception:  # noqa: BLE001 - opt-in export must never block startup
                self._span_exporter = self._metric_exporter = None
                self._note_drop("export")

        self._thread: threading.Thread | None = None
        if background:
            self._thread = threading.Thread(
                target=self._pump, name="podmuncher-telemetry", daemon=True)
            self._thread.start()

    # ------------------------------------------------------------------ drops

    def _note_drop(self, kind: str, count: int = 1) -> None:
        with self._counter_lock:
            self.dropped[kind] += count
        if kind == "log":
            return
        now = self._clock.monotonic()
        last = self._last_drop_note.get(kind)
        if last is None or now - last >= 1.0:
            self._last_drop_note[kind] = now
            self.log(logging.WARNING, "telemetry_dropped", component=Component.TELEMETRY,
                     reason_code=ReasonCode.TELEMETRY_DROPPED, kind=kind)

    @property
    def dropped_total(self) -> int:
        return sum(self.dropped.values())

    # ------------------------------------------------------------------ spans

    @contextmanager
    def span(
        self, name: str, attrs: Mapping[str, object] | None = None, *, root: bool = False,
    ) -> Iterator[Any]:
        if self._closed or type(name) is not str or name not in SPAN_NAMES:
            self._note_drop("span")
            yield _NoopSpan()
            return
        started = self._clock.monotonic()
        start_attrs, rejected = _clean_span_attrs(attrs)
        if rejected:
            self._note_drop("span", rejected)
        raw = self._tracer.start_span(
            name,
            context=Context() if root else None,
            attributes=dict(list(start_attrs.items())[:SPAN_MAX_ATTRIBUTES]),
            record_exception=False,
            set_status_on_exception=False,
        )
        handle = _SpanHandle(raw, lambda count: self._note_drop("span", count))
        stage = attrs.get("stage") if name == "stage" and attrs else None
        stage = stage.value if isinstance(stage, Enum) else stage
        try:
            with trace_api.use_span(
                raw, end_on_exit=False, record_exception=False, set_status_on_exception=False,
            ):
                yield handle
        except BaseException as error:
            handle.record_exception(error)
            raise
        finally:
            elapsed = max(0.0, self._clock.monotonic() - started)
            handle.set_attribute("duration_ms", int(elapsed * 1000))
            if type(stage) is str and stage in METRICS["podmuncher.stage.duration"][2]["stage"]:
                self._stage_seconds[stage] = elapsed
            raw.end()

    def service_span(self, name: str, attrs: Mapping[str, object] | None = None):
        """Non-job service operation: always a root context."""
        return self.span(name, attrs, root=True)

    def job_span(self, job_id: str, attrs: Mapping[str, object] | None = None):
        """The job root; stage spans opened inside it become its children."""
        merged = dict(attrs or {})
        merged["job_id"] = job_id
        return self.span("job", merged, root=True)

    def _enqueue_span(self, span: ReadableSpan) -> None:
        try:
            line = _span_line(span)
        except Exception:  # noqa: BLE001 - serialization must never reach the caller
            self._note_drop("span")
            return
        if not self._span_queue.offer(len(line), (line, span)):
            self._note_drop("span")
            return
        self._wake.set()

    # ------------------------------------------------------------------ logs

    def log(
        self,
        level: int,
        event_name: str,
        *,
        component: Component | str,
        reason_code: ReasonCode | str,
        message: str = "",
        job_id: str | None = None,
        attempt_id: str | None = None,
        duration_ms: int | None = None,
        **fields: object,
    ) -> bool:
        """Emit one structured log record; invalid fields or a rate limit drop it."""
        if self._closed:
            return False
        try:
            extra: dict[str, Any] = {
                "event_name": _clean_value(event_name) or "",
                "component": Component.parse(component).value,
                "reason_code": ReasonCode.parse(reason_code).value,
            }
        except DomainValidationError:
            self._note_drop("log")
            return False
        if job_id is not None:
            extra["job_id"] = job_id
        if attempt_id is not None:
            extra["attempt_id"] = attempt_id
        if duration_ms is not None:
            extra["duration_ms"] = duration_ms
        for key, value in _clean_attrs(fields).items():
            extra[f"fields.{key}"] = value
        if level >= logging.WARNING and not self._bucket.take():
            self._note_drop("log")
            return False
        try:
            self._pylogger.log(level, message or event_name, extra=extra)
        except Exception:  # noqa: BLE001 - logging never raises into callers
            self._note_drop("log")
            return False
        return True

    def _enqueue_log(self, log_record: ReadWriteLogRecord) -> None:
        try:
            line = _log_line(log_record, self._clock.now())
        except Exception:  # noqa: BLE001
            self._note_drop("log")
            return
        if not self._log_queue.offer(len(line), line):
            self._note_drop("log")
            return
        self._wake.set()

    # ------------------------------------------------------------------ metrics

    def metric(self, name: str, value: float, attrs: Mapping[str, object]) -> bool:
        """Record one measurement; unknown names, labels or label values are dropped."""
        spec = METRICS.get(name) if type(name) is str else None
        if spec is None or not isinstance(attrs, Mapping):
            self._note_drop("metric")
            return False
        kind, _unit, labels = spec
        if set(attrs) != set(labels):
            self._note_drop("metric")
            return False
        clean: dict[str, Any] = {}
        for key, allowed in labels.items():
            item = attrs[key]
            if isinstance(item, Enum):
                item = item.value
            if type(item) not in (str, int) or item not in allowed or isinstance(item, bool):
                self._note_drop("metric")
                return False
            clean[key] = item
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            self._note_drop("metric")
            return False
        if not math.isfinite(value) or value < 0 or (kind == "counter" and type(value) is not int):
            self._note_drop("metric")
            return False
        instrument = self._instruments[name]
        if kind == "counter":
            instrument.add(value, clean)
        elif kind == "histogram":
            instrument.record(value, clean)
        else:
            instrument.set(value, clean)
        return True

    def metric_points(self, name: str) -> list[tuple[dict[str, Any], float]]:
        """Local read-back: (attributes, value-or-count) for one metric."""
        data = self._metric_reader.get_metrics_data()
        points: list[tuple[dict[str, Any], float]] = []
        for resource in (data.resource_metrics if data else ()):
            for scope in resource.scope_metrics:
                for metric in scope.metrics:
                    if metric.name != name:
                        continue
                    for point in metric.data.data_points:
                        value = getattr(point, "value", None)
                        points.append((dict(point.attributes),
                                       point.count if value is None else value))
        return points

    # ------------------------------------------------------------------ pump / export

    def _pump(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(PUMP_IDLE_SECONDS)
            self._wake.clear()
            with contextlib.suppress(Exception):  # the pump never dies on a sink fault
                self._drain(force=False)
        with contextlib.suppress(Exception):
            self._drain(force=True)

    def flush(self) -> None:
        """Synchronously write queued records and run due exports (bounded)."""
        self._drain(force=True)

    def _drain(self, *, force: bool) -> None:
        with self._drain_lock:
            batch: list[ReadableSpan] = []
            for line, span in self._span_queue.drain():
                try:
                    self._traces.write(line)
                except OSError:
                    self._note_drop("span")
                batch.append(span)
            for line in self._log_queue.drain():
                try:
                    self._logs.write(line)
                except OSError:
                    with self._counter_lock:
                        self.dropped["log"] += 1
            span_failed = False
            if batch and self._span_exporter is not None:
                for start in range(0, len(batch), EXPORT_BATCH_RECORDS):
                    chunk = batch[start:start + EXPORT_BATCH_RECORDS]
                    if span_failed:
                        # The collector just exhausted its attempts; do not retry the rest.
                        self._note_drop("export", len(chunk))
                    elif not self._export(self._span_exporter.export, chunk, len(chunk)):
                        span_failed = True
            if self._metric_exporter is not None and not span_failed and (
                force or self._clock.monotonic() >= self._next_metric_export
            ):
                self._next_metric_export = (
                    self._clock.monotonic() + METRIC_EXPORT_INTERVAL_SECONDS)
                data = self._metric_reader.get_metrics_data()
                if data is not None and data.resource_metrics:
                    self._export(self._metric_exporter.export, data, 1)

    def _export(self, call: Callable[[Any], Any], payload: Any, count: int) -> bool:
        """At most 3 attempts inside 30 s with bounded backoff; then drop, never spool."""
        start = self._clock.monotonic()
        backoff = EXPORT_BACKOFF_SECONDS
        for attempt in range(1, EXPORT_ATTEMPTS + 1):
            self.export_attempts += 1
            try:
                ok = getattr(call(payload), "name", None) == "SUCCESS"
            except Exception:  # noqa: BLE001 - exporter faults are transport failures
                ok = False
            if ok:
                return True
            if attempt == EXPORT_ATTEMPTS:
                break
            elapsed = self._clock.monotonic() - start
            if elapsed + backoff + EXPORT_CALL_TIMEOUT_SECONDS > EXPORT_WINDOW_SECONDS:
                break
            self._sleep(backoff)
            backoff = min(backoff * 2, EXPORT_BACKOFF_MAX_SECONDS)
        self._note_drop("export", count)
        return False

    # ------------------------------------------------------------------ audit

    @property
    def admission_blocked(self) -> bool:
        return self.audit_blocked

    @property
    def audit_notices(self) -> tuple[AuditNotice, ...]:
        with self._audit_lock:
            return tuple(self._notices)

    @staticmethod
    def _exempt(action: str) -> bool:
        return action in EXEMPT_ACTIONS

    def audit(
        self,
        action: str,
        *,
        reason_code: ReasonCode | str,
        ids: Mapping[str, str] | None = None,
    ) -> bool:
        """Append one durable audit record.

        Returns True when the record was written and fsynced.  While the latch is
        set, exempt actions return False after a bounded in-memory notice; every
        other action raises ``TelemetryError(AUDIT_LOSS)``.
        """
        if type(action) is not str or _ACTION.fullmatch(action) is None:
            raise TelemetryError(ReasonCode.INVALID_INPUT)
        try:
            reason = ReasonCode.parse(reason_code)
        except DomainValidationError:
            raise TelemetryError(ReasonCode.INVALID_INPUT) from None
        clean_ids: dict[str, str] = {}
        for key, value in (ids or {}).items():
            if (type(key) is not str or _KEY.fullmatch(key) is None
                    or _ident(value) is None or len(clean_ids) >= 8):
                raise TelemetryError(ReasonCode.INVALID_INPUT)
            clean_ids[key] = value
        exempt = self._exempt(action)
        ts = self._clock.now()
        with self._audit_lock:
            if not self.audit_blocked:
                line = json.dumps(
                    {"ts": ts, "action": action, "reason_code": reason.value, "ids": clean_ids},
                    separators=(",", ":"), sort_keys=True,
                ).encode("utf-8") + b"\n"
                try:
                    self._audit_sink.write(line)
                    return True
                except Exception:  # noqa: BLE001 - any write/fsync failure latches
                    self.audit_blocked = True
            self._notices.append(AuditNotice(
                ts, action, ReasonCode.AUDIT_LOSS, tuple(sorted(clean_ids.items()))))
        self.log(logging.ERROR, "audit_loss", component=Component.TELEMETRY,
                 reason_code=ReasonCode.AUDIT_LOSS, kind="audit")
        if exempt:
            return False
        raise TelemetryError(ReasonCode.AUDIT_LOSS)

    def reset_audit(self) -> bool:
        """Clear the latch only after a successful probe write."""
        probe = json.dumps(
            {"ts": self._clock.now(), "action": "audit_probe",
             "reason_code": ReasonCode.OK.value, "ids": {}},
            separators=(",", ":"), sort_keys=True,
        ).encode("utf-8") + b"\n"
        with self._audit_lock:
            try:
                self._audit_sink.write(probe)
            except Exception:  # noqa: BLE001
                self.audit_blocked = True
                return False
            self.audit_blocked = False
            return True

    # ------------------------------------------------------------------ pressure

    def pressure_check(self) -> PressureReport:
        """Free space under the 512 MiB aggregate cap; never touches audit or artifacts."""
        files: dict[str, tuple[int, int | None, str]] = {}
        try:
            entries = list(os.scandir(self.diagnostics_dir))
        except OSError:
            entries = []
        total = 0
        unknown = False
        for entry in entries:
            try:
                if not entry.is_file(follow_symlinks=False):
                    continue
                size = entry.stat(follow_symlinks=False).st_size
            except OSError:
                unknown = True  # an unreadable entry is never counted as zero bytes
                continue
            total += size
            match = _ROTATED.fullmatch(entry.name)
            files[entry.name] = (size, int(match.group(2)) if match else None,
                                 match.group(1) if match else "")
        try:
            total += max(0, int(self._db_bytes()))
        except Exception:  # noqa: BLE001 - an unknown database size is never zero
            unknown = True
        deleted: list[str] = []

        def sweep(kind: str) -> None:
            nonlocal total
            rotated = sorted(
                ((index, name, size) for name, (size, index, group) in files.items()
                 if group == kind and index is not None),
                reverse=True,
            )
            for _index, name, size in rotated:
                if total <= self.total_cap:
                    return
                try:
                    (self.diagnostics_dir / name).unlink()
                except OSError:
                    continue
                total -= size
                deleted.append(name)

        if total > self.total_cap:
            sweep("traces")
        if total > self.total_cap:
            sweep("logs")
        if unknown:
            return PressureReport(total, tuple(deleted), True, ReasonCode.STORAGE_UNKNOWN)
        blocked = total > self.total_cap
        return PressureReport(
            total, tuple(deleted), blocked,
            ReasonCode.STORAGE_INSUFFICIENT if blocked else None)

    # ------------------------------------------------------------------ adapters

    def scheduler_telemetry(self) -> _SchedulerTelemetry:
        return _SchedulerTelemetry(self)

    def artifact_audit(self) -> Callable[[str, str | None, ReasonCode], None]:
        def audit(action: str, artifact_id: str | None, reason_code: ReasonCode) -> None:
            ids = {} if artifact_id is None else {"artifact_id": artifact_id}
            if not self.audit(action, reason_code=reason_code, ids=ids):
                raise TelemetryError(ReasonCode.AUDIT_LOSS)
        return audit

    def recovery_audit(self) -> Callable[[str, str | None, ReasonCode], None]:
        def audit(action: str, target_id: str | None, reason_code: ReasonCode) -> None:
            ids = {} if _ident(target_id) is None else {"target_id": target_id}
            if not self.audit(action, reason_code=reason_code, ids=ids):
                raise TelemetryError(ReasonCode.AUDIT_LOSS)
        return audit

    def cache_audit(self) -> _CacheAudit:
        return _CacheAudit(self)

    def cache_spans(self) -> _CacheSpans:
        return _CacheSpans(self)

    def preflight_telemetry(self) -> _PreflightTelemetry:
        return _PreflightTelemetry(self)

    # ------------------------------------------------------------------ shutdown

    def shutdown(self) -> None:
        """Flush and close everything; bounded by the pump join and export window."""
        if self._closed:
            return
        self._closed = True
        self._tracer_provider.shutdown()
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(SHUTDOWN_JOIN_SECONDS)
        else:
            self._drain(force=True)
        for exporter in (self._span_exporter, self._metric_exporter):
            if exporter is not None:
                with contextlib.suppress(Exception):
                    exporter.shutdown()
        if self._exporter_filter is not None:
            logging.getLogger(EXPORTER_LOGGER).removeFilter(self._exporter_filter)
        self._logger_provider.shutdown()
        self._meter_provider.shutdown()
        for sink in (self._traces, self._logs, self._audit_sink):
            sink.close()


class _SchedulerTelemetry:
    """Implements ``scheduler.Telemetry``."""

    def __init__(self, hub: TelemetryHub) -> None:
        self._hub = hub

    def span(self, name: str, attrs: Mapping[str, str]):
        return self._hub.span(name, attrs, root=(name == "job"))

    def count(self, name: str, attrs: Mapping[str, str]) -> None:
        hub = self._hub
        if name == "job":
            state = attrs.get("state")
            if state not in _TERMINAL_OUTCOMES:
                return
            source = attrs.get("source")
            if source is None:
                hub._note_drop("metric")
                return
            hub.metric("podmuncher.jobs", 1, {"outcome": state, "source": source})
        elif name == "stage":
            stage = attrs.get("stage")
            stage = stage.value if isinstance(stage, Enum) else stage
            seconds = hub._stage_seconds.pop(stage, None) if type(stage) is str else None
            if seconds is None:
                hub._note_drop("metric")
                return
            hub.metric("podmuncher.stage.duration", seconds,
                       {"stage": stage, "outcome": attrs.get("outcome")})
        elif name == "preflight":
            result = attrs.get("result")
            reason = attrs.get("reason", PreflightReason.UNKNOWN.value
                               if result == PreflightResult.BLOCKED.value else None)
            if result == PreflightResult.PASS.value and reason is None:
                reason = PreflightReason.UNKNOWN.value
            hub.metric("podmuncher.preflight", 1, {"result": result, "reason": reason})
        else:
            hub._note_drop("metric")


class _CacheAudit:
    """Implements ``model_cache.AuditSink``."""

    def __init__(self, hub: TelemetryHub) -> None:
        self._hub = hub

    def record(self, *, action: str, reason: str, asset_id: str, failure_code: str) -> None:
        # The cache only removes entries it has already classified as unused.
        mapped = "known_unused_removal" if action == "remove" else action
        ids = {"asset_id": asset_id} if _SNAPSHOT_ID.fullmatch(str(asset_id)) else {}
        self._hub.audit(mapped, reason_code=failure_code, ids=ids)


class _CacheSpans:
    """Implements ``model_cache.SpanSink``."""

    def __init__(self, hub: TelemetryHub) -> None:
        self._hub = hub

    def span(self, name: str, *, asset_id: str):
        attrs = {"asset_id": asset_id} if _SNAPSHOT_ID.fullmatch(str(asset_id)) else {}
        return self._hub.span(name, attrs)


class _PreflightTelemetry:
    """Implements ``volumes.PreflightTelemetry``."""

    def __init__(self, hub: TelemetryHub) -> None:
        self._hub = hub

    def span(self, name: str, *, component: str):
        try:
            attrs = {"component": Component.parse(component).value}
        except DomainValidationError:
            attrs = {}
        return self._hub.span(name, attrs)

    def result(self, *, result: str, reason: str, component: str) -> None:
        self._hub.metric("podmuncher.preflight", 1, {"result": result, "reason": reason})


def make_telemetry(
    state_dir: str | os.PathLike[str],
    config: _config.Telemetry,
    *,
    clock: Clock,
    exporter_factory: Callable[[str, str, float], Any] | None = None,
    db_bytes: Callable[[], int] = lambda: 0,
    background: bool = True,
    sleep: Callable[[float], None] = time.sleep,
) -> TelemetryHub:
    """Build a hub with private providers; OTLP export only when ``config.otlp_enabled``."""
    return TelemetryHub(
        state_dir, config, clock=clock, exporter_factory=exporter_factory,
        db_bytes=db_bytes, background=background, sleep=sleep,
    )
