"""Version 1 worker JSONL contract; no host paths or runtime side effects.

Request keys are schema_version, job_id, attempt_id, stage_id, capability,
operation, assigned_slots, source, outputs, input_basename, model_snapshot_id,
limits, selection. Source is null or exactly {kind,input}. Assigned slots map
closed intermediate worker slots to unique private basenames. Events have
schema_version, job_id, attempt_id, stage_id, sequence, event, payload.
Payloads: started/completed {}; progress {percent} (0..100, informational);
artifact {slot,bytes,sha256}; warning/error {reason_code,summary}. An artifact
is only an intermediate file claim; the consumer must verify it independently.
Completion is solely the explicit completed event, never progress coverage.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
import json
import math
import re
import time
from types import MappingProxyType
from typing import Callable, Mapping
import unicodedata

from .config import Config, ConfigError, Limits
from .domains import (Capability, DomainValidationError, ReasonCode, WorkerEvent as WorkerEventKind,
                      WorkerSlot, validate_json_payload, validate_uuid_string)
from .requests import RequestError, RequestOutputs, RequestSelection, classify_source


MAX_RECORD_BYTES = 65536
MAX_QUEUED_EVENTS = 256
MAX_QUEUED_BYTES = 4 * 1024 * 1024
_REQUEST_KEYS = frozenset(("schema_version", "job_id", "attempt_id", "stage_id",
                           "capability", "operation", "assigned_slots", "source",
                           "outputs", "input_basename", "model_snapshot_id", "limits", "selection"))
_EVENT_KEYS = frozenset(("schema_version", "job_id", "attempt_id", "stage_id",
                         "sequence", "event", "payload"))
_REMOTE_OPERATIONS = frozenset(("metadata", "download"))
_LOCAL_OPERATIONS = frozenset(("probe", "transcribe"))
_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,127}\Z")
_BASENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,254}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")
_SUMMARY_SIDEBAND = re.compile(r"(?i)(?:[a-z][a-z0-9+.-]*://|[a-z]:[\\/]|\\\\|(?:^|\s)/(?:[^\s]+)|\b[A-Za-z_][A-Za-z_0-9]*=)")


class ProtocolError(DomainValidationError):
    """Safe, latched worker protocol failure without untrusted text."""

    def __init__(self, reason_code: ReasonCode = ReasonCode.WORKER_PROTOCOL_MISMATCH) -> None:
        super().__init__(reason_code)


def _fail(reason: ReasonCode = ReasonCode.WORKER_PROTOCOL_MISMATCH) -> None:
    raise ProtocolError(reason)


def _exact(value: object, keys: frozenset[str]) -> dict:
    if type(value) is not dict or value.keys() != keys or any(type(key) is not str for key in value):
        _fail()
    return value


def _uuid(value: object) -> str:
    try:
        return validate_uuid_string(value)
    except DomainValidationError:
        _fail()


def _identifier(value: object) -> str:
    if type(value) is not str or _IDENTIFIER.fullmatch(value) is None or value in (".", ".."):
        _fail()
    return value


def _basename(value: object) -> str:
    if type(value) is not str or _BASENAME.fullmatch(value) is None or value in (".", "..") or ".." in value:
        _fail()
    return value


def _wire_bytes(data: dict) -> bytes:
    try:
        validate_json_payload(data, max_depth=8, max_string_bytes=8192)
        result = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except ProtocolError:
        raise
    except (DomainValidationError, ValueError, UnicodeError, OverflowError, RecursionError):
        _fail()
    if len(result) > MAX_RECORD_BYTES:
        _fail(ReasonCode.EVENT_LIMIT_REACHED)
    return result


def _unique(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            _fail()
        result[key] = value
    return result


def _decode(record: bytes | str) -> dict:
    try:
        if type(record) is bytes:
            if len(record) > MAX_RECORD_BYTES:
                _fail(ReasonCode.EVENT_LIMIT_REACHED)
            record = record.decode("utf-8", "strict")
        if type(record) is not str or len(record.encode("utf-8", "strict")) > MAX_RECORD_BYTES:
            _fail(ReasonCode.EVENT_LIMIT_REACHED)
        value = json.loads(record, object_pairs_hook=_unique, parse_constant=lambda _: _fail())
        validate_json_payload(value, max_depth=8, max_string_bytes=8192)
    except ProtocolError:
        raise
    except (UnicodeError, ValueError, OverflowError, RecursionError, DomainValidationError):
        _fail()
    if type(value) is not dict:
        _fail()
    return value


@dataclass(frozen=True, slots=True)
class WorkerRequest:
    schema_version: int
    job_id: str
    attempt_id: str
    stage_id: str
    capability: str
    operation: str
    assigned_slots: Mapping[str, str]
    source: Mapping[str, str] | None
    outputs: RequestOutputs
    input_basename: str | None
    model_snapshot_id: str | None
    limits: Limits
    selection: RequestSelection

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            _fail()
        for value in (self.job_id, self.attempt_id, self.stage_id):
            _uuid(value)
        try:
            Capability.parse(self.capability)
        except DomainValidationError:
            _fail()
        operation = _identifier(self.operation)
        if type(self.assigned_slots) not in (dict, MappingProxyType):
            _fail()
        slots = {}
        for slot, basename in self.assigned_slots.items():
            if type(slot) is not str:
                _fail()
            try:
                WorkerSlot.parse(slot)
            except DomainValidationError:
                _fail()
            slots[slot] = _basename(basename)
        if len(set(slots.values())) != len(slots):
            _fail()
        object.__setattr__(self, "assigned_slots", MappingProxyType(slots))
        if self.source is not None:
            source = _exact(self.source, frozenset(("kind", "input")))
            if source["kind"] not in ("direct", "youtube"):
                _fail()
            try:
                admitted = classify_source(source["input"])
            except RequestError:
                _fail()
            if (admitted.kind, admitted.input) != (source["kind"], source["input"]):
                _fail()
            object.__setattr__(self, "source", MappingProxyType(dict(source)))
        if self.input_basename is not None:
            _basename(self.input_basename)
        if self.model_snapshot_id is not None:
            _identifier(self.model_snapshot_id)
        if operation in _REMOTE_OPERATIONS:
            if self.capability != "media" or self.source is None or self.input_basename is not None:
                _fail()
        elif operation in _LOCAL_OPERATIONS:
            if self.source is not None or (self.input_basename is None and not (operation == "probe" and self.capability in ("asr_cpu", "asr_cuda"))):
                _fail()
        elif self.source is not None or self.input_basename is not None:
            _fail()
        if type(self.outputs) is not RequestOutputs or type(self.selection) is not RequestSelection or type(self.limits) is not Limits:
            _fail()
        try:
            RequestOutputs(**asdict(self.outputs))
            RequestSelection(**asdict(self.selection))
            Config.from_dict({"limits": {name: getattr(self.limits, name)
                                         for name in Limits.__dataclass_fields__}})
        except (RequestError, ConfigError):
            _fail()
        for name in ("model", "language"):
            _identifier(getattr(self.selection, name))

    @classmethod
    def from_dict(cls, value: object) -> WorkerRequest:
        data = _exact(value, _REQUEST_KEYS)
        for name, keys in (("outputs", RequestOutputs.__dataclass_fields__),
                           ("selection", RequestSelection.__dataclass_fields__),
                           ("limits", Limits.__dataclass_fields__)):
            _exact(data[name], frozenset(keys))
        try:
            return cls(**{**data, "outputs": RequestOutputs(**data["outputs"]),
                          "selection": RequestSelection(**data["selection"]),
                          "limits": Limits(**data["limits"])})
        except (TypeError, RequestError, ConfigError):
            _fail()

    @classmethod
    def from_json(cls, record: bytes | str) -> WorkerRequest:
        return cls.from_dict(_decode(record))

    def to_dict(self) -> dict:
        return {"schema_version": self.schema_version, "job_id": self.job_id,
                "attempt_id": self.attempt_id, "stage_id": self.stage_id,
                "capability": self.capability, "operation": self.operation,
                "assigned_slots": dict(self.assigned_slots),
                "source": None if self.source is None else dict(self.source),
                "outputs": asdict(self.outputs), "input_basename": self.input_basename,
                "model_snapshot_id": self.model_snapshot_id, "limits": asdict(self.limits),
                "selection": asdict(self.selection)}

    def to_json(self) -> bytes:
        return _wire_bytes(self.to_dict())


@dataclass(frozen=True, slots=True)
class WorkerEventRecord:
    schema_version: int
    job_id: str
    attempt_id: str
    stage_id: str
    sequence: int
    event: str
    payload: Mapping[str, object]

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            _fail()
        for value in (self.job_id, self.attempt_id, self.stage_id):
            _uuid(value)
        if type(self.sequence) is not int or self.sequence < 0:
            _fail()
        try:
            WorkerEventKind.parse(self.event)
        except DomainValidationError:
            _fail()
        if type(self.payload) not in (dict, MappingProxyType):
            _fail()
        payload = dict(self.payload)
        if self.event in ("started", "completed"):
            _exact(payload, frozenset())
        elif self.event == "progress":
            _exact(payload, frozenset(("percent",)))
            percent = payload["percent"]
            if type(percent) not in (int, float) or (type(percent) is float and not math.isfinite(percent)) or not 0 <= percent <= 100:
                _fail()
        elif self.event == "artifact":
            _exact(payload, frozenset(("slot", "bytes", "sha256")))
            try:
                WorkerSlot.parse(payload["slot"])
            except DomainValidationError:
                _fail()
            if type(payload["bytes"]) is not int or payload["bytes"] < 0 or type(payload["sha256"]) is not str or _SHA256.fullmatch(payload["sha256"]) is None:
                _fail()
        else:
            _exact(payload, frozenset(("reason_code", "summary")))
            try:
                reason = ReasonCode.parse(payload["reason_code"])
            except DomainValidationError:
                _fail()
            summary = payload["summary"]
            if (type(summary) is not str or len(summary) > 1024 or _CONTROL.search(summary)
                    or _SUMMARY_SIDEBAND.search(summary)
                    or any(unicodedata.category(char).startswith("C") for char in summary)):
                _fail()
            try:
                if len(summary.encode("utf-8", "strict")) > 4096:
                    _fail()
            except UnicodeError:
                _fail()
            payload["summary"] = reason.name.replace("_", " ").capitalize()
        _wire_bytes({"payload": payload})
        object.__setattr__(self, "payload", MappingProxyType(payload))

    @classmethod
    def from_dict(cls, value: object) -> WorkerEventRecord:
        data = _exact(value, _EVENT_KEYS)
        return cls(**data)

    @classmethod
    def from_json(cls, record: bytes | str) -> WorkerEventRecord:
        return cls.from_dict(_decode(record))

    def to_dict(self) -> dict:
        return {"schema_version": self.schema_version, "job_id": self.job_id,
                "attempt_id": self.attempt_id, "stage_id": self.stage_id,
                "sequence": self.sequence, "event": self.event, "payload": dict(self.payload)}

    def to_json(self) -> bytes:
        return _wire_bytes(self.to_dict())


@dataclass(slots=True)
class WorkerEventStream:
    """Bounded byte feeder; pop transfers ownership and frees queue capacity."""

    request: WorkerRequest
    clock: Callable[[], float] = time.monotonic
    fault_emitter: Callable[[ReasonCode], object] | None = None
    _partial: bytearray = field(default_factory=bytearray, init=False)
    _queue: deque[tuple[WorkerEventRecord, int]] = field(default_factory=deque, init=False)
    _queue_bytes: int = field(default=0, init=False)
    _sequence: int = field(default=-1, init=False)
    _progress_times: deque[float] = field(default_factory=deque, init=False)
    _warnings: int = field(default=0, init=False)
    _artifacts: int = field(default=0, init=False)
    _failure: ProtocolError | None = field(default=None, init=False)
    _finished: bool = field(default=False, init=False)

    @property
    def failure(self) -> ProtocolError | None:
        return self._failure

    def _reject(self, reason: ReasonCode = ReasonCode.WORKER_PROTOCOL_MISMATCH) -> None:
        if self._failure is not None:
            raise self._failure
        self._failure = ProtocolError(reason)
        if self.fault_emitter is not None:
            try:
                self.fault_emitter(reason)
            except Exception:
                pass  # Telemetry cannot replace or hide the original failure.
        raise self._failure

    def feed(self, data: bytes) -> None:
        if self._failure is not None:
            raise self._failure
        if self._finished:
            self._reject()
        if type(data) is not bytes:
            self._reject()
        for byte in data:
            if byte == 10:
                record = bytes(self._partial)
                self._partial.clear()
                if record.endswith(b"\r"):
                    self._reject()
                try:
                    event = WorkerEventRecord.from_json(record)
                    self._admit(event, len(record) + 1)
                except ProtocolError as error:
                    self._reject(error.reason_code)
            else:
                self._partial.append(byte)
                if len(self._partial) > MAX_RECORD_BYTES:
                    self._reject(ReasonCode.EVENT_LIMIT_REACHED)

    def _admit(self, event: WorkerEventRecord, size: int) -> None:
        if (event.job_id, event.attempt_id, event.stage_id) != (self.request.job_id, self.request.attempt_id, self.request.stage_id) or event.sequence <= self._sequence:
            self._reject()
        if event.event == "artifact":
            if event.payload["slot"] not in self.request.assigned_slots:
                self._reject()
            self._artifacts += 1
            if self._artifacts > 64:
                self._reject(ReasonCode.EVENT_LIMIT_REACHED)
        if event.event in ("warning", "error"):
            self._warnings += 1
            if self._warnings > 64:
                self._reject(ReasonCode.EVENT_LIMIT_REACHED)
        if event.event == "progress":
            now = self.clock()
            if type(now) not in (int, float) or not math.isfinite(now):
                self._reject()
            while self._progress_times and now - self._progress_times[0] >= 1:
                self._progress_times.popleft()
            self._progress_times.append(now)
            if len(self._progress_times) > 10:
                self._reject(ReasonCode.EVENT_LIMIT_REACHED)
        if len(self._queue) >= MAX_QUEUED_EVENTS or self._queue_bytes + size > MAX_QUEUED_BYTES:
            self._reject(ReasonCode.EVENT_LIMIT_REACHED)
        self._queue.append((event, size))
        self._queue_bytes += size
        self._sequence = event.sequence

    def pop(self) -> WorkerEventRecord | None:
        if not self._queue:
            return None
        event, size = self._queue.popleft()
        self._queue_bytes -= size
        return event

    def finish(self) -> None:
        if self._failure is not None:
            raise self._failure
        if self._partial:
            self._reject()
        self._finished = True


def emit_event(event: WorkerEventRecord, sink: Callable[[bytes], object]) -> None:
    """Validate before writing one complete JSONL record to an injected sink."""
    if type(event) is not WorkerEventRecord or not callable(sink):
        _fail()
    sink(event.to_json() + b"\n")


WorkerEvent = WorkerEventRecord


__all__ = ("ProtocolError", "WorkerRequest", "WorkerEvent", "WorkerEventRecord", "WorkerEventStream",
           "emit_event", "MAX_RECORD_BYTES", "MAX_QUEUED_EVENTS", "MAX_QUEUED_BYTES")
