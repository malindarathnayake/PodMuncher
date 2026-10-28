"""Immutable version-one state records and explicit protected/public projections.

``to_dict``/``from_dict`` and JSON methods are protected persistence codecs:
they retain admitted request data, internal paths, and provenance. ``public_dict``
omits internal paths and request input; ``telemetry_dict`` is an independent,
minimal allowlist. Neither projection should be used to restore state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import ntpath
import re
from types import MappingProxyType
from typing import Any, Generic, TypeVar

from .domains import (
    ArtifactFormat, ArtifactSlot, BudgetConfidence, Context, Device, DomainValidationError,
    FieldStatus, JobState, ReasonCode, ServiceState, Stage, StageOutcome, VolumeSlot,
    WorkerSlot, validate_finite_number, validate_json_payload,
    validate_nonnegative_int, validate_percentage, validate_schema_version,
    validate_utc_timestamp, validate_utf8_string, validate_uuid_string,
)
from .requests import JobRequest


T = TypeVar("T")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_MAX_JSON_BYTES = 335544320  # 600 bounded snapshots, including eight populated volumes each.


def _keys(value: object, names: tuple[str, ...]) -> dict[str, Any]:
    if type(value) is not dict or value.keys() != set(names) or any(type(k) is not str for k in value):
        raise DomainValidationError()
    return value


def _string(value: object, *, empty: bool = False) -> str:
    result = validate_utf8_string(value)
    if not empty and not result:
        raise DomainValidationError()
    return result


def _optional(value: object, validator: Any) -> Any:
    return None if value is None else validator(value)


def _enum(kind: type, value: object) -> Any:
    return kind.parse(value)


def _hash(value: object) -> str:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise DomainValidationError()
    return value


def _bool(value: object) -> bool:
    if type(value) is not bool:
        raise DomainValidationError()
    return value


def _freeze(value: object) -> object:
    validate_json_payload(value)
    if type(value) is dict:
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if type(value) is list:
        return tuple(_freeze(item) for item in value)
    return value


def _record(value: dict) -> None:
    """Apply the JSON tree limits to the complete persisted record."""
    validate_json_payload(value)
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        if len(encoded.encode("utf-8")) > _MAX_JSON_BYTES:
            raise DomainValidationError(ReasonCode.REQUEST_TOO_LARGE)
    except (OverflowError, ValueError):
        raise DomainValidationError() from None


def _thaw(value: object) -> object:
    if isinstance(value, MappingProxyType):
        return {key: _thaw(item) for key, item in value.items()}
    if type(value) is tuple:
        return [_thaw(item) for item in value]
    return value


def _read_json(payload: str | bytes) -> object:
    def unique(pairs: list[tuple[str, object]]) -> dict:
        result: dict = {}
        for key, value in pairs:
            if key in result:
                raise DomainValidationError()
            result[key] = value
        return result

    try:
        if type(payload) is bytes:
            if len(payload) > _MAX_JSON_BYTES:
                raise DomainValidationError(ReasonCode.REQUEST_TOO_LARGE)
            payload = payload.decode("utf-8", "strict")
        if type(payload) is not str:
            raise DomainValidationError()
        if len(payload.encode("utf-8", "strict")) > _MAX_JSON_BYTES:
            raise DomainValidationError(ReasonCode.REQUEST_TOO_LARGE)
        value = json.loads(payload, object_pairs_hook=unique,
                           parse_constant=lambda _: (_ for _ in ()).throw(DomainValidationError()))
        validate_json_payload(value)
        return value
    except RecursionError:
        raise DomainValidationError(ReasonCode.NESTING_LIMIT) from None
    except DomainValidationError:
        raise
    except (UnicodeError, ValueError):
        raise DomainValidationError() from None


class _Codec:
    @classmethod
    def from_json(cls, payload: str | bytes, **kwargs: object) -> Any:
        return cls.from_dict(_read_json(payload), **kwargs)

    def to_json(self) -> str:
        result = self.to_dict()
        _record(result)
        return json.dumps(result, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False)


@dataclass(frozen=True, slots=True)
class Measurement(_Codec, Generic[T]):
    value: T | None
    source: str
    status: FieldStatus
    sample_time: str
    reason: ReasonCode | None

    def __post_init__(self) -> None:
        if self.value is not None:
            object.__setattr__(self, "value", _freeze(self.value))
        _string(self.source)
        object.__setattr__(self, "status", _enum(FieldStatus, self.status))
        validate_utc_timestamp(self.sample_time)
        object.__setattr__(self, "reason", _optional(self.reason, lambda x: _enum(ReasonCode, x)))
        if self.status is FieldStatus.OK:
            if self.value is None or self.reason is not None:
                raise DomainValidationError()
        elif self.value is not None or self.reason is None or self.reason is ReasonCode.OK:
            raise DomainValidationError()
        _record(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> Measurement:
        validate_json_payload(data)
        d = _keys(data, ("value", "source", "status", "sample_time", "reason"))
        return cls(d["value"], d["source"], _enum(FieldStatus, d["status"]),
                   d["sample_time"], _optional(d["reason"], lambda x: _enum(ReasonCode, x)))

    def to_dict(self) -> dict:
        return {"value": _thaw(self.value), "source": self.source, "status": self.status.value,
                "sample_time": self.sample_time, "reason": self.reason.value if self.reason else None}


def _measurement(value: object, kind: str) -> Measurement:
    item = Measurement.from_dict(value) if type(value) is dict else value
    if type(item) is not Measurement:
        raise DomainValidationError()
    if item.value is not None:
        if kind == "bytes":
            validate_nonnegative_int(item.value)
        elif kind == "duration":
            if validate_finite_number(item.value) < 0:
                raise DomainValidationError()
        elif kind == "percentage":
            validate_percentage(item.value)
        elif kind == "service":
            _enum(ServiceState, item.value)
        elif kind == "bool":
            _bool(item.value)
        elif kind == "string":
            _string(item.value)
        elif kind == "number":
            validate_finite_number(item.value)
    return item


@dataclass(frozen=True, slots=True)
class StorageEntry(_Codec):
    volume_slot: VolumeSlot
    volume_identity: str
    context: Context
    capacity: Measurement[int]
    free: Measurement[int]
    reserve: Measurement[int]
    reserved: Measurement[int]
    required: Measurement[int]
    available_after_job: Measurement[int]
    source: str
    sample_time: str
    backing_volume_slot: VolumeSlot | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "volume_slot", _enum(VolumeSlot, self.volume_slot))
        _string(self.volume_identity)
        object.__setattr__(self, "context", _enum(Context, self.context))
        for name in ("capacity", "free", "reserve", "reserved", "required", "available_after_job"):
            object.__setattr__(self, name, _measurement(getattr(self, name), "bytes"))
        _string(self.source)
        validate_utc_timestamp(self.sample_time)
        object.__setattr__(self, "backing_volume_slot", _optional(self.backing_volume_slot, lambda x: _enum(VolumeSlot, x)))
        _record(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> StorageEntry:
        validate_json_payload(data)
        names = ("volume_slot", "volume_identity", "context", "capacity", "free", "reserve",
                 "reserved", "required", "available_after_job", "source", "sample_time", "backing_volume_slot")
        d = _keys(data, names)
        return cls(_enum(VolumeSlot, d["volume_slot"]), d["volume_identity"], _enum(Context, d["context"]),
                   *(_measurement(d[n], "bytes") for n in names[3:9]), d["source"], d["sample_time"],
                   _optional(d["backing_volume_slot"], lambda x: _enum(VolumeSlot, x)))

    def to_dict(self) -> dict:
        result = {"volume_slot": int(self.volume_slot), "volume_identity": self.volume_identity,
                  "context": self.context.value, "source": self.source, "sample_time": self.sample_time,
                  "backing_volume_slot": int(self.backing_volume_slot) if self.backing_volume_slot is not None else None}
        for name in ("capacity", "free", "reserve", "reserved", "required", "available_after_job"):
            result[name] = getattr(self, name).to_dict()
        return result

    def public_dict(self) -> dict:
        result = self.to_dict()
        result.pop("volume_identity")
        result.pop("source")
        for name in ("capacity", "free", "reserve", "reserved", "required", "available_after_job"):
            result[name].pop("source")
        return result


_RESOURCE_MEASURES = {
    "service": "service", "docker": "bool", "worker": "bool", "host_ram": "bytes",
    "worker_ram": "bytes", "cpu": "percentage", "gpu_capability": "bool",
    "gpu_utilization": "percentage", "gpu_vram": "bytes", "model": "string",
    "progress": "percentage", "elapsed_seconds": "duration", "stage_eta_seconds": "duration",
}


@dataclass(frozen=True, slots=True)
class ResourceSnapshot(_Codec):
    schema_version: int
    sample_time: str
    freshness: FieldStatus
    service: Measurement
    docker: Measurement
    worker: Measurement
    host_ram: Measurement
    worker_ram: Measurement
    cpu: Measurement
    gpu_capability: Measurement
    gpu_utilization: Measurement
    gpu_vram: Measurement
    model: Measurement
    storage: tuple[StorageEntry, ...]
    budget_confidence: BudgetConfidence
    job_id: str | None
    attempt_id: str | None
    stage: Stage | None
    progress: Measurement
    elapsed_seconds: Measurement
    stage_eta_seconds: Measurement

    def __post_init__(self) -> None:
        validate_schema_version(self.schema_version)
        validate_utc_timestamp(self.sample_time)
        object.__setattr__(self, "freshness", _enum(FieldStatus, self.freshness))
        for name, kind in _RESOURCE_MEASURES.items():
            object.__setattr__(self, name, _measurement(getattr(self, name), kind))
        if type(self.storage) not in (tuple, list) or len(self.storage) > 8:
            raise DomainValidationError()
        object.__setattr__(self, "storage", tuple(StorageEntry.from_dict(x) if type(x) is dict else x for x in self.storage))
        if any(type(x) is not StorageEntry for x in self.storage):
            raise DomainValidationError()
        if len({x.volume_slot for x in self.storage}) != len(self.storage):
            raise DomainValidationError()
        object.__setattr__(self, "storage", tuple(self.storage))
        object.__setattr__(self, "budget_confidence", _enum(BudgetConfidence, self.budget_confidence))
        _optional(self.job_id, validate_uuid_string)
        _optional(self.attempt_id, validate_uuid_string)
        object.__setattr__(self, "stage", _optional(self.stage, lambda x: _enum(Stage, x)))
        _record(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> ResourceSnapshot:
        validate_json_payload(data)
        names = ("schema_version", "sample_time", "freshness", *_RESOURCE_MEASURES,
                 "storage", "budget_confidence", "job_id", "attempt_id", "stage")
        d = _keys(data, names)
        if type(d["storage"]) is not list:
            raise DomainValidationError()
        values = {name: _measurement(d[name], kind) for name, kind in _RESOURCE_MEASURES.items()}
        return cls(d["schema_version"], d["sample_time"], _enum(FieldStatus, d["freshness"]),
                   *(values[name] for name in tuple(_RESOURCE_MEASURES)[:10]),
                   tuple(StorageEntry.from_dict(x) for x in d["storage"]),
                   _enum(BudgetConfidence, d["budget_confidence"]), d["job_id"], d["attempt_id"],
                   _optional(d["stage"], lambda x: _enum(Stage, x)),
                   *(values[name] for name in tuple(_RESOURCE_MEASURES)[10:]))

    def to_dict(self) -> dict:
        result = {"schema_version": self.schema_version, "sample_time": self.sample_time,
                  "freshness": self.freshness.value, "storage": [x.to_dict() for x in self.storage],
                  "budget_confidence": self.budget_confidence.value, "job_id": self.job_id,
                  "attempt_id": self.attempt_id, "stage": self.stage.value if self.stage else None}
        result.update({name: getattr(self, name).to_dict() for name in _RESOURCE_MEASURES})
        return result

    def public_dict(self) -> dict:
        result = self.to_dict()
        result["storage"] = [x.public_dict() for x in self.storage]
        for name in _RESOURCE_MEASURES:
            result[name].pop("source")
        return result

    def telemetry_dict(self) -> dict:
        """Allow only closed identifiers, statuses, counts and numeric measurements."""
        def safe(name: str) -> dict:
            item = getattr(self, name)
            return {"value": item.value, "status": item.status.value,
                    "reason": item.reason.value if item.reason else None}
        names = ("host_ram", "worker_ram", "cpu", "gpu_utilization", "gpu_vram",
                 "progress", "elapsed_seconds", "stage_eta_seconds")
        return {"schema_version": 1, "sample_time": self.sample_time,
                "freshness": self.freshness.value, "budget_confidence": self.budget_confidence.value,
                "job_id": self.job_id, "attempt_id": self.attempt_id,
                "stage": self.stage.value if self.stage else None,
                "storage_count": len(self.storage), "measurements": {n: safe(n) for n in names}}


@dataclass(frozen=True, slots=True)
class SnapshotHistory(_Codec):
    """A value-style ring; ``append`` returns a new history retaining at most 600 samples."""

    samples: tuple[ResourceSnapshot, ...] = ()

    def __post_init__(self) -> None:
        if type(self.samples) not in (tuple, list) or len(self.samples) > 600 or any(type(x) is not ResourceSnapshot for x in self.samples):
            raise DomainValidationError()
        object.__setattr__(self, "samples", tuple(self.samples))
        _record(self.to_dict())

    def append(self, sample: ResourceSnapshot) -> SnapshotHistory:
        if type(sample) is not ResourceSnapshot:
            raise DomainValidationError()
        return SnapshotHistory((self.samples + (sample,))[-600:])

    @classmethod
    def from_dict(cls, data: object) -> SnapshotHistory:
        validate_json_payload(data)
        d = _keys(data, ("samples",))
        if type(d["samples"]) is not list or len(d["samples"]) > 600:
            raise DomainValidationError()
        return cls(tuple(ResourceSnapshot.from_dict(x) for x in d["samples"]))

    def to_dict(self) -> dict:
        return {"samples": [x.to_dict() for x in self.samples]}


@dataclass(frozen=True, slots=True)
class Provenance(_Codec):
    model_id: str | None = None
    model_revision: str | None = None
    image_digest: str | None = None
    device: str | None = None
    compute_type: str | None = None
    source_sha256: str | None = None
    attempt_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("model_id", "model_revision", "compute_type"):
            _optional(getattr(self, name), _string)
        if self.device is not None:
            object.__setattr__(self, "device", _enum(Device, self.device).value)
        if self.image_digest is not None:
            digest = _string(self.image_digest)
            if not digest.startswith("sha256:") or _SHA.fullmatch(digest[7:]) is None:
                raise DomainValidationError()
        _optional(self.source_sha256, _hash)
        _optional(self.attempt_id, validate_uuid_string)
        _record(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> Provenance:
        validate_json_payload(data)
        d = _keys(data, tuple(cls.__dataclass_fields__))
        return cls(**d)

    def to_dict(self) -> dict:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class Artifact(_Codec):
    artifact_id: str
    slot: ArtifactSlot
    format: ArtifactFormat
    bytes: int
    sha256: str
    available: bool
    attempt_id: str
    provenance: Provenance | None
    internal_path: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        validate_uuid_string(self.artifact_id)
        object.__setattr__(self, "slot", _enum(ArtifactSlot, self.slot))
        object.__setattr__(self, "format", _enum(ArtifactFormat, self.format))
        validate_nonnegative_int(self.bytes)
        _hash(self.sha256)
        _bool(self.available)
        validate_uuid_string(self.attempt_id)
        if type(self.provenance) is dict:
            object.__setattr__(self, "provenance", Provenance.from_dict(self.provenance))
        if self.provenance is not None and type(self.provenance) is not Provenance:
            raise DomainValidationError()
        _optional(self.internal_path, _string)
        _record(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> Artifact:
        validate_json_payload(data)
        d = _keys(data, tuple(cls.__dataclass_fields__))
        return cls(d["artifact_id"], _enum(ArtifactSlot, d["slot"]),
                   _enum(ArtifactFormat, d["format"]), d["bytes"], d["sha256"],
                   d["available"], d["attempt_id"],
                   None if d["provenance"] is None else Provenance.from_dict(d["provenance"]), d["internal_path"])

    def to_dict(self) -> dict:
        return {"artifact_id": self.artifact_id, "slot": self.slot.value,
                "format": self.format.value, "bytes": self.bytes, "sha256": self.sha256,
                "available": self.available, "attempt_id": self.attempt_id,
                "provenance": self.provenance.to_dict() if self.provenance else None,
                "internal_path": self.internal_path}

    def public_dict(self) -> dict:
        return {"artifact_id": self.artifact_id, "slot": self.slot.value,
                "format": self.format.value, "bytes": self.bytes, "available": self.available,
                "basename": ntpath.basename(self.internal_path) if self.internal_path else None}

    def telemetry_dict(self) -> dict:
        return {"artifact_id": self.artifact_id, "slot": self.slot.value,
                "format": self.format.value, "bytes": self.bytes, "available": self.available}


@dataclass(frozen=True, slots=True)
class StageRecord(_Codec):
    stage_id: str
    attempt_id: str
    stage: Stage
    status: StageOutcome
    started_at: str
    ended_at: str | None
    verified_input_hashes: tuple[tuple[str, str], ...]
    settings_hash: str
    output_slots: tuple[WorkerSlot, ...]

    def __post_init__(self) -> None:
        validate_uuid_string(self.stage_id)
        validate_uuid_string(self.attempt_id)
        object.__setattr__(self, "stage", _enum(Stage, self.stage))
        object.__setattr__(self, "status", _enum(StageOutcome, self.status))
        validate_utc_timestamp(self.started_at)
        _optional(self.ended_at, validate_utc_timestamp)
        if type(self.verified_input_hashes) not in (tuple, list):
            raise DomainValidationError()
        pairs = tuple(self.verified_input_hashes)
        if any(type(p) is not tuple or len(p) != 2 for p in pairs):
            raise DomainValidationError()
        for key, value in pairs:
            _string(key)
            _hash(value)
        if len({p[0] for p in pairs}) != len(pairs):
            raise DomainValidationError()
        object.__setattr__(self, "verified_input_hashes", pairs)
        _hash(self.settings_hash)
        if type(self.output_slots) not in (tuple, list):
            raise DomainValidationError()
        object.__setattr__(self, "output_slots", tuple(_enum(WorkerSlot, x) for x in self.output_slots))
        _record(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> StageRecord:
        validate_json_payload(data)
        d = _keys(data, tuple(cls.__dataclass_fields__))
        hashes = d["verified_input_hashes"]
        if type(hashes) is not dict or any(type(k) is not str for k in hashes):
            raise DomainValidationError()
        if type(d["output_slots"]) is not list:
            raise DomainValidationError()
        return cls(d["stage_id"], d["attempt_id"], _enum(Stage, d["stage"]),
                   _enum(StageOutcome, d["status"]), d["started_at"], d["ended_at"],
                   tuple(hashes.items()), d["settings_hash"],
                   tuple(_enum(WorkerSlot, x) for x in d["output_slots"]))

    def to_dict(self) -> dict:
        return {"stage_id": self.stage_id, "attempt_id": self.attempt_id,
                "stage": self.stage.value, "status": self.status.value,
                "started_at": self.started_at, "ended_at": self.ended_at,
                "verified_input_hashes": dict(self.verified_input_hashes),
                "settings_hash": self.settings_hash,
                "output_slots": [x.value for x in self.output_slots]}


@dataclass(frozen=True, slots=True)
class Attempt(_Codec):
    attempt_id: str
    job_id: str
    ordinal: int
    selected_model: str | None
    provenance: Provenance | None
    started_at: str
    ended_at: str | None
    state: JobState
    reason_code: ReasonCode
    stage_records: tuple[StageRecord, ...]
    automatic_fallback_count: int

    def __post_init__(self) -> None:
        validate_uuid_string(self.attempt_id)
        validate_uuid_string(self.job_id)
        if validate_nonnegative_int(self.ordinal) == 0:
            raise DomainValidationError()
        _optional(self.selected_model, _string)
        if type(self.provenance) is dict:
            object.__setattr__(self, "provenance", Provenance.from_dict(self.provenance))
        if self.provenance is not None and type(self.provenance) is not Provenance:
            raise DomainValidationError()
        validate_utc_timestamp(self.started_at)
        _optional(self.ended_at, validate_utc_timestamp)
        object.__setattr__(self, "state", _enum(JobState, self.state))
        object.__setattr__(self, "reason_code", _enum(ReasonCode, self.reason_code))
        if type(self.stage_records) not in (tuple, list):
            raise DomainValidationError()
        object.__setattr__(self, "stage_records", tuple(StageRecord.from_dict(x) if type(x) is dict else x for x in self.stage_records))
        if any(type(x) is not StageRecord or x.attempt_id != self.attempt_id for x in self.stage_records):
            raise DomainValidationError()
        if len({x.stage_id for x in self.stage_records}) != len(self.stage_records):
            raise DomainValidationError()
        object.__setattr__(self, "stage_records", tuple(self.stage_records))
        if type(self.automatic_fallback_count) is not int or self.automatic_fallback_count not in (0, 1):
            raise DomainValidationError()
        _record(self.to_dict())

    @classmethod
    def from_dict(cls, data: object) -> Attempt:
        validate_json_payload(data)
        d = _keys(data, tuple(cls.__dataclass_fields__))
        if type(d["stage_records"]) is not list:
            raise DomainValidationError()
        return cls(d["attempt_id"], d["job_id"], d["ordinal"], d["selected_model"],
                   None if d["provenance"] is None else Provenance.from_dict(d["provenance"]),
                   d["started_at"], d["ended_at"], _enum(JobState, d["state"]),
                   _enum(ReasonCode, d["reason_code"]),
                   tuple(StageRecord.from_dict(x) for x in d["stage_records"]), d["automatic_fallback_count"])

    def to_dict(self) -> dict:
        return {"attempt_id": self.attempt_id, "job_id": self.job_id, "ordinal": self.ordinal,
                "selected_model": self.selected_model,
                "provenance": self.provenance.to_dict() if self.provenance else None,
                "started_at": self.started_at, "ended_at": self.ended_at,
                "state": self.state.value, "reason_code": self.reason_code.value,
                "stage_records": [x.to_dict() for x in self.stage_records],
                "automatic_fallback_count": self.automatic_fallback_count}


@dataclass(frozen=True, slots=True)
class Job(_Codec):
    job_id: str
    request: JobRequest
    request_hash: str
    state: JobState
    reason_code: ReasonCode
    next_action: str | None
    attempt_id: str | None
    created_at: str
    updated_at: str
    artifacts: tuple[Artifact, ...]
    preflight: object | None

    def __post_init__(self) -> None:
        validate_uuid_string(self.job_id)
        if type(self.request) is not JobRequest:
            raise DomainValidationError()
        _hash(self.request_hash)
        if self.request_hash != self.request.request_hash:
            raise DomainValidationError()
        object.__setattr__(self, "state", _enum(JobState, self.state))
        object.__setattr__(self, "reason_code", _enum(ReasonCode, self.reason_code))
        _optional(self.next_action, _string)
        _optional(self.attempt_id, validate_uuid_string)
        validate_utc_timestamp(self.created_at)
        validate_utc_timestamp(self.updated_at)
        if type(self.artifacts) not in (tuple, list):
            raise DomainValidationError()
        object.__setattr__(self, "artifacts", tuple(Artifact.from_dict(x) if type(x) is dict else x for x in self.artifacts))
        if any(type(x) is not Artifact for x in self.artifacts):
            raise DomainValidationError()
        if len({x.artifact_id for x in self.artifacts}) != len(self.artifacts):
            raise DomainValidationError()
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        if self.preflight is not None:
            object.__setattr__(self, "preflight", _freeze(self.preflight))
        _record(self.to_dict())

    @classmethod
    def from_dict(cls, data: object, *, verified_models: frozenset[str] | set[str] | None = None,
                  verified_languages: frozenset[str] | set[str] | None = None) -> Job:
        validate_json_payload(data)
        d = _keys(data, tuple(cls.__dataclass_fields__))
        if type(d["artifacts"]) is not list:
            raise DomainValidationError()
        request = JobRequest.from_dict(d["request"], verified_models=verified_models,
                                       verified_languages=verified_languages)
        return cls(d["job_id"], request, d["request_hash"], _enum(JobState, d["state"]),
                   _enum(ReasonCode, d["reason_code"]), d["next_action"], d["attempt_id"],
                   d["created_at"], d["updated_at"],
                   tuple(Artifact.from_dict(x) for x in d["artifacts"]), d["preflight"])

    def to_dict(self) -> dict:
        return {"job_id": self.job_id, "request": self.request.to_dict(),
                "request_hash": self.request_hash, "state": self.state.value,
                "reason_code": self.reason_code.value, "next_action": self.next_action,
                "attempt_id": self.attempt_id, "created_at": self.created_at,
                "updated_at": self.updated_at, "artifacts": [x.to_dict() for x in self.artifacts],
                "preflight": _thaw(self.preflight)}

    def public_dict(self) -> dict:
        """Expose state and artifact IDs while withholding request input and root."""
        return {"job_id": self.job_id, "source_kind": self.request.source.kind,
                "output_mode": self.request.output_mode.value, "state": self.state.value,
                "reason_code": self.reason_code.value, "next_action": self.next_action,
                "attempt_id": self.attempt_id, "created_at": self.created_at,
                "updated_at": self.updated_at,
                "artifacts": [x.public_dict() for x in self.artifacts]}

    def telemetry_dict(self) -> dict:
        """Allow only closed domain values, UUIDs, times, and artifact counts."""
        return {"job_id": self.job_id, "state": self.state.value,
                "reason_code": self.reason_code.value, "attempt_id": self.attempt_id,
                "created_at": self.created_at, "updated_at": self.updated_at,
                "artifact_count": len(self.artifacts)}


__all__ = ("Measurement", "StorageEntry", "ResourceSnapshot", "SnapshotHistory",
           "Provenance", "Artifact", "StageRecord", "Attempt", "Job")
