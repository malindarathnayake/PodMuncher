"""Closed contract domains and reusable, side-effect-free input validation.

All domain values are compiled into this module; importing it never reads Docs.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import IntEnum, StrEnum
import math
import re
from typing import Any, Mapping
from uuid import UUID


class DomainValidationError(ValueError):
    """A safe validation failure with a closed, machine-readable reason code."""

    def __init__(self, reason_code: ReasonCode | None = None) -> None:
        self.reason_code = reason_code if type(reason_code) is ReasonCode else ReasonCode.INVALID_INPUT
        super().__init__("Invalid input.")


class _StrictStrEnum(StrEnum):
    @classmethod
    def parse(cls, value: object) -> _StrictStrEnum:
        if isinstance(value, cls):
            return value
        if type(value) is not str:
            raise DomainValidationError(ReasonCode.INVALID_INPUT)
        try:
            return cls(value)
        except ValueError:
            raise DomainValidationError(ReasonCode.INVALID_INPUT) from None


class _StrictIntEnum(IntEnum):
    @classmethod
    def parse(cls, value: object) -> _StrictIntEnum:
        if isinstance(value, cls):
            return value
        if type(value) is not int:  # bool is an int subclass but never a slot.
            raise DomainValidationError(ReasonCode.INVALID_INPUT)
        try:
            return cls(value)
        except ValueError:
            raise DomainValidationError(ReasonCode.INVALID_INPUT) from None


# The declarations below are generated once from the versioned registry.
class JobState(_StrictStrEnum):
    QUEUED = 'queued'
    CHECKING = 'checking'
    BLOCKED = 'blocked'
    PREPARING = 'preparing'
    DOWNLOADING = 'downloading'
    PROBING = 'probing'
    TRANSCRIBING = 'transcribing'
    EXPORTING = 'exporting'
    COMPLETED = 'completed'
    PARTIAL = 'partial'
    FAILED = 'failed'
    CANCELLED = 'cancelled'
    INTERRUPTED = 'interrupted'


class Stage(_StrictStrEnum):
    CHECKING = 'checking'
    PREPARING = 'preparing'
    DOWNLOADING = 'downloading'
    PROBING = 'probing'
    TRANSCRIBING = 'transcribing'
    EXPORTING = 'exporting'


class Source(_StrictStrEnum):
    LOCAL = 'local'
    DIRECT = 'direct'
    YOUTUBE = 'youtube'


class Device(_StrictStrEnum):
    AUTO = 'auto'
    CPU = 'cpu'
    CUDA = 'cuda'


class Backend(_StrictStrEnum):
    CPU = 'cpu'
    CUDA = 'cuda'


class Profile(_StrictStrEnum):
    BALANCED = 'balanced'
    QUALITY = 'quality'


class FieldStatus(_StrictStrEnum):
    OK = 'ok'
    UNKNOWN = 'unknown'
    STALE = 'stale'
    ERROR = 'error'
    NOT_APPLICABLE = 'not_applicable'


class WorkerEvent(_StrictStrEnum):
    STARTED = 'started'
    PROGRESS = 'progress'
    ARTIFACT = 'artifact'
    WARNING = 'warning'
    ERROR = 'error'
    COMPLETED = 'completed'


class OutputMode(_StrictStrEnum):
    TRANSCRIPT = 'transcript'
    VIDEO_TRANSCRIPT = 'video_transcript'
    VIDEO = 'video'


class ArtifactSlot(_StrictStrEnum):
    ORIGINAL_VIDEO = 'original_video'
    TRANSCRIPT_TXT = 'transcript_txt'
    TRANSCRIPT_TIMESTAMPED_TXT = 'transcript_timestamped_txt'
    TRANSCRIPT_READABLE_TXT = 'transcript_readable_txt'
    TRANSCRIPT_SRT = 'transcript_srt'
    TRANSCRIPT_JSON = 'transcript_json'


class ArtifactFormat(_StrictStrEnum):
    VIDEO = 'video'
    TXT = 'txt'
    TIMESTAMPED_TXT = 'timestamped_txt'
    READABLE_TXT = 'readable_txt'
    SRT = 'srt'
    JSON = 'json'


class Theme(_StrictStrEnum):
    AUTO = 'auto'
    LIGHT = 'light'
    DARK = 'dark'


class Capability(_StrictStrEnum):
    MEDIA = 'media'
    ASR_CPU = 'asr_cpu'
    ASR_CUDA = 'asr_cuda'


class ModelTier(_StrictStrEnum):
    TINY = 'tiny'
    BASE = 'base'
    SMALL = 'small'
    MEDIUM = 'medium'
    LARGE = 'large'


class PreflightResult(_StrictStrEnum):
    PASS = 'pass'
    BLOCKED = 'blocked'


class PreflightReason(_StrictStrEnum):
    DISK = 'disk'
    RAM = 'ram'
    GPU = 'gpu'
    DOCKER = 'docker'
    INPUT = 'input'
    PERMISSION = 'permission'
    ASSET = 'asset'
    UNKNOWN = 'unknown'


class StageOutcome(_StrictStrEnum):
    OK = 'ok'
    ERROR = 'error'
    CANCELLED = 'cancelled'


class Resource(_StrictStrEnum):
    HOST_RAM = 'host_ram'
    WORKER_RAM = 'worker_ram'
    GPU_VRAM = 'gpu_vram'


class Context(_StrictStrEnum):
    HOST = 'host'
    DOCKER_VM = 'docker_vm'


class DeviceSlot(_StrictIntEnum):
    VALUE_0 = 0


class VolumeSlot(_StrictIntEnum):
    VALUE_0 = 0
    VALUE_1 = 1
    VALUE_2 = 2
    VALUE_3 = 3
    VALUE_4 = 4
    VALUE_5 = 5
    VALUE_6 = 6
    VALUE_7 = 7


class ServiceState(_StrictStrEnum):
    RUNNING = 'running'
    STOPPED = 'stopped'
    STOPPING = 'stopping'
    RECOVERING = 'recovering'


class BudgetConfidence(_StrictStrEnum):
    MEASURED = 'measured'
    CONSERVATIVE = 'conservative'
    UNKNOWN = 'unknown'


class VideoQuality(_StrictStrEnum):
    VALUE_1080P = '1080p'
    SOURCE_BEST = 'source_best'


class Component(_StrictStrEnum):
    API = 'api'
    SERVICE = 'service'
    STORAGE = 'storage'
    RUNTIME = 'runtime'
    ACQUISITION = 'acquisition'
    MODEL = 'model'
    WORKER = 'worker'
    ARTIFACTS = 'artifacts'
    TELEMETRY = 'telemetry'
    CLI = 'cli'
    TUI = 'tui'
    CONFIG = 'config'


class ReasonCode(_StrictStrEnum):
    OK = 'ok'
    INVALID_INPUT = 'invalid_input'
    INVALID_CONFIG = 'invalid_config'
    UNSUPPORTED_SCHEMA = 'unsupported_schema'
    UNAUTHORIZED = 'unauthorized'
    INVALID_HOST = 'invalid_host'
    BROWSER_ORIGIN_REJECTED = 'browser_origin_rejected'
    REQUEST_TOO_LARGE = 'request_too_large'
    REQUEST_TIMEOUT = 'request_timeout'
    RATE_LIMITED = 'rate_limited'
    UNSUPPORTED_TRANSFER_ENCODING = 'unsupported_transfer_encoding'
    NESTING_LIMIT = 'nesting_limit'
    QUEUE_FULL = 'queue_full'
    CONFLICT = 'conflict'
    ALREADY_TERMINAL = 'already_terminal'
    SERVICE_STOPPED = 'service_stopped'
    SERVICE_LOST = 'service_lost'
    SERVICE_UNAVAILABLE = 'service_unavailable'
    STORAGE_INSUFFICIENT = 'storage_insufficient'
    STORAGE_UNKNOWN = 'storage_unknown'
    STORAGE_STALE = 'storage_stale'
    STORAGE_GUARD_TRIPPED = 'storage_guard_tripped'
    QUOTA_EXCEEDED = 'quota_exceeded'
    STATE_WRITE_FAILED = 'state_write_failed'
    AUDIT_LOSS = 'audit_loss'
    DOCKER_UNAVAILABLE = 'docker_unavailable'
    DOCKER_CONTEXT_UNSUPPORTED = 'docker_context_unsupported'
    BACKING_UNRESOLVED = 'backing_unresolved'
    GPU_UNAVAILABLE = 'gpu_unavailable'
    RAM_INSUFFICIENT = 'ram_insufficient'
    ENVELOPE_UNVALIDATED = 'envelope_unvalidated'
    ASSET_VERIFICATION_FAILED = 'asset_verification_failed'
    ASSET_SOURCE_REJECTED = 'asset_source_rejected'
    WORKER_PROTOCOL_MISMATCH = 'worker_protocol_mismatch'
    WORKER_EXIT_UNEXPECTED = 'worker_exit_unexpected'
    MEDIA_PROBE_FAILED = 'media_probe_failed'
    MEDIA_LIMIT_REACHED = 'media_limit_reached'
    INPUT_REJECTED = 'input_rejected'
    DISPLAY_TEXT_SANITIZED = 'display_text_sanitized'
    EVENT_LIMIT_REACHED = 'event_limit_reached'
    DOWNLOAD_TIMEOUT = 'download_timeout'
    UPSTREAM_UNAVAILABLE = 'upstream_unavailable'
    CONTENT_DENIED = 'content_denied'
    SOURCE_CHANGED = 'source_changed'
    NO_AUDIO = 'no_audio'
    NO_VIDEO = 'no_video'
    DURATION_UNKNOWN = 'duration_unknown'
    DURATION_EXCEEDED = 'duration_exceeded'
    MODEL_OOM = 'model_oom'
    MODEL_LOAD_FAILED = 'model_load_failed'
    PUBLICATION_FAILED = 'publication_failed'
    DESTINATION_CONFLICT = 'destination_conflict'
    ARTIFACT_REJECTED = 'artifact_rejected'
    CANCELLATION_REQUESTED = 'cancellation_requested'
    CANCELLED = 'cancelled'
    WRITER_UNRESOLVED = 'writer_unresolved'
    CACHE_IN_USE = 'cache_in_use'
    CACHE_UNKNOWN = 'cache_unknown'
    EXPORT_LIMIT_REACHED = 'export_limit_reached'
    TELEMETRY_DROPPED = 'telemetry_dropped'


class WorkerSlot(_StrictStrEnum):
    DOWNLOADED_MEDIA = 'downloaded_media'
    DOWNLOADED_AUDIO = 'downloaded_audio'
    DOWNLOADED_VIDEO = 'downloaded_video'
    MERGED_VIDEO = 'merged_video'
    TRANSCRIPT_SEGMENTS = 'transcript_segments'
    PROBE_METADATA = 'probe_metadata'


_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_UTC_TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
    r"(?:\.\d+)?(?:Z|\+00:00)$"
)


def validate_object_keys(
    value: object,
    *,
    allowed: set[str] | frozenset[str],
    required: set[str] | frozenset[str] = frozenset(),
) -> Mapping[str, Any]:
    """Require an object with only allowed keys and all required keys."""
    if not isinstance(value, dict) or not all(type(key) is str for key in value):
        raise DomainValidationError(ReasonCode.INVALID_INPUT)
    if not value.keys() <= allowed or not required <= value.keys():
        raise DomainValidationError(ReasonCode.INVALID_INPUT)
    return value


def validate_uuid_string(value: object) -> str:
    """Accept only a conventional hyphenated UUID string."""
    if type(value) is not str or _UUID_RE.fullmatch(value) is None:
        raise DomainValidationError(ReasonCode.INVALID_INPUT)
    try:
        UUID(value)
    except ValueError:
        raise DomainValidationError(ReasonCode.INVALID_INPUT) from None
    return value


def validate_nonnegative_int(value: object) -> int:
    if type(value) is not int or value < 0:
        raise DomainValidationError(ReasonCode.INVALID_INPUT)
    return value


def validate_finite_number(value: object) -> int | float:
    if type(value) is int:
        return value
    if type(value) is not float or not math.isfinite(value):
        raise DomainValidationError(ReasonCode.INVALID_INPUT)
    return value


def validate_percentage(value: object) -> int | float:
    number = validate_finite_number(value)
    if not 0 <= number <= 100:
        raise DomainValidationError(ReasonCode.INVALID_INPUT)
    return number


def validate_utf8_string(value: object, *, max_bytes: int = 8192) -> str:
    if type(max_bytes) is not int or max_bytes < 0:
        raise ValueError("Invalid byte limit.")
    if type(value) is not str:
        raise DomainValidationError(ReasonCode.INVALID_INPUT)
    try:
        encoded = value.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        raise DomainValidationError(ReasonCode.INVALID_INPUT) from None
    if len(encoded) > max_bytes:
        raise DomainValidationError(ReasonCode.INVALID_INPUT)
    return value


def validate_schema_version(value: object) -> int:
    if type(value) is not int or value != 1:
        raise DomainValidationError(ReasonCode.UNSUPPORTED_SCHEMA)
    return value


def validate_utc_timestamp(value: object) -> str:
    """Require an RFC 3339 timestamp explicitly denoting UTC."""
    if type(value) is not str or _UTC_TIMESTAMP_RE.fullmatch(value) is None:
        raise DomainValidationError(ReasonCode.INVALID_INPUT)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise DomainValidationError(ReasonCode.INVALID_INPUT) from None
    if parsed.utcoffset() != timedelta(0):
        raise DomainValidationError(ReasonCode.INVALID_INPUT)
    return value


def validate_json_payload(
    value: object,
    *,
    max_depth: int = 8,
    max_string_bytes: int = 8192,
) -> object:
    """Validate a JSON data tree, including all nested keys and values.

    Root arrays and objects have depth one. Repeated references without a cycle
    are valid; an object visited again on its own active path is rejected.
    """
    if type(max_depth) is not int or max_depth < 0:
        raise ValueError("Invalid depth limit.")
    if type(max_string_bytes) is not int or max_string_bytes < 0:
        raise ValueError("Invalid byte limit.")
    active: set[int] = set()

    def visit(item: object, depth: int) -> None:
        if item is None or type(item) is bool:
            return
        if type(item) is str:
            validate_utf8_string(item, max_bytes=max_string_bytes)
            return
        if type(item) in (int, float):
            validate_finite_number(item)
            return
        if type(item) not in (dict, list):
            raise DomainValidationError(ReasonCode.INVALID_INPUT)
        if depth >= max_depth:
            raise DomainValidationError(ReasonCode.NESTING_LIMIT)
        identity = id(item)
        if identity in active:
            raise DomainValidationError(ReasonCode.INVALID_INPUT)
        active.add(identity)
        try:
            if type(item) is dict:
                for key, child in item.items():
                    validate_utf8_string(key, max_bytes=max_string_bytes)
                    visit(child, depth + 1)
            else:
                for child in item:
                    visit(child, depth + 1)
        finally:
            active.remove(identity)

    visit(value, 0)
    return value
