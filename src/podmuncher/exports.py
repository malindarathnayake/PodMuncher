"""Pure transcript exporters, export caps and sidecar provenance.

The input is the ASR worker's ``transcript_segments`` record. It is treated as
untrusted bytes and fully re-validated before any export is rendered. Every
builder is a pure function from a validated record to UTF-8 bytes (LF line
endings, no BOM). Nothing here performs I/O, publishes, re-runs ASR, infers
speakers, or adds summary or paraphrase content: exports only re-arrange the
validated segment text, in order, with its original timestamps.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from .domains import ArtifactFormat, ArtifactSlot, DomainValidationError, ReasonCode

EXPORT_SCHEMA_VERSION = 1
SIDECAR_SCHEMA_VERSION = 1
SUPPORTED_RECORD_SCHEMAS = frozenset({1})
SEGMENT_SLOP_SECONDS = 1.0
MAX_JSON_DEPTH = 8
PARAGRAPH_GAP_SECONDS = 2.0
PARAGRAPH_MAX_CHARS = 600

_LANGUAGE = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,127}\Z")
_RECORD_REQUIRED = frozenset({
    "schema_version", "language", "duration_seconds", "no_speech_detected",
    "segments", "provenance",
})
_RECORD_OPTIONAL = frozenset({"language_probability", "measurements"})
_SEGMENT_KEYS = frozenset({"start_seconds", "end_seconds", "text"})
_PROVENANCE_KEYS = frozenset({
    "model_id", "model_snapshot_id", "device", "compute_type",
    "source_sha256", "attempt_id", "image_digest",
})
# Model revision is never read from the worker; only the host stamps it.

EXPORT_SLOTS: Mapping[ArtifactSlot, ArtifactFormat] = {
    ArtifactSlot.TRANSCRIPT_TXT: ArtifactFormat.TXT,
    ArtifactSlot.TRANSCRIPT_TIMESTAMPED_TXT: ArtifactFormat.TIMESTAMPED_TXT,
    ArtifactSlot.TRANSCRIPT_READABLE_TXT: ArtifactFormat.READABLE_TXT,
    ArtifactSlot.TRANSCRIPT_SRT: ArtifactFormat.SRT,
    ArtifactSlot.TRANSCRIPT_JSON: ArtifactFormat.JSON,
}

# Upper-bound byte rates per second of audio. Fast speech is ~4 words/s
# (~25 UTF-8 bytes/s Latin, ~24 bytes/s CJK); 48 bytes/s of text plus one
# segment per second of framing overhead leaves generous headroom.
_TEXT_BYTES_PER_SECOND = 48
_SEGMENTS_PER_SECOND = 1
_PER_SEGMENT_OVERHEAD = {
    ArtifactFormat.TXT: 1,
    ArtifactFormat.TIMESTAMPED_TXT: 40,
    ArtifactFormat.READABLE_TXT: 2,
    ArtifactFormat.SRT: 48,
    ArtifactFormat.JSON: 160,
}
_FIXED_OVERHEAD_BYTES = 4096


class ExportError(DomainValidationError):
    """Closed reason code; the message never contains transcript text or paths."""

    def __init__(self, reason_code: ReasonCode = ReasonCode.INVALID_INPUT):
        super().__init__(reason_code)

    def __str__(self) -> str:
        return "Export failed."


@dataclass(frozen=True, slots=True)
class Segment:
    start_seconds: float
    end_seconds: float
    text: str


@dataclass(frozen=True, slots=True)
class RecordProvenance:
    model_id: str | None
    model_snapshot_id: str | None
    device: str | None
    compute_type: str | None
    source_sha256: str | None
    attempt_id: str | None
    image_digest: str | None
    model_revision: str | None = None


@dataclass(frozen=True, slots=True)
class TranscriptRecord:
    schema_version: int
    language: str | None
    duration_seconds: float
    no_speech_detected: bool
    segments: tuple[Segment, ...]
    provenance: RecordProvenance


# --------------------------------------------------------------------------
# Parsing and validation


def _fail() -> ExportError:
    return ExportError(ReasonCode.INVALID_INPUT)


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _fail()
        result[key] = value
    return result


def _reject_constant(_: str) -> Any:
    raise _fail()


def _check_depth(value: object, depth: int = 1) -> None:
    if type(value) is dict:
        children: Iterable[object] = value.values()
    elif type(value) is list:
        children = value
    else:
        return
    if depth > MAX_JSON_DEPTH:
        raise _fail()
    for child in children:
        _check_depth(child, depth + 1)


def _number(value: object) -> float:
    if type(value) not in (int, float):  # bool is excluded by the exact type check
        raise _fail()
    number = float(value)
    if not math.isfinite(number):
        raise _fail()
    return number


def _text(value: object) -> str:
    if type(value) is not str or "\x00" in value or not value.strip():
        raise _fail()
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:  # lone surrogates from JSON escapes
        raise _fail() from None
    return value


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    if type(value) is not str or "\x00" in value:
        raise _fail()
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise _fail() from None
    return value


def _provenance(value: object) -> RecordProvenance:
    if type(value) is not dict or set(value) != _PROVENANCE_KEYS:
        raise _fail()
    return RecordProvenance(**{key: _optional_str(value[key]) for key in sorted(_PROVENANCE_KEYS)})


def _segments(value: object, duration: float) -> tuple[Segment, ...]:
    if type(value) is not list:
        raise _fail()
    segments: list[Segment] = []
    previous_start = 0.0
    for item in value:
        if type(item) is not dict or set(item) != _SEGMENT_KEYS:
            raise _fail()
        start = _number(item["start_seconds"])
        end = _number(item["end_seconds"])
        if not 0.0 <= start <= end <= duration + SEGMENT_SLOP_SECONDS or start < previous_start:
            raise _fail()
        previous_start = start
        segments.append(Segment(start, end, _text(item["text"])))
    return tuple(segments)


def parse_record(raw: bytes, *, max_bytes: int) -> TranscriptRecord:
    """Parse and fully re-validate an untrusted worker transcript record."""
    if type(max_bytes) is not int or max_bytes <= 0:
        raise _fail()
    if not isinstance(raw, (bytes, bytearray)) or len(raw) > max_bytes:
        raise _fail()
    try:
        data = json.loads(bytes(raw).decode("utf-8"), object_pairs_hook=_unique_pairs,
                          parse_constant=_reject_constant)
    except ExportError:
        raise
    except (RecursionError, UnicodeDecodeError, ValueError):
        raise _fail() from None
    _check_depth(data)
    if type(data) is not dict:
        raise _fail()
    keys = set(data)
    if not _RECORD_REQUIRED <= keys or not keys <= _RECORD_REQUIRED | _RECORD_OPTIONAL:
        raise _fail()
    schema = data["schema_version"]
    if type(schema) is not int or schema not in SUPPORTED_RECORD_SCHEMAS:
        raise _fail()
    language = data["language"]
    if language is not None and (type(language) is not str or _LANGUAGE.fullmatch(language) is None):
        raise _fail()
    duration = _number(data["duration_seconds"])
    if duration < 0.0:
        raise _fail()
    if "language_probability" in data and data["language_probability"] is not None:
        _number(data["language_probability"])
    if "measurements" in data and type(data["measurements"]) is not dict:
        raise _fail()
    segments = _segments(data["segments"], duration)
    no_speech = data["no_speech_detected"]
    if type(no_speech) is not bool or no_speech != (len(segments) == 0):
        raise _fail()
    return TranscriptRecord(
        schema_version=schema,
        language=language,
        duration_seconds=duration,
        no_speech_detected=no_speech,
        segments=segments,
        provenance=_provenance(data["provenance"]),
    )


def with_host_provenance(record: TranscriptRecord, *, image_digest: str,
                         model_revision: str | None) -> TranscriptRecord:
    """Return a new record stamped with host-verified image digest and model revision."""
    if type(record) is not TranscriptRecord:
        raise _fail()
    if type(image_digest) is not str or not image_digest.strip():
        raise _fail()
    digest = _optional_str(image_digest)
    revision = _optional_str(model_revision)
    if revision is not None and not revision.strip():
        raise _fail()
    return replace(record, provenance=replace(record.provenance, image_digest=digest,
                                              model_revision=revision))


# --------------------------------------------------------------------------
# Builders


def _lines(text: str) -> list[str]:
    """Split on every line boundary, dropping blank lines; words are untouched."""
    return [line.strip() for line in text.splitlines() if line.strip()]


def _one_line(text: str) -> str:
    return " ".join(_lines(text))


def _milliseconds(seconds: float) -> int:
    return math.floor(seconds * 1000.0 + 0.5)


def _span(segment: Segment) -> tuple[int, int]:
    start = _milliseconds(segment.start_seconds)
    return start, max(start, _milliseconds(segment.end_seconds))


def _clock(ms: int, separator: str) -> str:
    hours, rest = divmod(ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}{separator}{millis:03d}"


def _encode(lines: list[str]) -> bytes:
    return ("\n".join(lines) + "\n").encode("utf-8") if lines else b""


def build_txt(record: TranscriptRecord) -> bytes:
    return _encode([_one_line(segment.text) for segment in record.segments])


def build_timestamped_txt(record: TranscriptRecord) -> bytes:
    lines = []
    for segment in record.segments:
        start, end = _span(segment)
        lines.append(f"[{_clock(start, '.')} --> {_clock(end, '.')}] {_one_line(segment.text)}")
    return _encode(lines)


def paragraphs(record: TranscriptRecord) -> list[str]:
    """Group consecutive segments into paragraphs at segment boundaries only."""
    result: list[str] = []
    current: list[str] = []
    length = 0
    previous_end: float | None = None
    for segment in record.segments:
        text = _one_line(segment.text)
        gap_break = previous_end is not None and segment.start_seconds - previous_end >= PARAGRAPH_GAP_SECONDS
        size_break = bool(current) and length + 1 + len(text) > PARAGRAPH_MAX_CHARS
        if current and (gap_break or size_break):
            result.append(" ".join(current))
            current, length = [], 0
        length += len(text) + (1 if current else 0)
        current.append(text)
        previous_end = segment.end_seconds if previous_end is None else max(previous_end, segment.end_seconds)
    if current:
        result.append(" ".join(current))
    return result


def build_readable_txt(record: TranscriptRecord) -> bytes:
    groups = paragraphs(record)
    return ("\n\n".join(groups) + "\n").encode("utf-8") if groups else b""


def build_srt(record: TranscriptRecord) -> bytes:
    cues = []
    for number, segment in enumerate(record.segments, start=1):
        start, end = _span(segment)
        body = "\n".join(_lines(segment.text))
        cues.append(f"{number}\n{_clock(start, ',')} --> {_clock(end, ',')}\n{body}\n")
    return "\n".join(cues).encode("utf-8")


def _provenance_document(provenance: RecordProvenance) -> dict[str, str | None]:
    return {
        "model_id": provenance.model_id,
        "model_revision": provenance.model_revision,
        "image_digest": provenance.image_digest,
        "device": provenance.device,
        "compute_type": provenance.compute_type,
        "source_sha256": provenance.source_sha256,
        "attempt_id": provenance.attempt_id,
    }


def _dump(document: object) -> bytes:
    try:
        text = json.dumps(document, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2)
    except ValueError:
        raise _fail() from None
    return (text + "\n").encode("utf-8")


def build_json(record: TranscriptRecord) -> bytes:
    return _dump({
        "schema_version": EXPORT_SCHEMA_VERSION,
        "language": record.language,
        "duration_seconds": record.duration_seconds,
        "no_speech_detected": record.no_speech_detected,
        "segments": [
            {"start_seconds": s.start_seconds, "end_seconds": s.end_seconds, "text": s.text}
            for s in record.segments
        ],
        "provenance": _provenance_document(record.provenance),
    })


_BUILDERS: Mapping[ArtifactFormat, Callable[[TranscriptRecord], bytes]] = {
    ArtifactFormat.TXT: build_txt,
    ArtifactFormat.TIMESTAMPED_TXT: build_timestamped_txt,
    ArtifactFormat.READABLE_TXT: build_readable_txt,
    ArtifactFormat.SRT: build_srt,
    ArtifactFormat.JSON: build_json,
}


def export_format(slot: object) -> ArtifactFormat:
    """Map a slot to its export format; non-transcript slots are not exportable."""
    parsed = ArtifactSlot.parse(slot)
    try:
        return EXPORT_SLOTS[parsed]
    except KeyError:
        raise _fail() from None


def build(slot: object, record: TranscriptRecord) -> bytes:
    if type(record) is not TranscriptRecord:
        raise _fail()
    return _BUILDERS[export_format(slot)](record)


# --------------------------------------------------------------------------
# Caps


def _slots(slots: Iterable[object]) -> list[ArtifactSlot]:
    if isinstance(slots, (str, bytes)):
        raise _fail()
    try:
        parsed = {ArtifactSlot.parse(slot) for slot in slots}
    except TypeError:
        raise _fail() from None
    if not parsed:
        raise _fail()
    for slot in parsed:
        export_format(slot)
    return sorted(parsed, key=lambda slot: slot.value)


def estimate_export_bytes(duration_seconds: float, slots: Iterable[object]) -> int:
    """Deterministic upper estimate of total export bytes, from duration alone."""
    duration = _number(duration_seconds)
    if duration < 0.0:
        raise _fail()
    seconds = math.ceil(duration)
    total = 0
    for slot in _slots(slots):
        rate = _TEXT_BYTES_PER_SECOND + _SEGMENTS_PER_SECOND * _PER_SEGMENT_OVERHEAD[EXPORT_SLOTS[slot]]
        total += _FIXED_OVERHEAD_BYTES + seconds * rate
    return total


def export_all(record: TranscriptRecord, slots: Iterable[object], *,
               cap_bytes: int) -> dict[ArtifactSlot, bytes]:
    """Render every requested slot, or none: the cap covers each output and the total."""
    if type(record) is not TranscriptRecord or type(cap_bytes) is not int or cap_bytes <= 0:
        raise _fail()
    ordered = _slots(slots)
    rendered: dict[ArtifactSlot, bytes] = {}
    total = 0
    for slot in ordered:
        data = _BUILDERS[EXPORT_SLOTS[slot]](record)
        total += len(data)
        if len(data) > cap_bytes or total > cap_bytes:
            raise ExportError(ReasonCode.EXPORT_LIMIT_REACHED)
        rendered[slot] = data
    return rendered


# --------------------------------------------------------------------------
# Sidecar provenance


def provenance_sidecar(record: TranscriptRecord, outputs: Mapping[ArtifactSlot, bytes]) -> bytes:
    """Schema-versioned provenance plus per-slot format, size and sha256; no text, no paths."""
    if type(record) is not TranscriptRecord or not isinstance(outputs, Mapping) or not outputs:
        raise _fail()
    entries: dict[str, dict[str, object]] = {}
    for slot, data in outputs.items():
        fmt = export_format(slot)
        if type(data) is not bytes:
            raise _fail()
        entries[ArtifactSlot.parse(slot).value] = {
            "format": fmt.value,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    return _dump({
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "provenance": _provenance_document(record.provenance),
        "outputs": entries,
    })
