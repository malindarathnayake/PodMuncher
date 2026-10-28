"""Pure transcript exporter, cap and provenance sidecar tests."""

import hashlib
import json
from pathlib import Path

import pytest

from podmuncher import exports
from podmuncher.domains import (
    ArtifactFormat,
    ArtifactSlot,
    DomainValidationError,
    ReasonCode,
)
from podmuncher.exports import ExportError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CAP = 268435456
TRANSCRIPT_SLOTS = frozenset(set(ArtifactSlot) - {ArtifactSlot.ORIGINAL_VIDEO})


def make_segments(*items):
    """Build worker-shaped segments from (start, end, text) tuples."""
    return [{"start_seconds": s, "end_seconds": e, "text": t} for s, e, t in items]


def make_raw(segments=(), *, duration=60.0, language="en", no_speech=None, **overrides):
    record = {
        "schema_version": 1,
        "language": language,
        "language_probability": 0.98,
        "duration_seconds": duration,
        "no_speech_detected": (len(segments) == 0) if no_speech is None else no_speech,
        "segments": list(segments),
        "provenance": {
            "model_id": "small", "model_snapshot_id": "abc123", "device": "cpu",
            "compute_type": "int8", "source_sha256": "0" * 64,
            "attempt_id": "11111111-1111-4111-8111-111111111111", "image_digest": None,
        },
        "measurements": {"load_seconds": 1.0, "peak_rss_bytes": None},
    }
    record.update(overrides)
    return json.dumps(record, ensure_ascii=False).encode("utf-8")


def make_record(segments=(), **kwargs):
    return exports.parse_record(make_raw(segments, **kwargs), max_bytes=CAP)


def assert_srt(data, expected_cues):
    """Assert SRT bytes are exactly the given (start, end, text) cues numbered from 1."""
    assert b"\r" not in data and not data.startswith(b"\xef\xbb\xbf")
    if not expected_cues:
        assert data == b""
        return
    blocks = data.decode("utf-8").split("\n\n")
    assert len(blocks) == len(expected_cues)
    for number, (block, (start, end, text)) in enumerate(zip(blocks, expected_cues), start=1):
        lines = block.rstrip("\n").split("\n")
        assert lines[0] == str(number)
        assert lines[1] == f"{start} --> {end}"
        assert "\n".join(lines[2:]) == text
        assert all(line.strip() for line in lines[2:])
    assert data.endswith(b"\n") and not data.endswith(b"\n\n")


def reason(excinfo):
    return excinfo.value.reason_code


# --- empty / silent -----------------------------------------------------------


def test_silent_audio_exports_are_valid_and_empty():
    record = make_record([], duration=12.5)
    assert record.no_speech_detected is True and record.segments == ()
    outputs = exports.export_all(record, TRANSCRIPT_SLOTS, cap_bytes=CAP)
    assert set(outputs) == TRANSCRIPT_SLOTS
    for slot in TRANSCRIPT_SLOTS - {ArtifactSlot.TRANSCRIPT_JSON}:
        assert outputs[slot] == b""
    document = json.loads(outputs[ArtifactSlot.TRANSCRIPT_JSON])
    assert document["segments"] == [] and document["no_speech_detected"] is True
    assert document["duration_seconds"] == 12.5
    assert_srt(outputs[ArtifactSlot.TRANSCRIPT_SRT], [])


@pytest.mark.parametrize("segments, flag", [([], False), (make_segments((0, 1, "hi")), True)])
def test_mismatched_no_speech_flag_rejected(segments, flag):
    with pytest.raises(ExportError) as excinfo:
        make_record(segments, no_speech=flag)
    assert reason(excinfo) is ReasonCode.INVALID_INPUT


# --- Unicode ------------------------------------------------------------------


def test_unicode_preserved_byte_exact():
    texts = ["日本語のテキスト", "مرحبا بالعالم", "emoji 🎙️👩🏽‍💻 ok", "Ünïcödé café"]
    segments = make_segments(*[(i * 2.0, i * 2.0 + 1.0, t) for i, t in enumerate(texts)])
    outputs = exports.export_all(make_record(segments), TRANSCRIPT_SLOTS, cap_bytes=CAP)
    assert outputs[ArtifactSlot.TRANSCRIPT_TXT] == ("\n".join(texts) + "\n").encode("utf-8")
    for data in outputs.values():
        assert not data.startswith(b"\xef\xbb\xbf") and b"\r" not in data
        decoded = data.decode("utf-8")
        for text in texts:
            assert text in decoded
    document = json.loads(outputs[ArtifactSlot.TRANSCRIPT_JSON])
    assert [s["text"] for s in document["segments"]] == texts


# --- timestamps -----------------------------------------------------------------


def test_timestamp_boundaries_and_rounding():
    segments = make_segments(
        (0, 0, "zero"),
        (0.0004, 0.0004, "sub-millisecond"),
        (1.2344, 1.2346, "rounding"),
        (3661.5, 3725.0, "over an hour"),
    )
    record = make_record(segments, duration=3725.0)
    assert_srt(exports.build_srt(record), [
        ("00:00:00,000", "00:00:00,000", "zero"),
        ("00:00:00,000", "00:00:00,000", "sub-millisecond"),
        ("00:00:01,234", "00:00:01,235", "rounding"),
        ("01:01:01,500", "01:02:05,000", "over an hour"),
    ])
    timestamped = exports.build_timestamped_txt(record).decode("utf-8").splitlines()
    assert timestamped[0] == "[00:00:00.000 --> 00:00:00.000] zero"
    assert timestamped[3] == "[01:01:01.500 --> 01:02:05.000] over an hour"


def test_rounding_never_makes_end_before_start():
    for start in [0.0005, 0.0015, 1.9995, 59.9995, 3599.9995, 0.1 + 0.2]:
        record = make_record(make_segments((start, start, "x")), duration=4000.0)
        assert exports._span(record.segments[0])[1] >= exports._span(record.segments[0])[0]
        line = exports.build_srt(record).decode().splitlines()[1]
        begin, end = line.split(" --> ")
        assert end >= begin


def test_overlap_kept_in_order_without_retiming():
    segments = make_segments((0, 5, "first"), (3, 4, "second"), (3, 8, "third"))
    record = make_record(segments)
    assert [s.text for s in record.segments] == ["first", "second", "third"]
    assert_srt(exports.build_srt(record), [
        ("00:00:00,000", "00:00:05,000", "first"),
        ("00:00:03,000", "00:00:04,000", "second"),
        ("00:00:03,000", "00:00:08,000", "third"),
    ])
    document = json.loads(exports.build_json(record))
    assert [(s["start_seconds"], s["end_seconds"]) for s in document["segments"]] == [(0, 5), (3, 4), (3, 8)]


@pytest.mark.parametrize("segments, duration", [
    ('[{"start_seconds": NaN, "end_seconds": 1, "text": "a"}]', 10),
    ('[{"start_seconds": 0, "end_seconds": Infinity, "text": "a"}]', 10),
    ('[{"start_seconds": -1, "end_seconds": 1, "text": "a"}]', 10),
    ('[{"start_seconds": 2, "end_seconds": 1, "text": "a"}]', 10),
    ('[{"start_seconds": 2, "end_seconds": 3, "text": "a"}, {"start_seconds": 1, "end_seconds": 3, "text": "b"}]', 10),
    ('[{"start_seconds": 0, "end_seconds": 12, "text": "a"}]', 10),
    ('[{"start_seconds": 0, "end_seconds": 1, "text": "   "}]', 10),
    ('[{"start_seconds": 0, "end_seconds": 1, "text": "a\\u0000b"}]', 10),
    ('[{"start_seconds": 0, "end_seconds": 1, "text": "\\ud800"}]', 10),
    ('[{"start_seconds": true, "end_seconds": 1, "text": "a"}]', 10),
    ('[{"start_seconds": 0, "end_seconds": 1, "text": "a", "speaker": "A"}]', 10),
    ('[]', -1),
    ('[]', "NaN"),
])
def test_invalid_segments_rejected(segments, duration):
    provenance = ('{"model_id": null, "model_snapshot_id": null, "device": null, "compute_type": null, '
                  '"source_sha256": null, "attempt_id": null, "image_digest": null}')
    raw = (f'{{"schema_version": 1, "language": "en", "duration_seconds": {duration}, '
           f'"no_speech_detected": false, "segments": {segments}, "provenance": {provenance}}}')
    with pytest.raises(ExportError) as excinfo:
        exports.parse_record(raw.encode(), max_bytes=CAP)
    assert reason(excinfo) is ReasonCode.INVALID_INPUT


@pytest.mark.parametrize("raw", [
    b"",
    b"\xef\xbb\xbf{}",
    b"\xff\xfe",
    b"[]",
    b'{"schema_version": 1, "schema_version": 1}',
    b"[" * 50 + b"]" * 50,
])
def test_malformed_json_rejected(raw):
    with pytest.raises(ExportError):
        exports.parse_record(raw, max_bytes=CAP)


def test_record_shape_rejections():
    with pytest.raises(ExportError):
        exports.parse_record(make_raw([]), max_bytes=10)
    with pytest.raises(ExportError):
        exports.parse_record(make_raw([], schema_version=2), max_bytes=CAP)
    with pytest.raises(ExportError):
        exports.parse_record(make_raw([], schema_version=True), max_bytes=CAP)
    with pytest.raises(ExportError):
        exports.parse_record(make_raw([], language=""), max_bytes=CAP)
    with pytest.raises(ExportError):
        exports.parse_record(make_raw([], speakers=["A"]), max_bytes=CAP)
    with pytest.raises(ExportError):
        exports.parse_record(make_raw([], provenance={"model_id": 3}), max_bytes=CAP)
    assert make_record([], language=None).language is None


def test_host_provenance_stamp():
    record = make_record(make_segments((0, 1, "a")))
    stamped = exports.with_host_provenance(record, image_digest="sha256:" + "f" * 64, model_revision="abc123")
    assert record.provenance.image_digest is None and record.provenance.model_revision is None
    provenance = json.loads(exports.build_json(stamped))["provenance"]
    assert provenance == {
        "model_id": "small", "model_revision": "abc123", "image_digest": "sha256:" + "f" * 64,
        "device": "cpu", "compute_type": "int8", "source_sha256": "0" * 64,
        "attempt_id": "11111111-1111-4111-8111-111111111111",
    }
    with pytest.raises(ExportError):
        exports.with_host_provenance(record, image_digest="", model_revision=None)


def test_json_shape_exact():
    record = make_record(make_segments((0.5, 1.5, "hello")), duration=2.0)
    data = exports.build_json(record)
    document = json.loads(data)
    assert set(document) == {"schema_version", "language", "duration_seconds",
                             "no_speech_detected", "segments", "provenance"}
    assert document["schema_version"] == exports.EXPORT_SCHEMA_VERSION == 1
    assert document["segments"] == [{"start_seconds": 0.5, "end_seconds": 1.5, "text": "hello"}]
    assert b"measurements" not in data and b"speaker" not in data and b"language_probability" not in data


# --- readable paragraphs --------------------------------------------------------


def test_readable_paragraphs_preserve_words_and_break_at_gaps():
    segments = make_segments(
        (0.0, 1.0, "One two."), (1.2, 2.0, "Three\nfour."), (5.0, 6.0, "Five."), (6.5, 7.0, "Six."),
    )
    record = make_record(segments)
    text = exports.build_readable_txt(record).decode("utf-8")
    assert text == "One two. Three four.\n\nFive. Six.\n"
    original_words = " ".join(s["text"] for s in segments).split()
    assert text.split() == original_words


def test_readable_paragraphs_split_long_runs_at_segment_boundaries():
    words = [f"word{i:03d}" for i in range(300)]
    segments = make_segments(*[(i * 0.5, i * 0.5 + 0.4, words[i]) for i in range(300)])
    record = make_record(segments, duration=200.0)
    paragraphs = exports.paragraphs(record)
    assert len(paragraphs) > 1
    assert all(len(p) <= exports.PARAGRAPH_MAX_CHARS for p in paragraphs)
    assert " ".join(paragraphs).split() == words


# --- caps -----------------------------------------------------------------------


def test_cap_is_all_or_nothing():
    segments = make_segments(*[(float(i), float(i) + 0.9, "repeat " * 20) for i in range(500)])
    record = make_record(segments, duration=600.0)
    full = exports.export_all(record, TRANSCRIPT_SLOTS, cap_bytes=CAP)
    total = sum(len(v) for v in full.values())
    largest = max(len(v) for v in full.values())
    with pytest.raises(ExportError) as excinfo:
        exports.export_all(record, TRANSCRIPT_SLOTS, cap_bytes=total - 1)
    assert reason(excinfo) is ReasonCode.EXPORT_LIMIT_REACHED
    with pytest.raises(ExportError) as excinfo:
        exports.export_all(record, [ArtifactSlot.TRANSCRIPT_JSON], cap_bytes=largest - 1)
    assert reason(excinfo) is ReasonCode.EXPORT_LIMIT_REACHED
    assert exports.export_all(record, TRANSCRIPT_SLOTS, cap_bytes=total) == full
    assert list(full) == sorted(full, key=lambda slot: slot.value)


def test_estimate_monotonic_and_above_typical_speech():
    values = [exports.estimate_export_bytes(d, TRANSCRIPT_SLOTS) for d in (0, 1, 10, 60, 3600, 21600)]
    assert values == sorted(values) and len(set(values)) == len(values)
    # ~160 wpm English, one segment every ~3 s.
    duration = 600.0
    words = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dogs"]
    segments, t, i = [], 0.0, 0
    while t + 3.0 <= duration:
        segments.append((t, t + 2.9, " ".join(words[(i + k) % len(words)] for k in range(8))))
        t, i = t + 3.0, i + 8
    record = make_record(make_segments(*segments), duration=duration)
    outputs = exports.export_all(record, TRANSCRIPT_SLOTS, cap_bytes=CAP)
    assert exports.estimate_export_bytes(duration, TRANSCRIPT_SLOTS) > sum(map(len, outputs.values()))
    for slot, data in outputs.items():
        assert exports.estimate_export_bytes(duration, [slot]) > len(data)
    with pytest.raises(ExportError):
        exports.estimate_export_bytes(float("nan"), TRANSCRIPT_SLOTS)
    with pytest.raises(ExportError):
        exports.estimate_export_bytes(-1, TRANSCRIPT_SLOTS)


# --- slots and domains -----------------------------------------------------------


def test_registry_covers_exactly_transcript_slots():
    assert set(exports.EXPORT_SLOTS) == set(ArtifactSlot) - {ArtifactSlot.ORIGINAL_VIDEO}
    assert set(exports.EXPORT_SLOTS.values()) == set(ArtifactFormat) - {ArtifactFormat.VIDEO}


def test_all_six_slots():
    record = make_record(make_segments((0, 1, "a")))
    for slot in ArtifactSlot:
        if slot is ArtifactSlot.ORIGINAL_VIDEO:
            with pytest.raises(ExportError) as excinfo:
                exports.build(slot, record)
            assert reason(excinfo) is ReasonCode.INVALID_INPUT
            with pytest.raises(ExportError):
                exports.export_all(record, [slot], cap_bytes=CAP)
        else:
            assert exports.build(slot.value, record) == exports.export_all(record, [slot], cap_bytes=CAP)[slot]


def test_domain_negative_examples_rejected():
    contract = json.loads((PROJECT_ROOT / "Docs" / "contract-domains.json").read_text(encoding="utf-8"))
    negatives = contract["negative_examples"]
    assert negatives == ["__invalid__", ""]
    record = make_record([])
    assert [s.value for s in ArtifactSlot] == contract["artifact_slot"]
    assert [f.value for f in ArtifactFormat] == contract["artifact_format"]
    for value in negatives:
        for parse in (ArtifactSlot.parse, ArtifactFormat.parse, ReasonCode.parse):
            with pytest.raises(DomainValidationError):
                parse(value)
        with pytest.raises(DomainValidationError):
            exports.build(value, record)
        with pytest.raises(DomainValidationError):
            exports.export_all(record, [value], cap_bytes=CAP)
        with pytest.raises(DomainValidationError):
            exports.estimate_export_bytes(10, [value])
    with pytest.raises(ExportError):
        exports.export_all(record, [], cap_bytes=CAP)


# --- sidecar --------------------------------------------------------------------


def test_sidecar_has_no_text_or_paths():
    secret = "uniquephrase-zebra"
    record = make_record(make_segments((0, 1, secret), (2, 3, "日本語")))
    record = exports.with_host_provenance(record, image_digest="sha256:" + "a" * 64, model_revision="rev1")
    outputs = exports.export_all(record, TRANSCRIPT_SLOTS, cap_bytes=CAP)
    sidecar = exports.provenance_sidecar(record, outputs)
    assert secret.encode() not in sidecar and "日本語".encode() not in sidecar
    assert b"/" not in sidecar.replace(b"sha256:", b"") and b"\\" not in sidecar
    document = json.loads(sidecar)
    assert document["schema_version"] == exports.SIDECAR_SCHEMA_VERSION
    assert document["provenance"]["model_revision"] == "rev1"
    for slot, data in outputs.items():
        entry = document["outputs"][slot.value]
        assert entry == {"format": exports.EXPORT_SLOTS[slot].value, "bytes": len(data),
                         "sha256": hashlib.sha256(data).hexdigest()}
    with pytest.raises(ExportError):
        exports.provenance_sidecar(record, {ArtifactSlot.ORIGINAL_VIDEO: b""})
