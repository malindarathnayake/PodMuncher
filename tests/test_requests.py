"""Request identity, strict codec, and side-effect-free source admission."""

from dataclasses import FrozenInstanceError, replace
import hashlib
import json
import os
from pathlib import Path
import socket

import pytest

from podmuncher.domains import ReasonCode
from podmuncher.requests import JobRequest, RequestError, RequestLimits, RequestSource, classify_source


def make_request(tmp_path: Path, **changes: object) -> JobRequest:
    data = {
        "schema_version": 1,
        "source": {"kind": "local", "input": "missing.mp4"},
        "outputs": {"transcript": True, "video": False, "video_quality": "1080p"},
        "selection": {"device": "auto", "profile": "balanced", "model": "auto", "language": "auto"},
        "output_root": str(tmp_path / "output"),
        "limits": {"media_bytes": 1024, "duration_seconds": 60, "export_bytes": 512},
    }
    for key, value in changes.items():
        if key in ("source", "outputs", "selection", "limits"):
            data[key].update(value)
        else:
            data[key] = value
    return JobRequest.from_dict(data)


def test_canonical_identity_and_round_trip(tmp_path: Path) -> None:
    request = make_request(tmp_path)
    expected = json.dumps(request.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    assert request.canonical_json() == expected
    assert request.request_hash == hashlib.sha256(expected).hexdigest()
    assert JobRequest.from_json(expected).request_hash == request.request_hash
    assert set(request.to_dict()) == {"schema_version", "source", "outputs", "selection", "output_root", "limits"}


@pytest.mark.parametrize("field", ("media_bytes", "duration_seconds", "export_bytes"))
def test_limits_reject_integers_that_cannot_be_serialized(tmp_path: Path, field: str) -> None:
    oversized = 10**5000
    base = make_request(tmp_path)
    with pytest.raises(RequestError, match="^Invalid input\\.$"):
        RequestLimits(**{field: oversized})
    with pytest.raises(RequestError, match="^Invalid input\\.$"):
        make_request(tmp_path, limits={field: oversized})
    with pytest.raises(RequestError, match="^Invalid input\\.$"):
        base.clone({"limits": {field: oversized}})


@pytest.mark.parametrize("field", ("media_bytes", "duration_seconds", "export_bytes"))
def test_large_serializable_limits_keep_canonical_identity(tmp_path: Path, field: str) -> None:
    request = make_request(tmp_path, limits={field: 10**1000})
    canonical = request.canonical_json()
    assert request.to_json() == canonical.decode("utf-8")
    assert request.request_hash == hashlib.sha256(canonical).hexdigest()
    assert JobRequest.from_json(canonical).canonical_json() == canonical
    assert JobRequest.from_json(canonical).request_hash == request.request_hash


@pytest.mark.parametrize("change", [
    {"source": {"input": "other.mp4"}},
    {"outputs": {"video": True}},
    {"outputs": {"video_quality": "source_best"}},
    {"selection": {"device": "cpu"}},
    {"selection": {"profile": "quality"}},
    {"output_root": "another"},
    {"limits": {"media_bytes": 2048}},
    {"limits": {"duration_seconds": 61}},
    {"limits": {"export_bytes": 513}},
])
def test_every_setting_changes_hash(tmp_path: Path, change: dict) -> None:
    request = make_request(tmp_path)
    assert request.clone(change).request_hash != request.request_hash


def test_registry_selection_changes_hash(tmp_path: Path) -> None:
    request = JobRequest.from_dict(make_request(tmp_path).to_dict(), verified_models={"tiny"}, verified_languages={"en"})
    assert request.clone({"selection": {"model": "tiny"}}).request_hash != request.request_hash
    assert request.clone({"selection": {"language": "en"}}).request_hash != request.request_hash
    with pytest.raises(RequestError):
        make_request(tmp_path).clone({"selection": {"model": "tiny"}})


def test_clone_is_deep_and_frozen(tmp_path: Path) -> None:
    request = make_request(tmp_path)
    clone = request.clone({"outputs": {"video": True}})
    assert request.outputs.video is False
    assert clone.outputs.video is True
    assert request.output_mode.value == "transcript"
    assert clone.output_mode.value == "video_transcript"
    with pytest.raises(FrozenInstanceError):
        request.outputs.video = True
    copy = request.to_dict()
    copy["source"]["input"] = "changed"
    assert request.source.input == os.path.abspath("missing.mp4")
    with pytest.raises(RequestError):
        request.clone({"outputs": {"unknown": 1}})


@pytest.mark.parametrize("payload", [
    {"schema_version": True}, {"schema_version": 2},
    {"unexpected": 1}, {"source": {"kind": "remote"}},
    {"source": {"input": ""}}, {"source": {"input": "\ud800"}},
    {"source": {"input": "é" * 4097}},
    {"outputs": {"transcript": False, "video": False}},
    {"outputs": {"transcript": 1}},
    {"outputs": {"video_quality": "__invalid__"}},
    {"selection": {"device": "cpu "}},
    {"selection": {"language": "en"}},
    {"limits": {"duration_seconds": True}},
    {"limits": {"export_bytes": 0}},
    {"output_root": ""}, {"output_root": "bad\x00path"},
])
def test_bad_data_rejected(tmp_path: Path, payload: dict) -> None:
    data = make_request(tmp_path).to_dict()
    for key, value in payload.items():
        if type(value) is dict and key in data:
            data[key].update(value)
        else:
            data[key] = value
    with pytest.raises(RequestError) as caught:
        JobRequest.from_dict(data)
    assert str(caught.value) == "Invalid input."


def test_missing_nested_keys_and_json_types(tmp_path: Path) -> None:
    data = make_request(tmp_path).to_dict()
    del data["source"]["kind"]
    with pytest.raises(RequestError):
        JobRequest.from_dict(data)
    with pytest.raises(RequestError):
        JobRequest.from_json(b"\xff")
    with pytest.raises(RequestError):
        JobRequest.from_json("{\"x\": NaN}")
    with pytest.raises(RequestError):
        JobRequest.from_json('{"schema_version":1,"schema_version":1}')


def test_json_ingress_limits(tmp_path: Path) -> None:
    request = make_request(tmp_path)
    payload = request.to_json()
    assert JobRequest.from_json(payload).request_hash == request.request_hash
    boundary = " " * (65536 - len(payload.encode("utf-8"))) + payload
    assert JobRequest.from_json(boundary).request_hash == request.request_hash
    for excessive in ("[" * 10000 + "]" * 10000, "[" * 1100 + "]" * 1100,
                      "[" * 9 + "0" + "]" * 9):
        with pytest.raises(RequestError) as caught:
            JobRequest.from_json(excessive)
        assert caught.value.reason_code is ReasonCode.NESTING_LIMIT
    for excessive in (" " * 65537 + payload, b" " * 65537 + payload.encode("utf-8"),
                      json.dumps({"extra": "é" * 32768}, ensure_ascii=False)):
        with pytest.raises(RequestError) as caught:
            JobRequest.from_json(excessive)
        assert caught.value.reason_code is ReasonCode.REQUEST_TOO_LARGE
    data = request.to_dict()
    data["extra"] = "x" * 8193
    with pytest.raises(RequestError):
        JobRequest.from_json(json.dumps(data))


def test_local_source_captures_absolute_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    expected = str(tmp_path / "missing file.mp4")
    source = classify_source("missing file.mp4")
    assert source.input == expected
    assert RequestSource("local", "missing file.mp4").input == expected
    request = make_request(tmp_path, source={"input": "missing file.mp4"})
    original_json, original_hash = request.to_json(), request.request_hash
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert request.source.input == expected
    assert request.to_json() == original_json
    assert request.request_hash == original_hash
    assert JobRequest.from_json(original_json).source.input == expected
    assert request.clone({"outputs": {"video": True}}).source.input == expected


def test_windows_local_paths_are_captured_without_host_resolution() -> None:
    assert classify_source(r"C:\missing folder\video.mp4").input == r"C:\missing folder\video.mp4"
    assert classify_source(r"\\server\share\missing file.mp4").input == r"\\server\share\missing file.mp4"
    with pytest.raises(RequestError):
        classify_source(r"C:relative\video.mp4")


def test_invalid_replacement_cannot_serialize_or_hash(tmp_path: Path) -> None:
    request = make_request(tmp_path)
    with pytest.raises(RequestError):
        replace(request, output_root="")
    object.__setattr__(request.limits, "media_bytes", False)
    with pytest.raises(RequestError):
        request.to_dict()
    with pytest.raises(RequestError):
        request.request_hash


def test_canonical_json_preserves_validation_reason(tmp_path: Path) -> None:
    request = make_request(tmp_path)
    object.__setattr__(request, "schema_version", 2)
    with pytest.raises(RequestError) as caught:
        request.canonical_json()
    assert caught.value.reason_code is ReasonCode.UNSUPPORTED_SCHEMA


def test_domain_negative_examples(tmp_path: Path) -> None:
    contract = json.loads((Path(__file__).resolve().parents[1] / "Docs" / "contract-domains.json").read_text(encoding="utf-8"))
    request = make_request(tmp_path)
    for value in contract["negative_examples"]:
        for section, field in (("source", "kind"), ("outputs", "video_quality"),
                               ("selection", "device"), ("selection", "profile")):
            with pytest.raises(RequestError):
                request.clone({section: {field: value}})


@pytest.mark.parametrize("url", [
    "http://example.com/file.mp4", "https://user:pass@example.com/file.mp4",
    "https://127.0.0.1/file", "https://10.1.2.3/file", "https://[::ffff:127.0.0.1]/file",
    "https://[fe80::1]/file", "https://[ff02::1]/file", "https://[2001:db8::1]/file",
    "https://localhost/file", "https://app.localhost/file", "https://localhost.localdomain/file",
    "https://example.com:bad/file", "https://example.com:0/file", "https://example.com:65536/file",
    "https://example.com/\nfile", "https://youtube.com/playlist?list=abc",
    "https://youtube.com/watch?v=abcdefghijk&list=abc", "https://youtube.com/live/abcdefghijk",
    "https://youtube.com/channel/abcdefghijk", "https://youtu.be/abcdefghijk?list=abc",
    "https://evil.youtube.com/watch?v=abcdefghijk", "ftp://example.com/file",
])
def test_rejected_urls(url: str) -> None:
    with pytest.raises(RequestError):
        classify_source(url)


def test_youtube_forms_have_one_identity(tmp_path: Path) -> None:
    urls = ["https://www.youtube.com/watch?v=abcdefghijk", "https://m.youtube.com/shorts/abcdefghijk",
            "https://youtube.com/embed/abcdefghijk", "https://youtu.be/abcdefghijk?si=tracking"]
    requests = [make_request(tmp_path, source={"kind": "youtube", "input": url}) for url in urls]
    assert len({request.request_hash for request in requests}) == 1
    assert requests[0].source.input == "https://www.youtube.com/watch?v=abcdefghijk"


def test_classification_has_no_dns_socket_or_existence_probes(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("unexpected I/O")
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setattr(socket, "gethostbyname", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(Path, "exists", forbidden)
    monkeypatch.setattr(Path, "resolve", forbidden)
    assert classify_source(r"C:\missing\video.mp4").kind == "local"
    assert classify_source("missing.mp4").kind == "local"
    assert classify_source("https://media.example.com/video.mp4").kind == "direct"
    assert classify_source("https://youtu.be/abcdefghijk").kind == "youtube"
