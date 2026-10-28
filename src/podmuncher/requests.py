"""Immutable, side-effect-free job request codec and source admission."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import ipaddress
import json
import ntpath
import os
import re
from typing import Mapping
from urllib.parse import parse_qs, urlsplit

from .domains import (
    Device, DomainValidationError, OutputMode, Profile, ReasonCode, Source,
    VideoQuality, validate_json_payload, validate_schema_version, validate_utf8_string,
)


class RequestError(DomainValidationError):
    """Validation error with no supplied value in its message."""

    def __init__(self, reason_code: ReasonCode = ReasonCode.INVALID_INPUT) -> None:
        super().__init__(reason_code)


_VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}\Z")
_WINDOWS_PATH = re.compile(r"^(?:[A-Za-z]:|\\\\)")
_DNS_LABEL = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\Z")
_TOP = frozenset(("schema_version", "source", "outputs", "selection", "output_root", "limits"))
_FIELDS = {
    "source": frozenset(("kind", "input")),
    "outputs": frozenset(("transcript", "video", "video_quality")),
    "selection": frozenset(("device", "profile", "model", "language")),
    "limits": frozenset(("media_bytes", "duration_seconds", "export_bytes")),
}
_MAX_JSON_BYTES = 65536


def _object(value: object, keys: frozenset[str]) -> dict:
    if type(value) is not dict or value.keys() != keys or any(type(k) is not str for k in value):
        raise RequestError()
    return value


def _string(value: object, *, nonempty: bool = True) -> str:
    try:
        result = validate_utf8_string(value)
    except DomainValidationError:
        raise RequestError() from None
    if nonempty and not result:
        raise RequestError()
    return result


def _enum(kind: type, value: object) -> str:
    try:
        return kind.parse(value).value
    except DomainValidationError:
        raise RequestError() from None


def _registry(verified: frozenset[str] | set[str] | None) -> frozenset[str] | None:
    if verified is None:
        return None
    if type(verified) not in (set, frozenset):
        raise RequestError()
    return frozenset(_string(item) for item in verified)


def _selected(value: object, verified: frozenset[str] | None) -> str:
    value = _string(value)
    if value != "auto" and (verified is None or value not in verified):
        raise RequestError()
    return value


def _output_root(value: object) -> str:
    value = _string(value)
    if "\x00" in value:
        raise RequestError()
    try:
        if _WINDOWS_PATH.match(value):
            if not ntpath.isabs(value):
                raise RequestError()
            result = ntpath.normpath(value)
        else:
            result = os.path.abspath(os.path.expanduser(value))
        return _string(result)
    except (OSError, RuntimeError, ValueError):
        raise RequestError() from None


def _local_source(value: str) -> str:
    try:
        if _WINDOWS_PATH.match(value):
            if not ntpath.isabs(value):
                raise RequestError()
            result = ntpath.normpath(value)
        else:
            result = os.path.abspath(os.path.expanduser(value))
        return _string(result)
    except (OSError, RuntimeError, ValueError):
        raise RequestError() from None


def _public_host(host: str) -> bool:
    host = host.rstrip(".").lower()
    if not host or host == "localhost" or host.endswith(".localhost") or host == "localhost.localdomain" or host.endswith(".localhost.localdomain"):
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if len(host) > 253 or "." not in host:
            return False
        labels = host.split(".")
        # Reject numeric and hexadecimal aliases that URL stacks can reinterpret.
        return all(_DNS_LABEL.fullmatch(label) for label in labels) and not all(
            re.fullmatch(r"(?:0[xX][0-9a-fA-F]+|0[0-7]*|[0-9]+)", label) for label in labels
        )
    if isinstance(address, ipaddress.IPv6Address) and address.ipv4_mapped:
        address = address.ipv4_mapped
    return address.is_global and not any((address.is_multicast, address.is_reserved,
                                          address.is_link_local, address.is_loopback,
                                          address.is_unspecified, address.is_private))


def _admit_source(value: object) -> tuple[str, str]:
    value = _string(value)
    if "\x00" in value or any(ord(ch) < 32 or ord(ch) == 127 for ch in value):
        raise RequestError()
    if _WINDOWS_PATH.match(value) or not re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", value):
        return "local", _local_source(value)
    try:
        url = urlsplit(value)
        host = url.hostname
        port = url.port
    except ValueError:
        raise RequestError() from None
    if url.scheme != "https" or not host or url.username is not None or url.password is not None or url.fragment or not _public_host(host):
        raise RequestError()
    if port is not None and not 1 <= port <= 65535:
        raise RequestError()
    host = host.rstrip(".").lower()
    if (host.endswith(".youtube.com") and host not in ("www.youtube.com", "m.youtube.com")) or host.endswith(".youtu.be"):
        raise RequestError()
    if host in ("youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be"):
        if port not in (None, 443):
            raise RequestError()
        if host == "youtu.be":
            candidate = url.path.removeprefix("/")
            if url.query:
                # Share links may carry tracking parameters, never playlist state.
                query = parse_qs(url.query, keep_blank_values=True)
                if "list" in query or "v" in query:
                    raise RequestError()
        else:
            path = url.path
            query = parse_qs(url.query, keep_blank_values=True)
            if "list" in query:
                raise RequestError()
            if path == "/watch":
                if len(query.get("v", [])) != 1:
                    raise RequestError()
                candidate = query["v"][0]
            elif path.startswith(("/shorts/", "/embed/")):
                candidate = path.split("/", 2)[2]
                if "v" in query:
                    raise RequestError()
            else:
                raise RequestError()
        if not _VIDEO_ID.fullmatch(candidate):
            raise RequestError()
        return "youtube", f"https://www.youtube.com/watch?v={candidate}"
    return "direct", value


def classify_source(value: object) -> RequestSource:
    """Classify a local path or admitted public HTTPS URL without I/O."""
    return RequestSource(*_admit_source(value))


@dataclass(frozen=True, slots=True)
class RequestSource:
    kind: str
    input: str

    def __post_init__(self) -> None:
        _enum(Source, self.kind)
        kind, admitted = _admit_source(self.input)
        if self.kind != kind or (kind != "local" and self.input != admitted):
            raise RequestError()
        if kind == "local":
            object.__setattr__(self, "input", admitted)


@dataclass(frozen=True, slots=True)
class RequestOutputs:
    transcript: bool = True
    video: bool = False
    video_quality: str = "1080p"

    def __post_init__(self) -> None:
        if type(self.transcript) is not bool or type(self.video) is not bool or not (self.transcript or self.video):
            raise RequestError()
        _enum(VideoQuality, self.video_quality)

    @property
    def mode(self) -> OutputMode:
        if self.transcript and self.video:
            return OutputMode.VIDEO_TRANSCRIPT
        return OutputMode.TRANSCRIPT if self.transcript else OutputMode.VIDEO


@dataclass(frozen=True, slots=True)
class RequestSelection:
    device: str = "auto"
    profile: str = "balanced"
    model: str = "auto"
    language: str = "auto"

    def __post_init__(self) -> None:
        _enum(Device, self.device)
        _enum(Profile, self.profile)
        _string(self.model)
        _string(self.language)


@dataclass(frozen=True, slots=True)
class RequestLimits:
    media_bytes: int = 10737418240
    duration_seconds: int = 21600
    export_bytes: int = 268435456

    def __post_init__(self) -> None:
        if any(type(item) is not int or item <= 0 for item in (self.media_bytes, self.duration_seconds, self.export_bytes)):
            raise RequestError()
        try:
            for item in (self.media_bytes, self.duration_seconds, self.export_bytes):
                str(item)
        except ValueError:
            raise RequestError() from None


@dataclass(frozen=True, slots=True)
class JobRequest:
    schema_version: int
    source: RequestSource
    outputs: RequestOutputs
    selection: RequestSelection
    output_root: str
    limits: RequestLimits
    _verified_models: frozenset[str] | None = field(default=None, repr=False, compare=False)
    _verified_languages: frozenset[str] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._validated_data()

    @classmethod
    def from_dict(cls, data: object, *, verified_models: frozenset[str] | set[str] | None = None,
                  verified_languages: frozenset[str] | set[str] | None = None) -> JobRequest:
        data = _object(data, _TOP)
        try:
            validate_schema_version(data["schema_version"])
        except DomainValidationError as error:
            raise RequestError(error.reason_code) from None
        source = _object(data["source"], _FIELDS["source"])
        outputs = _object(data["outputs"], _FIELDS["outputs"])
        selection = _object(data["selection"], _FIELDS["selection"])
        limits = _object(data["limits"], _FIELDS["limits"])
        models, languages = _registry(verified_models), _registry(verified_languages)
        kind = _enum(Source, source["kind"])
        admitted = classify_source(source["input"])
        if kind != admitted.kind:
            raise RequestError()
        return cls(1, admitted, RequestOutputs(**outputs), RequestSelection(**selection),
                   _output_root(data["output_root"]), RequestLimits(**limits), models, languages)

    @classmethod
    def from_json(cls, payload: str | bytes, **kwargs: object) -> JobRequest:
        def unique_object(pairs: list[tuple[str, object]]) -> dict:
            result: dict = {}
            for key, value in pairs:
                if key in result:
                    raise RequestError()
                result[key] = value
            return result

        try:
            if type(payload) is bytes:
                if len(payload) > _MAX_JSON_BYTES:
                    raise RequestError(ReasonCode.REQUEST_TOO_LARGE)
                payload = payload.decode("utf-8", errors="strict")
            if type(payload) is not str:
                raise RequestError()
            if len(payload) > _MAX_JSON_BYTES or len(payload.encode("utf-8", errors="strict")) > _MAX_JSON_BYTES:
                raise RequestError(ReasonCode.REQUEST_TOO_LARGE)
            data = json.loads(payload, object_pairs_hook=unique_object,
                              parse_constant=lambda _: (_ for _ in ()).throw(RequestError()))
            validate_json_payload(data, max_depth=8, max_string_bytes=8192)
        except RecursionError:
            raise RequestError(ReasonCode.NESTING_LIMIT) from None
        except DomainValidationError as error:
            raise RequestError(error.reason_code) from None
        except (UnicodeError, ValueError):
            raise RequestError() from None
        return cls.from_dict(data, **kwargs)

    def _validated_data(self) -> dict:
        if type(self) is not JobRequest or type(self.source) is not RequestSource or type(self.outputs) is not RequestOutputs or type(self.selection) is not RequestSelection or type(self.limits) is not RequestLimits:
            raise RequestError()
        if type(self._verified_models) not in (frozenset, type(None)) or type(self._verified_languages) not in (frozenset, type(None)):
            raise RequestError()
        try:
            validate_schema_version(self.schema_version)
        except DomainValidationError as error:
            raise RequestError(error.reason_code) from None
        RequestSource(**asdict(self.source))
        RequestOutputs(**asdict(self.outputs))
        RequestSelection(**asdict(self.selection))
        RequestLimits(**asdict(self.limits))
        _selected(self.selection.model, self._verified_models)
        _selected(self.selection.language, self._verified_languages)
        if _output_root(self.output_root) != self.output_root:
            raise RequestError()
        return {"schema_version": 1, "source": asdict(self.source), "outputs": asdict(self.outputs),
                "selection": asdict(self.selection), "output_root": self.output_root, "limits": asdict(self.limits)}

    def to_dict(self) -> dict:
        return self._validated_data()

    def canonical_json(self) -> bytes:
        data = self._validated_data()
        try:
            return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                              allow_nan=False).encode("utf-8")
        except (ValueError, UnicodeError):
            raise RequestError() from None

    def to_json(self) -> str:
        return self.canonical_json().decode("utf-8")

    @property
    def request_hash(self) -> str:
        return hashlib.sha256(self.canonical_json()).hexdigest()

    @property
    def output_mode(self) -> OutputMode:
        self._validated_data()
        return self.outputs.mode

    def clone(self, overrides: Mapping[str, object] | None = None, **changes: object) -> JobRequest:
        data = self._validated_data()
        for layer in (overrides, changes):
            if layer is None:
                continue
            if type(layer) is not dict or any(type(key) is not str or key not in _TOP for key in layer):
                raise RequestError()
            for key, value in layer.items():
                if key in _FIELDS:
                    if type(value) is not dict or any(type(k) is not str or k not in _FIELDS[key] for k in value):
                        raise RequestError()
                    data[key].update(value)
                else:
                    data[key] = value
        return JobRequest.from_dict(data, verified_models=self._verified_models,
                                    verified_languages=self._verified_languages)


__all__ = ("JobRequest", "RequestSource", "RequestOutputs", "RequestSelection", "RequestLimits",
           "RequestError", "classify_source")
