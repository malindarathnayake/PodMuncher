"""Strict, side-effect-free configuration and audited persistence boundary.

Loading never creates directories. A persistence writer must first admit the
target, durably record an audit event, and atomically replace a protected file.
It owns symlink/race checks, restrictive permissions, and preserving the old
file on any failure; no unaudited filesystem writer is supplied here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
import tomllib
from types import MappingProxyType
from typing import Mapping, Protocol
from urllib.parse import urlsplit

import platformdirs

from .domains import Device, DomainValidationError, Profile, ReasonCode, Theme, VideoQuality


class ConfigError(DomainValidationError):
    """Config failure whose message never contains supplied values."""

    def __init__(self, reason_code: ReasonCode = ReasonCode.INVALID_CONFIG) -> None:
        super().__init__(reason_code)


@dataclass(frozen=True, slots=True)
class Paths:
    data: str
    output: str
    cache: str
    scratch: str
    docker_backing: str = ""


@dataclass(frozen=True, slots=True)
class Selection:
    device: str = "auto"
    profile: str = "balanced"
    model: str = "auto"
    language: str = "auto"


@dataclass(frozen=True, slots=True)
class Outputs:
    transcript: bool = True
    video: bool = False
    video_quality: str = "1080p"


@dataclass(frozen=True, slots=True)
class Limits:
    media_bytes: int = 10737418240
    duration_seconds: int = 21600
    export_bytes: int = 268435456
    download_timeout_seconds: int = 3600
    decode_timeout_seconds: int = 1800
    worker_stop_grace_seconds: int = 10
    download_attempts: int = 3
    reserve_fraction: float = 0.05
    reserve_floor_bytes: int = 2147483648
    reserve_ceiling_bytes: int = 10737418240
    active_operations: int = 1
    pending_jobs: int = 100


@dataclass(frozen=True, slots=True)
class Monitoring:
    sample_seconds: int = 1
    history_samples: int = 600
    stale_after_seconds: int = 3


@dataclass(frozen=True, slots=True)
class Terminal:
    theme: str = "auto"
    reduced_motion: bool = True


@dataclass(frozen=True, slots=True)
class Telemetry:
    otlp_enabled: bool = False
    otlp_endpoint: str = ""


@dataclass(frozen=True, slots=True)
class TLS:
    ca_bundle: str = ""


class AuditedConfigWriter(Protocol):
    """Injected writer enforcing protected-file admission and atomicity.

    admit must reject unsafe/unprotected targets before any write. audit must
    durably record the intended change before atomic_replace commits it.
    atomic_replace must preserve the old file on failure and protect new files.
    """

    def admit(self, target: Path) -> None: ...

    def audit(self, target: Path, payload: bytes) -> None: ...

    def atomic_replace(self, target: Path, payload: bytes) -> None: ...


_SECTIONS = ("paths", "selection", "outputs", "limits", "monitoring", "terminal", "telemetry", "tls")
_SECTION_TYPES = dict(zip(_SECTIONS, (Paths, Selection, Outputs, Limits, Monitoring, Terminal, Telemetry, TLS)))
_FIELDS = {
    "paths": frozenset(Paths.__dataclass_fields__),
    "selection": frozenset(Selection.__dataclass_fields__),
    "outputs": frozenset(Outputs.__dataclass_fields__),
    "limits": frozenset(Limits.__dataclass_fields__),
    "monitoring": frozenset(Monitoring.__dataclass_fields__),
    "terminal": frozenset(Terminal.__dataclass_fields__),
    "telemetry": frozenset(Telemetry.__dataclass_fields__),
    "tls": frozenset(TLS.__dataclass_fields__),
}


def _mapping(value: object, allowed: frozenset[str] | set[str]) -> dict:
    if type(value) is not dict or any(type(k) is not str for k in value) or not value.keys() <= allowed:
        raise ConfigError()
    return value


def _string(value: object) -> str:
    if type(value) is not str:
        raise ConfigError()
    try:
        if len(value.encode("utf-8")) > 8192:
            raise ConfigError()
    except UnicodeEncodeError:
        raise ConfigError() from None
    return value


def _path(value: object, *, empty_ok: bool = False) -> str:
    value = _string(value)
    if "\x00" in value:
        raise ConfigError()
    if not value:
        if empty_ok:
            return ""
        raise ConfigError()
    try:
        return _string(str(Path(value).expanduser().resolve(strict=False)))
    except (OSError, RuntimeError, ValueError):
        raise ConfigError() from None


def _path_or_default(value: object, fallback: object) -> str:
    if type(value) is not str:
        raise ConfigError()
    return _path(value or fallback)


def _integer(value: object, minimum: int, maximum: int | None = None) -> int:
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ConfigError()
    # Accepted integers must be serializable before a writer is called.
    try:
        str(value)
    except ValueError:
        raise ConfigError() from None
    return value


def _number(value: object, *, minimum: float = 0, maximum: float | None = None, positive: bool = False) -> float | int:
    if type(value) not in (int, float):
        raise ConfigError()
    try:
        finite = math.isfinite(value)
    except OverflowError:
        raise ConfigError() from None
    if not finite:
        raise ConfigError()
    if value < minimum or (positive and value == 0) or (maximum is not None and value > maximum):
        raise ConfigError()
    return value


def _enum(enum: type, value: object) -> str:
    try:
        return enum.parse(value).value
    except DomainValidationError:
        raise ConfigError() from None


def _registry_value(value: object, verified: frozenset[str] | set[str] | None) -> str:
    value = _string(value)
    if not value or (value != "auto" and (verified is None or value not in verified)):
        raise ConfigError()
    return value


def _registry_snapshot(verified: frozenset[str] | set[str] | None) -> frozenset[str] | None:
    if verified is None:
        return None
    if type(verified) not in (set, frozenset):
        raise ConfigError()
    return frozenset(_string(value) for value in verified)


def _defaults() -> dict:
    return {
        "schema_version": 1,
        "paths": asdict(Paths("", "", "", "")),
        "selection": asdict(Selection()),
        "outputs": asdict(Outputs()),
        "limits": asdict(Limits()),
        "monitoring": asdict(Monitoring()),
        "terminal": asdict(Terminal()),
        "telemetry": asdict(Telemetry()),
        "tls": asdict(TLS()),
    }


def _layer(target: dict, origin: dict, layer: object, label: str) -> None:
    source = _mapping(layer, set(_SECTIONS) | {"schema_version"})
    if "schema_version" in source:
        if type(source["schema_version"]) is not int or source["schema_version"] != 1:
            raise ConfigError(ReasonCode.UNSUPPORTED_SCHEMA)
        origin["schema_version"] = label
    for section in _SECTIONS:
        if section in source:
            entries = _mapping(source[section], _FIELDS[section])
            target[section].update(entries)
            origin[section].update(dict.fromkeys(entries, label))


def _validate(values: dict, models: frozenset[str] | set[str] | None, languages: frozenset[str] | set[str] | None) -> None:
    paths = values["paths"]
    try:
        data_default = platformdirs.user_data_dir("podmuncher", appauthor=False)
        cache_default = platformdirs.user_cache_dir("podmuncher", appauthor=False)
    except (OSError, RuntimeError, ValueError):
        raise ConfigError() from None
    paths["data"] = _path_or_default(paths["data"], data_default)
    paths["cache"] = _path_or_default(paths["cache"], cache_default)
    paths["output"] = _path_or_default(paths["output"], str(Path(paths["data"]) / "outputs"))
    paths["scratch"] = _path_or_default(paths["scratch"], str(Path(paths["cache"]) / "scratch"))
    paths["docker_backing"] = _path(paths["docker_backing"], empty_ok=True)
    selection = values["selection"]
    selection["device"] = _enum(Device, selection["device"])
    selection["profile"] = _enum(Profile, selection["profile"])
    selection["model"] = _registry_value(selection["model"], models)
    selection["language"] = _registry_value(selection["language"], languages)
    outputs = values["outputs"]
    for key in ("transcript", "video"):
        if type(outputs[key]) is not bool:
            raise ConfigError()
    if not (outputs["transcript"] or outputs["video"]):
        raise ConfigError()
    outputs["video_quality"] = _enum(VideoQuality, outputs["video_quality"])
    limits = values["limits"]
    for key in ("media_bytes", "duration_seconds", "export_bytes", "download_timeout_seconds", "decode_timeout_seconds", "worker_stop_grace_seconds"):
        limits[key] = _integer(limits[key], 1)
    limits["download_attempts"] = _integer(limits["download_attempts"], 1, 3)
    limits["reserve_fraction"] = _number(limits["reserve_fraction"])
    limits["reserve_floor_bytes"] = _integer(limits["reserve_floor_bytes"], 0)
    limits["reserve_ceiling_bytes"] = _integer(limits["reserve_ceiling_bytes"], 0)
    if limits["reserve_floor_bytes"] > limits["reserve_ceiling_bytes"] or type(limits["active_operations"]) is not int or limits["active_operations"] != 1:
        raise ConfigError()
    limits["pending_jobs"] = _integer(limits["pending_jobs"], 1, 100)
    monitoring = values["monitoring"]
    monitoring["sample_seconds"] = _integer(monitoring["sample_seconds"], 1)
    monitoring["history_samples"] = _integer(monitoring["history_samples"], 1, 600)
    monitoring["stale_after_seconds"] = _integer(monitoring["stale_after_seconds"], 1)
    terminal = values["terminal"]
    terminal["theme"] = _enum(Theme, terminal["theme"])
    if type(terminal["reduced_motion"]) is not bool:
        raise ConfigError()
    telemetry = values["telemetry"]
    if type(telemetry["otlp_enabled"]) is not bool:
        raise ConfigError()
    endpoint = _string(telemetry["otlp_endpoint"])
    if endpoint and (endpoint != endpoint.strip() or any(ord(c) < 32 for c in endpoint)):
        raise ConfigError()
    if telemetry["otlp_enabled"]:
        try:
            url = urlsplit(endpoint)
            valid = url.scheme in ("http", "https") and bool(url.hostname) and url.port is not None and not url.username and not url.password and not url.path.strip("/") and not url.query and not url.fragment
        except ValueError:
            valid = False
        if not valid:
            raise ConfigError()
    values["tls"]["ca_bundle"] = _path(values["tls"]["ca_bundle"], empty_ok=True)


@dataclass(frozen=True, slots=True)
class Config:
    schema_version: int
    paths: Paths
    selection: Selection
    outputs: Outputs
    limits: Limits
    monitoring: Monitoring
    terminal: Terminal
    telemetry: Telemetry
    tls: TLS
    origins: Mapping[str, object] = field(repr=False, compare=False)
    _verified_models: frozenset[str] | None = field(default=None, init=False, repr=False, compare=False)
    _verified_languages: frozenset[str] | None = field(default=None, init=False, repr=False, compare=False)

    @classmethod
    def from_dict(cls, data: object = None, *, overrides: object = None, verified_models: frozenset[str] | set[str] | None = None, verified_languages: frozenset[str] | set[str] | None = None) -> Config:
        models = _registry_snapshot(verified_models)
        languages = _registry_snapshot(verified_languages)
        values = _defaults()
        origin: dict[str, object] = {"schema_version": "default", **{name: dict.fromkeys(_FIELDS[name], "default") for name in _SECTIONS}}
        if data is not None:
            _layer(values, origin, data, "config")
        if overrides is not None:
            _layer(values, origin, overrides, "override")
        _validate(values, models, languages)
        frozen_origins = MappingProxyType({key: MappingProxyType(value.copy()) if type(value) is dict else value for key, value in origin.items()})
        result = cls(1, Paths(**values["paths"]), Selection(**values["selection"]), Outputs(**values["outputs"]), Limits(**values["limits"]), Monitoring(**values["monitoring"]), Terminal(**values["terminal"]), Telemetry(**values["telemetry"]), TLS(**values["tls"]), frozen_origins)
        object.__setattr__(result, "_verified_models", models)
        object.__setattr__(result, "_verified_languages", languages)
        return result

    @classmethod
    def from_toml(cls, source: str | bytes, **kwargs: object) -> Config:
        try:
            if type(source) is bytes:
                source = source.decode("utf-8")
            if type(source) is not str:
                raise ConfigError()
            data = tomllib.loads(source)
        except (UnicodeError, tomllib.TOMLDecodeError, ValueError, OverflowError, RecursionError):
            raise ConfigError() from None
        return cls.from_dict(data, **kwargs)

    def to_dict(self) -> dict:
        return {"schema_version": self.schema_version, **{name: asdict(getattr(self, name)) for name in _SECTIONS}}

    def _validated_data(self) -> dict:
        if type(self) is not Config or any(type(getattr(self, name, None)) is not section_type for name, section_type in _SECTION_TYPES.items()):
            raise ConfigError()
        if any(registry is not None and type(registry) is not frozenset for registry in (self._verified_models, self._verified_languages)):
            raise ConfigError()
        data = self.to_dict()
        values = _defaults()
        origin: dict[str, object] = {"schema_version": "default", **{name: {} for name in _SECTIONS}}
        _layer(values, origin, data, "config")
        _validate(values, self._verified_models, self._verified_languages)
        if values != data:
            raise ConfigError()
        return data

    def to_toml(self) -> str:
        try:
            data = self._validated_data()
            lines = [f"schema_version = {data['schema_version']}"]
            for section in _SECTIONS:
                lines.extend(("", f"[{section}]"))
                for key, value in data[section].items():
                    if type(value) is str:
                        # JSON's ASCII escaping is not TOML; preserve Unicode literally.
                        encoded = json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007f")
                    elif type(value) is bool:
                        encoded = "true" if value else "false"
                    else:
                        encoded = str(value)
                    lines.append(f"{key} = {encoded}")
            return "\n".join(lines) + "\n"
        except ConfigError:
            raise
        except (ValueError, OverflowError, RecursionError):
            raise ConfigError() from None

    def persist(self, target: str | Path, *, writer: AuditedConfigWriter) -> None:
        """Persist only through an explicitly injected audited writer.

        The writer contract requires admission, durable audit, then protected
        atomic replacement. Config intentionally has no filesystem fallback.
        """
        if writer is None or not all(callable(getattr(writer, name, None)) for name in ("admit", "audit", "atomic_replace")):
            raise ConfigError()
        if type(target) is not str and not isinstance(target, Path):
            raise ConfigError()
        target_path = Path(_path(str(target) if isinstance(target, Path) else target))
        payload = self.to_toml().encode("utf-8")
        try:
            writer.admit(target_path)
        except DomainValidationError as error:
            raise ConfigError(error.reason_code) from None
        except Exception:
            raise ConfigError(ReasonCode.STATE_WRITE_FAILED) from None
        try:
            writer.audit(target_path, payload)
        except Exception:
            raise ConfigError(ReasonCode.AUDIT_LOSS) from None
        try:
            writer.atomic_replace(target_path, payload)
        except Exception:
            raise ConfigError(ReasonCode.STATE_WRITE_FAILED) from None
