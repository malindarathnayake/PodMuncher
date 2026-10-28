"""Configuration codec, validation, provenance, and persistence boundary tests."""

from dataclasses import FrozenInstanceError, replace
import json
import os
from pathlib import Path

import pytest

from podmuncher import config
from podmuncher.domains import DomainValidationError, ReasonCode


REGISTRY = Path(__file__).resolve().parents[1] / "Docs" / "contract-domains.json"
LITERAL_DEFAULTS = """schema_version = 1
[paths]
data = ''
output = ''
cache = ''
scratch = ''
docker_backing = ''
[selection]
device = 'auto'
profile = 'balanced'
model = 'auto'
language = 'auto'
[outputs]
transcript = true
video = false
video_quality = '1080p'
[limits]
media_bytes = 10737418240
duration_seconds = 21600
export_bytes = 268435456
download_timeout_seconds = 3600
decode_timeout_seconds = 1800
worker_stop_grace_seconds = 10
download_attempts = 3
reserve_fraction = 0.05
reserve_floor_bytes = 2147483648
reserve_ceiling_bytes = 10737418240
active_operations = 1
pending_jobs = 100
[monitoring]
sample_seconds = 1
history_samples = 600
stale_after_seconds = 3
[terminal]
theme = 'auto'
reduced_motion = true
[telemetry]
otlp_enabled = false
otlp_endpoint = ''
[tls]
ca_bundle = ''
"""


@pytest.fixture
def make_config(monkeypatch, tmp_path):
    monkeypatch.setattr(config.platformdirs, "user_data_dir", lambda *a, **k: str(tmp_path / "user-data"))
    monkeypatch.setattr(config.platformdirs, "user_cache_dir", lambda *a, **k: str(tmp_path / "user-cache"))
    return lambda data=None, **kwargs: config.Config.from_dict(data, **kwargs)


def test_defaults_roundtrip_no_directory_creation(make_config, tmp_path):
    value = make_config()
    assert value.paths.data == str(tmp_path / "user-data")
    assert value.paths.output == str(tmp_path / "user-data" / "outputs")
    assert value.paths.cache == str(tmp_path / "user-cache")
    assert value.paths.scratch == str(tmp_path / "user-cache" / "scratch")
    assert value.paths.docker_backing == ""
    assert not (tmp_path / "user-data").exists()
    assert not (tmp_path / "user-cache").exists()
    assert config.Config.from_toml(value.to_toml()).to_dict() == value.to_dict()
    assert config.Config.from_toml(value.to_toml().encode()).to_dict() == value.to_dict()
    assert value.origins["paths"]["data"] == "default"


def test_literal_default_toml_resolves_paths_without_writes(make_config, tmp_path):
    expected = make_config()
    parsed = config.Config.from_toml(LITERAL_DEFAULTS)
    assert parsed.to_dict() == expected.to_dict()
    assert not (tmp_path / "user-data").exists()
    assert not (tmp_path / "user-cache").exists()


def test_derived_paths_follow_resolved_parent_and_empty_values(make_config, tmp_path):
    value = make_config({"paths": {"data": str(tmp_path / "custom-data"), "cache": str(tmp_path / "custom-cache"), "output": "", "scratch": ""}})
    assert value.paths.output == str(tmp_path / "custom-data" / "outputs")
    assert value.paths.scratch == str(tmp_path / "custom-cache" / "scratch")
    assert value.origins["paths"]["output"] == "config"
    assert not (tmp_path / "custom-data").exists()


def test_precedence_false_zero_and_no_mutable_leaks(make_config):
    supplied = {"outputs": {"video": True}, "limits": {"reserve_fraction": 0.2}}
    overrides = {"outputs": {"video": False}, "limits": {"reserve_fraction": 0}}
    value = make_config(supplied, overrides=overrides)
    supplied["outputs"]["video"] = False
    overrides["limits"]["reserve_fraction"] = 1
    assert value.outputs.video is False
    assert value.limits.reserve_fraction == 0
    assert value.origins["outputs"]["video"] == "override"
    assert value.origins["limits"]["reserve_fraction"] == "override"
    assert value.origins["outputs"]["transcript"] == "default"
    exported = value.to_dict()
    exported["outputs"]["transcript"] = False
    assert value.outputs.transcript is True
    with pytest.raises((FrozenInstanceError, AttributeError)):
        value.outputs.video = True
    with pytest.raises(TypeError):
        value.origins["limits"]["reserve_fraction"] = "config"


@pytest.mark.parametrize("bad", [
    {"unknown": 1}, {"paths": {"unknown": "x"}}, {"outputs": []},
    {"paths": {"data": 3}}, {"paths": {"output": "bad\x00path"}},
    {"outputs": {"video": 1}}, {"outputs": {"transcript": False, "video": False}},
    {"limits": {"media_bytes": 0}}, {"limits": {"export_bytes": True}},
    {"limits": {"duration_seconds": -1}}, {"limits": {"reserve_fraction": float("nan")}},
    {"limits": {"reserve_fraction": -0.1}}, {"limits": {"reserve_floor_bytes": 12_000_000_000}},
    {"limits": {"active_operations": 2}}, {"limits": {"pending_jobs": 0}},
    {"limits": {"download_attempts": 4}}, {"monitoring": {"history_samples": 601}},
    {"terminal": {"reduced_motion": 0}}, {"telemetry": {"otlp_enabled": True}},
    {"telemetry": {"otlp_enabled": True, "otlp_endpoint": "https://name:secret@host:4317"}},
    {"tls": {"ca_bundle": 42}},
])
def test_invalid_input_is_safe(make_config, bad):
    with pytest.raises(config.ConfigError) as error:
        make_config(bad)
    assert error.value.reason_code is ReasonCode.INVALID_CONFIG
    assert "secret" not in str(error.value)


@pytest.mark.parametrize("version", [0, 2, True, "1", None])
def test_schema_migration_is_rejected(make_config, version):
    with pytest.raises(config.ConfigError) as error:
        make_config({"schema_version": version})
    assert error.value.reason_code is ReasonCode.UNSUPPORTED_SCHEMA


def test_closed_enums_use_contract_oracle(make_config):
    oracle = json.loads(REGISTRY.read_text(encoding="utf-8"))
    for name, values in (("device", oracle["device"]), ("profile", oracle["profile"])):
        for value in values:
            assert getattr(make_config({"selection": {name: value}}).selection, name) == value
        for bad in oracle["negative_examples"]:
            with pytest.raises(config.ConfigError):
                make_config({"selection": {name: bad}})
    for value in oracle["video_quality"]:
        assert make_config({"outputs": {"video_quality": value}}).outputs.video_quality == value
    for value in oracle["theme"]:
        assert make_config({"terminal": {"theme": value}}).terminal.theme == value
    for bad in oracle["negative_examples"]:
        for section, key in (("outputs", "video_quality"), ("terminal", "theme")):
            with pytest.raises(config.ConfigError):
                make_config({section: {key: bad}})


def test_model_language_must_come_from_injected_registry(make_config):
    requested = {"selection": {"model": "verified-model", "language": "en"}}
    with pytest.raises(config.ConfigError):
        make_config(requested)
    value = make_config(requested, verified_models={"verified-model"}, verified_languages={"en"})
    assert value.selection.model == "verified-model"
    assert value.selection.language == "en"
    with pytest.raises(config.ConfigError):
        make_config({"selection": {"model": "__invalid__"}}, verified_models={"verified-model"})


def test_large_numbers_and_utf8_strings_are_safely_rejected(make_config):
    for section, key in (("limits", "reserve_fraction"),):
        with pytest.raises(config.ConfigError) as error:
            make_config({section: {key: 10**1000}})
        assert error.value.reason_code is ReasonCode.INVALID_CONFIG
    for section, key in (("paths", "data"), ("selection", "model"), ("telemetry", "otlp_endpoint"), ("tls", "ca_bundle")):
        for bad in ("x" * 8193, "\ud800"):
            with pytest.raises(config.ConfigError) as error:
                make_config({section: {key: bad}})
            assert error.value.reason_code is ReasonCode.INVALID_CONFIG


@pytest.mark.parametrize("key", ["sample_seconds", "stale_after_seconds"])
@pytest.mark.parametrize("bad", [0, -1, 0.1, 0.5, 1.0, True, "1"])
def test_monitoring_times_require_positive_integers(make_config, key, bad):
    with pytest.raises(config.ConfigError) as error:
        make_config({"monitoring": {key: bad}})
    assert error.value.reason_code is ReasonCode.INVALID_CONFIG


def test_positive_integer_monitoring_times_roundtrip(make_config):
    value = make_config({"monitoring": {"sample_seconds": 2, "stale_after_seconds": 5}})
    assert value.monitoring.sample_seconds == 2
    assert value.monitoring.stale_after_seconds == 5
    assert config.Config.from_toml(value.to_toml()).monitoring == value.monitoring


def test_large_representable_integers_roundtrip(make_config):
    large = 10**1000
    value = make_config({"limits": {"media_bytes": large}, "monitoring": {"sample_seconds": large}})
    assert value.limits.media_bytes == large
    assert value.monitoring.sample_seconds == large
    assert config.Config.from_toml(value.to_toml()).to_dict() == value.to_dict()


@pytest.mark.parametrize("section,key", [
    ("limits", "media_bytes"), ("limits", "reserve_floor_bytes"),
    ("monitoring", "sample_seconds"), ("monitoring", "stale_after_seconds"),
])
def test_unserializable_integer_rejected_before_writer(make_config, tmp_path, section, key):
    huge = 10**5000
    with pytest.raises(config.ConfigError) as error:
        make_config({section: {key: huge}})
    assert error.value.reason_code is ReasonCode.INVALID_CONFIG

    value = make_config()
    altered = replace(value, **{section: replace(getattr(value, section), **{key: huge})})
    writer = _TinyAuditedWriter()
    with pytest.raises(config.ConfigError) as error:
        altered.persist(tmp_path / "config.toml", writer=writer)
    assert error.value.reason_code is ReasonCode.INVALID_CONFIG
    assert writer.events == []


def test_serializer_preserves_schema_reason(make_config):
    invalid = replace(make_config(), schema_version=2)
    with pytest.raises(config.ConfigError) as error:
        invalid.to_toml()
    assert error.value.reason_code is ReasonCode.UNSUPPORTED_SCHEMA


def test_toml_parser_resource_errors_are_private():
    for document in (
        "[limits]\nmedia_bytes = " + "9" * 5000,
        "[limits]\nmedia_bytes = " + "[" * 600 + "1" + "]" * 600,
    ):
        with pytest.raises(config.ConfigError) as error:
            config.Config.from_toml(document)
        assert error.value.reason_code is ReasonCode.INVALID_CONFIG


def test_nonnegative_reserve_fraction_has_no_arbitrary_one_cap(make_config):
    assert make_config({"limits": {"reserve_fraction": 1.5}}).limits.reserve_fraction == 1.5


def test_del_and_unicode_roundtrip(make_config):
    value = make_config({"telemetry": {"otlp_endpoint": "label-😀-\x7f"}})
    encoded = value.to_toml()
    assert "\\u007f" in encoded
    assert config.Config.from_toml(encoded).to_dict() == value.to_dict()


def test_toml_errors_are_private(make_config):
    for document in ("[telemetry\notlp_endpoint = 'secret'\n", "[x]\nsecret = 'value'\n"):
        with pytest.raises(config.ConfigError) as error:
            config.Config.from_toml(document)
        assert error.value.reason_code is ReasonCode.INVALID_CONFIG
        assert "secret" not in str(error.value)


class _TinyAuditedWriter:
    """Test-only protected writer; production must inject its own audited writer."""

    def __init__(self, fail=None):
        self.events = []
        self.fail = fail

    def admit(self, target):
        self.events.append("admit")
        assert target.parent.is_dir()
        if self.fail == "admit":
            raise OSError("private admission detail")
        if self.fail == "admit-domain":
            raise DomainValidationError(ReasonCode.UNAUTHORIZED)

    def audit(self, target, payload):
        self.events.append("audit")
        if self.fail == "audit":
            raise OSError("private audit detail")

    def atomic_replace(self, target, payload):
        self.events.append("replace")
        staging = target.with_name(target.name + ".staging")
        staging.write_bytes(payload)
        try:
            if self.fail == "replace":
                raise OSError("private replace detail")
            os.replace(staging, target)
        finally:
            staging.unlink(missing_ok=True)


@pytest.mark.parametrize("fail,events", [
    ("admit", ["admit"]), ("admit-domain", ["admit"]),
    ("audit", ["admit", "audit"]), ("replace", ["admit", "audit", "replace"]),
])
def test_writer_failure_preserves_old_file(make_config, tmp_path, fail, events):
    target = tmp_path / "config.toml"
    target.write_bytes(b"old protected config")
    writer = _TinyAuditedWriter(fail)
    with pytest.raises(config.ConfigError) as error:
        make_config().persist(target, writer=writer)
    expected = {"admit": ReasonCode.STATE_WRITE_FAILED, "admit-domain": ReasonCode.UNAUTHORIZED, "audit": ReasonCode.AUDIT_LOSS, "replace": ReasonCode.STATE_WRITE_FAILED}
    assert error.value.reason_code is expected[fail]
    assert "private" not in str(error.value)
    assert writer.events == events
    assert target.read_bytes() == b"old protected config"


def test_writer_required_and_successful_commit(make_config, tmp_path):
    value = make_config()
    target = tmp_path / "config.toml"
    with pytest.raises(config.ConfigError):
        value.persist(target, writer=None)
    assert not target.exists()
    writer = _TinyAuditedWriter()
    value.persist(target, writer=writer)
    assert writer.events == ["admit", "audit", "replace"]
    assert config.Config.from_toml(target.read_bytes()).to_dict() == value.to_dict()


def test_invalid_constructed_and_replaced_configs_never_call_writer(make_config, tmp_path):
    value = make_config()
    invalid = (
        replace(value, limits=replace(value.limits, reserve_fraction=-1)),
        replace(value, telemetry=replace(value.telemetry, otlp_endpoint="\ud800")),
        replace(value, schema_version=2),
        config.Config(1, value.paths, replace(value.selection, model="unverified"), value.outputs, value.limits, value.monitoring, value.terminal, value.telemetry, value.tls, value.origins),
    )
    for candidate in invalid:
        writer = _TinyAuditedWriter()
        with pytest.raises(config.ConfigError):
            candidate.persist(tmp_path / "config.toml", writer=writer)
        assert writer.events == []


def test_registry_snapshot_does_not_trust_replacement(make_config, tmp_path):
    models = {"verified-model"}
    value = make_config({"selection": {"model": "verified-model"}}, verified_models=models)
    models.add("late-model")
    writer = _TinyAuditedWriter()
    value.persist(tmp_path / "config.toml", writer=writer)
    assert writer.events == ["admit", "audit", "replace"]
    writer = _TinyAuditedWriter()
    with pytest.raises(config.ConfigError):
        replace(value, selection=replace(value.selection, model="late-model")).persist(tmp_path / "other.toml", writer=writer)
    assert writer.events == []


def test_target_requires_string_or_path(make_config):
    class Stringable:
        def __str__(self):
            return "config.toml"

    writer = _TinyAuditedWriter()
    with pytest.raises(config.ConfigError):
        make_config().persist(Stringable(), writer=writer)
    assert writer.events == []
