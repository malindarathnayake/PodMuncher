"""Registry conformance and adversarial tests for public domain validators."""

import json
from pathlib import Path

import pytest

from podmuncher import domains


REGISTRY = Path(__file__).resolve().parents[1] / "Docs" / "contract-domains.json"


def load_contract_domains():
    """Read the independent registry for conformance tests only."""
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def _enum_name(key):
    return "".join(part.capitalize() for part in key.split("_"))


@pytest.mark.parametrize(
    "key,accepted",
    [(key, accepted) for key, accepted in load_contract_domains().items()
     if key not in {"schema_version", "negative_examples"}],
)
def test_every_registry_domain_is_closed_and_strict(key, accepted):
    enum = getattr(domains, _enum_name(key))
    assert [member.value for member in enum] == accepted
    for value in accepted:
        member = enum.parse(value)
        assert member.value == value
        assert enum.parse(member) is member
    for value in load_contract_domains()["negative_examples"]:
        with pytest.raises(domains.DomainValidationError) as error:
            enum.parse(value)
        assert error.value.reason_code is domains.ReasonCode.INVALID_INPUT
    with pytest.raises(domains.DomainValidationError):
        enum.parse(None)
    with pytest.raises(domains.DomainValidationError):
        enum.parse([accepted[0]])


@pytest.mark.parametrize("key", ["device_slot", "volume_slot"])
@pytest.mark.parametrize("value", [-1, 99, True, False, 0.0, "0"])
def test_numeric_slots_do_not_coerce(key, value):
    enum = getattr(domains, _enum_name(key))
    with pytest.raises(domains.DomainValidationError):
        enum.parse(value)


def test_object_keys_and_nested_json_values():
    payload = {"outer": {"inner": [None, True, 17, 1.5, "ok"]}}
    assert domains.validate_object_keys(payload, allowed={"outer"}, required={"outer"}) is payload
    assert domains.validate_json_payload(payload) is payload
    for bad in ({"unknown": 1}, {}, {1: "bad"}, [], None):
        with pytest.raises(domains.DomainValidationError):
            domains.validate_object_keys(bad, allowed={"outer"}, required={"outer"})
    for bad in (
        {"a": {1: "bad"}},
        {"a": {"b": object()}},
        {"a": float("nan")},
        [float("inf")],
        [float("-inf")],
        {"a": "\ud800"},
    ):
        with pytest.raises(domains.DomainValidationError):
            domains.validate_json_payload(bad)


def test_json_payload_limits_cycles_and_shared_references():
    nested = {}
    for _ in range(7):
        nested = [nested]
    assert domains.validate_json_payload(nested) is nested
    with pytest.raises(domains.DomainValidationError) as error:
        domains.validate_json_payload([nested])
    assert error.value.reason_code is domains.ReasonCode.NESTING_LIMIT
    assert domains.validate_json_payload([nested], max_depth=9) is not None
    cycle = []
    cycle.append(cycle)
    with pytest.raises(domains.DomainValidationError):
        domains.validate_json_payload(cycle)
    shared = ["value"]
    assert domains.validate_json_payload([shared, shared]) is not None
    assert domains.validate_json_payload({"x": "é" * 4096}) is not None
    with pytest.raises(domains.DomainValidationError):
        domains.validate_json_payload({"x": "é" * 4097})
    assert domains.validate_json_payload(["é" * 4097], max_string_bytes=8194) is not None


@pytest.mark.parametrize("bad", ["", "abc", "../x", "{12345678-1234-1234-1234-123456789abc}", 1])
def test_uuid_rejects_non_uuid_identifiers(bad):
    with pytest.raises(domains.DomainValidationError):
        domains.validate_uuid_string(bad)


def test_uuid_accepts_hyphenated_uuid():
    value = "12345678-1234-4abc-8def-123456789abc"
    assert domains.validate_uuid_string(value) == value


@pytest.mark.parametrize("bad", [-1, True, 1.0, "1", None])
def test_nonnegative_int_rejects_invalid_types_and_values(bad):
    with pytest.raises(domains.DomainValidationError):
        domains.validate_nonnegative_int(bad)


def test_numeric_boundaries_and_finite_values():
    assert domains.validate_nonnegative_int(0) == 0
    assert domains.validate_finite_number(2) == 2
    assert domains.validate_finite_number(10 ** 1000) == 10 ** 1000
    assert domains.validate_percentage(0) == 0
    assert domains.validate_percentage(100.0) == 100.0
    for bad in (True, "1", None, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(domains.DomainValidationError):
            domains.validate_finite_number(bad)
        with pytest.raises(domains.DomainValidationError):
            domains.validate_percentage(bad)
    for bad in (-0.01, 100.01):
        with pytest.raises(domains.DomainValidationError):
            domains.validate_percentage(bad)


def test_utf8_byte_limits_and_surrogates():
    assert domains.validate_utf8_string("é" * 4, max_bytes=8) == "é" * 4
    with pytest.raises(domains.DomainValidationError):
        domains.validate_utf8_string("é" * 5, max_bytes=8)
    with pytest.raises(domains.DomainValidationError):
        domains.validate_utf8_string("\ud800")
    with pytest.raises(domains.DomainValidationError):
        domains.validate_utf8_string(b"bytes")


def test_schema_version_and_utc_rfc3339():
    assert domains.validate_schema_version(1) == 1
    for bad in (0, 2, True, "1", 1.0):
        with pytest.raises(domains.DomainValidationError) as error:
            domains.validate_schema_version(bad)
        assert error.value.reason_code is domains.ReasonCode.UNSUPPORTED_SCHEMA
    for valid in ("2026-10-09T12:30:45Z", "2026-10-09T12:30:45.123+00:00"):
        assert domains.validate_utc_timestamp(valid) == valid
    for bad in (
        "2026-10-09", "2026-10-09T12:30:45", "2026-10-09T12:30:45+01:00",
        "2026-02-30T12:30:45Z", "2026-10-09T24:30:45Z", "",
    ):
        with pytest.raises(domains.DomainValidationError):
            domains.validate_utc_timestamp(bad)


def test_errors_do_not_echo_rejected_values():
    secret = "credential=do-not-display"
    with pytest.raises(domains.DomainValidationError) as error:
        domains.validate_json_payload({secret: object()})
    assert secret not in str(error.value)
    assert len(str(error.value)) <= 80
    assert error.value.reason_code is domains.ReasonCode.INVALID_INPUT
