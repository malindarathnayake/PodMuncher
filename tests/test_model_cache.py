"""Verified model-cache transport, containment, lifecycle, and live smoke tests."""

from __future__ import annotations

import ctypes
import hashlib
import http.client
import http.server
import json
import os
import ssl
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import ClassVar
from urllib.parse import urlsplit, urlunsplit

import pytest

import podmuncher.model_cache as model_cache_module
from podmuncher.domains import ModelTier, ReasonCode
from podmuncher.manifests import ManifestFile, ManifestModel
from podmuncher.model_cache import CacheError, ModelCache
from podmuncher.model_policy import ModelArtifact

CERTIFICATE = """-----BEGIN CERTIFICATE-----
MIIC2jCCAcKgAwIBAgIJAJmmUWZzsyFMMA0GCSqGSIb3DQEBCwUAMBQxEjAQBgNV
BAMTCWxvY2FsaG9zdDAeFw0yNjEwMTAwMzExMjJaFw0zMTEwMTEwMzExMjJaMBQx
EjAQBgNVBAMTCWxvY2FsaG9zdDCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoC
ggEBAKF/+1wSPyhE1wVSCoUFqlAY/HIIC36fmYPVvXbUqi8YX3KADCNTwgrDaXYQ
NSzJtShDPrx6ff1hKyMjYnxuCjCdOBp648CwNkRtZU5rBH4ZbbZ9/Z5xh5b7NAgU
KsvCHHqkwNQ75Au2l+m4oU0YQjqs0YnEwO98x7Oz3ExaJDMk6pLtpys53uclKxgm
bF/NlhIkVAUuyMGBHb66TjenXJyU3NJrwqEy33cAt/0IVf/PKGdwaF9t930IFgJX
DdeozeVM9lVoFCmrTS+0l7KH9lSQwKdTwLVsKEK/lxh5NWL+72pO7ost68QiqLic
bI+Qb1gvwl1RSnGdO5ZjZQMlGE0CAwEAAaMvMC0wGgYDVR0RBBMwEYIJbG9jYWxo
b3N0hwR/AAABMA8GA1UdEwEB/wQFMAMBAf8wDQYJKoZIhvcNAQELBQADggEBAIf4
tLM/VyIjT6r4pCxx76zyFu20oOOWEVP6pGfnRUcZNWQ5Lt7IJ6oD7fjksizP2Rf1
v13FOz7k8QenmmcY0+bwjSPmLcqYOGXE6PXrFZPfuHjdkVsF6sSULeMp9R4b8xJF
lSUFRoquDu+3CDvtLScHW7gPYbNcc/oUjwtL98OnrJgi4K0X1NHxcNrtjG9Dp9gd
T1K5QIjUGL7+DP343cgZxHFeskBFW7dfgU3zcaEe8G9Qx0V+18/kPijOI62FOd6u
TmC3lczhKoREMuWITqD5t6a0UAUObCro6Dgqb+jC2QUzsDd60io/AGNZr/imTAaW
/vsOTdlAgE/4MIRMrWE=
-----END CERTIFICATE-----
"""

PRIVATE_KEY = """-----BEGIN PRIVATE KEY-----
MIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQChf/tcEj8oRNcF
UgqFBapQGPxyCAt+n5mD1b121KovGF9ygAwjU8IKw2l2EDUsybUoQz68en39YSsj
I2J8bgownTgaeuPAsDZEbWVOawR+GW22ff2ecYeW+zQIFCrLwhx6pMDUO+QLtpfp
uKFNGEI6rNGJxMDvfMezs9xMWiQzJOqS7acrOd7nJSsYJmxfzZYSJFQFLsjBgR2+
uk43p1yclNzSa8KhMt93ALf9CFX/zyhncGhfbfd9CBYCVw3XqM3lTPZVaBQpq00v
tJeyh/ZUkMCnU8C1bChCv5cYeTVi/u9qTu6LLevEIqi4nGyPkG9YL8JdUUpxnTuW
Y2UDJRhNAgMBAAECggEBAI+p5LVbufkde9VMzPTWPFIyV+cshuUmHo7rRTZ/KcDp
Qob9p06JxLSEbr1K37Ku4Esnvpg1svwCQzhn0sVzsfba9LyBkzamyi+GzTa79mwi
PLm5TDJ1NbO2VOvW3NcTHvhutOZXPJ662FpeFocZTFkbwO54K1UfQBr4pcD58vfW
tjfwht/6bzXsUlMnHKQJFFZY3Eudq2fR8EPxPho0smnTJ3IdCYuLV//Gm4xwK/EW
wmMZBYJVU6Roec33tCPN/qPf6/67xlAklFFGsy9Ql2AvBfwttsRez3TvQBwFZkwm
UiMstuTMpuAIIaHiHfSo5BqFzO009a4IAhSdkUsxDtUCgYEAxYIZVszYKiRkp5l3
h/8i/o6e30y/hUQ46pN3LC0ywX7iw0IW6HI5T2ifJgZJobBLuMn/wpIv5W34ZTK2
iUiZWWQersJV3O9DXvmpjsZysHEM7UlBkoFblGjjxySv9fIzQcldmGgAfx2XvgX7
KMsyZXbNl4Br+rDtHzadxlXf+EsCgYEA0VPzBIkcUzNsHnKohF1E0z6pqH+1txob
NA1Y8rKkXIWtACNIpFhqKInd1qVNq4A0MOxP6u89ULwmv2jhagr0zH/jJuEkfLST
wtF+jQwxNN1+L8mIX4HEoz0eHBtewslU74vqBCA8w5+Iyd6SxkyuwyMWuJQgQkfZ
XYFhBxK/gscCgYAhbMIDCYmvoBcUsgkjj88JS3OhkKh80i3brPcnF8MWMfYIW1/r
GKGQScFdOwUtAtvduvm4oOrQJgs6fRmdJBcnpKT4qFTFngxFv3Stf/Ia4Bby9UnO
lGu+Mv4RTqkks0dvY7eeOkOTP5vVgceegLzXpQJoGEit1VFbze9niFLt/wKBgB7X
d57bdMjtcqipzBp0g4d8XBLvfh6UQh3ROaHIoF7hczPp/vH39Ow0I/NUN7VlFkt/
BedfiHE0x6C8Gzo8kwTuctFvqqiVEpBJvZOakEVxvP3de7JWtFo4zw7lGXkpoNIK
k1g4R7uq7cSQfp+o/LbTnK1lZbteSrB0N4+LO5xRAoGBAMQYWwn1JN/+onM0nIFI
18wVKoMqGQhdF/pQ7Co9MLNWxqvTlT0hyoG3jvMzGJQFW2NtQvEvWPhQVtlDm2yT
fvoF5OWiAqtxnsK9Q3Ww4O30uayHlBTtLfpPxo2XaLm5NTZlaZrWv7yuglUArunO
S25bOO3ptueiGIFWaiRd61oj
-----END PRIVATE KEY-----
"""


class Audit:
    def __init__(self):
        self.events = []
        self.failure = False

    def record(self, *, action, reason, asset_id, failure_code):
        if self.failure:
            raise RuntimeError("private audit failure")
        self.events.append((action, reason, asset_id, failure_code))


class Spans:
    def __init__(self):
        self.names = []

    @contextmanager
    def span(self, name, *, asset_id):
        self.names.append((name, asset_id))
        yield


class FixtureHandler(http.server.BaseHTTPRequestHandler):
    payload = b""
    mode = "normal"
    requests: ClassVar[list[tuple[str, dict[str, str]]]] = []

    def _redirect(self, status, *locations):
        self.send_response(status)
        for location in locations:
            self.send_header("Location", location)
        self.send_header("Set-Cookie", "session=fixture-cookie; Path=/")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        FixtureHandler.requests.append(
            (self.path, {key.lower(): value for key, value in self.headers.items()})
        )
        port = self.server.server_port
        if self.path == "/redirect":
            self._redirect(302, "/asset.bin")
            return
        if self.path == "/relative-307/start":
            self._redirect(307, "../asset.bin")
            return
        if self.path == "/absolute-302":
            self._redirect(302, f"https://127.0.0.1:{port}/asset.bin")
            return
        if self.path.startswith("/chain/"):
            remaining = int(self.path.rsplit("/", 1)[1])
            if remaining:
                self._redirect(301, f"/chain/{remaining - 1}")
                return
        if self.path == "/to-http":
            self._redirect(308, f"http://localhost:{port}/asset.bin")
            return
        if self.path == "/userinfo":
            self._redirect(303, f"https://user:secret@localhost:{port}/asset.bin")
            return
        if self.path == "/fragment":
            self._redirect(302, "/asset.bin#part")
            return
        if self.path == "/missing-location":
            self._redirect(302)
            return
        if self.path == "/double-location":
            self._redirect(302, "/asset.bin", "/asset.bin")
            return
        if self.path == "/private-hop":
            self._redirect(302, f"https://10.0.0.1:{port}/asset.bin")
            return
        body = self.payload if self.mode != "corrupt" else self.payload + b"x"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        cutoff = len(body) // 2 if self.mode == "interrupt" else len(body)
        self.wfile.write(body[:cutoff])
        self.wfile.flush()
        if self.mode == "interrupt":
            self.connection.shutdown(2)

    def log_message(self, _format, *_args):
        return


_STRICT_PUBLIC_PEER = model_cache_module._public_peer


@pytest.fixture
def unseamed_https_fixture(tmp_path):
    certificate = tmp_path / "fixture-ca.pem"
    private_key = tmp_path / "fixture-key.pem"
    certificate.write_text(CERTIFICATE, encoding="ascii")
    private_key.write_text(PRIVATE_KEY, encoding="ascii")
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FixtureHandler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certificate, private_key)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    server_started = threading.Event()

    def serve():
        server_started.set()
        server.serve_forever()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    try:
        if not server_started.wait(timeout=2):
            pytest.fail("local HTTPS fixture server thread did not start")
        if not thread.is_alive():
            pytest.fail("local HTTPS fixture server thread exited during startup")
        FixtureHandler.payload = b""
        FixtureHandler.mode = "normal"
        FixtureHandler.requests = []
        yield f"https://localhost:{server.server_port}", certificate
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
        if thread.is_alive():
            pytest.fail("local HTTPS fixture server thread did not stop")


def allow_fixture_loopback(address):
    """Test seam: admit only the fixture's IPv4 loopback beyond the strict gate."""
    return address == "127.0.0.1" or _STRICT_PUBLIC_PEER(address)


@pytest.fixture
def loopback_peer_seam(monkeypatch):
    monkeypatch.setattr(model_cache_module, "_public_peer", allow_fixture_loopback)


@pytest.fixture
def local_https_fixture(unseamed_https_fixture, loopback_peer_seam):
    return unseamed_https_fixture


def model(url, payload, *, relative_path="model.bin", model_id="fixture-model"):
    return ManifestModel(
        ModelArtifact(model_id, ModelTier.TINY, "revision-1", "1" * 64),
        (url,),
        (ManifestFile(relative_path, len(payload), hashlib.sha256(payload).hexdigest()),),
        "MIT",
        "verified-test-provenance",
        ("int8",),
        1,
        0,
        1,
        1,
        1,
        ("wav",),
        "2" * 64,
        True,
        False,
    )


def ordered_files_model(url, payload, *, model_id):
    template = model(url, payload, model_id=model_id)
    digest = hashlib.sha256(payload).hexdigest()
    return replace(
        template,
        source_urls=(url, url),
        files=(
            ManifestFile("a.bin", len(payload), digest),
            ManifestFile("b.bin", len(payload), digest),
        ),
    )


def make_cache(tmp_path, *, ca_bundle=None, audit=None, admission=None, max_entries=8):
    return ModelCache(
        tmp_path / "cache",
        audit=audit or Audit(),
        spans=Spans(),
        admission=admission or (lambda **_values: True),
        ca_bundle=ca_bundle,
        max_entries=max_entries,
        chunk_size=4,
        timeout_seconds=30,
    )


def assert_remove_tombstones_reconciled(root):
    # A reconciled removal leaves nothing behind on any platform: the POSIX
    # private-name tombstones are unlinked once their identity is verified.
    names = {item.name for item in (root / "quarantine").iterdir()}
    assert names == set()
    return names


def retained_tombstones(root):
    return sorted(
        str(item.relative_to(root))
        for item in root.rglob(".retained-*")
    )


def test_local_https_fixture_transport_auth_and_stream_shape(tmp_path, local_https_fixture):
    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"transport-shape"
    FixtureHandler.mode = "normal"
    cache = make_cache(tmp_path, ca_bundle=ca_bundle)
    snapshot = cache.acquire(model(base_url + "/asset.bin", FixtureHandler.payload))
    assert snapshot.path.joinpath("model.bin").read_bytes() == FixtureHandler.payload
    assert snapshot.total_bytes == len(FixtureHandler.payload)


def test_atomic_verified_snapshot_metadata_and_exact_cleanup(tmp_path, local_https_fixture):
    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"complete-model"
    FixtureHandler.mode = "normal"
    audit, spans = Audit(), Spans()
    cache = ModelCache(tmp_path / "cache", audit=audit, spans=spans,
                       admission=lambda **_values: True, ca_bundle=ca_bundle, chunk_size=3)
    snapshot = cache.acquire(model(base_url + "/asset.bin", FixtureHandler.payload))
    assert cache.snapshots() == (snapshot,)
    assert not tuple((tmp_path / "cache" / "staging").iterdir())
    metadata = json.loads((tmp_path / "cache" / "metadata.json").read_text())
    assert metadata["snapshots"][0]["id"] == snapshot.snapshot_id
    assert {event[0] for event in audit.events} >= {"stage", "publish", "metadata"}
    assert spans.names == [("acquire_asset", snapshot.snapshot_id)]
    cache.remove(snapshot.snapshot_id)
    assert not snapshot.path.exists() and cache.snapshots() == ()


@pytest.mark.parametrize(("mode", "reason"), [
    ("corrupt", ReasonCode.ASSET_VERIFICATION_FAILED),
    ("interrupt", ReasonCode.UPSTREAM_UNAVAILABLE),
])
def test_corruption_and_interruption_are_quarantined_without_resume(
        tmp_path, local_https_fixture, mode, reason):
    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"expected-content"
    FixtureHandler.mode = mode
    audit = Audit()
    cache = make_cache(tmp_path, ca_bundle=ca_bundle, audit=audit)
    with pytest.raises(CacheError) as error:
        cache.acquire(model(base_url + "/asset.bin", FixtureHandler.payload))
    assert error.value.reason_code is reason
    assert not tuple((tmp_path / "cache" / "staging").iterdir())
    assert len(tuple((tmp_path / "cache" / "quarantine").iterdir())) == 1
    quarantine_events = [event for event in audit.events if event[0] == "quarantine"]
    assert len(quarantine_events) == 1
    assert quarantine_events[0][1] == "failed_or_unrecognized_partial"
    assert quarantine_events[0][3] == reason.value
    assert cache.audit_loss_events == ()
    FixtureHandler.mode = "normal"
    assert cache.acquire(model(base_url + "/asset.bin", FixtureHandler.payload))


def test_failure_path_quarantine_audit_loss_keeps_original_reason(
        tmp_path, local_https_fixture):
    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"expected-content"
    FixtureHandler.mode = "corrupt"
    audit = ActionAudit()
    audit.fail_action = "quarantine"
    cache = make_cache(tmp_path, ca_bundle=ca_bundle, audit=audit)
    with pytest.raises(CacheError) as error:
        cache.acquire(model(base_url + "/asset.bin", FixtureHandler.payload))
    assert error.value.reason_code is ReasonCode.ASSET_VERIFICATION_FAILED
    assert not tuple((tmp_path / "cache" / "staging").iterdir())
    assert len(tuple((tmp_path / "cache" / "quarantine").iterdir())) == 1
    assert not any(event[0] == "quarantine" for event in audit.events)
    assert len(cache.audit_loss_events) == 1
    assert cache.audit_loss_events[0][0] is ReasonCode.AUDIT_LOSS


def test_recovery_quarantines_audit_failure_codes_and_bookkeep_audit_loss(tmp_path):
    root = tmp_path / "cache"
    make_cache(tmp_path).close()
    (root / "staging" / "left-over.partial").mkdir()
    (root / "snapshots" / "orphan-snapshot").mkdir()

    audit = Audit()
    cache = make_cache(tmp_path, audit=audit)
    cache.close()
    quarantine_events = {
        (event[2], event[3]) for event in audit.events if event[0] == "quarantine"
    }
    assert quarantine_events == {
        ("unknown-partial", ReasonCode.ASSET_VERIFICATION_FAILED.value),
        ("unrecognized", ReasonCode.STATE_WRITE_FAILED.value),
    }
    assert all(
        event[1] == "failed_or_unrecognized_partial"
        for event in audit.events
        if event[0] == "quarantine"
    )
    assert cache.audit_loss_events == ()
    assert not tuple((root / "staging").iterdir())
    assert not tuple((root / "snapshots").iterdir())
    quarantined = len(tuple((root / "quarantine").iterdir()))
    assert quarantined == 2

    # Recovery quarantines of unknown or orphan entries are not exempt from
    # the audit-failure latch: a sink that loses the quarantine record leaves
    # the entries exactly where they are and initialization fails closed.
    (root / "staging" / "second.partial").mkdir()
    (root / "snapshots" / "second-orphan").mkdir()
    audit = ActionAudit()
    audit.fail_action = "quarantine"
    with pytest.raises(CacheError) as error:
        make_cache(tmp_path, audit=audit)
    assert error.value.reason_code is ReasonCode.AUDIT_LOSS
    assert not any(event[0] == "quarantine" for event in audit.events)
    assert (root / "staging" / "second.partial").is_dir()
    assert (root / "snapshots" / "second-orphan").is_dir()
    assert len(tuple((root / "quarantine").iterdir())) == quarantined

    # A healthy sink on the next open quarantines both left-over entries.
    audit = Audit()
    make_cache(tmp_path, audit=audit).close()
    assert {
        (event[2], event[3]) for event in audit.events if event[0] == "quarantine"
    } == {
        ("unknown-partial", ReasonCode.ASSET_VERIFICATION_FAILED.value),
        ("unrecognized", ReasonCode.STATE_WRITE_FAILED.value),
    }
    assert not tuple((root / "staging").iterdir())
    assert not tuple((root / "snapshots").iterdir())
    assert len(tuple((root / "quarantine").iterdir())) == quarantined + 2


def test_exempt_audit_loss_removal_defers_unrelated_recovery_quarantines(
        tmp_path, local_https_fixture):
    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"exempt-removal"
    FixtureHandler.mode = "normal"
    audit = ActionAudit()
    cache = make_cache(tmp_path, ca_bundle=ca_bundle, audit=audit)
    snapshot = cache.acquire(model(base_url + "/asset.bin", FixtureHandler.payload))
    root = tmp_path / "cache"
    (root / "staging" / "left-over.partial").mkdir()
    (root / "snapshots" / "orphan-snapshot").mkdir()
    quarantined = {item.name for item in (root / "quarantine").iterdir()}

    # A non-exempt removal stays fail-closed on the unrelated leftovers.
    audit.fail_action = "quarantine"
    with pytest.raises(CacheError) as error:
        cache.remove(snapshot.snapshot_id)
    assert error.value.reason_code is ReasonCode.AUDIT_LOSS
    assert snapshot.path.exists()
    assert (root / "staging" / "left-over.partial").is_dir()
    assert (root / "snapshots" / "orphan-snapshot").is_dir()

    # The exempt known-unused removal (spec 1194) proceeds; the unrelated
    # unknown partial and orphan are left in place, not moved unaudited.
    cache.remove(snapshot.snapshot_id, audit_loss_recovery=True)
    assert cache.snapshots() == ()
    assert not snapshot.path.exists()
    assert (root / "staging" / "left-over.partial").is_dir()
    assert (root / "snapshots" / "orphan-snapshot").is_dir()
    assert not any(event[0] == "quarantine" for event in audit.events)
    # The latched sink loses the remove record; the exemption bookkeeps it.
    assert (ReasonCode.AUDIT_LOSS, snapshot.snapshot_id) in cache.audit_loss_events
    assert {item.name for item in (root / "quarantine").iterdir()} == quarantined
    cache.close()

    # A later open with a healthy sink quarantines both deferred entries.
    audit = Audit()
    restarted = make_cache(tmp_path, ca_bundle=ca_bundle, audit=audit)
    assert restarted.snapshots() == ()
    assert {
        (event[2], event[3]) for event in audit.events if event[0] == "quarantine"
    } == {
        ("unknown-partial", ReasonCode.ASSET_VERIFICATION_FAILED.value),
        ("unrecognized", ReasonCode.STATE_WRITE_FAILED.value),
    }
    assert not tuple((root / "staging").iterdir())
    assert not tuple((root / "snapshots").iterdir())
    assert len(tuple((root / "quarantine").iterdir())) == len(quarantined) + 2
    restarted.close()


def test_orphan_recovery_audits_only_well_formed_snapshot_ids(tmp_path):
    root = tmp_path / "cache"
    make_cache(tmp_path).close()
    valid_id = "snapshot-" + "ab" * 16
    arbitrary = "arbitrary.orphan-name_v2"
    (root / "snapshots" / valid_id).mkdir()
    (root / "snapshots" / arbitrary).mkdir()
    (root / "snapshots" / "snapshot-notahexid").mkdir()

    audit = Audit()
    cache = make_cache(tmp_path, audit=audit)
    cache.close()
    quarantine_events = [
        (event[2], event[3]) for event in audit.events if event[0] == "quarantine"
    ]
    assert sorted(quarantine_events) == sorted([
        (valid_id, ReasonCode.STATE_WRITE_FAILED.value),
        ("unrecognized", ReasonCode.STATE_WRITE_FAILED.value),
        ("unrecognized", ReasonCode.STATE_WRITE_FAILED.value),
    ])
    assert not any(
        arbitrary in event[2] or "notahexid" in event[2] for event in audit.events
    )
    assert cache.audit_loss_events == ()
    assert not tuple((root / "snapshots").iterdir())
    assert len(tuple((root / "quarantine").iterdir())) == 3


def test_source_redirect_extra_file_and_unsafe_paths_are_rejected(tmp_path, local_https_fixture):
    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"content"
    FixtureHandler.mode = "normal"
    cache = make_cache(tmp_path, ca_bundle=ca_bundle)
    redirected = cache.acquire(model(base_url + "/redirect", FixtureHandler.payload))
    assert redirected.path.joinpath("model.bin").read_bytes() == FixtureHandler.payload
    cache.remove(redirected.snapshot_id)
    for unsafe in ("../escape.bin", "/absolute.bin", r"..\escape.bin"):
        with pytest.raises(CacheError):
            cache.acquire(model(base_url + "/asset.bin", FixtureHandler.payload,
                                relative_path=unsafe))
    unknown = tmp_path / "cache" / "staging" / "unknown"
    unknown.mkdir()
    (unknown / "extra.bin").write_bytes(b"x")
    cache.recover()
    assert not unknown.exists()
    assert tuple((tmp_path / "cache" / "quarantine").iterdir())


@pytest.mark.parametrize("address", [
    "8.8.8.8",
    "1.1.1.1",
    "2606:4700:4700::1111",
    "2001:4860:4860::8888",
])
def test_public_peer_admits_global_addresses(address):
    assert model_cache_module._public_peer(address) is True


@pytest.mark.parametrize("address", [
    "10.0.0.1",
    "172.16.0.1",
    "172.31.255.254",
    "192.168.1.1",
    "127.0.0.1",
    "127.10.0.1",
    "169.254.169.254",
    "0.0.0.0",
    "224.0.0.1",
    "239.255.255.250",
    "240.0.0.1",
    "100.64.0.1",
    "::",
    "::1",
    "fe80::1",
    "fe80::1%1",
    "fc00::1",
    "fd12:3456::1",
    "ff02::1",
    "::ffff:10.0.0.1",
    "::ffff:8.8.8.8",
    "::10.0.0.1",
    "2002:0a00:0001::1",
    "2001:0000:4136:e378:8000:63bf:f5ff:fffe",
    "64:ff9b::a00:1",
    "localhost",
    "",
    None,
])
def test_public_peer_rejects_non_global_and_embedded_private_addresses(address):
    assert model_cache_module._public_peer(address) is False


def test_strict_gate_rejects_loopback_fixture_without_seam(
        tmp_path, unseamed_https_fixture):
    base_url, ca_bundle = unseamed_https_fixture
    FixtureHandler.payload = b"loopback"
    cache = make_cache(tmp_path, ca_bundle=ca_bundle)
    with pytest.raises(CacheError) as error:
        cache.acquire(model(base_url + "/asset.bin", FixtureHandler.payload))
    assert error.value.reason_code is ReasonCode.ASSET_SOURCE_REJECTED
    assert error.value.reason_code.value == "asset_source_rejected"
    assert FixtureHandler.requests == []
    assert not tuple((tmp_path / "cache" / "staging").iterdir())


def test_resolution_filters_non_public_addresses(tmp_path, loopback_peer_seam):
    cache = make_cache(tmp_path)
    resolved = cache._resolve_addresses("localhost", 443, cache.monotonic() + 10)
    assert resolved and {item[4][0] for item in resolved} == {"127.0.0.1"}
    with pytest.raises(CacheError) as error:
        cache._resolve_addresses("10.0.0.1", 443, cache.monotonic() + 10)
    assert error.value.reason_code is ReasonCode.ASSET_SOURCE_REJECTED


@pytest.mark.parametrize("peer", [("127.0.0.2", 443), ("10.0.0.1", 443), (), None])
def test_post_connect_peer_mismatch_or_private_peer_is_rejected(
        tmp_path, monkeypatch, loopback_peer_seam, peer):
    class Stream:
        closed = False

        def settimeout(self, _timeout):
            return None

        def connect(self, _address):
            return None

        def getpeername(self):
            return peer

        def close(self):
            self.closed = True

    stream = Stream()
    monkeypatch.setattr(
        model_cache_module.socket, "socket", lambda *_args, **_kwargs: stream
    )
    cache = make_cache(tmp_path)
    with pytest.raises(CacheError) as error:
        cache._resolved_connection(
            [(2, 1, 6, "", ("127.0.0.1", 443))], cache.monotonic() + 1, None
        )
    assert error.value.reason_code is ReasonCode.ASSET_SOURCE_REJECTED
    assert stream.closed


@pytest.fixture
def owner_spy(monkeypatch):
    owners = []

    class SpyOwner(model_cache_module._HTTPResponseOwner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            owners.append(self)

    monkeypatch.setattr(model_cache_module, "_HTTPResponseOwner", SpyOwner)
    return owners


def assert_no_credentials_forwarded():
    assert FixtureHandler.requests
    for _path, headers in FixtureHandler.requests:
        assert "authorization" not in headers
        assert "cookie" not in headers


def test_relative_307_on_same_host_is_followed_and_hash_verified(
        tmp_path, local_https_fixture, owner_spy):
    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"relative-redirect-payload"
    manifest = model(base_url + "/relative-307/start", FixtureHandler.payload)
    cache = make_cache(tmp_path, ca_bundle=ca_bundle)
    snapshot = cache.acquire(manifest)
    assert snapshot.path.joinpath("model.bin").read_bytes() == FixtureHandler.payload
    assert [path for path, _headers in FixtureHandler.requests] == [
        "/relative-307/start", "/asset.bin",
    ]
    assert_no_credentials_forwarded()
    assert len(owner_spy) == 2
    assert all(owner._closed for owner in owner_spy)


def test_absolute_302_to_second_host_name_records_manifest_url_only(
        tmp_path, local_https_fixture, owner_spy):
    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"absolute-redirect-payload"
    manifest_url = base_url + "/absolute-302"
    manifest = model(manifest_url, FixtureHandler.payload)
    audit = Audit()
    cache = make_cache(tmp_path, ca_bundle=ca_bundle, audit=audit)
    snapshot = cache.acquire(manifest)
    assert snapshot.path.joinpath("model.bin").read_bytes() == FixtureHandler.payload
    assert [path for path, _headers in FixtureHandler.requests] == [
        "/absolute-302", "/asset.bin",
    ]
    assert FixtureHandler.requests[1][1]["host"].startswith("127.0.0.1:")
    assert_no_credentials_forwarded()
    assert all(owner._closed for owner in owner_spy) and len(owner_spy) == 2
    expected_id, _digest = model_cache_module._snapshot_identity(manifest)
    redirected_id, _digest = model_cache_module._snapshot_identity(
        replace(manifest, source_urls=(
            manifest_url.replace("localhost", "127.0.0.1").replace(
                "/absolute-302", "/asset.bin"),
        ))
    )
    assert snapshot.snapshot_id == expected_id != redirected_id
    metadata = (tmp_path / "cache" / "metadata.json").read_text()
    assert "127.0.0.1" not in metadata and "127.0.0.1" not in repr(audit.events)


def test_five_redirects_are_followed_and_sixth_is_rejected(
        tmp_path, local_https_fixture, owner_spy):
    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"chain-payload"
    cache = make_cache(tmp_path, ca_bundle=ca_bundle)
    snapshot = cache.acquire(model(base_url + "/chain/5", FixtureHandler.payload))
    assert snapshot.path.joinpath("model.bin").read_bytes() == FixtureHandler.payload
    FixtureHandler.requests = []
    with pytest.raises(CacheError) as error:
        cache.acquire(model(base_url + "/chain/6", FixtureHandler.payload,
                            model_id="fixture-chain-six"))
    assert error.value.reason_code is ReasonCode.ASSET_SOURCE_REJECTED
    assert len(FixtureHandler.requests) == 6
    assert all(owner._closed for owner in owner_spy)
    assert not tuple((tmp_path / "cache" / "staging").iterdir())


@pytest.mark.parametrize("path", [
    "/to-http",
    "/userinfo",
    "/fragment",
    "/missing-location",
    "/double-location",
    "/private-hop",
])
def test_unsafe_redirect_targets_are_rejected(
        tmp_path, local_https_fixture, owner_spy, path):
    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"unsafe-redirect"
    cache = make_cache(tmp_path, ca_bundle=ca_bundle)
    with pytest.raises(CacheError) as error:
        cache.acquire(model(base_url + path, FixtureHandler.payload))
    assert error.value.reason_code is ReasonCode.ASSET_SOURCE_REJECTED
    assert [item for item, _headers in FixtureHandler.requests] == [path]
    assert owner_spy and all(owner._closed for owner in owner_spy)
    assert not tuple((tmp_path / "cache" / "staging").iterdir())


@pytest.mark.parametrize(
    "prefix",
    [
        ".metadata.",
        ".metadata-control.",
        ".metadata-old.",
        ".metadata-transaction.",
        ".remove.",
        ".retained-directory.",
        ".retained-file.",
    ],
)
@pytest.mark.parametrize("nested", [False, True], ids=["top-level", "nested"])
def test_internal_cleanup_namespaces_are_rejected_in_every_path_component(
        tmp_path, local_https_fixture, prefix, nested):
    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"reserved-name"
    FixtureHandler.mode = "normal"
    relative_path = (
        f"models/{prefix}private/model.bin"
        if nested
        else f"{prefix}private/model.bin"
    )
    cache = make_cache(tmp_path, ca_bundle=ca_bundle)
    with pytest.raises(CacheError) as error:
        cache.acquire(
            model(
                base_url + "/asset.bin",
                FixtureHandler.payload,
                relative_path=relative_path,
            )
        )
    assert error.value.reason_code is ReasonCode.ASSET_VERIFICATION_FAILED
    assert not tuple((tmp_path / "cache" / "staging").iterdir())


def test_normal_dotted_manifest_components_remain_valid(
        tmp_path, local_https_fixture):
    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"dotted-name"
    FixtureHandler.mode = "normal"
    cache = make_cache(tmp_path, ca_bundle=ca_bundle)
    snapshot = cache.acquire(
        model(
            base_url + "/asset.bin",
            FixtureHandler.payload,
            relative_path=".models/v1.2/model.bin.part",
        )
    )
    assert snapshot.path.joinpath(
        ".models", "v1.2", "model.bin.part"
    ).read_bytes() == FixtureHandler.payload


def test_admission_and_root_or_link_swap_fail_closed(tmp_path, local_https_fixture, monkeypatch):
    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"guarded-content"
    FixtureHandler.mode = "normal"
    calls = []

    def admission(**values):
        calls.append(values["allocated_bytes"])
        return values["allocated_bytes"] < 4

    cache = make_cache(tmp_path, ca_bundle=ca_bundle, admission=admission)
    with pytest.raises(CacheError) as error:
        cache.acquire(model(base_url + "/asset.bin", FixtureHandler.payload))
    assert error.value.reason_code is ReasonCode.STORAGE_GUARD_TRIPPED
    assert calls[0] == 0 and calls[-1] == 4

    cache = make_cache(tmp_path / "swap", ca_bundle=ca_bundle)
    original = cache._verify_root
    count = 0

    def changed_root():
        nonlocal count
        count += 1
        if count > 2:
            raise OSError("private root swap")
        return original()

    monkeypatch.setattr(cache, "_verify_root", changed_root)
    with pytest.raises(CacheError):
        cache.acquire(model(base_url + "/asset.bin", FixtureHandler.payload))


def test_in_use_unknown_and_audit_failure_rules(tmp_path, local_https_fixture):
    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"managed"
    FixtureHandler.mode = "normal"
    audit = Audit()
    cache = make_cache(tmp_path, ca_bundle=ca_bundle, audit=audit)
    snapshot = cache.acquire(model(base_url + "/asset.bin", FixtureHandler.payload))
    cache.mark_in_use(snapshot.snapshot_id)
    with pytest.raises(CacheError) as error:
        cache.remove(snapshot.snapshot_id)
    assert error.value.reason_code is ReasonCode.CACHE_IN_USE
    cache.release(snapshot.snapshot_id)
    with pytest.raises(CacheError) as error:
        cache.remove("not-managed")
    assert error.value.reason_code is ReasonCode.CACHE_UNKNOWN
    audit.failure = True
    with pytest.raises(CacheError) as error:
        cache.acquire(model(base_url + "/other.bin", FixtureHandler.payload))
    assert error.value.reason_code is ReasonCode.AUDIT_LOSS
    cache.remove(snapshot.snapshot_id, audit_loss_recovery=True)
    assert cache.audit_loss_events
    assert all(event[0] is ReasonCode.AUDIT_LOSS for event in cache.audit_loss_events)
    assert snapshot.snapshot_id in {event[1] for event in cache.audit_loss_events}


def test_bounded_metadata_and_contract_domain_negatives(tmp_path, local_https_fixture):
    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"one"
    FixtureHandler.mode = "normal"
    cache = make_cache(tmp_path, ca_bundle=ca_bundle, max_entries=1)
    cache.acquire(model(base_url + "/asset.bin", FixtureHandler.payload))
    FixtureHandler.payload = b"two"
    with pytest.raises(CacheError) as error:
        cache.acquire(model(base_url + "/asset.bin", FixtureHandler.payload))
    assert error.value.reason_code is ReasonCode.STORAGE_INSUFFICIENT
    registry = json.loads(
        (Path(__file__).parents[1] / "Docs" / "contract-domains.json").read_text()
    )
    for invalid in registry["negative_examples"]:
        with pytest.raises(CacheError):
            cache.remove(invalid)


def test_restart_rehashes_persisted_declarations_and_accepts_manifest_ids(
        tmp_path, local_https_fixture):
    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"restart-integrity"
    FixtureHandler.mode = "normal"
    cache = make_cache(tmp_path, ca_bundle=ca_bundle)
    snapshot = cache.acquire(model(
        base_url + "/asset.bin",
        FixtureHandler.payload,
        model_id="Z" + "a" * 127,
    ))
    assert snapshot.snapshot_id.startswith("snapshot-")
    cache.close()
    snapshot.path.joinpath("model.bin").write_bytes(b"x" * len(FixtureHandler.payload))
    with pytest.raises(CacheError) as error:
        make_cache(tmp_path, ca_bundle=ca_bundle)
    assert error.value.reason_code is ReasonCode.ASSET_VERIFICATION_FAILED


def test_metadata_read_is_bounded_before_json_allocation(tmp_path):
    root = tmp_path / "cache"
    for path in (root, root / "snapshots", root / "staging", root / "quarantine"):
        path.mkdir()
    with (root / "metadata.json").open("wb") as stream:
        stream.seek(1024 * 1024)
        stream.write(b"x")
    with pytest.raises(CacheError) as error:
        make_cache(tmp_path)
    assert error.value.reason_code is ReasonCode.CACHE_UNKNOWN


def test_linked_root_is_rejected_before_admission_or_writes(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(real, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")
    calls = []
    with pytest.raises(CacheError):
        ModelCache(
            linked,
            audit=Audit(),
            admission=lambda **values: calls.append(values) or True,
        )
    assert calls == []
    assert not tuple(real.iterdir())


def test_final_guard_detects_mutation_immediately_before_publish(
        tmp_path, local_https_fixture):
    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"publish-mutation"
    FixtureHandler.mode = "normal"
    completed_calls = 0

    def admission(**values):
        nonlocal completed_calls
        if values["allocated_bytes"] == len(FixtureHandler.payload):
            completed_calls += 1
            if completed_calls == 2:
                target = next((tmp_path / "cache" / "staging").glob("*/model.bin"))
                target.write_bytes(b"x" * len(FixtureHandler.payload))
        return True

    cache = make_cache(tmp_path, ca_bundle=ca_bundle, admission=admission)
    with pytest.raises(CacheError) as error:
        cache.acquire(model(base_url + "/asset.bin", FixtureHandler.payload))
    assert error.value.reason_code is ReasonCode.ASSET_VERIFICATION_FAILED
    assert not tuple((tmp_path / "cache" / "snapshots").iterdir())


def test_in_use_and_remove_are_serialized_by_one_lock(
        tmp_path, local_https_fixture, monkeypatch):
    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"serialized"
    FixtureHandler.mode = "normal"
    cache = make_cache(tmp_path, ca_bundle=ca_bundle)
    snapshot = cache.acquire(model(base_url + "/asset.bin", FixtureHandler.payload))
    entered = threading.Event()
    proceed = threading.Event()
    original = cache._known_id

    def paused(snapshot_id):
        value = original(snapshot_id)
        if threading.current_thread().name == "marker":
            entered.set()
            assert proceed.wait(2)
        return value

    monkeypatch.setattr(cache, "_known_id", paused)
    marker = threading.Thread(
        target=cache.mark_in_use, args=(snapshot.snapshot_id,), name="marker"
    )
    outcomes = []

    def remove():
        try:
            cache.remove(snapshot.snapshot_id)
        except CacheError as error:
            outcomes.append(error.reason_code)

    remover = threading.Thread(target=remove, name="remover")
    marker.start()
    assert entered.wait(2)
    remover.start()
    time.sleep(0.05)
    assert remover.is_alive()
    proceed.set()
    marker.join(2)
    remover.join(2)
    assert outcomes == [ReasonCode.CACHE_IN_USE]


class ActionAudit(Audit):
    def __init__(self):
        super().__init__()
        self.fail_action = None

    def record(self, *, action, reason, asset_id, failure_code):
        if action == self.fail_action:
            raise RuntimeError("private audit failure")
        super().record(
            action=action,
            reason=reason,
            asset_id=asset_id,
            failure_code=failure_code,
        )


def test_initial_metadata_audit_failure_replays_exact_empty_index(tmp_path):
    audit = ActionAudit()
    audit.fail_action = "metadata"
    with pytest.raises(CacheError) as error:
        make_cache(tmp_path, audit=audit)
    assert error.value.reason_code is ReasonCode.AUDIT_LOSS
    root = tmp_path / "cache"
    marker = json.loads(
        (root / model_cache_module._METADATA_MARKER).read_text()
    )
    assert marker == {
        "action": "metadata",
        "asset_id": "cache",
        "failure_code": ReasonCode.OK.value,
        "had_previous": False,
        "reason": "managed_snapshot_index",
        "state": "audit_pending",
        "transaction_id": marker["transaction_id"],
    }
    assert json.loads((root / "metadata.json").read_text()) == {
        "schema_version": 2,
        "snapshots": [],
    }
    assert not (root / model_cache_module._METADATA_BACKUP).exists()

    replay_audit = Audit()
    restarted = make_cache(tmp_path, audit=replay_audit)
    assert (
        "metadata",
        "managed_snapshot_index",
        "cache",
        ReasonCode.OK.value,
    ) in replay_audit.events
    assert not (root / model_cache_module._METADATA_MARKER).exists()
    assert not (root / model_cache_module._METADATA_BACKUP).exists()
    restarted.close()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reason", "wrong"),
        ("asset_id", "wrong"),
        ("failure_code", ReasonCode.AUDIT_LOSS.value),
        ("had_previous", True),
        ("extra", "field"),
    ],
)
def test_initial_metadata_audit_pending_malformed_fields_fail_closed(
        tmp_path, field, value):
    audit = ActionAudit()
    audit.fail_action = "metadata"
    with pytest.raises(CacheError):
        make_cache(tmp_path, audit=audit)
    root = tmp_path / "cache"
    marker_path = root / model_cache_module._METADATA_MARKER
    marker = json.loads(marker_path.read_text())
    marker[field] = value
    marker_path.write_text(json.dumps(marker), encoding="ascii")
    pending_marker = marker_path.read_bytes()

    with pytest.raises(CacheError) as error:
        make_cache(tmp_path)

    assert error.value.reason_code is ReasonCode.CACHE_UNKNOWN
    assert marker_path.read_bytes() == pending_marker


def test_remove_audit_failure_keeps_replayable_tombstone_and_restart(
        tmp_path, local_https_fixture):
    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"transaction"
    FixtureHandler.mode = "normal"
    audit = ActionAudit()
    cache = make_cache(tmp_path, ca_bundle=ca_bundle, audit=audit)
    snapshot = cache.acquire(model(base_url + "/asset.bin", FixtureHandler.payload))
    before = (tmp_path / "cache" / "metadata.json").read_bytes()
    audit.fail_action = "remove"
    with pytest.raises(CacheError) as error:
        cache.remove(snapshot.snapshot_id)
    assert error.value.reason_code is ReasonCode.AUDIT_LOSS
    assert cache.snapshots() == ()
    assert not snapshot.path.exists()
    assert (tmp_path / "cache" / "metadata.json").read_bytes() != before
    assert json.loads(
        (
            tmp_path / "cache" / model_cache_module._METADATA_MARKER
        ).read_text()
    )["state"] == "audit_pending"
    cache.close()
    restarted = make_cache(tmp_path, ca_bundle=ca_bundle)
    assert restarted.snapshots() == ()
    root = tmp_path / "cache"
    retained = assert_remove_tombstones_reconciled(root)
    assert retained_tombstones(root) == []
    restarted.close()
    restarted = make_cache(tmp_path, ca_bundle=ca_bundle)
    assert assert_remove_tombstones_reconciled(root) == retained
    restarted.close()


def test_remove_frees_snapshot_directory_without_lingering_tombstones(
        tmp_path, local_https_fixture):
    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"free-the-disk"
    FixtureHandler.mode = "normal"
    cache = make_cache(tmp_path, ca_bundle=ca_bundle)
    snapshot = cache.acquire(
        model(
            base_url + "/asset.bin",
            FixtureHandler.payload,
            relative_path="models/nested/model.bin",
        )
    )
    root = tmp_path / "cache"
    assert (snapshot.path / "models" / "nested" / "model.bin").is_file()

    cache.remove(snapshot.snapshot_id)
    assert cache.snapshots() == ()
    assert not snapshot.path.exists()
    assert retained_tombstones(root) == []
    assert not tuple((root / "quarantine").iterdir())
    assert not tuple((root / "snapshots").iterdir())
    cache.close()


def test_recovery_removes_stale_private_name_tombstones_without_audit(
        tmp_path):
    root = tmp_path / "cache"
    make_cache(tmp_path).close()
    # An interrupted earlier deletion left verified private-name tombstones:
    # a directory with a truncated file, a nested stale tombstone, and a
    # stray top-level truncated file.
    stale_directory = root / "quarantine" / f".retained-directory.{'ab' * 16}"
    (stale_directory / "models").mkdir(parents=True)
    (stale_directory / "models" / "left.bin").write_bytes(b"")
    (stale_directory / f".retained-file.{'cd' * 16}").write_bytes(b"")
    (stale_directory / f".retained-directory.{'ef' * 16}").mkdir()
    (root / "quarantine" / f".retained-file.{'12' * 16}").write_bytes(b"")

    audit = Audit()
    cache = make_cache(tmp_path, audit=audit)
    assert not tuple((root / "quarantine").iterdir())
    assert retained_tombstones(root) == []
    assert not any(event[0] == "quarantine" for event in audit.events)
    assert cache.audit_loss_events == ()
    cache.close()


@pytest.mark.skipif(os.name != "posix", reason="POSIX private-name deletion")
def test_posix_private_delete_removes_entry_while_pinned(tmp_path):
    cache = make_cache(tmp_path)
    containment = cache._require_containment()
    parent_path = tmp_path / "cache" / "quarantine"
    (parent_path / "target").mkdir()
    (parent_path / "target" / "payload.bin").write_bytes(b"payload-bytes")
    payload_identity = model_cache_module._identity(
        (parent_path / "target" / "payload.bin").stat()
    )
    with containment.directory(("quarantine",)) as parent:
        with containment.child_directory(parent, "target") as child:
            containment.delete_retained_file(
                child, "payload.bin", payload_identity
            )
            assert os.listdir(child.handle) == []
            containment.delete_retained_directory(parent, "target", child)
        assert os.listdir(parent.handle) == []
    assert not (parent_path / "target").exists()
    assert retained_tombstones(tmp_path / "cache") == []
    cache.close()


def test_public_recover_settles_remove_before_namespace_cleanup(
        tmp_path, local_https_fixture):
    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"public-recover"
    FixtureHandler.mode = "normal"
    audit = ActionAudit()
    cache = make_cache(tmp_path, ca_bundle=ca_bundle, audit=audit)
    snapshot = cache.acquire(model(base_url + "/asset.bin", FixtureHandler.payload))
    audit.fail_action = "remove"
    with pytest.raises(CacheError) as error:
        cache.remove(snapshot.snapshot_id)
    assert error.value.reason_code is ReasonCode.AUDIT_LOSS

    root = tmp_path / "cache"
    marker_path = root / model_cache_module._METADATA_MARKER
    pending_marker = marker_path.read_bytes()
    tombstones = {item.name for item in (root / "quarantine").iterdir()}
    assert any(name.startswith(f".remove.{snapshot.snapshot_id}.") for name in tombstones)

    with pytest.raises(CacheError) as error:
        cache.recover()
    assert error.value.reason_code is ReasonCode.AUDIT_LOSS
    assert marker_path.read_bytes() == pending_marker
    assert {item.name for item in (root / "quarantine").iterdir()} == tombstones

    audit.fail_action = None
    cache.recover()
    assert cache.snapshots() == ()
    assert not marker_path.exists()
    assert not (root / model_cache_module._METADATA_BACKUP).exists()
    assert_remove_tombstones_reconciled(root)
    cache.close()

    restarted = make_cache(tmp_path, ca_bundle=ca_bundle)
    assert restarted.snapshots() == ()
    assert_remove_tombstones_reconciled(root)
    restarted.close()


def test_ca_bytes_are_pinned_before_bundle_path_changes(
        tmp_path, local_https_fixture):
    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"pinned-ca"
    FixtureHandler.mode = "normal"
    cache = make_cache(tmp_path, ca_bundle=ca_bundle)
    ca_bundle.write_text("not a certificate", encoding="ascii")
    snapshot = cache.acquire(model(base_url + "/asset.bin", FixtureHandler.payload))
    assert snapshot.path.joinpath("model.bin").read_bytes() == FixtureHandler.payload


def test_overall_deadline_closes_response_and_quarantines(tmp_path, monkeypatch):
    class Clock:
        value = 0.0

        def __call__(self):
            return self.value

    class Response:
        status = 200

        def __init__(self):
            self.headers = {"Content-Length": "8"}
            self.closed = False
            self.sock = self

        def geturl(self):
            return "https://example.com/model"

        def settimeout(self, _timeout):
            return None

        def fileno(self):
            return 1

        def read1(self, _size):
            clock.value += 0.6
            return b"1234"

        def close(self):
            self.closed = True

    clock = Clock()
    response = Response()
    cache = ModelCache(
        tmp_path / "cache",
        audit=Audit(),
        admission=lambda **_values: True,
        chunk_size=4,
        timeout_seconds=1,
        monotonic=clock,
    )
    monkeypatch.setattr(cache, "_open", lambda _url, _deadline: response)
    with pytest.raises(CacheError) as error:
        cache.acquire(model("https://example.com/model", b"12345678"))
    assert error.value.reason_code is ReasonCode.UPSTREAM_UNAVAILABLE
    assert response.closed
    assert not tuple((tmp_path / "cache" / "staging").iterdir())


def test_direct_http_client_deadline_closes_owned_connection(tmp_path, monkeypatch):
    closed = threading.Event()

    class Connection:
        def __init__(self, *_args, **_kwargs):
            self.closed = False

        def request(self, *_args, **_kwargs):
            assert closed.wait(2)
            raise OSError("private closed transport")

        def close(self):
            self.closed = True
            closed.set()

    cache = make_cache(tmp_path)
    connection = Connection()
    monkeypatch.setattr(
        model_cache_module.http.client,
        "HTTPSConnection",
        lambda *_args, **_kwargs: connection,
    )
    monkeypatch.setattr(cache, "_ssl_context", lambda: object())
    with pytest.raises(CacheError) as error:
        cache._open("https://example.com/model", cache.monotonic() + 0.1)
    assert error.value.reason_code is ReasonCode.UPSTREAM_UNAVAILABLE
    assert connection.closed


def test_base_exception_during_read_closes_and_quarantines(tmp_path, monkeypatch):
    class Response:
        status = 200

        def __init__(self):
            self.headers = {"Content-Length": "4"}
            self.closed = False
            self.sock = self

        def geturl(self):
            return "https://example.com/model"

        def settimeout(self, _timeout):
            return None

        def fileno(self):
            return 1

        def read1(self, _size):
            raise KeyboardInterrupt

        def close(self):
            self.closed = True

    response = Response()
    cache = make_cache(tmp_path)
    monkeypatch.setattr(cache, "_open", lambda _url, _deadline: response)
    with pytest.raises(KeyboardInterrupt):
        cache.acquire(model("https://example.com/model", b"1234"))
    assert response.closed
    assert not tuple((tmp_path / "cache" / "staging").iterdir())
    assert tuple((tmp_path / "cache" / "quarantine").iterdir())


@pytest.mark.skipif(os.name != "nt", reason="Windows handle ownership regression")
def test_windows_post_open_base_exception_closes_owned_handle(tmp_path, monkeypatch):
    closed = []
    original_close = model_cache_module._Containment._close_windows_handle

    def interrupt(_handle):
        raise KeyboardInterrupt

    def tracking_close(handle):
        closed.append(handle)
        original_close(handle)

    monkeypatch.setattr(
        model_cache_module._Containment, "_windows_path", staticmethod(interrupt)
    )
    monkeypatch.setattr(
        model_cache_module._Containment,
        "_close_windows_handle",
        staticmethod(tracking_close),
    )
    with pytest.raises(KeyboardInterrupt):
        ModelCache(
            tmp_path / "cache",
            audit=Audit(),
            admission=lambda **_values: True,
        )
    assert closed


def test_initialization_parent_swap_cannot_create_in_replacement(
        tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    parent.mkdir()
    moved = tmp_path / "moved-parent"
    swapped = False

    if os.name == "nt":
        original_mkdir = model_cache_module._Containment._mkdir

        def swap_before_create(self, directory, name, *, exist_ok):
            nonlocal swapped
            if not swapped and directory.path == parent and name == "cache":
                swapped = True
                parent.rename(moved)
                parent.mkdir()
            return original_mkdir(self, directory, name, exist_ok=exist_ok)

        monkeypatch.setattr(
            model_cache_module._Containment, "_mkdir", swap_before_create
        )
    else:
        assert os.name == "posix"
        original_os_mkdir = model_cache_module.os.mkdir
        expected_parent_identity = model_cache_module._identity(parent.stat())

        def swap_before_create(name, mode=0o777, *, dir_fd=None):
            nonlocal swapped
            if (
                not swapped
                and name == "cache"
                and dir_fd is not None
                and model_cache_module._identity(os.fstat(dir_fd))
                == expected_parent_identity
            ):
                verified_parent_handle = dir_fd
                swapped = True
                parent.rename(moved)
                original_os_mkdir(parent)
                return original_os_mkdir(
                    name, mode, dir_fd=verified_parent_handle
                )
            return original_os_mkdir(name, mode, dir_fd=dir_fd)

        monkeypatch.setattr(model_cache_module.os, "mkdir", swap_before_create)

    with pytest.raises(CacheError):
        ModelCache(parent / "cache", audit=Audit(), admission=lambda **_values: True)
    assert swapped
    assert not tuple(parent.iterdir())


def test_layout_parent_swaps_fail_without_replacement_side_effects(
        tmp_path, local_https_fixture):
    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"parent-swap"
    FixtureHandler.mode = "normal"
    cache = make_cache(tmp_path, ca_bundle=ca_bundle)
    root = tmp_path / "cache"
    staging = root / "staging"
    retained_staging = root / "retained-staging"
    if os.name == "nt":
        with pytest.raises(PermissionError):
            staging.rename(retained_staging)
        assert not retained_staging.exists()
        cache.close()
    else:
        staging.rename(retained_staging)
        staging.mkdir()
        with pytest.raises(CacheError):
            cache.acquire(model(base_url + "/asset.bin", FixtureHandler.payload))
        assert not tuple(staging.iterdir())
        cache.close()

    cache = make_cache(tmp_path / "remove", ca_bundle=ca_bundle)
    snapshot = cache.acquire(model(base_url + "/asset.bin", FixtureHandler.payload))
    quarantine = tmp_path / "remove" / "cache" / "quarantine"
    retained_quarantine = tmp_path / "remove" / "cache" / "retained-quarantine"
    if os.name == "nt":
        with pytest.raises(PermissionError):
            quarantine.rename(retained_quarantine)
        assert snapshot.path.exists()
        assert not retained_quarantine.exists()
        cache.close()
        return
    quarantine.rename(retained_quarantine)
    quarantine.mkdir()
    with pytest.raises(CacheError):
        cache.remove(snapshot.snapshot_id)
    assert snapshot.path.exists()
    assert not tuple(quarantine.iterdir())
    cache.close()


def test_metadata_transaction_marker_recovers_backup_before_empty_index(
        tmp_path, local_https_fixture):
    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"metadata-recovery"
    FixtureHandler.mode = "normal"
    cache = make_cache(tmp_path, ca_bundle=ca_bundle)
    snapshot = cache.acquire(model(base_url + "/asset.bin", FixtureHandler.payload))
    cache.close()
    root = tmp_path / "cache"
    previous = (root / "metadata.json").read_bytes()
    (root / ".metadata-transaction.backup").write_bytes(previous)
    (root / ".metadata-transaction.json").write_text(
        '{"had_previous":true}', encoding="ascii"
    )
    (root / "metadata.json").write_text(
        '{"schema_version":2,"snapshots":[]}', encoding="ascii"
    )
    restarted = make_cache(tmp_path, ca_bundle=ca_bundle)
    assert restarted.snapshots()[0].snapshot_id == snapshot.snapshot_id
    assert not (root / ".metadata-transaction.json").exists()
    assert not (root / ".metadata-transaction.backup").exists()
    restarted.close()


@pytest.mark.skipif(os.name != "posix", reason="POSIX retained ancestor regression")
def test_posix_root_uses_retained_nofollow_ancestor_chain(tmp_path, monkeypatch):
    reopened = []
    original = model_cache_module._Containment._open_directory_path

    def track_reopen(self, path):
        reopened.append(path)
        return original(self, path)

    monkeypatch.setattr(
        model_cache_module._Containment, "_open_directory_path", track_reopen
    )
    cache = make_cache(tmp_path)
    containment = cache._require_containment()
    handles = tuple(containment._posix_ancestor_handles)
    assert handles
    assert reopened == []
    assert all(os.fstat(handle) for handle in handles)
    cache.close()
    for handle in handles:
        with pytest.raises(OSError):
            os.fstat(handle)


def test_committed_metadata_cleanup_failure_never_rolls_back_or_quarantines(
        tmp_path, local_https_fixture, monkeypatch):
    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"committed-cleanup"
    FixtureHandler.mode = "normal"
    audit = Audit()
    cache = make_cache(tmp_path, ca_bundle=ca_bundle, audit=audit)
    containment = cache._require_containment()
    original_unlink = containment.unlink

    def fail_marker_cleanup(parent_parts, name):
        if name == model_cache_module._METADATA_MARKER:
            raise OSError("private cleanup failure")
        return original_unlink(parent_parts, name)

    monkeypatch.setattr(containment, "unlink", fail_marker_cleanup)
    snapshot = cache.acquire(model(base_url + "/asset.bin", FixtureHandler.payload))
    assert snapshot.path.exists()
    assert cache.snapshots() == (snapshot,)
    assert not tuple((tmp_path / "cache" / "quarantine").iterdir())
    assert any(event[0] == "publish" for event in audit.events)
    cache.close()
    monkeypatch.undo()
    restarted = make_cache(tmp_path, ca_bundle=ca_bundle)
    assert restarted.snapshots()[0].snapshot_id == snapshot.snapshot_id
    restarted_containment = restarted._require_containment()
    restarted_unlink = restarted_containment.unlink

    def fail_remove_backup_cleanup(parent_parts, name):
        if name == model_cache_module._METADATA_BACKUP:
            raise OSError("private remove cleanup failure")
        return restarted_unlink(parent_parts, name)

    monkeypatch.setattr(
        restarted_containment, "unlink", fail_remove_backup_cleanup
    )
    restarted.remove(snapshot.snapshot_id)
    assert restarted.snapshots() == ()
    assert not snapshot.path.exists()
    assert not (
        tmp_path / "cache" / model_cache_module._METADATA_MARKER
    ).exists()
    assert (
        tmp_path / "cache" / model_cache_module._METADATA_BACKUP
    ).exists()
    restarted.close()
    monkeypatch.undo()
    final_restart = make_cache(tmp_path, ca_bundle=ca_bundle)
    assert final_restart.snapshots() == ()
    final_restart.close()


def test_stale_committed_marker_is_cleared_before_next_transaction_payload(
        tmp_path, local_https_fixture, monkeypatch):
    base_url, ca_bundle = local_https_fixture
    FixtureHandler.mode = "normal"
    audit = Audit()
    cache = make_cache(tmp_path, ca_bundle=ca_bundle, audit=audit)
    containment = cache._require_containment()
    original_unlink = containment.unlink

    def retain_committed_marker(parent_parts, name):
        if name == model_cache_module._METADATA_MARKER:
            raise OSError("private stale committed marker")
        return original_unlink(parent_parts, name)

    monkeypatch.setattr(containment, "unlink", retain_committed_marker)
    FixtureHandler.payload = b"transaction-a"
    first = cache.acquire(
        model(base_url + "/asset-a.bin", FixtureHandler.payload, model_id="model-a")
    )
    marker = json.loads(
        (tmp_path / "cache" / model_cache_module._METADATA_MARKER).read_text()
    )
    assert marker["state"] == "committed"
    monkeypatch.setattr(containment, "unlink", original_unlink)

    FixtureHandler.payload = b"transaction-b"
    second = cache.acquire(
        model(base_url + "/asset-b.bin", FixtureHandler.payload, model_id="model-b")
    )
    assert {item.snapshot_id for item in cache.snapshots()} == {
        first.snapshot_id,
        second.snapshot_id,
    }
    persisted = json.loads((tmp_path / "cache" / "metadata.json").read_text())
    assert {item["id"] for item in persisted["snapshots"]} == {
        first.snapshot_id,
        second.snapshot_id,
    }
    assert any(
        action == "publish" and asset_id == second.snapshot_id
        for action, _reason, asset_id, _failure_code in audit.events
    )


def test_transaction_marker_is_atomic_when_publication_is_interrupted(
        tmp_path, local_https_fixture, monkeypatch):
    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"atomic-marker"
    FixtureHandler.mode = "normal"
    cache = make_cache(tmp_path, ca_bundle=ca_bundle)
    before = (tmp_path / "cache" / "metadata.json").read_bytes()
    containment = cache._require_containment()
    original_rename = containment.rename

    def interrupt_marker(source_parent, source_name, target_parent, target_name, **values):
        if (
            source_name.startswith(".metadata-control.")
            and target_name == model_cache_module._METADATA_MARKER
        ):
            raise KeyboardInterrupt
        return original_rename(
            source_parent, source_name, target_parent, target_name, **values
        )

    monkeypatch.setattr(containment, "rename", interrupt_marker)
    with pytest.raises(KeyboardInterrupt):
        cache.acquire(model(base_url + "/asset.bin", FixtureHandler.payload))
    root = tmp_path / "cache"
    assert not (root / model_cache_module._METADATA_MARKER).exists()
    assert not tuple(root.glob(".metadata-control.*.tmp"))
    assert (root / "metadata.json").read_bytes() == before
    cache.close()
    monkeypatch.undo()
    restarted = make_cache(tmp_path, ca_bundle=ca_bundle)
    assert restarted.snapshots() == ()
    restarted.close()


def test_owned_resolver_timeout_terminates_waits_and_closes_pipes(
        tmp_path, monkeypatch):
    class Pipe:
        closed = False

        def close(self):
            self.closed = True

    class Process:
        def __init__(self):
            self.stdin = Pipe()
            self.stdout = Pipe()
            self.returncode = None
            self.terminated = False
            self.waited = False
            self.calls = 0

        def communicate(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise model_cache_module.subprocess.TimeoutExpired("resolver", 1)
            self.returncode = -1
            return "", ""

        def terminate(self):
            self.terminated = True

        def poll(self):
            return self.returncode

        def kill(self):
            self.returncode = -1

        def wait(self, *, timeout):
            assert timeout == model_cache_module._RESOLVER_CLEANUP_SECONDS
            self.waited = True

    process = Process()
    monkeypatch.setattr(
        model_cache_module.subprocess, "Popen", lambda *_args, **_kwargs: process
    )
    cache = make_cache(tmp_path)
    with pytest.raises(CacheError) as error:
        cache._resolve_addresses("example.com", 443, cache.monotonic() + 1)
    assert error.value.reason_code is ReasonCode.UPSTREAM_UNAVAILABLE
    assert process.terminated and process.waited
    assert process.stdin.closed and process.stdout.closed


def test_active_base_exception_survives_all_transport_cleanup_failures():
    calls = []

    class Failing:
        def close(self):
            calls.append("close")
            raise RuntimeError("private cleanup failure")

    class Timer:
        def cancel(self):
            calls.append("cancel")
            raise RuntimeError("private cancel failure")

        def join(self):
            calls.append("join")
            raise RuntimeError("private join failure")

    response = Failing()
    response.status = 200
    response.headers = {}
    owner = model_cache_module._HTTPResponseOwner(
        response, Failing(), Timer(), "https://example.com/model"
    )
    with pytest.raises(KeyboardInterrupt):
        try:
            raise KeyboardInterrupt
        finally:
            owner.close()
    assert calls == ["cancel", "close", "close", "join"]


@pytest.mark.skipif(os.name != "nt", reason="Windows child handle regression")
def test_windows_child_stat_failure_closes_new_handle(tmp_path, monkeypatch):
    class FailingPath:
        def __truediv__(self, _name):
            return self

        def relative_to(self, _root):
            return Path(".")

        def stat(self, *, follow_symlinks):
            assert follow_symlinks is False
            raise KeyboardInterrupt

    containment = object.__new__(model_cache_module._Containment)
    containment.root = tmp_path
    parent = model_cache_module._Directory(FailingPath(), 1, (1, 1))
    containment._directories = {(): parent}
    closed = []

    def fail_close(handle):
        closed.append(handle)
        raise RuntimeError("private handle cleanup failure")

    monkeypatch.setattr(containment, "_verify_windows_handle", lambda *_args: None)
    monkeypatch.setattr(containment, "_windows_open", lambda *_args, **_kwargs: 123)
    monkeypatch.setattr(containment, "_close_windows_handle", fail_close)
    with pytest.raises(KeyboardInterrupt):
        containment._open_child_directory(parent, "child")
    assert closed == [123]


def test_remove_audit_truthfully_reports_tombstone_when_cleanup_is_deferred(
        tmp_path, local_https_fixture, monkeypatch):
    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"deferred-cleanup"
    FixtureHandler.mode = "normal"
    audit = Audit()
    cache = make_cache(tmp_path, ca_bundle=ca_bundle, audit=audit)
    snapshot = cache.acquire(model(base_url + "/asset.bin", FixtureHandler.payload))
    monkeypatch.setattr(
        cache,
        "_delete_tree",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            CacheError(ReasonCode.CACHE_UNKNOWN)
        ),
    )
    cache.remove(snapshot.snapshot_id)
    assert cache.snapshots() == ()
    assert any(
        action == "remove"
        and reason == "unused_snapshot_tombstoned"
        and asset_id == snapshot.snapshot_id
        for action, reason, asset_id, _failure_code in audit.events
    )
    assert tuple((tmp_path / "cache" / "quarantine").iterdir())
    cache.close()


@pytest.mark.skipif(os.name != "nt", reason="Windows exact-handle rename regression")
def test_windows_cache_mutations_do_not_call_os_rename(
        tmp_path, local_https_fixture, monkeypatch):
    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"handle-rename"
    FixtureHandler.mode = "normal"
    cache = make_cache(tmp_path, ca_bundle=ca_bundle)
    monkeypatch.setattr(
        model_cache_module.os,
        "rename",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("path rename used")
        ),
    )
    snapshot = cache.acquire(model(base_url + "/asset.bin", FixtureHandler.payload))
    cache.remove(snapshot.snapshot_id)
    cache.close()


@pytest.mark.parametrize(
    ("root_name", "destination_name"),
    [
        ("ascii-root", "ascii-destination"),
        ("emoji-\U0001f600-root", "destination-\U0001f680"),
    ],
    ids=["ascii", "non-bmp"],
)
@pytest.mark.skipif(os.name != "nt", reason="Windows UTF-16 rename regression")
def test_windows_rename_error_87_fallback_preserves_exact_utf16_name_and_identity(
        tmp_path, monkeypatch, root_name, destination_name):
    from ctypes import wintypes

    class RenameInfoHeader(ctypes.Structure):
        _fields_ = [
            ("replace_if_exists", wintypes.BOOLEAN),
            ("root_directory", wintypes.HANDLE),
            ("file_name_length", wintypes.DWORD),
            ("file_name", wintypes.WCHAR * 1),
        ]

    native_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    native = native_kernel32.SetFileInformationByHandle
    native.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
    )
    native.restype = wintypes.BOOL
    calls = []

    class SetFileInformationByHandle:
        argtypes = None
        restype = None

        def __call__(self, handle, information_class, value, value_size):
            header = ctypes.cast(
                value, ctypes.POINTER(RenameInfoHeader)
            ).contents
            name_bytes = ctypes.string_at(
                ctypes.addressof(header) + RenameInfoHeader.file_name.offset,
                header.file_name_length,
            )
            terminator = ctypes.string_at(
                ctypes.addressof(header)
                + RenameInfoHeader.file_name.offset
                + header.file_name_length,
                ctypes.sizeof(wintypes.WCHAR),
            )
            calls.append(
                (
                    header.root_directory,
                    name_bytes.decode("utf-16-le"),
                    header.file_name_length,
                    value_size,
                    terminator,
                )
            )
            if len(calls) == 1:
                ctypes.set_last_error(87)
                return False
            return native(handle, information_class, value, value_size)

    function = SetFileInformationByHandle()
    class Kernel32:
        SetFileInformationByHandle = function

        def __getattr__(self, name):
            return getattr(native_kernel32, name)

    kernel32 = Kernel32()

    workspace = tmp_path / root_name
    workspace.mkdir()
    cache = make_cache(workspace)
    containment = cache._require_containment()
    source_name = "rename-source"
    containment.mkdir(("staging",), source_name)
    with containment.directory(("staging", source_name)) as source:
        source_identity = source.identity

    monkeypatch.setattr(
        ctypes,
        "WinDLL",
        lambda *_args, **_kwargs: kernel32,
    )
    containment.rename(
        ("staging",),
        source_name,
        ("snapshots",),
        destination_name,
    )

    destination = workspace / "cache" / "snapshots" / destination_name
    with containment.directory(("snapshots", destination_name)) as renamed:
        assert renamed.identity == source_identity
    assert destination.is_dir()
    assert not (workspace / "cache" / "staging" / source_name).exists()
    assert [call[1] for call in calls] == [
        destination_name,
        os.fspath(destination),
    ]
    assert calls[0][0] and calls[1][0] is None
    for _root_handle, exact_name, byte_length, value_size, terminator in calls:
        assert byte_length == len(exact_name.encode("utf-16-le"))
        minimum_size = (
            RenameInfoHeader.file_name.offset
            + byte_length
            + ctypes.sizeof(wintypes.WCHAR)
        )
        alignment = ctypes.alignment(RenameInfoHeader)
        assert value_size == (
            (minimum_size + alignment - 1) // alignment
        ) * alignment
        assert terminator == b"\0" * ctypes.sizeof(wintypes.WCHAR)
    cache.close()


@pytest.mark.parametrize("phase", ["headers", "body"])
def test_absolute_deadline_shuts_down_underlying_socket_before_close(
        tmp_path, monkeypatch, phase):
    order = []
    interrupted = threading.Event()
    body_read_started = threading.Event()

    class Transport:
        def settimeout(self, _timeout):
            return None

        def fileno(self):
            return 1

        def shutdown(self, _how):
            order.append("shutdown")
            interrupted.set()

    class Response:
        status = 200

        def __init__(self, transport):
            self.headers = {"Content-Length": "4"}
            self.sock = transport

        def read1(self, _size):
            order.append("body-read")
            body_read_started.set()
            assert interrupted.wait(2)
            raise OSError("private interrupted body")

        def close(self):
            order.append("response-close")

    class Connection:
        def __init__(self):
            self.sock = Transport()
            self.response = Response(self.sock)

        def request(self, *_args, **_kwargs):
            return None

        def getresponse(self):
            if phase == "headers":
                assert interrupted.wait(2)
                raise OSError("private interrupted headers")
            return self.response

        def close(self):
            order.append("connection-close")

    cache = make_cache(tmp_path)
    deadline_seconds = 0.5 if phase == "body" else 0.05
    if phase == "body":
        cache.timeout_seconds = deadline_seconds
    connection = Connection()
    monkeypatch.setattr(
        model_cache_module.http.client,
        "HTTPSConnection",
        lambda *_args, **_kwargs: connection,
    )
    monkeypatch.setattr(cache, "_ssl_context", lambda: object())
    started = time.monotonic()
    deadline = cache.monotonic() + deadline_seconds
    if phase == "headers":
        with pytest.raises(CacheError):
            cache._open("https://example.com/model", deadline)
    else:
        with pytest.raises(CacheError):
            cache.acquire(model("https://example.com/model", b"1234"))
    elapsed = time.monotonic() - started
    shutdown_index = order.index("shutdown")
    if phase == "body":
        assert body_read_started.is_set()
        assert order.index("body-read") < shutdown_index
        assert shutdown_index < order.index("response-close")
    assert shutdown_index < order.index("connection-close")
    assert elapsed < deadline_seconds + 1


def test_timer_start_failure_closes_connection(tmp_path, monkeypatch):
    class Connection:
        closed = False

        def close(self):
            self.closed = True

    class Timer:
        daemon = False

        def __init__(self, *_args, **_kwargs):
            return None

        def start(self):
            raise KeyboardInterrupt

        def cancel(self):
            return None

        def is_alive(self):
            return False

    cache = make_cache(tmp_path)
    connection = Connection()
    monkeypatch.setattr(
        model_cache_module.http.client,
        "HTTPSConnection",
        lambda *_args, **_kwargs: connection,
    )
    monkeypatch.setattr(model_cache_module.threading, "Timer", Timer)
    monkeypatch.setattr(cache, "_ssl_context", lambda: object())
    with pytest.raises(KeyboardInterrupt):
        cache._open("https://example.com/model", cache.monotonic() + 1)
    assert connection.closed


@pytest.mark.parametrize("boundary", ["marker", "backup"])
def test_prepared_rollback_cleanup_restarts_at_each_deletion_boundary(
        tmp_path, monkeypatch, boundary):
    cache = make_cache(tmp_path)
    root = tmp_path / "cache"
    previous = (root / "metadata.json").read_bytes()
    (root / model_cache_module._METADATA_BACKUP).write_bytes(previous)
    (root / model_cache_module._METADATA_MARKER).write_text(
        '{"had_previous":true,"state":"prepared"}', encoding="ascii"
    )
    (root / "metadata.json").write_text(
        '{"schema_version":2,"snapshots":[]}', encoding="ascii"
    )
    containment = cache._require_containment()
    original_unlink = containment.unlink

    def interrupt(parent_parts, name):
        if name == (
            model_cache_module._METADATA_MARKER
            if boundary == "marker"
            else model_cache_module._METADATA_BACKUP
        ):
            raise KeyboardInterrupt
        return original_unlink(parent_parts, name)

    monkeypatch.setattr(containment, "unlink", interrupt)
    with pytest.raises(KeyboardInterrupt):
        cache._recover_metadata_transaction()
    assert (root / "metadata.json").read_bytes() == previous
    monkeypatch.setattr(containment, "unlink", original_unlink)
    cache._recover_metadata_transaction()
    assert not (root / model_cache_module._METADATA_MARKER).exists()
    assert not (root / model_cache_module._METADATA_BACKUP).exists()


def test_audit_pending_replays_when_committed_marker_publication_is_interrupted(
        tmp_path, local_https_fixture, monkeypatch):
    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"audit-replay"
    FixtureHandler.mode = "normal"
    audit = Audit()
    cache = make_cache(tmp_path, ca_bundle=ca_bundle, audit=audit)
    containment = cache._require_containment()
    original_rename = containment.rename
    root = tmp_path / "cache"

    def interrupt_committed(
            source_parent, source_name, target_parent, target_name, **values):
        if (
            source_name.startswith(".metadata-control.")
            and target_name == model_cache_module._METADATA_MARKER
            and json.loads((root / source_name).read_text())["state"] == "committed"
        ):
            raise KeyboardInterrupt
        return original_rename(
            source_parent, source_name, target_parent, target_name, **values
        )

    monkeypatch.setattr(containment, "rename", interrupt_committed)
    with pytest.raises(KeyboardInterrupt):
        cache.acquire(model(base_url + "/asset.bin", FixtureHandler.payload))
    snapshot = cache.snapshots()[0]
    assert snapshot.path.exists()
    assert not tuple((root / "quarantine").iterdir())
    assert json.loads(
        (root / model_cache_module._METADATA_MARKER).read_text()
    )["state"] == "audit_pending"
    cache.close()
    monkeypatch.undo()
    replay_audit = Audit()
    restarted = make_cache(tmp_path, ca_bundle=ca_bundle, audit=replay_audit)
    assert restarted.snapshots()[0].snapshot_id == snapshot.snapshot_id
    assert any(event[0] == "publish" for event in audit.events)
    assert any(event[0] == "publish" for event in replay_audit.events)


def test_durable_audit_pending_error_cannot_quarantine_committed_data(
        tmp_path, local_https_fixture, monkeypatch):
    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"durable-audit-pending"
    FixtureHandler.mode = "normal"
    cache = make_cache(tmp_path, ca_bundle=ca_bundle)
    original_replace = cache._replace_metadata_control

    def interrupt_after_replace(name, raw):
        original_replace(name, raw)
        if (
            name == model_cache_module._METADATA_MARKER
            and json.loads(raw)["state"] == "audit_pending"
        ):
            raise KeyboardInterrupt

    monkeypatch.setattr(cache, "_replace_metadata_control", interrupt_after_replace)
    with pytest.raises(KeyboardInterrupt):
        cache.acquire(model(base_url + "/asset.bin", FixtureHandler.payload))
    snapshot = cache.snapshots()[0]
    assert snapshot.path.exists()
    assert not tuple((tmp_path / "cache" / "quarantine").iterdir())


def test_interruption_after_audit_emission_leaves_replayable_intent(
        tmp_path, local_https_fixture):
    class InterruptingAudit(Audit):
        def record(self, *, action, reason, asset_id, failure_code):
            super().record(
                action=action,
                reason=reason,
                asset_id=asset_id,
                failure_code=failure_code,
            )
            if action == "publish":
                raise KeyboardInterrupt

    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"audit-interruption"
    FixtureHandler.mode = "normal"
    audit = InterruptingAudit()
    cache = make_cache(tmp_path, ca_bundle=ca_bundle, audit=audit)
    with pytest.raises(KeyboardInterrupt):
        cache.acquire(model(base_url + "/asset.bin", FixtureHandler.payload))
    snapshot = cache.snapshots()[0]
    marker = json.loads(
        (
            tmp_path / "cache" / model_cache_module._METADATA_MARKER
        ).read_text()
    )
    assert marker["state"] == "audit_pending"
    cache.close()
    replay_audit = Audit()
    restarted = make_cache(tmp_path, ca_bundle=ca_bundle, audit=replay_audit)
    assert restarted.snapshots()[0].snapshot_id == snapshot.snapshot_id
    assert any(event[0] == "publish" for event in replay_audit.events)


def test_exception_after_audit_append_keeps_published_mutation_for_replay(
        tmp_path, local_https_fixture):
    class AppendThenFailAudit(Audit):
        def record(self, *, action, reason, asset_id, failure_code):
            super().record(
                action=action,
                reason=reason,
                asset_id=asset_id,
                failure_code=failure_code,
            )
            if action == "publish":
                raise RuntimeError("private ambiguous acknowledgement")

    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"ambiguous-audit"
    FixtureHandler.mode = "normal"
    audit = AppendThenFailAudit()
    cache = make_cache(tmp_path, ca_bundle=ca_bundle, audit=audit)
    with pytest.raises(CacheError) as error:
        cache.acquire(model(base_url + "/asset.bin", FixtureHandler.payload))
    assert error.value.reason_code is ReasonCode.AUDIT_LOSS
    snapshot = cache.snapshots()[0]
    assert snapshot.path.exists()
    assert not tuple((tmp_path / "cache" / "quarantine").iterdir())
    assert json.loads(
        (
            tmp_path / "cache" / model_cache_module._METADATA_MARKER
        ).read_text()
    )["state"] == "audit_pending"
    cache.close()
    replay_audit = Audit()
    restarted = make_cache(tmp_path, ca_bundle=ca_bundle, audit=replay_audit)
    assert restarted.snapshots()[0].snapshot_id == snapshot.snapshot_id
    assert any(event[0] == "publish" for event in replay_audit.events)


def test_absolute_deadline_owns_raw_and_tls_sockets_during_handshake(
        tmp_path, monkeypatch, loopback_peer_seam):
    interrupted = threading.Event()
    shutdown = []

    class Stream:
        def settimeout(self, _timeout):
            return None

        def bind(self, _address):
            return None

        def connect(self, _address):
            return None

        def getpeername(self):
            return ("127.0.0.1", 443)

        def shutdown(self, _how):
            shutdown.append("raw")
            interrupted.set()

        def close(self):
            return None

    class TLSSocket(Stream):
        def do_handshake(self):
            assert interrupted.wait(1)
            raise OSError("private interrupted handshake")

        def shutdown(self, _how):
            shutdown.append("tls")
            interrupted.set()

    class Context:
        def wrap_socket(self, _stream, *_args, **kwargs):
            assert kwargs["do_handshake_on_connect"] is False
            return TLSSocket()

    class Connection:
        def __init__(self, *_args, context, **_kwargs):
            self.context = context
            self.sock = None

        def request(self, *_args, **_kwargs):
            raw = self._create_connection(("example.com", 443), 1)
            self.sock = self.context.wrap_socket(raw, server_hostname="example.com")

        def close(self):
            return None

    cache = make_cache(tmp_path)
    monkeypatch.setattr(
        cache,
        "_resolve_addresses",
        lambda *_args: [(2, 1, 6, "", ("127.0.0.1", 443))],
    )
    monkeypatch.setattr(cache, "_ssl_context", Context)
    monkeypatch.setattr(model_cache_module.socket, "socket", lambda *_args: Stream())
    monkeypatch.setattr(model_cache_module, "_HTTPS_CONNECTION_TYPE", Connection)
    monkeypatch.setattr(model_cache_module.http.client, "HTTPSConnection", Connection)
    started = time.monotonic()
    with pytest.raises(CacheError) as error:
        cache._open("https://example.com/model", started + 0.05)
    assert error.value.reason_code is ReasonCode.UPSTREAM_UNAVAILABLE
    assert time.monotonic() - started < 0.5
    assert "raw" in shutdown and "tls" in shutdown


def test_deadline_expiry_during_raw_to_tls_handoff_skips_handshake_and_cleans(
        tmp_path, monkeypatch, loopback_peer_seam):
    class Clock:
        value = 0.0

        def __call__(self):
            return self.value

    clock = Clock()
    timer_fired = threading.Event()
    closed = []
    handshake_called = False
    timeout_values = []

    class Stream:
        def settimeout(self, timeout):
            timeout_values.append(("raw", timeout))

        def connect(self, _address):
            return None

        def getpeername(self):
            return ("127.0.0.1", 443)

        def shutdown(self, _how):
            clock.value = 1.0
            timer_fired.set()

        def close(self):
            closed.append("raw")

    class TLSSocket(Stream):
        def settimeout(self, timeout):
            timeout_values.append(("tls", timeout))

        def do_handshake(self):
            nonlocal handshake_called
            handshake_called = True
            raise AssertionError("handshake started after absolute deadline")

        def close(self):
            closed.append("tls")

    class Context:
        def wrap_socket(self, _stream, *_args, **kwargs):
            assert kwargs["do_handshake_on_connect"] is False
            assert timer_fired.wait(1)
            return TLSSocket()

    class Connection:
        def __init__(self, *_args, context, **_kwargs):
            self.context = context
            self.sock = None

        def request(self, *_args, **_kwargs):
            raw = self._create_connection(("example.com", 443), 1)
            self.sock = self.context.wrap_socket(
                raw, server_hostname="example.com"
            )

        def close(self):
            closed.append("connection")

    cache = ModelCache(
        tmp_path / "cache",
        audit=Audit(),
        admission=lambda **_values: True,
        monotonic=clock,
    )
    monkeypatch.setattr(
        cache,
        "_resolve_addresses",
        lambda *_args: [(2, 1, 6, "", ("127.0.0.1", 443))],
    )
    monkeypatch.setattr(cache, "_ssl_context", Context)
    monkeypatch.setattr(model_cache_module.socket, "socket", lambda *_args: Stream())
    monkeypatch.setattr(model_cache_module, "_HTTPS_CONNECTION_TYPE", Connection)
    monkeypatch.setattr(model_cache_module.http.client, "HTTPSConnection", Connection)
    started = time.monotonic()
    with pytest.raises(CacheError) as error:
        cache._open("https://example.com/model", 0.05)
    assert error.value.reason_code is ReasonCode.UPSTREAM_UNAVAILABLE
    assert time.monotonic() - started < 0.5
    assert timer_fired.is_set()
    assert not handshake_called
    assert not any(kind == "tls" for kind, _timeout in timeout_values)
    assert {"raw", "tls", "connection"} <= set(closed)


@pytest.mark.parametrize("operation", ["publish", "remove"])
def test_post_rename_base_exception_reconciles_before_propagation(
        tmp_path, local_https_fixture, monkeypatch, operation):
    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"post-rename-reconcile"
    FixtureHandler.mode = "normal"
    cache = make_cache(tmp_path, ca_bundle=ca_bundle)
    snapshot = None
    if operation == "remove":
        snapshot = cache.acquire(
            model(base_url + "/asset.bin", FixtureHandler.payload)
        )
    containment = cache._require_containment()
    original_rename = containment.rename

    def rename_then_interrupt(
            source_parent, source_name, target_parent, target_name, **values):
        result = original_rename(
            source_parent, source_name, target_parent, target_name, **values
        )
        if (
            operation == "publish"
            and source_parent == ("staging",)
            and target_parent == ("snapshots",)
        ) or (
            operation == "remove"
            and source_parent == ("snapshots",)
            and target_parent == ("quarantine",)
        ):
            raise KeyboardInterrupt
        return result

    monkeypatch.setattr(containment, "rename", rename_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        if operation == "publish":
            snapshot = cache.acquire(
                model(base_url + "/asset.bin", FixtureHandler.payload)
            )
        else:
            assert snapshot is not None
            cache.remove(snapshot.snapshot_id)
    persisted = json.loads((tmp_path / "cache" / "metadata.json").read_text())
    if operation == "publish":
        assert len(cache.snapshots()) == 1
        published = cache.snapshots()[0]
        assert published.path.exists()
        assert persisted["snapshots"][0]["id"] == published.snapshot_id
    else:
        assert snapshot is not None
        assert cache.snapshots() == ()
        assert not snapshot.path.exists()
        assert persisted["snapshots"] == []
        assert_remove_tombstones_reconciled(tmp_path / "cache")


def test_resolver_base_exception_cleanup_is_bounded_and_preserves_cancellation(
        tmp_path, monkeypatch):
    class Pipe:
        closed = False

        def close(self):
            self.closed = True

    class Process:
        returncode = None

        def __init__(self):
            self.stdin = Pipe()
            self.stdout = Pipe()
            self.terminated = False
            self.killed = False

        def communicate(self, *_args, **_kwargs):
            raise KeyboardInterrupt

        def terminate(self):
            self.terminated = True

        def wait(self, *, timeout):
            assert timeout == model_cache_module._RESOLVER_CLEANUP_SECONDS
            if not self.killed:
                raise model_cache_module.subprocess.TimeoutExpired("resolver", timeout)
            self.returncode = -1

        def poll(self):
            return self.returncode

        def kill(self):
            self.killed = True

    process = Process()
    monkeypatch.setattr(
        model_cache_module.subprocess, "Popen", lambda *_args, **_kwargs: process
    )
    cache = make_cache(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        cache._resolve_addresses("example.com", 443, cache.monotonic() + 1)
    assert process.terminated and process.killed
    assert process.stdin.closed and process.stdout.closed


@pytest.mark.parametrize("failure", [KeyboardInterrupt, SystemExit, RuntimeError])
@pytest.mark.parametrize("phase", ["bind", "connect"])
def test_resolved_connection_closes_socket_for_every_base_exception(
        tmp_path, monkeypatch, failure, phase):
    class Stream:
        closed = False

        def settimeout(self, _timeout):
            return None

        def bind(self, _address):
            if phase == "bind":
                raise failure

        def connect(self, _address):
            if phase == "connect":
                raise failure

        def close(self):
            self.closed = True

    stream = Stream()
    monkeypatch.setattr(
        model_cache_module.socket, "socket", lambda *_args, **_kwargs: stream
    )
    cache = make_cache(tmp_path)
    with pytest.raises(failure):
        cache._resolved_connection(
            [(2, 1, 6, "", ("127.0.0.1", 443))],
            cache.monotonic() + 1,
            ("127.0.0.1", 0),
        )
    assert stream.closed


def test_resolved_connection_closes_allocated_socket_after_expired_deadline(
        tmp_path, monkeypatch):
    class Stream:
        closed = False

        def settimeout(self, _timeout):
            return None

        def close(self):
            self.closed = True

    stream = Stream()
    monkeypatch.setattr(
        model_cache_module.socket, "socket", lambda *_args, **_kwargs: stream
    )
    cache = make_cache(tmp_path)
    with pytest.raises(CacheError):
        cache._resolved_connection(
            [(2, 1, 6, "", ("127.0.0.1", 443))],
            cache.monotonic() - 1,
            None,
        )
    assert stream.closed


@pytest.mark.parametrize("boundary", ["listing", "final-removal"])
@pytest.mark.skipif(os.name != "posix", reason="POSIX retained-directory race")
def test_recursive_delete_never_mutates_swapped_root_replacement(
        tmp_path, monkeypatch, boundary):
    cache = make_cache(tmp_path)
    containment = cache._require_containment()
    containment.mkdir(("quarantine",), "target")
    containment.mkdir(("quarantine", "target"), "nested")
    descriptor = containment.open_file(
        ("quarantine", "target", "nested"),
        "model.bin",
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
    )
    os.write(descriptor, b"original")
    os.close(descriptor)
    with containment.directory(("quarantine", "target")) as target:
        expected_identity = target.identity
    original_verify = containment.verify
    original_delete = containment.delete_retained_directory
    replacement_payload = b"replacement-must-survive"
    swapped = False

    def swap(child):
        nonlocal swapped
        if swapped or child.path.name != "target":
            return
        retained = child.path.with_name(child.path.name + ".retained")
        os.rename(child.path, retained)
        child.path.mkdir()
        (child.path / "replacement.bin").write_bytes(replacement_payload)
        swapped = True

    def verify_then_swap(directory):
        original_verify(directory)
        if boundary == "listing":
            swap(directory)

    def swap_before_final(parent, name, child):
        if boundary == "final-removal" and child.path.name == "target":
            swap(child)
        return original_delete(parent, name, child)

    monkeypatch.setattr(containment, "verify", verify_then_swap)
    monkeypatch.setattr(
        containment, "delete_retained_directory", swap_before_final
    )
    with pytest.raises(CacheError) as error:
        cache._delete_tree(
            tmp_path / "cache" / "quarantine" / "target",
            expected_identity,
            parent="quarantine",
        )
    assert error.value.reason_code is ReasonCode.CACHE_UNKNOWN
    assert swapped
    assert (
        tmp_path / "cache" / "quarantine" / "target" / "replacement.bin"
    ).read_bytes() == replacement_payload


@pytest.mark.parametrize("kind", ["file", "directory"])
@pytest.mark.skipif(os.name != "posix", reason="POSIX private-name deletion")
def test_posix_delete_does_not_remove_swap_after_identity_check(
        tmp_path, monkeypatch, kind):
    cache = make_cache(tmp_path)
    containment = cache._require_containment()
    parent_path = tmp_path / "cache" / "quarantine"
    replacement_payload = b"replacement-must-survive"
    original_rename = model_cache_module.os.rename
    swapped = False

    if kind == "file":
        (parent_path / "target").write_bytes(b"original")
        expected_identity = model_cache_module._identity(
            (parent_path / "target").stat()
        )
    else:
        (parent_path / "target").mkdir()
        with containment.directory(("quarantine", "target")) as child:
            expected_identity = child.identity

    def swap_after_check(source, target, *args, **kwargs):
        nonlocal swapped
        if source == "target" and not swapped:
            swapped = True
            original_rename(
                source,
                "attacker-retained",
                *args,
                **kwargs,
            )
            if kind == "file":
                descriptor = os.open(
                    "target",
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=kwargs["src_dir_fd"],
                )
                try:
                    os.write(descriptor, replacement_payload)
                finally:
                    os.close(descriptor)
            else:
                os.mkdir("target", dir_fd=kwargs["src_dir_fd"])
                replacement_directory = os.open(
                    "target",
                    os.O_RDONLY | os.O_DIRECTORY,
                    dir_fd=kwargs["src_dir_fd"],
                )
                try:
                    descriptor = os.open(
                        "replacement.bin",
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                        0o600,
                        dir_fd=replacement_directory,
                    )
                    try:
                        os.write(descriptor, replacement_payload)
                    finally:
                        os.close(descriptor)
                finally:
                    os.close(replacement_directory)
        return original_rename(source, target, *args, **kwargs)

    monkeypatch.setattr(model_cache_module.os, "rename", swap_after_check)
    with (
        containment.directory(("quarantine",)) as parent,
        pytest.raises(CacheError),
    ):
        if kind == "file":
            containment.delete_retained_file(
                parent, "target", expected_identity
            )
        else:
            with containment.child_directory(parent, "target") as child:
                containment.delete_retained_directory(
                    parent, "target", child
                )
    assert swapped
    if kind == "file":
        survivors = tuple(
            item
            for item in parent_path.iterdir()
            if item.is_file() and item.read_bytes() == replacement_payload
        )
        assert len(survivors) == 1
    else:
        survivors = tuple(parent_path.rglob("replacement.bin"))
        assert len(survivors) == 1
        assert survivors[0].read_bytes() == replacement_payload


@pytest.mark.parametrize("kind", ["file", "directory"])
@pytest.mark.skipif(os.name != "posix", reason="POSIX final deletion race")
def test_posix_private_delete_preserves_public_final_instruction_replacement(
        tmp_path, monkeypatch, kind):
    cache = make_cache(tmp_path)
    containment = cache._require_containment()
    parent_path = tmp_path / "cache" / "quarantine"
    replacement_payload = b"replacement-at-final-instruction"
    original_open = model_cache_module.os.open
    original_rename = model_cache_module.os.rename
    swapped = False

    if kind == "file":
        target = parent_path / "target"
        target.write_bytes(b"original")
        expected_identity = model_cache_module._identity(target.stat())
    else:
        target = parent_path / "target"
        target.mkdir()
        with containment.directory(("quarantine", "target")) as opened:
            expected_identity = opened.identity

    def swap_private_before_pinned_open(name, flags, *args, **kwargs):
        nonlocal swapped
        prefix = ".retained-file." if kind == "file" else ".retained-directory."
        if isinstance(name, str) and name.startswith(prefix) and not swapped:
            swapped = True
            original_rename(
                name,
                "attacker-retained-original",
                src_dir_fd=kwargs["dir_fd"],
                dst_dir_fd=kwargs["dir_fd"],
            )
            if kind == "file":
                descriptor = original_open(
                    name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=kwargs["dir_fd"],
                )
                try:
                    os.write(descriptor, replacement_payload)
                finally:
                    os.close(descriptor)
            else:
                os.mkdir(name, dir_fd=kwargs["dir_fd"])
        return original_open(name, flags, *args, **kwargs)

    monkeypatch.setattr(model_cache_module.os, "open", swap_private_before_pinned_open)

    with (
        containment.directory(("quarantine",)) as parent,
        pytest.raises(CacheError),
    ):
        if kind == "file":
            containment.delete_retained_file(parent, "target", expected_identity)
        else:
            with containment.child_directory(parent, "target") as child:
                containment.delete_retained_directory(parent, "target", child)
    assert swapped
    replacement = next(
        item
        for item in parent_path.iterdir()
        if item.name.startswith(".retained-")
    )
    if kind == "file":
        assert replacement.read_bytes() == replacement_payload
    else:
        assert replacement.is_dir()


@pytest.mark.parametrize("operation", ["acquire", "remove"])
def test_next_mutation_resyncs_after_cancellation_before_commit_try(
        tmp_path, local_https_fixture, operation):
    class CancelAfterMutation(dict):
        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            raise KeyboardInterrupt

        def pop(self, key, *args):
            super().pop(key, *args)
            raise KeyboardInterrupt

    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"pre-commit-cancellation"
    FixtureHandler.mode = "normal"
    cache = make_cache(tmp_path, ca_bundle=ca_bundle)
    requested = model(base_url + "/asset.bin", FixtureHandler.payload)
    snapshot = cache.acquire(requested) if operation == "remove" else None
    original_load = cache._load_metadata
    cancellation_armed = False

    def load_then_arm_cancellation(**kwargs):
        nonlocal cancellation_armed
        original_load(**kwargs)
        if not cancellation_armed:
            cache._snapshots = CancelAfterMutation(cache._snapshots)
            cancellation_armed = True

    cache._load_metadata = load_then_arm_cancellation

    with pytest.raises(KeyboardInterrupt):
        if operation == "acquire":
            cache.acquire(requested)
        else:
            assert snapshot is not None
            cache.remove(snapshot.snapshot_id)

    root = tmp_path / "cache"
    assert not (root / model_cache_module._METADATA_MARKER).exists()
    if operation == "acquire":
        retried = cache.acquire(requested)
        assert cache.snapshots() == (retried,)
        assert retried.path.exists()
    else:
        assert snapshot is not None
        cache.remove(snapshot.snapshot_id)
        assert cache.snapshots() == ()
        assert not snapshot.path.exists()


def test_windows_delete_handle_identity_is_checked_before_disposition(
        tmp_path, monkeypatch):
    containment = object.__new__(model_cache_module._Containment)
    closed = []
    deleted = []
    monkeypatch.setattr(containment, "_windows_open", lambda *_args, **_kwargs: 17)
    monkeypatch.setattr(containment, "_verify_windows_handle", lambda *_args: None)
    monkeypatch.setattr(containment, "_windows_identity", lambda _handle: (2, 3))
    monkeypatch.setattr(
        containment, "_windows_delete_handle", lambda handle: deleted.append(handle)
    )
    monkeypatch.setattr(
        containment, "_close_windows_handle", lambda handle: closed.append(handle)
    )
    with pytest.raises(CacheError):
        containment._windows_delete(
            tmp_path / "target",
            ("target",),
            directory=False,
            expected_identity=(1, 1),
        )
    assert deleted == []
    assert closed == [17]


@pytest.mark.parametrize("operation", ["publish", "remove"])
def test_caller_boundary_cancellation_after_metadata_commit_never_rolls_back(
        tmp_path, local_https_fixture, monkeypatch, operation):
    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"caller-commit-boundary"
    FixtureHandler.mode = "normal"
    cache = make_cache(tmp_path, ca_bundle=ca_bundle)
    snapshot = None
    if operation == "remove":
        snapshot = cache.acquire(
            model(base_url + "/asset.bin", FixtureHandler.payload)
        )
    original_commit = cache._commit_metadata

    def commit_then_cancel(action, *args, **kwargs):
        original_commit(action, *args, **kwargs)
        if action == operation:
            raise KeyboardInterrupt

    monkeypatch.setattr(cache, "_commit_metadata", commit_then_cancel)
    with pytest.raises(KeyboardInterrupt):
        if operation == "publish":
            snapshot = cache.acquire(
                model(base_url + "/asset.bin", FixtureHandler.payload)
            )
        else:
            assert snapshot is not None
            cache.remove(snapshot.snapshot_id)
    if operation == "publish":
        snapshot = cache.snapshots()[0]
    assert snapshot is not None
    persisted = json.loads((tmp_path / "cache" / "metadata.json").read_text())
    if operation == "publish":
        assert snapshot.path.exists()
        assert persisted["snapshots"][0]["id"] == snapshot.snapshot_id
    else:
        assert not snapshot.path.exists()
        assert persisted["snapshots"] == []
        assert_remove_tombstones_reconciled(tmp_path / "cache")
    if operation == "publish":
        assert not tuple((tmp_path / "cache" / "quarantine").iterdir())


@pytest.mark.parametrize("failure", [KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("boundary", ["audit", "committed-marker"])
def test_post_commit_audit_region_never_rolls_back_on_base_exception(
        tmp_path, local_https_fixture, monkeypatch, failure, boundary):
    class InterruptingAudit(Audit):
        def record(self, *, action, reason, asset_id, failure_code):
            super().record(
                action=action,
                reason=reason,
                asset_id=asset_id,
                failure_code=failure_code,
            )
            if boundary == "audit" and action == "publish":
                raise failure

    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"post-commit-region"
    FixtureHandler.mode = "normal"
    cache = make_cache(
        tmp_path, ca_bundle=ca_bundle, audit=InterruptingAudit()
    )
    if boundary == "committed-marker":
        original_replace = cache._replace_metadata_control

        def interrupt_committed(name, raw):
            if (
                name == model_cache_module._METADATA_MARKER
                and json.loads(raw)["state"] == "committed"
            ):
                raise failure
            return original_replace(name, raw)

        monkeypatch.setattr(
            cache, "_replace_metadata_control", interrupt_committed
        )
    with pytest.raises(failure):
        cache.acquire(model(base_url + "/asset.bin", FixtureHandler.payload))
    snapshot = cache.snapshots()[0]
    assert snapshot.path.exists()
    assert json.loads(
        (
            tmp_path / "cache" / model_cache_module._METADATA_MARKER
        ).read_text()
    )["state"] == "audit_pending"


@pytest.mark.parametrize("failure", [KeyboardInterrupt, SystemExit])
def test_native_handshake_base_exception_closes_every_owned_transport(
        tmp_path, monkeypatch, loopback_peer_seam, failure):
    closed = []

    class Stream:
        def settimeout(self, _timeout):
            return None

        def connect(self, _address):
            return None

        def getpeername(self):
            return ("127.0.0.1", 443)

        def shutdown(self, _how):
            closed.append("raw-shutdown")

        def close(self):
            closed.append("raw-close")

    class TLSSocket(Stream):
        def do_handshake(self):
            raise failure

        def shutdown(self, _how):
            closed.append("tls-shutdown")

        def close(self):
            closed.append("tls-close")

    class Context:
        def wrap_socket(self, _stream, *_args, **kwargs):
            assert kwargs["do_handshake_on_connect"] is False
            return TLSSocket()

    class Connection:
        def __init__(self, *_args, context, **_kwargs):
            self.context = context
            self.sock = None

        def request(self, *_args, **_kwargs):
            raw = self._create_connection(("example.com", 443), 1)
            self.sock = self.context.wrap_socket(
                raw, server_hostname="example.com"
            )

        def close(self):
            closed.append("connection-close")

    class Timer:
        daemon = False

        def __init__(self, *_args):
            self.alive = False

        def start(self):
            self.alive = True

        def cancel(self):
            closed.append("timer-cancel")

        def is_alive(self):
            return self.alive

        def join(self):
            closed.append("timer-join")

    cache = make_cache(tmp_path)
    monkeypatch.setattr(
        cache,
        "_resolve_addresses",
        lambda *_args: [(2, 1, 6, "", ("127.0.0.1", 443))],
    )
    monkeypatch.setattr(cache, "_ssl_context", Context)
    monkeypatch.setattr(model_cache_module.socket, "socket", lambda *_args: Stream())
    monkeypatch.setattr(model_cache_module, "_HTTPS_CONNECTION_TYPE", Connection)
    monkeypatch.setattr(model_cache_module.http.client, "HTTPSConnection", Connection)
    monkeypatch.setattr(model_cache_module.threading, "Timer", Timer)
    with pytest.raises(failure):
        cache._open("https://example.com/model", cache.monotonic() + 1)
    assert {
        "raw-close",
        "tls-close",
        "connection-close",
        "timer-cancel",
        "timer-join",
    } <= set(closed)


def test_pending_audit_recovery_precedes_mutation_and_authorized_loss_classification(
        tmp_path, local_https_fixture, monkeypatch):
    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"pending-before-mutation"
    FixtureHandler.mode = "normal"
    audit = Audit()
    cache = make_cache(tmp_path, ca_bundle=ca_bundle, audit=audit)
    containment = cache._require_containment()
    original_replace = cache._replace_metadata_control

    def fail_publish_audit(name, raw):
        original_replace(name, raw)
        if (
            name == model_cache_module._METADATA_MARKER
            and json.loads(raw)["state"] == "audit_pending"
        ):
            audit.failure = True

    monkeypatch.setattr(cache, "_replace_metadata_control", fail_publish_audit)
    with pytest.raises(CacheError) as error:
        cache.acquire(model(base_url + "/asset.bin", FixtureHandler.payload))
    assert error.value.reason_code is ReasonCode.AUDIT_LOSS
    snapshot = cache.snapshots()[0]
    original_rename = containment.rename

    def assert_recovered_before_rename(*args, **kwargs):
        if args[0] == ("snapshots",):
            assert not (
                tmp_path / "cache" / model_cache_module._METADATA_MARKER
            ).exists()
        return original_rename(*args, **kwargs)

    monkeypatch.setattr(containment, "rename", assert_recovered_before_rename)
    with pytest.raises(CacheError) as error:
        cache.remove(snapshot.snapshot_id)
    assert error.value.reason_code is ReasonCode.AUDIT_LOSS
    assert snapshot.path.exists()
    cache.remove(snapshot.snapshot_id, audit_loss_recovery=True)
    assert not snapshot.path.exists()
    assert cache.audit_loss_events
    assert all(
        event == (ReasonCode.AUDIT_LOSS, snapshot.snapshot_id)
        for event in cache.audit_loss_events
    )


def test_emergency_remove_preserves_pending_publish_while_asset_is_in_use(
        tmp_path, local_https_fixture):
    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"in-use-pending-publish"
    FixtureHandler.mode = "normal"
    audit = ActionAudit()
    audit.fail_action = "publish"
    cache = make_cache(tmp_path, ca_bundle=ca_bundle, audit=audit)
    with pytest.raises(CacheError) as error:
        cache.acquire(model(base_url + "/asset.bin", FixtureHandler.payload))
    assert error.value.reason_code is ReasonCode.AUDIT_LOSS
    snapshot = cache.snapshots()[0]
    marker_path = tmp_path / "cache" / model_cache_module._METADATA_MARKER
    pending_marker = marker_path.read_bytes()

    cache.mark_in_use(snapshot.snapshot_id)
    with pytest.raises(CacheError) as error:
        cache.remove(snapshot.snapshot_id, audit_loss_recovery=True)
    assert error.value.reason_code is ReasonCode.CACHE_IN_USE
    assert marker_path.read_bytes() == pending_marker
    assert snapshot.path.exists()

    cache.release(snapshot.snapshot_id)
    cache.remove(snapshot.snapshot_id, audit_loss_recovery=True)
    assert not snapshot.path.exists()
    assert not marker_path.exists()


def test_asset_cannot_authorize_unrelated_pending_publish_loss(
        tmp_path, local_https_fixture):
    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"unrelated-pending-publish"
    FixtureHandler.mode = "normal"
    audit = ActionAudit()
    cache = make_cache(tmp_path, ca_bundle=ca_bundle, audit=audit)
    first = cache.acquire(
        model(
            base_url + "/first.bin",
            FixtureHandler.payload,
            model_id="first-model",
        )
    )
    audit.fail_action = "publish"
    with pytest.raises(CacheError) as error:
        cache.acquire(
            model(
                base_url + "/second.bin",
                FixtureHandler.payload,
                model_id="second-model",
            )
        )
    assert error.value.reason_code is ReasonCode.AUDIT_LOSS
    pending_marker = (
        tmp_path / "cache" / model_cache_module._METADATA_MARKER
    ).read_bytes()

    with pytest.raises(CacheError) as error:
        cache.remove(first.snapshot_id, audit_loss_recovery=True)
    assert error.value.reason_code is ReasonCode.AUDIT_LOSS
    assert first.path.exists()
    assert (
        tmp_path / "cache" / model_cache_module._METADATA_MARKER
    ).read_bytes() == pending_marker


def test_asset_cannot_authorize_unrelated_pending_remove_loss(
        tmp_path, local_https_fixture):
    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"unrelated-pending-remove"
    FixtureHandler.mode = "normal"
    audit = ActionAudit()
    cache = make_cache(tmp_path, ca_bundle=ca_bundle, audit=audit)
    first = cache.acquire(
        model(
            base_url + "/first.bin",
            FixtureHandler.payload,
            model_id="first-model",
        )
    )
    second = cache.acquire(
        model(
            base_url + "/second.bin",
            FixtureHandler.payload,
            model_id="second-model",
        )
    )
    audit.fail_action = "remove"
    with pytest.raises(CacheError) as error:
        cache.remove(second.snapshot_id)
    assert error.value.reason_code is ReasonCode.AUDIT_LOSS
    pending_marker = (
        tmp_path / "cache" / model_cache_module._METADATA_MARKER
    ).read_bytes()

    with pytest.raises(CacheError) as error:
        cache.remove(first.snapshot_id, audit_loss_recovery=True)
    assert error.value.reason_code is ReasonCode.AUDIT_LOSS
    assert first.path.exists()
    assert (
        tmp_path / "cache" / model_cache_module._METADATA_MARKER
    ).read_bytes() == pending_marker


def test_pending_remove_recovery_uses_backup_and_returns_settled_success(
        tmp_path, local_https_fixture):
    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"pending-remove-retry"
    FixtureHandler.mode = "normal"
    audit = ActionAudit()
    cache = make_cache(tmp_path, ca_bundle=ca_bundle, audit=audit)
    snapshot = cache.acquire(model(base_url + "/asset.bin", FixtureHandler.payload))
    audit.fail_action = "remove"
    with pytest.raises(CacheError) as error:
        cache.remove(snapshot.snapshot_id)
    assert error.value.reason_code is ReasonCode.AUDIT_LOSS

    cache.remove(snapshot.snapshot_id, audit_loss_recovery=True)

    root = tmp_path / "cache"
    assert cache.snapshots() == ()
    assert not snapshot.path.exists()
    assert not (root / model_cache_module._METADATA_MARKER).exists()
    assert not (root / model_cache_module._METADATA_BACKUP).exists()
    assert cache.audit_loss_events == (
        (ReasonCode.AUDIT_LOSS, snapshot.snapshot_id),
    )


def test_pending_remove_recovery_preserves_marker_while_asset_is_in_use(
        tmp_path, local_https_fixture):
    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"pending-remove-in-use"
    FixtureHandler.mode = "normal"
    audit = ActionAudit()
    cache = make_cache(tmp_path, ca_bundle=ca_bundle, audit=audit)
    snapshot = cache.acquire(model(base_url + "/asset.bin", FixtureHandler.payload))
    audit.fail_action = "remove"
    with pytest.raises(CacheError):
        cache.remove(snapshot.snapshot_id)
    marker_path = tmp_path / "cache" / model_cache_module._METADATA_MARKER
    pending_marker = marker_path.read_bytes()
    cache._in_use[snapshot.snapshot_id] = 1

    with pytest.raises(CacheError) as error:
        cache.remove(snapshot.snapshot_id, audit_loss_recovery=True)

    assert error.value.reason_code is ReasonCode.CACHE_IN_USE
    assert marker_path.read_bytes() == pending_marker
    assert tuple((tmp_path / "cache" / "quarantine").iterdir())


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_pending_publish_tree_damage_preserves_marker(
        tmp_path, local_https_fixture, damage):
    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"pending-publish-damage"
    FixtureHandler.mode = "normal"
    audit = ActionAudit()
    audit.fail_action = "publish"
    cache = make_cache(tmp_path, ca_bundle=ca_bundle, audit=audit)
    with pytest.raises(CacheError) as error:
        cache.acquire(model(base_url + "/asset.bin", FixtureHandler.payload))
    assert error.value.reason_code is ReasonCode.AUDIT_LOSS
    snapshot = cache.snapshots()[0]
    marker_path = tmp_path / "cache" / model_cache_module._METADATA_MARKER
    pending_marker = marker_path.read_bytes()
    if damage == "missing":
        snapshot.path.joinpath("model.bin").unlink()
    else:
        snapshot.path.joinpath("model.bin").write_bytes(
            b"x" * len(FixtureHandler.payload)
        )
    cache.close()

    with pytest.raises(CacheError) as error:
        make_cache(tmp_path, ca_bundle=ca_bundle)

    assert error.value.reason_code is ReasonCode.CACHE_UNKNOWN
    assert marker_path.read_bytes() == pending_marker


def test_pending_initial_publish_without_backup_replays(
        tmp_path, local_https_fixture):
    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"initial-publish"
    FixtureHandler.mode = "normal"
    audit = ActionAudit()
    audit.fail_action = "publish"
    cache = make_cache(tmp_path, ca_bundle=ca_bundle, audit=audit)
    with pytest.raises(CacheError):
        cache.acquire(model(base_url + "/asset.bin", FixtureHandler.payload))
    snapshot = cache.snapshots()[0]
    root = tmp_path / "cache"
    marker_path = root / model_cache_module._METADATA_MARKER
    marker = json.loads(marker_path.read_text())
    marker["had_previous"] = False
    marker_path.write_text(json.dumps(marker), encoding="ascii")
    (root / model_cache_module._METADATA_BACKUP).unlink()
    cache.close()

    replay_audit = Audit()
    restarted = make_cache(tmp_path, ca_bundle=ca_bundle, audit=replay_audit)

    assert restarted.snapshots() == (snapshot,)
    assert any(event[0] == "publish" for event in replay_audit.events)
    assert not marker_path.exists()
    restarted.close()


def test_metadata_audit_pending_rejects_nonempty_index(
        tmp_path, local_https_fixture):
    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"nonempty-metadata-index"
    FixtureHandler.mode = "normal"
    audit = ActionAudit()
    audit.fail_action = "publish"
    cache = make_cache(tmp_path, ca_bundle=ca_bundle, audit=audit)
    with pytest.raises(CacheError):
        cache.acquire(model(base_url + "/asset.bin", FixtureHandler.payload))
    root = tmp_path / "cache"
    marker_path = root / model_cache_module._METADATA_MARKER
    marker = json.loads(marker_path.read_text())
    marker.update({
        "action": "metadata",
        "asset_id": "cache",
        "failure_code": ReasonCode.OK.value,
        "had_previous": False,
        "reason": "managed_snapshot_index",
    })
    marker_path.write_text(json.dumps(marker), encoding="ascii")
    (root / model_cache_module._METADATA_BACKUP).unlink()
    pending_marker = marker_path.read_bytes()
    cache.close()

    with pytest.raises(CacheError) as error:
        make_cache(tmp_path, ca_bundle=ca_bundle)

    assert error.value.reason_code is ReasonCode.CACHE_UNKNOWN
    assert marker_path.read_bytes() == pending_marker


@pytest.mark.parametrize("operation", ["publish", "remove"])
def test_audit_loss_recovery_rejects_unrelated_index_record_change(
        tmp_path, local_https_fixture, operation):
    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"whole-index-transition"
    FixtureHandler.mode = "normal"
    audit = ActionAudit()
    cache = make_cache(tmp_path, ca_bundle=ca_bundle, audit=audit)
    first = cache.acquire(
        model(
            base_url + "/first.bin",
            FixtureHandler.payload,
            model_id="first-model",
        )
    )
    if operation == "publish":
        audit.fail_action = "publish"
        with pytest.raises(CacheError):
            cache.acquire(
                model(
                    base_url + "/second.bin",
                    FixtureHandler.payload,
                    model_id="second-model",
                )
            )
        target = next(
            snapshot
            for snapshot in cache.snapshots()
            if snapshot.snapshot_id != first.snapshot_id
        )
    else:
        target = cache.acquire(
            model(
                base_url + "/second.bin",
                FixtureHandler.payload,
                model_id="second-model",
            )
        )
        audit.fail_action = "remove"
        with pytest.raises(CacheError):
            cache.remove(target.snapshot_id)
    root = tmp_path / "cache"
    marker_path = root / model_cache_module._METADATA_MARKER
    pending_marker = marker_path.read_bytes()
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    unrelated = next(
        item for item in metadata["snapshots"] if item["id"] == first.snapshot_id
    )
    unrelated["files"][0]["sha256"] = "f" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(CacheError) as error:
        cache.remove(target.snapshot_id, audit_loss_recovery=True)

    assert error.value.reason_code is ReasonCode.CACHE_UNKNOWN
    assert marker_path.read_bytes() == pending_marker


@pytest.mark.parametrize("operation", ["publish", "remove"])
@pytest.mark.parametrize("reverse_unrelated", [False, True])
def test_audit_loss_recovery_preserves_unrelated_manifest_file_order(
        tmp_path, local_https_fixture, operation, reverse_unrelated):
    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"whole-index-declaration-order"
    FixtureHandler.mode = "normal"
    audit = ActionAudit()
    cache = make_cache(tmp_path, ca_bundle=ca_bundle, audit=audit)
    first = cache.acquire(
        ordered_files_model(
            base_url + "/first.bin",
            FixtureHandler.payload,
            model_id="first-model",
        )
    )
    if operation == "publish":
        audit.fail_action = "publish"
        with pytest.raises(CacheError):
            cache.acquire(
                model(
                    base_url + "/second.bin",
                    FixtureHandler.payload,
                    model_id="second-model",
                )
            )
        target = next(
            snapshot
            for snapshot in cache.snapshots()
            if snapshot.snapshot_id != first.snapshot_id
        )
    else:
        target = cache.acquire(
            model(
                base_url + "/second.bin",
                FixtureHandler.payload,
                model_id="second-model",
            )
        )
        audit.fail_action = "remove"
        with pytest.raises(CacheError):
            cache.remove(target.snapshot_id)

    root = tmp_path / "cache"
    marker_path = root / model_cache_module._METADATA_MARKER
    pending_marker = marker_path.read_bytes()
    metadata_path = root / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    unrelated = next(
        item for item in metadata["snapshots"] if item["id"] == first.snapshot_id
    )
    assert [item["path"] for item in unrelated["files"]] == ["a.bin", "b.bin"]
    if reverse_unrelated:
        unrelated["files"].reverse()
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

        with pytest.raises(CacheError) as error:
            cache.remove(target.snapshot_id, audit_loss_recovery=True)

        assert error.value.reason_code is ReasonCode.CACHE_UNKNOWN
        assert marker_path.read_bytes() == pending_marker
    else:
        cache.remove(target.snapshot_id, audit_loss_recovery=True)
        assert not marker_path.exists()
        assert cache.snapshots() == (first,)


class SurvivingResolverProcess:
    """A fake resolver child that ignores terminate and kill."""

    class Pipe:
        closed = False

        def close(self):
            self.closed = True

    def __init__(self, failure=None):
        self.stdin = self.Pipe()
        self.stdout = self.Pipe()
        self.returncode = None
        self.failure = failure
        self.terminated = False
        self.killed = False
        self.waits = 0

    def communicate(self, *_args, **_kwargs):
        if self.failure is not None:
            raise self.failure
        return "[]", ""

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, *, timeout):
        assert timeout == model_cache_module._RESOLVER_CLEANUP_SECONDS
        self.waits += 1
        raise model_cache_module.subprocess.TimeoutExpired("resolver", timeout)

    def poll(self):
        return self.returncode


def test_resolver_surviving_kill_is_reported_after_pipes_close():
    process = SurvivingResolverProcess()
    with pytest.raises(CacheError) as error:
        ModelCache._cleanup_resolver(process)
    assert error.value.reason_code is ReasonCode.CACHE_UNKNOWN
    assert process.terminated and process.killed and process.waits == 2
    assert process.stdin.closed and process.stdout.closed


@pytest.mark.parametrize("failure", [
    model_cache_module.subprocess.TimeoutExpired("resolver", 1),
    KeyboardInterrupt(),
])
def test_resolver_surviving_kill_keeps_propagating_exception_primary(
        tmp_path, monkeypatch, failure):
    process = SurvivingResolverProcess(failure)
    monkeypatch.setattr(
        model_cache_module.subprocess, "Popen", lambda *_args, **_kwargs: process
    )
    cache = make_cache(tmp_path)
    expected = (
        CacheError if isinstance(failure, model_cache_module.subprocess.TimeoutExpired)
        else type(failure)
    )
    with pytest.raises(expected) as error:
        cache._resolve_addresses("example.com", 443, cache.monotonic() + 1)
    if expected is CacheError:
        assert error.value.reason_code is ReasonCode.UPSTREAM_UNAVAILABLE
    assert process.terminated and process.killed
    assert process.stdin.closed and process.stdout.closed


class FailingSpans:
    """A span sink that raises at one chosen phase of the span lifecycle."""

    def __init__(self, phase):
        self.phase = phase
        self.names = []

    def span(self, name, *, asset_id):
        self.names.append((name, asset_id))
        if self.phase == "construct":
            raise RuntimeError("private span failure")
        return self.Scope(self.phase)

    class Scope:
        def __init__(self, phase):
            self.phase = phase

        def __enter__(self):
            if self.phase == "enter":
                raise RuntimeError("private span failure")
            return self

        def __exit__(self, *_exc_info):
            if self.phase == "exit":
                raise RuntimeError("private span failure")
            return True  # must never suppress the cache's own exceptions


@pytest.mark.parametrize("phase", ["construct", "enter", "exit"])
def test_span_sink_failure_drops_span_without_aborting_acquisition(
        tmp_path, local_https_fixture, phase):
    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"spanned-model"
    FixtureHandler.mode = "normal"
    spans = FailingSpans(phase)
    cache = ModelCache(
        tmp_path / "cache", audit=Audit(), spans=spans,
        admission=lambda **_values: True, ca_bundle=ca_bundle, chunk_size=4,
    )
    assert cache.spans_dropped == 0
    snapshot = cache.acquire(model(base_url + "/asset.bin", FixtureHandler.payload))
    assert snapshot.path.joinpath("model.bin").read_bytes() == FixtureHandler.payload
    assert spans.names == [("acquire_asset", snapshot.snapshot_id)]
    assert not tuple((tmp_path / "cache" / "quarantine").iterdir())
    assert cache.spans_dropped == 1


def test_span_sink_never_suppresses_cache_errors(tmp_path, local_https_fixture):
    base_url, ca_bundle = local_https_fixture
    FixtureHandler.payload = b"expected-content"
    FixtureHandler.mode = "corrupt"
    cache = ModelCache(
        tmp_path / "cache", audit=Audit(), spans=FailingSpans("none"),
        admission=lambda **_values: True, ca_bundle=ca_bundle, chunk_size=4,
    )
    with pytest.raises(CacheError) as error:
        cache.acquire(model(base_url + "/asset.bin", FixtureHandler.payload))
    assert error.value.reason_code is ReasonCode.ASSET_VERIFICATION_FAILED
    assert cache.spans_dropped == 0


def test_span_drop_counter_is_bounded(tmp_path):
    cache = make_cache(tmp_path)
    cache._spans_dropped = model_cache_module.MAX_SPANS_DROPPED - 1
    cache._count_dropped_span()
    cache._count_dropped_span()
    assert cache.spans_dropped == model_cache_module.MAX_SPANS_DROPPED
    cache.close()


@pytest.mark.live
def test_live_verified_asset_download_and_cleanup(tmp_path):
    url = os.environ.get("PODMUNCHER_TEST_ASSET_URL")
    expected_hash = os.environ.get("PODMUNCHER_TEST_ASSET_SHA256")
    if not url or not expected_hash:
        pytest.fail(
            "Set PODMUNCHER_TEST_ASSET_URL and PODMUNCHER_TEST_ASSET_SHA256 "
            "to run the registered model-cache live test."
        )
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.hostname is None
        or len(expected_hash) != 64
        or any(character not in "0123456789abcdef" for character in expected_hash)
    ):
        pytest.fail("PODMUNCHER_TEST_ASSET_URL must be a direct HTTPS URL.")
    maximum_live_bytes = 64 * 1024 * 1024
    connection = http.client.HTTPSConnection(
        parsed.hostname,
        parsed.port or 443,
        context=ssl.create_default_context(),
        timeout=5,
    )
    try:
        target = urlunsplit(("", "", parsed.path or "/", parsed.query, ""))
        connection.request("HEAD", target, headers={"Accept-Encoding": "identity"})
        response = connection.getresponse()
        content_length = response.getheader("Content-Length")
        response.read()
    finally:
        connection.close()
    if response.status != 200 or content_length is None:
        pytest.fail("Live asset must answer direct HTTPS HEAD with Content-Length.")
    payload_size = int(content_length)
    if not 0 < payload_size <= maximum_live_bytes:
        pytest.fail("Live asset Content-Length is outside the registered smoke bound.")
    template = model(url, b"x")
    live_model = ManifestModel(
        template.artifact, template.source_urls,
        (ManifestFile("live.bin", payload_size, expected_hash),),
        template.license, template.provenance, template.compute_types,
        template.worker_peak_ram_bytes, template.worker_peak_vram_bytes,
        template.validated_max_duration_seconds, template.decode_bytes_per_second,
        template.decode_timeout_seconds, template.codec_matrix,
        template.evidence_sha256, template.release_tested, template.synthetic,
    )
    cache = None
    snapshot = None
    try:
        cache = make_cache(
            tmp_path,
            admission=lambda **values: values["declared_bytes"] <= maximum_live_bytes
            and values["allocated_bytes"] <= maximum_live_bytes,
        )
        snapshot = cache.acquire(live_model)
        with snapshot.path.joinpath("live.bin").open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        assert digest == expected_hash
        cache.remove(snapshot.snapshot_id)
        assert not snapshot.path.exists()
    finally:
        if cache is not None:
            try:
                if snapshot is not None and snapshot.snapshot_id in {
                    item.snapshot_id for item in cache.snapshots()
                }:
                    cache.remove(snapshot.snapshot_id, audit_loss_recovery=True)
            finally:
                cache.close()
    print("PODMUNCHER_SMOKE_OK")
