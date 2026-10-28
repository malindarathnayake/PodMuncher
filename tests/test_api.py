"""Bounded loopback API: raw-socket evidence for transport caps and the route matrix."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import struct
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest

from podmuncher import api
from podmuncher.api import ApiActionError, ApiLimits, ApiServer
from podmuncher.domains import Capability, ReasonCode, ServiceState
from podmuncher.requests import JobRequest

CREDENTIAL = "test-credential-0123456789abcdef"
NONCE = "nonce-" + "a" * 26
ROOT = Path(__file__).resolve().parents[1]
JOB_ID = "11111111-1111-4111-8111-111111111111"
ARTIFACT_ID = "22222222-2222-4222-8222-222222222222"
KEY_A = "33333333-3333-4333-8333-333333333333"
KEY_B = "44444444-4444-4444-8444-444444444444"
KEY_C = "88888888-8888-4888-8888-888888888888"
OPERATION_ID = "55555555-5555-4555-8555-555555555555"
ARTIFACT_BYTES = bytes(range(256)) * 1024


class FakeJob:
    def __init__(self, job_id: str = JOB_ID, request_hash: str = "") -> None:
        self.job_id, self.request_hash = job_id, request_hash

    def public_dict(self) -> dict:
        return {"job_id": self.job_id, "state": "queued"}


class FakeActions:
    """In-memory ServiceActions; `fail` maps a method name to the error to raise."""

    def __init__(self, **fail: Exception) -> None:
        self.fail = fail
        self.calls: list[str] = []
        self.keys: dict[str, tuple[str, FakeJob]] = {}
        self.stop_result = (True, "ok")
        self.artifact_closed = asyncio.Event()
        self.artifact_size = len(ARTIFACT_BYTES)
        self.stop_delay = 0.0

    def _enter(self, name: str) -> None:
        self.calls.append(name)
        if name in self.fail:
            raise self.fail[name]

    async def identity(self):
        self._enter("identity")
        return {"pid": 4242, "service_state": ServiceState.RUNNING}

    async def snapshot(self):
        self._enter("snapshot")
        return {"schema_version": 1, "freshness": "fresh"}

    async def list_jobs(self, limit, cursor):
        self._enter("list_jobs")
        self.last_page = (limit, cursor)
        return [FakeJob()], ("next-token" if cursor is None else None)

    async def get_job(self, job_id):
        self._enter("get_job")
        return FakeJob(job_id)

    async def submit(self, request, key):
        self._enter("submit")
        if key in self.keys:
            digest, job = self.keys[key]
            if digest != request.request_hash:
                raise ApiActionError(ReasonCode.CONFLICT)
            return job, True
        job = FakeJob(request_hash=request.request_hash)
        self.keys[key] = (request.request_hash, job)
        return job, False

    async def recheck(self, job_id):
        self._enter("recheck")
        return FakeJob(job_id), {"result": "ready"}, True

    async def retry(self, job_id, key):
        self._enter("retry")
        replayed = key in self.keys
        self.keys[key] = ("", FakeJob(job_id))
        return FakeJob(job_id), "66666666-6666-4666-8666-666666666666", replayed

    async def clone(self, job_id, request, key):
        self._enter("clone")
        return FakeJob("77777777-7777-4777-8777-777777777777", request.request_hash)

    async def cancel(self, job_id):
        self._enter("cancel")
        return FakeJob(job_id), getattr(self, "cancel_status", "intent")

    async def open_artifact(self, job_id, artifact_id):
        self._enter("open_artifact")
        size = self.artifact_size

        async def chunks():
            try:
                for start in range(0, size, 65536):
                    yield ARTIFACT_BYTES[start % len(ARTIFACT_BYTES):][:min(65536, size - start)] \
                        if size == len(ARTIFACT_BYTES) else b"x" * min(65536, size - start)
                    await asyncio.sleep(0)
            finally:
                self.artifact_closed.set()

        return size, "audio/mpeg", chunks()

    async def list_cache(self, limit, cursor):
        self._enter("list_cache")
        return [{"entry_id": "model-a"}], None

    async def remove_cache(self, entry_id):
        self._enter("remove_cache")
        return {"entry_id": entry_id}

    async def setup_plan(self, capabilities, config):
        self._enter("setup_plan")
        assert all(isinstance(item, Capability) for item in capabilities)
        return {"plan_id": "plan-1", "bytes": 0}

    async def setup_apply(self, plan_id, key):
        self._enter("setup_apply")
        return {"operation_id": OPERATION_ID}

    async def setup_status(self, operation_id):
        self._enter("setup_status")
        return {"state": "running", "capabilities": [], "progress": 0.5,
                "reason_code": "ok", "snapshot": None}

    async def stop(self):
        self._enter("stop")
        if self.stop_delay:
            await asyncio.sleep(self.stop_delay)
        return self.stop_result


def make_api_service(tmp_path: Path, *, limits: ApiLimits | None = None, clock=None,
                     registries=None, **fakes):
    """Build (api, actions, audit_records); tmp_path keeps the signature stable for callers."""
    actions = FakeActions(**fakes)
    records: list[tuple[str, str, dict]] = []
    server = ApiServer(actions, credential=CREDENTIAL, instance_nonce=NONCE,
                       audit=lambda action, reason, ids: records.append((action, reason, ids)),
                       clock=clock or time.monotonic, limits=limits, registries=registries)
    return server, actions, records


@asynccontextmanager
async def running(server: ApiServer):
    port = await server.start()
    try:
        yield port
    finally:
        await server.close()


@dataclass
class Raw:
    data: bytes
    closed: bool

    @property
    def status(self) -> int:
        return int(self.data.split(b" ", 2)[1]) if self.data.startswith(b"HTTP/1.1 ") else 0

    @property
    def headers(self) -> dict[str, str]:
        head = self.data.partition(b"\r\n\r\n")[0].decode("latin-1").split("\r\n")[1:]
        return {k.strip().lower(): v.strip() for k, _, v in (line.partition(":") for line in head)}

    @property
    def body(self) -> bytes:
        return self.data.partition(b"\r\n\r\n")[2]

    @property
    def json(self) -> dict:
        return json.loads(self.body)

    @property
    def reason(self) -> str:
        return self.json["error"]["reason_code"]


def raw_socket_request(port: int, raw_bytes: bytes, *, read_timeout: float = 3.0,
                       chunk_delay: float = 0.0, recv_buffer: int | None = None) -> Raw:
    """Send raw bytes over a plain stdlib socket and read until close or timeout."""
    sock = socket.create_connection(("127.0.0.1", port), timeout=read_timeout)
    if recv_buffer is not None:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, recv_buffer)
    try:
        try:
            if chunk_delay:
                for index in range(0, len(raw_bytes), 8):
                    sock.sendall(raw_bytes[index:index + 8])
                    time.sleep(chunk_delay)
            else:
                sock.sendall(raw_bytes)
        except OSError:
            pass
        received = bytearray()
        deadline = time.monotonic() + read_timeout
        closed = False
        while time.monotonic() < deadline:
            sock.settimeout(max(0.01, deadline - time.monotonic()))
            try:
                data = sock.recv(65536)
            except TimeoutError:
                break
            except OSError:
                closed = True
                break
            if not data:
                closed = True
                break
            received += data
        return Raw(bytes(received), closed)
    finally:
        sock.close()


def build(port: int, method: str, path: str, body: object = None, *, headers: dict | None = None,
          drop: tuple[str, ...] = (), raw_body: bytes | None = None, auth: bool = True) -> bytes:
    payload = raw_body if raw_body is not None else (b"" if body is None else json.dumps(body).encode())
    head = {"Host": f"127.0.0.1:{port}", "X-Podmuncher-Schema": "1", "Connection": "close"}
    if auth:
        head["Authorization"] = f"Bearer {CREDENTIAL}"
    if method == "POST":
        head["Content-Length"] = str(len(payload))
        head["Content-Type"] = "application/json"
    head.update(headers or {})
    lines = [f"{method} {path} HTTP/1.1"] + [f"{k}: {v}" for k, v in head.items()
                                              if k.lower() not in {d.lower() for d in drop}]
    return ("\r\n".join(lines) + "\r\n\r\n").encode() + payload


async def call(port: int, method: str, path: str, body: object = None, **kw) -> Raw:
    timeout = kw.pop("read_timeout", 3.0)
    return await asyncio.to_thread(raw_socket_request, port,
                                   build(port, method, path, body, **kw), read_timeout=timeout)


def job_request_dict(tmp_path: Path, name: str = "missing.mp4") -> dict:
    return JobRequest.from_dict({
        "schema_version": 1,
        "source": {"kind": "local", "input": name},
        "outputs": {"transcript": True, "video": False, "video_quality": "1080p"},
        "selection": {"device": "auto", "profile": "balanced", "model": "auto", "language": "auto"},
        "output_root": str(tmp_path / "output"),
        "limits": {"media_bytes": 1024, "duration_seconds": 60, "export_bytes": 512},
    }).to_dict()


def run(coro):
    return asyncio.run(coro)


def fast_limits(**changes) -> ApiLimits:
    base = {"read_deadline_seconds": 0.4, "write_progress_seconds": 0.6,
            "keepalive_idle_seconds": 0.4, "linger_seconds": 0.2}
    base.update(changes)
    return ApiLimits(**base)


# -- constants -------------------------------------------------------------


def test_spec_constants():
    assert api.HEADER_LIMIT_BYTES == 16 * 1024
    assert api.BODY_LIMIT_BYTES == 64 * 1024
    assert api.MAX_CONNECTIONS == 16
    assert api.READ_DEADLINE_SECONDS == 5
    assert api.WRITE_PROGRESS_SECONDS == 10
    assert api.KEEPALIVE_IDLE_SECONDS <= 5
    assert (api.RATE_PER_SECOND, api.RATE_BURST) == (20, 40)
    assert (api.PAGE_MAX, api.MAX_JSON_DEPTH, api.MAX_STRING_BYTES) == (100, 8, 8192)
    assert ApiLimits() == ApiLimits(
        16384, 65536, 16, 5.0, 10.0, 5.0, 20.0, 40, 100, 8, 8192, 65536, 0.5)


# -- routes: success statuses ----------------------------------------------


def test_every_route_success_status_and_envelope(tmp_path):
    async def scenario():
        server, actions, _ = make_api_service(tmp_path)
        request = {"request": job_request_dict(tmp_path)}
        async with running(server) as port:
            rows = [
                ("GET", "/v1/identity", None, None, 200),
                ("GET", "/v1/snapshot", None, None, 200),
                ("GET", "/v1/jobs?limit=100", None, None, 200),
                ("GET", f"/v1/jobs/{JOB_ID}", None, None, 200),
                ("POST", "/v1/jobs", request, KEY_A, 201),
                ("POST", "/v1/jobs", request, KEY_A, 200),
                ("POST", f"/v1/jobs/{JOB_ID}/recheck", {}, None, 200),
                ("POST", f"/v1/jobs/{JOB_ID}/retry", {}, KEY_B, 202),
                ("POST", f"/v1/jobs/{JOB_ID}/retry", {}, KEY_B, 200),
                ("POST", f"/v1/jobs/{JOB_ID}/clone", request, KEY_C, 201),
                ("POST", f"/v1/jobs/{JOB_ID}/cancel", {}, None, 202),
                ("GET", "/v1/cache", None, None, 200),
                ("POST", "/v1/cache/remove", {"entry_id": "model-a"}, None, 200),
                ("POST", "/v1/setup/plan", {"capabilities": [], "config": {}}, None, 200),
                ("POST", "/v1/setup/apply", {"plan_id": "plan-1"}, KEY_A, 202),
                ("GET", f"/v1/setup/{OPERATION_ID}", None, None, 200),
            ]
            for method, path, body, key, status in rows:
                headers = {"Idempotency-Key": key} if key else None
                result = await call(port, method, path, body, headers=headers)
                assert result.status == status, (method, path, result.data[:300])
                assert result.json["schema_version"] == 1
                assert result.json["instance_nonce"] == NONCE
                assert not any(name.startswith("access-control") for name in result.headers)
            actions.cancel_status = "already_terminal"
            done = await call(port, "POST", f"/v1/jobs/{JOB_ID}/cancel", {})
            assert done.status == 200 and done.json["status"] == "already_terminal"
            ident = (await call(port, "GET", "/v1/identity")).json
            assert ident["pid"] == 4242 and ident["service_state"] == "running"
            page = (await call(port, "GET", "/v1/jobs?limit=7&cursor=abc")).json
            assert actions.last_page == (7, "abc") and page["next_cursor"] is None
            first = (await call(port, "GET", "/v1/jobs")).json
            assert actions.last_page == (100, None) and first["next_cursor"] == "next-token"
            assert first["jobs"] == [{"job_id": JOB_ID, "state": "queued"}]
    run(scenario())


def test_stop_ok_then_503_when_unresolved(tmp_path):
    async def scenario():
        server, actions, _ = make_api_service(tmp_path)
        actions.stop_result = (False, "writer_unresolved")
        async with running(server) as port:
            blocked = await call(port, "POST", "/v1/service/stop", {})
            assert blocked.status == 503 and blocked.reason == "writer_unresolved"
            assert not server.stop_requested.is_set()
            actions.stop_result = (True, "ok")
            done = await call(port, "POST", "/v1/service/stop", {})
            assert done.status == 200 and done.json["state"] == "stopped"
            await asyncio.wait_for(server.stop_requested.wait(), 2)
            after = await call(port, "GET", "/v1/identity")
            assert after.status == 503 and after.reason == "service_stopped"
    run(scenario())


def test_stop_response_completes_before_owner_signal(tmp_path):
    async def scenario():
        server, _, _ = make_api_service(tmp_path)
        async with running(server) as port:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(build(port, "POST", "/v1/service/stop", {}))
            await writer.drain()
            data = await asyncio.wait_for(reader.read(65536), 3)
            assert data.startswith(b"HTTP/1.1 200")
            length = int(Raw(data, False).headers["content-length"])
            assert len(Raw(data, False).body) == length
            await asyncio.wait_for(server.stop_requested.wait(), 2)
            writer.close()
    run(scenario())


# -- errors: action mapping ------------------------------------------------


@pytest.mark.parametrize("error,status,reason,action", [
    (ApiActionError(ReasonCode.INVALID_INPUT, not_found=True), 404, "invalid_input", "check_id"),
    (ApiActionError(ReasonCode.CONFLICT), 409, "conflict", "check_state"),
    (ApiActionError(ReasonCode.CACHE_IN_USE), 409, "cache_in_use", "retry_later"),
    (ApiActionError(ReasonCode.INVALID_CONFIG), 422, "invalid_config", "fix_config"),
    (ApiActionError(ReasonCode.INVALID_INPUT), 400, "invalid_input", "fix_request"),
    (ApiActionError(ReasonCode.QUEUE_FULL), 429, "queue_full", "retry_later"),
    (ApiActionError(ReasonCode.SERVICE_UNAVAILABLE), 503, "service_unavailable", "retry_later"),
    (ApiActionError(ReasonCode.SERVICE_STOPPED), 503, "service_stopped", "start_service"),
    (ApiActionError(ReasonCode.WRITER_UNRESOLVED), 503, "writer_unresolved", "wait_for_writers"),
    (RuntimeError("secret internal detail " + CREDENTIAL), 503, "service_unavailable", "retry_later"),
])
def test_action_errors_map_to_status_table(tmp_path, error, status, reason, action):
    async def scenario():
        server, _, _ = make_api_service(tmp_path, get_job=error)
        async with running(server) as port:
            result = await call(port, "GET", f"/v1/jobs/{JOB_ID}")
            assert result.status == status
            body = result.json
            assert body["error"]["reason_code"] == reason
            assert body["error"]["next_action"] == action
            assert isinstance(body["error"]["retryable"], bool)
            assert body["schema_version"] == 1 and body["instance_nonce"] == NONCE
            assert set(body) == {"schema_version", "instance_nonce", "error"}
            assert CREDENTIAL.encode() not in result.data and b"secret" not in result.data
    run(scenario())


def test_idempotency_replay_and_hash_conflict(tmp_path):
    async def scenario():
        server, _, _ = make_api_service(tmp_path)
        first = {"request": job_request_dict(tmp_path)}
        other = {"request": job_request_dict(tmp_path, "other.mp4")}
        hdr = {"Idempotency-Key": KEY_A}
        async with running(server) as port:
            created = await call(port, "POST", "/v1/jobs", first, headers=hdr)
            replay = await call(port, "POST", "/v1/jobs", first, headers=hdr)
            clash = await call(port, "POST", "/v1/jobs", other, headers=hdr)
            assert (created.status, created.json["replayed"]) == (201, False)
            assert (replay.status, replay.json["replayed"]) == (200, True)
            assert replay.json["job"] == created.json["job"]
            assert clash.status == 409 and clash.reason == "conflict"
    run(scenario())


@pytest.mark.parametrize("headers", [None, {"Idempotency-Key": "not-a-uuid"}])
def test_idempotency_key_must_be_uuid(tmp_path, headers):
    async def scenario():
        server, actions, _ = make_api_service(tmp_path)
        async with running(server) as port:
            body = {"request": job_request_dict(tmp_path)}
            result = await call(port, "POST", "/v1/jobs", body, headers=headers)
            assert result.status == 400 and result.reason == "invalid_input"
            assert "submit" not in actions.calls
    run(scenario())


# -- auth and identity -----------------------------------------------------

READ_ROUTES = [
    "/v1/identity", "/v1/snapshot", "/v1/jobs", f"/v1/jobs/{JOB_ID}", "/v1/cache",
    f"/v1/jobs/{JOB_ID}/artifacts/{ARTIFACT_ID}", f"/v1/setup/{OPERATION_ID}",
]


@pytest.mark.parametrize("path", READ_ROUTES)
def test_read_routes_require_auth_and_never_reach_actions(tmp_path, path):
    async def scenario():
        server, actions, records = make_api_service(tmp_path)
        async with running(server) as port:
            for headers, drop, auth in [(None, (), False), ({"Authorization": "Bearer wrong"}, (), True),
                                        ({"Authorization": "Basic " + CREDENTIAL}, (), True),
                                        ({"Authorization": "Bearer"}, (), True)]:
                result = await call(port, "GET", path, headers=headers, drop=drop, auth=auth)
                assert result.status == 401 and result.reason == "unauthorized"
                assert result.json["instance_nonce"] == NONCE
        assert actions.calls == []
        assert all(reason == "unauthorized" for _, reason, _ in records)
    run(scenario())


def test_unknown_route_requires_auth_then_404(tmp_path):
    async def scenario():
        server, _, _ = make_api_service(tmp_path)
        async with running(server) as port:
            for path in ("/docs", "/openapi.json", "/health", "/v1/nope", "/"):
                assert (await call(port, "GET", path, auth=False)).status == 401
                result = await call(port, "GET", path)
                assert result.status == 404
            assert (await call(port, "POST", "/v1/identity", {})).status == 400
    run(scenario())


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "192.168.1.10:{port}", "127.0.0.1:1",
                                  "localhost:{port}", "127.0.0.2:{port}", "[::1]:{port}", ""])
def test_host_variants_rejected(tmp_path, host):
    async def scenario():
        server, actions, _ = make_api_service(tmp_path)
        async with running(server) as port:
            result = await call(port, "GET", "/v1/identity", headers={"Host": host.format(port=port)})
            assert result.status == 403 and result.reason == "invalid_host"
            missing = await call(port, "GET", "/v1/identity", drop=("Host",))
            assert missing.status == 403
            assert actions.calls == []
    run(scenario())


@pytest.mark.parametrize("origin", ["http://evil.example", "null", "http://127.0.0.1"])
def test_any_origin_rejected_without_cors(tmp_path, origin):
    async def scenario():
        server, _, _ = make_api_service(tmp_path)
        async with running(server) as port:
            result = await call(port, "GET", "/v1/identity", headers={"Origin": origin})
            assert result.status == 403 and result.reason == "browser_origin_rejected"
            assert not any(name.startswith("access-control") or name == "vary" for name in result.headers)
            preflight = await call(port, "OPTIONS", "/v1/identity", headers={"Origin": origin})
            assert preflight.status in (0, 400, 403)
    run(scenario())


@pytest.mark.parametrize("schema", ["2", "0", "", "1.0", "01"])
def test_schema_mismatch_426(tmp_path, schema):
    async def scenario():
        server, _, _ = make_api_service(tmp_path)
        async with running(server) as port:
            result = await call(port, "GET", "/v1/identity", headers={"X-Podmuncher-Schema": schema})
            assert result.status == 426 and result.reason == "unsupported_schema"
            assert result.json["error"]["next_action"] == "upgrade_client"
        missing = await call(port, "GET", "/v1/identity", drop=("X-Podmuncher-Schema",)) \
            if False else None
        assert missing is None
    run(scenario())


def test_wrong_nonce_is_visible_to_client_in_every_envelope(tmp_path):
    async def scenario():
        server, _, _ = make_api_service(tmp_path)
        async with running(server) as port:
            for auth in (True, False):
                result = await call(port, "GET", "/v1/identity", auth=auth)
                assert result.json["instance_nonce"] == NONCE != "nonce-other"
    run(scenario())


# -- framing ---------------------------------------------------------------


def _framing_cases(port: int) -> list[tuple[str, bytes, int]]:
    base = build(port, "POST", "/v1/jobs", {}, headers={"Idempotency-Key": KEY_A})
    head, _, _ = base.partition(b"\r\n\r\n")
    common = head.replace(b"Content-Length: 2\r\n", b"")
    return [
        ("chunked", common + b"\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n", 400),
        ("te+cl", head + b"\r\nTransfer-Encoding: chunked\r\n\r\n2\r\n{}\r\n0\r\n\r\n", 400),
        ("double-cl", head + b"\r\nContent-Length: 2\r\n\r\n{}", 400),
        ("conflicting-cl", head + b"\r\nContent-Length: 5\r\n\r\n{}", 400),
        ("cl-list", head.replace(b"Content-Length: 2", b"Content-Length: 2, 2") + b"\r\n\r\n{}", 400),
        ("cl-sign", head.replace(b"Content-Length: 2", b"Content-Length: +2") + b"\r\n\r\n{}", 400),
        ("te-identity", common + b"\r\nTransfer-Encoding: identity\r\nContent-Length: 2\r\n\r\n{}", 400),
        ("obs-fold", common + b"\r\nContent-Length: 2\r\n X-Fold: a\r\n\r\n{}", 400),
        ("lf-only", b"GET /v1/identity HTTP/1.1\nHost: x\n\n", 400),
        ("lf-in-head", b"GET /v1/identity HTTP/1.1\r\nHost: x\nX-A: b\r\n\r\n", 400),
        ("http10", b"GET /v1/identity HTTP/1.0\r\nHost: x\r\n\r\n", 400),
        ("garbage", b"\x16\x03\x01\x02\x00\x01\x00\x01\xfc\x03\x03" + b"\r\n\r\n", 400),
        ("non-ascii", b"GET /v1/identity HTTP/1.1\r\nX-A: \xff\r\n\r\n", 400),
        ("trailing-garbage", common + b"\r\nContent-Length: 2\r\n\r\n{}ZZZZ", 400),
    ]


def test_framing_ambiguity_rejected_and_connection_closed(tmp_path):
    async def scenario():
        server, actions, records = make_api_service(tmp_path)
        async with running(server) as port:
            for name, raw, status in _framing_cases(port):
                result = await asyncio.to_thread(raw_socket_request, port, raw, read_timeout=3)
                assert result.status == status, (name, result.data[:120])
                assert result.closed, name
                assert result.headers["connection"] == "close"
            te = await asyncio.to_thread(raw_socket_request, port, _framing_cases(port)[0][1])
            assert te.reason == "unsupported_transfer_encoding"
        assert actions.calls == []
        assert records
    run(scenario())


# -- size caps -------------------------------------------------------------


def test_header_overflow_is_413_and_closes(tmp_path):
    async def scenario():
        server, actions, _ = make_api_service(tmp_path)
        async with running(server) as port:
            for pad in (api.HEADER_LIMIT_BYTES, 10 * api.HEADER_LIMIT_BYTES):
                raw = build(port, "GET", "/v1/identity", headers={"X-Pad": "a" * pad})
                result = await asyncio.to_thread(raw_socket_request, port, raw)
                assert result.status == 413 and result.reason == "request_too_large"
                assert result.closed
            endless = b"GET /v1/identity HTTP/1.1\r\n" + b"X-A: b\r\n" * 4000
            result = await asyncio.to_thread(raw_socket_request, port, endless)
            assert result.status == 413 and result.closed
            # Exactly at the cap is accepted: total head (incl. blank line) == 16 KiB.
            base = build(port, "GET", "/v1/identity", headers={"X-Pad": ""})
            filler = api.HEADER_LIMIT_BYTES - len(base)
            raw = build(port, "GET", "/v1/identity", headers={"X-Pad": "a" * filler})
            assert raw.index(b"\r\n\r\n") + 4 == api.HEADER_LIMIT_BYTES
            assert (await asyncio.to_thread(raw_socket_request, port, raw)).status == 200
            over = build(port, "GET", "/v1/identity", headers={"X-Pad": "a" * (filler + 1)})
            assert (await asyncio.to_thread(raw_socket_request, port, over)).status == 413
        assert actions.calls == ["identity"]
    run(scenario())


def test_body_cap_is_413_before_body_is_read(tmp_path):
    async def scenario():
        server, actions, _ = make_api_service(tmp_path)
        async with running(server) as port:
            ok_body = b'{"entry_id":"' + b"a" * (api.BODY_LIMIT_BYTES - 40) + b'"}'
            assert len(ok_body) <= api.BODY_LIMIT_BYTES
            big = build(port, "POST", "/v1/cache/remove", raw_body=b"x" * (api.BODY_LIMIT_BYTES + 1))
            result = await asyncio.to_thread(raw_socket_request, port, big)
            assert result.status == 413 and result.reason == "request_too_large" and result.closed
            declared = build(port, "POST", "/v1/cache/remove", raw_body=b"{}",
                             headers={"Content-Length": str(api.BODY_LIMIT_BYTES + 1)})
            result = await asyncio.to_thread(raw_socket_request, port, declared)
            assert result.status == 413
            # Unauthenticated oversized declarations are also capped (pre-auth).
            noauth = build(port, "POST", "/v1/cache/remove", raw_body=b"{}", auth=False,
                           headers={"Content-Length": "999999"})
            assert (await asyncio.to_thread(raw_socket_request, port, noauth)).status == 413
        assert actions.calls == []
    run(scenario())


def test_body_is_not_parsed_before_authentication(tmp_path):
    async def scenario():
        server, actions, _ = make_api_service(tmp_path)
        async with running(server) as port:
            raw = build(port, "POST", "/v1/jobs", raw_body=b"{not json" + b"[" * 100, auth=False)
            result = await asyncio.to_thread(raw_socket_request, port, raw)
            assert result.status == 401
            # Auth failure must not wait for a body that is declared but never sent.
            slow = build(port, "POST", "/v1/jobs", raw_body=b"", auth=False,
                         headers={"Content-Length": "100"})
            started = time.monotonic()
            result = await asyncio.to_thread(raw_socket_request, port, slow, read_timeout=2)
            assert result.status == 401 and time.monotonic() - started < 1.5
        assert actions.calls == []
    run(scenario())


def test_connection_cap_closes_the_17th(tmp_path):
    async def scenario():
        server, _, _ = make_api_service(tmp_path)
        async with running(server) as port:
            held = []
            for _ in range(api.MAX_CONNECTIONS):
                held.append(await asyncio.open_connection("127.0.0.1", port))
            for _ in range(50):
                if server.active_connections == api.MAX_CONNECTIONS:
                    break
                await asyncio.sleep(0.02)
            assert server.active_connections == api.MAX_CONNECTIONS
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            try:
                assert await asyncio.wait_for(reader.read(10), 2) == b""
            except ConnectionError:
                pass
            writer.close()
            # A held slot is still usable and frees the cap when closed.
            h_reader, h_writer = held[0]
            h_writer.write(build(port, "GET", "/v1/identity", headers={"Connection": "keep-alive"}))
            await h_writer.drain()
            assert (await asyncio.wait_for(h_reader.read(65536), 2)).startswith(b"HTTP/1.1 200")
            for _r, w in held:
                w.close()
            for _ in range(100):
                if server.active_connections == 0:
                    break
                await asyncio.sleep(0.02)
            assert server.active_connections == 0
    run(scenario())


# -- deadlines -------------------------------------------------------------


def test_slowloris_headers_get_408_and_close(tmp_path):
    async def scenario():
        server, _, records = make_api_service(tmp_path, limits=fast_limits())
        async with running(server) as port:
            raw = build(port, "GET", "/v1/identity")
            # Stalled mid-header: partial bytes then silence.
            stalled = await asyncio.to_thread(raw_socket_request, port, raw[:30], read_timeout=3)
            assert stalled.status == 408 and stalled.reason == "request_timeout" and stalled.closed
            # Trickle: one byte per 0.15 s never extends the total header deadline.
            reader, writer = await asyncio.open_connection("127.0.0.1", port)

            async def trickle():
                try:
                    for index in range(len(raw)):
                        writer.write(raw[index:index + 1])
                        await writer.drain()
                        await asyncio.sleep(0.15)
                except OSError:
                    pass

            task = asyncio.create_task(trickle())
            started = time.monotonic()
            data = await asyncio.wait_for(reader.read(65536), 3)
            elapsed = time.monotonic() - started
            task.cancel()
            writer.close()
            assert data.startswith(b"HTTP/1.1 408") and elapsed < 1.5
            idle = await asyncio.to_thread(raw_socket_request, port, b"", read_timeout=3)
            assert idle.closed and idle.data == b""
        assert any(reason == "request_timeout" for _, reason, _ in records)
    run(scenario())


def test_slow_body_is_408(tmp_path):
    async def scenario():
        server, actions, _ = make_api_service(tmp_path, limits=fast_limits())
        async with running(server) as port:
            raw = build(port, "POST", "/v1/cache/remove", raw_body=b'{"entry_id":"a"}')
            partial = raw[:-5]  # declared length never fully arrives
            started = time.monotonic()
            result = await asyncio.to_thread(raw_socket_request, port, partial, read_timeout=3)
            assert result.status == 408 and result.closed
            assert time.monotonic() - started < 2
            short = build(port, "POST", "/v1/cache/remove", raw_body=b"{}",
                          headers={"Content-Length": "3"})
            result = await asyncio.to_thread(raw_socket_request, port, short, read_timeout=3)
            assert result.status == 408
        assert actions.calls == []
    run(scenario())


def test_idle_keepalive_closes(tmp_path):
    async def scenario():
        server, _, _ = make_api_service(tmp_path, limits=fast_limits(keepalive_idle_seconds=0.3))
        async with running(server) as port:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(build(port, "GET", "/v1/identity", headers={"Connection": "keep-alive"}))
            await writer.drain()
            first = await asyncio.wait_for(reader.read(65536), 2)
            assert first.startswith(b"HTTP/1.1 200") and b"keep-alive" in first
            started = time.monotonic()
            assert await asyncio.wait_for(reader.read(10), 2) == b""
            assert time.monotonic() - started < 1.5
            writer.close()
    run(scenario())


def test_keepalive_serves_multiple_requests(tmp_path):
    async def scenario():
        server, actions, _ = make_api_service(tmp_path)
        async with running(server) as port:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            for _ in range(3):
                writer.write(build(port, "GET", "/v1/identity", headers={"Connection": "keep-alive"}))
                await writer.drain()
                data = await asyncio.wait_for(reader.read(65536), 2)
                assert data.startswith(b"HTTP/1.1 200")
            writer.close()
            assert actions.calls == ["identity"] * 3
    run(scenario())


def test_write_stall_closes_connection_and_releases_artifact(tmp_path):
    async def scenario():
        server, actions, _ = make_api_service(tmp_path, limits=fast_limits(write_progress_seconds=0.5))
        actions.artifact_size = 48 * 1024 * 1024
        async with running(server) as port:
            raw = build(port, "GET", f"/v1/jobs/{JOB_ID}/artifacts/{ARTIFACT_ID}")
            sock = socket.socket()
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
            sock.connect(("127.0.0.1", port))
            sock.sendall(raw)
            started = time.monotonic()
            await asyncio.wait_for(actions.artifact_closed.wait(), 6)
            assert time.monotonic() - started < 5
            sock.settimeout(3)
            total = 0
            try:
                while True:
                    chunk = sock.recv(65536)
                    if not chunk:
                        break
                    total += len(chunk)
            except OSError:
                pass
            sock.close()
            assert total < actions.artifact_size
            for _ in range(50):
                if server.active_connections == 0:
                    break
                await asyncio.sleep(0.05)
            assert server.active_connections == 0
    run(scenario())


def test_progressing_artifact_may_exceed_write_timeout(tmp_path):
    async def scenario():
        server, actions, _ = make_api_service(tmp_path, limits=fast_limits(write_progress_seconds=0.5))
        actions.artifact_size = 8 * 1024 * 1024
        async with running(server) as port:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(build(port, "GET", f"/v1/jobs/{JOB_ID}/artifacts/{ARTIFACT_ID}"))
            await writer.drain()
            total = 0
            started = time.monotonic()
            while True:
                data = await asyncio.wait_for(reader.read(65536), 2)
                if not data:
                    break
                total += len(data)
                if total < 2 * 1024 * 1024:
                    await asyncio.sleep(0.02)  # slow but always draining
            assert time.monotonic() - started > 0.5
            assert total > actions.artifact_size
            writer.close()
    run(scenario())


# -- rate limit ------------------------------------------------------------


def test_token_bucket_burst_40_then_429_and_refill(tmp_path):
    async def scenario():
        now = [100.0]
        server, _, records = make_api_service(tmp_path, clock=lambda: now[0])
        async with running(server) as port:
            statuses = []
            for _ in range(41):
                statuses.append((await call(port, "GET", "/v1/identity")).status)
            assert statuses[:40] == [200] * 40 and statuses[40] == 429
            limited = await call(port, "GET", "/v1/identity")
            assert limited.reason == "rate_limited" and limited.json["error"]["retryable"] is True
            now[0] += 0.11  # 20 req/s -> 2.2 tokens
            assert (await call(port, "GET", "/v1/identity")).status == 200
            assert (await call(port, "GET", "/v1/identity")).status == 200
            assert (await call(port, "GET", "/v1/identity")).status == 429
        assert sum(1 for _, reason, _ in records if reason == "rate_limited") == 3
    run(scenario())


# -- payload limits --------------------------------------------------------


def _nest(depth: int) -> object:
    value: object = "x"
    for _ in range(depth):
        value = [value]
    return value


def test_payload_limits(tmp_path):
    async def scenario():
        server, actions, _ = make_api_service(tmp_path)
        async with running(server) as port:
            deep = await call(port, "POST", "/v1/setup/plan", {"capabilities": [], "config": {"a": _nest(7)}})
            assert deep.status == 400 and deep.reason == "nesting_limit"
            ok = await call(port, "POST", "/v1/setup/plan", {"capabilities": [], "config": {"a": _nest(6)}})
            assert ok.status == 200
            long = await call(port, "POST", "/v1/setup/plan",
                              {"capabilities": [], "config": {"a": "x" * 8193}})
            assert long.status == 400 and long.reason == "invalid_input"
            exact = await call(port, "POST", "/v1/setup/plan",
                               {"capabilities": [], "config": {"a": "x" * 8192}})
            assert exact.status == 200
            no_schema = await call(port, "GET", "/v1/identity", drop=("X-Podmuncher-Schema",))
            assert no_schema.status == 426
            bomb = await call(port, "POST", "/v1/setup/plan", raw_body=b"[" * 60000)
            assert bomb.status in (400,) and bomb.reason in ("nesting_limit", "invalid_input")
            for query in ("limit=101", "limit=0", "limit=-1", "limit=abc", "limit=1&limit=2",
                          "limit=%31", "other=1", "cursor=bad cursor", "limit="):
                path = "/v1/jobs?" + query
                if " " in path:
                    continue
                result = await call(port, "GET", path)
                assert result.status == 400, query
            assert (await call(port, "GET", "/v1/jobs?limit=100")).status == 200
            assert (await call(port, "GET", "/v1/jobs?limit=1")).status == 200
            for body in (b"[]", b'"x"', b"null", b"NaN", b'{"a":1,"a":2}', b"\xff\xfe",
                         b'{"capabilities":[],"config":{},"extra":1}', b'{"capabilities":[]}'):
                result = await call(port, "POST", "/v1/setup/plan", raw_body=body)
                assert result.status == 400, body
            bad_cap = await call(port, "POST", "/v1/setup/plan",
                                 {"capabilities": ["__invalid__"], "config": {}})
            assert bad_cap.status == 400
        assert actions.calls.count("setup_plan") == 2
    run(scenario())


def test_one_job_per_submit_and_bad_ids(tmp_path):
    async def scenario():
        server, actions, _ = make_api_service(tmp_path)
        req = job_request_dict(tmp_path)
        hdr = {"Idempotency-Key": KEY_A}
        async with running(server) as port:
            for body in ({"requests": [req]}, {"request": [req]}, {"request": req, "more": req},
                         {"request": {**req, "extra": 1}}, {}, []):
                result = await call(port, "POST", "/v1/jobs", body, headers=hdr)
                assert result.status in (400, 422), body
            assert "submit" not in actions.calls
            bad = await call(port, "GET", "/v1/jobs/not-a-uuid")
            assert bad.status == 400
            bad = await call(port, "POST", "/v1/jobs/..%2f/cancel", {})
            assert bad.status in (400, 404)
            query = await call(port, "GET", f"/v1/jobs/{JOB_ID}?x=1")
            assert query.status == 400
    run(scenario())


# -- artifact --------------------------------------------------------------


def test_artifact_stream_headers_and_exact_bytes(tmp_path):
    async def scenario():
        server, actions, records = make_api_service(tmp_path)
        async with running(server) as port:
            result = await call(port, "GET", f"/v1/jobs/{JOB_ID}/artifacts/{ARTIFACT_ID}", read_timeout=5)
            assert result.status == 200
            assert result.body == ARTIFACT_BYTES
            headers = result.headers
            assert headers["content-length"] == str(len(ARTIFACT_BYTES))
            assert headers["content-type"] == "audio/mpeg"
            assert headers["x-podmuncher-schema"] == "1"
            assert headers["x-podmuncher-instance"] == NONCE
            assert "transfer-encoding" not in headers
            assert actions.artifact_closed.is_set()
            missing = await call(port, "GET", f"/v1/jobs/{JOB_ID}/artifacts/{ARTIFACT_ID}",
                                 auth=False)
            assert missing.status == 401 and b"audio" not in missing.data
        assert ("artifacts.get", "ok", {"job_id": JOB_ID, "artifact_id": ARTIFACT_ID}) in records
    run(scenario())


def test_artifact_unknown_id_is_json_404(tmp_path):
    async def scenario():
        server, _, _ = make_api_service(
            tmp_path, open_artifact=ApiActionError(ReasonCode.INVALID_INPUT, not_found=True))
        async with running(server) as port:
            result = await call(port, "GET", f"/v1/jobs/{JOB_ID}/artifacts/{ARTIFACT_ID}")
            assert result.status == 404 and result.json["instance_nonce"] == NONCE
            assert "x-podmuncher-instance" not in result.headers
    run(scenario())


# -- audit and redaction ---------------------------------------------------


def test_every_request_is_audited_without_secrets(tmp_path):
    async def scenario():
        server, _, records = make_api_service(tmp_path)
        body = {"request": job_request_dict(tmp_path, "secret-name.mp4")}
        async with running(server) as port:
            await call(port, "GET", "/v1/identity")
            await call(port, "GET", f"/v1/jobs/{JOB_ID}", auth=False)
            await call(port, "GET", "/v1/identity", headers={"Origin": "http://x"})
            await call(port, "GET", "/v1/identity", headers={"Host": "evil"})
            await call(port, "GET", "/v1/identity", headers={"X-Podmuncher-Schema": "9"})
            await call(port, "GET", "/v1/nope?token=" + CREDENTIAL)
            await call(port, "POST", "/v1/jobs", body, headers={"Idempotency-Key": KEY_A})
            await asyncio.to_thread(raw_socket_request, port, b"GET / HTTP/1.1\r\nTransfer-Encoding: x\r\n\r\n")
        assert [reason for _, reason, _ in records] == [
            "ok", "unauthorized", "browser_origin_rejected", "invalid_host", "unsupported_schema",
            "invalid_input", "ok", "unsupported_transfer_encoding"]
        assert records[1][0] == "jobs.get" and records[1][2] == {"job_id": JOB_ID}
        names = {name for name, _, _ in records}
        assert names <= {r.name for r in api._ROUTES} | {"unknown"}
        text = repr(records)
        assert CREDENTIAL not in text and "secret-name" not in text and "token=" not in text
    run(scenario())


def test_audit_sink_failure_does_not_break_requests(tmp_path):
    async def scenario():
        actions = FakeActions()

        def boom(*_args):
            raise RuntimeError("sink down " + CREDENTIAL)

        server = ApiServer(actions, credential=CREDENTIAL, instance_nonce=NONCE, audit=boom)
        async with running(server) as port:
            result = await call(port, "GET", "/v1/identity")
            assert result.status == 200 and CREDENTIAL.encode() not in result.data
    run(scenario())


def test_logs_never_contain_credential_or_body(tmp_path, caplog):
    async def scenario():
        server, _, _ = make_api_service(tmp_path, get_job=RuntimeError("boom " + CREDENTIAL))
        async with running(server) as port:
            await call(port, "GET", f"/v1/jobs/{JOB_ID}")
            await call(port, "GET", "/v1/identity", headers={"Authorization": "Bearer " + CREDENTIAL + "x"})
    with caplog.at_level("DEBUG"):
        run(scenario())
    assert CREDENTIAL not in caplog.text


# -- listener --------------------------------------------------------------


def test_exactly_one_listening_socket(tmp_path, monkeypatch):
    calls = []
    original = socket.socket.listen

    def counted(self, *args):
        calls.append(self.getsockname())
        return original(self, *args)

    async def scenario():
        # Patch after the loop exists: on Windows the loop's own self-pipe uses socketpair.
        monkeypatch.setattr(socket.socket, "listen", counted)
        server, _, _ = make_api_service(tmp_path)
        port = await server.start()
        try:
            assert len(calls) == 1 and calls[0] == ("127.0.0.1", port)
            assert [s.getsockname()[0] for s in server._server.sockets] == ["127.0.0.1"]
            assert server._runner is not None
            assert (await call(port, "GET", "/v1/identity")).status == 200
            assert len(calls) == 1  # dispatch opens no socket of its own
        finally:
            await server.close()
        refused = socket.socket()
        refused.settimeout(1)
        with pytest.raises(OSError):
            refused.connect(("127.0.0.1", port))
        refused.close()
    run(scenario())


def test_port_is_dynamic_and_loopback_only(tmp_path):
    async def scenario():
        one, _, _ = make_api_service(tmp_path)
        two, _, _ = make_api_service(tmp_path)
        async with running(one) as first, running(two) as second:
            assert first != second and first > 0
            assert one._server.sockets[0].getsockname()[0] == "127.0.0.1"
        with pytest.raises(RuntimeError):
            await one.start()
            await one.start()
        await one.close()
    run(scenario())


def test_requires_credential_and_nonce(tmp_path):
    with pytest.raises(ValueError):
        ApiServer(FakeActions(), credential="", instance_nonce=NONCE, audit=lambda *a: None)
    with pytest.raises(ValueError):
        ApiServer(FakeActions(), credential=CREDENTIAL, instance_nonce="", audit=lambda *a: None)


# -- domains ---------------------------------------------------------------


def test_negative_examples_rejected_by_reason_and_state_parsing():
    registry = json.loads((ROOT / "Docs" / "contract-domains.json").read_text(encoding="utf-8"))
    negatives = registry["negative_examples"]
    assert negatives
    for value in negatives:
        for enum in (ReasonCode, ServiceState):
            with pytest.raises(Exception) as error:
                enum.parse(value)
            assert error.value.reason_code is ReasonCode.INVALID_INPUT
    for code in (ReasonCode.UNAUTHORIZED, ReasonCode.INVALID_HOST, ReasonCode.RATE_LIMITED,
                 ReasonCode.WRITER_UNRESOLVED):
        assert api._describe(code)[0] in (401, 403, 429, 503)
    assert uuid.UUID(JOB_ID)


# -- P5 corrections --------------------------------------------------------


def _linger_rst() -> bytes:
    # struct linger is two u_short on Windows and two int elsewhere; (1, 0) => RST on close.
    return struct.pack("HH", 1, 0) if os.name == "nt" else struct.pack("ii", 1, 0)


def test_stop_signals_owner_even_when_client_resets_during_stop(tmp_path):
    async def scenario():
        server, actions, _ = make_api_service(tmp_path)
        actions.stop_delay = 0.5
        async with running(server) as port:
            def send_then_reset() -> None:
                sock = socket.create_connection(("127.0.0.1", port), timeout=3)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, _linger_rst())
                sock.sendall(build(port, "POST", "/v1/service/stop", {}))
                time.sleep(0.15)  # actions.stop() is still sleeping server-side
                sock.close()  # SO_LINGER(1, 0): abortive close, the peer sees RST

            await asyncio.to_thread(send_then_reset)
            await asyncio.wait_for(server.stop_requested.wait(), 3)
            assert actions.calls == ["stop"]
            after = await call(port, "GET", "/v1/identity")
            assert after.status == 503 and after.reason == "service_stopped"
    run(scenario())


def test_begin_stopping_refuses_new_work_with_service_stopped(tmp_path):
    async def scenario():
        server, actions, _ = make_api_service(tmp_path)
        async with running(server) as port:
            server.begin_stopping()
            server.begin_stopping()  # idempotent
            body = {"request": job_request_dict(tmp_path)}
            result = await call(port, "POST", "/v1/jobs", body, headers={"Idempotency-Key": KEY_A})
            assert result.status == 503 and result.reason == "service_stopped"
            assert actions.calls == [] and not server.stop_requested.is_set()
    run(scenario())


def _small_model_body(tmp_path: Path) -> dict:
    request = job_request_dict(tmp_path)
    request["selection"] = {**request["selection"], "model": "small"}
    return {"request": request}


def test_registry_seam_admits_verified_models_for_submit_and_clone(tmp_path):
    async def scenario():
        body = _small_model_body(tmp_path)
        seen = []

        def registries():
            seen.append(1)
            return frozenset({"small"}), None

        server, actions, _ = make_api_service(tmp_path, registries=registries)
        async with running(server) as port:
            created = await call(port, "POST", "/v1/jobs", body, headers={"Idempotency-Key": KEY_A})
            assert created.status == 201, created.data[:300]
            cloned = await call(port, "POST", f"/v1/jobs/{JOB_ID}/clone", body,
                                headers={"Idempotency-Key": KEY_B})
            assert cloned.status == 201, cloned.data[:300]
        assert actions.calls == ["submit", "clone"] and len(seen) == 2  # read per request

        plain, plain_actions, _ = make_api_service(tmp_path)
        async with running(plain) as port:
            refused = await call(port, "POST", "/v1/jobs", body, headers={"Idempotency-Key": KEY_A})
            assert refused.status == 400 and refused.reason == "invalid_input"
            refused = await call(port, "POST", f"/v1/jobs/{JOB_ID}/clone", body,
                                 headers={"Idempotency-Key": KEY_B})
            assert refused.status == 400 and refused.reason == "invalid_input"
        assert plain_actions.calls == []
    run(scenario())


@pytest.mark.parametrize("provider", [
    lambda: (_ for _ in ()).throw(RuntimeError("registry down " + CREDENTIAL)),
    lambda: (["small"], None),
    lambda: None,
])
def test_registry_provider_fault_is_503_never_500(tmp_path, provider):
    async def scenario():
        body = _small_model_body(tmp_path)
        server, actions, _ = make_api_service(tmp_path, registries=provider)
        async with running(server) as port:
            for path, key in (("/v1/jobs", KEY_A), (f"/v1/jobs/{JOB_ID}/clone", KEY_B)):
                result = await call(port, "POST", path, body, headers={"Idempotency-Key": key})
                assert result.status == 503 and result.reason == "service_unavailable"
                assert CREDENTIAL.encode() not in result.data
        assert actions.calls == []
    run(scenario())


def test_pretty_printed_json_in_one_packet_with_headers_is_accepted(tmp_path):
    async def scenario():
        server, actions, _ = make_api_service(tmp_path)
        async with running(server) as port:
            pretty = json.dumps({"request": job_request_dict(tmp_path)}, indent=2).encode()
            assert b"\n" in pretty and b"\r\n" not in pretty
            raw = build(port, "POST", "/v1/jobs", raw_body=pretty, headers={"Idempotency-Key": KEY_A})
            result = await asyncio.to_thread(raw_socket_request, port, raw)
            assert result.status == 201, result.data[:300]
        assert actions.calls == ["submit"]
    run(scenario())


def test_cursor_padding_rejected_consistently_and_cursors_round_trip_client(tmp_path):
    from podmuncher import service
    from podmuncher.recovery import ProcessIdentity, RuntimeRecord

    async def scenario():
        server, actions, _ = make_api_service(tmp_path)
        async with running(server) as port:
            for query in ("cursor=abc=", "cursor=abc%3D", "limit=5&cursor=a=="):
                raw = await call(port, "GET", "/v1/jobs?" + query)
                assert raw.status == 400 and raw.reason == "invalid_input", query
                assert raw.json["instance_nonce"] == NONCE and raw.closed
            record = RuntimeRecord(1, port, os.getpid(), NONCE, CREDENTIAL,
                                   ProcessIdentity(os.getpid(), 1.0))
            client = service.ServiceClient(record)
            for padded in ("abc=", "abc=="):
                for method in (client.list_jobs, client.list_cache):
                    with pytest.raises(service.ClientError) as error:
                        await asyncio.to_thread(method, 5, padded)
                    assert error.value.reason_code is ReasonCode.INVALID_INPUT
                    assert error.value.status == 400 and not error.value.identity_mismatch
            cursor = "eyJhIjoxfQ-_.~Zz9"
            reply = await asyncio.to_thread(client.list_jobs, 5, cursor)
            assert actions.last_page == (5, cursor) and reply["next_cursor"] is None
            first = await asyncio.to_thread(client.list_jobs)
            assert first["next_cursor"] == "next-token"
            assert (await asyncio.to_thread(client.list_cache, 3, cursor))["entries"]
        assert "list_jobs" in actions.calls and actions.calls.count("list_cache") == 1
    run(scenario())


# -- live smoke ------------------------------------------------------------


@pytest.mark.live
def test_own_transport_smoke(tmp_path):
    import aiohttp

    async def scenario():
        server, actions, records = make_api_service(tmp_path)
        port = await server.start()
        base = f"http://127.0.0.1:{port}"
        headers = {"Authorization": f"Bearer {CREDENTIAL}", "X-Podmuncher-Schema": "1"}
        try:
            async with aiohttp.ClientSession(headers=headers) as session:
                async with session.get(f"{base}/v1/identity") as response:
                    assert response.status == 200
                    assert (await response.json())["instance_nonce"] == NONCE
                body = {"request": job_request_dict(tmp_path)}
                key = {"Idempotency-Key": KEY_A}
                async with session.post(f"{base}/v1/jobs", json=body, headers=key) as response:
                    assert response.status == 201
                async with session.post(f"{base}/v1/jobs", json=body, headers=key) as response:
                    assert response.status == 200 and (await response.json())["replayed"] is True
                async with session.get(f"{base}/v1/jobs?limit=5") as response:
                    assert response.status == 200 and (await response.json())["jobs"]
                async with session.get(f"{base}/v1/jobs/{JOB_ID}") as response:
                    assert response.status == 200
                async with session.get(f"{base}/v1/jobs/{JOB_ID}/artifacts/{ARTIFACT_ID}") as response:
                    assert response.status == 200
                    assert response.headers["X-Podmuncher-Schema"] == "1"
                    assert await response.read() == ARTIFACT_BYTES
                async with session.get(f"{base}/v1/identity",
                                       headers={"Origin": "http://evil.example"}) as response:
                    assert response.status == 403
            probes = {
                "oversized-headers": (build(port, "GET", "/v1/identity", headers={"X-Pad": "a" * 20000}), 413),
                "bad-host": (build(port, "GET", "/v1/identity", headers={"Host": "localhost"}), 403),
                "origin": (build(port, "GET", "/v1/identity", headers={"Origin": "http://x"}), 403),
                "transfer-encoding": (build(port, "POST", "/v1/jobs", raw_body=b"0\r\n\r\n",
                                            headers={"Transfer-Encoding": "chunked"}), 400),
            }
            for name, (raw, status) in probes.items():
                result = await asyncio.to_thread(raw_socket_request, port, raw)
                assert result.status == status and result.closed, name
            started = time.monotonic()
            slow = await asyncio.to_thread(
                raw_socket_request, port, b"GET /v1/identity HTTP/1.1\r\nHost: 127.0.0.1\r\n",
                read_timeout=8)
            elapsed = time.monotonic() - started
            assert slow.status == 408 and slow.closed and 4.0 <= elapsed <= 7.5
            async with aiohttp.ClientSession(headers=headers) as session,                     session.post(f"{base}/v1/service/stop", json={}) as response:
                assert response.status == 200
                await response.read()
            await asyncio.wait_for(server.stop_requested.wait(), 3)
        finally:
            await server.close()
        probe = socket.socket()
        probe.settimeout(1)
        with pytest.raises(OSError):
            probe.connect(("127.0.0.1", port))
        probe.close()
        assert records and "stop" in actions.calls

    run(scenario())
    print("PODMUNCHER_SMOKE_OK")
