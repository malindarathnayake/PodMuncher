"""Bounded loopback control API with a raw asyncio front door.

Contract (Docs/spec.md:105-128), quoted for reviewers:

    JSON responses carry schema_version=1 and instance_nonce. Successful binary
    artifact responses instead carry X-Podmuncher-Schema: 1 and
    X-Podmuncher-Instance headers with Content-Length and Content-Type. Clients
    send X-Podmuncher-Schema: 1 and Authorization: Bearer credential. Bind
    127.0.0.1 only, dynamic allocated port. Accept Host exactly
    127.0.0.1:actual-port; reject every Origin header, disable CORS/docs.

    Fixed raw-transport limits: 16 KiB total headers, 64 KiB body, 16
    concurrent connections, 5s header/body read deadline, 10s write-progress/
    idle timeout (reset only when bytes drain; a progressing artifact response
    may exceed 10s total), <=5s idle keepalive, service token bucket 20
    requests/s with burst 40. Apply connection/header/deadline limits before
    authentication; authenticate headers before JSON body parsing. Reject any
    Transfer-Encoding and ambiguous/multiple Content-Length, enforce
    streamed/declaration length, nesting <=8, each string <=8192, page size
    <=100, one job per submit.

Transport design
================
``asyncio.start_server`` on 127.0.0.1 port 0 is the only listening socket. The
front door (``ApiServer._serve``) owns every pre-authentication limit: the
connection counter, the 16 KiB header cap, the read deadlines, framing checks
(Transfer-Encoding, Content-Length), the token bucket, Host/Origin/schema and
the Bearer credential. Only an authenticated request with a bounded, fully read
body is re-serialised as HTTP/1.1 and handed to aiohttp.

aiohttp is driven in process through ``_MemoryTransport``: a real
``aiohttp.web.Server`` request handler is attached to an in-memory
``asyncio.Transport`` so that aiohttp's own HTTP parser, router and response
writer run, but no socket of any kind is involved. A private
``socket.socketpair()`` was rejected on purpose: on Windows CPython implements
``socketpair`` by binding a temporary listening socket on 127.0.0.1, which
would create a second, briefly reachable listener. The in-memory transport
applies write back pressure (pause_writing/resume_writing) so artifact streams
are bounded by the front door's drain-based write-progress timeout.

The module is never imported from ``podmuncher/__init__`` (aiohttp is forbidden
on the ``--help``/``--version`` path).
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import http
import inspect
import json
import logging
import re
import time
from collections import deque
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import parse_qsl

from aiohttp import web

from .domains import (
    Capability,
    DomainValidationError,
    ReasonCode,
    validate_json_payload,
    validate_uuid_string,
)
from .requests import JobRequest

_LOG = logging.getLogger("podmuncher.api")

SCHEMA_VERSION = 1
HEADER_LIMIT_BYTES = 16 * 1024
BODY_LIMIT_BYTES = 64 * 1024
MAX_CONNECTIONS = 16
READ_DEADLINE_SECONDS = 5.0
WRITE_PROGRESS_SECONDS = 10.0
KEEPALIVE_IDLE_SECONDS = 5.0
RATE_PER_SECOND = 20.0
RATE_BURST = 40
PAGE_MAX = 100
MAX_JSON_DEPTH = 8
MAX_STRING_BYTES = 8192
CHUNK_BYTES = 64 * 1024
LINGER_SECONDS = 0.5
LINGER_BYTES = 128 * 1024
MAX_TARGET_BYTES = 2048
MAX_CURSOR_CHARS = 512
MAX_TOKEN_CHARS = 4096

_STREAMING = web.RequestKey("streaming", bool)
_REASON_HEADER = "X-Podmuncher-Reason"
_UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_TARGET_RE = re.compile(r"/[A-Za-z0-9._~/\-]*(\?[A-Za-z0-9._~=&\-]*)?")
_HEADER_NAME_RE = re.compile(r"[!#$&'*+.^_`|~0-9A-Za-z-]+")
_HEADER_VALUE_RE = re.compile(r"[\t\x20-\x7e]*")
# No "=": ServiceClient percent-encodes it and _TARGET_RE rejects "%", so both doors agree.
_CURSOR_RE = re.compile(r"[A-Za-z0-9._~\-]{1,512}")
_CONTENT_TYPE_RE = re.compile(r"[A-Za-z0-9!#$&^_.+\-]+/[A-Za-z0-9!#$&^_.+\-]+(; charset=[A-Za-z0-9._\-]+)?")
_REQUEST_LINE_RE = re.compile(rb"[A-Z]{3,7} /[\x21-\x7e]* HTTP/1\.1")
_BARE_LF_RE = re.compile(rb"(?<!\r)\n")
_LIMIT_RE = re.compile(r"[0-9]{1,3}")


@dataclass(frozen=True)
class ApiLimits:
    """Transport limits; defaults are the spec constants, tests may shrink times."""

    header_bytes: int = HEADER_LIMIT_BYTES
    body_bytes: int = BODY_LIMIT_BYTES
    max_connections: int = MAX_CONNECTIONS
    read_deadline_seconds: float = READ_DEADLINE_SECONDS
    write_progress_seconds: float = WRITE_PROGRESS_SECONDS
    keepalive_idle_seconds: float = KEEPALIVE_IDLE_SECONDS
    rate_per_second: float = RATE_PER_SECOND
    rate_burst: int = RATE_BURST
    max_page: int = PAGE_MAX
    max_depth: int = MAX_JSON_DEPTH
    max_string_bytes: int = MAX_STRING_BYTES
    chunk_bytes: int = CHUNK_BYTES
    linger_seconds: float = LINGER_SECONDS


class ApiActionError(Exception):
    """Raised by injected actions; mapped to the status table, never echoed."""

    def __init__(self, reason_code: ReasonCode, *, not_found: bool = False) -> None:
        self.reason_code = reason_code if isinstance(reason_code, ReasonCode) else ReasonCode.INVALID_INPUT
        self.not_found = not_found
        super().__init__(self.reason_code.value)


class ServiceActions(Protocol):
    """Service operations injected into the API; the API never touches storage."""

    async def identity(self) -> Mapping[str, Any]: ...
    async def snapshot(self) -> Any: ...
    async def list_jobs(self, limit: int, cursor: str | None) -> tuple[list[Any], str | None]: ...
    async def get_job(self, job_id: str) -> Any: ...
    async def submit(self, request: JobRequest, key: str) -> tuple[Any, bool]: ...
    async def recheck(self, job_id: str) -> tuple[Any, Mapping[str, Any], bool]: ...
    async def retry(self, job_id: str, key: str) -> tuple[Any, str, bool]: ...
    async def clone(self, job_id: str, request: JobRequest, key: str) -> Any: ...
    async def cancel(self, job_id: str) -> tuple[Any, str]: ...
    async def open_artifact(
        self, job_id: str, artifact_id: str
    ) -> tuple[int, str, AsyncIterator[bytes]]: ...
    async def list_cache(self, limit: int, cursor: str | None) -> tuple[list[Any], str | None]: ...
    async def remove_cache(self, entry_id: str) -> Any: ...
    async def setup_plan(self, capabilities: list[Capability], config: Mapping[str, Any]) -> Any: ...
    async def setup_apply(self, plan_id: str, key: str) -> Any: ...
    async def setup_status(self, operation_id: str) -> Mapping[str, Any]: ...
    async def stop(self) -> tuple[bool, str]: ...


AuditFn = Callable[[str, str, Mapping[str, str]], Any]
RegistryPair = tuple[frozenset[str] | None, frozenset[str] | None]
RegistriesFn = Callable[[], RegistryPair]

# reason -> (status, retryable, next_action). Unknown IDs use status 404 via not_found.
_REASONS: dict[ReasonCode, tuple[int, bool, str]] = {
    ReasonCode.INVALID_INPUT: (400, False, "fix_request"),
    ReasonCode.NESTING_LIMIT: (400, False, "fix_request"),
    ReasonCode.UNSUPPORTED_TRANSFER_ENCODING: (400, False, "fix_request"),
    ReasonCode.INVALID_CONFIG: (422, False, "fix_config"),
    ReasonCode.UNSUPPORTED_SCHEMA: (426, False, "upgrade_client"),
    ReasonCode.UNAUTHORIZED: (401, False, "check_credential"),
    ReasonCode.INVALID_HOST: (403, False, "use_loopback_host"),
    ReasonCode.BROWSER_ORIGIN_REJECTED: (403, False, "remove_origin"),
    ReasonCode.REQUEST_TOO_LARGE: (413, False, "reduce_request"),
    ReasonCode.REQUEST_TIMEOUT: (408, True, "retry"),
    ReasonCode.RATE_LIMITED: (429, True, "retry_later"),
    ReasonCode.QUEUE_FULL: (429, True, "retry_later"),
    ReasonCode.CONFLICT: (409, False, "check_state"),
    ReasonCode.ALREADY_TERMINAL: (409, False, "none"),
    ReasonCode.CACHE_IN_USE: (409, True, "retry_later"),
    ReasonCode.SERVICE_STOPPED: (503, False, "start_service"),
    ReasonCode.SERVICE_UNAVAILABLE: (503, True, "retry_later"),
    ReasonCode.WRITER_UNRESOLVED: (503, True, "wait_for_writers"),
    ReasonCode.AUDIT_LOSS: (503, True, "retry_later"),
}
_CONFLICT_CODES = frozenset({
    ReasonCode.DESTINATION_CONFLICT, ReasonCode.SOURCE_CHANGED, ReasonCode.CACHE_UNKNOWN,
})
_INVALID_CODES = frozenset({
    ReasonCode.INPUT_REJECTED, ReasonCode.MEDIA_PROBE_FAILED, ReasonCode.MEDIA_LIMIT_REACHED,
    ReasonCode.NO_AUDIO, ReasonCode.NO_VIDEO, ReasonCode.DURATION_UNKNOWN,
    ReasonCode.DURATION_EXCEEDED, ReasonCode.ASSET_SOURCE_REJECTED,
    ReasonCode.ARTIFACT_REJECTED, ReasonCode.CONTENT_DENIED, ReasonCode.EXPORT_LIMIT_REACHED,
})


def _describe(reason: ReasonCode, not_found: bool = False) -> tuple[int, bool, str]:
    if not_found:
        return 404, False, "check_id"
    found = _REASONS.get(reason)
    if found is not None:
        return found
    if reason in _CONFLICT_CODES:
        return 409, False, "check_state"
    if reason in _INVALID_CODES:
        return 422, False, "fix_request"
    return 503, True, "retry_later"


def _error_reason(exc: BaseException) -> tuple[ReasonCode, bool]:
    reason = getattr(exc, "reason_code", None)
    if not isinstance(reason, ReasonCode):
        return ReasonCode.SERVICE_UNAVAILABLE, False
    return reason, bool(getattr(exc, "not_found", False))


class _Route:
    __slots__ = ("handler", "method", "name", "params", "regex", "template")

    def __init__(self, method: str, template: str, name: str, handler: str) -> None:
        self.method, self.template, self.name, self.handler = method, template, name, handler
        self.params = tuple(re.findall(r"\{(\w+)\}", template))
        pattern = re.sub(r"\{(\w+)\}", r"(?P<\1>[^/]{1,128})", template)
        self.regex = re.compile(pattern)


_ROUTES = tuple(_Route(*row) for row in (
    ("GET", "/v1/identity", "identity", "identity"),
    ("GET", "/v1/snapshot", "snapshot", "snapshot"),
    ("GET", "/v1/jobs", "jobs.list", "list_jobs"),
    ("POST", "/v1/jobs", "jobs.submit", "submit"),
    ("GET", "/v1/jobs/{job_id}", "jobs.get", "get_job"),
    ("POST", "/v1/jobs/{job_id}/recheck", "jobs.recheck", "recheck"),
    ("POST", "/v1/jobs/{job_id}/retry", "jobs.retry", "retry"),
    ("POST", "/v1/jobs/{job_id}/clone", "jobs.clone", "clone"),
    ("POST", "/v1/jobs/{job_id}/cancel", "jobs.cancel", "cancel"),
    ("GET", "/v1/jobs/{job_id}/artifacts/{artifact_id}", "artifacts.get", "artifact"),
    ("GET", "/v1/cache", "cache.list", "list_cache"),
    ("POST", "/v1/cache/remove", "cache.remove", "remove_cache"),
    ("POST", "/v1/setup/plan", "setup.plan", "setup_plan"),
    ("POST", "/v1/setup/apply", "setup.apply", "setup_apply"),
    ("GET", "/v1/setup/{operation_id}", "setup.status", "setup_status"),
    ("POST", "/v1/service/stop", "service.stop", "stop"),
))
_ROUTE_NAMES = frozenset(route.name for route in _ROUTES)


def _classify(method: str, path: str) -> tuple[_Route | None, dict[str, str], bool]:
    """Return (route, params, path_known). Never raises; params are raw segments."""
    path_known = False
    for route in _ROUTES:
        found = route.regex.fullmatch(path)
        if found is None:
            continue
        path_known = True
        if route.method == method:
            return route, found.groupdict(), True
    return None, {}, path_known


def _audit_ids(params: Mapping[str, str]) -> dict[str, str]:
    """Only strict UUID path segments may reach the audit sink."""
    return {key: value.lower() for key, value in params.items() if _UUID_RE.fullmatch(value)}


class _TokenBucket:
    def __init__(self, rate: float, burst: int, clock: Callable[[], float]) -> None:
        self._rate, self._burst, self._clock = rate, float(burst), clock
        self._tokens = float(burst)
        self._last = clock()

    def take(self) -> bool:
        now = self._clock()
        self._tokens = min(self._burst, self._tokens + max(0.0, now - self._last) * self._rate)
        self._last = now
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False


@dataclass
class _Head:
    method: str
    target: str
    path: str
    query: str
    headers: dict[str, str]
    content_length: int
    keep_alive: bool


def _framing_error(reason: ReasonCode = ReasonCode.INVALID_INPUT) -> ApiActionError:
    return ApiActionError(reason)


def _parse_head(raw: bytes, limits: ApiLimits) -> _Head:
    """Strict HTTP/1.1 request-head parser: ASCII, CRLF only, no obs-fold."""
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError:
        raise _framing_error() from None
    if not text.endswith("\r\n\r\n"):
        raise _framing_error()
    lines = text[:-4].split("\r\n")
    if any("\r" in line or "\n" in line for line in lines):
        raise _framing_error()
    parts = lines[0].split(" ")
    if len(parts) != 3 or parts[2] != "HTTP/1.1" or parts[0] not in ("GET", "POST"):
        raise _framing_error()
    method, target = parts[0], parts[1]
    if len(target) > MAX_TARGET_BYTES or _TARGET_RE.fullmatch(target) is None:
        raise _framing_error()
    pairs: list[tuple[str, str]] = []
    for line in lines[1:]:
        if line[:1] in (" ", "\t"):
            raise _framing_error()
        name, sep, value = line.partition(":")
        if not sep or _HEADER_NAME_RE.fullmatch(name) is None:
            raise _framing_error()
        value = value.strip(" \t")
        if _HEADER_VALUE_RE.fullmatch(value) is None:
            raise _framing_error()
        pairs.append((name.lower(), value))
    if any(name == "transfer-encoding" for name, _ in pairs):
        raise _framing_error(ReasonCode.UNSUPPORTED_TRANSFER_ENCODING)
    headers: dict[str, str] = {}
    for name, value in pairs:
        if name in headers:
            raise _framing_error()  # duplicates (incl. multiple Content-Length) are ambiguous
        headers[name] = value
    length = 0
    if "content-length" in headers:
        value = headers["content-length"]
        if re.fullmatch(r"[0-9]{1,10}", value) is None:
            raise _framing_error()
        length = int(value)
        if length > limits.body_bytes:
            raise ApiActionError(ReasonCode.REQUEST_TOO_LARGE)
    path, _, query = target.partition("?")
    keep_alive = headers.get("connection", "").lower() != "close"
    return _Head(method, target, path, query, headers, length, keep_alive)


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=True, allow_nan=False, separators=(",", ":")).encode("ascii")


def _head_bytes(status: int, headers: list[tuple[str, str]]) -> bytes:
    phrase = http.HTTPStatus(status).phrase
    lines = [f"HTTP/1.1 {status} {phrase}"] + [f"{k}: {v}" for k, v in headers]
    return ("\r\n".join(lines) + "\r\n\r\n").encode("ascii")


def _public(value: Any) -> Any:
    method = getattr(value, "public_dict", None)
    return method() if callable(method) else value


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ApiActionError(ReasonCode.INVALID_INPUT)
        result[key] = value
    return result


def _reject_constant(_: str) -> Any:
    raise ApiActionError(ReasonCode.INVALID_INPUT)


class _MemoryTransport(asyncio.Transport):
    """In-memory write sink for aiohttp's response; bounded by write back pressure."""

    def __init__(self, high: int) -> None:
        super().__init__(extra={})
        self._chunks: deque[bytes] = deque()
        self._size = 0
        self._high = high
        self._closing = False
        self._paused = False
        self._wake = asyncio.Event()
        self._protocol: asyncio.BaseProtocol | None = None

    def set_protocol(self, protocol: asyncio.BaseProtocol) -> None:
        self._protocol = protocol

    def get_protocol(self) -> asyncio.BaseProtocol | None:
        return self._protocol

    def write(self, data: bytes | bytearray | memoryview) -> None:
        if self._closing or not data:
            return
        chunk = bytes(data)
        self._chunks.append(chunk)
        self._size += len(chunk)
        self._wake.set()
        if self._size > self._high and not self._paused and self._protocol is not None:
            self._paused = True
            self._protocol.pause_writing()  # type: ignore[attr-defined]

    def writelines(self, chunks: Any) -> None:
        for chunk in chunks:
            self.write(chunk)

    def write_eof(self) -> None:
        self.close()

    def can_write_eof(self) -> bool:
        return False

    def is_closing(self) -> bool:
        return self._closing

    def close(self) -> None:
        self._closing = True
        self._wake.set()

    def abort(self) -> None:
        self.close()

    def get_write_buffer_size(self) -> int:
        return self._size

    @property
    def drained_and_closed(self) -> bool:
        return self._closing and not self._chunks

    async def read(self, limit: int) -> bytes:
        """Return up to ``limit`` bytes, or b"" once closed and fully drained."""
        while not self._chunks:
            if self._closing:
                return b""
            self._wake.clear()
            await self._wake.wait()
        chunk = self._chunks.popleft()
        if len(chunk) > limit:
            self._chunks.appendleft(chunk[limit:])
            chunk = chunk[:limit]
        self._size -= len(chunk)
        if self._paused and self._size <= self._high // 4 and self._protocol is not None:
            self._paused = False
            self._protocol.resume_writing()  # type: ignore[attr-defined]
        return chunk


class _Stalled(Exception):
    """A client write or response producer made no progress in time."""


class _Handlers:
    """aiohttp handlers. They only run for requests the front door authenticated."""

    def __init__(self, server: ApiServer) -> None:
        self._s = server

    # -- helpers -------------------------------------------------------------
    @property
    def _actions(self) -> ServiceActions:
        return self._s._actions

    def _ok(self, status: int, payload: Mapping[str, Any]) -> web.Response:
        body = _json_bytes({"schema_version": SCHEMA_VERSION,
                            "instance_nonce": self._s._nonce, **payload})
        return web.Response(body=body, status=status, headers={
            "Content-Type": "application/json", _REASON_HEADER: ReasonCode.OK.value})

    def _uuid(self, value: str) -> str:
        return validate_uuid_string(value.lower() if _UUID_RE.fullmatch(value) else value).lower()

    def _key(self, request: web.Request) -> str:
        value = request.headers.get("Idempotency-Key")
        if value is None:
            raise ApiActionError(ReasonCode.INVALID_INPUT)
        return self._uuid(value)

    async def _body(self, request: web.Request) -> Any:
        raw = await request.read()
        if len(raw) > self._s._limits.body_bytes:
            raise ApiActionError(ReasonCode.REQUEST_TOO_LARGE)
        try:
            data = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_pairs,
                              parse_constant=_reject_constant)
        except RecursionError:
            raise ApiActionError(ReasonCode.NESTING_LIMIT) from None
        except (UnicodeDecodeError, ValueError):
            raise ApiActionError(ReasonCode.INVALID_INPUT) from None
        limits = self._s._limits
        validate_json_payload(data, max_depth=limits.max_depth, max_string_bytes=limits.max_string_bytes)
        return data

    async def _object(self, request: web.Request, keys: frozenset[str]) -> dict[str, Any]:
        data = await self._body(request)
        if type(data) is not dict or set(data) != keys:
            raise ApiActionError(ReasonCode.INVALID_INPUT)
        return data

    def _no_query(self, request: web.Request) -> None:
        if request.query_string:
            raise ApiActionError(ReasonCode.INVALID_INPUT)

    def _page(self, request: web.Request) -> tuple[int, str | None]:
        try:
            pairs = parse_qsl(request.query_string, keep_blank_values=True, strict_parsing=bool(request.query_string))
        except ValueError:
            raise ApiActionError(ReasonCode.INVALID_INPUT) from None
        names = [name for name, _ in pairs]
        if len(names) != len(set(names)) or not set(names) <= {"limit", "cursor"}:
            raise ApiActionError(ReasonCode.INVALID_INPUT)
        values = dict(pairs)
        limit = self._s._limits.max_page
        if "limit" in values:
            if _LIMIT_RE.fullmatch(values["limit"]) is None:
                raise ApiActionError(ReasonCode.INVALID_INPUT)
            limit = int(values["limit"])
            if not 1 <= limit <= self._s._limits.max_page:
                raise ApiActionError(ReasonCode.INVALID_INPUT)
        cursor = values.get("cursor")
        if cursor is not None and _CURSOR_RE.fullmatch(cursor) is None:
            raise ApiActionError(ReasonCode.INVALID_INPUT)
        return limit, cursor

    def _registries(self) -> RegistryPair:
        """Verified (models, languages) for this request; (None, None) keeps auto-only."""
        provider = self._s._registries
        if provider is None:
            return None, None
        try:
            models, languages = provider()
        except Exception:  # noqa: BLE001 - registry fault is the service's, never a 500
            _LOG.warning("api registry provider failed")
            raise ApiActionError(ReasonCode.SERVICE_UNAVAILABLE) from None
        if any(item is not None and type(item) not in (set, frozenset) for item in (models, languages)):
            _LOG.warning("api registry provider returned an invalid shape")
            raise ApiActionError(ReasonCode.SERVICE_UNAVAILABLE)
        return models, languages

    def _job_request(self, data: Any) -> JobRequest:
        models, languages = self._registries()
        return JobRequest.from_dict(data, verified_models=models, verified_languages=languages)

    def _next_cursor(self, value: Any) -> str | None:
        if value is not None and (type(value) is not str or _CURSOR_RE.fullmatch(value) is None):
            raise ApiActionError(ReasonCode.SERVICE_UNAVAILABLE)
        return value

    # -- routes --------------------------------------------------------------
    async def identity(self, request: web.Request) -> web.Response:
        self._no_query(request)
        info = await self._actions.identity()
        state = info["service_state"]
        return self._ok(200, {"pid": int(info["pid"]), "service_state": str(getattr(state, "value", state))})

    async def snapshot(self, request: web.Request) -> web.Response:
        self._no_query(request)
        return self._ok(200, {"snapshot": _public(await self._actions.snapshot())})

    async def list_jobs(self, request: web.Request) -> web.Response:
        limit, cursor = self._page(request)
        jobs, next_cursor = await self._actions.list_jobs(limit, cursor)
        return self._ok(200, {"jobs": [_public(job) for job in jobs[:limit]],
                              "next_cursor": self._next_cursor(next_cursor)})

    async def get_job(self, request: web.Request) -> web.Response:
        self._no_query(request)
        job = await self._actions.get_job(self._uuid(request.match_info["job_id"]))
        return self._ok(200, {"job": _public(job)})

    async def submit(self, request: web.Request) -> web.Response:
        self._no_query(request)
        key = self._key(request)
        body = await self._object(request, frozenset({"request"}))
        job, replayed = await self._actions.submit(self._job_request(body["request"]), key)
        return self._ok(200 if replayed else 201, {"job": _public(job), "replayed": bool(replayed)})

    async def _empty(self, request: web.Request) -> None:
        self._no_query(request)
        await self._object(request, frozenset())

    async def recheck(self, request: web.Request) -> web.Response:
        job_id = self._uuid(request.match_info["job_id"])
        await self._empty(request)
        job, preflight, ready = await self._actions.recheck(job_id)
        return self._ok(200, {"job": _public(job), "preflight": _public(preflight),
                              "ready_to_retry": bool(ready)})

    async def retry(self, request: web.Request) -> web.Response:
        job_id = self._uuid(request.match_info["job_id"])
        key = self._key(request)
        await self._empty(request)
        job, attempt_id, replayed = await self._actions.retry(job_id, key)
        return self._ok(200 if replayed else 202, {"job": _public(job), "attempt_id": attempt_id,
                                                   "replayed": bool(replayed)})

    async def clone(self, request: web.Request) -> web.Response:
        job_id = self._uuid(request.match_info["job_id"])
        self._no_query(request)
        key = self._key(request)
        body = await self._object(request, frozenset({"request"}))
        job = await self._actions.clone(job_id, self._job_request(body["request"]), key)
        return self._ok(201, {"job": _public(job)})

    async def cancel(self, request: web.Request) -> web.Response:
        job_id = self._uuid(request.match_info["job_id"])
        await self._empty(request)
        job, status = await self._actions.cancel(job_id)
        if status not in ("intent", "cancelled", "already_terminal"):
            raise ApiActionError(ReasonCode.SERVICE_UNAVAILABLE)
        return self._ok(202 if status == "intent" else 200, {"job": _public(job), "status": status})

    async def artifact(self, request: web.Request) -> web.StreamResponse:
        self._no_query(request)
        job_id = self._uuid(request.match_info["job_id"])
        artifact_id = self._uuid(request.match_info["artifact_id"])
        size, content_type, chunks = await self._actions.open_artifact(job_id, artifact_id)
        if type(size) is not int or size < 0 or _CONTENT_TYPE_RE.fullmatch(str(content_type)) is None:
            await _aclose(chunks)
            raise ApiActionError(ReasonCode.SERVICE_UNAVAILABLE)
        response = web.StreamResponse(status=200, headers={
            "Content-Type": content_type, "X-Podmuncher-Schema": str(SCHEMA_VERSION),
            "X-Podmuncher-Instance": self._s._nonce, _REASON_HEADER: ReasonCode.OK.value})
        response.content_length = size
        await response.prepare(request)
        request[_STREAMING] = True
        sent = 0
        step = self._s._limits.chunk_bytes
        try:
            async for chunk in chunks:
                for start in range(0, len(chunk), step):
                    piece = chunk[start:start + step]
                    sent += len(piece)
                    if sent > size:
                        raise ValueError("artifact longer than declared")
                    await response.write(piece)
            if sent != size:
                raise ValueError("artifact shorter than declared")
            await response.write_eof()
        except Exception:  # noqa: BLE001 - producer failure after headers: truncate
            # Headers are out; the only honest signal is a truncated stream.
            if request.transport is not None:
                request.transport.close()
        finally:
            await _aclose(chunks)
        return response

    async def list_cache(self, request: web.Request) -> web.Response:
        limit, cursor = self._page(request)
        entries, next_cursor = await self._actions.list_cache(limit, cursor)
        return self._ok(200, {"entries": [_public(entry) for entry in entries[:limit]],
                              "next_cursor": self._next_cursor(next_cursor)})

    async def remove_cache(self, request: web.Request) -> web.Response:
        self._no_query(request)
        body = await self._object(request, frozenset({"entry_id"}))
        entry_id = body["entry_id"]
        if type(entry_id) is not str or not 1 <= len(entry_id) <= 256:
            raise ApiActionError(ReasonCode.INVALID_INPUT)
        await self._actions.remove_cache(entry_id)
        return self._ok(200, {"removed": True, "entry_id": entry_id})

    async def setup_plan(self, request: web.Request) -> web.Response:
        self._no_query(request)
        body = await self._object(request, frozenset({"capabilities", "config"}))
        raw_caps, config = body["capabilities"], body["config"]
        if type(raw_caps) is not list or type(config) is not dict or len(raw_caps) > 32:
            raise ApiActionError(ReasonCode.INVALID_INPUT)
        capabilities = [Capability.parse(item) for item in raw_caps]
        return self._ok(200, {"plan": _public(await self._actions.setup_plan(capabilities, config))})

    async def setup_apply(self, request: web.Request) -> web.Response:
        self._no_query(request)
        key = self._key(request)
        body = await self._object(request, frozenset({"plan_id"}))
        plan_id = body["plan_id"]
        if type(plan_id) is not str or not 1 <= len(plan_id) <= 256:
            raise ApiActionError(ReasonCode.INVALID_INPUT)
        return self._ok(202, {"operation": _public(await self._actions.setup_apply(plan_id, key))})

    async def setup_status(self, request: web.Request) -> web.Response:
        self._no_query(request)
        status = await self._actions.setup_status(self._uuid(request.match_info["operation_id"]))
        payload = {k: _public(v) for k, v in status.items() if k not in ("schema_version", "instance_nonce")}
        return self._ok(200, payload)

    async def stop(self, request: web.Request) -> web.Response:
        await self._empty(request)
        ok, _reason = await self._actions.stop()
        if not ok:
            raise ApiActionError(ReasonCode.WRITER_UNRESOLVED)
        self._s._stopping = True
        self._s._stop_pending = True
        return self._ok(200, {"state": "stopped"})


async def _aclose(chunks: Any) -> None:
    closer = getattr(chunks, "aclose", None)
    if closer is not None:
        try:
            await closer()
        except Exception:  # noqa: BLE001 - best effort release of a contained handle
            _LOG.debug("artifact iterator close failed")


class ApiServer:
    """Loopback API owner: ``await start()`` returns the dynamic port."""

    def __init__(
        self,
        actions: ServiceActions,
        *,
        credential: str,
        instance_nonce: str,
        audit: AuditFn,
        clock: Callable[[], float] = time.monotonic,
        limits: ApiLimits | None = None,
        registries: RegistriesFn | None = None,
    ) -> None:
        if not credential or not instance_nonce:
            raise ValueError("credential and instance nonce are required")
        self._actions = actions
        self._registries = registries
        self._digest = hashlib.sha256(credential.encode("utf-8")).digest()
        self._nonce = instance_nonce
        self._audit = audit
        self._limits = limits or ApiLimits()
        self._bucket = _TokenBucket(self._limits.rate_per_second, self._limits.rate_burst, clock)
        self._server: asyncio.Server | None = None
        self._runner: web.AppRunner | None = None
        self._port = 0
        self._active = 0
        self._tasks: set[asyncio.Task[None]] = set()
        self._writers: set[asyncio.StreamWriter] = set()
        self._stopping = False
        self._stop_pending = False
        self.stop_requested = asyncio.Event()

    @property
    def port(self) -> int:
        return self._port

    @property
    def active_connections(self) -> int:
        return self._active

    def begin_stopping(self) -> None:
        """Refuse new requests with 503 service_stopped (idempotent; owner shutdown)."""
        self._stopping = True

    # -- lifecycle -----------------------------------------------------------
    def _build_app(self) -> web.Application:
        server = self

        @web.middleware
        async def errors(request: web.Request, handler: Any) -> web.StreamResponse:
            try:
                return await handler(request)
            except asyncio.CancelledError:
                raise
            except web.HTTPException as exc:
                reason = ReasonCode.INVALID_INPUT
                return server._error_response(reason, exc.status == 404)
            except Exception as exc:
                reason, not_found = _error_reason(exc)
                if request.get(_STREAMING):
                    raise
                if not isinstance(exc, (ApiActionError, DomainValidationError)) and \
                        getattr(exc, "reason_code", None) is None:
                    _LOG.warning("api handler failed type=%s", type(exc).__name__)
                return server._error_response(reason, not_found)

        app = web.Application(middlewares=[errors], client_max_size=self._limits.body_bytes)
        handlers = _Handlers(self)
        for route in _ROUTES:
            app.router.add_route(route.method, route.template, getattr(handlers, route.handler))
        return app

    def _error_body(self, reason: ReasonCode, not_found: bool = False) -> tuple[int, bytes]:
        status, retryable, action = _describe(reason, not_found)
        body = _json_bytes({"schema_version": SCHEMA_VERSION, "instance_nonce": self._nonce,
                            "error": {"reason_code": reason.value, "retryable": retryable,
                                      "next_action": action}})
        return status, body

    def _error_response(self, reason: ReasonCode, not_found: bool = False) -> web.Response:
        status, body = self._error_body(reason, not_found)
        return web.Response(body=body, status=status, headers={
            "Content-Type": "application/json", _REASON_HEADER: reason.value})

    async def start(self) -> int:
        if self._server is not None:
            raise RuntimeError("already started")
        self._runner = web.AppRunner(self._build_app(), access_log=None, handle_signals=False,
                                     max_line_size=self._limits.header_bytes,
                                     max_field_size=self._limits.header_bytes)
        await self._runner.setup()
        self._server = await asyncio.start_server(
            self._serve, host="127.0.0.1", port=0, start_serving=False)
        self._port = self._server.sockets[0].getsockname()[1]
        await self._server.start_serving()
        return self._port

    async def close(self) -> None:
        server, self._server = self._server, None
        if server is not None:
            server.close()
        for writer in list(self._writers):
            writer.transport.abort()
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if server is not None:
            await server.wait_closed()
        runner, self._runner = self._runner, None
        if runner is not None:
            await runner.cleanup()

    # -- audit ---------------------------------------------------------------
    async def _record(self, route: str, reason: ReasonCode | str, ids: Mapping[str, str]) -> None:
        code = reason.value if isinstance(reason, ReasonCode) else str(reason)
        _LOG.debug("api route=%s outcome=%s", route, code)
        try:
            result = self._audit(route, code, dict(ids))
            if inspect.isawaitable(result):
                await result
        except Exception:  # noqa: BLE001 - audit sink failure must not echo or crash
            _LOG.warning("api audit sink failed route=%s", route)

    # -- front door ----------------------------------------------------------
    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if self._active >= self._limits.max_connections:
            writer.transport.abort()
            return
        self._active += 1
        task = asyncio.current_task()
        if task is not None:
            self._tasks.add(task)
        self._writers.add(writer)
        try:
            await self._connection(reader, writer)
        except (_Stalled, ConnectionError, OSError, asyncio.IncompleteReadError):
            writer.transport.abort()
        finally:
            self._active -= 1
            self._writers.discard(writer)
            if task is not None:
                self._tasks.discard(task)
            writer.close()

    async def _read_head(self, reader: asyncio.StreamReader, buf: bytearray) -> bytes | None:
        limits = self._limits
        step = 4096
        if not buf:
            try:
                async with asyncio.timeout(limits.keepalive_idle_seconds):
                    data = await reader.read(step)
            except TimeoutError:
                return None
            if not data:
                return None
            buf += data
        line_checked = False
        try:
            async with asyncio.timeout(limits.read_deadline_seconds):
                while True:
                    end = buf.find(b"\r\n\r\n", 0, limits.header_bytes)
                    # Only head bytes are checked: body bytes that arrived in the same
                    # read (pretty-printed JSON) may legitimately contain bare LF.
                    if _BARE_LF_RE.search(buf, 0, end + 4 if end >= 0 else len(buf)) is not None:
                        raise ApiActionError(ReasonCode.INVALID_INPUT)
                    if not line_checked:
                        eol = buf.find(b"\r\n")
                        if eol >= 0:
                            if _REQUEST_LINE_RE.fullmatch(bytes(buf[:eol])) is None:
                                raise ApiActionError(ReasonCode.INVALID_INPUT)
                            line_checked = True
                        elif len(buf) > MAX_TARGET_BYTES + 32:
                            raise ApiActionError(ReasonCode.INVALID_INPUT)
                    if end >= 0:
                        head = bytes(buf[:end + 4])
                        del buf[:end + 4]
                        return head
                    if len(buf) >= limits.header_bytes:
                        raise ApiActionError(ReasonCode.REQUEST_TOO_LARGE)
                    data = await reader.read(step)
                    if not data:
                        return None
                    buf += data
        except TimeoutError:
            raise ApiActionError(ReasonCode.REQUEST_TIMEOUT) from None

    async def _read_body(self, reader: asyncio.StreamReader, buf: bytearray, length: int) -> bytes:
        try:
            async with asyncio.timeout(self._limits.read_deadline_seconds):
                while len(buf) < length:
                    data = await reader.read(min(65536, length - len(buf)))
                    if not data:
                        raise _Stalled()
                    buf += data
        except TimeoutError:
            raise ApiActionError(ReasonCode.REQUEST_TIMEOUT) from None
        body = bytes(buf[:length])
        del buf[:length]
        return body

    async def _write(self, writer: asyncio.StreamWriter, data: bytes) -> None:
        if writer.transport.is_closing():
            raise _Stalled()
        writer.write(data)
        try:
            async with asyncio.timeout(self._limits.write_progress_seconds):
                await writer.drain()
        except TimeoutError:
            raise _Stalled() from None

    async def _send_error(self, writer: asyncio.StreamWriter, error: ApiActionError, keep: bool) -> None:
        status, body = self._error_body(error.reason_code, error.not_found)
        head = _head_bytes(status, self._json_headers(len(body), keep))
        await self._write(writer, head + body)

    @staticmethod
    def _json_headers(length: int, keep: bool) -> list[tuple[str, str]]:
        return [("Content-Type", "application/json"), ("Content-Length", str(length)),
                ("Cache-Control", "no-store"), ("X-Content-Type-Options", "nosniff"),
                ("Connection", "keep-alive" if keep else "close")]

    async def _linger(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Half-close, then discard a bounded amount of unread input.

        Closing with unread bytes makes Windows send RST, which can destroy an
        error response the peer has not read yet.
        """
        try:
            writer.write_eof()
            remaining = LINGER_BYTES
            async with asyncio.timeout(self._limits.linger_seconds):
                while remaining > 0:
                    data = await reader.read(4096)
                    if not data:
                        return
                    remaining -= len(data)
        except (TimeoutError, OSError, RuntimeError):
            return

    async def _connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        buf = bytearray()
        while True:
            keep = await self._request(reader, writer, buf)
            if not keep:
                await self._linger(reader, writer)
                return

    def _check_credential(self, header: str | None) -> bool:
        presented = b""
        if header is not None and len(header) <= MAX_TOKEN_CHARS:
            scheme, sep, token = header.partition(" ")
            if sep and scheme.lower() == "bearer" and token and " " not in token:
                presented = token.encode("utf-8", "replace")
        digest = hashlib.sha256(presented).digest()
        return hmac.compare_digest(digest, self._digest) and bool(presented)

    async def _request(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter,
                       buf: bytearray) -> bool:
        """Serve one request; return True to keep the connection."""
        route_name, ids = "unknown", {}
        try:
            raw = await self._read_head(reader, buf)
            if raw is None:
                return False
            head = _parse_head(raw, self._limits)
            route, params, path_known = _classify(head.method, head.path)
            if route is not None:
                route_name, ids = route.name, _audit_ids(params)
            if not self._bucket.take():
                raise ApiActionError(ReasonCode.RATE_LIMITED)
            if head.headers.get("host") != f"127.0.0.1:{self._port}":
                raise ApiActionError(ReasonCode.INVALID_HOST)
            if "origin" in head.headers:
                raise ApiActionError(ReasonCode.BROWSER_ORIGIN_REJECTED)
            if head.headers.get("x-podmuncher-schema") != str(SCHEMA_VERSION):
                raise ApiActionError(ReasonCode.UNSUPPORTED_SCHEMA)
            if not self._check_credential(head.headers.get("authorization")):
                raise ApiActionError(ReasonCode.UNAUTHORIZED)
            if route is None:
                raise ApiActionError(ReasonCode.INVALID_INPUT, not_found=not path_known) \
                    if not path_known else ApiActionError(ReasonCode.INVALID_INPUT)
            if self._stopping:
                raise ApiActionError(ReasonCode.SERVICE_STOPPED)
            if route.method == "GET" and head.content_length:
                raise ApiActionError(ReasonCode.INVALID_INPUT)
        except ApiActionError as error:
            await self._record(route_name, error.reason_code, ids)
            await self._send_error(writer, error, keep=False)
            return False
        body = b""
        try:
            if head.content_length:
                body = await self._read_body(reader, buf, head.content_length)
            if buf and buf[:1] not in (b"G", b"P"):
                raise ApiActionError(ReasonCode.INVALID_INPUT)
        except ApiActionError as error:
            await self._record(route_name, error.reason_code, ids)
            await self._send_error(writer, error, keep=False)
            return False
        return await self._forward(writer, head, route, ids, body)

    async def _forward(self, writer: asyncio.StreamWriter, head: _Head, route: _Route,
                       ids: Mapping[str, str], body: bytes) -> bool:
        assert self._runner is not None and self._runner.server is not None
        lines = [f"{head.method} {head.target} HTTP/1.1", f"Host: 127.0.0.1:{self._port}",
                 "Connection: close", f"Content-Length: {len(body)}"]
        if head.method == "POST":
            lines.append("Content-Type: application/json")
        key = head.headers.get("idempotency-key")
        if key is not None:
            lines.append(f"Idempotency-Key: {key}")
        request_bytes = ("\r\n".join(lines) + "\r\n\r\n").encode("ascii") + body
        transport = _MemoryTransport(self._limits.chunk_bytes)
        handler = self._runner.server()
        transport.set_protocol(handler)
        handler.connection_made(transport)
        try:
            return await self._relay(writer, head, route, ids, transport, handler, request_bytes)
        finally:
            # A stop the service accepted must reach the owner even when the response
            # write fails (client reset, stall) or this task is cancelled; otherwise the
            # daemon answers service_stopped forever while holding the singleton lock.
            if route.name == "service.stop" and self._stop_pending:
                self._stop_pending = False
                self.stop_requested.set()

    async def _relay(self, writer: asyncio.StreamWriter, head: _Head, route: _Route,
                     ids: Mapping[str, str], transport: _MemoryTransport, handler: Any,
                     request_bytes: bytes) -> bool:
        try:
            handler.data_received(request_bytes)
            status, headers, length, rest = await self._read_response_head(transport)
            reason = headers.pop(_REASON_HEADER.lower(), ReasonCode.OK.value)
            await self._record(route.name, reason, ids)
            keep = head.keep_alive and not (route.name == "service.stop" and self._stop_pending)
            out = [("Content-Type", headers.get("content-type", "application/json")),
                   ("Content-Length", str(length)), ("Cache-Control", "no-store"),
                   ("X-Content-Type-Options", "nosniff")]
            for name in ("x-podmuncher-schema", "x-podmuncher-instance"):
                if name in headers:
                    out.append(("-".join(p.capitalize() for p in name.split("-")), headers[name]))
            out.append(("Connection", "keep-alive" if keep else "close"))
            await self._write(writer, _head_bytes(status, out))
            sent = 0
            while True:
                if rest:
                    chunk, rest = rest, b""
                else:
                    try:
                        async with asyncio.timeout(self._limits.write_progress_seconds):
                            chunk = await transport.read(self._limits.chunk_bytes)
                    except TimeoutError:
                        raise _Stalled() from None
                if not chunk:
                    break
                sent += len(chunk)
                if sent > length:
                    raise _Stalled()
                await self._write(writer, chunk)
                if sent == length:
                    break
            if sent != length:
                raise _Stalled()  # producer truncated: drop the client connection
        finally:
            transport.close()
            handler.connection_lost(None)
            await handler.shutdown(0)
        return keep

    async def _read_response_head(self, transport: _MemoryTransport) -> tuple[int, dict[str, str], int, bytes]:
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = await transport.read(self._limits.chunk_bytes)
            if not chunk or len(data) > self._limits.header_bytes:
                raise _Stalled()
            data += chunk
        raw, _, rest = data.partition(b"\r\n\r\n")
        lines = raw.decode("latin-1").split("\r\n")
        status = int(lines[0].split(" ", 2)[1])
        headers: dict[str, str] = {}
        for line in lines[1:]:
            name, _, value = line.partition(":")
            headers[name.strip().lower()] = value.strip()
        if "content-length" not in headers or "transfer-encoding" in headers:
            raise _Stalled()
        return status, headers, int(headers["content-length"]), rest


__all__ = (
    "BODY_LIMIT_BYTES",
    "HEADER_LIMIT_BYTES",
    "KEEPALIVE_IDLE_SECONDS",
    "MAX_CONNECTIONS",
    "MAX_JSON_DEPTH",
    "MAX_STRING_BYTES",
    "PAGE_MAX",
    "RATE_BURST",
    "RATE_PER_SECOND",
    "READ_DEADLINE_SECONDS",
    "WRITE_PROGRESS_SECONDS",
    "ApiActionError",
    "ApiLimits",
    "ApiServer",
    "AuditFn",
    "ServiceActions",
)
