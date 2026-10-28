"""Detached per-user service lifecycle and the shared loopback API client.

Phase 5 unit P5-U2 (Docs/spec.md:1341).  This module is imported by the CLI and TUI
and by the detached serving process itself (``python -m podmuncher.service --serve``).

Rules enforced here:

* ``status``/``jobs``/``doctor`` use ``discover``/``read_status`` only and never launch.
  Start-capable actions call ``start_service`` which preflights the state volume first.
* The serving process takes the singleton lock, builds the domain composition through an
  injected factory seam, generates the instance nonce and credential in process, starts the
  loopback API and only then writes the private runtime file.  The credential travels in
  the runtime file and in request headers only: never argv, URLs, logs or terminal output.
* ``ServiceClient`` verifies the instance nonce of every response, retries only
  connection errors for GETs and keyed POSTs, and ``close()`` never cancels server-side
  work: a client that goes away leaves the job running.
* ``stop_service`` succeeds only after the serving process is gone; an unresolved writer
  is a failure and the service keeps running with its reservation.

``config``, ``api``, ``recovery``, ``volumes`` and the bootstrap factory are imported inside
functions so that importing this module stays free of aiohttp, psutil, platformdirs,
opentelemetry and textual.  No OS boot service is created and no firewall is touched.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import http.client
import importlib
import inspect
import json
import logging
import os
import secrets
import signal
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol
from urllib.parse import quote, urlencode

from .domains import ReasonCode

if TYPE_CHECKING:
    from .recovery import RuntimeRecord

SCHEMA_VERSION = 1
SCHEMA_HEADER = "X-Podmuncher-Schema"
INSTANCE_HEADER = "X-Podmuncher-Instance"
DEFAULT_FACTORY = "podmuncher.bootstrap:make_service"
SNAPSHOT_FILE = "snapshot.json"
CLIENT_TIMEOUT_SECONDS = 10.0
RECONNECT_BACKOFF = (0.5, 1.0, 2.0)
START_TIMEOUT_SECONDS = 15.0
STOP_TIMEOUT_SECONDS = 30.0
POLL_SECONDS = 0.1
EXITED_GRACE_SECONDS = 2.0
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
DOWNLOAD_CHUNK_BYTES = 64 * 1024
EXIT_OK = 0
EXIT_SHUTDOWN_FAILED = 3
EXIT_UNAVAILABLE = 4
EXIT_CONFLICT = 5
EXIT_USAGE = 2
NOFILE_TARGET = 8192  # macOS defaults to a soft limit of 256, below what the stack opens

_LOG = logging.getLogger("podmuncher.service")

# Popen objects of services this process launched, so stop can reap them (no zombies).
_CHILDREN: dict[int, subprocess.Popen] = {}


# ----------------------------------------------------------------------------- errors


class ServiceError(Exception):
    """Closed-reason lifecycle failure.  The message never carries paths or credentials."""

    def __init__(self, reason_code: ReasonCode, message: str | None = None) -> None:
        self.reason_code = reason_code
        super().__init__(message or reason_code.value)


class ClientError(ServiceError):
    """Failure of an API call, mapped from the error envelope or the transport."""

    def __init__(self, reason_code: ReasonCode, *, retryable: bool = False,
                 next_action: str = "none", identity_mismatch: bool = False,
                 status: int | None = None) -> None:
        super().__init__(reason_code)
        self.retryable = retryable
        self.next_action = next_action
        self.identity_mismatch = identity_mismatch
        self.status = status


# ---------------------------------------------------------------------------- factory seam


class ServiceComposition(Protocol):
    """What the serving process needs from the domain layer (built by phase 6)."""

    actions: Any  # api.ServiceActions
    audit: Callable[[str, str, Mapping[str, str]], Any]

    async def startup(self) -> None: ...
    async def shutdown(self) -> tuple[bool, str]: ...


class ServiceFactory(Protocol):
    def __call__(self, state_dir: str) -> ServiceComposition: ...


def resolve_factory(spec: str | None = None, import_path: str | None = None) -> ServiceFactory:
    """Import ``module:attr`` at serve time.  Failure is a closed SERVICE_UNAVAILABLE."""
    text = spec or DEFAULT_FACTORY
    module_name, sep, attr = text.partition(":")
    if not sep or not module_name or not attr:
        raise ServiceError(ReasonCode.INVALID_CONFIG)
    if import_path:
        sys.path.insert(0, import_path)  # test factories only; production never passes it
    try:
        factory = getattr(importlib.import_module(module_name), attr)
    except (ImportError, AttributeError):
        raise ServiceError(ReasonCode.SERVICE_UNAVAILABLE) from None
    if not callable(factory):
        raise ServiceError(ReasonCode.SERVICE_UNAVAILABLE)
    return factory


# ------------------------------------------------------------------------- state directory


def default_state_dir() -> str:
    import platformdirs

    return platformdirs.user_state_dir("podmuncher", appauthor=False)


def _state(state_dir: str | os.PathLike | None) -> str:
    return os.fspath(state_dir) if state_dir is not None else default_state_dir()


def _runtime_path(state_dir: str) -> str:
    from .recovery import RUNTIME_FILE

    return os.path.join(state_dir, RUNTIME_FILE)


def preflight_state_path(state_dir: str | os.PathLike, *, volume_provider: Any = None) -> None:
    """The state volume must be provably local before anything is launched or created."""
    if volume_provider is None:
        from .volumes import make_volume_provider

        volume_provider = make_volume_provider()
    try:
        evidence = volume_provider.sample(os.fspath(state_dir), sqlite_state=True)
    except Exception as error:
        raise ServiceError(ReasonCode.STORAGE_UNKNOWN) from error
    allowed = evidence.state_allowed.value
    if allowed is True:
        return
    raise ServiceError(ReasonCode.INVALID_CONFIG if allowed is False else ReasonCode.STORAGE_UNKNOWN)


# ----------------------------------------------------------------------------- private files


def _write_private_file(path: str, payload: bytes) -> None:
    """Private file (0600 / current-user ACL, verified) through recovery's helper."""
    from . import recovery

    try:
        recovery._write_private(path, payload)
    except recovery.RecoveryError as error:
        raise ServiceError(error.reason_code) from None


def _read_private_file(path: str) -> bytes | None:
    from . import recovery

    try:
        return recovery._read_private(path)
    except recovery.RecoveryError:
        return None


def _public(value: Any) -> Any:
    method = getattr(value, "public_dict", None)
    return method() if callable(method) else value


def save_snapshot(state_dir: str | os.PathLike, snapshot: Any) -> bool:
    """Persist the last snapshot privately.  Best effort: returns False on any failure."""
    try:
        payload = json.dumps({"schema_version": SCHEMA_VERSION, "snapshot": _public(snapshot)},
                             sort_keys=True, allow_nan=False).encode("utf-8")
        _write_private_file(os.path.join(os.fspath(state_dir), SNAPSHOT_FILE), payload)
    except (ServiceError, TypeError, ValueError, OSError):
        return False
    return True


def persisted_snapshot(state_dir: str | os.PathLike | None = None) -> dict | None:
    """The protected persisted snapshot, or None when absent, unprotected or malformed."""
    raw = _read_private_file(os.path.join(_state(state_dir), SNAPSHOT_FILE))
    if not raw:
        return None
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    if type(doc) is not dict or doc.get("schema_version") != SCHEMA_VERSION:
        return None
    snapshot = doc.get("snapshot")
    return snapshot if type(snapshot) is dict else None


# ------------------------------------------------------------------------------ discovery


def discover(state_dir: str | os.PathLike | None = None, *, probe: Callable[[int], Any] | None = None,
             expected_nonce: str | None = None) -> RuntimeRecord | None:
    """The live runtime record or None.  Never launches.  Bad permissions count as no record."""
    from . import recovery

    directory = _state(state_dir)
    try:
        return recovery.read_runtime_file(
            _runtime_path(directory), process_identity=probe or recovery.psutil_identity,
            expected_nonce=expected_nonce)
    except recovery.RecoveryError:
        return None


# ------------------------------------------------------------------------------- client


class ServiceClient:
    """Stdlib HTTP client for the loopback API, shared by the CLI and the TUI."""

    def __init__(self, record: RuntimeRecord, *, timeout: float = CLIENT_TIMEOUT_SECONDS,
                 sleep: Callable[[float], None] = time.sleep,
                 connection_factory: Callable[..., Any] = http.client.HTTPConnection) -> None:
        self._port = record.port
        self._nonce = record.instance_nonce
        self._pid = record.pid
        self._credential = record.credential
        self._timeout = timeout
        self._sleep = sleep
        self._connection_factory = connection_factory
        self._open_connection: Any = None

    def __repr__(self) -> str:
        return f"ServiceClient(port={self._port}, instance_nonce={self._nonce!r})"

    @property
    def pid(self) -> int:
        return self._pid

    @property
    def instance_nonce(self) -> str:
        return self._nonce

    # -- transport -------------------------------------------------------------------
    def close(self) -> None:
        """Drop the local connection.  Nothing is cancelled server-side."""
        conn, self._open_connection = self._open_connection, None
        if conn is not None:
            with contextlib.suppress(Exception):
                conn.close()

    def _headers(self, body: bytes | None, key: str | None) -> dict[str, str]:
        headers = {"Host": f"127.0.0.1:{self._port}", SCHEMA_HEADER: str(SCHEMA_VERSION),
                   "Authorization": f"Bearer {self._credential}", "Connection": "close"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if key is not None:
            headers["Idempotency-Key"] = key
        return headers

    def _attempts(self, fn: Callable[[], Any], retryable: bool) -> Any:
        """Run ``fn``; only connection errors are retried, and only when ``retryable``."""
        delays = RECONNECT_BACKOFF if retryable else ()
        for attempt in range(len(delays) + 1):
            try:
                return fn()
            except (OSError, http.client.HTTPException):
                self.close()
                if attempt >= len(delays):
                    break
                self._sleep(delays[attempt])
        raise ClientError(ReasonCode.SERVICE_UNAVAILABLE, retryable=True,
                          next_action="retry_later") from None

    def _send(self, method: str, path: str, body: bytes | None, key: str | None) -> Any:
        self.close()
        conn = self._connection_factory("127.0.0.1", self._port, timeout=self._timeout)
        self._open_connection = conn
        conn.request(method, path, body=body, headers=self._headers(body, key))
        return conn.getresponse()

    def _call(self, method: str, path: str, payload: Mapping[str, Any] | None = None,
              key: str | None = None) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload, allow_nan=False).encode("utf-8")
        retryable = method == "GET" or key is not None

        def attempt() -> tuple[int, bytes]:
            response = self._send(method, path, body, key)
            data = response.read(MAX_RESPONSE_BYTES + 1)
            return response.status, data

        try:
            status, data = self._attempts(attempt, retryable)
        finally:
            self.close()
        return self._decode(status, data)

    def _check_nonce(self, doc: Any, status: int) -> dict[str, Any]:
        if type(doc) is not dict or doc.get("instance_nonce") != self._nonce \
                or doc.get("schema_version") != SCHEMA_VERSION:
            raise ClientError(ReasonCode.SERVICE_UNAVAILABLE, next_action="restart_client",
                              identity_mismatch=True, status=status)
        return doc

    def _decode(self, status: int, data: bytes) -> dict[str, Any]:
        try:
            doc = json.loads(data.decode("utf-8")) if len(data) <= MAX_RESPONSE_BYTES else None
        except (ValueError, UnicodeDecodeError):
            doc = None
        doc = self._check_nonce(doc, status)
        return self._result(status, doc)

    def _result(self, status: int, doc: dict[str, Any]) -> dict[str, Any]:
        if status == 426:
            raise ClientError(ReasonCode.UNSUPPORTED_SCHEMA, next_action="upgrade_client", status=status)
        if 200 <= status < 300:
            return doc
        error = doc.get("error")
        if type(error) is not dict:
            raise ClientError(ReasonCode.SERVICE_UNAVAILABLE, retryable=True,
                              next_action="retry_later", status=status)
        try:
            reason = ReasonCode.parse(error.get("reason_code"))
        except Exception:  # noqa: BLE001 - unknown reason is a closed unavailable
            reason = ReasonCode.SERVICE_UNAVAILABLE
        action = error.get("next_action")
        raise ClientError(reason, retryable=bool(error.get("retryable")),
                          next_action=action if type(action) is str else "none", status=status)

    # -- routes ----------------------------------------------------------------------
    def identity(self) -> dict[str, Any]:
        return self._call("GET", "/v1/identity")

    def snapshot(self) -> dict[str, Any]:
        return self._call("GET", "/v1/snapshot")

    @staticmethod
    def _page(limit: int | None, cursor: str | None) -> str:
        query = {}
        if limit is not None:
            query["limit"] = str(limit)
        if cursor is not None:
            query["cursor"] = cursor
        return f"?{urlencode(query)}" if query else ""

    def list_jobs(self, limit: int | None = None, cursor: str | None = None) -> dict[str, Any]:
        return self._call("GET", "/v1/jobs" + self._page(limit, cursor))

    def get_job(self, job_id: str) -> dict[str, Any]:
        return self._call("GET", f"/v1/jobs/{quote(job_id, safe='')}")

    def submit(self, request: Mapping[str, Any], key: str) -> dict[str, Any]:
        return self._call("POST", "/v1/jobs", {"request": dict(request)}, key)

    def recheck(self, job_id: str) -> dict[str, Any]:
        return self._call("POST", f"/v1/jobs/{quote(job_id, safe='')}/recheck", {})

    def retry(self, job_id: str, key: str) -> dict[str, Any]:
        return self._call("POST", f"/v1/jobs/{quote(job_id, safe='')}/retry", {}, key)

    def clone(self, job_id: str, request: Mapping[str, Any], key: str) -> dict[str, Any]:
        return self._call("POST", f"/v1/jobs/{quote(job_id, safe='')}/clone",
                          {"request": dict(request)}, key)

    def cancel(self, job_id: str) -> dict[str, Any]:
        """Explicit cancel.  Closing or dropping the client never calls this."""
        return self._call("POST", f"/v1/jobs/{quote(job_id, safe='')}/cancel", {})

    def list_cache(self, limit: int | None = None, cursor: str | None = None) -> dict[str, Any]:
        return self._call("GET", "/v1/cache" + self._page(limit, cursor))

    def remove_cache(self, entry_id: str) -> dict[str, Any]:
        return self._call("POST", "/v1/cache/remove", {"entry_id": entry_id})

    def setup_plan(self, capabilities: Sequence[str], config: Mapping[str, Any]) -> dict[str, Any]:
        return self._call("POST", "/v1/setup/plan",
                          {"capabilities": list(capabilities), "config": dict(config)})

    def setup_apply(self, plan_id: str, key: str) -> dict[str, Any]:
        return self._call("POST", "/v1/setup/apply", {"plan_id": plan_id}, key)

    def setup_status(self, operation_id: str) -> dict[str, Any]:
        return self._call("GET", f"/v1/setup/{quote(operation_id, safe='')}")

    def stop(self) -> dict[str, Any]:
        return self._call("POST", "/v1/service/stop", {})

    def download_artifact(self, job_id: str, artifact_id: str, sink: Callable[[bytes], Any]) -> int:
        """Stream an artifact into ``sink``; returns the byte count.  Verifies identity headers."""
        path = f"/v1/jobs/{quote(job_id, safe='')}/artifacts/{quote(artifact_id, safe='')}"
        try:
            response = self._attempts(lambda: self._send("GET", path, None, None), True)
            if response.status != 200:
                self._decode(response.status, response.read(MAX_RESPONSE_BYTES + 1))
                raise ClientError(ReasonCode.SERVICE_UNAVAILABLE, retryable=True, status=response.status)
            if response.getheader(INSTANCE_HEADER) != self._nonce \
                    or response.getheader(SCHEMA_HEADER) != str(SCHEMA_VERSION):
                raise ClientError(ReasonCode.SERVICE_UNAVAILABLE, next_action="restart_client",
                                  identity_mismatch=True, status=200)
            try:
                expected = int(response.getheader("Content-Length", ""))
            except ValueError:
                raise ClientError(ReasonCode.SERVICE_UNAVAILABLE, retryable=True) from None
            total = 0
            try:
                while True:
                    chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                    if not chunk:
                        break
                    total += len(chunk)
                    sink(chunk)
            except (OSError, http.client.HTTPException):
                raise ClientError(ReasonCode.SERVICE_UNAVAILABLE, retryable=True) from None
            if total != expected:
                raise ClientError(ReasonCode.SERVICE_UNAVAILABLE, retryable=True)
            return total
        finally:
            self.close()


def new_idempotency_key() -> str:
    return str(uuid.uuid4())


def connect(state_dir: str | os.PathLike | None = None, **client_options: Any) -> ServiceClient | None:
    """A client for the live service, or None when it is not running.  Never launches."""
    record = discover(state_dir)
    return None if record is None else ServiceClient(record, **client_options)


# ----------------------------------------------------------------------------- start / stop


def _package_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _launch_flags() -> dict[str, Any]:
    if os.name == "nt":
        flags = (getattr(subprocess, "DETACHED_PROCESS", 0x8)
                 | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x200)
                 | getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000))
        return {"creationflags": flags}
    return {"start_new_session": True}


def _child_env() -> dict[str, str]:
    env = dict(os.environ)
    parts = [_package_root()] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    env["PYTHONPATH"] = os.pathsep.join(parts)
    return env


def _kill_tree(pid: int) -> None:
    """Best effort teardown of a process we launched (launcher stub plus real interpreter)."""
    try:
        import psutil

        root = psutil.Process(pid)
        procs = [*root.children(recursive=True), root]
    except Exception:  # noqa: BLE001 - already gone or psutil missing
        return
    for proc in procs:
        with contextlib.suppress(Exception):
            proc.kill()
    with contextlib.suppress(Exception):
        psutil.wait_procs(procs, timeout=5)


def kill_launched_services() -> None:
    """Terminate and reap every service this process launched (test and failure cleanup)."""
    for pid, proc in list(_CHILDREN.items()):
        _kill_tree(pid)
        with contextlib.suppress(Exception):
            proc.wait(timeout=5)
        _CHILDREN.pop(pid, None)


def _reap_children() -> None:
    for pid, proc in list(_CHILDREN.items()):
        if proc.poll() is not None:
            _CHILDREN.pop(pid, None)


def start_service(state_dir: str | os.PathLike | None = None, *, factory: str | None = None,
                  import_path: str | None = None, timeout: float = START_TIMEOUT_SECONDS,
                  volume_provider: Any = None, popen: Callable[..., Any] = subprocess.Popen,
                  discover_fn: Callable[..., Any] | None = None,
                  client_factory: Callable[..., Any] | None = None,
                  sleep: Callable[[float], None] = time.sleep,
                  clock: Callable[[], float] = time.monotonic) -> RuntimeRecord:
    """Launch the detached service after a state-volume preflight, or return the live one."""
    directory = _state(state_dir)
    preflight_state_path(directory, volume_provider=volume_provider)
    find = discover_fn or discover
    existing = find(directory)
    if existing is not None:
        return existing
    os.makedirs(directory, mode=0o700, exist_ok=True)
    argv = [sys.executable, "-m", "podmuncher.service", "--serve", "--state-dir", directory]
    if factory:
        argv += ["--factory", factory]
    if import_path:
        argv += ["--import-path", import_path]
    try:
        child = popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                      stderr=subprocess.DEVNULL, close_fds=True, env=_child_env(), **_launch_flags())
    except OSError:
        raise ServiceError(ReasonCode.SERVICE_UNAVAILABLE) from None
    child_pid = getattr(child, "pid", None)
    if isinstance(child_pid, int):
        _CHILDREN[child_pid] = child
    make_client = client_factory or ServiceClient
    deadline = clock() + timeout
    exited_at: float | None = None
    while True:
        record = find(directory)
        if record is not None:
            try:
                reply = make_client(record).identity()
                if reply.get("instance_nonce") == record.instance_nonce:
                    return record
            except ClientError:
                pass
        now = clock()
        if child.poll() is not None:
            exited_at = now if exited_at is None else exited_at
            if now - exited_at >= EXITED_GRACE_SECONDS:
                break
        if now >= deadline:
            break
        sleep(POLL_SECONDS)
    if child.poll() is None and isinstance(child_pid, int):
        _kill_tree(child_pid)
        with contextlib.suppress(Exception):
            child.wait(timeout=5)
    _reap_children()
    raise ServiceError(ReasonCode.SERVICE_UNAVAILABLE)


@dataclass(frozen=True, slots=True)
class StopOutcome:
    ok: bool
    outcome: str  # stopped | already_stopped | writer_unresolved | unreachable | timeout
    reason_code: ReasonCode = ReasonCode.OK


def _process_gone(record: RuntimeRecord, probe: Callable[[int], Any]) -> bool:
    from . import recovery

    _reap_children()
    try:
        current = probe(record.pid)
    except Exception:  # noqa: BLE001 - unknown answer is not proof of exit
        return False
    return not recovery._same_process(record.process_identity, current)


def stop_service(state_dir: str | os.PathLike | None = None, *, timeout: float = STOP_TIMEOUT_SECONDS,
                 probe: Callable[[int], Any] | None = None,
                 discover_fn: Callable[..., Any] | None = None,
                 client_factory: Callable[..., Any] | None = None,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic) -> StopOutcome:
    """Stop the service.  Success only after the serving process has exited."""
    from . import recovery

    directory = _state(state_dir)
    probe = probe or recovery.psutil_identity
    record = (discover_fn or discover)(directory)
    if record is None:
        return StopOutcome(True, "already_stopped")
    client = (client_factory or ServiceClient)(record)
    stop_error: ClientError | None = None
    try:
        client.stop()
    except ClientError as error:
        if error.reason_code is ReasonCode.WRITER_UNRESOLVED:
            return StopOutcome(False, "writer_unresolved", ReasonCode.WRITER_UNRESOLVED)
        if error.identity_mismatch:
            return StopOutcome(False, "unreachable", error.reason_code)
        # A client timeout while writers are reaped, or a concurrent stop's 503
        # service_stopped, can still end in a clean exit: the process decides.
        stop_error = error
    finally:
        client.close()
    deadline = clock() + timeout
    while not _process_gone(record, probe):
        if clock() >= deadline:
            if stop_error is not None:
                return StopOutcome(False, "unreachable", stop_error.reason_code)
            return StopOutcome(False, "timeout", ReasonCode.SERVICE_UNAVAILABLE)
        sleep(POLL_SECONDS)
    for child in list(_CHILDREN.values()):
        if child.returncode == EXIT_SHUTDOWN_FAILED:
            return StopOutcome(False, "writer_unresolved", ReasonCode.WRITER_UNRESOLVED)
    return StopOutcome(True, "stopped")


def read_status(state_dir: str | os.PathLike | None = None, *,
                discover_fn: Callable[..., Any] | None = None,
                client_factory: Callable[..., Any] | None = None) -> dict[str, Any]:
    """Read-only status.  Never launches; a stopped service shows its persisted snapshot as stale."""
    directory = _state(state_dir)
    record = (discover_fn or discover)(directory)
    if record is not None:
        client = (client_factory or ServiceClient)(record)
        try:
            reply = client.snapshot()
            return {"state": "running", "snapshot": reply.get("snapshot"), "stale": False}
        except ClientError:
            return {"state": "unreachable", "snapshot": persisted_snapshot(directory), "stale": True}
        finally:
            client.close()
    return {"state": "stopped", "snapshot": persisted_snapshot(directory), "stale": True}


# ------------------------------------------------------------------------ serving process


class _SnapshotPersistingActions:
    """Delegates every action; persists snapshot reads to the protected state file.

    ``save_snapshot`` costs ACL work plus fsync (~130 ms on Windows), so it runs off the
    loop in a worker thread.  One writer task at most: a newer snapshot replaces a pending
    one, and an unchanged snapshot is not rewritten.  ``close()`` stops scheduling and
    waits for the in-flight write so the shutdown save is the final, unraced write.
    """

    def __init__(self, inner: Any, state_dir: str) -> None:
        self._inner = inner
        self._state_dir = state_dir
        self.last: Any = None
        self._pending: tuple[str, Any] | None = None
        self._persisted_key: str | None = None
        self._writer: asyncio.Task[None] | None = None
        self._closed = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def snapshot(self) -> Any:
        value = await self._inner.snapshot()
        self.last = _public(value)
        self._schedule(self.last)
        return value

    def _schedule(self, public: Any) -> None:
        if self._closed:
            return
        try:
            key = json.dumps(public, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError):
            return  # not persistable; save_snapshot would refuse it too
        if key == self._persisted_key and self._pending is None:
            return
        # A detached copy: the worker thread never touches a dict the loop may mutate.
        self._pending = (key, json.loads(key))
        if self._writer is None or self._writer.done():
            self._writer = asyncio.get_running_loop().create_task(self._drain())

    async def _drain(self) -> None:
        while self._pending is not None:
            (key, value), self._pending = self._pending, None
            if key == self._persisted_key:
                continue
            try:
                ok = await asyncio.to_thread(save_snapshot, self._state_dir, value)
            except Exception:  # noqa: BLE001 - best effort, like save_snapshot itself
                ok = False
            if ok:
                self._persisted_key = key

    async def close(self) -> None:
        """Stop background persistence and wait for the in-flight write to finish."""
        self._closed = True
        self._pending = None
        writer = self._writer
        if writer is not None and not writer.done():
            await asyncio.wait({writer})


async def _maybe_await(value: Any) -> Any:
    return await value if inspect.isawaitable(value) else value


def _install_signals(loop: asyncio.AbstractEventLoop, event: asyncio.Event) -> None:
    def trip(*_: Any) -> None:
        loop.call_soon_threadsafe(event.set)

    for name in ("SIGTERM", "SIGINT", "SIGBREAK"):
        number = getattr(signal, name, None)
        if number is not None:
            with contextlib.suppress(ValueError, OSError):
                signal.signal(number, trip)


def _raise_nofile_limit(*, resource_module: Any = None, os_name: str | None = None) -> str:
    """Raise the soft RLIMIT_NOFILE to ``min(hard, NOFILE_TARGET)`` on POSIX, only upward.

    Best effort: returns a closed outcome (``raised``, ``unchanged``, ``not_applicable``,
    ``unavailable`` or ``failed``) and never raises.  Windows has no RLIMIT_NOFILE.
    """
    if (os_name or os.name) == "nt":
        return "not_applicable"
    try:
        res = resource_module if resource_module is not None else importlib.import_module("resource")
        soft, hard = res.getrlimit(res.RLIMIT_NOFILE)
        infinity = res.RLIM_INFINITY
    except Exception:  # noqa: BLE001 - no resource module or no limit to read
        _LOG.warning("nofile limit not raised: reason=nofile_unavailable")
        return "unavailable"
    target = NOFILE_TARGET if hard == infinity else min(hard, NOFILE_TARGET)
    if soft == infinity or soft >= target:
        return "unchanged"
    try:
        res.setrlimit(res.RLIMIT_NOFILE, (target, hard))
    except (OSError, ValueError):
        _LOG.warning("nofile limit not raised: reason=nofile_raise_failed")
        return "failed"
    return "raised"


async def _serve_async(state_dir: str, factory_spec: str | None, import_path: str | None) -> int:
    from . import recovery
    from .api import ApiServer

    _raise_nofile_limit()  # before the composition opens sockets, pipes and state files
    try:
        lock = recovery.acquire_singleton(state_dir)
    except recovery.RecoveryError as error:
        return EXIT_CONFLICT if error.reason_code is ReasonCode.CONFLICT else EXIT_UNAVAILABLE
    server: ApiServer | None = None
    composition: Any = None
    persisting: _SnapshotPersistingActions | None = None
    started = False
    code = EXIT_UNAVAILABLE
    try:
        composition = await _maybe_await(resolve_factory(factory_spec, import_path)(state_dir))
        await composition.startup()
        started = True
        nonce, credential = str(uuid.uuid4()), secrets.token_urlsafe(32)
        persisting = _SnapshotPersistingActions(composition.actions, state_dir)
        server = ApiServer(persisting, credential=credential, instance_nonce=nonce,
                           audit=composition.audit,
                           registries=getattr(composition, "registries", None))
        port = await server.start()
        recovery.write_runtime_file(_runtime_path(state_dir), port=port, credential=credential,
                                    instance_nonce=nonce,
                                    identity=recovery.psutil_identity(os.getpid()))
        terminate = asyncio.Event()
        _install_signals(asyncio.get_running_loop(), terminate)
        waiters = [asyncio.ensure_future(server.stop_requested.wait()),
                   asyncio.ensure_future(terminate.wait())]
        try:
            await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for waiter in waiters:
                waiter.cancel()
        code = EXIT_OK
    except Exception:  # noqa: BLE001 - closed exit codes, never a traceback with secrets
        code = EXIT_UNAVAILABLE
    finally:
        ok = True
        if server is not None:
            server.begin_stopping()  # every exit path: new requests get service_stopped
        if persisting is not None:
            with contextlib.suppress(Exception):
                await persisting.close()  # the shutdown save below must be the last write
        if started:
            fresh = None
            try:
                fresh = await asyncio.wait_for(composition.actions.snapshot(), 5)
            except Exception:  # noqa: BLE001 - fall back to the last snapshot read
                fresh = persisting.last if persisting is not None else None
            if fresh is not None:
                save_snapshot(state_dir, fresh)
            try:
                ok = bool((await composition.shutdown())[0])
            except Exception:  # noqa: BLE001 - failed shutdown is an unresolved writer
                ok = False
        if server is not None:
            with contextlib.suppress(Exception):
                await server.close()
        with contextlib.suppress(OSError):
            os.unlink(_runtime_path(state_dir))
        lock.release()
        if started and not ok:
            code = EXIT_SHUTDOWN_FAILED
    return code


def serve(state_dir: str, *, factory: str | None = None, import_path: str | None = None) -> int:
    return asyncio.run(_serve_async(state_dir, factory, import_path))


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m podmuncher.service", add_help=False)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--factory", default=None)
    parser.add_argument("--import-path", default=None)
    try:
        args = parser.parse_args(argv)
    except SystemExit:
        return EXIT_USAGE
    if not args.serve:
        return EXIT_USAGE
    return serve(args.state_dir, factory=args.factory, import_path=args.import_path)


__all__ = (
    "DEFAULT_FACTORY",
    "ClientError",
    "ServiceClient",
    "ServiceComposition",
    "ServiceError",
    "ServiceFactory",
    "StopOutcome",
    "connect",
    "default_state_dir",
    "discover",
    "new_idempotency_key",
    "persisted_snapshot",
    "preflight_state_path",
    "read_status",
    "resolve_factory",
    "save_snapshot",
    "serve",
    "start_service",
    "stop_service",
)

if __name__ == "__main__":
    raise SystemExit(_main())
