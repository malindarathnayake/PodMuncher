"""Detached service lifecycle and the shared API client (P5-U2)."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from podmuncher import service
from podmuncher.domains import ReasonCode, ServiceState
from podmuncher.recovery import ProcessIdentity, RuntimeRecord

ROOT = Path(__file__).resolve().parents[1]
CREDENTIAL = "credential-" + "Z9" * 12
NONCE = "nonce-" + "b" * 26
KEY = "33333333-3333-4333-8333-333333333333"
JOB_ID = "11111111-1111-4111-8111-111111111111"

FACTORY_SOURCE = '''
import os
import uuid

from podmuncher.api import ApiActionError
from podmuncher.domains import ReasonCode, ServiceState


class FakeActions:
    def __init__(self):
        self.jobs = {}
        self.keys = {}
        self.cancels = 0

    def _job(self, job_id):
        return {"job_id": job_id, "state": "queued", "cancels": self.cancels}

    async def identity(self):
        return {"pid": os.getpid(), "service_state": ServiceState.RUNNING}

    async def snapshot(self):
        return {"schema_version": 1, "jobs": len(self.jobs), "freshness": "fresh"}

    async def list_jobs(self, limit, cursor):
        return [self._job(j) for j in self.jobs], None

    async def get_job(self, job_id):
        if job_id not in self.jobs:
            raise ApiActionError(ReasonCode.INVALID_INPUT, not_found=True)
        return self._job(job_id)

    async def submit(self, request, key):
        if key in self.keys:
            return self._job(self.keys[key]), True
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = request.request_hash
        self.keys[key] = job_id
        return self._job(job_id), False

    async def recheck(self, job_id):
        return self._job(job_id), {"result": "ready"}, True

    async def retry(self, job_id, key):
        return self._job(job_id), str(uuid.uuid4()), False

    async def clone(self, job_id, request, key):
        return self._job(job_id)

    async def cancel(self, job_id):
        self.cancels += 1
        return self._job(job_id), "intent"

    async def open_artifact(self, job_id, artifact_id):
        async def chunks():
            yield b"x" * 10
        return 10, "audio/mpeg", chunks()

    async def list_cache(self, limit, cursor):
        return [], None

    async def remove_cache(self, entry_id):
        return {"entry_id": entry_id}

    async def setup_plan(self, capabilities, config):
        return {"plan_id": "plan-1"}

    async def setup_apply(self, plan_id, key):
        return {"operation_id": str(uuid.uuid4())}

    async def setup_status(self, operation_id):
        return {"state": "done"}

    async def stop(self):
        return True, "ok"


class Composition:
    def __init__(self, state_dir):
        self.actions = FakeActions()
        self.state_dir = state_dir

    def audit(self, route, code, ids):
        return None

    async def startup(self):
        return None

    async def shutdown(self):
        return True, "ok"


def make_service(state_dir):
    return Composition(state_dir)
'''


def make_service_factory(tmp_path):
    """Write the in-memory factory module; return (module:attr, import_path)."""
    directory = tmp_path / "factory"
    directory.mkdir(exist_ok=True)
    (directory / "fake_service_factory.py").write_text(FACTORY_SOURCE, encoding="utf-8")
    return "fake_service_factory:make_service", str(directory)


def spawn_test_service(state_dir, factory):
    spec, import_path = factory
    return service.start_service(state_dir, factory=spec, import_path=import_path, timeout=30.0)


def job_request(tmp_path):
    from podmuncher.requests import JobRequest

    return JobRequest.from_dict({
        "schema_version": 1,
        "source": {"kind": "local", "input": "missing.mp4"},
        "outputs": {"transcript": True, "video": False, "video_quality": "1080p"},
        "selection": {"device": "auto", "profile": "balanced", "model": "auto", "language": "auto"},
        "output_root": str(tmp_path / "output"),
        "limits": {"media_bytes": 1024, "duration_seconds": 60, "export_bytes": 512},
    }).to_dict()


def make_record(port=5555, pid=4321, credential=CREDENTIAL, nonce=NONCE):
    return RuntimeRecord(1, port, pid, nonce, credential, ProcessIdentity(pid, 100.0))


class FakeResponse:
    def __init__(self, status, payload=None, headers=None, raw=None):
        self.status = status
        self._data = raw if raw is not None else json.dumps(payload).encode()
        self._headers = headers or {}

    def read(self, limit=None):
        data, self._data = self._data[:limit], self._data[limit:]
        return data

    def getheader(self, name, default=None):
        return self._headers.get(name, default)


class FakeConnection:
    """Scripted connection: each item is an exception to raise or a FakeResponse."""

    def __init__(self, script):
        self.script = script
        self.requests = []
        self.closed = 0

    def factory(self, host, port, timeout=None):
        self.host, self.port = host, port
        return self

    def request(self, method, path, body=None, headers=None):
        self.requests.append((method, path, body, dict(headers or {})))
        item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, BaseException):
            raise item
        self.current = item

    def getresponse(self):
        return self.current

    def close(self):
        self.closed += 1


def ok_payload(**extra):
    return {"schema_version": 1, "instance_nonce": NONCE, **extra}


def client_for(script, sleeps=None):
    conn = FakeConnection(script)
    log = [] if sleeps is None else sleeps
    return service.ServiceClient(make_record(), connection_factory=conn.factory, sleep=log.append), conn


def allowed_provider(value):
    evidence = SimpleNamespace(state_allowed=SimpleNamespace(value=value))
    return SimpleNamespace(sample=lambda path, sqlite_state=False: evidence)


# -- import hygiene ----------------------------------------------------------


def test_import_is_lazy_about_heavy_dependencies():
    code = (
        "import sys, podmuncher.service; bad = {'aiohttp', 'psutil', 'platformdirs', "
        "'opentelemetry', 'textual', 'podmuncher.api', 'podmuncher.recovery', "
        "'podmuncher.bootstrap'}; "
        "found = [m for m in sys.modules if m in bad or m.split('.')[0] in bad - {'podmuncher.api', "
        "'podmuncher.recovery', 'podmuncher.bootstrap'}]; assert not found, found"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True,
                            env={**os.environ, "PYTHONPATH": str(ROOT / "src")}, timeout=30, check=False)
    assert result.returncode == 0, result.stderr


# -- read-only never launches -------------------------------------------------


def test_read_status_never_launches_and_is_stale_when_stopped(tmp_path, monkeypatch):
    real_popen = subprocess.Popen
    launches = []

    def watching(args, *rest, **kwargs):
        if "podmuncher.service" in " ".join(map(str, args)):
            launches.append(args)
            raise AssertionError("read-only status must not launch")
        return real_popen(args, *rest, **kwargs)  # icacls for the private snapshot file

    monkeypatch.setattr(subprocess, "Popen", watching)
    state = tmp_path / "state"
    state.mkdir()
    first = service.read_status(state)
    assert first == {"state": "stopped", "snapshot": None, "stale": True}
    assert service.save_snapshot(state, {"schema_version": 1, "jobs": 2})
    second = service.read_status(state)
    assert second["state"] == "stopped" and second["stale"] is True
    assert second["snapshot"] == {"schema_version": 1, "jobs": 2}
    assert service.connect(state) is None
    assert launches == []


def test_unprotected_snapshot_is_not_trusted(tmp_path, monkeypatch):
    from podmuncher import recovery

    state = tmp_path / "state"
    state.mkdir()
    service.save_snapshot(state, {"jobs": 1})
    monkeypatch.setattr(recovery, "verify_runtime_permissions", lambda path: False)
    assert service.persisted_snapshot(state) is None


# -- start preflight and idempotence ------------------------------------------


@pytest.mark.parametrize(("allowed", "reason"), [
    (False, ReasonCode.INVALID_CONFIG), (None, ReasonCode.STORAGE_UNKNOWN)])
def test_start_preflight_refuses_non_local_state_path(tmp_path, allowed, reason):
    launched = []
    with pytest.raises(service.ServiceError) as error:
        service.start_service(tmp_path / "state", volume_provider=allowed_provider(allowed),
                              popen=lambda *a, **k: launched.append(a))
    assert error.value.reason_code is reason
    assert launched == []
    assert not (tmp_path / "state").exists()


def test_start_returns_existing_live_record_without_second_launch(tmp_path):
    record = make_record()
    launched = []
    result = service.start_service(
        tmp_path / "state", volume_provider=allowed_provider(True),
        popen=lambda *a, **k: launched.append(a), discover_fn=lambda d: record)
    assert result is record and launched == []


class FakeChild:
    pid = 987654

    def __init__(self, alive=True):
        self.alive = alive
        self.returncode = None

    def poll(self):
        return None if self.alive else 1

    def wait(self, timeout=None):
        return 0


def test_start_launch_flags_argv_and_polling(tmp_path):
    calls = []
    child = FakeChild()

    def popen(argv, **kwargs):
        calls.append((argv, kwargs))
        return child

    seen = iter([None, None, make_record()])
    clients = []

    class StubClient:
        def __init__(self, record):
            clients.append(record)

        def identity(self):
            return {"instance_nonce": NONCE}

    record = service.start_service(
        tmp_path / "state", factory="mod:attr", import_path=str(tmp_path),
        volume_provider=allowed_provider(True), popen=popen,
        discover_fn=lambda d: next(seen), client_factory=StubClient, sleep=lambda s: None)
    service._CHILDREN.pop(child.pid, None)
    assert record.instance_nonce == NONCE and len(calls) == 1
    argv, kwargs = calls[0]
    assert argv[1:4] == ["-m", "podmuncher.service", "--serve"]
    assert argv[argv.index("--factory") + 1] == "mod:attr"
    assert CREDENTIAL not in " ".join(argv)
    assert kwargs["stdin"] == kwargs["stdout"] == kwargs["stderr"] == subprocess.DEVNULL
    assert kwargs["close_fds"] is True
    if os.name == "nt":
        flags = kwargs["creationflags"]
        assert flags & subprocess.DETACHED_PROCESS
        assert flags & subprocess.CREATE_NEW_PROCESS_GROUP
        assert flags & subprocess.CREATE_NO_WINDOW
    else:
        assert kwargs["start_new_session"] is True


def test_start_times_out_when_child_dies_without_record(tmp_path):
    child = FakeChild(alive=False)
    ticks = iter(range(1000))
    with pytest.raises(service.ServiceError) as error:
        service.start_service(
            tmp_path / "state", timeout=5.0, volume_provider=allowed_provider(True),
            popen=lambda *a, **k: child, discover_fn=lambda d: None, sleep=lambda s: None,
            clock=lambda: next(ticks))
    service._CHILDREN.pop(child.pid, None)
    assert error.value.reason_code is ReasonCode.SERVICE_UNAVAILABLE


def test_missing_default_factory_is_service_unavailable():
    with pytest.raises(service.ServiceError) as error:
        service.resolve_factory("no_such_module_xyz:make", None)
    assert error.value.reason_code is ReasonCode.SERVICE_UNAVAILABLE


# -- discovery and identity ---------------------------------------------------


def test_stale_and_pid_reused_record_yields_no_client(tmp_path):
    from podmuncher import recovery

    state = tmp_path / "state"
    state.mkdir()
    identity = ProcessIdentity(os.getpid(), 100.0)
    with recovery.acquire_singleton(state):
        recovery.write_runtime_file(state / recovery.RUNTIME_FILE, port=5555, credential=CREDENTIAL,
                                    instance_nonce=NONCE, identity=identity)
    assert service.discover(state, probe=lambda pid: ProcessIdentity(pid, 100.0)) is not None
    assert service.discover(state, probe=lambda pid: ProcessIdentity(pid, 5000.0)) is None  # reuse
    assert service.discover(state, probe=lambda pid: None) is None  # dead
    assert service.discover(state, probe=lambda pid: ProcessIdentity(pid, 100.0),
                            expected_nonce="other-nonce") is None


def test_nonce_mismatch_is_identity_error():
    wrong = {"schema_version": 1, "instance_nonce": "someone-else"}
    client, _ = client_for([FakeResponse(200, wrong)])
    with pytest.raises(service.ClientError) as error:
        client.identity()
    assert error.value.identity_mismatch and error.value.reason_code is ReasonCode.SERVICE_UNAVAILABLE
    html, _ = client_for([FakeResponse(200, raw=b"<html>squatter</html>")])
    with pytest.raises(service.ClientError) as error:
        html.identity()
    assert error.value.identity_mismatch


def test_schema_426_maps_to_upgrade_action():
    body = ok_payload(error={"reason_code": "unsupported_schema", "retryable": False,
                             "next_action": "upgrade_client"})
    client, _ = client_for([FakeResponse(426, body)])
    with pytest.raises(service.ClientError) as error:
        client.snapshot()
    assert error.value.reason_code is ReasonCode.UNSUPPORTED_SCHEMA
    assert error.value.next_action == "upgrade_client" and error.value.status == 426


def test_error_envelope_maps_reason_code():
    body = ok_payload(error={"reason_code": "writer_unresolved", "retryable": True,
                             "next_action": "wait_for_writers"})
    client, _ = client_for([FakeResponse(503, body)])
    with pytest.raises(service.ClientError) as error:
        client.stop()
    assert error.value.reason_code is ReasonCode.WRITER_UNRESOLVED
    assert error.value.retryable and error.value.next_action == "wait_for_writers"


def test_request_headers_carry_credential_never_origin_or_url():
    client, conn = client_for([FakeResponse(200, ok_payload(jobs=[]))])
    client.list_jobs(limit=5, cursor="abc")
    method, path, body, headers = conn.requests[0]
    assert method == "GET" and path == "/v1/jobs?limit=5&cursor=abc" and body is None
    assert headers["Authorization"] == f"Bearer {CREDENTIAL}"
    assert headers["X-Podmuncher-Schema"] == "1" and headers["Host"] == "127.0.0.1:5555"
    assert "Origin" not in headers and CREDENTIAL not in path


# -- bounded reconnect ----------------------------------------------------------


def test_get_retries_connection_errors_with_backoff_then_gives_up():
    sleeps = []
    client, conn = client_for([ConnectionRefusedError(CREDENTIAL)], sleeps)
    with pytest.raises(service.ClientError) as error:
        client.snapshot()
    assert len(conn.requests) == 4 and sleeps == [0.5, 1.0, 2.0]
    assert error.value.reason_code is ReasonCode.SERVICE_UNAVAILABLE and error.value.retryable
    assert CREDENTIAL not in str(error.value) and CREDENTIAL not in repr(error.value)


def test_get_recovers_after_transient_connection_error():
    sleeps = []
    client, conn = client_for([ConnectionResetError(), FakeResponse(200, ok_payload(snapshot={}))], sleeps)
    assert client.snapshot()["snapshot"] == {}
    assert len(conn.requests) == 2 and sleeps == [0.5]


def test_post_without_key_is_never_retried():
    sleeps = []
    client, conn = client_for([ConnectionResetError()], sleeps)
    for call in (lambda: client.cancel(JOB_ID), lambda: client.recheck(JOB_ID), client.stop):
        with pytest.raises(service.ClientError):
            call()
    assert len(conn.requests) == 3 and sleeps == []


def test_keyed_post_is_retried_and_sends_idempotency_key():
    sleeps = []
    client, conn = client_for([ConnectionResetError(), FakeResponse(201, ok_payload(job={}, replayed=False))],
                              sleeps)
    client.submit({"schema_version": 1}, KEY)
    assert len(conn.requests) == 2 and sleeps == [0.5]
    assert all(request[3]["Idempotency-Key"] == KEY for request in conn.requests)


def test_http_error_status_is_not_a_reconnect():
    body = ok_payload(error={"reason_code": "rate_limited", "retryable": True, "next_action": "retry_later"})
    sleeps = []
    client, conn = client_for([FakeResponse(429, body)], sleeps)
    with pytest.raises(service.ClientError) as error:
        client.snapshot()
    assert error.value.reason_code is ReasonCode.RATE_LIMITED
    assert len(conn.requests) == 1 and sleeps == []


# -- credential hygiene and close ----------------------------------------------


def test_credential_never_in_repr_or_errors():
    client, _ = client_for([FakeResponse(200, {"schema_version": 1, "instance_nonce": "x"})])
    assert CREDENTIAL not in repr(client) and CREDENTIAL not in str(client)
    assert CREDENTIAL not in repr(make_record())
    with pytest.raises(service.ClientError) as error:
        client.identity()
    assert CREDENTIAL not in str(error.value) and CREDENTIAL not in repr(error.value)


def test_close_triggers_no_cancel_and_no_request():
    client, conn = client_for([FakeResponse(200, ok_payload(job={}))])
    client.get_job(JOB_ID)
    before = len(conn.requests)
    client.close()
    client.close()
    assert len(conn.requests) == before
    assert not any(path.endswith("/cancel") for _, path, _, _ in conn.requests)


def test_download_streams_and_verifies_identity_headers():
    good = FakeResponse(200, raw=b"x" * 10, headers={
        "X-Podmuncher-Instance": NONCE, "X-Podmuncher-Schema": "1", "Content-Length": "10"})
    client, _ = client_for([good])
    sink = bytearray()
    assert client.download_artifact(JOB_ID, JOB_ID, sink.extend) == 10 and bytes(sink) == b"x" * 10
    bad = FakeResponse(200, raw=b"x" * 10, headers={
        "X-Podmuncher-Instance": "other", "X-Podmuncher-Schema": "1", "Content-Length": "10"})
    client, _ = client_for([bad])
    with pytest.raises(service.ClientError) as error:
        client.download_artifact(JOB_ID, JOB_ID, sink.extend)
    assert error.value.identity_mismatch


# -- stop outcomes (pure) ---------------------------------------------------------


class StopClient:
    def __init__(self, error=None):
        self.error = error
        self.stopped = False

    def stop(self):
        self.stopped = True
        if self.error:
            raise self.error
        return {"state": "stopped"}

    def close(self):
        pass


def test_stop_not_running_is_already_stopped():
    outcome = service.stop_service("unused", discover_fn=lambda d: None)
    assert outcome.ok and outcome.outcome == "already_stopped"


def test_stop_writer_unresolved_is_failure_and_does_not_wait():
    error = service.ClientError(ReasonCode.WRITER_UNRESOLVED, retryable=True)
    outcome = service.stop_service("unused", discover_fn=lambda d: make_record(),
                                   client_factory=lambda r: StopClient(error),
                                   probe=lambda pid: ProcessIdentity(pid, 100.0))
    assert not outcome.ok and outcome.outcome == "writer_unresolved"
    assert outcome.reason_code is ReasonCode.WRITER_UNRESOLVED


def test_stop_succeeds_only_after_process_exits():
    states = iter([ProcessIdentity(4321, 100.0)] * 3 + [None])
    ticks = iter(range(100))
    outcome = service.stop_service("unused", discover_fn=lambda d: make_record(),
                                   client_factory=lambda r: StopClient(), probe=lambda pid: next(states),
                                   sleep=lambda s: None, clock=lambda: float(next(ticks)))
    assert outcome.ok and outcome.outcome == "stopped"


def test_stop_times_out_while_process_alive():
    ticks = iter(range(100))
    outcome = service.stop_service("unused", timeout=3.0, discover_fn=lambda d: make_record(),
                                   client_factory=lambda r: StopClient(),
                                   probe=lambda pid: ProcessIdentity(pid, 100.0),
                                   sleep=lambda s: None, clock=lambda: float(next(ticks)))
    assert not outcome.ok and outcome.outcome == "timeout"


def _stopped_body():
    return ok_payload(error={"reason_code": "service_stopped", "retryable": False,
                             "next_action": "start_service"})


@pytest.mark.parametrize("script", [
    [TimeoutError("client read timed out")],  # server still reaping writers past 10 s
    [FakeResponse(503, _stopped_body())],  # a concurrent second stop
], ids=["client-timeout", "service-stopped-503"])
def test_stop_error_then_process_exit_is_ok(script):
    states = iter([ProcessIdentity(4321, 100.0)] * 3 + [None])
    ticks = iter(range(100))
    outcome = service.stop_service("unused", discover_fn=lambda d: make_record(),
                                   client_factory=lambda r: client_for(list(script))[0],
                                   probe=lambda pid: next(states),
                                   sleep=lambda s: None, clock=lambda: float(next(ticks)))
    assert outcome.ok and outcome.outcome == "stopped"


def test_stop_error_and_process_alive_at_deadline_is_unreachable():
    ticks = iter(range(100))
    outcome = service.stop_service("unused", timeout=3.0, discover_fn=lambda d: make_record(),
                                   client_factory=lambda r: client_for([TimeoutError()])[0],
                                   probe=lambda pid: ProcessIdentity(pid, 100.0),
                                   sleep=lambda s: None, clock=lambda: float(next(ticks)))
    assert not outcome.ok and outcome.outcome == "unreachable"
    assert outcome.reason_code is ReasonCode.SERVICE_UNAVAILABLE


def test_stop_identity_mismatch_is_still_unreachable_without_waiting():
    probes = []
    wrong = {"schema_version": 1, "instance_nonce": "someone-else"}
    outcome = service.stop_service("unused", discover_fn=lambda d: make_record(),
                                   client_factory=lambda r: client_for([FakeResponse(200, wrong)])[0],
                                   probe=lambda pid: probes.append(pid),
                                   sleep=lambda s: pytest.fail("must not poll"))
    assert not outcome.ok and outcome.outcome == "unreachable"
    assert probes == []


# -- snapshot persistence off the loop --------------------------------------------


def test_snapshot_persistence_is_off_loop_serialized_and_deduplicated(monkeypatch):
    gate = threading.Event()
    saves, active, peak, lock = [], [0], [0], threading.Lock()

    def slow_save(state_dir, snapshot):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        gate.wait(5)
        saves.append(snapshot)
        with lock:
            active[0] -= 1
        return True

    monkeypatch.setattr(service, "save_snapshot", slow_save)

    class Inner:
        def __init__(self):
            self.value = {"n": 1}

        async def snapshot(self):
            return dict(self.value)

    async def scenario():
        inner = Inner()
        persisting = service._SnapshotPersistingActions(inner, "unused")
        # The handler returns while save_snapshot is still blocked in its worker thread.
        assert await asyncio.wait_for(persisting.snapshot(), 1) == {"n": 1}
        await asyncio.sleep(0.05)
        assert active[0] == 1 and saves == []
        inner.value = {"n": 2}
        await asyncio.wait_for(persisting.snapshot(), 1)
        inner.value = {"n": 3}
        await asyncio.wait_for(persisting.snapshot(), 1)  # replaces the pending n=2
        gate.set()
        await asyncio.wait({persisting._writer})
        assert saves == [{"n": 1}, {"n": 3}] and peak[0] == 1
        await persisting.snapshot()  # unchanged: not rewritten
        await asyncio.sleep(0.05)
        assert len(saves) == 2
        inner.value = {"n": 4}
        await persisting.snapshot()
        await asyncio.sleep(0)  # let the writer pick n=4 up
        await persisting.close()  # waits for the in-flight write
        assert saves[-1] == {"n": 4} and active[0] == 0
        inner.value = {"n": 5}
        await persisting.snapshot()  # closed: no more background writes
        await asyncio.sleep(0.05)
        assert len(saves) == 3 and persisting.last == {"n": 5}

    asyncio.run(scenario())


class _InProcessComposition:
    """Composition for _serve_async in this process; shutdown parks until released."""

    def __init__(self):
        self.actions = self
        self.snapshots = 0
        self.shutting = asyncio.Event()
        self.release = asyncio.Event()

    def registries(self):
        return frozenset({"small"}), None

    def audit(self, route, code, ids):
        return None

    async def startup(self):
        return None

    async def shutdown(self):
        self.shutting.set()
        await self.release.wait()
        return True, "ok"

    async def identity(self):
        return {"pid": os.getpid(), "service_state": ServiceState.RUNNING}

    async def snapshot(self):
        self.snapshots += 1
        return {"schema_version": 1, "n": self.snapshots}

    async def submit(self, request, key):
        return {"job_id": JOB_ID, "model": request.selection.model}, False


def test_serve_signal_shutdown_refuses_work_and_final_save_is_last(tmp_path, monkeypatch):
    state = tmp_path / "state"
    state.mkdir()
    holder = {}
    saves, active, peak, lock = [], [0], [0], threading.Lock()

    def slow_save(state_dir, snapshot):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
        time.sleep(0.3)
        saves.append(snapshot)
        with lock:
            active[0] -= 1
        return True

    def factory(state_dir):
        holder["composition"] = _InProcessComposition()
        return holder["composition"]

    monkeypatch.setattr(service, "save_snapshot", slow_save)
    monkeypatch.setattr(service, "resolve_factory", lambda spec, path: factory)
    monkeypatch.setattr(service, "_install_signals", lambda loop, event: holder.update(terminate=event))

    async def exercise():
        record = None
        for _ in range(250):
            record = service.discover(state)
            if record is not None:
                break
            await asyncio.sleep(0.02)
        assert record is not None
        client = service.ServiceClient(record, timeout=5.0)
        request = job_request(tmp_path)
        request["selection"] = {**request["selection"], "model": "small"}
        created = await asyncio.to_thread(client.submit, request, KEY)  # registry seam wired
        assert created["job"]["model"] == "small" and created["replayed"] is False
        started = time.monotonic()
        await asyncio.to_thread(client.snapshot)  # background write now in flight
        assert time.monotonic() - started < 0.25
        holder["terminate"].set()
        await asyncio.wait_for(holder["composition"].shutting.wait(), 5)
        with pytest.raises(service.ClientError) as error:
            await asyncio.to_thread(client.submit, request, str(uuid.uuid4()))
        assert error.value.reason_code is ReasonCode.SERVICE_STOPPED and error.value.status == 503

    async def scenario():
        task = asyncio.create_task(service._serve_async(str(state), None, None))
        try:
            await exercise()
        finally:  # never leave the serving task parked, even when an assertion fails
            if "terminate" in holder:
                holder["terminate"].set()
            if "composition" in holder:
                holder["composition"].release.set()
            await asyncio.wait({task}, timeout=10)
        assert task.done() and task.result() == service.EXIT_OK
        return holder["composition"]

    composition = asyncio.run(scenario())
    assert peak[0] == 1  # the shutdown save never raced the background writer
    assert saves[0] == {"schema_version": 1, "n": 1}
    assert saves[-1] == {"schema_version": 1, "n": composition.snapshots} and len(saves) == 2
    assert service.discover(state) is None


# -- open-file limit -------------------------------------------------------------


class FakeResource:
    RLIMIT_NOFILE = 7
    RLIM_INFINITY = -1

    def __init__(self, soft, hard, fail=None):
        self.limits = (soft, hard)
        self.fail = fail
        self.calls = []

    def getrlimit(self, which):
        assert which == self.RLIMIT_NOFILE
        return self.limits

    def setrlimit(self, which, limits):
        self.calls.append((which, limits))
        if self.fail is not None:
            raise self.fail
        self.limits = limits


@pytest.mark.parametrize(("soft", "hard", "expected"), [
    (256, -1, (8192, -1)),        # macOS default: soft 256, hard unlimited
    (256, 10240, (8192, 10240)),
    (1024, 4096, (4096, 4096)),   # capped at the hard limit
])
def test_nofile_soft_limit_is_raised_on_posix(soft, hard, expected):
    fake = FakeResource(soft, hard)
    assert service._raise_nofile_limit(resource_module=fake, os_name="posix") == "raised"
    assert fake.calls == [(FakeResource.RLIMIT_NOFILE, expected)]


@pytest.mark.parametrize(("soft", "hard"), [(8192, -1), (65536, 1048576), (-1, -1), (4096, 4096)])
def test_nofile_limit_is_never_lowered(soft, hard):
    fake = FakeResource(soft, hard)
    assert service._raise_nofile_limit(resource_module=fake, os_name="posix") == "unchanged"
    assert fake.calls == []


@pytest.mark.parametrize("error", [OSError(22, "Invalid argument"), ValueError("not allowed")])
def test_nofile_raise_failure_is_logged_and_survived(error, caplog):
    fake = FakeResource(256, -1, fail=error)
    with caplog.at_level("WARNING", logger="podmuncher.service"):
        assert service._raise_nofile_limit(resource_module=fake, os_name="posix") == "failed"
    assert "reason=nofile_raise_failed" in caplog.text


def test_nofile_unreadable_limit_is_unavailable(caplog):
    class Broken(FakeResource):
        def getrlimit(self, which):
            raise OSError("no limit")

    with caplog.at_level("WARNING", logger="podmuncher.service"):
        assert service._raise_nofile_limit(resource_module=Broken(0, 0), os_name="posix") == "unavailable"
    assert "reason=nofile_unavailable" in caplog.text


def test_nofile_is_not_applicable_on_windows():
    fake = FakeResource(256, -1)
    assert service._raise_nofile_limit(resource_module=fake, os_name="nt") == "not_applicable"
    assert fake.calls == []


@pytest.mark.skipif(os.name == "nt", reason="RLIMIT_NOFILE is POSIX only")
def test_nofile_real_resource_module_never_lowers():
    import resource

    before = resource.getrlimit(resource.RLIMIT_NOFILE)
    assert service._raise_nofile_limit() in {"raised", "unchanged", "failed"}
    after = resource.getrlimit(resource.RLIMIT_NOFILE)
    assert after[1] == before[1]
    assert before[0] == resource.RLIM_INFINITY or after[0] == resource.RLIM_INFINITY or after[0] >= before[0]


def test_serve_raises_nofile_before_the_composition(tmp_path, monkeypatch):
    order = []

    def factory_spec(spec, path):
        order.append("composition")
        raise service.ServiceError(ReasonCode.SERVICE_UNAVAILABLE)

    monkeypatch.setattr(service, "_raise_nofile_limit", lambda: order.append("nofile") or "raised")
    monkeypatch.setattr(service, "resolve_factory", factory_spec)
    code = asyncio.run(service._serve_async(str(tmp_path), None, None))
    assert code == service.EXIT_UNAVAILABLE and order == ["nofile", "composition"]


# -- domains ---------------------------------------------------------------------


def test_negative_examples_rejected_for_reason_code_and_service_state():
    registry = json.loads((ROOT / "Docs" / "contract-domains.json").read_text(encoding="utf-8"))
    negatives = registry["negative_examples"]
    assert negatives
    for value in negatives:
        for enum in (ReasonCode, ServiceState):
            with pytest.raises(Exception) as error:
                enum.parse(value)
            assert error.value.reason_code is ReasonCode.INVALID_INPUT
    client, _ = client_for([FakeResponse(
        503, ok_payload(error={"reason_code": "__invalid__", "retryable": False, "next_action": "x"}))])
    with pytest.raises(service.ClientError) as error:
        client.snapshot()
    assert error.value.reason_code is ReasonCode.SERVICE_UNAVAILABLE


# -- live smoke -------------------------------------------------------------------


@pytest.mark.live
def test_own_transport_smoke(tmp_path):
    import psutil

    from podmuncher import recovery

    state = str(tmp_path / "state")
    factory = make_service_factory(tmp_path)
    started = time.monotonic()
    record = None
    try:
        record = spawn_test_service(state, factory)
        assert service.discover(state) == record

        client = service.ServiceClient(record)
        assert client.identity()["instance_nonce"] == record.instance_nonce
        assert client.identity()["pid"] == record.pid
        assert CREDENTIAL not in repr(client)

        created = client.submit(job_request(tmp_path), service.new_idempotency_key())
        job_id = created["job"]["job_id"]
        client.close()

        again = service.connect(state)
        assert again is not None and again.instance_nonce == record.instance_nonce
        fetched = again.get_job(job_id)["job"]
        assert fetched["state"] == "queued" and fetched["cancels"] == 0  # client exit never cancels
        assert again.snapshot()["snapshot"]["jobs"] == 1
        sink = bytearray()
        assert again.download_artifact(job_id, str(uuid.uuid4()), sink.extend) == 10
        again.close()

        cmdline = " ".join(psutil.Process(record.pid).cmdline())
        assert record.credential not in cmdline and "--serve" in cmdline

        assert service.discover(state, expected_nonce="not-the-nonce") is None
        stale = service.read_status(state)
        assert stale["state"] == "running" and stale["stale"] is False

        outcome = service.stop_service(state, timeout=30.0)
        assert outcome.ok and outcome.outcome == "stopped"
        assert recovery.psutil_identity(record.pid) is None or not recovery._same_process(
            record.process_identity, recovery.psutil_identity(record.pid))
        assert service.discover(state) is None
        status = service.read_status(state)
        assert status["state"] == "stopped" and status["stale"] is True
        assert status["snapshot"]["jobs"] == 1  # protected persisted snapshot survives the stop
        assert service.stop_service(state).outcome == "already_stopped"
        assert not psutil.pid_exists(record.pid) or recovery.psutil_identity(record.pid) is None \
            or recovery.psutil_identity(record.pid).create_time != record.process_identity.create_time
        assert all(child.poll() is not None for child in service._CHILDREN.values())
        assert time.monotonic() - started < 60
    finally:
        service.kill_launched_services()
        if record is not None:
            service._kill_tree(record.pid)
    print("PODMUNCHER_SMOKE_OK")
