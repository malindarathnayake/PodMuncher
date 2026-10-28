"""Singleton lock, private runtime files, graceful stop and startup reconciliation.

Phase 4 unit P4-U5 (Docs/spec.md:1125).  Everything that touches the outside world is
injected: the process-identity probe, the Docker port (``OwnedContainers``), the audit
sink and the clock.  Rules enforced here:

* One daemon per state directory, proven by an OS file lock held for the process life.
* The runtime file (port, credential, identity) is private to the current user before any
  secret byte is written; a runtime record is live only when pid *and* create time match.
* A writer is stopped and reaped before a job becomes interrupted; an unresolved writer
  keeps every lease and marks nothing interrupted.
* Only containers carrying a ``podmuncher.instance`` label for an instance we recorded
  (stale runtime file or durable intent) are ever touched.
* An audit failure never blocks stop or reconcile: it becomes a bounded in-memory
  ``audit_loss`` notice plus a durable intent snapshot, and is never reported as durable.
* Reports carry reason codes and counts; no paths, no credentials.
"""
from __future__ import annotations

import contextlib
import getpass
import json
import os
import re
import stat
import subprocess
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Protocol, Self
from uuid import uuid4

from . import transitions as tr
from .artifacts import MANIFEST_NAME, MANIFEST_SCHEMA
from .domains import JobState, ReasonCode
from .repository import PAGE_MAX, Repository, RepositoryError

if TYPE_CHECKING:
    from .scheduler import Scheduler, WorkerLauncher

LOCK_NAME = "podmuncher.lock"
RUNTIME_FILE = "runtime.json"
INTENT_FILE = "intent.json"
RUNTIME_SCHEMA = 1
AUDIT_LOSS_MAX = 32
LABEL_INSTANCE = "podmuncher.instance"
ICACLS_TIMEOUT_SECONDS = 10
MAX_FILE_BYTES = 64 * 1024
MAX_MANIFEST_BYTES = 1024 * 1024
MISMATCH_IDS_MAX = 100
QUIESCE_TIMEOUT_SECONDS = 30.0
MANIFEST_SCAN_JOBS_MAX = 1000  # hard cap on jobs examined for manifest reconciliation
INTERRUPT_ROUNDS_MAX = 100  # active_jobs pages drained per stop or reconcile

_NONCE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z")
_PARTIAL = re.compile(r"\.[^\\/]+\.[0-9a-f]{32}\.partial\Z")  # artifacts.py staging name
_CONTAINER_ID = re.compile(r"[a-f0-9]{64}\Z")
_ACE = re.compile(r"(?P<who>\S(?:.*?\S)?):(?P<rights>(?:\([^()]*\))+)\s*$")
_BINARY = getattr(os, "O_BINARY", 0)
_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)
_CLOEXEC = getattr(os, "O_CLOEXEC", 0)
_IDENTITY_TOLERANCE = 0.001

_MESSAGES = {
    ReasonCode.CONFLICT: "Another Podmuncher instance is active.",
    ReasonCode.INVALID_CONFIG: "Runtime state is not private or not valid.",
    ReasonCode.WRITER_UNRESOLVED: "A writer could not be proven stopped.",
    ReasonCode.INVALID_INPUT: "Recovery input is invalid.",
    ReasonCode.DOCKER_UNAVAILABLE: "Docker did not answer.",
    ReasonCode.UNAUTHORIZED: "Container ownership could not be verified.",
}

# The singleton locks this process holds, by resolved state directory.
_HELD: dict[str, SingletonLock] = {}


class RecoveryError(Exception):
    """Closed-reason failure; the message never carries paths or credentials."""

    def __init__(self, reason_code: ReasonCode) -> None:
        self.reason_code = ReasonCode.parse(reason_code)
        super().__init__(_MESSAGES.get(self.reason_code, "Recovery operation failed."))


# --------------------------------------------------------------------- identity


@dataclass(frozen=True, slots=True)
class ProcessIdentity:
    pid: int
    create_time: float


ProcessIdentityProbe = Callable[[int], "ProcessIdentity | None"]


def psutil_identity(pid: int) -> ProcessIdentity | None:
    """Production probe: ``None`` only when the process provably does not exist."""
    import psutil

    try:
        return ProcessIdentity(pid, psutil.Process(pid).create_time())
    except psutil.NoSuchProcess:
        return None


def _same_process(recorded: ProcessIdentity, current: ProcessIdentity | None) -> bool:
    return (current is not None and current.pid == recorded.pid
            and abs(current.create_time - recorded.create_time) <= _IDENTITY_TOLERANCE)


# ----------------------------------------------------------------------- singleton


class SingletonLock:
    """Exclusive OS lock on ``<state_dir>/podmuncher.lock`` for the process lifetime."""

    def __init__(self, fd: int, key: str) -> None:
        self._fd: int | None = fd
        self._key = key

    @property
    def held(self) -> bool:
        return self._fd is not None

    def release(self) -> None:
        """Idempotent.  The lock file is never deleted (it may belong to a successor)."""
        fd, self._fd = self._fd, None
        if fd is None:
            return
        if _HELD.get(self._key) is self:
            del _HELD[self._key]
        try:
            _unlock(fd)
        except OSError:
            pass
        finally:
            try:
                os.close(fd)
            except OSError:
                pass

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()


def _lock(fd: int) -> None:
    os.lseek(fd, 0, os.SEEK_SET)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)


def _unlock(fd: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


def acquire_singleton(state_dir: str | os.PathLike) -> SingletonLock:
    """Take the non-blocking exclusive lock; contention means a duplicate daemon."""
    directory = os.fspath(state_dir)
    try:
        key = os.path.normcase(os.path.realpath(directory))
        fd = os.open(os.path.join(directory, LOCK_NAME),
                     os.O_RDWR | os.O_CREAT | _BINARY | _CLOEXEC, 0o600)
    except OSError:
        raise RecoveryError(ReasonCode.INVALID_CONFIG) from None
    try:
        _lock(fd)
    except OSError:
        os.close(fd)
        raise RecoveryError(ReasonCode.CONFLICT) from None
    lock = SingletonLock(fd, key)
    _HELD[key] = lock
    return lock


def _require_lock_for(path: str) -> None:
    key = os.path.normcase(os.path.realpath(os.path.dirname(os.path.abspath(path))))
    lock = _HELD.get(key)
    if lock is None or not lock.held:
        raise RecoveryError(ReasonCode.CONFLICT)


# -------------------------------------------------------------- private file helpers


def _windows_principal() -> str:
    domain = os.environ.get("USERDOMAIN")
    user = getpass.getuser()
    return f"{domain}\\{user}" if domain else user


def _icacls_exe() -> str:
    return os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "icacls.exe")


def _run_icacls(*args: str) -> str:
    try:
        done = subprocess.run((_icacls_exe(), *args), shell=False, capture_output=True,
                              timeout=ICACLS_TIMEOUT_SECONDS, check=False, stdin=subprocess.DEVNULL,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        raise RecoveryError(ReasonCode.INVALID_CONFIG) from None
    if done.returncode != 0:
        raise RecoveryError(ReasonCode.INVALID_CONFIG)
    return done.stdout.decode("utf-8", errors="replace")


def _parse_icacls(output: str, path: str) -> list[tuple[str, str]]:
    """ACEs as (principal, rights) from ``icacls <path>`` output; closed on any surprise."""
    aces: list[tuple[str, str]] = []
    prefix = os.path.normcase(path)
    for number, line in enumerate(output.splitlines()):
        text = line.strip()
        if not text or text.startswith("Successfully processed"):
            continue
        if number == 0:
            if not os.path.normcase(text).startswith(prefix):
                raise ValueError
            text = text[len(path):].strip()
        match = _ACE.fullmatch(text)
        if match is None:
            raise ValueError
        aces.append((match["who"], match["rights"]))
    return aces


def verify_runtime_permissions(path: str | os.PathLike) -> bool:
    """POSIX: regular file, mode 0600, owned by us.  Windows: exactly one ACE, ours."""
    target = os.fspath(path)
    try:
        info = os.lstat(target)
        if not stat.S_ISREG(info.st_mode) or _is_reparse(info):
            return False
        if os.name != "nt":
            return stat.S_IMODE(info.st_mode) == 0o600 and info.st_uid == os.getuid()
        aces = _parse_icacls(_run_icacls(target), target)
    except (OSError, ValueError, RecoveryError):
        return False
    return len(aces) == 1 and aces[0][0].casefold() == _windows_principal().casefold()


def _is_reparse(info: os.stat_result) -> bool:
    attributes = getattr(info, "st_file_attributes", 0)
    return stat.S_ISLNK(info.st_mode) or bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _write_private(path: str, payload: bytes) -> None:
    """Create-exclusive temp, restrict it, verify, write, fsync, then replace."""
    directory, name = os.path.split(os.path.abspath(path))
    temp = os.path.join(directory, f".{name}.{uuid4().hex}.tmp")
    created = False
    try:
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | _BINARY | _NOFOLLOW | _CLOEXEC, 0o600)
        created = True
        try:
            if os.name == "nt":
                os.close(fd)
                fd = -1
                _run_icacls(temp, "/inheritance:r", "/grant:r", f"{_windows_principal()}:(F)")
                if not verify_runtime_permissions(temp):
                    raise RecoveryError(ReasonCode.INVALID_CONFIG)
                fd = os.open(temp, os.O_WRONLY | _BINARY | _NOFOLLOW)
            else:
                os.fchmod(fd, 0o600)
            with os.fdopen(fd, "wb") as stream:
                fd = -1
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            if fd >= 0:
                os.close(fd)
        os.replace(temp, path)
        created = False
    except OSError:
        raise RecoveryError(ReasonCode.INVALID_CONFIG) from None
    finally:
        if created:
            try:
                os.unlink(temp)
            except OSError:
                pass


def _read_private(path: str, limit: int = MAX_FILE_BYTES) -> bytes | None:
    """``None`` when absent; ``INVALID_CONFIG`` when not private or not a plain file."""
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return None
    except OSError:
        raise RecoveryError(ReasonCode.INVALID_CONFIG) from None
    if not verify_runtime_permissions(path):
        raise RecoveryError(ReasonCode.INVALID_CONFIG)
    if info.st_size > limit:
        return b""
    try:
        with open(path, "rb") as stream:
            return stream.read(limit + 1)[:limit]
    except OSError:
        raise RecoveryError(ReasonCode.INVALID_CONFIG) from None


# ------------------------------------------------------------------ runtime record


@dataclass(frozen=True, slots=True)
class RuntimeRecord:
    schema_version: int
    port: int
    pid: int
    instance_nonce: str
    credential: str = field(repr=False)
    process_identity: ProcessIdentity


@dataclass(frozen=True, slots=True)
class RuntimeInspection:
    record: RuntimeRecord | None
    live: bool


def write_runtime_file(path: str | os.PathLike, *, port: int, credential: str, instance_nonce: str,
                       identity: ProcessIdentity) -> None:
    """Write the private runtime file.  Only valid while this process holds the singleton lock."""
    target = os.fspath(path)
    _require_lock_for(target)
    if (type(port) is not int or not 0 < port < 65536 or type(credential) is not str
            or not 0 < len(credential) <= 512 or type(instance_nonce) is not str
            or _NONCE.fullmatch(instance_nonce) is None or type(identity) is not ProcessIdentity
            or type(identity.pid) is not int or identity.pid <= 0):
        raise RecoveryError(ReasonCode.INVALID_INPUT)
    document = {"schema_version": RUNTIME_SCHEMA, "port": port, "pid": identity.pid,
                "instance_nonce": instance_nonce, "credential": credential,
                "process_identity": {"pid": identity.pid, "create_time": identity.create_time}}
    _write_private(target, json.dumps(document, sort_keys=True).encode("utf-8"))


def _parse_record(raw: bytes) -> RuntimeRecord | None:
    try:
        doc = json.loads(raw.decode("utf-8"))
        ident = doc["process_identity"]
        record = RuntimeRecord(doc["schema_version"], doc["port"], doc["pid"], doc["instance_nonce"],
                               doc["credential"], ProcessIdentity(ident["pid"], float(ident["create_time"])))
        valid = (
            type(doc) is dict and record.schema_version == RUNTIME_SCHEMA
            and type(record.port) is int and 0 < record.port < 65536
            and type(record.pid) is int and record.pid > 0
            and type(record.instance_nonce) is str and _NONCE.fullmatch(record.instance_nonce) is not None
            and type(record.credential) is str and 0 < len(record.credential) <= 512
            and type(ident["pid"]) is int and ident["pid"] == record.pid
        )
    except (ValueError, KeyError, TypeError, AttributeError, UnicodeDecodeError):
        return None
    return record if valid else None


def inspect_runtime_file(path: str | os.PathLike, *, process_identity: ProcessIdentityProbe) -> RuntimeInspection:
    """Parse the record and decide liveness.  Bad permissions raise ``INVALID_CONFIG``."""
    raw = _read_private(os.fspath(path))
    record = None if raw is None else _parse_record(raw)
    if record is None:
        return RuntimeInspection(None, False)
    try:
        current = process_identity(record.pid)
    except Exception:  # noqa: BLE001 - an unknown probe answer must count as live
        return RuntimeInspection(record, True)
    return RuntimeInspection(record, _same_process(record.process_identity, current))


def read_runtime_file(path: str | os.PathLike, *, process_identity: ProcessIdentityProbe,
                      expected_nonce: str | None = None) -> RuntimeRecord | None:
    """The live record, or ``None`` when missing, invalid, stale, a reused pid or a wrong nonce."""
    inspection = inspect_runtime_file(path, process_identity=process_identity)
    if inspection.record is None or not inspection.live:
        return None
    if expected_nonce is not None and inspection.record.instance_nonce != expected_nonce:
        return None
    return inspection.record


# ------------------------------------------------------------------------- intent


def write_intent(state_dir: str | os.PathLike, intent: Mapping[str, object]) -> None:
    """Durable last critical state (the safety-path record when audit fails)."""
    try:
        payload = json.dumps(dict(intent), sort_keys=True, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError):
        raise RecoveryError(ReasonCode.INVALID_INPUT) from None
    if len(payload) > MAX_FILE_BYTES:
        raise RecoveryError(ReasonCode.INVALID_INPUT)
    _write_private(os.path.join(os.fspath(state_dir), INTENT_FILE), payload)


def read_intent(state_dir: str | os.PathLike) -> dict | None:
    raw = _read_private(os.path.join(os.fspath(state_dir), INTENT_FILE))
    if raw is None:
        return None
    try:
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None
    return data if type(data) is dict else None


# ------------------------------------------------------------------ docker port


@dataclass(frozen=True, slots=True)
class ContainerRef:
    container_id: str
    labels: Mapping[str, str] = field(default_factory=dict)


class OwnedContainers(Protocol):
    """Docker port.  Implementations must act only on containers they re-verify by label."""

    async def list(self, instance: str) -> Sequence[ContainerRef]:
        """All containers (running or not) labelled ``podmuncher.instance=<instance>``."""

    async def stop(self, ref: ContainerRef) -> None: ...

    async def kill(self, ref: ContainerRef) -> None: ...

    async def remove(self, ref: ContainerRef) -> None: ...

    async def exists(self, ref: ContainerRef) -> bool | None:
        """``True``/``False`` only on proof; ``None`` when unknown."""


class NativeOwnedContainers:
    """Production adapter over ``runtime.NativeDockerCLI``; inspects labels before every action."""

    def __init__(self, cli: object, *, owner: str, timeout: float = 15) -> None:
        self._cli, self._owner, self._timeout = cli, owner, timeout

    async def _run(self, *args: str):
        return await self._cli.run(*args, owner=self._owner, timeout=self._timeout)  # type: ignore[attr-defined]

    async def _labels(self, container_id: str) -> Mapping[str, str] | None:
        result = await self._run("container", "inspect", container_id)
        if result.returncode:
            return None
        try:
            data, = json.loads(result.stdout)
            labels = data["Config"]["Labels"] or {}
            if data["Id"] != container_id or not all(type(v) is str for v in labels.values()):
                return None
        except (ValueError, TypeError, KeyError):
            return None
        return labels

    async def list(self, instance: str) -> Sequence[ContainerRef]:
        if _NONCE.fullmatch(instance) is None:
            raise RecoveryError(ReasonCode.INVALID_INPUT)
        result = await self._run("ps", "-a", "--no-trunc", "--format", "{{.ID}}",
                                 "--filter", f"label={LABEL_INSTANCE}={instance}")
        if result.returncode:
            raise RecoveryError(ReasonCode.DOCKER_UNAVAILABLE)
        refs = []
        for line in result.stdout.decode("ascii", errors="replace").splitlines():
            if _CONTAINER_ID.fullmatch(line) is None:
                raise RecoveryError(ReasonCode.DOCKER_UNAVAILABLE)
            labels = await self._labels(line)
            if labels is None:
                # ps listed it but its labels could not be read: only proven absence drops it.
                if await self._absent(line):
                    continue
                raise RecoveryError(ReasonCode.UNAUTHORIZED)  # unverified; never acted on
            if labels.get(LABEL_INSTANCE) == instance:
                refs.append(ContainerRef(line, dict(labels)))
        return refs

    async def _absent(self, container_id: str) -> bool:
        """True only when an engine-side listing answers cleanly and shows no such container."""
        try:
            listing = await self._run("ps", "-a", "-q", "--no-trunc", "--filter", f"id={container_id}")
        except Exception:  # noqa: BLE001 - an unanswered check is not proof
            return False
        return listing.returncode == 0 and not listing.stdout.strip()

    async def _act(self, ref: ContainerRef, *args: str) -> None:
        labels = await self._labels(ref.container_id)
        if labels is None or labels.get(LABEL_INSTANCE) != ref.labels.get(LABEL_INSTANCE):
            raise RecoveryError(ReasonCode.UNAUTHORIZED)
        result = await self._run(*args, ref.container_id)
        if result.returncode:
            raise RecoveryError(ReasonCode.DOCKER_UNAVAILABLE)

    async def stop(self, ref: ContainerRef) -> None:
        await self._act(ref, "stop", "--time", "1")

    async def kill(self, ref: ContainerRef) -> None:
        await self._act(ref, "kill")

    async def remove(self, ref: ContainerRef) -> None:
        await self._act(ref, "rm")

    async def exists(self, ref: ContainerRef) -> bool | None:
        inspected = await self._run("container", "inspect", ref.container_id)
        if inspected.returncode == 0:
            return True
        # A failed inspect alone is not proof; confirm with an engine-side listing.
        listing = await self._run("ps", "-a", "-q", "--no-trunc", "--filter", f"id={ref.container_id}")
        if listing.returncode == 0 and not listing.stdout.strip():
            return False
        return None


# ------------------------------------------------------------------------- reports


class Clock(Protocol):
    def now(self) -> str: ...


Audit = Callable[[str, str | None, ReasonCode], None]


@dataclass(frozen=True, slots=True)
class AuditLossNotice:
    sequence: int
    action: str
    reason_code: ReasonCode


class Recovery:
    """Process-wide stop flag and the bounded in-memory audit-loss notices."""

    def __init__(self, state_dir: str | os.PathLike | None = None, *, instance_nonce: str | None = None) -> None:
        self.state_dir = None if state_dir is None else os.fspath(state_dir)
        self.instance_nonce = instance_nonce
        self.accepting = True
        self._loss: deque[AuditLossNotice] = deque(maxlen=AUDIT_LOSS_MAX)
        self._sequence = 0

    @property
    def audit_loss(self) -> tuple[AuditLossNotice, ...]:
        return tuple(self._loss)

    @property
    def audit_loss_total(self) -> int:
        return self._sequence

    def note_audit_loss(self, action: str, reason: ReasonCode) -> None:
        self._sequence += 1
        self._loss.append(AuditLossNotice(self._sequence, action, reason))


@dataclass(frozen=True, slots=True)
class StopReport:
    ok: bool
    reason_code: ReasonCode
    interrupted: int
    queued_retained: int
    blocked_retained: int
    leases_released: int
    lease_retained: bool
    audit_delivered: bool  # the sink accepted the call; never a durability claim
    audit_loss: int
    intent_written: bool


@dataclass(frozen=True, slots=True)
class ReconcileReport:
    ok: bool
    reason_code: ReasonCode
    survivors_found: int
    survivors_cleared: int
    survivors_unresolved: int
    foreign_skipped: int
    interrupted: int
    leases_released: int
    partial_files_removed: int
    reclaimed_bytes: int
    manifests_checked: int
    manifest_mismatches: int
    mismatched_job_ids: tuple[str, ...]  # protected join evidence; never a metric label
    admission_latch: ReasonCode | None
    audit_delivered: bool
    audit_loss: int
    manifest_scan_truncated: bool = False  # the hard job cap was hit; later jobs were not checked


# -------------------------------------------------------------------- shared steps


def _iter_jobs(repository: Repository) -> Iterator:
    cursor = None
    while True:
        jobs, cursor = repository.list_jobs(PAGE_MAX, cursor)
        yield from jobs
        if cursor is None:
            return


def _nonces_in(document: object) -> set[str]:
    """Every valid instance nonce an intent-shaped mapping names."""
    found: set[str] = set()
    if not isinstance(document, Mapping):
        return found
    candidates = [document.get("instance_nonce")]
    listed = document.get("instance_nonces")
    if isinstance(listed, (list, tuple)):
        candidates.extend(listed)
    for nonce in candidates:
        if type(nonce) is str and _NONCE.fullmatch(nonce):
            found.add(nonce)
    return found


def _write_intent_merged(state_dir: str, snapshot: Mapping[str, object], clock: Clock, **extra: object) -> None:
    """Write the intent without ever dropping an instance nonce already recorded on disk."""
    try:
        existing = read_intent(state_dir)
    except RecoveryError:
        existing = None
    nonces = _nonces_in(existing) | _nonces_in(snapshot)
    document = dict(snapshot)
    if nonces:
        primary = document.get("instance_nonce")
        document["instance_nonce"] = primary if primary in nonces else min(nonces)
        document["instance_nonces"] = sorted(nonces)
    write_intent(state_dir, {**document, **extra, "ts": clock.now()})


def _audit_safe(recovery: Recovery, audit: Audit | None, action: str, reason: ReasonCode,
                clock: Clock, snapshot: Mapping[str, object]) -> tuple[bool, bool]:
    """Call the sink; on failure keep a bounded notice and a durable snapshot.

    Any exception from the sink (including ``TelemetryError(AUDIT_LOSS)`` from the telemetry
    adapter) is a loss, never a delivery.  Returns (delivered, intent_written); neither is a
    durability claim.
    """
    delivered = False
    if audit is not None:
        try:
            audit(action, None, reason)
            delivered = True
        except Exception:  # noqa: BLE001 - the sink is untrusted and must never block safety
            recovery.note_audit_loss(action, reason)
    written = False
    if not delivered and recovery.state_dir is not None:
        try:
            _write_intent_merged(recovery.state_dir, snapshot, clock, audit_lost=True)
            written = True
        except RecoveryError:
            pass
    return delivered, written


@dataclass(slots=True)
class _Tally:
    interrupted: int = 0
    released: int = 0


def _interrupt_job(repository: Repository, clock: Clock, job, event_type,
                   release_op: str | None) -> tuple[bool, bool]:
    """Interrupt one job whose writers are proven reaped.  Returns (interrupted, lease_released).

    With an attempt row the job, the attempt and (when ``release_op`` is given) its lease change
    in one closing transaction.  A job whose attempt row is missing is interrupted on its own;
    its lease is released afterwards by the caller.
    """
    for _ in range(2):  # one retry after a compare-and-set conflict
        job = repository.get_job(job.job_id) or job
        if job.state not in tr._ACTIVE:
            return False, False
        version = repository.get_job_version(job.job_id)
        if version is None:
            return False, False
        attempt = None if job.attempt_id is None else repository.get_attempt(job.attempt_id)
        now = clock.now()
        if attempt is None:
            reason = ReasonCode.SERVICE_STOPPED if event_type is tr.ServiceStopped else ReasonCode.SERVICE_LOST
            final_job = replace(job, state=JobState.INTERRUPTED, reason_code=reason, next_action=None,
                                updated_at=now)
            final_attempt = None
        else:
            reaped = tr.reduce(job, attempt, tr.WriterReaped(job.job_id, True, attempt.attempt_id), now=now)
            final = tr.reduce(reaped.job, reaped.attempt, event_type(job.job_id), now=now)
            final_job, final_attempt = final.job, final.attempt
        try:
            if final_attempt is None:
                repository.update_job(final_job, expected_version=version, closing=True)
                return True, False
            repository.close_attempt_and_release(
                final_job, final_attempt, expected_version=version,
                release=None if release_op is None else (release_op, True))
            return True, release_op is not None
        except RepositoryError as error:
            if error.reason_code is ReasonCode.CONFLICT:
                continue
            raise
    return False, False


def _interrupt_all(repository: Repository, clock: Clock, event_type, tally: _Tally) -> None:
    """Interrupt every active job (paged, bounded), then release every remaining lease.

    Call only after every owned writer is proven reaped.  The lease is released inside the
    job's own closing transaction when exactly one job and one lease exist; the rest are
    released in a final pass once no active job remains.
    """
    for _ in range(INTERRUPT_ROUNDS_MAX):
        active = repository.active_jobs(PAGE_MAX)
        if not active:
            break
        holders = [operation_id for operation_id, _unreaped in repository.lease_holders()]
        progress = False
        for job in active:
            paired = holders[0] if len(active) == 1 and len(holders) == 1 else None
            done, released = _interrupt_job(repository, clock, job, event_type, paired)
            progress = progress or done
            tally.interrupted += done
            tally.released += released
        if not progress:
            raise RepositoryError(ReasonCode.CONFLICT)
    else:
        raise RepositoryError(ReasonCode.CONFLICT)
    for operation_id, _unreaped in repository.lease_holders():
        repository.release(operation_id, writers_reaped=True)
        tally.released += 1


# ---------------------------------------------------------------------- graceful stop


def _reason_of(error: BaseException) -> ReasonCode:
    reason = getattr(error, "reason_code", None)
    return reason if type(reason) is ReasonCode else ReasonCode.STATE_WRITE_FAILED


async def graceful_stop(scheduler: Scheduler | None, repository: Repository, launcher: WorkerLauncher, *,
                        audit: Audit | None, clock: Clock, recovery: Recovery | None = None,
                        quiesce_timeout: float = QUIESCE_TIMEOUT_SECONDS) -> StopReport:
    """Reject new work, quiesce the scheduler, stop the owned writer, then record the interruption.

    Order: admission closes; ``scheduler.quiesce`` stops and awaits setup/run/export work;
    ``launcher.stop`` reaps the writer; only then is the repository read.  Unless both prove
    every writer reaped, nothing is interrupted and every lease stays (``writer_unresolved``).
    A repository read failure after the stop is reported with its reason, never as success.
    """
    recovery = recovery if recovery is not None else Recovery()
    recovery.accepting = False
    snapshot: dict[str, object] = {"action": "stop", "instance_nonce": recovery.instance_nonce,
                                   "active_job_id": None, "attempt_id": None}
    if recovery.state_dir is not None:
        try:
            _write_intent_merged(recovery.state_dir, snapshot, clock)
        except RecoveryError:
            pass
    quiesced = True
    if scheduler is not None:
        try:
            quiesced = await scheduler.quiesce(timeout=quiesce_timeout) is True
        except Exception:  # noqa: BLE001 - work that did not quiesce is an unreaped writer
            quiesced = False
    try:
        stopped = bool(await launcher.stop(ReasonCode.SERVICE_STOPPED))
    except Exception:  # noqa: BLE001 - a failed stop is an unreaped writer
        stopped = False
    reaped = quiesced and stopped

    active: list = []
    queued = blocked = 0
    read_failure: ReasonCode | None = None
    try:
        active = repository.active_jobs(PAGE_MAX)
        repository.lease_holders()
        queued = len(repository.queued_jobs(PAGE_MAX))  # bounded counts: no history scan
        blocked = sum(1 for job in repository.list_jobs(PAGE_MAX)[0] if job.state is JobState.BLOCKED)
    except Exception as error:  # noqa: BLE001 - reported below; never a success claim
        read_failure = _reason_of(error)
    if active:
        snapshot["active_job_id"], snapshot["attempt_id"] = active[0].job_id, active[0].attempt_id

    tally = _Tally()
    reason = ReasonCode.SERVICE_STOPPED
    if not reaped:
        reason = ReasonCode.WRITER_UNRESOLVED
        for job in active:  # record the unresolved reason; the lease stays
            try:
                attempt = None if job.attempt_id is None else repository.get_attempt(job.attempt_id)
                if attempt is not None:
                    decision = tr.reduce(job, attempt, tr.WriterReaped(job.job_id, False, attempt.attempt_id),
                                         now=clock.now())
                    repository.update_job(decision.job, expected_version=repository.get_job_version(job.job_id))
            except (RepositoryError, tr.TransitionError):
                pass
    elif read_failure is not None:
        reaped, reason = False, read_failure
    else:
        try:
            _interrupt_all(repository, clock, tr.ServiceStopped, tally)
        except (RepositoryError, tr.TransitionError) as error:
            reaped, reason = False, _reason_of(error)
    try:
        lease_retained = bool(repository.lease_holders())
    except Exception:  # noqa: BLE001 - unknown means retained
        lease_retained = True
    delivered, written = _audit_safe(recovery, audit, "service_stop", reason, clock, snapshot)
    return StopReport(
        ok=reaped, reason_code=reason, interrupted=tally.interrupted,
        queued_retained=queued, blocked_retained=blocked,
        leases_released=tally.released, lease_retained=lease_retained,
        audit_delivered=delivered, audit_loss=recovery.audit_loss_total, intent_written=written,
    )


# ---------------------------------------------------------------- startup reconcile


async def _reap_container(containers: OwnedContainers, ref: ContainerRef) -> bool:
    for action in (containers.stop, containers.kill, containers.remove):
        with contextlib.suppress(Exception):  # best effort; absence is checked below
            await action(ref)
    try:
        return await containers.exists(ref) is False
    except Exception:  # noqa: BLE001
        return False


def _owned_instances(state_dir: str, process_identity: ProcessIdentityProbe) -> set[str]:
    instances: set[str] = set()
    inspection = inspect_runtime_file(os.path.join(state_dir, RUNTIME_FILE), process_identity=process_identity)
    if inspection.record is not None and inspection.live:
        raise RecoveryError(ReasonCode.CONFLICT)
    if inspection.record is not None:
        instances.add(inspection.record.instance_nonce)
    intent = read_intent(state_dir)
    instances |= _nonces_in(intent)
    return instances


def _real_dir(path: str) -> bool:
    try:
        info = os.lstat(path)
    except OSError:
        return False
    return stat.S_ISDIR(info.st_mode) and not _is_reparse(info)


def _clean_partials(roots: Iterable[str], job_ids: frozenset[str]) -> tuple[int, int]:
    removed = reclaimed = 0
    for root in roots:
        try:
            names = os.listdir(root)
        except OSError:
            continue
        for name in names:
            if not name.startswith("podmuncher-") or name[len("podmuncher-"):] not in job_ids:
                continue
            job_dir = os.path.join(root, name)
            if not _real_dir(job_dir):
                continue
            try:
                entries = list(os.scandir(job_dir))
            except OSError:
                continue
            for entry in entries:
                if _PARTIAL.fullmatch(entry.name) is None:
                    continue
                try:
                    info = os.lstat(entry.path)
                    if not stat.S_ISREG(info.st_mode) or _is_reparse(info):
                        continue
                    os.unlink(entry.path)
                except OSError:
                    continue
                removed += 1
                reclaimed += info.st_size
    return removed, reclaimed


def _find_job_dir(roots: Iterable[str], job_id: str) -> str | None:
    for root in roots:
        candidate = os.path.join(root, f"podmuncher-{job_id}")
        if _real_dir(candidate):
            return candidate
    return None


def _manifest_ok(job_dir: str, available_ids: set[str]) -> bool:
    path = os.path.join(job_dir, MANIFEST_NAME)
    try:
        info = os.lstat(path)
        if not stat.S_ISREG(info.st_mode) or _is_reparse(info) or info.st_size > MAX_MANIFEST_BYTES:
            return False
        with open(path, "rb") as stream:
            doc = json.loads(stream.read().decode("utf-8"))
        if type(doc) is not dict or doc.get("schema_version") != MANIFEST_SCHEMA:
            return False
        listed = set()
        for item in doc["artifacts"]:
            name = item["file_name"]
            if type(name) is not str or name != os.path.basename(name) or name in ("", ".", ".."):
                return False
            file_info = os.lstat(os.path.join(job_dir, name))
            if (not stat.S_ISREG(file_info.st_mode) or _is_reparse(file_info)
                    or file_info.st_size != item["bytes"]):
                return False
            listed.add(item["artifact_id"])
    except (OSError, ValueError, KeyError, TypeError, UnicodeDecodeError):
        return False
    return available_ids <= listed


async def startup_reconcile(state_dir: str | os.PathLike, repository: Repository,
                            containers: OwnedContainers, *, process_identity: ProcessIdentityProbe,
                            audit: Audit | None, clock: Clock, output_roots: Sequence[str | os.PathLike],
                            recovery: Recovery | None = None) -> ReconcileReport:
    """Reconcile a previous instance.  The caller already holds the singleton lock."""
    directory = os.fspath(state_dir)
    roots = tuple(os.fspath(root) for root in output_roots)
    recovery = recovery if recovery is not None else Recovery(directory)
    instances = _owned_instances(directory, process_identity)  # CONFLICT when still live
    known = set(instances)
    if recovery.instance_nonce is not None:
        known.add(recovery.instance_nonce)  # an audit failure must never drop a known nonce
    found = cleared = unresolved = foreign = 0

    def report(reason: ReasonCode, **values: object) -> ReconcileReport:
        snapshot = {"action": "running", "instance_nonce": recovery.instance_nonce,
                    "instance_nonces": sorted(known)}
        delivered, _written = _audit_safe(recovery, audit, "startup_reconcile", reason, clock, snapshot)
        base = {
            "ok": reason is ReasonCode.OK, "reason_code": reason, "survivors_found": found,
            "survivors_cleared": cleared, "survivors_unresolved": unresolved, "foreign_skipped": foreign,
            "interrupted": 0, "leases_released": 0, "partial_files_removed": 0, "reclaimed_bytes": 0,
            "manifests_checked": 0, "manifest_mismatches": 0, "mismatched_job_ids": (),
            "admission_latch": repository.admission_blocked, "audit_delivered": delivered,
            "audit_loss": recovery.audit_loss_total}
        base.update(values)
        return ReconcileReport(**base)

    try:
        active = repository.active_jobs(PAGE_MAX)
        holders = repository.lease_holders()
    except RepositoryError as error:
        return report(error.reason_code)

    unidentified = bool(active or holders) and not instances
    for instance in sorted(instances):
        try:
            refs = list(await containers.list(instance))
        except Exception:  # noqa: BLE001 - an unanswered or unverifiable listing is not proof of absence
            unresolved += 1
            continue
        for ref in refs:
            if ref.labels.get(LABEL_INSTANCE) != instance:
                foreign += 1  # never touched
                continue
            found += 1
            if await _reap_container(containers, ref):
                cleared += 1
            else:
                unresolved += 1

    if unresolved or unidentified:
        return report(ReasonCode.WRITER_UNRESOLVED)  # every lease stays; nothing is interrupted

    tally = _Tally()
    try:
        _interrupt_all(repository, clock, tr.ServiceLost, tally)
    except (RepositoryError, tr.TransitionError) as error:
        return report(_reason_of(error), interrupted=tally.interrupted, leases_released=tally.released)

    jobs: list = []
    truncated = False
    try:
        for job in _iter_jobs(repository):
            if len(jobs) >= MANIFEST_SCAN_JOBS_MAX:
                truncated = True
                break
            jobs.append(job)
    except RepositoryError as error:
        return report(error.reason_code, interrupted=tally.interrupted, leases_released=tally.released)

    removed, reclaimed = _clean_partials(roots, frozenset(job.job_id for job in jobs))
    checked, mismatched = 0, []
    for job in jobs:
        try:
            artifacts = repository.list_artifacts(job.job_id)
        except RepositoryError:
            continue
        if not artifacts:
            continue
        checked += 1
        job_dir = _find_job_dir(roots, job.job_id)
        available = {a.artifact_id for a in artifacts if a.available}
        if job_dir is None or not _manifest_ok(job_dir, available):
            mismatched.append(job.job_id)
    return report(ReasonCode.OK, interrupted=tally.interrupted, leases_released=tally.released,
                  partial_files_removed=removed, reclaimed_bytes=reclaimed, manifests_checked=checked,
                  manifest_mismatches=len(mismatched), mismatched_job_ids=tuple(mismatched[:MISMATCH_IDS_MAX]),
                  manifest_scan_truncated=truncated)


__all__ = (
    "AUDIT_LOSS_MAX", "INTENT_FILE", "LOCK_NAME", "RUNTIME_FILE", "AuditLossNotice", "ContainerRef",
    "NativeOwnedContainers", "OwnedContainers", "ProcessIdentity", "ReconcileReport", "Recovery",
    "RecoveryError", "RuntimeInspection", "RuntimeRecord", "SingletonLock", "StopReport",
    "acquire_singleton", "graceful_stop", "inspect_runtime_file", "psutil_identity", "read_intent",
    "read_runtime_file", "startup_reconcile", "verify_runtime_permissions", "write_intent",
    "write_runtime_file",
)
