"""Local-only, single-writer SQLite state repository.

One ``Repository`` instance owns the only connection.  It exposes typed methods
and no SQL entry point, so client code has no write path.  Every write is one
``BEGIN IMMEDIATE`` transaction; any sqlite3/OS failure rolls back, latches
admission with ``STATE_WRITE_FAILED`` and raises, never reporting a lost commit
as success.  The repository never reads or writes outside its own database
files and never deletes artifact files; retention removes artifact rows only.
Which job transitions are legal is decided elsewhere; this layer only provides
compare-and-set on a version so late writers cannot overwrite newer state.
"""
from __future__ import annotations

import base64
import binascii
import json
import os
import sqlite3
import threading
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Self

from .admission import AdmissionPlan
from .config import Limits
from .domains import (
    DomainValidationError,
    JobState,
    ReasonCode,
    validate_utc_timestamp,
    validate_uuid_string,
)
from .snapshots import Artifact, Attempt, Job, StageRecord

MIB = 1024 * 1024
DB_BOUND = 128 * MIB
MAINTENANCE_ALLOWANCE = 128 * MIB
RETENTION_DAYS = 90
RETENTION_JOBS = 10000
PAGE_MAX = 100
KEY_MAX = 200

# States that count as finished for retention.  Interrupted is deliberately absent.
_FINISHED = (JobState.COMPLETED, JobState.PARTIAL, JobState.FAILED, JobState.CANCELLED)
_FINISHED_VALUES = tuple(s.value for s in _FINISHED)
# An attempt is no longer active once it is finished or interrupted.
_ATTEMPT_DONE = frozenset(_FINISHED + (JobState.INTERRUPTED,))
# Job states a closing write may leave a job in; closing writes bypass the size bound.
_CLOSING_JOB = _ATTEMPT_DONE | {JobState.BLOCKED}
# Non-terminal, non-queued, non-blocked states: a job in one of these holds an attempt.
_ACTIVE_VALUES = tuple(st.value for st in (JobState.CHECKING, JobState.PREPARING, JobState.DOWNLOADING,
                                           JobState.PROBING, JobState.TRANSCRIBING, JobState.EXPORTING))

_MIGRATIONS: tuple[tuple[str, ...], ...] = (
    (
        """CREATE TABLE jobs (
            job_id TEXT PRIMARY KEY, state TEXT NOT NULL, request_hash TEXT NOT NULL,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL, finished_at TEXT,
            version INTEGER NOT NULL, doc TEXT NOT NULL)""",
        """CREATE TABLE attempts (
            attempt_id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL REFERENCES jobs(job_id),
            ordinal INTEGER NOT NULL, retry_key TEXT UNIQUE, doc TEXT NOT NULL,
            UNIQUE (job_id, ordinal))""",
        """CREATE TABLE stages (
            stage_id TEXT PRIMARY KEY,
            attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id), doc TEXT NOT NULL)""",
        """CREATE TABLE artifacts (
            artifact_id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL REFERENCES jobs(job_id), doc TEXT NOT NULL)""",
        """CREATE TABLE idempotency_keys (
            key TEXT PRIMARY KEY, kind TEXT NOT NULL CHECK (kind IN ('submit', 'retry')),
            request_hash TEXT NOT NULL, target_id TEXT NOT NULL)""",
        """CREATE TABLE reservations (
            operation_id TEXT PRIMARY KEY, job_id TEXT, plan TEXT NOT NULL,
            acquired_at TEXT NOT NULL, writers_unreaped INTEGER NOT NULL)""",
        "CREATE INDEX jobs_paging ON jobs (created_at, job_id)",
        "CREATE INDEX jobs_finished ON jobs (finished_at)",
        "CREATE INDEX attempts_job ON attempts (job_id)",
        "CREATE INDEX stages_attempt ON stages (attempt_id)",
        "CREATE INDEX artifacts_job ON artifacts (job_id)",
    ),
    ("CREATE INDEX jobs_queue ON jobs (state, created_at, job_id)",),
)


class RepositoryError(Exception):
    """Closed-reason failure; the message never carries SQL, paths or request data."""

    def __init__(self, reason_code: ReasonCode) -> None:
        self.reason_code = ReasonCode.parse(reason_code)
        super().__init__("Repository operation failed.")


@dataclass(frozen=True, slots=True)
class MaintenanceResult:
    deleted: int
    bytes_before: int
    bytes_after: int
    blocked: ReasonCode | None


def _stamp(moment: datetime) -> str:
    if type(moment) is not datetime or moment.tzinfo is None:
        raise RepositoryError(ReasonCode.INVALID_INPUT)
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _key(value: object) -> str:
    if type(value) is not str or not 0 < len(value) <= KEY_MAX:
        raise RepositoryError(ReasonCode.INVALID_INPUT)
    return value


def _dump(data: dict) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _plan_doc(plan: AdmissionPlan) -> str:
    if type(plan) is not AdmissionPlan:
        raise RepositoryError(ReasonCode.INVALID_INPUT)
    return _dump({"admission": plan.admission.to_dict(), "reason": plan.reason.value,
                  "volume_count": len(plan.volumes), "interval_count": len(plan.intervals)})


def _encode_cursor(created_at: str, job_id: str) -> str:
    raw = _dump({"v": 1, "c": created_at, "j": job_id}).encode()
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def _decode_cursor(cursor: object) -> tuple[str, str]:
    try:
        if type(cursor) is not str or not cursor or len(cursor) > 512:
            raise ValueError
        data = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)))
        if type(data) is not dict or set(data) != {"v", "c", "j"} or data["v"] != 1 or type(data["v"]) is not int:
            raise ValueError
        return validate_utc_timestamp(data["c"]), validate_uuid_string(data["j"])
    except (ValueError, binascii.Error, DomainValidationError):
        raise RepositoryError(ReasonCode.INVALID_INPUT) from None


class Repository:
    """The only writer of service state.  Construct with :meth:`open`."""

    def __init__(self, conn: sqlite3.Connection, path: str, *, clock: Callable[[], datetime],
                 fault_hook: Callable[[str], None] | None, limits: Limits,
                 verified_models: frozenset[str] | set[str] | None,
                 verified_languages: frozenset[str] | set[str] | None) -> None:
        self._conn = conn
        self._path = path
        self._clock = clock
        self._fault_hook = fault_hook
        self._limits = limits
        self._models = verified_models
        self._languages = verified_languages
        self._lock = threading.Lock()
        self._latch: ReasonCode | None = None
        self._closed = False

    # ------------------------------------------------------------------ open

    @classmethod
    def open(cls, path: str | os.PathLike, *, clock: Callable[[], datetime],
             state_volume: Callable[[str], bool | None],
             fault_hook: Callable[[str], None] | None = None,
             limits: Limits | None = None,
             verified_models: frozenset[str] | set[str] | None = None,
             verified_languages: frozenset[str] | set[str] | None = None) -> Repository:
        db_path = os.fspath(path)
        try:
            allowed = state_volume(db_path)
        except Exception:  # noqa: BLE001 - any probe failure is an unknown volume
            raise RepositoryError(ReasonCode.STORAGE_UNKNOWN) from None
        if allowed is None:
            raise RepositoryError(ReasonCode.STORAGE_UNKNOWN)
        if allowed is not True:
            raise RepositoryError(ReasonCode.INVALID_CONFIG)
        conn = None
        old_umask = os.umask(0o077) if os.name == "posix" else None
        try:
            conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)
            # auto_vacuum only takes effect before the first table exists.
            conn.execute("PRAGMA auto_vacuum=INCREMENTAL")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute("PRAGMA foreign_keys=ON")
            cls._migrate(conn)
        except RepositoryError:
            if conn is not None:
                conn.close()
            raise
        except (sqlite3.Error, OSError):
            if conn is not None:
                conn.close()
            raise RepositoryError(ReasonCode.STATE_WRITE_FAILED) from None
        finally:
            if old_umask is not None:
                os.umask(old_umask)
        if os.name == "posix":
            for suffix in ("", "-wal", "-shm"):
                try:
                    os.chmod(db_path + suffix, 0o600)
                except OSError:
                    pass
        return cls(conn, db_path, clock=clock, fault_hook=fault_hook,
                   limits=limits if limits is not None else Limits(),
                   verified_models=verified_models, verified_languages=verified_languages)

    @staticmethod
    def _migrate(conn: sqlite3.Connection) -> None:
        current = conn.execute("PRAGMA user_version").fetchone()[0]
        if current > len(_MIGRATIONS):
            raise RepositoryError(ReasonCode.UNSUPPORTED_SCHEMA)
        if current == len(_MIGRATIONS):
            return
        conn.execute("BEGIN IMMEDIATE")
        try:
            for statements in _MIGRATIONS[current:]:
                for sql in statements:
                    conn.execute(sql)
            conn.execute(f"PRAGMA user_version={len(_MIGRATIONS)}")
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._closed = True
                self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------ infrastructure

    @property
    def admission_blocked(self) -> ReasonCode | None:
        return self._latch

    def db_bytes(self) -> int:
        total = 0
        for suffix in ("", "-wal"):
            try:
                total += os.path.getsize(self._path + suffix)
            except FileNotFoundError:
                pass
        return total

    def _require_open(self) -> None:
        if self._closed:
            raise RepositoryError(ReasonCode.SERVICE_STOPPED)

    def _check_bound(self) -> None:
        if self.db_bytes() > DB_BOUND:
            self._latch = ReasonCode.STORAGE_INSUFFICIENT
            raise RepositoryError(ReasonCode.STORAGE_INSUFFICIENT)

    def _rollback(self) -> None:
        try:
            if self._conn.in_transaction:
                self._conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass

    @contextmanager
    def _tx(self, op: str, *, check_bound: bool = True):
        """One locked BEGIN IMMEDIATE transaction with the fault point before COMMIT."""
        with self._lock:
            self._require_open()
            try:
                if check_bound:
                    self._check_bound()
                self._conn.execute("BEGIN IMMEDIATE")
                try:
                    yield self._conn
                    if self._fault_hook is not None:
                        self._fault_hook(op)
                    self._conn.execute("COMMIT")
                except BaseException:
                    self._rollback()
                    raise
            except RepositoryError:
                raise
            except sqlite3.IntegrityError:
                raise RepositoryError(ReasonCode.CONFLICT) from None
            except (sqlite3.Error, OSError):
                self._latch = ReasonCode.STATE_WRITE_FAILED
                raise RepositoryError(ReasonCode.STATE_WRITE_FAILED) from None
            except DomainValidationError as error:
                raise RepositoryError(error.reason_code) from None

    @contextmanager
    def _read(self):
        with self._lock:
            self._require_open()
            try:
                yield self._conn
            except RepositoryError:
                raise
            except DomainValidationError as error:
                raise RepositoryError(error.reason_code) from None
            except ValueError:  # undecodable stored JSON
                raise RepositoryError(ReasonCode.INVALID_INPUT) from None
            except (sqlite3.Error, OSError):
                raise RepositoryError(ReasonCode.SERVICE_UNAVAILABLE) from None

    def _now(self) -> str:
        return _stamp(self._clock())

    # ------------------------------------------------------------------ decoding

    def _job(self, row: tuple) -> Job:
        state, doc = row
        job = Job.from_dict(json.loads(doc), verified_models=self._models,
                            verified_languages=self._languages)
        if JobState.parse(state) is not job.state:
            raise RepositoryError(ReasonCode.INVALID_INPUT)
        return job

    def _attempt(self, doc: str) -> Attempt:
        return Attempt.from_dict(json.loads(doc))

    # -------------------------------------------------------------------- writes

    def submit(self, job: Job, idempotency_key: str) -> tuple[Job, bool]:
        key = _key(idempotency_key)
        if type(job) is not Job or job.state is not JobState.QUEUED:
            raise RepositoryError(ReasonCode.INVALID_INPUT)
        with self._tx("submit") as db:
            if self._latch is not None:
                raise RepositoryError(self._latch)
            row = db.execute("SELECT kind, request_hash, target_id FROM idempotency_keys WHERE key=?",
                             (key,)).fetchone()
            if row is not None:
                if row[0] != "submit" or row[1] != job.request_hash:
                    raise RepositoryError(ReasonCode.CONFLICT)
                stored = db.execute("SELECT state, doc FROM jobs WHERE job_id=?", (row[2],)).fetchone()
                if stored is None:
                    raise RepositoryError(ReasonCode.CONFLICT)
                return self._job(stored), True
            queued = db.execute("SELECT COUNT(*) FROM jobs WHERE state=?",
                                (JobState.QUEUED.value,)).fetchone()[0]
            if queued >= self._limits.pending_jobs:
                raise RepositoryError(ReasonCode.QUEUE_FULL)
            db.execute("INSERT INTO jobs (job_id, state, request_hash, created_at, updated_at,"
                       " finished_at, version, doc) VALUES (?,?,?,?,?,NULL,1,?)",
                       (job.job_id, job.state.value, job.request_hash, job.created_at,
                        job.updated_at, _dump(job.to_dict())))
            db.execute("INSERT INTO idempotency_keys (key, kind, request_hash, target_id)"
                       " VALUES (?,?,?,?)", (key, "submit", job.request_hash, job.job_id))
        return job, False

    def _update_job(self, db: sqlite3.Connection, job: Job, expected_version: int) -> None:
        if type(job) is not Job or type(expected_version) is not int or expected_version < 1:
            raise RepositoryError(ReasonCode.INVALID_INPUT)
        row = db.execute("SELECT request_hash, created_at, finished_at FROM jobs WHERE job_id=?",
                         (job.job_id,)).fetchone()
        if row is None or row[0] != job.request_hash or row[1] != job.created_at:
            raise RepositoryError(ReasonCode.CONFLICT)
        finished = None
        if job.state in _FINISHED:
            finished = row[2] if row[2] is not None else self._now()
        cursor = db.execute(
            "UPDATE jobs SET state=?, updated_at=?, finished_at=?, version=version+1, doc=?"
            " WHERE job_id=? AND version=?",
            (job.state.value, job.updated_at, finished, _dump(job.to_dict()), job.job_id,
             expected_version))
        if cursor.rowcount != 1:
            raise RepositoryError(ReasonCode.CONFLICT)

    @staticmethod
    def _require_closing_job(job: Job) -> None:
        if type(job) is not Job or job.state not in _CLOSING_JOB:
            raise RepositoryError(ReasonCode.INVALID_INPUT)

    @staticmethod
    def _require_closing_attempt(attempt: Attempt) -> None:
        if type(attempt) is not Attempt or attempt.state not in _ATTEMPT_DONE:
            raise RepositoryError(ReasonCode.INVALID_INPUT)

    def update_job(self, job: Job, *, expected_version: int, closing: bool = False) -> Job:
        if closing:
            self._require_closing_job(job)
        with self._tx("update_job", check_bound=not closing) as db:
            self._update_job(db, job, expected_version)
        return job

    def start_attempt(self, job_id: str, attempt: Attempt, *, retry_key: str | None,
                      expected_version: int) -> tuple[Attempt, bool]:
        validate_uuid_string(job_id)
        key = None if retry_key is None else _key(retry_key)
        if type(attempt) is not Attempt or attempt.job_id != job_id or type(expected_version) is not int:
            raise RepositoryError(ReasonCode.INVALID_INPUT)
        with self._tx("start_attempt") as db:
            if key is not None:
                row = db.execute("SELECT kind, target_id FROM idempotency_keys WHERE key=?",
                                 (key,)).fetchone()
                if row is not None:
                    stored = db.execute("SELECT job_id, doc FROM attempts WHERE attempt_id=?",
                                        (row[1],)).fetchone()
                    if row[0] != "retry" or stored is None or stored[0] != job_id:
                        raise RepositoryError(ReasonCode.CONFLICT)
                    return self._attempt(stored[1]), True
            job_row = db.execute("SELECT state, doc, version FROM jobs WHERE job_id=?",
                                 (job_id,)).fetchone()
            if job_row is None or job_row[2] != expected_version:
                raise RepositoryError(ReasonCode.CONFLICT)
            for (doc,) in db.execute("SELECT doc FROM attempts WHERE job_id=?", (job_id,)):
                if self._attempt(doc).state not in _ATTEMPT_DONE:
                    raise RepositoryError(ReasonCode.CONFLICT)
            job = replace(self._job(job_row[:2]), attempt_id=attempt.attempt_id, updated_at=self._now())
            db.execute("INSERT INTO attempts (attempt_id, job_id, ordinal, retry_key, doc)"
                       " VALUES (?,?,?,?,?)", (attempt.attempt_id, job_id, attempt.ordinal, key,
                                               _dump(attempt.to_dict())))
            if key is not None:
                db.execute("INSERT INTO idempotency_keys (key, kind, request_hash, target_id)"
                           " VALUES (?,?,?,?)", (key, "retry", job.request_hash, attempt.attempt_id))
            self._update_job(db, job, expected_version)
        return attempt, False

    def admit_attempt(self, job: Job, attempt: Attempt, *, retry_key: str | None,
                      expected_version: int,
                      acquire: tuple[str, AdmissionPlan] | None) -> tuple[Job, Attempt, bool]:
        """Admit a job, its attempt and the lease in one transaction.

        Returns ``(job, attempt, replayed)``.  A retry key already bound to an attempt of the
        same job replays the stored job and attempt with no mutation; a key bound elsewhere
        conflicts.  A held lease, a stale version or a second active attempt rolls back
        everything with CONFLICT.
        """
        key = None if retry_key is None else _key(retry_key)
        if (type(job) is not Job or type(attempt) is not Attempt or attempt.job_id != job.job_id
                or type(expected_version) is not int
                or (acquire is not None and (type(acquire) is not tuple or len(acquire) != 2))):
            raise RepositoryError(ReasonCode.INVALID_INPUT)
        with self._tx("admit_attempt") as db:
            if self._latch is not None:
                raise RepositoryError(self._latch)
            if key is not None:
                row = db.execute("SELECT kind, target_id FROM idempotency_keys WHERE key=?",
                                 (key,)).fetchone()
                if row is not None:
                    stored = db.execute("SELECT job_id, doc FROM attempts WHERE attempt_id=?",
                                        (row[1],)).fetchone()
                    if row[0] != "retry" or stored is None or stored[0] != job.job_id:
                        raise RepositoryError(ReasonCode.CONFLICT)
                    job_row = db.execute("SELECT state, doc FROM jobs WHERE job_id=?",
                                         (job.job_id,)).fetchone()
                    if job_row is None:
                        raise RepositoryError(ReasonCode.CONFLICT)
                    return self._job(job_row), self._attempt(stored[1]), True
            for (doc,) in db.execute("SELECT doc FROM attempts WHERE job_id=?", (job.job_id,)):
                if self._attempt(doc).state not in _ATTEMPT_DONE:
                    raise RepositoryError(ReasonCode.CONFLICT)
            self._update_job(db, job, expected_version)
            db.execute("INSERT INTO attempts (attempt_id, job_id, ordinal, retry_key, doc)"
                       " VALUES (?,?,?,?,?)", (attempt.attempt_id, job.job_id, attempt.ordinal, key,
                                               _dump(attempt.to_dict())))
            if key is not None:
                db.execute("INSERT INTO idempotency_keys (key, kind, request_hash, target_id)"
                           " VALUES (?,?,?,?)", (key, "retry", job.request_hash, attempt.attempt_id))
            if acquire is not None and not self._acquire(db, acquire[0], acquire[1], job.job_id):
                raise RepositoryError(ReasonCode.CONFLICT)
        return job, attempt, False

    def get_retry_attempt(self, retry_key: str) -> tuple[str, Attempt] | None:
        """Read-only lookup of ``(job_id, attempt)`` bound to a retry key."""
        key = _key(retry_key)
        with self._read() as db:
            row = db.execute(
                "SELECT a.job_id, a.doc FROM idempotency_keys k JOIN attempts a"
                " ON a.attempt_id = k.target_id WHERE k.key=? AND k.kind='retry'", (key,)).fetchone()
            return None if row is None else (row[0], self._attempt(row[1]))

    def update_attempt(self, attempt: Attempt, *, closing: bool = False) -> Attempt:
        if type(attempt) is not Attempt:
            raise RepositoryError(ReasonCode.INVALID_INPUT)
        if closing:
            self._require_closing_attempt(attempt)
        with self._tx("update_attempt", check_bound=not closing) as db:
            cursor = db.execute("UPDATE attempts SET doc=? WHERE attempt_id=? AND job_id=?",
                                (_dump(attempt.to_dict()), attempt.attempt_id, attempt.job_id))
            if cursor.rowcount != 1:
                raise RepositoryError(ReasonCode.CONFLICT)
        return attempt

    def record_stage(self, stage: StageRecord) -> None:
        if type(stage) is not StageRecord:
            raise RepositoryError(ReasonCode.INVALID_INPUT)
        with self._tx("record_stage") as db:
            # A duplicate stage_id or unknown attempt violates a constraint -> CONFLICT.
            db.execute("INSERT INTO stages (stage_id, attempt_id, doc) VALUES (?,?,?)",
                       (stage.stage_id, stage.attempt_id, _dump(stage.to_dict())))

    def record_artifact(self, artifact: Artifact) -> None:
        if type(artifact) is not Artifact:
            raise RepositoryError(ReasonCode.INVALID_INPUT)
        with self._tx("record_artifact") as db:
            row = db.execute("SELECT job_id FROM attempts WHERE attempt_id=?",
                             (artifact.attempt_id,)).fetchone()
            if row is None:
                raise RepositoryError(ReasonCode.CONFLICT)
            db.execute("INSERT INTO artifacts (artifact_id, job_id, doc) VALUES (?,?,?)",
                       (artifact.artifact_id, row[0], _dump(artifact.to_dict())))

    # --------------------------------------------------------------------- lease

    def _acquire(self, db: sqlite3.Connection, operation_id: str, plan: AdmissionPlan,
                 job_id: str | None) -> bool:
        if type(operation_id) is not str or not operation_id:
            raise RepositoryError(ReasonCode.INVALID_INPUT)
        doc = _plan_doc(plan)
        if db.execute("SELECT COUNT(*) FROM reservations").fetchone()[0] >= self._limits.active_operations:
            return False
        db.execute("INSERT INTO reservations (operation_id, job_id, plan, acquired_at, writers_unreaped)"
                   " VALUES (?,?,?,?,1)", (operation_id, job_id, doc, self._now()))
        return True

    def try_acquire(self, operation_id: str, plan: AdmissionPlan) -> bool:
        with self._tx("try_acquire") as db:
            if self._latch is not None:
                raise RepositoryError(self._latch)
            return self._acquire(db, operation_id, plan, None)

    @staticmethod
    def _release(db: sqlite3.Connection, operation_id: str, writers_reaped: bool) -> None:
        if type(operation_id) is not str or not operation_id or type(writers_reaped) is not bool:
            raise RepositoryError(ReasonCode.INVALID_INPUT)
        if not writers_reaped:
            raise RepositoryError(ReasonCode.WRITER_UNRESOLVED)
        db.execute("DELETE FROM reservations WHERE operation_id=?", (operation_id,))

    def release(self, operation_id: str, *, writers_reaped: bool) -> None:
        # Releasing shrinks state, so it is not refused by the size bound.
        with self._tx("release", check_bound=False) as db:
            self._release(db, operation_id, writers_reaped)

    def transition_with_lease(self, job: Job, *, expected_version: int,
                              acquire: tuple[str, AdmissionPlan] | None = None,
                              release: tuple[str, bool] | None = None,
                              closing: bool = False) -> Job:
        """Change job state and the lease together; either both happen or neither."""
        if closing:
            self._require_closing_job(job)
            if acquire is not None:
                raise RepositoryError(ReasonCode.INVALID_INPUT)
        with self._tx("transition_with_lease", check_bound=not closing) as db:
            if acquire is not None and self._latch is not None:
                raise RepositoryError(self._latch)
            self._update_job(db, job, expected_version)
            if release is not None:
                self._release(db, *release)
            if acquire is not None and not self._acquire(db, acquire[0], acquire[1], job.job_id):
                raise RepositoryError(ReasonCode.CONFLICT)
        return job

    def close_attempt_and_release(self, job: Job, attempt: Attempt, *, expected_version: int,
                                  release: tuple[str, bool] | None) -> Job:
        """Close the job, finish its attempt and release the lease in one transaction.

        Closing writes bypass the size bound and the admission latch so a verified reap can
        always be recorded.
        """
        self._require_closing_job(job)
        self._require_closing_attempt(attempt)
        if attempt.job_id != job.job_id:
            raise RepositoryError(ReasonCode.INVALID_INPUT)
        with self._tx("close_attempt_and_release", check_bound=False) as db:
            self._update_job(db, job, expected_version)
            cursor = db.execute("UPDATE attempts SET doc=? WHERE attempt_id=? AND job_id=?",
                                (_dump(attempt.to_dict()), attempt.attempt_id, attempt.job_id))
            if cursor.rowcount != 1:
                raise RepositoryError(ReasonCode.CONFLICT)
            if release is not None:
                self._release(db, *release)
        return job

    def lease_holders(self) -> tuple[tuple[str, bool], ...]:
        """Operation IDs holding the lease and whether writers remain unreaped."""
        with self._read() as db:
            return tuple((r[0], bool(r[1])) for r in db.execute(
                "SELECT operation_id, writers_unreaped FROM reservations ORDER BY acquired_at"))

    # --------------------------------------------------------------------- reads

    def get_job(self, job_id: str) -> Job | None:
        validate_uuid_string(job_id)
        with self._read() as db:
            row = db.execute("SELECT state, doc FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            return None if row is None else self._job(row)

    def get_job_version(self, job_id: str) -> int | None:
        validate_uuid_string(job_id)
        with self._read() as db:
            row = db.execute("SELECT version FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            return None if row is None else row[0]

    def get_attempt(self, attempt_id: str) -> Attempt | None:
        validate_uuid_string(attempt_id)
        with self._read() as db:
            row = db.execute("SELECT doc FROM attempts WHERE attempt_id=?", (attempt_id,)).fetchone()
            return None if row is None else self._attempt(row[0])

    def list_stages(self, attempt_id: str) -> list[StageRecord]:
        validate_uuid_string(attempt_id)
        with self._read() as db:
            # Bounded by the finite stage set per attempt; rowid keeps insertion order.
            rows = db.execute("SELECT doc FROM stages WHERE attempt_id=? ORDER BY rowid LIMIT 1000",
                              (attempt_id,)).fetchall()
            return [StageRecord.from_dict(json.loads(doc)) for (doc,) in rows]

    def list_artifacts(self, job_id: str) -> list[Artifact]:
        validate_uuid_string(job_id)
        with self._read() as db:
            rows = db.execute("SELECT doc FROM artifacts WHERE job_id=? ORDER BY rowid LIMIT 1000",
                              (job_id,)).fetchall()
            return [Artifact.from_dict(json.loads(doc)) for (doc,) in rows]

    def list_jobs(self, limit: int = PAGE_MAX, cursor: str | None = None) -> tuple[list[Job], str | None]:
        if type(limit) is not int or not 1 <= limit <= PAGE_MAX:
            raise RepositoryError(ReasonCode.INVALID_INPUT)
        after = None if cursor is None else _decode_cursor(cursor)
        with self._read() as db:
            if after is None:
                rows = db.execute("SELECT state, doc, created_at, job_id FROM jobs"
                                  " ORDER BY created_at DESC, job_id DESC LIMIT ?", (limit + 1,))
            else:
                rows = db.execute("SELECT state, doc, created_at, job_id FROM jobs"
                                  " WHERE (created_at, job_id) < (?, ?)"
                                  " ORDER BY created_at DESC, job_id DESC LIMIT ?",
                                  (*after, limit + 1))
            rows = rows.fetchall()
            page = rows[:limit]
            jobs = [self._job(row[:2]) for row in page]
            more = _encode_cursor(page[-1][2], page[-1][3]) if len(rows) > limit else None
            return jobs, more

    def queued_jobs(self, limit: int = PAGE_MAX) -> list[Job]:
        """Queued jobs oldest-first by (created_at, job_id), at most ``limit`` (1..100)."""
        if type(limit) is not int or not 1 <= limit <= PAGE_MAX:
            raise RepositoryError(ReasonCode.INVALID_INPUT)
        with self._read() as db:
            rows = db.execute("SELECT state, doc FROM jobs WHERE state=?"
                              " ORDER BY created_at, job_id LIMIT ?",
                              (JobState.QUEUED.value, limit)).fetchall()
            return [self._job(row) for row in rows]

    def active_jobs(self, limit: int = PAGE_MAX) -> list[Job]:
        """Jobs in checking..exporting, oldest-first by (created_at, job_id), at most ``limit``."""
        if type(limit) is not int or not 1 <= limit <= PAGE_MAX:
            raise RepositoryError(ReasonCode.INVALID_INPUT)
        marks = ",".join("?" * len(_ACTIVE_VALUES))
        with self._read() as db:
            rows = db.execute(f"SELECT state, doc FROM jobs WHERE state IN ({marks})"
                              " ORDER BY created_at, job_id LIMIT ?",
                              (*_ACTIVE_VALUES, limit)).fetchall()
            return [self._job(row) for row in rows]

    # --------------------------------------------------------------- maintenance

    def maintain(self, now: datetime) -> MaintenanceResult:
        """Idle cleanup: retention, incremental vacuum, WAL truncate; never full VACUUM."""
        cutoff = _stamp(now - timedelta(days=RETENTION_DAYS))
        before = self.db_bytes()
        deleted = 0
        # Maintenance may use up to the bound plus its allowance before deletes are refused.
        if before <= DB_BOUND + MAINTENANCE_ALLOWANCE:
            with self._tx("maintain", check_bound=False) as db:
                deleted = self._retain(db, cutoff)
        with self._lock:
            self._require_open()
            try:
                self._conn.execute("PRAGMA incremental_vacuum").fetchall()
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
                after = self.db_bytes()
            except (sqlite3.Error, OSError):
                self._latch = ReasonCode.STATE_WRITE_FAILED
                raise RepositoryError(ReasonCode.STATE_WRITE_FAILED) from None
            self._latch = ReasonCode.STORAGE_INSUFFICIENT if after > DB_BOUND else None
            return MaintenanceResult(deleted, before, after, self._latch)

    @staticmethod
    def _retain(db: sqlite3.Connection, cutoff: str) -> int:
        marks = ",".join("?" * len(_FINISHED_VALUES))
        db.execute("CREATE TEMP TABLE IF NOT EXISTS doomed (job_id TEXT PRIMARY KEY)")
        db.execute("DELETE FROM doomed")
        db.execute(
            f"INSERT INTO doomed SELECT job_id FROM jobs WHERE state IN ({marks})"
            " AND finished_at IS NOT NULL"
            " AND job_id NOT IN (SELECT job_id FROM reservations WHERE job_id IS NOT NULL)"
            " AND (finished_at < ? OR job_id IN (SELECT job_id FROM jobs"
            f"      WHERE state IN ({marks}) AND finished_at IS NOT NULL"
            "       ORDER BY finished_at DESC, job_id DESC LIMIT -1 OFFSET ?))",
            (*_FINISHED_VALUES, cutoff, *_FINISHED_VALUES, RETENTION_JOBS))
        count = db.execute("SELECT COUNT(*) FROM doomed").fetchone()[0]
        if count:
            db.execute("DELETE FROM stages WHERE attempt_id IN (SELECT attempt_id FROM attempts"
                       " WHERE job_id IN (SELECT job_id FROM doomed))")
            db.execute("DELETE FROM artifacts WHERE job_id IN (SELECT job_id FROM doomed)")
            db.execute("DELETE FROM idempotency_keys WHERE (kind='submit' AND target_id IN"
                       " (SELECT job_id FROM doomed)) OR (kind='retry' AND target_id IN"
                       " (SELECT attempt_id FROM attempts WHERE job_id IN (SELECT job_id FROM doomed)))")
            db.execute("DELETE FROM attempts WHERE job_id IN (SELECT job_id FROM doomed)")
            db.execute("DELETE FROM jobs WHERE job_id IN (SELECT job_id FROM doomed)")
        db.execute("DROP TABLE doomed")
        return count


__all__ = (
    "DB_BOUND",
    "MAINTENANCE_ALLOWANCE",
    "RETENTION_DAYS",
    "RETENTION_JOBS",
    "MaintenanceResult",
    "Repository",
    "RepositoryError",
)
