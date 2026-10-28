"""Single-writer SQLite repository: schema, atomic writes, paging, retention and bounds."""
import json
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from podmuncher import repository as repo_module
from podmuncher.admission import AdmissionPlan
from podmuncher.domains import (
    ArtifactFormat,
    ArtifactSlot,
    DomainValidationError,
    FieldStatus,
    JobState,
    ReasonCode,
    Stage,
    StageOutcome,
)
from podmuncher.repository import MaintenanceResult, Repository, RepositoryError
from podmuncher.requests import JobRequest
from podmuncher.snapshots import Artifact, Attempt, Job, Measurement, StageRecord

SHA = "a" * 64
NOW = datetime(2026, 10, 12, 12, 0, tzinfo=UTC)
ROOT = Path(__file__).resolve().parent.parent


class InjectedClock:
    """Advanceable UTC clock."""

    def __init__(self, start=NOW):
        self.now = start

    def __call__(self):
        return self.now

    def set(self, moment):
        self.now = moment

    def advance(self, delta):
        self.now += delta


class Fault:
    """fault_hook that fails once armed, before COMMIT."""

    def __init__(self):
        self.error = None
        self.ops = []

    def arm(self, error=None, *, op=None):
        self.error, self.only = error or sqlite3.OperationalError("disk I/O error"), op

    def __call__(self, op):
        self.ops.append(op)
        if self.error is not None and self.only in (None, op):
            error, self.error = self.error, None
            raise error


def ident(n, tag=1):
    return f"{n:08x}-{tag:04x}-4111-8111-111111111111"


def stamp(moment):
    return moment.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def make_request(root="./out"):
    return JobRequest.from_dict({
        "schema_version": 1, "source": {"kind": "local", "input": "./input.mp4"},
        "outputs": {"transcript": True, "video": False, "video_quality": "1080p"},
        "selection": {"device": "auto", "profile": "balanced", "model": "auto", "language": "auto"},
        "output_root": root,
        "limits": {"media_bytes": 100, "duration_seconds": 20, "export_bytes": 10}})


def make_job(n, *, created=None, root="./out", state=JobState.QUEUED):
    request = make_request(root)
    when = stamp(created or NOW + timedelta(seconds=n))
    return Job(ident(n), request, request.request_hash, state, ReasonCode.OK,
               None, None, when, when, (), None)


def make_attempt(job, n, *, state=JobState.CHECKING, ordinal=1):
    return Attempt(ident(n, 2), job.job_id, ordinal, None, None, stamp(NOW), None, state,
                   ReasonCode.OK, (), 0)


def make_stage(attempt, n):
    return StageRecord(ident(n, 3), attempt.attempt_id, Stage.PREPARING, StageOutcome.OK,
                       stamp(NOW), None, (), SHA, ())


def make_artifact(attempt, n, path=None):
    return Artifact(ident(n, 4), ArtifactSlot.TRANSCRIPT_TXT, ArtifactFormat.TXT, 5, SHA, True,
                    attempt.attempt_id, None, path)


def make_plan():
    return AdmissionPlan(Measurement(True, "test", FieldStatus.OK, stamp(NOW), None), (), ())


def make_repository(tmp_path, *, clock=None, state_volume=None, fault_hook=None, name="state.db"):
    return Repository.open(tmp_path / name, clock=clock or InjectedClock(),
                           state_volume=state_volume or (lambda path: True), fault_hook=fault_hook)


def count(path, table):
    conn = sqlite3.connect(path)
    try:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    finally:
        conn.close()


def raw_exec(path, sql, *params):
    conn = sqlite3.connect(path)
    try:
        conn.execute(sql, params)
        conn.commit()
    finally:
        conn.close()


def finish(repo, clock, job, when, state=JobState.COMPLETED):
    clock.set(when)
    version = repo.get_job_version(job.job_id)
    done = replace(job, state=state, updated_at=stamp(when))
    repo.update_job(done, expected_version=version)
    return done


def reason(excinfo):
    return excinfo.value.reason_code


# ---------------------------------------------------------------------- open

@pytest.mark.parametrize("answer,expected", [(False, ReasonCode.INVALID_CONFIG),
                                             (None, ReasonCode.STORAGE_UNKNOWN)])
def test_open_refuses_unsafe_state_volume_before_creating_files(tmp_path, answer, expected):
    with pytest.raises(RepositoryError) as caught:
        make_repository(tmp_path, state_volume=lambda path: answer)
    assert reason(caught) is expected
    assert list(tmp_path.iterdir()) == []


def test_open_refuses_failing_probe(tmp_path):
    def boom(path):
        raise OSError("secret path")
    with pytest.raises(RepositoryError) as caught:
        make_repository(tmp_path, state_volume=boom)
    assert reason(caught) is ReasonCode.STORAGE_UNKNOWN
    assert "secret" not in str(caught.value)
    assert list(tmp_path.iterdir()) == []


def test_open_migrates_empty_database_and_sets_pragmas(tmp_path):
    with make_repository(tmp_path) as repo:
        db = repo._conn
        assert db.execute("PRAGMA user_version").fetchone()[0] == len(repo_module._MIGRATIONS)
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert db.execute("PRAGMA synchronous").fetchone()[0] == 2
        assert db.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert db.execute("PRAGMA auto_vacuum").fetchone()[0] == 2
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"jobs", "attempts", "stages", "artifacts", "idempotency_keys",
                "reservations"} <= tables
        assert repo.admission_blocked is None
        assert repo.list_jobs() == ([], None)


def test_reopen_preserves_data(tmp_path):
    clock = InjectedClock()
    job = make_job(1)
    with make_repository(tmp_path, clock=clock) as repo:
        repo.submit(job, "key-1")
    with make_repository(tmp_path, clock=clock) as repo:
        assert repo.get_job(job.job_id) == job
        stored, replayed = repo.submit(job, "key-1")
        assert replayed and stored == job


def test_newer_schema_is_refused(tmp_path):
    path = tmp_path / "state.db"
    raw_exec(path, "PRAGMA user_version=99")
    with pytest.raises(RepositoryError) as caught:
        make_repository(tmp_path)
    assert reason(caught) is ReasonCode.UNSUPPORTED_SCHEMA


def test_closed_repository_refuses_use(tmp_path):
    repo = make_repository(tmp_path)
    repo.close()
    repo.close()
    with pytest.raises(RepositoryError) as caught:
        repo.get_job(ident(1))
    assert reason(caught) is ReasonCode.SERVICE_STOPPED


def test_repository_exposes_no_sql_entry_point(tmp_path):
    with make_repository(tmp_path) as repo:
        public = {n for n in dir(repo) if not n.startswith("_")}
        assert not {"execute", "executescript", "connection", "conn", "cursor", "query"} & public


# -------------------------------------------------------------------- submit

def test_submit_replay_returns_same_job(tmp_path):
    with make_repository(tmp_path) as repo:
        job = make_job(1)
        assert repo.submit(job, "key-1") == (job, False)
        again, replayed = repo.submit(job, "key-1")
        assert replayed is True and again == job
        assert count(tmp_path / "state.db", "jobs") == 1


def test_submit_same_key_different_hash_conflicts(tmp_path):
    with make_repository(tmp_path) as repo:
        repo.submit(make_job(1), "key-1")
        with pytest.raises(RepositoryError) as caught:
            repo.submit(make_job(2, root="./other"), "key-1")
        assert reason(caught) is ReasonCode.CONFLICT
        assert repo.get_job(ident(2)) is None


def test_submit_rejects_bad_key_and_non_queued(tmp_path):
    with make_repository(tmp_path) as repo:
        for key in ("", None, "x" * 201):
            with pytest.raises(RepositoryError) as caught:
                repo.submit(make_job(1), key)
            assert reason(caught) is ReasonCode.INVALID_INPUT
        with pytest.raises(RepositoryError) as caught:
            repo.submit(make_job(1, state=JobState.COMPLETED), "key-1")
        assert reason(caught) is ReasonCode.INVALID_INPUT


def test_submit_queue_full_on_101st(tmp_path):
    with make_repository(tmp_path) as repo:
        for n in range(1, 101):
            repo.submit(make_job(n), f"key-{n}")
        with pytest.raises(RepositoryError) as caught:
            repo.submit(make_job(101), "key-101")
        assert reason(caught) is ReasonCode.QUEUE_FULL
        assert count(tmp_path / "state.db", "jobs") == 100
        assert repo.admission_blocked is None
        # Replay of an admitted key still works at capacity.
        assert repo.submit(make_job(5), "key-5")[1] is True


# ---------------------------------------------------------------- update_job

def test_update_job_compare_and_set(tmp_path):
    with make_repository(tmp_path) as repo:
        job = make_job(1)
        repo.submit(job, "key-1")
        assert repo.get_job_version(job.job_id) == 1
        newer = replace(job, state=JobState.CHECKING, updated_at=stamp(NOW + timedelta(seconds=5)))
        repo.update_job(newer, expected_version=1)
        assert repo.get_job_version(job.job_id) == 2
        late = replace(job, state=JobState.FAILED, updated_at=stamp(NOW + timedelta(seconds=9)))
        with pytest.raises(RepositoryError) as caught:
            repo.update_job(late, expected_version=1)
        assert reason(caught) is ReasonCode.CONFLICT
        assert repo.get_job(job.job_id) == newer
        assert repo.get_job_version(job.job_id) == 2
        assert repo.admission_blocked is None


def test_update_job_unknown_job_conflicts(tmp_path):
    with make_repository(tmp_path) as repo:
        with pytest.raises(RepositoryError) as caught:
            repo.update_job(make_job(1), expected_version=1)
        assert reason(caught) is ReasonCode.CONFLICT


# ------------------------------------------------------------------ attempts

def test_start_attempt_replays_same_retry_key(tmp_path):
    with make_repository(tmp_path) as repo:
        job = make_job(1)
        repo.submit(job, "key-1")
        attempt = make_attempt(job, 1)
        assert repo.start_attempt(job.job_id, attempt, retry_key="retry-1", expected_version=1) == (attempt, False)
        stored_job = repo.get_job(job.job_id)
        assert stored_job.attempt_id == attempt.attempt_id
        assert repo.get_job_version(job.job_id) == 2
        again, replayed = repo.start_attempt(job.job_id, make_attempt(job, 9), retry_key="retry-1",
                                             expected_version=2)
        assert replayed is True and again == attempt
        assert count(tmp_path / "state.db", "attempts") == 1
        assert repo.get_attempt(attempt.attempt_id) == attempt


def test_start_attempt_second_active_attempt_conflicts(tmp_path):
    with make_repository(tmp_path) as repo:
        job = make_job(1)
        repo.submit(job, "key-1")
        repo.start_attempt(job.job_id, make_attempt(job, 1), retry_key=None, expected_version=1)
        with pytest.raises(RepositoryError) as caught:
            repo.start_attempt(job.job_id, make_attempt(job, 2, ordinal=2), retry_key="r2",
                               expected_version=2)
        assert reason(caught) is ReasonCode.CONFLICT
        assert count(tmp_path / "state.db", "attempts") == 1
        assert count(tmp_path / "state.db", "idempotency_keys") == 1
        # Once the first attempt is terminal a new one may start.
        first = make_attempt(job, 1, state=JobState.FAILED)
        repo.update_attempt(first)
        second = make_attempt(job, 2, ordinal=2)
        assert repo.start_attempt(job.job_id, second, retry_key="r2", expected_version=2) == (second, False)


def test_start_attempt_stale_version_conflicts(tmp_path):
    with make_repository(tmp_path) as repo:
        job = make_job(1)
        repo.submit(job, "key-1")
        with pytest.raises(RepositoryError) as caught:
            repo.start_attempt(job.job_id, make_attempt(job, 1), retry_key=None, expected_version=7)
        assert reason(caught) is ReasonCode.CONFLICT
        assert repo.get_job(job.job_id).attempt_id is None


def test_retry_key_reuse_across_jobs_conflicts(tmp_path):
    with make_repository(tmp_path) as repo:
        one, two = make_job(1), make_job(2)
        repo.submit(one, "k1")
        repo.submit(two, "k2")
        repo.start_attempt(one.job_id, make_attempt(one, 1), retry_key="shared", expected_version=1)
        with pytest.raises(RepositoryError) as caught:
            repo.start_attempt(two.job_id, make_attempt(two, 2), retry_key="shared", expected_version=1)
        assert reason(caught) is ReasonCode.CONFLICT


def test_stages_keep_distinct_ids_and_artifacts_record(tmp_path):
    with make_repository(tmp_path) as repo:
        job = make_job(1)
        repo.submit(job, "key-1")
        attempt = make_attempt(job, 1)
        repo.start_attempt(job.job_id, attempt, retry_key=None, expected_version=1)
        first, second = make_stage(attempt, 1), make_stage(attempt, 2)
        repo.record_stage(first)
        repo.record_stage(second)  # repeated preparing stage, distinct id
        assert repo.list_stages(attempt.attempt_id) == [first, second]
        with pytest.raises(RepositoryError) as caught:
            repo.record_stage(first)
        assert reason(caught) is ReasonCode.CONFLICT
        orphan = replace(first, stage_id=ident(9, 3), attempt_id=ident(77, 2))
        with pytest.raises(RepositoryError) as caught:
            repo.record_stage(orphan)
        assert reason(caught) is ReasonCode.CONFLICT
        artifact = make_artifact(attempt, 1)
        repo.record_artifact(artifact)
        assert repo.list_artifacts(job.job_id) == [artifact]
        assert repo.admission_blocked is None


# -------------------------------------------------------------- fault_hook

@pytest.mark.parametrize("error", [sqlite3.OperationalError("disk I/O error"), OSError(28, "no space")])
def test_fault_rolls_back_latches_and_maintain_clears(tmp_path, error):
    fault = Fault()
    with make_repository(tmp_path, fault_hook=fault) as repo:
        job = make_job(1)
        fault.arm(error, op="submit")
        with pytest.raises(RepositoryError) as caught:
            repo.submit(job, "key-1")
        assert reason(caught) is ReasonCode.STATE_WRITE_FAILED
        assert "disk" not in str(caught.value) and "space" not in str(caught.value)
        assert repo.admission_blocked is ReasonCode.STATE_WRITE_FAILED
        assert count(tmp_path / "state.db", "jobs") == 0
        assert count(tmp_path / "state.db", "idempotency_keys") == 0
        assert not repo._conn.in_transaction
        with pytest.raises(RepositoryError) as caught:
            repo.submit(job, "key-1")
        assert reason(caught) is ReasonCode.STATE_WRITE_FAILED
        result = repo.maintain(NOW)
        assert isinstance(result, MaintenanceResult) and result.blocked is None
        assert repo.admission_blocked is None
        assert repo.submit(job, "key-1") == (job, False)


def test_fault_in_start_attempt_leaves_no_partial_rows(tmp_path):
    fault = Fault()
    with make_repository(tmp_path, fault_hook=fault) as repo:
        job = make_job(1)
        repo.submit(job, "key-1")
        fault.arm(op="start_attempt")
        with pytest.raises(RepositoryError) as caught:
            repo.start_attempt(job.job_id, make_attempt(job, 1), retry_key="r1", expected_version=1)
        assert reason(caught) is ReasonCode.STATE_WRITE_FAILED
        db = tmp_path / "state.db"
        assert count(db, "attempts") == 0 and count(db, "idempotency_keys") == 1
        assert repo.get_job(job.job_id) == job and repo.get_job_version(job.job_id) == 1


def test_fault_in_record_stage_leaves_no_row(tmp_path):
    fault = Fault()
    with make_repository(tmp_path, fault_hook=fault) as repo:
        job = make_job(1)
        repo.submit(job, "key-1")
        attempt = make_attempt(job, 1)
        repo.start_attempt(job.job_id, attempt, retry_key=None, expected_version=1)
        fault.arm(op="record_stage")
        with pytest.raises(RepositoryError):
            repo.record_stage(make_stage(attempt, 1))
        assert count(tmp_path / "state.db", "stages") == 0


# --------------------------------------------------------------------- lease

def test_lease_second_acquire_false_and_release_unresolved(tmp_path):
    with make_repository(tmp_path) as repo:
        plan = make_plan()
        assert repo.try_acquire("op-1", plan) is True
        assert repo.try_acquire("op-2", plan) is False
        assert repo.try_acquire("op-1", plan) is False
        with pytest.raises(RepositoryError) as caught:
            repo.release("op-1", writers_reaped=False)
        assert reason(caught) is ReasonCode.WRITER_UNRESOLVED
        assert repo.lease_holders() == (("op-1", True),)
        assert count(tmp_path / "state.db", "reservations") == 1
        repo.release("op-1", writers_reaped=True)
        assert repo.lease_holders() == ()
        assert repo.try_acquire("op-2", plan) is True


def test_transition_with_lease_acquire_and_release(tmp_path):
    with make_repository(tmp_path) as repo:
        job = make_job(1)
        repo.submit(job, "key-1")
        checking = replace(job, state=JobState.CHECKING, updated_at=stamp(NOW + timedelta(seconds=1)))
        repo.transition_with_lease(checking, expected_version=1, acquire=("op-1", make_plan()))
        assert repo.get_job(job.job_id).state is JobState.CHECKING
        assert repo.lease_holders() == (("op-1", True),)
        failed = replace(checking, state=JobState.FAILED, updated_at=stamp(NOW + timedelta(seconds=2)))
        with pytest.raises(RepositoryError) as caught:
            repo.transition_with_lease(failed, expected_version=2, release=("op-1", False))
        assert reason(caught) is ReasonCode.WRITER_UNRESOLVED
        assert repo.get_job(job.job_id).state is JobState.CHECKING
        assert repo.lease_holders() == (("op-1", True),)
        repo.transition_with_lease(failed, expected_version=2, release=("op-1", True))
        assert repo.get_job(job.job_id).state is JobState.FAILED
        assert repo.lease_holders() == ()


def test_transition_with_lease_is_atomic_when_lease_held(tmp_path):
    with make_repository(tmp_path) as repo:
        job = make_job(1)
        repo.submit(job, "key-1")
        repo.try_acquire("other", make_plan())
        checking = replace(job, state=JobState.CHECKING, updated_at=stamp(NOW + timedelta(seconds=1)))
        with pytest.raises(RepositoryError) as caught:
            repo.transition_with_lease(checking, expected_version=1, acquire=("op-1", make_plan()))
        assert reason(caught) is ReasonCode.CONFLICT
        assert repo.get_job(job.job_id) == job and repo.get_job_version(job.job_id) == 1
        assert repo.lease_holders() == (("other", True),)


def test_transition_with_lease_is_atomic_under_injected_failure(tmp_path):
    fault = Fault()
    with make_repository(tmp_path, fault_hook=fault) as repo:
        job = make_job(1)
        repo.submit(job, "key-1")
        checking = replace(job, state=JobState.CHECKING, updated_at=stamp(NOW + timedelta(seconds=1)))
        fault.arm(op="transition_with_lease")
        with pytest.raises(RepositoryError) as caught:
            repo.transition_with_lease(checking, expected_version=1, acquire=("op-1", make_plan()))
        assert reason(caught) is ReasonCode.STATE_WRITE_FAILED
        assert repo.get_job(job.job_id) == job and repo.get_job_version(job.job_id) == 1
        assert repo.lease_holders() == ()
        assert repo.admission_blocked is ReasonCode.STATE_WRITE_FAILED


# -------------------------------------------------------------------- paging

@pytest.mark.parametrize("limit", [0, 101, -1, True, "7", None])
def test_list_jobs_limit_bounds(tmp_path, limit):
    with make_repository(tmp_path) as repo:
        with pytest.raises(RepositoryError) as caught:
            repo.list_jobs(limit=limit)
        assert reason(caught) is ReasonCode.INVALID_INPUT


@pytest.mark.parametrize("cursor", ["", "!!!", "e30", "bm90LWpzb24", 7,
                                    "eyJ2IjoyLCJjIjoieCIsImoiOiJ5In0"])
def test_list_jobs_malformed_cursor(tmp_path, cursor):
    with make_repository(tmp_path) as repo:
        with pytest.raises(RepositoryError) as caught:
            repo.list_jobs(limit=5, cursor=cursor)
        assert reason(caught) is ReasonCode.INVALID_INPUT


def test_list_jobs_walks_every_job_once_in_order(tmp_path):
    with make_repository(tmp_path) as repo:
        jobs = [make_job(n) for n in range(1, 31)]
        for job in jobs:
            repo.submit(job, f"key-{job.job_id}")
        seen, cursor, pages = [], None, 0
        while True:
            page, cursor = repo.list_jobs(limit=7, cursor=cursor)
            assert 1 <= len(page) <= 7
            seen.extend(j.job_id for j in page)
            pages += 1
            if cursor is None:
                break
        assert pages == 5
        assert seen == [j.job_id for j in reversed(jobs)]
        assert len(set(seen)) == 30
        page, cursor = repo.list_jobs(limit=30)
        assert len(page) == 30 and cursor is None


def test_list_jobs_ties_on_created_at_break_by_job_id(tmp_path):
    with make_repository(tmp_path) as repo:
        same = NOW
        jobs = [make_job(n, created=same) for n in range(1, 8)]
        for job in jobs:
            repo.submit(job, f"key-{job.job_id}")
        seen, cursor = [], None
        while True:
            page, cursor = repo.list_jobs(limit=2, cursor=cursor)
            seen.extend(j.job_id for j in page)
            if cursor is None:
                break
        assert seen == sorted((j.job_id for j in jobs), reverse=True)


# ----------------------------------------------------------------- retention

def test_retention_by_age_ninety_days(tmp_path):
    clock = InjectedClock()
    with make_repository(tmp_path, clock=clock) as repo:
        old, recent = make_job(1), make_job(2)
        repo.submit(old, "k1")
        repo.submit(recent, "k2")
        for job, n in ((old, 1), (recent, 2)):
            attempt = make_attempt(job, n)
            repo.start_attempt(job.job_id, attempt, retry_key=f"r{n}", expected_version=1)
            repo.record_stage(make_stage(attempt, n))
        finish(repo, clock, repo.get_job(old.job_id), NOW - timedelta(days=91))
        finish(repo, clock, repo.get_job(recent.job_id), NOW - timedelta(days=89))
        clock.set(NOW)
        result = repo.maintain(NOW)
        assert result.deleted == 1
        assert repo.get_job(old.job_id) is None
        assert repo.get_job(recent.job_id) is not None
        db = tmp_path / "state.db"
        assert (count(db, "jobs"), count(db, "attempts"), count(db, "stages")) == (1, 1, 1)
        assert count(db, "idempotency_keys") == 2  # recent job's submit and retry keys
        replayed = repo.submit(make_job(1), "k1")[1]
        assert replayed is False  # key removed with its job


def test_retention_by_count_removes_oldest_finished(tmp_path, monkeypatch):
    monkeypatch.setattr(repo_module, "RETENTION_JOBS", 2)
    clock = InjectedClock()
    with make_repository(tmp_path, clock=clock) as repo:
        jobs = [make_job(n) for n in range(1, 5)]
        for job in jobs:
            repo.submit(job, f"k{job.job_id}")
        for index, job in enumerate(jobs):
            finish(repo, clock, job, NOW - timedelta(days=10 - index))
        clock.set(NOW)
        assert repo.maintain(NOW).deleted == 2
        assert [repo.get_job(j.job_id) is not None for j in jobs] == [False, False, True, True]


def test_retention_never_removes_unfinished_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(repo_module, "RETENTION_JOBS", 0)
    clock = InjectedClock()
    with make_repository(tmp_path, clock=clock) as repo:
        ancient = NOW - timedelta(days=400)
        kept = []
        for n, state in enumerate((JobState.QUEUED, JobState.PREPARING, JobState.INTERRUPTED,
                                   JobState.CHECKING, JobState.BLOCKED), start=1):
            job = make_job(n, created=ancient)
            repo.submit(job, f"k{n}")
            if state is not JobState.QUEUED:
                clock.set(ancient)
                job = replace(job, state=state, updated_at=stamp(ancient))
                repo.update_job(job, expected_version=1)
            kept.append(job)
        clock.set(NOW)
        assert repo.maintain(NOW).deleted == 0
        assert [repo.get_job(j.job_id) for j in kept] == kept


def test_retention_leaves_artifact_files_untouched(tmp_path):
    clock = InjectedClock()
    artifact_file = tmp_path / "transcript.txt"
    artifact_file.write_text("hello")
    with make_repository(tmp_path, clock=clock) as repo:
        job = make_job(1)
        repo.submit(job, "k1")
        attempt = make_attempt(job, 1)
        repo.start_attempt(job.job_id, attempt, retry_key=None, expected_version=1)
        repo.record_artifact(make_artifact(attempt, 1, str(artifact_file)))
        finish(repo, clock, repo.get_job(job.job_id), NOW - timedelta(days=200))
        clock.set(NOW)
        assert repo.maintain(NOW).deleted == 1
        assert count(tmp_path / "state.db", "artifacts") == 0
    assert artifact_file.read_text() == "hello"
    assert sorted(p.name for p in tmp_path.iterdir() if not p.name.startswith("state.db")) == ["transcript.txt"]


# -------------------------------------------------------------------- bounds

def test_write_over_bound_is_refused_and_latched(tmp_path, monkeypatch):
    with make_repository(tmp_path) as repo:
        job = make_job(1)
        repo.submit(job, "k1")
        monkeypatch.setattr(repo_module, "DB_BOUND", 1)
        for call in (lambda: repo.submit(make_job(2), "k2"),
                     lambda: repo.update_job(job, expected_version=1),
                     lambda: repo.try_acquire("op", make_plan())):
            with pytest.raises(RepositoryError) as caught:
                call()
            assert reason(caught) is ReasonCode.STORAGE_INSUFFICIENT
        assert repo.admission_blocked is ReasonCode.STORAGE_INSUFFICIENT
        assert repo.get_job(job.job_id) == job  # reads still work
        assert count(tmp_path / "state.db", "jobs") == 1


def test_maintain_keeps_latch_while_unmaintainable_then_clears(tmp_path, monkeypatch):
    with make_repository(tmp_path) as repo:
        repo.submit(make_job(1), "k1")
        monkeypatch.setattr(repo_module, "DB_BOUND", 1)
        with pytest.raises(RepositoryError):
            repo.submit(make_job(2), "k2")
        result = repo.maintain(NOW)
        assert result.blocked is ReasonCode.STORAGE_INSUFFICIENT
        assert repo.admission_blocked is ReasonCode.STORAGE_INSUFFICIENT
        assert result.bytes_after <= result.bytes_before
        monkeypatch.setattr(repo_module, "DB_BOUND", 128 * 1024 * 1024)
        assert repo.maintain(NOW).blocked is None
        assert repo.admission_blocked is None
        assert repo.submit(make_job(2), "k2")[1] is False


def test_maintain_checkpoints_wal_and_never_runs_full_vacuum(tmp_path):
    clock = InjectedClock()
    with make_repository(tmp_path, clock=clock) as repo:
        executed = []
        repo._conn.set_trace_callback(executed.append)
        jobs = [make_job(n) for n in range(1, 6)]
        for job in jobs:
            repo.submit(job, f"k{job.job_id}")
        for job in jobs:
            finish(repo, clock, job, NOW - timedelta(days=120))
        before_wal = (tmp_path / "state.db-wal").stat().st_size
        assert before_wal > 0
        result = repo.maintain(NOW)
        assert result.deleted == 5 and result.bytes_before >= result.bytes_after
        assert (tmp_path / "state.db-wal").stat().st_size == 0
        statements = [s.strip().upper() for s in executed]
        assert not any(s.startswith("VACUUM") for s in statements)
        assert any("INCREMENTAL_VACUUM" in s for s in statements)
        assert repo.db_bytes() == result.bytes_after


def test_constants_match_contract():
    assert repo_module.DB_BOUND == 128 * 1024 * 1024
    assert repo_module.MAINTENANCE_ALLOWANCE == 128 * 1024 * 1024
    assert repo_module.RETENTION_DAYS == 90 and repo_module.RETENTION_JOBS == 10000
    assert repo_module.DB_BOUND + repo_module.MAINTENANCE_ALLOWANCE < 512 * 1024 * 1024


# ------------------------------------------------------------------- domains

def negative_examples():
    data = json.loads((ROOT / "Docs" / "contract-domains.json").read_text(encoding="utf-8"))
    return data["negative_examples"]


@pytest.mark.parametrize("bad", negative_examples())
def test_negative_domain_examples_rejected_on_parse(bad):
    for enum in (JobState, Stage, ReasonCode):
        with pytest.raises(DomainValidationError):
            enum.parse(bad)


@pytest.mark.parametrize("bad", negative_examples())
def test_corrupt_stored_domain_values_are_not_silently_defaulted(tmp_path, bad):
    db = tmp_path / "state.db"
    with make_repository(tmp_path) as repo:
        job = make_job(1)
        repo.submit(job, "k1")
        attempt = make_attempt(job, 1)
        repo.start_attempt(job.job_id, attempt, retry_key=None, expected_version=1)
        stage = make_stage(attempt, 1)
        repo.record_stage(stage)

        raw_exec(db, "UPDATE jobs SET state=? WHERE job_id=?", bad, job.job_id)
        with pytest.raises(RepositoryError):
            repo.get_job(job.job_id)
        with pytest.raises(RepositoryError):
            repo.list_jobs()
        raw_exec(db, "UPDATE jobs SET state='checking' WHERE job_id=?", job.job_id)

        doc = json.loads(repo._conn.execute("SELECT doc FROM jobs").fetchone()[0])
        doc["reason_code"] = bad
        raw_exec(db, "UPDATE jobs SET doc=?", json.dumps(doc))
        with pytest.raises(RepositoryError):
            repo.get_job(job.job_id)

        sdoc = json.loads(repo._conn.execute("SELECT doc FROM stages").fetchone()[0])
        sdoc["stage"] = bad
        raw_exec(db, "UPDATE stages SET doc=?", json.dumps(sdoc))
        with pytest.raises((RepositoryError, DomainValidationError)):
            repo.list_stages(attempt.attempt_id)


def test_state_column_and_document_must_agree(tmp_path):
    with make_repository(tmp_path) as repo:
        job = make_job(1)
        repo.submit(job, "k1")
        raw_exec(tmp_path / "state.db", "UPDATE jobs SET state='completed'")
        with pytest.raises(RepositoryError):
            repo.get_job(job.job_id)


def test_error_messages_hold_no_sensitive_text(tmp_path):
    secret = "SECRET-ROOT"
    fault = Fault()
    with make_repository(tmp_path, fault_hook=fault) as repo:
        fault.arm(sqlite3.OperationalError(f"{tmp_path} {secret} INSERT INTO jobs"))
        with pytest.raises(RepositoryError) as caught:
            repo.submit(make_job(1, root=f"./{secret}"), "k1")
        text = str(caught.value) + repr(caught.value)
        assert secret not in text and str(tmp_path) not in text and "INSERT" not in text
        assert caught.value.__cause__ is None


# ------------------------------------------------------------- queued_jobs (P4-U4)

def test_queued_jobs_oldest_first_and_only_queued(tmp_path):
    with make_repository(tmp_path) as repo:
        same = NOW + timedelta(minutes=5)
        late, tie_a, tie_b, early = (make_job(1, created=NOW + timedelta(hours=1)),
                                     make_job(2, created=same), make_job(3, created=same),
                                     make_job(4, created=NOW))
        started = make_job(5, created=NOW - timedelta(days=1))
        for n, job in enumerate((late, tie_b, tie_a, early, started)):
            repo.submit(job, f"key-{n}")
        repo.update_job(replace(started, state=JobState.CHECKING), expected_version=1)
        assert [j.job_id for j in repo.queued_jobs()] == [
            early.job_id, tie_a.job_id, tie_b.job_id, late.job_id]
        assert [j.job_id for j in repo.queued_jobs(1)] == [early.job_id]
        assert repo.queued_jobs(100) == repo.queued_jobs()


@pytest.mark.parametrize("limit", [0, -1, 101, True, 1.0, "1", None])
def test_queued_jobs_limit_bounds(tmp_path, limit):
    with make_repository(tmp_path) as repo:
        with pytest.raises(RepositoryError) as caught:
            repo.queued_jobs(limit)
        assert reason(caught) is ReasonCode.INVALID_INPUT


def test_queued_jobs_empty_and_closed(tmp_path):
    repo = make_repository(tmp_path)
    assert repo.queued_jobs() == []
    repo.close()
    with pytest.raises(RepositoryError) as caught:
        repo.queued_jobs()
    assert reason(caught) is ReasonCode.SERVICE_STOPPED


def test_migration_from_version_one_adds_queue_index(tmp_path, monkeypatch):
    clock = InjectedClock()
    job = make_job(1)
    assert len(repo_module._MIGRATIONS) == 2
    monkeypatch.setattr(repo_module, "_MIGRATIONS", repo_module._MIGRATIONS[:1])
    with make_repository(tmp_path, clock=clock) as repo:
        assert repo._conn.execute("PRAGMA user_version").fetchone()[0] == 1
        repo.submit(job, "key-1")
        names = {r[0] for r in repo._conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
        assert "jobs_queue" not in names
    monkeypatch.undo()
    with make_repository(tmp_path, clock=clock) as repo:
        assert repo._conn.execute("PRAGMA user_version").fetchone()[0] == 2
        names = {r[0] for r in repo._conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
        assert {"jobs_paging", "jobs_queue"} <= names
        assert [j.job_id for j in repo.queued_jobs()] == [job.job_id]
        assert repo.submit(job, "key-1")[1] is True


# ------------------------------------------------------- admission and closing writes

def checking_of(job, seconds=1):
    return replace(job, state=JobState.CHECKING, updated_at=stamp(NOW + timedelta(seconds=seconds)))


def snapshot(path):
    return {t: count(path, t) for t in ("jobs", "attempts", "idempotency_keys", "reservations")}


def test_admit_attempt_writes_job_attempt_key_and_lease_together(tmp_path):
    with make_repository(tmp_path) as repo:
        job = make_job(1)
        repo.submit(job, "key-1")
        attempt = make_attempt(job, 1)
        checking = checking_of(job)
        result = repo.admit_attempt(checking, attempt, retry_key="r1", expected_version=1,
                                    acquire=("op-1", make_plan()))
        assert result == (checking, attempt, False)
        assert repo.get_job(job.job_id) == checking and repo.get_job_version(job.job_id) == 2
        assert repo.get_attempt(attempt.attempt_id) == attempt
        assert repo.get_retry_attempt("r1") == (job.job_id, attempt)
        assert repo.lease_holders() == (("op-1", True),)


def test_admit_attempt_without_key_or_lease(tmp_path):
    with make_repository(tmp_path) as repo:
        job = make_job(1)
        repo.submit(job, "key-1")
        attempt = make_attempt(job, 1)
        assert repo.admit_attempt(checking_of(job), attempt, retry_key=None, expected_version=1,
                                  acquire=None)[2] is False
        assert repo.lease_holders() == () and repo.get_attempt(attempt.attempt_id) == attempt


def test_admit_attempt_is_atomic_under_injected_failure(tmp_path):
    fault = Fault()
    with make_repository(tmp_path, fault_hook=fault) as repo:
        job = make_job(1)
        repo.submit(job, "key-1")
        before = snapshot(tmp_path / "state.db")
        fault.arm(op="admit_attempt")
        with pytest.raises(RepositoryError) as caught:
            repo.admit_attempt(checking_of(job), make_attempt(job, 1), retry_key="r1",
                               expected_version=1, acquire=("op-1", make_plan()))
        assert reason(caught) is ReasonCode.STATE_WRITE_FAILED
        assert snapshot(tmp_path / "state.db") == before
        assert repo.get_job(job.job_id) == job and repo.get_job_version(job.job_id) == 1
        assert repo.get_retry_attempt("r1") is None and repo.lease_holders() == ()


def test_admit_attempt_replay_returns_stored_values_without_mutation(tmp_path):
    with make_repository(tmp_path) as repo:
        job = make_job(1)
        repo.submit(job, "key-1")
        attempt = make_attempt(job, 1)
        checking = checking_of(job)
        repo.admit_attempt(checking, attempt, retry_key="r1", expected_version=1, acquire=None)
        before = snapshot(tmp_path / "state.db")
        other = make_attempt(job, 2, ordinal=2)
        replay = repo.admit_attempt(checking_of(job, 5), other, retry_key="r1", expected_version=2,
                                    acquire=("op-1", make_plan()))
        assert replay == (checking, attempt, True)
        assert snapshot(tmp_path / "state.db") == before
        assert repo.get_job_version(job.job_id) == 2 and repo.lease_holders() == ()


def test_admit_attempt_key_bound_to_another_job_conflicts_without_mutation(tmp_path):
    with make_repository(tmp_path) as repo:
        first, second = make_job(1), make_job(2)
        repo.submit(first, "key-1")
        repo.submit(second, "key-2")
        repo.admit_attempt(checking_of(first), make_attempt(first, 1), retry_key="r1",
                           expected_version=1, acquire=None)
        before = snapshot(tmp_path / "state.db")
        with pytest.raises(RepositoryError) as caught:
            repo.admit_attempt(checking_of(second), make_attempt(second, 2), retry_key="r1",
                               expected_version=1, acquire=("op-1", make_plan()))
        assert reason(caught) is ReasonCode.CONFLICT
        assert snapshot(tmp_path / "state.db") == before
        assert repo.get_job(second.job_id) == second


def test_admit_attempt_held_lease_conflicts_and_rolls_back(tmp_path):
    with make_repository(tmp_path) as repo:
        job = make_job(1)
        repo.submit(job, "key-1")
        repo.try_acquire("other", make_plan())
        before = snapshot(tmp_path / "state.db")
        with pytest.raises(RepositoryError) as caught:
            repo.admit_attempt(checking_of(job), make_attempt(job, 1), retry_key="r1",
                               expected_version=1, acquire=("op-1", make_plan()))
        assert reason(caught) is ReasonCode.CONFLICT
        assert snapshot(tmp_path / "state.db") == before
        assert repo.get_job(job.job_id) == job and repo.get_retry_attempt("r1") is None
        assert repo.lease_holders() == (("other", True),)


def test_admit_attempt_active_attempt_and_stale_version_conflict(tmp_path):
    with make_repository(tmp_path) as repo:
        job = make_job(1)
        repo.submit(job, "key-1")
        with pytest.raises(RepositoryError) as caught:
            repo.admit_attempt(checking_of(job), make_attempt(job, 1), retry_key=None,
                               expected_version=7, acquire=None)
        assert reason(caught) is ReasonCode.CONFLICT
        repo.admit_attempt(checking_of(job), make_attempt(job, 1), retry_key=None,
                           expected_version=1, acquire=None)
        before = snapshot(tmp_path / "state.db")
        with pytest.raises(RepositoryError) as caught:
            repo.admit_attempt(checking_of(job, 2), make_attempt(job, 2, ordinal=2), retry_key="r2",
                               expected_version=2, acquire=None)
        assert reason(caught) is ReasonCode.CONFLICT
        assert snapshot(tmp_path / "state.db") == before


def test_admit_attempt_refused_while_latched_or_over_bound(tmp_path, monkeypatch):
    fault = Fault()
    with make_repository(tmp_path, fault_hook=fault) as repo:
        job = make_job(1)
        repo.submit(job, "key-1")
        fault.arm(op="update_job")
        with pytest.raises(RepositoryError):
            repo.update_job(job, expected_version=1)
        before = snapshot(tmp_path / "state.db")
        with pytest.raises(RepositoryError) as caught:
            repo.admit_attempt(checking_of(job), make_attempt(job, 1), retry_key=None,
                               expected_version=1, acquire=None)
        assert reason(caught) is ReasonCode.STATE_WRITE_FAILED
        assert repo.maintain(NOW).blocked is None
        monkeypatch.setattr(repo_module, "DB_BOUND", 1)
        with pytest.raises(RepositoryError) as caught:
            repo.admit_attempt(checking_of(job), make_attempt(job, 1), retry_key=None,
                               expected_version=1, acquire=None)
        assert reason(caught) is ReasonCode.STORAGE_INSUFFICIENT
        assert snapshot(tmp_path / "state.db") == before


def test_get_retry_attempt_is_read_only_and_validates_key(tmp_path):
    with make_repository(tmp_path) as repo:
        job = make_job(1)
        repo.submit(job, "key-1")
        before = snapshot(tmp_path / "state.db")
        assert repo.get_retry_attempt("missing") is None
        assert repo.get_retry_attempt("key-1") is None  # a submit key is not a retry key
        assert snapshot(tmp_path / "state.db") == before
        with pytest.raises(RepositoryError) as caught:
            repo.get_retry_attempt("")
        assert reason(caught) is ReasonCode.INVALID_INPUT


def test_acquire_refused_while_latched_release_still_works(tmp_path):
    fault = Fault()
    with make_repository(tmp_path, fault_hook=fault) as repo:
        job = make_job(1)
        repo.submit(job, "key-1")
        assert repo.try_acquire("op-1", make_plan()) is True
        fault.arm(op="update_job")
        with pytest.raises(RepositoryError):
            repo.update_job(job, expected_version=1)
        assert repo.admission_blocked is ReasonCode.STATE_WRITE_FAILED
        with pytest.raises(RepositoryError) as caught:
            repo.try_acquire("op-2", make_plan())
        assert reason(caught) is ReasonCode.STATE_WRITE_FAILED
        with pytest.raises(RepositoryError) as caught:
            repo.transition_with_lease(checking_of(job), expected_version=1,
                                       acquire=("op-3", make_plan()))
        assert reason(caught) is ReasonCode.STATE_WRITE_FAILED
        assert repo.get_job(job.job_id) == job
        repo.release("op-1", writers_reaped=True)
        assert repo.lease_holders() == ()


def active_repo_with_attempt(tmp_path, **kwargs):
    repo = make_repository(tmp_path, **kwargs)
    job = make_job(1)
    repo.submit(job, "key-1")
    attempt = make_attempt(job, 1)
    checking = checking_of(job)
    repo.admit_attempt(checking, attempt, retry_key="r1", expected_version=1,
                       acquire=("op-1", make_plan()))
    return repo, checking, attempt


def test_closing_writes_succeed_above_bound_while_admission_is_refused(tmp_path, monkeypatch):
    repo, checking, attempt = active_repo_with_attempt(tmp_path)
    with repo:
        monkeypatch.setattr(repo_module, "DB_BOUND", 1)
        with pytest.raises(RepositoryError) as caught:
            repo.update_job(checking, expected_version=2)
        assert reason(caught) is ReasonCode.STORAGE_INSUFFICIENT
        with pytest.raises(RepositoryError) as caught:
            repo.admit_attempt(checking, make_attempt(checking, 2, ordinal=2), retry_key=None,
                               expected_version=2, acquire=None)
        assert reason(caught) is ReasonCode.STORAGE_INSUFFICIENT
        done_attempt = replace(attempt, state=JobState.INTERRUPTED)
        assert repo.update_attempt(done_attempt, closing=True) == done_attempt
        assert repo.get_attempt(attempt.attempt_id) == done_attempt
        interrupted = replace(checking, state=JobState.INTERRUPTED,
                              updated_at=stamp(NOW + timedelta(seconds=2)))
        assert repo.update_job(interrupted, expected_version=2, closing=True) == interrupted
        cancelled = replace(interrupted, state=JobState.CANCELLED,
                            updated_at=stamp(NOW + timedelta(seconds=3)))
        repo.transition_with_lease(cancelled, expected_version=3, release=("op-1", True),
                                   closing=True)
        assert repo.get_job(checking.job_id) == cancelled and repo.lease_holders() == ()


def test_closing_with_non_closing_state_is_invalid_input(tmp_path):
    repo, checking, attempt = active_repo_with_attempt(tmp_path)
    with repo:
        preparing = replace(checking, state=JobState.PREPARING)
        for call in (lambda: repo.update_job(preparing, expected_version=2, closing=True),
                     lambda: repo.update_job(checking, expected_version=2, closing=True),
                     lambda: repo.update_attempt(attempt, closing=True),
                     lambda: repo.transition_with_lease(preparing, expected_version=2, closing=True),
                     lambda: repo.transition_with_lease(
                         replace(checking, state=JobState.FAILED), expected_version=2,
                         acquire=("op-9", make_plan()), closing=True)):
            with pytest.raises(RepositoryError) as caught:
                call()
            assert reason(caught) is ReasonCode.INVALID_INPUT
        assert repo.get_job(checking.job_id) == checking and repo.get_attempt(attempt.attempt_id) == attempt
        blocked = replace(checking, state=JobState.BLOCKED)
        assert repo.update_job(blocked, expected_version=2, closing=True) == blocked


def test_close_attempt_and_release_closes_everything_above_bound(tmp_path, monkeypatch):
    repo, checking, attempt = active_repo_with_attempt(tmp_path)
    with repo:
        monkeypatch.setattr(repo_module, "DB_BOUND", 1)
        failed = replace(checking, state=JobState.FAILED, updated_at=stamp(NOW + timedelta(seconds=2)))
        done = replace(attempt, state=JobState.FAILED)
        assert repo.close_attempt_and_release(failed, done, expected_version=2,
                                              release=("op-1", True)) == failed
        assert repo.get_job(checking.job_id) == failed and repo.get_attempt(attempt.attempt_id) == done
        assert repo.lease_holders() == ()


def test_close_attempt_and_release_is_atomic(tmp_path):
    fault = Fault()
    repo, checking, attempt = active_repo_with_attempt(tmp_path, fault_hook=fault)
    with repo:
        failed = replace(checking, state=JobState.FAILED, updated_at=stamp(NOW + timedelta(seconds=2)))
        done = replace(attempt, state=JobState.FAILED)
        fault.arm(op="close_attempt_and_release")
        with pytest.raises(RepositoryError) as caught:
            repo.close_attempt_and_release(failed, done, expected_version=2, release=("op-1", True))
        assert reason(caught) is ReasonCode.STATE_WRITE_FAILED
        assert repo.get_job(checking.job_id) == checking and repo.get_job_version(checking.job_id) == 2
        assert repo.get_attempt(attempt.attempt_id) == attempt
        assert repo.lease_holders() == (("op-1", True),)
        # Unresolved writers or a stale version also leave everything untouched.
        for kwargs, expected in (({"expected_version": 2, "release": ("op-1", False)},
                                  ReasonCode.WRITER_UNRESOLVED),
                                 ({"expected_version": 1, "release": ("op-1", True)},
                                  ReasonCode.CONFLICT)):
            with pytest.raises(RepositoryError) as caught:
                repo.close_attempt_and_release(failed, done, **kwargs)
            assert reason(caught) is expected
            assert repo.get_attempt(attempt.attempt_id) == attempt
            assert repo.lease_holders() == (("op-1", True),)
        with pytest.raises(RepositoryError) as caught:
            repo.close_attempt_and_release(failed, attempt, expected_version=2, release=None)
        assert reason(caught) is ReasonCode.INVALID_INPUT


def test_active_jobs_order_states_and_bounds(tmp_path):
    with make_repository(tmp_path) as repo:
        states = {1: JobState.EXPORTING, 2: JobState.QUEUED, 3: JobState.CHECKING, 4: JobState.BLOCKED,
                  5: JobState.COMPLETED, 6: JobState.INTERRUPTED, 7: JobState.TRANSCRIBING}
        for n, state in states.items():
            job = make_job(n)
            repo.submit(job, f"key-{n}")
            if state is not JobState.QUEUED:
                repo.update_job(replace(job, state=state), expected_version=1)
        assert [j.job_id for j in repo.active_jobs()] == [ident(1), ident(3), ident(7)]
        assert [j.job_id for j in repo.active_jobs(2)] == [ident(1), ident(3)]
        assert [j.job_id for j in repo.active_jobs(limit=1)] == [ident(1)]


@pytest.mark.parametrize("limit", [0, -1, 101, True, "5", None, 1.5])
def test_active_jobs_limit_bounds(tmp_path, limit):
    with make_repository(tmp_path) as repo:
        with pytest.raises(RepositoryError) as caught:
            repo.active_jobs(limit)
        assert reason(caught) is ReasonCode.INVALID_INPUT


# ---------------------------------------------------------------------- live

@pytest.mark.live
def test_own_transport_smoke(tmp_path):
    from podmuncher.volumes import make_volume_provider

    provider = make_volume_provider()

    def state_volume(path):
        return provider.sample(path, sqlite_state=True).state_allowed.value

    clock = InjectedClock()
    fault = Fault()
    db = tmp_path / "state.db"

    def open_repo():
        return Repository.open(db, clock=clock, state_volume=state_volume, fault_hook=fault)

    jobs = [make_job(n) for n in range(1, 4)]
    with open_repo() as repo:
        for job in jobs:
            assert repo.submit(job, f"key-{job.job_id}")[1] is False
        first = jobs[0]
        attempt = make_attempt(first, 1)
        repo.start_attempt(first.job_id, attempt, retry_key="retry-1", expected_version=1)
        repo.record_stage(make_stage(attempt, 1))
        repo.record_artifact(make_artifact(attempt, 1))
        checking = replace(repo.get_job(first.job_id), state=JobState.CHECKING,
                           updated_at=stamp(NOW + timedelta(seconds=30)))
        repo.transition_with_lease(checking, expected_version=2, acquire=("op-1", make_plan()))
        assert repo.try_acquire("op-2", make_plan()) is False
        done = replace(checking, state=JobState.COMPLETED, updated_at=stamp(NOW + timedelta(seconds=60)))
        repo.transition_with_lease(done, expected_version=3, release=("op-1", True))
        # Fault in the middle of a lease cycle: nothing from that transaction survives.
        fault.arm(op="transition_with_lease")
        second = jobs[1]
        moved = replace(second, state=JobState.CHECKING, updated_at=stamp(NOW + timedelta(seconds=90)))
        with pytest.raises(RepositoryError) as caught:
            repo.transition_with_lease(moved, expected_version=1, acquire=("op-3", make_plan()))
        assert reason(caught) is ReasonCode.STATE_WRITE_FAILED
        assert repo.admission_blocked is ReasonCode.STATE_WRITE_FAILED
        assert repo.maintain(NOW).blocked is None
    with open_repo() as repo:
        assert repo._conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        page, cursor = repo.list_jobs()
        assert cursor is None and len(page) == 3
        assert repo.get_job(first.job_id).state is JobState.COMPLETED
        assert repo.get_job(second.job_id) == second
        assert repo.lease_holders() == ()
        assert repo.list_stages(attempt.attempt_id) == [make_stage(attempt, 1)]
        assert len(repo.list_artifacts(first.job_id)) == 1
        assert repo.submit(second, f"key-{second.job_id}")[1] is True
    for path in tmp_path.iterdir():
        path.unlink()
    assert not tuple(tmp_path.iterdir())
    print("PODMUNCHER_SMOKE_OK")
