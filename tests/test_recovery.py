"""Singleton lock, private runtime files, graceful stop and startup reconciliation (P4-U5)."""
import asyncio
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from podmuncher import domains
from podmuncher import recovery as rec
from podmuncher.admission import AdmissionPlan
from podmuncher.domains import JobState, ReasonCode
from podmuncher.repository import Repository
from podmuncher.requests import JobRequest
from podmuncher.snapshots import Artifact, Attempt, Job, Measurement

NOW = datetime(2026, 10, 12, 12, 0, tzinfo=UTC)
ROOT = Path(__file__).resolve().parent.parent
SHA = "a" * 64
SECRET = "s3cr3t-credential-VALUE-0123456789"
NONCE = "instance-nonce-1"
OLD_PID = 4242
OLD_CREATE = 1_000.0


def ident(n, tag=1):
    return f"{n:08x}-{tag:04x}-4111-8111-111111111111"


def stamp(moment):
    return moment.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class FakeClock:
    def __init__(self):
        self.ticks = 0

    def now(self):
        self.ticks += 1
        return stamp(NOW + timedelta(milliseconds=self.ticks))


class FakeProcessIdentity:
    """Scripted pid -> create_time; an unknown pid does not exist."""

    def __init__(self, table=None):
        self.table = dict(table or {})
        self.calls = []

    def __call__(self, pid):
        self.calls.append(pid)
        if pid not in self.table:
            return None
        return rec.ProcessIdentity(pid, self.table[pid])


def make_recovery_dir(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (tmp_path / "out").mkdir()
    return state


def make_request():
    return JobRequest.from_dict({
        "schema_version": 1, "source": {"kind": "local", "input": "./input.mp4"},
        "outputs": {"transcript": True, "video": False, "video_quality": "1080p"},
        "selection": {"device": "auto", "profile": "balanced", "model": "auto", "language": "auto"},
        "output_root": "./out",
        "limits": {"media_bytes": 100, "duration_seconds": 20, "export_bytes": 10}},
        verified_models=frozenset({"small"}))


def make_job(n):
    request = make_request()
    when = stamp(NOW + timedelta(seconds=n))
    return Job(ident(n), request, request.request_hash, JobState.QUEUED, ReasonCode.OK, None, None,
               when, when, (), None)


def open_repo(path, **kwargs):
    return Repository.open(path, clock=lambda: NOW, state_volume=lambda p: True,
                           verified_models=frozenset({"small"}), **kwargs)


def plan():
    return AdmissionPlan(Measurement(True, "fake", domains.FieldStatus.OK, stamp(NOW), None), (), ())


def make_active(repo, n, *, lease=True, state=JobState.TRANSCRIBING):
    """A job in an active state with an attempt and (optionally) the single lease."""
    job = make_job(n)
    repo.submit(job, f"key-{n}")
    attempt = Attempt(ident(n, 2), job.job_id, 1, None, None, stamp(NOW), None, JobState.CHECKING,
                      ReasonCode.OK, (), 0)
    repo.start_attempt(job.job_id, attempt, retry_key=None, expected_version=1)
    stored = repo.get_job(job.job_id)
    repo.update_job(replace(stored, state=state), expected_version=repo.get_job_version(job.job_id))
    if lease:
        assert repo.try_acquire(f"op-{n}", plan())
    return repo.get_job(job.job_id)


class FakeContainers:
    """Scripted Docker port.  ``log`` is shared so tests can assert ordering."""

    def __init__(self, log, repo=None):
        self.log = log
        self.repo = repo
        self.by_instance = {}
        self.present = set()
        self.stubborn = set()
        self.fail_stop = set()
        self.unknown = set()
        self.list_error = False
        self.states_at_remove = []

    def add(self, instance, n, labels=None):
        ref = rec.ContainerRef(f"{n:064x}", {rec.LABEL_INSTANCE: instance} if labels is None else labels)
        self.by_instance.setdefault(instance, []).append(ref)
        self.present.add(ref.container_id)
        return ref

    async def list(self, instance):
        self.log.append(("list", instance))
        if self.list_error:
            raise rec.RecoveryError(ReasonCode.DOCKER_UNAVAILABLE)
        return list(self.by_instance.get(instance, ()))

    async def stop(self, ref):
        self.log.append(("stop", ref.container_id))
        if ref.container_id in self.fail_stop:
            raise rec.RecoveryError(ReasonCode.DOCKER_UNAVAILABLE)

    async def kill(self, ref):
        self.log.append(("kill", ref.container_id))

    async def remove(self, ref):
        self.log.append(("remove", ref.container_id))
        if self.repo is not None:
            self.states_at_remove.append([job.state for job in self.repo.list_jobs()[0]])
        if ref.container_id not in self.stubborn:
            self.present.discard(ref.container_id)

    async def exists(self, ref):
        self.log.append(("exists", ref.container_id))
        if ref.container_id in self.unknown:
            return None
        return ref.container_id in self.present


class FakeLauncher:
    def __init__(self, log, reaped=True):
        self.log = log
        self.reaped = reaped
        self.stops = []

    async def stop(self, reason):
        self.log.append(("launcher.stop", reason.value))
        self.stops.append(reason)
        return self.reaped


@pytest.fixture
def env(tmp_path):
    state = make_recovery_dir(tmp_path)
    log = []
    repo = open_repo(tmp_path / "repo.db")
    lock = rec.acquire_singleton(state)
    pids = FakeProcessIdentity()
    yield type("Env", (), {"tmp": tmp_path, "state": state, "out": tmp_path / "out", "log": log, "repo": repo,
                           "lock": lock, "pids": pids, "clock": FakeClock(),
                           "containers": FakeContainers(log, repo)})
    lock.release()
    repo.close()


def stale_runtime(env, nonce=NONCE):
    """A previous instance's private runtime file whose process is gone."""
    rec.write_runtime_file(env.state / rec.RUNTIME_FILE, port=8123, credential=SECRET, instance_nonce=nonce,
                           identity=rec.ProcessIdentity(OLD_PID, OLD_CREATE))


def reconcile(env, audit=None, recovery=None):
    return asyncio.run(rec.startup_reconcile(
        env.state, env.repo, env.containers, process_identity=env.pids, audit=audit, clock=env.clock,
        output_roots=[env.out], recovery=recovery))


def loosen(path):
    if os.name == "nt":
        subprocess.run([os.path.join(os.environ["SystemRoot"], "System32", "icacls.exe"), str(path),
                        "/grant", "Everyone:(R)"], shell=False, check=True, capture_output=True, timeout=20)
    else:
        os.chmod(path, 0o644)


# ------------------------------------------------------------------- singleton lock


def test_second_acquire_conflicts_then_release_allows_reacquire(tmp_path):
    state = make_recovery_dir(tmp_path)
    first = rec.acquire_singleton(state)
    with pytest.raises(rec.RecoveryError) as error:
        rec.acquire_singleton(state)
    assert error.value.reason_code is ReasonCode.CONFLICT
    first.release()
    first.release()  # idempotent
    assert not first.held and (state / rec.LOCK_NAME).exists()  # never deletes the lock file
    again = rec.acquire_singleton(state)
    assert again.held
    again.release()


def test_runtime_file_write_requires_the_held_lock(tmp_path):
    state = make_recovery_dir(tmp_path)
    with pytest.raises(rec.RecoveryError) as error:
        rec.write_runtime_file(state / rec.RUNTIME_FILE, port=1, credential=SECRET, instance_nonce=NONCE,
                               identity=rec.ProcessIdentity(1, 1.0))
    assert error.value.reason_code is ReasonCode.CONFLICT
    assert not (state / rec.RUNTIME_FILE).exists()


# ------------------------------------------------------- runtime file and identity


def test_live_record_roundtrips_and_pid_reuse_is_stale(env):
    path = env.state / rec.RUNTIME_FILE
    stale_runtime(env)
    env.pids.table[OLD_PID] = OLD_CREATE
    record = rec.read_runtime_file(path, process_identity=env.pids)
    assert record is not None and record.pid == OLD_PID and record.port == 8123
    assert record.credential == SECRET
    # Same pid, different create time: the pid was reused by another process.
    env.pids.table[OLD_PID] = OLD_CREATE + 50
    assert rec.read_runtime_file(path, process_identity=env.pids) is None
    # Pid gone.
    del env.pids.table[OLD_PID]
    assert rec.read_runtime_file(path, process_identity=env.pids) is None


def test_stale_nonce_missing_and_invalid_records_are_stale(env):
    path = env.state / rec.RUNTIME_FILE
    assert rec.read_runtime_file(path, process_identity=env.pids) is None  # missing
    stale_runtime(env)
    env.pids.table[OLD_PID] = OLD_CREATE
    assert rec.read_runtime_file(path, process_identity=env.pids, expected_nonce="other") is None
    assert rec.read_runtime_file(path, process_identity=env.pids, expected_nonce=NONCE) is not None
    path.write_text("{not json", encoding="utf-8")
    if os.name == "nt":
        rec._run_icacls(str(path), "/inheritance:r", "/grant:r", f"{rec._windows_principal()}:(F)")
    else:
        os.chmod(path, 0o600)
    assert rec.read_runtime_file(path, process_identity=env.pids) is None


def test_bad_permissions_are_refused(env):
    path = env.state / rec.RUNTIME_FILE
    stale_runtime(env)
    assert rec.verify_runtime_permissions(path) is True
    loosen(path)
    assert rec.verify_runtime_permissions(path) is False
    with pytest.raises(rec.RecoveryError) as error:
        rec.read_runtime_file(path, process_identity=env.pids)
    assert error.value.reason_code is ReasonCode.INVALID_CONFIG
    assert SECRET not in str(error.value)


def test_missing_file_and_link_do_not_verify(env):
    assert rec.verify_runtime_permissions(env.state / "nope.json") is False
    stale_runtime(env)
    link = env.state / "link.json"
    try:
        os.symlink(env.state / rec.RUNTIME_FILE, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation not permitted")
    assert rec.verify_runtime_permissions(link) is False


@pytest.mark.skipif(os.name == "nt", reason="POSIX mode check")
def test_posix_runtime_file_mode_is_0600(env):
    stale_runtime(env)
    assert (env.state / rec.RUNTIME_FILE).stat().st_mode & 0o777 == 0o600


def test_icacls_output_parser():
    path = r"C:\x y\runtime.json"
    one = f"{path} HOST\\me:(F)\n\nSuccessfully processed 1 files; Failed processing 0 files\n"
    assert rec._parse_icacls(one, path) == [("HOST\\me", "(F)")]
    two = (f"{path} NT AUTHORITY\\SYSTEM:(I)(F)\n"
           "      BUILTIN\\Administrators:(I)(F)\n\nSuccessfully processed 1 files; Failed processing 0 files\n")
    assert [who for who, _ in rec._parse_icacls(two, path)] == ["NT AUTHORITY\\SYSTEM",
                                                                "BUILTIN\\Administrators"]
    with pytest.raises(ValueError):
        rec._parse_icacls("garbage line\n", path)


def test_credential_never_in_repr_or_errors(env):
    stale_runtime(env)
    env.pids.table[OLD_PID] = OLD_CREATE
    record = rec.read_runtime_file(env.state / rec.RUNTIME_FILE, process_identity=env.pids)
    assert SECRET not in repr(record) and SECRET not in str(record)
    errors = []
    for call in (
        lambda: rec.write_runtime_file(env.state / rec.RUNTIME_FILE, port=0, credential=SECRET,
                                       instance_nonce=NONCE, identity=rec.ProcessIdentity(1, 1.0)),
        lambda: rec.write_runtime_file(env.tmp / "elsewhere" / rec.RUNTIME_FILE, port=5, credential=SECRET,
                                       instance_nonce=NONCE, identity=rec.ProcessIdentity(1, 1.0)),
        lambda: rec.write_intent(env.state, {"x": object()}),
    ):
        with pytest.raises(rec.RecoveryError) as error:
            call()
        errors.append(error.value)
    assert all(SECRET not in str(e) and SECRET not in repr(e) and str(env.tmp) not in str(e) for e in errors)
    del env.pids.table[OLD_PID]
    report = reconcile(env)
    assert SECRET not in repr(report) and str(env.tmp) not in repr(report)


# ------------------------------------------------------------ startup reconcile


def test_duplicate_daemon_live_record_refuses_reconcile(env):
    job = make_active(env.repo, 1)
    stale_runtime(env)
    env.pids.table[OLD_PID] = OLD_CREATE  # still alive
    env.containers.add(NONCE, 1)
    with pytest.raises(rec.RecoveryError) as error:
        reconcile(env)
    assert error.value.reason_code is ReasonCode.CONFLICT
    assert env.log == [] and env.repo.get_job(job.job_id).state is JobState.TRANSCRIBING
    assert env.repo.lease_holders() == (("op-1", True),)


def test_survivor_is_stopped_and_removed_before_any_job_is_interrupted(env):
    job = make_active(env.repo, 1)
    queued = make_job(2)
    env.repo.submit(queued, "key-2")
    stale_runtime(env)
    ref = env.containers.add(NONCE, 1)
    report = reconcile(env)
    assert [e for e in env.log if e[0] in ("stop", "kill", "remove")] == [
        ("stop", ref.container_id), ("kill", ref.container_id), ("remove", ref.container_id)]
    assert env.log[0] == ("list", NONCE)
    assert env.containers.states_at_remove == [[JobState.QUEUED, JobState.TRANSCRIBING]]
    after = env.repo.get_job(job.job_id)
    assert after.state is JobState.INTERRUPTED and after.reason_code is ReasonCode.SERVICE_LOST
    assert env.repo.get_attempt(after.attempt_id).state is JobState.INTERRUPTED
    assert env.repo.get_job(queued.job_id).state is JobState.QUEUED  # queue retained
    assert env.repo.lease_holders() == ()
    assert (report.ok, report.survivors_found, report.survivors_cleared, report.interrupted,
            report.leases_released) == (True, 1, 1, 1, 1)


def test_intent_alone_identifies_the_previous_instance(env):
    make_active(env.repo, 1)
    rec.write_intent(env.state, {"action": "running", "instance_nonce": "from-intent"})
    env.containers.add("from-intent", 1)
    report = reconcile(env)
    assert report.ok and report.survivors_cleared == 1


def test_unlabeled_and_foreign_containers_are_never_touched(env):
    make_active(env.repo, 1)
    stale_runtime(env)
    mine = env.containers.add(NONCE, 1)
    foreign = env.containers.add(NONCE, 2, labels={rec.LABEL_INSTANCE: "someone-else"})
    unlabeled = env.containers.add(NONCE, 3, labels={})
    other = env.containers.add("never-listed", 4)
    report = reconcile(env)
    touched = {e[1] for e in env.log if e[0] in ("stop", "kill", "remove", "exists")}
    assert touched == {mine.container_id}
    assert {foreign.container_id, unlabeled.container_id, other.container_id} <= env.containers.present
    assert report.foreign_skipped == 2 and report.ok


@pytest.mark.parametrize("mode", ["stubborn", "unknown", "list_error"])
def test_unresolved_survivor_keeps_lease_and_interrupts_nothing(env, mode):
    job = make_active(env.repo, 1)
    stale_runtime(env)
    ref = env.containers.add(NONCE, 1)
    if mode == "stubborn":
        env.containers.stubborn.add(ref.container_id)
        env.containers.fail_stop.add(ref.container_id)
    elif mode == "unknown":
        env.containers.unknown.add(ref.container_id)
    else:
        env.containers.list_error = True
    report = reconcile(env)
    assert report.reason_code is ReasonCode.WRITER_UNRESOLVED and not report.ok
    assert report.survivors_unresolved == 1 and report.interrupted == 0 and report.leases_released == 0
    assert env.repo.get_job(job.job_id).state is JobState.TRANSCRIBING
    assert env.repo.lease_holders() == (("op-1", True),)


def test_active_work_with_no_known_instance_is_unresolved(env):
    job = make_active(env.repo, 1)
    report = reconcile(env)  # no runtime file and no intent
    assert report.reason_code is ReasonCode.WRITER_UNRESOLVED
    assert env.repo.get_job(job.job_id).state is JobState.TRANSCRIBING
    assert env.repo.lease_holders() == (("op-1", True),)
    assert env.log == []


def test_clean_state_reconciles_with_nothing_to_do(env):
    report = reconcile(env)
    assert report.ok and (report.survivors_found, report.interrupted, report.leases_released) == (0, 0, 0)
    assert report.admission_latch is None


def test_reconcile_audit_failure_does_not_block(env):
    make_active(env.repo, 1)
    stale_runtime(env)
    env.containers.add(NONCE, 1)

    def broken(*args):
        raise RuntimeError(SECRET)

    recovery = rec.Recovery(env.state)
    report = reconcile(env, audit=broken, recovery=recovery)
    assert report.ok and not report.audit_delivered and report.audit_loss == 1
    assert SECRET not in repr(recovery.audit_loss)


def test_disk_state_failure_is_reported_and_restart_reconciles_safely(env):
    def boom(op):
        if op == "submit":
            raise sqlite3.OperationalError("disk full")

    faulty = open_repo(env.tmp / "faulty.db", fault_hook=boom)
    with pytest.raises(Exception):  # noqa: B017 - repository closed-reason error
        faulty.submit(make_job(9), "k9")
    assert faulty.admission_blocked is ReasonCode.STATE_WRITE_FAILED
    env.repo.close()
    env.repo = faulty
    report = reconcile(env)
    assert report.ok and report.admission_latch is ReasonCode.STATE_WRITE_FAILED
    assert faulty.admission_blocked is ReasonCode.STATE_WRITE_FAILED  # not cleared here
    faulty.close()
    # Restart: a fresh repository over the same file starts clean and reconciles.
    env.repo = open_repo(env.tmp / "faulty.db")
    assert reconcile(env).admission_latch is None


def test_partial_temp_files_are_removed_and_counted_others_untouched(env):
    job = make_active(env.repo, 1, lease=False)
    stale_runtime(env)
    job_dir = env.out / f"podmuncher-{job.job_id}"
    job_dir.mkdir()
    mine = [job_dir / f".transcript.txt.{'a' * 32}.partial", job_dir / f".podmuncher-manifest.json.{'b' * 32}.partial"]
    mine[0].write_bytes(b"x" * 10)
    mine[1].write_bytes(b"y" * 5)
    keep = [job_dir / "notes.partial", job_dir / ".x.partial", job_dir / "transcript.txt"]
    for path in keep:
        path.write_bytes(b"keep")
    other_dir = env.out / f"podmuncher-{ident(77)}"  # not a known job: never entered
    other_dir.mkdir()
    (other_dir / f".a.{'c' * 32}.partial").write_bytes(b"z")
    target = env.tmp / "outside"
    target.mkdir()
    (target / f".t.{'d' * 32}.partial").write_bytes(b"w" * 7)
    try:
        os.symlink(target, env.out / f"podmuncher-{ident(5)}", target_is_directory=True)
        env.repo.submit(make_job(5), "key-5")
    except (OSError, NotImplementedError):
        pass
    file_link = job_dir / f".l.{'e' * 32}.partial"
    try:
        os.symlink(target / f".t.{'d' * 32}.partial", file_link)
    except (OSError, NotImplementedError):
        file_link = None
    report = reconcile(env)
    assert report.partial_files_removed == 2 and report.reclaimed_bytes == 15
    assert not any(p.exists() for p in mine)
    assert all(p.exists() for p in keep)
    assert (other_dir / f".a.{'c' * 32}.partial").exists()
    assert (target / f".t.{'d' * 32}.partial").read_bytes() == b"w" * 7  # link target untouched
    if file_link is not None:
        assert os.path.lexists(file_link)


def _artifact(job, n, size, *, available=True):
    return Artifact(ident(n, 5), domains.ArtifactSlot.TRANSCRIPT_TXT, domains.ArtifactFormat.TXT, size, SHA,
                    available, job.attempt_id, None)


def _write_manifest(job_dir, artifacts):
    doc = {"schema_version": 1, "artifacts": [
        {"artifact_id": a.artifact_id, "file_name": name, "bytes": size} for a, name, size in artifacts]}
    (job_dir / "podmuncher-manifest.json").write_text(json.dumps(doc), encoding="utf-8")


def test_manifest_mismatch_is_reported_and_nothing_deleted(env):
    good = make_active(env.repo, 1, lease=False)
    bad = make_active(env.repo, 2, lease=False)
    gone = make_active(env.repo, 3, lease=False)
    stale_runtime(env)
    art_good, art_bad, art_gone = _artifact(good, 1, 5), _artifact(bad, 2, 5), _artifact(gone, 3, 5)
    for art in (art_good, art_bad, art_gone):
        env.repo.record_artifact(art)
    for job, art, size in ((good, art_good, 5), (bad, art_bad, 9)):
        job_dir = env.out / f"podmuncher-{job.job_id}"
        job_dir.mkdir()
        (job_dir / "transcript.txt").write_bytes(b"hello")  # 5 bytes on disk
        _write_manifest(job_dir, [(art, "transcript.txt", size)])
    before = sorted(p.relative_to(env.out).as_posix() for p in env.out.rglob("*"))
    report = reconcile(env)
    assert report.manifests_checked == 3 and report.manifest_mismatches == 2
    assert set(report.mismatched_job_ids) == {bad.job_id, gone.job_id}
    assert sorted(p.relative_to(env.out).as_posix() for p in env.out.rglob("*")) == before


# ----------------------------------------------------------------- graceful stop


def stop(env, launcher, audit=None, recovery=None):
    return asyncio.run(rec.graceful_stop(None, env.repo, launcher, audit=audit, clock=env.clock,
                                         recovery=recovery))


def test_graceful_stop_interrupts_active_job_and_retains_queue(env):
    job = make_active(env.repo, 1)
    queued = [make_job(2), make_job(3)]
    for item in queued:
        env.repo.submit(item, f"key-{item.job_id}")
    recovery = rec.Recovery(env.state, instance_nonce=NONCE)
    audits = []
    report = stop(env, FakeLauncher(env.log), audit=lambda *a: audits.append(a), recovery=recovery)
    assert report.ok and report.reason_code is ReasonCode.SERVICE_STOPPED
    assert recovery.accepting is False
    after = env.repo.get_job(job.job_id)
    assert after.state is JobState.INTERRUPTED and after.reason_code is ReasonCode.SERVICE_STOPPED
    assert env.repo.get_attempt(after.attempt_id).state is JobState.INTERRUPTED
    assert [env.repo.get_job(q.job_id).state for q in queued] == [JobState.QUEUED] * 2
    assert (report.interrupted, report.queued_retained, report.leases_released) == (1, 2, 1)
    assert env.repo.lease_holders() == () and not report.lease_retained
    assert audits == [("service_stop", None, ReasonCode.SERVICE_STOPPED)]
    assert report.audit_delivered and report.audit_loss == 0


def test_graceful_stop_unreaped_writer_fails_and_keeps_lease(env):
    job = make_active(env.repo, 1)
    report = stop(env, FakeLauncher(env.log, reaped=False))
    assert not report.ok and report.reason_code is ReasonCode.WRITER_UNRESOLVED
    assert report.lease_retained and report.interrupted == 0
    after = env.repo.get_job(job.job_id)
    assert after.state is JobState.TRANSCRIBING and after.reason_code is ReasonCode.WRITER_UNRESOLVED
    assert env.repo.lease_holders() == (("op-1", True),)


def test_graceful_stop_launcher_exception_is_unreaped(env):
    make_active(env.repo, 1)

    class Exploding:
        async def stop(self, reason):
            raise RuntimeError("boom")

    report = stop(env, Exploding())
    assert not report.ok and report.lease_retained


def test_audit_failure_during_stop_still_stops_and_writes_intent(env):
    job = make_active(env.repo, 1)
    recovery = rec.Recovery(env.state, instance_nonce=NONCE)

    def broken(*args):
        raise OSError(SECRET)

    report = stop(env, FakeLauncher(env.log), audit=broken, recovery=recovery)
    assert report.ok and env.repo.get_job(job.job_id).state is JobState.INTERRUPTED
    assert report.audit_delivered is False and report.audit_loss == 1 and report.intent_written
    (notice,) = recovery.audit_loss
    assert notice.action == "service_stop" and SECRET not in repr(notice)
    intent = rec.read_intent(env.state)
    assert intent["audit_lost"] is True and intent["action"] == "stop" and intent["instance_nonce"] == NONCE
    assert intent["active_job_id"] == job.job_id and SECRET not in json.dumps(intent)


def test_audit_loss_notices_are_bounded():
    recovery = rec.Recovery()
    for _ in range(rec.AUDIT_LOSS_MAX + 20):
        recovery.note_audit_loss("service_stop", ReasonCode.SERVICE_STOPPED)
    assert len(recovery.audit_loss) == rec.AUDIT_LOSS_MAX
    assert recovery.audit_loss_total == rec.AUDIT_LOSS_MAX + 20


class FakeScheduler:
    def __init__(self, log, result=True):
        self.log, self.result, self.timeouts = log, result, []

    async def quiesce(self, *, timeout):
        self.log.append(("quiesce", None))
        self.timeouts.append(timeout)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


class LoggedRepo:
    """Delegates to the real repository, logging reads and optionally failing them."""

    def __init__(self, repo, log, fail=()):
        self._repo, self._log, self._fail = repo, log, set(fail)

    def __getattr__(self, name):
        target = getattr(self._repo, name)
        if name not in ("active_jobs", "lease_holders", "queued_jobs", "list_jobs"):
            return target

        def call(*args, **kwargs):
            self._log.append(("repo", name))
            if name in self._fail:
                raise rec.RepositoryError(ReasonCode.STATE_WRITE_FAILED)
            return target(*args, **kwargs)
        return call


def stop_with(env, scheduler, launcher, repo=None, audit=None, recovery=None):
    return asyncio.run(rec.graceful_stop(scheduler, repo or env.repo, launcher, audit=audit, clock=env.clock,
                                         recovery=recovery))


def test_stop_order_is_quiesce_then_launcher_then_repository_reads(env):
    job = make_active(env.repo, 1)
    scheduler = FakeScheduler(env.log)
    report = stop_with(env, scheduler, FakeLauncher(env.log), repo=LoggedRepo(env.repo, env.log))
    assert [entry[0] for entry in env.log[:2]] == ["quiesce", "launcher.stop"]
    assert env.log[2:] and all(entry[0] == "repo" for entry in env.log[2:])  # no read before the stop
    assert scheduler.timeouts == [rec.QUIESCE_TIMEOUT_SECONDS]
    assert report.ok and env.repo.get_job(job.job_id).state is JobState.INTERRUPTED


@pytest.mark.parametrize("result", [False, RuntimeError("boom"), None])
def test_quiesce_not_proven_is_writer_unresolved(env, result):
    job = make_active(env.repo, 1)
    launcher = FakeLauncher(env.log)
    report = stop_with(env, FakeScheduler(env.log, result), launcher)
    assert launcher.stops  # the stop is still attempted
    assert not report.ok and report.reason_code is ReasonCode.WRITER_UNRESOLVED
    assert report.interrupted == 0 and report.leases_released == 0 and report.lease_retained
    assert env.repo.get_job(job.job_id).state is JobState.TRANSCRIBING
    assert env.repo.lease_holders() == (("op-1", True),)


def test_repository_read_failure_cannot_skip_the_stop(env):
    make_active(env.repo, 1)
    launcher = FakeLauncher(env.log)
    stop_with(env, None, launcher, repo=LoggedRepo(env.repo, env.log, fail={"active_jobs"}))
    assert launcher.stops == [ReasonCode.SERVICE_STOPPED]
    assert env.log[0] == ("launcher.stop", "service_stopped")


def test_repository_read_failure_after_stop_is_reported_without_success(env):
    job = make_active(env.repo, 1)
    launcher = FakeLauncher(env.log)
    report = stop_with(env, None, launcher, repo=LoggedRepo(env.repo, env.log, fail={"active_jobs"}))
    assert launcher.stops and not report.ok
    assert report.reason_code is ReasonCode.STATE_WRITE_FAILED
    assert report.interrupted == 0 and report.leases_released == 0
    assert env.repo.get_job(job.job_id).state is JobState.TRANSCRIBING
    assert env.repo.lease_holders() == (("op-1", True),)


def test_interrupt_above_the_size_bound_uses_closing_writes(env, monkeypatch):
    from podmuncher import repository as repository_module

    job = make_active(env.repo, 1)
    monkeypatch.setattr(repository_module, "DB_BOUND", 1)
    with pytest.raises(rec.RepositoryError):  # an ordinary write is refused above the bound
        env.repo.update_job(replace(env.repo.get_job(job.job_id), state=JobState.EXPORTING),
                            expected_version=env.repo.get_job_version(job.job_id))
    report = stop(env, FakeLauncher(env.log))
    assert report.ok and report.interrupted == 1 and report.leases_released == 1
    after = env.repo.get_job(job.job_id)
    assert after.state is JobState.INTERRUPTED
    assert env.repo.get_attempt(after.attempt_id).state is JobState.INTERRUPTED
    assert env.repo.lease_holders() == ()


def test_reconcile_above_the_size_bound_also_succeeds(env, monkeypatch):
    from podmuncher import repository as repository_module

    job = make_active(env.repo, 1)
    stale_runtime(env)
    env.containers.add(NONCE, 1)
    monkeypatch.setattr(repository_module, "DB_BOUND", 1)
    report = reconcile(env)
    assert report.ok and report.interrupted == 1 and report.leases_released == 1
    assert env.repo.get_job(job.job_id).state is JobState.INTERRUPTED


def test_atomic_close_fault_leaves_no_partial_state(env):
    fail = {"on": False}

    def hook(op):
        if fail["on"] and op == "close_attempt_and_release":
            raise sqlite3.OperationalError("disk full")

    repo = open_repo(env.tmp / "atomic.db", fault_hook=hook)
    job = make_active(repo, 1)
    fail["on"] = True
    env.repo.close()
    env.repo = repo
    report = stop(env, FakeLauncher(env.log))
    assert not report.ok and report.reason_code is ReasonCode.STATE_WRITE_FAILED
    assert report.interrupted == 0 and report.leases_released == 0 and report.lease_retained
    stored = repo.get_job(job.job_id)
    assert stored.state is JobState.TRANSCRIBING
    assert repo.get_attempt(stored.attempt_id).state is JobState.CHECKING
    assert repo.lease_holders() == (("op-1", True),)


def test_job_without_an_attempt_row_is_interrupted_and_lease_released_after_reap(env):
    job = make_job(1)
    env.repo.submit(job, "key-1")
    env.repo.update_job(replace(env.repo.get_job(job.job_id), state=JobState.TRANSCRIBING),
                        expected_version=env.repo.get_job_version(job.job_id))
    assert env.repo.try_acquire("op-1", plan())
    unresolved = stop(env, FakeLauncher(env.log, reaped=False))  # not reaped: nothing changes
    assert not unresolved.ok and env.repo.lease_holders() == (("op-1", True),)
    report = stop(env, FakeLauncher(env.log))
    assert report.ok and report.interrupted == 1 and report.leases_released == 1
    after = env.repo.get_job(job.job_id)
    assert after.state is JobState.INTERRUPTED and after.reason_code is ReasonCode.SERVICE_STOPPED
    assert env.repo.lease_holders() == ()


def test_several_active_jobs_are_drained_without_a_history_scan(env):
    for n in range(1, 4):
        make_active(env.repo, n, lease=False)
    assert env.repo.try_acquire("op-a", plan())
    report = stop(env, FakeLauncher(env.log))
    assert report.ok and report.interrupted == 3 and report.leases_released == 1
    assert env.repo.active_jobs() == [] and env.repo.lease_holders() == ()


def test_nonce_is_preserved_in_intent_when_reconcile_audit_fails(env):
    make_active(env.repo, 1)
    rec.write_intent(env.state, {"action": "running", "instance_nonce": "from-intent"})
    env.containers.add("from-intent", 1)

    def broken(*args):
        raise OSError(SECRET)

    report = reconcile(env, audit=broken, recovery=rec.Recovery(env.state))
    assert report.ok and not report.audit_delivered
    intent = rec.read_intent(env.state)
    assert intent["audit_lost"] is True and "from-intent" in intent["instance_nonces"]
    assert intent["instance_nonce"] == "from-intent" and SECRET not in json.dumps(intent)


def test_stop_audit_failure_keeps_a_nonce_already_on_disk(env):
    make_active(env.repo, 1)
    rec.write_intent(env.state, {"action": "running", "instance_nonce": "older-instance"})

    def broken(*args):
        raise OSError("down")

    stop(env, FakeLauncher(env.log), audit=broken, recovery=rec.Recovery(env.state, instance_nonce=NONCE))
    intent = rec.read_intent(env.state)
    assert intent["instance_nonce"] == NONCE and set(intent["instance_nonces"]) == {NONCE, "older-instance"}


def test_telemetry_audit_loss_counts_as_a_loss_never_a_delivery(env):
    from podmuncher.telemetry import TelemetryError

    make_active(env.repo, 1)
    recovery = rec.Recovery(env.state, instance_nonce=NONCE)
    calls = []

    def adapter(action, target, reason):
        calls.append(action)
        raise TelemetryError(ReasonCode.AUDIT_LOSS)

    report = stop(env, FakeLauncher(env.log), audit=adapter, recovery=recovery)
    assert calls == ["service_stop"] and report.ok
    assert report.audit_delivered is False and report.audit_loss == 1 and report.intent_written
    assert rec.read_intent(env.state)["audit_lost"] is True
    assert [n.reason_code for n in recovery.audit_loss] == [ReasonCode.SERVICE_STOPPED]


# ------------------------------------------------------ docker adapter and domains


class FakeCli:
    def __init__(self, labels_by_id):
        self.labels = labels_by_id
        self.calls = []

    async def run(self, *args, owner, timeout):
        self.calls.append(args)
        result = type("R", (), {})()
        result.returncode, result.stdout = 0, b""
        if args[:2] == ("container", "inspect"):
            cid = args[2]
            if cid in self.labels:
                result.stdout = json.dumps([{"Id": cid, "Config": {"Labels": self.labels[cid]}}]).encode()
            else:
                result.returncode = 1
        elif args[0] == "ps":
            if "-q" in args:
                result.stdout = b""
            else:
                result.stdout = "\n".join(self.labels).encode()
        return result


def test_native_adapter_verifies_labels_before_acting():
    mine, theirs = "a" * 64, "b" * 64
    cli = FakeCli({mine: {rec.LABEL_INSTANCE: NONCE}, theirs: {rec.LABEL_INSTANCE: "other"}})
    adapter = rec.NativeOwnedContainers(cli, owner="op")
    refs = asyncio.run(adapter.list(NONCE))
    assert [r.container_id for r in refs] == [mine]  # the foreign one is dropped at listing
    forged = rec.ContainerRef(theirs, {rec.LABEL_INSTANCE: NONCE})
    with pytest.raises(rec.RecoveryError) as error:
        asyncio.run(adapter.stop(forged))
    assert error.value.reason_code is ReasonCode.UNAUTHORIZED
    assert not any(call[0] in ("stop", "kill", "rm") for call in cli.calls)
    asyncio.run(adapter.remove(refs[0]))
    assert cli.calls[-1] == ("rm", mine)
    assert asyncio.run(adapter.exists(rec.ContainerRef("c" * 64, {}))) is False
    assert asyncio.run(adapter.exists(refs[0])) is True


class ScriptedCli:
    """ps lists ``listed``; inspect fails for ``bad_inspect``; the id filter answers from ``present``."""

    def __init__(self, listed, *, bad_inspect=(), present=(), filter_fails=False):
        self.listed, self.bad_inspect = list(listed), set(bad_inspect)
        self.present, self.filter_fails = set(present), filter_fails
        self.calls = []

    async def run(self, *args, owner, timeout):
        self.calls.append(args)
        result = type("R", (), {})()
        result.returncode, result.stdout = 0, b""
        if args[:2] == ("container", "inspect"):
            cid = args[2]
            if cid in self.bad_inspect:
                result.returncode = 1
            else:
                labels = {rec.LABEL_INSTANCE: NONCE}
                result.stdout = json.dumps([{"Id": cid, "Config": {"Labels": labels}}]).encode()
        elif args[0] == "ps" and "-q" in args:
            if self.filter_fails:
                result.returncode = 1
            else:
                cid = args[-1].split("=", 1)[1]
                result.stdout = cid.encode() if cid in self.present else b""
        elif args[0] == "ps":
            result.stdout = "\n".join(self.listed).encode()
        return result


GHOST = "d" * 64


def test_inspect_failure_with_container_still_present_is_unresolved_and_nothing_is_interrupted(env):
    job = make_active(env.repo, 1)
    stale_runtime(env)
    cli = ScriptedCli([GHOST], bad_inspect={GHOST}, present={GHOST})
    env.containers = rec.NativeOwnedContainers(cli, owner="op")
    report = reconcile(env)
    assert report.reason_code is ReasonCode.WRITER_UNRESOLVED and report.survivors_unresolved == 1
    assert report.interrupted == 0 and report.leases_released == 0
    assert env.repo.get_job(job.job_id).state is JobState.TRANSCRIBING
    assert env.repo.lease_holders() == (("op-1", True),)
    assert not any(call[0] in ("stop", "kill", "rm") for call in cli.calls)  # never acted on unverified


def test_inspect_failure_with_unknown_existence_is_unresolved(env):
    job = make_active(env.repo, 1)
    stale_runtime(env)
    cli = ScriptedCli([GHOST], bad_inspect={GHOST}, filter_fails=True)
    env.containers = rec.NativeOwnedContainers(cli, owner="op")
    report = reconcile(env)
    assert report.reason_code is ReasonCode.WRITER_UNRESOLVED
    assert env.repo.get_job(job.job_id).state is JobState.TRANSCRIBING
    assert env.repo.lease_holders() == (("op-1", True),)


def test_inspect_failure_with_proven_absence_proceeds(env):
    job = make_active(env.repo, 1)
    stale_runtime(env)
    cli = ScriptedCli([GHOST], bad_inspect={GHOST})
    env.containers = rec.NativeOwnedContainers(cli, owner="op")
    report = reconcile(env)
    assert report.ok and report.survivors_found == 0 and report.interrupted == 1
    assert ("ps", "-a", "-q", "--no-trunc", "--filter", f"id={GHOST}") in cli.calls
    assert env.repo.get_job(job.job_id).state is JobState.INTERRUPTED
    assert env.repo.lease_holders() == ()


@pytest.mark.parametrize("enum_name", ["ReasonCode", "JobState"])
def test_domains_reject_contract_negative_examples(enum_name):
    registry = json.loads((ROOT / "Docs" / "contract-domains.json").read_text(encoding="utf-8"))
    enum = getattr(domains, enum_name)
    for value in registry["negative_examples"]:
        with pytest.raises(domains.DomainValidationError):
            enum.parse(value)
    assert enum.parse("writer_unresolved" if enum_name == "ReasonCode" else "interrupted")


# ------------------------------------------------------------------------- live

_CHILD = """
import sys
sys.path.insert(0, sys.argv[1])
from podmuncher.recovery import RecoveryError, acquire_singleton
try:
    lock = acquire_singleton(sys.argv[2])
    print("HELD", flush=True)
except RecoveryError as error:
    print("CONFLICT", error.reason_code.value, flush=True)
    raise SystemExit(0)
sys.stdin.readline()
lock.release()
"""


@pytest.mark.live
def test_own_transport_smoke():
    base = tempfile.mkdtemp(prefix="podmuncher-smoke-")
    procs = []
    watchdog = None
    try:
        state = os.path.join(base, "state")
        os.mkdir(state)
        for _ in range(2):
            procs.append(subprocess.Popen(
                [sys.executable, "-I", "-c", _CHILD, str(ROOT / "src"), state], shell=False,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True))
        watchdog = threading.Timer(60, lambda: [p.kill() for p in procs])
        watchdog.start()
        answers = [p.stdout.readline().strip() for p in procs]
        assert sorted(a.split()[0] for a in answers) == ["CONFLICT", "HELD"], answers
        for p in procs:
            try:
                p.stdin.write("\n")
                p.stdin.close()
            except OSError:
                pass
        for p in procs:
            assert p.wait(timeout=30) == 0
        # With both children gone the lock is free; hold it here and write a real runtime file.
        lock = rec.acquire_singleton(state)
        try:
            me = rec.psutil_identity(os.getpid())
            assert me is not None
            path = os.path.join(state, rec.RUNTIME_FILE)
            rec.write_runtime_file(path, port=4321, credential=SECRET, instance_nonce=NONCE, identity=me)
            assert rec.verify_runtime_permissions(path) is True
            record = rec.read_runtime_file(path, process_identity=rec.psutil_identity)
            assert record is not None and record.pid == os.getpid() and SECRET not in repr(record)
            with pytest.raises(rec.RecoveryError):  # a duplicate daemon cannot take the lock
                rec.acquire_singleton(state)
        finally:
            lock.release()
    finally:
        if watchdog is not None:
            watchdog.cancel()
        for p in procs:
            if p.poll() is None:
                p.terminate()
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=15)
            for stream in (p.stdin, p.stdout):
                if stream is not None:
                    stream.close()
        shutil.rmtree(base, ignore_errors=True)
    assert not os.path.exists(base)
    print("PODMUNCHER_SMOKE_OK")
