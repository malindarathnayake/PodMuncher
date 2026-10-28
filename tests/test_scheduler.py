"""Serial scheduler: stage order, lease discipline, blocking, cancel, fallback, telemetry."""
import asyncio
import hashlib
import itertools
import json
import threading
import time
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from podmuncher import asr_worker, config, repository, telemetry
from podmuncher import scheduler as sch
from podmuncher.admission import AdmissionPlan
from podmuncher.artifacts import ArtifactError
from podmuncher.domains import (
    ArtifactFormat,
    ArtifactSlot,
    DomainValidationError,
    FieldStatus,
    JobState,
    PreflightReason,
    ReasonCode,
    Stage,
    WorkerSlot,
)
from podmuncher.hardware import AdmittedWorkerRef
from podmuncher.protocol import WorkerEventRecord
from podmuncher.repository import Repository, RepositoryError
from podmuncher.requests import JobRequest
from podmuncher.runtime import RuntimeFailure, WorkerResult
from podmuncher.scheduler import Scheduler, SchedulerError
from podmuncher.snapshots import Artifact, Job, Measurement
from podmuncher.transitions import MediaFacts, TransitionError

SHA = "a" * 64
NOW = datetime(2026, 10, 12, 12, 0, tzinfo=UTC)
ROOT = Path(__file__).resolve().parent.parent
T_SLOTS = sorted(s.value for s in ArtifactSlot if s is not ArtifactSlot.ORIGINAL_VIDEO)
FORMATS = {
    ArtifactSlot.ORIGINAL_VIDEO: ArtifactFormat.VIDEO, ArtifactSlot.TRANSCRIPT_TXT: ArtifactFormat.TXT,
    ArtifactSlot.TRANSCRIPT_TIMESTAMPED_TXT: ArtifactFormat.TIMESTAMPED_TXT,
    ArtifactSlot.TRANSCRIPT_READABLE_TXT: ArtifactFormat.READABLE_TXT,
    ArtifactSlot.TRANSCRIPT_SRT: ArtifactFormat.SRT, ArtifactSlot.TRANSCRIPT_JSON: ArtifactFormat.JSON,
}
REMOTE = "https://example.com/a.mp4"


def ident(n, tag=1):
    return f"{n:08x}-{tag:04x}-4111-8111-111111111111"


def stamp(moment):
    return moment.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def make_request(*, transcript=True, video=False, model="auto", device="auto", kind="local", source=None):
    source = source or ("./input.mp4" if kind == "local" else REMOTE)
    return JobRequest.from_dict({
        "schema_version": 1, "source": {"kind": "local" if kind == "local" else "direct", "input": source},
        "outputs": {"transcript": transcript, "video": video, "video_quality": "1080p"},
        "selection": {"device": device, "profile": "balanced", "model": model, "language": "auto"},
        "output_root": "./out",
        "limits": {"media_bytes": 100, "duration_seconds": 20, "export_bytes": 10}},
        verified_models=frozenset({"small"}))


def make_job(request, n):
    when = stamp(NOW + timedelta(seconds=n))
    return Job(ident(n), request, request.request_hash, JobState.QUEUED, ReasonCode.OK, None, None,
               when, when, (), None)


# ---------------------------------------------------------------------------- fakes


class FakeClock:
    def __init__(self):
        self.ticks = 0
        self.t = 0.0

    def now(self):
        self.ticks += 1
        return stamp(NOW + timedelta(milliseconds=self.ticks))

    def monotonic(self):
        return self.t


@dataclass(frozen=True)
class Verified:
    slot: str
    sha256: str = SHA


class FakeLauncher:
    """Scripted per-operation worker; the call log is the ground truth for stage order."""

    def __init__(self, log):
        self.log = log
        self.fail = {}
        self.gate = {}
        self.entered = {}
        self.events = {}
        self.fail_counts = {}
        self.unreaped_ops = set()
        self.reaped = True
        self.stopped = False
        self.stops = []
        self.prepared = set()
        self.callbacks = []
        self.runs = 0

    @property
    def writers_reaped(self):
        return self.reaped

    def ready(self, capability):
        return capability in self.prepared

    async def prepare(self, capability):
        self.log.append(("prepare", capability))
        self.prepared.add(capability)

    async def run(self, request, *, stage, on_event):
        op = request.operation
        self.runs += 1
        self.log.append(("run", op))
        self.callbacks.append((request, on_event))
        self.entered.setdefault(op, asyncio.Event()).set()
        for make in self.events.get(op, ()):
            on_event(make(request))
        if op in self.gate:
            await self.gate[op].wait()
            if self.stopped:
                self.reaped = op not in self.unreaped_ops and self.reaped
                raise RuntimeFailure(ReasonCode.CANCELLED)
        if op in self.unreaped_ops:
            self.reaped = False
            raise RuntimeFailure(ReasonCode.WRITER_UNRESOLVED)
        failure = self.fail.get(op)
        if failure is not None:
            left = self.fail_counts.get(op)
            if left is None or left > 0:
                if left is not None:
                    self.fail_counts[op] = left - 1
                raise RuntimeFailure(failure)
        slot = {"download": "downloaded_media", "probe": "probe_metadata",
                "transcribe": "transcript_segments"}[op]
        claim = WorkerEventRecord(1, request.job_id, request.attempt_id, request.stage_id, 1, "artifact",
                                  {"slot": slot, "bytes": 5, "sha256": SHA})
        ref = AdmittedWorkerRef("ctx", "npipe:////./pipe/docker_engine", "b" * 64, "inst", "job", "att")
        return WorkerResult(ref, True, 0, (claim,))

    async def stop(self, reason):
        self.stops.append(reason)
        self.stopped = True
        for gate in self.gate.values():
            gate.set()
        return self.reaped


class FakeExporter:
    def __init__(self, log):
        self.log = log

    def export(self, segments, slots):
        self.log.append(("export", tuple(sorted(s.value for s in slots))))
        return {slot: f"tmp-{slot.value}" for slot in slots}


class FakePublisher:
    def __init__(self, log):
        self.log = log
        self.facts = MediaFacts(True, True, 12.0)
        self.fail_publish = {}
        self.fail_manifest = None
        self.manifests = []
        self.content = {}
        self.n = 0

    def source_identity(self, job):
        # Stand-in for the content sha256: keyed by an editable "file content", never the path.
        key = job.request.source.input
        return hashlib.sha256(self.content.get(key, "content-v1").encode()).hexdigest()

    def write_manifest(self, job_id, attempt_id, published, provenance):
        self.log.append(("manifest", len(published)))
        if self.fail_manifest is not None:
            raise ArtifactError(self.fail_manifest)
        self.manifests.append((job_id, attempt_id, tuple(a.slot.value for a in published)))

    def assign(self, job, attempt, slots):
        self.n += 1
        return {WorkerSlot.parse(s): f"{WorkerSlot.parse(s).value}-{self.n}.bin" for s in slots}

    def verify(self, job, attempt, assignment, claim, *, writers_reaped):
        if not writers_reaped:
            raise ArtifactError(ReasonCode.WRITER_UNRESOLVED)
        assert claim.payload["slot"] in {s.value for s in assignment}
        return Verified(claim.payload["slot"], claim.payload["sha256"])

    def verify_local_video(self, job, attempt):
        return Verified("original_video")

    def verify_export(self, job, attempt, slot, ref):
        return Verified(slot.value)

    def probe_facts(self, verified):
        assert verified.slot == "probe_metadata"
        return self.facts

    def reuse(self, job, record):
        self.log.append(("reuse", record.stage.value))
        return {slot: (f"{slot.value}-reused.bin", Verified(slot.value)) for slot in record.output_slots}

    def discard(self, job, attempt):
        self.log.append(("discard",))

    async def publish(self, job, attempt, verified, slot):
        self.log.append(("publish", slot.value))
        reason = self.fail_publish.get(slot)
        if reason is not None:
            raise ArtifactError(reason)
        self.n += 1
        return Artifact(ident(self.n, 5), slot, FORMATS[slot], 5, SHA, True, attempt.attempt_id, None)


@dataclass(frozen=True)
class Choice:
    name: str


class FakeModels:
    TIERS = ("large", "medium", "small")

    def __init__(self, log):
        self.log = log
        self.prepared = []
        self.device = "cpu"

    def select(self, job, *, facts, lower_than=None):
        self.log.append(("select", None if lower_than is None else lower_than.name))
        if lower_than is None:
            return Choice(self.TIERS[0])
        index = self.TIERS.index(lower_than.name) + 1
        if index >= len(self.TIERS):
            raise DomainValidationError(ReasonCode.ENVELOPE_UNVALIDATED)
        return Choice(self.TIERS[index])

    def capability(self, choice):
        return f"asr_{self.device}"

    def selection(self, choice):
        return self.device, choice.name

    async def prepare(self, choice):
        self.log.append(("models.prepare", choice.name))
        self.prepared.append(choice.name)
        return f"snap-{choice.name}"


class FakePreflight:
    def __init__(self, log, clock):
        self.log = log
        self.clock = clock
        self.rule = lambda stage: None

    def plan(self, job, stage):
        self.log.append(("preflight", stage.value))
        reason = self.rule(stage)
        stamp_now = stamp(NOW)
        if reason is None:
            return AdmissionPlan(Measurement(True, "fake", FieldStatus.OK, stamp_now, None), (), ())
        status = FieldStatus.STALE if reason is ReasonCode.STORAGE_STALE else FieldStatus.ERROR
        return AdmissionPlan(Measurement(None, "fake", status, stamp_now, reason), (), ())


class FakeTelemetry:
    def __init__(self):
        self.events = []
        self.counts = []

    def span(self, name, attrs):
        telemetry = self

        class Span:
            def __enter__(self):
                telemetry.events.append(("enter", name, dict(attrs)))

            def __exit__(self, *exc):
                telemetry.events.append(("exit", name, dict(attrs)))

        return Span()

    def count(self, name, attrs):
        self.counts.append((name, dict(attrs)))


class Harness:
    def __init__(self, tmp_path, **repo_kwargs):
        self.log = []
        self.clock = FakeClock()
        self.repo = Repository.open(tmp_path / "state.db", clock=lambda: NOW, state_volume=lambda p: True,
                                    verified_models=frozenset({"small"}), **repo_kwargs)
        self.launcher = FakeLauncher(self.log)
        self.models = FakeModels(self.log)
        self.exporter = FakeExporter(self.log)
        self.publisher = FakePublisher(self.log)
        self.preflight = FakePreflight(self.log, self.clock)
        self.telemetry = FakeTelemetry()
        self.counter = itertools.count(1)
        self.scheduler = self.make_scheduler()

    def make_scheduler(self):
        return Scheduler(self.repo, preflight=self.preflight, launcher=self.launcher, models=self.models,
                         exporter=self.exporter, publisher=self.publisher, clock=self.clock,
                         telemetry=self.telemetry, ids=lambda: ident(next(self.counter), 9))

    async def submit(self, request, n=1):
        job = make_job(request, n)
        await self.scheduler.submit(job, f"key-{n}")
        return job

    async def run(self, request, n=1):
        job = await self.submit(request, n)
        assert await self.scheduler.run_once() is True
        return self.repo.get_job(job.job_id)

    def ops(self):
        return [entry for entry in self.log if entry[0] != "preflight"]

    def calls(self, kind):
        return [entry for entry in self.log if entry[0] == kind]

    def close(self):
        self.repo.close()


@pytest.fixture
def h(tmp_path):
    harness = Harness(tmp_path)
    yield harness
    harness.close()


# -------------------------------------------------------------------- matrix


def _expected(mode, kind, audio, fail):
    """Terminal (state, reason) per the outcome matrix for one scenario."""
    video, transcript = mode in ("video", "video_transcript"), mode in ("transcript", "video_transcript")
    if fail in ("download", "probe"):
        return JobState.FAILED, {"download": ReasonCode.DOWNLOAD_TIMEOUT,
                                 "probe": ReasonCode.MEDIA_PROBE_FAILED}[fail]
    if fail == "video_publish":
        return JobState.FAILED, ReasonCode.PUBLICATION_FAILED
    if transcript and not audio:
        if mode == "transcript":
            return JobState.FAILED, ReasonCode.NO_AUDIO
        return JobState.PARTIAL, ReasonCode.NO_AUDIO
    if fail == "transcribe":
        return (JobState.PARTIAL if video else JobState.FAILED), ReasonCode.WORKER_EXIT_UNEXPECTED
    if fail == "export":
        return JobState.PARTIAL, ReasonCode.PUBLICATION_FAILED
    return JobState.COMPLETED, ReasonCode.OK


def _cases():
    for mode, kind, audio, fail in itertools.product(
            ("transcript", "video_transcript", "video"), ("local", "remote"), (True, False),
            (None, "download", "probe", "transcribe", "export", "video_publish")):
        video, transcript = mode != "transcript", mode != "video"
        if fail == "download" and kind == "local":
            continue
        if fail == "video_publish" and not video:
            continue
        if fail in ("transcribe", "export") and not (transcript and audio):
            continue
        yield mode, kind, audio, fail


@pytest.mark.asyncio
@pytest.mark.parametrize("mode,kind,audio,fail", list(_cases()))
async def test_mode_outcome_error_matrix(h, mode, kind, audio, fail):
    request = make_request(transcript=mode != "video", video=mode != "transcript", kind=kind)
    h.publisher.facts = MediaFacts(audio, True, 12.0)
    if fail == "download":
        h.launcher.fail["download"] = ReasonCode.DOWNLOAD_TIMEOUT
    if fail == "probe":
        h.launcher.fail["probe"] = ReasonCode.MEDIA_PROBE_FAILED
    if fail == "transcribe":
        h.launcher.fail["transcribe"] = ReasonCode.WORKER_EXIT_UNEXPECTED
    if fail == "export":
        h.publisher.fail_publish[ArtifactSlot.TRANSCRIPT_SRT] = ReasonCode.PUBLICATION_FAILED
    if fail == "video_publish":
        h.publisher.fail_publish[ArtifactSlot.ORIGINAL_VIDEO] = ReasonCode.PUBLICATION_FAILED
    job = await h.run(request)
    state, reason = _expected(mode, kind, audio, fail)
    assert (job.state, job.reason_code) == (state, reason)
    assert h.repo.lease_holders() == ()  # every terminal outcome gives the lease back
    published = {a.slot.value for a in job.artifacts}
    if state is JobState.COMPLETED:
        want = set()
        if mode != "transcript":
            want.add("original_video")
        if mode != "video":
            want |= set(T_SLOTS)
        assert published == want
    if not audio and mode != "video":  # silent media never provisions ASR
        assert not h.calls("select") and not h.calls("models.prepare")
        assert ("run", "transcribe") not in h.log
    if mode == "video":
        assert not h.calls("select") and ("run", "transcribe") not in h.log
    attempt = h.repo.get_attempt(job.attempt_id)
    assert attempt.state is job.state


@pytest.mark.asyncio
async def test_exact_stage_order_remote_video_transcript(h):
    job = await h.run(make_request(video=True, kind="remote"))
    assert job.state is JobState.COMPLETED
    assert h.log[:17] == [
        ("preflight", "checking"),
        ("preflight", "preparing"), ("prepare", "media"),
        ("preflight", "downloading"), ("run", "download"),
        ("preflight", "probing"), ("run", "probe"),
        ("preflight", "probing"), ("publish", "original_video"), ("manifest", 1),
        ("preflight", "preparing"), ("select", None), ("models.prepare", "large"),
        ("preflight", "transcribing"), ("run", "transcribe"),
        ("preflight", "exporting"), ("export", tuple(T_SLOTS)),
    ]
    # Video publication precedes ASR provisioning; each export publication re-runs preflight.
    assert h.log.index(("publish", "original_video")) < h.log.index(("select", None))
    tail = h.log[17:]
    assert [e for e in tail if e[0] == "publish"] == [("publish", s) for s in T_SLOTS]
    assert [e for e in tail if e[0] == "preflight"] == [("preflight", "exporting")] * 5
    stages = [r.stage.value for r in h.repo.get_attempt(job.attempt_id).stage_records]
    assert stages == ["checking", "preparing", "downloading", "probing", "preparing", "transcribing",
                      "exporting"]
    assert len({r.stage_id for r in h.repo.get_attempt(job.attempt_id).stage_records}) == 7
    assert len(h.repo.list_stages(job.attempt_id)) == 7


@pytest.mark.asyncio
async def test_local_transcript_skips_download_and_prepared_media(h):
    h.launcher.prepared.add("media")
    job = await h.run(make_request(kind="local"))
    assert job.state is JobState.COMPLETED
    assert [e for e in h.log if e[0] in ("prepare", "run")] == [("run", "probe"), ("run", "transcribe")]
    stages = [r.stage.value for r in h.repo.get_attempt(job.attempt_id).stage_records]
    assert stages == ["checking", "probing", "preparing", "transcribing", "exporting"]


@pytest.mark.asyncio
async def test_request_and_slots_never_leak_host_paths(h):
    await h.run(make_request(kind="remote"))
    requests = [req for req, _ in h.launcher.callbacks]
    assert [r.operation for r in requests] == ["download", "probe", "transcribe"]
    assert requests[0].source == {"kind": "direct", "input": REMOTE}
    assert requests[1].source is None and requests[1].input_basename == requests[0].assigned_slots[
        "downloaded_media"]
    assert requests[2].model_snapshot_id == "snap-large" and requests[2].capability == "asr_cpu"
    assert all("/" not in v and "\\" not in v for r in requests for v in r.assigned_slots.values())


# ---------------------------------------------------------------- lease / queue


@pytest.mark.asyncio
async def test_setup_and_job_never_overlap(h):
    await h.submit(make_request())
    plan = AdmissionPlan(Measurement(True, "fake", FieldStatus.OK, stamp(NOW), None), (), ())
    gate, started = asyncio.Event(), asyncio.Event()

    async def action():
        started.set()
        await gate.wait()
        return True

    setup = asyncio.create_task(h.scheduler.run_setup("setup-1", plan, action))
    await started.wait()
    assert h.repo.lease_holders()[0][0] == "setup-1"
    assert await h.scheduler.run_once() is False
    with pytest.raises(SchedulerError) as caught:
        await h.scheduler.run_setup("setup-2", plan, action)
    assert caught.value.reason_code is ReasonCode.CONFLICT
    assert h.repo.try_acquire("other", plan) is False  # the lease is exclusive
    gate.set()
    await setup
    assert h.repo.lease_holders() == ()
    assert await h.scheduler.run_once() is True


@pytest.mark.asyncio
async def test_job_holds_lease_so_setup_is_refused(h):
    job = await h.submit(make_request())
    h.launcher.gate["probe"] = asyncio.Event()
    running = asyncio.create_task(h.scheduler.run_once())
    await h.launcher.entered.setdefault("probe", asyncio.Event()).wait()
    assert len(h.repo.lease_holders()) == 1
    plan = AdmissionPlan(Measurement(True, "fake", FieldStatus.OK, stamp(NOW), None), (), ())

    async def action():
        return True

    with pytest.raises(SchedulerError):
        await h.scheduler.run_setup("setup", plan, action)
    assert await h.scheduler.run_once() is False
    h.launcher.gate["probe"].set()
    assert await running is True
    assert h.repo.get_job(job.job_id).state is JobState.COMPLETED


@pytest.mark.asyncio
async def test_setup_unreaped_keeps_lease_and_refuses_scheduling(h):
    await h.submit(make_request())
    plan = AdmissionPlan(Measurement(True, "fake", FieldStatus.OK, stamp(NOW), None), (), ())

    async def action():
        return False

    await h.scheduler.run_setup("setup", plan, action)
    assert h.repo.lease_holders() == (("setup", True),)
    assert await h.scheduler.run_once() is False
    blocked = AdmissionPlan(Measurement(None, "fake", FieldStatus.STALE, stamp(NOW), ReasonCode.STORAGE_STALE),
                            (), ())
    with pytest.raises(SchedulerError) as caught:
        await Harness.__dict__["make_scheduler"](h).run_setup("s2", blocked, action)
    assert caught.value.reason_code is ReasonCode.CONFLICT


@pytest.mark.asyncio
async def test_blocked_job_never_takes_a_lease(h, monkeypatch):
    acquired = []
    original = h.repo.transition_with_lease

    def spy(job, *, expected_version, acquire=None, release=None, closing=False):
        acquired.append((acquire, release))
        return original(job, expected_version=expected_version, acquire=acquire, release=release,
                        closing=closing)

    monkeypatch.setattr(h.repo, "transition_with_lease", spy)
    monkeypatch.setattr(h.repo, "try_acquire", lambda *a, **k: pytest.fail("blocked job took a lease"))
    h.preflight.rule = lambda stage: ReasonCode.STORAGE_STALE
    job = await h.run(make_request())
    assert job.state is JobState.BLOCKED and job.reason_code is ReasonCode.STORAGE_STALE
    assert job.next_action == "recheck" and job.attempt_id is None
    assert h.repo.lease_holders() == () and acquired == []
    assert h.ops() == []  # nothing launched
    # A later, still-queued job is not starved by the blocked one.
    h.preflight.rule = lambda stage: None
    second = await h.run(make_request(kind="local"), n=2)
    assert second.state is JobState.COMPLETED


@pytest.mark.asyncio
async def test_run_once_is_fifo_and_serial(h):
    jobs = [await h.submit(make_request(), n) for n in (3, 1, 2)]
    order = []
    for _ in range(3):
        assert await h.scheduler.run_once() is True
        order.append([j.job_id for j in jobs if h.repo.get_job(j.job_id).state is JobState.COMPLETED])
    assert [len(x) for x in order] == [1, 2, 3]
    assert order[0] == [ident(1)] and set(order[1]) == {ident(1), ident(2)}
    assert await h.scheduler.run_once() is False
    attempts = [h.repo.get_job(j.job_id).attempt_id for j in jobs]
    assert len(set(attempts)) == 3


@pytest.mark.asyncio
async def test_queue_bound_is_enforced_by_the_repository(h):
    for n in range(1, 101):
        await h.submit(make_request(), n)
    with pytest.raises(RepositoryError) as caught:
        await h.submit(make_request(), 101)
    assert caught.value.reason_code is ReasonCode.QUEUE_FULL
    assert len(h.repo.queued_jobs(100)) == 100
    assert h.repo.lease_holders() == ()  # queued jobs hold no reservation


@pytest.mark.asyncio
async def test_repository_latch_stops_scheduling(h):
    await h.submit(make_request())
    h.repo._latch = ReasonCode.STORAGE_INSUFFICIENT
    assert await h.scheduler.run_once() is False


# ------------------------------------------------------- preflight / capacity


@pytest.mark.asyncio
async def test_stale_capacity_before_a_stage_stops_and_releases(h):
    h.preflight.rule = lambda stage: ReasonCode.STORAGE_STALE if stage is Stage.PROBING else None
    job = await h.run(make_request(kind="remote"))
    assert job.state is JobState.BLOCKED and job.reason_code is ReasonCode.STORAGE_STALE
    assert ("run", "probe") not in h.log and ("run", "transcribe") not in h.log
    assert h.launcher.stops == [ReasonCode.STORAGE_STALE]
    assert h.repo.lease_holders() == ()
    assert h.repo.get_attempt(job.attempt_id).state is JobState.BLOCKED


@pytest.mark.asyncio
async def test_stale_capacity_after_video_publication_is_partial(h):
    state = {"published": False}

    def rule(stage):
        state["published"] = state["published"] or ("publish", "original_video") in h.log
        return ReasonCode.STORAGE_STALE if state["published"] and stage is Stage.PREPARING else None

    h.preflight.rule = rule
    job = await h.run(make_request(video=True, kind="remote"))
    assert job.state is JobState.PARTIAL and job.reason_code is ReasonCode.STORAGE_STALE
    assert [a.slot for a in job.artifacts] == [ArtifactSlot.ORIGINAL_VIDEO]
    assert ("select", None) not in h.log and h.repo.lease_holders() == ()


@pytest.mark.asyncio
async def test_preflight_fault_fails_closed(h):
    def boom(stage):
        raise OSError("secret path")

    h.preflight.rule = boom
    job = await h.run(make_request())
    assert job.state is JobState.BLOCKED and job.reason_code is ReasonCode.STORAGE_UNKNOWN


@pytest.mark.asyncio
async def test_external_disk_drop_mid_stage_is_never_completed(h):
    h.launcher.fail["transcribe"] = ReasonCode.STORAGE_GUARD_TRIPPED
    job = await h.run(make_request())
    assert job.state is JobState.BLOCKED and job.reason_code is ReasonCode.STORAGE_GUARD_TRIPPED
    assert h.repo.lease_holders() == () and h.launcher.stops

    h.launcher.stops.clear()
    other = await h.run(make_request(video=True, kind="remote"), n=2)
    assert other.state is JobState.PARTIAL and other.reason_code is ReasonCode.STORAGE_GUARD_TRIPPED
    assert [a.slot for a in other.artifacts] == [ArtifactSlot.ORIGINAL_VIDEO]
    assert other.state is not JobState.COMPLETED and h.repo.lease_holders() == ()


@pytest.mark.asyncio
async def test_disk_drop_while_publishing_exports_is_partial(h):
    h.publisher.fail_publish[ArtifactSlot.TRANSCRIPT_TXT] = ReasonCode.STORAGE_GUARD_TRIPPED
    job = await h.run(make_request())
    assert job.state is JobState.PARTIAL and job.reason_code is ReasonCode.STORAGE_GUARD_TRIPPED
    assert {a.slot.value for a in job.artifacts} == set(T_SLOTS) - {"transcript_txt"}


# ------------------------------------------------------------ writer / cancel


@pytest.mark.asyncio
async def test_unresolved_writer_keeps_lease_and_blocks_scheduling(h):
    h.launcher.unreaped_ops.add("probe")
    await h.submit(make_request(), 2)
    job = await h.run(make_request(), 1)
    assert job.state is JobState.PROBING and job.reason_code is ReasonCode.WRITER_UNRESOLVED
    holders = h.repo.lease_holders()
    assert len(holders) == 1
    with pytest.raises(RepositoryError) as caught:
        h.repo.release(holders[0][0], writers_reaped=False)
    assert caught.value.reason_code is ReasonCode.WRITER_UNRESOLVED
    assert h.repo.lease_holders() == holders
    assert await h.scheduler.run_once() is False
    assert h.repo.get_job(ident(2)).state is JobState.QUEUED  # the queue is retained, untouched
    assert ("run", "transcribe") not in h.log


@pytest.mark.asyncio
async def test_cancel_running_job_waits_for_reap(h):
    job = await h.submit(make_request(), 1)
    h.launcher.gate["transcribe"] = asyncio.Event()
    running = asyncio.create_task(h.scheduler.run_once())
    await h.launcher.entered.setdefault("transcribe", asyncio.Event()).wait()
    cancelled = await h.scheduler.cancel(job.job_id)
    assert await running is True
    final = h.repo.get_job(job.job_id)
    assert cancelled.state is JobState.CANCELLED and final.state is JobState.CANCELLED
    assert final.reason_code is ReasonCode.CANCELLED and final.next_action is None
    assert h.launcher.stops[0] is ReasonCode.CANCELLED
    assert h.repo.lease_holders() == ()
    assert h.repo.get_attempt(job.attempt_id if job.attempt_id else final.attempt_id).state is JobState.CANCELLED
    assert ("run", "transcribe") in h.log and ("export", tuple(T_SLOTS)) not in h.log
    assert not h.calls("publish")
    assert (await h.scheduler.cancel(job.job_id)).state is JobState.CANCELLED  # idempotent
    h.launcher.stopped = False
    h.launcher.gate.clear()
    done = await h.run(make_request(), n=2)
    assert done.state is JobState.COMPLETED
    with pytest.raises(TransitionError) as caught:
        await h.scheduler.cancel(done.job_id)
    assert caught.value.reason_code is ReasonCode.ALREADY_TERMINAL


@pytest.mark.asyncio
async def test_cancel_with_unreaped_writer_retains_lease(h):
    job = await h.submit(make_request(), 1)
    await h.submit(make_request(), 2)
    h.launcher.gate["transcribe"] = asyncio.Event()
    running = asyncio.create_task(h.scheduler.run_once())
    await h.launcher.entered.setdefault("transcribe", asyncio.Event()).wait()
    h.launcher.reaped = False  # the worker ignores terminate/kill
    result = await h.scheduler.cancel(job.job_id)
    await running
    stored = h.repo.get_job(job.job_id)
    assert result.state is JobState.TRANSCRIBING and stored.state is JobState.TRANSCRIBING
    assert stored.reason_code is ReasonCode.WRITER_UNRESOLVED and stored.next_action == "cancel"
    assert len(h.repo.lease_holders()) == 1
    assert await h.scheduler.run_once() is False
    h.launcher.reaped = True  # recovery path: writer finally verified gone
    assert await h.scheduler.cancel(job.job_id) is not None
    assert h.repo.get_job(job.job_id).state is JobState.CANCELLED
    assert h.repo.lease_holders() == ()


@pytest.mark.asyncio
async def test_cancel_queued_and_blocked_jobs_hold_no_lease(h):
    queued = await h.submit(make_request(), 1)
    assert (await h.scheduler.cancel(queued.job_id)).state is JobState.CANCELLED
    h.preflight.rule = lambda stage: ReasonCode.STORAGE_STALE
    blocked = await h.run(make_request(), 2)
    assert blocked.state is JobState.BLOCKED
    assert (await h.scheduler.cancel(blocked.job_id)).state is JobState.CANCELLED
    assert h.launcher.stops == [] and h.repo.lease_holders() == ()


@pytest.mark.asyncio
async def test_late_writer_conflict_aborts_without_mutation(h):
    job = await h.submit(make_request(), 1)
    h.launcher.gate["probe"] = asyncio.Event()
    running = asyncio.create_task(h.scheduler.run_once())
    await h.launcher.entered.setdefault("probe", asyncio.Event()).wait()
    stored = h.repo.get_job(job.job_id)
    other = replace(stored, updated_at=stamp(NOW + timedelta(hours=1)))
    h.repo.update_job(other, expected_version=h.repo.get_job_version(job.job_id))
    h.launcher.gate["probe"].set()
    await running
    assert h.repo.get_job(job.job_id) == other  # the late writer changed nothing
    assert ("run", "transcribe") not in h.log
    assert h.repo.lease_holders() == ()  # reaped, so the lease was given back


@pytest.mark.asyncio
async def test_persistence_failure_stops_scheduling(tmp_path):
    from podmuncher.repository import RepositoryError as _RE  # noqa: F401
    fired = {"n": 0}

    def hook(op):
        if op == "update_job":
            fired["n"] += 1
            if fired["n"] == 2:
                raise __import__("sqlite3").OperationalError("disk I/O error")

    harness = Harness(tmp_path, fault_hook=hook)
    try:
        await harness.submit(make_request())
        assert await harness.scheduler.run_once() is True
        assert harness.scheduler.fault is ReasonCode.STATE_WRITE_FAILED
        assert harness.repo.admission_blocked is ReasonCode.STATE_WRITE_FAILED
        assert await harness.scheduler.run_once() is False
        assert harness.repo.get_job(ident(1)).state is not JobState.COMPLETED
    finally:
        harness.close()


# -------------------------------------------------------------------- AutoOOM


@pytest.mark.asyncio
async def test_auto_oom_retries_exactly_one_lower_candidate(h, monkeypatch):
    released = []
    original = h.repo.transition_with_lease

    def spy(job, *, expected_version, acquire=None, release=None, closing=False):
        released.append((job.state, acquire is not None, release is not None))
        return original(job, expected_version=expected_version, acquire=acquire, release=release,
                        closing=closing)

    monkeypatch.setattr(h.repo, "transition_with_lease", spy)
    events = []
    admit, release = h.repo.admit_attempt, h.repo.release

    def admit_spy(job, attempt, **kwargs):
        events.append(("admit", kwargs["acquire"] is not None))
        return admit(job, attempt, **kwargs)

    def release_spy(op, **kwargs):
        events.append(("release",))
        return release(op, **kwargs)

    monkeypatch.setattr(h.repo, "admit_attempt", admit_spy)
    monkeypatch.setattr(h.repo, "release", release_spy)
    h.launcher.fail["transcribe"] = ReasonCode.MODEL_OOM
    h.launcher.fail_counts["transcribe"] = 1
    job = await h.run(make_request())
    assert job.state is JobState.COMPLETED
    assert h.calls("select") == [("select", None), ("select", "large")]
    assert h.models.prepared == ["large", "medium"]
    assert [e for e in h.log if e == ("run", "transcribe")] == [("run", "transcribe")] * 2
    assert h.calls("discard") == [("discard",)]
    attempt = h.repo.get_attempt(job.attempt_id)
    assert (attempt.ordinal, attempt.automatic_fallback_count) == (2, 1)
    # Full preflight again before the fallback attempt, with the lease released in between.
    assert h.log.count(("preflight", "checking")) == 2
    # Fallback: attempt admitted without a lease, the old lease released, then re-acquired.
    assert events[:3] == [("admit", True), ("admit", False), ("release",)]
    assert h.repo.lease_holders() == ()


@pytest.mark.asyncio
async def test_second_oom_is_terminal(h):
    h.launcher.fail["transcribe"] = ReasonCode.MODEL_OOM
    job = await h.run(make_request())
    assert job.state is JobState.FAILED and job.reason_code is ReasonCode.MODEL_OOM
    assert h.log.count(("run", "transcribe")) == 2
    assert h.calls("select") == [("select", None), ("select", "large")]
    assert h.repo.get_attempt(job.attempt_id).ordinal == 2 and h.repo.lease_holders() == ()


@pytest.mark.asyncio
async def test_explicit_model_never_falls_back(h):
    h.launcher.fail["transcribe"] = ReasonCode.MODEL_OOM
    job = await h.run(make_request(model="small"))
    assert job.state is JobState.FAILED and job.reason_code is ReasonCode.MODEL_OOM
    assert h.log.count(("run", "transcribe")) == 1 and len(h.calls("select")) == 1
    assert h.repo.get_attempt(job.attempt_id).ordinal == 1


@pytest.mark.asyncio
async def test_oom_fallback_blocked_by_full_preflight(h):
    h.launcher.fail["transcribe"] = ReasonCode.MODEL_OOM

    def rule(stage):
        return ReasonCode.STORAGE_STALE if h.log.count(("run", "transcribe")) and stage is Stage.CHECKING \
            else None

    h.preflight.rule = rule
    job = await h.run(make_request())
    assert job.state is JobState.BLOCKED and job.reason_code is ReasonCode.STORAGE_STALE
    assert h.log.count(("run", "transcribe")) == 1 and h.repo.lease_holders() == ()


# ------------------------------------------------------------ retry / clone


@pytest.mark.asyncio
async def test_export_retry_never_launches_asr(h):
    h.publisher.fail_publish[ArtifactSlot.TRANSCRIPT_SRT] = ReasonCode.PUBLICATION_FAILED
    first = await h.run(make_request())
    assert first.state is JobState.PARTIAL and len(first.artifacts) == 4
    transcribes, selects = h.log.count(("run", "transcribe")), len(h.calls("select"))
    h.publisher.fail_publish.clear()
    mark = len(h.log)
    attempt = await h.scheduler.retry(first.job_id, "retry-1")
    job = h.repo.get_job(first.job_id)
    assert job.state is JobState.COMPLETED and attempt.attempt_id == job.attempt_id
    assert attempt.ordinal == 2
    assert h.log.count(("run", "transcribe")) == transcribes and len(h.calls("select")) == selects
    assert [e for e in h.log[mark:] if e[0] == "publish"] == [("publish", "transcript_srt")]
    assert ("export", ("transcript_srt",)) in h.log[mark:]
    assert {a.slot.value for a in job.artifacts} == set(T_SLOTS)
    records = h.repo.get_attempt(job.attempt_id).stage_records
    assert [r.stage.value for r in records] == ["checking", "transcribing", "exporting"]
    assert h.repo.lease_holders() == ()


@pytest.mark.asyncio
async def test_retry_is_idempotent_per_key(h):
    h.launcher.fail["probe"] = ReasonCode.MEDIA_PROBE_FAILED
    first = await h.run(make_request())
    assert first.state is JobState.FAILED
    h.launcher.fail["probe"] = ReasonCode.MEDIA_PROBE_FAILED
    again = await h.scheduler.retry(first.job_id, "retry-key")
    launches = h.launcher.runs
    state = h.repo.get_job(first.job_id)
    replay = await h.scheduler.retry(first.job_id, "retry-key")
    assert replay.attempt_id == again.attempt_id and h.launcher.runs == launches
    # A restarted scheduler (no in-memory key) is stopped by the repository's key.
    restarted = h.make_scheduler()
    stored = await restarted.retry(first.job_id, "retry-key")
    assert stored.attempt_id == again.attempt_id and h.launcher.runs == launches
    assert h.repo.get_job(first.job_id) == state and h.repo.lease_holders() == ()
    # A different key is a genuinely new attempt.
    h.launcher.fail.clear()
    third = await h.scheduler.retry(first.job_id, "other-key")
    assert third.attempt_id != again.attempt_id and third.ordinal == 3
    assert h.repo.get_job(first.job_id).state is JobState.COMPLETED


@pytest.mark.asyncio
async def test_retry_requires_admission_and_a_terminal_job(h):
    first = await h.run(make_request())
    with pytest.raises(TransitionError) as caught:
        await h.scheduler.retry(first.job_id, "k")
    assert caught.value.reason_code is ReasonCode.ALREADY_TERMINAL
    h.launcher.fail["probe"] = ReasonCode.MEDIA_PROBE_FAILED
    failed = await h.run(make_request(), n=2)
    h.preflight.rule = lambda stage: ReasonCode.STORAGE_STALE
    runs = h.launcher.runs
    with pytest.raises(SchedulerError) as refused:
        await h.scheduler.retry(failed.job_id, "k2")
    assert refused.value.reason_code is ReasonCode.STORAGE_STALE
    assert h.repo.get_job(failed.job_id) == failed and h.launcher.runs == runs
    assert await h.scheduler.recheck(failed.job_id) is False
    h.preflight.rule = lambda stage: None
    assert await h.scheduler.recheck(failed.job_id) is True
    assert h.repo.get_job(failed.job_id) == failed  # recheck never creates an attempt or output
    assert h.launcher.runs == runs


@pytest.mark.asyncio
async def test_clone_reuses_only_proven_stage_outputs(h):
    original = await h.run(make_request(kind="remote"))
    assert original.state is JobState.COMPLETED
    old_records = {r.stage: r for r in h.repo.get_attempt(original.attempt_id).stage_records}
    mark = len(h.log)
    clone = await h.scheduler.clone(original.job_id, {"selection": {"profile": "quality"}})
    assert clone.job_id != original.job_id and clone.request_hash != original.request_hash
    assert h.repo.get_job(original.job_id) == original
    assert await h.scheduler.run_once() is True
    done = h.repo.get_job(clone.job_id)
    assert done.state is JobState.COMPLETED
    new = h.log[mark:]
    assert ("run", "download") not in new and ("run", "probe") not in new
    assert ("reuse", "downloading") in new and ("reuse", "probing") in new and ("run", "transcribe") in new
    records = {r.stage: r for r in h.repo.get_attempt(done.attempt_id).stage_records}
    for stage in (Stage.DOWNLOADING, Stage.PROBING):
        proof = records[stage]
        assert proof.verified_input_hashes == old_records[stage].verified_input_hashes
        assert proof.settings_hash == old_records[stage].settings_hash
        assert proof.stage_id != old_records[stage].stage_id and proof.attempt_id == done.attempt_id
    assert records[Stage.TRANSCRIBING].settings_hash != old_records[Stage.TRANSCRIBING].settings_hash


@pytest.mark.asyncio
async def test_clone_with_different_source_has_no_reuse_proof(h):
    original = await h.run(make_request(kind="remote"))
    clone = await h.scheduler.clone(original.job_id, {"source": {"kind": "direct",
                                                                 "input": "https://example.org/b.mp4"}})
    mark = len(h.log)
    await h.scheduler.run_once()
    assert h.repo.get_job(clone.job_id).state is JobState.COMPLETED
    assert ("run", "download") in h.log[mark:] and ("run", "probe") in h.log[mark:]
    assert not [e for e in h.log[mark:] if e[0] == "reuse"]


# ------------------------------------------------------------------- events


@pytest.mark.asyncio
async def test_late_and_foreign_events_are_ignored(h):
    def foreign(request):
        return WorkerEventRecord(1, ident(99), request.attempt_id, request.stage_id, 1, "progress",
                                 {"percent": 77})

    def late_stage(request):
        return WorkerEventRecord(1, request.job_id, request.attempt_id, ident(98, 3), 2, "progress",
                                 {"percent": 66})

    def own(request):
        return WorkerEventRecord(1, request.job_id, request.attempt_id, request.stage_id, 3, "progress",
                                 {"percent": 10})

    h.launcher.events["probe"] = [foreign, late_stage, own]
    seen = []
    original = h.scheduler._on_event

    def watch(run, stage_id, event):
        original(run, stage_id, event)
        seen.append(h.scheduler.progress(run.job.job_id))

    h.scheduler._on_event = watch
    job = await h.run(make_request())
    assert job.state is JobState.COMPLETED
    assert h.scheduler.dropped_events == 2
    assert seen[-1] == 10  # kept while the job ran; evicted with the final state (S9)
    assert h.scheduler.progress(job.job_id) is None
    # An event delivered after its stage ended is late as well.
    request, callback = h.launcher.callbacks[0]
    callback(WorkerEventRecord(1, request.job_id, request.attempt_id, request.stage_id, 9, "progress",
                               {"percent": 55}))
    assert h.scheduler.dropped_events == 3 and h.scheduler.progress(job.job_id) is None


@pytest.mark.asyncio
async def test_claims_for_another_stage_are_rejected(h):
    original = h.launcher.run

    async def forged(request, *, stage, on_event):
        result = await original(request, stage=stage, on_event=on_event)
        claim = replace(result.artifact_claims[0], stage_id=ident(77, 3))
        return replace(result, artifact_claims=(claim,))

    h.launcher.run = forged
    job = await h.run(make_request())
    assert job.state is JobState.FAILED and job.reason_code is ReasonCode.ARTIFACT_REJECTED
    assert h.repo.lease_holders() == ()


@pytest.mark.asyncio
async def test_unreaped_writer_blocks_claim_verification(h):
    # verify() is handed writers_reaped from the launcher, never a constant.
    seen = []
    original = h.publisher.verify

    def spy(job, attempt, assignment, claim, *, writers_reaped):
        seen.append(writers_reaped)
        return original(job, attempt, assignment, claim, writers_reaped=writers_reaped)

    h.publisher.verify = spy
    await h.run(make_request())
    assert seen == [True, True]


@pytest.mark.asyncio
async def test_progress_is_coalesced_to_one_hz_and_never_completes(h):
    def make(index):
        def event(request):
            h.clock.t = index * 0.1
            return WorkerEventRecord(1, request.job_id, request.attempt_id, request.stage_id, index + 1,
                                     "progress", {"percent": min(100, index * 5)})
        return event

    h.launcher.events["transcribe"] = [make(i) for i in range(21)]
    h.launcher.fail["transcribe"] = ReasonCode.WORKER_EXIT_UNEXPECTED
    seen = []
    original = h.scheduler._on_event

    def watch(run, stage_id, event):
        original(run, stage_id, event)
        seen.append(h.scheduler.progress(run.job.job_id))

    h.scheduler._on_event = watch
    job = await h.run(make_request())
    changes = [v for i, v in enumerate(seen) if i == 0 or v != seen[i - 1]]
    assert len(seen) == 21 and len(changes) <= 3  # 21 events over 2 s -> at most 3 snapshots
    assert job.state is JobState.FAILED  # 100 percent progress never implies completion


# ---------------------------------------------------------------- telemetry


@pytest.mark.asyncio
async def test_telemetry_spans_and_metrics_are_bounded(h):
    await h.run(make_request(video=True, kind="remote"), n=1)
    h.preflight.rule = lambda stage: ReasonCode.RAM_INSUFFICIENT
    await h.run(make_request(), n=2)
    h.preflight.rule = lambda stage: ReasonCode.STORAGE_STALE if stage is Stage.PROBING else None
    await h.run(make_request(), n=3)
    names = {e[1] for e in h.telemetry.events}
    assert names == {"job", "stage", "preflight"}
    for kind, name, attrs in h.telemetry.events:
        assert set(attrs) <= sch.SPAN_ATTRIBUTES
        for value in attrs.values():
            assert not any(token in value for token in ("/", "\\", "://", ".mp4", "out"))
    for name, attrs in h.telemetry.counts:
        assert name in {"job", "stage", "preflight"} and set(attrs) <= sch.METRIC_ATTRIBUTES
        assert "job_id" not in attrs  # ids are span-only, never metric labels
        for value in attrs.values():
            assert not any(token in value for token in ("/", "\\", "://", "."))
    preflights = [a for n, a in h.telemetry.counts if n == "preflight"]
    assert {a["result"] for a in preflights} == {"pass", "blocked"}
    reasons = {a["reason"] for a in preflights if "reason" in a}
    assert reasons <= {r.value for r in PreflightReason} and {"ram", "disk"} <= reasons
    assert all("reason" not in a for a in preflights if a["result"] == "pass")
    assert ("job", {"state": "completed", "source": "direct"}) in h.telemetry.counts
    assert ("job", {"state": "blocked", "source": "local"}) in h.telemetry.counts


@pytest.mark.asyncio
async def test_job_span_encloses_stage_and_preflight_spans(h):
    await h.run(make_request())
    depth, seen = 0, set()
    for kind, name, attrs in h.telemetry.events:
        if kind == "enter":
            if name != "job":
                assert depth >= 1  # every stage/preflight span lives inside the job span
                assert name != "stage" or depth == 1
                seen.add(name)
            else:
                assert depth == 0
            depth += 1
        else:
            depth -= 1
    assert depth == 0 and seen == {"stage", "preflight"}
    stage_spans = [a for k, n, a in h.telemetry.events if k == "enter" and n == "stage"]
    assert [a["stage"] for a in stage_spans] == ["checking", "preparing", "probing", "preparing",
                                                 "transcribing", "exporting"]
    assert all({"job_id", "attempt_id", "stage_id"} <= set(a) for a in stage_spans)


@pytest.mark.asyncio
async def test_stage_counters_carry_enum_outcomes(h):
    h.launcher.fail["probe"] = ReasonCode.MEDIA_PROBE_FAILED
    await h.run(make_request())
    stages = [a for n, a in h.telemetry.counts if n == "stage"]
    assert {"stage": "probing", "outcome": "error"} in stages
    assert {"stage": "checking", "outcome": "ok"} in stages


# ------------------------------------------------------------------ domains


def test_job_state_and_stage_reject_contract_negative_examples():
    registry = json.loads((ROOT / "Docs" / "contract-domains.json").read_text(encoding="utf-8"))
    for enum in (JobState, Stage):
        for value in registry["negative_examples"]:
            with pytest.raises(DomainValidationError) as caught:
                enum.parse(value)
            assert caught.value.reason_code is ReasonCode.INVALID_INPUT
    assert [s.value for s in JobState] == registry["job_state"]
    assert [s.value for s in Stage] == registry["stage"]


# ------------------------------------------------- P4-U4 checkpoint corrections


def _worker_job(request, tmp_path):
    """The real asr_worker job object; its envelope check is the pure gate under test."""
    return asr_worker._Job(request, None, scratch=tmp_path, input_root=tmp_path, model_root=tmp_path,
                           model_factory=None, decode=None, probe=None, memory_limit=None,
                           peak_rss=None, clock=time.monotonic)


def _transcribe_requests(h):
    return [req for req, _ in h.launcher.callbacks if req.operation == "transcribe"]


@pytest.mark.asyncio
@pytest.mark.parametrize("device", ["cpu", "cuda"])
async def test_transcribe_request_passes_the_asr_worker_envelope_for_auto_jobs(h, tmp_path, device):
    h.models.device = device
    job = await h.run(make_request(model="auto", device="auto"))
    assert job.state is JobState.COMPLETED
    (request,) = _transcribe_requests(h)
    assert (request.selection.device, request.selection.model) == (device, "large")
    assert _worker_job(request, tmp_path)._check_envelope() == device
    # Non-transcribe operations keep the user's selection untouched.
    assert h.launcher.callbacks[0][0].selection.device == "auto"


@pytest.mark.asyncio
async def test_fallback_transcribe_request_passes_the_asr_worker_envelope(h, tmp_path):
    h.launcher.fail["transcribe"] = ReasonCode.MODEL_OOM
    h.launcher.fail_counts["transcribe"] = 1
    job = await h.run(make_request())
    assert job.state is JobState.COMPLETED
    first, second = _transcribe_requests(h)
    assert (first.selection.model, second.selection.model) == ("large", "medium")
    for request in (first, second):
        assert _worker_job(request, tmp_path)._check_envelope() == "cpu"


@pytest.mark.asyncio
async def test_admission_is_one_transaction_and_a_key_collision_leaves_nothing(h, monkeypatch):
    h.launcher.fail["probe"] = ReasonCode.MEDIA_PROBE_FAILED
    first = await h.run(make_request(), n=1)
    second = await h.run(make_request(), n=2)
    assert first.state is second.state is JobState.FAILED
    h.launcher.fail.clear()
    await h.scheduler.retry(first.job_id, "shared-key")
    before = h.repo.get_job(second.job_id), h.repo.get_job_version(second.job_id)
    runs = h.launcher.runs
    # Force the race: the pre-check misses, the repository transaction sees the other job's key.
    monkeypatch.setattr(h.repo, "get_retry_attempt", lambda key: None)
    with pytest.raises(RepositoryError) as caught:
        await h.scheduler.retry(second.job_id, "shared-key")
    assert caught.value.reason_code is ReasonCode.CONFLICT
    assert (h.repo.get_job(second.job_id), h.repo.get_job_version(second.job_id)) == before
    assert h.repo.get_attempt(second.attempt_id).ordinal == 1
    assert h.repo.lease_holders() == () and h.launcher.runs == runs
    assert h.scheduler._busy is False


@pytest.mark.asyncio
async def test_initial_admission_collision_leaves_no_lease_or_attempt(h, monkeypatch):
    job = await h.submit(make_request(), 1)

    def refuse(*args, **kwargs):
        raise RepositoryError(ReasonCode.CONFLICT)

    monkeypatch.setattr(h.repo, "admit_attempt", refuse)
    assert await h.scheduler.run_once() is True
    stored = h.repo.get_job(job.job_id)
    assert stored.state is JobState.QUEUED and stored.attempt_id is None
    assert h.repo.lease_holders() == () and h.ops() == []


@pytest.mark.asyncio
async def test_retry_replay_wins_over_every_other_check_after_restart(h):
    h.publisher.fail_publish[ArtifactSlot.TRANSCRIPT_SRT] = ReasonCode.PUBLICATION_FAILED
    first = await h.run(make_request())
    assert first.state is JobState.PARTIAL
    h.publisher.fail_publish.clear()
    attempt = await h.scheduler.retry(first.job_id, "retry-1")
    completed = h.repo.get_job(first.job_id)
    assert completed.state is JobState.COMPLETED
    runs, publishes = h.launcher.runs, len(h.calls("publish"))
    restarted = h.make_scheduler()
    h.preflight.rule = lambda stage: ReasonCode.STORAGE_STALE  # admission would refuse a new attempt
    replay = await restarted.retry(first.job_id, "retry-1")  # job is completed: still a replay
    assert replay.attempt_id == attempt.attempt_id
    assert h.repo.get_job(first.job_id) == completed
    assert (h.launcher.runs, len(h.calls("publish")), h.repo.lease_holders()) == (runs, publishes, ())
    h.preflight.rule = lambda stage: None
    other = await h.run(make_request(), n=2)
    with pytest.raises(SchedulerError) as caught:  # the same key on another job is a conflict
        await restarted.retry(other.job_id, "retry-1")
    assert caught.value.reason_code is ReasonCode.CONFLICT


@pytest.mark.asyncio
async def test_cancel_stops_the_writer_even_when_persisting_the_intent_fails(h, monkeypatch):
    job = await h.submit(make_request(), 1)
    h.launcher.gate["transcribe"] = asyncio.Event()
    running = asyncio.create_task(h.scheduler.run_once())
    await h.launcher.entered.setdefault("transcribe", asyncio.Event()).wait()
    original = h.repo.update_job
    calls = []

    def failing(job_, **kwargs):
        calls.append(kwargs)
        raise RepositoryError(ReasonCode.STATE_WRITE_FAILED)

    monkeypatch.setattr(h.repo, "update_job", failing)
    with pytest.raises(RepositoryError) as caught:
        await h.scheduler.cancel(job.job_id)
    assert caught.value.reason_code is ReasonCode.STATE_WRITE_FAILED
    assert h.launcher.stops and h.launcher.stops[0] is ReasonCode.CANCELLED  # writer stopped anyway
    assert len(calls) == 2  # persisted once, retried once after the stop
    monkeypatch.setattr(h.repo, "update_job", original)
    await running
    assert h.repo.lease_holders() == ()


@pytest.mark.asyncio
async def test_cancel_releases_only_its_own_lease_operation(h):
    job = await h.submit(make_request(), 1)
    h.launcher.gate["transcribe"] = asyncio.Event()
    running = asyncio.create_task(h.scheduler.run_once())
    await h.launcher.entered.setdefault("transcribe", asyncio.Event()).wait()
    ((own, _),) = h.repo.lease_holders()
    assert h.scheduler._ops[job.job_id] == own
    h.scheduler._ops[job.job_id] = "someone-elses-operation"  # not a held lease of this job
    final = await h.scheduler.cancel(job.job_id)
    assert final.state is JobState.CANCELLED
    assert h.repo.lease_holders() == ((own, True),)  # the holder was not released blindly
    await running
    assert h.repo.lease_holders() == ()  # the run's own abandon path gives its lease back


@pytest.mark.asyncio
async def test_source_identity_is_content_not_the_path_string(h):
    request = make_request(kind="local")
    job = await h.run(request)
    identity = hashlib.sha256(b"content-v1").hexdigest()
    path_hash = hashlib.sha256(request.source.input.encode()).hexdigest()
    for record in h.repo.get_attempt(job.attempt_id).stage_records:
        assert ("source", identity) in record.verified_input_hashes
        assert ("source", path_hash) not in record.verified_input_hashes
    # Same bytes: the clone reuses the probe.  Changed bytes behind the same path: it must not.
    clone = await h.scheduler.clone(job.job_id)
    mark = len(h.log)
    await h.scheduler.run_once()
    assert ("reuse", "probing") in h.log[mark:] and ("run", "probe") not in h.log[mark:]
    h.publisher.content[request.source.input] = "content-v2"
    changed = await h.scheduler.clone(job.job_id)
    mark = len(h.log)
    await h.scheduler.run_once()
    assert h.repo.get_job(clone.job_id).state is JobState.COMPLETED
    assert h.repo.get_job(changed.job_id).state is JobState.COMPLETED
    assert ("run", "probe") in h.log[mark:] and ("reuse", "probing") not in h.log[mark:]


@pytest.mark.asyncio
async def test_remote_records_the_verified_download_identity(h):
    job = await h.run(make_request(kind="remote"))
    identity = hashlib.sha256(b"content-v1").hexdigest()
    records = {r.stage: r for r in h.repo.get_attempt(job.attempt_id).stage_records}
    assert records[Stage.DOWNLOADING].verified_input_hashes == (("source", identity),)
    assert records[Stage.PROBING].verified_input_hashes == (("source", identity),)
    assert hashlib.sha256(REMOTE.encode()).hexdigest() != identity


@pytest.mark.asyncio
async def test_published_artifacts_are_recorded_and_manifested_with_all_of_them(h):
    job = await h.run(make_request(video=True, kind="remote"))
    assert job.state is JobState.COMPLETED
    assert {a.artifact_id for a in h.repo.list_artifacts(job.job_id)} == {a.artifact_id for a in job.artifacts}
    assert len(h.repo.list_artifacts(job.job_id)) == 6
    # One manifest after the video and one after every export; each lists everything so far.
    assert [m[2] for m in h.publisher.manifests] == [("original_video",)] + [
        ("original_video",) + tuple(T_SLOTS[: n + 1]) for n in range(len(T_SLOTS))]
    assert all(m[0] == job.job_id and m[1] == job.attempt_id for m in h.publisher.manifests)


@pytest.mark.asyncio
async def test_record_artifact_conflict_on_replay_is_tolerated(h, monkeypatch):
    def duplicate(artifact):
        raise RepositoryError(ReasonCode.CONFLICT)

    monkeypatch.setattr(h.repo, "record_artifact", duplicate)
    job = await h.run(make_request())
    assert job.state is JobState.COMPLETED


@pytest.mark.asyncio
async def test_manifest_failure_is_a_publication_failure(h):
    h.publisher.fail_manifest = ReasonCode.DESTINATION_CONFLICT
    job = await h.run(make_request(video=True, kind="remote"))
    assert job.state is JobState.PARTIAL and job.reason_code is ReasonCode.DESTINATION_CONFLICT
    assert [a.slot for a in job.artifacts] == [ArtifactSlot.ORIGINAL_VIDEO]
    assert ("run", "transcribe") not in h.log and h.repo.lease_holders() == ()


@pytest.mark.asyncio
async def test_per_job_memory_is_evicted_at_a_final_state(h):
    original = await h.run(make_request(kind="remote"))
    clone = await h.scheduler.clone(original.job_id, {"selection": {"profile": "quality"}})
    assert clone.job_id in h.scheduler._reuse
    await h.scheduler.run_once()
    assert h.repo.get_job(clone.job_id).state is JobState.COMPLETED
    assert not h.scheduler._reuse and not h.scheduler._progress and not h.scheduler._ops


@pytest.mark.asyncio
async def test_stage_measurement_is_emitted_for_cancel_and_terminal_probe_exits(h):
    job = await h.submit(make_request(), 1)  # cancel mid-stage
    h.launcher.gate["probe"] = asyncio.Event()
    running = asyncio.create_task(h.scheduler.run_once())
    await h.launcher.entered.setdefault("probe", asyncio.Event()).wait()
    await h.scheduler.cancel(job.job_id)
    await running
    stages = [a for n, a in h.telemetry.counts if n == "stage"]
    assert {"stage": "probing", "outcome": "cancelled"} in stages
    # Terminal probe (no audio, transcript only): the probe stage itself finished ok.
    h.launcher.stopped, h.launcher.gate = False, {}
    h.telemetry.counts.clear()
    h.publisher.facts = MediaFacts(False, True, 12.0)
    done = await h.run(make_request(), n=2)
    assert done.state is JobState.FAILED and done.reason_code is ReasonCode.NO_AUDIO
    stages = [a for n, a in h.telemetry.counts if n == "stage"]
    assert {"stage": "probing", "outcome": "ok"} in stages


@pytest.mark.asyncio
async def test_real_telemetry_hub_records_job_and_stage_points(tmp_path):
    harness = Harness(tmp_path)
    hub = telemetry.make_telemetry(tmp_path / "diag", config.Telemetry(otlp_enabled=False, otlp_endpoint=""),
                                   clock=harness.clock, background=False)
    try:
        harness.telemetry = hub.scheduler_telemetry()
        harness.scheduler = harness.make_scheduler()
        done = await harness.run(make_request(), n=1)
        harness.launcher.fail["probe"] = ReasonCode.MEDIA_PROBE_FAILED
        failed = await harness.run(make_request(kind="remote"), n=2)
        assert (done.state, failed.state) == (JobState.COMPLETED, JobState.FAILED)
        jobs = hub.metric_points("podmuncher.jobs")
        assert ({"outcome": "completed", "source": "local"}, 1) in jobs
        assert ({"outcome": "failed", "source": "direct"}, 1) in jobs
        durations = {(a["stage"], a["outcome"]) for a, _ in hub.metric_points("podmuncher.stage.duration")}
        assert {("checking", "ok"), ("probing", "ok"), ("transcribing", "ok"), ("exporting", "ok"),
                ("probing", "error")} <= durations
        assert hub.dropped["metric"] == 0
    finally:
        hub.shutdown()
        harness.close()


# -------------------------------------------------------------------- quiesce


@pytest.mark.asyncio
async def test_quiesce_when_idle_refuses_new_work_and_is_idempotent(h):
    await h.submit(make_request())
    assert await h.scheduler.quiesce(timeout=1) is True
    assert await h.scheduler.quiesce(timeout=1) is True
    assert await h.scheduler.run_once() is False
    plan = AdmissionPlan(Measurement(True, "fake", FieldStatus.OK, stamp(NOW), None), (), ())

    async def action():
        return True

    with pytest.raises(SchedulerError) as caught:
        await h.scheduler.run_setup("setup", plan, action)
    assert caught.value.reason_code is ReasonCode.SERVICE_STOPPED
    with pytest.raises(SchedulerError):
        await h.scheduler.retry(ident(1), "k")
    assert h.launcher.runs == 0 and h.repo.get_job(ident(1)).state is JobState.QUEUED


@pytest.mark.asyncio
async def test_quiesce_stops_the_running_writer_and_waits_for_the_task(h):
    job = await h.submit(make_request())
    h.launcher.gate["probe"] = asyncio.Event()
    running = asyncio.create_task(h.scheduler.run_once())
    await h.launcher.entered.setdefault("probe", asyncio.Event()).wait()
    assert await h.scheduler.quiesce(timeout=2) is True
    assert running.done() and await running is True
    assert h.launcher.stops and h.repo.lease_holders() == ()
    # Left for recovery: neither failed nor completed, and nothing further was started.
    assert h.repo.get_job(job.job_id).state is JobState.PROBING
    assert ("run", "transcribe") not in h.log
    assert await h.scheduler.quiesce(timeout=1) is True


@pytest.mark.asyncio
async def test_quiesce_reports_false_for_an_unreaped_writer(h):
    await h.submit(make_request())
    h.launcher.gate["probe"] = asyncio.Event()
    running = asyncio.create_task(h.scheduler.run_once())
    await h.launcher.entered.setdefault("probe", asyncio.Event()).wait()
    h.launcher.reaped = False
    assert await h.scheduler.quiesce(timeout=1) is False  # writer never reaped
    await running
    h.launcher.reaped = True
    assert await h.scheduler.quiesce(timeout=1) is True  # verified gone later


@pytest.mark.asyncio
async def test_quiesce_stops_the_setup_action_and_times_out_when_stop_hangs(h):
    plan = AdmissionPlan(Measurement(True, "fake", FieldStatus.OK, stamp(NOW), None), (), ())
    gate, started, stopped = asyncio.Event(), asyncio.Event(), []

    async def action():
        started.set()
        await gate.wait()
        return True

    async def stop():
        stopped.append(True)
        gate.set()
        return True

    setup = asyncio.create_task(h.scheduler.run_setup("setup", plan, action, stop=stop))
    await started.wait()
    assert await h.scheduler.quiesce(timeout=1) is True
    assert stopped == [True] and setup.done()
    assert h.repo.lease_holders() == ()

    async def hang(reason):
        await asyncio.Event().wait()

    other = h.make_scheduler()
    other._busy = True
    h.launcher.stop = hang
    assert await other.quiesce(timeout=0.05) is False


@pytest.mark.asyncio
async def test_quiesce_is_false_when_the_setup_stop_reports_an_unreaped_writer(h):
    plan = AdmissionPlan(Measurement(True, "fake", FieldStatus.OK, stamp(NOW), None), (), ())
    gate, started = asyncio.Event(), asyncio.Event()

    async def action():
        started.set()
        await gate.wait()
        return False  # the setup writer never reaped

    async def stop():
        gate.set()
        return False

    setup = asyncio.create_task(h.scheduler.run_setup("setup", plan, action, stop=stop))
    await started.wait()
    assert h.launcher.writers_reaped is True  # the launcher alone would have said "reaped"
    assert await h.scheduler.quiesce(timeout=1) is False
    await setup
    assert h.repo.lease_holders() == (("setup", True),)  # the lease stays held
    assert await h.scheduler.quiesce(timeout=1) is False


# ------------------------------------------------- closing writes, reuse, export retry


@pytest.mark.asyncio
async def test_terminal_commit_persists_and_releases_above_the_size_bound(h, monkeypatch):
    original = h.repo.record_artifact

    def shrink(artifact):
        original(artifact)
        if artifact.slot.value == T_SLOTS[-1]:  # every export is out and listed: only the closing commit remains
            monkeypatch.setattr(repository, "DB_BOUND", 1)

    monkeypatch.setattr(h.repo, "record_artifact", shrink)
    job = await h.run(make_request())
    assert job.state is JobState.COMPLETED and h.repo.db_bytes() > repository.DB_BOUND
    assert h.repo.get_attempt(job.attempt_id).state is JobState.COMPLETED
    assert h.repo.lease_holders() == ()


@pytest.mark.asyncio
async def test_blocked_commit_persists_above_the_size_bound(h, monkeypatch):
    await h.submit(make_request())
    monkeypatch.setattr(repository, "DB_BOUND", 1)
    h.preflight.rule = lambda stage: ReasonCode.STORAGE_STALE
    assert await h.scheduler.run_once() is True
    stored = h.repo.get_job(ident(1))
    assert stored.state is JobState.BLOCKED and stored.reason_code is ReasonCode.STORAGE_STALE
    assert h.repo.lease_holders() == ()


@pytest.mark.asyncio
async def test_cancel_after_reap_persists_and_releases_above_the_size_bound(h, monkeypatch):
    job = await h.submit(make_request(), 1)
    h.launcher.gate["transcribe"] = asyncio.Event()
    running = asyncio.create_task(h.scheduler.run_once())
    await h.launcher.entered.setdefault("transcribe", asyncio.Event()).wait()
    stop = h.launcher.stop

    async def shrinking_stop(reason):
        reaped = await stop(reason)
        monkeypatch.setattr(repository, "DB_BOUND", 1)  # the intent is stored; the closing write is not
        return reaped

    h.launcher.stop = shrinking_stop
    final = await h.scheduler.cancel(job.job_id)
    assert final.state is JobState.CANCELLED
    stored = h.repo.get_job(job.job_id)
    assert stored.state is JobState.CANCELLED and h.repo.get_attempt(stored.attempt_id).state is JobState.CANCELLED
    assert h.repo.db_bytes() > repository.DB_BOUND and h.repo.lease_holders() == ()
    await running


@pytest.mark.asyncio
async def test_clone_whose_redownload_changed_the_bytes_does_not_reuse_the_probe(h):
    original = await h.run(make_request(kind="remote"))
    clone = await h.scheduler.clone(original.job_id, {"outputs": {"video": True}})  # download settings differ
    assert Stage.PROBING in h.scheduler._reuse[clone.job_id] and Stage.DOWNLOADING not in h.scheduler._reuse[clone.job_id]
    h.publisher.content[REMOTE] = "content-v2"  # the re-download yields different bytes
    mark = len(h.log)
    await h.scheduler.run_once()
    assert h.repo.get_job(clone.job_id).state is JobState.COMPLETED
    assert ("run", "download") in h.log[mark:] and ("run", "probe") in h.log[mark:]
    assert ("reuse", "probing") not in h.log[mark:]
    records = {r.stage: r for r in h.repo.get_attempt(h.repo.get_job(clone.job_id).attempt_id).stage_records}
    new_identity = hashlib.sha256(b"content-v2").hexdigest()
    assert records[Stage.PROBING].verified_input_hashes == (("source", new_identity),)


@pytest.mark.asyncio
async def test_clone_whose_redownload_kept_the_bytes_still_reuses_the_probe(h):
    original = await h.run(make_request(kind="remote"))
    clone = await h.scheduler.clone(original.job_id, {"outputs": {"video": True}})
    mark = len(h.log)
    await h.scheduler.run_once()
    assert h.repo.get_job(clone.job_id).state is JobState.COMPLETED
    assert ("run", "download") in h.log[mark:] and ("reuse", "probing") in h.log[mark:]
    assert ("run", "probe") not in h.log[mark:]


@pytest.mark.asyncio
async def test_export_retry_does_not_reopen_a_deleted_source(h, monkeypatch):
    h.publisher.fail_publish[ArtifactSlot.TRANSCRIPT_SRT] = ReasonCode.PUBLICATION_FAILED
    first = await h.run(make_request())
    assert first.state is JobState.PARTIAL
    h.publisher.fail_publish.clear()
    reads = []

    def deleted(job):
        reads.append(job.job_id)
        raise ArtifactError(ReasonCode.INVALID_INPUT)

    monkeypatch.setattr(h.publisher, "source_identity", deleted)
    mark = len(h.log)
    await h.scheduler.retry(first.job_id, "retry-gone")
    job = h.repo.get_job(first.job_id)
    assert job.state is JobState.COMPLETED and reads == []
    assert [e for e in h.log[mark:] if e[0] == "publish"] == [("publish", "transcript_srt")]
    assert h.repo.lease_holders() == ()


@pytest.mark.asyncio
async def test_manifest_lists_the_first_export_when_the_second_fails_on_storage(h):
    h.publisher.fail_publish[ArtifactSlot.TRANSCRIPT_READABLE_TXT] = ReasonCode.STORAGE_INSUFFICIENT
    job = await h.run(make_request())
    assert job.state is JobState.PARTIAL
    assert [a.slot.value for a in job.artifacts] == [T_SLOTS[0]]
    assert h.publisher.manifests and h.publisher.manifests[-1][2] == (T_SLOTS[0],)
    assert h.repo.lease_holders() == ()


# ---------------------------------------------- sidecar before the terminal state


def _observe_closing(h):
    """Record every completed/partial job write with the sidecars that existed at that moment.

    The repository is the client's view: whatever it holds when such a write lands is what a
    client can observe, so the sidecar must already be among ``h.publisher.manifests``.
    """
    seen = []
    for name in ("update_job", "transition_with_lease"):
        original = getattr(h.repo, name)

        def wrapped(job, *args, _original=original, **kwargs):
            if job.state in (JobState.COMPLETED, JobState.PARTIAL):
                seen.append((job.state, job.attempt_id, tuple(h.publisher.manifests)))
            return _original(job, *args, **kwargs)

        setattr(h.repo, name, wrapped)
    return seen


def _observe_manifests(h):
    """Record the stored job state at the moment every sidecar is written."""
    states = []
    original = h.publisher.write_manifest

    def write(job_id, attempt_id, published, provenance):
        states.append(h.repo.get_job(job_id).state)
        return original(job_id, attempt_id, published, provenance)

    h.publisher.write_manifest = write
    return states


def _fail_manifest_when(h, newest, reason):
    """Fail the sidecar write whose newest listed artifact fills ``newest``."""
    original = h.publisher.write_manifest

    def write(job_id, attempt_id, published, provenance):
        if published and published[-1].slot.value == newest:
            h.log.append(("manifest", len(published)))
            raise ArtifactError(reason)
        return original(job_id, attempt_id, published, provenance)

    h.publisher.write_manifest = write


TERMINAL_PATHS = [
    # (case, request, has_audio, state, reason, sidecar slots)
    ("video_only", {"transcript": False, "video": True}, False, JobState.COMPLETED, ReasonCode.OK,
     ("original_video",)),
    ("combined_silent", {"video": True}, False, JobState.PARTIAL, ReasonCode.NO_AUDIO, ("original_video",)),
    ("transcript_export", {}, True, JobState.COMPLETED, ReasonCode.OK, tuple(T_SLOTS)),
    ("combined_export", {"video": True, "kind": "remote"}, True, JobState.COMPLETED, ReasonCode.OK,
     ("original_video", *T_SLOTS)),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("case,request_kwargs,audio,state,reason,slots", TERMINAL_PATHS,
                         ids=[c[0] for c in TERMINAL_PATHS])
async def test_sidecar_exists_when_the_terminal_state_becomes_visible(h, case, request_kwargs, audio,
                                                                      state, reason, slots):
    """completed/partial is committed only after the sidecar lists every claimed artifact (spec.md:140)."""
    h.publisher.facts = MediaFacts(audio, True, 12.0)
    closing, states = _observe_closing(h), _observe_manifests(h)
    job = await h.run(make_request(**request_kwargs))
    assert (job.state, job.reason_code) == (state, reason)
    assert tuple(a.slot.value for a in job.artifacts) == slots
    assert states and all(s not in (JobState.COMPLETED, JobState.PARTIAL) for s in states)
    ((seen, attempt_id, manifests),) = closing  # exactly one closing write, with the sidecar in place
    assert seen is state and attempt_id == job.attempt_id
    assert manifests[-1] == (job.job_id, job.attempt_id, slots)
    assert h.repo.lease_holders() == () and h.repo.get_attempt(job.attempt_id).state is state


@pytest.mark.asyncio
async def test_retry_over_an_earlier_video_lists_it_before_closing_partial(h):
    """A full retry whose probe now closes the job (silent media) writes the sidecar first."""
    h.launcher.fail["transcribe"] = ReasonCode.WORKER_EXIT_UNEXPECTED
    first = await h.run(make_request(video=True))
    assert first.state is JobState.PARTIAL and [a.slot for a in first.artifacts] == [ArtifactSlot.ORIGINAL_VIDEO]
    h.launcher.fail.clear()
    h.publisher.facts = MediaFacts(False, True, 12.0)
    closing, states = _observe_closing(h), _observe_manifests(h)
    mark = len(h.log)
    attempt = await h.scheduler.retry(first.job_id, "retry-silent")
    job = h.repo.get_job(first.job_id)
    assert (job.state, job.reason_code) == (JobState.PARTIAL, ReasonCode.NO_AUDIO)
    assert ("run", "transcribe") not in h.log[mark:] and ("publish", "original_video") not in h.log[mark:]
    assert states == [JobState.PROBING]  # written by the retry, before its closing probe commit
    ((seen, attempt_id, manifests),) = closing
    assert seen is JobState.PARTIAL and attempt_id == attempt.attempt_id
    assert manifests[-1] == (job.job_id, attempt.attempt_id, ("original_video",))
    assert h.repo.lease_holders() == ()


def _claim_every_export(h, job):
    """Legacy shape: a partial job that already claims every export without a current sidecar."""
    missing = set(T_SLOTS) - {a.slot.value for a in job.artifacts}
    extra = tuple(Artifact(ident(90 + n, 5), ArtifactSlot.parse(slot), FORMATS[ArtifactSlot.parse(slot)], 5, SHA,
                           True, job.attempt_id, None) for n, slot in enumerate(sorted(missing)))
    stale = replace(job, artifacts=job.artifacts + extra)
    h.repo.update_job(stale, expected_version=h.repo.get_job_version(job.job_id))
    return h.repo.get_job(job.job_id)


@pytest.mark.asyncio
async def test_export_retry_with_nothing_left_refreshes_the_sidecar_before_completed(h):
    h.publisher.fail_publish[ArtifactSlot.TRANSCRIPT_SRT] = ReasonCode.PUBLICATION_FAILED
    first = _claim_every_export(h, await h.run(make_request()))
    assert first.state is JobState.PARTIAL and {a.slot.value for a in first.artifacts} == set(T_SLOTS)
    h.publisher.fail_publish.clear()
    closing, states = _observe_closing(h), _observe_manifests(h)
    mark = len(h.log)
    attempt = await h.scheduler.retry(first.job_id, "retry-listed")
    job = h.repo.get_job(first.job_id)
    assert job.state is JobState.COMPLETED and attempt.ordinal == 2
    assert ("publish", "transcript_srt") not in h.log[mark:]  # nothing was left to publish
    assert states == [JobState.EXPORTING]
    ((seen, attempt_id, manifests),) = closing
    assert seen is JobState.COMPLETED and attempt_id == attempt.attempt_id
    assert manifests[-1][:2] == (job.job_id, attempt.attempt_id) and set(manifests[-1][2]) == set(T_SLOTS)


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", [ReasonCode.DESTINATION_CONFLICT, ReasonCode.STORAGE_INSUFFICIENT])
async def test_sidecar_failure_with_every_export_claimed_blocks_instead_of_completing(h, reason):
    """Once every requested artifact is claimed any closing event reduces to completed: block instead."""
    h.publisher.fail_publish[ArtifactSlot.TRANSCRIPT_SRT] = ReasonCode.PUBLICATION_FAILED
    first = _claim_every_export(h, await h.run(make_request()))
    h.publisher.fail_publish.clear()
    h.publisher.fail_manifest = reason
    closing = _observe_closing(h)
    attempt = await h.scheduler.retry(first.job_id, "retry-unlisted")
    job = h.repo.get_job(first.job_id)
    assert (job.state, job.reason_code, job.next_action) == (JobState.BLOCKED, reason, sch.RECHECK_ACTION)
    assert closing == [] and h.repo.get_attempt(attempt.attempt_id).state is JobState.BLOCKED
    assert h.launcher.stops[-1] is reason and h.repo.lease_holders() == ()
    h.publisher.fail_manifest = None  # the sidecar can be written again: the retry completes
    await h.scheduler.retry(first.job_id, "retry-listed")
    job = h.repo.get_job(first.job_id)
    assert job.state is JobState.COMPLETED and set(h.publisher.manifests[-1][2]) == set(T_SLOTS)
    ((seen, _, manifests),) = closing
    assert seen is JobState.COMPLETED and manifests[-1][1] == job.attempt_id


@pytest.mark.asyncio
@pytest.mark.parametrize("reason,state", [(ReasonCode.PUBLICATION_FAILED, JobState.FAILED),
                                          (ReasonCode.STORAGE_INSUFFICIENT, JobState.BLOCKED)])
@pytest.mark.parametrize("transcript", [False, True], ids=["video_only", "combined_silent"])
async def test_sidecar_failure_on_the_closing_video_publication_never_claims_it(h, transcript, reason, state):
    """The video closes the job: a failed sidecar leaves it unclaimed (failed, or blocked on storage)."""
    h.publisher.facts = MediaFacts(False, True, 12.0)
    h.publisher.fail_manifest = reason
    closing = _observe_closing(h)
    job = await h.run(make_request(transcript=transcript, video=True))
    assert (job.state, job.reason_code) == (state, reason)
    assert job.artifacts == () and h.repo.list_artifacts(job.job_id) == []
    assert closing == [] and h.publisher.manifests == []
    assert ("publish", "original_video") in h.log and ("run", "transcribe") not in h.log
    assert h.repo.get_attempt(job.attempt_id).state is state and h.repo.lease_holders() == ()
    if state is JobState.BLOCKED:
        assert h.launcher.stops == [reason] and job.next_action == sch.RECHECK_ACTION


EXPORT_SIDECAR_FAILURES = [
    # (case, reason, export whose sidecar fails, state, claimed exports)
    ("last_conflict", ReasonCode.DESTINATION_CONFLICT, T_SLOTS[-1], JobState.PARTIAL, T_SLOTS[:-1]),
    ("first_conflict", ReasonCode.DESTINATION_CONFLICT, T_SLOTS[0], JobState.PARTIAL, T_SLOTS[1:]),
    ("last_storage", ReasonCode.STORAGE_INSUFFICIENT, T_SLOTS[-1], JobState.PARTIAL, T_SLOTS[:-1]),
    ("first_storage", ReasonCode.STORAGE_INSUFFICIENT, T_SLOTS[0], JobState.BLOCKED, []),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("case,reason,newest,state,claimed", EXPORT_SIDECAR_FAILURES,
                         ids=[c[0] for c in EXPORT_SIDECAR_FAILURES])
async def test_sidecar_failure_on_an_export_never_claims_it_nor_completes(h, case, reason, newest, state, claimed):
    """An export is claimed only once listed; a storage reason halts, any other leaves the stage partial."""
    _fail_manifest_when(h, newest, reason)
    closing = _observe_closing(h)
    job = await h.run(make_request())
    assert (job.state, job.reason_code) == (state, reason)
    assert [a.slot.value for a in job.artifacts] == claimed
    assert [a.slot.value for a in h.repo.list_artifacts(job.job_id)] == claimed
    assert ("publish", newest) in h.log  # the file was written but, unlisted, is not claimed
    if claimed:
        assert h.publisher.manifests[-1][2] == tuple(claimed)
        ((seen, _, manifests),) = closing
        assert seen is JobState.PARTIAL and manifests[-1][2] == tuple(claimed)
    else:
        assert closing == [] and h.publisher.manifests == []
    if reason in sch._STORAGE_REASONS:
        assert h.launcher.stops == [reason]
    assert h.repo.get_attempt(job.attempt_id).state is state and h.repo.lease_holders() == ()


# ------------------------------------------------------- preflight off the loop


def _record_threads(h):
    """Wrap the fake preflight to record (stage, thread id) for every call."""
    calls = []
    original = h.preflight.plan

    def plan(job, stage):
        calls.append((stage.value, threading.get_ident()))
        return original(job, stage)

    h.preflight.plan = plan
    return calls


def _gate_preflight(h, when):
    """Block the preflight on a threading.Event while ``when(stage)`` holds."""
    entered, release = threading.Event(), threading.Event()
    original = h.preflight.plan

    def plan(job, stage):
        if when(stage) and not release.is_set():
            entered.set()
            assert release.wait(10), "test never released the blocked preflight"
        return original(job, stage)

    h.preflight.plan = plan
    return entered, release


@pytest.mark.asyncio
async def test_preflight_never_runs_on_the_event_loop_at_any_site(h):
    loop_thread = threading.get_ident()
    calls = _record_threads(h)
    # Queued admission, every stage boundary, the video publication and each export publication.
    job = await h.run(make_request(video=True, kind="remote"))
    assert job.state is JobState.COMPLETED
    stages = [stage for stage, _ in calls]
    assert stages[0] == "checking"  # queued admission
    assert {"preparing", "downloading", "transcribing"} <= set(stages)  # stage boundaries
    assert stages.count("probing") == 2  # probing boundary + video publication
    assert stages.count("exporting") == 1 + len(T_SLOTS)  # exporting boundary + one per export
    # AutoOOM fallback: the full preflight before the lower candidate.
    h.launcher.fail["transcribe"] = ReasonCode.MODEL_OOM
    h.launcher.fail_counts["transcribe"] = 1
    mark = len(calls)
    oom = await h.run(make_request(), n=2)
    assert oom.state is JobState.COMPLETED
    assert [s for s, _ in calls[mark:]].count("checking") == 2  # admission + fallback
    # Retry admission.
    h.publisher.fail_publish[ArtifactSlot.TRANSCRIPT_SRT] = ReasonCode.PUBLICATION_FAILED
    partial = await h.run(make_request(), n=3)
    assert partial.state is JobState.PARTIAL
    h.publisher.fail_publish.clear()
    mark = len(calls)
    await h.scheduler.retry(partial.job_id, "retry-offloop")
    assert calls[mark][0] == "checking"  # the retry's own admission preflight
    # Recheck.
    mark = len(calls)
    assert await h.scheduler.recheck(partial.job_id) is True
    assert [s for s, _ in calls[mark:]] == ["checking"]
    assert calls and all(thread != loop_thread for _, thread in calls)
    assert len(calls) == len(h.calls("preflight"))


@pytest.mark.asyncio
async def test_event_loop_stays_responsive_while_the_preflight_blocks(h):
    entered, release = _gate_preflight(h, lambda stage: stage is Stage.PROBING)
    job = await h.submit(make_request(), 1)
    running = asyncio.create_task(h.scheduler.run_once())
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        beats = 0
        for _ in range(50):  # the loop keeps scheduling other coroutines while the sample blocks
            await asyncio.sleep(0)
            beats += 1
        assert beats == 50 and not running.done()
        assert h.scheduler.progress(job.job_id) is None  # a sync control read is served too
    finally:
        release.set()
    assert await running is True
    assert h.repo.get_job(job.job_id).state is JobState.COMPLETED
    assert h.repo.lease_holders() == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("refused", [False, True])
async def test_cancel_during_a_stage_boundary_preflight_starts_no_new_stage(h, refused):
    if refused:  # a refused plan must not halt (stop with a storage reason) a cancelled job
        h.preflight.rule = lambda stage: ReasonCode.STORAGE_STALE if stage is Stage.PROBING else None
    entered, release = _gate_preflight(h, lambda stage: stage is Stage.PROBING)
    released = _record_releases(h)
    job = await h.submit(make_request(), 1)
    running = asyncio.create_task(h.scheduler.run_once())
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        ((op, _),) = h.repo.lease_holders()
        cancelled = await h.scheduler.cancel(job.job_id)  # completes while the preflight is blocked
        assert cancelled.state is JobState.CANCELLED and not running.done()
    finally:
        release.set()
    assert await running is True
    final = h.repo.get_job(job.job_id)
    assert final.state is JobState.CANCELLED and final.reason_code is ReasonCode.CANCELLED
    attempt = h.repo.get_attempt(final.attempt_id)
    assert attempt.state is JobState.CANCELLED
    assert Stage.PROBING not in {r.stage for r in attempt.stage_records}  # no new stage started
    assert ("run", "probe") not in h.log and not h.calls("publish")
    assert h.repo.lease_holders() == ()
    # Exactly once: the cancel's closing write releases the lease, and the run's abandon path
    # must not release the same operation a second time (repository idempotency would mask it).
    assert released == [op]
    assert set(h.launcher.stops) == {ReasonCode.CANCELLED}
    assert await h.scheduler.run_once() is False  # nothing queued; the scheduler is not wedged


@pytest.mark.asyncio
@pytest.mark.parametrize("refused", [False, True])
async def test_cancel_during_the_admission_preflight_admits_nothing(h, refused):
    if refused:  # the blocked write must not race the cancel's write (no CONFLICT escapes)
        h.preflight.rule = lambda stage: ReasonCode.STORAGE_STALE
    entered, release = _gate_preflight(h, lambda stage: stage is Stage.CHECKING)
    job = await h.submit(make_request(), 1)
    running = asyncio.create_task(h.scheduler.run_once())
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        assert (await h.scheduler.cancel(job.job_id)).state is JobState.CANCELLED
    finally:
        release.set()
    assert await running is True
    final = h.repo.get_job(job.job_id)
    assert final.state is JobState.CANCELLED and final.attempt_id is None
    assert h.repo.lease_holders() == () and h.launcher.runs == 0


# ------------------------------------------------ races at the off-loop preflight awaits


def _gate_job_preflight(h, job_id, stage):
    """Block only ``job_id``'s preflight for ``stage`` until released."""
    entered, release = threading.Event(), threading.Event()
    original = h.preflight.plan

    def plan(job, stage_):
        if job.job_id == job_id and stage_ is stage and not release.is_set():
            entered.set()
            assert release.wait(10), "test never released the blocked preflight"
        return original(job, stage_)

    h.preflight.plan = plan
    return entered, release


def _record_releases(h):
    """Record every lease release the scheduler asks the repository for, by operation id."""
    released = []
    release, transition = h.repo.release, h.repo.transition_with_lease
    close = h.repo.close_attempt_and_release

    def release_(operation_id, **kwargs):
        released.append(operation_id)
        return release(operation_id, **kwargs)

    def transition_(job, **kwargs):
        if kwargs.get("release") is not None:
            released.append(kwargs["release"][0])
        return transition(job, **kwargs)

    def close_(job, attempt, **kwargs):
        if kwargs.get("release") is not None:
            released.append(kwargs["release"][0])
        return close(job, attempt, **kwargs)

    h.repo.release, h.repo.transition_with_lease, h.repo.close_attempt_and_release = release_, transition_, close_
    return released


async def _blocked_without_attempt(h, n=1):
    h.preflight.rule = lambda stage: ReasonCode.STORAGE_STALE
    blocked = await h.run(make_request(), n=n)
    h.preflight.rule = lambda stage: None
    assert blocked.state is JobState.BLOCKED and blocked.attempt_id is None
    return blocked


@pytest.mark.asyncio
async def test_quiesce_during_an_export_preflight_publishes_nothing_further(h):
    exporting = itertools.count()  # call 0 is the exporting stage boundary; call 1 the first export
    entered, release = _gate_preflight(h, lambda stage: stage is Stage.EXPORTING and next(exporting) == 1)
    job = await h.submit(make_request(), 1)
    running = asyncio.create_task(h.scheduler.run_once())
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        stopping = asyncio.create_task(h.scheduler.quiesce(timeout=5))
        await asyncio.sleep(0)
        assert h.scheduler._quiesced  # the shutdown landed while the export preflight ran
    finally:
        release.set()
    assert await stopping is True
    assert await running is True
    assert not h.calls("publish") and not h.publisher.manifests  # no publication after the await
    stored = h.repo.get_job(job.job_id)
    assert stored.state is JobState.EXPORTING and stored.artifacts == ()  # left for recovery
    assert h.repo.lease_holders() == ()


@pytest.mark.asyncio
async def test_cancel_during_a_retry_preflight_is_not_overwritten_by_the_stale_retry(h):
    h.preflight.rule = lambda stage: ReasonCode.STORAGE_STALE if stage is Stage.PROBING else None
    blocked = await h.run(make_request())
    h.preflight.rule = lambda stage: None
    assert blocked.state is JobState.BLOCKED
    assert h.repo.get_attempt(blocked.attempt_id).state is JobState.BLOCKED  # retry would supersede it
    entered, release = _gate_job_preflight(h, blocked.job_id, Stage.CHECKING)
    retrying = asyncio.create_task(h.scheduler.retry(blocked.job_id, "retry-cancel"))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        assert (await h.scheduler.cancel(blocked.job_id)).state is JobState.CANCELLED
    finally:
        release.set()
    with pytest.raises((SchedulerError, RepositoryError)) as caught:
        await retrying
    assert caught.value.reason_code is ReasonCode.CONFLICT
    final = h.repo.get_job(blocked.job_id)
    assert final.state is JobState.CANCELLED and final.attempt_id == blocked.attempt_id
    assert h.repo.get_attempt(blocked.attempt_id).state is JobState.CANCELLED  # not the stale FAILED
    assert h.repo.get_retry_attempt("retry-cancel") is None
    assert h.repo.lease_holders() == ()
    assert h.scheduler._busy is False and h.scheduler._inflight is None


@pytest.mark.asyncio
async def test_task_cancelled_at_a_stage_boundary_preflight_releases_the_lease(h):
    entered, release = _gate_preflight(h, lambda stage: stage is Stage.PROBING)
    job = await h.submit(make_request(), 1)
    running = asyncio.create_task(h.scheduler.run_once())
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        assert h.repo.lease_holders()  # the run holds the lease across its preflight
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
    finally:
        release.set()
    assert h.repo.lease_holders() == ()  # the cancelled task gave the lease back
    assert h.launcher.stops == [ReasonCode.CANCELLED]
    assert h.scheduler._busy is False and h.scheduler._active is None
    assert h.repo.get_job(job.job_id).state not in (JobState.COMPLETED, JobState.FAILED)  # for recovery
    other = await h.submit(make_request(), 2)
    assert await h.scheduler.run_once() is True  # the scheduler is not wedged
    assert h.repo.get_job(other.job_id).state is JobState.COMPLETED


@pytest.mark.asyncio
async def test_retry_keeps_its_turn_while_its_admission_preflight_runs(h):
    blocked = await _blocked_without_attempt(h)
    queued = await h.submit(make_request(), 2)
    entered, release = _gate_job_preflight(h, blocked.job_id, Stage.CHECKING)
    retrying = asyncio.create_task(h.scheduler.retry(blocked.job_id, "retry-turn"))
    plan = AdmissionPlan(Measurement(True, "fake", FieldStatus.OK, stamp(NOW), None), (), ())

    async def action():
        return True

    try:
        assert await asyncio.to_thread(entered.wait, 10)
        assert h.repo.lease_holders() == ()  # a reservation, not the lease: that is taken at admission
        assert await h.scheduler.run_once() is False  # the queued job does not take the turn
        with pytest.raises(SchedulerError) as caught:
            await h.scheduler.run_setup("setup", plan, action)
        assert caught.value.reason_code is ReasonCode.CONFLICT
    finally:
        release.set()
    attempt = await retrying
    assert h.repo.get_job(blocked.job_id).state is JobState.COMPLETED
    assert h.repo.get_job(blocked.job_id).attempt_id == attempt.attempt_id
    assert h.repo.get_job(queued.job_id).state is JobState.QUEUED
    assert h.scheduler._busy is False and h.scheduler._inflight is None
    assert await h.scheduler.run_once() is True
    assert h.repo.get_job(queued.job_id).state is JobState.COMPLETED


@pytest.mark.asyncio
@pytest.mark.parametrize("exit_path", ["refused", "raised", "cancelled"])
async def test_retry_reservation_is_released_on_every_exit_path(h, monkeypatch, exit_path):
    blocked = await _blocked_without_attempt(h)
    if exit_path == "refused":
        h.preflight.rule = lambda stage: ReasonCode.STORAGE_STALE
        with pytest.raises(SchedulerError):
            await h.scheduler.retry(blocked.job_id, "retry-exit")
    elif exit_path == "raised":
        def refuse(*args, **kwargs):
            raise RepositoryError(ReasonCode.STATE_WRITE_FAILED)

        monkeypatch.setattr(h.repo, "admit_attempt", refuse)
        with pytest.raises(RepositoryError):
            await h.scheduler.retry(blocked.job_id, "retry-exit")
    else:
        entered, release = _gate_job_preflight(h, blocked.job_id, Stage.CHECKING)
        retrying = asyncio.create_task(h.scheduler.retry(blocked.job_id, "retry-exit"))
        try:
            assert await asyncio.to_thread(entered.wait, 10)
            retrying.cancel()
            with pytest.raises(asyncio.CancelledError):
                await retrying
        finally:
            release.set()
    assert h.scheduler._busy is False and h.scheduler._inflight is None
    assert not getattr(h.scheduler, "_retry_keys", {})
    assert h.repo.lease_holders() == ()
    h.preflight.rule = lambda stage: None
    monkeypatch.undo()
    await h.scheduler.retry(blocked.job_id, "retry-again")  # the scheduler is not wedged
    assert h.repo.get_job(blocked.job_id).state is JobState.COMPLETED


@pytest.mark.asyncio
async def test_concurrent_retries_with_one_key_replay_one_attempt(h):
    blocked = await _blocked_without_attempt(h)
    entered, release = _gate_job_preflight(h, blocked.job_id, Stage.CHECKING)
    first = asyncio.create_task(h.scheduler.retry(blocked.job_id, "same-key"))
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        second = asyncio.create_task(h.scheduler.retry(blocked.job_id, "same-key"))
        for _ in range(5):
            await asyncio.sleep(0)
        assert not second.done()  # neither refused nor replayed before the first is admitted
    finally:
        release.set()
    one, two = await asyncio.gather(first, second)
    assert one.attempt_id == two.attempt_id  # the second is a replay of the first
    job = h.repo.get_job(blocked.job_id)
    assert job.state is JobState.COMPLETED and job.attempt_id == one.attempt_id
    assert h.repo.get_attempt(one.attempt_id).ordinal == 1
    assert h.repo.lease_holders() == () and h.scheduler._busy is False


# ------------------------------------------- cancel waits for host publication writers


def _gate_publish(h, slot):
    """Block the publisher's copy for ``slot`` on a threading.Event, like a real to_thread copy."""
    entered, release = threading.Event(), threading.Event()
    original = h.publisher.publish

    async def publish(job, attempt, verified, slot_):
        if slot_ is slot and not release.is_set():
            entered.set()
            assert await asyncio.to_thread(release.wait, 10), "test never released the blocked publication"
        return await original(job, attempt, verified, slot_)

    h.publisher.publish = publish
    return entered, release


def _gate_manifest(h):
    """Block the sidecar write (it runs in the scheduler's writer thread) on a threading.Event."""
    entered, release = threading.Event(), threading.Event()
    original = h.publisher.write_manifest

    def write(job_id, attempt_id, published, provenance):
        if not release.is_set():
            entered.set()
            assert release.wait(10), "test never released the blocked sidecar write"
        return original(job_id, attempt_id, published, provenance)

    h.publisher.write_manifest = write
    return entered, release


async def _spin(n=20):
    for _ in range(n):
        await asyncio.sleep(0)


HOST_WRITERS = [
    # (case, request, blocked publication slot)
    ("video_publication", {"transcript": False, "video": True}, ArtifactSlot.ORIGINAL_VIDEO),
    ("export_publication", {}, ArtifactSlot.parse(T_SLOTS[0])),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("case,request_kwargs,slot", HOST_WRITERS, ids=[c[0] for c in HOST_WRITERS])
async def test_cancel_waits_for_a_blocked_publication_before_releasing_the_lease(h, case, request_kwargs, slot):
    """The lease is held and the job is not yet cancelled while the host copy still runs."""
    entered, release = _gate_publish(h, slot)
    released = _record_releases(h)
    job = await h.submit(make_request(**request_kwargs), 1)
    running = asyncio.create_task(h.scheduler.run_once())
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        ((op, _),) = h.repo.lease_holders()
        cancelling = asyncio.create_task(h.scheduler.cancel(job.job_id))
        await _spin()
        assert not cancelling.done() and not running.done()  # waiting on the writer, not past it
        assert h.launcher.stops == [ReasonCode.CANCELLED]  # the worker launcher was stopped first
        stored = h.repo.get_job(job.job_id)
        assert stored.state is not JobState.CANCELLED and stored.next_action == sch.CANCEL_ACTION
        assert h.repo.lease_holders() == ((op, True),) and released == []
    finally:
        release.set()
    assert (await cancelling).state is JobState.CANCELLED
    assert await running is True
    final = h.repo.get_job(job.job_id)
    assert final.state is JobState.CANCELLED and final.reason_code is ReasonCode.CANCELLED
    assert h.repo.get_attempt(final.attempt_id).state is JobState.CANCELLED
    # The copy ran to completion (the file stays on disk) but nothing claims or lists it.
    assert h.calls("publish") == [("publish", slot.value)]
    assert final.artifacts == () and h.repo.list_artifacts(job.job_id) == []
    assert h.publisher.manifests == []
    assert h.repo.lease_holders() == () and released == [op]  # released exactly once, after the writer
    assert await h.scheduler.run_once() is False  # nothing queued; the scheduler is not wedged


@pytest.mark.asyncio
async def test_cancel_waits_for_a_blocked_sidecar_write_before_releasing_the_lease(h):
    entered, release = _gate_manifest(h)
    released = _record_releases(h)
    job = await h.submit(make_request(transcript=False, video=True), 1)  # the video closes: sidecar first
    running = asyncio.create_task(h.scheduler.run_once())
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        ((op, _),) = h.repo.lease_holders()
        cancelling = asyncio.create_task(h.scheduler.cancel(job.job_id))
        await _spin()
        assert not cancelling.done() and not running.done()
        stored = h.repo.get_job(job.job_id)
        assert stored.state is not JobState.CANCELLED and stored.next_action == sch.CANCEL_ACTION
        assert h.repo.lease_holders() == ((op, True),) and released == []
    finally:
        release.set()
    assert (await cancelling).state is JobState.CANCELLED
    assert await running is True
    final = h.repo.get_job(job.job_id)
    assert final.state is JobState.CANCELLED and final.reason_code is ReasonCode.CANCELLED
    assert len(h.publisher.manifests) == 1  # the sidecar write finished; the cancel preempted the claim
    assert final.artifacts == () and h.repo.list_artifacts(job.job_id) == []
    assert h.repo.lease_holders() == () and released == [op]


@pytest.mark.asyncio
async def test_quiesce_waits_for_a_blocked_publication(h):
    entered, release = _gate_publish(h, ArtifactSlot.ORIGINAL_VIDEO)
    job = await h.submit(make_request(transcript=False, video=True), 1)
    running = asyncio.create_task(h.scheduler.run_once())
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        stopping = asyncio.create_task(h.scheduler.quiesce(timeout=5))
        await _spin()
        assert not stopping.done() and len(h.repo.lease_holders()) == 1  # held while the copy runs
    finally:
        release.set()
    assert await stopping is True
    assert await running is True
    assert h.repo.lease_holders() == () and not h.scheduler._host_writers
    stored = h.repo.get_job(job.job_id)
    assert stored.state not in (JobState.COMPLETED, JobState.CANCELLED) and stored.artifacts == ()


@pytest.mark.asyncio
async def test_quiesce_reports_not_reaped_while_a_publication_outlives_its_timeout(h):
    entered, release = _gate_publish(h, ArtifactSlot.ORIGINAL_VIDEO)
    await h.submit(make_request(transcript=False, video=True), 1)
    running = asyncio.create_task(h.scheduler.run_once())
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        assert await h.scheduler.quiesce(timeout=0.05) is False  # the host writer is still running
        assert len(h.repo.lease_holders()) == 1 and not running.done()
    finally:
        release.set()
    assert await running is True
    assert await h.scheduler.quiesce(timeout=1) is True  # verified gone once the copy finished
    assert h.repo.lease_holders() == ()


@pytest.mark.asyncio
async def test_cancel_keeps_the_lease_when_a_host_writer_outlives_the_stop_bound(h):
    h.scheduler._limits = replace(h.scheduler._limits, worker_stop_grace_seconds=1)  # the smallest valid bound
    entered, release = _gate_publish(h, ArtifactSlot.ORIGINAL_VIDEO)
    job = await h.submit(make_request(transcript=False, video=True), 1)
    queued = await h.submit(make_request(), 2)
    running = asyncio.create_task(h.scheduler.run_once())
    try:
        assert await asyncio.to_thread(entered.wait, 10)
        ((op, _),) = h.repo.lease_holders()
        started = time.monotonic()
        result = await h.scheduler.cancel(job.job_id)  # the bound expires with the copy still running
        assert time.monotonic() - started >= 0.9  # waited the whole bound before giving up
        assert result.state is not JobState.CANCELLED and result.reason_code is ReasonCode.WRITER_UNRESOLVED
        assert result.next_action == sch.CANCEL_ACTION
        stored = h.repo.get_job(job.job_id)
        assert (stored.state, stored.reason_code) == (result.state, ReasonCode.WRITER_UNRESOLVED)
        assert h.repo.lease_holders() == ((op, True),) and not running.done()
        assert h.scheduler._host_writers  # still tracked, never cancelled
    finally:
        release.set()
    assert await running is True
    assert h.repo.get_job(job.job_id).state is not JobState.CANCELLED
    assert h.repo.get_job(job.job_id).artifacts == ()
    assert h.repo.lease_holders() == ()  # the writer is proven finished: the run gave the lease back
    assert (await h.scheduler.cancel(job.job_id)).state is JobState.CANCELLED  # recovery: cancel again
    assert h.repo.get_job(job.job_id).state is JobState.CANCELLED
    assert await h.scheduler.run_once() is True  # the next job starts only now
    assert h.repo.get_job(queued.job_id).state is JobState.COMPLETED
