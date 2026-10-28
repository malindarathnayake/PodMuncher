"""Dependency-injected serial scheduler and stage orchestration (Docs/spec.md:1104).

Pure orchestration over injected ports: nothing here imports the future export or
bootstrap modules, samples storage, talks to Docker or touches the filesystem.  One
job (or one setup action) holds the single execution lease at a time; queued and
blocked jobs hold none.  Every state change is the result of ``transitions.reduce`` and
is persisted with a compare-and-set on the job version, so a late writer aborts its
step without mutating anything.

Approved rules implemented here:

* Sequence (spec.md:142): queued -> checking -> preparing(media if absent) ->
  downloading(remote only) -> probing -> [video publication] -> preparing(ASR) ->
  transcribing -> exporting.  ASR is never provisioned before the probe proved audio.
* Preflight runs before the lease (admission), before every later stage and before
  every publication.  A refused preflight stops at the safe boundary, reaps, releases
  the lease and ends blocked (or partial when an artifact already exists).
* Cancel records the intent, terminates the owned worker (bounded grace lives in the
  launcher), waits for every host-side writer this scheduler started (a publication copy
  or a sidecar write runs in a thread that cannot be interrupted) and only becomes
  ``cancelled`` after ``WriterReaped(True)``.  An unreaped writer keeps the lease and
  ``run_once`` refuses to schedule anything.
* AutoOOM: reap, release, full preflight, then exactly one lower candidate, then fail.
* Progress is informational: coalesced to a 1 Hz snapshot and never completion proof.
* Telemetry spans are job -> stage plus preflight; metric attributes are bounded enums.
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import re
from collections.abc import Awaitable, Callable, Iterable, Mapping
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass, field, replace
from typing import TYPE_CHECKING, Any, Protocol
from uuid import uuid4

from . import transitions as tr
from .admission import AdmissionPlan, may_acquire_lease
from .config import Limits
from .domains import (
    ArtifactSlot,
    Capability,
    DomainValidationError,
    FieldStatus,
    JobState,
    PreflightReason,
    PreflightResult,
    ReasonCode,
    Stage,
    StageOutcome,
    WorkerSlot,
)
from .protocol import WorkerEventRecord, WorkerRequest
from .repository import Repository, RepositoryError
from .snapshots import Artifact, Attempt, Job, Measurement, StageRecord

if TYPE_CHECKING:  # the runtime pulls in the Docker stack; the scheduler only needs its shape
    from .runtime import WorkerResult

PROGRESS_INTERVAL_SECONDS = 1.0
CANCEL_ACTION = "cancel"
RECHECK_ACTION = "recheck"
_FINAL = frozenset((JobState.COMPLETED, JobState.PARTIAL, JobState.FAILED, JobState.CANCELLED))
# Outcomes that claim published artifacts (spec.md:140): their sidecar manifest is written
# before the closing commit, so a client never sees them without it.
_PUBLISHED_FINAL = frozenset((JobState.COMPLETED, JobState.PARTIAL))
_COUNTED_STATES = _FINAL | {JobState.BLOCKED, JobState.INTERRUPTED}
# States the repository accepts as closing writes (bound-exempt): a verified reap must always
# be recordable, so every persisted transition into one of these is a closing write.
_CLOSING_JOB = _COUNTED_STATES
_CLOSING_ATTEMPT = _FINAL | {JobState.INTERRUPTED}
_STORAGE_REASONS = frozenset((
    ReasonCode.STORAGE_INSUFFICIENT, ReasonCode.STORAGE_UNKNOWN, ReasonCode.STORAGE_STALE,
    ReasonCode.STORAGE_GUARD_TRIPPED, ReasonCode.QUOTA_EXCEEDED,
))
_OOM_STAGES = frozenset((Stage.PREPARING, Stage.TRANSCRIBING))
_MEDIA_SLOTS = (WorkerSlot.DOWNLOADED_MEDIA, WorkerSlot.MERGED_VIDEO, WorkerSlot.DOWNLOADED_AUDIO)
_REUSABLE = (Stage.DOWNLOADING, Stage.PROBING)
_PREFLIGHT_REASONS: dict[ReasonCode, PreflightReason] = {
    ReasonCode.STORAGE_INSUFFICIENT: PreflightReason.DISK,
    ReasonCode.STORAGE_UNKNOWN: PreflightReason.DISK,
    ReasonCode.STORAGE_STALE: PreflightReason.DISK,
    ReasonCode.STORAGE_GUARD_TRIPPED: PreflightReason.DISK,
    ReasonCode.QUOTA_EXCEEDED: PreflightReason.DISK,
    ReasonCode.RAM_INSUFFICIENT: PreflightReason.RAM,
    ReasonCode.GPU_UNAVAILABLE: PreflightReason.GPU,
    ReasonCode.DOCKER_UNAVAILABLE: PreflightReason.DOCKER,
    ReasonCode.DOCKER_CONTEXT_UNSUPPORTED: PreflightReason.DOCKER,
    ReasonCode.BACKING_UNRESOLVED: PreflightReason.DOCKER,
    ReasonCode.INPUT_REJECTED: PreflightReason.INPUT,
    ReasonCode.INVALID_INPUT: PreflightReason.INPUT,
    ReasonCode.DURATION_EXCEEDED: PreflightReason.INPUT,
    ReasonCode.DURATION_UNKNOWN: PreflightReason.INPUT,
    ReasonCode.UNAUTHORIZED: PreflightReason.PERMISSION,
    ReasonCode.ASSET_VERIFICATION_FAILED: PreflightReason.ASSET,
    ReasonCode.ASSET_SOURCE_REJECTED: PreflightReason.ASSET,
    ReasonCode.ENVELOPE_UNVALIDATED: PreflightReason.ASSET,
}
# Closed attribute vocabulary: enum values and ids only, never paths, URLs or titles.
METRIC_ATTRIBUTES = frozenset(("state", "stage", "result", "reason", "outcome", "source"))
SPAN_ATTRIBUTES = METRIC_ATTRIBUTES | {"job_id", "attempt_id", "stage_id"}


# ----------------------------------------------------------------------------- ports


class Preflight(Protocol):
    def plan(self, job: Job, stage: Stage) -> AdmissionPlan:
        """Fresh evidence every call; a stale or unknown sample is a non-admitted plan."""


class WorkerLauncher(Protocol):
    """Async wrapper over a fresh ``DockerRuntime`` per stage."""

    @property
    def writers_reaped(self) -> bool:
        """True when no writer owned by the last call can still be running."""

    def ready(self, capability: str) -> bool: ...

    async def prepare(self, capability: str) -> None: ...

    async def run(self, request: WorkerRequest, *, stage: Stage,
                  on_event: Callable[[WorkerEventRecord], None]) -> WorkerResult: ...

    async def stop(self, reason: ReasonCode) -> bool:
        """Terminate, bounded grace, kill, reap.  Returns whether writers are reaped."""


class ModelProvider(Protocol):
    def select(self, job: Job, *, facts: tr.MediaFacts, lower_than: object | None = None) -> object: ...

    def capability(self, choice: object) -> str: ...

    def selection(self, choice: object) -> tuple[str, str]:
        """Concrete ``(device, model id)`` for a choice: device in {cpu, cuda}, never ``auto``."""

    async def prepare(self, choice: object) -> str:
        """Provision the verified snapshot and return its snapshot id."""


class Exporter(Protocol):
    """Placeholder until phase 6: transcript segments -> private temp file per slot."""

    def export(self, segments: object, slots: frozenset[ArtifactSlot]) -> Mapping[ArtifactSlot, object]: ...


class Publisher(Protocol):
    """Wraps ``artifacts`` (assign_slots, verify_artifact, publish, to_artifact)."""

    def assign(self, job: Job, attempt: Attempt, slots: Iterable[WorkerSlot]) -> dict[WorkerSlot, str]: ...

    def verify(self, job: Job, attempt: Attempt, assignment: Mapping[WorkerSlot, str],
               claim: WorkerEventRecord, *, writers_reaped: bool) -> object: ...

    def verify_local_video(self, job: Job, attempt: Attempt) -> object: ...

    def verify_export(self, job: Job, attempt: Attempt, slot: ArtifactSlot, ref: object) -> object: ...

    def probe_facts(self, verified: object) -> tr.MediaFacts: ...

    def source_identity(self, job: Job) -> str:
        """Content sha256 of the source: the contained local file, or the verified download."""

    def reuse(self, job: Job, record: StageRecord) -> Mapping[WorkerSlot, tuple[str, object]]:
        """Re-verify a prior stage's retained outputs: slot -> (basename, verified)."""

    def discard(self, job: Job, attempt: Attempt) -> None: ...

    async def publish(self, job: Job, attempt: Attempt, verified: object, slot: ArtifactSlot) -> Artifact: ...

    def write_manifest(self, job_id: str, attempt_id: str, published: list[Artifact],
                       provenance: object) -> object:
        """Write the sidecar manifest; it replaces itself, so ``published`` is every artifact."""


class Clock(Protocol):
    def now(self) -> str: ...

    def monotonic(self) -> float: ...


class Telemetry(Protocol):
    def span(self, name: str, attrs: Mapping[str, str]) -> AbstractContextManager[Any]: ...

    def count(self, name: str, attrs: Mapping[str, str]) -> None: ...


class SchedulerError(Exception):
    """Closed-reason refusal; carries no paths, URLs or request data."""

    def __init__(self, reason_code: ReasonCode) -> None:
        self.reason_code = ReasonCode.parse(reason_code)
        super().__init__("Scheduler operation refused.")


class _Failure(Exception):
    def __init__(self, reason: ReasonCode) -> None:
        self.reason = reason
        super().__init__("stage failed")


class _Stale(Exception):
    """Another writer changed the job: abort the step without mutation."""


@dataclass(slots=True)
class _Run:
    job: Job
    attempt: Attempt | None
    version: int
    op_id: str | None
    leased: bool = False
    facts: tr.MediaFacts | None = None
    media: object | None = None
    media_name: str | None = None
    segments: object | None = None
    choice: object | None = None
    snapshot: str | None = None
    lower_than: object | None = None
    export: tuple[StageRecord, Mapping[WorkerSlot, tuple[str, object]]] | None = None
    recorded: set[str] = field(default_factory=set)
    stage_id: str | None = None
    last_progress: float | None = None
    outputs: tuple[WorkerSlot, ...] = ()
    identity: str | None = None
    listed: frozenset[str] = frozenset()  # artifact ids the last sidecar written by this run lists


def _settings_hash(job: Job, stage: Stage) -> str:
    request = job.request
    base: dict[str, object] = {"stage": stage.value, "kind": request.source.kind,
                               "limits": asdict(request.limits)}
    if stage is Stage.DOWNLOADING:
        base.update(video=request.outputs.video, quality=request.outputs.video_quality)
    elif stage is not Stage.PROBING:
        base.update(outputs=asdict(request.outputs), selection=asdict(request.selection))
    return hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest()


def _reason(error: BaseException) -> ReasonCode:
    if isinstance(error, DomainValidationError):
        return error.reason_code
    return ReasonCode.WORKER_EXIT_UNEXPECTED


def _published(job: Job) -> frozenset[ArtifactSlot]:
    return frozenset(a.slot for a in job.artifacts if a.available)


def _all_claimed(job: Job) -> bool:
    """Every requested artifact is claimed: any closing event would reduce to completed."""
    return tr.requested_slots(job.request) <= _published(job)


def _source_value(record: StageRecord) -> str | None:
    return next((value for name, value in record.verified_input_hashes if name == "source"), None)


def _closes_job(job: Job) -> bool:
    return job.state in _CLOSING_JOB


def _closes_attempt(attempt: Attempt) -> bool:
    return attempt.state in _CLOSING_ATTEMPT


def _finished_ok(attempt: Attempt) -> list[StageRecord]:
    return [r for r in attempt.stage_records if r.ended_at is not None and r.status is StageOutcome.OK]


class Scheduler:
    def __init__(self, repository: Repository, *, preflight: Preflight, launcher: WorkerLauncher,
                 models: ModelProvider, exporter: Exporter, publisher: Publisher, clock: Clock,
                 telemetry: Telemetry, limits: Limits | None = None,
                 ids: Callable[[], str] | None = None) -> None:
        self._repo = repository
        self._preflight = preflight
        self._launcher = launcher
        self._models = models
        self._exporter = exporter
        self._publisher = publisher
        self._clock = clock
        self._tel = telemetry
        self._limits = limits if limits is not None else Limits()
        self._ids = ids if ids is not None else (lambda: str(uuid4()))
        self._active: _Run | None = None
        self._busy = False
        self._ops: dict[str, str] = {}  # job_id -> the lease operation this scheduler acquired for it
        self._reuse: dict[str, dict[Stage, StageRecord]] = {}
        self._progress: dict[str, float] = {}
        self._quiesced = False
        self._inflight: asyncio.Task[Any] | None = None
        self._setup_stop: Callable[[], Awaitable[bool]] | None = None
        self._setup_unreaped = False
        self._retry_keys: dict[str, asyncio.Future[None]] = {}  # retry keys mid-admission
        # Host-side writers (publication copies, sidecar writes) started for the current run.
        # Cancel, quiesce and abandon wait for them before the lease can be released.
        self._host_writers: set[asyncio.Future[Any]] = set()
        self.dropped_events = 0
        self.fault: ReasonCode | None = None

    # ------------------------------------------------------------------ public API

    async def submit(self, job: Job, idempotency_key: str) -> tuple[Job, bool]:
        return self._repo.submit(job, idempotency_key)

    def progress(self, job_id: str) -> float | None:
        """Latest coalesced progress snapshot; informational, never completion proof."""
        return self._progress.get(job_id)

    async def run_once(self) -> bool:
        if (self._quiesced or self._busy or self._repo.admission_blocked is not None
                or self._repo.lease_holders()):
            return False
        queued = self._repo.queued_jobs(1)
        if not queued:
            return False
        self._busy = True
        self._inflight = asyncio.current_task()
        try:
            await self._run_queued(queued[0])
        finally:
            self._busy = False
            self._inflight = None
        return True

    async def run_setup(self, operation_id: str, plan: AdmissionPlan,
                        action: Callable[[], Awaitable[bool]], *,
                        stop: Callable[[], Awaitable[bool]] | None = None) -> None:
        """Run setup work under the same single lease; ``action`` returns writers_reaped.

        ``stop`` (optional) asks the action to wind down and returns whether its writer is
        reaped; ``quiesce`` calls it and treats anything but ``True`` as unreaped.
        """
        if self._quiesced:
            raise SchedulerError(ReasonCode.SERVICE_STOPPED)
        if self._busy or self._repo.admission_blocked is not None or self._repo.lease_holders():
            raise SchedulerError(ReasonCode.CONFLICT)
        if not may_acquire_lease(JobState.CHECKING, plan):
            raise SchedulerError(plan.reason if not plan.admitted else ReasonCode.CONFLICT)
        if not self._repo.try_acquire(operation_id, plan):
            raise SchedulerError(ReasonCode.CONFLICT)
        self._busy = True
        self._inflight, self._setup_stop = asyncio.current_task(), stop
        reaped = False
        try:
            reaped = bool(await action())
        finally:
            self._busy = False
            self._inflight, self._setup_stop = None, None
            self._setup_unreaped = not reaped
            if reaped:
                self._repo.release(operation_id, writers_reaped=True)

    async def recheck(self, job_id: str) -> bool:
        """Fresh preflight for a blocked/terminal job; never creates an attempt or output."""
        job, attempt = self._load(job_id)
        tr.reduce(job, attempt, tr.RecheckRequested(job_id), now=self._clock.now())
        return (await self._plan(job, Stage.CHECKING)).admitted

    async def clone(self, job_id: str, overrides: Mapping[str, object] | None = None) -> Job:
        prior, attempt = self._load(job_id)
        request = tr.clone_request(prior, overrides)
        now = self._clock.now()
        clone = Job(self._ids(), request, request.request_hash, JobState.QUEUED, ReasonCode.OK, None,
                    None, now, now, (), None)
        stored, _ = self._repo.submit(clone, f"clone:{clone.job_id}")
        identity = await self._clone_identity(prior, stored)
        proofs: dict[Stage, StageRecord] = {}
        for record in () if attempt is None or identity is None else attempt.stage_records:
            if (record.stage in _REUSABLE and record.output_slots
                    and tr.reuse_allowed(record, identity, _settings_hash(stored, record.stage))):
                proofs[record.stage] = record
        if proofs:
            self._reuse[stored.job_id] = proofs
        return stored

    async def _clone_identity(self, prior: Job, clone: Job) -> str | None:
        """Content identity both jobs share, or None.  Never a hash of the path or URL string."""
        try:
            if clone.request.source.kind == "local":
                return await asyncio.to_thread(self._publisher.source_identity, clone)
            if prior.request.source == clone.request.source:  # same remote: the verified download
                return await asyncio.to_thread(self._publisher.source_identity, prior)
        except Exception:  # noqa: BLE001 - no provable identity means no reuse
            return None
        return None

    async def cancel(self, job_id: str) -> Job:
        job, attempt = self._load(job_id)
        version = self._repo.get_job_version(job_id)
        decision = tr.reduce(job, attempt, tr.CancelRequested(job_id), now=self._clock.now())
        persist_error: RepositoryError | None = None
        try:
            self._store(decision, version)
        except RepositoryError as error:
            if job.state not in tr._ACTIVE:
                raise
            persist_error = error  # a writer may be running: stop it before reporting
        if job.state not in tr._ACTIVE:
            self._count_state(job, decision.job)
            return decision.job
        reaped = await self._launcher.stop(ReasonCode.CANCELLED)
        # A host writer (publication or sidecar) may still be running: the lease is released
        # and the final state committed only once it is proven finished, within the stop bound.
        reaped = await self._drain_host_writers() and reaped
        if persist_error is not None:
            try:
                self._store(decision, version)
            except RepositoryError:
                pass
            raise persist_error
        job, attempt = self._load(job_id)
        version = self._repo.get_job_version(job_id)
        final = tr.reduce(job, attempt, tr.WriterReaped(job_id, reaped, attempt.attempt_id),
                          now=self._clock.now())
        op = self._ops.get(job_id)  # only this job's own lease operation, never another holder's
        held = {holder[0] for holder in self._repo.lease_holders()}
        release = (op, True) if reaped and op is not None and op in held else None
        self._store(final, version, release=release)
        if release is not None:
            self._ops.pop(job_id, None)
        self._evict(final.job)
        self._count_state(job, final.job)
        return final.job

    def _replay(self, job_id: str, retry_key: str) -> Attempt | None:
        """Durable retry-key replay: the stored attempt, or None; a key of another job conflicts."""
        known = self._repo.get_retry_attempt(retry_key)
        if known is None:
            return None
        if known[0] != job_id:
            raise SchedulerError(ReasonCode.CONFLICT)
        return known[1]

    async def retry(self, job_id: str, retry_key: str) -> Attempt:
        if self._quiesced:
            raise SchedulerError(ReasonCode.SERVICE_STOPPED)
        known = self._replay(job_id, retry_key)  # durable replay, before any other check
        if known is not None:
            return known
        pending = self._retry_keys.get(retry_key)
        if pending is not None:  # the same key is mid-admission here: wait for it, then replay
            await asyncio.shield(pending)
            known = self._replay(job_id, retry_key)
            if known is not None:
                return known
            if self._quiesced:
                raise SchedulerError(ReasonCode.SERVICE_STOPPED)
        job, attempt = self._load(job_id)
        version = self._repo.get_job_version(job_id)  # the decision below is reduced from this read
        if self._busy or self._repo.admission_blocked is not None or self._repo.lease_holders():
            raise SchedulerError(ReasonCode.CONFLICT)
        export = self._export_plan(job, attempt)
        new_id = self._ids()
        decision = tr.reduce(job, attempt, tr.RetryRequested(job_id, new_id, retry_key),
                             now=self._clock.now())
        # Reserve the scheduler across the off-loop preflight so no later run_once or setup
        # takes the turn.  This is not the lease: that is acquired only at admission.
        self._busy = True
        self._inflight = asyncio.current_task()
        admitted: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._retry_keys[retry_key] = admitted
        try:
            plan = await self._plan(job, Stage.CHECKING)
            # The preflight was an await: re-check what another coroutine may have changed.
            # The durable key comes first, so a same-key retry admitted meanwhile replays.
            known = self._replay(job_id, retry_key)
            if known is not None:
                return known
            if self._quiesced:
                raise SchedulerError(ReasonCode.SERVICE_STOPPED)
            if not plan.admitted or not may_acquire_lease(JobState.CHECKING, plan):
                raise SchedulerError(plan.reason if not plan.admitted else ReasonCode.CONFLICT)
            if self._repo.admission_blocked is not None or self._repo.lease_holders():
                raise SchedulerError(ReasonCode.CONFLICT)
            run = _Run(job, attempt, version, self._ids(), export=export)
            with self._tel.span("job", {"job_id": job_id}):
                decision = self._supersede(run, decision)
                try:
                    replayed = self._commit(run, decision, acquire=(run.op_id, plan), retry_key=retry_key)
                finally:
                    if not admitted.done():
                        admitted.set_result(None)
                if replayed is not None:  # lost a race on the same key: nothing mutated or leased
                    return replayed
                await self._drive(run)
        finally:
            self._busy = False
            self._inflight = None
            if self._retry_keys.get(retry_key) is admitted:
                del self._retry_keys[retry_key]
            if not admitted.done():
                admitted.set_result(None)
        return self._repo.get_attempt(new_id)

    def _supersede(self, run: _Run, decision: tr.Decision) -> tr.Decision:
        """Close the attempt a retry supersedes under the CAS on the version it was reduced from.

        A cancel (or any other writer) that landed during the retry's preflight moved the
        version, so this write fails with CONFLICT and the stale retry mutates nothing.  No
        await separates it from the admission transaction that follows.
        """
        superseded = decision.superseded_attempt
        if superseded is None:
            return decision
        self._repo.close_attempt_and_release(run.job, superseded, expected_version=run.version,
                                             release=None)
        run.version += 1
        self._record_closed(run, superseded)
        return replace(decision, superseded_attempt=None)

    async def quiesce(self, *, timeout: float) -> bool:
        """Stop accepting work, stop the current writer/setup, and wait for the in-flight task.

        True only when no scheduler-owned writer remains.  Idempotent.
        """
        self._quiesced = True
        task, current = self._inflight, asyncio.current_task()
        try:
            async with asyncio.timeout(timeout):
                if self._busy:
                    if self._setup_stop is not None and await self._setup_stop() is not True:
                        self._setup_unreaped = True  # spec.md:138: never report an unreaped writer as reaped
                    await self._launcher.stop(ReasonCode.CANCELLED)
                if task is not None and task is not current and not task.done():
                    await asyncio.wait({task})
                await self._drain_host_writers(bounded=False)  # host writers drain within this bound
        except TimeoutError:
            return False
        waiting = task is not None and task is not current and not task.done()
        return (bool(self._launcher.writers_reaped) and not self._setup_unreaped and not waiting
                and not self._pending_host_writers())

    # ------------------------------------------------------------- host writers

    def _pending_host_writers(self) -> set[asyncio.Future[Any]]:
        return {writer for writer in self._host_writers if not writer.done()}

    async def _host_write(self, awaitable: Awaitable[Any]) -> Any:
        """Run a host-side writer (publication copy or sidecar write) as a tracked future.

        The writer runs in a thread that cannot be interrupted, so it is shielded: cancelling
        the run leaves it running to completion, and ``cancel``/``quiesce``/``_abandon`` wait
        for it before the lease is released.  The run's own exception propagates unchanged.
        """
        writer = asyncio.ensure_future(awaitable)
        self._host_writers.add(writer)
        writer.add_done_callback(self._host_writer_done)
        return await asyncio.shield(writer)

    def _host_writer_done(self, writer: asyncio.Future[Any]) -> None:
        self._host_writers.discard(writer)
        if not writer.cancelled():
            writer.exception()  # an orphaned writer's failure is owned here: never "never retrieved"

    async def _drain_host_writers(self, *, bounded: bool = True) -> bool:
        """Wait for every host writer; True only when none can still be running.

        Bounded by the worker stop grace (the existing stop bound) unless the caller holds its
        own bound.  A writer still running when the bound expires is never cancelled: it keeps
        running, and the caller keeps the lease.
        """
        bound = float(self._limits.worker_stop_grace_seconds) if bounded else None
        try:
            async with asyncio.timeout(bound):
                while pending := self._pending_host_writers():
                    await asyncio.wait(pending)
        except TimeoutError:
            return False
        return not self._pending_host_writers()

    # --------------------------------------------------------------- persistence

    def _load(self, job_id: str) -> tuple[Job, Attempt | None]:
        job = self._repo.get_job(job_id)
        if job is None:
            raise SchedulerError(ReasonCode.INVALID_INPUT)
        attempt = None if job.attempt_id is None else self._repo.get_attempt(job.attempt_id)
        return job, attempt

    def _store(self, decision: tr.Decision, version: int, *, release: tuple[str, bool] | None = None) -> None:
        """Out-of-run persistence (cancel): CAS on the freshly read version.

        A transition into a closing state is a closing write, so a reaped job can always be
        recorded and its lease released even above the size bound.
        """
        closing = _closes_job(decision.job)
        if release is not None:
            self._repo.transition_with_lease(decision.job, expected_version=version, release=release,
                                             closing=closing)
        else:
            self._repo.update_job(decision.job, expected_version=version, closing=closing)
        if decision.attempt is not None:
            self._repo.update_attempt(decision.attempt, closing=_closes_attempt(decision.attempt))

    def _evict(self, job: Job) -> None:
        """Drop per-job in-memory state once the job reached a final state."""
        if job.state in _FINAL:
            self._reuse.pop(job.job_id, None)
            self._progress.pop(job.job_id, None)
            self._ops.pop(job.job_id, None)

    def _commit(self, run: _Run, decision: tr.Decision, *, acquire: tuple[str, AdmissionPlan] | None = None,
                retry_key: str | None = None) -> Attempt | None:
        """Persist one decision.  Returns the stored attempt when ``retry_key`` was a replay."""
        job, new = decision.job, decision.attempt
        release = bool(run.leased and self._launcher.writers_reaped
                       and tr.Effect.RELEASE_LEASE in decision.effects
                       and tr.Effect.KEEP_LEASE not in decision.effects)
        fresh = new is not None and (run.attempt is None or new.attempt_id != run.attempt.attempt_id)
        if decision.superseded_attempt is not None:
            self._repo.update_attempt(decision.superseded_attempt,
                                      closing=_closes_attempt(decision.superseded_attempt))
            self._record_closed(run, decision.superseded_attempt)
        if fresh:
            # Job, attempt, retry key and lease are admitted in one transaction: a collision
            # (CONFLICT) leaves no attempt-less active job and no held lease behind.
            _, stored, again = self._repo.admit_attempt(job, new, retry_key=retry_key,
                                                        expected_version=run.version, acquire=acquire)
            if again:
                return stored
            run.version += 1
            if acquire is not None:
                run.leased = True
                self._ops[job.job_id] = acquire[0]
            if release:
                self._repo.release(run.op_id, writers_reaped=True)
                run.leased = False
                self._ops.pop(job.job_id, None)
        else:
            lease = (run.op_id, True) if release else None
            closing = acquire is None and _closes_job(job)  # admission stays bound-checked
            if acquire is not None or lease is not None:
                self._repo.transition_with_lease(job, expected_version=run.version, acquire=acquire,
                                                 release=lease, closing=closing)
            else:
                self._repo.update_job(job, expected_version=run.version, closing=closing)
            run.version += 1
            if acquire is not None:
                run.leased = True
                self._ops[job.job_id] = acquire[0]
            if lease is not None:
                run.leased = False
                self._ops.pop(job.job_id, None)
            if new is not None and new != run.attempt:
                self._repo.update_attempt(new, closing=_closes_attempt(new))
                self._record_closed(run, new)
        self._count_state(run.job, job)
        run.job, run.attempt = job, new
        self._evict(job)
        return None

    def _record_closed(self, run: _Run, attempt: Attempt) -> None:
        for record in attempt.stage_records:
            if record.ended_at is not None and record.stage_id not in run.recorded:
                try:
                    self._repo.record_stage(record)
                except RepositoryError as error:
                    if error.reason_code is ReasonCode.STORAGE_INSUFFICIENT and attempt.state in _COUNTED_STATES:
                        continue  # derived row only: the closed attempt document already holds the record
                    if error.reason_code is not ReasonCode.CONFLICT:  # CONFLICT: already recorded
                        raise
                run.recorded.add(record.stage_id)

    def _count_state(self, before: Job, after: Job) -> None:
        if after.state in _COUNTED_STATES and after.state is not before.state:
            self._tel.count("job", {"state": after.state.value, "source": after.request.source.kind})

    def _decide(self, run: _Run, event: tr.Event, *, patch: Callable[[Attempt], Attempt] | None = None,
                ) -> tr.Decision:
        """Reduce one event against the run without persisting it."""
        decision = tr.reduce(run.job, run.attempt, event, now=self._clock.now(), facts=run.facts)
        if decision.ignored:
            raise _Stale
        if patch is not None and decision.attempt is not None:
            decision = replace(decision, attempt=patch(decision.attempt))
        return decision

    def _apply(self, run: _Run, event: tr.Event, *, patch: Callable[[Attempt], Attempt] | None = None,
               ) -> tr.Decision:
        decision = self._decide(run, event, patch=patch)
        self._commit(run, decision)
        return decision

    async def _settle(self, run: _Run, event: tr.Event, *, patch: Callable[[Attempt], Attempt] | None = None,
                      ) -> tr.Decision | None:
        """Apply an event that may close the job as completed/partial.

        Such an outcome claims published artifacts, so the sidecar manifest listing them is
        written first (spec.md:140); only then is the state committed and made visible.  When
        the manifest cannot be written the event is not committed and ``None`` is returned:
        a storage reason halts (blocked, or partial over the artifacts claimed earlier), any
        other reason fails the attempt (failed, or partial).  When every requested artifact is
        already claimed, any closing event would reduce to completed, so the job is halted
        into blocked for an explicit recheck/retry instead.  Nothing here claims completed.
        """
        decision = self._decide(run, event, patch=patch)
        if decision.job.state in _PUBLISHED_FINAL:
            try:
                if await self._ensure_manifest(run, decision.job):
                    self._resume(run)  # a cancel or quiesce that landed during the write preempts
            except _Failure as error:
                if error.reason in _STORAGE_REASONS or _all_claimed(run.job):
                    await self._halt(run, error.reason)
                else:
                    await self._fail(run, error.reason)
                return None
        self._commit(run, decision)
        return decision

    def _check_current(self, run: _Run) -> None:
        if self._repo.get_job_version(run.job.job_id) != run.version:
            raise _Stale

    def _resume(self, run: _Run) -> None:
        """Re-validate a run after an off-loop preflight: cancel or quiesce may have interleaved."""
        if self._quiesced:
            raise _Stale
        self._check_current(run)

    # ----------------------------------------------------------------- preflight

    async def _plan(self, job: Job, stage: Stage) -> AdmissionPlan:
        """Fresh preflight in a worker thread: it samples Docker storage and may block (spec.md:25)."""
        attrs = {"stage": stage.value, "job_id": job.job_id}
        with self._tel.span("preflight", attrs):
            try:
                plan = await asyncio.to_thread(self._preflight.plan, job, stage)
                if type(plan) is not AdmissionPlan:
                    raise DomainValidationError(ReasonCode.STORAGE_UNKNOWN)
            except Exception as error:  # noqa: BLE001 - fail closed on any preflight fault
                reason = _reason(error) if isinstance(error, DomainValidationError) else ReasonCode.STORAGE_UNKNOWN
                plan = AdmissionPlan(Measurement(None, "preflight", FieldStatus.ERROR,
                                                 self._clock.now(), reason), (), ())
        if plan.admitted:
            self._tel.count("preflight", {"result": PreflightResult.PASS.value})
        else:
            label = _PREFLIGHT_REASONS.get(plan.reason, PreflightReason.UNKNOWN)
            self._tel.count("preflight", {"result": PreflightResult.BLOCKED.value, "reason": label.value})
        return plan

    # ----------------------------------------------------------------- job runs

    async def _run_queued(self, job: Job) -> None:
        version = self._repo.get_job_version(job.job_id)
        with self._tel.span("job", {"job_id": job.job_id}):
            plan = await self._plan(job, Stage.CHECKING)
            if self._quiesced or self._repo.get_job_version(job.job_id) != version:
                return  # cancelled/changed or shutting down while the preflight ran: admit nothing
            now = self._clock.now()
            if not plan.admitted or not may_acquire_lease(JobState.CHECKING, plan):
                reason = plan.reason if not plan.admitted else ReasonCode.CONFLICT
                blocked = tr.reduce(job, None, tr.Blocked(job.job_id, reason, RECHECK_ACTION), now=now)
                self._repo.update_job(blocked.job, expected_version=version,
                                      closing=True)  # no lease is ever taken
                self._count_state(job, blocked.job)
                return
            decision = tr.reduce(job, None, tr.Admitted(job.job_id, self._ids()), now=now)
            run = _Run(job, None, version, self._ids())
            try:
                self._commit(run, decision, acquire=(run.op_id, plan))
            except RepositoryError as error:
                if run.leased:
                    await self._abandon(run)  # attempt row failed after the lease: give it back
                if error.reason_code is ReasonCode.CONFLICT:
                    return  # lost the race: nothing more to do
                raise
            await self._drive(run)

    async def _drive(self, run: _Run) -> None:
        self._active = run
        try:
            await self._loop(run)
        except _Stale:
            await self._abandon(run)
        except RepositoryError as error:
            if error.reason_code is ReasonCode.CONFLICT:
                await self._abandon(run)
            else:
                self.fault = error.reason_code  # stop scheduling; never report success
                await self._launcher.stop(error.reason_code)
        except tr.TransitionError:
            await self._abandon(run)
        except asyncio.CancelledError:
            # Task cancellation at any await (e.g. an off-loop preflight): the same cleanup as
            # the abandon path, so the lease never outlives the run that holds it.
            await self._abandon(run)
            raise
        finally:
            self._active = None

    async def _abandon(self, run: _Run) -> None:
        reaped = await self._launcher.stop(ReasonCode.CANCELLED)
        # A task cancellation at a shielded host write leaves that writer running: the lease
        # is given back only once it is proven finished (same bound as the launcher stop).
        reaped = await self._drain_host_writers() and reaped
        if reaped and run.leased:
            try:
                # Exactly once: a concurrent cancel may already have released this operation
                # in its closing write; only a still-held lease is released here.
                if any(holder[0] == run.op_id for holder in self._repo.lease_holders()):
                    self._repo.release(run.op_id, writers_reaped=True)
                run.leased = False
                self._ops.pop(run.job.job_id, None)
            except RepositoryError:
                pass

    async def _loop(self, run: _Run) -> None:
        while True:
            if self._quiesced:  # shutdown: leave the job for recovery, start nothing new
                raise _Stale
            job, attempt = run.job, run.attempt
            if job.state in _FINAL or job.state in (JobState.BLOCKED, JobState.INTERRUPTED):
                return
            if job.reason_code is ReasonCode.WRITER_UNRESOLVED or job.next_action == CANCEL_ACTION:
                return
            finished = _finished_ok(attempt)
            if not any(r.stage is Stage.CHECKING for r in finished):
                if not await self._stage(run, Stage.CHECKING):
                    return
                continue
            if run.export is not None and not any(r.stage is Stage.TRANSCRIBING for r in attempt.stage_records):
                self._adopt_export(run)
                continue
            probed = any(r.stage is Stage.PROBING for r in finished)
            if probed and job.request.outputs.video and ArtifactSlot.ORIGINAL_VIDEO not in _published(job):
                if not await self._publish_video(run):
                    return
                continue
            stage = tr.next_stage(job.request, attempt, run.facts,
                                  export_plan=None if run.export is None else self._export_marker(run))
            if stage is None:
                await self._fail(run, ReasonCode.CONFLICT)
                return
            if (stage is Stage.PREPARING and run.facts is None and self._launcher.ready(Capability.MEDIA.value)):
                stage = Stage.DOWNLOADING if job.request.source.kind != "local" else Stage.PROBING
            if not await self._stage(run, stage):
                return

    # ------------------------------------------------------------- export retry

    def _export_plan(self, job: Job, attempt: Attempt | None) -> Any:
        if attempt is None:
            return None
        try:
            plan = tr.export_retry_plan(job, [attempt])
        except tr.TransitionError as error:
            if error.reason_code is ReasonCode.ALREADY_TERMINAL:
                raise
            return None
        try:
            return plan.transcribing, self._publisher.reuse(job, plan.transcribing)
        except Exception:  # noqa: BLE001 - retained outputs gone: fall back to a full retry
            return None

    def _export_marker(self, run: _Run) -> tr.ExportRetryPlan:
        record, _ = run.export
        return tr.ExportRetryPlan(run.attempt.attempt_id, record, Stage.EXPORTING)

    def _adopt_export(self, run: _Run) -> None:
        """Carry the verified transcribing proof into the new attempt; never launches ASR."""
        prior, outputs = run.export
        now = self._clock.now()
        record = StageRecord(self._ids(), run.attempt.attempt_id, Stage.TRANSCRIBING, StageOutcome.OK, now, now,
                             prior.verified_input_hashes, prior.settings_hash, prior.output_slots)
        attempt = replace(run.attempt, stage_records=run.attempt.stage_records + (record,))
        self._repo.update_attempt(attempt)
        self._record_closed(run, attempt)
        run.attempt = attempt
        run.segments = outputs[WorkerSlot.TRANSCRIPT_SEGMENTS][1]
        run.facts = tr.MediaFacts(True, run.job.request.outputs.video, 0.0)

    # -------------------------------------------------------------------- stages

    async def _ensure_identity(self, run: _Run) -> None:
        """Content identity of a local source, computed once per run (never from the path)."""
        if run.identity is None and run.job.request.source.kind == "local":
            run.identity = await asyncio.to_thread(self._publisher.source_identity, run.job)

    def _hashes(self, run: _Run) -> tuple[tuple[str, str], ...]:
        return (("source", run.identity),) if run.identity else ()

    async def _stage(self, run: _Run, stage: Stage) -> bool:
        job = run.job
        if stage is not Stage.CHECKING:  # the pre-lease admission preflight covers checking
            plan = await self._plan(job, stage)
            self._resume(run)  # a cancel during the preflight preempts: no new stage starts
            if not plan.admitted:
                await self._halt(run, plan.reason)
                return False
        try:
            if run.export is None and stage is not Stage.EXPORTING:
                # An export retry publishes from the retained validated transcript: the source
                # (possibly deleted since) is never reopened, and exporting never needs it.
                await self._ensure_identity(run)
        except Exception as error:  # noqa: BLE001 - an unreadable source fails the attempt
            await self._fail(run, _reason(error))
            return False
        proof = self._proof(run, stage)
        stage_id = self._ids()
        settings = proof.settings_hash if proof else _settings_hash(job, stage)
        hashes = proof.verified_input_hashes if proof else self._hashes(run)
        self._apply(run, tr.StageStarted(job.job_id, run.attempt.attempt_id, stage, stage_id, settings, hashes))
        run.stage_id, run.last_progress, run.outputs = stage_id, None, ()
        attrs = {"stage": stage.value, "job_id": job.job_id, "attempt_id": run.attempt.attempt_id,
                 "stage_id": stage_id}
        outcome, measured = StageOutcome.ERROR, False
        try:
            with self._tel.span("stage", attrs):
                measured = True
                reason = ReasonCode.OK
                try:
                    await self._execute(run, stage, stage_id, proof)
                except _Failure as failure:
                    reason = failure.reason
                except Exception as error:  # noqa: BLE001 - every port failure maps to a closed reason
                    reason = _reason(error)
            run.stage_id = None  # events for a finished stage are late
            if self._quiesced:
                raise _Stale
            self._check_current(run)
            if not self._launcher.writers_reaped:
                self._apply(run, tr.WriterReaped(job.job_id, False, run.attempt.attempt_id))
                return False
            if reason is ReasonCode.OK:
                outcome = StageOutcome.OK  # a terminal probe result still finished its stage
                return await self._finish_stage(run, stage, stage_id)
            if reason is ReasonCode.MODEL_OOM and stage in _OOM_STAGES:
                return await self._fallback(run)
            if reason in _STORAGE_REASONS:
                await self._halt(run, reason)
                return False
            self._apply(run, tr.StageFinished(job.job_id, run.attempt.attempt_id, stage_id,
                                              StageOutcome.ERROR, reason))
            return False
        except (_Stale, asyncio.CancelledError):
            outcome = StageOutcome.CANCELLED
            raise
        finally:
            run.stage_id = None
            if measured:  # the span exit recorded the duration; every exit reports it once
                self._tel.count("stage", {"stage": stage.value, "outcome": outcome.value})

    def _proof(self, run: _Run, stage: Stage) -> StageRecord | None:
        proof = self._reuse.get(run.job.job_id, {}).get(stage)
        remote = run.job.request.source.kind != "local"
        if proof is None or (stage is Stage.PROBING and remote and run.media_name is None):
            return None
        # Local: the file's current content identity.  Remote download: the clone proved the same
        # source and publisher.reuse re-verifies the retained bytes.  Remote probe: the run's
        # verified download identity, never the proof's own hash (that would always match).
        identity = _source_value(proof) if remote and stage is Stage.DOWNLOADING else run.identity
        if identity is None:
            return None
        if tr.reuse_allowed(proof, identity, _settings_hash(run.job, stage)):
            return proof
        if remote and stage is Stage.PROBING and _source_value(proof) != identity:
            self._reuse.pop(run.job.job_id, None)  # the bytes changed: drop every stale proof
        return None

    async def _finish_stage(self, run: _Run, stage: Stage, stage_id: str) -> bool:
        job_id, attempt_id = run.job.job_id, run.attempt.attempt_id
        if stage is Stage.PROBING:
            facts = run.facts
            probed = await self._settle(run, tr.MediaProbed(job_id, attempt_id, facts.has_audio,
                                                            facts.has_video, facts.duration_seconds))
            if probed is None or probed.job.state in _FINAL:
                return False
        slots, hashes = run.outputs, self._hashes(run)

        def patch(attempt: Attempt) -> Attempt:
            def seal(record: StageRecord) -> StageRecord:
                if record.stage_id != stage_id:
                    return record
                if stage in _REUSABLE and hashes:  # the verified content identity, not the request
                    return replace(record, output_slots=slots, verified_input_hashes=hashes)
                return replace(record, output_slots=slots)
            return replace(attempt, stage_records=tuple(seal(r) for r in attempt.stage_records))

        # A finished probe (retry over an earlier video) or export closes the job: sidecar first.
        decision = await self._settle(run, tr.StageFinished(job_id, attempt_id, stage_id, StageOutcome.OK),
                                      patch=patch)
        return decision is not None and decision.job.state not in _FINAL

    async def _execute(self, run: _Run, stage: Stage, stage_id: str, proof: StageRecord | None) -> None:
        job = run.job
        if stage is Stage.CHECKING:
            return
        if stage is Stage.PREPARING:
            if run.facts is None:
                await self._launcher.prepare(Capability.MEDIA.value)
                return
            run.choice = self._models.select(job, facts=run.facts, lower_than=run.lower_than)
            run.snapshot = await self._models.prepare(run.choice)
            return
        if stage is Stage.DOWNLOADING:
            await self._download(run, stage, stage_id, proof)
        elif stage is Stage.PROBING:
            await self._probe(run, stage, stage_id, proof)
        elif stage is Stage.TRANSCRIBING:
            await self._transcribe(run, stage, stage_id)
        elif stage is Stage.EXPORTING:
            await self._export(run)
        else:
            raise _Failure(ReasonCode.CONFLICT)

    def _request(self, run: _Run, stage_id: str, *, capability: str, operation: str,
                 assignment: Mapping[WorkerSlot, str], remote: bool = False,
                 input_basename: str | None = None, snapshot: str | None = None,
                 selection: object | None = None) -> WorkerRequest:
        request = run.job.request
        limits = replace(self._limits, media_bytes=request.limits.media_bytes,
                         duration_seconds=request.limits.duration_seconds,
                         export_bytes=request.limits.export_bytes)
        source = {"kind": request.source.kind, "input": request.source.input} if remote else None
        return WorkerRequest(1, run.job.job_id, run.attempt.attempt_id, stage_id, capability, operation,
                             {slot.value: name for slot, name in assignment.items()}, source,
                             request.outputs, input_basename, snapshot, limits,
                             request.selection if selection is None else selection)

    async def _launch(self, run: _Run, request: WorkerRequest, stage: Stage, stage_id: str) -> WorkerResult:
        def on_event(event: WorkerEventRecord) -> None:
            self._on_event(run, stage_id, event)

        result = await self._launcher.run(request, stage=stage, on_event=on_event)
        if result.completed is not True:
            raise _Failure(ReasonCode.WORKER_PROTOCOL_MISMATCH)
        return result

    def _on_event(self, run: _Run, stage_id: str, event: WorkerEventRecord) -> None:
        ids = (run.job.job_id, run.attempt.attempt_id, stage_id)
        if (event.job_id, event.attempt_id, event.stage_id) != ids or run.stage_id != stage_id:
            self.dropped_events += 1  # late or foreign event: never mutates anything
            return
        if event.event == "progress":
            now = self._clock.monotonic()
            if run.last_progress is None or now - run.last_progress >= PROGRESS_INTERVAL_SECONDS:
                run.last_progress = now
                self._progress[run.job.job_id] = float(event.payload["percent"])

    def _claim(self, result: WorkerResult, run: _Run, stage_id: str, slot: WorkerSlot) -> WorkerEventRecord:
        ids = (run.job.job_id, run.attempt.attempt_id, stage_id)
        matching = [c for c in result.artifact_claims
                    if (c.job_id, c.attempt_id, c.stage_id) == ids and c.payload.get("slot") == slot.value]
        if len(matching) != 1:
            raise _Failure(ReasonCode.ARTIFACT_REJECTED)
        return matching[0]

    def _verify(self, run: _Run, assignment: Mapping[WorkerSlot, str], claim: WorkerEventRecord) -> object:
        # Claims are only trusted through verification, and only after the writer is reaped.
        return self._publisher.verify(run.job, run.attempt, assignment, claim,
                                      writers_reaped=self._launcher.writers_reaped)

    async def _download(self, run: _Run, stage: Stage, stage_id: str, proof: StageRecord | None) -> None:
        job = run.job
        if proof is not None:
            outputs = self._publisher.reuse(job, proof)
            slot = next(s for s in _MEDIA_SLOTS if s in outputs)
            run.media_name, run.media = outputs[slot]
            run.outputs = tuple(outputs)
            run.identity = _source_value(proof)
            return
        if job.request.source.kind == "direct":
            slots, media = (WorkerSlot.DOWNLOADED_MEDIA,), WorkerSlot.DOWNLOADED_MEDIA
        elif job.request.outputs.video:
            slots = (WorkerSlot.DOWNLOADED_VIDEO, WorkerSlot.DOWNLOADED_AUDIO, WorkerSlot.MERGED_VIDEO)
            media = WorkerSlot.MERGED_VIDEO
        else:
            slots, media = (WorkerSlot.DOWNLOADED_AUDIO,), WorkerSlot.DOWNLOADED_AUDIO
        assignment = self._publisher.assign(job, run.attempt, slots)
        request = self._request(run, stage_id, capability=Capability.MEDIA.value, operation="download",
                                assignment=assignment, remote=True)
        result = await self._launch(run, request, stage, stage_id)
        run.media = self._verify(run, assignment, self._claim(result, run, stage_id, media))
        run.media_name, run.outputs = assignment[media], (media,)
        run.identity = await asyncio.to_thread(self._publisher.source_identity, job)

    async def _probe(self, run: _Run, stage: Stage, stage_id: str, proof: StageRecord | None) -> None:
        job = run.job
        if run.media_name is None:  # local source: the input file itself
            run.media_name = re.split(r"[\\/]", job.request.source.input)[-1]
        if proof is not None:
            outputs = self._publisher.reuse(job, proof)
            run.facts = self._publisher.probe_facts(outputs[WorkerSlot.PROBE_METADATA][1])
            run.outputs = tuple(outputs)
            return
        assignment = self._publisher.assign(job, run.attempt, (WorkerSlot.PROBE_METADATA,))
        request = self._request(run, stage_id, capability=Capability.MEDIA.value, operation="probe",
                                assignment=assignment, input_basename=run.media_name)
        result = await self._launch(run, request, stage, stage_id)
        verified = self._verify(run, assignment, self._claim(result, run, stage_id, WorkerSlot.PROBE_METADATA))
        run.facts = self._publisher.probe_facts(verified)
        run.outputs = (WorkerSlot.PROBE_METADATA,)

    async def _transcribe(self, run: _Run, stage: Stage, stage_id: str) -> None:
        if run.facts is None or not run.facts.has_audio or run.choice is None or run.snapshot is None:
            raise _Failure(ReasonCode.CONFLICT)  # ASR only after audio was validated and provisioned
        assignment = self._publisher.assign(run.job, run.attempt, (WorkerSlot.TRANSCRIPT_SEGMENTS,))
        device, model = self._models.selection(run.choice)  # the worker rejects ``auto``
        selection = replace(run.job.request.selection, device=device, model=model)
        request = self._request(run, stage_id, capability=self._models.capability(run.choice),
                                operation="transcribe", assignment=assignment,
                                input_basename=run.media_name, snapshot=run.snapshot,
                                selection=selection)
        result = await self._launch(run, request, stage, stage_id)
        claim = self._claim(result, run, stage_id, WorkerSlot.TRANSCRIPT_SEGMENTS)
        run.segments = self._verify(run, assignment, claim)
        run.outputs = (WorkerSlot.TRANSCRIPT_SEGMENTS,)

    async def _export(self, run: _Run) -> None:
        job = run.job
        missing = tr.requested_slots(job.request) - _published(job) - {ArtifactSlot.ORIGINAL_VIDEO}
        if not missing:
            return  # the closing stage commit (``_settle``) still brings the sidecar up to date
        if run.segments is None:
            raise _Failure(ReasonCode.CONFLICT)
        exported = self._exporter.export(run.segments, frozenset(missing))
        if inspect.isawaitable(exported):
            exported = await exported
        failure: ReasonCode | None = None
        for slot in sorted(missing, key=lambda s: s.value):
            plan = await self._plan(run.job, Stage.EXPORTING)
            self._resume(run)  # never publish after a cancel or quiesce that landed during the preflight
            if not plan.admitted:
                raise _Failure(plan.reason)
            try:
                if slot not in exported:
                    raise _Failure(ReasonCode.PUBLICATION_FAILED)
                verified = self._publisher.verify_export(run.job, run.attempt, slot, exported[slot])
                artifact = await self._host_write(self._publisher.publish(run.job, run.attempt, verified, slot))
            except _Failure as error:
                failure = failure or error.reason
                continue
            except DomainValidationError as error:
                if error.reason_code in _STORAGE_REASONS:
                    raise _Failure(error.reason_code) from None
                failure = failure or error.reason_code
                continue
            decision = self._decide(run, tr.ArtifactPublished(run.job.job_id, artifact))  # never closes
            self._resume(run)  # a cancel or quiesce that landed during the copy starts no sidecar write
            try:
                # An export is claimed only once the sidecar lists it.  Once every requested slot
                # is claimed the closing stage commit can only reduce to completed (spec.md:140),
                # so a file whose sidecar write failed stays unclaimed and maps like a failed
                # publication: a storage reason halts, any other leaves the stage partial.
                await self._ensure_manifest(run, decision.job)
            except _Failure as error:
                if error.reason in _STORAGE_REASONS:
                    raise
                failure = failure or error.reason
                continue
            self._check_current(run)
            self._commit(run, decision)
            self._record_artifact(artifact)
        if failure is not None:
            raise _Failure(failure)

    def _record_artifact(self, artifact: Artifact) -> None:
        try:
            self._repo.record_artifact(artifact)
        except RepositoryError as error:
            if error.reason_code is not ReasonCode.CONFLICT:  # CONFLICT: already recorded (replay)
                raise

    async def _ensure_manifest(self, run: _Run, job: Job) -> bool:
        """Bring the sidecar up to date with ``job``'s available artifacts; True when it was written."""
        listed = frozenset(a.artifact_id for a in job.artifacts if a.available)
        if listed == run.listed:
            return False
        await self._write_manifest(run, job)
        return True

    async def _write_manifest(self, run: _Run, job: Job) -> None:
        """Sidecar manifest; it replaces itself, so it always lists every published artifact.

        ``job`` may be an uncommitted decision: the sidecar is written before a closing state
        becomes visible, so the artifacts it lists are the ones that state will claim.
        """
        artifacts = [a for a in job.artifacts if a.available]
        provenance = next((a.provenance for a in reversed(artifacts) if a.provenance is not None), None)
        try:
            await self._host_write(asyncio.to_thread(self._publisher.write_manifest, job.job_id,
                                                     run.attempt.attempt_id, artifacts, provenance))
        except Exception as error:  # noqa: BLE001
            reason = _reason(error)
            raise _Failure(reason if reason in _STORAGE_REASONS or isinstance(error, DomainValidationError)
                           else ReasonCode.PUBLICATION_FAILED) from None
        run.listed = frozenset(a.artifact_id for a in artifacts)

    # ---------------------------------------------------------------- publication

    async def _publish_video(self, run: _Run) -> bool:
        plan = await self._plan(run.job, Stage.PROBING)
        self._resume(run)
        if not plan.admitted:
            await self._halt(run, plan.reason)
            return False
        try:
            verified = run.media if run.job.request.source.kind != "local" else \
                self._publisher.verify_local_video(run.job, run.attempt)
            artifact = await self._host_write(
                self._publisher.publish(run.job, run.attempt, verified, ArtifactSlot.ORIGINAL_VIDEO))
        except Exception as error:  # noqa: BLE001
            reason = _reason(error)
            if reason in _STORAGE_REASONS:
                await self._halt(run, reason)
            else:
                await self._fail(run, reason)
            return False
        self._check_current(run)
        # Video-only, and combined silent media, close here (completed / partial no_audio): the
        # sidecar is written before that state is committed.  A failed sidecar leaves the video
        # unclaimed and maps like a failed publication (blocked on storage, otherwise failed).
        decision = await self._settle(run, tr.ArtifactPublished(run.job.job_id, artifact))
        if decision is None:
            return False
        self._record_artifact(artifact)
        if decision.job.state in _FINAL:
            return False
        try:  # the intermediate publication (ASR follows) is committed first, then listed
            await self._ensure_manifest(run, run.job)
        except _Failure as error:
            if error.reason in _STORAGE_REASONS:
                await self._halt(run, error.reason)
            else:
                await self._fail(run, error.reason)
            return False
        return True

    # ------------------------------------------------------- stop, fail, fallback

    async def _fail(self, run: _Run, reason: ReasonCode) -> None:
        self._apply(run, tr.AttemptFailed(run.job.job_id, run.attempt.attempt_id, reason))

    async def _halt(self, run: _Run, reason: ReasonCode) -> None:
        """Safe-boundary stop: reap, then blocked (nothing published) or partial, never completed.

        With every requested artifact already claimed (only the sidecar is missing) a stage or
        attempt failure would reduce to completed, so that case blocks for a recheck/retry too.
        """
        if not await self._launcher.stop(reason):
            self._apply(run, tr.WriterReaped(run.job.job_id, False, run.attempt.attempt_id))
            return
        job, attempt = run.job, run.attempt
        running = next((r for r in attempt.stage_records if r.ended_at is None), None)
        if any(a.available for a in job.artifacts) and not _all_claimed(job):
            event: tr.Event = (tr.StageFinished(job.job_id, attempt.attempt_id, running.stage_id,
                                                StageOutcome.ERROR, reason)
                               if running is not None else tr.AttemptFailed(job.job_id, attempt.attempt_id, reason))
        else:
            event = tr.Blocked(job.job_id, reason, RECHECK_ACTION, attempt.attempt_id)
        self._apply(run, event)

    async def _fallback(self, run: _Run) -> bool:
        """AutoOOM: discard, reap (already verified), release, full preflight, one lower candidate."""
        job, attempt = run.job, run.attempt
        self._publisher.discard(job, attempt)
        decision = tr.reduce(job, attempt, tr.ModelOom(job.job_id, attempt.attempt_id, self._ids()),
                             now=self._clock.now(), facts=run.facts)
        if tr.Effect.START_FALLBACK_ATTEMPT not in decision.effects:
            self._commit(run, decision)  # explicit model/device or second OOM: terminal
            return False
        self._commit(run, decision)  # releases the lease and starts the fallback attempt
        previous = run.choice
        run.facts = run.media = run.segments = run.choice = run.snapshot = None
        run.media_name, run.lower_than, run.export = None, previous, None
        plan = await self._plan(run.job, Stage.CHECKING)
        self._resume(run)  # no lease is held here; a cancel/quiesce leaves nothing to re-acquire
        if not plan.admitted or not may_acquire_lease(JobState.CHECKING, plan):
            reason = plan.reason if not plan.admitted else ReasonCode.CONFLICT
            self._apply(run, tr.Blocked(run.job.job_id, reason, RECHECK_ACTION, run.attempt.attempt_id))
            return False
        run.op_id = self._ids()
        refreshed = replace(run.job, updated_at=self._clock.now())
        self._repo.transition_with_lease(refreshed, expected_version=run.version, acquire=(run.op_id, plan))
        run.version += 1
        run.leased = True
        self._ops[run.job.job_id] = run.op_id
        run.job = refreshed
        return True


__all__ = (
    "CANCEL_ACTION",
    "METRIC_ATTRIBUTES",
    "PROGRESS_INTERVAL_SECONDS",
    "SPAN_ATTRIBUTES",
    "Clock",
    "Exporter",
    "ModelProvider",
    "Preflight",
    "Publisher",
    "Scheduler",
    "SchedulerError",
    "Telemetry",
    "WorkerLauncher",
)
