"""Pure lifecycle reducer, stage planner and outcome matrix.

Nothing here reads a clock, touches the filesystem or calls the repository.
Callers (scheduler, reconciliation) pass ``now`` in and apply the returned
``Decision`` through ``podmuncher.repository``.

Approved rules implemented here (Docs/spec.md:140-142, design-summary.md:61-121):

Stage order: "queued -> checking -> preparing(media if absent) -> downloading(remote
only) -> probing -> preparing(ASR if needed) -> transcribing -> exporting -> completed,
with video publication after probing. Skip absent stages explicitly. preparing occurs
with distinct stage IDs. Blocked requests require explicit recheck/retry; no background
retry loop. Recheck cannot create an attempt or output. Retry terminal/blocked jobs only
after admission; repeated retry key returns same attempt. Clone always creates a new
immutable job. Reuse only matching source identity/hash and stage settings. Export retry
uses validated transcript stage, never retranscribes. Cancellation records intent then
bounded terminate/kill/reap before cancelled; late attempt events cannot mutate newer
state. Stop -> interrupted/service_stopped, crash recovery -> interrupted/service_lost;
retain queue. Unreaped writer blocks scheduling and retains reservations."

Outcomes: "completed means every requested artifact is validated and published; partial
means at least one requested artifact exists and another requested result failed; failed
means no requested result exists after a terminal error. Video-only accepts a silent valid
video and never provisions ASR/GPU/model assets. Combined silent video publishes video
then partial/no_audio; transcript-only media without an audio stream is failed/no_audio.
... Audio-only plus video fails no_video."

AutoOOM: "Explicit model/device never silently substitutes. Auto OOM discards transcript
fragments, unloads/reaps, full re-admission, retries next lower candidate once, then
fails."

Unavailable persistence: "stop scheduling, retain a bounded in-memory error ... do not
report success."

Conventions
-----------
* Effects are ordered. ``RELEASE_LEASE`` may only be applied once the owned writer is
  verified reaped; a failed reap is reported with ``WriterReaped(False)`` which yields
  ``KEEP_LEASE``.
* A ``StageRecord`` whose ``ended_at`` is ``None`` is a running stage (its ``status`` is a
  placeholder ``OK`` until ``StageFinished`` closes it).
* Media facts (``MediaFacts``) are returned by the ``MediaProbed`` decision and must be
  passed back through ``reduce(..., facts=...)``; they are required once a transcript
  decision depends on them, and the reducer fails closed (CONFLICT) without them.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

from .domains import (
    ArtifactSlot,
    DomainValidationError,
    JobState,
    OutputMode,
    ReasonCode,
    Stage,
    StageOutcome,
    WorkerSlot,
    validate_utc_timestamp,
    validate_uuid_string,
)
from .requests import JobRequest
from .snapshots import Artifact, Attempt, Job, StageRecord

_TRANSCRIPT_SLOTS = frozenset((
    ArtifactSlot.TRANSCRIPT_TXT,
    ArtifactSlot.TRANSCRIPT_TIMESTAMPED_TXT,
    ArtifactSlot.TRANSCRIPT_READABLE_TXT,
    ArtifactSlot.TRANSCRIPT_SRT,
    ArtifactSlot.TRANSCRIPT_JSON,
))
_ACTIVE = frozenset((
    JobState.CHECKING, JobState.PREPARING, JobState.DOWNLOADING, JobState.PROBING,
    JobState.TRANSCRIBING, JobState.EXPORTING,
))
_FINAL = frozenset((JobState.COMPLETED, JobState.PARTIAL, JobState.FAILED, JobState.CANCELLED))
_ATTEMPT_DONE = _FINAL | {JobState.INTERRUPTED}
_CANCEL_ACTION = "cancel"
_RETRY_KEY_MAX = 200
_PERSISTENCE_REASONS = frozenset((ReasonCode.STATE_WRITE_FAILED, ReasonCode.AUDIT_LOSS))


class TransitionError(Exception):
    """Illegal transition; carries only a closed reason code."""

    def __init__(self, reason_code: ReasonCode) -> None:
        self.reason_code = reason_code if type(reason_code) is ReasonCode else ReasonCode.INVALID_INPUT
        super().__init__("Illegal transition.")


class Effect(StrEnum):
    SCHEDULE_NEXT_STAGE = "schedule_next_stage"
    RELEASE_LEASE = "release_lease"
    KEEP_LEASE = "keep_lease"
    TERMINATE_WORKER = "terminate_worker"
    START_FALLBACK_ATTEMPT = "start_fallback_attempt"
    HALT_ADMISSION = "halt_admission"
    NONE = "none"


@dataclass(frozen=True, slots=True)
class MediaFacts:
    has_audio: bool
    has_video: bool
    duration_seconds: float

    def __post_init__(self) -> None:
        if type(self.has_audio) is not bool or type(self.has_video) is not bool:
            raise DomainValidationError(ReasonCode.INVALID_INPUT)
        duration = self.duration_seconds
        if type(duration) not in (int, float) or not math.isfinite(duration) or duration < 0:
            raise DomainValidationError(ReasonCode.DURATION_UNKNOWN)


# --------------------------------------------------------------------------- events


def _uuid(value: object) -> str:
    return validate_uuid_string(value)


@dataclass(frozen=True, slots=True)
class Admitted:
    job_id: str
    attempt_id: str  # the new attempt

    def __post_init__(self) -> None:
        _uuid(self.job_id)
        _uuid(self.attempt_id)


@dataclass(frozen=True, slots=True)
class StageStarted:
    job_id: str
    attempt_id: str
    stage: Stage
    stage_id: str
    settings_hash: str
    verified_input_hashes: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _uuid(self.job_id)
        _uuid(self.attempt_id)
        _uuid(self.stage_id)
        object.__setattr__(self, "stage", Stage.parse(self.stage))
        if type(self.verified_input_hashes) not in (tuple, list):
            raise DomainValidationError(ReasonCode.INVALID_INPUT)
        object.__setattr__(self, "verified_input_hashes", tuple(tuple(p) for p in self.verified_input_hashes))


@dataclass(frozen=True, slots=True)
class StageFinished:
    job_id: str
    attempt_id: str
    stage_id: str
    outcome: StageOutcome
    reason_code: ReasonCode = ReasonCode.OK

    def __post_init__(self) -> None:
        _uuid(self.job_id)
        _uuid(self.attempt_id)
        _uuid(self.stage_id)
        object.__setattr__(self, "outcome", StageOutcome.parse(self.outcome))
        object.__setattr__(self, "reason_code", ReasonCode.parse(self.reason_code))
        if (self.outcome is StageOutcome.OK) != (self.reason_code is ReasonCode.OK):
            raise DomainValidationError(ReasonCode.INVALID_INPUT)


@dataclass(frozen=True, slots=True)
class MediaProbed:
    job_id: str
    attempt_id: str
    has_audio: bool
    has_video: bool
    duration_seconds: float

    def __post_init__(self) -> None:
        _uuid(self.job_id)
        _uuid(self.attempt_id)
        MediaFacts(self.has_audio, self.has_video, self.duration_seconds)

    @property
    def facts(self) -> MediaFacts:
        return MediaFacts(self.has_audio, self.has_video, self.duration_seconds)


@dataclass(frozen=True, slots=True)
class ArtifactPublished:
    job_id: str
    artifact: Artifact

    def __post_init__(self) -> None:
        _uuid(self.job_id)
        if type(self.artifact) is not Artifact or not self.artifact.available:
            raise DomainValidationError(ReasonCode.INVALID_INPUT)

    @property
    def attempt_id(self) -> str:
        return self.artifact.attempt_id


@dataclass(frozen=True, slots=True)
class Blocked:
    job_id: str
    reason_code: ReasonCode
    next_action: str
    attempt_id: str | None = None

    def __post_init__(self) -> None:
        _uuid(self.job_id)
        if self.attempt_id is not None:
            _uuid(self.attempt_id)
        object.__setattr__(self, "reason_code", ReasonCode.parse(self.reason_code))
        if self.reason_code is ReasonCode.OK or type(self.next_action) is not str or not self.next_action:
            raise DomainValidationError(ReasonCode.INVALID_INPUT)


@dataclass(frozen=True, slots=True)
class AttemptFailed:
    """Terminal error outside a running stage (for example a failed video publication)."""

    job_id: str
    attempt_id: str
    reason_code: ReasonCode

    def __post_init__(self) -> None:
        _uuid(self.job_id)
        _uuid(self.attempt_id)
        object.__setattr__(self, "reason_code", ReasonCode.parse(self.reason_code))
        if self.reason_code is ReasonCode.OK:
            raise DomainValidationError(ReasonCode.INVALID_INPUT)


@dataclass(frozen=True, slots=True)
class RecheckRequested:
    job_id: str

    def __post_init__(self) -> None:
        _uuid(self.job_id)


@dataclass(frozen=True, slots=True)
class RetryRequested:
    job_id: str
    attempt_id: str  # the attempt a successful retry creates
    retry_key: str

    def __post_init__(self) -> None:
        _uuid(self.job_id)
        _uuid(self.attempt_id)
        if type(self.retry_key) is not str or not 0 < len(self.retry_key) <= _RETRY_KEY_MAX:
            raise DomainValidationError(ReasonCode.INVALID_INPUT)


@dataclass(frozen=True, slots=True)
class CloneRequested:
    job_id: str
    overrides: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        _uuid(self.job_id)


@dataclass(frozen=True, slots=True)
class CancelRequested:
    job_id: str

    def __post_init__(self) -> None:
        _uuid(self.job_id)


@dataclass(frozen=True, slots=True)
class WriterReaped:
    job_id: str
    writers_reaped: bool
    attempt_id: str | None = None

    def __post_init__(self) -> None:
        _uuid(self.job_id)
        if self.attempt_id is not None:
            _uuid(self.attempt_id)
        if type(self.writers_reaped) is not bool:
            raise DomainValidationError(ReasonCode.INVALID_INPUT)


@dataclass(frozen=True, slots=True)
class ServiceStopped:
    job_id: str

    def __post_init__(self) -> None:
        _uuid(self.job_id)


@dataclass(frozen=True, slots=True)
class ServiceLost:
    job_id: str

    def __post_init__(self) -> None:
        _uuid(self.job_id)


@dataclass(frozen=True, slots=True)
class ModelOom:
    job_id: str
    attempt_id: str
    fallback_attempt_id: str | None = None  # needed only when a fallback is possible

    def __post_init__(self) -> None:
        _uuid(self.job_id)
        _uuid(self.attempt_id)
        if self.fallback_attempt_id is not None:
            _uuid(self.fallback_attempt_id)


@dataclass(frozen=True, slots=True)
class StatePersistenceFailed:
    job_id: str
    reason_code: ReasonCode = ReasonCode.STATE_WRITE_FAILED

    def __post_init__(self) -> None:
        _uuid(self.job_id)
        object.__setattr__(self, "reason_code", ReasonCode.parse(self.reason_code))
        if self.reason_code not in _PERSISTENCE_REASONS:
            raise DomainValidationError(ReasonCode.INVALID_INPUT)


Event = (
    Admitted | StageStarted | StageFinished | MediaProbed | ArtifactPublished | Blocked
    | AttemptFailed | RecheckRequested | RetryRequested | CloneRequested | CancelRequested
    | WriterReaped | ServiceStopped | ServiceLost | ModelOom | StatePersistenceFailed
)

_EVENT_TYPES: dict[str, type] = {
    "admitted": Admitted, "stage_started": StageStarted, "stage_finished": StageFinished,
    "media_probed": MediaProbed, "artifact_published": ArtifactPublished, "blocked": Blocked,
    "attempt_failed": AttemptFailed, "recheck_requested": RecheckRequested,
    "retry_requested": RetryRequested, "clone_requested": CloneRequested,
    "cancel_requested": CancelRequested, "writer_reaped": WriterReaped,
    "service_stopped": ServiceStopped, "service_lost": ServiceLost, "model_oom": ModelOom,
    "state_persistence_failed": StatePersistenceFailed,
}


def event_from_dict(data: object) -> Event:
    """Parsing entry: build a validated event from ``{"event": name, ...fields}``.

    Every enum is parsed strictly, so an invalid stage/state/reason raises
    ``DomainValidationError(INVALID_INPUT)``.
    """
    if not isinstance(data, dict) or type(data.get("event")) is not str:
        raise DomainValidationError(ReasonCode.INVALID_INPUT)
    cls = _EVENT_TYPES.get(data["event"])
    if cls is None:
        raise DomainValidationError(ReasonCode.INVALID_INPUT)
    fields = {k: v for k, v in data.items() if k != "event"}
    if cls is ArtifactPublished and isinstance(fields.get("artifact"), dict):
        fields["artifact"] = Artifact.from_dict(fields["artifact"])
    if cls is StageStarted and isinstance(fields.get("verified_input_hashes"), dict):
        fields["verified_input_hashes"] = tuple(fields["verified_input_hashes"].items())
    try:
        return cls(**fields)
    except TypeError:
        raise DomainValidationError(ReasonCode.INVALID_INPUT) from None


# ------------------------------------------------------------------------- decision


@dataclass(frozen=True, slots=True)
class Decision:
    job: Job
    attempt: Attempt | None
    effects: tuple[Effect, ...]
    ignored: bool = False
    next_stage: Stage | None = None
    superseded_attempt: Attempt | None = None
    facts: MediaFacts | None = None
    reason_code: ReasonCode | None = None  # bounded in-memory error (persistence failure)


@dataclass(frozen=True, slots=True)
class ExportRetryPlan:
    attempt_id: str
    transcribing: StageRecord
    start_stage: Stage


# ---------------------------------------------------------------------- pure helpers


def requested_slots(request: JobRequest) -> frozenset[ArtifactSlot]:
    slots: set[ArtifactSlot] = set()
    if request.outputs.video:
        slots.add(ArtifactSlot.ORIGINAL_VIDEO)
    if request.outputs.transcript:
        slots |= _TRANSCRIPT_SLOTS
    return frozenset(slots)


def outcome(
    request: JobRequest,
    published_slots: set[ArtifactSlot] | frozenset[ArtifactSlot],
    terminal_reason: ReasonCode | None,
) -> tuple[JobState, ReasonCode]:
    """Outcome matrix. Only published slots count; progress and VAD never imply completion.

    ``terminal_reason`` explains the failure of whatever is not published
    (``None`` defaults to ``publication_failed`` when something is missing).
    """
    requested = requested_slots(request)
    published = frozenset(published_slots) & requested
    if published == requested:
        return JobState.COMPLETED, ReasonCode.OK
    reason = ReasonCode.PUBLICATION_FAILED if terminal_reason in (None, ReasonCode.OK) else terminal_reason
    if published:
        return JobState.PARTIAL, reason
    return JobState.FAILED, reason


def _published(job: Job) -> frozenset[ArtifactSlot]:
    return frozenset(a.slot for a in job.artifacts if a.available)


def _plan(request: JobRequest, facts: MediaFacts | None) -> tuple[Stage, ...]:
    steps = [Stage.CHECKING, Stage.PREPARING]
    if request.source.kind != "local":
        steps.append(Stage.DOWNLOADING)
    steps.append(Stage.PROBING)
    # ASR is never planned for video-only, and only for media with a usable audio stream.
    if request.outputs.mode is not OutputMode.VIDEO and facts is not None and facts.has_audio:
        steps += [Stage.PREPARING, Stage.TRANSCRIBING, Stage.EXPORTING]
    return tuple(steps)


def _finished_ok(attempt: Attempt) -> list[StageRecord]:
    return [r for r in attempt.stage_records if r.ended_at is not None and r.status is StageOutcome.OK]


def _probed(attempt: Attempt) -> bool:
    return any(r.stage is Stage.PROBING for r in _finished_ok(attempt))


def _position(plan: Sequence[Stage], attempt: Attempt) -> int:
    pos = 0
    for record in _finished_ok(attempt):
        for index in range(pos, len(plan)):
            if plan[index] is record.stage:
                pos = index + 1
                break
        else:
            raise TransitionError(ReasonCode.CONFLICT)
    return pos


def _needs_facts(request: JobRequest, attempt: Attempt, facts: MediaFacts | None) -> None:
    if facts is None and request.outputs.transcript and _probed(attempt):
        raise TransitionError(ReasonCode.CONFLICT)


def next_stage(
    request: JobRequest,
    attempt: Attempt,
    facts: MediaFacts | None = None,
    *,
    export_plan: ExportRetryPlan | None = None,
) -> Stage | None:
    """Next stage in the approved order, or ``None`` when no stage remains.

    Local sources skip downloading; ASR preparing/transcribing only when a transcript is
    requested and the probe reported audio; video-only never plans ASR. With an
    ``export_plan`` only exporting is ever returned (never transcribing).
    """
    if export_plan is not None:
        done = any(r.stage is Stage.EXPORTING for r in _finished_ok(attempt))
        return None if done else Stage.EXPORTING
    _needs_facts(request, attempt, facts)
    plan = _plan(request, facts)
    pos = _position(plan, attempt)
    return plan[pos] if pos < len(plan) else None


def _allowed_next(request: JobRequest, attempt: Attempt, facts: MediaFacts | None) -> frozenset[Stage]:
    _needs_facts(request, attempt, facts)
    plan = _plan(request, facts)
    pos = _position(plan, attempt)
    if pos >= len(plan):
        return frozenset()
    allowed = {plan[pos]}
    if plan[pos] is Stage.PREPARING and pos + 1 < len(plan):
        allowed.add(plan[pos + 1])  # explicit skip of an absent preparing stage
    return frozenset(allowed)


def clone_request(job: Job, overrides: Mapping[str, object] | None = None) -> JobRequest:
    """Always a new immutable request; the source job is never changed."""
    return job.request.clone(dict(overrides) if overrides else None)


def reuse_allowed(stage_record: StageRecord, source_hash: str, settings_hash: str) -> bool:
    """Reuse only a finished OK stage whose verified source identity and settings match."""
    if type(stage_record) is not StageRecord or type(source_hash) is not str or type(settings_hash) is not str:
        return False
    return (
        stage_record.status is StageOutcome.OK
        and stage_record.ended_at is not None
        and stage_record.settings_hash == settings_hash
        and any(value == source_hash for _, value in stage_record.verified_input_hashes)
    )


def export_retry_plan(job: Job, attempts: Sequence[Attempt]) -> ExportRetryPlan:
    """Resume at exporting from a verified transcribing stage; never schedules transcribing."""
    if job.state is JobState.COMPLETED:
        raise TransitionError(ReasonCode.ALREADY_TERMINAL)
    if job.state in _ACTIVE or job.state is JobState.QUEUED or not job.request.outputs.transcript:
        raise TransitionError(ReasonCode.CONFLICT)
    for attempt in sorted(attempts, key=lambda a: a.ordinal, reverse=True):
        if attempt.job_id != job.job_id:
            continue
        for record in reversed(attempt.stage_records):
            if (
                record.stage is Stage.TRANSCRIBING
                and record.status is StageOutcome.OK
                and record.ended_at is not None
                and record.verified_input_hashes
                and WorkerSlot.TRANSCRIPT_SEGMENTS in record.output_slots
            ):
                return ExportRetryPlan(attempt.attempt_id, record, Stage.EXPORTING)
    raise TransitionError(ReasonCode.CONFLICT)


# --------------------------------------------------------------------- reducer core


def _touch(job: Job, now: str, **changes: object) -> Job:
    return replace(job, updated_at=now, **changes)


def _close_running(attempt: Attempt, status: StageOutcome, now: str) -> tuple[StageRecord, ...]:
    return tuple(
        replace(r, status=status, ended_at=now) if r.ended_at is None else r
        for r in attempt.stage_records
    )


def _end_attempt(attempt: Attempt, state: JobState, reason: ReasonCode, now: str,
                 stage_status: StageOutcome = StageOutcome.ERROR) -> Attempt:
    return replace(attempt, state=state, reason_code=reason, ended_at=now,
                   stage_records=_close_running(attempt, stage_status, now))


def _noop(job: Job, attempt: Attempt | None, *, ignored: bool = False,
          effects: tuple[Effect, ...] = (Effect.NONE,), facts: MediaFacts | None = None) -> Decision:
    return Decision(job, attempt, effects, ignored, facts=facts)


def _finalize(job: Job, attempt: Attempt, now: str, reason: ReasonCode | None,
              effects: tuple[Effect, ...] = (Effect.RELEASE_LEASE,)) -> Decision:
    state, final_reason = outcome(job.request, _published(job), reason)
    new_job = _touch(job, now, state=state, reason_code=final_reason, next_action=None)
    new_attempt = _end_attempt(attempt, state, final_reason, now)
    return Decision(new_job, new_attempt, effects)


def _cancel_pending(job: Job) -> bool:
    return job.next_action == _CANCEL_ACTION and job.state in _ACTIVE


def _require_active(job: Job) -> None:
    if job.state in _ACTIVE:
        return
    raise TransitionError(ReasonCode.ALREADY_TERMINAL if job.state in _FINAL else ReasonCode.CONFLICT)


def _find(attempt: Attempt, stage_id: str) -> StageRecord | None:
    return next((r for r in attempt.stage_records if r.stage_id == stage_id), None)


def _continue_or_finish(job: Job, attempt: Attempt, now: str, facts: MediaFacts | None) -> Decision:
    request = job.request
    probed = _probed(attempt)
    pending_video = request.outputs.video and ArtifactSlot.ORIGINAL_VIDEO not in _published(job)
    if probed and pending_video:
        return Decision(job, attempt, (Effect.NONE,), facts=facts)  # awaiting video publication
    nxt = next_stage(request, attempt, facts)
    if nxt is not None:
        return Decision(job, attempt, (Effect.SCHEDULE_NEXT_STAGE,), next_stage=nxt, facts=facts)
    if not probed:
        return Decision(job, attempt, (Effect.NONE,), facts=facts)
    silent = request.outputs.transcript and facts is not None and not facts.has_audio
    return _finalize(job, attempt, now, ReasonCode.NO_AUDIO if silent else None)


def _on_admitted(job: Job, attempt: Attempt | None, ev: Admitted, now: str, facts: MediaFacts | None) -> Decision:
    if job.state is not JobState.QUEUED:
        raise TransitionError(ReasonCode.CONFLICT)
    new_attempt = Attempt(ev.attempt_id, job.job_id, 1, None, None, now, None, JobState.CHECKING,
                          ReasonCode.OK, (), 0)
    new_job = _touch(job, now, state=JobState.CHECKING, reason_code=ReasonCode.OK, next_action=None,
                     attempt_id=ev.attempt_id)
    return Decision(new_job, new_attempt, (Effect.SCHEDULE_NEXT_STAGE,), next_stage=Stage.CHECKING)


def _on_stage_started(job: Job, attempt: Attempt | None, ev: StageStarted, now: str,
                      facts: MediaFacts | None) -> Decision:
    _require_active(job)
    assert attempt is not None
    if _cancel_pending(job):
        return _noop(job, attempt, ignored=True)
    if job.reason_code is ReasonCode.WRITER_UNRESOLVED:
        raise TransitionError(ReasonCode.CONFLICT)  # unreaped writer blocks scheduling
    if any(r.ended_at is None for r in attempt.stage_records):
        raise TransitionError(ReasonCode.CONFLICT)
    if any(r.stage_id == ev.stage_id for r in attempt.stage_records):
        raise TransitionError(ReasonCode.CONFLICT)  # preparing (and every stage) needs a new id
    if ev.stage not in _allowed_next(job.request, attempt, facts):
        raise TransitionError(ReasonCode.CONFLICT)
    if job.request.outputs.video and _probed(attempt) and ArtifactSlot.ORIGINAL_VIDEO not in _published(job):
        raise TransitionError(ReasonCode.CONFLICT)  # video publishes before ASR preparation
    record = StageRecord(ev.stage_id, attempt.attempt_id, ev.stage, StageOutcome.OK, now, None,
                         ev.verified_input_hashes, ev.settings_hash, ())
    state = JobState(ev.stage.value)
    new_attempt = replace(attempt, state=state, reason_code=ReasonCode.OK,
                          stage_records=attempt.stage_records + (record,))
    new_job = _touch(job, now, state=state, reason_code=ReasonCode.OK, next_action=None)
    return Decision(new_job, new_attempt, (Effect.NONE,), facts=facts)


def _on_stage_finished(job: Job, attempt: Attempt | None, ev: StageFinished, now: str,
                       facts: MediaFacts | None) -> Decision:
    _require_active(job)
    assert attempt is not None
    if _cancel_pending(job):
        return _noop(job, attempt, ignored=True)
    record = _find(attempt, ev.stage_id)
    if record is None or record.ended_at is not None or ev.outcome is StageOutcome.CANCELLED:
        raise TransitionError(ReasonCode.CONFLICT)
    closed = replace(record, status=ev.outcome, ended_at=now)
    attempt = replace(attempt, stage_records=tuple(closed if r is record else r for r in attempt.stage_records))
    if ev.outcome is StageOutcome.ERROR:
        return _finalize(job, attempt, now, ev.reason_code)
    if record.stage is Stage.EXPORTING:
        return _finalize(job, attempt, now, None)
    return _continue_or_finish(_touch(job, now), attempt, now, facts)


def _on_media_probed(job: Job, attempt: Attempt | None, ev: MediaProbed, now: str,
                     facts: MediaFacts | None) -> Decision:
    _require_active(job)
    assert attempt is not None
    if _cancel_pending(job):
        return _noop(job, attempt, ignored=True)
    if job.state is not JobState.PROBING or not any(r.stage is Stage.PROBING for r in attempt.stage_records):
        raise TransitionError(ReasonCode.CONFLICT)
    probed = ev.facts
    outputs = job.request.outputs
    if outputs.video and not probed.has_video:
        return replace(_finalize(job, attempt, now, ReasonCode.NO_VIDEO), facts=probed)
    if outputs.mode is OutputMode.TRANSCRIPT and not probed.has_audio:
        return replace(_finalize(job, attempt, now, ReasonCode.NO_AUDIO), facts=probed)
    return Decision(_touch(job, now), attempt, (Effect.NONE,), facts=probed)


def _on_artifact_published(job: Job, attempt: Attempt | None, ev: ArtifactPublished, now: str,
                           facts: MediaFacts | None) -> Decision:
    _require_active(job)
    assert attempt is not None
    slot = ev.artifact.slot
    if slot not in requested_slots(job.request) or slot in _published(job):
        raise TransitionError(ReasonCode.CONFLICT)  # unrequested slot or no-clobber replay
    if slot is ArtifactSlot.ORIGINAL_VIDEO:
        if not _probed(attempt):
            raise TransitionError(ReasonCode.CONFLICT)
    elif job.state is not JobState.EXPORTING:
        raise TransitionError(ReasonCode.CONFLICT)
    new_job = _touch(job, now, artifacts=job.artifacts + (ev.artifact,))
    if _cancel_pending(job):
        return _noop(new_job, attempt)
    if slot is ArtifactSlot.ORIGINAL_VIDEO:
        return _continue_or_finish(new_job, attempt, now, facts)
    return Decision(new_job, attempt, (Effect.NONE,), facts=facts)


def _on_blocked(job: Job, attempt: Attempt | None, ev: Blocked, now: str, facts: MediaFacts | None) -> Decision:
    if job.state in _ACTIVE and _cancel_pending(job):
        return _noop(job, attempt, ignored=True)
    if job.state not in _ACTIVE and job.state not in (JobState.QUEUED, JobState.BLOCKED):
        raise TransitionError(ReasonCode.ALREADY_TERMINAL if job.state in _FINAL else ReasonCode.CONFLICT)
    new_job = _touch(job, now, state=JobState.BLOCKED, reason_code=ev.reason_code, next_action=ev.next_action)
    new_attempt = None
    if attempt is not None:
        new_attempt = replace(attempt, state=JobState.BLOCKED, reason_code=ev.reason_code,
                              stage_records=_close_running(attempt, StageOutcome.ERROR, now))
    return Decision(new_job, new_attempt, (Effect.RELEASE_LEASE,))


def _on_attempt_failed(job: Job, attempt: Attempt | None, ev: AttemptFailed, now: str,
                       facts: MediaFacts | None) -> Decision:
    _require_active(job)
    assert attempt is not None
    if _cancel_pending(job):
        return _noop(job, attempt, ignored=True)
    return _finalize(job, attempt, now, ev.reason_code)


def _on_recheck(job: Job, attempt: Attempt | None, ev: RecheckRequested, now: str,
                facts: MediaFacts | None) -> Decision:
    if job.state in _ACTIVE or job.state is JobState.QUEUED:
        raise TransitionError(ReasonCode.CONFLICT)
    return _noop(job, attempt)  # never an attempt, output or queued work


def _on_retry(job: Job, attempt: Attempt | None, ev: RetryRequested, now: str,
              facts: MediaFacts | None) -> Decision:
    if job.state is JobState.COMPLETED:
        raise TransitionError(ReasonCode.ALREADY_TERMINAL)
    if job.state not in (JobState.BLOCKED, JobState.INTERRUPTED, JobState.PARTIAL, JobState.FAILED,
                         JobState.CANCELLED):
        raise TransitionError(ReasonCode.CONFLICT)  # queued or an attempt is active
    superseded = None
    if attempt is not None and attempt.state not in _ATTEMPT_DONE:
        superseded = _end_attempt(attempt, JobState.FAILED, attempt.reason_code, now)
    ordinal = 1 if attempt is None else attempt.ordinal + 1
    new_attempt = Attempt(ev.attempt_id, job.job_id, ordinal, None, None, now, None, JobState.CHECKING,
                          ReasonCode.OK, (), 0)
    new_job = _touch(job, now, state=JobState.CHECKING, reason_code=ReasonCode.OK, next_action=None,
                     attempt_id=ev.attempt_id)
    return Decision(new_job, new_attempt, (Effect.SCHEDULE_NEXT_STAGE,), next_stage=Stage.CHECKING,
                    superseded_attempt=superseded)


def _on_clone(job: Job, attempt: Attempt | None, ev: CloneRequested, now: str,
              facts: MediaFacts | None) -> Decision:
    return _noop(job, attempt)  # the clone is a new Job created by the caller


def _on_cancel(job: Job, attempt: Attempt | None, ev: CancelRequested, now: str,
               facts: MediaFacts | None) -> Decision:
    state = job.state
    if state in (JobState.COMPLETED, JobState.PARTIAL, JobState.FAILED):
        raise TransitionError(ReasonCode.ALREADY_TERMINAL)
    if state is JobState.CANCELLED:
        return _noop(job, attempt)  # idempotent
    if state in (JobState.QUEUED, JobState.BLOCKED, JobState.INTERRUPTED):
        new_job = _touch(job, now, state=JobState.CANCELLED, reason_code=ReasonCode.CANCELLED, next_action=None)
        new_attempt = attempt
        if attempt is not None and attempt.state not in _ATTEMPT_DONE:
            new_attempt = _end_attempt(attempt, JobState.CANCELLED, ReasonCode.CANCELLED, now,
                                       StageOutcome.CANCELLED)
        return Decision(new_job, new_attempt, (Effect.NONE,))
    assert attempt is not None
    if _cancel_pending(job) and job.reason_code is not ReasonCode.WRITER_UNRESOLVED:
        return _noop(job, attempt)  # intent already recorded
    new_job = _touch(job, now, reason_code=ReasonCode.CANCELLATION_REQUESTED, next_action=_CANCEL_ACTION)
    new_attempt = replace(attempt, reason_code=ReasonCode.CANCELLATION_REQUESTED)
    return Decision(new_job, new_attempt, (Effect.TERMINATE_WORKER,))


def _on_writer_reaped(job: Job, attempt: Attempt | None, ev: WriterReaped, now: str,
                      facts: MediaFacts | None) -> Decision:
    if not ev.writers_reaped:
        if job.state in _ACTIVE:
            new_job = _touch(job, now, reason_code=ReasonCode.WRITER_UNRESOLVED)
            return Decision(new_job, attempt, (Effect.KEEP_LEASE,))
        return _noop(job, attempt, effects=(Effect.KEEP_LEASE,))
    if _cancel_pending(job):
        assert attempt is not None
        new_job = _touch(job, now, state=JobState.CANCELLED, reason_code=ReasonCode.CANCELLED, next_action=None)
        new_attempt = _end_attempt(attempt, JobState.CANCELLED, ReasonCode.CANCELLED, now,
                                   StageOutcome.CANCELLED)
        return Decision(new_job, new_attempt, (Effect.RELEASE_LEASE,))
    if job.state in _ACTIVE and job.reason_code is ReasonCode.WRITER_UNRESOLVED:
        return Decision(_touch(job, now, reason_code=ReasonCode.OK), attempt, (Effect.RELEASE_LEASE,))
    return _noop(job, attempt, effects=(Effect.RELEASE_LEASE,))


def _interrupt(job: Job, attempt: Attempt | None, now: str, reason: ReasonCode) -> Decision:
    if job.state not in _ACTIVE:
        return _noop(job, attempt)  # queue, blocked and finished jobs are retained untouched
    assert attempt is not None
    new_job = _touch(job, now, state=JobState.INTERRUPTED, reason_code=reason, next_action=None)
    new_attempt = _end_attempt(attempt, JobState.INTERRUPTED, reason, now)
    return Decision(new_job, new_attempt, (Effect.TERMINATE_WORKER, Effect.RELEASE_LEASE))


def _on_stopped(job: Job, attempt: Attempt | None, ev: ServiceStopped, now: str,
                facts: MediaFacts | None) -> Decision:
    return _interrupt(job, attempt, now, ReasonCode.SERVICE_STOPPED)


def _on_lost(job: Job, attempt: Attempt | None, ev: ServiceLost, now: str, facts: MediaFacts | None) -> Decision:
    return _interrupt(job, attempt, now, ReasonCode.SERVICE_LOST)


def _on_model_oom(job: Job, attempt: Attempt | None, ev: ModelOom, now: str, facts: MediaFacts | None) -> Decision:
    _require_active(job)
    assert attempt is not None
    if job.state not in (JobState.PREPARING, JobState.TRANSCRIBING):
        raise TransitionError(ReasonCode.CONFLICT)
    if _cancel_pending(job):
        return _noop(job, attempt, ignored=True)
    selection = job.request.selection
    automatic = selection.model == "auto" and selection.device == "auto"
    if not (automatic and attempt.automatic_fallback_count == 0):
        decision = _finalize(job, attempt, now, ReasonCode.MODEL_OOM,
                             (Effect.TERMINATE_WORKER, Effect.RELEASE_LEASE))
        return decision
    if ev.fallback_attempt_id is None:
        raise TransitionError(ReasonCode.INVALID_INPUT)
    kept = tuple(
        a for a in job.artifacts
        if not (a.slot in _TRANSCRIPT_SLOTS and a.attempt_id == attempt.attempt_id)
    )
    superseded = _end_attempt(attempt, JobState.FAILED, ReasonCode.MODEL_OOM, now)
    new_attempt = Attempt(ev.fallback_attempt_id, job.job_id, attempt.ordinal + 1, None, None, now, None,
                          JobState.CHECKING, ReasonCode.OK, (), 1)
    new_job = _touch(job, now, state=JobState.CHECKING, reason_code=ReasonCode.OK, next_action=None,
                     attempt_id=ev.fallback_attempt_id, artifacts=kept)
    return Decision(new_job, new_attempt,
                    (Effect.TERMINATE_WORKER, Effect.RELEASE_LEASE, Effect.START_FALLBACK_ATTEMPT),
                    next_stage=Stage.CHECKING, superseded_attempt=superseded)


def _on_persistence_failed(job: Job, attempt: Attempt | None, ev: StatePersistenceFailed, now: str,
                           facts: MediaFacts | None) -> Decision:
    # Never advances the job: no completed/partial from this event.
    return Decision(job, attempt, (Effect.HALT_ADMISSION,), reason_code=ev.reason_code)


_HANDLERS = {
    Admitted: _on_admitted, StageStarted: _on_stage_started, StageFinished: _on_stage_finished,
    MediaProbed: _on_media_probed, ArtifactPublished: _on_artifact_published, Blocked: _on_blocked,
    AttemptFailed: _on_attempt_failed, RecheckRequested: _on_recheck, RetryRequested: _on_retry,
    CloneRequested: _on_clone, CancelRequested: _on_cancel, WriterReaped: _on_writer_reaped,
    ServiceStopped: _on_stopped, ServiceLost: _on_lost, ModelOom: _on_model_oom,
    StatePersistenceFailed: _on_persistence_failed,
}


def reduce(
    job: Job,
    attempt: Attempt | None,
    event: Event,
    *,
    now: str,
    facts: MediaFacts | None = None,
) -> Decision:
    """Apply one event. Pure: returns a new ``Decision`` or raises ``TransitionError``.

    Events for another job, or for an attempt other than ``job.attempt_id`` (late or
    future), are returned ``ignored`` with the job unchanged.
    """
    handler = _HANDLERS.get(type(event))
    if handler is None:
        raise TransitionError(ReasonCode.INVALID_INPUT)
    try:
        validate_utc_timestamp(now)
    except DomainValidationError:
        raise TransitionError(ReasonCode.INVALID_INPUT) from None
    if (job.attempt_id is None) != (attempt is None):
        raise TransitionError(ReasonCode.INVALID_INPUT)
    if attempt is not None and (attempt.attempt_id != job.attempt_id or attempt.job_id != job.job_id):
        raise TransitionError(ReasonCode.INVALID_INPUT)
    if event.job_id != job.job_id:
        return _noop(job, attempt, ignored=True)
    carried = getattr(event, "attempt_id", None)
    if type(event) in (Admitted, RetryRequested):
        if carried == job.attempt_id:
            return _noop(job, attempt, ignored=True)  # replay of the same attempt
    elif carried is not None and carried != job.attempt_id:
        return _noop(job, attempt, ignored=True)
    return handler(job, attempt, event, now, facts)


__all__ = (
    "Admitted",
    "ArtifactPublished",
    "AttemptFailed",
    "Blocked",
    "CancelRequested",
    "CloneRequested",
    "Decision",
    "Effect",
    "Event",
    "ExportRetryPlan",
    "MediaFacts",
    "MediaProbed",
    "ModelOom",
    "RecheckRequested",
    "RetryRequested",
    "ServiceLost",
    "ServiceStopped",
    "StageFinished",
    "StageStarted",
    "StatePersistenceFailed",
    "TransitionError",
    "WriterReaped",
    "clone_request",
    "event_from_dict",
    "export_retry_plan",
    "next_stage",
    "outcome",
    "reduce",
    "requested_slots",
    "reuse_allowed",
)
