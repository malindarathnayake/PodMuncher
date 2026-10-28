"""Pure lifecycle reducer: transition matrix, stage planner, outcome matrix, identity rules."""
import ast
import itertools
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from podmuncher import transitions as tr
from podmuncher.domains import (
    ArtifactFormat,
    ArtifactSlot,
    DomainValidationError,
    JobState,
    ReasonCode,
    Stage,
    StageOutcome,
    WorkerSlot,
)
from podmuncher.requests import JobRequest
from podmuncher.snapshots import Artifact, Attempt, Job, StageRecord
from podmuncher.transitions import (
    Admitted,
    ArtifactPublished,
    AttemptFailed,
    Blocked,
    CancelRequested,
    CloneRequested,
    Effect,
    MediaFacts,
    MediaProbed,
    ModelOom,
    RecheckRequested,
    RetryRequested,
    ServiceLost,
    ServiceStopped,
    StageFinished,
    StageStarted,
    StatePersistenceFailed,
    TransitionError,
    WriterReaped,
    reduce,
)

SHA = "a" * 64
SHB = "b" * 64
NOW = datetime(2026, 10, 12, 12, 0, tzinfo=UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
ROOT = Path(__file__).resolve().parent.parent
T_SLOTS = [
    ArtifactSlot.TRANSCRIPT_TXT, ArtifactSlot.TRANSCRIPT_TIMESTAMPED_TXT,
    ArtifactSlot.TRANSCRIPT_READABLE_TXT, ArtifactSlot.TRANSCRIPT_SRT, ArtifactSlot.TRANSCRIPT_JSON,
]
FORMATS = {
    ArtifactSlot.ORIGINAL_VIDEO: ArtifactFormat.VIDEO, ArtifactSlot.TRANSCRIPT_TXT: ArtifactFormat.TXT,
    ArtifactSlot.TRANSCRIPT_TIMESTAMPED_TXT: ArtifactFormat.TIMESTAMPED_TXT,
    ArtifactSlot.TRANSCRIPT_READABLE_TXT: ArtifactFormat.READABLE_TXT,
    ArtifactSlot.TRANSCRIPT_SRT: ArtifactFormat.SRT, ArtifactSlot.TRANSCRIPT_JSON: ArtifactFormat.JSON,
}
ACTIVE = [JobState.CHECKING, JobState.PREPARING, JobState.DOWNLOADING, JobState.PROBING,
          JobState.TRANSCRIBING, JobState.EXPORTING]
ALL_STATES = [JobState.QUEUED, JobState.BLOCKED, JobState.COMPLETED, JobState.PARTIAL,
              JobState.FAILED, JobState.CANCELLED, JobState.INTERRUPTED] + ACTIVE
FINAL_ERR = [JobState.COMPLETED, JobState.PARTIAL, JobState.FAILED]


def ident(n, tag=1):
    return f"{n:08x}-{tag:04x}-4111-8111-111111111111"


JOB_ID = ident(1)
ATT = ident(1, 2)


def make_request(*, transcript=True, video=False, model="auto", device="auto", kind="local"):
    source = "./input.mp4" if kind == "local" else "https://example.com/a.mp4"
    return JobRequest.from_dict({
        "schema_version": 1, "source": {"kind": kind, "input": source},
        "outputs": {"transcript": transcript, "video": video, "video_quality": "1080p"},
        "selection": {"device": device, "profile": "balanced", "model": model, "language": "auto"},
        "output_root": "./out",
        "limits": {"media_bytes": 100, "duration_seconds": 20, "export_bytes": 10}},
        verified_models=frozenset({"small"}))


def make_job(**overrides):
    request = overrides.pop("request", None) or make_request()
    fields = {
        "job_id": JOB_ID, "request": request, "request_hash": request.request_hash,
        "state": JobState.QUEUED, "reason_code": ReasonCode.OK, "next_action": None,
        "attempt_id": None, "created_at": NOW, "updated_at": NOW, "artifacts": (), "preflight": None,
    }
    fields.update(overrides)
    return Job(**fields)


def make_attempt(job, state=JobState.CHECKING, *, attempt_id=ATT, ordinal=1, records=(), fallback=0):
    return Attempt(attempt_id, job.job_id, ordinal, None, None, NOW, None, state, ReasonCode.OK,
                   tuple(records), fallback)


def rec(n, stage, *, ended=True, status=StageOutcome.OK, slots=(), hashes=(), attempt_id=ATT):
    return StageRecord(ident(n, 3), attempt_id, stage, status, NOW, NOW if ended else None, tuple(hashes),
                       SHA, tuple(slots))


def art(slot, n=1, attempt_id=ATT):
    return Artifact(ident(n, 4), slot, FORMATS[slot], 5, SHA, True, attempt_id, None)


HISTORY = {
    JobState.CHECKING: [(Stage.CHECKING, False)],
    JobState.PREPARING: [(Stage.CHECKING, True), (Stage.PREPARING, False)],
    JobState.DOWNLOADING: [(Stage.CHECKING, True), (Stage.PREPARING, True), (Stage.DOWNLOADING, False)],
    JobState.PROBING: [(Stage.CHECKING, True), (Stage.PREPARING, True), (Stage.DOWNLOADING, True),
                       (Stage.PROBING, False)],
    JobState.TRANSCRIBING: [(Stage.CHECKING, True), (Stage.PREPARING, True), (Stage.DOWNLOADING, True),
                            (Stage.PROBING, True), (Stage.PREPARING, True), (Stage.TRANSCRIBING, False)],
    JobState.EXPORTING: [(Stage.CHECKING, True), (Stage.PREPARING, True), (Stage.DOWNLOADING, True),
                         (Stage.PROBING, True), (Stage.PREPARING, True), (Stage.TRANSCRIBING, True),
                         (Stage.EXPORTING, False)],
}


def at_state(state, **req):
    """A consistent (job, attempt) pair sitting in ``state``."""
    request = make_request(kind="direct", **req)
    if state is JobState.QUEUED:
        return make_job(request=request), None
    history = HISTORY.get(state, [(Stage.CHECKING, True)])
    records = [rec(i + 1, stage, ended=ended) for i, (stage, ended) in enumerate(history)]
    job = make_job(request=request, state=state, attempt_id=ATT)
    attempt = make_attempt(job, state, records=records)
    return job, attempt


def apply_events(job, attempt, events, facts=None):
    effects = []
    for event in events:
        decision = reduce(job, attempt, event, now=NOW, facts=facts)
        job, attempt = decision.job, decision.attempt
        facts = decision.facts or facts
        effects.append(decision.effects)
    return job, attempt, effects


class Driver:
    """Drives one job through stage events with deterministic stage ids."""

    def __init__(self, request=None, *, attempt_id=ATT, admit=True):
        self.job = make_job(request=request or make_request())
        self.attempt = None
        self.facts = None
        self.counter = itertools.count(10)
        self.last = None
        if admit:
            self.send(Admitted(JOB_ID, attempt_id))

    def send(self, event):
        decision = reduce(self.job, self.attempt, event, now=NOW, facts=self.facts)
        self.job, self.attempt = decision.job, decision.attempt
        self.facts = decision.facts or self.facts
        self.last = decision
        return decision

    def start(self, stage):
        sid = ident(next(self.counter), 5)
        self.send(StageStarted(JOB_ID, self.job.attempt_id, stage, sid, SHA))
        return sid

    def finish(self, sid, outcome=StageOutcome.OK, reason=ReasonCode.OK):
        return self.send(StageFinished(JOB_ID, self.job.attempt_id, sid, outcome, reason))

    def stage(self, stage):
        return self.finish(self.start(stage))

    def probe(self, audio=True, video=True):
        return self.send(MediaProbed(JOB_ID, self.job.attempt_id, audio, video, 5.0))

    def publish(self, slot, n=None):
        n = n or next(self.counter)
        return self.send(ArtifactPublished(JOB_ID, art(slot, n, self.job.attempt_id)))

    def to_probed(self, audio=True, video=True):
        for stage in (Stage.CHECKING, Stage.PREPARING):
            self.stage(stage)
        if self.job.request.source.kind != "local":
            self.stage(Stage.DOWNLOADING)
        sid = self.start(Stage.PROBING)
        self.probe(audio, video)
        if self.job.state is JobState.PROBING:
            self.finish(sid)

    def to_exporting(self):
        """Audio present; video (if requested) published; stops with exporting running."""
        self.to_probed()
        if self.job.request.outputs.video:
            self.publish(ArtifactSlot.ORIGINAL_VIDEO)
        self.stage(Stage.PREPARING)
        self.stage(Stage.TRANSCRIBING)
        return self.start(Stage.EXPORTING)


def state_of(driver):
    return driver.job.state, driver.job.reason_code


# ------------------------------------------------------------------ matrix (state x event)

def _events(job):
    new = ident(77, 2)
    cur = job.attempt_id or ident(99, 2)
    return {
        "cancel": CancelRequested(job.job_id),
        "retry": RetryRequested(job.job_id, new, "key-1"),
        "recheck": RecheckRequested(job.job_id),
        "stopped": ServiceStopped(job.job_id),
        "lost": ServiceLost(job.job_id),
        "persist": StatePersistenceFailed(job.job_id),
        "blocked": Blocked(job.job_id, ReasonCode.STORAGE_INSUFFICIENT, "free_space"),
        "clone": CloneRequested(job.job_id),
        "reaped_true": WriterReaped(job.job_id, True),
        "reaped_false": WriterReaped(job.job_id, False),
        "oom": ModelOom(job.job_id, cur, ident(78, 2)),
        "failed": AttemptFailed(job.job_id, cur, ReasonCode.WORKER_EXIT_UNEXPECTED),
        "admitted": Admitted(job.job_id, new),
    }


def _expected(state, name):
    """Independently written expectations: ("state", s) | ("error", reason) | ("ignored",)."""
    active = state in ACTIVE
    terminal_err = state in FINAL_ERR or state is JobState.CANCELLED
    already = ("error", ReasonCode.ALREADY_TERMINAL)
    conflict = ("error", ReasonCode.CONFLICT)
    same = ("state", state)
    table = {
        "cancel": (already if state in FINAL_ERR else ("state", JobState.CANCELLED)
                   if state in (JobState.QUEUED, JobState.BLOCKED, JobState.INTERRUPTED, JobState.CANCELLED)
                   else same),
        "retry": (already if state is JobState.COMPLETED else conflict
                  if state is JobState.QUEUED or active else ("state", JobState.CHECKING)),
        "recheck": conflict if (state is JobState.QUEUED or active) else same,
        "stopped": ("state", JobState.INTERRUPTED) if active else same,
        "lost": ("state", JobState.INTERRUPTED) if active else same,
        "persist": same,
        "blocked": (already if terminal_err else conflict if state is JobState.INTERRUPTED
                    else ("state", JobState.BLOCKED)),
        "clone": same,
        "reaped_true": same,
        "reaped_false": same,
        "oom": (("ignored",) if state is JobState.QUEUED
                else already if terminal_err
                else ("state", JobState.CHECKING) if state in (JobState.PREPARING, JobState.TRANSCRIBING)
                else conflict),
        "failed": (("ignored",) if state is JobState.QUEUED
                   else already if terminal_err
                   else ("state", JobState.FAILED) if active else conflict),
        "admitted": ("state", JobState.CHECKING) if state is JobState.QUEUED else conflict,
    }
    return table[name]


@pytest.mark.parametrize("state", ALL_STATES)
@pytest.mark.parametrize("name", sorted(_events(make_job())))
def test_transition_matrix(state, name):
    job, attempt = at_state(state)
    event = _events(job)[name]
    kind, *detail = _expected(state, name)
    if kind == "error":
        with pytest.raises(TransitionError) as caught:
            reduce(job, attempt, event, now=NOW)
        assert caught.value.reason_code is detail[0]
        return
    decision = reduce(job, attempt, event, now=NOW)
    if kind == "ignored":
        assert decision.ignored and decision.job is job and decision.attempt is attempt
    else:
        assert not decision.ignored
        assert decision.job.state is detail[0]


@pytest.mark.parametrize("state", FINAL_ERR + [JobState.CANCELLED])
def test_cancel_on_terminal_states(state):
    job, attempt = at_state(state)
    event = CancelRequested(job.job_id)
    if state is JobState.CANCELLED:
        decision = reduce(job, attempt, event, now=NOW)
        assert decision.job is job and decision.effects == (Effect.NONE,)
    else:
        with pytest.raises(TransitionError) as caught:
            reduce(job, attempt, event, now=NOW)
        assert caught.value.reason_code is ReasonCode.ALREADY_TERMINAL


def test_terminal_attempt_events_never_mutate():
    for state in FINAL_ERR + [JobState.CANCELLED]:
        job, attempt = at_state(state)
        for event in (StageStarted(job.job_id, ATT, Stage.PROBING, ident(50, 5), SHA),
                      StageFinished(job.job_id, ATT, ident(1, 3), StageOutcome.OK),
                      MediaProbed(job.job_id, ATT, True, True, 1.0),
                      ArtifactPublished(job.job_id, art(ArtifactSlot.TRANSCRIPT_TXT))):
            with pytest.raises(TransitionError) as caught:
                reduce(job, attempt, event, now=NOW)
            assert caught.value.reason_code is ReasonCode.ALREADY_TERMINAL


# ------------------------------------------------------------------------ outcome matrix

def _request(mode):
    return make_request(transcript=mode != "video", video=mode != "transcript")


@pytest.mark.parametrize("mode", ["transcript", "video_transcript", "video"])
@pytest.mark.parametrize("audio", [True, False])
def test_output_modes_by_audio(mode, audio):
    d = Driver(_request(mode))
    d.to_probed(audio=audio)
    request = d.job.request
    if mode == "transcript":
        if audio:
            assert d.job.state is JobState.PROBING
            assert tr.next_stage(request, d.attempt, d.facts) is Stage.PREPARING
        else:
            assert state_of(d) == (JobState.FAILED, ReasonCode.NO_AUDIO)
        return
    if mode == "video":
        d.publish(ArtifactSlot.ORIGINAL_VIDEO)
        assert state_of(d) == (JobState.COMPLETED, ReasonCode.OK)
        assert not any(r.stage is Stage.TRANSCRIBING for r in d.attempt.stage_records)
        return
    d.publish(ArtifactSlot.ORIGINAL_VIDEO)
    if audio:
        assert d.job.state is JobState.PROBING
        assert d.last.next_stage is Stage.PREPARING and d.last.effects == (Effect.SCHEDULE_NEXT_STAGE,)
    else:
        assert state_of(d) == (JobState.PARTIAL, ReasonCode.NO_AUDIO)


def test_video_only_never_plans_asr_even_with_audio():
    d = Driver(_request("video"))
    d.to_probed(audio=True)
    assert tr.next_stage(d.job.request, d.attempt, d.facts) is None
    d.publish(ArtifactSlot.ORIGINAL_VIDEO)
    assert d.job.state is JobState.COMPLETED
    assert {r.stage for r in d.attempt.stage_records} == {Stage.CHECKING, Stage.PREPARING, Stage.PROBING}


@pytest.mark.parametrize("mode", ["transcript", "video_transcript", "video"])
def test_audio_only_with_video_requested_fails_no_video(mode):
    d = Driver(_request(mode))
    for stage in (Stage.CHECKING, Stage.PREPARING):
        d.stage(stage)
    d.start(Stage.PROBING)
    d.probe(audio=True, video=False)
    if mode == "transcript":
        assert d.job.state is JobState.PROBING  # no video requested: audio-only is fine
    else:
        assert state_of(d) == (JobState.FAILED, ReasonCode.NO_VIDEO)


def test_local_skips_download_and_remote_does_not():
    for kind, expected in (("local", [Stage.CHECKING, Stage.PREPARING, Stage.PROBING]),
                           ("direct", [Stage.CHECKING, Stage.PREPARING, Stage.DOWNLOADING, Stage.PROBING])):
        request = make_request(kind=kind)
        d = Driver(request)
        seen = []
        while True:
            stage = tr.next_stage(request, d.attempt, None)
            if stage is Stage.PROBING:
                seen.append(stage)
                break
            seen.append(stage)
            d.stage(stage)
        assert seen == expected


def test_absent_preparing_stage_can_be_skipped_but_not_others():
    d = Driver()
    d.stage(Stage.CHECKING)
    sid = d.start(Stage.PROBING)  # media runtime present: preparing skipped explicitly
    assert d.job.state is JobState.PROBING
    assert sid
    with pytest.raises(TransitionError) as caught:
        Driver().start(Stage.PROBING)  # checking cannot be skipped
    assert caught.value.reason_code is ReasonCode.CONFLICT


def test_video_survives_failed_asr_as_partial():
    d = Driver(_request("video_transcript"))
    d.to_probed()
    d.publish(ArtifactSlot.ORIGINAL_VIDEO)
    d.stage(Stage.PREPARING)
    sid = d.start(Stage.TRANSCRIBING)
    d.finish(sid, StageOutcome.ERROR, ReasonCode.WORKER_EXIT_UNEXPECTED)
    assert state_of(d) == (JobState.PARTIAL, ReasonCode.WORKER_EXIT_UNEXPECTED)
    assert [a.slot for a in d.job.artifacts] == [ArtifactSlot.ORIGINAL_VIDEO]
    assert d.last.effects == (Effect.RELEASE_LEASE,)


def test_video_published_before_asr_preparation():
    d = Driver(_request("video_transcript"))
    d.to_probed()
    with pytest.raises(TransitionError) as caught:
        d.start(Stage.PREPARING)
    assert caught.value.reason_code is ReasonCode.CONFLICT


@pytest.mark.parametrize("with_video", [False, True])
@pytest.mark.parametrize("published", [set(c) for r in range(6) for c in itertools.combinations(T_SLOTS, r)],
                         ids=lambda s: f"{len(s)}pub")
def test_export_failure_subsets_through_reducer(with_video, published):
    if len(published) == 5:
        return  # full set published is the completed case, covered below
    d = Driver(_request("video_transcript" if with_video else "transcript"))
    sid = d.to_exporting()
    for slot in sorted(published, key=lambda s: s.value):
        d.publish(slot)
    d.finish(sid, StageOutcome.ERROR, ReasonCode.ARTIFACT_REJECTED)
    if with_video or published:
        assert state_of(d) == (JobState.PARTIAL, ReasonCode.ARTIFACT_REJECTED)
    else:
        assert state_of(d) == (JobState.FAILED, ReasonCode.ARTIFACT_REJECTED)


@pytest.mark.parametrize("with_video", [False, True])
def test_full_export_set_completes_only_when_every_slot_published(with_video):
    d = Driver(_request("video_transcript" if with_video else "transcript"))
    sid = d.to_exporting()
    for slot in T_SLOTS:
        d.publish(slot)
    assert d.job.state is JobState.EXPORTING  # publication alone never completes
    d.finish(sid)
    assert state_of(d) == (JobState.COMPLETED, ReasonCode.OK)
    assert d.last.effects == (Effect.RELEASE_LEASE,)


def test_exporting_ok_with_missing_artifacts_is_never_completed():
    d = Driver(_request("transcript"))
    sid = d.to_exporting()
    d.publish(ArtifactSlot.TRANSCRIPT_TXT)
    d.finish(sid)
    assert state_of(d) == (JobState.PARTIAL, ReasonCode.PUBLICATION_FAILED)


def test_outcome_function_all_subsets():
    for video, transcript in ((True, False), (False, True), (True, True)):
        request = make_request(video=video, transcript=transcript)
        requested = tr.requested_slots(request)
        slots = sorted(requested, key=lambda s: s.value)
        for r in range(len(slots) + 1):
            for subset in itertools.combinations(slots, r):
                state, reason = tr.outcome(request, set(subset), ReasonCode.WORKER_EXIT_UNEXPECTED)
                if len(subset) == len(slots):
                    assert (state, reason) == (JobState.COMPLETED, ReasonCode.OK)
                elif subset:
                    assert (state, reason) == (JobState.PARTIAL, ReasonCode.WORKER_EXIT_UNEXPECTED)
                else:
                    assert (state, reason) == (JobState.FAILED, ReasonCode.WORKER_EXIT_UNEXPECTED)
    # Slots that were not requested never count.
    video_only = make_request(video=True, transcript=False)
    assert tr.outcome(video_only, {ArtifactSlot.TRANSCRIPT_TXT}, None)[0] is JobState.FAILED
    # Missing reason defaults to publication_failed, never to completion.
    assert tr.outcome(make_request(), set(), None) == (JobState.FAILED, ReasonCode.PUBLICATION_FAILED)


# ---------------------------------------------------------- retry, cancel, identity

def test_duplicate_retry_returns_same_attempt_and_active_conflicts():
    job, attempt = at_state(JobState.FAILED)
    new = ident(77, 2)
    event = RetryRequested(job.job_id, new, "key-1")
    first = reduce(job, attempt, event, now=NOW)
    assert first.job.state is JobState.CHECKING and first.job.attempt_id == new
    assert first.attempt.ordinal == 2 and first.attempt.automatic_fallback_count == 0
    assert first.effects == (Effect.SCHEDULE_NEXT_STAGE,) and first.next_stage is Stage.CHECKING
    replay = reduce(first.job, first.attempt, event, now=NOW)
    assert replay.ignored and replay.job is first.job
    other = RetryRequested(job.job_id, ident(78, 2), "key-2")
    with pytest.raises(TransitionError) as caught:
        reduce(first.job, first.attempt, other, now=NOW)
    assert caught.value.reason_code is ReasonCode.CONFLICT


def test_retry_of_blocked_job_supersedes_blocked_attempt():
    job, attempt = at_state(JobState.CHECKING)
    blocked = reduce(job, attempt, Blocked(JOB_ID, ReasonCode.DOCKER_UNAVAILABLE, "start_docker"), now=NOW)
    assert blocked.job.state is JobState.BLOCKED and blocked.job.next_action == "start_docker"
    retry = reduce(blocked.job, blocked.attempt, RetryRequested(JOB_ID, ident(77, 2), "k"), now=NOW)
    assert retry.attempt.ordinal == 2
    assert retry.superseded_attempt.attempt_id == ATT and retry.superseded_attempt.state is JobState.FAILED


def test_duplicate_cancel_is_idempotent_and_cancels_only_after_reap():
    job, attempt = at_state(JobState.TRANSCRIBING)
    first = reduce(job, attempt, CancelRequested(JOB_ID), now=NOW)
    assert first.job.state is JobState.TRANSCRIBING
    assert first.job.reason_code is ReasonCode.CANCELLATION_REQUESTED
    assert first.effects == (Effect.TERMINATE_WORKER,)
    second = reduce(first.job, first.attempt, CancelRequested(JOB_ID), now=NOW)
    assert second.job == first.job and second.effects == (Effect.NONE,)
    unresolved = reduce(second.job, second.attempt, WriterReaped(JOB_ID, False), now=NOW)
    assert unresolved.effects == (Effect.KEEP_LEASE,)
    assert unresolved.job.reason_code is ReasonCode.WRITER_UNRESOLVED
    assert unresolved.job.state is JobState.TRANSCRIBING
    again = reduce(unresolved.job, unresolved.attempt, CancelRequested(JOB_ID), now=NOW)
    assert again.effects == (Effect.TERMINATE_WORKER,)
    reaped = reduce(again.job, again.attempt, WriterReaped(JOB_ID, True), now=NOW)
    assert (reaped.job.state, reaped.job.reason_code) == (JobState.CANCELLED, ReasonCode.CANCELLED)
    assert reaped.effects == (Effect.RELEASE_LEASE,) and reaped.attempt.state is JobState.CANCELLED
    final = reduce(reaped.job, reaped.attempt, CancelRequested(JOB_ID), now=NOW)
    assert final.job == reaped.job


def test_cancel_keeps_published_artifacts_and_ignores_late_stage_events():
    job, attempt = at_state(JobState.EXPORTING)
    job = replace(job, artifacts=(art(ArtifactSlot.TRANSCRIPT_TXT),))
    pending = reduce(job, attempt, CancelRequested(JOB_ID), now=NOW)
    late = reduce(pending.job, pending.attempt,
                  StageStarted(JOB_ID, ATT, Stage.EXPORTING, ident(60, 5), SHA), now=NOW)
    assert late.ignored
    done = reduce(pending.job, pending.attempt, WriterReaped(JOB_ID, True), now=NOW)
    assert done.job.state is JobState.CANCELLED and len(done.job.artifacts) == 1


def test_late_and_future_attempt_events_are_ignored():
    job, attempt = at_state(JobState.TRANSCRIBING)
    other = ident(5, 2)
    for event in (StageFinished(JOB_ID, other, ident(6, 3), StageOutcome.OK),
                  StageStarted(JOB_ID, other, Stage.EXPORTING, ident(60, 5), SHA),
                  MediaProbed(JOB_ID, other, False, False, 1.0),
                  ArtifactPublished(JOB_ID, art(ArtifactSlot.TRANSCRIPT_TXT, 9, other)),
                  AttemptFailed(JOB_ID, other, ReasonCode.WORKER_EXIT_UNEXPECTED),
                  WriterReaped(JOB_ID, True, other),
                  ModelOom(JOB_ID, other, ident(8, 2)),
                  StageFinished(ident(2), ATT, ident(6, 3), StageOutcome.OK)):  # other job
        decision = reduce(job, attempt, event, now=NOW)
        assert decision.ignored and decision.job is job and decision.attempt is attempt


def test_old_attempt_event_cannot_mutate_newer_attempt():
    job, attempt = at_state(JobState.FAILED)
    retry = reduce(job, attempt, RetryRequested(JOB_ID, ident(77, 2), "k"), now=NOW)
    stale = StageFinished(JOB_ID, ATT, ident(1, 3), StageOutcome.ERROR, ReasonCode.WORKER_EXIT_UNEXPECTED)
    decision = reduce(retry.job, retry.attempt, stale, now=NOW)
    assert decision.ignored and decision.job is retry.job


def test_future_event_on_queued_job_is_ignored():
    job, _ = at_state(JobState.QUEUED)
    decision = reduce(job, None, StageStarted(JOB_ID, ATT, Stage.CHECKING, ident(3, 5), SHA), now=NOW)
    assert decision.ignored and decision.job is job


def test_preparing_stage_ids_are_distinct():
    d = Driver()
    d.stage(Stage.CHECKING)
    first = d.start(Stage.PREPARING)
    d.finish(first)
    sid = d.start(Stage.PROBING)
    d.probe()
    d.finish(sid)
    # ASR preparing must use a new id; reusing the media preparing id is a conflict.
    with pytest.raises(TransitionError) as caught:
        d.send(StageStarted(JOB_ID, ATT, Stage.PREPARING, first, SHA))
    assert caught.value.reason_code is ReasonCode.CONFLICT
    second = d.start(Stage.PREPARING)
    assert second != first
    ids = [r.stage_id for r in d.attempt.stage_records if r.stage is Stage.PREPARING]
    assert ids == [first, second]


def test_stage_rules_conflicts():
    d = Driver()
    sid = d.start(Stage.CHECKING)
    with pytest.raises(TransitionError):  # a stage is still running
        d.start(Stage.PREPARING)
    with pytest.raises(TransitionError):  # unknown stage id
        d.finish(ident(88, 5))
    d.finish(sid)
    with pytest.raises(TransitionError):  # already finished
        d.finish(sid)
    with pytest.raises(TransitionError):
        d.send(StageFinished(JOB_ID, ATT, d.start(Stage.PREPARING), StageOutcome.CANCELLED,
                                  ReasonCode.CANCELLED))


def test_unreaped_writer_blocks_scheduling():
    job, attempt = at_state(JobState.CHECKING)
    job = replace(job, attempt_id=ATT)
    unresolved = reduce(job, attempt, WriterReaped(JOB_ID, False), now=NOW)
    assert unresolved.effects == (Effect.KEEP_LEASE,)
    attempt2 = replace(unresolved.attempt, stage_records=tuple(
        replace(r, ended_at=NOW) for r in unresolved.attempt.stage_records))
    with pytest.raises(TransitionError):
        reduce(unresolved.job, attempt2, StageStarted(JOB_ID, ATT, Stage.PREPARING, ident(70, 5), SHA), now=NOW)
    cleared = reduce(unresolved.job, unresolved.attempt, WriterReaped(JOB_ID, True), now=NOW)
    assert cleared.job.reason_code is ReasonCode.OK and cleared.effects == (Effect.RELEASE_LEASE,)


# ------------------------------------------------------- recheck / reuse / clone / export

def test_recheck_creates_no_attempt_or_output_and_does_not_schedule():
    job, attempt = at_state(JobState.CHECKING)
    blocked = reduce(job, attempt, Blocked(JOB_ID, ReasonCode.STORAGE_UNKNOWN, "recheck"), now=NOW)
    for state_job, state_attempt in ((blocked.job, blocked.attempt), at_state(JobState.FAILED),
                                     at_state(JobState.COMPLETED)):
        decision = reduce(state_job, state_attempt, RecheckRequested(JOB_ID), now=NOW)
        assert decision.job is state_job and decision.attempt is state_attempt
        assert decision.effects == (Effect.NONE,) and decision.next_stage is None
        assert decision.job.attempt_id == state_job.attempt_id and decision.job.artifacts == state_job.artifacts


def test_blocked_never_auto_schedules():
    job, attempt = at_state(JobState.PREPARING)
    blocked = reduce(job, attempt, Blocked(JOB_ID, ReasonCode.QUOTA_EXCEEDED, "free_space"), now=NOW)
    assert blocked.effects == (Effect.RELEASE_LEASE,) and blocked.next_stage is None
    for event in (StageStarted(JOB_ID, ATT, Stage.DOWNLOADING, ident(70, 5), SHA),
                  StageFinished(JOB_ID, ATT, ident(2, 3), StageOutcome.OK),
                  RecheckRequested(JOB_ID), ServiceStopped(JOB_ID), ServiceLost(JOB_ID),
                  WriterReaped(JOB_ID, True)):
        try:
            decision = reduce(blocked.job, blocked.attempt, event, now=NOW)
        except TransitionError as error:
            assert error.reason_code is ReasonCode.CONFLICT
            continue
        assert decision.job.state is JobState.BLOCKED
        assert Effect.SCHEDULE_NEXT_STAGE not in decision.effects and decision.next_stage is None


def test_reuse_allowed_requires_ok_matching_source_and_settings():
    ok = rec(1, Stage.TRANSCRIBING, hashes=(("source", SHA),))
    assert tr.reuse_allowed(ok, SHA, SHA)
    assert not tr.reuse_allowed(ok, SHB, SHA)
    assert not tr.reuse_allowed(ok, SHA, SHB)
    assert not tr.reuse_allowed(replace(ok, status=StageOutcome.ERROR), SHA, SHA)
    assert not tr.reuse_allowed(replace(ok, status=StageOutcome.CANCELLED), SHA, SHA)
    assert not tr.reuse_allowed(rec(2, Stage.TRANSCRIBING, ended=False, hashes=(("source", SHA),)), SHA, SHA)
    assert not tr.reuse_allowed(rec(3, Stage.TRANSCRIBING), SHA, SHA)


def test_clone_is_new_immutable_request():
    job, _ = at_state(JobState.COMPLETED)
    same = tr.clone_request(job, None)
    changed = tr.clone_request(job, {"outputs": {"video": True}})
    assert same is not job.request and same.request_hash == job.request.request_hash
    assert changed.request_hash != job.request.request_hash and changed.outputs.video
    assert job.request.outputs.video is False  # the source request is untouched
    decision = reduce(job, make_attempt(job, JobState.COMPLETED), CloneRequested(JOB_ID, {"x": 1}), now=NOW)
    assert decision.job is job


def _transcribed_attempt(job, *, status=StageOutcome.OK, hashes=(("audio", SHA),), slots=(WorkerSlot.TRANSCRIPT_SEGMENTS,)):
    return make_attempt(job, JobState.FAILED, records=[
        rec(1, Stage.CHECKING), rec(2, Stage.TRANSCRIBING, status=status, hashes=hashes, slots=slots),
        rec(3, Stage.EXPORTING, status=StageOutcome.ERROR)])


def test_export_retry_reuses_transcript_and_never_transcribes():
    job = make_job(state=JobState.FAILED, attempt_id=ATT)
    attempt = _transcribed_attempt(job)
    plan = tr.export_retry_plan(job, [attempt])
    assert plan.start_stage is Stage.EXPORTING and plan.transcribing.stage_id == ident(2, 3)
    assert plan.attempt_id == ATT
    new_attempt = make_attempt(job, JobState.CHECKING, attempt_id=ident(9, 2), ordinal=2)
    assert tr.next_stage(job.request, new_attempt, None, export_plan=plan) is Stage.EXPORTING
    done = replace(new_attempt, stage_records=(replace(rec(4, Stage.EXPORTING), attempt_id=new_attempt.attempt_id),))
    assert tr.next_stage(job.request, done, None, export_plan=plan) is None


@pytest.mark.parametrize("kwargs", [
    {"status": StageOutcome.ERROR}, {"hashes": ()}, {"slots": ()},
])
def test_export_retry_refuses_without_verified_transcript(kwargs):
    job = make_job(state=JobState.FAILED, attempt_id=ATT)
    with pytest.raises(TransitionError) as caught:
        tr.export_retry_plan(job, [_transcribed_attempt(job, **kwargs)])
    assert caught.value.reason_code is ReasonCode.CONFLICT


def test_export_retry_refusals_by_state():
    completed = make_job(state=JobState.COMPLETED, attempt_id=ATT)
    with pytest.raises(TransitionError) as caught:
        tr.export_retry_plan(completed, [_transcribed_attempt(completed)])
    assert caught.value.reason_code is ReasonCode.ALREADY_TERMINAL
    active = make_job(state=JobState.EXPORTING, attempt_id=ATT)
    with pytest.raises(TransitionError):
        tr.export_retry_plan(active, [_transcribed_attempt(active)])
    foreign = make_job(state=JobState.FAILED, attempt_id=ATT)
    other = replace(_transcribed_attempt(foreign), job_id=ident(2))
    with pytest.raises(TransitionError):
        tr.export_retry_plan(foreign, [other])
    with pytest.raises(TransitionError):
        tr.export_retry_plan(foreign, [])


# ------------------------------------------------------------------------------ autoOOM

def _in_transcribing(request, *, fallback=0, attempt_id=ATT, artifacts=()):
    job = make_job(request=request, state=JobState.TRANSCRIBING, attempt_id=attempt_id, artifacts=artifacts)
    attempt = make_attempt(job, JobState.TRANSCRIBING, attempt_id=attempt_id, fallback=fallback,
                           records=[rec(1, Stage.CHECKING, attempt_id=attempt_id),
                                   rec(2, Stage.TRANSCRIBING, ended=False, attempt_id=attempt_id)])
    return job, attempt


def test_auto_oom_exactly_one_fallback_then_fails():
    request = make_request(video=True)
    video = art(ArtifactSlot.ORIGINAL_VIDEO, 20)
    fragment = art(ArtifactSlot.TRANSCRIPT_TXT, 21)
    job, attempt = _in_transcribing(request, artifacts=(video, fragment))
    first = reduce(job, attempt, ModelOom(JOB_ID, ATT, ident(9, 2)), now=NOW)
    assert first.effects == (Effect.TERMINATE_WORKER, Effect.RELEASE_LEASE, Effect.START_FALLBACK_ATTEMPT)
    assert first.attempt.automatic_fallback_count == 1 and first.attempt.ordinal == 2
    assert first.job.state is JobState.CHECKING and first.job.attempt_id == ident(9, 2)
    assert first.job.artifacts == (video,)  # transcript fragments discarded, video kept
    assert first.superseded_attempt.state is JobState.FAILED
    assert first.superseded_attempt.reason_code is ReasonCode.MODEL_OOM
    # Second OOM on the fallback attempt is terminal; the published video keeps it partial.
    job2, attempt2 = _in_transcribing(request, fallback=1, attempt_id=ident(9, 2), artifacts=(video,))
    second = reduce(job2, attempt2, ModelOom(JOB_ID, ident(9, 2), ident(10, 2)), now=NOW)
    assert Effect.START_FALLBACK_ATTEMPT not in second.effects
    assert (second.job.state, second.job.reason_code) == (JobState.PARTIAL, ReasonCode.MODEL_OOM)


def test_second_oom_without_video_fails():
    job, attempt = _in_transcribing(make_request(), fallback=1)
    decision = reduce(job, attempt, ModelOom(JOB_ID, ATT), now=NOW)
    assert (decision.job.state, decision.job.reason_code) == (JobState.FAILED, ReasonCode.MODEL_OOM)
    assert decision.effects == (Effect.TERMINATE_WORKER, Effect.RELEASE_LEASE)


@pytest.mark.parametrize("model,device", [("small", "auto"), ("auto", "cpu"), ("small", "cuda")])
def test_explicit_model_or_device_never_substitutes(model, device):
    job, attempt = _in_transcribing(make_request(model=model, device=device))
    decision = reduce(job, attempt, ModelOom(JOB_ID, ATT, ident(9, 2)), now=NOW)
    assert Effect.START_FALLBACK_ATTEMPT not in decision.effects
    assert decision.attempt.automatic_fallback_count == 0
    assert (decision.job.state, decision.job.reason_code) == (JobState.FAILED, ReasonCode.MODEL_OOM)


def test_auto_oom_requires_fallback_attempt_id():
    job, attempt = _in_transcribing(make_request())
    with pytest.raises(TransitionError) as caught:
        reduce(job, attempt, ModelOom(JOB_ID, ATT), now=NOW)
    assert caught.value.reason_code is ReasonCode.INVALID_INPUT


# ------------------------------------------------------ stop / crash / cancel distinct

def test_stop_crash_cancel_are_distinct():
    job, attempt = at_state(JobState.TRANSCRIBING)
    stopped = reduce(job, attempt, ServiceStopped(JOB_ID), now=NOW)
    lost = reduce(job, attempt, ServiceLost(JOB_ID), now=NOW)
    cancel = reduce(job, attempt, CancelRequested(JOB_ID), now=NOW)
    assert (stopped.job.state, stopped.job.reason_code) == (JobState.INTERRUPTED, ReasonCode.SERVICE_STOPPED)
    assert (lost.job.state, lost.job.reason_code) == (JobState.INTERRUPTED, ReasonCode.SERVICE_LOST)
    assert cancel.job.state is JobState.TRANSCRIBING
    assert cancel.job.reason_code is ReasonCode.CANCELLATION_REQUESTED
    assert stopped.attempt.state is JobState.INTERRUPTED and stopped.attempt.ended_at == NOW
    assert stopped.effects == (Effect.TERMINATE_WORKER, Effect.RELEASE_LEASE)
    final = reduce(cancel.job, cancel.attempt, WriterReaped(JOB_ID, True), now=NOW)
    assert final.job.state is JobState.CANCELLED
    assert len({stopped.job.reason_code, lost.job.reason_code, final.job.reason_code}) == 3


@pytest.mark.parametrize("event", [ServiceStopped(JOB_ID), ServiceLost(JOB_ID)])
def test_stop_and_crash_retain_queue_and_artifacts(event):
    queued, _ = at_state(JobState.QUEUED)
    decision = reduce(queued, None, event, now=NOW)
    assert decision.job is queued and decision.job.state is JobState.QUEUED
    job, attempt = at_state(JobState.EXPORTING)
    job = replace(job, artifacts=(art(ArtifactSlot.TRANSCRIPT_TXT),))
    interrupted = reduce(job, attempt, event, now=NOW)
    assert interrupted.job.artifacts == job.artifacts
    retry = reduce(interrupted.job, interrupted.attempt, RetryRequested(JOB_ID, ident(77, 2), "k"), now=NOW)
    assert retry.job.state is JobState.CHECKING and retry.superseded_attempt is None


# ------------------------------------------------------------ unavailable persistence

@pytest.mark.parametrize("state", ALL_STATES)
@pytest.mark.parametrize("reason", [ReasonCode.STATE_WRITE_FAILED, ReasonCode.AUDIT_LOSS])
def test_persistence_failure_never_reports_success(state, reason):
    job, attempt = at_state(state)
    decision = reduce(job, attempt, StatePersistenceFailed(JOB_ID, reason), now=NOW)
    assert decision.effects == (Effect.HALT_ADMISSION,) and decision.reason_code is reason
    assert decision.job is job and decision.attempt is attempt
    assert decision.job.state not in (JobState.COMPLETED, JobState.PARTIAL) or state in (
        JobState.COMPLETED, JobState.PARTIAL)


def test_persistence_failure_during_exporting_does_not_complete():
    d = Driver(_request("transcript"))
    d.to_exporting()
    for slot in T_SLOTS:
        d.publish(slot)
    before = d.job
    decision = d.send(StatePersistenceFailed(JOB_ID))
    assert decision.job is before and decision.job.state is JobState.EXPORTING
    with pytest.raises(DomainValidationError):
        StatePersistenceFailed(JOB_ID, ReasonCode.OK)


# ---------------------------------------------------------------- domains / parsing

def test_domain_negative_examples_rejected():
    negatives = json.loads((ROOT / "Docs" / "contract-domains.json").read_text(encoding="utf-8"))["negative_examples"]
    assert "__invalid__" in negatives and "" in negatives
    for bad in negatives:
        with pytest.raises(DomainValidationError):
            JobState.parse(bad)
        with pytest.raises(DomainValidationError):
            Stage.parse(bad)
        with pytest.raises(DomainValidationError):
            tr.event_from_dict({"event": "stage_started", "job_id": JOB_ID, "attempt_id": ATT,
                                "stage": bad, "stage_id": ident(3, 5), "settings_hash": SHA})
        with pytest.raises(DomainValidationError):
            tr.event_from_dict({"event": "blocked", "job_id": JOB_ID, "reason_code": bad, "next_action": "x"})
        with pytest.raises(DomainValidationError):
            tr.event_from_dict({"event": bad})
        with pytest.raises(DomainValidationError):
            StageStarted(JOB_ID, ATT, bad, ident(3, 5), SHA)


def test_event_from_dict_builds_valid_events():
    event = tr.event_from_dict({"event": "stage_started", "job_id": JOB_ID, "attempt_id": ATT,
                                "stage": "checking", "stage_id": ident(3, 5), "settings_hash": SHA})
    assert event.stage is Stage.CHECKING
    assert tr.event_from_dict({"event": "cancel_requested", "job_id": JOB_ID}) == CancelRequested(JOB_ID)
    with pytest.raises(DomainValidationError):
        tr.event_from_dict({"event": "cancel_requested", "job_id": JOB_ID, "extra": 1})
    with pytest.raises(DomainValidationError):
        tr.event_from_dict("not a dict")


@pytest.mark.parametrize("duration", [float("nan"), float("inf"), -1.0, "5"])
def test_media_probed_requires_finite_duration(duration):
    with pytest.raises(DomainValidationError):
        MediaProbed(JOB_ID, ATT, True, True, duration)
    with pytest.raises(DomainValidationError):
        MediaFacts(True, True, duration)


def test_invalid_reduce_inputs():
    job, attempt = at_state(JobState.CHECKING)
    with pytest.raises(TransitionError) as caught:
        reduce(job, attempt, CancelRequested(JOB_ID), now="yesterday")
    assert caught.value.reason_code is ReasonCode.INVALID_INPUT
    with pytest.raises(TransitionError):
        reduce(job, None, CancelRequested(JOB_ID), now=NOW)
    with pytest.raises(TransitionError):
        reduce(job, attempt, object(), now=NOW)
    with pytest.raises(TransitionError):
        reduce(job, replace(attempt, attempt_id=ident(9, 2), stage_records=()), CancelRequested(JOB_ID), now=NOW)


def test_facts_required_after_probe_for_transcript():
    d = Driver(_request("transcript"))
    d.to_probed()
    with pytest.raises(TransitionError) as caught:
        tr.next_stage(d.job.request, d.attempt, None)
    assert caught.value.reason_code is ReasonCode.CONFLICT


def test_unrequested_and_duplicate_artifacts_conflict():
    d = Driver(_request("transcript"))
    sid = d.to_exporting()
    with pytest.raises(TransitionError):
        d.publish(ArtifactSlot.ORIGINAL_VIDEO)
    d.publish(ArtifactSlot.TRANSCRIPT_TXT)
    with pytest.raises(TransitionError):
        d.publish(ArtifactSlot.TRANSCRIPT_TXT)
    assert sid


# -------------------------------------------------------------------- determinism / purity

def test_reducer_is_deterministic_and_leaves_inputs_unchanged():
    job, attempt = at_state(JobState.EXPORTING)
    event = StageFinished(JOB_ID, ATT, ident(7, 3), StageOutcome.ERROR, ReasonCode.ARTIFACT_REJECTED)
    before = (job.to_dict(), attempt.to_dict())
    results = [reduce(job, attempt, event, now=NOW) for _ in range(3)]
    assert results[0] == results[1] == results[2]
    assert (job.to_dict(), attempt.to_dict()) == before
    with pytest.raises(TransitionError):  # a refused transition returns nothing partial
        reduce(job, attempt, RetryRequested(JOB_ID, ident(77, 2), "k"), now=NOW)
    assert (job.to_dict(), attempt.to_dict()) == before


def test_module_has_no_io_clock_or_repository_dependency():
    source = (ROOT / "src" / "podmuncher" / "transitions.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add(("." * node.level) + (node.module or ""))
    forbidden = {"os", "time", "datetime", "random", "uuid", "sqlite3", "pathlib", "subprocess",
                 "socket", "threading", "asyncio", "shutil", ".repository"}
    assert not imported & forbidden, imported & forbidden
    calls = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert not calls & {"now", "utcnow", "time", "monotonic", "open", "read_text", "write_text"}
