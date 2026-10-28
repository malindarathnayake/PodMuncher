"""Writer-boundary guards, lease retention and real tiny-file adapters."""

import asyncio
import errno
import io
import math
from dataclasses import FrozenInstanceError

import pytest

from podmuncher.admission import AdmissionPlan
from podmuncher.domains import DomainValidationError, FieldStatus, ReasonCode
from podmuncher.guards import (
    AsyncFileReader,
    AsyncFileWriter,
    CancellationCoordinator,
    GuardError,
    OperationBinding,
    RetryableFailure,
    StopResult,
    StorageSample,
    WriteGuard,
    guarded_copy,
    guarded_write,
    retry_after_seconds,
    retry_operation,
)
from podmuncher.snapshots import Measurement


def plan(reason=ReasonCode.OK):
    okay = reason is ReasonCode.OK
    return AdmissionPlan(Measurement(True if okay else None, 'test',
                                    FieldStatus.OK if okay else FieldStatus.ERROR,
                                    '2026-10-09T12:00:00Z', None if okay else reason), (), ())


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = asyncio.Queue()

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        event = asyncio.Event()
        await self.sleeps.put((seconds, event))
        await event.wait()

    async def tick(self, seconds):
        _, event = await self.sleeps.get()
        self.now += seconds
        event.set()
        await asyncio.sleep(0)


class Lease:
    def __init__(self):
        self.held = False
        self.acquires = 0
        self.releases = 0

    def try_acquire(self, operation_id, admission):
        assert admission.admitted
        self.acquires += 1
        if self.held:
            return False
        self.held = True
        return True

    def release(self, operation_id, *, writers_reaped):
        assert writers_reaped and self.held
        self.releases += 1
        self.held = False


class FakeWriter:
    def __init__(self, *, short=None, failure=None, close_failure=None):
        self.data = bytearray()
        self.short = short
        self.failure = failure
        self.close_failure = close_failure
        self.closed = 0

    async def write(self, data):
        if self.failure is not None:
            raise self.failure
        count = len(data) if self.short is None else min(self.short, len(data))
        self.data.extend(data[:count])
        return count

    async def close(self):
        self.closed += 1
        if self.close_failure is not None:
            raise self.close_failure


class Operation:
    def __init__(self, *, graceful=True, kill_works=True, fail_stop=False):
        self.binding = OperationBinding('test-operation', 'verified-target-a')
        self.graceful = graceful
        self.kill_works = kill_works
        self.fail_stop = fail_stop
        self.exited = False
        self.terminations = 0
        self.kills = 0

    async def terminate(self, expected):
        assert expected is self.binding
        self.terminations += 1
        if self.fail_stop:
            raise RuntimeError('private path and stderr')
        self.exited = self.graceful

    async def kill(self, expected):
        assert expected is self.binding
        self.kills += 1
        self.exited = self.kill_works

    async def reap(self, expected):
        assert expected is self.binding
        return self.exited


def setup_guard(*, limit=100, timeout=60, sampler=None):
    clock = FakeClock()
    lease = Lease()
    coordinator = CancellationCoordinator('test-operation', lease, stop_grace_seconds=.03,
                                           reap_timeout_seconds=.03, callback_timeout_seconds=.03)

    async def sample():
        return StorageSample(plan(), clock())

    guard = WriteGuard(coordinator, sampler or sample, byte_limit=limit,
                       timeout_seconds=timeout, clock=clock, sleep=clock.sleep)
    return guard, clock, lease


async def chunks(*values):
    for value in values:
        yield value


@pytest.mark.asyncio
async def test_short_writes_count_actual_bytes_and_exact_cap_is_allowed():
    guard, _, lease = setup_guard(limit=6)
    writer = FakeWriter(short=2)
    assert await guarded_write(guard, chunks(b'abcdef'), writer) == 6
    assert writer.data == b'abcdef' and writer.closed == 1
    assert lease.releases == 1


@pytest.mark.asyncio
async def test_oversized_next_chunk_rejected_whole_before_crossing_cap():
    guard, _, lease = setup_guard(limit=5)
    writer = FakeWriter()
    with pytest.raises(GuardError) as error:
        await guarded_write(guard, chunks(b'abc', b'def'), writer)
    assert error.value.reason_code is ReasonCode.MEDIA_LIMIT_REACHED
    assert writer.data == b'abc' and guard.bytes_written == 3
    assert writer.closed == 1 and lease.releases == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('count', [0, -1, True, None, 4])
async def test_invalid_short_write_result_fails_closed(count):
    class InvalidWriter(FakeWriter):
        async def write(self, data):
            return count

    guard, _, lease = setup_guard()
    writer = InvalidWriter()
    with pytest.raises(GuardError) as error:
        await guarded_write(guard, chunks(b'abc'), writer)
    assert error.value.reason_code is ReasonCode.STATE_WRITE_FAILED
    assert writer.closed == 1 and lease.releases == 1


@pytest.mark.parametrize('limit', [True, -1, 1.5, '100', None])
def test_byte_budget_rejects_non_domain_values(limit):
    with pytest.raises(GuardError):
        setup_guard(limit=limit)


@pytest.mark.parametrize('reason', ['__invalid__', ''])
def test_guard_error_rejects_registry_negative_reasons(reason):
    with pytest.raises(DomainValidationError) as error:
        GuardError(reason)
    assert error.value.reason_code is ReasonCode.INVALID_INPUT


@pytest.mark.asyncio
@pytest.mark.parametrize('reason', ['__invalid__', ''])
async def test_write_guard_rejects_registry_negative_cap_reason_before_writing(reason):
    guard, _, lease = setup_guard()
    writer = FakeWriter()

    async def attempt_write():
        candidate = WriteGuard(guard.coordinator, guard.sampler, byte_limit=1,
                               timeout_seconds=1, cap_reason=reason)
        await guarded_write(candidate, chunks(b'x'), writer)

    with pytest.raises(DomainValidationError) as error:
        await attempt_write()
    assert error.value.reason_code is ReasonCode.INVALID_INPUT
    assert writer.data == b'' and writer.closed == 0
    assert lease.acquires == lease.releases == 0


@pytest.mark.asyncio
@pytest.mark.parametrize('reason', ['__invalid__', ''])
async def test_invalid_stop_reason_preserves_active_held_lease(reason):
    guard, _, lease = setup_guard()
    await guard.start()
    with pytest.raises(DomainValidationError) as error:
        guard.coordinator.request_stop(reason)
    assert error.value.reason_code is ReasonCode.INVALID_INPUT
    assert guard.coordinator.active and not guard.coordinator.stopped.is_set()
    assert guard.coordinator.reason is ReasonCode.OK
    assert lease.held and lease.releases == 0
    await guard.finish()


@pytest.mark.asyncio
@pytest.mark.parametrize('reason', [ReasonCode.STORAGE_UNKNOWN, ReasonCode.STORAGE_STALE,
                                   ReasonCode.STORAGE_INSUFFICIENT])
async def test_initial_denial_never_writes_or_acquires_and_closes(reason):
    async def sample():
        return StorageSample(plan(reason), 0)
    guard, _, lease = setup_guard(sampler=sample)
    writer = FakeWriter()
    with pytest.raises(GuardError) as error:
        await guarded_write(guard, chunks(b'x'), writer)
    assert error.value.reason_code is reason
    assert not writer.data and writer.closed == 1 and lease.acquires == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('close_failure', 'terminal_reason'),
    [
        (GuardError(ReasonCode.WRITER_UNRESOLVED), ReasonCode.WRITER_UNRESOLVED),
        (OSError(errno.EIO, 'secret/close'), ReasonCode.STORAGE_UNKNOWN),
    ],
)
async def test_initial_denial_close_failure_is_not_reported_reaped(
        close_failure, terminal_reason):
    async def sample():
        return StorageSample(plan(ReasonCode.STORAGE_UNKNOWN), 0)

    guard, _, lease = setup_guard(sampler=sample)
    writer = FakeWriter(close_failure=close_failure)
    with pytest.raises(GuardError) as error:
        await guarded_write(guard, chunks(b'secret'), writer)
    result = await guard.finish()

    assert error.value.reason_code is terminal_reason
    assert result.reason is terminal_reason
    assert not result.writers_reaped and not result.lease_released
    assert lease.acquires == lease.releases == 0 and not lease.held
    assert writer.closed == 1 and writer.data == b''
    assert 'secret' not in str(error.value) and 'secret' not in repr(result)


@pytest.mark.asyncio
async def test_cancelled_pre_admission_close_retains_unresolved_evidence():
    entered, release = asyncio.Event(), asyncio.Event()

    async def sample():
        return StorageSample(plan(ReasonCode.STORAGE_UNKNOWN), 0)

    class BlockingCloseWriter(FakeWriter):
        async def close(self):
            self.closed += 1
            entered.set()
            await release.wait()
            raise GuardError(ReasonCode.WRITER_UNRESOLVED)

    guard, _, lease = setup_guard(sampler=sample)
    writer = BlockingCloseWriter()
    writing = asyncio.create_task(guarded_write(guard, chunks(b'x'), writer))
    await entered.wait()
    writing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await writing

    result = await guard.finish()
    assert result.reason is ReasonCode.WRITER_UNRESOLVED
    assert not result.writers_reaped and not result.lease_released
    assert lease.acquires == lease.releases == 0 and not lease.held

    release.set()
    await asyncio.gather(*guard.coordinator._controls, return_exceptions=True)
    assert writer.closed == 1
    assert len(guard.coordinator._unregistered_close_results) == 1
    assert all(task.done() for task in guard.coordinator._controls)


@pytest.mark.asyncio
@pytest.mark.parametrize(('number', 'reason'), [(errno.ENOSPC, ReasonCode.STORAGE_GUARD_TRIPPED),
                                              (getattr(errno, 'EDQUOT', 122), ReasonCode.QUOTA_EXCEEDED)])
async def test_os_failures_are_closed_and_resources_finalized(number, reason):
    guard, _, lease = setup_guard()
    writer = FakeWriter(failure=OSError(number, 'secret/path'))
    with pytest.raises(GuardError) as error:
        await guarded_write(guard, chunks(b'x'), writer)
    assert error.value.reason_code is reason and 'secret' not in str(error.value)
    assert writer.closed == 1 and lease.releases == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('number', 'reason'),
    [
        (errno.ENOSPC, ReasonCode.STORAGE_GUARD_TRIPPED),
        (getattr(errno, 'EDQUOT', 122), ReasonCode.QUOTA_EXCEEDED),
    ],
)
async def test_concurrent_finish_keeps_discarded_tracked_write_failure(number, reason):
    write_entered, write_release = asyncio.Event(), asyncio.Event()
    reap_entered, reap_release = asyncio.Event(), asyncio.Event()

    class FailingWriter(FakeWriter):
        async def write(self, data):
            write_entered.set()
            await write_release.wait()
            raise OSError(number, 'secret/write')

    class BlockingReapOperation(Operation):
        async def reap(self, expected):
            assert expected is self.binding
            reap_entered.set()
            await reap_release.wait()
            return True

    guard, _, lease = setup_guard()
    writer = FailingWriter()
    writing = asyncio.create_task(guarded_write(guard, chunks(b'x'), writer))
    await write_entered.wait()
    guard.coordinator.register_operation(BlockingReapOperation())
    tracked_task, = {task for task in guard.coordinator._tasks if not task.done()}

    finishing = asyncio.create_task(guard.finish())
    await reap_entered.wait()
    write_release.set()
    await asyncio.wait({tracked_task})
    assert tracked_task.done() and tracked_task not in guard.coordinator._tasks
    reap_release.set()

    result = await finishing
    with pytest.raises(GuardError) as error:
        await writing
    assert error.value.reason_code is reason
    assert result.reason is reason
    assert result.writers_reaped and result.lease_released
    assert writer.closed == 1 and lease.releases == 1 and not lease.held


@pytest.mark.asyncio
@pytest.mark.parametrize(('number', 'reason'), [(errno.ENOSPC, ReasonCode.STORAGE_GUARD_TRIPPED),
                                              (getattr(errno, 'EDQUOT', 122), ReasonCode.QUOTA_EXCEEDED)])
async def test_close_os_failures_preserve_reason_and_retain_lease(number, reason):
    guard, _, lease = setup_guard()
    writer = FakeWriter(close_failure=OSError(number, 'secret/path'))
    with pytest.raises(GuardError) as error:
        await guarded_write(guard, chunks(b'x'), writer)
    assert error.value.reason_code is reason and 'secret' not in str(error.value)
    assert writer.data == b'x' and writer.closed == 1
    assert lease.held and lease.releases == 0
    result = await guard.finish()
    assert result.reason is reason
    assert result.writers_reaped is False and result.lease_released is False


@pytest.mark.asyncio
@pytest.mark.parametrize(('number', 'reason'), [(errno.ENOSPC, ReasonCode.STORAGE_GUARD_TRIPPED),
                                              (getattr(errno, 'EDQUOT', 122), ReasonCode.QUOTA_EXCEEDED)])
async def test_close_storage_failure_overrides_generic_write_failure(number, reason):
    guard, _, lease = setup_guard()
    writer = FakeWriter(failure=OSError(errno.EIO, 'secret/write'),
                        close_failure=OSError(number, 'secret/close'))
    with pytest.raises(GuardError) as error:
        await guarded_write(guard, chunks(b'x'), writer)
    assert error.value.reason_code is reason and 'secret' not in str(error.value)
    assert writer.closed == 1 and lease.held and lease.releases == 0
    result = await guard.finish()
    assert result.reason is reason
    assert result.writers_reaped is False and result.lease_released is False
    assert await guard.finish() == result


@pytest.mark.asyncio
@pytest.mark.parametrize('reverse_registration', [False, True])
@pytest.mark.parametrize(
    ('close_failure', 'reason'),
    [
        (OSError(errno.ENOSPC, 'secret/enospc'), ReasonCode.STORAGE_GUARD_TRIPPED),
        (OSError(getattr(errno, 'EDQUOT', 122), 'secret/edquot'),
         ReasonCode.QUOTA_EXCEEDED),
        (GuardError(ReasonCode.WRITER_UNRESOLVED), ReasonCode.WRITER_UNRESOLVED),
    ],
)
async def test_multiple_close_failures_use_order_independent_reason_precedence(
        reverse_registration, close_failure, reason):
    guard, _, lease = setup_guard()
    calls = [0, 0]

    async def fail(index, error):
        calls[index] += 1
        raise error

    closers = [
        lambda: fail(0, OSError(errno.EIO, 'secret/eio')),
        lambda: fail(1, close_failure),
    ]
    await guard.start()
    for closer in closers[::(-1 if reverse_registration else 1)]:
        guard.coordinator.register_resource(closer)

    result = await guard.finish()

    assert result.reason is reason
    assert not result.writers_reaped and not result.lease_released
    assert lease.held and lease.releases == 0 and calls == [1, 1]
    assert 'secret' not in repr(result) and 'secret' not in str(result.reason)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('body_reason', 'terminal_reason'),
    [
        (ReasonCode.CANCELLED, ReasonCode.CANCELLED),
        (ReasonCode.STORAGE_GUARD_TRIPPED, ReasonCode.STORAGE_GUARD_TRIPPED),
        (ReasonCode.OK, ReasonCode.STATE_WRITE_FAILED),
    ],
)
async def test_meaningful_body_reason_survives_generic_close_failure(
        body_reason, terminal_reason):
    guard, _, lease = setup_guard()

    async def fail_close():
        raise OSError(errno.EIO, 'secret/close')

    await guard.start()
    guard.coordinator.register_resource(fail_close)
    result = await guard.finish(body_reason)

    assert result.reason is terminal_reason
    assert not result.writers_reaped and not result.lease_released
    assert lease.held and lease.releases == 0
    assert 'secret' not in repr(result)


@pytest.mark.asyncio
async def test_close_exception_with_ok_reason_fails_closed():
    guard, _, lease = setup_guard()
    writer = FakeWriter(close_failure=GuardError(ReasonCode.OK))

    with pytest.raises(GuardError) as error:
        await guarded_write(guard, chunks(b'x'), writer)
    result = await guard.finish()

    assert error.value.reason_code is ReasonCode.STATE_WRITE_FAILED
    assert result.reason is ReasonCode.STATE_WRITE_FAILED
    assert not result.writers_reaped and not result.lease_released
    assert lease.held and lease.releases == 0
    assert writer.closed == 1
    assert await guard.finish() == result


@pytest.mark.asyncio
async def test_idle_external_writer_is_monitored_and_stopped():
    current = plan()

    async def sample():
        return StorageSample(current, clock())
    guard, clock, lease = setup_guard(sampler=sample)
    await guard.start()
    operation = Operation()
    guard.coordinator.register_operation(operation)
    current = plan(ReasonCode.STORAGE_INSUFFICIENT)
    await clock.tick(1)
    await asyncio.wait_for(guard.coordinator.stopped.wait(), .3)
    result = await guard.finish()
    assert result.reason is ReasonCode.STORAGE_GUARD_TRIPPED
    assert operation.terminations == 1 and lease.releases == 1


@pytest.mark.asyncio
async def test_transition_and_publication_each_resample():
    samples = 0

    async def sample():
        nonlocal samples
        samples += 1
        return StorageSample(plan(), clock())
    guard, clock, _ = setup_guard(sampler=sample)
    await guard.start()
    await guard.transition()
    await guard.publication()
    assert samples == 3
    await guard.finish()


@pytest.mark.asyncio
async def test_stale_and_exact_timeout_refuse_next_write():
    for now, expected in [(3.001, ReasonCode.STORAGE_STALE), (3, ReasonCode.DOWNLOAD_TIMEOUT)]:
        guard, clock, lease = setup_guard(timeout=3 if now == 3 else 60)
        writer = FakeWriter()
        await guard.start()
        guard.coordinator.register_resource(writer.close)
        clock.now = now
        with pytest.raises(GuardError) as error:
            await guard.write(writer, b'x')
        assert error.value.reason_code is expected
        assert writer.data == b'' and lease.releases == 1
        await guard.finish()


@pytest.mark.asyncio
async def test_competing_cancels_finalize_once_and_block_writes_immediately():
    guard, _, lease = setup_guard()
    await guard.start()
    operation = Operation(graceful=False)
    guard.coordinator.register_operation(operation)
    guard.coordinator.request_stop(ReasonCode.CANCELLED)
    with pytest.raises(GuardError):
        await guard.write(FakeWriter(), b'x')
    results = await asyncio.gather(guard.finish(ReasonCode.CANCELLED),
                                   guard.finish(ReasonCode.DOWNLOAD_TIMEOUT),
                                   guard.finish(ReasonCode.STORAGE_UNKNOWN))
    assert all(result == results[0] for result in results)
    assert operation.terminations == 1 and operation.kills == 1 and lease.releases == 1


@pytest.mark.asyncio
async def test_failed_stop_kill_reap_retains_lease_and_rejects_unowned():
    guard, _, lease = setup_guard()
    await guard.start()
    stranger = Operation()
    stranger.binding = OperationBinding('other-operation', 'verified-target-b')
    with pytest.raises(GuardError):
        guard.coordinator.register_operation(stranger)
    operation = Operation(graceful=False, kill_works=False, fail_stop=True)
    guard.coordinator.register_operation(operation)
    result = await guard.finish(ReasonCode.CANCELLED)
    assert result.reason is ReasonCode.WRITER_UNRESOLVED and lease.held
    assert lease.releases == 0 and stranger.terminations == stranger.kills == 0


@pytest.mark.asyncio
async def test_cancelled_waiter_does_not_cancel_write_or_release_early():
    guard, _, lease = setup_guard()
    entered, release = asyncio.Event(), asyncio.Event()

    class SlowWriter(FakeWriter):
        async def write(self, data):
            entered.set()
            await release.wait()
            return await super().write(data)

    writer = SlowWriter()
    await guard.start()
    guard.coordinator.register_resource(writer.close)
    task = asyncio.create_task(guard.write(writer, b'abc'))
    await entered.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert lease.held and writer.closed == 0
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert writer.data == b'abc' and guard.bytes_written == 3
    assert writer.closed == 1 and lease.releases == 1
    await guard.finish()


@pytest.mark.asyncio
async def test_unreaped_write_is_retained_and_not_closed_or_retried():
    guard, _, lease = setup_guard()
    entered, release = asyncio.Event(), asyncio.Event()

    class StuckWriter(FakeWriter):
        async def write(self, data):
            entered.set()
            await release.wait()
            return await super().write(data)

    writer = StuckWriter()
    await guard.start()
    guard.coordinator.register_resource(writer.close)
    task = asyncio.create_task(guard.write(writer, b'x'))
    await entered.wait()
    result = await guard.finish(ReasonCode.CANCELLED)
    with pytest.raises(GuardError):
        await task
    assert result.reason is ReasonCode.WRITER_UNRESOLVED
    assert lease.held and writer.closed == 0
    release.set()
    await asyncio.gather(*guard.coordinator._tasks, return_exceptions=True)
    assert writer.data == b'x' and guard.bytes_written == 1 and lease.held
    await writer.close()  # Explicit test-owned cleanup, after writer exited.


@pytest.mark.asyncio
async def test_ownership_change_cannot_kill_other_operation():
    guard, _, lease = setup_guard()
    await guard.start()
    operation = Operation()
    guard.coordinator.register_operation(operation)
    operation.binding = OperationBinding('unrelated', 'verified-target-b')
    result = await guard.finish(ReasonCode.CANCELLED)
    assert result.reason is ReasonCode.WRITER_UNRESOLVED and lease.held
    assert operation.terminations == operation.kills == 0


@pytest.mark.asyncio
async def test_hanging_lifecycle_callbacks_have_bounded_shutdown():
    guard, _, lease = setup_guard()
    release = asyncio.Event()

    class HungOperation(Operation):
        async def terminate(self, expected):
            assert expected is self.binding
            await release.wait()

        async def kill(self, expected):
            assert expected is self.binding
            await release.wait()

        async def reap(self, expected):
            assert expected is self.binding
            await release.wait()
            return True

    await guard.start()
    guard.coordinator.register_operation(HungOperation())
    result = await asyncio.wait_for(guard.finish(ReasonCode.CANCELLED), .5)
    assert result.reason is ReasonCode.WRITER_UNRESOLVED and lease.held
    release.set()
    await asyncio.gather(*guard.coordinator._controls)


@pytest.mark.parametrize('field', ['owner_id', 'target_id'])
@pytest.mark.parametrize('value', [None, True, 1, '', 'x' * 8193, object()])
def test_operation_binding_rejects_invalid_identity(field, value):
    fields = {'owner_id': 'test-operation', 'target_id': 'verified-target-a'}
    fields[field] = value
    with pytest.raises(GuardError) as error:
        OperationBinding(**fields)
    assert error.value.reason_code is ReasonCode.INVALID_INPUT


@pytest.mark.asyncio
async def test_immutable_binding_and_captured_callbacks_reap_and_release_once():
    guard, _, lease = setup_guard()
    await guard.start()
    operation = Operation()
    with pytest.raises(FrozenInstanceError):
        operation.binding.target_id = 'replacement'
    guard.coordinator.register_operation(operation)

    async def replacement(expected):
        pytest.fail('registered callbacks must not be replaced by method lookup')

    operation.terminate = operation.kill = operation.reap = replacement
    results = await asyncio.gather(guard.finish(ReasonCode.CANCELLED),
                                   guard.finish(ReasonCode.DOWNLOAD_TIMEOUT))
    assert all(result.writers_reaped and result.lease_released for result in results)
    assert operation.terminations == 1 and operation.kills == 0
    assert operation.exited and lease.releases == 1 and not lease.held


@pytest.mark.asyncio
@pytest.mark.parametrize('field', ['owner_id', 'target_id'])
@pytest.mark.parametrize('stage', ['terminate', 'first_reap', 'kill', 'final_reap'])
@pytest.mark.parametrize('outcome', ['success', 'failure', 'timeout', 'cancelled'])
async def test_retarget_during_each_lifecycle_wait_retains_original_lease(field, stage, outcome):
    guard, _, lease = setup_guard()
    entered, release = asyncio.Event(), asyncio.Event()
    calls, touched = [], []

    class RetargetedOperation(Operation):
        async def boundary(self, expected, name):
            assert expected is original
            assert self.binding is original  # Checked at the modeled OS use.
            calls.append(name)
            touched.append(expected.target_id)
            if name == stage:
                entered.set()
                await release.wait()
                if outcome == 'failure':
                    raise RuntimeError('private lifecycle failure')
                if outcome == 'cancelled':
                    raise asyncio.CancelledError

        async def terminate(self, expected):
            await self.boundary(expected, 'terminate')

        async def kill(self, expected):
            await self.boundary(expected, 'kill')

        async def reap(self, expected):
            name = 'final_reap' if 'first_reap' in calls else 'first_reap'
            await self.boundary(expected, name)
            # Model the reported defect: a result can be truthful for the new
            # exited target while the originally registered writer is live.
            return name == stage or name == 'final_reap'

    operation = RetargetedOperation()
    original = operation.binding
    await guard.start()
    guard.coordinator.register_operation(operation)
    stopping = asyncio.create_task(guard.finish(ReasonCode.CANCELLED))
    await asyncio.wait_for(entered.wait(), .5)
    fields = {'owner_id': original.owner_id, 'target_id': original.target_id}
    fields[field] = 'unrelated-owner' if field == 'owner_id' else 'exited-target-b'
    operation.binding = OperationBinding(**fields)
    if outcome != 'timeout':
        release.set()
    result = await asyncio.wait_for(stopping, .5)
    order = ['terminate', 'first_reap', 'kill', 'final_reap']
    assert calls == order[:order.index(stage) + 1]
    assert touched == [original.target_id] * len(calls)
    assert not operation.exited and lease.held and lease.releases == 0
    assert result.reason is ReasonCode.WRITER_UNRESOLVED
    assert not result.writers_reaped and not result.lease_released
    if outcome == 'timeout':
        assert any(not task.done() for task in guard.coordinator._controls)
        release.set()
    await asyncio.gather(*guard.coordinator._controls, return_exceptions=True)
    # Completion of stale callbacks cannot retroactively release the lease.
    operation.binding = original
    assert await guard.finish() == result
    assert lease.held and lease.releases == 0


@pytest.mark.asyncio
@pytest.mark.parametrize('during_close', [False, True])
async def test_binding_change_after_reap_during_other_shutdown_wait_retains_lease(during_close):
    guard, _, lease = setup_guard()
    entered, release = asyncio.Event(), asyncio.Event()
    await guard.start()
    operation = Operation()
    guard.coordinator.register_operation(operation)

    async def close():
        entered.set()
        await release.wait()

    if during_close:
        guard.coordinator.register_resource(close)
    else:
        class OtherOperation(Operation):
            async def reap(self, expected):
                await close()
                return await super().reap(expected)

        other = OtherOperation()
        other.binding = OperationBinding('test-operation', 'verified-other-target')
        guard.coordinator.register_operation(other)

    stopping = asyncio.create_task(guard.finish(ReasonCode.CANCELLED))
    await asyncio.wait_for(entered.wait(), .5)
    assert operation.exited
    operation.binding = OperationBinding('test-operation', 'replacement-target')
    release.set()
    result = await asyncio.wait_for(stopping, .5)
    assert result.reason is ReasonCode.WRITER_UNRESOLVED
    assert not result.lease_released and lease.held and lease.releases == 0


@pytest.mark.asyncio
async def test_copy_reads_until_eof_without_trusting_a_declared_size():
    guard, _, _ = setup_guard(limit=6)
    source = io.BytesIO(b'abcdef')
    writer = FakeWriter(short=1)
    assert await guarded_copy(guard, AsyncFileReader(source), writer, chunk_size=2) == 6
    assert source.closed and writer.data == b'abcdef'


@pytest.mark.asyncio
@pytest.mark.parametrize('priority_on_reader', [False, True])
@pytest.mark.parametrize(
    ('priority_failure', 'reason'),
    [
        (OSError(errno.ENOSPC, 'secret/enospc'), ReasonCode.STORAGE_GUARD_TRIPPED),
        (OSError(getattr(errno, 'EDQUOT', 122), 'secret/edquot'),
         ReasonCode.QUOTA_EXCEEDED),
        (GuardError(ReasonCode.WRITER_UNRESOLVED), ReasonCode.WRITER_UNRESOLVED),
    ],
)
async def test_copy_close_ranks_reader_and_writer_failures(
        priority_on_reader, priority_failure, reason):
    class FailingReader:
        def __init__(self, failure):
            self.failure = failure
            self.closed = 0

        async def read(self, size):
            return b''

        async def close(self):
            self.closed += 1
            raise self.failure

    reader_failure = priority_failure if priority_on_reader else OSError(errno.EIO, 'secret/eio')
    writer_failure = OSError(errno.EIO, 'secret/eio') if priority_on_reader else priority_failure
    reader = FailingReader(reader_failure)
    writer = FakeWriter(close_failure=writer_failure)
    guard, _, lease = setup_guard()

    with pytest.raises(GuardError) as error:
        await guarded_copy(guard, reader, writer)

    assert error.value.reason_code is reason
    assert reader.closed == writer.closed == 1
    assert lease.held and lease.releases == 0
    assert 'secret' not in str(error.value)


@pytest.mark.parametrize('value', [True, None, '', 'NaN', 'inf', '-1', ' 2', object(),
                                  math.nan, math.inf, -1, 'Wed, 21 Oct 2026 07:28:00 GMT'])
def test_invalid_retry_after_fails_closed(value):
    with pytest.raises(GuardError):
        retry_after_seconds(value)


def test_large_retry_after_is_bounded():
    assert retry_after_seconds('1000000000', maximum=2) == 2
    assert retry_after_seconds(0) == 0


@pytest.mark.asyncio
async def test_retry_attempts_need_reaping_and_explicit_reset():
    calls, resets, delays = [], [], []

    async def attempt(index):
        calls.append(index)
        if index < 3:
            raise RetryableFailure(StopResult(ReasonCode.UPSTREAM_UNAVAILABLE, True, True), 999)
        return 'done'

    async def reset(guard):
        assert guard.coordinator.active
        resets.append(True)

    async def sleep(seconds):
        delays.append(seconds)

    def factory():
        return setup_guard()[0]
    assert await retry_operation(attempt, reset=reset, reset_guard=factory,
                                 sleep=sleep, maximum_retry_after=2) == 'done'
    assert calls == [1, 2, 3] and len(resets) == 2 and delays == [2, 2]

    async def unresolved(index):
        raise RetryableFailure(StopResult(ReasonCode.WRITER_UNRESOLVED, False, False, False))

    with pytest.raises(GuardError) as error:
        await retry_operation(unresolved, reset=reset, reset_guard=factory, sleep=sleep)
    assert error.value.reason_code is ReasonCode.WRITER_UNRESOLVED and len(resets) == 2
    with pytest.raises(GuardError):
        await retry_operation(attempt, reset=reset, reset_guard=factory, attempts=4)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('drained', 'reason'),
    [
        (False, ReasonCode.UPSTREAM_UNAVAILABLE),
        (None, ReasonCode.STORAGE_UNKNOWN),
        (1, ReasonCode.STORAGE_UNKNOWN),
        ('yes', ReasonCode.STORAGE_UNKNOWN),
    ],
)
async def test_retry_rejects_unresolved_or_nonliteral_sampler_drain_flag(drained, reason):
    events = []

    async def attempt(index):
        events.append(('attempt', index))
        raise RetryableFailure(StopResult(ReasonCode.UPSTREAM_UNAVAILABLE,
                                          True, True, drained))

    async def reset(guard):
        events.append(('reset', guard))

    def factory():
        events.append(('factory',))
        return setup_guard()[0]

    async def sleep(seconds):
        events.append(('sleep', seconds))

    with pytest.raises(GuardError) as error:
        await retry_operation(attempt, reset=reset, reset_guard=factory, sleep=sleep)
    assert error.value.reason_code is reason
    assert events == [('attempt', 1)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('attempt_reason', 'terminal_reason'),
    [
        (ReasonCode.OK, ReasonCode.STORAGE_UNKNOWN),
        (ReasonCode.STORAGE_GUARD_TRIPPED, ReasonCode.STORAGE_GUARD_TRIPPED),
        (ReasonCode.QUOTA_EXCEEDED, ReasonCode.QUOTA_EXCEEDED),
    ],
)
async def test_retry_unresolved_sampler_never_retries_or_resets(
        attempt_reason, terminal_reason):
    events = []

    async def attempt(index):
        events.append(('attempt', index))
        raise RetryableFailure(StopResult(attempt_reason, True, True, False))

    async def reset(guard):
        events.append(('reset', guard))

    def factory():
        events.append(('factory',))
        return setup_guard()[0]

    async def sleep(seconds):
        events.append(('sleep', seconds))

    with pytest.raises(GuardError) as error:
        await retry_operation(attempt, reset=reset, reset_guard=factory, sleep=sleep)
    assert error.value.reason_code is terminal_reason
    assert events == [('attempt', 1)]


@pytest.mark.asyncio
async def test_retry_retains_live_sampler_after_real_finish():
    release = asyncio.Event()
    events = []
    retained = []
    block = False

    async def sample():
        if not block:
            return StorageSample(plan(), clock())
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                pass
        return StorageSample(plan(), clock())

    guard, clock, lease = setup_guard(sampler=sample)

    async def attempt(index):
        nonlocal block
        events.append(('attempt', index))
        await guard.start()
        guard.sample_timeout_seconds = .05
        block = True
        with pytest.raises(GuardError) as error:
            await guard.transition()
        assert error.value.reason_code is ReasonCode.STORAGE_UNKNOWN
        exact_task, = guard._sample_tasks
        retained.append(exact_task)
        result = await guard.finish()
        assert result.writers_reaped is True and result.lease_released is True
        assert result.samplers_drained is False and not exact_task.done()
        raise RetryableFailure(result)

    async def reset(reset_guard):
        events.append(('reset', reset_guard))

    def factory():
        events.append(('factory',))
        return setup_guard()[0]

    async def sleep(seconds):
        events.append(('sleep', seconds))

    try:
        with pytest.raises(GuardError) as error:
            await retry_operation(attempt, reset=reset, reset_guard=factory, sleep=sleep)
        assert error.value.reason_code is ReasonCode.STORAGE_UNKNOWN
        assert events == [('attempt', 1)]
        assert retained[0] in guard._sample_tasks and not retained[0].done()
        assert lease.acquires == lease.releases == 1 and not lease.held
    finally:
        release.set()
        if retained:
            await retained[0]
    assert not guard._sample_tasks


@pytest.mark.asyncio
@pytest.mark.parametrize('cancel', [False, True])
async def test_reset_timeout_or_cancellation_retains_owned_lease(cancel):
    guard, _, lease = setup_guard()
    entered, release = asyncio.Event(), asyncio.Event()
    attempts = []
    writer = FakeWriter()

    async def attempt(index):
        attempts.append(index)
        raise RetryableFailure(StopResult(ReasonCode.UPSTREAM_UNAVAILABLE, True, True))

    async def reset(reset_guard):
        assert reset_guard is guard and lease.held
        reset_guard.coordinator.register_resource(writer.close)
        entered.set()
        await release.wait()
        # This models an in-flight blocking reset completing after its waiter.
        await writer.write(b'x')

    task = asyncio.create_task(retry_operation(attempt, reset=reset,
                                               reset_guard=lambda: guard,
                                               reset_timeout_seconds=.01))
    await entered.wait()
    if cancel:
        task.cancel()
    with pytest.raises(GuardError) as error:
        await asyncio.wait_for(task, .5)
    assert error.value.reason_code is ReasonCode.WRITER_UNRESOLVED
    assert lease.held and lease.releases == 0 and attempts == [1] and writer.closed == 0
    release.set()
    await asyncio.gather(*guard.coordinator._tasks)
    assert lease.held and writer.data == b'x'
    await writer.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('close_failure', 'reason'),
    [
        (OSError(errno.ENOSPC, 'secret/enospc'), ReasonCode.STORAGE_GUARD_TRIPPED),
        (OSError(getattr(errno, 'EDQUOT', 122), 'secret/edquot'),
         ReasonCode.QUOTA_EXCEEDED),
    ],
)
async def test_reset_propagates_finalized_close_reason(close_failure, reason):
    guard, _, lease = setup_guard()
    attempts = []
    reset_calls = 0

    async def attempt(index):
        attempts.append(index)
        raise RetryableFailure(StopResult(ReasonCode.UPSTREAM_UNAVAILABLE, True, True))

    async def reset(reset_guard):
        nonlocal reset_calls
        reset_calls += 1

        async def close():
            raise close_failure

        reset_guard.coordinator.register_resource(close)
        raise OSError(errno.EIO, 'secret/reset')

    async def no_sleep(seconds):
        assert seconds == 0

    with pytest.raises(GuardError) as error:
        await retry_operation(attempt, reset=reset, reset_guard=lambda: guard,
                              sleep=no_sleep)

    assert error.value.reason_code is reason
    assert attempts == [1] and reset_calls == 1
    assert lease.held and lease.releases == 0
    assert 'secret' not in str(error.value)


@pytest.mark.asyncio
async def test_sampler_timeout_and_future_sample_fail_closed():
    release = asyncio.Event()
    finalized = asyncio.Event()

    async def hung():
        try:
            await release.wait()
        finally:
            finalized.set()
    guard, _, lease = setup_guard(sampler=hung)
    guard.sample_timeout_seconds = .01
    with pytest.raises(GuardError) as error:
        await guarded_write(guard, chunks(b'x'), FakeWriter())
    assert error.value.reason_code is ReasonCode.STORAGE_UNKNOWN and not lease.held
    assert finalized.is_set() and not guard._sample_tasks

    async def future():
        return StorageSample(plan(), 10)
    guard, _, _ = setup_guard(sampler=future)
    with pytest.raises(GuardError) as error:
        await guard.start()
    assert error.value.reason_code is ReasonCode.STORAGE_STALE


@pytest.mark.asyncio
@pytest.mark.parametrize('boundary', ['start', 'watchdog', 'transition', 'publication'])
async def test_sampling_timeout_drains_cooperative_adapter_at_each_boundary(boundary):
    entered, finalized = asyncio.Event(), asyncio.Event()
    sampling_tasks = []
    block = boundary == 'start'

    async def sample():
        if not block:
            return StorageSample(plan(), clock())
        sampling_tasks.append(asyncio.current_task())
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            finalized.set()

    guard, clock, lease = setup_guard(sampler=sample)
    guard.sample_timeout_seconds = .05
    if boundary != 'start':
        await guard.start()
        block = True
    if boundary == 'watchdog':
        await clock.tick(1)
        await asyncio.wait_for(entered.wait(), .5)
        await asyncio.wait_for(guard.coordinator.stopped.wait(), .5)
    else:
        checking = asyncio.create_task(getattr(guard, boundary)())
        await asyncio.wait_for(entered.wait(), .5)
        with pytest.raises(GuardError) as error:
            await asyncio.wait_for(checking, .5)
        assert error.value.reason_code is ReasonCode.STORAGE_UNKNOWN
    assert finalized.is_set() and sampling_tasks[0].done()
    assert not guard._sample_tasks
    result = await guard.finish(ReasonCode.STORAGE_UNKNOWN)
    assert result.reason is ReasonCode.STORAGE_UNKNOWN and result.samplers_drained
    assert lease.acquires == lease.releases == (boundary != 'start')


@pytest.mark.asyncio
@pytest.mark.parametrize('boundary', ['start', 'transition', 'publication'])
async def test_cancelled_sampling_waiter_drains_exact_sampler(boundary):
    entered, finalized = asyncio.Event(), asyncio.Event()
    sampling_tasks = []
    block = boundary == 'start'

    async def sample():
        if not block:
            return StorageSample(plan(), clock())
        sampling_tasks.append(asyncio.current_task())
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            finalized.set()

    guard, clock, lease = setup_guard(sampler=sample)
    if boundary != 'start':
        await guard.start()
        block = True
    checking = asyncio.create_task(getattr(guard, boundary)())
    await asyncio.wait_for(entered.wait(), .5)
    checking.cancel()
    with pytest.raises(asyncio.CancelledError):
        await checking
    assert finalized.is_set() and sampling_tasks[0].cancelled()
    assert not guard._sample_tasks
    result = await guard.finish(ReasonCode.CANCELLED)
    assert result.samplers_drained
    assert lease.acquires == lease.releases == (boundary != 'start')


@pytest.mark.asyncio
@pytest.mark.parametrize('boundary', ['watchdog', 'transition'])
async def test_finish_cancels_and_drains_inflight_sampling(boundary):
    entered, finalized = asyncio.Event(), asyncio.Event()
    block = False

    async def sample():
        if not block:
            return StorageSample(plan(), clock())
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            finalized.set()

    guard, clock, lease = setup_guard(sampler=sample)
    await guard.start()
    block = True
    checking = None
    if boundary == 'watchdog':
        await clock.tick(1)
    else:
        checking = asyncio.create_task(guard.transition())
    await asyncio.wait_for(entered.wait(), .5)
    exact_task, = guard._sample_tasks
    result = await asyncio.wait_for(guard.finish(ReasonCode.CANCELLED), .5)
    if checking is not None:
        with pytest.raises(GuardError):
            await checking
    assert finalized.is_set() and exact_task.cancelled()
    assert not guard._sample_tasks and result.samplers_drained
    assert result.writers_reaped and result.lease_released and lease.releases == 1


@pytest.mark.asyncio
@pytest.mark.parametrize('live_writer', [False, True])
async def test_cancellation_suppressing_sampler_is_bounded_retained_and_reported(live_writer):
    entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    writer_entered, writer_release = asyncio.Event(), asyncio.Event()
    block = False
    calls = 0

    async def sample():
        nonlocal calls
        calls += 1
        if not block:
            return StorageSample(plan(), clock())
        entered.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
        return StorageSample(plan(), clock())

    class PendingWriter(FakeWriter):
        async def write(self, data):
            writer_entered.set()
            await writer_release.wait()
            return await super().write(data)

    guard, clock, lease = setup_guard(sampler=sample)
    guard.sample_timeout_seconds = .05
    await guard.start()
    writer, writing = PendingWriter(), None
    if live_writer:
        guard.coordinator.register_resource(writer.close)
        writing = asyncio.create_task(guard.write(writer, b'partial'))
        await writer_entered.wait()
    block = True
    checking = asyncio.create_task(guard.transition())
    await asyncio.wait_for(entered.wait(), .5)
    exact_task, = guard._sample_tasks
    try:
        with pytest.raises(GuardError) as error:
            await asyncio.wait_for(checking, .5)
        assert error.value.reason_code is ReasonCode.STORAGE_UNKNOWN
        assert cancelled.is_set() and not exact_task.done()
        result = await asyncio.wait_for(guard.finish(), .5)
        assert not result.samplers_drained and exact_task in guard._sample_tasks
        assert not exact_task.done() and calls == 2
        if live_writer:
            with pytest.raises(GuardError):
                await writing
            assert result.reason is ReasonCode.WRITER_UNRESOLVED
            assert not result.writers_reaped and not result.lease_released
            assert lease.held and writer.closed == 0 and writer.data == b''
        else:
            assert result.reason is ReasonCode.STORAGE_UNKNOWN
            assert result.writers_reaped and result.lease_released
            assert not lease.held and lease.releases == 1
    finally:
        release.set()
        await exact_task
        writer_release.set()
        await asyncio.gather(*guard.coordinator._tasks, return_exceptions=True)
        if live_writer:
            await writer.close()
    assert not guard._sample_tasks
    if live_writer:
        assert writer.data == b'partial' and guard.bytes_written == 7 and lease.held


@pytest.mark.asyncio
async def test_repeated_waiter_cancellation_retains_sampler_finalizer_until_finish():
    entered, finalizing = asyncio.Event(), asyncio.Event()
    release = asyncio.Event()

    async def sample():
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            finalizing.set()
            await release.wait()

    guard, _, lease = setup_guard(sampler=sample)
    checking = asyncio.create_task(guard.start())
    await entered.wait()
    exact_task, = guard._sample_tasks
    checking.cancel()
    await finalizing.wait()
    checking.cancel()
    with pytest.raises(asyncio.CancelledError):
        await checking
    assert exact_task in guard._sample_tasks and not exact_task.done()
    release.set()
    result = await asyncio.wait_for(guard.finish(ReasonCode.CANCELLED), .5)
    assert exact_task.cancelled() and not guard._sample_tasks and result.samplers_drained
    assert lease.acquires == lease.releases == 0


@pytest.mark.asyncio
async def test_repeated_finish_cancellation_retains_original_sampling_and_shutdown():
    entered, finalizing, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    stopping, stop_release = asyncio.Event(), asyncio.Event()
    block = False

    async def sample():
        if not block:
            return StorageSample(plan(), clock())
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            finalizing.set()
            await release.wait()

    class StoppingOperation(Operation):
        async def terminate(self, expected):
            stopping.set()
            await stop_release.wait()
            await super().terminate(expected)

    guard, clock, lease = setup_guard(sampler=sample)
    guard.coordinator.callback_timeout_seconds = 1
    await guard.start()
    guard.coordinator.register_operation(StoppingOperation())
    block = True
    checking = asyncio.create_task(guard.transition())
    await entered.wait()
    exact_task, = guard._sample_tasks
    finishing = asyncio.create_task(guard.finish(ReasonCode.CANCELLED))
    await stopping.wait()
    finishing.cancel()
    await finalizing.wait()
    finishing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await finishing
    assert exact_task in guard._sample_tasks and not exact_task.done() and lease.held
    release.set()
    stop_release.set()
    with pytest.raises(asyncio.CancelledError):
        await exact_task
    with pytest.raises(GuardError):
        await checking
    result = await guard.finish()
    assert result.samplers_drained and result.writers_reaped and result.lease_released
    assert not guard._sample_tasks and lease.releases == 1


@pytest.mark.asyncio
async def test_sampling_lock_contention_is_bounded_without_starting_another_adapter():
    entered, finalizing, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    block = False
    calls = 0

    async def sample():
        nonlocal calls
        calls += 1
        if not block:
            return StorageSample(plan(), clock())
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            finalizing.set()
            await release.wait()

    guard, clock, lease = setup_guard(sampler=sample)
    await guard.start()
    block = True
    first = asyncio.create_task(guard.transition())
    await entered.wait()
    exact_task, = guard._sample_tasks
    first.cancel()
    await finalizing.wait()  # The first waiter still holds the sampling lock.
    guard.sample_timeout_seconds = .05
    try:
        with pytest.raises(GuardError) as error:
            await asyncio.wait_for(guard.publication(), .5)
        assert error.value.reason_code is ReasonCode.STORAGE_UNKNOWN and calls == 2
        assert exact_task in guard._sample_tasks and not exact_task.done()
    finally:
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await exact_task
        with pytest.raises(asyncio.CancelledError):
            await first
        result = await guard.finish()
    assert result.samplers_drained and not guard._sample_tasks and lease.releases == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('number', 'reason'),
    [
        (errno.ENOSPC, ReasonCode.STORAGE_GUARD_TRIPPED),
        (getattr(errno, 'EDQUOT', 122), ReasonCode.QUOTA_EXCEEDED),
    ],
)
async def test_unresolved_sampler_preserves_writer_failure_reason(number, reason):
    entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    block = False

    async def sample():
        if not block:
            return StorageSample(plan(), clock())
        entered.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
        return StorageSample(plan(), clock())

    guard, clock, lease = setup_guard(sampler=sample)
    guard.sample_timeout_seconds = .05
    writer = FakeWriter(failure=OSError(number, 'secret/write'))
    await guard.start()
    guard.coordinator.register_resource(writer.close)
    block = True
    await clock.tick(1)
    await entered.wait()
    exact_task, = guard._sample_tasks
    try:
        with pytest.raises(GuardError) as error:
            await guard.write(writer, b'x')
        assert error.value.reason_code is reason
        result = await guard.finish()
        assert result.reason is reason
        assert result.writers_reaped and result.lease_released
        assert not result.samplers_drained and not exact_task.done()
        assert cancelled.is_set() and writer.closed == 1 and lease.releases == 1
    finally:
        release.set()
        await exact_task
    assert not guard._sample_tasks


@pytest.mark.asyncio
async def test_pre_admission_finish_cannot_outrun_supplied_sink_close():
    entered, cancelled, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def sample():
        entered.set()
        while not release.is_set():
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
        return StorageSample(plan(), clock())

    guard, clock, lease = setup_guard(sampler=sample)
    guard.sample_timeout_seconds = .05
    writer = FakeWriter(close_failure=OSError(errno.ENOSPC, 'secret/close'))
    writing = asyncio.create_task(guarded_write(guard, chunks(b'x'), writer))
    await entered.wait()

    terminal = await guard.finish(ReasonCode.CANCELLED)
    assert terminal.reason is ReasonCode.WRITER_UNRESOLVED
    assert not terminal.writers_reaped and not terminal.lease_released
    assert lease.acquires == lease.releases == 0 and not lease.held
    assert writer.closed == 0 and cancelled.is_set()

    release.set()
    with pytest.raises(GuardError) as error:
        await writing
    assert error.value.reason_code is ReasonCode.WRITER_UNRESOLVED
    assert writer.closed == 1
    assert len(guard.coordinator._unregistered_close_results) == 1
    assert all(task.done() for task in guard.coordinator._controls)
    repeated = await guard.finish()
    assert repeated.reason is ReasonCode.WRITER_UNRESOLVED
    assert not repeated.writers_reaped and not repeated.lease_released
    assert lease.acquires == lease.releases == 0 and not lease.held


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_tiny_file_copy_and_bounded_enospc(tmp_path):
    source = tmp_path / 'owned-source.bin'
    target = tmp_path / 'owned-target.bin'
    partial = tmp_path / 'owned-partial.bin'
    source.write_bytes(b'abcdefgh')
    guard, _, lease = setup_guard(limit=8)
    reader = AsyncFileReader(source.open('rb', buffering=0))
    writer = AsyncFileWriter(target.open('wb', buffering=0))
    assert await guarded_copy(guard, reader, writer, chunk_size=2) == 8
    assert target.read_bytes() == source.read_bytes() and lease.releases == 1

    class TinyQuotaHandle:
        def __init__(self, handle):
            self.handle = handle
            self.written = 0

        def write(self, data):
            if self.written + len(data) > 4:
                raise OSError(errno.ENOSPC, 'injected tiny quota')
            count = self.handle.write(data)
            self.written += count
            return count

        def close(self):
            self.handle.close()

    guard, _, lease = setup_guard(limit=8)
    writer = AsyncFileWriter(TinyQuotaHandle(partial.open('wb', buffering=0)))
    with pytest.raises(GuardError) as error:
        await guarded_write(guard, chunks(b'ab', b'cd', b'ef'), writer)
    assert error.value.reason_code is ReasonCode.STORAGE_GUARD_TRIPPED
    assert partial.read_bytes() == b'abcd' and lease.releases == 1
    for owned in (source, target, partial):
        owned.unlink()
    print('PODMUNCHER_SMOKE_OK')
