"""Bounded storage guards and cancellation of explicitly owned writers.

The sampler recomputes admission using *remaining* growth, including bytes
already written. Paths, network clients and process discovery are deliberately
outside this module. Lifecycle adapters must address an already verified owned
operation; ``reap`` returns True only after every writer in that operation exits.
"""

from __future__ import annotations

# Injected adapters are trust boundaries: all ordinary failures are translated
# to closed domain reasons, without forwarding their text or filesystem paths.
# ruff: noqa: BLE001
import asyncio
import errno
import math
import time
from collections.abc import AsyncIterable, Awaitable, Callable
from dataclasses import dataclass
from typing import BinaryIO, Protocol

from .admission import AdmissionPlan, ExecutionLease
from .config import Limits
from .domains import DomainValidationError, ReasonCode


class GuardError(DomainValidationError):
    """A closed reason; never includes an adapter exception or private path."""

    def __init__(self, reason_code: ReasonCode):
        super().__init__(ReasonCode.parse(reason_code))


def _seconds(value: object, *, positive: bool = True) -> float:
    if type(value) not in (int, float):
        raise GuardError(ReasonCode.INVALID_INPUT)
    try:
        number = float(value)
    except (ValueError, OverflowError):
        raise GuardError(ReasonCode.INVALID_INPUT) from None
    if not math.isfinite(number) or (number <= 0 if positive else number < 0):
        raise GuardError(ReasonCode.INVALID_INPUT)
    return number


def _failure(error: BaseException) -> ReasonCode:
    if isinstance(error, DomainValidationError):
        return (ReasonCode.STATE_WRITE_FAILED if error.reason_code is ReasonCode.OK
                else error.reason_code)
    if isinstance(error, OSError):
        if error.errno == errno.ENOSPC:
            return ReasonCode.STORAGE_GUARD_TRIPPED
        if error.errno == getattr(errno, 'EDQUOT', 122):
            return ReasonCode.QUOTA_EXCEEDED
    return ReasonCode.STATE_WRITE_FAILED


def _terminal_reason(body_reason: ReasonCode, close_errors: tuple[BaseException, ...],
                     *, unresolved: bool) -> ReasonCode:
    """Select a closed terminal reason without depending on closer order."""
    close_reasons = tuple(_failure(error) for error in close_errors)
    if unresolved or ReasonCode.WRITER_UNRESOLVED in close_reasons:
        return ReasonCode.WRITER_UNRESOLVED
    if body_reason not in (ReasonCode.OK, ReasonCode.STATE_WRITE_FAILED):
        return body_reason
    if ReasonCode.QUOTA_EXCEEDED in close_reasons:
        return ReasonCode.QUOTA_EXCEEDED
    if ReasonCode.STORAGE_GUARD_TRIPPED in close_reasons:
        return ReasonCode.STORAGE_GUARD_TRIPPED
    explicit_reasons = sorted(
        {reason for reason in close_reasons if reason is not ReasonCode.STATE_WRITE_FAILED},
        key=lambda reason: reason.value,
    )
    if explicit_reasons:
        return explicit_reasons[0]
    if close_errors:
        return ReasonCode.STATE_WRITE_FAILED
    return body_reason


@dataclass(frozen=True, slots=True)
class StorageSample:
    plan: AdmissionPlan
    sampled_at: float


@dataclass(frozen=True, slots=True)
class OperationBinding:
    """One verified owner and non-reusable target identity for its lifetime."""

    owner_id: str
    target_id: str

    def __post_init__(self) -> None:
        for value in (self.owner_id, self.target_id):
            if type(value) is not str or not value or len(value) > 8192:
                raise GuardError(ReasonCode.INVALID_INPUT)


class OwnedOperation(Protocol):
    """Trusted adapter for a verified target, never caller-supplied discovery.

    Keep the same binding object for the entire registration. Callbacks receive
    that original binding and must address only its verified target at every OS
    use, including after their own awaits or inside a worker thread. Resolve a
    non-reusable identity (e.g. full container ID), never a mutable name or bare
    reusable PID. If the binding no longer applies, fail without touching any
    replacement. ``reap(expected)`` proves all writers of *expected* exited.

    Coordinator checks cannot inspect hidden adapter state or make an OS lookup
    atomic. This point-of-use guarantee is the adapter's trust boundary.
    """

    binding: OperationBinding

    async def terminate(self, expected: OperationBinding) -> None: ...
    async def kill(self, expected: OperationBinding) -> None: ...
    async def reap(self, expected: OperationBinding) -> bool: ...


@dataclass(frozen=True, slots=True)
class _RegisteredOperation:
    operation: OwnedOperation
    binding: OperationBinding
    terminate: Callable[[OperationBinding], Awaitable[None]]
    kill: Callable[[OperationBinding], Awaitable[None]]
    reap: Callable[[OperationBinding], Awaitable[bool]]


@dataclass(slots=True)
class _ResourceReservation:
    coordinator: CancellationCoordinator
    slot: int
    state: str = 'pending'


class AsyncWriter(Protocol):
    async def write(self, data: bytes) -> int: ...
    async def close(self) -> None: ...


class AsyncReader(Protocol):
    async def read(self, size: int) -> bytes: ...
    async def close(self) -> None: ...


class AsyncFileWriter:
    """Own an already policy-authorized binary handle; preserve partial files.

    Guard-owned shielded tasks retain the underlying thread until it finishes.
    This adapter neither selects paths nor removes files. Use an unbuffered file
    handle where physical short-write accounting is required.
    """

    def __init__(self, handle: BinaryIO):
        self._handle = handle

    async def write(self, data: bytes) -> int:
        return await asyncio.to_thread(self._handle.write, data)

    async def close(self) -> None:
        await asyncio.to_thread(self._handle.close)


class AsyncFileReader:
    def __init__(self, handle: BinaryIO):
        self._handle = handle

    async def read(self, size: int) -> bytes:
        return await asyncio.to_thread(self._handle.read, size)

    async def close(self) -> None:
        await asyncio.to_thread(self._handle.close)


@dataclass(frozen=True, slots=True)
class StopResult:
    reason: ReasonCode
    writers_reaped: bool
    lease_released: bool
    # Python sampler tasks only; native read lifetimes remain adapter-owned.
    samplers_drained: bool = True


def _observe(task: asyncio.Task) -> None:
    # Retrieve adapter failures even if an unresolved task completes later.
    if not task.cancelled():
        task.exception()


class CancellationCoordinator:
    """One finalization task shared by watchdog, timeout and user cancellation.

    Cancellation of a waiter never cancels the underlying writer or shutdown.
    Unresolved tasks stay referenced and keep the execution lease permanently;
    callers must use their recovery path before retrying such an operation.
    """

    def __init__(self, operation_id: str, lease: ExecutionLease, *,
                 stop_grace_seconds: float = 10, reap_timeout_seconds: float = 2,
                 callback_timeout_seconds: float = 1):
        if type(operation_id) is not str or not operation_id or len(operation_id) > 8192:
            raise GuardError(ReasonCode.INVALID_INPUT)
        self.operation_id = operation_id
        self.lease = lease
        self.stop_grace_seconds = _seconds(stop_grace_seconds)
        self.reap_timeout_seconds = _seconds(reap_timeout_seconds)
        self.callback_timeout_seconds = _seconds(callback_timeout_seconds)
        self.stopped = asyncio.Event()
        self.reason = ReasonCode.OK
        self._acquired = False
        self._released = False
        self._operations: list[_RegisteredOperation] = []
        self._ownership_lost: set[int] = set()
        self._tasks: set[asyncio.Task] = set()
        self._task_errors: list[BaseException] = []
        self._controls: set[asyncio.Task] = set()
        self._closers: list[Callable[[], Awaitable[None]]] = []
        self._reservations: list[_ResourceReservation] = []
        self._unregistered_close_results: list[tuple[bool, object]] = []
        self._unregistered_close_tasks: set[asyncio.Task] = set()
        self._shutdown_task: asyncio.Task[StopResult] | None = None

    @property
    def active(self) -> bool:
        return self._acquired and not self.stopped.is_set() and not self._released

    def acquire(self, plan: AdmissionPlan) -> None:
        if self._acquired or self.stopped.is_set() or type(plan) is not AdmissionPlan:
            raise GuardError(ReasonCode.CONFLICT)
        if not plan.admitted:
            raise GuardError(plan.reason)
        try:
            acquired = self.lease.try_acquire(self.operation_id, plan)
        except Exception:
            raise GuardError(ReasonCode.STATE_WRITE_FAILED) from None
        if acquired is not True:
            raise GuardError(ReasonCode.CONFLICT)
        self._acquired = True

    def ensure_active(self) -> None:
        if not self.active:
            raise GuardError(self.reason if self.reason is not ReasonCode.OK else ReasonCode.CONFLICT)

    def register_operation(self, operation: OwnedOperation) -> None:
        self.ensure_active()
        try:
            binding = operation.binding
            callbacks = tuple(getattr(operation, method) for method in ('terminate', 'kill', 'reap'))
            valid = (type(binding) is OperationBinding
                     and binding.owner_id == self.operation_id
                     and operation.binding is binding
                     and all(callable(callback) for callback in callbacks)
                     and not any(item.operation is operation for item in self._operations))
        except Exception:
            raise GuardError(ReasonCode.INVALID_INPUT) from None
        if not valid:
            raise GuardError(ReasonCode.INVALID_INPUT)
        self._operations.append(_RegisteredOperation(operation, binding, *callbacks))

    def register_resource(self, close: Callable[[], Awaitable[None]]) -> None:
        self.ensure_active()
        if not callable(close):
            raise GuardError(ReasonCode.INVALID_INPUT)
        self._closers.append(close)

    def _reserve_resource(self) -> _ResourceReservation:
        if self.stopped.is_set() or self._shutdown_task is not None:
            raise GuardError(ReasonCode.CONFLICT)
        reservation = _ResourceReservation(self, len(self._reservations))
        self._reservations.append(reservation)
        return reservation

    def _validate_reservation(self, reservation: _ResourceReservation) -> None:
        valid = (
            type(reservation) is _ResourceReservation
            and reservation.coordinator is self
            and type(reservation.slot) is int
            and 0 <= reservation.slot < len(self._reservations)
            and self._reservations[reservation.slot] is reservation
            and reservation.state == 'pending'
        )
        if not valid:
            raise GuardError(ReasonCode.WRITER_UNRESOLVED)

    def _register_reserved_resource(
            self, reservation: _ResourceReservation,
            close: Callable[[], Awaitable[None]]) -> None:
        self.ensure_active()
        self._validate_reservation(reservation)
        if not callable(close):
            raise GuardError(ReasonCode.INVALID_INPUT)
        self._closers.append(close)
        reservation.state = 'registered'

    def track(self, awaitable: Awaitable) -> asyncio.Task:
        """Register before yielding, so shutdown cannot miss a newly started IO."""
        self.ensure_active()
        task = asyncio.create_task(awaitable)
        self._tasks.add(task)
        task.add_done_callback(self._tracked_done)
        return task

    def _tracked_done(self, task: asyncio.Task) -> None:
        try:
            error = task.exception()
        except asyncio.CancelledError:
            error = None
        if error is not None and not isinstance(error, StopAsyncIteration):
            self._task_errors.append(error)
        self._tasks.discard(task)

    def request_stop(self, reason: ReasonCode) -> None:
        reason = ReasonCode.parse(reason)
        if not self.stopped.is_set():
            self.reason = reason
            self.stopped.set()

    async def stop(self, reason: ReasonCode = ReasonCode.CANCELLED) -> StopResult:
        self.request_stop(reason)
        if self._shutdown_task is None:
            self._shutdown_task = asyncio.create_task(self._shutdown())
        return await asyncio.shield(self._shutdown_task)

    async def _call(self, callback: Callable[[], Awaitable], seconds: float) -> tuple[bool, object]:
        async def invoke():
            return await callback()
        task = asyncio.create_task(invoke())
        self._controls.add(task)
        task.add_done_callback(_observe)
        done, _ = await asyncio.wait({task}, timeout=seconds)
        if not done:
            # Do not cancel a callback that may be awaiting a blocking thread.
            return False, None
        try:
            return True, task.result()
        except (Exception, asyncio.CancelledError) as error:
            # Keep the exception only as opaque shutdown evidence. Callers
            # translate it to a closed reason and never expose adapter text.
            return False, error

    async def _close_unregistered(
            self, close: Callable[[], Awaitable[None]],
            reservation: _ResourceReservation | None = None) -> None:
        if reservation is not None:
            self._validate_reservation(reservation)
            reservation.state = 'closing'

        async def close_and_record() -> None:
            try:
                result = await self._call(close, self.callback_timeout_seconds)
                self._unregistered_close_results.append(result)
            finally:
                if reservation is not None:
                    reservation.state = 'resolved'

        task = asyncio.create_task(close_and_record())
        self._unregistered_close_tasks.add(task)
        self._controls.add(task)
        task.add_done_callback(_observe)
        await asyncio.shield(task)

    def _binding_current(self, operation: _RegisteredOperation) -> bool:
        identity = id(operation)
        if identity in self._ownership_lost:
            return False
        try:
            current = operation.operation.binding is operation.binding
        except Exception:
            current = False
        if not current:
            self._ownership_lost.add(identity)
        return current

    async def _call_owned(self, operation: _RegisteredOperation,
                          callback: Callable[[OperationBinding], Awaitable],
                          seconds: float) -> tuple[bool, bool, object]:
        if not self._binding_current(operation):
            return False, False, None

        async def invoke():
            # Task scheduling is itself a yield boundary. Never look up a new
            # callback on the mutable adapter, and recheck before invoking it.
            if not self._binding_current(operation):
                raise GuardError(ReasonCode.WRITER_UNRESOLVED)
            try:
                return await callback(operation.binding)
            finally:
                # Latch changes at callback completion too, including callbacks
                # that outlive their timeout and finish during another wait.
                self._binding_current(operation)

        okay, value = await self._call(invoke, seconds)
        # Includes failed and timed-out callbacks; _call retains in-flight work.
        return self._binding_current(operation), okay, value

    async def _reap(self, operation: _RegisteredOperation) -> bool:
        if self.reason is not ReasonCode.OK:
            current, _, _ = await self._call_owned(
                operation, operation.terminate, self.callback_timeout_seconds)
            if not current:
                return False
        current, okay, reaped = await self._call_owned(
            operation, operation.reap, self.stop_grace_seconds)
        if not current:
            return False
        if okay and reaped is True:
            return True
        current, _, _ = await self._call_owned(
            operation, operation.kill, self.callback_timeout_seconds)
        if not current:
            return False
        current, okay, reaped = await self._call_owned(
            operation, operation.reap, self.reap_timeout_seconds)
        return current and okay and reaped is True

    async def _shutdown(self) -> StopResult:
        results = await asyncio.gather(*(self._reap(item) for item in self._operations))
        tracked = set(self._tasks)
        pending = {task for task in tracked if not task.done()}
        if pending:
            _, pending = await asyncio.wait(pending, timeout=self.reap_timeout_seconds)
        task_errors = list(self._task_errors)
        for task in tracked:
            if task.done() and not task.cancelled():
                error = task.exception()
                if error is not None and not isinstance(error, StopAsyncIteration):
                    task_errors.append(error)
        operations_reaped = (all(results) and not pending
                             and all(self._binding_current(item)
                                     for item in self._operations))
        unregistered_closes_done = all(
            task.done() for task in self._unregistered_close_tasks)
        recorded_closes = tuple(self._unregistered_close_results)
        close_errors = tuple(value for okay, value in recorded_closes
                             if not okay and isinstance(value, BaseException))
        close_errors += tuple(task_errors)
        close_unresolved = any(not okay and value is None
                               for okay, value in recorded_closes) or not unregistered_closes_done
        reservations_settled = all(
            reservation.state in ('registered', 'resolved')
            for reservation in self._reservations
        )
        reaped = (operations_reaped and unregistered_closes_done
                  and reservations_settled
                  and all(okay for okay, _ in recorded_closes))
        # A close racing a live write can corrupt files or block indefinitely.
        # Keep resources for explicit recovery when a writer cannot be reaped.
        if reaped:
            closed = await asyncio.gather(*(self._call(close, self.callback_timeout_seconds)
                                            for close in reversed(self._closers)))
            close_errors += tuple(value for okay, value in closed
                                  if not okay and isinstance(value, BaseException))
            close_unresolved = (close_unresolved
                                or any(not okay and value is None for okay, value in closed))
            reaped = all(okay for okay, _ in closed)
        # Lifecycle callbacks themselves can still perform side effects.
        controls_done = all(task.done() for task in self._controls)
        bindings_current = all(self._binding_current(item) for item in self._operations)
        reaped = reaped and controls_done and bindings_current
        unresolved = (not operations_reaped or close_unresolved or not reservations_settled
                      or not controls_done or not bindings_current)
        reason = _terminal_reason(self.reason, close_errors, unresolved=unresolved)
        if reaped and self._acquired and not self._released:
            try:
                self.lease.release(self.operation_id, writers_reaped=True)
                self._released = True
            except Exception:
                reason = ReasonCode.STATE_WRITE_FAILED
        return StopResult(reason, reaped, self._released)


class WriteGuard:
    """One operation's admission, byte/time budget and independent watchdog.

    Sampling is a trusted, read-only adapter boundary. Adapters must bound any
    native work they start: cancelling a Python task cannot prove that an
    offloaded OS read has ended. Pending sampler tasks stay referenced until
    completion and are reported separately from owned writer reaping.
    """

    def __init__(self, coordinator: CancellationCoordinator,
                 sampler: Callable[[], Awaitable[StorageSample]], *,
                 byte_limit: int, timeout_seconds: float,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
                 stale_after_seconds: float = 3, sample_timeout_seconds: float = 1,
                 cap_reason: ReasonCode = ReasonCode.MEDIA_LIMIT_REACHED):
        if type(byte_limit) is not int or byte_limit < 0:
            raise GuardError(ReasonCode.INVALID_INPUT)
        self.coordinator = coordinator
        self.sampler = sampler
        self.byte_limit = byte_limit
        self.timeout_seconds = _seconds(timeout_seconds)
        self.clock = clock
        self.sleep = sleep
        self.stale_after_seconds = _seconds(stale_after_seconds)
        self.sample_timeout_seconds = _seconds(sample_timeout_seconds)
        self.cap_reason = ReasonCode.parse(cap_reason)
        self.bytes_written = 0
        self._started: float | None = None
        self._last_sample: float | None = None
        self._watchdog: asyncio.Task | None = None
        self._sample_tasks: set[asyncio.Task] = set()
        self._write_lock = asyncio.Lock()
        self._sample_lock = asyncio.Lock()
        self._starting = False

    @classmethod
    def for_download(cls, coordinator: CancellationCoordinator,
                     sampler: Callable[[], Awaitable[StorageSample]], limits: Limits, **kwargs):
        if type(limits) is not Limits:
            raise GuardError(ReasonCode.INVALID_CONFIG)
        return cls(coordinator, sampler, byte_limit=limits.media_bytes,
                   timeout_seconds=limits.download_timeout_seconds, **kwargs)

    def _now(self) -> float:
        try:
            return _seconds(self.clock(), positive=False)
        except Exception:
            raise GuardError(ReasonCode.STORAGE_UNKNOWN) from None

    async def _sample(self, *, initial: bool = False) -> AdmissionPlan:
        # Bound lock contention as well as the adapter call. A transition must
        # not indefinitely hold the watchdog behind another sampling request.
        try:
            async with asyncio.timeout(self.sample_timeout_seconds):
                return await self._sample_once(initial=initial)
        except TimeoutError:
            raise GuardError(ReasonCode.STORAGE_UNKNOWN) from None

    async def _sample_once(self, *, initial: bool) -> AdmissionPlan:
        async with self._sample_lock:
            if self.coordinator.stopped.is_set():
                raise GuardError(self.coordinator.reason if self.coordinator.reason is not
                                 ReasonCode.OK else ReasonCode.CONFLICT)
            # Do not accumulate more reads behind an adapter ignoring cancel.
            if any(not task.done() for task in self._sample_tasks):
                raise GuardError(ReasonCode.STORAGE_UNKNOWN)
            task = asyncio.create_task(self.sampler())
            self._sample_tasks.add(task)
            task.add_done_callback(_observe)
            task.add_done_callback(self._sample_tasks.discard)
            try:
                # The outer timeout covers both this wait and lock contention.
                await asyncio.wait({task})
                try:
                    sample = task.result()
                except (Exception, asyncio.CancelledError):
                    raise GuardError(ReasonCode.STORAGE_UNKNOWN) from None
                if type(sample) is not StorageSample or type(sample.plan) is not AdmissionPlan:
                    raise GuardError(ReasonCode.STORAGE_UNKNOWN)
                try:
                    stamp = _seconds(sample.sampled_at, positive=False)
                except GuardError:
                    raise GuardError(ReasonCode.STORAGE_UNKNOWN) from None
                age = self._now() - stamp
                if not 0 <= age <= self.stale_after_seconds:
                    raise GuardError(ReasonCode.STORAGE_STALE)
                if not sample.plan.admitted:
                    reason = sample.plan.reason
                    if not initial and reason is ReasonCode.STORAGE_INSUFFICIENT:
                        reason = ReasonCode.STORAGE_GUARD_TRIPPED
                    raise GuardError(reason)
                self._last_sample = stamp
                return sample.plan
            finally:
                # This also runs when the outer deadline or waiter cancels us.
                # No detached cleanup task: a second cancellation can interrupt
                # this bounded drain, leaving the original task retained.
                await self._drain_samples({task})

    async def _drain_samples(self, tasks: set[asyncio.Task]) -> None:
        pending = {task for task in tasks if not task.done()}
        for task in pending:
            # A concurrent finish must not interrupt a cancellation finalizer.
            if not task.cancelling():
                task.cancel()
        if pending:
            await asyncio.wait(pending, timeout=self.sample_timeout_seconds)

    async def start(self) -> None:
        if self._starting or self._started is not None or self.coordinator.stopped.is_set():
            raise GuardError(ReasonCode.CONFLICT)
        self._starting = True
        try:
            plan = await self._sample(initial=True)
            started = self._now()
            self.coordinator.acquire(plan)
            self._started = started
            self._watchdog = asyncio.create_task(self._monitor())
        finally:
            self._starting = False

    def _boundary(self) -> None:
        self.coordinator.ensure_active()
        now = self._now()
        if self._started is None or now < self._started:
            raise GuardError(ReasonCode.STORAGE_UNKNOWN)
        if now - self._started >= self.timeout_seconds:
            raise GuardError(ReasonCode.DOWNLOAD_TIMEOUT)
        if self._last_sample is None or not 0 <= now - self._last_sample <= self.stale_after_seconds:
            raise GuardError(ReasonCode.STORAGE_STALE)

    async def _monitor(self) -> None:
        next_sample = self._now() + 1
        try:
            while not self.coordinator.stopped.is_set():
                await self.sleep(max(0, min(next_sample - self._now(),
                                          self._started + self.timeout_seconds - self._now())))
                self._boundary()
                await self._sample()
                self._boundary()
                # Cadence is measured between starts, including sampler time.
                next_sample += 1
                next_sample = max(next_sample, self._now())
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await self.coordinator.stop(_failure(error))

    async def transition(self) -> None:
        """Call immediately before every stage change and publication."""
        try:
            self._boundary()
            await self._sample()
            self._boundary()
        except Exception as error:
            await self.coordinator.stop(_failure(error))
            raise GuardError(_failure(error)) from None

    async def publication(self) -> None:
        await self.transition()

    async def io[T](self, callback: Callable[[], Awaitable[T]]) -> T:
        """Race IO against stop without cancelling or losing the IO task."""
        self._boundary()
        task = self.coordinator.track(callback())
        stop_task = asyncio.create_task(self.coordinator.stopped.wait())
        try:
            done, _ = await asyncio.wait({task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
            if task in done:
                return task.result()
            raise GuardError(self.coordinator.reason)
        finally:
            stop_task.cancel()
            await asyncio.gather(stop_task, return_exceptions=True)

    async def write(self, writer: AsyncWriter, chunk: bytes) -> None:
        async with self._write_lock:
            try:
                self._boundary()
                if type(chunk) is not bytes:
                    raise GuardError(ReasonCode.INVALID_INPUT)
                if len(chunk) > self.byte_limit - self.bytes_written:
                    raise GuardError(self.cap_reason)
                offset = 0
                while offset < len(chunk):
                    self._boundary()

                    async def write_part(start=offset):
                        count = await writer.write(chunk[start:])
                        if type(count) is not int or not 0 < count <= len(chunk) - start:
                            raise GuardError(ReasonCode.STATE_WRITE_FAILED)
                        # Account inside the tracked task even when its waiter
                        # is cancelled while a thread is completing the write.
                        self.bytes_written += count
                        return count

                    count = await self.io(write_part)
                    offset += count
                    self._boundary()
                    await asyncio.sleep(0)
            except asyncio.CancelledError:
                await self.coordinator.stop(ReasonCode.CANCELLED)
                raise
            except Exception as error:
                await self.coordinator.stop(_failure(error))
                raise GuardError(_failure(error)) from None

    async def finish(self, reason: ReasonCode = ReasonCode.OK) -> StopResult:
        try:
            result = await self.coordinator.stop(reason)
        finally:
            tasks = set(self._sample_tasks)
            if self._watchdog is not None and self._watchdog is not asyncio.current_task():
                tasks.add(self._watchdog)
            await self._drain_samples(tasks)
        drained = all(task.done() for task in self._sample_tasks)
        if not drained:
            # Writer failures retain precedence; read-only sampling does not
            # own a writer lease and cannot claim writers were not reaped.
            reason = (ReasonCode.STORAGE_UNKNOWN
                      if result.writers_reaped and result.reason is ReasonCode.OK
                      else result.reason)
            return StopResult(reason, result.writers_reaped, result.lease_released, False)
        return result


async def guarded_write(guard: WriteGuard, chunks: AsyncIterable[bytes],
                        writer: AsyncWriter) -> int:
    """Start, write and close a sink; no implicit deletion or retry."""
    reason = ReasonCode.OK
    registered = False
    reservation = guard.coordinator._reserve_resource()
    try:
        await guard.start()
        guard.coordinator._register_reserved_resource(reservation, writer.close)
        registered = True
        iterator = aiter(chunks)
        if callable(getattr(iterator, 'aclose', None)):
            guard.coordinator.register_resource(iterator.aclose)
        while True:
            try:
                chunk = await guard.io(iterator.__anext__)
            except StopAsyncIteration:
                break
            await guard.write(writer, chunk)
        await guard.transition()
    except asyncio.CancelledError:
        reason = ReasonCode.CANCELLED
        raise
    except Exception as error:
        reason = _failure(error)
        raise GuardError(reason) from None
    finally:
        if not registered:
            # The sink was handed to us even when initial admission failed.
            await guard.coordinator._close_unregistered(writer.close, reservation)
        result = await guard.finish(reason)
        if result.reason is not reason:
            raise GuardError(result.reason) from None
    return guard.bytes_written


async def guarded_copy(guard: WriteGuard, reader: AsyncReader, writer: AsyncWriter,
                       *, chunk_size: int = 1024 * 1024) -> int:
    if type(chunk_size) is not int or not 0 < chunk_size <= 16 * 1024 * 1024:
        raise GuardError(ReasonCode.INVALID_INPUT)

    async def chunks():
        while True:
            chunk = await reader.read(chunk_size)
            if type(chunk) is not bytes or len(chunk) > chunk_size:
                raise GuardError(ReasonCode.INVALID_INPUT)
            if not chunk:
                return
            yield chunk

    # Register the reader alongside the writer after admission, through a
    # wrapper whose close also covers a never-started generator.
    class CopySink:
        async def write(self, data: bytes) -> int:
            return await writer.write(data)

        async def close(self) -> None:
            errors: list[BaseException] = []
            for close in (reader.close, writer.close):
                try:
                    await close()
                except BaseException as error:
                    errors.append(error)
            unresolved = any(isinstance(error, asyncio.CancelledError) for error in errors)
            reason = _terminal_reason(ReasonCode.OK, tuple(errors), unresolved=unresolved)
            if reason is not ReasonCode.OK:
                raise GuardError(reason)

    return await guarded_write(guard, chunks(), CopySink())


def retry_after_seconds(value: object, *, maximum: float = 60) -> float:
    """Accept finite delta seconds only; dates and malformed values fail closed."""
    limit = _seconds(maximum)
    if type(value) is str:
        if len(value) > 128 or not value or value != value.strip():
            raise GuardError(ReasonCode.INVALID_INPUT)
        try:
            value = float(value)
        except (ValueError, OverflowError):
            raise GuardError(ReasonCode.INVALID_INPUT) from None
    return min(_seconds(value, positive=False), limit)


@dataclass(frozen=True, slots=True)
class RetryableFailure(Exception):
    """Only an explicitly retryable, fully finalized attempt can be repeated."""

    result: StopResult
    retry_after: object = 0

    def __str__(self) -> str:
        return 'Retryable operation failed.'


async def _reset_once(reset: Callable[[WriteGuard], Awaitable[None]],
                      factory: Callable[[], WriteGuard], timeout: float) -> None:
    guard = factory()
    if type(guard) is not WriteGuard:
        raise GuardError(ReasonCode.INVALID_INPUT)
    reason = ReasonCode.OK
    try:
        await guard.start()
        task = guard.coordinator.track(reset(guard))
        done, _ = await asyncio.wait({task}, timeout=timeout)
        if not done:
            raise GuardError(ReasonCode.DOWNLOAD_TIMEOUT)
        task.result()
    except asyncio.CancelledError:
        reason = ReasonCode.CANCELLED
        raise
    except Exception as error:
        reason = _failure(error)
        raise GuardError(reason) from None
    finally:
        result = await guard.finish(reason)
        if result.reason is not reason:
            raise GuardError(result.reason) from None


async def retry_operation[T](attempt: Callable[[int], Awaitable[T]], *,
                          reset: Callable[[WriteGuard], Awaitable[None]],
                          reset_guard: Callable[[], WriteGuard], attempts: int = 3,
                          maximum_retry_after: float = 60, reset_timeout_seconds: float = 5,
                          sleep: Callable[[float], Awaitable[None]] = asyncio.sleep) -> T:
    """Caller reset must recreate or safely rewind/truncate partial output.

    No reset is inferred from a seekable sink. Each attempt owns a fresh guard
    and must return/raise only after its lifecycle has been finalized. Each
    reset gets a fresh guard from reset_guard, acquires its lease and runs as
    a tracked writer. The callback registers any resources/owned subprocesses
    on that guard; it must not create detached writers. Timeout or cancellation
    retains the lease while reset is unresolved. Reset is never retried.
    """
    if (type(attempts) is not int or not 1 <= attempts <= 3
            or not callable(reset) or not callable(reset_guard)):
        raise GuardError(ReasonCode.INVALID_INPUT)
    _seconds(maximum_retry_after)
    reset_timeout = _seconds(reset_timeout_seconds)
    for index in range(1, attempts + 1):
        try:
            return await attempt(index)
        except RetryableFailure as error:
            result = error.result
            if (type(result) is not StopResult or result.writers_reaped is not True
                    or result.lease_released is not True):
                raise GuardError(ReasonCode.WRITER_UNRESOLVED) from None
            if result.samplers_drained is not True:
                reason = (result.reason
                          if result.samplers_drained is False
                          and result.reason is not ReasonCode.OK
                          else ReasonCode.STORAGE_UNKNOWN)
                raise GuardError(reason) from None
            if index == attempts:
                raise GuardError(result.reason) from None
            delay = retry_after_seconds(error.retry_after, maximum=maximum_retry_after)
            await sleep(delay)
            await _reset_once(reset, reset_guard, reset_timeout)
    raise AssertionError('Unreachable retry state.')
