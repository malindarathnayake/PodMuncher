"""Admitted, bounded local Docker workers. No artifact publication happens here.

The application supplies authorized host mounts, an image/entrypoint policy and
the singleton execution lease. Requests are untrusted data, never CLI options.
All errors crossing this boundary carry a closed reason, never Docker stderr.
"""
from __future__ import annotations

import asyncio
import collections
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from types import MappingProxyType
from uuid import uuid4

import psutil

from .admission import StoragePlacement, WorkloadGrowth, make_workload_plan
from .config import Limits
from .docker_storage import DockerStorageSession, _local_endpoint
from .domains import Capability, Device, DomainValidationError, JobState, ReasonCode
from .guards import OperationBinding, StorageSample, WriteGuard
from .hardware import AdmittedWorkerRef
from .protocol import WorkerEventRecord, WorkerEventStream, WorkerRequest

_ID = re.compile(r'[a-f0-9]{64}\Z')
_LABEL = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z')
_IMAGE = re.compile(r'[a-z0-9][a-z0-9./:_-]{0,240}@sha256:[a-f0-9]{64}\Z')
_LOCAL_IMAGE = re.compile(r'sha256:[a-f0-9]{64}\Z')
_GPU_UUID = re.compile(r'GPU-[a-fA-F0-9]{8}(?:-[a-fA-F0-9]{4}){3}-[a-fA-F0-9]{12}\Z')
_ROUTING = frozenset(('DOCKER_HOST', 'DOCKER_CONTEXT', 'DOCKER_TLS_VERIFY',
                      'DOCKER_CERT_PATH', 'DOCKER_API_VERSION'))
DEFAULT_SAMPLE_BUDGET_SECONDS = 0.95
# Docker Desktop for Mac installs its CLI (and credential helpers) here; it is
# not on PATH for non-login shells such as launchd services or ssh commands.
_DARWIN_DOCKER_DIR = '/usr/local/bin'


def _child_path(env, platform):
    """Child PATH for the docker CLI; only darwin gains the Desktop CLI dir."""
    if platform != 'darwin':
        return env
    entries = [p for p in env.get('PATH', '').split(':') if p]
    if _DARWIN_DOCKER_DIR not in entries:
        entries.append(_DARWIN_DOCKER_DIR)
    return dict(env, PATH=':'.join(entries))


class RuntimeFailure(DomainValidationError):
    def __init__(self, reason=ReasonCode.WORKER_PROTOCOL_MISMATCH):
        super().__init__(ReasonCode.parse(reason))


def _reason(error):
    if isinstance(error, DomainValidationError):
        return error.reason_code
    if isinstance(error, (TimeoutError, asyncio.TimeoutError)):
        return ReasonCode.REQUEST_TIMEOUT
    if isinstance(error, asyncio.CancelledError):
        return ReasonCode.CANCELLED
    return ReasonCode.DOCKER_UNAVAILABLE


@dataclass(frozen=True, slots=True)
class DockerTarget:
    context: str
    endpoint: str

    def __post_init__(self):
        if (type(self.context) is not str or not _LABEL.fullmatch(self.context)
            or not (_local_endpoint(self.endpoint, 'win32')
                    or _local_endpoint(self.endpoint, 'linux'))):
            raise RuntimeFailure(ReasonCode.DOCKER_CONTEXT_UNSUPPORTED)


@dataclass(frozen=True, slots=True)
class WorkerPolicy:
    """Trusted host policy, configured independently of a WorkerRequest.

    entrypoint contains a fixed executable and fixed arguments, not a shell.
    cached_proof is required for zero future image growth; it runs read-only
    before admission at every run and never overlaps the storage watchdog.
    """
    capability: Capability
    image: str
    entrypoint: tuple[str, ...]
    compressed_bytes: int
    unpacked_bytes: int
    cached_proof: Callable[[], object] | None = None
    allow_pull: bool = False
    cpus: float = 1.0
    memory_bytes: int = 512 * 1024 * 1024
    pids: int = 64
    selected_gpu_uuid: str | None = None
    development_image: bool = False

    def __post_init__(self):
        object.__setattr__(self, 'capability', Capability.parse(self.capability))
        if (type(self.development_image) is not bool or type(self.image) is not str
            or not (_LOCAL_IMAGE if self.development_image else _IMAGE).fullmatch(self.image)
            or type(self.entrypoint) is not tuple or not self.entrypoint
            or len(self.entrypoint) > 16 or any(type(v) is not str or not v
                or len(v) > 1024 or any(ord(c) < 32 for c in v) for v in self.entrypoint)
            or type(self.allow_pull) is not bool
            or type(self.cpus) not in (int, float) or not math.isfinite(self.cpus)
            or not 0.1 <= self.cpus <= 64
            or type(self.memory_bytes) is not int or not 64 * 1024**2 <= self.memory_bytes <= 128 * 1024**3
            or type(self.pids) is not int or not 8 <= self.pids <= 4096
            or any(type(v) is not int or v < 0 for v in (self.compressed_bytes, self.unpacked_bytes))):
            raise RuntimeFailure(ReasonCode.INVALID_INPUT)
        if self.development_image and self.allow_pull:
            raise RuntimeFailure(ReasonCode.UNAUTHORIZED)
        if self.selected_gpu_uuid is not None and (
            self.capability is not Capability.ASR_CUDA
            or type(self.selected_gpu_uuid) is not str
            or not _GPU_UUID.fullmatch(self.selected_gpu_uuid)):
            raise RuntimeFailure(ReasonCode.GPU_UNAVAILABLE)
        if not self.compressed_bytes or not self.unpacked_bytes:
            if self.allow_pull or not callable(self.cached_proof):
                raise RuntimeFailure(ReasonCode.STORAGE_UNKNOWN)


@dataclass(frozen=True, slots=True)
class WorkerMounts:
    """Host-authorized paths only. A probe file is host-authored executable code."""
    scratch: Path
    input_file: Path | None = None
    model: Path | None = None
    probe_file: Path | None = None

    def arguments(self, request):
        remote = request.operation in ('metadata', 'download')
        if remote and any(v is not None for v in (self.input_file, self.model, self.probe_file)):
            raise RuntimeFailure(ReasonCode.UNAUTHORIZED)
        if (self.input_file is None) != (request.input_basename is None):
            raise RuntimeFailure(ReasonCode.INVALID_INPUT)
        if (self.model is None) != (request.model_snapshot_id is None):
            raise RuntimeFailure(ReasonCode.INVALID_INPUT)
        paths = [(self.scratch, '/scratch', False, True)]
        if self.input_file is not None:
            paths.append((self.input_file, '/input/' + request.input_basename, True, False))
        if self.model is not None:
            paths.append((self.model, '/model', True, True))
        if self.probe_file is not None:
            paths.append((self.probe_file, '/probe.py', True, False))
        result, resolved = [], []
        for path, destination, readonly, directory in paths:
            if not isinstance(path, Path) or not path.is_absolute():
                raise RuntimeFailure(ReasonCode.INVALID_INPUT)
            raw = str(path)
            # --mount key/value syntax has its own separators, independent of shell=False.
            if any(c in raw for c in ',\r\n\x00') or raw.startswith(('\\\\', '//')):
                raise RuntimeFailure(ReasonCode.INVALID_INPUT)
            if any(item.is_symlink() or item.is_junction() for item in (path, *path.parents)):
                raise RuntimeFailure(ReasonCode.UNAUTHORIZED)
            if not path.exists() or path.is_dir() != directory:
                raise RuntimeFailure(ReasonCode.INVALID_INPUT)
            actual = path.resolve(strict=True)
            if 'docker.sock' in raw.lower() or actual == Path(actual.anchor):
                raise RuntimeFailure(ReasonCode.UNAUTHORIZED)
            resolved.append(actual)
            result.extend(('--mount', f'type=bind,src={actual},dst={destination}'
                           + (',readonly' if readonly else '')))
        scratch = resolved[0]
        if any(p == scratch or p.is_relative_to(scratch) or scratch.is_relative_to(p)
               for p in resolved[1:]):
            raise RuntimeFailure(ReasonCode.UNAUTHORIZED)
        return tuple(result)


class RuntimeStorage:
    """A single prepared storage identity, fresh complete budgets at each boundary; the completion budget is per instance and never exceeds the guard's staleness bound."""
    def __init__(self, session: DockerStorageSession, volumes, *, paths: Mapping[str, Path],
                 growth: WorkloadGrowth, limits: Limits, platform=os.name,
                 sample_budget_seconds: float = DEFAULT_SAMPLE_BUDGET_SECONDS):
        if (type(sample_budget_seconds) not in (int, float)
            or not math.isfinite(sample_budget_seconds)
            or not 0 < sample_budget_seconds <= 3):
            raise RuntimeFailure(ReasonCode.INVALID_INPUT)
        if (type(session) is not DockerStorageSession or not session.initial_evidence.can_create_cached
            or set(paths) != {'scratch', 'cache', 'output', 'state'}
            or type(growth) is not WorkloadGrowth or type(limits) is not Limits):
            raise RuntimeFailure(ReasonCode.STORAGE_UNKNOWN)
        self.session, self.volumes = session, volumes
        self.paths = MappingProxyType(dict(paths))
        self.growth, self.limits, self.platform = growth, limits, platform
        self.sample_budget_seconds = float(sample_budget_seconds)
        self._pending = None

    def _sample(self):
        deadline = time.monotonic() + self.sample_budget_seconds
        evidence = self.session.refresh(deadline=deadline)
        if not evidence.can_create_cached or evidence.oldest_observed_monotonic is None:
            raise RuntimeFailure(evidence.reason or ReasonCode.STORAGE_UNKNOWN)
        observed = [evidence.oldest_observed_monotonic]
        selected = {}
        for role, path in self.paths.items():
            observed.append(time.monotonic())
            selected[role] = self.volumes.sample(str(path), sqlite_state=role == 'state')
            if time.monotonic() >= deadline:
                raise RuntimeFailure(ReasonCode.STORAGE_STALE)
        roles = {role: c.volume.identity.value for c in evidence.constraints for role in c.roles}
        vm = roles.get('docker_vm', roles.get('docker_data'))
        backing = roles.get('docker_backing', vm)
        placement = StoragePlacement(*(selected[k].identity.value for k in
                                      ('scratch', 'cache', 'output', 'state')), vm, backing)
        optional = frozenset(v.identity.value for v in selected.values()) | {backing}
        plan = make_workload_plan(tuple(selected.values()), self.growth, placement, self.limits,
            now=datetime.now(timezone.utc), docker=evidence, state_evidence=selected['state'],
            inode_optional_identities=optional if self.platform in ('nt', 'win32') else frozenset())
        return StorageSample(plan, min(observed))

    async def sample(self):
        # Shield the actual thread and retain it if the watchdog's waiter expires.
        if self._pending is not None and not self._pending.done():
            raise RuntimeFailure(ReasonCode.STORAGE_UNKNOWN)
        self._pending = asyncio.create_task(asyncio.to_thread(self._sample))
        self._pending.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
        return await asyncio.shield(self._pending)


@dataclass(frozen=True, slots=True)
class CommandResult:
    returncode: int
    stdout: bytes


class NativeProcess:
    """An exact child and its IO, adopted before querying fallible OS identity.

    Windows Popen signals its native handle. Linux Popen signals through a
    verified pidfd. POSIX platforms without pidfd (macOS) signal through the
    owned, unreaped Popen: its PID stays reserved until this object reaps it,
    reaping only happens on the event-loop thread that also signals, and
    Popen.send_signal re-polls and skips a reaped child. The creation-time
    check remains mandatory with every signal route.
    ``posix``/``pidfd`` are test seams; None selects the running platform.
    """
    def __init__(self, process, owner, *, identity=psutil.Process, capture=True,
                 posix=None, pidfd=None):
        self.__process = process
        self.__identity = identity
        self.__posix = os.name != 'nt' if posix is None else bool(posix)
        self.__use_pidfd = (hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal')
                            if pidfd is None else bool(pidfd))
        self.__pid = self.__created = self.__pidfd = None
        self.__binding = OperationBinding(owner, 'native:' + uuid4().hex)
        self.tasks = set()
        self.closed = False
        if capture:
            self.capture_identity()

    def adopt(self, process):
        self.__process = process

    @property
    def process(self):
        return self.__process

    def capture_identity(self):
        self.__pid = self.__process.pid
        self.__created = self.__identity(self.__pid).create_time()
        if type(self.__created) not in (int, float) or not math.isfinite(self.__created):
            raise RuntimeFailure(ReasonCode.WRITER_UNRESOLVED)
        if (self.__posix and self.__use_pidfd and isinstance(self.__process, subprocess.Popen)
            and self.__process.poll() is None):
            self.__pidfd = os.pidfd_open(self.__pid)
            self._current(self.binding)

    @property
    def binding(self):
        return self.__binding

    def _current(self, expected):
        if expected is not self.__binding or self.__process.pid != self.__pid:
            raise RuntimeFailure(ReasonCode.WRITER_UNRESOLVED)
        if self.__process.poll() is not None:
            return False
        if self.__created is None:
            raise RuntimeFailure(ReasonCode.WRITER_UNRESOLVED)
        try:
            if self.__identity(self.__pid).create_time() != self.__created:
                raise RuntimeFailure(ReasonCode.WRITER_UNRESOLVED)
        except psutil.NoSuchProcess:
            if self.__process.poll() is not None:
                return False
            raise RuntimeFailure(ReasonCode.WRITER_UNRESOLVED) from None
        return True

    async def terminate(self, expected):
        if self._current(expected):
            self._signal(False)

    async def kill(self, expected):
        if self._current(expected):
            self._signal(True)

    def _signal(self, force):
        if self.__posix and isinstance(self.__process, subprocess.Popen) and self.__use_pidfd:
            if self.__pidfd is None:
                raise RuntimeFailure(ReasonCode.WRITER_UNRESOLVED)
            signal.pidfd_send_signal(self.__pidfd, signal.SIGKILL if force else signal.SIGTERM)
        else:
            # Windows handle, POSIX-without-pidfd owned Popen, or a trusted
            # test adapter. _current() has just confirmed the child unreaped.
            (self.__process.kill if force else self.__process.terminate)()

    def failed_identity_cleanup(self):
        # Only real Windows Popen guarantees kill/wait use the original handle.
        # A POSIX Popen object alone never authorizes a blind PID signal.
        if os.name == 'nt' and isinstance(self.__process, subprocess.Popen):
            failed = False
            for action in (self.__process.kill, lambda: self.__process.wait(timeout=0.1)):
                try:
                    action()
                except BaseException:
                    failed = True
            if failed:
                raise RuntimeFailure(ReasonCode.WRITER_UNRESOLVED)

    def track(self, awaitable):
        task = asyncio.create_task(awaitable)
        self.tasks.add(task)
        task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
        return task

    async def reap(self, expected):
        if expected is not self.__binding:
            raise RuntimeFailure(ReasonCode.WRITER_UNRESOLVED)
        if self.closed:
            return True
        if self.__process is None:
            return True
        if self._current(expected) or any(not t.done() or t.cancelled() for t in self.tasks):
            return False
        # poll alone is insufficient when wait or pipe closure fails.
        if self.__process.wait(timeout=0) is None:
            return False
        for pipe in (self.__process.stdin, self.__process.stdout, self.__process.stderr):
            if pipe is not None:
                pipe.close()
        if self.__pidfd is not None:
            os.close(self.__pidfd)
            self.__pidfd = None
        self.closed = True
        return True


def _worker_user():
    """Container --user value: the invoking host uid:gid on POSIX, fixed on Windows."""
    if not hasattr(os, 'getuid'):
        return '65532:65532'
    uid, gid = os.getuid(), os.getgid()
    if uid == 0:
        raise RuntimeFailure(ReasonCode.DOCKER_CONTEXT_UNSUPPORTED)
    return f'{uid}:{gid}'


class _NativeOwner:
    """Durable ownership includes commands spawned by stopped cleanup callbacks."""
    def __init__(self, owner):
        self.__binding = OperationBinding(owner, 'native-owner:' + uuid4().hex)
        self.children, self.cleanups = [], set()
        self.failed = self.sealed = False
        self.coordinator = None
        self.release = None

    @property
    def binding(self):
        return self.__binding

    def check(self, expected):
        if expected is not self.__binding:
            self.failed = True
            raise RuntimeFailure(ReasonCode.WRITER_UNRESOLVED)

    def spawn(self, factory, args, kwargs, identity):
        if self.sealed or self.failed or len(self.children) >= 128:
            raise RuntimeFailure(ReasonCode.WRITER_UNRESOLVED)
        child = NativeProcess(None, self.binding.owner_id, identity=identity, capture=False)
        self.children.append(child)
        # The record exists before Popen; adopt performs only the assignment.
        try:
            child.adopt(factory(args, **kwargs))
        except BaseException:
            self.children.remove(child)
            raise
        return child

    async def _signal(self, expected, method):
        self.check(expected)
        for child in tuple(self.children):
            try:
                await getattr(child, method)(child.binding)
            except BaseException:
                self.failed = True

    async def terminate(self, expected):
        await self._signal(expected, 'terminate')

    async def kill(self, expected):
        await self._signal(expected, 'kill')

    async def reap(self, expected):
        self.check(expected)
        # Container reapers run concurrently and can be draining a cleanup CLI
        # here. An immediate False would make the coordinator kill that child.
        # Await the actual IO, resampling because drains and cleanup callbacks
        # can register more work while an earlier task completes.
        seconds = self.coordinator.reap_timeout_seconds if self.coordinator is not None else 2
        deadline = time.monotonic() + seconds
        while True:
            pending = {task for child in self.children for task in child.tasks if not task.done()}
            pending.update(task for task in self.cleanups if not task.done())
            if not pending:
                break
            if asyncio.current_task() in pending:
                self.failed = True
                return False
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            await asyncio.wait(pending, timeout=remaining)
        okay = True
        for child in tuple(self.children):
            try:
                okay = await child.reap(child.binding) and okay
            except BaseException:
                self.failed = True
        return okay and not self.failed and all(t.done() and not t.cancelled() for t in self.cleanups)

    async def finish(self):
        # Resource closers run after all operation reapers. Seal before checking
        # again: concurrent container cleanup can have spawned children since
        # our first reap, and a late callback must never spawn after this proof.
        self.sealed = True
        if not await self.reap(self.binding):
            self.failed = True
            raise RuntimeFailure(ReasonCode.WRITER_UNRESOLVED)
        if self.release is not None:
            self.release()


class NativeDockerCLI:
    """Argument arrays, bounded pipes and deadlines, with owned native children.

    factory/identity are narrow test seams. They are trusted native adapters.
    Context metadata is revalidated and all operations pin the local endpoint
    using --host; changing a context between lookup and invocation cannot retarget.
    """
    def __init__(self, target: DockerTarget, *, factory=subprocess.Popen,
                 identity=psutil.Process, environ=None, max_output=4 * 1024**2,
                 max_stderr=65536, platform=sys.platform):
        if type(target) is not DockerTarget:
            raise RuntimeFailure(ReasonCode.DOCKER_CONTEXT_UNSUPPORTED)
        if any(type(n) is not int or not 0 < n <= 16 * 1024**2 for n in (max_output, max_stderr)):
            raise RuntimeFailure(ReasonCode.INVALID_INPUT)
        env = dict(os.environ if environ is None else environ)
        if any(env.get(k) for k in _ROUTING):
            raise RuntimeFailure(ReasonCode.DOCKER_CONTEXT_UNSUPPORTED)
        self.__target, self.factory, self.identity = target, factory, identity
        self.env = MappingProxyType(_child_path(
            {k: v for k, v in env.items() if k not in _ROUTING}, platform))
        self.max_output, self.max_stderr = max_output, max_stderr
        self._owners = {}
        self._released = collections.deque(maxlen=1024)

    @property
    def target(self):
        return self.__target

    def bind_owner(self, owner, coordinator=None):
        lifetime = self._owners.get(owner)
        if lifetime is None:
            # A finished owner stays sealed: a late command must not respawn it.
            if owner in self._released:
                raise RuntimeFailure(ReasonCode.WRITER_UNRESOLVED)
            lifetime = _NativeOwner(owner)
            lifetime.release = lambda: self._release(owner)
            self._owners[owner] = lifetime
        if coordinator is not None:
            coordinator.ensure_active()
            if coordinator.operation_id != owner or lifetime.coordinator not in (None, coordinator):
                raise RuntimeFailure(ReasonCode.WRITER_UNRESOLVED)
            if lifetime.coordinator is None:
                coordinator.register_operation(lifetime)
                coordinator.register_resource(lifetime.finish)
                lifetime.coordinator = coordinator
        return lifetime

    def _release(self, owner):
        if self._owners.pop(owner, None) is not None:
            self._released.append(owner)

    async def _raw(self, args, *, owner, timeout, coordinator=None, stdin=None, on_stdout=None):
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout:
            raise RuntimeFailure(ReasonCode.INVALID_INPUT)
        if stdin is not None and (type(stdin) is not bytes or len(stdin) > 65537):
            raise RuntimeFailure(ReasonCode.INVALID_INPUT)
        lifetime = self.bind_owner(owner, coordinator)
        owned = lifetime.spawn(self.factory, ('docker', *args), dict(
            stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False, env=dict(self.env),
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0), self.identity)
        process = owned.process
        try:
            owned.capture_identity()
        except BaseException:
            lifetime.failed = True
            try:
                owned.failed_identity_cleanup()
            except BaseException:
                pass
            try:
                await owned.reap(owned.binding)
            except BaseException:
                pass
            raise RuntimeFailure(ReasonCode.WRITER_UNRESOLVED) from None

        async def drain(stream, cap, consumer=None):
            size, output = 0, bytearray()
            while True:
                chunk = await asyncio.shield(owned.track(asyncio.to_thread(stream.read1, 4096)))
                if not chunk:
                    return bytes(output)
                size += len(chunk)
                if size > cap:
                    raise RuntimeFailure(ReasonCode.EVENT_LIMIT_REACHED)
                if consumer is not None:
                    await consumer(chunk)
                elif stream is process.stdout:
                    output.extend(chunk)
                # stderr is counted and discarded, never retained or published.

        async def write_request():
            if stdin is not None:
                def send():
                    # The pipe handle belongs to this exact spawned child.
                    owned._current(owned.binding)
                    process.stdin.write(stdin)
                    process.stdin.flush()
                    process.stdin.close()
                await asyncio.shield(owned.track(asyncio.to_thread(send)))

        async def collect():
            tasks = [owned.track(drain(process.stdout, self.max_output, on_stdout)),
                     owned.track(drain(process.stderr, self.max_stderr)),
                     owned.track(write_request())]
            done = await asyncio.gather(*(asyncio.shield(t) for t in tasks))
            while process.poll() is None:
                await asyncio.sleep(0.01)
            return CommandResult(process.returncode, done[0])

        task = owned.track(collect())
        failure = None
        try:
            async with asyncio.timeout(timeout):
                result = await asyncio.shield(task)
            if not await owned.reap(owned.binding):
                raise RuntimeFailure(ReasonCode.WRITER_UNRESOLVED)
            return result
        except BaseException as error:
            failure = error

        async def cleanup():
            for callback in (owned.terminate, owned.kill):
                try:
                    await callback(owned.binding)
                except BaseException:
                    lifetime.failed = True
            # Readers may schedule their next thread while the current read
            # finishes, so recheck the durable task set after each bounded wait.
            deadline = time.monotonic() + 2
            while pending := {t for t in owned.tasks if not t.done()}:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                await asyncio.wait(pending, timeout=remaining)
            try:
                if not await owned.reap(owned.binding):
                    lifetime.failed = True
            except BaseException:
                lifetime.failed = True
            if lifetime.failed:
                raise RuntimeFailure(ReasonCode.WRITER_UNRESOLVED)

        cleanup_task = asyncio.create_task(cleanup())
        lifetime.cleanups.add(cleanup_task)
        cleanup_task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
        try:
            await asyncio.shield(cleanup_task)
        except BaseException:
            lifetime.failed = True
            raise RuntimeFailure(ReasonCode.WRITER_UNRESOLVED) from None
        if isinstance(failure, asyncio.CancelledError):
            raise failure
        raise RuntimeFailure(_reason(failure)) from None

    async def run(self, *args, owner, timeout=15, coordinator=None, stdin=None, on_stdout=None):
        started = time.monotonic()
        result = await self._raw(('--context', self.target.context, 'context', 'inspect', self.target.context),
                                owner=owner, timeout=timeout, coordinator=coordinator)
        try:
            values = json.loads(result.stdout)
            if (result.returncode or len(values) != 1 or values[0]['Name'] != self.target.context
                or values[0]['Endpoints']['docker']['Host'] != self.target.endpoint):
                raise ValueError()
        except (ValueError, TypeError, KeyError):
            raise RuntimeFailure(ReasonCode.DOCKER_CONTEXT_UNSUPPORTED) from None
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise RuntimeFailure(ReasonCode.REQUEST_TIMEOUT)
        return await self._raw(('--host', self.target.endpoint, *args), owner=owner,
            timeout=remaining, coordinator=coordinator, stdin=stdin, on_stdout=on_stdout)


class OwnedContainer:
    def __init__(self, cli, reference: AdmittedWorkerRef, owner, token):
        self.__cli, self.__ref, self.__token = cli, reference, token
        self.__binding = OperationBinding(owner, reference.container_id)
        self.__removed = False

    @property
    def binding(self):
        return self.__binding

    @property
    def reference(self):
        return self.__ref

    def _expected(self, expected):
        if expected is not self.__binding or expected.target_id != self.__ref.container_id:
            raise RuntimeFailure(ReasonCode.WRITER_UNRESOLVED)

    async def inspect(self, expected):
        self._expected(expected)
        result = await self.__cli.run('container', 'inspect', expected.target_id, owner=expected.owner_id)
        self._expected(expected)
        try:
            data, = json.loads(result.stdout)
            labels = data['Config']['Labels']
            if (result.returncode or data['Id'] != expected.target_id
                or any(labels.get(k) != v for k, v in self.__ref.expected_labels.items())
                or labels.get('podmuncher.runtime') != self.__token):
                raise ValueError()
            return data
        except (ValueError, TypeError, KeyError):
            raise RuntimeFailure(ReasonCode.UNAUTHORIZED) from None

    async def _command(self, expected, command, *options):
        await self.inspect(expected)
        self._expected(expected)
        result = await self.__cli.run(command, *options, expected.target_id, owner=expected.owner_id)
        self._expected(expected)
        if result.returncode:
            raise RuntimeFailure(ReasonCode.DOCKER_UNAVAILABLE)
        return result

    async def verify_gpu(self, expected):
        data = await self.inspect(expected)
        try:
            requests = data['HostConfig']['DeviceRequests']
            if not self.__ref.gpu_visible:
                if requests not in (None, []):
                    raise ValueError()
                return
            device, = requests
            if (device['DeviceIDs'] != [self.__ref.gpu_uuid]
                or type(device['Count']) is not int or device['Count'] != 0
                or device['Driver'] not in ('', 'nvidia')
                or device['Capabilities'] != [['gpu']]
                or device['Options'] not in (None, {})):
                raise ValueError()
        except (ValueError, TypeError, KeyError):
            raise RuntimeFailure(ReasonCode.GPU_UNAVAILABLE) from None

    async def terminate(self, expected):
        self._expected(expected)
        if not self.__removed and (await self.inspect(expected))['State']['Running']:
            await self._command(expected, 'stop', '--time', '1')

    async def kill(self, expected):
        self._expected(expected)
        if not self.__removed and (await self.inspect(expected))['State']['Running']:
            await self._command(expected, 'kill')

    async def reap(self, expected):
        self._expected(expected)
        if self.__removed:
            return True
        info = await self.inspect(expected)
        self._expected(expected)
        if info['State']['Running'] or info['State'].get('Restarting'):
            return False
        await self._command(expected, 'rm')
        self._expected(expected)
        self.__removed = True
        return True

    async def logs(self, expected):
        """Bounded diagnostic bytes; callers must never forward raw sidebands."""
        return await self._command(expected, 'logs', '--tail', '256')


class _CreateRecovery:
    """The unique create transaction, including recovery from lost CLI stdout.

    Native CLI termination is never evidence that an engine writer stopped.
    This operation owns the actual create IO task, never the run controller.
    """
    def __init__(self, cli, owner, labels, token, gpu_uuid=None):
        self.cli, self.labels, self.token = cli, MappingProxyType(dict(labels)), token
        self.gpu_uuid = gpu_uuid
        self.__binding = OperationBinding(owner, 'create:' + token)
        self.task = None
        self.container = None
        self._lock = asyncio.Lock()

    @property
    def binding(self):
        return self.__binding

    def _check(self, expected):
        if expected is not self.__binding:
            raise RuntimeFailure(ReasonCode.WRITER_UNRESOLVED)

    async def _find(self, expected):
        async with self._lock:
            return await self._find_locked(expected)

    async def _find_locked(self, expected):
        self._check(expected)
        if self.container is not None:
            return self.container
        if self.task is None:
            return None
        if not self.task.done():
            await asyncio.wait({self.task}, timeout=0.1)
        self._check(expected)
        if not self.task.done():
            raise RuntimeFailure(ReasonCode.WRITER_UNRESOLVED)
        filters = tuple(v for k, value in self.labels.items() for v in ('--filter', f'label={k}={value}'))
        result = await self.cli.run('ps', '-a', '--no-trunc', '--format', '{{.ID}}', *filters,
                                    owner=expected.owner_id)
        self._check(expected)
        ids = result.stdout.decode('ascii').splitlines()
        if result.returncode or len(ids) != 1 or not _ID.fullmatch(ids[0]):
            # Empty output after an interrupted create is not proof of absence.
            raise RuntimeFailure(ReasonCode.WRITER_UNRESOLVED)
        ref = AdmittedWorkerRef(self.cli.target.context, self.cli.target.endpoint, ids[0],
            self.labels['podmuncher.instance'], self.labels['podmuncher.job'], self.labels['podmuncher.attempt'],
            self.gpu_uuid is not None, self.gpu_uuid)
        candidate = OwnedContainer(self.cli, ref, expected.owner_id, self.token)
        await candidate.inspect(candidate.binding)
        self._check(expected)
        self.container = candidate
        return candidate

    async def terminate(self, expected):
        item = await self._find(expected)
        self._check(expected)
        if item is not None:
            await item.terminate(item.binding)

    async def kill(self, expected):
        item = await self._find(expected)
        self._check(expected)
        if item is not None:
            await item.kill(item.binding)

    async def reap(self, expected):
        item = await self._find(expected)
        self._check(expected)
        return item is None or await item.reap(item.binding)


class _PullRecovery:
    """An interrupted daemon pull has no cancellation API: retain its lease."""
    def __init__(self, owner, image):
        self.__binding = OperationBinding(owner, 'pull:' + image)
        self.task = None

    @property
    def binding(self):
        return self.__binding

    async def terminate(self, expected):
        if expected is not self.__binding:
            raise RuntimeFailure(ReasonCode.WRITER_UNRESOLVED)

    async def kill(self, expected):
        await self.terminate(expected)

    async def reap(self, expected):
        await self.terminate(expected)
        if self.task is None:
            return True
        if not self.task.done() or self.task.cancelled():
            return False
        try:
            return self.task.result().returncode == 0
        except Exception:
            return False


@dataclass(frozen=True, slots=True)
class WorkerResult:
    reference: AdmittedWorkerRef
    completed: bool
    exit_code: int
    artifact_claims: tuple[WorkerEventRecord, ...]


class DockerRuntime:
    def __init__(self, cli: NativeDockerCLI, guard: WriteGuard, *, instance: str,
                 policy: WorkerPolicy, mounts: WorkerMounts):
        if type(instance) is not str or not _LABEL.fullmatch(instance):
            raise RuntimeFailure(ReasonCode.INVALID_INPUT)
        self.cli, self.guard, self.instance = cli, guard, instance
        self.policy, self.mounts = policy, mounts
        self.reference = None
        self._used = False
        self._current = True
        self._proof_task = None

    def obsolete(self):
        self._current = False
        self.guard.coordinator.request_stop(ReasonCode.CANCELLED)

    def create_arguments(self, request, token):
        if type(request) is not WorkerRequest or Capability.parse(request.capability) is not self.policy.capability:
            raise RuntimeFailure(ReasonCode.WORKER_PROTOCOL_MISMATCH)
        remote = request.operation in ('metadata', 'download')
        device = Device.parse(request.selection.device)
        if ((self.policy.capability is Capability.ASR_CPU and device is Device.CUDA)
            or (self.policy.capability is Capability.ASR_CUDA and device is Device.CPU)):
            raise RuntimeFailure(ReasonCode.WORKER_PROTOCOL_MISMATCH)
        gpu_args = ()
        if self.policy.capability is Capability.ASR_CUDA:
            if self.policy.selected_gpu_uuid is None:
                raise RuntimeFailure(ReasonCode.GPU_UNAVAILABLE)
            gpu_args = ('--gpus', 'device=' + self.policy.selected_gpu_uuid)
        labels = {'podmuncher.instance': self.instance, 'podmuncher.job': request.job_id,
                  'podmuncher.attempt': request.attempt_id, 'podmuncher.runtime': token}
        args = ('create', '--name', 'podmuncher-' + token, '--pull', 'never', '--interactive',
            '--read-only', '--user', _worker_user(), '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges',
            '--cpus', str(self.policy.cpus), '--memory', str(self.policy.memory_bytes),
            '--memory-swap', str(self.policy.memory_bytes), '--pids-limit', str(self.policy.pids),
            '--log-driver', 'json-file', '--log-opt', 'max-size=10m', '--log-opt', 'max-file=2',
            '--network', 'bridge' if remote else 'none', '--workdir', '/scratch',
            *gpu_args,
            '--env', 'PYTHONDONTWRITEBYTECODE=1', '--env', 'PYTHONUNBUFFERED=1',
            *tuple(v for k, value in labels.items() for v in ('--label', k + '=' + value)),
            *self.mounts.arguments(request), '--entrypoint', self.policy.entrypoint[0],
            self.policy.image, *self.policy.entrypoint[1:])
        return args, labels

    async def run(self, request: WorkerRequest, *, state=JobState.CHECKING, on_event=None):
        if JobState.parse(state) is not JobState.CHECKING or self._used:
            raise RuntimeFailure(ReasonCode.CONFLICT)
        self._used = True
        token = uuid4().hex
        args, labels = self.create_arguments(request, token)
        wire = request.to_json() + b'\n'
        coordinator = self.guard.coordinator
        owner = coordinator.operation_id
        reason, result = ReasonCode.OK, None
        started, completed = False, False
        claims = []
        stream = WorkerEventStream(request)

        async def receive(chunk):
            nonlocal started, completed
            if coordinator.stopped.is_set() or not self._current:
                return
            stream.feed(chunk)
            while (event := stream.pop()) is not None:
                if coordinator.stopped.is_set() or not self._current:
                    return
                if completed or (not started and event.event != 'started'):
                    raise RuntimeFailure()
                if event.event == 'started':
                    if started:
                        raise RuntimeFailure()
                    started = True
                elif event.event == 'completed':
                    completed = True
                elif event.event == 'artifact':
                    claims.append(event)
                elif event.event == 'error':
                    reported = ReasonCode.parse(event.payload['reason_code'])
                    raise RuntimeFailure(reported if reported is not ReasonCode.OK
                                         else ReasonCode.WORKER_PROTOCOL_MISMATCH)
                if on_event is not None:
                    on_event(event)

        unresolved = False
        try:
            if not self.policy.compressed_bytes or not self.policy.unpacked_bytes:
                # Read-only proof precedes admission so it never overlaps the
                # storage watchdog. The thread task is retained, never cancelled.
                self._proof_task = asyncio.create_task(asyncio.to_thread(self.policy.cached_proof))
                self._proof_task.add_done_callback(
                    lambda task: task.exception() if not task.cancelled() else None)
                proof = await asyncio.shield(self._proof_task)
                if proof is not True:
                    raise RuntimeFailure(ReasonCode.STORAGE_UNKNOWN)
            if not self._current or coordinator.stopped.is_set():
                raise RuntimeFailure(ReasonCode.CANCELLED)
            await self.guard.start()
            self.cli.bind_owner(owner, coordinator)
            if self.policy.allow_pull:
                await self.guard.transition()
                pull = _PullRecovery(owner, self.policy.image)
                coordinator.register_operation(pull)
                pull.task = coordinator.track(self.cli.run('pull', self.policy.image,
                    owner=owner, timeout=self.guard.timeout_seconds, coordinator=coordinator))
                pulled = await asyncio.shield(pull.task)
                if pulled.returncode:
                    raise RuntimeFailure(ReasonCode.DOCKER_UNAVAILABLE)
            await self.guard.transition()
            if self.policy.development_image:
                inspected = await self.guard.io(lambda: self.cli.run('image', 'inspect', self.policy.image,
                    owner=owner, coordinator=coordinator))
                try:
                    image, = json.loads(inspected.stdout)
                    if inspected.returncode or image['Id'] != self.policy.image:
                        raise ValueError()
                except (ValueError, TypeError, KeyError):
                    raise RuntimeFailure(ReasonCode.DOCKER_UNAVAILABLE) from None
                await self.guard.transition()
            recovery = _CreateRecovery(self.cli, owner, labels, token, self.policy.selected_gpu_uuid)
            coordinator.register_operation(recovery)
            # Register the actual IO, not this controller (which awaits finish).
            recovery.task = coordinator.track(self.cli.run(*args, owner=owner, coordinator=coordinator))
            created = await asyncio.shield(recovery.task)
            container = await recovery._find(recovery.binding)
            if created.returncode or created.stdout.strip().decode('ascii') != container.binding.target_id:
                raise RuntimeFailure(ReasonCode.DOCKER_UNAVAILABLE)
            await container.verify_gpu(container.binding)
            self.reference = container.reference
            await self.guard.transition()
            await container.verify_gpu(container.binding)
            attached = await self.guard.io(lambda: self.cli.run('start', '--attach', '--interactive',
                container.binding.target_id, owner=owner, timeout=self.guard.timeout_seconds,
                coordinator=coordinator, stdin=wire, on_stdout=receive))
            stream.finish()
            if not self._current or coordinator.stopped.is_set():
                raise RuntimeFailure(ReasonCode.CANCELLED)
            waited = await self.guard.io(lambda: container._command(container.binding, 'wait'))
            try:
                exit_code = int(waited.stdout.strip())
            except ValueError:
                raise RuntimeFailure() from None
            if not started or not completed:
                raise RuntimeFailure()
            if attached.returncode or exit_code != 0:
                raise RuntimeFailure(ReasonCode.WORKER_EXIT_UNEXPECTED)
            await self.guard.transition()
            result = WorkerResult(container.reference, True, exit_code, tuple(claims))
        except BaseException as error:
            reason = _reason(error)
            if reason is ReasonCode.OK:
                reason = ReasonCode.WORKER_PROTOCOL_MISMATCH
        finally:
            self._current = False
            if self._proof_task is not None and not self._proof_task.done():
                # Mirror the coordinator: an unfinished retained task is unresolved.
                await asyncio.wait({self._proof_task}, timeout=coordinator.reap_timeout_seconds)
                if not self._proof_task.done():
                    unresolved = True
                    reason = ReasonCode.WRITER_UNRESOLVED
            stopped = await self.guard.finish(reason)
            if stopped.reason is not ReasonCode.OK:
                reason = stopped.reason
            if unresolved:
                reason = ReasonCode.WRITER_UNRESOLVED
        if reason is not ReasonCode.OK:
            raise RuntimeFailure(reason) from None
        if result is None:
            raise RuntimeFailure() from None
        return result


__all__ = ('DockerTarget', 'WorkerPolicy', 'WorkerMounts', 'RuntimeStorage', 'NativeDockerCLI',
           'NativeProcess', 'OwnedContainer', 'DockerRuntime', 'RuntimeFailure', 'WorkerResult')
