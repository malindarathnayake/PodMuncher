"""Image-independent, read-only Docker storage evidence.

Only context inspection, info, and host/WSL metadata commands are used. No
container or image must exist. Every allocation needs all independent host and
VM constraints; aliases never contribute additive free space. Evidence is not
a reservation. Paths and daemon output are protected data, not diagnostics.

Native Linux requires locally bound runtime containerd storage exports. Native
Desktop discovery supports the 4.30+ 4.x WSL findmnt metadata layout, including
upgrades retaining docker-desktop-data. Version alone proves nothing: the actual
data mount and independent active-disk/VHDX association are required. Native
association is deliberately unresolved when no supported read-only API proves
it; a settings path, file existence, or raw VHDX header is insufficient. Injectable
platform providers may supply that independently verified association. Unknown,
remote, stale and unverified rootless layouts block pulls AND cached creation.

Darwin (Docker Desktop for Mac) keeps the engine inside a VM whose disk is one
sparse ``Docker.raw`` on a local host volume. The VM disk limit is that file's
apparent size, its allocation is the in-use bytes, and the host volume comes
from the ordinary volume provider, so the one ``docker_data`` constraint shares
the host volume's identity and budget. Free space is the tighter of the disk
headroom and the host volume; deleted VM data may still be allocated, so this
under-reports, never over-reports. An unresolved or ambiguous ``Docker.raw``
is ``backing_unresolved`` until the operator names it (``docker_backing``).
"""
from __future__ import annotations

import json
import math
import ntpath
import os
import posixpath
import re
import shutil
import stat
import subprocess
import sys
import threading
import time
import tomllib
import uuid
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from .domains import FieldStatus, ReasonCode
from .snapshots import Measurement
from .volumes import VolumeEvidence, make_volume_provider


class DiscoveryError(Exception):
    """Closed diagnostics; never interpolate command, path, or stderr."""
    def __init__(self, reason: ReasonCode):
        self.reason = ReasonCode.parse(reason)
        super().__init__(self.reason.value)


@dataclass(frozen=True, slots=True)
class ProcessResult:
    returncode: int
    stdout: bytes


class _OwnedProcesses:
    """A finite process group; native work never runs on detached Python threads."""
    def __init__(self, runner, deadline, env):
        self.runner, self.deadline, self.env = runner, deadline, dict(env)
        # Reserve termination/drain time *inside* the caller's absolute budget.
        self.work_deadline = deadline - min(0.075, max(0, deadline - time.monotonic()) / 4)
        self.children, self.buffers, self.readers = [], [], []
        self.supervised, self.tails = [], []
        self.lock, self.overflow = threading.Lock(), threading.Event()
        self.used = 0

    def check(self):
        if self.overflow.is_set():
            raise DiscoveryError(ReasonCode.REQUEST_TOO_LARGE)
        _check_deadline(self.work_deadline)

    def spawn(self, argv, *, interactive=False):
        self.check()
        try:
            child = self.runner.popen(tuple(argv), shell=False,
                stdin=subprocess.PIPE if interactive else subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=self.env)
        except OSError:
            raise DiscoveryError(ReasonCode.DOCKER_UNAVAILABLE) from None
        key = len(self.children)
        self.children.append(child)
        self.supervised.append(_WSL_SUPERVISOR in argv)
        self.tails.append(b'')
        buffer, readers = [], []
        self.buffers.append(buffer)
        self.readers.append(readers)
        def drain(stream, keep):
            try:
                read = getattr(stream, 'read1', stream.read)
                while True:
                    chunk = read(4096)
                    if not chunk:
                        break
                    with self.lock:
                        if keep:
                            self.tails[key] = (self.tails[key] + chunk)[-128:]
                        self.used += len(chunk)
                        if self.used > self.runner.max_output:
                            self.overflow.set()
                        elif keep:
                            buffer.append(chunk)
            except (OSError, ValueError):
                self.overflow.set()
            finally:
                stream.close()
        for stream, keep in ((child.stdout, True), (child.stderr, False)):
            thread = threading.Thread(target=drain, args=(stream, keep), daemon=True)
            readers.append(thread)
            thread.start()
        return key

    def output(self, key):
        with self.lock:
            return b''.join(self.buffers[key])

    def complete(self, key):
        return self.children[key].poll() is not None and not any(r.is_alive() for r in self.readers[key])

    def wait(self, predicate):
        while True:
            self.check()
            if any(c.poll() not in (None, 0) for c in self.children):
                raise DiscoveryError(ReasonCode.DOCKER_UNAVAILABLE)
            if predicate():
                return
            time.sleep(min(0.001, max(0, self.work_deadline - time.monotonic())))

    def result(self, key):
        self.check()
        return ProcessResult(self.children[key].returncode, self.output(key))

    def signal(self, key):
        self.check()
        # Exactly two tiny writes to an otherwise empty owned pipe; the fixed
        # witness protocol reads one line per stage, never arbitrary input.
        try:
            self.children[key].stdin.write(b'continue\n')
            self.children[key].stdin.flush()
        except (OSError, ValueError):
            raise DiscoveryError(ReasonCode.STORAGE_UNKNOWN) from None

    def close(self):
        failed = False
        # EOF reaches the Linux supervisor independently of a stalled worker.
        # Give its verified group cleanup time before terminating wsl.exe.
        for child in self.children:
            stream = getattr(child, 'stdin', None)
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    # An already closed peer can reject a buffered write. The
                    # wait/reader checks and Linux acknowledgement prove cleanup.
                    pass
        cleanup_end = min(self.deadline, time.monotonic() + 0.15)
        while (any(owned and not self.complete(i) for i, owned in enumerate(self.supervised))
               and time.monotonic() < cleanup_end):
            time.sleep(min(0.001, max(0, cleanup_end - time.monotonic())))
        for child in self.children:
            try:
                if child.poll() is None:
                    child.kill()
            except OSError:
                failed = True
        for child in self.children:
            try:
                child.wait(timeout=min(1, max(0, self.deadline - time.monotonic())))
            except (OSError, subprocess.TimeoutExpired):
                failed = True
        for readers in self.readers:
            for reader in readers:
                reader.join(timeout=max(0, self.deadline - time.monotonic()))
                failed |= reader.is_alive()
        failed |= any(owned and not self.tails[i].endswith(b'@@reaped\n')
                      for i, owned in enumerate(self.supervised))
        return not failed


class BoundedRunner:
    """Cap combined output and own/terminate/reap children under one deadline."""
    def __init__(self, *, popen=subprocess.Popen, max_output: int = 1048576):
        if type(max_output) is not int or not 1 <= max_output <= 8388608:
            raise ValueError('Invalid output bound')
        self.popen, self.max_output = popen, max_output

    def __call__(self, argv: tuple[str, ...], *, deadline: float,
                 env: Mapping[str, str]) -> ProcessResult:
        return self.parallel((argv,), deadline=deadline, env=env)[0]

    def parallel(self, commands, *, deadline, env):
        if not 1 <= len(commands) <= 4:
            raise ValueError('Invalid parallel bound')
        group = _OwnedProcesses(self, deadline, env)
        try:
            transports = [_wsl_transport_command(tuple(argv), deadline) for argv in commands]
            keys = [group.spawn(argv, interactive=supervised)
                    for argv, supervised in transports]
            group.wait(lambda: all(group.complete(key) for key in keys))
            results = tuple(_wsl_command_result(group.result(key)) if supervised else group.result(key)
                            for key, (_, supervised) in zip(keys, transports))
            if any(result.returncode for result in results):
                raise DiscoveryError(ReasonCode.DOCKER_UNAVAILABLE)
            return results
        finally:
            if not group.close():
                raise DiscoveryError(ReasonCode.SERVICE_UNAVAILABLE)

    def observation(self, context, info, host, witness, *, deadline, env, validate):
        """One WSL invocation brackets every host observation via stage signals.

        ``validate`` checks actual identity/context before allowing daemon info.
        WSL's final identity runs only after info, host and VM capacities finish.
        """
        group = _OwnedProcesses(self, deadline, env)
        try:
            first = group.spawn(context)
            vm = group.spawn(witness, interactive=True)
            group.wait(lambda: group.complete(vm) or
                       group.complete(first) and b'@@ready\n' in group.output(vm))
            if group.complete(vm) or not group.output(vm).endswith(b'@@ready\n'):
                raise DiscoveryError(ReasonCode.STORAGE_UNKNOWN)
            validate(group.result(first), _witness_identity(group.output(vm), b'before'))
            daemon, backing = group.spawn(info), group.spawn(host)
            group.signal(vm)
            group.wait(lambda: group.complete(vm) or group.complete(daemon) and group.complete(backing)
                       and b'@@sampled\n' in group.output(vm))
            if group.complete(vm) or not group.output(vm).endswith(b'@@sampled\n'):
                raise DiscoveryError(ReasonCode.STORAGE_UNKNOWN)
            last = group.spawn(context)
            group.signal(vm)
            group.wait(lambda: group.complete(last) and group.complete(vm))
            output = group.output(vm)
            if group.supervised[vm]:
                output = _witness_finished(output)
            validate(group.result(last), _witness_identity(output, b'after'))
            return group.result(daemon), _witness_batch(output), group.result(backing)
        finally:
            if not group.close():
                raise DiscoveryError(ReasonCode.SERVICE_UNAVAILABLE)


@dataclass(frozen=True, slots=True)
class Store:
    role: str
    path: str
    volume: VolumeEvidence


@dataclass(frozen=True, slots=True)
class Constraint:
    """One actual allocation volume, with all roles/aliases retained."""
    roles: tuple[str, ...]
    paths: tuple[str, ...]
    volume: VolumeEvidence


@dataclass(frozen=True, slots=True)
class BackingAssociation:
    vm_identity: Measurement[str]
    host_identity: Measurement[str]
    verified: Measurement[bool]


@dataclass(frozen=True, slots=True)
class LinuxLayout:
    roots: tuple[str, ...]
    verified: bool
    rootless_verified: bool = False
    identity: LinuxIdentity | None = None


@dataclass(frozen=True, slots=True)
class LinuxIdentity:
    """Process and owned listening-socket witnesses, never just PID strings."""
    daemon_pid: str
    daemon: tuple
    docker_socket: tuple
    container_pid: str
    container: tuple
    container_socket: tuple
    address: str


@dataclass(frozen=True, slots=True)
class WSLIdentity:
    fields: tuple[tuple[str, object], ...]


@dataclass(frozen=True, slots=True)
class WSLPreparedProof:
    mount: tuple[tuple[str, str], ...]
    container_mount: tuple[tuple[str, str], ...]
    identity: WSLIdentity
    witnesses: tuple


@dataclass(frozen=True, slots=True)
class WSLLayout:
    desktop_version: str
    data_root: str
    mount_target: str
    mount_device: str
    vm_volume: VolumeEvidence
    candidate_vhdx: str | None = None
    active_disk_id: str | None = None
    candidate_disk_id: str | None = None
    association_verified: bool = False
    data_path: str = ''
    proof: WSLPreparedProof | None = None
    host_file: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class DarwinIdentity:
    """The engine socket Docker Desktop creates per VM start; a restart renews it."""
    socket: str
    file: tuple[int, int]


@dataclass(frozen=True, slots=True)
class DarwinLayout:
    """Docker Desktop for Mac: one sparse Docker.raw holding the whole VM disk.

    ``limit`` is the apparent size (the configured disk limit); ``used`` is the
    allocated size; ``file`` is the (st_dev, st_ino) replacement witness.
    """
    desktop_version: str | None
    data_root: str
    backing: str
    file: tuple[int, int]
    limit: int
    used: int


class PlatformProvider(Protocol):
    def linux_layout(self, info: dict, call: Callable) -> LinuxLayout: ...
    def wsl_layout(self, info: dict, call: Callable, manual_backing: str) -> WSLLayout: ...
    def darwin_layout(self, info: dict, call: Callable, manual_backing: str) -> DarwinLayout: ...


@dataclass(frozen=True, slots=True)
class DockerStorageEvidence:
    context: Measurement[str]
    data_root: Measurement[str]
    desktop_version: Measurement[str]
    constraints: tuple[Constraint, ...]
    backing: tuple[BackingAssociation, ...]
    admission: Measurement[bool]
    # Conservative monotonic lower bound of the oldest real observation.
    # None on legacy constructed evidence or failed/invalidated observations.
    oldest_observed_monotonic: float | None = None

    @property
    def can_pull(self) -> bool:
        return self.admission.status is FieldStatus.OK and self.admission.value is True

    @property
    def can_create_cached(self) -> bool:
        return self.can_pull

    @property
    def reason(self) -> ReasonCode | None:
        return self.admission.reason


def _absolute_posix(value):
    return (type(value) is str and value.startswith('/') and not value.startswith('//')
            and '\x00' not in value and '..' not in value.split('/') and len(value) <= 8192)


def _json(result: ProcessResult):
    if result.returncode != 0:
        raise DiscoveryError(ReasonCode.DOCKER_UNAVAILABLE)
    if len(result.stdout) > 1048576:
        raise DiscoveryError(ReasonCode.REQUEST_TOO_LARGE)
    try:
        return json.loads(result.stdout)
    except (ValueError, UnicodeError, RecursionError):
        raise DiscoveryError(ReasonCode.STORAGE_UNKNOWN) from None


def _local_endpoint(endpoint: object, platform: str) -> bool:
    if type(endpoint) is not str:
        return False
    if platform == 'win32':
        return endpoint in ('npipe:////./pipe/docker_engine',
                            'npipe:////./pipe/dockerDesktopLinuxEngine')
    return endpoint.startswith('unix:///') and _absolute_posix(endpoint[7:])


def _usable(volume: VolumeEvidence, now: datetime, max_age: float, *, require_inodes=True) -> bool:
    required = ('identity', 'capacity', 'free', 'mounted', 'writable', 'network')
    for name in required:
        field = getattr(volume, name)
        if field.status is not FieldStatus.OK:
            return False
        age = (now - datetime.fromisoformat(field.sample_time)).total_seconds()
        if not 0 <= age <= max_age:
            raise DiscoveryError(ReasonCode.STORAGE_STALE)
    if require_inodes and volume.free_inodes.status is not FieldStatus.OK:
        return False
    if volume.free_inodes.status is FieldStatus.OK:
        age = (now - datetime.fromisoformat(volume.free_inodes.sample_time)).total_seconds()
        if not 0 <= age <= max_age:
            raise DiscoveryError(ReasonCode.STORAGE_STALE)
    return (type(volume.identity.value) is str and bool(volume.identity.value)
            and type(volume.capacity.value) is int and volume.capacity.value > 0
            and type(volume.free.value) is int and 0 < volume.free.value <= volume.capacity.value
            and volume.mounted.value is True and volume.writable.value is True
            and volume.network.value is False and volume.free_inodes.value != 0)


def _deduplicate(stores: list[Store]) -> tuple[Constraint, ...]:
    result: list[Constraint] = []
    for store in stores:
        found = next((i for i, item in enumerate(result)
                      if store.volume.identity.value is not None
                      and item.volume.identity.value == store.volume.identity.value), None)
        if found is None:
            result.append(Constraint((store.role,), (store.path,), store.volume))
            continue
        old = result[found]
        # All samples are checked before merging; preserve the tightest counts.
        volume = old.volume
        for key in ('capacity', 'free', 'free_inodes'):
            left, right = getattr(volume, key), getattr(store.volume, key)
            if left.value is not None and right.value is not None and right.value < left.value:
                volume = replace(volume, **{key: right})
        result[found] = Constraint(tuple(dict.fromkeys((*old.roles, store.role))),
                                   tuple(dict.fromkeys((*old.paths, store.path))), volume)
    return tuple(result)


_DARWIN_DOCKER = '/usr/local/bin/docker'


def _docker_executable(platform: str, environ: Mapping[str, str]) -> str:
    """Docker Desktop for Mac links its CLI into /usr/local/bin, which non-login
    shells (ssh, launchd) omit from PATH. Resolution follows the child's PATH."""
    if platform != 'darwin' or shutil.which('docker', path=os.pathsep.join(os.get_exec_path(environ))):
        return 'docker'
    return _DARWIN_DOCKER if os.access(_DARWIN_DOCKER, os.X_OK) else 'docker'


def _darwin_key(layout):
    # Counters (limit/used) are re-measured on refresh; the target is the key.
    return layout.desktop_version, layout.data_root, layout.backing, layout.file


def _darwin_store_volume(layout: DarwinLayout, host: VolumeEvidence) -> VolumeEvidence:
    """The VM can grow only into min(disk headroom, host free), on the host volume."""
    free = host.free
    if free.status is FieldStatus.OK and type(free.value) is int:
        headroom = max(0, layout.limit - layout.used)
        if headroom < free.value:
            free = replace(free, value=headroom, source='docker_desktop_disk')
    return replace(host, free=free)


def _darwin_desktop_version(call, context):
    name = _json(call(('docker', '--context', context, 'version', '--format',
                       '{{json .Server.Platform.Name}}')))
    match = re.fullmatch(r'Docker Desktop (\d+\.\d+\.\d+)(?: \(\d+\))?', name) if type(name) is str else None
    return match[1] if match else None


class DockerStorageProvider:
    def __init__(self, *, runner=None, volumes=None, platform_provider=None,
                 platform: str = sys.platform, environ: Mapping[str, str] | None = None,
                 clock: Callable[[], datetime] | None = None, timeout: float = 20,
                 max_age: float = 10):
        if not 0 < timeout <= 120 or not 0 < max_age <= 60:
            raise ValueError('Invalid discovery bounds')
        self.runner = runner if runner is not None else BoundedRunner()
        self._native_volume_provider = volumes is None
        self.volumes = volumes if volumes is not None else make_volume_provider()
        self.platform_provider = platform_provider if platform_provider is not None else NativePlatformProvider()
        self.platform, self.environ = platform, os.environ if environ is None else environ
        self.clock = clock if clock is not None else lambda: datetime.now(UTC)
        self.timeout, self.max_age = timeout, max_age

    def prepare(self, *, manual_backing: str = '') -> DockerStorageSession:
        """Discover once before writers start. Failed preparation stays invalid.

        Refresh never discovers a replacement. Stop/reap writers and prepare a
        new session after invalidation. Neither preparation nor refresh reserves
        storage; caller must retain the ordinary admission and writer lease.
        """
        captured = []
        evidence = self._sample(manual_backing=manual_backing, captured=captured)
        return DockerStorageSession(self, evidence, captured[0] if captured else None)

    def sample(self, *, manual_backing: str = '') -> DockerStorageEvidence:
        return self._sample(manual_backing=manual_backing)

    def _refresh_windows(self, baseline, deadline):
        observed = time.monotonic()
        stamp = self.clock().isoformat().replace('+00:00', 'Z')
        def measured(value, source='docker_metadata'):
            return Measurement(value, source, FieldStatus.OK, stamp, None)
        evidence = DockerStorageEvidence(measured(baseline.context), measured(baseline.root),
            measured(baseline.layout.desktop_version, 'desktop_install'), (), (), measured(True))
        try:
            original = dict(self.environ)
            if tuple(sorted(original.items())) != baseline.environment:
                raise DiscoveryError(ReasonCode.STORAGE_STALE)
            clean = {k: v for k, v in original.items() if k not in ('DOCKER_HOST', 'DOCKER_CONTEXT')}
            def parallel(commands):
                _check_deadline(deadline)
                if isinstance(self.runner, BoundedRunner):
                    results = self.runner.parallel(commands, deadline=deadline, env=clean)
                else:  # Explicit injected transport; same absolute deadline.
                    results = tuple(self.runner(c, deadline=deadline, env=clean) for c in commands)
                _check_deadline(deadline)
                if any(r.returncode for r in results):
                    raise DiscoveryError(ReasonCode.DOCKER_UNAVAILABLE)
                return results
            # No-name inspect resolves CurrentContext, like context show, while
            # returning Name + endpoint together. Named inspect honors the pinned
            # explicit context, after rejecting a changed original environment.
            context_command = ('docker', 'context', 'inspect') + (
                (original['DOCKER_CONTEXT'],) if original.get('DOCKER_CONTEXT') else ())
            identity_command = (*_WSL_PREFIX, '/bin/sh', '-c', _WSL_IDENTITY, 'storage-identity')
            def validate_bracket(context_result, identity_result):
                contexts = _json(context_result)
                if (type(contexts) is not list or len(contexts) != 1
                        or contexts[0]['Name'] != baseline.context
                        or contexts[0]['Endpoints']['docker']['Host'] != baseline.endpoint
                        or identity_result != baseline.identity):
                    raise DiscoveryError(ReasonCode.STORAGE_STALE)
            layout = baseline.layout
            if not isinstance(layout.proof, WSLPreparedProof) or layout.proof.identity != baseline.identity:
                raise DiscoveryError(ReasonCode.STORAGE_STALE)
            # These observations are independent, all strictly inside the
            # complete daemon/context identity bracket above and below.
            batch_stamp = self.clock().isoformat().replace('+00:00', 'Z')
            info_command = ('docker', '--context', baseline.context, 'info', '--format', '{{json .}}')
            if isinstance(self.runner, BoundedRunner):
                info, wsl, host = self.runner.observation(context_command, info_command,
                    _host_command(layout.candidate_vhdx), _wsl_observation_command(layout, deadline),
                    deadline=deadline, env=clean, validate=validate_bracket)
            else:
                before = parallel((context_command, identity_command))
                validate_bracket(before[0], _identity_record(_proof_sections(before[1], _IDENTITY_NAMES)))
                info, wsl, host = parallel((info_command, _wsl_command(layout),
                                           _host_command(layout.candidate_vhdx)))
            if _info_key(_json(info)) != baseline.info:
                raise DiscoveryError(ReasonCode.STORAGE_STALE)
            vm = _wsl_refresh(lambda argv: wsl, layout, observation_stamp=batch_stamp)
            host_data = _json(host)
            host_volume = _host_volume(host_data, layout)
            if layout.host_file is None or tuple(host_data['file']) != layout.host_file:
                raise DiscoveryError(ReasonCode.STORAGE_STALE)
            stores = [Store('docker_vm', baseline.root, vm),
                      Store('docker_backing', layout.candidate_vhdx, host_volume)]
            if not isinstance(self.runner, BoundedRunner):
                after = parallel((context_command, identity_command))
                validate_bracket(after[0], _identity_record(_proof_sections(after[1], _IDENTITY_NAMES)))
            if tuple(sorted(self.environ.items())) != baseline.environment or _store_keys(stores) != baseline.stores:
                raise DiscoveryError(ReasonCode.STORAGE_STALE)
            now = self.clock()
            for store in stores:
                if not _usable(store.volume, now, self.max_age, require_inodes=store.role != 'docker_backing'):
                    raise DiscoveryError(ReasonCode.STORAGE_UNKNOWN)
            _check_deadline(deadline)
            return replace(evidence, constraints=_deduplicate(stores),
                backing=(BackingAssociation(vm.identity, host_volume.identity,
                    Measurement(True, 'verified_disk_association', FieldStatus.OK, batch_stamp, None)),),
                admission=replace(evidence.admission, sample_time=now.isoformat().replace('+00:00', 'Z')),
                oldest_observed_monotonic=observed)
        except DiscoveryError as error:
            return _denied(evidence, error.reason, self.clock)
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            return _denied(evidence, ReasonCode.STORAGE_UNKNOWN, self.clock)

    def _sample(self, *, manual_backing='', deadline=None, captured=None,
                expected=None) -> DockerStorageEvidence:
        observed = time.monotonic()
        now = self.clock()
        if now.tzinfo is None or now.utcoffset().total_seconds() != 0:
            raise ValueError('UTC clock required')
        stamp = now.isoformat().replace('+00:00', 'Z')
        def measured(value, source='docker_metadata', reason=ReasonCode.STORAGE_UNKNOWN):
            return Measurement(value, source, FieldStatus.OK if value is not None else (
                FieldStatus.STALE if reason is ReasonCode.STORAGE_STALE else FieldStatus.UNKNOWN),
                stamp, None if value is not None else reason)
        context = root = version = None
        stores: list[Store] = []
        backing: list[BackingAssociation] = []
        failure = None
        original = dict(self.environ)
        clean_env = {key: value for key, value in original.items()
                     if key not in ('DOCKER_HOST', 'DOCKER_CONTEXT')}
        deadline = observed + self.timeout if deadline is None else deadline
        executable = _docker_executable(self.platform, clean_env)

        def call(argv):
            if time.monotonic() >= deadline:
                raise DiscoveryError(ReasonCode.REQUEST_TIMEOUT)
            if argv and argv[0] == 'docker' and executable != 'docker':
                argv = (executable, *argv[1:])
            result = self.runner(tuple(argv), deadline=deadline, env=clean_env)
            if result.returncode != 0:
                raise DiscoveryError(ReasonCode.DOCKER_UNAVAILABLE)
            return result

        call.deadline = deadline  # Native /proc work shares the transport budget.
        native = isinstance(self.platform_provider, NativePlatformProvider)

        def inspect(name):
            data = _json(call(('docker', 'context', 'inspect', name)))
            try:
                if type(data) is not list or len(data) != 1 or data[0]['Name'] != name:
                    raise ValueError()
                endpoint = data[0]['Endpoints']['docker']['Host']
            except (KeyError, TypeError, ValueError):
                raise DiscoveryError(ReasonCode.DOCKER_CONTEXT_UNSUPPORTED) from None
            if not _local_endpoint(endpoint, self.platform):
                raise DiscoveryError(ReasonCode.DOCKER_CONTEXT_UNSUPPORTED)
            return endpoint

        def selected():
            if original.get('DOCKER_CONTEXT'):
                return original['DOCKER_CONTEXT']
            result = call(('docker', 'context', 'show')).stdout
            try:
                name = result.decode('utf-8').strip()
            except UnicodeError:
                raise DiscoveryError(ReasonCode.DOCKER_CONTEXT_UNSUPPORTED) from None
            return name

        try:
            if expected is not None and tuple(sorted(original.items())) != expected.environment:
                raise DiscoveryError(ReasonCode.STORAGE_STALE)
            # Reject dangerous environment routing before talking to a daemon.
            host = original.get('DOCKER_HOST')
            if host and not _local_endpoint(host, self.platform):
                raise DiscoveryError(ReasonCode.DOCKER_CONTEXT_UNSUPPORTED)
            context = selected()
            if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', context):
                raise DiscoveryError(ReasonCode.DOCKER_CONTEXT_UNSUPPORTED)
            endpoint = inspect(context)
            if self.platform == 'win32' and endpoint != 'npipe:////./pipe/dockerDesktopLinuxEngine':
                raise DiscoveryError(ReasonCode.DOCKER_CONTEXT_UNSUPPORTED)
            if host and host != endpoint:
                raise DiscoveryError(ReasonCode.DOCKER_CONTEXT_UNSUPPORTED)
            if expected is not None and (context, endpoint) != (expected.context, expected.endpoint):
                raise DiscoveryError(ReasonCode.STORAGE_STALE)
            identity = (self.platform_provider.identity(self.platform, endpoint, call,
                        expected.identity if expected else None)
                        if native else None)
            info = _json(call(('docker', '--context', context, 'info', '--format', '{{json .}}')))
            if type(info) is not dict or info.get('OSType') != 'linux':
                raise DiscoveryError(ReasonCode.DOCKER_CONTEXT_UNSUPPORTED)
            if expected is not None and _info_key(info) != expected.info:
                raise DiscoveryError(ReasonCode.STORAGE_STALE)
            root = info.get('DockerRootDir')
            if not _absolute_posix(root) or root == '/':
                raise DiscoveryError(ReasonCode.STORAGE_UNKNOWN)
            security = info.get('SecurityOptions')
            if type(security) is not list or any(type(x) is not str for x in security):
                raise DiscoveryError(ReasonCode.STORAGE_UNKNOWN)
            rootless = any('rootless' in option for option in security)
            if self.platform == 'win32':
                if rootless:
                    raise DiscoveryError(ReasonCode.DOCKER_CONTEXT_UNSUPPORTED)
                layout = self.platform_provider.wsl_layout(info, call, manual_backing)
                # The layout provider has completed the active-disk/VHD recheck.
                # Capture this proof before potentially slow host volume sampling.
                association_stamp = self.clock().isoformat().replace('+00:00', 'Z')
                version = layout.desktop_version
                match = re.fullmatch(r'4\.(\d+)\.\d+(?:\.\d+)?', version)
                if not match or int(match[1]) < 30:
                    raise DiscoveryError(ReasonCode.DOCKER_CONTEXT_UNSUPPORTED)
                if (layout.data_root != root or not _absolute_posix(layout.mount_target)
                        or layout.mount_target == '/' or not layout.mount_device.startswith('/dev/')
                        or not layout.data_path.endswith(root)
                        or not (layout.data_path == layout.mount_target
                                or layout.data_path.startswith(layout.mount_target.rstrip('/') + '/'))
                        or not (layout.vm_volume.identity.value or '').startswith('wsl:')):
                    raise DiscoveryError(ReasonCode.STORAGE_UNKNOWN)
                stores.append(Store('docker_vm', root, layout.vm_volume))
                if (not layout.association_verified or not layout.active_disk_id
                        or layout.active_disk_id != layout.candidate_disk_id
                        or not layout.candidate_vhdx or not ntpath.isabs(layout.candidate_vhdx)
                        or not layout.candidate_vhdx.lower().endswith('.vhdx')):
                    raise DiscoveryError(ReasonCode.BACKING_UNRESOLVED)
                if native and self._native_volume_provider:
                    host_data = _json(call(_host_command(layout.candidate_vhdx)))
                    host_volume = _host_volume(host_data, layout)
                    if layout.host_file is not None and tuple(host_data['file']) != layout.host_file:
                        raise DiscoveryError(ReasonCode.STORAGE_STALE)
                    layout = replace(layout, host_file=tuple(host_data['file']))
                else:
                    host_volume = self.volumes.sample(ntpath.dirname(layout.candidate_vhdx))
                stores.append(Store('docker_backing', layout.candidate_vhdx, host_volume))
                backing.append(BackingAssociation(layout.vm_volume.identity, host_volume.identity,
                                                   Measurement(True, 'verified_disk_association',
                                                               FieldStatus.OK, association_stamp, None)))
            elif self.platform.startswith('linux'):
                layout = self.platform_provider.linux_layout(
                    {**info, '_local_endpoint': endpoint, '_identity': identity}, call)
                if (layout.verified is not True or (rootless and not layout.rootless_verified)
                        or not layout.roots or layout.roots[0] != root
                        or any(not _absolute_posix(p) or p == '/' for p in layout.roots)):
                    raise DiscoveryError(ReasonCode.DOCKER_CONTEXT_UNSUPPORTED)
                if expected is not None and layout != expected.layout:
                    raise DiscoveryError(ReasonCode.STORAGE_STALE)
                native_volumes = (_read_native_volumes(call, layout.roots)
                                  if native and self._native_volume_provider else None)
                for index, path in enumerate(layout.roots):
                    stores.append(Store('docker_data' if index == 0 else 'containerd', path,
                                        native_volumes[index] if native_volumes is not None
                                        else self.volumes.sample(path)))
            elif self.platform == 'darwin':
                # Only Docker Desktop keeps its VM disk in Docker.raw; Colima,
                # OrbStack or Podman engines would make that evidence meaningless.
                if rootless or info.get('OperatingSystem') != 'Docker Desktop':
                    raise DiscoveryError(ReasonCode.DOCKER_CONTEXT_UNSUPPORTED)
                layout = self.platform_provider.darwin_layout(info, call, manual_backing)
                if (not isinstance(layout, DarwinLayout) or layout.data_root != root
                        or not _absolute_posix(layout.backing) or not layout.backing.endswith('.raw')
                        or type(layout.limit) is not int or layout.limit <= 0
                        or type(layout.used) is not int or layout.used < 0
                        or type(layout.file) is not tuple or len(layout.file) != 2
                        or any(type(v) is not int for v in layout.file)):
                    raise DiscoveryError(ReasonCode.STORAGE_UNKNOWN)
                version = (expected.layout.desktop_version if expected is not None
                           else _darwin_desktop_version(call, context))
                layout = replace(layout, desktop_version=version)
                if expected is not None and _darwin_key(layout) != _darwin_key(expected.layout):
                    raise DiscoveryError(ReasonCode.STORAGE_STALE)
                # The host volume of Docker.raw, through the ordinary provider, so
                # its identity coalesces with cache/scratch samples on that volume.
                host_volume = self.volumes.sample(posixpath.dirname(layout.backing))
                again = self.platform_provider.darwin_layout(info, call, manual_backing)
                if _darwin_key(replace(again, desktop_version=version)) != _darwin_key(layout):
                    raise DiscoveryError(ReasonCode.STORAGE_STALE)
                stores.append(Store('docker_data', root, _darwin_store_volume(layout, host_volume)))
            else:
                raise DiscoveryError(ReasonCode.DOCKER_CONTEXT_UNSUPPORTED)
            # Detect active-context or endpoint changes over the complete sample.
            if (dict(self.environ) != original or selected() != context or inspect(context) != endpoint):
                raise DiscoveryError(ReasonCode.STORAGE_STALE)
            if native and self.platform_provider.identity(self.platform, endpoint, call, identity) != identity:
                raise DiscoveryError(ReasonCode.STORAGE_STALE)
            if (native and isinstance(layout, WSLLayout)
                    and (layout.proof is None or layout.proof.identity != identity)):
                raise DiscoveryError(ReasonCode.STORAGE_STALE)
            if time.monotonic() >= deadline:
                raise DiscoveryError(ReasonCode.REQUEST_TIMEOUT)
            # Check freshness after the final potentially slow context command.
            decision_now = self.clock()
            for store in stores:
                if not _usable(store.volume, decision_now, self.max_age,
                               require_inodes=store.role != 'docker_backing'):
                    raise DiscoveryError(ReasonCode.STORAGE_UNKNOWN)
            baseline = _PreparedBaseline(context, endpoint, tuple(sorted(original.items())),
                root, layout, identity, _store_keys(stores), manual_backing,
                _info_key(info))
            if expected is not None and _baseline_key(baseline) != _baseline_key(expected):
                raise DiscoveryError(ReasonCode.STORAGE_STALE)
            if captured is not None:
                captured.append(baseline)
        except DiscoveryError as error:
            failure = error.reason
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            failure = ReasonCode.STORAGE_UNKNOWN
        if failure:
            stores, backing = [], []
        # Decision time is independent of the original input measurement times.
        decision_stamp = self.clock().isoformat().replace('+00:00', 'Z')
        return DockerStorageEvidence(measured(context), measured(root), measured(version, 'desktop_install'),
                                     _deduplicate(stores), tuple(backing),
                                     replace(measured(None if failure else True, 'docker_storage',
                                              failure or ReasonCode.STORAGE_UNKNOWN), sample_time=decision_stamp),
                                     observed if not failure else None)


@dataclass(frozen=True, slots=True)
class _PreparedBaseline:
    context: str
    endpoint: str
    environment: tuple[tuple[str, str], ...]
    root: str
    layout: LinuxLayout | WSLLayout | DarwinLayout
    identity: LinuxIdentity | WSLIdentity | DarwinIdentity | None
    stores: tuple
    manual_backing: str
    info: tuple


def _info_key(info):
    return tuple((k, json.dumps(info.get(k), sort_keys=True)) for k in
                 ('OSType', 'DockerRootDir', 'SecurityOptions', 'Driver', 'DriverStatus'))


def _store_keys(stores):
    return tuple((s.role, s.path, s.volume.identity.value) for s in stores)


def _baseline_key(baseline):
    layout = baseline.layout
    if isinstance(layout, WSLLayout):
        layout_key = (layout.desktop_version, layout.data_root, layout.mount_target,
            layout.mount_device, layout.candidate_vhdx, layout.active_disk_id,
            layout.candidate_disk_id, layout.association_verified, layout.data_path, layout.proof,
            layout.host_file)
    elif isinstance(layout, DarwinLayout):
        layout_key = _darwin_key(layout)
    else:
        layout_key = (layout.roots, layout.verified, layout.rootless_verified, layout.identity)
    return (baseline.context, baseline.endpoint, baseline.environment, baseline.root,
            baseline.identity, baseline.stores, baseline.manual_backing, baseline.info, layout_key)


def _denied(evidence, reason, clock):
    return replace(evidence, constraints=(), backing=(), oldest_observed_monotonic=None,
        admission=Measurement(None, 'docker_storage',
            FieldStatus.STALE if reason is ReasonCode.STORAGE_STALE else FieldStatus.UNKNOWN,
            clock().isoformat().replace('+00:00', 'Z'), reason))


class DockerStorageSession:
    """Single-owner prepared observation, with an immutable comparison baseline.

    ``initial_evidence`` is the original discovery result, never retimestamped.
    Use refresh result's ``oldest_observed_monotonic`` for StorageSample's age;
    fields retain their actual UTC measurement times. Any failed refresh or
    close permanently invalidates this session. Context manager exit closes it.
    No persistent process or storage reservation belongs to this object.
    """
    __slots__ = ('_baseline', '_closed', '_initial', '_lock', '_provider')

    def __init__(self, provider, evidence, baseline):
        self._provider, self._baseline, self._initial = provider, baseline, evidence
        self._closed = baseline is None or not evidence.can_pull
        self._lock = threading.Lock()

    @property
    def initial_evidence(self):
        return self._initial

    @property
    def closed(self):
        return self._closed

    def close(self):
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def refresh(self, *, deadline: float) -> DockerStorageEvidence:
        if type(deadline) not in (int, float) or not math.isfinite(deadline):
            raise ValueError('Finite absolute monotonic deadline required')
        if self._closed or not self._lock.acquire(blocking=False):
            self.close()
            return _denied(self._initial, ReasonCode.STORAGE_STALE, self._provider.clock)
        try:
            if time.monotonic() >= deadline:
                result = _denied(self._initial, ReasonCode.REQUEST_TIMEOUT, self._provider.clock)
            elif (self._provider.platform == 'win32'
                    and isinstance(self._provider.platform_provider, NativePlatformProvider)):
                result = self._provider._refresh_windows(self._baseline, deadline)
            else:
                result = self._provider._sample(manual_backing=self._baseline.manual_backing,
                                                deadline=deadline, expected=self._baseline)
            if not result.can_pull or self._closed:
                self.close()
                return _denied(result, result.reason or ReasonCode.STORAGE_STALE, self._provider.clock)
            return result
        finally:
            self._lock.release()


def _bounded_file(path: Path, limit: int = 1048576) -> bytes:
    # Configuration is only supporting evidence, never proof of disk association.
    with path.open('rb') as stream:
        result = stream.read(limit + 1)
    if len(result) > limit:
        raise DiscoveryError(ReasonCode.REQUEST_TOO_LARGE)
    return result


# Fixed host-authored scripts. All variable paths are separate positional argv;
# the transport shares its existing output cap and deadline across these batches.
_WSL_METADATA = r'''
set -eu
export LC_ALL=C
emit() { printf '@%s\n' "$1"; shift; "$@"; }
n=0; found=''
for p in /proc/[0-9]*; do
    n=$((n+1)); [ "$n" -le 4096 ] || exit 1
    IFS= read -r comm < "$p/comm" 2>/dev/null || continue
    if [ "$comm" = dockerd ]; then [ -z "$found" ] || exit 1; found=${p##*/}; fi
done
[ -n "$found" ] || exit 1
emit pid printf '%s\n' "$found"
emit comm cat "/proc/$found/comm"
emit exe readlink "/proc/$found/exe"
emit process cat "/proc/$found/stat"
emit credentials sed -n '/^Uid:/p;/^Gid:/p;/^Groups:/p;/^CapPrm:/p;/^CapEff:/p' "/proc/$found/status"
emit label od -An -tx1 "/proc/$found/attr/current"
emit mnt readlink "/proc/$found/ns/mnt"
emit user readlink "/proc/$found/ns/user"
emit kernel uname -r
emit canonical readlink -f "$1"
emit block stat -c '%t:%T %F' "$1"
emit sysfs readlink -f "/sys/dev/block/$2"
emit dev cat "/sys/dev/block/$2/dev"
emit options cat "/proc/fs/ext4/${1##*/}/options"
emit superblock dumpe2fs -h "$1"
emit docker_stat stat -c '%d:%i' -- "$3"
emit container_stat stat -c '%d:%i' -- "$4"
emit docker_mount findmnt --json --target "$3" --output TARGET,SOURCE,FSTYPE,OPTIONS,UUID,MAJ:MIN
emit container_mount findmnt --json --target "$4" --output TARGET,SOURCE,FSTYPE,OPTIONS,UUID,MAJ:MIN
'''

_WSL_PERMISSION = r'''
set -eu
export LC_ALL=C
emit() { printf '@%s\n' "$1"; shift; "$@"; }
emit credentials sed -n '/^Uid:/p;/^Gid:/p;/^Groups:/p;/^CapPrm:/p;/^CapEff:/p' "/proc/$$/status"
emit label od -An -tx1 "/proc/$$/attr/current"
emit mnt readlink "/proc/$$/ns/mnt"
emit user readlink "/proc/$$/ns/user"
for p in "$1" "$2"; do
    /usr/bin/test -d "$p"
    /usr/bin/test -w "$p"
    /usr/bin/test -x "$p"
done
emit docker_stat stat -c '%d:%i' -- "$1"
emit container_stat stat -c '%d:%i' -- "$2"
emit docker_mount findmnt --json --target "$1" --output TARGET,SOURCE,FSTYPE,OPTIONS,UUID,MAJ:MIN
emit container_mount findmnt --json --target "$2" --output TARGET,SOURCE,FSTYPE,OPTIONS,UUID,MAJ:MIN
emit capacity stat -f -c '%S %b %a %d' -- "$1"
'''

_IDENTITY_NAMES = 'pid comm exe process credentials label mnt user'
_WSL_IDENTITY = _WSL_METADATA.split('emit kernel', 1)[0]
_WSL_PREFIX = ('wsl.exe', '--distribution', 'docker-desktop', '--user', 'root', '--exec')
_WSL_BLOCK = r'''
set -eu
export LC_ALL=C
device=${1##*/}
parent=$(lsblk --noheadings --nodeps --output PKNAME "$1")
if [ -n "$parent" ]; then device=$parent; fi
case "$device" in *[!a-zA-Z0-9_-]*|'') exit 1;; esac
leaf=$(readlink -f "/sys/dev/block/$2")
whole=$(readlink -f "/sys/class/block/$device")
[ "$leaf" = "$whole" ] || [ "${leaf%/*}" = "$whole" ] || exit 1
printf '@device\n%s\n@leaf\n%s\n@whole\n%s\n@vpd\n' "$device" "$leaf" "$whole"
od -An -tx1 "/sys/class/block/$device/device/vpd_pg83"
'''
_WSL_BATCH = ('set -eu\nmetadata() (\n' + _WSL_METADATA + '\n)\nblock() (\n' +
    _WSL_BLOCK + '\n)\n' + r'''
printf '@@block_before\n'
block "$1" "$2"
printf '@@before\n'
metadata "$1" "$2" "$3" "$4"
printf '@@permission\n'
nsenter --target="$6" --mount --pid --root --wdns=/ -- /bin/sh -c ''' +
    "'" + _WSL_PERMISSION.replace("'", "'\"'\"'") + "'" + r''' storage-probe "$5" /var/lib/containerd
printf '@@after\n'
metadata "$1" "$2" "$3" "$4"
printf '@@block_after\n'
block "$1" "$2"
''')
_WSL_OBSERVATION = ('set -eu\nidentity() (\n' + _WSL_IDENTITY +
    '\n)\nsample() (\n' + _WSL_BATCH + '\n)\n' + r'''
stage=waiting
trap 'stage=continue' USR1
advance() {
    while [ "$stage" = waiting ]; do sleep 0.002; done
    [ "$stage" = continue ]
    stage=waiting
}
printf '@@identity_before\n'
identity
printf '@@ready\n'
advance
sample "$@"
printf '@@sampled\n'
advance
printf '@@identity_after\n'
identity
''')

# The leader deliberately remains alive after success OR failure. Its PID,
# start time, process group and session therefore remain an ownership anchor
# until the supervisor's single group kill. None of these scripts daemonize.
_WSL_LEADER = r'''
controller=$1
shift
trap 'code=$?; trap "" USR1; printf "@@done:%s\n" "$code"; kill -USR1 "$controller"; while :; do sleep 1; done' EXIT
'''
_WSL_WORKER = _WSL_LEADER + _WSL_OBSERVATION
# Execute the original command as an argv array, retaining a live session
# leader even when the command exits nonzero. Binary output is not reframed.
_WSL_ONESHOT = _WSL_LEADER + 'set -e\n"$@"\n'

_WSL_SUPERVISOR = r'''
export LC_ALL=C
budget=$1
program=$2
stages=$3
shift 3
tick() {
    IFS=' ' read -r uptime unused < /proc/uptime || return 1
    seconds=${uptime%.*}; fraction=${uptime#*.}
    now=$((seconds * 100 + 1$fraction - 100))
}
record() {
    IFS= read -r line < "/proc/$1/stat" 2>/dev/null || return 1
    tail=${line##*) }
    set -- $tail
    [ "$#" -ge 20 ] || return 1
    group=$3; session=$4
    shift 19
    start=$1
}
tick || exit 1
end=$((now + budget))
owned=; anchor=; finished=0
trap 'finished=1' USR1
cleanup() {
    trap - EXIT HUP INT TERM
    [ -n "$owned" ] || return 1
    # Never signal an unproved group or a recycled PID. The fixed worker's
    # EXIT trap holds this exact leader until this point.
    if ! record "$owned" || [ "$start" != "$anchor" ] ||
       [ "$group" != "$owned" ] || [ "$session" != "$owned" ]; then
        return 1
    fi
    kill -KILL "-$owned" || return 1
    wait "$owned" 2>/dev/null || :
    # Orphans are reaped by the distro init. Prove their absence, including
    # zombies, before acknowledging cleanup; wait alone proves only leader.
    tick || return 1
    reap_end=$((now + 10))
    while :; do
        remains=0
        for node in /proc/[0-9]*; do
            record "${node##*/}" || continue
            if [ "$session" = "$owned" ]; then remains=1; fi
        done
        if [ "$remains" = 0 ]; then printf '@@reaped\n'; return 0; fi
        tick || return 1
        [ "$now" -lt "$reap_end" ] || return 1
        sleep 0.002
    done
}
trap 'cleanup' EXIT
trap 'exit 1' HUP INT TERM
/usr/bin/setsid /bin/sh -c "$program" storage-worker "$$" "$@" < /dev/null &
owned=$!
while :; do
    if record "$owned" && [ "$group" = "$owned" ] && [ "$session" = "$owned" ]; then
        anchor=$start
        break
    fi
    tick || exit 1
    [ "$now" -lt "$end" ] || exit 1
    sleep 0.002
done
read_byte() {
    tick || exit 1
    left=$((end - now))
    [ "$left" -gt 0 ] || exit 1
    # BusyBox ash returns 1 for timeout, signal interruption AND EOF. Only
    # EOF clears the destination; timeout/interruption leave it unchanged.
    # Read one byte with NUL as delimiter so partial lines cannot disappear
    # on timeout and NUL is explicitly rejected (successful empty read).
    byte=waiting
    if IFS= read -r -n 1 -d '' -t 0.01 byte; then
        [ -n "$byte" ] || exit 1
        return 0
    else
        read_status=$?
    fi
    [ "$read_status" = 1 ] && [ "$byte" = waiting ] || exit 1
    return 1
}
phase=0; stage=
newline='
'
while [ "$phase" -lt "$stages" ]; do
    if read_byte; then
        stage="$stage$byte"
        # At most the nine bytes of continue plus newline can accumulate.
        case "continue$newline" in "$stage"*) ;; *) exit 1 ;; esac
        [ "$finished" = 0 ] || exit 1
        if [ "$stage" = "continue$newline" ]; then
            record "$owned" && [ "$start" = "$anchor" ] && [ "$session" = "$owned" ] || exit 1
            kill -USR1 "$owned" || exit 1
            phase=$((phase + 1)); stage=
        fi
    else
        # Completion before all stages is a failed protocol, even when its
        # signal arrived immediately before the read began.
        [ "$finished" = 0 ] || exit 1
    fi
done
while :; do
    # Always poll before accepting completion: buffered unexpected bytes and
    # EOF must fail. A signal between a flag check and read cannot strand us
    # for the remaining budget; every read is bounded to one clock tick.
    # An interrupted poll can hide pending input: require a poll that began
    # after completion was observed before accepting its empty timeout.
    completion_seen=$finished
    if read_byte; then exit 1; fi
    [ "$completion_seen" = 0 ] || break
done
cleanup || exit 1
'''


def _witness_finished(output):
    footer = b'@@done:0\n@@reaped\n'
    if not output.endswith(footer) or output.count(b'@@done:') != 1:
        raise DiscoveryError(ReasonCode.STORAGE_UNKNOWN)
    return output[:-len(footer)]


def _witness_identity(output, phase):
    if phase == b'before':
        start, end = b'@@identity_before\n', b'@@ready\n'
        if not output.startswith(start) or not output.endswith(end) or output.count(end) != 1:
            raise DiscoveryError(ReasonCode.STORAGE_UNKNOWN)
        body = output[len(start):output.index(end)]
    else:
        if b'@@done:' in output:
            output = _witness_finished(output)
        marker = b'@@sampled\n@@identity_after\n'
        if output.count(marker) != 1:
            raise DiscoveryError(ReasonCode.STORAGE_UNKNOWN)
        body = output.split(marker, 1)[1]
    return _identity_record(_proof_sections(ProcessResult(0, body), _IDENTITY_NAMES))


def _witness_batch(output):
    if b'@@done:' in output:
        output = _witness_finished(output)
    start, end = b'@@ready\n', b'@@sampled\n@@identity_after\n'
    if output.count(start) != 1 or output.count(end) != 1:
        raise DiscoveryError(ReasonCode.STORAGE_UNKNOWN)
    return ProcessResult(0, output.split(start, 1)[1].split(end, 1)[0])


def _wsl_observation_command(layout, deadline):
    budget = max(1, int((deadline - time.monotonic() - 0.2) * 100))
    return (*_WSL_PREFIX, '/bin/sh', '-c', _WSL_SUPERVISOR,
            'storage-supervisor', str(budget), _WSL_WORKER, '2', *_wsl_command(layout)[10:])


def _wsl_transport_command(argv, deadline):
    # Both fixed discovery forms are supported. Trusted injected callable
    # runners retain their explicit raw-command contract; native BoundedRunner
    # routes every discovery call here, including findmnt/lsblk/binary VPD.
    for prefix in (_WSL_PREFIX, ('wsl.exe', '--distribution', 'docker-desktop', '--exec')):
        if argv[:len(prefix)] == prefix:
            command = argv[len(prefix):]
            if command[:3] == ('/bin/sh', '-c', _WSL_SUPERVISOR):
                return argv, False
            budget = max(1, int((deadline - time.monotonic() - 0.2) * 100))
            return (*prefix, '/bin/sh', '-c', _WSL_SUPERVISOR, 'storage-supervisor',
                    str(budget), _WSL_ONESHOT, '0', *command), True
    return argv, False


def _wsl_command_result(result):
    footer = re.search(rb'@@done:([0-9]{1,3})\n@@reaped\n\Z', result.stdout)
    if result.returncode != 0 or footer is None or int(footer[1]) > 255:
        raise DiscoveryError(ReasonCode.STORAGE_UNKNOWN)
    return ProcessResult(int(footer[1]), result.stdout[:footer.start()])


def _wsl_command(layout):
    return (*_WSL_PREFIX, '/bin/sh', '-c', _WSL_BATCH, 'storage-refresh',
        layout.mount_device, dict(layout.proof.mount)['maj:min'], layout.data_path,
        '/tmp/docker-desktop-root/var/lib/containerd', layout.data_root,
        dict(layout.proof.identity.fields)['pid'])


def _wsl_refresh(call, layout, *, observation_stamp=None):
    """One real WSL batch; parsing reuses the discovery proof validator."""
    proof = layout.proof
    if not isinstance(proof, WSLPreparedProof):
        raise DiscoveryError(ReasonCode.STORAGE_UNKNOWN)
    mount, container = dict(proof.mount), dict(proof.container_mount)
    stamp = observation_stamp or datetime.now(UTC).isoformat().replace('+00:00', 'Z')
    result = call(_wsl_command(layout))
    if len(result.stdout) > 524288:
        raise DiscoveryError(ReasonCode.REQUEST_TOO_LARGE)
    try:
        chunks = re.split(rb'^@@([a-z_]+)\n', result.stdout, flags=re.MULTILINE)
        if chunks[0] or chunks[1::2] != [b'block_before', b'before', b'permission', b'after', b'block_after']:
            raise ValueError()
        records = dict(zip(chunks[1::2], chunks[2::2]))
        blocks = [_proof_sections(ProcessResult(0, records[name]), 'device leaf whole vpd')
                  for name in (b'block_before', b'block_after')]
        if blocks[0] != blocks[1]:
            raise DiscoveryError(ReasonCode.STORAGE_STALE)
        block = blocks[0]
        if (block['leaf'] != dict(proof.witnesses)['sysfs']
                or block['whole'].rsplit('/', 1)[-1] != block['device']
                or block['leaf'] not in (block['whole'], block['whole'] + '/' + layout.mount_device.rsplit('/', 1)[-1])
                or parse_msft_vpd83(bytes.fromhex(block['vpd'])) != layout.active_disk_id):
            raise DiscoveryError(ReasonCode.STORAGE_STALE)
        observations = iter((records[b'before'], records[b'permission'], records[b'after']))
        captured = []
        volume = _wsl_storage_proof(lambda argv: ProcessResult(0, next(observations)),
            layout.mount_device, mount, container, layout.data_path,
            '/tmp/docker-desktop-root/var/lib/containerd', layout.data_root,
            captured=captured, observation_stamp=stamp)
        if captured != [proof]:
            raise DiscoveryError(ReasonCode.STORAGE_STALE)
        return replace(volume, identity=replace(volume.identity, value=layout.vm_volume.identity.value))
    except (ValueError, KeyError, TypeError, StopIteration):
        raise DiscoveryError(ReasonCode.STORAGE_UNKNOWN) from None


def _check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise DiscoveryError(ReasonCode.REQUEST_TIMEOUT)


def _identity_record(record):
    result = {key: record[key] for key in _IDENTITY_NAMES.split()}
    if (not re.fullmatch(r'[1-9]\d{0,9}', result['pid']) or result['comm'] != 'dockerd'
            or result['exe'] != '/usr/bin/dockerd'):
        raise ValueError()
    process = re.fullmatch(r'(\d+) \(dockerd\) (.*)', result['process'])
    if not process or process[1] != result['pid']:
        raise ValueError()
    fields = process[2].split()
    if len(fields) < 20 or not fields[19].isdigit():
        raise ValueError()
    result['process'] = fields[19]
    result['credentials'] = _credentials(result['credentials'])
    if (not re.fullmatch(r'(?:[0-9a-f]{2}\s*)+', result['label'])
            or not re.fullmatch(r'mnt:\[\d+\]', result['mnt'])
            or not re.fullmatch(r'user:\[\d+\]', result['user'])):
        raise ValueError()
    result['label'] = ''.join(result['label'].split())
    return WSLIdentity(tuple(result.items()))


def _wsl_identity(call, previous=None):
    try:
        result = _identity_record(_proof_sections(call((*_WSL_PREFIX, '/bin/sh', '-c',
            _WSL_IDENTITY, 'storage-identity')), _IDENTITY_NAMES))
        if previous is not None and result != previous:
            raise DiscoveryError(ReasonCode.STORAGE_STALE)
        return result
    except (ValueError, KeyError, TypeError):
        raise DiscoveryError(ReasonCode.STORAGE_UNKNOWN) from None


def _proof_sections(result, expected):
    if result.returncode or len(result.stdout) > 131072:
        raise DiscoveryError(ReasonCode.STORAGE_UNKNOWN)
    try:
        text = result.stdout.decode('ascii')
        parts = re.split(r'^@([a-z_]+)\n', text, flags=re.MULTILINE)
        if parts[0] or tuple(parts[1::2]) != tuple(expected.split()):
            raise ValueError()
        return dict(zip(parts[1::2], (p.strip() for p in parts[2::2])))
    except (UnicodeError, ValueError):
        raise DiscoveryError(ReasonCode.STORAGE_UNKNOWN) from None


def _credentials(text):
    keys = ('Uid', 'Gid', 'Groups', 'CapPrm', 'CapEff')
    lines = text.splitlines()
    if len(lines) != len(keys):
        raise ValueError()
    result = []
    for key, line in zip(keys, lines):
        name, values = line.split(':', 1)
        tokens = values.split()
        if name != key or not tokens:
            raise ValueError()
        if key.startswith('Cap'):
            if len(tokens) != 1 or not re.fullmatch('[0-9a-f]{16}', tokens[0]):
                raise ValueError()
        elif any(not re.fullmatch(r'\d+', token) for token in tokens):
            raise ValueError()
        if key in ('Uid', 'Gid') and (len(tokens) != 4 or set(tokens) != {'0'}):
            # This proof supports only the actual rootful Desktop daemon.
            raise ValueError()
        result.append(tuple(tokens))
    return tuple(result)


def _verified_quota(metadata, source, mount):
    # Only the reviewed Microsoft kernel semantics are admitted. In this
    # implementation noquota requires all quota mount flags to be clear, and
    # quota_on cannot enable hidden quotas without the ext4 quota feature.
    # https://github.com/microsoft/WSL2-Linux-Kernel/blob/linux-msft-wsl-6.6.87.2/fs/ext4/super.c
    if metadata['kernel'] != '6.6.87.2-microsoft-standard-WSL2':
        raise ValueError()
    major, minor = map(int, mount['maj:min'].split(':'))
    if (metadata['canonical'] != source or metadata['dev'] != mount['maj:min']
            or metadata['sysfs'].rsplit('/', 1)[-1] != source.rsplit('/', 1)[-1]
            or not metadata['sysfs'].startswith('/sys/devices/')):
        raise ValueError()
    block = re.fullmatch(r'([0-9a-f]+):([0-9a-f]+) block special file', metadata['block'])
    if not block or (int(block[1], 16), int(block[2], 16)) != (major, minor):
        raise ValueError()
    options = metadata['options'].splitlines()
    simple = {'rw', 'bsddf', 'nogrpid', 'block_validity', 'dioread_nolock', 'nodiscard',
              'delalloc', 'nowarn_on_error', 'journal_checksum', 'barrier', 'auto_da_alloc',
              'user_xattr', 'acl', 'noquota'}
    numeric = r'(?:resuid|resgid|commit|min_batch_time|max_batch_time|stripe|inode_readahead_blks|init_itable|max_dir_size_kb)=\d+'
    if (len(options) != len(set(options)) or not {'rw', 'noquota'} <= set(options)
            or any(o not in simple and o not in ('errors=continue', 'data=ordered')
                   and not re.fullmatch(numeric, o) for o in options)):
        raise ValueError()
    fields = {}
    for line in metadata['superblock'].splitlines():
        if not line.strip():
            continue
        key, value = line.split(':', 1)
        if key in fields:
            raise ValueError()
        fields[key] = value.strip()
    features = fields['Filesystem features'].split()
    recognized = {'has_journal', 'ext_attr', 'resize_inode', 'dir_index', 'filetype',
                  'needs_recovery', 'extent', '64bit', 'flex_bg', 'sparse_super',
                  'large_file', 'huge_file', 'dir_nlink', 'extra_isize', 'metadata_csum'}
    if (fields['Filesystem magic number'] != '0xEF53'
            or str(uuid.UUID(fields['Filesystem UUID'])) != mount['uuid']
            or not features or not set(features) <= recognized):
        raise ValueError()
    for key, value in fields.items():
        if 'quota' in key.lower() and (key not in ('User quota inode', 'Group quota inode',
                                                  'Project quota inode') or value != '0'):
            raise ValueError()
    return tuple(options), tuple(features)


def _wsl_storage_proof(call, source, mount, container_mount, data_path, container_path, root,
                       *, captured=None, observation_stamp=None):
    prefix = ('wsl.exe', '--distribution', 'docker-desktop', '--user', 'root', '--exec')
    names = ('pid comm exe process credentials label mnt user kernel canonical block sysfs dev '
             'options superblock docker_stat container_stat docker_mount container_mount')
    def snapshot():
        result = _proof_sections(call((*prefix, '/bin/sh', '-c', _WSL_METADATA, 'storage-proof',
                                       source, mount['maj:min'], data_path, container_path)), names)
        if (not re.fullmatch(r'[1-9]\d{0,9}', result['pid']) or result['comm'] != 'dockerd'
                or result['exe'] != '/usr/bin/dockerd'):
            raise ValueError()
        process = re.fullmatch(r'(\d+) \(dockerd\) (.*)', result['process'])
        if not process or process[1] != result['pid']:
            raise ValueError()
        fields = process[2].split()
        if len(fields) < 20 or not fields[19].isdigit():
            raise ValueError()
        result['process'] = fields[19]  # starttime, not mutable CPU counters
        result['credentials'] = _credentials(result['credentials'])
        if (not re.fullmatch(r'(?:[0-9a-f]{2}\s*)+', result['label'])
                or not re.fullmatch(r'mnt:\[\d+\]', result['mnt'])
                or not re.fullmatch(r'user:\[\d+\]', result['user'])):
            raise ValueError()
        result['label'] = ''.join(result['label'].split())
        result['superblock'] = _verified_quota(result, source, mount)
        return result
    def paths(record, scoped):
        for name, original, path in (('docker', mount, data_path if scoped else root),
                                     ('container', container_mount, container_path if scoped else '/var/lib/containerd')):
            data = json.loads(record[name + '_mount'])
            items = data['filesystems']
            if type(items) is not list or len(items) != 1:
                raise ValueError()
            item = items[0]
            target = item['target']
            if (not _absolute_posix(target) or not (path == target or path.startswith(target.rstrip('/') + '/'))
                    or any(item[k] != original[k] for k in ('uuid', 'maj:min', 'fstype'))
                    or item['source'].split('[', 1)[0] != source
                    or 'rw' not in item['options'].split(',')
                    or any('quota' in o and o != 'noquota' for o in item['options'].split(','))):
                raise ValueError()
            if scoped and item != original:
                raise ValueError()
            st = re.fullmatch(r'(\d+):([1-9]\d*)', record[name + '_stat'])
            if not st:
                raise ValueError()
            dev = int(st[1])
            major = ((dev >> 8) & 0xfff) | ((dev >> 32) & 0xfffff000)
            minor = (dev & 0xff) | ((dev >> 12) & 0xffffff00)
            if f'{major}:{minor}' != mount['maj:min']:
                raise ValueError()
    try:
        before = snapshot()
        paths(before, True)
        permission = _proof_sections(call((*prefix, 'nsenter', '--target=' + before['pid'],
            '--mount', '--pid', '--root', '--wdns=/', '--', '/bin/sh', '-c', _WSL_PERMISSION,
            'storage-probe', root, '/var/lib/containerd')),
            'credentials label mnt user docker_stat container_stat docker_mount container_mount capacity')
        stamp = observation_stamp or datetime.now(UTC).isoformat().replace('+00:00', 'Z')
        permission['credentials'] = _credentials(permission['credentials'])
        permission['label'] = ''.join(permission['label'].split())
        for key in ('credentials', 'label', 'mnt', 'user', 'docker_stat', 'container_stat'):
            if permission[key] != before[key]:
                raise ValueError()
        paths(permission, False)
        counters = permission['capacity'].split()
        if len(counters) != 4 or any(not re.fullmatch(r'\d+', x) for x in counters):
            raise ValueError()
        block, blocks, available, inodes = map(int, counters)
        if block <= 0 or blocks <= 0 or not 0 <= available <= blocks:
            raise ValueError()
        after = snapshot()
        paths(after, True)
        if before != after:
            raise DiscoveryError(ReasonCode.STORAGE_STALE)
        if captured is not None:
            captured.append(WSLPreparedProof(tuple(sorted(mount.items())),
                tuple(sorted(container_mount.items())),
                WSLIdentity(tuple((k, before[k]) for k in _IDENTITY_NAMES.split())),
                tuple(sorted(before.items()))))
        def value(item):
            return Measurement(item, 'wsl_daemon_quota_permission', FieldStatus.OK, stamp, None)
        return VolumeEvidence(value('wsl:' + mount['uuid'] + ':' + mount['maj:min']),
                              value(block * blocks), value(block * available), value(inodes),
                              value(True), value(True), value(False),
                              Measurement(None, 'wsl_daemon_quota_permission', FieldStatus.NOT_APPLICABLE,
                                          stamp, ReasonCode.STORAGE_UNKNOWN))
    except (ValueError, KeyError, TypeError, IndexError):
        raise DiscoveryError(ReasonCode.STORAGE_UNKNOWN) from None


def _linux_path(value):
    return (_absolute_posix(value) and value != '/'
            and posixpath.normpath(value) == value
            and not any(ord(c) < 32 or ord(c) == 127 for c in value))


def _linux_option(argv, *keys):
    values = []
    for index, arg in enumerate(argv):
        for key in keys:
            if arg.startswith(key + '='):
                values.append(arg[len(key) + 1:])
            elif arg == key:
                values.append(argv[index + 1] if index + 1 < len(argv) else '')
    if len(values) > 1 or values and not values[0]:
        raise ValueError()
    return values[0] if values else None


def _linux_process(pid, name, check):
    check()
    argv = tuple(_bounded_file(pid / 'cmdline', 16384).decode().rstrip('\x00').split('\x00'))
    process = _bounded_file(pid / 'stat', 16384).decode()
    match = re.fullmatch(r'(\d+) \(' + name + r'\) (.*)\s*', process)
    if not match or match[1] != pid.name:
        raise ValueError()
    fields = match[2].split()
    if len(fields) < 20 or not fields[19].isdigit():
        raise ValueError()
    executable = os.readlink(pid / 'exe')
    if (not argv or posixpath.basename(argv[0]) != name or not _linux_path(executable)
            or posixpath.basename(executable) != name):
        raise ValueError()
    namespaces = tuple(os.readlink(pid / 'ns' / key) for key in ('mnt', 'net', 'user'))
    host = tuple(os.readlink('/proc/self/ns/' + key) for key in ('mnt', 'net', 'user'))
    if namespaces != host or any(not re.fullmatch(key + r':\[\d+\]', value)
                                 for key, value in zip(('mnt', 'net', 'user'), namespaces)):
        raise ValueError()
    root, local = os.stat(pid / 'root'), os.stat('/')
    if (root.st_dev, root.st_ino) != (local.st_dev, local.st_ino):
        raise ValueError()
    binary = os.stat(pid / 'exe')
    check()
    return (argv, fields[19], executable, binary.st_dev, binary.st_ino,
            namespaces, root.st_dev, root.st_ino)


def _linux_socket(path, check):
    check()
    if not _linux_path(path):
        raise ValueError()
    node = os.stat(path)
    if not stat.S_ISSOCK(node.st_mode):
        raise ValueError()
    entries = _bounded_file(Path('/proc/net/unix')).decode().splitlines()[1:]
    matches = [line.split() for line in entries
               if len(line.split()) == 8 and line.split()[7] == path]
    if len(matches) != 1:
        raise ValueError()
    row = matches[0]
    # Match the listening stream socket, not a client connection or proxy path.
    if (int(row[3], 16) != 0x10000 or int(row[4], 16) != 1
            or int(row[5], 16) != 1 or not row[6].isdigit() or int(row[6]) <= 0):
        raise ValueError()
    check()
    return node.st_dev, node.st_ino, row[6]


def _linux_owns(pid, socket, check):
    owned = False
    for index, fd in enumerate((pid / 'fd').iterdir()):
        check()
        if index >= 4096:
            raise ValueError()
        try:
            owned |= os.readlink(fd) == 'socket:[' + socket[2] + ']'
        except FileNotFoundError:
            continue
    return owned


def _containerd_stores(result, snapshotter, *, runtime_state=None):
    # Supported protocol: unfiltered `ctr plugins list --detailed`, reviewed at
    # containerd v1.7.28 and v2.0.0, for the built-in Linux Docker profile below.
    # CRI/external storage and healthy blockfile/devmapper/zfs/aufs stores are
    # outside this subset; their backing storage is not proved by these exports.
    # https://github.com/containerd/containerd/blob/v2.0.0/docs/PLUGINS.md
    # https://github.com/containerd/containerd/blob/v2.0.0/cmd/ctr/commands/plugins/plugins.go
    # Persistent exports: plugins/content/local/plugin, plugins/metadata,
    # plugins/snapshots/{overlay,native,btrfs}/plugin and plugins/services/opt.
    # Runtime/service identities: cmd/containerd/builtins, core/runtime/v2,
    # core/metrics/cgroups, plugins/{services,restart,sandbox,transfer,streaming},
    # plugins/{nri,imageverifier} and pkg/tracing/plugin. The legacy identities
    # are also reviewed in v1.7.28 runtime/{v1/linux,v2} and services/restart.
    # Task managers have no exports but DO write persistent working directories.
    # Their roots are derived below from agreeing live, fixed-suffix exports:
    # https://github.com/containerd/containerd/blob/v2.0.0/cmd/containerd/server/server.go
    # https://github.com/containerd/containerd/blob/v2.0.0/core/runtime/v2/task_manager.go
    # v1.7.28 plugin/context.go gives the same root/URI construction.
    # ctr is unstable: reject additions/ambiguity rather than guessing a root.
    if result.returncode or len(result.stdout) > 1048576:
        raise ValueError()
    text = result.stdout.decode('utf-8')
    if any(ord(c) < 32 and c not in '\n\t' or ord(c) == 127 for c in text):
        raise ValueError()
    records, current, section = [], None, None
    order = ('Type', 'ID', 'Requires', 'Platforms', 'Exports', 'Capabilities', 'Error')
    for line in text.splitlines():
        if not line.strip():
            continue
        if line.startswith('Type:'):
            if len(records) >= 256:
                raise ValueError()
            current = {}
            records.append(current)
            section = None
        if current is None:
            raise ValueError()
        if line[:1].isspace():
            value = line.strip()
            if section == 'Requires':
                if not re.fullmatch(r'io\.[a-z0-9_.-]+', value) or value in current[section]:
                    raise ValueError()
                current[section].append(value)
            elif section in ('Exports', 'Error'):
                pair = value.split(None, 1)
                if len(pair) != 2 or pair[0] in current[section]:
                    raise ValueError()
                current[section][pair[0]] = pair[1]
            else:
                raise ValueError()
            continue
        key, separator, value = line.partition(':')
        if (not separator or key not in order or key in current
                or current and order.index(key) <= order.index(next(reversed(current)))):
            raise ValueError()
        section, value = key, value.strip()
        if key in ('Requires', 'Exports', 'Error'):
            if value:
                raise ValueError()
            current[key] = [] if key == 'Requires' else {}
        elif not value:
            raise ValueError()
        else:
            current[key] = value
    required = {('io.containerd.content.v1', 'content'),
                ('io.containerd.metadata.v1', 'bolt')}
    if snapshotter is not None:
        required.add(('io.containerd.snapshotter.v1', snapshotter))
    # Exact reviewed identities, never a prefix/type-wide non-storage bypass.
    # NRI is a CRI integration and bindir verifies transfer images; neither
    # registers an independent Docker store. Unexpected exports still block.
    wrappers = {
        'io.containerd.grpc.v1': {'containers', 'content', 'diff', 'events', 'healthcheck',
            'images', 'introspection', 'leases', 'namespaces', 'sandbox-controllers',
            'sandboxes', 'snapshots', 'streaming', 'tasks', 'transfer', 'version'},
        'io.containerd.service.v1': {name + '-service' for name in
            ('containers', 'content', 'diff', 'images', 'introspection', 'namespaces', 'snapshots', 'tasks')},
        'io.containerd.event.v1': {'exchange'}, 'io.containerd.gc.v1': {'scheduler'},
        'io.containerd.lease.v1': {'manager'}, 'io.containerd.differ.v1': {'walking'},
        'io.containerd.runtime.v1': {'linux'}, 'io.containerd.runtime.v2': {'task', 'shim'},
        'io.containerd.shim.v1': {'manager'}, 'io.containerd.monitor.v1': {'cgroups'},
        'io.containerd.monitor.task.v1': {'cgroups'},
        'io.containerd.monitor.container.v1': {'restart'},
        'io.containerd.internal.v1': {'restart', 'tracing'},
        'io.containerd.tracing.processor.v1': {'otlp'},
        'io.containerd.sandbox.controller.v1': {'shim', 'local'},
        'io.containerd.sandbox.store.v1': {'local'},
        'io.containerd.streaming.v1': {'manager'}, 'io.containerd.transfer.v1': {'local'},
        'io.containerd.nri.v1': {'nri'}, 'io.containerd.image-verifier.v1': {'bindir'},
        'io.containerd.warning.v1': {'deprecations'},
    }
    root_stores = {('io.containerd.content.v1', 'content'),
                   *(('io.containerd.snapshotter.v1', name) for name in ('overlayfs', 'native', 'btrfs'))}
    # Normal distributions report optional plugins which did not initialize.
    # Only these known failed plugins may be skipped without store semantics.
    failed_only = {('io.containerd.snapshotter.v1', name)
                   for name in ('aufs', 'blockfile', 'devmapper', 'zfs')}
    failed_only.update({('io.containerd.grpc.v1', 'cri'),
                        ('io.containerd.cri.v1', 'images'), ('io.containerd.cri.v1', 'runtime'),
                        ('io.containerd.podsandbox.controller.v1', 'podsandbox')})
    roots, seen, healthy, fixed_roots = [], set(), set(), {}
    for record in records:
        identity = record.get('Type'), record.get('ID')
        if (any(type(x) is not str or not re.fullmatch(r'[a-z0-9_.-]+', x) for x in identity)
                or identity in seen):
            raise ValueError()
        seen.add(identity)
        if 'Requires' in record and not record['Requires']:
            raise ValueError()
        wrapper = identity[1] in wrappers.get(identity[0], set())
        known = (wrapper or identity in root_stores or identity in failed_only
                 or identity in {('io.containerd.metadata.v1', 'bolt'), ('io.containerd.internal.v1', 'opt')})
        if not known:
            raise ValueError()
        if 'Error' in record:
            if set(record['Error']) != {'Code:', 'Message:'} or identity in required:
                raise ValueError()
            continue  # A failed plugin is not an active store.
        healthy.add(identity)
        exports = record.get('Exports', {})
        if identity in root_stores:
            if set(exports) != {'root'}:
                raise ValueError()
            root = exports['root']
        elif identity == ('io.containerd.metadata.v1', 'bolt'):
            if (set(exports) != {'path', 'policy'} or exports['policy'] not in ('shared', 'isolated')
                    or not _linux_path(exports['path']) or posixpath.basename(exports['path']) != 'meta.db'):
                raise ValueError()
            root = posixpath.dirname(exports['path'])
        elif identity == ('io.containerd.internal.v1', 'opt'):
            # Unlike bolt/path, opt/path names a directory (bin and lib live
            # below it), potentially on a separate filesystem.
            if set(exports) != {'path'}:
                raise ValueError()
            root = exports['path']
        elif wrapper and not exports:
            continue
        else:
            raise ValueError()
        if not _linux_path(root):
            raise ValueError()
        roots.append(root)
        if identity in required and identity[0] in ('io.containerd.content.v1', 'io.containerd.metadata.v1'):
            fixed_roots[identity] = root
    if not required <= healthy:
        raise ValueError()
    persistent_runtime = healthy & {('io.containerd.runtime.v1', 'linux'),
                                    ('io.containerd.runtime.v2', 'task'),
                                    ('io.containerd.sandbox.controller.v1', 'shim')}
    if persistent_runtime:
        # Unlike snapshotters, these two built-ins do not allow root_path
        # overrides. An unexpected suffix or differing bases cannot establish
        # the running containerd root. Each task root is sampled separately so
        # a mount beneath the containerd root is not hidden by its parent's free
        # space. Transient task state is not exposed by this protocol; require
        # the separate immutable --state argument from the verified process.
        if not _linux_path(runtime_state):
            raise ValueError()
        bases = set()
        for identity, path in fixed_roots.items():
            if posixpath.basename(path) != '.'.join(identity):
                raise ValueError()
            bases.add(posixpath.dirname(path))
        if len(bases) != 1 or not _linux_path(next(iter(bases))):
            raise ValueError()
        base = bases.pop()
        roots.extend(posixpath.join(base, '.'.join(identity)) for identity in sorted(persistent_runtime))
        roots.extend(posixpath.join(runtime_state, '.'.join(identity)) for identity in sorted(persistent_runtime))
    return tuple(dict.fromkeys(roots))


class NativePlatformProvider:
    """Native metadata provider; uncertainty is explicit, never a guessed path."""
    def __init__(self, *, virtual_disk=None):
        self.virtual_disk = virtual_disk

    def identity(self, platform, endpoint, call, previous=None):
        if platform == 'win32':
            return _wsl_identity(call, previous)
        if platform == 'darwin':
            try:
                return _darwin_identity(endpoint, previous)
            except (OSError, ValueError, TypeError):
                raise DiscoveryError(ReasonCode.STORAGE_UNKNOWN) from None
        if not platform.startswith('linux'):
            raise DiscoveryError(ReasonCode.DOCKER_CONTEXT_UNSUPPORTED)
        def check():
            return _check_deadline(call.deadline)
        try:
            return self._linux_identity(endpoint, check, previous)
        except (OSError, ValueError, TypeError, KeyError, UnicodeError):
            raise DiscoveryError(ReasonCode.STORAGE_UNKNOWN) from None

    def _linux_identity(self, endpoint, check, previous=None):
        processes = []
        entries = ((Path('/proc') / previous.daemon_pid, Path('/proc') / previous.container_pid)
                   if isinstance(previous, LinuxIdentity) else Path('/proc').iterdir())
        for index, entry in enumerate(entries):
            check()
            if index >= 8192 or len(processes) > 32:
                raise ValueError()
            if not entry.name.isdigit():
                continue
            try:
                argv = _bounded_file(entry / 'cmdline', 16384).decode().split('\x00')
            except (FileNotFoundError, ProcessLookupError):
                continue
            if argv and posixpath.basename(argv[0]) in ('dockerd', 'containerd'):
                processes.append((posixpath.basename(argv[0]), entry))
        daemons = [pid for name, pid in processes if name == 'dockerd']
        if len(daemons) != 1:
            raise ValueError()
        daemon = daemons[0]
        before_daemon = _linux_process(daemon, 'dockerd', check)
        docker_socket = _linux_socket(endpoint[7:], check)
        if not _linux_owns(daemon, docker_socket, check):
            raise ValueError()
        # Disk configuration cannot prove the running Docker endpoint.
        address = _linux_option(before_daemon[0], '--containerd')
        if not _linux_path(address):
            raise ValueError()
        container_socket = _linux_socket(address, check)
        owners = [pid for name, pid in processes if name == 'containerd'
                  and _linux_owns(pid, container_socket, check)]
        if len(owners) != 1:
            raise ValueError()
        container = owners[0]
        before_container = _linux_process(container, 'containerd', check)
        identity = LinuxIdentity(daemon.name, before_daemon, docker_socket,
                                 container.name, before_container, container_socket, address)
        if previous is not None and identity != previous:
            raise DiscoveryError(ReasonCode.STORAGE_STALE)
        return identity

    def linux_layout(self, info, call):
        deadline = getattr(call, 'deadline', time.monotonic() + 20)
        def check():
            if time.monotonic() >= deadline:
                raise DiscoveryError(ReasonCode.REQUEST_TIMEOUT)
        try:
            root = info['DockerRootDir']
            endpoint = info.get('_local_endpoint', '')
            driver, status = info.get('Driver'), info.get('DriverStatus')
            if (type(status) is not list or any(type(row) is not list or len(row) != 2
                    or any(type(value) is not str for value in row) for row in status)):
                return LinuxLayout((), False)
            if driver == 'overlayfs' and status == [['driver-type', 'io.containerd.snapshotter.v1']]:
                snapshotter = 'overlayfs'
            elif driver == 'overlay2' and not any(row[0] == 'driver-type' for row in status):
                # Classic Docker layers live under DockerRootDir. A separately
                # bound containerd still needs the same complete runtime query;
                # every active supported snapshotter it reports is budgeted,
                # although Docker does not require an image snapshotter here.
                snapshotter = None
            else:
                return LinuxLayout((), False)
            if (not _linux_path(root) or not endpoint.startswith('unix:///')
                    or any('rootless' in item for item in info.get('SecurityOptions', []))):
                return LinuxLayout((), False)
            identity = info.get('_identity')
            if identity is None:
                identity = self._linux_identity(endpoint, check)
            if not isinstance(identity, LinuxIdentity):
                raise TypeError()
            daemon, container = Path('/proc') / identity.daemon_pid, Path('/proc') / identity.container_pid
            before_daemon, before_container = identity.daemon, identity.container
            docker_socket, container_socket, address = identity.docker_socket, identity.container_socket, identity.address
            config_name = _linux_option(before_container[0], '--config', '-c')
            # Retain the conservative imports/proxy policy. Configuration is
            # supporting evidence only; neither its root nor address is used.
            try:
                cfg = tomllib.loads(_bounded_file(Path(config_name or '/etc/containerd/config.toml')).decode())
            except FileNotFoundError:
                if config_name:
                    raise
                cfg = {}
            if type(cfg.get('imports', [])) is not list or cfg.get('imports', []) or cfg.get('proxy_plugins'):
                raise ValueError()
            check()
            stores = _containerd_stores(call(('ctr', '--address', address, '--timeout', '5s',
                                               'plugins', 'list', '--detailed')), snapshotter,
                                        runtime_state=_linux_option(before_container[0], '--state'))
            # Recheck both joins after the runtime query; PID alone is reusable.
            if (before_daemon != _linux_process(daemon, 'dockerd', check)
                    or before_container != _linux_process(container, 'containerd', check)
                    or docker_socket != _linux_socket(endpoint[7:], check)
                    or container_socket != _linux_socket(address, check)
                    or not _linux_owns(daemon, docker_socket, check)
                    or not _linux_owns(container, container_socket, check)):
                raise DiscoveryError(ReasonCode.STORAGE_STALE)
            check()
            return LinuxLayout(tuple(dict.fromkeys((root, *stores))), True, identity=identity)
        except (OSError, ValueError, TypeError, KeyError, UnicodeError):
            return LinuxLayout((), False)

    def wsl_layout(self, info, call, manual_backing):
        # Even registry reads, candidate traversal and API construction can
        # block. Real Windows discovery belongs to the owned helper process.
        inventory = (_windows_inventory(call, manual_backing)
                     if self.virtual_disk is None else None)
        version = inventory['version'] if inventory is not None else _desktop_version()
        return self._discover_wsl(version, info, call, manual_backing, inventory)

    def _discover_wsl(self, version, info, call, manual_backing, inventory=None):
        root = info['DockerRootDir']
        # This namespace prefix is specific to the gated Desktop 4.x layout.
        # findmnt proves actual coverage; '/' can never substitute for data.
        data_path = '/tmp/docker-desktop-root' + root
        container_path = '/tmp/docker-desktop-root/var/lib/containerd'
        def mounted(path):
            metadata = _json(call(('wsl.exe', '--distribution', 'docker-desktop', '--exec',
                                   'findmnt', '--json', '--target', path, '--output',
                                   'TARGET,SOURCE,FSTYPE,OPTIONS,UUID,MAJ:MIN')))
            items = metadata.get('filesystems')
            if type(items) is not list or len(items) != 1:
                raise DiscoveryError(ReasonCode.STORAGE_UNKNOWN)
            item = items[0]
            target = item.get('target')
            if (not _absolute_posix(target) or target == '/'
                    or not (path == target or path.startswith(target.rstrip('/') + '/'))
                    or item.get('fstype') != 'ext4' or not item.get('uuid')
                    or not re.fullmatch(r'\d+:\d+', item.get('maj:min', ''))
                    or 'rw' not in item.get('options', '').split(',')):
                raise DiscoveryError(ReasonCode.STORAGE_UNKNOWN)
            return item
        mount, container_mount = mounted(data_path), mounted(container_path)
        def disk_key(item):
            return item.get('source', '').split('[', 1)[0], item['uuid'], item['maj:min']
        if disk_key(mount) != disk_key(container_mount):
            # Multiple guest disks need multiple associations, not a single sum.
            raise DiscoveryError(ReasonCode.BACKING_UNRESOLVED)
        source = mount.get('source', '').split('[', 1)[0]
        if not re.fullmatch(r'/dev/[A-Za-z0-9_-]+', source):
            raise DiscoveryError(ReasonCode.STORAGE_UNKNOWN)
        association = {}
        try:
            device = source.rsplit('/', 1)[1]
            # Resolve a partition to its actual parent block device from sysfs metadata.
            parent = call(('wsl.exe', '--distribution', 'docker-desktop', '--exec',
                           'lsblk', '--noheadings', '--nodeps', '--output', 'PKNAME', source))
            parent_name = parent.stdout.decode().strip()
            if parent_name:
                if not re.fullmatch(r'[A-Za-z0-9_-]+', parent_name):
                    raise DiscoveryError(ReasonCode.BACKING_UNRESOLVED)
                device = parent_name
            page = call(('wsl.exe', '--distribution', 'docker-desktop', '--exec',
                         'cat', '/sys/class/block/' + device + '/device/vpd_pg83')).stdout
            active_id = parse_msft_vpd83(page)
            api = self.virtual_disk
            matching = []
            if inventory is None:  # Explicit trusted fixture only.
                candidates = []
                for candidate in self._candidates(manual_backing):
                    try:
                        candidates.append((candidate, api.inspect(candidate), None))
                    except (OSError, DiscoveryError):
                        continue
            else:
                candidates = [(c['path'], c['disk'], tuple(c['file'])) for c in inventory['candidates']]
            for candidate, disk, file_key in candidates:
                disk_id, loaded, remote = disk
                if loaded is True and remote is False and disk_id == active_id:
                    matching.append((candidate, disk_id, file_key))
            if len(matching) == 1:
                candidate, candidate_id, file_key = matching[0]
                def inspect_candidate():
                    if inventory is None:
                        return api.inspect(candidate) == (candidate_id, True, False)
                    return _windows_inventory(call, candidate) == {'version': version,
                        'candidates': [{'path': candidate, 'disk': [candidate_id, True, False], 'file': list(file_key)}]}
                if (mounted(data_path) != mount or mounted(container_path) != container_mount
                        or parse_msft_vpd83(call(('wsl.exe', '--distribution', 'docker-desktop',
                            '--exec', 'cat', '/sys/class/block/' + device + '/device/vpd_pg83')).stdout) != active_id
                        or not inspect_candidate()):
                    raise DiscoveryError(ReasonCode.STORAGE_STALE)
                association = {'candidate_vhdx': candidate, 'active_disk_id': active_id,
                                   'candidate_disk_id': candidate_id, 'association_verified': True, 'host_file': file_key}
        except DiscoveryError as error:
            if error.reason in (ReasonCode.STORAGE_STALE, ReasonCode.REQUEST_TIMEOUT,
                                ReasonCode.REQUEST_TOO_LARGE, ReasonCode.SERVICE_UNAVAILABLE):
                raise
        except (OSError, UnicodeError, ValueError):
            pass
        # Only publish capacity after quota, actual daemon credentials and
        # allocation-path permissions have been proved and rechecked.
        captured = []
        volume = _wsl_storage_proof(call, source, mount, container_mount, data_path, container_path, root,
                                    captured=captured)
        if association:
            if (parse_msft_vpd83(call(('wsl.exe', '--distribution', 'docker-desktop', '--exec',
                    'cat', '/sys/class/block/' + device + '/device/vpd_pg83')).stdout) != association['active_disk_id']
                    or not inspect_candidate()):
                raise DiscoveryError(ReasonCode.STORAGE_STALE)
            volume = replace(volume, identity=replace(volume.identity,
                value='wsl:' + mount['uuid'] + ':' + association['active_disk_id']))
        return WSLLayout(version, root, mount['target'], source, volume,
                         data_path=data_path, proof=captured[0], **association)

    def darwin_layout(self, info, call, manual_backing):
        root = info['DockerRootDir']
        found = self._darwin_backing(manual_backing)
        if len(found) != 1:
            # Missing, or two disks and no proof which one the VM uses: the
            # operator names the Docker Desktop disk location (docker_backing).
            raise DiscoveryError(ReasonCode.BACKING_UNRESOLVED)
        path, node = found[0]
        blocks = getattr(node, 'st_blocks', None)
        if (type(node.st_size) is not int or node.st_size <= 0
                or type(blocks) is not int or blocks < 0
                or type(node.st_dev) is not int or type(node.st_ino) is not int):
            raise DiscoveryError(ReasonCode.STORAGE_UNKNOWN)
        # st_blocks counts 512-byte units on every POSIX platform, APFS included.
        return DarwinLayout(None, root, path, (node.st_dev, node.st_ino), node.st_size, blocks * 512)

    def _darwin_backing(self, manual_backing):
        """Regular, non-symlink Docker.raw candidates with their lstat records."""
        if type(manual_backing) is not str or len(manual_backing) > 8192 or '\x00' in manual_backing:
            raise DiscoveryError(ReasonCode.BACKING_UNRESOLVED)
        home = Path.home().as_posix()
        if manual_backing:
            manual = manual_backing.rstrip('/') or '/'
            # A file is used as given; a directory is the Desktop "disk image
            # location" or one of its ancestors.
            paths = ([manual] if manual.endswith('.raw') else
                     [posixpath.join(manual, 'Docker.raw'), posixpath.join(manual, 'data', 'Docker.raw'),
                      posixpath.join(manual, 'vms', '0', 'data', 'Docker.raw')])
        else:
            paths = []
            settings_dir = posixpath.join(home, 'Library', 'Group Containers', 'group.com.docker')
            # Settings select the location but prove nothing about the live disk.
            for filename in ('settings-store.json', 'settings.json'):
                try:
                    settings = json.loads(_bounded_file(Path(posixpath.join(settings_dir, filename))))
                except (OSError, ValueError, UnicodeError):
                    continue
                selected = settings.get('DataFolder', settings.get('dataFolder')) if type(settings) is dict else None
                if type(selected) is str and selected:
                    paths.append(posixpath.join(selected, 'Docker.raw'))
                    break
            default = posixpath.join(home, 'Library', 'Containers', 'com.docker.docker',
                                     'Data', 'vms', '0', 'data', 'Docker.raw')
            if default not in paths:
                paths.append(default)
        admitted = []
        for candidate in paths:
            if not _absolute_posix(candidate) or not candidate.endswith('.raw'):
                continue
            try:
                node = os.lstat(candidate)
            except OSError:
                continue
            if stat.S_ISREG(node.st_mode):
                admitted.append((candidate, node))
        return admitted

    def _candidates(self, manual_backing):
        if manual_backing:
            paths = [manual_backing]
        else:
            paths = []
            appdata = os.environ.get('APPDATA')
            local = os.environ.get('LOCALAPPDATA')
            # Settings select location but do not prove that any disk is active.
            if appdata:
                for filename in ('settings-store.json', 'settings.json'):
                    location = Path(appdata) / 'Docker' / filename
                    try:
                        settings = json.loads(_bounded_file(location))
                        selected = settings.get('diskImageLocation')
                        if type(selected) is str and selected:
                            paths.extend((str(Path(selected) / 'disk' / 'docker_data.vhdx'),
                                          str(Path(selected) / 'docker_data.vhdx'),
                                          str(Path(selected) / 'data' / 'ext4.vhdx')))
                            break
                    except FileNotFoundError:
                        continue
            if not paths and local:
                paths = [str(Path(local) / 'Docker' / 'wsl' / 'disk' / 'docker_data.vhdx'),
                         str(Path(local) / 'Docker' / 'wsl' / 'data' / 'ext4.vhdx')]
        admitted = []
        for candidate in paths:
            path = Path(candidate)
            if (not ntpath.isabs(candidate) or not candidate.lower().endswith('.vhdx')
                    or candidate.startswith(('\\\\', '//'))):
                continue
            try:
                # Reject candidate/ancestor reparse redirection. Host volume will
                # additionally be sampled using volumes.py's held handle walk.
                if not path.is_file() or any(p.is_symlink() or p.is_junction()
                                             for p in (path, *path.parents)):
                    continue
                admitted.append(str(path.absolute()))
            except OSError:
                continue
        return tuple(dict.fromkeys(admitted))


def _darwin_identity(endpoint, previous=None):
    socket_path = endpoint[7:]
    if not _absolute_posix(socket_path):
        raise ValueError()
    node = os.stat(socket_path)
    if not stat.S_ISSOCK(node.st_mode):
        raise ValueError()
    identity = DarwinIdentity(socket_path, (node.st_dev, node.st_ino))
    if previous is not None and identity != previous:
        raise DiscoveryError(ReasonCode.STORAGE_STALE)
    return identity


def _desktop_version():
    import winreg
    version = None
    for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        try:
            with winreg.OpenKey(hive, r'SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\Docker Desktop') as key:
                version = winreg.QueryValueEx(key, 'DisplayVersion')[0]
                break
        except OSError:
            continue
    if type(version) is not str or not re.fullmatch(r'4\.(?:[3-9]\d|\d{3,})\.\d+(?:\.\d+)?', version):
        raise DiscoveryError(ReasonCode.DOCKER_CONTEXT_UNSUPPORTED)
    return version


def parse_msft_vpd83(page: bytes) -> str:
    """WSL's binary, device-associated MSFT vendor descriptor, not textual IDs."""
    if (type(page) is not bytes or len(page) < 4 or len(page) > 65539
            or page[1] != 0x83 or int.from_bytes(page[2:4], 'big') != len(page) - 4):
        raise DiscoveryError(ReasonCode.BACKING_UNRESOLVED)
    identities = []
    offset = 4
    while offset < len(page):
        if len(page) - offset < 4:
            raise DiscoveryError(ReasonCode.BACKING_UNRESOLVED)
        head = page[offset:offset + 4]
        size = head[3]
        offset += 4
        if offset + size > len(page):
            raise DiscoveryError(ReasonCode.BACKING_UNRESOLVED)
        payload = page[offset:offset + size]
        offset += size
        if (head[0] & 15 == 1 and head[1] & 15 == 1 and (head[1] >> 4) & 3 == 0
                and size == 24 and payload[:8] == b'MSFT    '):
            identities.append(str(uuid.UUID(bytes_le=payload[8:24])))
    if len(identities) != 1 or identities[0] == str(uuid.UUID(int=0)):
        raise DiscoveryError(ReasonCode.BACKING_UNRESOLVED)
    return identities[0]


class WindowsVirtualDiskAPI:
    """Documented GET_INFO_ONLY/READ_ONLY V2 OpenVirtualDisk; never attach/detach.

    Query 14 is VirtualDiskId (SCSI join); query 2's Identifier is a different
    identifier. The union preserves the SDK's alignment and full V3 open size.
    https://learn.microsoft.com/en-us/windows/win32/api/virtdisk/nf-virtdisk-openvirtualdisk
    """
    def __init__(self, *, library=None, kernel=None):
        import ctypes as c
        self.c = c
        DWORD, BOOL = c.c_uint32, c.c_int32
        class GUID(c.Structure):
            _fields_ = [('bytes', c.c_ubyte * 16)]
        class StorageType(c.Structure):
            _fields_ = [('DeviceId', DWORD), ('VendorId', GUID)]
        class V1(c.Structure):
            _fields_ = [('RWDepth', DWORD)]
        class V2(c.Structure):
            _fields_ = [('GetInfoOnly', BOOL), ('ReadOnly', BOOL), ('ResiliencyGuid', GUID)]
        class V3(c.Structure):
            _fields_ = V2._fields_ + [('SnapshotId', GUID)]
        class OpenUnion(c.Union):
            _fields_ = [('V1', V1), ('V2', V2), ('V3', V3)]
        class OpenParameters(c.Structure):
            _fields_ = [('Version', DWORD), ('data', OpenUnion)]
        class Size(c.Structure):
            _fields_ = [('VirtualSize', c.c_uint64), ('PhysicalSize', c.c_uint64),
                        ('BlockSize', DWORD), ('SectorSize', DWORD)]
        class Physical(c.Structure):
            _fields_ = [('LogicalSectorSize', DWORD), ('PhysicalSectorSize', DWORD), ('IsRemote', BOOL)]
        class InfoUnion(c.Union):
            _fields_ = [('Size', Size), ('VirtualDiskId', GUID), ('IsLoaded', BOOL),
                        ('Physical', Physical)]
        class Info(c.Structure):
            _fields_ = [('Version', DWORD), ('data', InfoUnion)]
        self.StorageType, self.OpenParameters, self.Info = StorageType, OpenParameters, Info
        if library is None:
            library = c.WinDLL('virtdisk', use_last_error=True)
        if kernel is None:
            kernel = c.WinDLL('kernel32', use_last_error=True)
        self.open = library.OpenVirtualDisk
        self.open.argtypes = [c.POINTER(StorageType), c.c_wchar_p, DWORD, DWORD,
                              c.POINTER(OpenParameters), c.POINTER(c.c_void_p)]
        self.open.restype = DWORD
        self.query = library.GetVirtualDiskInformation
        self.query.argtypes = [c.c_void_p, c.POINTER(DWORD), c.POINTER(Info), c.POINTER(DWORD)]
        self.query.restype = DWORD
        self.close = kernel.CloseHandle
        self.close.argtypes, self.close.restype = [c.c_void_p], BOOL

    def inspect(self, path):
        c = self.c
        storage, params, handle = self.StorageType(), self.OpenParameters(), c.c_void_p()
        storage.DeviceId = 3
        storage.VendorId.bytes[:] = uuid.UUID('ec984aec-a0f9-47e9-901f-71415a66345b').bytes_le
        params.Version = 2
        params.data.V2.GetInfoOnly = params.data.V2.ReadOnly = 1
        if self.open(c.byref(storage), path, 0, 0, c.byref(params), c.byref(handle)) != 0:
            raise DiscoveryError(ReasonCode.BACKING_UNRESOLVED)
        try:
            values = []
            for version in (14, 13, 9):
                info, size, used = self.Info(), c.c_uint32(c.sizeof(self.Info)), c.c_uint32()
                info.Version = version
                if self.query(handle, c.byref(size), c.byref(info), c.byref(used)) != 0:
                    raise DiscoveryError(ReasonCode.BACKING_UNRESOLVED)
                if version == 14:
                    values.append(str(uuid.UUID(bytes_le=bytes(info.data.VirtualDiskId.bytes))))
                elif version == 13:
                    values.append(bool(info.data.IsLoaded))
                else:
                    values.append(bool(info.data.Physical.IsRemote))
            return tuple(values)
        finally:
            self.close(handle)


def make_docker_storage_provider(**kwargs) -> DockerStorageProvider:
    return DockerStorageProvider(**kwargs)


def _host_command(path):
    return (sys.executable, '-m', 'podmuncher.docker_storage', '--native-host-probe', path)


def _windows_inventory(call, manual):
    if type(manual) is not str or len(manual) > 8192 or '\x00' in manual:
        raise DiscoveryError(ReasonCode.BACKING_UNRESOLVED)
    data = _json(call((sys.executable, '-m', 'podmuncher.docker_storage',
                      '--native-windows-inventory', manual)))
    if (type(data) is not dict or set(data) != {'version', 'candidates'}
            or type(data['version']) is not str
            or not re.fullmatch(r'4\.(?:[3-9]\d|\d{3,})\.\d+(?:\.\d+)?', data['version'])
            or type(data['candidates']) is not list or len(data['candidates']) > 3):
        raise DiscoveryError(ReasonCode.STORAGE_UNKNOWN)
    for candidate in data['candidates']:
        if (type(candidate) is not dict or set(candidate) != {'path', 'disk', 'file'}
                or type(candidate['path']) is not str or len(candidate['path']) > 8192
                or '\x00' in candidate['path'] or not ntpath.isabs(candidate['path'])
                or candidate['path'].startswith(('\\\\', '//'))
                or not candidate['path'].lower().endswith('.vhdx')
                or type(candidate['disk']) is not list or len(candidate['disk']) != 3
                or type(candidate['disk'][0]) is not str
                or not re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', candidate['disk'][0])
                or any(type(v) is not bool for v in candidate['disk'][1:])
                or type(candidate['file']) is not list or len(candidate['file']) != 2
                or any(type(v) is not int or v < 0 for v in candidate['file'])):
            raise DiscoveryError(ReasonCode.STORAGE_UNKNOWN)
    return data


def _native_windows_inventory(manual):
    if type(manual) is not str or len(manual) > 8192 or '\x00' in manual:
        raise ValueError()
    native, version = NativePlatformProvider(), _desktop_version()
    paths = native._candidates(manual)
    if len(paths) > 3:
        raise ValueError()
    api, candidates = WindowsVirtualDiskAPI(), []
    for path in paths:
        before = os.stat(path, follow_symlinks=False)
        try:
            disk = api.inspect(path)
        except (OSError, DiscoveryError):
            continue
        after = os.stat(path, follow_symlinks=False)
        if ((before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
                or native._candidates(path) != (path,)):
            raise DiscoveryError(ReasonCode.STORAGE_STALE)
        candidates.append({'path': path, 'disk': disk, 'file': (before.st_dev, before.st_ino)})
    if _desktop_version() != version or native._candidates(manual) != paths:
        raise DiscoveryError(ReasonCode.STORAGE_STALE)
    return {'version': version, 'candidates': candidates}


def _decode_volume(data):
    if type(data) is not dict or set(data) != set(VolumeEvidence.__dataclass_fields__):
        raise ValueError()
    return VolumeEvidence(**{k: Measurement(**value) for k, value in data.items()})


def _host_volume(data, layout):
    if (type(data) is not dict or set(data) != {'version', 'disk', 'file', 'volume'}
            or data['version'] != layout.desktop_version
            or data['disk'] != [layout.candidate_disk_id, True, False]
            or type(data['file']) is not list or len(data['file']) != 2
            or any(type(v) is not int or v < 0 for v in data['file'])):
        raise DiscoveryError(ReasonCode.STORAGE_STALE)
    return _decode_volume(data['volume'])


def _read_native_volumes(call, paths):
    data = _json(call((sys.executable, '-m', 'podmuncher.docker_storage',
                      '--native-volume-probe', json.dumps(paths))))
    if type(data) is not list or len(data) != len(paths):
        raise ValueError()
    return tuple(_decode_volume(item) for item in data)


def _native_host_probe(path):
    # Synchronous Win32 APIs run only inside this short-lived owned subprocess;
    # the parent can kill/reap it even if a native disk query stalls indefinitely.
    native = NativePlatformProvider()
    version = _desktop_version()
    if native._candidates(path) != (path,):
        raise DiscoveryError(ReasonCode.BACKING_UNRESOLVED)
    def file_key():
        node = os.stat(path, follow_symlinks=False)
        return node.st_dev, node.st_ino
    before = file_key()
    api = WindowsVirtualDiskAPI()
    disk = api.inspect(path)
    if disk[1:] != (True, False):
        raise DiscoveryError(ReasonCode.BACKING_UNRESOLVED)
    volume = make_volume_provider().sample(ntpath.dirname(path))
    if (api.inspect(path) != disk or file_key() != before
            or native._candidates(path) != (path,) or _desktop_version() != version):
        raise DiscoveryError(ReasonCode.STORAGE_STALE)
    return {'version': version, 'disk': disk, 'file': before, 'volume': asdict(volume)}


if __name__ == '__main__':
    # Private fixed read-only protocol; no eval, shell, writes or CLI diagnostics.
    try:
        if len(sys.argv) != 3:
            raise ValueError()
        if sys.argv[1] == '--native-host-probe' and sys.platform == 'win32':
            result = _native_host_probe(sys.argv[2])
        elif sys.argv[1] == '--native-windows-inventory' and sys.platform == 'win32':
            result = _native_windows_inventory(sys.argv[2])
        elif sys.argv[1] == '--native-volume-probe' and sys.platform.startswith('linux'):
            paths = json.loads(sys.argv[2])
            if type(paths) is not list or not 1 <= len(paths) <= 256 or not all(_linux_path(p) for p in paths):
                raise ValueError()
            result = [asdict(make_volume_provider().sample(p)) for p in paths]
        else:
            raise ValueError()
        print(json.dumps(result))
    except (DiscoveryError, OSError, ValueError, TypeError, KeyError):
        sys.exit(2)
